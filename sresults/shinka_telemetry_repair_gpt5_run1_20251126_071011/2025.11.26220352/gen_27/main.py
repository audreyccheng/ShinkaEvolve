# EVOLVE-BLOCK-START
"""
Paired bounded scaling + adaptive redistribution telemetry repair.

Novel pipeline:
1) Pair hardening + status resolution with rate-aware tolerances
2) Per-router two-stage repair:
   - bounded multiplicative pre-step on less-trusted direction
   - adaptive, confidence- and residual-weighted additive redistribution
     with ramped caps and router total-delta cap
3) Limited post-router link symmetry reconciliation
4) Logistic, rate-aware confidence with penalties for caps and scaling
"""
from typing import Dict, Any, Tuple, List
from math import sqrt, exp


def repair_network_telemetry(telemetry: Dict[str, Dict[str, Any]],
                             topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:
    # Core tolerances
    HARDENING_THRESHOLD = 0.02
    TRAFFIC_EVIDENCE_MIN = 0.5
    EPS = 1e-9

    # Redistribution caps and penalties
    PASS_CAPS = [0.25, 0.35, 0.45]  # ramped per-interface caps across passes
    ROUTER_TOTAL_DELTA_CAP_FRAC = 0.25  # cap on total absolute redistribution at a router (per direction)
    MULT_SCALE_CLAMP = (0.85, 1.15)  # gentle bound on multiplicative pre-step scaling

    def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
        return max(lo, min(hi, x))

    def rel_diff(a: float, b: float) -> float:
        denom = max(abs(a), abs(b), 1e-9)
        return abs(a - b) / denom

    def rate_aware_tol(v: float) -> float:
        # Relax tolerance at low traffic; never below core hardening threshold; cap at 10%
        return min(0.10, max(HARDENING_THRESHOLD, 5.0 / max(v, 1.0)))

    def logistic_conf(residual: float, tol: float, k: float = 3.0) -> float:
        tol = max(tol, 1e-9)
        x = residual / tol
        return clamp(1.0 / (1.0 + exp(k * (x - 1.0))))

    # Build connected pairs and peer map
    visited = set()
    pairs: List[Tuple[str, str]] = []
    peer_of: Dict[str, str] = {}
    for if_id, data in telemetry.items():
        peer = data.get('connected_to')
        if peer and peer in telemetry:
            key = tuple(sorted([if_id, peer]))
            if key not in visited:
                visited.add(key)
                a_id, b_id = key[0], key[1]
                pairs.append((a_id, b_id))
                peer_of[a_id] = b_id
                peer_of[b_id] = a_id

    paired_ids = set()
    for a_id, b_id in pairs:
        paired_ids.add(a_id)
        paired_ids.add(b_id)

    # Initialize interim structure
    interim: Dict[str, Dict[str, Any]] = {}
    for if_id, data in telemetry.items():
        interim[if_id] = {
            'rx': float(data.get('rx_rate', 0.0)),
            'tx': float(data.get('tx_rate', 0.0)),
            'rx_conf': 1.0,
            'tx_conf': 1.0,
            'status': data.get('interface_status', 'unknown'),
            'status_conf': 1.0,
            'connected_to': data.get('connected_to'),
            'local_router': data.get('local_router'),
            'remote_router': data.get('remote_router'),
            'orig_rx': float(data.get('rx_rate', 0.0)),
            'orig_tx': float(data.get('tx_rate', 0.0)),
            'orig_status': data.get('interface_status', 'unknown'),
            # tracking for confidence penalties
            'hit_cap_rx': False,
            'hit_cap_tx': False,
            'mult_mag_rx': 0.0,
            'mult_mag_tx': 0.0,
        }

    # Resolve status per pair using traffic evidence and enforce down=>zero
    for a_id, b_id in pairs:
        a = telemetry[a_id]; b = telemetry[b_id]
        a_stat = a.get('interface_status', 'unknown')
        b_stat = b.get('interface_status', 'unknown')
        a_rx, a_tx = float(a.get('rx_rate', 0.0)), float(a.get('tx_rate', 0.0))
        b_rx, b_tx = float(b.get('rx_rate', 0.0)), float(b.get('tx_rate', 0.0))
        max_traffic = max(a_rx, a_tx, b_rx, b_tx)
        if a_stat == b_stat:
            st = a_stat
            sc = 0.95 if st in ('up', 'down') else 0.7
        else:
            if max_traffic > TRAFFIC_EVIDENCE_MIN:
                st, sc = 'up', 0.85
            else:
                st, sc = 'down', 0.75
        interim[a_id]['status'] = st
        interim[b_id]['status'] = st
        interim[a_id]['status_conf'] = min(interim[a_id]['status_conf'], sc)
        interim[b_id]['status_conf'] = min(interim[b_id]['status_conf'], sc)

        if st == 'down':
            for ifid, rx0, tx0 in [(a_id, a_rx, a_tx), (b_id, b_rx, b_tx)]:
                interim[ifid]['rx'] = 0.0
                interim[ifid]['tx'] = 0.0
                interim[ifid]['rx_conf'] = 0.9 if rx0 <= TRAFFIC_EVIDENCE_MIN else 0.3
                interim[ifid]['tx_conf'] = 0.9 if tx0 <= TRAFFIC_EVIDENCE_MIN else 0.3

    # For unpaired, keep stated status but if unknown use traffic evidence
    for if_id, data in telemetry.items():
        if if_id in paired_ids:
            continue
        st = data.get('interface_status', 'unknown')
        rx0 = float(data.get('rx_rate', 0.0)); tx0 = float(data.get('tx_rate', 0.0))
        if st not in ('up', 'down'):
            if max(rx0, tx0) > TRAFFIC_EVIDENCE_MIN:
                st, sc = 'up', 0.8
            else:
                st, sc = 'down', 0.7
        else:
            sc = 0.95
        interim[if_id]['status'] = st
        interim[if_id]['status_conf'] = sc
        if st == 'down':
            interim[if_id]['rx'] = 0.0
            interim[if_id]['tx'] = 0.0
            interim[if_id]['rx_conf'] = 0.9 if rx0 <= TRAFFIC_EVIDENCE_MIN else 0.3
            interim[if_id]['tx_conf'] = 0.9 if tx0 <= TRAFFIC_EVIDENCE_MIN else 0.3

    # Pair-level hardening with rate-aware tolerance
    for a_id, b_id in pairs:
        if interim[a_id]['status'] == 'down':
            continue
        a = interim[a_id]; b = interim[b_id]
        # Forward: a.tx vs b.rx
        a_tx0 = float(telemetry[a_id].get('tx_rate', 0.0))
        b_rx0 = float(telemetry[b_id].get('rx_rate', 0.0))
        traff_fwd = max(a_tx0, b_rx0, 1.0)
        tol_fwd = rate_aware_tol(traff_fwd)
        d_fwd = rel_diff(a_tx0, b_rx0)
        if d_fwd <= tol_fwd:
            v = 0.5 * (a_tx0 + b_rx0)
        else:
            # Favor peer's counterpart as stronger redundant signal; special-case zeros
            if (a_tx0 <= TRAFFIC_EVIDENCE_MIN) and (b_rx0 > 1.0):
                v = b_rx0
            elif (b_rx0 <= TRAFFIC_EVIDENCE_MIN) and (a_tx0 > 1.0):
                v = a_tx0
            else:
                v = b_rx0
        a['tx'] = max(0.0, v)
        b['rx'] = max(0.0, v)

        # Reverse: a.rx vs b.tx
        a_rx0 = float(telemetry[a_id].get('rx_rate', 0.0))
        b_tx0 = float(telemetry[b_id].get('tx_rate', 0.0))
        traff_rev = max(a_rx0, b_tx0, 1.0)
        tol_rev = rate_aware_tol(traff_rev)
        d_rev = rel_diff(a_rx0, b_tx0)
        if d_rev <= tol_rev:
            v2 = 0.5 * (a_rx0 + b_tx0)
        else:
            if (a_rx0 <= TRAFFIC_EVIDENCE_MIN) and (b_tx0 > 1.0):
                v2 = b_tx0
            elif (b_tx0 <= TRAFFIC_EVIDENCE_MIN) and (a_rx0 > 1.0):
                v2 = a_rx0
            else:
                v2 = b_tx0
        a['rx'] = max(0.0, v2)
        b['tx'] = max(0.0, v2)

    # Build router->interfaces map
    router_ifaces: Dict[str, List[str]] = {}
    if topology:
        router_ifaces = {r: [i for i in if_list if i in interim] for r, if_list in topology.items()}
    else:
        # Fall back to local_router metadata if topology absent
        for if_id, data in telemetry.items():
            r = data.get('local_router')
            if r is not None:
                router_ifaces.setdefault(r, []).append(if_id)

    # Helper: compute pair residual per direction for weight enrichment
    def pair_residual_for_dir(if_id: str, direction: str) -> float:
        peer = peer_of.get(if_id)
        if peer is None:
            return 0.3  # moderate default for unpaired
        if interim[if_id]['status'] != 'up' or interim.get(peer, {}).get('status') != 'up':
            return 0.3
        if direction == 'rx':
            a = interim[if_id]['rx']; b = interim[peer]['tx']
        else:
            a = interim[if_id]['tx']; b = interim[peer]['rx']
        return rel_diff(a, b)

    # Router-level two-stage repair per router
    for router, if_list in router_ifaces.items():
        up_list = [i for i in if_list if interim[i]['status'] == 'up']
        if not up_list:
            continue

        # Stage 1: bounded multiplicative pre-step on less-trusted direction
        sum_tx = sum(max(0.0, interim[i]['tx']) for i in up_list)
        sum_rx = sum(max(0.0, interim[i]['rx']) for i in up_list)
        avg_tx_conf = sum(interim[i]['tx_conf'] for i in up_list) / max(1, len(up_list))
        avg_rx_conf = sum(interim[i]['rx_conf'] for i in up_list) / max(1, len(up_list))
        scale_rx = avg_tx_conf >= avg_rx_conf  # scale less-trusted direction toward the other
        if scale_rx and sum_rx > 0.0:
            s_raw = sum_tx / sum_rx
        elif (not scale_rx) and sum_tx > 0.0:
            s_raw = sum_rx / sum_tx
        else:
            s_raw = 1.0
        s_bounded = clamp(s_raw, MULT_SCALE_CLAMP[0], MULT_SCALE_CLAMP[1])
        s_step = sqrt(s_bounded)  # gentle step
        if abs(1.0 - s_step) > 1e-6:
            if scale_rx:
                for i in up_list:
                    old = interim[i]['rx']
                    new = max(0.0, old * s_step)
                    interim[i]['rx'] = new
                    interim[i]['mult_mag_rx'] += abs(1.0 - s_step)
                    # tiny penalty for pre-scaling
                    interim[i]['rx_conf'] = clamp(interim[i]['rx_conf'] - 0.05 * abs(1.0 - s_step))
            else:
                for i in up_list:
                    old = interim[i]['tx']
                    new = max(0.0, old * s_step)
                    interim[i]['tx'] = new
                    interim[i]['mult_mag_tx'] += abs(1.0 - s_step)
                    interim[i]['tx_conf'] = clamp(interim[i]['tx_conf'] - 0.05 * abs(1.0 - s_step))

        # Recompute sums after pre-step
        sum_tx = sum(max(0.0, interim[i]['tx']) for i in up_list)
        sum_rx = sum(max(0.0, interim[i]['rx']) for i in up_list)

        # Stage 2: adaptive additive redistribution with caps and weights
        def additive_pass(pass_idx: int):
            nonlocal sum_tx, sum_rx
            if scale_rx:
                values = {i: max(0.0, interim[i]['rx']) for i in up_list}
                target_total = sum_tx
                sum_old = sum(values.values())
                need = target_total - sum_old
            else:
                values = {i: max(0.0, interim[i]['tx']) for i in up_list}
                target_total = sum_rx
                sum_old = sum(values.values())
                need = target_total - sum_old

            # Abort small residuals
            if abs(need) <= max(sum_old, target_total, 1.0) * (HARDENING_THRESHOLD * 0.5):
                return

            # Router total-delta cap
            total_base = max(sum_old, 1.0)
            total_cap = ROUTER_TOTAL_DELTA_CAP_FRAC * total_base
            remaining_router_cap = total_cap

            # Build weights and per-interface cap for this pass
            sum_v = sum(values.values()) + EPS
            weights = {}
            caps_pos = {}
            caps_neg = {}
            for i in up_list:
                conf = interim[i]['rx_conf'] if scale_rx else interim[i]['tx_conf']
                v = values[i]
                # Enrich weights with pair residual and volume share
                pr = pair_residual_for_dir(i, 'rx' if scale_rx else 'tx')
                w = 0.5 * (1.0 - conf) + 0.3 * pr + 0.2 * (v / sum_v)
                if v < 1.0:
                    w *= 0.5
                weights[i] = max(0.05, w)
                cap_frac = PASS_CAPS[pass_idx]
                # On final pass, allow larger cap only for low-confidence or tiny baselines
                if pass_idx == 2 and not (conf < 0.6 or v < 5.0):
                    cap_frac = PASS_CAPS[1]
                cap = cap_frac * max(v, 1.0)
                caps_pos[i] = cap
                caps_neg[i] = cap

            # Two allocation rounds per pass
            for _round in range(2):
                if abs(need) <= EPS or remaining_router_cap <= EPS:
                    break
                elig = [i for i in up_list if (caps_pos[i] if need > 0 else caps_neg[i]) > EPS]
                if not elig:
                    break
                sumW = sum(weights[i] for i in elig) + EPS
                for i in elig:
                    if abs(need) <= EPS or remaining_router_cap <= EPS:
                        break
                    quota = need * (weights[i] / sumW)
                    if need > 0:
                        d = min(max(0.0, quota), caps_pos[i], remaining_router_cap)
                        caps_pos[i] -= d
                        if d >= caps_pos[i] - 1e-12:
                            if scale_rx:
                                interim[i]['hit_cap_rx'] = True
                            else:
                                interim[i]['hit_cap_tx'] = True
                    else:
                        d = max(min(0.0, quota), -caps_neg[i], -remaining_router_cap)
                        # d is negative; update cap flags
                        if -d >= caps_neg[i] - 1e-12:
                            if scale_rx:
                                interim[i]['hit_cap_rx'] = True
                            else:
                                interim[i]['hit_cap_tx'] = True
                        caps_neg[i] -= -d
                    if abs(d) <= EPS:
                        continue
                    old_v = values[i]
                    new_v = max(0.0, old_v + d)
                    values[i] = new_v
                    remaining_router_cap -= abs(d)
                    # Confidence penalty proportional to per-interface relative change and pre-step magnitude
                    delta_rel = rel_diff(old_v, new_v)
                    if scale_rx:
                        interim[i]['rx'] = new_v
                        interim[i]['rx_conf'] = clamp(interim[i]['rx_conf'] - (0.3 * delta_rel))
                    else:
                        interim[i]['tx'] = new_v
                        interim[i]['tx_conf'] = clamp(interim[i]['tx_conf'] - (0.3 * delta_rel))
                    need -= d

            # Update sums for next passes
            sum_tx_local = sum(max(0.0, interim[i]['tx']) for i in up_list)
            sum_rx_local = sum(max(0.0, interim[i]['rx']) for i in up_list)
            sum_tx = sum_tx_local
            sum_rx = sum_rx_local

        # Run three passes with adaptive caps
        for pidx in range(3):
            additive_pass(pidx)

    # Limited link-symmetry reconciliation after router redistribution
    for a_id, b_id in pairs:
        if interim[a_id]['status'] != 'up' or interim[b_id]['status'] != 'up':
            continue
        # Forward: a.tx vs b.rx
        a_tx = interim[a_id]['tx']; b_rx = interim[b_id]['rx']
        traffic = max(a_tx, b_rx, 1.0)
        tol = rate_aware_tol(traffic)
        resid = rel_diff(a_tx, b_rx)
        if resid > tol:
            v_mid = 0.5 * (a_tx + b_rx)
            # Adjust by up to 30% of residual; smaller if barely over tol
            alpha = min(0.3, max(0.0, (resid - tol) / max(resid, 1e-9)))
            a_new = max(0.0, a_tx + alpha * (v_mid - a_tx))
            b_new = max(0.0, b_rx + alpha * (v_mid - b_rx))
            if a_new != a_tx:
                drel = rel_diff(a_tx, a_new)
                interim[a_id]['tx'] = a_new
                interim[a_id]['tx_conf'] = clamp(interim[a_id]['tx_conf'] - 0.5 * drel)
            if b_new != b_rx:
                drel = rel_diff(b_rx, b_new)
                interim[b_id]['rx'] = b_new
                interim[b_id]['rx_conf'] = clamp(interim[b_id]['rx_conf'] - 0.5 * drel)
        # Reverse: a.rx vs b.tx
        a_rx = interim[a_id]['rx']; b_tx = interim[b_id]['tx']
        traffic2 = max(a_rx, b_tx, 1.0)
        tol2 = rate_aware_tol(traffic2)
        resid2 = rel_diff(a_rx, b_tx)
        if resid2 > tol2:
            v_mid2 = 0.5 * (a_rx + b_tx)
            alpha2 = min(0.3, max(0.0, (resid2 - tol2) / max(resid2, 1e-9)))
            a_new2 = max(0.0, a_rx + alpha2 * (v_mid2 - a_rx))
            b_new2 = max(0.0, b_tx + alpha2 * (v_mid2 - b_tx))
            if a_new2 != a_rx:
                drel = rel_diff(a_rx, a_new2)
                interim[a_id]['rx'] = a_new2
                interim[a_id]['rx_conf'] = clamp(interim[a_id]['rx_conf'] - 0.5 * drel)
            if b_new2 != b_tx:
                drel = rel_diff(b_tx, b_new2)
                interim[b_id]['tx'] = b_new2
                interim[b_id]['tx_conf'] = clamp(interim[b_id]['tx_conf'] - 0.5 * drel)

    # Enforce "down implies zero" once more for safety
    for if_id, r in interim.items():
        if r['status'] == 'down':
            r['rx'] = 0.0
            r['tx'] = 0.0
            r['rx_conf'] = 0.9 if r['orig_rx'] <= TRAFFIC_EVIDENCE_MIN else 0.3
            r['tx_conf'] = 0.9 if r['orig_tx'] <= TRAFFIC_EVIDENCE_MIN else 0.3

    # Final confidence calibration
    # Compute per-router imbalance residuals after all repairs
    router_final_imbalance: Dict[str, float] = {}
    for router, if_list in router_ifaces.items():
        up_ifaces = [i for i in if_list if interim[i]['status'] == 'up']
        if not up_ifaces:
            router_final_imbalance[router] = 0.0
            continue
        sum_tx = sum(max(0.0, interim[i]['tx']) for i in up_ifaces)
        sum_rx = sum(max(0.0, interim[i]['rx']) for i in up_ifaces)
        router_final_imbalance[router] = rel_diff(sum_tx, sum_rx)

    # Compute final confidences per interface direction
    for if_id, r in interim.items():
        peer = peer_of.get(if_id)
        status_comp = clamp(r.get('status_conf', 0.8))
        # Pair components
        if peer and interim.get(peer, {}).get('status') == r.get('status'):
            # TX: my.tx vs peer.rx
            res_tx = rel_diff(r['tx'], interim[peer]['rx'])
            tol_tx_pair = rate_aware_tol(max(r['tx'], interim[peer]['rx'], 1.0))
            pair_comp_tx = logistic_conf(res_tx, tol_tx_pair)
            # RX: my.rx vs peer.tx
            res_rx = rel_diff(r['rx'], interim[peer]['tx'])
            tol_rx_pair = rate_aware_tol(max(r['rx'], interim[peer]['tx'], 1.0))
            pair_comp_rx = logistic_conf(res_rx, tol_rx_pair)
        else:
            pair_comp_tx = 0.6
            pair_comp_rx = 0.6

        # Router component
        router = r.get('local_router')
        router_imb = router_final_imbalance.get(router, 0.0)
        # Use tolerance scaled to router traffic magnitude (approx via my totals)
        my_tot = max(r['tx'] + r['rx'], 1.0)
        router_comp = logistic_conf(router_imb, rate_aware_tol(my_tot))

        # Change component (magnitude of edit)
        delta_tx_rel = rel_diff(r['orig_tx'], r['tx'])
        delta_rx_rel = rel_diff(r['orig_rx'], r['rx'])
        change_comp_tx = logistic_conf(delta_tx_rel, rate_aware_tol(max(r['orig_tx'], r['tx'], 1.0)))
        change_comp_rx = logistic_conf(delta_rx_rel, rate_aware_tol(max(r['orig_rx'], r['rx'], 1.0)))

        # Base composition
        base_tx_conf = 0.55 * pair_comp_tx + 0.35 * router_comp + 0.10 * status_comp
        base_rx_conf = 0.55 * pair_comp_rx + 0.35 * router_comp + 0.10 * status_comp

        # Penalties for caps and multiplicative scaling
        penalty_tx = 1.0
        penalty_rx = 1.0
        if r.get('hit_cap_tx', False):
            penalty_tx *= 0.85
        if r.get('hit_cap_rx', False):
            penalty_rx *= 0.85
        if r.get('mult_mag_tx', 0.0) > 0.0:
            penalty_tx *= (1.0 - 0.2 * clamp(r['mult_mag_tx'], 0.0, 1.0))
        if r.get('mult_mag_rx', 0.0) > 0.0:
            penalty_rx *= (1.0 - 0.2 * clamp(r['mult_mag_rx'], 0.0, 1.0))

        final_tx_conf = clamp(base_tx_conf * change_comp_tx * penalty_tx)
        final_rx_conf = clamp(base_rx_conf * change_comp_rx * penalty_rx)

        # Down-state override
        if r['status'] == 'down':
            final_tx_conf = 0.9 if r['orig_tx'] <= TRAFFIC_EVIDENCE_MIN else 0.3
            final_rx_conf = 0.9 if r['orig_rx'] <= TRAFFIC_EVIDENCE_MIN else 0.3

        # Subtle status calibration for idle-up
        if r['status'] == 'up' and r['tx'] <= TRAFFIC_EVIDENCE_MIN and r['rx'] <= TRAFFIC_EVIDENCE_MIN:
            r['status_conf'] = clamp(r['status_conf'] * 0.9)

        r['tx_conf'] = final_tx_conf
        r['rx_conf'] = final_rx_conf

    # Assemble final output
    result: Dict[str, Dict[str, Tuple]] = {}
    for if_id, r in interim.items():
        out: Dict[str, Tuple] = {}
        out['rx_rate'] = (r['orig_rx'], r['rx'], clamp(r['rx_conf']))
        out['tx_rate'] = (r['orig_tx'], r['tx'], clamp(r['tx_conf']))
        out['interface_status'] = (r['orig_status'], r['status'], clamp(r['status_conf']))
        out['connected_to'] = r['connected_to']
        out['local_router'] = r['local_router']
        out['remote_router'] = r['remote_router']
        result[if_id] = out

    return result

# EVOLVE-BLOCK-END


def run_repair(telemetry: Dict[str, Dict[str, Any]], topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:
    """
    Main entry point that will be called by the evaluator.

    Args:
        telemetry: Network interface telemetry data
        topology: Dictionary where key is router_id and value contains a list of interface_ids

    Returns:
        Dictionary containing repaired results with confidence scores
    """
    return repair_network_telemetry(telemetry, topology)


if __name__ == "__main__":
    # Simple test case
    test_telemetry = {
        'if1_to_if2': {
            'interface_status': 'up',
            'rx_rate': 100.0,
            'tx_rate': 95.0,
            'connected_to': 'if2_to_if1',
            'local_router': 'router1',
            'remote_router': 'router2'
        },
        'if2_to_if1': {
            'interface_status': 'up',
            'rx_rate': 95.0,  # Should match if1's TX
            'tx_rate': 100.0,  # Should match if1's RX
            'connected_to': 'if1_to_if2',
            'local_router': 'router2',
            'remote_router': 'router1'
        }
    }

    test_topology = {
        'router1': ['if1_to_if2'],
        'router2': ['if2_to_if1']
    }

    result = run_repair(test_telemetry, test_topology)

    print("Repair results:")
    for if_id, data in result.items():
        print(f"\n{if_id}:")
        print(f"  RX: {data['rx_rate']}")
        print(f"  TX: {data['tx_rate']}")
        print(f"  Status: {data['interface_status']}")