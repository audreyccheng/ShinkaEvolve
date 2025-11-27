# EVOLVE-BLOCK-START
"""
Network telemetry repair algorithm that detects and corrects inconsistencies
in network interface telemetry data using topology relationships.

Takes interface telemetry data and detects/repairs inconsistencies based on
network invariants like link symmetry and flow conservation.
"""
from typing import Dict, Any, Tuple, List


def repair_network_telemetry(telemetry: Dict[str, Dict[str, Any]],
                             topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:
    """
    Repair network interface telemetry by detecting and correcting inconsistencies.

    Core principle: Use network invariants to validate and repair telemetry:
    1. Link Symmetry (R3): my_tx_rate ≈ their_rx_rate for connected interfaces
    2. Flow Conservation (R1): Sum(incoming traffic) = Sum(outgoing traffic) at each router
    3. Interface Consistency: Status should be consistent across connected pairs

    Args:
        telemetry: Dictionary where key is interface_id and value contains:
            - interface_status: "up" or "down"
            - rx_rate: receive rate in Mbps
            - tx_rate: transmit rate in Mbps
            - connected_to: interface_id this interface connects to
            - local_router: router_id this interface belongs to
            - remote_router: router_id on the other side
        topology: Dictionary where key is router_id and value contains a list of interface_ids

    Returns:
        Dictionary with same structure but telemetry values become tuples of:
        (original_value, repaired_value, confidence_score)
        where confidence ranges from 0.0 (very uncertain) to 1.0 (very confident)
    """

    # Measurement timing tolerance (from Hodor research: ~2%)
    HARDENING_THRESHOLD = 0.02
    # Small traffic level used to infer link up when statuses disagree (Mbps)
    TRAFFIC_EVIDENCE_MIN = 0.5
    # Max fractional per-interface adjustment during router redistribution
    MAX_ROUTER_ADJ_FRAC = 0.35
    # Limited reconciliation strength to pull pair endpoints toward symmetry
    PAIR_RECONCILE_ALPHA = 0.30
    # Confidence/weighting helpers
    TIE_EPS_CONF = 0.05
    TOL_PAIR_BASE = HARDENING_THRESHOLD * 1.5
    TOL_ROUTER = HARDENING_THRESHOLD * 2.0
    # Router-wide safety guard to avoid over-adjustment
    ROUTER_TOTAL_DELTA_GUARD = 0.25  # ≤25% of average up-traffic as total router delta
    EPS = 1e-9

    def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
        return max(lo, min(hi, x))

    def rel_diff(a: float, b: float) -> float:
        denom = max(abs(a), abs(b), 1e-9)
        return abs(a - b) / denom

    def conf_from_residual(residual: float, tol: float) -> float:
        # Two-slope residual-to-confidence:
        # - near 0 residual -> ~1
        # - residual ≈ tol -> ~0.8
        # - large residuals decay faster to avoid overconfidence
        tol = max(tol, 1e-9)
        x = residual / tol
        conf = 1.0 - min(1.0, x / 5.0)
        if x > 3.0:
            conf -= 0.1 * (x - 3.0) / 2.0
        return clamp(conf)

    # Initialize structures
    result: Dict[str, Dict[str, Tuple]] = {}
    interim: Dict[str, Dict[str, Any]] = {}

    # Build connected pairs
    visited = set()
    pairs: List[Tuple[str, str]] = []
    for if_id, data in telemetry.items():
        peer = data.get('connected_to')
        if peer and peer in telemetry:
            # Use ordered tuple to avoid duplicates
            key = tuple(sorted([if_id, peer]))
            if key not in visited:
                visited.add(key)
                pairs.append((key[0], key[1]))

    # Map each interface to its peer for quick lookup and record paired IDs
    peer_of: Dict[str, str] = {}
    paired_ids = set()
    for a_id, b_id in pairs:
        peer_of[a_id] = b_id
        peer_of[b_id] = a_id
        paired_ids.add(a_id)
        paired_ids.add(b_id)

    # Initialize defaults for all interfaces
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
            # Keep originals for output tuples
            'orig_rx': float(data.get('rx_rate', 0.0)),
            'orig_tx': float(data.get('tx_rate', 0.0)),
            'orig_status': data.get('interface_status', 'unknown'),
            # Edit tracking for confidence calibration
            'edit_rx_rel': 0.0,
            'edit_tx_rel': 0.0,
            'cap_hit_rx': False,
            'cap_hit_tx': False,
            'touched': False,
        }

    # Pair-level hardening using link symmetry (R3) and interface consistency
    for a_id, b_id in pairs:
        a = telemetry[a_id]
        b = telemetry[b_id]
        a_stat = a.get('interface_status', 'unknown')
        b_stat = b.get('interface_status', 'unknown')
        a_rx, a_tx = float(a.get('rx_rate', 0.0)), float(a.get('tx_rate', 0.0))
        b_rx, b_tx = float(b.get('rx_rate', 0.0)), float(b.get('tx_rate', 0.0))
        max_traffic = max(a_rx, a_tx, b_rx, b_tx)

        # Resolve interface status consistency across the link
        if a_stat == b_stat:
            resolved_status = a_stat
            status_conf = 0.95 if resolved_status in ('up', 'down') else 0.7
        else:
            # Use traffic evidence: if there is noticeable traffic, link must be up
            if max_traffic > TRAFFIC_EVIDENCE_MIN:
                resolved_status = 'up'
                status_conf = 0.85
            else:
                resolved_status = 'down'
                status_conf = 0.75

        # Apply status to both ends
        interim[a_id]['status'] = resolved_status
        interim[b_id]['status'] = resolved_status
        interim[a_id]['status_conf'] = min(interim[a_id]['status_conf'], status_conf) if interim[a_id]['status_conf'] else status_conf
        interim[b_id]['status_conf'] = min(interim[b_id]['status_conf'], status_conf) if interim[b_id]['status_conf'] else status_conf

        if resolved_status == 'down':
            # Down interfaces cannot send or receive
            # Confidence is high if original values were already near zero, lower otherwise.
            for ifid, rx0, tx0 in [(a_id, a_rx, a_tx), (b_id, b_rx, b_tx)]:
                interim[ifid]['rx'] = 0.0
                interim[ifid]['tx'] = 0.0
                interim[ifid]['rx_conf'] = clamp(0.9 if rx0 <= TRAFFIC_EVIDENCE_MIN else 0.3)
                interim[ifid]['tx_conf'] = clamp(0.9 if tx0 <= TRAFFIC_EVIDENCE_MIN else 0.3)
            continue  # No need to harden rates if link is down

        # Link is up: harden both directions using symmetry
        # Forward direction: a.tx should match b.rx
        d_fwd = rel_diff(a_tx, b_rx)
        if d_fwd <= HARDENING_THRESHOLD:
            v = 0.5 * (a_tx + b_rx)
            conf = clamp(1.0 - 0.5 * d_fwd)  # near 1 when very close
        else:
            # Choose peer's counterpart as stronger signal
            v = b_rx if abs(b_rx) > 0 else a_tx
            conf = clamp(1.0 - d_fwd)  # lower confidence for larger violation
        interim[a_id]['tx'] = v
        interim[b_id]['rx'] = v
        interim[a_id]['tx_conf'] = min(interim[a_id]['tx_conf'], conf)
        interim[b_id]['rx_conf'] = min(interim[b_id]['rx_conf'], conf)

        # Reverse direction: a.rx should match b.tx
        d_rev = rel_diff(a_rx, b_tx)
        if d_rev <= HARDENING_THRESHOLD:
            v2 = 0.5 * (a_rx + b_tx)
            conf2 = clamp(1.0 - 0.5 * d_rev)
        else:
            v2 = b_tx if abs(b_tx) > 0 else a_rx
            conf2 = clamp(1.0 - d_rev)
        interim[a_id]['rx'] = v2
        interim[b_id]['tx'] = v2
        interim[a_id]['rx_conf'] = min(interim[a_id]['rx_conf'], conf2)
        interim[b_id]['tx_conf'] = min(interim[b_id]['tx_conf'], conf2)

    # Enforce "down implies zero traffic" also for unpaired interfaces
    for if_id, r in interim.items():
        if if_id not in paired_ids and r.get('status') == 'down':
            rx0 = r['rx']
            tx0 = r['tx']
            r['rx'] = 0.0
            r['tx'] = 0.0
            r['rx_conf'] = clamp(0.9 if rx0 <= TRAFFIC_EVIDENCE_MIN else 0.3)
            r['tx_conf'] = clamp(0.9 if tx0 <= TRAFFIC_EVIDENCE_MIN else 0.3)

    # Router-level dynamic flow conservation (R1)
    # Build router to interfaces map (use provided topology if available, else derive from telemetry)
    router_ifaces: Dict[str, List[str]] = {}
    if topology:
        router_ifaces = {r: [i for i in if_list if i in interim] for r, if_list in topology.items()}
    else:
        # If topology not provided, derive from telemetry metadata
        for if_id, data in telemetry.items():
            r = data.get('local_router')
            if r is not None:
                router_ifaces.setdefault(r, []).append(if_id)

    # Helper to compute pair residual for a given interface and direction
    def pair_residual_dir(iface_id: str, direction: str) -> float:
        peer = peer_of.get(iface_id)
        if not peer or interim.get(peer) is None:
            return 0.0
        if interim[iface_id]['status'] != 'up' or interim[peer]['status'] != 'up':
            return 0.0
        if direction == 'rx':
            return rel_diff(interim[iface_id]['rx'], interim[peer]['tx'])
        else:
            return rel_diff(interim[iface_id]['tx'], interim[peer]['rx'])

    # Uncertainty-weighted, capacity-capped redistribution towards flow conservation
    for router, if_list in router_ifaces.items():
        # Consider only interfaces present in telemetry
        interfaces = [i for i in if_list if i in interim]
        if not interfaces:
            continue

        # Compute sums over "up" interfaces
        sum_tx = 0.0
        sum_rx = 0.0
        tx_conf_acc = 0.0
        rx_conf_acc = 0.0
        up_list = []
        for i in interfaces:
            if interim[i]['status'] == 'up':
                up_list.append(i)
                sum_tx += max(0.0, interim[i]['tx'])
                sum_rx += max(0.0, interim[i]['rx'])
                tx_conf_acc += interim[i]['tx_conf']
                rx_conf_acc += interim[i]['rx_conf']

        if not up_list:
            continue

        avg_tx_conf = tx_conf_acc / max(1, len(up_list))
        avg_rx_conf = rx_conf_acc / max(1, len(up_list))

        # Evaluate flow imbalance
        imbalance = rel_diff(sum_tx, sum_rx)
        if imbalance <= HARDENING_THRESHOLD * 2:
            # Within tolerance; no router-level scaling needed
            pass
        else:
            # Decide which direction to scale: scale the less trusted direction; tie-break by larger absolute need
            need_rx = sum_tx - sum_rx  # need applied to RX to match TX
            need_tx = -need_rx         # need applied to TX to match RX
            if abs(avg_tx_conf - avg_rx_conf) < TIE_EPS_CONF:
                scale_rx = abs(need_rx) >= abs(need_tx)
            else:
                scale_rx = avg_tx_conf >= avg_rx_conf  # if TX more trusted, scale RX to match TX

            if scale_rx and sum_rx > 0.0:
                s = sum_tx / sum_rx
            elif (not scale_rx) and sum_tx > 0.0:
                s = sum_rx / sum_tx
            else:
                s = 1.0

            # Bound scaling magnitude for confidence penalty computation
            s_bounded = max(0.5, min(2.0, s))

            # Weighted additive redistribution toward target using lower-trust interfaces more
            if scale_rx:
                sum_old = sum(max(0.0, interim[i]['rx']) for i in up_list)
                target_total = sum_tx
                dir_key = 'rx'
            else:
                sum_old = sum(max(0.0, interim[i]['tx']) for i in up_list)
                target_total = sum_rx
                dir_key = 'tx'

            need = target_total - sum_old
            if abs(need) > max(sum_old, target_total, 1.0) * (HARDENING_THRESHOLD * 0.5):
                # Build weights from direction-specific confidence and pair residual (lower confidence -> larger weight)
                weights: Dict[str, float] = {}
                caps_pos: Dict[str, float] = {}
                caps_neg: Dict[str, float] = {}
                values: Dict[str, float] = {}

                sum_v_for_norm = sum(max(0.0, interim[i][dir_key]) for i in up_list) + EPS
                for i in up_list:
                    conf = float(interim[i][f'{dir_key}_conf'])
                    v = max(0.0, float(interim[i][dir_key]))
                    # Pair residual term scaled by tolerance
                    pair_resid = pair_residual_dir(i, dir_key)
                    pair_term = min(3.0, pair_resid / max(TOL_PAIR_BASE, 1e-9))
                    w = 0.6 * (1.0 - conf) + 0.25 * pair_term + 0.15 * (v / sum_v_for_norm)
                    weights[i] = max(0.02, w)
                    cap = MAX_ROUTER_ADJ_FRAC * max(v, 1.0)
                    caps_pos[i] = cap
                    caps_neg[i] = cap
                    values[i] = v

                # Router total delta guard to avoid over-adjustment
                avg_up_traffic = 0.5 * (sum_tx + sum_rx)
                router_delta_guard = ROUTER_TOTAL_DELTA_GUARD * avg_up_traffic
                router_delta_used = 0.0

                # Iterative allocation with capacity clipping and router guard
                alloc_passes = 2
                for _alloc in range(alloc_passes):
                    if abs(need) <= EPS:
                        break
                    if need > 0:
                        elig = [i for i in up_list if caps_pos[i] > EPS]
                    else:
                        elig = [i for i in up_list if caps_neg[i] > EPS]
                    if not elig:
                        break
                    sumW = sum(weights[i] for i in elig)
                    if sumW <= EPS:
                        break
                    for i in elig:
                        if abs(need) <= EPS:
                            break
                        quota = need * (weights[i] / sumW)
                        if need > 0:
                            d = min(max(0.0, quota), caps_pos[i])
                            caps_pos[i] -= d
                            cap_depleted = caps_pos[i] <= EPS * 10
                        else:
                            d = max(min(0.0, quota), -caps_neg[i])
                            caps_neg[i] -= -d
                            cap_depleted = caps_neg[i] <= EPS * 10

                        # Apply router guard
                        remaining_guard = router_delta_guard - router_delta_used
                        if abs(d) > remaining_guard:
                            d = max(-remaining_guard, min(remaining_guard, d))

                        if abs(d) <= EPS:
                            continue

                        old_v = values[i]
                        new_v = max(0.0, old_v + d)
                        values[i] = new_v
                        # Confidence drops with global imbalance, scaling magnitude and per-interface change
                        delta_rel = rel_diff(old_v, new_v)
                        if dir_key == 'rx':
                            interim[i]['rx'] = new_v
                            interim[i]['rx_conf'] = clamp(min(interim[i]['rx_conf'],
                                                              1.0 - min(1.0, imbalance + 0.5 * delta_rel + abs(1.0 - s_bounded) * 0.5)))
                            interim[i]['edit_rx_rel'] = max(interim[i]['edit_rx_rel'], delta_rel)
                            if cap_depleted:
                                interim[i]['cap_hit_rx'] = True
                        else:
                            interim[i]['tx'] = new_v
                            interim[i]['tx_conf'] = clamp(min(interim[i]['tx_conf'],
                                                              1.0 - min(1.0, imbalance + 0.5 * delta_rel + abs(1.0 - s_bounded) * 0.5)))
                            interim[i]['edit_tx_rel'] = max(interim[i]['edit_tx_rel'], delta_rel)
                            if cap_depleted:
                                interim[i]['cap_hit_tx'] = True

                        interim[i]['touched'] = True
                        router_delta_used += abs(d)
                        need -= d
                        if router_delta_used >= router_delta_guard - EPS:
                            break
                    if router_delta_used >= router_delta_guard - EPS:
                        break

    # Targeted, tolerance-gated pair reconciliation after router redistribution
    for a_id, b_id in pairs:
        if a_id not in interim or b_id not in interim:
            continue
        if interim[a_id].get('status') != 'up' or interim[b_id].get('status') != 'up':
            continue

        # Forward direction: a.tx vs b.rx
        a_tx_old = interim[a_id]['tx']
        b_rx_old = interim[b_id]['rx']
        traffic_tx = max(a_tx_old, b_rx_old, 1.0)
        tol_pair_post = max(0.02, 2.5 / traffic_tx)
        res_fwd = rel_diff(a_tx_old, b_rx_old)
        if res_fwd > tol_pair_post:
            v_mid = 0.5 * (a_tx_old + b_rx_old)
            move_a = PAIR_RECONCILE_ALPHA * (v_mid - a_tx_old)
            move_b = PAIR_RECONCILE_ALPHA * (v_mid - b_rx_old)
            cap_a = 0.20 * max(a_tx_old, 1.0)
            cap_b = 0.20 * max(b_rx_old, 1.0)
            move_a = max(-cap_a, min(cap_a, move_a))
            move_b = max(-cap_b, min(cap_b, move_b))
            a_tx_new = max(0.0, a_tx_old + move_a)
            b_rx_new = max(0.0, b_rx_old + move_b)
            if a_tx_new != a_tx_old:
                delta_rel = rel_diff(a_tx_old, a_tx_new)
                interim[a_id]['tx'] = a_tx_new
                interim[a_id]['tx_conf'] = clamp(interim[a_id]['tx_conf'] * (1.0 - 0.3 * min(1.0, res_fwd / max(tol_pair_post, 1e-9))))
                interim[a_id]['edit_tx_rel'] = max(interim[a_id]['edit_tx_rel'], delta_rel)
                interim[a_id]['touched'] = True
            if b_rx_new != b_rx_old:
                delta_rel = rel_diff(b_rx_old, b_rx_new)
                interim[b_id]['rx'] = b_rx_new
                interim[b_id]['rx_conf'] = clamp(interim[b_id]['rx_conf'] * (1.0 - 0.3 * min(1.0, res_fwd / max(tol_pair_post, 1e-9))))
                interim[b_id]['edit_rx_rel'] = max(interim[b_id]['edit_rx_rel'], delta_rel)
                interim[b_id]['touched'] = True

        # Reverse direction: a.rx vs b.tx
        a_rx_old = interim[a_id]['rx']
        b_tx_old = interim[b_id]['tx']
        traffic_rx = max(a_rx_old, b_tx_old, 1.0)
        tol_pair_post2 = max(0.02, 2.5 / traffic_rx)
        res_rev = rel_diff(a_rx_old, b_tx_old)
        if res_rev > tol_pair_post2:
            v_mid2 = 0.5 * (a_rx_old + b_tx_old)
            move_a2 = PAIR_RECONCILE_ALPHA * (v_mid2 - a_rx_old)
            move_b2 = PAIR_RECONCILE_ALPHA * (v_mid2 - b_tx_old)
            cap_a2 = 0.20 * max(a_rx_old, 1.0)
            cap_b2 = 0.20 * max(b_tx_old, 1.0)
            move_a2 = max(-cap_a2, min(cap_a2, move_a2))
            move_b2 = max(-cap_b2, min(cap_b2, move_b2))
            a_rx_new = max(0.0, a_rx_old + move_a2)
            b_tx_new = max(0.0, b_tx_old + move_b2)
            if a_rx_new != a_rx_old:
                delta_rel = rel_diff(a_rx_old, a_rx_new)
                interim[a_id]['rx'] = a_rx_new
                interim[a_id]['rx_conf'] = clamp(interim[a_id]['rx_conf'] * (1.0 - 0.3 * min(1.0, res_rev / max(tol_pair_post2, 1e-9))))
                interim[a_id]['edit_rx_rel'] = max(interim[a_id]['edit_rx_rel'], delta_rel)
                interim[a_id]['touched'] = True
            if b_tx_new != b_tx_old:
                delta_rel = rel_diff(b_tx_old, b_tx_new)
                interim[b_id]['tx'] = b_tx_new
                interim[b_id]['tx_conf'] = clamp(interim[b_id]['tx_conf'] * (1.0 - 0.3 * min(1.0, res_rev / max(tol_pair_post2, 1e-9))))
                interim[b_id]['edit_tx_rel'] = max(interim[b_id]['edit_tx_rel'], delta_rel)
                interim[b_id]['touched'] = True

    # Final confidence calibration based on post-repair invariants
    # Compute per-router imbalance residuals
    router_final_imbalance: Dict[str, float] = {}
    for router, if_list in router_ifaces.items():
        # only consider interfaces that are in interim and up
        up_ifaces = [i for i in if_list if i in interim and interim[i].get('status') == 'up']
        if not up_ifaces:
            router_final_imbalance[router] = 0.0
            continue
        sum_tx = sum(max(0.0, interim[i]['tx']) for i in up_ifaces)
        sum_rx = sum(max(0.0, interim[i]['rx']) for i in up_ifaces)
        router_final_imbalance[router] = rel_diff(sum_tx, sum_rx)

    # Weights and tolerances for confidence components
    w_pair, w_router, w_status = 0.6, 0.3, 0.1

    def finalize_conf(base: float, edit_rel: float, cap_hit: bool, orig_val: float, new_val: float) -> float:
        # Multiplicative penalty for large edits; softer near threshold
        pen_factor = max(0.0, 1.0 - 0.6 * max(0.0, edit_rel - HARDENING_THRESHOLD))
        conf = clamp(base * pen_factor)
        if cap_hit:
            conf *= 0.9
        # No-edit bonus
        if rel_diff(orig_val, new_val) <= 1e-3:
            conf = clamp(conf + 0.05)
        return clamp(conf)

    for if_id, r in interim.items():
        router = r.get('local_router')
        peer = peer_of.get(if_id)

        status_comp = clamp(r.get('status_conf', 0.8))
        resolved_status = r.get('status', 'unknown')

        if peer and interim.get(peer, {}).get('status') == resolved_status:
            res_fwd = rel_diff(r['tx'], interim[peer]['rx'])
            res_rev = rel_diff(r['rx'], interim[peer]['tx'])
            # Rate-aware pair tolerances to avoid over-penalizing low-rate links
            traffic_tx = max(r['tx'], interim[peer]['rx'], 1.0)
            traffic_rx = max(r['rx'], interim[peer]['tx'], 1.0)
            tol_pair_tx = min(0.12, max(TOL_PAIR_BASE, 5.0 / traffic_tx))
            tol_pair_rx = min(0.12, max(TOL_PAIR_BASE, 5.0 / traffic_rx))
            pair_comp_tx = conf_from_residual(res_fwd, tol_pair_tx)
            pair_comp_rx = conf_from_residual(res_rev, tol_pair_rx)
        else:
            pair_comp_tx = 0.55
            pair_comp_rx = 0.55

        router_imb = router_final_imbalance.get(router, 0.0)
        router_comp = conf_from_residual(router_imb, TOL_ROUTER)

        base_tx_conf = w_pair * pair_comp_tx + w_router * router_comp + w_status * status_comp
        base_rx_conf = w_pair * pair_comp_rx + w_router * router_comp + w_status * status_comp

        final_tx_conf = finalize_conf(base_tx_conf, r.get('edit_tx_rel', 0.0), r.get('cap_hit_tx', False), r['orig_tx'], r['tx'])
        final_rx_conf = finalize_conf(base_rx_conf, r.get('edit_rx_rel', 0.0), r.get('cap_hit_rx', False), r['orig_rx'], r['rx'])

        if resolved_status == 'down':
            final_rx_conf = 0.9 if r['orig_rx'] <= TRAFFIC_EVIDENCE_MIN else 0.3
            final_tx_conf = 0.9 if r['orig_tx'] <= TRAFFIC_EVIDENCE_MIN else 0.3

        r['tx_conf'] = clamp(final_tx_conf)
        r['rx_conf'] = clamp(final_rx_conf)

        # Subtle status calibration: if up but effectively idle, reduce status confidence slightly
        if resolved_status == 'up':
            if r['rx'] <= TRAFFIC_EVIDENCE_MIN and r['tx'] <= TRAFFIC_EVIDENCE_MIN:
                r['status_conf'] = clamp(r['status_conf'] * 0.9)
        elif resolved_status == 'down':
            if r['rx'] > TRAFFIC_EVIDENCE_MIN or r['tx'] > TRAFFIC_EVIDENCE_MIN:
                r['status_conf'] = clamp(min(r['status_conf'], 0.3))

    # Assemble final result with (original, repaired, confidence) tuples and unchanged metadata
    for if_id, data in telemetry.items():
        repaired_data: Dict[str, Tuple] = {}
        r = interim[if_id]

        repaired_data['rx_rate'] = (r['orig_rx'], r['rx'], clamp(r['rx_conf']))
        repaired_data['tx_rate'] = (r['orig_tx'], r['tx'], clamp(r['tx_conf']))
        repaired_data['interface_status'] = (r['orig_status'], r['status'], clamp(r['status_conf']))

        # Copy metadata unchanged
        repaired_data['connected_to'] = r['connected_to']
        repaired_data['local_router'] = r['local_router']
        repaired_data['remote_router'] = r['remote_router']

        result[if_id] = repaired_data

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