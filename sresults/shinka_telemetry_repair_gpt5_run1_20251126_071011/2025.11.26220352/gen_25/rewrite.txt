# EVOLVE-BLOCK-START
"""
Network telemetry repair algorithm that detects and corrects inconsistencies
in network interface telemetry data using topology relationships.

Novel approach:
- Rate-aware tolerance and logistic confidence
- Bounded multiplicative router pre-step, then confidence-weighted additive redistribution with ramped caps
- Limited post-router link symmetry reconciliation
- Change/cap/scale penalties for confidence calibration
"""
from typing import Dict, Any, Tuple, List
import math


def repair_network_telemetry(telemetry: Dict[str, Dict[str, Any]],
                             topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:
    """
    Repair network interface telemetry by detecting and correcting inconsistencies.

    Core principles (invariants):
    1. Link Symmetry (R3): my_tx_rate ≈ their_rx_rate for connected interfaces
    2. Flow Conservation (R1): Sum(incoming traffic) = Sum(outgoing traffic) at each router
    3. Interface Consistency: Status should be consistent across connected pairs

    Args:
        telemetry: Dictionary per interface_id with:
            - interface_status: "up" or "down"
            - rx_rate: receive rate in Mbps
            - tx_rate: transmit rate in Mbps
            - connected_to: peer interface_id
            - local_router: owning router_id
            - remote_router: remote router_id
        topology: Dictionary router_id -> list of interface_ids

    Returns:
        Dictionary where telemetry fields become tuples: (original_value, repaired_value, confidence)
        Non-telemetry fields (connected_to, local_router, remote_router) are copied unchanged.
    """
    # Tuning constants
    HARDENING_THRESHOLD = 0.02  # base tolerance for timing skew
    EPS = 1e-9

    # Helper functions
    def norm_status(s: Any) -> str:
        s = str(s).lower()
        return s if s in ("up", "down") else "up"

    def nz_float(x: Any) -> float:
        try:
            v = float(x)
        except Exception:
            v = 0.0
        return max(0.0, v)

    def rel_diff(a: float, b: float) -> float:
        a = float(a)
        b = float(b)
        denom = max(abs(a), abs(b), 1.0)
        return abs(a - b) / denom

    def clamp(x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))

    def pair_tolerance(a: float, b: float) -> float:
        # Rate-aware tolerance: at least HARDENING_THRESHOLD and looser for low traffic (~5 Mbps)
        traffic = max(abs(a), abs(b), 1.0)
        return max(HARDENING_THRESHOLD, 5.0 / traffic)

    def conf_from_residual(residual: float, tol: float, k: float = 3.0) -> float:
        # Logistic decay: high confidence within tolerance, lower beyond
        x = residual / max(tol, 1e-9)
        return 1.0 / (1.0 + math.exp(k * (x - 1.0)))

    # Build peer mapping
    peers: Dict[str, str] = {iface: data.get('connected_to') for iface, data in telemetry.items()}

    # First pass: link-level signal hardening (R3) with status-aware zeroing
    pre: Dict[str, Dict[str, Any]] = {}
    for iface, data in telemetry.items():
        status = norm_status(data.get('interface_status', 'unknown'))
        rx_orig = nz_float(data.get('rx_rate', 0.0))
        tx_orig = nz_float(data.get('tx_rate', 0.0))

        peer_id = peers.get(iface)
        had_peer = bool(peer_id and peer_id in telemetry)
        peer_status = 'unknown'
        peer_rx = peer_tx = 0.0
        if had_peer:
            p = telemetry[peer_id]
            peer_status = norm_status(p.get('interface_status', 'unknown'))
            peer_rx = nz_float(p.get('rx_rate', 0.0))
            peer_tx = nz_float(p.get('tx_rate', 0.0))

        pre_rx = rx_orig
        pre_tx = tx_orig
        rx_resid = 0.0
        tx_resid = 0.0
        rx_tol = HARDENING_THRESHOLD
        tx_tol = HARDENING_THRESHOLD

        if status == 'down':
            # Down interfaces cannot carry traffic
            pre_rx = 0.0
            pre_tx = 0.0
        else:
            if had_peer and peer_status == 'up':
                # My RX should match peer TX; My TX should match peer RX
                rx_resid = rel_diff(rx_orig, peer_tx)
                tx_resid = rel_diff(tx_orig, peer_rx)
                rx_tol = pair_tolerance(rx_orig, peer_tx)
                tx_tol = pair_tolerance(tx_orig, peer_rx)

                # Triage substitution vs averaging
                if tx_resid <= tx_tol and rx_resid > rx_tol:
                    pre_rx = peer_tx
                elif rx_resid <= rx_tol and tx_resid > tx_tol:
                    pre_tx = peer_rx
                elif rx_resid > rx_tol and tx_resid > tx_tol:
                    pre_rx = 0.5 * (rx_orig + peer_tx)
                    pre_tx = 0.5 * (tx_orig + peer_rx)

        pre[iface] = {
            'pre_rx': pre_rx,
            'pre_tx': pre_tx,
            'rx_resid': rx_resid,
            'tx_resid': tx_resid,
            'rx_tol': rx_tol,
            'tx_tol': tx_tol,
            'local_status': status,
            'peer_status': peer_status if had_peer else 'unknown',
            'had_peer': had_peer
        }

    # Working store for subsequent adjustments
    work: Dict[str, Dict[str, float]] = {i: {'rx': pre[i]['pre_rx'], 'tx': pre[i]['pre_tx']} for i in telemetry}

    # Penalty bookkeeping for confidence calibration
    penalties: Dict[str, Dict[str, float]] = {
        i: {'rx_cap': 0.0, 'tx_cap': 0.0, 'rx_pair_adj': 0.0, 'tx_pair_adj': 0.0,
            'rx_scale': 0.0, 'tx_scale': 0.0}
        for i in telemetry
    }

    # Second pass: router-level flow conservation (R1)
    # We use the topology to group interfaces per router.
    router_final_imbalance: Dict[str, float] = {}

    for router_id, iface_list in topology.items():
        # consider only known interfaces
        ifaces = [i for i in iface_list if i in telemetry]
        if not ifaces:
            router_final_imbalance[router_id] = 0.0
            continue

        # Use only locally up interfaces (down already zeroed)
        up_ifaces = [i for i in ifaces if pre[i]['local_status'] == 'up']
        if not up_ifaces:
            router_final_imbalance[router_id] = 0.0
            continue

        def router_sums() -> Tuple[float, float]:
            srx = sum(work[i]['rx'] for i in up_ifaces)
            stx = sum(work[i]['tx'] for i in up_ifaces)
            return srx, stx

        sum_rx, sum_tx = router_sums()
        if (sum_rx + sum_tx) <= 1e-9:
            router_final_imbalance[router_id] = 0.0
            continue

        denom = max(sum_rx, sum_tx, 1.0)
        imb = abs(sum_tx - sum_rx) / denom

        # Bounded multiplicative pre-step (gentle move toward balance)
        if imb > HARDENING_THRESHOLD and len(up_ifaces) >= 2:
            if sum_tx > sum_rx:
                target = clamp(sum_rx / max(sum_tx, EPS), 0.85, 1.15)
                s_step = target ** 0.5
                for i in up_ifaces:
                    old = work[i]['tx']
                    work[i]['tx'] = max(0.0, old * s_step)
                    penalties[i]['tx_scale'] = max(penalties[i]['tx_scale'], abs(1.0 - s_step))
            else:
                target = clamp(sum_tx / max(sum_rx, EPS), 0.85, 1.15)
                s_step = target ** 0.5
                for i in up_ifaces:
                    old = work[i]['rx']
                    work[i]['rx'] = max(0.0, old * s_step)
                    penalties[i]['rx_scale'] = max(penalties[i]['rx_scale'], abs(1.0 - s_step))

        # Recompute imbalance after pre-step
        sum_rx, sum_tx = router_sums()
        need = sum_tx - sum_rx
        need_abs = abs(need)
        denom = max(sum_rx, sum_tx, 1.0)

        # Confidence-weighted additive redistribution with ramped caps
        if need_abs / denom > HARDENING_THRESHOLD and len(up_ifaces) >= 2:
            total_traffic = sum_rx + sum_tx
            router_cap_total = 0.25 * total_traffic  # per-router total edit cap
            remaining = min(need_abs, router_cap_total)

            over_dir = 'tx' if need > 0 else 'rx'
            opp_dir = 'rx' if over_dir == 'tx' else 'tx'
            values = {i: work[i][over_dir] for i in up_ifaces}
            sum_v = sum(values.values()) + EPS

            # Directional confidence and residual vs peer
            dir_conf: Dict[str, float] = {}
            dir_resid: Dict[str, float] = {}
            for i in up_ifaces:
                peer_id = peers.get(i)
                if peer_id and peer_id in telemetry and pre.get(peer_id, {}).get('local_status') == 'up':
                    if over_dir == 'tx':
                        resid = rel_diff(work[i]['tx'], work[peer_id]['rx'])
                        tol = pair_tolerance(work[i]['tx'], work[peer_id]['rx'])
                    else:
                        resid = rel_diff(work[i]['rx'], work[peer_id]['tx'])
                        tol = pair_tolerance(work[i]['rx'], work[peer_id]['tx'])
                    dir_resid[i] = resid
                    dir_conf[i] = conf_from_residual(resid, tol)
                else:
                    dir_resid[i] = 0.0
                    dir_conf[i] = 0.6  # moderate default

            pass_caps = [0.25, 0.35, 0.45]
            for pass_idx, max_frac in enumerate(pass_caps):
                if remaining <= 1e-9:
                    break

                # Build weights emphasizing low confidence and high residual; de-emphasize near-idle
                weights: Dict[str, float] = {}
                for i in up_ifaces:
                    v = values[i]
                    resid = dir_resid.get(i, 0.0)
                    conf_d = dir_conf.get(i, 0.6)
                    w = 0.5 * (1.0 - conf_d) + 0.3 * resid + 0.2 * (v / sum_v)
                    if v < 5.0:
                        w *= 0.5  # avoid over-editing idle links
                    # In the last pass, allow large caps only for low-confidence or tiny baselines
                    if pass_idx == 2 and not (conf_d < 0.6 or v < 5.0):
                        w *= 0.1
                    weights[i] = max(w, 1e-6)

                sum_w = sum(weights.values())
                if sum_w <= 0:
                    break

                # Allocate edits this pass
                to_apply = remaining
                for i in up_ifaces:
                    if to_apply <= 1e-9:
                        break
                    v = values[i]
                    alloc = to_apply * (weights[i] / sum_w)
                    cap_i = max(0.1, v * max_frac)
                    delta_i = min(alloc, cap_i, v)  # do not cross zero
                    if delta_i <= 0:
                        continue
                    if over_dir == 'tx':
                        work[i]['tx'] = max(0.0, work[i]['tx'] - delta_i)
                        # penalize if we hit cap or nearly zeroed
                        if abs(delta_i - cap_i) < 1e-9 or delta_i >= v - 1e-9:
                            penalties[i]['tx_cap'] = max(penalties[i]['tx_cap'], 0.15)
                    else:
                        work[i]['rx'] = max(0.0, work[i]['rx'] - delta_i)
                        if abs(delta_i - cap_i) < 1e-9 or delta_i >= v - 1e-9:
                            penalties[i]['rx_cap'] = max(penalties[i]['rx_cap'], 0.15)
                    values[i] = max(0.0, v - delta_i)
                    remaining -= delta_i

        # Final imbalance after router adjustments
        sum_rx, sum_tx = router_sums()
        router_final_imbalance[router_id] = abs(sum_tx - sum_rx) / max(sum_rx, sum_tx, 1.0)

    # Third pass: limited link reconciliation after router edits (sustain R3)
    visited = set()
    for iface in telemetry:
        peer_id = peers.get(iface)
        if not (peer_id and peer_id in telemetry):
            continue
        key = tuple(sorted([iface, peer_id]))
        if key in visited:
            continue
        visited.add(key)

        if pre[iface]['local_status'] != 'up' or pre.get(peer_id, {}).get('local_status') != 'up':
            continue

        # A.tx <-> B.rx reconciliation
        a_tx = work[iface]['tx']
        b_rx = work[peer_id]['rx']
        tol_tx = pair_tolerance(a_tx, b_rx)
        resid_tx = rel_diff(a_tx, b_rx)
        if resid_tx > tol_tx:
            alpha = 0.3
            a_tx_new = a_tx + alpha * (b_rx - a_tx)
            b_rx_new = b_rx + alpha * (a_tx - b_rx)
            work[iface]['tx'] = max(0.0, a_tx_new)
            work[peer_id]['rx'] = max(0.0, b_rx_new)
            penalties[iface]['tx_pair_adj'] = max(penalties[iface]['tx_pair_adj'], 0.1)
            penalties[peer_id]['rx_pair_adj'] = max(penalties[peer_id]['rx_pair_adj'], 0.1)

        # A.rx <-> B.tx reconciliation
        a_rx = work[iface]['rx']
        b_tx = work[peer_id]['tx']
        tol_rx = pair_tolerance(a_rx, b_tx)
        resid_rx = rel_diff(a_rx, b_tx)
        if resid_rx > tol_rx:
            alpha = 0.3
            a_rx_new = a_rx + alpha * (b_tx - a_rx)
            b_tx_new = b_tx + alpha * (a_rx - b_tx)
            work[iface]['rx'] = max(0.0, a_rx_new)
            work[peer_id]['tx'] = max(0.0, b_tx_new)
            penalties[iface]['rx_pair_adj'] = max(penalties[iface]['rx_pair_adj'], 0.1)
            penalties[peer_id]['tx_pair_adj'] = max(penalties[peer_id]['tx_pair_adj'], 0.1)

    # Recompute final router imbalances for confidence (post pair reconciliation)
    for router_id, iface_list in topology.items():
        ifaces = [i for i in iface_list if i in telemetry and pre[i]['local_status'] == 'up']
        if not ifaces:
            router_final_imbalance[router_id] = router_final_imbalance.get(router_id, 0.0)
        else:
            sum_rx = sum(work[i]['rx'] for i in ifaces)
            sum_tx = sum(work[i]['tx'] for i in ifaces)
            router_final_imbalance[router_id] = abs(sum_tx - sum_rx) / max(sum_rx, sum_tx, 1.0)

    # Assemble results with calibrated confidence
    result: Dict[str, Dict[str, Tuple]] = {}
    for iface, data in telemetry.items():
        status_raw = data.get('interface_status', 'unknown')
        status = norm_status(status_raw)
        rx_orig = nz_float(data.get('rx_rate', 0.0))
        tx_orig = nz_float(data.get('tx_rate', 0.0))

        # Apply final values
        rx_repaired = work[iface]['rx']
        tx_repaired = work[iface]['tx']

        # Enforce zero for down status
        if status == 'down':
            rx_repaired = 0.0
            tx_repaired = 0.0

        # Link-based confidence using final residuals
        peer_id = peers.get(iface)
        if peer_id and peer_id in telemetry and pre[iface]['local_status'] == 'up' and pre.get(peer_id, {}).get('local_status') == 'up':
            peer_tx_final = work[peer_id]['tx']
            peer_rx_final = work[peer_id]['rx']
            rx_resid_f = rel_diff(rx_repaired, peer_tx_final)
            tx_resid_f = rel_diff(tx_repaired, peer_rx_final)
            rx_tol_f = pair_tolerance(rx_repaired, peer_tx_final)
            tx_tol_f = pair_tolerance(tx_repaired, peer_rx_final)
            rx_link_conf = conf_from_residual(rx_resid_f, rx_tol_f)
            tx_link_conf = conf_from_residual(tx_resid_f, tx_tol_f)
        elif status == 'down':
            rx_link_conf = 0.9 if rx_repaired == 0.0 else 0.5
            tx_link_conf = 0.9 if tx_repaired == 0.0 else 0.5
        else:
            rx_link_conf = 0.6
            tx_link_conf = 0.6

        # Router imbalance factor
        router_id = data.get('local_router')
        r_factor = max(0.2, 1.0 - router_final_imbalance.get(router_id, 0.0))

        # Change penalty from original
        rx_change = rel_diff(rx_orig, rx_repaired)
        tx_change = rel_diff(tx_orig, tx_repaired)
        rx_change_factor = max(0.2, 1.0 - 0.5 * min(1.0, rx_change))
        tx_change_factor = max(0.2, 1.0 - 0.5 * min(1.0, tx_change))

        # Cap/scale/reconciliation penalties
        cap_pen_rx = 1.0 - min(0.3, penalties[iface]['rx_cap'] + 0.2 * penalties[iface]['rx_pair_adj'] + 0.2 * penalties[iface]['rx_scale'])
        cap_pen_tx = 1.0 - min(0.3, penalties[iface]['tx_cap'] + 0.2 * penalties[iface]['tx_pair_adj'] + 0.2 * penalties[iface]['tx_scale'])

        rx_confidence = max(0.0, min(1.0, rx_link_conf * r_factor * rx_change_factor * cap_pen_rx))
        tx_confidence = max(0.0, min(1.0, tx_link_conf * r_factor * tx_change_factor * cap_pen_tx))

        # Status confidence: keep status unchanged but calibrate
        status_confidence = 1.0
        if peer_id and peer_id in telemetry:
            peer_status_raw = norm_status(telemetry[peer_id].get('interface_status', 'unknown'))
            if status != peer_status_raw:
                status_confidence = min(status_confidence, 0.5)
        if status == 'down' and (rx_orig > 0.0 or tx_orig > 0.0):
            status_confidence = min(status_confidence, 0.6)

        # Build output
        out: Dict[str, Tuple] = {}
        out['rx_rate'] = (rx_orig, rx_repaired, rx_confidence)
        out['tx_rate'] = (tx_orig, tx_repaired, tx_confidence)
        out['interface_status'] = (status_raw, status_raw, status_confidence)

        # Copy metadata unchanged
        out['connected_to'] = data.get('connected_to')
        out['local_router'] = data.get('local_router')
        out['remote_router'] = data.get('remote_router')

        result[iface] = out

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