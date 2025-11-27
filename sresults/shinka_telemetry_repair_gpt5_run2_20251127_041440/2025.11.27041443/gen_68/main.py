# EVOLVE-BLOCK-START
"""
Network telemetry repair via global alternating projections.

We iteratively project counters onto:
- Link symmetry constraints (directional equality across each connected pair)
- Router flow conservation (sum(tx) = sum(rx) at each router)

Hard constraints:
- If either endpoint reports 'down', both sides are set 'down' with zero rates.

Confidences are computed after convergence from:
- Direction-aware, share-aware router residual penalties
- Link symmetry fit after repair
- Magnitude of change (original vs repaired)
- With magnitude-aware floors and silent-side shaping
"""
from typing import Dict, Any, Tuple, List


def repair_network_telemetry(telemetry: Dict[str, Dict[str, Any]],
                             topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:
    # Tolerances and guards (Hodor-inspired)
    HARDENING_THRESHOLD = 0.02    # ~2% normal-rate tolerance
    LOW_RATE_CUTOFF = 10.0        # Mbps; low-rate handling
    LOW_RATE_THRESHOLD = 0.05     # 5% tolerance under low-rate
    ABS_GUARD_HI = 0.5            # Mbps; absolute guard for high-rate
    ABS_GUARD_LO = 0.3            # Mbps; absolute guard for low-rate
    QUIET_EPS = 0.1               # Mbps; silence threshold
    ROUTER_RESID_TRIGGER = 0.02   # router projection trigger (~2%)

    MAX_ITERS = 6                 # alternating projections iterations

    def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
        return max(lo, min(hi, x))

    def rel_diff(a: float, b: float) -> float:
        return abs(a - b) / max(1.0, a, b)

    def dir_threshold(a: float, b: float) -> float:
        return LOW_RATE_THRESHOLD if max(a, b) < LOW_RATE_CUTOFF else HARDENING_THRESHOLD

    # Build peer mapping and originals
    peers: Dict[str, str] = {}
    orig_tx: Dict[str, float] = {}
    orig_rx: Dict[str, float] = {}
    status_orig: Dict[str, str] = {}
    for if_id, d in telemetry.items():
        p = d.get('connected_to')
        peers[if_id] = p if p in telemetry else None
        orig_tx[if_id] = float(d.get('tx_rate', 0.0))
        orig_rx[if_id] = float(d.get('rx_rate', 0.0))
        status_orig[if_id] = d.get('interface_status', 'unknown')

    # Router->interfaces mapping (prefer topology, augment with telemetry)
    router_ifaces: Dict[str, List[str]] = {r: list(if_list) for r, if_list in topology.items()}
    for if_id, d in telemetry.items():
        lr = d.get('local_router')
        if lr:
            router_ifaces.setdefault(lr, [])
            if if_id not in router_ifaces[lr]:
                router_ifaces[lr].append(if_id)
        rr = d.get('remote_router')
        if rr and rr not in router_ifaces:
            router_ifaces[rr] = []

    # Determine forced-down set: if either endpoint reports down
    forced_down = set()
    for if_id, st in status_orig.items():
        peer = peers.get(if_id)
        peer_down = (status_orig.get(peer, 'unknown') == 'down') if peer else False
        if st == 'down' or peer_down:
            forced_down.add(if_id)
            if peer:
                forced_down.add(peer)

    # Working values (start from originals); force down ones to zero
    work_tx: Dict[str, float] = {}
    work_rx: Dict[str, float] = {}
    for if_id in telemetry.keys():
        if if_id in forced_down:
            work_tx[if_id] = 0.0
            work_rx[if_id] = 0.0
        else:
            work_tx[if_id] = orig_tx[if_id]
            work_rx[if_id] = orig_rx[if_id]

    # Helper: one pass of link projection (directional symmetry)
    def project_links():
        visited = set()
        for a_id, a_data in telemetry.items():
            b_id = peers.get(a_id)
            if not b_id:
                continue
            key = tuple(sorted((a_id, b_id)))
            if key in visited:
                continue
            visited.add(key)

            # Skip pairs that are forced down (already handled)
            if a_id in forced_down or b_id in forced_down:
                continue

            a_tx, a_rx = work_tx[a_id], work_rx[a_id]
            b_tx, b_rx = work_tx[b_id], work_rx[b_id]

            # Activity-based trust weights
            act_a = max(a_tx, a_rx)
            act_b = max(b_tx, b_rx)
            denom_act = max(1e-9, act_a + act_b)
            w_a = act_a / denom_act
            w_b = act_b / denom_act

            # a->b: a_tx vs b_rx
            abs_ab = abs(a_tx - b_rx)
            low_band_ab = max(a_tx, b_rx) < LOW_RATE_CUTOFF
            tol_ab = LOW_RATE_THRESHOLD if low_band_ab else HARDENING_THRESHOLD
            abs_guard_ab = ABS_GUARD_LO if low_band_ab else ABS_GUARD_HI
            full_mult_ab = 1.6 if low_band_ab else 2.0
            exp_ab = 1.2 if low_band_ab else 1.0
            diff_ab = abs_ab / max(1.0, a_tx, b_rx)

            if diff_ab > tol_ab and abs_ab > abs_guard_ab:
                consensus_ab = w_a * a_tx + w_b * b_rx
                if diff_ab <= full_mult_ab * tol_ab:
                    k_raw = (diff_ab - tol_ab) / max(tol_ab, 1e-9)
                    k = clamp(k_raw ** exp_ab, 0.0, 1.0)
                    work_tx[a_id] = a_tx * (1.0 - k) + consensus_ab * k
                    work_rx[b_id] = b_rx * (1.0 - k) + consensus_ab * k
                else:
                    work_tx[a_id] = consensus_ab
                    work_rx[b_id] = consensus_ab

            # b->a: b_tx vs a_rx
            a_tx, a_rx = work_tx[a_id], work_rx[a_id]  # refresh after potential change
            b_tx, b_rx = work_tx[b_id], work_rx[b_id]
            abs_ba = abs(b_tx - a_rx)
            low_band_ba = max(b_tx, a_rx) < LOW_RATE_CUTOFF
            tol_ba = LOW_RATE_THRESHOLD if low_band_ba else HARDENING_THRESHOLD
            abs_guard_ba = ABS_GUARD_LO if low_band_ba else ABS_GUARD_HI
            full_mult_ba = 1.6 if low_band_ba else 2.0
            exp_ba = 1.2 if low_band_ba else 1.0
            diff_ba = abs_ba / max(1.0, b_tx, a_rx)

            if diff_ba > tol_ba and abs_ba > abs_guard_ba:
                consensus_ba = w_b * b_tx + w_a * a_rx
                if diff_ba <= full_mult_ba * tol_ba:
                    k_raw = (diff_ba - tol_ba) / max(tol_ba, 1e-9)
                    k = clamp(k_raw ** exp_ba, 0.0, 1.0)
                    work_tx[b_id] = b_tx * (1.0 - k) + consensus_ba * k
                    work_rx[a_id] = a_rx * (1.0 - k) + consensus_ba * k
                else:
                    work_tx[b_id] = consensus_ba
                    work_rx[a_id] = consensus_ba

    # Helper: one pass of router projection (flow conservation)
    def project_routers(iter_idx: int):
        # Step scaling grows over iterations to be gentle early and decisive later
        step_scale = 0.6 + 0.4 * (iter_idx / max(1, MAX_ITERS - 1))
        for r, if_list in router_ifaces.items():
            # Consider only interfaces that exist and are not forced down
            members = [iid for iid in if_list if iid in telemetry and iid not in forced_down]
            if not members:
                continue
            sum_tx = sum(work_tx[iid] for iid in members)
            sum_rx = sum(work_rx[iid] for iid in members)
            delta = sum_tx - sum_rx
            denom = max(1.0, sum_tx, sum_rx)
            resid = abs(delta) / denom
            if resid < ROUTER_RESID_TRIGGER:
                continue

            alpha = 0.5 * abs(delta) * step_scale  # amount to transfer between sides this pass
            if delta > 0:
                # TX too high: subtract from tx and add to rx
                mass_tx = max(1e-9, sum_tx)
                mass_rx = max(1e-9, sum_rx)
                for iid in members:
                    # proportional shares; keep non-negative
                    dec = alpha * (work_tx[iid] / mass_tx)
                    inc = alpha * (work_rx[iid] / mass_rx)
                    work_tx[iid] = max(0.0, work_tx[iid] - dec)
                    work_rx[iid] = max(0.0, work_rx[iid] + inc)
            else:
                # RX too high: subtract from rx and add to tx
                mass_tx = max(1e-9, sum_tx)
                mass_rx = max(1e-9, sum_rx)
                for iid in members:
                    dec = alpha * (work_rx[iid] / mass_rx)
                    inc = alpha * (work_tx[iid] / mass_tx)
                    work_rx[iid] = max(0.0, work_rx[iid] - dec)
                    work_tx[iid] = max(0.0, work_tx[iid] + inc)

    # Run alternating projections
    for it in range(MAX_ITERS):
        project_links()
        project_routers(it)

    # Prepare final result with confidences
    # Compute router residuals on repaired values
    router_sum_tx: Dict[str, float] = {}
    router_sum_rx: Dict[str, float] = {}
    for r, if_list in router_ifaces.items():
        stx = srx = 0.0
        for iid in if_list:
            if iid in telemetry:
                stx += work_tx.get(iid, 0.0)
                srx += work_rx.get(iid, 0.0)
        router_sum_tx[r] = stx
        router_sum_rx[r] = srx
    router_resid: Dict[str, float] = {}
    for r in router_ifaces.keys():
        stx, srx = router_sum_tx.get(r, 0.0), router_sum_rx.get(r, 0.0)
        router_resid[r] = abs(stx - srx) / max(1.0, stx, srx)

    result: Dict[str, Dict[str, Tuple]] = {}

    for if_id, d in telemetry.items():
        lr = d.get('local_router')
        rr = d.get('remote_router')
        peer_id = peers.get(if_id)

        # Repaired status with hard consistency
        my_status = status_orig.get(if_id, 'unknown')
        peer_status = status_orig.get(peer_id, 'unknown') if peer_id else None
        force_down = (if_id in forced_down)

        rep_tx = 0.0 if force_down else work_tx[if_id]
        rep_rx = 0.0 if force_down else work_rx[if_id]
        org_tx = orig_tx[if_id]
        org_rx = orig_rx[if_id]

        # Confidence components
        # Direction-aware, share-aware router penalties
        resid_local = router_resid.get(lr, 0.0)
        resid_remote = router_resid.get(rr, 0.0)
        sum_tx_local = router_sum_tx.get(lr, 0.0)
        sum_rx_local = router_sum_rx.get(lr, 0.0)
        tx_share = rep_tx / max(1.0, sum_tx_local)
        rx_share = rep_rx / max(1.0, sum_rx_local)
        penalty_tx = clamp(1.0 - ((0.6 + 0.2 * tx_share) * resid_local + (0.4 - 0.2 * tx_share) * resid_remote), 0.5, 1.0)
        penalty_rx = clamp(1.0 - ((0.6 + 0.2 * rx_share) * resid_local + (0.4 - 0.2 * rx_share) * resid_remote), 0.5, 1.0)

        # Link symmetry fit after repair
        if peer_id in telemetry:
            peer_rep_tx = 0.0 if (peer_id in forced_down) else work_tx[peer_id]
            peer_rep_rx = 0.0 if (peer_id in forced_down) else work_rx[peer_id]
            diff_tx = rel_diff(rep_tx, peer_rep_rx)
            diff_rx = rel_diff(rep_rx, peer_rep_tx)
            c_sym_tx = clamp(1.0 - diff_tx)
            c_sym_rx = clamp(1.0 - diff_rx)
            # Magnitude-aware floors
            th_tx = dir_threshold(rep_tx, peer_rep_rx)
            th_rx = dir_threshold(rep_rx, peer_rep_tx)
            if max(rep_tx, peer_rep_rx) >= 10.0 and diff_tx <= 0.005:
                floor_tx = 0.99
            elif diff_tx <= th_tx:
                floor_tx = 0.98 if max(rep_tx, peer_rep_rx) >= 10.0 else 0.97
            else:
                floor_tx = 0.0
            if max(rep_rx, peer_rep_tx) >= 10.0 and diff_rx <= 0.005:
                floor_rx = 0.99
            elif diff_rx <= th_rx:
                floor_rx = 0.98 if max(rep_rx, peer_rep_tx) >= 10.0 else 0.97
            else:
                floor_rx = 0.0
        else:
            c_sym_tx = 0.9
            c_sym_rx = 0.9
            floor_tx = 0.0
            floor_rx = 0.0

        # Change magnitude
        m_tx = abs(rep_tx - org_tx) / max(1.0, rep_tx, org_tx)
        m_rx = abs(rep_rx - org_rx) / max(1.0, rep_rx, org_rx)
        c_delta_tx = clamp(1.0 - min(1.0, 1.5 * m_tx))
        c_delta_rx = clamp(1.0 - min(1.0, 1.5 * m_rx))

        # Compose confidences
        tx_conf = clamp(0.45 * penalty_tx + 0.35 * c_sym_tx + 0.20 * c_delta_tx)
        rx_conf = clamp(0.45 * penalty_rx + 0.35 * c_sym_rx + 0.20 * c_delta_rx)
        tx_conf = max(tx_conf, floor_tx)
        rx_conf = max(rx_conf, floor_rx)

        # Penalize silent side if peer shows traffic
        if peer_id in telemetry:
            peer_rep_tx = 0.0 if (peer_id in forced_down) else work_tx[peer_id]
            peer_rep_rx = 0.0 if (peer_id in forced_down) else work_rx[peer_id]
            if rep_tx <= QUIET_EPS and peer_rep_rx > QUIET_EPS:
                tx_conf = clamp(tx_conf * 0.88)
            if rep_rx <= QUIET_EPS and peer_rep_tx > QUIET_EPS:
                rx_conf = clamp(rx_conf * 0.88)

        # Unpaired interfaces (no redundancy): slightly lower baseline
        if peers.get(if_id) is None:
            tx_conf = min(tx_conf, 0.92)
            rx_conf = min(rx_conf, 0.92)

        # Status repair and confidence
        if force_down:
            repaired_status = 'down'
            both_report_down = (my_status == 'down' and ((peer_id in telemetry and status_orig.get(peer_id) == 'down') if peer_id else False))
            status_conf = 0.95 if both_report_down else 0.7
            # Scale by router penalties mildly
            status_scale = 0.85 + 0.15 * min(penalty_tx, penalty_rx)
            status_conf = clamp(status_conf * status_scale)
            tx_conf = clamp(tx_conf * (0.9 + 0.1 * penalty_tx))
            rx_conf = clamp(rx_conf * (0.9 + 0.1 * penalty_rx))
        else:
            repaired_status = my_status
            status_conf = 0.95
            # Status mismatch (neither side down) reduces confidence
            if peer_id in telemetry and status_orig.get(peer_id, 'unknown') != 'down' and my_status != status_orig.get(peer_id, 'unknown'):
                status_conf = min(status_conf, 0.6)
            # Align status confidence with penalties and per-direction confidences
            status_conf = clamp(status_conf * (0.85 + 0.15 * min(penalty_tx, penalty_rx)))
            status_conf = clamp(status_conf * (0.85 + 0.15 * min(tx_conf, rx_conf)))

        # Assemble output
        out: Dict[str, Tuple] = {}
        out['rx_rate'] = (orig_rx[if_id], rep_rx, rx_conf)
        out['tx_rate'] = (orig_tx[if_id], rep_tx, tx_conf)
        out['interface_status'] = (status_orig[if_id], repaired_status, status_conf)
        out['connected_to'] = d.get('connected_to')
        out['local_router'] = d.get('local_router')
        out['remote_router'] = d.get('remote_router')
        result[if_id] = out

    # Safety: ensure zeros for all down statuses
    for if_id, d in result.items():
        if d['interface_status'][1] == 'down':
            orx, _, rc = d['rx_rate']
            otx, _, tc = d['tx_rate']
            d['rx_rate'] = (orx, 0.0, rc)
            d['tx_rate'] = (otx, 0.0, tc)

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