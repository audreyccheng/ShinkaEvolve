# EVOLVE-BLOCK-START
"""
Network telemetry repair algorithm that detects and corrects inconsistencies
in network interface telemetry data using topology relationships.

This version implements "tilted_pair_harmony":
- Residual-tilted, direction-aware consensus weights in pair repair
- Gentle multiplicative pre-harmonization (ratio bias prescaling) before averaging
- Asymmetric partial averaging that moves the louder side more
- Guarded micro-adjustments on dominating unpaired interfaces with benefit checks
- Confidence from symmetry fit, residual-severity–adaptive router penalties, and correction magnitude
- Magnitude-aware confidence floors and conservative quiet-side damping at low rate
"""
from typing import Dict, Any, Tuple, List
from math import sqrt


def repair_network_telemetry(telemetry: Dict[str, Dict[str, Any]],
                             topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:
    """
    Repair network interface telemetry by detecting and correcting inconsistencies.

    Invariants:
    - Link symmetry: my_tx ≈ peer_rx, my_rx ≈ peer_tx
    - Flow conservation at routers: sum(tx) ≈ sum(rx)
    - Interface consistency: down on one side implies down on both sides with zero rates
    """

    # Tolerances and guards
    HARDENING_THRESHOLD = 0.02     # ~2% for normal rates
    LOW_RATE_CUTOFF = 10.0         # Mbps; use relaxed 5% when both sides are tiny
    LOW_RATE_THRESHOLD = 0.05      # 5% tolerance for low-rate flows
    ABS_GUARD = 0.5                # Mbps; absolute guard to avoid over-correcting tiny flows
    QUIET_EPS = 0.1                # Mbps; traffic "silence" threshold
    MICRO_TRIGGER = 0.03           # Router residual trigger for micro-adjustments
    EXT_LOW = 1.0                  # Mbps; skip prescaling below this to avoid amplifying noise

    def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
        return max(lo, min(hi, x))

    def rel_diff(a: float, b: float) -> float:
        return abs(a - b) / max(1.0, a, b)

    def dir_threshold(a: float, b: float) -> float:
        return LOW_RATE_THRESHOLD if max(a, b) < LOW_RATE_CUTOFF else HARDENING_THRESHOLD

    # Build peer mapping (validated)
    peers: Dict[str, str] = {}
    for if_id, data in telemetry.items():
        p = data.get('connected_to')
        peers[if_id] = p if p in telemetry else None

    # Initialize working and original values
    vals: Dict[str, Dict[str, float]] = {}
    orig: Dict[str, Dict[str, float]] = {}
    status_orig: Dict[str, str] = {}
    for if_id, d in telemetry.items():
        tx = float(d.get('tx_rate', 0.0))
        rx = float(d.get('rx_rate', 0.0))
        vals[if_id] = {'tx': tx, 'rx': rx}
        orig[if_id] = {'tx': tx, 'rx': rx}
        status_orig[if_id] = d.get('interface_status', 'unknown')

    # Build router->interfaces mapping (use topology; augment with telemetry hints)
    router_ifaces: Dict[str, List[str]] = {r: list(if_list) for r, if_list in topology.items()}
    for if_id, d in telemetry.items():
        lr = d.get('local_router')
        rr = d.get('remote_router')
        if lr:
            router_ifaces.setdefault(lr, [])
            if if_id not in router_ifaces[lr]:
                router_ifaces[lr].append(if_id)
        if rr and rr not in router_ifaces:
            router_ifaces[rr] = []

    # Helper to compute per-router sums
    def compute_router_sums(current_vals: Dict[str, Dict[str, float]]) -> Dict[str, Tuple[float, float]]:
        sums: Dict[str, Tuple[float, float]] = {}
        for r, if_list in router_ifaces.items():
            s_tx = 0.0
            s_rx = 0.0
            for iid in if_list:
                if iid in current_vals:
                    s_tx += float(current_vals[iid]['tx'])
                    s_rx += float(current_vals[iid]['rx'])
            sums[r] = (s_tx, s_rx)
        return sums

    # Residual-tilting precomputation: signed residuals on original telemetry
    orig_sums = compute_router_sums(orig)
    resid_signed: Dict[str, float] = {}
    for r, (s_tx, s_rx) in orig_sums.items():
        denom_r = max(1.0, s_tx, s_rx)
        resid_signed[r] = (s_tx - s_rx) / denom_r  # signed in [-1,1]

    # Stage 1: Directional pair consensus with residual-tilted weights and prescaling
    visited_pairs = set()
    for a_id, _ in telemetry.items():
        b_id = peers.get(a_id)
        if not b_id:
            continue
        key = tuple(sorted((a_id, b_id)))
        if key in visited_pairs:
            continue
        visited_pairs.add(key)

        sa = status_orig.get(a_id, 'unknown')
        sb = status_orig.get(b_id, 'unknown')
        if sa == 'down' or sb == 'down':
            continue

        a_tx, a_rx = vals[a_id]['tx'], vals[a_id]['rx']
        b_tx, b_rx = vals[b_id]['tx'], vals[b_id]['rx']

        # Base activity weights
        act_a = max(a_tx, a_rx)
        act_b = max(b_tx, b_rx)
        denom_act = max(1e-9, act_a + act_b)
        w_a_base = act_a / denom_act
        w_b_base = act_b / denom_act

        # Residual-tilt helpers
        lr_a = telemetry[a_id].get('local_router')
        lr_b = telemetry[b_id].get('local_router')
        resid_a = resid_signed.get(lr_a, 0.0)
        resid_b = resid_signed.get(lr_b, 0.0)

        # Function to perform one directional reconciliation with prescaling and tilted weights
        def reconcile(local_id: str, peer_id: str, local_side: str, peer_side: str,
                      resid_local: float, w_local_base: float, w_peer_base: float):
            nonlocal vals
            x = vals[local_id][local_side]
            y = vals[peer_id][peer_side]

            tol = dir_threshold(x, y)
            delta = x - y
            adiff = abs(delta)
            diff = adiff / max(1.0, x, y)

            if diff <= tol or adiff <= (0.3 if max(x, y) < LOW_RATE_CUTOFF else ABS_GUARD):
                return  # within tolerance or below absolute guard

            # Gentle multiplicative pair-bias prescaling (skip ultra-low)
            if max(x, y) >= EXT_LOW:
                s = sqrt(max(1e-9, y) / max(1e-9, x))
                s = clamp(s, 0.90, 1.10)
            else:
                s = 1.0

            x_ps = x * s
            y_ps = y / s

            # Residual-tilted weights (only if tilt direction matches residual sign)
            sign_match = (delta > 0 and resid_local > 0) or (delta < 0 and resid_local < 0)
            gamma = min(0.08, 0.1 * abs(resid_local)) if sign_match else 0.0
            w_loc = clamp(w_local_base - gamma, 0.2, 0.8)
            w_peer = 1.0 - w_loc

            # Compute diff in prescaled domain to shape partial averaging
            adiff_ps = abs(x_ps - y_ps)
            diff_ps = adiff_ps / max(1.0, x_ps, y_ps)

            # Low-rate shaping params
            low_band = max(x, y) < LOW_RATE_CUTOFF
            full_mult = 1.6 if low_band else 2.0
            exp = 1.2 if low_band else 1.0

            consensus_ps = w_loc * x_ps + w_peer * y_ps
            if diff_ps <= full_mult * tol:
                # Asymmetric partial averaging: louder side moves more
                k_raw = (diff_ps - tol) / max(tol, 1e-9)
                k_base = max(0.0, min(1.0, k_raw ** exp))
                loud = max(x, y)
                quiet = min(x, y)
                r = (loud - quiet) / max(1.0, loud)
                if x >= y:
                    k_local = clamp(k_base * (1.0 + 0.5 * r))
                    k_peer = clamp(k_base * (1.0 - 0.5 * r))
                else:
                    k_peer = clamp(k_base * (1.0 + 0.5 * r))
                    k_local = clamp(k_base * (1.0 - 0.5 * r))
                # Map back from prescaled domain
                new_x = x * (1.0 - k_local) + (consensus_ps / s) * k_local
                new_y = y * (1.0 - k_peer) + (consensus_ps * s) * k_peer
            else:
                # Clear violation: converge to consensus (map back)
                new_x = consensus_ps / s
                new_y = consensus_ps * s

            vals[local_id][local_side] = max(0.0, new_x)
            vals[peer_id][peer_side] = max(0.0, new_y)

        # Apply reconciliation for both directions
        reconcile(a_id, b_id, 'tx', 'rx', resid_a, w_a_base, w_b_base)  # a_tx vs b_rx
        reconcile(b_id, a_id, 'tx', 'rx', resid_b, w_b_base, w_a_base)  # b_tx vs a_rx

    # Router-level micro-adjustments: dominated unpaired interfaces only, guarded commits
    router_sums = compute_router_sums(vals)

    # Track which interfaces were micro-adjusted (direction) to optionally dampen confidence slightly
    micro_adjusted_tx = set()
    micro_adjusted_rx = set()

    for r, if_list in router_ifaces.items():
        sum_tx, sum_rx = router_sums.get(r, (0.0, 0.0))
        denom = max(1.0, sum_tx, sum_rx)
        imbalance = sum_tx - sum_rx
        resid_frac = abs(imbalance) / denom
        if resid_frac < MICRO_TRIGGER:
            continue

        # Candidate: unpaired, up, non-trivial traffic
        candidates = []
        for iid in if_list:
            if iid not in vals:
                continue
            peer = peers.get(iid)
            is_unpaired = not peer or peer not in telemetry
            if not is_unpaired:
                continue
            if status_orig.get(iid, 'unknown') == 'down':
                continue
            txv = vals[iid]['tx']
            rxv = vals[iid]['rx']
            if max(txv, rxv) < LOW_RATE_CUTOFF:
                continue
            contrib = abs(txv - rxv)
            candidates.append((contrib, iid, txv, rxv))

        if not candidates:
            continue
        candidates.sort(reverse=True)
        top_contrib, top_if, txv, rxv = candidates[0]
        # Require dominance relative to router imbalance
        if top_contrib < 0.5 * abs(imbalance):
            continue

        alpha = min(0.02, 0.5 * resid_frac)
        if alpha <= 0.0:
            continue

        # Helper to compute router residual (using current vals)
        def router_resid_local(router_id: str) -> float:
            stx, srx = 0.0, 0.0
            for iid in router_ifaces.get(router_id, []):
                if iid in vals:
                    stx += vals[iid]['tx']
                    srx += vals[iid]['rx']
            return abs(stx - srx) / max(1.0, stx, srx)

        # Simulate both options and pick larger improvement under internal-skew guard
        pre_resid = router_resid_local(r)
        pre_internal = abs(txv - rxv) / max(1.0, max(txv, rxv))

        # Option TX adjust
        new_tx1, new_rx1 = txv, rxv
        if imbalance > 0.0:
            new_tx1 = txv * (1.0 - alpha)
        else:
            new_tx1 = txv * (1.0 + alpha)
        saved_tx, saved_rx = vals[top_if]['tx'], vals[top_if]['rx']
        vals[top_if]['tx'], vals[top_if]['rx'] = new_tx1, new_rx1
        resid1 = router_resid_local(r)
        post_internal1 = abs(new_tx1 - new_rx1) / max(1.0, max(new_tx1, new_rx1))
        vals[top_if]['tx'], vals[top_if]['rx'] = saved_tx, saved_rx

        # Option RX adjust
        new_tx2, new_rx2 = txv, rxv
        if imbalance > 0.0:
            new_rx2 = rxv * (1.0 + alpha)
        else:
            new_rx2 = rxv * (1.0 - alpha)
        vals[top_if]['tx'], vals[top_if]['rx'] = new_tx2, new_rx2
        resid2 = router_resid_local(r)
        post_internal2 = abs(new_tx2 - new_rx2) / max(1.0, max(new_tx2, new_rx2))
        vals[top_if]['tx'], vals[top_if]['rx'] = saved_tx, saved_rx

        choice = None
        best_resid = pre_resid
        if resid1 <= resid2 and post_internal1 <= pre_internal + 0.03:
            choice = 'tx'
            best_resid = resid1
            commit_tx, commit_rx = new_tx1, new_rx1
        elif resid2 < resid1 and post_internal2 <= pre_internal + 0.03:
            choice = 'rx'
            best_resid = resid2
            commit_tx, commit_rx = new_tx2, new_rx2

        if choice and best_resid <= (1.0 - 0.20) * pre_resid:
            vals[top_if]['tx'] = commit_tx
            vals[top_if]['rx'] = commit_rx
            if choice == 'tx':
                micro_adjusted_tx.add(top_if)
            else:
                micro_adjusted_rx.add(top_if)

            # Optional second mini-step if residual remains sizable
            resid_after_first = router_resid_local(r)
            if resid_after_first >= 0.04:
                alpha2 = min(0.01, 0.5 * resid_after_first)
                if alpha2 > 0.0:
                    if choice == 'tx':
                        tx2 = commit_tx * (1.0 - alpha2) if imbalance > 0.0 else commit_tx * (1.0 + alpha2)
                        saved = vals[top_if]['tx']
                        vals[top_if]['tx'] = tx2
                        resid_second = router_resid_local(r)
                        post_internal2b = abs(tx2 - commit_rx) / max(1.0, max(tx2, commit_rx))
                        if resid_second <= (1.0 - 0.20) * resid_after_first and post_internal2b <= pre_internal + 0.03:
                            vals[top_if]['tx'] = tx2
                        else:
                            vals[top_if]['tx'] = saved
                    else:
                        rx2 = commit_rx * (1.0 + alpha2) if imbalance > 0.0 else commit_rx * (1.0 - alpha2)
                        saved = vals[top_if]['rx']
                        vals[top_if]['rx'] = rx2
                        resid_second = router_resid_local(r)
                        post_internal2b = abs(commit_tx - rx2) / max(1.0, max(commit_tx, rx2))
                        if resid_second <= (1.0 - 0.20) * resid_after_first and post_internal2b <= pre_internal + 0.03:
                            vals[top_if]['rx'] = rx2
                        else:
                            vals[top_if]['rx'] = saved

    # Final per-router residuals after all adjustments
    final_sums = compute_router_sums(vals)
    router_resid: Dict[str, float] = {}
    router_sum_tx: Dict[str, float] = {}
    router_sum_rx: Dict[str, float] = {}
    for r, (s_tx, s_rx) in final_sums.items():
        denom_r = max(1.0, s_tx, s_rx)
        router_resid[r] = abs(s_tx - s_rx) / denom_r
        router_sum_tx[r] = s_tx
        router_sum_rx[r] = s_rx

    # Assemble output with calibrated confidences
    result: Dict[str, Dict[str, Tuple]] = {}
    for if_id, d in telemetry.items():
        lr = d.get('local_router')
        rr = d.get('remote_router')
        peer_id = peers.get(if_id)
        peer_exists = peer_id in vals if peer_id else False

        o_tx = orig[if_id]['tx']
        o_rx = orig[if_id]['rx']
        r_tx = vals[if_id]['tx']
        r_rx = vals[if_id]['rx']

        st = status_orig.get(if_id, 'unknown')
        pst = status_orig.get(peer_id, 'unknown') if peer_exists else None

        # Enforce interface consistency: if either side is down, set both down and zero rates
        force_down = (st == 'down') or (peer_exists and pst == 'down')
        if force_down:
            repaired_status = 'down'
            both_report_down = (st == 'down' and (peer_exists and pst == 'down'))
            status_conf = 0.95 if both_report_down else 0.7
            tx_conf = status_conf
            rx_conf = status_conf
            r_tx = 0.0
            r_rx = 0.0
        else:
            repaired_status = st

            # Router penalties with residual-severity–adaptive tilt and shares
            resid_local = router_resid.get(lr, 0.0)
            resid_remote = router_resid.get(rr, 0.0)
            severity = max(resid_local, resid_remote)
            amp = 0.1 if severity < 0.03 else (0.2 if severity < 0.12 else 0.3)

            sum_tx_local = router_sum_tx.get(lr, 0.0)
            sum_rx_local = router_sum_rx.get(lr, 0.0)
            tx_share = r_tx / max(1.0, sum_tx_local)
            rx_share = r_rx / max(1.0, sum_rx_local)

            pen_tx = clamp(1.0 - ((0.6 + amp * tx_share) * resid_local + (0.4 - amp * tx_share) * resid_remote), 0.5, 1.0)
            pen_rx = clamp(1.0 - ((0.6 + amp * rx_share) * resid_local + (0.4 - amp * rx_share) * resid_remote), 0.5, 1.0)

            # Symmetry fit after repair
            if peer_exists:
                pr_tx = vals[peer_id]['tx']
                pr_rx = vals[peer_id]['rx']
                diff_tx = rel_diff(r_tx, pr_rx)  # my_tx vs their_rx
                diff_rx = rel_diff(r_rx, pr_tx)  # my_rx vs their_tx
                c_sym_tx = clamp(1.0 - diff_tx)
                c_sym_rx = clamp(1.0 - diff_rx)

                # Floors (magnitude-aware); ultra-agreement enhancement
                floor_tx = 0.0
                floor_rx = 0.0
                th_tx = dir_threshold(r_tx, pr_rx)
                th_rx = dir_threshold(r_rx, pr_tx)
                ultra = (diff_tx <= 0.003 and diff_rx <= 0.003 and max(resid_local, resid_remote) <= 0.02)
                if ultra and max(r_tx, pr_rx, r_rx, pr_tx) >= 10.0:
                    floor_tx = max(floor_tx, 0.995)
                    floor_rx = max(floor_rx, 0.995)
                else:
                    if max(r_tx, pr_rx) >= 10.0 and diff_tx <= 0.005:
                        floor_tx = max(floor_tx, 0.99)
                    elif diff_tx <= th_tx:
                        floor_tx = max(floor_tx, 0.98 if max(r_tx, pr_rx) >= 10.0 else 0.97)
                    if max(r_rx, pr_tx) >= 10.0 and diff_rx <= 0.005:
                        floor_rx = max(floor_rx, 0.99)
                    elif diff_rx <= th_rx:
                        floor_rx = max(floor_rx, 0.98 if max(r_rx, pr_tx) >= 10.0 else 0.97)
            else:
                pr_tx = None
                pr_rx = None
                c_sym_tx = 0.9
                c_sym_rx = 0.9
                floor_tx = 0.0
                floor_rx = 0.0

            # Correction magnitude component (bigger changes => lower confidence)
            m_tx = abs(r_tx - o_tx) / max(1.0, r_tx, o_tx)
            m_rx = abs(r_rx - o_rx) / max(1.0, r_rx, o_rx)
            c_delta_tx = clamp(1.0 - min(1.0, 1.5 * m_tx))
            c_delta_rx = clamp(1.0 - min(1.0, 1.5 * m_rx))

            # Compose confidence from three axes
            tx_conf = clamp(0.45 * pen_tx + 0.35 * c_sym_tx + 0.20 * c_delta_tx)
            rx_conf = clamp(0.45 * pen_rx + 0.35 * c_sym_rx + 0.20 * c_delta_rx)

            # Apply magnitude-aware floors
            tx_conf = max(tx_conf, floor_tx)
            rx_conf = max(rx_conf, floor_rx)

            # Asymmetric traffic-evidence shaping: penalize the silent side only
            if peer_exists:
                if r_tx <= QUIET_EPS and pr_rx is not None and pr_rx > QUIET_EPS:
                    # Conservative damping when tiny flows; stronger otherwise
                    factor = 0.92 if (max(r_tx, pr_rx) < LOW_RATE_CUTOFF and pr_rx <= 2.0) else 0.88
                    tx_conf = clamp(tx_conf * factor)
                if r_rx <= QUIET_EPS and pr_tx is not None and pr_tx > QUIET_EPS:
                    factor = 0.92 if (max(r_rx, pr_tx) < LOW_RATE_CUTOFF and pr_tx <= 2.0) else 0.88
                    rx_conf = clamp(rx_conf * factor)

            # Unpaired interfaces: cap confidences modestly to reflect lack of redundancy
            if not peer_exists:
                tx_conf = min(tx_conf, 0.92)
                rx_conf = min(rx_conf, 0.92)

            # Mild extra coupling if micro-adjusted (already reflected via c_delta; keep gentle)
            if if_id in micro_adjusted_tx:
                rx_conf = clamp(rx_conf * 0.98)
            if if_id in micro_adjusted_rx:
                tx_conf = clamp(tx_conf * 0.98)

            # Status confidence shaping
            status_conf = 0.95
            if peer_exists and repaired_status != pst and pst != 'down':
                status_conf = min(status_conf, 0.6)
            # Boost for strong bilateral agreement
            if peer_exists:
                dtx = rel_diff(r_tx, pr_rx)
                drx = rel_diff(r_rx, pr_tx)
                if (max(r_tx, pr_rx) >= 10.0 and dtx <= 0.005 and
                    max(r_rx, pr_tx) >= 10.0 and drx <= 0.005 and
                        max(resid_local, resid_remote) <= 0.02):
                    status_conf = max(status_conf, 0.995)
                elif dtx <= dir_threshold(r_tx, pr_rx) and drx <= dir_threshold(r_rx, pr_tx):
                    status_conf = max(status_conf, 0.97)
            # Mild alignment with per-direction confidences and router penalties
            status_scale = 0.85 + 0.15 * min(pen_tx, pen_rx)
            status_conf = clamp(status_conf * status_scale)
            status_conf = clamp(status_conf * (0.85 + 0.15 * min(tx_conf, rx_conf)))

        # Assemble output
        out: Dict[str, Tuple] = {}
        out['rx_rate'] = (o_rx, r_rx, clamp(rx_conf))
        out['tx_rate'] = (o_tx, r_tx, clamp(tx_conf))
        out['interface_status'] = (status_orig[if_id], repaired_status, clamp(status_conf))
        out['connected_to'] = d.get('connected_to')
        out['local_router'] = d.get('local_router')
        out['remote_router'] = d.get('remote_router')
        result[if_id] = out

    # Safety: ensure zero rates if repaired status is down
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