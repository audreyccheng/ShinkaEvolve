# EVOLVE-BLOCK-START
"""
Network telemetry repair algorithm that detects and corrects inconsistencies
in network interface telemetry data using topology relationships.

This version implements "consensus_router_balancer":
- Stage 1: Directional consensus hardening with magnitude-aware gating and partial averaging
- Stage 2: Router-balanced reconciliation that distributes small mirrored adjustments
- Final confidence computed from three axes: symmetry fit, router residuals (direction-aware),
  and correction magnitude; plus magnitude-aware floors and asymmetric traffic evidence shaping.
"""
from typing import Dict, Any, Tuple, List


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
    ROUTER_RESID_TRIGGER = 0.03    # Trigger router-balanced reconciliation at >=3% mismatch
    ROUTER_BUDGET_CAP = 0.015      # Cap router-wide adjustment budget to 1.5% of router traffic
    SMALL_RATE_CUTOFF = 1.0        # Mbps; avoid router adjustments on extremely small flows
    IMPROVE_REQ = 0.08             # Require ≥8% residual improvement to commit router adjustments

    def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
        return max(lo, min(hi, x))

    # Useful helpers
    def rel_diff(a: float, b: float) -> float:
        return abs(a - b) / max(1.0, a, b)

    def dir_threshold(a: float, b: float) -> float:
        return LOW_RATE_THRESHOLD if max(a, b) < LOW_RATE_CUTOFF else HARDENING_THRESHOLD

    # Build peer mapping (validated)
    peers: Dict[str, str] = {}
    for if_id, data in telemetry.items():
        p = data.get('connected_to')
        peers[if_id] = p if p in telemetry else None

    # Initialize working values (start from originals)
    vals: Dict[str, Dict[str, float]] = {}
    orig: Dict[str, Dict[str, float]] = {}
    status_orig: Dict[str, str] = {}
    for if_id, d in telemetry.items():
        tx = float(d.get('tx_rate', 0.0))
        rx = float(d.get('rx_rate', 0.0))
        vals[if_id] = {'tx': tx, 'rx': rx}
        orig[if_id] = {'tx': tx, 'rx': rx}
        status_orig[if_id] = d.get('interface_status', 'unknown')

    # Stage 0: Pair bias scaling (multiplicative harmonization) before consensus averaging.
    # Detect when one endpoint's counters are consistently scaled vs its peer in both directions.
    # Apply a partial multiplicative correction to the lower-activity side to preserve calibration.
    visited_pairs_bias = set()
    for a_id, a_d in telemetry.items():
        b_id = peers.get(a_id)
        if not b_id:
            continue
        key = tuple(sorted((a_id, b_id)))
        if key in visited_pairs_bias:
            continue
        visited_pairs_bias.add(key)

        sa = status_orig.get(a_id, 'unknown')
        sb = status_orig.get(b_id, 'unknown')
        if sa == 'down' or sb == 'down':
            continue

        a_tx, a_rx = vals[a_id]['tx'], vals[a_id]['rx']
        b_tx, b_rx = vals[b_id]['tx'], vals[b_id]['rx']

        # Require active traffic in both directions to infer multiplicative bias
        if (a_tx <= QUIET_EPS or b_rx <= QUIET_EPS or a_rx <= QUIET_EPS or b_tx <= QUIET_EPS):
            continue

        # Ratios that should both approximate the same bias factor if one endpoint is scaled
        r1 = b_rx / max(a_tx, 1e-9)  # their_rx / my_tx
        r2 = b_tx / max(a_rx, 1e-9)  # their_tx / my_rx
        # Geometric mean ratio as consensus scale
        s = (r1 * r2) ** 0.5

        # Consistency check between ratios and magnitude-aware gating
        ratio_consistent = (abs(r1 - r2) / max(1.0, r1, r2)) <= 0.02
        hi = max(a_tx, a_rx, b_tx, b_rx)
        tol = LOW_RATE_THRESHOLD if hi < LOW_RATE_CUTOFF else HARDENING_THRESHOLD
        abs_ab = abs(a_tx - b_rx)
        abs_ba = abs(a_rx - b_tx)

        if (not ratio_consistent) or (abs(s - 1.0) <= tol) or (abs_ab <= ABS_GUARD and abs_ba <= ABS_GUARD):
            continue

        # Adjust the lower-activity side toward the bias-corrected scale
        act_a = max(a_tx, a_rx)
        act_b = max(b_tx, b_rx)
        if act_a < act_b:
            target = a_id
            scale = s
        else:
            target = b_id
            scale = 1.0 / max(1e-9, s)

        # Partial application near threshold to avoid overcorrection
        k = (abs(s - 1.0) - tol) / max(tol, 1e-9)
        k = clamp(k, 0.0, 1.0)
        factor = 1.0 + k * (scale - 1.0)

        vals[target]['tx'] = max(0.0, vals[target]['tx'] * factor)
        vals[target]['rx'] = max(0.0, vals[target]['rx'] * factor)

    # Stage 1: Directional link consensus with magnitude-aware gating and partial averaging
    visited_pairs = set()
    for a_id, a_d in telemetry.items():
        b_id = peers.get(a_id)
        if not b_id:
            continue
        key = tuple(sorted((a_id, b_id)))
        if key in visited_pairs:
            continue
        visited_pairs.add(key)

        sa = status_orig.get(a_id, 'unknown')
        sb = status_orig.get(b_id, 'unknown')

        # Skip consensus if either side is explicitly down; will enforce later
        if sa == 'down' or sb == 'down':
            continue

        a_tx, a_rx = vals[a_id]['tx'], vals[a_id]['rx']
        b_tx, b_rx = vals[b_id]['tx'], vals[b_id]['rx']

        # Direction a->b: a_tx vs b_rx
        abs_ab = abs(a_tx - b_rx)
        tol_ab = dir_threshold(a_tx, b_rx)
        diff_ab = abs_ab / max(1.0, a_tx, b_rx)
        # Low-rate–aware absolute guard and convergence shaping
        low_band_ab = max(a_tx, b_rx) < LOW_RATE_CUTOFF
        abs_guard_ab = 0.3 if low_band_ab else ABS_GUARD
        full_mult_ab = 1.6 if low_band_ab else 2.0
        exp_ab = 1.2 if low_band_ab else 1.0

        if diff_ab > tol_ab and abs_ab > abs_guard_ab:
            # Activity-weighted consensus (bias to stronger magnitude)
            act_a = max(a_tx, a_rx)
            act_b = max(b_tx, b_rx)
            denom = max(1e-9, act_a + act_b)
            w_a = act_a / denom
            w_b = act_b / denom
            consensus_ab = w_a * a_tx + w_b * b_rx
            if diff_ab <= full_mult_ab * tol_ab:
                # Asymmetric partial averaging: louder side moves more toward consensus
                k_raw = (diff_ab - tol_ab) / max(tol_ab, 1e-9)
                k_base = max(0.0, min(1.0, k_raw ** exp_ab))
                s = max(1e-9, a_tx + b_rx)
                a_share = a_tx / s
                b_share = b_rx / s
                if a_tx >= b_rx:
                    k_a = min(1.0, k_base * (1.0 + 0.5 * a_share))
                    k_b = min(1.0, k_base * (1.0 - 0.5 * b_share))
                else:
                    k_b = min(1.0, k_base * (1.0 + 0.5 * b_share))
                    k_a = min(1.0, k_base * (1.0 - 0.5 * a_share))
                vals[a_id]['tx'] = a_tx * (1.0 - k_a) + consensus_ab * k_a
                vals[b_id]['rx'] = b_rx * (1.0 - k_b) + consensus_ab * k_b
            else:
                vals[a_id]['tx'] = consensus_ab
                vals[b_id]['rx'] = consensus_ab
        # else within tolerance -> no change (confidence floors applied later)

        # Direction b->a: b_tx vs a_rx
        # Refresh values for this direction to reflect any prior adjustments
        b_tx = vals[b_id]['tx']
        a_rx = vals[a_id]['rx']
        abs_ba = abs(b_tx - a_rx)
        tol_ba = dir_threshold(b_tx, a_rx)
        diff_ba = abs_ba / max(1.0, b_tx, a_rx)
        low_band_ba = max(b_tx, a_rx) < LOW_RATE_CUTOFF
        abs_guard_ba = 0.3 if low_band_ba else ABS_GUARD
        full_mult_ba = 1.6 if low_band_ba else 2.0
        exp_ba = 1.2 if low_band_ba else 1.0

        if diff_ba > tol_ba and abs_ba > abs_guard_ba:
            act_a = max(a_tx, a_rx)
            act_b = max(b_tx, b_rx)
            denom = max(1e-9, act_a + act_b)
            w_a = act_a / denom
            w_b = act_b / denom
            consensus_ba = w_b * b_tx + w_a * a_rx
            if diff_ba <= full_mult_ba * tol_ba:
                k_raw = (diff_ba - tol_ba) / max(tol_ba, 1e-9)
                k_base = max(0.0, min(1.0, k_raw ** exp_ba))
                s2 = max(1e-9, b_tx + a_rx)
                b_share = b_tx / s2
                a_share = a_rx / s2
                if b_tx >= a_rx:
                    k_b = min(1.0, k_base * (1.0 + 0.5 * b_share))
                    k_a = min(1.0, k_base * (1.0 - 0.5 * a_share))
                else:
                    k_a = min(1.0, k_base * (1.0 + 0.5 * a_share))
                    k_b = min(1.0, k_base * (1.0 - 0.5 * b_share))
                vals[b_id]['tx'] = b_tx * (1.0 - k_b) + consensus_ba * k_b
                vals[a_id]['rx'] = a_rx * (1.0 - k_a) + consensus_ba * k_a
            else:
                vals[b_id]['tx'] = consensus_ba
                vals[a_id]['rx'] = consensus_ba

    # Build router->interfaces mapping (prefer topology, augment with telemetry hints)
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

    # Helper to compute per-router sums from current vals (zeros for down enforced later)
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

    # Stage 2: Router-balanced reconciliation with mirrored adjustments on peers
    router_sums = compute_router_sums(vals)

    # Helper to compute residual for a single router on current vals
    def compute_router_resid_local(router_id: str) -> float:
        s_tx, s_rx = 0.0, 0.0
        for iid in router_ifaces.get(router_id, []):
            if iid in vals:
                s_tx += float(vals[iid]['tx'])
                s_rx += float(vals[iid]['rx'])
        return abs(s_tx - s_rx) / max(1.0, s_tx, s_rx)

    for r, if_list in router_ifaces.items():
        sum_tx, sum_rx = router_sums.get(r, (0.0, 0.0))
        denom_r = max(1.0, sum_tx, sum_rx)
        delta = sum_tx - sum_rx
        resid = abs(delta) / denom_r
        if resid < ROUTER_RESID_TRIGGER or not if_list:
            continue

        # Determine whether to reduce the dominant side
        reduce_tx = delta > 0.0  # too much TX
        reduce_rx = delta < 0.0  # too much RX

        # Collect candidates and total mass for proportional distribution
        if reduce_tx:
            # Outgoing directions at router r => this router's tx counters
            outs = [iid for iid in if_list
                    if status_orig.get(iid, 'unknown') != 'down' and peers.get(iid) and vals[iid]['tx'] >= SMALL_RATE_CUTOFF]
            total_mass = sum(vals[iid]['tx'] for iid in outs) or 0.0
            if total_mass <= 0.0:
                continue
            budget = min(abs(delta), ROUTER_BUDGET_CAP * denom_r)
            # Tentatively apply and track changes
            changes: List[Tuple[str, str, float]] = []
            for iid in outs:
                share = vals[iid]['tx'] / total_mass if total_mass > 0 else 0.0
                d_i = budget * share
                if d_i <= 0.0:
                    continue
                # Apply to local tx (tentative)
                old_tx = vals[iid]['tx']
                new_tx = max(0.0, old_tx - d_i)
                vals[iid]['tx'] = new_tx
                changes.append(('tx', iid, old_tx))
                # Mirror to peer's rx if peer exists and not explicitly down
                pid = peers.get(iid)
                if pid and status_orig.get(pid, 'unknown') != 'down':
                    old_prx = vals[pid]['rx']
                    vals[pid]['rx'] = max(0.0, old_prx - d_i)
                    changes.append(('rx', pid, old_prx))
            # Check improvement; revert if insufficient
            resid_after = compute_router_resid_local(r)
            if resid_after > (1.0 - IMPROVE_REQ) * resid:
                for kind, iid, old_val in changes:
                    vals[iid][kind] = old_val
        elif reduce_rx:
            # Incoming directions at router r => this router's rx counters
            ins = [iid for iid in if_list
                   if status_orig.get(iid, 'unknown') != 'down' and peers.get(iid) and vals[iid]['rx'] >= SMALL_RATE_CUTOFF]
            total_mass = sum(vals[iid]['rx'] for iid in ins) or 0.0
            if total_mass <= 0.0:
                continue
            budget = min(abs(delta), ROUTER_BUDGET_CAP * denom_r)
            changes: List[Tuple[str, str, float]] = []
            for iid in ins:
                share = vals[iid]['rx'] / total_mass if total_mass > 0 else 0.0
                d_i = budget * share
                if d_i <= 0.0:
                    continue
                # Apply to local rx (tentative)
                old_rx = vals[iid]['rx']
                new_rx = max(0.0, old_rx - d_i)
                vals[iid]['rx'] = new_rx
                changes.append(('rx', iid, old_rx))
                # Mirror to peer's tx if peer exists and not explicitly down
                pid = peers.get(iid)
                if pid and status_orig.get(pid, 'unknown') != 'down':
                    old_ptx = vals[pid]['tx']
                    vals[pid]['tx'] = max(0.0, old_ptx - d_i)
                    changes.append(('tx', pid, old_ptx))
            resid_after = compute_router_resid_local(r)
            if resid_after > (1.0 - IMPROVE_REQ) * resid:
                for kind, iid, old_val in changes:
                    vals[iid][kind] = old_val

    # Recompute per-router residuals after reconciliation
    router_sums2 = compute_router_sums(vals)
    router_resid: Dict[str, float] = {}
    for r, (s_tx, s_rx) in router_sums2.items():
        denom_r = max(1.0, s_tx, s_rx)
        router_resid[r] = abs(s_tx - s_rx) / denom_r

    # Final assembly with calibrated confidences
    result: Dict[str, Dict[str, Tuple]] = {}

    for if_id, d in telemetry.items():
        lr = d.get('local_router')
        rr = d.get('remote_router')
        peer_id = peers.get(if_id)
        peer_exists = peer_id is not None

        orig_tx = orig[if_id]['tx']
        orig_rx = orig[if_id]['rx']
        rep_tx = vals[if_id]['tx']
        rep_rx = vals[if_id]['rx']

        status = status_orig.get(if_id, 'unknown')
        peer_status = status_orig.get(peer_id, 'unknown') if peer_exists else None

        # Enforce interface consistency: if either side is down, set both down and zero rates
        force_down = (status == 'down') or (peer_exists and peer_status == 'down')
        if force_down:
            repaired_status = 'down'
            both_report_down = (status == 'down' and (peer_exists and peer_status == 'down'))
            status_conf = 0.95 if both_report_down else 0.7
            rep_tx = 0.0
            rep_rx = 0.0
            # Rate confidences mirror status confidence under forced-down condition
            tx_conf = status_conf
            rx_conf = status_conf
        else:
            repaired_status = status

            # Direction-aware router penalty
            resid_local = router_resid.get(lr, 0.0)
            resid_remote = router_resid.get(rr, 0.0)
            pen_tx = clamp(1.0 - (0.6 * resid_local + 0.4 * resid_remote), 0.0, 1.0)
            pen_rx = clamp(1.0 - (0.6 * resid_remote + 0.4 * resid_local), 0.0, 1.0)

            # Link symmetry fit after repair
            if peer_exists:
                peer_rep_tx = vals[peer_id]['tx']
                peer_rep_rx = vals[peer_id]['rx']
                diff_tx = rel_diff(rep_tx, peer_rep_rx)  # my_tx vs their_rx
                diff_rx = rel_diff(rep_rx, peer_rep_tx)  # my_rx vs their_tx
                c_sym_tx = clamp(1.0 - diff_tx)
                c_sym_rx = clamp(1.0 - diff_rx)
                # Magnitude-aware floors for very strong agreement
                floor_tx = 0.0
                floor_rx = 0.0
                th_tx = dir_threshold(rep_tx, peer_rep_rx)
                th_rx = dir_threshold(rep_rx, peer_rep_tx)
                if max(rep_tx, peer_rep_rx) >= 10.0 and diff_tx <= 0.005:
                    floor_tx = 0.99
                elif diff_tx <= th_tx:
                    floor_tx = 0.98 if max(rep_tx, peer_rep_rx) >= 10.0 else 0.97
                if max(rep_rx, peer_rep_tx) >= 10.0 and diff_rx <= 0.005:
                    floor_rx = 0.99
                elif diff_rx <= th_rx:
                    floor_rx = 0.98 if max(rep_rx, peer_rep_tx) >= 10.0 else 0.97
            else:
                peer_rep_tx = None
                peer_rep_rx = None
                c_sym_tx = 0.9
                c_sym_rx = 0.9
                floor_tx = 0.0
                floor_rx = 0.0

            # Correction magnitude factor (bigger changes => lower confidence)
            m_tx = abs(rep_tx - orig_tx) / max(1.0, rep_tx, orig_tx)
            m_rx = abs(rep_rx - orig_rx) / max(1.0, rep_rx, orig_rx)
            c_delta_tx = clamp(1.0 - min(1.0, 1.5 * m_tx))
            c_delta_rx = clamp(1.0 - min(1.0, 1.5 * m_rx))

            # Compose confidence from three axes + floors
            tx_conf = clamp(0.45 * pen_tx + 0.35 * c_sym_tx + 0.20 * c_delta_tx)
            rx_conf = clamp(0.45 * pen_rx + 0.35 * c_sym_rx + 0.20 * c_delta_rx)
            tx_conf = max(tx_conf, floor_tx)
            rx_conf = max(rx_conf, floor_rx)

            # Asymmetric traffic-evidence shaping: penalize the silent side only
            if peer_exists:
                if rep_tx <= QUIET_EPS and peer_rep_rx is not None and peer_rep_rx > QUIET_EPS:
                    tx_conf = clamp(tx_conf * 0.88)
                if rep_rx <= QUIET_EPS and peer_rep_tx is not None and peer_rep_tx > QUIET_EPS:
                    rx_conf = clamp(rx_conf * 0.88)

            # Status confidence shaping
            status_conf = 0.95
            if peer_exists and repaired_status != peer_status and peer_status != 'down':
                status_conf = min(status_conf, 0.6)
            # Strong bilateral agreement boosts status confidence
            if peer_exists:
                if (max(rep_tx, peer_rep_rx) >= 10.0 and rel_diff(rep_tx, peer_rep_rx) <= 0.005 and
                        max(rep_rx, peer_rep_tx) >= 10.0 and rel_diff(rep_rx, peer_rep_tx) <= 0.005):
                    status_conf = max(status_conf, 0.99)
                elif (rel_diff(rep_tx, peer_rep_rx) <= dir_threshold(rep_tx, peer_rep_rx) and
                      rel_diff(rep_rx, peer_rep_tx) <= dir_threshold(rep_rx, peer_rep_tx)):
                    status_conf = max(status_conf, 0.97)
            # Mildly align status confidence with per-direction confidences
            status_conf = clamp(status_conf * (0.85 + 0.15 * min(tx_conf, rx_conf)))

        # Assemble output
        out: Dict[str, Tuple] = {}
        out['rx_rate'] = (orig_rx, rep_rx, clamp(rx_conf))
        out['tx_rate'] = (orig_tx, rep_tx, clamp(tx_conf))
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