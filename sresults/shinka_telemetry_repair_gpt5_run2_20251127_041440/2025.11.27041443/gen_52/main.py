# EVOLVE-BLOCK-START
"""
Network telemetry repair algorithm that detects and corrects inconsistencies
in network interface telemetry data using topology relationships.

This version implements "tilted_balance_pipeline":
- Staged pipeline: signal_collection -> link_hardening -> router_balancing -> micro_adjustment -> confidence_assembly
- Residual-tilted, trust-weighted directional consensus with refined asymmetric partial averaging
- Magnitude-aware gating with absolute guard and strong agreement floors
- Soft router-balanced reconciliation with mirrored peer updates and strict caps
- Improvement-checked micro-adjustments for dominating unpaired interfaces
- Tri-axis confidence with interface-share–aware, direction-coupled router penalties,
  symmetry fit after repair, and correction magnitude; plus asymmetric traffic-evidence shaping
"""
from typing import Dict, Any, Tuple, List
from math import sqrt


def repair_network_telemetry(telemetry: Dict[str, Dict[str, Any]],
                             topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:
    """
    Repair network interface telemetry by detecting and correcting inconsistencies.

    Invariants:
    - Link Symmetry (R3): my_tx ≈ peer_rx, my_rx ≈ peer_tx
    - Flow Conservation (R1): Sum(tx) ≈ Sum(rx) at each router
    - Interface Consistency: Down on one side => both down with zero rates
    """

    # Thresholds and guards
    HARDENING_THRESHOLD = 0.02       # τh ≈ 2% (normal rates)
    LOW_RATE_CUTOFF = 10.0           # Mbps (small flows use relaxed tolerance)
    LOW_RATE_THRESHOLD = 0.05        # 5% for low rates
    ABS_GUARD = 0.5                  # Mbps absolute guard for tiny discrepancies
    QUIET_EPS = 0.1                  # Mbps silence threshold

    # Residual-tilt parameters (Recommendation 1)
    TILT_GAMMA_MAX = 0.08            # max absolute tilt on local weight
    TILT_GAMMA_SCALE = 0.10          # scale by residual fraction (γ = min(0.08, 0.1 × resid_local))

    # Soft router balancing (new Stage 2A)
    ROUTER_RESID_TRIGGER_SOFT = 0.02  # start soft reconciliation at ≥2% mismatch
    ROUTER_BUDGET_CAP = 0.012         # cap router-wide soft budget to 1.2% of router traffic

    # Micro-adjuster parameters (Recommendation 2)
    ROUTER_RESID_TRIGGER = 0.03      # trigger micro-adjust when residual ≥ 3%
    DOMINANCE_SHARE = 0.60           # candidate must contribute ≥60% of same-direction mass
    MI_ALPHA_CAP = 0.02              # cap nudge at 2%
    IMPROVE_REQ = 0.10               # require ≥10% residual improvement to commit

    def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
        return max(lo, min(hi, x))

    def rel_diff(a: float, b: float) -> float:
        return abs(a - b) / max(1.0, a, b)

    def dir_threshold(a: float, b: float) -> float:
        return LOW_RATE_THRESHOLD if max(a, b) < LOW_RATE_CUTOFF else HARDENING_THRESHOLD

    # Stage 0: Signal Collection
    def build_context():
        # peers: interface -> connected peer if exists in telemetry, else None
        peers: Dict[str, str] = {}
        vals: Dict[str, Dict[str, float]] = {}
        orig: Dict[str, Dict[str, float]] = {}
        status_orig: Dict[str, str] = {}

        for if_id, d in telemetry.items():
            p = d.get('connected_to')
            peers[if_id] = p if p in telemetry else None
            tx = float(d.get('tx_rate', 0.0))
            rx = float(d.get('rx_rate', 0.0))
            vals[if_id] = {'tx': tx, 'rx': rx}
            orig[if_id] = {'tx': tx, 'rx': rx}
            status_orig[if_id] = d.get('interface_status', 'unknown')

        # router_ifaces mapping; prefer topology, augment with telemetry metadata
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

        return peers, vals, orig, status_orig, router_ifaces

    def compute_router_sums(current_vals: Dict[str, Dict[str, float]], router_ifaces: Dict[str, List[str]]):
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

    peers, vals, orig, status_orig, router_ifaces = build_context()

    # Compute original router residuals for tilt computation
    router_sums_orig = compute_router_sums(orig, router_ifaces)
    router_delta_orig: Dict[str, float] = {}
    router_resid_frac_orig: Dict[str, float] = {}
    for r, (s_tx, s_rx) in router_sums_orig.items():
        router_delta_orig[r] = s_tx - s_rx
        denom = max(1.0, s_tx, s_rx)
        router_resid_frac_orig[r] = abs(s_tx - s_rx) / denom

    # Stage 1: Link Hardening (residual-tilted consensus with refined asymmetric partial averaging)
    def asym_partial_update(a_val: float, b_val: float, consensus: float, diff: float, tol: float) -> Tuple[float, float]:
        # Asymmetric partial averaging near threshold to move the louder side more than the quiet side
        k_base = (diff - tol) / max(tol, 1e-9)
        total = max(1e-9, a_val + b_val)
        a_ratio = a_val / total
        b_ratio = b_val / total
        if a_val >= b_val:
            # a is louder
            k_a = clamp(k_base * (1.0 + 0.5 * a_ratio), 0.0, 1.0)
            k_b = clamp(k_base * (1.0 - 0.3 * a_ratio), 0.0, 1.0)
        else:
            # b is louder
            k_b = clamp(k_base * (1.0 + 0.5 * b_ratio), 0.0, 1.0)
            k_a = clamp(k_base * (1.0 - 0.3 * b_ratio), 0.0, 1.0)
        new_a = a_val * (1.0 - k_a) + consensus * k_a
        new_b = b_val * (1.0 - k_b) + consensus * k_b
        return new_a, new_b

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

        # Activity-based base weights
        act_a = max(a_tx, a_rx)
        act_b = max(b_tx, b_rx)
        denom_act = max(1e-9, act_a + act_b)
        w_a_base = act_a / denom_act
        w_b_base = act_b / denom_act

        a_lr = telemetry[a_id].get('local_router')
        b_lr = telemetry[b_id].get('local_router')

        # Direction a->b
        abs_ab = abs(a_tx - b_rx)
        tol_ab = dir_threshold(a_tx, b_rx)
        diff_ab = abs_ab / max(1.0, a_tx, b_rx)
        if diff_ab > tol_ab and abs_ab > ABS_GUARD:
            resid_local = router_resid_frac_orig.get(a_lr, 0.0)
            delta_local = router_delta_orig.get(a_lr, 0.0)
            w_a, w_b = w_a_base, w_b_base
            if resid_local > 0.0 and (a_tx - b_rx) * delta_local > 0.0:
                gamma = min(TILT_GAMMA_MAX, TILT_GAMMA_SCALE * resid_local)
                w_a = clamp(w_a_base - gamma, 0.1, 0.9)
                w_b = 1.0 - w_a
            consensus_ab = w_a * a_tx + w_b * b_rx
            if diff_ab <= 2 * tol_ab:
                vals[a_id]['tx'], vals[b_id]['rx'] = asym_partial_update(a_tx, b_rx, consensus_ab, diff_ab, tol_ab)
            else:
                vals[a_id]['tx'], vals[b_id]['rx'] = consensus_ab, consensus_ab

        # Direction b->a
        abs_ba = abs(b_tx - a_rx)
        tol_ba = dir_threshold(b_tx, a_rx)
        diff_ba = abs_ba / max(1.0, b_tx, a_rx)
        if diff_ba > tol_ba and abs_ba > ABS_GUARD:
            resid_local = router_resid_frac_orig.get(b_lr, 0.0)
            delta_local = router_delta_orig.get(b_lr, 0.0)
            w_a2, w_b2 = w_a_base, w_b_base
            if resid_local > 0.0 and (b_tx - a_rx) * delta_local > 0.0:
                gamma = min(TILT_GAMMA_MAX, TILT_GAMMA_SCALE * resid_local)
                w_b2 = clamp(w_b_base - gamma, 0.1, 0.9)  # local side b
                w_a2 = 1.0 - w_b2
            consensus_ba = w_b2 * b_tx + w_a2 * a_rx
            if diff_ba <= 2 * tol_ba:
                vals[b_id]['tx'], vals[a_id]['rx'] = asym_partial_update(b_tx, a_rx, consensus_ba, diff_ba, tol_ba)
            else:
                vals[b_id]['tx'], vals[a_id]['rx'] = consensus_ba, consensus_ba

    # Stage 2A: Soft Router-Balanced Reconciliation (mirrored, capped, paired only)
    router_sums_after1 = compute_router_sums(vals, router_ifaces)
    for r, if_list in router_ifaces.items():
        sum_tx, sum_rx = router_sums_after1.get(r, (0.0, 0.0))
        denom_r = max(1.0, sum_tx, sum_rx)
        delta = sum_tx - sum_rx
        resid = abs(delta) / denom_r
        if resid < ROUTER_RESID_TRIGGER_SOFT or not if_list:
            continue

        reduce_tx = delta > 0.0
        reduce_rx = delta < 0.0

        # Eligible only if paired and not down
        if reduce_tx:
            cand = [iid for iid in if_list
                    if peers.get(iid) and status_orig.get(iid, 'unknown') != 'down' and vals[iid]['tx'] > 0.0]
            total_mass = sum(vals[iid]['tx'] for iid in cand)
        else:  # reduce_rx
            cand = [iid for iid in if_list
                    if peers.get(iid) and status_orig.get(iid, 'unknown') != 'down' and vals[iid]['rx'] > 0.0]
            total_mass = sum(vals[iid]['rx'] for iid in cand)

        if not cand or total_mass <= 0.0:
            continue

        budget = min(abs(delta), ROUTER_BUDGET_CAP * denom_r)
        if budget <= 0.0:
            continue

        for iid in cand:
            share = (vals[iid]['tx'] / total_mass) if reduce_tx else (vals[iid]['rx'] / total_mass)
            d_i = budget * share
            if d_i <= 0.0:
                continue
            if reduce_tx:
                vals[iid]['tx'] = max(0.0, vals[iid]['tx'] - d_i)
                pid = peers.get(iid)
                if pid and status_orig.get(pid, 'unknown') != 'down':
                    vals[pid]['rx'] = max(0.0, vals[pid]['rx'] - d_i)
            else:
                vals[iid]['rx'] = max(0.0, vals[iid]['rx'] - d_i)
                pid = peers.get(iid)
                if pid and status_orig.get(pid, 'unknown') != 'down':
                    vals[pid]['tx'] = max(0.0, vals[pid]['tx'] - d_i)

    # Stage 2B: Improvement-checked Micro-Adjustments (unpaired dominating interfaces)
    router_sums = compute_router_sums(vals, router_ifaces)
    for r, if_list in router_ifaces.items():
        sum_tx, sum_rx = router_sums.get(r, (0.0, 0.0))
        delta = sum_tx - sum_rx
        denom_r = max(1.0, sum_tx, sum_rx)
        resid_frac = abs(delta) / denom_r
        if resid_frac < ROUTER_RESID_TRIGGER:
            continue
        if not if_list:
            continue

        reduce_tx = delta > 0.0
        reduce_rx = delta < 0.0

        candidates = []
        total_mass = 0.0
        for iid in if_list:
            if iid not in vals:
                continue
            if status_orig.get(iid, 'unknown') != 'up':
                continue
            # unpaired only
            if peers.get(iid) is not None:
                continue
            vtx = float(vals[iid]['tx'])
            vrx = float(vals[iid]['rx'])
            mass = vtx if reduce_tx else (vrx if reduce_rx else 0.0)
            if mass > 0.0:
                candidates.append((mass, iid, vtx, vrx))
                total_mass += mass

        if not candidates or total_mass <= 0.0:
            continue

        candidates.sort(reverse=True)
        top_mass, top_if, old_tx, old_rx = candidates[0]
        if top_mass < DOMINANCE_SHARE * total_mass:
            continue

        alpha = min(MI_ALPHA_CAP, 0.5 * resid_frac)
        if alpha <= 0.0:
            continue

        def compute_router_resid_for(router: str) -> float:
            stx, srx = 0.0, 0.0
            for iid in router_ifaces.get(router, []):
                if iid in vals:
                    stx += float(vals[iid]['tx'])
                    srx += float(vals[iid]['rx'])
            return abs(stx - srx) / max(1.0, stx, srx)

        # Tentative change on the larger counter to reduce imbalance
        if reduce_tx:
            if old_tx >= old_rx:
                new_tx, new_rx = old_tx * (1.0 - alpha), old_rx
            else:
                new_tx, new_rx = old_tx, old_rx * (1.0 + alpha)
        else:
            if old_rx >= old_tx:
                new_rx, new_tx = old_rx * (1.0 - alpha), old_tx
            else:
                new_rx, new_tx = old_rx, old_tx * (1.0 + alpha)

        # Apply tentatively
        vals[top_if]['tx'], vals[top_if]['rx'] = new_tx, new_rx
        resid_before = resid_frac
        resid_after = compute_router_resid_for(r)

        if resid_after > (1.0 - IMPROVE_REQ) * resid_before:
            # Revert if not improved enough
            vals[top_if]['tx'], vals[top_if]['rx'] = old_tx, old_rx

    # Stage 3: Recompute residuals after balancing
    router_sums2 = compute_router_sums(vals, router_ifaces)
    router_resid2: Dict[str, float] = {}
    for r, (s_tx, s_rx) in router_sums2.items():
        router_resid2[r] = abs(s_tx - s_rx) / max(1.0, s_tx, s_rx)

    # Stage 4: Confidence composition and final assembly
    result: Dict[str, Dict[str, Tuple]] = {}
    router_sum_tx: Dict[str, float] = {r: stx for r, (stx, srx) in router_sums2.items()}
    router_sum_rx: Dict[str, float] = {r: srx for r, (stx, srx) in router_sums2.items()}

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

        # Enforce interface consistency: if either side is down, set both down with zero rates
        if (status == 'down') or (peer_exists and peer_status == 'down'):
            repaired_status = 'down'
            both_report_down = (status == 'down' and (peer_exists and peer_status == 'down'))
            status_conf = 0.95 if both_report_down else 0.7
            rep_tx = 0.0
            rep_rx = 0.0
            tx_conf = status_conf
            rx_conf = status_conf
        else:
            repaired_status = status

            resid_local = router_resid2.get(lr, 0.0)
            resid_remote = router_resid2.get(rr, 0.0)

            # Share-aware, direction-coupled penalties (Recommendation 3)
            sum_tx_local = router_sum_tx.get(lr, 0.0)
            sum_rx_local = router_sum_rx.get(lr, 0.0)
            iface_tx_share = rep_tx / max(1.0, sum_tx_local)
            iface_rx_share = rep_rx / max(1.0, sum_rx_local)

            pen_tx = clamp(1.0 - ((0.6 + 0.2 * iface_tx_share) * resid_local +
                                  (0.4 - 0.2 * iface_tx_share) * resid_remote), 0.0, 1.0)
            pen_rx = clamp(1.0 - ((0.6 + 0.2 * iface_rx_share) * resid_local +
                                  (0.4 - 0.2 * iface_rx_share) * resid_remote), 0.0, 1.0)

            # Link symmetry fit after repair
            if peer_exists:
                peer_rep_tx = vals[peer_id]['tx']
                peer_rep_rx = vals[peer_id]['rx']
                diff_tx = rel_diff(rep_tx, peer_rep_rx)  # my_tx vs their_rx
                diff_rx = rel_diff(rep_rx, peer_rep_tx)  # my_rx vs their_tx
                c_sym_tx = clamp(1.0 - diff_tx)
                c_sym_rx = clamp(1.0 - diff_rx)
                # Magnitude-aware floors for strong agreement
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

            # Correction magnitude component
            m_tx = abs(rep_tx - orig_tx) / max(1.0, rep_tx, orig_tx)
            m_rx = abs(rep_rx - orig_rx) / max(1.0, rep_rx, orig_rx)
            c_delta_tx = clamp(1.0 - min(1.0, 1.5 * m_tx))
            c_delta_rx = clamp(1.0 - min(1.0, 1.5 * m_rx))

            # Compose confidences (tri-axis)
            tx_conf = clamp(0.45 * pen_tx + 0.35 * c_sym_tx + 0.20 * c_delta_tx)
            rx_conf = clamp(0.45 * pen_rx + 0.35 * c_sym_rx + 0.20 * c_delta_rx)

            # Apply magnitude-aware floors
            tx_conf = max(tx_conf, floor_tx)
            rx_conf = max(rx_conf, floor_rx)

            # Asymmetric traffic-evidence shaping: penalize silent side if peer shows traffic
            if peer_exists and peer_rep_rx is not None and rep_tx <= QUIET_EPS and peer_rep_rx > QUIET_EPS:
                tx_conf = clamp(tx_conf * 0.88)
            if peer_exists and peer_rep_tx is not None and rep_rx <= QUIET_EPS and peer_rep_tx > QUIET_EPS:
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
            # Align mildly with per-direction confidences and penalties
            status_conf = clamp(status_conf * (0.85 + 0.15 * min(tx_conf, rx_conf)))
            status_conf = clamp(status_conf * (0.85 + 0.15 * min(pen_tx, pen_rx)))

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
