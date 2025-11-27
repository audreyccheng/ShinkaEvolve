# EVOLVE-BLOCK-START
"""
Network telemetry repair algorithm that detects and corrects inconsistencies
in network interface telemetry data using topology relationships.

This version implements "tilted_prescale_asym_v3":
- Gentle, gated multiplicative pair-bias prescaling used only to compute consensus
- Residual-tilted, trust-weighted directional consensus with asymmetric partial averaging
- Improvement-checked router micro-adjustments (evaluate both directions, optional second mini-step)
- Tri-axis confidence with residual-severity–adaptive, share-aware penalties; post-repair symmetry fit;
  correction magnitude; plus asymmetric traffic-evidence shaping and ultra-agreement floors
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

    # Core thresholds and guards
    HARDENING_THRESHOLD = 0.02       # 2% (normal rates)
    LOW_RATE_CUTOFF = 10.0           # Mbps (low-rate band)
    LOW_RATE_THRESHOLD = 0.05        # 5% for low rates
    ABS_GUARD = 0.5                  # Mbps absolute guard
    QUIET_EPS = 0.1                  # Mbps silence threshold

    # Prescale parameters (gentle harmonization, Recommendation 3)
    PRESCALE_MIN = 0.90
    PRESCALE_MAX = 1.10
    PRESCALE_SKIP_RATE = 1.0         # Mbps; skip prescale on very tiny links

    # Residual-tilt parameters (Recommendation 1)
    TILT_GAMMA_MAX = 0.08            # max absolute tilt
    TILT_GAMMA_SCALE = 0.10          # γ = min(0.08, 0.1 * |resid_signed|)

    # Partial averaging shaping (Recommendation 2)
    FULL_MULT_LOW = 1.6
    FULL_MULT_HIGH = 2.0
    EXP_LOW = 1.2
    EXP_HIGH = 1.0

    # Micro-adjuster parameters (Recommendation 5)
    ROUTER_RESID_TRIGGER = 0.03      # residual ≥ 3% to try micro-adjust
    MI_ALPHA_CAP = 0.02              # first-step cap 2%
    MI_ALPHA2_CAP = 0.01             # second mini-step cap 1%
    MI_IMPROVE_REQ_COMMIT = 0.10     # require ≥10% improvement for 1st commit
    MI_IMPROVE_REQ_SECOND = 0.20     # require ≥20% improvement to allow 2nd step
    DOM_CONTRIB_FACTOR = 0.5         # dominating contrib ≥ 50% of |imbalance|
    LOW_RATE_NUDGE_SKIP = LOW_RATE_CUTOFF  # skip nudging tiny flows

    def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
        return max(lo, min(hi, x))

    def rel_diff(a: float, b: float) -> float:
        return abs(a - b) / max(1.0, a, b)

    def dir_threshold(a: float, b: float) -> float:
        return LOW_RATE_THRESHOLD if max(a, b) < LOW_RATE_CUTOFF else HARDENING_THRESHOLD

    # Build peer mapping and collect originals
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

    # Helper to compute per-router sums from provided values
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

    # Compute original signed router residuals (for tilt)
    router_sums_orig = compute_router_sums(orig)
    router_resid_signed: Dict[str, float] = {}
    for r, (s_tx, s_rx) in router_sums_orig.items():
        denom = max(1.0, s_tx, s_rx)
        router_resid_signed[r] = (s_tx - s_rx) / denom  # signed

    # Stage 1: Residual-tilted, trust-weighted link consensus with gentle prescale and asymmetric partial averaging
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
        if sa == 'down' or sb == 'down':
            # Skip counter fusion; down handling applied later
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

        # Utility: asymmetric partial averaging with loud/quiet shaping
        def asym_partial(local_val: float, peer_val: float, consensus: float,
                         diff: float, tol: float, low_band: bool) -> Tuple[float, float]:
            exp = EXP_LOW if low_band else EXP_HIGH
            full_mult = FULL_MULT_LOW if low_band else FULL_MULT_HIGH
            if diff <= tol:
                return local_val, peer_val
            if diff > full_mult * tol:
                return consensus, consensus
            k_base = ((diff - tol) / max(tol, 1e-9)) ** exp
            # loud/quiet ratio
            loud = max(local_val, peer_val)
            quiet = min(local_val, peer_val)
            r = (loud - quiet) / max(1.0, loud)
            k_loud = clamp(k_base * (1.0 + 0.5 * r), 0.0, 1.0)
            k_quiet = clamp(k_base * (1.0 - 0.5 * r), 0.0, 1.0)
            if local_val >= peer_val:
                new_local = local_val * (1.0 - k_loud) + consensus * k_loud
                new_peer = peer_val * (1.0 - k_quiet) + consensus * k_quiet
            else:
                new_local = local_val * (1.0 - k_quiet) + consensus * k_quiet
                new_peer = peer_val * (1.0 - k_loud) + consensus * k_loud
            return new_local, new_peer

        # Gentle pair-bias prescaling for each direction (used only for computing consensus and k)
        def prescale_pair(local_side: float, peer_side: float, tol: float) -> Tuple[float, float]:
            # Gate prescale: only when diff > tol and |delta| > abs guard; skip on tiny rates
            abs_delta = abs(local_side - peer_side)
            if max(local_side, peer_side) < PRESCALE_SKIP_RATE:
                return local_side, peer_side
            if not (abs_delta > ABS_GUARD and rel_diff(local_side, peer_side) > tol):
                return local_side, peer_side
            s = sqrt(max(1e-9, peer_side) / max(1e-9, local_side))
            s = clamp(s, PRESCALE_MIN, PRESCALE_MAX)
            return local_side * s, peer_side / s

        # Direction a->b: compare a_tx with b_rx
        abs_ab = abs(a_tx - b_rx)
        tol_ab = dir_threshold(a_tx, b_rx)
        low_band_ab = max(a_tx, b_rx) < LOW_RATE_CUTOFF
        if abs_ab > ABS_GUARD and rel_diff(a_tx, b_rx) > tol_ab:
            # Residual-tilted weights
            w_a, w_b = w_a_base, w_b_base
            resid_signed = router_resid_signed.get(a_lr, 0.0)
            # If link discrepancy agrees with local router residual sign, tilt away from local
            if resid_signed != 0.0 and (a_tx - b_rx) * resid_signed > 0.0:
                gamma = min(TILT_GAMMA_MAX, TILT_GAMMA_SCALE * abs(resid_signed))
                w_a = clamp(w_a_base - gamma, 0.2, 0.8)
                w_b = 1.0 - w_a
            # Prescale for consensus computation
            a_tx_p, b_rx_p = prescale_pair(a_tx, b_rx, tol_ab)
            consensus_ab = w_a * a_tx_p + w_b * b_rx_p
            new_a_tx, new_b_rx = asym_partial(a_tx, b_rx, consensus_ab,
                                              rel_diff(a_tx_p, b_rx_p), tol_ab, low_band_ab)
            vals[a_id]['tx'], vals[b_id]['rx'] = new_a_tx, new_b_rx

        # Direction b->a: compare b_tx with a_rx
        abs_ba = abs(b_tx - a_rx)
        tol_ba = dir_threshold(b_tx, a_rx)
        low_band_ba = max(b_tx, a_rx) < LOW_RATE_CUTOFF
        if abs_ba > ABS_GUARD and rel_diff(b_tx, a_rx) > tol_ba:
            w_b2, w_a2 = w_b_base, w_a_base
            resid_signed = router_resid_signed.get(b_lr, 0.0)
            if resid_signed != 0.0 and (b_tx - a_rx) * resid_signed > 0.0:
                gamma = min(TILT_GAMMA_MAX, TILT_GAMMA_SCALE * abs(resid_signed))
                w_b2 = clamp(w_b_base - gamma, 0.2, 0.8)
                w_a2 = 1.0 - w_b2
            b_tx_p, a_rx_p = prescale_pair(b_tx, a_rx, tol_ba)
            consensus_ba = w_b2 * b_tx_p + w_a2 * a_rx_p
            new_b_tx, new_a_rx = asym_partial(b_tx, a_rx, consensus_ba,
                                              rel_diff(b_tx_p, a_rx_p), tol_ba, low_band_ba)
            vals[b_id]['tx'], vals[a_id]['rx'] = new_b_tx, new_a_rx

    # Stage 2: Router-level micro-adjustments (evaluate both directions + optional second mini-step)
    # Work on current vals; compute router sums and residuals
    router_sums = compute_router_sums(vals)

    def compute_router_resid_from_sums(stx: float, srx: float) -> float:
        return abs(stx - srx) / max(1.0, stx, srx)

    # Record micro-adjust actions to tune confidence later
    micro_adjust_dir: Dict[str, Tuple[str, float]] = {}  # iface -> (dir, alpha_total)

    for r, if_list in router_ifaces.items():
        sum_tx, sum_rx = router_sums.get(r, (0.0, 0.0))
        resid_before = compute_router_resid_from_sums(sum_tx, sum_rx)
        if resid_before < ROUTER_RESID_TRIGGER:
            continue
        delta = sum_tx - sum_rx  # positive: tx excess

        # Choose dominating unpaired, up interface with non-trivial traffic
        candidates = []
        for iid in if_list:
            if iid not in vals:
                continue
            if status_orig.get(iid, 'unknown') != 'up':
                continue
            # unpaired if peer missing in telemetry
            if peers.get(iid) is not None:
                continue
            txv = float(vals[iid]['tx'])
            rxv = float(vals[iid]['rx'])
            if max(txv, rxv) < LOW_RATE_NUDGE_SKIP:
                continue
            contrib = abs(txv - rxv)
            candidates.append((contrib, iid, txv, rxv))

        if not candidates:
            continue

        candidates.sort(reverse=True)
        top_contrib, top_if, txv, rxv = candidates[0]
        if top_contrib < DOM_CONTRIB_FACTOR * abs(delta):
            continue

        # Evaluate both nudge options and pick better
        alpha = min(MI_ALPHA_CAP, 0.5 * resid_before)
        if alpha <= 0.0:
            continue

        def eval_option(adjust_dir: str) -> Tuple[float, float, float]:
            # Return (resid_after, new_tx, new_rx)
            new_tx, new_rx = txv, rxv
            if delta > 0.0:  # tx excess
                if adjust_dir == 'tx':
                    if txv >= rxv:
                        new_tx = txv * (1.0 - alpha)
                    else:
                        return float('inf'), txv, rxv  # avoid worsening internal skew
                else:  # 'rx'
                    if rxv <= txv:
                        new_rx = rxv * (1.0 + alpha)
                    else:
                        return float('inf'), txv, rxv
            else:  # rx excess
                if adjust_dir == 'rx':
                    if rxv >= txv:
                        new_rx = rxv * (1.0 - alpha)
                    else:
                        return float('inf'), txv, rxv
                else:  # 'tx'
                    if txv <= rxv:
                        new_tx = txv * (1.0 + alpha)
                    else:
                        return float('inf'), txv, rxv
            stx = sum_tx - txv + new_tx
            srx = sum_rx - rxv + new_rx
            return compute_router_resid_from_sums(stx, srx), new_tx, new_rx

        resid1_tx, cand_tx, cand_rx1 = eval_option('tx')
        resid1_rx, cand_tx2, cand_rx2 = eval_option('rx')
        pick = None
        if resid1_tx < resid1_rx:
            if resid1_tx <= (1.0 - MI_IMPROVE_REQ_COMMIT) * resid_before:
                pick = ('tx', cand_tx, cand_rx1, resid1_tx)
        else:
            if resid1_rx <= (1.0 - MI_IMPROVE_REQ_COMMIT) * resid_before:
                pick = ('rx', cand_tx2, cand_rx2, resid1_rx)

        if pick is None:
            continue

        chosen_dir, new_tx, new_rx, resid_after = pick
        vals[top_if]['tx'], vals[top_if]['rx'] = new_tx, new_rx
        router_sums[r] = (sum_tx - txv + new_tx, sum_rx - rxv + new_rx)
        micro_adjust_dir[top_if] = (chosen_dir, alpha)

        # Optional second mini-step if conditions permit
        resid_now = resid_after
        if resid_before >= 0.04 and resid_now <= (1.0 - MI_IMPROVE_REQ_SECOND) * resid_before:
            alpha2 = min(MI_ALPHA2_CAP, 0.5 * resid_now)
            if alpha2 > 0.0:
                # Recompute sums baseline
                sum_tx2, sum_rx2 = router_sums[r]
                txv2, rxv2 = vals[top_if]['tx'], vals[top_if]['rx']
                # Try same direction again conservatively
                if chosen_dir == 'tx':
                    if delta > 0.0 and txv2 >= rxv2:
                        cand2_tx = txv2 * (1.0 - alpha2)
                        stx = sum_tx2 - txv2 + cand2_tx
                        srx = sum_rx2
                        resid2 = compute_router_resid_from_sums(stx, srx)
                        if resid2 < resid_now:
                            vals[top_if]['tx'] = cand2_tx
                            router_sums[r] = (stx, srx)
                            micro_adjust_dir[top_if] = (chosen_dir, alpha + alpha2)
                else:  # 'rx'
                    if delta < 0.0 and rxv2 >= txv2:
                        cand2_rx = rxv2 * (1.0 - alpha2)
                        stx = sum_tx2
                        srx = sum_rx2 - rxv2 + cand2_rx
                        resid2 = compute_router_resid_from_sums(stx, srx)
                        if resid2 < resid_now:
                            vals[top_if]['rx'] = cand2_rx
                            router_sums[r] = (stx, srx)
                            micro_adjust_dir[top_if] = (chosen_dir, alpha + alpha2)

    # Stage 3: Recompute residuals after micro-adjustments
    router_sums2 = compute_router_sums(vals)
    router_resid2: Dict[str, float] = {}
    for r, (s_tx, s_rx) in router_sums2.items():
        router_resid2[r] = abs(s_tx - s_rx) / max(1.0, s_tx, s_rx)

    # Stage 4: Final assembly with calibrated confidences (tri-axis + residual-severity refinement)
    result: Dict[str, Dict[str, Tuple]] = {}
    # Precompute per-router directional sums for share calculation
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
        force_down = (status == 'down') or (peer_exists and peer_status == 'down')
        if force_down:
            repaired_status = 'down'
            both_report_down = (status == 'down' and (peer_exists and peer_status == 'down'))
            status_conf = 0.95 if both_report_down else 0.7
            rep_tx = 0.0
            rep_rx = 0.0
            # Base conf when forced down
            tx_conf = status_conf
            rx_conf = status_conf
        else:
            repaired_status = status

            # Directional shares for confidence penalties
            sum_tx_local = router_sum_tx.get(lr, 0.0)
            sum_rx_local = router_sum_rx.get(lr, 0.0)
            iface_tx_share = rep_tx / max(1.0, sum_tx_local)
            iface_rx_share = rep_rx / max(1.0, sum_rx_local)

            resid_local = router_resid2.get(lr, 0.0)
            resid_remote = router_resid2.get(rr, 0.0)
            resid_sev = max(resid_local, resid_remote)
            # Residual-severity–adaptive penalties (Recommendation 4)
            if resid_sev < 0.03:
                amp = 0.1
            elif resid_sev < 0.12:
                amp = 0.2
            else:
                amp = 0.3
            pen_tx = clamp(1.0 - ((0.6 + amp * iface_tx_share) * resid_local +
                                  (0.4 - amp * iface_tx_share) * resid_remote), 0.0, 1.0)
            pen_rx = clamp(1.0 - ((0.6 + amp * iface_rx_share) * resid_local +
                                  (0.4 - amp * iface_rx_share) * resid_remote), 0.0, 1.0)

            # Link symmetry fit on repaired values
            if peer_exists:
                peer_rep_tx = vals[peer_id]['tx']
                peer_rep_rx = vals[peer_id]['rx']
                diff_tx = rel_diff(rep_tx, peer_rep_rx)  # my_tx vs their_rx
                diff_rx = rel_diff(rep_rx, peer_rep_tx)  # my_rx vs their_tx
                c_sym_tx = clamp(1.0 - diff_tx)
                c_sym_rx = clamp(1.0 - diff_rx)

                # Magnitude-aware floors for strong/ultra agreement
                floor_tx = 0.0
                floor_rx = 0.0
                th_tx = dir_threshold(rep_tx, peer_rep_rx)
                th_rx = dir_threshold(rep_rx, peer_rep_tx)
                ultra_agree = False
                if max(rep_tx, peer_rep_rx) >= 10.0 and diff_tx <= 0.005:
                    floor_tx = 0.99
                elif diff_tx <= th_tx:
                    floor_tx = 0.98 if max(rep_tx, peer_rep_rx) >= 10.0 else 0.97
                if max(rep_rx, peer_rep_tx) >= 10.0 and diff_rx <= 0.005:
                    floor_rx = 0.99
                elif diff_rx <= th_rx:
                    floor_rx = 0.98 if max(rep_rx, peer_rep_tx) >= 10.0 else 0.97
                # Ultra-agreement floor (Recommendation 4)
                if diff_tx <= 0.003 and diff_rx <= 0.003 and max(resid_local, resid_remote) <= 0.02:
                    ultra_agree = True
                    floor_tx = max(floor_tx, 0.995)
                    floor_rx = max(floor_rx, 0.995)
            else:
                peer_rep_tx = None
                peer_rep_rx = None
                c_sym_tx = 0.9
                c_sym_rx = 0.9
                floor_tx = 0.0
                floor_rx = 0.0
                ultra_agree = False

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

            # Asymmetric traffic-evidence shaping:
            # penalize silent side if peer shows traffic; relax to 0.92 on low-rate tiny peer traffic
            if peer_exists and peer_rep_rx is not None:
                if rep_tx <= QUIET_EPS and peer_rep_rx > QUIET_EPS:
                    if max(rep_tx, peer_rep_rx) < LOW_RATE_CUTOFF and peer_rep_rx <= 2.0:
                        tx_conf = clamp(tx_conf * 0.92)
                    else:
                        tx_conf = clamp(tx_conf * 0.88)
            if peer_exists and peer_rep_tx is not None:
                if rep_rx <= QUIET_EPS and peer_rep_tx > QUIET_EPS:
                    if max(rep_rx, peer_rep_tx) < LOW_RATE_CUTOFF and peer_rep_tx <= 2.0:
                        rx_conf = clamp(rx_conf * 0.92)
                    else:
                        rx_conf = clamp(rx_conf * 0.88)

            # Micro-adjustment confidence shaping: lower confidence on the adjusted direction
            if if_id in micro_adjust_dir:
                adj_dir, a_total = micro_adjust_dir[if_id]
                # map total alpha to a conservative confidence band [0.6, 0.8]
                scale = 0.6 + 0.2 * clamp(a_total / MI_ALPHA_CAP, 0.0, 1.0)
                if adj_dir == 'tx':
                    tx_conf = min(tx_conf, scale)
                    # mildly couple the other direction
                    rx_conf = min(rx_conf, max(0.0, 0.95 * rx_conf))
                else:
                    rx_conf = min(rx_conf, scale)
                    tx_conf = min(tx_conf, max(0.0, 0.95 * tx_conf))

            # Status confidence shaping
            status_conf = 0.95
            if peer_exists and repaired_status != peer_status and peer_status != 'down':
                status_conf = min(status_conf, 0.6)
            # Strong bilateral agreement boosts status confidence
            if peer_exists:
                if ultra_agree:
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