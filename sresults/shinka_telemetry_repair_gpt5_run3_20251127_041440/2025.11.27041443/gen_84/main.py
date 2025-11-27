# EVOLVE-BLOCK-START
"""
Bundle-aware consensus projection for network telemetry repair.

Algorithm summary:
1) Robust link hardening with adaptive tolerance and soft-zero snapping.
2) Router flow conservation via targeted, confidence-weighted corrections.
   - Bundle-aware scaling for parallel links (shared factor).
   - Per-interface clipping ±10% and 60% damping.
2.6) Conservation-preserving bundle finishing: median-residual zero-sum micro-alignment across parallel links.
3) Confidence-gap-proportional re-sync on links with scaling guard and absolute-per-dir caps.
4) Confidence calibrated from measurement residuals, link residuals, router residuals,
   plus a scale-factor term and peer smoothing.

Maintains inputs/outputs of the original function.
"""
from typing import Dict, Any, Tuple, List
import math


def repair_network_telemetry(telemetry: Dict[str, Dict[str, Any]],
                             topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:
    # Hyperparameters
    TAU_H_BASE = 0.02        # ~2% hardening threshold
    ZERO_EPS = 1e-6
    ZERO_THRESH = 1.0        # Mbps near-zero threshold
    DAMP_ROUTER = 0.60       # router damping factor
    PER_LINK_CLIP = 0.10     # per-interface relative change cap (±10%)
    BUNDLE_CLIP = 0.15       # bundle shared factor cap (±15%)
    STRONG_SCALE_GUARD = 0.08  # guard for re-sync when strong router scaling applied
    RESYNC_MAX_F = 0.40      # max one-sided nudge toward mean
    RESYNC_PER_DIR_CAP = 0.02  # per-direction absolute move cap as fraction of max value
    PEER_SMOOTH = 0.10       # 10% peer smoothing
    WEIGHT_FOCUS = 0.70      # focus router correction on lowest-confidence 70% weight
    DOMINANCE_CAP = 0.50     # cap any single interface's weight share to ≤50% in a pass
    INTRA_BUNDLE_CLIP = 0.05 # ±5% intra-bundle smoothing cap (reserved)
    CLIP_HIT_PENALTY = 0.95  # confidence penalty multiplier when clipping/strong scaling hit
    UNTOUCHED_BOOST = 0.02   # confidence boost for untouched, well-synced counters
    # New: bundle finishing micro-alignment params
    BUNDLE_FINISH_CLIP = 0.03      # ±3% micro alignment cap
    BUNDLE_FINISH_GAIN_MAX = 0.25  # max gain for bundle finishing pass

    def clamp01(x: float) -> float:
        return max(0.0, min(1.0, x))

    def rel_diff(a: float, b: float) -> float:
        return abs(a - b) / max(1.0, abs(a), abs(b))

    def adaptive_tau(v1: float, v2: float) -> float:
        # Adaptive symmetry tolerance:
        # tighter for high rates, looser for low/near-zero or low confidence regions.
        if v1 >= 100.0 and v2 >= 100.0:
            return 0.015
        if v1 < ZERO_THRESH or v2 < ZERO_THRESH:
            return 0.03
        return TAU_H_BASE

    # Build basic maps
    orig_rx: Dict[str, float] = {}
    orig_tx: Dict[str, float] = {}
    status: Dict[str, str] = {}
    peer_of: Dict[str, str] = {}
    local_router_of: Dict[str, Any] = {}
    remote_router_of: Dict[str, Any] = {}

    for iid, d in telemetry.items():
        orig_rx[iid] = float(d.get('rx_rate', 0.0))
        orig_tx[iid] = float(d.get('tx_rate', 0.0))
        status[iid] = d.get('interface_status', 'unknown')
        ct = d.get('connected_to')
        peer_of[iid] = ct if ct in telemetry else None
        local_router_of[iid] = d.get('local_router')
        remote_router_of[iid] = d.get('remote_router')

    # Build router->interfaces mapping, prefer provided topology
    router_ifaces: Dict[str, List[str]] = {}
    if topology:
        for r, ifs in topology.items():
            router_ifaces[r] = [i for i in ifs if i in telemetry]
    else:
        # Fallback to local_router fields
        for iid, d in telemetry.items():
            r = d.get('local_router')
            if r is not None:
                router_ifaces.setdefault(r, []).append(iid)

    # Derive missing remote_router via peer's local_router if needed
    for iid in telemetry:
        if not remote_router_of.get(iid):
            p = peer_of.get(iid)
            if p and p in telemetry:
                remote_router_of[iid] = telemetry[p].get('local_router')

    # Build link pairs (unique, undirected)
    link_pairs: List[Tuple[str, str]] = []
    seen = set()
    for a in telemetry:
        b = peer_of.get(a)
        if not b or b not in telemetry or a == b:
            continue
        if (b, a) in seen or (a, b) in seen:
            continue
        seen.add((a, b))
        link_pairs.append((a, b))

    # Initialize hardened values with originals
    hardened_rx: Dict[str, float] = {i: max(0.0, orig_rx[i]) for i in telemetry}
    hardened_tx: Dict[str, float] = {i: max(0.0, orig_tx[i]) for i in telemetry}
    # Initialize confidences (will be calibrated later)
    conf_rx: Dict[str, float] = {i: 0.7 for i in telemetry}
    conf_tx: Dict[str, float] = {i: 0.7 for i in telemetry}

    # Track cumulative router scaling factors for re-sync guard and confidence
    scaled_rx_factor: Dict[str, float] = {i: 1.0 for i in telemetry}
    scaled_tx_factor: Dict[str, float] = {i: 1.0 for i in telemetry}
    # Track if a direction hit clipping (±10% per-pass cap or strong scaling)
    clip_hit_rx: Dict[str, bool] = {i: False for i in telemetry}
    clip_hit_tx: Dict[str, bool] = {i: False for i in telemetry}

    # Stage 1: Robust link hardening with adaptive tolerance and soft-zero
    for a, b in link_pairs:
        a_up = (status.get(a) == 'up')
        b_up = (status.get(b) == 'up')
        a_rx, a_tx = orig_rx[a], orig_tx[a]
        b_rx, b_tx = orig_rx[b], orig_tx[b]

        # If either side is down: force both to zero on the link with high confidence
        if not a_up or not b_up:
            for i in (a, b):
                hardened_rx[i] = 0.0
                hardened_tx[i] = 0.0
                conf_rx[i] = max(conf_rx[i], 0.85)
                conf_tx[i] = max(conf_tx[i], 0.85)
            continue

        # Soft-zero stabilization for near-zero links
        if max(a_rx, a_tx, b_rx, b_tx) < 2.0 * ZERO_THRESH:
            for i in (a, b):
                hardened_rx[i] = 0.0
                hardened_tx[i] = 0.0
                conf_rx[i] = max(conf_rx[i], 0.95)
                conf_tx[i] = max(conf_tx[i], 0.95)
            continue

        # Direction 1: a.tx vs b.rx
        d1 = rel_diff(a_tx, b_rx)
        tau1 = adaptive_tau(a_tx, b_rx)
        if d1 <= tau1:
            v1 = 0.5 * (a_tx + b_rx)
            hardened_tx[a] = v1
            hardened_rx[b] = v1
            c1 = clamp01(0.9 + 0.1 * (1.0 - d1 / max(tau1, 1e-12)))
            conf_tx[a] = max(conf_tx[a], c1)
            conf_rx[b] = max(conf_rx[b], c1)
        else:
            choice = b_rx if abs(b_rx) >= abs(a_tx) else a_tx
            hardened_tx[a] = max(0.0, choice)
            hardened_rx[b] = max(0.0, choice)
            c1 = clamp01(1.0 - d1)
            conf_tx[a] = max(conf_tx[a], c1)
            conf_rx[b] = max(conf_rx[b], c1)

        # Direction 2: a.rx vs b.tx
        d2 = rel_diff(a_rx, b_tx)
        tau2 = adaptive_tau(a_rx, b_tx)
        if d2 <= tau2:
            v2 = 0.5 * (a_rx + b_tx)
            hardened_rx[a] = v2
            hardened_tx[b] = v2
            c2 = clamp01(0.9 + 0.1 * (1.0 - d2 / max(tau2, 1e-12)))
            conf_rx[a] = max(conf_rx[a], c2)
            conf_tx[b] = max(conf_tx[b], c2)
        else:
            choice = b_tx if abs(b_tx) >= abs(a_rx) else a_rx
            hardened_rx[a] = max(0.0, choice)
            hardened_tx[b] = max(0.0, choice)
            c2 = clamp01(1.0 - d2)
            conf_rx[a] = max(conf_rx[a], c2)
            conf_tx[b] = max(conf_tx[b], c2)

    # Unpaired interfaces: trust local with moderate confidence; zero if down
    in_pairs = {x for pair in link_pairs for x in pair}
    for i in telemetry:
        if i not in in_pairs:
            if status.get(i) != 'up':
                hardened_rx[i] = 0.0
                hardened_tx[i] = 0.0
                conf_rx[i] = max(conf_rx[i], 0.85)
                conf_tx[i] = max(conf_tx[i], 0.85)
            else:
                hardened_rx[i] = max(0.0, orig_rx[i])
                hardened_tx[i] = max(0.0, orig_tx[i])
                conf_rx[i] = max(conf_rx[i], 0.6)
                conf_tx[i] = max(conf_tx[i], 0.6)

    # Stage 2: Router flow conservation with targeted, bundle-aware corrections
    for r, ifs in router_ifaces.items():
        if not ifs:
            continue
        up_ifs = [i for i in ifs if status.get(i) == 'up']
        if len(up_ifs) < 2:
            continue

        sum_rx = sum(hardened_rx[i] for i in up_ifs)
        sum_tx = sum(hardened_tx[i] for i in up_ifs)
        denom = max(1.0, sum_rx, sum_tx)
        imbalance = (sum_rx - sum_tx)  # positive means rx > tx
        rel_gap = abs(imbalance) / denom

        # Adaptive router tolerance based on number of active interfaces
        n_active = len(up_ifs)
        tau_router = min(0.07, max(0.03, 0.05 * (2.0 / max(2, n_active)) ** 0.5))
        if rel_gap <= tau_router:
            continue

        # Choose side via expected-penalty simulation with dominance awareness
        def side_penalty(side: str) -> float:
            vals = [(i, (hardened_rx[i] if side == 'rx' else hardened_tx[i]),
                     clamp01(conf_rx[i] if side == 'rx' else conf_tx[i]))
                    for i in up_ifs]
            # weights w_i = (1 - conf_i) * rate_i
            w = {i: (1.0 - c) * max(v, ZERO_THRESH) for (i, v, c) in vals}
            total_w = sum(w.values()) or 1.0
            # Focused subset covering WEIGHT_FOCUS of total weight
            order = sorted(up_ifs, key=lambda x: w[x], reverse=True)
            focus: List[str] = []
            acc_w = 0.0
            for i2 in order:
                if acc_w / total_w >= WEIGHT_FOCUS:
                    break
                focus.append(i2)
                acc_w += w[i2]
            if not focus:
                focus = list(up_ifs)
                acc_w = total_w
            # Dominance-aware cap in focus
            cap_per = DOMINANCE_CAP * acc_w
            eff_w = {i3: min(w[i3], cap_per) for i3 in focus}
            eff_total = sum(eff_w.values()) or 1.0
            # HHI concentration metric
            hhi = sum((eff_w[i4] / eff_total) ** 2 for i4 in focus)

            # Simulate two-tier scaling using current weights
            delta = (-imbalance if side == 'rx' else imbalance)
            denom_k = 0.0
            for i5 in focus:
                vi = hardened_rx[i5] if side == 'rx' else hardened_tx[i5]
                denom_k += vi * eff_w[i5]
            k = 0.0 if denom_k == 0.0 else delta / (0.6 * denom_k)

            penalty = 0.0
            for i6 in focus:
                vi = hardened_rx[i6] if side == 'rx' else hardened_tx[i6]
                ci = clamp01(conf_rx[i6] if side == 'rx' else conf_tx[i6])
                # confidence-dependent ceiling: 1.12 if conf < 0.70 else 1.10
                clip_hi = 1.12 if ci < 0.70 else 1.10
                scale_sim = 1.0 + 0.6 * k * eff_w[i6]
                # apply lower bound and ceiling
                scale_sim = max(0.90, min(clip_hi, scale_sim))
                penalty += abs(scale_sim - 1.0) + 0.5 * (1.0 - ci)
            # concentration cost
            penalty += 0.05 * hhi
            return penalty

        pen_rx = side_penalty('rx')
        pen_tx = side_penalty('tx')
        if pen_rx + 1e-9 < 0.95 * (pen_tx + 1e-9):
            adjust_side = 'rx'
        elif pen_tx + 1e-9 < 0.95 * (pen_rx + 1e-9):
            adjust_side = 'tx'
        else:
            avg_rx_conf = sum(conf_rx[i] for i in up_ifs) / len(up_ifs)
            avg_tx_conf = sum(conf_tx[i] for i in up_ifs) / len(up_ifs)
            adjust_side = 'rx' if avg_rx_conf < avg_tx_conf else 'tx'
        total_adjust = (-imbalance if adjust_side == 'rx' else imbalance) * DAMP_ROUTER

        # Build per-interface values and confidences for the chosen side
        side_vals = {i: (hardened_rx[i] if adjust_side == 'rx' else hardened_tx[i]) for i in up_ifs}
        side_confs = {i: (conf_rx[i] if adjust_side == 'rx' else conf_tx[i]) for i in up_ifs}

        # Group interfaces into bundles by (local_router, remote_router)
        bundle_map: Dict[Tuple[Any, Any], List[str]] = {}
        for i in up_ifs:
            lr = local_router_of.get(i)
            rr = remote_router_of.get(i)
            if not rr:
                p = peer_of.get(i)
                if p:
                    rr = local_router_of.get(p)
            key = (lr, rr)
            bundle_map.setdefault(key, []).append(i)

        side_total = sum(side_vals.values())
        majority_bundles = []
        for key, members in bundle_map.items():
            s = sum(side_vals[m] for m in members)
            if side_total > ZERO_EPS and s >= 0.5 * side_total:
                majority_bundles.append((key, members, s))

        # Build weights for distribution: w_i = (1 - conf) * max(val, ZERO_THRESH)
        weights = {i: (1.0 - clamp01(side_confs[i])) * max(side_vals[i], ZERO_THRESH) + 1e-9 for i in up_ifs}
        total_w = sum(weights.values())
        if total_w <= 0:
            weights = {i: 1.0 for i in up_ifs}
            total_w = float(len(up_ifs))

        # Focus adjustments on the lowest-confidence subset covering WEIGHT_FOCUS of total weight
        sorted_ifs = sorted(up_ifs, key=lambda x: weights[x], reverse=True)
        focus_set: List[str] = []
        acc = 0.0
        for i in sorted_ifs:
            if acc / max(total_w, 1e-12) >= WEIGHT_FOCUS:
                break
            focus_set.append(i)
            acc += weights[i]
        if not focus_set:
            focus_set = list(up_ifs)
            acc = total_w
        focus_total_w = max(acc, 1e-9)

        # Apply bundle-aware scaling if there is a majority bundle; else per-interface adjustments
        if majority_bundles:
            bundle_weights: Dict[Tuple[Any, Any], float] = {}
            bundle_members_focused: Dict[Tuple[Any, Any], List[str]] = {}
            for key, members, _ in majority_bundles:
                focused_members = [m for m in members if m in focus_set]
                if not focused_members:
                    continue
                bundle_members_focused[key] = focused_members
                bundle_weights[key] = sum(weights[m] for m in focused_members)

            # First, scale majority bundles with a shared factor per bundle on focused members only
            for key, members, _ in majority_bundles:
                focused_members = bundle_members_focused.get(key, [])
                if not focused_members:
                    continue
                w_g = bundle_weights.get(key, 0.0)
                if w_g <= 0.0:
                    continue
                adj_g = total_adjust * (w_g / focus_total_w)
                s_sum_focus = sum(side_vals[m] for m in focused_members)
                if s_sum_focus <= ZERO_EPS:
                    continue
                target_sum = max(0.0, s_sum_focus + adj_g)
                scale_g = target_sum / s_sum_focus
                clipped = False
                if scale_g > 1.0 + BUNDLE_CLIP:
                    scale_g = 1.0 + BUNDLE_CLIP
                    clipped = True
                elif scale_g < 1.0 - BUNDLE_CLIP:
                    scale_g = 1.0 - BUNDLE_CLIP
                    clipped = True
                for m in focused_members:
                    old = side_vals[m]
                    new = max(0.0, scale_g * old)
                    if adjust_side == 'rx':
                        prev = hardened_rx[m]
                        if prev > ZERO_EPS:
                            scaled_rx_factor[m] *= (new / prev)
                        hardened_rx[m] = new
                        relc = abs(new - old) / max(1.0, abs(old))
                        conf_rx[m] = clamp01(conf_rx[m] * (1.0 - 0.6 * relc))
                        if clipped:
                            clip_hit_rx[m] = True
                    else:
                        prev = hardened_tx[m]
                        if prev > ZERO_EPS:
                            scaled_tx_factor[m] *= (new / prev)
                        hardened_tx[m] = new
                        relc = abs(new - old) / max(1.0, abs(old))
                        conf_tx[m] = clamp01(conf_tx[m] * (1.0 - 0.6 * relc))
                        if clipped:
                            clip_hit_tx[m] = True

            # Then, adjust remaining focused non-majority interfaces individually using capped shares
            all_major_members = [m for _, members, _ in majority_bundles for m in members]
            non_majority_all = [i for i in up_ifs if i not in all_major_members]
            non_majority = [i for i in non_majority_all if i in focus_set]
            nm_total_w = sum(weights[i] for i in non_majority)
            if nm_total_w > 0:
                cap_per = DOMINANCE_CAP * nm_total_w
                eff_weights = {i: min(weights[i], cap_per) for i in non_majority}
                eff_total_w = max(1e-9, sum(eff_weights.values()))
                adj_nm = total_adjust * (nm_total_w / focus_total_w)
                for i in non_majority:
                    v_old = side_vals[i]
                    w_i = eff_weights[i] / eff_total_w
                    adj_i_raw = adj_nm * w_i
                    cap = PER_LINK_CLIP * v_old
                    adj_i = min(max(adj_i_raw, -cap), cap)
                    if abs(adj_i) >= cap - 1e-12:
                        if adjust_side == 'rx':
                            clip_hit_rx[i] = True
                        else:
                            clip_hit_tx[i] = True
                    v_new = max(0.0, v_old + adj_i)
                    if adjust_side == 'rx':
                        prev = hardened_rx[i]
                        if prev > ZERO_EPS:
                            scl = (v_new / prev)
                            scaled_rx_factor[i] *= scl
                            if abs(scl - 1.0) >= 0.10:
                                clip_hit_rx[i] = True
                        hardened_rx[i] = v_new
                        relc = abs(adj_i) / max(1.0, abs(v_old))
                        conf_rx[i] = clamp01(conf_rx[i] * (1.0 - 0.6 * relc))
                    else:
                        prev = hardened_tx[i]
                        if prev > ZERO_EPS:
                            scl = (v_new / prev)
                            scaled_tx_factor[i] *= scl
                            if abs(scl - 1.0) >= 0.10:
                                clip_hit_tx[i] = True
                        hardened_tx[i] = v_new
                        relc = abs(adj_i) / max(1.0, abs(v_old))
                        conf_tx[i] = clamp01(conf_tx[i] * (1.0 - 0.6 * relc))
        else:
            # No dominant bundle: targeted per-interface corrections only over focus_set
            cap_per = DOMINANCE_CAP * focus_total_w
            eff_weights = {i: min(weights[i], cap_per) for i in focus_set}
            eff_total_w = max(1e-9, sum(eff_weights.values()))
            for i in focus_set:
                v_old = side_vals[i]
                w_i = eff_weights[i] / eff_total_w
                adj_i_raw = total_adjust * w_i
                cap = PER_LINK_CLIP * v_old
                adj_i = min(max(adj_i_raw, -cap), cap)
                if abs(adj_i) >= cap - 1e-12:
                    if adjust_side == 'rx':
                        clip_hit_rx[i] = True
                    else:
                        clip_hit_tx[i] = True
                v_new = max(0.0, v_old + adj_i)
                if adjust_side == 'rx':
                    prev = hardened_rx[i]
                    if prev > ZERO_EPS:
                        scl = (v_new / prev)
                        scaled_rx_factor[i] *= scl
                        if abs(scl - 1.0) >= 0.10:
                            clip_hit_rx[i] = True
                    hardened_rx[i] = v_new
                    relc = abs(adj_i) / max(1.0, abs(v_old))
                    conf_rx[i] = clamp01(conf_rx[i] * (1.0 - 0.6 * relc))
                else:
                    prev = hardened_tx[i]
                    if prev > ZERO_EPS:
                        scl = (v_new / prev)
                        scaled_tx_factor[i] *= scl
                        if abs(scl - 1.0) >= 0.10:
                            clip_hit_tx[i] = True
                    hardened_tx[i] = v_new
                    relc = abs(adj_i) / max(1.0, abs(v_old))
                    conf_tx[i] = clamp01(conf_tx[i] * (1.0 - 0.6 * relc))

    # Stage 2.5: Post-router soft-zero stabilization and residual computation for re-sync attenuation
    router_residual_mid: Dict[str, float] = {}
    for r, ifs in router_ifaces.items():
        ups = [i for i in ifs if status.get(i) == 'up']
        if not ups:
            router_residual_mid[r] = 0.0
            continue
        srx = sum(hardened_rx[i] for i in ups)
        stx = sum(hardened_tx[i] for i in ups)
        denomr = max(1.0, srx, stx)
        router_residual_mid[r] = abs(srx - stx) / denomr

    for a, b in link_pairs:
        if status.get(a) != 'up' or status.get(b) != 'up':
            continue
        if max(hardened_rx[a], hardened_tx[a], hardened_rx[b], hardened_tx[b]) < 2.0 * ZERO_THRESH:
            ra = local_router_of.get(a)
            rb = local_router_of.get(b)
            n_active_a = len([i for i in router_ifaces.get(ra, []) if status.get(i) == 'up'])
            n_active_b = len([i for i in router_ifaces.get(rb, []) if status.get(i) == 'up'])
            tau_ra = min(0.07, max(0.03, 0.05 * (2.0 / max(2, n_active_a)) ** 0.5))
            tau_rb = min(0.07, max(0.03, 0.05 * (2.0 / max(2, n_active_b)) ** 0.5))
            if router_residual_mid.get(ra, 0.0) <= tau_ra and router_residual_mid.get(rb, 0.0) <= tau_rb:
                hardened_rx[a] = hardened_tx[a] = hardened_rx[b] = hardened_tx[b] = 0.0
                conf_rx[a] = max(conf_rx[a], 0.95)
                conf_tx[a] = max(conf_tx[a], 0.95)
                conf_rx[b] = max(conf_rx[b], 0.95)
                conf_tx[b] = max(conf_tx[b], 0.95)

    # Stage 2.6: Conservation-preserving bundle finishing (median residual alignment across parallel links)
    bundle_map_links: Dict[Tuple[Any, Any], List[Tuple[str, str]]] = {}
    for a, b in link_pairs:
        if status.get(a) == 'up' and status.get(b) == 'up':
            ra = local_router_of.get(a)
            rb = local_router_of.get(b)
            bundle_map_links.setdefault((ra, rb), []).append((a, b))

    def _median(vals: List[float]) -> float:
        s = sorted(vals)
        n = len(s)
        if n == 0:
            return 0.0
        if n % 2 == 1:
            return s[n // 2]
        return 0.5 * (s[n // 2 - 1] + s[n // 2])

    for (ra, rb), pairs in bundle_map_links.items():
        if len(pairs) < 2:
            continue

        e_list: List[float] = []
        w_list: List[float] = []
        idx_list: List[int] = []
        rd_list: List[float] = []
        tau_list: List[float] = []
        caps: List[float] = []

        for idx, (a, b) in enumerate(pairs):
            v_tx = hardened_tx[a]
            v_rx_peer = hardened_rx[b]
            if max(v_tx, v_rx_peer) < ZERO_THRESH:
                continue
            e = v_tx - v_rx_peer
            cdir = 0.5 * (clamp01(conf_tx.get(a, 0.7)) + clamp01(conf_rx.get(b, 0.7)))
            w = (1.0 - cdir) * max(v_tx, ZERO_THRESH) + 1e-12
            e_list.append(e)
            w_list.append(w)
            idx_list.append(idx)
            rd_list.append(rel_diff(v_tx, v_rx_peer))
            tau_list.append(adaptive_tau(v_tx, v_rx_peer))
            caps.append(BUNDLE_FINISH_CLIP * max(v_tx, ZERO_THRESH))

        n = len(e_list)
        if n < 2:
            continue

        e_med = _median(e_list)
        residuals = [e - e_med for e in e_list]

        mean_rd = sum(rd_list) / n
        mean_tau = sum(tau_list) / n
        gap_norm = max(0.0, (mean_rd - mean_tau) / max(mean_tau, 1e-9))
        gamma = min(BUNDLE_FINISH_GAIN_MAX, 0.4 * gap_norm)
        if gamma <= 1e-12:
            continue

        deltas = []
        for r_val, cap in zip(residuals, caps):
            d = -gamma * r_val
            d = min(max(d, -cap), cap)
            deltas.append(d)

        sum_delta = sum(deltas)
        if abs(sum_delta) > 1e-9:
            free_idx = [k for k, (d, cap) in enumerate(zip(deltas, caps)) if abs(d) < cap - 1e-12]
            sw = sum(w_list[k] for k in free_idx)
            if sw > 0.0:
                for k in free_idx:
                    adj = sum_delta * (w_list[k] / sw)
                    nd = deltas[k] - adj
                    deltas[k] = min(max(nd, -caps[k]), caps[k])

        for local_idx, delta in enumerate(deltas):
            a, b = pairs[idx_list[local_idx]]
            if delta == 0.0:
                continue
            prev_tx = hardened_tx[a]
            prev_rx = hardened_rx[b]
            new_tx = max(0.0, prev_tx + delta)
            new_rx = max(0.0, prev_rx - delta)

            if prev_tx > ZERO_EPS:
                scaled_tx_factor[a] *= (new_tx / prev_tx)
            if prev_rx > ZERO_EPS:
                scaled_rx_factor[b] *= (new_rx / prev_rx)

            hardened_tx[a] = new_tx
            hardened_rx[b] = new_rx

            if prev_tx > ZERO_EPS:
                relc_tx = rel_diff(new_tx, prev_tx)
                conf_tx[a] = clamp01(conf_tx[a] * (1.0 - 0.2 * relc_tx))
            if prev_rx > ZERO_EPS:
                relc_rx = rel_diff(new_rx, prev_rx)
                conf_rx[b] = clamp01(conf_rx[b] * (1.0 - 0.2 * relc_rx))

    # Stage 3: Confidence-gap-proportional re-sync with scaling guard and router-imbalance attenuation
    def nudge_toward_mean(val_lo: float, val_hi: float, frac: float) -> float:
        target = 0.5 * (val_lo + val_hi)
        return val_lo + frac * (target - val_lo)

    for a, b in link_pairs:
        if status.get(a) != 'up' or status.get(b) != 'up':
            continue

        # Attenuation from local router imbalances
        ra = local_router_of.get(a)
        rb = local_router_of.get(b)
        att = clamp01(1.0 - max(router_residual_mid.get(ra, 0.0), router_residual_mid.get(rb, 0.0)))

        # Direction 1: a.tx vs b.rx
        a_tx, b_rx = hardened_tx[a], hardened_rx[b]
        if max(a_tx, b_rx) > ZERO_EPS:
            d1 = rel_diff(a_tx, b_rx)
            tau1 = adaptive_tau(a_tx, b_rx)
            if d1 > tau1:
                ca, cb = conf_tx.get(a, 0.6), conf_rx.get(b, 0.6)
                gap_norm = clamp01((d1 - tau1) / max(tau1, 1e-9))
                f_base = 0.4 * (1.0 / (1.0 + math.exp(-5.0 * (gap_norm - 0.5))))
                f_base *= att
                moved = False
                if ca >= cb and abs(scaled_rx_factor.get(b, 1.0) - 1.0) <= STRONG_SCALE_GUARD:
                    f = min(RESYNC_MAX_F, f_base)
                    if f > 0.0:
                        old = b_rx
                        target = a_tx
                        step = nudge_toward_mean(old, target, f)
                        cap_abs = RESYNC_PER_DIR_CAP * max(target, old, 1.0)
                        step = max(0.0, min(old + cap_abs, max(0.0, step)))
                        step = max(0.0, max(old - cap_abs, step))
                        new = step
                        hardened_rx[b] = new
                        relc = rel_diff(new, old)
                        conf_rx[b] = clamp01(conf_rx[b] * (1.0 - 0.3 * relc))
                        moved = True
                elif cb > ca and abs(scaled_tx_factor.get(a, 1.0) - 1.0) <= STRONG_SCALE_GUARD:
                    f = min(RESYNC_MAX_F, f_base)
                    if f > 0.0:
                        old = a_tx
                        target = b_rx
                        step = nudge_toward_mean(old, target, f)
                        cap_abs = RESYNC_PER_DIR_CAP * max(target, old, 1.0)
                        step = max(0.0, min(old + cap_abs, max(0.0, step)))
                        step = max(0.0, max(old - cap_abs, step))
                        new = step
                        hardened_tx[a] = new
                        relc = rel_diff(new, old)
                        conf_tx[a] = clamp01(conf_tx[a] * (1.0 - 0.3 * relc))
                        moved = True
                if not moved and ca < 0.70 and cb < 0.70:
                    ra = local_router_of.get(a); rb = local_router_of.get(b)
                    na = len([i for i in router_ifaces.get(ra, []) if status.get(i) == 'up'])
                    nb = len([i for i in router_ifaces.get(rb, []) if status.get(i) == 'up'])
                    tau_ra = min(0.07, max(0.03, 0.05 * (2.0 / max(2, na)) ** 0.5))
                    tau_rb = min(0.07, max(0.03, 0.05 * (2.0 / max(2, nb)) ** 0.5))
                    if router_residual_mid.get(ra, 0.0) <= tau_ra and router_residual_mid.get(rb, 0.0) <= tau_rb:
                        f_bi = min(0.10, 0.5 * gap_norm) * att
                        if f_bi > 0.0:
                            old_a = a_tx; old_b = b_rx
                            tgt = 0.5 * (old_a + old_b)
                            cap_a = RESYNC_PER_DIR_CAP * max(old_a, tgt, 1.0)
                            cap_b = RESYNC_PER_DIR_CAP * max(old_b, tgt, 1.0)
                            new_a = old_a + clamp01(f_bi) * (tgt - old_a)
                            new_b = old_b + clamp01(f_bi) * (tgt - old_b)
                            new_a = max(0.0, min(old_a + cap_a, max(old_a - cap_a, new_a)))
                            new_b = max(0.0, min(old_b + cap_b, max(old_b - cap_b, new_b)))
                            hardened_tx[a] = new_a
                            hardened_rx[b] = new_b
                            conf_tx[a] = clamp01(conf_tx[a] * (1.0 - 0.2 * rel_diff(new_a, old_a)))
                            conf_rx[b] = clamp01(conf_rx[b] * (1.0 - 0.2 * rel_diff(new_b, old_b)))

        # Direction 2: a.rx vs b.tx
        a_rx, b_tx = hardened_rx[a], hardened_tx[b]
        if max(a_rx, b_tx) > ZERO_EPS:
            d2 = rel_diff(a_rx, b_tx)
            tau2 = adaptive_tau(a_rx, b_tx)
            if d2 > tau2:
                ca, cb = conf_rx.get(a, 0.6), conf_tx.get(b, 0.6)
                gap_norm = clamp01((d2 - tau2) / max(tau2, 1e-9))
                f_base = 0.4 * (1.0 / (1.0 + math.exp(-5.0 * (gap_norm - 0.5))))
                f_base *= att
                moved = False
                if ca >= cb and abs(scaled_tx_factor.get(b, 1.0) - 1.0) <= STRONG_SCALE_GUARD:
                    f = min(RESYNC_MAX_F, f_base)
                    if f > 0.0:
                        old = b_tx
                        target = a_rx
                        step = nudge_toward_mean(old, target, f)
                        cap_abs = RESYNC_PER_DIR_CAP * max(target, old, 1.0)
                        step = max(0.0, min(old + cap_abs, max(0.0, step)))
                        step = max(0.0, max(old - cap_abs, step))
                        new = step
                        hardened_tx[b] = new
                        relc = rel_diff(new, old)
                        conf_tx[b] = clamp01(conf_tx[b] * (1.0 - 0.3 * relc))
                        moved = True
                elif cb > ca and abs(scaled_rx_factor.get(a, 1.0) - 1.0) <= STRONG_SCALE_GUARD:
                    f = min(RESYNC_MAX_F, f_base)
                    if f > 0.0:
                        old = a_rx
                        target = b_tx
                        step = nudge_toward_mean(old, target, f)
                        cap_abs = RESYNC_PER_DIR_CAP * max(target, old, 1.0)
                        step = max(0.0, min(old + cap_abs, max(0.0, step)))
                        step = max(0.0, max(old - cap_abs, step))
                        new = step
                        hardened_rx[a] = new
                        relc = rel_diff(new, old)
                        conf_rx[a] = clamp01(conf_rx[a] * (1.0 - 0.3 * relc))
                        moved = True
                if not moved and ca < 0.70 and cb < 0.70:
                    ra = local_router_of.get(a); rb = local_router_of.get(b)
                    na = len([i for i in router_ifaces.get(ra, []) if status.get(i) == 'up'])
                    nb = len([i for i in router_ifaces.get(rb, []) if status.get(i) == 'up'])
                    tau_ra = min(0.07, max(0.03, 0.05 * (2.0 / max(2, na)) ** 0.5))
                    tau_rb = min(0.07, max(0.03, 0.05 * (2.0 / max(2, nb)) ** 0.5))
                    if router_residual_mid.get(ra, 0.0) <= tau_ra and router_residual_mid.get(rb, 0.0) <= tau_rb:
                        f_bi = min(0.10, 0.5 * gap_norm) * att
                        if f_bi > 0.0:
                            old_a = a_rx; old_b = b_tx
                            tgt = 0.5 * (old_a + old_b)
                            cap_a = RESYNC_PER_DIR_CAP * max(old_a, tgt, 1.0)
                            cap_b = RESYNC_PER_DIR_CAP * max(old_b, tgt, 1.0)
                            new_a = old_a + clamp01(f_bi) * (tgt - old_a)
                            new_b = old_b + clamp01(f_bi) * (tgt - old_b)
                            new_a = max(0.0, min(old_a + cap_a, max(old_a - cap_a, new_a)))
                            new_b = max(0.0, min(old_b + cap_b, max(old_b - cap_b, new_b)))
                            hardened_rx[a] = new_a
                            hardened_tx[b] = new_b
                            conf_rx[a] = clamp01(conf_rx[a] * (1.0 - 0.2 * rel_diff(new_a, old_a)))
                            conf_tx[b] = clamp01(conf_tx[b] * (1.0 - 0.2 * rel_diff(new_b, old_b)))

    # Enforce status down => zero as a final safety
    for i in telemetry:
        if status.get(i) != 'up':
            hardened_rx[i] = 0.0
            hardened_tx[i] = 0.0
            conf_rx[i] = max(conf_rx[i], 0.85)
            conf_tx[i] = max(conf_tx[i], 0.85)

    # Compute router residuals after all adjustments (for confidence calibration)
    router_residual: Dict[str, float] = {}
    for r, ifs in router_ifaces.items():
        ups = [i for i in ifs if status.get(i) == 'up']
        if not ups:
            router_residual[r] = 0.0
            continue
        sum_rx = sum(hardened_rx[i] for i in ups)
        sum_tx = sum(hardened_tx[i] for i in ups)
        denom = max(1.0, sum_rx, sum_tx)
        router_residual[r] = abs(sum_rx - sum_tx) / denom

    # Stage 4: Confidence calibration enriched with stability, bundle-consistency, scale penalty, and router/peer smoothing
    # Pre-compute router totals per direction for stability term
    router_sum_rx: Dict[str, float] = {}
    router_sum_tx: Dict[str, float] = {}
    for r, ifs in router_ifaces.items():
        ups = [i for i in ifs if status.get(i) == 'up']
        router_sum_rx[r] = sum(hardened_rx.get(i, 0.0) for i in ups)
        router_sum_tx[r] = sum(hardened_tx.get(i, 0.0) for i in ups)

    # Pre-compute bundle median residuals for bundle-consistency
    bundle_e_map_tx: Dict[Tuple[Any, Any], List[float]] = {}
    bundle_e_map_rx: Dict[Tuple[Any, Any], List[float]] = {}
    for a, b in link_pairs:
        if status.get(a) != 'up' or status.get(b) != 'up':
            continue
        ra = local_router_of.get(a)
        rb = local_router_of.get(b)
        e1 = hardened_tx[a] - hardened_rx[b]
        e2 = hardened_tx[b] - hardened_rx[a]
        bundle_e_map_tx.setdefault((ra, rb), []).append(e1)
        bundle_e_map_rx.setdefault((rb, ra), []).append(e2)

    def _median_local(vals: List[float]) -> float:
        s = sorted(vals)
        n = len(s)
        if n == 0:
            return 0.0
        if n % 2 == 1:
            return s[n // 2]
        return 0.5 * (s[n // 2 - 1] + s[n // 2])

    bundle_e_med_tx: Dict[Tuple[Any, Any], float] = {k: _median_local(v) for k, v in bundle_e_map_tx.items()}
    bundle_e_med_rx: Dict[Tuple[Any, Any], float] = {k: _median_local(v) for k, v in bundle_e_map_rx.items()}

    def compute_conf(i: str) -> Tuple[float, float]:
        p = peer_of.get(i)
        # Measurement residuals
        r_meas_rx = rel_diff(hardened_rx[i], orig_rx[i])
        r_meas_tx = rel_diff(hardened_tx[i], orig_tx[i])
        # Link residuals
        if p and p in telemetry and status.get(i) == 'up' and status.get(p) == 'up':
            r_link_rx = rel_diff(hardened_rx[i], hardened_tx[p])
            r_link_tx = rel_diff(hardened_tx[i], hardened_rx[p])
        else:
            r_link_rx = 0.2
            r_link_tx = 0.2
        # Router residual
        rtr = router_residual.get(local_router_of.get(i), 0.0)
        # Base confidence blend (slightly more weight to router residual than before)
        base_rx = 1.0 - (0.53 * r_meas_rx + 0.33 * r_link_rx + 0.14 * rtr)
        base_tx = 1.0 - (0.53 * r_meas_tx + 0.33 * r_link_tx + 0.14 * rtr)
        base_rx = clamp01(base_rx)
        base_tx = clamp01(base_tx)

        # Scale-factor term (penalize big routed adjustments)
        alpha_rx = abs(scaled_rx_factor.get(i, 1.0) - 1.0)
        alpha_tx = abs(scaled_tx_factor.get(i, 1.0) - 1.0)
        scale_term_rx = clamp01(1.0 - min(0.5, alpha_rx))
        scale_term_tx = clamp01(1.0 - min(0.5, alpha_tx))

        c_rx = clamp01(0.90 * base_rx + 0.10 * scale_term_rx)
        c_tx = clamp01(0.90 * base_tx + 0.10 * scale_term_tx)

        # Stability term: interfaces dominating router totals are less reliable
        r_id = local_router_of.get(i)
        sum_r_rx = max(1.0, router_sum_rx.get(r_id, 1.0))
        sum_r_tx = max(1.0, router_sum_tx.get(r_id, 1.0))
        share_rx = 0.0 if sum_r_rx <= 0 else hardened_rx[i] / sum_r_rx
        share_tx = 0.0 if sum_r_tx <= 0 else hardened_tx[i] / sum_r_tx
        stab_rx = clamp01(1.0 - 0.5 * share_rx)
        stab_tx = clamp01(1.0 - 0.5 * share_tx)
        c_rx = clamp01(c_rx + 0.05 * stab_rx)
        c_tx = clamp01(c_tx + 0.05 * stab_tx)

        # Bundle-consistency term: closeness to bundle median residual
        if p and p in telemetry and status.get(i) == 'up' and status.get(p) == 'up':
            lr = local_router_of.get(i)
            rr = local_router_of.get(p)
            e_tx = hardened_tx[i] - hardened_rx[p]
            e_rx = hardened_rx[i] - hardened_tx[p]
            med_tx = bundle_e_med_tx.get((lr, rr), 0.0)
            med_rx = bundle_e_med_rx.get((lr, rr), 0.0)
            bcons_tx = clamp01(1.0 - abs(e_tx - med_tx) / (abs(med_tx) + max(hardened_tx[i], 1.0)))
            bcons_rx = clamp01(1.0 - abs(e_rx - med_rx) / (abs(med_rx) + max(hardened_rx[i], 1.0)))
            c_tx = clamp01(c_tx + 0.06 * bcons_tx)
            c_rx = clamp01(c_rx + 0.06 * bcons_rx)

        # If interface is down, zero is a strong invariant; raise confidence floor
        if status.get(i) != 'up':
            c_rx = max(c_rx, 0.85)
            c_tx = max(c_tx, 0.85)
        return c_rx, c_tx

    # Compute confidences (preliminary)
    for i in telemetry:
        cr, ct = compute_conf(i)
        conf_rx[i], conf_tx[i] = cr, ct

    # Apply penalties for clip/strong scaling and if router imbalance worsened
    for i in telemetry:
        r_id = local_router_of.get(i)
        if abs(scaled_rx_factor.get(i, 1.0) - 1.0) >= 0.10 or clip_hit_rx.get(i, False):
            conf_rx[i] = clamp01(conf_rx[i] * CLIP_HIT_PENALTY)
        if abs(scaled_tx_factor.get(i, 1.0) - 1.0) >= 0.10 or clip_hit_tx.get(i, False):
            conf_tx[i] = clamp01(conf_tx[i] * CLIP_HIT_PENALTY)
        # extra small penalty when both clipping and router got worse (mid -> final)
        if router_residual.get(r_id, 0.0) > router_residual_mid.get(r_id, 0.0):
            if clip_hit_rx.get(i, False):
                conf_rx[i] = clamp01(conf_rx[i] * 0.97)
            if clip_hit_tx.get(i, False):
                conf_tx[i] = clamp01(conf_tx[i] * 0.97)

    # Router-aware smoothing when |scale - 1| < 0.05
    router_mean_conf_rx: Dict[str, float] = {}
    router_mean_conf_tx: Dict[str, float] = {}
    for r, ifs in router_ifaces.items():
        ups = [i for i in ifs if i in telemetry and status.get(i) == 'up']
        if ups:
            router_mean_conf_rx[r] = sum(conf_rx[i] for i in ups) / len(ups)
            router_mean_conf_tx[r] = sum(conf_tx[i] for i in ups) / len(ups)
    for i in telemetry:
        r = local_router_of.get(i)
        if abs(scaled_rx_factor.get(i, 1.0) - 1.0) < 0.05 and r in router_mean_conf_rx:
            conf_rx[i] = clamp01(0.85 * conf_rx[i] + 0.15 * router_mean_conf_rx[r])
        if abs(scaled_tx_factor.get(i, 1.0) - 1.0) < 0.05 and r in router_mean_conf_tx:
            conf_tx[i] = clamp01(0.85 * conf_tx[i] + 0.15 * router_mean_conf_tx[r])

    # Untouched boost when minimal change (<1%) and good final symmetry on link
    for i in telemetry:
        p = peer_of.get(i)
        if p and p in telemetry and status.get(i) == 'up' and status.get(p) == 'up':
            if rel_diff(hardened_rx[i], orig_rx[i]) < 0.01:
                if rel_diff(hardened_rx[i], hardened_tx[p]) <= adaptive_tau(hardened_rx[i], hardened_tx[p]):
                    conf_rx[i] = min(0.98, conf_rx[i] + UNTOUCHED_BOOST)
            if rel_diff(hardened_tx[i], orig_tx[i]) < 0.01:
                if rel_diff(hardened_tx[i], hardened_rx[p]) <= adaptive_tau(hardened_tx[i], hardened_rx[p]):
                    conf_tx[i] = min(0.98, conf_tx[i] + UNTOUCHED_BOOST)

    # Peer smoothing (order-independent via staged update)
    new_conf_rx = dict(conf_rx)
    new_conf_tx = dict(conf_tx)
    for a, b in link_pairs:
        if status.get(a) == 'up' and status.get(b) == 'up':
            new_conf_tx[a] = clamp01((1.0 - PEER_SMOOTH) * conf_tx[a] + PEER_SMOOTH * conf_rx[b])
            new_conf_rx[b] = clamp01((1.0 - PEER_SMOOTH) * conf_rx[b] + PEER_SMOOTH * conf_tx[a])
            new_conf_rx[a] = clamp01((1.0 - PEER_SMOOTH) * conf_rx[a] + PEER_SMOOTH * conf_tx[b])
            new_conf_tx[b] = clamp01((1.0 - PEER_SMOOTH) * conf_tx[b] + PEER_SMOOTH * conf_rx[a])
    conf_rx, conf_tx = new_conf_rx, new_conf_tx

    # Assemble final result
    result: Dict[str, Dict[str, Tuple]] = {}
    for i, data in telemetry.items():
        my_status = status.get(i, 'unknown')
        peer_id = peer_of.get(i)
        # Status confidence: do not flip, but penalize inconsistent peer or down+nonzero readings
        status_conf = 1.0
        if peer_id and peer_id in telemetry:
            if telemetry[peer_id].get('interface_status', 'unknown') != my_status:
                status_conf = 0.6
        if my_status == 'down' and (orig_rx.get(i, 0.0) > ZERO_EPS or orig_tx.get(i, 0.0) > ZERO_EPS):
            status_conf = min(status_conf, 0.6)

        repaired: Dict[str, Any] = {}
        repaired['rx_rate'] = (orig_rx.get(i, 0.0), max(0.0, hardened_rx.get(i, 0.0)), clamp01(conf_rx.get(i, 0.6)))
        repaired['tx_rate'] = (orig_tx.get(i, 0.0), max(0.0, hardened_tx.get(i, 0.0)), clamp01(conf_tx.get(i, 0.6)))
        repaired['interface_status'] = (my_status, my_status, status_conf)

        # Copy metadata unchanged
        repaired['connected_to'] = data.get('connected_to')
        repaired['local_router'] = data.get('local_router')
        repaired['remote_router'] = data.get('remote_router')

        result[i] = repaired

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