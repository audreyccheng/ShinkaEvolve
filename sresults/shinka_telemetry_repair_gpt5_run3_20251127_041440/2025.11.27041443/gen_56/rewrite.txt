# EVOLVE-BLOCK-START
"""
Bundle-aware consensus projection for network telemetry repair (v2).

Enhancements:
- Adaptive, peer-biased fusion for large mismatches and early soft-zero pre-snap.
- Expected-penalty router side selection with softened weights (rate^0.95).
- Dominance-aware scaling and bundle-dominant group scaling with alpha_eff.
- Micro high-confidence finishing tier.
- Tighter confidence calibration with strong-scale guards and untouched boosts.

Maintains inputs/outputs of the original function.
"""
from typing import Dict, Any, Tuple, List


def repair_network_telemetry(telemetry: Dict[str, Dict[str, Any]],
                             topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:
    # Hyperparameters
    TAU_H_BASE = 0.02          # ~2% hardening threshold
    TAU_H_HIGH = 0.015         # tighter for high-rate pairs
    TAU_H_LOW = 0.03           # looser for near-zero
    ZERO_EPS = 1e-6
    ZERO_THRESH = 1.0          # Mbps near-zero threshold

    # Router projection
    DAMP_ROUTER = 0.60
    PER_LINK_CLIP = 0.10       # ±10%
    BUNDLE_CLIP = 0.15         # ±15%
    RATE_EXP = 0.95            # soften dominance via rate exponent
    WEIGHT_FOCUS = 0.70        # focus lowest-confidence 70% capacity
    DOMINANCE_CAP = 0.50       # ≤50% share in a pass
    BUNDLE_DOM_FRAC = 0.60     # bundle dominance threshold (≥60% of side traffic)

    # Re-sync / finishing
    STRONG_SCALE_GUARD = 0.08  # guard for re-sync skipping
    RESYNC_MAX_F = 0.40        # max nudge toward mean
    MICRO_DAMP = 0.25          # micro finishing damping
    MICRO_CLIP_REL = 0.03      # ±3% micro clip per-var

    # Confidence smoothing/penalties
    PEER_SMOOTH = 0.10         # 10% peer smoothing
    CLIP_HIT_PENALTY = 0.95    # clip/strong-scale penalty
    STRONG_PENALTY = 0.97      # explicit strong-scale penalty (>8%)
    UNTOUCHED_BOOST = 0.02     # small boost for untouched well-synced

    def clamp01(x: float) -> float:
        return max(0.0, min(1.0, x))

    def rel_diff(a: float, b: float) -> float:
        return abs(a - b) / max(1.0, abs(a), abs(b))

    def adaptive_tau(v1: float, v2: float) -> float:
        if v1 >= 100.0 and v2 >= 100.0:
            return TAU_H_HIGH
        if v1 < ZERO_THRESH or v2 < ZERO_THRESH:
            return TAU_H_LOW
        return TAU_H_BASE

    def router_tau(n_active: int) -> float:
        return min(0.07, max(0.03, 0.05 * (2.0 / max(2, n_active)) ** 0.5))

    # Collect originals and maps
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

    # Router->interfaces mapping, prefer provided topology
    router_ifaces: Dict[str, List[str]] = {}
    if topology:
        for r, ifs in topology.items():
            router_ifaces[r] = [i for i in ifs if i in telemetry]
    else:
        # Fallback to local_router fields (still useful to enforce R1)
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

    # Build unique link pairs
    link_pairs: List[Tuple[str, str]] = []
    seen = set()
    for a in telemetry:
        b = peer_of.get(a)
        if not b or b not in telemetry or a == b:
            continue
        if (b, a) in seen:
            continue
        seen.add((a, b))
        link_pairs.append((a, b))

    # Precompute pre-router residuals from original for early soft-zero pre-snap
    def compute_router_residuals_from(vals_rx: Dict[str, float], vals_tx: Dict[str, float]) -> Dict[str, float]:
        res: Dict[str, float] = {}
        for r, ifs in router_ifaces.items():
            ups = [i for i in ifs if status.get(i) == 'up']
            if not ups:
                res[r] = 0.0
                continue
            srx = sum(vals_rx.get(i, 0.0) for i in ups)
            stx = sum(vals_tx.get(i, 0.0) for i in ups)
            denom = max(1.0, srx, stx)
            res[r] = abs(srx - stx) / denom
        return res

    pre_router_resid = compute_router_residuals_from(orig_rx, orig_tx)

    # Initialize hardened with originals (non-negative)
    hardened_rx: Dict[str, float] = {i: max(0.0, orig_rx[i]) for i in telemetry}
    hardened_tx: Dict[str, float] = {i: max(0.0, orig_tx[i]) for i in telemetry}
    conf_rx: Dict[str, float] = {i: 0.7 for i in telemetry}
    conf_tx: Dict[str, float] = {i: 0.7 for i in telemetry}

    # Track cumulative scaling and clip hits per direction
    scaled_rx_factor: Dict[str, float] = {i: 1.0 for i in telemetry}
    scaled_tx_factor: Dict[str, float] = {i: 1.0 for i in telemetry}
    clip_hit_rx: Dict[str, bool] = {i: False for i in telemetry}
    clip_hit_tx: Dict[str, bool] = {i: False for i in telemetry}

    # Helper: adaptive peer-biased fusion for a direction
    def fuse_direction(local_val: float, peer_val: float, local_up: bool, peer_up: bool) -> Tuple[float, float]:
        # Returns fused value and confidence contribution
        if not local_up or not peer_up:
            return 0.0, 0.85
        d = rel_diff(local_val, peer_val)
        tau = adaptive_tau(local_val, peer_val)
        if d <= tau:
            fused = 0.5 * (local_val + peer_val)
            c = clamp01(0.9 + 0.1 * (1.0 - d / max(tau, 1e-12)))
            return fused, c
        # Adaptive snap-to-peer bias beta in [0.7, 0.9]
        # Stronger bias if mismatch >10%, or local near-zero and peer non-zero
        near_local_zero = (local_val < ZERO_THRESH and peer_val >= ZERO_THRESH)
        beta_base = 0.7
        beta = beta_base + 0.2 * clamp01((d - 0.10) / 0.20) + (0.1 if near_local_zero else 0.0)
        beta = max(0.7, min(0.9, beta))
        fused = (1.0 - beta) * local_val + beta * peer_val
        c = clamp01(1.0 - d)
        return fused, c

    # Stage 1: Link hardening with adaptive tolerance, early soft-zero, and peer-biased fusion
    for a, b in link_pairs:
        a_up = (status.get(a) == 'up')
        b_up = (status.get(b) == 'up')
        a_rx0, a_tx0 = orig_rx[a], orig_tx[a]
        b_rx0, b_tx0 = orig_rx[b], orig_tx[b]

        # Early soft-zero pre-snap when both routers balanced and link is tiny
        ra = local_router_of.get(a)
        rb = local_router_of.get(b)
        n_active_a = len([i for i in router_ifaces.get(ra, []) if status.get(i) == 'up'])
        n_active_b = len([i for i in router_ifaces.get(rb, []) if status.get(i) == 'up'])
        tau_ra = router_tau(n_active_a)
        tau_rb = router_tau(n_active_b)
        if max(a_rx0, a_tx0, b_rx0, b_tx0) < 1.5 * ZERO_THRESH and pre_router_resid.get(ra, 0.0) <= tau_ra and pre_router_resid.get(rb, 0.0) <= tau_rb:
            hardened_rx[a] = hardened_tx[a] = hardened_rx[b] = hardened_tx[b] = 0.0
            conf_rx[a] = max(conf_rx[a], 0.95)
            conf_tx[a] = max(conf_tx[a], 0.95)
            conf_rx[b] = max(conf_rx[b], 0.95)
            conf_tx[b] = max(conf_tx[b], 0.95)
            continue

        # If either side is down: strong invariant => zero on both ends of the link
        if not a_up or not b_up:
            hardened_rx[a] = hardened_tx[a] = 0.0
            hardened_rx[b] = hardened_tx[b] = 0.0
            conf_rx[a] = max(conf_rx[a], 0.85)
            conf_tx[a] = max(conf_tx[a], 0.85)
            conf_rx[b] = max(conf_rx[b], 0.85)
            conf_tx[b] = max(conf_tx[b], 0.85)
            continue

        # Soft-zero stabilization for near-zero links (post-invariant)
        if max(a_rx0, a_tx0, b_rx0, b_tx0) < 2.0 * ZERO_THRESH:
            hardened_rx[a] = hardened_tx[a] = 0.0
            hardened_rx[b] = hardened_tx[b] = 0.0
            conf_rx[a] = max(conf_rx[a], 0.95)
            conf_tx[a] = max(conf_tx[a], 0.95)
            conf_rx[b] = max(conf_rx[b], 0.95)
            conf_tx[b] = max(conf_tx[b], 0.95)
            continue

        # Direction 1: a.tx vs b.rx
        v1, c1 = fuse_direction(a_tx0, b_rx0, a_up, b_up)
        hardened_tx[a] = max(0.0, v1)
        hardened_rx[b] = max(0.0, v1)
        conf_tx[a] = max(conf_tx[a], c1)
        conf_rx[b] = max(conf_rx[b], c1)

        # Direction 2: a.rx vs b.tx
        v2, c2 = fuse_direction(a_rx0, b_tx0, a_up, b_up)
        hardened_rx[a] = max(0.0, v2)
        hardened_tx[b] = max(0.0, v2)
        conf_rx[a] = max(conf_rx[a], c2)
        conf_tx[b] = max(conf_tx[b], c2)

    # Unpaired interfaces: trust local; zero if down
    in_pairs = {x for ab in link_pairs for x in ab}
    for i in telemetry:
        if i not in in_pairs:
            if status.get(i) != 'up':
                hardened_rx[i] = hardened_tx[i] = 0.0
                conf_rx[i] = max(conf_rx[i], 0.85)
                conf_tx[i] = max(conf_tx[i], 0.85)
            else:
                hardened_rx[i] = max(0.0, orig_rx[i])
                hardened_tx[i] = max(0.0, orig_tx[i])
                conf_rx[i] = max(conf_rx[i], 0.6)
                conf_tx[i] = max(conf_tx[i], 0.6)

    # Stage 2: Router flow conservation with bundle-dominance and expected-penalty side selection
    for r, ifs in router_ifaces.items():
        up_ifs = [i for i in ifs if status.get(i) == 'up']
        if len(up_ifs) < 2:
            continue

        sum_rx = sum(hardened_rx[i] for i in up_ifs)
        sum_tx = sum(hardened_tx[i] for i in up_ifs)
        denom = max(1.0, sum_rx, sum_tx)
        imbalance = sum_rx - sum_tx
        rel_gap = abs(imbalance) / denom
        tau_router = router_tau(len(up_ifs))
        if rel_gap <= tau_router:
            continue

        # Helper: build side maps
        def side_maps(side: str):
            vals = {i: (hardened_rx[i] if side == 'rx' else hardened_tx[i]) for i in up_ifs}
            confs = {i: (conf_rx[i] if side == 'rx' else conf_tx[i]) for i in up_ifs}
            # Weights: w_i = (1 - conf) * max(val, ZERO_THRESH)^RATE_EXP
            weights = {i: (1.0 - clamp01(confs[i])) * (max(vals[i], ZERO_THRESH) ** RATE_EXP) + 1e-12 for i in up_ifs}
            total_w = sum(weights.values())
            # Focus set covering WEIGHT_FOCUS of total weight
            sorted_ifs = sorted(up_ifs, key=lambda x: weights[x], reverse=True)
            focus_set: List[str] = []
            acc = 0.0
            for it in sorted_ifs:
                if acc / max(total_w, 1e-12) >= WEIGHT_FOCUS:
                    break
                focus_set.append(it)
                acc += weights[it]
            if not focus_set:
                focus_set = list(up_ifs)
                acc = total_w
            focus_total_w = max(acc, 1e-9)
            return vals, confs, weights, total_w, focus_set, focus_total_w

        # Expected-penalty lookahead: choose side with lower predicted confidence loss
        def estimate_penalty(side: str) -> float:
            vals, confs, weights, total_w, focus_set, focus_total_w = side_maps(side)
            total_adjust = (-imbalance if side == 'rx' else imbalance) * DAMP_ROUTER
            if focus_total_w <= 0 or abs(total_adjust) < ZERO_EPS:
                return 0.0
            # Huber-like capped weights: w_i <= 0.5 * sum
            cap_per = DOMINANCE_CAP * focus_total_w
            eff_weights = {i: min(weights[i], cap_per) for i in focus_set}
            eff_total = max(1e-9, sum(eff_weights.values()))
            penalty = 0.0
            for i in focus_set:
                v_old = vals[i]
                w_i = eff_weights[i] / eff_total
                adj_raw = total_adjust * w_i
                cap = PER_LINK_CLIP * max(v_old, ZERO_THRESH)
                adj = max(-cap, min(cap, adj_raw))
                rel_change = abs(adj) / max(1.0, abs(v_old))
                c = clamp01(confs[i])
                # Penalty: higher for changing high-confidence counters
                penalty += rel_change * (0.5 + 0.5 * c)
            return penalty

        pen_rx = estimate_penalty('rx')
        pen_tx = estimate_penalty('tx')
        adjust_side = 'rx' if pen_rx <= pen_tx else 'tx'
        total_adjust = (-imbalance if adjust_side == 'rx' else imbalance) * DAMP_ROUTER

        # Build maps for chosen side
        side_vals, side_confs, side_weights, side_total_w, focus_set, focus_total_w = side_maps(adjust_side)
        # Bundle detection by (local_router, remote_router)
        bundles: Dict[Tuple[Any, Any], List[str]] = {}
        for i in up_ifs:
            key = (local_router_of.get(i), remote_router_of.get(i))
            bundles.setdefault(key, []).append(i)
        side_total = sum(side_vals.values()) if up_ifs else 0.0

        # Dominant bundles (≥60% of side traffic)
        dominant_bundles = []
        for key, members in bundles.items():
            s = sum(side_vals[m] for m in members)
            if side_total > ZERO_EPS and (s / side_total) >= BUNDLE_DOM_FRAC:
                dominant_bundles.append((key, members, s))

        # Huber-like weight cap for distribution
        cap_per = DOMINANCE_CAP * focus_total_w
        eff_weights = {i: min(side_weights[i], cap_per) for i in focus_set}
        eff_total_w = max(1e-9, sum(eff_weights.values()))

        used_adjust = 0.0

        # Dominant bundle shared scaling
        for key, members, s_sum in dominant_bundles:
            # Allocate bundle share proportional to its eff weight within focus set
            members_focus = [m for m in members if m in focus_set]
            if not members_focus or s_sum <= ZERO_EPS or eff_total_w <= 0:
                continue
            w_g = sum(eff_weights.get(m, 0.0) for m in members_focus)
            if w_g <= 0.0:
                continue
            adj_g = total_adjust * (w_g / eff_total_w)
            used_adjust += adj_g
            target_sum = max(0.0, s_sum + adj_g)
            raw_ratio = target_sum / max(s_sum, 1e-12)
            clipped_ratio = max(1.0 - BUNDLE_CLIP, min(1.0 + BUNDLE_CLIP, raw_ratio))
            alpha_eff = 1.0 + 0.6 * (clipped_ratio - 1.0)
            for m in members:
                old = side_vals[m]
                new = max(0.0, alpha_eff * old)
                if adjust_side == 'rx':
                    prev = hardened_rx[m]
                    if prev > ZERO_EPS:
                        scl = new / prev
                        scaled_rx_factor[m] *= scl
                        if abs(scl - 1.0) >= 0.10:
                            clip_hit_rx[m] = True
                    hardened_rx[m] = new
                    relc = abs(new - old) / max(1.0, abs(old))
                    conf_rx[m] = clamp01(conf_rx[m] * (1.0 - 0.6 * relc))
                else:
                    prev = hardened_tx[m]
                    if prev > ZERO_EPS:
                        scl = new / prev
                        scaled_tx_factor[m] *= scl
                        if abs(scl - 1.0) >= 0.10:
                            clip_hit_tx[m] = True
                    hardened_tx[m] = new
                    relc = abs(new - old) / max(1.0, abs(old))
                    conf_tx[m] = clamp01(conf_tx[m] * (1.0 - 0.6 * relc))
                # Update side_vals to reflect applied change for residual handling
                side_vals[m] = new

        # Distribute remaining adjustment to non-dominant links on focus set
        residual_adjust = total_adjust - used_adjust
        if abs(residual_adjust) > ZERO_EPS and eff_total_w > 0:
            # Eligible: focus_set excluding members of dominant bundles
            dominant_members = {m for _, mems, _ in dominant_bundles for m in mems}
            eligible = [i for i in focus_set if i not in dominant_members]
            if eligible:
                eff_total_e = max(1e-9, sum(eff_weights.get(i, 0.0) for i in eligible))
                for i in eligible:
                    v_old = side_vals[i]
                    w_i = eff_weights.get(i, 0.0) / eff_total_e
                    adj_raw = residual_adjust * w_i
                    cap = PER_LINK_CLIP * max(v_old, ZERO_THRESH)
                    adj = max(-cap, min(cap, adj_raw))
                    v_new = max(0.0, v_old + adj)
                    if abs(adj) >= cap - 1e-12:
                        if adjust_side == 'rx':
                            clip_hit_rx[i] = True
                        else:
                            clip_hit_tx[i] = True
                    if adjust_side == 'rx':
                        prev = hardened_rx[i]
                        if prev > ZERO_EPS:
                            scl = v_new / prev
                            scaled_rx_factor[i] *= scl
                            if abs(scl - 1.0) >= 0.10:
                                clip_hit_rx[i] = True
                        hardened_rx[i] = v_new
                        relc = abs(adj) / max(1.0, abs(v_old))
                        conf_rx[i] = clamp01(conf_rx[i] * (1.0 - 0.6 * relc))
                    else:
                        prev = hardened_tx[i]
                        if prev > ZERO_EPS:
                            scl = v_new / prev
                            scaled_tx_factor[i] *= scl
                            if abs(scl - 1.0) >= 0.10:
                                clip_hit_tx[i] = True
                        hardened_tx[i] = v_new
                        relc = abs(adj) / max(1.0, abs(v_old))
                        conf_tx[i] = clamp01(conf_tx[i] * (1.0 - 0.6 * relc))
                    side_vals[i] = v_new

        # Micro finishing tier if residual remains stubborn
        # Recompute residual for this router
        sum_rx2 = sum(hardened_rx[i] for i in up_ifs)
        sum_tx2 = sum(hardened_tx[i] for i in up_ifs)
        denom2 = max(1.0, sum_rx2, sum_tx2)
        imbalance2 = sum_rx2 - sum_tx2
        rel_gap2 = abs(imbalance2) / denom2
        if rel_gap2 > 0.6 * tau_router:
            # Choose side again by expected penalty
            pen_rx2 = estimate_penalty('rx')
            pen_tx2 = estimate_penalty('tx')
            side_micro = 'rx' if pen_rx2 <= pen_tx2 else 'tx'
            total_adjust_micro = (-imbalance2 if side_micro == 'rx' else imbalance2) * MICRO_DAMP

            # High-confidence set (>=0.85), fallback to all if empty
            if side_micro == 'rx':
                vals = {i: hardened_rx[i] for i in up_ifs}
                confs = {i: conf_rx[i] for i in up_ifs}
            else:
                vals = {i: hardened_tx[i] for i in up_ifs}
                confs = {i: conf_tx[i] for i in up_ifs}
            hi = [i for i in up_ifs if confs[i] >= 0.85]
            elig = hi if hi else list(up_ifs)
            # Weights simple magnitude based to spread tiny changes
            w_map = {i: max(vals[i], ZERO_THRESH) for i in elig}
            totw = max(1e-9, sum(w_map.values()))
            for i in elig:
                v_old = vals[i]
                w_i = w_map[i] / totw
                adj_raw = total_adjust_micro * w_i
                cap = MICRO_CLIP_REL * max(v_old, ZERO_THRESH)
                adj = max(-cap, min(cap, adj_raw))
                v_new = max(0.0, v_old + adj)
                if side_micro == 'rx':
                    prev = hardened_rx[i]
                    if prev > ZERO_EPS:
                        scl = v_new / prev
                        scaled_rx_factor[i] *= scl
                        if abs(scl - 1.0) >= 0.08:
                            clip_hit_rx[i] = True
                    hardened_rx[i] = v_new
                    relc = abs(adj) / max(1.0, abs(v_old))
                    conf_rx[i] = clamp01(conf_rx[i] * (1.0 - 0.25 * relc))
                else:
                    prev = hardened_tx[i]
                    if prev > ZERO_EPS:
                        scl = v_new / prev
                        scaled_tx_factor[i] *= scl
                        if abs(scl - 1.0) >= 0.08:
                            clip_hit_tx[i] = True
                    hardened_tx[i] = v_new
                    relc = abs(adj) / max(1.0, abs(v_old))
                    conf_tx[i] = clamp01(conf_tx[i] * (1.0 - 0.25 * relc))

    # Stage 2.5: Post-router soft-zero stabilization and residuals for re-sync attenuation
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

    # Soft-zero stabilization: if link tiny and adjacent routers near balanced, snap to zero
    for a, b in link_pairs:
        if status.get(a) != 'up' or status.get(b) != 'up':
            continue
        max_link = max(hardened_rx[a], hardened_tx[a], hardened_rx[b], hardened_tx[b])
        if max_link < 2.0 * ZERO_THRESH:
            ra = local_router_of.get(a)
            rb = local_router_of.get(b)
            na = len([i for i in router_ifaces.get(ra, []) if status.get(i) == 'up'])
            nb = len([i for i in router_ifaces.get(rb, []) if status.get(i) == 'up'])
            if router_residual_mid.get(ra, 0.0) <= router_tau(na) and router_residual_mid.get(rb, 0.0) <= router_tau(nb):
                hardened_rx[a] = hardened_tx[a] = 0.0
                hardened_rx[b] = hardened_tx[b] = 0.0
                conf_rx[a] = max(conf_rx[a], 0.95)
                conf_tx[a] = max(conf_tx[a], 0.95)
                conf_rx[b] = max(conf_rx[b], 0.95)
                conf_tx[b] = max(conf_tx[b], 0.95)

    # Stage 3: Confidence-gap-proportional re-sync with scaling guard and router attenuation
    def nudge_toward_mean(val_lo: float, val_hi: float, frac: float) -> float:
        target = 0.5 * (val_lo + val_hi)
        return val_lo + frac * (target - val_lo)

    for a, b in link_pairs:
        if status.get(a) != 'up' or status.get(b) != 'up':
            continue

        ra = local_router_of.get(a)
        rb = local_router_of.get(b)
        att = clamp01(1.0 - max(router_residual_mid.get(ra, 0.0), router_residual_mid.get(rb, 0.0)))

        # Direction 1: a.tx vs b.rx
        a_tx, b_rx = hardened_tx[a], hardened_rx[b]
        if max(a_tx, b_rx) > ZERO_EPS:
            d1 = rel_diff(a_tx, b_rx)
            tau1 = adaptive_tau(a_tx, b_rx)
            if d1 > tau1:
                ca, cb = conf_tx.get(a, 0.7), conf_rx.get(b, 0.7)
                if ca >= cb and abs(scaled_rx_factor.get(b, 1.0) - 1.0) <= STRONG_SCALE_GUARD:
                    f = min(RESYNC_MAX_F, max(0.0, ca - cb)) * att
                    if f > 0.0:
                        old = b_rx
                        new = max(0.0, nudge_toward_mean(old, a_tx, f))
                        hardened_rx[b] = new
                        move_rel = rel_diff(new, old)
                        conf_rx[b] = clamp01(conf_rx[b] * (1.0 - 0.3 * move_rel))
                elif cb > ca and abs(scaled_tx_factor.get(a, 1.0) - 1.0) <= STRONG_SCALE_GUARD:
                    f = min(RESYNC_MAX_F, max(0.0, cb - ca)) * att
                    if f > 0.0:
                        old = a_tx
                        new = max(0.0, nudge_toward_mean(old, b_rx, f))
                        hardened_tx[a] = new
                        move_rel = rel_diff(new, old)
                        conf_tx[a] = clamp01(conf_tx[a] * (1.0 - 0.3 * move_rel))

        # Direction 2: a.rx vs b.tx
        a_rx, b_tx = hardened_rx[a], hardened_tx[b]
        if max(a_rx, b_tx) > ZERO_EPS:
            d2 = rel_diff(a_rx, b_tx)
            tau2 = adaptive_tau(a_rx, b_tx)
            if d2 > tau2:
                ca, cb = conf_rx.get(a, 0.7), conf_tx.get(b, 0.7)
                if ca >= cb and abs(scaled_tx_factor.get(b, 1.0) - 1.0) <= STRONG_SCALE_GUARD:
                    f = min(RESYNC_MAX_F, max(0.0, ca - cb)) * att
                    if f > 0.0:
                        old = b_tx
                        new = max(0.0, nudge_toward_mean(old, a_rx, f))
                        hardened_tx[b] = new
                        move_rel = rel_diff(new, old)
                        conf_tx[b] = clamp01(conf_tx[b] * (1.0 - 0.3 * move_rel))
                elif cb > ca and abs(scaled_rx_factor.get(a, 1.0) - 1.0) <= STRONG_SCALE_GUARD:
                    f = min(RESYNC_MAX_F, max(0.0, cb - ca)) * att
                    if f > 0.0:
                        old = a_rx
                        new = max(0.0, nudge_toward_mean(old, b_tx, f))
                        hardened_rx[a] = new
                        move_rel = rel_diff(new, old)
                        conf_rx[a] = clamp01(conf_rx[a] * (1.0 - 0.3 * move_rel))

    # Final safety: status down => zero with high confidence
    for i in telemetry:
        if status.get(i) != 'up':
            hardened_rx[i] = 0.0
            hardened_tx[i] = 0.0
            conf_rx[i] = max(conf_rx[i], 0.85)
            conf_tx[i] = max(conf_tx[i], 0.85)

    # Compute router residuals for confidence calibration
    router_residual: Dict[str, float] = {}
    for r, ifs in router_ifaces.items():
        ups = [i for i in ifs if i in telemetry]
        if not ups:
            router_residual[r] = 0.0
            continue
        srx = sum(hardened_rx[i] for i in ups)
        stx = sum(hardened_tx[i] for i in ups)
        denom = max(1.0, srx, stx)
        router_residual[r] = abs(srx - stx) / denom

    # Stage 4: Confidence calibration with strong-scale guards and peer smoothing
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
        # Base blend
        base_rx = clamp01(1.0 - (0.55 * r_meas_rx + 0.35 * r_link_rx + 0.10 * rtr))
        base_tx = clamp01(1.0 - (0.55 * r_meas_tx + 0.35 * r_link_tx + 0.10 * rtr))
        # Scale-factor term
        alpha_rx = abs(scaled_rx_factor.get(i, 1.0) - 1.0)
        alpha_tx = abs(scaled_tx_factor.get(i, 1.0) - 1.0)
        scale_term_rx = clamp01(1.0 - min(0.5, alpha_rx))
        scale_term_tx = clamp01(1.0 - min(0.5, alpha_tx))
        c_rx = clamp01(0.90 * base_rx + 0.10 * scale_term_rx)
        c_tx = clamp01(0.90 * base_tx + 0.10 * scale_term_tx)

        # Down => high floor
        if status.get(i) != 'up':
            c_rx = max(c_rx, 0.85)
            c_tx = max(c_tx, 0.85)
        return c_rx, c_tx

    for i in telemetry:
        cr, ct = compute_conf(i)
        conf_rx[i], conf_tx[i] = cr, ct
        # Clip-hit and strong-scale penalties
        if abs(scaled_rx_factor.get(i, 1.0) - 1.0) >= 0.10 or clip_hit_rx.get(i, False):
            conf_rx[i] = clamp01(conf_rx[i] * CLIP_HIT_PENALTY)
        if abs(scaled_tx_factor.get(i, 1.0) - 1.0) >= 0.10 or clip_hit_tx.get(i, False):
            conf_tx[i] = clamp01(conf_tx[i] * CLIP_HIT_PENALTY)
        if abs(scaled_rx_factor.get(i, 1.0) - 1.0) > 0.08:
            conf_rx[i] = clamp01(conf_rx[i] * STRONG_PENALTY)
        if abs(scaled_tx_factor.get(i, 1.0) - 1.0) > 0.08:
            conf_tx[i] = clamp01(conf_tx[i] * STRONG_PENALTY)
        # Untouched boost: <1% change and good symmetry
        p = peer_of.get(i)
        if p and p in telemetry and status.get(i) == 'up' and status.get(p) == 'up':
            if rel_diff(hardened_rx[i], orig_rx[i]) < 0.01:
                if rel_diff(hardened_rx[i], hardened_tx[p]) <= adaptive_tau(hardened_rx[i], hardened_tx[p]):
                    conf_rx[i] = min(0.98, conf_rx[i] + UNTOUCHED_BOOST)
            if rel_diff(hardened_tx[i], orig_tx[i]) < 0.01:
                if rel_diff(hardened_tx[i], hardened_rx[p]) <= adaptive_tau(hardened_tx[i], hardened_rx[p]):
                    conf_tx[i] = min(0.98, conf_tx[i] + UNTOUCHED_BOOST)

    # Peer smoothing (order-independent staged)
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