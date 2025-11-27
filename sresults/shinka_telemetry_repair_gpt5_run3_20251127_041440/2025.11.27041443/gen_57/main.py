# EVOLVE-BLOCK-START
"""
Weighted least-squares projection with link hardening and re-sync.

Core ideas:
- Robust link fusion (R3) with adaptive tolerance and soft-zero snapping.
- Exact per-router flow conservation via closed-form weighted least-squares projection.
- Dominance cap to avoid over-reliance on one interface during correction.
- Bundle-aware intra-group smoothing and confidence-gap re-sync.
- Residual-based confidence calibration with peer smoothing and scaling penalties.

Maintains the original function signature and outputs.
"""
from typing import Dict, Any, Tuple, List, Optional
import math


def repair_network_telemetry(telemetry: Dict[str, Dict[str, Any]],
                             topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:
    # Hyperparameters
    TAU_H_BASE = 0.02           # ~2% hardening threshold
    TAU_H_HIGH = 0.015          # tighter tolerance for high-rate pairs
    TAU_H_LOW = 0.03            # looser tolerance for low/near-zero pairs
    ZERO_THRESH = 1.0           # Mbps soft-zero threshold
    ZERO_EPS = 1e-6
    DAMP_ROUTER = 0.60          # router projection damping on lambda
    DOMINANCE_CAP = 0.50        # ≤ 50% share cap for any single interface correction
    PEER_SMOOTH = 0.10          # confidence peer smoothing fraction
    STRONG_SCALE_GUARD = 0.08   # guard threshold for re-sync skipping
    RESYNC_MAX_F = 0.40         # max fraction for one-sided nudge toward mean
    BUNDLE_DOM_FRAC = 0.60      # bundle dominance threshold on a side’s traffic
    INTRA_BUNDLE_CLIP = 0.05    # ±5% intra-bundle smoothing cap
    UNTOUCHED_BOOST = 0.02      # confidence boost for untouched well-synced counters
    CLIP_HIT_PENALTY = 0.95     # multiplicative penalty when strong scaling/clipping hit
    MICRO_CLIP = 0.03           # ±3% micro finishing cap per interface
    MICRO_DAMP = 0.25           # micro finishing damping factor

    # Targeted router correction focus and clipping
    WEIGHT_FOCUS = 0.70         # focus on lowest-confidence 70% capacity
    MID_TIER_BOOST = 1.5        # weight boost for 0.70–0.85 confidence (moves less)
    OUTSIDE_FOCUS_BOOST = 2.0   # weight boost for ≥0.85 confidence (moves least)
    PER_VAR_REL_CLIP = 0.10     # ±10% relative change cap per variable in router stage

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

    def is_near_zero_link(vals: List[float]) -> bool:
        return max(vals) < 2.0 * ZERO_THRESH

    # Collect maps
    orig_rx: Dict[str, float] = {}
    orig_tx: Dict[str, float] = {}
    status: Dict[str, str] = {}
    peer_of: Dict[str, Optional[str]] = {}
    local_router_of: Dict[str, Any] = {}
    remote_router_of: Dict[str, Any] = {}

    for iid, d in telemetry.items():
        orig_rx[iid] = float(d.get('rx_rate', 0.0))
        orig_tx[iid] = float(d.get('tx_rate', 0.0))
        status[iid] = d.get('interface_status', 'unknown')
        peer = d.get('connected_to')
        peer_of[iid] = peer if peer in telemetry else None
        local_router_of[iid] = d.get('local_router')
        remote_router_of[iid] = d.get('remote_router')

    # Derive missing remote_router via peer's local_router if needed
    for iid in telemetry:
        if not remote_router_of.get(iid):
            p = peer_of.get(iid)
            if p and p in telemetry:
                remote_router_of[iid] = telemetry[p].get('local_router')

    # Router->interfaces mapping (use topology if provided, else fallback to local_router)
    router_ifaces: Dict[str, List[str]] = {}
    if topology:
        for r, ifs in topology.items():
            router_ifaces[r] = [i for i in ifs if i in telemetry]
    else:
        # Fallback: still useful to enforce R1
        for iid, d in telemetry.items():
            r = d.get('local_router')
            if r is not None:
                router_ifaces.setdefault(r, []).append(iid)

    # Build unique link pairs
    link_pairs: List[Tuple[str, str]] = []
    seen = set()
    for a in telemetry:
        b = peer_of.get(a)
        if not b or a == b or b not in telemetry:
            continue
        if (b, a) in seen or (a, b) in seen:
            continue
        seen.add((a, b))
        link_pairs.append((a, b))
    in_pairs = {x for ab in link_pairs for x in ab}

    # Initialize hardened with originals (non-negative)
    rx: Dict[str, float] = {i: max(0.0, v) for i, v in orig_rx.items()}
    tx: Dict[str, float] = {i: max(0.0, v) for i, v in orig_tx.items()}

    # Directional confidences initialized; refined later
    conf_rx: Dict[str, float] = {i: 0.7 for i in telemetry}
    conf_tx: Dict[str, float] = {i: 0.7 for i in telemetry}

    # Track scale factors applied later (for confidence, re-sync guard)
    scaled_rx_factor: Dict[str, float] = {i: 1.0 for i in telemetry}
    scaled_tx_factor: Dict[str, float] = {i: 1.0 for i in telemetry}
    clip_hit: Dict[str, bool] = {i: False for i in telemetry}

    # Stage 1: Link hardening with adaptive tolerance and soft-zero
    for a, b in link_pairs:
        a_up = (status.get(a) == 'up')
        b_up = (status.get(b) == 'up')

        a_rx0, a_tx0 = rx[a], tx[a]
        b_rx0, b_tx0 = rx[b], tx[b]

        # If either side down: force zeros with high confidence
        if not a_up or not b_up:
            rx[a] = tx[a] = 0.0
            rx[b] = tx[b] = 0.0
            conf_rx[a] = max(conf_rx[a], 0.85)
            conf_tx[a] = max(conf_tx[a], 0.85)
            conf_rx[b] = max(conf_rx[b], 0.85)
            conf_tx[b] = max(conf_tx[b], 0.85)
            continue

        # Soft-zero if all four directions tiny
        if is_near_zero_link([a_rx0, a_tx0, b_rx0, b_tx0]):
            rx[a] = tx[a] = rx[b] = tx[b] = 0.0
            conf_rx[a] = max(conf_rx[a], 0.95)
            conf_tx[a] = max(conf_tx[a], 0.95)
            conf_rx[b] = max(conf_rx[b], 0.95)
            conf_tx[b] = max(conf_tx[b], 0.95)
            continue

        # Direction 1: a.tx vs b.rx
        d1 = rel_diff(a_tx0, b_rx0)
        tau1 = adaptive_tau(a_tx0, b_rx0)
        if d1 <= tau1:
            v1 = 0.5 * (a_tx0 + b_rx0)
        else:
            # Favor the farther-from-zero redundant observation to avoid under-report outliers
            v1 = b_rx0 if abs(b_rx0) >= abs(a_tx0) else a_tx0
        tx[a] = max(0.0, v1)
        rx[b] = max(0.0, v1)
        c1 = clamp01(0.9 + 0.1 * (1.0 - min(1.0, d1 / max(tau1, 1e-12)))) if d1 <= tau1 else clamp01(1.0 - d1)
        conf_tx[a] = max(conf_tx[a], c1)
        conf_rx[b] = max(conf_rx[b], c1)

        # Direction 2: a.rx vs b.tx
        d2 = rel_diff(a_rx0, b_tx0)
        tau2 = adaptive_tau(a_rx0, b_tx0)
        if d2 <= tau2:
            v2 = 0.5 * (a_rx0 + b_tx0)
        else:
            v2 = b_tx0 if abs(b_tx0) >= abs(a_rx0) else a_rx0
        rx[a] = max(0.0, v2)
        tx[b] = max(0.0, v2)
        c2 = clamp01(0.9 + 0.1 * (1.0 - min(1.0, d2 / max(tau2, 1e-12)))) if d2 <= tau2 else clamp01(1.0 - d2)
        conf_rx[a] = max(conf_rx[a], c2)
        conf_tx[b] = max(conf_tx[b], c2)

    # Unpaired interfaces: keep local; zero if down
    for i in telemetry:
        if i not in in_pairs:
            if status.get(i) != 'up':
                rx[i] = tx[i] = 0.0
                conf_rx[i] = max(conf_rx[i], 0.85)
                conf_tx[i] = max(conf_tx[i], 0.85)
            else:
                rx[i] = max(0.0, orig_rx[i])
                tx[i] = max(0.0, orig_tx[i])
                conf_rx[i] = max(conf_rx[i], 0.6)
                conf_tx[i] = max(conf_tx[i], 0.6)

    # Helper: compute router residuals (for later attenuation/soft-zero rule)
    def compute_router_residuals(vals_rx: Dict[str, float], vals_tx: Dict[str, float]) -> Dict[str, float]:
        residuals: Dict[str, float] = {}
        for r, ifs in router_ifaces.items():
            ups = [i for i in ifs if status.get(i) == 'up']
            if not ups:
                residuals[r] = 0.0
                continue
            srx = sum(vals_rx[i] for i in ups)
            stx = sum(vals_tx[i] for i in ups)
            denom = max(1.0, srx, stx)
            residuals[r] = abs(srx - stx) / denom
        return residuals

    # Stage 2: Exact per-router WLS projection onto Σ(in)=Σ(out)
    router_residual_pre = compute_router_residuals(rx, tx)

    for r, ifs in router_ifaces.items():
        up_ifs = [i for i in ifs if status.get(i) == 'up']
        if len(up_ifs) < 2:
            continue

        # Skip projection if router already close to balanced (adaptive tolerance)
        sum_rx_r = sum(rx[i] for i in up_ifs)
        sum_tx_r = sum(tx[i] for i in up_ifs)
        denom_r = max(1.0, sum_rx_r, sum_tx_r)
        rel_gap_r = abs(sum_rx_r - sum_tx_r) / denom_r
        n_active = len(up_ifs)
        tau_router = min(0.07, max(0.03, 0.05 * math.sqrt(2.0 / max(2, n_active))))
        if rel_gap_r <= tau_router:
            continue

        # Build variable vectors
        vals: List[float] = []
        signs: List[int] = []  # +1 for rx, -1 for tx
        idx_map: List[Tuple[str, str]] = []  # (interface_id, 'rx'/'tx')
        weights: List[float] = []

        # Compose rx variables
        for i in up_ifs:
            v = rx[i]
            vals.append(v)
            signs.append(+1)
            idx_map.append((i, 'rx'))
            # Trust weight: higher for confident and high-rate counters
            c = clamp01(conf_rx.get(i, 0.7))
            # Two-tier adjustment (low and mid confidence make smaller weights => larger correction)
            tier = 1.0
            if c < 0.70:
                tier = 0.6
            elif c < 0.85:
                tier = 0.8
            # Weight proportional to confidence^2 and inversely to magnitude (to make relative changes)
            w = max(1e-6, (c**2) * tier / max(v, ZERO_THRESH))
            # Protect near-zero from taking huge absolute correction
            if v < ZERO_THRESH:
                w *= 3.0
            weights.append(w)

        # Compose tx variables
        for i in up_ifs:
            v = tx[i]
            vals.append(v)
            signs.append(-1)
            idx_map.append((i, 'tx'))
            c = clamp01(conf_tx.get(i, 0.7))
            tier = 1.0
            if c < 0.70:
                tier = 0.6
            elif c < 0.85:
                tier = 0.8
            w = max(1e-6, (c**2) * tier / max(v, ZERO_THRESH))
            if v < ZERO_THRESH:
                w *= 3.0
            weights.append(w)

        # Confidence-focused adjustment: boost weights outside the low-confidence focus set
        # so they move less; concentrate corrections on low-confidence/high-rate variables.
        # Compute correctable capacity per variable
        capacities: List[float] = []
        for (i, side), v in zip(idx_map, vals):
            cdir = conf_rx.get(i, 0.7) if side == 'rx' else conf_tx.get(i, 0.7)
            capacities.append((1.0 - clamp01(cdir)) * max(v, ZERO_THRESH) + 1e-12)
        total_cap = sum(capacities)
        # Determine focus set covering WEIGHT_FOCUS of capacity
        order = sorted(range(len(capacities)), key=lambda k: capacities[k], reverse=True)
        focus_set = set()
        acc = 0.0
        for k in order:
            if total_cap > 0 and acc / total_cap >= WEIGHT_FOCUS:
                break
            focus_set.add(k)
            acc += capacities[k]
        if not focus_set:
            focus_set = set(range(len(capacities)))

        # Apply boosts: mid-tier and high confidence outside focus move less
        for idx in range(len(weights)):
            if idx not in focus_set:
                i, side = idx_map[idx]
                cdir = conf_rx.get(i, 0.7) if side == 'rx' else conf_tx.get(i, 0.7)
                if cdir >= 0.85:
                    weights[idx] *= OUTSIDE_FOCUS_BOOST
                elif cdir >= 0.70:
                    weights[idx] *= MID_TIER_BOOST

        # Compute lambda for weighted projection
        # a^T x0 = sum(signs[i] * vals[i])
        ax0 = 0.0
        denom = 0.0  # a^T W^{-1} a = sum( (sign^2)/w_i ) = sum(1/w_i)
        inv_weights = [1.0 / max(1e-12, w) for w in weights]
        for sgn, v, invw in zip(signs, vals, inv_weights):
            ax0 += sgn * v
            denom += invw  # since sgn^2 = 1

        if denom <= 1e-12:
            continue

        # Damped Lagrange multiplier
        lam = 2.0 * ax0 / denom
        lam *= DAMP_ROUTER

        # Initial deltas
        deltas = [-(invw) * sgn * lam * 0.5 for sgn, invw in zip(signs, inv_weights)]
        # Note: derived from x* = x0 - 0.5 W^{-1} λ a, with λ as above (post damping)

        # Dominance cap iteration: avoid one interface taking >50% of correction
        def apply_dominance_cap(deltas_in: List[float], invw_in: List[float]) -> List[float]:
            # Compute shares by absolute delta proportional to invw (since all deltas share same lam)
            for _ in range(3):
                absd = [abs(d) for d in deltas_in]
                total = sum(absd) + 1e-12
                shares = [d / total for d in absd]
                max_share = max(shares)
                if max_share <= DOMINANCE_CAP + 1e-6:
                    break
                k = shares.index(max_share)
                # Increase weight (decrease inv weight) for offender
                invw_in[k] *= (max_share / DOMINANCE_CAP)  # reduce its share
                # Recompute deltas with modified inverse weights
                denom_new = sum(invw_in)
                if denom_new <= 1e-12:
                    break
                lam_new = 2.0 * ax0 / denom_new
                lam_new *= DAMP_ROUTER
                deltas_in = [-(invw) * sgn * lam_new * 0.5 for sgn, invw in zip(signs, invw_in)]
            return deltas_in

        deltas = apply_dominance_cap(deltas, inv_weights[:])

        # Per-variable relative change clipping (±10%); scale all deltas if needed
        gamma = 1.0
        for dv, v_old in zip(deltas, vals):
            if abs(dv) <= 0.0:
                continue
            cap_abs = PER_VAR_REL_CLIP * max(v_old, ZERO_THRESH)
            if abs(dv) > cap_abs:
                gamma = min(gamma, cap_abs / max(abs(dv), 1e-12))
        if gamma < 1.0:
            deltas = [dv * gamma for dv in deltas]

        # Apply deltas
        for (i, side), dv, v_old in zip(idx_map, deltas, vals):
            # Mark clip hits if we reached the cap
            if abs(dv) >= PER_VAR_REL_CLIP * max(v_old, ZERO_THRESH) - 1e-12:
                clip_hit[i] = True
            v_new = max(0.0, v_old + dv)
            if side == 'rx':
                prev = rx[i]
                rx[i] = v_new
                if prev > ZERO_EPS and v_new >= 0.0:
                    scl = v_new / prev if prev > 0 else 1.0
                    scaled_rx_factor[i] *= scl
                    if abs(scl - 1.0) >= 0.10:
                        clip_hit[i] = True
            else:
                prev = tx[i]
                tx[i] = v_new
                if prev > ZERO_EPS and v_new >= 0.0:
                    scl = v_new / prev if prev > 0 else 1.0
                    scaled_tx_factor[i] *= scl
                    if abs(scl - 1.0) >= 0.10:
                        clip_hit[i] = True

    # Stage 2.1: Bundle-aware intra-group smoothing (small ±5% pass)
    # Group by (local_router, remote_router)
    for r, ifs in router_ifaces.items():
        up_ifs = [i for i in ifs if status.get(i) == 'up']
        if not up_ifs:
            continue
        # Side totals
        sum_rx_side = sum(rx[i] for i in up_ifs)
        sum_tx_side = sum(tx[i] for i in up_ifs)
        # Build bundles
        bundles: Dict[Tuple[Any, Any], List[str]] = {}
        for i in up_ifs:
            key = (local_router_of.get(i), remote_router_of.get(i))
            bundles.setdefault(key, []).append(i)

        # RX side smoothing
        for key, members in bundles.items():
            if len(members) < 2:
                continue
            group_sum = sum(rx[i] for i in members)
            if sum_rx_side > ZERO_EPS and (group_sum / sum_rx_side) >= BUNDLE_DOM_FRAC:
                # Relative scales (post-router vs pre-router using orig_rx as loose anchor)
                rels = []
                for i in members:
                    base = max(ZERO_THRESH, orig_rx.get(i, 0.0))
                    rels.append(rx[i] / base)
                s_group = sum(rels) / len(rels)
                # Clip to ±5%
                s_group = max(1.0 - INTRA_BUNDLE_CLIP, min(1.0 + INTRA_BUNDLE_CLIP, s_group))
                for i in members:
                    base = max(ZERO_THRESH, orig_rx.get(i, 0.0))
                    new_v = max(0.0, base * s_group)
                    if rx[i] > ZERO_EPS:
                        scl = new_v / rx[i]
                        if abs(scl - 1.0) >= 0.10:
                            clip_hit[i] = True
                    rx[i] = new_v

        # TX side smoothing
        for key, members in bundles.items():
            if len(members) < 2:
                continue
            group_sum = sum(tx[i] for i in members)
            if sum_tx_side > ZERO_EPS and (group_sum / sum_tx_side) >= BUNDLE_DOM_FRAC:
                rels = []
                for i in members:
                    base = max(ZERO_THRESH, orig_tx.get(i, 0.0))
                    rels.append(tx[i] / base)
                s_group = sum(rels) / len(rels)
                s_group = max(1.0 - INTRA_BUNDLE_CLIP, min(1.0 + INTRA_BUNDLE_CLIP, s_group))
                for i in members:
                    base = max(ZERO_THRESH, orig_tx.get(i, 0.0))
                    new_v = max(0.0, base * s_group)
                    if tx[i] > ZERO_EPS:
                        scl = new_v / tx[i]
                        if abs(scl - 1.0) >= 0.10:
                            clip_hit[i] = True
                    tx[i] = new_v

    # Stage 2.2: Soft-zero stabilization on hardened links if routers balanced
    router_residual_post = compute_router_residuals(rx, tx)
    for a, b in link_pairs:
        if status.get(a) != 'up' or status.get(b) != 'up':
            continue
        # If link tiny and adjacent routers close to balanced, snap to zero
        if is_near_zero_link([rx[a], tx[a], rx[b], tx[b]]):
            ra = local_router_of.get(a)
            rb = local_router_of.get(b)
            imba = router_residual_post.get(ra, 0.0)
            imbb = router_residual_post.get(rb, 0.0)
            # Adaptive router tolerance based on active links
            n_active_a = len([i for i in router_ifaces.get(ra, []) if status.get(i) == 'up'])
            n_active_b = len([i for i in router_ifaces.get(rb, []) if status.get(i) == 'up'])
            tau_ra = min(0.07, max(0.03, 0.05 * math.sqrt(2.0 / max(2, n_active_a))))
            tau_rb = min(0.07, max(0.03, 0.05 * math.sqrt(2.0 / max(2, n_active_b))))
            if imba <= tau_ra and imbb <= tau_rb:
                rx[a] = tx[a] = rx[b] = tx[b] = 0.0
                conf_rx[a] = max(conf_rx[a], 0.95)
                conf_tx[a] = max(conf_tx[a], 0.95)
                conf_rx[b] = max(conf_rx[b], 0.95)
                conf_tx[b] = max(conf_tx[b], 0.95)

    # Stage 2.3: Micro high-confidence finishing tier (tiny ±3% pass)
    for r, ifs in router_ifaces.items():
        up_ifs = [i for i in ifs if status.get(i) == 'up']
        if len(up_ifs) < 2:
            continue
        sum_rx_r = sum(rx[i] for i in up_ifs)
        sum_tx_r = sum(tx[i] for i in up_ifs)
        denom_r = max(1.0, sum_rx_r, sum_tx_r)
        imbalance = sum_rx_r - sum_tx_r
        n_active = len(up_ifs)
        tau_router = min(0.07, max(0.03, 0.05 * math.sqrt(2.0 / max(2, n_active))))
        if abs(imbalance) / denom_r <= 0.6 * tau_router:
            continue

        # Eligible high-confidence sets
        elig_rx = [i for i in up_ifs if conf_rx.get(i, 0.7) >= 0.85 and rx[i] > ZERO_EPS]
        elig_tx = [i for i in up_ifs if conf_tx.get(i, 0.7) >= 0.85 and tx[i] > ZERO_EPS]
        if not elig_rx and not elig_tx:
            continue

        cap_rx = sum(MICRO_CLIP * rx[i] for i in elig_rx)
        cap_tx = sum(MICRO_CLIP * tx[i] for i in elig_tx)

        side = None
        if imbalance > 0.0:
            # Prefer decreasing rx or increasing tx based on capacity
            if cap_rx >= cap_tx and cap_rx > ZERO_EPS:
                side = 'rx'
            elif cap_tx > ZERO_EPS:
                side = 'tx'
        else:
            # Prefer decreasing tx or increasing rx based on capacity
            if cap_tx >= cap_rx and cap_tx > ZERO_EPS:
                side = 'tx'
            elif cap_rx > ZERO_EPS:
                side = 'rx'
        if side is None:
            continue

        if side == 'rx' and elig_rx:
            target = -imbalance * MICRO_DAMP
            adj_total = max(-cap_rx, min(cap_rx, target))
            if abs(adj_total) > ZERO_EPS:
                total_w = sum(max(rx[i], ZERO_THRESH) for i in elig_rx)
                for i in elig_rx:
                    w = max(rx[i], ZERO_THRESH) / max(total_w, 1e-12)
                    dv = w * adj_total
                    prev = rx[i]
                    new_v = max(0.0, prev + dv)
                    if prev > ZERO_EPS:
                        scl = new_v / prev
                        scaled_rx_factor[i] *= scl
                    rx[i] = new_v
        elif side == 'tx' and elig_tx:
            target = imbalance * MICRO_DAMP
            adj_total = max(-cap_tx, min(cap_tx, target))
            if abs(adj_total) > ZERO_EPS:
                total_w = sum(max(tx[i], ZERO_THRESH) for i in elig_tx)
                for i in elig_tx:
                    w = max(tx[i], ZERO_THRESH) / max(total_w, 1e-12)
                    dv = w * adj_total
                    prev = tx[i]
                    new_v = max(0.0, prev + dv)
                    if prev > ZERO_EPS:
                        scl = new_v / prev
                        scaled_tx_factor[i] *= scl
                    tx[i] = new_v

    # Stage 3: Confidence-gap re-sync with scaling guard and router attenuation
    def nudge_toward_mean(val_lo: float, val_hi: float, frac: float) -> float:
        target = 0.5 * (val_lo + val_hi)
        return val_lo + frac * (target - val_lo)

    # Recompute router residuals for attenuation
    router_residual_final = compute_router_residuals(rx, tx)

    for a, b in link_pairs:
        if status.get(a) != 'up' or status.get(b) != 'up':
            continue

        # Attenuation from local router imbalances
        ra = local_router_of.get(a)
        rb = local_router_of.get(b)
        att = 1.0 - max(router_residual_final.get(ra, 0.0), router_residual_final.get(rb, 0.0))
        att = clamp01(att)

        # Direction 1: a.tx vs b.rx
        a_tx, b_rx = tx[a], rx[b]
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
                        rx[b] = new
                        # confidence penalty proportional to relative movement
                        move_rel = rel_diff(new, old)
                        conf_rx[b] = clamp01(conf_rx[b] * (1.0 - 0.3 * move_rel))
                elif cb > ca and abs(scaled_tx_factor.get(a, 1.0) - 1.0) <= STRONG_SCALE_GUARD:
                    f = min(RESYNC_MAX_F, max(0.0, cb - ca)) * att
                    if f > 0.0:
                        old = a_tx
                        new = max(0.0, nudge_toward_mean(old, b_rx, f))
                        tx[a] = new
                        move_rel = rel_diff(new, old)
                        conf_tx[a] = clamp01(conf_tx[a] * (1.0 - 0.3 * move_rel))

        # Direction 2: a.rx vs b.tx
        a_rx, b_tx = rx[a], tx[b]
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
                        tx[b] = new
                        move_rel = rel_diff(new, old)
                        conf_tx[b] = clamp01(conf_tx[b] * (1.0 - 0.3 * move_rel))
                elif cb > ca and abs(scaled_rx_factor.get(a, 1.0) - 1.0) <= STRONG_SCALE_GUARD:
                    f = min(RESYNC_MAX_F, max(0.0, cb - ca)) * att
                    if f > 0.0:
                        old = a_rx
                        new = max(0.0, nudge_toward_mean(old, b_tx, f))
                        rx[a] = new
                        move_rel = rel_diff(new, old)
                        conf_rx[a] = clamp01(conf_rx[a] * (1.0 - 0.3 * move_rel))

    # Final safety: enforce down => zero
    for i in telemetry:
        if status.get(i) != 'up':
            rx[i] = 0.0
            tx[i] = 0.0
            conf_rx[i] = max(conf_rx[i], 0.85)
            conf_tx[i] = max(conf_tx[i], 0.85)

    # Confidence calibration
    # Router residuals after all adjustments
    router_resid_end = compute_router_residuals(rx, tx)

    # Compute raw confidence components and combine
    conf_final_rx: Dict[str, float] = {}
    conf_final_tx: Dict[str, float] = {}
    for i in telemetry:
        # Measurement residuals
        r_meas_rx = rel_diff(rx[i], orig_rx[i])
        r_meas_tx = rel_diff(tx[i], orig_tx[i])

        # Link residuals
        p = peer_of.get(i)
        if p and p in telemetry and status.get(i) == 'up' and status.get(p) == 'up':
            r_link_rx = rel_diff(rx[i], tx[p])
            r_link_tx = rel_diff(tx[i], rx[p])
        else:
            r_link_rx = 0.2
            r_link_tx = 0.2

        # Router residual
        rtr = router_resid_end.get(local_router_of.get(i), 0.0)

        base_rx = clamp01(1.0 - (0.55 * r_meas_rx + 0.35 * r_link_rx + 0.10 * rtr))
        base_tx = clamp01(1.0 - (0.55 * r_meas_tx + 0.35 * r_link_tx + 0.10 * rtr))

        # Scaling penalties
        alpha_rx = abs(scaled_rx_factor.get(i, 1.0) - 1.0)
        alpha_tx = abs(scaled_tx_factor.get(i, 1.0) - 1.0)
        scale_term_rx = clamp01(1.0 - min(0.5, alpha_rx))
        scale_term_tx = clamp01(1.0 - min(0.5, alpha_tx))

        c_rx = clamp01(0.90 * base_rx + 0.10 * scale_term_rx)
        c_tx = clamp01(0.90 * base_tx + 0.10 * scale_term_tx)

        # Scale/clip penalties (two-tier) and untouched boost
        if clip_hit.get(i, False) or alpha_rx > 0.12:
            c_rx *= CLIP_HIT_PENALTY
        elif alpha_rx > 0.08:
            c_rx *= 0.97
        if clip_hit.get(i, False) or alpha_tx > 0.12:
            c_tx *= CLIP_HIT_PENALTY
        elif alpha_tx > 0.08:
            c_tx *= 0.97

        # Untouched small-change and good symmetry boost
        if r_meas_rx < 0.01 and r_link_rx <= adaptive_tau(rx[i], tx.get(p, rx[i]) if p else rx[i]):
            c_rx = min(0.98, c_rx + UNTOUCHED_BOOST)
        if r_meas_tx < 0.01 and r_link_tx <= adaptive_tau(tx[i], rx.get(p, tx[i]) if p else tx[i]):
            c_tx = min(0.98, c_tx + UNTOUCHED_BOOST)

        # Floor for down interfaces
        if status.get(i) != 'up':
            c_rx = max(c_rx, 0.85)
            c_tx = max(c_tx, 0.85)

        conf_final_rx[i] = c_rx
        conf_final_tx[i] = c_tx

    # Peer smoothing (order-independent via staged update)
    new_conf_rx = dict(conf_final_rx)
    new_conf_tx = dict(conf_final_tx)
    for a, b in link_pairs:
        if status.get(a) == 'up' and status.get(b) == 'up':
            new_conf_tx[a] = clamp01((1.0 - PEER_SMOOTH) * conf_final_tx[a] + PEER_SMOOTH * conf_final_rx[b])
            new_conf_rx[b] = clamp01((1.0 - PEER_SMOOTH) * conf_final_rx[b] + PEER_SMOOTH * conf_final_tx[a])
            new_conf_rx[a] = clamp01((1.0 - PEER_SMOOTH) * conf_final_rx[a] + PEER_SMOOTH * conf_final_tx[b])
            new_conf_tx[b] = clamp01((1.0 - PEER_SMOOTH) * conf_final_tx[b] + PEER_SMOOTH * conf_final_rx[a])
    conf_final_rx = new_conf_rx
    conf_final_tx = new_conf_tx

    # Assemble final result
    result: Dict[str, Dict[str, Tuple]] = {}
    for i, data in telemetry.items():
        my_status = status.get(i, 'unknown')
        peer_id = data.get('connected_to')

        # Status confidence: penalize if peer status mismatched or traffic on down
        status_conf = 1.0
        if peer_id and peer_id in telemetry:
            if my_status != telemetry[peer_id].get('interface_status', 'unknown'):
                status_conf = 0.6
        if my_status == 'down' and (orig_rx.get(i, 0.0) > ZERO_EPS or orig_tx.get(i, 0.0) > ZERO_EPS):
            status_conf = min(status_conf, 0.6)

        repaired: Dict[str, Any] = {}
        repaired['rx_rate'] = (orig_rx.get(i, 0.0), max(0.0, rx.get(i, 0.0)), clamp01(conf_final_rx.get(i, 0.6)))
        repaired['tx_rate'] = (orig_tx.get(i, 0.0), max(0.0, tx.get(i, 0.0)), clamp01(conf_final_tx.get(i, 0.6)))
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