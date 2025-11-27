# EVOLVE-BLOCK-START
"""
Hybrid Hodor repair: robust link hardening with adaptive fusion + conservative router projection.

This implementation follows and combines:
1) Signal Collection: use redundant signals from both sides of a link.
2) Signal Hardening: direction-wise fusion with adaptive regimes:
   - tiny mismatch: keep local
   - moderate mismatch: average
   - large mismatch: snap to peer with status/zero-aware bias
3) Dynamic Checking / Projection: conservatively enforce router flow conservation with
   damped scaling and clipped factors. After projection, gently re-sync links, adjusting
   only the lower-confidence side if needed.

Outputs repaired telemetry with calibrated confidence scores.
"""
from typing import Dict, Any, Tuple, List
import math


def repair_network_telemetry(telemetry: Dict[str, Dict[str, Any]],
                             topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:
    """
    Repair network telemetry using hybrid per-link fusion and conservative router-level flow projection.

    Args:
        telemetry: per-interface telemetry dictionary with fields:
            - interface_status: "up" or "down"
            - rx_rate: float Mbps
            - tx_rate: float Mbps
            - connected_to: peer interface id
            - local_router: router id
            - remote_router: router id on the other side
        topology: router_id -> list of interface_ids

    Returns:
        Same structure as telemetry, but rx_rate, tx_rate, interface_status become tuples:
        (original_value, repaired_value, confidence) in [0, 1].
        Non-telemetry fields are copied unchanged.
    """
    # Tolerances/thresholds inspired by Hodor
    TAU_H = 0.02          # symmetry tolerance 2%
    TAU_ROUTER = 0.05     # router imbalance tolerance 5% (baseline; adaptively refined)
    EPS = 1e-9
    ZERO_THRESH = 0.1     # Mbps considered near-zero

    # Helper functions
    def safe_rate(x: Any) -> float:
        try:
            v = float(x)
            if not math.isfinite(v) or v < 0:
                return 0.0
            return v
        except Exception:
            return 0.0

    def rel_diff(a: float, b: float) -> float:
        m = max(abs(a), abs(b), 1.0)
        return abs(a - b) / m

    def clamp01(x: float) -> float:
        if x < 0.0: return 0.0
        if x > 1.0: return 1.0
        return x

    def clamp(x: float, lo: float, hi: float) -> float:
        return lo if x < lo else hi if x > hi else x

    def sigmoid(x: float) -> float:
        # Numerically safe sigmoid
        if x >= 0:
            z = math.exp(-x)
            return 1.0 / (1.0 + z)
        else:
            z = math.exp(x)
            return z / (1.0 + z)

    def tau_h_dir(v1: float, v2: float, c1: float = None, c2: float = None) -> float:
        """
        Adaptive hardening tolerance:
        - 1.5% when both directions are high-rate (>100 Mbps) and confidences (if provided) are high (>=0.8)
        - 3% when either direction is low-rate (<1 Mbps) or any confidence is low (<0.7)
        - 2% baseline otherwise
        """
        high = (v1 > 100.0 and v2 > 100.0)
        low = (v1 < 1.0 or v2 < 1.0)
        high_conf = (c1 is not None and c2 is not None and c1 >= 0.8 and c2 >= 0.8)
        low_conf = (c1 is not None and c2 is not None and (c1 < 0.7 or c2 < 0.7))
        if high and high_conf:
            return 0.015
        if low or low_conf:
            return 0.03
        return 0.02

    def tau_router(n_active: int) -> float:
        """
        Adaptive router imbalance tolerance based on number of active interfaces.
        0.05 * sqrt(2 / max(2, n_active)) clamped to [0.03, 0.07].
        """
        base = 0.05 * math.sqrt(2.0 / max(2, n_active))
        return max(0.03, min(0.07, base))

    # Build peer mapping
    peers: Dict[str, str] = {}
    for if_id, data in telemetry.items():
        peer = data.get("connected_to")
        if isinstance(peer, str) and peer in telemetry:
            peers[if_id] = peer

    # Build router->interfaces from topology with fallback to local_router
    router_ifaces: Dict[str, List[str]] = {}
    for r, if_list in topology.items():
        router_ifaces.setdefault(r, [])
        for i in if_list:
            if i in telemetry:
                router_ifaces[r].append(i)
    for if_id, data in telemetry.items():
        r = data.get("local_router")
        if r is None:
            r = f"unknown_router::{if_id}"
        router_ifaces.setdefault(r, [])
        if if_id not in router_ifaces[r]:
            router_ifaces[r].append(if_id)

    # Prepare originals and status
    orig_tx: Dict[str, float] = {}
    orig_rx: Dict[str, float] = {}
    status: Dict[str, str] = {}
    router_of: Dict[str, str] = {}
    for r, ifs in router_ifaces.items():
        for i in ifs:
            router_of[i] = r

    for if_id, data in telemetry.items():
        orig_tx[if_id] = safe_rate(data.get("tx_rate", 0.0))
        orig_rx[if_id] = safe_rate(data.get("rx_rate", 0.0))
        s = data.get("interface_status", "unknown")
        status[if_id] = s if s in ("up", "down") else "unknown"

    # Precompute router activity-based near-zero thresholds (orig-rate based)
    router_thr_pre: Dict[str, float] = {}
    for r, ifs in router_ifaces.items():
        sum_tx_o = sum(orig_tx.get(i, 0.0) for i in ifs)
        sum_rx_o = sum(orig_rx.get(i, 0.0) for i in ifs)
        router_thr_pre[r] = max(ZERO_THRESH, 0.002 * (sum_tx_o + sum_rx_o))

    # Stage 1: Link hardening with adaptive fusion
    hard_tx: Dict[str, float] = {}
    hard_rx: Dict[str, float] = {}
    conf_tx_link: Dict[str, float] = {}
    conf_rx_link: Dict[str, float] = {}
    pre_mismatch_tx: Dict[str, float] = {}
    pre_mismatch_rx: Dict[str, float] = {}

    visited = set()

    def fuse_direction(v_local: float, v_peer: float, s_local: str, s_peer: str, zero_thr: float) -> Tuple[float, float]:
        mismatch = rel_diff(v_local, v_peer)
        th = tau_h_dir(v_local, v_peer)

        # Both near-zero => zero with high confidence
        if max(v_local, v_peer) < ZERO_THRESH:
            return 0.0, 0.95

        # If within adaptive hardening tolerance, keep local reading (minimal change)
        if mismatch <= th:
            return v_local, 0.95

        # Moderate mismatch: average
        if mismatch <= 0.10:
            fused = 0.5 * v_local + 0.5 * v_peer
            return fused, clamp01(1.0 - mismatch)

        # Large mismatch: prefer the more plausible side (router-aware near-zero)
        nz_thr = max(ZERO_THRESH, zero_thr)
        if v_local < nz_thr and v_peer >= nz_thr:
            return v_peer, clamp01(1.0 - mismatch)
        if v_peer < nz_thr and v_local >= nz_thr:
            return v_local, clamp01(1.0 - mismatch)

        # Status-aware bias
        if s_local == "down" and s_peer == "up":
            return v_peer, clamp01(1.0 - mismatch)
        if s_peer == "down" and s_local == "up":
            return v_local, clamp01(1.0 - mismatch)

        # Otherwise, snap mostly to peer to resolve asymmetry decisively
        fused = 0.3 * v_local + 0.7 * v_peer
        return fused, clamp01(1.0 - mismatch)

    for if_id, data in telemetry.items():
        if if_id in visited:
            continue
        peer = peers.get(if_id)
        if not peer:
            # Isolated interface: keep as-is with conservative confidence
            hard_tx[if_id] = orig_tx[if_id]
            hard_rx[if_id] = orig_rx[if_id]
            conf_tx_link[if_id] = 0.6
            conf_rx_link[if_id] = 0.6
            pre_mismatch_tx[if_id] = 0.4
            pre_mismatch_rx[if_id] = 0.4
            visited.add(if_id)
            continue

        visited.add(if_id)
        visited.add(peer)

        a, b = if_id, peer
        a_tx, a_rx = orig_tx[a], orig_rx[a]
        b_tx, b_rx = orig_tx[b], orig_rx[b]
        sa, sb = status[a], status[b]

        # If both ends down: force zeros
        if sa == "down" and sb == "down":
            hard_tx[a] = 0.0
            hard_rx[a] = 0.0
            hard_tx[b] = 0.0
            hard_rx[b] = 0.0
            conf_tx_link[a] = 0.98
            conf_rx_link[a] = 0.98
            conf_tx_link[b] = 0.98
            conf_rx_link[b] = 0.98
            pre_mismatch_tx[a] = rel_diff(a_tx, b_rx)
            pre_mismatch_rx[a] = rel_diff(a_rx, b_tx)
            pre_mismatch_tx[b] = rel_diff(b_tx, a_rx)
            pre_mismatch_rx[b] = rel_diff(b_rx, a_tx)
            continue

        # Directional mismatches
        diff_ab = rel_diff(a_tx, b_rx)
        diff_ba = rel_diff(b_tx, a_rx)
        pre_mismatch_tx[a] = diff_ab
        pre_mismatch_rx[b] = diff_ab
        pre_mismatch_tx[b] = diff_ba
        pre_mismatch_rx[a] = diff_ba

        # Pair-adaptive near-zero threshold for fusion using precomputed router activity
        ra_pair = router_of.get(a, None)
        rb_pair = router_of.get(b, None)
        zero_thr_pair = max(router_thr_pre.get(ra_pair, ZERO_THRESH), router_thr_pre.get(rb_pair, ZERO_THRESH))
        fused_ab, c_ab = fuse_direction(a_tx, b_rx, sa, sb, zero_thr_pair)
        fused_ba, c_ba = fuse_direction(b_tx, a_rx, sb, sa, zero_thr_pair)

        # Assign hardened values per direction maintaining symmetry mapping
        hard_tx[a] = fused_ab
        hard_rx[b] = fused_ab
        hard_tx[b] = fused_ba
        hard_rx[a] = fused_ba

        conf_tx_link[a] = c_ab
        conf_rx_link[b] = c_ab
        conf_tx_link[b] = c_ba
        conf_rx_link[a] = c_ba

    # Ensure all interfaces have hardened values
    for if_id in telemetry.keys():
        if if_id not in hard_tx:
            hard_tx[if_id] = orig_tx[if_id]
            conf_tx_link[if_id] = 0.6
        if if_id not in hard_rx:
            hard_rx[if_id] = orig_rx[if_id]
            conf_rx_link[if_id] = 0.6
        if if_id not in pre_mismatch_tx:
            pre_mismatch_tx[if_id] = 0.4
        if if_id not in pre_mismatch_rx:
            pre_mismatch_rx[if_id] = 0.4

    # Early dynamic soft-zero pre-pass using router-aware thresholds
    def compute_router_thresholds() -> Dict[str, float]:
        thr: Dict[str, float] = {}
        for r, ifs in router_ifaces.items():
            stx = sum(hard_tx.get(i, 0.0) for i in ifs)
            srx = sum(hard_rx.get(i, 0.0) for i in ifs)
            thr[r] = max(ZERO_THRESH, 0.002 * (stx + srx))
        return thr

    def router_imbalance_now(router_id: str) -> float:
        if not router_id or router_id not in router_ifaces:
            return 0.0
        ifs = router_ifaces[router_id]
        stx = sum(hard_tx.get(i, 0.0) for i in ifs)
        srx = sum(hard_rx.get(i, 0.0) for i in ifs)
        return rel_diff(stx, srx)

    router_thr_early = compute_router_thresholds()
    processed_pairs_pre = set()
    for a, data_a in telemetry.items():
        b = data_a.get('connected_to')
        if not isinstance(b, str) or b not in telemetry:
            continue
        key = tuple(sorted([a, b]))
        if key in processed_pairs_pre:
            continue
        processed_pairs_pre.add(key)
        ra = router_of.get(a)
        rb = router_of.get(b)
        thr_pair = 1.5 * max(router_thr_early.get(ra, ZERO_THRESH), router_thr_early.get(rb, ZERO_THRESH))
        if max(hard_tx.get(a, 0.0), hard_rx.get(b, 0.0), hard_tx.get(b, 0.0), hard_rx.get(a, 0.0)) < thr_pair:
            # Check both routers roughly balanced under adaptive tolerance
            def tau_for_router(rid: str) -> float:
                if rid not in router_ifaces:
                    return TAU_ROUTER
                ifs = router_ifaces[rid]
                n_tx = sum(1 for i in ifs if hard_tx.get(i, 0.0) >= ZERO_THRESH)
                n_rx = sum(1 for i in ifs if hard_rx.get(i, 0.0) >= ZERO_THRESH)
                return tau_router(max(n_tx, n_rx))
            if router_imbalance_now(ra) <= tau_for_router(ra) and router_imbalance_now(rb) <= tau_for_router(rb):
                hard_tx[a] = 0.0
                hard_rx[b] = 0.0
                hard_tx[b] = 0.0
                hard_rx[a] = 0.0
                conf_tx_link[a] = max(conf_tx_link.get(a, 0.6), 0.95)
                conf_rx_link[b] = max(conf_rx_link.get(b, 0.6), 0.95)
                conf_tx_link[b] = max(conf_tx_link.get(b, 0.6), 0.95)
                conf_rx_link[a] = max(conf_rx_link.get(a, 0.6), 0.95)

    # Stage 2: Dominance-aware router-level projection with three-tier targeted scaling
    router_imbalance_before: Dict[str, float] = {}
    scaled_tx_factor: Dict[str, float] = {if_id: 1.0 for if_id in telemetry}
    scaled_rx_factor: Dict[str, float] = {if_id: 1.0 for if_id in telemetry}
    strong_scaled_tx: Dict[str, bool] = {if_id: False for if_id in telemetry}
    strong_scaled_rx: Dict[str, bool] = {if_id: False for if_id in telemetry}
    clip_hit_tx: Dict[str, bool] = {if_id: False for if_id in telemetry}
    clip_hit_rx: Dict[str, bool] = {if_id: False for if_id in telemetry}

    for router, if_list in router_ifaces.items():
        # Ignore trivial routers
        if len(if_list) <= 1:
            router_imbalance_before[router] = 0.0
            continue

        sum_tx = sum(hard_tx.get(i, 0.0) for i in if_list)
        sum_rx = sum(hard_rx.get(i, 0.0) for i in if_list)
        mismatch = rel_diff(sum_tx, sum_rx)
        router_imbalance_before[router] = mismatch

        if max(sum_tx, sum_rx) < EPS:
            continue  # nothing to project

        n_active_tx = sum(1 for i in if_list if hard_tx.get(i, 0.0) >= ZERO_THRESH)
        n_active_rx = sum(1 for i in if_list if hard_rx.get(i, 0.0) >= ZERO_THRESH)
        TAU_ROUTER_LOCAL = tau_router(max(n_active_tx, n_active_rx))

        if mismatch > TAU_ROUTER_LOCAL:
            # Rec.1: Expected-penalty lookahead to decide which side to scale
            def side_penalty_rec1(side: str) -> float:
                vals = [hard_tx.get(i, 0.0) if side == "tx" else hard_rx.get(i, 0.0) for i in if_list]
                confs = [clamp01(conf_tx_link.get(i, 0.6) if side == "tx" else conf_rx_link.get(i, 0.6)) for i in if_list]
                active_idx = [k for k, v in enumerate(vals) if v >= ZERO_THRESH]
                if not active_idx:
                    return float('inf')
                w = [(1.0 - confs[k]) * vals[k] for k in active_idx]
                sum_w = sum(w) if w else 0.0
                if sum_w <= 0.0:
                    return float('inf')
                current = sum(vals[k] for k in active_idx)
                target = sum_rx if side == "tx" else sum_tx
                delta = target - current
                denom = sum(vals[k] * w[j] for j, k in enumerate(active_idx))
                if denom == 0:
                    return float('inf')
                k_lin = delta / (0.6 * denom)
                # HHI on weights
                sw = sum_w
                hhi = sum((wi / sw) ** 2 for wi in w) if sw > 0 else 1.0
                # Per-interface penalty preview with confidence-based clip ceilings
                pen_sum = 0.0
                conf_pen = 0.0
                for j, k in enumerate(active_idx):
                    conf_i = confs[k]
                    v_i = vals[k]
                    w_i = w[j]
                    clip_hi_i = 1.12 if conf_i < 0.70 else 1.10
                    scale_raw = 1.0 + 0.6 * k_lin * w_i
                    scale_eff = clamp(scale_raw, 0.90, clip_hi_i)
                    pen_sum += abs(scale_eff - 1.0)
                    conf_pen += (1.0 - conf_i)
                # Dominance penalty via HHI
                penalty_dom = 0.05 * hhi
                return pen_sum + 0.5 * conf_pen + penalty_dom

            pen_tx = side_penalty_rec1("tx")
            pen_rx = side_penalty_rec1("rx")
            if abs(pen_tx - pen_rx) / max(1e-6, max(pen_tx, pen_rx)) < 0.05:
                # Near-tie: fallback to aggregate link confidence
                c_tx_total = sum(conf_tx_link.get(i, 0.5) for i in if_list)
                c_rx_total = sum(conf_rx_link.get(i, 0.5) for i in if_list)
                adjust_side = "tx" if c_tx_total < c_rx_total else "rx"
            else:
                adjust_side = "tx" if pen_tx < pen_rx else "rx"

            # Adaptive dominance-aware per-tier scaling
            def apply_tier(side: str, conf_lo: float, conf_hi: float, delta_in: float, tier_kind: str) -> float:
                if abs(delta_in) < EPS:
                    return 0.0
                vals = {i: (hard_tx[i] if side == "tx" else hard_rx[i]) for i in if_list}
                confs = {i: clamp01(conf_tx_link.get(i, 0.6) if side == "tx" else conf_rx_link.get(i, 0.6)) for i in if_list}
                elig = [i for i in if_list if vals[i] >= ZERO_THRESH and conf_lo <= confs[i] < conf_hi]
                if not elig:
                    return delta_in
                # Weights
                w = {i: (1.0 - confs[i]) * (vals[i]) for i in elig}
                sum_w = sum(w.values())
                if sum_w <= 0.0:
                    return delta_in
                denom_vw = sum(vals[i] * w[i] for i in elig)
                if denom_vw <= 0.0:
                    return delta_in
                # Dominance via HHI
                hhi = sum((w[i] / sum_w) ** 2 for i in elig) if sum_w > 0 else 1.0
                k = delta_in / (0.6 * denom_vw)
                # Dominance-aware absolute cap
                cap_abs = (0.5 - 0.2 * math.sqrt(max(0.0, hhi))) * abs(delta_in) if len(elig) >= 2 else None

                applied = 0.0
                for i in elig:
                    v = vals[i]
                    conf_i = confs[i]
                    wi = w[i]
                    scale_raw = 1.0 + 0.6 * k * wi
                    # Confidence-adaptive clip ceiling
                    if tier_kind == "verylow":
                        clip_hi_i = 1.0 + min(0.15, 0.10 + 0.05 * max(0.0, 0.70 - conf_i))
                        # small bonus ceiling for extremely low conf but we still cap change below
                        per_iface_abs_cap = 0.35 * abs(delta_in)
                    elif tier_kind == "low":
                        clip_hi_i = 1.0 + min(0.12, 0.10 + 0.05 * max(0.0, 0.70 - conf_i))
                        per_iface_abs_cap = None
                    else:  # moderate
                        base_hi = 1.10
                        # Reduce for dominance on the set
                        clip_hi_i = 1.0 + (base_hi - 1.0) * (1.0 - math.sqrt(max(0.0, hhi)))
                        per_iface_abs_cap = None
                    scale_eff = clamp(scale_raw, 0.90, clip_hi_i)
                    change_i = v * (scale_eff - 1.0)
                    # Absolute caps
                    if per_iface_abs_cap is not None and abs(change_i) > per_iface_abs_cap and v > 0:
                        scale_eff = 1.0 + math.copysign(per_iface_abs_cap, change_i) / v
                        change_i = v * (scale_eff - 1.0)
                    if cap_abs is not None and abs(change_i) > cap_abs and v > 0:
                        scale_eff = 1.0 + math.copysign(cap_abs, change_i) / v
                        change_i = v * (scale_eff - 1.0)
                    # Apply
                    if side == "tx":
                        hard_tx[i] = v * scale_eff
                        scaled_tx_factor[i] *= scale_eff
                        pen = abs(scale_eff - 1.0)
                        conf_tx_link[i] *= clamp01(1.0 - 0.4 * clamp01(pen))
                        if pen >= 0.10 or scale_eff <= 0.90 or scale_eff >= clip_hi_i:
                            clip_hit_tx[i] = True
                        if pen > 0.08:
                            strong_scaled_tx[i] = True
                    else:
                        hard_rx[i] = v * scale_eff
                        scaled_rx_factor[i] *= scale_eff
                        pen = abs(scale_eff - 1.0)
                        conf_rx_link[i] *= clamp01(1.0 - 0.4 * clamp01(pen))
                        if pen >= 0.10 or scale_eff <= 0.90 or scale_eff >= clip_hi_i:
                            clip_hit_rx[i] = True
                        if pen > 0.08:
                            strong_scaled_rx[i] = True
                    applied += change_i
                return delta_in - applied

            if adjust_side == "tx" and sum_tx > 0:
                current = sum_tx
                target = sum_rx
                delta = target - current
                # Three tiers
                delta = apply_tier("tx", -1.0, 0.50, delta, "verylow")
                # Recompute context
                sum_tx = sum(hard_tx.get(i, 0.0) for i in if_list)
                sum_rx = sum(hard_rx.get(i, 0.0) for i in if_list)
                current = sum_tx
                target = sum_rx
                side_mag = max(current, target, EPS)
                if abs(delta) > 0.7 * TAU_ROUTER_LOCAL * side_mag:
                    delta = apply_tier("tx", 0.50, 0.70, delta, "low")
                if abs(delta) > 0.5 * TAU_ROUTER_LOCAL * side_mag:
                    delta = apply_tier("tx", 0.70, 0.85, delta, "moderate")

                # Final small uniform damped scaling if imbalance persists; exclude near-zero
                sum_tx2 = sum(hard_tx.get(i, 0.0) for i in if_list)
                sum_rx2 = sum(hard_rx.get(i, 0.0) for i in if_list)
                if rel_diff(sum_tx2, sum_rx2) > TAU_ROUTER_LOCAL and sum_tx2 > 0:
                    alpha = clamp(sum_rx2 / max(sum_tx2, EPS), 0.95, 1.05)
                    alpha_eff = 1.0 + 0.4 * (alpha - 1.0)
                    thr_router = router_thr_early.get(router, ZERO_THRESH)
                    for i in if_list:
                        v = hard_tx.get(i, 0.0)
                        if v < max(ZERO_THRESH, thr_router):
                            continue
                        hard_tx[i] = v * alpha_eff
                        scaled_tx_factor[i] *= alpha_eff
                        pen = abs(alpha_eff - 1.0)
                        conf_tx_link[i] *= clamp01(1.0 - 0.3 * clamp01(pen))
                        if pen > 0.08:
                            strong_scaled_tx[i] = True
                        if alpha_eff <= 0.90 or alpha_eff >= 1.10:
                            clip_hit_tx[i] = True

            elif adjust_side == "rx" and sum_rx > 0:
                current = sum_rx
                target = sum_tx
                delta = target - current
                delta = apply_tier("rx", -1.0, 0.50, delta, "verylow")
                sum_tx = sum(hard_tx.get(i, 0.0) for i in if_list)
                sum_rx = sum(hard_rx.get(i, 0.0) for i in if_list)
                current = sum_rx
                target = sum_tx
                side_mag = max(current, target, EPS)
                if abs(delta) > 0.7 * TAU_ROUTER_LOCAL * side_mag:
                    delta = apply_tier("rx", 0.50, 0.70, delta, "low")
                if abs(delta) > 0.5 * TAU_ROUTER_LOCAL * side_mag:
                    delta = apply_tier("rx", 0.70, 0.85, delta, "moderate")

                sum_tx2 = sum(hard_tx.get(i, 0.0) for i in if_list)
                sum_rx2 = sum(hard_rx.get(i, 0.0) for i in if_list)
                if rel_diff(sum_tx2, sum_rx2) > TAU_ROUTER_LOCAL and sum_rx2 > 0:
                    alpha = clamp(sum_tx2 / max(sum_rx2, EPS), 0.95, 1.05)
                    alpha_eff = 1.0 + 0.4 * (alpha - 1.0)
                    thr_router = router_thr_early.get(router, ZERO_THRESH)
                    for i in if_list:
                        v = hard_rx.get(i, 0.0)
                        if v < max(ZERO_THRESH, thr_router):
                            continue
                        hard_rx[i] = v * alpha_eff
                        scaled_rx_factor[i] *= alpha_eff
                        pen = abs(alpha_eff - 1.0)
                        conf_rx_link[i] *= clamp01(1.0 - 0.3 * clamp01(pen))
                        if pen > 0.08:
                            strong_scaled_rx[i] = True
                        if alpha_eff <= 0.90 or alpha_eff >= 1.10:
                            clip_hit_rx[i] = True

    # Stage 2.5: Post-projection gentle link re-sync (saturating gain; ±2% cap; tiny bilateral when safe)
    def router_imbalance(router_id: str) -> float:
        if not router_id or router_id not in router_ifaces:
            return 0.0
        ifs = router_ifaces[router_id]
        stx = sum(hard_tx.get(i, 0.0) for i in ifs)
        srx = sum(hard_rx.get(i, 0.0) for i in ifs)
        return rel_diff(stx, srx)

    def tiny_bilateral_allowed(ra: str, rb: str) -> bool:
        # Allow tiny bilateral nudge when both routers are within tolerance
        if ra not in router_ifaces or rb not in router_ifaces:
            return False
        ifs_a = router_ifaces[ra]
        ifs_b = router_ifaces[rb]
        n_act_a = max(1, sum(1 for i in ifs_a if max(hard_tx.get(i, 0.0), hard_rx.get(i, 0.0)) >= ZERO_THRESH))
        n_act_b = max(1, sum(1 for i in ifs_b if max(hard_tx.get(i, 0.0), hard_rx.get(i, 0.0)) >= ZERO_THRESH))
        tau_a = tau_router(n_act_a)
        tau_b = tau_router(n_act_b)
        return router_imbalance(ra) <= tau_a and router_imbalance(rb) <= tau_b

    for a, data in telemetry.items():
        b = peers.get(a)
        if not b or a > b:
            continue
        ra = router_of.get(a, "")
        rb = router_of.get(b, "")

        # a->b direction
        tx_a = hard_tx.get(a, 0.0)
        rx_b = hard_rx.get(b, 0.0)
        ca = clamp01(conf_tx_link.get(a, 0.6))
        cb = clamp01(conf_rx_link.get(b, 0.6))
        th_ab = tau_h_dir(tx_a, rx_b, ca, cb)
        diff_ab = rel_diff(tx_a, rx_b)
        if diff_ab > th_ab and max(tx_a, rx_b) >= ZERO_THRESH and not (strong_scaled_tx.get(a, False) or strong_scaled_rx.get(b, False)):
            mean_ab = 0.5 * (tx_a + rx_b)
            gap_norm = clamp01((diff_ab - th_ab) / max(th_ab, 1e-9))
            f_gain = 0.4 * sigmoid(5.0 * (gap_norm - 0.5))
            att = clamp01(1.0 - max(router_imbalance(ra), router_imbalance(rb)))
            f_gain *= att
            # Optional tiny bilateral when both low confidence and routers balanced
            if ca < 0.70 and cb < 0.70 and tiny_bilateral_allowed(ra, rb):
                f_bi = min(0.10, 0.5 * gap_norm, f_gain)
                tx_new = (1.0 - f_bi) * tx_a + f_bi * mean_ab
                rx_new = (1.0 - f_bi) * rx_b + f_bi * mean_ab
                tx_clip = clamp(tx_new, tx_a * 0.98, tx_a * 1.02)
                rx_clip = clamp(rx_new, rx_b * 0.98, rx_b * 1.02)
                hard_tx[a] = tx_clip
                hard_rx[b] = rx_clip
                conf_tx_link[a] *= 0.95
                conf_rx_link[b] *= 0.95
            else:
                if ca <= cb:
                    f = min(0.4, max(0.0, cb - ca)) * f_gain
                    tx_new = (1.0 - f) * tx_a + f * mean_ab
                    hard_tx[a] = clamp(tx_new, tx_a * 0.98, tx_a * 1.02)
                    conf_tx_link[a] *= 0.95
                else:
                    f = min(0.4, max(0.0, ca - cb)) * f_gain
                    rx_new = (1.0 - f) * rx_b + f * mean_ab
                    hard_rx[b] = clamp(rx_new, rx_b * 0.98, rx_b * 1.02)
                    conf_rx_link[b] *= 0.95

        # b->a direction
        tx_b = hard_tx.get(b, 0.0)
        rx_a = hard_rx.get(a, 0.0)
        cb_tx = clamp01(conf_tx_link.get(b, 0.6))
        ca_rx = clamp01(conf_rx_link.get(a, 0.6))
        th_ba = tau_h_dir(tx_b, rx_a, cb_tx, ca_rx)
        diff_ba = rel_diff(tx_b, rx_a)
        if diff_ba > th_ba and max(tx_b, rx_a) >= ZERO_THRESH and not (strong_scaled_tx.get(b, False) or strong_scaled_rx.get(a, False)):
            mean_ba = 0.5 * (tx_b + rx_a)
            gap_norm = clamp01((diff_ba - th_ba) / max(th_ba, 1e-9))
            f_gain = 0.4 * sigmoid(5.0 * (gap_norm - 0.5))
            att = clamp01(1.0 - max(router_imbalance(ra), router_imbalance(rb)))
            f_gain *= att
            if cb_tx < 0.70 and ca_rx < 0.70 and tiny_bilateral_allowed(ra, rb):
                f_bi = min(0.10, 0.5 * gap_norm, f_gain)
                tx_new = (1.0 - f_bi) * tx_b + f_bi * mean_ba
                rx_new = (1.0 - f_bi) * rx_a + f_bi * mean_ba
                hard_tx[b] = clamp(tx_new, tx_b * 0.98, tx_b * 1.02)
                hard_rx[a] = clamp(rx_new, rx_a * 0.98, rx_a * 1.02)
                conf_tx_link[b] *= 0.95
                conf_rx_link[a] *= 0.95
            else:
                if cb_tx <= ca_rx:
                    f = min(0.4, max(0.0, ca_rx - cb_tx)) * f_gain
                    tx_new = (1.0 - f) * tx_b + f * mean_ba
                    hard_tx[b] = clamp(tx_new, tx_b * 0.98, tx_b * 1.02)
                    conf_tx_link[b] *= 0.95
                else:
                    f = min(0.4, max(0.0, cb_tx - ca_rx)) * f_gain
                    rx_new = (1.0 - f) * rx_a + f * mean_ba
                    hard_rx[a] = clamp(rx_new, rx_a * 0.98, rx_a * 1.02)
                    conf_rx_link[a] *= 0.95

    # Stage 2.6: Conservation-preserving bundle finishing pass for parallel links (dispersion-aware)
    bundles: Dict[Tuple[str, str], List[Tuple[str, str]]] = {}
    seen_pairs = set()
    for a, data_a in telemetry.items():
        b = peers.get(a)
        if not b:
            continue
        key = tuple(sorted([a, b]))
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        ra = telemetry[a].get("local_router")
        rb = telemetry[b].get("local_router")
        if not isinstance(ra, str) or not isinstance(rb, str):
            continue
        rp = tuple(sorted([ra, rb]))
        bundles.setdefault(rp, []).append((a, b))

    def median(vals: List[float]) -> float:
        s = sorted(vals)
        n = len(s)
        if n == 0:
            return 0.0
        if n % 2 == 1:
            return s[n // 2]
        return 0.5 * (s[n // 2 - 1] + s[n // 2])

    def mad(vals: List[float], med: float) -> float:
        if not vals:
            return 0.0
        dev = [abs(v - med) for v in vals]
        return median(dev)

    def bundle_direction_align(pairs: List[Tuple[str, str]], ab_dir: bool = True):
        if not pairs:
            return
        es = []
        rates = []
        confs = []
        idxs = []
        for (a, b) in pairs:
            if ab_dir:
                tx = hard_tx.get(a, 0.0)
                rx = hard_rx.get(b, 0.0)
                conf = 0.5 * (clamp01(conf_tx_link.get(a, 0.6)) + clamp01(conf_rx_link.get(b, 0.6)))
            else:
                tx = hard_tx.get(b, 0.0)
                rx = hard_rx.get(a, 0.0)
                conf = 0.5 * (clamp01(conf_tx_link.get(b, 0.6)) + clamp01(conf_rx_link.get(a, 0.6)))
            if max(tx, rx) < ZERO_THRESH:
                continue
            e = tx - rx
            rate = max(tx, rx)
            es.append(e)
            rates.append(rate)
            confs.append(conf)
            idxs.append((a, b))
        m = len(es)
        if m <= 1:
            return
        e_med = median(es)
        mad_val = mad(es, e_med)
        k_huber = 1.5 * mad_val if mad_val > 0 else max(1.0, 0.01 * sum(rates) / m)
        # Base deltas with Huber weights and dispersion-aware gamma
        deltas = []
        for i in range(m):
            e = es[i]
            rate = rates[i]
            # Huber weight
            r = abs(e - e_med)
            w_h = 1.0 if r <= k_huber else (k_huber / r if r > 0 else 1.0)
            # mismatch
            a, b = idxs[i]
            if ab_dir:
                tx = hard_tx.get(a, 0.0)
                rx = hard_rx.get(b, 0.0)
            else:
                tx = hard_tx.get(b, 0.0)
                rx = hard_rx.get(a, 0.0)
            mismatch = rel_diff(tx, rx)
            gamma = min(0.30, 0.5 * TAU_H / max(mismatch, 1e-9)) * w_h
            deltas.append(-gamma * (e - e_med))
        # Recenter to bundle zero-sum
        mean_d = sum(deltas) / m
        deltas = [d - mean_d for d in deltas]
        # Clip per-link delta to 2-4% depending on bundle size
        clip_frac = 0.02 if m <= 2 else 0.03 if m <= 4 else 0.04
        # Apply equal/opposite on peer
        for i, (a, b) in enumerate(idxs):
            di = clamp(deltas[i], -clip_frac * rates[i], clip_frac * rates[i])
            if ab_dir:
                hard_tx[a] = max(0.0, hard_tx.get(a, 0.0) + di)
                hard_rx[b] = max(0.0, hard_rx.get(b, 0.0) - di)
            else:
                hard_tx[b] = max(0.0, hard_tx.get(b, 0.0) + di)
                hard_rx[a] = max(0.0, hard_rx.get(a, 0.0) - di)

    for rp, pairs in bundles.items():
        bundle_direction_align(pairs, ab_dir=True)
        bundle_direction_align(pairs, ab_dir=False)

    # Soft-zero rule (late pass): dynamic thresholds based on router totals
    processed_pairs = set()
    router_thr_late = compute_router_thresholds()
    for a, data_a in telemetry.items():
        b = data_a.get('connected_to')
        if not isinstance(b, str) or b not in telemetry:
            continue
        key = tuple(sorted([a, b]))
        if key in processed_pairs:
            continue
        processed_pairs.add(key)
        ra = router_of.get(a)
        rb = router_of.get(b)
        thr_pair = 1.5 * max(router_thr_late.get(ra, ZERO_THRESH), router_thr_late.get(rb, ZERO_THRESH))
        tx_a = hard_tx.get(a, 0.0)
        rx_b = hard_rx.get(b, 0.0)
        tx_b = hard_tx.get(b, 0.0)
        rx_a = hard_rx.get(a, 0.0)
        if max(tx_a, rx_b, tx_b, rx_a) < thr_pair:
            def tau_for_router(rid: str) -> float:
                if rid not in router_ifaces:
                    return TAU_ROUTER
                ifs = router_ifaces[rid]
                n_tx = sum(1 for i in ifs if hard_tx.get(i, 0.0) >= ZERO_THRESH)
                n_rx = sum(1 for i in ifs if hard_rx.get(i, 0.0) >= ZERO_THRESH)
                return tau_router(max(n_tx, n_rx))
            if router_imbalance(ra) <= tau_for_router(ra) and router_imbalance(rb) <= tau_for_router(rb):
                hard_tx[a] = 0.0
                hard_rx[b] = 0.0
                hard_tx[b] = 0.0
                hard_rx[a] = 0.0
                conf_tx_link[a] = max(conf_tx_link.get(a, 0.6), 0.95)
                conf_rx_link[b] = max(conf_rx_link.get(b, 0.6), 0.95)
                conf_tx_link[b] = max(conf_tx_link.get(b, 0.6), 0.95)
                conf_rx_link[a] = max(conf_rx_link.get(a, 0.6), 0.95)

    # Status repair (conservative and symmetry-aware)
    repaired_status: Dict[str, str] = {}
    status_conf: Dict[str, float] = {}
    processed = set()
    for if_id in telemetry.keys():
        if if_id in processed:
            continue
        peer = peers.get(if_id)
        s_local = status.get(if_id, "unknown")
        if not peer:
            repaired_status[if_id] = s_local
            status_conf[if_id] = 0.95
            processed.add(if_id)
            continue

        s_peer = status.get(peer, "unknown")
        rep_local = s_local
        rep_peer = s_peer
        c_local = 0.95
        c_peer = 0.95

        # Both report down
        if s_local == "down" and s_peer == "down":
            rep_local = "down"
            rep_peer = "down"
            c_local = 0.98
            c_peer = 0.98
        elif s_local != s_peer:
            # If any traffic on link, set both up
            link_has_traffic = (hard_tx[if_id] >= ZERO_THRESH or hard_rx[if_id] >= ZERO_THRESH or
                                hard_tx.get(peer, 0.0) >= ZERO_THRESH or hard_rx.get(peer, 0.0) >= ZERO_THRESH)
            if link_has_traffic:
                rep_local = "up"
                rep_peer = "up"
                c_local = 0.7
                c_peer = 0.7
            else:
                # Ambiguous mismatch; keep as-is but lower confidence
                c_local = 0.6
                c_peer = 0.6
        else:
            rep_local = s_local
            rep_peer = s_peer
            c_local = 0.95
            c_peer = 0.95

        repaired_status[if_id] = rep_local
        repaired_status[peer] = rep_peer
        status_conf[if_id] = c_local
        status_conf[peer] = c_peer
        processed.add(if_id)
        processed.add(peer)

    # Calibrate confidence using post-projection invariants
    # 1) Router imbalance AFTER projection
    router_imbalance_after: Dict[str, float] = {}
    for router, if_list in router_ifaces.items():
        if not if_list:
            router_imbalance_after[router] = 0.0
            continue
        sum_tx_after = sum(hard_tx.get(i, 0.0) for i in if_list)
        sum_rx_after = sum(hard_rx.get(i, 0.0) for i in if_list)
        router_imbalance_after[router] = rel_diff(sum_tx_after, sum_rx_after)

    # 2) Final per-direction symmetry residuals AFTER all adjustments
    post_mismatch_tx_dir: Dict[str, float] = {}
    post_mismatch_rx_dir: Dict[str, float] = {}
    for if_id in telemetry.keys():
        peer = peers.get(if_id)
        if peer:
            post_mismatch_tx_dir[if_id] = rel_diff(hard_tx.get(if_id, 0.0), hard_rx.get(peer, 0.0))
            post_mismatch_rx_dir[if_id] = rel_diff(hard_rx.get(if_id, 0.0), hard_tx.get(peer, 0.0))
        else:
            # No redundant signal available: use moderate default uncertainty
            post_mismatch_tx_dir[if_id] = 0.4
            post_mismatch_rx_dir[if_id] = 0.4

    # Stability term per router and direction: stab_i = clamp01(1 - 0.5 * share_i)
    stab_tx: Dict[str, float] = {}
    stab_rx: Dict[str, float] = {}
    for r, ifs in router_ifaces.items():
        sum_tx_r = sum(max(hard_tx.get(i, 0.0), 0.0) for i in ifs)
        sum_rx_r = sum(max(hard_rx.get(i, 0.0), 0.0) for i in ifs)
        for i in ifs:
            vtx = max(hard_tx.get(i, 0.0), 0.0)
            vrx = max(hard_rx.get(i, 0.0), 0.0)
            share_tx = vtx / sum_tx_r if sum_tx_r > EPS else 0.0
            share_rx = vrx / sum_rx_r if sum_rx_r > EPS else 0.0
            stab_tx[i] = clamp01(1.0 - 0.5 * share_tx)
            stab_rx[i] = clamp01(1.0 - 0.5 * share_rx)

    # Bundle-consistency term: closeness to bundle median residual
    bcons_tx: Dict[str, float] = {}
    bcons_rx: Dict[str, float] = {}
    bundles_cons: Dict[Tuple[str, str], List[Tuple[str, str]]] = {}
    seen_pairs_cons = set()
    for a, data_a in telemetry.items():
        b = peers.get(a)
        if not b:
            continue
        key = tuple(sorted([a, b]))
        if key in seen_pairs_cons:
            continue
        seen_pairs_cons.add(key)
        ra = telemetry[a].get("local_router")
        rb = telemetry[b].get("local_router")
        if isinstance(ra, str) and isinstance(rb, str):
            rp = tuple(sorted([ra, rb]))
            bundles_cons.setdefault(rp, []).append((a, b))
    for rp, pairs in bundles_cons.items():
        if not pairs:
            continue
        e_ab_list = []
        e_ba_list = []
        for (a, b) in pairs:
            e_ab_list.append(hard_tx.get(a, 0.0) - hard_rx.get(b, 0.0))
            e_ba_list.append(hard_tx.get(b, 0.0) - hard_rx.get(a, 0.0))
        med_ab = median(e_ab_list)
        med_ba = median(e_ba_list)
        for (a, b) in pairs:
            tx_a = max(hard_tx.get(a, 0.0), 0.0)
            rx_b = max(hard_rx.get(b, 0.0), 0.0)
            rate_ab = max(tx_a, rx_b, 1.0)
            e_ab = tx_a - rx_b
            bcons_tx[a] = clamp01(1.0 - abs(e_ab - med_ab) / (abs(med_ab) + rate_ab))
            bcons_rx[b] = clamp01(1.0 - abs(e_ab - med_ab) / (abs(med_ab) + rate_ab))
            tx_b = max(hard_tx.get(b, 0.0), 0.0)
            rx_a = max(hard_rx.get(a, 0.0), 0.0)
            rate_ba = max(tx_b, rx_a, 1.0)
            e_ba = tx_b - rx_a
            bcons_tx[b] = clamp01(1.0 - abs(e_ba - med_ba) / (abs(med_ba) + rate_ba))
            bcons_rx[a] = clamp01(1.0 - abs(e_ba - med_ba) / (abs(med_ba) + rate_ba))
    for if_id in telemetry.keys():
        if if_id not in bcons_tx:
            bcons_tx[if_id] = 0.8
        if if_id not in bcons_rx:
            bcons_rx[if_id] = 0.8
        if if_id not in stab_tx:
            stab_tx[if_id] = 0.85
        if if_id not in stab_rx:
            stab_rx[if_id] = 0.85

    # Compose final results with calibrated confidences
    result: Dict[str, Dict[str, Tuple]] = {}
    for if_id, data in telemetry.items():
        rep_tx = hard_tx.get(if_id, orig_tx[if_id])
        rep_rx = hard_rx.get(if_id, orig_rx[if_id])

        # Compute change magnitude
        change_tx = rel_diff(orig_tx[if_id], rep_tx)
        change_rx = rel_diff(orig_rx[if_id], rep_rx)

        # Pre-fusion mismatch (redundancy before hardening)
        pre_tx = pre_mismatch_tx.get(if_id, 0.4)
        pre_rx = pre_mismatch_rx.get(if_id, 0.4)

        # Post-fusion symmetry agreement (redundancy after final hardening)
        fin_sym_tx = clamp01(1.0 - post_mismatch_tx_dir.get(if_id, 0.4))
        fin_sym_rx = clamp01(1.0 - post_mismatch_rx_dir.get(if_id, 0.4))

        # Improvement credit (how much mismatch improved)
        imp_tx = clamp01(pre_tx - post_mismatch_tx_dir.get(if_id, 0.4))
        imp_rx = clamp01(pre_rx - post_mismatch_rx_dir.get(if_id, 0.4))

        # Router context penalty AFTER projection
        r = router_of.get(if_id, None)
        router_penalty_after = router_imbalance_after.get(r, 0.0) if r is not None else 0.0
        router_factor_after = clamp01(1.0 - min(0.5, router_penalty_after))

        base_tx_conf = clamp01(conf_tx_link.get(if_id, 0.6))
        base_rx_conf = clamp01(conf_rx_link.get(if_id, 0.6))

        red_tx = clamp01(1.0 - pre_tx)
        red_rx = clamp01(1.0 - pre_rx)

        ch_tx = clamp01(1.0 - change_tx)
        ch_rx = clamp01(1.0 - change_rx)

        # Scale penalty term
        scale_tx_term = clamp01(1.0 - min(0.5, abs(scaled_tx_factor.get(if_id, 1.0) - 1.0)))
        scale_rx_term = clamp01(1.0 - min(0.5, abs(scaled_rx_factor.get(if_id, 1.0) - 1.0)))
        stab_term_tx = clamp01(stab_tx.get(if_id, 0.85))
        stab_term_rx = clamp01(stab_rx.get(if_id, 0.85))
        bcons_term_tx = clamp01(bcons_tx.get(if_id, 0.8))
        bcons_term_rx = clamp01(bcons_rx.get(if_id, 0.8))

        conf_tx_final = clamp01(
            0.18 * base_tx_conf +
            0.16 * red_tx +
            0.22 * fin_sym_tx +
            0.10 * ch_tx +
            0.10 * router_factor_after +
            0.05 * scale_tx_term +
            0.08 * imp_tx +
            0.05 * stab_term_tx +
            0.06 * bcons_term_tx
        )
        conf_rx_final = clamp01(
            0.18 * base_rx_conf +
            0.16 * red_rx +
            0.22 * fin_sym_rx +
            0.10 * ch_rx +
            0.10 * router_factor_after +
            0.05 * scale_rx_term +
            0.08 * imp_rx +
            0.05 * stab_term_rx +
            0.06 * bcons_term_rx
        )

        # Confidence refinements: clip-hit penalty and untouched/strong-scale adjustments
        if abs(scaled_tx_factor.get(if_id, 1.0) - 1.0) >= 0.10:
            conf_tx_final *= 0.95
        if abs(scaled_rx_factor.get(if_id, 1.0) - 1.0) >= 0.10:
            conf_rx_final *= 0.95
        if clip_hit_tx.get(if_id, False):
            conf_tx_final *= 0.95
        if clip_hit_rx.get(if_id, False):
            conf_rx_final *= 0.95
        if strong_scaled_tx.get(if_id, False):
            conf_tx_final *= 0.97
        if strong_scaled_rx.get(if_id, False):
            conf_rx_final *= 0.97
        # Penalize when a clip hit occurred and router imbalance worsened
        r_before = router_imbalance_before.get(router_of.get(if_id, ""), 0.0)
        r_after = router_imbalance_after.get(router_of.get(if_id, ""), 0.0)
        if clip_hit_tx.get(if_id, False) and r_after > r_before + 1e-12:
            conf_tx_final *= 0.97
        if clip_hit_rx.get(if_id, False) and r_after > r_before + 1e-12:
            conf_rx_final *= 0.97
        # Untouched small-change high-symmetry boost
        if change_tx < 0.01 and fin_sym_tx >= (1.0 - TAU_H):
            conf_tx_final = min(0.98, conf_tx_final + 0.02)
        if change_rx < 0.01 and fin_sym_rx >= (1.0 - TAU_H):
            conf_rx_final = min(0.98, conf_rx_final + 0.02)

        # Status enforcement: down implies zero counters
        rep_status = repaired_status.get(if_id, status.get(if_id, "unknown"))
        conf_status = status_conf.get(if_id, 0.9)
        if rep_status == "down":
            rep_tx = 0.0
            rep_rx = 0.0
            if orig_tx[if_id] >= ZERO_THRESH or orig_rx[if_id] >= ZERO_THRESH:
                conf_tx_final = min(conf_tx_final, 0.7)
                conf_rx_final = min(conf_rx_final, 0.7)
            else:
                conf_tx_final = max(conf_tx_final, 0.9)
                conf_rx_final = max(conf_rx_final, 0.9)

        # Assemble output record
        out = {}
        out["rx_rate"] = (orig_rx[if_id], rep_rx, conf_rx_final)
        out["tx_rate"] = (orig_tx[if_id], rep_tx, conf_tx_final)
        out["interface_status"] = (status[if_id], rep_status, conf_status)

        # Copy metadata unchanged
        out["connected_to"] = data.get("connected_to")
        out["local_router"] = data.get("local_router")
        out["remote_router"] = data.get("remote_router")

        result[if_id] = out

    # Router-aware confidence smoothing for stable directions
    # Compute router mean confidences per direction
    mean_conf_tx: Dict[str, float] = {}
    mean_conf_rx: Dict[str, float] = {}
    for r, ifs in router_ifaces.items():
        if not ifs:
            mean_conf_tx[r] = 0.8
            mean_conf_rx[r] = 0.8
            continue
        tx_cs = [clamp01(result[i]['tx_rate'][2]) for i in ifs if i in result]
        rx_cs = [clamp01(result[i]['rx_rate'][2]) for i in ifs if i in result]
        mean_conf_tx[r] = sum(tx_cs) / len(tx_cs) if tx_cs else 0.8
        mean_conf_rx[r] = sum(rx_cs) / len(rx_cs) if rx_cs else 0.8

    for if_id in telemetry.keys():
        if if_id not in result:
            continue
        r = router_of.get(if_id, None)
        if r is None:
            continue
        # Only smooth stable (small scale) directions
        if abs(scaled_tx_factor.get(if_id, 1.0) - 1.0) < 0.05:
            old = clamp01(result[if_id]['tx_rate'][2])
            sm = 0.85 * old + 0.15 * clamp01(mean_conf_tx.get(r, old))
            result[if_id]['tx_rate'] = (result[if_id]['tx_rate'][0], result[if_id]['tx_rate'][1], clamp01(sm))
        if abs(scaled_rx_factor.get(if_id, 1.0) - 1.0) < 0.05:
            old = clamp01(result[if_id]['rx_rate'][2])
            sm = 0.85 * old + 0.15 * clamp01(mean_conf_rx.get(r, old))
            result[if_id]['rx_rate'] = (result[if_id]['rx_rate'][0], result[if_id]['rx_rate'][1], clamp01(sm))

    # Final confidence touch-up: incorporate final symmetry residuals and peer smoothing when both ends are up
    def _safe_get_rate(tup):
        return safe_rate(tup[1]) if isinstance(tup, tuple) and len(tup) >= 2 else 0.0

    for i, data in telemetry.items():
        peer = data.get('connected_to')
        if not isinstance(peer, str) or peer not in telemetry or i not in result or peer not in result:
            continue
        if result[i]['interface_status'][1] != 'up' or result[peer]['interface_status'][1] != 'up':
            continue
        mis_tx = rel_diff(_safe_get_rate(result[i]['tx_rate']), _safe_get_rate(result[peer]['rx_rate']))
        mis_rx = rel_diff(_safe_get_rate(result[i]['rx_rate']), _safe_get_rate(result[peer]['tx_rate']))
        old_tx_c = clamp01(result[i]['tx_rate'][2])
        old_rx_c = clamp01(result[i]['rx_rate'][2])
        base_tx = clamp01(0.70 * old_tx_c + 0.30 * clamp01(1.0 - mis_tx))
        base_rx = clamp01(0.70 * old_rx_c + 0.30 * clamp01(1.0 - mis_rx))
        peer_rx_c = clamp01(result[peer]['rx_rate'][2])
        peer_tx_c = clamp01(result[peer]['tx_rate'][2])
        result[i]['tx_rate'] = (result[i]['tx_rate'][0], result[i]['tx_rate'][1],
                                clamp01(0.90 * base_tx + 0.10 * peer_rx_c))
        result[i]['rx_rate'] = (result[i]['rx_rate'][0], result[i]['rx_rate'][1],
                                clamp01(0.90 * base_rx + 0.10 * peer_tx_c))

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