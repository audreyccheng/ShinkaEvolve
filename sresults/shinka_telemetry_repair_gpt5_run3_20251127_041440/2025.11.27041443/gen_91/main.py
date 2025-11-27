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
    TAU_ROUTER = 0.05     # router imbalance tolerance 5% (more conservative)
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

    # Stage 1: Link hardening with adaptive fusion
    hard_tx: Dict[str, float] = {}
    hard_rx: Dict[str, float] = {}
    conf_tx_link: Dict[str, float] = {}
    conf_rx_link: Dict[str, float] = {}
    pre_mismatch_tx: Dict[str, float] = {}
    pre_mismatch_rx: Dict[str, float] = {}

    visited = set()

    def fuse_direction(v_local: float, v_peer: float, s_local: str, s_peer: str) -> Tuple[float, float]:
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

        # Large mismatch: prefer the more plausible side
        # If one is near-zero and the other is not, take the non-zero
        if v_local < ZERO_THRESH and v_peer >= ZERO_THRESH:
            return v_peer, clamp01(1.0 - mismatch)
        if v_peer < ZERO_THRESH and v_local >= ZERO_THRESH:
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

        fused_ab, c_ab = fuse_direction(a_tx, b_rx, sa, sb)
        fused_ba, c_ba = fuse_direction(b_tx, a_rx, sb, sa)

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

    # Stage 2: Conservative router-level flow projection (two-tier targeted scaling with dominance cap)
    router_imbalance_before: Dict[str, float] = {}
    scaled_tx_factor: Dict[str, float] = {if_id: 1.0 for if_id in telemetry}
    scaled_rx_factor: Dict[str, float] = {if_id: 1.0 for if_id in telemetry}
    # Track directions that received strong scaling to guard against double-adjustment in Stage 2.5
    strong_scaled_tx: Dict[str, bool] = {if_id: False for if_id in telemetry}
    strong_scaled_rx: Dict[str, bool] = {if_id: False for if_id in telemetry}
    # Clip-hit trackers for confidence calibration
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

        # Adaptive router tolerance based on number of active interfaces
        n_active_tx = sum(1 for i in if_list if hard_tx.get(i, 0.0) >= ZERO_THRESH)
        n_active_rx = sum(1 for i in if_list if hard_rx.get(i, 0.0) >= ZERO_THRESH)
        TAU_ROUTER_LOCAL = tau_router(max(n_active_tx, n_active_rx))

        if mismatch > TAU_ROUTER_LOCAL:
            # Choose side with lower aggregate link confidence to adjust
            c_tx_total = sum(conf_tx_link.get(i, 0.5) for i in if_list)
            c_rx_total = sum(conf_rx_link.get(i, 0.5) for i in if_list)
            adjust_side = "tx" if c_tx_total < c_rx_total else "rx"

            if adjust_side == "tx" and sum_tx > 0:
                target = sum_rx
                current = sum_tx
                delta = target - current

                def apply_tier_tx(conf_lo: float, conf_hi: float, clip_hi: float, delta_in: float) -> float:
                    elig = []
                    for i in if_list:
                        v = hard_tx.get(i, 0.0)
                        c = clamp01(conf_tx_link.get(i, 0.6))
                        if v >= ZERO_THRESH and conf_lo <= c < conf_hi:
                            elig.append(i)
                    if not elig:
                        return delta_in
                    # Weights favor low-confidence and higher-rate interfaces
                    weights = {i: (1.0 - clamp01(conf_tx_link.get(i, 0.6))) * hard_tx.get(i, 0.0) for i in elig}
                    denom = sum(hard_tx.get(i, 0.0) * weights[i] for i in elig)
                    if denom < EPS:
                        return delta_in
                    k = delta_in / (0.6 * denom)
                    applied = 0.0
                    cap_abs = 0.5 * abs(delta_in) if len(elig) >= 2 else None
                    for i in elig:
                        v = hard_tx.get(i, 0.0)
                        w = weights[i]
                        scale_i = 1.0 + 0.6 * (k * w)
                        # Clip factor
                        raw_scale = scale_i
                        scale_i = max(0.90, min(clip_hi, scale_i))
                        change_i = v * (scale_i - 1.0)
                        # Dominance cap
                        if cap_abs is not None and abs(change_i) > cap_abs:
                            scale_i = 1.0 + math.copysign(cap_abs, change_i) / max(v, EPS)
                            change_i = v * (scale_i - 1.0)
                        hard_tx[i] = v * scale_i
                        scaled_tx_factor[i] *= scale_i
                        pen = clamp01(abs(scale_i - 1.0))
                        conf_tx_link[i] *= clamp01(1.0 - 0.4 * pen)
                        if pen > 0.08:
                            strong_scaled_tx[i] = True
                        if scale_i <= 0.90 or scale_i >= clip_hi or abs(raw_scale - scale_i) > 1e-12:
                            clip_hit_tx[i] = True
                        applied += change_i
                    return delta_in - applied

                # Tier 1: low confidence < 0.70 with slightly wider clip to 1.12
                delta = apply_tier_tx(conf_lo=-1.0, conf_hi=0.70, clip_hi=1.12, delta_in=delta)
                # Recompute residual context
                sum_tx = sum(hard_tx.get(i, 0.0) for i in if_list)
                sum_rx = sum(hard_rx.get(i, 0.0) for i in if_list)
                current = sum_tx
                target = sum_rx
                side_mag = max(current, target, EPS)

                # Tier 2: moderate confidence [0.70, 0.85) if residual still significant
                if abs(delta) > 0.5 * TAU_ROUTER_LOCAL * side_mag:
                    delta = apply_tier_tx(conf_lo=0.70, conf_hi=0.85, clip_hi=1.10, delta_in=delta)

                # Final small uniform damped scaling if imbalance persists
                sum_tx2 = sum(hard_tx.get(i, 0.0) for i in if_list)
                sum_rx2 = sum(hard_rx.get(i, 0.0) for i in if_list)
                if rel_diff(sum_tx2, sum_rx2) > TAU_ROUTER_LOCAL and sum_tx2 > 0:
                    alpha = sum_rx2 / max(sum_tx2, EPS)
                    alpha = max(0.95, min(1.05, alpha))
                    alpha_eff = 1.0 + 0.4 * (alpha - 1.0)
                    for i in if_list:
                        v = hard_tx.get(i, 0.0)
                        if v < ZERO_THRESH:
                            continue
                        hard_tx[i] = v * alpha_eff
                        scaled_tx_factor[i] *= alpha_eff
                        pen = clamp01(abs(alpha_eff - 1.0))
                        conf_tx_link[i] *= clamp01(1.0 - 0.3 * pen)
                        if pen > 0.08:
                            strong_scaled_tx[i] = True
                        if alpha_eff <= 0.90 or alpha_eff >= 1.10:
                            clip_hit_tx[i] = True

            elif adjust_side == "rx" and sum_rx > 0:
                target = sum_tx
                current = sum_rx
                delta = target - current

                def apply_tier_rx(conf_lo: float, conf_hi: float, clip_hi: float, delta_in: float) -> float:
                    elig = []
                    for i in if_list:
                        v = hard_rx.get(i, 0.0)
                        c = clamp01(conf_rx_link.get(i, 0.6))
                        if v >= ZERO_THRESH and conf_lo <= c < conf_hi:
                            elig.append(i)
                    if not elig:
                        return delta_in
                    weights = {i: (1.0 - clamp01(conf_rx_link.get(i, 0.6))) * hard_rx.get(i, 0.0) for i in elig}
                    denom = sum(hard_rx.get(i, 0.0) * weights[i] for i in elig)
                    if denom < EPS:
                        return delta_in
                    k = delta_in / (0.6 * denom)
                    applied = 0.0
                    cap_abs = 0.5 * abs(delta_in) if len(elig) >= 2 else None
                    for i in elig:
                        v = hard_rx.get(i, 0.0)
                        w = weights[i]
                        scale_i = 1.0 + 0.6 * (k * w)
                        raw_scale = scale_i
                        scale_i = max(0.90, min(clip_hi, scale_i))
                        change_i = v * (scale_i - 1.0)
                        if cap_abs is not None and abs(change_i) > cap_abs:
                            scale_i = 1.0 + math.copysign(cap_abs, change_i) / max(v, EPS)
                            change_i = v * (scale_i - 1.0)
                        hard_rx[i] = v * scale_i
                        scaled_rx_factor[i] *= scale_i
                        pen = clamp01(abs(scale_i - 1.0))
                        conf_rx_link[i] *= clamp01(1.0 - 0.4 * pen)
                        if pen > 0.08:
                            strong_scaled_rx[i] = True
                        if scale_i <= 0.90 or scale_i >= clip_hi or abs(raw_scale - scale_i) > 1e-12:
                            clip_hit_rx[i] = True
                        applied += change_i
                    return delta_in - applied

                # Tier 1 and Tier 2 as above
                delta = apply_tier_rx(conf_lo=-1.0, conf_hi=0.70, clip_hi=1.12, delta_in=delta)
                sum_tx = sum(hard_tx.get(i, 0.0) for i in if_list)
                sum_rx = sum(hard_rx.get(i, 0.0) for i in if_list)
                current = sum_rx
                target = sum_tx
                side_mag = max(current, target, EPS)
                if abs(delta) > 0.5 * TAU_ROUTER_LOCAL * side_mag:
                    delta = apply_tier_rx(conf_lo=0.70, conf_hi=0.85, clip_hi=1.10, delta_in=delta)

                sum_tx2 = sum(hard_tx.get(i, 0.0) for i in if_list)
                sum_rx2 = sum(hard_rx.get(i, 0.0) for i in if_list)
                if rel_diff(sum_tx2, sum_rx2) > TAU_ROUTER_LOCAL and sum_rx2 > 0:
                    alpha = sum_tx2 / max(sum_rx2, EPS)
                    alpha = max(0.95, min(1.05, alpha))
                    alpha_eff = 1.0 + 0.4 * (alpha - 1.0)
                    for i in if_list:
                        v = hard_rx.get(i, 0.0)
                        if v < ZERO_THRESH:
                            continue
                        hard_rx[i] = v * alpha_eff
                        scaled_rx_factor[i] *= alpha_eff
                        pen = clamp01(abs(alpha_eff - 1.0))
                        conf_rx_link[i] *= clamp01(1.0 - 0.3 * pen)
                        if pen > 0.08:
                            strong_scaled_rx[i] = True
                        if alpha_eff <= 0.90 or alpha_eff >= 1.10:
                            clip_hit_rx[i] = True

    # Stage 2.5: Post-projection gentle link re-sync (only adjust lower-confidence side)
    # Helper for router imbalance attenuation in re-sync
    def router_imbalance(router_id: str) -> float:
        if not router_id or router_id not in router_ifaces:
            return 0.0
        ifs = router_ifaces[router_id]
        stx = sum(hard_tx.get(i, 0.0) for i in ifs)
        srx = sum(hard_rx.get(i, 0.0) for i in ifs)
        return rel_diff(stx, srx)

    for a, data in telemetry.items():
        b = peers.get(a)
        if not b or a > b:
            # Process each pair once; ensure deterministic order by a > b check
            continue
        # a->b direction
        tx_a = hard_tx.get(a, 0.0)
        rx_b = hard_rx.get(b, 0.0)
        diff_ab = rel_diff(tx_a, rx_b)
        ca = clamp01(conf_tx_link.get(a, 0.6))
        cb = clamp01(conf_rx_link.get(b, 0.6))
        th_ab = tau_h_dir(tx_a, rx_b, ca, cb)
        if diff_ab > th_ab and max(tx_a, rx_b) >= ZERO_THRESH and not (strong_scaled_tx.get(a, False) or strong_scaled_rx.get(b, False)):
            mean_ab = 0.5 * (tx_a + rx_b)
            if ca < cb:
                # Nudge lower-confidence side proportionally to confidence gap with router-imbalance attenuation
                gap = max(0.0, cb - ca)
                f = min(0.4, gap)
                att = clamp01(1.0 - router_imbalance(router_of.get(a, "")))
                f *= att
                hard_tx[a] = (1.0 - f) * tx_a + f * mean_ab
                conf_tx_link[a] *= 0.95
            elif cb < ca:
                gap = max(0.0, ca - cb)
                f = min(0.4, gap)
                att = clamp01(1.0 - router_imbalance(router_of.get(b, "")))
                f *= att
                hard_rx[b] = (1.0 - f) * rx_b + f * mean_ab
                conf_rx_link[b] *= 0.95
            else:
                # Both similar confidence and low: small cautious symmetric nudge
                if ca < 0.7:
                    f = 0.15
                    hard_tx[a] = (1.0 - f) * tx_a + f * mean_ab
                    hard_rx[b] = (1.0 - f) * rx_b + f * mean_ab
                    conf_tx_link[a] *= 0.93
                    conf_rx_link[b] *= 0.93
        # b->a direction
        tx_b = hard_tx.get(b, 0.0)
        rx_a = hard_rx.get(a, 0.0)
        diff_ba = rel_diff(tx_b, rx_a)
        cb_tx = clamp01(conf_tx_link.get(b, 0.6))
        ca_rx = clamp01(conf_rx_link.get(a, 0.6))
        th_ba = tau_h_dir(tx_b, rx_a, cb_tx, ca_rx)
        if diff_ba > th_ba and max(tx_b, rx_a) >= ZERO_THRESH and not (strong_scaled_tx.get(b, False) or strong_scaled_rx.get(a, False)):
            mean_ba = 0.5 * (tx_b + rx_a)
            if cb_tx < ca_rx:
                gap = max(0.0, ca_rx - cb_tx)
                f = min(0.4, gap)
                att = clamp01(1.0 - router_imbalance(router_of.get(b, "")))
                f *= att
                hard_tx[b] = (1.0 - f) * tx_b + f * mean_ba
                conf_tx_link[b] *= 0.95
            elif ca_rx < cb_tx:
                gap = max(0.0, cb_tx - ca_rx)
                f = min(0.4, gap)
                att = clamp01(1.0 - router_imbalance(router_of.get(a, "")))
                f *= att
                hard_rx[a] = (1.0 - f) * rx_a + f * mean_ba
                conf_rx_link[a] *= 0.95
            else:
                if cb_tx < 0.7:
                    f = 0.15
                    hard_tx[b] = (1.0 - f) * tx_b + f * mean_ba
                    hard_rx[a] = (1.0 - f) * rx_a + f * mean_ba
                    conf_tx_link[b] *= 0.93
                    conf_rx_link[a] *= 0.93

    # Soft-zero rule: if both directions on a link are tiny, snap all four to 0 with high confidence
    processed_pairs = set()
    for a, data_a in telemetry.items():
        b = data_a.get('connected_to')
        if not isinstance(b, str) or b not in telemetry:
            continue
        key = tuple(sorted([a, b]))
        if key in processed_pairs:
            continue
        processed_pairs.add(key)
        tx_a = hard_tx.get(a, 0.0)
        rx_b = hard_rx.get(b, 0.0)
        tx_b = hard_tx.get(b, 0.0)
        rx_a = hard_rx.get(a, 0.0)
        if max(tx_a, rx_b, tx_b, rx_a) < 2.0 * ZERO_THRESH:
            # Only snap to zero when both adjacent routers are reasonably balanced under adaptive tau
            ra = router_of.get(a)
            rb = router_of.get(b)
            def tau_for_router(rid: str) -> float:
                if rid not in router_ifaces:
                    return 0.05
                ifs = router_ifaces[rid]
                n_tx = sum(1 for i in ifs if hard_tx.get(i, 0.0) >= ZERO_THRESH)
                n_rx = sum(1 for i in ifs if hard_rx.get(i, 0.0) >= ZERO_THRESH)
                return tau_router(max(n_tx, n_rx))
            def router_imbalance_local(rid: str) -> float:
                if not rid or rid not in router_ifaces:
                    return 0.0
                ifs = router_ifaces[rid]
                stx = sum(hard_tx.get(i, 0.0) for i in ifs)
                srx = sum(hard_rx.get(i, 0.0) for i in ifs)
                return rel_diff(stx, srx)
            if router_imbalance_local(ra) <= tau_for_router(ra) and router_imbalance_local(rb) <= tau_for_router(rb):
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
            share_tx = (vtx / sum_tx_r) if sum_tx_r > EPS else 0.0
            share_rx = (vrx / sum_rx_r) if sum_rx_r > EPS else 0.0
            stab_tx[i] = clamp01(1.0 - 0.5 * share_tx)
            stab_rx[i] = clamp01(1.0 - 0.5 * share_rx)

    # Bundle-consistency term: compare residuals to bundle median per unordered router pair
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

    def _median(vals: List[float]) -> float:
        s = sorted(vals)
        n = len(s)
        if n == 0:
            return 0.0
        if n % 2 == 1:
            return s[n // 2]
        return 0.5 * (s[n // 2 - 1] + s[n // 2])

    for rp, pairs in bundles_cons.items():
        if not pairs:
            continue
        e_ab_list = []
        e_ba_list = []
        for (a, b) in pairs:
            e_ab_list.append(hard_tx.get(a, 0.0) - hard_rx.get(b, 0.0))
            e_ba_list.append(hard_tx.get(b, 0.0) - hard_rx.get(a, 0.0))
        med_ab = _median(e_ab_list)
        med_ba = _median(e_ba_list)
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

    # Defaults for interfaces not covered above
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
        # Note: rel_diff uses max(1.0, ...) in denominator, stabilizing near-zero cases
        change_tx = rel_diff(orig_tx[if_id], rep_tx)
        change_rx = rel_diff(orig_rx[if_id], rep_rx)

        # Pre-fusion mismatch (redundancy before hardening)
        pre_tx = pre_mismatch_tx.get(if_id, 0.4)
        pre_rx = pre_mismatch_rx.get(if_id, 0.4)

        # Post-fusion symmetry agreement (redundancy after final hardening)
        fin_sym_tx = clamp01(1.0 - post_mismatch_tx_dir.get(if_id, 0.4))
        fin_sym_rx = clamp01(1.0 - post_mismatch_rx_dir.get(if_id, 0.4))

        # Router context penalty AFTER projection
        r = router_of.get(if_id, None)
        router_penalty_after = router_imbalance_after.get(r, 0.0) if r is not None else 0.0
        router_factor_after = clamp01(1.0 - min(0.5, router_penalty_after))

        base_tx_conf = conf_tx_link.get(if_id, 0.6)
        base_rx_conf = conf_rx_link.get(if_id, 0.6)

        red_tx = clamp01(1.0 - pre_tx)
        red_rx = clamp01(1.0 - pre_rx)

        ch_tx = clamp01(1.0 - change_tx)
        ch_rx = clamp01(1.0 - change_rx)

        # Blend confidence components with emphasis on final invariant satisfaction and scale penalty
        scale_tx_term = clamp01(1.0 - min(0.5, abs(scaled_tx_factor.get(if_id, 1.0) - 1.0)))
        scale_rx_term = clamp01(1.0 - min(0.5, abs(scaled_rx_factor.get(if_id, 1.0) - 1.0)))
        # Add improvement credit and topology-aware stability/bundle-consistency terms
        stab_term_tx = clamp01(stab_tx.get(if_id, 0.85))
        stab_term_rx = clamp01(stab_rx.get(if_id, 0.85))
        bcons_term_tx = clamp01(bcons_tx.get(if_id, 0.8))
        bcons_term_rx = clamp01(bcons_rx.get(if_id, 0.8))
        imp_tx = clamp01(pre_tx - post_mismatch_tx_dir.get(if_id, 0.4))
        imp_rx = clamp01(pre_rx - post_mismatch_rx_dir.get(if_id, 0.4))

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

        # Confidence refinements: clip-hit penalty and untouched boost
        if abs(scaled_tx_factor.get(if_id, 1.0) - 1.0) >= 0.10:
            conf_tx_final *= 0.95
        if abs(scaled_rx_factor.get(if_id, 1.0) - 1.0) >= 0.10:
            conf_rx_final *= 0.95
        # Additional penalty when scaling was clipped
        if 'clip_hit_tx' in locals() and clip_hit_tx.get(if_id, False):
            conf_tx_final *= 0.95
        if 'clip_hit_rx' in locals() and clip_hit_rx.get(if_id, False):
            conf_rx_final *= 0.95
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

    # Final confidence touch-up: incorporate final symmetry residuals like the top-calibrated variant.
    # Only adjust when both ends are up to avoid inflating confidence for downed links.
    for i, data in telemetry.items():
        peer = data.get('connected_to')
        if not isinstance(peer, str) or peer not in telemetry or i not in result or peer not in result:
            continue
        if result[i]['interface_status'][1] != 'up' or result[peer]['interface_status'][1] != 'up':
            continue
        mis_tx = rel_diff(safe_rate(result[i]['tx_rate'][1]), safe_rate(result[peer]['rx_rate'][1]))
        mis_rx = rel_diff(safe_rate(result[i]['rx_rate'][1]), safe_rate(result[peer]['tx_rate'][1]))
        old_tx_c = clamp01(result[i]['tx_rate'][2])
        old_rx_c = clamp01(result[i]['rx_rate'][2])
        result[i]['tx_rate'] = (
            result[i]['tx_rate'][0],
            result[i]['tx_rate'][1],
            clamp01(0.7 * old_tx_c + 0.3 * clamp01(1.0 - mis_tx))
        )
        result[i]['rx_rate'] = (
            result[i]['rx_rate'][0],
            result[i]['rx_rate'][1],
            clamp01(0.7 * old_rx_c + 0.3 * clamp01(1.0 - mis_rx))
        )

    # Peer confidence smoothing (10%) when both ends are up
    for i, data in telemetry.items():
        p = data.get('connected_to')
        if not isinstance(p, str) or p not in telemetry:
            continue
        if i not in result or p not in result:
            continue
        if result[i]['interface_status'][1] != 'up' or result[p]['interface_status'][1] != 'up':
            continue
        my_tx_c = clamp01(result[i]['tx_rate'][2])
        my_rx_c = clamp01(result[i]['rx_rate'][2])
        peer_rx_c = clamp01(result[p]['rx_rate'][2])
        peer_tx_c = clamp01(result[p]['tx_rate'][2])
        result[i]['tx_rate'] = (result[i]['tx_rate'][0], result[i]['tx_rate'][1],
                                clamp01(0.90 * my_tx_c + 0.10 * peer_rx_c))
        result[i]['rx_rate'] = (result[i]['rx_rate'][0], result[i]['rx_rate'][1],
                                clamp01(0.90 * my_rx_c + 0.10 * peer_tx_c))

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