# EVOLVE-BLOCK-START
"""
Targeted-bundle router sync: robust link fusion, targeted router projection with bundle awareness,
confidence-gap re-sync, adaptive tolerances, and calibrated confidence.

This algorithm:
1) Harden links using redundant peer signals with adaptive tolerance.
2) Enforce router flow conservation by targeted scaling of low-confidence, active interfaces
   (per-interface damped factors, clipped), with bundle-aware scaling for parallel links.
3) Perform confidence-gap proportional re-sync on links (one-sided nudge), skipping directions
   that already received strong router scaling.
4) Apply adaptive tolerances and a soft-zero rule to stabilize near-zero noise.
5) Calibrate confidence with scale penalties, final invariant satisfaction, and peer smoothing.
"""
from typing import Dict, Any, Tuple, List
import math


def repair_network_telemetry(telemetry: Dict[str, Dict[str, Any]],
                             topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:
    # Base thresholds
    ZERO_THRESH = 0.1  # Mbps considered near-zero
    EPS = 1e-9

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
        # Adaptive hardening tolerance:
        # - Stricter (1.5%) when both > 100 Mbps and confidences (if provided) are high
        # - Looser (3%) when either < 1 Mbps or any low confidence
        # - Baseline 2% otherwise
        high = (v1 > 100.0 and v2 > 100.0)
        low = (v1 < 1.0 or v2 < 1.0)
        if c1 is not None and c2 is not None:
            high_conf = (c1 >= 0.8 and c2 >= 0.8)
            low_conf = (c1 < 0.7 or c2 < 0.7)
        else:
            high_conf = False
            low_conf = False
        if high and high_conf:
            return 0.015
        if low or low_conf:
            return 0.03
        return 0.02

    def tau_router(n_active: int) -> float:
        # Adaptive router tolerance based on active interfaces
        # Clamp in [0.03, 0.07]
        base = 0.05 * math.sqrt(2.0 / max(2, n_active))
        return max(0.03, min(0.07, base))

    # Build peers mapping
    peers: Dict[str, str] = {}
    for if_id, data in telemetry.items():
        peer = data.get("connected_to")
        if isinstance(peer, str) and peer in telemetry:
            peers[if_id] = peer

    # Build router->interfaces from topology; also record router_of for each interface
    router_ifaces: Dict[str, List[str]] = {}
    for r, if_list in topology.items():
        router_ifaces.setdefault(r, [])
        for i in if_list:
            if i in telemetry:
                router_ifaces[r].append(i)
    router_of: Dict[str, str] = {}
    for if_id, data in telemetry.items():
        r = data.get("local_router")
        if r is None:
            r = f"unknown_router::{if_id}"
        router_ifaces.setdefault(r, [])
        if if_id not in router_ifaces[r]:
            router_ifaces[r].append(if_id)
        router_of[if_id] = r

    # Prepare originals and statuses
    orig_tx: Dict[str, float] = {}
    orig_rx: Dict[str, float] = {}
    status_raw: Dict[str, str] = {}
    for if_id, data in telemetry.items():
        orig_tx[if_id] = safe_rate(data.get("tx_rate", 0.0))
        orig_rx[if_id] = safe_rate(data.get("rx_rate", 0.0))
        s = data.get("interface_status", "unknown")
        status_raw[if_id] = s if s in ("up", "down") else "unknown"

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

        # Both near-zero => stabilize at 0 with high confidence
        if max(v_local, v_peer) < ZERO_THRESH:
            return 0.0, 0.95

        # Within tolerance: trust local
        if mismatch <= th:
            return v_local, 0.95

        # Moderate mismatch: average
        if mismatch <= 0.10:
            fused = 0.5 * v_local + 0.5 * v_peer
            return fused, clamp01(1.0 - mismatch)

        # Large mismatch: decide by plausibility
        if v_local < ZERO_THRESH and v_peer >= ZERO_THRESH:
            return v_peer, clamp01(1.0 - mismatch)
        if v_peer < ZERO_THRESH and v_local >= ZERO_THRESH:
            return v_local, clamp01(1.0 - mismatch)

        # Status-aware: prefer side that is up
        if s_local == "down" and s_peer == "up":
            return v_peer, clamp01(1.0 - mismatch)
        if s_peer == "down" and s_local == "up":
            return v_local, clamp01(1.0 - mismatch)

        # Default: lean toward peer to reconcile asymmetry
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
        sa, sb = status_raw[a], status_raw[b]

        # Both ends down: force zeros strongly
        if sa == "down" and sb == "down":
            for i in (a, b):
                hard_tx[i] = 0.0
                hard_rx[i] = 0.0
                conf_tx_link[i] = 0.98
                conf_rx_link[i] = 0.98
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

        # Map fused directions to both ends
        hard_tx[a] = fused_ab
        hard_rx[b] = fused_ab
        hard_tx[b] = fused_ba
        hard_rx[a] = fused_ba

        conf_tx_link[a] = c_ab
        conf_rx_link[b] = c_ab
        conf_tx_link[b] = c_ba
        conf_rx_link[a] = c_ba

    # Ensure all interfaces have values
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

    # Stage 2: Targeted router-level flow projection with bundle awareness
    scaled_tx_factor: Dict[str, float] = {if_id: 1.0 for if_id in telemetry}
    scaled_rx_factor: Dict[str, float] = {if_id: 1.0 for if_id in telemetry}
    strong_scaled_tx: Dict[str, bool] = {if_id: False for if_id in telemetry}
    strong_scaled_rx: Dict[str, bool] = {if_id: False for if_id in telemetry}

    for router, if_list in router_ifaces.items():
        if len(if_list) <= 1:
            continue

        sum_tx = sum(hard_tx.get(i, 0.0) for i in if_list)
        sum_rx = sum(hard_rx.get(i, 0.0) for i in if_list)
        n_active_tx = sum(1 for i in if_list if hard_tx.get(i, 0.0) >= ZERO_THRESH)
        n_active_rx = sum(1 for i in if_list if hard_rx.get(i, 0.0) >= ZERO_THRESH)
        n_active = max(n_active_tx, n_active_rx)
        if max(sum_tx, sum_rx) < EPS:
            continue
        mismatch = rel_diff(sum_tx, sum_rx)
        TAU_ROUTER = tau_router(n_active)
        if mismatch <= TAU_ROUTER:
            continue

        # Decide side by aggregate confidence
        c_tx_total = sum(conf_tx_link.get(i, 0.6) for i in if_list)
        c_rx_total = sum(conf_rx_link.get(i, 0.6) for i in if_list)
        adjust_side = "tx" if c_tx_total < c_rx_total else "rx"

        # Build per-interface values, confidences, and bundle groups by remote router
        def group_key(iid: str) -> str:
            rr = telemetry.get(iid, {}).get("remote_router")
            return f"{rr}" if rr is not None else f"unknown_remote::{iid}"

        if adjust_side == "tx" and sum_tx > 0:
            vals = {i: hard_tx.get(i, 0.0) for i in if_list}
            confs = {i: clamp01(conf_tx_link.get(i, 0.6)) for i in if_list}
            weights = {}
            for i, v in vals.items():
                w = (1.0 - confs[i]) * (v if v >= ZERO_THRESH else 0.0)
                weights[i] = max(0.0, w)
            denom = sum(weights.values())
            target, current = sum_rx, sum_tx

            if denom < EPS:
                # Fallback: uniform damped scaling
                alpha = target / max(current, EPS)
                alpha = max(0.90, min(1.10, alpha))
                alpha_eff = 1.0 + 0.6 * (alpha - 1.0)
                for i in if_list:
                    hard_tx[i] = vals[i] * alpha_eff
                    scaled_tx_factor[i] *= alpha_eff
                    penalty = abs(alpha_eff - 1.0)
                    if penalty > 0.08:
                        strong_scaled_tx[i] = True
                    conf_tx_link[i] *= clamp01(1.0 - 0.4 * clamp01(penalty))
            else:
                k = (target - current) / denom

                # Compute group-major bundles
                total_side = sum(vals.values())
                bundles: Dict[str, List[str]] = {}
                for i in if_list:
                    bundles.setdefault(group_key(i), []).append(i)
                bundle_share: Dict[str, float] = {}
                for g, members in bundles.items():
                    share = sum(vals[m] for m in members) / max(total_side, EPS)
                    bundle_share[g] = share

                # Precompute group-level ratio for dominant bundles
                group_alpha_raw: Dict[str, float] = {}
                for g, members in bundles.items():
                    if bundle_share[g] >= 0.55:
                        sum_w = sum(weights[m] for m in members)
                        sum_v = sum(vals[m] for m in members)
                        if sum_v > EPS:
                            alpha_raw_g = 1.0 + k * (sum_w / sum_v)
                        else:
                            alpha_raw_g = 1.0
                        group_alpha_raw[g] = alpha_raw_g

                # Apply per-interface or bundle-shared scaling
                for i, v in vals.items():
                    if v < EPS:
                        continue
                    g = group_key(i)
                    if g in group_alpha_raw:
                        alpha_raw = group_alpha_raw[g]
                    else:
                        alpha_raw = 1.0 + k * (weights[i] / max(v, EPS))
                    # Damped and clipped
                    alpha_eff = 1.0 + 0.6 * (alpha_raw - 1.0)
                    alpha_eff = max(0.90, min(1.10, alpha_eff))
                    hard_tx[i] = v * alpha_eff
                    scaled_tx_factor[i] *= alpha_eff
                    penalty = abs(alpha_eff - 1.0)
                    if penalty > 0.08:
                        strong_scaled_tx[i] = True
                    conf_tx_link[i] *= clamp01(1.0 - 0.4 * clamp01(penalty))

        elif adjust_side == "rx" and sum_rx > 0:
            vals = {i: hard_rx.get(i, 0.0) for i in if_list}
            confs = {i: clamp01(conf_rx_link.get(i, 0.6)) for i in if_list}
            weights = {}
            for i, v in vals.items():
                w = (1.0 - confs[i]) * (v if v >= ZERO_THRESH else 0.0)
                weights[i] = max(0.0, w)
            denom = sum(weights.values())
            target, current = sum_tx, sum_rx

            if denom < EPS:
                alpha = target / max(current, EPS)
                alpha = max(0.90, min(1.10, alpha))
                alpha_eff = 1.0 + 0.6 * (alpha - 1.0)
                for i in if_list:
                    hard_rx[i] = vals[i] * alpha_eff
                    scaled_rx_factor[i] *= alpha_eff
                    penalty = abs(alpha_eff - 1.0)
                    if penalty > 0.08:
                        strong_scaled_rx[i] = True
                    conf_rx_link[i] *= clamp01(1.0 - 0.4 * clamp01(penalty))
            else:
                k = (target - current) / denom

                total_side = sum(vals.values())
                bundles: Dict[str, List[str]] = {}
                for i in if_list:
                    bundles.setdefault(group_key(i), []).append(i)
                bundle_share: Dict[str, float] = {}
                for g, members in bundles.items():
                    share = sum(vals[m] for m in members) / max(total_side, EPS)
                    bundle_share[g] = share

                group_alpha_raw: Dict[str, float] = {}
                for g, members in bundles.items():
                    if bundle_share[g] >= 0.55:
                        sum_w = sum(weights[m] for m in members)
                        sum_v = sum(vals[m] for m in members)
                        if sum_v > EPS:
                            alpha_raw_g = 1.0 + k * (sum_w / sum_v)
                        else:
                            alpha_raw_g = 1.0
                        group_alpha_raw[g] = alpha_raw_g

                for i, v in vals.items():
                    if v < EPS:
                        continue
                    g = group_key(i)
                    if g in group_alpha_raw:
                        alpha_raw = group_alpha_raw[g]
                    else:
                        alpha_raw = 1.0 + k * (weights[i] / max(v, EPS))
                    alpha_eff = 1.0 + 0.6 * (alpha_raw - 1.0)
                    alpha_eff = max(0.90, min(1.10, alpha_eff))
                    hard_rx[i] = v * alpha_eff
                    scaled_rx_factor[i] *= alpha_eff
                    penalty = abs(alpha_eff - 1.0)
                    if penalty > 0.08:
                        strong_scaled_rx[i] = True
                    conf_rx_link[i] *= clamp01(1.0 - 0.4 * clamp01(penalty))

    # Stage 2.5: Confidence-gap proportional link re-sync (skip directions with strong scaling)
    processed_pairs = set()
    for a, data_a in telemetry.items():
        b = data_a.get('connected_to')
        if not isinstance(b, str) or b not in telemetry:
            continue
        key = tuple(sorted([a, b]))
        if key in processed_pairs:
            continue
        processed_pairs.add(key)

        # a->b direction
        tx_a = hard_tx.get(a, 0.0)
        rx_b = hard_rx.get(b, 0.0)
        ca = clamp01(conf_tx_link.get(a, 0.6))
        cb = clamp01(conf_rx_link.get(b, 0.6))
        th_ab = tau_h_dir(tx_a, rx_b, ca, cb)
        if rel_diff(tx_a, rx_b) > th_ab and max(tx_a, rx_b) >= ZERO_THRESH:
            if not (strong_scaled_tx.get(a, False) or strong_scaled_rx.get(b, False)):
                mean_ab = 0.5 * (tx_a + rx_b)
                high, low = (ca, 'a'), (cb, 'b')
                if cb > ca:
                    high, low = (cb, 'b'), (ca, 'a')
                # Weight nudge by both confidence gap and mismatch severity beyond tolerance
                mis_ab = rel_diff(tx_a, rx_b)
                sev_ab = clamp01((mis_ab - th_ab) / max(th_ab, 1e-6))
                f = min(0.4, max(0.0, high[0] - low[0])) * sev_ab
                if f > 0.0:
                    if low[1] == 'a':
                        hard_tx[a] = (1.0 - f) * tx_a + f * mean_ab
                        conf_tx_link[a] *= 0.97
                    else:
                        hard_rx[b] = (1.0 - f) * rx_b + f * mean_ab
                        conf_rx_link[b] *= 0.97

        # b->a direction
        tx_b = hard_tx.get(b, 0.0)
        rx_a = hard_rx.get(a, 0.0)
        cb_tx = clamp01(conf_tx_link.get(b, 0.6))
        ca_rx = clamp01(conf_rx_link.get(a, 0.6))
        th_ba = tau_h_dir(tx_b, rx_a, cb_tx, ca_rx)
        if rel_diff(tx_b, rx_a) > th_ba and max(tx_b, rx_a) >= ZERO_THRESH:
            if not (strong_scaled_tx.get(b, False) or strong_scaled_rx.get(a, False)):
                mean_ba = 0.5 * (tx_b + rx_a)
                high, low = (cb_tx, 'b'), (ca_rx, 'a')
                if ca_rx > cb_tx:
                    high, low = (ca_rx, 'a'), (cb_tx, 'b')
                # Weight nudge by both confidence gap and mismatch severity beyond tolerance
                mis_ba = rel_diff(tx_b, rx_a)
                sev_ba = clamp01((mis_ba - th_ba) / max(th_ba, 1e-6))
                f = min(0.4, max(0.0, high[0] - low[0])) * sev_ba
                if f > 0.0:
                    if low[1] == 'b':
                        hard_tx[b] = (1.0 - f) * tx_b + f * mean_ba
                        conf_tx_link[b] *= 0.97
                    else:
                        hard_rx[a] = (1.0 - f) * rx_a + f * mean_ba
                        conf_rx_link[a] *= 0.97

    # Soft-zero rule: stabilize tiny bidirectional links
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
            hard_tx[a] = 0.0
            hard_rx[b] = 0.0
            hard_tx[b] = 0.0
            hard_rx[a] = 0.0
            conf_tx_link[a] = max(conf_tx_link.get(a, 0.6), 0.95)
            conf_rx_link[b] = max(conf_rx_link.get(b, 0.6), 0.95)
            conf_tx_link[b] = max(conf_tx_link.get(b, 0.6), 0.95)
            conf_rx_link[a] = max(conf_rx_link.get(a, 0.6), 0.95)

    # Status repair (symmetry-aware)
    repaired_status: Dict[str, str] = {}
    status_conf: Dict[str, float] = {}
    handled = set()
    for a, data_a in telemetry.items():
        if a in handled:
            continue
        b = data_a.get('connected_to')
        sa = status_raw.get(a, "unknown")
        if not isinstance(b, str) or b not in telemetry:
            repaired_status[a] = sa
            status_conf[a] = 0.95
            handled.add(a)
            continue
        sb = status_raw.get(b, "unknown")
        # Traffic-based consistency
        any_traffic = (hard_tx.get(a, 0.0) >= ZERO_THRESH or hard_rx.get(a, 0.0) >= ZERO_THRESH or
                       hard_tx.get(b, 0.0) >= ZERO_THRESH or hard_rx.get(b, 0.0) >= ZERO_THRESH)
        if sa == "down" and sb == "down":
            repaired_status[a] = "down"
            repaired_status[b] = "down"
            status_conf[a] = 0.98
            status_conf[b] = 0.98
        elif sa != sb:
            if any_traffic:
                repaired_status[a] = "up"
                repaired_status[b] = "up"
                status_conf[a] = 0.70
                status_conf[b] = 0.70
            else:
                # ambiguous; keep as-is with lower confidence
                repaired_status[a] = sa
                repaired_status[b] = sb
                status_conf[a] = 0.6
                status_conf[b] = 0.6
        else:
            repaired_status[a] = sa
            repaired_status[b] = sb
            status_conf[a] = 0.95
            status_conf[b] = 0.95
        handled.add(a)
        handled.add(b)

    # Prepare final invariant metrics for confidence calibration
    # Router imbalance AFTER projection
    router_imbalance_after: Dict[str, float] = {}
    for r, if_list in router_ifaces.items():
        sum_tx_a = sum(hard_tx.get(i, 0.0) for i in if_list)
        sum_rx_a = sum(hard_rx.get(i, 0.0) for i in if_list)
        router_imbalance_after[r] = rel_diff(sum_tx_a, sum_rx_a)

    # Final per-direction symmetry residuals
    post_mismatch_tx_dir: Dict[str, float] = {}
    post_mismatch_rx_dir: Dict[str, float] = {}
    for i, data in telemetry.items():
        p = data.get("connected_to")
        if isinstance(p, str) and p in telemetry:
            post_mismatch_tx_dir[i] = rel_diff(hard_tx.get(i, 0.0), hard_rx.get(p, 0.0))
            post_mismatch_rx_dir[i] = rel_diff(hard_rx.get(i, 0.0), hard_tx.get(p, 0.0))
        else:
            post_mismatch_tx_dir[i] = 0.4
            post_mismatch_rx_dir[i] = 0.4

    # Compose results with confidence calibration
    result: Dict[str, Dict[str, Tuple]] = {}
    for if_id, data in telemetry.items():
        rep_tx = hard_tx.get(if_id, orig_tx[if_id])
        rep_rx = hard_rx.get(if_id, orig_rx[if_id])

        # Change magnitude relative to original
        change_tx = rel_diff(orig_tx[if_id], rep_tx)
        change_rx = rel_diff(orig_rx[if_id], rep_rx)

        # Redundancy (pre-fusion mismatch)
        pre_tx = pre_mismatch_tx.get(if_id, 0.4)
        pre_rx = pre_mismatch_rx.get(if_id, 0.4)

        # Final symmetry agreement
        fin_sym_tx = clamp01(1.0 - post_mismatch_tx_dir.get(if_id, 0.4))
        fin_sym_rx = clamp01(1.0 - post_mismatch_rx_dir.get(if_id, 0.4))

        # Router factor AFTER projection
        r = router_of.get(if_id, None)
        router_penalty_after = router_imbalance_after.get(r, 0.0) if r is not None else 0.0
        router_factor_after = clamp01(1.0 - min(0.5, router_penalty_after))

        base_tx_conf = clamp01(conf_tx_link.get(if_id, 0.6))
        base_rx_conf = clamp01(conf_rx_link.get(if_id, 0.6))

        red_tx = clamp01(1.0 - pre_tx)
        red_rx = clamp01(1.0 - pre_rx)

        ch_tx = clamp01(1.0 - change_tx)
        ch_rx = clamp01(1.0 - change_rx)

        # Scale penalty term (larger scaling => lower confidence)
        scale_term_tx = clamp01(1.0 - min(0.5, abs(scaled_tx_factor.get(if_id, 1.0) - 1.0)))
        scale_term_rx = clamp01(1.0 - min(0.5, abs(scaled_rx_factor.get(if_id, 1.0) - 1.0)))

        conf_tx_final = clamp01(
            0.22 * base_tx_conf +
            0.18 * red_tx +
            0.28 * fin_sym_tx +
            0.12 * ch_tx +
            0.10 * router_factor_after +
            0.10 * scale_term_tx
        )
        conf_rx_final = clamp01(
            0.22 * base_rx_conf +
            0.18 * red_rx +
            0.28 * fin_sym_rx +
            0.12 * ch_rx +
            0.10 * router_factor_after +
            0.10 * scale_term_rx
        )

        # Status enforcement: down implies zero counters
        rep_status = repaired_status.get(if_id, status_raw.get(if_id, "unknown"))
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

        # Assemble output
        out = {}
        out["rx_rate"] = (orig_rx[if_id], rep_rx, conf_rx_final)
        out["tx_rate"] = (orig_tx[if_id], rep_tx, conf_tx_final)
        out["interface_status"] = (status_raw[if_id], rep_status, conf_status)

        # Copy metadata unchanged
        out["connected_to"] = data.get("connected_to")
        out["local_router"] = data.get("local_router")
        out["remote_router"] = data.get("remote_router")

        result[if_id] = out

    # Peer confidence smoothing with residual-informed touch (when both ends are up)
    for i, data in telemetry.items():
        p = data.get("connected_to")
        if not isinstance(p, str) or p not in telemetry:
            continue
        if i not in result or p not in result:
            continue
        if result[i]["interface_status"][1] != "up" or result[p]["interface_status"][1] != "up":
            continue
        # Compute final symmetry residuals
        tx_i = safe_rate(result[i]["tx_rate"][1])
        rx_p = safe_rate(result[p]["rx_rate"][1])
        rx_i = safe_rate(result[i]["rx_rate"][1])
        tx_p = safe_rate(result[p]["tx_rate"][1])
        mis_tx = rel_diff(tx_i, rx_p)
        mis_rx = rel_diff(rx_i, tx_p)
        # Blend my confidence with 10% peer opposite and 10% residual agreement
        my_tx_c = clamp01(result[i]["tx_rate"][2])
        my_rx_c = clamp01(result[i]["rx_rate"][2])
        peer_rx_c = clamp01(result[p]["rx_rate"][2])
        peer_tx_c = clamp01(result[p]["tx_rate"][2])
        result[i]["tx_rate"] = (
            result[i]["tx_rate"][0],
            result[i]["tx_rate"][1],
            clamp01(0.80 * my_tx_c + 0.10 * peer_rx_c + 0.10 * clamp01(1.0 - mis_tx))
        )
        result[i]["rx_rate"] = (
            result[i]["rx_rate"][0],
            result[i]["rx_rate"][1],
            clamp01(0.80 * my_rx_c + 0.10 * peer_tx_c + 0.10 * clamp01(1.0 - mis_rx))
        )

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