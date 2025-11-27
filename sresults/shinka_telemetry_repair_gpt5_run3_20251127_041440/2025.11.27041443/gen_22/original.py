# EVOLVE-BLOCK-START
"""
Targeted Hodor repair: robust link hardening + targeted router projection with adaptive tolerances
and calibrated confidences.

Enhancements:
- Targeted router correction on low-confidence interfaces with damped, clipped per-interface scaling.
- Bundle-aware harmonization for parallel links (same remote_router) when they dominate traffic.
- Adaptive tolerances (TAU_H and router imbalance) to avoid jitter at low rates.
- Confidence-gap-proportional post-projection re-sync with scaling guard.
- Confidence calibration includes scale-penalty and peer smoothing, plus soft-zero stabilization.
"""
from typing import Dict, Any, Tuple, List
import math


def repair_network_telemetry(telemetry: Dict[str, Dict[str, Any]],
                             topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:
    """
    Repair network telemetry using robust per-link fusion and targeted router-level flow projection.

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
    # Baseline tolerances/thresholds inspired by Hodor
    TAU_H_BASE = 0.02      # baseline symmetry tolerance 2%
    EPS = 1e-9
    ZERO_THRESH = 0.1      # Mbps considered near-zero

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

    def adapt_tau_h(v1: float, v2: float, c1: float, c2: float) -> float:
        # Adaptive symmetry tolerance:
        # - tighter (1.5%) when both rates high and confidences >= 0.8
        # - looser (3%) when either rate very low or any confidence < 0.7
        # - baseline otherwise (2%)
        high = (v1 > 100.0 and v2 > 100.0)
        low = (v1 < 1.0 or v2 < 1.0)
        if high and c1 >= 0.8 and c2 >= 0.8:
            return 0.015
        if low or c1 < 0.7 or c2 < 0.7:
            return 0.03
        return TAU_H_BASE

    def clamp(x: float, lo: float, hi: float) -> float:
        return lo if x < lo else hi if x > hi else x

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

    # Stage 1: Link hardening with adaptive fusion and soft-zero stabilization
    hard_tx: Dict[str, float] = {}
    hard_rx: Dict[str, float] = {}
    conf_tx_link: Dict[str, float] = {}
    conf_rx_link: Dict[str, float] = {}
    pre_mismatch_tx: Dict[str, float] = {}
    pre_mismatch_rx: Dict[str, float] = {}

    visited = set()

    def fuse_direction(v_local: float, v_peer: float, s_local: str, s_peer: str) -> Tuple[float, float]:
        mismatch = rel_diff(v_local, v_peer)

        # Directional soft-zero: both near-zero => zero with high confidence
        if max(v_local, v_peer) < ZERO_THRESH:
            return 0.0, 0.95

        # Within tolerance, keep local reading
        if mismatch <= TAU_H_BASE:
            return v_local, 0.95

        # Moderate mismatch: average
        if mismatch <= 0.10:
            fused = 0.5 * v_local + 0.5 * v_peer
            return fused, clamp01(1.0 - mismatch)

        # Large mismatch: prefer the more plausible side (non-zero and "up")
        if v_local < ZERO_THRESH and v_peer >= ZERO_THRESH:
            return v_peer, clamp01(1.0 - mismatch)
        if v_peer < ZERO_THRESH and v_local >= ZERO_THRESH:
            return v_local, clamp01(1.0 - mismatch)

        if s_local == "down" and s_peer == "up":
            return v_peer, clamp01(1.0 - mismatch)
        if s_peer == "down" and s_local == "up":
            return v_local, clamp01(1.0 - mismatch)

        # Default: bias toward peer to resolve asymmetry
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

        # Link-level soft-zero: all four directions tiny -> snap both to zero
        if max(a_tx, b_rx, a_rx, b_tx) < 2.0 * ZERO_THRESH:
            hard_tx[a] = 0.0
            hard_rx[a] = 0.0
            hard_tx[b] = 0.0
            hard_rx[b] = 0.0
            conf_tx_link[a] = 0.96
            conf_rx_link[a] = 0.96
            conf_tx_link[b] = 0.96
            conf_rx_link[b] = 0.96
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

        # Assign hardened values maintaining symmetry mapping
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

    # Stage 2: Targeted router-level flow projection
    # Track scale factors to guard post-sync and calibrate confidence
    scaled_tx_factor: Dict[str, float] = {if_id: 1.0 for if_id in telemetry}
    scaled_rx_factor: Dict[str, float] = {if_id: 1.0 for if_id in telemetry}

    def adaptive_router_tol(n_active: int) -> float:
        # Adaptive router imbalance tolerance
        # 0.05 * sqrt(2 / max(2, n_active)) clamped in [0.03, 0.07]
        base = 0.05 * math.sqrt(2.0 / max(2, n_active))
        return clamp(base, 0.03, 0.07)

    for router, if_list in router_ifaces.items():
        # Ignore empty routers
        if not if_list:
            continue

        # Active interfaces for tolerance calculation
        n_active_tx = sum(1 for i in if_list if hard_tx.get(i, 0.0) >= ZERO_THRESH)
        n_active_rx = sum(1 for i in if_list if hard_rx.get(i, 0.0) >= ZERO_THRESH)
        tau_router_tx = adaptive_router_tol(n_active_tx)
        tau_router_rx = adaptive_router_tol(n_active_rx)
        tau_router = max(tau_router_tx, tau_router_rx)

        sum_tx = sum(hard_tx.get(i, 0.0) for i in if_list)
        sum_rx = sum(hard_rx.get(i, 0.0) for i in if_list)
        if max(sum_tx, sum_rx) < EPS:
            continue  # nothing to project

        mismatch = rel_diff(sum_tx, sum_rx)
        if mismatch <= tau_router:
            continue

        # Choose side with lower aggregate link confidence to adjust
        c_tx_total = sum(conf_tx_link.get(i, 0.6) for i in if_list)
        c_rx_total = sum(conf_rx_link.get(i, 0.6) for i in if_list)
        adjust_side = "tx" if c_tx_total < c_rx_total else "rx"

        # Build weights: w_i = (1 - conf) * rate, zero weight for near-zero rates
        if adjust_side == "tx" and sum_tx > 0:
            vals = [hard_tx.get(i, 0.0) for i in if_list]
            confs = [conf_tx_link.get(i, 0.6) for i in if_list]
            weights = []
            for v, c in zip(vals, confs):
                w = (1.0 - clamp01(c)) * (v if v >= ZERO_THRESH else 0.0)
                weights.append(max(0.0, w))
            denom = sum(weights)
            target = sum_rx
            current = sum_tx
            delta = target - current
            if abs(denom) < EPS:
                # Fallback: uniform damped scaling
                alpha = clamp(target / max(current, EPS), 0.90, 1.10)
                alpha_eff = 1.0 + 0.6 * (alpha - 1.0)
                for i in if_list:
                    old = hard_tx[i]
                    new = old * alpha_eff
                    hard_tx[i] = new
                    scaled_tx_factor[i] *= alpha_eff
                    penalty = abs(alpha_eff - 1.0)
                    conf_tx_link[i] *= clamp01(1.0 - 0.4 * clamp01(penalty))
            else:
                # Proposed additive targeted change with damping
                k = 0.6 * (delta / denom)
                # Compute per-interface new value and scale factor proposal
                proposed_scales: Dict[str, float] = {}
                # Bundle-aware: compute group by remote_router shares
                group_sums: Dict[Any, float] = {}
                for idx, i in enumerate(if_list):
                    rr = telemetry[i].get("remote_router")
                    group_sums.setdefault(rr, 0.0)
                    group_sums[rr] += vals[idx]
                total_side = max(sum(vals), EPS)
                # Identify dominant group if any (>50%)
                dominant_group = None
                for rr, s in group_sums.items():
                    if s / total_side > 0.5:
                        dominant_group = rr
                        break

                # First pass: compute proposed scales
                for idx, i in enumerate(if_list):
                    v = vals[idx]
                    w = weights[idx]
                    if v < EPS or w <= 0.0:
                        proposed_scales[i] = 1.0
                        continue
                    new_v = v + k * w
                    scale_i = clamp(new_v / v, 0.90, 1.10)
                    proposed_scales[i] = scale_i

                # If a dominant bundle exists, harmonize scales within it
                if dominant_group is not None:
                    members = [i for i in if_list if telemetry[i].get("remote_router") == dominant_group]
                    if members:
                        mean_scale = sum(proposed_scales[m] for m in members) / len(members)
                        mean_scale = clamp(mean_scale, 0.90, 1.10)
                        for m in members:
                            proposed_scales[m] = mean_scale

                # Apply scales and update confidence
                for i in if_list:
                    scale_i = proposed_scales.get(i, 1.0)
                    hard_tx[i] = hard_tx[i] * scale_i
                    scaled_tx_factor[i] *= scale_i
                    penalty = abs(scale_i - 1.0)
                    conf_tx_link[i] *= clamp01(1.0 - 0.4 * clamp01(penalty))

        elif adjust_side == "rx" and sum_rx > 0:
            vals = [hard_rx.get(i, 0.0) for i in if_list]
            confs = [conf_rx_link.get(i, 0.6) for i in if_list]
            weights = []
            for v, c in zip(vals, confs):
                w = (1.0 - clamp01(c)) * (v if v >= ZERO_THRESH else 0.0)
                weights.append(max(0.0, w))
            denom = sum(weights)
            target = sum_tx
            current = sum_rx
            delta = target - current
            if abs(denom) < EPS:
                alpha = clamp(target / max(current, EPS), 0.90, 1.10)
                alpha_eff = 1.0 + 0.6 * (alpha - 1.0)
                for i in if_list:
                    old = hard_rx[i]
                    new = old * alpha_eff
                    hard_rx[i] = new
                    scaled_rx_factor[i] *= alpha_eff
                    penalty = abs(alpha_eff - 1.0)
                    conf_rx_link[i] *= clamp01(1.0 - 0.4 * clamp01(penalty))
            else:
                k = 0.6 * (delta / denom)
                proposed_scales: Dict[str, float] = {}
                group_sums: Dict[Any, float] = {}
                for idx, i in enumerate(if_list):
                    rr = telemetry[i].get("remote_router")
                    group_sums.setdefault(rr, 0.0)
                    group_sums[rr] += vals[idx]
                total_side = max(sum(vals), EPS)
                dominant_group = None
                for rr, s in group_sums.items():
                    if s / total_side > 0.5:
                        dominant_group = rr
                        break
                for idx, i in enumerate(if_list):
                    v = vals[idx]
                    w = weights[idx]
                    if v < EPS or w <= 0.0:
                        proposed_scales[i] = 1.0
                        continue
                    new_v = v + k * w
                    scale_i = clamp(new_v / v, 0.90, 1.10)
                    proposed_scales[i] = scale_i
                if dominant_group is not None:
                    members = [i for i in if_list if telemetry[i].get("remote_router") == dominant_group]
                    if members:
                        mean_scale = sum(proposed_scales[m] for m in members) / len(members)
                        mean_scale = clamp(mean_scale, 0.90, 1.10)
                        for m in members:
                            proposed_scales[m] = mean_scale
                for i in if_list:
                    scale_i = proposed_scales.get(i, 1.0)
                    hard_rx[i] = hard_rx[i] * scale_i
                    scaled_rx_factor[i] *= scale_i
                    penalty = abs(scale_i - 1.0)
                    conf_rx_link[i] *= clamp01(1.0 - 0.4 * clamp01(penalty))

    # Stage 2.5: Post-projection gentle link re-sync (confidence-gap-proportional, with scaling guard)
    for a, data in telemetry.items():
        b = peers.get(a)
        if not b or a > b:
            continue  # process each pair once

        # a->b direction
        tx_a = hard_tx.get(a, 0.0)
        rx_b = hard_rx.get(b, 0.0)
        ca = clamp01(conf_tx_link.get(a, 0.6))
        cb = clamp01(conf_rx_link.get(b, 0.6))
        tau_h_ab = adapt_tau_h(tx_a, rx_b, ca, cb)
        diff_ab = rel_diff(tx_a, rx_b)
        strong_scaled_a = abs(scaled_tx_factor.get(a, 1.0) - 1.0) > 0.08
        strong_scaled_b = abs(scaled_rx_factor.get(b, 1.0) - 1.0) > 0.08
        if diff_ab > tau_h_ab and max(tx_a, rx_b) >= ZERO_THRESH:
            mean_ab = 0.5 * (tx_a + rx_b)
            if ca < cb and not strong_scaled_a:
                f = min(0.4, max(0.0, cb - ca))
                hard_tx[a] = (1.0 - f) * tx_a + f * mean_ab
                conf_tx_link[a] *= 0.95
            elif cb < ca and not strong_scaled_b:
                f = min(0.4, max(0.0, ca - cb))
                hard_rx[b] = (1.0 - f) * rx_b + f * mean_ab
                conf_rx_link[b] *= 0.95
            else:
                # similar confidence and low: cautious small nudge both ways
                if min(ca, cb) < 0.7:
                    f = 0.15
                    if not strong_scaled_a:
                        hard_tx[a] = (1.0 - f) * tx_a + f * mean_ab
                        conf_tx_link[a] *= 0.95
                    if not strong_scaled_b:
                        hard_rx[b] = (1.0 - f) * rx_b + f * mean_ab
                        conf_rx_link[b] *= 0.95

        # b->a direction
        tx_b = hard_tx.get(b, 0.0)
        rx_a = hard_rx.get(a, 0.0)
        cb_tx = clamp01(conf_tx_link.get(b, 0.6))
        ca_rx = clamp01(conf_rx_link.get(a, 0.6))
        tau_h_ba = adapt_tau_h(tx_b, rx_a, cb_tx, ca_rx)
        diff_ba = rel_diff(tx_b, rx_a)
        strong_scaled_btx = abs(scaled_tx_factor.get(b, 1.0) - 1.0) > 0.08
        strong_scaled_arx = abs(scaled_rx_factor.get(a, 1.0) - 1.0) > 0.08
        if diff_ba > tau_h_ba and max(tx_b, rx_a) >= ZERO_THRESH:
            mean_ba = 0.5 * (tx_b + rx_a)
            if cb_tx < ca_rx and not strong_scaled_btx:
                f = min(0.4, max(0.0, ca_rx - cb_tx))
                hard_tx[b] = (1.0 - f) * tx_b + f * mean_ba
                conf_tx_link[b] *= 0.95
            elif ca_rx < cb_tx and not strong_scaled_arx:
                f = min(0.4, max(0.0, cb_tx - ca_rx))
                hard_rx[a] = (1.0 - f) * rx_a + f * mean_ba
                conf_rx_link[a] *= 0.95
            else:
                if min(cb_tx, ca_rx) < 0.7:
                    f = 0.15
                    if not strong_scaled_btx:
                        hard_tx[b] = (1.0 - f) * tx_b + f * mean_ba
                        conf_tx_link[b] *= 0.95
                    if not strong_scaled_arx:
                        hard_rx[a] = (1.0 - f) * rx_a + f * mean_ba
                        conf_rx_link[a] *= 0.95

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

        if s_local == "down" and s_peer == "down":
            rep_local = "down"
            rep_peer = "down"
            c_local = 0.98
            c_peer = 0.98
        elif s_local != s_peer:
            # If any traffic on link, set both up
            link_has_traffic = (hard_tx.get(if_id, 0.0) >= ZERO_THRESH or hard_rx.get(if_id, 0.0) >= ZERO_THRESH or
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

    # Router imbalance AFTER projection (for confidence context)
    router_imbalance_after: Dict[str, float] = {}
    for router, if_list in router_ifaces.items():
        if not if_list:
            router_imbalance_after[router] = 0.0
            continue
        sum_tx_after = sum(hard_tx.get(i, 0.0) for i in if_list)
        sum_rx_after = sum(hard_rx.get(i, 0.0) for i in if_list)
        router_imbalance_after[router] = rel_diff(sum_tx_after, sum_rx_after)

    # Post-adjustment symmetry residuals
    post_mismatch_tx_dir: Dict[str, float] = {}
    post_mismatch_rx_dir: Dict[str, float] = {}
    for if_id in telemetry.keys():
        peer = peers.get(if_id)
        if peer:
            post_mismatch_tx_dir[if_id] = rel_diff(hard_tx.get(if_id, 0.0), hard_rx.get(peer, 0.0))
            post_mismatch_rx_dir[if_id] = rel_diff(hard_rx.get(if_id, 0.0), hard_tx.get(peer, 0.0))
        else:
            post_mismatch_tx_dir[if_id] = 0.4
            post_mismatch_rx_dir[if_id] = 0.4

    # Compose final results with calibrated confidences
    result: Dict[str, Dict[str, Tuple]] = {}
    for if_id, data in telemetry.items():
        rep_tx = hard_tx.get(if_id, orig_tx[if_id])
        rep_rx = hard_rx.get(if_id, orig_rx[if_id])

        # Change magnitude from original
        change_tx = rel_diff(orig_tx[if_id], rep_tx)
        change_rx = rel_diff(orig_rx[if_id], rep_rx)

        # Pre-fusion mismatch (redundancy before hardening)
        pre_tx = pre_mismatch_tx.get(if_id, 0.4)
        pre_rx = pre_mismatch_rx.get(if_id, 0.4)

        # Final symmetry agreement
        fin_sym_tx = clamp01(1.0 - post_mismatch_tx_dir.get(if_id, 0.4))
        fin_sym_rx = clamp01(1.0 - post_mismatch_rx_dir.get(if_id, 0.4))

        # Router context AFTER projection
        r = router_of.get(if_id, None)
        router_penalty_after = router_imbalance_after.get(r, 0.0) if r is not None else 0.0
        router_factor_after = clamp01(1.0 - min(0.5, router_penalty_after))

        base_tx_conf = clamp01(conf_tx_link.get(if_id, 0.6))
        base_rx_conf = clamp01(conf_rx_link.get(if_id, 0.6))

        red_tx = clamp01(1.0 - pre_tx)
        red_rx = clamp01(1.0 - pre_rx)

        ch_tx = clamp01(1.0 - change_tx)
        ch_rx = clamp01(1.0 - change_rx)

        # Scale-factor penalty term (higher when little scaling)
        scale_tx_term = clamp01(1.0 - min(0.5, abs(scaled_tx_factor.get(if_id, 1.0) - 1.0)))
        scale_rx_term = clamp01(1.0 - min(0.5, abs(scaled_rx_factor.get(if_id, 1.0) - 1.0)))

        # Blend confidence components with emphasis on final invariant satisfaction and scale penalty
        conf_tx_final = clamp01(
            0.23 * base_tx_conf +
            0.17 * red_tx +
            0.30 * fin_sym_tx +
            0.10 * ch_tx +
            0.10 * router_factor_after +
            0.10 * scale_tx_term
        )
        conf_rx_final = clamp01(
            0.23 * base_rx_conf +
            0.17 * red_rx +
            0.30 * fin_sym_rx +
            0.10 * ch_rx +
            0.10 * router_factor_after +
            0.10 * scale_rx_term
        )

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

    # Final confidence touch-up: peer-aware smoothing when both ends are up
    for i, data in telemetry.items():
        peer = data.get('connected_to')
        if not isinstance(peer, str) or peer not in telemetry or i not in result or peer not in result:
            continue
        if result[i]['interface_status'][1] != 'up' or result[peer]['interface_status'][1] != 'up':
            continue
        # Symmetry residuals
        mis_tx = rel_diff(safe_rate(result[i]['tx_rate'][1]), safe_rate(result[peer]['rx_rate'][1]))
        mis_rx = rel_diff(safe_rate(result[i]['rx_rate'][1]), safe_rate(result[peer]['tx_rate'][1]))
        # 10% blend from opposite peer direction + residual-informed touch
        old_tx_c = clamp01(result[i]['tx_rate'][2])
        old_rx_c = clamp01(result[i]['rx_rate'][2])
        peer_rx_c = clamp01(result[peer]['rx_rate'][2])
        peer_tx_c = clamp01(result[peer]['tx_rate'][2])
        result[i]['tx_rate'] = (
            result[i]['tx_rate'][0],
            result[i]['tx_rate'][1],
            clamp01(0.80 * old_tx_c + 0.10 * peer_rx_c + 0.10 * clamp01(1.0 - mis_tx))
        )
        result[i]['rx_rate'] = (
            result[i]['rx_rate'][0],
            result[i]['rx_rate'][1],
            clamp01(0.80 * old_rx_c + 0.10 * peer_tx_c + 0.10 * clamp01(1.0 - mis_rx))
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