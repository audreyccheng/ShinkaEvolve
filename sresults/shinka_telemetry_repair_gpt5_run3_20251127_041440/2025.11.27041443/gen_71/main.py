# EVOLVE-BLOCK-START
"""
Targeted Hodor repair: robust link hardening + targeted router projection with adaptive tolerances
and calibrated confidences.

Enhancements:
- Two-tier targeted router correction on low-confidence interfaces (dominance cap, damped/clipped scaling).
- Bundle-aware harmonization for dominant parallel links with intra-bundle smoothing.
- Adaptive tolerances (TAU_H and router imbalance) to avoid jitter at low rates.
- Confidence-gap-proportional post-projection re-sync with scaling guard and router-imbalance attenuation.
- Confidence calibration includes scale-penalty, clip-hit penalties, untouched boosts, peer smoothing, and soft-zero stabilization.
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
    # Baseline thresholds
    EPS = 1e-9
    ZERO_THRESH = 0.1  # Mbps considered near-zero

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
        # Numerically stable sigmoid for adaptive gains
        if x >= 0:
            z = math.exp(-x)
            return 1.0 / (1.0 + z)
        else:
            z = math.exp(x)
            return z / (1.0 + z)

    def tau_h_dir(v1: float, v2: float, c1: float = None, c2: float = None) -> float:
        # Adaptive symmetry tolerance:
        # - stricter (1.5%) when both rates are high (>100 Mbps) and confidences high (>=0.8)
        # - looser (3%) when any low rate (<1 Mbps) or any low confidence (<0.7)
        # - baseline 2% otherwise
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
        # Adaptive router imbalance tolerance
        # 0.05 * sqrt(2 / max(2, n_active)) clamped in [0.03, 0.07]
        base = 0.05 * math.sqrt(2.0 / max(2, n_active))
        return clamp(base, 0.03, 0.07)

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
        th = tau_h_dir(v_local, v_peer)

        # Both near-zero => zero with high confidence
        if max(v_local, v_peer) < ZERO_THRESH:
            return 0.0, 0.95

        # Within adaptive tolerance: keep local reading
        if mismatch <= th:
            return v_local, 0.95

        # Moderate mismatch: average
        if mismatch <= 0.10:
            fused = 0.5 * v_local + 0.5 * v_peer
            return fused, clamp01(1.0 - mismatch)

        # Large mismatch: prefer plausible side (non-zero and up)
        if v_local < ZERO_THRESH and v_peer >= ZERO_THRESH:
            return v_peer, clamp01(1.0 - mismatch)
        if v_peer < ZERO_THRESH and v_local >= ZERO_THRESH:
            return v_local, clamp01(1.0 - mismatch)

        if s_local == "down" and s_peer == "up":
            return v_peer, clamp01(1.0 - mismatch)
        if s_peer == "down" and s_local == "up":
            return v_local, clamp01(1.0 - mismatch)

        # Default: bias to peer to reconcile asymmetry
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

        # Link-level soft-zero: all four tiny -> snap both ends to zero
        if max(a_tx, b_rx, a_rx, b_tx) < 2.0 * ZERO_THRESH:
            for i in (a, b):
                hard_tx[i] = 0.0
                hard_rx[i] = 0.0
                conf_tx_link[i] = 0.96
                conf_rx_link[i] = 0.96
            pre_mismatch_tx[a] = rel_diff(a_tx, b_rx)
            pre_mismatch_rx[a] = rel_diff(a_rx, b_tx)
            pre_mismatch_tx[b] = rel_diff(b_tx, a_rx)
            pre_mismatch_rx[b] = rel_diff(b_rx, a_tx)
            continue

        # Directional mismatches (pre-fusion)
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

    # Early dynamic soft-zero pre-pass using router-aware thresholds
    def compute_router_thresholds() -> Dict[str, float]:
        thr: Dict[str, float] = {}
        for r, ifs in router_ifaces.items():
            stx = sum(hard_tx.get(i, 0.0) for i in ifs)
            srx = sum(hard_rx.get(i, 0.0) for i in ifs)
            thr[r] = max(ZERO_THRESH, 0.002 * (stx + srx))
        return thr

    def _router_imbalance_now(rid: str) -> float:
        if rid not in router_ifaces:
            return 0.0
        ifs = router_ifaces[rid]
        stx = sum(hard_tx.get(i, 0.0) for i in ifs)
        srx = sum(hard_rx.get(i, 0.0) for i in ifs)
        return rel_diff(stx, srx)

    router_thr_early = compute_router_thresholds()
    processed_pairs_pre = set()
    for a, data_a in telemetry.items():
        b = data_a.get("connected_to")
        if not isinstance(b, str) or b not in telemetry:
            continue
        key = tuple(sorted([a, b]))
        if key in processed_pairs_pre:
            continue
        processed_pairs_pre.add(key)
        ra = router_of.get(a)
        rb = router_of.get(b)
        thr_pair = 1.5 * max(router_thr_early.get(ra, ZERO_THRESH), router_thr_early.get(rb, ZERO_THRESH))
        tx_a = hard_tx.get(a, 0.0)
        rx_b = hard_rx.get(b, 0.0)
        tx_b = hard_tx.get(b, 0.0)
        rx_a = hard_rx.get(a, 0.0)
        if max(tx_a, rx_b, tx_b, rx_a) < thr_pair:
            def tau_for_router(rid: str) -> float:
                if rid not in router_ifaces:
                    return 0.05
                ifs = router_ifaces[rid]
                n_act = max(1, sum(1 for i in ifs if max(hard_tx.get(i, 0.0), hard_rx.get(i, 0.0)) >= ZERO_THRESH))
                return tau_router(n_act)
            if _router_imbalance_now(ra) <= tau_for_router(ra) and _router_imbalance_now(rb) <= tau_for_router(rb):
                hard_tx[a] = 0.0
                hard_rx[b] = 0.0
                hard_tx[b] = 0.0
                hard_rx[a] = 0.0
                conf_tx_link[a] = max(conf_tx_link.get(a, 0.6), 0.95)
                conf_rx_link[b] = max(conf_rx_link.get(b, 0.6), 0.95)
                conf_tx_link[b] = max(conf_tx_link.get(b, 0.6), 0.95)
                conf_rx_link[a] = max(conf_rx_link.get(a, 0.6), 0.95)

    # Stage 2: Two-tier targeted router-level flow projection with bundle awareness
    # Track scale factors, clip hits and strong scaling for guards and calibration
    scaled_tx_factor: Dict[str, float] = {if_id: 1.0 for if_id in telemetry}
    scaled_rx_factor: Dict[str, float] = {if_id: 1.0 for if_id in telemetry}
    strong_scaled_tx: Dict[str, bool] = {if_id: False for if_id in telemetry}
    strong_scaled_rx: Dict[str, bool] = {if_id: False for if_id in telemetry}
    clip_hit_tx: Dict[str, bool] = {if_id: False for if_id in telemetry}
    clip_hit_rx: Dict[str, bool] = {if_id: False for if_id in telemetry}

    def router_imbalance(router_id: str) -> float:
        if router_id not in router_ifaces:
            return 0.0
        ifs = router_ifaces[router_id]
        stx = sum(hard_tx.get(i, 0.0) for i in ifs)
        srx = sum(hard_rx.get(i, 0.0) for i in ifs)
        return rel_diff(stx, srx)

    # Helper: apply bundle shared scaling for dominant bundle (≥60% share)
    def apply_bundle_shared(router: str, side: str, target_sum: float, current_sum: float, if_list: List[str]) -> float:
        # Returns residual delta after applying the group scaling (delta = target - current)
        delta = target_sum - current_sum
        if abs(current_sum) < EPS:
            return delta
        # Group by remote_router
        group_vals: Dict[Any, float] = {}
        for i in if_list:
            rr = telemetry.get(i, {}).get("remote_router")
            v = hard_tx[i] if side == "tx" else hard_rx[i]
            group_vals[rr] = group_vals.get(rr, 0.0) + v
        total_side = sum(group_vals.values())
        if total_side < EPS:
            return delta
        # Find dominant group
        dom_rr = None
        share = 0.0
        for rr, s in group_vals.items():
            sh = s / total_side
            if sh >= 0.60 and sh > share:
                share = sh
                dom_rr = rr
        if dom_rr is None:
            return delta
        # Compute shared alpha for dominant group
        alpha = clamp(target_sum / max(current_sum, EPS), 0.85, 1.15)
        alpha_eff = 1.0 + 0.6 * (alpha - 1.0)  # damping
        # Apply to members
        applied = 0.0
        for i in if_list:
            rr = telemetry.get(i, {}).get("remote_router")
            if rr != dom_rr:
                continue
            old = hard_tx[i] if side == "tx" else hard_rx[i]
            new = old * alpha_eff
            change = new - old
            if side == "tx":
                hard_tx[i] = new
                scaled_tx_factor[i] *= alpha_eff
                pen = abs(alpha_eff - 1.0)
                conf_tx_link[i] *= clamp01(1.0 - 0.4 * clamp01(pen))
                if pen >= 0.10 or alpha_eff <= 0.90 or alpha_eff >= 1.10:
                    clip_hit_tx[i] = True
                if pen > 0.08:
                    strong_scaled_tx[i] = True
            else:
                hard_rx[i] = new
                scaled_rx_factor[i] *= alpha_eff
                pen = abs(alpha_eff - 1.0)
                conf_rx_link[i] *= clamp01(1.0 - 0.4 * clamp01(pen))
                if pen >= 0.10 or alpha_eff <= 0.90 or alpha_eff >= 1.10:
                    clip_hit_rx[i] = True
                if pen > 0.08:
                    strong_scaled_rx[i] = True
            applied += change
        return delta - applied

    # Helper: intra-bundle smoothing within ±5% while preserving bundle sum
    def bundle_smoothing(router: str, side: str, if_list: List[str]):
        # Build bundles by remote_router
        bundles: Dict[Any, List[str]] = {}
        for i in if_list:
            rr = telemetry.get(i, {}).get("remote_router")
            bundles.setdefault(rr, []).append(i)
        for rr, members in bundles.items():
            if len(members) < 2:
                continue
            vals = [hard_tx[m] if side == "tx" else hard_rx[m] for m in members]
            confs = [clamp01(conf_tx_link.get(m, 0.6) if side == "tx" else conf_rx_link.get(m, 0.6)) for m in members]
            total = sum(vals)
            if total < EPS:
                continue
            mean = total / len(vals)
            # Proposed per-link small scale to reduce deviation (±5%)
            s_props = []
            for v, c in zip(vals, confs):
                weight = (1.0 - c) * (v if v >= ZERO_THRESH else 0.0)
                # Move a fraction toward mean; beta small
                beta = 0.25
                desired = 1.0 + beta * (mean - v) / max(mean, EPS)
                s_prop = clamp(desired, 0.95, 1.05)
                # If weight very small, reduce effect
                if weight <= 0.0:
                    s_prop = 1.0
                s_props.append(s_prop)
            # Renormalize to preserve group sum
            sum_new = sum(v * s for v, s in zip(vals, s_props))
            if sum_new < EPS:
                continue
            renorm = total / sum_new
            for idx, m in enumerate(members):
                s_eff = s_props[idx] * renorm
                if side == "tx":
                    hard_tx[m] = hard_tx[m] * s_eff
                    scaled_tx_factor[m] *= s_eff
                    pen = abs(s_eff - 1.0)
                    conf_tx_link[m] *= clamp01(1.0 - 0.2 * clamp01(pen))  # lighter penalty for smoothing
                else:
                    hard_rx[m] = hard_rx[m] * s_eff
                    scaled_rx_factor[m] *= s_eff
                    pen = abs(s_eff - 1.0)
                    conf_rx_link[m] *= clamp01(1.0 - 0.2 * clamp01(pen))

    # Helper to run two-tier targeted scaling for one side
    def targeted_scale(router: str, side: str, if_list: List[str], tau_r: float):
        # Compute current and target sums
        sum_tx = sum(hard_tx.get(i, 0.0) for i in if_list)
        sum_rx = sum(hard_rx.get(i, 0.0) for i in if_list)
        current = sum_tx if side == "tx" else sum_rx
        target = sum_rx if side == "tx" else sum_tx
        if max(current, target) < EPS:
            return
        mismatch = rel_diff(sum_tx, sum_rx)
        if mismatch <= tau_r:
            return

        # Bundle shared pass for dominant bundle
        delta = target - current
        delta = apply_bundle_shared(router, side, target, current, if_list)
        # Update current after bundle
        sum_tx = sum(hard_tx.get(i, 0.0) for i in if_list)
        sum_rx = sum(hard_rx.get(i, 0.0) for i in if_list)
        current = sum_tx if side == "tx" else sum_rx
        target = sum_rx if side == "tx" else sum_tx
        delta = target - current

        # Prepare per-interface values and confidences
        vals = {i: (hard_tx[i] if side == "tx" else hard_rx[i]) for i in if_list}
        confs = {i: clamp01(conf_tx_link.get(i, 0.6) if side == "tx" else conf_rx_link.get(i, 0.6)) for i in if_list}

        def apply_tier(conf_lo: float, conf_hi: float, clip_hi: float, delta_in: float) -> float:
            # Select eligible interfaces
            elig = [i for i in if_list if vals[i] >= ZERO_THRESH and conf_lo <= confs[i] < conf_hi]
            if not elig:
                return delta_in
            weights = {i: (1.0 - confs[i]) * vals[i] for i in elig}
            # denom over v*w (as per linearized change derivation)
            denom_vw = sum(vals[i] * weights[i] for i in elig)
            if denom_vw < EPS:
                return delta_in
            # Solve for k in sum(v*0.6*k*w) = delta_in
            k = delta_in / (0.6 * denom_vw)
            cap_abs = 0.5 * abs(delta_in) if len(elig) >= 2 else None
            applied_total = 0.0
            for i in elig:
                v = vals[i]
                w = weights[i]
                # Proposed scale
                scale_raw = 1.0 + 0.6 * k * w
                scale_eff = clamp(scale_raw, 0.90, clip_hi)
                # Dominance cap per-interface change
                change_i = v * (scale_eff - 1.0)
                if cap_abs is not None and abs(change_i) > cap_abs:
                    scale_eff = 1.0 + math.copysign(cap_abs, change_i) / max(v, EPS)
                    change_i = v * (scale_eff - 1.0)
                # Apply
                if side == "tx":
                    hard_tx[i] = v * scale_eff
                    scaled_tx_factor[i] *= scale_eff
                    pen = abs(scale_eff - 1.0)
                    conf_tx_link[i] *= clamp01(1.0 - 0.4 * clamp01(pen))
                    if pen >= 0.10 or scale_eff <= 0.90 or scale_eff >= clip_hi:
                        clip_hit_tx[i] = True
                    if pen > 0.08:
                        strong_scaled_tx[i] = True
                else:
                    hard_rx[i] = v * scale_eff
                    scaled_rx_factor[i] *= scale_eff
                    pen = abs(scale_eff - 1.0)
                    conf_rx_link[i] *= clamp01(1.0 - 0.4 * clamp01(pen))
                    if pen >= 0.10 or scale_eff <= 0.90 or scale_eff >= clip_hi:
                        clip_hit_rx[i] = True
                    if pen > 0.08:
                        strong_scaled_rx[i] = True
                applied_total += change_i
            return delta_in - applied_total

        # Tier 1: low-confidence (<0.70), allow up to 12% clip
        delta = apply_tier(conf_lo=-1.0, conf_hi=0.70, clip_hi=1.12, delta_in=delta)

        # Recompute residual mismatch threshold check for tier 2
        # If residual still > 0.5 * tau_r * side magnitude, run tier 2
        side_mag = max(current, target, EPS)
        if abs(delta) > 0.5 * tau_r * side_mag:
            delta = apply_tier(conf_lo=0.70, conf_hi=0.85, clip_hi=1.10, delta_in=delta)

        # If still residual > tau_r * side magnitude, perform small uniform damped scaling
        # across active interfaces on adjusted side
        # This minimizes remaining imbalance with conservative bounds
        # Recompute current after tiers
        sum_tx2 = sum(hard_tx.get(i, 0.0) for i in if_list)
        sum_rx2 = sum(hard_rx.get(i, 0.0) for i in if_list)
        current2 = sum_tx2 if side == "tx" else sum_rx2
        target2 = sum_rx2 if side == "tx" else sum_tx2
        mismatch2 = rel_diff(sum_tx2, sum_rx2)
        if mismatch2 > tau_r:
            alpha = clamp(target2 / max(current2, EPS), 0.95, 1.05)
            alpha_eff = 1.0 + 0.4 * (alpha - 1.0)
            for i in if_list:
                v = hard_tx[i] if side == "tx" else hard_rx[i]
                if v < ZERO_THRESH:
                    continue
                if side == "tx":
                    hard_tx[i] = v * alpha_eff
                    scaled_tx_factor[i] *= alpha_eff
                    pen = abs(alpha_eff - 1.0)
                    conf_tx_link[i] *= clamp01(1.0 - 0.3 * clamp01(pen))
                    if pen >= 0.10 or alpha_eff <= 0.90 or alpha_eff >= 1.10:
                        clip_hit_tx[i] = True
                    if pen > 0.08:
                        strong_scaled_tx[i] = True
                else:
                    hard_rx[i] = v * alpha_eff
                    scaled_rx_factor[i] *= alpha_eff
                    pen = abs(alpha_eff - 1.0)
                    conf_rx_link[i] *= clamp01(1.0 - 0.3 * clamp01(pen))
                    if pen >= 0.10 or alpha_eff <= 0.90 or alpha_eff >= 1.10:
                        clip_hit_rx[i] = True
                    if pen > 0.08:
                        strong_scaled_rx[i] = True

        # Intra-bundle smoothing (preserve sums)
        bundle_smoothing(router, side, if_list)

    # Decide which side to adjust per-router and apply targeted scaling
    for router, if_list in router_ifaces.items():
        if len(if_list) <= 1:
            continue
        # Compute router mismatch and adaptive tau
        sum_tx = sum(hard_tx.get(i, 0.0) for i in if_list)
        sum_rx = sum(hard_rx.get(i, 0.0) for i in if_list)
        if max(sum_tx, sum_rx) < EPS:
            continue
        n_active_tx = sum(1 for i in if_list if hard_tx.get(i, 0.0) >= ZERO_THRESH)
        n_active_rx = sum(1 for i in if_list if hard_rx.get(i, 0.0) >= ZERO_THRESH)
        tau_r = tau_router(max(n_active_tx, n_active_rx))
        if rel_diff(sum_tx, sum_rx) <= tau_r:
            continue
        # Choose side with lower aggregate confidence
        c_tx_total = sum(conf_tx_link.get(i, 0.6) for i in if_list)
        c_rx_total = sum(conf_rx_link.get(i, 0.6) for i in if_list)
        if c_tx_total < c_rx_total:
            targeted_scale(router, "tx", if_list, tau_r)
        else:
            targeted_scale(router, "rx", if_list, tau_r)

    # Stage 2.5: Post-projection link re-sync (adaptive saturating gain; guarded and attenuated)
    for a, data in telemetry.items():
        b = peers.get(a)
        if not b or a > b:
            continue  # process each pair once

        ra = router_of.get(a, "")
        rb = router_of.get(b, "")

        # a->b direction
        tx_a = hard_tx.get(a, 0.0)
        rx_b = hard_rx.get(b, 0.0)
        ca = clamp01(conf_tx_link.get(a, 0.6))
        cb = clamp01(conf_rx_link.get(b, 0.6))
        th_ab = tau_h_dir(tx_a, rx_b, ca, cb)
        mis_ab = rel_diff(tx_a, rx_b)
        if mis_ab > th_ab and max(tx_a, rx_b) >= ZERO_THRESH:
            if not (strong_scaled_tx.get(a, False) or strong_scaled_rx.get(b, False)):
                mean_ab = 0.5 * (tx_a + rx_b)
                gap_norm = clamp01((mis_ab - th_ab) / max(th_ab, 1e-9))
                f_gain = 0.4 * sigmoid(5.0 * (gap_norm - 0.5))
                # Attenuate by maximum adjacent router imbalance
                att = clamp01(1.0 - max(router_imbalance(ra), router_imbalance(rb)))
                f_gain *= att
                if ca < cb:
                    gap = max(0.0, cb - ca)
                    f = min(0.4, gap) * f_gain
                    hard_tx[a] = (1.0 - f) * tx_a + f * mean_ab
                    conf_tx_link[a] *= 0.97
                elif cb < ca:
                    gap = max(0.0, ca - cb)
                    f = min(0.4, gap) * f_gain
                    hard_rx[b] = (1.0 - f) * rx_b + f * mean_ab
                    conf_rx_link[b] *= 0.97
                else:
                    # Tiny bilateral nudge with ±2% clip when both low-confidence
                    if ca < 0.7 and cb < 0.7:
                        f_bi = min(0.10, f_gain)
                        tx_new = (1.0 - f_bi) * tx_a + f_bi * mean_ab
                        rx_new = (1.0 - f_bi) * rx_b + f_bi * mean_ab
                        tx_clip = max(tx_a * 0.98, min(tx_a * 1.02, tx_new))
                        rx_clip = max(rx_b * 0.98, min(rx_b * 1.02, rx_new))
                        hard_tx[a] = tx_clip
                        hard_rx[b] = rx_clip
                        conf_tx_link[a] *= 0.96
                        conf_rx_link[b] *= 0.96

        # b->a direction
        tx_b = hard_tx.get(b, 0.0)
        rx_a = hard_rx.get(a, 0.0)
        cb_tx = clamp01(conf_tx_link.get(b, 0.6))
        ca_rx = clamp01(conf_rx_link.get(a, 0.6))
        th_ba = tau_h_dir(tx_b, rx_a, cb_tx, ca_rx)
        mis_ba = rel_diff(tx_b, rx_a)
        if mis_ba > th_ba and max(tx_b, rx_a) >= ZERO_THRESH:
            if not (strong_scaled_tx.get(b, False) or strong_scaled_rx.get(a, False)):
                mean_ba = 0.5 * (tx_b + rx_a)
                gap_norm = clamp01((mis_ba - th_ba) / max(th_ba, 1e-9))
                f_gain = 0.4 * sigmoid(5.0 * (gap_norm - 0.5))
                att = clamp01(1.0 - max(router_imbalance(ra), router_imbalance(rb)))
                f_gain *= att
                if cb_tx < ca_rx:
                    gap = max(0.0, ca_rx - cb_tx)
                    f = min(0.4, gap) * f_gain
                    hard_tx[b] = (1.0 - f) * tx_b + f * mean_ba
                    conf_tx_link[b] *= 0.97
                elif ca_rx < cb_tx:
                    gap = max(0.0, cb_tx - ca_rx)
                    f = min(0.4, gap) * f_gain
                    hard_rx[a] = (1.0 - f) * rx_a + f * mean_ba
                    conf_rx_link[a] *= 0.97
                else:
                    if cb_tx < 0.7 and ca_rx < 0.7:
                        f_bi = min(0.10, f_gain)
                        tx_new = (1.0 - f_bi) * tx_b + f_bi * mean_ba
                        rx_new = (1.0 - f_bi) * rx_a + f_bi * mean_ba
                        tx_clip = max(tx_b * 0.98, min(tx_b * 1.02, tx_new))
                        rx_clip = max(rx_a * 0.98, min(rx_a * 1.02, rx_new))
                        hard_tx[b] = tx_clip
                        hard_rx[a] = rx_clip
                        conf_tx_link[b] *= 0.96
                        conf_rx_link[a] *= 0.96

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

    # Conservation-preserving bundle finishing pass for parallel links (tighten symmetry without altering router totals)
    bundles: Dict[Tuple[str, str], List[Tuple[str, str]]] = {}
    seen_pairs_fin = set()
    for a, data_a in telemetry.items():
        b = peers.get(a)
        if not b:
            continue
        key = tuple(sorted([a, b]))
        if key in seen_pairs_fin:
            continue
        seen_pairs_fin.add(key)
        ra = telemetry[a].get("local_router")
        rb = telemetry[b].get("local_router")
        if not isinstance(ra, str) or not isinstance(rb, str):
            continue
        rp = tuple(sorted([ra, rb]))
        bundles.setdefault(rp, []).append((a, b))

    def _bundle_direction_align(pairs: List[Tuple[str, str]], ab_dir: bool = True):
        if not pairs:
            return
        es, ws, rs, idxs = [], [], [], []
        for (a, b) in pairs:
            if ab_dir:
                tx = hard_tx.get(a, 0.0)
                rx = hard_rx.get(b, 0.0)
                c = 0.5 * (clamp01(conf_tx_link.get(a, 0.6)) + clamp01(conf_rx_link.get(b, 0.6)))
            else:
                tx = hard_tx.get(b, 0.0)
                rx = hard_rx.get(a, 0.0)
                c = 0.5 * (clamp01(conf_tx_link.get(b, 0.6)) + clamp01(conf_rx_link.get(a, 0.6)))
            if max(tx, rx) < ZERO_THRESH:
                continue
            e = tx - rx
            rate = max(tx, rx)
            w = (1.0 - c) * rate
            es.append(e)
            ws.append(w)
            rs.append(rate)
            idxs.append((a, b))
        if len(es) <= 1:
            return
        # Robust center using median residual
        es_sorted = sorted(es)
        mid = len(es_sorted) // 2
        e_center = es_sorted[mid] if len(es_sorted) % 2 == 1 else 0.5 * (es_sorted[mid - 1] + es_sorted[mid])
        # Build deltas with adaptive gamma, then weight and re-center to zero-sum
        base = []
        for k, e in enumerate(es):
            if ab_dir:
                tx = hard_tx.get(idxs[k][0], 0.0)
                rx = hard_rx.get(idxs[k][1], 0.0)
                c1 = clamp01(conf_tx_link.get(idxs[k][0], 0.6))
                c2 = clamp01(conf_rx_link.get(idxs[k][1], 0.6))
            else:
                tx = hard_tx.get(idxs[k][1], 0.0)
                rx = hard_rx.get(idxs[k][0], 0.0)
                c1 = clamp01(conf_tx_link.get(idxs[k][1], 0.6))
                c2 = clamp01(conf_rx_link.get(idxs[k][0], 0.6))
            mismatch = rel_diff(tx, rx)
            tau_loc = tau_h_dir(tx, rx, c1, c2)
            gamma = min(0.25, 0.5 * tau_loc / max(mismatch, 1e-9))
            base.append(-gamma * (e - e_center))
        wbar = (sum(ws) / len(ws)) if ws else 0.0
        scaled = [base[k] * (ws[k] / max(wbar, EPS)) for k in range(len(base))]
        mean_scaled = sum(scaled) / len(scaled)
        deltas = [d - mean_scaled for d in scaled]
        # Apply with ±3% clip per-link, conservation by construction (zero-sum)
        for k, (a, b) in enumerate(idxs):
            clip = 0.03 * rs[k]
            di = max(-clip, min(clip, deltas[k]))
            if ab_dir:
                hard_tx[a] = max(0.0, hard_tx.get(a, 0.0) + di)
                hard_rx[b] = max(0.0, hard_rx.get(b, 0.0) - di)
            else:
                hard_tx[b] = max(0.0, hard_tx.get(b, 0.0) + di)
                hard_rx[a] = max(0.0, hard_rx.get(a, 0.0) - di)

    for rp, pairs in bundles.items():
        _bundle_direction_align(pairs, ab_dir=True)
        _bundle_direction_align(pairs, ab_dir=False)

    # Router imbalance AFTER projection (for confidence context and soft-zero rule)
    router_imbalance_after: Dict[str, float] = {}
    for router, if_list in router_ifaces.items():
        if not if_list:
            router_imbalance_after[router] = 0.0
            continue
        sum_tx_after = sum(hard_tx.get(i, 0.0) for i in if_list)
        sum_rx_after = sum(hard_rx.get(i, 0.0) for i in if_list)
        router_imbalance_after[router] = rel_diff(sum_tx_after, sum_rx_after)

    # Soft-zero stabilization post-projection: snap tiny links to 0 if adjacent routers balanced
    seen_pairs = set()
    for a, data in telemetry.items():
        b = data.get("connected_to")
        if not isinstance(b, str) or b not in telemetry:
            continue
        key = tuple(sorted([a, b]))
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        tx_a = hard_tx.get(a, 0.0)
        rx_b = hard_rx.get(b, 0.0)
        tx_b = hard_tx.get(b, 0.0)
        rx_a = hard_rx.get(a, 0.0)
        if max(tx_a, rx_b, tx_b, rx_a) < 2.0 * ZERO_THRESH:
            ra = router_of.get(a)
            rb = router_of.get(b)
            # Local adaptive TAU per router
            def tau_for_router(rid: str) -> float:
                if rid not in router_ifaces:
                    return 0.05
                ifs = router_ifaces[rid]
                n_tx = sum(1 for i in ifs if hard_tx.get(i, 0.0) >= ZERO_THRESH)
                n_rx = sum(1 for i in ifs if hard_rx.get(i, 0.0) >= ZERO_THRESH)
                return tau_router(max(n_tx, n_rx))
            if router_imbalance_after.get(ra, 0.0) <= tau_for_router(ra) and router_imbalance_after.get(rb, 0.0) <= tau_for_router(rb):
                hard_tx[a] = 0.0
                hard_rx[b] = 0.0
                hard_tx[b] = 0.0
                hard_rx[a] = 0.0
                conf_tx_link[a] = max(conf_tx_link.get(a, 0.6), 0.95)
                conf_rx_link[b] = max(conf_rx_link.get(b, 0.6), 0.95)
                conf_tx_link[b] = max(conf_tx_link.get(b, 0.6), 0.95)
                conf_rx_link[a] = max(conf_rx_link.get(a, 0.6), 0.95)

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

        # Improvement credit: how much symmetry mismatch improved (pre vs post)
        imp_tx = clamp01(pre_tx - post_mismatch_tx_dir.get(if_id, 0.4))
        imp_rx = clamp01(pre_rx - post_mismatch_rx_dir.get(if_id, 0.4))

        # Blend confidence components (weights sum to 1.0)
        conf_tx_final = clamp01(
            0.20 * base_tx_conf +
            0.16 * red_tx +
            0.28 * fin_sym_tx +
            0.12 * ch_tx +
            0.08 * router_factor_after +
            0.08 * scale_tx_term +
            0.08 * imp_tx
        )
        conf_rx_final = clamp01(
            0.20 * base_rx_conf +
            0.16 * red_rx +
            0.28 * fin_sym_rx +
            0.12 * ch_rx +
            0.08 * router_factor_after +
            0.08 * scale_rx_term +
            0.08 * imp_rx
        )

        # Confidence refinements: clip-hit penalty, strong-scale penalty, and untouched boost
        if clip_hit_tx.get(if_id, False) or abs(scaled_tx_factor.get(if_id, 1.0) - 1.0) >= 0.10:
            conf_tx_final *= 0.95
        if clip_hit_rx.get(if_id, False) or abs(scaled_rx_factor.get(if_id, 1.0) - 1.0) >= 0.10:
            conf_rx_final *= 0.95
        # Small penalty when strong scaling (>8%) occurred
        if strong_scaled_tx.get(if_id, False):
            conf_tx_final *= 0.97
        if strong_scaled_rx.get(if_id, False):
            conf_rx_final *= 0.97
        if change_tx < 0.01 and fin_sym_tx >= (1.0 - tau_h_dir(rep_tx, rep_tx, base_tx_conf, base_tx_conf)):
            conf_tx_final = min(0.98, conf_tx_final + 0.02)
        if change_rx < 0.01 and fin_sym_rx >= (1.0 - tau_h_dir(rep_rx, rep_rx, base_rx_conf, base_rx_conf)):
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

    # Final confidence touch-up: residual-informed blend then peer smoothing when both ends are up
    for i, data in telemetry.items():
        peer = data.get('connected_to')
        if not isinstance(peer, str) or peer not in telemetry or i not in result or peer not in result:
            continue
        if result[i]['interface_status'][1] != 'up' or result[peer]['interface_status'][1] != 'up':
            continue
        # Symmetry residuals
        mis_tx = rel_diff(safe_rate(result[i]['tx_rate'][1]), safe_rate(result[peer]['rx_rate'][1]))
        mis_rx = rel_diff(safe_rate(result[i]['rx_rate'][1]), safe_rate(result[peer]['tx_rate'][1]))
        # 70/30 residual-informed touch-up
        old_tx_c = clamp01(result[i]['tx_rate'][2])
        old_rx_c = clamp01(result[i]['rx_rate'][2])
        base_tx_c = clamp01(0.70 * old_tx_c + 0.30 * clamp01(1.0 - mis_tx))
        base_rx_c = clamp01(0.70 * old_rx_c + 0.30 * clamp01(1.0 - mis_rx))
        # 10% peer smoothing
        peer_rx_c = clamp01(result[peer]['rx_rate'][2])
        peer_tx_c = clamp01(result[peer]['tx_rate'][2])
        final_tx_c = clamp01(0.90 * base_tx_c + 0.10 * peer_rx_c)
        final_rx_c = clamp01(0.90 * base_rx_c + 0.10 * peer_tx_c)
        result[i]['tx_rate'] = (
            result[i]['tx_rate'][0],
            result[i]['tx_rate'][1],
            final_tx_c
        )
        result[i]['rx_rate'] = (
            result[i]['rx_rate'][0],
            result[i]['rx_rate'][1],
            final_rx_c
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