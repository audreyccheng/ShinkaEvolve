# EVOLVE-BLOCK-START
"""
Pipeline Hodor repair with bundle-aware projection and expected-penalty side selection.

Phases:
  0) Signal collection and topology shaping
  1) Early soft-zero pre-pass on near-zero links when adjacent routers are balanced
  2) Link hardening: adaptive fusion with mismatch-adaptive bias
  3) Router projection:
     - Choose side by expected-penalty lookahead
     - Dominance-aware targeted scaling (two-tier) with Huber-like caps
     - Bundle-shared scaling for dominant bundles (>=60% share) + intra-bundle smoothing
     - Micro high-confidence finishing tier
  4) Post-projection link re-sync (adjust lower-confidence side with scaling guard)
  5) Status repair (symmetry/traffic-aware)
  6) Confidence calibration: invariant-based blend, strong-scale guards, residual-informed blend, peer smoothing
"""
from typing import Dict, Any, Tuple, List
import math


def repair_network_telemetry(telemetry: Dict[str, Dict[str, Any]],
                             topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:
    # Constants and tolerances
    EPS = 1e-9
    ZERO_THRESH = 0.1  # Mbps deemed near-zero

    # Safe helpers
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
        return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x

    def clamp(x: float, lo: float, hi: float) -> float:
        return lo if x < lo else hi if x > hi else x

    # Adaptive tolerances
    def tau_h_dir(v1: float, v2: float, c1: float = None, c2: float = None) -> float:
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
        base = 0.05 * math.sqrt(2.0 / max(2, n_active))
        return clamp(base, 0.03, 0.07)

    # Build peer mapping
    peers: Dict[str, str] = {}
    for if_id, data in telemetry.items():
        peer = data.get("connected_to")
        if isinstance(peer, str) and peer in telemetry:
            peers[if_id] = peer

    # Build router->interfaces using topology plus local_router fallback
    router_ifaces: Dict[str, List[str]] = {}
    for r, ifs in topology.items():
        router_ifaces.setdefault(r, [])
        for i in ifs:
            if i in telemetry:
                router_ifaces[r].append(i)
    for if_id, data in telemetry.items():
        r = data.get("local_router")
        if r is None:
            r = f"unknown_router::{if_id}"
        router_ifaces.setdefault(r, [])
        if if_id not in router_ifaces[r]:
            router_ifaces[r].append(if_id)

    # Originals and status
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

    # Utility: router sums and imbalance for arbitrary tx/rx maps
    def router_sums(tx_map: Dict[str, float], rx_map: Dict[str, float], rid: str) -> Tuple[float, float]:
        if rid not in router_ifaces:
            return 0.0, 0.0
        ifs = router_ifaces[rid]
        return sum(tx_map.get(i, 0.0) for i in ifs), sum(rx_map.get(i, 0.0) for i in ifs)

    def router_imbalance(tx_map: Dict[str, float], rx_map: Dict[str, float], rid: str) -> float:
        stx, srx = router_sums(tx_map, rx_map, rid)
        return rel_diff(stx, srx)

    # Phase 1: Early soft-zero pre-pass based on original readings and router balance
    hard_tx: Dict[str, float] = {}
    hard_rx: Dict[str, float] = {}
    conf_tx_link: Dict[str, float] = {}
    conf_rx_link: Dict[str, float] = {}
    pre_mismatch_tx: Dict[str, float] = {}
    pre_mismatch_rx: Dict[str, float] = {}

    # Compute adaptive router tau using counts of active original directions
    def tau_for_router_orig(rid: str) -> float:
        if rid not in router_ifaces:
            return 0.05
        ifs = router_ifaces[rid]
        n_tx = sum(1 for i in ifs if orig_tx.get(i, 0.0) >= ZERO_THRESH)
        n_rx = sum(1 for i in ifs if orig_rx.get(i, 0.0) >= ZERO_THRESH)
        return tau_router(max(n_tx, n_rx))

    visited_pairs = set()
    for a, data_a in telemetry.items():
        b = data_a.get("connected_to")
        if not isinstance(b, str) or b not in telemetry:
            continue
        key = tuple(sorted([a, b]))
        if key in visited_pairs:
            continue
        visited_pairs.add(key)

        # Check near-zero on all four original directions
        if max(orig_tx.get(a, 0.0), orig_rx.get(b, 0.0), orig_tx.get(b, 0.0), orig_rx.get(a, 0.0)) < 1.5 * ZERO_THRESH:
            ra = router_of.get(a)
            rb = router_of.get(b)
            imb_ra = router_imbalance(orig_tx, orig_rx, ra)
            imb_rb = router_imbalance(orig_tx, orig_rx, rb)
            if imb_ra <= tau_for_router_orig(ra) and imb_rb <= tau_for_router_orig(rb):
                # Snap all four to zero with high confidence
                hard_tx[a] = 0.0
                hard_rx[a] = 0.0
                hard_tx[b] = 0.0
                hard_rx[b] = 0.0
                conf_tx_link[a] = 0.96
                conf_rx_link[a] = 0.96
                conf_tx_link[b] = 0.96
                conf_rx_link[b] = 0.96

    # Phase 2: Link hardening with adaptive fusion (mismatch-adaptive bias)
    def fuse_direction(v_local: float, v_peer: float, s_local: str, s_peer: str) -> Tuple[float, float]:
        mismatch = rel_diff(v_local, v_peer)
        th = tau_h_dir(v_local, v_peer)

        # Both near-zero => strong zero
        if max(v_local, v_peer) < ZERO_THRESH:
            return 0.0, 0.95

        # Within tolerance: keep local
        if mismatch <= th:
            return v_local, 0.95

        # Moderate mismatch: average
        if mismatch <= 0.10:
            fused = 0.5 * v_local + 0.5 * v_peer
            return fused, clamp01(1.0 - mismatch)

        # Large mismatch: adaptive bias toward peer
        local_zero_bias = (v_local < ZERO_THRESH and v_peer >= ZERO_THRESH)
        peer_zero_bias = (v_peer < ZERO_THRESH and v_local >= ZERO_THRESH)
        local_down = (s_local == "down")
        peer_down = (s_peer == "down")

        if local_zero_bias:
            return v_peer, clamp01(1.0 - mismatch)
        if peer_zero_bias:
            return v_local, clamp01(1.0 - mismatch)

        beta_base = 0.7
        beta_scale = 0.2 * clamp01((mismatch - 0.10) / 0.20)
        beta_bonus = 0.1 if (local_down or local_zero_bias) else 0.0
        beta_malus = -0.1 if peer_down else 0.0
        beta = clamp(beta_base + beta_scale + beta_bonus + beta_malus, 0.7, 0.9)
        fused = (1.0 - beta) * v_local + beta * v_peer
        return fused, clamp01(1.0 - mismatch)

    # Process all interfaces, pairing when possible
    processed = set()
    for if_id, data in telemetry.items():
        if if_id in processed:
            continue
        peer = peers.get(if_id)
        if not peer:
            if if_id not in hard_tx:
                hard_tx[if_id] = orig_tx[if_id]
                hard_rx[if_id] = orig_rx[if_id]
                conf_tx_link[if_id] = conf_tx_link.get(if_id, 0.6)
                conf_rx_link[if_id] = conf_rx_link.get(if_id, 0.6)
                pre_mismatch_tx[if_id] = pre_mismatch_tx.get(if_id, 0.4)
                pre_mismatch_rx[if_id] = pre_mismatch_rx.get(if_id, 0.4)
            processed.add(if_id)
            continue

        a, b = if_id, peer
        if b in processed:
            processed.add(a)
            continue
        processed.add(a)
        processed.add(b)

        # If early soft-zero already set both, skip fusion
        if (hard_tx.get(a) == 0.0 and hard_rx.get(a) == 0.0 and
            hard_tx.get(b) == 0.0 and hard_rx.get(b) == 0.0):
            # Pre-mismatch for confidence context
            pre_mismatch_tx[a] = rel_diff(orig_tx[a], orig_rx[b])
            pre_mismatch_rx[a] = rel_diff(orig_rx[a], orig_tx[b])
            pre_mismatch_tx[b] = rel_diff(orig_tx[b], orig_rx[a])
            pre_mismatch_rx[b] = rel_diff(orig_rx[b], orig_tx[a])
            continue

        a_tx, a_rx = orig_tx[a], orig_rx[a]
        b_tx, b_rx = orig_tx[b], orig_rx[b]
        sa, sb = status[a], status[b]

        # Both ends down: force zeros
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

        # Record directional pre-mismatch
        diff_ab = rel_diff(a_tx, b_rx)
        diff_ba = rel_diff(b_tx, a_rx)
        pre_mismatch_tx[a] = diff_ab
        pre_mismatch_rx[b] = diff_ab
        pre_mismatch_tx[b] = diff_ba
        pre_mismatch_rx[a] = diff_ba

        fused_ab, c_ab = fuse_direction(a_tx, b_rx, sa, sb)
        fused_ba, c_ba = fuse_direction(b_tx, a_rx, sb, sa)

        # Map directions to sides
        hard_tx[a] = fused_ab
        hard_rx[b] = fused_ab
        hard_tx[b] = fused_ba
        hard_rx[a] = fused_ba

        conf_tx_link[a] = c_ab
        conf_rx_link[b] = c_ab
        conf_tx_link[b] = c_ba
        conf_rx_link[a] = c_ba

    # Ensure all interfaces have hardened values
    for if_id in telemetry:
        if if_id not in hard_tx:
            hard_tx[if_id] = orig_tx[if_id]
            conf_tx_link[if_id] = conf_tx_link.get(if_id, 0.6)
        if if_id not in hard_rx:
            hard_rx[if_id] = orig_rx[if_id]
            conf_rx_link[if_id] = conf_rx_link.get(if_id, 0.6)
        if if_id not in pre_mismatch_tx:
            pre_mismatch_tx[if_id] = 0.4
        if if_id not in pre_mismatch_rx:
            pre_mismatch_rx[if_id] = 0.4

    # Phase 3: Router projection with bundle-aware, dominance-aware targeted scaling

    # Track cumulative scaling for confidence calibration and guards
    scaled_tx_factor: Dict[str, float] = {i: 1.0 for i in telemetry}
    scaled_rx_factor: Dict[str, float] = {i: 1.0 for i in telemetry}
    strong_scaled_tx: Dict[str, bool] = {i: False for i in telemetry}
    strong_scaled_rx: Dict[str, bool] = {i: False for i in telemetry}

    # Expected penalty lookahead for choosing side (tx or rx)
    def expected_penalty(router: str, side: str) -> float:
        ifs = router_ifaces.get(router, [])
        cur = sum((hard_tx[i] if side == "tx" else hard_rx[i]) for i in ifs)
        tgt = sum((hard_rx[i] if side == "tx" else hard_tx[i]) for i in ifs)
        if max(cur, tgt) < EPS:
            return 0.0
        delta = tgt - cur
        # Two pseudo tiers combined to estimate penalty
        elig_all = []
        for i in ifs:
            v = hard_tx[i] if side == "tx" else hard_rx[i]
            c = clamp01(conf_tx_link.get(i, 0.6) if side == "tx" else conf_rx_link.get(i, 0.6))
            if v >= ZERO_THRESH and c < 0.85:
                w = (1.0 - c) * (v ** 0.95)
                elig_all.append((i, v, c, w))
        if not elig_all:
            # Uniform small scaling estimate
            alpha = clamp(tgt / max(cur, EPS), 0.95, 1.05)
            alpha_eff = 1.0 + 0.4 * (alpha - 1.0)
            return sum(abs(alpha_eff - 1.0) for _ in ifs)  # rough
        sumW = sum(w for (_, _, _, w) in elig_all)
        if sumW < EPS:
            return abs(delta)  # high cost
        # Clip any single weight to <= 0.5 * sumW
        cap = 0.5 * sumW
        weights = []
        for (i, v, c, w) in elig_all:
            weights.append(min(w, cap))
        denom = sum((v * w) for (_, v, _, w) in zip(elig_all, weights))
        if denom < EPS:
            return abs(delta)
        k = delta / (0.6 * denom)
        penalty_sum = 0.0
        for (_, v, c, w) in zip(elig_all, [e[1] for e in elig_all], [e[2] for e in elig_all], weights):
            scale = 1.0 + 0.6 * (k * w)
            # tier-dependent clip: use moderate worst-case
            clip_hi = 1.12 if c < 0.70 else 1.10
            scale = clamp(scale, 0.90, clip_hi)
            penalty_sum += abs(scale - 1.0)
        return penalty_sum

    # Bundle helpers
    def dominant_bundle_members(router: str, side: str, if_list: List[str]) -> List[str]:
        # Group by remote_router
        bundle_sum: Dict[str, float] = {}
        for i in if_list:
            rr = telemetry.get(i, {}).get("remote_router")
            v = hard_tx[i] if side == "tx" else hard_rx[i]
            bundle_sum[rr] = bundle_sum.get(rr, 0.0) + v
        total = sum(bundle_sum.values())
        if total < EPS:
            return []
        # Find bundle with >=60% share
        rr_dom = None
        share = 0.0
        for rr, s in bundle_sum.items():
            sh = s / total
            if sh >= 0.60 and sh > share:
                share = sh
                rr_dom = rr
        if rr_dom is None:
            return []
        return [i for i in if_list if telemetry.get(i, {}).get("remote_router") == rr_dom]

    def apply_bundle_shared(router: str, side: str, if_list: List[str]):
        # Apply shared alpha for the dominant bundle if any
        members = dominant_bundle_members(router, side, if_list)
        if not members:
            return
        cur = sum((hard_tx[i] if side == "tx" else hard_rx[i]) for i in if_list)
        tgt = sum((hard_rx[i] if side == "tx" else hard_tx[i]) for i in if_list)
        if max(cur, tgt) < EPS:
            return
        alpha = clamp(tgt / max(cur, EPS), 0.85, 1.15)
        alpha_eff = 1.0 + 0.6 * (alpha - 1.0)
        for i in members:
            old = hard_tx[i] if side == "tx" else hard_rx[i]
            new = old * alpha_eff
            if side == "tx":
                hard_tx[i] = new
                scaled_tx_factor[i] *= alpha_eff
                pen = abs(alpha_eff - 1.0)
                conf_tx_link[i] *= clamp01(1.0 - 0.4 * clamp01(pen))
                if pen > 0.08:
                    strong_scaled_tx[i] = True
            else:
                hard_rx[i] = new
                scaled_rx_factor[i] *= alpha_eff
                pen = abs(alpha_eff - 1.0)
                conf_rx_link[i] *= clamp01(1.0 - 0.4 * clamp01(pen))
                if pen > 0.08:
                    strong_scaled_rx[i] = True

    def bundle_smoothing(router: str, side: str, if_list: List[str]):
        # Smooth within each bundle (remote_router) with ±5% per-link, preserving bundle sum
        bundles: Dict[str, List[str]] = {}
        for i in if_list:
            rr = telemetry.get(i, {}).get("remote_router")
            bundles.setdefault(rr, []).append(i)
        for rr, members in bundles.items():
            if len(members) < 2:
                continue
            vals = [hard_tx[i] if side == "tx" else hard_rx[i] for i in members]
            confs = [clamp01(conf_tx_link.get(i, 0.6) if side == "tx" else conf_rx_link.get(i, 0.6)) for i in members]
            total = sum(vals)
            if total < EPS:
                continue
            mean = total / len(vals)
            # Propose small move towards mean with damping
            props = []
            for v, c in zip(vals, confs):
                beta = 0.25
                desired = 1.0 + beta * (mean - v) / max(mean, EPS)
                s_prop = clamp(desired, 0.95, 1.05)
                if (1.0 - c) * v <= 0.0:
                    s_prop = 1.0
                props.append(s_prop)
            new_total = sum(v * s for v, s in zip(vals, props))
            if new_total < EPS:
                continue
            renorm = total / new_total
            for idx, i in enumerate(members):
                s_eff = props[idx] * renorm
                if side == "tx":
                    hard_tx[i] *= s_eff
                    scaled_tx_factor[i] *= s_eff
                    conf_tx_link[i] *= clamp01(1.0 - 0.2 * abs(s_eff - 1.0))
                else:
                    hard_rx[i] *= s_eff
                    scaled_rx_factor[i] *= s_eff
                    conf_rx_link[i] *= clamp01(1.0 - 0.2 * abs(s_eff - 1.0))

    # Targeted scaling tiers with dominance caps and Huber-like clipping
    def targeted_scale(router: str, side: str, if_list: List[str], tau_r: float):
        # Bundle-shared pre-pass
        apply_bundle_shared(router, side, if_list)

        # Helper to apply one tier
        def apply_tier(conf_lo: float, conf_hi: float, clip_hi: float, delta_in: float) -> float:
            # Select eligible
            elig = []
            for i in if_list:
                v = hard_tx[i] if side == "tx" else hard_rx[i]
                c = clamp01(conf_tx_link.get(i, 0.6) if side == "tx" else conf_rx_link.get(i, 0.6))
                if v >= ZERO_THRESH and conf_lo <= c < conf_hi:
                    w = (1.0 - c) * (v ** 0.95)
                    elig.append((i, v, c, w))
            if not elig:
                return delta_in
            sumW = sum(w for (_, _, _, w) in elig)
            if sumW <= 0.0:
                return delta_in
            cap = 0.5 * sumW
            weights = {}
            for (i, v, c, w) in elig:
                weights[i] = min(w, cap)
            denom = sum((v * weights[i]) for (i, v, _, _) in elig)
            if denom < EPS:
                return delta_in
            k = delta_in / (0.6 * denom)
            applied = 0.0
            # Additional absolute dominance cap per-interface: ≤ 50% of router correction
            abs_cap = 0.5 * abs(delta_in) if len(elig) >= 2 else None
            for (i, v, c, _) in elig:
                w = weights[i]
                scale_raw = 1.0 + 0.6 * (k * w)
                scale_eff = clamp(scale_raw, 0.90, clip_hi)
                change_i = v * (scale_eff - 1.0)
                if abs_cap is not None and abs(change_i) > abs_cap:
                    scale_eff = 1.0 + math.copysign(abs_cap, change_i) / max(v, EPS)
                    change_i = v * (scale_eff - 1.0)
                if side == "tx":
                    hard_tx[i] = v * scale_eff
                    scaled_tx_factor[i] *= scale_eff
                    pen = abs(scale_eff - 1.0)
                    conf_tx_link[i] *= clamp01(1.0 - 0.4 * clamp01(pen))
                    if pen > 0.08:
                        strong_scaled_tx[i] = True
                else:
                    hard_rx[i] = v * scale_eff
                    scaled_rx_factor[i] *= scale_eff
                    pen = abs(scale_eff - 1.0)
                    conf_rx_link[i] *= clamp01(1.0 - 0.4 * clamp01(pen))
                    if pen > 0.08:
                        strong_scaled_rx[i] = True
                applied += change_i
            return delta_in - applied

        # Compute initial mismatch
        sum_tx = sum(hard_tx[i] for i in if_list)
        sum_rx = sum(hard_rx[i] for i in if_list)
        current = sum_tx if side == "tx" else sum_rx
        target = sum_rx if side == "tx" else sum_tx
        if max(current, target) < EPS:
            return
        if rel_diff(sum_tx, sum_rx) <= tau_r:
            return
        delta = target - current

        # Tier 1: low confidence < 0.70 with slightly wider clip
        delta = apply_tier(-1.0, 0.70, 1.12, delta)

        # Recompute residual and magnitude
        sum_tx = sum(hard_tx[i] for i in if_list)
        sum_rx = sum(hard_rx[i] for i in if_list)
        current = sum_tx if side == "tx" else sum_rx
        target = sum_rx if side == "tx" else sum_tx
        side_mag = max(current, target, EPS)

        # Tier 2: moderate confidence [0.70, 0.85)
        if abs(delta) > 0.5 * tau_r * side_mag:
            delta = apply_tier(0.70, 0.85, 1.10, delta)

        # Uniform small damped scaling if necessary
        sum_tx2 = sum(hard_tx[i] for i in if_list)
        sum_rx2 = sum(hard_rx[i] for i in if_list)
        if rel_diff(sum_tx2, sum_rx2) > tau_r:
            current2 = sum_tx2 if side == "tx" else sum_rx2
            target2 = sum_rx2 if side == "tx" else sum_tx2
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
                    if pen > 0.08:
                        strong_scaled_tx[i] = True
                else:
                    hard_rx[i] = v * alpha_eff
                    scaled_rx_factor[i] *= alpha_eff
                    pen = abs(alpha_eff - 1.0)
                    conf_rx_link[i] *= clamp01(1.0 - 0.3 * clamp01(pen))
                    if pen > 0.08:
                        strong_scaled_rx[i] = True

        # Intra-bundle smoothing (preserve sums)
        bundle_smoothing(router, side, if_list)

        # Micro finishing tier on high-confidence links if residual persists > 0.6 * tau_r
        sum_tx3 = sum(hard_tx[i] for i in if_list)
        sum_rx3 = sum(hard_rx[i] for i in if_list)
        if rel_diff(sum_tx3, sum_rx3) > 0.6 * tau_r:
            # tiny adjustments on conf >= 0.85
            def micro(conf_lo: float, clip_hi: float, damp: float):
                cur = sum(hard_tx[i] for i in if_list) if side == "tx" else sum(hard_rx[i] for i in if_list)
                tgt = sum(hard_rx[i] for i in if_list) if side == "tx" else sum(hard_tx[i] for i in if_list)
                delta_m = tgt - cur
                elig = []
                for i in if_list:
                    v = hard_tx[i] if side == "tx" else hard_rx[i]
                    c = clamp01(conf_tx_link.get(i, 0.6) if side == "tx" else conf_rx_link.get(i, 0.6))
                    if v >= ZERO_THRESH and c >= conf_lo:
                        w = (v ** 0.95)  # confidence high, rely more on rate
                        elig.append((i, v, w))
                if not elig:
                    return
                denom = sum(v * w for (_, v, w) in elig)
                if denom < EPS:
                    return
                k = delta_m / (damp * denom)
                for (i, v, w) in elig:
                    scale = 1.0 + damp * (k * w)
                    scale = clamp(scale, 1.0 / clip_hi, clip_hi)  # tiny clip around 1.0
                    if side == "tx":
                        hard_tx[i] = v * scale
                        scaled_tx_factor[i] *= scale
                        pen = abs(scale - 1.0)
                        conf_tx_link[i] *= clamp01(1.0 - 0.2 * clamp01(pen))
                        if pen > 0.08:
                            strong_scaled_tx[i] = True
                    else:
                        hard_rx[i] = v * scale
                        scaled_rx_factor[i] *= scale
                        pen = abs(scale - 1.0)
                        conf_rx_link[i] *= clamp01(1.0 - 0.2 * clamp01(pen))
                        if pen > 0.08:
                            strong_scaled_rx[i] = True

            micro(conf_lo=0.85, clip_hi=1.03, damp=0.25)

    # Iterate routers with expected-penalty side selection
    for router, if_list in router_ifaces.items():
        if len(if_list) <= 1:
            continue
        # Check imbalance
        sum_tx_r = sum(hard_tx[i] for i in if_list)
        sum_rx_r = sum(hard_rx[i] for i in if_list)
        if max(sum_tx_r, sum_rx_r) < EPS:
            continue
        n_active_tx = sum(1 for i in if_list if hard_tx[i] >= ZERO_THRESH)
        n_active_rx = sum(1 for i in if_list if hard_rx[i] >= ZERO_THRESH)
        tau_r = tau_router(max(n_active_tx, n_active_rx))
        if rel_diff(sum_tx_r, sum_rx_r) <= tau_r:
            continue
        # Choose side with lower expected penalty (fallback to lower aggregate confidence)
        pen_tx = expected_penalty(router, "tx")
        pen_rx = expected_penalty(router, "rx")
        if pen_tx < pen_rx:
            targeted_scale(router, "tx", if_list, tau_r)
        elif pen_rx < pen_tx:
            targeted_scale(router, "rx", if_list, tau_r)
        else:
            c_tx_total = sum(conf_tx_link.get(i, 0.6) for i in if_list)
            c_rx_total = sum(conf_rx_link.get(i, 0.6) for i in if_list)
            targeted_scale(router, "tx" if c_tx_total < c_rx_total else "rx", if_list, tau_r)

    # Phase 4: Post-projection gentle link re-sync (lower-confidence side, guarded)
    def router_imbalance_after_router(rid: str) -> float:
        if rid not in router_ifaces:
            return 0.0
        ifs = router_ifaces[rid]
        stx = sum(hard_tx[i] for i in ifs)
        srx = sum(hard_rx[i] for i in ifs)
        return rel_diff(stx, srx)

    processed_pairs = set()
    for a, data_a in telemetry.items():
        b = data_a.get("connected_to")
        if not isinstance(b, str) or b not in telemetry:
            continue
        key = tuple(sorted([a, b]))
        if key in processed_pairs:
            continue
        processed_pairs.add(key)

        # a->b
        tx_a = hard_tx.get(a, 0.0)
        rx_b = hard_rx.get(b, 0.0)
        ca = clamp01(conf_tx_link.get(a, 0.6))
        cb = clamp01(conf_rx_link.get(b, 0.6))
        th_ab = tau_h_dir(tx_a, rx_b, ca, cb)
        mis_ab = rel_diff(tx_a, rx_b)
        if mis_ab > th_ab and max(tx_a, rx_b) >= ZERO_THRESH:
            if not (strong_scaled_tx.get(a, False) or strong_scaled_rx.get(b, False)):
                mean_ab = 0.5 * (tx_a + rx_b)
                if ca < cb:
                    f = min(0.4, max(0.0, cb - ca))
                    att = clamp01(1.0 - router_imbalance_after_router(router_of.get(a, "")))
                    f *= att
                    hard_tx[a] = (1.0 - f) * tx_a + f * mean_ab
                    conf_tx_link[a] *= 0.97
                elif cb < ca:
                    f = min(0.4, max(0.0, ca - cb))
                    att = clamp01(1.0 - router_imbalance_after_router(router_of.get(b, "")))
                    f *= att
                    hard_rx[b] = (1.0 - f) * rx_b + f * mean_ab
                    conf_rx_link[b] *= 0.97

        # b->a
        tx_b = hard_tx.get(b, 0.0)
        rx_a = hard_rx.get(a, 0.0)
        cb_tx = clamp01(conf_tx_link.get(b, 0.6))
        ca_rx = clamp01(conf_rx_link.get(a, 0.6))
        th_ba = tau_h_dir(tx_b, rx_a, cb_tx, ca_rx)
        mis_ba = rel_diff(tx_b, rx_a)
        if mis_ba > th_ba and max(tx_b, rx_a) >= ZERO_THRESH:
            if not (strong_scaled_tx.get(b, False) or strong_scaled_rx.get(a, False)):
                mean_ba = 0.5 * (tx_b + rx_a)
                if cb_tx < ca_rx:
                    f = min(0.4, max(0.0, ca_rx - cb_tx))
                    att = clamp01(1.0 - router_imbalance_after_router(router_of.get(b, "")))
                    f *= att
                    hard_tx[b] = (1.0 - f) * tx_b + f * mean_ba
                    conf_tx_link[b] *= 0.97
                elif ca_rx < cb_tx:
                    f = min(0.4, max(0.0, cb_tx - ca_rx))
                    att = clamp01(1.0 - router_imbalance_after_router(router_of.get(a, "")))
                    f *= att
                    hard_rx[a] = (1.0 - f) * rx_a + f * mean_ba
                    conf_rx_link[a] *= 0.97

    # Phase 5: Status repair
    repaired_status: Dict[str, str] = {}
    status_conf: Dict[str, float] = {}
    processed = set()
    for if_id in telemetry:
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
        rep_local, rep_peer = s_local, s_peer
        c_local, c_peer = 0.95, 0.95
        if s_local == "down" and s_peer == "down":
            rep_local = "down"
            rep_peer = "down"
            c_local, c_peer = 0.98, 0.98
        elif s_local != s_peer:
            link_has_traffic = (hard_tx.get(if_id, 0.0) >= ZERO_THRESH or hard_rx.get(if_id, 0.0) >= ZERO_THRESH or
                                hard_tx.get(peer, 0.0) >= ZERO_THRESH or hard_rx.get(peer, 0.0) >= ZERO_THRESH)
            if link_has_traffic:
                rep_local = "up"
                rep_peer = "up"
                c_local = 0.7
                c_peer = 0.7
            else:
                c_local = 0.6
                c_peer = 0.6
        repaired_status[if_id] = rep_local
        repaired_status[peer] = rep_peer
        status_conf[if_id] = c_local
        status_conf[peer] = c_peer
        processed.add(if_id)
        processed.add(peer)

    # Phase 6: Confidence calibration and result assembly

    # Router imbalance AFTER projection
    router_imbalance_after: Dict[str, float] = {}
    for router, if_list in router_ifaces.items():
        if not if_list:
            router_imbalance_after[router] = 0.0
            continue
        stx = sum(hard_tx.get(i, 0.0) for i in if_list)
        srx = sum(hard_rx.get(i, 0.0) for i in if_list)
        router_imbalance_after[router] = rel_diff(stx, srx)

    # Final per-direction symmetry residuals
    post_mismatch_tx_dir: Dict[str, float] = {}
    post_mismatch_rx_dir: Dict[str, float] = {}
    for if_id, data in telemetry.items():
        peer = peers.get(if_id)
        if peer:
            post_mismatch_tx_dir[if_id] = rel_diff(hard_tx.get(if_id, 0.0), hard_rx.get(peer, 0.0))
            post_mismatch_rx_dir[if_id] = rel_diff(hard_rx.get(if_id, 0.0), hard_tx.get(peer, 0.0))
        else:
            post_mismatch_tx_dir[if_id] = 0.4
            post_mismatch_rx_dir[if_id] = 0.4

    # Compose output
    result: Dict[str, Dict[str, Tuple]] = {}
    for if_id, data in telemetry.items():
        rep_tx = hard_tx.get(if_id, orig_tx[if_id])
        rep_rx = hard_rx.get(if_id, orig_rx[if_id])

        change_tx = rel_diff(orig_tx[if_id], rep_tx)
        change_rx = rel_diff(orig_rx[if_id], rep_rx)

        pre_tx = pre_mismatch_tx.get(if_id, 0.4)
        pre_rx = pre_mismatch_rx.get(if_id, 0.4)

        fin_sym_tx = clamp01(1.0 - post_mismatch_tx_dir.get(if_id, 0.4))
        fin_sym_rx = clamp01(1.0 - post_mismatch_rx_dir.get(if_id, 0.4))

        r = router_of.get(if_id, None)
        router_penalty_after = router_imbalance_after.get(r, 0.0) if r is not None else 0.0
        router_factor_after = clamp01(1.0 - min(0.5, router_penalty_after))

        base_tx_conf = clamp01(conf_tx_link.get(if_id, 0.6))
        base_rx_conf = clamp01(conf_rx_link.get(if_id, 0.6))

        red_tx = clamp01(1.0 - pre_tx)
        red_rx = clamp01(1.0 - pre_rx)

        ch_tx = clamp01(1.0 - change_tx)
        ch_rx = clamp01(1.0 - change_rx)

        scale_tx_term = clamp01(1.0 - min(0.5, abs(scaled_tx_factor.get(if_id, 1.0) - 1.0)))
        scale_rx_term = clamp01(1.0 - min(0.5, abs(scaled_rx_factor.get(if_id, 1.0) - 1.0)))

        conf_tx_final = clamp01(
            0.22 * base_tx_conf +
            0.18 * red_tx +
            0.28 * fin_sym_tx +
            0.12 * ch_tx +
            0.10 * router_factor_after +
            0.10 * scale_tx_term
        )
        conf_rx_final = clamp01(
            0.22 * base_rx_conf +
            0.18 * red_rx +
            0.28 * fin_sym_rx +
            0.12 * ch_rx +
            0.10 * router_factor_after +
            0.10 * scale_rx_term
        )

        # Strong-scale guard penalties and untouched boosts
        if abs(scaled_tx_factor.get(if_id, 1.0) - 1.0) > 0.08:
            conf_tx_final *= 0.97
        if abs(scaled_rx_factor.get(if_id, 1.0) - 1.0) > 0.08:
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

    # Residual-informed confidence touch-up and peer-opposite smoothing (only when both ends up)
    for i, data in telemetry.items():
        p = data.get('connected_to')
        if not isinstance(p, str) or p not in telemetry or i not in result or p not in result:
            continue
        if result[i]['interface_status'][1] != 'up' or result[p]['interface_status'][1] != 'up':
            continue
        mis_tx = rel_diff(safe_rate(result[i]['tx_rate'][1]), safe_rate(result[p]['rx_rate'][1]))
        mis_rx = rel_diff(safe_rate(result[i]['rx_rate'][1]), safe_rate(result[p]['tx_rate'][1]))
        old_tx_c = clamp01(result[i]['tx_rate'][2])
        old_rx_c = clamp01(result[i]['rx_rate'][2])
        base_tx_c = clamp01(0.70 * old_tx_c + 0.30 * clamp01(1.0 - mis_tx))
        base_rx_c = clamp01(0.70 * old_rx_c + 0.30 * clamp01(1.0 - mis_rx))
        peer_rx_c = clamp01(result[p]['rx_rate'][2])
        peer_tx_c = clamp01(result[p]['tx_rate'][2])
        result[i]['tx_rate'] = (result[i]['tx_rate'][0], result[i]['tx_rate'][1],
                                clamp01(0.90 * base_tx_c + 0.10 * peer_rx_c))
        result[i]['rx_rate'] = (result[i]['rx_rate'][0], result[i]['rx_rate'][1],
                                clamp01(0.90 * base_rx_c + 0.10 * peer_tx_c))

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