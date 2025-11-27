# EVOLVE-BLOCK-START
"""
Bundle-Huber flow sync: adaptive link fusion, dominance- and bundle-aware router projection,
confidence-gap link resync, and calibrated confidences.

Implements:
- Link hardening via adaptive fusion with status/zero awareness and severity-biased snap.
- Router flow conservation via dominance-aware targeted scaling (rate^0.95 weights, 50% cap),
  bundle shared scaling for dominant parallel links, and intra-bundle smoothing.
- Expected-penalty lookahead to choose the router side to adjust.
- Micro finishing tier on high-confidence links for stubborn residuals.
- Confidence calibration with scale penalties, residual satisfaction, strong-scale guards, and peer smoothing.
"""
from typing import Dict, Any, Tuple, List
import math


def repair_network_telemetry(telemetry: Dict[str, Dict[str, Any]],
                             topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:
    # Constants
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

    def clamp(x: float, lo: float, hi: float) -> float:
        return lo if x < lo else (hi if x > hi else x)

    def tau_h_dir(v1: float, v2: float, c1: float = None, c2: float = None) -> float:
        # Adaptive symmetry tolerance:
        # stricter (1.5%) when both >100 and confidences high; looser (3%) for low rates or low confidences.
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
        # Adaptive router imbalance tolerance (clamped)
        base = 0.05 * math.sqrt(2.0 / max(2, n_active))
        return clamp(base, 0.03, 0.07)

    # Build peers map
    peers: Dict[str, str] = {}
    for if_id, data in telemetry.items():
        p = data.get("connected_to")
        if isinstance(p, str) and p in telemetry:
            peers[if_id] = p

    # Build router->interfaces from topology with fallback to local_router
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

    # Originals and status
    orig_tx: Dict[str, float] = {}
    orig_rx: Dict[str, float] = {}
    status_raw: Dict[str, str] = {}
    for if_id, data in telemetry.items():
        orig_tx[if_id] = safe_rate(data.get("tx_rate", 0.0))
        orig_rx[if_id] = safe_rate(data.get("rx_rate", 0.0))
        s = data.get("interface_status", "unknown")
        status_raw[if_id] = s if s in ("up", "down") else "unknown"

    # Stage 1: Link hardening (adaptive fusion)
    hard_tx: Dict[str, float] = {}
    hard_rx: Dict[str, float] = {}
    conf_tx_link: Dict[str, float] = {}
    conf_rx_link: Dict[str, float] = {}
    pre_mismatch_tx: Dict[str, float] = {}
    pre_mismatch_rx: Dict[str, float] = {}

    def fuse_direction(v_local: float, v_peer: float, s_local: str, s_peer: str) -> Tuple[float, float]:
        # Severity-aware beta and status/zero-aware decision
        mis = rel_diff(v_local, v_peer)
        th = tau_h_dir(v_local, v_peer)
        if max(v_local, v_peer) < ZERO_THRESH:
            return 0.0, 0.95
        if mis <= th:
            return v_local, 0.95
        if mis <= 0.10:
            fused = 0.5 * v_local + 0.5 * v_peer
            return fused, clamp01(1.0 - mis)
        # Large mismatch: adaptive snap-to-peer bias
        # Base beta 0.7, add up to +0.2 as mismatch grows beyond 10%, status/zero cues add ±0.1
        beta = 0.7 + 0.2 * clamp01((mis - 0.10) / 0.20)
        if (v_local < ZERO_THRESH and v_peer >= ZERO_THRESH) or (s_local == "down" and s_peer == "up"):
            beta = min(0.9, beta + 0.1)
        if (v_peer < ZERO_THRESH and v_local >= ZERO_THRESH) or (s_peer == "down" and s_local == "up"):
            beta = max(0.7, beta - 0.1)
        fused = (1.0 - beta) * v_local + beta * v_peer
        return fused, clamp01(1.0 - mis)

    # Pairwise hardening
    visited = set()
    for a in telemetry.keys():
        if a in visited:
            continue
        b = peers.get(a)
        if not b:
            # Isolated interface: keep as-is with conservative confidence
            hard_tx[a] = orig_tx[a]
            hard_rx[a] = orig_rx[a]
            conf_tx_link[a] = 0.6
            conf_rx_link[a] = 0.6
            pre_mismatch_tx[a] = 0.4
            pre_mismatch_rx[a] = 0.4
            visited.add(a)
            continue
        visited.add(a); visited.add(b)
        a_tx, a_rx = orig_tx[a], orig_rx[a]
        b_tx, b_rx = orig_tx[b], orig_rx[b]
        sa, sb = status_raw[a], status_raw[b]

        # Both ends down -> zeros with high confidence
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

        # Record pre-fusion directional mismatches
        mis_ab = rel_diff(a_tx, b_rx)
        mis_ba = rel_diff(b_tx, a_rx)
        pre_mismatch_tx[a] = mis_ab
        pre_mismatch_rx[b] = mis_ab
        pre_mismatch_tx[b] = mis_ba
        pre_mismatch_rx[a] = mis_ba

        fused_ab, c_ab = fuse_direction(a_tx, b_rx, sa, sb)
        fused_ba, c_ba = fuse_direction(b_tx, a_rx, sb, sa)

        # Map fused directions to both ends to enforce per-link symmetry
        hard_tx[a] = fused_ab
        hard_rx[b] = fused_ab
        hard_tx[b] = fused_ba
        hard_rx[a] = fused_ba

        conf_tx_link[a] = c_ab
        conf_rx_link[b] = c_ab
        conf_tx_link[b] = c_ba
        conf_rx_link[a] = c_ba

    # Ensure completeness for any missed interfaces
    for i in telemetry.keys():
        if i not in hard_tx:
            hard_tx[i] = orig_tx[i]
            conf_tx_link[i] = 0.6
        if i not in hard_rx:
            hard_rx[i] = orig_rx[i]
            conf_rx_link[i] = 0.6
        if i not in pre_mismatch_tx:
            pre_mismatch_tx[i] = 0.4
        if i not in pre_mismatch_rx:
            pre_mismatch_rx[i] = 0.4

    # Stage 2: Router-level projection with bundle + dominance-aware targeted scaling
    scaled_tx_factor: Dict[str, float] = {i: 1.0 for i in telemetry}
    scaled_rx_factor: Dict[str, float] = {i: 1.0 for i in telemetry}
    strong_scaled_tx: Dict[str, bool] = {i: False for i in telemetry}
    strong_scaled_rx: Dict[str, bool] = {i: False for i in telemetry}
    clip_hit_tx: Dict[str, bool] = {i: False for i in telemetry}
    clip_hit_rx: Dict[str, bool] = {i: False for i in telemetry}

    def router_imbalance_now(rid: str) -> float:
        if rid not in router_ifaces: return 0.0
        ifs = router_ifaces[rid]
        stx = sum(hard_tx.get(i, 0.0) for i in ifs)
        srx = sum(hard_rx.get(i, 0.0) for i in ifs)
        return rel_diff(stx, srx)

    # Expected-penalty lookahead for choosing side
    def expected_penalty(side: str, if_list: List[str], delta: float) -> float:
        # Use weights w = (1 - conf) * rate^0.95; predict per-interface relative scale change
        if side == "tx":
            vals = [hard_tx[i] for i in if_list]
            confs = [clamp01(conf_tx_link.get(i, 0.6)) for i in if_list]
        else:
            vals = [hard_rx[i] for i in if_list]
            confs = [clamp01(conf_rx_link.get(i, 0.6)) for i in if_list]
        w = [(1.0 - c) * (max(v, 0.0) ** 0.95 if v >= ZERO_THRESH else 0.0) for v, c in zip(vals, confs)]
        # Clip individual weights to <= 0.5 * total to avoid dominance
        sumw = sum(w)
        if sumw < EPS:
            # fallback penalty ~ magnitude
            return abs(delta)
        w = [min(x, 0.5 * sumw) for x in w]
        denom_vw = sum(v * wi for v, wi in zip(vals, w))
        if denom_vw < EPS:
            return abs(delta)
        k = delta / (0.6 * denom_vw)
        # Huber-like penalty on anticipated |scale-1|
        penalty = 0.0
        for v, wi, c in zip(vals, w, confs):
            if v <= 0.0 or wi <= 0.0:
                continue
            scale_i = 1.0 + 0.6 * k * wi
            x = abs(scale_i - 1.0)
            # Huber: quadratic for small x, linear for large
            x0 = 0.05
            rho = (x * x / (2 * x0)) if x <= x0 else (x - x0 / 2)
            # weight by expected confidence loss (1 - conf)
            penalty += (1.0 - c) * rho
        return penalty

    def apply_bundle_shared(side: str, if_list: List[str], target_sum: float, current_sum: float) -> float:
        # Returns residual delta after bundle scaling
        delta = target_sum - current_sum
        total = max(current_sum, EPS)
        # Group by remote_router
        bundles: Dict[Any, List[str]] = {}
        for i in if_list:
            rr = telemetry.get(i, {}).get("remote_router")
            bundles.setdefault(rr, []).append(i)
        # Compute shares
        side_vals = {i: (hard_tx[i] if side == "tx" else hard_rx[i]) for i in if_list}
        shares: Dict[Any, float] = {rr: sum(side_vals[m] for m in members) / total for rr, members in bundles.items()}
        # Find dominant bundle
        dom_rr = None
        dom_share = 0.0
        for rr, sh in shares.items():
            if sh >= 0.60 and sh > dom_share:
                dom_rr, dom_share = rr, sh
        if dom_rr is None:
            return delta
        # Shared alpha on dominant bundle
        ratio = clamp(target_sum / max(current_sum, EPS), 0.85, 1.15)
        alpha_eff = 1.0 + 0.6 * (ratio - 1.0)
        applied = 0.0
        for i in bundles[dom_rr]:
            v = side_vals[i]
            new = v * alpha_eff
            change = new - v
            if side == "tx":
                hard_tx[i] = new
                scaled_tx_factor[i] *= alpha_eff
                pen = abs(alpha_eff - 1.0)
                conf_tx_link[i] *= clamp01(1.0 - 0.4 * clamp01(pen))
                if pen > 0.08: strong_scaled_tx[i] = True
                if alpha_eff <= 0.90 or alpha_eff >= 1.10: clip_hit_tx[i] = True
            else:
                hard_rx[i] = new
                scaled_rx_factor[i] *= alpha_eff
                pen = abs(alpha_eff - 1.0)
                conf_rx_link[i] *= clamp01(1.0 - 0.4 * clamp01(pen))
                if pen > 0.08: strong_scaled_rx[i] = True
                if alpha_eff <= 0.90 or alpha_eff >= 1.10: clip_hit_rx[i] = True
            applied += change
        return delta - applied

    def bundle_smoothing(side: str, if_list: List[str]):
        # Small ±5% smoothing within bundles preserving bundle sum
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
            if total < EPS: continue
            mean = total / len(vals)
            # propose small move toward mean
            s_prop = []
            for v, c in zip(vals, confs):
                beta = 0.25
                desired = 1.0 + beta * (mean - v) / max(mean, EPS)
                desired = clamp(desired, 0.95, 1.05)
                # lower effect when confidence high
                desired = 1.0 + (desired - 1.0) * (1.0 - c)
                s_prop.append(desired)
            sum_new = sum(v * s for v, s in zip(vals, s_prop))
            if sum_new < EPS: continue
            renorm = total / sum_new
            for idx, m in enumerate(members):
                s_eff = s_prop[idx] * renorm
                if side == "tx":
                    hard_tx[m] *= s_eff
                    scaled_tx_factor[m] *= s_eff
                    conf_tx_link[m] *= clamp01(1.0 - 0.2 * clamp01(abs(s_eff - 1.0)))
                else:
                    hard_rx[m] *= s_eff
                    scaled_rx_factor[m] *= s_eff
                    conf_rx_link[m] *= clamp01(1.0 - 0.2 * clamp01(abs(s_eff - 1.0)))

    def targeted_scale(side: str, if_list: List[str], tau_r: float):
        # Compute sums
        sum_tx = sum(hard_tx[i] for i in if_list)
        sum_rx = sum(hard_rx[i] for i in if_list)
        current = sum_tx if side == "tx" else sum_rx
        target = sum_rx if side == "tx" else sum_tx
        if max(current, target) < EPS:
            return
        mismatch = rel_diff(sum_tx, sum_rx)
        if mismatch <= tau_r:
            return

        # Bundle shared scaling first for dominant bundle
        resid = apply_bundle_shared(side, if_list, target, current)
        # Recompute sums after bundle
        sum_tx = sum(hard_tx[i] for i in if_list)
        sum_rx = sum(hard_rx[i] for i in if_list)
        current = sum_tx if side == "tx" else sum_rx
        target = sum_rx if side == "tx" else sum_tx
        resid = target - current

        # Two-tier targeted per-interface scaling with dominance cap and weight clip
        def apply_tier(conf_lo: float, conf_hi: float, clip_hi: float, delta_in: float) -> float:
            if delta_in == 0.0:
                return 0.0
            if side == "tx":
                vals = {i: hard_tx[i] for i in if_list}
                confs = {i: clamp01(conf_tx_link.get(i, 0.6)) for i in if_list}
            else:
                vals = {i: hard_rx[i] for i in if_list}
                confs = {i: clamp01(conf_rx_link.get(i, 0.6)) for i in if_list}
            elig = [i for i in if_list if vals[i] >= ZERO_THRESH and conf_lo <= confs[i] < conf_hi]
            if not elig:
                return delta_in
            raw_w = {i: (1.0 - confs[i]) * (vals[i] ** 0.95) for i in elig}
            sumw = sum(raw_w.values())
            if sumw < EPS:
                return delta_in
            # individual weight cap
            cap_w = 0.5 * sumw
            w = {i: min(raw_w[i], cap_w) for i in elig}
            denom_vw = sum(vals[i] * w[i] for i in elig)
            if denom_vw < EPS:
                return delta_in
            k = delta_in / (0.6 * denom_vw)
            applied = 0.0
            # dominance cap on per-interface absolute change
            cap_abs = 0.5 * abs(delta_in) if len(elig) >= 2 else None
            for i in elig:
                v = vals[i]
                wi = w[i]
                scale_raw = 1.0 + 0.6 * k * wi
                scale_eff = clamp(scale_raw, 0.90, clip_hi)
                change_i = v * (scale_eff - 1.0)
                if cap_abs is not None and abs(change_i) > cap_abs:
                    scale_eff = 1.0 + math.copysign(cap_abs, change_i) / max(v, EPS)
                    change_i = v * (scale_eff - 1.0)
                if side == "tx":
                    hard_tx[i] = v * scale_eff
                    scaled_tx_factor[i] *= scale_eff
                    pen = abs(scale_eff - 1.0)
                    conf_tx_link[i] *= clamp01(1.0 - 0.4 * clamp01(pen))
                    if pen > 0.08: strong_scaled_tx[i] = True
                    if scale_eff <= 0.90 or scale_eff >= clip_hi: clip_hit_tx[i] = True
                else:
                    hard_rx[i] = v * scale_eff
                    scaled_rx_factor[i] *= scale_eff
                    pen = abs(scale_eff - 1.0)
                    conf_rx_link[i] *= clamp01(1.0 - 0.4 * clamp01(pen))
                    if pen > 0.08: strong_scaled_rx[i] = True
                    if scale_eff <= 0.90 or scale_eff >= clip_hi: clip_hit_rx[i] = True
                applied += change_i
            return delta_in - applied

        resid = apply_tier(-1.0, 0.70, 1.12, resid)
        # Check residual magnitude vs tau
        sum_tx2 = sum(hard_tx[i] for i in if_list)
        sum_rx2 = sum(hard_rx[i] for i in if_list)
        cur2 = sum_tx2 if side == "tx" else sum_rx2
        tgt2 = sum_rx2 if side == "tx" else sum_tx2
        side_mag = max(cur2, tgt2, EPS)
        if abs(resid) > 0.5 * tau_r * side_mag:
            resid = apply_tier(0.70, 0.85, 1.10, resid)

        # Micro finishing tier on high-confidence links if residual still notable
        sum_tx3 = sum(hard_tx[i] for i in if_list)
        sum_rx3 = sum(hard_rx[i] for i in if_list)
        if rel_diff(sum_tx3, sum_rx3) > 0.6 * tau_r:
            # adjust conf >= 0.85 with tiny clip and gentle damping
            if side == "tx":
                vals = {i: hard_tx[i] for i in if_list}
                confs = {i: clamp01(conf_tx_link.get(i, 0.6)) for i in if_list}
            else:
                vals = {i: hard_rx[i] for i in if_list}
                confs = {i: clamp01(conf_rx_link.get(i, 0.6)) for i in if_list}
            elig = [i for i in if_list if vals[i] >= ZERO_THRESH and confs[i] >= 0.85]
            if elig:
                delta = (sum_rx3 - sum_tx3) if side == "tx" else (sum_tx3 - sum_rx3)
                w = {i: vals[i] for i in elig}
                denom = sum(vals[i] * w[i] for i in elig)
                if denom > EPS:
                    k = delta / (0.25 * denom)  # damping 0.25
                    for i in elig:
                        v = vals[i]
                        scale = 1.0 + 0.25 * k * w[i]
                        scale = clamp(scale, 0.97, 1.03)
                        if side == "tx":
                            hard_tx[i] = v * scale
                            scaled_tx_factor[i] *= scale
                            conf_tx_link[i] *= clamp01(1.0 - 0.2 * abs(scale - 1.0))
                        else:
                            hard_rx[i] = v * scale
                            scaled_rx_factor[i] *= scale
                            conf_rx_link[i] *= clamp01(1.0 - 0.2 * abs(scale - 1.0))

        # Intra-bundle smoothing to preserve sums while aligning parallel links
        bundle_smoothing(side, if_list)

    # Router pass: choose side using expected-penalty lookahead
    for r, if_list in router_ifaces.items():
        if len(if_list) <= 1:
            continue
        stx = sum(hard_tx[i] for i in if_list)
        srx = sum(hard_rx[i] for i in if_list)
        if max(stx, srx) < EPS:
            continue
        n_active = max(sum(1 for i in if_list if hard_tx[i] >= ZERO_THRESH),
                       sum(1 for i in if_list if hard_rx[i] >= ZERO_THRESH))
        tau_r = tau_router(n_active)
        if rel_diff(stx, srx) <= tau_r:
            continue
        delta_tx = srx - stx
        delta_rx = stx - srx
        pen_tx = expected_penalty("tx", if_list, delta_tx)
        pen_rx = expected_penalty("rx", if_list, delta_rx)
        side = "tx" if pen_tx < pen_rx else "rx"
        targeted_scale(side, if_list, tau_r)

    # Stage 2.5: Post-projection link re-sync with adaptive beta and guards
    processed_pairs = set()
    for a, data_a in telemetry.items():
        b = data_a.get("connected_to")
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
        mis_ab = rel_diff(tx_a, rx_b)
        if mis_ab > th_ab and max(tx_a, rx_b) >= ZERO_THRESH:
            # adaptive beta with status/zero cues
            sa, sb = status_raw.get(a, "unknown"), status_raw.get(b, "unknown")
            beta = 0.7 + 0.2 * clamp01((mis_ab - 0.10) / 0.20)
            if (tx_a < ZERO_THRESH and rx_b >= ZERO_THRESH) or (sa == "down" and sb == "up"):
                beta = min(0.9, beta + 0.1)
            if (rx_b < ZERO_THRESH and tx_a >= ZERO_THRESH) or (sb == "down" and sa == "up"):
                beta = max(0.7, beta - 0.1)
            mean_ab = (1.0 - beta) * tx_a + beta * rx_b
            # adjust lower-confidence side only, attenuated by router balance
            if not (strong_scaled_tx.get(a, False) or strong_scaled_rx.get(b, False)):
                if ca < cb:
                    att = clamp01(1.0 - router_imbalance_now(router_of.get(a, "")))
                    hard_tx[a] = (1.0 - att) * tx_a + att * mean_ab
                    conf_tx_link[a] *= 0.97
                elif cb < ca:
                    att = clamp01(1.0 - router_imbalance_now(router_of.get(b, "")))
                    hard_rx[b] = (1.0 - att) * rx_b + att * mean_ab
                    conf_rx_link[b] *= 0.97

        # b->a direction
        tx_b = hard_tx.get(b, 0.0)
        rx_a = hard_rx.get(a, 0.0)
        cb_tx = clamp01(conf_tx_link.get(b, 0.6))
        ca_rx = clamp01(conf_rx_link.get(a, 0.6))
        th_ba = tau_h_dir(tx_b, rx_a, cb_tx, ca_rx)
        mis_ba = rel_diff(tx_b, rx_a)
        if mis_ba > th_ba and max(tx_b, rx_a) >= ZERO_THRESH:
            sb, sa = status_raw.get(b, "unknown"), status_raw.get(a, "unknown")
            beta = 0.7 + 0.2 * clamp01((mis_ba - 0.10) / 0.20)
            if (tx_b < ZERO_THRESH and rx_a >= ZERO_THRESH) or (sb == "down" and sa == "up"):
                beta = min(0.9, beta + 0.1)
            if (rx_a < ZERO_THRESH and tx_b >= ZERO_THRESH) or (sa == "down" and sb == "up"):
                beta = max(0.7, beta - 0.1)
            mean_ba = (1.0 - beta) * tx_b + beta * rx_a
            if not (strong_scaled_tx.get(b, False) or strong_scaled_rx.get(a, False)):
                if cb_tx < ca_rx:
                    att = clamp01(1.0 - router_imbalance_now(router_of.get(b, "")))
                    hard_tx[b] = (1.0 - att) * tx_b + att * mean_ba
                    conf_tx_link[b] *= 0.97
                elif ca_rx < cb_tx:
                    att = clamp01(1.0 - router_imbalance_now(router_of.get(a, "")))
                    hard_rx[a] = (1.0 - att) * rx_a + att * mean_ba
                    conf_rx_link[a] *= 0.97

    # Soft-zero post rule: if all four link directions tiny and both routers balanced enough, snap to 0
    seen_pairs = set()
    for a, data_a in telemetry.items():
        b = data_a.get("connected_to")
        if not isinstance(b, str) or b not in telemetry: continue
        key = tuple(sorted([a, b]))
        if key in seen_pairs: continue
        seen_pairs.add(key)
        tx_a, rx_a = hard_tx.get(a, 0.0), hard_rx.get(a, 0.0)
        tx_b, rx_b = hard_tx.get(b, 0.0), hard_rx.get(b, 0.0)
        if max(tx_a, rx_b, tx_b, rx_a) < 1.5 * ZERO_THRESH:
            ra, rb = router_of.get(a), router_of.get(b)
            # local adaptive tau
            def tau_for_router(rid: str) -> float:
                if rid not in router_ifaces: return 0.05
                ifs = router_ifaces[rid]
                n_act = max(sum(1 for i in ifs if hard_tx.get(i, 0.0) >= ZERO_THRESH),
                            sum(1 for i in ifs if hard_rx.get(i, 0.0) >= ZERO_THRESH))
                return tau_router(n_act)
            if router_imbalance_now(ra) <= tau_for_router(ra) and router_imbalance_now(rb) <= tau_for_router(rb):
                hard_tx[a] = 0.0; hard_rx[b] = 0.0
                hard_tx[b] = 0.0; hard_rx[a] = 0.0
                conf_tx_link[a] = max(conf_tx_link.get(a, 0.6), 0.95)
                conf_rx_link[b] = max(conf_rx_link.get(b, 0.6), 0.95)
                conf_tx_link[b] = max(conf_tx_link.get(b, 0.6), 0.95)
                conf_rx_link[a] = max(conf_rx_link.get(a, 0.6), 0.95)

    # Status repair (symmetry- and traffic-aware)
    repaired_status: Dict[str, str] = {}
    status_conf: Dict[str, float] = {}
    handled = set()
    for a in telemetry.keys():
        if a in handled:
            continue
        b = telemetry[a].get("connected_to")
        sa = status_raw.get(a, "unknown")
        if not isinstance(b, str) or b not in telemetry:
            repaired_status[a] = sa
            status_conf[a] = 0.95
            handled.add(a)
            continue
        sb = status_raw.get(b, "unknown")
        any_traffic = (hard_tx.get(a, 0.0) >= ZERO_THRESH or hard_rx.get(a, 0.0) >= ZERO_THRESH or
                       hard_tx.get(b, 0.0) >= ZERO_THRESH or hard_rx.get(b, 0.0) >= ZERO_THRESH)
        if sa == "down" and sb == "down":
            repaired_status[a] = "down"; repaired_status[b] = "down"
            status_conf[a] = 0.98; status_conf[b] = 0.98
        elif sa != sb:
            if any_traffic:
                repaired_status[a] = "up"; repaired_status[b] = "up"
                status_conf[a] = 0.70; status_conf[b] = 0.70
            else:
                repaired_status[a] = sa; repaired_status[b] = sb
                status_conf[a] = 0.6; status_conf[b] = 0.6
        else:
            repaired_status[a] = sa; repaired_status[b] = sb
            status_conf[a] = 0.95; status_conf[b] = 0.95
        handled.add(a); handled.add(b)

    # Confidence calibration components
    # Router imbalance after projection
    router_imbalance_after: Dict[str, float] = {}
    for r, if_list in router_ifaces.items():
        stx = sum(hard_tx.get(i, 0.0) for i in if_list)
        srx = sum(hard_rx.get(i, 0.0) for i in if_list)
        router_imbalance_after[r] = rel_diff(stx, srx)

    # Final symmetry residuals
    post_mismatch_tx_dir: Dict[str, float] = {}
    post_mismatch_rx_dir: Dict[str, float] = {}
    for i in telemetry.keys():
        p = peers.get(i)
        if p:
            post_mismatch_tx_dir[i] = rel_diff(hard_tx.get(i, 0.0), hard_rx.get(p, 0.0))
            post_mismatch_rx_dir[i] = rel_diff(hard_rx.get(i, 0.0), hard_tx.get(p, 0.0))
        else:
            post_mismatch_tx_dir[i] = 0.4
            post_mismatch_rx_dir[i] = 0.4

    # Compose final results
    result: Dict[str, Dict[str, Tuple]] = {}
    for i, data in telemetry.items():
        rep_tx = hard_tx.get(i, orig_tx[i])
        rep_rx = hard_rx.get(i, orig_rx[i])
        change_tx = rel_diff(orig_tx[i], rep_tx)
        change_rx = rel_diff(orig_rx[i], rep_rx)

        pre_tx = pre_mismatch_tx.get(i, 0.4)
        pre_rx = pre_mismatch_rx.get(i, 0.4)
        fin_sym_tx = clamp01(1.0 - post_mismatch_tx_dir.get(i, 0.4))
        fin_sym_rx = clamp01(1.0 - post_mismatch_rx_dir.get(i, 0.4))

        r = router_of.get(i, None)
        router_penalty_after = router_imbalance_after.get(r, 0.0) if r is not None else 0.0
        router_factor_after = clamp01(1.0 - min(0.5, router_penalty_after))

        base_tx_conf = clamp01(conf_tx_link.get(i, 0.6))
        base_rx_conf = clamp01(conf_rx_link.get(i, 0.6))

        red_tx = clamp01(1.0 - pre_tx)
        red_rx = clamp01(1.0 - pre_rx)
        ch_tx = clamp01(1.0 - change_tx)
        ch_rx = clamp01(1.0 - change_rx)

        scale_tx_term = clamp01(1.0 - min(0.5, abs(scaled_tx_factor.get(i, 1.0) - 1.0)))
        scale_rx_term = clamp01(1.0 - min(0.5, abs(scaled_rx_factor.get(i, 1.0) - 1.0)))

        conf_tx_final = clamp01(
            0.23 * base_tx_conf +
            0.18 * red_tx +
            0.28 * fin_sym_tx +
            0.11 * ch_tx +
            0.10 * router_factor_after +
            0.10 * scale_tx_term
        )
        conf_rx_final = clamp01(
            0.23 * base_rx_conf +
            0.18 * red_rx +
            0.28 * fin_sym_rx +
            0.11 * ch_rx +
            0.10 * router_factor_after +
            0.10 * scale_rx_term
        )

        # Strong-scale guards and clip penalties
        if abs(scaled_tx_factor.get(i, 1.0) - 1.0) > 0.08:
            conf_tx_final *= 0.97
        if abs(scaled_rx_factor.get(i, 1.0) - 1.0) > 0.08:
            conf_rx_final *= 0.97
        if clip_hit_tx.get(i, False):
            conf_tx_final *= 0.95
        if clip_hit_rx.get(i, False):
            conf_rx_final *= 0.95
        # Untouched boosts when minimal change and strong invariants
        if change_tx < 0.01 and fin_sym_tx >= (1.0 - tau_h_dir(rep_tx, rep_tx, base_tx_conf, base_tx_conf)):
            conf_tx_final = min(0.98, conf_tx_final + 0.02)
        if change_rx < 0.01 and fin_sym_rx >= (1.0 - tau_h_dir(rep_rx, rep_rx, base_rx_conf, base_rx_conf)):
            conf_rx_final = min(0.98, conf_rx_final + 0.02)

        # Status enforcement: down implies zero counters and calibrated confidence
        rep_status = repaired_status.get(i, status_raw.get(i, "unknown"))
        conf_status = status_conf.get(i, 0.9)
        if rep_status == "down":
            rep_tx = 0.0
            rep_rx = 0.0
            if orig_tx[i] >= ZERO_THRESH or orig_rx[i] >= ZERO_THRESH:
                conf_tx_final = min(conf_tx_final, 0.7)
                conf_rx_final = min(conf_rx_final, 0.7)
            else:
                conf_tx_final = max(conf_tx_final, 0.9)
                conf_rx_final = max(conf_rx_final, 0.9)

        out = {}
        out["rx_rate"] = (orig_rx[i], rep_rx, conf_rx_final)
        out["tx_rate"] = (orig_tx[i], rep_tx, conf_tx_final)
        out["interface_status"] = (status_raw[i], rep_status, conf_status)
        # Copy metadata unchanged
        out["connected_to"] = data.get("connected_to")
        out["local_router"] = data.get("local_router")
        out["remote_router"] = data.get("remote_router")
        result[i] = out

    # Final confidence touch-up: residual-informed blend and peer smoothing when both ends up
    for i, data in telemetry.items():
        p = data.get("connected_to")
        if not isinstance(p, str) or p not in telemetry:
            continue
        if i not in result or p not in result:
            continue
        if result[i]["interface_status"][1] != "up" or result[p]["interface_status"][1] != "up":
            continue
        mis_tx = rel_diff(safe_rate(result[i]["tx_rate"][1]), safe_rate(result[p]["rx_rate"][1]))
        mis_rx = rel_diff(safe_rate(result[i]["rx_rate"][1]), safe_rate(result[p]["tx_rate"][1]))
        old_tx_c = clamp01(result[i]["tx_rate"][2])
        old_rx_c = clamp01(result[i]["rx_rate"][2])
        base_tx_c = clamp01(0.70 * old_tx_c + 0.30 * clamp01(1.0 - mis_tx))
        base_rx_c = clamp01(0.70 * old_rx_c + 0.30 * clamp01(1.0 - mis_rx))
        peer_rx_c = clamp01(result[p]["rx_rate"][2])
        peer_tx_c = clamp01(result[p]["tx_rate"][2])
        final_tx_c = clamp01(0.90 * base_tx_c + 0.10 * peer_rx_c)
        final_rx_c = clamp01(0.90 * base_rx_c + 0.10 * peer_tx_c)
        result[i]["tx_rate"] = (result[i]["tx_rate"][0], result[i]["tx_rate"][1], final_tx_c)
        result[i]["rx_rate"] = (result[i]["rx_rate"][0], result[i]["rx_rate"][1], final_rx_c)

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