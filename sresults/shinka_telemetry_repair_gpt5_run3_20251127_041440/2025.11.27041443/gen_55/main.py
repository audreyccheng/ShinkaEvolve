# EVOLVE-BLOCK-START
"""
Global consensus projection: alternating projections onto link symmetry and router conservation
with bundle-aware smoothing and confidence-calibrated weighted least-change updates.

This approach:
- Builds a global solution by iteratively projecting to invariant sets
  (link symmetry and router flow conservation) using weighted least squares.
- Incorporates bundle-aware smoothing across parallel links.
- Calibrates confidences from final invariant satisfaction and adjustment magnitude.
"""
from typing import Dict, Any, Tuple, List
import math


def repair_network_telemetry(telemetry: Dict[str, Dict[str, Any]],
                             topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:
    # Constants
    TAU_H = 0.02           # symmetry tolerance ~2%
    ZERO_THRESH = 0.1      # Mbps considered near-zero
    EPS = 1e-9

    # Iteration & damping parameters
    MAX_ITERS = 8
    LINK_W_BIAS = 1.0      # weight exponent for confidences in link projection
    ROUTER_DAMP = 0.85     # damping for router projection k
    PER_ITER_CLIP = 0.12   # per-iteration relative change cap (12%)

    # Bundle smoothing parameters
    BUNDLE_SMOOTH_FRACTION = 0.10  # 10% move toward bundle mean per iteration

    # Helper functions
    def safe_rate(x: Any) -> float:
        try:
            v = float(x)
            if not math.isfinite(v) or v < 0:
                return 0.0
            return v
        except Exception:
            return 0.0

    def clamp01(x: float) -> float:
        if x < 0.0: return 0.0
        if x > 1.0: return 1.0
        return x

    def rel_diff(a: float, b: float) -> float:
        m = max(abs(a), abs(b), 1.0)
        return abs(a - b) / m

    def tau_router(n_active: int) -> float:
        base = 0.05 * math.sqrt(2.0 / max(2, n_active))
        return max(0.03, min(0.07, base))

    # Build peers
    peers: Dict[str, str] = {}
    for if_id, data in telemetry.items():
        peer = data.get("connected_to")
        if isinstance(peer, str) and peer in telemetry:
            peers[if_id] = peer

    # Build router_ifaces using topology and fallback to local_router
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

    # Router mapping for quick lookup
    router_of: Dict[str, str] = {}
    for r, ifs in router_ifaces.items():
        for i in ifs:
            router_of[i] = r

    # Originals and status
    orig_tx: Dict[str, float] = {}
    orig_rx: Dict[str, float] = {}
    status: Dict[str, str] = {}
    for if_id, data in telemetry.items():
        orig_tx[if_id] = safe_rate(data.get("tx_rate", 0.0))
        orig_rx[if_id] = safe_rate(data.get("rx_rate", 0.0))
        s = data.get("interface_status", "unknown")
        status[if_id] = s if s in ("up", "down") else "unknown"

    # Initialize working variables
    tx_hat: Dict[str, float] = {i: orig_tx[i] for i in telemetry}
    rx_hat: Dict[str, float] = {i: orig_rx[i] for i in telemetry}

    # Initialize base directional confidence from link mismatches (redundant signals)
    base_conf_tx: Dict[str, float] = {}
    base_conf_rx: Dict[str, float] = {}
    for i, data in telemetry.items():
        p = peers.get(i)
        if p:
            mis_tx = rel_diff(orig_tx[i], orig_rx.get(p, 0.0))  # my_tx vs peer_rx
            mis_rx = rel_diff(orig_rx[i], orig_tx.get(p, 0.0))  # my_rx vs peer_tx
            base_conf_tx[i] = clamp01(1.0 - mis_tx)
            base_conf_rx[i] = clamp01(1.0 - mis_rx)
        else:
            base_conf_tx[i] = 0.6
            base_conf_rx[i] = 0.6

    # Status down => initialize to zero, boost zero confidence a bit
    for i in telemetry:
        if status.get(i) == "down":
            tx_hat[i] = 0.0
            rx_hat[i] = 0.0
            base_conf_tx[i] = max(base_conf_tx[i], 0.9)
            base_conf_rx[i] = max(base_conf_rx[i], 0.9)

    # Build bundles: group by (local_router, remote_router)
    bundles: Dict[Tuple[str, str], List[str]] = {}
    for i, data in telemetry.items():
        lr = data.get("local_router")
        rr = data.get("remote_router")
        key = (str(lr), str(rr))
        bundles.setdefault(key, []).append(i)

    # Utility: compute router imbalance with current hats
    def router_imbalance(router_id: str) -> float:
        if not router_id or router_id not in router_ifaces:
            return 0.0
        ifs = router_ifaces[router_id]
        stx = sum(tx_hat.get(i, 0.0) for i in ifs)
        srx = sum(rx_hat.get(i, 0.0) for i in ifs)
        return rel_diff(stx, srx)

    # Soft-zero pre-snap: snap small links to 0 when both adjacent routers are already balanced
    def soft_zero_pass():
        processed = set()
        for a, data_a in telemetry.items():
            b = peers.get(a)
            if not b:
                continue
            key = tuple(sorted([a, b]))
            if key in processed:
                continue
            processed.add(key)
            # four directions on link
            if max(tx_hat.get(a, 0.0), rx_hat.get(b, 0.0),
                   tx_hat.get(b, 0.0), rx_hat.get(a, 0.0)) < 1.5 * ZERO_THRESH:
                ra = router_of.get(a)
                rb = router_of.get(b)
                # adaptive tau based on current active ports
                def tau_for_router(rid: str) -> float:
                    if rid not in router_ifaces:
                        return 0.05
                    ifs = router_ifaces[rid]
                    n_act = max(
                        sum(1 for i in ifs if tx_hat.get(i, 0.0) >= ZERO_THRESH),
                        sum(1 for i in ifs if rx_hat.get(i, 0.0) >= ZERO_THRESH)
                    )
                    return tau_router(n_act)
                if router_imbalance(ra) <= tau_for_router(ra) and router_imbalance(rb) <= tau_for_router(rb):
                    tx_hat[a] = rx_hat[b] = 0.0
                    tx_hat[b] = rx_hat[a] = 0.0
                    base_conf_tx[a] = max(base_conf_tx[a], 0.95)
                    base_conf_rx[a] = max(base_conf_rx[a], 0.95)
                    base_conf_tx[b] = max(base_conf_tx[b], 0.95)
                    base_conf_rx[b] = max(base_conf_rx[b], 0.95)

    soft_zero_pass()

    # Track cumulative scaling from original for confidence penalties
    cum_scale_tx: Dict[str, float] = {i: 1.0 if orig_tx[i] > EPS else (0.0 if tx_hat[i] == 0.0 else 1.0) for i in telemetry}
    cum_scale_rx: Dict[str, float] = {i: 1.0 if orig_rx[i] > EPS else (0.0 if rx_hat[i] == 0.0 else 1.0) for i in telemetry}
    strong_scaled_tx: Dict[str, bool] = {i: False for i in telemetry}
    strong_scaled_rx: Dict[str, bool] = {i: False for i in telemetry}

    # Alternating projections
    for it in range(MAX_ITERS):
        # 1) Link symmetry projection (weighted consensus)
        processed = set()
        for a, data_a in telemetry.items():
            b = peers.get(a)
            if not b:
                # isolated: nothing to project
                continue
            key = tuple(sorted([a, b]))
            if key in processed:
                continue
            processed.add(key)
            sa = status.get(a, "unknown")
            sb = status.get(b, "unknown")

            # If both down -> keep zeros (enforced earlier), skip
            if sa == "down" and sb == "down":
                tx_hat[a] = rx_hat[b] = 0.0
                tx_hat[b] = rx_hat[a] = 0.0
                continue

            # Direction a->b: my_tx[a] vs their_rx[b]
            va = tx_hat.get(a, 0.0)
            vb = rx_hat.get(b, 0.0)
            ca = clamp01(base_conf_tx.get(a, 0.6)) ** LINK_W_BIAS
            cb = clamp01(base_conf_rx.get(b, 0.6)) ** LINK_W_BIAS
            if max(va, vb) < ZERO_THRESH:
                fused_ab = 0.0
            else:
                # Status-aware: if one side is down favor the up side by inflating its weight
                if sa == "down" and sb == "up":
                    cb *= 1.4
                if sb == "down" and sa == "up":
                    ca *= 1.4
                wsum = max(EPS, ca + cb)
                fused_ab = (ca * va + cb * vb) / wsum
            # Clip per-iteration change
            def clipped(old: float, new: float) -> float:
                if old <= 0.0:
                    return max(0.0, new)
                lo = old * (1.0 - PER_ITER_CLIP)
                hi = old * (1.0 + PER_ITER_CLIP)
                return min(hi, max(lo, new))

            tx_hat[a] = clipped(va, fused_ab)
            rx_hat[b] = clipped(vb, fused_ab)

            # Direction b->a: my_tx[b] vs their_rx[a]
            vc = tx_hat.get(b, 0.0)
            vd = rx_hat.get(a, 0.0)
            cc = clamp01(base_conf_tx.get(b, 0.6)) ** LINK_W_BIAS
            cd = clamp01(base_conf_rx.get(a, 0.6)) ** LINK_W_BIAS
            if max(vc, vd) < ZERO_THRESH:
                fused_ba = 0.0
            else:
                if sb == "down" and sa == "up":
                    cd *= 1.4
                if sa == "down" and sb == "up":
                    cc *= 1.4
                wsum2 = max(EPS, cc + cd)
                fused_ba = (cc * vc + cd * vd) / wsum2
            tx_hat[b] = clipped(vc, fused_ba)
            rx_hat[a] = clipped(vd, fused_ba)

        # 2) Bundle-aware smoothing (soft, preserves bundle totals approximately)
        for key, ifs in bundles.items():
            if len(ifs) <= 1:
                continue
            # Smooth TX within bundle
            mean_tx = sum(tx_hat.get(i, 0.0) for i in ifs) / max(1, len(ifs))
            mean_rx = sum(rx_hat.get(i, 0.0) for i in ifs) / max(1, len(ifs))
            for i in ifs:
                old_tx = tx_hat.get(i, 0.0)
                old_rx = rx_hat.get(i, 0.0)
                target_tx = (1.0 - BUNDLE_SMOOTH_FRACTION) * old_tx + BUNDLE_SMOOTH_FRACTION * mean_tx
                target_rx = (1.0 - BUNDLE_SMOOTH_FRACTION) * old_rx + BUNDLE_SMOOTH_FRACTION * mean_rx
                # Per-iteration clip
                lo_tx, hi_tx = old_tx * (1.0 - PER_ITER_CLIP), old_tx * (1.0 + PER_ITER_CLIP)
                lo_rx, hi_rx = old_rx * (1.0 - PER_ITER_CLIP), old_rx * (1.0 + PER_ITER_CLIP)
                tx_hat[i] = min(hi_tx, max(lo_tx, target_tx))
                rx_hat[i] = min(hi_rx, max(lo_rx, target_rx))

        # 3) Router conservation projection (weighted LS across both sides)
        for r, ifs in router_ifaces.items():
            if not ifs or len(ifs) == 1:
                continue
            T = sum(tx_hat.get(i, 0.0) for i in ifs)
            R = sum(rx_hat.get(i, 0.0) for i in ifs)
            imbalance = T - R  # want zero
            if abs(imbalance) < max(T, R, 1.0) * tau_router(
                max(
                    sum(1 for i in ifs if tx_hat.get(i, 0.0) >= ZERO_THRESH),
                    sum(1 for i in ifs if rx_hat.get(i, 0.0) >= ZERO_THRESH)
                )
            ):
                continue
            # Adjustability weights (bigger => more change)
            A_t: Dict[str, float] = {}
            A_r: Dict[str, float] = {}
            sum_At = 0.0
            sum_Ar = 0.0
            for i in ifs:
                vt = tx_hat.get(i, 0.0)
                vr = rx_hat.get(i, 0.0)
                ct = clamp01(base_conf_tx.get(i, 0.6))
                cr = clamp01(base_conf_rx.get(i, 0.6))
                at = (1.0 - ct) * max(vt, ZERO_THRESH / 4.0)
                ar = (1.0 - cr) * max(vr, ZERO_THRESH / 4.0)
                A_t[i] = at
                A_r[i] = ar
                sum_At += at
                sum_Ar += ar
            denom = sum_At + sum_Ar
            if denom < EPS:
                continue
            # Distribute both ways to minimize squared change with damping
            k = -imbalance / denom  # because we apply Δt = k*At and Δr = -k*Ar
            k *= ROUTER_DAMP
            # Apply with per-iter clip and non-negativity
            for i in ifs:
                vt = tx_hat.get(i, 0.0)
                vr = rx_hat.get(i, 0.0)
                dt = k * A_t[i]
                dr = -k * A_r[i]
                new_tx = vt + dt
                new_rx = vr + dr
                # Per-iteration relative clip
                def clip_rel(old: float, new: float) -> float:
                    if old <= 0.0:
                        return max(0.0, new)
                    lo = old * (1.0 - PER_ITER_CLIP)
                    hi = old * (1.0 + PER_ITER_CLIP)
                    return min(hi, max(lo, new))
                new_tx = clip_rel(vt, new_tx)
                new_rx = clip_rel(vr, new_rx)
                # Non-negativity
                new_tx = max(0.0, new_tx)
                new_rx = max(0.0, new_rx)
                # Track strong scaling flags
                if orig_tx[i] > EPS:
                    scale_tx = new_tx / orig_tx[i]
                    cum_scale_tx[i] = scale_tx
                    if abs(scale_tx - 1.0) > 0.08:
                        strong_scaled_tx[i] = True
                if orig_rx[i] > EPS:
                    scale_rx = new_rx / orig_rx[i]
                    cum_scale_rx[i] = scale_rx
                    if abs(scale_rx - 1.0) > 0.08:
                        strong_scaled_rx[i] = True
                tx_hat[i] = new_tx
                rx_hat[i] = new_rx

        # Late-iteration soft-zero for tiny links
        if it in (1, MAX_ITERS - 1):
            soft_zero_pass()

    # Final deterministic link micro-sync: one more weighted projection
    processed = set()
    for a in telemetry:
        b = peers.get(a)
        if not b:
            continue
        key = tuple(sorted([a, b]))
        if key in processed:
            continue
        processed.add(key)
        # a->b
        va = tx_hat.get(a, 0.0)
        vb = rx_hat.get(b, 0.0)
        ca = clamp01(base_conf_tx.get(a, 0.6))
        cb = clamp01(base_conf_rx.get(b, 0.6))
        mis_ab = rel_diff(va, vb)
        if mis_ab > TAU_H and max(va, vb) >= ZERO_THRESH:
            fused = (ca * va + cb * vb) / max(EPS, (ca + cb))
            tx_hat[a] = fused
            rx_hat[b] = fused
        # b->a
        vc = tx_hat.get(b, 0.0)
        vd = rx_hat.get(a, 0.0)
        cc = clamp01(base_conf_tx.get(b, 0.6))
        cd = clamp01(base_conf_rx.get(a, 0.6))
        mis_ba = rel_diff(vc, vd)
        if mis_ba > TAU_H and max(vc, vd) >= ZERO_THRESH:
            fused2 = (cc * vc + cd * vd) / max(EPS, (cc + cd))
            tx_hat[b] = fused2
            rx_hat[a] = fused2

    # Status repair (symmetry-aware, conservative)
    repaired_status: Dict[str, str] = {}
    status_conf: Dict[str, float] = {}
    visited = set()
    for i in telemetry:
        if i in visited:
            continue
        p = peers.get(i)
        si = status.get(i, "unknown")
        if not p:
            repaired_status[i] = si
            status_conf[i] = 0.95
            visited.add(i)
            continue
        sp = status.get(p, "unknown")
        rep_i = si
        rep_p = sp
        ci = 0.95
        cp = 0.95
        # If both down, keep down
        if si == "down" and sp == "down":
            rep_i = "down"
            rep_p = "down"
            ci, cp = 0.98, 0.98
        elif si != sp:
            # If any traffic present on the link after repair -> both up
            link_has_traffic = (tx_hat.get(i, 0.0) >= ZERO_THRESH or rx_hat.get(i, 0.0) >= ZERO_THRESH or
                                tx_hat.get(p, 0.0) >= ZERO_THRESH or rx_hat.get(p, 0.0) >= ZERO_THRESH)
            if link_has_traffic:
                rep_i = "up"
                rep_p = "up"
                ci, cp = 0.7, 0.7
            else:
                # Inconclusive, keep as-is but reduce confidence
                ci, cp = 0.6, 0.6
        else:
            rep_i = si
            rep_p = sp
            ci, cp = 0.95, 0.95
        repaired_status[i] = rep_i
        repaired_status[p] = rep_p
        status_conf[i] = ci
        status_conf[p] = cp
        visited.add(i)
        visited.add(p)

    # Final router imbalances for confidence
    router_imbalance_after: Dict[str, float] = {}
    for r, ifs in router_ifaces.items():
        if not ifs:
            router_imbalance_after[r] = 0.0
            continue
        stx = sum(tx_hat.get(i, 0.0) for i in ifs)
        srx = sum(rx_hat.get(i, 0.0) for i in ifs)
        router_imbalance_after[r] = rel_diff(stx, srx)

    # Compose output with calibrated confidences
    result: Dict[str, Dict[str, Tuple]] = {}
    for i, data in telemetry.items():
        rep_tx = tx_hat.get(i, orig_tx[i])
        rep_rx = rx_hat.get(i, orig_rx[i])

        # Status enforcement: down implies zero counters
        rep_status = repaired_status.get(i, status.get(i, "unknown"))
        if rep_status == "down":
            rep_tx = 0.0
            rep_rx = 0.0

        # Changes from original
        ch_tx = rel_diff(orig_tx[i], rep_tx)
        ch_rx = rel_diff(orig_rx[i], rep_rx)

        # Symmetry residuals
        p = peers.get(i)
        if p:
            mis_tx = rel_diff(rep_tx, rx_hat.get(p, 0.0))
            mis_rx = rel_diff(rep_rx, tx_hat.get(p, 0.0))
            fin_sym_tx = clamp01(1.0 - mis_tx)
            fin_sym_rx = clamp01(1.0 - mis_rx)
        else:
            fin_sym_tx = 0.6
            fin_sym_rx = 0.6

        # Router context: average of my router and peer's router (when available)
        r_local = router_of.get(i)
        r_peer = router_of.get(p) if p else None
        r_pen_local = router_imbalance_after.get(r_local, 0.0)
        r_pen_peer = router_imbalance_after.get(r_peer, 0.0) if r_peer is not None else r_pen_local
        router_factor = clamp01(1.0 - min(0.5, 0.5 * (r_pen_local + r_pen_peer)))

        # Base confidences adjusted
        base_tx = clamp01(base_conf_tx.get(i, 0.6))
        base_rx = clamp01(base_conf_rx.get(i, 0.6))

        # Cumulative scaling penalty terms
        scale_tx_term = clamp01(1.0 - min(0.6, abs(cum_scale_tx.get(i, 1.0) - 1.0)))
        scale_rx_term = clamp01(1.0 - min(0.6, abs(cum_scale_rx.get(i, 1.0) - 1.0)))

        conf_tx = clamp01(
            0.30 * fin_sym_tx +
            0.25 * (1.0 - ch_tx) +
            0.20 * base_tx +
            0.15 * router_factor +
            0.10 * scale_tx_term
        )
        conf_rx = clamp01(
            0.30 * fin_sym_rx +
            0.25 * (1.0 - ch_rx) +
            0.20 * base_rx +
            0.15 * router_factor +
            0.10 * scale_rx_term
        )

        # Strong-scale guard and untouched boost
        if strong_scaled_tx.get(i, False):
            conf_tx *= 0.97
        if strong_scaled_rx.get(i, False):
            conf_rx *= 0.97
        if ch_tx < 0.01 and fin_sym_tx >= (1.0 - TAU_H):
            conf_tx = min(0.98, conf_tx + 0.02)
        if ch_rx < 0.01 and fin_sym_rx >= (1.0 - TAU_H):
            conf_rx = min(0.98, conf_rx + 0.02)

        # If status is down, adjust confidences appropriately
        sc = status_conf.get(i, 0.9)
        if rep_status == "down":
            if orig_tx[i] >= ZERO_THRESH or orig_rx[i] >= ZERO_THRESH:
                conf_tx = min(conf_tx, 0.7)
                conf_rx = min(conf_rx, 0.7)
            else:
                conf_tx = max(conf_tx, 0.9)
                conf_rx = max(conf_rx, 0.9)

        # Assemble record
        out = {}
        out["rx_rate"] = (orig_rx[i], rep_rx, clamp01(conf_rx))
        out["tx_rate"] = (orig_tx[i], rep_tx, clamp01(conf_tx))
        out["interface_status"] = (status[i], rep_status, sc)

        # Copy metadata unchanged
        out["connected_to"] = data.get("connected_to")
        out["local_router"] = data.get("local_router")
        out["remote_router"] = data.get("remote_router")

        result[i] = out

    # Final peer confidence smoothing (10%) when both ends are up and present
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