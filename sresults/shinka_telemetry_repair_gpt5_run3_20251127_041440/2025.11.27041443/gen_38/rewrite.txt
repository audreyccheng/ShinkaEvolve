# EVOLVE-BLOCK-START
"""
Consensus projection solver: global weighted projections onto link-equality and router-conservation
subspaces with non-negativity and status pinning. Provides calibrated confidences.

Key steps:
1) Reliability estimation from redundant link signals and statuses.
2) Alternating projections:
   - Link equality: weighted orthogonal projection setting my_tx ≡ peer_rx and vice versa.
   - Router conservation: weighted hyperplane projection so Σ(tx) = Σ(rx) per router.
3) Non-negativity and "down" pinning enforced each iteration; early stopping on invariant satisfaction.
4) Conservative status repair and confidence calibration from residuals and adjustment magnitudes.
"""
from typing import Dict, Any, Tuple, List
import math


def repair_network_telemetry(telemetry: Dict[str, Dict[str, Any]],
                             topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:
    # Base thresholds
    ZERO_THRESH = 0.1  # Mbps considered near-zero
    EPS = 1e-12

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

    def tau_router(n_active: int) -> float:
        # Adaptive router tolerance based on active interfaces
        base = 0.05 * math.sqrt(2.0 / max(2, n_active))
        return clamp(base, 0.03, 0.07)

    def tau_hardening(v1: float, v2: float, c1: float = 0.8, c2: float = 0.8) -> float:
        # Adaptive symmetry tolerance:
        high = (v1 > 100.0 and v2 > 100.0 and c1 >= 0.8 and c2 >= 0.8)
        low = (v1 < 1.0 or v2 < 1.0 or c1 < 0.7 or c2 < 0.7)
        if high: return 0.015
        if low: return 0.03
        return 0.02

    # Build peer mapping
    peers: Dict[str, str] = {}
    for if_id, data in telemetry.items():
        peer = data.get("connected_to")
        if isinstance(peer, str) and peer in telemetry:
            peers[if_id] = peer

    # Build router->interfaces from topology, with fallback using local_router
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

    # Originals and statuses
    orig_tx: Dict[str, float] = {}
    orig_rx: Dict[str, float] = {}
    status_raw: Dict[str, str] = {}
    for if_id, data in telemetry.items():
        orig_tx[if_id] = safe_rate(data.get("tx_rate", 0.0))
        orig_rx[if_id] = safe_rate(data.get("rx_rate", 0.0))
        s = data.get("interface_status", "unknown")
        status_raw[if_id] = s if s in ("up", "down") else "unknown"

    # Pre-fusion mismatches from redundant signals
    pre_mismatch_tx: Dict[str, float] = {}
    pre_mismatch_rx: Dict[str, float] = {}

    processed_pairs = set()
    for a, data in telemetry.items():
        b = data.get("connected_to")
        if not isinstance(b, str) or b not in telemetry:
            continue
        key = tuple(sorted([a, b]))
        if key in processed_pairs:
            continue
        processed_pairs.add(key)
        # a->b and b->a directional mismatches
        mis_ab = rel_diff(orig_tx.get(a, 0.0), orig_rx.get(b, 0.0))
        mis_ba = rel_diff(orig_tx.get(b, 0.0), orig_rx.get(a, 0.0))
        pre_mismatch_tx[a] = mis_ab
        pre_mismatch_rx[b] = mis_ab
        pre_mismatch_tx[b] = mis_ba
        pre_mismatch_rx[a] = mis_ba

    # Fill defaults for isolated interfaces
    for if_id in telemetry.keys():
        if if_id not in pre_mismatch_tx:
            pre_mismatch_tx[if_id] = 0.4
        if if_id not in pre_mismatch_rx:
            pre_mismatch_rx[if_id] = 0.4

    # Baseline reliability r in [0,1] per direction from redundancy and status
    base_r_tx: Dict[str, float] = {}
    base_r_rx: Dict[str, float] = {}
    for if_id in telemetry.keys():
        r_tx = 0.65 + 0.35 * clamp01(1.0 - pre_mismatch_tx.get(if_id, 0.4))
        r_rx = 0.65 + 0.35 * clamp01(1.0 - pre_mismatch_rx.get(if_id, 0.4))
        # Penalize unknown status slightly; down is handled by pinning later
        if status_raw.get(if_id, "unknown") == "unknown":
            r_tx *= 0.9
            r_rx *= 0.9
        # Low-rate jitter tolerance
        if orig_tx.get(if_id, 0.0) < 1.0:
            r_tx *= 0.95
        if orig_rx.get(if_id, 0.0) < 1.0:
            r_rx *= 0.95
        base_r_tx[if_id] = clamp01(r_tx)
        base_r_rx[if_id] = clamp01(r_rx)

    # Quadratic weights q (bigger q resists change more)
    q_tx: Dict[str, float] = {i: 0.3 + 1.0 * base_r_tx[i] for i in telemetry.keys()}
    q_rx: Dict[str, float] = {i: 0.3 + 1.0 * base_r_rx[i] for i in telemetry.keys()}

    # Initialize variables with observations; pin "down" interfaces to zero
    x_tx: Dict[str, float] = {i: orig_tx[i] for i in telemetry.keys()}
    x_rx: Dict[str, float] = {i: orig_rx[i] for i in telemetry.keys()}
    pinned_down: Dict[str, bool] = {i: (status_raw.get(i, "unknown") == "down") for i in telemetry.keys()}
    for i, pin in pinned_down.items():
        if pin:
            x_tx[i] = 0.0
            x_rx[i] = 0.0

    # Build unique link pairs
    link_pairs: List[Tuple[str, str]] = []
    seen = set()
    for a, data in telemetry.items():
        b = data.get("connected_to")
        if isinstance(b, str) and b in telemetry:
            key = tuple(sorted([a, b]))
            if key not in seen:
                seen.add(key)
                link_pairs.append((key[0], key[1]))

    # Alternating projections
    MAX_ITERS = 10
    for it in range(MAX_ITERS):
        # 1) Link-equality projections (weighted)
        for a, b in link_pairs:
            # a->b: x_tx[a] == x_rx[b]
            if not (pinned_down[a] and pinned_down[b]):
                if pinned_down[a]:
                    x_rx[b] = max(0.0, x_tx[a])  # pinned to 0
                elif pinned_down[b]:
                    x_tx[a] = max(0.0, x_rx[b])  # pinned to 0
                else:
                    qa = q_tx.get(a, 1.0)
                    qb = q_rx.get(b, 1.0)
                    denom = qa + qb
                    if denom > EPS:
                        v = (qa * x_tx[a] + qb * x_rx[b]) / denom
                        x_tx[a] = max(0.0, v)
                        x_rx[b] = max(0.0, v)

            # b->a: x_tx[b] == x_rx[a]
            if not (pinned_down[a] and pinned_down[b]):
                if pinned_down[b]:
                    x_rx[a] = max(0.0, x_tx[b])
                elif pinned_down[a]:
                    x_tx[b] = max(0.0, x_rx[a])
                else:
                    qb = q_tx.get(b, 1.0)
                    qa = q_rx.get(a, 1.0)
                    denom = qa + qb
                    if denom > EPS:
                        v = (qb * x_tx[b] + qa * x_rx[a]) / denom
                        x_tx[b] = max(0.0, v)
                        x_rx[a] = max(0.0, v)

        # 2) Router-conservation projections (weighted hyperplane)
        for r, if_list in router_ifaces.items():
            if not if_list:
                continue
            # Compute imbalance s = sum_tx - sum_rx
            sum_tx_r = sum(x_tx.get(i, 0.0) for i in if_list)
            sum_rx_r = sum(x_rx.get(i, 0.0) for i in if_list)
            s = sum_tx_r - sum_rx_r
            if abs(s) < 1e-9:
                continue

            # Build eligible variables (exclude pinned zeros)
            inv_q_sum = 0.0
            elig_tx: List[str] = []
            elig_rx: List[str] = []
            for i in if_list:
                if not pinned_down[i]:
                    if q_tx.get(i, 1.0) > EPS:
                        inv_q_sum += 1.0 / q_tx[i]
                        elig_tx.append(i)
                    if q_rx.get(i, 1.0) > EPS:
                        inv_q_sum += 1.0 / q_rx[i]
                        elig_rx.append(i)
            if inv_q_sum <= EPS:
                continue

            # Solve λ and deltas (exact weighted projection onto sum_tx - sum_rx = -s)
            lam = 2.0 * s / inv_q_sum
            # Apply with mild damping to reduce overshoot sensitivity
            gamma = 0.9
            for i in elig_tx:
                delta = -lam / (2.0 * q_tx[i])
                x_tx[i] = max(0.0, x_tx[i] + gamma * delta)
            for i in elig_rx:
                delta = +lam / (2.0 * q_rx[i])
                x_rx[i] = max(0.0, x_rx[i] + gamma * delta)

        # 3) Early stopping check: router imbalances within adaptive tolerance and link symmetry good
        all_ok = True
        for r, if_list in router_ifaces.items():
            if not if_list:
                continue
            n_active_tx = sum(1 for i in if_list if x_tx.get(i, 0.0) >= ZERO_THRESH)
            n_active_rx = sum(1 for i in if_list if x_rx.get(i, 0.0) >= ZERO_THRESH)
            tau_r = tau_router(max(n_active_tx, n_active_rx))
            sum_tx_r = sum(x_tx.get(i, 0.0) for i in if_list)
            sum_rx_r = sum(x_rx.get(i, 0.0) for i in if_list)
            if rel_diff(sum_tx_r, sum_rx_r) > tau_r:
                all_ok = False
                break
        if all_ok:
            # Check links
            for a, b in link_pairs:
                mis1 = rel_diff(x_tx.get(a, 0.0), x_rx.get(b, 0.0))
                mis2 = rel_diff(x_tx.get(b, 0.0), x_rx.get(a, 0.0))
                th1 = tau_hardening(x_tx.get(a, 0.0), x_rx.get(b, 0.0), base_r_tx.get(a, 0.8), base_r_rx.get(b, 0.8))
                th2 = tau_hardening(x_tx.get(b, 0.0), x_rx.get(a, 0.0), base_r_tx.get(b, 0.8), base_r_rx.get(a, 0.8))
                if mis1 > th1 or mis2 > th2:
                    all_ok = False
                    break
        if all_ok:
            break

    # Soft-zero stabilization on tiny links
    seen_pairs = set()
    for a, data in telemetry.items():
        b = data.get("connected_to")
        if not isinstance(b, str) or b not in telemetry:
            continue
        key = tuple(sorted([a, b]))
        if key in seen_pairs:
            continue
        seen_pairs.add(key)
        if max(x_tx.get(a, 0.0), x_rx.get(b, 0.0), x_tx.get(b, 0.0), x_rx.get(a, 0.0)) < 2.0 * ZERO_THRESH:
            x_tx[a] = 0.0
            x_rx[b] = 0.0
            x_tx[b] = 0.0
            x_rx[a] = 0.0

    # Status repair (symmetry-aware, conservative)
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
        any_traffic = (x_tx.get(a, 0.0) >= ZERO_THRESH or x_rx.get(a, 0.0) >= ZERO_THRESH or
                       x_tx.get(b, 0.0) >= ZERO_THRESH or x_rx.get(b, 0.0) >= ZERO_THRESH)
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
                # ambiguous; keep as-is but lower confidence
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

    # Enforce "down" => zero
    for i, st in repaired_status.items():
        if st == "down":
            x_tx[i] = 0.0
            x_rx[i] = 0.0

    # Post-invariant metrics for confidence
    # Router imbalance AFTER projection
    router_imbalance_after: Dict[str, float] = {}
    for r, if_list in router_ifaces.items():
        sum_tx_r = sum(x_tx.get(i, 0.0) for i in if_list)
        sum_rx_r = sum(x_rx.get(i, 0.0) for i in if_list)
        router_imbalance_after[r] = rel_diff(sum_tx_r, sum_rx_r)

    # Final per-direction symmetry residuals
    post_mismatch_tx_dir: Dict[str, float] = {}
    post_mismatch_rx_dir: Dict[str, float] = {}
    for i, data in telemetry.items():
        p = data.get("connected_to")
        if isinstance(p, str) and p in telemetry:
            post_mismatch_tx_dir[i] = rel_diff(x_tx.get(i, 0.0), x_rx.get(p, 0.0))
            post_mismatch_rx_dir[i] = rel_diff(x_rx.get(i, 0.0), x_tx.get(p, 0.0))
        else:
            post_mismatch_tx_dir[i] = 0.4
            post_mismatch_rx_dir[i] = 0.4

    # Compose results with confidence calibration
    result: Dict[str, Dict[str, Tuple]] = {}
    for if_id, data in telemetry.items():
        rep_tx = x_tx.get(if_id, orig_tx[if_id])
        rep_rx = x_rx.get(if_id, orig_rx[if_id])

        # Change magnitude relative to original (stable near-zero)
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

        # Baseline reliability
        base_tx_conf = clamp01(base_r_tx.get(if_id, 0.7))
        base_rx_conf = clamp01(base_r_rx.get(if_id, 0.7))

        red_tx = clamp01(1.0 - pre_tx)
        red_rx = clamp01(1.0 - pre_rx)

        ch_tx = clamp01(1.0 - change_tx)
        ch_rx = clamp01(1.0 - change_rx)

        # Blend confidence components
        conf_tx_final = clamp01(
            0.24 * base_tx_conf +
            0.18 * red_tx +
            0.28 * fin_sym_tx +
            0.12 * ch_tx +
            0.10 * router_factor_after +
            0.08 * clamp01(1.0 - min(0.5, post_mismatch_tx_dir.get(if_id, 0.4)))  # residual touch
        )
        conf_rx_final = clamp01(
            0.24 * base_rx_conf +
            0.18 * red_rx +
            0.28 * fin_sym_rx +
            0.12 * ch_rx +
            0.10 * router_factor_after +
            0.08 * clamp01(1.0 - min(0.5, post_mismatch_rx_dir.get(if_id, 0.4)))
        )

        # Status enforcement: down implies zero counters with calibrated confidence
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

        # Assemble output record
        out = {}
        out["rx_rate"] = (orig_rx[if_id], rep_rx, conf_rx_final)
        out["tx_rate"] = (orig_tx[if_id], rep_tx, conf_tx_final)
        out["interface_status"] = (status_raw[if_id], rep_status, conf_status)

        # Copy metadata unchanged
        out["connected_to"] = data.get("connected_to")
        out["local_router"] = data.get("local_router")
        out["remote_router"] = data.get("remote_router")

        result[if_id] = out

    # Final confidence peer smoothing (70/30 residual blend then 10% peer smoothing) when both ends are up
    for i, data in telemetry.items():
        p = data.get("connected_to")
        if not isinstance(p, str) or p not in telemetry:
            continue
        if i not in result or p not in result:
            continue
        if result[i]["interface_status"][1] != "up" or result[p]["interface_status"][1] != "up":
            continue
        tx_i = safe_rate(result[i]["tx_rate"][1])
        rx_p = safe_rate(result[p]["rx_rate"][1])
        rx_i = safe_rate(result[i]["rx_rate"][1])
        tx_p = safe_rate(result[p]["tx_rate"][1])
        mis_tx = rel_diff(tx_i, rx_p)
        mis_rx = rel_diff(rx_i, tx_p)
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