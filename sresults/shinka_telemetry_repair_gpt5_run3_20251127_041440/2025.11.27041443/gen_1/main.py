# EVOLVE-BLOCK-START
"""
Robust link hardening + router projection repair algorithm.

This implementation follows the three-step Hodor approach:
1) Signal Collection: use redundant signals from both sides of a link.
2) Signal Hardening: robustly fuse per-link-direction rates (my_tx ~ their_rx).
3) Dynamic Checking / Projection: enforce router-level flow conservation by
   scaling the less-trustworthy side to match the other (projection).

Outputs repaired telemetry with calibrated confidence scores.
"""
from typing import Dict, Any, Tuple, List
import math


def repair_network_telemetry(telemetry: Dict[str, Dict[str, Any]], 
                             topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:
    """
    Repair network telemetry using robust per-link fusion and router-level flow projection.

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
    # Timing tolerance from Hodor: ~2% for symmetry checks
    TAU_H = 0.02
    EPS = 1e-9
    ZERO_THRESH = 0.1  # Mbps considered operationally near-zero for decision heuristics

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

    # Build peer mapping and sanitize inputs
    peers: Dict[str, str] = {}
    for if_id, data in telemetry.items():
        peer = data.get("connected_to")
        if isinstance(peer, str) and peer in telemetry:
            peers[if_id] = peer

    # Build router->interfaces using topology with fallback to local_router fields
    router_ifaces: Dict[str, List[str]] = {}
    # Start with provided topology
    for r, if_list in topology.items():
        router_ifaces.setdefault(r, [])
        for i in if_list:
            if i in telemetry:
                router_ifaces[r].append(i)
    # Ensure all interfaces are assigned using local_router if missing in topology
    for if_id, data in telemetry.items():
        r = data.get("local_router")
        if r is None:
            # If missing, create a pseudo-router bucket
            r = f"unknown_router::{if_id}"
        router_ifaces.setdefault(r, [])
        if if_id not in router_ifaces[r]:
            router_ifaces[r].append(if_id)

    # Prepare storage
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

    # Stage 1: Link hardening (fuse redundant signals per direction)
    hard_tx: Dict[str, float] = {}
    hard_rx: Dict[str, float] = {}
    conf_tx_link: Dict[str, float] = {}  # confidence contribution from link fusion
    conf_rx_link: Dict[str, float] = {}
    # Pre-compute pre-fusion mismatches for calibration
    pre_mismatch_tx: Dict[str, float] = {}  # my_tx vs peer_rx
    pre_mismatch_rx: Dict[str, float] = {}  # my_rx vs peer_tx

    visited = set()
    for if_id, data in telemetry.items():
        if if_id in visited:
            continue
        peer = peers.get(if_id)
        if not peer:
            # No peer known; carry forward raw, conservative confidence
            hard_tx[if_id] = orig_tx[if_id]
            hard_rx[if_id] = orig_rx[if_id]
            conf_tx_link[if_id] = 0.6
            conf_rx_link[if_id] = 0.6
            pre_mismatch_tx[if_id] = 0.4  # unknown -> moderate mismatch
            pre_mismatch_rx[if_id] = 0.4
            visited.add(if_id)
            continue

        visited.add(if_id)
        visited.add(peer)

        # Original candidates for each direction
        a, b = if_id, peer
        a_tx, a_rx = orig_tx[a], orig_rx[a]
        b_tx, b_rx = orig_tx[b], orig_rx[b]
        sa, sb = status[a], status[b]

        # If both ends report down, force zeros (clear-cut)
        both_down = (sa == "down" and sb == "down")
        if both_down:
            hard_ab = 0.0
            hard_ba = 0.0
            # Set both directions to zero
            hard_tx[a] = hard_ab
            hard_rx[b] = hard_ab
            hard_tx[b] = hard_ba
            hard_rx[a] = hard_ba
            # High confidence due to strong agreement (both down)
            conf_tx_link[a] = 0.95
            conf_rx_link[b] = 0.95
            conf_tx_link[b] = 0.95
            conf_rx_link[a] = 0.95
            pre_mismatch_tx[a] = rel_diff(a_tx, b_rx)
            pre_mismatch_rx[a] = rel_diff(a_rx, b_tx)
            pre_mismatch_tx[b] = rel_diff(b_tx, a_rx)
            pre_mismatch_rx[b] = rel_diff(b_rx, a_tx)
            continue

        # Direction a->b: fuse a_tx with b_rx
        diff_ab = rel_diff(a_tx, b_rx)
        pre_mismatch_tx[a] = diff_ab
        pre_mismatch_rx[b] = diff_ab

        # Direction b->a: fuse b_tx with a_rx
        diff_ba = rel_diff(b_tx, a_rx)
        pre_mismatch_tx[b] = diff_ba
        pre_mismatch_rx[a] = diff_ba

        def fuse(v1: float, v2: float, s1: str, s2: str, mismatch: float) -> Tuple[float, float]:
            # Base weights prefer consistent, non-zero values
            w1 = 1.0
            w2 = 1.0
            # If one is near-zero and the other non-zero, favor the non-zero
            if v1 < ZERO_THRESH and v2 >= ZERO_THRESH:
                w1 *= 0.5
                w2 *= 1.5
            elif v2 < ZERO_THRESH and v1 >= ZERO_THRESH:
                w2 *= 0.5
                w1 *= 1.5
            # If one side reports down, slightly discount that side unless both are zero
            if s1 == "down" and (v1 >= ZERO_THRESH or v2 >= ZERO_THRESH):
                w1 *= 0.7
            if s2 == "down" and (v1 >= ZERO_THRESH or v2 >= ZERO_THRESH):
                w2 *= 0.7
            # If mismatch is small, equal weighting; if large, weights still apply
            # Fuse
            denom = w1 + w2
            fused_val = (w1 * v1 + w2 * v2) / denom if denom > 0 else 0.0
            # Confidence from link symmetry: higher if mismatch small
            c = clamp01(1.0 - mismatch)
            return fused_val, c

        # Fuse both directions
        fused_ab, c_ab = fuse(a_tx, b_rx, sa, sb, diff_ab)
        fused_ba, c_ba = fuse(b_tx, a_rx, sb, sa, diff_ba)

        # Assign hardened values symmetrically
        hard_tx[a] = fused_ab
        hard_rx[b] = fused_ab
        hard_tx[b] = fused_ba
        hard_rx[a] = fused_ba

        conf_tx_link[a] = c_ab
        conf_rx_link[b] = c_ab
        conf_tx_link[b] = c_ba
        conf_rx_link[a] = c_ba

    # For any remaining interfaces (isolated entries)
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

    # Stage 2: Router-level flow conservation projection
    # Compute per-router totals and mismatch; scale the lower-confidence side
    router_imbalance_before: Dict[str, float] = {}
    scaled_tx_factor: Dict[str, float] = {if_id: 1.0 for if_id in telemetry}
    scaled_rx_factor: Dict[str, float] = {if_id: 1.0 for if_id in telemetry}

    # Build reverse index router->interfaces (already have), ensure each listed iface exists
    for router, if_list in router_ifaces.items():
        # Totals using hardened values
        sum_tx = sum(hard_tx.get(i, 0.0) for i in if_list)
        sum_rx = sum(hard_rx.get(i, 0.0) for i in if_list)
        mismatch = rel_diff(sum_tx, sum_rx)
        router_imbalance_before[router] = mismatch

        if max(sum_tx, sum_rx) < EPS:
            continue  # nothing to project

        if mismatch > TAU_H:
            # Decide which side to adjust based on aggregate link confidence
            c_tx_total = sum(conf_tx_link.get(i, 0.5) for i in if_list)
            c_rx_total = sum(conf_rx_link.get(i, 0.5) for i in if_list)
            adjust_side = "tx" if c_tx_total < c_rx_total else "rx"
            if adjust_side == "tx" and sum_tx > 0:
                alpha = sum_rx / max(sum_tx, EPS)
                for i in if_list:
                    hard_tx[i] *= alpha
                    scaled_tx_factor[i] *= alpha
                    # Confidence penalty due to scaling magnitude
                    penalty = clamp01(abs(alpha - 1.0))
                    conf_tx_link[i] *= clamp01(1.0 - 0.5 * penalty)
            elif adjust_side == "rx" and sum_rx > 0:
                alpha = sum_tx / max(sum_rx, EPS)
                for i in if_list:
                    hard_rx[i] *= alpha
                    scaled_rx_factor[i] *= alpha
                    penalty = clamp01(abs(alpha - 1.0))
                    conf_rx_link[i] *= clamp01(1.0 - 0.5 * penalty)
        # else within tolerance, no scaling

    # Status repair (conservative):
    repaired_status: Dict[str, str] = {}
    status_conf: Dict[str, float] = {}
    processed = set()
    for if_id, data in telemetry.items():
        if if_id in processed:
            continue
        peer = peers.get(if_id)
        s_local = status[if_id]
        if not peer:
            # No peer: keep status as is with high confidence
            repaired_status[if_id] = s_local
            status_conf[if_id] = 0.95
            processed.add(if_id)
            continue
        s_peer = status[peer]
        # Default: keep local
        rep_local = s_local
        rep_peer = s_peer
        c_local = 0.95
        c_peer = 0.95

        # If both report down, keep down
        if s_local == "down" and s_peer == "down":
            rep_local = "down"
            rep_peer = "down"
            c_local = 0.98
            c_peer = 0.98
        elif s_local != s_peer:
            # If traffic clearly exists on the link, status must be up
            link_has_traffic = (hard_tx[if_id] >= ZERO_THRESH or hard_rx[if_id] >= ZERO_THRESH or
                                hard_tx[peer] >= ZERO_THRESH or hard_rx[peer] >= ZERO_THRESH)
            if link_has_traffic:
                rep_local = "up"
                rep_peer = "up"
                c_local = 0.7
                c_peer = 0.7
            else:
                # Ambiguous: keep originals but lower confidence
                rep_local = s_local
                rep_peer = s_peer
                c_local = 0.6
                c_peer = 0.6
        else:
            # Both equal, keep and set high confidence
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

    # Build final results with calibrated confidences
    result: Dict[str, Dict[str, Tuple]] = {}
    for if_id, data in telemetry.items():
        # Final repaired values
        rep_tx = hard_tx.get(if_id, orig_tx[if_id])
        rep_rx = hard_rx.get(if_id, orig_rx[if_id])

        # Compute change magnitudes
        change_tx = rel_diff(orig_tx[if_id], rep_tx)
        change_rx = rel_diff(orig_rx[if_id], rep_rx)

        # Pre-fusion mismatch contextual confidence
        pre_tx = pre_mismatch_tx.get(if_id, 0.4)
        pre_rx = pre_mismatch_rx.get(if_id, 0.4)

        # Router environment penalty from pre-projection imbalance
        r = router_of.get(if_id, None)
        router_penalty = router_imbalance_before.get(r, 0.0) if r is not None else 0.0
        router_factor = clamp01(1.0 - min(0.5, router_penalty))  # cap penalty at 0.5

        # Combine link confidence with change penalty and router factor
        base_tx_conf = conf_tx_link.get(if_id, 0.6)
        base_rx_conf = conf_rx_link.get(if_id, 0.6)

        # Redundancy support (higher if pre-mismatch small)
        red_tx = clamp01(1.0 - pre_tx)
        red_rx = clamp01(1.0 - pre_rx)

        # Penalize big changes
        ch_tx = clamp01(1.0 - change_tx)
        ch_rx = clamp01(1.0 - change_rx)

        # Final confidence: blend redundancy, link base, change, and router context
        conf_tx_final = clamp01((0.35 * base_tx_conf + 0.35 * red_tx + 0.20 * ch_tx + 0.10 * router_factor))
        conf_rx_final = clamp01((0.35 * base_rx_conf + 0.35 * red_rx + 0.20 * ch_rx + 0.10 * router_factor))

        # Status confidence from earlier step
        rep_status = repaired_status.get(if_id, status.get(if_id, "unknown"))
        conf_status = status_conf.get(if_id, 0.9)

        # Ensure "down" implies zero counters in repaired output for consistency
        if rep_status == "down":
            rep_tx = 0.0
            rep_rx = 0.0
            # High confidence that down implies zero, but if original was non-zero, modestly reduce
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
        out["interface_status"] = (status[if_id], rep_status, conf_status)

        # Copy metadata unchanged
        out["connected_to"] = data.get("connected_to")
        out["local_router"] = data.get("local_router")
        out["remote_router"] = data.get("remote_router")

        result[if_id] = out

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

