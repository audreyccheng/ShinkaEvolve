# EVOLVE-BLOCK-START
"""
Network telemetry repair algorithm that detects and corrects inconsistencies
in network interface telemetry data using topology relationships.

Takes interface telemetry data and detects/repairs inconsistencies based on
network invariants like link symmetry and flow conservation.
"""
from typing import Dict, Any, Tuple, List


def repair_network_telemetry(telemetry: Dict[str, Dict[str, Any]],
                             topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:
    """
    Repair network interface telemetry by detecting and correcting inconsistencies.

    Core principle: Use network invariants to validate and repair telemetry:
    1. Link Symmetry (R3): my_tx_rate ≈ their_rx_rate for connected interfaces
    2. Flow Conservation (R1): Sum(incoming traffic) = Sum(outgoing traffic) at each router
    3. Interface Consistency: Status should be consistent across connected pairs

    Args:
        telemetry: Dictionary where key is interface_id and value contains:
            - interface_status: "up" or "down"
            - rx_rate: receive rate in Mbps
            - tx_rate: transmit rate in Mbps
            - connected_to: interface_id this interface connects to
            - local_router: router_id this interface belongs to
            - remote_router: router_id on the other side
        topology: Dictionary where key is router_id and value contains a list of interface_ids

    Returns:
        Dictionary with same structure but telemetry values become tuples of:
        (original_value, repaired_value, confidence_score)
        where confidence ranges from 0.0 (very uncertain) to 1.0 (very confident)
    """

    # Measurement timing tolerance (from Hodor research: ~2%)
    HARDENING_THRESHOLD = 0.02
    # Router-level imbalance tolerance (slightly looser)
    ROUTER_TOL = 0.05
    # Near-zero threshold to stabilize tiny rates
    ZERO_THRESH = 0.1
    EPS = 1e-9

    def safe_rate(x: Any) -> float:
        try:
            v = float(x)
            if v < 0:
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

    result = {}

    # First pass: collect all measurements and check link symmetry
    link_symmetry_violations = {}

    for interface_id, data in telemetry.items():
        interface_status = data.get('interface_status', 'unknown')
        rx_rate = safe_rate(data.get('rx_rate', 0.0))
        tx_rate = safe_rate(data.get('tx_rate', 0.0))
        connected_to = data.get('connected_to')

        # Check link symmetry if connected interface exists
        if connected_to and connected_to in telemetry:
            peer_data = telemetry[connected_to]
            peer_rx = safe_rate(peer_data.get('rx_rate', 0.0))
            peer_tx = safe_rate(peer_data.get('tx_rate', 0.0))

            # My TX should match their RX (within tolerance)
            tx_rx_diff = rel_diff(tx_rate, peer_rx)
            # My RX should match their TX (within tolerance)
            rx_tx_diff = rel_diff(rx_rate, peer_tx)

            link_symmetry_violations[interface_id] = {
                'tx_rx_diff': tx_rx_diff,
                'rx_tx_diff': rx_tx_diff,
                'peer_rx': peer_rx,
                'peer_tx': peer_tx,
                'peer_status': peer_data.get('interface_status', 'unknown'),
            }

    # Second pass: repair using redundant signals with small-mismatch and near-zero handling
    for interface_id, data in telemetry.items():
        repaired_data = {}

        interface_status = data.get('interface_status', 'unknown')
        rx_rate = safe_rate(data.get('rx_rate', 0.0))
        tx_rate = safe_rate(data.get('tx_rate', 0.0))
        connected_to = data.get('connected_to')

        # Default: no repair, conservative high confidence
        repaired_rx = rx_rate
        repaired_tx = tx_rate
        repaired_status = interface_status
        rx_confidence = 0.95
        tx_confidence = 0.95
        status_confidence = 0.95

        # Check for issues and attempt repair
        if interface_id in link_symmetry_violations:
            violations = link_symmetry_violations[interface_id]
            peer_tx = violations['peer_tx']
            peer_rx = violations['peer_rx']
            peer_status = violations['peer_status']
            tx_rx_diff = violations['tx_rx_diff']
            rx_tx_diff = violations['rx_tx_diff']

            # If both ends down, snap to zero with high confidence
            if interface_status == 'down' and peer_status == 'down':
                repaired_rx, repaired_tx = 0.0, 0.0
                rx_confidence, tx_confidence = 0.98, 0.98
            else:
                # Near-zero stabilization: if both sides are ~0 in a direction, set to 0
                if max(tx_rate, peer_rx) < ZERO_THRESH:
                    repaired_tx = 0.0
                    tx_confidence = 0.95
                if max(rx_rate, peer_tx) < ZERO_THRESH:
                    repaired_rx = 0.0
                    rx_confidence = 0.95

                # RX repair from peer TX
                if rx_tx_diff <= HARDENING_THRESHOLD:
                    # within tolerance: keep local, reinforce confidence
                    rx_confidence = max(rx_confidence, 0.95)
                elif rx_tx_diff <= 0.10:
                    # moderate mismatch: average
                    fused = 0.5 * rx_rate + 0.5 * peer_tx
                    repaired_rx = fused
                    rx_confidence = max(0.6, clamp01(1.0 - rx_tx_diff))
                else:
                    # large mismatch: prefer plausible side
                    if rx_rate < ZERO_THRESH and peer_tx >= ZERO_THRESH:
                        repaired_rx = peer_tx
                    elif peer_tx < ZERO_THRESH and rx_rate >= ZERO_THRESH:
                        repaired_rx = rx_rate
                    else:
                        # bias to peer if peer is up and local is down
                        if interface_status == 'down' and peer_status == 'up':
                            repaired_rx = peer_tx
                        elif peer_status == 'down' and interface_status == 'up':
                            repaired_rx = rx_rate
                        else:
                            # default to weighted toward peer
                            repaired_rx = 0.3 * rx_rate + 0.7 * peer_tx
                    rx_confidence = max(0.4, clamp01(1.0 - rx_tx_diff))

                # TX repair from peer RX
                if tx_rx_diff <= HARDENING_THRESHOLD:
                    tx_confidence = max(tx_confidence, 0.95)
                elif tx_rx_diff <= 0.10:
                    fused = 0.5 * tx_rate + 0.5 * peer_rx
                    repaired_tx = fused
                    tx_confidence = max(0.6, clamp01(1.0 - tx_rx_diff))
                else:
                    if tx_rate < ZERO_THRESH and peer_rx >= ZERO_THRESH:
                        repaired_tx = peer_rx
                    elif peer_rx < ZERO_THRESH and tx_rate >= ZERO_THRESH:
                        repaired_tx = tx_rate
                    else:
                        if interface_status == 'down' and peer_status == 'up':
                            repaired_tx = peer_rx
                        elif peer_status == 'down' and interface_status == 'up':
                            repaired_tx = tx_rate
                        else:
                            repaired_tx = 0.3 * tx_rate + 0.7 * peer_rx
                    tx_confidence = max(0.4, clamp01(1.0 - tx_rx_diff))

        # Check status consistency
        if connected_to and connected_to in telemetry:
            peer_status = telemetry[connected_to].get('interface_status', 'unknown')
            # If statuses don't match, lower confidence
            if interface_status != peer_status:
                status_confidence = 0.6
                # If interface is down but has non-zero rates, that's suspicious -> enforce zeros
                if interface_status == 'down' and (rx_rate > ZERO_THRESH or tx_rate > ZERO_THRESH):
                    repaired_rx = 0.0
                    repaired_tx = 0.0
                    rx_confidence = min(rx_confidence, 0.5)
                    tx_confidence = min(tx_confidence, 0.5)

        # Store repaired values with confidence scores
        repaired_data['rx_rate'] = (rx_rate, repaired_rx, clamp01(rx_confidence))
        repaired_data['tx_rate'] = (tx_rate, repaired_tx, clamp01(tx_confidence))
        repaired_data['interface_status'] = (interface_status, repaired_status, clamp01(status_confidence))

        # Copy metadata unchanged
        repaired_data['connected_to'] = connected_to
        repaired_data['local_router'] = data.get('local_router')
        repaired_data['remote_router'] = data.get('remote_router')

        result[interface_id] = repaired_data

    # Third pass: Router-level flow conservation projection (uses topology)
    # Build router->interfaces from topology with fallback to local_router fields.
    router_ifaces: Dict[str, List[str]] = {}
    for r, if_list in topology.items():
        router_ifaces.setdefault(r, [])
        for i in if_list:
            if i in telemetry:
                router_ifaces[r].append(i)
    for if_id, data in telemetry.items():
        r = data.get('local_router')
        if r is None:
            r = f"unknown_router::{if_id}"
        router_ifaces.setdefault(r, [])
        if if_id not in router_ifaces[r]:
            router_ifaces[r].append(if_id)

    # For each router, scale the less-trustworthy aggregate (tx or rx) toward the other
    for router, if_list in router_ifaces.items():
        if not if_list:
            continue
        sum_tx = 0.0
        sum_rx = 0.0
        sum_tx_conf = 0.0
        sum_rx_conf = 0.0
        for i in if_list:
            if i not in result:
                continue
            sum_tx += safe_rate(result[i]['tx_rate'][1])
            sum_rx += safe_rate(result[i]['rx_rate'][1])
            sum_tx_conf += clamp01(result[i]['tx_rate'][2])
            sum_rx_conf += clamp01(result[i]['rx_rate'][2])
        if max(sum_tx, sum_rx) < EPS:
            continue
        mismatch = rel_diff(sum_tx, sum_rx)
        if mismatch > ROUTER_TOL:
            # choose the side with lower aggregate confidence to adjust
            adjust_side = 'tx' if sum_tx_conf < sum_rx_conf else 'rx'
            if adjust_side == 'tx' and sum_tx > 0:
                alpha = sum_rx / max(sum_tx, EPS)
                # clip and damp scaling
                alpha = max(0.85, min(1.15, alpha))
                alpha_eff = 1.0 + 0.6 * (alpha - 1.0)
                penalty = abs(alpha_eff - 1.0)
                for i in if_list:
                    if i not in result:
                        continue
                    cur_tx = safe_rate(result[i]['tx_rate'][1])
                    new_tx = cur_tx * alpha_eff
                    result[i]['tx_rate'] = (
                        result[i]['tx_rate'][0],
                        new_tx,
                        clamp01(result[i]['tx_rate'][2] * (1.0 - 0.4 * clamp01(penalty)))
                    )
            elif adjust_side == 'rx' and sum_rx > 0:
                alpha = sum_tx / max(sum_rx, EPS)
                alpha = max(0.85, min(1.15, alpha))
                alpha_eff = 1.0 + 0.6 * (alpha - 1.0)
                penalty = abs(alpha_eff - 1.0)
                for i in if_list:
                    if i not in result:
                        continue
                    cur_rx = safe_rate(result[i]['rx_rate'][1])
                    new_rx = cur_rx * alpha_eff
                    result[i]['rx_rate'] = (
                        result[i]['rx_rate'][0],
                        new_rx,
                        clamp01(result[i]['rx_rate'][2] * (1.0 - 0.4 * clamp01(penalty)))
                    )

    # Stage 2.5: Post-projection gentle link re-sync and confidence calibration
    processed_pairs = set()
    for a, data_a in telemetry.items():
        b = data_a.get('connected_to')
        if not isinstance(b, str) or b not in telemetry:
            continue
        key = tuple(sorted([a, b]))
        if key in processed_pairs:
            continue
        processed_pairs.add(key)

        # a->b direction: my_tx[a] vs their_rx[b]
        if a in result and b in result:
            tx_a = safe_rate(result[a]['tx_rate'][1])
            rx_b = safe_rate(result[b]['rx_rate'][1])
            diff_ab = rel_diff(tx_a, rx_b)
            if diff_ab > HARDENING_THRESHOLD and max(tx_a, rx_b) >= ZERO_THRESH:
                ca = clamp01(result[a]['tx_rate'][2])
                cb = clamp01(result[b]['rx_rate'][2])
                mean_ab = 0.5 * (tx_a + rx_b)
                if ca < cb:
                    # Nudge lower-confidence side toward mean
                    new_tx_a = 0.5 * mean_ab + 0.5 * tx_a
                    result[a]['tx_rate'] = (result[a]['tx_rate'][0], new_tx_a, clamp01(ca * 0.95))
                elif cb < ca:
                    new_rx_b = 0.5 * mean_ab + 0.5 * rx_b
                    result[b]['rx_rate'] = (result[b]['rx_rate'][0], new_rx_b, clamp01(cb * 0.95))
                else:
                    # Both similar and low confidence: cautiously set both to mean
                    if ca < 0.7:
                        result[a]['tx_rate'] = (result[a]['tx_rate'][0], mean_ab, clamp01(ca * 0.93))
                        result[b]['rx_rate'] = (result[b]['rx_rate'][0], mean_ab, clamp01(cb * 0.93))

        # b->a direction: my_tx[b] vs their_rx[a]
        if a in result and b in result:
            tx_b = safe_rate(result[b]['tx_rate'][1])
            rx_a = safe_rate(result[a]['rx_rate'][1])
            diff_ba = rel_diff(tx_b, rx_a)
            if diff_ba > HARDENING_THRESHOLD and max(tx_b, rx_a) >= ZERO_THRESH:
                cb_tx = clamp01(result[b]['tx_rate'][2])
                ca_rx = clamp01(result[a]['rx_rate'][2])
                mean_ba = 0.5 * (tx_b + rx_a)
                if cb_tx < ca_rx:
                    new_tx_b = 0.5 * mean_ba + 0.5 * tx_b
                    result[b]['tx_rate'] = (result[b]['tx_rate'][0], new_tx_b, clamp01(cb_tx * 0.95))
                elif ca_rx < cb_tx:
                    new_rx_a = 0.5 * mean_ba + 0.5 * rx_a
                    result[a]['rx_rate'] = (result[a]['rx_rate'][0], new_rx_a, clamp01(ca_rx * 0.95))
                else:
                    if cb_tx < 0.7:
                        result[b]['tx_rate'] = (result[b]['tx_rate'][0], mean_ba, clamp01(cb_tx * 0.93))
                        result[a]['rx_rate'] = (result[a]['rx_rate'][0], mean_ba, clamp01(ca_rx * 0.93))

    # Final confidence touch-up: incorporate final symmetry residuals
    for i, data in telemetry.items():
        peer = data.get('connected_to')
        if not isinstance(peer, str) or peer not in telemetry or i not in result or peer not in result:
            continue
        mis_tx = rel_diff(safe_rate(result[i]['tx_rate'][1]), safe_rate(result[peer]['rx_rate'][1]))
        mis_rx = rel_diff(safe_rate(result[i]['rx_rate'][1]), safe_rate(result[peer]['tx_rate'][1]))
        result[i]['tx_rate'] = (
            result[i]['tx_rate'][0],
            result[i]['tx_rate'][1],
            clamp01(0.7 * clamp01(result[i]['tx_rate'][2]) + 0.3 * clamp01(1.0 - mis_tx))
        )
        result[i]['rx_rate'] = (
            result[i]['rx_rate'][0],
            result[i]['rx_rate'][1],
            clamp01(0.7 * clamp01(result[i]['rx_rate'][2]) + 0.3 * clamp01(1.0 - mis_rx))
        )

    # Final pass: status enforcement - down implies zero counters with calibrated confidence
    for i, data in telemetry.items():
        status = data.get('interface_status', 'unknown')
        if i not in result:
            continue
        if status == 'down':
            orig_tx = safe_rate(result[i]['tx_rate'][0])
            orig_rx = safe_rate(result[i]['rx_rate'][0])
            result[i]['tx_rate'] = (
                result[i]['tx_rate'][0],
                0.0,
                0.9 if (orig_tx < ZERO_THRESH and orig_rx < ZERO_THRESH) else min(result[i]['tx_rate'][2], 0.7)
            )
            result[i]['rx_rate'] = (
                result[i]['rx_rate'][0],
                0.0,
                0.9 if (orig_tx < ZERO_THRESH and orig_rx < ZERO_THRESH) else min(result[i]['rx_rate'][2], 0.7)
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