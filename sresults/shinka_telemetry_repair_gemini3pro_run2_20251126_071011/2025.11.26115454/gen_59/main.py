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

    result = {}

    # First pass: collect all measurements and check link symmetry
    link_symmetry_violations = {}

    for interface_id, data in telemetry.items():
        interface_status = data.get('interface_status', 'unknown')
        rx_rate = data.get('rx_rate', 0.0)
        tx_rate = data.get('tx_rate', 0.0)
        connected_to = data.get('connected_to')

        # Check link symmetry if connected interface exists
        if connected_to and connected_to in telemetry:
            peer_data = telemetry[connected_to]
            peer_rx = peer_data.get('rx_rate', 0.0)
            peer_tx = peer_data.get('tx_rate', 0.0)

            # My TX should match their RX (within tolerance)
            tx_rx_diff = abs(tx_rate - peer_rx) / max(tx_rate, peer_rx, 1.0)
            # My RX should match their TX (within tolerance)
            rx_tx_diff = abs(rx_rate - peer_tx) / max(rx_rate, peer_tx, 1.0)

            link_symmetry_violations[interface_id] = {
                'tx_rx_diff': tx_rx_diff,
                'rx_tx_diff': rx_tx_diff,
                'peer_rx': peer_rx,
                'peer_tx': peer_tx
            }

    # Helper to calculate flow imbalance impact
    def check_flow_consistency(router_id, if_id, direction, value):
        if not router_id or router_id not in topology:
            return None

        sum_in = 0.0
        sum_out = 0.0

        for neighbor_if in topology[router_id]:
            if neighbor_if not in telemetry: continue

            # Determine neighbor values
            n_data = telemetry[neighbor_if]
            # Use fixed status logic for flow calc: if marked down, treat as 0
            # But we don't have repaired status yet, so use raw status + traffic check heuristic
            n_status = n_data.get('interface_status', 'unknown')
            n_rx = n_data.get('rx_rate', 0.0)
            n_tx = n_data.get('tx_rate', 0.0)

            # If status is down but traffic is high, treat as UP for calculation
            if n_status == 'down' and (n_rx <= 1.0 and n_tx <= 1.0):
                n_rx, n_tx = 0.0, 0.0

            if neighbor_if == if_id:
                # Use our candidate value
                if direction == 'rx':
                    sum_in += value
                    sum_out += n_tx
                else:
                    sum_in += n_rx
                    sum_out += value
            else:
                sum_in += n_rx
                sum_out += n_tx

        imbalance = abs(sum_in - sum_out)
        total_flow = max(sum_in, sum_out, 1.0)
        return imbalance / total_flow

    # Second pass: repair using redundant signals
    for interface_id, data in telemetry.items():
        repaired_data = {}

        rx_rate = data.get('rx_rate', 0.0)
        tx_rate = data.get('tx_rate', 0.0)
        interface_status = data.get('interface_status', 'unknown')

        repaired_rx = rx_rate
        repaired_tx = tx_rate
        repaired_status = interface_status

        rx_confidence = 1.0
        tx_confidence = 1.0
        status_confidence = 1.0

        connected_to = data.get('connected_to')
        peer_data = telemetry.get(connected_to) if connected_to else None

        # 1. Status Repair
        if peer_data:
            peer_status = peer_data.get('interface_status', 'unknown')
            if interface_status != peer_status:
                # Disagreement. Check traffic.
                local_active = rx_rate > 1.0 or tx_rate > 1.0
                peer_active = peer_data.get('rx_rate', 0.0) > 1.0 or peer_data.get('tx_rate', 0.0) > 1.0

                if local_active or peer_active:
                    repaired_status = 'up'
                    status_confidence = 0.9
                else:
                    repaired_status = 'down'
                    status_confidence = 0.9

        # Enforce Down
        if repaired_status == 'down':
            repaired_rx = 0.0
            repaired_tx = 0.0
            # If we force zero, confidence is high
            if rx_rate > 1.0 or tx_rate > 1.0:
                rx_confidence = 0.95
                tx_confidence = 0.95

        # 2. Rate Repair (Only if UP)
        elif interface_id in link_symmetry_violations:
            violations = link_symmetry_violations[interface_id]
            local_router = data.get('local_router')

            # --- RX Repair ---
            if violations['rx_tx_diff'] > HARDENING_THRESHOLD:
                cand_peer = violations['peer_tx']

                # Check flow consistency
                score_local = check_flow_consistency(local_router, interface_id, 'rx', rx_rate)
                score_peer = check_flow_consistency(local_router, interface_id, 'rx', cand_peer)

                # Heuristic: Prefer Peer unless Local is MUCH better for flow
                use_peer = True
                if score_local is not None and score_peer is not None:
                    # If local fits flow perfectly (<1%) and peer creates significant imbalance (>5%)
                    if score_local < 0.01 and score_peer > 0.05:
                        use_peer = False

                if use_peer:
                    repaired_rx = cand_peer
                    # High confidence if verified by flow
                    if score_peer is not None and score_peer < 0.02:
                        rx_confidence = 0.95
                    else:
                        rx_confidence = max(0.0, 1.0 - violations['rx_tx_diff'])
                else:
                    repaired_rx = rx_rate
                    rx_confidence = 0.9

            # --- TX Repair ---
            if violations['tx_rx_diff'] > HARDENING_THRESHOLD:
                cand_peer = violations['peer_rx']

                score_local = check_flow_consistency(local_router, interface_id, 'tx', tx_rate)
                score_peer = check_flow_consistency(local_router, interface_id, 'tx', cand_peer)

                use_peer = True
                if score_local is not None and score_peer is not None:
                     if score_local < 0.01 and score_peer > 0.05:
                        use_peer = False

                if use_peer:
                    repaired_tx = cand_peer
                    if score_peer is not None and score_peer < 0.02:
                        tx_confidence = 0.95
                    else:
                        tx_confidence = max(0.0, 1.0 - violations['tx_rx_diff'])
                else:
                    repaired_tx = tx_rate
                    tx_confidence = 0.9

        # Store repaired values with confidence scores
        repaired_data['rx_rate'] = (rx_rate, repaired_rx, rx_confidence)
        repaired_data['tx_rate'] = (tx_rate, repaired_tx, tx_confidence)
        repaired_data['interface_status'] = (interface_status, repaired_status, status_confidence)

        # Copy metadata unchanged
        repaired_data['connected_to'] = connected_to
        repaired_data['local_router'] = data.get('local_router')
        repaired_data['remote_router'] = data.get('remote_router')

        result[interface_id] = repaired_data

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
