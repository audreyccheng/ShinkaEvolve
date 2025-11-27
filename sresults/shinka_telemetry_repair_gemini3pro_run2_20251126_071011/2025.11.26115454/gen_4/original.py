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
    
    # Second pass: repair using redundant signals
    for interface_id, data in telemetry.items():
        repaired_data = {}
        
        interface_status = data.get('interface_status', 'unknown')
        rx_rate = data.get('rx_rate', 0.0)
        tx_rate = data.get('tx_rate', 0.0)
        connected_to = data.get('connected_to')
        
        # Default: no repair, high confidence
        repaired_rx = rx_rate
        repaired_tx = tx_rate
        repaired_status = interface_status
        rx_confidence = 1.0
        tx_confidence = 1.0
        status_confidence = 1.0
        
        # Check for issues and attempt repair
        if interface_id in link_symmetry_violations:
            violations = link_symmetry_violations[interface_id]
            
            # Repair RX rate if link symmetry is violated
            if violations['rx_tx_diff'] > HARDENING_THRESHOLD:
                # Use peer's TX as more reliable signal
                repaired_rx = violations['peer_tx']
                # Confidence decreases with magnitude of violation
                rx_confidence = max(0.0, 1.0 - violations['rx_tx_diff'])
            
            # Repair TX rate if link symmetry is violated
            if violations['tx_rx_diff'] > HARDENING_THRESHOLD:
                # Use peer's RX as more reliable signal
                repaired_tx = violations['peer_rx']
                # Confidence decreases with magnitude of violation
                tx_confidence = max(0.0, 1.0 - violations['tx_rx_diff'])
        
        # Check status consistency
        if connected_to and connected_to in telemetry:
            peer_status = telemetry[connected_to].get('interface_status', 'unknown')
            # If statuses don't match, lower confidence
            if interface_status != peer_status:
                status_confidence = 0.5
                # If interface is down but has non-zero rates, that's suspicious
                if interface_status == 'down' and (rx_rate > 0 or tx_rate > 0):
                    repaired_rx = 0.0
                    repaired_tx = 0.0
                    rx_confidence = 0.3
                    tx_confidence = 0.3
        
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

