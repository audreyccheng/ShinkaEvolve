# EVOLVE-BLOCK-START
"""
Network telemetry repair algorithm using Flow Invariants Consensus.
Prioritizes traffic evidence for status, and uses Flow Conservation hints
to resolve Link Symmetry violations with calibrated confidence.
"""
from typing import Dict, Any, Tuple, List


def repair_network_telemetry(telemetry: Dict[str, Dict[str, Any]], 
                             topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:
    """
    Repair network interface telemetry by checking consistency against
    redundant signals:
    1. Status vs Traffic: Traffic presence overrides 'down' status.
    2. Link Symmetry: Connected interfaces should match rates.
    3. Flow Conservation: Router ingress should match egress.
    
    The algorithm calculates 'Flow Hints' for each interface (what the rate *should* be
    to balance the router). These hints are used as a reliable tie-breaker when
    Link Symmetry is violated.
    """
    
    # Thresholds
    HARDENING_THRESHOLD = 0.02  # 2% tolerance for measurement timing differences
    TRAFFIC_THRESHOLD = 1.0     # 1 Mbps threshold to consider a link active
    
    # --- Phase 1: Calculate Flow Conservation Hints ---
    # For every interface, calculate the expected rate required to satisfy
    # Kirchhoff's current law (Sum In = Sum Out) at the local router.
    
    flow_hints = {} # interface_id -> {'rx': float, 'tx': float}
    
    for router_id, iface_ids in topology.items():
        # Only consider interfaces present in this telemetry snapshot
        valid_ifaces = [i for i in iface_ids if i in telemetry]
        
        # Calculate aggregate router throughput
        sum_rx = sum(telemetry[i].get('rx_rate', 0.0) for i in valid_ifaces)
        sum_tx = sum(telemetry[i].get('tx_rate', 0.0) for i in valid_ifaces)
        
        for iface in valid_ifaces:
            curr_data = telemetry[iface]
            curr_rx = curr_data.get('rx_rate', 0.0)
            curr_tx = curr_data.get('tx_rate', 0.0)
            
            # Calculate what RX/TX *should* be to close the balance
            # Hint_RX: The RX needed to match total TX, given other RX flows
            # hint_rx + (sum_rx - curr_rx) = sum_tx
            rx_hint = max(0.0, sum_tx - (sum_rx - curr_rx))
            
            # Hint_TX: The TX needed to match total RX, given other TX flows
            # hint_tx + (sum_tx - curr_tx) = sum_rx
            tx_hint = max(0.0, sum_rx - (sum_tx - curr_tx))
            
            flow_hints[iface] = {'rx': rx_hint, 'tx': tx_hint}

    result = {}

    # --- Phase 2: Repair Telemetry ---
    for iface_id, data in telemetry.items():
        # Original values
        orig_rx = data.get('rx_rate', 0.0)
        orig_tx = data.get('tx_rate', 0.0)
        orig_status = data.get('interface_status', 'unknown')
        
        # Peer Context
        connected_to = data.get('connected_to')
        peer_data = telemetry.get(connected_to, {}) if (connected_to and connected_to in telemetry) else {}
        has_peer = bool(peer_data)
        
        # --- A. Repair Status ---
        # Heuristic: Trust traffic over status flags.
        
        # Collect all traffic signals on the link
        traffic_signals = [orig_rx, orig_tx, peer_data.get('rx_rate', 0.0), peer_data.get('tx_rate', 0.0)]
        max_traffic = max(traffic_signals) if traffic_signals else 0.0
        
        rep_status = orig_status
        conf_status = 1.0
        
        if max_traffic > TRAFFIC_THRESHOLD:
            # Active traffic implies link is UP
            if orig_status != 'up':
                rep_status = 'up'
                conf_status = 0.95 # Correcting a likely flag error
        elif has_peer and orig_status == 'up':
             # I say UP, Traffic is 0. Check peer.
             if peer_data.get('interface_status') == 'down':
                 # Contradiction with peer and no traffic -> likely DOWN
                 rep_status = 'down'
                 conf_status = 0.8
        
        # --- B. Repair Rates ---
        
        rep_rx, rep_tx = orig_rx, orig_tx
        conf_rx, conf_tx = 1.0, 1.0
        
        if rep_status == 'down':
            # Enforce physics: Down links have 0 rate
            rep_rx = 0.0
            rep_tx = 0.0
            # If we suppressed real traffic, inherit status confidence
            if orig_rx > TRAFFIC_THRESHOLD or orig_tx > TRAFFIC_THRESHOLD:
                conf_rx = conf_status
                conf_tx = conf_status
        elif has_peer:
            # Link is UP: Apply Consensus Logic
            
            # Function to select best value between Local, Peer using Hint
            def resolve_rate(local_val, peer_val, hint_val):
                # 1. Check Symmetry (Local vs Peer)
                denom_sym = max(local_val, peer_val, 1.0)
                diff_sym = abs(local_val - peer_val) / denom_sym
                
                if diff_sym <= HARDENING_THRESHOLD:
                    # Consensus reached: Average the values to reduce jitter
                    return (local_val + peer_val) / 2.0, 1.0
                
                # 2. Symmetry Broken: Use Hint as Arbiter
                denom_l = max(local_val, hint_val, 1.0)
                diff_l = abs(local_val - hint_val) / denom_l
                
                denom_p = max(peer_val, hint_val, 1.0)
                diff_p = abs(peer_val - hint_val) / denom_p
                
                # Favor the value closer to the Flow Hint
                if diff_l < diff_p:
                    # Local is closer to hint. Trust Local.
                    # Confidence depends on how much better it is
                    margin = diff_p - diff_l
                    confidence = 0.9 if margin > 0.1 else 0.7
                    return local_val, confidence
                else:
                    # Peer is closer to hint. Trust Peer.
                    margin = diff_l - diff_p
                    confidence = 0.9 if margin > 0.1 else 0.7
                    return peer_val, confidence

            # Repair RX: Local RX vs Peer TX
            hint_rx = flow_hints.get(iface_id, {}).get('rx', orig_rx)
            rep_rx, conf_rx = resolve_rate(orig_rx, peer_data.get('tx_rate', 0.0), hint_rx)
            
            # Repair TX: Local TX vs Peer RX
            hint_tx = flow_hints.get(iface_id, {}).get('tx', orig_tx)
            rep_tx, conf_tx = resolve_rate(orig_tx, peer_data.get('rx_rate', 0.0), hint_tx)

        # Construct Result
        result[iface_id] = {
            'rx_rate': (orig_rx, rep_rx, conf_rx),
            'tx_rate': (orig_tx, rep_tx, conf_tx),
            'interface_status': (orig_status, rep_status, conf_status),
            'connected_to': connected_to,
            'local_router': data.get('local_router'),
            'remote_router': data.get('remote_router')
        }
        
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

