# EVOLVE-BLOCK-START
"""
Triangulated Consensus Repair
Uses a three-estimator voting model (Self, Peer, Flow) to validate and repair
network telemetry with high confidence calibration.
"""
from typing import Dict, Any, Tuple, List
import math

def repair_network_telemetry(telemetry: Dict[str, Dict[str, Any]], 
                             topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:
    """
    Repairs telemetry using a Triangulated Consensus model.
    
    For each interface, we generate three estimates for TX/RX:
    1. Self (Local Sensor)
    2. Peer (Remote Sensor via Link Symmetry)
    3. Flow (Contextual via Flow Conservation)
    
    We arbitrate these estimates to find the most likely true value and assign
    confidence based on the level of agreement (Consensus).
    """
    
    # --- Configuration ---
    TOLERANCE = 0.02          # 2% deviation allowed for "Agreement"
    MIN_RATE_THRESHOLD = 0.1  # Mbps, traffic below this is considered noise/zero
    
    results = {}
    
    # --- Helper Functions ---
    def get_rate(data: Dict, key: str) -> float:
        val = data.get(key, 0.0)
        return float(val) if val is not None else 0.0

    def agrees(v1: float, v2: float) -> bool:
        """Check if two values agree within tolerance."""
        # If both are effectively zero, they agree
        if v1 < MIN_RATE_THRESHOLD and v2 < MIN_RATE_THRESHOLD:
            return True
        # If one is zero and other is large, they disagree
        if v1 < MIN_RATE_THRESHOLD or v2 < MIN_RATE_THRESHOLD:
            return False
            
        diff = abs(v1 - v2)
        denom = max(abs(v1), abs(v2), 1.0)
        return (diff / denom) <= TOLERANCE

    # --- Phase 1: Status Alignment & Working State Initialization ---
    # Detect Zombie (Down but Active) and Ghost (Up but Dead) interfaces.
    
    working_state = {}
    
    for if_id, data in telemetry.items():
        orig_status = data.get('interface_status', 'unknown')
        rx = get_rate(data, 'rx_rate')
        tx = get_rate(data, 'tx_rate')
        
        peer_id = data.get('connected_to')
        peer_data = telemetry.get(peer_id, {}) if peer_id else {}
        peer_status = peer_data.get('interface_status', 'unknown')
        peer_rx = get_rate(peer_data, 'rx_rate')
        peer_tx = get_rate(peer_data, 'tx_rate')
        
        repaired_status = orig_status
        status_conf = 1.0
        
        # Check for signal presence (Self or Peer seeing traffic)
        has_traffic = (rx > MIN_RATE_THRESHOLD or tx > MIN_RATE_THRESHOLD or 
                       peer_rx > MIN_RATE_THRESHOLD or peer_tx > MIN_RATE_THRESHOLD)
        
        # Logic: If status says DOWN but we see significant traffic, it's UP.
        if orig_status == 'down' and has_traffic:
            repaired_status = 'up'
            status_conf = 0.9 # High confidence correction
            
        # Logic: If status says UP, but Peer is DOWN and NO traffic exists, likely DOWN.
        # This aligns status inconsistency when links are idle.
        elif orig_status == 'up' and peer_status == 'down' and not has_traffic:
            repaired_status = 'down'
            status_conf = 0.8
            
        working_state[if_id] = {
            'status': repaired_status,
            'orig_status': orig_status,
            'status_conf': status_conf,
            'rx': rx,
            'tx': tx,
            'peer_id': peer_id
        }

    # --- Phase 2: Flow Estimator Preparation ---
    # To use Flow Conservation as a voter, we need an initial "Best Guess" for every interface.
    # Peer measurements are generally trusted over Self if they disagree (Symmetry Principle).
    
    router_estimates = {} # router_id -> {if_id: {'rx': val, 'tx': val}}
    
    # Initialize estimates using Link Symmetry (Peer > Self)
    for router_id, interfaces in topology.items():
        router_estimates[router_id] = {}
        for if_id in interfaces:
            if if_id not in working_state: continue
            
            curr = working_state[if_id]
            est_rx = curr['rx']
            est_tx = curr['tx']
            
            # If we have a peer, check symmetry.
            if curr['peer_id'] and curr['peer_id'] in telemetry:
                peer_data = telemetry[curr['peer_id']]
                peer_tx = get_rate(peer_data, 'tx_rate') # Peer TX predicts My RX
                peer_rx = get_rate(peer_data, 'rx_rate') # Peer RX predicts My TX
                
                # If local disagrees with peer, prefer peer for the *initial* flow calculation
                # This makes the Flow Vote robust against single local sensor failures
                if not agrees(est_rx, peer_tx):
                    est_rx = peer_tx
                if not agrees(est_tx, peer_rx):
                    est_tx = peer_rx
            
            # If effective status is down, rates must be zero
            if curr['status'] == 'down':
                est_rx = 0.0
                est_tx = 0.0
                
            router_estimates[router_id][if_id] = {'rx': est_rx, 'tx': est_tx}

    # --- Phase 3: Consensus Voting & Repair ---
    # Iterate via topology to allow Flow Conservation calculations
    
    processed_interfaces = set()
    
    for router_id, interfaces in topology.items():
        # Calculate Router Totals (Constraint: Sum RX approx Sum TX)
        # We use the "Best Guesses" from Phase 2
        total_rx = sum(router_estimates[router_id][i]['rx'] for i in interfaces if i in router_estimates[router_id])
        total_tx = sum(router_estimates[router_id][i]['tx'] for i in interfaces if i in router_estimates[router_id])
        
        for if_id in interfaces:
            if if_id not in working_state: continue
            processed_interfaces.add(if_id)
            
            curr = working_state[if_id]
            
            # 1. Self Estimator
            v_self_rx = curr['rx']
            v_self_tx = curr['tx']
            
            # 2. Peer Estimator
            peer_id = curr['peer_id']
            if peer_id and peer_id in telemetry:
                v_peer_tx = get_rate(telemetry[peer_id], 'tx_rate') # Peer TX -> My RX
                v_peer_rx = get_rate(telemetry[peer_id], 'rx_rate') # Peer RX -> My TX
            else:
                # No peer: self is the only direct signal
                v_peer_tx = v_self_rx
                v_peer_rx = v_self_tx
                
            # 3. Flow Estimator
            # Logic: My RX = Total TX - (Total RX - My Estimated RX)
            # This asks: "What must My RX be to balance the router, given everyone else?"
            if if_id in router_estimates[router_id]:
                my_est = router_estimates[router_id][if_id]
                est_others_rx = total_rx - my_est['rx']
                est_others_tx = total_tx - my_est['tx']
                
                # Flow Target for RX: Balance output traffic + imbalance of others
                v_flow_rx = max(0.0, total_tx - est_others_rx)
                # Flow Target for TX: Balance input traffic - imbalance of others
                v_flow_tx = max(0.0, total_rx - est_others_tx)
            else:
                # Fallback if estimation failed
                v_flow_rx = v_self_rx
                v_flow_tx = v_self_tx

            # --- ARBITRATION LOGIC ---
            
            # Repair RX
            if curr['status'] == 'down':
                final_rx = 0.0
                # Confidence: High unless we saw phantom traffic on self sensor
                conf_rx = 1.0 if v_self_rx < MIN_RATE_THRESHOLD else 0.8
            else:
                # Check Consensus
                s_p = agrees(v_self_rx, v_peer_tx) # Self-Peer
                p_f = agrees(v_peer_tx, v_flow_rx) # Peer-Flow
                s_f = agrees(v_self_rx, v_flow_rx) # Self-Flow
                
                if s_p and p_f:
                    # Unanimous agreement (or close enough)
                    final_rx = (v_self_rx + v_peer_tx) / 2.0
                    conf_rx = 1.0
                elif s_p:
                    # Self and Peer agree, Flow disagrees. 
                    # Trust Link Symmetry, but lower confidence due to Flow anomaly (maybe another interface is bad)
                    final_rx = (v_self_rx + v_peer_tx) / 2.0
                    conf_rx = 0.90
                elif p_f:
                    # Peer and Flow agree, Self disagrees. 
                    # Strong evidence of local sensor failure.
                    final_rx = v_peer_tx
                    conf_rx = 0.95
                elif s_f:
                    # Self and Flow agree, Peer disagrees.
                    # Evidence of remote sensor failure.
                    final_rx = v_self_rx
                    conf_rx = 0.85 # Flow is derivative, so slightly less trust than direct peer
                else:
                    # Total disagreement. Fallback to Peer (usually best proxy).
                    final_rx = v_peer_tx
                    conf_rx = 0.5
            
            # Repair TX (Mirror logic)
            if curr['status'] == 'down':
                final_tx = 0.0
                conf_tx = 1.0 if v_self_tx < MIN_RATE_THRESHOLD else 0.8
            else:
                s_p = agrees(v_self_tx, v_peer_rx)
                p_f = agrees(v_peer_rx, v_flow_tx)
                s_f = agrees(v_self_tx, v_flow_tx)
                
                if s_p and p_f:
                    final_tx = (v_self_tx + v_peer_rx) / 2.0
                    conf_tx = 1.0
                elif s_p:
                    final_tx = (v_self_tx + v_peer_rx) / 2.0
                    conf_tx = 0.90
                elif p_f:
                    final_tx = v_peer_rx
                    conf_tx = 0.95
                elif s_f:
                    final_tx = v_self_tx
                    conf_tx = 0.85
                else:
                    final_tx = v_peer_rx
                    conf_tx = 0.5

            # Store Results
            results[if_id] = {
                'rx_rate': (curr['rx'], final_rx, conf_rx),
                'tx_rate': (curr['tx'], final_tx, conf_tx),
                'interface_status': (curr['orig_status'], curr['status'], curr['status_conf']),
                'connected_to': telemetry[if_id].get('connected_to'),
                'local_router': telemetry[if_id].get('local_router'),
                'remote_router': telemetry[if_id].get('remote_router')
            }

    # --- Phase 4: Cleanup Orphans ---
    # Handle interfaces that weren't in the topology map (rare but possible)
    for if_id, data in telemetry.items():
        if if_id not in results:
            # Basic repair without Flow Context
            curr = working_state[if_id]
            peer_id = curr['peer_id']
            
            # RX Repair
            if curr['status'] == 'down':
                final_rx = 0.0
                conf_rx = 1.0
            elif peer_id and peer_id in telemetry:
                peer_tx = get_rate(telemetry[peer_id], 'tx_rate')
                if agrees(curr['rx'], peer_tx):
                    final_rx = (curr['rx'] + peer_tx) / 2.0
                    conf_rx = 0.9
                else:
                    final_rx = peer_tx # Trust peer
                    conf_rx = 0.6
            else:
                final_rx = curr['rx']
                conf_rx = 0.5
                
            # TX Repair
            if curr['status'] == 'down':
                final_tx = 0.0
                conf_tx = 1.0
            elif peer_id and peer_id in telemetry:
                peer_rx = get_rate(telemetry[peer_id], 'rx_rate')
                if agrees(curr['tx'], peer_rx):
                    final_tx = (curr['tx'] + peer_rx) / 2.0
                    conf_tx = 0.9
                else:
                    final_tx = peer_rx
                    conf_tx = 0.6
            else:
                final_tx = curr['tx']
                conf_tx = 0.5
            
            results[if_id] = {
                'rx_rate': (curr['rx'], final_rx, conf_rx),
                'tx_rate': (curr['tx'], final_tx, conf_tx),
                'interface_status': (curr['orig_status'], curr['status'], curr['status_conf']),
                'connected_to': data.get('connected_to'),
                'local_router': data.get('local_router'),
                'remote_router': data.get('remote_router')
            }

    return results
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

