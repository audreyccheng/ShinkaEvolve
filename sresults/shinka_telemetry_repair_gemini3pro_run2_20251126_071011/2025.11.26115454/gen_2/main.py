# EVOLVE-BLOCK-START
"""
Network telemetry repair algorithm that detects and corrects inconsistencies
in network interface telemetry data using topology relationships.
"""
from typing import Dict, Any, Tuple, List
import math

def repair_network_telemetry(telemetry: Dict[str, Dict[str, Any]], 
                             topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:
    """
    Repairs network telemetry using a Weighted Consensus approach based on 
    Link Symmetry and Flow Conservation invariants.
    """
    
    # Parameters
    HARDENING_THRESHOLD = 0.02  # 2% tolerance
    NOISE_FLOOR = 1.0          # 1 Mbps noise floor
    WEIGHTS = {
        'self': 1.0,
        'peer': 1.5,           # High trust in link symmetry
        'flow': 0.8            # Moderate trust in flow conservation (accumulates noise)
    }
    
    # --- Helper Data Structures ---
    # Working copy of rates and status: {if_id: {'rx': float, 'tx': float, 'status': str}}
    working_state = {}
    
    # Initialize working state
    for if_id, data in telemetry.items():
        working_state[if_id] = {
            'rx': float(data.get('rx_rate', 0.0)),
            'tx': float(data.get('tx_rate', 0.0)),
            'status': data.get('interface_status', 'unknown'),
            'peer': data.get('connected_to'),
            'router': data.get('local_router')
        }

    # --- Step 1: Status Repair ---
    # Logic: Traffic implies UP. Peer traffic implies I am UP.
    for if_id, state in working_state.items():
        original_status = state['status']
        inferred_status = original_status
        
        # Evidence for UP
        has_traffic = state['rx'] > NOISE_FLOOR or state['tx'] > NOISE_FLOOR
        
        peer_active = False
        peer_id = state['peer']
        if peer_id and peer_id in working_state:
            peer_state = working_state[peer_id]
            if peer_state['status'] == 'up' and (peer_state['tx'] > NOISE_FLOOR or peer_state['rx'] > NOISE_FLOOR):
                peer_active = True
        
        if has_traffic or peer_active:
            inferred_status = 'up'
        
        # Apply status repair
        state['status'] = inferred_status
        
        # Enforce consistency: if DOWN, rates must be 0
        if inferred_status == 'down':
            state['rx'] = 0.0
            state['tx'] = 0.0

    # --- Step 2: Iterative Rate Repair ---
    # We perform coordinate descent to satisfy Symmetry and Flow invariants
    
    for iteration in range(2):
        # Calculate Router Aggregates for Flow Conservation
        # router_aggregates: {router_id: {'sum_rx': val, 'sum_tx': val}}
        router_aggregates = {}
        for r_id, if_list in topology.items():
            sum_rx = 0.0
            sum_tx = 0.0
            for if_id in if_list:
                if if_id in working_state:
                    sum_rx += working_state[if_id]['rx']
                    sum_tx += working_state[if_id]['tx']
            router_aggregates[r_id] = {'sum_rx': sum_rx, 'sum_tx': sum_tx}

        # Update each interface
        new_rates = {}
        
        for if_id, state in working_state.items():
            if state['status'] != 'up':
                new_rates[if_id] = {'rx': 0.0, 'tx': 0.0}
                continue
                
            # --- RX Repair ---
            # Signals for RX:
            # 1. Self RX
            # 2. Peer TX
            # 3. Flow suggestion: RX = TX + (Sum_TX_other - Sum_RX_other)
            
            rx_signals = []
            rx_signals.append({'val': state['rx'], 'w': WEIGHTS['self']})
            
            peer_id = state['peer']
            if peer_id and peer_id in working_state:
                rx_signals.append({'val': working_state[peer_id]['tx'], 'w': WEIGHTS['peer']})
            
            r_id = state['router']
            if r_id and r_id in router_aggregates:
                # Current router balance
                aggs = router_aggregates[r_id]
                # To balance: Sum_RX = Sum_TX
                # RX_this + Sum_RX_other = TX_this + Sum_TX_other
                # RX_this = TX_this + (Sum_TX_other - Sum_RX_other)
                # Note: TX_this is also being repaired, but we use current state value
                
                # We calculate 'net_out_others'
                rx_other = aggs['sum_rx'] - state['rx']
                tx_other = aggs['sum_tx'] - state['tx']
                
                flow_suggestion = state['tx'] + (tx_other - rx_other)
                # Only use flow suggestion if positive
                if flow_suggestion >= 0:
                    rx_signals.append({'val': flow_suggestion, 'w': WEIGHTS['flow']})

            # Consensus for RX
            weighted_sum = sum(s['val'] * s['w'] for s in rx_signals)
            total_weight = sum(s['w'] for s in rx_signals)
            repaired_rx = weighted_sum / total_weight if total_weight > 0 else 0.0
            
            # --- TX Repair ---
            # Signals for TX:
            # 1. Self TX
            # 2. Peer RX
            # 3. Flow suggestion: TX = RX - (Sum_TX_other - Sum_RX_other)
            
            tx_signals = []
            tx_signals.append({'val': state['tx'], 'w': WEIGHTS['self']})
            
            if peer_id and peer_id in working_state:
                tx_signals.append({'val': working_state[peer_id]['rx'], 'w': WEIGHTS['peer']})
                
            if r_id and r_id in router_aggregates:
                aggs = router_aggregates[r_id]
                rx_other = aggs['sum_rx'] - state['rx']
                tx_other = aggs['sum_tx'] - state['tx']
                
                # TX_this = RX_this - (Sum_TX_other - Sum_RX_other)
                flow_suggestion = state['rx'] - (tx_other - rx_other)
                if flow_suggestion >= 0:
                    tx_signals.append({'val': flow_suggestion, 'w': WEIGHTS['flow']})

            # Consensus for TX
            weighted_sum = sum(s['val'] * s['w'] for s in tx_signals)
            total_weight = sum(s['w'] for s in tx_signals)
            repaired_tx = weighted_sum / total_weight if total_weight > 0 else 0.0
            
            new_rates[if_id] = {'rx': repaired_rx, 'tx': repaired_tx}
        
        # Apply updates
        for if_id, rates in new_rates.items():
            working_state[if_id]['rx'] = rates['rx']
            working_state[if_id]['tx'] = rates['tx']

    # --- Step 3: Confidence Calculation & Output Generation ---
    result = {}
    
    for if_id, orig_data in telemetry.items():
        final_state = working_state[if_id]
        
        # Helpers for confidence
        def calc_confidence(original, repaired, peer_val, flow_support):
            # Base confidence
            conf = 1.0
            
            # If we changed the value significantly
            if abs(original - repaired) > max(original, repaired, 1.0) * HARDENING_THRESHOLD:
                # We changed it. Do we have support?
                
                # Check consistency with repaired value
                support_score = 0.0
                max_score = 0.0
                
                # Peer support
                if peer_val is not None:
                    max_score += WEIGHTS['peer']
                    if abs(repaired - peer_val) < max(repaired, 1.0) * 0.05:
                        support_score += WEIGHTS['peer']
                
                # Flow support (passed as bool or weight)
                if flow_support:
                    max_score += WEIGHTS['flow']
                    support_score += WEIGHTS['flow']
                
                if max_score > 0:
                    ratio = support_score / max_score
                    # If high support, high confidence. If low support, low confidence.
                    conf = 0.4 + (0.6 * ratio)
                else:
                    conf = 0.5 # Changed without external validation? risky.
            
            else:
                # We kept original (mostly).
                # Reduce confidence if it actually disagrees with invariants but we couldn't fix it?
                # Or if it agrees, high confidence.
                if peer_val is not None and abs(repaired - peer_val) > max(repaired, 1.0) * 0.1:
                    conf = 0.6 # Disagreement remains
            
            return min(1.0, max(0.0, conf))

        # Check Flow Support for final values
        # Re-calc router balance with final values
        r_id = final_state['router']
        flow_ok_rx = True
        flow_ok_tx = True
        if r_id and r_id in topology:
            # We need to calc aggregates again locally or pass them
            # Approximation: check if final state is balanced
            # Ideally we check if this specific value fits the balance
            pass # simplified in calc_confidence logic below
            
        peer_id = final_state['peer']
        peer_tx = working_state[peer_id]['tx'] if (peer_id and peer_id in working_state) else None
        peer_rx = working_state[peer_id]['rx'] if (peer_id and peer_id in working_state) else None

        # Check if router is balanced
        router_balanced = False
        if r_id in topology:
             # Quick check of final balance
             s_rx = sum(working_state[i]['rx'] for i in topology[r_id] if i in working_state)
             s_tx = sum(working_state[i]['tx'] for i in topology[r_id] if i in working_state)
             imbalance = abs(s_rx - s_tx) / max(s_rx, s_tx, 1.0)
             router_balanced = imbalance < 0.05
        
        # Calculate confidences
        rx_conf = calc_confidence(
            float(orig_data.get('rx_rate', 0.0)), 
            final_state['rx'], 
            peer_tx, 
            router_balanced
        )
        
        tx_conf = calc_confidence(
            float(orig_data.get('tx_rate', 0.0)), 
            final_state['tx'], 
            peer_rx, 
            router_balanced
        )
        
        # Status confidence
        status_conf = 1.0
        if orig_data.get('interface_status') != final_state['status']:
            # If we flipped status, how sure are we?
            # If traffic exists, very sure.
            if final_state['rx'] > 1.0 or final_state['tx'] > 1.0:
                status_conf = 0.95
            else:
                status_conf = 0.7
        elif final_state['status'] == 'up' and not (final_state['rx'] > 0 or final_state['tx'] > 0):
             # Reports UP but no traffic?
             status_conf = 0.8
             
        repaired_data = {}
        repaired_data['rx_rate'] = (orig_data.get('rx_rate', 0.0), final_state['rx'], rx_conf)
        repaired_data['tx_rate'] = (orig_data.get('tx_rate', 0.0), final_state['tx'], tx_conf)
        repaired_data['interface_status'] = (orig_data.get('interface_status', 'unknown'), final_state['status'], status_conf)
        
        # Copy metadata
        repaired_data['connected_to'] = orig_data.get('connected_to')
        repaired_data['local_router'] = orig_data.get('local_router')
        repaired_data['remote_router'] = orig_data.get('remote_router')
        
        result[if_id] = repaired_data
        
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

