# EVOLVE-BLOCK-START
"""
Network telemetry repair algorithm using State-Aware Flow Consensus.
Combines robust state initialization with evidence-based consensus logic.
1. Establishes effective status/rates to clean input noise.
2. Calculates Flow Conservation hints from cleaned state.
3. Resolves conflicts using a prioritized Symmetry > Hint architecture with calibrated confidence.
"""
from typing import Dict, Any, Tuple, List

def repair_network_telemetry(telemetry: Dict[str, Dict[str, Any]], 
                             topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:
    
    HARDENING_THRESHOLD = 0.02  # 2% tolerance for measurement timing
    TRAFFIC_THRESHOLD = 1.0     # 1 Mbps threshold for active link detection
    
    # --- Phase 1: State Initialization ---
    # Create a cleaned view of the network where Status flags are corrected 
    # based on traffic evidence. This ensures that Flow Hints (calculated next)
    # use realistic traffic volumes even if the original flags were wrong.
    
    prepared_data = {}
    
    for iface_id, data in telemetry.items():
        raw_rx = data.get('rx_rate', 0.0)
        raw_tx = data.get('tx_rate', 0.0)
        raw_status = data.get('interface_status', 'unknown')
        
        peer_id = data.get('connected_to')
        peer_data = telemetry.get(peer_id, {}) if (peer_id and peer_id in telemetry) else {}
        
        # 1. Status Repair Logic
        # Heuristic: Traffic flow is the ground truth.
        traffic_signals = [raw_rx, raw_tx, peer_data.get('rx_rate', 0.0), peer_data.get('tx_rate', 0.0)]
        max_traffic = max(traffic_signals) if traffic_signals else 0.0
        
        eff_status = raw_status
        status_conf = 1.0
        
        if max_traffic > TRAFFIC_THRESHOLD:
            # Traffic implies UP
            if raw_status != 'up':
                eff_status = 'up'
                status_conf = 0.95
        elif raw_status == 'up' and peer_data.get('interface_status') == 'down':
             # Contradiction with no significant traffic -> Trust Peer's DOWN
             eff_status = 'down'
             status_conf = 0.8
             
        # 2. Determine Effective Rates for Sums
        # If link is deemed DOWN, it contributes 0 to the router balance.
        if eff_status == 'down':
            eff_rx, eff_tx = 0.0, 0.0
        else:
            eff_rx, eff_tx = raw_rx, raw_tx
            
        prepared_data[iface_id] = {
            'rx': eff_rx,
            'tx': eff_tx,
            'status': eff_status,
            'status_conf': status_conf,
            'orig_rx': raw_rx,
            'orig_tx': raw_tx,
            'orig_status': raw_status,
            'peer_id': peer_id,
            'has_peer': bool(peer_data)
        }

    # --- Phase 2: Flow Hint Calculation ---
    # Calculate what each interface's rate *should* be to balance the router.
    # Uses the 'prepared' effective rates.
    
    flow_hints = {}
    
    for router_id, iface_ids in topology.items():
        valid_ifaces = [i for i in iface_ids if i in prepared_data]
        
        # Calculate router totals
        sum_rx = sum(prepared_data[i]['rx'] for i in valid_ifaces)
        sum_tx = sum(prepared_data[i]['tx'] for i in valid_ifaces)
        
        for iface in valid_ifaces:
            curr = prepared_data[iface]
            
            # Hint RX: sum_tx - (sum_rx - my_rx)
            # "The RX needed to feed all current TX, given other inputs"
            rx_hint = max(0.0, sum_tx - (sum_rx - curr['rx']))
            
            # Hint TX: sum_rx - (sum_tx - my_tx)
            # "The TX supported by all current RX, given other outputs"
            tx_hint = max(0.0, sum_rx - (sum_tx - curr['tx']))
            
            flow_hints[iface] = {'rx': rx_hint, 'tx': tx_hint}

    # --- Phase 3: Resolution & Repair ---
    result = {}
    
    for iface_id, p_data in prepared_data.items():
        orig_rx = p_data['orig_rx']
        orig_tx = p_data['orig_tx']
        
        rep_rx, rep_tx = orig_rx, orig_tx
        conf_rx, conf_tx = 1.0, 1.0
        
        if p_data['status'] == 'down':
            rep_rx, rep_tx = 0.0, 0.0
            # If we suppressed real traffic, confidence is tied to our status decision
            if orig_rx > TRAFFIC_THRESHOLD or orig_tx > TRAFFIC_THRESHOLD:
                conf_rx = p_data['status_conf']
                conf_tx = p_data['status_conf']
        elif p_data['has_peer']:
            peer_id = p_data['peer_id']
            # We use the peer's *effective* rates from prepared_data
            # This handles cases where peer status was corrected
            peer_rx = prepared_data[peer_id]['rx']
            peer_tx = prepared_data[peer_id]['tx']
            
            hints = flow_hints.get(iface_id, {'rx': orig_rx, 'tx': orig_tx})

            def resolve(local_val, peer_val, hint_val):
                # 1. Symmetry Check
                # Does Local match Peer?
                denom_sym = max(local_val, peer_val, 1.0)
                diff_sym = abs(local_val - peer_val) / denom_sym
                
                if diff_sym <= HARDENING_THRESHOLD:
                    # Consistent: Average them
                    return (local_val + peer_val) / 2.0, 1.0
                
                # 2. Symmetry Broken: Flow Hint Arbiter
                # Which candidate is closer to the Flow Conservation requirement?
                
                denom_l = max(local_val, hint_val, 1.0)
                diff_l = abs(local_val - hint_val) / denom_l
                
                denom_p = max(peer_val, hint_val, 1.0)
                diff_p = abs(peer_val - hint_val) / denom_p
                
                if diff_l < diff_p:
                    # Local is closer to Hint.
                    # Confidence depends on the margin of victory
                    margin = diff_p - diff_l
                    # If margin is wide, we are very sure. If narrow, less sure.
                    # 0.9/0.7 bucketing has proven effective for calibration.
                    conf = 0.9 if margin > 0.1 else 0.7
                    return local_val, conf
                else:
                    # Peer is closer to Hint.
                    margin = diff_l - diff_p
                    conf = 0.9 if margin > 0.1 else 0.7
                    return peer_val, conf

            # Repair RX (Local RX vs Peer TX)
            rep_rx, conf_rx = resolve(orig_rx, peer_tx, hints['rx'])
            
            # Repair TX (Local TX vs Peer RX)
            rep_tx, conf_tx = resolve(orig_tx, peer_rx, hints['tx'])

        # Final Result
        result[iface_id] = {
            'rx_rate': (orig_rx, rep_rx, conf_rx),
            'tx_rate': (orig_tx, rep_tx, conf_tx),
            'interface_status': (p_data['orig_status'], p_data['status'], p_data['status_conf']),
            'connected_to': p_data['peer_id'],
            'local_router': telemetry[iface_id].get('local_router'),
            'remote_router': telemetry[iface_id].get('remote_router')
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

