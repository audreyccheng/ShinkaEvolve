# EVOLVE-BLOCK-START
"""
Network telemetry repair algorithm using Topology-Aware Flow Consensus.
Handles partial observability by selectively applying Flow Conservation 
only to fully monitored routers. Uses a prioritized evidence hierarchy:
1. Symmetry (Direct agreement)
2. Flow Conservation (Indirect validation, if router fully visible)
3. Traffic Maximization (Heuristic for partial visibility/broken links)
"""
from typing import Dict, Any, Tuple, List

def repair_network_telemetry(telemetry: Dict[str, Dict[str, Any]], 
                             topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:
    
    # Constants
    HARDENING_THRESHOLD = 0.02   # 2% tolerance for measurement timing
    TRAFFIC_THRESHOLD = 1.0      # 1 Mbps threshold for "active" link
    ITERATIONS = 5               # Propagation passes
    CONFIDENCE_SLOPE = 5.0       # Linear decay slope for confidence
    
    # --- Phase 1: Assessment & Initialization ---
    
    # Identify Fully Monitored Routers
    # Flow Conservation (Sum In = Sum Out) can strictly apply only if we see all interfaces.
    # If interfaces are missing from telemetry, we cannot enforce flow balance on the router.
    fully_monitored_routers = set()
    for r_id, ifaces in topology.items():
        if all(i in telemetry for i in ifaces):
            fully_monitored_routers.add(r_id)
            
    # Initialize State
    state = {}
    for iface_id, data in telemetry.items():
        raw_rx = data.get('rx_rate', 0.0)
        raw_tx = data.get('tx_rate', 0.0)
        raw_status = data.get('interface_status', 'unknown')
        
        peer_id = data.get('connected_to')
        peer_data = telemetry.get(peer_id, {}) if (peer_id and peer_id in telemetry) else {}
        
        # Max traffic signal on the link
        signals = [raw_rx, raw_tx, peer_data.get('rx_rate', 0.0), peer_data.get('tx_rate', 0.0)]
        max_traffic = max(signals) if signals else 0.0
        
        # Status Inference
        status = raw_status
        status_conf = 1.0
        
        if max_traffic > TRAFFIC_THRESHOLD:
            if raw_status != 'up':
                status = 'up'
                status_conf = 0.95
        elif raw_status == 'up' and peer_data.get('interface_status') == 'down':
            # Contradiction with no traffic support -> likely down
            status = 'down'
            status_conf = 0.8
            
        # Rate Initialization
        if status == 'down':
            cur_rx, cur_tx = 0.0, 0.0
        else:
            cur_rx = raw_rx
            cur_tx = raw_tx
            
        state[iface_id] = {
            'rx': cur_rx,
            'tx': cur_tx,
            'status': status,
            'status_conf': status_conf,
            'orig_rx': raw_rx,
            'orig_tx': raw_tx,
            'orig_status': raw_status,
            'peer_id': peer_id,
            'local_router': data.get('local_router')
        }

    # --- Phase 2: Iterative Refinement ---
    
    for _ in range(ITERATIONS):
        next_rates = {}
        
        # Calculate Router Balances (Flow Hints) for verifiable routers
        router_balances = {} 
        for r_id in fully_monitored_routers:
            ifaces = topology.get(r_id, [])
            sum_rx = sum(state[i]['rx'] for i in ifaces if i in state)
            sum_tx = sum(state[i]['tx'] for i in ifaces if i in state)
            router_balances[r_id] = {'rx': sum_rx, 'tx': sum_tx}
            
        for iface_id, curr in state.items():
            if curr['status'] == 'down':
                next_rates[iface_id] = {'rx': 0.0, 'tx': 0.0}
                continue
                
            peer_id = curr['peer_id']
            has_peer = peer_id and peer_id in state
            r_id = curr.get('local_router')
            
            # Rate Resolution Logic
            def resolve(local_val, peer_val, is_rx):
                # 1. Get Flow Hint if available
                val_hint = None
                if r_id in router_balances:
                    rb = router_balances[r_id]
                    if is_rx:
                        # RX needed to match Total TX
                        others_rx = rb['rx'] - local_val
                        val_hint = max(0.0, rb['tx'] - others_rx)
                    else:
                        # TX needed to match Total RX
                        others_tx = rb['tx'] - local_val
                        val_hint = max(0.0, rb['rx'] - others_tx)
                
                # 2. Check Symmetry
                denom = max(local_val, peer_val, 1.0)
                diff = abs(local_val - peer_val) / denom
                
                if diff <= HARDENING_THRESHOLD:
                    # Symmetry holds: Average
                    return (local_val + peer_val) / 2.0
                
                # 3. Symmetry Broken
                
                if val_hint is not None:
                    # Case A: Have Hint (Flow Conservation)
                    
                    # Double Dead Check: If sensors say 0 but flow says X
                    if local_val < TRAFFIC_THRESHOLD and peer_val < TRAFFIC_THRESHOLD:
                        if val_hint > 5.0: # Significant missing flow
                             return val_hint
                        return 0.0
                    
                    # Pick value closer to hint
                    denom_l = max(local_val, val_hint, 1.0)
                    err_l = abs(local_val - val_hint) / denom_l
                    
                    denom_p = max(peer_val, val_hint, 1.0)
                    err_p = abs(peer_val - val_hint) / denom_p
                    
                    if err_l < err_p:
                        return local_val
                    else:
                        return peer_val
                        
                else:
                    # Case B: No Hint (Partial/Unmonitored Router)
                    # Heuristic: Trust positive signal (assuming counters stick to 0 on fail)
                    if local_val < TRAFFIC_THRESHOLD and peer_val > TRAFFIC_THRESHOLD:
                        return peer_val
                    elif peer_val < TRAFFIC_THRESHOLD and local_val > TRAFFIC_THRESHOLD:
                        return local_val
                    
                    # If both significant but different, average them to minimize max error
                    return (local_val + peer_val) / 2.0

            # Resolve RX (vs Peer TX)
            peer_tx = state[peer_id]['tx'] if has_peer else curr['rx']
            next_rx = resolve(curr['rx'], peer_tx, is_rx=True)
            
            # Resolve TX (vs Peer RX)
            peer_rx = state[peer_id]['rx'] if has_peer else curr['tx']
            next_tx = resolve(curr['tx'], peer_rx, is_rx=False)
            
            next_rates[iface_id] = {'rx': next_rx, 'tx': next_tx}
            
        # Synchronous State Update
        for i_id, vals in next_rates.items():
            state[i_id]['rx'] = vals['rx']
            state[i_id]['tx'] = vals['tx']

    # --- Phase 3: Calibration ---
    result = {}
    
    # Recalculate final balances for scoring
    final_balances = {}
    for r_id in fully_monitored_routers:
        ifaces = topology.get(r_id, [])
        sum_rx = sum(state[i]['rx'] for i in ifaces if i in state)
        sum_tx = sum(state[i]['tx'] for i in ifaces if i in state)
        final_balances[r_id] = {'rx': sum_rx, 'tx': sum_tx}
        
    for iface_id, data in state.items():
        final_rx = data['rx']
        final_tx = data['tx']
        peer_id = data['peer_id']
        has_peer = peer_id and peer_id in state
        
        peer_tx = state[peer_id]['tx'] if has_peer else final_rx
        peer_rx = state[peer_id]['rx'] if has_peer else final_tx
        
        def calibrate(val, peer_val, hint_val, status_conf, is_down):
            if is_down:
                return status_conf
            
            # Evidence Errors
            err_sym = 0.0
            if has_peer:
                denom = max(val, peer_val, 1.0)
                err_sym = abs(val - peer_val) / denom
            
            err_flow = None
            if hint_val is not None:
                denom = max(val, hint_val, 1.0)
                err_flow = abs(val - hint_val) / denom
            
            # Select Best Evidence
            if err_flow is not None:
                best_err = min(err_sym, err_flow)
            else:
                best_err = err_sym
                
            # Linear Decay Calibration
            # 1.0 at 0 error, decaying to 0.0 at 20% error (slope=5.0)
            base_conf = max(0.0, 1.0 - (best_err * CONFIDENCE_SLOPE))
            
            # Minor penalty if supporting evidence is weak/contradictory
            if err_flow is not None and err_sym > 0.1 and err_flow < 0.05:
                # Flow saved us, but link is asymmetric (broken sensor)
                base_conf = min(base_conf, 0.95)
            
            return base_conf

        # Get Hint Reference
        r_id = data['local_router']
        hint_rx, hint_tx = None, None
        if r_id in final_balances:
            rb = final_balances[r_id]
            hint_rx = max(0.0, rb['tx'] - (rb['rx'] - final_rx))
            hint_tx = max(0.0, rb['rx'] - (rb['tx'] - final_tx))
            
        conf_rx = calibrate(final_rx, peer_tx, hint_rx, data['status_conf'], data['status'] == 'down')
        conf_tx = calibrate(final_tx, peer_rx, hint_tx, data['status_conf'], data['status'] == 'down')
        
        result[iface_id] = {
            'rx_rate': (data['orig_rx'], final_rx, conf_rx),
            'tx_rate': (data['orig_tx'], final_tx, conf_tx),
            'interface_status': (data['orig_status'], data['status'], data['status_conf']),
            'connected_to': peer_id,
            'local_router': data['local_router'],
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
