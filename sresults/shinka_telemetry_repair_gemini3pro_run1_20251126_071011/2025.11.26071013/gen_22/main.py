# EVOLVE-BLOCK-START
"""
Network telemetry repair algorithm using Iterative Flow Consensus.
Refines telemetry data through multiple passes of constraint satisfaction,
solving for the most likely network state that satisfies Symmetry and Flow Conservation.
Uses continuous confidence calibration and synthetic repair for dead links.
"""
from typing import Dict, Any, Tuple, List
import math

def repair_network_telemetry(telemetry: Dict[str, Dict[str, Any]], 
                             topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:
    
    # Constants
    HARDENING_THRESHOLD = 0.02   # 2% tolerance for measurement timing
    TRAFFIC_THRESHOLD = 1.0      # 1 Mbps threshold for "active" link
    ITERATIONS = 5               # Increased iterations for better propagation
    CONFIDENCE_DECAY = 4.0       # Exponential decay factor for confidence (k)
    
    # --- Phase 1: Initialization & Status Repair ---
    state = {}
    
    for iface_id, data in telemetry.items():
        raw_rx = data.get('rx_rate', 0.0)
        raw_tx = data.get('tx_rate', 0.0)
        raw_status = data.get('interface_status', 'unknown')
        
        peer_id = data.get('connected_to')
        peer_data = telemetry.get(peer_id) if (peer_id and peer_id in telemetry) else {}
        
        # Traffic Evidence
        signals = [raw_rx, raw_tx, peer_data.get('rx_rate', 0.0), peer_data.get('tx_rate', 0.0)]
        max_traffic = max(signals) if signals else 0.0
        
        # Status Repair: Traffic presence overrides 'down' status
        status = raw_status
        status_conf = 1.0
        
        if max_traffic > TRAFFIC_THRESHOLD:
            if raw_status != 'up':
                status = 'up'
                status_conf = 0.95
        elif raw_status == 'up' and peer_data.get('interface_status') == 'down':
            # Peer says DOWN, I say UP, but no traffic -> Likely DOWN
            status = 'down'
            status_conf = 0.8
            
        # Initial Rate Beliefs
        # Clamp to 0.0 if status is down or noise
        if status == 'down':
            cur_rx, cur_tx = 0.0, 0.0
        else:
            cur_rx = raw_rx if raw_rx > TRAFFIC_THRESHOLD else 0.0
            cur_tx = raw_tx if raw_tx > TRAFFIC_THRESHOLD else 0.0
            
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
        
    # --- Phase 2: Iterative Constraint Satisfaction ---
    # We iterate to allow corrections (e.g., fixing a bad link) to propagate 
    # to the Flow Hints of neighbors.
    
    for _ in range(ITERATIONS):
        next_rates = {}
        
        # 1. Calculate Router Balances (Flow Hints) based on CURRENT beliefs
        router_balances = {} # router_id -> {'rx': float, 'tx': float}
        
        for r_id, ifaces in topology.items():
            sum_rx = 0.0
            sum_tx = 0.0
            for i in ifaces:
                if i in state:
                    sum_rx += state[i]['rx']
                    sum_tx += state[i]['tx']
            router_balances[r_id] = {'rx': sum_rx, 'tx': sum_tx}
            
        # 2. Evaluate each interface
        for iface_id, curr in state.items():
            if curr['status'] == 'down':
                next_rates[iface_id] = {'rx': 0.0, 'tx': 0.0}
                continue
                
            peer_id = curr['peer_id']
            has_peer = peer_id and peer_id in state
            r_id = curr.get('local_router')
            
            # Helper to resolve a rate (RX or TX)
            def resolve_rate(local_val, peer_val, is_rx):
                # Calculate Flow Hint for this specific direction
                val_hint = local_val # Default fallback
                has_hint = False
                
                if r_id and r_id in router_balances:
                    rb = router_balances[r_id]
                    if is_rx:
                        # RX must match Total TX
                        val_hint = max(0.0, rb['tx'] - (rb['rx'] - local_val))
                    else:
                        # TX must match Total RX
                        val_hint = max(0.0, rb['rx'] - (rb['tx'] - local_val))
                    has_hint = True
                
                # Decision Logic
                
                # 1. Check Symmetry (Direct Measurement)
                denom_sym = max(local_val, peer_val, 1.0)
                diff_sym = abs(local_val - peer_val) / denom_sym
                
                if diff_sym <= HARDENING_THRESHOLD:
                    # Symmetry holds: Reinforce by averaging
                    return (local_val + peer_val) / 2.0
                
                # 2. Symmetry Broken: Check for "Double Dead" scenario
                # If both sides are near zero, but Physics says there should be flow.
                if local_val < TRAFFIC_THRESHOLD and peer_val < TRAFFIC_THRESHOLD:
                    if has_hint and val_hint > 5.0: # Significant missing flow detected
                        return val_hint # Synthesize traffic from flow residual
                    return 0.0
                
                # 3. Standard Broken Symmetry: Minimize Error
                if not has_hint:
                    return peer_val # Trust peer if no local flow info (heuristic)

                denom_l = max(local_val, val_hint, 1.0)
                err_l = abs(local_val - val_hint) / denom_l
                
                denom_p = max(peer_val, val_hint, 1.0)
                err_p = abs(peer_val - val_hint) / denom_p
                
                if err_l < err_p:
                    return local_val
                else:
                    return peer_val

            # Resolve RX (Target: Peer TX)
            peer_tx = state[peer_id]['tx'] if has_peer else curr['rx']
            next_rx = resolve_rate(curr['rx'], peer_tx, is_rx=True)
            
            # Resolve TX (Target: Peer RX)
            peer_rx = state[peer_id]['rx'] if has_peer else curr['tx']
            next_tx = resolve_rate(curr['tx'], peer_rx, is_rx=False)
            
            next_rates[iface_id] = {'rx': next_rx, 'tx': next_tx}
            
        # Synchronous Update
        for iface, rates in next_rates.items():
            state[iface]['rx'] = rates['rx']
            state[iface]['tx'] = rates['tx']
            
    # --- Phase 3: Final Confidence Calibration ---
    result = {}
    
    # Recalculate final router sums for accurate calibration
    final_balances = {}
    for r_id, ifaces in topology.items():
        sum_rx = sum(state[i]['rx'] for i in ifaces if i in state)
        sum_tx = sum(state[i]['tx'] for i in ifaces if i in state)
        final_balances[r_id] = {'rx': sum_rx, 'tx': sum_tx}
        
    for iface_id, data in state.items():
        final_rx = data['rx']
        final_tx = data['tx']
        peer_id = data['peer_id']
        has_peer = peer_id and peer_id in state
        
        # Calibration Function
        def get_confidence(val, peer_val, hint_val, status_conf, is_down):
            if is_down:
                # If we suppressed real traffic, reduce confidence
                if val > TRAFFIC_THRESHOLD: 
                    return status_conf
                return status_conf 
                
            # Calculate Errors
            err_sym = 1.0
            if has_peer:
                denom = max(val, peer_val, 1.0)
                err_sym = abs(val - peer_val) / denom
            else:
                # Self-symmetry is perfect if no peer (edge case)
                err_sym = 0.0
                
            err_flow = 1.0
            if hint_val is not None:
                denom = max(val, hint_val, 1.0)
                err_flow = abs(val - hint_val) / denom
                
            # We trust the result if it is supported by EITHER Symmetry OR Flow.
            # We take the minimum error (best support).
            best_error = min(err_sym, err_flow)
            
            # Continuous Calibration: Exponential Decay
            # 0 error -> 1.0
            # 2% error -> ~0.92
            # 10% error -> ~0.67
            conf = math.exp(-CONFIDENCE_DECAY * best_error)
            
            # Slight penalty if the best support is Flow but Symmetry is totally broken
            # (Indicates local sensor is right, but link is noisy/peer is wrong)
            # We don't penalize heavily because we found a valid explanation (Flow).
            
            return max(0.0, min(1.0, conf))

        # Get Hints for final verification
        r_id = data['local_router']
        hint_rx = None
        hint_tx = None
        if r_id and r_id in final_balances:
            rb = final_balances[r_id]
            hint_rx = max(0.0, rb['tx'] - (rb['rx'] - final_rx))
            hint_tx = max(0.0, rb['rx'] - (rb['tx'] - final_tx))
            
        peer_tx = state[peer_id]['tx'] if has_peer else final_rx
        peer_rx = state[peer_id]['rx'] if has_peer else final_tx
        
        conf_rx = get_confidence(final_rx, peer_tx, hint_rx, data['status_conf'], data['status'] == 'down')
        conf_tx = get_confidence(final_tx, peer_rx, hint_tx, data['status_conf'], data['status'] == 'down')
        
        result[iface_id] = {
            'rx_rate': (data['orig_rx'], final_rx, conf_rx),
            'tx_rate': (data['orig_tx'], final_tx, conf_tx),
            'interface_status': (data['orig_status'], data['status'], data['status_conf']),
            'connected_to': peer_id,
            'local_router': r_id,
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
