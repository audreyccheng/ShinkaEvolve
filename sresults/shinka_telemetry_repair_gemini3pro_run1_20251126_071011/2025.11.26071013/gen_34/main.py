# EVOLVE-BLOCK-START
"""
Network telemetry repair algorithm using Weighted Consensus with Virtual Interface Inference.
1. Infers "Virtual" interfaces (present in topology but missing in telemetry) to act as flow slack variables.
2. Solves for rates using an iterative weighted average of Measurement, Symmetry, and Flow Constraints.
3. Dynamically adjusts weights to synthesize traffic on "Double Dead" links using flow residuals.
"""
import math
from typing import Dict, Any, Tuple, List

def repair_network_telemetry(telemetry: Dict[str, Dict[str, Any]], 
                             topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:
    
    # --- Configuration ---
    TRAFFIC_THRESHOLD = 1.0
    ITERATIONS = 5
    CONFIDENCE_DECAY_K = 4.0
    
    # --- Phase 1: Global Graph Construction ---
    # We need to model the entire network, including unmonitored (virtual) interfaces
    # to correctly enforce flow conservation.
    
    # 1. Identify all interfaces
    all_interfaces = set(telemetry.keys())
    router_map = {} # iface -> router_id
    
    for r_id, ifaces in topology.items():
        for iface in ifaces:
            all_interfaces.add(iface)
            router_map[iface] = r_id
            
    # 2. Build Adjacency (Connectivity)
    # Monitored interfaces tell us their peers. We infer the reverse link.
    adjacency = {} # iface -> peer_iface
    for iface, data in telemetry.items():
        peer = data.get('connected_to')
        if peer:
            adjacency[iface] = peer
            adjacency[peer] = iface # Assume bidirectional physical link
            
    # 3. Initialize State
    state = {}
    
    for iface in all_interfaces:
        is_monitored = iface in telemetry
        
        if is_monitored:
            data = telemetry[iface]
            raw_rx = data.get('rx_rate', 0.0)
            raw_tx = data.get('tx_rate', 0.0)
            raw_status = data.get('interface_status', 'unknown')
            peer_id = data.get('connected_to')
            
            # Use peer data for initial status check
            peer_data = telemetry.get(peer_id, {}) if peer_id else {}
            
            # Smart Status Inference
            traffic_max = max(raw_rx, raw_tx, peer_data.get('rx_rate', 0.0), peer_data.get('tx_rate', 0.0))
            
            status = raw_status
            status_conf = 1.0
            
            if traffic_max > TRAFFIC_THRESHOLD:
                if raw_status != 'up':
                    status = 'up'
                    status_conf = 0.95
            elif raw_status == 'up' and peer_data.get('interface_status') == 'down':
                # Peer says DOWN, I say UP, but no significant traffic -> Trust Peer
                status = 'down'
                status_conf = 0.8
            
            # Initialize rates based on status
            rx = raw_rx if status == 'up' else 0.0
            tx = raw_tx if status == 'up' else 0.0
            
            state[iface] = {
                'rx': rx, 'tx': tx,
                'status': status, 'status_conf': status_conf,
                'orig_rx': raw_rx, 'orig_tx': raw_tx,
                'orig_status': raw_status,
                'monitored': True,
                'router': data.get('local_router', router_map.get(iface)),
                'peer': peer_id
            }
        else:
            # Virtual Interface Initialization
            # We assume it exists and is UP (if needed). Rate starts at 0.
            # It will serve as a slack variable during flow optimization.
            state[iface] = {
                'rx': 0.0, 'tx': 0.0,
                'status': 'up', 'status_conf': 0.5, # Low confidence since inferred
                'orig_rx': 0.0, 'orig_tx': 0.0,
                'orig_status': 'unknown',
                'monitored': False,
                'router': router_map.get(iface),
                'peer': adjacency.get(iface)
            }

    # --- Phase 2: Iterative Weighted Consensus ---
    
    for _ in range(ITERATIONS):
        next_state = {}
        
        # 1. Pre-calculate Router Flow Sums
        router_sums = {}
        for r_id, ifaces in topology.items():
            sum_rx = sum(state[i]['rx'] for i in ifaces if i in state)
            sum_tx = sum(state[i]['tx'] for i in ifaces if i in state)
            router_sums[r_id] = {'rx': sum_rx, 'tx': sum_tx}
            
        # 2. Update Every Interface (Monitored & Virtual)
        for iface, curr in state.items():
            # If logically DOWN, keep at 0
            if curr['status'] == 'down':
                next_state[iface] = {'rx': 0.0, 'tx': 0.0}
                continue
                
            peer_id = curr['peer']
            has_peer = peer_id and peer_id in state
            r_id = curr['router']
            
            # --- Calculate Constraint Targets ---
            
            # A. Symmetry Constraint (Peer's Value)
            peer_rx = state[peer_id]['rx'] if has_peer else 0.0
            peer_tx = state[peer_id]['tx'] if has_peer else 0.0
            
            target_sym_rx = peer_tx # My RX should match Peer TX
            target_sym_tx = peer_rx # My TX should match Peer RX
            
            # B. Flow Constraint (Router Balance)
            # The rate required on THIS interface to balance the router totals.
            # New_Val = Current_Val + (Total_Required - Total_Current)
            target_flow_rx = curr['rx']
            target_flow_tx = curr['tx']
            has_flow = False
            
            if r_id and r_id in router_sums:
                rs = router_sums[r_id]
                # To balance: Sum_RX must equal Sum_TX
                
                # Target RX: Absorb the TX surplus / RX deficit
                # if Sum_TX > Sum_RX, we need more RX.
                target_flow_rx = max(0.0, curr['rx'] + (rs['tx'] - rs['rx']))
                
                # Target TX: Absorb the RX surplus / TX deficit
                # if Sum_RX > Sum_TX, we need more TX.
                target_flow_tx = max(0.0, curr['tx'] + (rs['rx'] - rs['tx']))
                has_flow = True

            # --- Weighted Consensus Update ---
            
            def compute_consensus(current, original, sym_target, flow_target, is_monitored):
                # Base Weights
                w_orig = 0.1 if is_monitored else 0.0  # Anchor lightly to measurement
                w_sym = 1.0 if has_peer else 0.0       # Strong pull to symmetry
                w_flow = 1.2 if has_flow else 0.0      # Strongest pull to physics
                
                # Dynamic Adjustments
                
                # 1. "Double Dead" Link Synthesis
                # If both ends read ~0, but Flow Hint implies significant traffic,
                # we assume sensors are dead and trust Flow exclusively.
                if current < TRAFFIC_THRESHOLD and sym_target < TRAFFIC_THRESHOLD:
                    if flow_target > 5.0:
                         w_flow = 10.0 
                         w_sym = 0.1
                         w_orig = 0.0
                
                # 2. Virtual Interface Handling
                if not is_monitored:
                    # Virtuals have no measurement anchor. 
                    # They exist purely to satisfy Flow and Symmetry.
                    w_orig = 0.0
                    w_flow = 2.0 # Allow them to move easily to satisfy flow
                    
                total_w = w_orig + w_sym + w_flow
                if total_w == 0: return current
                
                val = (w_orig * original + w_sym * sym_target + w_flow * flow_target) / total_w
                return val

            next_rx = compute_consensus(curr['rx'], curr['orig_rx'], target_sym_rx, target_flow_rx, curr['monitored'])
            next_tx = compute_consensus(curr['tx'], curr['orig_tx'], target_sym_tx, target_flow_tx, curr['monitored'])
            
            next_state[iface] = {'rx': next_rx, 'tx': next_tx}
            
        # Synchronous Update
        for i, v in next_state.items():
            state[i]['rx'] = v['rx']
            state[i]['tx'] = v['tx']

    # --- Phase 3: Result Generation & Confidence Calibration ---
    result = {}
    
    # Final Flow Calculations for Confidence
    final_sums = {}
    for r_id, ifaces in topology.items():
        sum_rx = sum(state[i]['rx'] for i in ifaces if i in state)
        sum_tx = sum(state[i]['tx'] for i in ifaces if i in state)
        final_sums[r_id] = {'rx': sum_rx, 'tx': sum_tx}

    # Only return repaired telemetry for original inputs
    for iface_id, data in telemetry.items():
        curr = state[iface_id]
        
        # Get References
        peer_id = curr['peer']
        peer_rx = state[peer_id]['rx'] if (peer_id and peer_id in state) else 0.0
        peer_tx = state[peer_id]['tx'] if (peer_id and peer_id in state) else 0.0
        
        r_id = curr['router']
        hint_rx, hint_tx = None, None
        
        if r_id in final_sums:
            rs = final_sums[r_id]
            # Hint is the value that would perfectly balance the router given others
            hint_rx = max(0.0, curr['rx'] + (rs['tx'] - rs['rx']))
            hint_tx = max(0.0, curr['tx'] + (rs['rx'] - rs['tx']))
            
        # Confidence Function
        def calc_conf(val, peer_ref, hint_ref, status_conf, is_down):
            if is_down: return status_conf
            
            # Calculate normalized errors
            err_sym = 1.0
            if peer_id:
                denom = max(val, peer_ref, 1.0)
                err_sym = abs(val - peer_ref) / denom
            else:
                err_sym = 0.0 # No peer implies no symmetry conflict
                
            err_flow = 1.0
            if hint_ref is not None:
                denom = max(val, hint_ref, 1.0)
                err_flow = abs(val - hint_ref) / denom
                
            # We are confident if EITHER constraint is satisfied.
            # This handles cases where one signal is bad (e.g. broken sensor) 
            # but the other validates our repaired value.
            best_err = min(err_sym, err_flow)
            
            # Continuous Exponential Decay
            # err 0.0 -> 1.0
            # err 0.1 -> ~0.67
            # err 0.2 -> ~0.45
            conf = math.exp(-CONFIDENCE_DECAY_K * best_err)
            
            # Bonus: Corroboration (Both signals agree)
            if err_sym < 0.05 and err_flow < 0.05:
                conf = max(conf, 0.98)
                
            return max(0.0, min(1.0, conf))

        conf_rx = calc_conf(curr['rx'], peer_tx, hint_rx, curr['status_conf'], curr['status']=='down')
        conf_tx = calc_conf(curr['tx'], peer_rx, hint_tx, curr['status_conf'], curr['status']=='down')

        result[iface_id] = {
            'rx_rate': (curr['orig_rx'], curr['rx'], conf_rx),
            'tx_rate': (curr['orig_tx'], curr['tx'], conf_tx),
            'interface_status': (curr['orig_status'], curr['status'], curr['status_conf']),
            'connected_to': data.get('connected_to'),
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
