# EVOLVE-BLOCK-START
import math
from typing import Dict, Any, Tuple, List

def repair_network_telemetry(telemetry: Dict[str, Dict[str, Any]], 
                             topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:
    
    # Constants
    HARDENING_THRESHOLD = 0.02   # 2% tolerance
    TRAFFIC_THRESHOLD = 1.0      # 1 Mbps active threshold
    ITERATIONS = 5               # Convergence passes
    
    # Weights for cost function
    WEIGHT_SYMMETRY = 1.0
    WEIGHT_FLOW = 2.5            # Trust physics (Flow) significantly more than sensors
    
    # --- Phase 1: Initialization & Status Logic ---
    state = {}
    
    # Identify Verifiable Routers (All interfaces monitored)
    verifiable_routers = set()
    for r_id, ifaces in topology.items():
        if all(i in telemetry for i in ifaces):
            verifiable_routers.add(r_id)
            
    for iface_id, data in telemetry.items():
        raw_rx = data.get('rx_rate', 0.0)
        raw_tx = data.get('tx_rate', 0.0)
        raw_status = data.get('interface_status', 'unknown')
        
        peer_id = data.get('connected_to')
        peer_data = telemetry.get(peer_id, {}) if (peer_id and peer_id in telemetry) else {}
        
        # Traffic Evidence
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
            status = 'down'
            status_conf = 0.8
            
        # Initial Rates
        if status == 'down':
            cur_rx, cur_tx = 0.0, 0.0
        else:
            cur_rx = raw_rx if raw_rx > 0 else 0.0
            cur_tx = raw_tx if raw_tx > 0 else 0.0
            
        state[iface_id] = {
            'rx': cur_rx,
            'tx': cur_tx,
            'status': status,
            'status_conf': status_conf,
            'orig_rx': raw_rx,
            'orig_tx': raw_tx,
            'orig_status': raw_status,
            'peer_id': peer_id,
            'local_router': data.get('local_router'),
            'remote_router': data.get('remote_router')
        }
        
    # --- Phase 2: Iterative Consensus ---
    
    for _ in range(ITERATIONS):
        # 1. Calculate Flow Potentials
        router_balances = {}
        for r_id in verifiable_routers:
            ifaces = topology[r_id]
            sum_rx = sum(state[i]['rx'] for i in ifaces)
            sum_tx = sum(state[i]['tx'] for i in ifaces)
            router_balances[r_id] = {'rx': sum_rx, 'tx': sum_tx}
            
        next_updates = {}
        
        for iface_id, curr in state.items():
            if curr['status'] == 'down':
                next_updates[iface_id] = {'rx': 0.0, 'tx': 0.0}
                continue
                
            peer_id = curr['peer_id']
            has_peer = peer_id and peer_id in state
            r_id = curr.get('local_router')
            
            # Rate Solver
            def solve_rate(local_val, peer_val, is_rx):
                # Candidates
                candidates = {local_val, peer_val}
                
                # Flow Hint
                flow_hint = None
                if r_id in router_balances:
                    rb = router_balances[r_id]
                    if is_rx:
                        # RX must match Total TX
                        flow_hint = max(0.0, rb['tx'] - (rb['rx'] - local_val))
                    else:
                        flow_hint = max(0.0, rb['rx'] - (rb['tx'] - local_val))
                    candidates.add(flow_hint)
                    
                # Evaluate Candidates
                best_val = local_val
                min_cost = float('inf')
                
                for cand in candidates:
                    cost = 0.0
                    
                    # Symmetry Penalty
                    denom_s = max(cand, peer_val, 1.0)
                    err_s = abs(cand - peer_val) / denom_s
                    if err_s > HARDENING_THRESHOLD:
                        cost += min(1.0, err_s) * WEIGHT_SYMMETRY
                        
                    # Flow Penalty
                    if flow_hint is not None:
                        denom_f = max(cand, flow_hint, 1.0)
                        err_f = abs(cand - flow_hint) / denom_f
                        if err_f > HARDENING_THRESHOLD:
                            cost += min(1.0, err_f) * WEIGHT_FLOW
                            
                    if cost < min_cost:
                        min_cost = cost
                        best_val = cand
                    elif abs(cost - min_cost) < 1e-6:
                        # Tie-breaking: Prefer higher value (Traffic Existence Heuristic)
                        # This helps in partial observability where one side is 0 and other is High
                        if cand > best_val:
                            best_val = cand
                            
                return best_val

            # Resolve RX (vs Peer TX)
            peer_tx = state[peer_id]['tx'] if has_peer else curr['rx']
            nxt_rx = solve_rate(curr['rx'], peer_tx, is_rx=True)
            
            # Resolve TX (vs Peer RX)
            peer_rx = state[peer_id]['rx'] if has_peer else curr['tx']
            nxt_tx = solve_rate(curr['tx'], peer_rx, is_rx=False)
            
            next_updates[iface_id] = {'rx': nxt_rx, 'tx': nxt_tx}
            
        # Apply Updates
        for i, vals in next_updates.items():
            state[i]['rx'] = vals['rx']
            state[i]['tx'] = vals['tx']
            
    # --- Phase 3: Discrete Confidence Calibration ---
    result = {}
    
    # Final Flow Check
    final_balances = {}
    for r_id in verifiable_routers:
        ifaces = topology[r_id]
        sum_rx = sum(state[i]['rx'] for i in ifaces)
        sum_tx = sum(state[i]['tx'] for i in ifaces)
        final_balances[r_id] = {'rx': sum_rx, 'tx': sum_tx}
        
    for iface_id, data in state.items():
        final_rx = data['rx']
        final_tx = data['tx']
        peer_id = data['peer_id']
        has_peer = peer_id and peer_id in state
        
        peer_tx = state[peer_id]['tx'] if has_peer else final_rx
        peer_rx = state[peer_id]['rx'] if has_peer else final_tx
        
        def get_confidence(val, peer_val, is_rx, status_conf, is_down):
            if is_down:
                return status_conf if val > TRAFFIC_THRESHOLD else status_conf
            
            # 1. Calculate Errors
            denom_s = max(val, peer_val, 1.0)
            err_sym = abs(val - peer_val) / denom_s
            
            err_flow = None
            r_id = data['local_router']
            if r_id in final_balances:
                rb = final_balances[r_id]
                if is_rx:
                    hint = max(0.0, rb['tx'] - (rb['rx'] - val))
                else:
                    hint = max(0.0, rb['rx'] - (rb['tx'] - val))
                denom_f = max(val, hint, 1.0)
                err_flow = abs(val - hint) / denom_f
                
            # 2. Assign Tiered Confidence
            
            # Case: Flow Verified (Strongest)
            if err_flow is not None and err_flow <= HARDENING_THRESHOLD:
                if err_sym <= HARDENING_THRESHOLD:
                    return 1.0  # Perfect (Verified by both)
                else:
                    return 0.95 # Rescued (Verified by Flow, Sensor corrected)
            
            # Case: Symmetry Verified (No Flow)
            if err_sym <= HARDENING_THRESHOLD:
                if err_flow is None:
                    return 0.90 # Corroborated (Verified by Peer)
                else:
                    return 0.75 # Contradiction (Sym OK, Flow Bad - likely unseen interface issue)
                    
            # Case: Unverified / Heuristic
            if err_flow is None:
                # We guessed (e.g., max traffic).
                return 0.60 
            
            # Case: Failure (Both Bad)
            return 0.20

        conf_rx = get_confidence(final_rx, peer_tx, True, data['status_conf'], data['status']=='down')
        conf_tx = get_confidence(final_tx, peer_rx, False, data['status_conf'], data['status']=='down')
        
        result[iface_id] = {
            'rx_rate': (data['orig_rx'], final_rx, conf_rx),
            'tx_rate': (data['orig_tx'], final_tx, conf_tx),
            'interface_status': (data['orig_status'], data['status'], data['status_conf']),
            'connected_to': peer_id,
            'local_router': data['local_router'],
            'remote_router': data['remote_router']
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
