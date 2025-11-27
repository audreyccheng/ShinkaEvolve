# EVOLVE-BLOCK-START
"""
Network telemetry repair algorithm using weighted constraint satisfaction 
and flow conservation heuristics.
"""
from typing import Dict, Any, Tuple, List

def repair_network_telemetry(telemetry: Dict[str, Dict[str, Any]], 
                             topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:
    
    HARDENING_THRESHOLD = 0.02
    MIN_TRAFFIC_THRESHOLD = 0.1 # Mbps
    
    # --- 1. State Initialization ---
    working_state = {}
    for if_id, data in telemetry.items():
        working_state[if_id] = {
            'rx': float(data.get('rx_rate', 0.0)),
            'tx': float(data.get('tx_rate', 0.0)),
            'status': data.get('interface_status', 'unknown'),
            'rx_conf': 0.5,
            'tx_conf': 0.5,
            'status_conf': 1.0,
            'orig': data
        }

    # --- 2. Status Repair (Consensus & Physics) ---
    for if_id, state in working_state.items():
        orig = state['orig']
        peer_id = orig.get('connected_to')
        
        # Determine Status
        # If peer exists, check for consensus
        if peer_id and peer_id in working_state:
            peer_state = working_state[peer_id]
            
            # Logic: If ANY traffic is detected on link, it's UP.
            # (Assuming counters don't increment when down)
            traffic_detected = (
                state['rx'] > MIN_TRAFFIC_THRESHOLD or 
                state['tx'] > MIN_TRAFFIC_THRESHOLD or
                peer_state['orig'].get('rx_rate', 0.0) > MIN_TRAFFIC_THRESHOLD or
                peer_state['orig'].get('tx_rate', 0.0) > MIN_TRAFFIC_THRESHOLD
            )
            
            if traffic_detected:
                state['status'] = 'up'
                # High confidence if traffic confirms it, even if label was wrong
                state['status_conf'] = 0.9 if state['status'] != orig.get('interface_status') else 1.0
            else:
                state['status'] = 'down'
                state['status_conf'] = 0.9 if state['status'] != orig.get('interface_status') else 1.0
        
        # Enforce Physics: Down interfaces have 0 rate
        if state['status'] == 'down':
            state['rx'] = 0.0
            state['tx'] = 0.0
            state['rx_conf'] = 1.0
            state['tx_conf'] = 1.0

    # --- 3. Iterative Flow Optimization ---
    # We solve for flow rates that minimize global imbalance + deviation from trustworthy measurements
    
    ITERATIONS = 5
    sorted_interfaces = sorted(working_state.keys())
    
    # Helper to compute current imbalances
    def get_imbalances_and_flow():
        imbalances = {} # rid -> (in - out)
        flows = {} # rid -> sum(in + out)
        
        for rid in topology:
            imbalances[rid] = 0.0
            flows[rid] = 0.0
            
        for if_id, st in working_state.items():
            rid = st['orig'].get('local_router')
            if rid in topology:
                imbalances[rid] += st['rx'] - st['tx']
                flows[rid] += st['rx'] + st['tx']
        return imbalances, flows

    for it_num in range(ITERATIONS):
        current_imbalances, current_flows = get_imbalances_and_flow()
        
        # Calculate Reliability Q for each router
        router_q = {}
        for rid, flow in current_flows.items():
            if flow < MIN_TRAFFIC_THRESHOLD:
                router_q[rid] = 0.1 # Low trust if inactive
            else:
                imb = current_imbalances[rid]
                # Q = 1.0 at 0 imbalance, decays as imbalance grows relative to flow
                # This acts as a confidence weight for the router's constraints
                router_q[rid] = 1.0 / (1.0 + 20.0 * abs(imb)/flow)

        processed_pairs = set()

        for if_id in sorted_interfaces:
            state = working_state[if_id]
            if state['status'] == 'down': continue
            
            peer_id = state['orig'].get('connected_to')
            if not peer_id or peer_id not in working_state: continue
            
            pair_key = tuple(sorted([if_id, peer_id]))
            if pair_key in processed_pairs: continue
            processed_pairs.add(pair_key)
            
            peer_state = working_state[peer_id]
            if peer_state['status'] == 'down': continue
            
            # Identify Routers
            r_local = state['orig'].get('local_router')
            r_remote = peer_state['orig'].get('local_router')
            
            # --- Direction 1: Local TX -> Peer RX ---
            # Current values
            cur_tx = state['tx']
            cur_rx = peer_state['rx'] # Remote side
            
            # Reliability Weights
            q_local = router_q.get(r_local, 0.0)
            q_remote = router_q.get(r_remote, 0.0)
            
            # Candidates
            candidates = {cur_tx, cur_rx, state['orig'].get('tx_rate', 0.0), peer_state['orig'].get('rx_rate', 0.0)}
            
            # Add Inferred Candidates (Flow Conservation)
            # Infer from Local: Imb_new = Imb_cur + cur_tx - v = 0 => v = Imb_cur + cur_tx
            if r_local in current_imbalances:
                inf_local = current_imbalances[r_local] + cur_tx
                if inf_local > 0: candidates.add(inf_local)
                
            # Infer from Remote: Imb_new = Imb_cur - cur_rx + v = 0 => v = cur_rx - Imb_cur
            if r_remote in current_imbalances:
                inf_remote = cur_rx - current_imbalances[r_remote]
                if inf_remote > 0: candidates.add(inf_remote)
                
            # Average if close (Symmetry assumption)
            if abs(cur_tx - cur_rx) < max(cur_tx, 1.0) * 0.1:
                candidates.add((cur_tx + cur_rx)/2.0)
                
            # Evaluation
            best_val = cur_tx
            min_cost = float('inf')
            
            for v in candidates:
                if v < 0: continue
                cost = 0.0
                
                # Flow Constraint Cost
                # Local (TX is Out): New Imb = Cur + cur_tx - v
                if r_local in current_imbalances:
                    new_imb_local = current_imbalances[r_local] + cur_tx - v
                    cost += q_local * abs(new_imb_local)
                    
                # Remote (RX is In): New Imb = Cur - cur_rx + v
                if r_remote in current_imbalances:
                    new_imb_remote = current_imbalances[r_remote] - cur_rx + v
                    cost += q_remote * abs(new_imb_remote)
                
                # Anchor Cost (Prefer measurements)
                m_tx = state['orig'].get('tx_rate', 0.0)
                m_rx = peer_state['orig'].get('rx_rate', 0.0)
                # Small penalty for deviating from measurements
                # This breaks ties in favor of data and prevents drift
                cost += 0.05 * (abs(v - m_tx) + abs(v - m_rx))
                
                if cost < min_cost:
                    min_cost = cost
                    best_val = v
            
            # Update State & Imbalances immediately (Gauss-Seidel)
            diff_local = best_val - state['tx']
            state['tx'] = best_val
            if r_local in current_imbalances:
                current_imbalances[r_local] -= diff_local # TX increases -> Imb decreases (In-Out)

            diff_remote = best_val - peer_state['rx']
            peer_state['rx'] = best_val
            if r_remote in current_imbalances:
                current_imbalances[r_remote] += diff_remote # RX increases -> Imb increases
                
            # --- Direction 2: Local RX <- Peer TX ---
            # Symmetric logic
            cur_rx_local = state['rx']
            cur_tx_remote = peer_state['tx']
            
            candidates_2 = {cur_rx_local, cur_tx_remote, state['orig'].get('rx_rate', 0.0), peer_state['orig'].get('tx_rate', 0.0)}
            
            # Infer from Local (RX is In): Imb_new = Imb + v - cur_rx = 0 => v = cur_rx - Imb
            if r_local in current_imbalances:
                inf_local_rx = cur_rx_local - current_imbalances[r_local]
                if inf_local_rx > 0: candidates_2.add(inf_local_rx)

            # Infer from Remote (TX is Out): Imb_new = Imb - v + cur_tx = 0 => v = Imb + cur_tx
            if r_remote in current_imbalances:
                inf_remote_tx = current_imbalances[r_remote] + cur_tx_remote
                if inf_remote_tx > 0: candidates_2.add(inf_remote_tx)
                
            if abs(cur_rx_local - cur_tx_remote) < max(cur_rx_local, 1.0) * 0.1:
                candidates_2.add((cur_rx_local + cur_tx_remote)/2.0)

            best_val_2 = cur_rx_local
            min_cost_2 = float('inf')
            
            for v in candidates_2:
                if v < 0: continue
                cost = 0.0
                if r_local in current_imbalances:
                    # Local RX (In): Imb changes by +(v - cur_rx)
                    new_imb = current_imbalances[r_local] - cur_rx_local + v
                    cost += q_local * abs(new_imb)
                if r_remote in current_imbalances:
                    # Remote TX (Out): Imb changes by -(v - cur_tx)
                    new_imb = current_imbalances[r_remote] + cur_tx_remote - v
                    cost += q_remote * abs(new_imb)
                
                m_rx = state['orig'].get('rx_rate', 0.0)
                m_tx = peer_state['orig'].get('tx_rate', 0.0)
                cost += 0.05 * (abs(v - m_rx) + abs(v - m_tx))
                
                if cost < min_cost_2:
                    min_cost_2 = cost
                    best_val_2 = v
            
            diff_l = best_val_2 - state['rx']
            state['rx'] = best_val_2
            if r_local in current_imbalances:
                current_imbalances[r_local] += diff_l
                
            diff_r = best_val_2 - peer_state['tx']
            peer_state['tx'] = best_val_2
            if r_remote in current_imbalances:
                current_imbalances[r_remote] -= diff_r

    # --- 4. Final Confidence Calibration ---
    final_imbalances, final_flows = get_imbalances_and_flow()
    
    for if_id, st in working_state.items():
        if st['status'] == 'down': continue
        
        orig = st['orig']
        peer_id = orig.get('connected_to')
        peer_st = working_state.get(peer_id)
        
        r_local = orig.get('local_router')
        r_remote = peer_st['orig'].get('local_router') if peer_st else None
        
        # Helper: Get Quality of Router (Solidity)
        def get_solidity(rid):
            if not rid or rid not in final_flows: return 0.0
            f = final_flows[rid]
            i = final_imbalances[rid]
            if f < 0.1: return 1.0
            ratio = abs(i) / f
            return 1.0 if ratio < HARDENING_THRESHOLD else max(0.0, 1.0 - ratio * 10.0)

        sol_local = get_solidity(r_local)
        sol_remote = get_solidity(r_remote)
        
        # Calculate for RX and TX
        def calc_conf(val, meas, peer_meas, s_loc, s_rem):
            # Agreement checks
            agrees_meas = abs(val - meas) < max(meas, 1.0) * HARDENING_THRESHOLD
            agrees_peer = abs(val - peer_meas) < max(peer_meas, 1.0) * HARDENING_THRESHOLD
            
            score = 0.5 # Base
            
            # Tiered Confidence
            if s_loc > 0.9 and s_rem > 0.9:
                score = 0.95 # Consensus in flow: both routers balanced
            elif agrees_meas and agrees_peer:
                score = 1.0 # Perfect agreement between measurements
            elif agrees_peer:
                score = 0.9 # Trust peer measurement
            elif agrees_meas and (s_loc > 0.8):
                score = 0.9 # Trust self if self is solid
            else:
                # Inferred or Averaged or just messy
                if s_loc > 0.8 or s_rem > 0.8:
                    score = 0.85 # One side is solid, likely inferred correctly
                else:
                    score = 0.4 # Low confidence
            
            # Penalty for residual imbalance (Global context)
            # If the router is still messy, reduce confidence slightly
            penalty = (1.0 - s_loc) * 0.2 + (1.0 - s_rem) * 0.2
            return max(0.0, min(1.0, score - penalty))

        # RX Confidence
        peer_tx = peer_st['orig'].get('tx_rate', 0.0) if peer_st else 0.0
        st['rx_conf'] = calc_conf(st['rx'], orig.get('rx_rate', 0.0), peer_tx, sol_local, sol_remote)
                                 
        # TX Confidence
        peer_rx = peer_st['orig'].get('rx_rate', 0.0) if peer_st else 0.0
        st['tx_conf'] = calc_conf(st['tx'], orig.get('tx_rate', 0.0), peer_rx, sol_local, sol_remote)

    # --- 5. Assemble Output ---
    result = {}
    for if_id, st in working_state.items():
        orig = st['orig']
        result[if_id] = {
            'rx_rate': (orig.get('rx_rate', 0.0), st['rx'], st['rx_conf']),
            'tx_rate': (orig.get('tx_rate', 0.0), st['tx'], st['tx_conf']),
            'interface_status': (orig.get('interface_status', 'unknown'), st['status'], st['status_conf']),
            'connected_to': orig.get('connected_to'),
            'local_router': orig.get('local_router'),
            'remote_router': orig.get('remote_router')
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