# EVOLVE-BLOCK-START
"""
Network telemetry repair algorithm that detects and corrects inconsistencies
in network interface telemetry data using topology relationships.
"""
from typing import Dict, Any, Tuple, List

def repair_network_telemetry(telemetry: Dict[str, Dict[str, Any]],
                             topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:
    """
    Repair network interface telemetry by detecting and correcting inconsistencies.
    
    Key Innovations:
    1. Golden Truth Verification: Prioritizes values that satisfy flow conservation 
       at BOTH ends of a link simultaneously.
    2. Dynamic Router Quality: Uses current flow imbalance to weight the reliability 
       of constraints (trust balanced routers more).
    3. Residual Confidence Penalty: Penalizes confidence if the final repair still 
       leaves the router unbalanced.
    """

    HARDENING_THRESHOLD = 0.02

    # 1. Initialization
    state = {}
    for if_id, data in telemetry.items():
        state[if_id] = {
            'rx': float(data.get('rx_rate', 0.0)),
            'tx': float(data.get('tx_rate', 0.0)),
            'status': data.get('interface_status', 'unknown'),
            'rx_conf': 0.0,
            'tx_conf': 0.0,
            'status_conf': 1.0,
            'orig': data
        }

    # 2. Status Consistency (Pass 0)
    for if_id, s in state.items():
        peer_id = s['orig'].get('connected_to')
        if peer_id and peer_id in state:
            peer = state[peer_id]
            # Traffic existence check (ignore tiny noise)
            has_traffic = (s['rx'] > 1.0 or s['tx'] > 1.0 or
                           peer['rx'] > 1.0 or peer['tx'] > 1.0)
            
            if s['status'] != peer['status']:
                if has_traffic:
                    s['status'] = 'up'
                    s['status_conf'] = 0.9
                else:
                    s['status'] = 'down'
                    s['status_conf'] = 0.9
            
        if s['status'] == 'down':
            s['rx'] = 0.0
            s['tx'] = 0.0

    # Helper: Analyze Router State
    # Returns (imbalance, total_flow)
    # Imbalance = Sum(In) - Sum(Out)
    def analyze_router(rid):
        if not rid or rid not in topology:
            return 0.0, 1.0 # Default safe values
        
        in_sum = 0.0
        out_sum = 0.0
        total = 0.0
        
        for iid in topology[rid]:
            if iid in state:
                r = state[iid]['rx']
                t = state[iid]['tx']
                in_sum += r
                out_sum += t
                total += r + t
                
        return in_sum - out_sum, max(total, 1.0)

    # 3. Iterative Repair (Gauss-Seidel)
    ITERATIONS = 5
    sorted_ifs = sorted(state.keys())
    
    for iteration in range(ITERATIONS):
        processed_pairs = set()
        
        for if_id in sorted_ifs:
            s = state[if_id]
            peer_id = s['orig'].get('connected_to')
            
            # Skip if invalid link or already processed
            if not peer_id or peer_id not in state:
                continue
            
            pair_key = tuple(sorted([if_id, peer_id]))
            if pair_key in processed_pairs:
                continue
            processed_pairs.add(pair_key)
            
            peer = state[peer_id]
            
            # Skip if link is down
            if s['status'] == 'down':
                continue

            r_local = s['orig'].get('local_router')
            r_remote = peer['orig'].get('local_router')

            # --- Direction A: Local TX -> Peer RX ---
            # Variables to optimize: s['tx'] and peer['rx'] (should be equal)
            
            meas_tx = s['orig']['tx_rate']
            meas_prx = peer['orig']['rx_rate']
            
            # Get router states
            imb_loc, flow_loc = analyze_router(r_local)
            imb_rem, flow_rem = analyze_router(r_remote)
            
            # Calculate "Base Imbalance" (Imbalance excluding this link's contribution)
            # Local TX is OUT. It contributed -TX to Imbalance (In-Out).
            # So Base = Current + TX
            base_imb_loc = imb_loc + s['tx']
            
            # Remote RX is IN. It contributed +RX to Imbalance.
            # So Base = Current - RX
            base_imb_rem = imb_rem - peer['rx']
            
            # Calculate Router Quality (Reliability) derived from the REST of the interfaces
            # If the rest of the router is balanced, we trust its constraint highly.
            # Quality = 1.0 / (1.0 + Imbalance_Ratio_of_rest)
            q_loc = 1.0 / (1.0 + 10.0 * abs(base_imb_loc) / flow_loc)
            q_rem = 1.0 / (1.0 + 10.0 * abs(base_imb_rem) / flow_rem)
            
            # Candidates to check
            candidates = [meas_tx, meas_prx]
            # If measurements are close, the average is a strong candidate
            if abs(meas_tx - meas_prx) / max(meas_tx, meas_prx, 1.0) < HARDENING_THRESHOLD:
                candidates.append((meas_tx + meas_prx) / 2.0)
                
            # Golden Truth Search
            # A value is "Golden" if it satisfies BOTH routers' flow constraints simultaneously.
            tol = max(meas_tx, meas_prx, 1.0) * HARDENING_THRESHOLD
            
            best_val = meas_tx # Fallback
            best_cost = float('inf')
            
            # Check Golden First
            found_golden = False
            for val in candidates:
                # Predict new imbalances with this candidate
                # Loc New = Base - val (since val is OUT)
                # Rem New = Base + val (since val is IN)
                err_loc = abs(base_imb_loc - val)
                err_rem = abs(base_imb_rem + val)
                
                if err_loc < tol and err_rem < tol:
                    best_val = val
                    found_golden = True
                    break
            
            if not found_golden:
                # Arbitration based on Quality
                # We want to minimize: w1*Err1 + w2*Err2 + w3*Drift
                for val in candidates:
                    err_loc = abs(base_imb_loc - val)
                    err_rem = abs(base_imb_rem + val)
                    drift = min(abs(val - meas_tx), abs(val - meas_prx))
                    
                    # Cost function
                    # If Q is high, we pay high cost for error.
                    cost = (q_loc * err_loc) + (q_rem * err_rem) + (0.1 * drift)
                    
                    if cost < best_cost:
                        best_cost = cost
                        best_val = val
            
            # Apply update
            s['tx'] = best_val
            peer['rx'] = best_val


            # --- Direction B: Local RX <- Peer TX ---
            # Variables to optimize: s['rx'] and peer['tx']
            
            meas_rx = s['orig']['rx_rate']
            meas_ptx = peer['orig']['tx_rate']
            
            # Re-fetch router states (Direction A update changed flows)
            imb_loc, flow_loc = analyze_router(r_local)
            imb_rem, flow_rem = analyze_router(r_remote)
            
            # Loc RX is IN. Contribution +RX. Base = Current - RX
            base_imb_loc = imb_loc - s['rx']
            # Rem TX is OUT. Contribution -TX. Base = Current + TX
            base_imb_rem = imb_rem + peer['tx']
            
            q_loc = 1.0 / (1.0 + 10.0 * abs(base_imb_loc) / flow_loc)
            q_rem = 1.0 / (1.0 + 10.0 * abs(base_imb_rem) / flow_rem)
            
            candidates_b = [meas_rx, meas_ptx]
            if abs(meas_rx - meas_ptx) / max(meas_rx, meas_ptx, 1.0) < HARDENING_THRESHOLD:
                candidates_b.append((meas_rx + meas_ptx) / 2.0)
            
            tol_b = max(meas_rx, meas_ptx, 1.0) * HARDENING_THRESHOLD
            best_val_b = meas_ptx # Fallback
            best_cost_b = float('inf')
            
            found_golden_b = False
            for val in candidates_b:
                # Loc New = Base + val (IN)
                # Rem New = Base - val (OUT)
                err_loc = abs(base_imb_loc + val)
                err_rem = abs(base_imb_rem - val)
                
                if err_loc < tol_b and err_rem < tol_b:
                    best_val_b = val
                    found_golden_b = True
                    break
            
            if not found_golden_b:
                for val in candidates_b:
                    err_loc = abs(base_imb_loc + val)
                    err_rem = abs(base_imb_rem - val)
                    drift = min(abs(val - meas_rx), abs(val - meas_ptx))
                    
                    cost = (q_loc * err_loc) + (q_rem * err_rem) + (0.1 * drift)
                    if cost < best_cost_b:
                        best_cost_b = cost
                        best_val_b = val
                        
            # Apply update
            s['rx'] = best_val_b
            peer['tx'] = best_val_b

    # 4. Final Confidence Calibration
    for if_id, s in state.items():
        if s['status'] == 'down':
            s['rx_conf'] = 1.0
            s['tx_conf'] = 1.0
            continue
            
        peer_id = s['orig'].get('connected_to')
        peer = state.get(peer_id)
        r_local = s['orig'].get('local_router')
        
        # Calculate residual router imbalance
        imb, flow = analyze_router(r_local)
        imbalance_ratio = abs(imb) / flow
        
        # 1. Base Confidence from Measurement Agreement (Symmetry)
        orig_tx = s['orig']['tx_rate']
        orig_rx = s['orig']['rx_rate']
        
        peer_agreement_tx = False
        if peer:
             peer_rx_orig = peer['orig']['rx_rate']
             if abs(orig_tx - peer_rx_orig) / max(orig_tx, 1.0) < HARDENING_THRESHOLD:
                 peer_agreement_tx = True

        peer_agreement_rx = False
        if peer:
             peer_tx_orig = peer['orig']['tx_rate']
             if abs(orig_rx - peer_tx_orig) / max(orig_rx, 1.0) < HARDENING_THRESHOLD:
                 peer_agreement_rx = True

        # 2. Confidence Calculation
        # High confidence if measurements agreed initially
        # Medium confidence if we had to repair but router is now balanced
        # Low confidence if router remains unbalanced
        
        # TX Confidence
        if peer_agreement_tx:
            s['tx_conf'] = 0.95
        else:
            s['tx_conf'] = 0.7 # Default for arbitration
            
        # RX Confidence
        if peer_agreement_rx:
            s['rx_conf'] = 0.95
        else:
            s['rx_conf'] = 0.7
            
        # 3. Residual Penalty
        # If the router is still unbalanced, subtract confidence.
        # This prevents overconfidence in "best effort" repairs that didn't quite work.
        if imbalance_ratio > HARDENING_THRESHOLD:
            # Penalty scales with imbalance
            penalty = min(0.6, imbalance_ratio * 3.0)
            s['rx_conf'] -= penalty
            s['tx_conf'] -= penalty
            
        # Clamp
        s['rx_conf'] = max(0.0, min(1.0, s['rx_conf']))
        s['tx_conf'] = max(0.0, min(1.0, s['tx_conf']))
        
        # Status Confidence Coupling
        if s['rx_conf'] > 0.8 and s['tx_conf'] > 0.8:
            s['status_conf'] = max(s['status_conf'], 0.95)

    # 5. Result Formatting
    result = {}
    for if_id, s in state.items():
        orig = s['orig']
        result[if_id] = {
            'rx_rate': (orig.get('rx_rate', 0.0), s['rx'], s['rx_conf']),
            'tx_rate': (orig.get('tx_rate', 0.0), s['tx'], s['tx_conf']),
            'interface_status': (orig.get('interface_status', 'unknown'), s['status'], s['status_conf']),
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