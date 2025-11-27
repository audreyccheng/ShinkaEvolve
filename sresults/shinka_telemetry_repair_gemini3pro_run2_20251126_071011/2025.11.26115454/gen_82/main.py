# EVOLVE-BLOCK-START
"""
Network telemetry repair algorithm that detects and corrects inconsistencies
in network interface telemetry data using topology relationships.

Takes interface telemetry data and detects/repairs inconsistencies based on
network invariants like link symmetry and flow conservation.
"""
from typing import Dict, Any, Tuple, List


def repair_network_telemetry(telemetry: Dict[str, Dict[str, Any]],
                             topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:
    """
    Repair network interface telemetry by detecting and correcting inconsistencies.

    Core principle: Use network invariants to validate and repair telemetry:
    1. Link Symmetry (R3): my_tx_rate ≈ their_rx_rate for connected interfaces
    2. Flow Conservation (R1): Sum(incoming traffic) = Sum(outgoing traffic) at each router
    3. Interface Consistency: Status should be consistent across connected pairs

    Args:
        telemetry: Dictionary where key is interface_id and value contains:
            - interface_status: "up" or "down"
            - rx_rate: receive rate in Mbps
            - tx_rate: transmit rate in Mbps
            - connected_to: interface_id this interface connects to
            - local_router: router_id this interface belongs to
            - remote_router: router_id on the other side
        topology: Dictionary where key is router_id and value contains a list of interface_ids

    Returns:
        Dictionary with same structure but telemetry values become tuples of:
        (original_value, repaired_value, confidence_score)
        where confidence ranges from 0.0 (very uncertain) to 1.0 (very confident)
    """

    HARDENING_THRESHOLD = 0.02

    # Initialize working state
    state = {}
    
    # Pre-fill state
    for if_id, data in telemetry.items():
        state[if_id] = {
            'rx': float(data.get('rx_rate', 0.0)),
            'tx': float(data.get('tx_rate', 0.0)),
            'status': data.get('interface_status', 'unknown'),
            'rx_conf': 0.5, 
            'tx_conf': 0.5,
            'status_conf': 1.0,
            'orig': data
        }

    # --- Phase 1: Status Consensus ---
    # Status dictates whether rates are possible. We repair status first.
    for if_id, s in state.items():
        peer_id = s['orig'].get('connected_to')
        if peer_id and peer_id in state:
            peer = state[peer_id]
            
            # If statuses disagree, check for traffic evidence
            if s['status'] != peer['status']:
                # Any significant traffic implies the link is physically UP
                has_traffic = (s['rx'] > 1.0 or s['tx'] > 1.0 or 
                               peer['rx'] > 1.0 or peer['tx'] > 1.0)
                
                if has_traffic:
                    s['status'] = 'up'
                    s['status_conf'] = 0.9
                else:
                    s['status'] = 'down'
                    s['status_conf'] = 0.9
        
        # Enforce physics: If DOWN, rates must be zero.
        if s['status'] == 'down':
            s['rx'] = 0.0
            s['tx'] = 0.0
            s['rx_conf'] = 1.0
            s['tx_conf'] = 1.0

    # --- Phase 2: Iterative Flow & Symmetry Repair ---
    ITERATIONS = 5
    sorted_ifs = sorted(state.keys()) # Deterministic order
    
    # Helper: Calculate router imbalance excluding a specific interface
    # Imbalance = Sum(In) - Sum(Out)
    # We want to find what value 'v' for the excluded interface would make Imbalance = 0.
    def get_router_partial_imbalance(rid, exclude_if_id):
        if not rid or rid not in topology:
            return 0.0
        
        in_sum = 0.0
        out_sum = 0.0
        
        for iid in topology[rid]:
            if iid == exclude_if_id:
                continue
            if iid in state:
                in_sum += state[iid]['rx']
                out_sum += state[iid]['tx']
                
        return in_sum - out_sum

    # Helper: Estimate router reliability based on neighbor confidences
    # Used to weight constraints (Trust reliable routers more)
    def get_router_quality(rid):
        if not rid or rid not in topology:
            return 0.0
        
        total_conf = 0.0
        count = 0
        for iid in topology[rid]:
            if iid in state:
                # Average of RX/TX confidence
                c = (state[iid]['rx_conf'] + state[iid]['tx_conf']) / 2.0
                total_conf += c
                count += 1
        
        if count == 0: return 0.0
        return total_conf / count

    for iteration in range(ITERATIONS):
        processed_pairs = set()
        
        for if_id in sorted_ifs:
            s = state[if_id]
            peer_id = s['orig'].get('connected_to')
            
            # Skip if unconnected or already processed
            if not peer_id or peer_id not in state:
                continue
                
            pair_key = tuple(sorted([if_id, peer_id]))
            if pair_key in processed_pairs:
                continue
            processed_pairs.add(pair_key)
            
            peer = state[peer_id]
            
            # Skip repair if link is effectively dead
            if s['status'] == 'down' and peer['status'] == 'down':
                continue

            # === Direction 1: Local TX -> Peer RX ===
            m_tx = s['orig'].get('tx_rate', 0.0)
            m_rx = peer['orig'].get('rx_rate', 0.0)
            
            # Initial Symmetry Lock: If measurements agree perfectly, anchor them.
            if iteration == 0:
                denom = max(m_tx, m_rx, 1.0)
                if abs(m_tx - m_rx) / denom < HARDENING_THRESHOLD:
                    avg_val = (m_tx + m_rx) / 2.0
                    s['tx'] = avg_val
                    peer['rx'] = avg_val
                    s['tx_conf'] = 1.0
                    peer['rx_conf'] = 1.0
            
            r_local = s['orig'].get('local_router')
            r_remote = peer['orig'].get('local_router')
            
            # Calculate Flow Targets
            # Target Local (Sender): Needs to balance Outbound flow. 
            # Imbalance = In - (Out_other + current). current = In - Out_other.
            imb_local = get_router_partial_imbalance(r_local, if_id)
            target_local = imb_local 
            
            # Target Remote (Receiver): Needs to balance Inbound flow.
            # Imbalance = (In_other + current) - Out. current = Out - In_other.
            imb_remote = get_router_partial_imbalance(r_remote, peer_id)
            target_remote = -imb_remote 
            
            # Router Qualities
            q_local = get_router_quality(r_local) if r_local in topology else 0.0
            q_remote = get_router_quality(r_remote) if r_remote in topology else 0.0
            q_local = max(q_local, 0.1)
            q_remote = max(q_remote, 0.1)
            
            # Generate Candidates
            candidates = [m_tx, m_rx]
            if abs(m_tx - m_rx) < max(m_tx, m_rx, 1.0) * 0.1:
                candidates.append((m_tx + m_rx) / 2.0)
            
            best_val = s['tx'] # Start with current
            best_conf = s['tx_conf']
            
            # "Golden Truth" Search
            # Find a value that satisfies BOTH routers' constraints within threshold
            golden_found = False
            for v in candidates:
                if v < 0: continue
                
                fit_local = abs(v - target_local) / max(v, 1.0)
                fit_remote = abs(v - target_remote) / max(v, 1.0)
                
                # Check if fits both routers well
                if (r_local in topology and fit_local < HARDENING_THRESHOLD and
                    r_remote in topology and fit_remote < HARDENING_THRESHOLD):
                    best_val = v
                    best_conf = 1.0
                    golden_found = True
                    break
            
            if not golden_found:
                # Weighted Arbitration
                min_cost = float('inf')
                for v in candidates:
                    if v < 0: continue
                    cost = 0.0
                    if r_local in topology: cost += q_local * abs(v - target_local)
                    if r_remote in topology: cost += q_remote * abs(v - target_remote)
                    
                    # Preference for original measurements (Bias towards TX as source)
                    cost += 0.01 * abs(v - m_tx)
                    cost += 0.02 * abs(v - m_rx)
                    
                    if cost < min_cost:
                        min_cost = cost
                        best_val = v
                
                # Confidence Calibration based on Fit
                err_local = abs(best_val - target_local) / max(best_val, 1.0) if r_local in topology else 1.0
                err_remote = abs(best_val - target_remote) / max(best_val, 1.0) if r_remote in topology else 1.0
                
                sym_score = 1.0 if abs(m_tx - m_rx)/max(m_tx, 1.0) < HARDENING_THRESHOLD else 0.5
                
                flow_fit = 0.0
                if r_local in topology and r_remote in topology:
                    if err_local < 0.05 and err_remote < 0.05: flow_fit = 0.95
                    elif err_local < 0.05: flow_fit = 0.8 * q_local
                    elif err_remote < 0.05: flow_fit = 0.8 * q_remote
                elif r_local in topology:
                    if err_local < 0.05: flow_fit = 0.9 * q_local
                elif r_remote in topology:
                    if err_remote < 0.05: flow_fit = 0.9 * q_remote
                    
                best_conf = max(sym_score, flow_fit)

            # Apply updates A->B
            s['tx'] = best_val
            peer['rx'] = best_val
            s['tx_conf'] = best_conf
            peer['rx_conf'] = best_conf
            
            
            # === Direction 2: Local RX <- Peer TX ===
            m_rx = s['orig'].get('rx_rate', 0.0)
            m_tx = peer['orig'].get('tx_rate', 0.0)
            
            if iteration == 0:
                denom = max(m_rx, m_tx, 1.0)
                if abs(m_rx - m_tx) / denom < HARDENING_THRESHOLD:
                    avg_val = (m_rx + m_tx) / 2.0
                    s['rx'] = avg_val
                    peer['tx'] = avg_val
                    s['rx_conf'] = 1.0
                    peer['tx_conf'] = 1.0

            # Target Local (Receiver): -Partial_Imb
            imb_local = get_router_partial_imbalance(r_local, if_id)
            target_local = -imb_local
            
            # Target Remote (Sender): Partial_Imb
            imb_remote = get_router_partial_imbalance(r_remote, peer_id)
            target_remote = imb_remote
            
            candidates_b = [m_rx, m_tx]
            if abs(m_rx - m_tx) < max(m_rx, m_tx, 1.0) * 0.1:
                candidates_b.append((m_rx + m_tx) / 2.0)
                
            best_val_b = s['rx']
            best_conf_b = s['rx_conf']
            
            golden_found_b = False
            for v in candidates_b:
                if v < 0: continue
                fit_local = abs(v - target_local) / max(v, 1.0)
                fit_remote = abs(v - target_remote) / max(v, 1.0)
                
                if (r_local in topology and fit_local < HARDENING_THRESHOLD and
                    r_remote in topology and fit_remote < HARDENING_THRESHOLD):
                    best_val_b = v
                    best_conf_b = 1.0
                    golden_found_b = True
                    break
            
            if not golden_found_b:
                min_cost_b = float('inf')
                for v in candidates_b:
                    if v < 0: continue
                    cost = 0.0
                    if r_local in topology: cost += q_local * abs(v - target_local)
                    if r_remote in topology: cost += q_remote * abs(v - target_remote)
                    
                    cost += 0.01 * abs(v - m_tx)
                    cost += 0.02 * abs(v - m_rx)
                    
                    if cost < min_cost_b:
                        min_cost_b = cost
                        best_val_b = v
                        
                err_local = abs(best_val_b - target_local) / max(best_val_b, 1.0) if r_local in topology else 1.0
                err_remote = abs(best_val_b - target_remote) / max(best_val_b, 1.0) if r_remote in topology else 1.0
                
                sym_score = 1.0 if abs(m_rx - m_tx)/max(m_tx, 1.0) < HARDENING_THRESHOLD else 0.5
                
                flow_fit = 0.0
                if r_local in topology and r_remote in topology:
                    if err_local < 0.05 and err_remote < 0.05: flow_fit = 0.95
                    elif err_local < 0.05: flow_fit = 0.8 * q_local
                    elif err_remote < 0.05: flow_fit = 0.8 * q_remote
                elif r_local in topology:
                    if err_local < 0.05: flow_fit = 0.9 * q_local
                elif r_remote in topology:
                    if err_remote < 0.05: flow_fit = 0.9 * q_remote
                
                best_conf_b = max(sym_score, flow_fit)
            
            # Apply updates B->A
            s['rx'] = best_val_b
            peer['tx'] = best_val_b
            s['rx_conf'] = best_conf_b
            peer['tx_conf'] = best_conf_b

    # --- Phase 3: Final Calibration with Residual Penalty ---
    # If a router is strictly imbalanced after all repairs, penalize confidence.
    for rid in topology:
        in_sum = 0.0
        out_sum = 0.0
        total_flow = 0.0
        
        for iid in topology[rid]:
            if iid in state:
                in_sum += state[iid]['rx']
                out_sum += state[iid]['tx']
                total_flow += state[iid]['rx'] + state[iid]['tx']
                
        if total_flow < 1.0: continue
        
        imbalance = abs(in_sum - out_sum)
        imb_ratio = imbalance / total_flow
        
        # Penalize if residual imbalance exists
        if imb_ratio > HARDENING_THRESHOLD:
            penalty = min(0.6, (imb_ratio - HARDENING_THRESHOLD) * 4.0)
            
            for iid in topology[rid]:
                if iid in state:
                    state[iid]['rx_conf'] = max(0.0, state[iid]['rx_conf'] - penalty)
                    state[iid]['tx_conf'] = max(0.0, state[iid]['tx_conf'] - penalty)

    # --- Assemble Output ---
    result = {}
    for if_id, s in state.items():
        orig = s['orig']
        
        # Consistency: High rate confidence implies high status confidence
        if s['rx_conf'] > 0.8 and s['tx_conf'] > 0.8:
            s['status_conf'] = max(s['status_conf'], 0.95)
            
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