# EVOLVE-BLOCK-START
from typing import Dict, Any, Tuple, List
import math

def repair_network_telemetry(telemetry: Dict[str, Dict[str, Any]], 
                             topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:
    """
    Repairs network telemetry using a Bayesian-inspired iterative consensus algorithm.
    Features:
    - Link Symmetry and Flow Conservation models.
    - Explicit 'Null Flow' (0.0) hypothesis testing to detect phantom traffic.
    - 'Computed' hypothesis for external links to allow edge correction.
    - Confidence calibration based on both relative probability and absolute fit.
    """
    
    # --- Configuration ---
    SYMMETRY_TOLERANCE = 0.02
    CONSERVATION_TOLERANCE_PCT = 0.03
    MIN_SIGNIFICANT_FLOW = 0.5
    ITERATIONS = 5
    
    # --- 1. Preprocessing & Topology ---
    if_to_router = {}
    for r_id, if_list in topology.items():
        for i_id in if_list:
            if_to_router[i_id] = r_id
            
    # Current State Dictionary
    # if_id -> {'rx': val, 'tx': val}
    current_state = {}
    
    # Initialize with measurements
    for if_id, data in telemetry.items():
        current_state[if_id] = {
            'rx': data.get('rx_rate', 0.0),
            'tx': data.get('tx_rate', 0.0)
        }

    # Helper to compute router imbalance
    def get_router_imbalance(rid, state):
        if rid not in topology: return 0.0, 1.0
        total_in = 0.0
        total_out = 0.0
        max_flow = 1.0
        
        for iid in topology[rid]:
            if iid in state:
                r = state[iid]['rx']
                t = state[iid]['tx']
                total_in += r
                total_out += t
                max_flow = max(max_flow, r, t)
        
        return (total_in - total_out), max_flow

    # --- 2. Iterative Improvement ---
    
    for _ in range(ITERATIONS):
        next_state = {k: v.copy() for k, v in current_state.items()}
        
        for if_id, data in telemetry.items():
            peer_id = data.get('connected_to')
            is_internal = peer_id and peer_id in telemetry
            
            # --- A. Handle TX Flow (if_id -> peer) ---
            # For internal links, this handles the peer's RX as well.
            
            src_if = if_id
            dst_if = peer_id
            
            val_local_tx = data.get('tx_rate', 0.0)
            
            # Gather Candidates
            candidates_tx = {val_local_tx, 0.0}
            
            if is_internal:
                val_peer_rx = telemetry[dst_if].get('rx_rate', 0.0)
                candidates_tx.add(val_peer_rx)
                # Average for smoothing
                if abs(val_local_tx - val_peer_rx) < max(val_local_tx, val_peer_rx, 1.0) * 0.2:
                    candidates_tx.add((val_local_tx + val_peer_rx) / 2.0)
            else:
                # External TX: Can we infer from router balance?
                # We want Out = In - Other_Out.
                # My_TX = (In - Other_Out_Excl_Me)
                rid = if_to_router.get(src_if)
                if rid:
                    # Current Imb = In - Out_Total
                    # Wanted Imb = 0
                    # Imb = In - (Other + My_TX)
                    # My_New_TX = In - Other = Imb + My_Old_TX
                    cur_imb, _ = get_router_imbalance(rid, current_state)
                    computed = cur_imb + current_state[src_if]['tx']
                    if computed > 0:
                        candidates_tx.add(computed)
            
            # Select Best Candidate for TX
            best_tx_val = current_state[src_if]['tx']
            best_tx_score = -1.0
            
            r_src = if_to_router.get(src_if)
            r_dst = if_to_router.get(dst_if) if is_internal else None
            
            for val in candidates_tx:
                # 1. Source Conservation (TX leaves router)
                score_src = 1.0
                if r_src:
                    old_val = current_state[src_if]['tx']
                    imb, flow = get_router_imbalance(r_src, current_state)
                    # New Imb = Old_Imb - (New_Val - Old_Val)
                    new_imb = imb - (val - old_val)
                    sigma = max(flow * CONSERVATION_TOLERANCE_PCT, 0.5)
                    score_src = math.exp(-abs(new_imb) / sigma)
                
                # 2. Dest Conservation (RX enters router) - only if internal
                score_dst = 1.0
                if r_dst:
                    old_val = current_state[dst_if]['rx']
                    imb, flow = get_router_imbalance(r_dst, current_state)
                    # New Imb = Old_Imb + (New_Val - Old_Val)
                    new_imb = imb + (val - old_val)
                    sigma = max(flow * CONSERVATION_TOLERANCE_PCT, 0.5)
                    score_dst = math.exp(-abs(new_imb) / sigma)
                
                # 3. Measurement Prior
                # Prior favors local TX and Peer RX (if exists)
                dist_local = abs(val - val_local_tx)
                sigma_meas = max(val_local_tx * 0.05, 1.0)
                prior = math.exp(-dist_local / sigma_meas)
                
                if is_internal:
                    val_peer_rx = telemetry[dst_if].get('rx_rate', 0.0)
                    dist_peer = abs(val - val_peer_rx)
                    sigma_peer = max(val_peer_rx * 0.05, 1.0)
                    prior = max(prior, math.exp(-dist_peer / sigma_peer))
                
                # 0.0 Prior Penalty/Boost
                if val == 0.0:
                    # If measurements are large, 0.0 is unlikely unless conservation demands it strongly
                    measured_max = max(val_local_tx, telemetry[dst_if].get('rx_rate', 0.0) if is_internal else 0)
                    if measured_max > 5.0:
                        prior *= 0.1
                
                total = score_src * score_dst * math.sqrt(prior + 0.001)
                if total > best_tx_score:
                    best_tx_score = total
                    best_tx_val = val
            
            next_state[src_if]['tx'] = best_tx_val
            if is_internal:
                next_state[dst_if]['rx'] = best_tx_val
            
            # --- B. Handle External RX Flow (Cloud -> if_id) ---
            if not is_internal:
                val_local_rx = data.get('rx_rate', 0.0)
                candidates_rx = {val_local_rx, 0.0}
                
                # Infer from router balance?
                # My_RX = Out_Total - Other_In = Out_Total - (In_Total - My_Old_RX)
                # My_New_RX = My_Old_RX - Imb (since Imb = In - Out)
                rid = if_to_router.get(if_id)
                if rid:
                    cur_imb, _ = get_router_imbalance(rid, current_state)
                    computed = current_state[if_id]['rx'] - cur_imb
                    if computed > 0:
                        candidates_rx.add(computed)
                
                best_rx_val = current_state[if_id]['rx']
                best_rx_score = -1.0
                
                for val in candidates_rx:
                    # 1. Router Conservation (RX enters router)
                    score_r = 1.0
                    if rid:
                        old_val = current_state[if_id]['rx']
                        imb, flow = get_router_imbalance(rid, current_state)
                        # New Imb = Old_Imb + (New_Val - Old_Val)
                        new_imb = imb + (val - old_val)
                        sigma = max(flow * CONSERVATION_TOLERANCE_PCT, 0.5)
                        score_r = math.exp(-abs(new_imb) / sigma)
                    
                    # 2. Prior
                    dist = abs(val - val_local_rx)
                    sigma_m = max(val_local_rx * 0.05, 1.0)
                    prior = math.exp(-dist / sigma_m)
                    
                    if val == 0.0 and val_local_rx > 5.0:
                        prior *= 0.1
                        
                    total = score_r * math.sqrt(prior + 0.001)
                    if total > best_rx_score:
                        best_rx_score = total
                        best_rx_val = val
                        
                next_state[if_id]['rx'] = best_rx_val
                
        current_state = next_state

    # --- 3. Final Assembly ---
    result = {}
    
    for if_id, data in telemetry.items():
        rx_val = current_state[if_id]['rx']
        tx_val = current_state[if_id]['tx']
        
        rid = if_to_router.get(if_id)
        
        # Conservation Quality
        cons_score = 0.95
        if rid:
            imb, flow = get_router_imbalance(rid, current_state)
            # Use a slightly loose tolerance for confidence scoring to avoid punishing minor noise
            sigma = max(flow * 0.05, 2.0) 
            cons_score = math.exp(-abs(imb) / sigma)
            
        # Measurement agreement
        orig_rx = data.get('rx_rate', 0.0)
        orig_tx = data.get('tx_rate', 0.0)
        
        # Determine Confidence
        # Base confidence is conservation score.
        conf_rx = cons_score
        conf_tx = cons_score
        
        # Boost if matches measurement
        if abs(rx_val - orig_rx) < max(orig_rx, 1.0) * 0.05:
            conf_rx = max(conf_rx, 0.8 + 0.2 * cons_score)
        # Boost if matches peer (for internal)
        peer_id = data.get('connected_to')
        if peer_id and peer_id in telemetry:
            peer_tx = telemetry[peer_id].get('tx_rate', 0.0)
            if abs(rx_val - peer_tx) < max(peer_tx, 1.0) * 0.05:
                 conf_rx = max(conf_rx, 0.9 + 0.1 * cons_score)
        
        if abs(tx_val - orig_tx) < max(orig_tx, 1.0) * 0.05:
            conf_tx = max(conf_tx, 0.8 + 0.2 * cons_score)
            
        # --- Status ---
        orig_status = data.get('interface_status', 'unknown')
        peer_status = telemetry.get(peer_id, {}).get('interface_status', 'unknown') if peer_id in telemetry else 'unknown'
        
        has_traffic = (rx_val > MIN_SIGNIFICANT_FLOW) or (tx_val > MIN_SIGNIFICANT_FLOW)
        
        rep_status = orig_status
        conf_status = 1.0
        
        if has_traffic:
            rep_status = 'up'
            if orig_status != 'up':
                conf_status = (conf_rx + conf_tx) / 2.0
        elif peer_status == 'down':
            rep_status = 'down'
            if orig_status != 'down':
                conf_status = 0.9
        elif orig_status == 'up' and not has_traffic:
            # Idle is valid
            rep_status = 'up'
            
        # Post-process down
        if rep_status == 'down':
            rx_val = 0.0
            tx_val = 0.0
            conf_rx = max(conf_rx, 0.95)
            conf_tx = max(conf_tx, 0.95)
            
        entry = {}
        entry['rx_rate'] = (orig_rx, rx_val, conf_rx)
        entry['tx_rate'] = (orig_tx, tx_val, conf_tx)
        entry['interface_status'] = (orig_status, rep_status, conf_status)
        
        for k in ['connected_to', 'local_router', 'remote_router']:
            if k in data: entry[k] = data[k]
            
        result[if_id] = entry
        
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
