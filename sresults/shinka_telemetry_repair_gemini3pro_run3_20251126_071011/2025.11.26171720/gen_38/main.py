# EVOLVE-BLOCK-START
from typing import Dict, Any, Tuple, List
import math

def repair_network_telemetry(telemetry: Dict[str, Dict[str, Any]], 
                             topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:
    """
    Repairs network telemetry using a Momentum-based Bayesian Consensus algorithm.
    
    Key Innovations:
    1.  **Hypothesis Generation**: Explicitly tests '0.0' (Phantom Traffic) and 'Implied' (Router Balance) 
        hypotheses alongside measured values.
    2.  **Non-Linear Noise Model**: Uses a tolerance model scaling with sqrt(Flow) to better model 
        Poisson packet process noise vs systematic error.
    3.  **Soft Updates**: Uses momentum to stabilize the iterative solver across the network graph.
    4.  **Calibrated Confidence**: Confidence scores combine hypothesis probability (relative) with 
        goodness-of-fit (absolute) to penalize "best of a bad bunch" solutions.
    """
    
    # --- Configuration ---
    SYMMETRY_TOLERANCE = 0.02
    ABS_TOLERANCE = 1.0
    CONSERVATION_TOLERANCE_PCT = 0.02
    ITERATIONS = 5
    MOMENTUM = 0.5 # Retain 50% of old value, accept 50% of new
    
    # --- Data Structures ---
    # estimates[if_id] = {'rx': val, 'tx': val}
    estimates = {}
    # confidences[if_id] = {'rx': val, 'tx': val}
    confidences = {}
    
    # Map interface to router
    if_to_router = {}
    for r_id, if_list in topology.items():
        for i_id in if_list:
            if_to_router[i_id] = r_id
            
    # Link Identification
    links = {} 
    processed_ifs = set()
    
    for if_id, data in telemetry.items():
        if if_id in processed_ifs: continue
        
        peer = data.get('connected_to')
        if peer and peer in telemetry:
            key = tuple(sorted([if_id, peer]))
            links[key] = {'type': 'internal', 'if1': if_id, 'if2': peer}
            processed_ifs.add(if_id)
            processed_ifs.add(peer)
        else:
            links[(if_id,)] = {'type': 'external', 'if1': if_id}
            processed_ifs.add(if_id)

    # --- Step 1: Initialization & Symmetry Check ---
    
    for link_key, info in links.items():
        if1 = info['if1']
        d1 = telemetry[if1]
        
        if info['type'] == 'internal':
            if2 = info['if2']
            d2 = telemetry[if2]
            
            # Forward: IF1 TX -> IF2 RX
            tx1 = d1.get('tx_rate', 0.0)
            rx2 = d2.get('rx_rate', 0.0)
            
            diff = abs(tx1 - rx2)
            denom = max(tx1, rx2, 1.0)
            
            # If symmetric, initialize to average (Hardening step)
            if diff < ABS_TOLERANCE or (diff/denom) < SYMMETRY_TOLERANCE:
                avg = (tx1 + rx2) / 2.0
                estimates[if1] = estimates.get(if1, {})
                estimates[if1]['tx'] = avg
                estimates[if2] = estimates.get(if2, {})
                estimates[if2]['rx'] = avg
            else:
                # Trust local initially
                estimates[if1] = estimates.get(if1, {})
                estimates[if1]['tx'] = tx1
                estimates[if2] = estimates.get(if2, {})
                estimates[if2]['rx'] = rx2
                
            # Backward: IF2 TX -> IF1 RX
            tx2 = d2.get('tx_rate', 0.0)
            rx1 = d1.get('rx_rate', 0.0)
            
            diff = abs(tx2 - rx1)
            denom = max(tx2, rx1, 1.0)
            
            if diff < ABS_TOLERANCE or (diff/denom) < SYMMETRY_TOLERANCE:
                avg = (tx2 + rx1) / 2.0
                estimates[if2]['tx'] = avg
                estimates[if1]['rx'] = avg
            else:
                estimates[if2]['tx'] = tx2
                estimates[if1]['rx'] = rx1
        else:
            # External
            estimates[if1] = estimates.get(if1, {})
            estimates[if1]['tx'] = d1.get('tx_rate', 0.0)
            estimates[if1]['rx'] = d1.get('rx_rate', 0.0)
            
    # Default confidence
    for if_id in telemetry:
        confidences[if_id] = {'rx': 0.5, 'tx': 0.5}
        
    # --- Step 2: Iterative Solver ---
    
    def get_sigma(flow):
        # Noise model: Scales with sqrt(flow) for Poisson noise, plus linear term
        return max(0.5 * math.sqrt(flow), flow * CONSERVATION_TOLERANCE_PCT, 1.0)

    for iteration in range(ITERATIONS):
        # Calculate Router Imbalances
        router_balances = {}
        for rid, ifs in topology.items():
            total_in = 0.0
            total_out = 0.0
            for iid in ifs:
                total_in += estimates.get(iid, {}).get('rx', 0.0)
                total_out += estimates.get(iid, {}).get('tx', 0.0)
            router_balances[rid] = {
                'in': total_in, 'out': total_out, 
                'imb': total_in - total_out, # In - Out
                'max_flow': max(total_in, total_out, 1.0)
            }
            
        batch_updates = {}
        
        for link_key, info in links.items():
            # Strategy: Generate hypotheses for correct flow value, score them, pick winner
            
            if info['type'] == 'internal':
                if1, if2 = info['if1'], info['if2']
                
                # Helper for bidirectional solving
                def solve_internal(src, dst, m_src, m_dst, type_src, type_dst):
                    r_src = if_to_router.get(src)
                    r_dst = if_to_router.get(dst)
                    
                    # Hypotheses: Measured Src, Measured Dst, Zero
                    hyps = sorted(list(set([m_src, m_dst, 0.0])))
                    scores = []
                    
                    for h in hyps:
                        # Score Src Conservation
                        p_src = 1.0
                        if r_src:
                            # Imb = In - Out. 
                            # If we change Src (TX), we change Out.
                            # New Imb = Old Imb + Old Val - New Val
                            old_val = estimates[src]['tx']
                            bal = router_balances[r_src]
                            pred_imb = bal['imb'] + old_val - h
                            sigma = get_sigma(bal['max_flow'])
                            p_src = math.exp(-abs(pred_imb) / sigma)
                            
                        # Score Dst Conservation
                        p_dst = 1.0
                        if r_dst:
                            # Imb = In - Out
                            # If we change Dst (RX), we change In.
                            # New Imb = Old Imb - Old Val + New Val
                            old_val = estimates[dst]['rx']
                            bal = router_balances[r_dst]
                            pred_imb = bal['imb'] - old_val + h
                            sigma = get_sigma(bal['max_flow'])
                            p_dst = math.exp(-abs(pred_imb) / sigma)
                            
                        # Prior
                        prior = 1.0
                        # Weak penalty for 0 if measurements are high
                        if h == 0.0 and max(m_src, m_dst) > 10.0:
                            prior = 0.8
                            
                        scores.append(p_src * p_dst * prior)
                    
                    # Selection
                    total_s = sum(scores) + 1e-12
                    probs = [s/total_s for s in scores]
                    best_idx = scores.index(max(scores))
                    
                    winner = hyps[best_idx]
                    win_prob = probs[best_idx]
                    fit_quality = scores[best_idx] # Raw likelihood acts as absolute quality
                    
                    # If prior < 1, normalize fit quality for confidence
                    if winner == 0.0 and max(m_src, m_dst) > 10.0:
                        fit_quality /= 0.8
                        
                    conf = win_prob * math.sqrt(fit_quality)
                    conf = max(0.01, min(0.99, conf))
                    
                    return winner, conf

                # 1 -> 2
                val1 = telemetry[if1].get('tx_rate', 0.0)
                val2 = telemetry[if2].get('rx_rate', 0.0)
                win_val, win_conf = solve_internal(if1, if2, val1, val2, 'tx', 'rx')
                
                # Soft Update
                old = estimates[if1]['tx']
                new_val = (1 - MOMENTUM) * win_val + MOMENTUM * old
                batch_updates[(if1, 'tx')] = (new_val, win_conf)
                batch_updates[(if2, 'rx')] = (new_val, win_conf)
                
                # 2 -> 1
                val2 = telemetry[if2].get('tx_rate', 0.0)
                val1 = telemetry[if1].get('rx_rate', 0.0)
                win_val, win_conf = solve_internal(if2, if1, val2, val1, 'tx', 'rx')
                
                old = estimates[if2]['tx']
                new_val = (1 - MOMENTUM) * win_val + MOMENTUM * old
                batch_updates[(if2, 'tx')] = (new_val, win_conf)
                batch_updates[(if1, 'rx')] = (new_val, win_conf)
                
            else:
                # External Link
                if1 = info['if1']
                rid = if_to_router.get(if1)
                
                if not rid: continue 
                
                bal = router_balances[rid]
                sigma = get_sigma(bal['max_flow'])
                
                # Solve TX (Out)
                # Imb = In - Out. We want Imb = 0.
                # Out_new = In_others + My_TX_new.
                # My_TX_new = In - Out_others = My_TX_old + Imb
                curr_tx = estimates[if1]['tx']
                implied_tx = max(0.0, curr_tx + bal['imb'])
                meas_tx = telemetry[if1].get('tx_rate', 0.0)
                
                hyps = sorted(list(set([meas_tx, implied_tx, 0.0])))
                scores = []
                
                for h in hyps:
                    pred_imb = bal['imb'] + curr_tx - h
                    p = math.exp(-abs(pred_imb) / sigma)
                    
                    prior = 1.0
                    if abs(h - meas_tx) > sigma: prior = 0.6 # Penalty for deviating from measurement
                    scores.append(p * prior)
                    
                total_s = sum(scores) + 1e-12
                probs = [s/total_s for s in scores]
                best_idx = scores.index(max(scores))
                
                win_tx = hyps[best_idx]
                conf_tx = probs[best_idx] * math.sqrt(scores[best_idx])
                
                new_tx = (1 - MOMENTUM) * win_tx + MOMENTUM * curr_tx
                batch_updates[(if1, 'tx')] = (new_tx, conf_tx)
                
                # Solve RX (In)
                # Imb = In - Out.
                # My_RX_new = My_RX_old - Imb
                curr_rx = estimates[if1]['rx']
                implied_rx = max(0.0, curr_rx - bal['imb'])
                meas_rx = telemetry[if1].get('rx_rate', 0.0)
                
                hyps = sorted(list(set([meas_rx, implied_rx, 0.0])))
                scores = []
                
                for h in hyps:
                    pred_imb = bal['imb'] - curr_rx + h
                    p = math.exp(-abs(pred_imb) / sigma)
                    
                    prior = 1.0
                    if abs(h - meas_rx) > sigma: prior = 0.6
                    scores.append(p * prior)
                
                total_s = sum(scores) + 1e-12
                probs = [s/total_s for s in scores]
                best_idx = scores.index(max(scores))
                
                win_rx = hyps[best_idx]
                conf_rx = probs[best_idx] * math.sqrt(scores[best_idx])
                
                new_rx = (1 - MOMENTUM) * win_rx + MOMENTUM * curr_rx
                batch_updates[(if1, 'rx')] = (new_rx, conf_rx)
                
        # Apply Batch
        for (iid, metric), (val, conf) in batch_updates.items():
            estimates[iid][metric] = val
            confidences[iid][metric] = conf

    # --- Step 3: Status & Final Assembly ---
    result = {}
    MIN_SIGNIFICANT_FLOW = 0.1
    
    for if_id, data in telemetry.items():
        orig_rx = data.get('rx_rate', 0.0)
        orig_tx = data.get('tx_rate', 0.0)
        orig_status = data.get('interface_status', 'unknown')
        
        rep_rx = estimates[if_id]['rx']
        rep_tx = estimates[if_id]['tx']
        conf_rx = confidences[if_id]['rx']
        conf_tx = confidences[if_id]['tx']
        
        peer_id = data.get('connected_to')
        peer_status = 'unknown'
        if peer_id and peer_id in telemetry:
            peer_status = telemetry[peer_id].get('interface_status', 'unknown')
            
        # Status Logic
        rep_status = orig_status
        conf_status = 1.0
        
        has_traffic = (rep_rx > MIN_SIGNIFICANT_FLOW) or (rep_tx > MIN_SIGNIFICANT_FLOW)
        
        if has_traffic:
            rep_status = 'up'
            if orig_status != 'up':
                conf_status = max(conf_rx, conf_tx)
        elif peer_status == 'down':
            rep_status = 'down'
            if orig_status != 'down':
                conf_status = 0.95
        elif orig_status == 'up' and not has_traffic:
            # Ambiguous / Idle
            rep_status = 'up'
        
        # Consistency
        if rep_status == 'down':
            rep_rx = 0.0
            rep_tx = 0.0
            conf_rx = max(conf_rx, conf_status)
            conf_tx = max(conf_tx, conf_status)
            
        entry = {}
        entry['rx_rate'] = (orig_rx, rep_rx, conf_rx)
        entry['tx_rate'] = (orig_tx, rep_tx, conf_tx)
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
