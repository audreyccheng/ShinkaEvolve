# EVOLVE-BLOCK-START
from typing import Dict, Any, Tuple, List
import math

def repair_network_telemetry(telemetry: Dict[str, Dict[str, Any]], 
                             topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:
    """
    Repairs network telemetry using a Momentum-based Bayesian Flow Optimizer.
    
    Key Improvements:
    1.  **Hypothesis Expansion**: Considers 0.0 (Null) and Implied (Conservation-derived) values 
        alongside measurements to handle phantom traffic and external link errors.
    2.  **Soft Updates**: Uses momentum/weighted averaging during iterations to stabilize convergence.
    3.  **Calibrated Confidence**: Confidence = Probability(Winner) * GoodnessOfFit(Winner).
        This penalizes "least bad" choices that still violate conservation.
    4.  **Non-Linear Noise Model**: Tolerances scale with sqrt(flow) to handle low-flow noise better.
    """
    
    # --- Configuration ---
    SYMMETRY_TOLERANCE = 0.02
    MIN_SIGMA = 1.0           # Minimum noise tolerance (Mbps)
    FLOW_SIGMA_COEFF = 0.03   # 3% Linear noise component
    ITERATIONS = 10           # Iterations for soft update convergence
    LEARNING_RATE = 0.5       # Momentum factor for updates
    
    # --- 1. Topology Mapping ---
    if_to_router = {}
    router_to_ifs = topology
    for rid, ifs in topology.items():
        for iid in ifs:
            if_to_router[iid] = rid
            
    # --- 2. Initial State & Pre-processing ---
    # estimates[if_id] = {'rx': val, 'tx': val}
    estimates = {}
    
    # Identify Links
    links = []
    processed = set()
    
    for if_id, data in telemetry.items():
        if if_id in processed: continue
        
        peer = data.get('connected_to')
        if peer and peer in telemetry:
            links.append({
                'type': 'internal',
                'if1': if_id,
                'if2': peer
            })
            processed.add(if_id)
            processed.add(peer)
        else:
            links.append({
                'type': 'external',
                'if1': if_id
            })
            processed.add(if_id)
            
    # Initialize estimates with raw data
    for if_id, data in telemetry.items():
        estimates[if_id] = {
            'rx': float(data.get('rx_rate', 0.0)),
            'tx': float(data.get('tx_rate', 0.0))
        }

    # --- 3. Symmetry Check & Hypothesis Setup ---
    # Identify which flows need optimization vs which are solid
    optimization_targets = [] 
    
    for link in links:
        if link['type'] == 'internal':
            if1, if2 = link['if1'], link['if2']
            d1, d2 = telemetry[if1], telemetry[if2]
            
            # --- Forward: IF1 TX -> IF2 RX ---
            tx1 = d1.get('tx_rate', 0.0)
            rx2 = d2.get('rx_rate', 0.0)
            diff_fwd = abs(tx1 - rx2)
            denom_fwd = max(tx1, rx2, 1.0)
            
            if diff_fwd / denom_fwd <= SYMMETRY_TOLERANCE:
                # Consistent: Harden immediately
                avg = (tx1 + rx2) / 2.0
                estimates[if1]['tx'] = avg
                estimates[if2]['rx'] = avg
            else:
                # Suspect: Add to optimization
                optimization_targets.append({
                    'type': 'internal_flow',
                    'src': if1, 'dst': if2,
                    'metric_src': 'tx', 'metric_dst': 'rx',
                    'candidates': [tx1, rx2, 0.0] # Candidates: Local, Peer, Null
                })

            # --- Backward: IF2 TX -> IF1 RX ---
            tx2 = d2.get('tx_rate', 0.0)
            rx1 = d1.get('rx_rate', 0.0)
            diff_bwd = abs(tx2 - rx1)
            denom_bwd = max(tx2, rx1, 1.0)
            
            if diff_bwd / denom_bwd <= SYMMETRY_TOLERANCE:
                avg = (tx2 + rx1) / 2.0
                estimates[if2]['tx'] = avg
                estimates[if1]['rx'] = avg
            else:
                optimization_targets.append({
                    'type': 'internal_flow',
                    'src': if2, 'dst': if1,
                    'metric_src': 'tx', 'metric_dst': 'rx',
                    'candidates': [tx2, rx1, 0.0]
                })
                
        else:
            # --- External Links ---
            # Treat external measurements as hypotheses to be validated by conservation
            if1 = link['if1']
            d1 = telemetry[if1]
            
            # Add targets for both TX and RX
            optimization_targets.append({
                'type': 'external_flow',
                'if_id': if1, 'metric': 'tx',
                'candidates': [d1.get('tx_rate', 0.0), 0.0] # Implied added dynamically
            })
            optimization_targets.append({
                'type': 'external_flow',
                'if_id': if1, 'metric': 'rx',
                'candidates': [d1.get('rx_rate', 0.0), 0.0]
            })

    # --- 4. Iterative Optimization ---
    
    def get_router_stats(rid):
        """Returns (net_imbalance, max_flow_on_router)"""
        if rid not in router_to_ifs: return 0.0, 1.0
        in_sum = 0.0
        out_sum = 0.0
        max_f = 0.0
        for iid in router_to_ifs[rid]:
            r = estimates[iid]['rx']
            t = estimates[iid]['tx']
            in_sum += r
            out_sum += t
            max_f = max(max_f, r, t)
        return (in_sum - out_sum), max(max_f, 1.0)

    for it in range(ITERATIONS):
        updates = []
        
        for target in optimization_targets:
            if target['type'] == 'internal_flow':
                src, dst = target['src'], target['dst']
                ms, md = target['metric_src'], target['metric_dst']
                rid_src = if_to_router.get(src)
                rid_dst = if_to_router.get(dst)
                
                current_val = estimates[src][ms]
                cands = target['candidates']
                scores = []
                
                for val in cands:
                    # Apply Candidate
                    estimates[src][ms] = val
                    estimates[dst][md] = val
                    
                    # Eval Imbalance
                    imb_src, max_src = get_router_stats(rid_src)
                    imb_dst, max_dst = get_router_stats(rid_dst)
                    
                    # Non-linear Sigma: sqrt term dominates at low flow, linear at high
                    sig_src = max(math.sqrt(max_src), max_src * FLOW_SIGMA_COEFF, MIN_SIGMA)
                    sig_dst = max(math.sqrt(max_dst), max_dst * FLOW_SIGMA_COEFF, MIN_SIGMA)
                    
                    # Likelihood
                    l_src = math.exp(-abs(imb_src) / sig_src) if rid_src else 1.0
                    l_dst = math.exp(-abs(imb_dst) / sig_dst) if rid_dst else 1.0
                    scores.append(l_src * l_dst)
                
                # Restore
                estimates[src][ms] = current_val
                estimates[dst][md] = current_val
                
                # Soft Update
                total_s = sum(scores) + 1e-12
                probs = [s / total_s for s in scores]
                target_val = sum(c * p for c, p in zip(cands, probs))
                
                new_val = current_val * (1 - LEARNING_RATE) + target_val * LEARNING_RATE
                updates.append(((src, ms), new_val))
                updates.append(((dst, md), new_val))
                
            elif target['type'] == 'external_flow':
                iid = target['if_id']
                metric = target['metric']
                rid = if_to_router.get(iid)
                current_val = estimates[iid][metric]
                
                # Dynamic Hypothesis: Implied Value
                # What value would satisfy conservation given other flows?
                estimates[iid][metric] = 0.0
                net_imb, max_f = get_router_stats(rid)
                # net_imb = In - Out (excluding self)
                # If RX (In): (Others_In + Val) - Others_Out = 0 -> Val = Out - Others_In = -net_imb
                # If TX (Out): Others_In - (Others_Out + Val) = 0 -> Val = In - Others_Out = net_imb
                implied = -net_imb if metric == 'rx' else net_imb
                implied = max(0.0, implied)
                estimates[iid][metric] = current_val # Restore
                
                cands = target['candidates'] + [implied]
                scores = []
                
                for val in cands:
                    estimates[iid][metric] = val
                    imb, max_f = get_router_stats(rid)
                    sig = max(math.sqrt(max_f), max_f * FLOW_SIGMA_COEFF, MIN_SIGMA)
                    scores.append(math.exp(-abs(imb) / sig) if rid else 1.0)
                    
                estimates[iid][metric] = current_val
                
                total_s = sum(scores) + 1e-12
                probs = [s / total_s for s in scores]
                target_val = sum(c * p for c, p in zip(cands, probs))
                
                new_val = current_val * (1 - LEARNING_RATE) + target_val * LEARNING_RATE
                updates.append(((iid, metric), new_val))
                
        for (key, val) in updates:
            estimates[key[0]][key[1]] = val
            
    # --- 5. Final Selection & Calibration ---
    final_confidences = {}
    
    for target in optimization_targets:
        if target['type'] == 'internal_flow':
            src, dst = target['src'], target['dst']
            ms, md = target['metric_src'], target['metric_dst']
            rid_src = if_to_router.get(src)
            rid_dst = if_to_router.get(dst)
            cands = target['candidates']
            
            best_score = -1.0
            best_val = 0.0
            best_raw = 0.0
            total_s = 0.0
            
            scores = []
            
            for val in cands:
                estimates[src][ms] = val
                estimates[dst][md] = val
                imb_src, max_src = get_router_stats(rid_src)
                imb_dst, max_dst = get_router_stats(rid_dst)
                
                sig_src = max(math.sqrt(max_src), max_src * FLOW_SIGMA_COEFF, MIN_SIGMA)
                sig_dst = max(math.sqrt(max_dst), max_dst * FLOW_SIGMA_COEFF, MIN_SIGMA)
                
                s1 = math.exp(-abs(imb_src) / sig_src) if rid_src else 1.0
                s2 = math.exp(-abs(imb_dst) / sig_dst) if rid_dst else 1.0
                
                combined = s1 * s2
                scores.append(combined)
                total_s += combined
                
                if combined > best_score:
                    best_score = combined
                    best_val = val
                    best_raw = min(s1, s2) # Conservative fit quality
            
            # Probability of winning
            win_prob = best_score / (total_s + 1e-12)
            # Calibrated Conf = Prob * Fit
            conf = win_prob * best_raw
            
            estimates[src][ms] = best_val
            estimates[dst][md] = best_val
            final_confidences[(src, ms)] = conf
            final_confidences[(dst, md)] = conf
            
        elif target['type'] == 'external_flow':
            iid = target['if_id']
            metric = target['metric']
            rid = if_to_router.get(iid)
            
            estimates[iid][metric] = 0.0
            net_imb, max_f = get_router_stats(rid)
            implied = max(0.0, -net_imb if metric == 'rx' else net_imb)
            
            cands = target['candidates'] + [implied]
            scores = []
            
            best_score = -1.0
            best_val = 0.0
            best_raw = 0.0
            total_s = 0.0
            
            for val in cands:
                estimates[iid][metric] = val
                imb, max_f = get_router_stats(rid)
                sig = max(math.sqrt(max_f), max_f * FLOW_SIGMA_COEFF, MIN_SIGMA)
                s = math.exp(-abs(imb) / sig) if rid else 1.0
                
                scores.append(s)
                total_s += s
                
                if s > best_score:
                    best_score = s
                    best_val = val
                    best_raw = s
            
            win_prob = best_score / (total_s + 1e-12)
            conf = win_prob * best_raw
            
            estimates[iid][metric] = best_val
            final_confidences[(iid, metric)] = conf

    # --- 6. Status & Output ---
    result = {}
    
    for if_id, data in telemetry.items():
        orig_rx = data.get('rx_rate', 0.0)
        orig_tx = data.get('tx_rate', 0.0)
        orig_status = data.get('interface_status', 'unknown')
        
        est_rx = estimates[if_id]['rx']
        est_tx = estimates[if_id]['tx']
        conf_rx = final_confidences.get((if_id, 'rx'), 0.95)
        conf_tx = final_confidences.get((if_id, 'tx'), 0.95)
        
        peer_id = data.get('connected_to')
        peer_status = 'unknown'
        if peer_id and peer_id in telemetry:
            peer_status = telemetry[peer_id].get('interface_status', 'unknown')
            
        # Infer Status
        has_traffic = (est_rx > 0.5) or (est_tx > 0.5)
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
        else:
            rep_status = orig_status
            if orig_status == 'up' and peer_status == 'down':
                rep_status = 'down'
                conf_status = 0.8
                
        if rep_status == 'down':
            est_rx = 0.0
            est_tx = 0.0
            conf_rx = max(conf_rx, 0.95)
            conf_tx = max(conf_tx, 0.95)
            
        entry = {}
        entry['rx_rate'] = (orig_rx, est_rx, conf_rx)
        entry['tx_rate'] = (orig_tx, est_tx, conf_tx)
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

