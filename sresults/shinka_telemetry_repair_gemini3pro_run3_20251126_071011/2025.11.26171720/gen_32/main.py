# EVOLVE-BLOCK-START
from typing import Dict, Any, Tuple, List
import math

def repair_network_telemetry(telemetry: Dict[str, Dict[str, Any]], 
                             topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:
    """
    Repairs network telemetry using a Momentum-Based Bayesian Consensus algorithm.
    
    Key Innovations:
    1. Square-Root Noise Model: Tolerances scale with sqrt(flow) to match Poisson statistics.
    2. Implied & Zero Hypotheses: Explicitly tests 'Link Down' and 'Conservation-Implied' values.
    3. Momentum Updates: Uses soft updates (alpha=0.5) to stably converge on flow conservation.
    4. Absolute Confidence Calibration: Confidence scales with goodness-of-fit, not just relative probability.
    """
    
    # --- Configuration ---
    SYMMETRY_TOLERANCE = 0.02   # 2% difference allowed for hardening
    MIN_SIG_FLOW = 0.1          # Minimum flow to consider interface UP
    ITERATIONS = 10             # Solver iterations
    LEARNING_RATE = 0.5         # Alpha for momentum updates (0.0 = no change, 1.0 = immediate overwrite)
    SIGMA_K = 0.5               # Noise scaling constant: sigma = K * sqrt(flow)
                                # At 100Mbps, sigma=5 (5%). At 10Gbps, sigma=50 (0.5%).
    
    # --- Helper Structures ---
    if_to_router = {}
    for r_id, if_list in topology.items():
        for i_id in if_list:
            if_to_router[i_id] = r_id
            
    # --- Step 1: Link Identification & Hardening ---
    # Classify links into 'Internal' (pairs) and 'External' (singletons).
    # Check symmetry for Internal links.
    
    # estimates: {if_id: {'rx': val, 'tx': val}}
    estimates = {}
    # confidences: {if_id: {'rx': conf, 'tx': conf}}
    confidences = {}
    # processed_links: set of keys to avoid double processing
    processed_links = set()
    
    # Work items for the solver
    # List of dicts describing what to solve
    suspect_items = []
    
    # Initialize all interfaces first
    for if_id, data in telemetry.items():
        estimates[if_id] = {
            'rx': float(data.get('rx_rate', 0.0)), 
            'tx': float(data.get('tx_rate', 0.0))
        }
        # Default low confidence
        confidences[if_id] = {'rx': 0.5, 'tx': 0.5}

    # Process Links
    for if_id, data in telemetry.items():
        if if_id in processed_links: continue
        
        peer_id = data.get('connected_to')
        
        if peer_id and peer_id in telemetry:
            # Internal Link
            processed_links.add(if_id)
            processed_links.add(peer_id)
            
            # Check Symmetry: My TX vs Peer RX
            tx_val = estimates[if_id]['tx']
            rx_val = estimates[peer_id]['rx']
            diff = abs(tx_val - rx_val)
            denom = max(tx_val, rx_val, 1.0)
            
            if diff / denom < SYMMETRY_TOLERANCE:
                # Consistent: Harden
                consensus = (tx_val + rx_val) / 2.0
                estimates[if_id]['tx'] = consensus
                estimates[peer_id]['rx'] = consensus
                confidences[if_id]['tx'] = 0.95
                confidences[peer_id]['rx'] = 0.95
                # Do NOT add to suspect_items (anchored)
            else:
                # Inconsistent: Suspect
                suspect_items.append({
                    'type': 'internal_link_dir',
                    'src': if_id, 'dst': peer_id,
                    'val_src': tx_val, 'val_dst': rx_val
                })
                
            # Check Symmetry: Peer TX vs My RX
            tx_val = estimates[peer_id]['tx']
            rx_val = estimates[if_id]['rx']
            diff = abs(tx_val - rx_val)
            denom = max(tx_val, rx_val, 1.0)
            
            if diff / denom < SYMMETRY_TOLERANCE:
                consensus = (tx_val + rx_val) / 2.0
                estimates[peer_id]['tx'] = consensus
                estimates[if_id]['rx'] = consensus
                confidences[peer_id]['tx'] = 0.95
                confidences[if_id]['rx'] = 0.95
            else:
                suspect_items.append({
                    'type': 'internal_link_dir',
                    'src': peer_id, 'dst': if_id,
                    'val_src': tx_val, 'val_dst': rx_val
                })
        else:
            # External Link
            processed_links.add(if_id)
            # Add both RX and TX as suspect/optimizable
            # We initialize confidence slightly higher as there is no peer to contradict,
            # but we allow conservation to correct it.
            confidences[if_id]['rx'] = 0.8
            confidences[if_id]['tx'] = 0.8
            
            suspect_items.append({
                'type': 'external_rx',
                'if': if_id,
                'val': estimates[if_id]['rx']
            })
            suspect_items.append({
                'type': 'external_tx',
                'if': if_id,
                'val': estimates[if_id]['tx']
            })

    # --- Step 2: Iterative Bayesian Solver with Momentum ---
    
    def get_router_stats(rid):
        """Returns (net_imbalance, total_flow) for a router."""
        if not rid: return 0.0, 1.0
        net = 0.0
        total = 0.0
        for iid in topology.get(rid, []):
            r = estimates[iid]['rx']
            t = estimates[iid]['tx']
            net += (r - t)
            total += (r + t)
        return net, max(total/2.0, 1.0) # Avg flow per direction

    def get_implied_value(rid, if_id, metric):
        """
        Calculates the value for interface[metric] that would perfectly balance the router,
        assuming all other interfaces are fixed.
        """
        if not rid: return 0.0
        
        # Calculate net flow of ALL OTHER interfaces
        other_net = 0.0
        for iid in topology.get(rid, []):
            if iid == if_id: continue
            other_net += (estimates[iid]['rx'] - estimates[iid]['tx'])
            
        # Balance equation: other_net + My_Net = 0
        # If I am RX: My_Net = +RX.  => RX = -other_net
        # If I am TX: My_Net = -TX.  => -TX = -other_net => TX = other_net
        
        if metric == 'rx':
            return max(0.0, -other_net)
        else:
            return max(0.0, other_net)

    for _ in range(ITERATIONS):
        updates = []
        
        for item in suspect_items:
            # Generate Hypotheses
            candidates = [] # List of (value, type)
            
            if item['type'] == 'internal_link_dir':
                # Candidates: Src measurement, Dst measurement, Zero
                candidates.append((item['val_src'], 'src'))
                candidates.append((item['val_dst'], 'dst'))
                candidates.append((0.0, 'zero'))
                
                src_if, dst_if = item['src'], item['dst']
                r_src = if_to_router.get(src_if)
                r_dst = if_to_router.get(dst_if)
                
                # Setup targets for update
                targets = [(src_if, 'tx'), (dst_if, 'rx')]
                
                # Context Management
                old_src = estimates[src_if]['tx']
                old_dst = estimates[dst_if]['rx']
                
                scored_hyps = []
                for val, _ in candidates:
                    # Apply
                    estimates[src_if]['tx'] = val
                    estimates[dst_if]['rx'] = val
                    
                    # Score Src
                    imb_s, flow_s = get_router_stats(r_src)
                    sigma_s = max(math.sqrt(flow_s) * SIGMA_K, 1.0)
                    prob_s = math.exp(-abs(imb_s)/sigma_s)
                    
                    # Score Dst
                    imb_d, flow_d = get_router_stats(r_dst)
                    sigma_d = max(math.sqrt(flow_d) * SIGMA_K, 1.0)
                    prob_d = math.exp(-abs(imb_d)/sigma_d)
                    
                    # Combined
                    scored_hyps.append((val, prob_s * prob_d))
                    
                # Restore
                estimates[src_if]['tx'] = old_src
                estimates[dst_if]['rx'] = old_dst
                
            else: # external_rx or external_tx
                iid = item['if']
                metric = 'rx' if item['type'] == 'external_rx' else 'tx'
                rid = if_to_router.get(iid)
                
                # Candidates: Local, Zero, Implied
                candidates.append((item['val'], 'local'))
                candidates.append((0.0, 'zero'))
                if rid:
                    implied = get_implied_value(rid, iid, metric)
                    candidates.append((implied, 'implied'))
                
                targets = [(iid, metric)]
                old_val = estimates[iid][metric]
                
                scored_hyps = []
                for val, _ in candidates:
                    estimates[iid][metric] = val
                    imb, flow = get_router_stats(rid)
                    sigma = max(math.sqrt(flow) * SIGMA_K, 1.0)
                    prob = math.exp(-abs(imb)/sigma)
                    scored_hyps.append((val, prob))
                    
                estimates[iid][metric] = old_val

            # Selection Strategy
            # Find best hypothesis
            best_val, best_score = max(scored_hyps, key=lambda x: x[1])
            
            # Calibration
            total_score = sum(s for v,s in scored_hyps) + 1e-12
            rel_prob = best_score / total_score
            
            # Confidence = Relative Probability * Absolute Goodness
            # If the best solution still leaves the router imbalanced (low best_score),
            # confidence drops.
            conf = rel_prob * best_score
            conf = max(0.01, min(0.99, conf))
            
            updates.append({
                'targets': targets,
                'val': best_val,
                'conf': conf
            })
            
        # Apply Updates (Momentum)
        for u in updates:
            target_val = u['val']
            conf = u['conf']
            for (iid, metric) in u['targets']:
                curr = estimates[iid][metric]
                # Update rule: New = (LR * Target) + ((1-LR) * Old)
                estimates[iid][metric] = (LEARNING_RATE * target_val) + ((1 - LEARNING_RATE) * curr)
                confidences[iid][metric] = conf

    # --- Step 3: Result Assembly & Status Inference ---
    result = {}
    
    for if_id, data in telemetry.items():
        orig_rx = data.get('rx_rate', 0.0)
        orig_tx = data.get('tx_rate', 0.0)
        orig_status = data.get('interface_status', 'unknown')
        
        rep_rx = estimates[if_id]['rx']
        conf_rx = confidences[if_id]['rx']
        
        rep_tx = estimates[if_id]['tx']
        conf_tx = confidences[if_id]['tx']
        
        # Determine Status
        peer_id = data.get('connected_to')
        peer_status = 'unknown'
        if peer_id and peer_id in telemetry:
            peer_status = telemetry[peer_id].get('interface_status', 'unknown')
            
        has_traffic = (rep_rx > MIN_SIG_FLOW) or (rep_tx > MIN_SIG_FLOW)
        
        rep_status = orig_status
        conf_status = 1.0
        
        if has_traffic:
            rep_status = 'up'
            if orig_status != 'up':
                # If we overturn status, our confidence is limited by flow confidence
                conf_status = (conf_rx + conf_tx) / 2.0
        elif peer_status == 'down':
            rep_status = 'down'
            if orig_status != 'down':
                conf_status = 0.95
        elif orig_status == 'up':
            # No traffic, Peer UP or Unknown. Likely Idle.
            rep_status = 'up'
        else:
            # Original DOWN, no traffic, peer not DOWN. Trust original.
            rep_status = 'down'
            
        # Consistency Check
        if rep_status == 'down':
            rep_rx = 0.0
            rep_tx = 0.0
            # If we are sure it's down, we are sure rate is 0
            conf_rx = max(conf_rx, conf_status)
            conf_tx = max(conf_tx, conf_status)
            
        # Build Entry
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