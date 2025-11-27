# EVOLVE-BLOCK-START
"""
Probabilistic Flow Inference Repair
Treats telemetry repair as a Bayesian hypothesis selection problem.
For each value, generates candidate hypotheses (Self, Peer, Flow, Zero) and scores them
based on Gaussian likelihood against available evidence, weighted by source reliability.
"""
from typing import Dict, Any, Tuple, List
import math

def repair_network_telemetry(telemetry: Dict[str, Dict[str, Any]], 
                             topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:
    
    # --- Configuration ---
    REL_TOL = 0.02
    ABS_TOL = 0.5
    ITERATIONS = 5
    ALPHA = 0.5  # Momentum factor (New estimate = (1-A)*Old + A*New)

    # --- Helper: Likelihood Function ---
    def get_likelihood(candidate: float, evidence: float) -> float:
        if evidence is None: return 0.0
        # Heteroscedastic noise model: Sigma scales with magnitude
        # We use a floor (ABS_TOL) for low values and relative for high values
        sigma = max(ABS_TOL, max(abs(candidate), abs(evidence)) * REL_TOL)
        diff = abs(candidate - evidence)
        # Gaussian Kernel: returns 1.0 at perfect match, decays with distance
        return math.exp(-0.5 * (diff / sigma) ** 2)

    # --- Phase 1: Initialization & Status Normalization ---
    working_state = {}
    
    for if_id, data in telemetry.items():
        s_rx = float(data.get('rx_rate', 0.0))
        s_tx = float(data.get('tx_rate', 0.0))
        s_status = data.get('interface_status', 'unknown')
        
        peer_id = data.get('connected_to')
        has_peer = False
        p_rx, p_tx, p_status = None, None, 'unknown'
        
        if peer_id and peer_id in telemetry:
            has_peer = True
            p_data = telemetry[peer_id]
            p_rx = float(p_data.get('rx_rate', 0.0))
            p_tx = float(p_data.get('tx_rate', 0.0))
            p_status = p_data.get('interface_status', 'unknown')
            
        # Robust Status Logic
        # Traffic is defined as any significant signal (> ABS_TOL) from any source
        traffic_detected = (s_rx > ABS_TOL or s_tx > ABS_TOL or 
                            (p_rx is not None and p_rx > ABS_TOL) or 
                            (p_tx is not None and p_tx > ABS_TOL))
        
        final_status = s_status
        status_conf = 1.0
        
        if s_status == 'down' and traffic_detected:
            final_status = 'up'
            status_conf = 0.95
        elif s_status == 'up' and not traffic_detected and p_status == 'down':
            final_status = 'down'
            status_conf = 0.90
            
        # Initial Estimates (Seeding)
        if final_status == 'down':
            est_rx, est_tx = 0.0, 0.0
        else:
            # Seed: Trust Peer > Self
            est_rx = p_tx if (p_tx is not None) else s_rx
            est_tx = p_rx if (p_rx is not None) else s_tx
            
        working_state[if_id] = {
            's_rx': s_rx, 's_tx': s_tx,
            'p_rx': p_rx, 'p_tx': p_tx,
            'est_rx': est_rx, 'est_tx': est_tx,
            'status': final_status, 'status_conf': status_conf,
            'orig_status': s_status,
            'has_peer': has_peer,
            'r_id': data.get('local_router')
        }

    # --- Phase 2: Iterative Probabilistic Refinement ---
    for _ in range(ITERATIONS):
        # 1. Build Router Conservation Context
        router_stats = {}
        for r_id, if_list in topology.items():
            valid_ifs = [i for i in if_list if i in working_state]
            
            sum_in = sum(working_state[i]['est_rx'] for i in valid_ifs)
            sum_out = sum(working_state[i]['est_tx'] for i in valid_ifs)
            
            imbalance = abs(sum_in - sum_out)
            scale = max(sum_in, sum_out, 1.0)
            
            # Reliability Score (0.0 to 1.0)
            # Decays exponentially based on imbalance ratio.
            # At 5% imbalance (0.05), exp(-1) ≈ 0.36
            reliability = math.exp(- (imbalance / scale) * 20.0)
            
            router_stats[r_id] = {
                'in': sum_in, 'out': sum_out,
                'reliability': reliability
            }
            
        # 2. Update Interface Estimates
        for if_id, d in working_state.items():
            if d['status'] == 'down': 
                continue # Fixed at 0.0

            r_id = d['r_id']
            r_stats = router_stats.get(r_id)
            
            # Helper to perform selection for one counter
            def select_best(s_val, p_val, flow_val, flow_weight):
                candidates = [s_val, 0.0]
                if p_val is not None: candidates.append(p_val)
                if flow_val is not None: candidates.append(flow_val)
                
                best_val = s_val
                best_score = -1.0
                
                for cand in candidates:
                    score = 0.0
                    # Evidence Scoring
                    score += 1.0 * get_likelihood(cand, s_val)      # Self
                    score += 2.0 * get_likelihood(cand, p_val)      # Peer (Stronger)
                    score += flow_weight * get_likelihood(cand, flow_val) # Flow (Variable)
                    
                    # Zero Bias: Favor 0.0 if signals are low (suppress noise)
                    if cand == 0.0 and s_val < 1.0 and (p_val is None or p_val < 1.0):
                        score += 1.5
                    
                    if score > best_score:
                        best_score = score
                        best_val = cand
                return best_val

            # --- RX Update ---
            # Flow Target: RX_i = Total_Out - (Total_In - RX_i)
            flow_rx = None
            flow_w = 0.0
            if r_stats:
                other_in = r_stats['in'] - d['est_rx']
                flow_rx = max(0.0, r_stats['out'] - other_in)
                # Weight: External links (no peer) rely heavily on flow if router is reliable
                base_w = 2.5 if not d['has_peer'] else 1.0
                flow_w = base_w * r_stats['reliability']

            target_rx = select_best(d['s_rx'], d['p_tx'], flow_rx, flow_w)
            d['est_rx'] = (1 - ALPHA) * d['est_rx'] + ALPHA * target_rx
            
            # --- TX Update ---
            # Flow Target: TX_i = Total_In - (Total_Out - TX_i)
            flow_tx = None
            flow_w = 0.0
            if r_stats:
                other_out = r_stats['out'] - d['est_tx']
                flow_tx = max(0.0, r_stats['in'] - other_out)
                base_w = 2.5 if not d['has_peer'] else 1.0
                flow_w = base_w * r_stats['reliability']
                
            target_tx = select_best(d['s_tx'], d['p_rx'], flow_tx, flow_w)
            d['est_tx'] = (1 - ALPHA) * d['est_tx'] + ALPHA * target_tx

    # --- Phase 3: Final Output & Calibration ---
    # Final check of router balance for confidence calibration
    final_stats = {}
    for r_id, if_list in topology.items():
        valid_ifs = [i for i in if_list if i in working_state]
        s_in = sum(working_state[i]['est_rx'] for i in valid_ifs)
        s_out = sum(working_state[i]['est_tx'] for i in valid_ifs)
        mx = max(s_in, s_out, 1.0)
        final_stats[r_id] = abs(s_in - s_out) / mx

    results = {}
    for if_id, d in working_state.items():
        orig = telemetry[if_id]
        res = orig.copy()
        
        if d['status'] == 'down':
            c_rx = 0.95 if d['s_rx'] > ABS_TOL else 1.0
            c_tx = 0.95 if d['s_tx'] > ABS_TOL else 1.0
            res['rx_rate'] = (d['s_rx'], 0.0, c_rx)
            res['tx_rate'] = (d['s_tx'], 0.0, c_tx)
        else:
            # Calibrate Confidence
            def calc_conf(val, s_val, p_val, has_peer, r_id):
                lik_s = get_likelihood(val, s_val)
                lik_p = get_likelihood(val, p_val)
                
                conf = 0.5
                if has_peer:
                    # If we match Peer (Strongest signal), high confidence
                    if lik_p > 0.8: conf = 0.95
                    elif lik_s > 0.8: conf = 0.8 # Match self, mismatch peer?
                    else: conf = 0.7 # Match neither (Flow repair?)
                else:
                    # No peer
                    if lik_s > 0.9: conf = 0.9 # Trust self
                    else: conf = 0.7 # Trust flow
                    
                    # External Link Validation: Adjust based on final router balance
                    # If we blindly trusted flow on a broken router, this metric saves us.
                    if r_id in final_stats:
                        imb = final_stats[r_id]
                        if imb < 0.02: conf = max(conf, 0.95) # Validated by flow
                        elif imb > 0.05: conf = min(conf, 0.6) # Invalidated by bad flow
                return conf

            c_rx = calc_conf(d['est_rx'], d['s_rx'], d['p_tx'], d['has_peer'], d['r_id'])
            c_tx = calc_conf(d['est_tx'], d['s_tx'], d['p_rx'], d['has_peer'], d['r_id'])
            
            res['rx_rate'] = (d['s_rx'], d['est_rx'], c_rx)
            res['tx_rate'] = (d['s_tx'], d['est_tx'], c_tx)

        res['interface_status'] = (d['orig_status'], d['status'], d['status_conf'])
        results[if_id] = res

    return results
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
