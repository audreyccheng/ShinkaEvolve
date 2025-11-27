# EVOLVE-BLOCK-START
"""
annealed_trust_propagation
Implements Global Trust Propagation with Annealed Consensus.
Uses physics-based variance summation for flow target uncertainty and 
progressively anneals (reduces) weight of self-telemetry if it conflicts 
with verified peer signals, allowing strong signals to propagate through 
trusted routers to edge links.
"""
import math
from typing import Dict, Any, Tuple, List

def repair_network_telemetry(telemetry: Dict[str, Dict[str, Any]], 
                             topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:
    
    # --- Configuration ---
    REL_TOL = 0.02
    ABS_TOL = 0.5
    ITERATIONS = 5
    
    # --- Helper Functions ---
    def get_sigma(val: float) -> float:
        """Adaptive standard deviation for Gaussian kernels."""
        return max(ABS_TOL, abs(val) * REL_TOL)

    def gaussian_score(x: float, target: float, sigma: float) -> float:
        """Unnormalized Gaussian Likelihood."""
        if target is None: return 0.0
        diff = abs(x - target)
        return math.exp(-0.5 * (diff / sigma) ** 2)
    
    def is_zero(val: float) -> bool:
        return val < ABS_TOL

    # --- Phase 1: Initialization & Status Repair ---
    state = {}
    
    for if_id, data in telemetry.items():
        s_rx = float(data.get('rx_rate', 0.0))
        s_tx = float(data.get('tx_rate', 0.0))
        s_stat = data.get('interface_status', 'unknown')
        
        peer_id = data.get('connected_to')
        has_peer = False
        p_rx, p_tx, p_stat = 0.0, 0.0, 'unknown'
        
        if peer_id and peer_id in telemetry:
            has_peer = True
            p_data = telemetry[peer_id]
            p_rx = float(p_data.get('rx_rate', 0.0))
            p_tx = float(p_data.get('tx_rate', 0.0))
            p_stat = p_data.get('interface_status', 'unknown')

        # Traffic Analysis
        proven_active = (s_rx > ABS_TOL or s_tx > ABS_TOL or 
                         p_rx > ABS_TOL or p_tx > ABS_TOL)
        
        final_stat = s_stat
        stat_conf = 1.0
        
        # Status Repair Logic
        if s_stat == 'down' and proven_active:
            final_stat = 'up'
            stat_conf = 0.95
        elif s_stat == 'up' and not proven_active:
            if p_stat == 'down':
                final_stat = 'down'
                stat_conf = 0.90
            
        # Initial Value Estimation
        if final_stat == 'down':
            est_rx, est_tx = 0.0, 0.0
        else:
            est_rx = p_tx if has_peer else s_rx
            est_tx = p_rx if has_peer else s_tx
            
        state[if_id] = {
            's_rx': s_rx, 's_tx': s_tx,
            'p_rx': p_rx, 'p_tx': p_tx,
            'est_rx': est_rx, 'est_tx': est_tx,
            'status': final_stat,
            'stat_conf': stat_conf,
            'orig_stat': s_stat,
            'has_peer': has_peer
        }

    # --- Phase 2: Iterative Consensus ---
    
    for iteration in range(ITERATIONS):
        # 1. Router Reliability & Variance Analysis
        router_metrics = {}
        for r_id, if_list in topology.items():
            valid_ifs = [i for i in if_list if i in state]
            if not valid_ifs: continue
            
            sum_in, sum_out = 0.0, 0.0
            sum_agreement = 0.0
            count_peers = 0
            
            # Variance accumulation for flow sigma
            # We want: sigma_in^2 = sum(sigma_i^2) for i in inputs
            var_sum_in = 0.0
            var_sum_out = 0.0
            
            for i in valid_ifs:
                d = state[i]
                sum_in += d['est_rx']
                sum_out += d['est_tx']
                
                s_in = get_sigma(d['est_rx'])
                s_out = get_sigma(d['est_tx'])
                var_sum_in += s_in**2
                var_sum_out += s_out**2
                
                # Agreement (Trust) - Only verify against PEERS
                if d['has_peer']:
                    ag_rx = gaussian_score(d['est_rx'], d['p_tx'], s_in)
                    ag_tx = gaussian_score(d['est_tx'], d['p_rx'], s_out)
                    sum_agreement += (ag_rx + ag_tx) / 2.0
                    count_peers += 1
            
            # Router Trust is based on PEER verification primarily
            # If no peers, we assume neutral/baseline trust (0.5)
            if count_peers > 0:
                router_trust = sum_agreement / count_peers
            else:
                router_trust = 0.5
            
            # Balance Score
            imb = abs(sum_in - sum_out)
            mag = max(sum_in, sum_out, 1.0)
            balance_score = math.exp(- (imb / (mag * 0.05))**2 )
            
            router_metrics[r_id] = {
                'sin': sum_in, 'sout': sum_out,
                'var_in': var_sum_in, 'var_out': var_sum_out,
                'trust': router_trust, 'balance': balance_score
            }
            
        # 2. Update Estimates
        for if_id, d in state.items():
            if d['status'] == 'down': continue
            
            r_id = telemetry[if_id].get('local_router')
            r_info = router_metrics.get(r_id)
            
            def solve_direction(current, s_val, p_val, is_rx):
                f_val = None
                f_weight = 0.0
                f_sigma = get_sigma(current)
                
                w_self = 1.0
                w_peer = 1.2
                
                # Annealing: Reduce Self weight in later iterations if it's likely wrong
                # This prevents "stickiness" to bad local sensors if we have better data
                if iteration >= 2 and d['has_peer']:
                     # If Self is far from Peer, practically ignore Self
                     diff = abs(s_val - p_val)
                     if diff > 3 * get_sigma(p_val):
                         w_self = 0.1
                
                if r_info:
                    # Flow Target & Uncertainty Calculation
                    if is_rx:
                        others = r_info['sin'] - current
                        f_val = max(0.0, r_info['sout'] - others)
                        # Sigma of flow target = sqrt(Var(Others_In) + Var(Total_Out))
                        # Var(Others_In) = Var(Total_In) - Var(Current)
                        v_others = max(0.0, r_info['var_in'] - get_sigma(current)**2)
                        v_target = r_info['var_out'] + v_others
                    else:
                        others = r_info['sout'] - current
                        f_val = max(0.0, r_info['sin'] - others)
                        v_others = max(0.0, r_info['var_out'] - get_sigma(current)**2)
                        v_target = r_info['var_in'] + v_others
                    
                    f_sigma = math.sqrt(v_target)
                    # Widen sigma if balance is bad (conservation assumption weak)
                    f_sigma *= (1.0 + 3.0 * (1.0 - r_info['balance']))
                    
                    # Weighting Strategy
                    # Trust Transfer: If router is highly trusted, flow is king.
                    base_flow_w = 1.5 * r_info['trust'] * (0.2 + 0.8 * r_info['balance'])
                    
                    if r_info['trust'] > 0.85:
                        # Strong Anchor Mode: Enforce flow conservation strictly
                        f_weight = 3.0 * r_info['balance'] 
                    else:
                        f_weight = base_flow_w
                
                # Hypothesis Generation
                candidates = [s_val, 0.0]
                if d['has_peer']: candidates.append(p_val)
                if f_val is not None: candidates.append(f_val)
                
                # Mean injection for noise handling (merging close clusters)
                if d['has_peer']:
                    diff = abs(s_val - p_val)
                    if diff > ABS_TOL and (diff/max(s_val, p_val, 1.0)) < 0.2:
                        candidates.append((s_val + p_val)/2.0)
                
                best_val = current
                best_score = -1.0
                
                unique_cands = set(candidates)
                max_mag = max(unique_cands) if unique_cands else 0.0
                
                for cand in unique_cands:
                    sigma = get_sigma(cand)
                    
                    # Likelihoods
                    l_s = w_self * gaussian_score(cand, s_val, sigma)
                    l_p = w_peer * gaussian_score(cand, p_val, sigma) if d['has_peer'] else 0.0
                    
                    l_f = 0.0
                    if f_val is not None:
                        # Use physics-based flow sigma
                        l_f = f_weight * gaussian_score(cand, f_val, max(sigma, f_sigma))
                    
                    score = l_s + l_p + l_f
                    
                    # Magnitude-Dependent Zero Penalty
                    if is_zero(cand):
                        # Strong penalty if max_mag is high (Phantom Traffic Check)
                        # 0.2 at 5Mbps, less at higher speeds
                        penalty = 0.2 / (1.0 + max_mag / 5.0)
                        score *= penalty
                        
                    if score > best_score:
                        best_score = score
                        best_val = cand
                
                return best_val

            new_rx = solve_direction(d['est_rx'], d['s_rx'], d['p_tx'], True)
            new_tx = solve_direction(d['est_tx'], d['s_tx'], d['p_rx'], False)
            
            # Momentum update
            d['est_rx'] = 0.5 * d['est_rx'] + 0.5 * new_rx
            d['est_tx'] = 0.5 * d['est_tx'] + 0.5 * new_tx

    # --- Phase 3: Final Output & Calibration ---
    results = {}
    
    # Recalculate context for final scoring
    final_metrics = {}
    for r_id, if_list in topology.items():
        valid_ifs = [i for i in if_list if i in state]
        if not valid_ifs: continue
        sum_in = sum(state[i]['est_rx'] for i in valid_ifs)
        sum_out = sum(state[i]['est_tx'] for i in valid_ifs)
        
        sum_ag = 0
        cnt = 0
        var_sum_in, var_sum_out = 0.0, 0.0
        
        for i in valid_ifs:
            d = state[i]
            s_in = get_sigma(d['est_rx'])
            s_out = get_sigma(d['est_tx'])
            var_sum_in += s_in**2
            var_sum_out += s_out**2
            
            if d['has_peer']:
                ag = (gaussian_score(d['est_rx'], d['p_tx'], s_in) + 
                      gaussian_score(d['est_tx'], d['p_rx'], s_out))/2.0
                sum_ag += ag
                cnt += 1
        
        trust = sum_ag / cnt if cnt > 0 else 0.5
        bal = math.exp(-(abs(sum_in-sum_out)/max(sum_in,sum_out,1.0)*20)**2)
        
        final_metrics[r_id] = {
            'sin': sum_in, 'sout': sum_out,
            'var_in': var_sum_in, 'var_out': var_sum_out,
            'trust': trust, 'bal': bal
        }

    for if_id, d in state.items():
        res = telemetry[if_id].copy()
        
        if d['status'] == 'down':
            res['rx_rate'] = (d['s_rx'], 0.0, 1.0 if d['s_rx'] <= ABS_TOL else 0.95)
            res['tx_rate'] = (d['s_tx'], 0.0, 1.0 if d['s_tx'] <= ABS_TOL else 0.95)
        else:
            r_id = telemetry[if_id].get('local_router')
            r_info = final_metrics.get(r_id)
            
            def calc_conf(val, s_val, p_val, is_rx):
                f_val = None
                f_weight = 0.0
                f_sigma = get_sigma(val)
                w_self = 1.0
                
                # Apply same annealing check for consistency
                if d['has_peer'] and abs(s_val - p_val) > 3 * get_sigma(p_val):
                    w_self = 0.1

                if r_info:
                    if is_rx:
                        others_var = max(0.0, r_info['var_in'] - get_sigma(val)**2)
                        v_target = r_info['var_out'] + others_var
                        # Reconstruct target (exclude self from sum)
                        target = r_info['sout'] - (r_info['sin'] - val)
                    else:
                        others_var = max(0.0, r_info['var_out'] - get_sigma(val)**2)
                        v_target = r_info['var_in'] + others_var
                        target = r_info['sin'] - (r_info['sout'] - val)
                    
                    f_val = max(0.0, target)
                    f_sigma = math.sqrt(v_target) * (1.0 + 3.0 * (1.0 - r_info['bal']))
                    
                    if r_info['trust'] > 0.85:
                        f_weight = 3.0 * r_info['bal']
                    else:
                        f_weight = 1.5 * r_info['trust'] * (0.2 + 0.8 * r_info['bal'])

                # Mass Calculation
                hyps = {val, s_val, 0.0}
                if d['has_peer']: hyps.add(p_val)
                if f_val is not None: hyps.add(f_val)
                if d['has_peer']: hyps.add((s_val + p_val)/2.0)
                
                max_mag = max(hyps)
                sigma_win = get_sigma(val)
                
                cluster_mass = 0.0
                total_mass = 0.0
                
                for h in hyps:
                    sigma = get_sigma(h)
                    l_s = w_self * gaussian_score(h, s_val, sigma)
                    l_p = 1.2 * gaussian_score(h, p_val, sigma) if d['has_peer'] else 0.0
                    l_f = 0.0
                    if f_val is not None:
                        l_f = f_weight * gaussian_score(h, f_val, max(sigma, f_sigma))
                    
                    score = l_s + l_p + l_f
                    if is_zero(h):
                        score *= 0.2 / (1.0 + max_mag / 5.0)
                    
                    total_mass += score
                    if abs(h - val) <= sigma_win:
                        cluster_mass += score
                        
                if total_mass < 1e-9: return 0.5
                prob = cluster_mass / total_mass
                
                # --- Calibration Adjustments ---
                
                # 1. Fit Quality (Absolute Score)
                # Check if winner score is reasonably high compared to theoretical max
                max_possible = w_self + (1.2 if d['has_peer'] else 0.0) + (f_weight if f_val is not None else 0.0)
                
                l_s = w_self * gaussian_score(val, s_val, sigma_win)
                l_p = 1.2 * gaussian_score(val, p_val, sigma_win) if d['has_peer'] else 0.0
                l_f = f_weight * gaussian_score(val, f_val, max(sigma_win, f_sigma)) if f_val is not None else 0.0
                winner_score = l_s + l_p + l_f
                
                fit_ratio = winner_score / max(1.0, max_possible)
                if fit_ratio < 0.5:
                    prob *= (fit_ratio * 2.0)
                    
                # 2. Support Logic
                # Did we just pick Self or 0.0 without external backup?
                peer_support = d['has_peer'] and abs(val - p_val) <= sigma_win
                flow_support = f_val is not None and abs(val - f_val) <= max(sigma_win, f_sigma)
                
                if not peer_support and not flow_support:
                    # Isolated decision -> cap confidence
                    prob = min(prob, 0.70)
                
                # Edge Case: Supported ONLY by Flow (no Peer)
                if not d['has_peer'] and flow_support:
                    # Trust flow only if router is trusted globally
                    if r_info['trust'] < 0.7:
                        prob = min(prob, 0.6 + 0.4 * r_info['trust'])
                
                return max(0.5, min(1.0, prob))

            conf_rx = calc_conf(d['est_rx'], d['s_rx'], d['p_tx'], True)
            conf_tx = calc_conf(d['est_tx'], d['s_tx'], d['p_rx'], False)
            
            res['rx_rate'] = (d['s_rx'], d['est_rx'], conf_rx)
            res['tx_rate'] = (d['s_tx'], d['est_tx'], conf_tx)
            
        res['interface_status'] = (d['orig_stat'], d['status'], d['stat_conf'])
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

