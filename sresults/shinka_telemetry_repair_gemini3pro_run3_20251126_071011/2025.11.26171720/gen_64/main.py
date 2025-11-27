# EVOLVE-BLOCK-START
"""
Bayesian Hypothesis Competition Telemetry Repair
Implements a probabilistic solver that scores discrete hypotheses (Self, Peer, Flow, Zero)
and calibrates confidence based on the margin of victory between the best and second-best explanations.
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
        """Returns standard deviation for Gaussian scoring based on value magnitude."""
        return max(ABS_TOL, abs(val) * REL_TOL)

    def gaussian_sim(x: float, mu: float, sigma: float) -> float:
        """Calculates unnormalized Gaussian likelihood."""
        if mu is None: return 0.0
        diff = abs(x - mu)
        return math.exp(-0.5 * (diff / sigma) ** 2)

    def looks_like_zero(val: float) -> bool:
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

        # Status Logic
        traffic = (s_rx > ABS_TOL or s_tx > ABS_TOL or p_rx > ABS_TOL or p_tx > ABS_TOL)
        final_stat = s_stat
        stat_conf = 1.0
        
        if s_stat == 'down' and traffic:
            final_stat = 'up'
            stat_conf = 0.95
        elif s_stat == 'up' and not traffic and p_stat == 'down':
            final_stat = 'down'
            stat_conf = 0.90
            
        # Initial Value Estimation (Prefer Peer)
        if final_stat == 'down':
            est_rx, est_tx = 0.0, 0.0
        else:
            est_rx = p_tx if has_peer else s_rx
            est_tx = p_rx if has_peer else s_tx
            
        # Initial Agreement Score (used to determine if this interface is an "Anchor")
        agree_rx = 0.5
        agree_tx = 0.5
        if has_peer:
             sigma_rx = get_sigma(max(s_rx, p_tx))
             agree_rx = gaussian_sim(s_rx, p_tx, sigma_rx)
             sigma_tx = get_sigma(max(s_tx, p_rx))
             agree_tx = gaussian_sim(s_tx, p_rx, sigma_tx)
        
        state[if_id] = {
            's_rx': s_rx, 's_tx': s_tx,
            'p_rx': p_rx, 'p_tx': p_tx,
            'est_rx': est_rx, 'est_tx': est_tx,
            'status': final_stat,
            'status_conf': stat_conf,
            'orig_stat': s_stat,
            'has_peer': has_peer,
            'agree_score': (agree_rx + agree_tx) / 2.0
        }

    # --- Phase 2: Iterative Bayesian Solver ---
    
    for iteration in range(ITERATIONS):
        # 1. Router Analysis (Compute "Anchor Scores")
        router_stats = {}
        for r_id, if_list in topology.items():
            valid_ifs = [i for i in if_list if i in state]
            if not valid_ifs: continue
            
            sum_in = 0.0
            sum_out = 0.0
            total_agreement = 0.0
            
            for i in valid_ifs:
                sum_in += state[i]['est_rx']
                sum_out += state[i]['est_tx']
                total_agreement += state[i]['agree_score']
                
            # Anchor Score: Average agreement of interfaces on this router.
            # High score means the router's environment is trustworthy.
            anchor_score = total_agreement / len(valid_ifs)
            
            # Balance Score: How well current estimates balance
            imbalance = abs(sum_in - sum_out)
            magnitude = max(sum_in, sum_out, 1.0)
            balance_score = math.exp(- (imbalance / (magnitude * 0.05))**2 )
            
            router_stats[r_id] = {
                'sum_in': sum_in,
                'sum_out': sum_out,
                'anchor': anchor_score,
                'balance': balance_score
            }
            
        # 2. Update Estimates
        for if_id, d in state.items():
            if d['status'] == 'down':
                continue # Fixed at 0
            
            r_id = telemetry[if_id].get('local_router')
            r_info = router_stats.get(r_id)
            
            # Helper to solve for one direction
            def solve_direction(current_est, s_val, p_val, is_rx):
                # Calculate Flow Target
                flow_val = None
                flow_weight = 0.0
                
                if r_info:
                    # Target = what the rest of the router demands
                    # Sum_Others = Total - Me
                    if is_rx:
                        other_in = r_info['sum_in'] - current_est
                        flow_val = max(0.0, r_info['sum_out'] - other_in)
                    else:
                        other_out = r_info['sum_out'] - current_est
                        flow_val = max(0.0, r_info['sum_in'] - other_out)
                    
                    # Trust Flow?
                    # Weight = 2.0 (High) if Anchor & Balance are perfect.
                    # This allows Flow to override Self+Peer if the router context is very strong.
                    flow_weight = 2.5 * r_info['anchor'] * (0.4 + 0.6 * r_info['balance'])
                
                # Generate Hypotheses
                hyps = []
                hyps.append(s_val) # H_self
                if d['has_peer']: hyps.append(p_val) # H_peer
                if flow_val is not None: hyps.append(flow_val) # H_flow
                hyps.append(0.0) # H_zero
                
                # H_mean: Add average if Self/Peer are close (noise model)
                if d['has_peer']:
                     diff = abs(s_val - p_val)
                     mx = max(s_val, p_val, 1.0)
                     if diff / mx < 0.2: # 20% divergence threshold
                         hyps.append((s_val + p_val) / 2.0)

                candidates = sorted(list(set(hyps)))
                
                best_val = current_est
                best_score = -1.0
                
                for c in candidates:
                    sigma = get_sigma(c)
                    
                    # Likelihoods
                    s_lik = gaussian_sim(c, s_val, sigma)
                    p_lik = gaussian_sim(c, p_val, sigma) if d['has_peer'] else 0.0
                    f_lik = gaussian_sim(c, flow_val, sigma) if flow_val is not None else 0.0
                    
                    # Score = Weighted Sum of Likelihoods
                    # Peer=1.0, Self=0.8, Flow=Variable
                    score = (0.8 * s_lik) + (1.0 * p_lik) + (flow_weight * f_lik)
                    
                    # Prior: Penalize Zero if status is UP
                    if looks_like_zero(c):
                        score *= 0.2 
                        
                    if score > best_score:
                        best_score = score
                        best_val = c
                
                return best_val

            # Solve RX
            new_rx = solve_direction(d['est_rx'], d['s_rx'], d['p_tx'], True)
            d['est_rx'] = 0.5 * d['est_rx'] + 0.5 * new_rx
            
            # Solve TX
            new_tx = solve_direction(d['est_tx'], d['s_tx'], d['p_rx'], False)
            d['est_tx'] = 0.5 * d['est_tx'] + 0.5 * new_tx

            # Update Agreement Score for next iteration
            new_agree_rx = 0.0
            new_agree_tx = 0.0
            if d['has_peer']:
                # Check convergence against Peer Raw
                new_agree_rx = gaussian_sim(d['est_rx'], d['p_tx'], get_sigma(d['est_rx']))
                new_agree_tx = gaussian_sim(d['est_tx'], d['p_rx'], get_sigma(d['est_tx']))
            else:
                new_agree_rx = gaussian_sim(d['est_rx'], d['s_rx'], get_sigma(d['est_rx']))
                new_agree_tx = gaussian_sim(d['est_tx'], d['s_tx'], get_sigma(d['est_tx']))
            
            d['agree_score'] = (new_agree_rx + new_agree_tx) / 2.0

    # --- Phase 3: Final Output & Confidence Calibration ---
    results = {}
    
    # Final Router Context
    final_router_stats = {}
    for r_id, if_list in topology.items():
        valid_ifs = [i for i in if_list if i in state]
        s_in = sum(state[i]['est_rx'] for i in valid_ifs)
        s_out = sum(state[i]['est_tx'] for i in valid_ifs)
        tot_agree = sum(state[i]['agree_score'] for i in valid_ifs)
        anchor = tot_agree / len(valid_ifs) if valid_ifs else 0.0
        imb = abs(s_in - s_out)
        bal = math.exp(-(imb/max(s_in, s_out, 1.0)*20.0)**2) 
        final_router_stats[r_id] = {'sin': s_in, 'sout': s_out, 'anchor': anchor, 'bal': bal}
    
    for if_id, d in state.items():
        res = telemetry[if_id].copy()
        
        if d['status'] == 'down':
             conf_rx = 0.95 if d['s_rx'] > ABS_TOL else 1.0
             conf_tx = 0.95 if d['s_tx'] > ABS_TOL else 1.0
             res['rx_rate'] = (d['s_rx'], 0.0, conf_rx)
             res['tx_rate'] = (d['s_tx'], 0.0, conf_tx)
        else:
            r_id = telemetry[if_id].get('local_router')
            r_info = final_router_stats.get(r_id)
            
            def calc_conf(val, s_val, p_val, is_rx):
                # Reconstruct Targets
                f_val = None
                f_w = 0.0
                if r_info:
                    # Note: Must subtract FINAL estimate to get the network invariant
                    if is_rx:
                        target = r_info['sout'] - (r_info['sin'] - d['est_rx'])
                    else:
                        target = r_info['sin'] - (r_info['sout'] - d['est_tx'])
                    f_val = max(0.0, target)
                    f_w = 2.5 * r_info['anchor'] * r_info['bal']
                
                # Calculate Score of Winner
                sigma = get_sigma(val)
                sc_s = 0.8 * gaussian_sim(val, s_val, sigma)
                sc_p = 1.0 * gaussian_sim(val, p_val, sigma) if d['has_peer'] else 0.0
                sc_f = f_w * gaussian_sim(val, f_val, sigma) if f_val is not None else 0.0
                winner_score = sc_s + sc_p + sc_f
                
                # Calculate Score of Best Runner-Up
                alts = [s_val, 0.0]
                if d['has_peer']: alts.append(p_val)
                if f_val is not None: alts.append(f_val)
                
                runner_up_score = 0.0
                for alt in alts:
                    if abs(alt - val) < sigma: continue # Same cluster
                    
                    asigma = get_sigma(alt)
                    a_s = 0.8 * gaussian_sim(alt, s_val, asigma)
                    a_p = 1.0 * gaussian_sim(alt, p_val, asigma) if d['has_peer'] else 0.0
                    a_f = f_w * gaussian_sim(alt, f_val, asigma) if f_val is not None else 0.0
                    
                    score = a_s + a_p + a_f
                    if looks_like_zero(alt): score *= 0.2
                    
                    if score > runner_up_score:
                        runner_up_score = score
                        
                # Probability = Winner / (Winner + RunnerUp)
                # This measures dominance of the explanation
                total_mass = winner_score + runner_up_score + 0.001
                prob = winner_score / total_mass
                return max(0.5, min(1.0, prob))

            conf_rx = calc_conf(d['est_rx'], d['s_rx'], d['p_tx'], True)
            conf_tx = calc_conf(d['est_tx'], d['s_tx'], d['p_rx'], False)
            
            res['rx_rate'] = (d['s_rx'], d['est_rx'], conf_rx)
            res['tx_rate'] = (d['s_tx'], d['est_tx'], conf_tx)

        res['interface_status'] = (d['orig_stat'], d['status'], d['status_conf'])
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