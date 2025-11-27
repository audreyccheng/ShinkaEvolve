# EVOLVE-BLOCK-START
"""
Hybrid Gaussian-Cluster Telemetry Repair
Combines Gaussian hypothesis selection for high-accuracy estimation with
discrete supporter-based confidence tiers for robust calibration.
"""
from typing import Dict, Any, Tuple, List
import math

def repair_network_telemetry(telemetry: Dict[str, Dict[str, Any]],
                             topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:
    """
    Repairs network telemetry using a hybrid approach:
    1. Estimation: Candidates (Self, Peer, Flow, Averages) are scored using
       Gaussian likelihoods to select the most probable value.
    2. Calibration: Confidence is assigned based on the set of 'Supporters'
       (signals agreeing with the selected value), adhering to a hierarchy of trust
       (Peer+Self > Peer+Flow > etc.).
    """

    # --- Configuration ---
    REL_TOL = 0.02
    ABS_TOL = 0.5
    ITERATIONS = 3
    MOMENTUM = 0.5

    results = {}

    # --- Phase 1: Initialization ---
    working_state = {}

    for if_id, data in telemetry.items():
        s_rx = float(data.get('rx_rate', 0.0))
        s_tx = float(data.get('tx_rate', 0.0))
        s_status = data.get('interface_status', 'unknown')

        peer_id = data.get('connected_to')
        has_peer = False
        p_rx, p_tx, p_status = 0.0, 0.0, 'unknown'

        if peer_id and peer_id in telemetry:
            has_peer = True
            p_data = telemetry[peer_id]
            p_rx = float(p_data.get('rx_rate', 0.0))
            p_tx = float(p_data.get('tx_rate', 0.0))
            p_status = p_data.get('interface_status', 'unknown')

        # Status Logic
        has_traffic = (s_rx > ABS_TOL or s_tx > ABS_TOL or
                       p_rx > ABS_TOL or p_tx > ABS_TOL)

        final_status = s_status
        status_conf = 1.0

        if s_status == 'down' and has_traffic:
            final_status = 'up'
            status_conf = 0.95
        elif s_status == 'up' and not has_traffic and p_status == 'down':
            final_status = 'down'
            status_conf = 0.90

        # Initial Estimate (Prefer Peer, else Self)
        if final_status == 'down':
            est_rx, est_tx = 0.0, 0.0
        else:
            est_rx = p_tx if has_peer else s_rx
            est_tx = p_rx if has_peer else s_tx

        working_state[if_id] = {
            's_rx': s_rx, 's_tx': s_tx,
            'p_rx': p_rx, 'p_tx': p_tx,
            'est_rx': est_rx, 'est_tx': est_tx,
            'status': final_status, 'status_conf': status_conf,
            'orig_status': s_status,
            'has_peer': has_peer
        }

    # --- Phase 2: Iterative Refinement ---
    for _ in range(ITERATIONS):
        # 1. Calculate Router Reliability
        router_stats = {}
        for r_id, if_list in topology.items():
            valid_ifs = [i for i in if_list if i in working_state]
            t_in = sum(working_state[i]['est_rx'] for i in valid_ifs)
            t_out = sum(working_state[i]['est_tx'] for i in valid_ifs)

            imb = abs(t_in - t_out)
            mx = max(t_in, t_out, 1.0)
            # Smooth sigmoid decay for reliability
            # 8% imbalance = 50% reliability
            reliability = 1.0 / (1.0 + (imb / (mx * 0.08))**2)
            router_stats[r_id] = {'in': t_in, 'out': t_out, 'rel': reliability}

        # 2. Update Estimates
        for if_id, d in working_state.items():
            if d['status'] == 'down':
                d['est_rx'], d['est_tx'] = 0.0, 0.0
                continue

            r_id = telemetry[if_id].get('local_router')

            # Calculate Flow Targets
            f_rx, f_tx, f_qual = d['est_rx'], d['est_tx'], 0.0
            if r_id and r_id in router_stats:
                rs = router_stats[r_id]
                f_tx = max(0.0, rs['in'] - (rs['out'] - d['est_tx']))
                f_rx = max(0.0, rs['out'] - (rs['in'] - d['est_rx']))
                f_qual = rs['rel']

            # Arbitrate
            val_rx, _ = arbitrate(d['s_rx'], d['p_tx'], f_rx, d['has_peer'], f_qual, d['status'], REL_TOL, ABS_TOL)
            val_tx, _ = arbitrate(d['s_tx'], d['p_rx'], f_tx, d['has_peer'], f_qual, d['status'], REL_TOL, ABS_TOL)

            # Update with momentum
            d['est_rx'] = (1.0 - MOMENTUM) * d['est_rx'] + MOMENTUM * val_rx
            d['est_tx'] = (1.0 - MOMENTUM) * d['est_tx'] + MOMENTUM * val_tx

    # --- Phase 3: Final Output ---
    # Final context calculation for accurate confidence
    final_router_stats = {}
    for r_id, if_list in topology.items():
        valid_ifs = [i for i in if_list if i in working_state]
        t_in = sum(working_state[i]['est_rx'] for i in valid_ifs)
        t_out = sum(working_state[i]['est_tx'] for i in valid_ifs)
        imb = abs(t_in - t_out)
        mx = max(t_in, t_out, 1.0)
        final_router_stats[r_id] = {'in': t_in, 'out': t_out, 'rel': 1.0 / (1.0 + (imb/(mx*0.08))**2)}

    for if_id, d in working_state.items():
        res = telemetry[if_id].copy()

        if d['status'] == 'down':
            # High confidence if signals are indeed zero
            crx = 1.0 if d['s_rx'] <= ABS_TOL else 0.95
            ctx = 1.0 if d['s_tx'] <= ABS_TOL else 0.95
            res['rx_rate'] = (d['s_rx'], 0.0, crx)
            res['tx_rate'] = (d['s_tx'], 0.0, ctx)
        else:
            r_id = telemetry[if_id].get('local_router')
            f_rx, f_tx, f_qual = d['est_rx'], d['est_tx'], 0.0
            if r_id and r_id in final_router_stats:
                rs = final_router_stats[r_id]
                f_qual = rs['rel']
                f_tx = max(0.0, rs['in'] - (rs['out'] - d['est_tx']))
                f_rx = max(0.0, rs['out'] - (rs['in'] - d['est_rx']))

            # Final arbitration (no momentum)
            final_rx, conf_rx = arbitrate(d['s_rx'], d['p_tx'], f_rx, d['has_peer'], f_qual, d['status'], REL_TOL, ABS_TOL)
            final_tx, conf_tx = arbitrate(d['s_tx'], d['p_rx'], f_tx, d['has_peer'], f_qual, d['status'], REL_TOL, ABS_TOL)

            res['rx_rate'] = (d['s_rx'], final_rx, conf_rx)
            res['tx_rate'] = (d['s_tx'], final_tx, conf_tx)

        res['interface_status'] = (d['orig_status'], d['status'], d['status_conf'])
        results[if_id] = res

    return results

def arbitrate(v_self, v_peer, v_flow, has_peer, flow_qual, status, rel_tol, abs_tol):
    """
    Selects best value using Gaussian scoring, then calibrates confidence 
    based on the tier of agreement (Supporters).
    """
    # 1. Candidates
    # We include averages to catch the center of noisy agreement
    candidates = [v_self, 0.0]
    if has_peer:
        candidates.append(v_peer)
        candidates.append((v_self + v_peer) / 2.0)
    
    # Flow candidates included if quality is sufficient
    if flow_qual > 0.1:
        candidates.append(v_flow)
        if has_peer:
            candidates.append((v_peer + v_flow) / 2.0)
        candidates.append((v_self + v_flow) / 2.0)

    # 2. Gaussian Scoring (Winner Selection)
    # Weights reflecting signal trustworthiness
    w_s = 1.0
    w_p = 1.2 if has_peer else 0.0
    # Flow weight scales with quality
    w_f = 1.5 * flow_qual if flow_qual > 0.0 else 0.0

    best_val = v_self
    best_score = -1.0
    
    # Helper: Gaussian Kernel
    def get_score(cand):
        score = 0.0
        # Sigma scales with magnitude to handle relative noise
        # We use a multiplier (2.0) to make the kernel wide enough to catch agreement
        sigma_base = max(abs(cand), 1.0) * rel_tol
        sigma = max(abs_tol, sigma_base) * 2.0 
        
        def kernel(target, w):
            if w <= 0: return 0.0
            diff = abs(cand - target)
            return w * math.exp(-0.5 * (diff / sigma) ** 2)

        score += kernel(v_self, w_s)
        if has_peer: score += kernel(v_peer, w_p)
        if w_f > 0.0: score += kernel(v_flow, w_f)
        
        # Penalize Zero hypothesis if status is UP (unless data supports it)
        if status == 'up' and cand < abs_tol:
            score *= 0.5
            
        return score

    for c in candidates:
        s = get_score(c)
        if s > best_score:
            best_score = s
            best_val = c
            
    # 3. Supporter-Based Confidence (Calibration)
    # Identify which signals support the winner
    supporters = set()
    
    def supports(target):
        # Strict check for confidence assignment
        diff = abs(best_val - target)
        limit = max(abs_tol, max(abs(best_val), abs(target)) * rel_tol)
        return diff <= limit

    if supports(v_self): supporters.add('self')
    if has_peer and supports(v_peer): supporters.add('peer')
    if w_f > 0.2 and supports(v_flow): supporters.add('flow')
    
    # Hierarchy of Trust
    base_conf = 0.5
    if 'peer' in supporters and 'self' in supporters:
        base_conf = 0.95
    elif 'peer' in supporters and 'flow' in supporters:
        base_conf = 0.90
    elif 'self' in supporters and 'flow' in supporters:
        base_conf = 0.85
    elif 'peer' in supporters:
        base_conf = 0.75
    elif 'flow' in supporters:
        base_conf = 0.60 + 0.2 * flow_qual
    else:
        # Fallback: if only self supports, or nothing supports (e.g. forced zero)
        if 'self' in supporters: base_conf = 0.5
        else: base_conf = 0.4 # Uncertainty
        
    # Unanimity Bonus
    if len(supporters) == 3:
        base_conf = min(1.0, base_conf + 0.04)
        
    # Distance Penalty (Fine-tuning)
    # Reduces confidence if the agreement is barely within tolerance
    dist_penalty = 0.0
    total_w = 0.0
    
    if 'self' in supporters:
        d = abs(best_val - v_self) / max(1.0, abs(best_val))
        dist_penalty += d * w_s; total_w += w_s
    if 'peer' in supporters:
        d = abs(best_val - v_peer) / max(1.0, abs(best_val))
        dist_penalty += d * w_p; total_w += w_p
        
    if total_w > 0:
        # Scale penalty: 1% deviation -> 0.05 drop in confidence
        avg_dist = dist_penalty / total_w
        base_conf -= avg_dist * 5.0

    return best_val, max(0.0, min(1.0, base_conf))
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