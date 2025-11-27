# EVOLVE-BLOCK-START
"""
Hypothesis-Based Consensus Repair
Replaces heuristic arbitration with a Gaussian Mixture hypothesis selection model.
This approach explicitly scores candidate values (Self, Peer, Flow, Zero) based on
their likelihood given the observed signals and their respective reliabilities.
"""
from typing import Dict, Any, Tuple, List
import math

def repair_network_telemetry(telemetry: Dict[str, Dict[str, Any]],
                             topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:
    """
    Repairs telemetry using an Iterative Hypothesis Selection model.

    Key Innovations:
    1. Hypothesis Competition: Instead of merging values, we generate specific candidates
       (Self, Peer, Flow, Average, Zero) and score them using a weighted Gaussian kernel.
    2. Dynamic Trust Weights:
       - Peer (Link Symmetry) is trusted most (Weight 1.2).
       - Self is trusted baseline (Weight 1.0).
       - Flow is trusted dynamically based on router balance (Weight up to 1.5).
    3. Probabilistic Confidence: Confidence scores reflect the normalized likelihood
       of the winning hypothesis.
    """

    # --- Configuration ---
    REL_TOL = 0.02
    ABS_TOL = 0.5

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

        # Initial Estimate: Symmetry > Self
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
    ITERATIONS = 4

    for _ in range(ITERATIONS):
        # 1. Calculate Router Context
        router_stats = {}
        for r_id, if_list in topology.items():
            valid_ifs = [i for i in if_list if i in working_state]
            t_in = sum(working_state[i]['est_rx'] for i in valid_ifs)
            t_out = sum(working_state[i]['est_tx'] for i in valid_ifs)

            # Reliability score using sigmoid-like decay
            imb = abs(t_in - t_out)
            mx = max(t_in, t_out, 1.0)
            # Knee at 8% imbalance (relaxed to allow flow logic to work through noise)
            reliability = 1.0 / (1.0 + (imb / (mx * 0.08))**2)

            router_stats[r_id] = {'in': t_in, 'out': t_out, 'rel': reliability}

        # 2. Update Interface Estimates
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

            # Hypothesis Selection
            target_rx, _ = solve_hypothesis(
                d['s_rx'], d['p_tx'], f_rx,
                d['has_peer'], f_qual, REL_TOL, ABS_TOL
            )

            target_tx, _ = solve_hypothesis(
                d['s_tx'], d['p_rx'], f_tx,
                d['has_peer'], f_qual, REL_TOL, ABS_TOL
            )

            # Momentum Update (0.5 to keep stability)
            d['est_rx'] = 0.5 * d['est_rx'] + 0.5 * target_rx
            d['est_tx'] = 0.5 * d['est_tx'] + 0.5 * target_tx

    # --- Phase 3: Final Output Generation ---
    # Re-calc router stats for final flow context
    final_router_stats = {}
    for r_id, if_list in topology.items():
        valid_ifs = [i for i in if_list if i in working_state]
        t_in = sum(working_state[i]['est_rx'] for i in valid_ifs)
        t_out = sum(working_state[i]['est_tx'] for i in valid_ifs)
        imb = abs(t_in - t_out)
        mx = max(t_in, t_out, 1.0)
        final_router_stats[r_id] = {'in': t_in, 'out': t_out, 'rel': 1.0 / (1.0 + (imb/(mx*0.08))**2)}

    results = {}
    for if_id, d in working_state.items():
        r_id = telemetry[if_id].get('local_router')
        f_rx, f_tx, f_qual = d['est_rx'], d['est_tx'], 0.0

        if r_id and r_id in final_router_stats:
            rs = final_router_stats[r_id]
            f_tx = max(0.0, rs['in'] - (rs['out'] - d['est_tx']))
            f_rx = max(0.0, rs['out'] - (rs['in'] - d['est_rx']))
            f_qual = rs['rel']

        res_entry = telemetry[if_id].copy()

        if d['status'] == 'down':
             conf_rx = 1.0 if d['s_rx'] < ABS_TOL else 0.95
             conf_tx = 1.0 if d['s_tx'] < ABS_TOL else 0.95
             res_entry['rx_rate'] = (d['s_rx'], 0.0, conf_rx)
             res_entry['tx_rate'] = (d['s_tx'], 0.0, conf_tx)
        else:
             final_rx, conf_rx = solve_hypothesis(
                 d['s_rx'], d['p_tx'], f_rx,
                 d['has_peer'], f_qual, REL_TOL, ABS_TOL
             )
             final_tx, conf_tx = solve_hypothesis(
                 d['s_tx'], d['p_rx'], f_tx,
                 d['has_peer'], f_qual, REL_TOL, ABS_TOL
             )
             res_entry['rx_rate'] = (d['s_rx'], final_rx, conf_rx)
             res_entry['tx_rate'] = (d['s_tx'], final_tx, conf_tx)

        res_entry['interface_status'] = (d['orig_status'], d['status'], d['status_conf'])
        results[if_id] = res_entry

    return results

def solve_hypothesis(v_self, v_peer, v_flow, has_peer, flow_qual, rel_tol, abs_tol):
    """
    Selects the best estimate by scoring candidates against available signals.
    """
    # 1. Define Candidates
    candidates = {v_self, 0.0}
    if has_peer:
        candidates.add(v_peer)
        candidates.add((v_self + v_peer) / 2.0)

    # Allow flow if it has even minimal quality
    if flow_qual > 0.05:
        candidates.add(v_flow)
        if has_peer:
            candidates.add((v_peer + v_flow) / 2.0)
        candidates.add((v_self + v_flow) / 2.0)

    # 2. Weights
    # Widen sigma: 3x tolerance window to improve confidence calibration and recall
    # This reflects that "within 3 sigma" (approx 6%) is still strongly supportive
    SIGMA_SCALE = 3.0

    W_SELF = 1.0
    W_PEER = 1.5 # Boosted symmetry importance

    # Adaptive Flow Weight
    # If no peer, flow is the only external validator. Trust it more if quality is decent.
    if not has_peer:
        W_FLOW = 3.0 * flow_qual
    else:
        W_FLOW = 1.5 * flow_qual

    # 3. Score Candidates
    best_score = -1.0
    best_val = v_self

    for cand in candidates:
        score = 0.0

        # Helper for Gaussian Kernel
        def kernel(val, ref, w):
            sigma = SIGMA_SCALE * max(abs(val) * rel_tol, abs_tol)
            z = abs(val - ref) / sigma
            return w * math.exp(-0.5 * min(z*z, 20.0))

        # Self Score
        score += kernel(cand, v_self, W_SELF)

        # Peer Score
        if has_peer:
            score += kernel(cand, v_peer, W_PEER)

        # Flow Score
        if flow_qual > 0.05:
            score += kernel(cand, v_flow, W_FLOW)

        if score > best_score:
            best_score = score
            best_val = cand

    # 4. Confidence
    # Normalize by potential max score to get [0,1] confidence
    max_weight = W_SELF + (W_PEER if has_peer else 0) + (W_FLOW if flow_qual > 0.05 else 0)
    confidence = best_score / max(max_weight, 1.0)

    return best_val, min(1.0, max(0.0, confidence))
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