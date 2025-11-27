# EVOLVE-BLOCK-START
"""
Network telemetry repair algorithm that detects and corrects inconsistencies
in network interface telemetry data using topology relationships.

Takes interface telemetry data and detects/repairs inconsistencies based on
network invariants like link symmetry and flow conservation.
"""
from typing import Dict, Any, Tuple, List


import math

def repair_network_telemetry(telemetry: Dict[str, Dict[str, Any]],
                             topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:
    """
    Repairs telemetry using an Iterative Consensus model with Momentum.

    Key Features:
    1. Hypothesis Selection: Arbitrates between Self, Peer, Flow, and Zero.
    2. Momentum Updates: Uses a rolling average during iteration to stabilize flow calculations.
    3. Dynamic Flow Trust: Flow signals are weighted by the router's current balance quality.
    4. Calibration: Confidence reflects the Gaussian likelihood of the chosen value against available signals.
    """

    # --- Configuration ---
    REL_TOL = 0.02
    ABS_TOL = 0.5
    MOMENTUM_ALPHA = 0.4 # How much new value contributes (0.4 means 60% retention)
    ITERATIONS = 4

    results = {}

    # --- Helper: Similarity Score ---
    def gaussian_similarity(v1, v2):
        if v1 is None or v2 is None: return 0.0
        diff = abs(v1 - v2)
        # Square Root Noise Model: sigma proportional to sqrt of magnitude
        # This models physical counting processes better than linear scaling
        mag = max(abs(v1), abs(v2))
        sigma = max(ABS_TOL, math.sqrt(mag) * 0.5)
        return math.exp(- (diff**2) / (2 * (sigma**2)))

    # --- Phase 1: Initialization & Status Repair ---
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

        # Initial Estimate Seeding
        if final_status == 'down':
            est_rx, est_tx = 0.0, 0.0
        else:
            # Seed with Peer if available (Symmetry), else Self
            est_rx = p_tx if has_peer else s_rx
            est_tx = p_rx if has_peer else s_tx

        working_state[if_id] = {
            's_rx': s_rx, 's_tx': s_tx,
            'p_rx': p_rx if has_peer else None,
            'p_tx': p_tx if has_peer else None,
            'est_rx': est_rx, 'est_tx': est_tx,
            'status': final_status,
            'status_conf': status_conf,
            'orig_status': s_status,
            'has_peer': has_peer
        }

    # --- Phase 2: Iterative Refinement ---
    for _ in range(ITERATIONS):
        # 1. Build Router Context
        router_context = {}
        for r_id, if_list in topology.items():
            valid_ifs = [i for i in if_list if i in working_state]
            sum_in = sum(working_state[i]['est_rx'] for i in valid_ifs)
            sum_out = sum(working_state[i]['est_tx'] for i in valid_ifs)

            # Reliability based on imbalance
            # We want a score that is high when balance is good
            imbalance = abs(sum_in - sum_out)
            magnitude = max(sum_in, sum_out, 1.0)

            # Score: 1.0 at 0 imbalance, drops to ~0.0 at 10% imbalance
            # Using exponential decay for sharper distinction
            rel_imbalance = imbalance / magnitude
            reliability = math.exp(- rel_imbalance * 20.0) # at 5% (0.05), exp(-1) = 0.36

            router_context[r_id] = {
                'sum_in': sum_in, 'sum_out': sum_out,
                'reliability': reliability
            }

        # 2. Update Estimates
        for if_id, d in working_state.items():
            if d['status'] == 'down':
                continue # Fixed at 0.0

            r_id = telemetry[if_id].get('local_router')

            # Define Signal Hypotheses
            # We will score candidate values.

            # --- RX Update ---
            candidates_rx = [d['s_rx']]
            if d['p_tx'] is not None: candidates_rx.append(d['p_tx'])
            candidates_rx.append(0.0) # Zero hypothesis

            f_rx = None
            f_weight_rx = 0.0

            if r_id and r_id in router_context:
                rc = router_context[r_id]
                # Implied flow value: Total_Out - (Total_In - Me)
                other_in = rc['sum_in'] - d['est_rx']
                f_rx = max(0.0, rc['sum_out'] - other_in)
                candidates_rx.append(f_rx)

                # Dynamic weight for flow
                # If the router is consistent, flow is strong.
                # However, if we are the CAUSE of inconsistency, the router looks bad.
                # We use a heuristic: Trust flow more if we have no peer.
                base_flow_w = 2.0
                if not d['has_peer']: base_flow_w = 3.0 # External link needs flow
                f_weight_rx = base_flow_w * rc['reliability']

            # Scoring function
            def score_candidate(val, s_val, p_val, f_val, f_w):
                score = 0.0
                # Support from Self
                score += 1.0 * gaussian_similarity(val, s_val)
                # Support from Peer
                if p_val is not None:
                    score += 2.0 * gaussian_similarity(val, p_val)
                # Support from Flow
                if f_val is not None:
                    score += f_w * gaussian_similarity(val, f_val)
                return score

            best_rx = d['est_rx']
            best_score_rx = -1.0

            # Optimization: Try the candidates themselves as the best value
            for c in candidates_rx:
                s = score_candidate(c, d['s_rx'], d['p_tx'], f_rx, f_weight_rx)
                if s > best_score_rx:
                    best_score_rx = s
                    best_rx = c

            # Momentum Update
            d['est_rx'] = (1.0 - MOMENTUM_ALPHA) * d['est_rx'] + MOMENTUM_ALPHA * best_rx

            # --- TX Update ---
            candidates_tx = [d['s_tx']]
            if d['p_rx'] is not None: candidates_tx.append(d['p_rx'])
            candidates_tx.append(0.0)

            f_tx = None
            f_weight_tx = 0.0

            if r_id and r_id in router_context:
                rc = router_context[r_id]
                other_out = rc['sum_out'] - d['est_tx']
                f_tx = max(0.0, rc['sum_in'] - other_out)
                candidates_tx.append(f_tx)

                base_flow_w = 2.0
                if not d['has_peer']: base_flow_w = 3.0
                f_weight_tx = base_flow_w * rc['reliability']

            best_tx = d['est_tx']
            best_score_tx = -1.0

            for c in candidates_tx:
                s = score_candidate(c, d['s_tx'], d['p_rx'], f_tx, f_weight_tx)
                if s > best_score_tx:
                    best_score_tx = s
                    best_tx = c

            d['est_tx'] = (1.0 - MOMENTUM_ALPHA) * d['est_tx'] + MOMENTUM_ALPHA * best_tx

    # --- Phase 3: Final Output Construction ---
    # 1. Calculate Final Router Context for Confidence Scoring
    final_router_context = {}
    for r_id, if_list in topology.items():
        valid_ifs = [i for i in if_list if i in working_state]
        s_in = sum(working_state[i]['est_rx'] for i in valid_ifs)
        s_out = sum(working_state[i]['est_tx'] for i in valid_ifs)
        imb = abs(s_in - s_out)
        mag = max(s_in, s_out, 1.0)
        rel = math.exp(-(imb/mag) * 20.0)
        final_router_context[r_id] = {'sum_in': s_in, 'sum_out': s_out, 'rel': rel}

    # 2. Construct Results with Calibrated Confidence
    for if_id, d in working_state.items():
        res = telemetry[if_id].copy()

        if d['status'] == 'down':
            c_rx = 0.95 if d['s_rx'] > ABS_TOL else 1.0
            c_tx = 0.95 if d['s_tx'] > ABS_TOL else 1.0
            res['rx_rate'] = (d['s_rx'], 0.0, c_rx)
            res['tx_rate'] = (d['s_tx'], 0.0, c_tx)
        else:
            # Flow context for this interface
            r_id = telemetry[if_id].get('local_router')
            f_rx, f_tx, f_rel = None, None, 0.0

            if r_id in final_router_context:
                rc = final_router_context[r_id]
                f_rel = rc['rel']
                # Infer flow target from final router state
                other_in = rc['sum_in'] - d['est_rx']
                f_rx = max(0.0, rc['sum_out'] - other_in)
                other_out = rc['sum_out'] - d['est_tx']
                f_tx = max(0.0, rc['sum_in'] - other_out)

            # Confidence Calculator: Weighted Consensus
            def get_conf(val, s_val, p_val, f_val, f_rel, has_peer):
                # Weights for sources
                w_s = 1.0 # Self baseline
                w_p = 2.0 if has_peer else 0.0 # Peer is strong
                w_f = 1.5 * f_rel # Flow depends on router balance reliability

                # Similarities
                sim_s = gaussian_similarity(val, s_val)
                sim_p = gaussian_similarity(val, p_val) if has_peer else 0.0
                sim_f = gaussian_similarity(val, f_val) if f_val is not None else 0.0

                weighted_sim = (sim_s * w_s) + (sim_p * w_p) + (sim_f * w_f)
                total_weight = w_s + w_p + w_f

                if total_weight < 0.1: return 0.5

                return weighted_sim / total_weight

            final_rx = d['est_rx']
            final_tx = d['est_tx']

            conf_rx = get_conf(final_rx, d['s_rx'], d['p_tx'], f_rx, f_rel, d['has_peer'])
            conf_tx = get_conf(final_tx, d['s_tx'], d['p_rx'], f_tx, f_rel, d['has_peer'])

            # Boost confidence slightly if it's very high (rounding/precision)
            if conf_rx > 0.95: conf_rx = 1.0
            if conf_tx > 0.95: conf_tx = 1.0

            res['rx_rate'] = (d['s_rx'], final_rx, conf_rx)
            res['tx_rate'] = (d['s_tx'], final_tx, conf_tx)

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