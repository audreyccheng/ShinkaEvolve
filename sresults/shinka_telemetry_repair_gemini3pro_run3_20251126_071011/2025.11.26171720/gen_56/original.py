# EVOLVE-BLOCK-START
"""
Calibrated Consensus Telemetry Repair
Implements a hierarchical truth-selection algorithm (Link Symmetry > Flow Conservation)
with decoupled confidence calibration to ensure high confidence in large-magnitude repairs.
"""
from typing import Dict, Any, Tuple, List

import math
def repair_network_telemetry(telemetry: Dict[str, Dict[str, Any]],
                             topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:
    """
    Repairs telemetry using an Iterative Probabilistic Hypothesis Selection model.

    This approach moves beyond simple arbitration by:
    1. Generating hypothesis candidates (Self, Peer, Flow, Zero).
    2. Scoring them using a Gaussian kernel based on source reliability.
    3. Iteratively refining estimates to improve flow conservation signals.
    4. Calibrating confidence based on the absolute goodness-of-fit of the winning hypothesis.
    """

    # --- Configuration ---
    REL_TOL = 0.02
    ABS_TOL = 0.5
    ITERATIONS = 4
    MOMENTUM = 0.6  # Momentum for iterative updates (higher = slower adaptation)

    results = {}

    # --- Phase 1: Status Normalization & Initialization ---
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

        # Status Repair
        has_traffic = (s_rx > ABS_TOL or s_tx > ABS_TOL or
                       p_rx > ABS_TOL or p_tx > ABS_TOL)

        final_status = s_status
        status_conf = 1.0

        if s_status == 'down' and has_traffic:
            final_status = 'up'
            status_conf = 0.95
        elif s_status == 'up' and not has_traffic:
            if p_status == 'down':
                final_status = 'down'
                status_conf = 0.90

        # Initial Estimates (Seed with Peer/Symmetry if available, else Self)
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

    # --- Helper: Gaussian Scoring ---
    def get_score(candidate: float, target: float, weight: float) -> float:
        """Calculates score based on Gaussian likelihood."""
        if target is None: return 0.0
        diff = abs(candidate - target)
        # Sigma scales with magnitude (Linear approx of physical variance)
        sigma = max(ABS_TOL, max(abs(candidate), abs(target)) * REL_TOL)
        return weight * math.exp(-0.5 * (diff / sigma) ** 2)

    # --- Phase 2: Iterative Refinement ---
    for _ in range(ITERATIONS):
        # 1. Build Router Context
        router_stats = {}
        for r_id, if_list in topology.items():
            valid_ifs = [i for i in if_list if i in working_state]
            sum_in = sum(working_state[i]['est_rx'] for i in valid_ifs)
            sum_out = sum(working_state[i]['est_tx'] for i in valid_ifs)

            # Reliability score (Sigmoid decay based on imbalance)
            diff = abs(sum_in - sum_out)
            mag = max(sum_in, sum_out, 1.0)
            # 5% imbalance -> ~0.36 reliability score
            rel = math.exp(- (diff / (mag * 0.05)) ** 2)

            router_stats[r_id] = {'in': sum_in, 'out': sum_out, 'rel': rel}

        # 2. Update Estimates
        for if_id, d in working_state.items():
            if d['status'] == 'down': continue

            r_id = telemetry[if_id].get('local_router')
            rs = router_stats.get(r_id)

            # --- RX Update ---
            f_rx, f_w_rx = None, 0.0
            if rs:
                # Flow Target: Total_Out - (Total_In - Me)
                other_in = rs['in'] - d['est_rx']
                f_rx = max(0.0, rs['out'] - other_in)
                # Boost flow weight if isolated (no peer to contradict)
                base_w = 2.0 if not d['has_peer'] else 1.5
                f_w_rx = base_w * rs['rel']

            # Candidates: Self, Zero, Peer, Flow
            candidates = [d['s_rx'], 0.0]
            if d['has_peer']: candidates.append(d['p_tx'])
            if f_rx is not None: candidates.append(f_rx)

            best_val_rx = d['est_rx']
            best_score_rx = -1.0

            for cand in candidates:
                score = 0.0
                score += get_score(cand, d['s_rx'], 1.0) # Self
                if d['has_peer']: score += get_score(cand, d['p_tx'], 1.2) # Peer
                if f_rx is not None: score += get_score(cand, f_rx, f_w_rx) # Flow

                if score > best_score_rx:
                    best_score_rx = score
                    best_val_rx = cand

            # Momentum Update
            d['est_rx'] = MOMENTUM * d['est_rx'] + (1 - MOMENTUM) * best_val_rx

            # --- TX Update ---
            f_tx, f_w_tx = None, 0.0
            if rs:
                other_out = rs['out'] - d['est_tx']
                f_tx = max(0.0, rs['in'] - other_out)
                base_w = 2.0 if not d['has_peer'] else 1.5
                f_w_tx = base_w * rs['rel']

            candidates_tx = [d['s_tx'], 0.0]
            if d['has_peer']: candidates_tx.append(d['p_rx'])
            if f_tx is not None: candidates_tx.append(f_tx)

            best_val_tx = d['est_tx']
            best_score_tx = -1.0

            for cand in candidates_tx:
                score = 0.0
                score += get_score(cand, d['s_tx'], 1.0)
                if d['has_peer']: score += get_score(cand, d['p_rx'], 1.2)
                if f_tx is not None: score += get_score(cand, f_tx, f_w_tx)

                if score > best_score_tx:
                    best_score_tx = score
                    best_val_tx = cand

            d['est_tx'] = MOMENTUM * d['est_tx'] + (1 - MOMENTUM) * best_val_tx

    # --- Phase 3: Final Output & Confidence ---
    # Re-calculate router stats one last time for final confidence assessment
    final_rs = {}
    for r_id, if_list in topology.items():
        valid_ifs = [i for i in if_list if i in working_state]
        sum_in = sum(working_state[i]['est_rx'] for i in valid_ifs)
        sum_out = sum(working_state[i]['est_tx'] for i in valid_ifs)
        diff = abs(sum_in - sum_out)
        mag = max(sum_in, sum_out, 1.0)
        rel = math.exp(- (diff / (mag * 0.05)) ** 2)
        final_rs[r_id] = {'in': sum_in, 'out': sum_out, 'rel': rel}

    for if_id, d in working_state.items():
        res = telemetry[if_id].copy()

        if d['status'] == 'down':
             # High confidence if signals are quiet
             c_rx = 0.95 if d['s_rx'] > ABS_TOL else 1.0
             c_tx = 0.95 if d['s_tx'] > ABS_TOL else 1.0
             res['rx_rate'] = (d['s_rx'], 0.0, c_rx)
             res['tx_rate'] = (d['s_tx'], 0.0, c_tx)
        else:
             r_id = telemetry[if_id].get('local_router')
             rs = final_rs.get(r_id)

             # --- RX Confidence ---
             f_rx, f_w_rx = None, 0.0
             if rs:
                 other_in = rs['in'] - d['est_rx']
                 f_rx = max(0.0, rs['out'] - other_in)
                 base_w = 2.0 if not d['has_peer'] else 1.5
                 f_w_rx = base_w * rs['rel']

             score_rx = 0.0
             max_score_rx = 1.0 # Self
             score_rx += get_score(d['est_rx'], d['s_rx'], 1.0)

             if d['has_peer']:
                 max_score_rx += 1.2
                 score_rx += get_score(d['est_rx'], d['p_tx'], 1.2)

             if f_rx is not None:
                 max_score_rx += f_w_rx
                 score_rx += get_score(d['est_rx'], f_rx, f_w_rx)

             conf_rx = score_rx / max_score_rx
             # Bonus for perfect agreement
             conf_rx = min(1.0, conf_rx * 1.1)

             # --- TX Confidence ---
             f_tx, f_w_tx = None, 0.0
             if rs:
                 other_out = rs['out'] - d['est_tx']
                 f_tx = max(0.0, rs['in'] - other_out)
                 base_w = 2.0 if not d['has_peer'] else 1.5
                 f_w_tx = base_w * rs['rel']

             score_tx = 0.0
             max_score_tx = 1.0
             score_tx += get_score(d['est_tx'], d['s_tx'], 1.0)

             if d['has_peer']:
                 max_score_tx += 1.2
                 score_tx += get_score(d['est_tx'], d['p_rx'], 1.2)

             if f_tx is not None:
                 max_score_tx += f_w_tx
                 score_tx += get_score(d['est_tx'], f_tx, f_w_tx)

             conf_tx = score_tx / max_score_tx
             conf_tx = min(1.0, conf_tx * 1.1)

             res['rx_rate'] = (d['s_rx'], d['est_rx'], conf_rx)
             res['tx_rate'] = (d['s_tx'], d['est_tx'], conf_tx)

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