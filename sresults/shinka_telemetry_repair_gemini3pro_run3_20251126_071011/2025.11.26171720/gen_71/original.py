# EVOLVE-BLOCK-START
"""
Calibrated Consensus Telemetry Repair
Implements a hierarchical truth-selection algorithm (Link Symmetry > Flow Conservation)
with decoupled confidence calibration to ensure high confidence in large-magnitude repairs.
"""
from typing import Dict, Any, Tuple, List

def repair_network_telemetry(telemetry: Dict[str, Dict[str, Any]],
                             topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:
    """
    Repairs telemetry using an Iterative Calibrated Consensus model.

    Strategy:
    1. Initial Guess: Based on Link Symmetry (Peer > Self).
    2. Iterative Refinement (3 Passes):
       - Updates estimates by arbitrating between Self, Peer, and Flow.
       - Uses 'Weighted Cluster Voting' to robustly identify the consensus value.
       - Applies momentum to smooth convergence of Flow calculations.
    3. Final Arbitration: Explicitly re-runs arbitration on stabilized values
       to generate consistent value-confidence pairs.
    """

    # --- Configuration ---
    REL_TOL = 0.02  # 2% Relative Tolerance
    ABS_TOL = 0.5   # 0.5 Mbps Noise Floor

    results = {}

    # --- Phase 1: Status Normalization & Initialization ---
    working_state = {}

    for if_id, data in telemetry.items():
        # Extract Signals
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
        elif s_status == 'up' and not has_traffic and p_status == 'down':
            final_status = 'down'
            status_conf = 0.90

        # Initial Estimates (Prefer Peer, else Self)
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
    for iteration in range(3):
        # 1. Calculate Router Totals from current estimates
        router_totals = {}
        for r_id, if_list in topology.items():
            valid_ifs = [i for i in if_list if i in working_state]
            t_in = sum(working_state[i]['est_rx'] for i in valid_ifs)
            t_out = sum(working_state[i]['est_tx'] for i in valid_ifs)

            imb = abs(t_in - t_out)
            mx = max(t_in, t_out, 1.0)

            # Reliability decays with imbalance
            ratio = imb / mx
            quality = max(0.0, 1.0 - (ratio * 10.0))
            router_totals[r_id] = {'in': t_in, 'out': t_out, 'quality': quality}

        # 2. Update Estimates
        for if_id, d in working_state.items():
            if d['status'] == 'down':
                 d['est_rx'], d['est_tx'] = 0.0, 0.0
                 continue

            r_id = telemetry[if_id].get('local_router')

            # Flow Targets
            f_rx, f_tx, f_qual = d['est_rx'], d['est_tx'], 0.0
            if r_id and r_id in router_totals:
                rt = router_totals[r_id]
                f_qual = rt['quality']
                f_rx = max(0.0, rt['out'] - (rt['in'] - d['est_rx']))
                f_tx = max(0.0, rt['in'] - (rt['out'] - d['est_tx']))

            # Soft Update with Momentum
            val_rx, _ = arbitrate(d['s_rx'], d['p_tx'], f_rx, d['has_peer'], f_qual, REL_TOL, ABS_TOL)
            val_tx, _ = arbitrate(d['s_tx'], d['p_rx'], f_tx, d['has_peer'], f_qual, REL_TOL, ABS_TOL)

            d['est_rx'] = 0.6 * d['est_rx'] + 0.4 * val_rx
            d['est_tx'] = 0.6 * d['est_tx'] + 0.4 * val_tx

    # --- Phase 3: Final Arbitration ---
    # Recalculate context one last time for final consistent output
    final_router_totals = {}
    for r_id, if_list in topology.items():
        valid_ifs = [i for i in if_list if i in working_state]
        t_in = sum(working_state[i]['est_rx'] for i in valid_ifs)
        t_out = sum(working_state[i]['est_tx'] for i in valid_ifs)
        ratio = abs(t_in - t_out) / max(t_in, t_out, 1.0)
        final_router_totals[r_id] = {'in': t_in, 'out': t_out, 'quality': max(0.0, 1.0 - (ratio * 10.0))}

    for if_id, d in working_state.items():
        res = telemetry[if_id].copy()

        if d['status'] == 'down':
            crx = 1.0 if d['s_rx'] <= ABS_TOL else 0.95
            ctx = 1.0 if d['s_tx'] <= ABS_TOL else 0.95
            res['rx_rate'] = (d['s_rx'], 0.0, crx)
            res['tx_rate'] = (d['s_tx'], 0.0, ctx)
        else:
            r_id = telemetry[if_id].get('local_router')
            f_rx, f_tx, f_qual = d['est_rx'], d['est_tx'], 0.0

            if r_id and r_id in final_router_totals:
                rt = final_router_totals[r_id]
                f_qual = rt['quality']
                f_rx = max(0.0, rt['out'] - (rt['in'] - d['est_rx']))
                f_tx = max(0.0, rt['in'] - (rt['out'] - d['est_tx']))

            # Final clean arbitration without momentum
            final_rx, conf_rx = arbitrate(d['s_rx'], d['p_tx'], f_rx, d['has_peer'], f_qual, REL_TOL, ABS_TOL)
            final_tx, conf_tx = arbitrate(d['s_tx'], d['p_rx'], f_tx, d['has_peer'], f_qual, REL_TOL, ABS_TOL)

            res['rx_rate'] = (d['s_rx'], final_rx, conf_rx)
            res['tx_rate'] = (d['s_tx'], final_tx, conf_tx)

        res['interface_status'] = (d['orig_status'], d['status'], d['status_conf'])
        results[if_id] = res

    return results

def arbitrate(v_self: float, v_peer: float, v_flow: float,
              has_peer: bool, flow_qual: float, rel_tol: float, abs_tol: float) -> Tuple[float, float]:
    """
    Arbitrates using Weighted Cluster Voting.
    Groups agreeing signals and selects the strongest cluster.
    """
    # 1. Define Votes
    votes = []
    votes.append({'val': v_self, 'w': 0.8, 'src': 'self'})

    if has_peer:
        votes.append({'val': v_peer, 'w': 1.0, 'src': 'peer'})

    if flow_qual > 0.0:
        votes.append({'val': v_flow, 'w': 1.0 * flow_qual, 'src': 'flow'})

    # 2. Form Clusters
    clusters = []
    processed = [False] * len(votes)

    for i in range(len(votes)):
        if processed[i]: continue

        seed = votes[i]
        c = {'val_sum': seed['val'] * seed['w'], 'w_sum': seed['w'],
             'srcs': {seed['src']}, 'max_dist': 0.0}
        processed[i] = True

        for j in range(i+1, len(votes)):
            if processed[j]: continue
            cand = votes[j]

            # Check agreement
            diff = abs(seed['val'] - cand['val'])
            limit = max(abs(seed['val']), abs(cand['val']), 1.0) * rel_tol

            if diff <= max(abs_tol, limit):
                c['val_sum'] += cand['val'] * cand['w']
                c['w_sum'] += cand['w']
                c['srcs'].add(cand['src'])

                # Update cluster spread metric
                norm_d = diff / max(abs(seed['val']), abs(cand['val']), 1.0)
                c['max_dist'] = max(c['max_dist'], norm_d)
                processed[j] = True
        clusters.append(c)

    # 3. Select Winner
    if not clusters: return v_self, 0.5
    best = max(clusters, key=lambda x: x['w_sum'])

    final_val = best['val_sum'] / best['w_sum']

    # 4. Calibrate Confidence
    srcs = best['srcs']
    base_conf = 0.5

    if 'peer' in srcs and 'self' in srcs:
        base_conf = 0.95
    elif 'peer' in srcs and 'flow' in srcs:
        base_conf = 0.90
    elif 'self' in srcs and 'flow' in srcs:
        base_conf = 0.85
    elif 'peer' in srcs:
        base_conf = 0.70
    elif 'flow' in srcs:
        base_conf = 0.60 + (0.2 * flow_qual)
    else:
        base_conf = 0.50

    # Penalty for cluster width (internal disagreement)
    penalty = 5.0 * best['max_dist']

    # Unanimity Bonus
    if len(srcs) == 3:
        base_conf = min(1.0, base_conf + 0.05)

    return final_val, max(0.0, base_conf - penalty)
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