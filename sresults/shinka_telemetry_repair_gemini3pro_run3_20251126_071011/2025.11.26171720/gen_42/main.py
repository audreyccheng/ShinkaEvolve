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
    Repairs telemetry using a calibrated consensus model.

    Key Improvements:
    1. Decoupled Confidence: Confidence scores are based on the *reliability of the source*
       (e.g., Peer + Flow Agreement = 0.95), not the magnitude of the error.
    2. Hierarchical Validation:
       - Tier 1: Link Symmetry (Peer) - Strongest physical signal.
       - Tier 2: Flow Conservation - Contextual verification to resolve conflicts.
    3. Flow Gating: Flow signals are only used if the router's overall state is reasonably balanced,
       preventing noise propagation.
    """

    # --- Configuration ---
    REL_TOL = 0.02  # 2% Relative Tolerance
    ABS_TOL = 0.5   # 0.5 Mbps Absolute Tolerance (Noise Floor)

    results = {}

    # --- Phase 1: Status Normalization & Signal Collection ---
    working_state = {}

    # Pre-check agreement for initial estimates
    def check_agrees(v1, v2):
        d = abs(v1 - v2)
        return d <= ABS_TOL or (d / max(abs(v1), abs(v2), 1.0)) <= REL_TOL

    for if_id, data in telemetry.items():
        # 1. Extract Raw Signals
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

        # 2. Status Repair
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

        # 3. Initial Estimation (Symmetry > Self)
        if final_status == 'down':
            est_rx, est_tx = 0.0, 0.0
        else:
            # RX estimate
            if has_peer:
                est_rx = (s_rx + p_tx) / 2.0 if check_agrees(s_rx, p_tx) else p_tx
            else:
                est_rx = s_rx
            # TX estimate
            if has_peer:
                est_tx = (s_tx + p_rx) / 2.0 if check_agrees(s_tx, p_rx) else p_rx
            else:
                est_tx = s_tx

        working_state[if_id] = {
            's_rx': s_rx, 's_tx': s_tx,
            'p_rx': p_rx, 'p_tx': p_tx,
            'est_rx': est_rx, 'est_tx': est_tx,
            'status': final_status, 'status_conf': status_conf,
            'orig_status': s_status,
            'has_peer': has_peer
        }

    # --- Phase 2: Iterative Refinement ---
    # We refine estimates in a loop to improve Flow Conservation signals
    for iteration in range(2):
        # 1. Calculate Router Totals & Reliability
        router_totals = {}
        for router_id, if_list in topology.items():
            valid_ifs = [i for i in if_list if i in working_state]
            t_in = sum(working_state[i]['est_rx'] for i in valid_ifs)
            t_out = sum(working_state[i]['est_tx'] for i in valid_ifs)

            imb = abs(t_in - t_out)
            mx = max(t_in, t_out, 1.0)

            # Continuous reliability: 1.0 at 0% imbalance, 0.0 at 10% imbalance
            ratio = imb / mx
            reliability = max(0.0, 1.0 - (ratio * 10.0))
            router_totals[router_id] = {'in': t_in, 'out': t_out, 'rel': reliability}

        # 2. Update Estimates
        for if_id, d in working_state.items():
            r_id = telemetry[if_id].get('local_router')
            f_rx, f_tx, f_qual = d['est_rx'], d['est_tx'], 0.0

            if r_id and r_id in router_totals:
                rt = router_totals[r_id]
                f_rx = max(0.0, rt['out'] - (rt['in'] - d['est_rx']))
                f_tx = max(0.0, rt['in'] - (rt['out'] - d['est_tx']))
                f_qual = rt['rel']

            if d['status'] == 'down':
                d['est_rx'], d['est_tx'] = 0.0, 0.0
            else:
                # Update estimates using weighted voting
                v_rx, _ = arbitrate_voting(d['s_rx'], d['p_tx'], f_rx, d['has_peer'], f_qual, REL_TOL, ABS_TOL)
                v_tx, _ = arbitrate_voting(d['s_tx'], d['p_rx'], f_tx, d['has_peer'], f_qual, REL_TOL, ABS_TOL)

                d['est_rx'] = 0.5 * d['est_rx'] + 0.5 * v_rx
                d['est_tx'] = 0.5 * d['est_tx'] + 0.5 * v_tx

    # --- Phase 3: Final Arbitration ---

    # Recalculate Flow Targets one last time
    flow_context = {}
    for router_id, if_list in topology.items():
        valid_ifs = [i for i in if_list if i in working_state]
        t_in = sum(working_state[i]['est_rx'] for i in valid_ifs)
        t_out = sum(working_state[i]['est_tx'] for i in valid_ifs)

        ratio = abs(t_in - t_out) / max(t_in, t_out, 1.0)
        reliability = max(0.0, 1.0 - (ratio * 10.0))

        for i in valid_ifs:
            ws = working_state[i]
            target_rx = max(0.0, t_out - (t_in - ws['est_rx']))
            target_tx = max(0.0, t_in - (t_out - ws['est_tx']))
            flow_context[i] = {'rx': target_rx, 'tx': target_tx, 'qual': reliability}

    for if_id, d in working_state.items():
        f_ctx = flow_context.get(if_id, {'rx': 0.0, 'tx': 0.0, 'qual': 0.0})

        if d['status'] == 'down':
            final_rx, final_tx = 0.0, 0.0
            conf_rx = 0.95 if d['s_rx'] > ABS_TOL else 1.0
            conf_tx = 0.95 if d['s_tx'] > ABS_TOL else 1.0
        else:
            final_rx, conf_rx = arbitrate_voting(
                d['s_rx'], d['p_tx'], f_ctx['rx'], d['has_peer'], f_ctx['qual'], REL_TOL, ABS_TOL
            )
            final_tx, conf_tx = arbitrate_voting(
                d['s_tx'], d['p_rx'], f_ctx['tx'], d['has_peer'], f_ctx['qual'], REL_TOL, ABS_TOL
            )

        # Construct Result
        orig_data = telemetry[if_id]
        res_entry = orig_data.copy()
        res_entry['rx_rate'] = (d['s_rx'], final_rx, conf_rx)
        res_entry['tx_rate'] = (d['s_tx'], final_tx, conf_tx)
        res_entry['interface_status'] = (d['orig_status'], d['status'], d['status_conf'])

        results[if_id] = res_entry

    return results

def arbitrate_voting(v_self: float, v_peer: float, v_flow: float,
                     has_peer: bool, flow_qual: float,
                     rel_tol: float = 0.02, abs_tol: float = 0.5) -> Tuple[float, float]:
    """
    Arbitrates using a Weighted Cluster Voting model.
    Groups signals (Self, Peer, Flow) into agreement clusters and picks the strongest.
    """
    # 1. Define Votes
    votes = []

    # Self: Base trust.
    votes.append({'val': v_self, 'weight': 0.8, 'src': 'self'})

    # Peer: High trust (independent hardware).
    if has_peer:
        votes.append({'val': v_peer, 'weight': 1.0, 'src': 'peer'})

    # Flow: Trust scaled by router reliability.
    if flow_qual > 0.0:
        votes.append({'val': v_flow, 'weight': flow_qual * 1.0, 'src': 'flow'})

    # 2. Form Clusters
    # Greedy clustering: Pick a seed, find friends, merge.
    clusters = []
    processed = [False] * len(votes)

    for i in range(len(votes)):
        if processed[i]: continue

        # Start new cluster
        seed = votes[i]
        cluster = {'val_sum': seed['val'] * seed['weight'],
                   'weight_sum': seed['weight'],
                   'sources': {seed['src']},
                   'max_dist': 0.0}
        processed[i] = True

        # Find matches
        for j in range(i + 1, len(votes)):
            if processed[j]: continue

            # Check agreement
            cand = votes[j]
            diff = abs(seed['val'] - cand['val'])
            limit = max(abs(seed['val']), abs(cand['val']), 1.0) * rel_tol

            if diff <= max(abs_tol, limit):
                # Match found
                cluster['val_sum'] += cand['val'] * cand['weight']
                cluster['weight_sum'] += cand['weight']
                cluster['sources'].add(cand['src'])

                # Update max intra-cluster distance ratio
                norm_dist = diff / max(abs(seed['val']), abs(cand['val']), 1.0)
                cluster['max_dist'] = max(cluster['max_dist'], norm_dist)

                processed[j] = True

        clusters.append(cluster)

    # 3. Select Winner
    best_cluster = max(clusters, key=lambda c: c['weight_sum'])
    final_val = best_cluster['val_sum'] / best_cluster['weight_sum']

    # 4. Calibrate Confidence
    w = best_cluster['weight_sum']
    srcs = best_cluster['sources']

    if 'peer' in srcs and 'self' in srcs:
        base_conf = 0.95 # Physical agreement
    elif 'peer' in srcs and 'flow' in srcs:
        base_conf = 0.90 # Remote + Logic agreement
    elif 'self' in srcs and 'flow' in srcs:
        base_conf = 0.85 # Local + Logic agreement
    elif 'peer' in srcs:
        base_conf = 0.70 # Peer only fallback
    elif 'flow' in srcs:
        base_conf = 0.60 + (0.2 * flow_qual) # Flow only (isolated)
    else:
        base_conf = 0.50 # Self only

    # Penalize for distance within cluster
    penalty = 5.0 * best_cluster['max_dist']

    # Bonus for unanimity
    if len(votes) == 3 and len(srcs) == 3:
        base_conf = min(1.0, base_conf + 0.05)

    return final_val, max(0.5, base_conf - penalty)
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