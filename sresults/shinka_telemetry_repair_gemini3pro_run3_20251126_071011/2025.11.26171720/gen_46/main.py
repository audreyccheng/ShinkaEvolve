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
    Repairs telemetry using an Iterative Trusted Consensus model.

    Key Concepts:
    1. Trust Scores: Each interface maintains a 'trust' score (0.0-1.0) based on
       agreement with Peer and Flow.
    2. Trusted Flow: Flow targets are weighted by the trust of *other* interfaces.
       High-trust neighbors can force a correction on a low-trust interface.
    3. Weighted Voting: Arbitration uses weighted clustering of Self, Peer, and Flow.
    """

    # --- Configuration ---
    REL_TOL = 0.02
    ABS_TOL = 0.5

    results = {}
    working_state = {}

    # --- Phase 1: Initialization & Status Repair ---
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
        has_traffic = (s_rx > ABS_TOL or s_tx > ABS_TOL or p_rx > ABS_TOL or p_tx > ABS_TOL)
        final_status = s_status
        status_conf = 1.0

        if s_status == 'down' and has_traffic:
            final_status = 'up'
            status_conf = 0.95
        elif s_status == 'up' and not has_traffic and p_status == 'down':
            final_status = 'down'
            status_conf = 0.90

        # Initial Trust & Estimate
        # Start with a bias towards Peer (if available)
        trust = 0.5
        est_rx, est_tx = s_rx, s_tx

        if final_status == 'down':
            est_rx, est_tx = 0.0, 0.0
            trust = 1.0 # Confident in zero
        elif has_peer:
            # Check agreement
            diff_rx = abs(s_rx - p_tx)
            match_rx = diff_rx <= ABS_TOL or (diff_rx / max(abs(s_rx), abs(p_tx), 1.0)) <= REL_TOL

            if match_rx:
                est_rx = (s_rx + p_tx) / 2.0
                trust = 0.9
            else:
                est_rx = p_tx
                trust = 0.7 # Trust Peer more, but not fully since conflict exists

            # Similar logic for TX (checking against Peer RX)
            diff_tx = abs(s_tx - p_rx)
            match_tx = diff_tx <= ABS_TOL or (diff_tx / max(abs(s_tx), abs(p_rx), 1.0)) <= REL_TOL
            if match_tx:
                est_tx = (s_tx + p_rx) / 2.0
            else:
                est_tx = p_rx
            # Averaging trust for simplicity in this struct
            trust = 0.9 if (match_rx and match_tx) else 0.7
        else:
            trust = 0.5 # Isolated, low trust

        working_state[if_id] = {
            's_rx': s_rx, 's_tx': s_tx,
            'p_rx': p_rx, 'p_tx': p_tx,
            'est_rx': est_rx, 'est_tx': est_tx,
            'trust': trust,
            'status': final_status, 'status_conf': status_conf,
            'orig_status': s_status,
            'has_peer': has_peer
        }

    # --- Phase 2: Iterative Trusted Consensus ---
    for iteration in range(3):
        # 1. Calculate Router Stats (Totals & Trust Mass)
        router_stats = {}
        for r_id, if_list in topology.items():
            valid_ifs = [i for i in if_list if i in working_state]

            sum_in = sum(working_state[i]['est_rx'] for i in valid_ifs)
            sum_out = sum(working_state[i]['est_tx'] for i in valid_ifs)

            # Calculate average trust of the router's interfaces
            total_trust = sum(working_state[i]['trust'] for i in valid_ifs)
            avg_trust = total_trust / max(len(valid_ifs), 1)

            router_stats[r_id] = {
                'sum_in': sum_in, 'sum_out': sum_out,
                'avg_trust': avg_trust,
                'count': len(valid_ifs)
            }

        # 2. Update Interfaces
        for if_id, d in working_state.items():
            if d['status'] == 'down': continue

            r_id = telemetry[if_id].get('local_router')

            # Flow Calculation
            f_rx, f_tx = d['est_rx'], d['est_tx']
            neighbor_trust = 0.0

            if r_id and r_id in router_stats:
                rs = router_stats[r_id]
                # To get trust of *others*, we subtract our contribution
                # (Approximation for performance)
                other_trust_sum = (rs['avg_trust'] * rs['count']) - d['trust']
                other_count = max(1, rs['count'] - 1)
                neighbor_trust = other_trust_sum / other_count

                # Flow Targets
                other_rx = rs['sum_in'] - d['est_rx']
                f_rx = max(0.0, rs['sum_out'] - other_rx)

                other_tx = rs['sum_out'] - d['est_tx']
                f_tx = max(0.0, rs['sum_in'] - other_tx)

            # Arbitrate RX
            new_rx, conf_rx = arbitrate_voting(
                d['s_rx'], d['p_tx'], f_rx,
                d['has_peer'], neighbor_trust, REL_TOL, ABS_TOL
            )
            # Soft Update
            d['est_rx'] = 0.5 * d['est_rx'] + 0.5 * new_rx

            # Arbitrate TX
            new_tx, conf_tx = arbitrate_voting(
                d['s_tx'], d['p_rx'], f_tx,
                d['has_peer'], neighbor_trust, REL_TOL, ABS_TOL
            )
            d['est_tx'] = 0.5 * d['est_tx'] + 0.5 * new_tx

            # Update Trust (Momentum)
            iter_trust = (conf_rx + conf_tx) / 2.0
            d['trust'] = 0.5 * d['trust'] + 0.5 * iter_trust

            # Store confidences for final output
            d['final_conf_rx'] = conf_rx
            d['final_conf_tx'] = conf_tx

    # --- Phase 3: Final Output ---
    for if_id, d in working_state.items():
        res = telemetry[if_id].copy()

        if d['status'] == 'down':
            crx = 1.0 if d['s_rx'] <= ABS_TOL else 0.95
            ctx = 1.0 if d['s_tx'] <= ABS_TOL else 0.95
            res['rx_rate'] = (d['s_rx'], 0.0, crx)
            res['tx_rate'] = (d['s_tx'], 0.0, ctx)
        else:
            res['rx_rate'] = (d['s_rx'], d['est_rx'], d.get('final_conf_rx', 0.5))
            res['tx_rate'] = (d['s_tx'], d['est_tx'], d.get('final_conf_tx', 0.5))

        res['interface_status'] = (d['orig_status'], d['status'], d['status_conf'])
        results[if_id] = res

    return results

def arbitrate_voting(v_self, v_peer, v_flow, has_peer, flow_weight, rel_tol, abs_tol):
    """
    Weighted Clustering Voting for Consensus.

    Weights:
    - Self: 0.6 (Base truth, but often noisy or broken)
    - Peer: 1.2 (Strong hardware indication)
    - Flow: 1.5 * flow_weight (Can override all if neighbors are highly trusted)
    """
    candidates = []

    # Add Self
    candidates.append({'val': v_self, 'w': 0.6, 'type': 'self'})

    # Add Peer
    if has_peer:
        candidates.append({'val': v_peer, 'w': 1.2, 'type': 'peer'})

    # Add Flow
    # Flow weight scales with the trust of the neighborhood
    w_flow = 1.5 * flow_weight
    # If neighborhood is untrusted, flow is weak (0.0). If trusted (0.9), flow is strong (1.35).
    candidates.append({'val': v_flow, 'w': w_flow, 'type': 'flow'})

    # Clustering
    clusters = []
    for cand in candidates:
        if cand['val'] is None: continue

        # Try to add to existing cluster
        best_cluster = None
        min_dist = float('inf')

        for cl in clusters:
            # Check compatibility using cluster centroid
            centroid = cl['val_sum'] / cl['w_sum']
            diff = abs(cand['val'] - centroid)
            limit = max(abs(cand['val']), abs(centroid), 1.0) * rel_tol

            if diff <= max(abs_tol, limit):
                if diff < min_dist:
                    min_dist = diff
                    best_cluster = cl

        if best_cluster:
            best_cluster['val_sum'] += cand['val'] * cand['w']
            best_cluster['w_sum'] += cand['w']
            best_cluster['members'].add(cand['type'])
        else:
            clusters.append({
                'val_sum': cand['val'] * cand['w'],
                'w_sum': cand['w'],
                'members': {cand['type']}
            })

    # Pick Winner
    winner = max(clusters, key=lambda c: c['w_sum'])
    final_val = winner['val_sum'] / winner['w_sum']

    # Calculate Confidence
    # Base confidence on who agreed
    m = winner['members']
    if 'peer' in m and 'self' in m:
        conf = 0.95
        if 'flow' in m: conf = 0.98
    elif 'peer' in m and 'flow' in m:
        conf = 0.90
    elif 'self' in m and 'flow' in m:
        conf = 0.85
    elif 'peer' in m:
        conf = 0.70
    elif 'flow' in m:
        # Trust flow only if it was heavily weighted
        conf = 0.6 + (0.3 * flow_weight)
    else:
        conf = 0.5

    return final_val, conf
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