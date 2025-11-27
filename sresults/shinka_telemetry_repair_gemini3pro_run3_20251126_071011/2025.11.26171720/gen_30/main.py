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

    Key Improvements:
    1. Trusted Mass Flow Verification: Flow conservation is weighted by the
       trustworthiness of the *other* interfaces on the router. This allows
       isolated/external links to be repaired if the rest of the router is healthy.
    2. Weighted Clustering Arbitration: Selects values based on weighted votes
       from Self (1.0), Peer (1.5), and Flow (Dynamic).
    3. Dynamic Confidence: Confidence evolves based on consensus quality.
    """

    # --- Configuration ---
    REL_TOL = 0.02
    ABS_TOL = 0.5

    # Helper: Agreement Check
    def check_agrees(v1, v2):
        d = abs(v1 - v2)
        return d <= ABS_TOL or (d / max(abs(v1), abs(v2), 1.0)) <= REL_TOL

    results = {}
    working_state = {}

    # --- Phase 1: Initialization & Status Repair ---
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

        # 3. Initial Estimation & Trust
        # Initial Trust is high if Symmetry holds, low otherwise.
        trust_score = 0.5

        if final_status == 'down':
            est_rx, est_tx = 0.0, 0.0
            trust_score = 1.0
        else:
            # RX Estimate
            if has_peer:
                if check_agrees(s_rx, p_tx):
                    est_rx = (s_rx + p_tx) / 2.0
                    trust_score = 1.0
                else:
                    est_rx = p_tx # Trust Peer initially
                    trust_score = 0.8
            else:
                est_rx = s_rx
                trust_score = 0.5 # Isolated

            # TX Estimate
            if has_peer:
                est_tx = (s_tx + p_rx) / 2.0 if check_agrees(s_tx, p_rx) else p_rx
            else:
                est_tx = s_tx

        working_state[if_id] = {
            's_rx': s_rx, 's_tx': s_tx,
            'p_rx': p_rx if has_peer else None,
            'p_tx': p_tx if has_peer else None,
            'est_rx': est_rx, 'est_tx': est_tx,
            'status': final_status, 'status_conf': status_conf,
            'orig_status': s_status,
            'has_peer': has_peer,
            'trust': trust_score
        }

    # --- Phase 2: Iterative Refinement ---
    for iteration in range(2):
        # 1. Calculate Router 'Trusted Mass'
        # We sum up traffic that we are confident in.
        router_stats = {}
        for r_id, if_list in topology.items():
            valid_ifs = [i for i in if_list if i in working_state]

            sum_in, sum_out = 0.0, 0.0
            trusted_in, trusted_out = 0.0, 0.0

            for i in valid_ifs:
                ws = working_state[i]
                # We weight the contribution by trust score
                # To be conservative, only high trust counts as 'verified mass'
                weight = 1.0 if ws['trust'] >= 0.8 else 0.0

                sum_in += ws['est_rx']
                sum_out += ws['est_tx']
                trusted_in += ws['est_rx'] * weight
                trusted_out += ws['est_tx'] * weight

            router_stats[r_id] = {
                'sum_in': sum_in, 'sum_out': sum_out,
                'trusted_in': trusted_in, 'trusted_out': trusted_out
            }

        # 2. Update Estimates
        for if_id, d in working_state.items():
            if d['status'] == 'down': continue

            r_id = telemetry[if_id].get('local_router')

            # Flow Hypothesis
            f_rx, f_tx = d['est_rx'], d['est_tx']
            flow_w_rx, flow_w_tx = 0.0, 0.0 # Weight of flow hypothesis

            if r_id and r_id in router_stats:
                rs = router_stats[r_id]

                # RX Target: Balances Total Out
                # My_RX = Total_Out - Other_In
                other_in = rs['sum_in'] - d['est_rx']
                f_rx = max(0.0, rs['sum_out'] - other_in)

                # Flow Weight Calculation
                # Trust flow if the *rest* of the router is trusted.
                # Specifically, we need to trust Total_Out and Other_In.
                other_trusted_in = rs['trusted_in'] - (d['est_rx'] if d['trust']>=0.8 else 0)

                # Fraction of the equation components that are trusted
                supporting_mass = rs['sum_out'] + other_in
                trusted_mass = rs['trusted_out'] + other_trusted_in

                if supporting_mass > 1.0:
                    flow_w_rx = trusted_mass / supporting_mass
                elif supporting_mass == 0.0:
                    flow_w_rx = 1.0 # Quiet router is trustworthy

                # TX Target
                other_out = rs['sum_out'] - d['est_tx']
                f_tx = max(0.0, rs['sum_in'] - other_out)

                other_trusted_out = rs['trusted_out'] - (d['est_tx'] if d['trust']>=0.8 else 0)
                supporting_mass_tx = rs['sum_in'] + other_out
                trusted_mass_tx = rs['trusted_in'] + other_trusted_out

                if supporting_mass_tx > 1.0:
                    flow_w_tx = trusted_mass_tx / supporting_mass_tx
                elif supporting_mass_tx == 0.0:
                    flow_w_tx = 1.0

            # Arbitrate
            # RX
            val_rx, conf_rx = arbitrate_weighted(
                d['s_rx'], d['p_tx'], f_rx,
                d['has_peer'], flow_w_rx, REL_TOL, ABS_TOL
            )
            d['est_rx'] = 0.5 * d['est_rx'] + 0.5 * val_rx
            d['trust'] = 0.5 * d['trust'] + 0.5 * conf_rx # Update trust for next pass

            # TX
            val_tx, conf_tx = arbitrate_weighted(
                d['s_tx'], d['p_rx'], f_tx,
                d['has_peer'], flow_w_tx, REL_TOL, ABS_TOL
            )
            d['est_tx'] = 0.5 * d['est_tx'] + 0.5 * val_tx
            # Average confidence for the interface trust
            d['trust'] = (d['trust'] + conf_tx) / 2.0

    # --- Phase 3: Final Output ---
    for if_id, d in working_state.items():
        orig_data = telemetry[if_id]
        res = orig_data.copy()

        if d['status'] == 'down':
            crx = 0.95 if d['s_rx'] > ABS_TOL else 1.0
            ctx = 0.95 if d['s_tx'] > ABS_TOL else 1.0
            res['rx_rate'] = (d['s_rx'], 0.0, crx)
            res['tx_rate'] = (d['s_tx'], 0.0, ctx)
        else:
            # Recalculate Final Confidence with final estimates
            # We use the last computed trust/conf from loop
            res['rx_rate'] = (d['s_rx'], d['est_rx'], max(0.5, min(1.0, d['trust'])))
            res['tx_rate'] = (d['s_tx'], d['est_tx'], max(0.5, min(1.0, d['trust'])))

        res['interface_status'] = (d['orig_status'], d['status'], d['status_conf'])
        results[if_id] = res

    return results

def arbitrate_weighted(v_self, v_peer, v_flow, has_peer, flow_weight, rel_tol, abs_tol):
    """
    Weighted Consensus Arbitration.
    """
    candidates = []
    # Self
    candidates.append({'val': v_self, 'w': 1.0, 'src': 'self'})
    # Peer (Stronger than Self)
    if has_peer:
        candidates.append({'val': v_peer, 'w': 1.8, 'src': 'peer'})
    # Flow (Variable strength)
    # If flow_weight is high (e.g. >0.9), it can override Self (w=1.0) and even Peer (w=1.8) if combined with Self
    # If flow_weight is low, it's ignored.
    candidates.append({'val': v_flow, 'w': 2.5 * flow_weight, 'src': 'flow'})

    # Cluster
    clusters = []
    for c in candidates:
        if c['val'] is None: continue
        added = False
        for cl in clusters:
            # Check agreement with cluster centroid
            diff = abs(c['val'] - cl['val'])
            limit = max(abs(c['val']), abs(cl['val']), 1.0) * rel_tol
            if diff <= abs_tol or diff <= limit:
                # Merge
                total_w = cl['w'] + c['w']
                cl['val'] = (cl['val'] * cl['w'] + c['val'] * c['w']) / total_w
                cl['w'] = total_w
                cl['srcs'].add(c['src'])
                added = True
                break
        if not added:
            clusters.append({'val': c['val'], 'w': c['w'], 'srcs': {c['src']}})

    # Pick winner
    best = max(clusters, key=lambda x: x['w'])
    winner_val = best['val']
    srcs = best['srcs']

    # Confidence Logic
    conf = 0.5
    if 'peer' in srcs and 'self' in srcs:
        conf = 1.0
    elif 'peer' in srcs and 'flow' in srcs:
        conf = 0.95
    elif 'self' in srcs and 'flow' in srcs:
        conf = 0.90 if has_peer else 0.85 # If Peer missing, S+F is best we have
    elif 'peer' in srcs:
        conf = 0.8
    elif 'flow' in srcs:
        # Trust flow alone? Only if flow_weight is very high
        conf = 0.6 + (0.3 * flow_weight)

    return winner_val, conf
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