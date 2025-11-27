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

    Key Improvements:
    1. Flow Trust Scoring: Calculates flow reliability based on the 'verified mass' of
       neighboring interfaces, allowing repair of external links if the rest of the router is stable.
    2. Hypothesis Selection: Arbitrates between Self, Peer, and Flow using a
       weighted consensus approach, including a 'Zero' hypothesis for phantom traffic.
    3. Gaussian Confidence: Uses exponential decay for confidence calibration based on agreement quality.
    """

    # --- Configuration ---
    REL_TOL = 0.02
    ABS_TOL = 0.5

    # Helper: Agreement Check
    def check_agrees(v1, v2):
        if v1 is None or v2 is None: return False
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

        # 3. Initial Estimation & Confidence
        # We assign an initial 'trust' score to our estimate to help Flow calculation later.
        if final_status == 'down':
            est_rx, est_tx = 0.0, 0.0
            cur_conf = 1.0
        else:
            # RX Estimate
            if has_peer:
                if check_agrees(s_rx, p_tx):
                    est_rx = (s_rx + p_tx) / 2.0
                    cur_conf = 1.0
                else:
                    est_rx = p_tx
                    cur_conf = 0.8 # Trust Peer, but conflict exists
            else:
                est_rx = s_rx
                cur_conf = 0.5 # Low trust in isolated self

            # TX Estimate (same logic)
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
            'conf': cur_conf
        }

    # --- Phase 2: Iterative Refinement ---
    for iteration in range(2):
        # 1. Calculate Router Stats (Total Traffic & Verified Traffic)
        router_stats = {}
        for r_id, if_list in topology.items():
            valid_ifs = [i for i in if_list if i in working_state]

            sum_in, sum_out = 0.0, 0.0
            ver_in, ver_out = 0.0, 0.0

            for i in valid_ifs:
                ws = working_state[i]
                c_weight = ws['conf']
                # Weighted contribution to 'verified mass'
                # If confidence is high (e.g. >0.7), we treat it as verified mass
                is_trusted = 1.0 if c_weight >= 0.7 else 0.0

                sum_in += ws['est_rx']
                sum_out += ws['est_tx']
                ver_in += ws['est_rx'] * is_trusted
                ver_out += ws['est_tx'] * is_trusted

            router_stats[r_id] = {
                'sum_in': sum_in, 'sum_out': sum_out,
                'ver_in': ver_in, 'ver_out': ver_out
            }

        # 2. Update Estimates
        for if_id, d in working_state.items():
            if d['status'] == 'down':
                continue # Stay 0.0

            r_id = telemetry[if_id].get('local_router')

            # Flow Hypothesis
            f_rx, f_tx = d['est_rx'], d['est_tx']
            flow_trust_rx, flow_trust_tx = 0.0, 0.0

            if r_id and r_id in router_stats:
                rs = router_stats[r_id]

                # RX Target: Balances Total Out
                # My_RX = Total_Out - Other_In
                # Other_In = Total_In - My_Est_RX
                other_in = rs['sum_in'] - d['est_rx']
                f_rx = max(0.0, rs['sum_out'] - other_in)

                # Calculate Trust in this calculation
                # Trust depends on: Trusted_Out + Trusted_Other_In
                other_ver_in = rs['ver_in'] - (d['est_rx'] if d['conf'] >= 0.7 else 0)
                supporting_mass = rs['sum_out'] + other_in
                verified_mass = rs['ver_out'] + other_ver_in

                if supporting_mass > 1.0:
                    flow_trust_rx = verified_mass / supporting_mass

                # TX Target
                other_out = rs['sum_out'] - d['est_tx']
                f_tx = max(0.0, rs['sum_in'] - other_out)

                other_ver_out = rs['ver_out'] - (d['est_tx'] if d['conf'] >= 0.7 else 0)
                supporting_mass_tx = rs['sum_in'] + other_out
                verified_mass_tx = rs['ver_in'] + other_ver_out

                if supporting_mass_tx > 1.0:
                    flow_trust_tx = verified_mass_tx / supporting_mass_tx

            # Arbitrate and Soft Update
            # Update RX
            val_rx, conf_rx = arbitrate_signals(
                d['s_rx'], d['p_tx'], f_rx,
                d['has_peer'], flow_trust_rx, REL_TOL, ABS_TOL
            )
            d['est_rx'] = 0.5 * d['est_rx'] + 0.5 * val_rx

            # Update TX
            val_tx, conf_tx = arbitrate_signals(
                d['s_tx'], d['p_rx'], f_tx,
                d['has_peer'], flow_trust_tx, REL_TOL, ABS_TOL
            )
            d['est_tx'] = 0.5 * d['est_tx'] + 0.5 * val_tx

            # Update Confidence (Average)
            avg_conf = (conf_rx + conf_tx) / 2.0
            d['conf'] = 0.5 * d['conf'] + 0.5 * avg_conf

    # --- Phase 3: Final Output ---
    for if_id, d in working_state.items():
        orig_data = telemetry[if_id]
        res = orig_data.copy()

        if d['status'] == 'down':
            # Signal noise check
            crx = 0.95 if d['s_rx'] > ABS_TOL else 1.0
            ctx = 0.95 if d['s_tx'] > ABS_TOL else 1.0
            res['rx_rate'] = (d['s_rx'], 0.0, crx)
            res['tx_rate'] = (d['s_tx'], 0.0, ctx)
        else:
            final_rx, final_tx = d['est_rx'], d['est_tx']

            # Final Confidence Calculation based on Goodness-of-Fit
            # RX Confidence
            dist_p = abs(final_rx - d['p_tx']) if d['p_tx'] is not None else 999.0
            if dist_p < ABS_TOL or (dist_p/max(final_rx, 1.0) < REL_TOL):
                c_rx = 1.0 - min(0.2, dist_p/max(final_rx, 1.0))
            elif d['has_peer']:
                 c_rx = 0.8 # Disagreement with peer
            else:
                 c_rx = 0.8 if d['conf'] > 0.7 else 0.5

            # TX Confidence
            dist_p_tx = abs(final_tx - d['p_rx']) if d['p_rx'] is not None else 999.0
            if dist_p_tx < ABS_TOL or (dist_p_tx/max(final_tx, 1.0) < REL_TOL):
                c_tx = 1.0 - min(0.2, dist_p_tx/max(final_tx, 1.0))
            elif d['has_peer']:
                 c_tx = 0.8
            else:
                 c_tx = 0.8 if d['conf'] > 0.7 else 0.5

            res['rx_rate'] = (d['s_rx'], final_rx, max(0.5, min(1.0, c_rx)))
            res['tx_rate'] = (d['s_tx'], final_tx, max(0.5, min(1.0, c_tx)))

        res['interface_status'] = (d['orig_status'], d['status'], d['status_conf'])
        results[if_id] = res

    return results

def arbitrate_signals(v_self: float, v_peer: float, v_flow: float,
                      has_peer: bool, flow_trust: float,
                      rel_tol: float, abs_tol: float) -> Tuple[float, float]:
    """
    Selects the best value among Self, Peer, and Flow based on consistency and trust.
    Returns (best_value, confidence).
    """

    # Define Source Weights
    W_SELF = 1.0
    W_PEER = 2.0
    W_FLOW = 2.5 * flow_trust

    clusters = [] # List of {'val': float, 'weight': float, 'sources': []}

    def add_vote(val, weight, source):
        if val is None: return
        merged = False
        for c in clusters:
            if abs(c['val'] - val) <= abs_tol or \
               abs(c['val'] - val) / max(abs(c['val']), 1.0) <= rel_tol:
                total_w = c['weight'] + weight
                c['val'] = (c['val'] * c['weight'] + val * weight) / total_w
                c['weight'] = total_w
                c['sources'].append(source)
                merged = True
                break
        if not merged:
            clusters.append({'val': val, 'weight': weight, 'sources': [source]})

    add_vote(v_self, W_SELF, 'self')
    if has_peer:
        add_vote(v_peer, W_PEER, 'peer')
    add_vote(v_flow, W_FLOW, 'flow')

    # Select Winner
    best_c = max(clusters, key=lambda x: x['weight'])
    winner_val = best_c['val']

    # Calculate Confidence based on source composition
    sources = best_c['sources']
    conf = 0.5

    if 'peer' in sources and 'self' in sources:
        conf = 1.0
    elif 'peer' in sources and 'flow' in sources:
        conf = 0.95
    elif 'self' in sources and 'flow' in sources:
        conf = 0.85 if has_peer else 0.90
    elif 'peer' in sources:
        conf = 0.8
    elif 'flow' in sources:
        conf = 0.6 + (0.2 * flow_trust)
    else:
        conf = 0.5

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