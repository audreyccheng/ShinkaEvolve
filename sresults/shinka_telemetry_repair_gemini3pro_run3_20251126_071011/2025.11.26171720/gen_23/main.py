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
    Repairs telemetry using an Iterative Consensus model.

    Improvements:
    1. Iterative Solver: Runs 3 passes of estimate refinement to allow flow conservation
       logic to stabilize and isolate faulty counters.
    2. Soft Flow Reliability: Uses continuous reliability scores for flow targets instead
       of binary valid/invalid flags.
    3. Continuous Confidence: Returns confidence scores proportional to the goodness-of-fit
       of the chosen solution, improving calibration.
    4. Null Hypothesis: Explicitly checks for 'phantom traffic' (Flow≈0) scenarios.
    """

    # --- Configuration ---
    REL_TOL = 0.02
    ABS_TOL = 0.5

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
    ITERATIONS = 3

    for _ in range(ITERATIONS):
        # 1. Calculate Router Context
        router_stats = {}
        for r_id, if_list in topology.items():
            valid_ifs = [i for i in if_list if i in working_state]
            t_in = sum(working_state[i]['est_rx'] for i in valid_ifs)
            t_out = sum(working_state[i]['est_tx'] for i in valid_ifs)

            # Reliability score based on current imbalance (Soft Gating)
            imb = abs(t_in - t_out)
            mx = max(t_in, t_out, 1.0)
            reliability = 1.0 / (1.0 + (imb / (mx * 0.05))**2)

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
                # TX_i = Total_In - (Total_Out - TX_i)
                f_tx = max(0.0, rs['in'] - (rs['out'] - d['est_tx']))
                # RX_i = Total_Out - (Total_In - RX_i)
                f_rx = max(0.0, rs['out'] - (rs['in'] - d['est_rx']))
                f_qual = rs['rel']

            # Arbitration with Momentum
            target_rx, _ = solve_triangulation(
                d['s_rx'], d['p_tx'], f_rx,
                d['has_peer'], f_qual, REL_TOL, ABS_TOL
            )

            target_tx, _ = solve_triangulation(
                d['s_tx'], d['p_rx'], f_tx,
                d['has_peer'], f_qual, REL_TOL, ABS_TOL
            )

            # Soft Update
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
        final_router_stats[r_id] = {'in': t_in, 'out': t_out, 'rel': 1.0 / (1.0 + (imb/(mx*0.05))**2)}

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
             final_rx, conf_rx = solve_triangulation(
                 d['s_rx'], d['p_tx'], f_rx,
                 d['has_peer'], f_qual, REL_TOL, ABS_TOL
             )
             final_tx, conf_tx = solve_triangulation(
                 d['s_tx'], d['p_rx'], f_tx,
                 d['has_peer'], f_qual, REL_TOL, ABS_TOL
             )
             res_entry['rx_rate'] = (d['s_rx'], final_rx, conf_rx)
             res_entry['tx_rate'] = (d['s_tx'], final_tx, conf_tx)

        res_entry['interface_status'] = (d['orig_status'], d['status'], d['status_conf'])
        results[if_id] = res_entry

    return results

def solve_triangulation(v_self, v_peer, v_flow, has_peer, flow_qual, rel_tol, abs_tol):
    """
    Arbitrates between Self, Peer, and Flow.
    Returns (value, confidence).
    """
    # Normalized Distance
    def d(a, b):
        if a is None or b is None: return 10.0
        diff = abs(a - b)
        if diff <= abs_tol: return 0.0
        return diff / (max(abs(a), abs(b), 1.0) * rel_tol)

    d_sp = d(v_self, v_peer) if has_peer else 10.0
    d_sf = d(v_self, v_flow)
    d_pf = d(v_peer, v_flow) if has_peer else 10.0

    # 1. Unanimous (S ~= P ~= F)
    if d_sp <= 1.0 and d_pf <= 1.0:
        val = (v_self + v_peer + v_flow) / 3.0
        conf = 1.0 - 0.05 * max(d_sp, d_pf, d_sf)
        return val, max(0.9, conf)

    # 2. Symmetry (S ~= P)
    if d_sp <= 1.0:
        val = (v_self + v_peer) / 2.0
        conf = 0.95 - 0.1 * d_sp
        # Penalty if flow contradicts strongly and is reliable
        if flow_qual > 0.8 and d_pf > 2.0:
            conf -= 0.1
        return val, max(0.8, conf)

    # 3. Peer-Flow Agreement
    if d_pf <= 1.0:
        val = (v_peer + v_flow) / 2.0
        base = 0.85 + 0.1 * flow_qual
        conf = base - 0.1 * d_pf
        return val, max(0.7, conf)

    # 4. Self-Flow Agreement
    if d_sf <= 1.0:
        val = (v_self + v_flow) / 2.0
        base = 0.80 + 0.1 * flow_qual
        conf = base - 0.1 * d_sf
        return val, max(0.7, conf)

    # 5. No Consensus
    # Check for Phantom Traffic (Flow says 0)
    if flow_qual > 0.7 and v_flow < abs_tol:
        return 0.0, 0.8

    if has_peer:
        return v_peer, 0.6
    elif flow_qual > 0.6:
        return v_flow, 0.6
    else:
        return v_self, 0.5
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
