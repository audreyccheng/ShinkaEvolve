# EVOLVE-BLOCK-START
"""
Iterative Triangulated Consensus Repair
Combines the iterative refinement of the Calibrated Consensus model with the 
multi-perspective voting of Triangulated Consensus. Introduces continuous 
reliability scoring for flow targets to improve confidence calibration.
"""
from typing import Dict, Any, Tuple, List

def repair_network_telemetry(telemetry: Dict[str, Dict[str, Any]],
                             topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:
    """
    Repairs telemetry using an Iterative Triangulated Consensus model.

    Phases:
    1. Initialization: Detects status inconsistencies and seeds initial estimates
       using Link Symmetry (Peer > Self).
    2. Iterative Refinement: Recalculates Flow Conservation targets in a loop.
       Updates estimates using a soft-triangulation arbiter that balances 
       Self, Peer, and Flow signals.
    3. Final Arbitration: Determines final values and confidence scores by 
       classifying the agreement pattern (Unanimous, Symmetry, Peer-Flow, Self-Flow).
    """

    # --- Configuration ---
    REL_TOL = 0.02   # 2% Relative Tolerance
    ABS_TOL = 0.25   # 0.25 Mbps Noise Floor

    results = {}

    # --- Helper: Normalized Distance ---
    def get_dist(v1: float, v2: float) -> float:
        """Returns normalized distance where <= 1.0 implies agreement."""
        diff = abs(v1 - v2)
        if diff <= ABS_TOL: return 0.0
        # Denominator ensures relative comparison
        return diff / (max(abs(v1), abs(v2), 1.0) * REL_TOL)

    # --- Phase 1: Status Normalization & Initialization ---
    working_state = {}

    for if_id, data in telemetry.items():
        # 1. Extract Signals
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

        # 2. Status Repair logic
        # Traffic implies UP. 
        has_traffic = (s_rx > ABS_TOL or s_tx > ABS_TOL or
                       p_rx > ABS_TOL or p_tx > ABS_TOL)

        final_status = s_status
        status_conf = 1.0

        if s_status == 'down' and has_traffic:
            final_status = 'up'
            status_conf = 0.95
        elif s_status == 'up' and not has_traffic:
            # If peer is DOWN, we are likely DOWN
            if p_status == 'down':
                final_status = 'down'
                status_conf = 0.90
            # Else: Up but Idle.

        # 3. Initial Estimation
        # Symmetry > Self. 
        if final_status == 'down':
            est_rx, est_tx = 0.0, 0.0
        else:
            # RX: Peer TX is the strongest predictor
            if has_peer and get_dist(s_rx, p_tx) <= 1.0:
                est_rx = (s_rx + p_tx) / 2.0
            elif has_peer:
                est_rx = p_tx
            else:
                est_rx = s_rx
            
            # TX: Peer RX is the strongest predictor
            if has_peer and get_dist(s_tx, p_rx) <= 1.0:
                est_tx = (s_tx + p_rx) / 2.0
            elif has_peer:
                est_tx = p_rx
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
    # Loop to allow Flow Conservation to propagate and correct local errors
    for iteration in range(2):
        # 1. Calculate Router Totals & Flow Reliability
        router_context = {}
        for r_id, if_list in topology.items():
            valid_ifs = [i for i in if_list if i in working_state]
            t_in = sum(working_state[i]['est_rx'] for i in valid_ifs)
            t_out = sum(working_state[i]['est_tx'] for i in valid_ifs)

            # Reliability Score: 1.0 (perfect) to 0.0 (bad)
            # We tolerate up to 5% imbalance before degrading reliability
            imbalance = abs(t_in - t_out)
            max_traffic = max(t_in, t_out, 1.0)
            
            # Linear decay from 0% to 10% imbalance
            # At 0%, ratio=0 -> score=1.0
            # At 10%, ratio=0.1 -> score=0.0
            ratio = imbalance / max_traffic
            reliability = max(0.0, 1.0 - (ratio * 10.0)) 
            
            router_context[r_id] = {'in': t_in, 'out': t_out, 'reliability': reliability}

        # 2. Update Estimates
        for if_id, d in working_state.items():
            if d['status'] == 'down':
                d['est_rx'], d['est_tx'] = 0.0, 0.0
                continue

            r_id = telemetry[if_id].get('local_router')
            
            # Calculate Flow Targets
            f_rx, f_tx, f_qual = d['est_rx'], d['est_tx'], 0.0
            if r_id and r_id in router_context:
                rc = router_context[r_id]
                f_qual = rc['reliability']
                # RX Target = Out - (In - MyRX)
                other_rx = rc['in'] - d['est_rx']
                f_rx = max(0.0, rc['out'] - other_rx)
                # TX Target = In - (Out - MyTX)
                other_tx = rc['out'] - d['est_tx']
                f_tx = max(0.0, rc['in'] - other_tx)

            # Soft Update
            # We use the arbiter to find the "best" current value, then blend it
            # Blending helps stability across iterations
            target_rx, _ = arbitrate(d['s_rx'], d['p_tx'], f_rx, d['has_peer'], f_qual, REL_TOL, ABS_TOL)
            target_tx, _ = arbitrate(d['s_tx'], d['p_rx'], f_tx, d['has_peer'], f_qual, REL_TOL, ABS_TOL)
            
            d['est_rx'] = 0.6 * d['est_rx'] + 0.4 * target_rx
            d['est_tx'] = 0.6 * d['est_tx'] + 0.4 * target_tx

    # --- Phase 3: Final Arbitration ---
    # Recalculate context one last time with converged estimates
    final_flow_context = {}
    for r_id, if_list in topology.items():
        valid_ifs = [i for i in if_list if i in working_state]
        t_in = sum(working_state[i]['est_rx'] for i in valid_ifs)
        t_out = sum(working_state[i]['est_tx'] for i in valid_ifs)
        ratio = abs(t_in - t_out) / max(t_in, t_out, 1.0)
        reliability = max(0.0, 1.0 - (ratio * 10.0))
        
        for i in valid_ifs:
            ws = working_state[i]
            tgt_rx = max(0.0, t_out - (t_in - ws['est_rx']))
            tgt_tx = max(0.0, t_in - (t_out - ws['est_tx']))
            final_flow_context[i] = {'rx': tgt_rx, 'tx': tgt_tx, 'qual': reliability}

    for if_id, d in working_state.items():
        f_ctx = final_flow_context.get(if_id, {'rx': 0.0, 'tx': 0.0, 'qual': 0.0})

        if d['status'] == 'down':
            final_rx, final_tx = 0.0, 0.0
            conf_rx = 0.95 if d['s_rx'] > ABS_TOL else 1.0
            conf_tx = 0.95 if d['s_tx'] > ABS_TOL else 1.0
        else:
            final_rx, conf_rx = arbitrate(
                d['s_rx'], d['p_tx'], f_ctx['rx'], d['has_peer'], f_ctx['qual'], REL_TOL, ABS_TOL
            )
            final_tx, conf_tx = arbitrate(
                d['s_tx'], d['p_rx'], f_ctx['tx'], d['has_peer'], f_ctx['qual'], REL_TOL, ABS_TOL
            )

        orig_data = telemetry[if_id]
        res = orig_data.copy()
        res['rx_rate'] = (d['s_rx'], final_rx, conf_rx)
        res['tx_rate'] = (d['s_tx'], final_tx, conf_tx)
        res['interface_status'] = (d['orig_status'], d['status'], d['status_conf'])
        results[if_id] = res

    return results

def arbitrate(v_self: float, v_peer: float, v_flow: float, 
              has_peer: bool, flow_qual: float, 
              rel_tol: float, abs_tol: float) -> Tuple[float, float]:
    """
    Triangulated Arbitration: Decides based on consensus of (Self, Peer, Flow).
    Returns (Repaired Value, Confidence Score).
    """
    def get_d(a, b):
        diff = abs(a - b)
        if diff <= abs_tol: return 0.0
        return diff / (max(abs(a), abs(b), 1.0) * rel_tol)

    # 1. Calculate Pairwise Distances (Normalized: <=1.0 means Agreement)
    d_sp = get_d(v_self, v_peer) if has_peer else 999.0
    d_sf = get_d(v_self, v_flow)
    d_pf = get_d(v_peer, v_flow) if has_peer else 999.0
    
    # Flow is only a valid voter if the router context is reliable
    flow_reliable = flow_qual > 0.6 

    # --- Scenario A: Unanimous Consensus (Self ~ Peer ~ Flow) ---
    if d_sp <= 1.0 and d_pf <= 1.0 and flow_reliable:
        # All three agree. Highest confidence.
        avg = (v_self + v_peer + v_flow) / 3.0
        # Confidence decays slightly if agreement is on the edge of tolerance
        conf = 1.0 - (0.05 * max(d_sp, d_pf, d_sf))
        return avg, max(0.9, conf)

    # --- Scenario B: Link Symmetry (Self ~ Peer) ---
    if d_sp <= 1.0:
        # Strong physical signal.
        base_val = (v_self + v_peer) / 2.0
        
        # Sub-case: Flow Disagreement
        if flow_reliable and d_pf > 2.0:
            # Flow contradicts S & P (e.g. Router imbalance elsewhere).
            # Trust S & P, but lower confidence.
            return base_val, 0.85
            
        # Normal Case
        conf = 1.0 - (0.1 * d_sp)
        return base_val, max(0.85, conf)

    # --- Scenario C: Peer Verified by Flow (Peer ~ Flow) ---
    # Implies Self is wrong/noisy.
    if has_peer and d_pf <= 1.0 and flow_reliable:
        avg = (v_peer + v_flow) / 2.0
        # Confidence depends on flow quality
        conf = 0.9 * flow_qual - (0.1 * d_pf)
        return avg, max(0.75, conf)

    # --- Scenario D: Self Verified by Flow (Self ~ Flow) ---
    # Implies Peer is wrong/dead.
    if d_sf <= 1.0 and flow_reliable:
        avg = (v_self + v_flow) / 2.0
        # Slightly lower base confidence than Peer-Flow because Peer is usually better
        conf = 0.85 * flow_qual - (0.1 * d_sf)
        return avg, max(0.70, conf)

    # --- Scenario E: Fallback (Disagreement) ---
    # No consensus found.
    
    # Special Case: Phantom Traffic Check
    # If Flow says 0, and Flow is very reliable, but sensors show noise.
    if flow_reliable and v_flow < abs_tol and flow_qual > 0.8:
        return 0.0, 0.8

    if has_peer:
        # Trust Peer as best single source
        return v_peer, 0.6
    
    if flow_reliable:
        # Trust Flow if decent quality
        return v_flow, 0.6
        
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
