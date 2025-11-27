# EVOLVE-BLOCK-START
"""
Iterative Verified Consensus Repair
Combines the robust "Triangulated Arbitration" of the high-performing model with
the "Verified Mass" flow reliability concept. This allows flow conservation to 
be used more surgically—trusting flow targets only when the supporting data 
(other interfaces) is high-confidence.
"""
from typing import Dict, Any, Tuple, List

def repair_network_telemetry(telemetry: Dict[str, Dict[str, Any]],
                             topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:
    """
    Repairs telemetry using an Iterative Verified Consensus model.

    Key Mechanisms:
    1. Verified Mass: Flow targets are weighed by the confidence of the *other* 
       interfaces contributing to the sum. This prevents bad data from polluting 
       good neighbors via flow conservation.
    2. Triangulated Arbitration: Determines truth by classifying the agreement 
       pattern between Self, Peer, and Flow (Unanimous, Symmetry, Peer-Flow, Self-Flow).
    3. Iterative Refinement: Recalculates flow targets as confidence in the 
       network state evolves.
    """

    # --- Configuration ---
    REL_TOL = 0.02
    ABS_TOL = 0.25
    VERIFIED_THRESHOLD = 0.7  # Confidence threshold to count as "verified mass"

    # --- Helper: Normalized Distance ---
    def get_dist(v1: float, v2: float) -> float:
        """Returns normalized distance where <= 1.0 implies agreement."""
        diff = abs(v1 - v2)
        if diff <= ABS_TOL: return 0.0
        return diff / (max(abs(v1), abs(v2), 1.0) * REL_TOL)

    results = {}
    working_state = {}

    # --- Phase 1: Status Normalization & Initialization ---
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

        # 3. Initial Estimation & Confidence Seeding
        # We perform an initial arbitration check to seed confidence
        if final_status == 'down':
            est_rx, est_tx = 0.0, 0.0
            conf_rx, conf_tx = 1.0, 1.0
        else:
            # RX Estimate
            if has_peer:
                if get_dist(s_rx, p_tx) <= 1.0:
                    est_rx = (s_rx + p_tx) / 2.0
                    conf_rx = 1.0
                else:
                    est_rx = p_tx
                    conf_rx = 0.8 # Trust peer, but note the conflict
            else:
                est_rx = s_rx
                conf_rx = 0.5 # Isolated, low trust

            # TX Estimate
            if has_peer:
                if get_dist(s_tx, p_rx) <= 1.0:
                    est_tx = (s_tx + p_rx) / 2.0
                    conf_tx = 1.0
                else:
                    est_tx = p_rx
                    conf_tx = 0.8
            else:
                est_tx = s_tx
                conf_tx = 0.5

        working_state[if_id] = {
            's_rx': s_rx, 's_tx': s_tx,
            'p_rx': p_rx, 'p_tx': p_tx,
            'est_rx': est_rx, 'est_tx': est_tx,
            'conf_rx': conf_rx, 'conf_tx': conf_tx,
            'status': final_status, 'status_conf': status_conf,
            'orig_status': s_status,
            'has_peer': has_peer
        }

    # --- Phase 2: Iterative Refinement ---
    for iteration in range(2):
        # 1. Calculate Router Stats (Total & Verified Mass)
        router_stats = {}
        for r_id, if_list in topology.items():
            valid_ifs = [i for i in if_list if i in working_state]
            
            sum_in, sum_out = 0.0, 0.0
            ver_in, ver_out = 0.0, 0.0
            
            for i in valid_ifs:
                ws = working_state[i]
                sum_in += ws['est_rx']
                sum_out += ws['est_tx']
                
                # Check if this signal contributes to verified mass
                if ws['conf_rx'] >= VERIFIED_THRESHOLD:
                    ver_in += ws['est_rx']
                if ws['conf_tx'] >= VERIFIED_THRESHOLD:
                    ver_out += ws['est_tx']
            
            router_stats[r_id] = {
                'in': sum_in, 'out': sum_out,
                'v_in': ver_in, 'v_out': ver_out
            }

        # 2. Update Estimates
        for if_id, d in working_state.items():
            if d['status'] == 'down':
                continue

            r_id = telemetry[if_id].get('local_router')
            
            # Default to current estimates (no flow info)
            f_rx, f_tx = d['est_rx'], d['est_tx']
            q_rx, q_tx = 0.0, 0.0 # Flow Quality/Trust
            
            if r_id and r_id in router_stats:
                rs = router_stats[r_id]
                
                # --- RX Flow Calculation ---
                # RX_Target = Total_Out - Other_In
                # Other_In = Total_In - My_RX
                other_in = rs['in'] - d['est_rx']
                f_rx = max(0.0, rs['out'] - other_in)
                
                # Verified Mass Calculation for this specific target
                # Trust = (Verified_Out + Verified_Other_In) / (Total_Out + Total_Other_In)
                # Note: We subtract our own contribution from verification to avoid self-loop
                other_v_in = rs['v_in'] - (d['est_rx'] if d['conf_rx'] >= VERIFIED_THRESHOLD else 0.0)
                
                mass_total = rs['out'] + other_in
                mass_verified = rs['v_out'] + other_v_in
                
                if mass_total > 1.0:
                    q_rx = mass_verified / mass_total
                
                # --- TX Flow Calculation ---
                # TX_Target = Total_In - Other_Out
                other_out = rs['out'] - d['est_tx']
                f_tx = max(0.0, rs['in'] - other_out)
                
                other_v_out = rs['v_out'] - (d['est_tx'] if d['conf_tx'] >= VERIFIED_THRESHOLD else 0.0)
                mass_total_tx = rs['in'] + other_out
                mass_verified_tx = rs['v_in'] + other_v_out
                
                if mass_total_tx > 1.0:
                    q_tx = mass_verified_tx / mass_total_tx

            # Arbitrate using Triangulated Logic
            val_rx, conf_rx = arbitrate(d['s_rx'], d['p_tx'], f_rx, d['has_peer'], q_rx, REL_TOL, ABS_TOL)
            val_tx, conf_tx = arbitrate(d['s_tx'], d['p_rx'], f_tx, d['has_peer'], q_tx, REL_TOL, ABS_TOL)
            
            # Blend values for stability, adopt new confidence
            d['est_rx'] = 0.6 * d['est_rx'] + 0.4 * val_rx
            d['est_tx'] = 0.6 * d['est_tx'] + 0.4 * val_tx
            d['conf_rx'] = conf_rx
            d['conf_tx'] = conf_tx

    # --- Phase 3: Final Output Generation ---
    # Re-calculate stats one last time for the final pass
    final_stats = {}
    for r_id, if_list in topology.items():
        valid_ifs = [i for i in if_list if i in working_state]
        s_in = sum(working_state[i]['est_rx'] for i in valid_ifs)
        s_out = sum(working_state[i]['est_tx'] for i in valid_ifs)
        v_in = sum(working_state[i]['est_rx'] for i in valid_ifs if working_state[i]['conf_rx'] >= VERIFIED_THRESHOLD)
        v_out = sum(working_state[i]['est_tx'] for i in valid_ifs if working_state[i]['conf_tx'] >= VERIFIED_THRESHOLD)
        final_stats[r_id] = {'in': s_in, 'out': s_out, 'v_in': v_in, 'v_out': v_out}

    for if_id, d in working_state.items():
        orig = telemetry[if_id]
        res = orig.copy()
        
        if d['status'] == 'down':
            final_rx, final_tx = 0.0, 0.0
            # If signals are actually quiet, very high confidence. If noisy, slightly less.
            c_rx = 0.95 if d['s_rx'] > ABS_TOL else 1.0
            c_tx = 0.95 if d['s_tx'] > ABS_TOL else 1.0
        else:
            r_id = orig.get('local_router')
            # Derive final flow context
            f_rx, f_tx, q_rx, q_tx = 0.0, 0.0, 0.0, 0.0
            
            if r_id in final_stats:
                rs = final_stats[r_id]
                
                other_in = rs['in'] - d['est_rx']
                f_rx = max(0.0, rs['out'] - other_in)
                mass_total_rx = rs['out'] + other_in
                if mass_total_rx > 1.0:
                    other_v_in = rs['v_in'] - (d['est_rx'] if d['conf_rx'] >= VERIFIED_THRESHOLD else 0)
                    q_rx = (rs['v_out'] + other_v_in) / mass_total_rx
                
                other_out = rs['out'] - d['est_tx']
                f_tx = max(0.0, rs['in'] - other_out)
                mass_total_tx = rs['in'] + other_out
                if mass_total_tx > 1.0:
                    other_v_out = rs['v_out'] - (d['est_tx'] if d['conf_tx'] >= VERIFIED_THRESHOLD else 0)
                    q_tx = (rs['v_in'] + other_v_out) / mass_total_tx

            final_rx, c_rx = arbitrate(d['s_rx'], d['p_tx'], f_rx, d['has_peer'], q_rx, REL_TOL, ABS_TOL)
            final_tx, c_tx = arbitrate(d['s_tx'], d['p_rx'], f_tx, d['has_peer'], q_tx, REL_TOL, ABS_TOL)

        res['rx_rate'] = (d['s_rx'], final_rx, c_rx)
        res['tx_rate'] = (d['s_tx'], final_tx, c_tx)
        res['interface_status'] = (d['orig_status'], d['status'], d['status_conf'])
        
        results[if_id] = res

    return results

def arbitrate(v_self: float, v_peer: float, v_flow: float, 
              has_peer: bool, flow_qual: float, 
              rel_tol: float, abs_tol: float) -> Tuple[float, float]:
    """
    Arbitrates between Self, Peer, and Flow based on agreement distance.
    Uses 'flow_qual' (Verified Mass) to determine if Flow is a valid voter.
    """
    def get_d(a, b):
        diff = abs(a - b)
        if diff <= abs_tol: return 0.0
        return diff / (max(abs(a), abs(b), 1.0) * rel_tol)

    d_sp = get_d(v_self, v_peer) if has_peer else 999.0
    d_sf = get_d(v_self, v_flow)
    d_pf = get_d(v_peer, v_flow) if has_peer else 999.0
    
    # Flow is usable if it is backed by enough verified mass
    flow_usable = flow_qual > 0.6

    # 1. Unanimous (S ≈ P ≈ F)
    if d_sp <= 1.0 and d_pf <= 1.0 and flow_usable:
        return (v_self + v_peer + v_flow)/3.0, max(0.9, 1.0 - 0.05 * max(d_sp, d_pf))

    # 2. Symmetry (S ≈ P)
    if d_sp <= 1.0:
        base = (v_self + v_peer)/2.0
        # If flow is very trusted but disagrees, we doubt slightly, but symmetry wins.
        if flow_usable and d_pf > 2.0:
            return base, 0.85 
        return base, max(0.85, 1.0 - 0.1 * d_sp)

    # 3. Peer ≈ Flow (S is wrong)
    if has_peer and d_pf <= 1.0 and flow_usable:
        # High confidence because two independent sources agree (Remote Peer + Local Router Context)
        conf = 0.8 + (0.15 * flow_qual) - (0.05 * d_pf)
        return (v_peer + v_flow)/2.0, max(0.75, min(1.0, conf))

    # 4. Self ≈ Flow (P is wrong)
    if d_sf <= 1.0 and flow_usable:
        conf = 0.75 + (0.15 * flow_qual) - (0.05 * d_sf)
        return (v_self + v_flow)/2.0, max(0.70, min(1.0, conf))

    # 5. Fallback Scenarios
    
    # Phantom Traffic: Flow says 0 (verified), but sensors show noise.
    if flow_usable and v_flow < abs_tol and flow_qual > 0.8:
        return 0.0, 0.9

    if has_peer:
        return v_peer, 0.6 # Trust Peer over ambiguous Flow
    
    if flow_usable:
        return v_flow, 0.6 # Trust Flow over isolated Self
        
    return v_self, 0.5 # Blind trust
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
