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
    2. Iterative Refinement (2 Passes):
       - Calculate Flow Targets for each interface based on current estimates of neighbors.
       - Update estimates by arbitrating between Self, Peer, and Flow.
       - This allows Flow Conservation to correct bad initial estimates (e.g., dead peer)
         that initially make the router appear unbalanced.
    3. Final Arbitration: Produces value and confidence score based on consensus quality.
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
        # Note: We rely on iteration to fix these if Peer is wrong but Flow is right.
        if final_status == 'down':
            est_rx, est_tx = 0.0, 0.0
        else:
            # Simple initial guess: Trust Peer if available, else Self
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
    # We refine estimates in a loop to resolve circular dependencies in flow calculation.

    for iteration in range(2):
        # 1. Calculate Router Totals from current estimates
        router_totals = {}
        for r_id, if_list in topology.items():
            valid_ifs = [i for i in if_list if i in working_state]
            t_in = sum(working_state[i]['est_rx'] for i in valid_ifs)
            t_out = sum(working_state[i]['est_tx'] for i in valid_ifs)

            imb = abs(t_in - t_out)
            mx = max(t_in, t_out, 1.0)
            reliable = (imb / mx) < 0.05
            router_totals[r_id] = {'in': t_in, 'out': t_out, 'reliable': reliable}

        # 2. Update Estimates based on Flow Context
        for if_id, d in working_state.items():
            r_id = telemetry[if_id].get('local_router')

            # Calculate Flow Hypothesis
            f_rx, f_tx, f_valid = d['est_rx'], d['est_tx'], False

            if r_id and r_id in router_totals:
                rt = router_totals[r_id]
                # Target RX = Total_Out - (Total_In - My_RX)
                other_rx = rt['in'] - d['est_rx']
                f_rx = max(0.0, rt['out'] - other_rx)

                # Target TX = Total_In - (Total_Out - My_TX)
                other_tx = rt['out'] - d['est_tx']
                f_tx = max(0.0, rt['in'] - other_tx)

                f_valid = rt['reliable']

            # Update Estimates using Arbitration
            if d['status'] == 'down':
                 d['est_rx'], d['est_tx'] = 0.0, 0.0
            else:
                # RX
                val_rx, conf_rx = arbitrate(
                    d['s_rx'], d['p_tx'], f_rx,
                    d['has_peer'], f_valid, REL_TOL, ABS_TOL
                )
                d['est_rx'] = val_rx
                d['conf_rx'] = conf_rx # Store for final pass

                # TX
                val_tx, conf_tx = arbitrate(
                    d['s_tx'], d['p_rx'], f_tx,
                    d['has_peer'], f_valid, REL_TOL, ABS_TOL
                )
                d['est_tx'] = val_tx
                d['conf_tx'] = conf_tx

    # --- Phase 3: Final Result Generation ---
    for if_id, d in working_state.items():
        res = telemetry[if_id].copy()

        if d['status'] == 'down':
            # High confidence if signal is actually quiet
            crx = 1.0 if d['s_rx'] <= ABS_TOL else 0.95
            ctx = 1.0 if d['s_tx'] <= ABS_TOL else 0.95
            res['rx_rate'] = (d['s_rx'], 0.0, crx)
            res['tx_rate'] = (d['s_tx'], 0.0, ctx)
        else:
            res['rx_rate'] = (d['s_rx'], d['est_rx'], d.get('conf_rx', 0.5))
            res['tx_rate'] = (d['s_tx'], d['est_tx'], d.get('conf_tx', 0.5))

        res['interface_status'] = (d['orig_status'], d['status'], d['status_conf'])
        results[if_id] = res

    return results

def arbitrate(v_self: float, v_peer: float, v_flow: float,
              has_peer: bool, flow_valid: bool, rel_tol: float, abs_tol: float) -> Tuple[float, float]:
    """
    Arbitrates between signals using goodness-of-fit to determine confidence.
    Prioritizes consensus (e.g. Self+Flow or Peer+Flow) over reliability flags.
    """

    def dist(a, b):
        if a is None or b is None: return 999.0
        diff = abs(a - b)
        if diff <= abs_tol: return 0.0
        return diff / max(abs(a), abs(b), 1.0) / rel_tol

    # Calculate Distances (0.0 = Perfect match, <=1.0 = Agree)
    d_sp = dist(v_self, v_peer) if has_peer else 999.0
    d_sf = dist(v_self, v_flow)
    d_pf = dist(v_peer, v_flow) if has_peer else 999.0

    # 1. Unanimous Agreement (S ≈ P ≈ F)
    # Strongest possible signal.
    if d_sp <= 1.0 and d_pf <= 1.0:
         return (v_self + v_peer + v_flow)/3.0, max(0.9, 1.0 - 0.05 * max(d_sp, d_pf))

    # 2. Self == Peer (Flow Disagrees or Invalid)
    # Link symmetry is the physical baseline.
    if d_sp <= 1.0:
         return (v_self + v_peer)/2.0, max(0.8, 0.95 - 0.1 * d_sp)

    # 3. Peer == Flow (Self Disagrees)
    # Strong evidence that local sensor is broken.
    if d_pf <= 1.0:
         return (v_peer + v_flow)/2.0, max(0.8, 0.95 - 0.1 * d_pf)

    # 4. Self == Flow (Peer Disagrees)
    # Evidence that Peer is dead/wrong, provided neighbors are consistent.
    # We trust this even if 'flow_valid' was loose, because exact numerical match is rare by chance.
    if d_sf <= 1.0:
         return (v_self + v_flow)/2.0, max(0.7, 0.90 - 0.1 * d_sf)

    # 5. Fallbacks (No Consensus)
    if has_peer:
         return v_peer, 0.6 # Trust Symmetry over unknown Flow
    elif flow_valid:
         # External Link with valid flow context
         # If Self is close to Flow (relaxed), trust Self verified by Flow
         if d_sf <= 2.0: return v_self, 0.8
         return v_flow, 0.75 # Trust Flow
    else:
         return v_self, 0.5 # Blind Trust
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
