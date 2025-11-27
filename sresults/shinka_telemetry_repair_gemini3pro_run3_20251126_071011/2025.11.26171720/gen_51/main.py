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
    Repairs telemetry using an Iterative Bayesian Consensus model.

    Strategy:
    1. Initial Guess: Based on Link Symmetry, but with agreement checks to avoid
       seeding the system with bad peer data.
    2. Iterative Refinement:
       - Update estimates using Bayesian arbitration (Self vs Peer vs Flow vs Zero).
       - Use momentum (soft updates) to dampen oscillations in circular dependencies.
    3. Final Arbitration: Produces value and confidence score.
    """

    # --- Configuration ---
    REL_TOL = 0.02  # 2% Relative Tolerance
    ABS_TOL = 0.5   # 0.5 Mbps Noise Floor

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
        elif s_status == 'up' and not has_traffic and p_status == 'down':
            final_status = 'down'
            status_conf = 0.90

        # Initial Estimates
        if final_status == 'down':
            est_rx, est_tx = 0.0, 0.0
        else:
            # Initialize with agreement check
            # If Self and Peer agree, average them. If not, favor Peer but don't commit fully.
            def init_val(self_v, peer_v, has_p):
                if not has_p: return self_v
                diff = abs(self_v - peer_v)
                limit = max(self_v, peer_v, 1.0) * REL_TOL
                if diff <= max(ABS_TOL, limit):
                    return (self_v + peer_v) / 2.0
                return peer_v

            est_rx = init_val(s_rx, p_tx, has_peer)
            est_tx = init_val(s_tx, p_rx, has_peer)

        working_state[if_id] = {
            's_rx': s_rx, 's_tx': s_tx,
            'p_rx': p_rx, 'p_tx': p_tx,
            'est_rx': est_rx, 'est_tx': est_tx,
            'status': final_status, 'status_conf': status_conf,
            'orig_status': s_status,
            'has_peer': has_peer
        }

    # --- Phase 2: Iterative Refinement ---
    for iteration in range(2):
        # 1. Calculate Router Totals
        router_totals = {}
        for r_id, if_list in topology.items():
            valid_ifs = [i for i in if_list if i in working_state]
            t_in = sum(working_state[i]['est_rx'] for i in valid_ifs)
            t_out = sum(working_state[i]['est_tx'] for i in valid_ifs)

            imb = abs(t_in - t_out)
            mx = max(t_in, t_out, 1.0)

            # Reliability: 5% imbalance = 0.5 reliability
            reliability = max(0.0, 1.0 - (imb / (mx * 0.10)))
            router_totals[r_id] = {'in': t_in, 'out': t_out, 'reliability': reliability}

        # 2. Update Estimates
        for if_id, d in working_state.items():
            r_id = telemetry[if_id].get('local_router')

            f_rx, f_tx = d['est_rx'], d['est_tx']
            f_trust = 0.0

            if r_id and r_id in router_totals:
                rt = router_totals[r_id]
                f_rx = max(0.0, rt['out'] - (rt['in'] - d['est_rx']))
                f_tx = max(0.0, rt['in'] - (rt['out'] - d['est_tx']))
                f_trust = rt['reliability']

            if d['status'] == 'down':
                 d['est_rx'], d['est_tx'] = 0.0, 0.0
            else:
                # Bayesian Update
                val_rx, _ = arbitrate_bayesian(
                    d['s_rx'], d['p_tx'], f_rx, d['has_peer'], f_trust, REL_TOL, ABS_TOL
                )
                val_tx, _ = arbitrate_bayesian(
                    d['s_tx'], d['p_rx'], f_tx, d['has_peer'], f_trust, REL_TOL, ABS_TOL
                )

                # Momentum (Soft Update)
                d['est_rx'] = 0.5 * d['est_rx'] + 0.5 * val_rx
                d['est_tx'] = 0.5 * d['est_tx'] + 0.5 * val_tx

    # --- Phase 3: Final Arbitration ---
    # Recalculate Flow Context one last time
    router_totals = {}
    for r_id, if_list in topology.items():
        valid_ifs = [i for i in if_list if i in working_state]
        t_in = sum(working_state[i]['est_rx'] for i in valid_ifs)
        t_out = sum(working_state[i]['est_tx'] for i in valid_ifs)
        imb = abs(t_in - t_out)
        mx = max(t_in, t_out, 1.0)
        reliability = max(0.0, 1.0 - (imb / (mx * 0.10)))
        router_totals[r_id] = {'in': t_in, 'out': t_out, 'reliability': reliability}

    for if_id, d in working_state.items():
        res = telemetry[if_id].copy()

        r_id = telemetry[if_id].get('local_router')
        f_rx, f_tx = d['est_rx'], d['est_tx']
        f_trust = 0.0
        if r_id and r_id in router_totals:
            rt = router_totals[r_id]
            f_rx = max(0.0, rt['out'] - (rt['in'] - d['est_rx']))
            f_tx = max(0.0, rt['in'] - (rt['out'] - d['est_tx']))
            f_trust = rt['reliability']

        if d['status'] == 'down':
            crx = 1.0 if d['s_rx'] <= ABS_TOL else 0.95
            ctx = 1.0 if d['s_tx'] <= ABS_TOL else 0.95
            res['rx_rate'] = (d['s_rx'], 0.0, crx)
            res['tx_rate'] = (d['s_tx'], 0.0, ctx)
        else:
            final_rx, conf_rx = arbitrate_bayesian(d['s_rx'], d['p_tx'], f_rx, d['has_peer'], f_trust, REL_TOL, ABS_TOL)
            final_tx, conf_tx = arbitrate_bayesian(d['s_tx'], d['p_rx'], f_tx, d['has_peer'], f_trust, REL_TOL, ABS_TOL)

            res['rx_rate'] = (d['s_rx'], final_rx, conf_rx)
            res['tx_rate'] = (d['s_tx'], final_tx, conf_tx)

        res['interface_status'] = (d['orig_status'], d['status'], d['status_conf'])
        results[if_id] = res

    return results

def arbitrate_bayesian(v_self: float, v_peer: float, v_flow: float,
                       has_peer: bool, flow_trust: float,
                       rel_tol: float, abs_tol: float) -> Tuple[float, float]:
    """
    Selects the best hypothesis using a Gaussian Kernel score.
    Candidates: Self, Peer, Flow, Zero.
    """
    # 1. Define Sources
    sources = []
    sources.append({'val': v_self, 'weight': 0.8, 'type': 'self'})
    if has_peer:
        sources.append({'val': v_peer, 'weight': 1.0, 'type': 'peer'})
    if flow_trust > 0.1:
        sources.append({'val': v_flow, 'weight': 1.0 * flow_trust, 'type': 'flow'})

    # 2. Define Hypotheses
    candidates = {v_self, 0.0}
    if has_peer: candidates.add(v_peer)
    if flow_trust > 0.1: candidates.add(v_flow)

    best_val = v_self
    best_score = -1.0

    # 3. Score
    for h in candidates:
        score = 0.0
        for src in sources:
            diff = abs(h - src['val'])
            sigma = max(abs_tol, src['val'] * rel_tol)
            # Gaussian Kernel
            if diff < 1e-9:
                match = 1.0
            else:
                match = math.exp(-0.5 * (diff / sigma)**2)
            score += src['weight'] * match

        if score > best_score:
            best_score = score
            best_val = h

    # 4. Discrete Confidence Logic (Robust)
    # Check who supports the winner
    supporting = set()
    for src in sources:
        if abs(best_val - src['val']) <= max(abs_tol, src['val']*rel_tol):
             supporting.add(src['type'])

    if 'peer' in supporting and 'self' in supporting:
        conf = 1.0
    elif 'peer' in supporting and 'flow' in supporting:
        conf = 0.95
    elif 'self' in supporting and 'flow' in supporting:
        conf = 0.90
    elif 'peer' in supporting:
        conf = 0.8
    elif 'flow' in supporting:
        conf = 0.7
    else:
        conf = 0.5

    return best_val, conf
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