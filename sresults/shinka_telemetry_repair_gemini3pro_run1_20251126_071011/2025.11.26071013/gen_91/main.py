# EVOLVE-BLOCK-START
"""
Network telemetry repair algorithm using Consistency-Based Flow Consensus.
Key Innovations:
1. Consistency Scoring: Evaluates 'Local Consistency' vs 'Remote Consistency' 
   to identify which side of a link fits its local flow constraints, rather than 
   just averaging hints.
2. SNR-Adaptive Confidence: Replaces static heuristic confidence with a continuous 
   Signal-to-Noise Ratio (SNR) curve, assigning higher trust to stronger signals 
   in heuristic scenarios.
"""
from typing import Dict, Any, Tuple, List
import math

def repair_network_telemetry(telemetry: Dict[str, Dict[str, Any]],
                             topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:

    # Constants
    HARDENING_THRESHOLD = 0.02   # 2% tolerance for measurement timing
    TRAFFIC_THRESHOLD = 1.0      # 1 Mbps threshold for "active" link
    ITERATIONS = 5               # Propagation passes

    # --- Phase 1: Initialization & Status Repair ---
    state = {}
    
    # Identify Verifiable Routers (Fully Monitored)
    verifiable_routers = set()
    for r_id, ifaces in topology.items():
        if all(i in telemetry for i in ifaces):
            verifiable_routers.add(r_id)

    for iface_id, data in telemetry.items():
        raw_rx = data.get('rx_rate', 0.0)
        raw_tx = data.get('tx_rate', 0.0)
        raw_status = data.get('interface_status', 'unknown')

        peer_id = data.get('connected_to')
        peer_data = telemetry.get(peer_id) if (peer_id and peer_id in telemetry) else {}

        # Traffic Evidence
        signals = [raw_rx, raw_tx, peer_data.get('rx_rate', 0.0), peer_data.get('tx_rate', 0.0)]
        max_traffic = max(signals) if signals else 0.0

        # Status Inference
        status = raw_status
        status_conf = 1.0

        if max_traffic > TRAFFIC_THRESHOLD:
            if raw_status != 'up':
                status = 'up'
                status_conf = 0.95
        elif raw_status == 'up' and peer_data.get('interface_status') == 'down':
            status = 'down'
            status_conf = 0.8

        # Initial Rate Beliefs
        if status == 'down':
            cur_rx, cur_tx = 0.0, 0.0
        else:
            cur_rx = raw_rx
            cur_tx = raw_tx

        state[iface_id] = {
            'rx': cur_rx,
            'tx': cur_tx,
            'status': status,
            'status_conf': status_conf,
            'orig_rx': raw_rx,
            'orig_tx': raw_tx,
            'orig_status': raw_status,
            'peer_id': peer_id,
            'local_router': data.get('local_router'),
            'remote_router': data.get('remote_router')
        }

    # --- Phase 2: Iterative Consensus ---
    for _ in range(ITERATIONS):
        next_state = {}

        # Calculate Flow Balances
        router_balances = {}
        for r_id in verifiable_routers:
            ifaces = topology[r_id]
            sum_rx = sum(state[i]['rx'] for i in ifaces)
            sum_tx = sum(state[i]['tx'] for i in ifaces)
            router_balances[r_id] = {'rx': sum_rx, 'tx': sum_tx}

        for iface_id, curr in state.items():
            if curr['status'] == 'down':
                next_state[iface_id] = {'rx': 0.0, 'tx': 0.0}
                continue

            peer_id = curr['peer_id']
            has_peer = peer_id and peer_id in state

            # --- Rate Resolution Logic ---
            def resolve_rate(local_val, peer_val, is_rx):
                # 1. Generate Flow Hints
                hint_local = None
                r_id = curr.get('local_router')
                if r_id and r_id in router_balances:
                    rb = router_balances[r_id]
                    # Calc what local_val should be to satisfy Local Flow
                    if is_rx: hint_local = max(0.0, rb['tx'] - (rb['rx'] - local_val))
                    else:     hint_local = max(0.0, rb['rx'] - (rb['tx'] - local_val))

                hint_remote = None
                rr_id = curr.get('remote_router')
                if rr_id and rr_id in router_balances:
                    rb_r = router_balances[rr_id]
                    # Calc what peer_val (which corresponds to local_val) should be
                    # to satisfy Remote Flow.
                    # Note: peer_val is from the perspective of the remote router.
                    if is_rx: 
                        # I am RX. Peer is TX (Out). Remote constraint: Out balances In.
                        hint_remote = max(0.0, rb_r['rx'] - (rb_r['tx'] - peer_val))
                    else:
                        # I am TX. Peer is RX (In). Remote constraint: In balances Out.
                        hint_remote = max(0.0, rb_r['tx'] - (rb_r['rx'] - peer_val))

                hints_avail = [h for h in [hint_local, hint_remote] if h is not None]

                # 2. Symmetry Check
                denom = max(local_val, peer_val, 1.0)
                diff_sym = abs(local_val - peer_val) / denom

                # Double Dead Check (Flow Rescue)
                # If measurements are effectively zero, but hints suggest significant flow.
                avg_val = (local_val + peer_val) / 2.0
                if avg_val < TRAFFIC_THRESHOLD and hints_avail:
                    max_hint = max(hints_avail)
                    if max_hint > 5.0:
                        # Trust the high hints
                        high_hints = [h for h in hints_avail if h > 5.0]
                        return sum(high_hints) / len(high_hints)

                # If Symmetric, trust the average
                if diff_sym <= HARDENING_THRESHOLD:
                    return avg_val

                # 3. Consistency Selection
                # If symmetry is broken, check which measurement is consistent with its own router.
                
                score_local = None
                if hint_local is not None:
                    score_local = abs(local_val - hint_local) / max(local_val, hint_local, 1.0)
                
                score_remote = None
                if hint_remote is not None:
                    score_remote = abs(peer_val - hint_remote) / max(peer_val, hint_remote, 1.0)

                # Thresholds
                CONSISTENT = 0.05
                
                is_cons_l = score_local is not None and score_local < CONSISTENT
                is_cons_r = score_remote is not None and score_remote < CONSISTENT

                # If one is consistent and the other isn't, trust the consistent one.
                # This handles "Broken Router" vs "Healthy Router" scenarios.
                if is_cons_l and not is_cons_r:
                    return local_val
                if is_cons_r and not is_cons_l:
                    return peer_val
                
                # If both consistent (but different) or neither consistent:
                # Use standard target proximity or heuristic.
                
                target = None
                if hints_avail:
                    target = sum(hints_avail) / len(hints_avail)

                if target is not None:
                    # Pick candidate closest to the consensus of flow hints
                    dist_l = abs(local_val - target)
                    dist_p = abs(peer_val - target)
                    return local_val if dist_l < dist_p else peer_val

                # 4. Fallback Heuristic: Trust Non-Zero
                if local_val > TRAFFIC_THRESHOLD and peer_val < TRAFFIC_THRESHOLD:
                    return local_val
                if peer_val > TRAFFIC_THRESHOLD and local_val < TRAFFIC_THRESHOLD:
                    return peer_val
                
                return avg_val

            # Resolve RX
            peer_tx = state[peer_id]['tx'] if has_peer else curr['rx']
            next_rx = resolve_rate(curr['rx'], peer_tx, is_rx=True)

            # Resolve TX
            peer_rx = state[peer_id]['rx'] if has_peer else curr['tx']
            next_tx = resolve_rate(curr['tx'], peer_rx, is_rx=False)

            next_state[iface_id] = {'rx': next_rx, 'tx': next_tx}

        # Apply Updates
        for i_id, vals in next_state.items():
            state[i_id]['rx'] = vals['rx']
            state[i_id]['tx'] = vals['tx']

    # --- Phase 3: Final Calibration ---
    result = {}

    # Recalculate Final Balances
    final_balances = {}
    for r_id in verifiable_routers:
        ifaces = topology[r_id]
        sum_rx = sum(state[i]['rx'] for i in ifaces)
        sum_tx = sum(state[i]['tx'] for i in ifaces)
        final_balances[r_id] = {'rx': sum_rx, 'tx': sum_tx}

    for iface_id, data in state.items():
        final_rx = data['rx']
        final_tx = data['tx']
        peer_id = data['peer_id']
        has_peer = peer_id and peer_id in state

        def get_confidence(val, peer_val, local_r, remote_r, is_rx, is_down, stat_conf):
            if is_down:
                return stat_conf
            
            # 1. Calculate Errors
            err_sym = 1.0
            if has_peer:
                denom = max(val, peer_val, 1.0)
                err_sym = abs(val - peer_val) / denom

            hints = []
            if local_r in final_balances:
                b = final_balances[local_r]
                if is_rx: h = b['tx'] - (b['rx'] - val)
                else:     h = b['rx'] - (b['tx'] - val)
                hints.append(max(0.0, h))
            
            if remote_r in final_balances:
                b = final_balances[remote_r]
                if is_rx: h = b['rx'] - (b['tx'] - peer_val)
                else:     h = b['tx'] - (b['rx'] - peer_val)
                hints.append(max(0.0, h))

            err_flow = None
            if hints:
                target = sum(hints) / len(hints)
                denom = max(val, target, 1.0)
                err_flow = abs(val - target) / denom

            # 2. Assign Confidence
            
            # Case A: Strong Corroboration (Flow & Sym agree)
            if err_flow is not None and err_flow < 0.05 and err_sym < 0.05:
                return 1.0
            
            # Case B: Flow Rescue (Trusted flow overrides broken symmetry)
            if err_flow is not None and err_flow < 0.05:
                return 0.90
            
            # Case C: Symmetry Verified (No flow info available)
            if err_sym < 0.05:
                if err_flow is None: return 0.95
                else: return 0.70 # Conflict between Sym and Flow
            
            # Case D: Heuristic / Unverified
            # Use SNR Curve to prevent overconfidence in noise
            # Range: 0.50 (at 0 Mbps) to 0.85 (at high traffic)
            snr_conf = 0.50 + 0.35 * (1.0 - math.exp(-val / 20.0))
            
            if err_flow is None:
                return snr_conf
            
            # Case E: Conflict (Nothing matches)
            return 0.40

        peer_tx = state[peer_id]['tx'] if has_peer else final_rx
        peer_rx = state[peer_id]['rx'] if has_peer else final_tx

        conf_rx = get_confidence(final_rx, peer_tx, data['local_router'], data['remote_router'], True, data['status']=='down', data['status_conf'])
        conf_tx = get_confidence(final_tx, peer_rx, data['local_router'], data['remote_router'], False, data['status']=='down', data['status_conf'])

        result[iface_id] = {
            'rx_rate': (data['orig_rx'], final_rx, conf_rx),
            'tx_rate': (data['orig_tx'], final_tx, conf_tx),
            'interface_status': (data['orig_status'], data['status'], data['status_conf']),
            'connected_to': peer_id,
            'local_router': data['local_router'],
            'remote_router': data['remote_router']
        }

    return result
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