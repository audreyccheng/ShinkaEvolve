# EVOLVE-BLOCK-START
"""
Network telemetry repair algorithm that detects and corrects inconsistencies
in network interface telemetry data using topology relationships.

Takes interface telemetry data and detects/repairs inconsistencies based on
network invariants like link symmetry and flow conservation.
"""
from typing import Dict, Any, Tuple, List


def repair_network_telemetry(telemetry: Dict[str, Dict[str, Any]],
                             topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:
    """
    Repair network interface telemetry by detecting and correcting inconsistencies.

    Strategy:
    1. Infer 'status' based on activity (local or peer).
    2. Iteratively repair 'rx' and 'tx' rates using a Consensus approach.
       - Candidates: Self, Peer, Flow (if valid).
       - Selection: Weighted Median / Closest Cluster to resolve conflicts.
    3. Calculate confidence based on the level of consensus among candidates.
    """

    HARDENING_THRESHOLD = 0.02
    NOISE_FLOOR = 0.1
    ITERATIONS = 5

    # Initialize working state
    state = {}
    for if_id, data in telemetry.items():
        state[if_id] = {
            'rx': float(data.get('rx_rate', 0.0)),
            'tx': float(data.get('tx_rate', 0.0)),
            'status': data.get('interface_status', 'unknown'),
            'peer': data.get('connected_to'),
            'router': data.get('local_router')
        }

    # --- Step 1: Robust Status Repair ---
    for if_id, s in state.items():
        # Evidence for UP
        has_local_traffic = s['rx'] > NOISE_FLOOR or s['tx'] > NOISE_FLOOR

        peer_says_active = False
        peer_id = s['peer']
        if peer_id and peer_id in state:
            p_s = state[peer_id]
            # If peer is sending to me, I must be UP to receive (conceptually)
            # and if peer says UP, I am likely UP
            if p_s['status'] == 'up' and (p_s['tx'] > NOISE_FLOOR or p_s['rx'] > NOISE_FLOOR):
                peer_says_active = True

        if s['status'] == 'down':
            if has_local_traffic or peer_says_active:
                s['status'] = 'up'
        # If status is up but no traffic, we leave it as up (could be idle)

    # Enforce down = zero traffic
    for if_id, s in state.items():
        if s['status'] != 'up':
            s['rx'] = 0.0
            s['tx'] = 0.0

    # --- Step 2: Iterative Rate Repair ---
    for _ in range(ITERATIONS):

        # Build router totals for Flow Conservation
        router_totals = {}
        for r_id, if_list in topology.items():
            # Only calculate flow if we have info for all interfaces (or most)
            valid_ifs = [i for i in if_list if i in state]
            sum_rx = sum(state[i]['rx'] for i in valid_ifs)
            sum_tx = sum(state[i]['tx'] for i in valid_ifs)
            # Flag if complete visibility of router
            router_totals[r_id] = {'rx': sum_rx, 'tx': sum_tx, 'complete': len(valid_ifs) == len(if_list)}

        next_rates = {}

        for if_id, s in state.items():
            if s['status'] != 'up':
                next_rates[if_id] = {'rx': 0.0, 'tx': 0.0}
                continue

            peer_id = s['peer']
            r_id = s['router']

            # --- RX Repair ---
            candidates = []
            # 1. Self
            candidates.append({'val': s['rx'], 'src': 'self'})

            # 2. Peer TX
            if peer_id and peer_id in state:
                candidates.append({'val': state[peer_id]['tx'], 'src': 'peer'})

            # 3. Flow
            if r_id and r_id in router_totals:
                totals = router_totals[r_id]
                if totals['complete']: # Only trust flow if we see whole router
                    flow_val = totals['tx'] - (totals['rx'] - s['rx'])
                    candidates.append({'val': max(0.0, flow_val), 'src': 'flow'})

            # Selection Logic
            vals = [c['val'] for c in candidates]

            if not vals:
                repaired_rx = 0.0
            elif len(vals) == 1:
                repaired_rx = vals[0]
            else:
                # If all agree (low variance), average
                mean_val = sum(vals) / len(vals)
                max_dev = max(abs(v - mean_val) for v in vals)
                if max_dev <= max(mean_val, 1.0) * HARDENING_THRESHOLD:
                    repaired_rx = mean_val
                else:
                    # Disagreement. Use Voting / Median strategy.
                    if len(vals) >= 3:
                        sorted_vals = sorted(vals)
                        repaired_rx = sorted_vals[len(vals) // 2]
                    else:
                        # 2 values (Self, Peer) or (Self, Flow)
                        peer_c = next((c for c in candidates if c['src'] == 'peer'), None)
                        if peer_c:
                            repaired_rx = peer_c['val'] # Trust peer in 1v1
                        else:
                            repaired_rx = s['rx']

            # --- TX Repair ---
            candidates = []
            candidates.append({'val': s['tx'], 'src': 'self'})
            if peer_id and peer_id in state:
                candidates.append({'val': state[peer_id]['rx'], 'src': 'peer'})
            if r_id and r_id in router_totals and router_totals[r_id]['complete']:
                flow_val = router_totals[r_id]['rx'] - (router_totals[r_id]['tx'] - s['tx'])
                candidates.append({'val': max(0.0, flow_val), 'src': 'flow'})

            vals = [c['val'] for c in candidates]
            if not vals:
                repaired_tx = 0.0
            elif len(vals) == 1:
                repaired_tx = vals[0]
            else:
                mean_val = sum(vals) / len(vals)
                max_dev = max(abs(v - mean_val) for v in vals)
                if max_dev <= max(mean_val, 1.0) * HARDENING_THRESHOLD:
                    repaired_tx = mean_val
                else:
                    if len(vals) >= 3:
                        sorted_vals = sorted(vals)
                        repaired_tx = sorted_vals[len(vals) // 2]
                    else:
                        peer_c = next((c for c in candidates if c['src'] == 'peer'), None)
                        if peer_c:
                            repaired_tx = peer_c['val']
                        else:
                            repaired_tx = s['tx']

            next_rates[if_id] = {'rx': repaired_rx, 'tx': repaired_tx}

        # Update state
        for if_id, rates in next_rates.items():
            state[if_id]['rx'] = rates['rx']
            state[if_id]['tx'] = rates['tx']

    # --- Step 3: Confidence Calculation ---
    result = {}
    for if_id, orig_data in telemetry.items():
        final = state[if_id]
        orig_rx = float(orig_data.get('rx_rate', 0.0))
        orig_tx = float(orig_data.get('tx_rate', 0.0))
        orig_status = orig_data.get('interface_status', 'unknown')

        # Re-gather final candidates context
        peer_id = final['peer']
        r_id = final['router']

        def calc_confidence(repaired_val, is_rx):
            supports = []
            # Self support?
            orig_val = orig_rx if is_rx else orig_tx
            if abs(repaired_val - orig_val) <= max(repaired_val, 1.0) * HARDENING_THRESHOLD:
                supports.append('self')

            # Peer support?
            if peer_id and peer_id in state:
                p_val = state[peer_id]['tx'] if is_rx else state[peer_id]['rx']
                if abs(repaired_val - p_val) <= max(repaired_val, 1.0) * HARDENING_THRESHOLD:
                    supports.append('peer')

            # Flow support?
            if r_id and r_id in topology:
                 valid_ifs = [i for i in topology[r_id] if i in state]
                 s_rx = sum(state[i]['rx'] for i in valid_ifs)
                 s_tx = sum(state[i]['tx'] for i in valid_ifs)
                 imbalance = abs(s_rx - s_tx)
                 if imbalance <= max(s_rx, s_tx, 1.0) * HARDENING_THRESHOLD:
                     supports.append('flow')

            # Scoring
            if 'peer' in supports and 'flow' in supports: return 0.95
            if 'peer' in supports and 'self' in supports: return 1.0
            if 'peer' in supports: return 0.85
            if 'self' in supports and 'flow' in supports: return 0.8
            if 'self' in supports: return 0.8 # Fallback
            return 0.5

        rx_conf = calc_confidence(final['rx'], True)
        tx_conf = calc_confidence(final['tx'], False)

        st_conf = 1.0
        if final['status'] != orig_status:
            if final['status'] == 'up':
                st_conf = 0.95
            else:
                st_conf = 0.7

        result[if_id] = {
            'rx_rate': (orig_rx, final['rx'], rx_conf),
            'tx_rate': (orig_tx, final['tx'], tx_conf),
            'interface_status': (orig_status, final['status'], st_conf),
            'connected_to': orig_data.get('connected_to'),
            'local_router': orig_data.get('local_router'),
            'remote_router': orig_data.get('remote_router')
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