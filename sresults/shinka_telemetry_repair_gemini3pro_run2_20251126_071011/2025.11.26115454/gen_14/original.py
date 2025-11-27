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

    Uses an iterative consensus algorithm based on:
    1. Link Symmetry: TX(A) ≈ RX(B)
    2. Flow Conservation: Σ RX(Router) ≈ Σ TX(Router)
    3. Status Consistency
    """

    HARDENING_THRESHOLD = 0.02
    NOISE_FLOOR = 0.1

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
                s['status'] = 'up' # Repair status
        elif s['status'] == 'up':
            # If completely silent and peer is down/silent, maybe down?
            # But stick to original unless strong evidence
            pass

    # Enforce down = zero traffic
    for if_id, s in state.items():
        if s['status'] != 'up':
            s['rx'] = 0.0
            s['tx'] = 0.0

    # --- Step 2: Iterative Rate Repair ---
    # We iterate to allow flow conservation to propagate corrections
    for _ in range(3):

        # Calculate Router Flow Totals
        router_totals = {}
        for r_id, if_list in topology.items():
            sum_rx = sum(state[i]['rx'] for i in if_list if i in state)
            sum_tx = sum(state[i]['tx'] for i in if_list if i in state)
            router_totals[r_id] = {'rx': sum_rx, 'tx': sum_tx}

        next_rates = {}

        for if_id, s in state.items():
            if s['status'] != 'up':
                next_rates[if_id] = {'rx': 0.0, 'tx': 0.0}
                continue

            # --- RX Repair ---
            # Candidates:
            # 1. Self RX (s['rx'])
            # 2. Peer TX (peer['tx'])
            # 3. Flow RX (Total_TX - (Total_RX - Self_RX))

            candidates_rx = [{'val': s['rx'], 'src': 'self'}]

            peer_id = s['peer']
            if peer_id and peer_id in state:
                candidates_rx.append({'val': state[peer_id]['tx'], 'src': 'peer'})

            r_id = s['router']
            if r_id and r_id in router_totals:
                totals = router_totals[r_id]
                # Flow balance: RX_this + RX_others = TX_total
                # RX_this = TX_total - RX_others = TX_total - (Total_RX - RX_this)
                flow_val = totals['tx'] - (totals['rx'] - s['rx'])
                if flow_val >= 0:
                     candidates_rx.append({'val': flow_val, 'src': 'flow'})

            # Decision Logic for RX
            # If peer is available, it's the strongest check.
            best_rx = s['rx']

            # Find peer candidate
            peer_val = next((c['val'] for c in candidates_rx if c['src'] == 'peer'), None)
            flow_val = next((c['val'] for c in candidates_rx if c['src'] == 'flow'), None)

            if peer_val is not None:
                # Symmetry Check
                diff = abs(s['rx'] - peer_val)
                denom = max(s['rx'], peer_val, 1.0)

                if diff / denom < HARDENING_THRESHOLD:
                    # Agree
                    best_rx = (s['rx'] + peer_val) / 2.0
                else:
                    # Disagree. Use Flow as tiebreaker if available
                    if flow_val is not None:
                        diff_self_flow = abs(s['rx'] - flow_val)
                        diff_peer_flow = abs(peer_val - flow_val)

                        # If peer is much closer to flow expectation than self is
                        if diff_peer_flow < diff_self_flow * 0.5:
                            best_rx = peer_val
                        # If self is much closer to flow
                        elif diff_self_flow < diff_peer_flow * 0.5:
                            best_rx = s['rx']
                        else:
                            # Ambiguous. Trust peer slightly more for RX/TX symmetry?
                            # Research says inputs are often wrong.
                            # If we have to choose, usually peer tx is a better indicator of what was sent
                            # than rx is of what was received (if drops occur).
                            # But here we assume no drops.
                            # Let's trust peer.
                            best_rx = peer_val
                    else:
                        # No flow info. Trust peer (Symmetry is R3)
                        best_rx = peer_val

            # --- TX Repair ---
            # Candidates: Self TX, Peer RX, Flow TX
            candidates_tx = [{'val': s['tx'], 'src': 'self'}]
            if peer_id and peer_id in state:
                candidates_tx.append({'val': state[peer_id]['rx'], 'src': 'peer'})

            if r_id and r_id in router_totals:
                totals = router_totals[r_id]
                # Flow balance: TX_this + TX_others = RX_total
                flow_val = totals['rx'] - (totals['tx'] - s['tx'])
                if flow_val >= 0:
                    candidates_tx.append({'val': flow_val, 'src': 'flow'})

            best_tx = s['tx']
            peer_val = next((c['val'] for c in candidates_tx if c['src'] == 'peer'), None)
            flow_val = next((c['val'] for c in candidates_tx if c['src'] == 'flow'), None)

            if peer_val is not None:
                diff = abs(s['tx'] - peer_val)
                denom = max(s['tx'], peer_val, 1.0)

                if diff / denom < HARDENING_THRESHOLD:
                    best_tx = (s['tx'] + peer_val) / 2.0
                else:
                    if flow_val is not None:
                        diff_self_flow = abs(s['tx'] - flow_val)
                        diff_peer_flow = abs(peer_val - flow_val)
                        if diff_peer_flow < diff_self_flow * 0.5:
                            best_tx = peer_val
                        elif diff_self_flow < diff_peer_flow * 0.5:
                            best_tx = s['tx']
                        else:
                            best_tx = peer_val
                    else:
                        best_tx = peer_val

            next_rates[if_id] = {'rx': best_rx, 'tx': best_tx}

        # Apply updates for next iteration
        for if_id, rates in next_rates.items():
            state[if_id]['rx'] = rates['rx']
            state[if_id]['tx'] = rates['tx']

    # --- Step 3: Confidence Calibration ---
    result = {}
    for if_id, orig_data in telemetry.items():
        final = state[if_id]
        orig_rx = float(orig_data.get('rx_rate', 0.0))
        orig_tx = float(orig_data.get('tx_rate', 0.0))
        orig_status = orig_data.get('interface_status', 'unknown')

        # Helper to calc confidence
        def get_conf(orig, repaired, peer_val, flow_val):
            # If we didn't change much, high confidence unless peer disagrees strongly
            if abs(orig - repaired) < max(orig, 1.0) * HARDENING_THRESHOLD:
                if peer_val is not None and abs(repaired - peer_val) > max(repaired, 1.0) * 0.1:
                    return 0.8 # We kept value but peer disagrees
                return 1.0

            # We changed value. Justification?
            score = 0.5 # base

            # Peer agreement?
            if peer_val is not None and abs(repaired - peer_val) < max(repaired, 1.0) * 0.05:
                score += 0.3

            # Flow agreement?
            if flow_val is not None and abs(repaired - flow_val) < max(repaired, 1.0) * 0.05:
                score += 0.2

            return min(1.0, score)

        # Get aux values for confidence
        peer_id = final['peer']
        peer_tx = state[peer_id]['tx'] if (peer_id and peer_id in state) else None
        peer_rx = state[peer_id]['rx'] if (peer_id and peer_id in state) else None

        r_id = final['router']
        flow_rx = None
        flow_tx = None
        if r_id and r_id in topology:
             # Recalc finals
             s_rx = sum(state[i]['rx'] for i in topology[r_id] if i in state)
             s_tx = sum(state[i]['tx'] for i in topology[r_id] if i in state)
             # flow implied rx = s_tx - (s_rx - my_rx)
             flow_rx = s_tx - (s_rx - final['rx'])
             flow_tx = s_rx - (s_tx - final['tx'])

        rx_conf = get_conf(orig_rx, final['rx'], peer_tx, flow_rx)
        tx_conf = get_conf(orig_tx, final['tx'], peer_rx, flow_tx)

        # Status confidence
        st_conf = 1.0
        if final['status'] != orig_status:
            # We changed status.
            # If we have traffic, we are very confident it is UP
            if final['rx'] > 1.0 or final['tx'] > 1.0:
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
