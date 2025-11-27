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
    Repair network interface telemetry using a Physical Constraint & Flow Validation approach.

    Key Strategies:
    1. **Physical Status Logic**:
       - If Peer is DOWN, Local MUST be DOWN (physically impossible to link).
       - If Traffic > Noise, Interface MUST be UP.
    2. **Symmetry-First Rate Repair with Flow Veto**:
       - Primary Truth: Peer's counter (Link Symmetry).
       - Sanity Check: Does adopting Peer's value improve/maintain Flow Conservation at the router?
       - If Peer's value creates massive Flow Imbalance, reject it (Peer likely corrupted).
    3. **Calibrated Confidence**:
       - Confidence scores derived from the agreement between Original, Peer, and Flow signals.
       - Uses a 'Noise Floor' to prevent over-reacting to small variances in low-bandwidth links.
    """

    # Configuration
    HARDENING_THRESHOLD = 0.02
    MIN_RATE_THRESHOLD = 10.0  # Mbps, noise floor for normalization
    ITERATIONS = 2             # Convergence iterations

    # --- Initialize State ---
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
    # Apply physical constraints to status
    for if_id, s in state.items():
        # Check Peer Status (if available)
        peer_id = s['peer']
        peer_is_down = False
        if peer_id and peer_id in telemetry:
            if telemetry[peer_id].get('interface_status') == 'down':
                peer_is_down = True

        # Check Traffic Activity
        # We consider 'active' if rates > threshold
        is_active = s['rx'] > MIN_RATE_THRESHOLD or s['tx'] > MIN_RATE_THRESHOLD

        # Decision
        if peer_is_down:
            # Physical Constraint: Cannot be UP if connected to DOWN
            s['status'] = 'down'
        elif is_active:
            # Physical Observation: Cannot be DOWN if passing traffic
            s['status'] = 'up'

        # Enforce consistency
        if s['status'] != 'up':
            s['rx'] = 0.0
            s['tx'] = 0.0

    # --- Step 2: Iterative Rate Repair ---
    # Helper to calculate normalized difference
    def get_diff(v1, v2):
        return abs(v1 - v2) / max(v1, v2, MIN_RATE_THRESHOLD)

    for _ in range(ITERATIONS):
        next_state = {}

        # Calculate Router Flow States
        router_flows = {}
        for r_id, if_ids in topology.items():
            r_rx = sum(state[i]['rx'] for i in if_ids if i in state)
            r_tx = sum(state[i]['tx'] for i in if_ids if i in state)
            router_flows[r_id] = {'rx': r_rx, 'tx': r_tx}

        for if_id, s in state.items():
            if s['status'] != 'up':
                next_state[if_id] = {'rx': 0.0, 'tx': 0.0}
                continue

            peer_id = s['peer']
            r_id = s['router']
            has_peer = peer_id and peer_id in state

            curr_rx = s['rx']
            curr_tx = s['tx']

            # --- RX Repair ---
            # Constraint: RX <= Peer TX (Physical).
            final_rx = curr_rx
            if has_peer:
                peer_tx = state[peer_id]['tx']
                diff = get_diff(curr_rx, peer_tx)

                if diff <= HARDENING_THRESHOLD:
                     final_rx = (curr_rx + peer_tx) / 2.0
                else:
                    # Disagreement.
                    # If RX > Peer TX (Impossible), prefer Peer TX unless Flow demands RX.
                    if curr_rx > peer_tx * (1.0 + HARDENING_THRESHOLD):
                        keep_self = False
                        if r_id in router_flows:
                            # If we have a deficit (TX > RX), keeping high RX helps balance.
                            # If we have surplus (RX > TX), keeping high RX hurts.
                            f = router_flows[r_id]
                            if f['tx'] > f['rx']: keep_self = True

                        final_rx = curr_rx if keep_self else peer_tx
                    else:
                        # RX < Peer TX (Possible Loss or Error).
                        # Use Flow to Arbitrate.
                        use_peer = True
                        if r_id in router_flows:
                            f = router_flows[r_id]
                            # Current Imbalance (with curr_rx)
                            imb_curr = abs(f['rx'] - f['tx'])
                            # Proposed Imbalance (with peer_tx) -> Total RX increases
                            imb_peer = abs((f['rx'] - curr_rx + peer_tx) - f['tx'])
                            if imb_curr < imb_peer:
                                use_peer = False

                        final_rx = peer_tx if use_peer else curr_rx

            # --- TX Repair ---
            # Constraint: TX >= Peer RX (Physical).
            final_tx = curr_tx
            if has_peer:
                peer_rx = state[peer_id]['rx']
                diff = get_diff(curr_tx, peer_rx)

                if diff <= HARDENING_THRESHOLD:
                    final_tx = (curr_tx + peer_rx) / 2.0
                else:
                    # Disagreement.
                    # If TX < Peer RX (Impossible), must repair up to Peer RX.
                    if curr_tx < peer_rx * (1.0 - HARDENING_THRESHOLD):
                        final_tx = peer_rx
                    else:
                        # TX > Peer RX (Packet Loss?).
                        # Only keep high TX if Flow supports it (we have enough RX to send this).
                        keep_self = False
                        if r_id in router_flows:
                            f = router_flows[r_id]
                            # Surplus (RX >= TX) supports high TX.
                            # Deficit (TX > RX) suggests TX is too high (phantom traffic).
                            if f['rx'] >= f['tx']:
                                keep_self = True
                            else:
                                # Check if lowering TX to Peer RX improves balance
                                imb_curr = abs(f['rx'] - f['tx'])
                                imb_peer = abs(f['rx'] - (f['tx'] - curr_tx + peer_rx))
                                if imb_curr < imb_peer:
                                    keep_self = True

                        final_tx = curr_tx if keep_self else peer_rx

            next_state[if_id] = {'rx': final_rx, 'tx': final_tx}

        # Update State
        for if_id, vals in next_state.items():
            state[if_id]['rx'] = vals['rx']
            state[if_id]['tx'] = vals['tx']

    # --- Step 3: Confidence & Result ---
    result = {}
    for if_id, orig_data in telemetry.items():
        s = state[if_id]
        orig_rx = float(orig_data.get('rx_rate', 0.0))
        orig_tx = float(orig_data.get('tx_rate', 0.0))

        peer_id = s['peer']
        has_peer = peer_id and peer_id in state
        r_id = s['router']

        # Check Flow Balance
        flow_balanced = False
        if r_id in topology:
             final_rx_sum = sum(state[i]['rx'] for i in topology[r_id] if i in state)
             final_tx_sum = sum(state[i]['tx'] for i in topology[r_id] if i in state)
             if get_diff(final_rx_sum, final_tx_sum) < 0.05:
                 flow_balanced = True

        def calculate_confidence(orig, final, peer_val, is_tx):
            # Error distances
            err_orig = get_diff(final, orig)
            err_peer = get_diff(final, peer_val) if peer_val is not None else 0.0

            # 1. We trusted Original (approx)
            if err_orig < HARDENING_THRESHOLD:
                if peer_val is not None and err_peer > HARDENING_THRESHOLD:
                    # Disagreement with Peer
                    if is_tx and final > peer_val:
                         # TX > Peer RX (Loss). Valid if Flow supports.
                         return 0.95 if flow_balanced else 0.8
                    elif not is_tx and final < peer_val:
                         # RX < Peer TX (Loss). Valid if Flow supports.
                         return 0.95 if flow_balanced else 0.8
                    else:
                         # "Impossible" state kept (e.g. RX > Peer TX).
                         # Only likely if Peer sensor is broken.
                         return 0.85
                return 1.0

            # 2. We Repaired
            # If we matched Peer
            if peer_val is not None and err_peer < HARDENING_THRESHOLD:
                return 0.95

            # If we didn't match Peer (e.g. flow arbitration selected something else or averaged)
            return 0.8

        rx_conf = calculate_confidence(orig_rx, s['rx'], state[peer_id]['tx'] if has_peer else None, False)
        tx_conf = calculate_confidence(orig_tx, s['tx'], state[peer_id]['rx'] if has_peer else None, True)

        st_conf = 1.0
        if s['status'] != orig_data.get('interface_status'):
            st_conf = 0.95

        result[if_id] = {
            'rx_rate': (orig_rx, s['rx'], rx_conf),
            'tx_rate': (orig_tx, s['tx'], tx_conf),
            'interface_status': (orig_data.get('interface_status'), s['status'], st_conf),
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