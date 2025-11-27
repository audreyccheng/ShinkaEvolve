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
    Repair network interface telemetry using Iterative Physics-Flow Consensus.

    Strategies:
    1. Status: Traffic > Noise implies UP. Peer DOWN implies Local DOWN (unless Local Traffic exists).
    2. Rate Repair (Asymmetric Physics + Flow Arbitration):
       - Distinguish between "Impossible" (RX > PeerTX) and "Plausible" (RX < PeerTX) errors.
       - Use Router Flow State (Surplus vs Deficit) to arbitrate disagreements.
       - If Local Router has Deficit (TX > RX), it effectively "votes" for higher RX input or lower TX output.
       - If Local Router has Surplus (RX > TX), it votes for lower RX or higher TX.
    3. Confidence:
       - Continuous score based on Peer Agreement and Final Router Flow Quality.
    """

    # Configuration
    HARDENING_THRESHOLD = 0.02
    MIN_RATE_THRESHOLD = 10.0
    ITERATIONS = 3

    # --- Initialize State ---
    state = {}
    for if_id, data in telemetry.items():
        state[if_id] = {
            'rx': float(data.get('rx_rate', 0.0)),
            'tx': float(data.get('tx_rate', 0.0)),
            'status': data.get('interface_status', 'unknown'),
            'peer': data.get('connected_to'),
            'router': data.get('local_router'),
            'orig_rx': float(data.get('rx_rate', 0.0)),
            'orig_tx': float(data.get('tx_rate', 0.0)),
            'orig_status': data.get('interface_status', 'unknown')
        }

    # --- Step 1: Status Repair ---
    for if_id, s in state.items():
        peer_id = s['peer']
        peer_is_down = False
        peer_has_traffic = False

        if peer_id and peer_id in telemetry:
            p_data = telemetry[peer_id]
            if p_data.get('interface_status') == 'down':
                peer_is_down = True
            if float(p_data.get('rx_rate', 0)) > MIN_RATE_THRESHOLD or \
               float(p_data.get('tx_rate', 0)) > MIN_RATE_THRESHOLD:
                peer_has_traffic = True

        local_traffic = s['rx'] > MIN_RATE_THRESHOLD or s['tx'] > MIN_RATE_THRESHOLD

        if peer_is_down and not local_traffic:
            s['status'] = 'down'
        elif local_traffic or peer_has_traffic:
            s['status'] = 'up'

        if s['status'] != 'up':
            s['rx'] = 0.0
            s['tx'] = 0.0

    # --- Step 2: Rate Repair ---
    def get_diff(v1, v2):
        return abs(v1 - v2) / max(v1, v2, MIN_RATE_THRESHOLD)

    for _ in range(ITERATIONS):
        # Calculate Flow Nets (RX - TX)
        router_net = {}
        for r_id, if_ids in topology.items():
            r_rx = sum(state[i]['rx'] for i in if_ids if i in state)
            r_tx = sum(state[i]['tx'] for i in if_ids if i in state)
            router_net[r_id] = r_rx - r_tx

        updates = {}

        for if_id, s in state.items():
            if s['status'] != 'up':
                updates[if_id] = {'rx': 0.0, 'tx': 0.0}
                continue

            peer_id = s['peer']
            r_id = s['router']
            has_peer = peer_id and peer_id in state

            curr_rx = s['rx']
            curr_tx = s['tx']

            # --- RX Repair ---
            final_rx = curr_rx
            if has_peer:
                peer_tx = state[peer_id]['tx']
                diff = get_diff(curr_rx, peer_tx)

                if diff < HARDENING_THRESHOLD:
                    final_rx = (curr_rx + peer_tx) / 2.0
                else:
                    # Disagreement
                    net_flow = router_net.get(r_id, 0.0) # RX - TX

                    if curr_rx > peer_tx:
                        # Impossible (RX > Peer TX).
                        # Enforce Physics (assuming Peer is not drastically under-reporting)
                        final_rx = peer_tx
                    else:
                        # curr_rx < peer_tx (Loss).
                        # If Deficit (net < 0), we need more RX to balance TX. Trust Peer TX (recover loss).
                        # If Surplus (net > 0), we have too much RX already. Keep Low RX (real loss).
                        if net_flow < -MIN_RATE_THRESHOLD:
                            final_rx = peer_tx
                        else:
                            final_rx = curr_rx

            # --- TX Repair ---
            final_tx = curr_tx
            if has_peer:
                peer_rx = state[peer_id]['rx']
                diff = get_diff(curr_tx, peer_rx)

                if diff < HARDENING_THRESHOLD:
                    final_tx = (curr_tx + peer_rx) / 2.0
                else:
                    net_flow = router_net.get(r_id, 0.0)

                    if curr_tx < peer_rx:
                        # Impossible (TX < Peer RX).
                        # Must repair up to Peer RX.
                        final_tx = peer_rx
                    else:
                        # curr_tx > peer_rx (Loss on wire).
                        # If Surplus (net > 0), we have RX to support this TX. Keep high TX.
                        # If Deficit (net < 0), we don't have inputs for this TX. Phantom? Repair to Peer RX.
                        if net_flow < -MIN_RATE_THRESHOLD:
                            final_tx = peer_rx
                        else:
                            final_tx = curr_tx

            updates[if_id] = {'rx': final_rx, 'tx': final_tx}

        for if_id, vals in updates.items():
            state[if_id]['rx'] = vals['rx']
            state[if_id]['tx'] = vals['tx']

    # --- Step 3: Confidence ---
    # Final Flow Check
    router_quality = {}
    for r_id, if_ids in topology.items():
        r_rx = sum(state[i]['rx'] for i in if_ids if i in state)
        r_tx = sum(state[i]['tx'] for i in if_ids if i in state)
        mx = max(r_rx, r_tx, MIN_RATE_THRESHOLD)
        imbalance = abs(r_rx - r_tx) / mx
        # Quality: 1.0 (perfect) to 0.0 (bad)
        router_quality[r_id] = max(0.0, 1.0 - (imbalance * 5.0))

    result = {}
    for if_id, s in state.items():
        orig_rx = s['orig_rx']
        orig_tx = s['orig_tx']
        peer_id = s['peer']
        has_peer = peer_id and peer_id in state
        r_id = s['router']
        flow_q = router_quality.get(r_id, 0.5)

        peer_tx = state[peer_id]['tx'] if has_peer else None
        peer_rx = state[peer_id]['rx'] if has_peer else None

        def calc_conf(final, orig, peer_val, flow_q):
            # Agreement with Peer
            peer_agree = False
            if peer_val is not None:
                if get_diff(final, peer_val) < HARDENING_THRESHOLD:
                    peer_agree = True

            # Change magnitude
            changed = get_diff(final, orig) > HARDENING_THRESHOLD

            if changed:
                if peer_agree:
                    # We repaired to match Peer.
                    # Confidence depends on Flow Quality (did it work?).
                    return 0.9 + (0.09 * flow_q)
                else:
                    # We repaired but not to Peer? (Unlikely in this logic, maybe average)
                    return 0.75
            else:
                # Kept Original
                if peer_val is not None and not peer_agree:
                    # Disagreement.
                    # If Flow is perfect (1.0), we are very confident in our Self value.
                    # If Flow is bad, we are uncertain.
                    return 0.7 + (0.25 * flow_q)

                return 1.0

        rx_conf = calc_conf(s['rx'], orig_rx, peer_tx, flow_q)
        tx_conf = calc_conf(s['tx'], orig_tx, peer_rx, flow_q)

        st_conf = 1.0
        if s['status'] != s['orig_status']:
            st_conf = 0.95

        result[if_id] = {
            'rx_rate': (orig_rx, s['rx'], rx_conf),
            'tx_rate': (orig_tx, s['tx'], tx_conf),
            'interface_status': (s['orig_status'], s['status'], st_conf),
            'connected_to': telemetry[if_id].get('connected_to'),
            'local_router': telemetry[if_id].get('local_router'),
            'remote_router': telemetry[if_id].get('remote_router')
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