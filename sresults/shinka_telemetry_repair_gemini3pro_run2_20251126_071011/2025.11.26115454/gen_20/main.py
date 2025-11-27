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

    Uses an iterative consensus algorithm with physical asymmetry logic:
    1. RX cannot exceed Peer TX (physically impossible).
    2. TX can exceed Peer RX (packet loss).
    3. Flow Conservation acts as arbiter.
    """

    HARDENING_THRESHOLD = 0.02
    NOISE_FLOOR = 0.1
    MIN_RATE_THRESHOLD = 10.0

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
        # Check Peer Status
        peer_id = s['peer']
        peer_down = False
        if peer_id and peer_id in state:
            if state[peer_id]['status'] == 'down':
                peer_down = True

        # Evidence for UP
        has_local_traffic = s['rx'] > NOISE_FLOOR or s['tx'] > NOISE_FLOOR

        peer_says_active = False
        if peer_id and peer_id in state:
            p_s = state[peer_id]
            if p_s['status'] == 'up' and (p_s['tx'] > NOISE_FLOOR or p_s['rx'] > NOISE_FLOOR):
                peer_says_active = True

        if s['status'] == 'down':
            if (has_local_traffic or peer_says_active) and not peer_down:
                s['status'] = 'up'
        elif s['status'] == 'up':
            pass

    # Enforce down = zero traffic
    for if_id, s in state.items():
        force_down = False
        if s['peer'] and s['peer'] in state:
            if state[s['peer']]['status'] == 'down':
                force_down = True

        if s['status'] != 'up' or force_down:
            s['rx'] = 0.0
            s['tx'] = 0.0
            if force_down:
                s['status'] = 'down'

    # --- Step 2: Iterative Rate Repair ---
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

            peer_id = s['peer']
            r_id = s['router']
            has_peer = peer_id and peer_id in state

            def agrees(a, b):
                return abs(a - b) <= max(a, b, MIN_RATE_THRESHOLD) * HARDENING_THRESHOLD

            # --- RX Repair ---
            val_self = s['rx']
            val_peer = state[peer_id]['tx'] if has_peer else None

            val_flow = None
            if r_id and r_id in router_totals:
                totals = router_totals[r_id]
                val_flow = max(0.0, totals['tx'] - (totals['rx'] - val_self))

            new_rx = val_self
            if val_peer is not None:
                if agrees(val_self, val_peer):
                    new_rx = (val_self + val_peer) / 2.0
                else:
                    supports_self = val_flow is not None and agrees(val_self, val_flow)
                    supports_peer = val_flow is not None and agrees(val_peer, val_flow)

                    if supports_peer and not supports_self:
                        new_rx = val_peer
                    elif supports_self and not supports_peer:
                        # Flow matches self.
                        # Physical Constraint: I cannot be > Peer TX
                        if val_self > val_peer * (1.0 + HARDENING_THRESHOLD):
                            new_rx = val_peer
                        else:
                            new_rx = val_self
                    else:
                        new_rx = val_peer

            # --- TX Repair ---
            val_self = s['tx']
            val_peer = state[peer_id]['rx'] if has_peer else None

            val_flow = None
            if r_id and r_id in router_totals:
                totals = router_totals[r_id]
                val_flow = max(0.0, totals['rx'] - (totals['tx'] - val_self))

            new_tx = val_self
            if val_peer is not None:
                if agrees(val_self, val_peer):
                    new_tx = (val_self + val_peer) / 2.0
                else:
                    supports_self = val_flow is not None and agrees(val_self, val_flow)
                    supports_peer = val_flow is not None and agrees(val_peer, val_flow)

                    if supports_self and not supports_peer:
                        new_tx = val_self
                    elif supports_peer and not supports_self:
                        new_tx = val_peer
                    else:
                        # Asymmetry: Trust Self if Loss Scenario (Self > Peer)
                        if val_self > val_peer:
                            new_tx = val_self
                        else:
                            new_tx = val_peer

            next_rates[if_id] = {'rx': new_rx, 'tx': new_tx}

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

        peer_id = final['peer']
        peer_tx = state[peer_id]['tx'] if (peer_id and peer_id in state) else None
        peer_rx = state[peer_id]['rx'] if (peer_id and peer_id in state) else None

        r_id = final['router']
        flow_ok_rx = False
        flow_ok_tx = False
        if r_id and r_id in topology:
             s_rx = sum(state[i]['rx'] for i in topology[r_id] if i in state)
             s_tx = sum(state[i]['tx'] for i in topology[r_id] if i in state)
             if abs(s_rx - s_tx) < max(s_rx, s_tx, MIN_RATE_THRESHOLD) * 0.05:
                 flow_ok_rx = True
                 flow_ok_tx = True

        def get_conf(orig, repaired, peer_val, flow_ok):
            changed = abs(orig - repaired) > max(orig, MIN_RATE_THRESHOLD) * HARDENING_THRESHOLD
            agrees_peer = False
            if peer_val is not None:
                agrees_peer = abs(repaired - peer_val) <= max(repaired, MIN_RATE_THRESHOLD) * HARDENING_THRESHOLD

            if not changed:
                if peer_val is not None and not agrees_peer and not flow_ok:
                    return 0.8
                return 1.0

            if agrees_peer and flow_ok: return 0.95
            if agrees_peer: return 0.9
            if flow_ok: return 0.8
            return 0.7

        rx_conf = get_conf(orig_rx, final['rx'], peer_tx, flow_ok_rx)
        tx_conf = get_conf(orig_tx, final['tx'], peer_rx, flow_ok_tx)

        st_conf = 1.0
        if final['status'] != orig_status:
            if final['status'] == 'down':
                 st_conf = 0.95
            elif final['rx'] > 1.0 or final['tx'] > 1.0:
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