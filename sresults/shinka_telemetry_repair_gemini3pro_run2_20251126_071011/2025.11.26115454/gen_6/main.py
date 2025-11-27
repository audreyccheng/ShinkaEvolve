# EVOLVE-BLOCK-START
"""
Network telemetry repair algorithm that detects and corrects inconsistencies
in network interface telemetry data using topology relationships.

Takes interface telemetry data and detects/repairs inconsistencies based on
network invariants like link symmetry and flow conservation.
"""
from typing import Dict, Any, Tuple, List
import math

def repair_network_telemetry(telemetry: Dict[str, Dict[str, Any]],
                             topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:
    """
    Repairs network telemetry using an Iterative Consensus algorithm.
    Resolves conflicts between Self, Peer, and Flow Conservation signals
    by selecting the most consistent candidate rather than averaging.
    """

    TOLERANCE = 0.05  # 5% tolerance
    ITERATIONS = 5    # Number of repair passes

    # --- Step 1: Initialize State ---
    state = {}
    for if_id, data in telemetry.items():
        state[if_id] = {
            'rx': float(data.get('rx_rate', 0.0)),
            'tx': float(data.get('tx_rate', 0.0)),
            'status': data.get('interface_status', 'unknown'),
            'orig_rx': float(data.get('rx_rate', 0.0)),
            'orig_tx': float(data.get('tx_rate', 0.0)),
            'orig_status': data.get('interface_status', 'unknown'),
            'peer': data.get('connected_to'),
            'router': data.get('local_router')
        }

    # --- Step 2: Status Inference ---
    for if_id, s in state.items():
        # Heuristic: Link is UP if any valid traffic is seen locally or remotely
        local_active = s['rx'] > 1.0 or s['tx'] > 1.0

        peer_active = False
        if s['peer'] and s['peer'] in state:
            ps = state[s['peer']]
            if ps['orig_status'] == 'up' and (ps['rx'] > 1.0 or ps['tx'] > 1.0):
                peer_active = True

        if local_active or peer_active:
            s['status'] = 'up'
        else:
            # If originally down, stay down. If originally up but idle, stay up?
            # Trust explicit 'down' if no traffic.
            if s['orig_status'] == 'down' and not (local_active or peer_active):
                 s['status'] = 'down'
            else:
                 s['status'] = 'up'

        # Enforce Down = 0 rates
        if s['status'] == 'down':
            s['rx'] = 0.0
            s['tx'] = 0.0

    # --- Step 3: Iterative Repair of Rates ---
    for _ in range(ITERATIONS):
        # Pre-calculate Router Aggregates (Sum RX/TX) for Flow Conservation
        router_aggregates = {}
        for r_id, if_ids in topology.items():
            sum_rx = sum(state[i]['rx'] for i in if_ids if i in state)
            sum_tx = sum(state[i]['tx'] for i in if_ids if i in state)
            router_aggregates[r_id] = {'sum_rx': sum_rx, 'sum_tx': sum_tx}

        for if_id, s in state.items():
            if s['status'] == 'down': continue

            peer_id = s['peer']
            has_peer = peer_id and peer_id in state

            # Helper to check if values agree within tolerance
            def is_close(a, b):
                if a is None or b is None: return False
                return abs(a - b) <= max(a, b, 1.0) * TOLERANCE

            # --- Repair RX ---
            # Candidates
            val_self = s['rx']
            val_peer = state[peer_id]['tx'] if has_peer else None

            # Flow Candidate: RX = Sum_TX_all - Sum_RX_others
            # Uses current loop start aggregates.
            old_rx = s['rx'] # Store for TX calculation later
            val_flow_rx = None
            if s['router'] in router_aggregates:
                aggs = router_aggregates[s['router']]
                rx_others = aggs['sum_rx'] - s['rx']
                tx_all = aggs['sum_tx']
                val_flow_rx = max(0.0, tx_all - rx_others)

            # Decision Logic for RX
            new_rx = val_self
            if val_peer is not None:
                if is_close(val_self, val_peer):
                    # Agreement -> High confidence average
                    new_rx = (val_self + val_peer) / 2.0
                else:
                    # Disagreement -> Use Flow to arbitrate
                    peer_matches_flow = is_close(val_peer, val_flow_rx)
                    self_matches_flow = is_close(val_self, val_flow_rx)

                    if peer_matches_flow and not self_matches_flow:
                        new_rx = val_peer
                    elif self_matches_flow and not peer_matches_flow:
                        new_rx = val_self
                    else:
                        # Fallback: Trust Peer (Redundancy principle)
                        new_rx = val_peer

            state[if_id]['rx'] = new_rx

            # --- Repair TX ---
            val_self = s['tx']
            val_peer = state[peer_id]['rx'] if has_peer else None

            # Flow Candidate: TX = RX_new + RX_others - TX_others
            val_flow_tx = None
            if s['router'] in router_aggregates:
                aggs = router_aggregates[s['router']]
                # Reconstruct others using start-of-loop values (approximate)
                rx_others = aggs['sum_rx'] - old_rx
                tx_others = aggs['sum_tx'] - s['tx']
                val_flow_tx = max(0.0, new_rx + rx_others - tx_others)

            # Decision Logic for TX
            new_tx = val_self
            if val_peer is not None:
                if is_close(val_self, val_peer):
                    new_tx = (val_self + val_peer) / 2.0
                else:
                    peer_matches_flow = is_close(val_peer, val_flow_tx)
                    self_matches_flow = is_close(val_self, val_flow_tx)

                    if peer_matches_flow and not self_matches_flow:
                        new_tx = val_peer
                    elif self_matches_flow and not peer_matches_flow:
                        new_tx = val_self
                    else:
                        new_tx = val_peer

            state[if_id]['tx'] = new_tx

    # --- Step 4: Confidence & Output ---
    result = {}
    for if_id, s in state.items():
        peer_id = s['peer']
        p_vals = state[peer_id] if (peer_id and peer_id in state) else None

        def get_conf(orig, final, peer_val):
            # Did we change the value?
            changed = abs(final - orig) > max(orig, 1.0) * TOLERANCE

            # Does it match peer?
            matches_peer = False
            if peer_val is not None:
                matches_peer = abs(final - peer_val) <= max(final, 1.0) * TOLERANCE

            # Does it match local flow?
            matches_flow = False
            if s['router'] in topology:
                r_ifs = topology[s['router']]
                sum_rx = sum(state[k]['rx'] for k in r_ifs if k in state)
                sum_tx = sum(state[k]['tx'] for k in r_ifs if k in state)
                if abs(sum_rx - sum_tx) < max(sum_rx, sum_tx, 1.0) * TOLERANCE:
                    matches_flow = True
            else:
                matches_flow = True # Assume OK if no router info

            if not changed:
                # Kept original
                if not matches_peer and peer_val is not None:
                    return 0.8 # Peer disagrees but we stuck to guns
                return 1.0
            else:
                # Changed
                if matches_peer and matches_flow: return 0.95
                if matches_peer: return 0.8
                if matches_flow: return 0.7
                return 0.5 # Low confidence guess

        rx_conf = get_conf(s['orig_rx'], s['rx'], p_vals['tx'] if p_vals else None)
        tx_conf = get_conf(s['orig_tx'], s['tx'], p_vals['rx'] if p_vals else None)

        st_conf = 1.0
        if s['status'] != s['orig_status']:
            st_conf = 0.9

        result[if_id] = {
            'rx_rate': (s['orig_rx'], s['rx'], rx_conf),
            'tx_rate': (s['orig_tx'], s['tx'], tx_conf),
            'interface_status': (s['orig_status'], s['status'], st_conf),
            'connected_to': s['peer'],
            'local_router': s['router'],
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
