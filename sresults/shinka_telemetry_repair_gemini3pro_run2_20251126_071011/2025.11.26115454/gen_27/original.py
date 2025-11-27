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

    Core principle: Use network invariants to validate and repair telemetry:
    1. Link Symmetry (R3): my_tx_rate ≈ their_rx_rate for connected interfaces
    2. Flow Conservation (R1): Sum(incoming traffic) = Sum(outgoing traffic) at each router
    3. Interface Consistency: Status should be consistent across connected pairs

    Args:
        telemetry: Dictionary where key is interface_id and value contains:
            - interface_status: "up" or "down"
            - rx_rate: receive rate in Mbps
            - tx_rate: transmit rate in Mbps
            - connected_to: interface_id this interface connects to
            - local_router: router_id this interface belongs to
            - remote_router: router_id on the other side
        topology: Dictionary where key is router_id and value contains a list of interface_ids

    Returns:
        Dictionary with same structure but telemetry values become tuples of:
        (original_value, repaired_value, confidence_score)
        where confidence ranges from 0.0 (very uncertain) to 1.0 (very confident)
    """

    HARDENING_THRESHOLD = 0.02

    # Initialize working data structure
    working_state = {}

    # Pre-process: Copy initial data
    for if_id, data in telemetry.items():
        working_state[if_id] = {
            'rx': float(data.get('rx_rate', 0.0)),
            'tx': float(data.get('tx_rate', 0.0)),
            'status': data.get('interface_status', 'unknown'),
            'rx_conf': 0.5, # Default to moderate/uncertain until validated
            'tx_conf': 0.5,
            'status_conf': 1.0,
            'original': data
        }

    # Step 1: Link Symmetry & Status Consistency
    for if_id, state in working_state.items():
        orig_data = state['original']
        peer_id = orig_data.get('connected_to')

        if peer_id and peer_id in working_state:
            peer_state = working_state[peer_id]

            # 1a. Status Consistency
            s1 = state['status']
            s2 = peer_state['status']

            # Check for active traffic to resolve status conflicts
            traffic_active = (state['rx'] > 1.0 or state['tx'] > 1.0 or
                              peer_state['rx'] > 1.0 or peer_state['tx'] > 1.0)

            if s1 != s2:
                # Inconsistency found
                if traffic_active:
                    # If traffic flows, link is likely UP
                    state['status'] = 'up'
                    state['status_conf'] = 0.8
                else:
                    # No traffic, likely DOWN
                    state['status'] = 'down'
                    state['status_conf'] = 0.8

            # If status is DOWN, force rates to 0
            if state['status'] == 'down':
                state['rx'] = 0.0
                state['tx'] = 0.0
                state['rx_conf'] = 1.0
                state['tx_conf'] = 1.0
                continue

            # 1b. Rate Symmetry Candidates
            # Use ORIGINAL values for symmetry checks to avoid order-dependence bias
            my_rx = float(orig_data.get('rx_rate', 0.0))
            peer_tx = float(peer_state['original'].get('tx_rate', 0.0))
            denom = max(my_rx, peer_tx, 1.0)

            if abs(my_rx - peer_tx) / denom <= HARDENING_THRESHOLD:
                # Signals agree
                avg_val = (my_rx + peer_tx) / 2.0
                state['rx'] = avg_val
                state['rx_conf'] = 1.0
            else:
                # Signals disagree, keep original but mark uncertain
                state['rx_conf'] = 0.5

            # Compare My TX vs Peer RX
            my_tx = float(orig_data.get('tx_rate', 0.0))
            peer_rx = float(peer_state['original'].get('rx_rate', 0.0))
            denom_tx = max(my_tx, peer_rx, 1.0)

            if abs(my_tx - peer_rx) / denom_tx <= HARDENING_THRESHOLD:
                state['tx'] = (my_tx + peer_rx) / 2.0
                state['tx_conf'] = 1.0
            else:
                state['tx_conf'] = 0.5

    # Step 2: Flow Conservation (Router Level)
    # Use flow conservation to resolve uncertainties (conf=0.5)
    for router_id, if_list in topology.items():
        router_ifs = [i for i in if_list if i in working_state]
        if not router_ifs:
            continue

        total_rx = sum(working_state[i]['rx'] for i in router_ifs)
        total_tx = sum(working_state[i]['tx'] for i in router_ifs)
        imbalance = total_rx - total_tx

        # Only attempt repair if imbalance is significant
        if abs(imbalance) > max(total_rx, total_tx, 1.0) * HARDENING_THRESHOLD:
            best_fix = None
            min_residual = abs(imbalance)

            # Find the best single interface repair that fixes flow conservation
            for i in router_ifs:
                st = working_state[i]
                peer_id = st['original'].get('connected_to')
                if not peer_id or peer_id not in working_state:
                    continue
                peer_st = working_state[peer_id]

                # Check if replacing RX with peer's TX helps (if RX is uncertain)
                if st['rx_conf'] <= 0.5:
                    candidate_rx = peer_st['tx']
                    # New imbalance = (total_rx - old_rx + new_rx) - total_tx
                    #               = imbalance - old_rx + new_rx
                    new_imbalance = imbalance - st['rx'] + candidate_rx
                    if abs(new_imbalance) < min_residual:
                        min_residual = abs(new_imbalance)
                        best_fix = (i, 'rx', candidate_rx)

                # Check if replacing TX with peer's RX helps (if TX is uncertain)
                if st['tx_conf'] <= 0.5:
                    candidate_tx = peer_st['rx']
                    # New imbalance = total_rx - (total_tx - old_tx + new_tx)
                    #               = imbalance + old_tx - new_tx
                    new_imbalance = imbalance + st['tx'] - candidate_tx
                    if abs(new_imbalance) < min_residual:
                        min_residual = abs(new_imbalance)
                        best_fix = (i, 'tx', candidate_tx)

            # Apply the best fix
            if best_fix:
                fid, ftype, fval = best_fix
                working_state[fid][ftype] = fval
                working_state[fid][f'{ftype}_conf'] = 0.9 # High confidence: confirmed by flow conservation

                # Update peer to match for consistency
                peer_id = working_state[fid]['original'].get('connected_to')
                if peer_id:
                    peer_ftype = 'tx' if ftype == 'rx' else 'rx'
                    working_state[peer_id][peer_ftype] = fval
                    working_state[peer_id][f'{peer_ftype}_conf'] = 0.9

    # Final Output Construction
    result = {}
    for if_id, st in working_state.items():
        orig = st['original']

        # Ensure consistency for DOWN interfaces
        if st['status'] == 'down':
             st['rx_conf'] = 1.0
             st['tx_conf'] = 1.0

        result[if_id] = {
            'rx_rate': (orig.get('rx_rate', 0.0), st['rx'], st['rx_conf']),
            'tx_rate': (orig.get('tx_rate', 0.0), st['tx'], st['tx_conf']),
            'interface_status': (orig.get('interface_status', 'unknown'), st['status'], st['status_conf']),
            'connected_to': orig.get('connected_to'),
            'local_router': orig.get('local_router'),
            'remote_router': orig.get('remote_router')
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