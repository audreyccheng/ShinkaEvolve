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
    NOISE_FLOOR = 10.0  # Mbps, treat differences below this as noise/unimportant

    # Initialize working state
    state = {}
    for if_id, data in telemetry.items():
        state[if_id] = {
            'rx': float(data.get('rx_rate', 0.0)),
            'tx': float(data.get('tx_rate', 0.0)),
            'status': data.get('interface_status', 'unknown'),
            'rx_conf': 0.5,  # Start uncertain
            'tx_conf': 0.5,
            'status_conf': 1.0,
            'orig': data,
            'fixed': False
        }

    # Pass 1: Status Consensus & Zeroing
    # Implements Peer-Status Dominance with traffic validation
    for if_id, s in state.items():
        peer_id = s['orig'].get('connected_to')
        if peer_id and peer_id in state:
            peer = state[peer_id]

            # If explicit mismatch
            if s['status'] != peer['status']:
                s_traffic = max(s['rx'], s['tx'])
                p_traffic = max(peer['rx'], peer['tx'])

                # If peer says DOWN, we should likely be DOWN, unless we have significant traffic
                if peer['status'] == 'down':
                    if s_traffic > NOISE_FLOOR:
                        # Conflict: Peer down but we have high traffic.
                        # Trust traffic (Network is UP, Peer Status is Stale)
                        s['status'] = 'up'
                        s['status_conf'] = 0.8
                    else:
                        # Consistent with peer down (Low traffic might be noise)
                        s['status'] = 'down'
                        s['status_conf'] = 0.9
                elif s['status'] == 'down':
                    # We say DOWN, Peer says UP
                    if p_traffic > NOISE_FLOOR:
                        # Peer has traffic, we are likely UP (Our status stale)
                        s['status'] = 'up'
                        s['status_conf'] = 0.8
                    else:
                        # No traffic on peer either, likely DOWN
                        s['status'] = 'down' # Stay down
                        s['status_conf'] = 0.9
            else:
                 # Statuses match.
                 pass

    # Enforce DOWN invariants (Highest Priority)
    for s in state.values():
        if s['status'] == 'down':
            s['rx'], s['tx'] = 0.0, 0.0
            s['rx_conf'], s['tx_conf'] = 1.0, 1.0
            s['fixed'] = True

    # Pass 2: Symmetry Check with Noise Dampening
    pairs_processed = set()
    for if_id, s in state.items():
        if s['fixed']: continue

        peer_id = s['orig'].get('connected_to')
        if not peer_id or peer_id not in state: continue

        pair = tuple(sorted([if_id, peer_id]))
        if pair in pairs_processed: continue
        pairs_processed.add(pair)

        peer = state[peer_id]
        if peer['fixed']: continue

        # Check TX -> Peer RX
        tx = s['tx']
        prx = peer['rx']
        # Use noise floor to avoid penalizing small absolute differences on low links
        denom_tx = max(tx, prx, NOISE_FLOOR)

        if abs(tx - prx) / denom_tx <= HARDENING_THRESHOLD:
            avg = (tx + prx) / 2.0
            s['tx'] = avg
            peer['rx'] = avg
            s['tx_conf'] = 1.0
            peer['rx_conf'] = 1.0

        # Check RX <- Peer TX
        rx = s['rx']
        ptx = peer['tx']
        denom_rx = max(rx, ptx, NOISE_FLOOR)

        if abs(rx - ptx) / denom_rx <= HARDENING_THRESHOLD:
            avg = (rx + ptx) / 2.0
            s['rx'] = avg
            peer['tx'] = avg
            s['rx_conf'] = 1.0
            peer['tx_conf'] = 1.0

    # Pass 3: Flow Conservation Repair
    def get_imbalance(rid):
        if rid not in topology: return 0.0
        in_flow = sum(state[i]['rx'] for i in topology[rid] if i in state)
        out_flow = sum(state[i]['tx'] for i in topology[rid] if i in state)
        return in_flow - out_flow

    ITERATIONS = 2
    for _ in range(ITERATIONS):
        # Identify imbalanced routers
        imbalanced_routers = []
        for rid in topology:
            imb = get_imbalance(rid)
            flow = sum(state[i]['rx'] + state[i]['tx'] for i in topology[rid] if i in state)
            # Use noise floor here too
            if abs(imb) > max(flow, NOISE_FLOOR) * HARDENING_THRESHOLD:
                imbalanced_routers.append(rid)

        imbalanced_routers.sort()

        for rid in imbalanced_routers:
            current_imb = get_imbalance(rid)
            flow = sum(state[i]['rx'] + state[i]['tx'] for i in topology[rid] if i in state)
            if abs(current_imb) <= max(flow, NOISE_FLOOR) * HARDENING_THRESHOLD:
                continue

            # Candidates: Interfaces with uncertain values
            candidates = []
            for if_id in topology[rid]:
                if if_id not in state: continue
                # We can try to repair any interface, but prioritize those not fixed by symmetry
                if state[if_id]['rx_conf'] < 1.0:
                    candidates.append((if_id, 'rx'))
                if state[if_id]['tx_conf'] < 1.0:
                    candidates.append((if_id, 'tx'))

            if not candidates: continue

            best_fix = None
            min_global_error = abs(current_imb) # Baseline to beat

            for if_id, metric in candidates:
                s = state[if_id]
                peer_id = s['orig'].get('connected_to')
                if not peer_id or peer_id not in state: continue

                peer = state[peer_id]

                curr_val = s[metric]
                peer_metric = 'tx' if metric == 'rx' else 'rx'
                peer_val = peer[peer_metric]

                # Try adopting Peer's Value (Trust Peer)
                # Update Local Imbalance: Imb = In - Out
                # RX (In): New = Old - curr + peer
                # TX (Out): New = Old - (-curr) + (-peer) = Old + curr - peer

                delta_local = (-curr_val + peer_val) if metric == 'rx' else (curr_val - peer_val)
                new_local_imb = current_imb + delta_local

                cost = abs(new_local_imb)

                if cost < min_global_error:
                    min_global_error = cost
                    best_fix = (if_id, metric, peer_val, cost)

            if best_fix:
                fid, ftype, fval, fcost = best_fix

                # Apply
                state[fid][ftype] = fval

                # Update Peer to maintain symmetry
                pid = state[fid]['orig'].get('connected_to')
                if pid in state:
                    ptype = 'tx' if ftype == 'rx' else 'rx'
                    state[pid][ptype] = fval

                    # Sigmoid-like Confidence Decay
                    rate_mag = max(fval, NOISE_FLOOR)
                    rel_err = fcost / rate_mag
                    # Sigmoid shape approximation: 1 / (1 + k*x)
                    conf = 1.0 / (1.0 + 10.0 * rel_err)

                    state[fid][f'{ftype}_conf'] = conf
                    state[pid][f'{ptype}_conf'] = conf

    # Final Result Construction
    result = {}
    for if_id, s in state.items():
        result[if_id] = {
            'rx_rate': (s['orig'].get('rx_rate', 0.0), s['rx'], s['rx_conf']),
            'tx_rate': (s['orig'].get('tx_rate', 0.0), s['tx'], s['tx_conf']),
            'interface_status': (s['orig'].get('interface_status', 'unknown'), s['status'], s['status_conf']),
            'connected_to': s['orig'].get('connected_to'),
            'local_router': s['orig'].get('local_router'),
            'remote_router': s['orig'].get('remote_router')
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