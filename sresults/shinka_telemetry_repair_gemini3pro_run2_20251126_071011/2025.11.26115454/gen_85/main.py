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

    # Measurement timing tolerance (from Hodor research: ~2%)
    HARDENING_THRESHOLD = 0.02

    # Initialize working state
    state = {}
    for if_id, data in telemetry.items():
        # Initialize confidence based on link symmetry
        init_conf = 0.5
        peer_id = data.get('connected_to')
        if peer_id and peer_id in telemetry:
            tx = float(data.get('tx_rate', 0.0))
            prx = float(telemetry[peer_id].get('rx_rate', 0.0))
            if abs(tx - prx) / max(tx, prx, 1.0) < HARDENING_THRESHOLD:
                init_conf = 1.0

        state[if_id] = {
            'rx': float(data.get('rx_rate', 0.0)),
            'tx': float(data.get('tx_rate', 0.0)),
            'status': data.get('interface_status', 'unknown'),
            'rx_conf': init_conf,
            'tx_conf': init_conf,
            'status_conf': 1.0,
            'orig': data,
            'locked_rx': False,
            'locked_tx': False
        }

    # Pass 1: Status Consensus
    for if_id, s in state.items():
        connected_to = s['orig'].get('connected_to')
        if connected_to and connected_to in state:
            peer = state[connected_to]
            # If mismatch, prefer UP if traffic exists
            if s['status'] != peer['status']:
                has_traffic = (s['rx'] > 1.0 or s['tx'] > 1.0 or
                             peer['rx'] > 1.0 or peer['tx'] > 1.0)
                if has_traffic:
                    s['status'] = 'up'
                    s['status_conf'] = 0.8
                else:
                    s['status'] = 'down'
                    s['status_conf'] = 0.8

        # Enforce DOWN means zero rates
        if s['status'] == 'down':
            s['rx'] = 0.0
            s['tx'] = 0.0
            s['rx_conf'] = 1.0
            s['tx_conf'] = 1.0
            s['locked_rx'] = True
            s['locked_tx'] = True

    # Pass 2: Rate Repair with Symmetry & Flow Conservation

    def get_router_imbalance(router_id):
        if not router_id or router_id not in topology:
            return 0.0
        in_sum = 0.0
        out_sum = 0.0
        for if_id in topology[router_id]:
            if if_id in state:
                in_sum += state[if_id]['rx']
                out_sum += state[if_id]['tx']
        return in_sum - out_sum

    def get_router_total_flow(router_id):
        if not router_id or router_id not in topology:
            return 1.0
        total = 0.0
        for if_id in topology[router_id]:
            if if_id in state:
                total += state[if_id]['rx'] + state[if_id]['tx']
        return max(total, 1.0)

    # Reliability: how much we trust neighbors
    def get_router_reliability(router_id, exclude_if_id):
        if not router_id or router_id not in topology:
            return 0.0
        total_conf = 0.0
        count = 0
        for if_id in topology[router_id]:
            if if_id == exclude_if_id: continue
            if if_id in state:
                c = (state[if_id]['rx_conf'] + state[if_id]['tx_conf']) / 2.0
                total_conf += c
                count += 1
        return total_conf / count if count > 0 else 0.0

    # Iterative refinement
    ITERATIONS = 5
    sorted_interfaces = sorted(state.keys())

    for iteration in range(ITERATIONS):
        processed_pairs = set()

        for if_id in sorted_interfaces:
            s = state[if_id]
            connected_to = s['orig'].get('connected_to')
            if not connected_to or connected_to not in state:
                continue

            pair_id = tuple(sorted([if_id, connected_to]))
            if pair_id in processed_pairs:
                continue
            processed_pairs.add(pair_id)

            peer = state[connected_to]
            if s['status'] == 'down' and peer['status'] == 'down':
                continue

            # --- 1. Fix Direction A: Local TX -> Peer RX ---
            if not s['locked_tx']:
                c_tx = float(s['orig'].get('tx_rate', 0.0))
                c_prx = float(peer['orig'].get('rx_rate', 0.0))

                r_local = s['orig'].get('local_router')
                r_remote = peer['orig'].get('local_router')

                w_local = get_router_reliability(r_local, if_id)
                w_remote = get_router_reliability(r_remote, connected_to)

                # Flow Targets
                imb_local = get_router_imbalance(r_local)
                imb_remote = get_router_imbalance(r_remote)

                target_local = s['tx'] + imb_local
                target_remote = peer['rx'] - imb_remote

                # Golden Truth Check
                golden_avg = (target_local + target_remote) / 2.0
                golden_diff = abs(target_local - target_remote)
                is_golden = False

                if (w_local > 0.0 or w_remote > 0.0):
                    denom = max(golden_avg, 1.0)
                    if golden_diff / denom < HARDENING_THRESHOLD:
                         is_golden = True

                def eval_cost(v):
                    cost = 0.0
                    cost += w_local * abs(v - target_local)
                    cost += w_remote * abs(v - target_remote)
                    cost += 0.01 * abs(v - c_tx)
                    cost += 0.01 * abs(v - c_prx)
                    return cost

                if is_golden and w_local > 0.4 and w_remote > 0.4:
                    best_val = golden_avg
                    s['locked_tx'] = True
                    peer['locked_rx'] = True
                    s['tx_conf'] = 0.95
                    peer['rx_conf'] = 0.95
                else:
                    candidates = [c_tx, c_prx, target_local, target_remote]
                    if abs(c_tx - c_prx) / max(c_tx, c_prx, 1.0) < HARDENING_THRESHOLD:
                        candidates.append((c_tx + c_prx)/2.0)

                    best_val = s['tx']
                    min_cost = float('inf')

                    for v in candidates:
                        if v < 0: continue
                        c = eval_cost(v)
                        if c < min_cost:
                            min_cost = c
                            best_val = v

                    # Update Confidence if we have decent solidity
                    if abs(best_val - c_tx) / max(best_val, 1.0) < HARDENING_THRESHOLD and \
                       abs(best_val - target_local) / max(best_val, 1.0) < HARDENING_THRESHOLD and \
                       w_local > 0.6:
                           s['locked_tx'] = True
                           s['tx_conf'] = 0.95
                           peer['rx_conf'] = 0.95
                    else:
                        # Dynamic confidence based on fit
                        s['tx_conf'] = 0.5
                        if abs(best_val - target_local) / max(best_val, 1.0) < HARDENING_THRESHOLD: s['tx_conf'] += 0.2
                        if abs(best_val - target_remote) / max(best_val, 1.0) < HARDENING_THRESHOLD: s['tx_conf'] += 0.2
                        s['tx_conf'] = min(0.9, s['tx_conf'])
                        peer['rx_conf'] = s['tx_conf']

                s['tx'] = best_val
                peer['rx'] = best_val


            # --- 2. Fix Direction B: Local RX <- Peer TX ---
            if not s['locked_rx']:
                c_rx = float(s['orig'].get('rx_rate', 0.0))
                c_ptx = float(peer['orig'].get('tx_rate', 0.0))

                r_local = s['orig'].get('local_router')
                r_remote = peer['orig'].get('local_router')

                w_local = get_router_reliability(r_local, if_id)
                w_remote = get_router_reliability(r_remote, connected_to)

                imb_local = get_router_imbalance(r_local)
                imb_remote = get_router_imbalance(r_remote)

                target_local = s['rx'] - imb_local
                target_remote = peer['tx'] + imb_remote

                golden_avg = (target_local + target_remote) / 2.0
                golden_diff = abs(target_local - target_remote)
                is_golden = False

                if (w_local > 0.0 or w_remote > 0.0):
                    denom = max(golden_avg, 1.0)
                    if golden_diff / denom < HARDENING_THRESHOLD:
                         is_golden = True

                def eval_cost_b(v):
                    cost = 0.0
                    cost += w_local * abs(v - target_local)
                    cost += w_remote * abs(v - target_remote)
                    cost += 0.01 * abs(v - c_rx)
                    cost += 0.01 * abs(v - c_ptx)
                    return cost

                if is_golden and w_local > 0.4 and w_remote > 0.4:
                    best_val_b = golden_avg
                    s['locked_rx'] = True
                    peer['locked_tx'] = True
                    s['rx_conf'] = 0.95
                    peer['tx_conf'] = 0.95
                else:
                    candidates = [c_rx, c_ptx, target_local, target_remote]
                    if abs(c_rx - c_ptx) / max(c_rx, c_ptx, 1.0) < HARDENING_THRESHOLD:
                        candidates.append((c_rx + c_ptx)/2.0)

                    best_val_b = s['rx']
                    min_cost = float('inf')

                    for v in candidates:
                        if v < 0: continue
                        c = eval_cost_b(v)
                        if c < min_cost:
                            min_cost = c
                            best_val_b = v

                    if abs(best_val_b - c_rx) / max(best_val_b, 1.0) < HARDENING_THRESHOLD and \
                       abs(best_val_b - target_local) / max(best_val_b, 1.0) < HARDENING_THRESHOLD and \
                       w_local > 0.6:
                           s['locked_rx'] = True
                           s['rx_conf'] = 0.95
                           peer['tx_conf'] = 0.95
                    else:
                        s['rx_conf'] = 0.5
                        if abs(best_val_b - target_local) / max(best_val_b, 1.0) < HARDENING_THRESHOLD: s['rx_conf'] += 0.2
                        if abs(best_val_b - target_remote) / max(best_val_b, 1.0) < HARDENING_THRESHOLD: s['rx_conf'] += 0.2
                        s['rx_conf'] = min(0.9, s['rx_conf'])
                        peer['tx_conf'] = s['rx_conf']

                s['rx'] = best_val_b
                peer['tx'] = best_val_b

    # Final Confidence Calibration (Residual Penalty)
    for if_id, s in state.items():
        if s['status'] == 'down': continue

        # Penalize confidence based on residual imbalance at the router
        r_local = s['orig'].get('local_router')
        flow_local = get_router_total_flow(r_local)
        imb_local = abs(get_router_imbalance(r_local))

        penalty = 0.0
        if flow_local > 1.0:
            penalty = (imb_local / flow_local) * 2.0

        if not s['locked_rx']:
            s['rx_conf'] = max(0.0, min(1.0, s.get('rx_conf', 0.5) - penalty))
        if not s['locked_tx']:
            s['tx_conf'] = max(0.0, min(1.0, s.get('tx_conf', 0.5) - penalty))

        peer_id = s['orig'].get('connected_to')
        if peer_id in state:
            peer = state[peer_id]
            peer['tx_conf'] = s['rx_conf']
            peer['rx_conf'] = s['tx_conf']

    # Assemble result
    result = {}
    for if_id, s in state.items():
        orig = s['orig']

        if s['rx_conf'] > 0.8 and s['tx_conf'] > 0.8:
            s['status_conf'] = max(s['status_conf'], 0.95)

        result[if_id] = {
            'rx_rate': (orig.get('rx_rate', 0.0), s['rx'], s['rx_conf']),
            'tx_rate': (orig.get('tx_rate', 0.0), s['tx'], s['tx_conf']),
            'interface_status': (orig.get('interface_status', 'unknown'), s['status'], s['status_conf']),
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