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

    # Initialize working state with originals
    state = {}
    for if_id, data in telemetry.items():
        state[if_id] = {
            'rx': float(data.get('rx_rate', 0.0)),
            'tx': float(data.get('tx_rate', 0.0)),
            'status': data.get('interface_status', 'unknown'),
            'rx_conf': 1.0,
            'tx_conf': 1.0,
            'status_conf': 1.0,
            'orig': data
        }

    # Pass 1: Fix Status Consistency (Peer Dominance)
    sorted_ifs = sorted(state.keys())
    for if_id in sorted_ifs:
        s = state[if_id]
        peer_id = s['orig'].get('connected_to')

        if peer_id and peer_id in state:
            peer = state[peer_id]
            # If peer reports DOWN, trust it implies no traffic possible
            if peer['orig'].get('interface_status') == 'down':
                s['status'] = 'down'
                s['status_conf'] = 0.95
                s['rx'] = 0.0
                s['tx'] = 0.0
                s['rx_conf'] = 1.0
                s['tx_conf'] = 1.0
                continue

            # If mismatch but peer UP, trust traffic
            if s['status'] != peer['status']:
                traffic_active = max(s['rx'], s['tx'], peer['rx'], peer['tx']) > 1.0
                if traffic_active:
                    s['status'] = 'up'
                    s['status_conf'] = 0.9
                else:
                    s['status'] = 'down'
                    s['status_conf'] = 0.8

    # Enforce DOWN means zero rates for any other status changes
    for s in state.values():
        if s['status'] == 'down':
            s['rx'] = 0.0
            s['tx'] = 0.0
            s['rx_conf'] = 1.0
            s['tx_conf'] = 1.0

    # Helper: Calculate target flow for a router interface to achieve balance
    def get_target_flow(router_id, exclude_if_id, direction):
        if not router_id or router_id not in topology:
            return None
        sum_in = 0.0
        sum_out = 0.0
        valid = True
        for rid in topology[router_id]:
            if rid not in state:
                valid = False; break
            if rid == exclude_if_id:
                if direction == 'in': sum_out += state[rid]['tx']
                else: sum_in += state[rid]['rx']
            else:
                sum_in += state[rid]['rx']
                sum_out += state[rid]['tx']

        if not valid: return None
        return max(0.0, sum_out - sum_in) if direction == 'in' else max(0.0, sum_in - sum_out)

    # Pass 2: Rate Repair (Iterative)
    for _ in range(2):
        for if_id in sorted_ifs:
            s = state[if_id]
            if s['status'] == 'down': continue

            peer_id = s['orig'].get('connected_to')
            if not peer_id or peer_id not in state: continue
            peer = state[peer_id]
            if peer['status'] == 'down': continue # Should be handled, but safe check

            # --- Link Flow 1: My RX should match Peer TX ---
            val_rx = s['rx']
            val_ptx = peer['tx']
            avg = (val_rx + val_ptx) / 2.0
            base = max(avg, 10.0)

            if abs(val_rx - val_ptx) / base > HARDENING_THRESHOLD:
                # Violation
                t_local_rx = get_target_flow(s['orig'].get('local_router'), if_id, 'in')
                t_remote_tx = get_target_flow(peer['orig'].get('local_router'), peer_id, 'out')

                candidates = [val_rx, val_ptx]
                best_v = val_rx
                min_cost = float('inf')
                valid_constraints = 0

                for v in candidates:
                    cost = 0.0
                    cnt = 0
                    if t_local_rx is not None:
                        cost += abs(v - t_local_rx); cnt += 1
                    if t_remote_tx is not None:
                        cost += abs(v - t_remote_tx); cnt += 1

                    if cnt > 0:
                        if cost < min_cost:
                            min_cost = cost; best_v = v; valid_constraints = cnt
                        elif cost == min_cost:
                            best_v = min(best_v, v)

                if valid_constraints > 0:
                    norm_err = min_cost / (valid_constraints * base)
                    # Sigmoid confidence decay
                    conf = 1.0 / (1.0 + 20.0 * norm_err)
                else:
                    # No constraints: Heuristic "RX cannot > Peer TX"
                    if val_rx > val_ptx:
                        best_v = val_ptx
                        conf = 0.8
                    else:
                        best_v = (val_rx + val_ptx) / 2.0
                        conf = 0.5

                s['rx'] = best_v
                peer['tx'] = best_v
                s['rx_conf'] = conf
                peer['tx_conf'] = conf
            else:
                s['rx_conf'] = 1.0
                peer['tx_conf'] = 1.0
                # Smooth small noise
                avg_v = (val_rx + val_ptx) / 2.0
                s['rx'] = avg_v
                peer['tx'] = avg_v

            # --- Link Flow 2: My TX should match Peer RX ---
            val_tx = s['tx']
            val_prx = peer['rx']
            avg = (val_tx + val_prx) / 2.0
            base = max(avg, 10.0)

            if abs(val_tx - val_prx) / base > HARDENING_THRESHOLD:
                t_local_tx = get_target_flow(s['orig'].get('local_router'), if_id, 'out')
                t_remote_rx = get_target_flow(peer['orig'].get('local_router'), peer_id, 'in')

                candidates = [val_tx, val_prx]
                best_v = val_tx
                min_cost = float('inf')
                valid_constraints = 0

                for v in candidates:
                    cost = 0.0
                    cnt = 0
                    if t_local_tx is not None:
                        cost += abs(v - t_local_tx); cnt += 1
                    if t_remote_rx is not None:
                        cost += abs(v - t_remote_rx); cnt += 1

                    if cnt > 0:
                        if cost < min_cost:
                            min_cost = cost; best_v = v; valid_constraints = cnt
                        elif cost == min_cost:
                            best_v = min(best_v, v)

                if valid_constraints > 0:
                    norm_err = min_cost / (valid_constraints * base)
                    conf = 1.0 / (1.0 + 20.0 * norm_err)
                else:
                    # No constraints: Heuristic "Peer RX cannot > My TX"
                    if val_prx > val_tx:
                        best_v = val_tx
                        conf = 0.8
                    else:
                        best_v = val_tx # Trust Source
                        conf = 0.7

                s['tx'] = best_v
                peer['rx'] = best_v
                s['tx_conf'] = conf
                peer['rx_conf'] = conf
            else:
                s['tx_conf'] = 1.0
                peer['rx_conf'] = 1.0
                avg_v = (val_tx + val_prx) / 2.0
                s['tx'] = avg_v
                peer['rx'] = avg_v

    # Final result construction
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