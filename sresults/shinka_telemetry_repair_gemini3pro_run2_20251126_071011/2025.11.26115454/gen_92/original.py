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
        # If the link is symmetric, we trust it more initially.
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
            'orig': data
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

    # Pass 2: Rate Repair with Symmetry & Flow Conservation

    # Helper: Calculate router flow imbalance (In - Out)
    def get_router_imbalance(router_id):
        if not router_id or router_id not in topology:
            return None
        in_sum = 0.0
        out_sum = 0.0
        for if_id in topology[router_id]:
            if if_id in state:
                in_sum += state[if_id]['rx']
                out_sum += state[if_id]['tx']
        return in_sum - out_sum

    # Helper: Calculate router reliability (Average confidence of *other* interfaces)
    def get_router_reliability(router_id, exclude_if_id):
        if not router_id or router_id not in topology:
            return 0.0 # No info -> unreliable constraint

        total_conf = 0.0
        count = 0
        for if_id in topology[router_id]:
            if if_id == exclude_if_id:
                continue
            if if_id in state:
                # Use average of RX/TX conf as proxy for interface health
                c = (state[if_id]['rx_conf'] + state[if_id]['tx_conf']) / 2.0
                total_conf += c
                count += 1

        if count == 0:
            return 0.0 # Isolated link, cannot verify flow
        return total_conf / count

    # Iterative refinement (Gauss-Seidel style)
    # Allows flow corrections to propagate through the network
    ITERATIONS = 4
    sorted_interfaces = sorted(state.keys())  # Deterministic order

    for iteration in range(ITERATIONS):
        # We track processed PAIRS per iteration to avoid double processing
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

            # Skip if both down
            if s['status'] == 'down' and peer['status'] == 'down':
                continue

            # --- 1. Fix Direction A: Local TX -> Peer RX ---
            # Candidates
            c_tx = s['tx']
            c_prx = peer['rx']

            # Context
            r_local = s['orig'].get('local_router')
            r_remote = peer['orig'].get('local_router')

            # Reliability of routers (used as weights)
            w_local = get_router_reliability(r_local, if_id)
            w_remote = get_router_reliability(r_remote, connected_to)

            # Boost weights slightly to avoid zero
            w_local = max(w_local, 0.1)
            w_remote = max(w_remote, 0.1)

            # Imbalances
            imb_local = get_router_imbalance(r_local)
            imb_remote = get_router_imbalance(r_remote)

            # Cost function with Quality Weights
            def eval_direction_a(v):
                cost = 0.0
                valid_checks = 0
                if imb_local is not None:
                    # w_local * deviation
                    cost += w_local * abs(imb_local + c_tx - v)
                    valid_checks += 1
                if imb_remote is not None:
                    cost += w_remote * abs(imb_remote - c_prx + v)
                    valid_checks += 1

                # Small penalty for deviating from measurements to break ties
                # Penalize deviating from TX slightly more (Source of Truth bias)
                cost += 0.001 * abs(v - c_tx)
                cost += 0.001 * abs(v - c_prx)

                return cost, valid_checks

            cost_tx, n_tx = eval_direction_a(c_tx)
            cost_prx, n_prx = eval_direction_a(c_prx)

            # Decision Logic
            best_val = c_tx
            conf = 0.5 # Default

            denom = max(c_tx, c_prx, 1.0)
            diff = abs(c_tx - c_prx)

            # If signals agree, trust them
            if diff / denom <= HARDENING_THRESHOLD:
                best_val = (c_tx + c_prx) / 2.0
                conf = 1.0
            elif n_tx == 0:
                # No topology info, fallback to average
                best_val = (c_tx + c_prx) / 2.0
                conf = max(0.0, 1.0 - (diff / denom))
            else:
                # Topology info available, check costs
                if cost_tx < cost_prx:
                    best_val = c_tx
                else:
                    best_val = c_prx

                # Hybrid Confidence Calibration
                # Calculate "Solidity": How well does best_val fit flow?

                # Local check
                err_local = float('inf')
                if imb_local is not None:
                     err_local = abs(imb_local + c_tx - best_val) / max(best_val, 1.0)

                # Remote check
                err_remote = float('inf')
                if imb_remote is not None:
                     err_remote = abs(imb_remote - c_prx + best_val) / max(best_val, 1.0)

                is_solid_local = (err_local < HARDENING_THRESHOLD) and (imb_local is not None)
                is_solid_remote = (err_remote < HARDENING_THRESHOLD) and (imb_remote is not None)

                # Tiered Base Confidence
                base_conf = 0.5
                if is_solid_local and is_solid_remote:
                    base_conf = 0.95
                elif is_solid_local:
                    # Trust local if local is reliable
                    base_conf = 0.7 + (0.2 * w_local)
                elif is_solid_remote:
                    # Trust remote if remote is reliable
                    base_conf = 0.7 + (0.2 * w_remote)

                # Residual Penalty
                min_err = min(err_local, err_remote)
                if min_err == float('inf'): min_err = 0.0

                conf = base_conf - min(0.4, min_err)
                conf = max(0.0, min(1.0, conf))

            # Apply
            s['tx'] = best_val
            peer['rx'] = best_val
            s['tx_conf'] = conf
            peer['rx_conf'] = conf


            # --- 2. Fix Direction B: Local RX <- Peer TX ---
            c_rx = s['rx']
            c_ptx = peer['tx']

            imb_local = get_router_imbalance(r_local)
            imb_remote = get_router_imbalance(r_remote)

            def eval_direction_b(v):
                cost = 0.0
                valid_checks = 0
                if imb_local is not None:
                    cost += w_local * abs(imb_local - c_rx + v)
                    valid_checks += 1
                if imb_remote is not None:
                    cost += w_remote * abs(imb_remote + c_ptx - v)
                    valid_checks += 1

                cost += 0.001 * abs(v - c_rx)
                cost += 0.001 * abs(v - c_ptx)

                return cost, valid_checks

            cost_rx, n_rx = eval_direction_b(c_rx)
            cost_ptx, n_ptx = eval_direction_b(c_ptx)

            best_val_b = c_rx
            conf_b = 0.5

            denom_b = max(c_rx, c_ptx, 1.0)
            diff_b = abs(c_rx - c_ptx)

            if diff_b / denom_b <= HARDENING_THRESHOLD:
                best_val_b = (c_rx + c_ptx) / 2.0
                conf_b = 1.0
            elif n_rx == 0:
                best_val_b = (c_rx + c_ptx) / 2.0
                conf_b = max(0.0, 1.0 - (diff_b / denom_b))
            else:
                if cost_rx < cost_ptx:
                    best_val_b = c_rx
                else:
                    best_val_b = c_ptx

                err_local = float('inf')
                if imb_local is not None:
                     err_local = abs(imb_local - c_rx + best_val_b) / max(best_val_b, 1.0)

                err_remote = float('inf')
                if imb_remote is not None:
                     err_remote = abs(imb_remote + c_ptx - best_val_b) / max(best_val_b, 1.0)

                is_solid_local = (err_local < HARDENING_THRESHOLD) and (imb_local is not None)
                is_solid_remote = (err_remote < HARDENING_THRESHOLD) and (imb_remote is not None)

                base_conf = 0.5
                if is_solid_local and is_solid_remote:
                    base_conf = 0.95
                elif is_solid_local:
                    base_conf = 0.7 + (0.2 * w_local)
                elif is_solid_remote:
                    base_conf = 0.7 + (0.2 * w_remote)

                min_err = min(err_local, err_remote)
                if min_err == float('inf'): min_err = 0.0

                conf_b = base_conf - min(0.4, min_err)
                conf_b = max(0.0, min(1.0, conf_b))

            s['rx'] = best_val_b
            peer['tx'] = best_val_b
            s['rx_conf'] = conf_b
            peer['tx_conf'] = conf_b

    # Assemble result
    result = {}
    for if_id, s in state.items():
        orig = s['orig']

        # If we are very confident about rates, we should be confident about status
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