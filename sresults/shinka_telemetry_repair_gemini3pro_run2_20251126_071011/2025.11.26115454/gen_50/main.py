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
        state[if_id] = {
            'rx': float(data.get('rx_rate', 0.0)),
            'tx': float(data.get('tx_rate', 0.0)),
            'status': data.get('interface_status', 'unknown'),
            'rx_conf': 1.0,
            'tx_conf': 1.0,
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

    # Iterative refinement (Gauss-Seidel style)
    # Allows flow corrections to propagate through the network
    ITERATIONS = 3
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
            imb_local = get_router_imbalance(r_local) # Current imbalance including c_tx

            r_remote = peer['orig'].get('local_router')
            imb_remote = get_router_imbalance(r_remote) # Current imbalance including c_prx

            # Inferred Budgets (Flow Conservation Targets)
            # What value would balance each router perfectly?
            budget_local = None
            if imb_local is not None:
                # Imb = In - Out. Out includes c_tx.
                # Wanted: In - (Out_others + v) = 0 => v = Imb + c_tx
                budget_local = max(0.0, imb_local + c_tx)

            budget_remote = None
            if imb_remote is not None:
                # Imb = In - Out. In includes c_prx.
                # Wanted: (In_others + v) - Out = 0 => v = c_prx - Imb
                budget_remote = max(0.0, c_prx - imb_remote)

            # Build Candidates List
            candidates = [c_tx, c_prx]

            # 1. Consensus Candidate: Both routers agree on a value
            if budget_local is not None and budget_remote is not None:
                avg_budget = (budget_local + budget_remote) / 2.0
                # If budgets agree within tolerance, this is a very strong signal
                if abs(budget_local - budget_remote) / max(avg_budget, 1.0) < HARDENING_THRESHOLD * 2:
                    candidates.append(avg_budget)

            # 2. Individual Budgets (to see if they confirm a measurement)
            if budget_local is not None: candidates.append(budget_local)
            if budget_remote is not None: candidates.append(budget_remote)

            # Scoring Function
            def eval_direction_a(v):
                cost = 0.0
                valid_checks = 0
                if imb_local is not None:
                    cost += abs(imb_local + c_tx - v)
                    valid_checks += 1
                if imb_remote is not None:
                    cost += abs(imb_remote - c_prx + v)
                    valid_checks += 1
                return cost, valid_checks

            # Decision Logic
            best_val = c_tx
            conf = 1.0

            # Check for Link Symmetry first (Ground Truth if consistent)
            denom = max(c_tx, c_prx, 1.0)
            if abs(c_tx - c_prx) / denom <= HARDENING_THRESHOLD:
                best_val = (c_tx + c_prx) / 2.0
                winner_cost, _ = eval_direction_a(best_val)
                # Calculate loser cost based on hypothetical deviation (not used for high conf)
                loser_cost = winner_cost
                conf = 1.0
            else:
                # Check for Topology Availability
                _, n_checks = eval_direction_a(0.0)
                if n_checks == 0:
                    best_val = (c_tx + c_prx) / 2.0
                    winner_cost = 0; loser_cost = 0
                    conf = 0.5
                else:
                    # Evaluate all candidates
                    min_cost = float('inf')
                    unique_candidates = sorted(list(set(candidates)))

                    for cand in unique_candidates:
                        cost, _ = eval_direction_a(cand)
                        if cost < min_cost:
                            min_cost = cost
                            best_val = cand

                    winner_cost = min_cost
                    # Loser cost is the max cost of the raw measurements
                    # This represents the "cost of being wrong" if we picked the wrong measurement
                    cost_tx, _ = eval_direction_a(c_tx)
                    cost_prx, _ = eval_direction_a(c_prx)
                    loser_cost = max(cost_tx, cost_prx)

                # Confidence Calibration
                # Check for "Solidity" (Recommendation 2):
                # If the chosen value results in perfect balance at one router,
                # it is highly trustworthy even if the other router is noisy.

                w_cost_local = 0.0
                w_cost_remote = 0.0
                if imb_local is not None:
                    w_cost_local = abs(imb_local + c_tx - best_val)
                if imb_remote is not None:
                    w_cost_remote = abs(imb_remote - c_prx + best_val)

                err_local = w_cost_local / max(best_val, 1.0)
                err_remote = w_cost_remote / max(best_val, 1.0)

                is_solid_local = (err_local < HARDENING_THRESHOLD) and (imb_local is not None)
                is_solid_remote = (err_remote < HARDENING_THRESHOLD) and (imb_remote is not None)

                if is_solid_local and is_solid_remote:
                    conf = 1.0
                elif is_solid_local or is_solid_remote:
                    conf = 0.9
                else:
                    # Fallback to standard scoring
                    margin = (loser_cost - winner_cost) / max(winner_cost + loser_cost, 1.0)
                    residual_ratio = winner_cost / max(best_val, 1.0)

                    distinctness_score = min(1.0, margin * 2.0)
                    fit_score = max(0.0, 1.0 - residual_ratio * 2.0)

                    conf = 0.5 + 0.45 * distinctness_score * fit_score

            # Apply
            s['tx'] = best_val
            peer['rx'] = best_val
            s['tx_conf'] = conf
            peer['rx_conf'] = conf


            # --- 2. Fix Direction B: Local RX <- Peer TX ---
            c_rx = s['rx']
            c_ptx = peer['tx']

            # Re-fetch imbalances as they might have changed from previous step
            imb_local = get_router_imbalance(r_local)
            imb_remote = get_router_imbalance(r_remote)

            # Inferred Budgets
            budget_local_b = None
            if imb_local is not None:
                # Imb = In - Out. In includes c_rx.
                # Wanted: (In_others + v) - Out = 0 => v = c_rx - Imb
                budget_local_b = max(0.0, c_rx - imb_local)

            budget_remote_b = None
            if imb_remote is not None:
                # Imb = In - Out. Out includes c_ptx.
                # Wanted: In - (Out_others + v) = 0 => v = Imb + c_ptx
                budget_remote_b = max(0.0, imb_remote + c_ptx)

            candidates_b = [c_rx, c_ptx]
            if budget_local_b is not None and budget_remote_b is not None:
                avg_budget = (budget_local_b + budget_remote_b) / 2.0
                if abs(budget_local_b - budget_remote_b) / max(avg_budget, 1.0) < HARDENING_THRESHOLD * 2:
                    candidates_b.append(avg_budget)
            if budget_local_b is not None: candidates_b.append(budget_local_b)
            if budget_remote_b is not None: candidates_b.append(budget_remote_b)

            def eval_direction_b(v):
                cost = 0.0
                valid_checks = 0
                if imb_local is not None:
                    cost += abs(imb_local - c_rx + v)
                    valid_checks += 1
                if imb_remote is not None:
                    cost += abs(imb_remote + c_ptx - v)
                    valid_checks += 1
                return cost, valid_checks

            best_val_b = c_rx
            conf_b = 1.0

            denom_b = max(c_rx, c_ptx, 1.0)
            if abs(c_rx - c_ptx) / denom_b <= HARDENING_THRESHOLD:
                best_val_b = (c_rx + c_ptx) / 2.0
                winner_cost, _ = eval_direction_b(best_val_b)
                loser_cost = winner_cost
                conf_b = 1.0
            else:
                _, n_checks = eval_direction_b(0.0)
                if n_checks == 0:
                    best_val_b = (c_rx + c_ptx) / 2.0
                    winner_cost = 0; loser_cost = 0
                    conf_b = 0.5
                else:
                    min_cost = float('inf')
                    unique_candidates = sorted(list(set(candidates_b)))

                    for cand in unique_candidates:
                        cost, _ = eval_direction_b(cand)
                        if cost < min_cost:
                            min_cost = cost
                            best_val_b = cand

                    winner_cost = min_cost
                    cost_rx, _ = eval_direction_b(c_rx)
                    cost_ptx, _ = eval_direction_b(c_ptx)
                    loser_cost = max(cost_rx, cost_ptx)

                # Confidence Calibration (B)
                w_cost_local = 0.0
                w_cost_remote = 0.0
                if imb_local is not None:
                    w_cost_local = abs(imb_local - c_rx + best_val_b)
                if imb_remote is not None:
                    w_cost_remote = abs(imb_remote + c_ptx - best_val_b)

                err_local = w_cost_local / max(best_val_b, 1.0)
                err_remote = w_cost_remote / max(best_val_b, 1.0)

                is_solid_local = (err_local < HARDENING_THRESHOLD) and (imb_local is not None)
                is_solid_remote = (err_remote < HARDENING_THRESHOLD) and (imb_remote is not None)

                if is_solid_local and is_solid_remote:
                    conf_b = 1.0
                elif is_solid_local or is_solid_remote:
                    conf_b = 0.9
                else:
                    margin = (loser_cost - winner_cost) / max(winner_cost + loser_cost, 1.0)
                    residual_ratio = winner_cost / max(best_val_b, 1.0)

                    distinctness_score = min(1.0, margin * 2.0)
                    fit_score = max(0.0, 1.0 - residual_ratio * 2.0)

                    conf_b = 0.5 + 0.45 * distinctness_score * fit_score

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