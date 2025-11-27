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

    # Iterative refinement (Gauss-Seidel style) with Flow-Implied Candidates
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
            # Current values
            c_tx = s['tx']
            c_prx = peer['rx']

            # Router contexts
            r_local = s['orig'].get('local_router')
            imb_local = get_router_imbalance(r_local) # Current imbalance

            r_remote = peer['orig'].get('local_router')
            imb_remote = get_router_imbalance(r_remote) # Current imbalance

            # Determine candidates
            candidates = {c_tx, c_prx}

            # Flow-implied candidates (what would zero the imbalance?)
            # Local (TX is Out): Imb = In - Out.
            # Imb_new = Imb_old + c_tx - v = 0 => v = Imb_old + c_tx
            if imb_local is not None:
                target_local = imb_local + c_tx
                if target_local > 0: candidates.add(target_local)

            # Remote (RX is In): Imb = In - Out.
            # Imb_new = Imb_old - c_prx + v = 0 => v = c_prx - Imb_old
            if imb_remote is not None:
                target_remote = c_prx - imb_remote
                if target_remote > 0: candidates.add(target_remote)

            # Cost function: Balance Flow + Anchor to Reality
            def eval_direction_a(v):
                flow_cost = 0.0
                valid_checks = 0
                if imb_local is not None:
                    flow_cost += abs(imb_local + c_tx - v)
                    valid_checks += 1
                if imb_remote is not None:
                    flow_cost += abs(imb_remote - c_prx + v)
                    valid_checks += 1

                # Anchor cost: Penalty for deviating from measurements
                # We trust measurements more than "hallucinating" values, unless flow strict.
                dist_tx = abs(v - c_tx)
                dist_prx = abs(v - c_prx)
                anchor_cost = min(dist_tx, dist_prx)

                # Weighting: Flow is primary (1.0), Anchor is secondary (0.01)
                return flow_cost + 0.01 * anchor_cost, valid_checks

            # Find best candidate
            best_val = c_tx
            min_cost = float('inf')

            # Default to average if measurements agree (shortcut)
            denom = max(c_tx, c_prx, 1.0)
            if abs(c_tx - c_prx) / denom <= HARDENING_THRESHOLD:
                 best_val = (c_tx + c_prx) / 2.0
                 min_cost = 0 # effectively
            else:
                for cand in candidates:
                    cost, checks = eval_direction_a(cand)
                    if checks == 0:
                        cost = abs(cand - (c_tx + c_prx)/2.0)

                    if cost < min_cost:
                        min_cost = cost
                        best_val = cand

            # Confidence Calculation
            conf = 0.5

            # Check agreement with measurements
            agrees_tx = abs(best_val - c_tx) / max(best_val, 1.0) < HARDENING_THRESHOLD
            agrees_prx = abs(best_val - c_prx) / max(best_val, 1.0) < HARDENING_THRESHOLD

            if agrees_tx and agrees_prx:
                conf = 1.0
            else:
                # Check solidity of flow fit
                w_cost_local = 0.0
                w_cost_remote = 0.0
                if imb_local is not None:
                    w_cost_local = abs(imb_local + c_tx - best_val)
                if imb_remote is not None:
                    w_cost_remote = abs(imb_remote - c_prx + best_val)

                err_local = w_cost_local / max(best_val, 1.0)
                err_remote = w_cost_remote / max(best_val, 1.0)

                # Thresholds for "Solid" fit
                solid_local = (err_local < HARDENING_THRESHOLD) and (imb_local is not None)
                solid_remote = (err_remote < HARDENING_THRESHOLD) and (imb_remote is not None)

                if solid_local and solid_remote:
                    # Fits both routers perfectly -> High confidence even if meas disagree
                    conf = 0.95
                elif solid_local or solid_remote:
                    # Fits one router perfectly -> Good confidence
                    conf = 0.85
                    # Bonus if it matches one measurement
                    if agrees_tx or agrees_prx:
                        conf = 0.95
                else:
                    # Matches neither flow nor both measurements
                    conf = 0.5
                    if agrees_tx or agrees_prx:
                         conf = 0.7

            s['tx'] = best_val
            peer['rx'] = best_val
            s['tx_conf'] = conf
            peer['rx_conf'] = conf


            # --- 2. Fix Direction B: Local RX <- Peer TX ---
            c_rx = s['rx']
            c_ptx = peer['tx']

            imb_local = get_router_imbalance(r_local)
            imb_remote = get_router_imbalance(r_remote)

            candidates_b = {c_rx, c_ptx}

            # Local (RX is In): Imb = In - Out.
            # Imb_new = Imb_old - c_rx + v = 0 => v = c_rx - Imb_old
            if imb_local is not None:
                target_local_b = c_rx - imb_local
                if target_local_b > 0: candidates_b.add(target_local_b)

            # Remote (TX is Out): Imb = In - Out.
            # Imb_new = Imb_old + c_ptx - v = 0 => v = Imb_old + c_ptx
            if imb_remote is not None:
                target_remote_b = imb_remote + c_ptx
                if target_remote_b > 0: candidates_b.add(target_remote_b)

            def eval_direction_b(v):
                flow_cost = 0.0
                valid_checks = 0
                if imb_local is not None:
                    flow_cost += abs(imb_local - c_rx + v)
                    valid_checks += 1
                if imb_remote is not None:
                    flow_cost += abs(imb_remote + c_ptx - v)
                    valid_checks += 1

                dist_rx = abs(v - c_rx)
                dist_ptx = abs(v - c_ptx)
                anchor_cost = min(dist_rx, dist_ptx)

                return flow_cost + 0.01 * anchor_cost, valid_checks

            best_val_b = c_rx
            min_cost_b = float('inf')

            denom_b = max(c_rx, c_ptx, 1.0)
            if abs(c_rx - c_ptx) / denom_b <= HARDENING_THRESHOLD:
                 best_val_b = (c_rx + c_ptx) / 2.0
                 min_cost_b = 0
            else:
                for cand in candidates_b:
                    cost, checks = eval_direction_b(cand)
                    if checks == 0:
                         cost = abs(cand - (c_rx + c_ptx)/2.0)

                    if cost < min_cost_b:
                        min_cost_b = cost
                        best_val_b = cand

            # Confidence B
            conf_b = 0.5
            agrees_rx = abs(best_val_b - c_rx) / max(best_val_b, 1.0) < HARDENING_THRESHOLD
            agrees_ptx = abs(best_val_b - c_ptx) / max(best_val_b, 1.0) < HARDENING_THRESHOLD

            if agrees_rx and agrees_ptx:
                conf_b = 1.0
            else:
                w_cost_local = 0.0
                w_cost_remote = 0.0
                if imb_local is not None:
                    w_cost_local = abs(imb_local - c_rx + best_val_b)
                if imb_remote is not None:
                    w_cost_remote = abs(imb_remote + c_ptx - best_val_b)

                err_local = w_cost_local / max(best_val_b, 1.0)
                err_remote = w_cost_remote / max(best_val_b, 1.0)

                solid_local = (err_local < HARDENING_THRESHOLD) and (imb_local is not None)
                solid_remote = (err_remote < HARDENING_THRESHOLD) and (imb_remote is not None)

                if solid_local and solid_remote:
                    conf_b = 0.95
                elif solid_local or solid_remote:
                    conf_b = 0.85
                    if agrees_rx or agrees_ptx:
                        conf_b = 0.95
                else:
                    conf_b = 0.5
                    if agrees_rx or agrees_ptx:
                         conf_b = 0.7

            s['rx'] = best_val_b
            peer['tx'] = best_val_b
            s['rx_conf'] = conf_b
            peer['tx_conf'] = conf_b

    # Assemble result
    result = {}
    for if_id, s in state.items():
        orig = s['orig']
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