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
            'rx_conf': 0.5, # Initialize as uncertain
            'tx_conf': 0.5,
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
                    s['status_conf'] = 0.9
                else:
                    s['status'] = 'down'
                    s['status_conf'] = 0.9

        # Enforce DOWN means zero rates
        if s['status'] == 'down':
            s['rx'] = 0.0
            s['tx'] = 0.0
            s['rx_conf'] = 1.0
            s['tx_conf'] = 1.0

    # Pass 2: Anchor Reliable Links (Pre-pass)
    # If links are symmetric within tolerance, we trust them.
    # This provides a stable base for the flow solver.
    for if_id, s in state.items():
        if s['status'] == 'down': continue
        connected_to = s['orig'].get('connected_to')
        if connected_to and connected_to in state:
            peer = state[connected_to]

            # Check TX->RX
            tx = s['tx']
            prx = peer['rx']
            denom = max(tx, prx, 1.0)
            if abs(tx - prx) / denom <= HARDENING_THRESHOLD:
                avg = (tx + prx) / 2.0
                s['tx'] = avg
                peer['rx'] = avg
                s['tx_conf'] = 1.0
                peer['rx_conf'] = 1.0

            # Check RX<-TX
            rx = s['rx']
            ptx = peer['tx']
            denom = max(rx, ptx, 1.0)
            if abs(rx - ptx) / denom <= HARDENING_THRESHOLD:
                avg = (rx + ptx) / 2.0
                s['rx'] = avg
                peer['tx'] = avg
                s['rx_conf'] = 1.0
                peer['tx_conf'] = 1.0

    # Pass 3: Rate Repair with Symmetry & Flow Conservation

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
    ITERATIONS = 5
    sorted_interfaces = sorted(state.keys())  # Deterministic order

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

            # Shared Arbitration Logic
            def arbitrate_flow(meas_src, meas_dst, router_src, router_dst, component_src, component_dst):
                # meas_src: e.g., local TX
                # meas_dst: e.g., peer RX
                # component_src: 'out' if TX, 'in' if RX (contribution sign to Imb)
                # component_dst: 'in' if RX, 'out' if TX

                imb_src = get_router_imbalance(router_src)
                imb_dst = get_router_imbalance(router_dst)

                candidates = {meas_src, meas_dst}
                # Add Average
                candidates.add((meas_src + meas_dst) / 2.0)

                # Add Implied Candidates (Flow Conservation Targets)
                # If we assume all other links are correct, what must this value be?

                # Src Target: Imb_Src = In - Out.
                # If TX (Out): NewImb = (OldImb - (-meas_src)) + (-v) = OldImb + meas_src - v = 0 => v = OldImb + meas_src
                # If RX (In):  NewImb = (OldImb - meas_src) + v = 0 => v = meas_src - OldImb
                if imb_src is not None:
                    if component_src == 'out': # TX
                        target = imb_src + meas_src
                    else: # RX
                        target = meas_src - imb_src
                    if target >= 0: candidates.add(target)

                # Dst Target:
                if imb_dst is not None:
                    if component_dst == 'out': # TX
                        target = imb_dst + meas_dst
                    else: # RX
                        target = meas_dst - imb_dst
                    if target >= 0: candidates.add(target)

                best_v = meas_src
                min_cost = float('inf')

                for v in candidates:
                    cost = 0.0
                    # Physics check: if v > capacity or significantly different from both measurements without strong flow reason
                    # Simple cost: sum of residual imbalances

                    if imb_src is not None:
                        # Reconstruct new imbalance based on v
                        # If Out (TX): delta = -v - (-meas_src) = meas_src - v
                        # If In (RX): delta = v - meas_src
                        delta = (meas_src - v) if component_src == 'out' else (v - meas_src)
                        cost += abs(imb_src + delta)

                    if imb_dst is not None:
                        delta = (meas_dst - v) if component_dst == 'out' else (v - meas_dst)
                        cost += abs(imb_dst + delta)

                    # Measurement fidelity cost (regularization)
                    # We trust measurements somewhat.
                    # If flow conservation is perfect at 0 but measurements say 100, we shouldn't just pick 0 unless strongly supported.
                    # Add small penalty for deviating from measurements
                    cost += 0.05 * (abs(v - meas_src) + abs(v - meas_dst))

                    if cost < min_cost:
                        min_cost = cost
                        best_v = v

                # Confidence Calculation
                # 1. Measurement Agreement
                denom = max(meas_src, meas_dst, 1.0)
                agreement = abs(meas_src - meas_dst) / denom < HARDENING_THRESHOLD

                # 2. Flow solidity (how well does best_v balance routers?)
                solid_src = False
                solid_dst = False

                if imb_src is not None:
                    delta = (meas_src - best_v) if component_src == 'out' else (best_v - meas_src)
                    resid_src = abs(imb_src + delta)
                    if resid_src / max(best_v, 1.0) < HARDENING_THRESHOLD:
                        solid_src = True

                if imb_dst is not None:
                    delta = (meas_dst - best_v) if component_dst == 'out' else (best_v - meas_dst)
                    resid_dst = abs(imb_dst + delta)
                    if resid_dst / max(best_v, 1.0) < HARDENING_THRESHOLD:
                        solid_dst = True

                conf = 0.5
                if solid_src and solid_dst:
                    conf = 1.0
                elif (solid_src or solid_dst) and agreement:
                    conf = 0.98
                elif solid_src or solid_dst:
                    conf = 0.90
                elif agreement:
                    conf = 0.8
                else:
                    # Low confidence if nothing matches
                    conf = 0.4
                    # slightly higher if we found a flow compromise
                    if imb_src is not None or imb_dst is not None:
                         conf = 0.6

                return best_v, conf

            # --- 1. Fix Direction A: Local TX -> Peer RX ---
            r_local = s['orig'].get('local_router')
            r_remote = peer['orig'].get('local_router')

            best_val, conf = arbitrate_flow(
                s['tx'], peer['rx'],
                r_local, r_remote,
                'out', 'in'
            )

            s['tx'] = best_val
            peer['rx'] = best_val
            s['tx_conf'] = conf
            peer['rx_conf'] = conf

            # --- 2. Fix Direction B: Local RX <- Peer TX ---
            best_val_b, conf_b = arbitrate_flow(
                s['rx'], peer['tx'],
                r_local, r_remote,
                'in', 'out'
            )

            s['rx'] = best_val_b
            peer['tx'] = best_val_b
            s['rx_conf'] = conf_b
            peer['tx_conf'] = conf_b

    # Assemble result
    result = {}
    for if_id, s in state.items():
        orig = s['orig']

        # If we are very confident about rates, we should be confident about status
        if s['rx_conf'] > 0.9 and s['tx_conf'] > 0.9:
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