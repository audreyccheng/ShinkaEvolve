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
    Repair network interface telemetry using Iterative Flow Optimization.
    Refines telemetry data through multiple passes of constraint satisfaction,
    solving for the most likely network state that satisfies Symmetry and Flow Conservation.
    """
    # Constants
    HARDENING_THRESHOLD = 0.02   # 2% tolerance for measurement timing
    TRAFFIC_THRESHOLD = 1.0      # 1 Mbps threshold for "active" link
    ITERATIONS = 5               # Propagation passes

    # --- Phase 1: Initialization & Status Repair ---
    state = {}

    # Identify Verified Routers (Full Visibility)
    # We only apply flow conservation constraints to routers where we see ALL interfaces.
    verified_routers = set()
    for r_id, ifaces in topology.items():
        if all(i in telemetry for i in ifaces):
            verified_routers.add(r_id)

    for iface_id, data in telemetry.items():
        raw_rx = data.get('rx_rate', 0.0)
        raw_tx = data.get('tx_rate', 0.0)
        raw_status = data.get('interface_status', 'unknown')

        peer_id = data.get('connected_to')
        peer_data = telemetry.get(peer_id) if (peer_id and peer_id in telemetry) else {}

        # Traffic Evidence
        signals = [raw_rx, raw_tx, peer_data.get('rx_rate', 0.0), peer_data.get('tx_rate', 0.0)]
        max_traffic = max(signals) if signals else 0.0

        # Status Repair: Traffic presence overrides 'down' status
        status = raw_status
        status_conf = 1.0

        if max_traffic > TRAFFIC_THRESHOLD:
            if raw_status != 'up':
                status = 'up'
                status_conf = 0.95
        elif raw_status == 'up' and peer_data.get('interface_status') == 'down':
            # Peer says DOWN, I say UP, but no traffic -> Likely DOWN
            status = 'down'
            status_conf = 0.8

        # Initial Rate Beliefs
        if status == 'down':
            cur_rx, cur_tx = 0.0, 0.0
        else:
            cur_rx, cur_tx = raw_rx, raw_tx

        state[iface_id] = {
            'rx': cur_rx,
            'tx': cur_tx,
            'status': status,
            'status_conf': status_conf,
            'orig_rx': raw_rx,
            'orig_tx': raw_tx,
            'orig_status': raw_status,
            'peer_id': peer_id,
            'local_router': data.get('local_router')
        }

    # --- Phase 2: Iterative Constraint Satisfaction ---
    for _ in range(ITERATIONS):
        next_rates = {}

        # 1. Calculate Router Balances (Flow Hints) for verified routers only
        router_balances = {}
        for r_id in verified_routers:
            ifaces = topology[r_id]
            sum_rx = sum(state[i]['rx'] for i in ifaces) # Safe to look up as all checked in phase 1
            sum_tx = sum(state[i]['tx'] for i in ifaces)
            router_balances[r_id] = {'rx': sum_rx, 'tx': sum_tx}

        # 2. Evaluate each interface
        for iface_id, curr in state.items():
            if curr['status'] == 'down':
                next_rates[iface_id] = {'rx': 0.0, 'tx': 0.0}
                continue

            peer_id = curr['peer_id']
            has_peer = peer_id and peer_id in state
            r_id = curr.get('local_router')

            def resolve_rate(local_val, peer_val, is_rx):
                # Calculate Flow Hint (Strict Validation)
                val_hint = None
                if r_id in router_balances:
                    rb = router_balances[r_id]
                    if is_rx:
                        val_hint = max(0.0, rb['tx'] - (rb['rx'] - local_val))
                    else:
                        val_hint = max(0.0, rb['rx'] - (rb['tx'] - local_val))

                # Adaptive Tolerance
                # Use relaxed threshold for low-traffic links to handle noise/bursts
                adaptive_thresh = HARDENING_THRESHOLD
                max_val = max(local_val, peer_val, 1.0)
                if max_val < 50.0:
                    adaptive_thresh = 0.05

                # 1. Check Symmetry
                denom_sym = max(local_val, peer_val, 1.0)
                diff_sym = abs(local_val - peer_val) / denom_sym

                if diff_sym <= adaptive_thresh:
                    # Symmetry holds
                    return (local_val + peer_val) / 2.0

                # 2. Symmetry Broken

                if val_hint is not None:
                    # We have a strict flow hint.

                    # Double Dead / Blackhole Check
                    if local_val < TRAFFIC_THRESHOLD and peer_val < TRAFFIC_THRESHOLD:
                        if val_hint > 5.0: return val_hint
                        return 0.0

                    # Synthesized Repair: If both sensors deviate significantly from Hint,
                    # trust the Hint (router conservation) over the sensors.
                    denom_l = max(local_val, val_hint, 1.0)
                    dist_l = abs(local_val - val_hint) / denom_l

                    denom_p = max(peer_val, val_hint, 1.0)
                    dist_p = abs(peer_val - val_hint) / denom_p

                    if dist_l > 0.2 and dist_p > 0.2:
                        # Both are wrong (>20% error). Trust the hint.
                        return val_hint

                    # Standard Selection
                    if dist_l < dist_p:
                        return local_val
                    else:
                        return peer_val

                else:
                    # No Hint (Partial Router)
                    # Heuristic: Trust active signal
                    if local_val < TRAFFIC_THRESHOLD and peer_val > TRAFFIC_THRESHOLD:
                        return peer_val
                    if peer_val < TRAFFIC_THRESHOLD and local_val > TRAFFIC_THRESHOLD:
                        return local_val

                    return (local_val + peer_val) / 2.0

            # Resolve RX (Target: Peer TX)
            peer_tx = state[peer_id]['tx'] if has_peer else curr['rx']
            next_rx = resolve_rate(curr['rx'], peer_tx, is_rx=True)

            # Resolve TX (Target: Peer RX)
            peer_rx = state[peer_id]['rx'] if has_peer else curr['tx']
            next_tx = resolve_rate(curr['tx'], peer_rx, is_rx=False)

            next_rates[iface_id] = {'rx': next_rx, 'tx': next_tx}

        # Synchronous Update
        for iface, rates in next_rates.items():
            state[iface]['rx'] = rates['rx']
            state[iface]['tx'] = rates['tx']

    # --- Phase 3: Final Confidence Calibration ---
    result = {}

    final_balances = {}
    for r_id in verified_routers:
        ifaces = topology[r_id]
        sum_rx = sum(state[i]['rx'] for i in ifaces)
        sum_tx = sum(state[i]['tx'] for i in ifaces)
        final_balances[r_id] = {'rx': sum_rx, 'tx': sum_tx}

    for iface_id, data in state.items():
        final_rx = data['rx']
        final_tx = data['tx']
        peer_id = data['peer_id']
        has_peer = peer_id and peer_id in state

        def get_confidence(val, peer_val, hint_val, status_conf, is_down):
            if is_down:
                return status_conf if val > TRAFFIC_THRESHOLD else 1.0

            # Errors
            err_sym = 0.0
            if has_peer:
                denom = max(val, peer_val, 1.0)
                err_sym = abs(val - peer_val) / denom

            err_flow = None
            if hint_val is not None:
                denom = max(val, hint_val, 1.0)
                err_flow = abs(val - hint_val) / denom

            # Score Logic
            if err_flow is not None:
                # Corroboration: We have two checks.
                primary_err = min(err_sym, err_flow)

                # Steeper penalty curve
                score = 1.0 - (primary_err * 3.0)

                # Conflict Penalty: If one says X and other says Y (and they differ)
                conflict = abs(err_sym - err_flow)
                if conflict > 0.1:
                    score -= (conflict * 0.5)
            else:
                # Only Symmetry
                score = 1.0 - (err_sym * 3.0)
                # Cap confidence if we lack flow verification, unless symmetry is perfect
                if err_sym > HARDENING_THRESHOLD:
                    score = min(score, 0.85)

            return max(0.0, score)

        r_id = data['local_router']
        hint_rx, hint_tx = None, None
        if r_id in final_balances:
            rb = final_balances[r_id]
            hint_rx = max(0.0, rb['tx'] - (rb['rx'] - final_rx))
            hint_tx = max(0.0, rb['rx'] - (rb['tx'] - final_tx))

        peer_tx = state[peer_id]['tx'] if has_peer else final_rx
        peer_rx = state[peer_id]['rx'] if has_peer else final_tx

        conf_rx = get_confidence(final_rx, peer_tx, hint_rx, data['status_conf'], data['status'] == 'down')
        conf_tx = get_confidence(final_tx, peer_rx, hint_tx, data['status_conf'], data['status'] == 'down')

        result[iface_id] = {
            'rx_rate': (data['orig_rx'], final_rx, conf_rx),
            'tx_rate': (data['orig_tx'], final_tx, conf_tx),
            'interface_status': (data['orig_status'], data['status'], data['status_conf']),
            'connected_to': peer_id,
            'local_router': r_id,
            'remote_router': telemetry[iface_id].get('remote_router')
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