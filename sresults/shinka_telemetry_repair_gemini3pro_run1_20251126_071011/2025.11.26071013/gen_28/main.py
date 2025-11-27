# EVOLVE-BLOCK-START
"""
Network telemetry repair algorithm using Iterative Flow Optimization.
Refines telemetry data through multiple passes of constraint satisfaction,
solving for the most likely network state that satisfies Symmetry and Flow Conservation.
"""
from typing import Dict, Any, Tuple, List

def repair_network_telemetry(telemetry: Dict[str, Dict[str, Any]],
                             topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:

    # Constants
    HARDENING_THRESHOLD = 0.02   # 2% tolerance for measurement timing
    TRAFFIC_THRESHOLD = 1.0      # 1 Mbps threshold for "active" link
    ITERATIONS = 3               # Number of relaxation passes

    # --- Phase 1: Initialization & Status Repair ---
    # We maintain a 'belief' state for rates and status
    state = {}

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
            'local_router': data.get('local_router'),
            'remote_router': data.get('remote_router')
        }

    # --- Phase 2: Iterative Constraint Satisfaction ---

    # Pre-computation: Identify fully monitored routers
    # We can only strictly enforce flow conservation on routers where we see all interfaces.
    fully_monitored_routers = set()
    for r_id, ifaces in topology.items():
        if all(i in telemetry for i in ifaces):
            fully_monitored_routers.add(r_id)

    for _ in range(ITERATIONS):
        next_rates = {}

        # 1. Calculate Router Balances (Flow Hints) based on CURRENT beliefs
        # Only for fully monitored routers to avoid garbage hints
        router_balances = {}

        for r_id in fully_monitored_routers:
            ifaces = topology[r_id]
            sum_rx = sum(state[i]['rx'] for i in ifaces if i in state)
            sum_tx = sum(state[i]['tx'] for i in ifaces if i in state)
            router_balances[r_id] = {'rx': sum_rx, 'tx': sum_tx}

        # 2. Evaluate each interface
        for iface_id, curr in state.items():
            if curr['status'] == 'down':
                next_rates[iface_id] = {'rx': 0.0, 'tx': 0.0}
                continue

            peer_id = curr['peer_id']
            has_peer = peer_id and peer_id in state
            local_r_id = curr.get('local_router')
            remote_r_id = curr.get('remote_router')

            # Helper to resolve a rate (RX or TX)
            def resolve_rate(local_val, peer_val, is_rx):

                # --- A. Get Hints ---
                # Local Hint
                local_hint = None
                if local_r_id and local_r_id in router_balances:
                    rb = router_balances[local_r_id]
                    if is_rx:
                        local_hint = max(0.0, rb['tx'] - (rb['rx'] - local_val))
                    else:
                        local_hint = max(0.0, rb['rx'] - (rb['tx'] - local_val))

                # Peer Hint (Check consistency of peer value against its own router)
                # Note: peer_val is the value coming FROM the peer.
                # If is_rx=True (Local RX), peer_val is Peer TX. Peer TX should match Peer RX sums.
                peer_hint = None
                if remote_r_id and remote_r_id in router_balances:
                    rb_p = router_balances[remote_r_id]
                    if is_rx:
                        # Peer Val is Peer TX. Hint is Peer TX needed to balance Peer RX.
                        peer_hint = max(0.0, rb_p['rx'] - (rb_p['tx'] - peer_val))
                    else:
                        # Peer Val is Peer RX. Hint is Peer RX needed to balance Peer TX.
                        peer_hint = max(0.0, rb_p['tx'] - (rb_p['rx'] - peer_val))

                # --- B. Decision Logic ---

                # 1. Symmetry Check
                denom_sym = max(local_val, peer_val, 1.0)
                diff_sym = abs(local_val - peer_val) / denom_sym

                if diff_sym <= HARDENING_THRESHOLD:
                    return (local_val + peer_val) / 2.0

                # 2. Double Dead Check
                if local_val < TRAFFIC_THRESHOLD and peer_val < TRAFFIC_THRESHOLD:
                    if local_hint is not None and local_hint > 5.0:
                        return local_hint
                    if peer_hint is not None and peer_hint > 5.0:
                        return peer_hint
                    return 0.0

                # 3. Bilateral Consistency Check
                # Calculate consistency scores (Lower is better)
                score_local = float('inf')
                if local_hint is not None:
                    denom = max(local_val, local_hint, 1.0)
                    score_local = abs(local_val - local_hint) / denom

                score_peer = float('inf')
                if peer_hint is not None:
                    denom = max(peer_val, peer_hint, 1.0)
                    score_peer = abs(peer_val - peer_hint) / denom

                # Compare scores
                if score_local < 0.1 and score_peer < 0.1:
                    if score_local < score_peer:
                        return local_val
                    else:
                        return peer_val

                if score_local < score_peer:
                    return local_val
                elif score_peer < score_local:
                    return peer_val
                else:
                    return local_val

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

    # Recalculate final router sums for accurate calibration
    final_balances = {}
    for r_id in fully_monitored_routers:
        ifaces = topology[r_id]
        sum_rx = sum(state[i]['rx'] for i in ifaces if i in state)
        sum_tx = sum(state[i]['tx'] for i in ifaces if i in state)
        final_balances[r_id] = {'rx': sum_rx, 'tx': sum_tx}

    for iface_id, data in state.items():
        final_rx = data['rx']
        final_tx = data['tx']
        peer_id = data['peer_id']
        has_peer = peer_id and peer_id in state

        # Calibration Function
        def get_confidence(val, peer_val, hint_val, status_conf, is_down):
            if is_down:
                return status_conf

            # Calculate Residuals (Errors)
            err_sym = 0.0
            if has_peer:
                denom = max(val, peer_val, 1.0)
                err_sym = abs(val - peer_val) / denom

            err_flow = None
            if hint_val is not None:
                denom = max(val, hint_val, 1.0)
                err_flow = abs(val - hint_val) / denom

            # Calibration Logic
            # Determine Best Support
            support_err = err_sym
            if err_flow is not None:
                support_err = min(err_sym, err_flow)

            # Base Score: 1.0 - 2.5 * error
            base_score = max(0.0, 1.0 - (2.5 * support_err))

            # Contradiction Penalty
            if err_flow is not None:
                conflict = max(err_sym, err_flow)
                if conflict > 0.2 and support_err > 0.1:
                     # Both are kinda bad.
                     base_score *= 0.8

            return base_score * status_conf

        # Get Hints for final verification
        r_id = data['local_router']
        hint_rx = None
        hint_tx = None
        if r_id and r_id in final_balances:
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
