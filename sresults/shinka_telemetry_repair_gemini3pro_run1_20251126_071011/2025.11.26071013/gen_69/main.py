# EVOLVE-BLOCK-START
"""
Network telemetry repair algorithm using Topology-Aware Flow Consensus.
Handles partial observability by selectively applying Flow Conservation
only to fully monitored routers. Uses a prioritized evidence hierarchy:
1. Symmetry (Direct agreement)
2. Flow Conservation (Indirect validation, if router fully visible)
3. Traffic Maximization (Heuristic for partial visibility/broken links)
"""
from typing import Dict, Any, Tuple, List

def repair_network_telemetry(telemetry: Dict[str, Dict[str, Any]],
                             topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:

    # Constants
    HARDENING_THRESHOLD = 0.02   # 2% tolerance for measurement timing
    TRAFFIC_THRESHOLD = 1.0      # 1 Mbps threshold for "active" link
    ITERATIONS = 5               # Convergence passes

    # --- Phase 1: Assessment & Initialization ---

    # Identify Fully Monitored Routers
    # Flow Conservation only applies if we see all interfaces.
    verifiable_routers = set()
    for r_id, ifaces in topology.items():
        if all(i in telemetry for i in ifaces):
            verifiable_routers.add(r_id)

    # Initialize State with Status Inference
    state = {}
    for iface_id, data in telemetry.items():
        raw_rx = data.get('rx_rate', 0.0)
        raw_tx = data.get('tx_rate', 0.0)
        raw_status = data.get('interface_status', 'unknown')

        peer_id = data.get('connected_to')
        peer_data = telemetry.get(peer_id, {}) if (peer_id and peer_id in telemetry) else {}

        # Traffic signals
        signals = [raw_rx, raw_tx, peer_data.get('rx_rate', 0.0), peer_data.get('tx_rate', 0.0)]
        max_traffic = max(signals) if signals else 0.0

        # Status Logic
        status = raw_status
        status_conf = 1.0

        if max_traffic > TRAFFIC_THRESHOLD:
            if raw_status != 'up':
                status = 'up'
                status_conf = 0.95
        elif raw_status == 'up' and peer_data.get('interface_status') == 'down':
            status = 'down'
            status_conf = 0.8

        # Rate Initialization
        if status == 'down':
            cur_rx, cur_tx = 0.0, 0.0
        else:
            cur_rx = raw_rx if raw_rx > 0 else 0.0
            cur_tx = raw_tx if raw_tx > 0 else 0.0

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

    # --- Phase 2: Iterative Consensus ---
    for _ in range(ITERATIONS):
        next_state = {}

        # 1. Compute Flow Balances for Verifiable Routers
        router_balances = {}
        for r_id in verifiable_routers:
            ifaces = topology[r_id]
            sum_rx = sum(state[i]['rx'] for i in ifaces)
            sum_tx = sum(state[i]['tx'] for i in ifaces)
            router_balances[r_id] = {'rx': sum_rx, 'tx': sum_tx}

        # 2. Resolve Link Rates
        for iface_id, curr in state.items():
            if curr['status'] == 'down':
                next_state[iface_id] = {'rx': 0.0, 'tx': 0.0}
                continue

            peer_id = curr['peer_id']
            has_peer = peer_id and peer_id in state

            def resolve_direction(local_val, peer_val, is_rx):
                # A. Generate Hints (Triangulation)
                hints = []

                # Local Hint
                r_id = curr['local_router']
                if r_id in router_balances:
                    rb = router_balances[r_id]
                    if is_rx: h = rb['tx'] - (rb['rx'] - local_val)
                    else:     h = rb['rx'] - (rb['tx'] - local_val)
                    hints.append(max(0.0, h))

                # Remote Hint
                rr_id = curr['remote_router']
                if rr_id in router_balances:
                    rb_r = router_balances[rr_id]
                    # Link RX is Remote TX. Hint = Remote Total In - (Remote Total Out - Link RX)
                    # Link TX is Remote RX. Hint = Remote Total Out - (Remote Total In - Link TX)
                    if is_rx: h = rb_r['rx'] - (rb_r['tx'] - peer_val)
                    else:     h = rb_r['tx'] - (rb_r['rx'] - peer_val)
                    hints.append(max(0.0, h))

                # Consolidated Hint Target
                target = None
                if len(hints) == 2:
                    # Check consistency
                    diff = abs(hints[0] - hints[1])
                    avg_h = (hints[0] + hints[1]) / 2.0
                    if avg_h > 5.0:
                        rel_diff = diff / avg_h
                        if rel_diff < 0.15: # 15% tolerance for hint agreement
                             target = avg_h
                        else:
                             # Hints disagree significantly.
                             # Fallback to local hint if available as it's topologically closer.
                             target = hints[0]
                    else:
                        target = avg_h
                elif len(hints) == 1:
                    target = hints[0]

                # B. Decision Logic based on Evidence

                # 1. Symmetry Check
                denom = max(local_val, peer_val, 1.0)
                diff_sym = abs(local_val - peer_val) / denom

                if diff_sym <= HARDENING_THRESHOLD:
                    # Symmetric - Trust Average
                    avg = (local_val + peer_val) / 2.0

                    # Double Dead Check: If sensors 0, but hint says > 5, trust hint.
                    if target is not None and avg < TRAFFIC_THRESHOLD and target > 5.0:
                        return target
                    return avg

                # 2. Asymmetric (Conflict)
                if target is not None:
                    # We have a guiding hint

                    # Double Dead Check
                    if local_val < TRAFFIC_THRESHOLD and peer_val < TRAFFIC_THRESHOLD:
                         if target > 5.0: return target
                         return 0.0

                    # Select candidate closest to target
                    dist_l = abs(local_val - target)
                    dist_p = abs(peer_val - target)

                    if dist_l < dist_p:
                        return local_val
                    else:
                        return peer_val

                else:
                    # No Hint, Asymmetric
                    # Heuristic: Trust Non-Zero (Failure to count is more common)
                    if local_val > TRAFFIC_THRESHOLD and peer_val < TRAFFIC_THRESHOLD:
                        return local_val
                    if peer_val > TRAFFIC_THRESHOLD and local_val < TRAFFIC_THRESHOLD:
                        return peer_val

                    # Both non-zero, disagree, no hint -> Average
                    return (local_val + peer_val) / 2.0

            # Resolve RX
            peer_tx = state[peer_id]['tx'] if has_peer else curr['rx']
            next_rx = resolve_direction(curr['rx'], peer_tx, is_rx=True)

            # Resolve TX
            peer_rx = state[peer_id]['rx'] if has_peer else curr['tx']
            next_tx = resolve_direction(curr['tx'], peer_rx, is_rx=False)

            next_state[iface_id] = {'rx': next_rx, 'tx': next_tx}

        # Apply Updates
        for i_id, vals in next_state.items():
            state[i_id]['rx'] = vals['rx']
            state[i_id]['tx'] = vals['tx']

    # --- Phase 3: Calibration ---
    result = {}

    # Final Balances for Calibration
    final_balances = {}
    for r_id in verifiable_routers:
        ifaces = topology[r_id]
        sum_rx = sum(state[i]['rx'] for i in ifaces)
        sum_tx = sum(state[i]['tx'] for i in ifaces)
        final_balances[r_id] = {'rx': sum_rx, 'tx': sum_tx}

    for iface_id, data in state.items():
        final_rx = data['rx']
        final_tx = data['tx']
        peer_id = data['peer_id']
        has_peer = peer_id and peer_id in state

        peer_tx = state[peer_id]['tx'] if has_peer else final_rx
        peer_rx = state[peer_id]['rx'] if has_peer else final_tx

        def calibrate(val, peer_val, local_r, remote_r, is_rx, status_down, stat_conf):
            if status_down:
                return stat_conf

            # 1. Symmetry Error
            err_sym = 0.0
            if has_peer:
                denom = max(val, peer_val, 1.0)
                err_sym = abs(val - peer_val) / denom
            else:
                err_sym = 1.0

            # 2. Flow Error (Find Best Matching Hint)
            hints = []
            if local_r in final_balances:
                rb = final_balances[local_r]
                if is_rx: h = rb['tx'] - (rb['rx'] - val)
                else:     h = rb['rx'] - (rb['tx'] - val)
                hints.append(max(0.0, h))
            if remote_r in final_balances:
                rb = final_balances[remote_r]
                if is_rx: h = rb['rx'] - (rb['tx'] - peer_val)
                else:     h = rb['tx'] - (rb['rx'] - peer_val)
                hints.append(max(0.0, h))

            err_flow = None
            if hints:
                # Find best matching hint to the final decision
                best_hint_err = 1.0
                for h in hints:
                    denom = max(val, h, 1.0)
                    e = abs(val - h) / denom
                    if e < best_hint_err:
                        best_hint_err = e
                err_flow = best_hint_err

            # Confidence Assignment

            # A. Strong Corroboration (Flow & Symmetry Agree)
            if err_flow is not None and err_flow < 0.05 and err_sym < 0.05:
                return 1.0

            # B. Flow Rescue (Symmetry Broken, Flow Verified)
            if err_flow is not None and err_flow < 0.05:
                # High confidence, we used conservation to pick the winner
                return 0.90

            # C. Symmetry Only (No Flow Hint, or Flow Disagrees)
            if err_sym < 0.05:
                if err_flow is None:
                    # Verified by peer only (Partial Observability)
                    return 0.95
                else:
                    # Symmetry holds, but flow disagrees?
                    # Likely a third interface issue, but our link is consistent.
                    return 0.80

            # D. Heuristic (No Flow, Broken Sym)
            if err_flow is None:
                # We trusted non-zero. Usually correct.
                return 0.60

            # E. Conflict / Unknown
            return 0.30

        conf_rx = calibrate(final_rx, peer_tx, data['local_router'], data['remote_router'], True, data['status']=='down', data['status_conf'])
        conf_tx = calibrate(final_tx, peer_rx, data['local_router'], data['remote_router'], False, data['status']=='down', data['status_conf'])

        result[iface_id] = {
            'rx_rate': (data['orig_rx'], final_rx, conf_rx),
            'tx_rate': (data['orig_tx'], final_tx, conf_tx),
            'interface_status': (data['orig_status'], data['status'], data['status_conf']),
            'connected_to': peer_id,
            'local_router': data['local_router'],
            'remote_router': data['remote_router']
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