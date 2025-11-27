# EVOLVE-BLOCK-START
"""
Network telemetry repair algorithm using Adaptive Consensus and Flow Conservation.
1. Identifies Verifiable Routers.
2. Resolves rates using a priority logic: Symmetry > Flow Consensus > Heuristic.
3. Implements 'Double Dead' detection where valid flow implies traffic despite zero readings.
4. Calibrates confidence using a multi-factor model (Symmetry, Flow, SNR).
"""
from typing import Dict, Any, Tuple, List
import math

def repair_network_telemetry(telemetry: Dict[str, Dict[str, Any]],
                             topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:

    # Constants
    HARDENING_THRESHOLD = 0.02   # 2% tolerance
    TRAFFIC_THRESHOLD = 1.0      # 1 Mbps active threshold
    ITERATIONS = 5               # Refinement passes

    # --- Phase 1: Initialization & Status Repair ---
    state = {}

    # Helper to get peer data safely
    def get_peer_data(curr_data):
        pid = curr_data.get('connected_to')
        if pid and pid in telemetry:
            return telemetry[pid]
        return {}

    for iface_id, data in telemetry.items():
        raw_rx = data.get('rx_rate', 0.0)
        raw_tx = data.get('tx_rate', 0.0)
        raw_status = data.get('interface_status', 'unknown')
        peer_data = get_peer_data(data)

        # Traffic Evidence: Look for any sign of life
        signals = [raw_rx, raw_tx, peer_data.get('rx_rate', 0.0), peer_data.get('tx_rate', 0.0)]
        max_traffic = max(signals) if signals else 0.0

        # Status Logic
        status = raw_status
        status_conf = 1.0

        # If any significant traffic is detected, link must be UP
        if max_traffic > TRAFFIC_THRESHOLD:
            if raw_status != 'up':
                status = 'up'
                status_conf = 0.95
        # If I say UP but Peer says DOWN, and no traffic -> Likely DOWN
        elif raw_status == 'up' and peer_data.get('interface_status') == 'down':
            status = 'down'
            status_conf = 0.8

        cur_rx = raw_rx if raw_rx > 0 else 0.0
        cur_tx = raw_tx if raw_tx > 0 else 0.0

        # If status determined DOWN, zero out rates initially
        if status == 'down':
            cur_rx, cur_tx = 0.0, 0.0

        state[iface_id] = {
            'rx': cur_rx,
            'tx': cur_tx,
            'status': status,
            'status_conf': status_conf,
            'orig_rx': raw_rx,
            'orig_tx': raw_tx,
            'orig_status': raw_status,
            'peer_id': data.get('connected_to'),
            'local_router': data.get('local_router'),
            'remote_router': data.get('remote_router')
        }

    # --- Phase 2: Iterative Consensus ---

    # Identify Verifiable Routers (All interfaces monitored)
    verifiable_routers = set()
    for r_id, ifaces in topology.items():
        if all(i in state for i in ifaces):
            verifiable_routers.add(r_id)

    for _ in range(ITERATIONS):
        next_state = {}

        # 1. Calculate Flow Balances for Verifiable Routers
        router_balances = {}
        for r_id in verifiable_routers:
            ifaces = topology[r_id]
            s_rx = sum(state[i]['rx'] for i in ifaces)
            s_tx = sum(state[i]['tx'] for i in ifaces)
            router_balances[r_id] = {'rx': s_rx, 'tx': s_tx}

        # 2. Resolve Links
        for iface_id, curr in state.items():
            if curr['status'] == 'down':
                next_state[iface_id] = {'rx': 0.0, 'tx': 0.0}
                continue

            peer_id = curr['peer_id']
            has_peer = peer_id and peer_id in state

            # Helper to solve for one direction (e.g., My RX)
            def resolve_rate(local_val, peer_val, is_rx):
                # Gather Hints
                hints = []

                # Local Hint
                r_id = curr.get('local_router')
                if r_id in router_balances:
                    rb = router_balances[r_id]
                    if is_rx: h = rb['tx'] - (rb['rx'] - local_val)
                    else:     h = rb['rx'] - (rb['tx'] - local_val)
                    hints.append(max(0.0, h))

                # Remote Hint
                rr_id = curr.get('remote_router')
                if rr_id in router_balances:
                    rb = router_balances[rr_id]
                    if is_rx: h = rb['rx'] - (rb['tx'] - peer_val)
                    else:     h = rb['tx'] - (rb['rx'] - peer_val)
                    hints.append(max(0.0, h))

                # Consensus Hint
                target = None
                if hints:
                    if len(hints) > 1 and abs(hints[0] - hints[1]) > (0.2 * max(hints) + 1.0):
                        # Hints disagree. Use the one closer to the max measured value.
                        # Logic: Failures usually drop values. The higher measured value is likely more correct.
                        ref = max(local_val, peer_val)
                        target = min(hints, key=lambda x: abs(x - ref))
                    else:
                        target = sum(hints) / len(hints)

                # Decision Priority

                # 1. Symmetry (Agreement between sensors)
                denom = max(local_val, peer_val, 1.0)
                if abs(local_val - peer_val) <= HARDENING_THRESHOLD * denom:
                    avg = (local_val + peer_val) / 2.0
                    # Double Dead Check: Sensors agree on 0, but Physics says Traffic?
                    if target is not None and avg < TRAFFIC_THRESHOLD and target > 5.0:
                        return target
                    return avg

                # 2. Flow Conservation (External Truth)
                if target is not None:
                    # Double Dead Check
                    if local_val < TRAFFIC_THRESHOLD and peer_val < TRAFFIC_THRESHOLD:
                        if target > 5.0: return target
                        return 0.0

                    # Trust Target if sensors disagree
                    return target

                # 3. Heuristic (Fallback)
                # Trust Non-Zero / Max
                if local_val > TRAFFIC_THRESHOLD and peer_val < TRAFFIC_THRESHOLD: return local_val
                if peer_val > TRAFFIC_THRESHOLD and local_val < TRAFFIC_THRESHOLD: return peer_val
                return (local_val + peer_val) / 2.0

            # Resolve RX (Target Peer TX)
            peer_tx = state[peer_id]['tx'] if has_peer else curr['rx']
            next_rx = resolve_rate(curr['rx'], peer_tx, True)

            # Resolve TX (Target Peer RX)
            peer_rx = state[peer_id]['rx'] if has_peer else curr['tx']
            next_tx = resolve_rate(curr['tx'], peer_rx, False)

            next_state[iface_id] = {'rx': next_rx, 'tx': next_tx}

        # Update State
        for i_id, vals in next_state.items():
            state[i_id]['rx'] = vals['rx']
            state[i_id]['tx'] = vals['tx']

    # --- Phase 3: Final Calibration ---
    result = {}

    # Recompute balances for final checking
    final_balances = {}
    for r_id in verifiable_routers:
        ifaces = topology[r_id]
        s_rx = sum(state[i]['rx'] for i in ifaces)
        s_tx = sum(state[i]['tx'] for i in ifaces)
        final_balances[r_id] = {'rx': s_rx, 'tx': s_tx}

    for iface_id, data in state.items():
        final_rx = data['rx']
        final_tx = data['tx']
        peer_id = data['peer_id']
        has_peer = peer_id and peer_id in state

        peer_tx = state[peer_id]['tx'] if has_peer else final_rx
        peer_rx = state[peer_id]['rx'] if has_peer else final_tx

        def calibrate(val, peer_val, local_r, remote_r, is_rx, is_down, stat_conf):
            if is_down: return stat_conf

            # 1. Symmetry Error
            denom_s = max(val, peer_val, 1.0)
            err_sym = abs(val - peer_val) / denom_s

            # 2. Flow Error
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
                # Use best matching hint for calibration (optimistic verification)
                best_hint = min(hints, key=lambda x: abs(x - val))
                denom_f = max(val, best_hint, 1.0)
                err_flow = abs(val - best_hint) / denom_f

            # 3. SNR Scaling (Penalty for low signal noise)
            snr_score = 1.0
            if val < 5.0: snr_score = 0.8
            elif val < 20.0: snr_score = 0.95

            # Confidence Tiers

            # A. Verified (Symmetry + Flow)
            if err_flow is not None and err_flow < 0.05 and err_sym < 0.05:
                return 1.0 * snr_score

            # B. Flow Rescue (Symmetry Broken, Flow Verified)
            if err_flow is not None and err_flow < 0.05:
                # We trusted flow.
                return 0.90 * snr_score

            # C. Symmetry Verified (Flow Unknown/Disagree)
            if err_sym < 0.05:
                if err_flow is None: return 0.95 * snr_score
                return 0.75 * snr_score # Flow disagrees

            # D. Heuristic (Trust Non-Zero)
            if err_flow is None:
                return 0.60 * snr_score

            # E. Conflict
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