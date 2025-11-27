# EVOLVE-BLOCK-START
"""
Consensus-based Network Telemetry Repair Algorithm
Uses iterative constraint satisfaction with flow conservation and link symmetry
to detect and repair corrupted network counters and status flags.
"""
from typing import Dict, Any, Tuple, List
import collections

def repair_network_telemetry(telemetry: Dict[str, Dict[str, Any]],
                             topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:
    """
    Repairs network telemetry using an iterative consensus approach.

    The algorithm treats telemetry as a system of constraints:
    1. Link Symmetry: Tx(A->B) should equal Rx(B<-A)
    2. Flow Conservation: Sum(Rx) should equal Sum(Tx) at every router
    3. Status Consistency: Active traffic implies UP status

    It iteratively optimizes these values to find the most consistent network state.
    """

    # --- Constants ---
    TOLERANCE = 0.02         # 2% deviation allowed for symmetry
    MIN_ACTIVITY = 0.01      # Mbps threshold to consider an interface active
    FLOW_TOLERANCE = 0.05    # 5% flow imbalance allowed

    # --- 1. Initialization & Data Structure Setup ---

    # Working state dictionary: We will mutate this iteratively
    # Structure: if_id -> {rx, tx, status, ...}
    state = {}

    # Map for router verification
    # router_id -> list of interface_ids
    router_interfaces = collections.defaultdict(list)
    verifiable_routers = set()

    # Build topology map and identify routers where we can check Flow Conservation
    # (We can only check flow if we have telemetry for ALL interfaces on the router)
    for rid, if_list in topology.items():
        router_interfaces[rid] = if_list
        if all(if_id in telemetry for if_id in if_list):
            verifiable_routers.add(rid)

    # Initialize state from input telemetry
    for if_id, data in telemetry.items():
        state[if_id] = {
            'rx': float(data.get('rx_rate', 0.0)),
            'tx': float(data.get('tx_rate', 0.0)),
            'status': data.get('interface_status', 'down'),
            'connected_to': data.get('connected_to'),
            'local_router': data.get('local_router')
        }

    # --- 2. Status Repair ---
    # Logic: Activity implies UP. Peer UP + Activity implies UP.
    status_conf_map = {}

    for if_id, s in state.items():
        orig_status = s['status']
        peer_id = s['connected_to']

        # Local activity check
        local_active = (s['rx'] > MIN_ACTIVITY) or (s['tx'] > MIN_ACTIVITY)

        # Peer activity/status check
        peer_active = False
        peer_status = 'unknown'
        if peer_id and peer_id in state:
            p = state[peer_id]
            peer_active = (p['rx'] > MIN_ACTIVITY) or (p['tx'] > MIN_ACTIVITY)
            peer_status = p['status']

        # Decision Logic
        new_status = orig_status
        conf = 1.0

        # If there is traffic, the link must be UP
        if local_active or peer_active:
            new_status = 'up'
            if orig_status == 'down':
                conf = 0.95  # High confidence repair
        # If I say UP, but Peer says DOWN and there is NO traffic -> likely DOWN
        elif orig_status == 'up' and peer_status == 'down':
            new_status = 'down'
            conf = 0.8
        # If statuses disagree and no traffic, we are uncertain
        elif orig_status != peer_status:
            # We stick to original but with low confidence
            conf = 0.5

        state[if_id]['status'] = new_status
        status_conf_map[if_id] = conf

    # --- 3. Rate Repair (Iterative Consensus) ---

    # Helper: Calculate relative flow error for a router if we force a specific value
    def calc_flow_error(rid, if_target, field, value):
        if rid not in verifiable_routers:
            return None

        sum_rx = 0.0
        sum_tx = 0.0

        for iface in router_interfaces[rid]:
            # Use current state values
            r = state[iface]['rx']
            t = state[iface]['tx']

            # Substitute the target value we are testing
            if iface == if_target:
                if field == 'rx': r = value
                else: t = value

            sum_rx += r
            sum_tx += t

        err = abs(sum_rx - sum_tx)
        denom = max(sum_rx, sum_tx, 1.0)
        return err / denom

    # Helper: Synthesize the value required to balance the router flow
    def get_flow_suggestion(rid, if_target, field):
        if rid not in verifiable_routers:
            return None

        sum_in = 0.0
        sum_out = 0.0

        for iface in router_interfaces[rid]:
            if iface == if_target: continue
            sum_in += state[iface]['rx']
            sum_out += state[iface]['tx']

        # Flow Conservation: Sum(In) = Sum(Out)
        if field == 'tx':
            # Target_Tx = Sum(In) - Sum(Other_Out)
            val = sum_in - sum_out
        else:
            # Target_Rx = Sum(Out) - Sum(Other_In)
            val = sum_out - sum_in

        return max(0.0, val)

    # Run 3 passes to allow corrections to propagate across the network
    for _ in range(3):
        for if_id, s in state.items():
            if s['status'] == 'down': continue

            peer_id = s['connected_to']
            if not peer_id or peer_id not in state: continue

            # Link: Local(Tx) -> Remote(Rx)
            cand_tx = s['tx']
            cand_rx = state[peer_id]['rx']

            # Check Agreement
            diff = abs(cand_tx - cand_rx)
            avg = (cand_tx + cand_rx) / 2.0

            if diff < max(avg * TOLERANCE, MIN_ACTIVITY):
                best_val = avg
            else:
                # Disagreement. Generate candidates.
                candidates = {cand_tx, cand_rx, avg}

                # Synthesize values from flow conservation
                rid_local = s['local_router']
                synth_tx = get_flow_suggestion(rid_local, if_id, 'tx')
                if synth_tx is not None: candidates.add(synth_tx)

                rid_remote = state[peer_id]['local_router']
                synth_rx = get_flow_suggestion(rid_remote, peer_id, 'rx')
                if synth_rx is not None: candidates.add(synth_rx)

                # Filter candidates
                candidates = [c for c in candidates if c >= 0.0]

                # Score candidates
                best_score = float('inf')
                best_val = avg

                for val in candidates:
                    # Score based on flow error magnitude
                    err_local = calc_flow_error(rid_local, if_id, 'tx', val)
                    err_remote = calc_flow_error(rid_remote, peer_id, 'rx', val)

                    score = 0.0

                    # Cost Function: Unverified = small constant cost, Verified = actual error cost
                    # This prefers verified solutions (if error is small) over unverified guesses.
                    if err_local is None: score += 0.05
                    else: score += min(err_local, 1.0) * 2.0

                    if err_remote is None: score += 0.05
                    else: score += min(err_remote, 1.0) * 2.0

                    # Heuristic: Avoid Zero if alternatives exist
                    if val < MIN_ACTIVITY and any(c > MIN_ACTIVITY for c in candidates):
                        score += 0.5

                    # Preference for Average (Smoothing) if errors are similar
                    if abs(val - avg) < 0.001:
                        score -= 0.01

                    if score < best_score:
                        best_score = score
                        best_val = val

            state[if_id]['tx'] = best_val
            state[peer_id]['rx'] = best_val

    # --- 4. Final Result Generation & Confidence Calibration ---
    result = {}

    for if_id, data in telemetry.items():
        # Get Final Values
        final_rx = state[if_id]['rx']
        final_tx = state[if_id]['tx']
        final_st = state[if_id]['status']

        orig_rx = data.get('rx_rate', 0.0)
        orig_tx = data.get('tx_rate', 0.0)
        orig_st = data.get('interface_status', 'unknown')

        # -- Calibrate Confidence --
        def get_rate_confidence(orig, final, field):
            changed = abs(orig - final) > 0.001

            # Calculate final flow error for verification
            rid = data.get('local_router')
            flow_err = calc_flow_error(rid, if_id, field, final)
            is_verified = flow_err is not None and flow_err < FLOW_TOLERANCE

            # Remote verification check
            remote_verified = False
            peer_id = data.get('connected_to')
            rem_rid = data.get('remote_router')
            if rem_rid and peer_id:
                check_field = 'tx' if field == 'rx' else 'rx'
                rem_err = calc_flow_error(rem_rid, peer_id, check_field, final)
                if rem_err is not None and rem_err < FLOW_TOLERANCE:
                    remote_verified = True

            # 1. High Confidence: Verified by Flow
            if is_verified:
                # Continuous confidence based on residual error
                # 0% error -> 0.99 confidence
                # 5% error -> 0.90 confidence
                base = 0.99
                penalty = (flow_err / FLOW_TOLERANCE) * 0.09
                return base - penalty

            if remote_verified:
                return 0.90

            # 2. Unverified Scenarios
            if not changed:
                return 0.9 # Default trust in data

            # Smoothing (changed but < 5%)
            if orig > MIN_ACTIVITY and abs(orig - final) / orig < 0.05:
                return 0.95

            # Dead Counter Repair
            if orig < MIN_ACTIVITY and final > MIN_ACTIVITY:
                return 0.8

            # Forced Agreement / Guess
            return 0.6

        rx_conf = get_rate_confidence(orig_rx, final_rx, 'rx')
        tx_conf = get_rate_confidence(orig_tx, final_tx, 'tx')
        st_conf = status_conf_map.get(if_id, 1.0)

        # Sanity Check: If Down but Traffic, lower confidence
        if final_st == 'down' and (final_rx > 1.0 or final_tx > 1.0):
            rx_conf *= 0.5
            tx_conf *= 0.5
            st_conf *= 0.5

        # Construct Output Tuple
        res_entry = {
            'rx_rate': (orig_rx, final_rx, rx_conf),
            'tx_rate': (orig_tx, final_tx, tx_conf),
            'interface_status': (orig_st, final_st, st_conf),
            'connected_to': data.get('connected_to'),
            'local_router': data.get('local_router'),
            'remote_router': data.get('remote_router')
        }
        result[if_id] = res_entry

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
