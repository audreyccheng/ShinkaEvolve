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

        # If we are confident the link is DOWN, force rates to zero
        # This prevents "ghost traffic" from confusing the rate consensus
        if new_status == 'down':
            state[if_id]['rx'] = 0.0
            state[if_id]['tx'] = 0.0

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

    # Helper: Calculate the value required to perfectly balance a router
    def get_residual(rid, if_target, field):
        if rid not in verifiable_routers:
            return None

        sum_in = 0.0
        sum_out = 0.0

        for iface in router_interfaces[rid]:
            if iface == if_target: continue # Skip the target
            sum_in += state[iface]['rx']
            sum_out += state[iface]['tx']

        # Target must balance the difference
        # If calculating Tx: Out_Target = Sum_In - Sum_Out_Others
        # If calculating Rx: In_Target = Sum_Out - Sum_In_Others
        if field == 'tx':
            val = sum_in - sum_out
        else:
            val = sum_out - sum_in

        return max(0.0, val)

    # Run 3 passes to allow corrections to propagate across the network
    for _ in range(3):
        for if_id, s in state.items():
            if s['status'] == 'down': continue

            peer_id = s['connected_to']
            if not peer_id or peer_id not in state:
                continue

            # Consensus: Tx(Local) -> Rx(Peer)
            cand_tx = s['tx']
            cand_rx = state[peer_id]['rx']

            rid_local = s['local_router']
            rid_remote = state[peer_id]['local_router']

            # Collect candidates
            candidates = {cand_tx, cand_rx}

            # Add Residuals (Synthesized values from Flow Conservation)
            res_local = get_residual(rid_local, if_id, 'tx')
            if res_local is not None: candidates.add(res_local)

            res_remote = get_residual(rid_remote, peer_id, 'rx')
            if res_remote is not None: candidates.add(res_remote)

            # Evaluate Candidates
            best_score = float('inf')
            best_val = cand_tx

            # Pre-check: If all candidates are very close, just average them
            cands_list = list(candidates)
            if max(cands_list) - min(cands_list) < max(max(cands_list) * TOLERANCE, MIN_ACTIVITY):
                best_val = sum(cands_list) / len(cands_list)
            else:
                for cand in candidates:
                    # Calculate System Error for this candidate
                    err_local = calc_flow_error(rid_local, if_id, 'tx', cand)
                    err_remote = calc_flow_error(rid_remote, peer_id, 'rx', cand)

                    def get_score(err):
                        if err is None: return 0.05
                        return min(err, 1.0)

                    score = get_score(err_local) + get_score(err_remote)

                    # Heuristic: Penalize zero if alternatives exist (Dead counter heuristic)
                    if cand < MIN_ACTIVITY and max(cands_list) > MIN_ACTIVITY:
                        score += 0.5

                    if score < best_score:
                        best_score = score
                        best_val = cand

            state[if_id]['tx'] = best_val
            state[peer_id]['rx'] = best_val

    # --- 4. Final Result Generation & Confidence Calibration ---
    result = {}

    # Map final flow errors
    final_errors = {}
    for rid in verifiable_routers:
        # Re-calc based on final state
        sum_rx = sum(state[iface]['rx'] for iface in router_interfaces[rid])
        sum_tx = sum(state[iface]['tx'] for iface in router_interfaces[rid])
        denom = max(sum_rx, sum_tx, 1.0)
        final_errors[rid] = abs(sum_rx - sum_tx) / denom

    for if_id, data in telemetry.items():
        # Get Final Values
        final_rx = state[if_id]['rx']
        final_tx = state[if_id]['tx']
        final_st = state[if_id]['status']

        orig_rx = data.get('rx_rate', 0.0)
        orig_tx = data.get('tx_rate', 0.0)
        orig_st = data.get('interface_status', 'unknown')

        rid = data.get('local_router')
        peer_id = data.get('connected_to')

        def get_rate_confidence(orig, final, field):
            # 1. Verification Logic
            # Local Verification (strongest)
            local_err = final_errors.get(rid)
            local_verified = (local_err is not None and local_err < FLOW_TOLERANCE)

            # Remote Verification
            remote_verified = False
            rem_rid = data.get('remote_router')
            if rem_rid and rem_rid in final_errors:
                 if final_errors[rem_rid] < FLOW_TOLERANCE:
                     remote_verified = True

            # Peer Consistency (Symmetry)
            peer_consistent = True
            if peer_id and peer_id in state:
                peer_val = state[peer_id]['tx'] if field == 'rx' else state[peer_id]['rx']
                if abs(final - peer_val) > max(final, peer_val, 1.0) * TOLERANCE:
                    peer_consistent = False

            # 2. Change Logic
            changed = abs(orig - final) > 0.001
            # Smoothing (small change)
            smoothed = changed and (abs(orig - final) < max(orig * 0.05, 0.1))

            # --- Scoring Buckets ---

            if not changed:
                # We kept the original value
                if local_verified and remote_verified: return 1.0

                # Check for "Broken Router" (Verifiable but High Error)
                # If we are verifying locally, but the error is high, we shouldn't trust it.
                if local_err is not None and local_err > FLOW_TOLERANCE:
                     return 0.6

                if local_verified: return 0.98
                if not peer_consistent: return 0.7 # Conflict existed, but we didn't change (ambiguous)
                # If unverifiable but consistent with peer
                if remote_verified: return 0.95
                return 0.9

            if smoothed:
                return 0.95 # High confidence in small adjustments

            # Significant Repairs
            if local_verified and remote_verified:
                return 0.98
            if local_verified:
                return 0.95
            if remote_verified:
                return 0.90

            # Heuristic Repairs (Unverified)
            # 0 -> Value
            if orig < MIN_ACTIVITY and final > MIN_ACTIVITY:
                return 0.85

            # Fallback for unverified repairs
            return 0.6

        rx_conf = get_rate_confidence(orig_rx, final_rx, 'rx')
        tx_conf = get_rate_confidence(orig_tx, final_tx, 'tx')
        st_conf = status_conf_map.get(if_id, 1.0)

        # Sanity Check: If Down but Traffic, lower confidence
        if final_st == 'down' and (final_rx > 1.0 or final_tx > 1.0):
            rx_conf = 0.0
            tx_conf = 0.0

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