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
        # If statuses disagree and no traffic, we assume DOWN is correct (or keep DOWN)
        elif orig_status != peer_status:
            # If we are DOWN and peer is UP (idle), we trust our DOWN status more than their UP
            # because "Activity implies UP" didn't trigger.
            conf = 0.8

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

    # Run 3 passes to allow corrections to propagate across the network
    for _ in range(3):
        # Iterate over all interfaces to check Link Symmetry
        for if_id, s in state.items():
            if s['status'] == 'down': continue

            peer_id = s['connected_to']
            if not peer_id or peer_id not in state:
                continue

            # We process the "outgoing" link: Tx(Local) -> Rx(Peer)
            cand_tx = s['tx']              # Candidate 1: Local Tx
            cand_rx = state[peer_id]['rx'] # Candidate 2: Peer Rx

            # 3a. Check for agreement
            diff = abs(cand_tx - cand_rx)
            mag = max(cand_tx, cand_rx, 1.0)

            if diff < max(mag * TOLERANCE, MIN_ACTIVITY):
                # Agree: Average to smooth out small noise
                best_val = (cand_tx + cand_rx) / 2.0
            else:
                # Disagree: Conflict Resolution using Flow Conservation
                rid_local = s['local_router']
                rid_remote = state[peer_id]['local_router']

                # Helper to score a candidate value based on flow impact
                # Lower score is better
                def get_candidate_score(val):
                    # Check local router (TX side)
                    err_local = calc_flow_error(rid_local, if_id, 'tx', val)
                    # Check remote router (RX side)
                    err_remote = calc_flow_error(rid_remote, peer_id, 'rx', val)

                    score = 0.0

                    # Cost for Local
                    if err_local is None: score += 0.02 # Unverifiable = small cost
                    else: score += min(err_local, 1.0)  # Verifiable = actual error

                    # Cost for Remote
                    if err_remote is None: score += 0.02
                    else: score += min(err_remote, 1.0)

                    return score

                score_tx = get_candidate_score(cand_tx)
                score_rx = get_candidate_score(cand_rx)

                # Heuristic: Dead counters (0) are often wrong if the other side is active
                # Penalize the zero value
                if cand_tx < MIN_ACTIVITY and cand_rx > MIN_ACTIVITY: score_tx += 0.5
                if cand_rx < MIN_ACTIVITY and cand_tx > MIN_ACTIVITY: score_rx += 0.5

                # Weighted combination based on scores (lower score = higher weight)
                # This naturally handles ties and slight preferences better than hard switching
                w_tx = 1.0 / (score_tx + 1e-4)
                w_rx = 1.0 / (score_rx + 1e-4)

                best_val = (cand_tx * w_tx + cand_rx * w_rx) / (w_tx + w_rx)

            # Apply repair to state
            state[if_id]['tx'] = best_val
            state[peer_id]['rx'] = best_val

    # --- 4. Final Result Generation & Confidence Calibration ---
    result = {}

    # Identify Broken Routers (High residual error after all repairs)
    # These routers are verifiable but failed to converge to a consistent state
    broken_routers = set()
    for rid in verifiable_routers:
        err = calc_flow_error(rid, None, None, None) # Calc error with current state
        if err is not None and err > FLOW_TOLERANCE:
            broken_routers.add(rid)

    for if_id, data in telemetry.items():
        final_rx = state[if_id]['rx']
        final_tx = state[if_id]['tx']
        final_st = state[if_id]['status']

        orig_rx = data.get('rx_rate', 0.0)
        orig_tx = data.get('tx_rate', 0.0)
        orig_st = data.get('interface_status', 'unknown')

        rid_local = data.get('local_router')
        rid_remote = data.get('remote_router')

        # -- Calibrate Confidence --

        # Helper: check for peer consistency
        def is_peer_consistent(val, field):
            peer_id = data.get('connected_to')
            if not peer_id or peer_id not in state:
                return True # Can't check, assume consistent

            # Compare with peer's finalized value (which should be symmetric)
            peer_val = state[peer_id]['tx'] if field == 'rx' else state[peer_id]['rx']

            # Use same tolerance as repair loop
            diff = abs(val - peer_val)
            mag = max(val, peer_val, 1.0)
            return diff < max(mag * TOLERANCE, MIN_ACTIVITY)

        # Helper: check verification
        def get_verification_level(val, field):
            # Local Verification
            local_err = calc_flow_error(rid_local, if_id, field, val)
            local_ok = (local_err is not None and local_err < FLOW_TOLERANCE)
            if rid_local in broken_routers: local_ok = False

            # Remote Verification
            remote_ok = False
            peer_id = data.get('connected_to')

            if rid_remote and peer_id:
                check_field = 'tx' if field == 'rx' else 'rx'
                rem_err = calc_flow_error(rid_remote, peer_id, check_field, val)
                if rem_err is not None and rem_err < FLOW_TOLERANCE:
                    remote_ok = True
                if rid_remote in broken_routers: remote_ok = False

            if local_ok and remote_ok: return 3 # Both
            if local_ok: return 2 # Local only
            if remote_ok: return 1 # Remote only
            return 0 # None

        def get_rate_confidence(orig, final, field):
            changed = abs(orig - final) > 0.001
            ver_level = get_verification_level(final, field)
            consistent_with_peer = is_peer_consistent(final, field)

            # 1. High Confidence Scenarios (Verified)
            if ver_level == 3: return 0.99
            if ver_level == 2: return 0.96

            # 2. Unchanged Data
            if not changed:
                # If we didn't change it, but it contradicts the peer, confidence drops
                if not consistent_with_peer:
                    return 0.7
                # If consistent and unchanged
                if ver_level == 1: return 0.95 # Verified remote

                # If unverifiable but stable
                return 0.9

            # 3. Changed Data

            # Smoothing (small change)
            if orig > MIN_ACTIVITY and abs(orig - final) / orig < 0.05:
                return 0.95

            # Significant changes
            if ver_level == 1: return 0.92 # Verified remote

            # Unverified Repairs
            if orig < MIN_ACTIVITY and final > MIN_ACTIVITY:
                # Dead counter repair.
                # If connected to a broken router, be careful
                if rid_remote in broken_routers: return 0.75
                return 0.85

            # Trust Peer (Unverified change to match peer)
            if rid_remote in broken_routers: return 0.6 # Don't trust peer if their router is broken
            return 0.75

        rx_conf = get_rate_confidence(orig_rx, final_rx, 'rx')
        tx_conf = get_rate_confidence(orig_tx, final_tx, 'tx')
        st_conf = status_conf_map.get(if_id, 1.0)

        # Sanity Check: If Down but Traffic, lower confidence
        if final_st == 'down' and (final_rx > 1.0 or final_tx > 1.0):
            rx_conf = 0.0
            tx_conf = 0.0
            st_conf = 0.0

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