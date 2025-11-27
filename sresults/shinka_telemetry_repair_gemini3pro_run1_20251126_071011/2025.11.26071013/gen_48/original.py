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

    # Run 2 passes to allow corrections to propagate across the network
    for _ in range(2):
        # Iterate over all interfaces to check Link Symmetry
        for if_id, s in state.items():
            peer_id = s['connected_to']
            if not peer_id or peer_id not in state:
                continue

            # We process the "outgoing" link: Tx(Local) -> Rx(Peer)
            # (The incoming link will be processed when we iterate the peer)

            cand_tx = s['tx']              # Candidate 1: Local Tx
            cand_rx = state[peer_id]['rx'] # Candidate 2: Peer Rx

            # 3a. Check for agreement
            diff = abs(cand_tx - cand_rx)
            mag = max(cand_tx, cand_rx, 1.0)

            best_val = cand_tx # Default to local

            if diff / mag < TOLERANCE:
                # Agree: Average to smooth out small noise
                best_val = (cand_tx + cand_rx) / 2.0
            else:
                # Disagree: Conflict Resolution using Flow Conservation
                rid_a = s['local_router']
                rid_b = state[peer_id]['local_router']

                # Test Candidate 1 (cand_tx)
                # Does it fit Router A's flow (as Tx)? Does it fit Router B's flow (as Rx)?
                err_a_1 = calc_flow_error(rid_a, if_id, 'tx', cand_tx)
                err_b_1 = calc_flow_error(rid_b, peer_id, 'rx', cand_tx)

                # Test Candidate 2 (cand_rx)
                err_a_2 = calc_flow_error(rid_a, if_id, 'tx', cand_rx)
                err_b_2 = calc_flow_error(rid_b, peer_id, 'rx', cand_rx)

                # Vote counting
                votes_1 = 0
                votes_2 = 0

                # A candidate gets a vote if it results in low flow error (<5%)
                if err_a_1 is not None and err_a_1 < FLOW_TOLERANCE: votes_1 += 1
                if err_b_1 is not None and err_b_1 < FLOW_TOLERANCE: votes_1 += 1

                if err_a_2 is not None and err_a_2 < FLOW_TOLERANCE: votes_2 += 1
                if err_b_2 is not None and err_b_2 < FLOW_TOLERANCE: votes_2 += 1

                # Decision
                if votes_1 > votes_2:
                    best_val = cand_tx
                elif votes_2 > votes_1:
                    best_val = cand_rx
                else:
                    # Tie or No Info (e.g. edge routers). Use Heuristics.
                    # Heuristic: Dead counters often report 0. Trust non-zero.
                    if cand_tx > MIN_ACTIVITY and cand_rx <= MIN_ACTIVITY:
                        best_val = cand_tx
                    elif cand_rx > MIN_ACTIVITY and cand_tx <= MIN_ACTIVITY:
                        best_val = cand_rx
                    else:
                        # Compare raw error magnitudes if flow info exists but was ambiguous
                        sum_err_1 = (err_a_1 or 100) + (err_b_1 or 100)
                        sum_err_2 = (err_a_2 or 100) + (err_b_2 or 100)

                        if sum_err_1 < sum_err_2 and sum_err_1 < 200:
                            best_val = cand_tx
                        elif sum_err_2 < sum_err_1 and sum_err_2 < 200:
                            best_val = cand_rx
                        else:
                            # Total ambiguity -> Average
                            best_val = (cand_tx + cand_rx) / 2.0

            # Apply repair to state
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

        # Helper to check remote verification
        def is_remotely_verified(val, field):
            rem_rid = data.get('remote_router')
            peer_id = data.get('connected_to')
            if not rem_rid or not peer_id:
                return False
            # If checking local RX, we check remote TX flow
            # If checking local TX, we check remote RX flow
            check_field = 'tx' if field == 'rx' else 'rx'

            err = calc_flow_error(rem_rid, peer_id, check_field, val)
            return err is not None and err < FLOW_TOLERANCE

        def get_rate_confidence(orig, final, field):
            # 1. Did we change the value?
            if abs(orig - final) < 0.001:
                return 1.0 # High confidence in original/unchanged data

            # 2. Is it just smoothing? (Small relative change < 5%)
            if orig > MIN_ACTIVITY and abs(orig - final) / orig < 0.05:
                return 0.95 # Smoothing is safe

            # 3. Noise Consensus (Both < Min Activity)
            if orig < MIN_ACTIVITY and final < MIN_ACTIVITY:
                return 0.95

            # 4. If changed, was it verified by LOCAL flow?
            rid = data.get('local_router')
            flow_err = calc_flow_error(rid, if_id, field, final)

            if flow_err is not None and flow_err < FLOW_TOLERANCE:
                return 0.95 # Very high confidence: Mathematics supports this

            # 5. Was it verified by REMOTE flow?
            if is_remotely_verified(final, field):
                return 0.90 # High confidence: Peer is verified

            # 6. Was it a "Dead Counter" repair (0 -> Non-Zero)?
            if orig < MIN_ACTIVITY and final > MIN_ACTIVITY:
                return 0.8 # Moderately high: 0 is a common error mode

            # 7. Fallback
            return 0.5 # Low confidence: Changed based on peer but no verification

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