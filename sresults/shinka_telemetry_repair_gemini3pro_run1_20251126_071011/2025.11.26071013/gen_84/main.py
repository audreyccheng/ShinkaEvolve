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

        # If status is DOWN, force rates to zero to prevent pollution of flow calc
        if new_status == 'down':
            state[if_id]['rx'] = 0.0
            state[if_id]['tx'] = 0.0

    # --- 3. Rate Repair (Iterative Consensus) ---

    def calc_flow_error(rid, if_target, field, value):
        """Calculates flow error for a router given a hypothetical value."""
        if rid not in verifiable_routers: return None
        sum_rx, sum_tx = 0.0, 0.0
        for iface in router_interfaces[rid]:
            r = state[iface]['rx']
            t = state[iface]['tx']
            if iface == if_target:
                if field == 'rx': r = value
                else: t = value
            sum_rx += r
            sum_tx += t
        return abs(sum_rx - sum_tx) / max(sum_rx, sum_tx, 1.0)

    def calculate_residual(rid, if_target, field):
        """Calculates the value required to perfectly balance the router."""
        if rid not in verifiable_routers: return None
        sum_rx, sum_tx = 0.0, 0.0
        for iface in router_interfaces[rid]:
            # Exclude the target component from the sum
            if iface == if_target:
                r = 0.0 if field == 'rx' else state[iface]['rx']
                t = 0.0 if field == 'tx' else state[iface]['tx']
            else:
                r = state[iface]['rx']
                t = state[iface]['tx']
            sum_rx += r
            sum_tx += t
        # Balance: Sum(Rx) = Sum(Tx) -> Target = Total_Opposite - Sum_Others
        val = sum_tx - sum_rx if field == 'rx' else sum_rx - sum_tx
        return max(val, 0.0)

    # Run 3 passes to allow corrections to propagate
    for _ in range(3):
        for if_id, s in state.items():
            if s['status'] == 'down': continue
            peer_id = s['connected_to']
            if not peer_id or peer_id not in state: continue

            # Link: Tx(Local) -> Rx(Peer)
            meas_tx = s['tx']
            meas_rx = state[peer_id]['rx']

            # Candidates: (value, weight, source_type)
            candidates = []
            candidates.append((meas_tx, 1.0, 'meas'))
            candidates.append((meas_rx, 1.0, 'meas'))

            # Calculate Residuals (Derived candidates)
            rid_loc = s['local_router']
            res_tx = calculate_residual(rid_loc, if_id, 'tx')
            if res_tx is not None:
                # High weight: Residuals represent the consensus of N other interfaces
                candidates.append((res_tx, 2.0, 'resid'))

            rid_rem = state[peer_id]['local_router']
            res_rx = calculate_residual(rid_rem, peer_id, 'rx')
            if res_rx is not None:
                candidates.append((res_rx, 2.0, 'resid'))

            # Clustering
            clusters = []
            for val, w, src in candidates:
                matched = False
                for c in clusters:
                    avg = c['sum'] / c['count']
                    diff = abs(val - avg)
                    if diff < max(avg * TOLERANCE, MIN_ACTIVITY):
                        c['sum'] += val
                        c['w'] += w
                        c['count'] += 1
                        matched = True
                        break
                if not matched:
                    clusters.append({'sum': val, 'w': w, 'count': 1})

            # Select Best Cluster
            best_val = meas_tx
            best_score = -1.0

            # Find max value to detect "Dead vs Alive" scenarios
            max_val = max((c['sum']/c['count'] for c in clusters), default=0.0)

            for c in clusters:
                val = c['sum'] / c['count']
                score = c['w']

                # Heuristic: If we have a significant "Alive" signal, penalize "Dead" signals
                # This fixes "Double Dead" or "One Dead" scenarios where flow implies traffic
                if val < MIN_ACTIVITY and max_val > MIN_ACTIVITY:
                    score *= 0.1

                if score > best_score:
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
            # Check verification status (Flow Conservation)
            rid = data.get('local_router')
            flow_err = calc_flow_error(rid, if_id, field, final)
            local_verified = (flow_err is not None and flow_err < FLOW_TOLERANCE)

            remote_verified = is_remotely_verified(final, field)

            # Check consistency with peer (Symmetry)
            peer_consistent = True
            peer_id = data.get('connected_to')
            if peer_id and peer_id in state:
                peer_val = state[peer_id]['tx'] if field == 'rx' else state[peer_id]['rx']
                if abs(final - peer_val) > max(final, peer_val, 1.0) * TOLERANCE:
                    peer_consistent = False

            changed = abs(orig - final) > 0.001

            # --- Confidence Scoring ---

            # Case 1: Verified Correctness (Gold Standard)
            if local_verified and remote_verified:
                return 0.99 if not changed else 0.98

            # Case 2: Local Verification (Silver Standard)
            if local_verified:
                if not changed: return 0.98
                return 0.95

            # Case 3: Remote Verification (Silver Standard)
            if remote_verified:
                if not changed: return 0.95
                return 0.90

            # Case 4: No Verification (Unverifiable Routers)
            if flow_err is None:
                if not peer_consistent:
                    return 0.6 # Disagreement and no way to check

                if not changed:
                    return 0.9 # Assume innocence

                # We changed it: Why?
                # Smoothing
                if orig > MIN_ACTIVITY and abs(orig - final) / orig < 0.05:
                    return 0.95
                # Dead Repair (0 -> Active)
                if orig < MIN_ACTIVITY and final > MIN_ACTIVITY:
                    return 0.85

                # Changed to match peer?
                return 0.75

            # Case 5: Verification Failed (Broken Router or Bad Value)
            # If we are here, flow_err is High (>= TOLERANCE)
            if not changed:
                return 0.6 # Retaining a value that breaks flow conservation is suspicious

            # We changed it, but it still doesn't satisfy flow?
            # Maybe the router is just broken (packet loss).
            return 0.5

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