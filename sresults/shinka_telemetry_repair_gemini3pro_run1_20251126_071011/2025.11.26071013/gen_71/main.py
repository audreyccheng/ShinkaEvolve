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
    Repairs network telemetry using a hybrid consensus approach.
    Combines iterative constraint satisfaction with residual synthesis for robust repair.
    """

    # --- Constants ---
    TOLERANCE = 0.02          # 2% symmetry tolerance
    FLOW_TOLERANCE = 0.05     # 5% flow conservation tolerance
    MIN_ACTIVITY = 0.05       # Mbps threshold for "active" traffic

    # --- 1. Initialization ---
    state = {}
    router_map = collections.defaultdict(list)
    verifiable_routers = set()

    # Identify verifiable routers (all interfaces monitored)
    for rid, if_list in topology.items():
        router_map[rid] = if_list
        if all(if_id in telemetry for if_id in if_list):
            verifiable_routers.add(rid)

    # Initialize state
    for if_id, data in telemetry.items():
        state[if_id] = {
            'rx': float(data.get('rx_rate', 0.0)),
            'tx': float(data.get('tx_rate', 0.0)),
            'status': data.get('interface_status', 'down'),
            'connected_to': data.get('connected_to'),
            'local_router': data.get('local_router'),
            'remote_router': data.get('remote_router'),
            # Keep originals
            'orig_rx': float(data.get('rx_rate', 0.0)),
            'orig_tx': float(data.get('tx_rate', 0.0)),
            'orig_status': data.get('interface_status', 'down')
        }

    # --- 2. Status Repair ---
    status_conf_map = {}

    for if_id, s in state.items():
        orig_st = s['orig_status']
        peer_id = s['connected_to']

        # Traffic check (on originals)
        active = (s['orig_rx'] > MIN_ACTIVITY) or (s['orig_tx'] > MIN_ACTIVITY)
        peer_active = False
        peer_st = 'unknown'

        if peer_id and peer_id in state:
            peer = state[peer_id]
            peer_active = (peer['orig_rx'] > MIN_ACTIVITY) or (peer['orig_tx'] > MIN_ACTIVITY)
            peer_st = peer['orig_status']

        # Logic
        final_st = orig_st
        conf = 1.0

        if active or peer_active:
            final_st = 'up'
            if orig_st == 'down': conf = 0.95
        elif orig_st == 'up' and peer_st == 'down':
            final_st = 'down'
            conf = 0.8
        elif orig_st != peer_st:
            # Conflict with no traffic -> Conservative Down
            final_st = 'down'
            conf = 0.7

        state[if_id]['status'] = final_st
        status_conf_map[if_id] = conf

        # Zero out down links
        if final_st == 'down':
            state[if_id]['rx'] = 0.0
            state[if_id]['tx'] = 0.0

    # --- Helpers for Rate Repair ---
    def get_flow_error(rid, if_target=None, field=None, value=None):
        if rid not in verifiable_routers: return None
        sum_rx, sum_tx = 0.0, 0.0
        for iface in router_map[rid]:
            r = value if (iface == if_target and field == 'rx') else state[iface]['rx']
            t = value if (iface == if_target and field == 'tx') else state[iface]['tx']
            sum_rx += r
            sum_tx += t
        return abs(sum_rx - sum_tx) / max(sum_rx, sum_tx, 1.0)

    def get_residual(rid, if_target, field):
        """Calculates value needed to balance router perfectly."""
        if rid not in verifiable_routers: return None
        sum_in, sum_out = 0.0, 0.0
        for iface in router_map[rid]:
            r = 0.0 if (iface == if_target and field == 'rx') else state[iface]['rx']
            t = 0.0 if (iface == if_target and field == 'tx') else state[iface]['tx']
            sum_in += r
            sum_out += t

        # Balance: Sum_In = Sum_Out
        if field == 'rx': val = sum_out - sum_in
        else: val = sum_in - sum_out
        return max(val, 0.0)

    # --- 3. Rate Repair (Iterative) ---
    for _ in range(3):
        for if_id, s in state.items():
            if s['status'] == 'down': continue

            peer_id = s['connected_to']
            if not peer_id or peer_id not in state: continue

            # Candidates
            candidates = [s['tx'], state[peer_id]['rx']]

            # Add Residuals as candidates
            res_tx = get_residual(s['local_router'], if_id, 'tx')
            if res_tx is not None: candidates.append(res_tx)

            res_rx = get_residual(s['remote_router'], peer_id, 'rx')
            if res_rx is not None: candidates.append(res_rx)

            # Evaluate
            best_val = s['tx']
            best_score = float('inf')

            # Deduplicate and sort
            unique_cands = sorted(list(set([c for c in candidates if c >= 0])))

            for cand in unique_cands:
                # Calculate penalties
                # Local router error
                err_loc = get_flow_error(s['local_router'], if_id, 'tx', cand)
                # Remote router error
                err_rem = get_flow_error(s['remote_router'], peer_id, 'rx', cand)

                score = 0.0
                # Continuous scoring
                if err_loc is None: score += 0.05
                else: score += min(err_loc, 1.0)

                if err_rem is None: score += 0.05
                else: score += min(err_rem, 1.0)

                # Heuristic: Avoid zero if possible when alternates exist
                if cand < MIN_ACTIVITY and max(unique_cands) > MIN_ACTIVITY:
                    score += 0.5

                if score < best_score:
                    best_score = score
                    best_val = cand

            # Smoothing: If best val is close to average of telemetry, use average
            avg_telemetry = (s['tx'] + state[peer_id]['rx']) / 2.0
            if abs(best_val - avg_telemetry) < max(avg_telemetry * 0.1, 1.0):
                 if abs(s['tx'] - state[peer_id]['rx']) < max(avg_telemetry * TOLERANCE, MIN_ACTIVITY):
                     best_val = avg_telemetry

            state[if_id]['tx'] = best_val
            state[peer_id]['rx'] = best_val

    # --- 4. Confidence Calibration ---
    result = {}

    # Calculate final health of routers
    final_router_health = {}
    for rid in verifiable_routers:
        final_router_health[rid] = get_flow_error(rid)

    for if_id, s in state.items():
        orig_rx = s['orig_rx']
        final_rx = s['rx']
        orig_tx = s['orig_tx']
        final_tx = s['tx']

        def get_conf(orig, final, field):
            # 1. Verification
            rid_loc = s['local_router']
            loc_err = final_router_health.get(rid_loc)
            loc_ok = (loc_err is not None and loc_err < FLOW_TOLERANCE)

            # Remote Verification (via Peer)
            peer_id = s['connected_to']
            rem_ok = False
            if peer_id and peer_id in state:
                rid_rem = state[peer_id]['local_router'] # The OTHER router
                rem_err = final_router_health.get(rid_rem)
                rem_ok = (rem_err is not None and rem_err < FLOW_TOLERANCE)

            # Peer Consistency
            peer_consistent = True
            if peer_id and peer_id in state:
                peer_val = state[peer_id]['tx'] if field == 'rx' else state[peer_id]['rx']
                if abs(final - peer_val) > max(final, peer_val, 1.0) * TOLERANCE:
                    peer_consistent = False

            changed = abs(orig - final) > 0.001
            smoothed = changed and abs(orig - final) < max(orig * 0.05, 0.1)

            # Scoring
            if not changed:
                if loc_ok and rem_ok: return 1.0
                if loc_ok: return 0.98
                if not peer_consistent: return 0.7
                if rem_ok: return 0.95
                # Critical: If not changed, but local router is BROKEN, confidence is low
                if loc_err is not None and loc_err > FLOW_TOLERANCE:
                    return 0.6
                return 0.9

            if smoothed: return 0.95

            # Significant changes
            if loc_ok and rem_ok: return 0.99
            if loc_ok: return 0.95 # Local math adds up
            if rem_ok: return 0.90 # Peer math adds up

            # Heuristics
            if orig < MIN_ACTIVITY and final > MIN_ACTIVITY: return 0.85

            return 0.6

        rx_conf = get_conf(orig_rx, final_rx, 'rx')
        tx_conf = get_conf(orig_tx, final_tx, 'tx')
        st_conf = status_conf_map.get(if_id, 1.0)

        if s['status'] == 'down' and (final_rx > 1.0 or final_tx > 1.0):
             rx_conf = 0.0; tx_conf = 0.0; st_conf = 0.0

        result[if_id] = {
            'rx_rate': (orig_rx, final_rx, rx_conf),
            'tx_rate': (orig_tx, final_tx, tx_conf),
            'interface_status': (s['orig_status'], s['status'], st_conf),
            'connected_to': s['connected_to'],
            'local_router': s['local_router'],
            'remote_router': s['remote_router']
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