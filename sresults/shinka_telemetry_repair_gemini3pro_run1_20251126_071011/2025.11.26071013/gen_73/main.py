# EVOLVE-BLOCK-START
"""
Consensus-based Network Telemetry Repair V2
Combines iterative constraint satisfaction with robust confidence scoring.
Base logic from high-performing inspiration program, refined with "Broken Router" detection from prior work.
"""
from typing import Dict, Any, Tuple, List
import collections

def repair_network_telemetry(telemetry: Dict[str, Dict[str, Any]],
                             topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:

    # --- Constants ---
    TOLERANCE = 0.02          # 2% symmetry tolerance
    FLOW_TOLERANCE = 0.05     # 5% flow conservation tolerance
    MIN_ACTIVITY = 0.05       # Mbps threshold for "active" traffic

    # --- 1. Initialization ---
    state = {}
    router_map = collections.defaultdict(list)
    verifiable_routers = set()

    # Build topology map and identify verifiable routers
    # A router is verifiable if we have telemetry for ALL its interfaces
    for rid, if_list in topology.items():
        router_map[rid] = if_list
        if all(if_id in telemetry for if_id in if_list):
            verifiable_routers.add(rid)

    # Initialize state from telemetry
    for if_id, data in telemetry.items():
        state[if_id] = {
            'rx': float(data.get('rx_rate', 0.0)),
            'tx': float(data.get('tx_rate', 0.0)),
            'status': data.get('interface_status', 'down'),
            'orig_rx': float(data.get('rx_rate', 0.0)),
            'orig_tx': float(data.get('tx_rate', 0.0)),
            'orig_status': data.get('interface_status', 'down'),
            'local_router': data.get('local_router'),
            'connected_to': data.get('connected_to'),
            'remote_router': data.get('remote_router')
        }

    # --- Helper: Flow Error Calculation ---
    def get_flow_error(rid, if_target=None, field=None, value=None):
        """
        Calculates the relative flow error (imbalance) for a router.
        Optionally allows substituting a speculative value for one interface.
        """
        if rid not in verifiable_routers:
            return None

        sum_rx, sum_tx = 0.0, 0.0
        for iface in router_map[rid]:
            # Use speculative value if this is the target interface
            if iface == if_target:
                r = value if field == 'rx' else state[iface]['rx']
                t = value if field == 'tx' else state[iface]['tx']
            else:
                r = state[iface]['rx']
                t = state[iface]['tx']
            sum_rx += r
            sum_tx += t

        diff = abs(sum_rx - sum_tx)
        denom = max(sum_rx, sum_tx, 1.0)
        return diff / denom

    def get_residual(rid, if_target, field):
        """Calculates the value needed at an interface to perfectly balance a router."""
        if rid not in verifiable_routers:
            return None

        sum_in, sum_out = 0.0, 0.0
        for iface in router_map[rid]:
            if iface == if_target: continue
            sum_in += state[iface]['rx']
            sum_out += state[iface]['tx']

        # We want Sum_In = Sum_Out
        # If solving for Tx (out): x = Sum_In - Sum_Out_Other
        # If solving for Rx (in):  x = Sum_Out - Sum_In_Other
        val = sum_in - sum_out if field == 'tx' else sum_out - sum_in
        return max(val, 0.0)

    # --- 2. Status Repair ---
    status_confidence = {}

    for if_id, s in state.items():
        orig = s['orig_status']
        peer_id = s['connected_to']

        # Check for traffic activity
        local_traffic = (s['orig_rx'] > MIN_ACTIVITY) or (s['orig_tx'] > MIN_ACTIVITY)
        peer_traffic = False
        peer_status = 'unknown'

        if peer_id and peer_id in state:
            peer = state[peer_id]
            peer_traffic = (peer['orig_rx'] > MIN_ACTIVITY) or (peer['orig_tx'] > MIN_ACTIVITY)
            peer_status = peer['orig_status']

        # Decision Logic
        final_status = orig
        conf = 1.0

        if local_traffic or peer_traffic:
            # Traffic implies UP
            final_status = 'up'
            if orig == 'down':
                conf = 0.95 # Correcting a false negative
        elif orig == 'up' and peer_status == 'down':
            # Peer says down + no traffic -> likely down
            final_status = 'down'
            conf = 0.8
        elif orig != peer_status:
            # Conflict with no traffic info. Conservative choice: Down.
            final_status = 'down'
            conf = 0.7

        state[if_id]['status'] = final_status
        status_confidence[if_id] = conf

        # Enforce consistency: Down interfaces have zero rate
        if final_status == 'down':
            state[if_id]['rx'] = 0.0
            state[if_id]['tx'] = 0.0

    # --- 3. Rate Repair (Iterative Consensus) ---
    # Run multiple passes to allow flow corrections to propagate
    for _ in range(3):
        for if_id, s in state.items():
            if s['status'] == 'down': continue

            peer_id = s['connected_to']
            if not peer_id or peer_id not in state: continue

            # The link connects Local(Tx) -> Remote(Rx)
            cand_tx = s['tx']              # Local view
            cand_rx = state[peer_id]['rx'] # Remote view

            # Agreement Check
            diff = abs(cand_tx - cand_rx)
            mag = max(cand_tx, cand_rx, 1.0)

            if diff < max(mag * TOLERANCE, MIN_ACTIVITY):
                new_val = (cand_tx + cand_rx) / 2.0
            else:
                # Disagreement: Use Flow Conservation
                rid_local = s['local_router']
                rid_remote = s['remote_router']

                # Gather Candidates
                candidates = [cand_tx, cand_rx]

                # Try to synthesize values from flow conservation
                res_local = get_residual(rid_local, if_id, 'tx')
                if res_local is not None: candidates.append(res_local)

                res_remote = get_residual(rid_remote, peer_id, 'rx')
                if res_remote is not None: candidates.append(res_remote)

                # Deduplicate and Score
                best_score = float('inf')
                best_val = cand_tx

                # Filter candidates close to each other
                unique_cands = []
                for c in candidates:
                    if not any(abs(c - u) < 1e-4 for u in unique_cands):
                        unique_cands.append(c)

                for cand in unique_cands:
                    err_local = get_flow_error(rid_local, if_id, 'tx', cand)
                    err_remote = get_flow_error(rid_remote, peer_id, 'rx', cand)

                    def get_penalty(err):
                        if err is None: return 0.05 # Unverifiable: small penalty
                        return min(err, 2.0)        # Verifiable: proportional penalty

                    score = get_penalty(err_local) + get_penalty(err_remote)

                    # Heuristic: Penalize zero if alternatives exist (Dead counter)
                    if cand < MIN_ACTIVITY and max(unique_cands) > MIN_ACTIVITY:
                        score += 0.5

                    if score < best_score:
                        best_score = score
                        best_val = cand

                new_val = best_val

            # Update state immediately
            state[if_id]['tx'] = new_val
            state[peer_id]['rx'] = new_val

    # --- 4. Confidence Calibration ---
    result = {}

    # Pre-calculate final flow errors for context
    final_router_errors = {rid: get_flow_error(rid) for rid in verifiable_routers}

    for if_id, s in state.items():
        orig_rx, final_rx = s['orig_rx'], s['rx']
        orig_tx, final_tx = s['orig_tx'], s['tx']

        rid = s['local_router']
        peer_id = s['connected_to']

        def calculate_confidence(orig, final, field):
            # 1. Verification Status
            # Local
            local_err = final_router_errors.get(rid)
            local_verified = (local_err is not None and local_err < FLOW_TOLERANCE)

            # Remote
            remote_verified = False
            rem_rid = s['remote_router']
            if rem_rid in final_router_errors:
                 if final_router_errors[rem_rid] < FLOW_TOLERANCE:
                     remote_verified = True

            # 2. Change Analysis
            changed = abs(orig - final) > max(orig * 0.001, 0.001)
            # Smoothing = Changed, but within 5%
            is_smoothing = changed and (abs(orig - final) < max(orig * 0.05, 0.1))

            # 3. Peer Consistency (Symmetry)
            peer_consistent = True
            if peer_id in state:
                # If checking Rx, compare with Peer Tx
                peer_val = state[peer_id]['tx'] if field == 'rx' else state[peer_id]['rx']
                if abs(final - peer_val) > max(final, peer_val, 1.0) * TOLERANCE:
                    peer_consistent = False

            # --- Scoring ---

            if not changed:
                # Unchanged
                if local_verified and remote_verified: return 1.0
                if local_verified: return 0.98
                if not peer_consistent: return 0.7 # Kept original despite conflict

                # Check for Broken Router (Unverified locally, but demonstrably bad)
                if local_err is not None and local_err >= FLOW_TOLERANCE:
                    return 0.6

                return 0.9 # Default for unchanged

            if is_smoothing:
                return 0.95 # High confidence in smoothing

            # Significant Correction
            if local_verified and remote_verified:
                return 0.98 # Validated by both ends
            if local_verified:
                return 0.95 # Validated by local math
            if remote_verified:
                return 0.90 # Validated by remote math

            # Unverified Corrections
            # Repairing a "Dead" counter (0 -> Active)
            if orig < MIN_ACTIVITY and final > MIN_ACTIVITY:
                return 0.85

            # Total guess / forced agreement
            return 0.6

        rx_conf = calculate_confidence(orig_rx, final_rx, 'rx')
        tx_conf = calculate_confidence(orig_tx, final_tx, 'tx')
        st_conf = status_confidence.get(if_id, 1.0)

        # Sanity override for Down state
        if s['status'] == 'down':
             if final_rx > MIN_ACTIVITY or final_tx > MIN_ACTIVITY:
                 rx_conf = 0.0
                 tx_conf = 0.0

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