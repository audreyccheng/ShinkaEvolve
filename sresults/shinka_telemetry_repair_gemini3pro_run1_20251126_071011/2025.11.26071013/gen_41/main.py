# EVOLVE-BLOCK-START
"""
Network Telemetry Repair using Iterative Constraint Satisfaction
with Discrete Voting and Global Consistency Penalties.

Key features:
- Three-stage pipeline: Status Repair -> Rate Repair -> Confidence Scoring
- Iterative consensus for rates using Link Symmetry and Flow Conservation
- "Discrete Voting" mechanism to resolve conflicts (robust against measurement noise)
- Confidence calibration with Global Consistency Penalty for broken routers
"""
from typing import Dict, Any, Tuple, List
import collections

def repair_network_telemetry(telemetry: Dict[str, Dict[str, Any]],
                             topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:

    # --- Constants ---
    TOLERANCE = 0.02          # 2% symmetry tolerance
    FLOW_TOLERANCE = 0.05     # 5% flow conservation tolerance
    MIN_ACTIVITY = 0.05       # 50 Kbps threshold (Higher threshold filters noise better)

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
    # Helper: Synthesize the value required to balance the router flow
    def get_flow_target(rid, if_exclude, direction):
        """
        Returns the rate required at if_exclude to balance flow at rid.
        direction: 'tx' (outbound from rid) or 'rx' (inbound to rid)
        """
        if rid not in verifiable_routers:
            return None

        sum_in = 0.0
        sum_out = 0.0

        for iface in router_map[rid]:
            if iface == if_exclude: continue
            sum_in += state[iface]['rx']
            sum_out += state[iface]['tx']

        if direction == 'tx':
            # Target = Sum_In - Sum_Others_Out
            val = sum_in - sum_out
        else:
            # Target = Sum_Out - Sum_Others_In
            val = sum_out - sum_in

        return max(0.0, val)

    # Run multiple passes to allow flow corrections to propagate
    for _ in range(3):
        for if_id, s in state.items():
            if s['status'] == 'down': continue

            peer_id = s['connected_to']
            if not peer_id or peer_id not in state: continue

            val_tx = s['tx']              # Local view
            val_rx = state[peer_id]['rx'] # Remote view

            # 1. Agreement Check
            diff = abs(val_tx - val_rx)
            avg = (val_tx + val_rx) / 2.0

            is_agreed = diff < max(avg * TOLERANCE, MIN_ACTIVITY)

            # consistency check: If agreed on ~Zero, but router demands traffic, invalidate agreement.
            if is_agreed and avg < MIN_ACTIVITY:
                rid_local = s['local_router']
                if rid_local in verifiable_routers:
                     if get_flow_error(rid_local, if_id, 'tx', avg) > FLOW_TOLERANCE:
                         is_agreed = False

                rid_remote = s['remote_router']
                if is_agreed and rid_remote in verifiable_routers:
                     if get_flow_error(rid_remote, peer_id, 'rx', avg) > FLOW_TOLERANCE:
                         is_agreed = False

            if is_agreed:
                state[if_id]['tx'] = avg
                state[peer_id]['rx'] = avg
                continue

            # 2. Conflict Resolution / Synthesis
            candidates = {val_tx, val_rx, avg}

            # Add synthesized candidates (What flow conservation suggests)
            rid_local = s['local_router']
            synth_tx = get_flow_target(rid_local, if_id, 'tx')
            if synth_tx is not None: candidates.add(synth_tx)

            rid_remote = s['remote_router']
            synth_rx = get_flow_target(rid_remote, peer_id, 'rx')
            if synth_rx is not None: candidates.add(synth_rx)

            # Score candidates
            best_val = avg
            min_cost = float('inf')

            for cand in candidates:
                # Cost function: Sum of flow errors at both ends
                # 0.0 = Verified Perfect
                # 0.05 = Unverifiable (Neutral cost)
                # >1.0 = Verified Violation

                err_local = get_flow_error(rid_local, if_id, 'tx', cand)
                if err_local is None: c_local = 0.05
                elif err_local < FLOW_TOLERANCE: c_local = err_local
                else: c_local = 1.0 + err_local

                err_remote = get_flow_error(rid_remote, peer_id, 'rx', cand)
                if err_remote is None: c_remote = 0.05
                elif err_remote < FLOW_TOLERANCE: c_remote = err_remote
                else: c_remote = 1.0 + err_remote

                total_cost = c_local + c_remote

                # Heuristic: Avoid Zero if alternatives exist and look valid
                if cand < MIN_ACTIVITY and any(c > MIN_ACTIVITY for c in candidates):
                     total_cost += 0.5

                if total_cost < min_cost:
                    min_cost = total_cost
                    best_val = cand

            # Update state
            state[if_id]['tx'] = best_val
            state[peer_id]['rx'] = best_val

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
            # Check verification of the FINAL value
            # Note: We re-calculate specific error for this value to ensure
            # we don't punish this interface for OTHER interfaces' errors.
            local_err = get_flow_error(rid, if_id, field, final)
            local_verified = (local_err is not None and local_err < FLOW_TOLERANCE)

            # Remote check
            remote_verified = False
            rem_rid = s['remote_router']
            if rem_rid:
                 check_field = 'tx' if field == 'rx' else 'rx'
                 rem_err = get_flow_error(rem_rid, peer_id, check_field, final)
                 if rem_err is not None and rem_err < FLOW_TOLERANCE:
                     remote_verified = True

            # Peer Symmetry check
            peer_consistent = True
            if peer_id in state:
                peer_val = state[peer_id]['tx'] if field == 'rx' else state[peer_id]['rx']
                if abs(final - peer_val) > max(final, peer_val, 1.0) * TOLERANCE:
                    peer_consistent = False

            changed = abs(orig - final) > max(orig * 0.001, 0.001)
            is_smoothing = changed and (abs(orig - final) < max(orig * 0.05, 0.1))

            # --- Confidence Scoring ---

            if not changed:
                if local_verified and remote_verified: return 1.0
                if local_verified: return 0.98
                if remote_verified: return 0.95
                if not peer_consistent: return 0.7
                return 0.9

            if is_smoothing:
                return 0.95

            # Significant Repairs
            if local_verified and remote_verified: return 0.99
            if local_verified: return 0.95
            if remote_verified: return 0.90

            # Heuristic Repairs
            if orig < MIN_ACTIVITY and final > MIN_ACTIVITY:
                return 0.85 # Dead counter repair

            return 0.6 # Best guess

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