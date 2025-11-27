# EVOLVE-BLOCK-START
"""
Consensus+Residual Network Telemetry Repair
Combines iterative constraint satisfaction with Residual Synthesis to generate
high-quality repair candidates from Flow Conservation constraints.
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
    
    # Build topology and identify verifiable routers
    # A router is verifiable if we have telemetry for ALL its interfaces
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
            'orig_rx': float(data.get('rx_rate', 0.0)),
            'orig_tx': float(data.get('tx_rate', 0.0)),
            'orig_status': data.get('interface_status', 'down'),
            'local_router': data.get('local_router'),
            'connected_to': data.get('connected_to'),
            'remote_router': data.get('remote_router')
        }

    # --- 2. Status Repair ---
    status_conf_map = {}
    
    for if_id, s in state.items():
        orig_status = s['orig_status']
        peer_id = s['connected_to']
        
        # Check activity
        local_active = (s['rx'] > MIN_ACTIVITY) or (s['tx'] > MIN_ACTIVITY)
        peer_active = False
        peer_status = 'unknown'
        
        if peer_id and peer_id in state:
            p = state[peer_id]
            peer_active = (p['rx'] > MIN_ACTIVITY) or (p['tx'] > MIN_ACTIVITY)
            peer_status = p['orig_status']
            
        # Status Decision
        new_status = orig_status
        conf = 1.0
        
        if local_active or peer_active:
            new_status = 'up'
            if orig_status == 'down': conf = 0.95
        elif orig_status == 'up' and peer_status == 'down':
            new_status = 'down'
            conf = 0.8
        elif orig_status != peer_status:
            new_status = 'down'
            conf = 0.7
            
        state[if_id]['status'] = new_status
        status_conf_map[if_id] = conf
        
        # Enforce consistency for DOWN links
        if new_status == 'down':
            state[if_id]['rx'] = 0.0
            state[if_id]['tx'] = 0.0

    # --- 3. Rate Repair (Iterative Consensus with Residual Synthesis) ---

    def get_router_imbalance(rid, exclude_if=None, exclude_field=None):
        """
        Calculates Sum(In) and Sum(Out) for a router, optionally excluding a specific component.
        """
        if rid not in verifiable_routers:
            return None, None
            
        sum_in = 0.0
        sum_out = 0.0
        
        for iface in router_map[rid]:
            r = state[iface]['rx']
            t = state[iface]['tx']
            
            if iface == exclude_if:
                if exclude_field == 'rx': r = 0.0
                elif exclude_field == 'tx': t = 0.0
                
            sum_in += r
            sum_out += t
            
        return sum_in, sum_out

    def calc_error_with_val(rid, if_target, field, val):
        """Calculates flow error for a router if we force a specific value."""
        if rid not in verifiable_routers: return None
        s_in, s_out = get_router_imbalance(rid, if_target, field)
        
        if field == 'rx': s_in += val
        else: s_out += val
        
        diff = abs(s_in - s_out)
        denom = max(s_in, s_out, 1.0)
        return diff / denom

    # 3 Iterations allow updates to propagate
    for _ in range(3):
        for if_id, s in state.items():
            if s['status'] == 'down': continue
            
            peer_id = s['connected_to']
            if not peer_id or peer_id not in state: continue
            
            # We are solving for the traffic flow: Local TX -> Remote RX
            
            # 1. Gather Candidates
            candidates = []
            
            # A. Actual Measurements
            val_tx = s['tx']
            val_rx = state[peer_id]['rx']
            candidates.append(val_tx)
            candidates.append(val_rx)
            
            # B. Synthetic (Residual) Candidates
            # What value is needed to balance the Local router?
            rid_loc = s['local_router']
            sin_loc, sout_loc = get_router_imbalance(rid_loc, if_id, 'tx')
            if sin_loc is not None:
                # In = Out => In = (Out_Others + X) => X = In - Out_Others
                synth_loc = sin_loc - sout_loc
                if synth_loc > MIN_ACTIVITY: candidates.append(synth_loc)
                
            # What value is needed to balance the Remote router?
            rid_rem = s['remote_router']
            sin_rem, sout_rem = get_router_imbalance(rid_rem, peer_id, 'rx')
            if sin_rem is not None:
                # In = Out => (In_Others + X) = Out => X = Out - In_Others
                synth_rem = sout_rem - sin_rem
                if synth_rem > MIN_ACTIVITY: candidates.append(synth_rem)

            # Deduplicate (within small epsilon)
            unique_cands = []
            for c in candidates:
                if c < -1e-9: continue # Filter negative residuals
                c = max(c, 0.0)
                is_dupe = False
                for u in unique_cands:
                    if abs(c - u) < 1e-4:
                        is_dupe = True
                        break
                if not is_dupe:
                    unique_cands.append(c)
            
            # 2. Score Candidates (Lower is better)
            best_score = float('inf')
            best_val = val_tx
            
            # Context: Do original measurements agree?
            meas_diff = abs(val_tx - val_rx)
            meas_mag = max(val_tx, val_rx, 1.0)
            measurements_agree = meas_diff < max(meas_mag * TOLERANCE, MIN_ACTIVITY)
            avg_measurement = (val_tx + val_rx) / 2.0
            
            for c in unique_cands:
                score = 0.0
                
                # Flow Verification Scores
                # Local Router
                err_loc = calc_error_with_val(rid_loc, if_id, 'tx', c)
                if err_loc is not None: score += min(err_loc, 1.0)
                else: score += 0.05 # Small penalty for unverifiability
                
                # Remote Router
                err_rem = calc_error_with_val(rid_rem, peer_id, 'rx', c)
                if err_rem is not None: score += min(err_rem, 1.0)
                else: score += 0.05
                
                # Heuristic: Dead Counter Penalty
                # If this candidate is near zero, but other viable candidates exist, penalize zero.
                if c < MIN_ACTIVITY and max(unique_cands) > MIN_ACTIVITY:
                    score += 0.5
                    
                # Heuristic: Measurement Consistency Bonus
                # If measurements agree, penalize candidates that deviate significantly.
                # This protects symmetric links from being distorted by errors elsewhere in the topology.
                if measurements_agree:
                    dist = abs(c - avg_measurement)
                    if dist > max(avg_measurement * 0.1, 1.0):
                        score += 0.5
                        
                if score < best_score:
                    best_score = score
                    best_val = c
            
            state[if_id]['tx'] = best_val
            state[peer_id]['rx'] = best_val

    # --- 4. Confidence Calibration ---
    result = {}
    
    # Calculate final router flow errors for verification
    final_errors = {}
    for rid in verifiable_routers:
        sin, sout = get_router_imbalance(rid)
        final_errors[rid] = abs(sin - sout) / max(sin, sout, 1.0)
        
    for if_id, data in telemetry.items():
        s = state[if_id]
        orig_rx, final_rx = s['orig_rx'], s['rx']
        orig_tx, final_tx = s['orig_tx'], s['tx']
        
        def is_verified(rid):
            return rid in final_errors and final_errors[rid] < FLOW_TOLERANCE

        def get_conf(orig, final, field):
            # Verification Status
            rid = s['local_router']
            loc_ver = is_verified(rid)
            
            rem_ver = False
            peer_id = s['connected_to']
            rem_rid = s['remote_router']
            if peer_id and rem_rid:
                rem_ver = is_verified(rem_rid)
                
            # Peer Consistency
            peer_consistent = True
            if peer_id and peer_id in state:
                p_val = state[peer_id]['tx'] if field == 'rx' else state[peer_id]['rx']
                if abs(final - p_val) > max(final, 1.0) * TOLERANCE:
                    peer_consistent = False

            changed = abs(orig - final) > 0.001
            
            # --- Scoring Logic ---
            
            # Gold Standard: Verified locally and remotely
            if loc_ver and rem_ver: return 1.0
            
            # Silver Standard: Verified locally (most direct math)
            if loc_ver: return 0.98
            
            if not changed:
                if rem_ver: return 0.95
                if not peer_consistent: return 0.7
                # "Broken Router" check: Unchanged, but local flow is bad
                if rid in final_errors and not loc_ver:
                    return 0.6 
                return 0.9
            
            # Changed Value Logic
            if rem_ver: return 0.90 # Peer forced change, validated by peer's router
            
            # Heuristics (Unverified)
            # Dead Counter Repair (0 -> Value)
            if orig < MIN_ACTIVITY and final > MIN_ACTIVITY:
                return 0.85 
                
            # Smoothing (Small change)
            if abs(orig - final) < max(orig * 0.05, 0.1):
                return 0.95
                
            # Fallback
            return 0.6

        rx_c = get_conf(orig_rx, final_rx, 'rx')
        tx_c = get_conf(orig_tx, final_tx, 'tx')
        st_c = status_conf_map.get(if_id, 1.0)
        
        # Sanity Check
        if s['status'] == 'down' and (final_rx > 1.0 or final_tx > 1.0):
            rx_c = 0.0; tx_c = 0.0
            
        result[if_id] = {
            'rx_rate': (orig_rx, final_rx, rx_c),
            'tx_rate': (orig_tx, final_tx, tx_c),
            'interface_status': (s['orig_status'], s['status'], st_c),
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