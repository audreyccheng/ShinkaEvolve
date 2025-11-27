# EVOLVE-BLOCK-START
"""
Network Telemetry Repair using Iterative Constraint Satisfaction
with Residual Synthesis and Gradient-based Voting.

Key Features:
1. Residual Synthesis: Detects and repairs "silent failures" (double-dead links) by calculating 
   residual flow imbalances on connected routers and synthesizing missing traffic rates.
2. Gradient-based Voting: Resolves conflicts using a continuous penalty function based on 
   flow conservation error, allowing for nuanced tie-breaking.
3. Tiered Confidence Calibration: Assigns confidence scores based on strict verification 
   tiers (Gold/Silver/Bronze) and propagates trust transitively.
"""
from typing import Dict, Any, Tuple, List
import collections
import math

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
    
    # Build topology map
    for rid, if_list in topology.items():
        router_map[rid] = if_list
        # A router is verifiable if we have telemetry for ALL its interfaces
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
            # Keep originals for reference
            'orig_rx': float(data.get('rx_rate', 0.0)),
            'orig_tx': float(data.get('tx_rate', 0.0)),
            'orig_status': data.get('interface_status', 'down')
        }

    # --- Helper: Flow Calculations ---
    def get_flow_error(rid, override_if=None, override_field=None, override_val=0.0):
        """Calculates relative flow error with optional value override."""
        if rid not in verifiable_routers:
            return None
            
        sum_rx, sum_tx = 0.0, 0.0
        for iface in router_map[rid]:
            r = state[iface]['rx']
            t = state[iface]['tx']
            
            if iface == override_if:
                if override_field == 'rx': r = override_val
                else: t = override_val
                
            sum_rx += r
            sum_tx += t
            
        diff = abs(sum_rx - sum_tx)
        denom = max(sum_rx, sum_tx, 1.0)
        return diff / denom

    def solve_residual(rid, if_target, field):
        """
        Solves for the value of if_target that would perfectly balance the router.
        Returns None if not solvable (router not verifiable) or implied value is negative.
        """
        if rid not in verifiable_routers:
            return None
            
        # Calculate sums EXCLUDING the target interface/field
        sum_rx, sum_tx = 0.0, 0.0
        for iface in router_map[rid]:
            r = state[iface]['rx']
            t = state[iface]['tx']
            
            if iface == if_target:
                if field == 'rx': r = 0.0
                else: t = 0.0
            
            sum_rx += r
            sum_tx += t
            
        # Balance equation: sum_rx + (target if rx) = sum_tx + (target if tx)
        if field == 'rx':
            required = sum_tx - sum_rx
        else: # tx
            required = sum_rx - sum_tx
            
        if required < 0: return 0.0 # Can't have negative rate
        return required

    # --- 2. Status Repair ---
    status_conf_map = {}
    
    for if_id, s in state.items():
        orig_st = s['orig_status']
        peer_id = s['connected_to']
        
        # Traffic check
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
            # Conflict, no traffic -> Conservative Down
            final_st = 'down'
            conf = 0.7
            
        state[if_id]['status'] = final_st
        status_conf_map[if_id] = conf
        
        # Zero out down links to help flow conservation
        if final_st == 'down':
            state[if_id]['rx'] = 0.0
            state[if_id]['tx'] = 0.0

    # --- 3. Rate Repair (Iterative) ---
    for _ in range(3):
        for if_id, s in state.items():
            if s['status'] == 'down': continue
            
            peer_id = s['connected_to']
            if not peer_id or peer_id not in state: continue
            
            # Candidates
            val_tx = s['tx']
            val_rx = state[peer_id]['rx']
            
            candidates = [val_tx, val_rx]
            
            # --- Residual Synthesis (The "Double-Dead" Fix) ---
            # If both ends report low traffic, but connected routers have imbalances,
            # we might need to synthesize a value.
            if val_tx < MIN_ACTIVITY and val_rx < MIN_ACTIVITY:
                rid_loc = s['local_router']
                rid_rem = s['remote_router']
                
                # What value does Local Router want?
                ideal_tx = solve_residual(rid_loc, if_id, 'tx')
                # What value does Remote Router want?
                ideal_rx = solve_residual(rid_rem, peer_id, 'rx')
                
                if ideal_tx is not None and ideal_rx is not None:
                    # If both routers imply a missing flow of similar magnitude
                    if ideal_tx > MIN_ACTIVITY and ideal_rx > MIN_ACTIVITY:
                        avg_ideal = (ideal_tx + ideal_rx) / 2.0
                        diff_ideal = abs(ideal_tx - ideal_rx)
                        if diff_ideal < max(avg_ideal * 0.1, 5.0): # 10% or 5Mbps tolerance
                            candidates.append(avg_ideal)
            
            # --- Voting ---
            rid_loc = s['local_router']
            rid_rem = s['remote_router']
            
            best_val = val_tx
            best_score = float('inf')
            
            # Evaluate candidates
            # We filter duplicates to save compute
            unique_candidates = sorted(list(set(candidates)))
            
            for cand in unique_candidates:
                # Local Penalty (TX side)
                err_loc = get_flow_error(rid_loc, if_id, 'tx', cand)
                if err_loc is None:
                    pen_loc = 0.05 # Small penalty for uncertainty
                else:
                    # Continuous penalty: square the error to punish large deviations more
                    # But cap it so huge errors don't break float math
                    pen_loc = min(err_loc * 10, 2.0) 
                
                # Remote Penalty (RX side)
                err_rem = get_flow_error(rid_rem, peer_id, 'rx', cand)
                if err_rem is None:
                    pen_rem = 0.05
                else:
                    pen_rem = min(err_rem * 10, 2.0)
                
                total_score = pen_loc + pen_rem
                
                # Heuristic: Penalize Zeros if we have a non-zero candidate that fits well
                # This helps the solver prefer the "Active" candidate over the "Dead" one
                # if flow conservation allows it.
                if cand < MIN_ACTIVITY and max(unique_candidates) > MIN_ACTIVITY:
                    total_score += 0.5
                    
                if total_score < best_score:
                    best_score = total_score
                    best_val = cand
                    
            # Check for symmetry agreement (Smoothing)
            # If the original candidates agree closely, prefer their average over a potentially
            # slightly-better-fitting but synthetic value, unless the synthetic score is MUCH better.
            diff_orig = abs(val_tx - val_rx)
            if diff_orig < max(val_tx * TOLERANCE, MIN_ACTIVITY):
                avg = (val_tx + val_rx) / 2.0
                # If average is essentially as good as the best found (within small margin)
                # use average to reduce noise.
                if abs(avg - best_val) < max(avg * 0.1, 1.0):
                    best_val = avg

            state[if_id]['tx'] = best_val
            state[peer_id]['rx'] = best_val

    # --- 4. Confidence Calibration ---
    result = {}
    
    # Calculate Final Router Health
    router_health = {} # rid -> error
    for rid in verifiable_routers:
        router_health[rid] = get_flow_error(rid) or 0.0
        
    for if_id, s in state.items():
        orig_rx = s['orig_rx']
        final_rx = s['rx']
        orig_tx = s['orig_tx']
        final_tx = s['tx']
        
        rid_loc = s['local_router']
        rid_rem = s['remote_router']
        peer_id = s['connected_to']
        
        def calculate_confidence(orig, final, field):
            # 1. Verification Tiers
            loc_err = router_health.get(rid_loc)
            loc_verifiable = (rid_loc in verifiable_routers)
            loc_ok = loc_verifiable and (loc_err < FLOW_TOLERANCE)
            
            rem_err = router_health.get(rid_rem)
            rem_verifiable = (rid_rem in verifiable_routers)
            rem_ok = rem_verifiable and (rem_err < FLOW_TOLERANCE)
            
            # Peer Consistency
            peer_consistent = True
            if peer_id in state:
                peer_val = state[peer_id]['tx'] if field == 'rx' else state[peer_id]['rx']
                if abs(final - peer_val) > max(final, peer_val, 1.0) * TOLERANCE:
                    peer_consistent = False
            
            changed = abs(orig - final) > max(orig * 0.001, 0.001)
            smoothing = changed and abs(orig - final) < max(orig * 0.05, 0.5)
            
            # --- Tiered Scoring ---
            
            # TIER 1: Gold Standard (Double Verified)
            if loc_ok and rem_ok:
                base = 0.99 if not changed else 0.98
                return base
                
            # TIER 2: Silver (Single Verified)
            if loc_ok:
                if not changed: return 0.98
                if smoothing: return 0.96
                return 0.95
                
            if rem_ok:
                if not changed: return 0.95
                if smoothing: return 0.93
                return 0.90
            
            # TIER 3: Bronze (Consistent but Unverified)
            if peer_consistent:
                # Transitive trust: If peer is on a verified router, trust this link more
                peer_router_ok = rem_ok # Remote is the peer's local
                if peer_router_ok:
                    return 0.92 # Trust propagation
                
                if not changed: return 0.90
                if smoothing: return 0.90
                
                # Heuristic: Dead Repair
                if orig < MIN_ACTIVITY and final > MIN_ACTIVITY:
                    return 0.85
                
                return 0.75
                
            # TIER 4: Iron (Inconsistent / Guess)
            if not changed: return 0.7 # Kept bad value?
            return 0.6
            
        rx_conf = calculate_confidence(orig_rx, final_rx, 'rx')
        tx_conf = calculate_confidence(orig_tx, final_tx, 'tx')
        st_conf = status_conf_map.get(if_id, 1.0)
        
        # Penalty for Down state inconsistency
        if s['status'] == 'down' and (final_rx > MIN_ACTIVITY or final_tx > MIN_ACTIVITY):
            rx_conf = 0.0
            tx_conf = 0.0
            st_conf = 0.0
            
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