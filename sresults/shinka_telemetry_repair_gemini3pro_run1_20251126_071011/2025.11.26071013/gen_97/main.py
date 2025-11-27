# EVOLVE-BLOCK-START
"""
Weighted Consensus Network Telemetry Repair
Uses iterative constraint satisfaction with probabilistic scoring based on flow conservation errors.
Key improvements:
- Continuous scoring function instead of discrete votes for better consensus
- Outlier rejection via clustering
- Dynamic confidence calibration based on verification strength
"""
from typing import Dict, Any, Tuple, List
import collections
import math

def repair_network_telemetry(telemetry: Dict[str, Dict[str, Any]], 
                             topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:
    
    # --- Configuration ---
    # Tolerances
    TOLERANCE_SYMMETRY = 0.02    # 2%
    TOLERANCE_FLOW = 0.05        # 5%
    THRESHOLD_ACTIVITY = 0.01    # Mbps
    
    # --- Data Preparation ---
    state = {}
    
    # Topology helpers
    router_to_interfaces = collections.defaultdict(list)
    verifiable_routers = set()
    
    for rid, if_ids in topology.items():
        router_to_interfaces[rid] = if_ids
        # A router is verifiable if we have data for all its interfaces
        if all(if_id in telemetry for if_id in if_ids):
            verifiable_routers.add(rid)

    # Initial State Load
    for if_id, data in telemetry.items():
        rx = float(data.get('rx_rate', 0.0))
        tx = float(data.get('tx_rate', 0.0))
        status = data.get('interface_status', 'down')
        
        state[if_id] = {
            'rx': rx,
            'tx': tx,
            'status': status,
            # Keep originals for reference
            'orig_rx': rx,
            'orig_tx': tx,
            'orig_status': status,
            # Topology
            'local_router': data.get('local_router'),
            'remote_router': data.get('remote_router'),
            'connected_to': data.get('connected_to'),
        }

    # --- Phase 1: Status Repair ---
    # Logic: Traffic Activity > Peer Status > Local Status
    status_conf_map = {}
    
    for if_id, s in state.items():
        orig_st = s['orig_status']
        peer_id = s['connected_to']
        
        # 1. Activity Detection
        has_traffic_local = (s['orig_rx'] > THRESHOLD_ACTIVITY) or (s['orig_tx'] > THRESHOLD_ACTIVITY)
        
        has_traffic_peer = False
        peer_st = 'unknown'
        if peer_id and peer_id in state:
            peer = state[peer_id]
            has_traffic_peer = (peer['orig_rx'] > THRESHOLD_ACTIVITY) or (peer['orig_tx'] > THRESHOLD_ACTIVITY)
            peer_st = peer['orig_status']
            
        # 2. Consensus Logic
        final_st = orig_st
        conf = 1.0
        
        if has_traffic_local or has_traffic_peer:
            # Presence of traffic is strong evidence of UP
            final_st = 'up'
            if orig_st == 'down':
                conf = 0.95 # Strong correction
        elif orig_st == 'up' and peer_st == 'down':
            # No traffic, peer says down. Likely down.
            final_st = 'down'
            conf = 0.85
        elif orig_st != peer_st:
            # Disagreement, no traffic. Safe default -> Down.
            final_st = 'down'
            conf = 0.75
            
        state[if_id]['status'] = final_st
        status_conf_map[if_id] = conf
        
        # Consistency: Down interfaces should have 0 rate
        if final_st == 'down':
            state[if_id]['rx'] = 0.0
            state[if_id]['tx'] = 0.0

    # --- Phase 2: Counter Repair (Iterative) ---
    
    def get_router_balance_error(rid, target_if=None, target_field=None, target_val=None):
        """
        Calculate flow conservation error for a router.
        Returns (absolute_diff, relative_error)
        """
        if rid not in verifiable_routers:
            return 0.0, None # Cannot verify
            
        sum_in = 0.0
        sum_out = 0.0
        
        for iface in router_to_interfaces[rid]:
            # Determine values to use
            if iface == target_if:
                r = target_val if target_field == 'rx' else state[iface]['rx']
                t = target_val if target_field == 'tx' else state[iface]['tx']
            else:
                r = state[iface]['rx']
                t = state[iface]['tx']
                
            sum_in += r
            sum_out += t
            
        diff = abs(sum_in - sum_out)
        denom = max(sum_in, sum_out, 1.0)
        return diff, diff / denom

    def get_residual(rid, target_if, target_field):
        """
        Calculate what value is needed at target_if to balance flow.
        """
        if rid not in verifiable_routers:
            return None
            
        # Sum everything else
        partial_in = 0.0
        partial_out = 0.0
        for iface in router_to_interfaces[rid]:
            if iface == target_if: continue
            partial_in += state[iface]['rx']
            partial_out += state[iface]['tx']
            
        # Flow Eq: Sum_In = Sum_Out
        # If target is Rx (input), In_Target + Partial_In = Partial_Out + Out_Target
        # In_Target = Partial_Out + Out_Target - Partial_In
        
        target_tx_curr = state[target_if]['tx']
        target_rx_curr = state[target_if]['rx']
        
        if target_field == 'rx':
            val = (partial_out + target_tx_curr) - partial_in
        else:
            # Out_Target = In_Target + Partial_In - Partial_Out
            val = (target_rx_curr + partial_in) - partial_out
            
        return max(0.0, val)

    # Iteration loop
    ITERATIONS = 5
    for i in range(ITERATIONS):
        
        for if_id, s in state.items():
            if s['status'] == 'down': continue
            
            peer_id = s['connected_to']
            if not peer_id or peer_id not in state: continue
            
            # Link: Local(Tx) -> Remote(Rx)
            curr_tx = s['tx']
            curr_rx = state[peer_id]['rx']
            
            # 1. Gather Candidates
            candidates = [curr_tx, curr_rx]
            
            # 2. Residual Candidates (Flow Conservation)
            rid_loc = s['local_router']
            rid_rem = s['remote_router']
            
            res_loc = get_residual(rid_loc, if_id, 'tx')
            res_rem = get_residual(rid_rem, peer_id, 'rx')
            
            if res_loc is not None: candidates.append(res_loc)
            if res_rem is not None: candidates.append(res_rem)
            
            # Filter candidates (remove negative or absurd values)
            candidates = [c for c in candidates if c >= 0]
            if not candidates: candidates = [0.0]
            
            # 3. Score Candidates
            best_val = curr_tx
            best_score = float('inf')
            
            # Deduplicate to save compute
            unique_candidates = sorted(list(set([round(c, 4) for c in candidates])))
            
            # If candidates are close, average them
            if unique_candidates[-1] - unique_candidates[0] < max(unique_candidates[-1]*TOLERANCE_SYMMETRY, THRESHOLD_ACTIVITY):
                best_val = sum(unique_candidates) / len(unique_candidates)
            else:
                for cand in unique_candidates:
                    score = 0.0
                    
                    # Local Fit
                    _, err_loc = get_router_balance_error(rid_loc, if_id, 'tx', cand)
                    if err_loc is None:
                        score += 0.5 # Neutral penalty for unverifiable
                    else:
                        score += min(err_loc * 10, 2.0) # Scale error
                        
                    # Remote Fit
                    _, err_rem = get_router_balance_error(rid_rem, peer_id, 'rx', cand)
                    if err_rem is None:
                        score += 0.5
                    else:
                        score += min(err_rem * 10, 2.0)
                        
                    # Heuristic: Penalize Zero if competing with high activity
                    # (Only if unverified or weakly verified)
                    if cand < THRESHOLD_ACTIVITY and max(unique_candidates) > 1.0:
                        is_verified = (err_loc is not None and err_loc < TOLERANCE_FLOW) or \
                                      (err_rem is not None and err_rem < TOLERANCE_FLOW)
                        if not is_verified:
                            score += 1.0
                            
                    if score < best_score:
                        best_score = score
                        best_val = cand
                    elif abs(score - best_score) < 1e-4:
                        # Tie-break: prefer non-zero
                        if cand > best_val:
                            best_val = cand
                            
            # Update State
            state[if_id]['tx'] = best_val
            state[peer_id]['rx'] = best_val

    # --- Phase 3: Confidence Calibration ---
    result = {}
    
    # Final error map
    final_errors = {}
    for rid in verifiable_routers:
        _, err = get_router_balance_error(rid)
        final_errors[rid] = err

    for if_id, s in state.items():
        orig_rx = s['orig_rx']
        final_rx = s['rx']
        orig_tx = s['orig_tx']
        final_tx = s['tx']
        
        # Helper to calculate confidence for a specific rate field
        def get_rate_conf(orig, final, field):
            # 1. Verification
            rid = s['local_router']
            is_loc_ver = False
            if rid in verifiable_routers:
                if final_errors[rid] < TOLERANCE_FLOW:
                    is_loc_ver = True
            
            rid_rem = s['remote_router']
            is_rem_ver = False
            if rid_rem in verifiable_routers:
                if final_errors[rid_rem] < TOLERANCE_FLOW:
                    is_rem_ver = True
                    
            # 2. Peer Consistency
            is_consistent = True
            peer_id = s['connected_to']
            if peer_id in state:
                peer_val = state[peer_id]['tx'] if field == 'rx' else state[peer_id]['rx']
                if abs(final - peer_val) > max(final, peer_val, 1.0) * TOLERANCE_SYMMETRY:
                    is_consistent = False
            
            # 3. Change Magnitude
            changed = abs(orig - final) > max(orig * 0.01, THRESHOLD_ACTIVITY)
            
            # --- Scoring Logic ---
            
            # Case A: Fully Verified (Both ends flow consistent)
            if is_loc_ver and is_rem_ver:
                return 1.0 
                
            # Case B: One-side Verified
            if is_loc_ver:
                return 0.98 if is_consistent else 0.95
            if is_rem_ver:
                return 0.98 if is_consistent else 0.90
                
            # Case C: Unverifiable but Consistent
            if is_consistent:
                if not changed: return 0.9
                # Smoothing
                if abs(orig - final) < max(orig, 1.0) * 0.1: return 0.9
                # Significant repair (blind trust in peer)
                return 0.75 
            
            # Case D: Inconsistent and Unverifiable
            if not changed: return 0.6
            return 0.5 # We guessed and it still doesn't match peer? Bad.
            
        rx_conf = get_rate_conf(orig_rx, final_rx, 'rx')
        tx_conf = get_rate_conf(orig_tx, final_tx, 'tx')
        st_conf = status_conf_map.get(if_id, 0.0)
        
        # DOWN Override
        if s['status'] == 'down':
             rx_conf = st_conf
             tx_conf = st_conf
             
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