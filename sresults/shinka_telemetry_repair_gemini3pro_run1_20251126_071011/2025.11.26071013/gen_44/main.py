# EVOLVE-BLOCK-START
"""
Network Telemetry Repair using Residual-Enhanced Consensus
and Continuous Confidence Scoring.

Key features:
- "Residual Synthesis": Infers missing traffic from flow conservation to propose candidates for dead links.
- Continuous Scoring: Uses gradient flow error for fine-grained candidate selection.
- Verification Tiers: confidence based on 4 tiers of verification (Dual, Single, Peer, Heuristic).
- Iterative propagation of flow constraints.
"""
from typing import Dict, Any, Tuple, List
import collections

def repair_network_telemetry(telemetry: Dict[str, Dict[str, Any]], 
                             topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:
    
    # --- Constants ---
    TOLERANCE = 0.02          # 2% symmetry tolerance
    FLOW_TOLERANCE = 0.05     # 5% flow conservation tolerance
    MIN_ACTIVITY = 0.01       # 10 Kbps threshold (Low to catch trickle traffic)
    
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
            
    # Initialize state
    for if_id, data in telemetry.items():
        state[if_id] = {
            'rx': float(data.get('rx_rate', 0.0)),
            'tx': float(data.get('tx_rate', 0.0)),
            'status': data.get('interface_status', 'down'),
            'connected_to': data.get('connected_to'),
            'local_router': data.get('local_router'),
            'remote_router': data.get('remote_router')
        }

    # --- 2. Status Repair ---
    status_conf_map = {}
    
    for if_id, s in state.items():
        orig_st = s['status']
        peer_id = s['connected_to']
        
        # Check Traffic (Local & Peer)
        loc_active = (s['rx'] > MIN_ACTIVITY) or (s['tx'] > MIN_ACTIVITY)
        
        peer_active = False
        peer_st = 'unknown'
        if peer_id and peer_id in state:
            peer_data = state[peer_id]
            peer_active = (peer_data['rx'] > MIN_ACTIVITY) or (peer_data['tx'] > MIN_ACTIVITY)
            peer_st = peer_data['status']
            
        # Repair Logic
        new_st = orig_st
        conf = 1.0
        
        if loc_active or peer_active:
            new_st = 'up'
            if orig_st == 'down': conf = 0.95
        elif orig_st == 'up' and peer_st == 'down':
            new_st = 'down'
            conf = 0.8
        elif orig_st != peer_st:
            # Conflict, no traffic -> Assume Down
            new_st = 'down'
            conf = 0.7
            
        state[if_id]['status'] = new_st
        status_conf_map[if_id] = conf
        
        if new_st == 'down':
            state[if_id]['rx'] = 0.0
            state[if_id]['tx'] = 0.0

    # --- 3. Rate Repair (Iterative Residual Consensus) ---
    
    def get_flow_error_for_val(rid, if_target, field, val):
        """Calc relative flow error if we force state[if_target][field] = val"""
        if rid not in verifiable_routers:
            return None # Cannot verify
            
        sum_in, sum_out = 0.0, 0.0
        for iface in router_map[rid]:
            r = val if (iface == if_target and field == 'rx') else state[iface]['rx']
            t = val if (iface == if_target and field == 'tx') else state[iface]['tx']
            sum_in += r
            sum_out += t
            
        diff = abs(sum_in - sum_out)
        denom = max(sum_in, sum_out, 1.0)
        return diff / denom

    def get_residual_val(rid, if_target, field):
        """Infer value required to balance flow at router"""
        if rid not in verifiable_routers:
            return None
        
        # Calculate residuals excluding the target interface
        # Flow Eq: sum_in = sum_out
        # if field='tx' (outgoing): tx = sum_in - sum_out_others
        # if field='rx' (incoming): rx = sum_out - sum_in_others
        
        sum_in_all = 0.0
        sum_out_all = 0.0
        
        # We need to iterate all interfaces.
        # Note: We rely on current state.
        for iface in router_map[rid]:
            # For the target interface, we only include the OTHER direction
            # If we are solving for TX, we include RX in Sum_IN.
            # We exclude TX from Sum_OUT (that's what we are solving for)
            
            if iface == if_target:
                sum_in_all += state[iface]['rx']
                # exclude tx
            else:
                sum_in_all += state[iface]['rx']
                sum_out_all += state[iface]['tx']
        
        if field == 'tx':
            val = sum_in_all - sum_out_all
        else: # rx
            val = sum_out_all - sum_in_all
            
        return max(0.0, val)

    for _ in range(3):
        for if_id, s in state.items():
            if s['status'] == 'down': continue
            
            peer_id = s['connected_to']
            if not peer_id or peer_id not in state: continue
            
            # Setup Candidates
            cand_tx = s['tx']
            cand_rx = state[peer_id]['rx']
            
            # Basic Candidates
            candidates = {cand_tx, cand_rx, (cand_tx + cand_rx)/2}
            
            # Residual Candidates (Synthesis)
            rid_loc = s['local_router']
            rid_rem = state[peer_id]['local_router'] # Remote router
            
            res_tx = get_residual_val(rid_loc, if_id, 'tx')
            if res_tx is not None: candidates.add(res_tx)
            
            res_rx = get_residual_val(rid_rem, peer_id, 'rx')
            if res_rx is not None: candidates.add(res_rx)
            
            # Score Candidates
            best_val = cand_tx
            best_score = float('inf')
            
            for val in candidates:
                # Calc Flow Errors
                err_loc = get_flow_error_for_val(rid_loc, if_id, 'tx', val)
                err_rem = get_flow_error_for_val(rid_rem, peer_id, 'rx', val)
                
                # Continuous Score
                # None (Unverifiable) => 0.05 penalty (neutral but slightly worse than perfect verification)
                s_loc = min(err_loc, 1.0) if err_loc is not None else 0.05
                s_rem = min(err_rem, 1.0) if err_rem is not None else 0.05
                
                score = s_loc + s_rem
                
                # Heuristic: Prefer Non-Zero
                # If val is 0 but the other side of the link originally reported high traffic,
                # we penalize the 0 val unless flow conservation STRICTLY demands it.
                if val < MIN_ACTIVITY:
                    # If originally one side was active, 0 is suspicious
                    if cand_tx > MIN_ACTIVITY or cand_rx > MIN_ACTIVITY:
                        score += 0.2
                
                if score < best_score:
                    best_score = score
                    best_val = val
                elif score == best_score:
                    # Tie-breaking
                    # Prefer values closer to original measurements if scores are identical
                    # or prefer non-zero
                    if val > best_val: best_val = val
            
            state[if_id]['tx'] = best_val
            state[peer_id]['rx'] = best_val

    # --- 4. Confidence Calibration ---
    result = {}
    
    # Final Flow Errors map
    router_errors = {}
    for rid in verifiable_routers:
        sin = sum(state[i]['rx'] for i in router_map[rid])
        sout = sum(state[i]['tx'] for i in router_map[rid])
        router_errors[rid] = abs(sin - sout) / max(sin, sout, 1.0)
        
    for if_id, data in telemetry.items():
        # Retrieve final data
        final_rx = state[if_id]['rx']
        final_tx = state[if_id]['tx']
        final_st = state[if_id]['status']
        
        orig_rx = data.get('rx_rate', 0.0)
        orig_tx = data.get('tx_rate', 0.0)
        orig_st = data.get('interface_status', 'down')
        
        rid = data.get('local_router')
        peer_id = data.get('connected_to')
        remote_rid = data.get('remote_router')
        
        def get_conf(orig, final, field):
            # Verification Flags
            loc_err = router_errors.get(rid)
            loc_ok = (loc_err is not None and loc_err < FLOW_TOLERANCE)
            
            rem_ok = False
            if remote_rid and remote_rid in router_errors:
                if router_errors[remote_rid] < FLOW_TOLERANCE:
                    rem_ok = True
                    
            # Peer Consistency
            peer_consistent = True
            if peer_id and peer_id in state:
                peer_val = state[peer_id]['tx'] if field == 'rx' else state[peer_id]['rx']
                if abs(final - peer_val) > max(final, 1.0) * TOLERANCE:
                    peer_consistent = False
            
            changed = abs(orig - final) > 0.001
            
            # Tier 1: Dual Verification
            if loc_ok and rem_ok: return 0.99
            
            # Tier 2: Single Verification
            if loc_ok: return 0.95
            if rem_ok: return 0.92
            
            # Tier 3: Unverifiable but Consistent
            if not changed:
                # We kept original
                if not peer_consistent: return 0.65 # Kept it, but peer disagrees
                # If unverifiable (edge), but consistent with peer
                return 0.9
            else:
                # We changed it
                # Smoothing
                if orig > MIN_ACTIVITY and abs(orig - final) / orig < 0.05:
                    return 0.95
                
                # Heuristic Repair (0 -> X)
                if orig < MIN_ACTIVITY and final > MIN_ACTIVITY:
                    return 0.85
                    
                # Consistent with peer now
                if peer_consistent: return 0.75
                
                return 0.5
                
        rx_c = get_conf(orig_rx, final_rx, 'rx')
        tx_c = get_conf(orig_tx, final_tx, 'tx')
        st_c = status_conf_map.get(if_id, 1.0)
        
        # Sanity: Down = 0
        if final_st == 'down' and (final_rx > 1.0 or final_tx > 1.0):
             rx_c, tx_c, st_c = 0.0, 0.0, 0.0

        result[if_id] = {
            'rx_rate': (orig_rx, final_rx, rx_c),
            'tx_rate': (orig_tx, final_tx, tx_c),
            'interface_status': (orig_st, final_st, st_c),
            'connected_to': peer_id,
            'local_router': rid,
            'remote_router': remote_rid
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