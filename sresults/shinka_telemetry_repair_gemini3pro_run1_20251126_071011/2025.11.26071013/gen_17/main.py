# EVOLVE-BLOCK-START
"""
Network Telemetry Repair with Flow Synthesis and Continuous Confidence Calibration.

Key features:
- Candidate Synthesis: Generates hypothetical rates that satisfy flow conservation to fix "double-dead" links.
- Granular Scoring: Continuous error functions for precise candidate selection.
- Continuous Confidence: Confidence scores decay based on residual flow errors.
"""
from typing import Dict, Any, Tuple, List
import collections

def repair_network_telemetry(telemetry: Dict[str, Dict[str, Any]], 
                             topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:
    
    # --- Constants ---
    TOLERANCE = 0.02          # 2% symmetry tolerance
    FLOW_TOLERANCE = 0.05     # 5% flow conservation tolerance
    MIN_ACTIVITY = 0.05       # Mbps threshold for active traffic
    
    # --- 1. Initialization ---
    state = {}
    router_map = collections.defaultdict(list)
    verifiable_routers = set()
    
    # Identify verifiable routers (all interfaces monitored)
    for rid, if_list in topology.items():
        router_map[rid] = if_list
        if all(if_id in telemetry for if_id in if_list):
            verifiable_routers.add(rid)
            
    # Initialize State
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

    # --- Helper: Flow Error & Balance Calculation ---
    def get_router_stats(rid, if_target=None, field=None, value=None):
        """
        Returns (flow_error, required_value_for_target) for a router.
        If if_target is None, returns (current_error, None).
        """
        if rid not in verifiable_routers:
            return None, None
            
        sum_rx = 0.0
        sum_tx = 0.0
        
        # Calculate sums (using spec value for target)
        for iface in router_map[rid]:
            if iface == if_target:
                r = value if field == 'rx' else state[iface]['rx']
                t = value if field == 'tx' else state[iface]['tx']
            else:
                r = state[iface]['rx']
                t = state[iface]['tx']
            
            sum_rx += r
            sum_tx += t
            
        # Calculate Error
        diff = abs(sum_rx - sum_tx)
        denom = max(sum_rx, sum_tx, 1.0)
        error = diff / denom
        
        # Calculate Required Value for Target (Synthesis)
        required = None
        if if_target is not None:
            current_target_r = value if field == 'rx' else state[if_target]['rx']
            current_target_t = value if field == 'tx' else state[if_target]['tx']
            
            if field == 'rx':
                # Sum_RX should equal Sum_TX
                # (Other_RX + Required) = Sum_TX
                other_rx = sum_rx - current_target_r
                required = sum_tx - other_rx
            else: # field == 'tx'
                # Sum_TX should equal Sum_RX
                # (Other_TX + Required) = Sum_RX
                other_tx = sum_tx - current_target_t
                required = sum_rx - other_tx
             
        return error, required

    # --- 2. Status Repair ---
    status_confidence = {}
    
    for if_id, s in state.items():
        orig = s['orig_status']
        peer_id = s['connected_to']
        
        local_active = (s['orig_rx'] > MIN_ACTIVITY) or (s['orig_tx'] > MIN_ACTIVITY)
        peer_active = False
        peer_status = 'unknown'
        
        if peer_id and peer_id in state:
            p = state[peer_id]
            peer_active = (p['orig_rx'] > MIN_ACTIVITY) or (p['orig_tx'] > MIN_ACTIVITY)
            peer_status = p['orig_status']
            
        final_st = orig
        conf = 1.0
        
        if local_active or peer_active:
            final_st = 'up'
            if orig == 'down': conf = 0.95
        elif orig == 'up' and peer_status == 'down':
            final_st = 'down'
            conf = 0.8
        elif orig != peer_status:
            final_st = 'down'
            conf = 0.7
            
        state[if_id]['status'] = final_st
        status_confidence[if_id] = conf
        
        if final_st == 'down':
            state[if_id]['rx'] = 0.0
            state[if_id]['tx'] = 0.0

    # --- 3. Rate Repair (Iterative) ---
    for i in range(4): # 4 passes to allow synthesis propagation
        for if_id, s in state.items():
            if s['status'] == 'down': continue
            
            peer_id = s['connected_to']
            if not peer_id or peer_id not in state: continue
            
            # Goal: Consensus on Rate X (Tx_local -> Rx_remote)
            val_tx = s['tx']
            val_rx = state[peer_id]['rx']
            
            # Generate Candidates
            candidates = [val_tx, val_rx]
            
            # Synthesis: What does the Local Router need this TX to be?
            _, req_tx = get_router_stats(s['local_router'], if_id, 'tx', val_tx)
            if req_tx is not None and req_tx > MIN_ACTIVITY:
                candidates.append(req_tx)
                
            # Synthesis: What does the Remote Router need this RX to be?
            _, req_rx = get_router_stats(s['remote_router'], peer_id, 'rx', val_rx)
            if req_rx is not None and req_rx > MIN_ACTIVITY:
                candidates.append(req_rx)
                
            # Filter Candidates (Deduplicate, positive, round for stability)
            candidates = sorted(list(set([round(c, 4) for c in candidates if c >= 0])))
            if not candidates: candidates = [0.0]
            
            # Scoring
            best_val = val_tx
            best_score = float('inf')
            
            for c in candidates:
                # Calculate costs (flow errors at both ends)
                err_loc, _ = get_router_stats(s['local_router'], if_id, 'tx', c)
                err_rem, _ = get_router_stats(s['remote_router'], peer_id, 'rx', c)
                
                def get_cost(err):
                    if err is None: return 0.02 # Unverifiable -> neutral low cost
                    if err < FLOW_TOLERANCE: return err # Valid: cost is the actual error (0.00-0.05)
                    return 0.5 + err # Invalid: Heavy penalty + error magnitude
                    
                score = get_cost(err_loc) + get_cost(err_rem)
                
                # Heuristic Penalties
                # Penalty for choosing 0 if the peer/system suggests activity
                # We identify "system suggests activity" if any candidate is significantly large
                if c < MIN_ACTIVITY and max(candidates) > MIN_ACTIVITY:
                     score += 0.25
                
                if score < best_score:
                    best_score = score
                    best_val = c
            
            # Apply
            state[if_id]['tx'] = best_val
            state[peer_id]['rx'] = best_val

    # --- 4. Confidence Calibration ---
    result = {}
    
    # Final Flow Errors map
    final_errors = {}
    for rid in verifiable_routers:
        err, _ = get_router_stats(rid)
        final_errors[rid] = err
        
    for if_id, s in state.items():
        rid = s['local_router']
        peer_id = s['connected_to']
        rem_rid = s['remote_router']
        
        rx_val = s['rx']
        tx_val = s['tx']
        
        # Helper for individual rate confidence
        def calibrate(orig, final, field):
            # Base Confidence
            conf = 0.9
            
            changed = abs(orig - final) > 0.01
            smoothed = changed and abs(orig - final) < max(orig * 0.1, 0.5)
            
            # Verification Logic
            loc_err = final_errors.get(rid)
            rem_err = final_errors.get(rem_rid)
            
            loc_ver = (loc_err is not None and loc_err < FLOW_TOLERANCE)
            rem_ver = (rem_err is not None and rem_err < FLOW_TOLERANCE)
            
            # Peer Consistency
            peer_ver = True
            if peer_id in state:
                p_val = state[peer_id]['tx'] if field == 'rx' else state[peer_id]['rx']
                if abs(final - p_val) > max(final, p_val, 1.0) * TOLERANCE:
                    peer_ver = False
            
            if not changed:
                if loc_ver and rem_ver: conf = 1.0
                elif loc_ver: conf = 0.98
                elif not peer_ver: conf = 0.7
                else: conf = 0.95
            elif smoothed:
                conf = 0.95
            else:
                # Changed significantly
                if loc_ver and rem_ver: conf = 0.99
                elif loc_ver: conf = 0.95
                elif rem_ver: conf = 0.90
                # Dead counter repair (0 -> High)
                elif orig < MIN_ACTIVITY and final > MIN_ACTIVITY: conf = 0.85
                else: conf = 0.6
            
            # Penalty for residual errors (Continuous Calibration)
            # If the router is still imbalanced, reduce confidence in its interfaces
            if loc_err is not None and loc_err > FLOW_TOLERANCE:
                # Decay factor: 1.0 at 5% error -> 0.5 at 100% error
                # This ensures we don't express high confidence in a broken router's data
                decay = min(0.5, loc_err)
                conf *= (1.0 - decay)
                
            return float(conf)

        rx_conf = calibrate(s['orig_rx'], rx_val, 'rx')
        tx_conf = calibrate(s['orig_tx'], tx_val, 'tx')
        st_conf = status_confidence.get(if_id, 1.0)
        
        # Override for Down
        if s['status'] == 'down' and (rx_val > 0.1 or tx_val > 0.1):
             rx_conf = 0.0
             tx_conf = 0.0

        result[if_id] = {
            'rx_rate': (s['orig_rx'], rx_val, rx_conf),
            'tx_rate': (s['orig_tx'], tx_val, tx_conf),
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

