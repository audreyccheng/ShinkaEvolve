# EVOLVE-BLOCK-START
"""
Consensus-based Network Telemetry Repair V3
Combines iterative constraint satisfaction with robust confidence scoring.
Improvements:
- Full candidate generation including flow residuals for better outlier rejection.
- SNR-based confidence scaling for dead counter repairs.
- Explicit categorization of router states (Verified, Unverifiable, Broken).
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

    # --- Helpers ---
    def get_flow_stats(rid, if_target=None, field=None, value=None):
        """Returns (sum_rx, sum_tx) for a router, optionally substituting one value."""
        if rid not in verifiable_routers:
            return None, None
            
        sum_rx, sum_tx = 0.0, 0.0
        for iface in router_map[rid]:
            if iface == if_target:
                r = value if field == 'rx' else state[iface]['rx']
                t = value if field == 'tx' else state[iface]['tx']
            else:
                r = state[iface]['rx']
                t = state[iface]['tx']
            sum_rx += r
            sum_tx += t
        return sum_rx, sum_tx

    def get_flow_error(rid, if_target=None, field=None, value=None):
        sum_rx, sum_tx = get_flow_stats(rid, if_target, field, value)
        if sum_rx is None: return None
        diff = abs(sum_rx - sum_tx)
        denom = max(sum_rx, sum_tx, 1.0)
        return diff / denom

    def get_residual(rid, if_target, field):
        """Calculates the value needed for perfect flow conservation."""
        if rid not in verifiable_routers: return None
        
        sum_rx, sum_tx = 0.0, 0.0
        for iface in router_map[rid]:
            if iface == if_target: continue # Skip self
            sum_rx += state[iface]['rx']
            sum_tx += state[iface]['tx']
            
        # We want Total_Rx = Total_Tx
        # If target is Rx: Rx_Target + Other_Rx = Total_Tx -> Rx_Target = Total_Tx - Other_Rx
        # If target is Tx: Total_Rx = Tx_Target + Other_Tx -> Tx_Target = Total_Rx - Other_Tx
        
        if field == 'rx':
            val = sum_tx - sum_rx
        else:
            val = sum_rx - sum_tx
            
        return max(0.0, val)

    # --- 2. Status Repair ---
    status_confidence = {}
    for if_id, s in state.items():
        orig = s['orig_status']
        peer_id = s['connected_to']
        
        local_traffic = (s['orig_rx'] > MIN_ACTIVITY) or (s['orig_tx'] > MIN_ACTIVITY)
        peer_traffic = False
        peer_status = 'unknown'
        
        if peer_id and peer_id in state:
            peer = state[peer_id]
            peer_traffic = (peer['orig_rx'] > MIN_ACTIVITY) or (peer['orig_tx'] > MIN_ACTIVITY)
            peer_status = peer['orig_status']
            
        final_status = orig
        conf = 1.0
        
        if local_traffic or peer_traffic:
            final_status = 'up'
            if orig == 'down': conf = 0.95
        elif orig == 'up' and peer_status == 'down':
            final_status = 'down'
            conf = 0.8
        elif orig != peer_status:
            final_status = 'down'
            conf = 0.7
            
        state[if_id]['status'] = final_status
        status_confidence[if_id] = conf
        
        if final_status == 'down':
            state[if_id]['rx'] = 0.0
            state[if_id]['tx'] = 0.0

    # --- 3. Rate Repair (Consensus) ---
    for _ in range(4): # 4 iterations to allow flow corrections to propagate
        for if_id, s in state.items():
            if s['status'] == 'down': continue
            
            peer_id = s['connected_to']
            if not peer_id or peer_id not in state: continue
            
            val_tx = s['tx']
            val_rx = state[peer_id]['rx']
            
            # Simple average smoothing if close
            diff = abs(val_tx - val_rx)
            avg = (val_tx + val_rx) / 2.0
            
            if diff < max(avg * TOLERANCE, MIN_ACTIVITY):
                new_val = avg
            else:
                # Disagreement - Use flow constraints to vote
                rid_local = s['local_router']
                rid_remote = s['remote_router']
                
                candidates = [val_tx, val_rx]
                
                # Add residuals as candidates (The "Third Opinion")
                res_tx = get_residual(rid_local, if_id, 'tx')
                if res_tx is not None: candidates.append(res_tx)
                
                res_rx = get_residual(rid_remote, peer_id, 'rx')
                if res_rx is not None: candidates.append(res_rx)
                
                # Deduplicate candidates with epsilon
                unique_cands = []
                for c in candidates:
                    if not any(abs(c - x) < 1e-4 for x in unique_cands):
                        unique_cands.append(c)
                        
                # Score candidates
                best_score = float('inf')
                best_cands = []
                
                for cand in unique_cands:
                    err_loc = get_flow_error(rid_local, if_id, 'tx', cand)
                    err_rem = get_flow_error(rid_remote, peer_id, 'rx', cand)
                    
                    # Scoring Logic
                    # Verified (err < TOL) -> 0 cost
                    # Unverified (err is None) -> 0.2 cost (slight penalty vs perfect verification)
                    # Broken (err > TOL) -> 1.0 + err cost (major penalty)
                    
                    def get_cost(err):
                        if err is None: return 0.2 
                        if err < FLOW_TOLERANCE: return 0.0 
                        return 1.0 + min(err, 1.0) 
                        
                    score = get_cost(err_loc) + get_cost(err_rem)
                    
                    # Heuristic: Penalty for Zero if competing with valid signal
                    # If candidate is < MIN, but max candidate is > MIN, penalize 0
                    if cand < MIN_ACTIVITY and max(unique_cands) > MIN_ACTIVITY:
                        score += 0.4
                        
                    if score < best_score:
                        best_score = score
                        best_cands = [cand]
                    elif abs(score - best_score) < 1e-4:
                        best_cands.append(cand)
                
                new_val = sum(best_cands) / len(best_cands)
                
            state[if_id]['tx'] = new_val
            state[peer_id]['rx'] = new_val

    # --- 4. Confidence Calibration ---
    result = {}
    
    # Calculate final errors to identify broken routers
    router_errors = {rid: get_flow_error(rid) for rid in verifiable_routers}
    
    for if_id, s in state.items():
        orig_rx, final_rx = s['orig_rx'], s['rx']
        orig_tx, final_tx = s['orig_tx'], s['tx']
        
        rid_local = s['local_router']
        rid_remote = s['remote_router']
        
        def calibrate(orig, final, field):
            # 1. State Classification
            # Check Local Verification
            loc_err = router_errors.get(rid_local)
            local_verified = (loc_err is not None and loc_err < FLOW_TOLERANCE)
            local_broken = (loc_err is not None and loc_err >= FLOW_TOLERANCE)
            
            # Check Remote Verification
            rem_err = router_errors.get(rid_remote)
            remote_verified = (rem_err is not None and rem_err < FLOW_TOLERANCE)
            
            # Check Change
            changed = abs(orig - final) > max(orig * 0.001, 0.001)
            is_smoothing = changed and (abs(orig - final) < max(orig * 0.05, 0.1))
            
            # Check Peer Consistency
            peer_id = s['connected_to']
            peer_consistent = True
            if peer_id in state:
                peer_val = state[peer_id]['tx'] if field == 'rx' else state[peer_id]['rx']
                if abs(final - peer_val) > max(final, peer_val, 1.0) * TOLERANCE:
                    peer_consistent = False

            # 2. Logic
            
            if not changed:
                if local_verified and remote_verified: return 1.0
                if local_verified: return 0.99
                if not peer_consistent: return 0.7 
                return 0.95
                
            if is_smoothing:
                return 0.95
                
            # Significant Changes
            if local_verified and remote_verified: return 0.99
            if local_verified: return 0.95
            if remote_verified: return 0.90
            
            # Heuristics (Unverified)
            if orig < MIN_ACTIVITY and final > MIN_ACTIVITY:
                # Dead counter repair
                # SNR logic: 0.75 base + up to 0.2 depending on magnitude
                # Cap at 10 Mbps for max confidence bonus
                snr_bonus = 0.2 * min(1.0, final / 10.0)
                base = 0.75
                
                # If connected to a broken router, reduce confidence
                if local_broken or (rem_err is not None and rem_err >= FLOW_TOLERANCE):
                    base = 0.6
                    
                return base + snr_bonus
                
            # Trust Peer (Agreement without verification)
            if local_broken: return 0.5 # Local is broken, so we are guessing
            return 0.6 # Low confidence default
            
        rx_conf = calibrate(orig_rx, final_rx, 'rx')
        tx_conf = calibrate(orig_tx, final_tx, 'tx')
        st_conf = status_confidence.get(if_id, 1.0)
        
        # Down sanity
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
