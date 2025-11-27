# EVOLVE-BLOCK-START
"""
Hybrid Flow Consensus Network Telemetry Repair
Combines "Residual Synthesis" for candidate generation (recovering lost data from flow constraints)
with "Broken Router" confidence calibration to prevent overconfidence in messy topologies.
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
        """Calculates relative flow imbalance. Allows hypothetical substitution."""
        if rid not in verifiable_routers:
            return None
            
        sum_rx, sum_tx = 0.0, 0.0
        for iface in router_map[rid]:
            r = value if (iface == if_target and field == 'rx') else state[iface]['rx']
            t = value if (iface == if_target and field == 'tx') else state[iface]['tx']
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
        
        # Activity checks
        local_traffic = (s['orig_rx'] > MIN_ACTIVITY) or (s['orig_tx'] > MIN_ACTIVITY)
        peer_status = 'unknown'
        peer_traffic = False
        
        if peer_id and peer_id in state:
            p = state[peer_id]
            peer_traffic = (p['orig_rx'] > MIN_ACTIVITY) or (p['orig_tx'] > MIN_ACTIVITY)
            peer_status = p['orig_status']
            
        # Decision Logic
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

    # --- 3. Rate Repair (Hybrid Consensus) ---
    for _ in range(3): 
        for if_id, s in state.items():
            if s['status'] == 'down': continue
            
            peer_id = s['connected_to']
            if not peer_id or peer_id not in state: continue
            
            # Setup
            val_tx = s['tx']
            val_rx = state[peer_id]['rx']
            rid_local = s['local_router']
            rid_remote = s['remote_router']
            
            # 1. Base Candidates
            candidates = [val_tx, val_rx]
            
            # 2. Residual Synthesis (Flow Conservation)
            # Try to calculate what TX *should* be to balance the local router
            if rid_local in verifiable_routers:
                # TX_target = Sum(In) - Sum(Out_others)
                s_in, s_out_others = 0.0, 0.0
                for iface in router_map[rid_local]:
                    s_in += state[iface]['rx']
                    if iface != if_id: s_out_others += state[iface]['tx']
                
                synth_tx = s_in - s_out_others
                if synth_tx > -1e-3: candidates.append(max(0.0, synth_tx))

            # Try to calculate what RX *should* be to balance the remote router
            if rid_remote in verifiable_routers:
                # RX_target = Sum(Out) - Sum(In_others)
                s_out, s_in_others = 0.0, 0.0
                for iface in router_map[rid_remote]:
                    s_out += state[iface]['tx']
                    if iface != peer_id: s_in_others += state[iface]['rx']
                    
                synth_rx = s_out - s_in_others
                if synth_rx > -1e-3: candidates.append(max(0.0, synth_rx))

            # 3. Deduplicate
            unique_cands = []
            for c in candidates:
                if not any(abs(c - u) < 1e-4 for u in unique_cands):
                    unique_cands.append(c)
            
            # 4. Scoring
            best_val = val_tx
            best_score = float('inf')
            
            for cand in unique_cands:
                # Error check
                err_local = get_flow_error(rid_local, if_id, 'tx', cand)
                err_remote = get_flow_error(rid_remote, peer_id, 'rx', cand)
                
                # Continuous scoring: Prefer Verified Low Error
                # None (unverifiable) gets 0.05 penalty (neutral but worse than verified 0.0)
                score_local = min(err_local, 1.0) if err_local is not None else 0.05
                score_remote = min(err_remote, 1.0) if err_remote is not None else 0.05
                
                score = score_local + score_remote
                
                # Heuristic: Penalize zero if active alternatives exist
                if cand < MIN_ACTIVITY and max(unique_cands) > MIN_ACTIVITY:
                    score += 0.5
                    
                if score < best_score:
                    best_score = score
                    best_val = cand
            
            # Update
            state[if_id]['tx'] = best_val
            state[peer_id]['rx'] = best_val

    # --- 4. Confidence Calibration ---
    result = {}
    
    # Final error context
    final_errors = {rid: get_flow_error(rid) for rid in verifiable_routers}

    for if_id, s in state.items():
        orig_rx, final_rx = s['orig_rx'], s['rx']
        orig_tx, final_tx = s['orig_tx'], s['tx']
        
        rid = s['local_router']
        peer_id = s['connected_to']
        
        def calculate_confidence(orig, final, field):
            # Verification
            local_err = final_errors.get(rid)
            local_verified = (local_err is not None and local_err < FLOW_TOLERANCE)
            
            rem_rid = s['remote_router']
            remote_verified = False
            if rem_rid in final_errors and final_errors[rem_rid] < FLOW_TOLERANCE:
                remote_verified = True
                
            # Peer Consistency
            peer_consistent = True
            if peer_id in state:
                peer_val = state[peer_id]['tx'] if field == 'rx' else state[peer_id]['rx']
                if abs(final - peer_val) > max(final, peer_val, 1.0) * TOLERANCE:
                    peer_consistent = False
            
            # Change Analysis
            changed = abs(orig - final) > max(orig * 0.001, 0.001)
            is_smoothing = changed and (abs(orig - final) < max(orig * 0.05, 0.1))
            
            # Scoring
            if not changed:
                if local_verified and remote_verified: return 1.0
                if local_verified: return 0.98
                if not peer_consistent: return 0.7
                
                # "Broken Router" Penalty: Unverified locally + High Error = unreliable
                if local_err is not None and local_err >= FLOW_TOLERANCE:
                    return 0.6
                return 0.9
                
            if is_smoothing: return 0.95
            
            if local_verified and remote_verified: return 0.98
            if local_verified: return 0.95
            if remote_verified: return 0.90
            
            # Unverified heuristics
            if orig < MIN_ACTIVITY and final > MIN_ACTIVITY: return 0.85
            return 0.6
            
        rx_conf = calculate_confidence(orig_rx, final_rx, 'rx')
        tx_conf = calculate_confidence(orig_tx, final_tx, 'tx')
        st_conf = status_confidence.get(if_id, 1.0)
        
        # Down sanity check
        if s['status'] == 'down' and (final_rx > MIN_ACTIVITY or final_tx > MIN_ACTIVITY):
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