# EVOLVE-BLOCK-START
"""
Hybrid Consensus Network Telemetry Repair
Combines iterative constraint satisfaction with global consistency scoring.
Features:
- Status Repair: Conservative consensus (Activity > Peer Status > Local Status)
- Rate Repair: Global candidate pooling (Local, Remote, Residuals) with flow error minimization
- Confidence: Verification-based scoring with penalties for unbalanced routers
"""
from typing import Dict, Any, Tuple, List
import collections

def repair_network_telemetry(telemetry: Dict[str, Dict[str, Any]], 
                             topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:
    
    # --- Constants ---
    TOLERANCE = 0.02          # 2% symmetry tolerance
    MIN_ACTIVITY = 0.01       # Mbps threshold for active
    FLOW_TOLERANCE = 0.05     # 5% flow tolerance
    ITERATIONS = 5            # Number of consensus passes

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
            'orig_rx': float(data.get('rx_rate', 0.0)),
            'orig_tx': float(data.get('tx_rate', 0.0)),
            'orig_status': data.get('interface_status', 'down'),
            'connected_to': data.get('connected_to'),
            'local_router': data.get('local_router'),
            'remote_router': data.get('remote_router')
        }

    # --- 2. Status Repair ---
    status_conf = {}
    
    for if_id, s in state.items():
        orig_s = s['orig_status']
        peer_id = s['connected_to']
        
        # Check Activity
        has_activity = (s['orig_rx'] > MIN_ACTIVITY) or (s['orig_tx'] > MIN_ACTIVITY)
        
        peer_active = False
        peer_s = 'unknown'
        if peer_id and peer_id in state:
            p = state[peer_id]
            peer_active = (p['orig_rx'] > MIN_ACTIVITY) or (p['orig_tx'] > MIN_ACTIVITY)
            peer_s = p['orig_status']
            
        # Repair Logic
        final_s = orig_s
        conf = 1.0
        
        if has_activity or peer_active:
            final_s = 'up'
            if orig_s == 'down':
                conf = 0.95
        elif orig_s == 'up' and peer_s == 'down':
            final_s = 'down'
            conf = 0.8
        elif orig_s != peer_s:
            final_s = 'down'
            conf = 0.7
            
        state[if_id]['status'] = final_s
        status_conf[if_id] = conf
        
        # Enforce Down = 0.0
        if final_s == 'down':
            state[if_id]['rx'] = 0.0
            state[if_id]['tx'] = 0.0

    # --- 3. Rate Repair (Iterative) ---
    
    def get_flow_residual(rid, if_exclude, field_exclude):
        """Calculates the value needed at if_exclude to balance the router."""
        if rid not in verifiable_routers: return None
        
        sum_rx, sum_tx = 0.0, 0.0
        for iface in router_map[rid]:
            if iface == if_exclude:
                # Add the OTHER component (known)
                if field_exclude == 'rx': sum_tx += state[iface]['tx']
                else: sum_rx += state[iface]['rx']
            else:
                sum_rx += state[iface]['rx']
                sum_tx += state[iface]['tx']
        
        # We want sum_rx == sum_tx
        if field_exclude == 'rx': return max(0.0, sum_tx - sum_rx)
        else: return max(0.0, sum_rx - sum_tx)

    def get_flow_error(rid, if_target, field, val):
        """Calculates relative flow error if if_target takes value val."""
        if rid not in verifiable_routers: return None
        
        sum_rx, sum_tx = 0.0, 0.0
        for iface in router_map[rid]:
            r, t = state[iface]['rx'], state[iface]['tx']
            if iface == if_target:
                if field == 'rx': r = val
                else: t = val
            sum_rx += r
            sum_tx += t
            
        return abs(sum_rx - sum_tx) / max(sum_rx, sum_tx, 1.0)

    for _ in range(ITERATIONS):
        processed = set()
        
        for if_id, s in state.items():
            if s['status'] == 'down': continue
            
            peer_id = s['connected_to']
            if not peer_id or peer_id not in state: continue
            
            # Process link once (handles both directions)
            link_key = tuple(sorted([if_id, peer_id]))
            if link_key in processed: continue
            processed.add(link_key)
            
            # --- Direction 1: Local TX -> Remote RX ---
            rid_loc = s['local_router']
            rid_rem = state[peer_id]['local_router']
            
            val_tx = s['tx']
            val_rx = state[peer_id]['rx']
            
            # Candidate Pool
            cands = {val_tx, val_rx, (val_tx + val_rx)/2.0}
            res_loc = get_flow_residual(rid_loc, if_id, 'tx')
            if res_loc is not None: cands.add(res_loc)
            res_rem = get_flow_residual(rid_rem, peer_id, 'rx')
            if res_rem is not None: cands.add(res_rem)
            
            # Selection
            best_val = val_tx
            best_score = float('inf')
            
            # Dedupe for efficiency
            unique_cands = []
            for c in cands:
                if not any(abs(c - x) < 1e-5 for x in unique_cands):
                    unique_cands.append(c)
            
            for c in unique_cands:
                e1 = get_flow_error(rid_loc, if_id, 'tx', c)
                e2 = get_flow_error(rid_rem, peer_id, 'rx', c)
                
                # If error is None (unverifiable), use small constant cost
                score = (e1 if e1 is not None else 0.02) + (e2 if e2 is not None else 0.02)
                
                # Penalty for Zero if valid non-zero exists
                if c < MIN_ACTIVITY and max(unique_cands) > MIN_ACTIVITY:
                    score += 0.5
                
                if score < best_score:
                    best_score = score
                    best_val = c
            
            state[if_id]['tx'] = best_val
            state[peer_id]['rx'] = best_val
            
            # --- Direction 2: Remote TX -> Local RX ---
            # Swap roles: Peer is TX source, Local is RX sink
            val_tx_2 = state[peer_id]['tx']
            val_rx_2 = s['rx']
            
            cands_2 = {val_tx_2, val_rx_2, (val_tx_2 + val_rx_2)/2.0}
            res_loc_2 = get_flow_residual(rid_rem, peer_id, 'tx')
            if res_loc_2 is not None: cands_2.add(res_loc_2)
            res_rem_2 = get_flow_residual(rid_loc, if_id, 'rx')
            if res_rem_2 is not None: cands_2.add(res_rem_2)
            
            best_val_2 = val_tx_2
            best_score_2 = float('inf')
            
            unique_cands_2 = []
            for c in cands_2:
                if not any(abs(c - x) < 1e-5 for x in unique_cands_2):
                    unique_cands_2.append(c)

            for c in unique_cands_2:
                e1 = get_flow_error(rid_rem, peer_id, 'tx', c)
                e2 = get_flow_error(rid_loc, if_id, 'rx', c)
                
                score = (e1 if e1 is not None else 0.02) + (e2 if e2 is not None else 0.02)
                
                if c < MIN_ACTIVITY and max(unique_cands_2) > MIN_ACTIVITY:
                    score += 0.5
                    
                if score < best_score_2:
                    best_score_2 = score
                    best_val_2 = c
            
            state[peer_id]['tx'] = best_val_2
            state[if_id]['rx'] = best_val_2

    # --- 4. Confidence Calibration ---
    result = {}
    
    # Calculate final errors
    final_errs = {}
    for rid in verifiable_routers:
        sum_rx = sum(state[i]['rx'] for i in router_map[rid])
        sum_tx = sum(state[i]['tx'] for i in router_map[rid])
        final_errs[rid] = abs(sum_rx - sum_tx) / max(sum_rx, sum_tx, 1.0)
        
    for if_id, data in telemetry.items():
        s = state[if_id]
        
        def get_conf(orig, final, field):
            # Verification Status
            rid_loc = s['local_router']
            loc_ok = (rid_loc in final_errs and final_errs[rid_loc] < FLOW_TOLERANCE)
            
            rid_rem = s['remote_router']
            rem_ok = (rid_rem in final_errs and final_errs[rid_rem] < FLOW_TOLERANCE)
            
            # Peer Consistency
            peer_id = s['connected_to']
            peer_consistent = True
            if peer_id in state:
                peer_val = state[peer_id]['tx'] if field == 'rx' else state[peer_id]['rx']
                if abs(final - peer_val) > max(final, peer_val, 1.0) * TOLERANCE:
                    peer_consistent = False
            
            # Base Confidence
            c = 0.6
            if loc_ok and rem_ok: c = 0.99
            elif loc_ok: c = 0.95
            elif rem_ok: c = 0.90
            elif peer_consistent: c = 0.85
            
            # Change Logic
            changed = abs(orig - final) > 0.001
            
            if not changed:
                # Retained value is generally trusted unless flow is broken
                c = max(c, 0.9)
                # If local router is broken, cap confidence
                if rid_loc in final_errs and not loc_ok:
                    c = min(c, 0.7)
            else:
                # Changed value
                if not (loc_ok or rem_ok):
                    # Unverified change (Heuristic)
                    if orig < MIN_ACTIVITY and final > MIN_ACTIVITY: c = 0.8
                    else: c = 0.6
            
            return c

        rx_c = get_conf(s['orig_rx'], s['rx'], 'rx')
        tx_c = get_conf(s['orig_tx'], s['tx'], 'tx')
        st_c = status_conf[if_id]
        
        # Sanity: Down = 0
        if s['status'] == 'down':
            if s['rx'] > 1.0 or s['tx'] > 1.0:
                rx_c = 0.0
                tx_c = 0.0
        
        result[if_id] = {
            'rx_rate': (s['orig_rx'], s['rx'], rx_c),
            'tx_rate': (s['orig_tx'], s['tx'], tx_c),
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