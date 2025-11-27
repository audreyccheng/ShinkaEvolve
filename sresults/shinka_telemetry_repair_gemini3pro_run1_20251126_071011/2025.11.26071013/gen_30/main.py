# EVOLVE-BLOCK-START
"""
Iterative Flow-Balance Optimization with Synthetic Candidates
and Continuous Confidence Calibration.

Key Improvements:
1. Synthetic Candidates: Generates potential rate values based on router flow residuals to fix "Double Dead" links.
2. Continuous Voting: Selects rates by minimizing a continuous cost function (Flow Error + Deviation), avoiding binary threshold pitfalls.
3. Granular Confidence: Confidence scores decay continuously with residual error, improving calibration.
"""
from typing import Dict, Any, Tuple, List
import collections
import math

def repair_network_telemetry(telemetry: Dict[str, Dict[str, Any]], 
                             topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:

    # --- Constants ---
    TOLERANCE = 0.02           # 2% symmetry/consistency tolerance
    FLOW_TOLERANCE = 0.05      # 5% flow conservation tolerance
    MIN_ACTIVITY = 0.01        # Threshold for "active" traffic
    
    # --- Data Prep ---
    state = {}
    router_map = collections.defaultdict(list)
    verifiable_routers = set()
    
    # Identify verifiable routers (all interfaces monitored)
    for rid, if_list in topology.items():
        router_map[rid] = if_list
        if all(if_id in telemetry for if_id in if_list):
            verifiable_routers.add(rid)
            
    # Initial Load
    for if_id, data in telemetry.items():
        state[if_id] = {
            'rx': float(data.get('rx_rate', 0.0)),
            'tx': float(data.get('tx_rate', 0.0)),
            'status': data.get('interface_status', 'down'),
            'orig_rx': float(data.get('rx_rate', 0.0)),
            'orig_tx': float(data.get('tx_rate', 0.0)),
            'orig_st': data.get('interface_status', 'down'),
            'local_router': data.get('local_router'),
            'connected_to': data.get('connected_to'),
            'remote_router': data.get('remote_router')
        }

    # --- Helper: Flow Calculation ---
    def get_flow_imbalance(rid, hypothetical_updates=None):
        """
        Calculate (Sum_Rx - Sum_Tx) for a router.
        hypothetical_updates: dict {if_id: {'rx': v, 'tx': v}} to override current state
        Returns: (diff, total_traffic)
        """
        if rid not in verifiable_routers:
            return None, 0.0
        
        sum_rx, sum_tx = 0.0, 0.0
        for iface in router_map[rid]:
            # Get values
            r = state[iface]['rx']
            t = state[iface]['tx']
            
            # Override if hypothetical
            if hypothetical_updates and iface in hypothetical_updates:
                upd = hypothetical_updates[iface]
                if 'rx' in upd: r = upd['rx']
                if 'tx' in upd: t = upd['tx']
            
            sum_rx += r
            sum_tx += t
            
        return abs(sum_rx - sum_tx), max(sum_rx, sum_tx, 1.0)

    # --- 1. Status Repair ---
    # Traffic implies UP. Peer UP+Traffic implies UP.
    status_conf = {}
    for if_id, s in state.items():
        orig = s['orig_st']
        peer_id = s['connected_to']
        
        local_act = (s['orig_rx'] > MIN_ACTIVITY) or (s['orig_tx'] > MIN_ACTIVITY)
        peer_act = False
        peer_st = 'unknown'
        
        if peer_id and peer_id in state:
            peer_act = (state[peer_id]['orig_rx'] > MIN_ACTIVITY) or \
                       (state[peer_id]['orig_tx'] > MIN_ACTIVITY)
            peer_st = state[peer_id]['orig_st']
            
        new_st = orig
        conf = 1.0
        
        if local_act or peer_act:
            new_st = 'up'
            if orig == 'down': conf = 0.95
        elif orig == 'up' and peer_st == 'down':
            new_st = 'down'
            conf = 0.8
        elif orig != peer_st:
            # Conflict, no traffic. Trust Down.
            new_st = 'down'
            conf = 0.7
            
        state[if_id]['status'] = new_st
        status_conf[if_id] = conf
        
        # Zero out rates if DOWN
        if new_st == 'down':
            state[if_id]['rx'] = 0.0
            state[if_id]['tx'] = 0.0

    # --- 2. Iterative Rate Repair ---
    # We use a scoring function to pick the best consensus value for each link
    
    for _ in range(5): # 5 passes for convergence
        for if_id, s in state.items():
            if s['status'] == 'down': continue
            
            peer_id = s['connected_to']
            if not peer_id or peer_id not in state: continue
            
            # Focusing on the link flow: Local(Tx) -> Remote(Rx)
            # We must choose ONE value for this flow that satisfies both routers.
            
            rid_local = s['local_router']
            rid_remote = s['remote_router']
            
            curr_tx = s['tx']
            curr_rx = state[peer_id]['rx']
            
            # --- Generate Candidates ---
            candidates = set()
            candidates.add(curr_tx)
            candidates.add(curr_rx)
            candidates.add((curr_tx + curr_rx) / 2.0)
            
            # Synthesize Flow-Based Candidates (The "Missing Piece" logic)
            # If I am Local, what Tx value would balance me?
            # Sum_Rx - Sum_Other_Tx = My_Tx
            if rid_local in verifiable_routers:
                s_rx, s_tx = 0.0, 0.0
                for iface in router_map[rid_local]:
                    s_rx += state[iface]['rx']
                    if iface != if_id:
                        s_tx += state[iface]['tx']
                target = s_rx - s_tx
                if target > -MIN_ACTIVITY: 
                    candidates.add(max(0.0, target))
                    
            # If Remote, what Rx value would balance it?
            if rid_remote in verifiable_routers:
                r_rx, r_tx = 0.0, 0.0
                for iface in router_map[rid_remote]:
                    r_tx += state[iface]['tx']
                    if iface != peer_id:
                        r_rx += state[iface]['rx']
                target = r_tx - r_rx
                if target > -MIN_ACTIVITY:
                    candidates.add(max(0.0, target))

            # --- Score Candidates ---
            best_val = curr_tx
            best_score = float('inf')
            
            for val in candidates:
                if val < 0: continue
                
                # Check Local Error
                err_local = 0.0
                if rid_local in verifiable_routers:
                    # Hypothesize this TX value
                    diff, mag = get_flow_imbalance(rid_local, {if_id: {'tx': val}})
                    err_local = diff / mag if mag > 0 else 0
                    
                # Check Remote Error
                err_remote = 0.0
                if rid_remote in verifiable_routers:
                    # Hypothesize this RX value (at peer)
                    diff, mag = get_flow_imbalance(rid_remote, {peer_id: {'rx': val}})
                    err_remote = diff / mag if mag > 0 else 0
                
                # Deviation Penalty (Soft constraint to prefer original measurements if flow is satisfied)
                # We penalize squared relative deviation to prioritize the average in ties
                dev_cost = 0.0
                orig_tx_nz = s['orig_tx'] if s['orig_tx'] > MIN_ACTIVITY else 0
                orig_rx_nz = state[peer_id]['orig_rx'] if state[peer_id]['orig_rx'] > MIN_ACTIVITY else 0
                
                if orig_tx_nz: 
                    d = (val - orig_tx_nz) / orig_tx_nz
                    dev_cost += d*d
                if orig_rx_nz: 
                    d = (val - orig_rx_nz) / orig_rx_nz
                    dev_cost += d*d
                
                # Combined Score
                # Flow Error is dominant (Weight 10). Deviation is tie-breaker (Weight 0.1).
                score = (err_local * 10.0) + (err_remote * 10.0) + (dev_cost * 0.1)
                
                # Heuristic: Avoid zero if possible (Dead Counter avoidance)
                # If we have non-zero measurements but candidate is zero, penalize heavily
                if val < MIN_ACTIVITY and (orig_tx_nz or orig_rx_nz):
                    score += 5.0 
                
                if score < best_score:
                    best_score = score
                    best_val = val
            
            # Apply
            state[if_id]['tx'] = best_val
            state[peer_id]['rx'] = best_val

    # --- 3. Confidence & Formatting ---
    result = {}
    
    # Pre-calc verification status
    router_errors = {}
    for rid in verifiable_routers:
        d, m = get_flow_imbalance(rid)
        router_errors[rid] = d / m if m > 0 else 0.0
        
    for if_id, s in state.items():
        # Retrieve originals
        orig_rx = s['orig_rx']
        orig_tx = s['orig_tx']
        final_rx = s['rx']
        final_tx = s['tx']
        final_st = s['status']
        
        # Determine confidence for Rates
        def calc_conf(val, field_type):
            rid = s['local_router']
            rem_rid = s['remote_router']
            
            # 1. Verification Level
            loc_err = router_errors.get(rid)
            loc_ok = (loc_err is not None and loc_err < FLOW_TOLERANCE)
            
            rem_ok = False
            if rem_rid in router_errors and router_errors[rem_rid] < FLOW_TOLERANCE:
                 rem_ok = True
            
            # Base Confidence logic
            if loc_ok and rem_ok:
                base = 1.0
            elif loc_ok or rem_ok:
                base = 0.95
            else:
                # Unverifiable (e.g. edge router)
                # Trust consistency with original data if relatively unchanged
                orig = orig_rx if field_type == 'rx' else orig_tx
                if abs(val - orig) < max(orig*0.05, MIN_ACTIVITY):
                    base = 0.9
                else:
                    base = 0.7 
            
            # 2. Residual Error Penalty (Continuous Decay)
            # If "Verified" but still has error (e.g. 4.9%), confidence should drop slightly
            relevant_err = loc_err if loc_err is not None else 0.0
            # Decay: Linear scale within tolerance
            # 0% error -> 1.0 multiplier, 5% error -> ~0.9 multiplier
            if relevant_err > 0.001:
                 if relevant_err > FLOW_TOLERANCE:
                     base *= 0.5
                 else:
                     base *= (1.0 - (relevant_err * 2.0))
            
            # 3. Dead Counter Repair Bonus
            # If we revived a dead counter to a flow-verified value, high confidence
            orig = orig_rx if field_type == 'rx' else orig_tx
            if orig < MIN_ACTIVITY and val > MIN_ACTIVITY and (loc_ok or rem_ok):
                base = max(base, 0.9)
                
            return float(f"{base:.3f}")

        rx_conf = calc_conf(final_rx, 'rx')
        tx_conf = calc_conf(final_tx, 'tx')
        st_conf = status_conf.get(if_id, 1.0)
        
        # Override for DOWN
        if final_st == 'down':
             if final_rx > MIN_ACTIVITY or final_tx > MIN_ACTIVITY:
                 rx_conf = 0.0
                 tx_conf = 0.0
        
        result[if_id] = {
            'rx_rate': (orig_rx, final_rx, rx_conf),
            'tx_rate': (orig_tx, final_tx, tx_conf),
            'interface_status': (s['orig_st'], final_st, st_conf),
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