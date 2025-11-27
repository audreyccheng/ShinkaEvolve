# EVOLVE-BLOCK-START
"""
Continuous Synthesis Repair Algorithm
Combines iterative constraint satisfaction with residual-based value synthesis
and continuous error scoring to repair network telemetry.
"""
from typing import Dict, Any, Tuple, List
import collections

def repair_network_telemetry(telemetry: Dict[str, Dict[str, Any]], 
                             topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:
    
    # --- Constants ---
    TOLERANCE = 0.02          # 2% symmetry tolerance
    FLOW_TOLERANCE = 0.05     # 5% flow conservation tolerance
    MIN_ACTIVITY = 0.01       # 10 Kbps floor for active links
    
    # --- 1. Initialization & Data Structure Setup ---
    state = {}
    router_ifs = collections.defaultdict(list)
    verifiable_routers = set()
    
    # Map topology and identify verifiable routers (complete telemetry visibility)
    for rid, if_list in topology.items():
        router_ifs[rid] = if_list
        if all(if_id in telemetry for if_id in if_list):
            verifiable_routers.add(rid)
            
    # Initialize working state
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
    # Strategy: Traffic implies UP. Peer UP implies UP (if traffic).
    # Conflict resolution: Trust DOWN if no traffic is observed.
    status_conf = {}
    
    for if_id, s in state.items():
        orig_status = s['status']
        peer_id = s['connected_to']
        
        # Check for local traffic
        has_traffic = (s['rx'] > MIN_ACTIVITY) or (s['tx'] > MIN_ACTIVITY)
        
        # Check peer status and traffic
        peer_status = 'unknown'
        peer_traffic = False
        if peer_id and peer_id in state:
            p = state[peer_id]
            peer_status = p['status']
            peer_traffic = (p['rx'] > MIN_ACTIVITY) or (p['tx'] > MIN_ACTIVITY)
            
        final_status = orig_status
        conf = 1.0
        
        if has_traffic or peer_traffic:
            # Traffic is the strongest signal for UP
            final_status = 'up'
            if orig_status == 'down':
                conf = 0.95 # Correcting a false negative
        elif orig_status == 'up' and peer_status == 'down':
            # No traffic, peer says down -> Likely DOWN
            final_status = 'down'
            conf = 0.8
        elif orig_status != peer_status:
            # Conflict with no traffic info. 
            # Trust the DOWN signal as it's a specific error condition vs a default UP.
            final_status = 'down' if peer_status == 'down' else orig_status
            conf = 0.7
            
        state[if_id]['status'] = final_status
        status_conf[if_id] = conf
        
        # Enforce Consistency: Down interfaces must have zero rate
        if final_status == 'down':
            state[if_id]['rx'] = 0.0
            state[if_id]['tx'] = 0.0
            
    # --- 3. Rate Repair (Iterative Continuous Consensus) ---
    
    def get_flow_error_with_val(rid, if_target, field, val):
        """ 
        Calculate relative flow error for a router if we speculatively 
        set if_target.field = val. Returns None if router not verifiable.
        """
        if rid not in verifiable_routers:
            return None
            
        sum_rx = 0.0
        sum_tx = 0.0
        
        for iface in router_ifs[rid]:
            if iface == if_target:
                r = val if field == 'rx' else state[iface]['rx']
                t = val if field == 'tx' else state[iface]['tx']
            else:
                r = state[iface]['rx']
                t = state[iface]['tx']
            sum_rx += r
            sum_tx += t
            
        diff = abs(sum_rx - sum_tx)
        denom = max(sum_rx, sum_tx, 1.0)
        return diff / denom
        
    def get_synth_val(rid, if_target, field):
        """ 
        Synthesize the value required for if_target.field to perfectly 
        balance the router's flow conservation equation.
        """
        if rid not in verifiable_routers: return None
        
        sum_in = 0.0
        sum_out = 0.0
        for iface in router_ifs[rid]:
            if iface == if_target: continue
            sum_in += state[iface]['rx']
            sum_out += state[iface]['tx']
            
        # If calculating Tx (Outgoing), it must equal (Total_In - Other_Out)
        if field == 'tx':
            return max(0.0, sum_in - sum_out)
        # If calculating Rx (Incoming), it must equal (Total_Out - Other_In)
        else:
            return max(0.0, sum_out - sum_in)

    # Iterative optimization loop (Gauss-Seidel style)
    for _ in range(3):
        for if_id, s in state.items():
            if s['status'] == 'down': continue
            
            # Link Direction: We primarily repair Local(Tx) -> Remote(Rx)
            # The other direction (Local(Rx) <- Remote(Tx)) is handled when visiting the peer.
            
            rid_local = s['local_router']
            val_tx = s['tx']
            
            peer_id = s['connected_to']
            has_peer = peer_id and peer_id in state
            
            # --- Candidate Generation ---
            candidates = {val_tx}
            
            if has_peer:
                val_rx = state[peer_id]['rx']
                rid_remote = s['remote_router']
                candidates.add(val_rx)
                candidates.add((val_tx + val_rx)/2.0)
            else:
                rid_remote = None
                val_rx = 0.0 # Placeholder
                
            # Synthesis: Calculate what values would satisfy flow constraints
            synth_tx = get_synth_val(rid_local, if_id, 'tx')
            if synth_tx is not None: candidates.add(synth_tx)
            
            if has_peer:
                synth_rx = get_synth_val(rid_remote, peer_id, 'rx')
                if synth_rx is not None: candidates.add(synth_rx)
                
            # --- Candidate Selection ---
            best_val = val_tx
            best_score = float('inf')
            
            # Heuristic trigger: If any candidate is non-zero, we are skeptical of zeros
            active_signal = any(c > MIN_ACTIVITY for c in candidates)
            
            for c in candidates:
                if c < 0: continue
                
                # Continuous Scoring Function
                score = 0.0
                
                # Local Flow Penalty
                err_loc = get_flow_error_with_val(rid_local, if_id, 'tx', c)
                if err_loc is not None:
                    score += min(err_loc, 1.0) # Cap error at 100%
                else:
                    score += 0.05 # Small penalty for unverifiability
                
                # Remote Flow Penalty
                if has_peer:
                    err_rem = get_flow_error_with_val(rid_remote, peer_id, 'rx', c)
                    if err_rem is not None:
                        score += min(err_rem, 1.0)
                    else:
                        score += 0.05
                
                # Penalties & Bonuses
                
                # "Dead Counter" Penalty: If signal exists, 0 is likely wrong
                if active_signal and c < MIN_ACTIVITY:
                    score += 0.5
                    
                # Consensus Bonus: Slight preference for the average value if multiple sources exist
                if has_peer:
                    avg = (val_tx + val_rx) / 2.0
                    if abs(c - avg) < 0.0001:
                        score -= 0.01 
                        
                if score < best_score:
                    best_score = score
                    best_val = c
            
            # Apply repair
            state[if_id]['tx'] = best_val
            if has_peer:
                state[peer_id]['rx'] = best_val
                
            # --- Edge Case: Unmonitored Peer ---
            # If there is no peer, we never visit the other side to fix RX.
            # We must attempt to fix RX here using only local synthesis.
            if not has_peer:
                val_rx_local = s['rx']
                synth_rx_local = get_synth_val(rid_local, if_id, 'rx')
                
                if synth_rx_local is not None:
                     # Check if synthesis is drastically better than current
                     err_curr = get_flow_error_with_val(rid_local, if_id, 'rx', val_rx_local)
                     # Synthetic value has 0 error by definition
                     
                     if err_curr is not None and err_curr > FLOW_TOLERANCE:
                         # Trust synthesis if current is broken
                         state[if_id]['rx'] = synth_rx_local

    # --- 4. Final Result & Confidence Calibration ---
    result = {}
    
    # Pre-calculate final flow errors to detect broken routers
    final_errors = {}
    for rid in verifiable_routers:
        sum_rx = sum(state[i]['rx'] for i in router_ifs[rid])
        sum_tx = sum(state[i]['tx'] for i in router_ifs[rid])
        err = abs(sum_rx - sum_tx) / max(sum_rx, sum_tx, 1.0)
        final_errors[rid] = err
        
    for if_id, data in telemetry.items():
        orig_rx = data.get('rx_rate', 0.0)
        orig_tx = data.get('tx_rate', 0.0)
        final_rx = state[if_id]['rx']
        final_tx = state[if_id]['tx']
        
        rid = data.get('local_router')
        peer_id = data.get('connected_to')
        rem_rid = data.get('remote_router')
        
        def get_conf(orig, final, field):
            # Special Case: Interface Down
            if state[if_id]['status'] == 'down':
                # If we forced down, rates must be 0. High confidence if they are.
                return 0.95 if abs(final) < MIN_ACTIVITY else 0.0
                
            changed = abs(orig - final) > max(orig*0.01, 0.001)
            
            # Check Verification Status
            loc_err = final_errors.get(rid)
            loc_ok = loc_err is not None and loc_err < FLOW_TOLERANCE
            
            rem_err = final_errors.get(rem_rid)
            rem_ok = rem_err is not None and rem_err < FLOW_TOLERANCE
            
            # Check Peer Consistency
            peer_consistent = True
            if peer_id and peer_id in state:
                peer_val = state[peer_id]['tx'] if field == 'rx' else state[peer_id]['rx']
                if abs(final - peer_val) > max(final * TOLERANCE, MIN_ACTIVITY):
                    peer_consistent = False
            
            # --- Scoring Logic ---
            if not changed:
                if loc_ok and rem_ok: return 1.0     # Perfect
                if loc_ok: return 0.98               # Verified locally
                if not peer_consistent: return 0.7   # Conflict, but we kept original (uncertain)
                if rem_ok: return 0.95               # Verified remotely
                return 0.9                           # Default trust in original
            
            # Changed
            if loc_ok and rem_ok: return 0.99        # Verified repair
            if loc_ok: return 0.96                   # Local math confirms repair
            if rem_ok: return 0.92                   # Remote math confirms repair
            
            # Unverified Repairs
            if orig < MIN_ACTIVITY and final > MIN_ACTIVITY:
                return 0.85 # "Dead counter" repair (0 -> Value) is common
            
            if not peer_consistent:
                return 0.5 # We changed it, but it still doesn't match peer? Bad.
                
            return 0.75 # Best guess based on peer, but unverified
            
        rx_c = get_conf(orig_rx, final_rx, 'rx')
        tx_c = get_conf(orig_tx, final_tx, 'tx')
        
        # Global Consistency Penalty
        # If the local router is still reporting high flow error, cap confidence.
        # This prevents overconfidence when the algorithm fails to find a solution.
        if rid in final_errors and final_errors[rid] > FLOW_TOLERANCE:
            rx_c = min(rx_c, 0.6)
            tx_c = min(tx_c, 0.6)
            
        result[if_id] = {
            'rx_rate': (orig_rx, final_rx, rx_c),
            'tx_rate': (orig_tx, final_tx, tx_c),
            'interface_status': (data.get('interface_status'), state[if_id]['status'], status_conf.get(if_id, 1.0)),
            'connected_to': peer_id,
            'local_router': rid,
            'remote_router': rem_rid
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
