# EVOLVE-BLOCK-START
"""
Network telemetry repair algorithm that detects and corrects inconsistencies
in network interface telemetry data using topology relationships.

Takes interface telemetry data and detects/repairs inconsistencies based on
network invariants like link symmetry and flow conservation.
"""
from typing import Dict, Any, Tuple, List
import math

def repair_network_telemetry(telemetry: Dict[str, Dict[str, Any]], 
                             topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:
    """
    Repair network interface telemetry using a consensus-based approach.
    
    Strategies:
    1. Status Repair: Infer UP status from local or remote activity.
    2. Rate Repair: Iterative consensus between Self, Peer, and Flow signals.
       - Priority 1: Link Symmetry (if Self and Peer agree).
       - Priority 2: Flow Conservation (use Flow to arbitrate between Self and Peer).
       - Priority 3: Peer Authority (trust the other side of the link).
    3. Confidence: Calculated based on the convergence of multiple signals.
    """
    
    HARDENING_THRESHOLD = 0.02
    NOISE_FLOOR = 0.1
    ITERATIONS = 3
    
    # --- Step 1: Initialization & Status Repair ---
    state = {}
    
    # Initial pass to build state and infer status
    for if_id, data in telemetry.items():
        # Read raw values
        rx = float(data.get('rx_rate', 0.0))
        tx = float(data.get('tx_rate', 0.0))
        status = data.get('interface_status', 'unknown')
        connected_to = data.get('connected_to')
        
        # Check for remote activity to infer status
        peer_active = False
        if connected_to and connected_to in telemetry:
            peer_data = telemetry[connected_to]
            p_status = peer_data.get('interface_status', 'unknown')
            p_rx = float(peer_data.get('rx_rate', 0.0))
            p_tx = float(peer_data.get('tx_rate', 0.0))
            
            # If peer says UP and is sending/receiving, link is likely UP
            if p_status == 'up' and (p_rx > NOISE_FLOOR or p_tx > NOISE_FLOOR):
                peer_active = True
        
        # Infer Status
        # If we have local traffic, or peer has traffic, we are UP.
        if (rx > NOISE_FLOOR or tx > NOISE_FLOOR) or peer_active:
            final_status = 'up'
        elif status == 'down':
             final_status = 'down'
        else:
             # Status is 'up' but no traffic. Keep as 'up' (idle).
             final_status = 'up'
             
        # Enforce Down consistency
        if final_status != 'up':
            rx, tx = 0.0, 0.0
            
        state[if_id] = {
            'rx': rx,
            'tx': tx,
            'status': final_status,
            'orig_rx': float(data.get('rx_rate', 0.0)),
            'orig_tx': float(data.get('tx_rate', 0.0)),
            'orig_status': data.get('interface_status', 'unknown'),
            'peer': connected_to,
            'router': data.get('local_router')
        }

    # --- Step 2: Iterative Rate Repair ---
    for _ in range(ITERATIONS):
        # Pre-calculate router totals for Flow checks
        router_totals = {}
        for r_id, if_list in topology.items():
            sum_rx = sum(state[i]['rx'] for i in if_list if i in state)
            sum_tx = sum(state[i]['tx'] for i in if_list if i in state)
            router_totals[r_id] = {'rx': sum_rx, 'tx': sum_tx}
            
        updates = {}
        
        for if_id, s in state.items():
            if s['status'] != 'up':
                updates[if_id] = {'rx': 0.0, 'tx': 0.0}
                continue
                
            peer_id = s['peer']
            has_peer = peer_id and peer_id in state
            r_id = s['router']
            has_flow = r_id and r_id in router_totals
            
            # Helper to decide best value
            def solve_rate(current_val, is_rx):
                # 1. Gather Candidates
                candidates = {}
                candidates['self'] = current_val
                
                if has_peer:
                    # For RX, peer metric is TX. For TX, peer metric is RX.
                    candidates['peer'] = state[peer_id]['tx' if is_rx else 'rx']
                
                if has_flow:
                    totals = router_totals[r_id]
                    # Calculate what value would balance the router
                    # RX_balance = Sum_TX - Sum_RX_others
                    # TX_balance = Sum_RX - Sum_TX_others
                    if is_rx:
                        others_rx = totals['rx'] - current_val
                        flow_target = totals['tx'] - others_rx
                    else:
                        others_tx = totals['tx'] - current_val
                        flow_target = totals['rx'] - others_tx
                    
                    # Only accept valid flow targets
                    if flow_target >= 0:
                        candidates['flow'] = flow_target

                # 2. Consensus Logic
                val_self = candidates.get('self')
                val_peer = candidates.get('peer')
                val_flow = candidates.get('flow')
                
                # Case A: Symmetry (Self ≈ Peer)
                if val_peer is not None:
                    diff = abs(val_self - val_peer)
                    if diff <= max(val_self, val_peer, 1.0) * HARDENING_THRESHOLD:
                        return (val_self + val_peer) / 2.0
                
                # Case B: Arbitration via Flow
                if val_flow is not None and val_peer is not None:
                    dist_self = abs(val_self - val_flow)
                    dist_peer = abs(val_peer - val_flow)
                    
                    # If Peer is significantly closer to Flow than Self is
                    if dist_peer < dist_self * 0.5: 
                        return val_peer
                    # If Self is significantly closer
                    elif dist_self < dist_peer * 0.5:
                        return val_self
                    
                    # If both are far or both are close (Ambiguous)
                    # Verify if Peer and Flow agree
                    if dist_peer <= max(val_peer, 1.0) * HARDENING_THRESHOLD * 2:
                        return (val_peer + val_flow) / 2.0
                        
                    return val_peer # Fallback to Peer

                # Case C: No Flow, Disagreement
                if val_peer is not None:
                    return val_peer
                
                return val_self

            new_rx = solve_rate(s['rx'], True)
            new_tx = solve_rate(s['tx'], False)
            
            updates[if_id] = {'rx': new_rx, 'tx': new_tx}
            
        # Apply updates
        for if_id, u in updates.items():
            state[if_id]['rx'] = u['rx']
            state[if_id]['tx'] = u['tx']

    # --- Step 3: Confidence & Result ---
    result = {}
    for if_id, s in state.items():
        orig_rx = s['orig_rx']
        orig_tx = s['orig_tx']
        
        # Recalculate context for confidence
        peer_id = s['peer']
        peer_tx = state[peer_id]['tx'] if (peer_id and peer_id in state) else None
        peer_rx = state[peer_id]['rx'] if (peer_id and peer_id in state) else None
        
        r_id = s['router']
        flow_ok = False
        if r_id and r_id in router_totals:
            # Check final balance using updated state locally
             curr_sum_rx = sum(state[i]['rx'] for i in topology[r_id] if i in state)
             curr_sum_tx = sum(state[i]['tx'] for i in topology[r_id] if i in state)
             flow_ok = abs(curr_sum_rx - curr_sum_tx) <= max(curr_sum_rx, curr_sum_tx, 1.0) * 0.05

        def get_confidence(orig, final, peer_val, flow_is_valid):
            # Check if changed
            changed = abs(orig - final) > max(orig, 1.0) * HARDENING_THRESHOLD
            
            # Check agreements
            agrees_peer = False
            if peer_val is not None:
                agrees_peer = abs(final - peer_val) <= max(final, 1.0) * HARDENING_THRESHOLD
            
            if not changed:
                # We kept the value.
                if peer_val is not None and not agrees_peer:
                    # We defied the peer. Did we have flow support?
                    if flow_is_valid: return 0.9 # Trusted Self+Flow over Peer
                    return 0.7 # Trusted Self over Peer with no Flow support? Risky.
                return 1.0 # All good
            else:
                # We changed the value.
                if agrees_peer and flow_is_valid:
                    return 0.95 # Strongest repair
                if agrees_peer:
                    return 0.85 # Good repair
                if flow_is_valid:
                    return 0.75 # Flow-based repair (rare)
                
                return 0.5 # Low confidence

        rx_conf = get_confidence(orig_rx, s['rx'], peer_tx, flow_ok)
        tx_conf = get_confidence(orig_tx, s['tx'], peer_rx, flow_ok)
        
        # Status confidence
        st_conf = 1.0
        if s['status'] != s['orig_status']:
            if s['status'] == 'up': st_conf = 0.95
            else: st_conf = 0.8
            
        result[if_id] = {
            'rx_rate': (orig_rx, s['rx'], rx_conf),
            'tx_rate': (orig_tx, s['tx'], tx_conf),
            'interface_status': (s['orig_status'], s['status'], st_conf),
            'connected_to': telemetry[if_id].get('connected_to'),
            'local_router': telemetry[if_id].get('local_router'),
            'remote_router': telemetry[if_id].get('remote_router')
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
