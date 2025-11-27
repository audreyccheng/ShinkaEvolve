# EVOLVE-BLOCK-START
"""
Network telemetry repair algorithm using Dual-Sided Flow Consensus.
"""
from typing import Dict, Any, Tuple, List
import math

def repair_network_telemetry(telemetry: Dict[str, Dict[str, Any]], 
                             topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:
    """
    Repair network telemetry using Dual-Sided Flow Consensus.
    
    Strategy:
    1. Status Repair: Infer status from traffic activity.
    2. Rate Repair (Iterative):
       - Calculate Flow Imbalance and Quality for all routers.
       - Use 'Dual Solidity' to arbitrate disagreements:
         - A value is 'Local Solid' if it balances the local router.
         - A value is 'Remote Solid' if it comes from a balanced remote router.
       - Physics constraints (RX <= Peer TX) enforced strictly.
    3. Confidence Calibration:
       - Hybrid score based on Peer Agreement, Local Flow Consistency, and Remote Flow Quality.
    """
    
    # --- Configuration ---
    HARDENING_THRESHOLD = 0.02   # 2% relative error considered 'match'
    PHYSICS_THRESHOLD = 0.005    # 0.5% tolerance for physical violations
    BASE_NOISE_FLOOR = 10.0      # Minimum Mbps to consider 'active'
    ITERATIONS = 4               # Convergence count
    
    # --- Helper: Dynamic Noise Floor ---
    def get_noise_floor(rate_a, rate_b=0.0):
        # Scale noise floor for high speed links (0.1%), but keep base floor
        mx = max(rate_a, rate_b)
        return max(BASE_NOISE_FLOOR, mx * 0.001)

    # --- Helper: Normalized Error ---
    def calc_error(v1, v2):
        nf = get_noise_floor(v1, v2)
        return abs(v1 - v2) / max(v1, v2, nf)

    # --- Step 1: Initialization ---
    state = {}
    for if_id, data in telemetry.items():
        state[if_id] = {
            'rx': float(data.get('rx_rate', 0.0)),
            'tx': float(data.get('tx_rate', 0.0)),
            'status': data.get('interface_status', 'unknown'),
            'peer': data.get('connected_to'),
            'router': data.get('local_router'),
            'remote_router': data.get('remote_router'),
            'orig_rx': float(data.get('rx_rate', 0.0)),
            'orig_tx': float(data.get('tx_rate', 0.0)),
            'orig_status': data.get('interface_status', 'unknown')
        }

    # --- Step 2: Robust Status Repair ---
    for if_id, s in state.items():
        # Evidence
        nf = get_noise_floor(s['rx'], s['tx'])
        local_traffic = s['rx'] > nf or s['tx'] > nf
        
        peer_traffic = False
        peer_is_down = False
        if s['peer'] and s['peer'] in state:
            p = state[s['peer']]
            nf_p = get_noise_floor(p['orig_rx'], p['orig_tx'])
            if p['orig_rx'] > nf_p or p['orig_tx'] > nf_p:
                peer_traffic = True
            if p['orig_status'] == 'down':
                peer_is_down = True
        
        # Decision Matrix
        if local_traffic or peer_traffic:
            s['status'] = 'up'
        elif peer_is_down and not local_traffic:
            s['status'] = 'down'
        # Else: keep original
        
        # Consistency enforce
        if s['status'] != 'up':
            s['rx'] = 0.0
            s['tx'] = 0.0

    # --- Step 3: Iterative Rate Repair ---
    for _ in range(ITERATIONS):
        
        # 3.1: Pre-calculate Router Flow States
        router_stats = {}
        for r_id, if_ids in topology.items():
            sum_rx = sum(state[i]['rx'] for i in if_ids if i in state)
            sum_tx = sum(state[i]['tx'] for i in if_ids if i in state)
            vol = max(sum_rx, sum_tx, BASE_NOISE_FLOOR)
            imbalance = sum_rx - sum_tx
            quality = max(0.0, 1.0 - (abs(imbalance) / vol * 10.0))
            router_stats[r_id] = {
                'imbalance': imbalance,
                'quality': quality
            }

        next_values = {}

        for if_id, s in state.items():
            if s['status'] != 'up':
                next_values[if_id] = {'rx': 0.0, 'tx': 0.0}
                continue

            peer_id = s['peer']
            local_r = s['router']
            remote_r = s['remote_router']
            has_peer = peer_id and peer_id in state
            
            # --- Flow Implied Values ---
            # What value would balance the local router?
            local_flow_rx = None
            local_flow_tx = None
            local_q = 0.5
            
            if local_r in router_stats:
                ls = router_stats[local_r]
                local_q = ls['quality']
                # RX_new = RX_old - Imbalance
                local_flow_rx = max(0.0, s['rx'] - ls['imbalance'])
                # TX_new = TX_old + Imbalance
                local_flow_tx = max(0.0, s['tx'] + ls['imbalance'])
                
            # Remote Quality
            remote_q = 0.5
            if remote_r and remote_r in router_stats:
                remote_q = router_stats[remote_r]['quality']

            # --- RX Repair ---
            # Constraint: RX <= Peer TX
            val_self = s['rx']
            val_peer = state[peer_id]['tx'] if has_peer else None
            
            final_rx = val_self
            
            if val_peer is not None:
                # 1. Physics Violation (RX > Peer TX)
                if val_self > val_peer * (1.0 + PHYSICS_THRESHOLD):
                    final_rx = val_peer
                
                # 2. Agreement
                elif calc_error(val_self, val_peer) < HARDENING_THRESHOLD:
                    final_rx = (val_self + val_peer) / 2.0
                    
                # 3. Disagreement (RX < Peer TX) or Ambiguity
                else:
                    # Logic: Is Peer Value "Solid" w.r.t Local Flow?
                    # i.e., Does accepting Peer TX fix my imbalance?
                    peer_matches_local_flow = False
                    if local_flow_rx is not None:
                        if calc_error(val_peer, local_flow_rx) < HARDENING_THRESHOLD:
                            peer_matches_local_flow = True
                            
                    # Does Local Value match Local Flow? (Real Loss?)
                    self_matches_local_flow = False
                    if local_flow_rx is not None:
                         if calc_error(val_self, local_flow_rx) < HARDENING_THRESHOLD:
                            self_matches_local_flow = True
                    
                    if peer_matches_local_flow:
                        # Best Case: Peer fixes my flow.
                        final_rx = val_peer
                    elif self_matches_local_flow:
                        # My current value fits my flow. Peer is higher. 
                        # This implies Real Loss (packets dropped) or Peer is over-reporting.
                        # We keep Self.
                        final_rx = val_self
                    else:
                        # Neither fits perfectly. Use Flow Quality to arbitrate.
                        if remote_q > local_q:
                            # Peer router is healthier. Trust Peer.
                            final_rx = val_peer
                        elif local_q > 0.8:
                            # Local is healthy enough, and we checked self_matches_local_flow above?
                            # If self didn't match flow, but local is generally healthy, 
                            # we are just imbalanced on this link.
                            # Default to Peer (Link Symmetry).
                            final_rx = val_peer
                        else:
                             final_rx = val_peer

            # --- TX Repair ---
            # Constraint: TX >= Peer RX
            val_self = s['tx']
            val_peer = state[peer_id]['rx'] if has_peer else None
            
            final_tx = val_self
            
            if val_peer is not None:
                # 1. Physics Violation (TX < Peer RX)
                if val_self < val_peer * (1.0 - PHYSICS_THRESHOLD):
                    final_tx = val_peer
                
                # 2. Agreement
                elif calc_error(val_self, val_peer) < HARDENING_THRESHOLD:
                    final_tx = (val_self + val_peer) / 2.0
                    
                # 3. Disagreement
                else:
                    # Does Peer RX fit my flow?
                    peer_matches_local_flow = False
                    if local_flow_tx is not None:
                        if calc_error(val_peer, local_flow_tx) < HARDENING_THRESHOLD:
                            peer_matches_local_flow = True
                            
                    self_matches_local_flow = False
                    if local_flow_tx is not None:
                        if calc_error(val_self, local_flow_tx) < HARDENING_THRESHOLD:
                            self_matches_local_flow = True
                            
                    if peer_matches_local_flow:
                        final_tx = val_peer
                    elif self_matches_local_flow:
                        final_tx = val_self
                    else:
                        if remote_q > local_q:
                            final_tx = val_peer
                        else:
                            final_tx = val_peer

            next_values[if_id] = {'rx': final_rx, 'tx': final_tx}

        # Apply updates
        for if_id, vals in next_values.items():
            state[if_id]['rx'] = vals['rx']
            state[if_id]['tx'] = vals['tx']

    # --- Step 4: Confidence Calibration ---
    result = {}
    
    # Final Flow Qualities
    final_router_qual = {}
    for r_id, if_ids in topology.items():
        sum_rx = sum(state[i]['rx'] for i in if_ids if i in state)
        sum_tx = sum(state[i]['tx'] for i in if_ids if i in state)
        vol = max(sum_rx, sum_tx, BASE_NOISE_FLOOR)
        imb = abs(sum_rx - sum_tx) / vol
        final_router_qual[r_id] = max(0.0, 1.0 - (imb * 10.0))

    for if_id, s in state.items():
        orig_rx = s['orig_rx']
        orig_tx = s['orig_tx']
        
        peer_id = s['peer']
        has_peer = peer_id and peer_id in state
        
        peer_tx = state[peer_id]['tx'] if has_peer else None
        peer_rx = state[peer_id]['rx'] if has_peer else None
        
        r_id = s['router']
        remote_r = s['remote_router']
        
        l_q = final_router_qual.get(r_id, 0.5)
        r_q = final_router_qual.get(remote_r, 0.5) if remote_r else 0.5
        
        def get_confidence(final, orig, peer_val, l_q, r_q):
            # 1. Did we change it?
            was_repaired = calc_error(final, orig) > HARDENING_THRESHOLD
            
            # 2. Peer Agreement
            matches_peer = False
            if peer_val is not None and calc_error(final, peer_val) < HARDENING_THRESHOLD:
                matches_peer = True
                
            conf = 1.0
            
            if was_repaired:
                if matches_peer:
                    # Consensus Repair (Aligned with Peer)
                    # Boost based on Source Quality (Remote) and Destination Balance (Local)
                    # Base: 0.85
                    # Remote Quality is crucial: if source is bad, peer value is suspect.
                    conf = 0.85 + (0.07 * r_q) + (0.06 * l_q)
                else:
                    # Non-Consensus Repair (e.g. Averaging or Flow Override)
                    # We changed value, but NOT to Peer.
                    # This happens if we split difference or trust Flow over Peer.
                    if l_q > 0.9:
                        conf = 0.85
                    else:
                        conf = 0.60
            else:
                # Kept Original
                if peer_val is not None and not matches_peer:
                    # Disagreement (Self != Peer)
                    # We trusted Self over Peer.
                    # This implies we think Peer is wrong/lossy OR we are Solid.
                    if l_q > 0.9:
                        # My router is balanced.
                        if r_q > 0.8:
                            # Remote is also balanced. This is a stalemate/loss.
                            conf = 0.92
                        else:
                            # Remote is bad. I am good. Trust me.
                            conf = 0.96
                    elif l_q > 0.7:
                        conf = 0.80
                    else:
                        conf = 0.70 # Low confidence
                else:
                    # Agreement or No Peer
                    conf = 1.0
                    
            return max(0.0, min(1.0, conf))

        rx_conf = get_confidence(s['rx'], orig_rx, peer_tx, l_q, r_q)
        tx_conf = get_confidence(s['tx'], orig_tx, peer_rx, l_q, r_q)
        
        # Status confidence
        st_conf = 1.0
        if s['status'] != s['orig_status']:
            st_conf = 0.95

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

