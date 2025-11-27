# EVOLVE-BLOCK-START
"""
Network telemetry repair algorithm using Dual-Source Flow Consensus.
"""
from typing import Dict, Any, Tuple, List
import math

def repair_network_telemetry(telemetry: Dict[str, Dict[str, Any]], 
                             topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:
    """
    Repair network telemetry using Dual-Source Flow Consensus.
    
    Strategy:
    1. Status Repair: Infer status from Local AND Peer traffic evidence.
    2. Rate Repair (Iterative):
       - Candidates: Compare Local Measurement vs Peer Measurement.
       - Physics: strictly enforce RX <= Peer TX (with tolerance).
       - Arbitration: Calculate a 'Flow Cost' for each candidate.
         - Cost = Local_Router_Imbalance + Remote_Router_Imbalance.
         - Select the candidate that minimizes total network flow error.
       - Source Bias: In tie-breaks, trust the Transmitting side (TX) over Receiving side (RX).
    3. Confidence: Hybrid scoring based on repair magnitude, peer agreement, and linear flow penalty.
    """
    
    # --- Configuration ---
    HARDENING_THRESHOLD = 0.02   # 2% relative error considered 'match'
    IMPOSSIBLE_THRESHOLD = 0.01  # 1% tolerance for physical violations
    NOISE_FLOOR = 10.0           # Minimum Mbps for significant operations
    ITERATIONS = 4               # Convergence count
    
    # --- Helper: Normalized Error ---
    def calc_error(v1, v2):
        return abs(v1 - v2) / max(v1, v2, NOISE_FLOOR)

    # --- Step 1: Initialization ---
    state = {}
    for if_id, data in telemetry.items():
        state[if_id] = {
            'rx': float(data.get('rx_rate', 0.0)),
            'tx': float(data.get('tx_rate', 0.0)),
            'status': data.get('interface_status', 'unknown'),
            'peer': data.get('connected_to'),
            'router': data.get('local_router'),
            'remote': data.get('remote_router'),
            'orig_rx': float(data.get('rx_rate', 0.0)),
            'orig_tx': float(data.get('tx_rate', 0.0)),
            'orig_status': data.get('interface_status', 'unknown')
        }

    # --- Step 2: Robust Status Repair ---
    for if_id, s in state.items():
        # Evidence check
        local_active = s['rx'] > NOISE_FLOOR or s['tx'] > NOISE_FLOOR
        
        peer_id = s['peer']
        peer_down = False
        peer_active = False
        
        if peer_id and peer_id in telemetry:
            p = telemetry[peer_id]
            if p.get('interface_status') == 'down':
                peer_down = True
            if float(p.get('rx_rate', 0.0)) > NOISE_FLOOR or float(p.get('tx_rate', 0.0)) > NOISE_FLOOR:
                peer_active = True
                
        # Decision Logic
        # 1. Traffic implies UP
        # 2. Peer DOWN implies DOWN (unless local traffic overrides)
        if local_active or peer_active:
            s['status'] = 'up'
        elif peer_down and not local_active:
            s['status'] = 'down'
        # Else: keep original (e.g., admin up but idle)

        # Consistency
        if s['status'] != 'up':
            s['rx'] = 0.0
            s['tx'] = 0.0

    # --- Step 3: Iterative Rate Repair ---
    for _ in range(ITERATIONS):
        
        # Pre-calculate sums for all routers to speed up cost checks
        router_sums = {}
        for r_id, if_ids in topology.items():
            sum_rx = sum(state[i]['rx'] for i in if_ids if i in state)
            sum_tx = sum(state[i]['tx'] for i in if_ids if i in state)
            router_sums[r_id] = {'rx': sum_rx, 'tx': sum_tx}
            
        updates = {}
        
        for if_id, s in state.items():
            if s['status'] != 'up':
                updates[if_id] = {'rx': 0.0, 'tx': 0.0}
                continue
                
            peer_id = s['peer']
            r_id = s['router']
            rem_id = s['remote']
            has_peer = peer_id and peer_id in state
            
            # --- RX Repair ---
            # Candidates: Self RX (Measurement), Peer TX (Source Truth)
            c_self = s['rx']
            c_peer = state[peer_id]['tx'] if has_peer else None
            
            final_rx = c_self
            
            if c_peer is not None:
                # 1. Physics: Impossible for RX > Peer TX (Phantom Traffic)
                if c_self > c_peer * (1.0 + IMPOSSIBLE_THRESHOLD):
                    final_rx = c_peer
                
                # 2. Agreement
                elif calc_error(c_self, c_peer) < HARDENING_THRESHOLD:
                    final_rx = (c_self + c_peer) / 2.0
                    
                # 3. Disagreement -> Dual Flow Arbitration
                else:
                    # Calculate Cost for "Self" Candidate
                    # Local Cost: Imbalance using current RX (c_self)
                    # Remote Cost: Imbalance if Peer TX was actually c_self
                    
                    cost_self = 0.0
                    if r_id in router_sums:
                        rs = router_sums[r_id]
                        cost_self += abs(rs['rx'] - rs['tx'])
                    if rem_id in router_sums:
                        rs = router_sums[rem_id]
                        # Adjust Remote Sum: Remove Peer TX, Add c_self
                        new_tx = rs['tx'] - state[peer_id]['tx'] + c_self
                        cost_self += abs(rs['rx'] - new_tx)
                        
                    # Calculate Cost for "Peer" Candidate
                    # Local Cost: Imbalance if we adopt Peer TX (c_peer)
                    # Remote Cost: Imbalance using current Peer TX (c_peer)
                    
                    cost_peer = 0.0
                    if r_id in router_sums:
                        rs = router_sums[r_id]
                        # Adjust Local Sum: Remove Self RX, Add c_peer
                        new_rx = rs['rx'] - c_self + c_peer
                        cost_peer += abs(new_rx - rs['tx'])
                    if rem_id in router_sums:
                        rs = router_sums[rem_id]
                        cost_peer += abs(rs['rx'] - rs['tx'])
                        
                    # Decision: Min Cost
                    # Bias: Trust Peer (Source) if costs are similar
                    if cost_peer <= cost_self:
                        final_rx = c_peer
                    else:
                        final_rx = c_self

            # --- TX Repair ---
            # Candidates: Self TX (Source Truth), Peer RX (Measurement)
            c_self = s['tx']
            c_peer = state[peer_id]['rx'] if has_peer else None
            
            final_tx = c_self
            
            if c_peer is not None:
                # 1. Physics: Impossible for TX < Peer RX (Creation on wire)
                if c_self < c_peer * (1.0 - IMPOSSIBLE_THRESHOLD):
                    final_tx = c_peer
                
                # 2. Agreement
                elif calc_error(c_self, c_peer) < HARDENING_THRESHOLD:
                    final_tx = (c_self + c_peer) / 2.0
                    
                # 3. Disagreement -> Dual Flow Arbitration
                else:
                    # Cost Self (Keep Self TX)
                    cost_self = 0.0
                    if r_id in router_sums:
                        rs = router_sums[r_id]
                        cost_self += abs(rs['rx'] - rs['tx'])
                    if rem_id in router_sums:
                        rs = router_sums[rem_id]
                        # Remote RX should be c_self
                        new_rx = rs['rx'] - state[peer_id]['rx'] + c_self
                        cost_self += abs(new_rx - rs['tx'])
                        
                    # Cost Peer (Adopt Peer RX)
                    cost_peer = 0.0
                    if r_id in router_sums:
                        rs = router_sums[r_id]
                        # Local TX becomes c_peer
                        new_tx = rs['tx'] - c_self + c_peer
                        cost_peer += abs(rs['rx'] - new_tx)
                    if rem_id in router_sums:
                        rs = router_sums[rem_id]
                        cost_peer += abs(rs['rx'] - rs['tx'])
                        
                    # Decision: Min Cost
                    # Bias: Trust Self (Source) if costs are similar
                    if cost_self <= cost_peer:
                        final_tx = c_self
                    else:
                        final_tx = c_peer

            updates[if_id] = {'rx': final_rx, 'tx': final_tx}

        # Apply updates
        for if_id, vals in updates.items():
            state[if_id]['rx'] = vals['rx']
            state[if_id]['tx'] = vals['tx']

    # --- Step 4: Confidence Calibration ---
    result = {}
    
    # Calculate Final Flow Quality for Penalty
    router_imbalance = {}
    for r_id, if_ids in topology.items():
        sum_rx = sum(state[i]['rx'] for i in if_ids if i in state)
        sum_tx = sum(state[i]['tx'] for i in if_ids if i in state)
        vol = max(sum_rx, sum_tx, NOISE_FLOOR)
        router_imbalance[r_id] = abs(sum_rx - sum_tx) / vol

    for if_id, s in state.items():
        orig_rx = s['orig_rx']
        orig_tx = s['orig_tx']
        peer_id = s['peer']
        has_peer = peer_id and peer_id in state
        r_id = s['router']
        
        # Imbalance Ratio for penalty (0.0 to 1.0)
        imb = router_imbalance.get(r_id, 0.0)
        
        peer_tx = state[peer_id]['tx'] if has_peer else None
        peer_rx = state[peer_id]['rx'] if has_peer else None
        
        def get_confidence(final, orig, peer_val):
            # Base Confidence
            conf = 1.0
            
            is_changed = calc_error(final, orig) > HARDENING_THRESHOLD
            
            # Check Peer Support
            peer_matches = False
            if peer_val is not None and calc_error(final, peer_val) < HARDENING_THRESHOLD:
                peer_matches = True
            
            if is_changed:
                if peer_matches:
                    conf = 0.98 # Validated Repair
                else:
                    conf = 0.75 # Unverified Repair (likely forced by flow)
            else:
                if peer_val is not None and not peer_matches:
                    conf = 0.85 # Disagreement maintained (Source Bias)
                else:
                    conf = 1.0  # Agreement or No Peer
            
            # Linear Penalty based on Residual Router Imbalance
            # If router is still messy, our confidence decreases.
            # Penalty factor: 2.0. (5% imbalance -> -0.1 confidence)
            penalty = imb * 2.0
            conf = max(0.0, conf - penalty)
            
            return conf

        rx_conf = get_confidence(s['rx'], orig_rx, peer_tx)
        tx_conf = get_confidence(s['tx'], orig_tx, peer_rx)
        
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