# EVOLVE-BLOCK-START
"""
Network telemetry repair algorithm using Asymmetric Flow Physics Consensus.
"""
from typing import Dict, Any, Tuple, List
import math

def repair_network_telemetry(telemetry: Dict[str, Dict[str, Any]], 
                             topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:
    """
    Repair network telemetry using Asymmetric Flow Physics Consensus.
    
    Strategy:
    1. Status Repair: Infer status from local/remote traffic activity.
       - Enforce strict DOWN propagation if Peer is DOWN and local is idle.
    2. Rate Repair (Iterative):
       - Calculate 'Flow Implied' rate that would perfectly balance the router.
       - Asymmetric Logic:
         - RX > Peer TX: 'Impossible' -> Clamp to Peer TX (Physics).
         - RX < Peer TX: 'Plausible' -> Check Flow Implied.
           - If Flow Implied matches Peer TX -> Repair (Phantom Loss).
           - If Flow Implied matches Local RX -> Keep (Real Loss).
       - Uses a 'Solidity' check: if Peer's value would fix our flow, trust it.
    3. Confidence Calibration:
       - Tiered scoring: Highest for Physics/Link agreement, lower for Flow-only deductions.
       - Penalize confidence by residual flow imbalance.
    """
    
    # --- Configuration ---
    HARDENING_THRESHOLD = 0.02   # 2% relative error considered 'match'
    PHYSICS_THRESHOLD = 0.005    # 0.5% tolerance for physical violations
    BASE_NOISE_FLOOR = 10.0      # Minimum Mbps to consider 'active'
    ITERATIONS = 4               # Convergence count
    
    # --- Helper: Dynamic Noise Floor ---
    def get_noise_floor(rate_a, rate_b=0.0):
        # Scale noise floor for high speed links, but keep base floor
        mx = max(rate_a, rate_b)
        return max(BASE_NOISE_FLOOR, mx * 0.001) # 0.1% dynamic floor

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
            'orig_rx': float(data.get('rx_rate', 0.0)),
            'orig_tx': float(data.get('tx_rate', 0.0)),
            'orig_status': data.get('interface_status', 'unknown')
        }

    # --- Step 2: Robust Status Repair ---
    for if_id, s in state.items():
        # Evidence
        local_traffic = s['rx'] > BASE_NOISE_FLOOR or s['tx'] > BASE_NOISE_FLOOR
        
        peer_traffic = False
        peer_is_down = False
        if s['peer'] and s['peer'] in state:
            p = state[s['peer']]
            if p['orig_rx'] > BASE_NOISE_FLOOR or p['orig_tx'] > BASE_NOISE_FLOOR:
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
            total_vol = max(sum_rx, sum_tx, BASE_NOISE_FLOOR)
            imbalance = sum_rx - sum_tx
            router_stats[r_id] = {
                'imbalance': imbalance,
                'quality': max(0.0, 1.0 - (abs(imbalance) / total_vol * 10.0)) 
            }

        next_values = {}

        for if_id, s in state.items():
            if s['status'] != 'up':
                next_values[if_id] = {'rx': 0.0, 'tx': 0.0}
                continue

            peer_id = s['peer']
            r_id = s['router']
            has_peer = peer_id and peer_id in state
            
            # --- Flow Implied Values ---
            flow_rx_target = None
            flow_tx_target = None
            
            if r_id in router_stats:
                rs = router_stats[r_id]
                # To balance: RX_new = RX_old - Imbalance
                flow_rx_target = max(0.0, s['rx'] - rs['imbalance'])
                # To balance: TX_new = TX_old + Imbalance
                flow_tx_target = max(0.0, s['tx'] + rs['imbalance'])

            # --- RX Repair ---
            # Constraint: RX <= Peer TX
            val_self = s['rx']
            val_peer = state[peer_id]['tx'] if has_peer else None
            
            final_rx = val_self
            
            if val_peer is not None:
                # 1. Physics Violation (RX > Peer TX)
                # Strict enforcement
                if val_self > val_peer * (1.0 + PHYSICS_THRESHOLD):
                    final_rx = val_peer
                
                # 2. Agreement
                elif calc_error(val_self, val_peer) < HARDENING_THRESHOLD:
                    final_rx = (val_self + val_peer) / 2.0
                    
                # 3. Plausible Discrepancy (RX < Peer TX)
                else:
                    # Use Flow to Arbitrate
                    if flow_rx_target is not None:
                        # Distances
                        d_peer = calc_error(val_peer, flow_rx_target)
                        d_self = calc_error(val_self, flow_rx_target)
                        
                        # Logic: If Peer is significantly closer to Flow Target, 
                        # it implies the missing packets *did* arrive (to balance flow).
                        # If Self is closer, it implies packets were lost (Flow balanced without them).
                        if d_peer < d_self:
                            final_rx = val_peer
                        elif d_self < d_peer:
                            final_rx = val_self
                        else:
                            final_rx = val_peer # Fallback to Link Truth
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
                    
                # 3. Plausible Discrepancy (TX > Peer RX)
                else:
                    if flow_tx_target is not None:
                        d_peer = calc_error(val_peer, flow_tx_target)
                        d_self = calc_error(val_self, flow_tx_target)
                        
                        if d_peer < d_self:
                            final_tx = val_peer
                        elif d_self < d_peer:
                            final_tx = val_self
                        else:
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
    
    # Recalculate Final Flow Quality
    final_router_qual = {}
    for r_id, if_ids in topology.items():
        sum_rx = sum(state[i]['rx'] for i in if_ids if i in state)
        sum_tx = sum(state[i]['tx'] for i in if_ids if i in state)
        vol = max(sum_rx, sum_tx, BASE_NOISE_FLOOR)
        imb = abs(sum_rx - sum_tx) / vol
        # Quality: 1.0 = Perfect. 0.0 = >5% Imbalance.
        final_router_qual[r_id] = max(0.0, 1.0 - (imb * 20.0))

    for if_id, s in state.items():
        orig_rx = s['orig_rx']
        orig_tx = s['orig_tx']
        
        peer_id = s['peer']
        has_peer = peer_id and peer_id in state
        
        peer_tx = state[peer_id]['tx'] if has_peer else None
        peer_rx = state[peer_id]['rx'] if has_peer else None
        
        r_id = s['router']
        flow_q = final_router_qual.get(r_id, 0.5)

        def get_confidence(final, orig, peer_val, flow_q):
            # Error metrics
            err_orig = calc_error(final, orig)
            
            matches_peer = False
            if peer_val is not None:
                if calc_error(final, peer_val) < HARDENING_THRESHOLD:
                    matches_peer = True
            
            conf = 1.0
            
            # Logic Branching
            if err_orig > HARDENING_THRESHOLD:
                # We Repaired
                if matches_peer:
                    # We aligned with Peer.
                    # This is the "Gold Standard" repair if Flow supports it.
                    # Base 0.95, + 0.04 * FlowQ
                    conf = 0.95 + (0.04 * flow_q)
                else:
                    # We repaired to something else (e.g. Average or Flow Target).
                    # This implies Peer was also wrong?
                    if flow_q > 0.9:
                        conf = 0.85
                    else:
                        conf = 0.60
            else:
                # Kept Original
                if peer_val is not None and not matches_peer:
                    # Disagreement.
                    # We defied Peer. We need Flow support.
                    if flow_q > 0.9:
                        conf = 0.95 # Flow verifies Self
                    elif flow_q > 0.7:
                        conf = 0.80 # Flow okay
                    else:
                        conf = 0.70 # Ambiguous
                else:
                    # Agreement or No Peer
                    conf = 1.0
            
            return conf

        rx_conf = get_confidence(s['rx'], orig_rx, peer_tx, flow_q)
        tx_conf = get_confidence(s['tx'], orig_tx, peer_rx, flow_q)
        
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