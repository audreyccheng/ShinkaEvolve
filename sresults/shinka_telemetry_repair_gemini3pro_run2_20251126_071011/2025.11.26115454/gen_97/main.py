# EVOLVE-BLOCK-START
"""
Network telemetry repair algorithm using Symmetric Flow Consensus.
"""
from typing import Dict, Any, Tuple, List
import math

def repair_network_telemetry(telemetry: Dict[str, Dict[str, Any]], 
                             topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:
    """
    Repair network telemetry using Symmetric Flow Consensus.
    
    Strategy:
    1. Status Repair: Traffic > Status indicators.
    2. Iterative Rate Repair:
       - Compute Router Quality and Imbalance.
       - Calculate 'Flow Targets' for both Local (RX) and Remote (TX).
       - Golden Truth: If Peer value satisfies Local Flow Target, trust it.
       - Arbitration:
         - Physics Violations (RX > Peer TX) -> Clamp to Peer.
         - Loss (RX < Peer TX) -> Trust Peer if it helps balance Local, or if Remote is reliable.
       - Convergence: Repeat to propagate corrections.
    3. Confidence Calibration:
       - Score based on Agreement, Repair magnitude, and Router Quality.
       - Penalize score by final Residual Imbalance of the local router.
    """
    
    # --- Configuration ---
    HARDENING_THRESHOLD = 0.02   # 2% relative error considered 'match'
    PHYSICS_THRESHOLD = 0.005    # 0.5% tolerance for physical violations
    BASE_NOISE_FLOOR = 10.0      # Minimum Mbps
    ITERATIONS = 5               # Increased iterations for propagation
    
    # --- Helper: Dynamic Noise Floor ---
    def get_noise_floor(rate_a, rate_b=0.0):
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
        local_active = s['rx'] > BASE_NOISE_FLOOR or s['tx'] > BASE_NOISE_FLOOR
        peer_active = False
        peer_down = False
        
        if s['peer'] and s['peer'] in state:
            p = state[s['peer']]
            if p['orig_rx'] > BASE_NOISE_FLOOR or p['orig_tx'] > BASE_NOISE_FLOOR:
                peer_active = True
            if p['orig_status'] == 'down':
                peer_down = True
        
        if local_active or peer_active:
            s['status'] = 'up'
        elif peer_down and not local_active:
            s['status'] = 'down'
        
        if s['status'] != 'up':
            s['rx'] = 0.0
            s['tx'] = 0.0

    # --- Step 3: Iterative Rate Repair ---
    for _ in range(ITERATIONS):
        
        # 3.1: Calculate Router Stats
        router_stats = {}
        for r_id, if_ids in topology.items():
            sum_rx = sum(state[i]['rx'] for i in if_ids if i in state)
            sum_tx = sum(state[i]['tx'] for i in if_ids if i in state)
            vol = max(sum_rx, sum_tx, BASE_NOISE_FLOOR)
            imbalance = sum_rx - sum_tx
            router_stats[r_id] = {
                'imbalance': imbalance,
                'quality': max(0.0, 1.0 - (abs(imbalance) / vol * 10.0))
            }

        next_values = {}

        for if_id, s in state.items():
            if s['status'] != 'up':
                next_values[if_id] = {'rx': 0.0, 'tx': 0.0}
                continue

            peer_id = s['peer']
            r_id = s['router']
            has_peer = peer_id and peer_id in state
            
            # Local Flow Target
            rs = router_stats.get(r_id, {'imbalance': 0.0, 'quality': 0.5})
            # If RX is x, Imbalance is I. Correct RX should be x - I.
            flow_rx_target = max(0.0, s['rx'] - rs['imbalance'])
            flow_tx_target = max(0.0, s['tx'] + rs['imbalance'])
            
            # Remote Stats
            remote_r_id = s['remote_router']
            remote_q = 0.5
            if remote_r_id and remote_r_id in router_stats:
                remote_q = router_stats[remote_r_id]['quality']

            # --- RX Repair (Incoming) ---
            val_self = s['rx']
            val_peer = state[peer_id]['tx'] if has_peer else None
            final_rx = val_self
            
            if val_peer is not None:
                # Physics: RX cannot exceed Peer TX (Phantom)
                if val_self > val_peer * (1.0 + PHYSICS_THRESHOLD):
                    # Violation. Clamp to Peer.
                    final_rx = val_peer
                
                # Agreement
                elif calc_error(val_self, val_peer) < HARDENING_THRESHOLD:
                    final_rx = (val_self + val_peer) / 2.0
                
                # Disagreement (Likely RX < Peer TX, Loss)
                else:
                    # Logic: Trust Peer if it helps Local Balance, or if Remote is Healthy.
                    dist_peer_flow = calc_error(val_peer, flow_rx_target)
                    dist_self_flow = calc_error(val_self, flow_rx_target)
                    
                    if dist_peer_flow < dist_self_flow:
                        # Peer value makes us balanced. Trust Peer.
                        final_rx = val_peer
                    elif remote_q > 0.8 and rs['quality'] < 0.5:
                        # Remote is healthy, Local is messy. Trust Remote.
                        final_rx = val_peer
                    elif rs['quality'] > 0.8 and remote_q < 0.5:
                         # Local is healthy, Remote is messy. Trust Self.
                         final_rx = val_self
                    else:
                         # Ambiguous. Default to Peer (Source Truth) for RX loss scenarios
                         final_rx = val_peer

            # --- TX Repair (Outgoing) ---
            val_self = s['tx']
            val_peer = state[peer_id]['rx'] if has_peer else None
            final_tx = val_self
            
            if val_peer is not None:
                # Physics: TX cannot be less than Peer RX (Impossible)
                if val_self < val_peer * (1.0 - PHYSICS_THRESHOLD):
                    final_tx = val_peer
                
                # Agreement
                elif calc_error(val_self, val_peer) < HARDENING_THRESHOLD:
                    final_tx = (val_self + val_peer) / 2.0
                
                # Disagreement
                else:
                    dist_peer_flow = calc_error(val_peer, flow_tx_target)
                    dist_self_flow = calc_error(val_self, flow_tx_target)
                    
                    if dist_peer_flow < dist_self_flow:
                        final_tx = val_peer
                    elif remote_q > 0.8 and rs['quality'] < 0.5:
                        # Trust Peer RX (Remote) over my TX
                        final_tx = val_peer
                    elif rs['quality'] > 0.8 and remote_q < 0.5:
                        final_tx = val_self
                    else:
                        # Default to Self (TX is source)
                         final_tx = val_self

            next_values[if_id] = {'rx': final_rx, 'tx': final_tx}

        # Update
        for if_id, vals in next_values.items():
            state[if_id]['rx'] = vals['rx']
            state[if_id]['tx'] = vals['tx']

    # --- Step 4: Confidence Calibration ---
    result = {}
    
    # Final Qualities
    final_router_qual = {}
    final_router_imb_ratio = {}
    
    for r_id, if_ids in topology.items():
        sum_rx = sum(state[i]['rx'] for i in if_ids if i in state)
        sum_tx = sum(state[i]['tx'] for i in if_ids if i in state)
        vol = max(sum_rx, sum_tx, BASE_NOISE_FLOOR)
        imb_ratio = abs(sum_rx - sum_tx) / vol
        
        final_router_imb_ratio[r_id] = imb_ratio
        final_router_qual[r_id] = max(0.0, 1.0 - (imb_ratio * 10.0))

    for if_id, s in state.items():
        orig_rx = s['orig_rx']
        orig_tx = s['orig_tx']
        
        peer_id = s['peer']
        has_peer = peer_id and peer_id in state
        
        peer_tx = state[peer_id]['tx'] if has_peer else None
        peer_rx = state[peer_id]['rx'] if has_peer else None
        
        r_id = s['router']
        remote_r_id = s['remote_router']
        
        l_q = final_router_qual.get(r_id, 0.5)
        r_q = final_router_qual.get(remote_r_id, 0.5) if remote_r_id else 0.5
        
        l_imb = final_router_imb_ratio.get(r_id, 0.0)

        def get_confidence(final, orig, peer_val, l_q, r_q, imb_ratio):
            # 1. Base Score
            score = 1.0
            
            repaired = calc_error(final, orig) > HARDENING_THRESHOLD
            matches_peer = peer_val is not None and calc_error(final, peer_val) < HARDENING_THRESHOLD
            
            if repaired:
                if matches_peer:
                    # Strongest repair.
                    score = 0.90 + (0.05 * l_q) + (0.04 * r_q)
                else:
                    # Weak repair (Flow forced, or Averaged?)
                    if l_q > 0.8: score = 0.85
                    else: score = 0.60
            else:
                # Kept Original
                if peer_val is not None and not matches_peer:
                    # Conflict.
                    if l_q > 0.9: score = 0.95
                    elif l_q > 0.7: score = 0.80
                    else: score = 0.70
                else:
                    # Agreement or No Peer
                    score = 1.0
            
            # 2. Imbalance Penalty (Damping)
            # If router is still messed up, reduce confidence.
            # E.g. 10% imbalance -> 0.15 penalty.
            score -= (imb_ratio * 1.5)
            
            return max(0.0, min(1.0, score))

        rx_conf = get_confidence(s['rx'], orig_rx, peer_tx, l_q, r_q, l_imb)
        tx_conf = get_confidence(s['tx'], orig_tx, peer_rx, l_q, r_q, l_imb)
        
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