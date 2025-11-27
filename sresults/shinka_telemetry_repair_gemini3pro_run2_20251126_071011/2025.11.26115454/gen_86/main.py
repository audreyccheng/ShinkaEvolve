# EVOLVE-BLOCK-START
"""
Network telemetry repair algorithm using Symmetric Flow Anchor Consensus.
"""
from typing import Dict, Any, Tuple, List
import math

def repair_network_telemetry(telemetry: Dict[str, Dict[str, Any]], 
                             topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:
    """
    Repair network telemetry using Symmetric Flow Anchor Consensus.
    
    Strategy:
    1. Status Repair: Infer status from traffic evidence.
    2. Iterative Symmetric Repair:
       - Calculates Flow Targets for both Local and Remote routers.
       - "Golden Truth": If Local and Remote flow targets for a link agree, 
         lock that value (Anchor) to prevent drift.
       - Asymmetric Physics: Strict clamping for Phantom Traffic (RX > Peer TX).
       - Dominant Quality: Trust high-quality remote routers over noisy local ones.
    3. Residual Confidence Calibration:
       - Confidence scores penalized by final residual router imbalance.
    """
    
    # --- Configuration ---
    HARDENING_THRESHOLD = 0.02    # 2% error considered 'match'
    GOLDEN_THRESHOLD = 0.01       # 1% error to declare 'Golden Truth'
    PHYSICS_THRESHOLD = 0.005     # 0.5% tolerance for physics violations
    BASE_NOISE_FLOOR = 10.0       # Minimum Mbps to consider 'active'
    ITERATIONS = 5                # Convergence count
    
    # --- Helper: Dynamic Noise Floor ---
    def get_noise_floor(rate_a, rate_b=0.0):
        # 0.1% dynamic floor to handle high-speed links without excessive noise sensitivity
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
            'orig_status': data.get('interface_status', 'unknown'),
            'rx_locked': False,
            'tx_locked': False
        }

    # --- Step 2: Robust Status Repair ---
    for if_id, s in state.items():
        local_traffic = s['rx'] > BASE_NOISE_FLOOR or s['tx'] > BASE_NOISE_FLOOR
        
        peer_down = False
        peer_traffic = False
        if s['peer'] and s['peer'] in state:
            p = state[s['peer']]
            if p['orig_rx'] > BASE_NOISE_FLOOR or p['orig_tx'] > BASE_NOISE_FLOOR:
                peer_traffic = True
            if p['orig_status'] == 'down':
                peer_down = True
        
        # Traffic evidence overrides Status flags
        if local_traffic or peer_traffic:
            s['status'] = 'up'
        elif peer_down and not local_traffic:
            s['status'] = 'down'
        # Else: keep original status
        
        # Enforce consistency
        if s['status'] != 'up':
            s['rx'] = 0.0
            s['tx'] = 0.0
            s['rx_locked'] = True
            s['tx_locked'] = True

    # --- Step 3: Iterative Rate Repair ---
    for i in range(ITERATIONS):
        
        # 3.1: Calculate Router Stats
        router_stats = {}
        for r_id, if_ids in topology.items():
            sum_rx = sum(state[if_id]['rx'] for if_id in if_ids if if_id in state)
            sum_tx = sum(state[if_id]['tx'] for if_id in if_ids if if_id in state)
            imbalance = sum_rx - sum_tx
            vol = max(sum_rx, sum_tx, BASE_NOISE_FLOOR)
            router_stats[r_id] = {
                'imbalance': imbalance,
                'quality': max(0.0, 1.0 - (abs(imbalance) / vol * 10.0))
            }

        updates = {}

        for if_id, s in state.items():
            if s['status'] != 'up': 
                continue

            peer_id = s['peer']
            r_id = s['router']
            has_peer = peer_id and peer_id in state
            
            # Get Peer Info
            peer_tx = state[peer_id]['tx'] if has_peer else None
            peer_rx = state[peer_id]['rx'] if has_peer else None
            peer_router = state[peer_id]['router'] if has_peer else None

            # Get Router Qualities
            local_q = router_stats.get(r_id, {'quality': 0.5})['quality']
            remote_q = 0.5
            if peer_router and peer_router in router_stats:
                remote_q = router_stats[peer_router]['quality']

            # --- RX Repair ---
            if not s['rx_locked']:
                curr_rx = s['rx']
                
                # 1. Flow Targets
                # Local Target: What RX makes Local Imbalance 0? (RX - Imbalance)
                local_target_rx = None
                if r_id in router_stats:
                    local_target_rx = max(0.0, curr_rx - router_stats[r_id]['imbalance'])
                
                # Remote Target: What Peer TX makes Remote Imbalance 0? (Peer TX + Peer Imbalance)
                remote_target_tx = None
                if peer_router and peer_router in router_stats:
                    remote_target_tx = max(0.0, peer_tx + router_stats[peer_router]['imbalance'])
                
                # 2. Golden Truth Check
                # If Local Target and Remote Target agree, we have a global consensus
                if local_target_rx is not None and remote_target_tx is not None:
                    if calc_error(local_target_rx, remote_target_tx) < GOLDEN_THRESHOLD:
                        updates[if_id] = updates.get(if_id, {})
                        updates[if_id]['rx'] = (local_target_rx + remote_target_tx) / 2.0
                        updates[if_id]['rx_locked'] = True
                        # If locked, we don't need further arbitration for this counter
                        
                # 3. Arbitration (if not locked)
                if if_id not in updates or 'rx' not in updates[if_id]:
                    final_rx = curr_rx
                    
                    if peer_tx is not None:
                        # Physics: RX > Peer TX (Phantom Traffic)
                        if curr_rx > peer_tx * (1.0 + PHYSICS_THRESHOLD):
                            # Strict Clamp unless Remote is terrible and Local is great
                            if remote_q < 0.3 and local_q > 0.8:
                                final_rx = curr_rx
                            else:
                                final_rx = peer_tx
                        
                        # Agreement
                        elif calc_error(curr_rx, peer_tx) < HARDENING_THRESHOLD:
                            final_rx = (curr_rx + peer_tx) / 2.0
                        
                        # Physics: RX < Peer TX (Loss) or Disagreement
                        else:
                            # Dominant Quality Override
                            if local_q < 0.5 and remote_q > 0.9:
                                final_rx = peer_tx
                            elif local_target_rx is not None:
                                # Which is closer to Local Target?
                                d_self = calc_error(curr_rx, local_target_rx)
                                d_peer = calc_error(peer_tx, local_target_rx)
                                
                                if d_peer < d_self:
                                    final_rx = peer_tx # Peer helps balance
                                elif d_self < d_peer:
                                    final_rx = curr_rx # Self helps balance
                                else:
                                    final_rx = peer_tx
                            else:
                                final_rx = peer_tx

                    updates[if_id] = updates.get(if_id, {})
                    updates[if_id]['rx'] = final_rx

            # --- TX Repair ---
            if not s['tx_locked']:
                curr_tx = s['tx']
                
                # 1. Flow Targets
                local_target_tx = None
                if r_id in router_stats:
                    local_target_tx = max(0.0, curr_tx + router_stats[r_id]['imbalance'])
                
                remote_target_rx = None
                if peer_router and peer_router in router_stats:
                    remote_target_rx = max(0.0, peer_rx - router_stats[peer_router]['imbalance'])

                # 2. Golden Truth
                if local_target_tx is not None and remote_target_rx is not None:
                    if calc_error(local_target_tx, remote_target_rx) < GOLDEN_THRESHOLD:
                        updates[if_id] = updates.get(if_id, {})
                        updates[if_id]['tx'] = (local_target_tx + remote_target_rx) / 2.0
                        updates[if_id]['tx_locked'] = True
                
                # 3. Arbitration
                if if_id not in updates or 'tx' not in updates[if_id]:
                    final_tx = curr_tx
                    
                    if peer_rx is not None:
                        # Physics: TX < Peer RX (Impossible - Packet Creation?)
                        if curr_tx < peer_rx * (1.0 - PHYSICS_THRESHOLD):
                            if remote_q < 0.3 and local_q > 0.8:
                                final_tx = curr_tx
                            else:
                                final_tx = peer_rx
                        
                        elif calc_error(curr_tx, peer_rx) < HARDENING_THRESHOLD:
                            final_tx = (curr_tx + peer_rx) / 2.0
                        
                        else:
                            # Dominant Quality
                            if local_q < 0.5 and remote_q > 0.9:
                                final_tx = peer_rx
                            elif local_target_tx is not None:
                                d_self = calc_error(curr_tx, local_target_tx)
                                d_peer = calc_error(peer_rx, local_target_tx)
                                
                                if d_peer < d_self:
                                    final_tx = peer_rx
                                elif d_self < d_peer:
                                    final_tx = curr_tx
                                else:
                                    final_tx = peer_rx
                            else:
                                final_tx = peer_rx
                                
                    updates[if_id] = updates.get(if_id, {})
                    updates[if_id]['tx'] = final_tx

        # Apply Updates
        for if_id, u in updates.items():
            if 'rx' in u: state[if_id]['rx'] = u['rx']
            if 'tx' in u: state[if_id]['tx'] = u['tx']
            if 'rx_locked' in u: state[if_id]['rx_locked'] = u['rx_locked']
            if 'tx_locked' in u: state[if_id]['tx_locked'] = u['tx_locked']

    # --- Step 4: Confidence Calibration ---
    result = {}
    
    # Final Router Qualities
    final_r_stats = {}
    for r_id, if_ids in topology.items():
        sum_rx = sum(state[i]['rx'] for i in if_ids if i in state)
        sum_tx = sum(state[i]['tx'] for i in if_ids if i in state)
        vol = max(sum_rx, sum_tx, BASE_NOISE_FLOOR)
        imb = abs(sum_rx - sum_tx)
        final_r_stats[r_id] = {
            'quality': max(0.0, 1.0 - (imb / vol * 10.0)),
            'imb_ratio': imb / vol
        }

    for if_id, s in state.items():
        orig_rx = s['orig_rx']
        orig_tx = s['orig_tx']
        peer_id = s['peer']
        has_peer = peer_id and peer_id in state
        
        peer_tx = state[peer_id]['tx'] if has_peer else None
        peer_rx = state[peer_id]['rx'] if has_peer else None
        
        r_id = s['router']
        remote_r_id = s['remote_router']
        
        l_stat = final_r_stats.get(r_id, {'quality': 0.5, 'imb_ratio': 0.1})
        r_stat = final_r_stats.get(remote_r_id, {'quality': 0.5, 'imb_ratio': 0.1}) if remote_r_id else {'quality': 0.5, 'imb_ratio': 0.1}
        
        l_q = l_stat['quality']
        r_q = r_stat['quality']
        l_imb = l_stat['imb_ratio']

        def get_confidence(final, orig, peer_val, l_q, r_q, l_imb):
            matches_peer = False
            if peer_val is not None:
                if calc_error(final, peer_val) < HARDENING_THRESHOLD:
                    matches_peer = True
            
            repaired = calc_error(final, orig) > HARDENING_THRESHOLD
            
            # Base Score
            if repaired:
                if matches_peer:
                    # Consensus Repair - High Confidence
                    base = 0.90 + (0.05 * l_q) + (0.04 * r_q)
                else:
                    # Unilateral Repair (Flow Arbitration)
                    if l_q > 0.9: base = 0.85
                    else: base = 0.60
            else:
                # Kept Original
                if peer_val is not None and not matches_peer:
                    # Conflict / Disagreement
                    if l_q > 0.9: base = 0.95
                    elif l_q > 0.7: base = 0.80
                    else: base = 0.65
                else:
                    # Agreement
                    base = 1.0
            
            # Residual Penalty: If router is still unbalanced, penalize confidence
            penalty = l_imb * 2.0 # Significant penalty for residual imbalance
            
            return max(0.0, min(1.0, base - penalty))

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