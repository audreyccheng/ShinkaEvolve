# EVOLVE-BLOCK-START
"""
Network telemetry repair algorithm using Dual-Flow Consensus.
"""
from typing import Dict, Any, Tuple, List
import math

def repair_network_telemetry(telemetry: Dict[str, Dict[str, Any]], 
                             topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:
    """
    Repair network telemetry using Dual-Flow Consensus.
    
    Strategy:
    1. Status Repair: Priority on Traffic Evidence > Peer Status > Original Status.
    2. Rate Repair (Iterative):
       - Calculates Flow Imbalance for all routers.
       - Uses Dual-Source Arbitration:
         - Validates candidates (Self vs Peer) against Local Flow Target.
         - Incorporates Remote Router Quality (Peer's health) to weight Peer reliability.
         - Applies Asymmetric Thresholds: Strict for Physics Violations (RX > TX), 
           Lenient for Plausible Loss.
    3. Confidence Calibration:
       - Discrete confidence tiers based on repair type (Agreement vs Flow Override).
       - Linear penalty based on residual flow imbalance.
    """
    
    # --- Configuration ---
    # Thresholds
    T_HARD = 0.02       # 2% standard matching for agreement
    T_STRICT = 0.005    # 0.5% strict physics violation limit (RX > Peer TX)
    NOISE_FLOOR = 10.0  # Mbps, ignore variance below this
    ITERATIONS = 3      # Convergence steps
    
    # --- Helper: Normalized Error ---
    def get_noise_floor(v1, v2=0.0):
        # 0.1% scaling for high speed links to avoid floating point noise issues
        return max(NOISE_FLOOR, max(v1, v2) * 0.001)

    def calc_error(v1, v2):
        nf = get_noise_floor(v1, v2)
        return abs(v1 - v2) / max(v1, v2, nf)

    # --- Initialization ---
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

    # --- Step 1: Status Repair ---
    for if_id, s in state.items():
        # Evidence: Traffic
        local_active = s['rx'] > NOISE_FLOOR or s['tx'] > NOISE_FLOOR
        
        peer_down = False
        peer_active = False
        if s['peer'] and s['peer'] in state:
            p = state[s['peer']]
            if p['orig_status'] == 'down':
                peer_down = True
            if p['orig_rx'] > NOISE_FLOOR or p['orig_tx'] > NOISE_FLOOR:
                peer_active = True
        
        # Logic: Traffic overrides status flags
        if local_active or peer_active:
            s['status'] = 'up'
        elif peer_down:
            s['status'] = 'down'
        # Else: keep original status
        
        # Consistency
        if s['status'] != 'up':
            s['rx'] = 0.0
            s['tx'] = 0.0

    # --- Step 2: Rate Repair Iterations ---
    for _ in range(ITERATIONS):
        
        # 2.1 Calculate Router Flow Metrics
        r_stats = {}
        for r_id, if_list in topology.items():
            sum_rx = sum(state[i]['rx'] for i in if_list if i in state)
            sum_tx = sum(state[i]['tx'] for i in if_list if i in state)
            vol = max(sum_rx, sum_tx, NOISE_FLOOR)
            imb = sum_rx - sum_tx # Net Imbalance (RX - TX)
            r_stats[r_id] = {
                'imb': imb,
                'vol': vol,
                'quality': max(0.0, 1.0 - (abs(imb) / vol * 10.0)) # 1.0 = Balanced
            }

        updates = {}
        
        for if_id, s in state.items():
            if s['status'] != 'up':
                updates[if_id] = {'rx': 0.0, 'tx': 0.0}
                continue
                
            peer_id = s['peer']
            r_id = s['router']
            rem_id = s['remote_router']
            has_peer = peer_id and peer_id in state
            
            # Retrieve Current Values
            curr_rx = s['rx']
            curr_tx = s['tx']
            
            peer_tx = state[peer_id]['tx'] if has_peer else None
            peer_rx = state[peer_id]['rx'] if has_peer else None
            
            # Calculate Local Flow Targets
            # Target is the value that zeros the imbalance (curr - Imbalance)
            f_rx = None
            f_tx = None
            q_loc = 0.5
            if r_id in r_stats:
                st = r_stats[r_id]
                f_rx = max(0.0, curr_rx - st['imb'])
                f_tx = max(0.0, curr_tx + st['imb'])
                q_loc = st['quality']
            
            # Determine Remote Router Quality
            q_rem = 0.5
            if rem_id and rem_id in r_stats:
                q_rem = r_stats[rem_id]['quality']
            elif rem_id is None and has_peer:
                # Infer from Peer's local router
                p_r = state[peer_id]['router']
                if p_r in r_stats:
                    q_rem = r_stats[p_r]['quality']

            # --- Arbitration Logic ---
            def arbitrate(val_self, val_peer, val_flow, q_self, q_peer, is_rx):
                if val_peer is None:
                    return val_self
                
                # 1. Strict Physics Constraint
                if is_rx:
                    # RX > Peer TX is physically impossible (Phantom Traffic)
                    # Allow T_STRICT tolerance
                    if val_self > val_peer * (1.0 + T_STRICT):
                         return val_peer
                else:
                    # TX < Peer RX is physically impossible (Traffic Creation on Wire)
                    if val_self < val_peer * (1.0 - T_STRICT):
                         return val_peer
                
                # 2. Agreement Check
                if calc_error(val_self, val_peer) < T_HARD:
                    return (val_self + val_peer) / 2.0
                
                # 3. Flow-Based Arbitration (Disagreement > T_HARD)
                if val_flow is not None:
                    # Calculate distances to Flow Target
                    err_self_flow = calc_error(val_self, val_flow)
                    err_peer_flow = calc_error(val_peer, val_flow)
                    
                    # Determining Preference
                    prefer_peer = err_peer_flow < err_self_flow
                    
                    # Sanity Check on Source Reliability
                    # If Flow suggests Peer, but Peer's router is chaotic (Low Quality),
                    # we might hesitate. But usually, linking to a chaotic router is the source of error,
                    # so trusting the link equation (Peer value) helps stabilize local.
                    
                    # However, if Peer is chaotic, its value might be garbage. 
                    # If Local is Stable (High Quality) and Self fits Flow, keep Self.
                    if prefer_peer:
                        if q_peer < 0.3 and q_self > 0.8:
                            # Peer is chaotic, Local is good. Trust Local.
                            return val_self
                        return val_peer
                    else:
                        # Flow suggests Self is better.
                        if q_peer > 0.9 and q_self < 0.3:
                            # Peer is solid, Local is chaotic. Trust Peer.
                            return val_peer
                        return val_self
                
                # No Flow info -> Trust Peer (R3 Invariant)
                return val_peer

            new_rx = arbitrate(curr_rx, peer_tx, f_rx, q_loc, q_rem, is_rx=True)
            new_tx = arbitrate(curr_tx, peer_rx, f_tx, q_loc, q_rem, is_rx=False)
            
            updates[if_id] = {'rx': new_rx, 'tx': new_tx}
            
        # Apply updates
        for if_id, vals in updates.items():
            state[if_id]['rx'] = vals['rx']
            state[if_id]['tx'] = vals['tx']

    # --- Step 3: Confidence Calibration ---
    result = {}
    
    # Calculate Final Router Imbalance Ratios
    final_imb_ratios = {}
    for r_id, if_list in topology.items():
        sum_rx = sum(state[i]['rx'] for i in if_list if i in state)
        sum_tx = sum(state[i]['tx'] for i in if_list if i in state)
        vol = max(sum_rx, sum_tx, NOISE_FLOOR)
        final_imb_ratios[r_id] = abs(sum_rx - sum_tx) / vol

    for if_id, s in state.items():
        orig_rx = s['orig_rx']
        orig_tx = s['orig_tx']
        peer_id = s['peer']
        has_peer = peer_id and peer_id in state
        
        peer_tx = state[peer_id]['tx'] if has_peer else None
        peer_rx = state[peer_id]['rx'] if has_peer else None
        
        r_id = s['router']
        imb_ratio = final_imb_ratios.get(r_id, 0.0)
        
        def get_conf(final, orig, peer_val):
            conf = 1.0
            
            changed = calc_error(final, orig) > T_HARD
            matches_peer = peer_val is not None and calc_error(final, peer_val) < T_HARD
            
            if changed:
                if matches_peer:
                    # Validated Repair (Tier 2)
                    conf = 0.95
                else:
                    # Flow Override / Averaging (Tier 3)
                    conf = 0.85
            else:
                # Kept Original
                if peer_val is not None and not matches_peer:
                    # Disagreement (Self Trusted) (Tier 4)
                    conf = 0.80
                else:
                    # Agreement (Tier 1)
                    conf = 1.0
            
            # Linear Penalty for Residual Imbalance
            # If router is still imbalanced, confidence decreases.
            conf -= (imb_ratio * 2.0)
            
            return max(0.0, min(1.0, conf))

        rx_conf = get_conf(s['rx'], orig_rx, peer_tx)
        tx_conf = get_conf(s['tx'], orig_tx, peer_rx)
        
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