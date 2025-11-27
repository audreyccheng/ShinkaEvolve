# EVOLVE-BLOCK-START
"""
Network telemetry repair algorithm using Symmetric Flow Consensus and Trust Heuristics.
"""
from typing import Dict, Any, Tuple, List
import math

def repair_network_telemetry(telemetry: Dict[str, Dict[str, Any]],
                             topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:
    """
    Repair network telemetry using Symmetric Flow Consensus.

    Key Strategies:
    1. Symmetric Flow Verification: Candidates are checked against flow constraints of
       BOTH the local and remote routers. If a value satisfies both, it is prioritized.
    2. Trust Healthy Heuristic: In disagreements, if one router is significantly
       healthier (better balanced) than the other, its data is trusted.
    3. Residual Damping: Confidence scores are penalized if the final repaired state
       of the router remains unbalanced.
    """

    # --- Configuration ---
    HARDENING_THRESHOLD = 0.02   # 2% relative error considered 'match'
    STRICT_THRESHOLD = 0.01      # 1% strictness for physics violations
    BASE_NOISE_FLOOR = 10.0      # Minimum Mbps to consider 'active'
    ITERATIONS = 5               # Convergence count

    # --- Helper: Dynamic Noise Floor ---
    def get_noise_floor(v1, v2=0.0):
        # Scale noise floor for high speed links, but keep base floor
        return max(BASE_NOISE_FLOOR, max(v1, v2) * 0.001)

    # --- Helper: Relative Error ---
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
        # Else keep original
        
        if s['status'] != 'up':
            s['rx'] = 0.0
            s['tx'] = 0.0

    # --- Step 3: Iterative Rate Repair ---
    for _ in range(ITERATIONS):
        
        # 3.1 Calculate Router Stats
        router_stats = {}
        for r_id, if_ids in topology.items():
            sum_rx = sum(state[i]['rx'] for i in if_ids if i in state)
            sum_tx = sum(state[i]['tx'] for i in if_ids if i in state)
            imb = sum_rx - sum_tx
            vol = max(sum_rx, sum_tx, BASE_NOISE_FLOOR)
            router_stats[r_id] = {
                'imbalance': imb,
                'quality': max(0.0, 1.0 - (abs(imb) / vol * 10.0))
            }

        next_values = {}
        
        for if_id, s in state.items():
            if s['status'] != 'up':
                next_values[if_id] = {'rx': 0.0, 'tx': 0.0}
                continue

            peer_id = s['peer']
            r_id = s['router']
            rr_id = s['remote_router']
            has_peer = peer_id and peer_id in state
            
            # Retrieve Stats
            rs_local = router_stats.get(r_id, {'imbalance': 0.0, 'quality': 0.5})
            q_local = rs_local['quality']
            
            rs_remote = router_stats.get(rr_id, {'imbalance': 0.0, 'quality': 0.5})
            q_remote = rs_remote['quality']
            
            # --- Flow Targets ---
            # Local RX Target (Balances Local Router)
            # RX_new = RX_old - Imbalance
            target_rx_local = max(0.0, s['rx'] - rs_local['imbalance'])
            
            # Local TX Target
            # TX_new = TX_old + Imbalance
            target_tx_local = max(0.0, s['tx'] + rs_local['imbalance'])
            
            # Remote Targets (Balances Remote Router)
            # Calculated relative to Peer's current values
            peer_tx_val = state[peer_id]['tx'] if has_peer else 0.0
            peer_rx_val = state[peer_id]['rx'] if has_peer else 0.0
            
            # Peer TX Target (aligns with Local RX)
            target_peer_tx = max(0.0, peer_tx_val + rs_remote['imbalance'])
            
            # Peer RX Target (aligns with Local TX)
            target_peer_rx = max(0.0, peer_rx_val - rs_remote['imbalance'])
            
            # --- RX Repair ---
            val_self = s['rx']
            val_peer = peer_tx_val if has_peer else None
            final_rx = val_self
            
            if val_peer is not None:
                # 1. Symmetric Golden Truth Check
                # Does Peer value satisfy BOTH Local and Remote requirements?
                d_peer_local = calc_error(val_peer, target_rx_local)
                d_peer_remote = calc_error(val_peer, target_peer_tx)
                
                is_golden_peer = (d_peer_local < HARDENING_THRESHOLD and 
                                  d_peer_remote < HARDENING_THRESHOLD)
                
                # 2. Trust Healthy Override
                # If Remote is pristine (>0.9) and Local is messy (<0.6), trust Peer
                trust_remote = (q_remote > 0.9 and q_local < 0.6)
                
                if is_golden_peer:
                    final_rx = val_peer
                elif trust_remote:
                    # Trust Peer unless strictly impossible (RX >> Peer TX)
                    if val_self > val_peer * (1.0 + STRICT_THRESHOLD):
                         final_rx = val_peer
                    else:
                         final_rx = val_peer
                else:
                    # Standard Arbitration
                    
                    # Physics: Phantom Traffic (RX > Peer TX)
                    if val_self > val_peer * (1.0 + STRICT_THRESHOLD):
                        final_rx = val_peer
                        
                    # Physics: Packet Loss (RX < Peer TX)
                    elif val_self < val_peer * (1.0 - STRICT_THRESHOLD):
                        # Is the loss real?
                        # If Peer fits Local Target better, it's likely measurement error -> Repair
                        d_self_local = calc_error(val_self, target_rx_local)
                        
                        if d_peer_local < d_self_local:
                            final_rx = val_peer
                        else:
                            # Self fits better (Real Loss)
                            final_rx = val_self
                    else:
                        # Agreement range
                        final_rx = (val_self + val_peer) / 2.0

            # --- TX Repair ---
            val_self = s['tx']
            val_peer = peer_rx_val if has_peer else None
            final_tx = val_self
            
            if val_peer is not None:
                # 1. Symmetric Golden Truth
                d_peer_local = calc_error(val_peer, target_tx_local)
                d_peer_remote = calc_error(val_peer, target_peer_rx)
                
                is_golden_peer = (d_peer_local < HARDENING_THRESHOLD and 
                                  d_peer_remote < HARDENING_THRESHOLD)

                trust_remote = (q_remote > 0.9 and q_local < 0.6)
                
                if is_golden_peer:
                    final_tx = val_peer
                elif trust_remote:
                    if val_self < val_peer * (1.0 - STRICT_THRESHOLD):
                        final_tx = val_peer
                    else:
                        final_tx = val_peer
                else:
                    # Physics: Impossible (TX < Peer RX)
                    if val_self < val_peer * (1.0 - STRICT_THRESHOLD):
                        final_tx = val_peer
                    # Physics: Loss on wire (TX > Peer RX)
                    elif val_self > val_peer * (1.0 + STRICT_THRESHOLD):
                        # If Peer (lower) fits Local Target better -> Phantom TX
                        d_self_local = calc_error(val_self, target_tx_local)
                        d_peer_local = calc_error(val_peer, target_tx_local)
                        
                        if d_peer_local < d_self_local:
                            final_tx = val_peer
                        else:
                            final_tx = val_self
                    else:
                        final_tx = (val_self + val_peer) / 2.0
            
            next_values[if_id] = {'rx': final_rx, 'tx': final_tx}
            
        # Update State
        for if_id, vals in next_values.items():
            state[if_id]['rx'] = vals['rx']
            state[if_id]['tx'] = vals['tx']

    # --- Step 4: Confidence Calibration ---
    # Recalculate Final Stats for Damping
    final_qual = {}
    final_imb_ratio = {}
    for r_id, if_ids in topology.items():
        sum_rx = sum(state[i]['rx'] for i in if_ids if i in state)
        sum_tx = sum(state[i]['tx'] for i in if_ids if i in state)
        vol = max(sum_rx, sum_tx, BASE_NOISE_FLOOR)
        imb = abs(sum_rx - sum_tx)
        ratio = imb / vol
        final_qual[r_id] = max(0.0, 1.0 - (ratio * 10.0))
        final_imb_ratio[r_id] = ratio

    result = {}
    for if_id, s in state.items():
        orig_rx = s['orig_rx']
        orig_tx = s['orig_tx']
        peer_id = s['peer']
        has_peer = peer_id and peer_id in state
        
        peer_tx = state[peer_id]['tx'] if has_peer else None
        peer_rx = state[peer_id]['rx'] if has_peer else None
        
        r_id = s['router']
        rr_id = s['remote_router']
        l_q = final_qual.get(r_id, 0.5)
        r_q = final_qual.get(rr_id, 0.5) if rr_id else 0.5
        
        # Damping: Penalize confidence if Local Router remains unbalanced
        # 0% imbalance -> 1.0 factor
        # 10% imbalance -> 0.8 factor
        damp_factor = 1.0 - min(0.2, final_imb_ratio.get(r_id, 0.0) * 2.0)

        def get_conf(final, orig, peer_val, l_q, r_q, damp):
            dist_orig = calc_error(final, orig)
            matches_peer = False
            if peer_val is not None and calc_error(final, peer_val) < HARDENING_THRESHOLD:
                matches_peer = True
            
            conf = 1.0
            
            if dist_orig > HARDENING_THRESHOLD:
                # Repaired
                if matches_peer:
                    # Consensus with Link
                    # Base 0.90 + Bonuses for Router Quality
                    conf = 0.90 + (0.05 * l_q) + (0.04 * r_q)
                else:
                    # Repaired, no match peer (Arbitration)
                    # Relies on Local Flow
                    if l_q > 0.9:
                        conf = 0.85
                    else:
                        conf = 0.60
            else:
                # Kept Original
                if peer_val is not None and not matches_peer:
                    # Disagreement
                    if l_q > 0.9:
                        conf = 0.95
                    else:
                        conf = 0.70
                else:
                    # Agreement or No Peer
                    conf = 1.0
            
            return max(0.0, min(1.0, conf * damp))

        rx_conf = get_conf(s['rx'], orig_rx, peer_tx, l_q, r_q, damp_factor)
        tx_conf = get_conf(s['tx'], orig_tx, peer_rx, l_q, r_q, damp_factor)
        
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