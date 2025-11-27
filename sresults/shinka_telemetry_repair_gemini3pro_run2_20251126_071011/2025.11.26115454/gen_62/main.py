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
    1. Status Repair: Standard traffic-based inference.
    2. Rate Repair (Iterative):
       - Calculate Flow Imbalance for ALL routers.
       - For each link, evaluate 'Peer Validity' (does Peer fit its own router's flow?)
         and 'Peer Solidity' (does Peer fit MY router's flow?).
       - Arbitrate between Self and Peer:
         - Reject Physically Impossible values (RX > Peer TX).
         - Trust Peer if it is Valid (Remote Consensus) or Solid (Local Consensus).
         - Trust Self if Peer is Invalid/Phantom and Self fits Local Flow.
    3. Confidence:
       - Calibrated based on Link Agreement + Local Flow Quality + Remote Flow Quality.
    """
    
    # --- Configuration ---
    HARDENING_THRESHOLD = 0.02   # 2% relative error considered 'match'
    PHYSICS_THRESHOLD = 0.005    # 0.5% tolerance
    BASE_NOISE_FLOOR = 10.0      # Minimum Mbps
    ITERATIONS = 4               # Convergence count
    
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
        
        if local_traffic or peer_traffic:
            s['status'] = 'up'
        elif peer_is_down and not local_traffic:
            s['status'] = 'down'
        
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
            imbalance = sum_rx - sum_tx
            router_stats[r_id] = {
                'imbalance': imbalance
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
            
            # Helper to get flow target for a specific value type (RX or TX)
            def get_flow_target(router_id, current_val, is_rx):
                if router_id not in router_stats: return None
                imb = router_stats[router_id]['imbalance']
                # If RX: New_RX = Old_RX - Imbalance (to reduce surplus)
                if is_rx: return max(0.0, current_val - imb)
                # If TX: New_TX = Old_TX + Imbalance (to utilize surplus)
                else: return max(0.0, current_val + imb)

            # --- RX Repair ---
            # Self RX should match Peer TX
            val_self = s['rx']
            val_peer = state[peer_id]['tx'] if has_peer else None
            
            final_rx = val_self
            
            if val_peer is not None:
                # 1. Physics: RX > Peer TX is Impossible
                if val_self > val_peer * (1.0 + PHYSICS_THRESHOLD):
                    final_rx = val_peer
                elif calc_error(val_self, val_peer) < HARDENING_THRESHOLD:
                    final_rx = (val_self + val_peer) / 2.0
                else:
                    # Disagreement (Likely Self < Peer)
                    # Check Peer Validity (Does Peer TX match Remote Flow?)
                    peer_validity = False
                    # Peer TX is a TX on the Remote Router
                    ft_remote = get_flow_target(rr_id, val_peer, is_rx=False)
                    if ft_remote is not None and calc_error(val_peer, ft_remote) < HARDENING_THRESHOLD:
                        peer_validity = True
                        
                    # Check Peer Solidity (Does Peer TX fix Local Flow?)
                    # If we replaced Self RX with Peer TX, would it work?
                    # Target for Self RX is:
                    ft_local = get_flow_target(r_id, val_self, is_rx=True)
                    peer_solidity = False
                    if ft_local is not None and calc_error(val_peer, ft_local) < HARDENING_THRESHOLD:
                        peer_solidity = True
                        
                    if peer_validity or peer_solidity:
                        final_rx = val_peer
                    else:
                        # Peer is neither supported by remote nor helps local.
                        # Does Self match Local Flow?
                        if ft_local is not None and calc_error(val_self, ft_local) < HARDENING_THRESHOLD:
                            final_rx = val_self
                        else:
                            # Ambiguous. Default to Peer (Link Symmetry).
                            final_rx = val_peer

            # --- TX Repair ---
            # Self TX should match Peer RX
            val_self = s['tx']
            val_peer = state[peer_id]['rx'] if has_peer else None
            
            final_tx = val_self
            
            if val_peer is not None:
                # 1. Physics: TX < Peer RX is Impossible
                if val_self < val_peer * (1.0 - PHYSICS_THRESHOLD):
                    final_tx = val_peer
                elif calc_error(val_self, val_peer) < HARDENING_THRESHOLD:
                    final_tx = (val_self + val_peer) / 2.0
                else:
                    # Disagreement
                    # Check Peer Validity (Peer RX on Remote)
                    peer_validity = False
                    ft_remote = get_flow_target(rr_id, val_peer, is_rx=True)
                    if ft_remote is not None and calc_error(val_peer, ft_remote) < HARDENING_THRESHOLD:
                        peer_validity = True
                        
                    # Check Peer Solidity (Peer RX as Self TX target)
                    peer_solidity = False
                    ft_local = get_flow_target(r_id, val_self, is_rx=False)
                    if ft_local is not None and calc_error(val_peer, ft_local) < HARDENING_THRESHOLD:
                        peer_solidity = True
                        
                    if peer_validity or peer_solidity:
                        final_tx = val_peer
                    else:
                        if ft_local is not None and calc_error(val_self, ft_local) < HARDENING_THRESHOLD:
                            final_tx = val_self
                        else:
                            final_tx = val_peer

            next_values[if_id] = {'rx': final_rx, 'tx': final_tx}

        # Apply
        for if_id, vals in next_values.items():
            state[if_id]['rx'] = vals['rx']
            state[if_id]['tx'] = vals['tx']

    # --- Step 4: Confidence ---
    result = {}
    
    # Final Flow Qualities
    router_qual = {}
    for r_id, if_ids in topology.items():
        sum_rx = sum(state[i]['rx'] for i in if_ids if i in state)
        sum_tx = sum(state[i]['tx'] for i in if_ids if i in state)
        vol = max(sum_rx, sum_tx, BASE_NOISE_FLOOR)
        imb = abs(sum_rx - sum_tx) / vol
        router_qual[r_id] = max(0.0, 1.0 - (imb * 10.0))

    for if_id, s in state.items():
        orig_rx = s['orig_rx']
        orig_tx = s['orig_tx']
        peer_id = s['peer']
        has_peer = peer_id and peer_id in state
        
        peer_tx = state[peer_id]['tx'] if has_peer else None
        peer_rx = state[peer_id]['rx'] if has_peer else None
        
        l_q = router_qual.get(s['router'], 0.5)
        r_q = router_qual.get(s['remote_router'], 0.5)

        def get_conf(final, orig, peer_val, lq, rq):
            # Check Link Agreement
            link_match = False
            if peer_val is not None and calc_error(final, peer_val) < HARDENING_THRESHOLD:
                link_match = True
            
            # Check Change
            changed = calc_error(final, orig) > HARDENING_THRESHOLD
            
            conf = 1.0
            
            if changed:
                if link_match:
                    # Consensus with Peer.
                    # Boost with Router Qualities.
                    # Base 0.88 + contributions
                    conf = 0.88 + (0.07 * lq) + (0.04 * rq)
                else:
                    # No Consensus with Peer (e.g. Flow Override).
                    # Needs strong Local support.
                    if lq > 0.9: conf = 0.85
                    else: conf = 0.60
            else:
                if peer_val is not None and not link_match:
                    # Disagreement, kept Self.
                    # Dangerous. Needs strong Local support.
                    if lq > 0.9: 
                        # Self is flow-verified.
                        if rq < 0.5:
                            # Remote is bad, so Peer is likely wrong. High confidence.
                            conf = 0.95
                        else:
                            # Remote is good? Why mismatch?
                            conf = 0.90
                    else:
                        conf = 0.70
                else:
                    conf = 1.0
            
            return max(0.0, min(1.0, conf))

        rx_conf = get_conf(s['rx'], orig_rx, peer_tx, l_q, r_q)
        tx_conf = get_conf(s['tx'], orig_tx, peer_rx, l_q, r_q)
        
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
