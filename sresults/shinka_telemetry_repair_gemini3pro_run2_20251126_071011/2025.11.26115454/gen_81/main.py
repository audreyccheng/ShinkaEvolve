# EVOLVE-BLOCK-START
"""
Network telemetry repair algorithm using Hybrid Solidity Dual Consensus.
"""
from typing import Dict, Any, Tuple, List
import math

def repair_network_telemetry(telemetry: Dict[str, Dict[str, Any]], 
                             topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:
    """
    Repair network telemetry using Hybrid Solidity Dual Consensus.
    
    Strategy:
    1. Status Repair: Uses robust traffic detection (Local OR Peer traffic implies UP).
    2. Rate Repair (Iterative):
       - Calculates Flow Imbalance for both Local and Remote routers.
       - 'Solidity' Checks:
         - Local Solidity: Does Peer value fix Local imbalance? (Implies Phantom Loss/Gain)
         - Remote Solidity: Is Peer value consistent with Remote imbalance? (Implies Peer is correct)
       - Logic:
         - Enforce Physics (RX <= Peer TX).
         - If Disagreement (RX < Peer TX):
           - If Peer is needed locally (Local Solidity) -> Repair to Peer.
           - If Peer is NOT needed locally (Local OK) -> Keep Local (Real Loss).
    3. Confidence Calibration:
       - Uses Dual-Source Quality (Local Router Q + Remote Router Q).
       - High confidence when repair aligns with Peer AND both routers are healthy.
    """
    
    # --- Configuration ---
    HARDENING_THRESHOLD = 0.02   # 2% relative error considered 'match'
    STRICT_THRESHOLD = 0.005     # 0.5% tolerance for physical impossibility
    SOLIDITY_THRESHOLD = 0.015   # 1.5% tolerance for flow consistency
    BASE_NOISE_FLOOR = 10.0      # Minimum Mbps
    ITERATIONS = 4               # Convergence count
    
    # --- Helper: Dynamic Noise Floor ---
    def get_noise_floor(rate_a, rate_b=0.0):
        # 0.1% dynamic floor to handle high-speed jitter
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
        # Evidence check
        nf = get_noise_floor(s['rx'], s['tx'])
        local_traffic = s['rx'] > nf or s['tx'] > nf
        
        peer_traffic = False
        peer_down_flag = False
        
        if s['peer'] and s['peer'] in state:
            p = state[s['peer']]
            p_nf = get_noise_floor(p['orig_rx'], p['orig_tx'])
            if p['orig_rx'] > p_nf or p['orig_tx'] > p_nf:
                peer_traffic = True
            if p['orig_status'] == 'down':
                peer_down_flag = True
        
        # Traffic trumps flags
        if local_traffic or peer_traffic:
            s['status'] = 'up'
        elif peer_down_flag and not local_traffic:
            s['status'] = 'down'
        # Else keep original
        
        if s['status'] != 'up':
            s['rx'] = 0.0
            s['tx'] = 0.0

    # --- Step 3: Iterative Rate Repair ---
    for _ in range(ITERATIONS):
        
        # 3.1: Analyze Router Flows
        router_stats = {}
        for r_id, if_ids in topology.items():
            sum_rx = sum(state[i]['rx'] for i in if_ids if i in state)
            sum_tx = sum(state[i]['tx'] for i in if_ids if i in state)
            router_stats[r_id] = {
                'imbalance': sum_rx - sum_tx
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
            
            # Context
            # Local Flow Target: The value required to make Local Imbalance = 0
            # RX_target = RX_current - Imbalance
            local_flow_rx = None
            local_flow_tx = None
            if local_r in router_stats:
                ls = router_stats[local_r]
                local_flow_rx = max(0.0, s['rx'] - ls['imbalance'])
                local_flow_tx = max(0.0, s['tx'] + ls['imbalance'])
            
            # Remote Solidity: Does Peer match Remote Flow?
            peer_tx_solid = False
            peer_rx_solid = False
            
            if has_peer and remote_r in router_stats:
                rs = router_stats[remote_r]
                # Peer TX is an Outgoing interface on Remote. 
                # To balance Remote: New_TX = Old_TX + Imbalance
                p_tx = state[peer_id]['tx']
                p_tx_target = max(0.0, p_tx + rs['imbalance'])
                if calc_error(p_tx, p_tx_target) < SOLIDITY_THRESHOLD:
                    peer_tx_solid = True
                
                # Peer RX is Incoming on Remote.
                # To balance Remote: New_RX = Old_RX - Imbalance
                p_rx = state[peer_id]['rx']
                p_rx_target = max(0.0, p_rx - rs['imbalance'])
                if calc_error(p_rx, p_rx_target) < SOLIDITY_THRESHOLD:
                    peer_rx_solid = True

            # --- RX Repair ---
            # Constraint: RX <= Peer TX
            val_self = s['rx']
            val_peer = state[peer_id]['tx'] if has_peer else None
            
            final_rx = val_self
            
            if val_peer is not None:
                # 1. Physics: Impossible if Self > Peer (significantly)
                if val_self > val_peer * (1.0 + STRICT_THRESHOLD):
                    final_rx = val_peer
                
                # 2. Agreement
                elif calc_error(val_self, val_peer) < HARDENING_THRESHOLD:
                    final_rx = (val_self + val_peer) / 2.0
                
                # 3. Disagreement (Self < Peer)
                else:
                    # Check Local Solidity (Do we need the Peer's bytes?)
                    local_needs_peer = False
                    if local_flow_rx is not None:
                        if calc_error(val_peer, local_flow_rx) < SOLIDITY_THRESHOLD:
                            local_needs_peer = True
                    
                    if local_needs_peer:
                        # Phantom Loss detected.
                        final_rx = val_peer
                    else:
                        # We don't need Peer's bytes to balance.
                        # Likely Real Loss.
                        # Unless Remote is VERY solid and we are not?
                        # Use Arbitration.
                        if local_flow_rx is not None:
                            d_peer = calc_error(val_peer, local_flow_rx)
                            d_self = calc_error(val_self, local_flow_rx)
                            if d_peer < d_self:
                                final_rx = val_peer
                            else:
                                final_rx = val_self
                        else:
                            final_rx = val_peer

            # --- TX Repair ---
            # Constraint: TX >= Peer RX
            val_self = s['tx']
            val_peer = state[peer_id]['rx'] if has_peer else None
            
            final_tx = val_self
            
            if val_peer is not None:
                # 1. Physics: Impossible if Self < Peer
                if val_self < val_peer * (1.0 - STRICT_THRESHOLD):
                    final_tx = val_peer
                
                # 2. Agreement
                elif calc_error(val_self, val_peer) < HARDENING_THRESHOLD:
                    final_tx = (val_self + val_peer) / 2.0
                
                # 3. Disagreement (Self > Peer)
                else:
                    # Check Local Solidity (Do we need to reduce to Peer?)
                    # If Local Flow says "Lower TX is better", then Phantom TX.
                    local_needs_peer = False
                    if local_flow_tx is not None:
                         if calc_error(val_peer, local_flow_tx) < SOLIDITY_THRESHOLD:
                             local_needs_peer = True
                    
                    if local_needs_peer:
                        final_tx = val_peer
                    else:
                        # Arbitration
                        if local_flow_tx is not None:
                            d_peer = calc_error(val_peer, local_flow_tx)
                            d_self = calc_error(val_self, local_flow_tx)
                            if d_peer < d_self:
                                final_tx = val_peer
                            else:
                                final_tx = val_self
                        else:
                            final_tx = val_peer

            next_values[if_id] = {'rx': final_rx, 'tx': final_tx}

        # Apply
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
        
        local_q = final_router_qual.get(s['router'], 0.5)
        remote_q = final_router_qual.get(s['remote_router'], 0.5) if s['remote_router'] else 0.5
        
        def get_conf(final, orig, peer_val, l_q, r_q):
            # 1. Did we change it?
            is_repaired = calc_error(final, orig) > HARDENING_THRESHOLD
            
            # 2. Do we match peer?
            matches_peer = False
            if peer_val is not None and calc_error(final, peer_val) < HARDENING_THRESHOLD:
                matches_peer = True
                
            conf = 1.0
            
            if is_repaired:
                if matches_peer:
                    # Consensus Repair. 
                    # Trust is high, modulated by router health.
                    # Range: 0.88 (Base) + 0.08 (Local) + 0.04 (Remote) -> 1.0
                    conf = 0.88 + (0.08 * l_q) + (0.04 * r_q)
                else:
                    # Flow Arbitration Repair (Not Peer).
                    # We defied Peer and Original. Must have strong Local Flow.
                    if l_q > 0.9:
                        conf = 0.85
                    else:
                        conf = 0.60
            else:
                # Kept Original
                if peer_val is not None and not matches_peer:
                    # Disagreement.
                    # We defied Peer. Valid if Local is strong.
                    if l_q > 0.9:
                        # If Remote is also strong, it's a stalemate -> Lower confidence.
                        if r_q > 0.8:
                            conf = 0.90
                        else:
                            conf = 0.95
                    elif l_q > 0.7:
                        conf = 0.80
                    else:
                        conf = 0.70
                else:
                    # Agreement or No Peer
                    conf = 1.0
            
            return max(0.0, min(1.0, conf))

        rx_conf = get_conf(s['rx'], orig_rx, peer_tx, local_q, remote_q)
        tx_conf = get_conf(s['tx'], orig_tx, peer_rx, local_q, remote_q)
        
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