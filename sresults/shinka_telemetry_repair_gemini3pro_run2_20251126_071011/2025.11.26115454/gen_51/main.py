# EVOLVE-BLOCK-START
import math
from typing import Dict, Any, Tuple, List

def repair_network_telemetry(telemetry: Dict[str, Dict[str, Any]], 
                             topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:
    """
    Repair network telemetry using Flow-Aware Constraint Consensus and Tiered Linear Confidence.
    
    Strategy:
    1. Status Repair: Infer status from local/remote traffic activity.
    2. Rate Repair (Iterative):
       - Calculate 'Flow Implied' rate that balances the router.
       - Enforce strict physical constraints (RX <= Peer TX) with tight tolerance.
       - Arbitrate loss scenarios (RX < Peer TX) using Flow Implied value.
       - Use 'Solidity' check: if Peer matches Flow Implied, trust it highly.
    3. Confidence Calibration:
       - Tiered Linear model: Discrete base score minus penalty for residual flow imbalance.
    """
    
    # --- Configuration ---
    HARDENING_THRESHOLD = 0.02   # 2% relative error considered 'match'
    STRICT_THRESHOLD = 0.005     # 0.5% for physical impossibility checks
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
        
        # Consistency
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
            # Imbalance = RX - TX
            router_stats[r_id] = {
                'imbalance': sum_rx - sum_tx
            }

        next_values = {}

        for if_id, s in state.items():
            if s['status'] != 'up':
                next_values[if_id] = {'rx': 0.0, 'tx': 0.0}
                continue

            peer_id = s['peer']
            r_id = s['router']
            has_peer = peer_id and peer_id in state
            
            # Flow Implied Values
            flow_rx = None
            flow_tx = None
            
            if r_id in router_stats:
                rs = router_stats[r_id]
                # To balance: RX_new = RX_old - Imbalance
                flow_rx = max(0.0, s['rx'] - rs['imbalance'])
                # To balance: TX_new = TX_old + Imbalance
                flow_tx = max(0.0, s['tx'] + rs['imbalance'])

            # --- RX Repair ---
            # Constraint: RX <= Peer TX
            val_self = s['rx']
            val_peer = state[peer_id]['tx'] if has_peer else None
            
            final_rx = val_self
            
            if val_peer is not None:
                # Check Solidity: Does Peer match Flow Implied?
                peer_matches_flow = False
                if flow_rx is not None:
                    if calc_error(val_peer, flow_rx) < HARDENING_THRESHOLD:
                        peer_matches_flow = True

                # 1. Impossible Case (RX > Peer TX)
                if val_self > val_peer * (1.0 + STRICT_THRESHOLD):
                    final_rx = val_peer
                
                # 2. Agreement
                elif calc_error(val_self, val_peer) < HARDENING_THRESHOLD:
                    final_rx = (val_self + val_peer) / 2.0
                    
                # 3. Loss Case (RX < Peer TX) or Disagreement
                else:
                    if peer_matches_flow:
                        # Flow confirms Peer (Packet Loss detected, Peer is correct source rate)
                        final_rx = val_peer
                    elif flow_rx is not None:
                        err_self_flow = calc_error(val_self, flow_rx)
                        err_peer_flow = calc_error(val_peer, flow_rx)
                        
                        if err_self_flow < err_peer_flow:
                            # Self is closer to Flow -> Real Link Loss
                            final_rx = val_self
                        else:
                            # Peer is closer -> Measurement error
                            final_rx = val_peer
                    else:
                        final_rx = val_peer

            # --- TX Repair ---
            # Constraint: TX >= Peer RX
            val_self = s['tx']
            val_peer = state[peer_id]['rx'] if has_peer else None
            
            final_tx = val_self
            
            if val_peer is not None:
                # Solidity Check
                peer_matches_flow = False
                if flow_tx is not None:
                    if calc_error(val_peer, flow_tx) < HARDENING_THRESHOLD:
                        peer_matches_flow = True

                # 1. Impossible Case (TX < Peer RX)
                if val_self < val_peer * (1.0 - STRICT_THRESHOLD):
                     final_tx = val_peer
                
                # 2. Agreement
                elif calc_error(val_self, val_peer) < HARDENING_THRESHOLD:
                    final_tx = (val_self + val_peer) / 2.0
                    
                # 3. Surplus Case (TX > Peer RX) or Disagreement
                else:
                    if peer_matches_flow:
                        # Flow confirms Peer (Phantom TX repaired)
                        final_tx = val_peer
                    elif flow_tx is not None:
                        err_self_flow = calc_error(val_self, flow_tx)
                        err_peer_flow = calc_error(val_peer, flow_tx)
                        
                        if err_self_flow < err_peer_flow:
                            # Self matches flow -> Loss on wire downstream
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
    
    # Recalculate Final Flow Quality for Penalty
    final_router_imb_ratio = {}
    for r_id, if_ids in topology.items():
        sum_rx = sum(state[i]['rx'] for i in if_ids if i in state)
        sum_tx = sum(state[i]['tx'] for i in if_ids if i in state)
        vol = max(sum_rx, sum_tx, BASE_NOISE_FLOOR)
        final_router_imb_ratio[r_id] = abs(sum_rx - sum_tx) / vol

    for if_id, s in state.items():
        orig_rx = s['orig_rx']
        orig_tx = s['orig_tx']
        
        peer_id = s['peer']
        has_peer = peer_id and peer_id in state
        
        peer_tx = state[peer_id]['tx'] if has_peer else None
        peer_rx = state[peer_id]['rx'] if has_peer else None
        
        r_id = s['router']
        imb_ratio = final_router_imb_ratio.get(r_id, 0.0)

        def get_confidence(final, orig, peer_val, imb_ratio):
            dist_orig = calc_error(final, orig)
            matches_peer = peer_val is not None and calc_error(final, peer_val) < HARDENING_THRESHOLD
            
            # Base Confidence Buckets
            score = 1.0
            
            if dist_orig > HARDENING_THRESHOLD:
                # REPAIRED
                if matches_peer:
                    # Verified by Link Symmetry
                    score = 0.98
                else:
                    # Repaired to Flow or Averaged
                    score = 0.85
            else:
                # KEPT ORIGINAL
                if peer_val is not None and not matches_peer:
                    # Disagreement. Trusted Self.
                    # This is high confidence only if we are balanced (handled by penalty)
                    score = 0.90
                else:
                    # Agreement or No Peer
                    score = 1.0
            
            # Linear Penalty for Residual Imbalance
            # If the router is unbalanced, reduce confidence.
            # E.g. 5% imbalance -> -0.1 confidence
            score -= (imb_ratio * 2.0)
            
            return max(0.0, min(1.0, score))

        rx_conf = get_confidence(s['rx'], orig_rx, peer_tx, imb_ratio)
        tx_conf = get_confidence(s['tx'], orig_tx, peer_rx, imb_ratio)
        
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

