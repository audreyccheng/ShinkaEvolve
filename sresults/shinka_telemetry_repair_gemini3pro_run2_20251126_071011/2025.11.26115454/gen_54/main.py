# EVOLVE-BLOCK-START
"""
Network telemetry repair algorithm using Dual-Flow Consensus and Asymmetric Logic.
"""
from typing import Dict, Any, Tuple, List
import math

def repair_network_telemetry(telemetry: Dict[str, Dict[str, Any]], 
                             topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:
    """
    Repair network telemetry using Dual-Flow Consensus and Asymmetric Logic.
    
    Key Innovations:
    1. Dual-Flow Arbitration: Validates repairs using both Local and Remote router flow states.
       Trusts peers more if they belong to a balanced (healthy) router.
    2. Asymmetric Thresholds: 
       - Strict (0.5%) for physical violations (RX > Peer TX).
       - Loose (2.0%) for agreement/loss scenarios.
    3. Hybrid Confidence:
       - Discrete tiers based on repair source (Peer/Self/Flow).
       - Linear penalty based on residual flow imbalance.
    """
    
    # --- Configuration ---
    AGREEMENT_THRESHOLD = 0.02    # 2% relative error for match
    STRICT_THRESHOLD = 0.005      # 0.5% for physical violations
    BASE_NOISE_FLOOR = 10.0       # Ignore differences below 10 Mbps
    ITERATIONS = 5                # Convergence count
    
    # --- Helpers ---
    def get_noise_floor(v1, v2=0.0):
        # 0.1% dynamic floor to handle high-speed links without ignoring real errors
        return max(BASE_NOISE_FLOOR, max(v1, v2) * 0.001)

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
        # Evidence Gathering
        local_traffic = s['rx'] > BASE_NOISE_FLOOR or s['tx'] > BASE_NOISE_FLOOR
        
        peer_traffic = False
        peer_is_down = False
        if s['peer'] and s['peer'] in state:
            p = state[s['peer']]
            if p['orig_rx'] > BASE_NOISE_FLOOR or p['orig_tx'] > BASE_NOISE_FLOOR:
                peer_traffic = True
            if p['orig_status'] == 'down':
                peer_is_down = True
        
        # Inference Logic
        if local_traffic or peer_traffic:
            # Traffic proves UP
            s['status'] = 'up'
        elif peer_is_down and not local_traffic:
            # Peer DOWN + No Local Traffic -> DOWN
            s['status'] = 'down'
        # Else: Default to original (preserves 'up' but idle, or 'down' correctly)

        # Consistency enforcement
        if s['status'] != 'up':
            s['rx'] = 0.0
            s['tx'] = 0.0

    # --- Step 3: Iterative Rate Repair ---
    for _ in range(ITERATIONS):
        
        # 3.1 Calculate Router Flow States (Local Quality)
        router_stats = {}
        for r_id, if_ids in topology.items():
            sum_rx = sum(state[i]['rx'] for i in if_ids if i in state)
            sum_tx = sum(state[i]['tx'] for i in if_ids if i in state)
            imbalance = sum_rx - sum_tx
            vol = max(sum_rx, sum_tx, BASE_NOISE_FLOOR)
            
            router_stats[r_id] = {
                'imbalance': imbalance,
                'vol': vol,
                'quality': max(0.0, 1.0 - (abs(imbalance) / vol * 10.0)) # 1.0=Good
            }

        next_values = {}

        for if_id, s in state.items():
            if s['status'] != 'up':
                next_values[if_id] = {'rx': 0.0, 'tx': 0.0}
                continue

            peer_id = s['peer']
            r_id = s['router']
            has_peer = peer_id and peer_id in state
            
            # Get Peer Context
            peer_tx = state[peer_id]['tx'] if has_peer else None
            peer_rx = state[peer_id]['rx'] if has_peer else None
            peer_router = state[peer_id]['router'] if has_peer else None
            
            # Assess Remote Quality
            remote_quality = 0.5
            if peer_router and peer_router in router_stats:
                remote_quality = router_stats[peer_router]['quality']

            # Calculate Flow Implied Local Targets
            # Target RX: What RX makes Local Imbalance = 0?
            # Imbalance = SumRX - SumTX.
            # NewImbalance = (SumRX - old_rx + new_rx) - SumTX = 0
            # new_rx = old_rx - Imbalance
            flow_target_rx = None
            flow_target_tx = None
            
            if r_id in router_stats:
                rs = router_stats[r_id]
                flow_target_rx = max(0.0, s['rx'] - rs['imbalance'])
                flow_target_tx = max(0.0, s['tx'] + rs['imbalance'])

            # === RX Repair Logic ===
            curr_rx = s['rx']
            final_rx = curr_rx
            
            if peer_tx is not None:
                # 1. Strict Physical Violation Check (RX > Peer TX)
                if curr_rx > peer_tx * (1.0 + STRICT_THRESHOLD):
                    # Impossible. Clamp to Peer.
                    final_rx = peer_tx
                
                # 2. Agreement Check
                elif calc_error(curr_rx, peer_tx) < AGREEMENT_THRESHOLD:
                    # Consistent. Average to reduce noise.
                    final_rx = (curr_rx + peer_tx) / 2.0
                    
                # 3. Disagreement / Loss Arbitration
                else:
                    # Candidates: curr_rx (Self), peer_tx (Peer)
                    # We check which one is supported by Flow.
                    
                    if flow_target_rx is not None:
                        err_self_flow = calc_error(curr_rx, flow_target_rx)
                        err_peer_flow = calc_error(peer_tx, flow_target_rx)
                        
                        # Dual-Source Heuristic:
                        # If Peer matches Flow -> Trust Peer (Phantom Loss or Measurement Error)
                        if err_peer_flow < AGREEMENT_THRESHOLD:
                             final_rx = peer_tx
                        # If Self matches Flow -> Trust Self (Real Packet Loss)
                        elif err_self_flow < AGREEMENT_THRESHOLD:
                             final_rx = curr_rx
                        else:
                            # Both disagree with Flow (or Flow is ambiguous).
                            # Tie-breaker using Remote Quality.
                            # If Remote is healthy, Peer TX is likely reliable.
                            if remote_quality > 0.8:
                                final_rx = peer_tx
                            elif err_peer_flow < err_self_flow:
                                # Peer is closer to flow balance than Self
                                final_rx = peer_tx
                            else:
                                # Self is closer (e.g. huge imbalance if we use Peer)
                                final_rx = curr_rx
                    else:
                        # No flow info. Trust Peer (Link Symmetry).
                        final_rx = peer_tx

            # === TX Repair Logic ===
            curr_tx = s['tx']
            final_tx = curr_tx
            
            if peer_rx is not None:
                # 1. Strict Physical Violation (TX < Peer RX)
                if curr_tx < peer_rx * (1.0 - STRICT_THRESHOLD):
                    # Impossible. Boost to Peer.
                    final_tx = peer_rx
                    
                # 2. Agreement
                elif calc_error(curr_tx, peer_rx) < AGREEMENT_THRESHOLD:
                    final_tx = (curr_tx + peer_rx) / 2.0
                    
                # 3. Disagreement
                else:
                    if flow_target_tx is not None:
                        err_self_flow = calc_error(curr_tx, flow_target_tx)
                        err_peer_flow = calc_error(peer_rx, flow_target_tx)
                        
                        if err_peer_flow < AGREEMENT_THRESHOLD:
                            # Peer matches Flow (Phantom TX)
                            final_tx = peer_rx
                        elif err_self_flow < AGREEMENT_THRESHOLD:
                            # Self matches Flow (Real Loss downstream)
                            final_tx = curr_tx
                        else:
                            # Ambiguous.
                            # If Remote is healthy, Peer RX is reliable observation.
                            if remote_quality > 0.8:
                                final_tx = peer_rx
                            elif err_peer_flow < err_self_flow:
                                final_tx = peer_rx
                            else:
                                final_tx = curr_tx
                    else:
                        final_tx = peer_rx

            next_values[if_id] = {'rx': final_rx, 'tx': final_tx}

        # Apply updates
        for if_id, vals in next_values.items():
            state[if_id]['rx'] = vals['rx']
            state[if_id]['tx'] = vals['tx']

    # --- Step 4: Confidence Calibration ---
    result = {}
    
    # Final Flow Assessment
    router_balance_score = {}
    for r_id, if_ids in topology.items():
        sum_rx = sum(state[i]['rx'] for i in if_ids if i in state)
        sum_tx = sum(state[i]['tx'] for i in if_ids if i in state)
        vol = max(sum_rx, sum_tx, BASE_NOISE_FLOOR)
        # Ratio of imbalance
        imb_ratio = abs(sum_rx - sum_tx) / vol
        router_balance_score[r_id] = imb_ratio

    for if_id, s in state.items():
        orig_rx = s['orig_rx']
        orig_tx = s['orig_tx']
        peer_id = s['peer']
        has_peer = peer_id and peer_id in state
        r_id = s['router']
        
        peer_tx = state[peer_id]['tx'] if has_peer else None
        peer_rx = state[peer_id]['rx'] if has_peer else None
        
        # Residual Imbalance Penalty
        imb_penalty = router_balance_score.get(r_id, 0.0) * 2.0 # Penalty multiplier
        
        def get_confidence(final, orig, peer_val):
            # Base Confidence Logic
            dist_orig = calc_error(final, orig)
            
            matches_peer = False
            if peer_val is not None and calc_error(final, peer_val) < AGREEMENT_THRESHOLD:
                matches_peer = True
            
            # Tiered Scoring
            score = 1.0
            
            if dist_orig > AGREEMENT_THRESHOLD:
                # REPAIRED
                if matches_peer:
                    # High confidence: Matched Peer
                    score = 0.98
                else:
                    # Repaired to value distinct from Peer? 
                    # (e.g. Flow Override or Averaging)
                    score = 0.75
            else:
                # KEPT ORIGINAL
                if peer_val is not None and not matches_peer:
                    # Disagreement. Kept Self.
                    # This implies we trusted Self + Flow over Peer.
                    score = 0.85
                else:
                    # Agreement or No Peer
                    score = 1.0
            
            # Apply Residual Penalty
            # If the router is still imbalanced, reduce confidence in all its values
            # Clamp between 0.0 and 1.0
            final_score = max(0.0, min(1.0, score - imb_penalty))
            
            return final_score

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