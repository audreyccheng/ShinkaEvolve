# EVOLVE-BLOCK-START
"""
Network telemetry repair algorithm using Dual-Perspective Flow Consensus.
"""
from typing import Dict, Any, Tuple, List
import math

def repair_network_telemetry(telemetry: Dict[str, Dict[str, Any]], 
                             topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:
    """
    Repair network interface telemetry using Dual-Perspective Flow Consensus.
    
    Strategy:
    1. Status Repair: Infer status from traffic activity, respecting Peer DOWN signals unless contradicted by physics.
    2. Rate Repair (Iterative):
       - Calculate 'Flow Implied' rates for both Local and Remote routers.
       - Use 'Solidity' (agreement with Flow Implied) to identify trustworthy signals.
       - Enforce physical constraints (RX <= Peer TX) with a 'Flow Veto' for broken peer counters.
       - Arbitrate Loss/Error based on which signal (Self vs Peer) aligns best with its respective Flow context.
    3. Confidence:
       - Continuous scoring based on 'Context Quality' (avg flow balance of Local & Remote routers).
    """
    
    # --- Constants ---
    HARDENING_THRESHOLD = 0.02   # 2% tolerance for matches
    IMPOSSIBLE_THRESHOLD = 0.01  # 1% Strict threshold for physics violations
    BASE_NOISE_FLOOR = 10.0      # Minimum meaningful bandwidth (Mbps)
    ITERATIONS = 4               # Convergence iterations
    
    # --- Helpers ---
    def get_noise_floor(val):
        # Scale noise floor: max(10 Mbps, 0.5% of rate)
        # Prevents micro-mismatches on 100G links from triggering errors
        return max(BASE_NOISE_FLOOR, val * 0.005)

    def calc_error(v1, v2):
        nf = max(get_noise_floor(v1), get_noise_floor(v2))
        return abs(v1 - v2) / max(v1, v2, nf)

    # --- Step 1: Initialization & Status Repair ---
    state = {}
    for if_id, data in telemetry.items():
        state[if_id] = {
            'rx': float(data.get('rx_rate', 0.0)),
            'tx': float(data.get('tx_rate', 0.0)),
            'status': data.get('interface_status', 'unknown'),
            'peer': data.get('connected_to'),
            'local_router': data.get('local_router'),
            'remote_router': data.get('remote_router'),
            'orig_rx': float(data.get('rx_rate', 0.0)),
            'orig_tx': float(data.get('tx_rate', 0.0)),
            'orig_status': data.get('interface_status', 'unknown')
        }

    # Robust Status Logic
    for if_id, s in state.items():
        # Traffic Evidence
        nf = get_noise_floor(max(s['rx'], s['tx']))
        local_active = s['rx'] > nf or s['tx'] > nf
        
        peer_down = False
        peer_active = False
        if s['peer'] and s['peer'] in state:
            p = state[s['peer']]
            if p['orig_status'] == 'down':
                peer_down = True
            nf_p = get_noise_floor(max(p['rx'], p['tx']))
            if p['rx'] > nf_p or p['tx'] > nf_p:
                peer_active = True

        # Decision Matrix
        if local_active or peer_active:
            s['status'] = 'up'
        elif peer_down and not local_active:
            s['status'] = 'down'
        # Else keep original (e.g. up but idle)

        # Consistency
        if s['status'] != 'up':
            s['rx'] = 0.0
            s['tx'] = 0.0

    # --- Step 2: Iterative Rate Repair ---
    for _ in range(ITERATIONS):
        # Calculate Router Imbalances
        # Imbalance = Sum(RX) - Sum(TX)
        router_imb = {}
        for r_id, if_ids in topology.items():
            sum_rx = sum(state[i]['rx'] for i in if_ids if i in state)
            sum_tx = sum(state[i]['tx'] for i in if_ids if i in state)
            router_imb[r_id] = sum_rx - sum_tx

        updates = {}
        
        for if_id, s in state.items():
            if s['status'] != 'up':
                updates[if_id] = {'rx': 0.0, 'tx': 0.0}
                continue

            peer_id = s['peer']
            local_r = s['local_router']
            remote_r = s['remote_router'] 
            has_peer = peer_id and peer_id in state
            
            curr_rx = s['rx']
            curr_tx = s['tx']
            
            peer_tx = state[peer_id]['tx'] if has_peer else None
            peer_rx = state[peer_id]['rx'] if has_peer else None
            
            # --- Flow Target Calculation ---
            # Local Target: The value that would make Local Router Imbalance = 0
            # New_RX = Curr_RX - Imbalance (since Imbalance = Rx - Tx)
            local_target_rx = None
            local_target_tx = None
            if local_r in router_imb:
                imb = router_imb[local_r]
                local_target_rx = max(0.0, curr_rx - imb)
                local_target_tx = max(0.0, curr_tx + imb)
            
            # Remote Target (for Peer): The value that makes Remote Router Imbalance = 0
            # Note: We infer this using Peer's router ID.
            # Peer New_RX = Peer_RX - Imb_remote
            remote_target_peer_rx = None
            remote_target_peer_tx = None
            if remote_r in router_imb and has_peer:
                imb_r = router_imb[remote_r]
                remote_target_peer_rx = max(0.0, peer_rx - imb_r)
                remote_target_peer_tx = max(0.0, peer_tx + imb_r)

            # --- RX REPAIR ---
            # Constraint: RX <= Peer TX
            next_rx = curr_rx
            
            if peer_tx is not None:
                # 1. Impossible Check (Strict)
                if curr_rx > peer_tx * (1.0 + IMPOSSIBLE_THRESHOLD):
                    # RX > Peer TX is physically impossible (Phantom Traffic).
                    # EXCEPTION: If Local Flow strongly supports Self (score < Threshold), 
                    # assume Peer Counter is broken/stuck and keep Self.
                    
                    is_flow_validated = False
                    if local_target_rx is not None:
                         if calc_error(curr_rx, local_target_rx) < HARDENING_THRESHOLD:
                             is_flow_validated = True
                    
                    if is_flow_validated:
                        next_rx = curr_rx
                    else:
                        next_rx = peer_tx # Clip to Peer
                
                # 2. Plausible Loss or Match
                else:
                    if calc_error(curr_rx, peer_tx) < HARDENING_THRESHOLD:
                        next_rx = (curr_rx + peer_tx) / 2.0
                    else:
                        # Disagreement (Likely RX < Peer TX).
                        # Arbitration: Who is more "Solid"?
                        
                        score_self_local = calc_error(curr_rx, local_target_rx) if local_target_rx is not None else 1.0
                        score_peer_local = calc_error(peer_tx, local_target_rx) if local_target_rx is not None else 1.0
                        
                        # Check Remote Solidity for Peer
                        score_peer_remote = 1.0
                        if remote_target_peer_tx is not None:
                            score_peer_remote = calc_error(peer_tx, remote_target_peer_tx)
                        
                        if score_peer_local < HARDENING_THRESHOLD:
                            # Peer TX solves my local flow -> Under-counting detected.
                            next_rx = peer_tx
                        elif score_self_local < HARDENING_THRESHOLD:
                            # Local Flow confirms Self -> Real Loss detected.
                            next_rx = curr_rx
                        elif score_peer_remote < HARDENING_THRESHOLD:
                            # Peer is backed by Remote Flow, I am ambiguous -> Trust Peer.
                            next_rx = peer_tx
                        else:
                            # Ambiguous. Default to Peer TX (R3 Symmetry) as it's the source.
                            next_rx = peer_tx

            # --- TX REPAIR ---
            # Constraint: TX >= Peer RX
            next_tx = curr_tx
            
            if peer_rx is not None:
                # 1. Impossible Check (Strict)
                if curr_tx < peer_rx * (1.0 - IMPOSSIBLE_THRESHOLD):
                    # TX < Peer RX is physically impossible.
                    # EXCEPTION: If Local Flow supports Self (e.g. TX=0 and Flow=Balanced), 
                    # assume Peer RX is reading noise/phantom.
                    
                    is_flow_validated = False
                    if local_target_tx is not None:
                        if calc_error(curr_tx, local_target_tx) < HARDENING_THRESHOLD:
                            is_flow_validated = True
                            
                    if is_flow_validated:
                        next_tx = curr_tx
                    else:
                        next_tx = peer_rx
                
                # 2. Plausible Loss (downstream) or Match
                else:
                    if calc_error(curr_tx, peer_rx) < HARDENING_THRESHOLD:
                         next_tx = (curr_tx + peer_rx) / 2.0
                    else:
                        # Disagreement (TX > Peer RX).
                        
                        score_self_local = calc_error(curr_tx, local_target_tx) if local_target_tx is not None else 1.0
                        score_peer_local = calc_error(peer_rx, local_target_tx) if local_target_tx is not None else 1.0
                        
                        score_peer_remote = 1.0
                        if remote_target_peer_rx is not None:
                            score_peer_remote = calc_error(peer_rx, remote_target_peer_rx)
                            
                        if score_peer_local < HARDENING_THRESHOLD:
                            # Peer RX fits my flow better -> Phantom TX.
                            next_tx = peer_rx
                        elif score_self_local < HARDENING_THRESHOLD:
                            # My TX fits my flow -> Real Loss downstream.
                            next_tx = curr_tx
                        elif score_peer_remote < HARDENING_THRESHOLD:
                            # Peer is backed by remote flow -> Trust Peer.
                            next_tx = peer_rx
                        else:
                            next_tx = peer_rx

            updates[if_id] = {'rx': next_rx, 'tx': next_tx}
            
        # Apply synchronous updates
        for if_id, vals in updates.items():
            state[if_id]['rx'] = vals['rx']
            state[if_id]['tx'] = vals['tx']

    # --- Step 3: Confidence Calibration ---
    result = {}
    
    # Calculate Context Quality (Flow Health)
    router_q = {}
    for r_id, if_ids in topology.items():
        sum_rx = sum(state[i]['rx'] for i in if_ids if i in state)
        sum_tx = sum(state[i]['tx'] for i in if_ids if i in state)
        vol = max(sum_rx, sum_tx, BASE_NOISE_FLOOR)
        imb = abs(sum_rx - sum_tx)
        # Quality: 1.0 (perfect) -> 0.0 (bad, >5% imbalance)
        router_q[r_id] = max(0.0, 1.0 - (imb / vol) * 20.0)

    for if_id, s in state.items():
        orig_rx = s['orig_rx']
        orig_tx = s['orig_tx']
        
        peer_id = s['peer']
        has_peer = peer_id and peer_id in state
        
        peer_tx = state[peer_id]['tx'] if has_peer else None
        peer_rx = state[peer_id]['rx'] if has_peer else None
        
        local_q = router_q.get(s['local_router'], 0.5)
        remote_q = router_q.get(s['remote_router'], 0.5)
        # Context Quality: Average health of the link environment
        context_q = (local_q + remote_q) / 2.0
        
        def get_conf(final, orig, peer_val):
            # 1. Did we repair?
            was_repaired = calc_error(final, orig) > HARDENING_THRESHOLD
            
            # 2. Do we match Peer?
            matches_peer = False
            if peer_val is not None:
                if calc_error(final, peer_val) < HARDENING_THRESHOLD:
                    matches_peer = True
            
            conf = 1.0
            
            if was_repaired:
                if matches_peer:
                    # Best Case: We repaired to match Peer, supported by context.
                    # Base 0.90 + boost from context quality
                    conf = 0.90 + (0.09 * context_q) # Max 0.99
                else:
                    # Repaired, but NOT matching Peer? (e.g. Flow Override or Averaging)
                    # Relies heavily on Local Flow Quality
                    conf = 0.70 + (0.20 * local_q) # Max 0.90
            else:
                # Kept Original
                if peer_val is not None and not matches_peer:
                    # Disagreement (e.g. Loss detected and kept).
                    # Valid only if Local Flow confirms it.
                    conf = 0.75 + (0.20 * local_q) # Max 0.95
                else:
                    # Agreement or No Peer
                    conf = 1.0
            
            return conf

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
