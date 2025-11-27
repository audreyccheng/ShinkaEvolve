# EVOLVE-BLOCK-START
"""
Network telemetry repair algorithm using Trust-Weighted Flow Consensus.
1. Establishes effective status/rates.
2. Identifies 'Trust Anchors' (fully monitored routers).
3. Iteratively repairs rates by anchoring to Trust Hints when available.
4. Calibrates confidence based on the verification source (Flow vs Symmetry).
"""
from typing import Dict, Any, Tuple, List
import math

def repair_network_telemetry(telemetry: Dict[str, Dict[str, Any]], 
                             topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:
    
    # --- Configuration ---
    HARDENING_THRESHOLD = 0.02   # 2% tolerance
    TRAFFIC_THRESHOLD = 1.0      # 1 Mbps active threshold
    ITERATIONS = 5               # Refinement passes
    
    # --- Phase 1: Initialization & Status Repair ---
    state = {}
    
    for iface_id, data in telemetry.items():
        raw_rx = data.get('rx_rate', 0.0)
        raw_tx = data.get('tx_rate', 0.0)
        raw_status = data.get('interface_status', 'unknown')
        
        peer_id = data.get('connected_to')
        peer_data = telemetry.get(peer_id) if (peer_id and peer_id in telemetry) else {}
        
        # Traffic Evidence
        signals = [raw_rx, raw_tx, peer_data.get('rx_rate', 0.0), peer_data.get('tx_rate', 0.0)]
        max_traffic = max(signals) if signals else 0.0
        
        # Status Inference
        status = raw_status
        status_conf = 1.0
        
        if max_traffic > TRAFFIC_THRESHOLD:
            if raw_status != 'up':
                status = 'up'
                status_conf = 0.95
        elif raw_status == 'up' and peer_data.get('interface_status') == 'down':
            status = 'down'
            status_conf = 0.8
            
        # Initial Rate Beliefs
        if status == 'down':
            cur_rx, cur_tx = 0.0, 0.0
        else:
            cur_rx = raw_rx if raw_rx > 0 else 0.0
            cur_tx = raw_tx if raw_tx > 0 else 0.0
            
        state[iface_id] = {
            'rx': cur_rx,
            'tx': cur_tx,
            'status': status,
            'status_conf': status_conf,
            'orig_rx': raw_rx,
            'orig_tx': raw_tx,
            'orig_status': raw_status,
            'peer_id': peer_id,
            'local_router': data.get('local_router'),
            'remote_router': data.get('remote_router')
        }
        
    # --- Phase 2: Iterative Consensus ---
    
    # Identify Trust Anchors (Routers with full observability)
    # Flow hints from these routers are treated as "Constraints" rather than just votes.
    trust_anchors = set()
    for r_id, ifaces in topology.items():
        if all(i in telemetry for i in ifaces):
            trust_anchors.add(r_id)
            
    for _ in range(ITERATIONS):
        next_state = {}
        
        # Calculate Flow Hints from Trust Anchors
        router_balances = {}
        for r_id in trust_anchors:
            ifaces = topology[r_id]
            sum_rx = sum(state[i]['rx'] for i in ifaces)
            sum_tx = sum(state[i]['tx'] for i in ifaces)
            router_balances[r_id] = {'rx': sum_rx, 'tx': sum_tx}
            
        for iface_id, curr in state.items():
            if curr['status'] == 'down':
                next_state[iface_id] = {'rx': 0.0, 'tx': 0.0}
                continue
                
            peer_id = curr['peer_id']
            has_peer = peer_id and peer_id in state
            
            # --- Logic to Resolve One Direction ---
            def resolve_direction(local_val, peer_val, is_rx):
                # 1. Gather Evidence
                # Hint: What should this value be to balance the LOCAL router?
                local_hint = None
                r_id = curr.get('local_router')
                if r_id and r_id in router_balances:
                    rb = router_balances[r_id]
                    if is_rx:
                        # RX must match Total TX
                        local_hint = max(0.0, rb['tx'] - (rb['rx'] - local_val))
                    else:
                        # TX must match Total RX
                        local_hint = max(0.0, rb['rx'] - (rb['tx'] - local_val))
                
                # Peer Hint: What should this value be to balance the REMOTE router?
                # Note: If is_rx=True (Local RX), we are looking for Peer TX.
                # Remote router balance implies a value for Peer TX.
                remote_hint = None
                rr_id = curr.get('remote_router')
                if rr_id and rr_id in router_balances:
                    rb_r = router_balances[rr_id]
                    if is_rx:
                        # We are receiving. Peer is sending. Peer TX Hint.
                        # Peer TX = Remote RX Sum - (Remote TX Sum - Peer TX) -> No.
                        # Remote Router Balance: Sum In = Sum Out.
                        # Peer TX is an Out from Remote.
                        # Peer TX = Sum In(Remote) - Sum Out(Remote, excluding Peer TX)
                        # Peer Val is currently being used in Sum Out.
                        remote_hint = max(0.0, rb_r['rx'] - (rb_r['tx'] - peer_val))
                    else:
                        # We are sending. Peer is receiving. Peer RX Hint.
                        remote_hint = max(0.0, rb_r['tx'] - (rb_r['rx'] - peer_val))
                
                # 2. Consensus Logic
                
                # A. Symmetry Check (Base Baseline)
                denom = max(local_val, peer_val, 1.0)
                diff_sym = abs(local_val - peer_val) / denom
                
                if diff_sym <= HARDENING_THRESHOLD:
                    return (local_val + peer_val) / 2.0
                
                # B. Flow Verification / Arbitration
                # Identify a "Target" value from Hints
                targets = []
                if local_hint is not None: targets.append(local_hint)
                if remote_hint is not None: targets.append(remote_hint)
                
                if not targets:
                    # No Hints. Fallback to heuristic.
                    # Heuristic: Trust positive value over zero (assuming sensor failure)
                    if local_val < TRAFFIC_THRESHOLD and peer_val > TRAFFIC_THRESHOLD:
                        return peer_val
                    if peer_val < TRAFFIC_THRESHOLD and local_val > TRAFFIC_THRESHOLD:
                        return local_val
                    # Default to local
                    return local_val
                
                # We have targets. Average them to get consensus target.
                target = sum(targets) / len(targets)
                
                # Check "Double Dead" (Synthesis)
                # If measurements are ~0 but Target is high.
                if local_val < TRAFFIC_THRESHOLD and peer_val < TRAFFIC_THRESHOLD:
                    if target > 5.0:
                        return target
                    return 0.0
                
                # Arbitration: Pick measurement closest to Target
                denom_l = max(local_val, target, 1.0)
                err_l = abs(local_val - target) / denom_l
                
                denom_p = max(peer_val, target, 1.0)
                err_p = abs(peer_val - target) / denom_p
                
                if err_l < err_p:
                    return local_val
                else:
                    return peer_val

            # Resolve RX
            peer_tx = state[peer_id]['tx'] if has_peer else curr['rx']
            next_rx = resolve_direction(curr['rx'], peer_tx, is_rx=True)
            
            # Resolve TX
            peer_rx = state[peer_id]['rx'] if has_peer else curr['tx']
            next_tx = resolve_direction(curr['tx'], peer_rx, is_rx=False)
            
            next_state[iface_id] = {'rx': next_rx, 'tx': next_tx}
            
        # Apply Updates
        for i_id, vals in next_state.items():
            state[i_id]['rx'] = vals['rx']
            state[i_id]['tx'] = vals['tx']
            
    # --- Phase 3: Final Calibration ---
    result = {}
    
    # Final Balances for scoring
    final_balances = {}
    for r_id in trust_anchors:
        ifaces = topology[r_id]
        sum_rx = sum(state[i]['rx'] for i in ifaces)
        sum_tx = sum(state[i]['tx'] for i in ifaces)
        final_balances[r_id] = {'rx': sum_rx, 'tx': sum_tx}
        
    for iface_id, data in state.items():
        final_rx = data['rx']
        final_tx = data['tx']
        peer_id = data['peer_id']
        has_peer = peer_id and peer_id in state
        
        # Calibration Helper
        def calibrate(val, peer_val, hint_val, is_down, stat_conf):
            if is_down:
                return stat_conf if val > TRAFFIC_THRESHOLD else stat_conf # Inherit status confidence
            
            # Calculate Errors
            err_sym = 1.0
            if has_peer:
                denom = max(val, peer_val, 1.0)
                err_sym = abs(val - peer_val) / denom
            else:
                err_sym = 0.0 # No peer -> perfect symmetry by definition
            
            err_flow = None
            if hint_val is not None:
                denom = max(val, hint_val, 1.0)
                err_flow = abs(val - hint_val) / denom
                
            # Tiered Confidence Logic
            base_score = 0.5
            penalty = 0.0
            
            if err_flow is not None:
                if err_flow < HARDENING_THRESHOLD:
                    # Tier 1: Flow Verified (Gold Standard)
                    base_score = 1.0
                    penalty = err_flow * 2.0 # Slight penalty for imperfection
                else:
                    # Tier 3: Flow Available but Mismatched (Constraint violation)
                    # We are essentially guessing/synthesizing
                    base_score = 0.85
                    penalty = err_flow * 1.5
            elif err_sym < HARDENING_THRESHOLD:
                # Tier 2: Symmetry Verified (Silver Standard)
                base_score = 0.95
                penalty = err_sym * 2.0
            else:
                # Tier 4: Heuristic / Broken
                base_score = 0.6
                penalty = min(err_sym, 1.0) * 0.5
            
            # Double Dead Special Case
            # If we synthesized value (val > 0, peer ~0, hint > 0), and err_flow is low
            if val > 5.0 and peer_val < 1.0 and err_flow is not None and err_flow < 0.05:
                # We trusted flow over peer.
                base_score = 0.9
            
            return max(0.0, base_score - penalty) * stat_conf

        # Get Final Hints
        r_id = data['local_router']
        hint_rx = None
        hint_tx = None
        if r_id and r_id in final_balances:
            rb = final_balances[r_id]
            hint_rx = max(0.0, rb['tx'] - (rb['rx'] - final_rx))
            hint_tx = max(0.0, rb['rx'] - (rb['tx'] - final_tx))
            
        peer_tx = state[peer_id]['tx'] if has_peer else final_rx
        peer_rx = state[peer_id]['rx'] if has_peer else final_tx
        
        conf_rx = calibrate(final_rx, peer_tx, hint_rx, data['status']=='down', data['status_conf'])
        conf_tx = calibrate(final_tx, peer_rx, hint_tx, data['status']=='down', data['status_conf'])
        
        result[iface_id] = {
            'rx_rate': (data['orig_rx'], final_rx, conf_rx),
            'tx_rate': (data['orig_tx'], final_tx, conf_tx),
            'interface_status': (data['orig_status'], data['status'], data['status_conf']),
            'connected_to': peer_id,
            'local_router': r_id,
            'remote_router': data['remote_router']
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