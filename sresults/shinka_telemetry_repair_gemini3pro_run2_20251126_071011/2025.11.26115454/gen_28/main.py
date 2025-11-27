# EVOLVE-BLOCK-START
"""
Network telemetry repair algorithm that detects and corrects inconsistencies
in network interface telemetry data using topology relationships.
"""
from typing import Dict, Any, Tuple, List
import math

def repair_network_telemetry(telemetry: Dict[str, Dict[str, Any]], 
                             topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:
    """
    Repair network interface telemetry using Asymmetric Constraint Consensus.
    
    Strategies:
    1. Status: Peer DOWN strongly implies Local DOWN (zero traffic).
    2. Rate Repair:
       - Enforce 'Physically Impossible' bounds (RX <= Peer TX) strictly.
       - Allow 'Plausible Loss' (RX < Peer TX) unless Flow Conservation suggests under-counting.
       - Use Flow Imbalance at router level to arbitrate ambiguous cases.
    3. Calibration:
       - Confidence decays based on residual error and support from Peer/Flow.
    """
    
    # --- Constants ---
    HARDENING_THRESHOLD = 0.02   # 2% tolerance
    NOISE_FLOOR = 10.0           # Ignore variations below 10 Mbps
    ITERATIONS = 3               # Allow convergence
    
    # --- Helper Functions ---
    def safe_get(d, key, default):
        val = d.get(key)
        return float(val) if val is not None else default

    def calc_error(v1, v2):
        return abs(v1 - v2) / max(v1, v2, NOISE_FLOOR)

    # --- Step 1: Initialization & Status ---
    state = {}
    for if_id, data in telemetry.items():
        state[if_id] = {
            'rx': safe_get(data, 'rx_rate', 0.0),
            'tx': safe_get(data, 'tx_rate', 0.0),
            'status': data.get('interface_status', 'unknown'),
            'peer': data.get('connected_to'),
            'router': data.get('local_router'),
            'orig_rx': safe_get(data, 'rx_rate', 0.0),
            'orig_tx': safe_get(data, 'tx_rate', 0.0),
            'orig_status': data.get('interface_status', 'unknown')
        }

    # Robust Status Inference
    for if_id, s in state.items():
        peer_id = s['peer']
        
        # Check Traffic
        local_traffic = s['rx'] > NOISE_FLOOR or s['tx'] > NOISE_FLOOR
        
        peer_down = False
        peer_traffic = False
        if peer_id and peer_id in telemetry:
            p = telemetry[peer_id]
            if p.get('interface_status') == 'down':
                peer_down = True
            if safe_get(p, 'rx_rate', 0.0) > NOISE_FLOOR or safe_get(p, 'tx_rate', 0.0) > NOISE_FLOOR:
                peer_traffic = True
        
        # Logic: 
        # 1. If Peer is explicitly DOWN, I should be DOWN, unless I see massive traffic (safety valve).
        # 2. If I see traffic (or peer sees traffic), I must be UP.
        
        if peer_down and not local_traffic:
            s['status'] = 'down'
        elif local_traffic or peer_traffic:
            s['status'] = 'up'
        # Else keep original status (e.g. up but idle)
        
        # Consistency
        if s['status'] != 'up':
            s['rx'] = 0.0
            s['tx'] = 0.0

    # --- Step 2: Rate Repair Iterations ---
    for _ in range(ITERATIONS):
        # 2.1 Calculate Flow Balance per Router
        router_flow = {}
        for r_id, if_list in topology.items():
            r_rx = sum(state[i]['rx'] for i in if_list if i in state)
            r_tx = sum(state[i]['tx'] for i in if_list if i in state)
            # Flow Balance = RX - TX. Positive = Accumulation/Drop? Negative = Creation?
            # Ideal is 0.
            router_flow[r_id] = r_rx - r_tx

        updates = {}
        for if_id, s in state.items():
            if s['status'] != 'up':
                updates[if_id] = {'rx': 0.0, 'tx': 0.0}
                continue

            peer_id = s['peer']
            r_id = s['router']
            has_peer = peer_id and peer_id in state
            
            curr_rx = s['rx']
            curr_tx = s['tx']
            
            # Retrieve Peer Values (Use current state for fastest convergence)
            peer_tx = state[peer_id]['tx'] if has_peer else None
            peer_rx = state[peer_id]['rx'] if has_peer else None
            
            # --- RX Repair Strategy ---
            # Ideal: RX_self == TX_peer
            # Constraint: RX_self <= TX_peer
            
            next_rx = curr_rx
            if peer_tx is not None:
                err = calc_error(curr_rx, peer_tx)
                if err < HARDENING_THRESHOLD:
                    next_rx = (curr_rx + peer_tx) / 2.0
                else:
                    # Disagreement
                    if curr_rx > peer_tx:
                        # Impossible (I received more than they sent).
                        # Only valid if Flow demands High RX.
                        # Check flow: If we lower RX to PeerTX, does flow error increase?
                        if r_id in router_flow:
                            # Current Imbalance = Flow
                            # Proposed Imbalance = Flow - curr_rx + peer_tx (RX decreases, so Net decreases)
                            # If Flow is negative (TX > RX), reducing RX makes it worse.
                            # If Flow is positive (RX > TX), reducing RX helps.
                            net = router_flow[r_id]
                            # Simple heuristic: Does Peer TX minimize abs(net)?
                            # Error with Self: abs(net)
                            # Error with Peer: abs(net - curr_rx + peer_tx)
                            if abs(net - curr_rx + peer_tx) < abs(net):
                                next_rx = peer_tx # Peer is better for flow
                            else:
                                # Self is better for flow? 
                                # But Self is physically impossible (RX > PeerTX).
                                # Trust Physics over Flow (Flow might be messy due to other links).
                                # Use Peer TX, but maybe soften if flow is strongly opposed?
                                # Let's stick to Physics.
                                next_rx = peer_tx
                        else:
                            next_rx = peer_tx
                    else:
                        # curr_rx < peer_tx (Possible Loss).
                        # Check Flow: Do we need more RX?
                        if r_id in router_flow:
                            net = router_flow[r_id]
                            # If net < 0 (TX > RX), we are creating packets. We need more RX.
                            # If increasing RX to PeerTX helps balance:
                            if abs(net - curr_rx + peer_tx) < abs(net):
                                next_rx = peer_tx # Assume it was error/loss we should count
                            else:
                                next_rx = curr_rx # Keep loss (it balances flow)
                        else:
                            # No flow info. Assume Loss is real? 
                            # Or repair to Peer TX? 
                            # Usually assume loss is real (keep Self).
                            # But if error is huge, maybe repair?
                            # Let's keep Self for loss scenarios without flow evidence.
                            next_rx = curr_rx

            # --- TX Repair Strategy ---
            # Ideal: TX_self == RX_peer
            # Constraint: TX_self >= RX_peer
            
            next_tx = curr_tx
            if peer_rx is not None:
                err = calc_error(curr_tx, peer_rx)
                if err < HARDENING_THRESHOLD:
                    next_tx = (curr_tx + peer_rx) / 2.0
                else:
                    if curr_tx < peer_rx:
                        # Impossible (I sent less than they received).
                        # Must repair upwards.
                        next_tx = peer_rx
                    else:
                        # curr_tx > peer_rx (Possible Loss).
                        # Check Flow: Do we need less TX?
                        if r_id in router_flow:
                            net = router_flow[r_id]
                            # TX is output. 
                            # If net > 0 (RX > TX), we are dropping at router. 
                            # If net < 0 (TX > RX), we are creating. High TX is suspect.
                            
                            # Error with Self: abs(net)
                            # Error with Peer: abs(net + curr_tx - peer_rx) (TX decreases, net increases)
                            
                            if abs(net + curr_tx - peer_rx) < abs(net):
                                next_tx = peer_rx # Lowering TX helps balance
                            else:
                                next_tx = curr_tx # Keeping High TX helps balance
                        else:
                            next_tx = curr_tx

            updates[if_id] = {'rx': next_rx, 'tx': next_tx}

        # Apply updates
        for if_id, vals in updates.items():
            state[if_id]['rx'] = vals['rx']
            state[if_id]['tx'] = vals['tx']

    # --- Step 3: Confidence Calibration ---
    result = {}
    
    # Assess Final Flow Quality
    router_scores = {}
    for r_id, if_list in topology.items():
        r_rx = sum(state[i]['rx'] for i in if_list if i in state)
        r_tx = sum(state[i]['tx'] for i in if_list if i in state)
        mx = max(r_rx, r_tx, NOISE_FLOOR)
        imb = abs(r_rx - r_tx) / mx
        # Score: 1.0 = good, 0.0 = bad (>10% imbalance)
        router_scores[r_id] = max(0.0, 1.0 - imb * 10.0)

    for if_id, s in state.items():
        orig_rx = s['orig_rx']
        orig_tx = s['orig_tx']
        peer_id = s['peer']
        has_peer = peer_id and peer_id in state
        r_id = s['router']
        flow_score = router_scores.get(r_id, 0.5)

        peer_tx = state[peer_id]['tx'] if has_peer else None
        peer_rx = state[peer_id]['rx'] if has_peer else None

        def get_conf(final, orig, peer_val, flow_q):
            # 1. Did we change significant value?
            is_changed = calc_error(final, orig) > HARDENING_THRESHOLD
            
            # 2. Agreement with Peer
            peer_agrees = False
            if peer_val is not None:
                if calc_error(final, peer_val) < HARDENING_THRESHOLD:
                    peer_agrees = True
            
            # Base Confidence
            conf = 1.0
            
            if is_changed:
                if peer_agrees:
                    # Supported repair
                    conf = 0.90 + (0.09 * flow_q) # 0.90 to 0.99
                else:
                    # Unsupported repair (e.g. flow forced, or impossible constraint forced)
                    if flow_q > 0.8:
                        conf = 0.85
                    else:
                        conf = 0.60
            else:
                # Kept Original
                if peer_val is not None and not peer_agrees:
                    # Disagreement remains (e.g. trusted Loss)
                    if flow_q > 0.8:
                        conf = 0.95 # Flow confirms our value
                    else:
                        conf = 0.80 # Ambiguous
                else:
                    conf = 1.0
            
            return conf

        rx_conf = get_conf(s['rx'], orig_rx, peer_tx, flow_score)
        tx_conf = get_conf(s['tx'], orig_tx, peer_rx, flow_score)
        
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