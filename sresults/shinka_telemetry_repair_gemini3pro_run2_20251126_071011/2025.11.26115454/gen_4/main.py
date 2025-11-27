# EVOLVE-BLOCK-START
"""
Network telemetry repair algorithm that detects and corrects inconsistencies
in network interface telemetry data using topology relationships.

Takes interface telemetry data and detects/repairs inconsistencies based on
network invariants like link symmetry and flow conservation.
"""
from typing import Dict, Any, Tuple, List
import math

def repair_network_telemetry(telemetry: Dict[str, Dict[str, Any]], 
                             topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:
    """
    Repairs network telemetry using an Iterative Analytical Solver that optimizes
    for Link Symmetry and Flow Conservation simultaneously.
    """
    
    # --- Configuration ---
    HARDENING_THRESHOLD = 0.02   # 2% tolerance for measurement noise
    WEIGHT_SELF = 1.0            # Trust in own measurement
    WEIGHT_PEER = 2.0            # Trust in peer measurement (Symmetry)
    WEIGHT_FLOW = 1.0            # Trust in flow conservation
    ITERATIONS = 5               # Convergence passes
    
    # --- Step 1: Initialization & Status Inference ---
    # We build a working state dictionary.
    # Logic: If I see traffic, or my peer sees traffic, the link is UP.
    
    state = {}
    
    # First pass: load data and infer status
    for if_id, data in telemetry.items():
        # Raw inputs
        raw_rx = float(data.get('rx_rate', 0.0))
        raw_tx = float(data.get('tx_rate', 0.0))
        connected_to = data.get('connected_to')
        
        # Check Peer Activity
        peer_active = False
        if connected_to and connected_to in telemetry:
            peer_data = telemetry[connected_to]
            p_rx = float(peer_data.get('rx_rate', 0.0))
            p_tx = float(peer_data.get('tx_rate', 0.0))
            p_status = peer_data.get('interface_status', 'unknown')
            
            # If peer claims UP and sees significant traffic
            if p_status == 'up' and (p_rx > 1.0 or p_tx > 1.0):
                peer_active = True
        
        # Infer Status
        # "Alive" if local traffic > 1Mbps or peer sees us active
        # Or if status says UP and traffic is present
        raw_active = (raw_rx > 1.0 or raw_tx > 1.0)
        is_active = raw_active or peer_active
        
        # We trust the 'up' status if there's any evidence, otherwise 'down'
        if data.get('interface_status') == 'up' or is_active:
            new_status = 'up'
        else:
            new_status = 'down'
        
        # Initial working values
        # If DOWN, force rates to 0 to stop noise propagating to flow calculations
        if new_status == 'down':
            cur_rx, cur_tx = 0.0, 0.0
        else:
            cur_rx, cur_tx = raw_rx, raw_tx
            
        state[if_id] = {
            'rx': cur_rx,
            'tx': cur_tx,
            'status': new_status,
            'orig_rx': raw_rx,
            'orig_tx': raw_tx,
            'orig_status': data.get('interface_status', 'unknown'),
            'peer': connected_to,
            'router': data.get('local_router')
        }

    # --- Step 2: Iterative Analytic Repair ---
    # We solve for TX and RX that minimize error against Self, Peer, and Flow constraints.
    
    for _ in range(ITERATIONS):
        # 2a. Pre-calculate Router Net Flows based on current state
        # RouterNet = Sum(RX) - Sum(TX) for all interfaces on router
        router_nets = {}
        for r_id, if_list in topology.items():
            net_flow = 0.0
            for if_id in if_list:
                if if_id in state:
                    net_flow += (state[if_id]['rx'] - state[if_id]['tx'])
            router_nets[r_id] = net_flow
            
        # 2b. Update each interface
        for if_id, s in state.items():
            if s['status'] == 'down':
                continue
                
            # --- Gather Constraints ---
            
            # Constraint 1 & 2: Self and Peer (Symmetry)
            # We treat these as weighted votes for the "True Value"
            
            # Target for TX: Self TX (w=1) + Peer RX (w=2)
            sum_tx_votes = s['orig_tx'] * WEIGHT_SELF
            w_tx_sum = WEIGHT_SELF
            
            peer_id = s['peer']
            if peer_id and peer_id in state:
                # Peer's RX is a measurement of our TX
                sum_tx_votes += state[peer_id]['rx'] * WEIGHT_PEER
                w_tx_sum += WEIGHT_PEER
                
            avg_tx = sum_tx_votes / w_tx_sum
            
            # Target for RX: Self RX (w=1) + Peer TX (w=2)
            sum_rx_votes = s['orig_rx'] * WEIGHT_SELF
            w_rx_sum = WEIGHT_SELF
            
            if peer_id and peer_id in state:
                # Peer's TX is a measurement of our RX
                sum_rx_votes += state[peer_id]['tx'] * WEIGHT_PEER
                w_rx_sum += WEIGHT_PEER
            
            avg_rx = sum_rx_votes / w_rx_sum
            
            # Constraint 3: Flow Conservation
            # We want to solve for x (TX) and y (RX) minimizing:
            # Cost = Kx(x - X0)^2 + Ky(y - Y0)^2 + W(x - y - D)^2
            # Where:
            #   X0 = avg_tx (Target TX from symmetry/self)
            #   Y0 = avg_rx (Target RX from symmetry/self)
            #   Kx = w_tx_sum
            #   Ky = w_rx_sum
            #   W = WEIGHT_FLOW
            #   D = delta_rest (Flow imbalance from *other* interfaces)
            
            r_id = s['router']
            if r_id and r_id in router_nets:
                # current_diff = RX - TX
                # router_net = current_diff + others_diff
                # others_diff = router_net - current_diff
                # We want: x - y = -others_diff (since incoming = outgoing -> sum(rx)=sum(tx) -> sum(rx-tx)=0)
                # Actually, strictly: sum(rx) = sum(tx)
                # => (rx_i + sum_rx_others) = (tx_i + sum_tx_others)
                # => tx_i - rx_i = sum_rx_others - sum_tx_others
                # Let D = sum_rx_others - sum_tx_others
                # D is exactly (router_nets[r_id] - (s['rx'] - s['tx']))
                
                current_net = s['rx'] - s['tx']
                D = router_nets[r_id] - current_net
                
                # Minimizing: Kx(x - X0)^2 + Ky(y - Y0)^2 + W(x - y - D)^2
                X0 = avg_tx
                Y0 = avg_rx
                Kx = w_tx_sum
                Ky = w_rx_sum
                W = WEIGHT_FLOW
                
                # Analytical Solution (solving the linear system of partial derivatives):
                # (Kx + W)x - Wy = KxX0 + WD
                # -Wx + (Ky + W)y = KyY0 - WD
                
                # Cramer's rule or substitution
                det = (Kx + W) * (Ky + W) - (W * W)
                if det == 0: det = 1.0 # Safety
                
                rhs1 = Kx * X0 + W * D
                rhs2 = Ky * Y0 - W * D
                
                new_tx = (rhs1 * (Ky + W) + rhs2 * W) / det
                new_rx = (rhs1 * W + rhs2 * (Kx + W)) / det

            else:
                # No router context, just use weighted symmetry
                new_tx = avg_tx
                new_rx = avg_rx
            
            # Update State (clamped to positive)
            state[if_id]['tx'] = max(0.0, new_tx)
            state[if_id]['rx'] = max(0.0, new_rx)

    # --- Step 3: Result Formatting & Confidence ---
    result = {}
    
    for if_id, s in state.items():
        repaired_rx = s['rx']
        repaired_tx = s['tx']
        repaired_status = s['status']
        
        # --- Confidence Calculation ---
        # Goal: High confidence if consistent with Peer, or if we kept original.
        # Low confidence if we changed it and it still disagrees or is messy.
        
        def calculate_confidence(original, repaired, peer_val):
            # 1. Magnitude of change
            if max(original, 1.0) == 0: change = 0 
            else: change = abs(original - repaired) / max(original, 1.0)
            
            # 2. Consistency with Peer (using final repaired peer value)
            if peer_val is not None:
                diff_peer = abs(repaired - peer_val) / max(repaired, 1.0)
            else:
                diff_peer = 0.0
            
            # Logic
            if change < HARDENING_THRESHOLD:
                # We essentially kept the original value.
                if peer_val is not None and diff_peer > HARDENING_THRESHOLD * 2:
                    # But it disagrees with peer? 
                    # If we kept it, it means Symmetry weight wasn't enough to pull it 
                    # (maybe strong Flow opposition? or just Weighted avg landed here).
                    # This is suspicious.
                    return 0.75
                return 1.0
            else:
                # We changed the value significantly.
                if peer_val is not None and diff_peer < HARDENING_THRESHOLD * 2:
                    # We changed it to match the peer. Good repair.
                    return 0.95
                elif peer_val is not None:
                    # We changed it, but it still doesn't match peer?
                    # This implies a compromise between Peer and Flow.
                    # Less confident.
                    return max(0.0, 0.7 - diff_peer)
                else:
                    # Changed without peer reference? (e.g. flow only)
                    return 0.6

        # Get Peer Final Values for confidence check
        peer_id = s['peer']
        p_final_rx = None
        p_final_tx = None
        if peer_id and peer_id in state:
            p_final_rx = state[peer_id]['rx']
            p_final_tx = state[peer_id]['tx']

        rx_conf = calculate_confidence(s['orig_rx'], repaired_rx, p_final_tx)
        tx_conf = calculate_confidence(s['orig_tx'], repaired_tx, p_final_rx)
        
        # Status Confidence
        st_conf = 1.0
        if repaired_status != s['orig_status']:
            # Inferred status change
            st_conf = 0.95
        elif repaired_status == 'up' and (repaired_rx < 0.1 and repaired_tx < 0.1):
            # UP but no traffic?
            st_conf = 0.8
            
        result[if_id] = {
            'rx_rate': (s['orig_rx'], repaired_rx, rx_conf),
            'tx_rate': (s['orig_tx'], repaired_tx, tx_conf),
            'interface_status': (s['orig_status'], repaired_status, st_conf),
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

