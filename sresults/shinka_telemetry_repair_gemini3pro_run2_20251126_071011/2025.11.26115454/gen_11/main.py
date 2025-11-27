# EVOLVE-BLOCK-START
"""
Network telemetry repair algorithm that detects and corrects inconsistencies
in network interface telemetry data using topology relationships.

Takes interface telemetry data and detects/repairs inconsistencies based on
network invariants like link symmetry and flow conservation.
"""
from typing import Dict, Any, Tuple, List


def repair_network_telemetry(telemetry: Dict[str, Dict[str, Any]],
                             topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:
    """
    Repair network interface telemetry by detecting and correcting inconsistencies.

    Core principle: Use network invariants to validate and repair telemetry:
    1. Link Symmetry (R3): my_tx_rate ≈ their_rx_rate for connected interfaces
    2. Flow Conservation (R1): Sum(incoming traffic) = Sum(outgoing traffic) at each router
    3. Interface Consistency: Status should be consistent across connected pairs

    Args:
        telemetry: Dictionary where key is interface_id and value contains:
            - interface_status: "up" or "down"
            - rx_rate: receive rate in Mbps
            - tx_rate: transmit rate in Mbps
            - connected_to: interface_id this interface connects to
            - local_router: router_id this interface belongs to
            - remote_router: router_id on the other side
        topology: Dictionary where key is router_id and value contains a list of interface_ids

    Returns:
        Dictionary with same structure but telemetry values become tuples of:
        (original_value, repaired_value, confidence_score)
        where confidence ranges from 0.0 (very uncertain) to 1.0 (very confident)
    """

    HARDENING_THRESHOLD = 0.02
    
    # 1. Initialize State
    # We store mutable current estimates
    state = {}
    for if_id, data in telemetry.items():
        state[if_id] = {
            'rx': float(data.get('rx_rate', 0.0)),
            'tx': float(data.get('tx_rate', 0.0)),
            'status': data.get('interface_status', 'unknown'),
            'rx_conf': 0.5,
            'tx_conf': 0.5,
            'status_conf': 1.0,
            'rx_fixed': False, # True if validated by symmetry
            'tx_fixed': False,
            'orig': data
        }

    # 2. Status Repair
    # Process pairs to ensure consistency
    processed_pairs = set()
    sorted_ifs = sorted(state.keys()) # Deterministic order
    
    for if_id in sorted_ifs:
        if if_id in processed_pairs: continue
        
        s = state[if_id]
        peer_id = s['orig'].get('connected_to')
        
        if peer_id and peer_id in state:
            processed_pairs.add(if_id)
            processed_pairs.add(peer_id)
            p = state[peer_id]
            
            # Resolve Status
            if s['status'] != p['status']:
                # Traffic check
                max_traffic = max(s['rx'], s['tx'], p['rx'], p['tx'])
                if max_traffic > 1.0:
                    final_status = 'up'
                else:
                    final_status = 'down'
                
                s['status'] = final_status
                p['status'] = final_status
                s['status_conf'] = 0.8
                p['status_conf'] = 0.8
            
            # Enforce Down
            if s['status'] == 'down':
                for node in [s, p]:
                    node['rx'] = 0.0
                    node['tx'] = 0.0
                    node['rx_conf'] = 1.0
                    node['tx_conf'] = 1.0
                    node['rx_fixed'] = True
                    node['tx_fixed'] = True
    
    # 3. Symmetry Check (High Confidence Anchors)
    for if_id in sorted_ifs:
        s = state[if_id]
        if s['status'] == 'down': continue
        
        peer_id = s['orig'].get('connected_to')
        if not peer_id or peer_id not in state: continue
        p = state[peer_id]
        
        # Check TX -> Peer RX
        if not s['tx_fixed']:
            tx_val = float(s['orig'].get('tx_rate', 0.0))
            peer_rx_val = float(p['orig'].get('rx_rate', 0.0))
            denom = max(tx_val, peer_rx_val, 1.0)
            
            if abs(tx_val - peer_rx_val) / denom <= HARDENING_THRESHOLD:
                avg = (tx_val + peer_rx_val) / 2.0
                s['tx'] = avg
                p['rx'] = avg
                s['tx_conf'] = 1.0
                p['rx_conf'] = 1.0
                s['tx_fixed'] = True
                p['rx_fixed'] = True
                
        # RX -> Peer TX check
        # Explicitly check "My RX" vs "Peer TX"
        if not s['rx_fixed']:
            rx_val = float(s['orig'].get('rx_rate', 0.0))
            peer_tx_val = float(p['orig'].get('tx_rate', 0.0))
            denom = max(rx_val, peer_tx_val, 1.0)
            
            if abs(rx_val - peer_tx_val) / denom <= HARDENING_THRESHOLD:
                avg = (rx_val + peer_tx_val) / 2.0
                s['rx'] = avg
                p['tx'] = avg
                s['rx_conf'] = 1.0
                p['tx_conf'] = 1.0
                s['rx_fixed'] = True
                p['tx_fixed'] = True

    # Helper for Flow Balance Sums
    def get_router_sums(router_id):
        if not router_id or router_id not in topology:
            return None, None
        sum_in = 0.0
        sum_out = 0.0
        valid_ifs = 0
        for rid in topology[router_id]:
            if rid in state:
                sum_in += state[rid]['rx']
                sum_out += state[rid]['tx']
                valid_ifs += 1
        if valid_ifs == 0: return None, None
        return sum_in, sum_out

    # 4. Iterative Flow Repair
    # Propagate constraints
    ITERATIONS = 3
    for _ in range(ITERATIONS):
        # We iterate over links by looking at each interface's TX direction
        # This covers all flows in the network exactly once per direction
        for if_id in sorted_ifs:
            s = state[if_id]
            if s['status'] == 'down' or s['tx_fixed']: continue
            
            peer_id = s['orig'].get('connected_to')
            if not peer_id or peer_id not in state: continue
            p = state[peer_id]
            
            # Flow: Local(TX) -> Remote(RX)
            # Candidates
            c1 = float(s['orig'].get('tx_rate', 0.0))
            c2 = float(p['orig'].get('rx_rate', 0.0))
            
            r_local = s['orig'].get('local_router')
            r_remote = s['orig'].get('remote_router')
            
            cost1 = 0.0
            cost2 = 0.0
            constraints = 0
            
            # Local Router Constraint (Source)
            lin, lout = get_router_sums(r_local)
            if lin is not None:
                # Flow Conservation: Sum(In) = Sum(Out)
                # Current lout includes current s['tx']. We want to test candidates.
                # Target TX = Sum(In) - (Sum(Out) - Current_TX)
                target_tx = max(0.0, lin - (lout - s['tx']))
                cost1 += abs(c1 - target_tx)
                cost2 += abs(c2 - target_tx)
                constraints += 1
                
            # Remote Router Constraint (Dest)
            rin, rout = get_router_sums(r_remote)
            if rin is not None:
                # Target RX = Sum(Out) - (Sum(In) - Current_RX)
                # Current_RX is p['rx']
                target_rx = max(0.0, rout - (rin - p['rx']))
                cost1 += abs(c1 - target_rx)
                cost2 += abs(c2 - target_rx)
                constraints += 1
            
            # Selection
            if constraints == 0:
                # No topology info, fallback to average with low confidence
                best_val = (c1 + c2) / 2.0
                conf = 0.5
            else:
                if cost1 < cost2:
                    best_val = c1
                    best_err = cost1
                    other_err = cost2
                else:
                    best_val = c2
                    best_err = cost2
                    other_err = cost1
                
                # Update estimates
                s['tx'] = best_val
                p['rx'] = best_val
                
                # Confidence Calibration
                flow_mag = max(best_val, 1.0)
                rel_err = best_err / (flow_mag * constraints) # Normalize error per constraint
                
                if rel_err < HARDENING_THRESHOLD:
                    # Good fit
                    if abs(c1 - c2) / flow_mag < HARDENING_THRESHOLD:
                         # Candidates agreed (rare here since symmetry failed, but possible)
                         conf = 0.95
                    elif best_err < other_err * 0.5:
                         # Distinct winner
                         conf = 0.95
                    else:
                         # Winner but close call
                         conf = 0.8
                else:
                    # Poor fit - degrade confidence
                    conf = max(0.0, 0.8 - (rel_err * 5.0))
                
                s['tx_conf'] = conf
                p['rx_conf'] = conf

    # Final Result Construction
    result = {}
    for if_id, s in state.items():
        result[if_id] = {
            'rx_rate': (s['orig'].get('rx_rate', 0.0), s['rx'], s['rx_conf']),
            'tx_rate': (s['orig'].get('tx_rate', 0.0), s['tx'], s['tx_conf']),
            'interface_status': (s['orig'].get('interface_status', 'unknown'), s['status'], s['status_conf']),
            'connected_to': s['orig'].get('connected_to'),
            'local_router': s['orig'].get('local_router'),
            'remote_router': s['orig'].get('remote_router')
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
