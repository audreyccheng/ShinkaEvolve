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
    NOISE_FLOOR = 5.0  # Mbps, prevents noise amplification on low-bandwidth links

    # Initialize working state
    state = {}
    for if_id, data in telemetry.items():
        state[if_id] = {
            'rx': float(data.get('rx_rate', 0.0)),
            'tx': float(data.get('tx_rate', 0.0)),
            'status': data.get('interface_status', 'unknown'),
            'rx_conf': 0.0,  # Will be calculated later
            'tx_conf': 0.0,
            'status_conf': 1.0,
            'orig': data,
            'fixed_rx': False, # Flag for values confirmed by symmetry
            'fixed_tx': False
        }

    # Helper to check if interface has significant traffic
    def has_traffic(s):
        return s['rx'] > 1.0 or s['tx'] > 1.0

    # 1. Status Repair
    # Resolve UP/DOWN conflicts and enforce zero rates for DOWN interfaces
    for if_id, s in state.items():
        connected_to = s['orig'].get('connected_to')
        if connected_to and connected_to in state:
            peer = state[connected_to]
            
            if s['status'] != peer['status']:
                # Conflict: If traffic exists, link is UP, otherwise DOWN
                if has_traffic(s) or has_traffic(peer):
                    s['status'] = 'up'
                    s['status_conf'] = 0.8
                else:
                    s['status'] = 'down'
                    s['status_conf'] = 0.8
            else:
                # Agreement
                s['status_conf'] = 1.0
        
        # Enforce invariant: DOWN interfaces must have zero rates
        if s['status'] == 'down':
            s['rx'] = 0.0
            s['tx'] = 0.0
            s['rx_conf'] = 1.0
            s['tx_conf'] = 1.0
            s['fixed_rx'] = True
            s['fixed_tx'] = True

    # 2. Symmetry Check (Pass 1)
    # Identify links that satisfy symmetry (within tolerance) and mark them as fixed
    for if_id, s in state.items():
        if s['fixed_rx'] and s['fixed_tx']: continue
        
        connected_to = s['orig'].get('connected_to')
        if not connected_to or connected_to not in state: continue
        
        peer = state[connected_to]
        if peer['status'] == 'down': continue 

        # Check Direction: Local TX -> Peer RX
        if not s['fixed_tx']:
            tx = s['tx']
            prx = peer['rx']
            denom = max(tx, prx, NOISE_FLOOR)
            
            if abs(tx - prx) / denom <= HARDENING_THRESHOLD:
                avg = (tx + prx) / 2.0
                s['tx'] = avg
                peer['rx'] = avg
                s['tx_conf'] = 1.0
                peer['rx_conf'] = 1.0
                s['fixed_tx'] = True
                peer['fixed_rx'] = True
        
        # Check Direction: Local RX <- Peer TX
        if not s['fixed_rx']:
            rx = s['rx']
            ptx = peer['tx']
            denom = max(rx, ptx, NOISE_FLOOR)
            
            if abs(rx - ptx) / denom <= HARDENING_THRESHOLD:
                avg = (rx + ptx) / 2.0
                s['rx'] = avg
                peer['tx'] = avg
                s['rx_conf'] = 1.0
                peer['tx_conf'] = 1.0
                s['fixed_rx'] = True
                peer['fixed_tx'] = True

    # 3. Flow Conservation Solver
    # Use router flow balance to resolve disagreements for unfixed links
    
    def get_router_imbalance_parts(router_id, exclude_if=None, exclude_type=None):
        """Calculate sum(in) and sum(out) for a router, optionally excluding a flow."""
        if not router_id or router_id not in topology: return 0.0, 0.0
        
        sum_in = 0.0
        sum_out = 0.0
        
        for if_id in topology[router_id]:
            if if_id not in state: continue
            st = state[if_id]
            
            if if_id == exclude_if:
                # If we exclude 'rx', we still include 'tx'
                if exclude_type == 'rx': 
                    sum_out += st['tx']
                elif exclude_type == 'tx': 
                    sum_in += st['rx']
            else:
                sum_in += st['rx']
                sum_out += st['tx']
        
        return sum_in, sum_out

    # Iterate to allow corrections to propagate (2 passes)
    for _ in range(2):
        # Iterate over unique links
        sorted_ids = sorted(state.keys())
        
        for if_id in sorted_ids:
            s = state[if_id]
            peer_id = s['orig'].get('connected_to')
            if not peer_id or peer_id not in state: continue
            if if_id > peer_id: continue # Process each pair once
            
            p = state[peer_id]
            if s['status'] == 'down': continue

            # --- Resolve Direction 1: Local TX -> Peer RX ---
            if not s['fixed_tx']:
                cand_tx = s['tx'] 
                cand_rx = p['rx']
                
                r_s = s['orig'].get('local_router') # Source Router
                r_p = p['orig'].get('local_router') # Dest Router
                
                # Calculate what the imbalance would be for each candidate
                in_s, out_s_partial = get_router_imbalance_parts(r_s, if_id, 'tx')
                in_p_partial, out_p = get_router_imbalance_parts(r_p, peer_id, 'rx')
                
                def eval_flow(val):
                    cost = 0.0
                    if r_s: cost += abs(in_s - (out_s_partial + val))
                    if r_p: cost += abs((in_p_partial + val) - out_p)
                    return cost
                
                cost_tx = eval_flow(cand_tx)
                cost_rx = eval_flow(cand_rx)
                
                # Selection Logic
                denom = max(cand_tx, cand_rx, NOISE_FLOOR)
                
                if cost_tx < cost_rx:
                    best_val = cand_tx
                    winner_cost = cost_tx
                    loser_cost = cost_rx
                else:
                    best_val = cand_rx
                    winner_cost = cost_rx
                    loser_cost = cost_tx
                
                # Confidence Calibration
                # High confidence requires:
                # 1. High Distinctness (Winner is much better than Loser)
                # 2. High Fit Quality (Winner actually balances the router well)
                
                distinctness = (loser_cost - winner_cost) / (loser_cost + winner_cost + 1.0)
                rel_winner_cost = winner_cost / denom
                fit_quality = max(0.0, 1.0 - rel_winner_cost)
                
                conf = 0.5 + 0.45 * distinctness * fit_quality
                
                # Ambiguity Fallback: if candidates are close but no clear winner, average them
                diff_ratio = abs(cand_tx - cand_rx) / denom
                if conf < 0.6 and diff_ratio < 0.1:
                    best_val = (cand_tx + cand_rx) / 2.0
                    conf = 0.5

                s['tx'] = best_val
                p['rx'] = best_val
                s['tx_conf'] = conf
                p['rx_conf'] = conf
            
            # --- Resolve Direction 2: Local RX <- Peer TX ---
            if not s['fixed_rx']:
                cand_rx = s['rx']
                cand_tx = p['tx']
                
                r_s = s['orig'].get('local_router') # Dest Router
                r_p = p['orig'].get('local_router') # Source Router
                
                in_s_partial, out_s = get_router_imbalance_parts(r_s, if_id, 'rx')
                in_p, out_p_partial = get_router_imbalance_parts(r_p, peer_id, 'tx')
                
                def eval_flow_2(val):
                    cost = 0.0
                    if r_s: cost += abs((in_s_partial + val) - out_s)
                    if r_p: cost += abs(in_p - (out_p_partial + val))
                    return cost
                
                cost_rx = eval_flow_2(cand_rx)
                cost_tx = eval_flow_2(cand_tx)
                
                denom = max(cand_rx, cand_tx, NOISE_FLOOR)
                
                if cost_rx < cost_tx:
                    best_val = cand_rx
                    winner_cost = cost_rx
                    loser_cost = cost_tx
                else:
                    best_val = cand_tx
                    winner_cost = cost_tx
                    loser_cost = cost_rx
                
                distinctness = (loser_cost - winner_cost) / (loser_cost + winner_cost + 1.0)
                rel_winner_cost = winner_cost / denom
                fit_quality = max(0.0, 1.0 - rel_winner_cost)
                
                conf = 0.5 + 0.45 * distinctness * fit_quality

                diff_ratio = abs(cand_rx - cand_tx) / denom
                if conf < 0.6 and diff_ratio < 0.1:
                    best_val = (cand_rx + cand_tx) / 2.0
                    conf = 0.5

                s['rx'] = best_val
                p['tx'] = best_val
                s['rx_conf'] = conf
                p['tx_conf'] = conf

    # Build final result dictionary
    result = {}
    for if_id, s in state.items():
        orig = s['orig']
        result[if_id] = {
            'rx_rate': (orig.get('rx_rate', 0.0), s['rx'], s['rx_conf']),
            'tx_rate': (orig.get('tx_rate', 0.0), s['tx'], s['tx_conf']),
            'interface_status': (orig.get('interface_status', 'unknown'), s['status'], s['status_conf']),
            'connected_to': orig.get('connected_to'),
            'local_router': orig.get('local_router'),
            'remote_router': orig.get('remote_router')
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
