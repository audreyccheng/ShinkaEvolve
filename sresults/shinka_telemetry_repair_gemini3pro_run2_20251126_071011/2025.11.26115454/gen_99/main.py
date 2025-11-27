# EVOLVE-BLOCK-START
"""
Network telemetry repair algorithm that detects and corrects inconsistencies
in network interface telemetry data using topology relationships.
"""
from typing import Dict, Any, Tuple, List

def repair_network_telemetry(telemetry: Dict[str, Dict[str, Any]],
                             topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:
    """
    Repair network interface telemetry by detecting and correcting inconsistencies.
    
    Uses Iterative Flow Conservation with "Golden Truth" locking and Reliability Weighting.
    1. Identifies symmetric links as anchors.
    2. Calculates flow targets for each link from both Router A and Router B perspectives.
    3. If targets agree ("Golden Truth"), locks the value.
    4. Otherwise, arbitrates between measurements and flow targets based on router reliability.
    5. Calibrates confidence based on the final residual imbalance of the endpoints.
    """

    # Constants
    HARDENING_THRESHOLD = 0.02
    ITERATIONS = 5

    # Initialize working state
    state = {}
    for if_id, data in telemetry.items():
        state[if_id] = {
            'rx': float(data.get('rx_rate', 0.0)),
            'tx': float(data.get('tx_rate', 0.0)),
            'status': data.get('interface_status', 'unknown'),
            'rx_conf': 0.0,
            'tx_conf': 0.0,
            'status_conf': 1.0,
            'orig': data,
            'locked': False 
        }

    # --- Pass 1: Status Consensus ---
    # Determine UP/DOWN status based on traffic presence and peer agreement
    for if_id, s in state.items():
        peer_id = s['orig'].get('connected_to')
        if peer_id and peer_id in state:
            peer = state[peer_id]
            if s['status'] != peer['status']:
                # Traffic check: traffic > 1.0 Mbps implies UP
                max_traffic = max(s['rx'], s['tx'], peer['rx'], peer['tx'])
                if max_traffic > 1.0:
                    s['status'] = 'up'
                    s['status_conf'] = 0.9
                else:
                    s['status'] = 'down'
                    s['status_conf'] = 0.9

        # Enforce DOWN consistency
        if s['status'] == 'down':
            s['rx'] = 0.0
            s['tx'] = 0.0
            s['locked'] = True
            s['rx_conf'] = 1.0
            s['tx_conf'] = 1.0

    # --- Pass 2: Anchor Identification ---
    # Identify initially symmetric links to serve as anchors for the solver
    sorted_ifs = sorted(state.keys())
    for if_id in sorted_ifs:
        s = state[if_id]
        if s['locked']: continue
        
        peer_id = s['orig'].get('connected_to')
        if not peer_id or peer_id not in state: continue
        peer = state[peer_id]
        
        # Check symmetry
        tx, prx = s['tx'], peer['rx']
        rx, ptx = s['rx'], peer['tx']
        
        sym_a = abs(tx - prx) / max(tx, prx, 1.0) < HARDENING_THRESHOLD
        sym_b = abs(rx - ptx) / max(rx, ptx, 1.0) < HARDENING_THRESHOLD
        
        if sym_a and sym_b:
            # Symmetrize and lock
            avg_tx = (tx + prx) / 2.0
            avg_rx = (rx + ptx) / 2.0
            s['tx'], peer['rx'] = avg_tx, avg_tx
            s['rx'], peer['tx'] = avg_rx, avg_rx
            s['locked'] = True
            peer['locked'] = True

    # --- Helper: Router Stats ---
    def get_router_stats(router_id, exclude_if=None):
        """
        Calculates Router Imbalance and a Reliability Score based on neighbor symmetry.
        Returns: (imbalance, reliability, total_flow)
        """
        if not router_id or router_id not in topology:
            return 0.0, 0.0, 1.0
            
        in_flow = 0.0
        out_flow = 0.0
        total = 0.0
        
        symmetric_neighbors = 0
        neighbor_count = 0
        
        for if_id in topology[router_id]:
            if if_id not in state: continue
            
            r, t = state[if_id]['rx'], state[if_id]['tx']
            in_flow += r
            out_flow += t
            total += (r + t)
            
            if if_id != exclude_if:
                neighbor_count += 1
                pid = state[if_id]['orig'].get('connected_to')
                if pid and pid in state:
                    # Check if this neighbor link is currently symmetric/consistent
                    # We use a looser threshold for reliability metric to be robust
                    p_rx = state[pid]['rx']
                    if abs(t - p_rx) / max(t, p_rx, 1.0) < HARDENING_THRESHOLD * 2.0:
                        symmetric_neighbors += 1
                        
        imbalance = in_flow - out_flow
        reliability = symmetric_neighbors / max(neighbor_count, 1) if neighbor_count > 0 else 0.0
        return imbalance, reliability, max(total, 1.0)

    # --- Pass 3: Iterative Repair (Gauss-Seidel) ---
    for _ in range(ITERATIONS):
        processed_pairs = set()
        
        for if_id in sorted_ifs:
            s = state[if_id]
            peer_id = s['orig'].get('connected_to')
            if not peer_id or peer_id not in state: continue
            
            pair = tuple(sorted([if_id, peer_id]))
            if pair in processed_pairs: continue
            processed_pairs.add(pair)
            
            peer = state[peer_id]
            if s['locked'] and peer['locked']: continue
            if s['status'] == 'down': continue
            
            r_local = s['orig'].get('local_router')
            r_remote = peer['orig'].get('local_router')
            
            # --- Direction A: Local TX -> Peer RX ---
            imb_loc, rel_loc, _ = get_router_stats(r_local, if_id)
            imb_rem, rel_rem, _ = get_router_stats(r_remote, peer_id)
            
            # Flow Targets: Value needed to zero out imbalance
            # Local Imb = In - Out. Increasing Out (TX) decreases Imb.
            target_local = s['tx'] + imb_loc
            # Remote Imb = In - Out. Increasing In (RX) increases Imb.
            # If Imb is positive (surplus), we need less In. Target = Current - Imb.
            target_remote = peer['rx'] - imb_rem
            
            # Golden Truth Check
            denom_g = max(target_local, target_remote, 1.0)
            if abs(target_local - target_remote) / denom_g < HARDENING_THRESHOLD:
                # Strong agreement between flow constraints
                best_val = (target_local + target_remote) / 2.0
                if best_val >= 0:
                    s['tx'] = best_val
                    peer['rx'] = best_val
                    # Dynamic locking if routers are somewhat reliable
                    if rel_loc > 0.5 or rel_rem > 0.5:
                         pass # Could lock, but let's allow micro-adjustments
                    # Skip to next direction
            else:
                # Weighted Arbitration
                meas_tx = float(s['orig'].get('tx_rate', 0.0))
                meas_prx = float(peer['orig'].get('rx_rate', 0.0))
                
                candidates = [
                    {'val': meas_tx, 'w': 1.0},
                    {'val': meas_prx, 'w': 1.0},
                    {'val': target_local, 'w': rel_loc * 6.0}, # Higher weight for reliable flow
                    {'val': target_remote, 'w': rel_rem * 6.0}
                ]
                
                # Check for Phantom Traffic (RX >> TX)
                # If meas_prx >> meas_tx, likely phantom unless Local needs to dump traffic
                if meas_prx > meas_tx * 1.5 and meas_tx > 1.0:
                    # Penalty on RX measurement
                    candidates[1]['w'] = 0.1
                
                best_val = s['tx']
                min_cost = float('inf')
                
                # Test points: candidates + averages
                test_points = [c['val'] for c in candidates]
                test_points.append((meas_tx + meas_prx)/2.0)
                
                for v in test_points:
                    if v < 0: continue
                    cost = sum(c['w'] * abs(v - c['val']) for c in candidates)
                    if cost < min_cost:
                        min_cost = cost
                        best_val = v
                
                s['tx'] = best_val
                peer['rx'] = best_val


            # --- Direction B: Peer TX -> Local RX ---
            imb_loc, rel_loc, _ = get_router_stats(r_local, if_id)
            imb_rem, rel_rem, _ = get_router_stats(r_remote, peer_id)
            
            # Targets
            target_local_rx = s['rx'] - imb_loc
            target_remote_tx = peer['tx'] + imb_rem
            
            denom_g = max(target_local_rx, target_remote_tx, 1.0)
            if abs(target_local_rx - target_remote_tx) / denom_g < HARDENING_THRESHOLD:
                best_val_b = (target_local_rx + target_remote_tx) / 2.0
                if best_val_b >= 0:
                    s['rx'] = best_val_b
                    peer['tx'] = best_val_b
            else:
                meas_rx = float(s['orig'].get('rx_rate', 0.0))
                meas_ptx = float(peer['orig'].get('tx_rate', 0.0))
                
                candidates_b = [
                    {'val': meas_rx, 'w': 1.0},
                    {'val': meas_ptx, 'w': 1.0},
                    {'val': target_local_rx, 'w': rel_loc * 6.0},
                    {'val': target_remote_tx, 'w': rel_rem * 6.0}
                ]
                
                if meas_rx > meas_ptx * 1.5 and meas_ptx > 1.0:
                     candidates_b[0]['w'] = 0.1
                
                best_val_b = s['rx']
                min_cost_b = float('inf')
                
                test_points_b = [c['val'] for c in candidates_b]
                test_points_b.append((meas_rx + meas_ptx)/2.0)
                
                for v in test_points_b:
                    if v < 0: continue
                    cost = sum(c['w'] * abs(v - c['val']) for c in candidates_b)
                    if cost < min_cost_b:
                        min_cost_b = cost
                        best_val_b = v
                        
                s['rx'] = best_val_b
                peer['tx'] = best_val_b

    # --- Pass 4: Final Confidence Calibration ---
    # Calculate final stats to judge quality
    final_stats = {rid: get_router_stats(rid) for rid in topology}
    
    for if_id, s in state.items():
        if s['status'] == 'down': continue
        
        peer_id = s['orig'].get('connected_to')
        if not peer_id or peer_id not in state:
            # Fallback for isolated links
            s['tx_conf'] = 0.5
            s['rx_conf'] = 0.5
            continue
            
        r_loc = s['orig'].get('local_router')
        r_rem = state[peer_id]['orig'].get('local_router')
        
        imb_loc, _, flow_loc = final_stats.get(r_loc, (0,0,1))
        imb_rem, _, flow_rem = final_stats.get(r_rem, (0,0,1))
        
        # Calculate residual error ratio
        err_loc = abs(imb_loc) / flow_loc
        err_rem = abs(imb_rem) / flow_rem
        
        # Confidence Function: "Trust Healthy"
        # If one side is perfect, we trust the link value is correct
        def get_score(err):
            if err < HARDENING_THRESHOLD: return 1.0
            if err < HARDENING_THRESHOLD * 2: return 0.9
            return max(0.0, 1.0 - err * 4.0)
            
        s_loc = get_score(err_loc)
        s_rem = get_score(err_rem)
        
        # If one side is very healthy, trust it dominates the solution
        if s_loc >= 0.95 or s_rem >= 0.95:
            final_conf = max(s_loc, s_rem)
        else:
            final_conf = (s_loc + s_rem) / 2.0
            
        s['tx_conf'] = final_conf
        s['rx_conf'] = final_conf
        
        # Edge case: Zero flow
        if s['tx'] < 0.1 and s['rx'] < 0.1:
            s['tx_conf'] = 1.0
            s['rx_conf'] = 1.0

    # Build Result
    result = {}
    for if_id, s in state.items():
        orig = s['orig']
        
        # Status confidence boost if rates are confident
        if s['rx_conf'] > 0.9 and s['tx_conf'] > 0.9:
            s['status_conf'] = max(s['status_conf'], 0.98)
            
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