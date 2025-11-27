# EVOLVE-BLOCK-START
"""
Network telemetry repair algorithm using consensus-based flow conservation 
and dual-sided validation.
"""
from typing import Dict, Any, Tuple, List

def repair_network_telemetry(telemetry: Dict[str, Dict[str, Any]],
                             topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:
    
    HARDENING_THRESHOLD = 0.02
    
    # 1. Initialize Working State
    state = {}
    for if_id, data in telemetry.items():
        state[if_id] = {
            'rx': float(data.get('rx_rate', 0.0)),
            'tx': float(data.get('tx_rate', 0.0)),
            'status': data.get('interface_status', 'unknown'),
            'rx_conf': 0.5,
            'tx_conf': 0.5,
            'status_conf': 1.0,
            'locked_rx': False, # If true, value is fixed (anchor)
            'locked_tx': False,
            'orig': data
        }

    # 2. Status Repair & Consensus
    for if_id, s in state.items():
        conn = s['orig'].get('connected_to')
        if conn and conn in state:
            peer = state[conn]
            # Status Consensus
            if s['status'] != peer['status']:
                # Traffic check: If significant traffic, link is likely UP
                traffic = max(s['rx'], s['tx'], peer['rx'], peer['tx'])
                if traffic > 1.0:
                    s['status'] = 'up'
                    s['status_conf'] = 0.9
                else:
                    s['status'] = 'down'
                    s['status_conf'] = 0.9
        
        # Enforce DOWN invariants
        if s['status'] == 'down':
            s['rx'], s['tx'] = 0.0, 0.0
            s['rx_conf'], s['tx_conf'] = 1.0, 1.0
            s['locked_rx'], s['locked_tx'] = True, True

    # 3. Symmetry Anchoring (Golden Truth Init)
    # If raw measurements agree, lock them immediately.
    sorted_ids = sorted(state.keys())
    for if_id in sorted_ids:
        s = state[if_id]
        if s['status'] == 'down': continue
        
        conn = s['orig'].get('connected_to')
        if not conn or conn not in state: continue
        peer = state[conn]
        
        # Check TX -> RX symmetry
        if not s['locked_tx']:
            tx = s['tx']
            prx = peer['rx']
            denom = max(tx, prx, 1.0)
            if abs(tx - prx) / denom < HARDENING_THRESHOLD:
                avg = (tx + prx) / 2.0
                s['tx'] = avg
                peer['rx'] = avg
                s['tx_conf'] = 1.0
                peer['rx_conf'] = 1.0
                s['locked_tx'] = True
                peer['locked_rx'] = True

        # Check RX <- TX symmetry
        if not s['locked_rx']:
            rx = s['rx']
            ptx = peer['tx']
            denom = max(rx, ptx, 1.0)
            if abs(rx - ptx) / denom < HARDENING_THRESHOLD:
                avg = (rx + ptx) / 2.0
                s['rx'] = avg
                peer['tx'] = avg
                s['rx_conf'] = 1.0
                peer['tx_conf'] = 1.0
                s['locked_rx'] = True
                peer['locked_tx'] = True

    # 4. Iterative Flow Solver
    # Pre-calculate router balances
    router_balance = {} # rid -> sum(in) - sum(out)
    for rid, if_list in topology.items():
        bal = 0.0
        for iid in if_list:
            if iid in state:
                bal += state[iid]['rx'] - state[iid]['tx']
        router_balance[rid] = bal

    ITERATIONS = 5
    for _ in range(ITERATIONS):
        # We iterate by interface, processing outgoing TX for each.
        # This covers all links exactly once per direction.
        
        for if_id in sorted_ids:
            s = state[if_id]
            if s['status'] == 'down' or s['locked_tx']: continue
            
            conn = s['orig'].get('connected_to')
            if not conn or conn not in state: continue
            peer = state[conn]
            
            # Context: Flow F (Local TX -> Remote RX)
            # Local Router: Balance L = (In - Out_Others) - F
            # Remote Router: Balance R = (In_Others - Out) + F
            # We want L -> 0, R -> 0.
            
            cur_f = s['tx']
            rid_loc = s['orig'].get('local_router')
            rid_rem = peer['orig'].get('local_router')
            
            targets = []
            
            # Local Target (Source)
            if rid_loc and rid_loc in router_balance:
                # To zero balance: F_new = F_old + Balance
                bal_loc = router_balance[rid_loc]
                tgt_loc = cur_f + bal_loc
                # Sanity check: Non-negative
                if tgt_loc >= -0.01: targets.append(max(0.0, tgt_loc))
            
            # Remote Target (Dest)
            if rid_rem and rid_rem in router_balance:
                # To zero balance: F_new = F_old - Balance
                bal_rem = router_balance[rid_rem]
                tgt_rem = cur_f - bal_rem
                if tgt_rem >= -0.01: targets.append(max(0.0, tgt_rem))
            
            # Candidates from measurements
            m_tx = float(s['orig'].get('tx_rate', 0.0))
            m_prx = float(peer['orig'].get('rx_rate', 0.0))
            
            # Candidate set: Measurements, Average, and Targets themselves
            candidates = [m_tx, m_prx, (m_tx + m_prx)/2.0]
            # Adding targets as candidates allows "inference" where measurements are totally wrong
            for t in targets:
                candidates.append(t)
            
            # Evaluate Candidates
            best_val = cur_f
            min_cost = float('inf')
            
            for val in candidates:
                cost = 0.0
                valid_t = 0
                for t in targets:
                    cost += abs(val - t)
                    valid_t += 1
                
                # Tie-breaker/Anchor: consistency with measurements
                # If no topology constraints, this dominates.
                # If topology exists, this acts as regularization.
                measurement_cost = abs(val - m_tx) + abs(val - m_prx)
                
                if valid_t == 0:
                    cost = measurement_cost
                else:
                    # Weight topology higher than measurement drift
                    cost += 0.1 * measurement_cost
                
                if cost < min_cost:
                    min_cost = cost
                    best_val = val
            
            # Update State & Incremental Balances
            diff = best_val - cur_f
            if abs(diff) > 1e-6:
                s['tx'] = best_val
                peer['rx'] = best_val
                if rid_loc in router_balance: router_balance[rid_loc] -= diff
                if rid_rem in router_balance: router_balance[rid_rem] += diff
                
            # Dynamic Locking (Golden Truth)
            # If we hit a value that satisfies both router constraints perfectly
            if len(targets) == 2:
                t1 = targets[0]
                t2 = targets[1]
                limit = max(best_val, 1.0) * 0.001
                if abs(best_val - t1) < limit and abs(best_val - t2) < limit:
                    s['locked_tx'] = True
                    peer['locked_rx'] = True

    # 5. Final Confidence Calibration
    for if_id in sorted_ids:
        s = state[if_id]
        if s['status'] == 'down': continue
        
        conn = s['orig'].get('connected_to')
        peer = state.get(conn)
        
        # Calculate TX Confidence (and apply to peer RX)
        # We re-evaluate fit globally
        val = s['tx']
        rid_loc = s['orig'].get('local_router')
        rid_rem = peer['orig'].get('local_router') if peer else None
        
        bal_loc = router_balance.get(rid_loc)
        bal_rem = router_balance.get(rid_rem)
        
        m_tx = float(s['orig'].get('tx_rate', 0.0))
        m_prx = float(peer['orig'].get('rx_rate', 0.0)) if peer else 0.0
        
        disagreement = abs(m_tx - m_prx) / max(m_tx, m_prx, 1.0)
        
        has_loc = bal_loc is not None
        has_rem = bal_rem is not None
        
        # Check Solidity: Is the router balanced relative to this flow?
        solid_threshold = max(val, 1.0) * HARDENING_THRESHOLD
        solid_loc = has_loc and (abs(bal_loc) < solid_threshold)
        solid_rem = has_rem and (abs(bal_rem) < solid_threshold)
        
        score = 0.5 # Default ambiguous
        
        # Tiered Scoring
        if disagreement < HARDENING_THRESHOLD:
            # Measurements agree - extremely reliable
            if solid_loc and solid_rem: score = 1.0
            else: score = 0.95
        elif solid_loc and solid_rem:
            # "Dual Solid": Measurements disagreed, but physics proved a specific value fixes everything
            score = 0.95
        elif solid_loc or solid_rem:
            # "Single Solid": One side confirms the value
            score = 0.85
        else:
            # Ambiguous: No topology confirmation, measurements disagree.
            # Confidence degrades with disagreement magnitude.
            score = max(0.0, 1.0 - disagreement)
            
        s['tx_conf'] = score
        if peer: peer['rx_conf'] = score

    # Construct Result
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