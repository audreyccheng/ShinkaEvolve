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

    # Initialize Working State
    state = {}
    for if_id, data in telemetry.items():
        state[if_id] = {
            'rx': float(data.get('rx_rate', 0.0)),
            'tx': float(data.get('tx_rate', 0.0)),
            'status': data.get('interface_status', 'unknown'),
            'rx_conf': 0.5,
            'tx_conf': 0.5,
            'status_conf': 1.0,
            'orig': data,
            # Lock flags to anchor solved parts of the network
            'locked_rx': False,
            'locked_tx': False
        }

    # --- Pass 1: Status Consensus ---
    for if_id, s in state.items():
        peer_id = s['orig'].get('connected_to')
        if peer_id and peer_id in state:
            peer = state[peer_id]
            # Mismatch detection
            if s['status'] != peer['status']:
                # Traffic check: if any significant traffic, link is likely UP
                traffic = max(s['rx'], s['tx'], peer['rx'], peer['tx'])
                if traffic > 1.0:
                    s['status'] = 'up'
                    s['status_conf'] = 0.9
                else:
                    s['status'] = 'down'
                    s['status_conf'] = 0.9

        # Consistency enforcement
        if s['status'] == 'down':
            s['rx'] = 0.0
            s['tx'] = 0.0
            s['rx_conf'] = 1.0
            s['tx_conf'] = 1.0
            s['locked_rx'] = True
            s['locked_tx'] = True

    # Helper: Calculate router metrics (Imbalance, Reliability, Total Flow)
    def get_router_metrics(router_id):
        if not router_id or router_id not in topology:
            return 0.0, 0.0, 1.0

        in_sum = 0.0
        out_sum = 0.0
        conf_sum = 0.0
        count = 0
        total_flow = 0.0

        for if_id in topology[router_id]:
            if if_id not in state: continue
            
            curr = state[if_id]
            in_sum += curr['rx']
            out_sum += curr['tx']
            total_flow += (curr['rx'] + curr['tx'])

            # Reliability calculation (neighbors' confidence)
            # Use average of RX/TX conf as proxy for interface health
            c = (curr['rx_conf'] + curr['tx_conf']) / 2.0
            conf_sum += c
            count += 1

        reliability = conf_sum / max(count, 1) if count > 0 else 0.0
        imbalance = in_sum - out_sum
        return imbalance, reliability, max(total_flow, 1.0)

    # --- Pass 2: Anchor Reliable Links ---
    # If a link is perfectly symmetric initially, we treat it as Ground Truth anchors.
    sorted_ifs = sorted(state.keys())
    
    for if_id in sorted_ifs:
        s = state[if_id]
        if s['status'] == 'down': continue
        peer_id = s['orig'].get('connected_to')
        if not peer_id or peer_id not in state: continue
        peer = state[peer_id]

        # Check Direction A (My TX -> Peer RX)
        tx = float(s['orig'].get('tx_rate', 0.0))
        prx = float(peer['orig'].get('rx_rate', 0.0))
        if abs(tx - prx) / max(tx, prx, 1.0) < HARDENING_THRESHOLD:
            avg = (tx + prx) / 2.0
            s['tx'] = avg
            peer['rx'] = avg
            s['locked_tx'] = True
            peer['locked_rx'] = True
            s['tx_conf'] = 1.0
            peer['rx_conf'] = 1.0

        # Check Direction B (Peer TX -> My RX)
        rx = float(s['orig'].get('rx_rate', 0.0))
        ptx = float(peer['orig'].get('tx_rate', 0.0))
        if abs(rx - ptx) / max(rx, ptx, 1.0) < HARDENING_THRESHOLD:
            avg = (rx + ptx) / 2.0
            s['rx'] = avg
            peer['tx'] = avg
            s['locked_rx'] = True
            peer['locked_tx'] = True
            s['rx_conf'] = 1.0
            peer['tx_conf'] = 1.0

    # --- Pass 3: Iterative Flow Repair ---
    ITERATIONS = 5
    for iteration in range(ITERATIONS):
        processed_pairs = set()

        for if_id in sorted_ifs:
            s = state[if_id]
            if s['status'] == 'down': continue
            peer_id = s['orig'].get('connected_to')
            if not peer_id or peer_id not in state: continue

            pair_key = tuple(sorted([if_id, peer_id]))
            if pair_key in processed_pairs: continue
            processed_pairs.add(pair_key)

            peer = state[peer_id]

            # --- Flow Direction A: Local TX -> Peer RX ---
            if not s['locked_tx']:
                r_local = s['orig'].get('local_router')
                r_remote = peer['orig'].get('local_router')

                imb_loc, rel_loc, flow_loc = get_router_metrics(r_local)
                imb_rem, rel_rem, flow_rem = get_router_metrics(r_remote)

                curr_tx = s['tx']
                curr_prx = peer['rx']

                # Calculate Targets (Golden Truth candidates)
                # Local TX is Out. Imb = In - Out.
                # To Fix Imbalance: New_TX = Current_TX + Imbalance (Increase Out to reduce positive Imb)
                target_loc = curr_tx + imb_loc
                # Remote RX is In. Imb = In - Out.
                # To Fix Imbalance: New_RX = Current_RX - Imbalance (Decrease In to reduce positive Imb)
                target_rem = curr_prx - imb_rem

                # 1. Golden Truth Verification
                golden_diff = abs(target_loc - target_rem)
                golden_denom = max(target_loc, target_rem, 1.0)
                
                # Check if targets derived from completely different routers agree
                is_golden = (golden_denom > 1.0 and golden_diff / golden_denom < HARDENING_THRESHOLD)

                best_val = curr_tx
                
                if is_golden:
                    # Strongest possible signal: Flow conservation matches on both sides
                    best_val = (target_loc + target_rem) / 2.0
                    s['locked_tx'] = True
                    peer['locked_rx'] = True
                else:
                    # 2. Heuristic Arbitration
                    # Determine which router is "Solid" (Balanced)
                    solid_loc = (abs(imb_loc) / flow_loc < HARDENING_THRESHOLD)
                    solid_rem = (abs(imb_rem) / flow_rem < HARDENING_THRESHOLD)
                    
                    if solid_loc and not solid_rem:
                        # Local is balanced, trust its requirement
                        best_val = target_loc
                    elif solid_rem and not solid_loc:
                        # Remote is balanced, trust its requirement
                        best_val = target_rem
                    elif rel_loc > rel_rem + 0.3:
                        # Local is significantly more reliable
                        best_val = target_loc
                    elif rel_rem > rel_loc + 0.3:
                        # Remote is significantly more reliable
                        best_val = target_rem
                    else:
                        # Weighted Average
                        w_l = max(rel_loc, 0.1)
                        w_r = max(rel_rem, 0.1)
                        best_val = (w_l * target_loc + w_r * target_rem) / (w_l + w_r)
                        
                        # Sanity clamp to measurements if targets are diverging wildly
                        meas_tx = float(s['orig'].get('tx_rate', 0.0))
                        meas_prx = float(peer['orig'].get('rx_rate', 0.0))
                        if abs(best_val - meas_tx) > 20 and abs(best_val - meas_prx) > 20:
                             best_val = (meas_tx + meas_prx) / 2.0

                s['tx'] = max(0.0, best_val)
                peer['rx'] = max(0.0, best_val)


            # --- Flow Direction B: Peer TX -> Local RX ---
            if not s['locked_rx']:
                r_local = s['orig'].get('local_router')
                r_remote = peer['orig'].get('local_router')

                imb_loc, rel_loc, flow_loc = get_router_metrics(r_local)
                imb_rem, rel_rem, flow_rem = get_router_metrics(r_remote)

                curr_rx = s['rx']
                curr_ptx = peer['tx']

                # Local RX is In. Target = Current - Imb
                target_loc = curr_rx - imb_loc
                # Remote TX is Out. Target = Current + Imb
                target_rem = curr_ptx + imb_rem

                golden_diff = abs(target_loc - target_rem)
                golden_denom = max(target_loc, target_rem, 1.0)
                is_golden = (golden_denom > 1.0 and golden_diff / golden_denom < HARDENING_THRESHOLD)

                best_val_b = curr_rx

                if is_golden:
                    best_val_b = (target_loc + target_rem) / 2.0
                    s['locked_rx'] = True
                    peer['locked_tx'] = True
                else:
                    solid_loc = (abs(imb_loc) / flow_loc < HARDENING_THRESHOLD)
                    solid_rem = (abs(imb_rem) / flow_rem < HARDENING_THRESHOLD)
                    
                    if solid_loc and not solid_rem:
                        best_val_b = target_loc
                    elif solid_rem and not solid_loc:
                        best_val_b = target_rem
                    elif rel_loc > rel_rem + 0.3:
                        best_val_b = target_loc
                    elif rel_rem > rel_loc + 0.3:
                        best_val_b = target_rem
                    else:
                        w_l = max(rel_loc, 0.1)
                        w_r = max(rel_rem, 0.1)
                        best_val_b = (w_l * target_loc + w_r * target_rem) / (w_l + w_r)
                        
                        meas_rx = float(s['orig'].get('rx_rate', 0.0))
                        meas_ptx = float(peer['orig'].get('tx_rate', 0.0))
                        if abs(best_val_b - meas_rx) > 20 and abs(best_val_b - meas_ptx) > 20:
                             best_val_b = (meas_rx + meas_ptx) / 2.0

                s['rx'] = max(0.0, best_val_b)
                peer['tx'] = max(0.0, best_val_b)

    # --- Final Confidence Calibration ---
    for if_id, s in state.items():
        if s['status'] == 'down': continue
        peer_id = s['orig'].get('connected_to')
        if not peer_id or peer_id not in state: continue
        peer = state[peer_id]

        r_local = s['orig'].get('local_router')
        r_remote = peer['orig'].get('local_router')

        imb_loc, _, flow_loc = get_router_metrics(r_local)
        imb_rem, _, flow_rem = get_router_metrics(r_remote)

        err_loc = abs(imb_loc) / flow_loc
        err_rem = abs(imb_rem) / flow_rem
        
        # Calculate penalty based on residual imbalance (Recommendation 5)
        # We use the minimum error of the two routers to be charitable (one side might be unobservable)
        # But we still damp confidence if the "best" side is still broken.
        min_err = min(err_loc, err_rem)
        penalty_factor = max(0.0, 1.0 - (min_err * 5.0))

        def calibrate(val, m1, m2, is_locked):
            # Base confidence
            conf = 0.5
            # If signals matched originally
            if abs(m1 - m2) / max(m1, m2, 1.0) < HARDENING_THRESHOLD:
                conf = 1.0
            # If locked by Golden Truth
            elif is_locked:
                conf = 1.0
            # If flow is solid
            elif err_loc < HARDENING_THRESHOLD or err_rem < HARDENING_THRESHOLD:
                conf = 0.9
            
            # Apply residual damping
            return conf * penalty_factor

        # TX Confidence
        tx_meas = float(s['orig'].get('tx_rate', 0.0))
        prx_meas = float(peer['orig'].get('rx_rate', 0.0))
        s['tx_conf'] = calibrate(s['tx'], tx_meas, prx_meas, s['locked_tx'])
        peer['rx_conf'] = s['tx_conf']

        # RX Confidence
        rx_meas = float(s['orig'].get('rx_rate', 0.0))
        ptx_meas = float(peer['orig'].get('tx_rate', 0.0))
        s['rx_conf'] = calibrate(s['rx'], rx_meas, ptx_meas, s['locked_rx'])
        peer['tx_conf'] = s['rx_conf']

    # Result Assembly
    result = {}
    for if_id, s in state.items():
        orig = s['orig']
        
        if s['rx_conf'] > 0.8 and s['tx_conf'] > 0.8:
            s['status_conf'] = max(s['status_conf'], 0.95)

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