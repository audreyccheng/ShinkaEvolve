# EVOLVE-BLOCK-START
"""
Robust Consensus Telemetry Repair
Implements a three-phase repair strategy (Sanitization, Flow Context, Arbitration)
to validate network telemetry using Link Symmetry and Flow Conservation.
"""
from typing import Dict, Any, Tuple, List
import math

def repair_network_telemetry(telemetry: Dict[str, Dict[str, Any]],
                             topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:
    """
    Repairs network telemetry using a robust consensus model.

    Phases:
    1. Status & Clean Estimates: Fix status based on traffic. Generate 'Clean' RX/TX
       estimates for each interface using Link Symmetry (Peer > Self).
    2. Flow Calculation: Use Clean Estimates to compute Flow Conservation targets
       for every interface on a router.
    3. Arbitration: Compare Local, Peer, and Flow signals to determine the final
       value and assign calibrated confidence.
    """

    # --- Configuration ---
    # Tolerance for "Agreement": 2% relative or 0.1 Mbps absolute
    REL_TOL = 0.02
    ABS_TOL = 0.1

    results = {}

    # Helper to safely extract float rates
    def get_rate(data: Dict, key: str) -> float:
        val = data.get(key, 0.0)
        return float(val) if val is not None else 0.0

    # Helper to check agreement between two values
    def agrees(v1: float, v2: float) -> bool:
        diff = abs(v1 - v2)
        if diff < ABS_TOL: return True
        return (diff / max(abs(v1), abs(v2))) <= REL_TOL

    # --- Phase 1: Status Sanitization & Clean Estimates ---
    # We build an intermediate 'state' dictionary to hold purified data for Flow calc
    state = {}

    for if_id, data in telemetry.items():
        # Raw Signals
        s_rx = get_rate(data, 'rx_rate')
        s_tx = get_rate(data, 'tx_rate')
        orig_status = data.get('interface_status', 'unknown')

        # Peer Signals
        peer_id = data.get('connected_to')
        if peer_id and peer_id in telemetry:
            peer_data = telemetry[peer_id]
            p_rx = get_rate(peer_data, 'rx_rate') # My TX -> Peer RX
            p_tx = get_rate(peer_data, 'tx_rate') # My RX -> Peer TX
            p_status = peer_data.get('interface_status', 'unknown')
            has_peer = True
        else:
            p_rx, p_tx, p_status = 0.0, 0.0, 'unknown'
            has_peer = False

        # 1. Status Repair
        # Logic: Traffic implies UP. No traffic + Peer Down implies DOWN.
        has_traffic = (s_rx > ABS_TOL or s_tx > ABS_TOL or
                       (has_peer and (p_rx > ABS_TOL or p_tx > ABS_TOL)))

        final_status = orig_status
        status_conf = 1.0

        if orig_status == 'down' and has_traffic:
            final_status = 'up'
            status_conf = 0.9 # Correcting a false negative
        elif orig_status == 'up' and not has_traffic:
            # If peer is confirmed down, we are likely down too
            if p_status == 'down':
                final_status = 'down'
                status_conf = 0.85
            # If peer is UP but no traffic, it might just be idle. Keep UP.

        # 2. Generate "Clean" Estimates for Flow Calculation
        # We need a robust guess for this interface's contribution to the router's flow.
        # Strategy: Use Peer values if available (Link Symmetry), else Self.
        # This prevents local corruption from polluting the router-wide flow sum.

        if final_status == 'down':
            clean_rx, clean_tx = 0.0, 0.0
        else:
            # Clean RX Estimate (Incoming to Router)
            if has_peer:
                # If Self and Peer agree, average them. If distinct, trust Peer.
                if agrees(s_rx, p_tx):
                    clean_rx = (s_rx + p_tx) / 2.0
                else:
                    clean_rx = p_tx
            else:
                clean_rx = s_rx

            # Clean TX Estimate (Outgoing from Router)
            if has_peer:
                if agrees(s_tx, p_rx):
                    clean_tx = (s_tx + p_rx) / 2.0
                else:
                    clean_tx = p_rx
            else:
                clean_tx = s_tx

        state[if_id] = {
            's_rx': s_rx, 's_tx': s_tx,
            'p_rx': p_rx if has_peer else None,
            'p_tx': p_tx if has_peer else None,
            'clean_rx': clean_rx, 'clean_tx': clean_tx,
            'status': final_status, 'status_conf': status_conf,
            'orig_status': orig_status
        }

    # --- Phase 2: Flow Target Calculation ---
    # For every interface, calculate what the rate SHOULD be based on Flow Conservation
    # utilizing the "Clean" estimates from other interfaces.

    flow_targets = {} # if_id -> {'rx': val, 'tx': val}

    for router_id, interfaces in topology.items():
        # Filter for known interfaces
        valid_ifs = [i for i in interfaces if i in state]

        # Calculate Router-wide totals (using clean estimates)
        total_in = sum(state[i]['clean_rx'] for i in valid_ifs)
        total_out = sum(state[i]['clean_tx'] for i in valid_ifs)

        # Calculate individual flow targets
        # Conservation: Sum(Incoming) = Sum(Outgoing)
        # For interface i: RX_i + Other_RXs = TX_i + Other_TXs
        # Therefore: RX_i = (TX_i + Other_TXs) - Other_RXs
        # Wait, simple flow conservation is Total_In = Total_Out.
        # So we estimate RX_i to be the value that balances the equation.
        # RX_i_target = Total_Out - (Total_In - RX_i_current)

        for if_id in valid_ifs:
            curr = state[if_id]

            # Estimate RX target
            # What RX is needed to balance the Total Output?
            # other_rx = total_in - curr['clean_rx']
            # target_rx = total_out - other_rx
            rx_target = max(0.0, total_out - (total_in - curr['clean_rx']))

            # Estimate TX target
            # What TX is needed to balance the Total Input?
            # other_tx = total_out - curr['clean_tx']
            # target_tx = total_in - other_tx
            tx_target = max(0.0, total_in - (total_out - curr['clean_tx']))

            flow_targets[if_id] = {'rx': rx_target, 'tx': tx_target}

    # --- Phase 3: Arbitration & Repair ---

    for if_id, d in state.items():
        # Get Flow Signals (fallback to self if calculation failed/missing)
        f_sig = flow_targets.get(if_id, {'rx': d['clean_rx'], 'tx': d['clean_tx']})

        # --- Repair RX ---
        if d['status'] == 'down':
            final_rx = 0.0
            # Confidence is high unless we are suppressing significant noise
            conf_rx = 1.0 if d['s_rx'] < ABS_TOL else 0.9
        else:
            # Inputs: Self (s_rx), Peer (p_tx), Flow (f_sig['rx'])
            # Note: Peer TX is the source for My RX
            final_rx, conf_rx = arbitrate(d['s_rx'], d['p_tx'], f_sig['rx'], agrees)

        # --- Repair TX ---
        if d['status'] == 'down':
            final_tx = 0.0
            conf_tx = 1.0 if d['s_tx'] < ABS_TOL else 0.9
        else:
            # Inputs: Self (s_tx), Peer (p_rx), Flow (f_sig['tx'])
            # Note: Peer RX is the destination for My TX
            final_tx, conf_tx = arbitrate(d['s_tx'], d['p_rx'], f_sig['tx'], agrees)

        # Store Result
        orig_data = telemetry[if_id]
        results[if_id] = {
            'rx_rate': (d['s_rx'], final_rx, conf_rx),
            'tx_rate': (d['s_tx'], final_tx, conf_tx),
            'interface_status': (d['orig_status'], d['status'], d['status_conf']),
            'connected_to': orig_data.get('connected_to'),
            'local_router': orig_data.get('local_router'),
            'remote_router': orig_data.get('remote_router')
        }

    return results

def arbitrate(v_self: float, v_peer: float, v_flow: float, agree_func) -> Tuple[float, float]:
    """
    Arbitrates between three signals: Self, Peer, and Flow.
    Returns (repaired_value, confidence).
    """
    # 1. Handle Missing Peer (No redundancy from Link Symmetry)
    if v_peer is None:
        if agree_func(v_self, v_flow):
            return (v_self + v_flow) / 2.0, 0.9
        else:
            # Low confidence fallback to Self
            return v_self, 0.6

    # 2. Check Consensus Agreements
    sp = agree_func(v_self, v_peer) # Self-Peer
    pf = agree_func(v_peer, v_flow) # Peer-Flow
    sf = agree_func(v_self, v_flow) # Self-Flow

    # 3. Decision Logic

    # CASE A: Unanimous Agreement
    if sp and pf:
        return (v_self + v_peer + v_flow) / 3.0, 1.0

    # CASE B: Self & Peer agree (Flow is outlier)
    # Common case: One bad interface elsewhere on router corrupts Flow.
    # Link Symmetry is very strong.
    if sp:
        return (v_self + v_peer) / 2.0, 0.95

    # CASE C: Peer & Flow agree (Self is outlier)
    # Strong evidence that local sensor is broken.
    if pf:
        return (v_peer + v_flow) / 2.0, 0.90

    # CASE D: Self & Flow agree (Peer is outlier)
    # Ambiguous: Either Peer is broken, OR I am forwarding noise/amplification.
    # "I see it (Self) and I forwarded it (Flow), but Peer didn't send/receive it".
    # Since Peer is usually the ground truth for what is on the wire, this is lower confidence.
    if sf:
        return (v_self + v_flow) / 2.0, 0.80

    # CASE E: Total Disagreement
    # Fallback to Peer. Link Symmetry is the most robust physical invariant.
    return v_peer, 0.60
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
