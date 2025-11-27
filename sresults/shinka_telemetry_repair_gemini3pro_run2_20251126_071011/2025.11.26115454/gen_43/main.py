# EVOLVE-BLOCK-START
"""
Network telemetry repair algorithm using Analytic Flow Consensus.
"""
from typing import Dict, Any, Tuple, List
import math

def repair_network_telemetry(telemetry: Dict[str, Dict[str, Any]],
                             topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:
    """
    Repair network telemetry using Dual-Flow Consensus.

    Strategy:
    1. Status Repair: Infer status from local/remote traffic activity.
    2. Rate Repair (Iterative):
       - Calculate 'Flow Implied' values for both Local and Peer routers.
       - Assess 'Solidity' of Peer values (consistency with Peer Router balance).
       - Arbitrate using a consensus of: Self, Peer, Local Flow, Peer Flow.
    3. Confidence Calibration:
       - Continuous scoring based on residual errors and agreement strength.
    """

    # --- Configuration ---
    HARDENING_THRESHOLD = 0.02   # 2% relative error considered 'match'
    BASE_NOISE_FLOOR = 10.0      # Minimum Mbps to consider 'active'
    ITERATIONS = 4               # Convergence count

    # --- Helper: Dynamic Noise Floor ---
    def get_noise_floor(rate_a, rate_b=0.0):
        # Scale noise floor for high speed links (0.5%)
        mx = max(rate_a, rate_b)
        return max(BASE_NOISE_FLOOR, mx * 0.005)

    # --- Helper: Normalized Error ---
    def calc_error(v1, v2):
        nf = get_noise_floor(v1, v2)
        return abs(v1 - v2) / max(v1, v2, nf)

    # --- Step 1: Initialization ---
    state = {}
    for if_id, data in telemetry.items():
        state[if_id] = {
            'rx': float(data.get('rx_rate', 0.0)),
            'tx': float(data.get('tx_rate', 0.0)),
            'status': data.get('interface_status', 'unknown'),
            'peer': data.get('connected_to'),
            'router': data.get('local_router'),
            'orig_rx': float(data.get('rx_rate', 0.0)),
            'orig_tx': float(data.get('tx_rate', 0.0)),
            'orig_status': data.get('interface_status', 'unknown')
        }

    # --- Step 2: Robust Status Repair ---
    for if_id, s in state.items():
        # Traffic Evidence
        local_traffic = s['rx'] > BASE_NOISE_FLOOR or s['tx'] > BASE_NOISE_FLOOR

        peer_traffic = False
        peer_is_down = False
        if s['peer'] and s['peer'] in state:
            p = state[s['peer']]
            # Use original values for peer status evidence to avoid feedback loops initially
            if p['orig_rx'] > BASE_NOISE_FLOOR or p['orig_tx'] > BASE_NOISE_FLOOR:
                peer_traffic = True
            if p['orig_status'] == 'down':
                peer_is_down = True

        # Decision Matrix
        if local_traffic or peer_traffic:
            s['status'] = 'up'
        elif peer_is_down and not local_traffic:
            s['status'] = 'down'

        # Consistency enforce
        if s['status'] != 'up':
            s['rx'] = 0.0
            s['tx'] = 0.0

    # --- Step 3: Iterative Rate Repair ---
    for _ in range(ITERATIONS):

        # 3.1: Pre-calculate Router Flow States
        router_stats = {}
        for r_id, if_ids in topology.items():
            sum_rx = sum(state[i]['rx'] for i in if_ids if i in state)
            sum_tx = sum(state[i]['tx'] for i in if_ids if i in state)
            total_vol = max(sum_rx, sum_tx, BASE_NOISE_FLOOR)
            imbalance = sum_rx - sum_tx
            router_stats[r_id] = {
                'imbalance': imbalance,
                'quality': max(0.0, 1.0 - (abs(imbalance) / total_vol * 10.0))
            }

        next_values = {}

        for if_id, s in state.items():
            if s['status'] != 'up':
                next_values[if_id] = {'rx': 0.0, 'tx': 0.0}
                continue

            peer_id = s['peer']
            r_id = s['router']
            has_peer = peer_id and peer_id in state

            # --- Gather Signals ---
            # 1. Local Flow Implied
            local_flow_rx = None
            local_flow_tx = None
            if r_id in router_stats:
                rs = router_stats[r_id]
                # NewRX = OldRX - Imbalance
                local_flow_rx = max(0.0, s['rx'] - rs['imbalance'])
                # NewTX = OldTX + Imbalance
                local_flow_tx = max(0.0, s['tx'] + rs['imbalance'])

            # 2. Peer Flow Implied (Validity check for Peer)
            peer_flow_tx = None # What Peer TX should be
            peer_flow_rx = None # What Peer RX should be
            peer_router_valid = False

            if has_peer:
                p_r_id = state[peer_id]['router']
                if p_r_id in router_stats:
                    prs = router_stats[p_r_id]
                    p_tx = state[peer_id]['tx']
                    p_rx = state[peer_id]['rx']
                    # Peer TX implied = P_TX + P_Imbalance
                    peer_flow_tx = max(0.0, p_tx + prs['imbalance'])
                    # Peer RX implied = P_RX - P_Imbalance
                    peer_flow_rx = max(0.0, p_rx - prs['imbalance'])
                    peer_router_valid = True

            # --- RX Repair ---
            # Conflict: Self RX vs Peer TX
            val_self = s['rx']
            val_peer = state[peer_id]['tx'] if has_peer else None

            final_rx = val_self

            if val_peer is not None:
                # 1. Check Impossible (RX > Peer TX)
                if val_self > val_peer * (1.0 + HARDENING_THRESHOLD):
                    # Usually clamp to Peer TX.
                    # Exception: If Peer TX is demonstrably wrong (Peer Flow requires higher TX).
                    is_peer_invalid = False
                    if peer_router_valid:
                         if calc_error(val_peer, peer_flow_tx) > HARDENING_THRESHOLD:
                             if calc_error(val_self, peer_flow_tx) < HARDENING_THRESHOLD:
                                 is_peer_invalid = True

                    if is_peer_invalid:
                        final_rx = val_self # Keep higher value
                    else:
                        final_rx = val_peer # Clamp

                # 2. Check Loss/Agreement (RX < Peer TX)
                else:
                     if calc_error(val_self, val_peer) < HARDENING_THRESHOLD:
                         final_rx = (val_self + val_peer) / 2.0
                     else:
                         # Arbitration
                         use_peer = True # Default to Source Truth

                         if local_flow_rx is not None:
                             err_self_flow = calc_error(val_self, local_flow_rx)
                             err_peer_flow = calc_error(val_peer, local_flow_rx)

                             if err_self_flow < err_peer_flow * 0.5:
                                 use_peer = False # Local Flow confirms Self
                             elif err_peer_flow < err_self_flow * 0.5:
                                 use_peer = True # Local Flow confirms Peer
                             else:
                                 # Ambiguous Local Flow. Check Peer Solidity.
                                 if peer_router_valid:
                                     # If Peer TX matches Peer Flow, it's solid.
                                     if calc_error(val_peer, peer_flow_tx) < HARDENING_THRESHOLD:
                                         use_peer = True
                                     else:
                                         use_peer = False

                         final_rx = val_peer if use_peer else val_self

            # --- TX Repair ---
            # Conflict: Self TX vs Peer RX
            val_self = s['tx']
            val_peer = state[peer_id]['rx'] if has_peer else None

            final_tx = val_self

            if val_peer is not None:
                 # 1. Impossible (TX < Peer RX)
                 if val_self < val_peer * (1.0 - HARDENING_THRESHOLD):
                     # Usually boost to Peer RX.
                     # Exception: Peer RX is over-reporting?
                     is_peer_invalid = False
                     if peer_router_valid:
                         if calc_error(val_peer, peer_flow_rx) > HARDENING_THRESHOLD:
                             if calc_error(val_self, peer_flow_rx) < HARDENING_THRESHOLD:
                                 is_peer_invalid = True

                     if is_peer_invalid:
                         final_tx = val_self
                     else:
                         final_tx = val_peer

                 # 2. Surplus (TX > Peer RX)
                 else:
                     if calc_error(val_self, val_peer) < HARDENING_THRESHOLD:
                         final_tx = (val_self + val_peer) / 2.0
                     else:
                         # Arbitration
                         # TX > RX is often Packet Loss on the wire. Source (TX) is usually right.
                         use_peer = False

                         if local_flow_tx is not None:
                             err_self_flow = calc_error(val_self, local_flow_tx)
                             err_peer_flow = calc_error(val_peer, local_flow_tx)

                             if err_peer_flow < err_self_flow * 0.5:
                                 # Flow says we didn't send that much (Phantom TX)
                                 use_peer = True
                             elif err_self_flow < err_peer_flow * 0.5:
                                 # Flow confirms we sent it
                                 use_peer = False

                         final_tx = val_peer if use_peer else val_self

            next_values[if_id] = {'rx': final_rx, 'tx': final_tx}

        # Apply updates
        for if_id, vals in next_values.items():
            state[if_id]['rx'] = vals['rx']
            state[if_id]['tx'] = vals['tx']

    # --- Step 4: Confidence Calibration ---
    result = {}

    # Final Quality
    final_router_qual = {}
    for r_id, if_ids in topology.items():
        sum_rx = sum(state[i]['rx'] for i in if_ids if i in state)
        sum_tx = sum(state[i]['tx'] for i in if_ids if i in state)
        vol = max(sum_rx, sum_tx, BASE_NOISE_FLOOR)
        imb = abs(sum_rx - sum_tx) / vol
        final_router_qual[r_id] = max(0.0, 1.0 - (imb * 10.0))

    for if_id, s in state.items():
        orig_rx = s['orig_rx']
        orig_tx = s['orig_tx']

        peer_id = s['peer']
        has_peer = peer_id and peer_id in state

        peer_tx = state[peer_id]['tx'] if has_peer else None
        peer_rx = state[peer_id]['rx'] if has_peer else None

        r_id = s['router']
        flow_q = final_router_qual.get(r_id, 0.5)

        def get_conf(final, orig, peer_val):
            # 1. Change Penalty
            changed = calc_error(final, orig) > HARDENING_THRESHOLD

            base_conf = 1.0
            if changed: base_conf = 0.85

            # 2. Peer Agreement Reward/Penalty
            if peer_val is not None:
                err_peer = calc_error(final, peer_val)
                if err_peer < HARDENING_THRESHOLD:
                    base_conf += 0.1 # Boost
                else:
                    base_conf -= 0.1 # Penalty for disagreeing with Peer

            # 3. Flow Quality Scaling
            flow_mult = 0.8 + (0.2 * flow_q)

            return min(1.0, max(0.0, base_conf * flow_mult))

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
