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
    Repair network interface telemetry using Flow Consensus with Physics Clamping.

    Strategy:
    1. Status Repair: Infer status from Local AND Peer traffic evidence.
    2. Rate Repair (Iterative):
       - Calculate 'Flow Implied' value for each interface to balance routers.
       - Apply Strict Physics Constraints:
         - RX cannot exceed Peer TX (Phantom Traffic).
         - TX cannot be less than Peer RX (Creation on wire).
       - Arbitrate remaining disagreements (Loss/Noise) using Flow Consensus.
         - Compare distance of Self/Peer to the Flow Implied target.
    3. Confidence: Calibration based on repair magnitude, Peer agreement, and Flow Quality.
    """

    # --- Configuration ---
    HARDENING_THRESHOLD = 0.02    # 2% relative error considered 'match'
    STRICT_PHYSICS_LIMIT = 0.005  # 0.5% threshold for impossible physics (e.g. RX > Peer TX)
    BASE_NOISE_FLOOR = 10.0       # Minimum Mbps to consider 'active' or valid
    ITERATIONS = 4                # Convergence count

    # --- Helper: Dynamic Error Calculation ---
    def get_noise_floor(v1, v2):
        # Scale noise floor for high speed links (0.5%), but keep base floor
        return max(BASE_NOISE_FLOOR, max(v1, v2) * 0.005)

    def calc_error(v1, v2):
        nf = get_noise_floor(v1, v2)
        return abs(v1 - v2) / max(v1, v2, nf)

    # --- Step 1: Initialization & Status Repair ---
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

    # Robust Status Logic
    for if_id, s in state.items():
        has_local_traffic = s['rx'] > BASE_NOISE_FLOOR or s['tx'] > BASE_NOISE_FLOOR

        peer_has_traffic = False
        peer_is_down = False
        if s['peer'] and s['peer'] in state:
            p = state[s['peer']]
            if p['rx'] > BASE_NOISE_FLOOR or p['tx'] > BASE_NOISE_FLOOR:
                peer_has_traffic = True
            if p['status'] == 'down':
                peer_is_down = True

        if has_local_traffic or peer_has_traffic:
            s['status'] = 'up'
        elif peer_is_down and not has_local_traffic:
            s['status'] = 'down'

    for s in state.values():
        if s['status'] != 'up':
            s['rx'] = 0.0
            s['tx'] = 0.0

    # --- Step 2: Iterative Rate Repair ---
    for _ in range(ITERATIONS):

        # Calculate Router Flow Imbalances & Quality
        router_stats = {}
        for r_id, if_list in topology.items():
            sum_rx = sum(state[i]['rx'] for i in if_list if i in state)
            sum_tx = sum(state[i]['tx'] for i in if_list if i in state)
            net = sum_rx - sum_tx
            vol = max(sum_rx, sum_tx, BASE_NOISE_FLOOR)
            # Quality: 1.0 = perfect, 0.0 = >10% imbalance
            qual = max(0.0, 1.0 - (abs(net) / vol * 10.0))
            router_stats[r_id] = {'net': net, 'qual': qual}

        updates = {}

        for if_id, s in state.items():
            if s['status'] != 'up':
                updates[if_id] = {'rx': 0.0, 'tx': 0.0}
                continue

            peer_id = s['peer']
            r_id = s['router']
            has_peer = peer_id and peer_id in state

            # Context for Arbitration
            rs_local = router_stats.get(r_id, {'net': 0.0, 'qual': 0.5})

            # --- RX Repair ---
            val_self = s['rx']
            val_peer = state[peer_id]['tx'] if has_peer else None
            # Flow Implied: val_self - net (if net > 0, we need less RX; if net < 0, we need more RX)
            val_flow = max(0.0, val_self - rs_local['net'])

            final_rx = val_self
            if val_peer is not None:
                # 1. Physics Clamping (Strict)
                # RX > Peer TX is physically impossible (no creation on wire).
                if val_self > val_peer * (1.0 + STRICT_PHYSICS_LIMIT):
                    final_rx = val_peer

                # 2. Agreement
                elif calc_error(val_self, val_peer) < HARDENING_THRESHOLD:
                    final_rx = (val_self + val_peer) / 2.0

                # 3. Flow Arbitration (RX < Peer TX likely)
                else:
                    # Dual-Source Arbitration (Recommendation 1)
                    # Check Remote Router Quality
                    remote_qual = 0.5
                    if has_peer:
                        remote_rid = state[peer_id]['router']
                        remote_qual = router_stats.get(remote_rid, {}).get('qual', 0.5)

                    # If Local is noisy but Remote is clean, trust Peer (Remote TX)
                    if remote_qual > 0.95 and rs_local['qual'] < 0.8:
                        final_rx = val_peer
                    else:
                        # Use Distance to Local Flow Target
                        err_self = calc_error(val_self, val_flow)
                        err_peer = calc_error(val_peer, val_flow)

                        if err_peer < err_self:
                            final_rx = val_peer
                        elif err_self < err_peer:
                            final_rx = val_self
                        else:
                            final_rx = val_peer # Default to Peer

            # --- TX Repair ---
            val_self = s['tx']
            val_peer = state[peer_id]['rx'] if has_peer else None
            # Flow Implied: val_self + net (if net > 0, we need more TX; if net < 0, we need less TX)
            val_flow = max(0.0, val_self + rs_local['net'])

            final_tx = val_self
            if val_peer is not None:
                # 1. Physics Clamping (Strict)
                # TX < Peer RX is physically impossible
                if val_self < val_peer * (1.0 - STRICT_PHYSICS_LIMIT):
                    final_tx = val_peer

                # 2. Agreement
                elif calc_error(val_self, val_peer) < HARDENING_THRESHOLD:
                    final_tx = (val_self + val_peer) / 2.0

                # 3. Flow Arbitration (TX > Peer RX likely)
                else:
                    remote_qual = 0.5
                    if has_peer:
                        remote_rid = state[peer_id]['router']
                        remote_qual = router_stats.get(remote_rid, {}).get('qual', 0.5)

                    if remote_qual > 0.95 and rs_local['qual'] < 0.8:
                        final_tx = val_peer
                    else:
                        err_self = calc_error(val_self, val_flow)
                        err_peer = calc_error(val_peer, val_flow)

                        if err_peer < err_self:
                            final_tx = val_peer
                        elif err_self < err_peer:
                            final_tx = val_self
                        else:
                            final_tx = val_peer

            updates[if_id] = {'rx': final_rx, 'tx': final_tx}

        # Apply updates synchronously
        for if_id, vals in updates.items():
            state[if_id]['rx'] = vals['rx']
            state[if_id]['tx'] = vals['tx']

    # --- Step 3: Confidence Calibration ---
    result = {}

    # Final Flow Quality
    router_final_stats = {}
    for r_id, if_list in topology.items():
        sum_rx = sum(state[i]['rx'] for i in if_list if i in state)
        sum_tx = sum(state[i]['tx'] for i in if_list if i in state)
        vol = max(sum_rx, sum_tx, BASE_NOISE_FLOOR)
        imbalance = abs(sum_rx - sum_tx)
        router_final_stats[r_id] = {'imb_ratio': imbalance / vol}

    for if_id, s in state.items():
        orig_rx = s['orig_rx']
        orig_tx = s['orig_tx']

        peer_id = s['peer']
        has_peer = peer_id and peer_id in state
        r_id = s['router']

        peer_tx = state[peer_id]['tx'] if has_peer else None
        peer_rx = state[peer_id]['rx'] if has_peer else None

        rs = router_final_stats.get(r_id, {'imb_ratio': 0.0})
        flow_penalty = rs['imb_ratio'] * 2.0 # Linear penalty (Rec 3)

        def get_conf(final, orig, peer_val):
            was_repaired = calc_error(final, orig) > HARDENING_THRESHOLD
            peer_supports = (peer_val is not None) and (calc_error(final, peer_val) < HARDENING_THRESHOLD)

            conf = 1.0

            # Base Buckets
            if was_repaired:
                if peer_supports:
                    conf = 0.98 # Validated by Link
                else:
                    conf = 0.85 # Flow forced / Averaging
            else:
                if peer_val is not None and not peer_supports:
                    conf = 0.80 # Trusted Self over Peer
                else:
                    conf = 1.0 # Agreement

            # Apply Flow Penalty (Hybrid Model)
            conf = max(0.0, conf - flow_penalty)

            # Cap at 1.0
            return min(1.0, conf)

        rx_conf = get_conf(s['rx'], orig_rx, peer_tx)
        tx_conf = get_conf(s['tx'], orig_tx, peer_rx)

        # Status confidence
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