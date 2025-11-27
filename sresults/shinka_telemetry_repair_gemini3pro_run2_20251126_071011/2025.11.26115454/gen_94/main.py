# EVOLVE-BLOCK-START
"""
Network telemetry repair algorithm using Symmetric Golden Truth & Differential Trust.
"""
from typing import Dict, Any, Tuple, List
import math

def repair_network_telemetry(telemetry: Dict[str, Dict[str, Any]],
                             topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:
    """
    Repair network telemetry using Symmetric Golden Truth & Differential Trust.

    Strategy:
    1. Status Repair: Infer status from local/remote traffic activity.
    2. Rate Repair (Iterative):
       - Calculate 'Dual Flow Targets':
         - Local Target: Value needed to balance local router.
         - Remote Target: Value implied by Peer balancing the remote router.
       - Identify 'Golden Truth': If Local Target ≈ Remote Target, this value satisfies
         Global Conservation and is extremely reliable.
       - Logic:
         - Prioritize Golden Truth.
         - Apply 'Differential Trust': If Remote Router is healthy and Local is bad,
           trust the Remote Target.
         - Enforce Physics (RX <= Peer TX), handling Phantom Traffic vs Loss.
    3. Confidence Calibration:
       - High scores for Golden Truth repairs.
       - Damping factor based on final residual imbalance (penalize if we failed to balance).
    """

    # --- Configuration ---
    HARDENING_THRESHOLD = 0.02   # 2% relative error considered 'match'
    STRICT_THRESHOLD = 0.005     # 0.5% tolerance for physical impossibility
    GOLDEN_THRESHOLD = 0.01      # 1% agreement for Golden Truth
    BASE_NOISE_FLOOR = 10.0      # Minimum Mbps
    ITERATIONS = 5               # Convergence count

    # --- Helper: Dynamic Noise Floor ---
    def get_noise_floor(rate_a, rate_b=0.0):
        mx = max(rate_a, rate_b)
        return max(BASE_NOISE_FLOOR, mx * 0.001)

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
            'remote_router': data.get('remote_router'),
            'orig_rx': float(data.get('rx_rate', 0.0)),
            'orig_tx': float(data.get('tx_rate', 0.0)),
            'orig_status': data.get('interface_status', 'unknown')
        }

    # --- Step 2: Robust Status Repair ---
    for if_id, s in state.items():
        nf = get_noise_floor(s['rx'], s['tx'])
        local_traffic = s['rx'] > nf or s['tx'] > nf

        peer_traffic = False
        peer_down_flag = False

        if s['peer'] and s['peer'] in state:
            p = state[s['peer']]
            p_nf = get_noise_floor(p['orig_rx'], p['orig_tx'])
            if p['orig_rx'] > p_nf or p['orig_tx'] > p_nf:
                peer_traffic = True
            if p['orig_status'] == 'down':
                peer_down_flag = True

        if local_traffic or peer_traffic:
            s['status'] = 'up'
        elif peer_down_flag and not local_traffic:
            s['status'] = 'down'

        if s['status'] != 'up':
            s['rx'] = 0.0
            s['tx'] = 0.0

    # --- Step 3: Iterative Rate Repair ---
    for _ in range(ITERATIONS):

        # 3.1: Analyze Router Flows
        router_stats = {}
        for r_id, if_ids in topology.items():
            sum_rx = sum(state[i]['rx'] for i in if_ids if i in state)
            sum_tx = sum(state[i]['tx'] for i in if_ids if i in state)
            vol = max(sum_rx, sum_tx, BASE_NOISE_FLOOR)
            imbalance = sum_rx - sum_tx
            router_stats[r_id] = {
                'imbalance': imbalance,
                'vol': vol,
                'quality': max(0.0, 1.0 - (abs(imbalance) / vol * 10.0))
            }

        next_values = {}

        for if_id, s in state.items():
            if s['status'] != 'up':
                next_values[if_id] = {'rx': 0.0, 'tx': 0.0}
                continue

            peer_id = s['peer']
            local_r = s['router']
            remote_r = s['remote_router']
            has_peer = peer_id and peer_id in state

            # --- RX Targets ---
            # 1. Local Target: RX needed to balance Local Router
            local_target_rx = None
            local_q = 0.5
            if local_r in router_stats:
                ls = router_stats[local_r]
                local_q = ls['quality']
                local_target_rx = max(0.0, s['rx'] - ls['imbalance'])

            # 2. Remote Target: Peer TX needed to balance Remote Router (which equals Local RX)
            # Peer TX is outgoing on Remote. To balance Remote (SumRX - SumTX = Imb),
            # we need to increase Peer TX by Imbalance.
            remote_target_rx = None
            remote_q = 0.5
            if has_peer and remote_r in router_stats:
                rs = router_stats[remote_r]
                remote_q = rs['quality']
                peer_tx_current = state[peer_id]['tx']
                remote_target_rx = max(0.0, peer_tx_current + rs['imbalance'])

            # 3. Golden Truth Check
            golden_rx = None
            if local_target_rx is not None and remote_target_rx is not None:
                if calc_error(local_target_rx, remote_target_rx) < GOLDEN_THRESHOLD:
                    golden_rx = (local_target_rx + remote_target_rx) / 2.0

            # --- RX Decision ---
            val_self = s['rx']
            val_peer = state[peer_id]['tx'] if has_peer else None
            final_rx = val_self

            if val_peer is not None:
                if golden_rx is not None:
                    final_rx = golden_rx
                elif val_self > val_peer * (1.0 + STRICT_THRESHOLD):
                    # Physics (Phantom).
                    # Exception: If Local strictly needs this extra traffic to balance, AND Remote is bad/ambiguous.
                    # But if Remote is Good, we trust Peer (and clamp).
                    if remote_q > 0.8:
                        final_rx = val_peer
                    else:
                        # Trust local if local target matches current
                        if local_target_rx is not None and calc_error(val_self, local_target_rx) < HARDENING_THRESHOLD:
                            final_rx = val_self
                        else:
                            final_rx = val_peer
                elif calc_error(val_self, val_peer) < HARDENING_THRESHOLD:
                    final_rx = (val_self + val_peer) / 2.0
                else:
                    # Disagreement (Likely Loss: RX < Peer TX)
                    # Differential Trust
                    if remote_q > 0.9 and local_q < 0.6:
                        final_rx = remote_target_rx if remote_target_rx is not None else val_peer
                    elif local_q > 0.9 and remote_q < 0.6:
                        final_rx = local_target_rx if local_target_rx is not None else val_self
                    else:
                        # Arbitration
                        if local_target_rx is not None:
                            d_peer = calc_error(val_peer, local_target_rx)
                            d_self = calc_error(val_self, local_target_rx)
                            if d_peer < d_self:
                                final_rx = val_peer
                            else:
                                final_rx = val_self
                        else:
                            final_rx = val_peer

            # --- TX Targets ---
            # 1. Local Target: TX needed to balance Local
            # To balance (SumRX - SumTX = Imb), increase TX by Imb.
            local_target_tx = None
            if local_r in router_stats:
                ls = router_stats[local_r]
                local_target_tx = max(0.0, s['tx'] + ls['imbalance'])

            # 2. Remote Target: Peer RX needed to balance Remote (equals Local TX)
            # Peer RX is incoming on Remote. To balance: Peer_RX_new = Peer_RX_old - Imbalance
            remote_target_tx = None
            if has_peer and remote_r in router_stats:
                rs = router_stats[remote_r]
                peer_rx_current = state[peer_id]['rx']
                remote_target_tx = max(0.0, peer_rx_current - rs['imbalance'])

            # 3. Golden Truth
            golden_tx = None
            if local_target_tx is not None and remote_target_tx is not None:
                if calc_error(local_target_tx, remote_target_tx) < GOLDEN_THRESHOLD:
                    golden_tx = (local_target_tx + remote_target_tx) / 2.0

            # --- TX Decision ---
            val_self = s['tx']
            val_peer = state[peer_id]['rx'] if has_peer else None
            final_tx = val_self

            if val_peer is not None:
                if golden_tx is not None:
                    final_tx = golden_tx
                elif val_self < val_peer * (1.0 - STRICT_THRESHOLD):
                    # Physics (Impossible TX < Peer RX).
                    final_tx = val_peer
                elif calc_error(val_self, val_peer) < HARDENING_THRESHOLD:
                    final_tx = (val_self + val_peer) / 2.0
                else:
                    # Disagreement (TX > Peer RX)
                    if remote_q > 0.9 and local_q < 0.6:
                        final_tx = remote_target_tx if remote_target_tx is not None else val_peer
                    elif local_q > 0.9 and remote_q < 0.6:
                        final_tx = local_target_tx if local_target_tx is not None else val_self
                    else:
                        if local_target_tx is not None:
                            d_peer = calc_error(val_peer, local_target_tx)
                            d_self = calc_error(val_self, local_target_tx)
                            if d_peer < d_self:
                                final_tx = val_peer
                            else:
                                final_tx = val_self
                        else:
                            final_tx = val_peer

            next_values[if_id] = {'rx': final_rx, 'tx': final_tx}

        # Apply
        for if_id, vals in next_values.items():
            state[if_id]['rx'] = vals['rx']
            state[if_id]['tx'] = vals['tx']

    # --- Step 4: Confidence Calibration ---
    result = {}

    # Final Stats for Calibration
    final_stats = {}
    for r_id, if_ids in topology.items():
        sum_rx = sum(state[i]['rx'] for i in if_ids if i in state)
        sum_tx = sum(state[i]['tx'] for i in if_ids if i in state)
        vol = max(sum_rx, sum_tx, BASE_NOISE_FLOOR)
        imb = abs(sum_rx - sum_tx)
        final_stats[r_id] = {
            'quality': max(0.0, 1.0 - (imb / vol * 10.0)),
            'imbalance_ratio': min(1.0, imb / vol)
        }

    for if_id, s in state.items():
        orig_rx = s['orig_rx']
        orig_tx = s['orig_tx']

        peer_id = s['peer']
        has_peer = peer_id and peer_id in state

        peer_tx = state[peer_id]['tx'] if has_peer else None
        peer_rx = state[peer_id]['rx'] if has_peer else None

        ls = final_stats.get(s['router'], {'quality': 0.5, 'imbalance_ratio': 0.0})
        rs = final_stats.get(s['remote_router'], {'quality': 0.5}) if s['remote_router'] else {'quality': 0.5}

        local_q = ls['quality']
        remote_q = rs['quality']

        # Check for Golden Truth status again (post-repair consistency)
        # Re-calc targets with final values
        # (Simplified: just check if Local Target matches Peer Target)
        # Note: This is computationally expensive to redo exactly, so we infer from Q.

        def get_conf(final, orig, peer_val, l_q, r_q, imb_ratio):
            # 1. Base Score from Decision Type
            is_repaired = calc_error(final, orig) > HARDENING_THRESHOLD
            matches_peer = False
            if peer_val is not None and calc_error(final, peer_val) < HARDENING_THRESHOLD:
                matches_peer = True

            score = 1.0

            if is_repaired:
                if matches_peer:
                    # Consensus.
                    # If both routers are healthy, this is Golden.
                    if l_q > 0.9 and r_q > 0.9:
                        score = 0.98
                    else:
                        score = 0.88 + (0.05 * l_q) + (0.05 * r_q)
                else:
                    # Repaired to something else (e.g. Local Flow Target)
                    if l_q > 0.9:
                        score = 0.85
                    else:
                        score = 0.60
            else:
                # Kept Original
                if peer_val is not None and not matches_peer:
                    # Disagreement.
                    if l_q > 0.9 and r_q < 0.7:
                        score = 0.95 # Trust Self (I am good, you are bad)
                    elif l_q > 0.9 and r_q > 0.9:
                        score = 0.85 # Stalemate
                    elif l_q < 0.7:
                        score = 0.60 # I am bad, but kept my value? Low confidence.
                    else:
                        score = 0.75
                else:
                    score = 1.0

            # 2. Residual Damping
            # If the router is still unbalanced, reduce confidence in all its interfaces.
            # Factor: (1 - ImbalanceRatio)
            # e.g. 10% imbalance -> score * 0.9
            damping = 1.0 - (imb_ratio * 2.0) # Aggressive damping
            damping = max(0.5, min(1.0, damping))

            return max(0.0, min(1.0, score * damping))

        rx_conf = get_conf(s['rx'], orig_rx, peer_tx, local_q, remote_q, ls['imbalance_ratio'])
        tx_conf = get_conf(s['tx'], orig_tx, peer_rx, local_q, remote_q, ls['imbalance_ratio'])

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