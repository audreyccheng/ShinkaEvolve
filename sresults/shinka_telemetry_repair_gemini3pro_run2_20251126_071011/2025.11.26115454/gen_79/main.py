# EVOLVE-BLOCK-START
"""
Network telemetry repair algorithm using Hybrid Flow Solidity Consensus.
"""
from typing import Dict, Any, Tuple, List
import math

def repair_network_telemetry(telemetry: Dict[str, Dict[str, Any]],
                             topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:
    """
    Repair network telemetry using Hybrid Flow Solidity Consensus.

    Combination Strategy:
    1. Repair Logic (from Dual-Flow Asymmetric): Uses 'Solidity' checks (verifying if
       Peer values match Remote Router flow) and 'Impossibility' logic to achieve
       high counter accuracy.
    2. Confidence Calibration (from Dual-Source Analytic): Uses continuous Router
       Quality metrics (Local and Remote) to calibrate confidence scores, significantly
       improving upon the subtractive penalty model.
    """

    # --- Configuration ---
    HARDENING_THRESHOLD = 0.02    # 2% general matching tolerance
    PHYSICS_THRESHOLD = 0.005     # 0.5% strict tolerance for physical impossibility
    SOLIDITY_THRESHOLD = 0.01     # 1% strict tolerance for Flow Validation
    BASE_NOISE_FLOOR = 10.0       # Minimum Mbps to consider 'active'
    ITERATIONS = 4                # Convergence count

    # --- Helper: Dynamic Noise Floor ---
    def get_noise_floor(rate_a, rate_b=0.0):
        # Scale noise floor for high speed links (0.1% of max rate)
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
        # Evidence
        local_traffic = s['rx'] > BASE_NOISE_FLOOR or s['tx'] > BASE_NOISE_FLOOR

        peer_traffic = False
        peer_is_down = False
        if s['peer'] and s['peer'] in state:
            p = state[s['peer']]
            if p['orig_rx'] > BASE_NOISE_FLOOR or p['orig_tx'] > BASE_NOISE_FLOOR:
                peer_traffic = True
            if p['orig_status'] == 'down':
                peer_is_down = True

        # Decision Matrix
        if local_traffic or peer_traffic:
            s['status'] = 'up'
        elif peer_is_down and not local_traffic:
            s['status'] = 'down'
        # Else: keep original

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
            vol = max(sum_rx, sum_tx, BASE_NOISE_FLOOR)
            imbalance = sum_rx - sum_tx
            quality = max(0.0, 1.0 - (abs(imbalance) / vol * 10.0))
            router_stats[r_id] = {
                'imbalance': imbalance,
                'quality': quality
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

            # --- Gather Context ---
            ls = router_stats.get(local_r, {'imbalance': 0.0, 'quality': 0.5})
            rs = router_stats.get(remote_r, {'imbalance': 0.0, 'quality': 0.5}) if remote_r else {'imbalance': 0.0, 'quality': 0.5}

            # --- Calculate Targets ---
            # 1. Local Implied Target (Value that balances Local Router)
            local_target_rx = max(0.0, s['rx'] - ls['imbalance'])
            local_target_tx = max(0.0, s['tx'] + ls['imbalance'])

            # 2. Remote Implied Target (Value at PEER that balances Remote Router)
            # Note: We need to translate this to Local Perspective.
            remote_target_tx_peer = 0.0 # Maps to Local RX
            remote_target_rx_peer = 0.0 # Maps to Local TX

            if has_peer:
                p_tx = state[peer_id]['tx']
                p_rx = state[peer_id]['rx']
                # Remote Imbalance = Rx - Tx.
                # To fix: Tx_new = Tx_old + Imbalance.
                remote_target_tx_peer = max(0.0, p_tx + rs['imbalance'])
                # To fix: Rx_new = Rx_old - Imbalance.
                remote_target_rx_peer = max(0.0, p_rx - rs['imbalance'])

            # --- RX Repair (Local RX vs Peer TX) ---
            val_self = s['rx']
            val_peer = state[peer_id]['tx'] if has_peer else 0.0

            final_rx = val_self

            if has_peer:
                # A. Golden Truth: Do Local and Remote targets agree?
                golden_rx = None
                if calc_error(local_target_rx, remote_target_tx_peer) < HARDENING_THRESHOLD:
                    golden_rx = (local_target_rx + remote_target_tx_peer) / 2.0

                if golden_rx is not None:
                    final_rx = golden_rx

                # B. Physics Violation (RX > Peer TX)
                elif val_self > val_peer * (1.0 + PHYSICS_THRESHOLD):
                     final_rx = val_peer

                # C. Agreement
                elif calc_error(val_self, val_peer) < HARDENING_THRESHOLD:
                    final_rx = (val_self + val_peer) / 2.0

                # D. Arbitration (Disagreement)
                else:
                    # Quality Dominance
                    if rs['quality'] > ls['quality'] + 0.3:
                         final_rx = val_peer
                    elif ls['quality'] > rs['quality'] + 0.3:
                         final_rx = val_self
                    else:
                        # Flow Error Minimization
                        err_self = calc_error(val_self, local_target_rx)
                        err_peer = calc_error(val_peer, local_target_rx)
                        if err_peer < err_self:
                             final_rx = val_peer
                        elif err_self < err_peer:
                             final_rx = val_self
                        else:
                             final_rx = val_peer

            # --- TX Repair (Local TX vs Peer RX) ---
            val_self = s['tx']
            val_peer = state[peer_id]['rx'] if has_peer else 0.0

            final_tx = val_self

            if has_peer:
                # A. Golden Truth
                golden_tx = None
                if calc_error(local_target_tx, remote_target_rx_peer) < HARDENING_THRESHOLD:
                    golden_tx = (local_target_tx + remote_target_rx_peer) / 2.0

                if golden_tx is not None:
                    final_tx = golden_tx

                # B. Physics Violation (TX < Peer RX)
                elif val_self < val_peer * (1.0 - PHYSICS_THRESHOLD):
                    final_tx = val_peer

                # C. Agreement
                elif calc_error(val_self, val_peer) < HARDENING_THRESHOLD:
                    final_tx = (val_self + val_peer) / 2.0

                # D. Arbitration
                else:
                    if rs['quality'] > ls['quality'] + 0.3:
                        final_tx = val_peer
                    elif ls['quality'] > rs['quality'] + 0.3:
                        final_tx = val_self
                    else:
                        err_self = calc_error(val_self, local_target_tx)
                        err_peer = calc_error(val_peer, local_target_tx)
                        if err_peer < err_self:
                            final_tx = val_peer
                        elif err_self < err_peer:
                            final_tx = val_self
                        else:
                            final_tx = val_peer

            next_values[if_id] = {'rx': final_rx, 'tx': final_tx}

        # Apply updates
        for if_id, vals in next_values.items():
            state[if_id]['rx'] = vals['rx']
            state[if_id]['tx'] = vals['tx']

    # --- Step 4: Confidence Calibration (Dual-Source Analytic) ---
    result = {}

    # Recalculate Final Flow Quality (1.0 = Perfect, 0.0 = Poor)
    final_router_qual = {}
    for r_id, if_ids in topology.items():
        sum_rx = sum(state[i]['rx'] for i in if_ids if i in state)
        sum_tx = sum(state[i]['tx'] for i in if_ids if i in state)
        vol = max(sum_rx, sum_tx, BASE_NOISE_FLOOR)
        imb = abs(sum_rx - sum_tx) / vol
        # Decay to 0.0 at 10% imbalance
        final_router_qual[r_id] = max(0.0, 1.0 - (imb * 10.0))

    for if_id, s in state.items():
        orig_rx = s['orig_rx']
        orig_tx = s['orig_tx']

        peer_id = s['peer']
        has_peer = peer_id and peer_id in state

        peer_tx = state[peer_id]['tx'] if has_peer else None
        peer_rx = state[peer_id]['rx'] if has_peer else None

        r_id = s['router']
        remote_r_id = s['remote_router']

        local_q = final_router_qual.get(r_id, 0.5)
        remote_q = final_router_qual.get(remote_r_id, 0.5) if remote_r_id else 0.5

        def get_confidence(final, orig, peer_val, l_q, r_q):
            dist_orig = calc_error(final, orig)

            matches_peer = False
            if peer_val is not None and calc_error(final, peer_val) < HARDENING_THRESHOLD:
                matches_peer = True

            conf = 1.0

            if dist_orig > HARDENING_THRESHOLD:
                # REPAIRED
                if matches_peer:
                    # Best case: Link Consensus + Flow Support
                    # Base 0.90
                    conf = 0.90 + (0.05 * l_q) + (0.04 * r_q)
                else:
                    # Repaired but no peer match (Arbitration)
                    # Relies on Local Quality
                    conf = 0.70 + (0.20 * l_q)
            else:
                # KEPT ORIGINAL
                if peer_val is not None and not matches_peer:
                    # Conflict: We defied Peer.
                    # Trustworthiness depends on Local Quality vs Remote Quality
                    if l_q > 0.9:
                        if r_q > 0.9:
                            conf = 0.85 # Stalemate
                        else:
                            conf = 0.95 # Strong Local, Weak Remote
                    elif l_q > 0.7:
                        conf = 0.80
                    else:
                        conf = 0.60 # Weak Local support
                else:
                    # Agreement or No Peer
                    conf = 1.0

            # Residual Penalty: If Local Router is still unbalanced, decrease confidence
            # If l_q is 1.0, penalty is 0. If l_q is 0.5, penalty is significant.
            penalty = (1.0 - l_q) * 0.2
            conf -= penalty

            return max(0.0, min(1.0, conf))

        rx_conf = get_confidence(s['rx'], orig_rx, peer_tx, local_q, remote_q)
        tx_conf = get_confidence(s['tx'], orig_tx, peer_rx, local_q, remote_q)

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
