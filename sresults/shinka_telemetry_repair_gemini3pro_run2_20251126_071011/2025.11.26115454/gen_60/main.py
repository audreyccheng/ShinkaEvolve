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
    Repair network telemetry using Dual-Source Flow Consensus and Physics Clamping.

    Strategy:
    1. Status Repair: Infer status from Local AND Peer traffic evidence.
    2. Rate Repair (Iterative):
       - Calculate 'Flow Implied' targets for both Local and Remote routers.
       - Arbitration Strategy (Minimize System Energy):
         - Cost(Value) = Distance(Value, Local_Flow_Target) + Distance(Value, Remote_Flow_Target)
         - Select value (Self vs Peer) that minimizes total flow error.
       - Physics Clamping: Strictly enforce RX <= Peer TX (Impossible region).
    3. Confidence:
       - Base confidence from decision path (Repaired vs Kept, Peer Match vs Mismatch).
       - Penalty: Linear reduction based on residual flow imbalance of the router.
    """

    # --- Configuration ---
    HARDENING_THRESHOLD = 0.02   # 2% relative error considered 'match'
    PHYSICS_THRESHOLD = 0.005    # 0.5% tolerance for physical impossibilities
    BASE_NOISE_FLOOR = 10.0      # Minimum Mbps
    ITERATIONS = 4               # Convergence count

    # --- Helper: Dynamic Error Calculation ---
    def get_noise_floor(v1, v2):
        return max(BASE_NOISE_FLOOR, max(v1, v2) * 0.005)

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
            'remote': data.get('remote_router'),
            'orig_rx': float(data.get('rx_rate', 0.0)),
            'orig_tx': float(data.get('tx_rate', 0.0)),
            'orig_status': data.get('interface_status', 'unknown')
        }

    # --- Status Repair ---
    for if_id, s in state.items():
        # Evidence
        local_active = s['rx'] > BASE_NOISE_FLOOR or s['tx'] > BASE_NOISE_FLOOR

        peer_active = False
        peer_down = False
        if s['peer'] and s['peer'] in state:
            p = state[s['peer']]
            if p['rx'] > BASE_NOISE_FLOOR or p['tx'] > BASE_NOISE_FLOOR:
                peer_active = True
            if p['status'] == 'down':
                peer_down = True

        if local_active or peer_active:
            s['status'] = 'up'
        elif peer_down and not local_active:
            s['status'] = 'down'

    # Consistency
    for s in state.values():
        if s['status'] != 'up':
            s['rx'] = 0.0
            s['tx'] = 0.0

    # --- Step 2: Iterative Rate Repair ---
    for _ in range(ITERATIONS):

        # 1. Calculate Router Imbalances
        router_stats = {}
        for r_id, if_list in topology.items():
            sum_rx = sum(state[i]['rx'] for i in if_list if i in state)
            sum_tx = sum(state[i]['tx'] for i in if_list if i in state)
            router_stats[r_id] = {
                'imbalance': sum_rx - sum_tx,
                'vol': max(sum_rx, sum_tx, BASE_NOISE_FLOOR)
            }

        updates = {}

        for if_id, s in state.items():
            if s['status'] != 'up':
                updates[if_id] = {'rx': 0.0, 'tx': 0.0}
                continue

            peer_id = s['peer']
            r_id = s['router']
            rem_id = s['remote']
            has_peer = peer_id and peer_id in state

            # Helper to get flow targets
            def get_flow_target(router_id, current_val, is_rx):
                if router_id not in router_stats:
                    return None
                imb = router_stats[router_id]['imbalance']
                # RX Target: val - imb. TX Target: val + imb.
                return max(0.0, current_val - imb) if is_rx else max(0.0, current_val + imb)

            # --- RX Repair ---
            val_self = s['rx']
            val_peer = state[peer_id]['tx'] if has_peer else None

            # Targets
            ft_local = get_flow_target(r_id, val_self, is_rx=True)
            # Remote TX Target: For the peer interface (TX), what value balances remote?
            ft_remote = get_flow_target(rem_id, val_peer, is_rx=False) if has_peer else None

            final_rx = val_self

            if val_peer is not None:
                # 1. Physics: RX > Peer TX (Impossible)
                if val_self > val_peer * (1.0 + PHYSICS_THRESHOLD):
                    final_rx = val_peer

                # 2. Agreement
                elif calc_error(val_self, val_peer) < HARDENING_THRESHOLD:
                    final_rx = (val_self + val_peer) / 2.0

                # 3. Arbitration
                else:
                    # Cost Analysis
                    # Cost = Dist(Val, LocalTarget) + Dist(Val, RemoteTarget)

                    def get_cost(val):
                        c = 0.0
                        cnt = 0
                        if ft_local is not None:
                            c += calc_error(val, ft_local)
                            cnt += 1
                        if ft_remote is not None:
                            c += calc_error(val, ft_remote)
                            cnt += 1
                        return c / max(1, cnt)

                    c_self = get_cost(val_self)
                    c_peer = get_cost(val_peer)

                    if c_peer < c_self:
                        final_rx = val_peer
                    elif c_self < c_peer:
                        final_rx = val_self
                    else:
                        final_rx = val_peer # Tie-break: Source Truth

            # --- TX Repair ---
            val_self = s['tx']
            val_peer = state[peer_id]['rx'] if has_peer else None

            ft_local = get_flow_target(r_id, val_self, is_rx=False)
            ft_remote = get_flow_target(rem_id, val_peer, is_rx=True) if has_peer else None

            final_tx = val_self

            if val_peer is not None:
                # 1. Physics: TX < Peer RX (Impossible)
                if val_self < val_peer * (1.0 - PHYSICS_THRESHOLD):
                    final_tx = val_peer

                # 2. Agreement
                elif calc_error(val_self, val_peer) < HARDENING_THRESHOLD:
                    final_tx = (val_self + val_peer) / 2.0

                # 3. Arbitration
                else:
                    def get_cost(val):
                        c = 0.0
                        cnt = 0
                        if ft_local is not None:
                            c += calc_error(val, ft_local)
                            cnt += 1
                        if ft_remote is not None:
                            c += calc_error(val, ft_remote)
                            cnt += 1
                        return c / max(1, cnt)

                    c_self = get_cost(val_self)
                    c_peer = get_cost(val_peer)

                    if c_peer < c_self:
                        final_tx = val_peer
                    elif c_self < c_peer:
                        final_tx = val_self
                    else:
                        final_tx = val_peer

            updates[if_id] = {'rx': final_rx, 'tx': final_tx}

        for if_id, vals in updates.items():
            state[if_id]['rx'] = vals['rx']
            state[if_id]['tx'] = vals['tx']

    # --- Step 3: Confidence Calibration ---
    result = {}

    # Final Imbalance Ratios
    router_imb_ratio = {}
    for r_id, if_list in topology.items():
        sum_rx = sum(state[i]['rx'] for i in if_list if i in state)
        sum_tx = sum(state[i]['tx'] for i in if_list if i in state)
        vol = max(sum_rx, sum_tx, BASE_NOISE_FLOOR)
        router_imb_ratio[r_id] = abs(sum_rx - sum_tx) / vol

    for if_id, s in state.items():
        orig_rx = s['orig_rx']
        orig_tx = s['orig_tx']

        peer_id = s['peer']
        has_peer = peer_id and peer_id in state
        r_id = s['router']

        peer_tx = state[peer_id]['tx'] if has_peer else None
        peer_rx = state[peer_id]['rx'] if has_peer else None

        imb = router_imb_ratio.get(r_id, 0.0)

        def get_conf(final, orig, peer_val, router_imb):
            is_changed = calc_error(final, orig) > HARDENING_THRESHOLD

            peer_match = False
            if peer_val is not None and calc_error(final, peer_val) < HARDENING_THRESHOLD:
                peer_match = True

            conf = 1.0

            if is_changed:
                if peer_match:
                    # Validated Repair (Link Agreement)
                    conf = 0.98
                else:
                    # Unvalidated Repair (Flow Forced)
                    conf = 0.75
            else:
                if peer_val is not None and not peer_match:
                    # Disagreement maintained (Source/Self bias)
                    # This is better than a bad repair, but still uncertain.
                    conf = 0.85
                else:
                    # Agreement or No Peer
                    conf = 1.0

            # Linear Penalty for Residual Imbalance
            # If router is messy, reduce confidence significantly
            # e.g. 5% imbalance -> 0.1 penalty.
            penalty = router_imb * 2.0
            conf = max(0.0, conf - penalty)

            return conf

        rx_conf = get_conf(s['rx'], orig_rx, peer_tx, imb)
        tx_conf = get_conf(s['tx'], orig_tx, peer_rx, imb)

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