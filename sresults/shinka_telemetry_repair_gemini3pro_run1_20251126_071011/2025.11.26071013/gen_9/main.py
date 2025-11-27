# EVOLVE-BLOCK-START
"""
Network telemetry repair algorithm using Iterative Invariant Consensus.
Refines telemetry data through multiple passes of constraint satisfaction,
allowing Flow Conservation hints to become cleaner and more accurate with each iteration.
"""
from typing import Dict, Any, Tuple, List
import copy


def repair_network_telemetry(telemetry: Dict[str, Dict[str, Any]],
                             topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:
    """
    Repair network telemetry using Bilateral Flow Consensus.

    Key Logic:
    1. Validate Status: Traffic implies UP.
    2. Check Link Symmetry: If Local matches Peer, trust the average.
    3. If Symmetry Broken:
       - Check 'Self-Consistency' of Local side (Does Local Val match Local Flow Hint?)
       - Check 'Self-Consistency' of Peer side (Does Peer Val match Peer Flow Hint?)
       - Trust the side that is more self-consistent (validates against its own router).
    """

    HARDENING_THRESHOLD = 0.02
    TRAFFIC_THRESHOLD = 1.0

    # --- Phase 1: Calculate Flow Hints (The "Should Be" values) ---
    flow_hints = {}

    for router_id, iface_ids in topology.items():
        valid_ifaces = [i for i in iface_ids if i in telemetry]
        sum_rx = sum(telemetry[i].get('rx_rate', 0.0) for i in valid_ifaces)
        sum_tx = sum(telemetry[i].get('tx_rate', 0.0) for i in valid_ifaces)

        for iface in valid_ifaces:
            data = telemetry[iface]
            # Hint RX = Sum_TX - (Sum_RX - My_RX)
            rx_hint = max(0.0, sum_tx - (sum_rx - data.get('rx_rate', 0.0)))
            # Hint TX = Sum_RX - (Sum_TX - My_TX)
            tx_hint = max(0.0, sum_rx - (sum_tx - data.get('tx_rate', 0.0)))
            flow_hints[iface] = {'rx': rx_hint, 'tx': tx_hint}

    result = {}

    # --- Phase 2: Repair Logic ---
    for iface_id, data in telemetry.items():
        orig_rx = data.get('rx_rate', 0.0)
        orig_tx = data.get('tx_rate', 0.0)
        orig_status = data.get('interface_status', 'unknown')

        peer_id = data.get('connected_to')
        peer_data = telemetry.get(peer_id, {}) if (peer_id and peer_id in telemetry) else {}
        has_peer = bool(peer_data)

        # --- A. Status Repair ---
        traffic_signals = [orig_rx, orig_tx, peer_data.get('rx_rate', 0.0), peer_data.get('tx_rate', 0.0)]
        max_traffic = max(traffic_signals) if traffic_signals else 0.0

        rep_status = orig_status
        conf_status = 1.0

        if max_traffic > TRAFFIC_THRESHOLD:
            # Traffic > Threshold implies UP
            if orig_status != 'up':
                rep_status = 'up'
                conf_status = 0.95
        elif orig_status == 'up' and peer_data.get('interface_status') == 'down':
            # Contradiction, no traffic -> likely DOWN
            rep_status = 'down'
            conf_status = 0.8

        # --- B. Rate Repair ---
        if rep_status == 'down':
            rep_rx, rep_tx = 0.0, 0.0
            # If we are changing values significantly, inherit status confidence
            conf_rx = conf_status if orig_rx > TRAFFIC_THRESHOLD else 1.0
            conf_tx = conf_status if orig_tx > TRAFFIC_THRESHOLD else 1.0
        elif has_peer:
            # Helper to resolve conflict using Bilateral Consistency
            def resolve(local_val, peer_val, local_hint, peer_hint):
                # 1. Check Symmetry
                denom_sym = max(local_val, peer_val, 1.0)
                diff_sym = abs(local_val - peer_val) / denom_sym

                if diff_sym <= HARDENING_THRESHOLD:
                    return (local_val + peer_val) / 2.0, 1.0

                # 2. Symmetry Broken: Check Self-Consistency
                # How well does Local match its own router's balance?
                denom_l = max(local_val, local_hint, 1.0) if local_hint is not None else 1.0
                dist_local = abs(local_val - local_hint) / denom_l if local_hint is not None else float('inf')

                # How well does Peer match its own router's balance?
                denom_p = max(peer_val, peer_hint, 1.0) if peer_hint is not None else 1.0
                dist_peer = abs(peer_val - peer_hint) / denom_p if peer_hint is not None else float('inf')

                if dist_local < dist_peer:
                    # Local is more self-consistent -> Trust Local
                    # Confidence reduced by how inconsistent it actually is
                    conf = max(0.6, 1.0 - dist_local)
                    return local_val, conf
                elif dist_peer < dist_local:
                    # Peer is more self-consistent -> Trust Peer
                    conf = max(0.6, 1.0 - dist_peer)
                    return peer_val, conf
                else:
                    # Both equally bad/good. Trust average with low confidence.
                    return (local_val + peer_val) / 2.0, 0.5

            # Repair RX (matches Peer TX)
            hint_rx = flow_hints.get(iface_id, {}).get('rx')
            peer_hint_tx = flow_hints.get(peer_id, {}).get('tx')
            rep_rx, conf_rx = resolve(orig_rx, peer_data.get('tx_rate', 0.0), hint_rx, peer_hint_tx)

            # Repair TX (matches Peer RX)
            hint_tx = flow_hints.get(iface_id, {}).get('tx')
            peer_hint_rx = flow_hints.get(peer_id, {}).get('rx')
            rep_tx, conf_tx = resolve(orig_tx, peer_data.get('rx_rate', 0.0), hint_tx, peer_hint_rx)
        else:
            # No peer, trust local values but check hint
            rep_rx, rep_tx = orig_rx, orig_tx
            conf_rx, conf_tx = 1.0, 1.0

            # Sanity check against hint if available
            hint_rx = flow_hints.get(iface_id, {}).get('rx')
            if hint_rx is not None:
                denom = max(orig_rx, hint_rx, 1.0)
                if abs(orig_rx - hint_rx) / denom > 0.5:
                    conf_rx = 0.6

            hint_tx = flow_hints.get(iface_id, {}).get('tx')
            if hint_tx is not None:
                denom = max(orig_tx, hint_tx, 1.0)
                if abs(orig_tx - hint_tx) / denom > 0.5:
                    conf_tx = 0.6

        result[iface_id] = {
            'rx_rate': (orig_rx, rep_rx, conf_rx),
            'tx_rate': (orig_tx, rep_tx, conf_tx),
            'interface_status': (orig_status, rep_status, conf_status),
            'connected_to': peer_id,
            'local_router': data.get('local_router'),
            'remote_router': data.get('remote_router')
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
