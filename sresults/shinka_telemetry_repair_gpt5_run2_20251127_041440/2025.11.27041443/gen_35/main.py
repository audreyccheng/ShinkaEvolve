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
    """
    # Hardened tolerances (Hodor guidance)
    HARDENING_THRESHOLD = 0.02   # 2% for normal rates
    LOW_RATE_CUTOFF = 10.0       # Mbps threshold for tiny flows
    LOW_RATE_THRESHOLD = 0.05    # 5% when small
    ABS_GUARD = 0.5              # Mbps; absolute guard to avoid over-correcting tiny flows

    def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
        return max(lo, min(hi, x))

    result: Dict[str, Dict[str, Tuple]] = {}

    # Build peer mapping for quick, validated lookup
    peers: Dict[str, str] = {}
    for if_id, data in telemetry.items():
        peer_id = data.get('connected_to')
        peers[if_id] = peer_id if peer_id in telemetry else None

    # Plan pairwise consensus adjustments so both ends change consistently
    field_value_adjust: Dict[Tuple[str, str], float] = {}   # (iface, 'tx'|'rx') -> new_value
    field_conf_assign: Dict[Tuple[str, str], float] = {}    # set confidence when adjusted
    field_conf_floor: Dict[Tuple[str, str], float] = {}     # high floors when in strong agreement

    visited_pairs = set()

    # Pairwise consensus-and-hardening with magnitude-aware gating and partial averaging
    for a_id, a_data in telemetry.items():
        b_id = peers.get(a_id)
        if not b_id:
            continue
        key = tuple(sorted([a_id, b_id]))
        if key in visited_pairs:
            continue
        visited_pairs.add(key)

        a_status = a_data.get('interface_status', 'unknown')
        b_status = telemetry[b_id].get('interface_status', 'unknown')

        a_tx = float(a_data.get('tx_rate', 0.0))
        a_rx = float(a_data.get('rx_rate', 0.0))
        b_tx = float(telemetry[b_id].get('tx_rate', 0.0))
        b_rx = float(telemetry[b_id].get('rx_rate', 0.0))

        # Only attempt counter fusion if neither side is explicitly down; down logic handled later
        if a_status != 'down' and b_status != 'down':
            # Direction a->b: compare a_tx with b_rx
            abs_ab = abs(a_tx - b_rx)
            max_ab = max(1.0, a_tx, b_rx)
            diff_ab = abs_ab / max_ab
            tol_ab = LOW_RATE_THRESHOLD if max(a_tx, b_rx) < LOW_RATE_CUTOFF else HARDENING_THRESHOLD

            if diff_ab > tol_ab and abs_ab > ABS_GUARD:
                avg_ab = 0.5 * (a_tx + b_rx)
                if diff_ab <= 2 * tol_ab:
                    # Partial averaging near threshold
                    k = (diff_ab - tol_ab) / max(tol_ab, 1e-9)
                    new_a_tx = a_tx * (1.0 - k) + avg_ab * k
                    new_b_rx = b_rx * (1.0 - k) + avg_ab * k
                else:
                    # Full averaging for clear violations
                    new_a_tx = avg_ab
                    new_b_rx = avg_ab
                field_value_adjust[(a_id, 'tx')] = new_a_tx
                field_value_adjust[(b_id, 'rx')] = new_b_rx
                conf_ab = clamp(1.0 - diff_ab)
                field_conf_assign[(a_id, 'tx')] = conf_ab
                field_conf_assign[(b_id, 'rx')] = conf_ab
            else:
                # Within tolerance: set strong confidence floors
                floor = 0.99 if (max(a_tx, b_rx) >= 10.0 and diff_ab <= 0.005) else 0.98
                field_conf_floor[(a_id, 'tx')] = max(field_conf_floor.get((a_id, 'tx'), 0.0), floor)
                field_conf_floor[(b_id, 'rx')] = max(field_conf_floor.get((b_id, 'rx'), 0.0), floor)

            # Direction b->a: compare b_tx with a_rx
            abs_ba = abs(b_tx - a_rx)
            max_ba = max(1.0, b_tx, a_rx)
            diff_ba = abs_ba / max_ba
            tol_ba = LOW_RATE_THRESHOLD if max(b_tx, a_rx) < LOW_RATE_CUTOFF else HARDENING_THRESHOLD

            if diff_ba > tol_ba and abs_ba > ABS_GUARD:
                avg_ba = 0.5 * (b_tx + a_rx)
                if diff_ba <= 2 * tol_ba:
                    k = (diff_ba - tol_ba) / max(tol_ba, 1e-9)
                    new_b_tx = b_tx * (1.0 - k) + avg_ba * k
                    new_a_rx = a_rx * (1.0 - k) + avg_ba * k
                else:
                    new_b_tx = avg_ba
                    new_a_rx = avg_ba
                field_value_adjust[(b_id, 'tx')] = new_b_tx
                field_value_adjust[(a_id, 'rx')] = new_a_rx
                conf_ba = clamp(1.0 - diff_ba)
                field_conf_assign[(b_id, 'tx')] = conf_ba
                field_conf_assign[(a_id, 'rx')] = conf_ba
            else:
                floor = 0.99 if (max(b_tx, a_rx) >= 10.0 and diff_ba <= 0.005) else 0.98
                field_conf_floor[(b_id, 'tx')] = max(field_conf_floor.get((b_id, 'tx'), 0.0), floor)
                field_conf_floor[(a_id, 'rx')] = max(field_conf_floor.get((a_id, 'rx'), 0.0), floor)

    # Second pass: apply planned adjustments and assign calibrated confidences
    for interface_id, data in telemetry.items():
        repaired = {}

        interface_status = data.get('interface_status', 'unknown')
        rx_rate = float(data.get('rx_rate', 0.0))
        tx_rate = float(data.get('tx_rate', 0.0))
        connected_to = data.get('connected_to')

        # Defaults: identity with conservative base
        repaired_rx = rx_rate
        repaired_tx = tx_rate
        repaired_status = interface_status
        rx_conf = 0.95
        tx_conf = 0.95
        status_conf = 0.95

        # Peer snapshot
        peer_status = None
        if connected_to and connected_to in telemetry:
            peer_status = telemetry[connected_to].get('interface_status', 'unknown')

        # Enforce interface consistency: if either side is down, set both down with zero rates
        if interface_status == 'down' or (peer_status == 'down' if peer_status is not None else False):
            both_down = (interface_status == 'down' and (peer_status == 'down' if peer_status is not None else False))
            repaired_status = 'down'
            repaired_rx = 0.0
            repaired_tx = 0.0
            status_conf = 0.95 if both_down else 0.7
            rx_conf = status_conf
            tx_conf = status_conf
        else:
            # Apply pairwise counter adjustments
            if (interface_id, 'rx') in field_value_adjust:
                repaired_rx = float(field_value_adjust[(interface_id, 'rx')])
                rx_conf = field_conf_assign.get((interface_id, 'rx'), rx_conf)
            if (interface_id, 'tx') in field_value_adjust:
                repaired_tx = float(field_value_adjust[(interface_id, 'tx')])
                tx_conf = field_conf_assign.get((interface_id, 'tx'), tx_conf)

            # Status mismatch (neither side down) reduces status confidence
            if connected_to and connected_to in telemetry:
                if interface_status != peer_status:
                    status_conf = min(status_conf, 0.6)

            # Apply confidence floors for in-tolerance agreements
            rx_floor = field_conf_floor.get((interface_id, 'rx'))
            tx_floor = field_conf_floor.get((interface_id, 'tx'))
            if rx_floor is not None:
                rx_conf = max(rx_conf, rx_floor)
            if tx_floor is not None:
                tx_conf = max(tx_conf, tx_floor)

        # Store
        repaired['rx_rate'] = (rx_rate, repaired_rx, clamp(rx_conf))
        repaired['tx_rate'] = (tx_rate, repaired_tx, clamp(tx_conf))
        repaired['interface_status'] = (interface_status, repaired_status, clamp(status_conf))
        repaired['connected_to'] = connected_to
        repaired['local_router'] = data.get('local_router')
        repaired['remote_router'] = data.get('remote_router')
        result[interface_id] = repaired

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
