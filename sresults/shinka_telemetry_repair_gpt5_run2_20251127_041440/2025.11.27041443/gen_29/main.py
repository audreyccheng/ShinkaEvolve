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

    # Measurement timing tolerance (from Hodor research: ~2%)
    HARDENING_THRESHOLD = 0.02
    # Magnitude-aware tolerance and absolute guard to avoid over-correcting tiny flows
    LOW_RATE_CUTOFF = 10.0  # Mbps
    ABS_GUARD = 0.5         # Mbps; require this absolute delta to trigger a repair

    def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
        return max(lo, min(hi, x))

    result: Dict[str, Dict[str, Tuple]] = {}

    # First pass: collect all measurements and compute link symmetry stats
    link_stats: Dict[str, Dict[str, float]] = {}
    for interface_id, data in telemetry.items():
        rx_rate = float(data.get('rx_rate', 0.0))
        tx_rate = float(data.get('tx_rate', 0.0))
        connected_to = data.get('connected_to')

        if connected_to and connected_to in telemetry:
            peer = telemetry[connected_to]
            peer_rx = float(peer.get('rx_rate', 0.0))
            peer_tx = float(peer.get('tx_rate', 0.0))

            # My TX should match their RX
            abs_tx_rx = abs(tx_rate - peer_rx)
            max_tx_rx = max(1.0, tx_rate, peer_rx)
            rel_tx_rx = abs_tx_rx / max_tx_rx

            # My RX should match their TX
            abs_rx_tx = abs(rx_rate - peer_tx)
            max_rx_tx = max(1.0, rx_rate, peer_tx)
            rel_rx_tx = abs_rx_tx / max_rx_tx

            # Dynamic thresholds
            th_tx_rx = 0.05 if max(tx_rate, peer_rx) < LOW_RATE_CUTOFF else HARDENING_THRESHOLD
            th_rx_tx = 0.05 if max(rx_rate, peer_tx) < LOW_RATE_CUTOFF else HARDENING_THRESHOLD

            link_stats[interface_id] = {
                'peer_rx': peer_rx,
                'peer_tx': peer_tx,
                'abs_tx_rx': abs_tx_rx,
                'abs_rx_tx': abs_rx_tx,
                'rel_tx_rx': rel_tx_rx,
                'rel_rx_tx': rel_rx_tx,
                'th_tx_rx': th_tx_rx,
                'th_rx_tx': th_rx_tx,
                'max_tx_rx': max(tx_rate, peer_rx),
                'max_rx_tx': max(rx_rate, peer_tx),
            }

    # Second pass: repair using redundancy and assign calibrated confidences
    for interface_id, data in telemetry.items():
        repaired_data: Dict[str, Any] = {}

        interface_status = data.get('interface_status', 'unknown')
        rx_rate = float(data.get('rx_rate', 0.0))
        tx_rate = float(data.get('tx_rate', 0.0))
        connected_to = data.get('connected_to')

        # Start with conservative defaults; will raise when evidence is strong
        repaired_rx = rx_rate
        repaired_tx = tx_rate
        repaired_status = interface_status
        rx_confidence = 0.95
        tx_confidence = 0.95
        status_confidence = 0.95

        # Peer snapshot for redundancy use
        peer_status = None
        peer_rx = None
        peer_tx = None
        if connected_to and connected_to in telemetry:
            peer = telemetry[connected_to]
            peer_status = peer.get('interface_status', 'unknown')
            peer_rx = float(peer.get('rx_rate', 0.0))
            peer_tx = float(peer.get('tx_rate', 0.0))

        # Enforce interface consistency: if either side is down, set effective down and zero rates
        if interface_status == 'down' or (peer_status == 'down' if peer_status is not None else False):
            both_down = (interface_status == 'down' and (peer_status == 'down' if peer_status is not None else False))
            repaired_status = 'down'
            repaired_rx = 0.0
            repaired_tx = 0.0
            status_confidence = 0.95 if both_down else 0.7
            rx_confidence = status_confidence
            tx_confidence = status_confidence
        else:
            # Use magnitude-aware symmetry check with absolute guard for value repairs
            if interface_id in link_stats:
                stats = link_stats[interface_id]

                # RX should match peer's TX
                abs_diff_rx = stats['abs_rx_tx']
                rel_diff_rx = stats['rel_rx_tx']
                tol_rx = stats['th_rx_tx']
                max_pair_rx = stats['max_rx_tx']

                if rel_diff_rx > tol_rx and abs_diff_rx > ABS_GUARD:
                    repaired_rx = stats['peer_tx']
                    rx_confidence = clamp(1.0 - rel_diff_rx)
                else:
                    # Within tolerance: strong agreement floors
                    if max_pair_rx >= 10.0 and rel_diff_rx <= 0.005:
                        rx_confidence = max(rx_confidence, 0.99)
                    else:
                        rx_confidence = max(rx_confidence, 0.97 if max_pair_rx < 10.0 else 0.98)

                # TX should match peer's RX
                abs_diff_tx = stats['abs_tx_rx']
                rel_diff_tx = stats['rel_tx_rx']
                tol_tx = stats['th_tx_rx']
                max_pair_tx = stats['max_tx_rx']

                if rel_diff_tx > tol_tx and abs_diff_tx > ABS_GUARD:
                    repaired_tx = stats['peer_rx']
                    tx_confidence = clamp(1.0 - rel_diff_tx)
                else:
                    if max_pair_tx >= 10.0 and rel_diff_tx <= 0.005:
                        tx_confidence = max(tx_confidence, 0.99)
                    else:
                        tx_confidence = max(tx_confidence, 0.97 if max_pair_tx < 10.0 else 0.98)
            else:
                # No redundancy: keep values but slightly lower confidence
                rx_confidence = max(rx_confidence, 0.9)
                tx_confidence = max(tx_confidence, 0.9)

            # If statuses differ (but neither side is down), reduce status confidence
            if peer_status is not None and interface_status != peer_status:
                status_confidence = min(status_confidence, 0.6)

        # Store repaired values with confidence scores
        repaired_data['rx_rate'] = (float(rx_rate), float(repaired_rx), clamp(rx_confidence))
        repaired_data['tx_rate'] = (float(tx_rate), float(repaired_tx), clamp(tx_confidence))
        repaired_data['interface_status'] = (interface_status, repaired_status, clamp(status_confidence))

        # Copy metadata unchanged
        repaired_data['connected_to'] = connected_to
        repaired_data['local_router'] = data.get('local_router')
        repaired_data['remote_router'] = data.get('remote_router')

        result[interface_id] = repaired_data

    # Router-level flow conservation: direction-aware confidence attenuation
    # Build router->interfaces mapping (prefer topology, augment with telemetry hints)
    router_ifaces: Dict[str, List[str]] = {r: list(if_list) for r, if_list in topology.items()}
    for if_id, d in telemetry.items():
        lr = d.get('local_router')
        if lr:
            router_ifaces.setdefault(lr, [])
            if if_id not in router_ifaces[lr]:
                router_ifaces[lr].append(if_id)
        rr = d.get('remote_router')
        if rr and rr not in router_ifaces:
            router_ifaces[rr] = []

    # Compute per-router residual mismatch from repaired values
    router_resid: Dict[str, float] = {}
    for r, if_list in router_ifaces.items():
        sum_tx = 0.0
        sum_rx = 0.0
        for if_id in if_list:
            if if_id in result:
                sum_tx += float(result[if_id]['tx_rate'][1])
                sum_rx += float(result[if_id]['rx_rate'][1])
        denom = max(1.0, sum_tx, sum_rx)
        router_resid[r] = abs(sum_tx - sum_rx) / denom

    # Apply direction-aware penalties with a gentle floor to avoid over-penalizing noisy sites
    for if_id, d in telemetry.items():
        lr = d.get('local_router')
        rr = d.get('remote_router')
        resid_local = router_resid.get(lr, 0.0)
        resid_remote = router_resid.get(rr, 0.0)
        # TX leaves local, RX arrives from remote
        penalty_tx = clamp(1.0 - (0.6 * resid_local + 0.4 * resid_remote), 0.7, 1.0)
        penalty_rx = clamp(1.0 - (0.6 * resid_remote + 0.4 * resid_local), 0.7, 1.0)
        orx, rrx, rc = result[if_id]['rx_rate']
        otx, rtx, tc = result[if_id]['tx_rate']
        result[if_id]['rx_rate'] = (orx, rrx, clamp(rc * penalty_rx))
        result[if_id]['tx_rate'] = (otx, rtx, clamp(tc * penalty_tx))

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
