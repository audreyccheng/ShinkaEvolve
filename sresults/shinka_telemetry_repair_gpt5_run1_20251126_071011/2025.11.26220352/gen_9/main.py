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
    EPS = 1e-9

    # Helpers
    def norm_status(s: Any) -> str:
        s = str(s).lower()
        return s if s in ("up", "down") else "up"  # treat unknown as up conservatively

    def nz_float(x: Any) -> float:
        try:
            v = float(x)
        except Exception:
            v = 0.0
        return max(0.0, v)

    def rel_diff(a: float, b: float) -> float:
        a = float(a)
        b = float(b)
        denom = max(abs(a), abs(b), 1.0)
        return abs(a - b) / denom

    # Precompute peer mapping
    peers: Dict[str, str] = {iface: data.get('connected_to') for iface, data in telemetry.items()}

    # First pass: link-aware hardening with status-aware zeroing and triage substitution
    pre: Dict[str, Dict[str, Any]] = {}
    for iface, data in telemetry.items():
        local_status = norm_status(data.get('interface_status', 'unknown'))
        rx_orig = nz_float(data.get('rx_rate', 0.0))
        tx_orig = nz_float(data.get('tx_rate', 0.0))

        peer_id = peers.get(iface)
        had_peer = bool(peer_id and peer_id in telemetry)
        peer_status = 'unknown'
        peer_rx = peer_tx = 0.0

        if had_peer:
            pd = telemetry[peer_id]
            peer_status = norm_status(pd.get('interface_status', 'unknown'))
            peer_rx = nz_float(pd.get('rx_rate', 0.0))
            peer_tx = nz_float(pd.get('tx_rate', 0.0))

        pre_rx = rx_orig
        pre_tx = tx_orig
        rx_link_diff = 0.0
        tx_link_diff = 0.0

        # Status-aware zeroing: local down => zero traffic
        if local_status == 'down':
            pre_rx = 0.0
            pre_tx = 0.0
        else:
            if had_peer and peer_status == 'up':
                # Link symmetry: A.rx ≈ B.tx, A.tx ≈ B.rx
                rx_link_diff = rel_diff(rx_orig, peer_tx)
                tx_link_diff = rel_diff(tx_orig, peer_rx)

                pair_rx_ok = rx_link_diff <= HARDENING_THRESHOLD
                pair_tx_ok = tx_link_diff <= HARDENING_THRESHOLD

                # Triage: if one pair matches and the other violates, do direct substitution for the violating side
                if pair_tx_ok and not pair_rx_ok:
                    pre_rx = peer_tx
                elif pair_rx_ok and not pair_tx_ok:
                    pre_tx = peer_rx
                elif not pair_rx_ok and not pair_tx_ok:
                    # Multiple faults or timing skew: average redundant signals to harden
                    pre_rx = 0.5 * (rx_orig + peer_tx)
                    pre_tx = 0.5 * (tx_orig + peer_rx)
                # else: both ok -> keep as is
            else:
                # No reliable peer signal (no peer or peer down): keep local measurements
                rx_link_diff = 0.0
                tx_link_diff = 0.0

        pre[iface] = {
            'pre_rx': pre_rx,
            'pre_tx': pre_tx,
            'rx_link_diff': rx_link_diff,
            'tx_link_diff': tx_link_diff,
            'local_status': local_status,
            'peer_status': peer_status if had_peer else 'unknown',
            'had_peer': had_peer
        }

    # Second pass: router-level flow conservation using topology (R1)
    # Scale only the larger aggregate (RX or TX) to match the smaller when imbalance exceeds tolerance.
    scale_rx: Dict[str, float] = {iface: 1.0 for iface in telemetry}
    scale_tx: Dict[str, float] = {iface: 1.0 for iface in telemetry}
    router_imbalance: Dict[str, float] = {}

    for router_id, iface_list in topology.items():
        # Consider only interfaces present in telemetry
        candidate_ifaces = [i for i in iface_list if i in telemetry]
        if not candidate_ifaces:
            router_imbalance[router_id] = 0.0
            continue

        # Active interfaces (local up). Down links already zeroed.
        up_ifaces = [i for i in candidate_ifaces if pre[i]['local_status'] == 'up']
        if not up_ifaces:
            router_imbalance[router_id] = 0.0
            continue

        sum_rx = sum(pre[i]['pre_rx'] for i in up_ifaces)
        sum_tx = sum(pre[i]['pre_tx'] for i in up_ifaces)

        denom = max(sum_rx, sum_tx, 1.0)
        imbalance = abs(sum_tx - sum_rx) / denom
        router_imbalance[router_id] = imbalance

        # Avoid scaling for tiny volumes or single-link routers
        if (sum_rx + sum_tx) <= 1e-6 or len(up_ifaces) < 2:
            continue

        if imbalance > HARDENING_THRESHOLD:
            applied_targeted = False
            if sum_tx > sum_rx:
                # Prefer to correct the most suspicious TX interface based on link symmetry residual
                suspects = [i for i in up_ifaces if pre[i]['pre_tx'] > 0.0]
                if suspects:
                    sus = max(suspects, key=lambda i: pre[i]['tx_link_diff'])
                    tx_sus = pre[sus]['pre_tx']
                    # Choose factor to make router TX equal RX after adjusting only the suspect
                    k_sus = (sum_rx - (sum_tx - tx_sus)) / max(tx_sus, EPS)
                    if 0.0 < k_sus < 1.0:
                        scale_tx[sus] = k_sus
                        applied_targeted = True
                if not applied_targeted:
                    # Fallback to uniform scaling of TX if a single-target fix is infeasible
                    k = sum_rx / max(sum_tx, EPS)
                    for i in up_ifaces:
                        scale_tx[i] = k
            else:
                # sum_rx > sum_tx: correct the most suspicious RX interface
                suspects = [i for i in up_ifaces if pre[i]['pre_rx'] > 0.0]
                if suspects:
                    sus = max(suspects, key=lambda i: pre[i]['rx_link_diff'])
                    rx_sus = pre[sus]['pre_rx']
                    k_sus = (sum_tx - (sum_rx - rx_sus)) / max(rx_sus, EPS)
                    if 0.0 < k_sus < 1.0:
                        scale_rx[sus] = k_sus
                        applied_targeted = True
                if not applied_targeted:
                    # Fallback to uniform scaling of RX
                    k = sum_tx / max(sum_rx, EPS)
                    for i in up_ifaces:
                        scale_rx[i] = k

        # Recompute post-scaling imbalance for this router for confidence calibration
        new_sum_rx = sum(pre[i]['pre_rx'] * scale_rx.get(i, 1.0) for i in up_ifaces)
        new_sum_tx = sum(pre[i]['pre_tx'] * scale_tx.get(i, 1.0) for i in up_ifaces)
        new_denom = max(new_sum_rx, new_sum_tx, 1.0)
        router_imbalance[router_id] = abs(new_sum_tx - new_sum_rx) / new_denom

    # Assemble final results with calibrated confidence
    result: Dict[str, Dict[str, Tuple]] = {}
    for iface, data in telemetry.items():
        rx_orig = nz_float(data.get('rx_rate', 0.0))
        tx_orig = nz_float(data.get('tx_rate', 0.0))
        local_status = pre[iface]['local_status']
        peer_status = pre[iface]['peer_status']
        had_peer = pre[iface]['had_peer']

        # Apply router scaling
        rx_repaired = pre[iface]['pre_rx'] * scale_rx.get(iface, 1.0)
        tx_repaired = pre[iface]['pre_tx'] * scale_tx.get(iface, 1.0)

        # Down interfaces must have zero traffic
        repaired_status = data.get('interface_status', 'unknown')
        if norm_status(repaired_status) == 'down':
            rx_repaired = 0.0
            tx_repaired = 0.0

        # Compute link residuals after scaling using peer's scaled counters
        peer_id = peers.get(iface)
        if had_peer and peer_id in pre and local_status == 'up' and (peer_status == 'up'):
            peer_tx_after = pre[peer_id]['pre_tx'] * scale_tx.get(peer_id, 1.0)
            peer_rx_after = pre[peer_id]['pre_rx'] * scale_rx.get(peer_id, 1.0)
            rx_resid = rel_diff(rx_repaired, peer_tx_after)
            tx_resid = rel_diff(tx_repaired, peer_rx_after)
            rx_link_conf = max(0.0, 1.0 - rx_resid)
            tx_link_conf = max(0.0, 1.0 - tx_resid)
        elif norm_status(repaired_status) == 'down':
            rx_link_conf = 0.85 if rx_repaired == 0.0 else 0.5
            tx_link_conf = 0.85 if tx_repaired == 0.0 else 0.5
        else:
            # No reliable peer information
            rx_link_conf = 0.6
            tx_link_conf = 0.6

        # Router factor from imbalance
        router_id = data.get('local_router')
        imbalance = router_imbalance.get(router_id, 0.0)
        router_factor = max(0.2, 1.0 - imbalance)

        # Change penalty to avoid overconfidence on large corrections
        rx_change = rel_diff(rx_orig, rx_repaired)
        tx_change = rel_diff(tx_orig, tx_repaired)
        rx_change_factor = max(0.2, 1.0 - 0.5 * min(1.0, rx_change))
        tx_change_factor = max(0.2, 1.0 - 0.5 * min(1.0, tx_change))

        rx_confidence = max(0.0, min(1.0, rx_link_conf * router_factor * rx_change_factor))
        tx_confidence = max(0.0, min(1.0, tx_link_conf * router_factor * tx_change_factor))

        # Status handling: keep status unchanged but calibrate confidence
        status_confidence = 1.0
        # If peer status differs, reduce confidence
        if peer_id and peer_id in telemetry:
            peer_status_raw = norm_status(telemetry[peer_id].get('interface_status', 'unknown'))
            if norm_status(repaired_status) != peer_status_raw:
                status_confidence = min(status_confidence, 0.5)
        # If interface reports down but had non-zero original counters, lower status confidence
        if norm_status(repaired_status) == 'down' and (rx_orig > 0.0 or tx_orig > 0.0):
            status_confidence = min(status_confidence, 0.6)

        # Build output entry
        entry: Dict[str, Tuple] = {}
        entry['rx_rate'] = (rx_orig, rx_repaired, rx_confidence)
        entry['tx_rate'] = (tx_orig, tx_repaired, tx_confidence)
        entry['interface_status'] = (data.get('interface_status', 'unknown'), repaired_status, status_confidence)

        # Copy metadata unchanged
        entry['connected_to'] = data.get('connected_to')
        entry['local_router'] = data.get('local_router')
        entry['remote_router'] = data.get('remote_router')

        result[iface] = entry

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
