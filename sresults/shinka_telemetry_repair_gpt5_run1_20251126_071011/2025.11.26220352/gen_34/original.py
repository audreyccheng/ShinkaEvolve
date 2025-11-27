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
            p = telemetry[peer_id]
            peer_status = norm_status(p.get('interface_status', 'unknown'))
            peer_rx = nz_float(p.get('rx_rate', 0.0))
            peer_tx = nz_float(p.get('tx_rate', 0.0))

        pre_rx = rx_orig
        pre_tx = tx_orig
        rx_link_diff = 0.0
        tx_link_diff = 0.0

        # Status-aware zeroing: only zero when local interface is down
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

                # Triage: if only one direction violates, substitute; if both violate, average
                if pair_tx_ok and not pair_rx_ok:
                    pre_rx = peer_tx
                elif pair_rx_ok and not pair_tx_ok:
                    pre_tx = peer_rx
                elif not pair_rx_ok and not pair_tx_ok:
                    pre_rx = 0.5 * (rx_orig + peer_tx)
                    pre_tx = 0.5 * (tx_orig + peer_rx)
                # else: both within tolerance -> keep as-is

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
    # Scale only the larger aggregate to match the smaller when imbalance exceeds tolerance.
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
            if sum_tx > sum_rx:
                k = sum_rx / max(sum_tx, EPS)
                for i in up_ifaces:
                    scale_tx[i] = k
            else:
                k = sum_tx / max(sum_rx, EPS)
                for i in up_ifaces:
                    scale_rx[i] = k

    # Third pass: pairwise reconciliation to enforce link symmetry post router scaling
    post: Dict[str, Dict[str, float]] = {}
    for iface in telemetry:
        post[iface] = {
            'rx': pre[iface]['pre_rx'] * scale_rx.get(iface, 1.0),
            'tx': pre[iface]['pre_tx'] * scale_tx.get(iface, 1.0),
        }

    visited_pairs = set()
    for iface, data in telemetry.items():
        peer_id = peers.get(iface)
        if not (peer_id and peer_id in telemetry):
            continue
        pair_key = tuple(sorted([iface, peer_id]))
        if pair_key in visited_pairs:
            continue
        visited_pairs.add(pair_key)

        # Only reconcile when both interfaces are locally up
        if pre[iface]['local_status'] != 'up' or pre.get(peer_id, {}).get('local_status') != 'up':
            continue

        # Residuals after router scaling
        resid_tx = rel_diff(post[iface]['tx'], post[peer_id]['rx'])  # A.tx vs B.rx
        resid_rx = rel_diff(post[iface]['rx'], post[peer_id]['tx'])  # A.rx vs B.tx

        def suspicion(ifc: str, kind: str) -> float:
            rid = telemetry.get(ifc, {}).get('local_router')
            imb = router_imbalance.get(rid, 0.0)
            if kind == 'tx':
                return pre[ifc]['tx_link_diff'] + 0.5 * abs(scale_tx.get(ifc, 1.0) - 1.0) + 0.3 * imb
            else:
                return pre[ifc]['rx_link_diff'] + 0.5 * abs(scale_rx.get(ifc, 1.0) - 1.0) + 0.3 * imb

        # Reconcile A.tx <-> B.rx
        if resid_tx > HARDENING_THRESHOLD:
            a_tx_s = suspicion(iface, 'tx')
            b_rx_s = suspicion(peer_id, 'rx')
            if a_tx_s >= b_rx_s:
                post[iface]['tx'] = post[peer_id]['rx']
            else:
                post[peer_id]['rx'] = post[iface]['tx']

        # Reconcile A.rx <-> B.tx
        if resid_rx > HARDENING_THRESHOLD:
            a_rx_s = suspicion(iface, 'rx')
            b_tx_s = suspicion(peer_id, 'tx')
            if a_rx_s >= b_tx_s:
                post[iface]['rx'] = post[peer_id]['tx']
            else:
                post[peer_id]['tx'] = post[iface]['rx']

    # Assemble final results with calibrated confidence (post-repair residuals)
    result: Dict[str, Dict[str, Tuple]] = {}

    for iface, data in telemetry.items():
        local_status = pre[iface]['local_status']
        peer_status = pre[iface]['peer_status']
        had_peer = pre[iface]['had_peer']
        rx_orig = nz_float(data.get('rx_rate', 0.0))
        tx_orig = nz_float(data.get('tx_rate', 0.0))

        rx_repaired = post[iface]['rx']
        tx_repaired = post[iface]['tx']

        # Enforce zero on down interfaces
        repaired_status = data.get('interface_status', 'unknown')
        if norm_status(repaired_status) == 'down':
            rx_repaired = 0.0
            tx_repaired = 0.0

        # Confidence based on post-repair residuals vs peer
        peer_id = peers.get(iface)
        if had_peer and peer_id in pre and local_status == 'up' and (peer_status == 'up'):
            peer_tx_after = post[peer_id]['tx']
            peer_rx_after = post[peer_id]['rx']
            rx_resid = rel_diff(rx_repaired, peer_tx_after)
            tx_resid = rel_diff(tx_repaired, peer_rx_after)
            rx_link_conf = max(0.0, 1.0 - rx_resid)
            tx_link_conf = max(0.0, 1.0 - tx_resid)
        elif norm_status(repaired_status) == 'down':
            rx_link_conf = 0.9 if rx_repaired == 0.0 else 0.5
            tx_link_conf = 0.9 if tx_repaired == 0.0 else 0.5
        else:
            rx_link_conf = 0.6
            tx_link_conf = 0.6

        # Router imbalance factor
        router_id = data.get('local_router')
        imbalance = router_imbalance.get(router_id, 0.0)
        router_factor = max(0.2, 1.0 - imbalance)

        # Change penalty: reduce confidence for large corrections from original
        rx_change = rel_diff(rx_orig, rx_repaired)
        tx_change = rel_diff(tx_orig, tx_repaired)
        rx_change_factor = max(0.2, 1.0 - 0.5 * min(1.0, rx_change))
        tx_change_factor = max(0.2, 1.0 - 0.5 * min(1.0, tx_change))

        rx_confidence = max(0.0, min(1.0, rx_link_conf * router_factor * rx_change_factor))
        tx_confidence = max(0.0, min(1.0, tx_link_conf * router_factor * tx_change_factor))

        # Status confidence adjustments (keep status unchanged; calibrate confidence)
        status_confidence = 1.0

        if peer_id and peer_id in telemetry:
            peer_status_raw = norm_status(telemetry[peer_id].get('interface_status', 'unknown'))
            if norm_status(repaired_status) != peer_status_raw:
                status_confidence = min(status_confidence, 0.5)

        if norm_status(repaired_status) == 'down' and (rx_orig > 0.0 or tx_orig > 0.0):
            status_confidence = min(status_confidence, 0.6)

        # Build output
        repaired_entry: Dict[str, Tuple] = {}
        repaired_entry['rx_rate'] = (rx_orig, rx_repaired, rx_confidence)
        repaired_entry['tx_rate'] = (tx_orig, tx_repaired, tx_confidence)
        repaired_entry['interface_status'] = (data.get('interface_status', 'unknown'), repaired_status, status_confidence)

        # Copy metadata unchanged
        repaired_entry['connected_to'] = data.get('connected_to')
        repaired_entry['local_router'] = data.get('local_router')
        repaired_entry['remote_router'] = data.get('remote_router')

        result[iface] = repaired_entry

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