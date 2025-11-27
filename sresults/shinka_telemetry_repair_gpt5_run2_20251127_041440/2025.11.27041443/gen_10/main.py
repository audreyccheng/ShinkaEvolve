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
    EPS = 1e-6  # small absolute tolerance for "no traffic"

    def norm_diff(a: float, b: float) -> float:
        return abs(a - b) / max(a, b, 1.0)

    def has_traffic(d: Dict[str, Any]) -> bool:
        return (float(d.get('rx_rate', 0.0)) > EPS) or (float(d.get('tx_rate', 0.0)) > EPS)

    def change_ratio(orig: float, rep: float) -> float:
        denom = max(abs(orig), abs(rep), 1.0)
        return abs(rep - orig) / denom

    # Build link pairs (unique, undirected)
    pairs = {}  # key: tuple(sorted(if1, if2)) -> (if1_id, if2_id)
    for if_id, data in telemetry.items():
        peer = data.get('connected_to')
        if peer and peer in telemetry:
            key = tuple(sorted([if_id, peer]))
            if key not in pairs:
                pairs[key] = (if_id, peer)

    # First pass: pairwise hardening and status harmonization using link symmetry
    per_if_repair = {}  # interface_id -> interim repaired values and base confidences

    for _, (a_id, b_id) in pairs.items():
        a = telemetry[a_id]
        b = telemetry[b_id]

        a_status = a.get('interface_status', 'unknown')
        b_status = b.get('interface_status', 'unknown')

        a_rx, a_tx = float(a.get('rx_rate', 0.0) or 0.0), float(a.get('tx_rate', 0.0) or 0.0)
        b_rx, b_tx = float(b.get('rx_rate', 0.0) or 0.0), float(b.get('tx_rate', 0.0) or 0.0)

        # Decide pair status using interface consistency + traffic evidence
        a_has = has_traffic(a)
        b_has = has_traffic(b)

        if a_status == 'down' and b_status == 'down':
            pair_status = 'down'
        elif a_status == 'up' and b_status == 'up':
            pair_status = 'up'
        else:
            # Mismatch: if any traffic exists on either end, consider link up; otherwise down
            pair_status = 'up' if (a_has or b_has) else 'down'

        # Initialize with originals
        rep_a_tx, rep_b_rx = a_tx, b_rx
        rep_b_tx, rep_a_rx = b_tx, a_rx
        rx_conf_a = 1.0
        tx_conf_a = 1.0
        rx_conf_b = 1.0
        tx_conf_b = 1.0

        if pair_status == 'down':
            # No traffic on a down link
            rep_a_tx, rep_b_rx, rep_b_tx, rep_a_rx = 0.0, 0.0, 0.0, 0.0
            # Confidence is high if there was no traffic observed; otherwise moderate
            base_conf = 0.95 if not (a_has or b_has) else 0.7
            rx_conf_a = tx_conf_a = rx_conf_b = tx_conf_b = base_conf
        else:
            # Link is up: enforce link symmetry with hardening threshold
            diff_ab = norm_diff(a_tx, b_rx)  # A.tx vs B.rx
            diff_ba = norm_diff(b_tx, a_rx)  # B.tx vs A.rx

            if diff_ab > HARDENING_THRESHOLD:
                avg_ab = 0.5 * (a_tx + b_rx)
                rep_a_tx = avg_ab
                rep_b_rx = avg_ab
            conf_ab = max(0.0, 1.0 - diff_ab)
            if diff_ab <= HARDENING_THRESHOLD:
                conf_ab = min(1.0, max(conf_ab, 0.98))
            tx_conf_a = conf_ab
            rx_conf_b = conf_ab

            if diff_ba > HARDENING_THRESHOLD:
                avg_ba = 0.5 * (b_tx + a_rx)
                rep_b_tx = avg_ba
                rep_a_rx = avg_ba
            conf_ba = max(0.0, 1.0 - diff_ba)
            if diff_ba <= HARDENING_THRESHOLD:
                conf_ba = min(1.0, max(conf_ba, 0.98))
            tx_conf_b = conf_ba
            rx_conf_a = conf_ba

        # Status confidence based on agreement and evidence
        if pair_status == 'down':
            if a_status == 'down' and b_status == 'down' and not (a_has or b_has):
                status_conf = 0.98
            else:
                status_conf = 0.7
        else:  # up
            if a_status == 'up' and b_status == 'up':
                status_conf = 0.95
            else:
                # we decided up due to traffic evidence
                status_conf = 0.8

        # Confidence refinement: penalize by magnitude of applied change
        # This improves calibration by lowering confidence when we make large adjustments.
        a_rx_change = change_ratio(a_rx, rep_a_rx)
        a_tx_change = change_ratio(a_tx, rep_a_tx)
        b_rx_change = change_ratio(b_rx, rep_b_rx)
        b_tx_change = change_ratio(b_tx, rep_b_tx)

        rx_conf_a = max(0.0, min(1.0, min(rx_conf_a, 1.0 - a_rx_change)))
        tx_conf_a = max(0.0, min(1.0, min(tx_conf_a, 1.0 - a_tx_change)))
        rx_conf_b = max(0.0, min(1.0, min(rx_conf_b, 1.0 - b_rx_change)))
        tx_conf_b = max(0.0, min(1.0, min(tx_conf_b, 1.0 - b_tx_change)))

        per_if_repair[a_id] = {
            'repaired_rx': rep_a_rx,
            'repaired_tx': rep_a_tx,
            'rx_conf': rx_conf_a,
            'tx_conf': tx_conf_a,
            'repaired_status': pair_status,
            'status_conf': status_conf
        }
        per_if_repair[b_id] = {
            'repaired_rx': rep_b_rx,
            'repaired_tx': rep_b_tx,
            'rx_conf': rx_conf_b,
            'tx_conf': tx_conf_b,
            'repaired_status': pair_status,
            'status_conf': status_conf
        }

    # Handle interfaces without a valid peer (dangling or missing peer data)
    for if_id, data in telemetry.items():
        if if_id in per_if_repair:
            continue
        status = data.get('interface_status', 'unknown')
        rx = float(data.get('rx_rate', 0.0) or 0.0)
        tx = float(data.get('tx_rate', 0.0) or 0.0)
        if status == 'down':
            # Enforce no traffic on down interfaces
            per_if_repair[if_id] = {
                'repaired_rx': 0.0,
                'repaired_tx': 0.0,
                'rx_conf': 0.9,
                'tx_conf': 0.9,
                'repaired_status': 'down',
                'status_conf': 0.95
            }
        else:
            # No peer to cross-check; keep values but lower confidence slightly.
            per_if_repair[if_id] = {
                'repaired_rx': rx,
                'repaired_tx': tx,
                'rx_conf': 0.6,
                'tx_conf': 0.6,
                'repaired_status': status if status in ('up', 'down') else 'up',
                'status_conf': 0.6
            }

    # Router-level flow conservation repair (small balancing step preserving link equality)
    # Build per-direction flows from pairwise-hardened values and adjust to reduce router imbalance.
    # This uses local_router metadata; if missing for some interfaces, those flows are skipped.
    flows = []  # list of dicts: {'tx_if': str, 'rx_if': str, 'tx_r': router, 'rx_r': router, 'val': float}
    # Map of router -> indices of flows where router is tx or rx endpoint
    flows_tx_by_router: Dict[Any, List[int]] = {}
    flows_rx_by_router: Dict[Any, List[int]] = {}

    for _, (a_id, b_id) in pairs.items():
        # Skip if pair was determined down
        if a_id not in per_if_repair or b_id not in per_if_repair:
            continue
        if per_if_repair[a_id].get('repaired_status') == 'down':
            continue

        a_lr = telemetry[a_id].get('local_router')
        b_lr = telemetry[b_id].get('local_router')

        # Direction A -> B
        val_ab = float(per_if_repair[a_id]['repaired_tx'])
        idx_ab = len(flows)
        flows.append({'tx_if': a_id, 'rx_if': b_id, 'tx_r': a_lr, 'rx_r': b_lr, 'val': val_ab})
        flows_tx_by_router.setdefault(a_lr, []).append(idx_ab)
        flows_rx_by_router.setdefault(b_lr, []).append(idx_ab)

        # Direction B -> A
        val_ba = float(per_if_repair[b_id]['repaired_tx'])
        idx_ba = len(flows)
        flows.append({'tx_if': b_id, 'rx_if': a_id, 'tx_r': b_lr, 'rx_r': a_lr, 'val': val_ba})
        flows_tx_by_router.setdefault(b_lr, []).append(idx_ba)
        flows_rx_by_router.setdefault(a_lr, []).append(idx_ba)

    # Compute per-router imbalance and balancing steps
    s_by_router: Dict[Any, float] = {}
    for r in set(list(flows_tx_by_router.keys()) + list(flows_rx_by_router.keys())):
        tx_indices = flows_tx_by_router.get(r, [])
        rx_indices = flows_rx_by_router.get(r, [])
        sum_tx_r = sum(flows[i]['val'] for i in tx_indices)
        sum_rx_r = sum(flows[i]['val'] for i in rx_indices)
        delta_r = sum_tx_r - sum_rx_r
        resid_r = abs(delta_r) / max(sum_tx_r, sum_rx_r, 1.0)
        denom = len(tx_indices) + len(rx_indices)
        if denom == 0 or resid_r <= HARDENING_THRESHOLD:
            s_by_router[r] = 0.0
        else:
            s_by_router[r] = delta_r / float(denom)

    # Apply balancing: v' = v - s_tx_router + s_rx_router, clamp non-negative
    if s_by_router:
        for i, f in enumerate(flows):
            s_tx = s_by_router.get(f['tx_r'], 0.0)
            s_rx = s_by_router.get(f['rx_r'], 0.0)
            new_val = f['val'] - s_tx + s_rx
            if new_val < 0.0:
                new_val = 0.0
            flows[i]['val'] = new_val

        # Push adjusted flow values back to per-interface repaired counters
        for f in flows:
            tx_if = f['tx_if']
            rx_if = f['rx_if']
            v = f['val']
            if tx_if in per_if_repair:
                per_if_repair[tx_if]['repaired_tx'] = v
            if rx_if in per_if_repair:
                per_if_repair[rx_if]['repaired_rx'] = v

        # Update confidences based on total change from the original observation
        for if_id, rep in per_if_repair.items():
            rx_orig = float(telemetry.get(if_id, {}).get('rx_rate', 0.0) or 0.0)
            tx_orig = float(telemetry.get(if_id, {}).get('tx_rate', 0.0) or 0.0)
            rx_change = change_ratio(rx_orig, float(rep['repaired_rx']))
            tx_change = change_ratio(tx_orig, float(rep['repaired_tx']))
            rep['rx_conf'] = max(0.0, min(1.0, min(rep.get('rx_conf', 0.6), 1.0 - rx_change)))
            rep['tx_conf'] = max(0.0, min(1.0, min(rep.get('tx_conf', 0.6), 1.0 - tx_change)))

    # Second pass: compute router-level flow conservation residuals using repaired rates
    router_residual: Dict[str, float] = {}
    for router, if_list in topology.items():
        sum_tx = 0.0
        sum_rx = 0.0
        for if_id in if_list:
            if if_id in per_if_repair:
                rep = per_if_repair[if_id]
                # Include all interfaces; down interfaces contribute 0 traffic (already enforced)
                sum_tx += float(rep['repaired_tx'])
                sum_rx += float(rep['repaired_rx'])
        resid = abs(sum_tx - sum_rx) / max(sum_tx, sum_rx, 1.0)
        router_residual[router] = resid

    # Final assembly: scale confidences by router residuals (dynamic checking)
    result: Dict[str, Dict[str, Tuple]] = {}
    for if_id, data in telemetry.items():
        repaired = per_if_repair.get(if_id, {})
        repaired_rx = repaired.get('repaired_rx', float(data.get('rx_rate', 0.0) or 0.0))
        repaired_tx = repaired.get('repaired_tx', float(data.get('tx_rate', 0.0) or 0.0))
        repaired_status = repaired.get('repaired_status', data.get('interface_status', 'unknown'))

        # Base confidences from link hardening
        rx_conf = float(repaired.get('rx_conf', 0.6))
        tx_conf = float(repaired.get('tx_conf', 0.6))
        status_conf = float(repaired.get('status_conf', 0.6))

        # Apply router-based penalty using both local and remote routers when available
        local_router = data.get('local_router')
        remote_router = data.get('remote_router')

        resid_local = router_residual.get(local_router, 0.0)
        resid_remote = router_residual.get(remote_router, 0.0)

        penalty = 1.0 - 0.5 * (resid_local + resid_remote)
        penalty = max(0.0, min(1.0, penalty))

        rx_conf = max(0.0, min(1.0, rx_conf * penalty))
        tx_conf = max(0.0, min(1.0, tx_conf * penalty))
        # Status confidence is less sensitive to flow residuals; apply mild scaling
        status_conf = max(0.0, min(1.0, status_conf * (0.75 + 0.25 * penalty)))

        repaired_data: Dict[str, Any] = {}
        # Store repaired values with confidence scores
        rx_orig = float(data.get('rx_rate', 0.0) or 0.0)
        tx_orig = float(data.get('tx_rate', 0.0) or 0.0)
        status_orig = data.get('interface_status', 'unknown')

        repaired_data['rx_rate'] = (rx_orig, repaired_rx, rx_conf)
        repaired_data['tx_rate'] = (tx_orig, repaired_tx, tx_conf)
        repaired_data['interface_status'] = (status_orig, repaired_status, status_conf)

        # Copy metadata unchanged
        repaired_data['connected_to'] = data.get('connected_to')
        repaired_data['local_router'] = local_router
        repaired_data['remote_router'] = remote_router

        result[if_id] = repaired_data

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