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

    Strategy inspired by Hodor:
    1) Signal Collection: use redundant bilateral measurements on links.
    2) Signal Hardening: pair-wise hardening via link symmetry (R3) with 2% tolerance.
    3) Dynamic Checking: router-level flow conservation (R1) when router has >= 2 interfaces.
    Additionally, enforce interface consistency for rates when status is down.

    Confidence calibration:
    - High confidence when redundant signals agree and small/zero corrections.
    - Confidence reduced proportionally to symmetry deviations and applied router-level adjustments.

    Note: We intentionally do not flip interface statuses to preserve high status accuracy,
    but we reduce status confidence when peers disagree.
    """
    # Measurement timing tolerance (from Hodor research: ~2%)
    HARDENING_THRESHOLD = 0.02

    def clamp01(x: float) -> float:
        return max(0.0, min(1.0, x))

    # Precompute originals
    orig_rx: Dict[str, float] = {}
    orig_tx: Dict[str, float] = {}
    status: Dict[str, str] = {}
    peer_of: Dict[str, str] = {}
    for if_id, data in telemetry.items():
        orig_rx[if_id] = float(data.get('rx_rate', 0.0))
        orig_tx[if_id] = float(data.get('tx_rate', 0.0))
        status[if_id] = data.get('interface_status', 'unknown')
        peer_of[if_id] = data.get('connected_to')

    # Pair hardening: use bilateral agreement to estimate direction rates
    hardened_rx: Dict[str, float] = {}
    hardened_tx: Dict[str, float] = {}
    conf_rx: Dict[str, float] = {}
    conf_tx: Dict[str, float] = {}

    for if_id, data in telemetry.items():
        my_status = status.get(if_id, 'unknown')
        my_up = (my_status == 'up')
        peer_id = peer_of.get(if_id)
        peer_data = telemetry.get(peer_id, {}) if peer_id in telemetry else None
        peer_status = peer_data.get('interface_status', 'unknown') if peer_data else 'unknown'
        peer_up = (peer_status == 'up')

        my_rx = orig_rx[if_id]
        my_tx = orig_tx[if_id]
        peer_rx = float(peer_data.get('rx_rate', 0.0)) if peer_data else 0.0
        peer_tx = float(peer_data.get('tx_rate', 0.0)) if peer_data else 0.0

        # Relative diffs for both directions
        tx_to_peer_rx_diff = abs(my_tx - peer_rx) / max(1.0, abs(my_tx), abs(peer_rx))
        rx_from_peer_tx_diff = abs(my_rx - peer_tx) / max(1.0, abs(my_rx), abs(peer_tx))

        # Base hardened values
        if my_up and peer_up and peer_data:
            # Average redundant signals to reduce measurement noise
            hardened_tx_val = 0.5 * (my_tx + peer_rx)
            hardened_rx_val = 0.5 * (my_rx + peer_tx)
            # Confidence primarily from agreement; boost when within tolerance
            base_tx_conf = clamp01(1.0 - tx_to_peer_rx_diff)
            base_rx_conf = clamp01(1.0 - rx_from_peer_tx_diff)
            # Slight bump when within tolerance to reflect redundancy agreement
            if tx_to_peer_rx_diff <= HARDENING_THRESHOLD:
                base_tx_conf = clamp01(0.9 + 0.1 * (1.0 - tx_to_peer_rx_diff / max(HARDENING_THRESHOLD, 1e-9)))
            if rx_from_peer_tx_diff <= HARDENING_THRESHOLD:
                base_rx_conf = clamp01(0.9 + 0.1 * (1.0 - rx_from_peer_tx_diff / max(HARDENING_THRESHOLD, 1e-9)))
        else:
            # If peer missing or either side is down/unknown: rely on local, but reduce confidence
            if my_up:
                hardened_tx_val = my_tx
                hardened_rx_val = my_rx
                # Lower baseline confidence without redundancy
                base_tx_conf = 0.6
                base_rx_conf = 0.6
            else:
                # Interface down cannot send/receive
                hardened_tx_val = 0.0
                hardened_rx_val = 0.0
                base_tx_conf = 0.8  # Strong invariant: down -> zero traffic
                base_rx_conf = 0.8

        hardened_tx[if_id] = max(0.0, hardened_tx_val)
        hardened_rx[if_id] = max(0.0, hardened_rx_val)
        conf_tx[if_id] = clamp01(base_tx_conf)
        conf_rx[if_id] = clamp01(base_rx_conf)

    # Router-level flow conservation (R1) for routers with >= 2 interfaces
    # Only adjust when mismatch exceeds tolerance, and adjust the lower-confidence side.
    for router_id, if_list in topology.items():
        if_list = [i for i in if_list if i in telemetry]
        if len(if_list) < 2:
            # Avoid over-correcting single-link routers; not enough redundancy
            continue

        sum_rx = sum(hardened_rx.get(i, 0.0) for i in if_list)
        sum_tx = sum(hardened_tx.get(i, 0.0) for i in if_list)
        max_sum = max(1.0, sum_rx, sum_tx)
        rel_gap = abs(sum_rx - sum_tx) / max_sum

        if rel_gap <= HARDENING_THRESHOLD:
            continue  # within tolerance

        delta = sum_rx - sum_tx  # positive: rx larger than tx
        # Aggregate confidences per side
        agg_conf_rx = sum(conf_rx.get(i, 0.5) for i in if_list)
        agg_conf_tx = sum(conf_tx.get(i, 0.5) for i in if_list)

        # Choose side with lower aggregate confidence to adjust
        adjust_side = 'rx' if agg_conf_rx < agg_conf_tx else 'tx'

        # Build weights: favor larger links and lower-confidence signals
        if adjust_side == 'rx':
            vals = [hardened_rx.get(i, 0.0) for i in if_list]
            confs = [conf_rx.get(i, 0.5) for i in if_list]
            # We need sum(new_rx) = sum_tx -> total adjustment = -delta
            total_adjust = -delta
        else:
            vals = [hardened_tx.get(i, 0.0) for i in if_list]
            confs = [conf_tx.get(i, 0.5) for i in if_list]
            # We need sum(new_tx) = sum_rx -> total adjustment = +delta
            total_adjust = delta

        weights = []
        for v, c in zip(vals, confs):
            # Larger v and lower confidence -> larger weight
            w = (abs(v) + 1e-6) * (1.0 - clamp01(c)) + 1e-6
            weights.append(w)
        total_w = sum(weights)
        if total_w <= 0:
            # Fallback to uniform weights
            weights = [1.0 for _ in if_list]
            total_w = float(len(if_list))

        # Apply distributed adjustments and reduce confidence proportionally to relative change
        for idx, if_id in enumerate(if_list):
            adj = total_adjust * (weights[idx] / total_w)
            if adjust_side == 'rx':
                old = hardened_rx[if_id]
                new_val = max(0.0, old + adj)
                hardened_rx[if_id] = new_val
                rel_change = abs(adj) / max(1.0, abs(old) + 1e-9)
                conf_rx[if_id] = clamp01(conf_rx[if_id] * (1.0 - rel_change))
            else:
                old = hardened_tx[if_id]
                new_val = max(0.0, old + adj)
                hardened_tx[if_id] = new_val
                rel_change = abs(adj) / max(1.0, abs(old) + 1e-9)
                conf_tx[if_id] = clamp01(conf_tx[if_id] * (1.0 - rel_change))

    # Finalize: enforce zero rates on down interfaces, and prepare output
    result: Dict[str, Dict[str, Tuple]] = {}
    for if_id, data in telemetry.items():
        my_status = status.get(if_id, 'unknown')
        my_up = (my_status == 'up')
        peer_id = peer_of.get(if_id)
        repaired_rx = hardened_rx.get(if_id, 0.0)
        repaired_tx = hardened_tx.get(if_id, 0.0)
        rx_conf = conf_rx.get(if_id, 0.5)
        tx_conf = conf_tx.get(if_id, 0.5)

        # Enforce down => zero traffic with strong confidence
        if not my_up:
            repaired_rx = 0.0
            repaired_tx = 0.0
            rx_conf = max(rx_conf, 0.8)
            tx_conf = max(tx_conf, 0.8)

        # Status confidence handling (we do not change statuses)
        status_conf = 1.0
        if peer_id and peer_id in telemetry:
            peer_status = telemetry[peer_id].get('interface_status', 'unknown')
            if my_status != peer_status:
                status_conf = 0.6  # inconsistent pair statuses

        repaired_data: Dict[str, Any] = {}
        repaired_data['rx_rate'] = (orig_rx.get(if_id, 0.0), repaired_rx, clamp01(rx_conf))
        repaired_data['tx_rate'] = (orig_tx.get(if_id, 0.0), repaired_tx, clamp01(tx_conf))
        repaired_data['interface_status'] = (my_status, my_status, status_conf)

        # Copy metadata unchanged
        repaired_data['connected_to'] = data.get('connected_to')
        repaired_data['local_router'] = data.get('local_router')
        repaired_data['remote_router'] = data.get('remote_router')

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
