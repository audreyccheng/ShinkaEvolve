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
    3) Dynamic Checking: router-level flow conservation (R1) with guarded, capped scaling.
    Additionally, enforce interface consistency for rates when status is down.

    Confidence calibration:
    - High confidence when redundant signals agree and small/zero corrections.
    - Confidence reduced proportionally to symmetry deviations and applied router-level adjustments.
    - Final calibration blends measurement residuals, link residuals, and router residuals.

    Note: We intentionally do not flip interface statuses to preserve status accuracy,
    but we reduce status confidence when peers disagree.
    """
    # Measurement timing tolerance (from Hodor research: ~2%)
    HARDENING_THRESHOLD = 0.02
    ZERO_EPS = 1e-3

    def clamp01(x: float) -> float:
        return max(0.0, min(1.0, x))

    def rel_diff(a: float, b: float) -> float:
        return abs(a - b) / max(1.0, abs(a), abs(b))

    def near_zero(x: float) -> bool:
        return abs(x) < ZERO_EPS

    # Precompute originals and peers
    orig_rx: Dict[str, float] = {}
    orig_tx: Dict[str, float] = {}
    status: Dict[str, str] = {}
    peer_of: Dict[str, str] = {}

    for if_id, data in telemetry.items():
        orig_rx[if_id] = float(data.get('rx_rate', 0.0))
        orig_tx[if_id] = float(data.get('tx_rate', 0.0))
        status[if_id] = data.get('interface_status', 'unknown')
        ct = data.get('connected_to')
        peer_of[if_id] = ct if ct in telemetry else None

    # Initialize hardened values with originals
    hardened_rx: Dict[str, float] = {i: max(0.0, v) for i, v in orig_rx.items()}
    hardened_tx: Dict[str, float] = {i: max(0.0, v) for i, v in orig_tx.items()}
    conf_rx: Dict[str, float] = {i: 0.7 for i in telemetry}
    conf_tx: Dict[str, float] = {i: 0.7 for i in telemetry}

    processed_pairs = set()

    # Pairwise hardening using link symmetry (R3) with "both_idle" quiescence detection
    for a, data in telemetry.items():
        b = peer_of.get(a)
        if not b or (b, a) in processed_pairs or a == b:
            continue
        processed_pairs.add((a, b))

        a_stat = status.get(a, 'unknown')
        b_stat = status.get(b, 'unknown')
        a_up = (a_stat == 'up')
        b_up = (b_stat == 'up')

        a_rx = orig_rx[a]
        a_tx = orig_tx[a]
        b_rx = orig_rx[b]
        b_tx = orig_tx[b]

        # Quiescent link or both down -> zero traffic with high confidence
        both_idle = (near_zero(a_rx) and near_zero(a_tx) and near_zero(b_rx) and near_zero(b_tx))
        if not a_up or not b_up or both_idle:
            hardened_rx[a] = 0.0
            hardened_tx[a] = 0.0
            hardened_rx[b] = 0.0
            hardened_tx[b] = 0.0
            # Confidence high due to strong invariant
            conf_rx[a] = max(conf_rx[a], 0.95 if both_idle else 0.85)
            conf_tx[a] = max(conf_tx[a], 0.95 if both_idle else 0.85)
            conf_rx[b] = max(conf_rx[b], 0.95 if both_idle else 0.85)
            conf_tx[b] = max(conf_tx[b], 0.95 if both_idle else 0.85)
            continue

        # Direction 1: a.tx should match b.rx
        d1 = rel_diff(a_tx, b_rx)
        if d1 <= HARDENING_THRESHOLD:
            v1 = 0.5 * (a_tx + b_rx)
            hardened_tx[a] = max(0.0, v1)
            hardened_rx[b] = max(0.0, v1)
            c1 = clamp01(0.9 + 0.1 * (1.0 - d1 / max(HARDENING_THRESHOLD, 1e-12)))
            conf_tx[a] = max(conf_tx[a], c1)
            conf_rx[b] = max(conf_rx[b], c1)
        else:
            # Snap to peer's measurement for strong symmetry
            hardened_tx[a] = max(0.0, b_rx)
            hardened_rx[b] = max(0.0, b_rx)
            c1 = clamp01(1.0 - d1)
            conf_tx[a] = max(conf_tx[a], c1)
            conf_rx[b] = max(conf_rx[b], c1)

        # Direction 2: a.rx should match b.tx
        d2 = rel_diff(a_rx, b_tx)
        if d2 <= HARDENING_THRESHOLD:
            v2 = 0.5 * (a_rx + b_tx)
            hardened_rx[a] = max(0.0, v2)
            hardened_tx[b] = max(0.0, v2)
            c2 = clamp01(0.9 + 0.1 * (1.0 - d2 / max(HARDENING_THRESHOLD, 1e-12)))
            conf_rx[a] = max(conf_rx[a], c2)
            conf_tx[b] = max(conf_tx[b], c2)
        else:
            hardened_rx[a] = max(0.0, b_tx)
            hardened_tx[b] = max(0.0, a_rx)
            c2 = clamp01(1.0 - d2)
            conf_rx[a] = max(conf_rx[a], c2)
            conf_tx[b] = max(conf_tx[b], c2)

    # Unpaired interfaces: keep own values with moderate confidence
    in_any_pair = set([x for pair in processed_pairs for x in pair])
    for i, d in telemetry.items():
        if i not in in_any_pair:
            # If interface is down, enforce zero with strong confidence
            if status.get(i) == 'down':
                hardened_rx[i] = 0.0
                hardened_tx[i] = 0.0
                conf_rx[i] = max(conf_rx[i], 0.85)
                conf_tx[i] = max(conf_tx[i], 0.85)
            else:
                # Keep local but acknowledge weaker redundancy
                hardened_rx[i] = max(0.0, orig_rx[i])
                hardened_tx[i] = max(0.0, orig_tx[i])
                conf_rx[i] = max(conf_rx[i], 0.6)
                conf_tx[i] = max(conf_tx[i], 0.6)

    # Build router membership using provided topology (preferred)
    router_ifaces: Dict[str, List[str]] = {}
    if topology:
        for r, ifs in topology.items():
            router_ifaces[r] = [i for i in ifs if i in telemetry]
    else:
        # Fallback to local_router if topology not supplied (still useful for R1)
        for iid, d in telemetry.items():
            r = d.get('local_router')
            router_ifaces.setdefault(r, []).append(iid)

    # Guarded router-level flow conservation (R1)
    # Apply when router has >= 2 interfaces and imbalance > threshold, capped scaling (<=10% change).
    for r, ifs in router_ifaces.items():
        if len(ifs) < 2:
            continue
        # Consider only interfaces that are up to avoid double-penalizing down links
        up_ifs = [i for i in ifs if status.get(i) == 'up']
        if len(up_ifs) < 2:
            continue

        sum_rx = sum(hardened_rx[i] for i in up_ifs)
        sum_tx = sum(hardened_tx[i] for i in up_ifs)
        denom = max(1.0, sum_rx, sum_tx)
        imbalance = abs(sum_rx - sum_tx) / denom
        if imbalance <= HARDENING_THRESHOLD:
            continue

        avg_rx_conf = sum(conf_rx[i] for i in up_ifs) / len(up_ifs)
        avg_tx_conf = sum(conf_tx[i] for i in up_ifs) / len(up_ifs)

        # Decide which side to adjust based on lower confidence
        adjust_side = 'rx' if avg_rx_conf < avg_tx_conf else 'tx'
        ratio = (sum_rx + 1e-9) / (sum_tx + 1e-9)
        if adjust_side == 'rx':
            s_req = 1.0 / max(1e-9, ratio)  # scale rx by s_req
        else:
            s_req = max(1e-9, ratio)        # scale tx by s_req

        # Cap the magnitude of change to be gentle (<=10%)
        s_cap = 0.10
        s = max(1.0 - s_cap, min(1.0 + s_cap, s_req))

        # Apply scaling and penalize confidence by relative change
        rel_change = abs(s - 1.0)
        for i in up_ifs:
            if adjust_side == 'rx':
                old = hardened_rx[i]
                hardened_rx[i] = max(0.0, old * s)
                # Penalize proportional to change (10% => max penalty)
                conf_rx[i] = clamp01(conf_rx[i] * (1.0 - rel_change / s_cap))
            else:
                old = hardened_tx[i]
                hardened_tx[i] = max(0.0, old * s)
                conf_tx[i] = clamp01(conf_tx[i] * (1.0 - rel_change / s_cap))

    # Final symmetry touch-up to keep links consistent (R3)
    for a, b in list(processed_pairs):
        if a not in telemetry or b not in telemetry:
            continue
        v1 = 0.5 * (hardened_tx[a] + hardened_rx[b])
        v2 = 0.5 * (hardened_rx[a] + hardened_tx[b])
        hardened_tx[a] = max(0.0, v1)
        hardened_rx[b] = max(0.0, v1)
        hardened_rx[a] = max(0.0, v2)
        hardened_tx[b] = max(0.0, v2)
        # Slight confidence reduction due to final adjustment
        conf_rx[a] = clamp01(conf_rx[a] * 0.95)
        conf_tx[a] = clamp01(conf_tx[a] * 0.95)
        conf_rx[b] = clamp01(conf_rx[b] * 0.95)
        conf_tx[b] = clamp01(conf_tx[b] * 0.95)

    # Enforce interface down => zero traffic (final safeguard)
    for i in telemetry:
        if status.get(i) == 'down':
            hardened_rx[i] = 0.0
            hardened_tx[i] = 0.0
            conf_rx[i] = max(conf_rx[i], 0.85)
            conf_tx[i] = max(conf_tx[i], 0.85)

    # Residual-based confidence calibration phase
    # Compute router residuals after all adjustments
    router_residual: Dict[str, float] = {}
    for r, ifs in router_ifaces.items():
        if not ifs:
            router_residual[r] = 0.0
            continue
        sum_rx = sum(hardened_rx[i] for i in ifs)
        sum_tx = sum(hardened_tx[i] for i in ifs)
        denom = max(1.0, sum_rx, sum_tx)
        router_residual[r] = abs(sum_rx - sum_tx) / denom

    # Map iface -> router for residual lookup
    iface_router: Dict[str, str] = {}
    for r, ifs in router_ifaces.items():
        for iid in ifs:
            iface_router[iid] = r

    # Calibrate confidences using measurement, link, and router residuals
    for i in telemetry:
        # Measurement residuals
        r_meas_rx = rel_diff(hardened_rx[i], orig_rx[i])
        r_meas_tx = rel_diff(hardened_tx[i], orig_tx[i])

        # Link residuals
        p = peer_of.get(i)
        if p:
            r_link_tx = rel_diff(hardened_tx[i], hardened_rx[p])
            r_link_rx = rel_diff(hardened_rx[i], hardened_tx[p])
        else:
            r_link_tx = 0.2
            r_link_rx = 0.2

        # Router residual
        rtr = router_residual.get(iface_router.get(i, ""), 0.0)

        # Blend residuals; emphasize measurement and link agreement
        new_rx_conf = 1.0 - (0.55 * r_meas_rx + 0.35 * r_link_rx + 0.10 * rtr)
        new_tx_conf = 1.0 - (0.55 * r_meas_tx + 0.35 * r_link_tx + 0.10 * rtr)

        # Do not allow post-calibration to increase confidence beyond previously earned
        conf_rx[i] = clamp01(min(conf_rx.get(i, 0.6), new_rx_conf))
        conf_tx[i] = clamp01(min(conf_tx.get(i, 0.6), new_tx_conf))

        # If interface is down, zero is a strong invariant; raise floor
        if status.get(i) != 'up':
            conf_rx[i] = max(conf_rx[i], 0.85)
            conf_tx[i] = max(conf_tx[i], 0.85)

    # Assemble result with confidence calibration
    result: Dict[str, Dict[str, Tuple]] = {}
    for i, data in telemetry.items():
        interface_status = status.get(i, 'unknown')
        connected_to = data.get('connected_to')

        # Status confidence: penalize when peer status inconsistent or traffic present while down
        status_confidence = 1.0
        if connected_to and connected_to in telemetry:
            peer_status = telemetry[connected_to].get('interface_status', 'unknown')
            if interface_status != peer_status:
                status_confidence = 0.6
        if interface_status == 'down' and (orig_rx.get(i, 0.0) > ZERO_EPS or orig_tx.get(i, 0.0) > ZERO_EPS):
            status_confidence = min(status_confidence, 0.6)

        rx_c = clamp01(conf_rx.get(i, 0.6))
        tx_c = clamp01(conf_tx.get(i, 0.6))

        repaired: Dict[str, Any] = {}
        repaired['rx_rate'] = (orig_rx.get(i, 0.0), hardened_rx.get(i, 0.0), rx_c)
        repaired['tx_rate'] = (orig_tx.get(i, 0.0), hardened_tx.get(i, 0.0), tx_c)
        repaired['interface_status'] = (interface_status, interface_status, status_confidence)

        # Copy metadata unchanged
        repaired['connected_to'] = connected_to
        repaired['local_router'] = data.get('local_router')
        repaired['remote_router'] = data.get('remote_router')

        result[i] = repaired

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
