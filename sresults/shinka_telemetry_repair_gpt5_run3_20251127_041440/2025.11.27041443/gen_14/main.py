# EVOLVE-BLOCK-START
"""
Consensus-projection telemetry repair:
- Solve a constrained consensus (least-squares with equality constraints) by iterative projections.
- Factors: measurement anchoring, link symmetry (R3), router flow conservation (R1), and status constraints.
- Confidence calibrated from final residuals to measurement, link symmetry, and router balance.

This implementation uses only Python stdlib and does not rely on external solvers.
"""
from typing import Dict, Any, Tuple, List


def repair_network_telemetry(telemetry: Dict[str, Dict[str, Any]],
                             topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:
    # Tolerances and hyperparameters (tuned to be stable and responsive)
    TAU = 0.02  # hardening threshold ~2%
    EPS = 1e-9
    ITER = 12

    # Step sizes for projections
    ETA_MEAS = 0.35
    ETA_LINK = 0.60
    ETA_RTR = 0.25
    ETA_DOWN = 0.90

    def clamp01(x: float) -> float:
        return max(0.0, min(1.0, x))

    def rel_diff(a: float, b: float) -> float:
        return abs(a - b) / max(1.0, abs(a), abs(b))

    # Collect originals and metadata
    orig_rx: Dict[str, float] = {}
    orig_tx: Dict[str, float] = {}
    status: Dict[str, str] = {}
    peer_of: Dict[str, str] = {}

    for if_id, data in telemetry.items():
        orig_rx[if_id] = float(data.get('rx_rate', 0.0))
        orig_tx[if_id] = float(data.get('tx_rate', 0.0))
        status[if_id] = data.get('interface_status', 'unknown')
        peer = data.get('connected_to')
        peer_of[if_id] = peer if peer in telemetry else None

    # Build router membership using topology; fallback to local_router if needed
    router_ifaces: Dict[str, List[str]] = {}
    if topology:
        for r, ifs in topology.items():
            router_ifaces[r] = [i for i in ifs if i in telemetry]
    else:
        # If topology missing, leverage local_router fields; still useful for R1
        for iid, data in telemetry.items():
            r = data.get('local_router')
            if r is not None:
                router_ifaces.setdefault(r, []).append(iid)

    # Initialize solution with originals (clamped non-negative)
    x_rx: Dict[str, float] = {i: max(0.0, v) for i, v in orig_rx.items()}
    x_tx: Dict[str, float] = {i: max(0.0, v) for i, v in orig_tx.items()}

    # Precompute per-variable measurement weights based on redundancy quality
    w_meas_rx: Dict[str, float] = {}
    w_meas_tx: Dict[str, float] = {}

    for i in telemetry:
        p = peer_of.get(i)
        my_up = (status.get(i) == 'up')
        base_rx = 0.75
        base_tx = 0.75
        if p:
            # If we have redundancy, weight anchors based on how consistent the pair is
            d_tx_to_peer_rx = rel_diff(orig_tx[i], float(telemetry[p].get('rx_rate', 0.0)))
            d_rx_to_peer_tx = rel_diff(orig_rx[i], float(telemetry[p].get('tx_rate', 0.0)))
            # If measurements agree, trust them more; if they disagree, anchor less to avoid locking in errors
            base_tx = 0.9 if d_tx_to_peer_rx <= TAU else max(0.55, 0.9 * (1.0 - min(1.0, d_tx_to_peer_rx)))
            base_rx = 0.9 if d_rx_to_peer_tx <= TAU else max(0.55, 0.9 * (1.0 - min(1.0, d_rx_to_peer_tx)))
        else:
            # No redundancy, moderate anchor
            base_rx = 0.7
            base_tx = 0.7

        # If interface is down, we will hard-project to zero anyway; keep small anchor to avoid conflicts
        if not my_up:
            base_rx = 0.4
            base_tx = 0.4

        w_meas_rx[i] = base_rx
        w_meas_tx[i] = base_tx

    # Precompute link pairs
    link_pairs = []
    visited = set()
    for a in telemetry:
        b = peer_of.get(a)
        if not b or b in visited or a == b or b not in telemetry:
            continue
        visited.add(a)
        visited.add(b)
        link_pairs.append((a, b))

    # Iterative consensus projections
    for _ in range(ITER):
        # 1) Measurement anchoring
        for i in telemetry:
            # Move towards original measurement with a weight reflecting redundancy agreement
            x_rx[i] += ETA_MEAS * w_meas_rx[i] * (orig_rx[i] - x_rx[i])
            x_tx[i] += ETA_MEAS * w_meas_tx[i] * (orig_tx[i] - x_tx[i])
            # Keep non-negative
            if x_rx[i] < 0.0:
                x_rx[i] = 0.0
            if x_tx[i] < 0.0:
                x_tx[i] = 0.0

        # 2) Link symmetry (R3) projections
        for a, b in link_pairs:
            a_up = (status.get(a) == 'up')
            b_up = (status.get(b) == 'up')

            if not a_up or not b_up:
                # If either endpoint is down, drive both directions on the link to zero rapidly
                x_tx[a] += ETA_DOWN * (0.0 - x_tx[a])
                x_rx[a] += ETA_DOWN * (0.0 - x_rx[a])
                x_tx[b] += ETA_DOWN * (0.0 - x_tx[b])
                x_rx[b] += ETA_DOWN * (0.0 - x_rx[b])
                # Clamp
                x_tx[a] = max(0.0, x_tx[a])
                x_rx[a] = max(0.0, x_rx[a])
                x_tx[b] = max(0.0, x_tx[b])
                x_rx[b] = max(0.0, x_rx[b])
                continue

            # Direction 1: a.tx == b.rx
            d1 = x_tx[a] - x_rx[b]
            mid1 = 0.5 * (x_tx[a] + x_rx[b])
            # Weight stronger when current mismatch is large relative to threshold
            rel1 = abs(d1) / max(1.0, abs(mid1))
            w1 = min(1.5, max(0.2, rel1 / max(TAU, 1e-6)))
            alpha1 = ETA_LINK * clamp01(w1)
            x_tx[a] += alpha1 * (mid1 - x_tx[a])
            x_rx[b] += alpha1 * (mid1 - x_rx[b])

            # Direction 2: a.rx == b.tx
            d2 = x_rx[a] - x_tx[b]
            mid2 = 0.5 * (x_rx[a] + x_tx[b])
            rel2 = abs(d2) / max(1.0, abs(mid2))
            w2 = min(1.5, max(0.2, rel2 / max(TAU, 1e-6)))
            alpha2 = ETA_LINK * clamp01(w2)
            x_rx[a] += alpha2 * (mid2 - x_rx[a])
            x_tx[b] += alpha2 * (mid2 - x_tx[b])

        # 3) Router flow conservation (R1) small distributed correction steps
        for r, ifs in router_ifaces.items():
            if not ifs:
                continue
            # Consider up interfaces only to avoid double-imposing zero for down links
            up_ifs = [i for i in ifs if status.get(i) == 'up']
            if len(up_ifs) < 1:
                continue
            sum_rx = sum(x_rx[i] for i in up_ifs)
            sum_tx = sum(x_tx[i] for i in up_ifs)
            total = max(1.0, sum_rx, sum_tx)
            imbalance = (sum_rx - sum_tx)
            rel_gap = abs(imbalance) / total
            if rel_gap <= TAU:
                continue
            # Weight stronger when greater imbalance
            w_r = min(2.0, max(0.25, rel_gap / TAU))
            alpha_r = ETA_RTR * clamp01(w_r)

            # Distribute equal per-variable step; move rx down and tx up (or vice versa) to close the gap
            n = len(up_ifs)
            if n <= 0:
                continue
            per = imbalance / n
            # Update with small step scaling
            for i in up_ifs:
                # rx := rx - alpha * per, tx := tx + alpha * per
                x_rx[i] -= alpha_r * per
                x_tx[i] += alpha_r * per
                # Clamp non-negative
                if x_rx[i] < 0.0:
                    x_rx[i] = 0.0
                if x_tx[i] < 0.0:
                    x_tx[i] = 0.0

        # 4) Enforce hard status constraints after all adjustments
        for i in telemetry:
            if status.get(i) != 'up':
                x_rx[i] = 0.0
                x_tx[i] = 0.0

    # Final strict projection on links to remove residual asymmetry
    for a, b in link_pairs:
        if status.get(a) != 'up' or status.get(b) != 'up':
            x_tx[a] = x_rx[a] = x_tx[b] = x_rx[b] = 0.0
        else:
            v1 = 0.5 * (x_tx[a] + x_rx[b])
            v2 = 0.5 * (x_rx[a] + x_tx[b])
            x_tx[a] = v1
            x_rx[b] = v1
            x_rx[a] = v2
            x_tx[b] = v2

    # Prepare per-router residuals for confidence calibration
    router_residual: Dict[str, float] = {}
    for r, ifs in router_ifaces.items():
        if not ifs:
            router_residual[r] = 0.0
            continue
        sum_rx = sum(x_rx[i] for i in ifs)
        sum_tx = sum(x_tx[i] for i in ifs)
        denom = max(1.0, sum_rx, sum_tx)
        router_residual[r] = abs(sum_rx - sum_tx) / denom

    # Build mapping interface->router (for residual lookup)
    iface_router: Dict[str, str] = {}
    for r, ifs in router_ifaces.items():
        for i in ifs:
            iface_router[i] = r

    # Confidence estimation per variable from residuals
    def compute_confidence(i: str) -> Tuple[float, float]:
        p = peer_of.get(i)
        # Measurement residuals
        r_meas_rx = rel_diff(x_rx[i], orig_rx[i])
        r_meas_tx = rel_diff(x_tx[i], orig_tx[i])
        # Link residuals (if peer)
        if p:
            r_link_tx = rel_diff(x_tx[i], x_rx[p])
            r_link_rx = rel_diff(x_rx[i], x_tx[p])
        else:
            # No redundancy: assume moderate residual
            r_link_tx = 0.2
            r_link_rx = 0.2
        # Router residual at this interface's router
        rtr = router_residual.get(iface_router.get(i, ""), 0.0)

        # Blend residuals with weights emphasizing measurement and linkage
        c_rx = 1.0 - (0.55 * r_meas_rx + 0.35 * r_link_rx + 0.10 * rtr)
        c_tx = 1.0 - (0.55 * r_meas_tx + 0.35 * r_link_tx + 0.10 * rtr)
        c_rx = clamp01(c_rx)
        c_tx = clamp01(c_tx)

        # If interface is down, zero is a strong invariant; raise floor confidence
        if status.get(i) != 'up':
            c_rx = max(c_rx, 0.85)
            c_tx = max(c_tx, 0.85)
        return c_rx, c_tx

    # Assemble result with calibrated confidence and unchanged metadata
    result: Dict[str, Dict[str, Tuple]] = {}
    for i, data in telemetry.items():
        repaired: Dict[str, Any] = {}

        rx_conf, tx_conf = compute_confidence(i)

        # Status handling: do not flip, but reduce confidence on inconsistency
        peer_id = peer_of.get(i)
        my_status = status.get(i, 'unknown')
        status_confidence = 1.0
        if peer_id and peer_id in telemetry:
            peer_status = telemetry[peer_id].get('interface_status', 'unknown')
            if my_status != peer_status:
                status_confidence = 0.6
        if my_status == 'down' and (orig_rx.get(i, 0.0) > 1e-3 or orig_tx.get(i, 0.0) > 1e-3):
            status_confidence = min(status_confidence, 0.6)

        repaired['rx_rate'] = (orig_rx.get(i, 0.0), x_rx.get(i, 0.0), rx_conf)
        repaired['tx_rate'] = (orig_tx.get(i, 0.0), x_tx.get(i, 0.0), tx_conf)
        repaired['interface_status'] = (my_status, my_status, status_confidence)

        # Copy metadata unchanged
        repaired['connected_to'] = data.get('connected_to')
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