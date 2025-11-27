# EVOLVE-BLOCK-START
"""
POCS-based network telemetry repair:
- Iteratively projects counters onto link-symmetry and router flow-conservation sets.
- Harmonizes interface status using traffic evidence.
- Calibrates confidence via change magnitude, link residuals, and router residuals.

Key invariants:
- Link Symmetry (R3): my_tx ≈ their_rx; my_rx ≈ their_tx
- Flow Conservation (R1): Σ incoming = Σ outgoing at each router
- Interface Consistency: status aligned across link pairs

This algorithm uses a global, iterative solver (projections onto convex sets) to
jointly reconcile all counters across the network, which improves accuracy when
errors are correlated across multiple links/routers.
"""
from typing import Dict, Any, Tuple, List


def repair_network_telemetry(telemetry: Dict[str, Dict[str, Any]],
                             topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:
    # Parameters
    HARDENING_THRESHOLD = 0.02  # 2% tolerance for symmetry differences
    EPS = 1e-6                  # numeric floor for zero detection
    LINK_DAMP = 0.6             # damping for link projection (0..1)
    ROUTER_DAMP = 0.5           # damping for router projection (0..1)
    ITERATIONS = 10             # number of POCS iterations

    def clamp01(x: float) -> float:
        return max(0.0, min(1.0, float(x)))

    def rel_diff(a: float, b: float) -> float:
        return abs(a - b) / max(a, b, 1.0)

    def change_ratio(orig: float, rep: float) -> float:
        # magnitude of change normalized by scale
        denom = max(abs(orig), abs(rep), 1.0)
        return abs(rep - orig) / denom

    def has_traffic_val(rx: float, tx: float) -> bool:
        return (rx > EPS) or (tx > EPS)

    # Build unique link pairs (undirected, canonicalized)
    pairs = {}  # key: tuple(sorted([if1, if2])) -> (ifA, ifB)
    for if_id, data in telemetry.items():
        peer = data.get('connected_to')
        if peer and peer in telemetry:
            key = tuple(sorted([if_id, peer]))
            if key not in pairs:
                pairs[key] = (if_id, peer)

    # Build router->interfaces from topology; fall back to telemetry if needed
    router_if = {}
    if topology:
        # Use provided topology mapping
        for r, ifs in topology.items():
            router_if[r] = list(ifs)
    else:
        # Fallback: infer from telemetry
        for if_id, d in telemetry.items():
            r = d.get('local_router')
            if r is not None:
                router_if.setdefault(r, []).append(if_id)
        # If no topology present, we still apply link projections.
        # Router projections will be skipped for routers with no list.

    # Initialize estimates (start at observation, zero if interface is down)
    rx_hat: Dict[str, float] = {}
    tx_hat: Dict[str, float] = {}
    for if_id, d in telemetry.items():
        status = d.get('interface_status', 'unknown')
        rx0 = float(d.get('rx_rate', 0.0) or 0.0)
        tx0 = float(d.get('tx_rate', 0.0) or 0.0)
        if status == 'down':
            rx_hat[if_id] = 0.0
            tx_hat[if_id] = 0.0
        else:
            rx_hat[if_id] = rx0
            tx_hat[if_id] = tx0

    # Iterative POCS: alternate link and router projections
    for _ in range(ITERATIONS):
        # 1) Link symmetry projection for each pair
        for _, (a_id, b_id) in pairs.items():
            a = telemetry[a_id]
            b = telemetry[b_id]

            a_status = a.get('interface_status', 'unknown')
            b_status = b.get('interface_status', 'unknown')

            # If both down, keep zero
            if a_status == 'down' and b_status == 'down':
                rx_hat[a_id] = 0.0
                tx_hat[a_id] = 0.0
                rx_hat[b_id] = 0.0
                tx_hat[b_id] = 0.0
                continue

            # Project (A.tx, B.rx) to equality
            atx = tx_hat[a_id]
            brx = rx_hat[b_id]
            m1 = 0.5 * (atx + brx)
            tx_hat[a_id] = (1 - LINK_DAMP) * atx + LINK_DAMP * m1
            rx_hat[b_id] = (1 - LINK_DAMP) * brx + LINK_DAMP * m1

            # Project (B.tx, A.rx) to equality
            btx = tx_hat[b_id]
            arx = rx_hat[a_id]
            m2 = 0.5 * (btx + arx)
            tx_hat[b_id] = (1 - LINK_DAMP) * btx + LINK_DAMP * m2
            rx_hat[a_id] = (1 - LINK_DAMP) * arx + LINK_DAMP * m2

            # Clamp non-negativity
            if tx_hat[a_id] < 0.0: tx_hat[a_id] = 0.0
            if rx_hat[b_id] < 0.0: rx_hat[b_id] = 0.0
            if tx_hat[b_id] < 0.0: tx_hat[b_id] = 0.0
            if rx_hat[a_id] < 0.0: rx_hat[a_id] = 0.0

        # 2) Router flow conservation projection
        for router, if_list in router_if.items():
            # Consider only interfaces present in telemetry
            vars_if = [iid for iid in if_list if iid in telemetry]
            if not vars_if:
                continue
            # Exclude strictly-down interfaces (kept at zero)
            active = []
            for iid in vars_if:
                status = telemetry[iid].get('interface_status', 'unknown')
                if status != 'down':
                    active.append(iid)
            if not active:
                continue

            sum_tx = sum(tx_hat[iid] for iid in active)
            sum_rx = sum(rx_hat[iid] for iid in active)
            delta = sum_tx - sum_rx  # want delta -> 0

            k = 2 * len(active)  # number of variables in this hyperplane
            if k <= 0:
                continue
            step = ROUTER_DAMP * (delta / k)

            # Project onto c^T v = 0: subtract step from tx, add step to rx
            for iid in active:
                tx_hat[iid] = max(0.0, tx_hat[iid] - step)
                rx_hat[iid] = max(0.0, rx_hat[iid] + step)

    # After convergence, compute router residuals for confidence scaling
    router_residual: Dict[str, float] = {}
    for router, if_list in router_if.items():
        sum_tx = 0.0
        sum_rx = 0.0
        for iid in if_list:
            if iid in rx_hat:
                sum_tx += float(tx_hat[iid])
                sum_rx += float(rx_hat[iid])
        resid = abs(sum_tx - sum_rx) / max(sum_tx, sum_rx, 1.0)
        router_residual[router] = resid

    # Compute link residuals after final projection
    # Map each interface to the relevant link residuals for RX and TX
    link_tx_resid: Dict[str, float] = {}
    link_rx_resid: Dict[str, float] = {}
    for _, (a_id, b_id) in pairs.items():
        # Residual for A.tx vs B.rx
        res_ab = rel_diff(tx_hat[a_id], rx_hat[b_id])
        # Residual for B.tx vs A.rx
        res_ba = rel_diff(tx_hat[b_id], rx_hat[a_id])
        link_tx_resid[a_id] = res_ab
        link_rx_resid[b_id] = res_ab
        link_tx_resid[b_id] = res_ba
        link_rx_resid[a_id] = res_ba

    # Derive repaired status using evidence + pair harmonization
    repaired_status_map: Dict[str, str] = {}
    status_conf_map: Dict[str, float] = {}

    for if_id, d in telemetry.items():
        my_status = d.get('interface_status', 'unknown')
        peer_id = d.get('connected_to')
        rx_final = float(rx_hat.get(if_id, 0.0))
        tx_final = float(tx_hat.get(if_id, 0.0))
        my_has = has_traffic_val(rx_final, tx_final)

        if peer_id and peer_id in telemetry:
            peer_status = telemetry[peer_id].get('interface_status', 'unknown')
            peer_has = has_traffic_val(float(rx_hat.get(peer_id, 0.0)), float(tx_hat.get(peer_id, 0.0)))

            if my_status == 'down' and peer_status == 'down':
                # Both say down: keep down if no traffic on either side
                pair_status = 'down' if not (my_has or peer_has) else 'up'
                conf = 0.98 if pair_status == 'down' else 0.75
            elif my_status == 'up' and peer_status == 'up':
                pair_status = 'up'
                conf = 0.95 if (my_has or peer_has or (rx_final <= EPS and tx_final <= EPS)) else 0.9
            else:
                # Mixed statuses: use traffic evidence
                pair_status = 'up' if (my_has or peer_has) else 'down'
                conf = 0.8 if pair_status == 'up' else 0.7
        else:
            # No peer info: rely on local status and traffic
            if my_status == 'down':
                pair_status = 'down'
                conf = 0.95 if not my_has else 0.7
            else:
                pair_status = 'up' if my_status == 'up' or my_has else 'down'
                conf = 0.85 if pair_status == 'up' else 0.7

        repaired_status_map[if_id] = pair_status
        status_conf_map[if_id] = clamp01(conf)

    # Assemble final results and confidence for counters
    result: Dict[str, Dict[str, Tuple]] = {}
    for if_id, d in telemetry.items():
        rx_orig = float(d.get('rx_rate', 0.0) or 0.0)
        tx_orig = float(d.get('tx_rate', 0.0) or 0.0)
        local_router = d.get('local_router')
        remote_router = d.get('remote_router')

        rx_rep = float(rx_hat.get(if_id, rx_orig))
        tx_rep = float(tx_hat.get(if_id, tx_orig))
        status_rep = repaired_status_map.get(if_id, d.get('interface_status', 'unknown'))

        # Base confidence from change magnitude
        base_rx = 1.0 - change_ratio(rx_orig, rx_rep)
        base_tx = 1.0 - change_ratio(tx_orig, tx_rep)

        # Link penalty (higher residual -> lower confidence). Tolerate <= τh.
        lrx_res = link_rx_resid.get(if_id, 0.0)
        ltx_res = link_tx_resid.get(if_id, 0.0)
        p_link_rx = 1.0 - max(0.0, lrx_res - HARDENING_THRESHOLD)
        p_link_tx = 1.0 - max(0.0, ltx_res - HARDENING_THRESHOLD)

        # Router penalty uses both local and remote routers if available
        resid_local = router_residual.get(local_router, 0.0)
        resid_remote = router_residual.get(remote_router, 0.0)
        p_router = 1.0 - 0.5 * (resid_local + resid_remote)
        p_router = clamp01(p_router)

        # Final calibrated confidences: weighted blend for calibration stability
        rx_conf = clamp01(0.55 * base_rx + 0.25 * p_router + 0.20 * p_link_rx)
        tx_conf = clamp01(0.55 * base_tx + 0.25 * p_router + 0.20 * p_link_tx)

        # Special case: if interface determined down -> counters must be 0 with confidence depending on original traffic
        if status_rep == 'down':
            rx_rep = 0.0
            tx_rep = 0.0
            if not has_traffic_val(rx_orig, tx_orig):
                rx_conf = max(rx_conf, 0.95)
                tx_conf = max(tx_conf, 0.95)
            else:
                # We zeroed non-zero traffic; lower confidence
                rx_conf = min(rx_conf, 0.7)
                tx_conf = min(tx_conf, 0.7)

        status_conf = status_conf_map.get(if_id, 0.8)
        # Mild scaling of status confidence by router penalty for dynamic checking
        status_conf = clamp01(status_conf * (0.75 + 0.25 * p_router))

        repaired_data: Dict[str, Any] = {}
        repaired_data['rx_rate'] = (rx_orig, rx_rep, rx_conf)
        repaired_data['tx_rate'] = (tx_orig, tx_rep, tx_conf)
        repaired_data['interface_status'] = (d.get('interface_status', 'unknown'), status_rep, status_conf)

        # Copy metadata unchanged
        repaired_data['connected_to'] = d.get('connected_to')
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

