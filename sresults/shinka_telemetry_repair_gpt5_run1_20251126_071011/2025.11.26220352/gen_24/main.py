# EVOLVE-BLOCK-START
"""
Variance-weighted consensus telemetry repair:
- Factor-graph inspired projections with Gaussian beliefs.
- Link equality and router flow conservation enforced iteratively.
- Confidence derived from residuals (logistic), changes, and posterior variances.

Maintains the same inputs/outputs as prior implementations.
"""
from typing import Dict, Any, Tuple, List


def repair_network_telemetry(telemetry: Dict[str, Dict[str, Any]],
                             topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:
    # Core tolerances
    HARDENING_THRESHOLD = 0.02  # ~2% timing tolerance
    TRAFFIC_EVIDENCE_MIN = 0.5  # Mbps
    EPS = 1e-9

    # Iteration parameters
    ITERATIONS = 4
    ALPHA_LINK = 0.8
    ALPHA_ROUTER = 0.8
    FINAL_LINK_ALPHA = 0.35

    # Noise model for priors (sigma = rel*value + abs_floor)
    REL_SIGMA = 0.03
    ABS_SIGMA = 0.5  # Mbps

    # Helper functions
    def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
        return max(lo, min(hi, x))

    def rel_diff(a: float, b: float) -> float:
        denom = max(abs(a), abs(b), 1e-9)
        return abs(a - b) / denom

    def conf_from_residual_logistic(residual: float, tol: float) -> float:
        # Logistic decay: 1 at residual << tol, ~0.5 near tol, down thereafter
        tol = max(tol, 1e-9)
        x = residual / tol
        k = 3.0
        # 1 / (1 + e^{k(x-1)})
        import math
        return clamp(1.0 / (1.0 + math.e ** (k * (x - 1.0))))

    def init_variance(v: float) -> float:
        # Variance from absolute and relative noise
        scale = max(abs(v), 1.0)
        sigma = REL_SIGMA * scale + ABS_SIGMA
        return sigma * sigma

    # Build connected pairs
    visited_pairs = set()
    pairs: List[Tuple[str, str]] = []
    for if_id, data in telemetry.items():
        peer = data.get('connected_to')
        if peer and peer in telemetry:
            key = tuple(sorted([if_id, peer]))
            if key not in visited_pairs:
                visited_pairs.add(key)
                pairs.append((key[0], key[1]))

    peer_of: Dict[str, str] = {}
    paired_ids = set()
    for a_id, b_id in pairs:
        peer_of[a_id] = b_id
        peer_of[b_id] = a_id
        paired_ids.add(a_id)
        paired_ids.add(b_id)

    # Initialize working state
    work: Dict[str, Dict[str, Any]] = {}
    for if_id, data in telemetry.items():
        rx0 = float(data.get('rx_rate', 0.0))
        tx0 = float(data.get('tx_rate', 0.0))
        status0 = data.get('interface_status', 'unknown')
        work[if_id] = {
            'rx': rx0,
            'tx': tx0,
            'rx_var': init_variance(rx0),
            'tx_var': init_variance(tx0),
            'status': status0,
            'status_conf': 1.0,
            'connected_to': data.get('connected_to'),
            'local_router': data.get('local_router'),
            'remote_router': data.get('remote_router'),
            'orig_rx': rx0,
            'orig_tx': tx0,
            'orig_status': status0,
            'base_rx_var': init_variance(rx0),
            'base_tx_var': init_variance(tx0),
        }

    # Resolve interface status consistency across links with traffic evidence
    for a_id, b_id in pairs:
        a = work[a_id]; b = work[b_id]
        a_stat = a['status']; b_stat = b['status']
        max_traffic = max(a['rx'], a['tx'], b['rx'], b['tx'])
        if a_stat == b_stat:
            resolved = a_stat
            sconf = 0.95 if resolved in ('up', 'down') else 0.7
        else:
            if max_traffic > TRAFFIC_EVIDENCE_MIN:
                resolved = 'up'; sconf = 0.85
            else:
                resolved = 'down'; sconf = 0.75
        a['status'] = resolved; b['status'] = resolved
        a['status_conf'] = min(a['status_conf'], sconf)
        b['status_conf'] = min(b['status_conf'], sconf)

    # If status down, force zero rates and set small variance (high confidence)
    for if_id, r in work.items():
        if r['status'] == 'down':
            rx0, tx0 = r['rx'], r['tx']
            r['rx'] = 0.0; r['tx'] = 0.0
            # Very small variance to keep at zero
            r['rx_var'] = (ABS_SIGMA * 0.1) ** 2
            r['tx_var'] = (ABS_SIGMA * 0.1) ** 2
            r['status_conf'] = 0.9 if (rx0 <= TRAFFIC_EVIDENCE_MIN and tx0 <= TRAFFIC_EVIDENCE_MIN) else 0.5

    # Build router interfaces using provided topology else derive best-effort from telemetry
    router_ifaces: Dict[str, List[str]] = {}
    if topology:
        router_ifaces = {r: [i for i in lst if i in work] for r, lst in topology.items()}
    else:
        # Use telemetry metadata if topology not provided
        for if_id, data in work.items():
            rtr = data.get('local_router')
            if rtr is not None:
                router_ifaces.setdefault(rtr, []).append(if_id)

    # Iterative variance-weighted consensus
    import math
    for it in range(ITERATIONS):
        # Link consensus step
        for a_id, b_id in pairs:
            a = work[a_id]; b = work[b_id]
            if a['status'] != 'up' or b['status'] != 'up':
                continue
            # Forward direction: a.tx <-> b.rx
            x = float(a['tx']); vx = float(a['tx_var'])
            y = float(b['rx']); vy = float(b['rx_var'])
            vscale = max((x + y) / 2.0, 1.0)
            tau_link = max(HARDENING_THRESHOLD * vscale, 0.01 * vscale)
            p_link = 1.0 / (tau_link * tau_link)
            px = 1.0 / max(vx, EPS)
            py = 1.0 / max(vy, EPS)
            denom = px + py + p_link
            if denom > EPS:
                v_cons = (px * x + py * y) / denom
                new = (1.0 - ALPHA_LINK) * x + ALPHA_LINK * v_cons
                a['tx'] = max(0.0, new)
                b['rx'] = max(0.0, new)
                # Posterior variance for both ends (approximate sharing)
                v_post = 1.0 / denom
                a['tx_var'] = max(v_post, 0.01 * a['tx_var'])
                b['rx_var'] = max(v_post, 0.01 * b['rx_var'])

            # Reverse direction: a.rx <-> b.tx
            x = float(a['rx']); vx = float(a['rx_var'])
            y = float(b['tx']); vy = float(b['tx_var'])
            vscale = max((x + y) / 2.0, 1.0)
            tau_link = max(HARDENING_THRESHOLD * vscale, 0.01 * vscale)
            p_link = 1.0 / (tau_link * tau_link)
            px = 1.0 / max(vx, EPS)
            py = 1.0 / max(vy, EPS)
            denom = px + py + p_link
            if denom > EPS:
                v_cons = (px * x + py * y) / denom
                new = (1.0 - ALPHA_LINK) * x + ALPHA_LINK * v_cons
                a['rx'] = max(0.0, new)
                b['tx'] = max(0.0, new)
                v_post = 1.0 / denom
                a['rx_var'] = max(v_post, 0.01 * a['rx_var'])
                b['tx_var'] = max(v_post, 0.01 * b['tx_var'])

        # Router conservation step
        for router, if_list in router_ifaces.items():
            # Collect up interfaces
            up_ifaces = [i for i in if_list if work[i]['status'] == 'up']
            if not up_ifaces:
                continue

            sum_tx = sum(max(0.0, work[i]['tx']) for i in up_ifaces)
            sum_rx = sum(max(0.0, work[i]['rx']) for i in up_ifaces)
            need = sum_tx - sum_rx  # want this to be zero

            # If within tolerance skip
            if rel_diff(sum_tx, sum_rx) <= HARDENING_THRESHOLD * 1.5:
                continue

            # Use Lagrange multiplier solution: minimize sum (delta^2 / var) s.t. sum_tx+dt - (sum_rx+dr) = 0
            var_tx_sum = sum(max(work[i]['tx_var'], EPS) for i in up_ifaces)
            var_rx_sum = sum(max(work[i]['rx_var'], EPS) for i in up_ifaces)
            denom = var_tx_sum + var_rx_sum
            if denom <= EPS:
                continue
            lam = need / denom

            # Apply updates with damping and non-negativity; shrink variance slightly
            for i in up_ifaces:
                vtx = work[i]['tx_var']; vrx = work[i]['rx_var']
                d_tx = -ALPHA_ROUTER * lam * vtx
                d_rx = +ALPHA_ROUTER * lam * vrx
                # Clip to avoid negative
                if work[i]['tx'] + d_tx < 0.0:
                    d_tx = -work[i]['tx']
                if work[i]['rx'] + d_rx < 0.0:
                    d_rx = -work[i]['rx']
                work[i]['tx'] = max(0.0, work[i]['tx'] + d_tx)
                work[i]['rx'] = max(0.0, work[i]['rx'] + d_rx)
                # Posterior variance contraction (heuristic)
                work[i]['tx_var'] = max(0.8 * vtx, (ABS_SIGMA * 0.2) ** 2)
                work[i]['rx_var'] = max(0.8 * vrx, (ABS_SIGMA * 0.2) ** 2)

    # Final gentle link reconciliation to maintain symmetry after router step
    for a_id, b_id in pairs:
        a = work[a_id]; b = work[b_id]
        if a['status'] != 'up' or b['status'] != 'up':
            continue
        # Forward
        x = a['tx']; y = b['rx']
        vmid = 0.5 * (x + y)
        a['tx'] = max(0.0, (1 - FINAL_LINK_ALPHA) * x + FINAL_LINK_ALPHA * vmid)
        b['rx'] = max(0.0, (1 - FINAL_LINK_ALPHA) * y + FINAL_LINK_ALPHA * vmid)
        # Reverse
        x = a['rx']; y = b['tx']
        vmid = 0.5 * (x + y)
        a['rx'] = max(0.0, (1 - FINAL_LINK_ALPHA) * x + FINAL_LINK_ALPHA * vmid)
        b['tx'] = max(0.0, (1 - FINAL_LINK_ALPHA) * y + FINAL_LINK_ALPHA * vmid)

    # Enforce "down implies zero" for any unpaired interface as well
    for if_id, r in work.items():
        if if_id not in paired_ids and r.get('status') == 'down':
            r['rx'] = 0.0; r['tx'] = 0.0
            r['rx_var'] = (ABS_SIGMA * 0.1) ** 2
            r['tx_var'] = (ABS_SIGMA * 0.1) ** 2

    # Compute router imbalance for confidence
    router_final_imbalance: Dict[str, float] = {}
    for router, if_list in router_ifaces.items():
        up_ifaces = [i for i in if_list if i in work and work[i]['status'] == 'up']
        if not up_ifaces:
            router_final_imbalance[router] = 0.0
            continue
        sum_tx = sum(max(0.0, work[i]['tx']) for i in up_ifaces)
        sum_rx = sum(max(0.0, work[i]['rx']) for i in up_ifaces)
        router_final_imbalance[router] = rel_diff(sum_tx, sum_rx)

    # Assemble output with confidence calibration
    result: Dict[str, Dict[str, Tuple]] = {}

    # Component weights
    w_pair, w_router, w_status = 0.6, 0.3, 0.1

    for if_id, r in work.items():
        out: Dict[str, Tuple] = {}

        router = r.get('local_router')
        peer = peer_of.get(if_id)
        resolved_status = r.get('status', 'unknown')
        status_conf = clamp(r.get('status_conf', 0.8))

        # Pair component confidences (rate-aware tolerance)
        if peer and work.get(peer, {}).get('status') == resolved_status:
            # forward a.tx vs peer.rx
            res_fwd = rel_diff(r['tx'], work[peer]['rx'])
            traffic_tx = max(r['tx'], work[peer]['rx'], 1.0)
            tol_pair_tx = max(HARDENING_THRESHOLD, 5.0 / traffic_tx)
            pair_comp_tx = conf_from_residual_logistic(res_fwd, tol_pair_tx)

            # reverse a.rx vs peer.tx
            res_rev = rel_diff(r['rx'], work[peer]['tx'])
            traffic_rx = max(r['rx'], work[peer]['tx'], 1.0)
            tol_pair_rx = max(HARDENING_THRESHOLD, 5.0 / traffic_rx)
            pair_comp_rx = conf_from_residual_logistic(res_rev, tol_pair_rx)
        else:
            pair_comp_tx = 0.55
            pair_comp_rx = 0.55

        router_imb = router_final_imbalance.get(router, 0.0)
        router_comp = conf_from_residual_logistic(router_imb, HARDENING_THRESHOLD * 2.0)

        base_tx_conf = w_pair * pair_comp_tx + w_router * router_comp + w_status * status_conf
        base_rx_conf = w_pair * pair_comp_rx + w_router * router_comp + w_status * status_conf

        # Penalties: change magnitude and posterior variance
        def change_penalty(orig: float, new: float) -> float:
            d = rel_diff(orig, new)
            excess = max(0.0, d - HARDENING_THRESHOLD)
            return excess

        pen_tx = change_penalty(r['orig_tx'], r['tx'])
        pen_rx = change_penalty(r['orig_rx'], r['rx'])
        CHANGE_PENALTY_WEIGHT = 0.5

        # Posterior variance penalty relative to base variance
        var_factor_tx = r['tx_var'] / max(r['base_tx_var'], 1e-6)
        var_factor_rx = r['rx_var'] / max(r['base_rx_var'], 1e-6)
        # Map var_factor to [0,1] multiplier (~1 when reduced variance, <1 when large)
        var_mult_tx = 1.0 / (1.0 + 0.6 * var_factor_tx)
        var_mult_rx = 1.0 / (1.0 + 0.6 * var_factor_rx)

        tx_conf = clamp(base_tx_conf * (1.0 - CHANGE_PENALTY_WEIGHT * pen_tx) * var_mult_tx)
        rx_conf = clamp(base_rx_conf * (1.0 - CHANGE_PENALTY_WEIGHT * pen_rx) * var_mult_rx)

        if resolved_status == 'down':
            # If we forced to zero, confidence reflects whether it was near zero originally
            rx_conf = 0.9 if r['orig_rx'] <= TRAFFIC_EVIDENCE_MIN else 0.3
            tx_conf = 0.9 if r['orig_tx'] <= TRAFFIC_EVIDENCE_MIN else 0.3

        # If up but both directions effectively idle, reduce status confidence slightly
        if resolved_status == 'up' and r['rx'] <= TRAFFIC_EVIDENCE_MIN and r['tx'] <= TRAFFIC_EVIDENCE_MIN:
            status_conf = clamp(status_conf * 0.9)

        out['rx_rate'] = (r['orig_rx'], float(r['rx']), rx_conf)
        out['tx_rate'] = (r['orig_tx'], float(r['tx']), tx_conf)
        out['interface_status'] = (r['orig_status'], resolved_status, status_conf)
        out['connected_to'] = r['connected_to']
        out['local_router'] = r['local_router']
        out['remote_router'] = r['remote_router']

        result[if_id] = out

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
