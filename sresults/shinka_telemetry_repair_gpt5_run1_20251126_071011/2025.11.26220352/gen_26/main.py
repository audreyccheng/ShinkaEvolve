# EVOLVE-BLOCK-START
"""
Global weighted-projection consensus solver for network telemetry repair.

This implementation treats rx/tx rates as variables with Gaussian priors
(centered at the measurements). It alternates projections onto:
  - Link symmetry constraints (a.tx == b.rx, a.rx == b.tx)
  - Router flow conservation constraints (sum(tx) - sum(rx) == 0)

Projection updates are weighted by per-variable trust (from redundancy),
so unreliable counters are moved more. Status consistency is enforced
first; down interfaces are set to zero. Non-negativity is enforced.

Confidence is computed from post-repair residuals (pair + router), with
a change penalty and rate-aware tolerances for better calibration.
"""
from typing import Dict, Any, Tuple, List
from math import exp


def repair_network_telemetry(telemetry: Dict[str, Dict[str, Any]],
                             topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:
    # Core tolerances and knobs
    HARDENING_THRESHOLD = 0.02  # base timing jitter tolerance (~2%)
    TRAFFIC_EVIDENCE_MIN = 0.5  # Mbps; traffic evidence for status correction
    TOL_PAIR_MIN = 0.02         # min pair tolerance
    TOL_PAIR_MAX = 0.10         # max pair tolerance
    TOL_ROUTER = 0.04           # router imbalance tolerance (~2x base)
    # Iteration controls
    NUM_ITERS = 5               # alternating projection iterations
    ALPHA_PAIR = 0.75           # under-relaxation for pair updates
    ALPHA_ROUTER = 0.85         # under-relaxation for router updates
    EPS = 1e-9

    def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
        return max(lo, min(hi, x))

    def rel_diff(a: float, b: float) -> float:
        denom = max(abs(a), abs(b), 1e-9)
        return abs(a - b) / denom

    def rate_aware_pair_tol(v1: float, v2: float) -> float:
        traffic = max(abs(v1), abs(v2), 1.0)
        # relax tolerance on very low rates; cap at 10%
        return clamp(max(TOL_PAIR_MIN, 5.0 / traffic), TOL_PAIR_MIN, TOL_PAIR_MAX)

    def conf_logistic(residual: float, tol: float, k: float = 3.0) -> float:
        # Logistic decay: residual ≈ tol gives ~0.5 confidence
        tol = max(tol, 1e-9)
        x = residual / tol
        return clamp(1.0 / (1.0 + exp(k * (x - 1.0))))

    # 1) Prepare interim structure with originals
    interim: Dict[str, Dict[str, Any]] = {}
    for if_id, data in telemetry.items():
        interim[if_id] = {
            'rx': float(data.get('rx_rate', 0.0)),
            'tx': float(data.get('tx_rate', 0.0)),
            'orig_rx': float(data.get('rx_rate', 0.0)),
            'orig_tx': float(data.get('tx_rate', 0.0)),
            'status': data.get('interface_status', 'unknown'),
            'orig_status': data.get('interface_status', 'unknown'),
            'status_conf': 1.0,
            'rx_conf': 1.0,
            'tx_conf': 1.0,
            'connected_to': data.get('connected_to'),
            'local_router': data.get('local_router'),
            'remote_router': data.get('remote_router'),
        }

    # 2) Build connected pairs
    visited = set()
    pairs: List[Tuple[str, str]] = []
    for if_id, data in telemetry.items():
        peer = data.get('connected_to')
        if peer and peer in telemetry:
            key = tuple(sorted([if_id, peer]))
            if key not in visited:
                visited.add(key)
                pairs.append((key[0], key[1]))

    peer_of: Dict[str, str] = {}
    for a_id, b_id in pairs:
        peer_of[a_id] = b_id
        peer_of[b_id] = a_id

    # 3) Resolve status across links (signal collection + hardening)
    for a_id, b_id in pairs:
        a = interim[a_id]
        b = interim[b_id]
        a_stat = a['status']
        b_stat = b['status']
        a_rx, a_tx = a['rx'], a['tx']
        b_rx, b_tx = b['rx'], b['tx']
        max_traffic = max(a_rx, a_tx, b_rx, b_tx)

        if a_stat == b_stat:
            resolved_status = a_stat
            status_conf = 0.95 if resolved_status in ('up', 'down') else 0.7
        else:
            if max_traffic > TRAFFIC_EVIDENCE_MIN:
                resolved_status = 'up'
                status_conf = 0.85
            else:
                resolved_status = 'down'
                status_conf = 0.75

        a['status'] = resolved_status
        b['status'] = resolved_status
        a['status_conf'] = min(a['status_conf'], status_conf)
        b['status_conf'] = min(b['status_conf'], status_conf)

        if resolved_status == 'down':
            # enforce zero traffic with calibrated confidence
            for r, rx0, tx0 in [(a, a_rx, a_tx), (b, b_rx, b_tx)]:
                r['rx'] = 0.0
                r['tx'] = 0.0
                r['rx_conf'] = 0.9 if rx0 <= TRAFFIC_EVIDENCE_MIN else 0.3
                r['tx_conf'] = 0.9 if tx0 <= TRAFFIC_EVIDENCE_MIN else 0.3

    # Ensure unpaired down interfaces also set to zero
    for if_id, r in interim.items():
        if if_id not in peer_of and r.get('status') == 'down':
            rx0 = r['rx']; tx0 = r['tx']
            r['rx'] = 0.0; r['tx'] = 0.0
            r['rx_conf'] = 0.9 if rx0 <= TRAFFIC_EVIDENCE_MIN else 0.3
            r['tx_conf'] = 0.9 if tx0 <= TRAFFIC_EVIDENCE_MIN else 0.3

    # 4) Build router -> interfaces mapping
    if topology:
        router_ifaces: Dict[str, List[str]] = {r: [i for i in if_list if i in interim] for r, if_list in topology.items()}
    else:
        router_ifaces = {}
        for if_id, r in interim.items():
            rr = r.get('local_router')
            if rr is not None:
                router_ifaces.setdefault(rr, []).append(if_id)

    # Helper: compute pre-weights using redundancy (pair + router imbalance)
    def compute_router_imbalance_current() -> Dict[str, float]:
        imb: Dict[str, float] = {}
        for router, if_list in router_ifaces.items():
            up_ifaces = [i for i in if_list if interim[i]['status'] == 'up']
            if not up_ifaces:
                imb[router] = 0.0
                continue
            sum_tx = sum(max(0.0, interim[i]['tx']) for i in up_ifaces)
            sum_rx = sum(max(0.0, interim[i]['rx']) for i in up_ifaces)
            imb[router] = rel_diff(sum_tx, sum_rx)
        return imb

    router_imb0 = compute_router_imbalance_current()

    # Initialize variable values (x) and trust weights (w)
    # Variables: key is (if_id, 'tx') and (if_id, 'rx') for status up
    x: Dict[Tuple[str, str], float] = {}
    w: Dict[Tuple[str, str], float] = {}

    # Pair-based preliminary confidences and weights
    pair_dir_conf: Dict[Tuple[str, str], float] = {}
    for a_id, b_id in pairs:
        a = interim[a_id]; b = interim[b_id]
        if a['status'] != 'up' or b['status'] != 'up':
            continue
        # a.tx vs b.rx
        tol_tx = rate_aware_pair_tol(a['tx'], b['rx'])
        res_tx = rel_diff(a['tx'], b['rx'])
        conf_tx = conf_logistic(res_tx, tol_tx)
        pair_dir_conf[(a_id, 'tx')] = conf_tx
        pair_dir_conf[(b_id, 'rx')] = conf_tx
        # a.rx vs b.tx
        tol_rx = rate_aware_pair_tol(a['rx'], b['tx'])
        res_rx = rel_diff(a['rx'], b['tx'])
        conf_rx = conf_logistic(res_rx, tol_rx)
        pair_dir_conf[(a_id, 'rx')] = conf_rx
        pair_dir_conf[(b_id, 'tx')] = conf_rx

    for if_id, r in interim.items():
        # initialize x for all (including down, but we'll ignore down in constraints)
        x[(if_id, 'tx')] = max(0.0, float(r['tx']))
        x[(if_id, 'rx')] = max(0.0, float(r['rx']))

        # derive weights for up only
        for d in ('tx', 'rx'):
            key = (if_id, d)
            if r['status'] != 'up':
                # down variables won't be used in constraints, but keep finite weight
                w[key] = 2.0
                continue
            base_conf = 0.9 if r['status'] == 'up' else 0.6
            conf_pair = pair_dir_conf.get(key, 0.7)
            # combine: more weight to pair agreement
            comb_conf = 0.6 * conf_pair + 0.4 * base_conf
            weight = 0.2 + 1.8 * comb_conf  # map to [0.2, 2.0]
            # router imbalance factor (penalize reliability if router is inconsistent)
            rr = r.get('local_router')
            imb = router_imb0.get(rr, 0.0)
            weight *= clamp(1.0 - 0.2 * min(imb / max(TOL_ROUTER, 1e-9), 1.0), 0.5, 1.0)
            # protect near-zero traffic from aggressive edits
            if x[key] < 1.0:
                weight *= 1.3
            w[key] = clamp(weight, 0.2, 3.5)

    # 5) Build constraints lists (only for up interfaces)
    link_constraints: List[Tuple[Tuple[str, str], Tuple[str, str]]] = []
    for a_id, b_id in pairs:
        if interim[a_id]['status'] == 'up' and interim[b_id]['status'] == 'up':
            link_constraints.append(((a_id, 'tx'), (b_id, 'rx')))
            link_constraints.append(((a_id, 'rx'), (b_id, 'tx')))

    router_constraints: Dict[str, List[Tuple[Tuple[str, str], int]]] = {}
    # each constraint is a list of (var_key, sign) where sign=+1 for tx, -1 for rx
    for router, if_list in router_ifaces.items():
        vars_list: List[Tuple[Tuple[str, str], int]] = []
        for i in if_list:
            if interim[i]['status'] != 'up':
                continue
            vars_list.append(((i, 'tx'), +1))
            vars_list.append(((i, 'rx'), -1))
        if vars_list:
            router_constraints[router] = vars_list

    # Weighted projection helpers
    def project_pair_equal(k1: Tuple[str, str], k2: Tuple[str, str], alpha: float) -> None:
        # Enforce x[k1] == x[k2] by minimal weighted change
        v1, v2 = x[k1], x[k2]
        rdiff = v1 - v2
        if abs(rdiff) <= 0.0:
            return
        w1 = max(w[k1], EPS)
        w2 = max(w[k2], EPS)
        denom = (1.0 / w1) + (1.0 / w2)
        if denom <= EPS:
            return
        delta1 = - rdiff * (1.0 / w1) / denom
        delta2 = + rdiff * (1.0 / w2) / denom
        if abs(delta1) > 0.0 or abs(delta2) > 0.0:
            x[k1] = max(0.0, v1 + alpha * delta1)
            x[k2] = max(0.0, v2 + alpha * delta2)

    def project_router_flow(vars_signs: List[Tuple[Tuple[str, str], int]], alpha: float) -> None:
        # Enforce sum(sign * x) == 0 by minimal weighted change
        if not vars_signs:
            return
        residual = 0.0
        sum_inv = 0.0
        for k, s in vars_signs:
            residual += s * x[k]
            sum_inv += 1.0 / max(w[k], EPS)
        if abs(residual) <= 0.0 or sum_inv <= EPS:
            return
        factor = residual / sum_inv
        # delta_k = - factor * s / w_k
        for k, s in vars_signs:
            delta = - factor * s / max(w[k], EPS)
            if delta != 0.0:
                x[k] = max(0.0, x[k] + alpha * delta)

    # 6) Alternating projections
    for _ in range(NUM_ITERS):
        # Pair equalities
        for k1, k2 in link_constraints:
            # rate-aware relaxation: smaller alpha on tiny links
            v_scale = max(x[k1], x[k2], 1.0)
            tol_pair = clamp(max(TOL_PAIR_MIN, 5.0 / v_scale), TOL_PAIR_MIN, TOL_PAIR_MAX)
            alpha_pair = ALPHA_PAIR * (0.8 + 0.2 * conf_logistic(rel_diff(x[k1], x[k2]), tol_pair))
            project_pair_equal(k1, k2, alpha_pair)

        # Router flow conservation
        for router, vars_signs in router_constraints.items():
            # scale alpha by router imbalance heuristic
            sum_tx = sum(x[k] for k, s in vars_signs if s == +1)
            sum_rx = sum(x[k] for k, s in vars_signs if s == -1)
            imb = rel_diff(sum_tx, sum_rx)
            alpha_r = ALPHA_ROUTER * (0.7 + 0.3 * conf_logistic(imb, TOL_ROUTER))
            project_router_flow(vars_signs, alpha_r)

    # 7) Write back repaired values
    for if_id, r in interim.items():
        # Keep zeros for down; otherwise use x
        if r['status'] == 'up':
            r['tx'] = x[(if_id, 'tx')]
            r['rx'] = x[(if_id, 'rx')]
        else:
            r['tx'] = 0.0
            r['rx'] = 0.0

    # 8) Final confidence calibration based on post-repair invariants
    # Compute per-router imbalance residuals after repair
    router_final_imb: Dict[str, float] = {}
    for router, if_list in router_ifaces.items():
        up_ifaces = [i for i in if_list if interim[i]['status'] == 'up']
        if not up_ifaces:
            router_final_imb[router] = 0.0
            continue
        sum_tx = sum(interim[i]['tx'] for i in up_ifaces)
        sum_rx = sum(interim[i]['rx'] for i in up_ifaces)
        router_final_imb[router] = rel_diff(sum_tx, sum_rx)

    # Confidence composition weights
    w_pair_comp, w_router_comp, w_status_comp = 0.6, 0.3, 0.1

    for if_id, r in interim.items():
        resolved_status = r.get('status', 'unknown')
        status_conf = clamp(r.get('status_conf', 0.8))

        # Pair components
        peer = peer_of.get(if_id)
        if resolved_status == 'up' and peer and interim.get(peer, {}).get('status') == 'up':
            res_fwd = rel_diff(r['tx'], interim[peer]['rx'])
            tol_fwd = rate_aware_pair_tol(r['tx'], interim[peer]['rx'])
            pair_comp_tx = conf_logistic(res_fwd, tol_fwd)

            res_rev = rel_diff(r['rx'], interim[peer]['tx'])
            tol_rev = rate_aware_pair_tol(r['rx'], interim[peer]['tx'])
            pair_comp_rx = conf_logistic(res_rev, tol_rev)
        else:
            pair_comp_tx = 0.55
            pair_comp_rx = 0.55

        # Router component
        rr = r.get('local_router')
        router_comp = conf_logistic(router_final_imb.get(rr, 0.0), TOL_ROUTER)

        base_tx_conf = w_pair_comp * pair_comp_tx + w_router_comp * router_comp + w_status_comp * status_conf
        base_rx_conf = w_pair_comp * pair_comp_rx + w_router_comp * router_comp + w_status_comp * status_conf

        # Change penalty (larger edits -> lower confidence)
        d_tx = rel_diff(r['orig_tx'], r['tx'])
        d_rx = rel_diff(r['orig_rx'], r['rx'])
        pen_tx = max(0.0, d_tx - HARDENING_THRESHOLD)
        pen_rx = max(0.0, d_rx - HARDENING_THRESHOLD)
        CHANGE_PENALTY_WEIGHT = 0.55
        tx_conf = clamp(base_tx_conf * (1.0 - CHANGE_PENALTY_WEIGHT * pen_tx))
        rx_conf = clamp(base_rx_conf * (1.0 - CHANGE_PENALTY_WEIGHT * pen_rx))

        if resolved_status == 'down':
            rx_conf = 0.9 if r['orig_rx'] <= TRAFFIC_EVIDENCE_MIN else 0.3
            tx_conf = 0.9 if r['orig_tx'] <= TRAFFIC_EVIDENCE_MIN else 0.3

        # Idle-link slight penalty when up
        if resolved_status == 'up' and r['rx'] <= TRAFFIC_EVIDENCE_MIN and r['tx'] <= TRAFFIC_EVIDENCE_MIN:
            status_conf = clamp(status_conf * 0.9)

        r['tx_conf'] = tx_conf
        r['rx_conf'] = rx_conf
        r['status_conf'] = status_conf

    # 9) Assemble results: (original, repaired, confidence) + unchanged metadata
    result: Dict[str, Dict[str, Tuple]] = {}
    for if_id, r in interim.items():
        out: Dict[str, Tuple] = {}
        out['rx_rate'] = (r['orig_rx'], r['rx'], clamp(r['rx_conf']))
        out['tx_rate'] = (r['orig_tx'], r['tx'], clamp(r['tx_conf']))
        out['interface_status'] = (r['orig_status'], r['status'], clamp(r['status_conf']))
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