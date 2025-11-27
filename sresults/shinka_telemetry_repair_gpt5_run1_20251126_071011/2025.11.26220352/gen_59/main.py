# EVOLVE-BLOCK-START
"""
Least-squares factor-graph telemetry repair:
- Build a global quadratic objective from data fidelity, link symmetry, and router flow factors
- Solve the normal equations with conjugate gradient in a sparse structure
- Enforce non-negativity and "down implies zero"
- Calibrate confidence from invariant residuals, change sizes, and redundancy
"""
from typing import Dict, Any, Tuple, List
from math import isfinite

def repair_network_telemetry(telemetry: Dict[str, Dict[str, Any]],
                             topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:
    # Core tolerances and constants (from research)
    HARDENING_THRESHOLD = 0.02         # ≈2% timing tolerance
    TRAFFIC_EVIDENCE_MIN = 0.5         # Mbps: traffic implying link up
    TOL_ROUTER = HARDENING_THRESHOLD * 2.0
    EPS = 1e-9

    # Factor weights (relative strengths)
    W_DATA_BASE = 1.0                  # anchor to original telemetry
    W_DOWN_STRONG = 50.0               # strong zeroing for down interfaces
    W_PAIR_BASE = 4.0                  # pair equality strength
    W_ROUTER_BASE = 2.0                # router flow strength
    LAMBDA_RIDGE = 1e-6                # small ridge to ensure SPD

    def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
        if not isfinite(x):
            return lo
        return max(lo, min(hi, x))

    def rel_diff(a: float, b: float) -> float:
        denom = max(abs(a), abs(b), 1e-9)
        return abs(a - b) / denom

    # Map interfaces to indices (rx and tx separate variables)
    iface_ids = list(telemetry.keys())
    n = len(iface_ids)
    idx_rx: Dict[str, int] = {}
    idx_tx: Dict[str, int] = {}
    for k, if_id in enumerate(iface_ids):
        idx_rx[if_id] = 2 * k
        idx_tx[if_id] = 2 * k + 1
    Nvar = 2 * n

    # Sparse symmetric matrix H as dict of dicts, and vector g
    H: Dict[int, Dict[int, float]] = {}
    g: List[float] = [0.0] * Nvar

    def h_add(i: int, j: int, v: float):
        # Add v to H[i,j] and H[j,i] if i!=j; otherwise to diagonal
        if i not in H:
            H[i] = {}
        H[i][j] = H[i].get(j, 0.0) + v
        if i != j:
            if j not in H:
                H[j] = {}
            H[j][i] = H[j].get(i, 0.0) + v

    def add_data_anchor(vi: int, value: float, weight: float):
        # term: weight*(x_vi - value)^2 -> H[vi,vi] += weight; g[vi] += weight*value
        if weight <= 0.0:
            return
        h_add(vi, vi, weight)
        g[vi] += weight * value
        # redundancy count
        redundancy[vi] = redundancy.get(vi, 0) + 1

    def add_equality(i: int, j: int, weight: float):
        # term: weight*(x_i - x_j)^2 -> H[ii]+=w, H[jj]+=w, H[ij]-=w, H[ji]-=w
        if weight <= 0.0:
            return
        h_add(i, i, weight)
        h_add(j, j, weight)
        # for off-diagonals add -weight (we already maintain symmetry)
        if i not in H:
            H[i] = {}
        if j not in H:
            H[j] = {}
        H[i][j] = H[i].get(j, 0.0) - weight
        H[j][i] = H[j].get(i, 0.0) - weight
        # redundancy counts
        redundancy[i] = redundancy.get(i, 0) + 1
        redundancy[j] = redundancy.get(j, 0) + 1

    def add_sum_zero(indices: List[int], coeffs: List[float], weight: float):
        # term: weight*(sum_k coeffs[k]*x[indices[k]])^2
        # expands to weight * A^T A where A is the coefficient vector on these indices
        if weight <= 0.0 or not indices:
            return
        m = len(indices)
        for a in range(m):
            ia = indices[a]
            ca = coeffs[a]
            for b in range(m):
                ib = indices[b]
                cb = coeffs[b]
                h_add(ia, ib, weight * ca * cb)
        for ia in indices:
            redundancy[ia] = redundancy.get(ia, 0) + 1

    # Prepare topology per router; if not provided, derive from telemetry metadata
    router_ifaces: Dict[str, List[str]] = {}
    if topology:
        # Only include interfaces we have telemetry for
        router_ifaces = {r: [i for i in if_list if i in telemetry] for r, if_list in topology.items()}
    else:
        # Topology helps; derive best-effort from local_router metadata
        for if_id, data in telemetry.items():
            r = data.get('local_router')
            if r is not None:
                router_ifaces.setdefault(r, []).append(if_id)

    # Build peer mapping
    peer_of: Dict[str, str] = {}
    for if_id, data in telemetry.items():
        peer = data.get('connected_to')
        if peer and peer in telemetry:
            peer_of[if_id] = peer

    # Redundancy count per variable (used in confidence)
    redundancy: Dict[int, int] = {}

    # Add data anchors and down constraints
    x0: List[float] = [0.0] * Nvar
    status_resolved: Dict[str, str] = {}
    status_conf: Dict[str, float] = {}

    # First pass: determine link-up status by pair evidence
    # For unpaired, keep original status; for pairs with disagreement, use traffic proof
    # Build reverse pair map quickly
    for if_id in iface_ids:
        data = telemetry[if_id]
        peer = peer_of.get(if_id)
        my_stat = data.get('interface_status', 'unknown')
        # Default resolution is own status for unpaired
        if peer is None:
            status_resolved[if_id] = my_stat
            status_conf[if_id] = 0.9 if my_stat in ('up', 'down') else 0.7
            continue
        # Let the min tuple key handle once per pair
    visited_pairs = set()
    for if_id in iface_ids:
        peer = peer_of.get(if_id)
        if not peer:
            continue
        key = tuple(sorted([if_id, peer]))
        if key in visited_pairs:
            continue
        visited_pairs.add(key)
        a, b = key
        a_stat = telemetry[a].get('interface_status', 'unknown')
        b_stat = telemetry[b].get('interface_status', 'unknown')
        a_rx, a_tx = float(telemetry[a].get('rx_rate', 0.0)), float(telemetry[a].get('tx_rate', 0.0))
        b_rx, b_tx = float(telemetry[b].get('rx_rate', 0.0)), float(telemetry[b].get('tx_rate', 0.0))
        max_traffic = max(a_rx, a_tx, b_rx, b_tx)
        if a_stat == b_stat:
            resolved = a_stat
            conf = 0.95 if resolved in ('up', 'down') else 0.7
        else:
            if max_traffic > TRAFFIC_EVIDENCE_MIN:
                resolved = 'up'; conf = 0.85
            else:
                resolved = 'down'; conf = 0.75
        status_resolved[a] = resolved; status_resolved[b] = resolved
        status_conf[a] = conf; status_conf[b] = conf

    # Second pass: fill unpaired missing from previous
    for if_id in iface_ids:
        if if_id not in status_resolved:
            s = telemetry[if_id].get('interface_status', 'unknown')
            status_resolved[if_id] = s
            status_conf[if_id] = 0.9 if s in ('up', 'down') else 0.7

    # Add factors
    for if_id in iface_ids:
        rx0 = float(telemetry[if_id].get('rx_rate', 0.0))
        tx0 = float(telemetry[if_id].get('tx_rate', 0.0))
        irx = idx_rx[if_id]; itx = idx_tx[if_id]
        x0[irx] = rx0; x0[itx] = tx0

        # Data anchors (weight scaled by volume for relative robustness)
        # Use piecewise sigma: absolute floor 1 Mbps; relative via τh
        scale_rx = max(1.0, HARDENING_THRESHOLD * max(abs(rx0), 1.0))
        scale_tx = max(1.0, HARDENING_THRESHOLD * max(abs(tx0), 1.0))
        w_rx = W_DATA_BASE * (1.0 / scale_rx)
        w_tx = W_DATA_BASE * (1.0 / scale_tx)

        if status_resolved[if_id] == 'down':
            # Strong pull to zero for down interfaces
            add_data_anchor(irx, 0.0, W_DOWN_STRONG)
            add_data_anchor(itx, 0.0, W_DOWN_STRONG)
        else:
            add_data_anchor(irx, rx0, w_rx)
            add_data_anchor(itx, tx0, w_tx)

    # Pair symmetry factors, rate-aware weighting
    visited_pairs.clear()
    for if_id in iface_ids:
        peer = peer_of.get(if_id)
        if not peer:
            continue
        key = tuple(sorted([if_id, peer]))
        if key in visited_pairs:
            continue
        visited_pairs.add(key)
        a, b = key
        # Only enforce when link is up
        if not (status_resolved[a] == 'up' and status_resolved[b] == 'up'):
            continue
        # Forward: a.tx ≈ b.rx
        a_itx = idx_tx[a]; b_irx = idx_rx[b]
        # Reverse: a.rx ≈ b.tx
        a_irx = idx_rx[a]; b_itx = idx_tx[b]
        # Rate-aware tolerance: higher traffic => tighter; also bounded minimum tolerance
        a_tx0 = max(1.0, max(abs(x0[a_itx]), 1.0))
        b_rx0 = max(1.0, max(abs(x0[b_irx]), 1.0))
        a_rx0 = max(1.0, max(abs(x0[a_irx]), 1.0))
        b_tx0 = max(1.0, max(abs(x0[b_itx]), 1.0))
        tol_fwd = max(HARDENING_THRESHOLD, 5.0 / max(a_tx0, b_rx0))
        tol_rev = max(HARDENING_THRESHOLD, 5.0 / max(a_rx0, b_tx0))
        w_fwd = W_PAIR_BASE / max(tol_fwd, 1e-6)
        w_rev = W_PAIR_BASE / max(tol_rev, 1e-6)
        add_equality(a_itx, b_irx, w_fwd)
        add_equality(a_irx, b_itx, w_rev)

    # Router flow conservation factors
    for router, if_list in router_ifaces.items():
        # Only include interfaces that exist and are up
        up_ifaces = [i for i in if_list if i in telemetry and status_resolved.get(i, 'up') == 'up']
        if not up_ifaces:
            continue
        # Build indices and coefficients: sum(tx) - sum(rx) ≈ 0
        idxs: List[int] = []
        coeffs: List[float] = []
        for i in up_ifaces:
            idxs.append(idx_tx[i]); coeffs.append(1.0)
            idxs.append(idx_rx[i]); coeffs.append(-1.0)
        # Weight scaled modestly by router size to keep comparable influence
        scale = max(1.0, len(up_ifaces))
        w_router = W_ROUTER_BASE / scale
        add_sum_zero(idxs, coeffs, w_router)

    # Add ridge on diagonal to ensure SPD
    for vi in range(Nvar):
        h_add(vi, vi, LAMBDA_RIDGE)

    # Conjugate Gradient solver for Hx = g
    def matvec(vec: List[float]) -> List[float]:
        out = [0.0] * Nvar
        for i, row in H.items():
            s = 0.0
            # compute row dot vec
            acc = 0.0
            for j, v in row.items():
                acc += v * vec[j]
            out[i] = acc
        return out

    def dot(a: List[float], b: List[float]) -> float:
        return sum(ai * bi for ai, bi in zip(a, b))

    # Initialize with original vector x0
    x = x0[:]
    # r = g - Hx
    Hx = matvec(x)
    r = [g[i] - Hx[i] for i in range(Nvar)]
    p = r[:]
    rsold = dot(r, r)
    tol = 1e-8 * max(1.0, (rsold ** 0.5))
    max_iter = max(50, 5 * Nvar)  # modest iterations
    for _ in range(max_iter):
        Ap = matvec(p)
        denom = dot(p, Ap)
        if abs(denom) < 1e-18:
            break
        alpha = rsold / denom
        x = [xi + alpha * pi for xi, pi in zip(x, p)]
        r = [ri - alpha * api for ri, api in zip(r, Ap)]
        rsnew = dot(r, r)
        if (rsnew ** 0.5) < tol:
            break
        beta = rsnew / max(rsold, 1e-18)
        p = [ri + beta * pi for ri, pi in zip(r, p)]
        rsold = rsnew

    # Project non-negativity and enforce "down implies zero"
    repaired_rx: Dict[str, float] = {}
    repaired_tx: Dict[str, float] = {}
    for if_id in iface_ids:
        irx = idx_rx[if_id]; itx = idx_tx[if_id]
        rx_val = max(0.0, x[irx])
        tx_val = max(0.0, x[itx])
        if status_resolved[if_id] == 'down':
            rx_val = 0.0; tx_val = 0.0
        repaired_rx[if_id] = rx_val
        repaired_tx[if_id] = tx_val

    # Build confidence metrics
    # Per-router imbalance on repaired values
    router_final_imbalance: Dict[str, float] = {}
    for router, if_list in router_ifaces.items():
        up_ifaces = [i for i in if_list if i in telemetry and status_resolved.get(i, 'up') == 'up']
        if not up_ifaces:
            router_final_imbalance[router] = 0.0
            continue
        sum_tx = sum(max(0.0, repaired_tx[i]) for i in up_ifaces)
        sum_rx = sum(max(0.0, repaired_rx[i]) for i in up_ifaces)
        router_final_imbalance[router] = rel_diff(sum_tx, sum_rx)

    # Helper for pair residual-based confidence
    def pair_conf_for(if_id: str, direction: str) -> float:
        peer = peer_of.get(if_id)
        if not peer:
            return 0.55
        if not (status_resolved.get(if_id, 'up') == 'up' and status_resolved.get(peer, 'up') == 'up'):
            return 0.65
        if direction == 'tx':
            v1 = repaired_tx[if_id]; v2 = repaired_rx[peer]
            traffic = max(v1, v2, 1.0)
        else:
            v1 = repaired_rx[if_id]; v2 = repaired_tx[peer]
            traffic = max(v1, v2, 1.0)
        resid = rel_diff(v1, v2)
        tol_pair = min(0.12, max(HARDENING_THRESHOLD * 1.5, 5.0 / traffic))
        # Two-slope mapping
        xnorm = resid / max(tol_pair, 1e-9)
        conf = 1.0 - min(1.0, xnorm / 5.0)
        if xnorm > 3.0:
            conf -= 0.1 * (xnorm - 3.0) / 2.0
        return clamp(conf)

    # Assemble results with confidence calibration
    result: Dict[str, Dict[str, Tuple]] = {}
    for if_id in iface_ids:
        orig_rx = float(telemetry[if_id].get('rx_rate', 0.0))
        orig_tx = float(telemetry[if_id].get('tx_rate', 0.0))
        s_orig = telemetry[if_id].get('interface_status', 'unknown')
        s_res = status_resolved.get(if_id, s_orig)
        # Status confidence tweaks
        s_conf = status_conf.get(if_id, 0.8)
        if s_res == 'up':
            if repaired_rx[if_id] <= TRAFFIC_EVIDENCE_MIN and repaired_tx[if_id] <= TRAFFIC_EVIDENCE_MIN:
                s_conf = clamp(s_conf * 0.9)
        else:
            if repaired_rx[if_id] > TRAFFIC_EVIDENCE_MIN or repaired_tx[if_id] > TRAFFIC_EVIDENCE_MIN:
                s_conf = clamp(min(s_conf, 0.3))

        # Pair components
        pair_tx_conf = pair_conf_for(if_id, 'tx')
        pair_rx_conf = pair_conf_for(if_id, 'rx')

        # Router component
        router = telemetry[if_id].get('local_router')
        router_imb = router_final_imbalance.get(router, 0.0)
        tol_router = TOL_ROUTER
        xnorm_r = router_imb / max(tol_router, 1e-9)
        router_comp = 1.0 - min(1.0, xnorm_r / 5.0)
        if xnorm_r > 3.0:
            router_comp -= 0.1 * (xnorm_r - 3.0) / 2.0
        router_comp = clamp(router_comp)

        # Redundancy bonus based on number of factors touching variables
        vrx = idx_rx[if_id]; vtx = idx_tx[if_id]
        red_rx = redundancy.get(vrx, 1)
        red_tx = redundancy.get(vtx, 1)
        bonus_rx = clamp(0.01 * min(10, red_rx - 1), 0.0, 0.08)
        bonus_tx = clamp(0.01 * min(10, red_tx - 1), 0.0, 0.08)

        # Change penalties (avoid overconfidence on large edits)
        delta_rx_rel = rel_diff(orig_rx, repaired_rx[if_id])
        delta_tx_rel = rel_diff(orig_tx, repaired_tx[if_id])
        pen_rx = max(0.0, delta_rx_rel - HARDENING_THRESHOLD)
        pen_tx = max(0.0, delta_tx_rel - HARDENING_THRESHOLD)

        # Blend components
        w_pair, w_router, w_status = 0.6, 0.3, 0.1
        base_rx = w_pair * pair_rx_conf + w_router * router_comp + w_status * s_conf
        base_tx = w_pair * pair_tx_conf + w_router * router_comp + w_status * s_conf

        # Apply redundancy bonuses and change penalties
        conf_rx = clamp((base_rx + bonus_rx) * (1.0 - 0.5 * pen_rx))
        conf_tx = clamp((base_tx + bonus_tx) * (1.0 - 0.5 * pen_tx))

        # No-edit bonus
        if rel_diff(orig_rx, repaired_rx[if_id]) <= 1e-3:
            conf_rx = clamp(conf_rx + 0.05)
        if rel_diff(orig_tx, repaired_tx[if_id]) <= 1e-3:
            conf_tx = clamp(conf_tx + 0.05)

        # Down status confidence override
        if s_res == 'down':
            conf_rx = 0.9 if orig_rx <= TRAFFIC_EVIDENCE_MIN else 0.3
            conf_tx = 0.9 if orig_tx <= TRAFFIC_EVIDENCE_MIN else 0.3

        result[if_id] = {
            'rx_rate': (orig_rx, repaired_rx[if_id], conf_rx),
            'tx_rate': (orig_tx, repaired_tx[if_id], conf_tx),
            'interface_status': (s_orig, s_res, clamp(s_conf)),
            # Metadata unchanged
            'connected_to': telemetry[if_id].get('connected_to'),
            'local_router': telemetry[if_id].get('local_router'),
            'remote_router': telemetry[if_id].get('remote_router'),
        }

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