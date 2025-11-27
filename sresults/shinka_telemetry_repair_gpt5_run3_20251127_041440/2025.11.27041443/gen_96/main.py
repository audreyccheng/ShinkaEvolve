# EVOLVE-BLOCK-START
"""
Global POCS-based projection for network telemetry repair.

Approach:
- Formulate telemetry repair as a weighted least-squares projection onto convex
  sets encoding network invariants.
- Iteratively project current telemetry onto:
  1) Link symmetry sets: a.tx = b.rx and b.tx = a.rx (weighted averaging)
  2) Router conservation hyperplanes: Σrx = Σtx per router (weighted L2 projection)
- Enforce non-negativity and "down => zero" constraints throughout.
- Use share-aware and near-zero-aware weights so dominant links and tiny links
  are more stable; low-share links absorb more of the corrections safely.
- Confidence calibrated from: measurement deviation, final link residuals,
  and final router residuals, with light peer smoothing.

Maintains inputs/outputs of the original function signature.
"""
from typing import Dict, Any, Tuple, List
import math


def repair_network_telemetry(telemetry: Dict[str, Dict[str, Any]],
                             topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:
    # Hyperparameters
    TAU_H = 0.02            # ~2% hardening tolerance
    ZERO_EPS = 1e-9
    ZERO_THRESH = 1.0       # Mbps near-zero threshold
    N_ITER = 6              # number of POCS outer iterations
    LINK_ZERO_SNAP = 2.0    # soft-zero snap when link max < 2*ZERO_THRESH and routers balanced
    PEER_SMOOTH = 0.10      # peer confidence smoothing
    ROUTER_BAL_TAU_MIN = 0.03
    ROUTER_BAL_TAU_MAX = 0.07

    # Weighting parameters
    W_SHARE_K = 2.0         # added weight proportional to share within router side
    W_ZERO_K = 2.0          # extra weight when variable is near zero
    W_MIN = 0.15            # minimal weight to avoid division issues

    def clamp01(x: float) -> float:
        return max(0.0, min(1.0, x))

    def rel_diff(a: float, b: float) -> float:
        return abs(a - b) / max(1.0, abs(a), abs(b))

    def router_tau(n_active: int) -> float:
        # Adaptive router tolerance by active interface count
        return min(ROUTER_BAL_TAU_MAX, max(ROUTER_BAL_TAU_MIN, 0.05 * (2.0 / max(2, n_active)) ** 0.5))

    # Build basic maps
    orig_rx: Dict[str, float] = {}
    orig_tx: Dict[str, float] = {}
    status: Dict[str, str] = {}
    peer_of: Dict[str, str] = {}
    local_router_of: Dict[str, Any] = {}
    remote_router_of: Dict[str, Any] = {}

    for iid, d in telemetry.items():
        orig_rx[iid] = float(d.get('rx_rate', 0.0))
        orig_tx[iid] = float(d.get('tx_rate', 0.0))
        status[iid] = d.get('interface_status', 'unknown')
        ct = d.get('connected_to')
        peer_of[iid] = ct if ct in telemetry else None
        local_router_of[iid] = d.get('local_router')
        remote_router_of[iid] = d.get('remote_router')

    # Build router->interfaces mapping; prefer provided topology
    router_ifaces: Dict[str, List[str]] = {}
    if topology:
        for r, ifs in topology.items():
            router_ifaces[r] = [i for i in ifs if i in telemetry]
    else:
        # Fallback to local_router fields if topology not provided
        for iid, d in telemetry.items():
            r = d.get('local_router')
            if r is not None:
                router_ifaces.setdefault(r, []).append(iid)

    # Derive missing remote_router via peer's local_router if needed
    for iid in telemetry:
        if not remote_router_of.get(iid):
            p = peer_of.get(iid)
            if p and p in telemetry:
                remote_router_of[iid] = telemetry[p].get('local_router')

    # Build link pairs (unique, undirected)
    link_pairs: List[Tuple[str, str]] = []
    seen = set()
    for a in telemetry:
        b = peer_of.get(a)
        if not b or b not in telemetry or a == b:
            continue
        if (b, a) in seen or (a, b) in seen:
            continue
        seen.add((a, b))
        link_pairs.append((a, b))

    # Initialize working values
    rx: Dict[str, float] = {i: max(0.0, orig_rx[i]) for i in telemetry}
    tx: Dict[str, float] = {i: max(0.0, orig_tx[i]) for i in telemetry}

    # Enforce down interfaces -> zero
    for i in telemetry:
        if status.get(i) != 'up':
            rx[i] = 0.0
            tx[i] = 0.0

    # Quick helper: compute router sums
    def router_sums() -> Dict[str, Tuple[float, float, int]]:
        sums: Dict[str, Tuple[float, float, int]] = {}
        for r, ifs in router_ifaces.items():
            ups = [i for i in ifs if status.get(i) == 'up']
            srx = sum(rx[i] for i in ups)
            stx = sum(tx[i] for i in ups)
            sums[r] = (srx, stx, len(ups))
        return sums

    # Helper: compute share per interface for a given direction (rx or tx)
    def side_shares() -> Tuple[Dict[str, float], Dict[str, float]]:
        # tx_share[i]: share of tx[i] in local router's total TX; rx_share likewise
        tx_share: Dict[str, float] = {}
        rx_share: Dict[str, float] = {}
        # Precompute sums per router side
        sum_tx_r: Dict[str, float] = {}
        sum_rx_r: Dict[str, float] = {}
        for r, ifs in router_ifaces.items():
            ups = [i for i in ifs if status.get(i) == 'up']
            sum_tx_r[r] = sum(tx[i] for i in ups)
            sum_rx_r[r] = sum(rx[i] for i in ups)
        for i in telemetry:
            r = local_router_of.get(i)
            if r is None:
                tx_share[i] = 0.0
                rx_share[i] = 0.0
                continue
            st = sum_tx_r.get(r, 0.0)
            sr = sum_rx_r.get(r, 0.0)
            tx_share[i] = 0.0 if st <= ZERO_EPS else tx[i] / st
            rx_share[i] = 0.0 if sr <= ZERO_EPS else rx[i] / sr
        return tx_share, rx_share

    # Weighted averaging projection onto x = y constraint
    def project_pair_equal(x_val: float, y_val: float, wx: float, wy: float) -> float:
        wx = max(W_MIN, wx)
        wy = max(W_MIN, wy)
        return (wx * x_val + wy * y_val) / (wx + wy)

    # Weighted projection of router onto Σrx = Σtx hyperplane
    # Minimize Σ w_i * delta_i^2 s.t. Σ s_i * delta_i = -d, where s_i = +1 for rx, -1 for tx
    def project_router_balance(r: str) -> None:
        ifs = [i for i in router_ifaces.get(r, []) if status.get(i) == 'up']
        if len(ifs) < 2:
            return
        # Build variable lists
        z_vals: List[float] = []
        s_signs: List[int] = []
        weights: List[float] = []
        # Shares to bias weights
        tx_sh, rx_sh = shares_tx.get, shares_rx.get
        for i in ifs:
            # rx direction entry
            vi_rx = rx[i]
            si_rx = +1
            wi_rx = 1.0 + W_SHARE_K * max(0.0, rx_sh(i)) + (W_ZERO_K if vi_rx < 2.0 * ZERO_THRESH else 0.0)
            z_vals.append(vi_rx); s_signs.append(si_rx); weights.append(max(W_MIN, wi_rx))
            # tx direction entry
            vi_tx = tx[i]
            si_tx = -1
            wi_tx = 1.0 + W_SHARE_K * max(0.0, tx_sh(i)) + (W_ZERO_K if vi_tx < 2.0 * ZERO_THRESH else 0.0)
            z_vals.append(vi_tx); s_signs.append(si_tx); weights.append(max(W_MIN, wi_tx))

        # Compute imbalance d = Σ s_i * z_i
        d = 0.0
        for zi, si in zip(z_vals, s_signs):
            d += si * zi

        # Skip if within tolerance
        srx, stx, nup = router_totals.get(r, (0.0, 0.0, 0))
        denom = max(1.0, srx, stx)
        if abs(d) / denom <= router_tau(nup):
            return

        # Weighted solution: delta_i = -d * s_i / w_i / Σ (1/w_j)
        invw_sum = sum(1.0 / wi for wi in weights)
        if invw_sum <= ZERO_EPS:
            return
        deltas = [(-d) * si / wi / invw_sum for si, wi in zip(s_signs, weights)]

        # Determine maximal step to keep variables non-negative (damped if needed)
        lam_max = 1.0
        for zi, di in zip(z_vals, deltas):
            if di < 0.0:
                lam_max = min(lam_max, (zi / (-di)) if (-di) > ZERO_EPS else lam_max)
        lam = max(0.0, min(1.0, lam_max))

        # Apply updates
        idx = 0
        for i in ifs:
            d_rx = lam * deltas[idx]; d_tx = lam * deltas[idx + 1]
            rx[i] = max(0.0, rx[i] + d_rx)
            tx[i] = max(0.0, tx[i] + d_tx)
            idx += 2

    # POCS outer iterations
    for _ in range(N_ITER):
        # Refresh router totals and shares
        router_totals = router_sums()
        shares_tx, shares_rx = side_shares()

        # 1) Project onto link symmetry for each connected pair (two directions)
        for a, b in link_pairs:
            a_up = (status.get(a) == 'up')
            b_up = (status.get(b) == 'up')

            if not a_up or not b_up:
                # Down -> zero already enforced; force both sides of the link to zero
                rx[a] = tx[a] = 0.0
                rx[b] = tx[b] = 0.0
                continue

            # Soft-zero stabilization for near-zero links
            if max(rx[a], tx[a], rx[b], tx[b]) < LINK_ZERO_SNAP * ZERO_THRESH:
                rx[a] = tx[a] = 0.0
                rx[b] = tx[b] = 0.0
                continue

            # Direction 1: a.tx == b.rx
            x = tx[a]; y = rx[b]
            # Weights prefer large-share and non-near-zero to remain more stable
            wx = 1.0 + W_SHARE_K * max(0.0, shares_tx.get(a, 0.0)) + (W_ZERO_K if x < 2.0 * ZERO_THRESH else 0.0)
            wy = 1.0 + W_SHARE_K * max(0.0, shares_rx.get(b, 0.0)) + (W_ZERO_K if y < 2.0 * ZERO_THRESH else 0.0)
            v = project_pair_equal(x, y, wx, wy)
            tx[a] = max(0.0, v)
            rx[b] = max(0.0, v)

            # Direction 2: a.rx == b.tx
            x2 = rx[a]; y2 = tx[b]
            wx2 = 1.0 + W_SHARE_K * max(0.0, shares_rx.get(a, 0.0)) + (W_ZERO_K if x2 < 2.0 * ZERO_THRESH else 0.0)
            wy2 = 1.0 + W_SHARE_K * max(0.0, shares_tx.get(b, 0.0)) + (W_ZERO_K if y2 < 2.0 * ZERO_THRESH else 0.0)
            v2 = project_pair_equal(x2, y2, wx2, wy2)
            rx[a] = max(0.0, v2)
            tx[b] = max(0.0, v2)

        # 2) Project per-router onto Σrx = Σtx hyperplane (weighted, non-negativity aware)
        router_totals = router_sums()
        shares_tx, shares_rx = side_shares()
        for r in router_ifaces.keys():
            project_router_balance(r)

        # 3) Re-apply hard down constraint
        for i in telemetry:
            if status.get(i) != 'up':
                rx[i] = 0.0
                tx[i] = 0.0

    # Final soft-zero snap for tiny links when both adjacent routers are balanced
    router_totals = router_sums()
    for a, b in link_pairs:
        if status.get(a) != 'up' or status.get(b) != 'up':
            rx[a] = tx[a] = 0.0
            rx[b] = tx[b] = 0.0
            continue

        ra = local_router_of.get(a)
        rb = local_router_of.get(b)
        srx_a, stx_a, na = router_totals.get(ra, (0.0, 0.0, 0))
        srx_b, stx_b, nb = router_totals.get(rb, (0.0, 0.0, 0))
        bal_a = abs(srx_a - stx_a) / max(1.0, srx_a, stx_a) <= router_tau(na)
        bal_b = abs(srx_b - stx_b) / max(1.0, srx_b, stx_b) <= router_tau(nb)

        if bal_a and bal_b and max(rx[a], tx[a], rx[b], tx[b]) < LINK_ZERO_SNAP * ZERO_THRESH:
            rx[a] = tx[a] = 0.0
            rx[b] = tx[b] = 0.0

    # Confidence computation
    # Router residuals after all adjustments
    router_residual: Dict[str, float] = {}
    for r, (srx, stx, nup) in router_totals.items():
        denom = max(1.0, srx, stx)
        router_residual[r] = abs(srx - stx) / denom if denom > 0 else 0.0

    # Base confidences per direction from three terms:
    #  - measurement change
    #  - link residual (vs. peer opposite dir)
    #  - router residual
    conf_rx: Dict[str, float] = {}
    conf_tx: Dict[str, float] = {}

    for i in telemetry:
        # Measurement residuals
        r_meas_rx = rel_diff(rx[i], orig_rx[i])
        r_meas_tx = rel_diff(tx[i], orig_tx[i])

        # Link residuals
        p = peer_of.get(i)
        if p and p in telemetry and status.get(i) == 'up' and status.get(p) == 'up':
            r_link_rx = rel_diff(rx[i], tx[p])  # my rx vs peer tx
            r_link_tx = rel_diff(tx[i], rx[p])  # my tx vs peer rx
        else:
            r_link_rx = 0.2
            r_link_tx = 0.2

        # Router residual
        rtr = router_residual.get(local_router_of.get(i), 0.0)

        # Confidence as 1 − weighted residual blend
        c_rx = 1.0 - (0.52 * r_meas_rx + 0.33 * r_link_rx + 0.15 * rtr)
        c_tx = 1.0 - (0.52 * r_meas_tx + 0.33 * r_link_tx + 0.15 * rtr)
        # Enforce floors/ceilings
        conf_rx[i] = clamp01(c_rx)
        conf_tx[i] = clamp01(c_tx)

        # If interface is down, zero is a strong invariant; raise confidence floor
        if status.get(i) != 'up':
            conf_rx[i] = max(conf_rx[i], 0.85)
            conf_tx[i] = max(conf_tx[i], 0.85)

    # Peer smoothing for confidences (order-independent staging)
    new_conf_rx = dict(conf_rx)
    new_conf_tx = dict(conf_tx)
    for a, b in link_pairs:
        if status.get(a) == 'up' and status.get(b) == 'up':
            new_conf_tx[a] = clamp01((1.0 - PEER_SMOOTH) * conf_tx[a] + PEER_SMOOTH * conf_rx[b])
            new_conf_rx[b] = clamp01((1.0 - PEER_SMOOTH) * conf_rx[b] + PEER_SMOOTH * conf_tx[a])

            new_conf_rx[a] = clamp01((1.0 - PEER_SMOOTH) * conf_rx[a] + PEER_SMOOTH * conf_tx[b])
            new_conf_tx[b] = clamp01((1.0 - PEER_SMOOTH) * conf_tx[b] + PEER_SMOOTH * conf_rx[a])
    conf_rx = new_conf_rx
    conf_tx = new_conf_tx

    # Assemble final result in required tuple format
    result: Dict[str, Dict[str, Tuple]] = {}
    for i, data in telemetry.items():
        my_status = status.get(i, 'unknown')
        peer_id = peer_of.get(i)
        # Status confidence: do not flip, penalize inconsistent peer or down+nonzero readings
        status_conf = 1.0
        if peer_id and peer_id in telemetry:
            if telemetry[peer_id].get('interface_status', 'unknown') != my_status:
                status_conf = 0.6
        if my_status == 'down' and (orig_rx.get(i, 0.0) > ZERO_EPS or orig_tx.get(i, 0.0) > ZERO_EPS):
            status_conf = min(status_conf, 0.6)

        repaired: Dict[str, Any] = {}
        repaired['rx_rate'] = (orig_rx.get(i, 0.0), max(0.0, rx.get(i, 0.0)), clamp01(conf_rx.get(i, 0.6)))
        repaired['tx_rate'] = (orig_tx.get(i, 0.0), max(0.0, tx.get(i, 0.0)), clamp01(conf_tx.get(i, 0.6)))
        repaired['interface_status'] = (my_status, my_status, status_conf)

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