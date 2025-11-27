# EVOLVE-BLOCK-START
"""
Global flow balancer for repairing network telemetry.

Fundamental shift: treat each connected pair's two directions as shared variables
and minimize global router imbalances with regularization, enforcing link symmetry
exactly while allowing uncertain links to move more to satisfy flow conservation.
"""
from typing import Dict, Any, Tuple, List


def repair_network_telemetry(telemetry: Dict[str, Dict[str, Any]],
                             topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:
    """
    Inputs:
      telemetry: interface_id -> {
          interface_status: "up"|"down",
          rx_rate: float Mbps,
          tx_rate: float Mbps,
          connected_to: peer interface_id (optional),
          local_router: router_id,
          remote_router: router_id
      }
      topology: router_id -> [interface_ids]

    Output:
      interface_id -> {
        rx_rate: (orig, repaired, confidence),
        tx_rate: (orig, repaired, confidence),
        interface_status: (orig, repaired, confidence),
        connected_to/local_router/remote_router: unchanged
      }
    """

    # Core tolerances and caps
    HARDENING_THRESHOLD = 0.02          # ~2% timing tolerance
    TRAFFIC_EVIDENCE_MIN = 0.5          # Mbps, to infer link "up" when statuses disagree
    TOTAL_CAP_FRAC = 0.35               # ±35% max total change cap per variable
    PER_ITER_CAP_FRAC = 0.15            # ±15% per-iteration relative change cap
    EPS = 1e-9

    # Confidence blend weights
    W_PAIR, W_ROUTER, W_STATUS = 0.6, 0.3, 0.1

    def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
        return max(lo, min(hi, x))

    def rel_diff(a: float, b: float) -> float:
        denom = max(abs(a), abs(b), 1e-9)
        return abs(a - b) / denom

    def conf_from_residual(residual: float, tol: float) -> float:
        # Map residual to confidence: 1 at 0 residual, degrades linearly until near 0 at 5*tol
        denom = max(tol * 5.0, 1e-9)
        return clamp(1.0 - residual / denom)

    # Build connected pairs
    visited = set()
    pairs: List[Tuple[str, str]] = []
    for if_id, data in telemetry.items():
        peer = data.get('connected_to')
        if peer and peer in telemetry:
            key = tuple(sorted([if_id, peer]))
            if key not in visited:
                visited.add(key)
                pairs.append((key[0], key[1]))

    # Peer map and paired ids
    peer_of: Dict[str, str] = {}
    paired_ids = set()
    for a_id, b_id in pairs:
        peer_of[a_id] = b_id
        peer_of[b_id] = a_id
        paired_ids.add(a_id)
        paired_ids.add(b_id)

    # Build router->interfaces mapping using topology if available; derive otherwise
    router_ifaces: Dict[str, List[str]] = {}
    if topology:
        router_ifaces = {r: [i for i in lst if i in telemetry] for r, lst in topology.items()}
    else:
        for if_id, data in telemetry.items():
            r = data.get('local_router')
            if r is not None:
                router_ifaces.setdefault(r, []).append(if_id)

    # Interim per-interface store (status resolution + originals)
    interim_if: Dict[str, Dict[str, Any]] = {}
    for if_id, data in telemetry.items():
        rx0 = float(data.get('rx_rate', 0.0))
        tx0 = float(data.get('tx_rate', 0.0))
        st0 = data.get('interface_status', 'unknown')
        interim_if[if_id] = {
            'orig_rx': rx0, 'orig_tx': tx0, 'orig_status': st0,
            'rx': rx0, 'tx': tx0, 'status': st0,
            'rx_conf': 1.0, 'tx_conf': 1.0, 'status_conf': 1.0,
            'connected_to': data.get('connected_to'),
            'local_router': data.get('local_router'),
            'remote_router': data.get('remote_router'),
            'cap_hit_rx': False, 'cap_hit_tx': False,
        }

    # Pair-level status resolution and initial hardening (compute base values and base confidences)
    for a_id, b_id in pairs:
        a = telemetry[a_id]; b = telemetry[b_id]
        a_stat = a.get('interface_status', 'unknown')
        b_stat = b.get('interface_status', 'unknown')
        a_rx, a_tx = float(a.get('rx_rate', 0.0)), float(a.get('tx_rate', 0.0))
        b_rx, b_tx = float(b.get('rx_rate', 0.0)), float(b.get('tx_rate', 0.0))
        max_traffic = max(a_rx, a_tx, b_rx, b_tx)

        if a_stat == b_stat:
            resolved_status = a_stat
            status_conf = 0.95 if resolved_status in ('up', 'down') else 0.7
        else:
            if max_traffic > TRAFFIC_EVIDENCE_MIN:
                resolved_status = 'up'; status_conf = 0.85
            else:
                resolved_status = 'down'; status_conf = 0.75

        for ifid in (a_id, b_id):
            interim_if[ifid]['status'] = resolved_status
            interim_if[ifid]['status_conf'] = min(interim_if[ifid]['status_conf'], status_conf) if interim_if[ifid]['status_conf'] else status_conf

        if resolved_status == 'down':
            for (ifid, rx0i, tx0i) in [(a_id, a_rx, a_tx), (b_id, b_rx, b_tx)]:
                interim_if[ifid]['rx'] = 0.0
                interim_if[ifid]['tx'] = 0.0
                interim_if[ifid]['rx_conf'] = 0.9 if rx0i <= TRAFFIC_EVIDENCE_MIN else 0.3
                interim_if[ifid]['tx_conf'] = 0.9 if tx0i <= TRAFFIC_EVIDENCE_MIN else 0.3
            continue

        # Pair hardening values (we will later overwrite via global optimizer)
        # Forward: a.tx ↔ b.rx, Reverse: a.rx ↔ b.tx
        d_fwd = rel_diff(a_tx, b_rx)
        d_rev = rel_diff(a_rx, b_tx)
        v_fwd = 0.5 * (a_tx + b_rx) if (a_tx > 0 or b_rx > 0) else 0.0
        v_rev = 0.5 * (a_rx + b_tx) if (a_rx > 0 or b_tx > 0) else 0.0

        conf_fwd = clamp(1.0 - d_fwd) if d_fwd > HARDENING_THRESHOLD else clamp(1.0 - 0.5 * d_fwd)
        conf_rev = clamp(1.0 - d_rev) if d_rev > HARDENING_THRESHOLD else clamp(1.0 - 0.5 * d_rev)
        # Store provisional (will be replaced by global solution)
        interim_if[a_id]['tx'] = v_fwd
        interim_if[b_id]['rx'] = v_fwd
        interim_if[a_id]['tx_conf'] = min(interim_if[a_id]['tx_conf'], conf_fwd)
        interim_if[b_id]['rx_conf'] = min(interim_if[b_id]['rx_conf'], conf_fwd)

        interim_if[a_id]['rx'] = v_rev
        interim_if[b_id]['tx'] = v_rev
        interim_if[a_id]['rx_conf'] = min(interim_if[a_id]['rx_conf'], conf_rev)
        interim_if[b_id]['tx_conf'] = min(interim_if[b_id]['tx_conf'], conf_rev)

    # Down implies zero for unpaired interfaces as well
    for if_id, r in interim_if.items():
        if if_id not in paired_ids and r.get('status') == 'down':
            rx0 = r['rx']; tx0 = r['tx']
            r['rx'] = 0.0; r['tx'] = 0.0
            r['rx_conf'] = 0.9 if rx0 <= TRAFFIC_EVIDENCE_MIN else 0.3
            r['tx_conf'] = 0.9 if tx0 <= TRAFFIC_EVIDENCE_MIN else 0.3

    # Build directed "arc" variables:
    # For each connected pair (up): two arcs y_ab and y_ba.
    # For each unpaired (up): two arcs y_i_tx (src=router) and y_i_rx (dst=router).
    # Each arc has: src_router, dst_router, y (current), y0 (init), mu (regularization), cap_abs, cap_hit flag.
    arcs: Dict[str, Dict[str, Any]] = {}
    # Also map arcs to interface metric fields for final assignment
    arc_to_interface: Dict[str, Tuple[str, str]] = {}  # arc_id -> (iface_id, 'tx'|'rx')

    # Helper to get router of iface
    def router_of(iface_id: str) -> Any:
        return telemetry.get(iface_id, {}).get('local_router')

    # Create arcs for pairs (if up)
    for a_id, b_id in pairs:
        if interim_if[a_id]['status'] != 'up' or interim_if[b_id]['status'] != 'up':
            continue
        a_r = router_of(a_id)
        b_r = router_of(b_id)
        # forward arc: a -> b
        a_tx0 = float(telemetry[a_id].get('tx_rate', 0.0))
        b_rx0 = float(telemetry[b_id].get('rx_rate', 0.0))
        d_fwd = rel_diff(a_tx0, b_rx0)
        v_fwd0 = 0.5 * (a_tx0 + b_rx0) if (a_tx0 > 0 or b_rx0 > 0) else 0.0
        conf_fwd = clamp(1.0 - d_fwd) if d_fwd > HARDENING_THRESHOLD else clamp(1.0 - 0.5 * d_fwd)
        mu_fwd = 0.3 * (0.5 + 0.5 * conf_fwd)  # stronger regularization for higher agreement
        arc_id_fwd = f"{a_id}__to__{b_id}"
        arcs[arc_id_fwd] = {
            'src': a_r, 'dst': b_r, 'y': max(0.0, v_fwd0), 'y0': max(0.0, v_fwd0),
            'mu': mu_fwd, 'cap': TOTAL_CAP_FRAC, 'cap_hit': False
        }
        arc_to_interface[arc_id_fwd] = (a_id, 'tx')  # also implies b_id's rx

        # reverse arc: b -> a
        a_rx0 = float(telemetry[a_id].get('rx_rate', 0.0))
        b_tx0 = float(telemetry[b_id].get('tx_rate', 0.0))
        d_rev = rel_diff(a_rx0, b_tx0)
        v_rev0 = 0.5 * (a_rx0 + b_tx0) if (a_rx0 > 0 or b_tx0 > 0) else 0.0
        conf_rev = clamp(1.0 - d_rev) if d_rev > HARDENING_THRESHOLD else clamp(1.0 - 0.5 * d_rev)
        mu_rev = 0.3 * (0.5 + 0.5 * conf_rev)
        arc_id_rev = f"{b_id}__to__{a_id}"
        arcs[arc_id_rev] = {
            'src': b_r, 'dst': a_r, 'y': max(0.0, v_rev0), 'y0': max(0.0, v_rev0),
            'mu': mu_rev, 'cap': TOTAL_CAP_FRAC, 'cap_hit': False
        }
        arc_to_interface[arc_id_rev] = (b_id, 'tx')  # also implies a_id's rx

    # Unpaired: create tx and rx arcs if up
    for if_id, r in interim_if.items():
        if if_id in paired_ids:
            continue
        if r.get('status') != 'up':
            continue
        rtr = router_of(if_id)
        tx0 = float(telemetry[if_id].get('tx_rate', 0.0))
        rx0 = float(telemetry[if_id].get('rx_rate', 0.0))
        # With no peer, trust more the original (stronger mu)
        mu_u = 0.4
        # tx arc (source at local router)
        arc_id_tx = f"{if_id}__TX"
        arcs[arc_id_tx] = {
            'src': rtr, 'dst': None, 'y': max(0.0, tx0), 'y0': max(0.0, tx0),
            'mu': mu_u, 'cap': TOTAL_CAP_FRAC, 'cap_hit': False
        }
        arc_to_interface[arc_id_tx] = (if_id, 'tx')
        # rx arc (destination at local router)
        arc_id_rx = f"{if_id}__RX"
        arcs[arc_id_rx] = {
            'src': None, 'dst': rtr, 'y': max(0.0, rx0), 'y0': max(0.0, rx0),
            'mu': mu_u, 'cap': TOTAL_CAP_FRAC, 'cap_hit': False
        }
        arc_to_interface[arc_id_rx] = (if_id, 'rx')

    # Prepare router list from telemetry/topology
    routers = set(router_ifaces.keys())
    for if_id, data in telemetry.items():
        rtr = data.get('local_router')
        if rtr is not None:
            routers.add(rtr)

    # Global gradient descent to minimize sum_r (Out[r]-In[r])^2 + sum_arcs mu*(y - y0)^2
    def compute_router_sums(arcs_dict: Dict[str, Dict[str, Any]]):
        out_sum = {r: 0.0 for r in routers}
        in_sum = {r: 0.0 for r in routers}
        for a in arcs_dict.values():
            v = max(0.0, a['y'])
            if a['src'] is not None:
                out_sum[a['src']] = out_sum.get(a['src'], 0.0) + v
            if a['dst'] is not None:
                in_sum[a['dst']] = in_sum.get(a['dst'], 0.0) + v
        return out_sum, in_sum

    def max_rel_router_imbalance(out_sum: Dict[Any, float], in_sum: Dict[Any, float]) -> float:
        worst = 0.0
        for r in routers:
            o = out_sum.get(r, 0.0); i = in_sum.get(r, 0.0)
            rd = rel_diff(o, i)
            if rd > worst:
                worst = rd
        return worst

    # Iterative updates
    max_iters = 25
    for it in range(max_iters):
        out_sum, in_sum = compute_router_sums(arcs)
        # Check convergence on relative imbalance
        worst_rel = max_rel_router_imbalance(out_sum, in_sum)
        if worst_rel <= HARDENING_THRESHOLD * 2.0:
            break

        # Precompute router residuals C[r] = Out - In
        C = {r: out_sum.get(r, 0.0) - in_sum.get(r, 0.0) for r in routers}

        # Update each arc
        for arc_id, a in arcs.items():
            y = a['y']; y0 = a['y0']; mu = a['mu']
            src = a['src']; dst = a['dst']
            # Gradient: dJ/dy = 2*C[src] - 2*C[dst] + 2*mu*(y - y0) (missing terms if None)
            grad = 0.0
            if src is not None:
                grad += 2.0 * C.get(src, 0.0)
            if dst is not None:
                grad -= 2.0 * C.get(dst, 0.0)
            grad += 2.0 * mu * (y - y0)
            # Adaptive step: normalize by scale
            denom = (2.0 * (1.0 if src is not None else 0.0) +
                     2.0 * (1.0 if dst is not None else 0.0) +
                     2.0 * mu + 1e-6)
            eta = 0.5 / denom
            delta = -eta * grad

            # Per-iteration cap (relative ±15% and absolute on small values)
            rel_cap = PER_ITER_CAP_FRAC * max(y, 1.0)
            delta = max(-rel_cap, min(rel_cap, delta))

            # Apply and enforce non-negativity
            new_y = max(0.0, y + delta)

            # Total cap relative to y0 (±35%)
            y_hi = (1.0 + a['cap']) * max(y0, 1.0)
            y_lo = max(0.0, (1.0 - a['cap']) * max(y0, 1.0))
            clipped = False
            if new_y > y_hi:
                new_y = y_hi; clipped = True
            if new_y < y_lo:
                new_y = y_lo; clipped = True
            if clipped:
                a['cap_hit'] = True

            a['y'] = new_y

    # Final router residuals for confidence
    out_sum_final, in_sum_final = compute_router_sums(arcs)
    router_final_imbalance: Dict[Any, float] = {}
    for r in routers:
        o = out_sum_final.get(r, 0.0); i = in_sum_final.get(r, 0.0)
        router_final_imbalance[r] = rel_diff(o, i)

    # Assign repaired values back to interfaces from arcs
    # Initialize with current interim values (already zeroed for down)
    for arc_id, a in arcs.items():
        y = max(0.0, a['y'])
        iface_id, kind = arc_to_interface[arc_id]
        if kind == 'tx':
            interim_if[iface_id]['tx'] = y
        else:
            interim_if[iface_id]['rx'] = y
        # For the opposite side of a pair arc, we also need to set peer rx if applicable
        if "__to__" in arc_id:
            # arc_id pattern: "{src_if}__to__{dst_if}"
            src_if, dst_if = arc_id.split("__to__")
            src_if = src_if
            dst_if = dst_if
            # fwd arc represents src_if.tx and dst_if.rx
            interim_if[dst_if]['rx'] = y

    # For pair reverse arcs, similarly ensure opposite side tx is set (already handled via mapping)
    # Note: The mapping above already set both ends via arc_to_interface assignments.

    # Confidence calibration per interface
    TOL_PAIR_BASE = HARDENING_THRESHOLD * 1.5
    TOL_ROUTER = HARDENING_THRESHOLD * 2.0

    def finalize_conf(base: float, orig_val: float, new_val: float, cap_hit: bool) -> float:
        delta_rel = rel_diff(orig_val, new_val)
        pen = max(0.0, delta_rel - HARDENING_THRESHOLD)
        CHANGE_PENALTY_WEIGHT = 0.5
        conf = clamp(base * (1.0 - CHANGE_PENALTY_WEIGHT * pen))
        if cap_hit and delta_rel > 0.25:
            conf = max(0.0, conf - 0.05)
        # No-edit bonus
        if rel_diff(orig_val, new_val) <= 1e-3:
            conf = clamp(conf + 0.05)
        return clamp(conf)

    # Build result and compute confidences
    result: Dict[str, Dict[str, Tuple]] = {}

    for if_id, r in interim_if.items():
        # Re-resolve status evidence if needed for unpaired (already done); keep as is
        status_resolved = r.get('status', 'unknown')
        status_conf = r.get('status_conf', 0.8)

        # Pair residual components: after repair, for paired interfaces pair residual is zero.
        peer = peer_of.get(if_id)
        if peer and interim_if.get(peer, {}).get('status') == status_resolved and status_resolved == 'up':
            res_tx = rel_diff(r['tx'], interim_if[peer]['rx'])
            res_rx = rel_diff(r['rx'], interim_if[peer]['tx'])
            # Rate-aware tolerances to avoid punishing low-rate interfaces
            traffic_tx = max(r['tx'], interim_if[peer]['rx'], 1.0)
            traffic_rx = max(r['rx'], interim_if[peer]['tx'], 1.0)
            tol_pair_tx = min(0.12, max(TOL_PAIR_BASE, 5.0 / traffic_tx))
            tol_pair_rx = min(0.12, max(TOL_PAIR_BASE, 5.0 / traffic_rx))
            pair_comp_tx = conf_from_residual(res_tx, tol_pair_tx)
            pair_comp_rx = conf_from_residual(res_rx, tol_pair_rx)
        else:
            # Unpaired or down; rely less on pair component
            pair_comp_tx = 0.55
            pair_comp_rx = 0.55

        router = r.get('local_router')
        router_imb = router_final_imbalance.get(router, 0.0)
        router_comp = conf_from_residual(router_imb, TOL_ROUTER)

        base_tx_conf = W_PAIR * pair_comp_tx + W_ROUTER * router_comp + W_STATUS * status_conf
        base_rx_conf = W_PAIR * pair_comp_rx + W_ROUTER * router_comp + W_STATUS * status_conf

        # Determine cap hits from arcs (if exists)
        cap_hit_tx = False
        cap_hit_rx = False
        if if_id in paired_ids and status_resolved == 'up':
            # paired: find arcs that set our tx and rx
            # tx arc id is "{if_id}__to__{peer}"
            arc_tx_id = f"{if_id}__to__{peer}" if f"{if_id}__to__{peer}" in arcs else None
            arc_rx_id = f"{peer}__to__{if_id}" if f"{peer}__to__{if_id}" in arcs else None
            if arc_tx_id:
                cap_hit_tx = arcs[arc_tx_id]['cap_hit']
            if arc_rx_id:
                cap_hit_rx = arcs[arc_rx_id]['cap_hit']
        else:
            # unpaired arcs naming: "{if_id}__TX", "{if_id}__RX"
            if f"{if_id}__TX" in arcs:
                cap_hit_tx = arcs[f"{if_id}__TX"]['cap_hit']
            if f"{if_id}__RX" in arcs:
                cap_hit_rx = arcs[f"{if_id}__RX"]['cap_hit']

        # Finalize confidences with change penalty and cap-hit
        final_tx_conf = finalize_conf(base_tx_conf, r['orig_tx'], r['tx'], cap_hit_tx)
        final_rx_conf = finalize_conf(base_rx_conf, r['orig_rx'], r['rx'], cap_hit_rx)

        # Status calibration for down/up edge cases
        if status_resolved == 'down':
            final_rx_conf = 0.9 if r['orig_rx'] <= TRAFFIC_EVIDENCE_MIN else 0.3
            final_tx_conf = 0.9 if r['orig_tx'] <= TRAFFIC_EVIDENCE_MIN else 0.3
            # enforce zero
            r['rx'] = 0.0
            r['tx'] = 0.0
        elif status_resolved == 'up':
            if r['rx'] <= TRAFFIC_EVIDENCE_MIN and r['tx'] <= TRAFFIC_EVIDENCE_MIN:
                status_conf = clamp(status_conf * 0.9)

        # Assemble output for interface
        repaired_data: Dict[str, Tuple] = {}
        repaired_data['rx_rate'] = (r['orig_rx'], r['rx'], clamp(final_rx_conf))
        repaired_data['tx_rate'] = (r['orig_tx'], r['tx'], clamp(final_tx_conf))
        repaired_data['interface_status'] = (r['orig_status'], status_resolved, clamp(status_conf))
        # Copy metadata unchanged
        repaired_data['connected_to'] = r['connected_to']
        repaired_data['local_router'] = r['local_router']
        repaired_data['remote_router'] = r['remote_router']
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
