# EVOLVE-BLOCK-START
"""
Global weighted alternating projections for telemetry repair.

Approach:
- Initialize repaired rates from telemetry (down interfaces => 0).
- Iterate alternating projections:
  1) Link projection (R3): project each link's directional pairs onto equality
     using weighted least-squares averages with stiffness derived from proximity
     to original measurements; apply magnitude-aware gating/relaxation.
  2) Router projection (R1): for each router, project onto flow conservation
     by minimally reducing only the dominant side (TX if ΣTX>ΣRX else RX),
     solving a weighted least-squares with non-negativity bounds via an
     active-set reduction ("water-filling") scheme.

- Final status harmonization for paired links (down => no traffic).
- Confidence combines: change magnitude, peer mismatch (R3), and router residuals (R1).

Invariants:
- Link Symmetry (R3)
- Flow Conservation (R1)
- Interface Consistency
"""
from typing import Dict, Any, Tuple, List


def repair_network_telemetry(telemetry: Dict[str, Dict[str, Any]],
                             topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:
    # Constants and tolerances
    TH_REL_DEFAULT = 0.02   # 2% relative tolerance (τh)
    TH_REL_LOW = 0.05       # 5% for low-rate links (<10 Mbps)
    LOW_RATE_CUTOFF = 10.0  # Mbps
    ABS_GUARD = 0.5         # absolute diff guard for link projection (Mbps)
    ABS_GUARD_LOW = 0.3     # guard for low-rate
    ROUTER_ACT_TH_ABS = 0.5  # ignore tiny router residuals below this absolute
    MAX_ITERS = 8
    RELAX_EXP_LOW = 1.2
    EPS = 1e-9

    def to_float(x: Any) -> float:
        try:
            return float(x)
        except Exception:
            return 0.0

    # Build link pairs (unique undirected)
    pairs: Dict[Tuple[str, str], Tuple[str, str]] = {}
    for iid, d in telemetry.items():
        peer = d.get('connected_to')
        if peer and peer in telemetry:
            key = tuple(sorted([iid, peer]))
            if key not in pairs:
                pairs[key] = (iid, peer)

    # Router interfaces map (trust topology; ignore missing ids)
    router_if: Dict[str, List[str]] = {}
    for r, if_list in topology.items():
        router_if[r] = [iid for iid in if_list if iid in telemetry]

    # Precompute originals and initialize repaired values
    orig_tx: Dict[str, float] = {}
    orig_rx: Dict[str, float] = {}
    status: Dict[str, str] = {}
    local_router: Dict[str, Any] = {}
    remote_router: Dict[str, Any] = {}
    peer_of: Dict[str, str] = {}

    for iid, d in telemetry.items():
        orig_tx[iid] = to_float(d.get('tx_rate', 0.0))
        orig_rx[iid] = to_float(d.get('rx_rate', 0.0))
        status[iid] = d.get('interface_status', 'unknown')
        local_router[iid] = d.get('local_router')
        remote_router[iid] = d.get('remote_router')
        if d.get('connected_to') and d.get('connected_to') in telemetry:
            peer_of[iid] = d.get('connected_to')

    rep_tx: Dict[str, float] = {}
    rep_rx: Dict[str, float] = {}
    rep_status: Dict[str, str] = {}

    for iid in telemetry.keys():
        if status[iid] == 'down':
            rep_tx[iid] = 0.0
            rep_rx[iid] = 0.0
        else:
            rep_tx[iid] = orig_tx[iid]
            rep_rx[iid] = orig_rx[iid]
        rep_status[iid] = status[iid] if status[iid] in ('up', 'down') else 'up'

    def rel_diff(a: float, b: float) -> float:
        return abs(a - b) / max(a, b, 1.0)

    def magnitude_aware_gate(a: float, b: float) -> Tuple[bool, float, float, float]:
        m = max(a, b)
        thr = TH_REL_LOW if m < LOW_RATE_CUTOFF else TH_REL_DEFAULT
        guard = ABS_GUARD_LOW if m < LOW_RATE_CUTOFF else ABS_GUARD
        d = rel_diff(a, b)
        absd = abs(a - b)
        return (d > thr) and (absd > guard), d, thr, m

    def stiffness_weight(cur_val: float, orig_val: float) -> float:
        # Stiffer when current is close to original; softer when it already moved.
        scale = max(1.0, abs(cur_val), abs(orig_val))
        frac_change = abs(cur_val - orig_val) / scale
        # Base noise proportional to scale (2%) with an absolute floor (0.3 Mbps)
        base_sigma = 0.02 + (0.3 / scale)
        sigma = base_sigma + 0.8 * frac_change  # more change => higher sigma => lower weight
        return 1.0 / max(EPS, sigma * sigma)

    def project_link_pair(a_id: str, b_id: str) -> float:
        # Returns cumulative absolute change applied in this projection
        delta_sum = 0.0
        # A.tx vs B.rx
        x1 = rep_tx[a_id]; x2 = rep_rx[b_id]
        do_fix, d, thr, m = magnitude_aware_gate(x1, x2)
        if rep_status[a_id] != 'down' and rep_status[b_id] != 'down' and do_fix:
            w1 = stiffness_weight(x1, orig_tx[a_id])
            w2 = stiffness_weight(x2, orig_rx[b_id])
            v = (w1 * x1 + w2 * x2) / max(EPS, (w1 + w2))
            # Relaxation increases with severity
            k = min(1.0, max(0.0, (d - thr) / max(thr, EPS)))
            if m < LOW_RATE_CUTOFF:
                k = k ** RELAX_EXP_LOW
            new1 = (1.0 - k) * x1 + k * v
            new2 = (1.0 - k) * x2 + k * v
            new1 = max(0.0, new1); new2 = max(0.0, new2)
            delta_sum += abs(new1 - rep_tx[a_id]) + abs(new2 - rep_rx[b_id])
            rep_tx[a_id] = new1; rep_rx[b_id] = new2

        # B.tx vs A.rx
        x1 = rep_tx[b_id]; x2 = rep_rx[a_id]
        do_fix, d, thr, m = magnitude_aware_gate(x1, x2)
        if rep_status[a_id] != 'down' and rep_status[b_id] != 'down' and do_fix:
            w1 = stiffness_weight(x1, orig_tx[b_id])
            w2 = stiffness_weight(x2, orig_rx[a_id])
            v = (w1 * x1 + w2 * x2) / max(EPS, (w1 + w2))
            k = min(1.0, max(0.0, (d - thr) / max(thr, EPS)))
            if m < LOW_RATE_CUTOFF:
                k = k ** RELAX_EXP_LOW
            new1 = (1.0 - k) * x1 + k * v
            new2 = (1.0 - k) * x2 + k * v
            new1 = max(0.0, new1); new2 = max(0.0, new2)
            delta_sum += abs(new1 - rep_tx[b_id]) + abs(new2 - rep_rx[a_id])
            rep_tx[b_id] = new1; rep_rx[a_id] = new2

        return delta_sum

    def router_balance(router: str) -> float:
        # Project onto ΣTX = ΣRX by reducing only the louder side, minimal weighted L2,
        # subject to non-negativity. Returns cumulative absolute change.
        if_list = router_if.get(router, [])
        if not if_list:
            return 0.0
        sum_tx = sum(rep_tx[iid] for iid in if_list if rep_status[iid] != 'down')
        sum_rx = sum(rep_rx[iid] for iid in if_list if rep_status[iid] != 'down')
        scale = max(1.0, sum_tx, sum_rx)
        delta = sum_tx - sum_rx
        if abs(delta) <= max(ROUTER_ACT_TH_ABS, 0.02 * scale):
            return 0.0

        delta_sum = 0.0
        if delta > 0:
            # Reduce TX across up interfaces
            vars_list = []
            for iid in if_list:
                if rep_status[iid] == 'down':
                    continue
                cur = rep_tx[iid]
                if cur <= 0.0:
                    continue
                w = stiffness_weight(cur, orig_tx[iid])
                vars_list.append((iid, cur, w))
            remaining = delta
            active = {iid: (cur, w) for iid, cur, w in vars_list}
            while remaining > 1e-9 and active:
                inv_sum = sum(1.0 / max(EPS, w) for (_, (cur, w)) in active.items())
                if inv_sum <= EPS:
                    break
                changed_any = False
                for iid in list(active.keys()):
                    cur, w = active[iid]
                    share = (1.0 / max(EPS, w)) / inv_sum
                    reduce_amt = share * remaining
                    if reduce_amt >= cur - 0.0:
                        used = cur  # reduce to zero
                        rep_tx[iid] = 0.0
                        delta_sum += used
                        remaining -= used
                        del active[iid]
                        changed_any = True
                    else:
                        rep_tx[iid] = cur - reduce_amt
                        delta_sum += reduce_amt
                        remaining -= reduce_amt
                if not changed_any:
                    break
        else:
            # delta < 0 => Reduce RX across up interfaces
            need = -delta
            vars_list = []
            for iid in if_list:
                if rep_status[iid] == 'down':
                    continue
                cur = rep_rx[iid]
                if cur <= 0.0:
                    continue
                w = stiffness_weight(cur, orig_rx[iid])
                vars_list.append((iid, cur, w))
            remaining = need
            active = {iid: (cur, w) for iid, cur, w in vars_list}
            while remaining > 1e-9 and active:
                inv_sum = sum(1.0 / max(EPS, w) for (_, (cur, w)) in active.items())
                if inv_sum <= EPS:
                    break
                changed_any = False
                for iid in list(active.keys()):
                    cur, w = active[iid]
                    share = (1.0 / max(EPS, w)) / inv_sum
                    reduce_amt = share * remaining
                    if reduce_amt >= cur - 0.0:
                        used = cur
                        rep_rx[iid] = 0.0
                        delta_sum += used
                        remaining -= used
                        del active[iid]
                        changed_any = True
                    else:
                        rep_rx[iid] = cur - reduce_amt
                        delta_sum += reduce_amt
                        remaining -= reduce_amt
                if not changed_any:
                    break

        return delta_sum

    # Alternating projections loop
    for _ in range(MAX_ITERS):
        total_change = 0.0
        # Link projections
        for _, (a_id, b_id) in pairs.items():
            total_change += project_link_pair(a_id, b_id)
        # Router projections
        for r in router_if.keys():
            total_change += router_balance(r)
        # Early stopping: network-scale normalized change small
        net_scale = max(1.0,
                        sum(rep_tx[i] + rep_rx[i] for i in telemetry.keys()))
        if total_change / net_scale < 1e-4:
            break

    # Status harmonization for paired links: down => no traffic
    for _, (a_id, b_id) in pairs.items():
        a_up = rep_status[a_id] != 'down'
        b_up = rep_status[b_id] != 'down'
        a_has = (rep_tx[a_id] > EPS) or (rep_rx[a_id] > EPS)
        b_has = (rep_tx[b_id] > EPS) or (rep_rx[b_id] > EPS)
        if (not a_up) and (not b_up):
            rep_status[a_id] = 'down'; rep_status[b_id] = 'down'
            rep_tx[a_id] = rep_rx[a_id] = 0.0
            rep_tx[b_id] = rep_rx[b_id] = 0.0
        elif a_up and b_up:
            rep_status[a_id] = 'up'; rep_status[b_id] = 'up'
        else:
            # If any traffic observed, force up; else down
            if a_has or b_has:
                rep_status[a_id] = 'up'; rep_status[b_id] = 'up'
            else:
                rep_status[a_id] = 'down'; rep_status[b_id] = 'down'
                rep_tx[a_id] = rep_rx[a_id] = 0.0
                rep_tx[b_id] = rep_rx[b_id] = 0.0

    # Router residuals after repairs (for confidence)
    def router_residuals() -> Dict[str, float]:
        res: Dict[str, float] = {}
        for r, if_list in router_if.items():
            s_tx = sum(rep_tx[iid] for iid in if_list if rep_status[iid] != 'down')
            s_rx = sum(rep_rx[iid] for iid in if_list if rep_status[iid] != 'down')
            res[r] = abs(s_tx - s_rx) / max(1.0, s_tx, s_rx)
        return res

    router_resid = router_residuals()

    # Confidence calculation helpers
    def change_ratio(orig: float, rep: float) -> float:
        denom = max(1.0, abs(orig), abs(rep))
        return abs(rep - orig) / denom

    # Build result
    result: Dict[str, Dict[str, Tuple]] = {}
    for iid, d in telemetry.items():
        # Peer mismatch residuals
        peer = peer_of.get(iid)
        # Directional peer diffs
        d_peer_rx = 0.0
        d_peer_tx = 0.0
        if peer:
            # my tx vs peer rx
            d_peer_tx = rel_diff(rep_tx[iid], rep_rx[peer])
            # my rx vs peer tx
            d_peer_rx = rel_diff(rep_rx[iid], rep_tx[peer])

        # Router residual penalties (direction-aware)
        r_local = local_router.get(iid)
        r_remote = remote_router.get(iid)
        resid_local = router_resid.get(r_local, 0.0)
        resid_remote = router_resid.get(r_remote, 0.0)

        # Base confidences from own change magnitude
        tx_ch = change_ratio(orig_tx[iid], rep_tx[iid])
        rx_ch = change_ratio(orig_rx[iid], rep_rx[iid])
        tx_conf = max(0.0, 1.0 - 0.9 * tx_ch)
        rx_conf = max(0.0, 1.0 - 0.9 * rx_ch)

        # Peer agreement scaling; strong floors when within tight tolerance
        def peer_scale(diff_val: float, rep_mag: float) -> float:
            thr = TH_REL_LOW if rep_mag < LOW_RATE_CUTOFF else TH_REL_DEFAULT
            if diff_val <= min(0.005, 0.5 * thr) and rep_mag >= LOW_RATE_CUTOFF:
                return max(0.99, 1.0 - diff_val)
            return max(0.0, 1.0 - diff_val)

        tx_conf *= peer_scale(d_peer_tx, max(rep_tx[iid], rep_rx[peer]) if peer else rep_tx[iid])
        rx_conf *= peer_scale(d_peer_rx, max(rep_rx[iid], rep_tx[peer]) if peer else rep_rx[iid])

        # Router residual penalties: TX leans on local, RX on remote
        penalty_tx = 1.0 - (0.6 * resid_local + 0.4 * resid_remote)
        penalty_rx = 1.0 - (0.6 * resid_remote + 0.4 * resid_local)
        penalty_tx = max(0.0, min(1.0, penalty_tx))
        penalty_rx = max(0.0, min(1.0, penalty_rx))
        tx_conf = max(0.0, min(1.0, tx_conf * penalty_tx))
        rx_conf = max(0.0, min(1.0, rx_conf * penalty_rx))

        # Status decision and confidence
        final_status = rep_status[iid]
        orig_status = status[iid]
        has_traf = (rep_tx[iid] > EPS) or (rep_rx[iid] > EPS)
        if final_status == 'down':
            status_conf = 0.98 if (orig_status == 'down' and not has_traf) else 0.75
        else:
            if orig_status == 'up':
                status_conf = 0.95
            else:
                # We promoted to up due to evidence
                status_conf = 0.8

        # Assemble output tuples
        out: Dict[str, Any] = {}
        out['rx_rate'] = (orig_rx[iid], rep_rx[iid], float(rx_conf))
        out['tx_rate'] = (orig_tx[iid], rep_tx[iid], float(tx_conf))
        out['interface_status'] = (orig_status, final_status, float(status_conf))
        out['connected_to'] = d.get('connected_to')
        out['local_router'] = local_router.get(iid)
        out['remote_router'] = remote_router.get(iid)
        result[iid] = out

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