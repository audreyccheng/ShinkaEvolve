# EVOLVE-BLOCK-START
"""
Network telemetry repair with residual-tilted consensus, asymmetric near-threshold
averaging, router-guarded micro-adjustments, and share-aware confidence penalties.

Key invariants:
- Link Symmetry (R3): my_tx ≈ their_rx and my_rx ≈ their_tx
- Flow Conservation (R1): Σ incoming = Σ outgoing per router
- Interface Consistency: link pair status aligned; down links carry no traffic

Parameters:
- tau_default: 0.02 (2%) baseline tolerance
- tau_low: 0.05 (5%) for low-rate links (< LOW_RATE_THRESH)
- abs_guard: 0.5 Mbps absolute difference minimum to trigger corrections
- LOW_RATE_THRESH: 10.0 Mbps magnitude boundary
- gamma_scale: 0.1, gamma_cap: 0.08 for residual-tilted consensus
- Partial band: T < diff ≤ 2T with asymmetric k_loud/k_quiet
- Router micro-adjustment guard:
  - R1_TRIGGER = 0.03, DOMINANCE = 0.60, IMPROVE_MIN = 0.10, NUDGE_CAP_FRAC = 0.02
- Share-aware confidence penalties (direction-coupled)
"""
from typing import Dict, Any, Tuple, List
import math


def repair_network_telemetry(telemetry: Dict[str, Dict[str, Any]], 
                             topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:
    # Core parameters
    tau_default = 0.02            # 2% baseline tolerance
    tau_low = 0.05                # 5% for low-rate links
    LOW_RATE_THRESH = 10.0        # Mbps
    abs_guard = 0.5               # Mbps absolute guard
    EPS = 1e-6

    # Residual-tilt parameters
    gamma_scale = 0.1
    gamma_cap = 0.08

    # Micro-adjustment (router R1 guard) parameters
    R1_TRIGGER = 0.03
    DOMINANCE = 0.60
    IMPROVE_MIN = 0.10
    NUDGE_CAP_FRAC = 0.02

    def to_float(x: Any) -> float:
        try:
            v = float(x)
            if math.isnan(v) or math.isinf(v):
                return 0.0
            return v
        except Exception:
            return 0.0

    def clamp(x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))

    def rel_diff(a: float, b: float) -> float:
        return abs(a - b) / max(a, b, 1.0)

    def has_traffic(d: Dict[str, Any]) -> bool:
        return (to_float(d.get('rx_rate', 0.0)) > EPS) or (to_float(d.get('tx_rate', 0.0)) > EPS)

    def dyn_tau(a: float, b: float) -> float:
        return tau_low if max(a, b) < LOW_RATE_THRESH else tau_default

    # Precompute raw router sums and residuals (before any repair) for residual-tilted weighting
    raw_router_sums = {}  # router -> (sum_tx_raw, sum_rx_raw)
    for router, if_list in topology.items():
        s_tx = 0.0
        s_rx = 0.0
        for iid in if_list:
            d = telemetry.get(iid)
            if not d:
                continue
            status = d.get('interface_status', 'unknown')
            # Treat raw as reported; do not zero-out for residual tilt since it's a detection step
            s_tx += to_float(d.get('tx_rate', 0.0))
            s_rx += to_float(d.get('rx_rate', 0.0))
        raw_router_sums[router] = (s_tx, s_rx)

    raw_router_resid = {}  # router -> resid_frac
    for r, (s_tx, s_rx) in raw_router_sums.items():
        raw_router_resid[r] = abs(s_tx - s_rx) / max(s_tx, s_rx, 1.0)

    # Build unique link pairs
    pairs: Dict[Tuple[str, str], Tuple[str, str]] = {}
    for if_id, data in telemetry.items():
        peer = data.get('connected_to')
        if peer and peer in telemetry:
            key = tuple(sorted([if_id, peer]))
            if key not in pairs:
                pairs[key] = (if_id, peer)

    per_if = {}  # interface_id -> interim repaired values and base confidences

    # Helper: apply residual-tilted consensus repair on a pair of measurements
    def tilted_consensus(a_val: float, b_val: float,
                         a_local_router: str,
                         diff: float, tau: float) -> Tuple[float, float]:
        # only used when diff > tau and abs difference > abs_guard by caller
        # initial equal weights
        w_a = 0.5
        # compute residual tilt
        s_tx, s_rx = raw_router_sums.get(a_local_router, (0.0, 0.0))
        resid_loc = raw_router_resid.get(a_local_router, 0.0)
        delta_loc = s_tx - s_rx  # signed
        # If (a - b) and delta_loc have same sign, shrink local weight to move toward reducing delta
        sign = (1 if (a_val - b_val) > 0 else (-1 if (a_val - b_val) < 0 else 0)) * (1 if delta_loc > 0 else (-1 if delta_loc < 0 else 0))
        gamma = min(gamma_cap, gamma_scale * resid_loc)
        if sign > 0:
            w_a = clamp(w_a - gamma, 0.1, 0.9)
        elif sign < 0:
            w_a = clamp(w_a + gamma, 0.1, 0.9)
        v = w_a * a_val + (1.0 - w_a) * b_val
        return v, v

    # Pairwise hardening and status harmonization
    for _, (a_id, b_id) in pairs.items():
        a = telemetry[a_id]
        b = telemetry[b_id]

        a_status = a.get('interface_status', 'unknown')
        b_status = b.get('interface_status', 'unknown')

        a_rx, a_tx = to_float(a.get('rx_rate', 0.0)), to_float(a.get('tx_rate', 0.0))
        b_rx, b_tx = to_float(b.get('rx_rate', 0.0)), to_float(b.get('tx_rate', 0.0))

        a_has = has_traffic(a)
        b_has = has_traffic(b)

        # Pair status determination
        if a_status == 'down' and b_status == 'down':
            pair_status = 'down'
        elif a_status == 'up' and b_status == 'up':
            pair_status = 'up'
        else:
            pair_status = 'up' if (a_has or b_has) else 'down'

        # Initialize
        rep_a_tx, rep_b_rx = a_tx, b_rx
        rep_b_tx, rep_a_rx = b_tx, a_rx
        rx_conf_a = 1.0
        tx_conf_a = 1.0
        rx_conf_b = 1.0
        tx_conf_b = 1.0

        if pair_status == 'down':
            rep_a_tx = rep_b_rx = rep_b_tx = rep_a_rx = 0.0
            base_conf = 0.95 if not (a_has or b_has) else 0.7
            rx_conf_a = tx_conf_a = rx_conf_b = tx_conf_b = base_conf
        else:
            # Direction A->B: A.tx vs B.rx
            diff_ab = rel_diff(a_tx, b_rx)
            tau_ab = dyn_tau(a_tx, b_rx)
            abs_ab = abs(a_tx - b_rx)
            if (diff_ab > tau_ab) and (abs_ab > abs_guard):
                # residual-tilted weighted consensus target
                v_ab_a, v_ab_b = tilted_consensus(a_tx, b_rx, a.get('local_router'), diff_ab, tau_ab)
                # asymmetric partial averaging within 2*tau
                if diff_ab <= 2.0 * tau_ab:
                    k_base = clamp((diff_ab - tau_ab) / max(tau_ab, 1e-9), 0.0, 1.0)
                    loud = max(a_tx, b_rx)
                    quiet = min(a_tx, b_rx)
                    loud_ratio = clamp((loud - quiet) / max(loud, 1.0), 0.0, 1.0)
                    k_loud = clamp(k_base * (1.0 + 0.5 * loud_ratio), 0.0, 1.0)
                    k_quiet = clamp(k_base * (1.0 - 0.3 * loud_ratio), 0.0, 1.0)
                    # Apply k per endpoint based on which is louder
                    if a_tx >= b_rx:
                        rep_a_tx = a_tx * (1.0 - k_loud) + v_ab_a * k_loud
                        rep_b_rx = b_rx * (1.0 - k_quiet) + v_ab_b * k_quiet
                    else:
                        rep_a_tx = a_tx * (1.0 - k_quiet) + v_ab_a * k_quiet
                        rep_b_rx = b_rx * (1.0 - k_loud) + v_ab_b * k_loud
                else:
                    rep_a_tx, rep_b_rx = v_ab_a, v_ab_b
                # confidence for A->B
                change_a = abs(rep_a_tx - a_tx) / max(abs(rep_a_tx), abs(a_tx), 1.0)
                change_b = abs(rep_b_rx - b_rx) / max(abs(rep_b_rx), abs(b_rx), 1.0)
                conf_base = max(0.0, 1.0 - diff_ab)
                tx_conf_a = min(conf_base, 1.0 - 0.8 * change_a)
                rx_conf_b = min(conf_base, 1.0 - 0.8 * change_b)
            else:
                # within tolerance -> strong floors (higher for high-rate with very tight match)
                max_ab = max(a_tx, b_rx)
                if max_ab >= LOW_RATE_THRESH and diff_ab <= 0.005:
                    floor = 0.99
                else:
                    floor = 0.98 if max_ab >= LOW_RATE_THRESH else 0.97
                tx_conf_a = max(tx_conf_a, floor)
                rx_conf_b = max(rx_conf_b, floor)

            # Direction B->A: B.tx vs A.rx
            diff_ba = rel_diff(b_tx, a_rx)
            tau_ba = dyn_tau(b_tx, a_rx)
            abs_ba = abs(b_tx - a_rx)
            if (diff_ba > tau_ba) and (abs_ba > abs_guard):
                v_ba_b, v_ba_a = tilted_consensus(b_tx, a_rx, b.get('local_router'), diff_ba, tau_ba)
                if diff_ba <= 2.0 * tau_ba:
                    k_base2 = clamp((diff_ba - tau_ba) / max(tau_ba, 1e-9), 0.0, 1.0)
                    loud2 = max(b_tx, a_rx)
                    quiet2 = min(b_tx, a_rx)
                    loud_ratio2 = clamp((loud2 - quiet2) / max(loud2, 1.0), 0.0, 1.0)
                    k_loud2 = clamp(k_base2 * (1.0 + 0.5 * loud_ratio2), 0.0, 1.0)
                    k_quiet2 = clamp(k_base2 * (1.0 - 0.3 * loud_ratio2), 0.0, 1.0)
                    if b_tx >= a_rx:
                        rep_b_tx = b_tx * (1.0 - k_loud2) + v_ba_b * k_loud2
                        rep_a_rx = a_rx * (1.0 - k_quiet2) + v_ba_a * k_quiet2
                    else:
                        rep_b_tx = b_tx * (1.0 - k_quiet2) + v_ba_b * k_quiet2
                        rep_a_rx = a_rx * (1.0 - k_loud2) + v_ba_a * k_loud2
                else:
                    rep_b_tx, rep_a_rx = v_ba_b, v_ba_a
                change_b2 = abs(rep_b_tx - b_tx) / max(abs(rep_b_tx), abs(b_tx), 1.0)
                change_a2 = abs(rep_a_rx - a_rx) / max(abs(rep_a_rx), abs(a_rx), 1.0)
                conf_base2 = max(0.0, 1.0 - diff_ba)
                tx_conf_b = min(conf_base2, 1.0 - 0.8 * change_b2)
                rx_conf_a = min(conf_base2, 1.0 - 0.8 * change_a2)
            else:
                max_ba = max(b_tx, a_rx)
                if max_ba >= LOW_RATE_THRESH and diff_ba <= 0.005:
                    floor2 = 0.99
                else:
                    floor2 = 0.98 if max_ba >= LOW_RATE_THRESH else 0.97
                tx_conf_b = max(tx_conf_b, floor2)
                rx_conf_a = max(rx_conf_a, floor2)

            # Penalize one-sided traffic evidence on "up" decision
            if a_has != b_has:
                if not a_has:
                    rx_conf_a *= 0.88
                    tx_conf_a *= 0.88
                if not b_has:
                    rx_conf_b *= 0.88
                    tx_conf_b *= 0.88

        # Status confidence
        if pair_status == 'down':
            if a_status == 'down' and b_status == 'down' and not (a_has or b_has):
                status_conf = 0.98
            else:
                status_conf = 0.7
        else:
            status_conf = 0.95 if (a_status == 'up' and b_status == 'up') else 0.8

        # Change-aware calibration (cap overly optimistic confidences)
        def change_ratio(orig: float, rep: float) -> float:
            return abs(rep - orig) / max(abs(orig), abs(rep), 1.0)

        a_rx_ch = change_ratio(a_rx, rep_a_rx)
        a_tx_ch = change_ratio(a_tx, rep_a_tx)
        b_rx_ch = change_ratio(b_rx, rep_b_rx)
        b_tx_ch = change_ratio(b_tx, rep_b_tx)

        rx_conf_a = clamp(min(rx_conf_a, 1.0 - a_rx_ch), 0.0, 1.0)
        tx_conf_a = clamp(min(tx_conf_a, 1.0 - a_tx_ch), 0.0, 1.0)
        rx_conf_b = clamp(min(rx_conf_b, 1.0 - b_rx_ch), 0.0, 1.0)
        tx_conf_b = clamp(min(tx_conf_b, 1.0 - b_tx_ch), 0.0, 1.0)

        per_if[a_id] = {
            'repaired_rx': rep_a_rx,
            'repaired_tx': rep_a_tx,
            'rx_conf': rx_conf_a,
            'tx_conf': tx_conf_a,
            'repaired_status': pair_status,
            'status_conf': status_conf
        }
        per_if[b_id] = {
            'repaired_rx': rep_b_rx,
            'repaired_tx': rep_b_tx,
            'rx_conf': rx_conf_b,
            'tx_conf': tx_conf_b,
            'repaired_status': pair_status,
            'status_conf': status_conf
        }

    # Dangling interfaces (no valid peer)
    for if_id, data in telemetry.items():
        if if_id in per_if:
            continue
        status = data.get('interface_status', 'unknown')
        rx = to_float(data.get('rx_rate', 0.0))
        tx = to_float(data.get('tx_rate', 0.0))
        if status == 'down':
            per_if[if_id] = {
                'repaired_rx': 0.0,
                'repaired_tx': 0.0,
                'rx_conf': 0.9,
                'tx_conf': 0.9,
                'repaired_status': 'down',
                'status_conf': 0.95
            }
        else:
            per_if[if_id] = {
                'repaired_rx': rx,
                'repaired_tx': tx,
                'rx_conf': 0.6,
                'tx_conf': 0.6,
                'repaired_status': status if status in ('up', 'down') else 'up',
                'status_conf': 0.6
            }

    # Router-level residuals after pair repair (pre micro-adjust)
    def compute_router_residuals(per_if_state: Dict[str, Dict[str, Any]]) -> Dict[str, float]:
        resids: Dict[str, float] = {}
        for router, if_list in topology.items():
            s_tx = 0.0
            s_rx = 0.0
            for iid in if_list:
                rep = per_if_state.get(iid)
                if not rep:
                    continue
                s_tx += to_float(rep['repaired_tx'])
                s_rx += to_float(rep['repaired_rx'])
            resids[router] = abs(s_tx - s_rx) / max(s_tx, s_rx, 1.0)
        return resids

    def compute_router_sums(per_if_state: Dict[str, Dict[str, Any]]) -> Dict[str, Tuple[float, float]]:
        sums: Dict[str, Tuple[float, float]] = {}
        for router, if_list in topology.items():
            s_tx = 0.0
            s_rx = 0.0
            for iid in if_list:
                rep = per_if_state.get(iid)
                if not rep:
                    continue
                s_tx += to_float(rep['repaired_tx'])
                s_rx += to_float(rep['repaired_rx'])
            sums[router] = (s_tx, s_rx)
        return sums

    router_sums = compute_router_sums(per_if)
    router_residual = compute_router_residuals(per_if)

    # Micro-adjustment: only on dangling interfaces, with improvement check and dominance test
    paired_ids = set()
    for _, (aid, bid) in pairs.items():
        paired_ids.add(aid)
        paired_ids.add(bid)

    for router, if_list in topology.items():
        resid = router_residual.get(router, 0.0)
        if resid < R1_TRIGGER:
            continue
        s_tx, s_rx = router_sums.get(router, (0.0, 0.0))
        scale = max(s_tx, s_rx, 1.0)
        delta = s_tx - s_rx  # positive: too much TX; negative: too much RX

        # candidates: dangling, up
        cand = []
        for iid in if_list:
            if iid in paired_ids:
                continue
            rep = per_if.get(iid)
            if not rep:
                continue
            if rep.get('repaired_status', 'up') == 'down':
                continue
            cand.append(iid)

        if not cand:
            continue

        # dominance in same-direction contribution
        if delta > 0.0:
            # need to increase RX
            total_rx_cand = sum(to_float(per_if[i]['repaired_rx']) for i in cand) + EPS
            leader = max(cand, key=lambda x: to_float(per_if[x]['repaired_rx']))
            leader_share = to_float(per_if[leader]['repaired_rx']) / total_rx_cand
            if leader_share < DOMINANCE:
                continue  # no dominant candidate; skip to avoid risky adjustments
            nudge = min(delta, NUDGE_CAP_FRAC * scale)
            # tentative apply
            new_rx = to_float(per_if[leader]['repaired_rx']) + nudge
            old_rx = to_float(per_if[leader]['repaired_rx'])
            per_if[leader]['repaired_rx'] = new_rx
            # recompute residual
            router_sums[router] = (s_tx, s_rx + nudge)
            new_resid = abs((s_tx) - (s_rx + nudge)) / max(s_tx, s_rx + nudge, 1.0)
            if new_resid <= (1.0 - IMPROVE_MIN) * resid:
                # commit and downgrade confidence based on change
                rx_orig = to_float(telemetry.get(leader, {}).get('rx_rate', 0.0))
                ch = abs(new_rx - rx_orig) / max(abs(new_rx), abs(rx_orig), 1.0)
                per_if[leader]['rx_conf'] = clamp(min(per_if[leader].get('rx_conf', 0.6), 1.0 - ch), 0.0, 1.0)
                router_residual[router] = new_resid
            else:
                # revert
                per_if[leader]['repaired_rx'] = old_rx
                router_sums[router] = (s_tx, s_rx)
        elif delta < 0.0:
            # need to increase TX
            need = -delta
            total_tx_cand = sum(to_float(per_if[i]['repaired_tx']) for i in cand) + EPS
            leader = max(cand, key=lambda x: to_float(per_if[x]['repaired_tx']))
            leader_share = to_float(per_if[leader]['repaired_tx']) / total_tx_cand
            if leader_share < DOMINANCE:
                continue
            nudge = min(need, NUDGE_CAP_FRAC * scale)
            new_tx = to_float(per_if[leader]['repaired_tx']) + nudge
            old_tx = to_float(per_if[leader]['repaired_tx'])
            per_if[leader]['repaired_tx'] = new_tx
            router_sums[router] = (s_tx + nudge, s_rx)
            new_resid = abs((s_tx + nudge) - (s_rx)) / max(s_tx + nudge, s_rx, 1.0)
            if new_resid <= (1.0 - IMPROVE_MIN) * resid:
                tx_orig = to_float(telemetry.get(leader, {}).get('tx_rate', 0.0))
                ch = abs(new_tx - tx_orig) / max(abs(new_tx), abs(tx_orig), 1.0)
                per_if[leader]['tx_conf'] = clamp(min(per_if[leader].get('tx_conf', 0.6), 1.0 - ch), 0.0, 1.0)
                router_residual[router] = new_resid
            else:
                per_if[leader]['repaired_tx'] = old_tx
                router_sums[router] = (s_tx, s_rx)

    # Final residuals (post micro-adjust) for confidence scaling
    router_residual = compute_router_residuals(per_if)

    # Final assembly with share-aware, direction-coupled penalties
    result: Dict[str, Dict[str, Tuple]] = {}
    # Precompute per-router directional sums to get interface shares
    router_tx_sum = {r: 0.0 for r in topology}
    router_rx_sum = {r: 0.0 for r in topology}
    for r, if_list in topology.items():
        for iid in if_list:
            rep = per_if.get(iid)
            if not rep:
                continue
            router_tx_sum[r] += to_float(rep['repaired_tx'])
            router_rx_sum[r] += to_float(rep['repaired_rx'])

    for if_id, data in telemetry.items():
        rep = per_if.get(if_id, {})
        repaired_rx = to_float(rep.get('repaired_rx', data.get('rx_rate', 0.0)))
        repaired_tx = to_float(rep.get('repaired_tx', data.get('tx_rate', 0.0)))
        repaired_status = rep.get('repaired_status', data.get('interface_status', 'unknown'))

        rx_conf = float(rep.get('rx_conf', 0.6))
        tx_conf = float(rep.get('tx_conf', 0.6))
        status_conf = float(rep.get('status_conf', 0.6))

        local_router = data.get('local_router')
        remote_router = data.get('remote_router')

        resid_local = router_residual.get(local_router, 0.0)
        resid_remote = router_residual.get(remote_router, 0.0)

        # Interface shares
        sum_tx_loc = router_tx_sum.get(local_router, 0.0)
        sum_rx_loc = router_rx_sum.get(local_router, 0.0)
        tx_share = repaired_tx / max(sum_tx_loc, 1.0)
        rx_share = repaired_rx / max(sum_rx_loc, 1.0)

        # Share-aware, direction-coupled penalties
        pen_tx = 1.0 - ((0.6 + 0.2 * tx_share) * resid_local + (0.4 - 0.2 * tx_share) * resid_remote)
        pen_rx = 1.0 - ((0.6 + 0.2 * rx_share) * resid_remote + (0.4 - 0.2 * rx_share) * resid_local)
        pen_tx = clamp(pen_tx, 0.0, 1.0)
        pen_rx = clamp(pen_rx, 0.0, 1.0)

        tx_conf = clamp(tx_conf * pen_tx, 0.0, 1.0)
        rx_conf = clamp(rx_conf * pen_rx, 0.0, 1.0)
        status_conf = clamp(status_conf * (0.85 + 0.15 * min(pen_tx, pen_rx)), 0.0, 1.0)

        # Output tuples: (original, repaired, confidence)
        out: Dict[str, Any] = {}
        rx_orig = to_float(data.get('rx_rate', 0.0))
        tx_orig = to_float(data.get('tx_rate', 0.0))
        status_orig = data.get('interface_status', 'unknown')

        out['rx_rate'] = (rx_orig, repaired_rx, rx_conf)
        out['tx_rate'] = (tx_orig, repaired_tx, tx_conf)
        out['interface_status'] = (status_orig, repaired_status, status_conf)

        # Copy metadata unchanged
        out['connected_to'] = data.get('connected_to')
        out['local_router'] = local_router
        out['remote_router'] = remote_router

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

