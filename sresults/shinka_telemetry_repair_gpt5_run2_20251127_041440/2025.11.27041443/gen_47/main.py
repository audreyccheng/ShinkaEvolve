# EVOLVE-BLOCK-START
"""
Residual-tilted asymmetric consensus repair with share-aware confidence.

Core invariants:
- Link Symmetry (R3): my_tx ≈ their_rx and my_rx ≈ their_tx
- Flow Conservation (R1): Σ incoming = Σ outgoing per router
- Interface Consistency: paired statuses aligned; down links carry no traffic

Novelty:
- Residual-tilted consensus weights when repairing pairs: tilt away from the side
  that would worsen the local router's imbalance, bounded and magnitude-aware.
- Asymmetric partial averaging: near-threshold violations move the louder side more.
- Improvement-checked micro-adjustments on dangling interfaces only.
- Interface-share–aware, direction-coupled router residual penalties for confidence.
"""
from typing import Dict, Any, Tuple, List


def repair_network_telemetry(telemetry: Dict[str, Dict[str, Any]],
                             topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:
    # Parameters
    TH_REL_DEFAULT = 0.02      # 2% default tolerance
    TH_REL_LOW = 0.05          # 5% tolerance for low-rate links
    ABS_GUARD = 0.5            # 0.5 Mbps absolute guard for triggering repairs
    STRONG_AGREE = 0.005       # 0.5% strong agreement floor
    LOW_RATE_MAX = 10.0        # low-rate threshold in Mbps
    EPS = 1e-6

    def f64(x: Any, default: float = 0.0) -> float:
        try:
            v = float(x)
            if v != v:  # NaN guard
                return default
            return v
        except Exception:
            return default

    def rel_diff(a: float, b: float) -> float:
        return abs(a - b) / max(a, b, 1.0)

    def dyn_tau(a: float, b: float) -> float:
        return TH_REL_LOW if max(a, b) < LOW_RATE_MAX else TH_REL_DEFAULT

    def has_traffic(d: Dict[str, Any]) -> bool:
        return (f64(d.get('rx_rate', 0.0)) > EPS) or (f64(d.get('tx_rate', 0.0)) > EPS)

    # Precompute original router residuals and deltas for residual-tilted consensus
    # If topology lacks some routers, we gracefully skip them.
    orig_router_delta: Dict[str, float] = {}
    orig_router_scale: Dict[str, float] = {}
    orig_router_resid: Dict[str, float] = {}
    for router, if_list in topology.items():
        sum_tx = 0.0
        sum_rx = 0.0
        for iid in if_list:
            d = telemetry.get(iid)
            if not d:
                continue
            sum_tx += f64(d.get('tx_rate', 0.0))
            sum_rx += f64(d.get('rx_rate', 0.0))
        delta = sum_tx - sum_rx
        scale = max(sum_tx, sum_rx, 1.0)
        resid = abs(delta) / scale
        orig_router_delta[router] = delta
        orig_router_scale[router] = scale
        orig_router_resid[router] = resid

    # Build unique undirected link pairs
    pairs: Dict[Tuple[str, str], Tuple[str, str]] = {}
    for if_id, data in telemetry.items():
        peer = data.get('connected_to')
        if peer and peer in telemetry:
            key = tuple(sorted([if_id, peer]))
            if key not in pairs:
                pairs[key] = (if_id, peer)

    # First pass: pairwise hardening with residual-tilted consensus and status harmonization
    per_if: Dict[str, Dict[str, Any]] = {}

    for _, (a_id, b_id) in pairs.items():
        a = telemetry[a_id]
        b = telemetry[b_id]

        a_stat = a.get('interface_status', 'unknown')
        b_stat = b.get('interface_status', 'unknown')

        a_rx, a_tx = f64(a.get('rx_rate', 0.0)), f64(a.get('tx_rate', 0.0))
        b_rx, b_tx = f64(b.get('rx_rate', 0.0)), f64(b.get('tx_rate', 0.0))

        a_has = has_traffic(a)
        b_has = has_traffic(b)

        # Status harmonization with traffic evidence
        if a_stat == 'down' and b_stat == 'down':
            pair_status = 'down'
        elif a_stat == 'up' and b_stat == 'up':
            pair_status = 'up'
        else:
            pair_status = 'up' if (a_has or b_has) else 'down'

        # Initialize repaired values
        rep_a_tx, rep_b_rx = a_tx, b_rx
        rep_b_tx, rep_a_rx = b_tx, a_rx
        rx_conf_a = tx_conf_a = rx_conf_b = tx_conf_b = 1.0

        if pair_status == 'down':
            # Enforce no traffic
            rep_a_tx = rep_b_rx = rep_b_tx = rep_a_rx = 0.0
            base_conf = 0.95 if not (a_has or b_has) else 0.7
            rx_conf_a = tx_conf_a = rx_conf_b = tx_conf_b = base_conf
        else:
            # Prepare residual-tilted consensus for each direction
            a_loc = a.get('local_router')
            b_loc = b.get('local_router')
            delta_a_loc = orig_router_delta.get(a_loc, 0.0)
            resid_a_loc = orig_router_resid.get(a_loc, 0.0)
            delta_b_loc = orig_router_delta.get(b_loc, 0.0)
            resid_b_loc = orig_router_resid.get(b_loc, 0.0)

            # Direction A->B: A.tx vs B.rx
            d_ab = rel_diff(a_tx, b_rx)
            tau_ab = dyn_tau(a_tx, b_rx)
            abs_ab = abs(a_tx - b_rx)

            if (d_ab > tau_ab) and (abs_ab > ABS_GUARD):
                # Base equal weights
                w_a = 0.5
                # Residual-tilting toward reducing imbalance at A's local router
                # If (a_tx - b_rx) and delta_a_loc share sign, decreasing a_tx helps (so shrink a's weight).
                if (a_tx - b_rx) * delta_a_loc > 0:
                    gamma = min(0.08, 0.1 * resid_a_loc)
                    w_a = max(0.1, w_a - gamma)
                # Weighted consensus
                v = w_a * a_tx + (1.0 - w_a) * b_rx

                if d_ab <= 2.0 * tau_ab:
                    # Asymmetric partial move: louder side moves more
                    k_base = (d_ab - tau_ab) / max(tau_ab, 1e-9)  # in (0,1]
                    max_ab = max(a_tx, b_rx)
                    sum_ab = max(a_tx + b_rx, EPS)
                    loud_is_a = a_tx >= b_rx
                    loud_share = max_ab / sum_ab  # in (0.5, ~1)
                    k_loud = min(1.0, k_base * (1.0 + 0.5 * loud_share))
                    k_quiet = min(1.0, k_base * (1.0 - 0.3 * loud_share))
                    # Small extra tilt on k toward reducing imbalance at A's router
                    if (a_tx - b_rx) * delta_a_loc > 0:
                        tilt_scale = min(1.0, (resid_a_loc or 0.0) * 0.5)
                        if loud_is_a:
                            k_a = min(1.0, k_loud * (1.0 + 0.25 * tilt_scale))
                            k_b = max(0.0, k_quiet * (1.0 - 0.2 * tilt_scale))
                        else:
                            k_a = min(1.0, k_quiet * (1.0 + 0.25 * tilt_scale))
                            k_b = max(0.0, k_loud * (1.0 - 0.2 * tilt_scale))
                    else:
                        k_a = k_loud if loud_is_a else k_quiet
                        k_b = k_quiet if loud_is_a else k_loud

                    rep_a_tx = a_tx * (1.0 - k_a) + v * k_a
                    rep_b_rx = b_rx * (1.0 - k_b) + v * k_b
                else:
                    rep_a_tx = v
                    rep_b_rx = v

            # Confidence for A->B direction: depends on violation and change magnitude
            d_after_ab = rel_diff(rep_a_tx, rep_b_rx)
            base_ab = max(0.0, 1.0 - max(d_ab, d_after_ab))
            if d_ab <= STRONG_AGREE and max(a_tx, b_rx) >= LOW_RATE_MAX:
                base_ab = max(base_ab, 0.99)
            elif d_ab <= tau_ab:
                base_ab = max(base_ab, 0.98 if max(a_tx, b_rx) >= LOW_RATE_MAX else 0.97)
            change_a_tx = abs(rep_a_tx - a_tx) / max(abs(rep_a_tx), abs(a_tx), 1.0)
            change_b_rx = abs(rep_b_rx - b_rx) / max(abs(rep_b_rx), abs(b_rx), 1.0)
            tx_conf_a = min(base_ab, 1.0 - 0.8 * change_a_tx)
            rx_conf_b = min(base_ab, 1.0 - 0.8 * change_b_rx)

            # Direction B->A: B.tx vs A.rx
            d_ba = rel_diff(b_tx, a_rx)
            tau_ba = dyn_tau(b_tx, a_rx)
            abs_ba = abs(b_tx - a_rx)

            if (d_ba > tau_ba) and (abs_ba > ABS_GUARD):
                w_b = 0.5
                if (b_tx - a_rx) * delta_b_loc > 0:
                    gamma2 = min(0.08, 0.1 * resid_b_loc)
                    w_b = max(0.1, w_b - gamma2)
                v2 = w_b * b_tx + (1.0 - w_b) * a_rx

                if d_ba <= 2.0 * tau_ba:
                    k_base2 = (d_ba - tau_ba) / max(tau_ba, 1e-9)
                    max_ba = max(b_tx, a_rx)
                    sum_ba = max(b_tx + a_rx, EPS)
                    loud_is_b = b_tx >= a_rx
                    loud_share2 = max_ba / sum_ba
                    k_loud2 = min(1.0, k_base2 * (1.0 + 0.5 * loud_share2))
                    k_quiet2 = min(1.0, k_base2 * (1.0 - 0.3 * loud_share2))
                    if (b_tx - a_rx) * delta_b_loc > 0:
                        tilt_scale2 = min(1.0, (resid_b_loc or 0.0) * 0.5)
                        if loud_is_b:
                            k_b_ = min(1.0, k_loud2 * (1.0 + 0.25 * tilt_scale2))
                            k_a_ = max(0.0, k_quiet2 * (1.0 - 0.2 * tilt_scale2))
                        else:
                            k_b_ = min(1.0, k_quiet2 * (1.0 + 0.25 * tilt_scale2))
                            k_a_ = max(0.0, k_loud2 * (1.0 - 0.2 * tilt_scale2))
                    else:
                        k_b_ = k_loud2 if loud_is_b else k_quiet2
                        k_a_ = k_quiet2 if loud_is_b else k_loud2

                    rep_b_tx = b_tx * (1.0 - k_b_) + v2 * k_b_
                    rep_a_rx = a_rx * (1.0 - k_a_) + v2 * k_a_
                else:
                    rep_b_tx = v2
                    rep_a_rx = v2

            d_after_ba = rel_diff(rep_b_tx, rep_a_rx)
            base_ba = max(0.0, 1.0 - max(d_ba, d_after_ba))
            if d_ba <= STRONG_AGREE and max(b_tx, a_rx) >= LOW_RATE_MAX:
                base_ba = max(base_ba, 0.99)
            elif d_ba <= tau_ba:
                base_ba = max(base_ba, 0.98 if max(b_tx, a_rx) >= LOW_RATE_MAX else 0.97)
            change_b_tx = abs(rep_b_tx - b_tx) / max(abs(rep_b_tx), abs(b_tx), 1.0)
            change_a_rx = abs(rep_a_rx - a_rx) / max(abs(rep_a_rx), abs(a_rx), 1.0)
            tx_conf_b = min(base_ba, 1.0 - 0.8 * change_b_tx)
            rx_conf_a = min(base_ba, 1.0 - 0.8 * change_a_rx)

        # Status confidence
        if pair_status == 'down':
            if a_stat == 'down' and b_stat == 'down' and not (a_has or b_has):
                status_conf = 0.98
            else:
                status_conf = 0.7
        else:
            status_conf = 0.95 if (a_stat == 'up' and b_stat == 'up') else 0.8
            # Boost on strong bilateral agreement at high rates
            if (max(a_tx, b_rx, b_tx, a_rx) >= LOW_RATE_MAX) and \
               (rel_diff(rep_a_tx, rep_b_rx) <= STRONG_AGREE) and \
               (rel_diff(rep_b_tx, rep_a_rx) <= STRONG_AGREE):
                status_conf = max(status_conf, 0.99)

        per_if[a_id] = {
            'repaired_rx': rep_a_rx,
            'repaired_tx': rep_a_tx,
            'rx_conf': max(0.0, min(1.0, rx_conf_a)),
            'tx_conf': max(0.0, min(1.0, tx_conf_a)),
            'repaired_status': pair_status,
            'status_conf': status_conf
        }
        per_if[b_id] = {
            'repaired_rx': rep_b_rx,
            'repaired_tx': rep_b_tx,
            'rx_conf': max(0.0, min(1.0, rx_conf_b)),
            'tx_conf': max(0.0, min(1.0, tx_conf_b)),
            'repaired_status': pair_status,
            'status_conf': status_conf
        }

    # Handle interfaces without a valid peer (dangling or missing peer data)
    paired_ids = set()
    for _, (aid, bid) in pairs.items():
        paired_ids.add(aid)
        paired_ids.add(bid)

    for if_id, data in telemetry.items():
        if if_id in per_if:
            continue
        status = data.get('interface_status', 'unknown')
        rx = f64(data.get('rx_rate', 0.0))
        tx = f64(data.get('tx_rate', 0.0))
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

    # Micro-adjustment on dangling interfaces only, with commit check
    # This uses topology; without it, there's no meaningful per-router residual to optimize.
    for router, if_list in topology.items():
        present = [iid for iid in if_list if iid in per_if]
        if not present:
            continue
        sum_tx = sum(f64(per_if[iid]['repaired_tx']) for iid in present)
        sum_rx = sum(f64(per_if[iid]['repaired_rx']) for iid in present)
        delta = sum_tx - sum_rx
        scale = max(sum_tx, sum_rx, 1.0)
        resid = abs(delta) / scale

        # Trigger only on sufficiently large residuals
        if resid < 0.03 or abs(delta) <= EPS:
            continue

        # Candidates: dangling and up
        candidates: List[Tuple[str, float]] = []
        for iid in present:
            if iid in paired_ids:
                continue
            if per_if[iid].get('repaired_status', 'up') == 'down':
                continue
            contrib = f64(per_if[iid]['repaired_tx']) - f64(per_if[iid]['repaired_rx'])
            # Only those aligned with imbalance direction
            if (delta > 0 and contrib > 0) or (delta < 0 and contrib < 0):
                candidates.append((iid, contrib))

        if not candidates:
            continue

        # Dominance test: pick interface contributing most to the imbalance
        dom_iid, dom_contrib = max(candidates, key=lambda t: abs(t[1]))
        total_same_dir = sum(abs(c[1]) for c in candidates) or EPS
        if abs(dom_contrib) / total_same_dir < 0.6:
            continue

        # Tentative step: very conservative, capped at 2% of router scale
        alpha = min(0.02, 0.5 * resid)
        step = alpha * scale
        cur_tx = f64(per_if[dom_iid]['repaired_tx'])
        cur_rx = f64(per_if[dom_iid]['repaired_rx'])

        # Apply tentative nudge that should reduce residual
        new_tx, new_rx = cur_tx, cur_rx
        if delta > 0 and dom_contrib > 0:
            # Too much TX: reduce TX on this interface
            new_tx = max(0.0, cur_tx - min(step, cur_tx))
        elif delta < 0 and dom_contrib < 0:
            # Too much RX: reduce RX on this interface
            new_rx = max(0.0, cur_rx - min(step, cur_rx))
        else:
            continue  # signs mismatch, skip

        # Commit check: recompute residual; require at least 10% relative improvement
        sum_tx_new = sum_tx - cur_tx + new_tx
        sum_rx_new = sum_rx - cur_rx + new_rx
        resid_new = abs(sum_tx_new - sum_rx_new) / max(sum_tx_new, sum_rx_new, 1.0)
        if resid_new <= 0.9 * resid:
            # Commit change and penalize confidence by change magnitude
            per_if[dom_iid]['repaired_tx'] = new_tx
            per_if[dom_iid]['repaired_rx'] = new_rx
            # Confidence penalty proportional to change
            tx_orig = f64(telemetry.get(dom_iid, {}).get('tx_rate', 0.0))
            rx_orig = f64(telemetry.get(dom_iid, {}).get('rx_rate', 0.0))
            tx_change = abs(new_tx - tx_orig) / max(abs(new_tx), abs(tx_orig), 1.0)
            rx_change = abs(new_rx - rx_orig) / max(abs(new_rx), abs(rx_orig), 1.0)
            per_if[dom_iid]['tx_conf'] = max(0.0, min(per_if[dom_iid].get('tx_conf', 0.6), 1.0 - tx_change))
            per_if[dom_iid]['rx_conf'] = max(0.0, min(per_if[dom_iid].get('rx_conf', 0.6), 1.0 - rx_change))

    # Compute final router-level residuals using repaired rates
    router_residual: Dict[str, float] = {}
    router_sum_tx: Dict[str, float] = {}
    router_sum_rx: Dict[str, float] = {}
    for router, if_list in topology.items():
        s_tx = 0.0
        s_rx = 0.0
        for iid in if_list:
            if iid in per_if:
                s_tx += f64(per_if[iid]['repaired_tx'])
                s_rx += f64(per_if[iid]['repaired_rx'])
        router_sum_tx[router] = s_tx
        router_sum_rx[router] = s_rx
        router_residual[router] = abs(s_tx - s_rx) / max(s_tx, s_rx, 1.0)

    # Final assembly with interface-share–aware, direction-coupled confidence penalties
    result: Dict[str, Dict[str, Tuple]] = {}
    for if_id, data in telemetry.items():
        rep = per_if.get(if_id, {})
        repaired_rx = f64(rep.get('repaired_rx', data.get('rx_rate', 0.0)))
        repaired_tx = f64(rep.get('repaired_tx', data.get('tx_rate', 0.0)))
        repaired_status = rep.get('repaired_status', data.get('interface_status', 'unknown'))

        rx_conf = float(rep.get('rx_conf', 0.6))
        tx_conf = float(rep.get('tx_conf', 0.6))
        status_conf = float(rep.get('status_conf', 0.6))

        local_router = data.get('local_router')
        remote_router = data.get('remote_router')

        resid_local = router_residual.get(local_router, 0.0)
        resid_remote = router_residual.get(remote_router, 0.0)

        sum_tx_local = router_sum_tx.get(local_router, 0.0)
        sum_rx_local = router_sum_rx.get(local_router, 0.0)

        # Interface shares for direction-coupled penalties
        tx_share = repaired_tx / max(1.0, sum_tx_local)
        rx_share = repaired_rx / max(1.0, sum_rx_local)

        # Direction-coupled penalties with share-aware weighting
        pen_tx = 1.0 - ((0.6 + 0.2 * tx_share) * resid_local + (0.4 - 0.2 * tx_share) * resid_remote)
        pen_rx = 1.0 - ((0.6 + 0.2 * rx_share) * resid_remote + (0.4 - 0.2 * rx_share) * resid_local)
        pen_tx = max(0.0, min(1.0, pen_tx))
        pen_rx = max(0.0, min(1.0, pen_rx))

        tx_conf = max(0.0, min(1.0, tx_conf * pen_tx))
        rx_conf = max(0.0, min(1.0, rx_conf * pen_rx))
        status_conf = max(0.0, min(1.0, status_conf * (0.8 + 0.2 * min(pen_tx, pen_rx))))

        # Assemble output tuples with original, repaired, confidence
        out: Dict[str, Any] = {}
        rx_orig = f64(data.get('rx_rate', 0.0))
        tx_orig = f64(data.get('tx_rate', 0.0))
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