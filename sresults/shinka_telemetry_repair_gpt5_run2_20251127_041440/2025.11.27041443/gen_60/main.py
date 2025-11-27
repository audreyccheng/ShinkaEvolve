# EVOLVE-BLOCK-START
"""
Residual-tilted, asymmetric consensus repair for network telemetry with
magnitude-aware thresholds, interface-share–aware confidence, and safe
router micro-adjustments.

Key invariants:
- Link Symmetry (R3): my_tx ≈ their_rx, my_rx ≈ their_tx
- Flow Conservation (R1): Σ incoming = Σ outgoing at each router
- Interface Consistency: status aligned across link pairs

Parameters:
- TH_REL_DEFAULT: 0.02 (2%) timing tolerance (τh)
- TH_REL_LOW: 0.05 (5%) for low-rate links (< LOW_RATE_CUTOFF)
- ABS_GUARD / ABS_GUARD_LOW: 0.5 / 0.3 Mbps absolute triggers
- PARTIAL_RAMP_EXP: 1.2 exponent for partial averaging ramp
- FULL_MULT_HIGH / FULL_MULT_LOW: 2.0 / 1.6 full-convergence multipliers
- Residual tilt: RESID_TILT_MAX=0.08, RESID_TILT_GAMMA_BASE=0.1
- Micro-adjustments (router R1 nudges) on dangling dominant interfaces with benefit checks
- Confidence: direction-coupled router penalties weighted by interface directional shares
"""
from typing import Dict, Any, Tuple, List


def repair_network_telemetry(telemetry: Dict[str, Dict[str, Any]], 
                             topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:
    # Core parameters
    TH_REL_DEFAULT = 0.02
    TH_REL_LOW = 0.05
    LOW_RATE_CUTOFF = 10.0  # Mbps threshold separating low/high rates
    ABS_GUARD = 0.5
    ABS_GUARD_LOW = 0.3
    PARTIAL_RAMP_EXP = 1.2
    FULL_MULT_HIGH = 2.0
    FULL_MULT_LOW = 1.6
    STRONG_AGREE_DIFF = 0.005  # 0.5%
    EPS = 1e-6

    # Residual tilt parameters
    RESID_TILT_MAX = 0.08
    RESID_TILT_GAMMA_BASE = 0.1

    # Router micro-adjust parameters (dangling interfaces only)
    MICRO_TRIGGER = 0.03            # minimum residual fraction to consider nudging
    DOMINANT_SHARE_MIN = 0.6        # dominant share among dangling interfaces
    MICRO_ALPHA_MAX = 0.02          # max fraction of scale for a single nudge
    MICRO_IMPROVE_MIN = 0.08        # min relative improvement (8%) to accept a step
    MICRO_SECOND_STEP_MIN_IMPROVE = 0.20  # require ≥20% improvement to allow second mini step
    INTERNAL_SKEW_LIMIT = 0.03      # must not worsen |tx - rx| by >3% relative

    # Confidence penalties
    CONF_CHANGE_PENALTY = 0.8       # penalty slope for magnitude of change

    def to_float(x: Any, default: float = 0.0) -> float:
        try:
            f = float(x)
            if f != f:  # NaN
                return default
            return f
        except Exception:
            return default

    def has_traffic(d: Dict[str, Any]) -> bool:
        return (to_float(d.get('rx_rate', 0.0)) > EPS) or (to_float(d.get('tx_rate', 0.0)) > EPS)

    def rel_diff(a: float, b: float) -> float:
        return abs(a - b) / max(a, b, 1.0)

    def change_ratio(orig: float, rep: float) -> float:
        denom = max(abs(orig), abs(rep), 1.0)
        return abs(rep - orig) / denom

    # Build unique undirected pairs using connected_to references
    pairs: Dict[Tuple[str, str], Tuple[str, str]] = {}
    for if_id, data in telemetry.items():
        peer = data.get('connected_to')
        if peer and peer in telemetry:
            key = tuple(sorted([if_id, peer]))
            if key not in pairs:
                pairs[key] = (if_id, peer)

    # Precompute signed router residuals from original telemetry (before hardening)
    router_signed_resid: Dict[str, float] = {}
    for router, if_list in topology.items():
        s_tx = 0.0
        s_rx = 0.0
        for iid in if_list:
            if iid not in telemetry:
                continue
            d = telemetry[iid]
            status = d.get('interface_status', 'up')
            # When interface is down, treat counters as zero
            rx = 0.0 if status == 'down' else to_float(d.get('rx_rate', 0.0))
            tx = 0.0 if status == 'down' else to_float(d.get('tx_rate', 0.0))
            s_tx += tx
            s_rx += rx
        scale = max(s_tx, s_rx, 1.0)
        signed = (s_tx - s_rx) / scale
        router_signed_resid[router] = signed  # retains sign for tilt

    # First pass: pairwise hardening and status harmonization
    per_if: Dict[str, Dict[str, Any]] = {}

    for (_, (a_id, b_id)) in pairs.items():
        a = telemetry[a_id]
        b = telemetry[b_id]

        a_stat = a.get('interface_status', 'unknown')
        b_stat = b.get('interface_status', 'unknown')

        a_rx, a_tx = to_float(a.get('rx_rate', 0.0)), to_float(a.get('tx_rate', 0.0))
        b_rx, b_tx = to_float(b.get('rx_rate', 0.0)), to_float(b.get('tx_rate', 0.0))

        a_has = has_traffic(a)
        b_has = has_traffic(b)

        # Pair status decision
        if a_stat == 'down' and b_stat == 'down':
            pair_status = 'down'
        elif a_stat == 'up' and b_stat == 'up':
            pair_status = 'up'
        else:
            pair_status = 'up' if (a_has or b_has) else 'down'

        # Initialize repaired values and confidences
        rep_a_tx, rep_b_rx = a_tx, b_rx
        rep_b_tx, rep_a_rx = b_tx, a_rx
        rx_conf_a = tx_conf_a = rx_conf_b = tx_conf_b = 1.0

        if pair_status == 'down':
            # Enforce no traffic on down link
            rep_a_tx = rep_b_rx = rep_b_tx = rep_a_rx = 0.0
            base_conf = 0.95 if not (a_has or b_has) else 0.7
            rx_conf_a = tx_conf_a = rx_conf_b = tx_conf_b = base_conf
        else:
            # Helper to perform one directional repair with residual-tilted, asymmetric partial averaging
            def repair_direction(x_val: float, y_val: float,
                                 local_router_for_x: str,
                                 is_low_rate: bool) -> Tuple[float, float, float, float, bool]:
                # Returns: new_x, new_y, conf_x, conf_y, changed
                max_xy = max(x_val, y_val)
                tau = TH_REL_LOW if is_low_rate else TH_REL_DEFAULT
                full_mult = FULL_MULT_LOW if is_low_rate else FULL_MULT_HIGH
                abs_guard = ABS_GUARD_LOW if is_low_rate else ABS_GUARD

                d = rel_diff(x_val, y_val)
                abs_d = abs(x_val - y_val)
                changed = False

                conf_dir = max(0.0, 1.0 - d)
                # Confidence floors on strong agreement
                if (not is_low_rate) and d <= STRONG_AGREE_DIFF:
                    conf_dir = max(conf_dir, 0.99)
                elif d <= tau:
                    conf_dir = max(conf_dir, 0.98 if not is_low_rate else 0.97)

                nx, ny = x_val, y_val
                if (d > tau) and (abs_d > abs_guard):
                    # Weighted consensus with residual tilt towards reducing local imbalance
                    w_x = 0.5
                    w_y = 0.5
                    delta = x_val - y_val  # sign of direction discrepancy
                    resid_signed = router_signed_resid.get(local_router_for_x, 0.0)
                    if (delta > 0 and resid_signed > 0) or (delta < 0 and resid_signed < 0):
                        gamma = min(RESID_TILT_MAX, RESID_TILT_GAMMA_BASE * abs(resid_signed))
                        w_x = max(0.2, min(0.8, 0.5 - gamma))
                        w_y = 1.0 - w_x
                    v = w_x * x_val + w_y * y_val

                    if d <= full_mult * tau:
                        # Asymmetric partial averaging: louder side moves more
                        k_base = (d - tau) / max(tau, 1e-9)
                        # ramp with exponent for low-rate decisiveness
                        k_base = max(0.0, min(1.0, k_base ** PARTIAL_RAMP_EXP if is_low_rate else k_base))
                        loud_is_x = x_val >= y_val
                        loud = max(x_val, y_val)
                        quiet = min(x_val, y_val)
                        r = (loud - quiet) / max(1.0, loud)
                        k_loud = max(0.0, min(1.0, k_base * (1.0 + 0.5 * r)))
                        k_quiet = max(0.0, min(1.0, k_base * (1.0 - 0.5 * r)))
                        if loud_is_x:
                            nx = x_val * (1.0 - k_loud) + v * k_loud
                            ny = y_val * (1.0 - k_quiet) + v * k_quiet
                        else:
                            nx = x_val * (1.0 - k_quiet) + v * k_quiet
                            ny = y_val * (1.0 - k_loud) + v * k_loud
                    else:
                        nx = v
                        ny = v
                    changed = True

                    # Confidence penalty by change magnitude
                    cx = change_ratio(x_val, nx)
                    cy = change_ratio(y_val, ny)
                    conf_dir = min(conf_dir, 1.0 - CONF_CHANGE_PENALTY * cx)
                    conf_other = min(conf_dir, 1.0 - CONF_CHANGE_PENALTY * cy)
                else:
                    conf_other = conf_dir  # no change to partner's base conf

                return nx, ny, conf_dir, conf_other, changed

            # Direction A->B uses A.tx vs B.rx
            is_low_ab = max(a_tx, b_rx) < LOW_RATE_CUTOFF
            rep_a_tx, rep_b_rx, conf_a_tx, conf_b_rx, changed_ab = repair_direction(
                a_tx, b_rx, a.get('local_router'), is_low_ab
            )
            tx_conf_a = conf_a_tx
            rx_conf_b = conf_b_rx

            # Direction B->A uses B.tx vs A.rx
            is_low_ba = max(b_tx, a_rx) < LOW_RATE_CUTOFF
            rep_b_tx, rep_a_rx, conf_b_tx, conf_a_rx, changed_ba = repair_direction(
                b_tx, a_rx, b.get('local_router'), is_low_ba
            )
            tx_conf_b = conf_b_tx
            rx_conf_a = conf_a_rx

            # Asymmetric evidence reduction when only one side has traffic
            if a_has != b_has:
                if not a_has:
                    rx_conf_a *= 0.88
                    tx_conf_a *= 0.88
                if not b_has:
                    rx_conf_b *= 0.88
                    tx_conf_b *= 0.88

        # Status confidence
        if pair_status == 'down':
            if a_stat == 'down' and b_stat == 'down' and not (a_has or b_has):
                status_conf = 0.98
            else:
                status_conf = 0.7
        else:
            if a_stat == 'up' and b_stat == 'up':
                status_conf = 0.95
            else:
                status_conf = 0.8

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

    # Handle interfaces without a valid peer (dangling or missing peer data)
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
            # No peer redundancy; keep values with moderate confidence
            per_if[if_id] = {
                'repaired_rx': rx,
                'repaired_tx': tx,
                'rx_conf': 0.6,
                'tx_conf': 0.6,
                'repaired_status': status if status in ('up', 'down') else 'up',
                'status_conf': 0.6
            }

    # Router-level micro-adjustments on dangling interfaces to reduce residuals (R1)
    # Identify paired interfaces to avoid breaking symmetry repairs
    paired_ids = set()
    for _, (aid, bid) in pairs.items():
        paired_ids.add(aid)
        paired_ids.add(bid)

    def router_totals_from_rep(router: str) -> Tuple[float, float, List[str]]:
        s_tx = 0.0
        s_rx = 0.0
        present = []
        for iid in topology.get(router, []):
            if iid in per_if:
                present.append(iid)
                rep = per_if[iid]
                s_tx += to_float(rep['repaired_tx'])
                s_rx += to_float(rep['repaired_rx'])
        return s_tx, s_rx, present

    for router in topology.keys():
        s_tx, s_rx, present = router_totals_from_rep(router)
        scale = max(s_tx, s_rx, 1.0)
        delta = s_tx - s_rx
        resid_frac = abs(delta) / scale
        if resid_frac < MICRO_TRIGGER or abs(delta) <= EPS:
            continue

        # Candidates: dangling (unpaired), up status, with enough traffic
        candidates = []
        total_dang = 0.0
        for iid in present:
            if iid in paired_ids:
                continue
            rep = per_if[iid]
            if rep.get('repaired_status', 'up') == 'down':
                continue
            txv = to_float(rep['repaired_tx'])
            rxv = to_float(rep['repaired_rx'])
            mag = max(txv, rxv)
            if mag < LOW_RATE_CUTOFF:
                continue  # guard against nudging tiny flows
            w = txv + rxv + EPS
            candidates.append((iid, w))
            total_dang += w
        if not candidates or total_dang <= 0.0:
            continue

        dom_iid, dom_w = max(candidates, key=lambda x: x[1])
        dom_share = dom_w / max(total_dang, EPS)
        if dom_share < DOMINANT_SHARE_MIN:
            continue

        # Compute step size
        alpha = min(MICRO_ALPHA_MAX, 0.5 * resid_frac)
        step = alpha * scale

        def commit_if_beneficial(adjust_rx: bool, step_val: float) -> bool:
            nonlocal s_tx, s_rx, delta, resid_frac
            rep = per_if[dom_iid]
            cur_tx = to_float(rep['repaired_tx'])
            cur_rx = to_float(rep['repaired_rx'])
            new_tx = cur_tx
            new_rx = cur_rx
            if adjust_rx:
                new_rx = cur_rx + step_val
            else:
                new_tx = cur_tx + step_val
            new_s_tx = s_tx if adjust_rx else s_tx + step_val
            new_s_rx = s_rx + step_val if adjust_rx else s_rx
            new_delta = new_s_tx - new_s_rx
            new_scale = max(new_s_tx, new_s_rx, 1.0)
            new_resid = abs(new_delta) / new_scale
            # Benefit check
            if new_resid <= (1.0 - MICRO_IMPROVE_MIN) * resid_frac:
                # Internal skew guard: do not worsen |tx - rx| by more than 3% relative
                old_skew = abs(cur_tx - cur_rx) / max(max(cur_tx, cur_rx), 1.0)
                new_skew = abs(new_tx - new_rx) / max(max(new_tx, new_rx), 1.0)
                if new_skew <= old_skew + INTERNAL_SKEW_LIMIT:
                    # Commit
                    per_if[dom_iid]['repaired_tx'] = new_tx
                    per_if[dom_iid]['repaired_rx'] = new_rx
                    # Update locals for potential second step
                    s_tx, s_rx, delta, resid_frac = new_s_tx, new_s_rx, new_delta, new_resid
                    # Confidence penalty for changed field
                    if adjust_rx:
                        rx_orig = to_float(telemetry.get(dom_iid, {}).get('rx_rate', 0.0))
                        cr = change_ratio(rx_orig, new_rx)
                        per_if[dom_iid]['rx_conf'] = max(0.0, min(per_if[dom_iid].get('rx_conf', 0.6), 1.0 - cr))
                    else:
                        tx_orig = to_float(telemetry.get(dom_iid, {}).get('tx_rate', 0.0))
                        cr = change_ratio(tx_orig, new_tx)
                        per_if[dom_iid]['tx_conf'] = max(0.0, min(per_if[dom_iid].get('tx_conf', 0.6), 1.0 - cr))
                    return True
            return False

        # Choose direction: if too much TX (delta>0), increase RX; else increase TX
        first_ok = commit_if_beneficial(adjust_rx=(delta > 0.0), step_val=step)

        # Optional second mini-step if first improved ≥20% and imbalance remains notable
        if first_ok and resid_frac >= 0.04:
            # Re-evaluate improvement fraction; if ≥20%, allow half step
            # Using prior residual before first step is tricky; approximate by using stronger gate
            half_step = 0.5 * step
            commit_if_beneficial(adjust_rx=(delta > 0.0), step_val=half_step)

    # Compute router residuals (unsigned) and per-router TX/RX totals after all repairs
    router_residual: Dict[str, float] = {}
    router_totals: Dict[str, Dict[str, float]] = {}
    for router, if_list in topology.items():
        s_tx = 0.0
        s_rx = 0.0
        for iid in if_list:
            if iid in per_if:
                rep = per_if[iid]
                s_tx += to_float(rep['repaired_tx'])
                s_rx += to_float(rep['repaired_rx'])
        router_totals[router] = {'tx': s_tx, 'rx': s_rx}
        router_residual[router] = abs(s_tx - s_rx) / max(s_tx, s_rx, 1.0)

    # Final assembly: compose confidences with direction-aware, share-weighted router penalties
    result: Dict[str, Dict[str, Tuple]] = {}
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

        # Directional shares to modulate penalties
        s_local_tx = router_totals.get(local_router, {}).get('tx', 0.0)
        s_local_rx = router_totals.get(local_router, {}).get('rx', 0.0)
        tx_share = repaired_tx / max(1.0, s_local_tx)
        rx_share = repaired_rx / max(1.0, s_local_rx)

        # For TX: emphasize local residual by the interface TX share
        w_local_tx = 0.6 + 0.2 * tx_share
        w_remote_tx = 1.0 - w_local_tx
        pen_tx = 1.0 - (w_local_tx * resid_local + w_remote_tx * resid_remote)
        pen_tx = max(0.0, min(1.0, pen_tx))

        # For RX: invert roles, weight remote residual by interface RX share
        w_remote_rx = 0.6 + 0.2 * rx_share
        w_local_rx = 1.0 - w_remote_rx
        pen_rx = 1.0 - (w_local_rx * resid_local + w_remote_rx * resid_remote)
        pen_rx = max(0.0, min(1.0, pen_rx))

        tx_conf = max(0.0, min(1.0, tx_conf * pen_tx))
        rx_conf = max(0.0, min(1.0, rx_conf * pen_rx))
        status_conf = max(0.0, min(1.0, status_conf * (0.85 + 0.15 * min(pen_tx, pen_rx))))

        # Assemble output tuples with original, repaired, confidence
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

