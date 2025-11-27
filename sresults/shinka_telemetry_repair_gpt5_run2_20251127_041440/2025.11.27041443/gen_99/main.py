# EVOLVE-BLOCK-START
"""
Robust bi-directional consensus repair for network telemetry with
magnitude-aware thresholds and direction-aware confidence penalties.

Principles:
- Link Symmetry (R3): my_tx ≈ their_rx and my_rx ≈ their_tx
- Flow Conservation (R1): Σ incoming = Σ outgoing per router (used for confidence)
- Interface Consistency: paired statuses aligned; down links carry no traffic

Enhancements:
- Magnitude-aware tolerance and absolute guard (0.5 Mbps) to avoid over-correction on tiny flows.
- Strong agreement confidence floors (≥0.99 on clear high-rate agreement).
- Direction-aware router residual penalties for RX/TX confidences.
"""
from typing import Dict, Any, Tuple, List


def repair_network_telemetry(telemetry: Dict[str, Dict[str, Any]],
                             topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:
    # Base thresholds
    TH_REL_DEFAULT = 0.02   # 2%
    TH_REL_LOW = 0.05       # 5% for low-rate links
    TH_ABS_GUARD = 0.5      # 0.5 Mbps absolute guard
    ABS_GUARD_LOW = 0.3     # 0.3 Mbps absolute guard for low-rate links
    LOW_RATE_CUTOFF = 10.0  # Mbps threshold for low-rate behavior
    # One-sided trust (when one side ~0 and peer is high)
    ONE_SIDED_FRAC = 0.05      # near-zero if smaller side <= 5% of larger
    ONE_SIDED_ABS_MULT = 2.0   # require abs diff > 2x guard to trust peer
    # Gentle prescaling clamp for bias reduction (used only for consensus/k computations)
    PRESCALE_MIN = 0.90
    PRESCALE_MAX = 1.10
    # Ultra-agreement confidence floor thresholds
    ULTRA_DIFF_FLOOR = 0.003       # 0.3% relative diff
    RESID_AGREE_THRESH = 0.02      # 2% router residual
    EPS = 1e-6

    def to_float(x: Any) -> float:
        try:
            return float(x)
        except Exception:
            return 0.0

    def has_traffic(d: Dict[str, Any]) -> bool:
        return (to_float(d.get('rx_rate', 0.0)) > EPS) or (to_float(d.get('tx_rate', 0.0)) > EPS)

    def rel_diff(a: float, b: float) -> float:
        return abs(a - b) / max(a, b, 1.0)

    def clamp(x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))

    def sgn(x: float) -> int:
        return 1 if x > 0 else (-1 if x < 0 else 0)

    # Build unique undirected pairs
    pairs: Dict[Tuple[str, str], Tuple[str, str]] = {}
    for if_id, data in telemetry.items():
        peer = data.get('connected_to')
        if peer and peer in telemetry:
            key = tuple(sorted([if_id, peer]))
            if key not in pairs:
                pairs[key] = (if_id, peer)

    # Compute signed router residuals from original telemetry (down links contribute zero)
    resid_signed: Dict[str, float] = {}
    for router, if_list in topology.items():
        s_tx = 0.0
        s_rx = 0.0
        for iid in if_list:
            d = telemetry.get(iid)
            if not d:
                continue
            status = d.get('interface_status', 'unknown')
            tx = to_float(d.get('tx_rate', 0.0))
            rx = to_float(d.get('rx_rate', 0.0))
            if status == 'down':
                tx = 0.0
                rx = 0.0
            s_tx += tx
            s_rx += rx
        scale = max(s_tx, s_rx, 1.0)
        resid_signed[router] = (s_tx - s_rx) / scale if scale > 0 else 0.0

    # First pass: pairwise consensus hardening and status harmonization
    per_if: Dict[str, Dict[str, Any]] = {}

    for _, (a_id, b_id) in pairs.items():
        a = telemetry[a_id]
        b = telemetry[b_id]

        a_stat = a.get('interface_status', 'unknown')
        b_stat = b.get('interface_status', 'unknown')

        a_rx, a_tx = to_float(a.get('rx_rate', 0.0)), to_float(a.get('tx_rate', 0.0))
        b_rx, b_tx = to_float(b.get('rx_rate', 0.0)), to_float(b.get('tx_rate', 0.0))

        # Decide link status
        a_has = has_traffic(a)
        b_has = has_traffic(b)
        if a_stat == 'down' and b_stat == 'down':
            pair_status = 'down'
        elif a_stat == 'up' and b_stat == 'up':
            pair_status = 'up'
        else:
            pair_status = 'up' if (a_has or b_has) else 'down'

        # Initialize repaired values
        rep_a_tx, rep_b_rx = a_tx, b_rx
        rep_b_tx, rep_a_rx = b_tx, a_rx
        rx_conf_a = 1.0
        tx_conf_a = 1.0
        rx_conf_b = 1.0
        tx_conf_b = 1.0

        if pair_status == 'down':
            # Enforce no traffic on down link
            rep_a_tx = rep_b_rx = rep_b_tx = rep_a_rx = 0.0
            base_conf = 0.95 if not (a_has or b_has) else 0.7
            rx_conf_a = tx_conf_a = rx_conf_b = tx_conf_b = base_conf
        else:
            # Magnitude-aware thresholds with prescaling, one-sided trust, residual-tilted consensus,
            # and asymmetric partial averaging
            # Direction A->B (A.tx vs B.rx)
            max_ab = max(a_tx, b_rx)
            thr_rel_ab = TH_REL_LOW if max_ab < LOW_RATE_CUTOFF else TH_REL_DEFAULT
            guard_ab = ABS_GUARD_LOW if max_ab < LOW_RATE_CUTOFF else TH_ABS_GUARD
            d_ab = rel_diff(a_tx, b_rx)
            abs_ab = abs(a_tx - b_rx)
            partial_upper_ab = (1.6 * thr_rel_ab) if max_ab < LOW_RATE_CUTOFF else (2.0 * thr_rel_ab)

            if (d_ab > thr_rel_ab) and (abs_ab > guard_ab):
                # One-sided trust: when one side is near-zero and peer is high (avoid averaging)
                min_ab = min(a_tx, b_rx)
                if (max_ab >= LOW_RATE_CUTOFF and
                    min_ab <= ONE_SIDED_FRAC * max_ab and
                    abs_ab > ONE_SIDED_ABS_MULT * guard_ab):
                    v_ab = max(a_tx, b_rx)
                    rep_a_tx = v_ab
                    rep_b_rx = v_ab
                    conf_base = max(0.0, 1.0 - 0.5 * d_ab)
                    change_a = abs(rep_a_tx - a_tx) / max(abs(rep_a_tx), abs(a_tx), 1.0)
                    change_b = abs(rep_b_rx - b_rx) / max(abs(rep_b_rx), abs(b_rx), 1.0)
                    tx_conf_a = min(0.95, max(0.65, conf_base, 1.0 - 0.8 * change_a))
                    rx_conf_b = min(0.95, max(0.65, conf_base, 1.0 - 0.8 * change_b))
                else:
                    # Gentle prescaling to reduce ratio bias (used only for consensus/k calculations)
                    s_ab = (max(EPS, b_rx) / max(EPS, a_tx)) ** 0.5
                    s_ab = clamp(s_ab, PRESCALE_MIN, PRESCALE_MAX)
                    a_tx_s = a_tx * s_ab
                    b_rx_s = b_rx / s_ab
                    d_ab_s = rel_diff(a_tx_s, b_rx_s)

                    # Residual-tilted weighted consensus target computed on prescaled values
                    v = 0.5 * (a_tx_s + b_rx_s)
                    resid_a = resid_signed.get(a.get('local_router'), 0.0)
                    if max_ab >= 1.0 and (a_tx_s - b_rx_s) != 0.0 and resid_a != 0.0 and (sgn(a_tx_s - b_rx_s) == sgn(resid_a)):
                        gamma = min(0.08, 0.1 * abs(resid_a))
                        w_a = clamp(0.5 - gamma, 0.2, 0.8)
                        w_b = 1.0 - w_a
                        v = w_a * a_tx_s + w_b * b_rx_s

                    if d_ab_s <= partial_upper_ab:
                        # Asymmetric partial averaging: louder side moves more; low-rate slightly steeper ramp
                        k_base = (d_ab_s - thr_rel_ab) / max(thr_rel_ab, 1e-9)
                        k_base = min(1.0, max(0.0, k_base))
                        if max_ab < LOW_RATE_CUTOFF:
                            k_base = k_base ** 1.2
                        if a_tx_s >= b_rx_s:
                            loud_s, quiet_s, loud_is_a = a_tx_s, b_rx_s, True
                        else:
                            loud_s, quiet_s, loud_is_a = b_rx_s, a_tx_s, False
                        r = (loud_s - quiet_s) / max(1.0, loud_s)
                        k_loud = min(1.0, max(0.0, k_base * (1.0 + 0.5 * r)))
                        k_quiet = min(1.0, max(0.0, k_base * (1.0 - 0.5 * r)))
                        if loud_is_a:
                            rep_a_tx = a_tx * (1.0 - k_loud) + v * k_loud
                            rep_b_rx = b_rx * (1.0 - k_quiet) + v * k_quiet
                        else:
                            rep_a_tx = a_tx * (1.0 - k_quiet) + v * k_quiet
                            rep_b_rx = b_rx * (1.0 - k_loud) + v * k_loud
                    else:
                        rep_a_tx = v
                        rep_b_rx = v

                    # Confidence decreases with raw violation magnitude and applied change
                    change_a = abs(rep_a_tx - a_tx) / max(abs(rep_a_tx), abs(a_tx), 1.0)
                    change_b = abs(rep_b_rx - b_rx) / max(abs(rep_b_rx), abs(b_rx), 1.0)
                    conf_base = max(0.0, 1.0 - d_ab)
                    tx_conf_a = min(conf_base, 1.0 - 0.8 * change_a)
                    rx_conf_b = min(conf_base, 1.0 - 0.8 * change_b)
            else:
                # Within tolerance: keep values and set strong confidence floors (using raw diff)
                if max_ab >= LOW_RATE_CUTOFF and d_ab <= 0.005:  # 0.5%
                    conf_floor = 0.99
                else:
                    conf_floor = 0.98 if max_ab >= LOW_RATE_CUTOFF else 0.97
                tx_conf_a = max(tx_conf_a, conf_floor)
                rx_conf_b = max(rx_conf_b, conf_floor)

            # Direction B->A (B.tx vs A.rx)
            max_ba = max(b_tx, a_rx)
            thr_rel_ba = TH_REL_LOW if max_ba < LOW_RATE_CUTOFF else TH_REL_DEFAULT
            guard_ba = ABS_GUARD_LOW if max_ba < LOW_RATE_CUTOFF else TH_ABS_GUARD
            d_ba = rel_diff(b_tx, a_rx)
            abs_ba = abs(b_tx - a_rx)
            partial_upper_ba = (1.6 * thr_rel_ba) if max_ba < LOW_RATE_CUTOFF else (2.0 * thr_rel_ba)

            if (d_ba > thr_rel_ba) and (abs_ba > guard_ba):
                min_ba = min(b_tx, a_rx)
                if (max_ba >= LOW_RATE_CUTOFF and
                    min_ba <= ONE_SIDED_FRAC * max_ba and
                    abs_ba > ONE_SIDED_ABS_MULT * guard_ba):
                    v_ba = max(b_tx, a_rx)
                    rep_b_tx = v_ba
                    rep_a_rx = v_ba
                    conf_base2 = max(0.0, 1.0 - 0.5 * d_ba)
                    change_b2 = abs(rep_b_tx - b_tx) / max(abs(rep_b_tx), abs(b_tx), 1.0)
                    change_a2 = abs(rep_a_rx - a_rx) / max(abs(rep_a_rx), abs(a_rx), 1.0)
                    tx_conf_b = min(0.95, max(0.65, conf_base2, 1.0 - 0.8 * change_b2))
                    rx_conf_a = min(0.95, max(0.65, conf_base2, 1.0 - 0.8 * change_a2))
                else:
                    # Prescaling (used only for consensus/k calculations)
                    s_ba = (max(EPS, a_rx) / max(EPS, b_tx)) ** 0.5
                    s_ba = clamp(s_ba, PRESCALE_MIN, PRESCALE_MAX)
                    b_tx_s = b_tx * s_ba
                    a_rx_s = a_rx / s_ba
                    d_ba_s = rel_diff(b_tx_s, a_rx_s)

                    v2 = 0.5 * (b_tx_s + a_rx_s)
                    resid_b = resid_signed.get(b.get('local_router'), 0.0)
                    if max_ba >= 1.0 and (b_tx_s - a_rx_s) != 0.0 and resid_b != 0.0 and (sgn(b_tx_s - a_rx_s) == sgn(resid_b)):
                        gamma2 = min(0.08, 0.1 * abs(resid_b))
                        w_b2 = clamp(0.5 - gamma2, 0.2, 0.8)
                        w_a2 = 1.0 - w_b2
                        v2 = w_b2 * b_tx_s + w_a2 * a_rx_s

                    if d_ba_s <= partial_upper_ba:
                        k_base2 = (d_ba_s - thr_rel_ba) / max(thr_rel_ba, 1e-9)
                        k_base2 = min(1.0, max(0.0, k_base2))
                        if max_ba < LOW_RATE_CUTOFF:
                            k_base2 = k_base2 ** 1.2
                        if b_tx_s >= a_rx_s:
                            loud2_s, quiet2_s, loud_is_b = b_tx_s, a_rx_s, True
                        else:
                            loud2_s, quiet2_s, loud_is_b = a_rx_s, b_tx_s, False
                        r2 = (loud2_s - quiet2_s) / max(1.0, loud2_s)
                        k_loud2 = min(1.0, max(0.0, k_base2 * (1.0 + 0.5 * r2)))
                        k_quiet2 = min(1.0, max(0.0, k_base2 * (1.0 - 0.5 * r2)))
                        if loud_is_b:
                            rep_b_tx = b_tx * (1.0 - k_loud2) + v2 * k_loud2
                            rep_a_rx = a_rx * (1.0 - k_quiet2) + v2 * k_quiet2
                        else:
                            rep_b_tx = b_tx * (1.0 - k_quiet2) + v2 * k_quiet2
                            rep_a_rx = a_rx * (1.0 - k_loud2) + v2 * k_loud2
                    else:
                        rep_b_tx = v2
                        rep_a_rx = v2

                    change_b2 = abs(rep_b_tx - b_tx) / max(abs(rep_b_tx), abs(b_tx), 1.0)
                    change_a2 = abs(rep_a_rx - a_rx) / max(abs(rep_a_rx), abs(a_rx), 1.0)
                    conf_base2 = max(0.0, 1.0 - d_ba)
                    tx_conf_b = min(conf_base2, 1.0 - 0.8 * change_b2)
                    rx_conf_a = min(conf_base2, 1.0 - 0.8 * change_a2)
            else:
                if max_ba >= LOW_RATE_CUTOFF and d_ba <= 0.005:
                    conf_floor2 = 0.99
                else:
                    conf_floor2 = 0.98 if max_ba >= LOW_RATE_CUTOFF else 0.97
                tx_conf_b = max(tx_conf_b, conf_floor2)
                rx_conf_a = max(rx_conf_a, conf_floor2)

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
                # Asymmetric confidence when "up" is evidence-driven and only one side has traffic
                if a_has != b_has:
                    if not a_has:
                        rx_conf_a *= 0.88
                        tx_conf_a *= 0.88
                    if not b_has:
                        rx_conf_b *= 0.88
                        tx_conf_b *= 0.88
            # Boost status confidence on strong bilateral agreement at high rates
            if (max(max_ab, max_ba) >= 10.0) and (d_ab <= 0.005) and (d_ba <= 0.005):
                status_conf = max(status_conf, 0.99)

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

    # Handle dangling interfaces (no valid peer in telemetry)
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

    # Compute router-level residuals (R1) using repaired rates
    router_residual: Dict[str, float] = {}
    for router, if_list in topology.items():
        s_tx = 0.0
        s_rx = 0.0
        for iid in if_list:
            if iid in per_if:
                rep = per_if[iid]
                s_tx += to_float(rep['repaired_tx'])
                s_rx += to_float(rep['repaired_rx'])
        router_residual[router] = abs(s_tx - s_rx) / max(s_tx, s_rx, 1.0)

    # Final assembly with direction-aware confidence penalties
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

        # Direction-aware penalties
        penalty_tx = 1.0 - (0.6 * resid_local + 0.4 * resid_remote)
        penalty_rx = 1.0 - (0.6 * resid_remote + 0.4 * resid_local)
        penalty_tx = max(0.0, min(1.0, penalty_tx))
        penalty_rx = max(0.0, min(1.0, penalty_rx))
        avg_penalty = 0.5 * (penalty_tx + penalty_rx)
        min_penalty = min(penalty_tx, penalty_rx)

        tx_conf = max(0.0, min(1.0, tx_conf * penalty_tx))
        rx_conf = max(0.0, min(1.0, rx_conf * penalty_rx))
        status_conf = max(0.0, min(1.0, status_conf * (0.85 + 0.15 * min_penalty)))

        # Ultra-agreement floor: if both directions on this link are tightly aligned post-repair
        # and both routers have very low residuals, raise confidences to at least 0.995.
        peer_id = data.get('connected_to')
        if peer_id and (peer_id in per_if) and (if_id in per_if):
            try:
                rep_tx_local = repaired_tx
                rep_rx_local = repaired_rx
                rep_rx_peer = float(per_if[peer_id]['repaired_rx'])
                rep_tx_peer = float(per_if[peer_id]['repaired_tx'])
                d1 = rel_diff(rep_tx_local, rep_rx_peer)
                d2 = rel_diff(rep_tx_peer, rep_rx_local)
                if (d1 <= ULTRA_DIFF_FLOOR) and (d2 <= ULTRA_DIFF_FLOOR) and (resid_local <= RESID_AGREE_THRESH) and (resid_remote <= RESID_AGREE_THRESH):
                    tx_conf = max(tx_conf, 0.995)
                    rx_conf = max(rx_conf, 0.995)
                    status_conf = max(status_conf, 0.995)
            except Exception:
                pass

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