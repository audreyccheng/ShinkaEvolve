# EVOLVE-BLOCK-START
"""
Network telemetry repair with residual-tilted asymmetric consensus, dynamic guards,
share-aware confidence penalties, and safeguarded micro-adjustments.

Core invariants:
- Link Symmetry (R3): my_tx ≈ their_rx and my_rx ≈ their_tx
- Flow Conservation (R1): Σ incoming = Σ outgoing per router
- Interface Consistency: paired statuses aligned; down links carry no traffic

Enhancements in this version:
- Residual-tilted, direction-aware consensus weights during pair repair
- Asymmetric partial averaging that moves the louder side more
- Low-rate sensitive absolute guard and ramp for decisive convergence
- Interface-share–aware, direction-coupled router residual penalties for confidence
- Safer, benefit-checked micro-adjustments on dominant unpaired interfaces
"""
from typing import Dict, Any, Tuple, List
import math


def repair_network_telemetry(telemetry: Dict[str, Dict[str, Any]],
                             topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:
    # ---------------------------
    # Configuration parameters
    # ---------------------------
    # Symmetry thresholds
    TH_REL_DEFAULT = 0.02       # 2% default tolerance
    TH_REL_LOW = 0.05           # 5% tolerance for low-rate links
    LOW_RATE_CUTOFF = 10.0      # Mbps threshold for "low" rate
    STRONG_AGREE_DIFF = 0.005   # 0.5% strong agreement floor
    ABS_GUARD_LOW = 0.3         # Mbps absolute guard for low-rate links
    ABS_GUARD_HIGH = 0.5        # Mbps absolute guard otherwise
    FULL_FACTOR_DEFAULT = 2.0   # full convergence trigger factor for regular rates
    FULL_FACTOR_LOW = 1.6       # tighter full convergence for low-rate links
    EPS = 1e-6                  # numeric floor

    # Residual-tilting
    TILT_GAMMA_MAX = 0.08       # maximum tilt away from local endpoint
    TILT_GAMMA_SCALE = 0.1      # gamma = min(0.08, 0.1 * |resid_local_frac|)

    # Router penalty for status scaling (mild)
    STATUS_PENALTY_BLEND = 0.25

    # Micro-adjustments for dangling interfaces (safeguarded)
    MICRO_ACTIVATE_RESID = 0.03  # require ≥3% residual
    MICRO_MAX_ALPHA = 0.02       # maximum first-step change
    MICRO_SECOND_ALPHA = 0.01    # maximum second-step change
    MICRO_ALPHA_SCALE = 0.5      # α = min(0.02, 0.5 * resid_frac)
    MICRO_REQ_IMPROVE = 0.08     # ≥8% improvement required to commit
    MICRO_REQ_IMPROVE_SECOND = 0.20  # ≥20% improvement to allow second step
    MICRO_RESID_FOR_SECOND = 0.04    # residual still ≥4% to consider second step
    MICRO_SKEW_LIMIT = 0.03      # do not increase |tx-rx| by more than 3%
    MICRO_RATE_FLOOR = LOW_RATE_CUTOFF  # only adjust if iface max(tx,rx) ≥ LOW_RATE_CUTOFF

    # ---------------------------
    # Helper functions
    # ---------------------------
    def to_float(x: Any) -> float:
        try:
            return float(x)
        except Exception:
            return 0.0

    def clamp(x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, float(x)))

    def clamp01(x: float) -> float:
        return clamp(x, 0.0, 1.0)

    def has_traffic(d: Dict[str, Any]) -> bool:
        return (to_float(d.get('rx_rate', 0.0)) > EPS) or (to_float(d.get('tx_rate', 0.0)) > EPS)

    def rel_diff(a: float, b: float) -> float:
        return abs(a - b) / max(a, b, 1.0)

    def dynamic_tau(a: float, b: float) -> float:
        return TH_REL_LOW if max(a, b) < LOW_RATE_CUTOFF else TH_REL_DEFAULT

    def dynamic_abs_guard(a: float, b: float) -> float:
        return ABS_GUARD_LOW if max(a, b) < LOW_RATE_CUTOFF else ABS_GUARD_HIGH

    def dynamic_full_factor(a: float, b: float) -> float:
        return FULL_FACTOR_LOW if max(a, b) < LOW_RATE_CUTOFF else FULL_FACTOR_DEFAULT

    # ---------------------------
    # Precompute signed router residuals (original telemetry)
    # resid_frac[router] = (Σtx - Σrx) / max(Σtx, Σrx, 1)
    # ---------------------------
    orig_router_sums: Dict[str, Tuple[float, float]] = {}
    for router, if_list in topology.items():
        s_tx = 0.0
        s_rx = 0.0
        for iid in if_list:
            d = telemetry.get(iid)
            if not d:
                continue
            s_tx += to_float(d.get('tx_rate', 0.0))
            s_rx += to_float(d.get('rx_rate', 0.0))
        orig_router_sums[router] = (s_tx, s_rx)
    orig_resid_signed: Dict[str, float] = {}
    for r, (s_tx, s_rx) in orig_router_sums.items():
        denom = max(s_tx, s_rx, 1.0)
        orig_resid_signed[r] = (s_tx - s_rx) / denom

    # ---------------------------
    # Build unique link pairs
    # ---------------------------
    pairs: Dict[Tuple[str, str], Tuple[str, str]] = {}
    paired_members: set = set()
    for if_id, data in telemetry.items():
        peer = data.get('connected_to')
        if peer and peer in telemetry:
            key = tuple(sorted([if_id, peer]))
            if key not in pairs:
                pairs[key] = (if_id, peer)
                paired_members.add(if_id)
                paired_members.add(peer)

    # ---------------------------
    # Pairwise hardening with residual-tilt and asymmetric partial averaging
    # ---------------------------
    per_if: Dict[str, Dict[str, Any]] = {}

    def repair_direction(x: float, y: float, local_router: str, diff_tol: float) -> Tuple[float, float, float]:
        """
        Repair a pair direction (x vs y) returning (x_rep, y_rep, conf_dir).

        - Uses dynamic absolute guard and full-trigger factor.
        - Applies residual-tilted weighted consensus when applicable.
        - Partial averaging moves the louder side more toward consensus.
        """
        d = rel_diff(x, y)
        abs_diff = abs(x - y)
        abs_guard = dynamic_abs_guard(x, y)
        full_factor = dynamic_full_factor(x, y)

        # Weighted consensus (residual tilt)
        w_x = 0.5
        resid_local = orig_resid_signed.get(local_router, 0.0)
        # Tilt only when violation is meaningful
        if (d > diff_tol) and (abs_diff > abs_guard):
            # If (x - y) and local residual have same sign, bias away from x to reduce local residual
            if (x - y) * resid_local > 0:
                gamma = min(TILT_GAMMA_MAX, TILT_GAMMA_SCALE * abs(resid_local))
                w_x = clamp(0.5 - gamma, 0.2, 0.8)
        w_y = 1.0 - w_x
        consensus = w_x * x + w_y * y

        x_rep, y_rep = x, y
        conf = max(0.0, 1.0 - d)

        if (d > diff_tol) and (abs_diff > abs_guard):
            # Partial vs full convergence band
            if d <= full_factor * diff_tol:
                # Partial averaging with asymmetric movement (louder side moves more)
                k_base = (d - diff_tol) / max(diff_tol, 1e-9)
                # Low-rate: steeper ramp
                if max(x, y) < LOW_RATE_CUTOFF:
                    k = clamp01(k_base ** 1.2)
                else:
                    k = clamp01(k_base)
                loud_is_x = x >= y
                loud = x if loud_is_x else y
                quiet = y if loud_is_x else x
                r = (loud - quiet) / max(1.0, loud)
                k_loud = clamp01(k * (1.0 + 0.5 * r))
                k_quiet = clamp01(k * (1.0 - 0.5 * r))
                new_loud = loud * (1.0 - k_loud) + consensus * k_loud
                new_quiet = quiet * (1.0 - k_quiet) + consensus * k_quiet
                if loud_is_x:
                    x_rep, y_rep = new_loud, new_quiet
                else:
                    x_rep, y_rep = new_quiet, new_loud
            else:
                # Full convergence to weighted consensus
                x_rep = consensus
                y_rep = consensus

        # Confidence floors on agreement
        max_mag = max(x, y)
        if d <= STRONG_AGREE_DIFF and max_mag >= LOW_RATE_CUTOFF:
            conf = max(conf, 0.99)
        elif d <= diff_tol:
            conf = max(conf, 0.98 if max_mag >= LOW_RATE_CUTOFF else 0.97)

        # Penalize by own change magnitude for calibration
        denom_x = max(abs(x), abs(x_rep), 1.0)
        denom_y = max(abs(y), abs(y_rep), 1.0)
        change_x = abs(x_rep - x) / denom_x
        change_y = abs(y_rep - y) / denom_y
        conf_x = clamp01(min(conf, 1.0 - change_x))
        conf_y = clamp01(min(conf, 1.0 - change_y))

        return x_rep, y_rep, (conf_x + conf_y) / 2.0  # directional consensus confidence

    for _, (a_id, b_id) in pairs.items():
        a = telemetry[a_id]
        b = telemetry[b_id]

        a_stat = a.get('interface_status', 'unknown')
        b_stat = b.get('interface_status', 'unknown')

        a_rx, a_tx = to_float(a.get('rx_rate', 0.0)), to_float(a.get('tx_rate', 0.0))
        b_rx, b_tx = to_float(b.get('rx_rate', 0.0)), to_float(b.get('tx_rate', 0.0))

        a_has = has_traffic(a)
        b_has = has_traffic(b)

        # Pair status via consistency + traffic evidence
        if a_stat == 'down' and b_stat == 'down':
            pair_status = 'down'
        elif a_stat == 'up' and b_stat == 'up':
            pair_status = 'up'
        else:
            pair_status = 'up' if (a_has or b_has) else 'down'

        # Initialize repaired values and confidences
        rep_a_rx, rep_a_tx = a_rx, a_tx
        rep_b_rx, rep_b_tx = b_rx, b_tx
        rx_conf_a = tx_conf_a = rx_conf_b = tx_conf_b = 1.0

        if pair_status == 'down':
            rep_a_rx = rep_a_tx = rep_b_rx = rep_b_tx = 0.0
            base_conf = 0.95 if not (a_has or b_has) else 0.7
            rx_conf_a = tx_conf_a = rx_conf_b = tx_conf_b = base_conf
        else:
            # Direction A->B: A.tx vs B.rx
            tau_ab = dynamic_tau(a_tx, b_rx)
            new_a_tx, new_b_rx, conf_ab = repair_direction(a_tx, b_rx, a.get('local_router'), tau_ab)
            rep_a_tx, rep_b_rx = new_a_tx, new_b_rx
            # Direction B->A: B.tx vs A.rx
            tau_ba = dynamic_tau(b_tx, a_rx)
            new_b_tx, new_a_rx, conf_ba = repair_direction(b_tx, a_rx, b.get('local_router'), tau_ba)
            rep_b_tx, rep_a_rx = new_b_tx, new_a_rx

            # Base confidences from directions
            tx_conf_a = conf_ab
            rx_conf_b = conf_ab
            tx_conf_b = conf_ba
            rx_conf_a = conf_ba

            # Asymmetric evidence-driven "up" dampening when only one side has traffic
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
            # Boost on strong bilateral agreement at high rates
            d_ab = rel_diff(rep_a_tx, rep_b_rx) if max(rep_a_tx, rep_b_rx) > 0 else 0.0
            d_ba = rel_diff(rep_b_tx, rep_a_rx) if max(rep_b_tx, rep_a_rx) > 0 else 0.0
            if (max(rep_a_tx, rep_b_rx, rep_b_tx, rep_a_rx) >= LOW_RATE_CUTOFF) and (d_ab <= STRONG_AGREE_DIFF) and (d_ba <= STRONG_AGREE_DIFF):
                status_conf = max(status_conf, 0.99)

        per_if[a_id] = {
            'repaired_rx': rep_a_rx,
            'repaired_tx': rep_a_tx,
            'rx_conf': clamp01(rx_conf_a),
            'tx_conf': clamp01(tx_conf_a),
            'repaired_status': pair_status,
            'status_conf': status_conf
        }
        per_if[b_id] = {
            'repaired_rx': rep_b_rx,
            'repaired_tx': rep_b_tx,
            'rx_conf': clamp01(rx_conf_b),
            'tx_conf': clamp01(tx_conf_b),
            'repaired_status': pair_status,
            'status_conf': status_conf
        }

    # ---------------------------
    # Dangling interfaces (no peer)
    # ---------------------------
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
            # No topology redundancy; keep with moderate confidence
            per_if[if_id] = {
                'repaired_rx': rx,
                'repaired_tx': tx,
                'rx_conf': 0.6,
                'tx_conf': 0.6,
                'repaired_status': status if status in ('up', 'down') else 'up',
                'status_conf': 0.6
            }

    # ---------------------------
    # Router residuals using repaired rates
    # ---------------------------
    def compute_router_sums_and_residuals(per_if_map: Dict[str, Dict[str, Any]]) -> Tuple[Dict[str, Tuple[float, float]], Dict[str, float]]:
        sums: Dict[str, Tuple[float, float]] = {}
        residuals: Dict[str, float] = {}
        for router, if_list in topology.items():
            s_tx = 0.0
            s_rx = 0.0
            for iid in if_list:
                rep = per_if_map.get(iid)
                if rep is None:
                    continue
                s_tx += to_float(rep['repaired_tx'])
                s_rx += to_float(rep['repaired_rx'])
            sums[router] = (s_tx, s_rx)
            residuals[router] = abs(s_tx - s_rx) / max(s_tx, s_rx, 1.0)
        return sums, residuals

    router_sums, router_residual = compute_router_sums_and_residuals(per_if)

    # ---------------------------
    # Safeguarded micro-adjustments on dominant unpaired interfaces
    # ---------------------------
    # Adjust only dangling interfaces and only if router residual is meaningful.
    for router, if_list in topology.items():
        s_tx, s_rx = router_sums.get(router, (0.0, 0.0))
        denom = max(s_tx, s_rx, 1.0)
        resid_frac = (s_tx - s_rx) / denom if denom > 0 else 0.0
        if abs(resid_frac) < MICRO_ACTIVATE_RESID:
            continue

        # Candidate set: unpaired, up interfaces on this router with sufficient activity
        candidates: List[str] = []
        for iid in if_list:
            rep = per_if.get(iid)
            if rep is None:
                continue
            if iid in paired_members:
                continue
            if rep.get('repaired_status', 'up') != 'up':
                continue
            if max(to_float(rep['repaired_tx']), to_float(rep['repaired_rx'])) < MICRO_RATE_FLOOR:
                continue
            candidates.append(iid)
        if not candidates:
            continue

        # Dominance pick: largest counter along imbalance direction
        tx_excess = resid_frac > 0.0  # Σtx > Σrx
        best_iid = None
        best_val = -1.0
        for iid in candidates:
            rep = per_if[iid]
            val = to_float(rep['repaired_tx']) if tx_excess else to_float(rep['repaired_rx'])
            if val > best_val:
                best_val = val
                best_iid = iid
        if best_iid is None or best_val <= 0.0:
            continue

        # Attempt first nudge
        alpha = min(MICRO_MAX_ALPHA, MICRO_ALPHA_SCALE * abs(resid_frac))
        rep = per_if[best_iid]
        old_tx = to_float(rep['repaired_tx'])
        old_rx = to_float(rep['repaired_rx'])
        new_tx, new_rx = old_tx, old_rx
        if tx_excess:
            new_tx = old_tx * (1.0 - alpha)
        else:
            new_rx = old_rx * (1.0 - alpha)

        # Recompute sums and residual
        s_tx_new = s_tx - old_tx + new_tx
        s_rx_new = s_rx - old_rx + new_rx
        resid_new = abs(s_tx_new - s_rx_new) / max(s_tx_new, s_rx_new, 1.0)
        resid_old = abs(s_tx - s_rx) / max(s_tx, s_rx, 1.0)
        improve = 0.0 if resid_old == 0 else (resid_old - resid_new) / resid_old

        # Skew guard: do not worsen |tx-rx| by >3% relative
        old_skew = abs(old_tx - old_rx) / max(1.0, max(old_tx, old_rx))
        new_skew = abs(new_tx - new_rx) / max(1.0, max(new_tx, new_rx))
        skew_ok = new_skew <= old_skew * (1.0 + MICRO_SKEW_LIMIT + 1e-9)

        if improve >= MICRO_REQ_IMPROVE and skew_ok:
            rep['repaired_tx'], rep['repaired_rx'] = new_tx, new_rx
            per_if[best_iid] = rep
            # Update router sums for potential second step and for final penalties
            router_sums[router] = (s_tx_new, s_rx_new)
            router_residual[router] = resid_new

            # Optional second mini-step if strong improvement and residual remains large
            if improve >= MICRO_REQ_IMPROVE_SECOND:
                s_tx2, s_rx2 = s_tx_new, s_rx_new
                denom2 = max(s_tx2, s_rx2, 1.0)
                resid_frac2 = (s_tx2 - s_rx2) / denom2 if denom2 > 0 else 0.0
                if abs(resid_frac2) >= MICRO_RESID_FOR_SECOND:
                    alpha2 = min(MICRO_SECOND_ALPHA, MICRO_ALPHA_SCALE * abs(resid_frac2))
                    rep2 = per_if[best_iid]
                    otx2 = to_float(rep2['repaired_tx'])
                    orx2 = to_float(rep2['repaired_rx'])
                    ntx2, nrx2 = otx2, orx2
                    if resid_frac2 > 0:
                        ntx2 = otx2 * (1.0 - alpha2)
                    else:
                        nrx2 = orx2 * (1.0 - alpha2)
                    s_tx3 = s_tx2 - otx2 + ntx2
                    s_rx3 = s_rx2 - orx2 + nrx2
                    resid3 = abs(s_tx3 - s_rx3) / max(s_tx3, s_rx3, 1.0)
                    resid2 = abs(s_tx2 - s_rx2) / max(s_tx2, s_rx2, 1.0)
                    improve2 = 0.0 if resid2 == 0 else (resid2 - resid3) / resid2
                    old_skew2 = abs(otx2 - orx2) / max(1.0, max(otx2, orx2))
                    new_skew2 = abs(ntx2 - nrx2) / max(1.0, max(ntx2, nrx2))
                    skew_ok2 = new_skew2 <= old_skew2 * (1.0 + MICRO_SKEW_LIMIT + 1e-9)
                    if improve2 >= MICRO_REQ_IMPROVE and skew_ok2:
                        rep2['repaired_tx'], rep2['repaired_rx'] = ntx2, nrx2
                        per_if[best_iid] = rep2
                        router_sums[router] = (s_tx3, s_rx3)
                        router_residual[router] = resid3

    # Ensure final router residuals recomputed (post micro-adjustments)
    router_sums, router_residual = compute_router_sums_and_residuals(per_if)

    # ---------------------------
    # Final assembly with share-aware, direction-coupled penalties
    # ---------------------------
    result: Dict[str, Dict[str, Tuple]] = {}
    # Precompute per-router directional sums for shares
    sums_tx: Dict[str, float] = {r: v[0] for r, v in router_sums.items()}
    sums_rx: Dict[str, float] = {r: v[1] for r, v in router_sums.items()}

    for if_id, data in telemetry.items():
        rep = per_if.get(if_id, {})
        repaired_rx = to_float(rep.get('repaired_rx', data.get('rx_rate', 0.0)))
        repaired_tx = to_float(rep.get('repaired_tx', data.get('tx_rate', 0.0)))
        repaired_status = rep.get('repaired_status', data.get('interface_status', 'unknown'))

        # Base confidences from hardening/micro
        rx_conf = float(rep.get('rx_conf', 0.6))
        tx_conf = float(rep.get('tx_conf', 0.6))
        status_conf = float(rep.get('status_conf', 0.6))

        local_router = data.get('local_router')
        remote_router = data.get('remote_router')
        resid_local = router_residual.get(local_router, 0.0)
        resid_remote = router_residual.get(remote_router, 0.0)

        # Direction-coupled penalties with share awareness
        tx_share = 0.0
        if local_router in sums_tx:
            tx_share = repaired_tx / max(1.0, sums_tx.get(local_router, 1.0))
        rx_share_local = 0.0
        if local_router in sums_rx:
            rx_share_local = repaired_rx / max(1.0, sums_rx.get(local_router, 1.0))

        # For TX: emphasize local residual more when iface has higher TX share
        w_local_tx = clamp01(0.6 + 0.2 * tx_share)
        w_remote_tx = clamp01(1.0 - w_local_tx)
        pen_tx = clamp01(1.0 - (w_local_tx * resid_local + w_remote_tx * resid_remote))

        # For RX: emphasize remote residual more; weight scales with this iface's RX share
        w_remote_rx = clamp01(0.6 + 0.2 * rx_share_local)
        w_local_rx = clamp01(1.0 - w_remote_rx)
        pen_rx = clamp01(1.0 - (w_local_rx * resid_local + w_remote_rx * resid_remote))

        # Apply penalties
        tx_conf = clamp01(tx_conf * pen_tx)
        rx_conf = clamp01(rx_conf * pen_rx)

        # Status confidence mild scaling using weaker direction penalty
        status_conf = clamp01(status_conf * (0.85 + 0.15 * min(pen_tx, pen_rx)))

        # Compose output
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
