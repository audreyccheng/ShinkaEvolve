# EVOLVE-BLOCK-START
"""
Network telemetry repair algorithm that detects and corrects inconsistencies
in network interface telemetry data using topology relationships.

Takes interface telemetry data and detects/repairs inconsistencies based on
network invariants like link symmetry and flow conservation.
"""
from typing import Dict, Any, Tuple, List


def repair_network_telemetry(telemetry: Dict[str, Dict[str, Any]],
                             topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:
    """
    Repair network interface telemetry by detecting and correcting inconsistencies.

    Core principle: Use network invariants to validate and repair telemetry:
    1. Link Symmetry (R3): my_tx_rate ≈ their_rx_rate for connected interfaces
    2. Flow Conservation (R1): Sum(incoming traffic) = Sum(outgoing traffic) at each router
    3. Interface Consistency: Status should be consistent across connected pairs

    Args:
        telemetry: Dictionary where key is interface_id and value contains:
            - interface_status: "up" or "down"
            - rx_rate: receive rate in Mbps
            - tx_rate: transmit rate in Mbps
            - connected_to: interface_id this interface connects to
            - local_router: router_id this interface belongs to
            - remote_router: router_id on the other side
        topology: Dictionary where key is router_id and value contains a list of interface_ids

    Returns:
        Dictionary with same structure but telemetry values become tuples of:
        (original_value, repaired_value, confidence_score)
        where confidence ranges from 0.0 (very uncertain) to 1.0 (very confident)
    """
    # Measurement timing tolerance (from Hodor research: ~2%)
    HARDENING_THRESHOLD = 0.02
    # Magnitude-aware thresholds and guards
    TH_REL_LOW = 0.05       # relaxed tolerance for low-rate links
    TH_ABS_GUARD = 0.5      # absolute guard (Mbps) to avoid correcting tiny diffs
    ABS_GUARD_LOW = 0.3     # smaller absolute guard for low-rate links
    LOW_RATE_CUTOFF = 10.0  # Mbps threshold for low-rate behavior
    EPS = 1e-6  # small absolute tolerance for "no traffic"

    def to_float(x: Any) -> float:
        try:
            return float(x)
        except Exception:
            return 0.0

    def norm_diff(a: float, b: float) -> float:
        return abs(a - b) / max(a, b, 1.0)

    def change_ratio(orig: float, rep: float) -> float:
        denom = max(abs(orig), abs(rep), 1.0)
        return abs(rep - orig) / denom

    def has_traffic(d: Dict[str, Any]) -> bool:
        return (to_float(d.get('rx_rate', 0.0)) > EPS) or (to_float(d.get('tx_rate', 0.0)) > EPS)

    # Build link pairs
    pairs = {}  # key: tuple(sorted(if1, if2)) -> (if1_id, if2_id)
    for if_id, data in telemetry.items():
        peer = data.get('connected_to')
        if peer and peer in telemetry:
            key = tuple(sorted([if_id, peer]))
            # Store canonical orientation (A=if_id, B=peer) for deterministic processing
            if key not in pairs:
                pairs[key] = (if_id, peer)

    # First pass: pairwise hardening and status harmonization
    per_if_repair = {}  # interface_id -> interim repaired values and base confidences

    for key, (a_id, b_id) in pairs.items():
        a = telemetry[a_id]
        b = telemetry[b_id]

        a_status = a.get('interface_status', 'unknown')
        b_status = b.get('interface_status', 'unknown')

        a_rx, a_tx = float(a.get('rx_rate', 0.0)), float(a.get('tx_rate', 0.0))
        b_rx, b_tx = float(b.get('rx_rate', 0.0)), float(b.get('tx_rate', 0.0))

        # Decide pair status using interface consistency + traffic evidence
        a_has = has_traffic(a)
        b_has = has_traffic(b)

        if a_status == 'down' and b_status == 'down':
            pair_status = 'down'
        elif a_status == 'up' and b_status == 'up':
            pair_status = 'up'
        else:
            # Mismatch: if any traffic exists on either end, consider link up; otherwise down
            pair_status = 'up' if (a_has or b_has) else 'down'

        # Prepare repairs for both directions
        # Direction A->B uses A.tx vs B.rx
        diff_ab = norm_diff(a_tx, b_rx)
        # Direction B->A uses B.tx vs A.rx
        diff_ba = norm_diff(b_tx, a_rx)

        # Initialize with originals
        rep_a_tx, rep_b_rx = a_tx, b_rx
        rep_b_tx, rep_a_rx = b_tx, a_rx
        rx_conf_a = 1.0
        tx_conf_a = 1.0
        rx_conf_b = 1.0
        tx_conf_b = 1.0

        if pair_status == 'down':
            # No traffic on a down link
            rep_a_tx, rep_b_rx, rep_b_tx, rep_a_rx = 0.0, 0.0, 0.0, 0.0
            # Confidence is high if there was no traffic observed; otherwise moderate
            base_conf = 0.95 if not (a_has or b_has) else 0.7
            rx_conf_a = tx_conf_a = rx_conf_b = tx_conf_b = base_conf
        else:
            # Link is up: magnitude-aware hardening with partial averaging and confidence calibrated by change magnitude
            # Direction A->B (A.tx vs B.rx)
            max_ab = max(a_tx, b_rx)
            thr_rel_ab = TH_REL_LOW if max_ab < LOW_RATE_CUTOFF else HARDENING_THRESHOLD
            guard_ab = ABS_GUARD_LOW if max_ab < LOW_RATE_CUTOFF else TH_ABS_GUARD
            abs_ab = abs(a_tx - b_rx)

            if (diff_ab > thr_rel_ab) and (abs_ab > guard_ab):
                v_ab = 0.5 * (a_tx + b_rx)
                # Partial averaging near threshold; full consensus beyond 2×thr (1.6× at low rate)
                partial_upper_ab = (1.6 * thr_rel_ab) if max_ab < LOW_RATE_CUTOFF else (2.0 * thr_rel_ab)
                if diff_ab <= partial_upper_ab:
                    k_ab = (diff_ab - thr_rel_ab) / max(thr_rel_ab, 1e-9)
                    k_ab = min(1.0, max(0.0, k_ab))
                    if max_ab < LOW_RATE_CUTOFF:
                        k_ab = k_ab ** 1.2  # slightly steeper ramp at low rates
                    rep_a_tx = a_tx * (1.0 - k_ab) + v_ab * k_ab
                    rep_b_rx = b_rx * (1.0 - k_ab) + v_ab * k_ab
                else:
                    rep_a_tx = v_ab
                    rep_b_rx = v_ab
                # Confidence decreases with violation and applied change magnitude
                change_a = change_ratio(a_tx, rep_a_tx)
                change_b = change_ratio(b_rx, rep_b_rx)
                conf_base_ab = max(0.0, 1.0 - diff_ab)
                tx_conf_a = min(conf_base_ab, 1.0 - 0.8 * change_a)
                rx_conf_b = min(conf_base_ab, 1.0 - 0.8 * change_b)
            else:
                # Within tolerance or under absolute guard: keep values and set strong confidence floors
                if max_ab >= LOW_RATE_CUTOFF and diff_ab <= 0.005:
                    conf_floor_ab = 0.99
                else:
                    conf_floor_ab = 0.98 if max_ab >= LOW_RATE_CUTOFF else 0.97
                tx_conf_a = max(tx_conf_a, conf_floor_ab)
                rx_conf_b = max(rx_conf_b, conf_floor_ab)

            # Direction B->A (B.tx vs A.rx)
            max_ba = max(b_tx, a_rx)
            thr_rel_ba = TH_REL_LOW if max_ba < LOW_RATE_CUTOFF else HARDENING_THRESHOLD
            guard_ba = ABS_GUARD_LOW if max_ba < LOW_RATE_CUTOFF else TH_ABS_GUARD
            abs_ba = abs(b_tx - a_rx)

            if (diff_ba > thr_rel_ba) and (abs_ba > guard_ba):
                v_ba = 0.5 * (b_tx + a_rx)
                partial_upper_ba = (1.6 * thr_rel_ba) if max_ba < LOW_RATE_CUTOFF else (2.0 * thr_rel_ba)
                if diff_ba <= partial_upper_ba:
                    k_ba = (diff_ba - thr_rel_ba) / max(thr_rel_ba, 1e-9)
                    k_ba = min(1.0, max(0.0, k_ba))
                    if max_ba < LOW_RATE_CUTOFF:
                        k_ba = k_ba ** 1.2
                    rep_b_tx = b_tx * (1.0 - k_ba) + v_ba * k_ba
                    rep_a_rx = a_rx * (1.0 - k_ba) + v_ba * k_ba
                else:
                    rep_b_tx = v_ba
                    rep_a_rx = v_ba
                change_b2 = change_ratio(b_tx, rep_b_tx)
                change_a2 = change_ratio(a_rx, rep_a_rx)
                conf_base_ba = max(0.0, 1.0 - diff_ba)
                tx_conf_b = min(conf_base_ba, 1.0 - 0.8 * change_b2)
                rx_conf_a = min(conf_base_ba, 1.0 - 0.8 * change_a2)
            else:
                if max_ba >= LOW_RATE_CUTOFF and diff_ba <= 0.005:
                    conf_floor_ba = 0.99
                else:
                    conf_floor_ba = 0.98 if max_ba >= LOW_RATE_CUTOFF else 0.97
                tx_conf_b = max(tx_conf_b, conf_floor_ba)
                rx_conf_a = max(rx_conf_a, conf_floor_ba)

        # Status confidence based on agreement and evidence
        if pair_status == 'down':
            if a_status == 'down' and b_status == 'down' and not (a_has or b_has):
                status_conf = 0.98
            else:
                status_conf = 0.7
        else:  # up
            if a_status == 'up' and b_status == 'up':
                status_conf = 0.95
            else:
                # we decided up due to traffic evidence
                status_conf = 0.8

        per_if_repair[a_id] = {
            'repaired_rx': rep_a_rx,
            'repaired_tx': rep_a_tx,
            'rx_conf': rx_conf_a,
            'tx_conf': tx_conf_a,
            'repaired_status': pair_status,
            'status_conf': status_conf
        }
        per_if_repair[b_id] = {
            'repaired_rx': rep_b_rx,
            'repaired_tx': rep_b_tx,
            'rx_conf': rx_conf_b,
            'tx_conf': tx_conf_b,
            'repaired_status': pair_status,
            'status_conf': status_conf
        }

    # Handle interfaces without a valid peer (dangling or missing peer data)
    for if_id, data in telemetry.items():
        if if_id in per_if_repair:
            continue
        status = data.get('interface_status', 'unknown')
        rx = float(data.get('rx_rate', 0.0))
        tx = float(data.get('tx_rate', 0.0))
        if status == 'down':
            # Enforce no traffic on down interfaces
            per_if_repair[if_id] = {
                'repaired_rx': 0.0,
                'repaired_tx': 0.0,
                'rx_conf': 0.9,
                'tx_conf': 0.9,
                'repaired_status': 'down',
                'status_conf': 0.95
            }
        else:
            # No peer to cross-check; keep values but lower confidence slightly.
            per_if_repair[if_id] = {
                'repaired_rx': rx,
                'repaired_tx': tx,
                'rx_conf': 0.6,
                'tx_conf': 0.6,
                'repaired_status': status if status in ('up', 'down') else 'up',
                'status_conf': 0.6
            }

    # Router-level micro-adjustments for flow conservation (R1) using dangling interfaces only
    # We only adjust interfaces without a known peer to avoid breaking link symmetry.
    paired_ids = set()
    for _, (aid, bid) in pairs.items():
        paired_ids.add(aid)
        paired_ids.add(bid)

    # Micro-adjustment parameters
    RESID_MIN = 0.03        # minimum router residual fraction to trigger micro-adjustments
    IMPROVE_REQ = 0.08      # require ≥8% residual improvement to commit first step
    SECOND_STEP_REQ = 0.20  # allow second mini-step if improvement ≥20%
    SECOND_STEP_MAX = 0.01  # second step cap (≤1% of router scale)

    for router, if_list in topology.items():
        present = [iid for iid in if_list if iid in per_if_repair]
        if not present:
            continue

        # Current sums for this router
        sum_tx = 0.0
        sum_rx = 0.0
        for iid in present:
            rep = per_if_repair[iid]
            sum_tx += to_float(rep['repaired_tx'])
            sum_rx += to_float(rep['repaired_rx'])
        delta = sum_tx - sum_rx  # want to drive toward zero
        scale = max(sum_tx, sum_rx, 1.0)
        resid = abs(delta) / scale

        # Trigger only on sufficiently large residuals
        if resid < RESID_MIN or abs(delta) <= EPS:
            continue

        # Candidates: dangling, up, sufficient magnitude, and aligned with imbalance
        candidates: List[Tuple[str, float, float, float]] = []
        for iid in present:
            if iid in paired_ids:
                continue
            if per_if_repair[iid].get('repaired_status', 'up') == 'down':
                continue
            cur_tx = to_float(per_if_repair[iid]['repaired_tx'])
            cur_rx = to_float(per_if_repair[iid]['repaired_rx'])
            if max(cur_tx, cur_rx) < LOW_RATE_CUTOFF:
                continue
            contrib = cur_tx - cur_rx
            if (delta > 0 and contrib > 0) or (delta < 0 and contrib < 0):
                candidates.append((iid, contrib, cur_tx, cur_rx))

        if not candidates:
            continue

        total_same_dir = sum(abs(c[1]) for c in candidates) or EPS
        dom_iid, dom_contrib, cur_tx, cur_rx = max(candidates, key=lambda t: abs(t[1]))
        if abs(dom_contrib) / total_same_dir < 0.6:
            continue

        # Tentative nudge: reduce the dominant direction on the dominant interface
        alpha = min(0.02, 0.5 * resid)
        step = alpha * scale
        new_tx, new_rx = cur_tx, cur_rx
        if delta > 0 and dom_contrib > 0:
            # Too much TX globally; reduce TX on this interface
            new_tx = max(0.0, cur_tx - min(step, cur_tx))
        elif delta < 0 and dom_contrib < 0:
            # Too much RX globally; reduce RX on this interface
            new_rx = max(0.0, cur_rx - min(step, cur_rx))
        else:
            continue

        # Internal TX/RX skew guard (≤3%)
        new_int = abs(new_tx - new_rx) / max(1.0, max(new_tx, new_rx, 1.0))
        if new_int > 0.03 + 1e-9:
            continue

        # Commit only if router residual improves sufficiently
        sum_tx_new = sum_tx - cur_tx + new_tx
        sum_rx_new = sum_rx - cur_rx + new_rx
        resid_new = abs(sum_tx_new - sum_rx_new) / max(sum_tx_new, sum_rx_new, 1.0)
        if resid_new <= (1.0 - IMPROVE_REQ) * resid:
            # Commit change
            per_if_repair[dom_iid]['repaired_tx'] = new_tx
            per_if_repair[dom_iid]['repaired_rx'] = new_rx

            # Optional second mini-step if strong improvement and residual still meaningful
            if resid_new <= (1.0 - SECOND_STEP_REQ) * resid and resid_new >= 0.04:
                alpha2 = min(SECOND_STEP_MAX, 0.5 * resid_new)
                step2 = alpha2 * scale
                cur_tx2, cur_rx2 = new_tx, new_rx
                if delta > 0 and dom_contrib > 0:
                    cand_tx2 = max(0.0, cur_tx2 - min(step2, cur_tx2))
                    cand_rx2 = cur_rx2
                else:
                    cand_rx2 = max(0.0, cur_rx2 - min(step2, cur_rx2))
                    cand_tx2 = cur_tx2
                new_int2 = abs(cand_tx2 - cand_rx2) / max(1.0, max(cand_tx2, cand_rx2, 1.0))
                sum_tx_new2 = sum_tx - cur_tx + cand_tx2
                sum_rx_new2 = sum_rx - cur_rx + cand_rx2
                resid_new2 = abs(sum_tx_new2 - sum_rx_new2) / max(sum_tx_new2, sum_rx_new2, 1.0)
                if new_int2 <= 0.03 + 1e-9 and resid_new2 < resid_new:
                    per_if_repair[dom_iid]['repaired_tx'] = cand_tx2
                    per_if_repair[dom_iid]['repaired_rx'] = cand_rx2
                    resid_new = resid_new2

            # Confidence penalty proportional to applied change for that interface
            tx_orig = to_float(telemetry.get(dom_iid, {}).get('tx_rate', 0.0))
            rx_orig = to_float(telemetry.get(dom_iid, {}).get('rx_rate', 0.0))
            tx_change = change_ratio(tx_orig, to_float(per_if_repair[dom_iid]['repaired_tx']))
            rx_change = change_ratio(rx_orig, to_float(per_if_repair[dom_iid]['repaired_rx']))
            per_if_repair[dom_iid]['tx_conf'] = max(0.0, min(float(per_if_repair[dom_iid].get('tx_conf', 0.6)), 1.0 - 0.8 * tx_change))
            per_if_repair[dom_iid]['rx_conf'] = max(0.0, min(float(per_if_repair[dom_iid].get('rx_conf', 0.6)), 1.0 - 0.8 * rx_change))

    # Second pass: compute router-level flow conservation residuals using repaired rates
    router_residual: Dict[str, float] = {}
    for router, if_list in topology.items():
        sum_tx = 0.0
        sum_rx = 0.0
        for if_id in if_list:
            if if_id in per_if_repair:
                rep = per_if_repair[if_id]
                # Include all interfaces; down interfaces contribute 0 traffic (already enforced)
                sum_tx += float(rep['repaired_tx'])
                sum_rx += float(rep['repaired_rx'])
            else:
                # Interface not present in telemetry; ignore (no contribution)
                continue
        resid = abs(sum_tx - sum_rx) / max(sum_tx, sum_rx, 1.0)
        router_residual[router] = resid

    # Final assembly: scale confidences by router residuals (dynamic checking)
    result: Dict[str, Dict[str, Tuple]] = {}
    for if_id, data in telemetry.items():
        repaired = per_if_repair.get(if_id, {})
        repaired_rx = repaired.get('repaired_rx', float(data.get('rx_rate', 0.0)))
        repaired_tx = repaired.get('repaired_tx', float(data.get('tx_rate', 0.0)))
        repaired_status = repaired.get('repaired_status', data.get('interface_status', 'unknown'))

        # Base confidences from link hardening
        rx_conf = float(repaired.get('rx_conf', 0.6))
        tx_conf = float(repaired.get('tx_conf', 0.6))
        status_conf = float(repaired.get('status_conf', 0.6))

        # Apply router-based penalty using both local and remote routers when available
        local_router = data.get('local_router')
        remote_router = data.get('remote_router')

        resid_local = router_residual.get(local_router, 0.0)
        resid_remote = router_residual.get(remote_router, 0.0)

        # Direction-aware penalties: TX depends more on local, RX more on remote
        penalty_tx = 1.0 - (0.6 * resid_local + 0.4 * resid_remote)
        penalty_rx = 1.0 - (0.6 * resid_remote + 0.4 * resid_local)
        penalty_tx = max(0.0, min(1.0, penalty_tx))
        penalty_rx = max(0.0, min(1.0, penalty_rx))
        min_penalty = min(penalty_tx, penalty_rx)

        tx_conf = max(0.0, min(1.0, tx_conf * penalty_tx))
        rx_conf = max(0.0, min(1.0, rx_conf * penalty_rx))
        # Status confidence mildly scaled by the weaker (more conservative) penalty
        status_conf = max(0.0, min(1.0, status_conf * (0.85 + 0.15 * min_penalty)))

        repaired_data: Dict[str, Any] = {}
        # Store repaired values with confidence scores
        rx_orig = float(data.get('rx_rate', 0.0))
        tx_orig = float(data.get('tx_rate', 0.0))
        status_orig = data.get('interface_status', 'unknown')

        repaired_data['rx_rate'] = (rx_orig, repaired_rx, rx_conf)
        repaired_data['tx_rate'] = (tx_orig, repaired_tx, tx_conf)
        repaired_data['interface_status'] = (status_orig, repaired_status, status_conf)

        # Copy metadata unchanged
        repaired_data['connected_to'] = data.get('connected_to')
        repaired_data['local_router'] = local_router
        repaired_data['remote_router'] = remote_router

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