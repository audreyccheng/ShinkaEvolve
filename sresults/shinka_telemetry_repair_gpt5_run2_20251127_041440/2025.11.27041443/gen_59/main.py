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
    HARDENING_THRESHOLD = 0.02  # base 2% timing tolerance
    # Magnitude-aware tolerances and guards to avoid over-correcting tiny flows
    TH_REL_DEFAULT = 0.02   # 2% for normal/high-rate links
    TH_REL_LOW = 0.05       # 5% for low-rate links (<10 Mbps)
    TH_ABS_GUARD = 0.5      # 0.5 Mbps absolute guard before averaging
    LOW_RATE_MAX = 10.0     # threshold between low/high-rate
    STRONG_AGREE_DIFF = 0.005  # 0.5% strong agreement floor
    EPS = 1e-6  # small absolute tolerance for "no traffic"

    def norm_diff(a: float, b: float) -> float:
        return abs(a - b) / max(a, b, 1.0)

    def has_traffic(d: Dict[str, Any]) -> bool:
        return (float(d.get('rx_rate', 0.0) or 0.0) > EPS) or (float(d.get('tx_rate', 0.0) or 0.0) > EPS)

    def change_ratio(orig: float, rep: float) -> float:
        # magnitude of change normalized by the larger magnitude and 1.0 to avoid division by zero
        denom = max(abs(orig), abs(rep), 1.0)
        return abs(rep - orig) / denom

    # Build link pairs (unique, undirected)
    pairs = {}  # key: tuple(sorted(if1, if2)) -> (if1_id, if2_id)
    for if_id, data in telemetry.items():
        peer = data.get('connected_to')
        if peer and peer in telemetry:
            key = tuple(sorted([if_id, peer]))
            if key not in pairs:
                pairs[key] = (if_id, peer)

    # First pass: pairwise hardening and status harmonization using link symmetry
    per_if_repair = {}  # interface_id -> interim repaired values and base confidences

    for _, (a_id, b_id) in pairs.items():
        a = telemetry[a_id]
        b = telemetry[b_id]

        a_status = a.get('interface_status', 'unknown')
        b_status = b.get('interface_status', 'unknown')

        a_rx, a_tx = float(a.get('rx_rate', 0.0) or 0.0), float(a.get('tx_rate', 0.0) or 0.0)
        b_rx, b_tx = float(b.get('rx_rate', 0.0) or 0.0), float(b.get('tx_rate', 0.0) or 0.0)

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
            # Link is up: enforce link symmetry with magnitude-aware thresholds and absolute guard
            # Direction A->B (A.tx vs B.rx)
            max_ab = max(a_tx, b_rx)
            thr_rel_ab = TH_REL_LOW if max_ab < LOW_RATE_MAX else TH_REL_DEFAULT
            d_ab = norm_diff(a_tx, b_rx)
            abs_ab = abs(a_tx - b_rx)

            if (d_ab > thr_rel_ab) and (abs_ab > TH_ABS_GUARD):
                # Repair by consensus average with partial averaging near threshold
                v = 0.5 * (a_tx + b_rx)
                if d_ab <= 2.0 * thr_rel_ab:
                    k = (d_ab - thr_rel_ab) / max(thr_rel_ab, 1e-9)
                    rep_a_tx = a_tx * (1.0 - k) + v * k
                    rep_b_rx = b_rx * (1.0 - k) + v * k
                else:
                    rep_a_tx = v
                    rep_b_rx = v
            # Confidence floors and violation-based reduction
            conf_ab = max(0.0, 1.0 - d_ab)
            if max_ab >= LOW_RATE_MAX and d_ab <= STRONG_AGREE_DIFF:
                conf_ab = max(conf_ab, 0.99)
            elif d_ab <= thr_rel_ab:
                conf_ab = max(conf_ab, 0.98 if max_ab >= LOW_RATE_MAX else 0.97)
            tx_conf_a = min(1.0, conf_ab)
            rx_conf_b = min(1.0, conf_ab)

            # Direction B->A (B.tx vs A.rx)
            max_ba = max(b_tx, a_rx)
            thr_rel_ba = TH_REL_LOW if max_ba < LOW_RATE_MAX else TH_REL_DEFAULT
            d_ba = norm_diff(b_tx, a_rx)
            abs_ba = abs(b_tx - a_rx)

            if (d_ba > thr_rel_ba) and (abs_ba > TH_ABS_GUARD):
                v2 = 0.5 * (b_tx + a_rx)
                if d_ba <= 2.0 * thr_rel_ba:
                    k2 = (d_ba - thr_rel_ba) / max(thr_rel_ba, 1e-9)
                    rep_b_tx = b_tx * (1.0 - k2) + v2 * k2
                    rep_a_rx = a_rx * (1.0 - k2) + v2 * k2
                else:
                    rep_b_tx = v2
                    rep_a_rx = v2
            conf_ba = max(0.0, 1.0 - d_ba)
            if max_ba >= LOW_RATE_MAX and d_ba <= STRONG_AGREE_DIFF:
                conf_ba = max(conf_ba, 0.99)
            elif d_ba <= thr_rel_ba:
                conf_ba = max(conf_ba, 0.98 if max_ba >= LOW_RATE_MAX else 0.97)
            tx_conf_b = min(1.0, conf_ba)
            rx_conf_a = min(1.0, conf_ba)

            # Asymmetric confidence reduction when only one side has traffic evidence
            if a_has != b_has:
                if not a_has:
                    rx_conf_a *= 0.88
                    tx_conf_a *= 0.88
                if not b_has:
                    rx_conf_b *= 0.88
                    tx_conf_b *= 0.88

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

        # Confidence refinement: penalize by magnitude of applied change to improve calibration
        a_rx_change = change_ratio(a_rx, rep_a_rx)
        a_tx_change = change_ratio(a_tx, rep_a_tx)
        b_rx_change = change_ratio(b_rx, rep_b_rx)
        b_tx_change = change_ratio(b_tx, rep_b_tx)

        rx_conf_a = max(0.0, min(1.0, min(rx_conf_a, 1.0 - a_rx_change)))
        tx_conf_a = max(0.0, min(1.0, min(tx_conf_a, 1.0 - a_tx_change)))
        rx_conf_b = max(0.0, min(1.0, min(rx_conf_b, 1.0 - b_rx_change)))
        tx_conf_b = max(0.0, min(1.0, min(tx_conf_b, 1.0 - b_tx_change)))

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
        rx = float(data.get('rx_rate', 0.0) or 0.0)
        tx = float(data.get('tx_rate', 0.0) or 0.0)
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

    # Router-level balancing using dangling interfaces to satisfy flow conservation (R1)
    # We only adjust interfaces without a known peer to avoid breaking link symmetry.
    paired_ids = set()
    for _, (aid, bid) in pairs.items():
        paired_ids.add(aid)
        paired_ids.add(bid)

    for router, if_list in topology.items():
        # Compute current router totals
        sum_tx = 0.0
        sum_rx = 0.0
        present = [iid for iid in if_list if iid in per_if_repair]
        if not present:
            continue
        for iid in present:
            rep = per_if_repair[iid]
            sum_tx += float(rep['repaired_tx'])
            sum_rx += float(rep['repaired_rx'])
        scale = max(sum_tx, sum_rx, 1.0)
        delta = sum_tx - sum_rx  # want delta -> 0
        resid_frac = abs(delta) / scale

        # Stronger trigger: only attempt micro-adjustment for noticeable imbalance (≥3%)
        if resid_frac < 0.03:
            continue

        # Candidate interfaces for adjustment: dangling and up
        dangling = [iid for iid in present
                    if (iid not in paired_ids) and (per_if_repair[iid].get('repaired_status', 'up') != 'down')]

        if not dangling:
            continue

        # Determine dominance among dangling interfaces
        weights = []
        sum_w = 0.0
        for iid in dangling:
            rep = per_if_repair[iid]
            w = float(rep['repaired_tx']) + float(rep['repaired_rx']) + EPS
            weights.append((iid, w))
            sum_w += w
        if sum_w <= 0.0:
            continue  # nothing meaningful to adjust

        # Pick dominant interface and require >60% share among dangling traffic
        dom_iid, dom_w = max(weights, key=lambda x: x[1])
        dom_share = dom_w / max(sum_w, EPS)
        if dom_share <= 0.6:
            continue

        # Tentative nudge capped at 2% of router scale
        alpha = min(0.02 * scale, abs(delta))

        # Compute tentative new residual after the nudge
        if delta > 0.0:
            # Too much TX; increase RX on dominant dangling interface
            new_sum_rx = sum_rx + alpha
            new_sum_tx = sum_tx
            new_delta = new_sum_tx - new_sum_rx
        else:
            # Too much RX; increase TX on dominant dangling interface
            new_sum_tx = sum_tx + alpha
            new_sum_rx = sum_rx
            new_delta = new_sum_tx - new_sum_rx

        new_scale = max(new_sum_tx, new_sum_rx, 1.0)
        new_resid_frac = abs(new_delta) / new_scale

        # Commit only if residual improves by at least 10% relative
        if new_resid_frac <= 0.9 * resid_frac:
            if delta > 0.0:
                new_rx = float(per_if_repair[dom_iid]['repaired_rx']) + alpha
                per_if_repair[dom_iid]['repaired_rx'] = new_rx
                # Confidence: penalize by change magnitude from original observation
                rx_orig = float(telemetry.get(dom_iid, {}).get('rx_rate', 0.0) or 0.0)
                cr = change_ratio(rx_orig, new_rx)
                per_if_repair[dom_iid]['rx_conf'] = max(0.0, min(float(per_if_repair[dom_iid].get('rx_conf', 0.6)), 1.0 - cr))
            else:
                new_tx = float(per_if_repair[dom_iid]['repaired_tx']) + alpha
                per_if_repair[dom_iid]['repaired_tx'] = new_tx
                tx_orig = float(telemetry.get(dom_iid, {}).get('tx_rate', 0.0) or 0.0)
                cr = change_ratio(tx_orig, new_tx)
                per_if_repair[dom_iid]['tx_conf'] = max(0.0, min(float(per_if_repair[dom_iid].get('tx_conf', 0.6)), 1.0 - cr))
        # else: skip adjustment to avoid degrading counters on noise

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
        repaired_rx = repaired.get('repaired_rx', float(data.get('rx_rate', 0.0) or 0.0))
        repaired_tx = repaired.get('repaired_tx', float(data.get('tx_rate', 0.0) or 0.0))
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

        # Direction-aware dynamic checking penalties: TX weighted by local residual, RX by remote
        pen_tx = 1.0 - (0.6 * resid_local + 0.4 * resid_remote)
        pen_rx = 1.0 - (0.6 * resid_remote + 0.4 * resid_local)
        pen_tx = max(0.0, min(1.0, pen_tx))
        pen_rx = max(0.0, min(1.0, pen_rx))

        rx_conf = max(0.0, min(1.0, rx_conf * pen_rx))
        tx_conf = max(0.0, min(1.0, tx_conf * pen_tx))
        # Status confidence is less sensitive to flow residuals; apply mild scaling using weaker direction
        status_conf = max(0.0, min(1.0, status_conf * (0.85 + 0.15 * min(pen_tx, pen_rx))))

        repaired_data: Dict[str, Any] = {}
        # Store repaired values with confidence scores
        rx_orig = float(data.get('rx_rate', 0.0) or 0.0)
        tx_orig = float(data.get('tx_rate', 0.0) or 0.0)
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