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
    EPS = 1e-9

    # Helpers
    def norm_status(s: Any) -> str:
        s = str(s).lower()
        return s if s in ("up", "down") else "up"  # treat unknown as up to avoid over-zeroing

    def nz_float(x: Any) -> float:
        try:
            v = float(x)
        except Exception:
            v = 0.0
        return max(0.0, v)

    def rel_diff(a: float, b: float) -> float:
        a = float(a)
        b = float(b)
        denom = max(abs(a), abs(b), 1.0)
        return abs(a - b) / denom

    def clamp(x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))

    def pair_tolerance(a: float, b: float) -> float:
        # Rate-aware tolerance: looser for very low traffic, at least HARDENING_THRESHOLD
        traffic = max(abs(a), abs(b), 1.0)
        return max(HARDENING_THRESHOLD, 2.5 / traffic)

    # Precompute peer mapping
    peers: Dict[str, str] = {iface: data.get('connected_to') for iface, data in telemetry.items()}

    # First pass: link-level hardening with status-aware zeroing and targeted substitution
    pre: Dict[str, Dict[str, Any]] = {}
    for iface, data in telemetry.items():
        local_status = norm_status(data.get('interface_status', 'unknown'))
        rx_orig = nz_float(data.get('rx_rate', 0.0))
        tx_orig = nz_float(data.get('tx_rate', 0.0))

        peer_id = peers.get(iface)
        had_peer = bool(peer_id and peer_id in telemetry)
        peer_status = 'unknown'
        peer_rx = peer_tx = 0.0

        if had_peer:
            p = telemetry[peer_id]
            peer_status = norm_status(p.get('interface_status', 'unknown'))
            peer_rx = nz_float(p.get('rx_rate', 0.0))
            peer_tx = nz_float(p.get('tx_rate', 0.0))

        pre_rx = rx_orig
        pre_tx = tx_orig
        rx_link_diff = 0.0
        tx_link_diff = 0.0

        if local_status == 'down':
            pre_rx = 0.0
            pre_tx = 0.0
        else:
            if had_peer and peer_status == 'up':
                # Link symmetry checks
                rx_link_diff = rel_diff(rx_orig, peer_tx)  # my RX vs peer TX
                tx_link_diff = rel_diff(tx_orig, peer_rx)  # my TX vs peer RX

                pair_rx_ok = rx_link_diff <= HARDENING_THRESHOLD
                pair_tx_ok = tx_link_diff <= HARDENING_THRESHOLD

                # Targeted substitution: fix only the violating direction
                if not pair_rx_ok and pair_tx_ok:
                    pre_rx = peer_tx
                elif not pair_tx_ok and pair_rx_ok:
                    pre_tx = peer_rx
                elif not pair_rx_ok and not pair_tx_ok:
                    # Both directions off -> average to reduce single-sided large errors
                    pre_rx = 0.5 * (rx_orig + peer_tx)
                    pre_tx = 0.5 * (tx_orig + peer_rx)
                # else both ok -> leave as-is

        pre[iface] = {
            'pre_rx': pre_rx,
            'pre_tx': pre_tx,
            'rx_link_diff': rx_link_diff,
            'tx_link_diff': tx_link_diff,
            'local_status': local_status,
            'peer_status': peer_status if had_peer else 'unknown',
            'had_peer': had_peer,
        }

    # Router-level multiplicative pre-step (flow conservation) with guarded caps
    # We adaptively scale only one direction (rx or tx) per router, picking the less-trusted direction.
    post_router: Dict[str, Dict[str, float]] = {i: {'rx': pre[i]['pre_rx'], 'tx': pre[i]['pre_tx']} for i in telemetry}
    router_imbalance_pre: Dict[str, float] = {}
    router_scale_dir: Dict[str, str] = {}  # 'rx' or 'tx' for diagnostics
    # Track per-interface relative delta from router scaling, used for confidence calibration
    router_delta_rel: Dict[str, Dict[str, float]] = {i: {'rx': 0.0, 'tx': 0.0} for i in telemetry}

    for router_id, iface_list in topology.items():
        # Select interfaces present and locally up
        up_ifaces = [i for i in iface_list if i in telemetry and pre[i]['local_status'] == 'up']
        if len(up_ifaces) < 2:
            router_imbalance_pre[router_id] = 0.0
            continue

        sum_rx = sum(pre[i]['pre_rx'] for i in up_ifaces)
        sum_tx = sum(pre[i]['pre_tx'] for i in up_ifaces)

        denom = max(sum_rx, sum_tx, 1.0)
        imb = abs(sum_tx - sum_rx) / denom
        router_imbalance_pre[router_id] = imb

        # Skip tiny traffic or minor imbalance
        if (sum_rx + sum_tx) <= 1e-6 or imb <= HARDENING_THRESHOLD:
            continue

        # Determine less trusted direction using peer residuals when available
        rx_resids = []
        tx_resids = []
        for i in up_ifaces:
            peer_id = peers.get(i)
            if peer_id and peer_id in pre and pre[peer_id]['local_status'] == 'up':
                # My TX should match peer RX; my RX should match peer TX
                tx_resids.append(rel_diff(pre[i]['pre_tx'], pre[peer_id]['pre_rx']))
                rx_resids.append(rel_diff(pre[i]['pre_rx'], pre[peer_id]['pre_tx']))

        avg_tx_resid = sum(tx_resids) / len(tx_resids) if tx_resids else None
        avg_rx_resid = sum(rx_resids) / len(rx_resids) if rx_resids else None

        # Choose direction to scale
        dir_to_scale = None
        if avg_tx_resid is not None and avg_rx_resid is not None:
            if avg_tx_resid > avg_rx_resid + 0.02:
                dir_to_scale = 'tx'
            elif avg_rx_resid > avg_tx_resid + 0.02:
                dir_to_scale = 'rx'
        if dir_to_scale is None:
            # Fallback: scale the larger aggregate to match the smaller
            dir_to_scale = 'tx' if sum_tx > sum_rx else 'rx'

        router_scale_dir[router_id] = dir_to_scale

        # Compute multiplicative target scale factor for the chosen direction
        if dir_to_scale == 'tx':
            s = sum_rx / max(sum_tx, EPS)
        else:
            s = sum_tx / max(sum_rx, EPS)

        # Bound the raw router ratio and temper with imbalance to avoid overshoot
        s_bounded = clamp(s, 0.85, 1.15)
        alpha = clamp(imb / 0.15, 0.25, 0.6)  # stronger when imbalance is large
        k = 1.0 + alpha * (s_bounded - 1.0)

        # Apply per-interface capped scaling on the chosen direction
        for i in up_ifaces:
            if dir_to_scale == 'tx':
                old = post_router[i]['tx']
                new = old * k
                # Cap per-interface multiplicative delta to ±15%
                new_capped = clamp(new, old * 0.85, old * 1.15)
                post_router[i]['tx'] = max(0.0, new_capped)
                router_delta_rel[i]['tx'] = abs(post_router[i]['tx'] - old) / max(old, 1.0)
            else:
                old = post_router[i]['rx']
                new = old * k
                new_capped = clamp(new, old * 0.85, old * 1.15)
                post_router[i]['rx'] = max(0.0, new_capped)
                router_delta_rel[i]['rx'] = abs(post_router[i]['rx'] - old) / max(old, 1.0)

    # Uncertainty-weighted additive redistribution (router-level) to further enforce R1
    for router_id, iface_list in topology.items():
        # Active, locally up interfaces
        up_ifaces = [i for i in iface_list if i in telemetry and pre[i]['local_status'] == 'up']
        if len(up_ifaces) < 2:
            continue

        sum_rx = sum(post_router[i]['rx'] for i in up_ifaces)
        sum_tx = sum(post_router[i]['tx'] for i in up_ifaces)
        total = sum_rx + sum_tx
        if total <= 1e-6:
            continue

        imbalance = abs(sum_tx - sum_rx) / max(sum_rx, sum_tx, 1.0)
        if imbalance <= HARDENING_THRESHOLD:
            continue

        # Choose direction with lower average link-confidence; tie-break by larger absolute aggregate
        confs_tx = [max(0.0, 1.0 - pre[i]['tx_link_diff']) for i in up_ifaces]
        confs_rx = [max(0.0, 1.0 - pre[i]['rx_link_diff']) for i in up_ifaces]
        avg_conf_tx = sum(confs_tx) / len(confs_tx) if confs_tx else 0.0
        avg_conf_rx = sum(confs_rx) / len(confs_rx) if confs_rx else 0.0

        dir_to_adj = None
        if (avg_conf_tx + 0.05) < avg_conf_rx:
            dir_to_adj = 'tx'
        elif (avg_conf_rx + 0.05) < avg_conf_tx:
            dir_to_adj = 'rx'
        else:
            dir_to_adj = 'tx' if sum_tx > sum_rx else 'rx'

        # Compute additive need on chosen direction (positive => increase that direction)
        if dir_to_adj == 'tx':
            need = sum_rx - sum_tx
            sum_dir = sum_tx
        else:
            need = sum_tx - sum_rx
            sum_dir = sum_rx

        # Skip tiny needs relative to volume (guard against noise)
        if abs(need) <= max(1e-6, HARDENING_THRESHOLD * max(sum_rx, sum_tx, 1.0)):
            continue

        # Build weights and caps: focus on least-trusted and material-volume edges
        cap_ratio = 0.15  # per-interface absolute delta cap (±15% of current)
        weights = {}
        caps = {}
        values = {}
        for i in up_ifaces:
            v = post_router[i][dir_to_adj]
            values[i] = v
            conf = max(0.0, 1.0 - (pre[i]['tx_link_diff'] if dir_to_adj == 'tx' else pre[i]['rx_link_diff']))
            w = 0.6 * (1.0 - conf) + 0.4 * (v / max(sum_dir, 1e-9))
            if v < 1.0:
                w *= 0.5  # protect tiny links
            weights[i] = max(0.0, w)
            caps[i] = cap_ratio * max(v, 1.0)

        # Iterative water-filling allocation honoring per-interface caps
        remain = need
        delta = {i: 0.0 for i in up_ifaces}
        active = set(i for i in up_ifaces if caps[i] > 0.0)
        for _ in range(3):
            if not active or abs(remain) <= 1e-9:
                break
            sum_w = sum(weights[i] for i in active)
            if sum_w <= 0.0:
                break
            progress = 0.0
            for i in list(active):
                raw = remain * (weights[i] / sum_w)
                # remaining capacity for this interface
                cap_rem = caps[i] - abs(delta[i])
                if cap_rem <= 0.0:
                    active.discard(i)
                    continue
                bounded = clamp(raw, -cap_rem, cap_rem)
                if abs(bounded) > 0.0:
                    delta[i] += bounded
                    progress += abs(bounded)
            remain = need - sum(delta.values())
            # Remove saturated
            for i in list(active):
                if abs(delta[i]) >= caps[i] * 0.999:
                    active.discard(i)
            if progress <= 1e-9:
                break

        # Apply deltas and track relative movement for confidence calibration
        for i in up_ifaces:
            if delta[i] != 0.0:
                prev = post_router[i][dir_to_adj]
                post_router[i][dir_to_adj] = max(0.0, prev + delta[i])
                router_delta_rel[i][dir_to_adj] += abs(delta[i]) / max(prev, 1.0)

    # Build values after router passes
    post = {i: {'rx': post_router[i]['rx'], 'tx': post_router[i]['tx']} for i in telemetry}

    # Targeted pair reconciliation only on links touched by router edits
    # Track magnitude of pair reconciliation for calibration
    pair_adj_rel: Dict[str, Dict[str, float]] = {i: {'rx': 0.0, 'tx': 0.0} for i in telemetry}

    visited_pairs = set()
    for iface in telemetry:
        peer_id = peers.get(iface)
        if not (peer_id and peer_id in telemetry):
            continue
        pair_key = tuple(sorted([iface, peer_id]))
        if pair_key in visited_pairs:
            continue
        visited_pairs.add(pair_key)

        # Only reconcile when both sides are locally up
        if pre[iface]['local_status'] != 'up' or pre[peer_id]['local_status'] != 'up':
            continue

        # Only reconcile if router pass touched at least one side in either direction
        touched = (router_delta_rel[iface]['rx'] > 0 or router_delta_rel[iface]['tx'] > 0 or
                   router_delta_rel[peer_id]['rx'] > 0 or router_delta_rel[peer_id]['tx'] > 0)
        if not touched:
            continue

        # Compute residuals after router scaling
        a_rx, a_tx = post[iface]['rx'], post[iface]['tx']
        b_rx, b_tx = post[peer_id]['rx'], post[peer_id]['tx']

        # A.tx ↔ B.rx
        resid_tx = rel_diff(a_tx, b_rx)
        tol_tx = pair_tolerance(a_tx, b_rx)
        if resid_tx > tol_tx:
            # Confidence proxies from pre-pass diffs (smaller diff => higher confidence)
            conf_a_tx = max(0.0, 1.0 - pre[iface]['tx_link_diff'])
            conf_b_rx = max(0.0, 1.0 - pre[peer_id]['rx_link_diff'])
            # Lower-confidence side moves more
            if conf_a_tx <= conf_b_rx:
                alpha_low, alpha_high = 0.35, 0.20  # move toward midpoint
                move_a = alpha_low * (b_rx - a_tx)
                move_b = alpha_high * (a_tx - b_rx)
            else:
                alpha_low, alpha_high = 0.35, 0.20
                move_a = alpha_high * (b_rx - a_tx)
                move_b = alpha_low * (a_tx - b_rx)

            # Cap moves to remaining per-interface budget (≤ 20% of current value)
            cap_a = 0.2 * max(a_tx, 1.0)
            cap_b = 0.2 * max(b_rx, 1.0)
            move_a = clamp(move_a, -cap_a, cap_a)
            move_b = clamp(move_b, -cap_b, cap_b)

            new_a_tx = max(0.0, a_tx + move_a)
            new_b_rx = max(0.0, b_rx + move_b)

            pair_adj_rel[iface]['tx'] = max(pair_adj_rel[iface]['tx'], abs(new_a_tx - a_tx) / max(a_tx, 1.0))
            pair_adj_rel[peer_id]['rx'] = max(pair_adj_rel[peer_id]['rx'], abs(new_b_rx - b_rx) / max(b_rx, 1.0))

            post[iface]['tx'] = new_a_tx
            post[peer_id]['rx'] = new_b_rx

        # A.rx ↔ B.tx
        a_rx, a_tx = post[iface]['rx'], post[iface]['tx']
        b_rx, b_tx = post[peer_id]['rx'], post[peer_id]['tx']
        resid_rx = rel_diff(a_rx, b_tx)
        tol_rx = pair_tolerance(a_rx, b_tx)
        if resid_rx > tol_rx:
            conf_a_rx = max(0.0, 1.0 - pre[iface]['rx_link_diff'])
            conf_b_tx = max(0.0, 1.0 - pre[peer_id]['tx_link_diff'])
            if conf_a_rx <= conf_b_tx:
                alpha_low, alpha_high = 0.35, 0.20
                move_a = alpha_low * (b_tx - a_rx)
                move_b = alpha_high * (a_rx - b_tx)
            else:
                alpha_low, alpha_high = 0.35, 0.20
                move_a = alpha_high * (b_tx - a_rx)
                move_b = alpha_low * (a_rx - b_tx)

            cap_a = 0.2 * max(a_rx, 1.0)
            cap_b = 0.2 * max(b_tx, 1.0)
            move_a = clamp(move_a, -cap_a, cap_a)
            move_b = clamp(move_b, -cap_b, cap_b)

            new_a_rx = max(0.0, a_rx + move_a)
            new_b_tx = max(0.0, b_tx + move_b)

            pair_adj_rel[iface]['rx'] = max(pair_adj_rel[iface]['rx'], abs(new_a_rx - a_rx) / max(a_rx, 1.0))
            pair_adj_rel[peer_id]['tx'] = max(pair_adj_rel[peer_id]['tx'], abs(new_b_tx - b_tx) / max(b_tx, 1.0))

            post[iface]['rx'] = new_a_rx
            post[peer_id]['tx'] = new_b_tx

    # Recompute router imbalance after all passes for confidence factor
    router_imbalance_post: Dict[str, float] = {}
    for router_id, iface_list in topology.items():
        up_ifaces = [i for i in iface_list if i in telemetry and pre[i]['local_status'] == 'up']
        if not up_ifaces:
            router_imbalance_post[router_id] = 0.0
            continue
        sum_rx = sum(post[i]['rx'] for i in up_ifaces)
        sum_tx = sum(post[i]['tx'] for i in up_ifaces)
        denom = max(sum_rx, sum_tx, 1.0)
        router_imbalance_post[router_id] = abs(sum_tx - sum_rx) / denom

    # Assemble final results with calibrated confidence
    result: Dict[str, Dict[str, Tuple]] = {}
    for iface, data in telemetry.items():
        local_status = pre[iface]['local_status']
        peer_id = peers.get(iface)
        had_peer = bool(peer_id and peer_id in telemetry)
        peer_status = norm_status(telemetry[peer_id].get('interface_status', 'unknown')) if had_peer else 'unknown'

        rx_orig = nz_float(data.get('rx_rate', 0.0))
        tx_orig = nz_float(data.get('tx_rate', 0.0))

        rx_repaired = post[iface]['rx']
        tx_repaired = post[iface]['tx']

        # Enforce zero on down interfaces
        repaired_status = data.get('interface_status', 'unknown')
        if norm_status(repaired_status) == 'down':
            rx_repaired = 0.0
            tx_repaired = 0.0

        # Confidence based on post-repair residuals vs peer
        if had_peer and local_status == 'up' and peer_status == 'up':
            peer_tx_after = post[peer_id]['tx']
            peer_rx_after = post[peer_id]['rx']
            rx_resid = rel_diff(rx_repaired, peer_tx_after)  # my RX vs peer TX
            tx_resid = rel_diff(tx_repaired, peer_rx_after)  # my TX vs peer RX
            rx_link_conf = max(0.0, 1.0 - rx_resid)
            tx_link_conf = max(0.0, 1.0 - tx_resid)
        elif norm_status(repaired_status) == 'down':
            rx_link_conf = 0.9 if rx_repaired == 0.0 else 0.5
            tx_link_conf = 0.9 if tx_repaired == 0.0 else 0.5
        else:
            # No reliable peer info
            rx_link_conf = 0.65 if abs(rx_repaired - rx_orig) < 1e-9 else 0.6
            tx_link_conf = 0.65 if abs(tx_repaired - tx_orig) < 1e-9 else 0.6

        # Router imbalance factor (post)
        router_id = data.get('local_router')
        imb_post = router_imbalance_post.get(router_id, 0.0)
        router_factor = max(0.2, 1.0 - imb_post)

        # Change penalties from original
        rx_change = rel_diff(rx_orig, rx_repaired)
        tx_change = rel_diff(tx_orig, tx_repaired)
        rx_change_factor = max(0.2, 1.0 - 0.5 * min(1.0, rx_change))
        tx_change_factor = max(0.2, 1.0 - 0.5 * min(1.0, tx_change))

        rx_conf = rx_link_conf * router_factor * rx_change_factor
        tx_conf = tx_link_conf * router_factor * tx_change_factor

        # Penalties for heavy router scaling usage and pair adjustments (helps calibration)
        rdel_rx = router_delta_rel[iface]['rx']
        rdel_tx = router_delta_rel[iface]['tx']
        if rdel_rx >= 0.12:
            rx_conf -= 0.03
        elif rdel_rx >= 0.07:
            rx_conf -= 0.02
        if rdel_tx >= 0.12:
            tx_conf -= 0.03
        elif rdel_tx >= 0.07:
            tx_conf -= 0.02

        padj_rx = pair_adj_rel[iface]['rx']
        padj_tx = pair_adj_rel[iface]['tx']
        rx_conf -= min(0.05, 0.25 * padj_rx)
        tx_conf -= min(0.05, 0.25 * padj_tx)

        # Bonus for untouched directions across all passes
        pre_unchanged_rx = abs(pre[iface]['pre_rx'] - rx_orig) < 1e-9
        pre_unchanged_tx = abs(pre[iface]['pre_tx'] - tx_orig) < 1e-9
        if pre_unchanged_rx and rdel_rx == 0.0 and padj_rx == 0.0:
            rx_conf += 0.03
        if pre_unchanged_tx and rdel_tx == 0.0 and padj_tx == 0.0:
            tx_conf += 0.03

        rx_confidence = max(0.0, min(1.0, rx_conf))
        tx_confidence = max(0.0, min(1.0, tx_conf))

        # Status confidence calibration (keep status unchanged)
        status_confidence = 1.0
        if had_peer:
            peer_status_raw = peer_status
            if norm_status(repaired_status) != peer_status_raw:
                status_confidence = min(status_confidence, 0.5)
        if norm_status(repaired_status) == 'down' and (rx_orig > 0.0 or tx_orig > 0.0):
            status_confidence = min(status_confidence, 0.6)

        # Build output
        repaired_entry: Dict[str, Tuple] = {}
        repaired_entry['rx_rate'] = (rx_orig, rx_repaired, rx_confidence)
        repaired_entry['tx_rate'] = (tx_orig, tx_repaired, tx_confidence)
        repaired_entry['interface_status'] = (data.get('interface_status', 'unknown'), repaired_status, status_confidence)

        # Copy metadata unchanged
        repaired_entry['connected_to'] = data.get('connected_to')
        repaired_entry['local_router'] = data.get('local_router')
        repaired_entry['remote_router'] = data.get('remote_router')

        result[iface] = repaired_entry

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
