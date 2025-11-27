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
    # Parameters (tunable)
    HARDENING_THRESHOLD = 0.02        # base timing tolerance
    EPS = 1e-9
    PRESTEP_RATIO_CLAMP = (0.5, 2.0)  # s_bounded clamp for router pre-step
    PER_IFACE_CAP = 0.15              # per-interface cap for router pre-step (relative)
    FINAL_SCALE_CLAMP = (0.85, 1.15)  # final uniform router scaling clamp
    PAIR_CAP = 0.20                   # per-interface pair reconciliation cap (absolute, as fraction of pre value)
    ALPHA_LOW = 0.35                  # lower-confidence side move fraction
    ALPHA_HIGH = 0.20                 # higher-confidence side move fraction

    # Helpers
    def norm_status(s: Any) -> str:
        s = str(s).lower()
        return s if s in ("up", "down") else "up"

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
        traffic = max(abs(a), abs(b), 1.0)
        return max(HARDENING_THRESHOLD, 2.5 / traffic)

    # Precompute peer mapping
    peers: Dict[str, str] = {iface: data.get('connected_to') for iface, data in telemetry.items()}

    # First pass: link-level hardening with tolerance and status-aware zeroing
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
        pre_rx_changed = False
        pre_tx_changed = False

        if local_status == 'down':
            pre_rx = 0.0
            pre_tx = 0.0
            pre_rx_changed = abs(pre_rx - rx_orig) > 1e-12
            pre_tx_changed = abs(pre_tx - tx_orig) > 1e-12
        else:
            if had_peer and peer_status == 'up':
                # A.rx ≈ B.tx, A.tx ≈ B.rx
                rx_link_diff = rel_diff(rx_orig, peer_tx)
                tx_link_diff = rel_diff(tx_orig, peer_rx)
                tol_rx = pair_tolerance(rx_orig, peer_tx)
                tol_tx = pair_tolerance(tx_orig, peer_rx)
                pair_rx_ok = rx_link_diff <= tol_rx
                pair_tx_ok = tx_link_diff <= tol_tx

                if pair_tx_ok and not pair_rx_ok:
                    pre_rx = peer_tx
                    pre_rx_changed = abs(pre_rx - rx_orig) > 1e-12
                elif pair_rx_ok and not pair_tx_ok:
                    pre_tx = peer_rx
                    pre_tx_changed = abs(pre_tx - tx_orig) > 1e-12
                elif not pair_rx_ok and not pair_tx_ok:
                    # Weighted average: trust direction with smaller diff more
                    w_rx = clamp(1.0 - rx_link_diff, 0.0, 1.0)
                    w_tx = clamp(1.0 - tx_link_diff, 0.0, 1.0)
                    # For rx, mix rx_orig with peer_tx; for tx, mix tx_orig with peer_rx
                    pre_rx = (w_rx * peer_tx + (1.0 - w_rx) * rx_orig)
                    pre_tx = (w_tx * peer_rx + (1.0 - w_tx) * tx_orig)
                    pre_rx_changed = abs(pre_rx - rx_orig) > 1e-12
                    pre_tx_changed = abs(pre_tx - tx_orig) > 1e-12
            # else: no reliable peer, leave as-is

        pre[iface] = {
            'pre_rx': pre_rx,
            'pre_tx': pre_tx,
            'rx_link_diff': rx_link_diff,
            'tx_link_diff': tx_link_diff,
            'local_status': local_status,
            'peer_status': peer_status if had_peer else 'unknown',
            'had_peer': had_peer,
            'pre_rx_changed': pre_rx_changed,
            'pre_tx_changed': pre_tx_changed,
        }

    # Second pass: router-level flow conservation using topology
    # Multiplicative bounded pre-step (per-interface, weighted, capped), then mild uniform finish scaling.
    router_imbalance: Dict[str, float] = {}
    post_router: Dict[str, Dict[str, float]] = {i: {'rx': pre[i]['pre_rx'], 'tx': pre[i]['pre_tx']} for i in telemetry}
    router_delta_rel: Dict[str, Dict[str, float]] = {i: {'rx': 0.0, 'tx': 0.0} for i in telemetry}
    router_delta_abs: Dict[str, Dict[str, float]] = {i: {'rx': 0.0, 'tx': 0.0} for i in telemetry}
    s_bounded_map: Dict[str, float] = {}

    for router_id, iface_list in topology.items():
        up_ifaces = [i for i in iface_list if i in telemetry and pre[i]['local_status'] == 'up']
        if len(up_ifaces) < 2:
            router_imbalance[router_id] = 0.0
            continue

        sum_rx = sum(pre[i]['pre_rx'] for i in up_ifaces)
        sum_tx = sum(pre[i]['pre_tx'] for i in up_ifaces)
        denom = max(sum_rx, sum_tx, 1.0)
        imbalance = abs(sum_tx - sum_rx) / denom
        router_imbalance[router_id] = imbalance

        if (sum_rx + sum_tx) <= 1e-6 or imbalance <= HARDENING_THRESHOLD:
            continue

        # Estimate avg directional confidences from link symmetry
        rx_confs: List[float] = []
        tx_confs: List[float] = []
        for i in up_ifaces:
            peer_id = peers.get(i)
            if peer_id and peer_id in telemetry and pre[peer_id]['local_status'] == 'up':
                rx_resid = rel_diff(pre[i]['pre_rx'], pre[peer_id]['pre_tx'])
                tx_resid = rel_diff(pre[i]['pre_tx'], pre[peer_id]['pre_rx'])
                rx_confs.append(max(0.0, 1.0 - rx_resid))
                tx_confs.append(max(0.0, 1.0 - tx_resid))
        avg_rx_conf = sum(rx_confs) / len(rx_confs) if rx_confs else 0.5
        avg_tx_conf = sum(tx_confs) / len(tx_confs) if tx_confs else 0.5

        # Choose direction to scale
        dir_to_scale = None
        if abs(avg_tx_conf - avg_rx_conf) > 0.05:
            dir_to_scale = 'tx' if avg_tx_conf < avg_rx_conf else 'rx'
        else:
            dir_to_scale = 'tx' if sum_tx > sum_rx else 'rx'

        # Pre-step bounded ratio and tempered factor
        if dir_to_scale == 'tx':
            s = sum_rx / max(sum_tx, EPS)
        else:
            s = sum_tx / max(sum_rx, EPS)
        s_bounded = clamp(s, PRESTEP_RATIO_CLAMP[0], PRESTEP_RATIO_CLAMP[1])
        alpha = clamp(imbalance / 0.15, 0.25, 0.6)
        k_pre = 1.0 + alpha * (s_bounded - 1.0)
        s_bounded_map[router_id] = s_bounded

        # Build totals for volumetric weighting on chosen dir
        if dir_to_scale == 'tx':
            sum_dir = sum(post_router[i]['tx'] for i in up_ifaces)
        else:
            sum_dir = sum(post_router[i]['rx'] for i in up_ifaces)
        sum_dir = max(sum_dir, 1.0)

        # Apply weighted, capped pre-step
        for i in up_ifaces:
            peer_id = peers.get(i)
            if dir_to_scale == 'tx':
                old = post_router[i]['tx']
                # Directional confidence proxy from pre-pass
                conf_dir = max(0.0, 1.0 - pre[i]['tx_link_diff'])
                if peer_id and peer_id in telemetry and pre[peer_id]['local_status'] == 'up':
                    resid = rel_diff(pre[i]['pre_tx'], pre[peer_id]['pre_rx'])
                    tol = pair_tolerance(pre[i]['pre_tx'], pre[peer_id]['pre_rx'])
                    sev = min(2.0, resid / max(tol, EPS))
                else:
                    sev = 0.0
                vol = old / sum_dir
                w = 0.6 * (1.0 - conf_dir) + 0.25 * sev + 0.15 * vol
                if old < 1.0:
                    w *= 0.5
                w = max(0.02, w)
                delta_target = old * (k_pre - 1.0)
                cap_abs = PER_IFACE_CAP * max(old, 1.0)
                move = clamp(delta_target * w, -cap_abs, cap_abs)
                new_val = max(0.0, old + move)
                post_router[i]['tx'] = new_val
                router_delta_abs[i]['tx'] += abs(move)
                router_delta_rel[i]['tx'] += abs(move) / max(old, 1.0)
            else:
                old = post_router[i]['rx']
                conf_dir = max(0.0, 1.0 - pre[i]['rx_link_diff'])
                if peer_id and peer_id in telemetry and pre[peer_id]['local_status'] == 'up':
                    resid = rel_diff(pre[i]['pre_rx'], pre[peer_id]['pre_tx'])
                    tol = pair_tolerance(pre[i]['pre_rx'], pre[peer_id]['pre_tx'])
                    sev = min(2.0, resid / max(tol, EPS))
                else:
                    sev = 0.0
                vol = old / sum_dir
                w = 0.6 * (1.0 - conf_dir) + 0.25 * sev + 0.15 * vol
                if old < 1.0:
                    w *= 0.5
                w = max(0.02, w)
                delta_target = old * (k_pre - 1.0)
                cap_abs = PER_IFACE_CAP * max(old, 1.0)
                move = clamp(delta_target * w, -cap_abs, cap_abs)
                new_val = max(0.0, old + move)
                post_router[i]['rx'] = new_val
                router_delta_abs[i]['rx'] += abs(move)
                router_delta_rel[i]['rx'] += abs(move) / max(old, 1.0)

        # Uniform finish scaling to reduce residual imbalance (mild, clamped)
        sum_rx2 = sum(post_router[i]['pre_rx' if False else 'rx'] if dir_to_scale == 'rx' else post_router[i]['rx'] for i in [])  # dummy to keep structure clear
        # Compute new sums after pre-step
        sum_rx_after = sum(post_router[i]['pre_rx' if False else 'rx'] for i in up_ifaces)
        sum_tx_after = sum(post_router[i]['pre_tx' if False else 'tx'] for i in up_ifaces)
        if dir_to_scale == 'tx':
            denom_after = max(sum_tx_after, EPS)
            k_final = clamp((sum_rx_after / denom_after), FINAL_SCALE_CLAMP[0], FINAL_SCALE_CLAMP[1])
            if abs(k_final - 1.0) > 1e-9:
                for i in up_ifaces:
                    old = post_router[i]['tx']
                    new = max(0.0, old * k_final)
                    delta_abs = abs(new - old)
                    post_router[i]['tx'] = new
                    router_delta_abs[i]['tx'] += delta_abs
                    router_delta_rel[i]['tx'] += abs(k_final - 1.0)
        else:
            denom_after = max(sum_rx_after, EPS)
            k_final = clamp((sum_tx_after / denom_after), FINAL_SCALE_CLAMP[0], FINAL_SCALE_CLAMP[1])
            if abs(k_final - 1.0) > 1e-9:
                for i in up_ifaces:
                    old = post_router[i]['rx']
                    new = max(0.0, old * k_final)
                    delta_abs = abs(new - old)
                    post_router[i]['rx'] = new
                    router_delta_abs[i]['rx'] += delta_abs
                    router_delta_rel[i]['rx'] += abs(k_final - 1.0)

    # Third pass: targeted asymmetric pair reconciliation with remaining budget
    post: Dict[str, Dict[str, float]] = {i: {'rx': post_router[i]['rx'], 'tx': post_router[i]['tx']} for i in telemetry}
    pair_adj_rel: Dict[str, Dict[str, float]] = {i: {'rx': 0.0, 'tx': 0.0} for i in telemetry}

    visited_pairs = set()
    for iface, data in telemetry.items():
        peer_id = peers.get(iface)
        if not (peer_id and peer_id in telemetry):
            continue
        key = tuple(sorted((iface, peer_id)))
        if key in visited_pairs:
            continue
        visited_pairs.add(key)

        if pre[iface]['local_status'] != 'up' or pre[peer_id]['local_status'] != 'up':
            continue

        # Reconcile only if router edits touched either iface/dir or residual > tolerance
        touched = any([
            router_delta_abs.get(iface, {}).get('rx', 0.0) > 0.0,
            router_delta_abs.get(iface, {}).get('tx', 0.0) > 0.0,
            router_delta_abs.get(peer_id, {}).get('rx', 0.0) > 0.0,
            router_delta_abs.get(peer_id, {}).get('tx', 0.0) > 0.0,
        ])

        a_rx, a_tx = post[iface]['rx'], post[iface]['tx']
        b_rx, b_tx = post[peer_id]['rx'], post[peer_id]['tx']

        # A.tx <-> B.rx
        resid_tx = rel_diff(a_tx, b_rx)
        tol_tx = pair_tolerance(a_tx, b_rx)
        if touched or (resid_tx > tol_tx):
            conf_a_tx = max(0.0, 1.0 - pre[iface]['tx_link_diff'])
            conf_b_rx = max(0.0, 1.0 - pre[peer_id]['rx_link_diff'])
            low_moves_a = conf_a_tx <= conf_b_rx
            alpha_a = ALPHA_LOW if low_moves_a else ALPHA_HIGH
            alpha_b = ALPHA_HIGH if low_moves_a else ALPHA_LOW
            move_a = alpha_a * (b_rx - a_tx)
            move_b = alpha_b * (a_tx - b_rx)
            # Remaining caps based on pre-router values minus router edits
            base_a = pre[iface]['pre_tx']
            base_b = pre[peer_id]['pre_rx']
            cap_a = max(0.0, PAIR_CAP * max(base_a, 1.0) - router_delta_abs.get(iface, {}).get('tx', 0.0))
            cap_b = max(0.0, PAIR_CAP * max(base_b, 1.0) - router_delta_abs.get(peer_id, {}).get('rx', 0.0))
            move_a = clamp(move_a, -cap_a, cap_a)
            move_b = clamp(move_b, -cap_b, cap_b)
            new_a_tx = max(0.0, a_tx + move_a)
            new_b_rx = max(0.0, b_rx + move_b)
            pair_adj_rel[iface]['tx'] = max(pair_adj_rel[iface]['tx'], abs(new_a_tx - a_tx) / max(a_tx, 1.0))
            pair_adj_rel[peer_id]['rx'] = max(pair_adj_rel[peer_id]['rx'], abs(new_b_rx - b_rx) / max(b_rx, 1.0))
            post[iface]['tx'] = new_a_tx
            post[peer_id]['rx'] = new_b_rx

        # Refresh after potential edit
        a_rx, a_tx = post[iface]['rx'], post[iface]['tx']
        b_rx, b_tx = post[peer_id]['rx'], post[peer_id]['tx']

        # A.rx <-> B.tx
        resid_rx = rel_diff(a_rx, b_tx)
        tol_rx = pair_tolerance(a_rx, b_tx)
        if touched or (resid_rx > tol_rx):
            conf_a_rx = max(0.0, 1.0 - pre[iface]['rx_link_diff'])
            conf_b_tx = max(0.0, 1.0 - pre[peer_id]['tx_link_diff'])
            low_moves_a = conf_a_rx <= conf_b_tx
            alpha_a = ALPHA_LOW if low_moves_a else ALPHA_HIGH
            alpha_b = ALPHA_HIGH if low_moves_a else ALPHA_LOW
            move_a = alpha_a * (b_tx - a_rx)
            move_b = alpha_b * (a_rx - b_tx)
            base_a = pre[iface]['pre_rx']
            base_b = pre[peer_id]['pre_tx']
            cap_a = max(0.0, PAIR_CAP * max(base_a, 1.0) - router_delta_abs.get(iface, {}).get('rx', 0.0))
            cap_b = max(0.0, PAIR_CAP * max(base_b, 1.0) - router_delta_abs.get(peer_id, {}).get('tx', 0.0))
            move_a = clamp(move_a, -cap_a, cap_a)
            move_b = clamp(move_b, -cap_b, cap_b)
            new_a_rx = max(0.0, a_rx + move_a)
            new_b_tx = max(0.0, b_tx + move_b)
            pair_adj_rel[iface]['rx'] = max(pair_adj_rel[iface]['rx'], abs(new_a_rx - a_rx) / max(a_rx, 1.0))
            pair_adj_rel[peer_id]['tx'] = max(pair_adj_rel[peer_id]['tx'], abs(new_b_tx - b_tx) / max(b_tx, 1.0))
            post[iface]['rx'] = new_a_rx
            post[peer_id]['tx'] = new_b_tx

    # Compute post-repair router imbalance for confidence
    router_imbalance_post: Dict[str, float] = {}
    for rid, if_list in topology.items():
        up_ifaces = [i for i in if_list if i in telemetry and pre[i]['local_status'] == 'up']
        if not up_ifaces:
            router_imbalance_post[rid] = 0.0
            continue
        sum_rx = sum(post[i]['rx'] for i in up_ifaces)
        sum_tx = sum(post[i]['tx'] for i in up_ifaces)
        denom = max(sum_rx, sum_tx, 1.0)
        router_imbalance_post[rid] = abs(sum_tx - sum_rx) / denom

    # Assemble final results with calibrated confidence
    result: Dict[str, Dict[str, Tuple]] = {}
    for iface, data in telemetry.items():
        local_status = pre[iface]['local_status']
        peer_status = pre[iface]['peer_status']
        had_peer = pre[iface]['had_peer']
        rx_orig = nz_float(data.get('rx_rate', 0.0))
        tx_orig = nz_float(data.get('tx_rate', 0.0))

        rx_repaired = post[iface]['rx']
        tx_repaired = post[iface]['tx']

        # Ensure down interfaces have zero repaired rates
        repaired_status = data.get('interface_status', 'unknown')
        if norm_status(repaired_status) == 'down':
            rx_repaired = 0.0
            tx_repaired = 0.0

        # Base link confidence from final residuals
        peer_id = peers.get(iface)
        if had_peer and peer_id in telemetry and local_status == 'up' and peer_status == 'up':
            peer_tx_after = post[peer_id]['tx']
            peer_rx_after = post[peer_id]['rx']
            rx_resid = rel_diff(rx_repaired, peer_tx_after)
            tx_resid = rel_diff(tx_repaired, peer_rx_after)
            rx_link_conf = max(0.0, 1.0 - rx_resid)
            tx_link_conf = max(0.0, 1.0 - tx_resid)
        elif norm_status(repaired_status) == 'down':
            rx_link_conf = 0.9 if rx_repaired == 0.0 else 0.5
            tx_link_conf = 0.9 if tx_repaired == 0.0 else 0.5
        else:
            rx_link_conf = 0.6
            tx_link_conf = 0.6

        # Router factor from post-repair imbalance
        router_id = data.get('local_router')
        imb_post = router_imbalance_post.get(router_id, 0.0)
        router_factor = max(0.2, 1.0 - imb_post)

        # Change penalty (two-slope taper)
        def change_factor(orig: float, rep: float) -> float:
            ch = rel_diff(orig, rep)
            weight = 0.4 if ch < 0.15 else 0.5
            return max(0.2, 1.0 - weight * min(1.0, ch))

        rx_change_factor = change_factor(rx_orig, rx_repaired)
        tx_change_factor = change_factor(tx_orig, tx_repaired)

        # Base confidence
        rx_confidence = rx_link_conf * router_factor * rx_change_factor
        tx_confidence = tx_link_conf * router_factor * tx_change_factor

        # Scale intensity penalty using bounded router ratio
        s_b = s_bounded_map.get(router_id, 1.0)
        if abs(1.0 - s_b) > 0.25:
            pen = min(0.05, 0.2 * abs(1.0 - s_b))
            rx_confidence -= pen
            tx_confidence -= pen

        # Penalties for heavy router/pair edits (cap intensity)
        rdel_rx = router_delta_abs.get(iface, {}).get('rx', 0.0) / max(pre[iface]['pre_rx'], 1.0)
        rdel_tx = router_delta_abs.get(iface, {}).get('tx', 0.0) / max(pre[iface]['pre_tx'], 1.0)
        padj_rx = pair_adj_rel.get(iface, {}).get('rx', 0.0)
        padj_tx = pair_adj_rel.get(iface, {}).get('tx', 0.0)
        cum_rx_rel = rdel_rx + padj_rx
        cum_tx_rel = rdel_tx + padj_tx
        cum_cap = PER_IFACE_CAP + PAIR_CAP  # 0.35
        thresh = 0.7 * cum_cap
        if cum_rx_rel > thresh:
            rx_confidence -= min(0.08, 0.76 * (cum_rx_rel - thresh))
        if cum_tx_rel > thresh:
            tx_confidence -= min(0.08, 0.76 * (cum_tx_rel - thresh))

        # Small penalty proportional to pair adjustment magnitude
        rx_confidence -= min(0.05, 0.25 * padj_rx)
        tx_confidence -= min(0.05, 0.25 * padj_tx)

        # Bonus for untouched directions
        rx_untouched = (not pre[iface]['pre_rx_changed']) and (router_delta_abs.get(iface, {}).get('rx', 0.0) == 0.0) and (padj_rx == 0.0)
        tx_untouched = (not pre[iface]['pre_tx_changed']) and (router_delta_abs.get(iface, {}).get('tx', 0.0) == 0.0) and (padj_tx == 0.0)
        if rx_untouched:
            rx_confidence += 0.03
        if tx_untouched:
            tx_confidence += 0.03

        rx_confidence = max(0.0, min(1.0, rx_confidence))
        tx_confidence = max(0.0, min(1.0, tx_confidence))

        # Status confidence
        status_confidence = 1.0
        if peer_id and peer_id in telemetry:
            peer_status_raw = norm_status(telemetry[peer_id].get('interface_status', 'unknown'))
            if norm_status(repaired_status) != peer_status_raw:
                status_confidence = min(status_confidence, 0.5)
        if norm_status(repaired_status) == 'down' and (rx_orig > 0.0 or tx_orig > 0.0):
            status_confidence = min(status_confidence, 0.6)

        # Build output
        repaired_entry: Dict[str, Tuple] = {}
        repaired_entry['rx_rate'] = (rx_orig, rx_repaired, rx_confidence)
        repaired_entry['tx_rate'] = (tx_orig, tx_repaired, tx_confidence)
        repaired_entry['interface_status'] = (data.get('interface_status', 'unknown'), repaired_status, status_confidence)
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