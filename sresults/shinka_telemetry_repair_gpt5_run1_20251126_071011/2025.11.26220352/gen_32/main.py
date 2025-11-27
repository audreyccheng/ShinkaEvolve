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
    # ------------------------------
    # Parameters (tunable)
    # ------------------------------
    HARDENING_THRESHOLD = 0.02                # τh timing tolerance for link symmetry
    EPS = 1e-9
    MULT_PRE_CAP_REL = 0.15                  # per-interface multiplicative cap (±15%)
    ADD_PASS_CAPS = [0.25, 0.35]             # staged additive per-interface caps (relative)
    ROUTER_TOTAL_DELTA_GUARD = 0.25          # router's total absolute delta guard fraction of avg traffic
    SMALL_LINK_Mbps = 1.0                    # threshold to protect tiny links in weighting
    WEIGHT_TOL_PAIR = 0.02                   # pair residual tolerance for weight normalization
    POST_PAIR_PULL_FRAC = 0.25               # fraction to pull during post-pair reconcile
    POST_PAIR_CAP_REL = 0.20                 # cap for post-reconcile per-interface relative move
    CONF_NO_EDIT_BONUS = 0.05                # confidence bonus when no meaningful edit applied

    # ------------------------------
    # Helpers
    # ------------------------------
    def clamp(x: float, lo: float, hi: float) -> float:
        return max(lo, min(hi, x))

    def norm_status(s: Any) -> str:
        s = str(s).lower()
        return s if s in ("up", "down") else "up"  # treat unknown as up, reflect uncertainty in confidence

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

    # Build peer mapping
    peers: Dict[str, str] = {iface: data.get('connected_to') for iface, data in telemetry.items()}

    # Build router -> interfaces mapping using topology plus any missing from telemetry.local_router
    router_ifaces: Dict[str, List[str]] = {r: list(if_list) for r, if_list in topology.items()}
    for iface, data in telemetry.items():
        r = data.get('local_router')
        if r is not None:
            if r not in router_ifaces:
                router_ifaces[r] = []
            if iface not in router_ifaces[r]:
                router_ifaces[r].append(iface)

    # ------------------------------
    # Pass 1: Link hardening (status-aware, peer-aided)
    # ------------------------------
    pre: Dict[str, Dict[str, Any]] = {}
    for iface, data in telemetry.items():
        local_status = norm_status(data.get('interface_status', 'unknown'))
        rx0 = nz_float(data.get('rx_rate', 0.0))
        tx0 = nz_float(data.get('tx_rate', 0.0))
        peer_id = peers.get(iface)
        had_peer = bool(peer_id and peer_id in telemetry)

        peer_status = 'unknown'
        peer_rx0 = peer_tx0 = 0.0
        if had_peer:
            p = telemetry[peer_id]
            peer_status = norm_status(p.get('interface_status', 'unknown'))
            peer_rx0 = nz_float(p.get('rx_rate', 0.0))
            peer_tx0 = nz_float(p.get('tx_rate', 0.0))

        # Defaults
        pre_rx = rx0
        pre_tx = tx0
        # Link residuals (relative)
        rx_link_diff = 0.0
        tx_link_diff = 0.0

        # Local-only zeroing to avoid over-zeroing when peer status may be wrong
        if local_status == 'down':
            pre_rx = 0.0
            pre_tx = 0.0
        else:
            # Use peer redundancy if available and peer up
            if had_peer and peer_status == 'up':
                rx_link_diff = rel_diff(rx0, peer_tx0)  # A.rx vs B.tx
                tx_link_diff = rel_diff(tx0, peer_rx0)  # A.tx vs B.rx

                if rx_link_diff > HARDENING_THRESHOLD:
                    pre_rx = peer_tx0  # direct substitution improves single-sided faults
                if tx_link_diff > HARDENING_THRESHOLD:
                    pre_tx = peer_rx0

        # Directional pre-confidence estimate from link residuals
        pre_conf_rx = 0.9 if local_status == 'down' else (max(0.0, 1.0 - rx_link_diff) if had_peer else 0.6)
        pre_conf_tx = 0.9 if local_status == 'down' else (max(0.0, 1.0 - tx_link_diff) if had_peer else 0.6)

        pre[iface] = {
            'pre_rx': pre_rx,
            'pre_tx': pre_tx,
            'rx_link_diff': rx_link_diff,
            'tx_link_diff': tx_link_diff,
            'local_status': local_status,
            'peer_status': peer_status if had_peer else 'unknown',
            'had_peer': had_peer,
            'pre_conf_rx': pre_conf_rx,
            'pre_conf_tx': pre_conf_tx,
        }

    # Working values after pass 1
    cur_rx: Dict[str, float] = {i: pre[i]['pre_rx'] for i in telemetry}
    cur_tx: Dict[str, float] = {i: pre[i]['pre_tx'] for i in telemetry}

    # Track per-interface edit budgets and penalties
    iface_mult_cap_hit = {i: False for i in telemetry}
    iface_add_consumed_rel = {i: 0.0 for i in telemetry}
    iface_any_edit = {i: False for i in telemetry}
    iface_post_recon_pen = {i: 1.0 for i in telemetry}  # multiplicative confidence penalty factor (<=1)

    # ------------------------------
    # Pass 2: Router-level conservation with imbalance-tuned multiplicative pre-step
    # ------------------------------
    router_imbalance: Dict[str, float] = {}

    for router_id, if_list_all in router_ifaces.items():
        # Consider only interfaces present and locally up (down were zeroed)
        if_list = [i for i in if_list_all if i in telemetry]
        up_ifaces = [i for i in if_list if pre[i]['local_status'] == 'up']
        if not up_ifaces:
            router_imbalance[router_id] = 0.0
            continue

        sum_rx = sum(cur_rx[i] for i in up_ifaces)
        sum_tx = sum(cur_tx[i] for i in up_ifaces)
        denom = max(sum_rx, sum_tx, 1.0)
        imb = abs(sum_tx - sum_rx) / denom
        router_imbalance[router_id] = imb

        # Skip tiny traffic routers or trivial imbalance
        if (sum_rx + sum_tx) <= 1e-6 or imb <= HARDENING_THRESHOLD or len(up_ifaces) < 2:
            continue

        # Decide which direction to scale toward balance
        dir_big = 'tx' if sum_tx > sum_rx else 'rx'
        sum_this = sum_tx if dir_big == 'tx' else sum_rx
        sum_other = sum_rx if dir_big == 'tx' else sum_tx

        s_bounded = clamp(sum_other / max(sum_this, EPS), 0.5, 2.0)
        alpha = clamp(imb / 0.15, 0.25, 0.6)  # imbalance-tuned aggressiveness

        target_scale = 1.0 + alpha * (s_bounded - 1.0)

        # Apply bounded multiplicative scaling with per-interface cap
        for i in up_ifaces:
            v = cur_tx[i] if dir_big == 'tx' else cur_rx[i]
            new_v = v * target_scale
            delta = new_v - v
            cap_rel = MULT_PRE_CAP_REL
            cap_abs = MULT_PRE_CAP_REL * max(v, 1.0)
            # enforce caps
            if abs(delta) > cap_rel * v or abs(delta) > cap_abs:
                iface_mult_cap_hit[i] = True
                # clamp delta
                max_delta = min(cap_rel * v, cap_abs)
                delta = clamp(delta, -max_delta, max_delta)
            if delta != 0.0:
                iface_any_edit[i] = True
            if dir_big == 'tx':
                cur_tx[i] = max(0.0, v + delta)
            else:
                cur_rx[i] = max(0.0, v + delta)

    # ------------------------------
    # Pass 3: Confidence-weighted additive redistribution with staged caps and router guard
    # ------------------------------
    def compute_pair_residual(i: str, direction: str) -> float:
        # direction 'tx': compare i.tx vs peer.rx; 'rx': compare i.rx vs peer.tx
        p = peers.get(i)
        if not (p and p in telemetry):
            return 0.0
        if pre[i]['local_status'] != 'up' or pre.get(p, {}).get('local_status') != 'up':
            return 0.0
        if direction == 'tx':
            return rel_diff(cur_tx[i], cur_rx[p])
        else:
            return rel_diff(cur_rx[i], cur_tx[p])

    for router_id, if_list_all in router_ifaces.items():
        if_list = [i for i in if_list_all if i in telemetry]
        up_ifaces = [i for i in if_list if pre[i]['local_status'] == 'up']
        if len(up_ifaces) < 2:
            continue

        # Recompute sums after multiplicative step
        def sums():
            return sum(cur_rx[i] for i in up_ifaces), sum(cur_tx[i] for i in up_ifaces)

        sum_rx, sum_tx = sums()
        if (sum_rx + sum_tx) <= 1e-6:
            continue

        # Router-level delta guard
        avg_tr = 0.5 * (sum_rx + sum_tx)
        router_abs_guard = ROUTER_TOTAL_DELTA_GUARD * avg_tr

        for pass_idx, cap_rel_stage in enumerate(ADD_PASS_CAPS):
            # Determine which direction to adjust (larger aggregate)
            sum_rx, sum_tx = sums()
            if abs(sum_tx - sum_rx) / max(sum_tx, sum_rx, 1.0) <= HARDENING_THRESHOLD:
                break  # good enough
            if sum_tx >= sum_rx:
                dir_scale = 'tx'
                sum_this, sum_other = sum_tx, sum_rx
            else:
                dir_scale = 'rx'
                sum_this, sum_other = sum_rx, sum_tx

            need_total = sum_other - sum_this  # negative means we must reduce this direction
            # Bound by router guard
            need_sign = 1.0 if need_total >= 0 else -1.0
            max_allowed_change = min(abs(need_total), router_abs_guard)
            need_total = need_sign * max_allowed_change
            if abs(need_total) <= 1e-12:
                break

            # Prepare weights
            vals = [cur_tx[i] if dir_scale == 'tx' else cur_rx[i] for i in up_ifaces]
            sum_v = max(sum(vals), EPS)

            weights: Dict[str, float] = {}
            for i in up_ifaces:
                conf_dir = pre[i]['pre_conf_tx'] if dir_scale == 'tx' else pre[i]['pre_conf_rx']
                resid = compute_pair_residual(i, dir_scale)
                w_base = 0.6 * (1.0 - conf_dir) + 0.25 * (resid / max(WEIGHT_TOL_PAIR, EPS)) + 0.15 * ((cur_tx[i] if dir_scale == 'tx' else cur_rx[i]) / sum_v)
                # Protect very small links
                if (cur_tx[i] if dir_scale == 'tx' else cur_rx[i]) < SMALL_LINK_Mbps:
                    w_base *= 0.5
                # Reduce if already used heavy cap in previous stages
                if iface_add_consumed_rel[i] > 0.7:
                    w_base *= 0.7
                # Avoid zero weights
                weights[i] = max(1e-6, w_base)

            # Normalize
            sum_w = sum(weights.values())
            if sum_w <= 0:
                weights = {i: 1.0 for i in up_ifaces}
                sum_w = float(len(up_ifaces))

            # Distribute need_total respecting per-interface caps
            remaining = need_total
            # Single pass proportional distribution with clamping
            for i in up_ifaces:
                share = (weights[i] / sum_w) * need_total
                v = cur_tx[i] if dir_scale == 'tx' else cur_rx[i]
                cap_amount = cap_rel_stage * v
                # remaining per-interface additive budget (subtract what has already been used)
                remaining_cap = max(0.0, cap_amount - iface_add_consumed_rel[i] * v)
                # Clamp share to remaining cap
                delta_i = share
                if abs(delta_i) > remaining_cap:
                    delta_i = clamp(delta_i, -remaining_cap, remaining_cap)
                if delta_i != 0.0:
                    iface_any_edit[i] = True
                if dir_scale == 'tx':
                    cur_tx[i] = max(0.0, v + delta_i)
                else:
                    cur_rx[i] = max(0.0, v + delta_i)
                # Update consumed relative cap
                iface_add_consumed_rel[i] = min(1.0, iface_add_consumed_rel[i] + (abs(delta_i) / max(v, EPS)))
                remaining -= delta_i

            # If significant remaining, a second internal sweep could be added, but we keep one pass per stage for stability.
            if abs(remaining) <= max(1e-6, 0.005 * avg_tr):
                # Good enough
                pass

    # ------------------------------
    # Pass 4: Targeted post-redistribution pair reconcile
    # ------------------------------
    visited_pairs = set()
    for i, data in telemetry.items():
        p = peers.get(i)
        if not (p and p in telemetry):
            continue
        key = tuple(sorted([i, p]))
        if key in visited_pairs:
            continue
        visited_pairs.add(key)

        # Reconcile only when both sides locally up
        if pre[i]['local_status'] != 'up' or pre[p]['local_status'] != 'up':
            continue

        # Only reconcile pairs where any endpoint was edited
        if not (iface_any_edit[i] or iface_any_edit[p]):
            continue

        # Residuals after router edits
        resid_tx = rel_diff(cur_tx[i], cur_rx[p])  # i.tx vs p.rx
        resid_rx = rel_diff(cur_rx[i], cur_tx[p])  # i.rx vs p.tx

        # Threshold scales with traffic to avoid overreacting on very small links
        traffic_scale_tx = max(cur_tx[i], cur_rx[p], 1.0)
        traffic_scale_rx = max(cur_rx[i], cur_tx[p], 1.0)
        tol_pair_post_tx = max(HARDENING_THRESHOLD, 2.5 / traffic_scale_tx)
        tol_pair_post_rx = max(HARDENING_THRESHOLD, 2.5 / traffic_scale_rx)

        # Helper to pull pair toward agreement
        def pull_toward(a_val: float, b_val: float, cap_a: float, cap_b: float):
            target = 0.5 * (a_val + b_val)
            move_a = clamp(target - a_val, -cap_a, cap_a)
            move_b = clamp(target - b_val, -cap_b, cap_b)
            return a_val + move_a, b_val + move_b, abs(move_a) + abs(move_b)

        total_move_penalty = 0.0

        # Reconcile tx direction
        if resid_tx > tol_pair_post_tx:
            cap_i = POST_PAIR_CAP_REL * max(cur_tx[i], 1.0)
            cap_p = POST_PAIR_CAP_REL * max(cur_rx[p], 1.0)
            new_i_tx, new_p_rx, move = pull_toward(cur_tx[i], cur_rx[p], cap_i, cap_p)
            cur_tx[i], cur_rx[p] = max(0.0, new_i_tx), max(0.0, new_p_rx)
            total_move_penalty += move
            iface_any_edit[i] = True
            iface_any_edit[p] = True

        # Reconcile rx direction
        if resid_rx > tol_pair_post_rx:
            cap_i = POST_PAIR_CAP_REL * max(cur_rx[i], 1.0)
            cap_p = POST_PAIR_CAP_REL * max(cur_tx[p], 1.0)
            new_i_rx, new_p_tx, move = pull_toward(cur_rx[i], cur_tx[p], cap_i, cap_p)
            cur_rx[i], cur_tx[p] = max(0.0, new_i_rx), max(0.0, new_p_tx)
            total_move_penalty += move
            iface_any_edit[i] = True
            iface_any_edit[p] = True

        # Confidence penalty factor for reconciliation magnitude
        if total_move_penalty > 0.0:
            # Penalize proportional to how large the residual was compared to tolerance
            x_tx = resid_tx / max(WEIGHT_TOL_PAIR, EPS)
            x_rx = resid_rx / max(WEIGHT_TOL_PAIR, EPS)
            penal = clamp(1.0 - 0.3 * max(x_tx, x_rx), 0.0, 1.0)
            iface_post_recon_pen[i] *= penal
            iface_post_recon_pen[p] *= penal

    # ------------------------------
    # Assemble final results with refined confidence calibration
    # ------------------------------
    result: Dict[str, Dict[str, Tuple]] = {}

    # Compute final router imbalance after all edits for confidence scaling
    final_router_imbalance: Dict[str, float] = {}
    for router_id, if_list_all in router_ifaces.items():
        if_list = [i for i in if_list_all if i in telemetry and pre[i]['local_status'] == 'up']
        if not if_list:
            final_router_imbalance[router_id] = 0.0
            continue
        sum_rx = sum(cur_rx[i] for i in if_list)
        sum_tx = sum(cur_tx[i] for i in if_list)
        final_router_imbalance[router_id] = abs(sum_tx - sum_rx) / max(sum_tx, sum_rx, 1.0)

    for iface, data in telemetry.items():
        local_status = pre[iface]['local_status']
        peer_status = pre[iface]['peer_status']
        had_peer = pre[iface]['had_peer']
        rx_orig = nz_float(data.get('rx_rate', 0.0))
        tx_orig = nz_float(data.get('tx_rate', 0.0))

        rx_repaired = cur_rx[iface]
        tx_repaired = cur_tx[iface]

        # Enforce zero rates if status down
        repaired_status = data.get('interface_status', 'unknown')
        if norm_status(repaired_status) == 'down':
            rx_repaired = 0.0
            tx_repaired = 0.0

        # Peer-based residuals after all edits
        peer_id = peers.get(iface)
        if had_peer and peer_id in telemetry and pre[peer_id]['local_status'] == 'up' and local_status == 'up':
            rx_resid = rel_diff(rx_repaired, cur_tx[peer_id])
            tx_resid = rel_diff(tx_repaired, cur_rx[peer_id])
        else:
            rx_resid = 0.4  # uncertainty placeholder when no reliable peer
            tx_resid = 0.4

        # Two-slope residual mapping for peer agreement
        def conf_from_residual(resid: float, tol: float) -> float:
            x = resid / max(tol, EPS)
            conf = 1.0 - min(1.0, x / 5.0)
            if x > 3.0:
                conf -= 0.1 * ((x - 3.0) / 2.0)
            return clamp(conf, 0.0, 1.0)

        rx_peer_conf = conf_from_residual(rx_resid, HARDENING_THRESHOLD)
        tx_peer_conf = conf_from_residual(tx_resid, HARDENING_THRESHOLD)

        # Router imbalance factor (post-edits)
        router_id = data.get('local_router')
        router_factor = clamp(1.0 - final_router_imbalance.get(router_id, 0.0), 0.2, 1.0)

        # Change penalty based on relative change from original
        rx_change = rel_diff(rx_orig, rx_repaired)
        tx_change = rel_diff(tx_orig, tx_repaired)
        rx_change_factor = max(0.2, 1.0 - 0.5 * min(1.0, rx_change))
        tx_change_factor = max(0.2, 1.0 - 0.5 * min(1.0, tx_change))

        # Scale penalties for cap hits and large multiplicative scaling
        cap_penalty_rx = 1.0
        cap_penalty_tx = 1.0
        if iface_mult_cap_hit[iface]:
            cap_penalty_rx *= 0.9
            cap_penalty_tx *= 0.9
        if iface_add_consumed_rel[iface] > 0.8:
            cap_penalty_rx *= 0.9
            cap_penalty_tx *= 0.9

        # No-edit bonus
        no_edit_bonus_rx = CONF_NO_EDIT_BONUS if rx_change <= 0.01 else 0.0
        no_edit_bonus_tx = CONF_NO_EDIT_BONUS if tx_change <= 0.01 else 0.0

        rx_confidence = clamp(rx_peer_conf * router_factor * rx_change_factor * iface_post_recon_pen[iface] * cap_penalty_rx + no_edit_bonus_rx, 0.0, 1.0)
        tx_confidence = clamp(tx_peer_conf * router_factor * tx_change_factor * iface_post_recon_pen[iface] * cap_penalty_tx + no_edit_bonus_tx, 0.0, 1.0)

        # Status confidence: keep status unchanged, calibrate confidence using peer status and counter sanity
        status_confidence = 1.0
        if peer_id and peer_id in telemetry:
            peer_status_raw = norm_status(telemetry[peer_id].get('interface_status', 'unknown'))
            if norm_status(repaired_status) != peer_status_raw:
                status_confidence = min(status_confidence, 0.5)
        if norm_status(repaired_status) == 'down' and (rx_orig > 0.0 or tx_orig > 0.0):
            status_confidence = min(status_confidence, 0.6)

        # Ensure down interfaces have zero repaired rates
        if norm_status(repaired_status) == 'down':
            rx_repaired = 0.0
            tx_repaired = 0.0

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