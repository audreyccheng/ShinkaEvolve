# EVOLVE-BLOCK-START
"""
Network telemetry repair algorithm that detects and corrects inconsistencies
in network interface telemetry data using topology relationships.

Takes interface telemetry data and detects/repairs inconsistencies based on
network invariants like link symmetry and flow conservation.
"""
from typing import Dict, Any, Tuple, List
from math import sqrt

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
    # Small traffic level used to infer link up when statuses disagree (Mbps)
    TRAFFIC_EVIDENCE_MIN = 0.5

    # Multiplicative pre-normalization (imbalance-tuned)
    MULT_PRE_CAP_FRAC = 0.15  # per-interface cap for pre-step
    TIE_EPS_CONF = 0.05

    # Additive redistribution staged caps (per-interface), and router safety guard
    CAP_PASSES = [0.25, 0.35, 0.45]  # pass1, pass2, pass3 (pass3 conditional)
    ROUTER_TOTAL_DELTA_FRAC = 0.25   # ≤25% of avg(router TX,RX) absolute delta guard
    MAX_ROUTER_ADJ_FRAC = 0.35       # legacy per-interface cap when not staged (kept for fallbacks)

    # Pair reconciliation (only for links touched by router edits)
    PAIR_RECONCILE_ALPHA_LOW = 0.35   # lower-confidence side
    PAIR_RECONCILE_ALPHA_HIGH = 0.20  # higher-confidence side
    PAIR_RECONCILE_CAP_FRAC = 0.20    # per-interface cap in reconciliation

    # Confidence blending weights
    W_PAIR, W_ROUTER, W_STATUS = 0.6, 0.3, 0.1

    EPS = 1e-9

    def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
        return max(lo, min(hi, x))

    def rel_diff(a: float, b: float) -> float:
        denom = max(abs(a), abs(b), 1e-9)
        return abs(a - b) / denom

    def conf_from_residual_two_slope(residual: float, tol: float) -> float:
        # Two-slope residual-to-confidence as used in strong prior generations
        tol = max(tol, 1e-9)
        x = residual / tol
        conf = 1.0 - min(1.0, x / 5.0)
        if x > 3.0:
            conf -= 0.1 * (x - 3.0) / 2.0
        return clamp(conf)

    # Build connected pairs (bidirectional mapping) and initial containers
    visited = set()
    pairs: List[Tuple[str, str]] = []
    for if_id, data in telemetry.items():
        peer = data.get('connected_to')
        if peer and peer in telemetry:
            key = tuple(sorted([if_id, peer]))
            if key not in visited:
                visited.add(key)
                pairs.append((key[0], key[1]))

    peer_of: Dict[str, str] = {}
    paired_ids = set()
    for a_id, b_id in pairs:
        peer_of[a_id] = b_id
        peer_of[b_id] = a_id
        paired_ids.add(a_id)
        paired_ids.add(b_id)

    # Initialize interim store
    interim: Dict[str, Dict[str, Any]] = {}
    for if_id, data in telemetry.items():
        interim[if_id] = {
            'rx': float(data.get('rx_rate', 0.0)),
            'tx': float(data.get('tx_rate', 0.0)),
            'rx_conf': 1.0,
            'tx_conf': 1.0,
            'status': data.get('interface_status', 'unknown'),
            'status_conf': 1.0,
            'connected_to': data.get('connected_to'),
            'local_router': data.get('local_router'),
            'remote_router': data.get('remote_router'),
            # Originals
            'orig_rx': float(data.get('rx_rate', 0.0)),
            'orig_tx': float(data.get('tx_rate', 0.0)),
            'orig_status': data.get('interface_status', 'unknown'),
            # Edit tracking
            'edit_rx_rel': 0.0,
            'edit_tx_rel': 0.0,
            'cap_used_rx': 0.0,
            'cap_used_tx': 0.0,
            'cap_total_rx': 0.0,
            'cap_total_tx': 0.0,
            'cap_hit_rx': False,
            'cap_hit_tx': False,
            'touched': False,
            'touched_by_router': False,
        }

    # Helper: robust blend for pair hardening (geometric + trimmed mean)
    def robust_pair_merge(x: float, y: float) -> float:
        if x < 0: x = 0.0
        if y < 0: y = 0.0
        mid = 0.5 * (x + y)
        arr = sorted([x, y, mid])
        trimmed = arr[1]
        if x > 0 and y > 0:
            g = sqrt(max(EPS, x) * max(EPS, y))
            # Blend geometric and trimmed mean (geometric dampens outliers)
            return 0.6 * g + 0.4 * trimmed
        return trimmed

    # Pair-level hardening: status consistency + link symmetry projection
    for a_id, b_id in pairs:
        a = interim[a_id]
        b = interim[b_id]
        a_stat = a['status']
        b_stat = b['status']
        a_rx0, a_tx0 = a['rx'], a['tx']
        b_rx0, b_tx0 = b['rx'], b['tx']
        max_traffic = max(a_rx0, a_tx0, b_rx0, b_tx0)

        # Resolve interface status using redundant signals
        if a_stat == b_stat:
            resolved_status = a_stat
            status_conf = 0.95 if resolved_status in ('up', 'down') else 0.7
        else:
            if max_traffic > TRAFFIC_EVIDENCE_MIN:
                resolved_status = 'up'
                status_conf = 0.85
            else:
                resolved_status = 'down'
                status_conf = 0.75

        a['status'] = resolved_status
        b['status'] = resolved_status
        a['status_conf'] = min(a['status_conf'], status_conf)
        b['status_conf'] = min(b['status_conf'], status_conf)

        if resolved_status == 'down':
            for r, rx0, tx0 in [(a, a_rx0, a_tx0), (b, b_rx0, b_tx0)]:
                r['rx'] = 0.0
                r['tx'] = 0.0
                r['rx_conf'] = clamp(0.9 if rx0 <= TRAFFIC_EVIDENCE_MIN else 0.3)
                r['tx_conf'] = clamp(0.9 if tx0 <= TRAFFIC_EVIDENCE_MIN else 0.3)
            continue

        # Link up: enforce symmetry using robust pair merge
        # Forward: a.tx ≈ b.rx
        res_fwd = rel_diff(a_tx0, b_rx0)
        tol = HARDENING_THRESHOLD
        if res_fwd <= tol:
            v = 0.5 * (a_tx0 + b_rx0)
            conf = clamp(1.0 - 0.5 * res_fwd)
        else:
            v = robust_pair_merge(a_tx0, b_rx0)
            conf = clamp(1.0 - min(1.0, res_fwd))
        a['tx'] = v
        b['rx'] = v
        a['tx_conf'] = min(a['tx_conf'], conf)
        b['rx_conf'] = min(b['rx_conf'], conf)

        # Reverse: a.rx ≈ b.tx
        res_rev = rel_diff(a_rx0, b_tx0)
        if res_rev <= tol:
            v2 = 0.5 * (a_rx0 + b_tx0)
            conf2 = clamp(1.0 - 0.5 * res_rev)
        else:
            v2 = robust_pair_merge(a_rx0, b_tx0)
            conf2 = clamp(1.0 - min(1.0, res_rev))
        a['rx'] = v2
        b['tx'] = v2
        a['rx_conf'] = min(a['rx_conf'], conf2)
        b['tx_conf'] = min(b['tx_conf'], conf2)

    # Enforce down => zero on unpaired
    for if_id, r in interim.items():
        if if_id not in paired_ids and r.get('status') == 'down':
            rx0, tx0 = r['rx'], r['tx']
            r['rx'] = 0.0
            r['tx'] = 0.0
            r['rx_conf'] = clamp(0.9 if rx0 <= TRAFFIC_EVIDENCE_MIN else 0.3)
            r['tx_conf'] = clamp(0.9 if tx0 <= TRAFFIC_EVIDENCE_MIN else 0.3)

    # Build router to interfaces map (prefer provided topology; derive if absent)
    if topology:
        router_ifaces: Dict[str, List[str]] = {r: [i for i in if_list if i in interim] for r, if_list in topology.items()}
    else:
        # If topology not provided, derive from telemetry metadata (best-effort)
        router_ifaces = {}
        for if_id, data in interim.items():
            r = data.get('local_router')
            if r is not None:
                router_ifaces.setdefault(r, []).append(if_id)

    # Helper: pair residual for interface in specific direction
    def pair_residual_dir(iface_id: str, direction: str) -> float:
        peer = peer_of.get(iface_id)
        if not peer:
            return 0.0
        if interim[iface_id]['status'] != 'up' or interim[peer]['status'] != 'up':
            return 0.0
        if direction == 'rx':
            return rel_diff(interim[iface_id]['rx'], interim[peer]['tx'])
        else:
            return rel_diff(interim[iface_id]['tx'], interim[peer]['rx'])

    # Router-level rebalancing with multiplicative pre-step and staged additive passes
    # Track router scaling magnitude to inform confidence penalties later
    router_scale_mag: Dict[str, float] = {}

    for router, if_list in router_ifaces.items():
        up_list = [i for i in if_list if interim[i]['status'] == 'up']
        if not up_list:
            router_scale_mag[router] = 0.0
            continue

        sum_tx = sum(max(0.0, interim[i]['tx']) for i in up_list)
        sum_rx = sum(max(0.0, interim[i]['rx']) for i in up_list)
        avg_tx_conf = sum(interim[i]['tx_conf'] for i in up_list) / max(1, len(up_list))
        avg_rx_conf = sum(interim[i]['rx_conf'] for i in up_list) / max(1, len(up_list))

        imbalance = rel_diff(sum_tx, sum_rx)
        if imbalance <= HARDENING_THRESHOLD * 2:
            router_scale_mag[router] = 0.0
        # Decide which direction is less trusted; break ties by scale factor sign
        if abs(avg_tx_conf - avg_rx_conf) < TIE_EPS_CONF:
            # Use ratio to decide: if sum_tx > sum_rx, scale RX up; else scale TX up
            scale_rx = (sum_tx >= sum_rx)
        else:
            scale_rx = (avg_tx_conf >= avg_rx_conf)  # scale less trusted side

        # Multiplicative pre-step
        if sum_tx > 0 and sum_rx > 0 and imbalance > HARDENING_THRESHOLD * 2:
            s = (sum_tx / sum_rx) if scale_rx else (sum_rx / sum_tx)
            s_bounded = clamp(s, 0.5, 2.0)
            # alpha grows with imbalance; clamp between 0.25 and 0.6 at ~15% imbalance
            alpha = clamp(imbalance / 0.15, 0.25, 0.6)
            step_scale = 1.0 + alpha * (s_bounded - 1.0)
            # Apply per-interface capped multiplicative step
            for i in up_list:
                if scale_rx:
                    old_v = max(0.0, interim[i]['rx'])
                    delta = old_v * (step_scale - 1.0)
                    cap = MULT_PRE_CAP_FRAC * max(old_v, 1.0)
                    delta = max(-cap, min(cap, delta))
                    new_v = max(0.0, old_v + delta)
                    if new_v != old_v:
                        interim[i]['rx'] = new_v
                        drel = rel_diff(old_v, new_v)
                        interim[i]['edit_rx_rel'] = max(interim[i]['edit_rx_rel'], drel)
                        interim[i]['touched'] = True
                        interim[i]['touched_by_router'] = True
                        # mild conf decrease due to global scaling magnitude and personal change
                        interim[i]['rx_conf'] = clamp(min(interim[i]['rx_conf'], 1.0 - min(1.0, 0.3 * drel + 0.2 * abs(1.0 - s_bounded)))))
                else:
                    old_v = max(0.0, interim[i]['tx'])
                    delta = old_v * (step_scale - 1.0)
                    cap = MULT_PRE_CAP_FRAC * max(old_v, 1.0)
                    delta = max(-cap, min(cap, delta))
                    new_v = max(0.0, old_v + delta)
                    if new_v != old_v:
                        interim[i]['tx'] = new_v
                        drel = rel_diff(old_v, new_v)
                        interim[i]['edit_tx_rel'] = max(interim[i]['edit_tx_rel'], drel)
                        interim[i]['touched'] = True
                        interim[i]['touched_by_router'] = True
                        interim[i]['tx_conf'] = clamp(min(interim[i]['tx_conf'], 1.0 - min(1.0, 0.3 * drel + 0.2 * abs(1.0 - s_bounded)))))
            router_scale_mag[router] = abs(1.0 - s_bounded)
            # Recompute sums after pre-step
            sum_tx = sum(max(0.0, interim[i]['tx']) for i in up_list)
            sum_rx = sum(max(0.0, interim[i]['rx']) for i in up_list)

        # Additive redistribution if significant need remains
        imbalance = rel_diff(sum_tx, sum_rx)
        if imbalance <= HARDENING_THRESHOLD * 2:
            continue

        # Choose direction again given new sums
        if abs(avg_tx_conf - avg_rx_conf) < TIE_EPS_CONF:
            scale_rx = (sum_tx >= sum_rx)
        else:
            scale_rx = (avg_tx_conf >= avg_rx_conf)

        dir_key = 'rx' if scale_rx else 'tx'
        target_total = sum_tx if scale_rx else sum_rx
        values = {i: max(0.0, interim[i][dir_key]) for i in up_list}
        sum_old = sum(values.values())
        need = target_total - sum_old
        if abs(need) <= max(sum_old, target_total, 1.0) * (HARDENING_THRESHOLD * 0.5):
            continue

        # Router guard: total absolute delta ≤ 25% of avg traffic
        avg_up_traffic = 0.5 * (sum_tx + sum_rx)
        router_delta_guard = ROUTER_TOTAL_DELTA_FRAC * avg_up_traffic
        router_delta_used = 0.0

        # Build weights: uncertainty + pair residual + volume share
        sum_v_for_norm = sum(values.values()) + EPS
        weights: Dict[str, float] = {}
        pair_tols: Dict[str, float] = {}
        for i in up_list:
            conf_i = float(interim[i][f'{dir_key}_conf'])
            v_i = values[i]
            # Pair residual term with rate-aware tolerance
            peer = peer_of.get(i)
            if peer and interim[peer]['status'] == 'up':
                traffic = max(v_i, interim[peer]['tx' if dir_key == 'rx' else 'rx'], 1.0)
                tol_pair = max(HARDENING_THRESHOLD * 1.5, 5.0 / traffic)
                pair_tols[i] = tol_pair
                resid = pair_residual_dir(i, dir_key)
                pair_term = min(2.0, resid / max(tol_pair, 1e-9))
            else:
                pair_tols[i] = HARDENING_THRESHOLD * 1.5
                pair_term = 1.0  # unknown peer -> modest weight
            vol_share = v_i / sum_v_for_norm
            w = 0.6 * (1.0 - conf_i) + 0.25 * pair_term + 0.15 * vol_share
            if v_i < 1.0:
                w *= 0.5
            weights[i] = max(0.02, w)

        # Staged per-interface caps and freeze logic
        pos_rem = {i: CAP_PASSES[0] * max(values[i], 1.0) for i in up_list}
        neg_rem = {i: CAP_PASSES[0] * max(values[i], 1.0) for i in up_list}
        inc_pass2 = {i: (CAP_PASSES[1] - CAP_PASSES[0]) * max(values[i], 1.0) for i in up_list}
        # pass3 only for low-confidence or very small links
        inc_pass3 = {i: ((CAP_PASSES[2] - CAP_PASSES[1]) * max(values[i], 1.0)
                         if (interim[i][f'{dir_key}_conf'] < 0.6 or values[i] < 5.0) else 0.0)
                    for i in up_list}
        # Track how much capacity consumed (for calibration and freezing)
        used_pos = {i: 0.0 for i in up_list}
        used_neg = {i: 0.0 for i in up_list}

        for pass_idx in range(3):
            if abs(need) <= EPS or router_delta_used >= router_delta_guard - EPS:
                break
            # Increase capacities for this pass
            if pass_idx == 1:
                for i in up_list:
                    pos_rem[i] += inc_pass2[i]
                    neg_rem[i] += inc_pass2[i]
            elif pass_idx == 2:
                for i in up_list:
                    pos_rem[i] += inc_pass3[i]
                    neg_rem[i] += inc_pass3[i]

            # Determine eligible set with freeze logic:
            # Prefer interfaces that have not consumed >80% of their total cap,
            # but if none available, include all with any remaining capacity.
            if need > 0:
                cand = [i for i in up_list if pos_rem[i] > EPS]
                pref = [i for i in cand if (used_pos[i] / max(pos_rem[i] + used_pos[i], EPS)) <= 0.8]
                elig = pref if pref else cand
            else:
                cand = [i for i in up_list if neg_rem[i] > EPS]
                pref = [i for i in cand if (used_neg[i] / max(neg_rem[i] + used_neg[i], EPS)) <= 0.8]
                elig = pref if pref else cand

            if not elig:
                continue

            # Reduce weight by 30% on later passes for cap-stressed links
            def eff_weight(i: str) -> float:
                w = weights[i]
                if pass_idx >= 1:
                    # consider if consumed >70% of total cap so far
                    tot_cap = (pos_rem[i] + used_pos[i]) if need > 0 else (neg_rem[i] + used_neg[i])
                    used = used_pos[i] if need > 0 else used_neg[i]
                    if tot_cap > 0 and (used / tot_cap) > 0.7:
                        w *= 0.7
                return w

            sumW = sum(eff_weight(i) for i in elig)
            if sumW <= EPS:
                continue

            for i in elig:
                if abs(need) <= EPS or router_delta_used >= router_delta_guard - EPS:
                    break
                quota = need * (eff_weight(i) / sumW)
                if need > 0:
                    d = min(max(0.0, quota), pos_rem[i], router_delta_guard - router_delta_used)
                    pos_rem[i] -= d
                    used_pos[i] += d
                else:
                    d = -min(max(0.0, -quota), neg_rem[i], router_delta_guard - router_delta_used)
                    neg_rem[i] -= -d
                    used_neg[i] += -d

                if abs(d) <= EPS:
                    continue

                old_v = values[i]
                new_v = max(0.0, old_v + d)
                values[i] = new_v
                router_delta_used += abs(d)
                delta_rel = rel_diff(old_v, new_v)
                # Apply edit to interim and track caps used for calibration
                if dir_key == 'rx':
                    interim[i]['rx'] = new_v
                    interim[i]['edit_rx_rel'] = max(interim[i]['edit_rx_rel'], delta_rel)
                    interim[i]['touched'] = True
                    interim[i]['touched_by_router'] = True
                    # accumulate total cap context for calibration
                    interim[i]['cap_total_rx'] = (pos_rem[i] + used_pos[i]) if need > 0 else (neg_rem[i] + used_neg[i])
                    interim[i]['cap_used_rx'] = max(interim[i]['cap_used_rx'], used_pos[i] if need > 0 else used_neg[i])
                    if (used_pos[i] if need > 0 else used_neg[i]) / max(interim[i]['cap_total_rx'], 1e-9) > 0.99:
                        interim[i]['cap_hit_rx'] = True
                else:
                    interim[i]['tx'] = new_v
                    interim[i]['edit_tx_rel'] = max(interim[i]['edit_tx_rel'], delta_rel)
                    interim[i]['touched'] = True
                    interim[i]['touched_by_router'] = True
                    interim[i]['cap_total_tx'] = (pos_rem[i] + used_pos[i]) if need > 0 else (neg_rem[i] + used_neg[i])
                    interim[i]['cap_used_tx'] = max(interim[i]['cap_used_tx'], used_pos[i] if need > 0 else used_neg[i])
                    if (used_pos[i] if need > 0 else used_neg[i]) / max(interim[i]['cap_total_tx'], 1e-9) > 0.99:
                        interim[i]['cap_hit_tx'] = True

                need -= d

    # Targeted, tolerance-gated pair reconciliation only on links touched by router edits
    for a_id, b_id in pairs:
        a = interim[a_id]; b = interim[b_id]
        if a['status'] != 'up' or b['status'] != 'up':
            continue
        if not (a.get('touched_by_router') or b.get('touched_by_router')):
            continue

        # Forward direction: a.tx vs b.rx
        a_tx_old, b_rx_old = a['tx'], b['rx']
        traffic_tx = max(a_tx_old, b_rx_old, 1.0)
        tol_pair_post = max(0.02, 2.5 / traffic_tx)
        res_fwd = rel_diff(a_tx_old, b_rx_old)
        if res_fwd > tol_pair_post:
            v_mid = 0.5 * (a_tx_old + b_rx_old)
            # Asymmetric pulls based on relative directional confidences
            conf_a = clamp(a['tx_conf'])
            conf_b = clamp(b['rx_conf'])
            alpha_a = PAIR_RECONCILE_ALPHA_LOW if conf_a < conf_b else PAIR_RECONCILE_ALPHA_HIGH
            alpha_b = PAIR_RECONCILE_ALPHA_LOW if conf_b < conf_a else PAIR_RECONCILE_ALPHA_HIGH
            move_a = alpha_a * (v_mid - a_tx_old)
            move_b = alpha_b * (v_mid - b_rx_old)
            cap_a = PAIR_RECONCILE_CAP_FRAC * max(a_tx_old, 1.0)
            cap_b = PAIR_RECONCILE_CAP_FRAC * max(b_rx_old, 1.0)
            move_a = max(-cap_a, min(cap_a, move_a))
            move_b = max(-cap_b, min(cap_b, move_b))
            a_tx_new = max(0.0, a_tx_old + move_a)
            b_rx_new = max(0.0, b_rx_old + move_b)
            if a_tx_new != a_tx_old:
                drel = rel_diff(a_tx_old, a_tx_new)
                a['tx'] = a_tx_new
                a['edit_tx_rel'] = max(a['edit_tx_rel'], drel)
                a['tx_conf'] = clamp(a['tx_conf'] * (1.0 - 0.3 * min(1.0, res_fwd)))
            if b_rx_new != b_rx_old:
                drel = rel_diff(b_rx_old, b_rx_new)
                b['rx'] = b_rx_new
                b['edit_rx_rel'] = max(b['edit_rx_rel'], drel)
                b['rx_conf'] = clamp(b['rx_conf'] * (1.0 - 0.3 * min(1.0, res_fwd)))

        # Reverse direction: a.rx vs b.tx
        a_rx_old, b_tx_old = a['rx'], b['tx']
        traffic_rx = max(a_rx_old, b_tx_old, 1.0)
        tol_pair_post2 = max(0.02, 2.5 / traffic_rx)
        res_rev = rel_diff(a_rx_old, b_tx_old)
        if res_rev > tol_pair_post2:
            v_mid2 = 0.5 * (a_rx_old + b_tx_old)
            conf_a2 = clamp(a['rx_conf'])
            conf_b2 = clamp(b['tx_conf'])
            alpha_a2 = PAIR_RECONCILE_ALPHA_LOW if conf_a2 < conf_b2 else PAIR_RECONCILE_ALPHA_HIGH
            alpha_b2 = PAIR_RECONCILE_ALPHA_LOW if conf_b2 < conf_a2 else PAIR_RECONCILE_ALPHA_HIGH
            move_a2 = alpha_a2 * (v_mid2 - a_rx_old)
            move_b2 = alpha_b2 * (v_mid2 - b_tx_old)
            cap_a2 = PAIR_RECONCILE_CAP_FRAC * max(a_rx_old, 1.0)
            cap_b2 = PAIR_RECONCILE_CAP_FRAC * max(b_tx_old, 1.0)
            move_a2 = max(-cap_a2, min(cap_a2, move_a2))
            move_b2 = max(-cap_b2, min(cap_b2, move_b2))
            a_rx_new = max(0.0, a_rx_old + move_a2)
            b_tx_new = max(0.0, b_tx_old + move_b2)
            if a_rx_new != a_rx_old:
                drel = rel_diff(a_rx_old, a_rx_new)
                a['rx'] = a_rx_new
                a['edit_rx_rel'] = max(a['edit_rx_rel'], drel)
                a['rx_conf'] = clamp(a['rx_conf'] * (1.0 - 0.3 * min(1.0, res_rev)))
            if b_tx_new != b_tx_old:
                drel = rel_diff(b_tx_old, b_tx_new)
                b['tx'] = b_tx_new
                b['edit_tx_rel'] = max(b['edit_tx_rel'], drel)
                b['tx_conf'] = clamp(b['tx_conf'] * (1.0 - 0.3 * min(1.0, res_rev)))

    # Final confidence calibration
    # Compute per-router final imbalance
    router_final_imbalance: Dict[str, float] = {}
    for router, if_list in router_ifaces.items():
        up_ifaces = [i for i in if_list if interim[i]['status'] == 'up']
        if not up_ifaces:
            router_final_imbalance[router] = 0.0
            continue
        sum_tx = sum(max(0.0, interim[i]['tx']) for i in up_ifaces)
        sum_rx = sum(max(0.0, interim[i]['rx']) for i in up_ifaces)
        router_final_imbalance[router] = rel_diff(sum_tx, sum_rx)

    TOL_ROUTER = HARDENING_THRESHOLD * 2.0

    def finalize_conf(base: float,
                      edit_rel: float,
                      small_edit: bool,
                      cap_used: float,
                      cap_total: float,
                      router_scale: float,
                      orig_val: float,
                      new_val: float) -> float:
        # Change penalty: taper weight for small edits
        pen = max(0.0, edit_rel - HARDENING_THRESHOLD)
        weight = 0.4 if small_edit else 0.5
        conf = clamp(base * (1.0 - weight * pen))
        # Router pre-scale penalty if large scale magnitude
        if router_scale > 0.25:
            conf -= 0.04
        elif router_scale > 0.15:
            conf -= 0.03
        # Cap stress penalty if consumed >70% of cap
        if cap_total > 0:
            frac = cap_used / cap_total
            if frac > 0.9:
                conf -= 0.08
            elif frac > 0.7:
                conf -= 0.05
        # No-edit bonus
        if rel_diff(orig_val, new_val) <= 1e-3:
            conf += 0.03
        return clamp(conf)

    # Assemble confidences
    for if_id, r in interim.items():
        router = r.get('local_router')
        peer = peer_of.get(if_id)
        status_comp = clamp(r.get('status_conf', 0.8))
        resolved_status = r.get('status', 'unknown')

        # Pair components with rate-aware tolerance
        if peer and interim.get(peer, {}).get('status') == resolved_status:
            res_fwd = rel_diff(r['tx'], interim[peer]['rx'])
            res_rev = rel_diff(r['rx'], interim[peer]['tx'])
            traffic_tx = max(r['tx'], interim[peer]['rx'], 1.0)
            traffic_rx = max(r['rx'], interim[peer]['tx'], 1.0)
            tol_pair_tx = min(0.12, max(HARDENING_THRESHOLD * 1.5, 5.0 / traffic_tx))
            tol_pair_rx = min(0.12, max(HARDENING_THRESHOLD * 1.5, 5.0 / traffic_rx))
            pair_comp_tx = conf_from_residual_two_slope(res_fwd, tol_pair_tx)
            pair_comp_rx = conf_from_residual_two_slope(res_rev, tol_pair_rx)
        else:
            pair_comp_tx = 0.55
            pair_comp_rx = 0.55

        router_imb = router_final_imbalance.get(router, 0.0)
        router_comp = conf_from_residual_two_slope(router_imb, TOL_ROUTER)

        base_tx_conf = W_PAIR * pair_comp_tx + W_ROUTER * router_comp + W_STATUS * status_comp
        base_rx_conf = W_PAIR * pair_comp_rx + W_ROUTER * router_comp + W_STATUS * status_comp

        # Directional finalize with penalties/bonuses
        tx_small_edit = r['edit_tx_rel'] < 0.15
        rx_small_edit = r['edit_rx_rel'] < 0.15

        tx_cap_used = r.get('cap_used_tx', 0.0)
        tx_cap_total = r.get('cap_total_tx', 0.0)
        rx_cap_used = r.get('cap_used_rx', 0.0)
        rx_cap_total = r.get('cap_total_rx', 0.0)

        rtr_scale = router_scale_mag.get(router, 0.0)

        final_tx_conf = finalize_conf(base_tx_conf, r['edit_tx_rel'], tx_small_edit,
                                      tx_cap_used, tx_cap_total, rtr_scale, r['orig_tx'], r['tx'])
        final_rx_conf = finalize_conf(base_rx_conf, r['edit_rx_rel'], rx_small_edit,
                                      rx_cap_used, rx_cap_total, rtr_scale, r['orig_rx'], r['rx'])

        # If down, align confidences with zero-enforcement logic
        if resolved_status == 'down':
            final_rx_conf = 0.9 if r['orig_rx'] <= TRAFFIC_EVIDENCE_MIN else 0.3
            final_tx_conf = 0.9 if r['orig_tx'] <= TRAFFIC_EVIDENCE_MIN else 0.3

        # Untouched-by-router small bonus to reward stable, consistent counters
        if not r.get('touched_by_router', False):
            final_tx_conf = clamp(final_tx_conf + 0.03)
            final_rx_conf = clamp(final_rx_conf + 0.03)

        r['tx_conf'] = clamp(final_tx_conf)
        r['rx_conf'] = clamp(final_rx_conf)

        # Idle-up slight status confidence damping; anomaly if down but traffic present
        if resolved_status == 'up':
            if r['rx'] <= TRAFFIC_EVIDENCE_MIN and r['tx'] <= TRAFFIC_EVIDENCE_MIN:
                r['status_conf'] = clamp(r['status_conf'] * 0.9)
        elif resolved_status == 'down':
            if r['rx'] > TRAFFIC_EVIDENCE_MIN or r['tx'] > TRAFFIC_EVIDENCE_MIN:
                r['status_conf'] = min(r['status_conf'], 0.3)

    # Prepare result with tuples and unchanged metadata
    result: Dict[str, Dict[str, Tuple]] = {}
    for if_id, r in interim.items():
        repaired_data: Dict[str, Tuple] = {}
        repaired_data['rx_rate'] = (r['orig_rx'], r['rx'], clamp(r['rx_conf']))
        repaired_data['tx_rate'] = (r['orig_tx'], r['tx'], clamp(r['tx_conf']))
        repaired_data['interface_status'] = (r['orig_status'], r['status'], clamp(r['status_conf']))
        repaired_data['connected_to'] = r['connected_to']
        repaired_data['local_router'] = r['local_router']
        repaired_data['remote_router'] = r['remote_router']
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