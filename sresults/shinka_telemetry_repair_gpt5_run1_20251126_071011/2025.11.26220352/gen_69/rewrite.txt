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
    # Small traffic level used to infer link up when statuses disagree (Mbps)
    TRAFFIC_EVIDENCE_MIN = 0.5

    # Pair reconciliation strength
    PAIR_RECONCILE_ALPHA_BASE = 0.30
    PAIR_RECONCILE_ALPHA_LOW = 0.35
    PAIR_RECONCILE_ALPHA_HIGH = 0.20

    # Additive redistribution per-interface staged caps (fraction of pre-router dir value)
    STAGE_CAP_FRACS = [0.25, 0.35, 0.45]
    # Multiplicative pre-step caps and tuning
    MULT_PRE_CAP_FRAC = 0.15
    ALPHA_REF_IMBAL = 0.15
    ALPHA_MIN, ALPHA_MAX = 0.25, 0.60

    # Tie-breaking for direction selection when avg confidences are close
    TIE_EPS_CONF = 0.05

    # Router guard base min/max fractions
    GUARD_MIN, GUARD_MAX = 0.15, 0.35

    # Weight enrichment parameters
    WEIGHT_FLOOR = 0.02
    TOL_PAIR_BASE = HARDENING_THRESHOLD * 1.5
    TOL_ROUTER = HARDENING_THRESHOLD * 2.0
    SMALL_LINK_MBPS = 1.0

    EPS = 1e-9

    def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
        return max(lo, min(hi, x))

    def rel_diff(a: float, b: float) -> float:
        denom = max(abs(a), abs(b), 1e-9)
        return abs(a - b) / denom

    def conf_from_residual(residual: float, tol: float) -> float:
        # Two-slope residual-to-confidence mapping
        tol = max(tol, 1e-9)
        x = residual / tol
        conf = 1.0 - min(1.0, x / 5.0)
        if x > 3.0:
            conf -= 0.1 * (x - 3.0) / 2.0
        return clamp(conf)

    # Initialize structures
    result: Dict[str, Dict[str, Tuple]] = {}
    interim: Dict[str, Dict[str, Any]] = {}

    # Build connected pairs
    visited = set()
    pairs: List[Tuple[str, str]] = []
    for if_id, data in telemetry.items():
        peer = data.get('connected_to')
        if peer and peer in telemetry:
            key = tuple(sorted([if_id, peer]))
            if key not in visited:
                visited.add(key)
                pairs.append((key[0], key[1]))

    # Map each interface to its peer for quick lookup and record paired IDs
    peer_of: Dict[str, str] = {}
    paired_ids = set()
    for a_id, b_id in pairs:
        peer_of[a_id] = b_id
        peer_of[b_id] = a_id
        paired_ids.add(a_id)
        paired_ids.add(b_id)

    # Initialize defaults for all interfaces
    for if_id, data in telemetry.items():
        rx0 = float(data.get('rx_rate', 0.0))
        tx0 = float(data.get('tx_rate', 0.0))
        interim[if_id] = {
            'rx': rx0,
            'tx': tx0,
            'rx_conf': 1.0,
            'tx_conf': 1.0,
            'status': data.get('interface_status', 'unknown'),
            'status_conf': 1.0,
            'connected_to': data.get('connected_to'),
            'local_router': data.get('local_router'),
            'remote_router': data.get('remote_router'),
            # Originals for output tuples
            'orig_rx': rx0,
            'orig_tx': tx0,
            'orig_status': data.get('interface_status', 'unknown'),

            # Pre-router snapshots and delta tracking (for budgeted reconciliation and calibration)
            'pre_router_rx': rx0,
            'pre_router_tx': tx0,
            'router_pos_used_rx': 0.0,
            'router_neg_used_rx': 0.0,
            'router_pos_used_tx': 0.0,
            'router_neg_used_tx': 0.0,

            # Edit tracking for calibration
            'edit_rx_rel': 0.0,
            'edit_tx_rel': 0.0,
            'touched_router': False,
            'touched': False,
        }

    # Pair-level status hardening and symmetry alignment
    for a_id, b_id in pairs:
        a = telemetry[a_id]
        b = telemetry[b_id]
        a_stat = a.get('interface_status', 'unknown')
        b_stat = b.get('interface_status', 'unknown')
        a_rx, a_tx = float(a.get('rx_rate', 0.0)), float(a.get('tx_rate', 0.0))
        b_rx, b_tx = float(b.get('rx_rate', 0.0)), float(b.get('tx_rate', 0.0))
        max_traffic = max(a_rx, a_tx, b_rx, b_tx)

        # Resolve status via redundancy + traffic evidence
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

        interim[a_id]['status'] = resolved_status
        interim[b_id]['status'] = resolved_status
        interim[a_id]['status_conf'] = min(interim[a_id]['status_conf'], status_conf) if interim[a_id]['status_conf'] else status_conf
        interim[b_id]['status_conf'] = min(interim[b_id]['status_conf'], status_conf) if interim[b_id]['status_conf'] else status_conf

        if resolved_status == 'down':
            # Down implies zero traffic
            for ifid, rxv, txv in [(a_id, a_rx, a_tx), (b_id, b_rx, b_tx)]:
                interim[ifid]['rx'] = 0.0
                interim[ifid]['tx'] = 0.0
                interim[ifid]['rx_conf'] = clamp(0.9 if rxv <= TRAFFIC_EVIDENCE_MIN else 0.3)
                interim[ifid]['tx_conf'] = clamp(0.9 if txv <= TRAFFIC_EVIDENCE_MIN else 0.3)
            continue

        # Symmetry hardening for rates on up links
        # a.tx vs b.rx
        d_fwd = rel_diff(a_tx, b_rx)
        if d_fwd <= HARDENING_THRESHOLD:
            v = 0.5 * (a_tx + b_rx)
            conf = clamp(1.0 - 0.5 * d_fwd)
        else:
            v = b_rx if abs(b_rx) > 0 else a_tx
            conf = clamp(1.0 - d_fwd)
        interim[a_id]['tx'] = v
        interim[b_id]['rx'] = v
        interim[a_id]['tx_conf'] = min(interim[a_id]['tx_conf'], conf)
        interim[b_id]['rx_conf'] = min(interim[b_id]['rx_conf'], conf)

        # a.rx vs b.tx
        d_rev = rel_diff(a_rx, b_tx)
        if d_rev <= HARDENING_THRESHOLD:
            v2 = 0.5 * (a_rx + b_tx)
            conf2 = clamp(1.0 - 0.5 * d_rev)
        else:
            v2 = b_tx if abs(b_tx) > 0 else a_rx
            conf2 = clamp(1.0 - d_rev)
        interim[a_id]['rx'] = v2
        interim[b_id]['tx'] = v2
        interim[a_id]['rx_conf'] = min(interim[a_id]['rx_conf'], conf2)
        interim[b_id]['tx_conf'] = min(interim[b_id]['tx_conf'], conf2)

    # Enforce "down implies zero traffic" for unpaired ifaces too
    for if_id, r in interim.items():
        if if_id not in paired_ids and r.get('status') == 'down':
            rx0 = r['rx']
            tx0 = r['tx']
            r['rx'] = 0.0
            r['tx'] = 0.0
            r['rx_conf'] = clamp(0.9 if rx0 <= TRAFFIC_EVIDENCE_MIN else 0.3)
            r['tx_conf'] = clamp(0.9 if tx0 <= TRAFFIC_EVIDENCE_MIN else 0.3)

    # Build router to interfaces map (use provided topology; else derive)
    router_ifaces: Dict[str, List[str]] = {}
    if topology:
        router_ifaces = {r: [i for i in if_list if i in interim] for r, if_list in topology.items()}
    else:
        # Using topology helps flow conservation; derive best-effort if absent.
        for if_id, data in telemetry.items():
            r = data.get('local_router')
            if r is not None and if_id in interim:
                router_ifaces.setdefault(r, []).append(if_id)

    # Helper: pair residual for weighting
    def pair_residual_dir(iface_id: str, direction: str) -> float:
        peer = peer_of.get(iface_id)
        if not peer or interim.get(peer) is None:
            return 0.0
        if interim[iface_id]['status'] != 'up' or interim[peer]['status'] != 'up':
            return 0.0
        if direction == 'rx':
            return rel_diff(interim[iface_id]['rx'], interim[peer]['tx'])
        else:
            return rel_diff(interim[iface_id]['tx'], interim[peer]['rx'])

    # Store per-router scale magnitude for calibration
    router_scale_mag: Dict[str, float] = {}

    # Router-level conservation with multiplicative pre-step and staged additive redistribution
    for router, if_list in router_ifaces.items():
        interfaces = [i for i in if_list if i in interim]
        if not interfaces:
            continue

        # Snapshot pre-router values per interface
        for i in interfaces:
            interim[i]['pre_router_rx'] = interim[i]['rx']
            interim[i]['pre_router_tx'] = interim[i]['tx']
            interim[i]['router_pos_used_rx'] = 0.0
            interim[i]['router_neg_used_rx'] = 0.0
            interim[i]['router_pos_used_tx'] = 0.0
            interim[i]['router_neg_used_tx'] = 0.0

        # Compute sums/conf on up interfaces
        up_list = [i for i in interfaces if interim[i]['status'] == 'up']
        if not up_list:
            continue

        def recalc_sums():
            sum_tx_ = sum(max(0.0, interim[i]['tx']) for i in up_list)
            sum_rx_ = sum(max(0.0, interim[i]['rx']) for i in up_list)
            avg_tx_c = sum(interim[i]['tx_conf'] for i in up_list) / max(1, len(up_list))
            avg_rx_c = sum(interim[i]['rx_conf'] for i in up_list) / max(1, len(up_list))
            return sum_tx_, sum_rx_, avg_tx_c, avg_rx_c

        sum_tx, sum_rx, avg_tx_conf, avg_rx_conf = recalc_sums()
        imbalance = rel_diff(sum_tx, sum_rx)
        if imbalance <= HARDENING_THRESHOLD * 2:
            router_scale_mag[router] = 0.0
        else:
            # Decide direction to scale (less-trusted direction). Tie-break by larger absolute need.
            need_rx = sum_tx - sum_rx
            need_tx = -need_rx
            if abs(avg_tx_conf - avg_rx_conf) < TIE_EPS_CONF:
                scale_rx = abs(need_rx) >= abs(need_tx)
            else:
                scale_rx = avg_tx_conf >= avg_rx_conf

            # Multiplicative pre-step to gently reduce skew
            if scale_rx and sum_rx > 0.0:
                s = sum_tx / sum_rx
            elif (not scale_rx) and sum_tx > 0.0:
                s = sum_rx / sum_tx
            else:
                s = 1.0
            s_bounded = max(0.5, min(2.0, s))
            router_scale_mag[router] = abs(1.0 - s_bounded)

            alpha = clamp(imbalance / ALPHA_REF_IMBAL if ALPHA_REF_IMBAL > 0 else ALPHA_MIN, ALPHA_MIN, ALPHA_MAX)
            m = 1.0 + alpha * (s_bounded - 1.0)
            if abs(m - 1.0) > EPS:
                for i in up_list:
                    if scale_rx:
                        v = max(0.0, interim[i]['rx'])
                    else:
                        v = max(0.0, interim[i]['tx'])
                    proposed = v * m
                    delta = proposed - v
                    cap = MULT_PRE_CAP_FRAC * max(v, 1.0)
                    if delta > 0:
                        delta = min(delta, cap)
                        if scale_rx:
                            interim[i]['router_pos_used_rx'] += abs(delta)
                        else:
                            interim[i]['router_pos_used_tx'] += abs(delta)
                    else:
                        delta = max(delta, -cap)
                        if scale_rx:
                            interim[i]['router_neg_used_rx'] += abs(delta)
                        else:
                            interim[i]['router_neg_used_tx'] += abs(delta)
                    if abs(delta) <= EPS:
                        continue
                    new_v = max(0.0, v + delta)
                    delta_rel = rel_diff(v, new_v)
                    if scale_rx:
                        interim[i]['rx'] = new_v
                        interim[i]['rx_conf'] = clamp(min(interim[i]['rx_conf'],
                                                          1.0 - min(1.0, 0.5 * imbalance + 0.5 * delta_rel)))
                    else:
                        interim[i]['tx'] = new_v
                        interim[i]['tx_conf'] = clamp(min(interim[i]['tx_conf'],
                                                          1.0 - min(1.0, 0.5 * imbalance + 0.5 * delta_rel)))
                    interim[i]['touched_router'] = True
                    interim[i]['touched'] = True
                    if scale_rx:
                        interim[i]['edit_rx_rel'] = max(interim[i]['edit_rx_rel'], delta_rel)
                    else:
                        interim[i]['edit_tx_rel'] = max(interim[i]['edit_tx_rel'], delta_rel)

            # Recompute after pre-step
            sum_tx, sum_rx, avg_tx_conf, avg_rx_conf = recalc_sums()
            imbalance = rel_diff(sum_tx, sum_rx)

            # Direction and targets for additive redistribution
            if abs(avg_tx_conf - avg_rx_conf) < TIE_EPS_CONF:
                # break ties by larger absolute need
                scale_rx = abs(sum_tx - sum_rx) >= abs(sum_rx - sum_tx)
            else:
                scale_rx = avg_tx_conf >= avg_rx_conf

            if scale_rx:
                dir_key = 'rx'
                sum_old = sum(max(0.0, interim[i]['rx']) for i in up_list)
                target_total = sum_tx
            else:
                dir_key = 'tx'
                sum_old = sum(max(0.0, interim[i]['tx']) for i in up_list)
                target_total = sum_rx

            need = target_total - sum_old
            if abs(need) > max(sum_old, target_total, 1.0) * (HARDENING_THRESHOLD * 0.5):
                # Dynamic router guard proportional to imbalance and trust gap
                avg_up_traffic = 0.5 * (sum_tx + sum_rx)
                guard_frac = clamp(GUARD_MIN + 0.5 * imbalance + 0.5 * abs(avg_tx_conf - avg_rx_conf),
                                   GUARD_MIN, GUARD_MAX)
                router_delta_guard = guard_frac * avg_up_traffic
                router_delta_used = 0.0

                # Per-interface remaining capacities per stage (sign-specific)
                def remaining_cap(i_id: str, sign: int, cap_frac: float) -> float:
                    base = max(interim[i_id]['pre_router_rx'] if dir_key == 'rx' else interim[i_id]['pre_router_tx'], 1.0)
                    cap_total = cap_frac * base
                    if sign > 0:
                        used = interim[i_id]['router_pos_used_rx'] if dir_key == 'rx' else interim[i_id]['router_pos_used_tx']
                    else:
                        used = interim[i_id]['router_neg_used_rx'] if dir_key == 'rx' else interim[i_id]['router_neg_used_tx']
                    return max(0.0, cap_total - used)

                # Helper to update used capacity tracking
                def consume_cap(i_id: str, delta: float, cap_frac: float):
                    if delta >= 0:
                        if dir_key == 'rx':
                            interim[i_id]['router_pos_used_rx'] += delta
                        else:
                            interim[i_id]['router_pos_used_tx'] += delta
                    else:
                        if dir_key == 'rx':
                            interim[i_id]['router_neg_used_rx'] += abs(delta)
                        else:
                            interim[i_id]['router_neg_used_tx'] += abs(delta)

                # Two to three passes with staged caps; optional third pass for low-confidence or tiny links
                num_passes = 3
                for pass_idx in range(num_passes):
                    cap_frac = STAGE_CAP_FRACS[pass_idx]
                    # Recompute residual need at pass start
                    if scale_rx:
                        sum_now = sum(max(0.0, interim[i]['rx']) for i in up_list)
                        target_total = sum_tx
                    else:
                        sum_now = sum(max(0.0, interim[i]['tx']) for i in up_list)
                        target_total = sum_rx
                    need = target_total - sum_now
                    if abs(need) <= EPS:
                        break

                    # Determine eligible interfaces for this pass
                    elig = []
                    for i in up_list:
                        # Optional pass 3 is selective: only low-conf or small links
                        if pass_idx == 2:
                            conf_dir = interim[i][f'{dir_key}_conf']
                            vdir = max(0.0, interim[i][dir_key])
                            if conf_dir >= 0.6 and vdir >= 5.0:
                                continue
                        # Freeze interfaces that consumed >80% of total sign-cap unless no other capacity remains
                        base = max(interim[i]['pre_router_rx'] if dir_key == 'rx' else interim[i]['pre_router_tx'], 1.0)
                        total_cap_sign = cap_frac * base
                        used_pos = interim[i]['router_pos_used_rx'] if dir_key == 'rx' else interim[i]['router_pos_used_tx']
                        used_neg = interim[i]['router_neg_used_rx'] if dir_key == 'rx' else interim[i]['router_neg_used_tx']
                        # For the sign we need right now:
                        sign_need = 1 if need > 0 else -1
                        used_sign = used_pos if sign_need > 0 else used_neg
                        consumed_frac = used_sign / max(total_cap_sign, EPS)
                        rem = remaining_cap(i, sign_need, cap_frac)
                        # Defer if over-consumed unless no other capacity exists (handled by elig empty check)
                        if consumed_frac > 0.8 and rem <= EPS:
                            continue
                        if rem > EPS:
                            elig.append(i)

                    if not elig:
                        # If nothing eligible and it's not the last pass, continue to next stage
                        continue

                    # Enriched weights
                    sum_v = sum(max(0.0, interim[i][dir_key]) for i in elig) + EPS
                    weights: Dict[str, float] = {}
                    for i in elig:
                        conf = float(interim[i][f'{dir_key}_conf'])
                        v = max(0.0, float(interim[i][dir_key]))
                        # Pair residual and tolerance
                        peer_resid = pair_residual_dir(i, dir_key)
                        traffic = max(v, 1.0)
                        tol_pair = max(TOL_PAIR_BASE, 5.0 / traffic)
                        resid_term = min(2.0, peer_resid / max(tol_pair, 1e-9))
                        w = 0.6 * (1.0 - conf) + 0.25 * resid_term + 0.15 * (v / sum_v)
                        if v < SMALL_LINK_MBPS:
                            w *= 0.5
                        # Reduce weight on later passes if >70% cap consumed for the needed sign
                        base = max(interim[i]['pre_router_rx'] if dir_key == 'rx' else interim[i]['pre_router_tx'], 1.0)
                        total_cap_sign = cap_frac * base
                        used_sign = (interim[i]['router_pos_used_rx'] if (dir_key == 'rx' and need > 0)
                                     else interim[i]['router_neg_used_rx'] if dir_key == 'rx'
                                     else interim[i]['router_pos_used_tx'] if need > 0
                                     else interim[i]['router_neg_used_tx'])
                        if pass_idx >= 1 and (used_sign / max(total_cap_sign, EPS)) > 0.7:
                            w *= 0.7
                        if peer_resid > 2.0 * tol_pair:
                            w += 0.1
                        weights[i] = max(WEIGHT_FLOOR, w)

                    # Allocation with capacity and router guard
                    sumW = sum(weights[i] for i in elig)
                    if sumW <= EPS:
                        continue

                    for i in elig:
                        if abs(need) <= EPS or router_delta_used >= router_delta_guard - EPS:
                            break
                        quota = need * (weights[i] / sumW)
                        # Clip by sign-specific remaining cap
                        sign_need = 1 if need > 0 else -1
                        rem_cap = remaining_cap(i, sign_need, cap_frac)
                        d = quota
                        if need > 0:
                            d = min(max(0.0, d), rem_cap)
                        else:
                            d = max(min(0.0, d), -rem_cap)

                        # Router guard
                        remaining_guard = router_delta_guard - router_delta_used
                        if abs(d) > remaining_guard:
                            d = max(-remaining_guard, min(remaining_guard, d))

                        if abs(d) <= EPS:
                            continue

                        old_v = interim[i][dir_key]
                        new_v = max(0.0, old_v + d)
                        delta_rel = rel_diff(old_v, new_v)

                        # Apply
                        interim[i][dir_key] = new_v
                        consume_cap(i, d, cap_frac)
                        router_delta_used += abs(d)
                        need -= d

                        # Confidence and edit tracking
                        penalty = 1.0 - min(1.0, imbalance + 0.5 * delta_rel + abs(1.0 - s_bounded) * 0.5)
                        if dir_key == 'rx':
                            interim[i]['rx_conf'] = clamp(min(interim[i]['rx_conf'], penalty))
                            interim[i]['edit_rx_rel'] = max(interim[i]['edit_rx_rel'], delta_rel)
                        else:
                            interim[i]['tx_conf'] = clamp(min(interim[i]['tx_conf'], penalty))
                            interim[i]['edit_tx_rel'] = max(interim[i]['edit_tx_rel'], delta_rel)
                        interim[i]['touched_router'] = True
                        interim[i]['touched'] = True

                    if router_delta_used >= router_delta_guard - EPS:
                        break

        # End router loop per-router

    # Targeted, asymmetric pair reconciliation with remaining per-interface budgets
    for a_id, b_id in pairs:
        if a_id not in interim or b_id not in interim:
            continue
        if interim[a_id].get('status') != 'up' or interim[b_id].get('status') != 'up':
            continue
        # Only reconcile if any side was changed by router edits
        if not (interim[a_id].get('touched_router') or interim[b_id].get('touched_router')):
            continue

        def remaining_pair_cap(i_id: str, direction: str, move: float) -> float:
            pre = interim[i_id]['pre_router_rx'] if direction == 'rx' else interim[i_id]['pre_router_tx']
            total_cap = 0.20 * max(pre, 1.0)
            if move >= 0:
                used = interim[i_id]['router_pos_used_rx'] if direction == 'rx' else interim[i_id]['router_pos_used_tx']
            else:
                used = interim[i_id]['router_neg_used_rx'] if direction == 'rx' else interim[i_id]['router_neg_used_tx']
            return max(0.0, total_cap - used)

        # Forward direction: a.tx vs b.rx
        a_tx = interim[a_id]['tx']
        b_rx = interim[b_id]['rx']
        traffic_tx = max(a_tx, b_rx, 1.0)
        tol_pair_post = max(0.02, 2.5 / traffic_tx)
        res_fwd = rel_diff(a_tx, b_rx)

        if res_fwd > tol_pair_post:
            v_mid = 0.5 * (a_tx + b_rx)
            # Asymmetric alphas based on confidences
            a_conf = interim[a_id]['tx_conf']
            b_conf = interim[b_id]['rx_conf']
            alpha_a = PAIR_RECONCILE_ALPHA_LOW if a_conf < b_conf else PAIR_RECONCILE_ALPHA_HIGH
            alpha_b = PAIR_RECONCILE_ALPHA_LOW if b_conf < a_conf else PAIR_RECONCILE_ALPHA_HIGH
            move_a = alpha_a * (v_mid - a_tx)
            move_b = alpha_b * (v_mid - b_rx)
            cap_a = remaining_pair_cap(a_id, 'tx', move_a)
            cap_b = remaining_pair_cap(b_id, 'rx', move_b)
            # Clamp by remaining caps
            if move_a >= 0:
                move_a = min(move_a, cap_a)
            else:
                move_a = max(move_a, -cap_a)
            if move_b >= 0:
                move_b = min(move_b, cap_b)
            else:
                move_b = max(move_b, -cap_b)

            a_tx_new = max(0.0, a_tx + move_a)
            b_rx_new = max(0.0, b_rx + move_b)

            if a_tx_new != a_tx:
                delta_rel = rel_diff(a_tx, a_tx_new)
                interim[a_id]['tx'] = a_tx_new
                # Mild penalty scaled by residual fraction
                pen = 1.0 - 0.3 * min(1.0, res_fwd / max(tol_pair_post, 1e-9))
                interim[a_id]['tx_conf'] = clamp(interim[a_id]['tx_conf'] * pen)
                interim[a_id]['edit_tx_rel'] = max(interim[a_id]['edit_tx_rel'], delta_rel)
                interim[a_id]['touched'] = True
            if b_rx_new != b_rx:
                delta_rel = rel_diff(b_rx, b_rx_new)
                interim[b_id]['rx'] = b_rx_new
                pen = 1.0 - 0.3 * min(1.0, res_fwd / max(tol_pair_post, 1e-9))
                interim[b_id]['rx_conf'] = clamp(interim[b_id]['rx_conf'] * pen)
                interim[b_id]['edit_rx_rel'] = max(interim[b_id]['edit_rx_rel'], delta_rel)
                interim[b_id]['touched'] = True

        # Reverse direction: a.rx vs b.tx
        a_rx = interim[a_id]['rx']
        b_tx = interim[b_id]['tx']
        traffic_rx = max(a_rx, b_tx, 1.0)
        tol_pair_post2 = max(0.02, 2.5 / traffic_rx)
        res_rev = rel_diff(a_rx, b_tx)

        if res_rev > tol_pair_post2:
            v_mid2 = 0.5 * (a_rx + b_tx)
            a_conf2 = interim[a_id]['rx_conf']
            b_conf2 = interim[b_id]['tx_conf']
            alpha_a2 = PAIR_RECONCILE_ALPHA_LOW if a_conf2 < b_conf2 else PAIR_RECONCILE_ALPHA_HIGH
            alpha_b2 = PAIR_RECONCILE_ALPHA_LOW if b_conf2 < a_conf2 else PAIR_RECONCILE_ALPHA_HIGH
            move_a2 = alpha_a2 * (v_mid2 - a_rx)
            move_b2 = alpha_b2 * (v_mid2 - b_tx)
            cap_a2 = remaining_pair_cap(a_id, 'rx', move_a2)
            cap_b2 = remaining_pair_cap(b_id, 'tx', move_b2)

            if move_a2 >= 0:
                move_a2 = min(move_a2, cap_a2)
            else:
                move_a2 = max(move_a2, -cap_a2)
            if move_b2 >= 0:
                move_b2 = min(move_b2, cap_b2)
            else:
                move_b2 = max(move_b2, -cap_b2)

            a_rx_new = max(0.0, a_rx + move_a2)
            b_tx_new = max(0.0, b_tx + move_b2)

            if a_rx_new != a_rx:
                delta_rel = rel_diff(a_rx, a_rx_new)
                interim[a_id]['rx'] = a_rx_new
                pen = 1.0 - 0.3 * min(1.0, res_rev / max(tol_pair_post2, 1e-9))
                interim[a_id]['rx_conf'] = clamp(interim[a_id]['rx_conf'] * pen)
                interim[a_id]['edit_rx_rel'] = max(interim[a_id]['edit_rx_rel'], delta_rel)
                interim[a_id]['touched'] = True
            if b_tx_new != b_tx:
                delta_rel = rel_diff(b_tx, b_tx_new)
                interim[b_id]['tx'] = b_tx_new
                pen = 1.0 - 0.3 * min(1.0, res_rev / max(tol_pair_post2, 1e-9))
                interim[b_id]['tx_conf'] = clamp(interim[b_id]['tx_conf'] * pen)
                interim[b_id]['edit_tx_rel'] = max(interim[b_id]['edit_tx_rel'], delta_rel)
                interim[b_id]['touched'] = True

    # Final confidence calibration based on post-repair invariants
    # Compute per-router imbalance residuals
    router_final_imbalance: Dict[str, float] = {}
    for router, if_list in router_ifaces.items():
        up_ifaces = [i for i in if_list if i in interim and interim[i].get('status') == 'up']
        if not up_ifaces:
            router_final_imbalance[router] = 0.0
            continue
        sum_tx = sum(max(0.0, interim[i]['tx']) for i in up_ifaces)
        sum_rx = sum(max(0.0, interim[i]['rx']) for i in up_ifaces)
        router_final_imbalance[router] = rel_diff(sum_tx, sum_rx)

    # Weights for components
    w_pair, w_router, w_status = 0.6, 0.3, 0.1

    for if_id, r in interim.items():
        router = r.get('local_router')
        peer = peer_of.get(if_id)

        status_comp = clamp(r.get('status_conf', 0.8))
        resolved_status = r.get('status', 'unknown')

        # Pair residual confidences with traffic-aware tolerances
        if peer and interim.get(peer, {}).get('status') == resolved_status:
            res_fwd = rel_diff(r['tx'], interim[peer]['rx'])
            res_rev = rel_diff(r['rx'], interim[peer]['tx'])
            traffic_tx = max(r['tx'], interim[peer]['rx'], 1.0)
            traffic_rx = max(r['rx'], interim[peer]['tx'], 1.0)
            tol_pair_tx = min(0.12, max(TOL_PAIR_BASE, 5.0 / traffic_tx))
            tol_pair_rx = min(0.12, max(TOL_PAIR_BASE, 5.0 / traffic_rx))
            pair_comp_tx = conf_from_residual(res_fwd, tol_pair_tx)
            pair_comp_rx = conf_from_residual(res_rev, tol_pair_rx)
        else:
            pair_comp_tx = 0.55
            pair_comp_rx = 0.55

        router_imb = router_final_imbalance.get(router, 0.0)
        router_comp = conf_from_residual(router_imb, TOL_ROUTER)

        base_tx_conf = w_pair * pair_comp_tx + w_router * router_comp + w_status * status_comp
        base_rx_conf = w_pair * pair_comp_rx + w_router * router_comp + w_status * status_comp

        # Edit penalty taper
        delta_tx_rel = r.get('edit_tx_rel', 0.0)
        delta_rx_rel = r.get('edit_rx_rel', 0.0)
        pen_tx = max(0.0, delta_tx_rel - HARDENING_THRESHOLD)
        pen_rx = max(0.0, delta_rx_rel - HARDENING_THRESHOLD)
        weight_tx = 0.4 if delta_tx_rel < 0.15 else 0.5
        weight_rx = 0.4 if delta_rx_rel < 0.15 else 0.5
        final_tx_conf = clamp(base_tx_conf * (1.0 - weight_tx * pen_tx))
        final_rx_conf = clamp(base_rx_conf * (1.0 - weight_rx * pen_rx))

        # Router scale penalty if strong scaling at router
        scale_mag = router_scale_mag.get(router, 0.0)
        if scale_mag > 0.25:
            # 0.03 to 0.05 penalty depending on excess scale
            excess = (scale_mag - 0.25) / 0.25  # 0..1 for 0.25..0.5+
            penalty_scale = 0.03 + 0.02 * clamp(excess, 0.0, 1.0)
            final_tx_conf = clamp(final_tx_conf - penalty_scale)
            final_rx_conf = clamp(final_rx_conf - penalty_scale)

        # Cap-intensity penalty: up to 0.08 if >70% of cumulative sign-cap consumed
        # Compute cumulative consumption vs max stage cap (0.45)
        pre_rx = max(r.get('pre_router_rx', r['orig_rx']), 1.0)
        pre_tx = max(r.get('pre_router_tx', r['orig_tx']), 1.0)
        max_cap_rx = STAGE_CAP_FRACS[-1] * pre_rx
        max_cap_tx = STAGE_CAP_FRACS[-1] * pre_tx
        used_pos_rx = r.get('router_pos_used_rx', 0.0)
        used_neg_rx = r.get('router_neg_used_rx', 0.0)
        used_pos_tx = r.get('router_pos_used_tx', 0.0)
        used_neg_tx = r.get('router_neg_used_tx', 0.0)
        # Use the relevant sign for each direction by actual edit direction if any; else take max
        cons_rx = max(used_pos_rx, used_neg_rx) / max(max_cap_rx, EPS)
        cons_tx = max(used_pos_tx, used_neg_tx) / max(max_cap_tx, EPS)
        if cons_rx > 0.7:
            final_rx_conf = clamp(final_rx_conf - 0.08 * (cons_rx - 0.7) / 0.3)
        if cons_tx > 0.7:
            final_tx_conf = clamp(final_tx_conf - 0.08 * (cons_tx - 0.7) / 0.3)

        # No-edit bonuses
        def no_edit_bonus(orig: float, new: float) -> float:
            return 0.05 if rel_diff(orig, new) <= 1e-3 else 0.0

        final_tx_conf = clamp(final_tx_conf + no_edit_bonus(r['orig_tx'], r['tx']))
        final_rx_conf = clamp(final_rx_conf + no_edit_bonus(r['orig_rx'], r['rx']))

        # Fully untouched interface small bonus
        if (r.get('edit_tx_rel', 0.0) <= 1e-6) and (r.get('edit_rx_rel', 0.0) <= 1e-6):
            final_tx_conf = clamp(final_tx_conf + 0.03)
            final_rx_conf = clamp(final_rx_conf + 0.03)

        # Down status calibration
        if resolved_status == 'down':
            final_rx_conf = 0.9 if r['orig_rx'] <= TRAFFIC_EVIDENCE_MIN else 0.3
            final_tx_conf = 0.9 if r['orig_tx'] <= TRAFFIC_EVIDENCE_MIN else 0.3

        r['tx_conf'] = clamp(final_tx_conf)
        r['rx_conf'] = clamp(final_rx_conf)

        # Subtle status calibration
        if resolved_status == 'up':
            if r['rx'] <= TRAFFIC_EVIDENCE_MIN and r['tx'] <= TRAFFIC_EVIDENCE_MIN:
                r['status_conf'] = clamp(r['status_conf'] * 0.9)
        elif resolved_status == 'down':
            if r['rx'] > TRAFFIC_EVIDENCE_MIN or r['tx'] > TRAFFIC_EVIDENCE_MIN:
                r['status_conf'] = clamp(min(r['status_conf'], 0.3))

    # Assemble final result with (original, repaired, confidence) tuples and unchanged metadata
    for if_id, data in telemetry.items():
        repaired_data: Dict[str, Tuple] = {}
        r = interim[if_id]
        repaired_data['rx_rate'] = (r['orig_rx'], r['rx'], clamp(r['rx_conf']))
        repaired_data['tx_rate'] = (r['orig_tx'], r['tx'], clamp(r['tx_conf']))
        repaired_data['interface_status'] = (r['orig_status'], r['status'], clamp(r['status_conf']))

        # Copy metadata unchanged
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