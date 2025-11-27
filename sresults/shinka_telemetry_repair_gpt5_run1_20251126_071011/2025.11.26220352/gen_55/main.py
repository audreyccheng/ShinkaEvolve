# EVOLVE-BLOCK-START
"""
Network telemetry repair algorithm that detects and corrects inconsistencies
in network interface telemetry data using topology relationships.

Staged pipeline:
  1) Pair status harmonization and link-symmetry hardening
  2) Router-level flow conservation with multiplicative pre-step + additive redistribution
  3) Targeted asymmetrical pair reconciliation
  4) Confidence calibration from pair/router residuals, edits, and guards
"""
from typing import Dict, Any, Tuple, List


def repair_network_telemetry(telemetry: Dict[str, Dict[str, Any]],
                             topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:
    # Constants from research and tuned recommendations
    HARDENING_THRESHOLD = 0.02
    TRAFFIC_EVIDENCE_MIN = 0.5
    # Multiplicative pre-step (Recommendation 1)
    MULT_PRESTEP_CAP_FRAC = 0.15
    TIE_EPS_CONF = 0.05
    # Additive redistribution (Recommendations 2,3)
    CAP_RAMP_FRACS = [0.25, 0.35, 0.45]
    ROUTER_TOTAL_DELTA_FRAC = 0.25
    # Pair reconciliation (Recommendation 4)
    PAIR_RECONCILE_ALPHA_LOW = 0.35
    PAIR_RECONCILE_ALPHA_HIGH = 0.20
    PAIR_RECONCILE_PER_IF_CAP = 0.20
    # Confidence calibration (Recommendation 5)
    CHANGE_PENALTY_WEIGHT_SMALL = 0.40
    CHANGE_PENALTY_WEIGHT_LARGE = 0.50
    SCALE_PENALTY_THRESH = 0.25
    TOUCH_BONUS = 0.03
    # Other tolerances
    TOL_PAIR_BASE = HARDENING_THRESHOLD * 1.5
    TOL_ROUTER = HARDENING_THRESHOLD * 2.0
    EPS = 1e-9

    def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
        return max(lo, min(hi, x))

    def rel_diff(a: float, b: float) -> float:
        denom = max(abs(a), abs(b), 1e-9)
        return abs(a - b) / denom

    def conf_from_residual(residual: float, tol: float) -> float:
        tol = max(tol, 1e-9)
        x = residual / tol
        conf = 1.0 - min(1.0, x / 5.0)
        if x > 3.0:
            conf -= 0.1 * (x - 3.0) / 2.0
        return clamp(conf)

    # Build pairs and peer map
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

    # Build router->interfaces map
    router_ifaces: Dict[str, List[str]] = {}
    if topology:
        router_ifaces = {r: [i for i in if_list if i in telemetry] for r, if_list in topology.items()}
    else:
        # Fallback: derive from telemetry metadata
        for if_id, data in telemetry.items():
            r = data.get('local_router')
            if r is not None:
                router_ifaces.setdefault(r, []).append(if_id)

    # Initialize state per interface
    st: Dict[str, Dict[str, Any]] = {}
    for if_id, data in telemetry.items():
        st[if_id] = {
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
            # Tracking for calibration
            'touched': False,
            'edit_rx_rel': 0.0,
            'edit_tx_rel': 0.0,
            'cap_hit_rx': False,
            'cap_hit_tx': False,
            'scaled_rel_rx': 0.0,
            'scaled_rel_tx': 0.0,
            'router_scale_mag': 0.0,
            'cap_consumed_rx': 0.0,
            'cap_consumed_tx': 0.0,
        }

    # Stage 1: Pair-level hardening and status reconciliation
    for a_id, b_id in pairs:
        a = telemetry[a_id]
        b = telemetry[b_id]
        a_stat = a.get('interface_status', 'unknown')
        b_stat = b.get('interface_status', 'unknown')
        a_rx, a_tx = float(a.get('rx_rate', 0.0)), float(a.get('tx_rate', 0.0))
        b_rx, b_tx = float(b.get('rx_rate', 0.0)), float(b.get('tx_rate', 0.0))
        max_traffic = max(a_rx, a_tx, b_rx, b_tx)

        # Status consistency using traffic evidence
        if a_stat == b_stat:
            resolved = a_stat
            s_conf = 0.95 if resolved in ('up', 'down') else 0.7
        else:
            if max_traffic > TRAFFIC_EVIDENCE_MIN:
                resolved = 'up'
                s_conf = 0.85
            else:
                resolved = 'down'
                s_conf = 0.75

        for ifid in (a_id, b_id):
            st[ifid]['status'] = resolved
            st[ifid]['status_conf'] = min(st[ifid]['status_conf'], s_conf)

        if resolved == 'down':
            # Down links carry no traffic
            for ifid, rx0, tx0 in [(a_id, a_rx, a_tx), (b_id, b_rx, b_tx)]:
                st[ifid]['rx'] = 0.0
                st[ifid]['tx'] = 0.0
                st[ifid]['rx_conf'] = clamp(0.9 if rx0 <= TRAFFIC_EVIDENCE_MIN else 0.3)
                st[ifid]['tx_conf'] = clamp(0.9 if tx0 <= TRAFFIC_EVIDENCE_MIN else 0.3)
            continue

        # Harden rates using symmetry
        # a.tx ~ b.rx
        d_fwd = rel_diff(a_tx, b_rx)
        if d_fwd <= HARDENING_THRESHOLD:
            v = 0.5 * (a_tx + b_rx)
            conf = clamp(1.0 - 0.5 * d_fwd)
        else:
            v = b_rx if b_rx > 0 else a_tx
            conf = clamp(1.0 - d_fwd)
        st[a_id]['tx'] = v
        st[b_id]['rx'] = v
        st[a_id]['tx_conf'] = min(st[a_id]['tx_conf'], conf)
        st[b_id]['rx_conf'] = min(st[b_id]['rx_conf'], conf)

        # a.rx ~ b.tx
        d_rev = rel_diff(a_rx, b_tx)
        if d_rev <= HARDENING_THRESHOLD:
            v2 = 0.5 * (a_rx + b_tx)
            conf2 = clamp(1.0 - 0.5 * d_rev)
        else:
            v2 = b_tx if b_tx > 0 else a_rx
            conf2 = clamp(1.0 - d_rev)
        st[a_id]['rx'] = v2
        st[b_id]['tx'] = v2
        st[a_id]['rx_conf'] = min(st[a_id]['rx_conf'], conf2)
        st[b_id]['tx_conf'] = min(st[b_id]['tx_conf'], conf2)

    # Also enforce down->zero for unpaired
    for if_id, r in st.items():
        if if_id not in paired_ids and r.get('status') == 'down':
            rx0, tx0 = r['rx'], r['tx']
            r['rx'] = 0.0
            r['tx'] = 0.0
            r['rx_conf'] = clamp(0.9 if rx0 <= TRAFFIC_EVIDENCE_MIN else 0.3)
            r['tx_conf'] = clamp(0.9 if tx0 <= TRAFFIC_EVIDENCE_MIN else 0.3)

    # Helper for pair residual in a specific direction
    def pair_residual_dir(iface_id: str, dir_key: str) -> float:
        peer = peer_of.get(iface_id)
        if not peer:
            return 0.0
        if st[iface_id]['status'] != 'up' or st[peer]['status'] != 'up':
            return 0.0
        if dir_key == 'rx':
            return rel_diff(st[iface_id]['rx'], st[peer]['tx'])
        else:
            return rel_diff(st[iface_id]['tx'], st[peer]['rx'])

    # Stage 2: Router-level flow conservation with pre-step + redistribution
    for router, if_list in router_ifaces.items():
        up_list = [i for i in if_list if i in st and st[i]['status'] == 'up']
        if not up_list:
            continue

        sum_tx = sum(max(0.0, st[i]['tx']) for i in up_list)
        sum_rx = sum(max(0.0, st[i]['rx']) for i in up_list)
        avg_tx_conf = sum(st[i]['tx_conf'] for i in up_list) / max(1, len(up_list))
        avg_rx_conf = sum(st[i]['rx_conf'] for i in up_list) / max(1, len(up_list))
        imbalance = rel_diff(sum_tx, sum_rx)
        if imbalance <= HARDENING_THRESHOLD * 2:
            continue

        need_rx = sum_tx - sum_rx
        need_tx = -need_rx
        # Choose less-trusted direction; tie-break with larger absolute need
        if abs(avg_tx_conf - avg_rx_conf) <= TIE_EPS_CONF:
            scale_rx = abs(need_rx) >= abs(need_tx)
        else:
            scale_rx = avg_tx_conf >= avg_rx_conf

        # Ratio to equalize and bounded
        if scale_rx and sum_rx > 0:
            s = sum_tx / sum_rx
        elif (not scale_rx) and sum_tx > 0:
            s = sum_rx / sum_tx
        else:
            s = 1.0
        s_bounded = max(0.5, min(2.0, s))
        for i in up_list:
            st[i]['router_scale_mag'] = max(st[i]['router_scale_mag'], abs(1.0 - s_bounded))

        # Multiplicative pre-step (bounded ±15%)
        alpha = clamp(imbalance / 0.15, 0.25, 0.60)
        m = 1.0 + alpha * (s_bounded - 1.0)
        if abs(1.0 - m) > 1e-6:
            for i in up_list:
                key = 'rx' if scale_rx else 'tx'
                old_v = max(0.0, float(st[i][key]))
                proposed = old_v * m
                delta = proposed - old_v
                cap = MULT_PRESTEP_CAP_FRAC * max(old_v, 1.0)
                delta = max(-cap, min(cap, delta))
                new_v = max(0.0, old_v + delta)
                if abs(delta) > EPS:
                    st[i][key] = new_v
                    delta_rel = rel_diff(old_v, new_v)
                    st[i]['touched'] = True
                    if key == 'rx':
                        st[i]['scaled_rel_rx'] = max(st[i]['scaled_rel_rx'], delta_rel)
                        st[i]['edit_rx_rel'] = max(st[i]['edit_rx_rel'], delta_rel)
                        st[i]['rx_conf'] = clamp(min(st[i]['rx_conf'],
                                                     1.0 - min(1.0, 0.5 * imbalance + 0.5 * delta_rel + 0.5 * abs(1.0 - s_bounded)))))
                    else:
                        st[i]['scaled_rel_tx'] = max(st[i]['scaled_rel_tx'], delta_rel)
                        st[i]['edit_tx_rel'] = max(st[i]['edit_tx_rel'], delta_rel)
                        st[i]['tx_conf'] = clamp(min(st[i]['tx_conf'],
                                                     1.0 - min(1.0, 0.5 * imbalance + 0.5 * delta_rel + 0.5 * abs(1.0 - s_bounded)))))

        # Recompute need after multiplicative step
        if scale_rx:
            sum_old = sum(max(0.0, st[i]['rx']) for i in up_list)
            target_total = sum(max(0.0, st[i]['tx']) for i in up_list)
            dir_key = 'rx'
            conf_key = 'rx_conf'
            cap_hit_key = 'cap_hit_rx'
            cap_frac_key = 'cap_consumed_rx'
        else:
            sum_old = sum(max(0.0, st[i]['tx']) for i in up_list)
            target_total = sum(max(0.0, st[i]['rx']) for i in up_list)
            dir_key = 'tx'
            conf_key = 'tx_conf'
            cap_hit_key = 'cap_hit_tx'
            cap_frac_key = 'cap_consumed_tx'

        need = target_total - sum_old
        tol_need = max(sum_old, target_total, 1.0) * (HARDENING_THRESHOLD * 0.5)
        if abs(need) <= tol_need:
            pass
        else:
            # Router total delta guard
            avg_up_traffic = 0.5 * (sum_tx + sum_rx)
            router_delta_guard = ROUTER_TOTAL_DELTA_FRAC * avg_up_traffic
            router_delta_used = 0.0

            # Base values and caps per pass
            values = {i: max(0.0, st[i][dir_key]) for i in up_list}
            base_confs = {i: float(st[i][conf_key]) for i in up_list}
            cap_total = {i: CAP_RAMP_FRACS[0] * max(values[i], 1.0) for i in up_list}
            cap_rem = {i: cap_total[i] for i in up_list}
            inc_pass2 = {i: (CAP_RAMP_FRACS[1] - CAP_RAMP_FRACS[0]) * max(values[i], 1.0) for i in up_list}
            inc_pass3 = {}
            for i in up_list:
                v0 = values[i]
                allow_more = (base_confs[i] < 0.6) or (v0 < 5.0)
                inc_pass3[i] = ((CAP_RAMP_FRACS[2] - CAP_RAMP_FRACS[1]) * max(v0, 1.0)) if allow_more else 0.0
            consumed_frac_prev = {i: 0.0 for i in up_list}

            def build_weights(pass_idx: int) -> Dict[str, float]:
                sum_v = sum(max(0.0, values[i]) for i in up_list) + EPS
                weights: Dict[str, float] = {}
                for i in up_list:
                    conf = base_confs[i]
                    v = max(0.0, values[i])
                    pr = pair_residual_dir(i, dir_key)
                    pair_term = min(2.0, pr / max(TOL_PAIR_BASE, 1e-9))
                    w = 0.6 * (1.0 - conf) + 0.25 * pair_term + 0.15 * (v / sum_v)
                    if v < 1.0:
                        w *= 0.5
                    if pass_idx > 0 and consumed_frac_prev.get(i, 0.0) > 0.7:
                        w *= 0.7
                    weights[i] = max(0.02, w)
                return weights

            for pass_idx in range(3):
                if abs(need) <= EPS or router_delta_used >= router_delta_guard - EPS:
                    break
                if pass_idx == 1:
                    for i in up_list:
                        cap_total[i] += inc_pass2[i]
                        cap_rem[i] += inc_pass2[i]
                elif pass_idx == 2:
                    for i in up_list:
                        cap_total[i] += inc_pass3[i]
                        cap_rem[i] += inc_pass3[i]

                weights = build_weights(pass_idx)

                # Freeze interfaces >80% consumed unless no other capacity remains
                def eligible_list() -> List[str]:
                    # Primary eligible set
                    elig = [i for i in up_list if cap_rem[i] > EPS and consumed_frac_prev[i] <= 0.8]
                    if not elig:
                        elig = [i for i in up_list if cap_rem[i] > EPS]
                    return elig

                elig = eligible_list()
                if not elig:
                    continue
                sumW = sum(weights[i] for i in elig)
                if sumW <= EPS:
                    continue

                for i in elig:
                    if abs(need) <= EPS or router_delta_used >= router_delta_guard - EPS:
                        break
                    quota = need * (weights[i] / sumW)
                    d = max(-cap_rem[i], min(cap_rem[i], quota))
                    remaining_guard = router_delta_guard - router_delta_used
                    if abs(d) > remaining_guard:
                        d = remaining_guard if d > 0 else -remaining_guard
                    if abs(d) <= EPS:
                        continue

                    old_v = values[i]
                    new_v = max(0.0, old_v + d)
                    values[i] = new_v
                    cap_rem[i] = max(0.0, cap_rem[i] - abs(d))
                    router_delta_used += abs(d)
                    need -= d

                    # Write back, confidence penalty, tracking
                    delta_rel = rel_diff(old_v, new_v)
                    st[i][dir_key] = new_v
                    st[i]['touched'] = True
                    if dir_key == 'rx':
                        st[i]['edit_rx_rel'] = max(st[i]['edit_rx_rel'], delta_rel)
                        if cap_rem[i] <= EPS * 10:
                            st[i]['cap_hit_rx'] = True
                        st[i]['rx_conf'] = clamp(min(st[i]['rx_conf'],
                                                     1.0 - min(1.0, imbalance + 0.5 * delta_rel + 0.5 * abs(1.0 - s_bounded)))))
                    else:
                        st[i]['edit_tx_rel'] = max(st[i]['edit_tx_rel'], delta_rel)
                        if cap_rem[i] <= EPS * 10:
                            st[i]['cap_hit_tx'] = True
                        st[i]['tx_conf'] = clamp(min(st[i]['tx_conf'],
                                                     1.0 - min(1.0, imbalance + 0.5 * delta_rel + 0.5 * abs(1.0 - s_bounded)))))

                for i in up_list:
                    total_cap_i = max(cap_total[i], EPS)
                    consumed_frac_prev[i] = 1.0 - (cap_rem[i] / total_cap_i)

            # Save cap consumption fraction for calibration
            for i in up_list:
                st[i][cap_frac_key] = max(st[i][cap_frac_key], consumed_frac_prev[i])

    # Stage 3: Targeted asymmetrical pair reconciliation (only touched pairs)
    for a_id, b_id in pairs:
        if st[a_id]['status'] != 'up' or st[b_id]['status'] != 'up':
            continue
        if not (st[a_id]['touched'] or st[b_id]['touched']):
            continue

        # Forward: a.tx vs b.rx
        a_tx = st[a_id]['tx']; b_rx = st[b_id]['rx']
        traffic_tx = max(a_tx, b_rx, 1.0)
        tol_tx = max(0.02, 2.5 / traffic_tx)
        res_fwd = rel_diff(a_tx, b_rx)
        if res_fwd > tol_tx:
            v_mid = 0.5 * (a_tx + b_rx)
            # Lower-confidence side gets stronger pull
            if st[a_id]['tx_conf'] <= st[b_id]['rx_conf']:
                alpha_a, alpha_b = PAIR_RECONCILE_ALPHA_LOW, PAIR_RECONCILE_ALPHA_HIGH
            else:
                alpha_a, alpha_b = PAIR_RECONCILE_ALPHA_HIGH, PAIR_RECONCILE_ALPHA_LOW
            move_a = max(-PAIR_RECONCILE_PER_IF_CAP * max(a_tx, 1.0), min(PAIR_RECONCILE_PER_IF_CAP * max(a_tx, 1.0), alpha_a * (v_mid - a_tx)))
            move_b = max(-PAIR_RECONCILE_PER_IF_CAP * max(b_rx, 1.0), min(PAIR_RECONCILE_PER_IF_CAP * max(b_rx, 1.0), alpha_b * (v_mid - b_rx)))
            a_tx_new = max(0.0, a_tx + move_a)
            b_rx_new = max(0.0, b_rx + move_b)
            if a_tx_new != a_tx:
                st[a_id]['tx'] = a_tx_new
                st[a_id]['touched'] = True
                st[a_id]['tx_conf'] = clamp(st[a_id]['tx_conf'] * (1.0 - 0.3 * min(1.0, res_fwd / tol_tx)))
            if b_rx_new != b_rx:
                st[b_id]['rx'] = b_rx_new
                st[b_id]['touched'] = True
                st[b_id]['rx_conf'] = clamp(st[b_id]['rx_conf'] * (1.0 - 0.3 * min(1.0, res_fwd / tol_tx)))

        # Reverse: a.rx vs b.tx
        a_rx = st[a_id]['rx']; b_tx = st[b_id]['tx']
        traffic_rx = max(a_rx, b_tx, 1.0)
        tol_rx = max(0.02, 2.5 / traffic_rx)
        res_rev = rel_diff(a_rx, b_tx)
        if res_rev > tol_rx:
            v_mid2 = 0.5 * (a_rx + b_tx)
            if st[a_id]['rx_conf'] <= st[b_id]['tx_conf']:
                alpha_a2, alpha_b2 = PAIR_RECONCILE_ALPHA_LOW, PAIR_RECONCILE_ALPHA_HIGH
            else:
                alpha_a2, alpha_b2 = PAIR_RECONCILE_ALPHA_HIGH, PAIR_RECONCILE_ALPHA_LOW
            move_a2 = max(-PAIR_RECONCILE_PER_IF_CAP * max(a_rx, 1.0), min(PAIR_RECONCILE_PER_IF_CAP * max(a_rx, 1.0), alpha_a2 * (v_mid2 - a_rx)))
            move_b2 = max(-PAIR_RECONCILE_PER_IF_CAP * max(b_tx, 1.0), min(PAIR_RECONCILE_PER_IF_CAP * max(b_tx, 1.0), alpha_b2 * (v_mid2 - b_tx)))
            a_rx_new = max(0.0, a_rx + move_a2)
            b_tx_new = max(0.0, b_tx + move_b2)
            if a_rx_new != a_rx:
                st[a_id]['rx'] = a_rx_new
                st[a_id]['touched'] = True
                st[a_id]['rx_conf'] = clamp(st[a_id]['rx_conf'] * (1.0 - 0.3 * min(1.0, res_rev / tol_rx)))
            if b_tx_new != b_tx:
                st[b_id]['tx'] = b_tx_new
                st[b_id]['touched'] = True
                st[b_id]['tx_conf'] = clamp(st[b_id]['tx_conf'] * (1.0 - 0.3 * min(1.0, res_rev / tol_rx)))

    # Stage 4: Confidence calibration
    # Compute router residuals after all edits
    router_final_imbalance: Dict[str, float] = {}
    for router, if_list in router_ifaces.items():
        up_ifaces = [i for i in if_list if i in st and st[i]['status'] == 'up']
        if not up_ifaces:
            router_final_imbalance[router] = 0.0
            continue
        s_tx = sum(max(0.0, st[i]['tx']) for i in up_ifaces)
        s_rx = sum(max(0.0, st[i]['rx']) for i in up_ifaces)
        router_final_imbalance[router] = rel_diff(s_tx, s_rx)

    # Build final confidences
    for if_id, r in st.items():
        router = r.get('local_router')
        peer = peer_of.get(if_id)
        resolved_status = r.get('status', 'unknown')
        status_comp = clamp(r.get('status_conf', 0.8))

        if peer and st.get(peer, {}).get('status') == resolved_status:
            res_fwd = rel_diff(r['tx'], st[peer]['rx'])
            res_rev = rel_diff(r['rx'], st[peer]['tx'])
            traffic_tx = max(r['tx'], st[peer]['rx'], 1.0)
            traffic_rx = max(r['rx'], st[peer]['tx'], 1.0)
            tol_pair_tx = min(0.12, max(TOL_PAIR_BASE, 5.0 / traffic_tx))
            tol_pair_rx = min(0.12, max(TOL_PAIR_BASE, 5.0 / traffic_rx))
            pair_comp_tx = conf_from_residual(res_fwd, tol_pair_tx)
            pair_comp_rx = conf_from_residual(res_rev, tol_pair_rx)
        else:
            pair_comp_tx = 0.55
            pair_comp_rx = 0.55

        router_imb = router_final_imbalance.get(router, 0.0)
        router_comp = conf_from_residual(router_imb, TOL_ROUTER)

        w_pair, w_router, w_status = 0.6, 0.3, 0.1
        base_tx = w_pair * pair_comp_tx + w_router * router_comp + w_status * status_comp
        base_rx = w_pair * pair_comp_rx + w_router * router_comp + w_status * status_comp

        # Tapered change penalty
        d_tx_rel = rel_diff(r['orig_tx'], r['tx'])
        d_rx_rel = rel_diff(r['orig_rx'], r['rx'])
        wt_tx = CHANGE_PENALTY_WEIGHT_SMALL if d_tx_rel < 0.15 else CHANGE_PENALTY_WEIGHT_LARGE
        wt_rx = CHANGE_PENALTY_WEIGHT_SMALL if d_rx_rel < 0.15 else CHANGE_PENALTY_WEIGHT_LARGE
        pen_tx = max(0.0, d_tx_rel - HARDENING_THRESHOLD)
        pen_rx = max(0.0, d_rx_rel - HARDENING_THRESHOLD)
        tx_conf = clamp(base_tx * (1.0 - wt_tx * pen_tx))
        rx_conf = clamp(base_rx * (1.0 - wt_rx * pen_rx))

        # Explicit penalties/bonuses
        scale_mag = r.get('router_scale_mag', 0.0)
        if scale_mag > SCALE_PENALTY_THRESH:
            # 0.03–0.05 depending on magnitude up to 0.5
            extra = 0.03 + 0.02 * min(1.0, (min(0.5, scale_mag) - SCALE_PENALTY_THRESH) / max(1e-9, 0.5 - SCALE_PENALTY_THRESH))
            tx_conf = clamp(tx_conf - extra)
            rx_conf = clamp(rx_conf - extra)

        # Penalty if consumed >70% cap across passes in that direction (up to 0.08)
        cap_frac_rx = r.get('cap_consumed_rx', 0.0)
        cap_frac_tx = r.get('cap_consumed_tx', 0.0)
        if cap_frac_tx > 0.7:
            tx_conf = clamp(tx_conf - min(0.08, 0.08 * (cap_frac_tx - 0.7) / 0.3))
        if cap_frac_rx > 0.7:
            rx_conf = clamp(rx_conf - min(0.08, 0.08 * (cap_frac_rx - 0.7) / 0.3))

        # Small stability bonus if untouched by router redistribution
        if not r.get('touched', False):
            tx_conf = clamp(tx_conf + TOUCH_BONUS)
            rx_conf = clamp(rx_conf + TOUCH_BONUS)

        if resolved_status == 'down':
            rx_conf = 0.9 if r['orig_rx'] <= TRAFFIC_EVIDENCE_MIN else 0.3
            tx_conf = 0.9 if r['orig_tx'] <= TRAFFIC_EVIDENCE_MIN else 0.3

        r['tx_conf'] = clamp(tx_conf)
        r['rx_conf'] = clamp(rx_conf)

        # Idle links slightly reduce status confidence if marked up
        if resolved_status == 'up':
            if r['rx'] <= TRAFFIC_EVIDENCE_MIN and r['tx'] <= TRAFFIC_EVIDENCE_MIN:
                r['status_conf'] = clamp(r['status_conf'] * 0.9)
        elif resolved_status == 'down':
            if r['rx'] > TRAFFIC_EVIDENCE_MIN or r['tx'] > TRAFFIC_EVIDENCE_MIN:
                r['status_conf'] = clamp(min(r['status_conf'], 0.3))

    # Assemble final output
    result: Dict[str, Dict[str, Tuple]] = {}
    for if_id, r in st.items():
        out: Dict[str, Tuple] = {}
        out['rx_rate'] = (r['orig_rx'], r['rx'], clamp(r['rx_conf']))
        out['tx_rate'] = (r['orig_tx'], r['tx'], clamp(r['tx_conf']))
        out['interface_status'] = (r['orig_status'], r['status'], clamp(r['status_conf']))
        out['connected_to'] = r['connected_to']
        out['local_router'] = r['local_router']
        out['remote_router'] = r['remote_router']
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