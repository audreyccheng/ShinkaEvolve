# EVOLVE-BLOCK-START
"""
Network telemetry repair algorithm that detects and corrects inconsistencies
in network interface telemetry data using topology relationships.

Takes interface telemetry data and detects/repairs inconsistencies based on
network invariants like link symmetry and flow conservation.
"""
from typing import Dict, Any, Tuple, List
from math import isfinite

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

    # Parameters
    HARDENING_THRESHOLD = 0.02          # τh measurement tolerance (~2%)
    TRAFFIC_EVIDENCE_MIN = 0.5          # Mbps: evidence threshold for link up
    EPS = 1e-9

    # Router rebalancing parameters
    MULT_PRE_CAP_FRAC = 0.15            # per-interface multiplicative pre-step cap (±15%)
    ALPHA_REF_IMBAL = 0.15              # imbalance reference for alpha tuning
    ALPHA_MIN, ALPHA_MAX = 0.25, 0.60   # bounds for multiplicative pre-step alpha
    STAGED_CAPS = [0.25, 0.35, 0.45]    # per-interface additive caps across passes
    ROUTER_TOTAL_DELTA_GUARD = 0.25     # ≤25% of average up-traffic as total router delta
    TIE_EPS_CONF = 0.05                 # tie-break threshold for direction selection

    # Weighting and reconciliation
    TOL_PAIR_BASE = HARDENING_THRESHOLD * 1.5
    TOL_ROUTER = HARDENING_THRESHOLD * 2.0
    PAIR_RECONCILE_FRACTION = 0.25      # move 25% toward each other
    PAIR_RECONCILE_CAP_FRAC = 0.20      # per-direction reconcile cap ≤ 20% of value

    def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
        if not isfinite(x):
            return lo
        return max(lo, min(hi, x))

    def rel_diff(a: float, b: float) -> float:
        denom = max(abs(a), abs(b), 1e-9)
        return abs(a - b) / denom

    def conf_two_slope(residual: float, tol: float) -> float:
        tol = max(tol, 1e-9)
        x = residual / tol
        # base decay: 1 at x=0, down to 0 at x>=5
        conf = 1.0 - min(1.0, x / 5.0)
        # extra penalty for very large residuals
        if x > 3.0:
            conf = conf - 0.1 * (x - 3.0) / 2.0
        return clamp(conf)

    def percentile(vals: List[float], q: float) -> float:
        arr = [v for v in vals if isfinite(v)]
        n = len(arr)
        if n == 0:
            return 0.0
        arr.sort()
        q = max(0.0, min(1.0, q))
        idx = int(q * (n - 1))
        return arr[idx]

    # Initialize structures
    result: Dict[str, Dict[str, Tuple]] = {}
    interim: Dict[str, Dict[str, Any]] = {}

    # Build connected pairs map
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

    # Initialize interim with originals and tracking fields
    for if_id, data in telemetry.items():
        rx0 = float(data.get('rx_rate', 0.0))
        tx0 = float(data.get('tx_rate', 0.0))
        interim[if_id] = {
            'rx': rx0, 'tx': tx0,
            'rx_conf': 1.0, 'tx_conf': 1.0,
            'status': data.get('interface_status', 'unknown'),
            'status_conf': 1.0,
            'connected_to': data.get('connected_to'),
            'local_router': data.get('local_router'),
            'remote_router': data.get('remote_router'),
            'orig_rx': rx0, 'orig_tx': tx0, 'orig_status': data.get('interface_status', 'unknown'),
            # edit tracking
            'edit_rx_abs': 0.0, 'edit_tx_abs': 0.0,
            'edit_rx_rel': 0.0, 'edit_tx_rel': 0.0,
            'cap_hit_rx': False, 'cap_hit_tx': False,
            'mult_scaled_rx': False, 'mult_scaled_tx': False,
            'scaled_rel_rx': 0.0, 'scaled_rel_tx': 0.0,
            'touched': False,
        }

    # Pair-level hardening (R3) and interface status consistency
    for a_id, b_id in pairs:
        a = telemetry[a_id]; b = telemetry[b_id]
        a_stat = a.get('interface_status', 'unknown')
        b_stat = b.get('interface_status', 'unknown')
        a_rx, a_tx = float(a.get('rx_rate', 0.0)), float(a.get('tx_rate', 0.0))
        b_rx, b_tx = float(b.get('rx_rate', 0.0)), float(b.get('tx_rate', 0.0))
        max_traffic = max(a_rx, a_tx, b_rx, b_tx)

        # Resolve link status using redundant evidence
        if a_stat == b_stat:
            resolved_status = a_stat
            status_conf = 0.95 if resolved_status in ('up', 'down') else 0.7
        else:
            if max_traffic > TRAFFIC_EVIDENCE_MIN:
                resolved_status = 'up'; status_conf = 0.85
            else:
                resolved_status = 'down'; status_conf = 0.75

        for ifid in (a_id, b_id):
            interim[ifid]['status'] = resolved_status
            interim[ifid]['status_conf'] = min(interim[ifid]['status_conf'], status_conf) if interim[ifid]['status_conf'] else status_conf

        if resolved_status == 'down':
            # Enforce zero traffic on down links with calibrated confidence
            for (ifid, rx0i, tx0i) in [(a_id, a_rx, a_tx), (b_id, b_rx, b_tx)]:
                interim[ifid]['rx'] = 0.0; interim[ifid]['tx'] = 0.0
                interim[ifid]['rx_conf'] = 0.9 if rx0i <= TRAFFIC_EVIDENCE_MIN else 0.3
                interim[ifid]['tx_conf'] = 0.9 if tx0i <= TRAFFIC_EVIDENCE_MIN else 0.3
            continue

        # Symmetry hardening (average if close; otherwise trust stronger peer reading)
        # Forward: a.tx ≈ b.rx
        d_fwd = rel_diff(a_tx, b_rx)
        if d_fwd <= HARDENING_THRESHOLD:
            v = 0.5 * (a_tx + b_rx); conf = clamp(1.0 - 0.5 * d_fwd)
        else:
            v = b_rx if abs(b_rx) > 0 else a_tx; conf = clamp(1.0 - d_fwd)
        interim[a_id]['tx'] = v; interim[b_id]['rx'] = v
        interim[a_id]['tx_conf'] = min(interim[a_id]['tx_conf'], conf)
        interim[b_id]['rx_conf'] = min(interim[b_id]['rx_conf'], conf)

        # Reverse: a.rx ≈ b.tx
        d_rev = rel_diff(a_rx, b_tx)
        if d_rev <= HARDENING_THRESHOLD:
            v2 = 0.5 * (a_rx + b_tx); conf2 = clamp(1.0 - 0.5 * d_rev)
        else:
            v2 = b_tx if abs(b_tx) > 0 else a_rx; conf2 = clamp(1.0 - d_rev)
        interim[a_id]['rx'] = v2; interim[b_id]['tx'] = v2
        interim[a_id]['rx_conf'] = min(interim[a_id]['rx_conf'], conf2)
        interim[b_id]['tx_conf'] = min(interim[b_id]['tx_conf'], conf2)

    # Enforce "down implies zero traffic" for unpaired down interfaces
    for if_id, r in interim.items():
        if if_id not in paired_ids and r.get('status') == 'down':
            rx0 = r['rx']; tx0 = r['tx']
            r['rx'] = 0.0; r['tx'] = 0.0
            r['rx_conf'] = 0.9 if rx0 <= TRAFFIC_EVIDENCE_MIN else 0.3
            r['tx_conf'] = 0.9 if tx0 <= TRAFFIC_EVIDENCE_MIN else 0.3

    # Router-level flow conservation (R1)
    # Build router->interfaces from topology if provided; else derive from telemetry metadata
    router_ifaces: Dict[str, List[str]] = {}
    if topology:
        router_ifaces = {r: [i for i in if_list if i in interim] for r, if_list in topology.items()}
    else:
        for if_id, data in telemetry.items():
            r = data.get('local_router')
            if r is not None:
                router_ifaces.setdefault(r, []).append(if_id)

    # Helper to compute pair residual for a given interface and direction
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

    # Uncertainty-weighted, capacity-capped redistribution using staged passes
    for router, if_list in router_ifaces.items():
        up_list = [i for i in if_list if interim[i]['status'] == 'up']
        if not up_list:
            continue

        def sums_and_conf():
            sum_tx = sum(max(0.0, interim[i]['tx']) for i in up_list)
            sum_rx = sum(max(0.0, interim[i]['rx']) for i in up_list)
            avg_tx_conf = sum(interim[i]['tx_conf'] for i in up_list) / max(1, len(up_list))
            avg_rx_conf = sum(interim[i]['rx_conf'] for i in up_list) / max(1, len(up_list))
            imb = rel_diff(sum_tx, sum_rx)
            return sum_tx, sum_rx, avg_tx_conf, avg_rx_conf, imb

        sum_tx, sum_rx, avg_tx_conf, avg_rx_conf, imbalance = sums_and_conf()
        if imbalance <= HARDENING_THRESHOLD * 2:
            pass  # small enough; skip router balancing
        else:
            # Decide which direction to scale: prefer less trusted; tie-break by larger absolute need
            need_rx = sum_tx - sum_rx  # need applied to RX to match TX
            need_tx = sum_rx - sum_tx  # need applied to TX to match RX
            if abs(avg_tx_conf - avg_rx_conf) < TIE_EPS_CONF:
                scale_rx = abs(need_rx) >= abs(need_tx)
            else:
                scale_rx = avg_tx_conf >= avg_rx_conf  # if TX is more trusted, scale RX
            # Pre-step multiplicative scaling (bounded)
            if scale_rx:
                s = (sum_tx / max(sum_rx, EPS))
            else:
                s = (sum_rx / max(sum_tx, EPS))
            s_bounded = max(0.5, min(2.0, s))
            alpha = max(ALPHA_MIN, min(ALPHA_MAX, (imbalance / ALPHA_REF_IMBAL) if ALPHA_REF_IMBAL > 0 else ALPHA_MIN))
            m = 1.0 + alpha * (s_bounded - 1.0)

            # Apply multiplicative pre-step with per-interface cap ±15%
            if m != 1.0:
                # Target only low-trust or high-residual interfaces in the scaled direction
                dir_key_pre = 'rx' if scale_rx else 'tx'
                confs = [interim[i][f'{dir_key_pre}_conf'] for i in up_list]
                conf_thresh = percentile(confs, 0.4)
                for i in up_list:
                    v = max(0.0, interim[i][dir_key_pre])
                    conf_i = interim[i][f'{dir_key_pre}_conf']
                    # Traffic-aware tolerance for pair residual
                    tol_pair_i = max(0.02, 2.5 / max(v, 1.0))
                    pair_resid_i = pair_residual_dir(i, dir_key_pre)
                    do_adjust = (conf_i <= conf_thresh) or (pair_resid_i > tol_pair_i)
                    if not do_adjust:
                        continue
                    allowed = MULT_PRE_CAP_FRAC * max(v, 1.0)
                    proposed = v * m
                    delta = proposed - v
                    # Clamp by per-interface cap
                    if delta > 0:
                        delta = min(delta, allowed)
                    else:
                        delta = max(delta, -allowed)
                    new_v = max(0.0, v + delta)
                    if abs(delta) > EPS:
                        if scale_rx:
                            interim[i]['rx'] = new_v
                            interim[i]['mult_scaled_rx'] = True
                            rel = rel_diff(v, new_v)
                            interim[i]['scaled_rel_rx'] = max(interim[i]['scaled_rel_rx'], rel)
                            interim[i]['edit_rx_abs'] += abs(delta)
                            interim[i]['edit_rx_rel'] = max(interim[i]['edit_rx_rel'], rel)
                            interim[i]['rx_conf'] = clamp(min(interim[i]['rx_conf'], 1.0 - min(1.0, 0.5 * imbalance + 0.5 * rel)))
                        else:
                            interim[i]['tx'] = new_v
                            interim[i]['mult_scaled_tx'] = True
                            rel = rel_diff(v, new_v)
                            interim[i]['scaled_rel_tx'] = max(interim[i]['scaled_rel_tx'], rel)
                            interim[i]['edit_tx_abs'] += abs(delta)
                            interim[i]['edit_tx_rel'] = max(interim[i]['edit_tx_rel'], rel)
                            interim[i]['tx_conf'] = clamp(min(interim[i]['tx_conf'], 1.0 - min(1.0, 0.5 * imbalance + 0.5 * rel)))
                        interim[i]['touched'] = True

            # Recompute sums after pre-step
            sum_tx, sum_rx, avg_tx_conf, avg_rx_conf, imbalance = sums_and_conf()
            if imbalance > HARDENING_THRESHOLD:
                # Target totals and initial need
                if scale_rx:
                    target_total = sum_tx
                    dir_key = 'rx'
                else:
                    target_total = sum_rx
                    dir_key = 'tx'

                # Prepare staged capacities and track remaining per interface
                base_vals = {i: (max(0.0, interim[i][dir_key])) for i in up_list}
                sum_dir = sum(base_vals.values())
                need = target_total - sum_dir

                # Elastic router total delta guard
                avg_up_traffic = 0.5 * (sum_tx + sum_rx)
                guard_frac = clamp(0.15 + 0.4 * imbalance + 0.4 * abs(avg_tx_conf - avg_rx_conf), 0.15, 0.35)
                router_delta_guard = guard_frac * avg_up_traffic
                router_delta_used = 0.0

                # Per-interface cap bookkeeping
                cap_total: Dict[str, float] = {i: 0.0 for i in up_list}
                cap_rem: Dict[str, float] = {i: 0.0 for i in up_list}
                consumed_frac_prev: Dict[str, float] = {i: 0.0 for i in up_list}

                def build_weights(pass_idx: int) -> Dict[str, float]:
                    nonlocal dir_key
                    sum_v = sum(max(0.0, interim[i][dir_key]) for i in up_list) + EPS
                    weights: Dict[str, float] = {}
                    for i in up_list:
                        v = max(0.0, interim[i][dir_key])
                        conf = interim[i][f'{dir_key}_conf']
                        pair_resid = pair_residual_dir(i, dir_key)
                        tol_pair_i = max(0.02, 2.5 / max(v, 1.0))
                        pair_term = min(2.0, pair_resid / max(tol_pair_i, 1e-9))
                        w = 0.6 * (1.0 - conf) + 0.25 * pair_term + 0.15 * (v / sum_v)
                        if v < 1.0:
                            w *= 0.5
                        if pass_idx > 0 and consumed_frac_prev.get(i, 0.0) > 0.7:
                            w *= 0.7
                        if pair_resid > 2.0 * tol_pair_i:
                            w += 0.1
                        # Ensure minimal positive weight
                        weights[i] = max(0.02, w)
                    return weights

                # Staged passes
                for pass_idx, cap_frac in enumerate(STAGED_CAPS):
                    # pass 3 only for low-confidence or low-traffic links
                    if pass_idx == 2:
                        eligible_subset = [i for i in up_list if (interim[i][f'{dir_key}_conf'] < 0.6 or interim[i][dir_key] < 5.0)]
                        if not eligible_subset:
                            continue
                        pass_list = eligible_subset
                    else:
                        pass_list = list(up_list)

                    # Update caps to include additional capacity this pass
                    for i in pass_list:
                        v_now = max(0.0, interim[i][dir_key])
                        cap_target = cap_frac * max(v_now, 1.0)
                        add_cap = max(0.0, cap_target - cap_total[i])
                        cap_total[i] += add_cap
                        cap_rem[i] += add_cap

                    # Recompute need for this pass
                    sum_dir = sum(max(0.0, interim[i][dir_key]) for i in up_list)
                    need = target_total - sum_dir
                    if abs(need) <= max(sum_dir, target_total, 1.0) * (HARDENING_THRESHOLD * 0.5):
                        break  # small residual remaining

                    # Choose eligible set based on need sign and remaining cap
                    if need > 0:
                        elig = [i for i in pass_list if cap_rem[i] > EPS]
                    else:
                        elig = [i for i in pass_list if cap_rem[i] > EPS]

                    # Freeze interfaces that consumed >80% of their cap if others remain
                    not_heavy = [i for i in elig if (cap_total[i] <= EPS) or (1.0 - cap_rem[i] / max(cap_total[i], EPS) <= 0.8)]
                    if not_heavy:
                        elig = not_heavy

                    if not elig:
                        continue

                    weights = build_weights(pass_idx)
                    # Single allocation sweep
                    sumW = sum(weights[i] for i in elig) + EPS
                    for i in elig:
                        if abs(need) <= EPS:
                            break
                        wshare = weights[i] / sumW
                        quota = need * wshare
                        # Clamp by remaining cap, and router guard
                        d = quota
                        d = max(-cap_rem[i], min(cap_rem[i], d))
                        # Apply router guard
                        remaining_guard = router_delta_guard - router_delta_used
                        if abs(d) > remaining_guard:
                            d = max(-remaining_guard, min(remaining_guard, d))
                        if abs(d) <= EPS:
                            continue

                        old_v = max(0.0, interim[i][dir_key])
                        new_v = max(0.0, old_v + d)
                        delta = new_v - old_v
                        # Update state
                        interim[i][dir_key] = new_v
                        cap_rem[i] = max(0.0, cap_rem[i] - abs(delta))
                        consumed_frac_prev[i] = 1.0 - (cap_rem[i] / max(cap_total[i], EPS))
                        router_delta_used += abs(delta)
                        need -= delta
                        interim[i]['touched'] = True
                        # Track edits
                        rel = rel_diff(old_v, new_v)
                        if dir_key == 'rx':
                            interim[i]['edit_rx_abs'] += abs(delta)
                            interim[i]['edit_rx_rel'] = max(interim[i]['edit_rx_rel'], rel)
                            if cap_rem[i] <= EPS * 10:
                                interim[i]['cap_hit_rx'] = True
                            # Confidence decay with imbalance and per-interface change
                            interim[i]['rx_conf'] = clamp(min(interim[i]['rx_conf'], 1.0 - min(1.0, imbalance + 0.5 * rel)))
                        else:
                            interim[i]['edit_tx_abs'] += abs(delta)
                            interim[i]['edit_tx_rel'] = max(interim[i]['edit_tx_rel'], rel)
                            if cap_rem[i] <= EPS * 10:
                                interim[i]['cap_hit_tx'] = True
                            interim[i]['tx_conf'] = clamp(min(interim[i]['tx_conf'], 1.0 - min(1.0, imbalance + 0.5 * rel)))

                        if router_delta_used >= router_delta_guard - EPS:
                            break
                    if router_delta_used >= router_delta_guard - EPS:
                        break

    # Targeted post-redistribution pair reconcile (only for touched pairs)
    for a_id, b_id in pairs:
        if interim[a_id].get('status') != 'up' or interim[b_id].get('status') != 'up':
            continue
        if not (interim[a_id]['touched'] or interim[b_id]['touched']):
            continue

        # Forward: a.tx vs b.rx
        a_tx_old, b_rx_old = interim[a_id]['tx'], interim[b_id]['rx']
        traffic_tx = max(a_tx_old, b_rx_old, 1.0)
        tol_pair_post = max(0.02, 2.5 / traffic_tx)
        res_fwd = rel_diff(a_tx_old, b_rx_old)
        if res_fwd > tol_pair_post:
            v_mid = 0.5 * (a_tx_old + b_rx_old)
            # Asymmetric alphas: move more on lower-confidence endpoint
            a_conf = interim[a_id]['tx_conf']
            b_conf = interim[b_id]['rx_conf']
            alpha_low, alpha_high = 0.40, 0.20
            alpha_a = alpha_low if a_conf < b_conf else alpha_high
            alpha_b = alpha_low if b_conf < a_conf else alpha_high
            move_a = alpha_a * (v_mid - a_tx_old)
            move_b = alpha_b * (v_mid - b_rx_old)
            cap_a = PAIR_RECONCILE_CAP_FRAC * max(a_tx_old, 1.0)
            cap_b = PAIR_RECONCILE_CAP_FRAC * max(b_rx_old, 1.0)
            move_a = max(-cap_a, min(cap_a, move_a))
            move_b = max(-cap_b, min(cap_b, move_b))
            a_tx_new = max(0.0, a_tx_old + move_a)
            b_rx_new = max(0.0, b_rx_old + move_b)
            if abs(a_tx_new - a_tx_old) > EPS:
                rel = rel_diff(a_tx_old, a_tx_new)
                interim[a_id]['tx'] = a_tx_new
                interim[a_id]['edit_tx_abs'] += abs(a_tx_new - a_tx_old)
                interim[a_id]['edit_tx_rel'] = max(interim[a_id]['edit_tx_rel'], rel)
                # Penalty proportional to residual
                penalty = 1.0 - 0.3 * min(1.0, res_fwd / max(TOL_PAIR_BASE, 1e-9))
                interim[a_id]['tx_conf'] = clamp(min(interim[a_id]['tx_conf'], penalty))
            if abs(b_rx_new - b_rx_old) > EPS:
                rel = rel_diff(b_rx_old, b_rx_new)
                interim[b_id]['rx'] = b_rx_new
                interim[b_id]['edit_rx_abs'] += abs(b_rx_new - b_rx_old)
                interim[b_id]['edit_rx_rel'] = max(interim[b_id]['edit_rx_rel'], rel)
                penalty = 1.0 - 0.3 * min(1.0, res_fwd / max(TOL_PAIR_BASE, 1e-9))
                interim[b_id]['rx_conf'] = clamp(min(interim[b_id]['rx_conf'], penalty))

        # Reverse: a.rx vs b.tx
        a_rx_old, b_tx_old = interim[a_id]['rx'], interim[b_id]['tx']
        traffic_rx = max(a_rx_old, b_tx_old, 1.0)
        tol_pair_post2 = max(0.02, 2.5 / traffic_rx)
        res_rev = rel_diff(a_rx_old, b_tx_old)
        if res_rev > tol_pair_post2:
            v_mid2 = 0.5 * (a_rx_old + b_tx_old)
            a_conf2 = interim[a_id]['rx_conf']
            b_conf2 = interim[b_id]['tx_conf']
            alpha_low, alpha_high = 0.40, 0.20
            alpha_a2 = alpha_low if a_conf2 < b_conf2 else alpha_high
            alpha_b2 = alpha_low if b_conf2 < a_conf2 else alpha_high
            move_a = alpha_a2 * (v_mid2 - a_rx_old)
            move_b = alpha_b2 * (v_mid2 - b_tx_old)
            cap_a = PAIR_RECONCILE_CAP_FRAC * max(a_rx_old, 1.0)
            cap_b = PAIR_RECONCILE_CAP_FRAC * max(b_tx_old, 1.0)
            move_a = max(-cap_a, min(cap_a, move_a))
            move_b = max(-cap_b, min(cap_b, move_b))
            a_rx_new = max(0.0, a_rx_old + move_a)
            b_tx_new = max(0.0, b_tx_old + move_b)
            if abs(a_rx_new - a_rx_old) > EPS:
                rel = rel_diff(a_rx_old, a_rx_new)
                interim[a_id]['rx'] = a_rx_new
                interim[a_id]['edit_rx_abs'] += abs(a_rx_new - a_rx_old)
                interim[a_id]['edit_rx_rel'] = max(interim[a_id]['edit_rx_rel'], rel)
                penalty = 1.0 - 0.3 * min(1.0, res_rev / max(TOL_PAIR_BASE, 1e-9))
                interim[a_id]['rx_conf'] = clamp(min(interim[a_id]['rx_conf'], penalty))
            if abs(b_tx_new - b_tx_old) > EPS:
                rel = rel_diff(b_tx_old, b_tx_new)
                interim[b_id]['tx'] = b_tx_new
                interim[b_id]['edit_tx_abs'] += abs(b_tx_new - b_tx_old)
                interim[b_id]['edit_tx_rel'] = max(interim[b_id]['edit_tx_rel'], rel)
                penalty = 1.0 - 0.3 * min(1.0, res_rev / max(TOL_PAIR_BASE, 1e-9))
                interim[b_id]['tx_conf'] = clamp(min(interim[b_id]['tx_conf'], penalty))

    # Final confidence calibration based on post-repair invariants
    # Compute original per-router imbalance from original telemetry (status up)
    router_orig_imbalance: Dict[str, float] = {}
    for router, if_list in router_ifaces.items():
        up_ifaces0 = [i for i in if_list if telemetry.get(i, {}).get('interface_status') == 'up']
        if not up_ifaces0:
            router_orig_imbalance[router] = 0.0
            continue
        sum_tx0 = sum(max(0.0, interim[i]['orig_tx']) for i in up_ifaces0)
        sum_rx0 = sum(max(0.0, interim[i]['orig_rx']) for i in up_ifaces0)
        router_orig_imbalance[router] = rel_diff(sum_tx0, sum_rx0)

    # Compute per-router imbalance residuals with current interim
    router_final_imbalance: Dict[str, float] = {}
    for router, if_list in router_ifaces.items():
        up_ifaces = [i for i in if_list if i in interim and interim[i].get('status') == 'up']
        if not up_ifaces:
            router_final_imbalance[router] = 0.0
            continue
        sum_tx = sum(max(0.0, interim[i]['tx']) for i in up_ifaces)
        sum_rx = sum(max(0.0, interim[i]['rx']) for i in up_ifaces)
        router_final_imbalance[router] = rel_diff(sum_tx, sum_rx)

    # Confidence weights
    w_pair, w_router, w_status = 0.6, 0.3, 0.1

    for if_id, r in interim.items():
        router = r.get('local_router')
        peer = peer_of.get(if_id)
        status_comp = clamp(r.get('status_conf', 0.8))
        resolved_status = r.get('status', 'unknown')

        # Pair-based components using rate-aware tolerances
        if peer and interim.get(peer, {}).get('status') == resolved_status:
            # Forward: tx vs peer.rx
            res_fwd = rel_diff(r['tx'], interim[peer]['rx'])
            traffic_tx = max(r['tx'], interim[peer]['rx'], 1.0)
            tol_pair_tx = min(0.12, max(TOL_PAIR_BASE, 5.0 / traffic_tx))
            pair_comp_tx = conf_two_slope(res_fwd, tol_pair_tx)
            # Reverse: rx vs peer.tx
            res_rev = rel_diff(r['rx'], interim[peer]['tx'])
            traffic_rx = max(r['rx'], interim[peer]['tx'], 1.0)
            tol_pair_rx = min(0.12, max(TOL_PAIR_BASE, 5.0 / traffic_rx))
            pair_comp_rx = conf_two_slope(res_rev, tol_pair_rx)
        else:
            pair_comp_tx = 0.55
            pair_comp_rx = 0.55

        router_imb = router_final_imbalance.get(router, 0.0)
        router_comp = conf_two_slope(router_imb, TOL_ROUTER)

        base_tx_conf = w_pair * pair_comp_tx + w_router * router_comp + w_status * status_comp
        base_rx_conf = w_pair * pair_comp_rx + w_router * router_comp + w_status * status_comp

        # Edit-aware calibration and penalties
        # No-edit bonus; larger edits => multiplicative penalty
        edit_tx_rel = r.get('edit_tx_rel', 0.0)
        edit_rx_rel = r.get('edit_rx_rel', 0.0)
        cap_hit_tx = r.get('cap_hit_tx', False)
        cap_hit_rx = r.get('cap_hit_rx', False)
        scaled_rel_tx = r.get('scaled_rel_tx', 0.0)
        scaled_rel_rx = r.get('scaled_rel_rx', 0.0)

        def finalize_conf(base: float, edit_rel: float, cap_hit: bool, scaled_rel: float, orig_val: float, new_val: float) -> float:
            # Two-slope change penalty
            pen_factor = max(0.0, 1.0 - 0.6 * max(0.0, edit_rel - HARDENING_THRESHOLD))
            conf = clamp(base * pen_factor)
            if cap_hit:
                conf *= 0.9
            if scaled_rel > 0.0:
                conf *= (1.0 - min(0.15, 0.3 * scaled_rel))
            # No-edit bonus
            if rel_diff(orig_val, new_val) <= 1e-3:
                conf = clamp(conf + 0.05)
            return clamp(conf)

        final_tx_conf = finalize_conf(base_tx_conf, edit_tx_rel, cap_hit_tx, scaled_rel_tx, r['orig_tx'], r['tx'])
        final_rx_conf = finalize_conf(base_rx_conf, edit_rx_rel, cap_hit_rx, scaled_rel_rx, r['orig_rx'], r['rx'])

        # Improvement bonus: reward large pair residual and router imbalance reductions
        impr_bonus = 0.0
        if peer:
            res_fwd0 = rel_diff(r['orig_tx'], interim[peer]['orig_rx'])
            res_rev0 = rel_diff(r['orig_rx'], interim[peer]['orig_tx'])
            res_fwd1 = rel_diff(r['tx'], interim[peer]['rx'])
            res_rev1 = rel_diff(r['rx'], interim[peer]['tx'])
            impr_fwd = (res_fwd0 > 0.0) and (res_fwd1 <= 0.5 * res_fwd0)
            impr_rev = (res_rev0 > 0.0) and (res_rev1 <= 0.5 * res_rev0)
        else:
            impr_fwd = False
            impr_rev = False
        orig_imb = router_orig_imbalance.get(router, 0.0)
        final_imb = router_final_imbalance.get(router, 0.0)
        router_impr = (orig_imb > 0.0 and final_imb <= 0.7 * orig_imb)
        if (impr_fwd and impr_rev) and router_impr:
            impr_bonus = 0.05
        elif (impr_fwd and impr_rev) or router_impr:
            impr_bonus = 0.03
        final_tx_conf = clamp(final_tx_conf + impr_bonus)
        final_rx_conf = clamp(final_rx_conf + impr_bonus)

        if resolved_status == 'down':
            final_rx_conf = 0.9 if r['orig_rx'] <= TRAFFIC_EVIDENCE_MIN else 0.3
            final_tx_conf = 0.9 if r['orig_tx'] <= TRAFFIC_EVIDENCE_MIN else 0.3

        r['tx_conf'] = clamp(final_tx_conf)
        r['rx_conf'] = clamp(final_rx_conf)

        # Status subtle calibration
        if resolved_status == 'up':
            if r['rx'] <= TRAFFIC_EVIDENCE_MIN and r['tx'] <= TRAFFIC_EVIDENCE_MIN:
                r['status_conf'] = clamp(r['status_conf'] * 0.9)
        elif resolved_status == 'down':
            if r['rx'] > TRAFFIC_EVIDENCE_MIN or r['tx'] > TRAFFIC_EVIDENCE_MIN:
                r['status_conf'] = clamp(min(r['status_conf'], 0.3))

    # Assemble final result
    result: Dict[str, Dict[str, Tuple]] = {}
    for if_id, data in telemetry.items():
        r = interim[if_id]
        repaired_data: Dict[str, Tuple] = {}
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