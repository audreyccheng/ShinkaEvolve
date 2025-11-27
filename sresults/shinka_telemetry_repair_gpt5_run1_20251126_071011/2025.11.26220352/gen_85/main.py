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
    # Max fractional per-interface adjustment during router redistribution
    MAX_ROUTER_ADJ_FRAC = 0.35
    # Pair reconciliation strength toward midpoint after router pass
    PAIR_RECONCILE_ALPHA = 0.30
    # Asymmetric pair reconciliation alphas (lower-confidence endpoint moves more)
    PAIR_ALPHA_LOW = 0.40
    PAIR_ALPHA_HIGH = 0.20
    # Secondary router rebalancing cap (gentler than primary)
    SECONDARY_ROUTER_ADJ_FRAC = 0.15

    # Confidence/weighting helpers and guards
    TIE_EPS_CONF = 0.05
    TOL_PAIR_BASE = HARDENING_THRESHOLD * 1.5
    TOL_ROUTER = HARDENING_THRESHOLD * 2.0
    ROUTER_TOTAL_DELTA_GUARD = 0.25  # ≤25% of average up-traffic as total router delta
    EPS = 1e-9

    def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
        return max(lo, min(hi, x))

    def rel_diff(a: float, b: float) -> float:
        denom = max(abs(a), abs(b), 1e-9)
        return abs(a - b) / denom

    def conf_from_residual(residual: float, tol: float) -> float:
        # Two-slope residual-to-confidence:
        # High near 0 residual; decays with residual/tolerance, with extra penalty for very large residuals.
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
            # Use ordered tuple to avoid duplicates
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
            # Keep originals for output tuples
            'orig_rx': float(data.get('rx_rate', 0.0)),
            'orig_tx': float(data.get('tx_rate', 0.0)),
            'orig_status': data.get('interface_status', 'unknown'),
            # Edit tracking for calibration
            'edit_rx_rel': 0.0,
            'edit_tx_rel': 0.0,
            'cap_hit_rx': False,
            'cap_hit_tx': False,
            'touched': False,
        }

    # Precompute baseline router imbalance and initial pair residuals for confidence bonuses
    router_ifaces_pre: Dict[str, List[str]] = {}
    if topology:
        router_ifaces_pre = {r: [i for i in if_list if i in interim] for r, if_list in topology.items()}
    else:
        # Derive from telemetry if topology absent
        for if_id2, data2 in telemetry.items():
            rtr2 = data2.get('local_router')
            if rtr2 is not None:
                router_ifaces_pre.setdefault(rtr2, []).append(if_id2)
    router_imbalance_before: Dict[str, float] = {}
    for router0, if_list0 in router_ifaces_pre.items():
        up0 = [i for i in if_list0 if interim[i].get('status') == 'up']
        if not up0:
            router_imbalance_before[router0] = 0.0
            continue
        sum_tx0 = sum(max(0.0, interim[i]['tx']) for i in up0)
        sum_rx0 = sum(max(0.0, interim[i]['rx']) for i in up0)
        router_imbalance_before[router0] = rel_diff(sum_tx0, sum_rx0)
    pre_pair_residuals: Dict[str, Dict[str, float]] = {}
    for if_id2, _data2 in telemetry.items():
        peer2 = peer_of.get(if_id2)
        res_tx0 = 0.0
        res_rx0 = 0.0
        if peer2 and interim[if_id2].get('status') == 'up' and interim.get(peer2, {}).get('status') == 'up':
            res_tx0 = rel_diff(interim[if_id2]['tx'], interim[peer2]['rx'])
            res_rx0 = rel_diff(interim[if_id2]['rx'], interim[peer2]['tx'])
        pre_pair_residuals[if_id2] = {'tx': res_tx0, 'rx': res_rx0}

    # Track router scaling magnitude and guard usage for calibration
    router_scale_mag: Dict[str, float] = {}
    router_delta_frac: Dict[str, float] = {}

    # Pair-level hardening using link symmetry (R3) and interface consistency
    for a_id, b_id in pairs:
        a = telemetry[a_id]
        b = telemetry[b_id]
        a_stat = a.get('interface_status', 'unknown')
        b_stat = b.get('interface_status', 'unknown')
        a_rx, a_tx = float(a.get('rx_rate', 0.0)), float(a.get('tx_rate', 0.0))
        b_rx, b_tx = float(b.get('rx_rate', 0.0)), float(b.get('tx_rate', 0.0))
        max_traffic = max(a_rx, a_tx, b_rx, b_tx)

        # Resolve interface status consistency across the link
        if a_stat == b_stat:
            resolved_status = a_stat
            status_conf = 0.95 if resolved_status in ('up', 'down') else 0.7
        else:
            # Use traffic evidence: if there is noticeable traffic, link must be up
            if max_traffic > TRAFFIC_EVIDENCE_MIN:
                resolved_status = 'up'
                status_conf = 0.85
            else:
                resolved_status = 'down'
                status_conf = 0.75

        # Apply status to both ends
        interim[a_id]['status'] = resolved_status
        interim[b_id]['status'] = resolved_status
        interim[a_id]['status_conf'] = min(interim[a_id]['status_conf'], status_conf) if interim[a_id]['status_conf'] else status_conf
        interim[b_id]['status_conf'] = min(interim[b_id]['status_conf'], status_conf) if interim[b_id]['status_conf'] else status_conf

        if resolved_status == 'down':
            # Down interfaces cannot send or receive
            # Confidence is high if original values were already near zero, lower otherwise.
            for ifid, rx0, tx0 in [(a_id, a_rx, a_tx), (b_id, b_rx, b_tx)]:
                interim[ifid]['rx'] = 0.0
                interim[ifid]['tx'] = 0.0
                interim[ifid]['rx_conf'] = clamp(0.9 if rx0 <= TRAFFIC_EVIDENCE_MIN else 0.3)
                interim[ifid]['tx_conf'] = clamp(0.9 if tx0 <= TRAFFIC_EVIDENCE_MIN else 0.3)
            continue  # No need to harden rates if link is down

        # Link is up: harden both directions using symmetry
        # Forward direction: a.tx should match b.rx
        d_fwd = rel_diff(a_tx, b_rx)
        if d_fwd <= HARDENING_THRESHOLD:
            v = 0.5 * (a_tx + b_rx)
            conf = clamp(1.0 - 0.5 * d_fwd)  # near 1 when very close
        else:
            # Choose peer's counterpart as stronger signal
            v = b_rx if abs(b_rx) > 0 else a_tx
            conf = clamp(1.0 - d_fwd)  # lower confidence for larger violation
        if v != interim[a_id]['tx']:
            interim[a_id]['edit_tx_rel'] = max(interim[a_id]['edit_tx_rel'], rel_diff(interim[a_id]['tx'], v))
            interim[a_id]['touched'] = True
        if v != interim[b_id]['rx']:
            interim[b_id]['edit_rx_rel'] = max(interim[b_id]['edit_rx_rel'], rel_diff(interim[b_id]['rx'], v))
            interim[b_id]['touched'] = True
        interim[a_id]['tx'] = v
        interim[b_id]['rx'] = v
        interim[a_id]['tx_conf'] = min(interim[a_id]['tx_conf'], conf)
        interim[b_id]['rx_conf'] = min(interim[b_id]['rx_conf'], conf)

        # Reverse direction: a.rx should match b.tx
        d_rev = rel_diff(a_rx, b_tx)
        if d_rev <= HARDENING_THRESHOLD:
            v2 = 0.5 * (a_rx + b_tx)
            conf2 = clamp(1.0 - 0.5 * d_rev)
        else:
            v2 = b_tx if abs(b_tx) > 0 else a_rx
            conf2 = clamp(1.0 - d_rev)
        if v2 != interim[a_id]['rx']:
            interim[a_id]['edit_rx_rel'] = max(interim[a_id]['edit_rx_rel'], rel_diff(interim[a_id]['rx'], v2))
            interim[a_id]['touched'] = True
        if v2 != interim[b_id]['tx']:
            interim[b_id]['edit_tx_rel'] = max(interim[b_id]['edit_tx_rel'], rel_diff(interim[b_id]['tx'], v2))
            interim[b_id]['touched'] = True
        interim[a_id]['rx'] = v2
        interim[b_id]['tx'] = v2
        interim[a_id]['rx_conf'] = min(interim[a_id]['rx_conf'], conf2)
        interim[b_id]['tx_conf'] = min(interim[b_id]['tx_conf'], conf2)

    # Enforce "down implies zero traffic" also for unpaired interfaces
    for if_id, r in interim.items():
        if if_id not in paired_ids and r.get('status') == 'down':
            rx0 = r['rx']
            tx0 = r['tx']
            r['rx'] = 0.0
            r['tx'] = 0.0
            r['rx_conf'] = clamp(0.9 if rx0 <= TRAFFIC_EVIDENCE_MIN else 0.3)
            r['tx_conf'] = clamp(0.9 if tx0 <= TRAFFIC_EVIDENCE_MIN else 0.3)

    # Build router to interfaces map (use provided topology if available, else derive from telemetry)
    router_ifaces: Dict[str, List[str]] = {}
    if topology:
        router_ifaces = {r: [i for i in if_list if i in interim] for r, if_list in topology.items()}
    else:
        # If topology not provided, derive from telemetry metadata
        # Note: Topology helps flow conservation; we derive a best-effort map when absent.
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

    def router_redistribute(max_adj_frac: float) -> None:
        # Uncertainty-weighted, capacity-capped redistribution towards flow conservation
        for router, if_list in router_ifaces.items():
            interfaces = [i for i in if_list if i in interim]
            if not interfaces:
                continue

            # Compute sums over "up" interfaces
            sum_tx = 0.0
            sum_rx = 0.0
            tx_conf_acc = 0.0
            rx_conf_acc = 0.0
            up_list = []
            for i in interfaces:
                if interim[i]['status'] == 'up':
                    up_list.append(i)
                    sum_tx += max(0.0, interim[i]['tx'])
                    sum_rx += max(0.0, interim[i]['rx'])
                    tx_conf_acc += interim[i]['tx_conf']
                    rx_conf_acc += interim[i]['rx_conf']

            if not up_list:
                continue

            # Evaluate flow imbalance
            imbalance = rel_diff(sum_tx, sum_rx)
            if imbalance <= HARDENING_THRESHOLD * 2:
                # Within tolerance; no router-level scaling needed
                continue

            avg_tx_conf = tx_conf_acc / max(1, len(up_list))
            avg_rx_conf = rx_conf_acc / max(1, len(up_list))

            # Decide which direction to scale: scale the less trusted direction; tie-break by larger absolute need
            need_rx = sum_tx - sum_rx  # need applied to RX to match TX
            need_tx = -need_rx         # need applied to TX to match RX
            if abs(avg_tx_conf - avg_rx_conf) < TIE_EPS_CONF:
                scale_rx = abs(need_rx) >= abs(need_tx)
            else:
                scale_rx = avg_tx_conf >= avg_rx_conf  # if TX more trusted, scale RX to match TX

            if scale_rx and sum_rx > 0.0:
                s = sum_tx / sum_rx
            elif (not scale_rx) and sum_tx > 0.0:
                s = sum_rx / sum_tx
            else:
                s = 1.0

            # Bound scaling magnitude and store for calibration
            s_bounded = max(0.5, min(2.0, s))
            router_scale_mag[router] = max(router_scale_mag.get(router, 0.0), abs(1.0 - s_bounded))

            # Target totals and working direction
            if scale_rx:
                sum_old = sum(max(0.0, interim[i]['rx']) for i in up_list)
                target_total = sum_tx
                dir_key = 'rx'
            else:
                sum_old = sum(max(0.0, interim[i]['tx']) for i in up_list)
                target_total = sum_rx
                dir_key = 'tx'

            need = target_total - sum_old
            if abs(need) <= max(sum_old, target_total, 1.0) * (HARDENING_THRESHOLD * 0.5):
                # Tiny residual; skip redistribution
                continue

            # Initialize per-interface caps for both signs and keep totals to track consumption
            caps_pos: Dict[str, float] = {}
            caps_neg: Dict[str, float] = {}
            caps_total_pos: Dict[str, float] = {}
            caps_total_neg: Dict[str, float] = {}
            for i in up_list:
                v = max(0.0, interim[i][dir_key])
                cap = max_adj_frac * max(v, 1.0)
                caps_pos[i] = cap
                caps_neg[i] = cap
                caps_total_pos[i] = cap
                caps_total_neg[i] = cap

            # Router total delta guard to avoid over-adjustment
            avg_up_traffic = 0.5 * (sum_tx + sum_rx)
            router_delta_guard = ROUTER_TOTAL_DELTA_GUARD * avg_up_traffic
            router_delta_used = 0.0

            # Iterative allocation with capacity clipping and router guard
            for pass_idx in range(2):
                if abs(need) <= EPS:
                    break
                if need > 0:
                    elig = [i for i in up_list if caps_pos[i] > EPS]
                else:
                    elig = [i for i in up_list if caps_neg[i] > EPS]
                if not elig:
                    break

                # Build traffic-aware weights this pass
                sum_v_for_norm = sum(max(0.0, interim[i][dir_key]) for i in elig) + EPS
                weights: Dict[str, float] = {}
                for i in elig:
                    conf = float(interim[i][f'{dir_key}_conf'])
                    v = max(0.0, float(interim[i][dir_key]))
                    # Pair residual term scaled by rate-aware tolerance
                    tol_pair = max(0.02, 2.5 / max(v, 1.0))
                    pair_resid = pair_residual_dir(i, dir_key)
                    resid_term = min(2.0, pair_resid / max(tol_pair, 1e-9))
                    w = 0.6 * (1.0 - conf) + 0.25 * resid_term + 0.15 * (v / sum_v_for_norm)
                    # Tiny-link protection
                    if v < 1.0:
                        w *= 0.5
                    # Reduce weight on later pass if >70% of sign-cap consumed
                    if need > 0:
                        consumed_frac = 1.0 - (caps_pos[i] / max(caps_total_pos[i], EPS))
                    else:
                        consumed_frac = 1.0 - (caps_neg[i] / max(caps_total_neg[i], EPS))
                    if pass_idx >= 1 and consumed_frac > 0.7:
                        w *= 0.7
                    # Boost when residual is very large
                    if pair_resid > 2.0 * tol_pair:
                        w += 0.1
                    weights[i] = max(0.02, w)

                sumW = sum(weights[i] for i in elig)
                if sumW <= EPS:
                    break

                for i in elig:
                    if abs(need) <= EPS or router_delta_used >= router_delta_guard - EPS:
                        break
                    quota = need * (weights[i] / sumW)
                    if need > 0:
                        d = min(max(0.0, quota), caps_pos[i])
                        caps_pos[i] -= d
                        cap_depleted = caps_pos[i] <= EPS * 10
                    else:
                        d = max(min(0.0, quota), -caps_neg[i])
                        caps_neg[i] -= -d
                        cap_depleted = caps_neg[i] <= EPS * 10

                    # Apply router guard
                    remaining_guard = router_delta_guard - router_delta_used
                    if abs(d) > remaining_guard:
                        d = max(-remaining_guard, min(remaining_guard, d))

                    if abs(d) <= EPS:
                        continue

                    old_v = max(0.0, interim[i][dir_key])
                    new_v = max(0.0, old_v + d)
                    delta_rel = rel_diff(old_v, new_v)
                    interim[i][dir_key] = new_v
                    # Confidence drops with global imbalance, scaling magnitude and per-interface change
                    penalty = 1.0 - min(1.0, imbalance + 0.5 * delta_rel + abs(1.0 - s_bounded) * 0.5)
                    if dir_key == 'rx':
                        interim[i]['rx_conf'] = clamp(min(interim[i]['rx_conf'], penalty))
                        interim[i]['edit_rx_rel'] = max(interim[i]['edit_rx_rel'], delta_rel)
                        if cap_depleted:
                            interim[i]['cap_hit_rx'] = True
                    else:
                        interim[i]['tx_conf'] = clamp(min(interim[i]['tx_conf'], penalty))
                        interim[i]['edit_tx_rel'] = max(interim[i]['edit_tx_rel'], delta_rel)
                        if cap_depleted:
                            interim[i]['cap_hit_tx'] = True
                    interim[i]['touched'] = True

                    router_delta_used += abs(d)
                    need -= d
                if router_delta_used >= router_delta_guard - EPS:
                    break

            # Track how much of router guard we used for confidence calibration
            if router_delta_guard > EPS:
                used_frac = clamp(router_delta_used / router_delta_guard, 0.0, 1.0)
                router_delta_frac[router] = max(router_delta_frac.get(router, 0.0), used_frac)

    # Primary router redistribution
    router_redistribute(MAX_ROUTER_ADJ_FRAC)

    # Limited pair-symmetry reconciliation after router redistribution (tolerance-gated with caps)
    for a_id, b_id in pairs:
        if a_id not in interim or b_id not in interim:
            continue
        if interim[a_id].get('status') != 'up' or interim[b_id].get('status') != 'up':
            continue

        # Two micro-iterations with asymmetric, residual-proportional moves and per-direction caps
        # Forward direction caps
        a_tx_cap_rem = 0.20 * max(interim[a_id]['tx'], 1.0)
        b_rx_cap_rem = 0.20 * max(interim[b_id]['rx'], 1.0)
        # Reverse direction caps
        a_rx_cap_rem = 0.20 * max(interim[a_id]['rx'], 1.0)
        b_tx_cap_rem = 0.20 * max(interim[b_id]['tx'], 1.0)

        for _iter in range(2):
            # Forward direction: a.tx vs b.rx
            a_tx_cur = interim[a_id]['tx']; b_rx_cur = interim[b_id]['rx']
            traffic_tx = max(a_tx_cur, b_rx_cur, 1.0)
            tol_pair_post = max(0.02, 2.5 / traffic_tx)
            res_fwd = rel_diff(a_tx_cur, b_rx_cur)
            if res_fwd > tol_pair_post and (a_tx_cap_rem > EPS or b_rx_cap_rem > EPS):
                v_mid = 0.5 * (a_tx_cur + b_rx_cur)
                a_conf = interim[a_id]['tx_conf']; b_conf = interim[b_id]['rx_conf']
                alpha_a = PAIR_ALPHA_LOW if a_conf < b_conf else PAIR_ALPHA_HIGH
                alpha_b = PAIR_ALPHA_LOW if b_conf < a_conf else PAIR_ALPHA_HIGH
                scale = min(2.0, res_fwd / max(tol_pair_post, 1e-9))
                move_a = alpha_a * scale * (v_mid - a_tx_cur)
                move_b = alpha_b * scale * (v_mid - b_rx_cur)
                # Clamp by remaining caps
                if move_a >= 0:
                    move_a = min(move_a, a_tx_cap_rem)
                else:
                    move_a = max(move_a, -a_tx_cap_rem)
                if move_b >= 0:
                    move_b = min(move_b, b_rx_cap_rem)
                else:
                    move_b = max(move_b, -b_rx_cap_rem)
                a_tx_new = max(0.0, a_tx_cur + move_a)
                b_rx_new = max(0.0, b_rx_cur + move_b)
                if a_tx_new != a_tx_cur:
                    drel = rel_diff(a_tx_cur, a_tx_new)
                    interim[a_id]['tx'] = a_tx_new
                    interim[a_id]['tx_conf'] = clamp(interim[a_id]['tx_conf'] * (1.0 - 0.3 * min(1.0, res_fwd / max(tol_pair_post, 1e-9))))
                    interim[a_id]['edit_tx_rel'] = max(interim[a_id]['edit_tx_rel'], drel)
                    interim[a_id]['touched'] = True
                    a_tx_cap_rem = max(0.0, a_tx_cap_rem - abs(move_a))
                if b_rx_new != b_rx_cur:
                    drel = rel_diff(b_rx_cur, b_rx_new)
                    interim[b_id]['rx'] = b_rx_new
                    interim[b_id]['rx_conf'] = clamp(interim[b_id]['rx_conf'] * (1.0 - 0.3 * min(1.0, res_fwd / max(tol_pair_post, 1e-9))))
                    interim[b_id]['edit_rx_rel'] = max(interim[b_id]['edit_rx_rel'], drel)
                    interim[b_id]['touched'] = True
                    b_rx_cap_rem = max(0.0, b_rx_cap_rem - abs(move_b))
                # Early stop if within tolerance
                a_tx_cur = interim[a_id]['tx']; b_rx_cur = interim[b_id]['rx']
                if rel_diff(a_tx_cur, b_rx_cur) <= tol_pair_post:
                    pass

            # Reverse direction: a.rx vs b.tx
            a_rx_cur = interim[a_id]['rx']; b_tx_cur = interim[b_id]['tx']
            traffic_rx = max(a_rx_cur, b_tx_cur, 1.0)
            tol_pair_post2 = max(0.02, 2.5 / traffic_rx)
            res_rev = rel_diff(a_rx_cur, b_tx_cur)
            if res_rev > tol_pair_post2 and (a_rx_cap_rem > EPS or b_tx_cap_rem > EPS):
                v_mid2 = 0.5 * (a_rx_cur + b_tx_cur)
                a_conf2 = interim[a_id]['rx_conf']; b_conf2 = interim[b_id]['tx_conf']
                alpha_a2 = PAIR_ALPHA_LOW if a_conf2 < b_conf2 else PAIR_ALPHA_HIGH
                alpha_b2 = PAIR_ALPHA_LOW if b_conf2 < a_conf2 else PAIR_ALPHA_HIGH
                scale2 = min(2.0, res_rev / max(tol_pair_post2, 1e-9))
                move_a2 = alpha_a2 * scale2 * (v_mid2 - a_rx_cur)
                move_b2 = alpha_b2 * scale2 * (v_mid2 - b_tx_cur)
                if move_a2 >= 0:
                    move_a2 = min(move_a2, a_rx_cap_rem)
                else:
                    move_a2 = max(move_a2, -a_rx_cap_rem)
                if move_b2 >= 0:
                    move_b2 = min(move_b2, b_tx_cap_rem)
                else:
                    move_b2 = max(move_b2, -b_tx_cap_rem)
                a_rx_new = max(0.0, a_rx_cur + move_a2)
                b_tx_new = max(0.0, b_tx_cur + move_b2)
                if a_rx_new != a_rx_cur:
                    drel = rel_diff(a_rx_cur, a_rx_new)
                    interim[a_id]['rx'] = a_rx_new
                    interim[a_id]['rx_conf'] = clamp(interim[a_id]['rx_conf'] * (1.0 - 0.3 * min(1.0, res_rev / max(tol_pair_post2, 1e-9))))
                    interim[a_id]['edit_rx_rel'] = max(interim[a_id]['edit_rx_rel'], drel)
                    interim[a_id]['touched'] = True
                    a_rx_cap_rem = max(0.0, a_rx_cap_rem - abs(move_a2))
                if b_tx_new != b_tx_cur:
                    drel = rel_diff(b_tx_cur, b_tx_new)
                    interim[b_id]['tx'] = b_tx_new
                    interim[b_id]['tx_conf'] = clamp(interim[b_id]['tx_conf'] * (1.0 - 0.3 * min(1.0, res_rev / max(tol_pair_post2, 1e-9))))
                    interim[b_id]['edit_tx_rel'] = max(interim[b_id]['edit_tx_rel'], drel)
                    interim[b_id]['touched'] = True
                    b_tx_cap_rem = max(0.0, b_tx_cap_rem - abs(move_b2))
            # Early exit if both within tolerance
            if rel_diff(interim[a_id]['tx'], interim[b_id]['rx']) <= max(0.02, 2.5 / max(interim[a_id]['tx'], interim[b_id]['rx'], 1.0)) and \
               rel_diff(interim[a_id]['rx'], interim[b_id]['tx']) <= max(0.02, 2.5 / max(interim[a_id]['rx'], interim[b_id]['tx'], 1.0)):
                break

    # Secondary (gentle) router redistribution to restore flow conservation after pair reconcile
    router_redistribute(SECONDARY_ROUTER_ADJ_FRAC)

    # Final confidence calibration based on post-repair invariants
    # Compute per-router imbalance residuals
    router_final_imbalance: Dict[str, float] = {}
    for router, if_list in router_ifaces.items():
        # only consider interfaces that are in interim and up
        up_ifaces = [i for i in if_list if i in interim and interim[i].get('status') == 'up']
        if not up_ifaces:
            router_final_imbalance[router] = 0.0
            continue
        sum_tx = sum(max(0.0, interim[i]['tx']) for i in up_ifaces)
        sum_rx = sum(max(0.0, interim[i]['rx']) for i in up_ifaces)
        router_final_imbalance[router] = rel_diff(sum_tx, sum_rx)

    # Weights and tolerances for confidence components
    w_pair, w_router, w_status = 0.6, 0.3, 0.1

    def finalize_conf(base: float, edit_rel: float, cap_hit: bool, orig_val: float, new_val: float) -> float:
        # Multiplicative penalty for large edits; softer near threshold
        pen_factor = max(0.0, 1.0 - 0.6 * max(0.0, edit_rel - HARDENING_THRESHOLD))
        conf = clamp(base * pen_factor)
        if cap_hit:
            conf *= 0.9
        # No-edit bonus
        if rel_diff(orig_val, new_val) <= 1e-3:
            conf = clamp(conf + 0.05)
        return clamp(conf)

    for if_id, r in interim.items():
        router = r.get('local_router')
        peer = peer_of.get(if_id)

        status_comp = clamp(r.get('status_conf', 0.8))
        resolved_status = r.get('status', 'unknown')

        # Pair-based component with rate-aware tolerance (bounded) to avoid over-penalizing low rates
        if peer and interim.get(peer, {}).get('status') == resolved_status:
            # Forward residual and tolerance
            res_fwd = rel_diff(r['tx'], interim[peer]['rx'])
            traffic_tx = max(r['tx'], interim[peer]['rx'], 1.0)
            tol_pair_tx = min(0.12, max(TOL_PAIR_BASE, 5.0 / traffic_tx))
            pair_comp_tx = conf_from_residual(res_fwd, tol_pair_tx)
            # Reverse residual and tolerance
            res_rev = rel_diff(r['rx'], interim[peer]['tx'])
            traffic_rx = max(r['rx'], interim[peer]['tx'], 1.0)
            tol_pair_rx = min(0.12, max(TOL_PAIR_BASE, 5.0 / traffic_rx))
            pair_comp_rx = conf_from_residual(res_rev, tol_pair_rx)
        else:
            pair_comp_tx = 0.55
            pair_comp_rx = 0.55

        router_imb = router_final_imbalance.get(router, 0.0)
        router_comp = conf_from_residual(router_imb, TOL_ROUTER)

        base_tx_conf = w_pair * pair_comp_tx + w_router * router_comp + w_status * status_comp
        base_rx_conf = w_pair * pair_comp_rx + w_router * router_comp + w_status * status_comp

        # Edit-aware penalties and cap-hit adjustments
        final_tx_conf = finalize_conf(base_tx_conf, r.get('edit_tx_rel', 0.0), r.get('cap_hit_tx', False), r['orig_tx'], r['tx'])
        final_rx_conf = finalize_conf(base_rx_conf, r.get('edit_rx_rel', 0.0), r.get('cap_hit_rx', False), r['orig_rx'], r['rx'])

        # Additional router-scale and guard-usage penalties to avoid overconfidence
        scale_mag = router_scale_mag.get(router, 0.0)
        if scale_mag > 0.25:
            excess = clamp((scale_mag - 0.25) / 0.25, 0.0, 1.0)
            penalty_scale = 0.03 + 0.02 * excess
            final_tx_conf = clamp(final_tx_conf - penalty_scale)
            final_rx_conf = clamp(final_rx_conf - penalty_scale)
        usage = router_delta_frac.get(router, 0.0)
        if usage >= 0.9:
            penalty_usage = 0.05 * clamp((usage - 0.9) / 0.1, 0.0, 1.0)
            final_tx_conf = clamp(final_tx_conf - penalty_usage)
            final_rx_conf = clamp(final_rx_conf - penalty_usage)

        # Heavy per-interface edit proxy penalty (>~35%)
        if r.get('edit_tx_rel', 0.0) > 0.35:
            final_tx_conf = clamp(final_tx_conf - 0.04)
        if r.get('edit_rx_rel', 0.0) > 0.35:
            final_rx_conf = clamp(final_rx_conf - 0.04)

        # Improvement bonus when both pair residuals drop by >=50% and router imbalance decreases by >=30%
        peer = peer_of.get(if_id)
        if peer and interim.get(peer, {}).get('status') == resolved_status:
            pre_tx_res = pre_pair_residuals.get(if_id, {}).get('tx', 0.0)
            pre_rx_res = pre_pair_residuals.get(if_id, {}).get('rx', 0.0)
            post_tx_res = rel_diff(r['tx'], interim[peer]['rx'])
            post_rx_res = rel_diff(r['rx'], interim[peer]['tx'])
            tx_improved = pre_tx_res > 0 and (pre_tx_res - post_tx_res) / pre_tx_res >= 0.5
            rx_improved = pre_rx_res > 0 and (pre_rx_res - post_rx_res) / pre_rx_res >= 0.5
            pre_router_imb = router_imbalance_before.get(router, 0.0)
            post_router_imb = router_imbalance_before.get(router, 0.0)
            # Use final imbalance computed earlier
            post_router_imb = router_final_imbalance.get(router, 0.0)
            router_improved = pre_router_imb > 0 and (pre_router_imb - post_router_imb) / pre_router_imb >= 0.3
            if router_improved:
                if tx_improved and rx_improved:
                    final_tx_conf = clamp(final_tx_conf + 0.04)
                    final_rx_conf = clamp(final_rx_conf + 0.04)
                else:
                    if tx_improved:
                        final_tx_conf = clamp(final_tx_conf + 0.02)
                    if rx_improved:
                        final_rx_conf = clamp(final_rx_conf + 0.02)

        if resolved_status == 'down':
            final_rx_conf = 0.9 if r['orig_rx'] <= TRAFFIC_EVIDENCE_MIN else 0.3
            final_tx_conf = 0.9 if r['orig_tx'] <= TRAFFIC_EVIDENCE_MIN else 0.3

        r['tx_conf'] = final_tx_conf
        r['rx_conf'] = final_rx_conf

        # Subtle status calibration: if up but effectively idle, reduce status confidence slightly
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