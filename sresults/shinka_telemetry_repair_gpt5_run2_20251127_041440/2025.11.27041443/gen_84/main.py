# EVOLVE-BLOCK-START
"""
Network telemetry repair algorithm that detects and corrects inconsistencies
in network interface telemetry data using topology relationships.

This version implements a consensus-and-residual fusion approach:
- Trust-weighted directional consensus with partial averaging near threshold
- Magnitude-aware gating with absolute guard and sharp agreement floors
- Asymmetric confidence shaping when only one side shows traffic
- Router-level micro-adjustments for dominating dangling interfaces
- Direction-aware confidence penalties informed by flow-conservation residuals
"""
from typing import Dict, Any, Tuple, List
from math import sqrt


def repair_network_telemetry(telemetry: Dict[str, Dict[str, Any]],
                             topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:
    """
    Repair network interface telemetry by detecting and correcting inconsistencies.

    Core invariants used:
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
    # Hardened tolerances (Hodor guidance)
    HARDENING_THRESHOLD = 0.02   # 2% for normal rates
    LOW_RATE_CUTOFF = 10.0       # Mbps threshold for tiny flows
    LOW_RATE_THRESHOLD = 0.05    # 5% tolerance when small
    ABS_GUARD = 0.5              # Mbps; absolute guard to avoid over-correcting tiny flows
    QUIET_EPS = 0.1              # Mbps; traffic "silence" threshold for asymmetry handling

    def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
        return max(lo, min(hi, x))

    result: Dict[str, Dict[str, Tuple]] = {}

    # Build peer mapping for quick, validated lookup
    peers: Dict[str, str] = {}
    for if_id, data in telemetry.items():
        peer_id = data.get('connected_to')
        peers[if_id] = peer_id if peer_id in telemetry else None

    # Precompute router signed residuals from raw telemetry (for residual-tilted consensus)
    pre_router_ifaces: Dict[str, List[str]] = {r: list(if_list) for r, if_list in topology.items()}
    for if_id, d in telemetry.items():
        lr = d.get('local_router')
        if lr:
            pre_router_ifaces.setdefault(lr, [])
            if if_id not in pre_router_ifaces[lr]:
                pre_router_ifaces[lr].append(if_id)
        rr = d.get('remote_router')
        if rr and rr not in pre_router_ifaces:
            pre_router_ifaces[rr] = []
    pre_router_resid_signed: Dict[str, float] = {}
    for r, if_list in pre_router_ifaces.items():
        sum_tx_raw = 0.0
        sum_rx_raw = 0.0
        for iid in if_list:
            if iid in telemetry:
                d = telemetry[iid]
                st = d.get('interface_status', 'unknown')
                txv = float(d.get('tx_rate', 0.0)) if st != 'down' else 0.0
                rxv = float(d.get('rx_rate', 0.0)) if st != 'down' else 0.0
                sum_tx_raw += txv
                sum_rx_raw += rxv
        denom = max(1.0, sum_tx_raw, sum_rx_raw)
        pre_router_resid_signed[r] = (sum_tx_raw - sum_rx_raw) / denom

    # Plan pairwise consensus adjustments so both ends change consistently
    field_value_adjust: Dict[Tuple[str, str], float] = {}   # (iface, 'tx'|'rx') -> new_value
    field_conf_assign: Dict[Tuple[str, str], float] = {}    # set confidence when adjusted
    field_conf_floor: Dict[Tuple[str, str], float] = {}     # high floors when in strong agreement
    field_conf_scale: Dict[Tuple[str, str], float] = {}     # multiplicative confidence scalers (asymmetry)
    status_conf_floor: Dict[str, float] = {}                # status confidence floors under strong bilateral agreement

    visited_pairs = set()

    # Utility for normalized difference
    def rel_diff(a: float, b: float) -> float:
        return abs(a - b) / max(1.0, a, b)

    # Pairwise consensus-and-hardening
    for a_id, a_data in telemetry.items():
        b_id = peers.get(a_id)
        if not b_id:
            continue
        key = tuple(sorted([a_id, b_id]))
        if key in visited_pairs:
            continue
        visited_pairs.add(key)

        a_status = a_data.get('interface_status', 'unknown')
        b_status = telemetry[b_id].get('interface_status', 'unknown')

        a_tx = float(a_data.get('tx_rate', 0.0))
        a_rx = float(a_data.get('rx_rate', 0.0))
        b_tx = float(telemetry[b_id].get('tx_rate', 0.0))
        b_rx = float(telemetry[b_id].get('rx_rate', 0.0))

        # Only attempt counter fusion if neither side is explicitly down; down logic handled later
        if a_status != 'down' and b_status != 'down':
            # Direction a->b: compare a_tx with b_rx
            abs_ab = abs(a_tx - b_rx)
            max_ab = max(1.0, a_tx, b_rx)
            diff_ab = abs_ab / max_ab
            tol_ab = LOW_RATE_THRESHOLD if max(a_tx, b_rx) < LOW_RATE_CUTOFF else HARDENING_THRESHOLD

            # Direction b->a: compare b_tx with a_rx
            abs_ba = abs(b_tx - a_rx)
            max_ba = max(1.0, b_tx, a_rx)
            diff_ba = abs_ba / max_ba
            tol_ba = LOW_RATE_THRESHOLD if max(b_tx, a_rx) < LOW_RATE_CUTOFF else HARDENING_THRESHOLD

            # Activity-based trust for consensus weighting (novel bias)
            # Bias toward the side that exhibits stronger traffic magnitude (more reliable signal)
            act_a = max(a_tx, a_rx)
            act_b = max(b_tx, b_rx)
            denom_act = max(1e-9, act_a + act_b)
            w_a = act_a / denom_act
            w_b = act_b / denom_act

            # a->b hardening
            if diff_ab > tol_ab:
                if abs_ab > ABS_GUARD:
                    # Trust-weighted consensus with residual-tilted weights to reduce local router residual
                    w_a_dir = w_a
                    w_b_dir = w_b
                    a_router = a_data.get('local_router')
                    resid_local_a = 0.0 if a_router is None else pre_router_resid_signed.get(a_router, 0.0)
                    sgn_delta = 1 if (a_tx - b_rx) > 0 else (-1 if (a_tx - b_rx) < 0 else 0)
                    sgn_resid = 1 if resid_local_a > 0 else (-1 if resid_local_a < 0 else 0)
                    if sgn_delta == sgn_resid and sgn_delta != 0:
                        gamma = min(0.08, 0.1 * abs(resid_local_a))
                        # Tilt weights away from local side to pull toward reducing residual
                        w_a_dir = clamp(max(0.2, min(0.8, w_a - gamma)))
                        w_b_dir = 1.0 - w_a_dir
                    consensus_ab = w_a_dir * a_tx + w_b_dir * b_rx
                    if diff_ab <= 2 * tol_ab:
                        # Partial averaging near threshold to reduce overcorrection
                        k = (diff_ab - tol_ab) / max(tol_ab, 1e-9)
                        new_a_tx = a_tx * (1.0 - k) + consensus_ab * k
                        new_b_rx = b_rx * (1.0 - k) + consensus_ab * k
                    else:
                        # Clear violation: converge fully to consensus
                        new_a_tx = consensus_ab
                        new_b_rx = consensus_ab
                    field_value_adjust[(a_id, 'tx')] = new_a_tx
                    field_value_adjust[(b_id, 'rx')] = new_b_rx
                    conf_ab = clamp(1.0 - diff_ab)
                    field_conf_assign[(a_id, 'tx')] = conf_ab
                    field_conf_assign[(b_id, 'rx')] = conf_ab
                else:
                    # Guard-blocked discrepancy: do not set high floors; downscale confidence instead
                    field_conf_scale[(a_id, 'tx')] = min(0.9, field_conf_scale.get((a_id, 'tx'), 1.0))
                    field_conf_scale[(b_id, 'rx')] = min(0.9, field_conf_scale.get((b_id, 'rx'), 1.0))
            else:
                # Within tolerance: floor high confidence
                if max(a_tx, b_rx) >= 10.0 and diff_ab <= 0.005:
                    floor = 0.99
                else:
                    floor = 0.98
                field_conf_floor[(a_id, 'tx')] = max(field_conf_floor.get((a_id, 'tx'), 0.0), floor)
                field_conf_floor[(b_id, 'rx')] = max(field_conf_floor.get((b_id, 'rx'), 0.0), floor)

            # b->a hardening
            if diff_ba > tol_ba:
                if abs_ba > ABS_GUARD:
                    # Trust-weighted consensus with residual-tilted weights (direction b->a)
                    w_b_dir = w_b
                    w_a_dir = w_a
                    b_router = telemetry[b_id].get('local_router')
                    resid_local_b = 0.0 if b_router is None else pre_router_resid_signed.get(b_router, 0.0)
                    sgn_delta = 1 if (b_tx - a_rx) > 0 else (-1 if (b_tx - a_rx) < 0 else 0)
                    sgn_resid = 1 if resid_local_b > 0 else (-1 if resid_local_b < 0 else 0)
                    if sgn_delta == sgn_resid and sgn_delta != 0:
                        gamma = min(0.08, 0.1 * abs(resid_local_b))
                        w_b_dir = clamp(max(0.2, min(0.8, w_b - gamma)))
                        w_a_dir = 1.0 - w_b_dir
                    consensus_ba = w_b_dir * b_tx + w_a_dir * a_rx
                    if diff_ba <= 2 * tol_ba:
                        k = (diff_ba - tol_ba) / max(tol_ba, 1e-9)
                        new_b_tx = b_tx * (1.0 - k) + consensus_ba * k
                        new_a_rx = a_rx * (1.0 - k) + consensus_ba * k
                    else:
                        new_b_tx = consensus_ba
                        new_a_rx = consensus_ba
                    field_value_adjust[(b_id, 'tx')] = new_b_tx
                    field_value_adjust[(a_id, 'rx')] = new_a_rx
                    conf_ba = clamp(1.0 - diff_ba)
                    field_conf_assign[(b_id, 'tx')] = conf_ba
                    field_conf_assign[(a_id, 'rx')] = conf_ba
                else:
                    # Guard-blocked discrepancy: do not set high floors; downscale confidence instead
                    field_conf_scale[(b_id, 'tx')] = min(0.9, field_conf_scale.get((b_id, 'tx'), 1.0))
                    field_conf_scale[(a_id, 'rx')] = min(0.9, field_conf_scale.get((a_id, 'rx'), 1.0))
            else:
                if max(b_tx, a_rx) >= 10.0 and diff_ba <= 0.005:
                    floor = 0.99
                else:
                    floor = 0.98
                field_conf_floor[(b_id, 'tx')] = max(field_conf_floor.get((b_id, 'tx'), 0.0), floor)
                field_conf_floor[(a_id, 'rx')] = max(field_conf_floor.get((a_id, 'rx'), 0.0), floor)

            # Asymmetric confidence when only one side shows traffic (traffic-evidence asymmetry)
            pair_up = True
            # Direction a->b
            if pair_up and a_tx > QUIET_EPS and b_rx <= QUIET_EPS:
                field_conf_scale[(b_id, 'rx')] = min(0.85, field_conf_scale.get((b_id, 'rx'), 1.0))
            if pair_up and b_rx > QUIET_EPS and a_tx <= QUIET_EPS:
                field_conf_scale[(a_id, 'tx')] = min(0.85, field_conf_scale.get((a_id, 'tx'), 1.0))
            # Direction b->a
            if pair_up and b_tx > QUIET_EPS and a_rx <= QUIET_EPS:
                field_conf_scale[(a_id, 'rx')] = min(0.85, field_conf_scale.get((a_id, 'rx'), 1.0))
            if pair_up and a_rx > QUIET_EPS and b_tx <= QUIET_EPS:
                field_conf_scale[(b_id, 'tx')] = min(0.85, field_conf_scale.get((b_id, 'tx'), 1.0))

            # Harmonize very strong agreements via geometric-mean floors
            strong_ab = (max(a_tx, b_rx) >= 10.0 and diff_ab <= 0.005)
            strong_ba = (max(b_tx, a_rx) >= 10.0 and diff_ba <= 0.005)
            if strong_ab:
                # Geometric mean harmonization for link-direction confidences
                fa = field_conf_floor.get((a_id, 'tx'), 0.98)
                fb = field_conf_floor.get((b_id, 'rx'), 0.98)
                gm = sqrt(max(1e-9, fa) * max(1e-9, fb))
                field_conf_floor[(a_id, 'tx')] = max(field_conf_floor.get((a_id, 'tx'), 0.0), gm)
                field_conf_floor[(b_id, 'rx')] = max(field_conf_floor.get((b_id, 'rx'), 0.0), gm)
            if strong_ba:
                fb = field_conf_floor.get((b_id, 'tx'), 0.98)
                fa = field_conf_floor.get((a_id, 'rx'), 0.98)
                gm = sqrt(max(1e-9, fa) * max(1e-9, fb))
                field_conf_floor[(b_id, 'tx')] = max(field_conf_floor.get((b_id, 'tx'), 0.0), gm)
                field_conf_floor[(a_id, 'rx')] = max(field_conf_floor.get((a_id, 'rx'), 0.0), gm)
            # If both directions are strongly consistent, raise status confidence floor
            if strong_ab and strong_ba:
                status_conf_floor[a_id] = max(status_conf_floor.get(a_id, 0.0), 0.99)
                status_conf_floor[b_id] = max(status_conf_floor.get(b_id, 0.0), 0.99)

    # Second pass: apply planned adjustments and assign calibrated confidences
    for interface_id, data in telemetry.items():
        repaired = {}

        interface_status = data.get('interface_status', 'unknown')
        rx_rate = float(data.get('rx_rate', 0.0))
        tx_rate = float(data.get('tx_rate', 0.0))
        connected_to = data.get('connected_to')

        # Defaults: identity with conservative base
        repaired_rx = rx_rate
        repaired_tx = tx_rate
        repaired_status = interface_status
        rx_conf = 0.95
        tx_conf = 0.95
        status_conf = 0.95

        # Peer snapshot
        peer_status = None
        if connected_to and connected_to in telemetry:
            peer_status = telemetry[connected_to].get('interface_status', 'unknown')

        # Enforce interface consistency: if either side is down, set both down with zero rates
        if interface_status == 'down' or (peer_status == 'down' if peer_status is not None else False):
            both_down = (interface_status == 'down' and (peer_status == 'down' if peer_status is not None else False))
            repaired_status = 'down'
            repaired_rx = 0.0
            repaired_tx = 0.0
            status_conf = 0.95 if both_down else 0.7
            rx_conf = status_conf
            tx_conf = status_conf
        else:
            # Apply pairwise counter adjustments
            if (interface_id, 'rx') in field_value_adjust:
                repaired_rx = float(field_value_adjust[(interface_id, 'rx')])
                rx_conf = field_conf_assign.get((interface_id, 'rx'), rx_conf)
            if (interface_id, 'tx') in field_value_adjust:
                repaired_tx = float(field_value_adjust[(interface_id, 'tx')])
                tx_conf = field_conf_assign.get((interface_id, 'tx'), tx_conf)

            # Status mismatch (neither side down) reduces status confidence
            if connected_to and connected_to in telemetry:
                if interface_status != peer_status:
                    status_conf = min(status_conf, 0.6)

            # For unpaired interfaces (no redundancy), use a slightly lower baseline confidence
            if not connected_to or connected_to not in telemetry:
                rx_conf = min(rx_conf, 0.9)
                tx_conf = min(tx_conf, 0.9)

            # Apply confidence floors for in-tolerance agreements
            rx_floor = field_conf_floor.get((interface_id, 'rx'))
            tx_floor = field_conf_floor.get((interface_id, 'tx'))
            if rx_floor is not None:
                rx_conf = max(rx_conf, rx_floor)
            if tx_floor is not None:
                tx_conf = max(tx_conf, tx_floor)

            # Apply asymmetric scaling if applicable
            if (interface_id, 'rx') in field_conf_scale:
                rx_conf = clamp(rx_conf * field_conf_scale[(interface_id, 'rx')])
            if (interface_id, 'tx') in field_conf_scale:
                tx_conf = clamp(tx_conf * field_conf_scale[(interface_id, 'tx')])

            # Apply status confidence floor when strong bilateral agreement detected
            sfloor = status_conf_floor.get(interface_id)
            if sfloor is not None:
                status_conf = max(status_conf, sfloor)

        # Store
        repaired['rx_rate'] = (rx_rate, repaired_rx, clamp(rx_conf))
        repaired['tx_rate'] = (tx_rate, repaired_tx, clamp(tx_conf))
        repaired['interface_status'] = (interface_status, repaired_status, clamp(status_conf))
        repaired['connected_to'] = connected_to
        repaired['local_router'] = data.get('local_router')
        repaired['remote_router'] = data.get('remote_router')
        result[interface_id] = repaired

    # Router-level flow conservation: micro-adjustments on dominating dangling interfaces
    # Build router->interfaces mapping using provided topology and telemetry fallbacks
    router_ifaces: Dict[str, List[str]] = {r: list(if_list) for r, if_list in topology.items()}
    for if_id, d in telemetry.items():
        lr = d.get('local_router')
        if lr:
            router_ifaces.setdefault(lr, [])
            if if_id not in router_ifaces[lr]:
                router_ifaces[lr].append(if_id)
        rr = d.get('remote_router')
        if rr and rr not in router_ifaces:
            router_ifaces[rr] = []

    # Compute per-router sums
    router_sums: Dict[str, Tuple[float, float]] = {}
    for r, if_list in router_ifaces.items():
        sum_tx = 0.0
        sum_rx = 0.0
        for if_id in if_list:
            if if_id in result:
                sum_tx += float(result[if_id]['tx_rate'][1])
                sum_rx += float(result[if_id]['rx_rate'][1])
        router_sums[r] = (sum_tx, sum_rx)

    # Apply tightly scoped micro-adjustments only on dominating dangling interfaces (recommendation 4)
    for r, if_list in router_ifaces.items():
        sum_tx, sum_rx = router_sums.get(r, (0.0, 0.0))
        imbalance = sum_tx - sum_rx
        abs_imb = abs(imbalance)
        if abs_imb <= 0.0:
            continue
        denom = max(1.0, sum_tx, sum_rx)
        resid_frac = abs_imb / denom

        # Identify unpaired, up interfaces
        candidates = []
        for if_id in if_list:
            if if_id not in result:
                continue
            # unpaired if peer missing in telemetry
            connected_to = result[if_id].get('connected_to')
            is_unpaired = not connected_to or connected_to not in telemetry
            status = result[if_id]['interface_status'][1]
            if is_unpaired and status == 'up':
                txv = float(result[if_id]['tx_rate'][1])
                rxv = float(result[if_id]['rx_rate'][1])
                contrib = abs(txv - rxv)
                candidates.append((contrib, if_id, txv, rxv))

        if not candidates:
            continue

        candidates.sort(reverse=True)
        top_contrib, top_if, txv, rxv = candidates[0]
        if top_contrib < 0.5 * abs_imb:
            continue  # not dominating enough

        alpha = min(0.02, 0.5 * resid_frac)
        if alpha <= 0.0:
            continue

        # Adjust only the larger counter toward reducing the router imbalance
        orx, rrx, rc = result[top_if]['rx_rate']
        otx, rtx, tc = result[top_if]['tx_rate']
        if imbalance > 0.0:
            # sum_tx > sum_rx: reduce tx or increase rx; nudge only larger counter
            if rtx >= rrx:
                new_tx = rtx * (1.0 - alpha)
                # reduce confidence to reflect heuristic adjustment
                new_conf = min(tc, 0.6 + 0.2 * (alpha / 0.02))
                result[top_if]['tx_rate'] = (otx, new_tx, clamp(new_conf))
            else:
                new_rx = rrx * (1.0 + alpha)
                new_conf = min(rc, 0.6 + 0.2 * (alpha / 0.02))
                result[top_if]['rx_rate'] = (orx, new_rx, clamp(new_conf))
        else:
            # sum_tx < sum_rx: reduce rx or increase tx; nudge only larger counter
            if rrx >= rtx:
                new_rx = rrx * (1.0 - alpha)
                new_conf = min(rc, 0.6 + 0.2 * (alpha / 0.02))
                result[top_if]['rx_rate'] = (orx, new_rx, clamp(new_conf))
            else:
                new_tx = rtx * (1.0 + alpha)
                new_conf = min(tc, 0.6 + 0.2 * (alpha / 0.02))
                result[top_if]['tx_rate'] = (otx, new_tx, clamp(new_conf))

    # Recompute residuals after micro-adjustments
    router_resid: Dict[str, float] = {}
    for r, if_list in router_ifaces.items():
        sum_tx = 0.0
        sum_rx = 0.0
        for if_id in if_list:
            if if_id in result:
                sum_tx += float(result[if_id]['tx_rate'][1])
                sum_rx += float(result[if_id]['rx_rate'][1])
        denom = max(1.0, sum_tx, sum_rx)
        router_resid[r] = abs(sum_tx - sum_rx) / denom

    # Apply tri-axis confidence composition:
    # - direction-aware router penalties
    # - link symmetry fit after repair
    # - correction magnitude (original vs repaired)
    # with magnitude-aware floors and asymmetric traffic-evidence shaping.
    for if_id, d in telemetry.items():
        if if_id not in result:
            continue

        lr = d.get('local_router')
        rr = d.get('remote_router')
        resid_local = router_resid.get(lr, 0.0)
        resid_remote = router_resid.get(rr, 0.0)
        pen_tx = clamp(1.0 - (0.6 * resid_local + 0.4 * resid_remote), 0.0, 1.0)
        pen_rx = clamp(1.0 - (0.6 * resid_remote + 0.4 * resid_local), 0.0, 1.0)

        orx, rrx, rc = result[if_id]['rx_rate']
        otx, rtx, tc = result[if_id]['tx_rate']
        ost, rst, sc = result[if_id]['interface_status']

        # If repaired status is down, keep zeros and current confidence (handled earlier)
        if rst == 'down':
            status_scale = 0.85 + 0.15 * min(pen_tx, pen_rx)
            result[if_id]['interface_status'] = (ost, rst, clamp(sc * status_scale))
            result[if_id]['tx_rate'] = (otx, 0.0, tc)
            result[if_id]['rx_rate'] = (orx, 0.0, rc)
            continue

        # Peer-repaired values for symmetry; may be missing
        peer_id = d.get('connected_to')
        peer_exists = peer_id in result if peer_id else False
        if peer_exists:
            porx, prrx, prc = result[peer_id]['rx_rate']
            potx, prtx, ptc = result[peer_id]['tx_rate']
        else:
            prrx = None
            prtx = None

        # Helpers
        def rel_diff(a: float, b: float) -> float:
            return abs(a - b) / max(1.0, a, b)

        def dir_threshold(a: float, b: float) -> float:
            return LOW_RATE_THRESHOLD if max(a, b) < LOW_RATE_CUTOFF else HARDENING_THRESHOLD

        # Compute symmetry fit on repaired values
        if peer_exists:
            diff_tx = rel_diff(rtx, prrx)  # my_tx vs their_rx
            diff_rx = rel_diff(rrx, prtx)  # my_rx vs their_tx
            c_sym_tx = clamp(1.0 - diff_tx)
            c_sym_rx = clamp(1.0 - diff_rx)
            # Magnitude-aware floors
            th_tx = dir_threshold(rtx, prrx)
            th_rx = dir_threshold(rrx, prtx)
            if max(rtx, prrx) >= 10.0 and diff_tx <= 0.005:
                floor_tx = 0.99
            elif diff_tx <= th_tx:
                floor_tx = 0.98 if max(rtx, prrx) >= 10.0 else 0.97
            else:
                floor_tx = 0.0
            if max(rrx, prtx) >= 10.0 and diff_rx <= 0.005:
                floor_rx = 0.99
            elif diff_rx <= th_rx:
                floor_rx = 0.98 if max(rrx, prtx) >= 10.0 else 0.97
            else:
                floor_rx = 0.0
        else:
            c_sym_tx = 0.9
            c_sym_rx = 0.9
            floor_tx = 0.0
            floor_rx = 0.0
            diff_tx = None
            diff_rx = None

        # Correction magnitude component (bigger changes => lower confidence)
        m_tx = abs(rtx - otx) / max(1.0, rtx, otx)
        m_rx = abs(rrx - orx) / max(1.0, rrx, orx)
        c_delta_tx = clamp(1.0 - min(1.0, 1.5 * m_tx))
        c_delta_rx = clamp(1.0 - min(1.0, 1.5 * m_rx))

        # Compose new confidences
        conf_tx_new = clamp(0.45 * pen_tx + 0.35 * c_sym_tx + 0.20 * c_delta_tx)
        conf_rx_new = clamp(0.45 * pen_rx + 0.35 * c_sym_rx + 0.20 * c_delta_rx)

        # Apply magnitude-aware floors
        conf_tx_new = max(conf_tx_new, floor_tx)
        conf_rx_new = max(conf_rx_new, floor_rx)

        # Asymmetric traffic-evidence shaping: penalize silent side if peer shows traffic
        if peer_exists and prrx is not None:
            if rtx <= QUIET_EPS and prrx > QUIET_EPS:
                conf_tx_new = clamp(conf_tx_new * 0.88)
        if peer_exists and prtx is not None:
            if rrx <= QUIET_EPS and prtx > QUIET_EPS:
                conf_rx_new = clamp(conf_rx_new * 0.88)

        result[if_id]['tx_rate'] = (otx, rtx, conf_tx_new)
        result[if_id]['rx_rate'] = (orx, rrx, conf_rx_new)

        # Status confidence shaping:
        # - Boost when both directions show strong bilateral agreement
        # - Mildly align with per-direction confidences
        status_conf = sc
        if peer_exists and diff_tx is not None and diff_rx is not None:
            th_tx_cur = dir_threshold(rtx, prrx)
            th_rx_cur = dir_threshold(rrx, prtx)
            strong_tx = (max(rtx, prrx) >= 10.0 and diff_tx <= 0.005)
            strong_rx = (max(rrx, prtx) >= 10.0 and diff_rx <= 0.005)
            if strong_tx and strong_rx:
                status_conf = max(status_conf, 0.99)
            elif (diff_tx <= th_tx_cur and diff_rx <= th_rx_cur):
                status_conf = max(status_conf, 0.97)
        # Mild status confidence scaling with penalties and alignment
        status_scale = 0.85 + 0.15 * min(pen_tx, pen_rx)
        status_conf = clamp(status_conf * status_scale)
        status_conf = clamp(status_conf * (0.85 + 0.15 * min(conf_tx_new, conf_rx_new)))
        result[if_id]['interface_status'] = (ost, rst, status_conf)

    # Ensure zero rates if repaired status is down (idempotent safety)
    for if_id, d in result.items():
        status = d['interface_status'][1]
        if status == 'down':
            orx, _, rc = d['rx_rate']
            otx, _, tc = d['tx_rate']
            d['rx_rate'] = (orx, 0.0, rc)
            d['tx_rate'] = (otx, 0.0, tc)

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