# EVOLVE-BLOCK-START
"""
Network telemetry repair algorithm that detects and corrects inconsistencies
in network interface telemetry data using topology relationships.

This version implements a consensus-and-residual fusion approach enhanced with pair-bias scaling:
- Pair-bias pre-scaling of the lower-activity endpoint when both directions show consistent ratios
- Trust-weighted directional consensus with partial averaging near threshold
- Magnitude-aware gating with absolute guard and sharp agreement floors
- Asymmetric confidence shaping when only one side shows traffic
- Router-level micro-adjustments for dominating dangling interfaces
- Share-aware, direction-aware confidence penalties informed by flow-conservation residuals
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
    # Working values used by consensus (can be pre-scaled by pair-bias detection)
    work_vals: Dict[str, Dict[str, float]] = {}
    for if_id, data in telemetry.items():
        peer_id = data.get('connected_to')
        peers[if_id] = peer_id if peer_id in telemetry else None
        work_vals[if_id] = {
            'tx': float(data.get('tx_rate', 0.0)),
            'rx': float(data.get('rx_rate', 0.0))
        }

    # Plan pairwise consensus adjustments so both ends change consistently
    field_value_adjust: Dict[Tuple[str, str], float] = {}   # (iface, 'tx'|'rx') -> new_value
    field_conf_assign: Dict[Tuple[str, str], float] = {}    # set confidence when adjusted
    field_conf_floor: Dict[Tuple[str, str], float] = {}     # high floors when in strong agreement
    field_conf_scale: Dict[Tuple[str, str], float] = {}     # multiplicative confidence scalers (asymmetry)
    status_conf_floor: Dict[str, float] = {}                # status confidence floors under strong bilateral agreement

    # Utility for normalized difference
    def rel_diff(a: float, b: float) -> float:
        return abs(a - b) / max(1.0, a, b)

    # Stage 0: Pair-bias scaling (detect multiplicative scaling across link and partially correct)
    visited_pairs_bias = set()
    for a_id, a_data in telemetry.items():
        b_id = peers.get(a_id)
        if not b_id:
            continue
        key = tuple(sorted([a_id, b_id]))
        if key in visited_pairs_bias:
            continue
        visited_pairs_bias.add(key)

        a_status = a_data.get('interface_status', 'unknown')
        b_status = telemetry[b_id].get('interface_status', 'unknown')
        if a_status == 'down' or b_status == 'down':
            continue

        a_tx = work_vals[a_id]['tx']
        a_rx = work_vals[a_id]['rx']
        b_tx = work_vals[b_id]['tx']
        b_rx = work_vals[b_id]['rx']

        # Require active traffic in both directions to infer multiplicative bias
        if (a_tx <= QUIET_EPS or b_rx <= QUIET_EPS or a_rx <= QUIET_EPS or b_tx <= QUIET_EPS):
            continue

        # Ratios that should both approximate the same bias factor if one endpoint is scaled
        r1 = b_rx / max(a_tx, 1e-9)  # their_rx / my_tx
        r2 = b_tx / max(a_rx, 1e-9)  # their_tx / my_rx
        s = (r1 * r2) ** 0.5         # geometric mean ratio

        # Consistency check between ratios and magnitude-aware gating
        ratio_consistent = (abs(r1 - r2) / max(1.0, r1, r2)) <= 0.02
        hi = max(a_tx, a_rx, b_tx, b_rx)
        tol = LOW_RATE_THRESHOLD if hi < LOW_RATE_CUTOFF else HARDENING_THRESHOLD
        abs_ab = abs(a_tx - b_rx)
        abs_ba = abs(a_rx - b_tx)

        if (not ratio_consistent) or (abs(s - 1.0) <= tol) or (abs_ab <= ABS_GUARD and abs_ba <= ABS_GUARD):
            continue

        # Adjust the lower-activity side toward the bias-corrected scale
        act_a = max(a_tx, a_rx)
        act_b = max(b_tx, b_rx)
        if act_a < act_b:
            target = a_id
            scale = s
        else:
            target = b_id
            scale = 1.0 / max(1e-9, s)

        # Partial application near threshold to avoid overcorrection
        k = (abs(s - 1.0) - tol) / max(tol, 1e-9)
        k = clamp(k, 0.0, 1.0)
        factor = 1.0 + k * (scale - 1.0)

        # Update working values (so consensus sees the improved scale)
        work_vals[target]['tx'] = max(0.0, work_vals[target]['tx'] * factor)
        work_vals[target]['rx'] = max(0.0, work_vals[target]['rx'] * factor)

        # Stage the same updates so final repaired values include this correction (unless later overridden)
        if (target, 'tx') not in field_value_adjust:
            field_value_adjust[(target, 'tx')] = work_vals[target]['tx']
            # Confidence based on correction magnitude
            c_delta_tx = clamp(1.0 - min(1.0, 1.5 * abs(factor - 1.0)))
            field_conf_assign[(target, 'tx')] = max(field_conf_assign.get((target, 'tx'), 0.0), c_delta_tx)
        if (target, 'rx') not in field_value_adjust:
            field_value_adjust[(target, 'rx')] = work_vals[target]['rx']
            c_delta_rx = clamp(1.0 - min(1.0, 1.5 * abs(factor - 1.0)))
            field_conf_assign[(target, 'rx')] = max(field_conf_assign.get((target, 'rx'), 0.0), c_delta_rx)

    # Stage 1: Pairwise consensus-and-hardening
    visited_pairs = set()
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

        a_tx = work_vals[a_id]['tx']
        a_rx = work_vals[a_id]['rx']
        b_tx = work_vals[b_id]['tx']
        b_rx = work_vals[b_id]['rx']

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

            # Activity-based trust for consensus weighting
            act_a = max(a_tx, a_rx)
            act_b = max(b_tx, b_rx)
            denom_act = max(1e-9, act_a + act_b)
            w_a = act_a / denom_act
            w_b = act_b / denom_act

            # a->b hardening with gentle prescale and asymmetric partial averaging
            if diff_ab > tol_ab:
                if abs_ab > ABS_GUARD:
                    # Gentle prescale (only for consensus/k computation; do not commit prescaled values)
                    if max(a_tx, b_rx) >= 1.0:
                        s = sqrt(max(1e-9, b_rx) / max(1e-9, a_tx))
                        s = clamp(s, 0.90, 1.10)
                    else:
                        s = 1.0
                    a_tx_p = a_tx * s
                    b_rx_p = b_rx / s
                    diff_ab_p = rel_diff(a_tx_p, b_rx_p)

                    # Trust-weighted consensus on prescaled values
                    consensus_ab = w_a * a_tx_p + w_b * b_rx_p

                    # Low-rate–aware shaping
                    low_band_ab = max(a_tx, b_rx) < LOW_RATE_CUTOFF
                    full_mult_ab = 1.6 if low_band_ab else 2.0
                    exp_ab = 1.2 if low_band_ab else 1.0

                    if diff_ab_p <= full_mult_ab * tol_ab:
                        # Asymmetric partial averaging: move louder side more
                        k_raw = (diff_ab_p - tol_ab) / max(tol_ab, 1e-9)
                        k_base = max(0.0, min(1.0, k_raw ** exp_ab))
                        loud = max(a_tx, b_rx)
                        quiet = min(a_tx, b_rx)
                        r = (loud - quiet) / max(1.0, loud)
                        if a_tx >= b_rx:
                            k_a = clamp(k_base * (1.0 + 0.5 * r))
                            k_b = clamp(k_base * (1.0 - 0.5 * r))
                        else:
                            k_b = clamp(k_base * (1.0 + 0.5 * r))
                            k_a = clamp(k_base * (1.0 - 0.5 * r))
                        new_a_tx = a_tx * (1.0 - k_a) + consensus_ab * k_a
                        new_b_rx = b_rx * (1.0 - k_b) + consensus_ab * k_b
                    else:
                        # Clear violation: converge fully to consensus
                        new_a_tx = consensus_ab
                        new_b_rx = consensus_ab

                    field_value_adjust[(a_id, 'tx')] = new_a_tx
                    field_value_adjust[(b_id, 'rx')] = new_b_rx
                    conf_ab = clamp(1.0 - diff_ab_p)
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

            # b->a hardening with gentle prescale and asymmetric partial averaging
            if diff_ba > tol_ba:
                if abs_ba > ABS_GUARD:
                    if max(b_tx, a_rx) >= 1.0:
                        s2 = sqrt(max(1e-9, a_rx) / max(1e-9, b_tx))
                        s2 = clamp(s2, 0.90, 1.10)
                    else:
                        s2 = 1.0
                    b_tx_p = b_tx * s2
                    a_rx_p = a_rx / s2
                    diff_ba_p = rel_diff(b_tx_p, a_rx_p)

                    consensus_ba = w_b * b_tx_p + w_a * a_rx_p

                    low_band_ba = max(b_tx, a_rx) < LOW_RATE_CUTOFF
                    full_mult_ba = 1.6 if low_band_ba else 2.0
                    exp_ba = 1.2 if low_band_ba else 1.0

                    if diff_ba_p <= full_mult_ba * tol_ba:
                        k_raw = (diff_ba_p - tol_ba) / max(tol_ba, 1e-9)
                        k_base = max(0.0, min(1.0, k_raw ** exp_ba))
                        loud = max(b_tx, a_rx)
                        quiet = min(b_tx, a_rx)
                        r = (loud - quiet) / max(1.0, loud)
                        if b_tx >= a_rx:
                            k_b = clamp(k_base * (1.0 + 0.5 * r))
                            k_a = clamp(k_base * (1.0 - 0.5 * r))
                        else:
                            k_a = clamp(k_base * (1.0 + 0.5 * r))
                            k_b = clamp(k_base * (1.0 - 0.5 * r))
                        new_b_tx = b_tx * (1.0 - k_b) + consensus_ba * k_b
                        new_a_rx = a_rx * (1.0 - k_a) + consensus_ba * k_a
                    else:
                        new_b_tx = consensus_ba
                        new_a_rx = consensus_ba

                    field_value_adjust[(b_id, 'tx')] = new_b_tx
                    field_value_adjust[(a_id, 'rx')] = new_a_rx
                    conf_ba = clamp(1.0 - diff_ba_p)
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
            # Direction a->b
            if a_tx > QUIET_EPS and b_rx <= QUIET_EPS:
                field_conf_scale[(b_id, 'rx')] = min(0.85, field_conf_scale.get((b_id, 'rx'), 1.0))
            if b_rx > QUIET_EPS and a_tx <= QUIET_EPS:
                field_conf_scale[(a_id, 'tx')] = min(0.85, field_conf_scale.get((a_id, 'tx'), 1.0))
            # Direction b->a
            if b_tx > QUIET_EPS and a_rx <= QUIET_EPS:
                field_conf_scale[(a_id, 'rx')] = min(0.85, field_conf_scale.get((a_id, 'rx'), 1.0))
            if a_rx > QUIET_EPS and b_tx <= QUIET_EPS:
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
            # Apply pairwise counter adjustments (Stage 0 and Stage 1 may have populated these)
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

    # Apply safer, benefit-checked micro-adjustments on dominating dangling interfaces
    for r, if_list in router_ifaces.items():
        sum_tx, sum_rx = router_sums.get(r, (0.0, 0.0))
        imbalance = sum_tx - sum_rx
        abs_imb = abs(imbalance)
        if abs_imb <= 0.0:
            continue
        denom = max(1.0, sum_tx, sum_rx)
        resid_frac = abs_imb / denom
        if resid_frac < 0.03:
            continue

        # Identify unpaired, up interfaces with non-trivial traffic
        candidates = []
        for if_id in if_list:
            if if_id not in result:
                continue
            connected_to = result[if_id].get('connected_to')
            is_unpaired = not connected_to or connected_to not in telemetry
            status = result[if_id]['interface_status'][1]
            if is_unpaired and status == 'up':
                txv = float(result[if_id]['tx_rate'][1])
                rxv = float(result[if_id]['rx_rate'][1])
                if max(txv, rxv) < LOW_RATE_CUTOFF:
                    continue
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

        # Helper to compute router residual for current snapshot
        def compute_router_resid_local(router_id: str) -> float:
            stx, srx = 0.0, 0.0
            for iid in router_ifaces.get(router_id, []):
                if iid in result:
                    stx += float(result[iid]['tx_rate'][1])
                    srx += float(result[iid]['rx_rate'][1])
            return abs(stx - srx) / max(1.0, stx, srx)

        # Snapshot current values and internal skew
        orx, rrx, rc = result[top_if]['rx_rate']
        otx, rtx, tc = result[top_if]['tx_rate']
        pre_internal = abs(rtx - rrx) / max(1.0, max(rtx, rrx))
        resid_before = compute_router_resid_local(r)

        # Simulate both options and compute improvement
        def simulate(option: str):
            new_tx, new_rx = rtx, rrx
            if imbalance > 0.0:
                # tx excess: prefer reducing tx or increasing rx
                if option == 'tx':
                    new_tx = rtx * (1.0 - alpha)
                else:
                    new_rx = rrx * (1.0 + alpha)
            else:
                # rx excess: prefer reducing rx or increasing tx
                if option == 'rx':
                    new_rx = rrx * (1.0 - alpha)
                else:
                    new_tx = rtx * (1.0 + alpha)
            saved_tx = result[top_if]['tx_rate']
            saved_rx = result[top_if]['rx_rate']
            result[top_if]['tx_rate'] = (otx, new_tx, tc)
            result[top_if]['rx_rate'] = (orx, new_rx, rc)
            resid_tmp = compute_router_resid_local(r)
            post_internal_tmp = abs(new_tx - new_rx) / max(1.0, max(new_tx, new_rx))
            # Revert
            result[top_if]['tx_rate'] = saved_tx
            result[top_if]['rx_rate'] = saved_rx
            return resid_tmp, post_internal_tmp, new_tx, new_rx

        r_tx, post_tx_internal, cand_tx, cand_rx1 = simulate('tx')
        r_rx, post_rx_internal, cand_tx2, cand_rx2 = simulate('rx')

        # Pick option with larger residual improvement and guarded internal skew
        choice = None
        resid_after = resid_before
        new_tx = rtx
        new_rx = rrx
        if (r_tx <= r_rx) and (post_tx_internal <= pre_internal + 0.03):
            choice = 'tx'
            resid_after = r_tx
            new_tx, new_rx = cand_tx, cand_rx1
        elif (r_rx < r_tx) and (post_rx_internal <= pre_internal + 0.03):
            choice = 'rx'
            resid_after = r_rx
            new_tx, new_rx = cand_tx2, cand_rx2

        # Commit only if residual improves by at least 10%
        if choice is not None and resid_after <= (1.0 - 0.10) * resid_before:
            if choice == 'tx':
                new_conf = min(tc, 0.6 + 0.2 * (alpha / 0.02))
                result[top_if]['tx_rate'] = (otx, new_tx, clamp(new_conf))
            else:
                new_conf = min(rc, 0.6 + 0.2 * (alpha / 0.02))
                result[top_if]['rx_rate'] = (orx, new_rx, clamp(new_conf))
        # else: no commit if guard fails or insufficient improvement

    # Recompute residuals and directional sums after micro-adjustments
    router_resid: Dict[str, float] = {}
    router_sum_tx: Dict[str, float] = {}
    router_sum_rx: Dict[str, float] = {}
    for r, if_list in router_ifaces.items():
        sum_tx = 0.0
        sum_rx = 0.0
        for if_id in if_list:
            if if_id in result:
                sum_tx += float(result[if_id]['tx_rate'][1])
                sum_rx += float(result[if_id]['rx_rate'][1])
        denom = max(1.0, sum_tx, sum_rx)
        router_resid[r] = abs(sum_tx - sum_rx) / denom
        router_sum_tx[r] = sum_tx
        router_sum_rx[r] = sum_rx

    # Apply tri-axis confidence composition:
    # - share-aware, direction-aware router penalties
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

        # Interface directional shares for share-aware penalties (conservative floor retained)
        _, rrx, rc = result[if_id]['rx_rate']
        _, rtx, tc = result[if_id]['tx_rate']
        tx_share = rtx / max(1.0, router_sum_tx.get(lr, 0.0))
        rx_share = rrx / max(1.0, router_sum_rx.get(lr, 0.0))
        penalty_tx = clamp(1.0 - ((0.6 + 0.2 * tx_share) * resid_local + (0.4 - 0.2 * tx_share) * resid_remote), 0.5, 1.0)
        penalty_rx = clamp(1.0 - ((0.6 + 0.2 * rx_share) * resid_local + (0.4 - 0.2 * rx_share) * resid_remote), 0.5, 1.0)

        orx, _, _ = result[if_id]['rx_rate']
        otx, _, _ = result[if_id]['tx_rate']
        ost, rst, sc = result[if_id]['interface_status']

        # If repaired status is down, keep zeros and only scale confidences by router penalties
        if rst == 'down':
            result[if_id]['rx_rate'] = (orx, 0.0, clamp(rc * penalty_rx))
            result[if_id]['tx_rate'] = (otx, 0.0, clamp(tc * penalty_tx))
            status_scale = 0.85 + 0.15 * min(penalty_tx, penalty_rx)
            result[if_id]['interface_status'] = (ost, rst, clamp(sc * status_scale))
            continue

        # Peer repaired values for symmetry assessment
        peer_id = d.get('connected_to')
        peer_exists = (peer_id in result) if peer_id else False
        if peer_exists:
            _, peer_rrx, _ = result[peer_id]['rx_rate']
            _, peer_rtx, _ = result[peer_id]['tx_rate']
        else:
            peer_rrx = None
            peer_rtx = None

        # Symmetry fit after repair
        if peer_exists and peer_rrx is not None and peer_rtx is not None:
            diff_tx = rel_diff(rtx, peer_rrx)  # my_tx vs their_rx
            diff_rx = rel_diff(rrx, peer_rtx)  # my_rx vs their_tx
            c_sym_tx = clamp(1.0 - diff_tx)
            c_sym_rx = clamp(1.0 - diff_rx)

            # Magnitude-aware thresholds and floors
            th_tx = LOW_RATE_THRESHOLD if max(rtx, peer_rrx) < LOW_RATE_CUTOFF else HARDENING_THRESHOLD
            th_rx = LOW_RATE_THRESHOLD if max(rrx, peer_rtx) < LOW_RATE_CUTOFF else HARDENING_THRESHOLD
            if max(rtx, peer_rrx) >= 10.0 and diff_tx <= 0.005:
                floor_tx = 0.99
            elif diff_tx <= th_tx:
                floor_tx = 0.98 if max(rtx, peer_rrx) >= 10.0 else 0.97
            else:
                floor_tx = 0.0
            if max(rrx, peer_rtx) >= 10.0 and diff_rx <= 0.005:
                floor_rx = 0.99
            elif diff_rx <= th_rx:
                floor_rx = 0.98 if max(rrx, peer_rtx) >= 10.0 else 0.97
            else:
                floor_rx = 0.0
        else:
            # No redundancy; use conservative defaults
            c_sym_tx = 0.9
            c_sym_rx = 0.9
            floor_tx = 0.0
            floor_rx = 0.0

        # Correction magnitude component (bigger changes => lower confidence)
        m_tx = abs(rtx - otx) / max(1.0, rtx, otx)
        m_rx = abs(rrx - orx) / max(1.0, rrx, orx)
        c_delta_tx = clamp(1.0 - min(1.0, 1.5 * m_tx))
        c_delta_rx = clamp(1.0 - min(1.0, 1.5 * m_rx))

        # Compose final confidences
        conf_tx_new = clamp(0.45 * penalty_tx + 0.35 * c_sym_tx + 0.20 * c_delta_tx)
        conf_rx_new = clamp(0.45 * penalty_rx + 0.35 * c_sym_rx + 0.20 * c_delta_rx)

        # Apply magnitude-aware floors
        conf_tx_new = max(conf_tx_new, floor_tx)
        conf_rx_new = max(conf_rx_new, floor_rx)

        # Asymmetric traffic-evidence shaping: penalize silent side if peer shows traffic
        if peer_exists and peer_rrx is not None and rtx <= QUIET_EPS and peer_rrx > QUIET_EPS:
            conf_tx_new = clamp(conf_tx_new * 0.88)
        if peer_exists and peer_rtx is not None and rrx <= QUIET_EPS and peer_rtx > QUIET_EPS:
            conf_rx_new = clamp(conf_rx_new * 0.88)

        result[if_id]['tx_rate'] = (otx, rtx, conf_tx_new)
        result[if_id]['rx_rate'] = (orx, rrx, conf_rx_new)

        # Status confidence shaping: residual-based scaling + mild alignment with per-direction confidences
        status_scale = 0.85 + 0.15 * min(penalty_tx, penalty_rx)
        status_conf_new = clamp(sc * status_scale)
        status_conf_new = clamp(status_conf_new * (0.85 + 0.15 * min(conf_tx_new, conf_rx_new)))
        result[if_id]['interface_status'] = (ost, rst, status_conf_new)

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