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

    def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
        return max(lo, min(hi, x))

    def rel_diff(a: float, b: float) -> float:
        denom = max(abs(a), abs(b), 1e-9)
        return abs(a - b) / denom

    def conf_from_residual(residual: float, tol: float) -> float:
        # Map residual to confidence: 1 at 0 residual, degrades linearly until ~0 near 5*tol
        denom = max(tol * 5.0, 1e-9)
        return clamp(1.0 - residual / denom)

    # Initialize structures
    result: Dict[str, Dict[str, Tuple]] = {}
    # Store interim repaired values and confidences per interface before router-level hardening
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
            'rx': data.get('rx_rate', 0.0),
            'tx': data.get('tx_rate', 0.0),
            'rx_conf': 1.0,
            'tx_conf': 1.0,
            'status': data.get('interface_status', 'unknown'),
            'status_conf': 1.0,
            'connected_to': data.get('connected_to'),
            'local_router': data.get('local_router'),
            'remote_router': data.get('remote_router'),
            # Keep originals for output tuples
            'orig_rx': data.get('rx_rate', 0.0),
            'orig_tx': data.get('tx_rate', 0.0),
            'orig_status': data.get('interface_status', 'unknown'),
        }

    # Pair-level hardening using link symmetry (R3) and interface consistency
    for a_id, b_id in pairs:
        a = telemetry[a_id]
        b = telemetry[b_id]
        a_stat = a.get('interface_status', 'unknown')
        b_stat = b.get('interface_status', 'unknown')
        a_rx, a_tx = a.get('rx_rate', 0.0), a.get('tx_rate', 0.0)
        b_rx, b_tx = b.get('rx_rate', 0.0), b.get('tx_rate', 0.0)
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
            for if_id, rx0, tx0 in [(a_id, a_rx, a_tx), (b_id, b_rx, b_tx)]:
                interim[if_id]['rx'] = 0.0
                interim[if_id]['tx'] = 0.0
                interim[if_id]['rx_conf'] = clamp(0.9 if rx0 <= TRAFFIC_EVIDENCE_MIN else 0.3)
                interim[if_id]['tx_conf'] = clamp(0.9 if tx0 <= TRAFFIC_EVIDENCE_MIN else 0.3)
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

    # Router-level dynamic flow conservation (R1)
    # Build router to interfaces map (use provided topology if available, else derive from telemetry)
    router_ifaces: Dict[str, List[str]] = {}
    if topology:
        router_ifaces = {r: list(if_list) for r, if_list in topology.items()}
    else:
        # If topology not provided, derive from telemetry metadata
        # Note: Topology helps flow conservation; we derive a best-effort map when absent.
        for if_id, data in telemetry.items():
            r = data.get('local_router')
            if r is not None:
                router_ifaces.setdefault(r, []).append(if_id)

    for router, if_list in router_ifaces.items():
        # Consider only interfaces present in telemetry
        interfaces = [i for i in if_list if i in interim]
        if not interfaces:
            continue

        # Compute sums over "up" interfaces
        sum_tx = 0.0
        sum_rx = 0.0
        tx_conf_acc = 0.0
        rx_conf_acc = 0.0
        up_count_tx = 0
        up_count_rx = 0
        for i in interfaces:
            if interim[i]['status'] == 'up':
                sum_tx += max(0.0, interim[i]['tx'])
                sum_rx += max(0.0, interim[i]['rx'])
                tx_conf_acc += interim[i]['tx_conf']
                rx_conf_acc += interim[i]['rx_conf']
                up_count_tx += 1
                up_count_rx += 1

        if up_count_tx == 0 or up_count_rx == 0:
            continue

        # Evaluate flow imbalance
        imbalance = rel_diff(sum_tx, sum_rx)
        if imbalance <= HARDENING_THRESHOLD * 2:
            # Within tolerance; no router-level scaling needed
            continue

        avg_tx_conf = tx_conf_acc / max(1, up_count_tx)
        avg_rx_conf = rx_conf_acc / max(1, up_count_rx)

        # Decide which direction to scale: scale the less trusted direction
        scale_rx = avg_tx_conf >= avg_rx_conf  # if TX more trusted, scale RX to match TX
        if scale_rx and sum_rx > 0.0:
            s = sum_tx / sum_rx
        elif (not scale_rx) and sum_tx > 0.0:
            s = sum_rx / sum_tx
        else:
            s = 1.0

        # Bound scaling to avoid extreme corrections
        s_bounded = max(0.5, min(2.0, s))

        # Apply scaling and update confidences, while propagating across the link to preserve symmetry (R3)
        for i in interfaces:
            if interim[i]['status'] != 'up':
                continue
            if scale_rx:
                old_val = interim[i]['rx']
                new_val = old_val * s_bounded
                interim[i]['rx'] = new_val
                # Confidence drops with imbalance, scaling magnitude, and per-interface change
                delta_rel = rel_diff(old_val, new_val)
                penalty = 1.0 - min(1.0, imbalance + abs(1.0 - s_bounded) + 0.5 * delta_rel)
                interim[i]['rx_conf'] = clamp(min(interim[i]['rx_conf'], penalty))
                # Symmetry propagation to peer's TX if peer is up on a different router
                peer = peer_of.get(i)
                if peer and (peer in interim) and interim[peer].get('status') == 'up' and interim[peer].get('local_router') != router:
                    peer_old = interim[peer]['tx']
                    interim[peer]['tx'] = new_val
                    # Peer confidence penalty considers same router penalty and its own relative change
                    peer_delta = rel_diff(peer_old, new_val)
                    peer_penalty = 1.0 - min(1.0, imbalance + abs(1.0 - s_bounded) + 0.5 * peer_delta)
                    interim[peer]['tx_conf'] = clamp(min(interim[peer]['tx_conf'], peer_penalty))
            else:
                old_val = interim[i]['tx']
                new_val = old_val * s_bounded
                interim[i]['tx'] = new_val
                delta_rel = rel_diff(old_val, new_val)
                penalty = 1.0 - min(1.0, imbalance + abs(1.0 - s_bounded) + 0.5 * delta_rel)
                interim[i]['tx_conf'] = clamp(min(interim[i]['tx_conf'], penalty))
                # Symmetry propagation to peer's RX if peer is up on a different router
                peer = peer_of.get(i)
                if peer and (peer in interim) and interim[peer].get('status') == 'up' and interim[peer].get('local_router') != router:
                    peer_old = interim[peer]['rx']
                    interim[peer]['rx'] = new_val
                    peer_delta = rel_diff(peer_old, new_val)
                    peer_penalty = 1.0 - min(1.0, imbalance + abs(1.0 - s_bounded) + 0.5 * peer_delta)
                    interim[peer]['rx_conf'] = clamp(min(interim[peer]['rx_conf'], peer_penalty))

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
    TOL_PAIR = HARDENING_THRESHOLD * 1.5
    TOL_ROUTER = HARDENING_THRESHOLD * 2.0

    for if_id, r in interim.items():
        router = r.get('local_router')
        # Peer lookup when available
        try:
            peer = peer_of.get(if_id)
        except NameError:
            peer = None

        status_comp = clamp(r.get('status_conf', 0.8))
        resolved_status = r.get('status', 'unknown')

        if peer and interim.get(peer, {}).get('status') == resolved_status:
            res_fwd = rel_diff(r['tx'], interim[peer]['rx'])
            res_rev = rel_diff(r['rx'], interim[peer]['tx'])
            pair_comp_tx = conf_from_residual(res_fwd, TOL_PAIR)
            pair_comp_rx = conf_from_residual(res_rev, TOL_PAIR)
        else:
            pair_comp_tx = 0.55
            pair_comp_rx = 0.55

        router_imb = router_final_imbalance.get(router, 0.0)
        router_comp = conf_from_residual(router_imb, TOL_ROUTER)

        base_tx_conf = w_pair * pair_comp_tx + w_router * router_comp + w_status * status_comp
        base_rx_conf = w_pair * pair_comp_rx + w_router * router_comp + w_status * status_comp

        # Change penalty to discourage overconfidence on large edits
        delta_tx_rel = rel_diff(r['orig_tx'], r['tx'])
        delta_rx_rel = rel_diff(r['orig_rx'], r['rx'])
        pen_tx = max(0.0, delta_tx_rel - HARDENING_THRESHOLD)
        pen_rx = max(0.0, delta_rx_rel - HARDENING_THRESHOLD)
        CHANGE_PENALTY_WEIGHT = 0.5
        final_tx_conf = clamp(base_tx_conf * (1.0 - CHANGE_PENALTY_WEIGHT * pen_tx))
        final_rx_conf = clamp(base_rx_conf * (1.0 - CHANGE_PENALTY_WEIGHT * pen_rx))

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