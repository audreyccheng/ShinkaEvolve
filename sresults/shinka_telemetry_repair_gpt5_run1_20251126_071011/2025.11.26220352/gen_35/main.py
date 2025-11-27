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
    """

    # Tolerances and caps
    HARDENING_THRESHOLD = 0.02           # ~2% timing tolerance
    TRAFFIC_EVIDENCE_MIN = 0.5           # Mbps, evidence of link up if statuses disagree
    MAX_ROUTER_ADJ_FRAC = 0.35           # per-interface adjustment cap at router balancing (±35%)
    EPS = 1e-9

    def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
        return max(lo, min(hi, x))

    def rel_diff(a: float, b: float) -> float:
        denom = max(abs(a), abs(b), 1e-9)
        return abs(a - b) / denom

    def conf_from_residual(residual: float, tol: float) -> float:
        denom = max(tol * 5.0, 1e-9)
        return clamp(1.0 - residual / denom)

    # Build connected pairs and peer map
    pairs: List[Tuple[str, str]] = []
    seen = set()
    for if_id, data in telemetry.items():
        peer = data.get('connected_to')
        if peer and peer in telemetry:
            key = tuple(sorted([if_id, peer]))
            if key not in seen:
                seen.add(key)
                pairs.append((key[0], key[1]))
    peer_of: Dict[str, str] = {}
    paired_ids = set()
    for a, b in pairs:
        peer_of[a] = b
        peer_of[b] = a
        paired_ids.add(a); paired_ids.add(b)

    # Interim store for repaired values and confidences
    interim: Dict[str, Dict[str, Any]] = {}
    for if_id, data in telemetry.items():
        rx0 = float(data.get('rx_rate', 0.0))
        tx0 = float(data.get('tx_rate', 0.0))
        st0 = data.get('interface_status', 'unknown')
        interim[if_id] = {
            'orig_rx': rx0, 'orig_tx': tx0, 'orig_status': st0,
            'rx': rx0, 'tx': tx0, 'status': st0,
            'rx_conf': 1.0, 'tx_conf': 1.0, 'status_conf': 1.0,
            'connected_to': data.get('connected_to'),
            'local_router': data.get('local_router'),
            'remote_router': data.get('remote_router'),
        }

    # Pair-level status resolution and symmetry hardening (R3)
    for a_id, b_id in pairs:
        a = telemetry[a_id]; b = telemetry[b_id]
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
            if max_traffic > TRAFFIC_EVIDENCE_MIN:
                resolved_status = 'up'
                status_conf = 0.85
            else:
                resolved_status = 'down'
                status_conf = 0.75

        # Apply status to both ends
        for ifid in (a_id, b_id):
            interim[ifid]['status'] = resolved_status
            interim[ifid]['status_conf'] = min(interim[ifid]['status_conf'], status_conf) if interim[ifid]['status_conf'] else status_conf

        # If link is down, enforce zero traffic and confidence based on original values
        if resolved_status == 'down':
            for (ifid, rx0i, tx0i) in [(a_id, a_rx, a_tx), (b_id, b_rx, b_tx)]:
                interim[ifid]['rx'] = 0.0
                interim[ifid]['tx'] = 0.0
                interim[ifid]['rx_conf'] = 0.9 if rx0i <= TRAFFIC_EVIDENCE_MIN else 0.3
                interim[ifid]['tx_conf'] = 0.9 if tx0i <= TRAFFIC_EVIDENCE_MIN else 0.3
            continue

        # Link up: enforce symmetry per direction
        # Forward: a.tx ≈ b.rx
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

        # Reverse: a.rx ≈ b.tx
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

    # Down implies zero for unpaired interfaces as well
    for if_id, r in interim.items():
        if if_id not in paired_ids and r.get('status') == 'down':
            rx0 = r['rx']; tx0 = r['tx']
            r['rx'] = 0.0; r['tx'] = 0.0
            r['rx_conf'] = 0.9 if rx0 <= TRAFFIC_EVIDENCE_MIN else 0.3
            r['tx_conf'] = 0.9 if tx0 <= TRAFFIC_EVIDENCE_MIN else 0.3

    # Build router->interfaces mapping using topology if available; derive otherwise
    router_ifaces: Dict[str, List[str]] = {}
    if topology:
        router_ifaces = {r: [i for i in if_list if i in interim] for r, if_list in topology.items()}
    else:
        # Use telemetry metadata when explicit topology is not provided
        for if_id, data in telemetry.items():
            rtr = data.get('local_router')
            if rtr is not None:
                router_ifaces.setdefault(rtr, []).append(if_id)

    # Router-level flow conservation (R1) with bounded scaling
    for router, if_list in router_ifaces.items():
        up_ifaces = [i for i in if_list if interim[i].get('status') == 'up']
        if not up_ifaces:
            continue
        sum_tx = sum(max(0.0, interim[i]['tx']) for i in up_ifaces)
        sum_rx = sum(max(0.0, interim[i]['rx']) for i in up_ifaces)
        imb = rel_diff(sum_tx, sum_rx)
        if imb <= HARDENING_THRESHOLD * 2.0:
            continue

        # Scale the direction with the smaller total toward the larger to satisfy conservation
        scale_rx = sum_tx > sum_rx  # if TX larger, scale RX upward; else scale TX upward
        if scale_rx and sum_rx > 0.0:
            s = sum_tx / sum_rx
        elif (not scale_rx) and sum_tx > 0.0:
            s = sum_rx / sum_tx
        else:
            s = 1.0
        s_bounded = max(0.5, min(2.0, s))

        for i in up_ifaces:
            if scale_rx:
                old_v = max(0.0, interim[i]['rx'])
                proposed = old_v * s_bounded
                delta = proposed - old_v
                cap = MAX_ROUTER_ADJ_FRAC * max(old_v, 1.0)
                delta = max(-cap, min(cap, delta))
                new_v = max(0.0, old_v + delta)
                interim[i]['rx'] = new_v
                # Confidence penalty grows with imbalance, scaling magnitude and relative change
                rel = abs(delta) / max(old_v, 1e-9)
                interim[i]['rx_conf'] = clamp(min(interim[i]['rx_conf'], 1.0 - min(1.0, imb + 0.5 * rel + 0.5 * abs(1.0 - s_bounded))))
            else:
                old_v = max(0.0, interim[i]['tx'])
                proposed = old_v * s_bounded
                delta = proposed - old_v
                cap = MAX_ROUTER_ADJ_FRAC * max(old_v, 1.0)
                delta = max(-cap, min(cap, delta))
                new_v = max(0.0, old_v + delta)
                interim[i]['tx'] = new_v
                rel = abs(delta) / max(old_v, 1e-9)
                interim[i]['tx_conf'] = clamp(min(interim[i]['tx_conf'], 1.0 - min(1.0, imb + 0.5 * rel + 0.5 * abs(1.0 - s_bounded))))

    # Slight status confidence adjustment for idle-but-up links
    for if_id, r in interim.items():
        if r.get('status') == 'up':
            if r['rx'] <= TRAFFIC_EVIDENCE_MIN and r['tx'] <= TRAFFIC_EVIDENCE_MIN:
                r['status_conf'] = clamp(r['status_conf'] * 0.9)
        elif r.get('status') == 'down':
            if r['rx'] > TRAFFIC_EVIDENCE_MIN or r['tx'] > TRAFFIC_EVIDENCE_MIN:
                r['status_conf'] = clamp(min(r['status_conf'], 0.3))

    # Assemble final result with (original, repaired, confidence) tuples and unchanged metadata
    result: Dict[str, Dict[str, Tuple]] = {}
    for if_id, r in interim.items():
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
