# EVOLVE-BLOCK-START
"""
Network telemetry repair algorithm that detects and corrects inconsistencies
in network interface telemetry data using topology relationships.

Evolved version:
- Robust pair fusion (symmetry hardening)
- Uncertainty-weighted router-level redistribution (flow conservation)
- Post-repair confidence calibration from invariant residuals
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

    # PARAMETERS (tunable)
    HARDENING_THRESHOLD = 0.02            # τh: 2% timing tolerance
    PAIR_STRICT_FACTOR = 1.5              # widens pair tolerance for confidence mapping
    ROUTER_TOL_FACTOR = 2.0               # widens router tolerance for confidence mapping
    TRAFFIC_EVIDENCE_MIN = 0.5            # Mbps threshold to infer link up when statuses disagree
    MAX_ROUTER_ADJ_FRAC = 0.35            # cap per-interface adjustment per iteration (fraction of current value)
    ROUTER_ADJ_ITERS = 2                  # iterations for router-level redistribution
    CHANGE_PENALTY_WEIGHT = 0.5           # weight of confidence penalty for large changes
    EPS = 1e-9

    def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
        return max(lo, min(hi, x))

    def rel_diff(a: float, b: float) -> float:
        denom = max(abs(a), abs(b), EPS)
        return abs(a - b) / denom

    def conf_from_residual(residual: float, tol: float) -> float:
        # Maps residual to confidence; at residual=0 -> 1, at residual=tol -> ~0.9, and decreases smoothly
        # Linear falloff up to 5*tol then floors near 0
        return clamp(1.0 - (residual / max(tol * 5.0, EPS)))

    def build_pairs(tel: Dict[str, Dict[str, Any]]) -> List[Tuple[str, str]]:
        visited = set()
        pairs_local: List[Tuple[str, str]] = []
        for if_id, data in tel.items():
            peer = data.get('connected_to')
            if not peer or peer not in tel:
                continue
            key = tuple(sorted((if_id, peer)))
            if key in visited:
                continue
            visited.add(key)
            pairs_local.append((key[0], key[1]))
        return pairs_local

    # Initialize structures
    result: Dict[str, Dict[str, Tuple]] = {}
    interim: Dict[str, Dict[str, Any]] = {}

    # Build connected pairs and a quick peer map
    pairs = build_pairs(telemetry)
    peer_of: Dict[str, str] = {}
    for a_id, b_id in pairs:
        peer_of[a_id] = b_id
        peer_of[b_id] = a_id

    # Initialize interim records
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
            'orig_rx': float(data.get('rx_rate', 0.0)),
            'orig_tx': float(data.get('tx_rate', 0.0)),
            'orig_status': data.get('interface_status', 'unknown'),
        }

    # Pair-level status resolution and symmetry fusion
    for a_id, b_id in pairs:
        a = telemetry[a_id]
        b = telemetry[b_id]
        a_stat = a.get('interface_status', 'unknown')
        b_stat = b.get('interface_status', 'unknown')
        a_rx, a_tx = float(a.get('rx_rate', 0.0)), float(a.get('tx_rate', 0.0))
        b_rx, b_tx = float(b.get('rx_rate', 0.0)), float(b.get('tx_rate', 0.0))
        max_traffic = max(a_rx, a_tx, b_rx, b_tx)

        # Resolve link status consistency
        if a_stat == b_stat:
            resolved_status = a_stat
            status_conf = 0.95 if resolved_status in ('up', 'down') else 0.7
        else:
            # Traffic evidence rule
            if max_traffic > TRAFFIC_EVIDENCE_MIN:
                resolved_status = 'up'
                status_conf = 0.85
            else:
                resolved_status = 'down'
                status_conf = 0.75

        # Apply status to both ends
        for if_id in (a_id, b_id):
            interim[if_id]['status'] = resolved_status
            # combine conservatively
            prev = interim[if_id]['status_conf']
            interim[if_id]['status_conf'] = min(prev, status_conf) if prev is not None else status_conf

        if resolved_status == 'down':
            # Enforce zero traffic on down links
            for if_id, rx0, tx0 in [(a_id, a_rx, a_tx), (b_id, b_rx, b_tx)]:
                interim[if_id]['rx'] = 0.0
                interim[if_id]['tx'] = 0.0
                # High confidence if already near zero, lower otherwise
                interim[if_id]['rx_conf'] = 0.9 if rx0 <= TRAFFIC_EVIDENCE_MIN else 0.3
                interim[if_id]['tx_conf'] = 0.9 if tx0 <= TRAFFIC_EVIDENCE_MIN else 0.3
            continue  # skip rate hardening for down links

        # Link is up: harden rates using link symmetry (R3)
        # Forward direction: a.tx vs b.rx
        d_fwd = rel_diff(a_tx, b_rx)
        if d_fwd <= HARDENING_THRESHOLD:
            v_fwd = 0.5 * (a_tx + b_rx)
        else:
            # Prefer the peer's counterpart (redundant signal)
            v_fwd = b_rx if b_rx > 0.0 or a_tx == 0.0 else a_tx

        # Reverse direction: a.rx vs b.tx
        d_rev = rel_diff(a_rx, b_tx)
        if d_rev <= HARDENING_THRESHOLD:
            v_rev = 0.5 * (a_rx + b_tx)
        else:
            v_rev = b_tx if b_tx > 0.0 or a_rx == 0.0 else a_rx

        # Apply symmetric hardening
        interim[a_id]['tx'] = v_fwd
        interim[b_id]['rx'] = v_fwd
        interim[a_id]['rx'] = v_rev
        interim[b_id]['tx'] = v_rev

        # Seed confidences based on pair residuals (will be recalibrated later)
        # Seed: higher when values originally agreed better
        interim[a_id]['tx_conf'] = clamp(1.0 - d_fwd)
        interim[b_id]['rx_conf'] = clamp(1.0 - d_fwd)
        interim[a_id]['rx_conf'] = clamp(1.0 - d_rev)
        interim[b_id]['tx_conf'] = clamp(1.0 - d_rev)

    # Router-level dynamic flow conservation (R1) with uncertainty-weighted redistribution
    # Build router_ifaces using provided topology when available; otherwise derive from telemetry
    if topology:
        router_ifaces: Dict[str, List[str]] = {r: [i for i in ifs if i in interim] for r, ifs in topology.items()}
    else:
        # If topology not provided, derive from local_router metadata
        router_ifaces = {}
        for if_id, data in interim.items():
            r = data.get('local_router')
            if r is not None:
                router_ifaces.setdefault(r, []).append(if_id)

    # Precompute seed confidences from pair residuals for weighting
    seed_tx_conf: Dict[str, float] = {}
    seed_rx_conf: Dict[str, float] = {}
    for if_id, r in interim.items():
        peer = peer_of.get(if_id)
        if peer and interim[if_id]['status'] == 'up' and interim[peer]['status'] == 'up':
            # Pair residuals based on current interim values
            res_tx = rel_diff(interim[if_id]['tx'], interim[peer]['rx'])
            res_rx = rel_diff(interim[if_id]['rx'], interim[peer]['tx'])
            seed_tx_conf[if_id] = clamp(1.0 - res_tx)
            seed_rx_conf[if_id] = clamp(1.0 - res_rx)
        else:
            # If no peer, use neutral seeds
            seed_tx_conf[if_id] = 0.6
            seed_rx_conf[if_id] = 0.6

    # Iterative redistribution to reduce router imbalance
    for _ in range(ROUTER_ADJ_ITERS):
        for router, if_list in router_ifaces.items():
            # Consider only interfaces that are up
            up_ifaces = [i for i in if_list if interim.get(i, {}).get('status') == 'up']
            if not up_ifaces:
                continue

            sum_tx = sum(max(0.0, interim[i]['tx']) for i in up_ifaces)
            sum_rx = sum(max(0.0, interim[i]['rx']) for i in up_ifaces)
            imbalance = sum_tx - sum_rx
            # If within tolerance, skip
            if max(sum_tx, sum_rx) <= EPS:
                continue
            rel_imb = abs(imbalance) / max(sum_tx, sum_rx, EPS)
            if rel_imb <= HARDENING_THRESHOLD * ROUTER_TOL_FACTOR:
                continue

            # Determine which side to adjust: adjust the less trusted direction
            avg_tx_seed = sum(seed_tx_conf.get(i, 0.6) for i in up_ifaces) / max(1, len(up_ifaces))
            avg_rx_seed = sum(seed_rx_conf.get(i, 0.6) for i in up_ifaces) / max(1, len(up_ifaces))
            adjust_rx = avg_tx_seed >= avg_rx_seed  # if TX more trusted, adjust RX

            if adjust_rx:
                # Target change in total RX is +imbalance (may be negative)
                weights = []
                for i in up_ifaces:
                    w = max(0.05, 1.0 - seed_rx_conf.get(i, 0.6))  # lower confidence -> larger weight
                    weights.append((i, w))
                sumW = sum(w for _, w in weights) or 1.0
                for i, w in weights:
                    target_delta = imbalance * (w / sumW)
                    # Cap per-interface change
                    cap = MAX_ROUTER_ADJ_FRAC * max(interim[i]['rx'], 1.0)
                    delta = max(-cap, min(cap, target_delta))
                    interim[i]['rx'] = max(0.0, interim[i]['rx'] + delta)
            else:
                # Target change in total TX is -imbalance
                weights = []
                for i in up_ifaces:
                    w = max(0.05, 1.0 - seed_tx_conf.get(i, 0.6))
                    weights.append((i, w))
                sumW = sum(w for _, w in weights) or 1.0
                for i, w in weights:
                    target_delta = (-imbalance) * (w / sumW)
                    cap = MAX_ROUTER_ADJ_FRAC * max(interim[i]['tx'], 1.0)
                    delta = max(-cap, min(cap, target_delta))
                    interim[i]['tx'] = max(0.0, interim[i]['tx'] + delta)

    # Final confidence calibration based on post-repair residuals
    # Compute per-router final imbalance for confidence component
    router_final_imbalance: Dict[str, float] = {}
    for router, if_list in router_ifaces.items():
        up_ifaces = [i for i in if_list if interim.get(i, {}).get('status') == 'up']
        if not up_ifaces:
            router_final_imbalance[router] = 0.0
            continue
        sum_tx = sum(max(0.0, interim[i]['tx']) for i in up_ifaces)
        sum_rx = sum(max(0.0, interim[i]['rx']) for i in up_ifaces)
        router_final_imbalance[router] = rel_diff(sum_tx, sum_rx)

    # Weights for confidence components
    w_pair = 0.6
    w_router = 0.3
    w_status = 0.1
    tol_pair = HARDENING_THRESHOLD * PAIR_STRICT_FACTOR
    tol_router = HARDENING_THRESHOLD * ROUTER_TOL_FACTOR

    for if_id, r in interim.items():
        peer = peer_of.get(if_id)
        router = r.get('local_router')

        # Status confidence: start from previously computed value
        status_comp = clamp(r.get('status_conf', 0.8))
        resolved_status = r.get('status', 'unknown')

        # Pair residual components
        if peer and interim[peer]['status'] == resolved_status:
            # Compute post-repair residuals for both directions
            res_fwd = rel_diff(interim[if_id]['tx'], interim[peer]['rx'])
            res_rev = rel_diff(interim[if_id]['rx'], interim[peer]['tx'])
            pair_comp_tx = conf_from_residual(res_fwd, tol_pair)
            pair_comp_rx = conf_from_residual(res_rev, tol_pair)
        else:
            # No reliable pair information -> neutral confidence from pair
            pair_comp_tx = 0.55
            pair_comp_rx = 0.55

        # Router residual component (same for both directions)
        router_imb = router_final_imbalance.get(router, 0.0)
        router_comp = conf_from_residual(router_imb, tol_router)

        # Aggregate base confidence
        base_tx_conf = w_pair * pair_comp_tx + w_router * router_comp + w_status * status_comp
        base_rx_conf = w_pair * pair_comp_rx + w_router * router_comp + w_status * status_comp

        # Change penalty: large changes without strong agreement => reduce confidence
        delta_tx_rel = rel_diff(r['orig_tx'], r['tx'])
        delta_rx_rel = rel_diff(r['orig_rx'], r['rx'])
        # Only penalize beyond tolerance
        pen_tx = max(0.0, delta_tx_rel - HARDENING_THRESHOLD)
        pen_rx = max(0.0, delta_rx_rel - HARDENING_THRESHOLD)
        final_tx_conf = clamp(base_tx_conf * (1.0 - CHANGE_PENALTY_WEIGHT * pen_tx))
        final_rx_conf = clamp(base_rx_conf * (1.0 - CHANGE_PENALTY_WEIGHT * pen_rx))

        # If link is down, override with down-specific confidence logic
        if resolved_status == 'down':
            final_rx_conf = 0.9 if r['orig_rx'] <= TRAFFIC_EVIDENCE_MIN else 0.3
            final_tx_conf = 0.9 if r['orig_tx'] <= TRAFFIC_EVIDENCE_MIN else 0.3

        interim[if_id]['rx_conf'] = final_rx_conf
        interim[if_id]['tx_conf'] = final_tx_conf

        # Status confidence slight adjustment if status contradicts traffic
        if resolved_status == 'up':
            if (interim[if_id]['rx'] <= TRAFFIC_EVIDENCE_MIN and interim[if_id]['tx'] <= TRAFFIC_EVIDENCE_MIN):
                # Up but no traffic: slightly reduce status confidence
                interim[if_id]['status_conf'] = clamp(interim[if_id]['status_conf'] * 0.9)
        elif resolved_status == 'down':
            if (interim[if_id]['rx'] > TRAFFIC_EVIDENCE_MIN or interim[if_id]['tx'] > TRAFFIC_EVIDENCE_MIN):
                interim[if_id]['status_conf'] = clamp(min(interim[if_id]['status_conf'], 0.3))

    # Assemble final result with (original, repaired, confidence) tuples and unchanged metadata
    result: Dict[str, Dict[str, Tuple]] = {}
    for if_id, r in interim.items():
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
