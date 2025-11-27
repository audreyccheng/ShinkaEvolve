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
    # Magnitude-aware thresholds per Hodor guidance
    HARDENING_THRESHOLD = 0.02  # ~2%
    LOW_RATE_THRESHOLD = 10.0   # Mbps; relax threshold for tiny flows
    LOW_RATE_THRESHOLD_VAL = 0.05  # 5% when small
    ABS_DIFF_GUARD = 0.5       # Mbps; require for any change

    def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
        return max(lo, min(hi, x))

    def norm_diff(a: float, b: float) -> float:
        return abs(a - b) / max(1.0, a, b)

    # Build peer mapping for quick access
    peers: Dict[str, str] = {}
    for if_id, data in telemetry.items():
        peer = data.get('connected_to')
        peers[if_id] = peer if peer in telemetry else None

    # Precompute forced-down interfaces (propagate across a link)
    forced_down_conf: Dict[str, float] = {}
    for if_id, data in telemetry.items():
        status = data.get('interface_status', 'unknown')
        peer_id = peers.get(if_id)
        if status == 'down':
            # If both sides report down: high confidence; else moderate
            both_down = False
            if peer_id and telemetry.get(peer_id, {}).get('interface_status', 'unknown') == 'down':
                both_down = True
            conf = 0.95 if both_down else 0.7
            forced_down_conf[if_id] = conf
            if peer_id:
                forced_down_conf[peer_id] = conf
        else:
            # If peer reports down, this side must also be effectively down
            if peer_id and telemetry.get(peer_id, {}).get('interface_status', 'unknown') == 'down':
                conf = 0.7
                forced_down_conf[if_id] = conf
                forced_down_conf[peer_id] = max(conf, forced_down_conf.get(peer_id, 0.0))

    # First pass: compute link symmetry stats (magnitude-aware)
    link_stats = {}
    for interface_id, data in telemetry.items():
        connected_to = data.get('connected_to')
        if not connected_to or connected_to not in telemetry:
            continue
        rx_rate = float(data.get('rx_rate', 0.0))
        tx_rate = float(data.get('tx_rate', 0.0))
        peer_data = telemetry[connected_to]
        peer_rx = float(peer_data.get('rx_rate', 0.0))
        peer_tx = float(peer_data.get('tx_rate', 0.0))

        max_tx_rx = max(1.0, tx_rate, peer_rx)
        max_rx_tx = max(1.0, rx_rate, peer_tx)
        tx_rx_diff = abs(tx_rate - peer_rx) / max_tx_rx
        rx_tx_diff = abs(rx_rate - peer_tx) / max_rx_tx

        # Dynamic thresholds
        th_tx_rx = LOW_RATE_THRESHOLD_VAL if max(tx_rate, peer_rx) < LOW_RATE_THRESHOLD else HARDENING_THRESHOLD
        th_rx_tx = LOW_RATE_THRESHOLD_VAL if max(rx_rate, peer_tx) < LOW_RATE_THRESHOLD else HARDENING_THRESHOLD

        link_stats[interface_id] = {
            'tx_rx_diff': tx_rx_diff,
            'rx_tx_diff': rx_tx_diff,
            'peer_rx': peer_rx,
            'peer_tx': peer_tx,
            'th_tx_rx': th_tx_rx,
            'th_rx_tx': th_rx_tx,
            'abs_tx_rx_diff': abs(tx_rate - peer_rx),
            'abs_rx_tx_diff': abs(rx_rate - peer_tx),
            'max_tx_rx': max(tx_rate, peer_rx),
            'max_rx_tx': max(rx_rate, peer_tx),
        }

    # Second pass: repair using redundant signals with magnitude-aware logic
    # Store interim results to apply router residual confidence later
    interim: Dict[str, Dict[str, Any]] = {}

    for interface_id, data in telemetry.items():
        interface_status = data.get('interface_status', 'unknown')
        rx_rate = float(data.get('rx_rate', 0.0))
        tx_rate = float(data.get('tx_rate', 0.0))
        connected_to = data.get('connected_to')

        # Defaults: identity repair, moderate-high confidence
        repaired_rx = rx_rate
        repaired_tx = tx_rate
        repaired_status = interface_status
        rx_confidence = 0.95
        tx_confidence = 0.95
        status_confidence = 0.95

        # Forced down takes precedence (paired consistency)
        if interface_id in forced_down_conf:
            conf_down = forced_down_conf[interface_id]
            repaired_status = 'down'
            repaired_rx = 0.0
            repaired_tx = 0.0
            rx_confidence = conf_down
            tx_confidence = conf_down
            status_confidence = conf_down
        else:
            # Link symmetry hardening if peer exists
            if interface_id in link_stats:
                stats = link_stats[interface_id]
                # RX side: my_rx should match peer_tx
                d_rx = stats['rx_tx_diff']
                th_rx = stats['th_rx_tx']
                abs_d_rx = stats['abs_rx_tx_diff']
                max_rx = stats['max_rx_tx']
                if d_rx > th_rx and abs_d_rx > ABS_DIFF_GUARD:
                    repaired_rx = stats['peer_tx']
                    rx_confidence = clamp(1.0 - d_rx)
                else:
                    # Within tolerance: set strong confidence floor
                    if max_rx >= 10.0 and d_rx <= 0.005:
                        rx_confidence = max(rx_confidence, 0.99)
                    else:
                        rx_confidence = max(rx_confidence, 0.97 if max_rx < 10.0 else 0.98)

                # TX side: my_tx should match peer_rx
                d_tx = stats['tx_rx_diff']
                th_tx = stats['th_tx_rx']
                abs_d_tx = stats['abs_tx_rx_diff']
                max_tx = stats['max_tx_rx']
                if d_tx > th_tx and abs_d_tx > ABS_DIFF_GUARD:
                    repaired_tx = stats['peer_rx']
                    tx_confidence = clamp(1.0 - d_tx)
                else:
                    if max_tx >= 10.0 and d_tx <= 0.005:
                        tx_confidence = max(tx_confidence, 0.99)
                    else:
                        tx_confidence = max(tx_confidence, 0.97 if max_tx < 10.0 else 0.98)
            else:
                # No redundancy: keep but slightly lower confidence
                rx_confidence = max(rx_confidence, 0.9)
                tx_confidence = max(tx_confidence, 0.9)

            # Status mismatch (non-down) implies uncertainty
            if connected_to and connected_to in telemetry:
                peer_status = telemetry[connected_to].get('interface_status', 'unknown')
                if interface_status != 'down' and peer_status != 'down' and interface_status != peer_status:
                    status_confidence = min(status_confidence, 0.6)

        # Assemble interim with metadata
        interim[interface_id] = {
            'repaired_rx': repaired_rx,
            'repaired_tx': repaired_tx,
            'rx_conf': clamp(rx_confidence),
            'tx_conf': clamp(tx_confidence),
            'repaired_status': repaired_status,
            'status_conf': clamp(status_confidence),
            'connected_to': connected_to,
            'local_router': data.get('local_router'),
            'remote_router': data.get('remote_router'),
            'orig_rx': rx_rate,
            'orig_tx': tx_rate,
            'orig_status': interface_status,
        }

    # Build router->interfaces mapping using topology and local_router fields
    router_ifaces: Dict[str, List[str]] = {r: list(if_list) for r, if_list in topology.items()}
    for if_id, data in telemetry.items():
        lr = data.get('local_router')
        if lr:
            router_ifaces.setdefault(lr, [])
            if if_id not in router_ifaces[lr]:
                router_ifaces[lr].append(if_id)

    # Compute router-level residual mismatches from repaired counters
    router_mismatch: Dict[str, float] = {}
    router_totals: Dict[str, Tuple[float, float]] = {}
    for router, if_list in router_ifaces.items():
        sum_tx = 0.0
        sum_rx = 0.0
        for if_id in if_list:
            if if_id in interim:
                sum_tx += float(interim[if_id]['repaired_tx'])
                sum_rx += float(interim[if_id]['repaired_rx'])
        denom = max(1.0, sum_tx, sum_rx)
        router_mismatch[router] = abs(sum_tx - sum_rx) / denom
        router_totals[router] = (sum_tx, sum_rx)

    # Targeted micro-adjustments on dangling interfaces that dominate imbalance
    RESID_TRIGGER = 0.10       # trigger micro-adjustment at >=10% residual
    SHARE_TRIGGER = 0.50       # interface must contribute >=50% of direction
    MAX_ADJUST_FRAC = 0.10     # cap change to 10% of that interface counter
    ALPHA = 0.50               # nudge by 50% of router residual at most

    for router, if_list in router_ifaces.items():
        sum_tx, sum_rx = router_totals.get(router, (0.0, 0.0))
        denom = max(1.0, sum_tx, sum_rx)
        delta = sum_tx - sum_rx
        resid = abs(delta) / denom
        if resid < RESID_TRIGGER or not if_list:
            continue

        # Candidates: unpaired, up interfaces only (no redundant peer to trust)
        candidates = [
            if_id for if_id in if_list
            if if_id in interim
            and interim[if_id]['repaired_status'] != 'down'
            and (peers.get(if_id) is None)
        ]
        if not candidates:
            continue

        if delta > 0.0:
            # Too much TX at router: reduce TX on dominant dangling interface
            best = max(candidates, key=lambda iid: float(interim[iid]['repaired_tx']))
            tx_val = float(interim[best]['repaired_tx'])
            share = tx_val / max(1.0, sum_tx)
            if tx_val > 0.0 and share >= SHARE_TRIGGER:
                adjust = min(delta * ALPHA, tx_val * MAX_ADJUST_FRAC)
                adjust = max(0.0, adjust)
                interim[best]['repaired_tx'] = max(0.0, tx_val - adjust)
                # Lower confidence to reflect heuristic adjustment
                interim[best]['tx_conf'] = min(float(interim[best]['tx_conf']), 0.65)
        elif delta < 0.0:
            # Too much RX at router: reduce RX on dominant dangling interface
            best = max(candidates, key=lambda iid: float(interim[iid]['repaired_rx']))
            rx_val = float(interim[best]['repaired_rx'])
            share = rx_val / max(1.0, sum_rx)
            if rx_val > 0.0 and share >= SHARE_TRIGGER:
                adjust = min(abs(delta) * ALPHA, rx_val * MAX_ADJUST_FRAC)
                adjust = max(0.0, adjust)
                interim[best]['repaired_rx'] = max(0.0, rx_val - adjust)
                interim[best]['rx_conf'] = min(float(interim[best]['rx_conf']), 0.65)

    # Recompute router-level residual mismatches after micro-adjustments
    router_mismatch2: Dict[str, float] = {}
    for router, if_list in router_ifaces.items():
        sum_tx = 0.0
        sum_rx = 0.0
        for if_id in if_list:
            if if_id in interim:
                sum_tx += float(interim[if_id]['repaired_tx'])
                sum_rx += float(interim[if_id]['repaired_rx'])
        denom = max(1.0, sum_tx, sum_rx)
        router_mismatch2[router] = abs(sum_tx - sum_rx) / denom

    # Direction-aware confidence attenuation based on updated residuals
    for if_id, item in interim.items():
        lr = item.get('local_router')
        rr = item.get('remote_router')
        resid_local = router_mismatch2.get(lr, 0.0)
        resid_remote = router_mismatch2.get(rr, 0.0)

        penalty_tx = clamp(1.0 - (0.6 * resid_local + 0.4 * resid_remote), 0.5, 1.0)
        penalty_rx = clamp(1.0 - (0.6 * resid_remote + 0.4 * resid_local), 0.5, 1.0)

        item['rx_conf'] = clamp(item['rx_conf'] * penalty_rx)
        item['tx_conf'] = clamp(item['tx_conf'] * penalty_tx)
        # Slight status confidence reduction under high residuals
        item['status_conf'] = clamp(item['status_conf'] * (1.0 - 0.3 * max(resid_local, resid_remote)))

    # Final assembly
    result: Dict[str, Dict[str, Tuple]] = {}
    for if_id, item in interim.items():
        repaired_status = item['repaired_status']
        repaired_rx = item['repaired_rx']
        repaired_tx = item['repaired_tx']
        # Enforce zero rates if repaired status is down
        if repaired_status == 'down':
            repaired_rx = 0.0
            repaired_tx = 0.0

        out = {
            'rx_rate': (item['orig_rx'], repaired_rx, item['rx_conf']),
            'tx_rate': (item['orig_tx'], repaired_tx, item['tx_conf']),
            'interface_status': (item['orig_status'], repaired_status, item['status_conf']),
            'connected_to': item.get('connected_to'),
            'local_router': item.get('local_router'),
            'remote_router': item.get('remote_router'),
        }
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