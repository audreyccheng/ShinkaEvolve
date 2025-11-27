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
    Repair network interface telemetry by detecting and correcting inconsistencies using:
    - Link symmetry hardening (R3): fuse my_tx with their_rx and my_rx with their_tx
    - Flow conservation awareness (R1): use router-level mismatch to calibrate confidence
    - Interface consistency: down interfaces cannot send/receive; paired statuses should match
    """
    HARDENING_THRESHOLD = 0.02  # ~2% timing tolerance from research

    def clamp(x: float, lo: float = 0.0, hi: float = 1.0) -> float:
        return max(lo, min(hi, x))

    # Precompute peer mapping
    peers: Dict[str, str] = {}
    for if_id, data in telemetry.items():
        peer = data.get('connected_to')
        peers[if_id] = peer if peer in telemetry else None

    # Initialize fused per-interface data with originals
    fused: Dict[str, Dict[str, Any]] = {}
    for if_id, data in telemetry.items():
        fused[if_id] = {
            'rx': float(data.get('rx_rate', 0.0)),
            'tx': float(data.get('tx_rate', 0.0)),
            'rx_conf': 0.8,  # default before seeing redundancy
            'tx_conf': 0.8,
            'status': data.get('interface_status', 'up'),
            'status_conf': 1.0
        }

    visited = set()

    # Helper to compute relative difference safely
    def rel_diff(a: float, b: float) -> float:
        return abs(a - b) / max(1.0, a, b)

    # Link-level hardening
    for a_id, a_data in telemetry.items():
        if a_id in visited:
            continue
        b_id = peers.get(a_id)
        if not b_id:
            continue
        if b_id in visited:
            continue
        if b_id not in telemetry:
            continue

        visited.add(a_id)
        visited.add(b_id)

        b_data = telemetry[b_id]

        sa = a_data.get('interface_status', 'up')
        sb = b_data.get('interface_status', 'up')

        a_tx = float(a_data.get('tx_rate', 0.0))
        a_rx = float(a_data.get('rx_rate', 0.0))
        b_tx = float(b_data.get('tx_rate', 0.0))
        b_rx = float(b_data.get('rx_rate', 0.0))

        # If either side reports down, enforce down on both and zero rates
        if sa == 'down' or sb == 'down':
            both_down = (sa == 'down' and sb == 'down')
            conf_down = 0.95 if both_down else 0.7
            fused[a_id]['status'] = 'down'
            fused[b_id]['status'] = 'down'
            fused[a_id]['status_conf'] = conf_down
            fused[b_id]['status_conf'] = conf_down
            fused[a_id]['rx'] = 0.0
            fused[a_id]['tx'] = 0.0
            fused[b_id]['rx'] = 0.0
            fused[b_id]['tx'] = 0.0
            fused[a_id]['rx_conf'] = conf_down
            fused[a_id]['tx_conf'] = conf_down
            fused[b_id]['rx_conf'] = conf_down
            fused[b_id]['tx_conf'] = conf_down
            continue

        # Both up: compute disagreements per direction
        diff_ab = rel_diff(a_tx, b_rx)  # a->b traffic
        diff_ba = rel_diff(b_tx, a_rx)  # b->a traffic

        # Device trust: average disagreement across both directions
        err_a = 0.5 * (rel_diff(a_tx, b_rx) + rel_diff(a_rx, b_tx))
        err_b = 0.5 * (rel_diff(b_tx, a_rx) + rel_diff(b_rx, a_tx))

        # Direction a->b: choose strategy
        if diff_ab <= HARDENING_THRESHOLD:
            fused[a_id]['tx'] = a_tx
            fused[b_id]['rx'] = b_rx
        else:
            # pick the more trustworthy side for this direction
            if err_a < err_b:
                fused[a_id]['tx'] = a_tx
                fused[b_id]['rx'] = a_tx
            else:
                fused[a_id]['tx'] = b_rx
                fused[b_id]['rx'] = b_rx
        # Confidence for direction a->b
        conf_ab_base = clamp(1.0 - diff_ab)
        trust_boost_ab = clamp(1.0 - min(err_a, err_b))
        conf_ab = clamp(conf_ab_base * (0.7 + 0.3 * trust_boost_ab))
        fused[a_id]['tx_conf'] = conf_ab
        fused[b_id]['rx_conf'] = conf_ab

        # Direction b->a
        if diff_ba <= HARDENING_THRESHOLD:
            fused[b_id]['tx'] = b_tx
            fused[a_id]['rx'] = a_rx
        else:
            if err_b < err_a:
                fused[b_id]['tx'] = b_tx
                fused[a_id]['rx'] = b_tx
            else:
                fused[b_id]['tx'] = a_rx
                fused[a_id]['rx'] = a_rx
        # Confidence for direction b->a
        conf_ba_base = clamp(1.0 - diff_ba)
        trust_boost_ba = clamp(1.0 - min(err_a, err_b))
        conf_ba = clamp(conf_ba_base * (0.7 + 0.3 * trust_boost_ba))
        fused[b_id]['tx_conf'] = conf_ba
        fused[a_id]['rx_conf'] = conf_ba

        # Status for both sides consistent up
        fused[a_id]['status'] = 'up'
        fused[b_id]['status'] = 'up'
        fused[a_id]['status_conf'] = 1.0
        fused[b_id]['status_conf'] = 1.0

    # Handle interfaces without valid peers
    for if_id, data in telemetry.items():
        if peers.get(if_id):
            continue
        status = data.get('interface_status', 'up')
        if status == 'down':
            # Cannot send/receive if down
            fused[if_id]['rx'] = 0.0
            fused[if_id]['tx'] = 0.0
            fused[if_id]['rx_conf'] = 0.9
            fused[if_id]['tx_conf'] = 0.9
            fused[if_id]['status'] = 'down'
            fused[if_id]['status_conf'] = 0.9
        else:
            # No redundancy: keep original with moderate confidence
            fused[if_id]['rx'] = float(data.get('rx_rate', 0.0))
            fused[if_id]['tx'] = float(data.get('tx_rate', 0.0))
            fused[if_id]['rx_conf'] = 0.75
            fused[if_id]['tx_conf'] = 0.75
            fused[if_id]['status'] = status
            fused[if_id]['status_conf'] = 0.85

    # Build router->interfaces mapping, prefer topology but include any missing via local_router
    router_ifaces: Dict[str, List[str]] = {r: list(if_list) for r, if_list in topology.items()}
    for if_id, data in telemetry.items():
        lr = data.get('local_router')
        if not lr:
            continue
        if lr not in router_ifaces:
            router_ifaces[lr] = []
        if if_id not in router_ifaces[lr]:
            router_ifaces[lr].append(if_id)

    # Compute router-level mismatch to adjust confidence
    router_mismatch: Dict[str, float] = {}
    for router, if_list in router_ifaces.items():
        sum_tx = 0.0
        sum_rx = 0.0
        for if_id in if_list:
            if if_id in fused:
                sum_tx += float(fused[if_id]['tx'])
                sum_rx += float(fused[if_id]['rx'])
        mismatch = 0.0
        denom = max(1.0, sum_tx, sum_rx)
        mismatch = abs(sum_tx - sum_rx) / denom
        router_mismatch[router] = mismatch

    # Apply router mismatch as confidence attenuation
    for if_id, data in telemetry.items():
        lr = data.get('local_router')
        if not lr:
            continue
        mismatch = router_mismatch.get(lr, 0.0)
        # Attenuate up to 50% at high mismatch (~10%+)
        router_factor = clamp(1.0 - 5.0 * mismatch, 0.5, 1.0)
        fused[if_id]['rx_conf'] = clamp(fused[if_id]['rx_conf'] * router_factor)
        fused[if_id]['tx_conf'] = clamp(fused[if_id]['tx_conf'] * router_factor)
        # Status confidence slightly reduced if mismatch is high (indicates potential broader inconsistency)
        fused[if_id]['status_conf'] = clamp(fused[if_id]['status_conf'] * (1.0 - 0.5 * mismatch))

    # Assemble result with original, repaired, and confidence
    result: Dict[str, Dict[str, Tuple]] = {}
    for if_id, data in telemetry.items():
        orig_rx = float(data.get('rx_rate', 0.0))
        orig_tx = float(data.get('tx_rate', 0.0))
        orig_status = data.get('interface_status', 'up')

        repaired_rx = fused[if_id]['rx']
        repaired_tx = fused[if_id]['tx']
        repaired_status = fused[if_id]['status']

        rx_conf = clamp(float(fused[if_id]['rx_conf']))
        tx_conf = clamp(float(fused[if_id]['tx_conf']))
        status_conf = clamp(float(fused[if_id]['status_conf']))

        # If final status is down, ensure rates are zero
        if repaired_status == 'down':
            repaired_rx = 0.0
            repaired_tx = 0.0

        # Prepare output
        out = {
            'rx_rate': (orig_rx, repaired_rx, rx_conf),
            'tx_rate': (orig_tx, repaired_tx, tx_conf),
            'interface_status': (orig_status, repaired_status, status_conf),
            'connected_to': data.get('connected_to'),
            'local_router': data.get('local_router'),
            'remote_router': data.get('remote_router'),
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
