# EVOLVE-BLOCK-START
"""
Network telemetry repair algorithm with pairwise hardening and router-aware
confidence calibration (Hodor-inspired).

Key invariants:
- Link Symmetry (R3): my_tx ≈ their_rx, my_rx ≈ their_tx
- Flow Conservation (R1): Σ incoming = Σ outgoing at each router
- Interface Consistency: status aligned across link pairs

Parameters:
- HARDENING_THRESHOLD (τh): 0.02 (2%) tolerance for symmetry differences
- EPS: 1e-6 numeric floor
- ROUTER_PENALTY_WEIGHT: 0.5 weight for combining local/remote router residuals
- STATUS_PENALTY_BLEND: 0.25 mild scaling of status confidence by router penalty
"""
from typing import Dict, Any, Tuple, List


def repair_network_telemetry(telemetry: Dict[str, Dict[str, Any]], 
                             topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:
    """
    Repair network interface telemetry by detecting and correcting inconsistencies.
    
    Uses pairwise link hardening to correct counters and router-level residuals
    to calibrate confidence, following Hodor's three-step validation strategy.
    """
    # Parameters
    HARDENING_THRESHOLD = 0.02  # τh: 2% tolerance
    EPS = 1e-6                  # numeric floor for zero traffic detection
    ROUTER_PENALTY_WEIGHT = 0.5
    STATUS_PENALTY_BLEND = 0.25

    def norm_diff(a: float, b: float) -> float:
        # Relative difference normalized by the larger magnitude (plus floor to avoid zero denom)
        return abs(a - b) / max(a, b, 1.0)

    def has_traffic(d: Dict[str, Any]) -> bool:
        # Detect any traffic beyond EPS in either direction
        return (float(d.get('rx_rate', 0.0)) > EPS) or (float(d.get('tx_rate', 0.0)) > EPS)

    # Build unique link pairs (undirected, canonicalized)
    pairs = {}  # key: tuple(sorted([if1, if2])) -> (ifA, ifB)
    for if_id, data in telemetry.items():
        peer = data.get('connected_to')
        if peer and peer in telemetry:
            key = tuple(sorted([if_id, peer]))
            if key not in pairs:
                pairs[key] = (if_id, peer)

    # First pass: pairwise hardening (link symmetry) and status harmonization
    per_if_repair = {}  # interface_id -> interim repaired values and base confidences

    for _, (a_id, b_id) in pairs.items():
        a = telemetry[a_id]
        b = telemetry[b_id]

        a_status = a.get('interface_status', 'unknown')
        b_status = b.get('interface_status', 'unknown')

        a_rx, a_tx = float(a.get('rx_rate', 0.0) or 0.0), float(a.get('tx_rate', 0.0) or 0.0)
        b_rx, b_tx = float(b.get('rx_rate', 0.0) or 0.0), float(b.get('tx_rate', 0.0) or 0.0)

        a_has = has_traffic(a)
        b_has = has_traffic(b)

        # Decide pair status using interface consistency + traffic evidence
        if a_status == 'down' and b_status == 'down':
            pair_status = 'down'
        elif a_status == 'up' and b_status == 'up':
            pair_status = 'up'
        else:
            pair_status = 'up' if (a_has or b_has) else 'down'

        # Initialize repaired values with originals and high base confidence
        rep_a_tx, rep_b_rx = a_tx, b_rx
        rep_b_tx, rep_a_rx = b_tx, a_rx
        rx_conf_a = 1.0
        tx_conf_a = 1.0
        rx_conf_b = 1.0
        tx_conf_b = 1.0

        if pair_status == 'down':
            # Enforce no traffic on a down link
            rep_a_tx = rep_b_rx = rep_b_tx = rep_a_rx = 0.0
            base_conf = 0.95 if not (a_has or b_has) else 0.7
            rx_conf_a = tx_conf_a = rx_conf_b = tx_conf_b = base_conf
        else:
            # Link is up: enforce link symmetry with τh tolerance
            # Direction A->B (A.tx vs B.rx)
            diff_ab = norm_diff(a_tx, b_rx)
            if diff_ab > HARDENING_THRESHOLD:
                avg_ab = 0.5 * (a_tx + b_rx)
                rep_a_tx = avg_ab
                rep_b_rx = avg_ab
            conf_ab = max(0.0, 1.0 - diff_ab)
            if diff_ab <= HARDENING_THRESHOLD:
                conf_ab = min(1.0, max(conf_ab, 0.98))
            tx_conf_a = conf_ab
            rx_conf_b = conf_ab

            # Direction B->A (B.tx vs A.rx)
            diff_ba = norm_diff(b_tx, a_rx)
            if diff_ba > HARDENING_THRESHOLD:
                avg_ba = 0.5 * (b_tx + a_rx)
                rep_b_tx = avg_ba
                rep_a_rx = avg_ba
            conf_ba = max(0.0, 1.0 - diff_ba)
            if diff_ba <= HARDENING_THRESHOLD:
                conf_ba = min(1.0, max(conf_ba, 0.98))
            tx_conf_b = conf_ba
            rx_conf_a = conf_ba

        # Status confidence based on agreement and evidence
        if pair_status == 'down':
            if a_status == 'down' and b_status == 'down' and not (a_has or b_has):
                status_conf = 0.98
            else:
                status_conf = 0.7
        else:  # pair_status == 'up'
            if a_status == 'up' and b_status == 'up':
                status_conf = 0.95
            else:
                status_conf = 0.8  # up due to traffic evidence

        per_if_repair[a_id] = {
            'repaired_rx': rep_a_rx,
            'repaired_tx': rep_a_tx,
            'rx_conf': rx_conf_a,
            'tx_conf': tx_conf_a,
            'repaired_status': pair_status,
            'status_conf': status_conf
        }
        per_if_repair[b_id] = {
            'repaired_rx': rep_b_rx,
            'repaired_tx': rep_b_tx,
            'rx_conf': rx_conf_b,
            'tx_conf': tx_conf_b,
            'repaired_status': pair_status,
            'status_conf': status_conf
        }

    # Handle interfaces without a valid peer (dangling or missing peer data)
    for if_id, data in telemetry.items():
        if if_id in per_if_repair:
            continue
        status = data.get('interface_status', 'unknown')
        rx = float(data.get('rx_rate', 0.0) or 0.0)
        tx = float(data.get('tx_rate', 0.0) or 0.0)
        if status == 'down':
            # Enforce no traffic on down interfaces
            per_if_repair[if_id] = {
                'repaired_rx': 0.0,
                'repaired_tx': 0.0,
                'rx_conf': 0.9,
                'tx_conf': 0.9,
                'repaired_status': 'down',
                'status_conf': 0.95
            }
        else:
            # No peer to cross-check; keep values but with moderate confidence.
            per_if_repair[if_id] = {
                'repaired_rx': rx,
                'repaired_tx': tx,
                'rx_conf': 0.6,
                'tx_conf': 0.6,
                'repaired_status': status if status in ('up', 'down') else 'up',
                'status_conf': 0.6
            }

    # Second pass: compute router-level flow conservation residuals using repaired rates
    router_residual: Dict[str, float] = {}
    for router, if_list in topology.items():
        sum_tx = 0.0
        sum_rx = 0.0
        for if_id in if_list:
            if if_id in per_if_repair:
                rep = per_if_repair[if_id]
                sum_tx += float(rep['repaired_tx'])
                sum_rx += float(rep['repaired_rx'])
        resid = abs(sum_tx - sum_rx) / max(sum_tx, sum_rx, 1.0)
        router_residual[router] = resid

    # Final assembly: scale confidences by router residuals (dynamic checking)
    result: Dict[str, Dict[str, Tuple]] = {}
    for if_id, data in telemetry.items():
        repaired = per_if_repair.get(if_id, {})
        repaired_rx = repaired.get('repaired_rx', float(data.get('rx_rate', 0.0) or 0.0))
        repaired_tx = repaired.get('repaired_tx', float(data.get('tx_rate', 0.0) or 0.0))
        repaired_status = repaired.get('repaired_status', data.get('interface_status', 'unknown'))

        # Base confidences from link hardening
        rx_conf = float(repaired.get('rx_conf', 0.6))
        tx_conf = float(repaired.get('tx_conf', 0.6))
        status_conf = float(repaired.get('status_conf', 0.6))

        # Apply router-based penalty using both local and remote routers when available
        local_router = data.get('local_router')
        remote_router = data.get('remote_router')

        resid_local = router_residual.get(local_router, 0.0)
        resid_remote = router_residual.get(remote_router, 0.0)

        penalty = 1.0 - ROUTER_PENALTY_WEIGHT * (resid_local + resid_remote)
        penalty = max(0.0, min(1.0, penalty))

        rx_conf = max(0.0, min(1.0, rx_conf * penalty))
        tx_conf = max(0.0, min(1.0, tx_conf * penalty))
        # Status confidence is less sensitive to flow residuals; apply mild scaling
        status_conf = max(0.0, min(1.0, status_conf * (0.75 + STATUS_PENALTY_BLEND * penalty)))

        repaired_data: Dict[str, Any] = {}
        # Store repaired values with confidence scores
        rx_orig = float(data.get('rx_rate', 0.0) or 0.0)
        tx_orig = float(data.get('tx_rate', 0.0) or 0.0)
        status_orig = data.get('interface_status', 'unknown')

        repaired_data['rx_rate'] = (rx_orig, repaired_rx, rx_conf)
        repaired_data['tx_rate'] = (tx_orig, repaired_tx, tx_conf)
        repaired_data['interface_status'] = (status_orig, repaired_status, status_conf)

        # Copy metadata unchanged
        repaired_data['connected_to'] = data.get('connected_to')
        repaired_data['local_router'] = local_router
        repaired_data['remote_router'] = remote_router

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

