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

    Strategy inspired by Hodor:
    1) Signal Collection: use redundant bilateral measurements on links.
    2) Signal Hardening: pair-wise hardening via link symmetry (R3) with 2% tolerance.
    3) Dynamic Checking: router-level flow conservation (R1) with guarded, capped scaling.
    Additionally, enforce interface consistency for rates when status is down.

    Confidence calibration:
    - High confidence when redundant signals agree and small/zero corrections.
    - Confidence reduced proportionally to symmetry deviations and applied router-level adjustments.

    Note: We intentionally do not flip interface statuses to preserve status accuracy,
    but we reduce status confidence when peers disagree.
    """
    # Measurement timing tolerance (from Hodor research: ~2%)
    HARDENING_THRESHOLD = 0.02
    ZERO_EPS = 1e-3
    # Soft-zero threshold for Mbps-scale counters
    ZERO_THRESH = 1.0

    def clamp01(x: float) -> float:
        return max(0.0, min(1.0, x))

    def rel_diff(a: float, b: float) -> float:
        return abs(a - b) / max(1.0, abs(a), abs(b))

    def compute_tau_h(v1: float, v2: float) -> float:
        """
        Adaptive symmetry tolerance:
        - 0.015 for high-rate pairs (>=100 Mbps both sides)
        - 0.03 for near-zero/low-rate (either side < 1 Mbps)
        - 0.02 baseline otherwise
        """
        if v1 >= 100.0 and v2 >= 100.0:
            return 0.015
        if v1 < ZERO_THRESH or v2 < ZERO_THRESH:
            return 0.03
        return HARDENING_THRESHOLD

    # Precompute originals and peers
    orig_rx: Dict[str, float] = {}
    orig_tx: Dict[str, float] = {}
    status: Dict[str, str] = {}
    peer_of: Dict[str, str] = {}

    for if_id, data in telemetry.items():
        orig_rx[if_id] = float(data.get('rx_rate', 0.0))
        orig_tx[if_id] = float(data.get('tx_rate', 0.0))
        status[if_id] = data.get('interface_status', 'unknown')
        ct = data.get('connected_to')
        peer_of[if_id] = ct if ct in telemetry else None

    # Initialize hardened values with originals
    hardened_rx: Dict[str, float] = {i: v for i, v in orig_rx.items()}
    hardened_tx: Dict[str, float] = {i: v for i, v in orig_tx.items()}
    conf_rx: Dict[str, float] = {i: 0.7 for i in telemetry}
    conf_tx: Dict[str, float] = {i: 0.7 for i in telemetry}

    processed_pairs = set()

    # Pairwise hardening using link symmetry (R3)
    for a, data in telemetry.items():
        b = peer_of.get(a)
        if not b or (b, a) in processed_pairs or a == b:
            continue
        processed_pairs.add((a, b))

        a_stat = status.get(a, 'unknown')
        b_stat = status.get(b, 'unknown')
        a_up = (a_stat == 'up')
        b_up = (b_stat == 'up')

        a_rx = orig_rx[a]
        a_tx = orig_tx[a]
        b_rx = orig_rx[b]
        b_tx = orig_tx[b]

        # If either side is down: enforce zero on both link directions with high confidence
        if not a_up or not b_up:
            hardened_rx[a] = 0.0
            hardened_tx[a] = 0.0
            hardened_rx[b] = 0.0
            hardened_tx[b] = 0.0
            conf_rx[a] = max(conf_rx[a], 0.85)
            conf_tx[a] = max(conf_tx[a], 0.85)
            conf_rx[b] = max(conf_rx[b], 0.85)
            conf_tx[b] = max(conf_tx[b], 0.85)
            continue

        # Soft-zero stabilization: both ends up and all four directions near zero
        if max(a_rx, a_tx, b_rx, b_tx) < 2.0 * ZERO_THRESH:
            hardened_rx[a] = 0.0
            hardened_tx[a] = 0.0
            hardened_rx[b] = 0.0
            hardened_tx[b] = 0.0
            conf_rx[a] = max(conf_rx[a], 0.95)
            conf_tx[a] = max(conf_tx[a], 0.95)
            conf_rx[b] = max(conf_rx[b], 0.95)
            conf_tx[b] = max(conf_tx[b], 0.95)
            continue

        # Direction 1: a.tx should match b.rx (adaptive tolerance)
        d1 = rel_diff(a_tx, b_rx)
        tau1 = compute_tau_h(a_tx, b_rx)
        if d1 <= tau1:
            v1 = 0.5 * (a_tx + b_rx)
            hardened_tx[a] = max(0.0, v1)
            hardened_rx[b] = max(0.0, v1)
            c1 = clamp01(0.9 + 0.1 * (1.0 - d1 / max(tau1, 1e-12)))
            conf_tx[a] = max(conf_tx[a], c1)
            conf_rx[b] = max(conf_rx[b], c1)
        else:
            # Snap to peer's measurement for strong symmetry
            hardened_tx[a] = max(0.0, b_rx)
            hardened_rx[b] = max(0.0, b_rx)
            c1 = clamp01(1.0 - d1)
            conf_tx[a] = max(conf_tx[a], c1)
            conf_rx[b] = max(conf_rx[b], c1)

        # Direction 2: a.rx should match b.tx (adaptive tolerance)
        d2 = rel_diff(a_rx, b_tx)
        tau2 = compute_tau_h(a_rx, b_tx)
        if d2 <= tau2:
            v2 = 0.5 * (a_rx + b_tx)
            hardened_rx[a] = max(0.0, v2)
            hardened_tx[b] = max(0.0, v2)
            c2 = clamp01(0.9 + 0.1 * (1.0 - d2 / max(tau2, 1e-12)))
            conf_rx[a] = max(conf_rx[a], c2)
            conf_tx[b] = max(conf_tx[b], c2)
        else:
            hardened_rx[a] = max(0.0, b_tx)
            hardened_tx[b] = max(0.0, b_tx)
            c2 = clamp01(1.0 - d2)
            conf_rx[a] = max(conf_rx[a], c2)
            conf_tx[b] = max(conf_tx[b], c2)

    # Unpaired interfaces: keep own values with moderate confidence
    for i, d in telemetry.items():
        if i not in [x for pair in processed_pairs for x in pair]:
            # If interface is down, enforce zero with strong confidence
            if status.get(i) == 'down':
                hardened_rx[i] = 0.0
                hardened_tx[i] = 0.0
                conf_rx[i] = max(conf_rx[i], 0.85)
                conf_tx[i] = max(conf_tx[i], 0.85)
            else:
                # Keep local but acknowledge weaker redundancy
                hardened_rx[i] = max(0.0, orig_rx[i])
                hardened_tx[i] = max(0.0, orig_tx[i])
                conf_rx[i] = max(conf_rx[i], 0.6)
                conf_tx[i] = max(conf_tx[i], 0.6)

    # Track per-direction scaled factors from router adjustments (for later guards)
    scaled_rx_factor = {i: 1.0 for i in telemetry}
    scaled_tx_factor = {i: 1.0 for i in telemetry}

    # Build router membership using provided topology (preferred)
    router_ifaces: Dict[str, List[str]] = {}
    if topology:
        for r, ifs in topology.items():
            router_ifaces[r] = [i for i in ifs if i in telemetry]
    else:
        # Fallback to local_router if topology not supplied (still useful for R1)
        for iid, d in telemetry.items():
            r = d.get('local_router')
            router_ifaces.setdefault(r, []).append(iid)

    # Guarded router-level flow conservation (R1)
    # Targeted adjustment: adjust only the lower-confidence side, distribute by weights,
    # clip per-interface relative change to +/-10%, and damp the total correction by 60%.
    for r, ifs in router_ifaces.items():
        # Consider only interfaces that are up to avoid double-penalizing down links
        up_ifs = [i for i in ifs if status.get(i) == 'up']
        if len(up_ifs) < 2:
            continue

        sum_rx = sum(hardened_rx[i] for i in up_ifs)
        sum_tx = sum(hardened_tx[i] for i in up_ifs)
        denom = max(1.0, sum_rx, sum_tx)
        imbalance = abs(sum_rx - sum_tx) / denom

        # Adaptive router tolerance depending on number of active links
        n_active = len(up_ifs)
        tau_router = min(0.07, max(0.03, 0.05 * (2.0 / max(2, n_active)) ** 0.5))
        if imbalance <= tau_router:
            continue

        # Aggregate confidences per side
        avg_rx_conf = sum(conf_rx[i] for i in up_ifs) / len(up_ifs)
        avg_tx_conf = sum(conf_tx[i] for i in up_ifs) / len(up_ifs)

        # Choose side with lower aggregate confidence to adjust
        adjust_side = 'rx' if avg_rx_conf < avg_tx_conf else 'tx'
        delta = sum_rx - sum_tx  # positive: rx larger than tx
        # We aim to reduce the gap; compute total additive adjustment with damping
        total_adjust = (-delta) if adjust_side == 'rx' else (delta)
        total_adjust *= 0.6  # damping to avoid over-correction

        # Build weights: favor larger links and lower-confidence signals
        vals = []
        confs = []
        if adjust_side == 'rx':
            vals = [hardened_rx[i] for i in up_ifs]
            confs = [conf_rx[i] for i in up_ifs]
        else:
            vals = [hardened_tx[i] for i in up_ifs]
            confs = [conf_tx[i] for i in up_ifs]

        weights = []
        for v, c in zip(vals, confs):
            w = (max(v, 0.0) + 1e-6) * (1.0 - clamp01(c)) + 1e-6
            weights.append(w)
        total_w = sum(weights)
        if total_w <= 0:
            continue

        # Apply distributed adjustments with per-interface clipping (+/-10%)
        for idx, i in enumerate(up_ifs):
            v_old = vals[idx]
            w_i = weights[idx] / total_w
            adj = total_adjust * w_i
            # Clip per-interface relative change to +/-10%
            cap = 0.10 * v_old
            adj_clipped = min(max(adj, -cap), cap)
            v_new = max(0.0, v_old + adj_clipped)

            # Update hardened values and track effective scale factor for re-sync guard
            if adjust_side == 'rx':
                prev = hardened_rx[i]
                hardened_rx[i] = v_new
                if prev > ZERO_EPS:
                    scaled_rx_factor[i] *= (v_new / prev)
                # Moderate confidence penalty proportional to applied change
                rel = abs(adj_clipped) / max(1.0, abs(prev))
                conf_rx[i] = clamp01(conf_rx[i] * (1.0 - 0.6 * rel))
            else:
                prev = hardened_tx[i]
                hardened_tx[i] = v_new
                if prev > ZERO_EPS:
                    scaled_tx_factor[i] *= (v_new / prev)
                rel = abs(adj_clipped) / max(1.0, abs(prev))
                conf_tx[i] = clamp01(conf_tx[i] * (1.0 - 0.6 * rel))

    # Final symmetry touch-up (R3) with one-sided, confidence-gap-proportional nudge
    for a, b in list(processed_pairs):
        if a not in telemetry or b not in telemetry:
            continue
        # Skip if either side is down (already zeroed)
        if status.get(a) != 'up' or status.get(b) != 'up':
            continue

        # Helper to perform one-sided nudge toward mean
        def nudge_one_side(val_lo: float, val_hi: float, f: float) -> float:
            target = 0.5 * (val_lo + val_hi)
            return val_lo + f * (target - val_lo)

        # Direction 1: a.tx vs b.rx
        a_tx = hardened_tx.get(a, 0.0)
        b_rx = hardened_rx.get(b, 0.0)
        if max(a_tx, b_rx) > ZERO_EPS:
            d1 = rel_diff(a_tx, b_rx)
            tau1 = compute_tau_h(a_tx, b_rx)
            if d1 > tau1:
                conf_a_tx = conf_tx.get(a, 0.6)
                conf_b_rx = conf_rx.get(b, 0.6)
                if conf_a_tx >= conf_b_rx:
                    # Nudge lower-confidence side (b.rx) unless it had strong router scaling
                    if abs(scaled_rx_factor.get(b, 1.0) - 1.0) <= 0.08:
                        f = min(0.4, max(0.0, conf_a_tx - conf_b_rx))
                        if f > 0.0:
                            old = b_rx
                            new = nudge_one_side(old, a_tx, f)
                            hardened_rx[b] = max(0.0, new)
                            relc = abs(new - old) / max(1.0, abs(old), abs(new))
                            conf_rx[b] = clamp01(conf_rx.get(b, 0.6) * (1.0 - 0.3 * relc))
                else:
                    # Nudge lower-confidence side (a.tx) unless it had strong router scaling
                    if abs(scaled_tx_factor.get(a, 1.0) - 1.0) <= 0.08:
                        f = min(0.4, max(0.0, conf_b_rx - conf_a_tx))
                        if f > 0.0:
                            old = a_tx
                            new = nudge_one_side(old, b_rx, f)
                            hardened_tx[a] = max(0.0, new)
                            relc = abs(new - old) / max(1.0, abs(old), abs(new))
                            conf_tx[a] = clamp01(conf_tx.get(a, 0.6) * (1.0 - 0.3 * relc))

        # Direction 2: a.rx vs b.tx
        a_rx = hardened_rx.get(a, 0.0)
        b_tx = hardened_tx.get(b, 0.0)
        if max(a_rx, b_tx) > ZERO_EPS:
            d2 = rel_diff(a_rx, b_tx)
            tau2 = compute_tau_h(a_rx, b_tx)
            if d2 > tau2:
                conf_a_rx = conf_rx.get(a, 0.6)
                conf_b_tx = conf_tx.get(b, 0.6)
                if conf_a_rx >= conf_b_tx:
                    # Nudge lower-confidence side (b.tx) unless it had strong router scaling
                    if abs(scaled_tx_factor.get(b, 1.0) - 1.0) <= 0.08:
                        f = min(0.4, max(0.0, conf_a_rx - conf_b_tx))
                        if f > 0.0:
                            old = b_tx
                            new = nudge_one_side(old, a_rx, f)
                            hardened_tx[b] = max(0.0, new)
                            relc = abs(new - old) / max(1.0, abs(old), abs(new))
                            conf_tx[b] = clamp01(conf_tx.get(b, 0.6) * (1.0 - 0.3 * relc))
                else:
                    # Nudge lower-confidence side (a.rx) unless it had strong router scaling
                    if abs(scaled_rx_factor.get(a, 1.0) - 1.0) <= 0.08:
                        f = min(0.4, max(0.0, conf_b_tx - conf_a_rx))
                        if f > 0.0:
                            old = a_rx
                            new = nudge_one_side(old, b_tx, f)
                            hardened_rx[a] = max(0.0, new)
                            relc = abs(new - old) / max(1.0, abs(old), abs(new))
                            conf_rx[a] = clamp01(conf_rx.get(a, 0.6) * (1.0 - 0.3 * relc))

    # Confidence peer smoothing: softly harmonize per-direction confidences across peers
    # Apply only when both ends are up
    new_conf_rx = dict(conf_rx)
    new_conf_tx = dict(conf_tx)
    for a, b in list(processed_pairs):
        if a not in telemetry or b not in telemetry:
            continue
        if status.get(a) != 'up' or status.get(b) != 'up':
            continue
        # Blend 10% from peer opposite direction
        new_conf_tx[a] = clamp01(0.9 * conf_tx.get(a, 0.6) + 0.1 * conf_rx.get(b, 0.6))
        new_conf_rx[b] = clamp01(0.9 * conf_rx.get(b, 0.6) + 0.1 * conf_tx.get(a, 0.6))
        new_conf_rx[a] = clamp01(0.9 * conf_rx.get(a, 0.6) + 0.1 * conf_tx.get(b, 0.6))
        new_conf_tx[b] = clamp01(0.9 * conf_tx.get(b, 0.6) + 0.1 * conf_rx.get(a, 0.6))
    conf_rx = new_conf_rx
    conf_tx = new_conf_tx

    # Enforce interface down => zero traffic (final safeguard)
    for i in telemetry:
        if status.get(i) == 'down':
            hardened_rx[i] = 0.0
            hardened_tx[i] = 0.0
            conf_rx[i] = max(conf_rx[i], 0.85)
            conf_tx[i] = max(conf_tx[i], 0.85)

    # Assemble result with confidence calibration
    result: Dict[str, Dict[str, Tuple]] = {}
    for i, data in telemetry.items():
        interface_status = status.get(i, 'unknown')
        connected_to = data.get('connected_to')

        # Status confidence: penalize when peer status inconsistent or traffic present while down
        status_confidence = 1.0
        if connected_to and connected_to in telemetry:
            peer_status = telemetry[connected_to].get('interface_status', 'unknown')
            if interface_status != peer_status:
                status_confidence = 0.6
        if interface_status == 'down' and (orig_rx.get(i, 0.0) > ZERO_EPS or orig_tx.get(i, 0.0) > ZERO_EPS):
            status_confidence = min(status_confidence, 0.6)

        rx_c_base = clamp01(conf_rx.get(i, 0.6))
        tx_c_base = clamp01(conf_tx.get(i, 0.6))
        # Include a small (10%) confidence component tied to the magnitude of router scaling
        alpha_rx = abs(scaled_rx_factor.get(i, 1.0) - 1.0)
        alpha_tx = abs(scaled_tx_factor.get(i, 1.0) - 1.0)
        scale_term_rx = clamp01(1.0 - min(0.5, alpha_rx))
        scale_term_tx = clamp01(1.0 - min(0.5, alpha_tx))
        rx_c = clamp01(0.9 * rx_c_base + 0.1 * scale_term_rx)
        tx_c = clamp01(0.9 * tx_c_base + 0.1 * scale_term_tx)

        repaired: Dict[str, Any] = {}
        repaired['rx_rate'] = (orig_rx.get(i, 0.0), hardened_rx.get(i, 0.0), rx_c)
        repaired['tx_rate'] = (orig_tx.get(i, 0.0), hardened_tx.get(i, 0.0), tx_c)
        repaired['interface_status'] = (interface_status, interface_status, status_confidence)

        # Copy metadata unchanged
        repaired['connected_to'] = connected_to
        repaired['local_router'] = data.get('local_router')
        repaired['remote_router'] = data.get('remote_router')

        result[i] = repaired

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