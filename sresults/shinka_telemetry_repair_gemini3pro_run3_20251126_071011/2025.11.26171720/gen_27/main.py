# EVOLVE-BLOCK-START
from typing import Dict, Any, Tuple, List
import math

def repair_network_telemetry(telemetry: Dict[str, Dict[str, Any]],
                             topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:
    """
    Repairs network telemetry using a consensus mechanism between Link Symmetry
    and Router Flow Conservation invariants, with ambiguity-aware confidence calibration.

    Algorithm:
    1. Identify 'clean' (consistent) and 'suspect' (inconsistent) links.
    2. For suspect links, generate hypotheses: True Value = Local vs Peer.
    3. Validate hypotheses using Router Flow Conservation.
    4. Select the hypothesis that minimizes global network violation.
    5. Detect ambiguity: If multiple hypotheses are similarly good, reduce confidence.
    """

    # --- Configuration ---
    SYMMETRY_TOLERANCE = 0.02
    CONSERVATION_TOLERANCE_PCT = 0.03
    MIN_SIGNIFICANT_FLOW = 0.1
    ITERATIONS = 5

    # --- Helper Structures ---
    if_to_router = {}
    for r_id, if_list in topology.items():
        for i_id in if_list:
            if_to_router[i_id] = r_id

    # Group interfaces into Links
    links = {}
    processed_ifs = set()

    for if_id, data in telemetry.items():
        if if_id in processed_ifs: continue

        peer_id = data.get('connected_to')
        if peer_id and peer_id in telemetry:
            # Internal Link
            link_key = tuple(sorted([if_id, peer_id]))
            links[link_key] = {'type': 'internal', 'if1': if_id, 'if2': peer_id}
            processed_ifs.add(if_id)
            processed_ifs.add(peer_id)
        else:
            # External Link
            links[(if_id,)] = {'type': 'external', 'if1': if_id, 'if2': None}
            processed_ifs.add(if_id)

    # --- Step 1: Initial Link Assessment ---
    # Current best estimates
    current_estimates = {} # {if_id: {'rx': val, 'tx': val}}
    estimate_confidence = {} # {if_id: {'rx': conf, 'tx': conf}}

    # Suspect flows: Flows that need resolution
    suspect_flows = []

    for link_key, info in links.items():
        if1 = info['if1']
        if2 = info['if2']
        d1 = telemetry[if1]

        # Initialize estimates
        if if2:
            d2 = telemetry[if2]

            # Check 1->2 (if1 TX, if2 RX)
            v_tx = d1.get('tx_rate', 0.0)
            v_rx = d2.get('rx_rate', 0.0)

            denom = max(v_tx, v_rx, 1.0)
            diff = abs(v_tx - v_rx)

            if diff / denom < SYMMETRY_TOLERANCE:
                # Solid
                avg = (v_tx + v_rx) / 2.0
                current_estimates[if1] = current_estimates.get(if1, {})
                current_estimates[if1]['tx'] = avg
                current_estimates[if2] = current_estimates.get(if2, {})
                current_estimates[if2]['rx'] = avg

                estimate_confidence[if1] = estimate_confidence.get(if1, {})
                estimate_confidence[if1]['tx'] = 0.95
                estimate_confidence[if2] = estimate_confidence.get(if2, {})
                estimate_confidence[if2]['rx'] = 0.95
            else:
                # Suspect
                suspect_flows.append({
                    'src': if1, 'dst': if2, 'dir': 'tx_to_rx',
                    'val_src': v_tx, 'val_dst': v_rx
                })
                # Init with average, low conf
                avg = (v_tx + v_rx) / 2.0
                current_estimates[if1] = current_estimates.get(if1, {})
                current_estimates[if1]['tx'] = avg
                current_estimates[if2] = current_estimates.get(if2, {})
                current_estimates[if2]['rx'] = avg
                estimate_confidence[if1] = estimate_confidence.get(if1, {})
                estimate_confidence[if1]['tx'] = 0.5
                estimate_confidence[if2] = estimate_confidence.get(if2, {})
                estimate_confidence[if2]['rx'] = 0.5

            # Check 2->1 (if2 TX, if1 RX)
            v_tx = d2.get('tx_rate', 0.0)
            v_rx = d1.get('rx_rate', 0.0)

            denom = max(v_tx, v_rx, 1.0)
            diff = abs(v_tx - v_rx)

            if diff / denom < SYMMETRY_TOLERANCE:
                avg = (v_tx + v_rx) / 2.0
                current_estimates[if2]['tx'] = avg
                current_estimates[if1]['rx'] = avg
                estimate_confidence[if2]['tx'] = 0.95
                estimate_confidence[if1]['rx'] = 0.95
            else:
                suspect_flows.append({
                    'src': if2, 'dst': if1, 'dir': 'tx_to_rx',
                    'val_src': v_tx, 'val_dst': v_rx
                })
                avg = (v_tx + v_rx) / 2.0
                current_estimates[if2]['tx'] = avg
                current_estimates[if1]['rx'] = avg
                estimate_confidence[if2]['tx'] = 0.5
                estimate_confidence[if1]['rx'] = 0.5

        else:
            # External Link
            # TX
            v_tx = d1.get('tx_rate', 0.0)
            current_estimates[if1] = current_estimates.get(if1, {})
            current_estimates[if1]['tx'] = v_tx
            estimate_confidence[if1] = estimate_confidence.get(if1, {})
            estimate_confidence[if1]['tx'] = 0.8
            suspect_flows.append({
                'src': if1, 'dst': None, 'dir': 'tx_to_void',
                'val_src': v_tx, 'val_dst': None
            })

            # RX
            v_rx = d1.get('rx_rate', 0.0)
            current_estimates[if1]['rx'] = v_rx
            estimate_confidence[if1]['rx'] = 0.8
            suspect_flows.append({
                'src': None, 'dst': if1, 'dir': 'void_to_rx',
                'val_src': None, 'val_dst': v_rx
            })

    # --- Step 2: Iterative Bayesian Refinement ---

    def get_router_imbalance(rid):
        if not rid: return 0.0, 1.0
        if_list = topology.get(rid, [])
        total_in = 0.0
        total_out = 0.0
        for iid in if_list:
            total_in += current_estimates.get(iid, {}).get('rx', 0.0)
            total_out += current_estimates.get(iid, {}).get('tx', 0.0)
        return total_in - total_out, max(total_in, total_out, 1.0)

    for iteration in range(ITERATIONS):
        updates = {}

        for flow in suspect_flows:
            src, dst = flow['src'], flow['dst']

            # Generate Hypotheses
            candidates = set()
            if flow['val_src'] is not None: candidates.add(flow['val_src'])
            if flow['val_dst'] is not None: candidates.add(flow['val_dst'])
            candidates.add(0.0)

            # Infer from Router Balance
            r_src = if_to_router.get(src) if src else None
            r_dst = if_to_router.get(dst) if dst else None

            if r_src and src:
                # Src is sending (TX). Balance: In = Out_others + My_TX
                # My_TX = In - Out_others = In - (Out - Current_TX) = Current_TX + (In - Out)
                imb, _ = get_router_imbalance(r_src)
                implied = current_estimates[src]['tx'] + imb
                if implied > 0: candidates.add(implied)

            if r_dst and dst:
                # Dst is receiving (RX). Balance: Out = In_others + My_RX
                # My_RX = Out - In_others = Out - (In - Current_RX) = Current_RX - (In - Out)
                imb, _ = get_router_imbalance(r_dst)
                implied = current_estimates[dst]['rx'] - imb
                if implied > 0: candidates.add(implied)

            # Check conservation for each hypothesis
            hyps = list(candidates)
            scores = []

            # Save current
            old_src_tx = current_estimates[src]['tx'] if src else 0
            old_dst_rx = current_estimates[dst]['rx'] if dst else 0

            for h in hyps:
                # Apply h
                if src: current_estimates[src]['tx'] = h
                if dst: current_estimates[dst]['rx'] = h

                # Score Src
                score_src = 1.0
                if r_src:
                    imb, flow_mag = get_router_imbalance(r_src)
                    sigma = max(flow_mag * CONSERVATION_TOLERANCE_PCT, 1.0)
                    score_src = math.exp(-abs(imb) / sigma)

                # Score Dst
                score_dst = 1.0
                if r_dst:
                    imb, flow_mag = get_router_imbalance(r_dst)
                    sigma = max(flow_mag * CONSERVATION_TOLERANCE_PCT, 1.0)
                    score_dst = math.exp(-abs(imb) / sigma)

                scores.append(score_src * score_dst)

            # Restore
            if src: current_estimates[src]['tx'] = old_src_tx
            if dst: current_estimates[dst]['rx'] = old_dst_rx

            # Normalize scores
            total_score = sum(scores) + 1e-12
            probs = [s / total_score for s in scores]

            # Winner
            best_idx = 0
            best_p = -1
            for i, p in enumerate(probs):
                if p > best_p:
                    best_p = p
                    best_idx = i

            winner_val = hyps[best_idx]

            # Confidence Calibration
            raw_quality = scores[best_idx] # 0 to 1
            conf = best_p * math.sqrt(raw_quality)
            conf = min(0.99, max(0.01, conf))

            # Soft Update (Momentum)
            alpha = 0.5 + 0.5 * conf

            if src:
                prev = current_estimates[src]['tx']
                new_val = alpha * winner_val + (1 - alpha) * prev
                updates[(src, 'tx')] = (new_val, conf)
            if dst:
                prev = current_estimates[dst]['rx']
                new_val = alpha * winner_val + (1 - alpha) * prev
                updates[(dst, 'rx')] = (new_val, conf)

        # Apply Updates
        for (if_id, metric), (val, conf) in updates.items():
            current_estimates[if_id][metric] = val
            estimate_confidence[if_id][metric] = conf

    # --- Step 3: Status & Result Construction ---
    result = {}
    for if_id, data in telemetry.items():
        orig_rx = data.get('rx_rate', 0.0)
        orig_tx = data.get('tx_rate', 0.0)
        orig_status = data.get('interface_status', 'unknown')

        est_rx = current_estimates[if_id]['rx']
        est_tx = current_estimates[if_id]['tx']
        conf_rx = estimate_confidence[if_id]['rx']
        conf_tx = estimate_confidence[if_id]['tx']

        # Determine peer status
        peer_id = data.get('connected_to')
        peer_status = 'unknown'
        if peer_id and peer_id in telemetry:
            peer_status = telemetry[peer_id].get('interface_status', 'unknown')

        # Status Logic
        has_rx = est_rx > MIN_SIGNIFICANT_FLOW
        has_tx = est_tx > MIN_SIGNIFICANT_FLOW

        rep_status = orig_status
        conf_status = 1.0

        if has_rx or has_tx:
            rep_status = 'up'
            if orig_status != 'up':
                conf_status = max(conf_rx if has_rx else 0, conf_tx if has_tx else 0)
        elif peer_status == 'down':
            rep_status = 'down'
            if orig_status != 'down':
                conf_status = 0.9
        elif orig_status == 'up' and not has_rx and not has_tx:
            rep_status = 'up'

        # Post-process: If status is DOWN, force rates to 0
        if rep_status == 'down':
            est_rx = 0.0
            est_tx = 0.0
            conf_rx = max(conf_rx, conf_status)
            conf_tx = max(conf_tx, conf_status)

        entry = {}
        entry['rx_rate'] = (orig_rx, est_rx, conf_rx)
        entry['tx_rate'] = (orig_tx, est_tx, conf_tx)
        entry['interface_status'] = (orig_status, rep_status, conf_status)
        for k in ['connected_to', 'local_router', 'remote_router']:
            if k in data: entry[k] = data[k]
        result[if_id] = entry

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