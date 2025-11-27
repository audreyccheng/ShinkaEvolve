# EVOLVE-BLOCK-START
from typing import Dict, Any, Tuple, List
import math

def repair_network_telemetry(telemetry: Dict[str, Dict[str, Any]],
                             topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:
    """
    Repairs network telemetry using a Bayesian Flow Consensus algorithm.
    It fuses evidence from Link Symmetry and Router Flow Conservation to
    probabilistically determine the most likely true state of network counters.
    """

    # --- Configuration ---
    SYMMETRY_TOLERANCE = 0.02
    CONSERVATION_TOLERANCE_PCT = 0.03
    MIN_SIGNIFICANT_FLOW = 0.5
    ITERATIONS = 5

    # --- Helper Structures ---
    # Map interface to router
    if_to_router = {}
    for r_id, if_list in topology.items():
        for i_id in if_list:
            if_to_router[i_id] = r_id

    # Group interfaces into Links for processing
    # Link ID = tuple of sorted interface IDs to handle bidirectionality uniquely
    links = {}
    processed_ifs = set()

    for if_id, data in telemetry.items():
        if if_id in processed_ifs: continue

        peer_id = data.get('connected_to')
        if peer_id and peer_id in telemetry:
            # Internal Link
            link_key = tuple(sorted([if_id, peer_id]))
            links[link_key] = {
                'type': 'internal',
                'if1': if_id,
                'if2': peer_id
            }
            processed_ifs.add(if_id)
            processed_ifs.add(peer_id)
        else:
            # Edge/External Link
            links[(if_id,)] = {
                'type': 'external',
                'if1': if_id,
                'if2': None
            }
            processed_ifs.add(if_id)

    # --- Step 1: Initial Link Assessment ---
    # Determine which links are 'Solid' (agreeing) and which are 'Suspect'.
    # For suspect links, generate hypotheses.

    # Store current best estimates for RX and TX flow on every interface
    # structure: {if_id: {'rx': val, 'tx': val}}
    current_estimates = {}

    # Store reliability/confidence of these estimates
    # structure: {if_id: {'rx': conf, 'tx': conf}} (0.0 to 1.0)
    estimate_confidence = {}

    # Suspect flows to solve: list of (link_key, metric_type ('rx_flow' or 'tx_flow'))
    # Note: 'rx_flow' for link (A,B) means A->B traffic (A TX, B RX).
    suspect_flows = []

    for link_key, info in links.items():
        if1 = info['if1']
        if2 = info['if2']

        d1 = telemetry[if1]
        d2 = telemetry[if2] if if2 else {}

        # Analyze Flow IF1 -> IF2 (IF1 TX, IF2 RX)
        val1_tx = d1.get('tx_rate', 0.0)
        val2_rx = d2.get('rx_rate', 0.0) if if2 else None

        if if2:
            # Internal Link: Check Symmetry
            denom = max(val1_tx, val2_rx, 1.0)
            diff = abs(val1_tx - val2_rx)

            if diff / denom < SYMMETRY_TOLERANCE:
                # Consistent
                consensus = (val1_tx + val2_rx) / 2.0
                current_estimates[if1] = current_estimates.get(if1, {})
                current_estimates[if1]['tx'] = consensus
                current_estimates[if2] = current_estimates.get(if2, {})
                current_estimates[if2]['rx'] = consensus

                estimate_confidence[if1] = estimate_confidence.get(if1, {})
                estimate_confidence[if1]['tx'] = 0.95
                estimate_confidence[if2] = estimate_confidence.get(if2, {})
                estimate_confidence[if2]['rx'] = 0.95
            else:
                # Suspect
                suspect_flows.append({'key': link_key, 'dir': '1_to_2', 'candidates': [val1_tx, val2_rx]})
                # Initialize with average but low confidence
                consensus = (val1_tx + val2_rx) / 2.0
                current_estimates[if1] = current_estimates.get(if1, {})
                current_estimates[if1]['tx'] = consensus
                current_estimates[if2] = current_estimates.get(if2, {})
                current_estimates[if2]['rx'] = consensus

                estimate_confidence[if1] = estimate_confidence.get(if1, {})
                estimate_confidence[if1]['tx'] = 0.5
                estimate_confidence[if2] = estimate_confidence.get(if2, {})
                estimate_confidence[if2]['rx'] = 0.5
        else:
            # External Link: Trust local blindly for now (no peer to contradict)
            current_estimates[if1] = current_estimates.get(if1, {})
            current_estimates[if1]['tx'] = val1_tx
            estimate_confidence[if1] = estimate_confidence.get(if1, {})
            estimate_confidence[if1]['tx'] = 0.90 # Less confident than hardened links

        # Analyze Flow IF2 -> IF1 (IF2 TX, IF1 RX)
        if if2:
            val2_tx = d2.get('tx_rate', 0.0)
            val1_rx = d1.get('rx_rate', 0.0)

            denom = max(val2_tx, val1_rx, 1.0)
            diff = abs(val2_tx - val1_rx)

            if diff / denom < SYMMETRY_TOLERANCE:
                consensus = (val2_tx + val1_rx) / 2.0
                current_estimates[if2]['tx'] = consensus
                current_estimates[if1]['rx'] = consensus
                estimate_confidence[if2]['tx'] = 0.95
                estimate_confidence[if1]['rx'] = 0.95
            else:
                suspect_flows.append({'key': link_key, 'dir': '2_to_1', 'candidates': [val2_tx, val1_rx]})
                consensus = (val2_tx + val1_rx) / 2.0
                current_estimates[if2]['tx'] = consensus
                current_estimates[if1]['rx'] = consensus
                estimate_confidence[if2]['tx'] = 0.5
                estimate_confidence[if1]['rx'] = 0.5
        else:
             # External RX
            val1_rx = d1.get('rx_rate', 0.0)
            current_estimates[if1]['rx'] = val1_rx
            estimate_confidence[if1]['rx'] = 0.90

    # --- Step 2: Iterative Bayesian Refinement ---

    def get_router_state(rid):
        if rid not in topology: return 0.0, 1.0
        tin, tout = 0.0, 0.0
        max_f = 0.0
        for iid in topology[rid]:
            if iid in current_estimates:
                r = current_estimates[iid].get('rx', 0.0)
                t = current_estimates[iid].get('tx', 0.0)
                tin += r
                tout += t
                max_f = max(max_f, r, t)
        return (tin - tout), max(max_f, 1.0)

    def calc_sigma(flow_val):
        return max(math.sqrt(flow_val), flow_val * CONSERVATION_TOLERANCE_PCT, 1.0)

    solver_confidences = {}
    MOMENTUM = 0.5

    for _ in range(ITERATIONS):
        updates = []

        # 1. Internal Suspect Flows
        for flow_prob in suspect_flows:
            link_key = flow_prob['key']
            direction = flow_prob['dir']
            candidates = flow_prob['candidates']

            info = links[link_key]
            if direction == '1_to_2':
                src_if, dst_if = info['if1'], info['if2']
            else:
                src_if, dst_if = info['if2'], info['if1']

            r_src = if_to_router.get(src_if)
            r_dst = if_to_router.get(dst_if)

            # Hypotheses: [Meas1, Meas2, 0.0] + Mean
            hyps = sorted(list(set([c for c in candidates if c >= 0] + [0.0])))
            if len(candidates) == 2:
                v1, v2 = candidates
                if abs(v1-v2) < max(v1,v2)*0.2 + 5.0:
                    hyps.append((v1+v2)/2.0)
            hyps = sorted(list(set(hyps)))

            curr_tx = current_estimates[src_if]['tx']
            curr_rx = current_estimates[dst_if]['rx']

            s_src = telemetry[src_if].get('interface_status', 'unknown')
            s_dst = telemetry[dst_if].get('interface_status', 'unknown')

            scores = []
            for h in hyps:
                current_estimates[src_if]['tx'] = h
                current_estimates[dst_if]['rx'] = h

                imb_s, flow_s = get_router_state(r_src)
                score_s = math.exp(-abs(imb_s)/calc_sigma(flow_s)) if r_src else 1.0

                imb_d, flow_d = get_router_state(r_dst)
                score_d = math.exp(-abs(imb_d)/calc_sigma(flow_d)) if r_dst else 1.0

                # Prior
                prior = 1.0
                if h == 0.0:
                    if s_src == 'down' or s_dst == 'down': prior = 0.95
                    else:
                        m_val = max(candidates)
                        if m_val > 10.0: prior = 0.01
                        elif m_val > 1.0: prior = 0.2
                else:
                    dist = min([abs(h-c) for c in candidates])
                    prior = math.exp(-dist / max(h*0.05, 1.0))

                scores.append(score_s * score_d * prior)

            current_estimates[src_if]['tx'] = curr_tx
            current_estimates[dst_if]['rx'] = curr_rx

            best_idx = scores.index(max(scores))
            win_val = hyps[best_idx]

            # Calibration: Probability mass near winner
            total = sum(scores) + 1e-20
            probs = [s/total for s in scores]
            win_prob = sum(p for i, p in enumerate(probs)
                           if abs(hyps[i] - win_val) < max(win_val*0.05, 1.0))

            updates.append((src_if, 'tx', win_val, win_prob))
            updates.append((dst_if, 'rx', win_val, win_prob))

        # 2. External Flows
        for key, info in links.items():
            if info['type'] != 'external': continue
            if_id = info['if1']
            r_id = if_to_router.get(if_id)
            if not r_id: continue

            metrics = []
            if 'tx' in current_estimates.get(if_id, {}): metrics.append('tx')
            if 'rx' in current_estimates.get(if_id, {}): metrics.append('rx')

            stat = telemetry[if_id].get('interface_status', 'unknown')

            for metric in metrics:
                curr_val = current_estimates[if_id][metric]
                meas = telemetry[if_id].get(f'{metric}_rate', 0.0)

                imb, r_flow = get_router_state(r_id)
                if metric == 'tx': implied = max(0.0, curr_val + imb)
                else: implied = max(0.0, curr_val - imb)

                hyps = sorted(list(set([meas, implied, 0.0])))
                scores = []
                for h in hyps:
                    current_estimates[if_id][metric] = h
                    imb, rf = get_router_state(r_id)
                    lik = math.exp(-abs(imb)/calc_sigma(rf))

                    prior = 1.0
                    if h == 0.0:
                        if stat == 'down': prior = 0.95
                        elif meas > 10.0: prior = 0.01
                    elif abs(h - meas) < 1e-6:
                        prior = 0.6
                    elif abs(h - implied) < 1e-6:
                        prior = 0.4

                    if h > 0:
                        prior *= math.exp(-abs(h-meas)/max(meas*0.05, 1.0))

                    scores.append(lik * prior)

                current_estimates[if_id][metric] = curr_val

                best_idx = scores.index(max(scores))
                win_val = hyps[best_idx]
                total = sum(scores) + 1e-20
                probs = [s/total for s in scores]
                win_prob = sum(p for i, p in enumerate(probs)
                           if abs(hyps[i] - win_val) < max(win_val*0.05, 1.0))

                updates.append((if_id, metric, win_val, win_prob))

        # Apply
        for if_id, metric, val, prob in updates:
            old = current_estimates[if_id][metric]
            current_estimates[if_id][metric] = (old * (1 - MOMENTUM)) + (val * MOMENTUM)
            solver_confidences[(if_id, metric)] = prob

    # --- Step 3: Final Assembly & Status Repair ---

    result = {}

    # Calculate final router conservation fits
    router_fits = {}
    for rid in topology:
        imb, flow = get_router_state(rid)
        fit = math.exp(-abs(imb) / calc_sigma(flow))
        router_fits[rid] = fit

    for if_id, data in telemetry.items():
        orig_rx = data.get('rx_rate', 0.0)
        orig_tx = data.get('tx_rate', 0.0)
        orig_status = data.get('interface_status', 'unknown')

        rep_rx = current_estimates[if_id]['rx']
        rep_tx = current_estimates[if_id]['tx']

        rid = if_to_router.get(if_id)
        r_fit = router_fits.get(rid, 0.8)

        def get_final_conf(metric, val, orig_val):
            # Check if this was a Step 1 Anchor (high confidence in estimate_confidence)
            step1_conf = estimate_confidence.get(if_id, {}).get(metric, 0.0)
            if step1_conf > 0.9:
                return 0.95

            # Otherwise use Solver Confidence
            sol_prob = solver_confidences.get((if_id, metric), 0.5)

            # Combine: Probability * Fit Quality
            base = sol_prob * (0.3 + 0.7 * r_fit)

            # If value didn't change much from measurement, boost confidence slightly
            if abs(val - orig_val) < max(orig_val * 0.1, 1.0):
                base = max(base, 0.8 * r_fit + 0.1)

            return max(0.01, min(0.99, base))

        conf_rx = get_final_conf('rx', rep_rx, orig_rx)
        conf_tx = get_final_conf('tx', rep_tx, orig_tx)

        # Status Inference
        peer_id = data.get('connected_to')
        peer_status = 'unknown'
        if peer_id and peer_id in telemetry:
            peer_status = telemetry[peer_id].get('interface_status', 'unknown')

        has_traffic = (rep_rx > MIN_SIGNIFICANT_FLOW) or (rep_tx > MIN_SIGNIFICANT_FLOW)

        rep_status = orig_status
        conf_status = 1.0

        if has_traffic:
            rep_status = 'up'
            if orig_status != 'up':
                conf_status = (conf_rx + conf_tx) / 2.0
        elif peer_status == 'down':
            rep_status = 'down'
            if orig_status != 'down':
                conf_status = 0.9
        else:
            rep_status = orig_status

        if rep_status == 'down':
            rep_rx, rep_tx = 0.0, 0.0
            conf_rx = max(conf_rx, 0.95)
            conf_tx = max(conf_tx, 0.95)

        entry = {}
        entry['rx_rate'] = (orig_rx, rep_rx, conf_rx)
        entry['tx_rate'] = (orig_tx, rep_tx, conf_tx)
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