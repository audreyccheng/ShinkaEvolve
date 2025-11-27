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
            estimate_confidence[if1]['tx'] = 0.90 # Tentative

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

    # Helper to calculate router imbalance given current estimates
    def get_router_imbalance(rid):
        if_list = topology.get(rid, [])
        total_in = 0.0
        total_out = 0.0
        for iid in if_list:
            total_in += current_estimates.get(iid, {}).get('rx', 0.0)
            total_out += current_estimates.get(iid, {}).get('tx', 0.0)
        return total_in - total_out, max(total_in, total_out, 1.0)

    # Iteration loop
    for _ in range(ITERATIONS):
        updates = {} # Store updates to apply after full pass

        for flow_prob in suspect_flows:
            link_key = flow_prob['key']
            direction = flow_prob['dir']
            candidates = flow_prob['candidates']

            # Identify interfaces and routers involved
            info = links[link_key]
            if direction == '1_to_2':
                src_if, dst_if = info['if1'], info['if2']
                val_src, val_dst = candidates[0], candidates[1] # src measured tx, dst measured rx
            else:
                src_if, dst_if = info['if2'], info['if1']
                val_src, val_dst = candidates[0], candidates[1]

            router_src = if_to_router.get(src_if)
            router_dst = if_to_router.get(dst_if)

            # Hypotheses:
            # 1. val_src is correct
            # 2. val_dst is correct
            # 3. 0.0 is correct (Link Down / Phantom traffic)

            hyps = [val_src, val_dst, 0.0]
            scores = []

            for h_val in hyps:
                # Probability score based on Gaussian likelihood of imbalance

                # Check Source Router Imbalance if we use h_val for this TX interface
                # Temporarily replace value in calculation
                old_tx = current_estimates[src_if]['tx']
                current_estimates[src_if]['tx'] = h_val
                imb_src, flow_src = get_router_imbalance(router_src)
                # Restore
                current_estimates[src_if]['tx'] = old_tx

                # Check Dest Router Imbalance if we use h_val for this RX interface
                old_rx = current_estimates[dst_if]['rx']
                current_estimates[dst_if]['rx'] = h_val
                imb_dst, flow_dst = get_router_imbalance(router_dst)
                current_estimates[dst_if]['rx'] = old_rx

                # Likelihood function: P ~ exp(- |imbalance| / sigma)
                # sigma is tolerance proportional to flow
                sigma_src = max(flow_src * CONSERVATION_TOLERANCE_PCT, 1.0)
                sigma_dst = max(flow_dst * CONSERVATION_TOLERANCE_PCT, 1.0)

                score_src = math.exp(-abs(imb_src) / sigma_src) if router_src else 1.0
                score_dst = math.exp(-abs(imb_dst) / sigma_dst) if router_dst else 1.0

                # Combined score
                scores.append(score_src * score_dst)

            # Select Winner
            total_s = sum(scores) + 1e-12
            probs = [s / total_s for s in scores]

            # Find best hypothesis
            best_idx = 0
            best_p = -1.0
            for i, p in enumerate(probs):
                if p > best_p:
                    best_p = p
                    best_idx = i

            winner_val = hyps[best_idx]

            # Calibration: Confidence should reflect:
            # 1. How much better this hypothesis is (best_p)
            # 2. How good the fit is absolutely (scores[best_idx])
            # If fit is perfect (score=1), conf = best_p.
            # If fit is poor (score low), conf reduces.
            # We use sqrt of score to not penalize too harshly for minor noise.

            raw_score = scores[best_idx]
            goodness = math.sqrt(raw_score)

            conf = best_p * goodness
            conf = max(0.01, min(0.99, conf))

            # Store update
            updates[(src_if, 'tx')] = (winner_val, conf)
            updates[(dst_if, 'rx')] = (winner_val, conf)

        # Apply updates
        for (if_id, metric), (val, conf) in updates.items():
            current_estimates[if_id][metric] = val
            estimate_confidence[if_id][metric] = conf

    # --- Step 2.5: Global Confidence Penalty for Residual Imbalance ---
    # If a router is still imbalanced, it means our repairs (or external links) are inconsistent.
    # Reduce confidence for all connected interfaces.

    for rid, if_list in topology.items():
        imb, flow = get_router_imbalance(rid)
        tol = max(flow * CONSERVATION_TOLERANCE_PCT, 1.0)

        if abs(imb) > tol:
            # Calculate penalty based on severity
            # If imb is significantly above tolerance, penalty increases.
            ratio = abs(imb) / tol
            # Exponential penalty
            penalty_factor = math.exp(-(ratio - 1.0) * 0.5)
            penalty_factor = min(1.0, max(0.0, penalty_factor))

            for iid in if_list:
                if iid in estimate_confidence:
                    for metric in ['rx', 'tx']:
                        if metric in estimate_confidence[iid]:
                            estimate_confidence[iid][metric] *= penalty_factor

    # --- Step 3: Final Assembly & Status Repair ---

    result = {}

    for if_id, data in telemetry.items():
        orig_rx = data.get('rx_rate', 0.0)
        orig_tx = data.get('tx_rate', 0.0)
        orig_status = data.get('interface_status', 'unknown')

        # Repaired Rates
        est_rx = current_estimates[if_id]['rx']
        conf_rx = estimate_confidence[if_id]['rx']

        est_tx = current_estimates[if_id]['tx']
        conf_tx = estimate_confidence[if_id]['tx']

        # Status Inference
        # Determine peer status
        peer_id = data.get('connected_to')
        peer_status = 'unknown'
        if peer_id and peer_id in telemetry:
            peer_status = telemetry[peer_id].get('interface_status', 'unknown')

        # 1. Existence of significant traffic implies UP
        has_rx = est_rx > MIN_SIGNIFICANT_FLOW
        has_tx = est_tx > MIN_SIGNIFICANT_FLOW

        rep_status = orig_status
        conf_status = 1.0 # Baseline

        if has_rx or has_tx:
            rep_status = 'up'
            if orig_status != 'up':
                # We are overturning status based on traffic.
                # Confidence depends on traffic confidence.
                flow_conf = max(conf_rx if has_rx else 0, conf_tx if has_tx else 0)
                conf_status = flow_conf

        elif peer_status == 'down':
            # Peer is down and we have no traffic -> We should be down (usually)
            rep_status = 'down'
            if orig_status != 'down':
                conf_status = 0.9

        elif orig_status == 'up' and not has_rx and not has_tx:
            # We say UP, but no traffic.
            # Could be idle.
            # If peer is UP, likely idle -> Keep UP.
            # If peer is DOWN (caught above), then DOWN.
            # If peer unknown, keep UP.
            rep_status = 'up'

        # Post-process: If status is DOWN, force rates to 0
        if rep_status == 'down':
            est_rx = 0.0
            est_tx = 0.0
            # High confidence in 0 if we are confident it's down
            conf_rx = max(conf_rx, conf_status)
            conf_tx = max(conf_tx, conf_status)

        # Result Construction
        entry = {}
        entry['rx_rate'] = (orig_rx, est_rx, conf_rx)
        entry['tx_rate'] = (orig_tx, est_tx, conf_tx)
        entry['interface_status'] = (orig_status, rep_status, conf_status)

        # Metadata
        for k in ['connected_to', 'local_router', 'remote_router']:
            if k in data:
                entry[k] = data[k]

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
