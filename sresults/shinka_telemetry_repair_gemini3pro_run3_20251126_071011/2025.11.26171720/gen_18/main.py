# EVOLVE-BLOCK-START
from typing import Dict, Any, Tuple, List
import math

def repair_network_telemetry(telemetry: Dict[str, Dict[str, Any]],
                             topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:
    """
    Repairs network telemetry using a Bayesian Flow Consensus algorithm with
    Zero-Flow Hypothesis testing and calibrated confidence scoring.
    """

    # --- Configuration ---
    # Tolerances for detecting inconsistencies
    SYMMETRY_TOLERANCE = 0.02
    CONSERVATION_TOLERANCE = 0.03

    # Thresholds
    MIN_SIGNIFICANT_FLOW = 0.5
    ITERATIONS = 5

    # --- Data Structure Setup ---
    # Map interface to router for quick lookups
    if_to_router = {}
    for r_id, if_list in topology.items():
        for i_id in if_list:
            if_to_router[i_id] = r_id

    # Identification of Links (Internal vs External)
    links = {} # Key: tuple(sorted(ids)) or (id,)
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
            # External/Edge Link
            links[(if_id,)] = {
                'type': 'external',
                'if1': if_id,
                'if2': None
            }
            processed_ifs.add(if_id)

    # --- Step 1: Initial Assessment & Symmetry ---
    # Initialize estimates.
    # Validated links are "hardened" (high confidence).
    # Discrepant links are "suspect" (low confidence, to be solved).

    current_estimates = {}   # {if_id: {'rx': val, 'tx': val}}
    estimate_confidence = {} # {if_id: {'rx': conf, 'tx': conf}}
    suspect_flows = []       # List of problems to solve

    for link_key, info in links.items():
        if1 = info['if1']

        if info['type'] == 'external':
            # Trust external blindly initially (can't check symmetry)
            val_rx = telemetry[if1].get('rx_rate', 0.0)
            val_tx = telemetry[if1].get('tx_rate', 0.0)

            current_estimates[if1] = {'rx': val_rx, 'tx': val_tx}
            estimate_confidence[if1] = {'rx': 0.9, 'tx': 0.9}
            continue

        if2 = info['if2']
        d1 = telemetry[if1]
        d2 = telemetry[if2]

        # Check Flow 1 -> 2 (if1 TX, if2 RX)
        val1 = d1.get('tx_rate', 0.0)
        val2 = d2.get('rx_rate', 0.0)

        diff = abs(val1 - val2)
        denom = max(val1, val2, 1.0)

        # Initialize dictionary structure
        if if1 not in current_estimates: current_estimates[if1] = {}
        if if2 not in current_estimates: current_estimates[if2] = {}
        if if1 not in estimate_confidence: estimate_confidence[if1] = {}
        if if2 not in estimate_confidence: estimate_confidence[if2] = {}

        if diff / denom < SYMMETRY_TOLERANCE:
            # Consistent
            avg = (val1 + val2) / 2.0
            current_estimates[if1]['tx'] = avg
            current_estimates[if2]['rx'] = avg
            estimate_confidence[if1]['tx'] = 0.95
            estimate_confidence[if2]['rx'] = 0.95
        else:
            # Suspect
            suspect_flows.append({
                'src': if1, 'dst': if2, 'dir': '1_to_2',
                'candidates': [val1, val2, 0.0] # Hypothesis: Src, Dst, or Zero (phantom)
            })
            # Init with average
            avg = (val1 + val2) / 2.0
            current_estimates[if1]['tx'] = avg
            current_estimates[if2]['rx'] = avg
            estimate_confidence[if1]['tx'] = 0.5
            estimate_confidence[if2]['rx'] = 0.5

        # Check Flow 2 -> 1 (if2 TX, if1 RX)
        val1 = d2.get('tx_rate', 0.0)
        val2 = d1.get('rx_rate', 0.0)

        diff = abs(val1 - val2)
        denom = max(val1, val2, 1.0)

        if diff / denom < SYMMETRY_TOLERANCE:
            avg = (val1 + val2) / 2.0
            current_estimates[if2]['tx'] = avg
            current_estimates[if1]['rx'] = avg
            estimate_confidence[if2]['tx'] = 0.95
            estimate_confidence[if1]['rx'] = 0.95
        else:
            suspect_flows.append({
                'src': if2, 'dst': if1, 'dir': '2_to_1',
                'candidates': [val1, val2, 0.0]
            })
            avg = (val1 + val2) / 2.0
            current_estimates[if2]['tx'] = avg
            current_estimates[if1]['rx'] = avg
            estimate_confidence[if2]['tx'] = 0.5
            estimate_confidence[if1]['rx'] = 0.5

    # --- Step 2: Iterative Bayesian Solver ---

    def calculate_router_imbalance(rid):
        if not rid: return 0.0, 1.0
        if_list = topology.get(rid, [])
        total_in = 0.0
        total_out = 0.0
        for iid in if_list:
            est = current_estimates.get(iid, {})
            total_in += est.get('rx', 0.0)
            total_out += est.get('tx', 0.0)
        return (total_in - total_out), max(total_in, total_out, 1.0)

    for _ in range(ITERATIONS):
        updates = []

        for flow in suspect_flows:
            src = flow['src']
            dst = flow['dst']

            # We want to find the value 'v' that maximizes P(Conservation|v)
            # P(Conservation|v) ~ P(Src_Cons|v) * P(Dst_Cons|v)

            r_src = if_to_router.get(src)
            r_dst = if_to_router.get(dst)

            # Filter unique candidates to avoid redundant calc
            candidates = sorted(list(set(flow['candidates'])))

            best_val = candidates[0]
            scores = []

            # Original values to restore
            orig_src_tx = current_estimates[src]['tx']
            orig_dst_rx = current_estimates[dst]['rx']

            for val in candidates:
                # Apply Hypothesis
                current_estimates[src]['tx'] = val
                current_estimates[dst]['rx'] = val

                # Check Source
                imb_src, flow_src = calculate_router_imbalance(r_src)
                # Sigma calculation: Non-linear noise model
                # Allows higher absolute error for high flows, but enforces tighter bounds for low flows
                sigma_src = max(flow_src * CONSERVATION_TOLERANCE, math.sqrt(flow_src), 1.0)
                score_src = math.exp(-abs(imb_src) / sigma_src)

                # Check Dest
                imb_dst, flow_dst = calculate_router_imbalance(r_dst)
                sigma_dst = max(flow_dst * CONSERVATION_TOLERANCE, math.sqrt(flow_dst), 1.0)
                score_dst = math.exp(-abs(imb_dst) / sigma_dst)

                combined_score = score_src * score_dst
                scores.append(combined_score)

            # Restore
            current_estimates[src]['tx'] = orig_src_tx
            current_estimates[dst]['rx'] = orig_dst_rx

            # Normalize scores to probabilities
            total_score = sum(scores) + 1e-12
            probs = [s / total_score for s in scores]

            # Pick winner
            max_p = -1.0
            winner_idx = 0
            for i, p in enumerate(probs):
                if p > max_p:
                    max_p = p
                    winner_idx = i

            winner_val = candidates[winner_idx]

            # Confidence Calibration
            # Use Goodness of Fit (absolute score) to scale the relative probability
            # If the best hypothesis is still garbage (low score), confidence should be low.
            raw_score = scores[winner_idx]
            goodness_of_fit = math.sqrt(raw_score) # Geometric mean of src/dst agreement

            # Final confidence is probability mass * quality of fit
            conf = max_p * goodness_of_fit
            # Clamp minimal confidence
            conf = max(0.01, min(0.99, conf))

            updates.append((src, 'tx', winner_val, conf))
            updates.append((dst, 'rx', winner_val, conf))

        # Apply updates
        for iid, metric, val, conf in updates:
            current_estimates[iid][metric] = val
            estimate_confidence[iid][metric] = conf

    # --- Step 3: Status Inference & Formatting ---

    result = {}
    for if_id, data in telemetry.items():
        orig_rx = data.get('rx_rate', 0.0)
        orig_tx = data.get('tx_rate', 0.0)
        orig_status = data.get('interface_status', 'unknown')

        rep_rx = current_estimates[if_id]['rx']
        conf_rx = estimate_confidence[if_id]['rx']

        rep_tx = current_estimates[if_id]['tx']
        conf_tx = estimate_confidence[if_id]['tx']

        # Determine Status
        # 1. Traffic existence (High confidence UP)
        has_traffic = (rep_rx > MIN_SIGNIFICANT_FLOW) or (rep_tx > MIN_SIGNIFICANT_FLOW)

        # 2. Peer status
        peer_id = data.get('connected_to')
        peer_status = 'unknown'
        if peer_id and peer_id in telemetry:
            peer_status = telemetry[peer_id].get('interface_status', 'unknown')

        rep_status = orig_status
        conf_status = 1.0

        if has_traffic:
            rep_status = 'up'
            if orig_status != 'up':
                conf_status = (conf_rx + conf_tx) / 2.0
        elif peer_status == 'down':
            rep_status = 'down'
            if orig_status != 'down':
                conf_status = 0.95
        elif orig_status == 'up' and not has_traffic:
            # Ambiguous: Could be idle.
            # If peer is UP, likely idle.
            # If peer is unknown/external, likely idle.
            rep_status = 'up'

        # Consistency: Down interfaces have 0 rate
        if rep_status == 'down':
            rep_rx = 0.0
            rep_tx = 0.0
            # If we are confident it's down, we are confident rates are 0
            conf_rx = max(conf_rx, conf_status)
            conf_tx = max(conf_tx, conf_status)

        # Final Record
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
