# EVOLVE-BLOCK-START
from typing import Dict, Any, Tuple, List
import math

def repair_network_telemetry(telemetry: Dict[str, Dict[str, Any]],
                             topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:
    """
    Repairs network telemetry using an Iterative Bayesian Consensus algorithm.
    It systematically fuses Link Symmetry and Router Flow Conservation evidence.

    Algorithm:
    1.  **Symmetry Analysis**: Identify consistent links and harden them.
    2.  **Iterative Optimization**: For suspect links, test hypotheses (Local, Peer, Zero)
        against router flow balance constraints.
    3.  **Bayesian Confidence**: Calculate confidence based on hypothesis likelihood and
        absolute goodness-of-fit.
    """

    # --- Configuration ---
    SYMMETRY_TOLERANCE = 0.02
    CONSERVATION_TOLERANCE_PCT = 0.03
    MIN_SIGNIFICANT_FLOW = 0.5
    ITERATIONS = 5

    # --- Helper Structures ---
    if_to_router = {}
    for r_id, if_list in topology.items():
        for i_id in if_list:
            if_to_router[i_id] = r_id

    # Group interfaces into Links for unique processing
    links = {}
    processed_ifs = set()

    for if_id, data in telemetry.items():
        if if_id in processed_ifs: continue

        peer_id = data.get('connected_to')
        if peer_id and peer_id in telemetry:
            link_key = tuple(sorted([if_id, peer_id]))
            links[link_key] = {'type': 'internal', 'if1': if_id, 'if2': peer_id}
            processed_ifs.add(if_id)
            processed_ifs.add(peer_id)
        else:
            links[(if_id,)] = {'type': 'external', 'if1': if_id, 'if2': None}
            processed_ifs.add(if_id)

    # --- Step 1: Initialization & Link Symmetry ---

    # Current best estimates: {if_id: {'rx': val, 'tx': val}}
    current_estimates = {}
    # Confidence scores: {if_id: {'rx': conf, 'tx': conf}}
    estimate_confidence = {}

    # List of suspect flows (link + direction) to solve
    suspect_flows = []

    for link_key, info in links.items():
        if1 = info['if1']
        if2 = info['if2']
        d1 = telemetry[if1]

        # Initialize default estimates (will be overwritten if consistent)
        current_estimates[if1] = {'rx': d1.get('rx_rate', 0.0), 'tx': d1.get('tx_rate', 0.0)}
        estimate_confidence[if1] = {'rx': 0.5, 'tx': 0.5} # Low default

        if if2:
            d2 = telemetry[if2]
            current_estimates[if2] = {'rx': d2.get('rx_rate', 0.0), 'tx': d2.get('tx_rate', 0.0)}
            estimate_confidence[if2] = {'rx': 0.5, 'tx': 0.5}

            # Check Forward: if1 TX -> if2 RX
            val_tx = d1.get('tx_rate', 0.0)
            val_rx = d2.get('rx_rate', 0.0)
            denom = max(val_tx, val_rx, 1.0)
            if abs(val_tx - val_rx) / denom < SYMMETRY_TOLERANCE:
                avg = (val_tx + val_rx) / 2.0
                current_estimates[if1]['tx'] = avg
                current_estimates[if2]['rx'] = avg
                estimate_confidence[if1]['tx'] = 0.95
                estimate_confidence[if2]['rx'] = 0.95
            else:
                # Suspect: Add to solver
                suspect_flows.append({
                    'src': if1, 'dst': if2,
                    'val_src': val_tx, 'val_dst': val_rx,
                    'type': 'internal'
                })

            # Check Backward: if2 TX -> if1 RX
            val_tx = d2.get('tx_rate', 0.0)
            val_rx = d1.get('rx_rate', 0.0)
            denom = max(val_tx, val_rx, 1.0)
            if abs(val_tx - val_rx) / denom < SYMMETRY_TOLERANCE:
                avg = (val_tx + val_rx) / 2.0
                current_estimates[if2]['tx'] = avg
                current_estimates[if1]['rx'] = avg
                estimate_confidence[if2]['tx'] = 0.95
                estimate_confidence[if1]['rx'] = 0.95
            else:
                suspect_flows.append({
                    'src': if2, 'dst': if1,
                    'val_src': val_tx, 'val_dst': val_rx,
                    'type': 'internal'
                })
        else:
            # External Link - Mark as suspect to allow validation against router balance
            # even though we only have one measurement source.
            suspect_flows.append({
                'src': if1, 'dst': None,
                'val_src': d1.get('tx_rate', 0.0), 'val_dst': None,
                'type': 'external_tx'
            })
            # External RX is effectively an input to the router
            suspect_flows.append({
                'src': None, 'dst': if1,
                'val_src': None, 'val_dst': d1.get('rx_rate', 0.0),
                'type': 'external_rx'
            })

            # Set tentative high confidence for external, will be lowered if conservation fails
            estimate_confidence[if1]['tx'] = 0.8
            estimate_confidence[if1]['rx'] = 0.8

    # --- Step 2: Iterative Bayesian Resolution ---

    def get_router_imbalance(rid):
        if not rid: return 0.0, 1.0
        if_list = topology.get(rid, [])
        total_in = 0.0
        total_out = 0.0
        for iid in if_list:
            total_in += current_estimates[iid]['rx']
            total_out += current_estimates[iid]['tx']
        return (total_in - total_out), max(total_in, total_out, 1.0)

    for _ in range(ITERATIONS):
        updates = []

        for flow in suspect_flows:
            src = flow['src']
            dst = flow['dst']

            # Setup Hypotheses
            hypotheses = []

            if flow['type'] == 'internal':
                # Hypotheses: Source Meas, Dest Meas, Zero (Phantom/Down)
                hypotheses.append(flow['val_src'])
                hypotheses.append(flow['val_dst'])
                hypotheses.append(0.0)
                # Remove duplicates to avoid split probability
                hypotheses = sorted(list(set(hypotheses)))

                rid_src = if_to_router.get(src)
                rid_dst = if_to_router.get(dst)

                saved_src_tx = current_estimates[src]['tx']
                saved_dst_rx = current_estimates[dst]['rx']

                scores = []
                for h in hypotheses:
                    # Apply
                    current_estimates[src]['tx'] = h
                    current_estimates[dst]['rx'] = h

                    # Eval
                    imb_src, f_src = get_router_imbalance(rid_src)
                    imb_dst, f_dst = get_router_imbalance(rid_dst)

                    sig_src = max(f_src * CONSERVATION_TOLERANCE_PCT, 1.0)
                    sig_dst = max(f_dst * CONSERVATION_TOLERANCE_PCT, 1.0)

                    p_src = math.exp(-abs(imb_src) / sig_src)
                    p_dst = math.exp(-abs(imb_dst) / sig_dst)

                    scores.append(p_src * p_dst)

                # Restore
                current_estimates[src]['tx'] = saved_src_tx
                current_estimates[dst]['rx'] = saved_dst_rx

                # Pick Winner
                total_score = sum(scores) + 1e-20
                probs = [s / total_score for s in scores]
                best_idx = probs.index(max(probs))
                winner = hypotheses[best_idx]

                # Confidence: Relative Certainty * Absolute Fit Quality
                # Fit Quality is sqrt of likelihoods (geometric mean)
                fit_quality = math.sqrt(scores[best_idx])
                conf = probs[best_idx] * fit_quality

                updates.append((src, 'tx', winner, conf))
                updates.append((dst, 'rx', winner, conf))

            elif flow['type'] == 'external_tx':
                # Src is local interface, Dst is None.
                # Hypotheses: Local Meas, Zero
                val = flow['val_src']
                rid = if_to_router.get(src)
                saved = current_estimates[src]['tx']

                # Test Local Val
                current_estimates[src]['tx'] = val
                imb, f = get_router_imbalance(rid)
                sig = max(f * CONSERVATION_TOLERANCE_PCT, 1.0)
                score_val = math.exp(-abs(imb) / sig)

                # Test Zero (optional, but good if external link is dead)
                current_estimates[src]['tx'] = 0.0
                imb_z, f_z = get_router_imbalance(rid)
                sig_z = max(f_z * CONSERVATION_TOLERANCE_PCT, 1.0)
                score_zero = math.exp(-abs(imb_z) / sig_z)

                current_estimates[src]['tx'] = saved

                if score_val >= score_zero:
                    winner = val
                    conf = score_val # No peer, so just absolute fit
                else:
                    winner = 0.0
                    conf = score_zero

                updates.append((src, 'tx', winner, conf))

            elif flow['type'] == 'external_rx':
                # Dst is local interface, Src is None
                val = flow['val_dst']
                rid = if_to_router.get(dst)
                saved = current_estimates[dst]['rx']

                current_estimates[dst]['rx'] = val
                imb, f = get_router_imbalance(rid)
                sig = max(f * CONSERVATION_TOLERANCE_PCT, 1.0)
                score_val = math.exp(-abs(imb) / sig)

                current_estimates[dst]['rx'] = 0.0
                imb_z, f_z = get_router_imbalance(rid)
                sig_z = max(f_z * CONSERVATION_TOLERANCE_PCT, 1.0)
                score_zero = math.exp(-abs(imb_z) / sig_z)

                current_estimates[dst]['rx'] = saved

                if score_val >= score_zero:
                    winner = val
                    conf = score_val
                else:
                    winner = 0.0
                    conf = score_zero

                updates.append((dst, 'rx', winner, conf))

        # Apply Updates
        for if_id, metric, val, conf in updates:
            current_estimates[if_id][metric] = val
            estimate_confidence[if_id][metric] = conf

    # --- Step 3: Status Inference ---

    result = {}

    for if_id, data in telemetry.items():
        orig_rx = data.get('rx_rate', 0.0)
        orig_tx = data.get('tx_rate', 0.0)
        orig_status = data.get('interface_status', 'unknown')

        est_rx = current_estimates[if_id]['rx']
        conf_rx = estimate_confidence[if_id]['rx']
        est_tx = current_estimates[if_id]['tx']
        conf_tx = estimate_confidence[if_id]['tx']

        peer_id = data.get('connected_to')
        peer_status = 'unknown'
        if peer_id and peer_id in telemetry:
            peer_status = telemetry[peer_id].get('interface_status', 'unknown')

        has_traffic = (est_rx > MIN_SIGNIFICANT_FLOW) or (est_tx > MIN_SIGNIFICANT_FLOW)

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
        elif orig_status == 'up':
            # Traffic is 0, Peer is UP (or unknown).
            # Could be idle. Keep UP.
            rep_status = 'up'

        # Post-process: If status is DOWN, force rates to 0
        if rep_status == 'down':
            est_rx = 0.0
            est_tx = 0.0
            conf_rx = max(conf_rx, 0.95)
            conf_tx = max(conf_tx, 0.95)

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
