# EVOLVE-BLOCK-START
from typing import Dict, Any, Tuple, List
import math

def repair_network_telemetry(telemetry: Dict[str, Dict[str, Any]],
                             topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:
    """
    Repairs network telemetry using a consensus mechanism between Link Symmetry
    and Router Flow Conservation invariants.

    Algorithm:
    1. Identify 'clean' (consistent) and 'suspect' (inconsistent) links.
    2. For suspect links, generate hypotheses: True Value = Local Measurement vs Peer Measurement.
    3. Validate hypotheses using Router Flow Conservation (Input Flux == Output Flux).
    4. Select the hypothesis that minimizes global network violation.
    5. Calibrate confidence based on the agreement strength of invariants.
    """

    # Configuration
    TOLERANCE = 0.02  # 2% hardening threshold
    MIN_FLOW_SIGNIFICANCE = 0.1  # Ignore micro-flows for status logic

    # Data Structures for repair
    # Structure: {if_id: {'rx': (val, conf), 'tx': (val, conf), 'status': (val, conf)}}
    repairs = {}

    # Helper: Build Interface->Router map from topology
    if_to_router = {}
    for r_id, if_list in topology.items():
        for i_id in if_list:
            if_to_router[i_id] = r_id

    # --- PHASE 1: Link Symmetry Analysis & Hypothesis Generation ---
    # We gather candidate values for every interface's RX and TX.
    # Candidates = [Local_Measurement, Peer_Measurement]

    link_analysis = {}

    for if_id, data in telemetry.items():
        # Get Local Data
        local_rx = data.get('rx_rate', 0.0)
        local_tx = data.get('tx_rate', 0.0)

        # Get Peer Data
        peer_id = data.get('connected_to')
        peer_data = telemetry.get(peer_id, {}) if peer_id else {}

        peer_tx = peer_data.get('tx_rate', None) # Peer's TX corresponds to My RX
        peer_rx = peer_data.get('rx_rate', None) # Peer's RX corresponds to My TX

        # Analyze RX Symmetry
        rx_candidates = {'local': local_rx}
        rx_symmetry_score = 0.0 # 0=Bad, 1=Good

        if peer_tx is not None:
            rx_candidates['peer'] = peer_tx
            # Calculate symmetry error
            denom = max(local_rx, peer_tx, 1.0)
            diff = abs(local_rx - peer_tx)
            if diff / denom < TOLERANCE:
                rx_symmetry_score = 1.0
                # If they agree, average them for higher precision
                rx_candidates['consensus'] = (local_rx + peer_tx) / 2.0
            else:
                # Disagreement
                rx_symmetry_score = max(0.0, 1.0 - (diff / denom))
        else:
            # No peer data, trust local but with lower confidence baseline?
            # Actually, without peer, symmetry is unknown. Assume 1.0 for processing but track missing.
            rx_symmetry_score = 0.5

        # Analyze TX Symmetry
        tx_candidates = {'local': local_tx}
        tx_symmetry_score = 0.0

        if peer_rx is not None:
            tx_candidates['peer'] = peer_rx
            denom = max(local_tx, peer_rx, 1.0)
            diff = abs(local_tx - peer_rx)
            if diff / denom < TOLERANCE:
                tx_symmetry_score = 1.0
                tx_candidates['consensus'] = (local_tx + peer_rx) / 2.0
            else:
                tx_symmetry_score = max(0.0, 1.0 - (diff / denom))
        else:
            tx_symmetry_score = 0.5

        link_analysis[if_id] = {
            'rx': {'candidates': rx_candidates, 'symmetry': rx_symmetry_score},
            'tx': {'candidates': tx_candidates, 'symmetry': tx_symmetry_score}
        }

    # --- PHASE 2: Global Flow Optimization ---
    # We use an iterative approach to solve for flow conservation while maintaining link symmetry.

    # 1. Initialize Estimates & Confidence
    # Structure: estimates[if_id] = {'rx': val, 'tx': val}
    estimates = {}
    confidences = {} # {if_id: {'rx': conf, 'tx': conf}}

    # Track which flows are suspect and need solving
    # list of {'src_if': id, 'dst_if': id, 'candidates': [v1, v2, 0.0]}
    suspect_flows = []

    # Initialize default structure for all interfaces
    for if_id, data in telemetry.items():
        if if_id not in estimates: estimates[if_id] = {}
        if if_id not in confidences: confidences[if_id] = {}

        # RX Handling
        rx_an = link_analysis[if_id]['rx']
        if rx_an['symmetry'] > 0.95:
            # Reliable
            val = rx_an['candidates'].get('consensus', rx_an['candidates']['local'])
            estimates[if_id]['rx'] = val
            confidences[if_id]['rx'] = 0.95
        else:
            # Unreliable - wait for flow processing
            estimates[if_id]['rx'] = rx_an['candidates'].get('consensus', rx_an['candidates']['local'])
            confidences[if_id]['rx'] = 0.5

        # TX Handling
        tx_an = link_analysis[if_id]['tx']
        if tx_an['symmetry'] > 0.95:
            val = tx_an['candidates'].get('consensus', tx_an['candidates']['local'])
            estimates[if_id]['tx'] = val
            confidences[if_id]['tx'] = 0.95
        else:
            estimates[if_id]['tx'] = tx_an['candidates'].get('consensus', tx_an['candidates']['local'])
            confidences[if_id]['tx'] = 0.5

    # 2. Build Suspect Flow List
    for if_id, data in telemetry.items():
        peer_id = data.get('connected_to')

        # Handle TX Flow (My TX -> Peer RX)
        if peer_id and peer_id in telemetry:
            # My TX analysis
            tx_an = link_analysis[if_id]['tx']
            if tx_an['symmetry'] <= 0.95:
                # This flow is suspect
                cands = [tx_an['candidates']['local']]
                if 'peer' in tx_an['candidates']:
                    cands.append(tx_an['candidates']['peer'])

                # Add 0.0 as candidate for suspect links (phantom traffic hypothesis)
                cands.append(0.0)

                # Deduplicate
                cands = sorted(list(set(cands)))

                suspect_flows.append({
                    'src': if_id,
                    'dst': peer_id,
                    'candidates': cands
                })
        else:
            # Edge link or no peer.
            tx_an = link_analysis[if_id]['tx']
            if tx_an['symmetry'] <= 0.9: # Lower threshold for edge trust
                 cands = [tx_an['candidates']['local'], 0.0]
                 suspect_flows.append({
                     'src': if_id,
                     'dst': None,
                     'candidates': cands
                 })

            # Edge RX
            rx_an = link_analysis[if_id]['rx']
            if rx_an['symmetry'] <= 0.9:
                 cands = [rx_an['candidates']['local'], 0.0]
                 suspect_flows.append({
                     'src': None,
                     'dst': if_id,
                     'candidates': cands
                 })

    # 3. Iterative Solver
    ITERATIONS = 5

    def get_imbalance(rid):
        if not rid: return 0.0, 1.0
        in_flow = 0.0
        out_flow = 0.0
        for iid in topology.get(rid, []):
            if iid in estimates:
                in_flow += estimates[iid].get('rx', 0.0)
                out_flow += estimates[iid].get('tx', 0.0)
        return (in_flow - out_flow), max(in_flow, out_flow, 1.0)

    for _ in range(ITERATIONS):
        updates = [] # Store updates to apply synchronously

        for flow in suspect_flows:
            src = flow['src']
            dst = flow['dst']
            cands = flow['candidates']

            # Routers
            r_src = if_to_router.get(src) if src else None
            r_dst = if_to_router.get(dst) if dst else None

            # Current values to restore
            old_src_val = estimates[src]['tx'] if src else 0.0
            old_dst_val = estimates[dst]['rx'] if dst else 0.0

            best_val = old_src_val

            # Score candidates
            # Score = P(src_conservation) * P(dst_conservation)
            total_prob = 0.0
            scored_cands = []

            for val in cands:
                # Apply hypothesis
                if src: estimates[src]['tx'] = val
                if dst: estimates[dst]['rx'] = val

                # Check Imbalance
                score_src = 1.0
                if r_src:
                    imb, flux = get_imbalance(r_src)
                    sigma = max(flux * 0.03, 1.0) # 3% tolerance
                    score_src = math.exp(-abs(imb)/sigma)

                score_dst = 1.0
                if r_dst:
                    imb, flux = get_imbalance(r_dst)
                    sigma = max(flux * 0.03, 1.0)
                    score_dst = math.exp(-abs(imb)/sigma)

                score = score_src * score_dst
                scored_cands.append((val, score))
                total_prob += score

            # Restore
            if src: estimates[src]['tx'] = old_src_val
            if dst: estimates[dst]['rx'] = old_dst_val

            # Normalize and pick winner
            if total_prob > 0:
                # Find max score
                scored_cands.sort(key=lambda x: x[1], reverse=True)
                winner_val, winner_raw_score = scored_cands[0]

                # Confidence: Probability of winner * Absolute fit
                prob = winner_raw_score / total_prob
                # Absolute fit: geometric mean of agreement (0-1)
                fit = math.sqrt(winner_raw_score)

                conf = prob * fit
                conf = max(0.01, min(0.99, conf))

                updates.append((src, dst, winner_val, conf))

        # Apply updates
        for src, dst, val, conf in updates:
            if src:
                estimates[src]['tx'] = val
                confidences[src]['tx'] = conf
            if dst:
                estimates[dst]['rx'] = val
                confidences[dst]['rx'] = conf

    # 4. Map back to final_decisions format for Phase 3
    final_decisions = {}
    for if_id in telemetry:
        final_decisions[if_id] = {}
        final_decisions[if_id]['rx'] = (estimates[if_id].get('rx', 0.0), confidences[if_id].get('rx', 1.0))
        final_decisions[if_id]['tx'] = (estimates[if_id].get('tx', 0.0), confidences[if_id].get('tx', 1.0))

    # --- PHASE 3: Status Inference & Final Assembly ---

    result = {}

    for if_id, data in telemetry.items():
        orig_rx = data.get('rx_rate', 0.0)
        orig_tx = data.get('tx_rate', 0.0)
        orig_status = data.get('interface_status', 'unknown')

        # Get Repaired Rates
        dec = final_decisions.get(if_id, {})
        rep_rx, conf_rx = dec.get('rx', (orig_rx, 0.0))
        rep_tx, conf_tx = dec.get('tx', (orig_tx, 0.0))

        # Get Peer Status for consistency
        peer_id = data.get('connected_to')
        peer_status = 'unknown'
        if peer_id and peer_id in telemetry:
            peer_status = telemetry[peer_id].get('interface_status', 'unknown')

        # Status Logic
        # Rule 1: Significant Traffic -> UP
        has_traffic = rep_rx > MIN_FLOW_SIGNIFICANCE or rep_tx > MIN_FLOW_SIGNIFICANCE

        rep_status = orig_status
        conf_status = 1.0

        if has_traffic:
            rep_status = 'up'
            # If we contradicted original status, confidence depends on rate confidence
            if orig_status != 'up':
                conf_status = (conf_rx + conf_tx) / 2.0
        elif peer_status == 'down':
            # Rule 2: Peer is down and no traffic -> DOWN
            rep_status = 'down'
            if orig_status != 'down':
                conf_status = 0.9
        else:
            # Rule 3: No traffic, peer UP. Ambiguous.
            # If original says DOWN, trust it. If UP, trust it (idle).
            # Lower confidence if mismatch with peer
            rep_status = orig_status
            if orig_status == 'up' and peer_status == 'down':
                rep_status = 'down' # Safe default
                conf_status = 0.7

        # Consistency enforcement: If DOWN, rates must be 0
        if rep_status == 'down':
            rep_rx = 0.0
            rep_tx = 0.0
            conf_rx = max(conf_rx, 0.9) # High confidence in 0 if down
            conf_tx = max(conf_tx, 0.9)

        # Structure Output
        repaired_entry = {}
        repaired_entry['rx_rate'] = (orig_rx, rep_rx, conf_rx)
        repaired_entry['tx_rate'] = (orig_tx, rep_tx, conf_tx)
        repaired_entry['interface_status'] = (orig_status, rep_status, conf_status)

        # Metadata pass-through
        for k in ['connected_to', 'local_router', 'remote_router']:
            if k in data:
                repaired_entry[k] = data[k]

        result[if_id] = repaired_entry

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
