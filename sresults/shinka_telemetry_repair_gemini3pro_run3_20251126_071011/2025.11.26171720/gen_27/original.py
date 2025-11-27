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

    # Configuration
    TOLERANCE = 0.02  # 2% hardening threshold
    ABS_TOLERANCE = 1.0 # 1 Mbps absolute threshold for noise/idle checks
    MIN_FLOW_SIGNIFICANCE = 0.1  # Threshold for "active" link status

    # Helper: Build Interface->Router map from topology
    if_to_router = {}
    for r_id, if_list in topology.items():
        for i_id in if_list:
            if_to_router[i_id] = r_id

    # --- PHASE 1: Link Symmetry Analysis & Hypothesis Generation ---
    link_analysis = {}

    for if_id, data in telemetry.items():
        # Get Local Data
        local_rx = data.get('rx_rate', 0.0)
        local_tx = data.get('tx_rate', 0.0)

        # Get Peer Data
        peer_id = data.get('connected_to')
        peer_data = telemetry.get(peer_id, {}) if peer_id else {}

        peer_tx = peer_data.get('tx_rate', None)
        peer_rx = peer_data.get('rx_rate', None)

        # Analyze RX Symmetry
        rx_candidates = {'local': local_rx}
        rx_symmetry_score = 0.0

        if peer_tx is not None:
            rx_candidates['peer'] = peer_tx
            denom = max(local_rx, peer_tx, 1.0)
            diff = abs(local_rx - peer_tx)

            # Use absolute or relative tolerance
            if diff < ABS_TOLERANCE or diff / denom < TOLERANCE:
                rx_symmetry_score = 1.0
                # Consensus logic: if one is 0 and match is good, likely 0. Else average.
                if min(local_rx, peer_tx) == 0.0:
                    rx_candidates['consensus'] = 0.0
                else:
                    rx_candidates['consensus'] = (local_rx + peer_tx) / 2.0
            else:
                # Disagreement
                rx_symmetry_score = max(0.0, 1.0 - (diff / denom))
        else:
            rx_symmetry_score = 0.5

        # Analyze TX Symmetry
        tx_candidates = {'local': local_tx}
        tx_symmetry_score = 0.0

        if peer_rx is not None:
            tx_candidates['peer'] = peer_rx
            denom = max(local_tx, peer_rx, 1.0)
            diff = abs(local_tx - peer_rx)

            if diff < ABS_TOLERANCE or diff / denom < TOLERANCE:
                tx_symmetry_score = 1.0
                if min(local_tx, peer_rx) == 0.0:
                    tx_candidates['consensus'] = 0.0
                else:
                    tx_candidates['consensus'] = (local_tx + peer_rx) / 2.0
            else:
                tx_symmetry_score = max(0.0, 1.0 - (diff / denom))
        else:
            tx_symmetry_score = 0.5

        link_analysis[if_id] = {
            'rx': {'candidates': rx_candidates, 'symmetry': rx_symmetry_score},
            'tx': {'candidates': tx_candidates, 'symmetry': tx_symmetry_score}
        }

    # --- PHASE 2: Router Conservation Optimization ---
    final_decisions = {}

    for router_id, if_list in topology.items():
        # Identify variables for optimization
        reliable_inputs = 0.0
        reliable_outputs = 0.0
        unreliable_ifs = [] # List of {'key': (if_id, type), 'local': val, 'peer': val, 'impact': float}

        # Base flow from reliable links
        for if_id in if_list:
            if if_id not in link_analysis: continue

            for metric, inputs_dict, outputs_dict in [('rx', True, False), ('tx', False, True)]:
                info = link_analysis[if_id][metric]
                # High symmetry -> Reliable
                if info['symmetry'] > 0.95:
                    val = info['candidates'].get('consensus', info['candidates']['local'])
                    if inputs_dict: reliable_inputs += val
                    if outputs_dict: reliable_outputs += val
                else:
                    # Unreliable -> Variable
                    cands = info['candidates']
                    # Candidates: Default to Peer (H0), Local (H1)
                    peer_val = cands.get('peer', cands['local'])
                    local_val = cands['local']

                    # Impact on (In - Out) if we switch from Peer -> Local
                    diff = local_val - peer_val
                    impact = diff if inputs_dict else -diff

                    unreliable_ifs.append({
                        'key': (if_id, metric),
                        'local': local_val,
                        'peer': peer_val,
                        'impact': impact,
                        'is_input': inputs_dict
                    })

                    # Add baseline (Peer) to sums
                    if inputs_dict: reliable_inputs += peer_val
                    else: reliable_outputs += peer_val

        # Optimization: Select subset of swaps to minimize |In - Out|
        # Base Imbalance (with all Peer values)
        base_net_flow = reliable_inputs - reliable_outputs

        n_vars = len(unreliable_ifs)
        best_mask = 0
        min_imbalance = abs(base_net_flow)

        # Brute force if small (covers most routers)
        valid_masks = []

        if n_vars <= 12:
            # Find global minimum
            for mask in range(1 << n_vars):
                current_impact = 0.0
                for i in range(n_vars):
                    if (mask >> i) & 1:
                        current_impact += unreliable_ifs[i]['impact']

                imbalance = abs(base_net_flow + current_impact)
                if imbalance < min_imbalance:
                    min_imbalance = imbalance
                    best_mask = mask

            # Find ambiguous solutions (those close to min_imbalance)
            total_flow = max(reliable_inputs, reliable_outputs, 1.0) # Approx
            ambiguity_threshold = min_imbalance + max(1.0, 0.05 * total_flow)

            for mask in range(1 << n_vars):
                current_impact = 0.0
                for i in range(n_vars):
                    if (mask >> i) & 1:
                        current_impact += unreliable_ifs[i]['impact']

                if abs(base_net_flow + current_impact) <= ambiguity_threshold:
                    valid_masks.append(mask)
        else:
            # Greedy fallback for huge routers (rare)
            # Just use base_mask=0 for huge routers to be safe
            valid_masks = [0]

        # Determine Final Values & Ambiguity
        decisions = {}
        ambiguity_scores = {} # key -> 0.0 to 1.0 (1.0 = ambiguous)

        # Calculate ambiguity per variable
        if valid_masks:
            for i in range(n_vars):
                # Check if bit i varies across valid_masks
                first_val = (valid_masks[0] >> i) & 1
                is_constant = all(((m >> i) & 1) == first_val for m in valid_masks)
                ambiguity_scores[unreliable_ifs[i]['key']] = 0.0 if is_constant else 1.0

        # Apply Best Mask
        h1_inputs = reliable_inputs
        h1_outputs = reliable_outputs

        for i in range(n_vars):
            item = unreliable_ifs[i]
            key = item['key']
            use_local = (best_mask >> i) & 1

            val = item['local'] if use_local else item['peer']
            decisions[key] = val

        # Re-sum for accurate final imbalance
        final_inputs = 0.0
        final_outputs = 0.0

        for if_id in if_list:
            if if_id not in link_analysis: continue

            # RX
            if (if_id, 'rx') in decisions:
                final_inputs += decisions[(if_id, 'rx')]
            elif link_analysis[if_id]['rx']['symmetry'] > 0.95:
                val = link_analysis[if_id]['rx']['candidates'].get('consensus', link_analysis[if_id]['rx']['candidates']['local'])
                final_inputs += val

            # TX
            if (if_id, 'tx') in decisions:
                final_outputs += decisions[(if_id, 'tx')]
            elif link_analysis[if_id]['tx']['symmetry'] > 0.95:
                val = link_analysis[if_id]['tx']['candidates'].get('consensus', link_analysis[if_id]['tx']['candidates']['local'])
                final_outputs += val

        # Conservation Score
        final_imbalance = abs(final_inputs - final_outputs)
        max_flow = max(final_inputs, final_outputs, 1.0)
        conservation_score = max(0.0, 1.0 - (final_imbalance / max_flow))

        # Store Results
        for if_id in if_list:
            if if_id not in final_decisions: final_decisions[if_id] = {}

            for metric in ['rx', 'tx']:
                key = (if_id, metric)
                if key in decisions:
                    val = decisions[key]
                    is_ambiguous = ambiguity_scores.get(key, 0.0)
                    # Confidence: Base 0.5. Bonus from Conservation. Penalty from Ambiguity.
                    # If ambiguous (score=1.0), factor is 0.5. If stable (score=0.0), factor is 1.0.
                    stability = 1.0 - (0.5 * is_ambiguous)
                    conf = 0.5 + (0.4 * conservation_score * stability)
                    final_decisions[if_id][metric] = (val, conf)
                else:
                    # Reliable
                    info = link_analysis[if_id][metric]
                    val = info['candidates'].get('consensus', info['candidates']['local'])
                    # Reliable links are trusted, but extreme router violation reduces conf slightly
                    final_decisions[if_id][metric] = (val, 0.9 + 0.1 * conservation_score)

    # --- PHASE 3: Status Inference & Assembly ---
    result = {}

    for if_id, data in telemetry.items():
        orig_rx = data.get('rx_rate', 0.0)
        orig_tx = data.get('tx_rate', 0.0)
        orig_status = data.get('interface_status', 'unknown')

        dec = final_decisions.get(if_id, {})
        rep_rx, conf_rx = dec.get('rx', (orig_rx, 0.0))
        rep_tx, conf_tx = dec.get('tx', (orig_tx, 0.0))

        peer_id = data.get('connected_to')
        peer_status = 'unknown'
        if peer_id and peer_id in telemetry:
            peer_status = telemetry[peer_id].get('interface_status', 'unknown')

        # Status Logic
        has_traffic = rep_rx > MIN_FLOW_SIGNIFICANCE or rep_tx > MIN_FLOW_SIGNIFICANCE

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
            if orig_status == 'up' and peer_status == 'down':
                rep_status = 'down'
                conf_status = 0.7

        if rep_status == 'down':
            rep_rx = 0.0
            rep_tx = 0.0
            conf_rx = max(conf_rx, 0.9)
            conf_tx = max(conf_tx, 0.9)

        repaired_entry = {}
        repaired_entry['rx_rate'] = (orig_rx, rep_rx, conf_rx)
        repaired_entry['tx_rate'] = (orig_tx, rep_tx, conf_tx)
        repaired_entry['interface_status'] = (orig_status, rep_status, conf_status)

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