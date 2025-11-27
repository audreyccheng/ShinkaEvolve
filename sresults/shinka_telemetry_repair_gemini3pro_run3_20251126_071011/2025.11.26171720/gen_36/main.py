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

    # --- PHASE 2: Router Conservation Voting ---
    # We iterate over routers to resolve conflicts using Flow Conservation.
    # We use an iterative "Coordinate Descent" approach to allow decisions to propagate.

    ITERATIONS = 4

    # Store the currently accepted flow value for every interface metric
    # Key: (if_id, metric_type) where metric_type is 'rx' or 'tx'
    # Value: float flow_rate
    current_decisions = {}

    # Track which decisions are fixed (Reliable links)
    fixed_decisions = set()

    # Initialization
    for if_id, analysis in link_analysis.items():
        for metric in ['rx', 'tx']:
            data = analysis[metric]
            candidates = data['candidates']

            if data['symmetry'] > 0.95:
                # High symmetry - lock it in
                val = candidates.get('consensus', candidates['local'])
                current_decisions[(if_id, metric)] = val
                fixed_decisions.add((if_id, metric))
            else:
                # Low symmetry - init with average to be neutral
                # If we have peer, avg. If not, local.
                if 'peer' in candidates:
                    val = (candidates['local'] + candidates['peer']) / 2.0
                else:
                    val = candidates['local']
                current_decisions[(if_id, metric)] = val

    # Iterative Refinement Loop
    for _ in range(ITERATIONS):
        # We verify conservation at each router and update the 'mutable' decisions

        for router_id, if_list in topology.items():
            # Calculate current flows and imbalance
            router_in = 0.0
            router_out = 0.0

            # Identify mutable interfaces on this router
            mutable_ifs = []

            for if_id in if_list:
                # RX Flow
                rx_val = current_decisions.get((if_id, 'rx'), 0.0)
                router_in += rx_val
                if (if_id, 'rx') not in fixed_decisions and if_id in link_analysis:
                    mutable_ifs.append((if_id, 'rx'))

                # TX Flow
                tx_val = current_decisions.get((if_id, 'tx'), 0.0)
                router_out += tx_val
                if (if_id, 'tx') not in fixed_decisions and if_id in link_analysis:
                    mutable_ifs.append((if_id, 'tx'))

            current_imbalance = router_in - router_out

            if not mutable_ifs:
                continue

            # Local optimization: Adjust each mutable interface to minimize router imbalance.
            # We try specific candidates: Local Measurement, Peer Measurement, and Zero.

            for key in mutable_ifs:
                if_id, metric = key
                current_val = current_decisions[key]

                # Candidates
                candidates = link_analysis[if_id][metric]['candidates']
                possible_values = [candidates['local']]
                if 'peer' in candidates:
                    possible_values.append(candidates['peer'])
                possible_values.append(0.0) # Zero flow hypothesis (Link Down / Phantom Traffic)

                best_val = current_val
                best_abs_imb = abs(current_imbalance)

                # Calculate contribution of this flow to imbalance:
                # Imb = In - Out.
                # RX contributes (+), TX contributes (-)
                sign = 1.0 if metric == 'rx' else -1.0
                remainder = current_imbalance - (sign * current_val)

                for val in possible_values:
                    # New Imb = Remainder + (sign * val)
                    new_imb = remainder + (sign * val)
                    if abs(new_imb) < best_abs_imb:
                        best_abs_imb = abs(new_imb)
                        best_val = val

                if best_val != current_val:
                    # Apply update immediately (Gauss-Seidel)
                    diff = best_val - current_val
                    current_decisions[key] = best_val
                    current_imbalance += (sign * diff)

                    # Propagate to Peer (Link Consistency)
                    # If I change my TX, I should change peer's RX to match, if mutable.
                    peer_id = telemetry[if_id].get('connected_to')
                    if peer_id:
                        peer_metric = 'rx' if metric == 'tx' else 'tx'
                        peer_key = (peer_id, peer_metric)
                        if peer_key in current_decisions and peer_key not in fixed_decisions:
                            current_decisions[peer_key] = best_val
                            # Note: We don't update current_imbalance here as peer is on a different router

    # Final Scoring and Assembly
    final_decisions = {}

    # Calculate final conservation scores per router
    router_scores = {}
    for router_id, if_list in topology.items():
        r_in = sum(current_decisions.get((i, 'rx'), 0.0) for i in if_list)
        r_out = sum(current_decisions.get((i, 'tx'), 0.0) for i in if_list)

        imb = abs(r_in - r_out)
        flow = max(r_in, r_out, 1.0)

        # Exponential score: 1.0 -> Perfect balance, decays with imbalance
        score = math.exp(-imb / (flow * 0.05 + 1.0))
        router_scores[router_id] = score

    for if_id in telemetry:
        final_decisions[if_id] = {}

        # Calculate final confidence
        local_rid = if_to_router.get(if_id)
        r_score = router_scores.get(local_rid, 0.5) if local_rid else 0.5

        for metric in ['rx', 'tx']:
            val = current_decisions.get((if_id, metric), 0.0)
            is_reliable = (if_id, metric) in fixed_decisions

            if is_reliable:
                # High confidence for reliable links
                base_conf = 0.95
                # Slight modulation by router health (if router is chaos, maybe we are wrong?)
                conf = base_conf * (0.9 + 0.1 * r_score)
            else:
                # Lower baseline for repaired links
                base_conf = 0.6
                # Strong dependency on router conservation success
                # If r_score is 1.0 (perfect balance), conf -> 0.9
                # If r_score is 0.0 (bad balance), conf -> 0.3
                conf = base_conf * 0.5 + (0.6 * r_score)

            conf = min(1.0, max(0.0, conf))
            final_decisions[if_id][metric] = (val, conf)

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