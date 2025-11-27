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
    
    final_decisions = {} # if_id -> {'rx': (val, conf), ...}
    
    for router_id, if_list in topology.items():
        
        # 2a. Identify Reliable vs Unreliable flows on this router
        reliable_inputs = 0.0
        reliable_outputs = 0.0
        
        unreliable_ifs = [] # List of (if_id, type 'rx'|'tx')
        
        # Preliminary pass: calculate "Trusted" flow mass
        for if_id in if_list:
            if if_id not in link_analysis: continue
            
            # RX Check
            rx_info = link_analysis[if_id]['rx']
            if rx_info['symmetry'] > 0.95:
                # High symmetry, assume correct
                val = rx_info['candidates'].get('consensus', rx_info['candidates']['local'])
                reliable_inputs += val
            else:
                unreliable_ifs.append((if_id, 'rx'))
                
            # TX Check
            tx_info = link_analysis[if_id]['tx']
            if tx_info['symmetry'] > 0.95:
                val = tx_info['candidates'].get('consensus', tx_info['candidates']['local'])
                reliable_outputs += val
            else:
                unreliable_ifs.append((if_id, 'tx'))
        
        # 2b. Resolve Unreliable Interfaces
        # We try to pick candidates (Local vs Peer) for unreliable flows that balance the router.
        # Imbalance equation: Reliable_In + Sum(Unreliable_In) ≈ Reliable_Out + Sum(Unreliable_Out)
        
        # Baseline: Assume "Peer" measurements are correct for unreliable links (External Observer Principle)
        # This is our H0 hypothesis.
        h0_inputs = reliable_inputs
        h0_outputs = reliable_outputs
        
        decisions = {} # (if_id, type) -> val
        
        for if_id, metric in unreliable_ifs:
            candidates = link_analysis[if_id][metric]['candidates']
            # Default to peer if available, else local
            val = candidates.get('peer', candidates['local'])
            decisions[(if_id, metric)] = val
            if metric == 'rx': h0_inputs += val
            else: h0_outputs += val
            
        baseline_imbalance = abs(h0_inputs - h0_outputs)
        
        # 2c. Optimization / Correction
        # Check if swapping any decision from Peer -> Local improves balance significantly
        
        for if_id, metric in unreliable_ifs:
            candidates = link_analysis[if_id][metric]['candidates']
            if 'peer' not in candidates: continue # Can't swap if only local exists
            
            local_val = candidates['local']
            peer_val = candidates['peer']
            
            # If we switch this specific interface to Local, how does balance change?
            current_diff = local_val - peer_val
            
            # If metric is RX, it adds to Inputs. Change = +diff
            # If metric is TX, it adds to Outputs. Change = +diff (on the output side) -> effectively -diff to Net Flow
            
            if metric == 'rx':
                # New imbalance = |(In + diff) - Out| = |(In - Out) + diff|
                # Current Net = h0_inputs - h0_outputs
                new_imbalance = abs((h0_inputs - h0_outputs) + current_diff)
            else:
                # New imbalance = |In - (Out + diff)| = |(In - Out) - diff|
                new_imbalance = abs((h0_inputs - h0_outputs) - current_diff)
            
            # Decision Rule: If Local reduces imbalance significantly compared to Peer, pick Local.
            # "Significantly" means better balance and the imbalance is < 5% of total flow
            total_flow = max(h0_inputs, h0_outputs, 1.0)
            
            # We prefer Local if it fixes the conservation problem
            if new_imbalance < baseline_imbalance and new_imbalance < (total_flow * 0.05):
                decisions[(if_id, metric)] = local_val
                # Update running sums to reflect the swap (greedy approach)
                if metric == 'rx': h0_inputs += current_diff
                else: h0_outputs += current_diff
                baseline_imbalance = new_imbalance
        
        # 2d. Calculate Final Conservation Score for Confidence
        final_imbalance = abs(h0_inputs - h0_outputs)
        max_flow = max(h0_inputs, h0_outputs, 1.0)
        conservation_score = max(0.0, 1.0 - (final_imbalance / max_flow))
        
        # 2e. Store Decisions and compute Confidence
        for if_id in if_list:
            if if_id not in final_decisions: final_decisions[if_id] = {}
            
            # Process RX
            if (if_id, 'rx') in decisions:
                # It was unreliable
                val = decisions[(if_id, 'rx')]
                # Confidence is combination of how well it matches peer (0 by def if unreliable) + conservation
                # If we picked Peer (usually means Link Symmetry error is ignored/fixed), Conf relies on Conservation
                # If we picked Local (means Peer was wrong), Conf relies on Conservation
                final_decisions[if_id]['rx'] = (val, 0.5 + 0.5 * conservation_score)
            else:
                # It was reliable (Symmetry > 0.95)
                # Confidence is high, modulated slightly by conservation
                val = link_analysis[if_id]['rx']['candidates'].get('consensus', link_analysis[if_id]['rx']['candidates']['local'])
                final_decisions[if_id]['rx'] = (val, 0.9 + 0.1 * conservation_score)

            # Process TX
            if (if_id, 'tx') in decisions:
                val = decisions[(if_id, 'tx')]
                final_decisions[if_id]['tx'] = (val, 0.5 + 0.5 * conservation_score)
            else:
                val = link_analysis[if_id]['tx']['candidates'].get('consensus', link_analysis[if_id]['tx']['candidates']['local'])
                final_decisions[if_id]['tx'] = (val, 0.9 + 0.1 * conservation_score)

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

