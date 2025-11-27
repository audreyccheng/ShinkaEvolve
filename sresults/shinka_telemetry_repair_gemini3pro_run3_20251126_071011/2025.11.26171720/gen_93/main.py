# EVOLVE-BLOCK-START
from typing import Dict, Any, Tuple, List
import math

def repair_network_telemetry(telemetry: Dict[str, Dict[str, Any]],
                             topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:
    """
    Repairs network telemetry using Reliability-Weighted Bayesian Consensus with Continuous Priors.

    Improvements:
    - Continuous Phantom Prior Decay: Smoothly handles 0-flow hypothesis probability.
    - Reliability-Weighted priors: Uses router anchor strength to weight conservation vs measurement.
    - Trust Transfer: Tightens constraints on external links of reliable routers.
    - Neighbor Consensus: Incorporates peer router fit into confidence calibration.
    """

    # --- Configuration ---
    SYMMETRY_TOLERANCE = 0.02
    CONSERVATION_TOLERANCE_PCT = 0.02
    MIN_SIGNIFICANT_FLOW = 0.5
    ITERATIONS = 10
    LOCKING_ITERATION = 4
    LOCKING_THRESHOLD = 0.85
    MOMENTUM = 0.6

    # --- Helper Structures ---
    if_to_router = {}
    for r_id, if_list in topology.items():
        for i_id in if_list:
            if_to_router[i_id] = r_id

    # Estimates state: {if_id: {'rx': val, 'tx': val}}
    estimates = {}

    # Classification
    links = {}
    processed_ifs = set()

    # Initial data load
    for if_id, data in telemetry.items():
        estimates[if_id] = {
            'rx': data.get('rx_rate', 0.0),
            'tx': data.get('tx_rate', 0.0)
        }

        if if_id in processed_ifs: continue

        peer = data.get('connected_to')
        if peer and peer in telemetry:
            link_key = tuple(sorted([if_id, peer]))
            links[link_key] = {'type': 'internal', 'if1': if_id, 'if2': peer}
            processed_ifs.add(if_id)
            processed_ifs.add(peer)
        else:
            links[(if_id,)] = {'type': 'external', 'if1': if_id, 'if2': None}
            processed_ifs.add(if_id)

    # --- Step 1: Symmetry & Anchoring ---
    suspect_flows = []
    anchors = set()
    locked_flows = set()

    for key, info in links.items():
        if1 = info['if1']

        if info['type'] == 'internal':
            if2 = info['if2']

            # Forward: 1(TX) -> 2(RX)
            v1, v2 = estimates[if1]['tx'], estimates[if2]['rx']
            diff = abs(v1 - v2)
            denom = max(v1, v2, 1.0)

            if diff / denom < SYMMETRY_TOLERANCE:
                avg = (v1 + v2) / 2.0
                estimates[if1]['tx'] = avg
                estimates[if2]['rx'] = avg
                anchors.add((if1, 'tx'))
                anchors.add((if2, 'rx'))
                locked_flows.add((if1, 'tx'))
                locked_flows.add((if2, 'rx'))
            else:
                suspect_flows.append({
                    'type': 'internal', 'src': if1, 'dst': if2,
                    'cands': [v1, v2], 'dir': '1_to_2'
                })

            # Backward: 2(TX) -> 1(RX)
            v1, v2 = estimates[if2]['tx'], estimates[if1]['rx']
            diff = abs(v1 - v2)
            denom = max(v1, v2, 1.0)

            if diff / denom < SYMMETRY_TOLERANCE:
                avg = (v1 + v2) / 2.0
                estimates[if2]['tx'] = avg
                estimates[if1]['rx'] = avg
                anchors.add((if2, 'tx'))
                anchors.add((if1, 'rx'))
                locked_flows.add((if2, 'tx'))
                locked_flows.add((if1, 'rx'))
            else:
                suspect_flows.append({
                    'type': 'internal', 'src': if2, 'dst': if1,
                    'cands': [v1, v2], 'dir': '2_to_1'
                })

        else:
            suspect_flows.append({
                'type': 'external', 'src': if1, 'dst': None,
                'cands': [estimates[if1]['tx']],
                'metric': 'tx'
            })
            suspect_flows.append({
                'type': 'external', 'src': None, 'dst': if1,
                'cands': [estimates[if1]['rx']],
                'metric': 'rx'
            })

    # --- Step 2: Iterative Solver ---

    def get_router_state(rid):
        if rid not in topology: return 0.0, 1.0
        tin, tout = 0.0, 0.0
        max_f = 0.0
        for iid in topology[rid]:
            if iid in estimates:
                r, t = estimates[iid]['rx'], estimates[iid]['tx']
                tin += r
                tout += t
                max_f = max(max_f, r, t)
        return (tin - tout), max(max_f, 1.0)

    def calc_sigma(flow_val, reliability=0.5):
        base = max(1.5 * math.sqrt(flow_val), flow_val * CONSERVATION_TOLERANCE_PCT, 1.0)
        # Reliable routers (1.0) -> 1.0x sigma, Unreliable (0.0) -> 2.0x sigma
        return base * (2.0 - reliability)

    solver_probs = {}

    for iter_idx in range(ITERATIONS):
        # Calculate Router Reliability
        router_reliability = {}
        for rid in topology:
            trusted_flow = 0.0
            total_flow = 0.0
            for iid in topology[rid]:
                if iid in estimates:
                    r, t = estimates[iid]['rx'], estimates[iid]['tx']
                    total_flow += (r + t)
                    if (iid, 'rx') in locked_flows: trusted_flow += r
                    if (iid, 'tx') in locked_flows: trusted_flow += t
            router_reliability[rid] = trusted_flow / max(total_flow, 1.0)

        updates = []
        new_locks = []
        can_lock = (iter_idx >= LOCKING_ITERATION)

        for f_idx, flow in enumerate(suspect_flows):
            if flow.get('locked', False): continue

            # Internal Logic
            if flow['type'] == 'internal':
                src, dst = flow['src'], flow['dst']
                r_src = if_to_router.get(src)
                r_dst = if_to_router.get(dst)

                rel_src = router_reliability.get(r_src, 0.5)
                rel_dst = router_reliability.get(r_dst, 0.5)

                cands = [c for c in flow['cands'] if c >= 0]
                if len(cands) == 2:
                    v1, v2 = cands
                    if abs(v1-v2) < max(v1,v2)*0.2 + 5.0:
                        cands.append((v1+v2)/2.0)

                hyps = sorted(list(set(cands + [0.0])))
                curr_tx = estimates[src]['tx']
                curr_rx = estimates[dst]['rx']

                scores = []
                for h in hyps:
                    estimates[src]['tx'] = h
                    estimates[dst]['rx'] = h

                    imb_s, flow_s = get_router_state(r_src)
                    score_s = math.exp(-abs(imb_s)/calc_sigma(flow_s, rel_src)) if r_src else 1.0

                    imb_d, flow_d = get_router_state(r_dst)
                    score_d = math.exp(-abs(imb_d)/calc_sigma(flow_d, rel_dst)) if r_dst else 1.0

                    prior = 1.0
                    if h == 0.0:
                        s_src = flow.get('status_src')
                        s_dst = flow.get('status_dst')
                        if s_src == 'down' or s_dst == 'down': prior = 0.98
                        else:
                            max_c = max(flow['cands'])
                            # Continuous decay centered at 5.0
                            prior = 1.0 / (1.0 + math.exp(0.5 * (max_c - 5.0)))
                    else:
                        min_dist = min([abs(h - c) for c in flow['cands']])
                        prior = math.exp(-min_dist / max(h*0.05, 1.0))

                    scores.append(score_s * score_d * prior)

                estimates[src]['tx'] = curr_tx
                estimates[dst]['rx'] = curr_rx

                total = sum(scores) + 1e-20
                probs = [s/total for s in scores]
                best_idx = scores.index(max(scores))
                win_val = hyps[best_idx]
                win_prob = sum(p for i, p in enumerate(probs) if abs(hyps[i] - win_val) < max(win_val*0.05, 1.0))

                updates.append((src, 'tx', win_val, win_prob))
                updates.append((dst, 'rx', win_val, win_prob))

                if can_lock and win_prob > LOCKING_THRESHOLD:
                    new_locks.append(f_idx)
                    locked_flows.add((src, 'tx'))
                    locked_flows.add((dst, 'rx'))

            # External Logic
            elif flow['type'] == 'external':
                if flow['src']:
                    if_id = flow['src']; metric = 'tx'; meas = flow['cands'][0]
                    r_id = if_to_router.get(if_id)
                    curr = estimates[if_id]['tx']
                    stat = flow.get('status_src')
                    imb, r_flow = get_router_state(r_id)
                    implied = max(0.0, curr + imb)
                else:
                    if_id = flow['dst']; metric = 'rx'; meas = flow['cands'][0]
                    r_id = if_to_router.get(if_id)
                    curr = estimates[if_id]['rx']
                    stat = flow.get('status_dst')
                    imb, r_flow = get_router_state(r_id)
                    implied = max(0.0, curr - imb)

                hyps = sorted(list(set([meas, implied, 0.0])))
                scores = []
                rel = router_reliability.get(r_id, 0.5)

                for h in hyps:
                    estimates[if_id][metric] = h
                    imb, rf = get_router_state(r_id)
                    sigma = calc_sigma(rf, rel)
                    lik = math.exp(-abs(imb)/sigma) if r_id else 0.5

                    prior = 1.0
                    if h == 0.0:
                        if stat == 'down': prior = 0.98
                        else: prior = 1.0 / (1.0 + math.exp(0.5 * (meas - 5.0)))
                    elif abs(h - implied) < 1e-6:
                        prior = 0.2 + 0.7 * rel
                    elif abs(h - meas) < 1e-6:
                        prior = 0.9 - 0.7 * rel

                    scores.append(lik * prior)

                estimates[if_id][metric] = curr

                total = sum(scores) + 1e-20
                probs = [s/total for s in scores]
                best_idx = scores.index(max(scores))
                win_val = hyps[best_idx]
                win_prob = sum(p for i, p in enumerate(probs) if abs(hyps[i] - win_val) < max(win_val*0.05, 1.0))

                updates.append((if_id, metric, win_val, win_prob))

        for if_id, metric, val, prob in updates:
            old = estimates[if_id][metric]
            estimates[if_id][metric] = (old * (1 - MOMENTUM)) + (val * MOMENTUM)
            solver_probs[(if_id, metric)] = prob

        for idx in new_locks:
            suspect_flows[idx]['locked'] = True

    # --- Step 3: Confidence & Status ---
    result = {}

    # Recalculate reliability for final fit
    final_reliability = {}
    for rid in topology:
        trusted_flow = 0.0
        total_flow = 0.0
        for iid in topology[rid]:
            if iid in estimates:
                r, t = estimates[iid]['rx'], estimates[iid]['tx']
                total_flow += (r + t)
                if (iid, 'rx') in locked_flows: trusted_flow += r
                if (iid, 'tx') in locked_flows: trusted_flow += t
        final_reliability[rid] = trusted_flow / max(total_flow, 1.0)

    router_fits = {}
    for rid in topology:
        imb, flow = get_router_state(rid)
        rel = final_reliability.get(rid, 0.5)
        router_fits[rid] = math.exp(-abs(imb) / calc_sigma(flow, rel))

    for if_id, data in telemetry.items():
        orig_rx = data.get('rx_rate', 0.0)
        orig_tx = data.get('tx_rate', 0.0)
        orig_status = data.get('interface_status', 'unknown')

        rep_rx = estimates[if_id]['rx']
        rep_tx = estimates[if_id]['tx']

        rid = if_to_router.get(if_id)
        r_fit = router_fits.get(rid, 0.8)

        # Neighbor Consensus
        peer_router = data.get('remote_router')
        effective_fit = r_fit
        if peer_router and peer_router in router_fits:
            effective_fit = math.sqrt(r_fit * router_fits[peer_router])

        def calc_conf(metric, rep_val, orig_val):
            if (if_id, metric) in anchors: return 0.98

            sol_prob = solver_probs.get((if_id, metric), 0.5)
            h_mean = 2 * (sol_prob * effective_fit) / (sol_prob + effective_fit + 1e-9)

            if abs(rep_val - orig_val) < max(orig_val * 0.05, 1.0):
                return max(h_mean, 0.85 * effective_fit + 0.1)
            else:
                return h_mean

        conf_rx = calc_conf('rx', rep_rx, orig_rx)
        conf_tx = calc_conf('tx', rep_tx, orig_tx)

        conf_rx = max(0.01, min(0.99, conf_rx))
        conf_tx = max(0.01, min(0.99, conf_tx))

        # Status Logic
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
                conf_status = 0.95
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
