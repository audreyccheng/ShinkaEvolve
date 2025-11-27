# EVOLVE-BLOCK-START
"""
weighted_anchor_consensus
Combines Iterative Bayesian Consensus with a Traffic-Weighted Anchor mechanism.
The router reliability score (Anchor) is weighted by interface traffic magnitude, ensuring
that high-volume links (which dictate flow conservation) have a proportional impact on
whether we trust the router's flow target. Retains Probability Dominance calibration.
"""
import math
from typing import Dict, Any, Tuple, List

def repair_network_telemetry(telemetry: Dict[str, Dict[str, Any]],
                             topology: Dict[str, List[str]]) -> Dict[str, Dict[str, Tuple]]:

    # --- Configuration ---
    REL_TOL = 0.02
    ABS_TOL = 0.5
    ITERATIONS = 5

    # --- Helper Functions ---
    def get_sigma(val: float) -> float:
        """Adaptive standard deviation for Gaussian kernels."""
        return max(ABS_TOL, abs(val) * REL_TOL)

    def gaussian_score(x: float, target: float, sigma: float) -> float:
        """Unnormalized Gaussian Likelihood (0.0 to 1.0)."""
        if target is None: return 0.0
        diff = abs(x - target)
        return math.exp(-0.5 * (diff / sigma) ** 2)

    def is_zero(val: float) -> bool:
        return val < ABS_TOL

    # --- Phase 1: Initialization & Status Repair ---
    state = {}

    for if_id, data in telemetry.items():
        s_rx = float(data.get('rx_rate', 0.0))
        s_tx = float(data.get('tx_rate', 0.0))
        s_stat = data.get('interface_status', 'unknown')

        peer_id = data.get('connected_to')
        has_peer = False
        p_rx, p_tx, p_stat = 0.0, 0.0, 'unknown'

        if peer_id and peer_id in telemetry:
            has_peer = True
            p_data = telemetry[peer_id]
            p_rx = float(p_data.get('rx_rate', 0.0))
            p_tx = float(p_data.get('tx_rate', 0.0))
            p_stat = p_data.get('interface_status', 'unknown')

        # Traffic Analysis: Significant signal on Self or Peer
        proven_active = (s_rx > ABS_TOL or s_tx > ABS_TOL or
                         p_rx > ABS_TOL or p_tx > ABS_TOL)

        final_stat = s_stat
        stat_conf = 1.0

        # Status Repair Logic
        if s_stat == 'down' and proven_active:
            final_stat = 'up'
            stat_conf = 0.95
        elif s_stat == 'up' and not proven_active:
            if p_stat == 'down':
                final_stat = 'down'
                stat_conf = 0.90

        # Initial Value Estimation
        if final_stat == 'down':
            est_rx, est_tx = 0.0, 0.0
        else:
            est_rx = p_tx if has_peer else s_rx
            est_tx = p_rx if has_peer else s_tx

        state[if_id] = {
            's_rx': s_rx, 's_tx': s_tx,
            'p_rx': p_rx, 'p_tx': p_tx,
            'est_rx': est_rx, 'est_tx': est_tx,
            'status': final_stat,
            'stat_conf': stat_conf,
            'orig_stat': s_stat,
            'has_peer': has_peer
        }

    # --- Phase 2: Iterative Bayesian Consensus ---

    for iteration in range(ITERATIONS):
        # 1. Weighted Router Anchor Analysis
        router_metrics = {}
        for r_id, if_list in topology.items():
            valid_ifs = [i for i in if_list if i in state]
            if not valid_ifs: continue

            sum_in, sum_out = 0.0, 0.0
            weighted_agreement_sum = 0.0
            total_weight = 0.0

            for i in valid_ifs:
                d = state[i]
                sum_in += d['est_rx']
                sum_out += d['est_tx']

                # Agreement Score: How well does this link match its peer?
                agreement = 0.5 # Neutral default
                if d['has_peer']:
                    ag_rx = gaussian_score(d['est_rx'], d['p_tx'], get_sigma(d['est_rx']))
                    ag_tx = gaussian_score(d['est_tx'], d['p_rx'], get_sigma(d['est_tx']))
                    agreement = (ag_rx + ag_tx) / 2.0

                # Weight by Traffic Magnitude (Log Scale)
                # High traffic links (which drive the flow) contribute more to the
                # "trustworthiness" of the router context.
                mag = max(d['est_rx'], d['est_tx'], 1.0)
                weight = math.log(10.0 + mag) # Log weighting prevents dominance but respects size

                weighted_agreement_sum += agreement * weight
                total_weight += weight

            anchor = weighted_agreement_sum / max(total_weight, 1.0)

            # Balance Score
            imb = abs(sum_in - sum_out)
            mag = max(sum_in, sum_out, 1.0)
            balance = math.exp(- (imb / (mag * 0.05))**2 )

            router_metrics[r_id] = {
                'sin': sum_in, 'sout': sum_out,
                'anchor': anchor, 'balance': balance
            }

        # 2. Update Estimates
        for if_id, d in state.items():
            if d['status'] == 'down': continue

            r_id = telemetry[if_id].get('local_router')
            r_info = router_metrics.get(r_id)

            def solve_direction(current, s_val, p_val, is_rx):
                f_val = None
                f_weight = 0.0
                f_sigma_scale = 1.0

                if r_info:
                    if is_rx:
                        others = r_info['sin'] - current
                        f_val = max(0.0, r_info['sout'] - others)
                    else:
                        others = r_info['sout'] - current
                        f_val = max(0.0, r_info['sin'] - others)

                    f_weight = 2.5 * r_info['anchor'] * (0.2 + 0.8 * r_info['balance'])
                    # Adaptive Variance: allow wider flow match if router is imbalanced
                    f_sigma_scale = 1.0 + 3.0 * (1.0 - r_info['balance'])

                # Hypothesis Generation
                candidates = [s_val, 0.0]
                if d['has_peer']: candidates.append(p_val)
                if f_val is not None: candidates.append(f_val)

                if d['has_peer']:
                    diff = abs(s_val - p_val)
                    if diff > ABS_TOL and (diff / max(s_val, p_val, 1.0)) < 0.15:
                        candidates.append((s_val + p_val) / 2.0)

                if f_val is not None and d['has_peer']:
                    diff_p_f = abs(p_val - f_val)
                    if diff_p_f > ABS_TOL and (diff_p_f / max(p_val, f_val, 1.0)) < 0.15:
                        candidates.append((p_val + f_val) / 2.0)

                best_val = current
                best_score = -1.0

                unique_candidates = sorted(list(set(candidates)))
                max_mag = max(unique_candidates) if unique_candidates else 0.0

                for cand in unique_candidates:
                    sigma = get_sigma(cand)

                    l_self = 0.8 * gaussian_score(cand, s_val, sigma)
                    l_peer = 1.0 * gaussian_score(cand, p_val, sigma) if d['has_peer'] else 0.0
                    l_flow = 0.0
                    if f_val is not None:
                        l_flow = f_weight * gaussian_score(cand, f_val, sigma * f_sigma_scale)

                    score = l_self + l_peer + l_flow

                    # Magnitude-Dependent Zero Penalty
                    if is_zero(cand):
                        penalty = 0.1 / (1.0 + max_mag / 10.0)
                        score *= penalty

                    if score > best_score:
                        best_score = score
                        best_val = cand

                return best_val

            new_rx = solve_direction(d['est_rx'], d['s_rx'], d['p_tx'], True)
            new_tx = solve_direction(d['est_tx'], d['s_tx'], d['p_rx'], False)

            # Momentum Update
            d['est_rx'] = 0.5 * d['est_rx'] + 0.5 * new_rx
            d['est_tx'] = 0.5 * d['est_tx'] + 0.5 * new_tx

    # --- Phase 3: Final Output & Probability Calibration ---
    results = {}

    # Final Router Context (Re-calculate with weighted anchor for consistency)
    final_metrics = {}
    for r_id, if_list in topology.items():
        valid_ifs = [i for i in if_list if i in state]
        if not valid_ifs: continue
        sin = sum(state[i]['est_rx'] for i in valid_ifs)
        sout = sum(state[i]['est_tx'] for i in valid_ifs)

        w_ag_sum = 0.0
        tot_w = 0.0
        for i in valid_ifs:
            entry = state[i]
            ag = 0.5
            if entry['has_peer']:
                 ag = (gaussian_score(entry['est_rx'], entry['p_tx'], get_sigma(entry['est_rx'])) +
                       gaussian_score(entry['est_tx'], entry['p_rx'], get_sigma(entry['est_tx']))) / 2.0
            mag = max(entry['est_rx'], entry['est_tx'], 1.0)
            w = math.log(10.0 + mag)
            w_ag_sum += ag * w
            tot_w += w

        anchor = w_ag_sum / max(tot_w, 1.0)
        bal = math.exp(-(abs(sin-sout)/max(sin,sout,1.0)*20)**2)
        final_metrics[r_id] = {'sin': sin, 'sout': sout, 'anchor': anchor, 'bal': bal}

    for if_id, d in state.items():
        res = telemetry[if_id].copy()

        if d['status'] == 'down':
            # High confidence repair for Down
            crx = 1.0 if d['s_rx'] <= ABS_TOL else 0.95
            ctx = 1.0 if d['s_tx'] <= ABS_TOL else 0.95
            res['rx_rate'] = (d['s_rx'], 0.0, crx)
            res['tx_rate'] = (d['s_tx'], 0.0, ctx)
        else:
            r_id = telemetry[if_id].get('local_router')
            r_info = final_metrics.get(r_id)

            def calc_conf(val, s_val, p_val, is_rx):
                f_val = None
                f_w = 0.0
                f_sigma_scale = 1.0

                if r_info:
                    if is_rx: target = r_info['sout'] - (r_info['sin'] - d['est_rx'])
                    else:     target = r_info['sin'] - (r_info['sout'] - d['est_tx'])
                    f_val = max(0.0, target)
                    f_w = 2.5 * r_info['anchor'] * (0.2 + 0.8 * r_info['bal'])
                    f_sigma_scale = 1.0 + 3.0 * (1.0 - r_info['bal'])

                hyps = {val, s_val, 0.0}
                if d['has_peer']: hyps.add(p_val)
                if f_val is not None: hyps.add(f_val)
                if d['has_peer']: hyps.add((s_val + p_val)/2.0)

                max_mag = max(hyps)
                sigma_win = get_sigma(val)

                total_mass = 0.0
                winner_mass = 0.0

                for h in hyps:
                    sigma = get_sigma(h)
                    sc_s = 0.8 * gaussian_score(h, s_val, sigma)
                    sc_p = 1.0 * gaussian_score(h, p_val, sigma) if d['has_peer'] else 0.0
                    sc_f = 0.0
                    if f_val is not None:
                         sc_f = f_w * gaussian_score(h, f_val, sigma * f_sigma_scale)

                    score = sc_s + sc_p + sc_f

                    if is_zero(h):
                        penalty = 0.1 / (1.0 + max_mag / 10.0)
                        score *= penalty

                    total_mass += score
                    if abs(h - val) <= sigma_win:
                        winner_mass += score

                if total_mass < 1e-9: return 0.5

                prob = winner_mass / total_mass

                # Absolute Fit Check
                w_sigma = get_sigma(val)
                ws_s = 0.8 * gaussian_score(val, s_val, w_sigma)
                ws_p = 1.0 * gaussian_score(val, p_val, w_sigma) if d['has_peer'] else 0.0
                ws_f = 0.0
                if f_val is not None:
                    ws_f = f_w * gaussian_score(val, f_val, w_sigma * f_sigma_scale)

                winner_raw_score = ws_s + ws_p + ws_f
                max_possible = 0.8 + (1.0 if d['has_peer'] else 0.0) + (f_w if f_val is not None else 0.0)

                fit_ratio = winner_raw_score / max(1.0, max_possible)

                final_conf = prob
                if fit_ratio < 0.5:
                    final_conf *= fit_ratio * 2.0

                # Support Verification
                peer_agrees = d['has_peer'] and abs(val - p_val) <= sigma_win
                flow_agrees = f_val is not None and abs(val - f_val) <= (sigma_win * f_sigma_scale)

                if not peer_agrees and not flow_agrees:
                    final_conf = min(final_conf, 0.75)

                return max(0.5, min(1.0, final_conf))

            conf_rx = calc_conf(d['est_rx'], d['s_rx'], d['p_tx'], True)
            conf_tx = calc_conf(d['est_tx'], d['s_tx'], d['p_rx'], False)

            res['rx_rate'] = (d['s_rx'], d['est_rx'], conf_rx)
            res['tx_rate'] = (d['s_tx'], d['est_tx'], conf_tx)

        res['interface_status'] = (d['orig_stat'], d['status'], d['stat_conf'])
        results[if_id] = res

    return results
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