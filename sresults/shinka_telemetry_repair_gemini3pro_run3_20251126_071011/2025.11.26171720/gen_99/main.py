# EVOLVE-BLOCK-START
"""
neighbor_aware_consensus
Extends Variance-Adaptive Consensus with:
1. Reliability-Weighted Self-Confidence: Dynamically weights local sensor data based on
   the router's global agreement score (Trust).
2. Neighbor-Aware Confidence Calibration: Penalizes confidence scores if the connected
   neighbor router exhibits poor flow conservation (Balance), indicating local instability.
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
        """Adaptive standard deviation."""
        return max(ABS_TOL, abs(val) * REL_TOL)

    def gaussian_score(x: float, target: float, sigma: float) -> float:
        """Unnormalized Gaussian Likelihood."""
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

        # Traffic Analysis
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

    # --- Phase 2: Iterative Consensus ---

    for iteration in range(ITERATIONS):
        # 1. Router Reliability Analysis
        router_metrics = {}
        for r_id, if_list in topology.items():
            valid_ifs = [i for i in if_list if i in state]
            if not valid_ifs: continue

            sum_in, sum_out = 0.0, 0.0
            total_agreement = 0.0

            for i in valid_ifs:
                d = state[i]
                sum_in += d['est_rx']
                sum_out += d['est_tx']

                # Interface Agreement Score
                ag = 0.5
                if d['has_peer']:
                    s_rx = get_sigma(d['est_rx'])
                    s_tx = get_sigma(d['est_tx'])
                    ag_rx = gaussian_score(d['est_rx'], d['p_tx'], s_rx)
                    ag_tx = gaussian_score(d['est_tx'], d['p_rx'], s_tx)
                    ag = (ag_rx + ag_tx) / 2.0
                else:
                    s_rx = get_sigma(d['est_rx'])
                    s_tx = get_sigma(d['est_tx'])
                    ag_rx = gaussian_score(d['est_rx'], d['s_rx'], s_rx)
                    ag_tx = gaussian_score(d['est_tx'], d['s_tx'], s_tx)
                    ag = (ag_rx + ag_tx) / 2.0 * 0.8

                total_agreement += ag

            router_trust = total_agreement / len(valid_ifs)

            imb = abs(sum_in - sum_out)
            mag = max(sum_in, sum_out, 1.0)
            balance = math.exp(- (imb / (mag * 0.05))**2 )

            router_metrics[r_id] = {
                'sin': sum_in, 'sout': sum_out,
                'trust': router_trust, 'balance': balance
            }

        # 2. Update Estimates
        for if_id, d in state.items():
            if d['status'] == 'down': continue

            r_id = telemetry[if_id].get('local_router')
            r_info = router_metrics.get(r_id)

            def solve_direction(current, s_val, p_val, is_rx):
                f_val = None
                f_weight = 0.0
                f_sigma_mult = 1.0

                # Dynamic Self-Weight based on Router Trust
                # If the router is generally trustworthy (consistent with peers),
                # we give more weight to its own measurements.
                w_self = 0.4 + 0.6 * (r_info['trust'] if r_info else 0.5)

                if r_info:
                    if is_rx:
                        others = r_info['sin'] - current
                        f_val = max(0.0, r_info['sout'] - others)
                    else:
                        others = r_info['sout'] - current
                        f_val = max(0.0, r_info['sin'] - others)

                    base_weight = 2.0
                    if not d['has_peer'] and r_info['trust'] > 0.8:
                        base_weight = 3.0

                    f_weight = base_weight * r_info['trust'] * (0.3 + 0.7 * r_info['balance'])
                    f_sigma_mult = 1.0 + 3.0 * (1.0 - r_info['balance'])

                candidates = [s_val, 0.0]
                if d['has_peer']: candidates.append(p_val)
                if f_val is not None: candidates.append(f_val)

                if d['has_peer']:
                    diff = abs(s_val - p_val)
                    if diff > ABS_TOL and (diff/max(s_val, p_val, 1.0)) < 0.2:
                        candidates.append((s_val + p_val)/2.0)

                unique_cands = sorted(list(set(candidates)))
                max_cand_mag = max(unique_cands) if unique_cands else 0.0

                best_val = current
                best_score = -1.0

                for cand in unique_cands:
                    sigma = get_sigma(cand)

                    sc_s = w_self * gaussian_score(cand, s_val, sigma)
                    sc_p = 1.0 * gaussian_score(cand, p_val, sigma) if d['has_peer'] else 0.0

                    sc_f = 0.0
                    if f_val is not None:
                         eff_sigma = sigma * f_sigma_mult
                         sc_f = f_weight * gaussian_score(cand, f_val, eff_sigma)

                    score = sc_s + sc_p + sc_f

                    if is_zero(cand):
                        deg = 1.0 / (1.0 + max_cand_mag / 10.0)
                        score *= (0.1 + 0.1 * deg)

                    if score > best_score:
                        best_score = score
                        best_val = cand

                return best_val

            new_rx = solve_direction(d['est_rx'], d['s_rx'], d['p_tx'], True)
            new_tx = solve_direction(d['est_tx'], d['s_tx'], d['p_rx'], False)

            d['est_rx'] = 0.5 * d['est_rx'] + 0.5 * new_rx
            d['est_tx'] = 0.5 * d['est_tx'] + 0.5 * new_tx

    # --- Phase 3: Final Output & Calibration ---
    results = {}

    final_metrics = {}
    for r_id, if_list in topology.items():
        valid_ifs = [i for i in if_list if i in state]
        if not valid_ifs: continue
        sin = sum(state[i]['est_rx'] for i in valid_ifs)
        sout = sum(state[i]['est_tx'] for i in valid_ifs)

        tot_ag = 0.0
        for i in valid_ifs:
            d = state[i]
            s_rx = get_sigma(d['est_rx'])
            s_tx = get_sigma(d['est_tx'])
            if d['has_peer']:
                 ag = (gaussian_score(d['est_rx'], d['p_tx'], s_rx) +
                       gaussian_score(d['est_tx'], d['p_rx'], s_tx)) / 2.0
            else:
                 ag = (gaussian_score(d['est_rx'], d['s_rx'], s_rx) +
                       gaussian_score(d['est_tx'], d['s_tx'], s_tx)) / 2.0 * 0.8
            tot_ag += ag

        trust = tot_ag / len(valid_ifs)
        bal = math.exp(-(abs(sin-sout)/max(sin,sout,1.0)*20)**2)
        # We use 'balance' key to be consistent with Phase 2
        final_metrics[r_id] = {'sin': sin, 'sout': sout, 'trust': trust, 'balance': bal}

    for if_id, d in state.items():
        res = telemetry[if_id].copy()

        if d['status'] == 'down':
            crx = 1.0 if d['s_rx'] <= ABS_TOL else 0.95
            ctx = 1.0 if d['s_tx'] <= ABS_TOL else 0.95
            res['rx_rate'] = (d['s_rx'], 0.0, crx)
            res['tx_rate'] = (d['s_tx'], 0.0, ctx)
        else:
            r_id = telemetry[if_id].get('local_router')
            r_info = final_metrics.get(r_id)

            def calc_conf(val, s_val, p_val, is_rx):
                f_val = None
                f_weight = 0.0
                f_sigma_mult = 1.0
                w_self = 0.4 + 0.6 * (r_info['trust'] if r_info else 0.5)

                if r_info:
                    if is_rx: target = r_info['sout'] - (r_info['sin'] - d['est_rx'])
                    else:     target = r_info['sin'] - (r_info['sout'] - d['est_tx'])
                    f_val = max(0.0, target)

                    base_weight = 2.0
                    if not d['has_peer'] and r_info['trust'] > 0.8: base_weight = 3.0
                    f_weight = base_weight * r_info['trust'] * (0.3 + 0.7 * r_info['balance'])
                    f_sigma_mult = 1.0 + 3.0 * (1.0 - r_info['balance'])

                hyps = {val, s_val, 0.0}
                if d['has_peer']: hyps.add(p_val)
                if f_val is not None: hyps.add(f_val)
                if d['has_peer']: hyps.add((s_val + p_val)/2.0)

                sorted_hyps = sorted(list(hyps))
                max_cand_mag = max(sorted_hyps) if sorted_hyps else 0.0

                sigma_win = get_sigma(val)

                def get_score(h, sigma):
                    sc_s = w_self * gaussian_score(h, s_val, sigma)
                    sc_p = 1.0 * gaussian_score(h, p_val, sigma) if d['has_peer'] else 0.0
                    sc_f = 0.0
                    if f_val is not None:
                        sc_f = f_weight * gaussian_score(h, f_val, sigma * f_sigma_mult)

                    tot = sc_s + sc_p + sc_f

                    if is_zero(h):
                        deg = 1.0 / (1.0 + max_cand_mag / 10.0)
                        tot *= (0.1 + 0.1 * deg)

                    return tot

                winner_score = get_score(val, sigma_win)

                runner_up_score = 0.0
                for h in sorted_hyps:
                    if abs(h - val) <= sigma_win: continue
                    sigma_h = get_sigma(h)
                    s = get_score(h, sigma_h)
                    if s > runner_up_score:
                        runner_up_score = s

                total_mass = winner_score + runner_up_score + 1e-9
                prob = winner_score / total_mass

                max_possible = w_self + (1.0 if d['has_peer'] else 0.0) + (f_weight if f_val is not None else 0.0)
                fit_ratio = winner_score / max(1.0, max_possible)

                if fit_ratio < 0.5:
                    prob *= fit_ratio * 2.0

                # Neighbor Consensus Check
                # If the remote router is imbalanced, reduce confidence
                remote_id = telemetry[if_id].get('remote_router')
                if remote_id and remote_id in final_metrics:
                    rem_bal = final_metrics[remote_id]['balance']
                    # Penalty factor: 0.85 + 0.15 * neighbor_balance
                    # If neighbor is broken (balance=0), penalty is 0.85
                    prob *= (0.85 + 0.15 * rem_bal)

                # Sanity Check
                peer_agrees = d['has_peer'] and abs(val - p_val) <= sigma_win
                flow_agrees = f_val is not None and abs(val - f_val) <= (sigma_win * f_sigma_mult)

                if not peer_agrees and not flow_agrees:
                    prob = min(prob, 0.75)

                return max(0.5, min(1.0, prob))

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