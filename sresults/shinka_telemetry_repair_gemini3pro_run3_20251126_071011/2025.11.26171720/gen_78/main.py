# EVOLVE-BLOCK-START
"""
clustered_consensus_flow
Implements a dual-phase Bayesian repair strategy with "Cluster Mass" confidence calibration.
1. Uses traffic-magnitude weighted "Anchor" scores to determine router trustworthiness.
2. Incorporates strict Status-Aware Priors: penalizes the zero-hypothesis heavily if status is UP, unless "Flow" explicitly validates zero traffic (Phantom Traffic handling).
3. Calibration groups hypotheses into clusters (Winner vs Rest) to accurately calculate probability mass, boosting confidence when Peer or Flow signals align with the repair.
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
        """Unnormalized Gaussian Likelihood."""
        if target is None: return 0.0
        diff = abs(x - target)
        return math.exp(-0.5 * (diff / sigma) ** 2)

    def is_zero(val: float) -> bool:
        return val < ABS_TOL

    # --- Phase 1: Initialization & Status Logic ---
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

        # Traffic evidence
        traffic_detected = (s_rx > ABS_TOL or s_tx > ABS_TOL or
                            p_rx > ABS_TOL or p_tx > ABS_TOL)

        final_stat = s_stat
        stat_conf = 1.0

        # Status Inference
        # If config says DOWN but traffic flows, it's UP.
        if s_stat == 'down' and traffic_detected:
            final_stat = 'up'
            stat_conf = 0.95
        # If config says UP but it's dead silent and peer is DOWN, it's likely DOWN.
        elif s_stat == 'up' and not traffic_detected:
            if p_stat == 'down':
                final_stat = 'down'
                stat_conf = 0.90

        # Initial Estimates
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

    # --- Phase 2: Consensus Loop ---

    for iteration in range(ITERATIONS):
        # 1. Router Analytics
        router_metrics = {}
        for r_id, if_list in topology.items():
            valid_ifs = [i for i in if_list if i in state]
            if not valid_ifs: continue

            sum_in, sum_out = 0.0, 0.0
            sum_trusted_mass, total_mass = 0.0, 0.0

            for i in valid_ifs:
                d = state[i]
                sum_in += d['est_rx']
                sum_out += d['est_tx']

                mass = d['est_rx'] + d['est_tx']
                total_mass += mass

                # Trust Assessment: High Agreement with Peer (or Self if isolated)
                agree = 0.0
                if d['has_peer']:
                    ag_rx = gaussian_score(d['est_rx'], d['p_tx'], get_sigma(d['est_rx']))
                    ag_tx = gaussian_score(d['est_tx'], d['p_rx'], get_sigma(d['est_tx']))
                    agree = (ag_rx + ag_tx) / 2.0
                else:
                    # Isolated: check self-consistency (weaker)
                    ag_rx = gaussian_score(d['est_rx'], d['s_rx'], get_sigma(d['est_rx']))
                    ag_tx = gaussian_score(d['est_tx'], d['s_tx'], get_sigma(d['est_tx']))
                    agree = (ag_rx + ag_tx) / 2.0

                # Threshold for "Trusted" Link
                if agree > 0.8:
                    sum_trusted_mass += mass

            # Trust Ratio: Fraction of traffic that is verified
            trust_ratio = sum_trusted_mass / max(total_mass, 1.0)

            # Balance Score
            imb = abs(sum_in - sum_out)
            mag = max(sum_in, sum_out, 1.0)
            balance = math.exp(- (imb / (mag * 0.05))**2 )

            router_metrics[r_id] = {
                'sin': sum_in, 'sout': sum_out,
                'trust': trust_ratio, 'balance': balance
            }

        # 2. Estimate Updates
        for if_id, d in state.items():
            if d['status'] == 'down': continue

            r_id = telemetry[if_id].get('local_router')
            r_info = router_metrics.get(r_id)

            def solve(current, s_val, p_val, is_rx):
                # Flow Target
                f_val = None
                f_w = 0.0
                f_sigma_mult = 1.0

                if r_info:
                    if is_rx:
                        others = r_info['sin'] - current
                        f_val = max(0.0, r_info['sout'] - others)
                    else:
                        others = r_info['sout'] - current
                        f_val = max(0.0, r_info['sin'] - others)

                    # Trust Boost: If router is verified (Trust > 0.8) and Balanced,
                    # we boost flow weight to override local errors.
                    base_w = 2.5 * r_info['trust'] * (0.2 + 0.8 * r_info['balance'])
                    if r_info['trust'] > 0.8 and r_info['balance'] > 0.8:
                        f_weight = base_w * 2.0
                    else:
                        f_weight = base_w

                    f_w = f_weight

                    # Adaptive Sigma: Widen acceptance window if router is imbalanced
                    f_sigma_mult = 1.0 + 3.0 * (1.0 - r_info['balance'])

                # Candidates
                candidates = [s_val, 0.0]
                if d['has_peer']: candidates.append(p_val)
                if f_val is not None: candidates.append(f_val)

                # Noise Cluster Mean
                if d['has_peer']:
                    diff = abs(s_val - p_val)
                    if diff > ABS_TOL and (diff / max(s_val, p_val, 1.0)) < 0.2:
                        candidates.append((s_val + p_val) / 2.0)

                # Evaluation
                best_val = current
                best_score = -1.0

                unique_cands = sorted(list(set(candidates)))

                for cand in unique_cands:
                    sigma = get_sigma(cand)

                    # Sources
                    sc_s = 0.8 * gaussian_score(cand, s_val, sigma)
                    sc_p = 1.0 * gaussian_score(cand, p_val, sigma) if d['has_peer'] else 0.0

                    # Flow score with adaptive sigma
                    sc_f = 0.0
                    if f_val is not None:
                         eff_sigma = sigma * f_sigma_mult
                         sc_f = f_w * gaussian_score(cand, f_val, eff_sigma)

                    score = sc_s + sc_p + sc_f

                    # Status-Aware Priors
                    if is_zero(cand):
                        flow_says_zero = (f_val is not None) and (f_val < ABS_TOL)
                        if not flow_says_zero:
                            score *= 0.05
                        else:
                            score *= 0.5

                    if score > best_score:
                        best_score = score
                        best_val = cand

                return best_val

            new_rx = solve(d['est_rx'], d['s_rx'], d['p_tx'], True)
            new_tx = solve(d['est_tx'], d['s_tx'], d['p_rx'], False)

            # Momentum
            d['est_rx'] = 0.5 * d['est_rx'] + 0.5 * new_rx
            d['est_tx'] = 0.5 * d['est_tx'] + 0.5 * new_tx

    # --- Phase 3: Final Computation & Calibration ---
    results = {}

    # Final context computation
    final_metrics = {}
    for r_id, if_list in topology.items():
        valid_ifs = [i for i in if_list if i in state]
        if not valid_ifs: continue
        sin = sum(state[i]['est_rx'] for i in valid_ifs)
        sout = sum(state[i]['est_tx'] for i in valid_ifs)

        sum_trusted_mass, total_mass = 0.0, 0.0
        for i in valid_ifs:
            d = state[i]
            mass = d['est_rx'] + d['est_tx']
            total_mass += mass
            agree = 0.0
            if d['has_peer']:
                 agree = (gaussian_score(d['est_rx'], d['p_tx'], get_sigma(d['est_rx'])) +
                       gaussian_score(d['est_tx'], d['p_rx'], get_sigma(d['est_tx']))) / 2.0
            else:
                 agree = (gaussian_score(d['est_rx'], d['s_rx'], get_sigma(d['est_rx'])) +
                       gaussian_score(d['est_tx'], d['s_tx'], get_sigma(d['est_tx']))) / 2.0
            if agree > 0.8:
                sum_trusted_mass += mass

        trust = sum_trusted_mass / max(total_mass, 1.0)
        bal = math.exp(-(abs(sin-sout)/max(sin,sout,1.0)*20)**2)
        final_metrics[r_id] = {'sin': sin, 'sout': sout, 'trust': trust, 'bal': bal}

    for if_id, d in state.items():
        res = telemetry[if_id].copy()

        if d['status'] == 'down':
            rx_clean = d['s_rx'] <= ABS_TOL
            tx_clean = d['s_tx'] <= ABS_TOL
            res['rx_rate'] = (d['s_rx'], 0.0, 1.0 if rx_clean else 0.95)
            res['tx_rate'] = (d['s_tx'], 0.0, 1.0 if tx_clean else 0.95)
        else:
            r_id = telemetry[if_id].get('local_router')
            r_info = final_metrics.get(r_id)

            def calibrate(val, s_val, p_val, is_rx):
                f_val = None
                f_w = 0.0
                f_sigma_mult = 1.0

                if r_info:
                    if is_rx: target = r_info['sout'] - (r_info['sin'] - d['est_rx'])
                    else:     target = r_info['sin'] - (r_info['sout'] - d['est_tx'])
                    f_val = max(0.0, target)

                    base_w = 2.5 * r_info['trust'] * (0.2 + 0.8 * r_info['bal'])
                    if r_info['trust'] > 0.8 and r_info['bal'] > 0.8:
                        f_w = base_w * 2.0
                    else:
                        f_w = base_w

                    f_sigma_mult = 1.0 + 3.0 * (1.0 - r_info['bal'])

                hyps = {val, s_val, 0.0}
                if d['has_peer']: hyps.add(p_val)
                if f_val is not None: hyps.add(f_val)
                if d['has_peer']: hyps.add((s_val + p_val)/2.0)

                cluster_mass = 0.0
                total_mass = 0.0
                sigma_win = get_sigma(val)

                def get_score(h, sigma):
                    sc_s = 0.8 * gaussian_score(h, s_val, sigma)
                    sc_p = 1.0 * gaussian_score(h, p_val, sigma) if d['has_peer'] else 0.0
                    sc_f = 0.0
                    if f_val is not None:
                        sc_f = f_w * gaussian_score(h, f_val, sigma * f_sigma_mult)

                    tot = sc_s + sc_p + sc_f

                    if is_zero(h):
                        flow_zero = (f_val is not None) and (f_val < ABS_TOL)
                        if not flow_zero: tot *= 0.05
                        else: tot *= 0.5
                    return tot

                for h in hyps:
                    sigma = get_sigma(h)
                    s = get_score(h, sigma)
                    total_mass += s
                    if abs(h - val) <= sigma_win:
                        cluster_mass += s

                if total_mass < 1e-9: return 0.5

                raw_prob = cluster_mass / total_mass

                peer_support = d['has_peer'] and (abs(val - p_val) <= sigma_win)
                flow_support = (f_val is not None) and (abs(val - f_val) <= (sigma_win * f_sigma_mult))

                if not peer_support and not flow_support:
                    raw_prob = min(raw_prob, 0.75)

                if not peer_support and flow_support:
                    max_conf = 0.6 + 0.4 * r_info['trust']
                    raw_prob = min(raw_prob, max_conf)

                return max(0.5, min(1.0, raw_prob))

            conf_rx = calibrate(d['est_rx'], d['s_rx'], d['p_tx'], True)
            conf_tx = calibrate(d['est_tx'], d['s_tx'], d['p_rx'], False)

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