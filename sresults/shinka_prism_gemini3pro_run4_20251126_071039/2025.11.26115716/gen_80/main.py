# EVOLVE-BLOCK-START
"""Model placement algorithm for minimizing maximum KV cache pressure across GPUs"""

import random
import math

GPU_MEM_SIZE = 80.0  # GB

def compute_model_placement(gpu_num, models):
    """
    Minimizes max KVPR using Binary Search on the target pressure K.
    The feasibility check transforms the problem into Bin Packing with item size (w + K*s).
    Uses deterministic and stochastic ordering strategies for robust packing.
    """

    # 0. Precompute model data
    m_data = []
    total_w = 0.0
    total_s = 0.0
    for i, m in enumerate(models):
        w = m.req_rate / m.slo
        s = m.model_size
        m_data.append({'w': w, 's': s, 'obj': m, 'id': i})
        total_w += w
        total_s += s

    # Theoretical Lower Bound
    # sum(w) / (N*C - sum(s))
    rem_global = gpu_num * GPU_MEM_SIZE - total_s
    if rem_global <= 1e-9:
        if total_w > 0: lb = float('inf')
        else: lb = 0.0
    else:
        lb = total_w / rem_global

    # -------------------------------------------------------
    # Helper: Feasibility Check for Pressure K
    # -------------------------------------------------------
    def try_pack(target_k, max_random_trials=0):
        """
        Attempts to pack all models such that for every GPU:
        sum(w) / (C - sum(s)) <= target_k
        Equivalent to: sum(w + target_k * s) <= target_k * C
        Returns placement dict if successful, else None.
        """

        # Strategies: (key_function, reverse_bool, packing_mode)
        # packing_mode: 'best_fit' (min slack) or 'worst_fit' (max slack/load balancing)
        strategies = [
            (lambda x: x['w'] + target_k * x['s'], True, 'best_fit'),
            (lambda x: x['s'], True, 'best_fit'),
            (lambda x: x['w'], True, 'best_fit'),
            (lambda x: x['w'] / (x['s'] + 1e-9), True, 'best_fit'),
        ]

        # Indices for trials
        trials = list(range(len(strategies)))
        if max_random_trials > 0:
            trials.extend(['rand'] * max_random_trials)

        failed_items = set()

        for t in trials:
            mode = 'best_fit'
            # Sort indices based on strategy
            if isinstance(t, int):
                key_func, reverse, mode = strategies[t]
                ordered_indices = sorted(range(len(m_data)), key=lambda i: key_func(m_data[i]), reverse=reverse)
            else:
                # Stochastic: Perturbed Virtual Size with Failure Feedback
                # Randomly choose between best_fit and worst_fit to diversify search
                if random.random() < 0.15:
                    mode = 'worst_fit'

                def sort_key(i):
                    item = m_data[i]
                    val = (item['w'] + target_k * item['s'])
                    val *= random.uniform(0.85, 1.15)
                    # Boost if failed previously
                    if item['id'] in failed_items:
                        val += 1e9
                    return val

                ordered_indices = sorted(range(len(m_data)), key=sort_key, reverse=True)

            # Perform Packing
            bins = [{'w': 0.0, 's': 0.0, 'idxs': []} for _ in range(gpu_num)]
            possible = True
            first_fail_idx = None

            for idx in ordered_indices:
                item = m_data[idx]
                w, s = item['w'], item['s']

                best_bin = None

                # Initialize comparison metric
                if mode == 'best_fit':
                    best_metric = float('inf') # Minimize slack
                else:
                    best_metric = -1.0 # Maximize slack

                for b_idx in range(gpu_num):
                    b = bins[b_idx]

                    # 1. Physical Fit
                    if b['s'] + s > GPU_MEM_SIZE: continue

                    # 2. Pressure Fit
                    rem_new = GPU_MEM_SIZE - (b['s'] + s)
                    if rem_new < 0: rem_new = 0.0

                    max_w = target_k * rem_new
                    new_w = b['w'] + w

                    if new_w <= max_w + 1e-5:
                        # Feasible
                        slack = max_w - new_w
                        if mode == 'best_fit':
                            if slack < best_metric:
                                best_metric = slack
                                best_bin = b_idx
                        else:
                            if slack > best_metric:
                                best_metric = slack
                                best_bin = b_idx

                if best_bin is None:
                    possible = False
                    first_fail_idx = idx
                    break

                bins[best_bin]['idxs'].append(idx)
                bins[best_bin]['w'] += w
                bins[best_bin]['s'] += s

            if possible:
                return {i: bins[i]['idxs'] for i in range(gpu_num)}
            else:
                if first_fail_idx is not None:
                    failed_items.add(m_data[first_fail_idx]['id'])

        return None

    # -------------------------------------------------------
    # 1. Search Initialization
    # -------------------------------------------------------
    best_placement = None
    best_max_kvpr = float('inf')

    # Establish an upper bound with a very loose pressure constraint (essentially physical packing)
    # Then evaluate its real pressure.
    init_k = 2000.0
    initial_sol = try_pack(init_k, max_random_trials=0)

    if initial_sol:
        # Evaluate
        curr_max = 0.0
        for idxs in initial_sol.values():
            w = sum(m_data[i]['w'] for i in idxs)
            s = sum(m_data[i]['s'] for i in idxs)
            rem = GPU_MEM_SIZE - s
            if rem > 1e-9:
                val = w/rem
            elif w > 0:
                val = float('inf')
            else:
                val = 0.0
            curr_max = max(curr_max, val)
        best_max_kvpr = curr_max
        best_placement = initial_sol

    # Define Binary Search Range
    high = best_max_kvpr if best_max_kvpr != float('inf') else 5000.0
    low = lb

    # -------------------------------------------------------
    # 2. Binary Search
    # -------------------------------------------------------
    # Perform search to push K down.
    # The stochastic packer gives us a good chance to find a valid config if one exists near 'mid'.
    if high > low + 1e-4:
        # Number of BS iterations
        for bs_iter in range(32):
            mid = (low + high) / 2.0

            # Adaptive Stochastic Depth: Increase trials as we narrow down (bounds converge)
            # Early iterations: loose bounds, easy to check -> few trials
            # Late iterations: tight bounds, hard to check -> many trials
            adaptive_trials = 10 + int(40 * (bs_iter / 31.0))

            # Use randomization to try harder to fit into 'mid'
            sol = try_pack(mid, max_random_trials=adaptive_trials)

            if sol:
                # Found a valid packing.
                # However, the packing only guarantees KVPR <= mid (approx).
                # We save it and try to find an even smaller mid.

                # Recalculate actual max pressure to keep the best real solution found
                curr_max = 0.0
                for idxs in sol.values():
                    w = sum(m_data[i]['w'] for i in idxs)
                    s = sum(m_data[i]['s'] for i in idxs)
                    rem = GPU_MEM_SIZE - s
                    val = w/rem if rem > 1e-9 else (float('inf') if w > 0 else 0.0)
                    curr_max = max(curr_max, val)

                if curr_max < best_max_kvpr:
                    best_max_kvpr = curr_max
                    best_placement = sol

                high = mid
            else:
                low = mid

    if best_placement is None:
         raise ValueError("Unable to place models on GPUs with available memory.")

    # -------------------------------------------------------
    # 3. Local Search Refinement
    # -------------------------------------------------------
    # Convert best_placement (indices) to lists of objects for output,
    # but keep working with indices for local search speed.
    curr_map = best_placement # dict: gpu_id -> list of indices

    # Precalculate stats
    g_stats = []
    for g in range(gpu_num):
        idxs = curr_map[g]
        w = sum(m_data[i]['w'] for i in idxs)
        s = sum(m_data[i]['s'] for i in idxs)
        rem = GPU_MEM_SIZE - s
        p = w / rem if rem > 1e-9 else (float('inf') if w > 0 else 0.0)
        g_stats.append({'w': w, 's': s, 'p': p})

    # Optimization Loop (Steepest Descent with Tie-Breaking)
    # Objective: Minimize Max KVPR. Secondary: Minimize Sum of Squared KVPR.

    # Calculate initial sum of squares
    sum_sq_p = sum(gs['p']**2 for gs in g_stats)

    for _ in range(300):
        # Identify bottleneck
        max_p = -1.0
        src_gpu = -1
        for g in range(gpu_num):
            if g_stats[g]['p'] > max_p:
                max_p = g_stats[g]['p']
                src_gpu = g

        if src_gpu == -1 or max_p < 1e-9: break

        best_action = None # (type, improvement_metric, details)
        # improvement_metric: (new_max_pair, new_sum_sq_delta) - minimize this

        src_list = curr_map[src_gpu]

        # 3.1 Evaluate Moves (Src -> Dst)
        for list_idx, m_idx in enumerate(src_list):
            m = m_data[m_idx]

            src_rem_new = GPU_MEM_SIZE - (g_stats[src_gpu]['s'] - m['s'])
            src_w_new = g_stats[src_gpu]['w'] - m['w']
            src_p_new = src_w_new / src_rem_new if src_rem_new > 1e-9 else (float('inf') if src_w_new > 0 else 0.0)

            for dst in range(gpu_num):
                if dst == src_gpu: continue
                if g_stats[dst]['s'] + m['s'] > GPU_MEM_SIZE: continue

                dst_rem_new = GPU_MEM_SIZE - (g_stats[dst]['s'] + m['s'])
                dst_w_new = g_stats[dst]['w'] + m['w']
                dst_p_new = dst_w_new / dst_rem_new if dst_rem_new > 1e-9 else (float('inf') if dst_w_new > 0 else 0.0)

                new_max_pair = max(src_p_new, dst_p_new)

                # Check strict improvement on max_p for the involved pair
                if new_max_pair >= max_p - 1e-7: continue

                # Delta Sum Sq: (new_src^2 + new_dst^2) - (old_src^2 + old_dst^2)
                delta_sq = (src_p_new**2 + dst_p_new**2) - (g_stats[src_gpu]['p']**2 + g_stats[dst]['p']**2)

                metric = (new_max_pair, delta_sq)

                if best_action is None or metric < best_action[1]:
                    best_action = ('move', metric, (list_idx, dst, m_idx, src_p_new, dst_p_new))

        # 3.2 Evaluate Swaps (Src <-> Dst)
        for s_list_idx, m_src_idx in enumerate(src_list):
            m_src = m_data[m_src_idx]

            for dst in range(gpu_num):
                if dst == src_gpu: continue
                dst_list = curr_map[dst]

                for d_list_idx, m_dst_idx in enumerate(dst_list):
                    m_dst = m_data[m_dst_idx]

                    new_src_s = g_stats[src_gpu]['s'] - m_src['s'] + m_dst['s']
                    if new_src_s > GPU_MEM_SIZE: continue
                    new_dst_s = g_stats[dst]['s'] - m_dst['s'] + m_src['s']
                    if new_dst_s > GPU_MEM_SIZE: continue

                    new_src_rem = GPU_MEM_SIZE - new_src_s
                    new_src_w = g_stats[src_gpu]['w'] - m_src['w'] + m_dst['w']
                    new_src_p = new_src_w / new_src_rem if new_src_rem > 1e-9 else float('inf')

                    new_dst_rem = GPU_MEM_SIZE - new_dst_s
                    new_dst_w = g_stats[dst]['w'] - m_dst['w'] + m_src['w']
                    new_dst_p = new_dst_w / new_dst_rem if new_dst_rem > 1e-9 else float('inf')

                    new_max_pair = max(new_src_p, new_dst_p)

                    if new_max_pair >= max_p - 1e-7: continue

                    delta_sq = (new_src_p**2 + new_dst_p**2) - (g_stats[src_gpu]['p']**2 + g_stats[dst]['p']**2)
                    metric = (new_max_pair, delta_sq)

                    if best_action is None or metric < best_action[1]:
                        best_action = ('swap', metric, (s_list_idx, dst, d_list_idx, m_src_idx, m_dst_idx, new_src_p, new_dst_p))

        # Execute Best Action
        if best_action:
            act_type, _, details = best_action
            if act_type == 'move':
                s_list_idx, dst_gpu, m_idx, sp, dp = details
                m = m_data[m_idx]

                curr_map[src_gpu].pop(s_list_idx)
                curr_map[dst_gpu].append(m_idx)

                g_stats[src_gpu]['w'] -= m['w']
                g_stats[src_gpu]['s'] -= m['s']
                g_stats[src_gpu]['p'] = sp

                g_stats[dst_gpu]['w'] += m['w']
                g_stats[dst_gpu]['s'] += m['s']
                g_stats[dst_gpu]['p'] = dp

                sum_sq_p += best_action[1][1]

            elif act_type == 'swap':
                s_l_idx, dst_gpu, d_l_idx, m_s_idx, m_d_idx, sp, dp = details
                m_s = m_data[m_s_idx]
                m_d = m_data[m_d_idx]

                curr_map[src_gpu][s_l_idx] = m_d_idx
                curr_map[dst_gpu][d_l_idx] = m_s_idx

                g_stats[src_gpu]['w'] = g_stats[src_gpu]['w'] - m_s['w'] + m_d['w']
                g_stats[src_gpu]['s'] = g_stats[src_gpu]['s'] - m_s['s'] + m_d['s']
                g_stats[src_gpu]['p'] = sp

                g_stats[dst_gpu]['w'] = g_stats[dst_gpu]['w'] - m_d['w'] + m_s['w']
                g_stats[dst_gpu]['s'] = g_stats[dst_gpu]['s'] - m_d['s'] + m_s['s']
                g_stats[dst_gpu]['p'] = dp

                sum_sq_p += best_action[1][1]
        else:
            break

    # Final conversion
    result = {}
    for g, idxs in curr_map.items():
        result[g] = [m_data[i]['obj'] for i in idxs]

    return result

# EVOLVE-BLOCK-END


def run_placement(gpu_num, models):
    """
    Main entry point that will be called by the evaluator.

    Args:
        gpu_num: Number of GPUs
        models: List of models to place

    Returns:
        Dictionary containing GPU placements
    """
    return compute_model_placement(gpu_num, models)


if __name__ == "__main__":
    # Test the algorithm
    import os
    import sys

    # Add the openevolve_examples directory to the path to import evaluator
    def find_repo_root(start_path):
        """Find the repository root by looking for openevolve_examples directory."""
        current = os.path.abspath(start_path)
        while current != os.path.dirname(current):  # Stop at filesystem root
            if os.path.exists(os.path.join(current, 'openevolve_examples', 'prism')):
                return current
            current = os.path.dirname(current)
        raise RuntimeError("Could not find openevolve_examples directory")

    repo_root = find_repo_root(os.path.dirname(__file__))
    sys.path.insert(0, os.path.join(repo_root, 'openevolve_examples', 'prism'))

    from evaluator import generate_test_gpu_models, calculate_kvcache_pressure, safe_float
    import numpy as np

    test_cases = generate_test_gpu_models()
    all_kvpr = []
    for i, (gpu_num, gpu_models) in enumerate(test_cases):
        results = compute_model_placement(gpu_num, gpu_models)
        max_kvpr = calculate_kvcache_pressure(results)
        all_kvpr.append(safe_float(max_kvpr))

    avg_kvpr = np.mean(all_kvpr)
    if avg_kvpr != 0:
        avg_kvpr = 1.0 / avg_kvpr

    print(f"Max KVPR: {avg_kvpr:.3f}")