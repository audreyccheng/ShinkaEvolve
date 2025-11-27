# EVOLVE-BLOCK-START
"""Model placement algorithm for minimizing maximum KV cache pressure across GPUs"""

import random

GPU_MEM_SIZE = 80.0  # GB

def compute_model_placement(gpu_num, models):
    """
    Compute a model placement that minimizes the maximum KVPR across all GPUs.

    Phases:
    1. Greedy Ensemble: Quick valid solutions to establish upper bounds.
    2. Binary Search: Minimizes K using Bin Packing with Effective Size (s + w/K).
    3. Local Search: Move and Swap operations to refine the bottleneck GPU.
    """

    # 0. Precompute model data for faster access
    m_data = []
    total_w = 0.0
    total_s = 0.0
    for i, m in enumerate(models):
        w = m.req_rate / m.slo
        s = m.model_size
        m_data.append({
            'id': i,
            'w': w,
            's': s,
            'obj': m
        })
        total_w += w
        total_s += s

    # ---------------------------------------------------------
    # Helper Functions
    # ---------------------------------------------------------
    def evaluate_indices(placement_indices):
        """Calc max KVPR given dict {gpu_idx: [list_of_m_data_indices]}"""
        max_p = 0.0
        for indices in placement_indices.values():
            w_sum = sum(m_data[idx]['w'] for idx in indices)
            s_sum = sum(m_data[idx]['s'] for idx in indices)
            rem = GPU_MEM_SIZE - s_sum

            if rem <= 1e-9:
                if w_sum > 0: return float('inf')
                else: continue

            p = w_sum / rem
            if p > max_p: max_p = p
        return max_p

    def indices_to_objects(placement_indices):
        res = {}
        for g, indices in placement_indices.items():
            res[g] = [m_data[idx]['obj'] for idx in indices]
        return res

    best_placement_indices = None
    best_max_kvpr = float('inf')

    # ---------------------------------------------------------
    # 1. Greedy Ensemble Initialization
    # ---------------------------------------------------------
    # Heuristics: (Sort Key, Strategy)
    # Strategies: 'min_peak' (minimize resulting KVPR), 'min_load' (minimize current KVPR)
    heuristics = [
        (lambda x: x['w'] / (x['s'] + 1e-6), 'min_peak'),  # Density
        (lambda x: x['w'], 'min_load'),                    # Weight (Load Balance)
        (lambda x: x['s'], 'min_peak'),                    # Size (Bin Packing)
        (lambda x: x['w'] / (GPU_MEM_SIZE - x['s'] + 1e-6), 'min_peak') # Isolated Pressure
    ]

    for key_fn, strategy in heuristics:
        sorted_indices = sorted(range(len(m_data)), key=lambda i: key_fn(m_data[i]), reverse=True)

        p_indices = {i: [] for i in range(gpu_num)}
        gpu_state = [{'w': 0.0, 's': 0.0} for _ in range(gpu_num)]
        possible = True

        for idx in sorted_indices:
            item = m_data[idx]
            best_gpu = None
            best_val = float('inf')

            for i in range(gpu_num):
                if gpu_state[i]['s'] + item['s'] > GPU_MEM_SIZE: continue

                rem = GPU_MEM_SIZE - gpu_state[i]['s']

                if strategy == 'min_peak':
                    # Min-Max Greedy
                    new_rem = rem - item['s']
                    if new_rem <= 1e-9: val = float('inf')
                    else: val = (gpu_state[i]['w'] + item['w']) / new_rem
                else:
                    # Load Balancing
                    if rem <= 1e-9: val = float('inf')
                    else: val = gpu_state[i]['w'] / rem

                if val < best_val:
                    best_val = val
                    best_gpu = i

            if best_gpu is None:
                possible = False
                break

            p_indices[best_gpu].append(idx)
            gpu_state[best_gpu]['w'] += item['w']
            gpu_state[best_gpu]['s'] += item['s']

        if possible:
            score = evaluate_indices(p_indices)
            if score < best_max_kvpr:
                best_max_kvpr = score
                best_placement_indices = p_indices

    # ---------------------------------------------------------
    # 2. Binary Search with Multi-Heuristic Checking
    # ---------------------------------------------------------
    # Theoretical Lower Bound
    rem_global = gpu_num * GPU_MEM_SIZE - total_s
    low = total_w / rem_global if rem_global > 1e-6 else best_max_kvpr
    high = best_max_kvpr if best_max_kvpr != float('inf') else 1000.0

    if high > low + 1e-4:
        for _ in range(15):
            mid = (low + high) / 2.0

            # Check feasibility for target K = mid
            # Effective Size = s + w/mid.
            # We try multiple sorting orders to pack better.
            check_sorts = [
                lambda x: x['s'] + x['w'] / mid, # Effective Size
                lambda x: x['s'],                # Physical Size
                lambda x: x['w']                 # Weight
            ]

            found_config = None

            for sort_key in check_sorts:
                sorted_idx = sorted(range(len(m_data)), key=lambda i: sort_key(m_data[i]), reverse=True)

                temp_p = {i: [] for i in range(gpu_num)}
                temp_state = [{'w': 0.0, 's': 0.0} for _ in range(gpu_num)]
                feasible = True

                for idx in sorted_idx:
                    item = m_data[idx]
                    eff_size = item['s'] + item['w'] / mid

                    best_gpu = None
                    min_slack = float('inf')

                    # Best Fit Decreasing on Effective Constraint
                    for i in range(gpu_num):
                        # 1. Physical Fit
                        if temp_state[i]['s'] + item['s'] > GPU_MEM_SIZE: continue

                        # 2. Effective Fit: (Current_W + w) <= mid * (Rem_Physical - s)
                        phys_rem_after = GPU_MEM_SIZE - (temp_state[i]['s'] + item['s'])
                        if phys_rem_after < 0: phys_rem_after = 0.0

                        lhs = temp_state[i]['w'] + item['w']
                        rhs = mid * phys_rem_after

                        if lhs <= rhs + 1e-6:
                            # Slack represents unused effective capacity
                            slack = rhs - lhs
                            if slack < min_slack:
                                min_slack = slack
                                best_gpu = i

                    if best_gpu is None:
                        feasible = False
                        break

                    temp_p[best_gpu].append(idx)
                    temp_state[best_gpu]['w'] += item['w']
                    temp_state[best_gpu]['s'] += item['s']

                if feasible:
                    found_config = temp_p
                    break

            if found_config:
                score = evaluate_indices(found_config)
                if score < best_max_kvpr:
                    best_max_kvpr = score
                    best_placement_indices = found_config
                high = mid
            else:
                low = mid

    if best_placement_indices is None:
         raise ValueError("Unable to place models on GPUs with available memory.")

    # ---------------------------------------------------------
    # 3. Local Search (Moves & Swaps)
    # ---------------------------------------------------------
    # Use best_placement_indices as starting point
    # Convert to a mutable map: gpu_idx -> list of indices
    curr_map = best_placement_indices

    # Helper to get stats of a GPU
    def get_gpu_stats(g_idx, indices_list):
        w = sum(m_data[i]['w'] for i in indices_list)
        s = sum(m_data[i]['s'] for i in indices_list)
        rem = GPU_MEM_SIZE - s
        p = w / rem if rem > 1e-9 else float('inf')
        return w, s, p

    # Precalculate stats
    g_stats = []
    for g in range(gpu_num):
        w, s, p = get_gpu_stats(g, curr_map[g])
        g_stats.append({'w': w, 's': s, 'p': p})

    for _ in range(100):
        # Find bottleneck
        max_p = -1.0
        src_gpu = -1
        for g in range(gpu_num):
            if g_stats[g]['p'] > max_p:
                max_p = g_stats[g]['p']
                src_gpu = g

        if src_gpu == -1 or max_p < 1e-9: break

        improved = False
        src_list = curr_map[src_gpu]

        # 3.1 Try MOVE (Src -> Dst)
        for i_idx, m_idx in enumerate(src_list):
            m = m_data[m_idx]

            # Predict Src stats
            src_rem_s = GPU_MEM_SIZE - (g_stats[src_gpu]['s'] - m['s'])
            src_new_w = g_stats[src_gpu]['w'] - m['w']
            src_new_p = src_new_w / src_rem_s if src_rem_s > 1e-9 else float('inf')

            best_dst = None

            for dst in range(gpu_num):
                if dst == src_gpu: continue
                if g_stats[dst]['s'] + m['s'] > GPU_MEM_SIZE: continue

                dst_rem_s = GPU_MEM_SIZE - (g_stats[dst]['s'] + m['s'])
                dst_new_w = g_stats[dst]['w'] + m['w']
                dst_new_p = dst_new_w / dst_rem_s if dst_rem_s > 1e-9 else float('inf')

                # Condition: Reduce global max
                if max(src_new_p, dst_new_p) < max_p - 1e-5:
                    best_dst = dst
                    break

            if best_dst is not None:
                # Apply Move
                curr_map[src_gpu].pop(i_idx)
                curr_map[best_dst].append(m_idx)

                # Update Stats
                g_stats[src_gpu] = dict(zip(['w','s','p'], get_gpu_stats(src_gpu, curr_map[src_gpu])))
                g_stats[best_dst] = dict(zip(['w','s','p'], get_gpu_stats(best_dst, curr_map[best_dst])))
                improved = True
                break

        if improved: continue

        # 3.2 Try SWAP (Src <-> Dst)
        # Only if Move failed to improve
        for s_i_idx, m_src_idx in enumerate(src_list):
            m_src = m_data[m_src_idx]

            for dst in range(gpu_num):
                if dst == src_gpu: continue
                dst_list = curr_map[dst]

                for d_i_idx, m_dst_idx in enumerate(dst_list):
                    m_dst = m_data[m_dst_idx]

                    # Check Capacities
                    new_src_s = g_stats[src_gpu]['s'] - m_src['s'] + m_dst['s']
                    if new_src_s > GPU_MEM_SIZE: continue

                    new_dst_s = g_stats[dst]['s'] - m_dst['s'] + m_src['s']
                    if new_dst_s > GPU_MEM_SIZE: continue

                    # Check Pressures
                    new_src_rem = GPU_MEM_SIZE - new_src_s
                    new_src_w = g_stats[src_gpu]['w'] - m_src['w'] + m_dst['w']
                    new_src_p = new_src_w / new_src_rem if new_src_rem > 1e-9 else float('inf')

                    new_dst_rem = GPU_MEM_SIZE - new_dst_s
                    new_dst_w = g_stats[dst]['w'] - m_dst['w'] + m_src['w']
                    new_dst_p = new_dst_w / new_dst_rem if new_dst_rem > 1e-9 else float('inf')

                    if max(new_src_p, new_dst_p) < max_p - 1e-5:
                        # Apply Swap
                        curr_map[src_gpu].pop(s_i_idx)
                        curr_map[src_gpu].append(m_dst_idx)

                        curr_map[dst].pop(d_i_idx)
                        curr_map[dst].append(m_src_idx)

                        g_stats[src_gpu] = dict(zip(['w','s','p'], get_gpu_stats(src_gpu, curr_map[src_gpu])))
                        g_stats[dst] = dict(zip(['w','s','p'], get_gpu_stats(dst, curr_map[dst])))
                        improved = True
                        break
                if improved: break
            if improved: break

        if not improved: break

    return indices_to_objects(curr_map)

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