# EVOLVE-BLOCK-START
"""Model placement algorithm for minimizing maximum KV cache pressure across GPUs"""

import random

GPU_MEM_SIZE = 80  # GB

def compute_model_placement(gpu_num, models):
    """
    Compute a model placement that minimizes the maximum KVPR across all GPUs.
    Combines Greedy Heuristics, Binary Search (Bin Packing), and Local Search.
    """

    # Helper to calculate max KVPR of a placement
    def get_max_kvpr(placement):
        max_p = 0.0
        for assigned in placement.values():
            w = sum(m.req_rate / m.slo for m in assigned)
            s = sum(m.model_size for m in assigned)
            rem = GPU_MEM_SIZE - s
            if rem <= 1e-9:
                if w > 0: return float('inf')
                else: continue
            max_p = max(max_p, w / rem)
        return max_p

    best_placement = None
    best_score = float('inf')

    # ---------------------------------------------------------
    # 1. Greedy Heuristics Ensemble
    # ---------------------------------------------------------
    # Fast initial solutions to set an upper bound

    # Estimate K for linearized heuristic
    t_w = sum(m.req_rate / m.slo for m in models)
    t_s = sum(m.model_size for m in models)
    t_rem = gpu_num * GPU_MEM_SIZE - t_s
    k_est = t_w / t_rem if t_rem > 1e-9 else 1.0

    heuristics = [
        # (Key Function, Strategy Name)
        # Strategy 'min_result': Place on GPU minimizing resulting KVPR
        # Strategy 'min_current': Place on GPU minimizing current KVPR (Load Balancing)
        (lambda m: (m.req_rate / m.slo, m.model_size), 'min_result'),
        (lambda m: m.model_size, 'min_result'),
        (lambda m: (m.req_rate / m.slo) / (GPU_MEM_SIZE - m.model_size + 1e-6), 'min_result'),
        (lambda m: m.req_rate / m.slo, 'min_current'),
        # Linearized pressure heuristic: w + K*s
        (lambda m: (m.req_rate / m.slo) + k_est * m.model_size, 'min_result'),
    ]

    for key_fn, strategy in heuristics:
        sorted_models = sorted(models, key=key_fn, reverse=True)
        placement = {i: [] for i in range(gpu_num)}
        gpu_w = [0.0] * gpu_num
        gpu_s = [0.0] * gpu_num
        possible = True

        for model in sorted_models:
            w = model.req_rate / model.slo
            s = model.model_size
            best_idx = None
            best_val = float('inf')

            for i in range(gpu_num):
                if gpu_s[i] + s > GPU_MEM_SIZE: continue
                rem = GPU_MEM_SIZE - gpu_s[i]

                if strategy == 'min_result':
                    new_rem = rem - s
                    if new_rem > 1e-9:
                        val = (gpu_w[i] + w) / new_rem
                    else:
                        val = float('inf')
                else: # min_current
                    if rem > 1e-9:
                        val = gpu_w[i] / rem
                    else:
                        val = float('inf')

                if val < best_val:
                    best_val = val
                    best_idx = i
                elif val == best_val and best_idx is None:
                    best_idx = i

            if best_idx is None:
                possible = False
                break

            placement[best_idx].append(model)
            gpu_w[best_idx] += w
            gpu_s[best_idx] += s

        if possible:
            score = get_max_kvpr(placement)
            if score < best_score:
                best_score = score
                best_placement = placement

    # ---------------------------------------------------------
    # 2. Binary Search on Target KVPR (Transformation to Bin Packing)
    # ---------------------------------------------------------
    # Problem: Minimize K such that sum(w_i)/ (C - sum(s_i)) <= K
    # Equivalent to Bin Packing with item size v_i(K) = s_i + w_i/K, bin capacity C.

    total_w = sum(m.req_rate / m.slo for m in models)
    total_s = sum(m.model_size for m in models)
    rem_global = gpu_num * GPU_MEM_SIZE - total_s

    def refine_solution(placement, iterations=20):
        """Optimizes placement via steepest descent Move/Swap."""
        # Setup working stats
        g_map = {}
        for g, m_list in placement.items():
            w_sum = sum(m.req_rate / m.slo for m in m_list)
            s_sum = sum(m.model_size for m in m_list)
            g_map[g] = {'models': list(m_list), 'w': w_sum, 's': s_sum}

        # Cache GPU pressures calculation
        def calc_pressure(g_idx):
            rem = GPU_MEM_SIZE - g_map[g_idx]['s']
            return g_map[g_idx]['w'] / rem if rem > 1e-9 else (float('inf') if g_map[g_idx]['w'] > 0 else 0.0)

        for _ in range(iterations):
            # Identify Bottleneck
            max_p = -1.0
            src = -1
            current_pressures = [calc_pressure(g) for g in range(gpu_num)]
            for g, p in enumerate(current_pressures):
                if p > max_p:
                    max_p = p
                    src = g

            if src == -1 or max_p < 1e-9: break

            # Steepest Descent: Find the single best action across all moves/swaps
            best_action = None # (type, score, details...)
            best_val = max_p

            src_models = g_map[src]['models']

            # 1. Evaluate MOVES (Src -> Dst)
            for i, m in enumerate(src_models):
                w, s = m.req_rate / m.slo, m.model_size

                # Hypothetical Src after move
                src_rem_new = GPU_MEM_SIZE - (g_map[src]['s'] - s)
                src_w_new = g_map[src]['w'] - w
                src_p_new = src_w_new / src_rem_new if src_rem_new > 1e-9 else (float('inf') if src_w_new > 0 else 0.0)

                for dst in range(gpu_num):
                    if dst == src: continue
                    if g_map[dst]['s'] + s > GPU_MEM_SIZE: continue

                    dst_rem_new = GPU_MEM_SIZE - (g_map[dst]['s'] + s)
                    if dst_rem_new <= 1e-9: continue
                    dst_w_new = g_map[dst]['w'] + w
                    dst_p_new = dst_w_new / dst_rem_new

                    # Minimize the bottleneck of the involved GPUs
                    local_max = max(src_p_new, dst_p_new)
                    if local_max < best_val - 1e-6:
                        best_val = local_max
                        best_action = ('move', i, dst)

            # 2. Evaluate SWAPS (Src <-> Dst)
            for i, m_src in enumerate(src_models):
                w_src, s_src = m_src.req_rate / m_src.slo, m_src.model_size

                for dst in range(gpu_num):
                    if dst == src: continue
                    dst_models = g_map[dst]['models']

                    for j, m_dst in enumerate(dst_models):
                        w_dst, s_dst = m_dst.req_rate / m_dst.slo, m_dst.model_size

                        # Capacity Check
                        new_src_s = g_map[src]['s'] - s_src + s_dst
                        if new_src_s > GPU_MEM_SIZE: continue
                        new_dst_s = g_map[dst]['s'] - s_dst + s_src
                        if new_dst_s > GPU_MEM_SIZE: continue

                        # Pressure Check
                        rem_src = GPU_MEM_SIZE - new_src_s
                        if rem_src <= 1e-9: continue
                        p_src = (g_map[src]['w'] - w_src + w_dst) / rem_src

                        rem_dst = GPU_MEM_SIZE - new_dst_s
                        if rem_dst <= 1e-9: continue
                        p_dst = (g_map[dst]['w'] - w_dst + w_src) / rem_dst

                        local_max = max(p_src, p_dst)
                        if local_max < best_val - 1e-6:
                            best_val = local_max
                            best_action = ('swap', i, dst, j)

            if best_action:
                if best_action[0] == 'move':
                    _, i, dst = best_action
                    m = src_models.pop(i)
                    g_map[src]['w'] -= m.req_rate / m.slo
                    g_map[src]['s'] -= m.model_size
                    g_map[dst]['models'].append(m)
                    g_map[dst]['w'] += m.req_rate / m.slo
                    g_map[dst]['s'] += m.model_size
                else:
                    _, i, dst, j = best_action
                    m_src = src_models[i]
                    m_dst = g_map[dst]['models'][j]

                    # Update lists
                    src_models[i] = m_dst
                    g_map[dst]['models'][j] = m_src

                    # Update stats
                    w_src, s_src = m_src.req_rate / m_src.slo, m_src.model_size
                    w_dst, s_dst = m_dst.req_rate / m_dst.slo, m_dst.model_size

                    g_map[src]['w'] += (w_dst - w_src)
                    g_map[src]['s'] += (s_dst - s_src)
                    g_map[dst]['w'] += (w_src - w_dst)
                    g_map[dst]['s'] += (s_src - s_dst)
            else:
                break # Local optimum

        return {g: g_map[g]['models'] for g in range(gpu_num)}

    if rem_global > 1e-6:
        low = total_w / rem_global
        high = best_score if best_score != float('inf') else 2000.0

        if high > low + 1e-4:
            for _ in range(16):
                mid = (low + high) / 2

                found_placement = None
                failed_items = set()
                strategies = [
                    (lambda m: m.model_size + (m.req_rate / m.slo) / mid, 'Effective'),
                    (lambda m: m.model_size, 'Physical'),
                    (lambda m: m.req_rate / m.slo, 'Weight'),
                    (lambda m: (m.req_rate / m.slo) / (m.model_size + 1e-6), 'Density')
                ]

                for attempt in range(30):
                    base_key, _ = strategies[attempt % len(strategies)]
                    noise = 0.0
                    if attempt >= 4:
                        noise = 0.01 + (attempt * 0.003)

                    # Randomly toggle between Best Fit and Worst Fit to escape traps
                    fit_mode = 'best_fit'
                    if attempt > 10 and random.random() < 0.25:
                        fit_mode = 'worst_fit'

                    def sort_key(m):
                        val = base_key(m)
                        if noise > 0: val *= random.uniform(1.0 - noise, 1.0 + noise)
                        if id(m) in failed_items: val += 1e9
                        return val

                    bs_models = sorted(models, key=sort_key, reverse=True)
                    temp_placement = {i: [] for i in range(gpu_num)}
                    gpu_w = [0.0] * gpu_num
                    gpu_s = [0.0] * gpu_num
                    possible_k = True
                    first_fail_model = None

                    for model in bs_models:
                        w = model.req_rate / model.slo
                        s = model.model_size
                        eff = s + w/mid

                        best_idx = None

                        if fit_mode == 'best_fit':
                            best_metric = float('inf') # Minimize remaining effective capacity
                        else:
                            best_metric = -1.0 # Maximize remaining effective capacity

                        for i in range(gpu_num):
                            if gpu_s[i] + s > GPU_MEM_SIZE: continue
                            curr_eff = gpu_s[i] + gpu_w[i]/mid
                            if curr_eff + eff <= GPU_MEM_SIZE + 1e-6:
                                rem_eff = GPU_MEM_SIZE - (curr_eff + eff)

                                if fit_mode == 'best_fit':
                                    if rem_eff < best_metric:
                                        best_metric = rem_eff
                                        best_idx = i
                                else:
                                    if rem_eff > best_metric:
                                        best_metric = rem_eff
                                        best_idx = i

                        if best_idx is None:
                            possible_k = False
                            first_fail_model = model
                            break

                        temp_placement[best_idx].append(model)
                        gpu_w[best_idx] += w
                        gpu_s[best_idx] += s

                    if possible_k:
                        found_placement = temp_placement
                        break
                    else:
                        if first_fail_model: failed_items.add(id(first_fail_model))

                if found_placement:
                    # Instant refinement
                    refined = refine_solution(found_placement, iterations=20)
                    actual_score = get_max_kvpr(refined)
                    if actual_score < best_score:
                        best_score = actual_score
                        best_placement = refined
                    high = min(mid, actual_score)
                else:
                    low = mid

    if best_placement is None:
        raise ValueError("Unable to place models on GPUs with available memory.")

    # ---------------------------------------------------------
    # 3. Local Search Refinement
    # ---------------------------------------------------------
    # Final cleanup using the same refinement logic
    best_placement = refine_solution(best_placement, iterations=50)
    return best_placement

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