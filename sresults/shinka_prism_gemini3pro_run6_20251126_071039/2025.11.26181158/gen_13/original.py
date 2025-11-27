# EVOLVE-BLOCK-START
"""Model placement algorithm for minimizing maximum KV cache pressure across GPUs"""

GPU_MEM_SIZE = 80  # GB

def compute_model_placement(gpu_num, models):
    """
    Compute a model placement that minimizes the maximum KVPR across all GPUs.

    Algorithm:
    1. Binary Search on target KVPR 'K'.
       - Checks feasibility by solving a Bin Packing problem with Best Fit.
       - Uses multiple sorting heuristics (Linearized Weight, Size) to maximize packing success.
    2. Local Search Refinement.
       - Greedily moves models from the bottleneck GPU (max KVPR) to other GPUs
         to reduce the global maximum KVPR.
    """

    # Pre-calculate model properties
    model_data = []
    for i, m in enumerate(models):
        model_data.append({
            'model': m,
            'l': m.req_rate / m.slo,
            's': m.model_size
        })

    def solve_packing(target_k):
        """
        Attempts to place models into gpu_num bins given a target KVPR 'K'.
        Constraint per GPU: sum(L) + K * sum(S) <= K * M
        Returns placement dict if successful, None otherwise.
        """
        capacity = target_k * GPU_MEM_SIZE

        # Helper to try a specific packing order with Best Fit
        def try_best_fit(items):
            gpu_l = [0.0] * gpu_num
            gpu_s = [0.0] * gpu_num
            gpu_models = [[] for _ in range(gpu_num)]

            for item in items:
                best_idx = -1
                min_remaining = float('inf')

                w = item['l'] + target_k * item['s']

                for i in range(gpu_num):
                    # Hard memory constraint
                    if gpu_s[i] + item['s'] >= GPU_MEM_SIZE - 1e-6:
                        continue

                    # Linearized KVPR constraint
                    curr_w = gpu_l[i] + target_k * gpu_s[i]
                    if curr_w + w <= capacity + 1e-9:
                        # Best Fit: Choose bin with minimum remaining linearized capacity
                        rem = capacity - (curr_w + w)
                        if rem < min_remaining:
                            min_remaining = rem
                            best_idx = i

                if best_idx != -1:
                    gpu_l[best_idx] += item['l']
                    gpu_s[best_idx] += item['s']
                    gpu_models[best_idx].append(item['model'])
                else:
                    return None
            return gpu_models

        # Strategy 1: Sort by Linearized Weight Descending
        # Effective for generic bin packing logic
        items_1 = sorted(model_data, key=lambda x: x['l'] + target_k * x['s'], reverse=True)
        res = try_best_fit(items_1)
        if res: return res

        # Strategy 2: Sort by Size Descending
        # Effective when memory is the primary constraint
        items_2 = sorted(model_data, key=lambda x: x['s'], reverse=True)
        res = try_best_fit(items_2)
        if res: return res

        # Strategy 3: Sort by Load Descending
        # Effective when load is the primary constraint
        items_3 = sorted(model_data, key=lambda x: x['l'], reverse=True)
        res = try_best_fit(items_3)
        if res: return res

        return None

    # --- Phase 1: Binary Search for Optimal K ---

    # Find valid upper bound
    low = 0.0
    high = 1.0
    best_placement_list = None

    # Exponential search
    for _ in range(20):
        res = solve_packing(high)
        if res is not None:
            best_placement_list = res
            break
        low = high
        high *= 2.0
    else:
        # Fallback
        high = 1e9

    # Refine K
    for _ in range(25):
        mid = (low + high) / 2
        res = solve_packing(mid)
        if res is not None:
            best_placement_list = res
            high = mid
        else:
            low = mid

    # Ensure we have a placement
    if best_placement_list is None:
        best_placement_list = solve_packing(high)
        if best_placement_list is None:
            raise ValueError("Unable to place models even with infinite KVPR.")

    placement = {i: best_placement_list[i] for i in range(gpu_num)}

    # --- Phase 2: Local Search Refinement with Moves and Swaps ---

    # Initialize state for faster computation
    gpu_states = []
    for g in range(gpu_num):
        models_g = placement[g]
        s_g = sum(m.model_size for m in models_g)
        l_g = sum(m.req_rate / m.slo for m in models_g)
        gpu_states.append({'models': models_g, 's': s_g, 'l': l_g})

    def get_kvpr(l, s):
        if s >= GPU_MEM_SIZE - 1e-6: return 1e15 # Penalty for full/overflow
        return l / (GPU_MEM_SIZE - s)

    # Iterative improvement
    for _ in range(100):
        # Identify bottleneck GPU
        current_kvprs = [get_kvpr(gs['l'], gs['s']) for gs in gpu_states]
        max_kvpr = max(current_kvprs)
        max_gpu = current_kvprs.index(max_kvpr)

        if max_kvpr < 1e-9: break

        best_action = None
        best_new_max = max_kvpr

        # 1. Try Moving a model from max_gpu to any other GPU
        src_models = gpu_states[max_gpu]['models']
        for s_idx, model in enumerate(src_models):
            m_l = model.req_rate / model.slo
            m_s = model.model_size

            for t_gpu in range(gpu_num):
                if t_gpu == max_gpu: continue

                # Quick capacity check
                if gpu_states[t_gpu]['s'] + m_s >= GPU_MEM_SIZE: continue

                # Calc projected KVPRs
                new_src_l = gpu_states[max_gpu]['l'] - m_l
                new_src_s = gpu_states[max_gpu]['s'] - m_s
                new_tgt_l = gpu_states[t_gpu]['l'] + m_l
                new_tgt_s = gpu_states[t_gpu]['s'] + m_s

                ks = get_kvpr(new_src_l, new_src_s)
                kt = get_kvpr(new_tgt_l, new_tgt_s)

                local_max = max(ks, kt)

                # We want to reduce the max KVPR. Strict improvement required.
                if local_max < best_new_max - 1e-6:
                    best_new_max = local_max
                    best_action = ('move', s_idx, t_gpu)

        # 2. Try Swapping a model from max_gpu with a model from another GPU
        for s_idx, s_model in enumerate(src_models):
            s_l = s_model.req_rate / s_model.slo
            s_s = s_model.model_size

            for t_gpu in range(gpu_num):
                if t_gpu == max_gpu: continue

                # Optimization: Don't swap with a GPU that is already worse or equal
                if current_kvprs[t_gpu] >= max_kvpr: continue

                tgt_models = gpu_states[t_gpu]['models']
                for t_idx, t_model in enumerate(tgt_models):
                    t_l = t_model.req_rate / t_model.slo
                    t_s = t_model.model_size

                    # Capacity check
                    new_src_s = gpu_states[max_gpu]['s'] - s_s + t_s
                    new_tgt_s = gpu_states[t_gpu]['s'] - t_s + s_s

                    if new_src_s >= GPU_MEM_SIZE or new_tgt_s >= GPU_MEM_SIZE:
                        continue

                    new_src_l = gpu_states[max_gpu]['l'] - s_l + t_l
                    new_tgt_l = gpu_states[t_gpu]['l'] - t_l + s_l

                    ks = get_kvpr(new_src_l, new_src_s)
                    kt = get_kvpr(new_tgt_l, new_tgt_s)

                    local_max = max(ks, kt)

                    if local_max < best_new_max - 1e-6:
                        best_new_max = local_max
                        best_action = ('swap', s_idx, t_gpu, t_idx)

        # Apply best action
        if best_action:
            if best_action[0] == 'move':
                _, s_idx, t_gpu = best_action
                model = gpu_states[max_gpu]['models'].pop(s_idx)

                gpu_states[max_gpu]['l'] -= model.req_rate / model.slo
                gpu_states[max_gpu]['s'] -= model.model_size

                gpu_states[t_gpu]['models'].append(model)
                gpu_states[t_gpu]['l'] += model.req_rate / model.slo
                gpu_states[t_gpu]['s'] += model.model_size

            elif best_action[0] == 'swap':
                _, s_idx, t_gpu, t_idx = best_action
                s_model = gpu_states[max_gpu]['models'][s_idx]
                t_model = gpu_states[t_gpu]['models'][t_idx]

                # Swap in lists
                gpu_states[max_gpu]['models'][s_idx] = t_model
                gpu_states[t_gpu]['models'][t_idx] = s_model

                # Update stats
                sl_diff = (t_model.req_rate/t_model.slo) - (s_model.req_rate/s_model.slo)
                ss_diff = t_model.model_size - s_model.model_size

                gpu_states[max_gpu]['l'] += sl_diff
                gpu_states[max_gpu]['s'] += ss_diff

                gpu_states[t_gpu]['l'] -= sl_diff
                gpu_states[t_gpu]['s'] -= ss_diff
        else:
            break

    return {g: state['models'] for g, state in enumerate(gpu_states)}

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