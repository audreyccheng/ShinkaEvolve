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
        items_1 = sorted(model_data, key=lambda x: x['l'] + target_k * x['s'], reverse=True)
        res = try_best_fit(items_1)
        if res: return res

        # Strategy 2: Sort by Size Descending
        items_2 = sorted(model_data, key=lambda x: x['s'], reverse=True)
        res = try_best_fit(items_2)
        if res: return res

        # Strategy 3: Sort by Load Descending
        items_3 = sorted(model_data, key=lambda x: x['l'], reverse=True)
        res = try_best_fit(items_3)
        if res: return res

        return None

    # --- Phase 1: Binary Search for Optimal K ---

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
        high = 1e9

    # Refine K
    for _ in range(30):
        mid = (low + high) / 2
        res = solve_packing(mid)
        if res is not None:
            best_placement_list = res
            high = mid
        else:
            low = mid

    if best_placement_list is None:
        best_placement_list = solve_packing(high)
        if best_placement_list is None:
            raise ValueError("Unable to place models even with infinite KVPR.")

    # --- Phase 2: Local Search Refinement (Moves + Swaps) ---

    # Initialize working state
    gpu_state = []
    for g in range(gpu_num):
        m_list = best_placement_list[g]
        s_l = sum(m.req_rate / m.slo for m in m_list)
        s_s = sum(m.model_size for m in m_list)
        gpu_state.append({'l': s_l, 's': s_s, 'models': m_list})

    def get_kvpr(cl, cs):
        if cs >= GPU_MEM_SIZE - 1e-9: return float('inf')
        return cl / (GPU_MEM_SIZE - cs)

    for _ in range(100):
        # Identify bottleneck GPU
        max_kvpr = -1.0
        max_gpu = -1
        for g in range(gpu_num):
            val = get_kvpr(gpu_state[g]['l'], gpu_state[g]['s'])
            if val > max_kvpr:
                max_kvpr = val
                max_gpu = g

        if max_kvpr == 0: break

        best_action = None
        best_gain = 0.0

        src_st = gpu_state[max_gpu]

        # 1. Try Moving a model from max_gpu -> tgt_gpu
        for m_idx, model in enumerate(src_st['models']):
            m_l = model.req_rate / model.slo
            m_s = model.model_size

            # Src state after move
            n_src_l = src_st['l'] - m_l
            n_src_s = src_st['s'] - m_s
            n_src_kvpr = get_kvpr(n_src_l, n_src_s)

            for tgt_idx in range(gpu_num):
                if tgt_idx == max_gpu: continue
                tgt_st = gpu_state[tgt_idx]

                if tgt_st['s'] + m_s >= GPU_MEM_SIZE: continue

                n_tgt_l = tgt_st['l'] + m_l
                n_tgt_s = tgt_st['s'] + m_s
                n_tgt_kvpr = get_kvpr(n_tgt_l, n_tgt_s)

                new_max = max(n_src_kvpr, n_tgt_kvpr)
                if new_max < max_kvpr - 1e-6:
                    gain = max_kvpr - new_max
                    if gain > best_gain:
                        best_gain = gain
                        best_action = ('move', m_idx, tgt_idx)

        # 2. Try Swapping a model from max_gpu <-> model from tgt_gpu
        for m1_idx, m1 in enumerate(src_st['models']):
            m1_l = m1.req_rate / m1.slo
            m1_s = m1.model_size

            for tgt_idx in range(gpu_num):
                if tgt_idx == max_gpu: continue
                tgt_st = gpu_state[tgt_idx]

                for m2_idx, m2 in enumerate(tgt_st['models']):
                    m2_l = m2.req_rate / m2.slo
                    m2_s = m2.model_size

                    # Verify capacity
                    n_src_s = src_st['s'] - m1_s + m2_s
                    if n_src_s >= GPU_MEM_SIZE: continue

                    n_tgt_s = tgt_st['s'] - m2_s + m1_s
                    if n_tgt_s >= GPU_MEM_SIZE: continue

                    # Verify KVPR improvement
                    n_src_l = src_st['l'] - m1_l + m2_l
                    n_src_kvpr = get_kvpr(n_src_l, n_src_s)

                    n_tgt_l = tgt_st['l'] - m2_l + m1_l
                    n_tgt_kvpr = get_kvpr(n_tgt_l, n_tgt_s)

                    new_max = max(n_src_kvpr, n_tgt_kvpr)
                    if new_max < max_kvpr - 1e-6:
                        gain = max_kvpr - new_max
                        if gain > best_gain:
                            best_gain = gain
                            best_action = ('swap', m1_idx, tgt_idx, m2_idx)

        # Execute best action
        if best_action:
            if best_action[0] == 'move':
                _, m_idx, tgt_idx = best_action
                model = gpu_state[max_gpu]['models'].pop(m_idx)
                m_l = model.req_rate / model.slo
                m_s = model.model_size

                gpu_state[max_gpu]['l'] -= m_l
                gpu_state[max_gpu]['s'] -= m_s

                gpu_state[tgt_idx]['models'].append(model)
                gpu_state[tgt_idx]['l'] += m_l
                gpu_state[tgt_idx]['s'] += m_s

            elif best_action[0] == 'swap':
                _, m1_idx, tgt_idx, m2_idx = best_action
                # We need to pop strictly but we have indices.
                # Pop from different lists is safe.
                m1 = gpu_state[max_gpu]['models'].pop(m1_idx)
                m2 = gpu_state[tgt_idx]['models'].pop(m2_idx)

                gpu_state[max_gpu]['models'].append(m2)
                gpu_state[tgt_idx]['models'].append(m1)

                m1_l = m1.req_rate / m1.slo
                m1_s = m1.model_size
                m2_l = m2.req_rate / m2.slo
                m2_s = m2.model_size

                gpu_state[max_gpu]['l'] += (m2_l - m1_l)
                gpu_state[max_gpu]['s'] += (m2_s - m1_s)

                gpu_state[tgt_idx]['l'] += (m1_l - m2_l)
                gpu_state[tgt_idx]['s'] += (m1_s - m2_s)
        else:
            break

    return {i: gpu_state[i]['models'] for i in range(gpu_num)}

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
