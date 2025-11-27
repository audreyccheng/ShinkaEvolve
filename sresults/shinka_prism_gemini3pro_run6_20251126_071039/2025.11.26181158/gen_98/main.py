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
    for _ in range(200):
        # Identify bottleneck GPU
        current_kvprs = [get_kvpr(gs['l'], gs['s']) for gs in gpu_states]
        max_kvpr = max(current_kvprs)
        max_gpu = current_kvprs.index(max_kvpr)

        if max_kvpr < 1e-9: break

        best_action = None
        best_new_max = max_kvpr

        # Helper to check if a move/swap improves global state
        def check_improvement(new_src_l, new_src_s, new_tgt_l, new_tgt_s, action):
            nonlocal best_new_max, best_action
            if new_src_s >= GPU_MEM_SIZE or new_tgt_s >= GPU_MEM_SIZE:
                return
            ks = get_kvpr(new_src_l, new_src_s)
            kt = get_kvpr(new_tgt_l, new_tgt_s)
            local_max = max(ks, kt)
            if local_max < best_new_max - 1e-7:
                best_new_max = local_max
                best_action = action

        src = gpu_states[max_gpu]
        src_data = [(m.req_rate/m.slo, m.model_size) for m in src['models']]
        n_src = len(src_data)

        for t_gpu in range(gpu_num):
            if t_gpu == max_gpu: continue
            if current_kvprs[t_gpu] >= max_kvpr: continue # Optimization

            tgt = gpu_states[t_gpu]
            tgt_data = [(m.req_rate/m.slo, m.model_size) for m in tgt['models']]
            n_tgt = len(tgt_data)

            # 1. Move: src -> tgt
            for i in range(n_src):
                l_i, s_i = src_data[i]
                check_improvement(
                    src['l'] - l_i, src['s'] - s_i,
                    tgt['l'] + l_i, tgt['s'] + s_i,
                    ('move', i, t_gpu)
                )

            # 2. Swap 1-1: src[i] <-> tgt[j]
            for i in range(n_src):
                l_i, s_i = src_data[i]
                for j in range(n_tgt):
                    l_j, s_j = tgt_data[j]
                    check_improvement(
                        src['l'] - l_i + l_j, src['s'] - s_i + s_j,
                        tgt['l'] - l_j + l_i, tgt['s'] - s_j + s_i,
                        ('swap11', i, t_gpu, j)
                    )

            # 3. Swap 2-1: src[i1, i2] <-> tgt[j]
            # Moving two small items out of bottleneck, taking one large item?
            # Or just moving mass out.
            for i1 in range(n_src):
                for i2 in range(i1 + 1, n_src):
                    l_out = src_data[i1][0] + src_data[i2][0]
                    s_out = src_data[i1][1] + src_data[i2][1]
                    for j in range(n_tgt):
                        l_j, s_j = tgt_data[j]
                        check_improvement(
                            src['l'] - l_out + l_j, src['s'] - s_out + s_j,
                            tgt['l'] - l_j + l_out, tgt['s'] - s_j + s_out,
                            ('swap21', i1, i2, t_gpu, j)
                        )

        # Apply best action
        if best_action:
            if best_action[0] == 'move':
                _, i, t_gpu = best_action
                m = gpu_states[max_gpu]['models'].pop(i)
                gpu_states[t_gpu]['models'].append(m)

                # Update stats manually to avoid drift/recalc
                l_m, s_m = m.req_rate/m.slo, m.model_size
                gpu_states[max_gpu]['l'] -= l_m; gpu_states[max_gpu]['s'] -= s_m
                gpu_states[t_gpu]['l'] += l_m; gpu_states[t_gpu]['s'] += s_m

            elif best_action[0] == 'swap11':
                _, i, t_gpu, j = best_action
                m_src = gpu_states[max_gpu]['models'][i]
                m_tgt = gpu_states[t_gpu]['models'][j]

                gpu_states[max_gpu]['models'][i] = m_tgt
                gpu_states[t_gpu]['models'][j] = m_src

                l_src, s_src = m_src.req_rate/m_src.slo, m_src.model_size
                l_tgt, s_tgt = m_tgt.req_rate/m_tgt.slo, m_tgt.model_size

                diff_l = l_tgt - l_src
                diff_s = s_tgt - s_src

                gpu_states[max_gpu]['l'] += diff_l; gpu_states[max_gpu]['s'] += diff_s
                gpu_states[t_gpu]['l'] -= diff_l; gpu_states[t_gpu]['s'] -= diff_s

            elif best_action[0] == 'swap21':
                _, i1, i2, t_gpu, j = best_action
                # Pop indices carefully (larger first)
                m1 = gpu_states[max_gpu]['models'].pop(i2)
                m2 = gpu_states[max_gpu]['models'].pop(i1)
                m_tgt = gpu_states[t_gpu]['models'][j]

                gpu_states[max_gpu]['models'].append(m_tgt)
                gpu_states[t_gpu]['models'][j] = m1
                gpu_states[t_gpu]['models'].append(m2)

                l_out = (m1.req_rate/m1.slo) + (m2.req_rate/m2.slo)
                s_out = m1.model_size + m2.model_size
                l_in = m_tgt.req_rate/m_tgt.slo
                s_in = m_tgt.model_size

                gpu_states[max_gpu]['l'] += l_in - l_out
                gpu_states[max_gpu]['s'] += s_in - s_out
                gpu_states[t_gpu]['l'] += l_out - l_in
                gpu_states[t_gpu]['s'] += s_out - s_in
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