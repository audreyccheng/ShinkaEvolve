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

        # Strategy 4: Sort by Density (Load/Size) Descending
        # Good for packing "efficient" but heavy items first
        items_4 = sorted(model_data, key=lambda x: x['l']/x['s'], reverse=True)
        res = try_best_fit(items_4)
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

    # --- Phase 2: Iterated Local Search ---
    import random

    def get_kvpr(l, s):
        if s >= GPU_MEM_SIZE - 1e-6: return 1e15
        return l / (GPU_MEM_SIZE - s)

    def run_local_search(current_placement):
        # Deepish copy for state tracking
        gpu_states = []
        for g in range(gpu_num):
            models_g = list(current_placement[g])
            s_g = sum(m.model_size for m in models_g)
            l_g = sum(m.req_rate / m.slo for m in models_g)
            gpu_states.append({'models': models_g, 's': s_g, 'l': l_g})

        for _ in range(50):
            current_kvprs = [get_kvpr(gs['l'], gs['s']) for gs in gpu_states]
            max_kvpr = max(current_kvprs)

            if max_kvpr < 1e-9: break

            # Find all GPUs near the max to allow moving from any bottleneck
            candidates = [g for g, v in enumerate(current_kvprs) if v >= max_kvpr - 1e-6]

            best_action = None
            best_new_max = max_kvpr

            for max_gpu in candidates:
                src_models = gpu_states[max_gpu]['models']

                # 1. Move
                for s_idx, model in enumerate(src_models):
                    m_l = model.req_rate / model.slo
                    m_s = model.model_size
                    for t_gpu in range(gpu_num):
                        if t_gpu == max_gpu: continue
                        if gpu_states[t_gpu]['s'] + m_s >= GPU_MEM_SIZE: continue

                        ks = get_kvpr(gpu_states[max_gpu]['l'] - m_l, gpu_states[max_gpu]['s'] - m_s)
                        kt = get_kvpr(gpu_states[t_gpu]['l'] + m_l, gpu_states[t_gpu]['s'] + m_s)

                        local_max = max(ks, kt)
                        if local_max < best_new_max - 1e-6:
                            best_new_max = local_max
                            best_action = ('move', max_gpu, s_idx, t_gpu)

                # 2. Swap
                for s_idx, s_model in enumerate(src_models):
                    s_l = s_model.req_rate / s_model.slo
                    s_s = s_model.model_size
                    for t_gpu in range(gpu_num):
                        if t_gpu == max_gpu: continue
                        if current_kvprs[t_gpu] >= max_kvpr: continue

                        tgt_models = gpu_states[t_gpu]['models']
                        for t_idx, t_model in enumerate(tgt_models):
                            t_l = t_model.req_rate / t_model.slo
                            t_s = t_model.model_size

                            ns_s = gpu_states[max_gpu]['s'] - s_s + t_s
                            nt_s = gpu_states[t_gpu]['s'] - t_s + s_s
                            if ns_s >= GPU_MEM_SIZE or nt_s >= GPU_MEM_SIZE: continue

                            ks = get_kvpr(gpu_states[max_gpu]['l'] - s_l + t_l, ns_s)
                            kt = get_kvpr(gpu_states[t_gpu]['l'] - t_l + s_l, nt_s)

                            local_max = max(ks, kt)
                            if local_max < best_new_max - 1e-6:
                                best_new_max = local_max
                                best_action = ('swap', max_gpu, s_idx, t_gpu, t_idx)

            if best_action:
                if best_action[0] == 'move':
                    mg, si, tg = best_action[1:]
                    m = gpu_states[mg]['models'].pop(si)
                    gpu_states[mg]['l'] -= m.req_rate/m.slo
                    gpu_states[mg]['s'] -= m.model_size
                    gpu_states[tg]['models'].append(m)
                    gpu_states[tg]['l'] += m.req_rate/m.slo
                    gpu_states[tg]['s'] += m.model_size
                elif best_action[0] == 'swap':
                    mg, si, tg, ti = best_action[1:]
                    m1 = gpu_states[mg]['models'][si]
                    m2 = gpu_states[tg]['models'][ti]
                    gpu_states[mg]['models'][si] = m2
                    gpu_states[tg]['models'][ti] = m1

                    diff_l = (m2.req_rate/m2.slo) - (m1.req_rate/m1.slo)
                    diff_s = m2.model_size - m1.model_size
                    gpu_states[mg]['l'] += diff_l
                    gpu_states[mg]['s'] += diff_s
                    gpu_states[tg]['l'] -= diff_l
                    gpu_states[tg]['s'] -= diff_s
            else:
                break

        return {g: gpu_states[g]['models'] for g in range(gpu_num)}

    def calc_score(plc):
        max_k = 0.0
        for g in range(gpu_num):
            sl = sum(m.req_rate/m.slo for m in plc[g])
            ss = sum(m.model_size for m in plc[g])
            if ss >= GPU_MEM_SIZE: return float('inf')
            k = sl / (GPU_MEM_SIZE - ss)
            if k > max_k: max_k = k
        return max_k

    # ILS Loop
    best_solution = placement
    best_score = calc_score(best_solution)

    current_solution = best_solution

    # 5 iterations of ILS
    for iteration in range(5):
        # Local Search
        current_solution = run_local_search(current_solution)
        current_score = calc_score(current_solution)

        if current_score < best_score:
            best_score = current_score
            best_solution = current_solution

        # Perturbation
        # Clone current best to perturb
        next_start = {g: list(best_solution[g]) for g in range(gpu_num)}

        # Identify bottleneck in best solution
        kvprs = []
        for g in range(gpu_num):
            sl = sum(m.req_rate/m.slo for m in next_start[g])
            ss = sum(m.model_size for m in next_start[g])
            if ss >= GPU_MEM_SIZE: kvprs.append(float('inf'))
            else: kvprs.append(sl/(GPU_MEM_SIZE - ss))

        max_k = max(kvprs)
        candidates = [g for g, v in enumerate(kvprs) if v >= max_k - 1e-4]

        if not candidates or max_k == 0: break

        # Move random model from bottleneck to random target
        perturb_gpu = random.choice(candidates)
        if next_start[perturb_gpu]:
             # Try up to 3 times to find a valid move
            for _ in range(3):
                m_idx = random.randrange(len(next_start[perturb_gpu]))
                model = next_start[perturb_gpu][m_idx]

                targets = list(range(gpu_num))
                random.shuffle(targets)
                moved = False
                for t in targets:
                    if t == perturb_gpu: continue
                    tgt_s = sum(m.model_size for m in next_start[t])
                    if tgt_s + model.model_size < GPU_MEM_SIZE:
                        next_start[perturb_gpu].pop(m_idx)
                        next_start[t].append(model)
                        moved = True
                        break
                if moved: break

        current_solution = next_start

    return best_solution

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