# EVOLVE-BLOCK-START
"""Model placement algorithm for minimizing maximum KV cache pressure across GPUs"""

import random

GPU_MEM_SIZE = 80  # GB

def compute_model_placement(gpu_num, models):
    """
    Compute a model placement that minimizes the maximum KVPR across all GPUs.

    Algorithm: Multi-Start Perturbation-Based Iterated Local Search.

    1. Initialization: Randomized Bin Packing based on Linearized Weight (L + K*S).
       Runs multiple independent restarts to explore diverse starting basins.
    2. Local Search: Steepest Descent Hill Climbing.
       - Moves: Transfer, Swap(1-1), Swap(2-1).
       - Selection: Evaluates ALL valid moves from the bottleneck GPU and selects
         the one yielding the best immediate reduction in KVPR.
    3. Perturbation: targeted 'kick' that moves a model from the bottleneck GPU
       to the least-loaded feasible GPU to escape local optima while maintaining balance.
    """

    # Pre-calculate model properties
    model_data = []
    for m in models:
        model_data.append({
            'model': m,
            'l': m.req_rate / m.slo,
            's': m.model_size
        })

    def solve_packing(target_k, randomize=False):
        """Generates a placement. Randomize=True adds noise to weights for diversity."""
        capacity = target_k * GPU_MEM_SIZE
        items = []
        for d in model_data:
            w = d['l'] + target_k * d['s']
            if randomize:
                w *= random.uniform(0.9, 1.1)
            items.append((w, d))

        items.sort(key=lambda x: x[0], reverse=True)

        gpu_l = [0.0] * gpu_num
        gpu_s = [0.0] * gpu_num
        gpu_models = [[] for _ in range(gpu_num)]

        # Helper indices for random tie-breaking
        indices = list(range(gpu_num))

        for _, item in items:
            best_idx = -1
            min_rem = float('inf')
            w_item = item['l'] + target_k * item['s']

            if randomize: random.shuffle(indices)

            for i in indices:
                if gpu_s[i] + item['s'] >= GPU_MEM_SIZE - 1e-6: continue

                curr_w = gpu_l[i] + target_k * gpu_s[i]
                if curr_w + w_item <= capacity + 1e-9:
                    rem = capacity - (curr_w + w_item)
                    if rem < min_rem:
                        min_rem = rem
                        best_idx = i

            if best_idx != -1:
                gpu_l[best_idx] += item['l']
                gpu_s[best_idx] += item['s']
                gpu_models[best_idx].append(item['model'])
            else:
                return None
        return gpu_models

    def get_kvpr(l, s):
        if s >= GPU_MEM_SIZE - 1e-7: return 1e15
        return l / (GPU_MEM_SIZE - s)

    # --- Phase 1: Determine Baseline K ---
    low, high = 0.0, 1.0
    for _ in range(20):
        if solve_packing(high) is not None: break
        low, high = high, high * 2.0
    else: high = 1e9

    for _ in range(20):
        mid = (low + high) / 2
        if solve_packing(mid) is not None: high = mid
        else: low = mid
    base_k = high

    # --- Phase 2: Multi-Start ILS ---
    best_global_plc = None
    best_global_score = float('inf')

    # Run 5 restarts (Time permitting, this is very fast)
    for restart_idx in range(5):
        # Generate initial solution (Randomize subsequent starts)
        init_plc_list = solve_packing(base_k, randomize=(restart_idx > 0))
        if init_plc_list is None:
            # Fallback if base_k is too tight for random packing
            init_plc_list = solve_packing(base_k * 1.5, randomize=False)
            if init_plc_list is None: continue

        current_plc = {i: list(init_plc_list[i]) for i in range(gpu_num)}

        # Calculate stats
        l_vec = [sum(m.req_rate/m.slo for m in current_plc[g]) for g in range(gpu_num)]
        s_vec = [sum(m.model_size for m in current_plc[g]) for g in range(gpu_num)]

        cur_max_k = max(get_kvpr(l_vec[g], s_vec[g]) for g in range(gpu_num))

        no_imp_iter = 0
        iter_limit = 100

        for _ in range(iter_limit):
            if cur_max_k < 1e-9: break

            # Find bottleneck
            bottleneck = -1
            max_val = -1.0
            for g in range(gpu_num):
                val = get_kvpr(l_vec[g], s_vec[g])
                if val > max_val:
                    max_val = val
                    bottleneck = g

            # Steepest Descent: Find BEST move from bottleneck
            best_move = None
            best_imp = 0.0

            src_l = l_vec[bottleneck]
            src_s = s_vec[bottleneck]

            # Helper to check validity and improvement
            def check_update(new_src_l, new_src_s, new_tgt_l, new_tgt_s):
                if new_src_s >= GPU_MEM_SIZE or new_tgt_s >= GPU_MEM_SIZE: return -1.0
                nk_src = get_kvpr(new_src_l, new_src_s)
                nk_tgt = get_kvpr(new_tgt_l, new_tgt_s)
                new_local_max = max(nk_src, nk_tgt)
                if new_local_max < cur_max_k - 1e-7:
                    return cur_max_k - new_local_max
                return -1.0

            src_models = current_plc[bottleneck]
            n_src = len(src_models)

            # Iterate targets
            for t in range(gpu_num):
                if t == bottleneck: continue
                # Pruning: if target is already heavily loaded, skip
                if get_kvpr(l_vec[t], s_vec[t]) > cur_max_k * 0.95: continue

                tgt_models = current_plc[t]
                n_tgt = len(tgt_models)
                tgt_l = l_vec[t]
                tgt_s = s_vec[t]

                # 1. Move
                for i in range(n_src):
                    m = src_models[i]
                    ml, ms = m.req_rate/m.slo, m.model_size
                    imp = check_update(src_l - ml, src_s - ms, tgt_l + ml, tgt_s + ms)
                    if imp > best_imp:
                        best_imp = imp
                        best_move = ('move', bottleneck, i, t)

                # 2. Swap 1-1
                for i in range(n_src):
                    m1 = src_models[i]
                    m1l, m1s = m1.req_rate/m1.slo, m1.model_size
                    for j in range(n_tgt):
                        m2 = tgt_models[j]
                        m2l, m2s = m2.req_rate/m2.slo, m2.model_size
                        imp = check_update(src_l - m1l + m2l, src_s - m1s + m2s,
                                         tgt_l - m2l + m1l, tgt_s - m2s + m1s)
                        if imp > best_imp:
                            best_imp = imp
                            best_move = ('swap11', bottleneck, i, t, j)

                # 3. Swap 2-1 (2 from Bottleneck, 1 from Target)
                if n_src >= 2:
                    for i1 in range(n_src):
                        for i2 in range(i1 + 1, n_src):
                            m1, m2 = src_models[i1], src_models[i2]
                            pl = (m1.req_rate/m1.slo) + (m2.req_rate/m2.slo)
                            ps = m1.model_size + m2.model_size

                            for j in range(n_tgt):
                                m3 = tgt_models[j]
                                m3l, m3s = m3.req_rate/m3.slo, m3.model_size
                                imp = check_update(src_l - pl + m3l, src_s - ps + m3s,
                                                 tgt_l - m3l + pl, tgt_s - m3s + ps)
                                if imp > best_imp:
                                    best_imp = imp
                                    best_move = ('swap21', bottleneck, i1, i2, t, j)

            # Execute best move
            if best_move:
                mtype = best_move[0]
                if mtype == 'move':
                    _, b, i, t = best_move
                    m = current_plc[b].pop(i)
                    current_plc[t].append(m)

                    ml, ms = m.req_rate/m.slo, m.model_size
                    l_vec[b] -= ml; s_vec[b] -= ms
                    l_vec[t] += ml; s_vec[t] += ms

                elif mtype == 'swap11':
                    _, b, i, t, j = best_move
                    m1 = current_plc[b][i]
                    m2 = current_plc[t][j]
                    current_plc[b][i] = m2
                    current_plc[t][j] = m1

                    diff_l = (m2.req_rate/m2.slo) - (m1.req_rate/m1.slo)
                    diff_s = m2.model_size - m1.model_size
                    l_vec[b] += diff_l; s_vec[b] += diff_s
                    l_vec[t] -= diff_l; s_vec[t] -= diff_s

                elif mtype == 'swap21':
                    _, b, i1, i2, t, j = best_move
                    # Pop carefully: larger index first
                    m2 = current_plc[b].pop(i2)
                    m1 = current_plc[b].pop(i1)
                    m3 = current_plc[t].pop(j)
                    current_plc[b].append(m3)
                    current_plc[t].append(m1)
                    current_plc[t].append(m2)

                    # Full recalc for safety on multi-item moves
                    l_vec[b] = sum(m.req_rate/m.slo for m in current_plc[b])
                    s_vec[b] = sum(m.model_size for m in current_plc[b])
                    l_vec[t] = sum(m.req_rate/m.slo for m in current_plc[t])
                    s_vec[t] = sum(m.model_size for m in current_plc[t])

                cur_max_k = max(get_kvpr(l_vec[g], s_vec[g]) for g in range(gpu_num))
                no_imp_iter = 0
            else:
                # Perturbation
                if no_imp_iter > 3: break

                # Move a model from bottleneck to LEAST LOADED feasible GPU
                min_k = float('inf')
                min_g = -1
                for g in range(gpu_num):
                    if g == bottleneck: continue
                    k_val = get_kvpr(l_vec[g], s_vec[g])
                    if k_val < min_k:
                        min_k = k_val
                        min_g = g

                moved = False
                if min_g != -1 and current_plc[bottleneck]:
                    # Try first model (simple heuristic)
                    m = current_plc[bottleneck][0]
                    if s_vec[min_g] + m.model_size < GPU_MEM_SIZE:
                        current_plc[bottleneck].pop(0)
                        current_plc[min_g].append(m)

                        l_vec[bottleneck] = sum(m.req_rate/m.slo for m in current_plc[bottleneck])
                        s_vec[bottleneck] = sum(m.model_size for m in current_plc[bottleneck])
                        l_vec[min_g] = sum(m.req_rate/m.slo for m in current_plc[min_g])
                        s_vec[min_g] = sum(m.model_size for m in current_plc[min_g])

                        cur_max_k = max(get_kvpr(l_vec[g], s_vec[g]) for g in range(gpu_num))
                        moved = True

                if not moved: break
                no_imp_iter += 1

        if cur_max_k < best_global_score:
            best_global_score = cur_max_k
            best_global_plc = {i: list(current_plc[i]) for i in range(gpu_num)}

    # Fallback guarantees
    if best_global_plc is None:
        init = solve_packing(base_k)
        if init is None: init = solve_packing(1e9)
        if init is None: raise ValueError("No valid placement found.")
        best_global_plc = {i: init[i] for i in range(gpu_num)}

    return best_global_plc
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