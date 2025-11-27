# EVOLVE-BLOCK-START
"""Model placement algorithm for minimizing maximum KV cache pressure across GPUs"""

import random
import time

GPU_MEM_SIZE = 80  # GB

def compute_model_placement(gpu_num, models):
    """
    Compute a model placement that minimizes the maximum KVPR across all GPUs.

    Algorithm: Multi-Start Perturbation-Based Iterated Local Search.

    1. Initialization: Randomized Bin Packing based on Linearized Weight (L + K*S).
       Runs multiple independent restarts to explore diverse starting basins.
    2. Local Search: Steepest Descent Hill Climbing.
       - Moves: Transfer, Swap(1-1), Swap(2-1), Swap(2-2).
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
        """
        Generates a placement.
        If randomize=False, tries multiple deterministic heuristics.
        If randomize=True, adds noise to the linearized weight heuristic.
        """
        capacity = target_k * GPU_MEM_SIZE

        def get_strategies():
            if randomize:
                # Randomized Linearized
                items = []
                for d in model_data:
                    w = (d['l'] + target_k * d['s']) * random.uniform(0.9, 1.1)
                    items.append((w, d))
                items.sort(key=lambda x: x[0], reverse=True)
                return [[x[1] for x in items]]
            else:
                # Deterministic Heuristics
                strats = []
                # 1. Linearized Descending
                strats.append(sorted(model_data, key=lambda x: x['l'] + target_k * x['s'], reverse=True))
                # 2. Size Descending
                strats.append(sorted(model_data, key=lambda x: x['s'], reverse=True))
                # 3. Load Descending
                strats.append(sorted(model_data, key=lambda x: x['l'], reverse=True))
                return strats

        for items in get_strategies():
            gpu_l = [0.0] * gpu_num
            gpu_s = [0.0] * gpu_num
            gpu_models = [[] for _ in range(gpu_num)]
            indices = list(range(gpu_num))
            feasible = True

            for item in items:
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
                    feasible = False
                    break

            if feasible:
                return gpu_models
        return None

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

    # Run restarts. Since Swap22 is expensive, reduce restarts if needed, but 4-5 is usually fine for 1s limit.
    for restart_idx in range(4):
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

                # 4. Swap 1-2 (1 from Bottleneck, 2 from Target)
                if n_tgt >= 2:
                    for i in range(n_src):
                        m1 = src_models[i]
                        m1l, m1s = m1.req_rate/m1.slo, m1.model_size

                        for j1 in range(n_tgt):
                            for j2 in range(j1 + 1, n_tgt):
                                m2, m3 = tgt_models[j1], tgt_models[j2]
                                pl_tgt = (m2.req_rate/m2.slo) + (m3.req_rate/m3.slo)
                                ps_tgt = m2.model_size + m3.model_size

                                imp = check_update(src_l - m1l + pl_tgt, src_s - m1s + ps_tgt,
                                                 tgt_l - pl_tgt + m1l, tgt_s - ps_tgt + m1s)
                                if imp > best_imp:
                                    best_imp = imp
                                    best_move = ('swap12', bottleneck, i, t, j1, j2)

                # 5. Swap 2-2 (2 from Bottleneck, 2 from Target)
                if n_src >= 2 and n_tgt >= 2:
                    for i1 in range(n_src):
                        for i2 in range(i1 + 1, n_src):
                            m1, m2 = src_models[i1], src_models[i2]
                            pl_src = (m1.req_rate/m1.slo) + (m2.req_rate/m2.slo)
                            ps_src = m1.model_size + m2.model_size

                            for j1 in range(n_tgt):
                                for j2 in range(j1 + 1, n_tgt):
                                    m3, m4 = tgt_models[j1], tgt_models[j2]
                                    pl_tgt = (m3.req_rate/m3.slo) + (m4.req_rate/m4.slo)
                                    ps_tgt = m3.model_size + m4.model_size

                                    imp = check_update(src_l - pl_src + pl_tgt, src_s - ps_src + ps_tgt,
                                                     tgt_l - pl_tgt + pl_src, tgt_s - ps_tgt + ps_src)
                                    if imp > best_imp:
                                        best_imp = imp
                                        best_move = ('swap22', bottleneck, i1, i2, t, j1, j2)

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
                    # Pop larger index first
                    m2 = current_plc[b].pop(i2)
                    m1 = current_plc[b].pop(i1)
                    m3 = current_plc[t].pop(j)
                    current_plc[b].append(m3)
                    current_plc[t].append(m1)
                    current_plc[t].append(m2)

                    l_vec[b] = sum(m.req_rate/m.slo for m in current_plc[b])
                    s_vec[b] = sum(m.model_size for m in current_plc[b])
                    l_vec[t] = sum(m.req_rate/m.slo for m in current_plc[t])
                    s_vec[t] = sum(m.model_size for m in current_plc[t])

                elif mtype == 'swap12':
                    _, b, i, t, j1, j2 = best_move
                    # Pop larger indices first from target
                    m3 = current_plc[t].pop(j2)
                    m2 = current_plc[t].pop(j1)
                    m1 = current_plc[b].pop(i)
                    current_plc[t].append(m1)
                    current_plc[b].append(m2)
                    current_plc[b].append(m3)

                    l_vec[b] = sum(m.req_rate/m.slo for m in current_plc[b])
                    s_vec[b] = sum(m.model_size for m in current_plc[b])
                    l_vec[t] = sum(m.req_rate/m.slo for m in current_plc[t])
                    s_vec[t] = sum(m.model_size for m in current_plc[t])

                elif mtype == 'swap22':
                    _, b, i1, i2, t, j1, j2 = best_move
                    # Pop larger indices first from both lists
                    m_b2 = current_plc[b].pop(i2)
                    m_b1 = current_plc[b].pop(i1)
                    m_t2 = current_plc[t].pop(j2)
                    m_t1 = current_plc[t].pop(j1)

                    current_plc[b].append(m_t1)
                    current_plc[b].append(m_t2)
                    current_plc[t].append(m_b1)
                    current_plc[t].append(m_b2)

                    l_vec[b] = sum(m.req_rate/m.slo for m in current_plc[b])
                    s_vec[b] = sum(m.model_size for m in current_plc[b])
                    l_vec[t] = sum(m.req_rate/m.slo for m in current_plc[t])
                    s_vec[t] = sum(m.model_size for m in current_plc[t])

                cur_max_k = max(get_kvpr(l_vec[g], s_vec[g]) for g in range(gpu_num))
                no_imp_iter = 0
            else:
                # Perturbation
                if no_imp_iter > 3: break

                # Robust Perturbation: Move a random model from bottleneck to the
                # BEST feasible GPU (lowest KVPR), not just the first one found.
                candidates = [] # (kvpr, gpu_idx)
                for g in range(gpu_num):
                    if g == bottleneck: continue
                    candidates.append((get_kvpr(l_vec[g], s_vec[g]), g))
                candidates.sort(key=lambda x: x[0])

                moved = False
                if current_plc[bottleneck]:
                    # Randomize which model to move to avoid cycles
                    indices = list(range(len(current_plc[bottleneck])))
                    random.shuffle(indices)

                    for idx in indices:
                        m = current_plc[bottleneck][idx]
                        # Try to move to the best possible candidate
                        for _, g in candidates:
                            if s_vec[g] + m.model_size < GPU_MEM_SIZE:
                                # Execute perturbation
                                current_plc[bottleneck].pop(idx)
                                current_plc[g].append(m)

                                # Update stats
                                l_vec[bottleneck] -= m.req_rate/m.slo
                                s_vec[bottleneck] -= m.model_size
                                l_vec[g] += m.req_rate/m.slo
                                s_vec[g] += m.model_size

                                moved = True
                                break
                        if moved: break

                if not moved: break
                cur_max_k = max(get_kvpr(l_vec[g], s_vec[g]) for g in range(gpu_num))
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