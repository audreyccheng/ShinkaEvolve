# EVOLVE-BLOCK-START
"""Model placement algorithm for minimizing maximum KV cache pressure across GPUs"""

import random
import time

GPU_MEM_SIZE = 80.0

def compute_model_placement(gpu_num, models):
    """
    Compute a model placement that minimizes the maximum KVPR across all GPUs.

    Algorithm:
    1. Multi-Heuristic Initialization: Binary search using multiple sorting strategies.
    2. Enhanced Local Search (VND): Move, Swap(1-1), Swap(2-1), Swap(1-2).
    3. Perturbation: Greedy Reinsertion.
    """
    start_time = time.time()

    model_data = []
    for i, m in enumerate(models):
        model_data.append({
            'model': m,
            'l': m.req_rate / m.slo,
            's': m.model_size,
            'id': i
        })

    def get_kvpr(l, s):
        if s >= GPU_MEM_SIZE - 1e-7: return 1e15
        return l / (GPU_MEM_SIZE - s)

    # --- Phase 1: Construction ---

    def solve_packing(target_k, strategy='linear', seed=None):
        capacity = target_k * GPU_MEM_SIZE
        items = list(model_data)

        # Strategies
        if strategy == 'linear':
            key_func = lambda x: x['l'] + target_k * x['s']
        elif strategy == 'size':
            key_func = lambda x: x['s']
        elif strategy == 'load':
            key_func = lambda x: x['l']
        elif strategy == 'density':
            key_func = lambda x: x['l'] / x['s'] if x['s'] > 0 else 0
        else:
            key_func = lambda x: x['l'] + target_k * x['s']

        if seed is not None:
            rng = random.Random(seed)
            # Add noise only to linear for randomization
            base_key = key_func
            key_func = lambda x: base_key(x) * rng.uniform(0.9, 1.1)

        items.sort(key=key_func, reverse=True)

        gpu_l = [0.0] * gpu_num
        gpu_s = [0.0] * gpu_num
        gpu_items = [[] for _ in range(gpu_num)]

        for item in items:
            w = item['l'] + target_k * item['s']
            best_idx = -1
            min_rem = float('inf')

            # Best Fit
            for i in range(gpu_num):
                if gpu_s[i] + item['s'] >= GPU_MEM_SIZE - 1e-6: continue

                curr_w = gpu_l[i] + target_k * gpu_s[i]
                if curr_w + w <= capacity + 1e-9:
                    rem = capacity - (curr_w + w)
                    if rem < min_rem:
                        min_rem = rem
                        best_idx = i

            if best_idx != -1:
                gpu_l[best_idx] += item['l']
                gpu_s[best_idx] += item['s']
                gpu_items[best_idx].append(item)
            else:
                return None
        return gpu_items

    # Binary Search
    low, high = 0.0, 1.0
    for _ in range(20):
        if solve_packing(high) is not None: break
        low, high = high, high * 2.0
    else: high = 1e9

    strategies = ['linear', 'size', 'load', 'density']

    # Coarse search
    for _ in range(20):
        mid = (low + high) / 2
        feasible = False
        for strat in strategies:
            if solve_packing(mid, strategy=strat) is not None:
                feasible = True
                break
        if feasible: high = mid
        else: low = mid

    base_k = high

    # Multi-start selection
    best_init_plc = None
    best_init_score = float('inf')

    # Try deterministic strategies at base_k
    candidates = []
    for strat in strategies:
        res = solve_packing(base_k, strategy=strat)
        if res: candidates.append(res)

    # Try randomized linear
    for seed in range(15):
        res = solve_packing(base_k * 1.001, strategy='linear', seed=seed)
        if res: candidates.append(res)

    for res in candidates:
        max_k = 0
        for g in range(gpu_num):
            l = sum(x['l'] for x in res[g])
            s = sum(x['s'] for x in res[g])
            k = get_kvpr(l, s)
            if k > max_k: max_k = k
        if max_k < best_init_score:
            best_init_score = max_k
            best_init_plc = res

    if best_init_plc is None:
        best_init_plc = solve_packing(1e9) # Fallback

    # --- Phase 2: Local Search ---
    plc = [list(p) for p in best_init_plc]

    # Maintain stats
    gpu_stats = []
    for g in range(gpu_num):
        l = sum(x['l'] for x in plc[g])
        s = sum(x['s'] for x in plc[g])
        gpu_stats.append({'l': l, 's': s})

    global_best_plc = [list(p) for p in plc]
    global_best_score = best_init_score

    def evaluate(stats):
        max_k = -1.0
        bottlenecks = []
        for g in range(gpu_num):
            k = get_kvpr(stats[g]['l'], stats[g]['s'])
            if k > max_k:
                max_k = k
                bottlenecks = [g]
            elif abs(k - max_k) < 1e-9:
                bottlenecks.append(g)
        return max_k, bottlenecks

    while time.time() - start_time < 0.8:
        # Steepest Descent
        while True:
            cur_max, bottlenecks = evaluate(gpu_stats)
            if cur_max < 1e-9: break

            best_move = None
            best_imp = 0.0

            # Sort non-bottleneck targets by KVPR to optimize search order (heuristic)
            targets = []
            for t in range(gpu_num):
                if t not in bottlenecks:
                    k_t = get_kvpr(gpu_stats[t]['l'], gpu_stats[t]['s'])
                    if k_t < cur_max - 1e-6:
                        targets.append((k_t, t))
            targets.sort(key=lambda x: x[0])
            sorted_targets = [t for _, t in targets]

            if not sorted_targets: break

            # Only check worst bottleneck to save time
            b_idx = bottlenecks[0]
            b_l = gpu_stats[b_idx]['l']
            b_s = gpu_stats[b_idx]['s']
            src_items = plc[b_idx]

            # 1. Move
            for i, item in enumerate(src_items):
                for t in sorted_targets:
                    if gpu_stats[t]['s'] + item['s'] >= GPU_MEM_SIZE: continue
                    nk_src = get_kvpr(b_l - item['l'], b_s - item['s'])
                    nk_tgt = get_kvpr(gpu_stats[t]['l'] + item['l'], gpu_stats[t]['s'] + item['s'])
                    new_global = max(nk_src, nk_tgt)
                    if new_global < cur_max - 1e-7:
                        imp = cur_max - new_global
                        if imp > best_imp:
                            best_imp = imp
                            best_move = ('move', b_idx, i, t)

            # 2. Swap 1-1
            for i, s_item in enumerate(src_items):
                for t in sorted_targets:
                    tgt_items = plc[t]
                    for j, t_item in enumerate(tgt_items):
                        if b_s - s_item['s'] + t_item['s'] >= GPU_MEM_SIZE: continue
                        if gpu_stats[t]['s'] - t_item['s'] + s_item['s'] >= GPU_MEM_SIZE: continue

                        nk_src = get_kvpr(b_l - s_item['l'] + t_item['l'], b_s - s_item['s'] + t_item['s'])
                        nk_tgt = get_kvpr(gpu_stats[t]['l'] - t_item['l'] + s_item['l'], gpu_stats[t]['s'] - t_item['s'] + s_item['s'])
                        new_global = max(nk_src, nk_tgt)
                        if new_global < cur_max - 1e-7:
                            imp = cur_max - new_global
                            if imp > best_imp:
                                best_imp = imp
                                best_move = ('swap11', b_idx, i, t, j)

            # 3. Swap 2-1
            if len(src_items) >= 2:
                pair_limit = 100
                checked = 0
                for i1 in range(len(src_items)):
                    if checked >= pair_limit: break
                    for i2 in range(i1+1, len(src_items)):
                        m1, m2 = src_items[i1], src_items[i2]
                        pl, ps = m1['l'] + m2['l'], m1['s'] + m2['s']
                        for t in sorted_targets:
                            for j, m3 in enumerate(plc[t]):
                                if b_s - ps + m3['s'] >= GPU_MEM_SIZE: continue
                                if gpu_stats[t]['s'] - m3['s'] + ps >= GPU_MEM_SIZE: continue
                                nk_src = get_kvpr(b_l - pl + m3['l'], b_s - ps + m3['s'])
                                nk_tgt = get_kvpr(gpu_stats[t]['l'] - m3['l'] + pl, gpu_stats[t]['s'] - m3['s'] + ps)
                                if max(nk_src, nk_tgt) < cur_max - 1e-7:
                                    imp = cur_max - max(nk_src, nk_tgt)
                                    if imp > best_imp:
                                        best_imp = imp
                                        best_move = ('swap21', b_idx, i1, i2, t, j)
                        checked += 1

            # 4. Swap 1-2 (1 from bottleneck, 2 from target)
            for i, m1 in enumerate(src_items):
                for t in sorted_targets:
                    tgt_items = plc[t]
                    if len(tgt_items) < 2: continue

                    pair_limit = 50
                    checked = 0
                    for j1 in range(len(tgt_items)):
                        if checked >= pair_limit: break
                        for j2 in range(j1+1, len(tgt_items)):
                            m2, m3 = tgt_items[j1], tgt_items[j2]
                            pl, ps = m2['l'] + m3['l'], m2['s'] + m3['s']

                            if b_s - m1['s'] + ps >= GPU_MEM_SIZE: continue
                            if gpu_stats[t]['s'] - ps + m1['s'] >= GPU_MEM_SIZE: continue

                            nk_src = get_kvpr(b_l - m1['l'] + pl, b_s - m1['s'] + ps)
                            nk_tgt = get_kvpr(gpu_stats[t]['l'] - pl + m1['l'], gpu_stats[t]['s'] - ps + m1['s'])

                            if max(nk_src, nk_tgt) < cur_max - 1e-7:
                                imp = cur_max - max(nk_src, nk_tgt)
                                if imp > best_imp:
                                    best_imp = imp
                                    best_move = ('swap12', b_idx, i, t, j1, j2)
                            checked += 1

            if best_move:
                op = best_move[0]
                if op == 'move':
                    _, b, i, t = best_move
                    item = plc[b].pop(i)
                    plc[t].append(item)
                elif op == 'swap11':
                    _, b, i, t, j = best_move
                    plc[b][i], plc[t][j] = plc[t][j], plc[b][i]
                elif op == 'swap21':
                    _, b, i1, i2, t, j = best_move
                    it2 = plc[b].pop(i2)
                    it1 = plc[b].pop(i1)
                    it3 = plc[t].pop(j)
                    plc[b].append(it3)
                    plc[t].extend([it1, it2])
                elif op == 'swap12':
                    _, b, i, t, j1, j2 = best_move
                    it3 = plc[t].pop(j2)
                    it2 = plc[t].pop(j1)
                    it1 = plc[b].pop(i)
                    plc[t].append(it1)
                    plc[b].extend([it2, it3])

                # Update stats fully to avoid errors
                b_idx = best_move[1]
                t_idx = best_move[3] if op == 'move' else best_move[3] if op == 'swap11' else best_move[4] if op == 'swap21' else best_move[3]

                gpu_stats[b_idx]['l'] = sum(x['l'] for x in plc[b_idx])
                gpu_stats[b_idx]['s'] = sum(x['s'] for x in plc[b_idx])
                gpu_stats[t_idx]['l'] = sum(x['l'] for x in plc[t_idx])
                gpu_stats[t_idx]['s'] = sum(x['s'] for x in plc[t_idx])

            else:
                break # Local optima

        # Check global best
        cur_max, bottlenecks = evaluate(gpu_stats)
        if cur_max < global_best_score:
            global_best_score = cur_max
            global_best_plc = [list(p) for p in plc]

        # Perturbation
        if not bottlenecks: break
        b = random.choice(bottlenecks)
        if not plc[b]: break

        # Remove 1 item
        idx = random.randrange(len(plc[b]))
        item = plc[b].pop(idx)
        gpu_stats[b]['l'] -= item['l']
        gpu_stats[b]['s'] -= item['s']

        # Insert into best available
        best_t = -1
        best_k = float('inf')

        cands = list(range(gpu_num))
        random.shuffle(cands)
        for t in cands:
            if t == b: continue
            if gpu_stats[t]['s'] + item['s'] < GPU_MEM_SIZE:
                k = get_kvpr(gpu_stats[t]['l'] + item['l'], gpu_stats[t]['s'] + item['s'])
                if k < best_k:
                    best_k = k
                    best_t = t

        if best_t != -1:
            plc[best_t].append(item)
            gpu_stats[best_t]['l'] += item['l']
            gpu_stats[best_t]['s'] += item['s']
        else:
            # Revert if failed (unlikely)
            plc[b].append(item)
            gpu_stats[b]['l'] += item['l']
            gpu_stats[b]['s'] += item['s']

    result = {}
    for i in range(gpu_num):
        result[i] = [x['model'] for x in global_best_plc[i]]
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