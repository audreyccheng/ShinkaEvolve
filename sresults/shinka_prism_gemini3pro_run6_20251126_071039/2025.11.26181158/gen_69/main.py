# EVOLVE-BLOCK-START
"""Model placement algorithm for minimizing maximum KV cache pressure across GPUs"""

import random
import time

GPU_MEM_SIZE = 80.0

def compute_model_placement(gpu_num, models):
    """
    Compute a model placement that minimizes the maximum KVPR across all GPUs.

    Algorithm:
    1. Multi-Start Randomized Construction:
       - Uses Binary Search to find a baseline 'target K'.
       - Generates multiple initial solutions by adding random noise to sorting weights
         during Bin Packing. Selects the best start.
    2. Iterated Local Search (ILS):
       - Uses Steepest Descent (Best Improvement) on bottleneck GPUs.
       - Neighborhoods: Move, Swap(1-1), Swap(2-1).
       - Perturbation: Ruins & Recreate (LNS) on bottlenecks when stuck.
    """
    start_time = time.time()

    # Pre-process models
    model_data = []
    for i, m in enumerate(models):
        model_data.append({
            'model': m,
            'l': m.req_rate / m.slo,
            's': m.model_size
        })

    def get_kvpr(l, s):
        if s >= GPU_MEM_SIZE - 1e-7: return 1e15
        return l / (GPU_MEM_SIZE - s)

    # --- Phase 1: Randomized Binary Search Construction ---

    def solve_packing(target_k, random_seed=None):
        capacity = target_k * GPU_MEM_SIZE
        items = list(model_data)

        # Sort Key: l + K*s with optional noise
        if random_seed is not None:
            rng = random.Random(random_seed)
            # Add noise to weight: w * random_factor
            items.sort(key=lambda x: (x['l'] + target_k * x['s']) * rng.uniform(0.9, 1.1), reverse=True)
        else:
            items.sort(key=lambda x: x['l'] + target_k * x['s'], reverse=True)

        gpu_l = [0.0] * gpu_num
        gpu_s = [0.0] * gpu_num
        gpu_models = [[] for _ in range(gpu_num)]

        for item in items:
            w = item['l'] + target_k * item['s']
            best_idx = -1
            min_rem = float('inf')

            # Best Fit logic
            for i in range(gpu_num):
                if gpu_s[i] + item['s'] >= GPU_MEM_SIZE - 1e-6:
                    continue

                curr_w = gpu_l[i] + target_k * gpu_s[i]
                if curr_w + w <= capacity + 1e-9:
                    rem = capacity - (curr_w + w)
                    if rem < min_rem:
                        min_rem = rem
                        best_idx = i

            if best_idx != -1:
                gpu_l[best_idx] += item['l']
                gpu_s[best_idx] += item['s']
                gpu_models[best_idx].append(item)
            else:
                return None
        return gpu_models

    # 1. Deterministic Binary Search for Baseline K
    low, high = 0.0, 1.0
    for _ in range(30):
        mid = (low + high) / 2
        if solve_packing(mid) is not None: high = mid
        else: low = mid
    base_k = high

    # 2. Multi-Start Initialization
    best_init_plc = None
    best_init_max_kvpr = float('inf')

    # Run multiple randomized initializations
    # Relax K slightly for randomized packing
    search_k = base_k * 1.001

    seeds = [None] + list(range(10))
    for seed in seeds:
        if time.time() - start_time > 0.3: break

        res = solve_packing(search_k, random_seed=seed)
        if res:
            max_k = max(get_kvpr(sum(x['l'] for x in res[g]), sum(x['s'] for x in res[g])) for g in range(gpu_num))
            if max_k < best_init_max_kvpr:
                best_init_max_kvpr = max_k
                best_init_plc = res

    if best_init_plc is None:
        best_init_plc = solve_packing(1e9)
        if best_init_plc is None: best_init_plc = [[] for _ in range(gpu_num)]
        best_init_max_kvpr = 1e9

    # --- Phase 2: Iterated Local Search (VND) ---
    plc = [list(best_init_plc[g]) for g in range(gpu_num)]

    gpu_stats = []
    for g in range(gpu_num):
        gpu_stats.append({
            'l': sum(m['l'] for m in plc[g]),
            's': sum(m['s'] for m in plc[g])
        })

    best_global_plc = [list(p) for p in plc]
    best_global_score = best_init_max_kvpr

    def evaluate_max_k():
        max_k = -1.0
        bottlenecks = []
        for g in range(gpu_num):
            k = get_kvpr(gpu_stats[g]['l'], gpu_stats[g]['s'])
            if k > max_k:
                max_k = k
                bottlenecks = [g]
            elif abs(k - max_k) < 1e-9:
                bottlenecks.append(g)
        return max_k, bottlenecks

    while time.time() - start_time < 0.95:
        # Variable Neighborhood Descent
        # Priorities: Move > Swap11 > Swap21
        # Uses First Improvement with Target Sorting

        improved = True
        while improved:
            improved = False
            cur_max, bottlenecks = evaluate_max_k()
            if cur_max < 1e-9: break

            # Identify valid targets (KVPR < cur_max) sorted by KVPR asc
            targets = []
            for g in range(gpu_num):
                if g not in bottlenecks:
                    k = get_kvpr(gpu_stats[g]['l'], gpu_stats[g]['s'])
                    if k < cur_max - 1e-7:
                        targets.append((k, g))
            targets.sort(key=lambda x: x[0])
            target_indices = [x[1] for x in targets]

            if not target_indices: break

            # Process primary bottleneck
            b = bottlenecks[0]
            b_l, b_s = gpu_stats[b]['l'], gpu_stats[b]['s']
            b_items = plc[b]

            # Helper to update stats
            def update_stats(g, dl, ds):
                gpu_stats[g]['l'] += dl
                gpu_stats[g]['s'] += ds

            # 1. MOVE
            for i, item in enumerate(b_items):
                for t in target_indices:
                    t_l, t_s = gpu_stats[t]['l'], gpu_stats[t]['s']
                    if t_s + item['s'] >= GPU_MEM_SIZE: continue

                    nk_src = get_kvpr(b_l - item['l'], b_s - item['s'])
                    nk_tgt = get_kvpr(t_l + item['l'], t_s + item['s'])

                    if max(nk_src, nk_tgt) < cur_max - 1e-7:
                        # Apply Move
                        item = plc[b].pop(i)
                        plc[t].append(item)
                        update_stats(b, -item['l'], -item['s'])
                        update_stats(t, item['l'], item['s'])
                        improved = True
                        break
                if improved: break
            if improved: continue

            # 2. SWAP 1-1
            for i, item1 in enumerate(b_items):
                for t in target_indices:
                    t_items = plc[t]
                    t_l, t_s = gpu_stats[t]['l'], gpu_stats[t]['s']
                    for j, item2 in enumerate(t_items):
                        if b_s - item1['s'] + item2['s'] >= GPU_MEM_SIZE: continue
                        if t_s - item2['s'] + item1['s'] >= GPU_MEM_SIZE: continue

                        nk_src = get_kvpr(b_l - item1['l'] + item2['l'], b_s - item1['s'] + item2['s'])
                        nk_tgt = get_kvpr(t_l - item2['l'] + item1['l'], t_s - item2['s'] + item1['s'])

                        if max(nk_src, nk_tgt) < cur_max - 1e-7:
                            # Apply Swap
                            plc[b][i] = item2
                            plc[t][j] = item1
                            dl = item2['l'] - item1['l']
                            ds = item2['s'] - item1['s']
                            update_stats(b, dl, ds)
                            update_stats(t, -dl, -ds)
                            improved = True
                            break
                    if improved: break
                if improved: break
            if improved: continue

            # 3. SWAP 2-1
            if len(b_items) >= 2:
                for i1 in range(len(b_items)):
                    for i2 in range(i1 + 1, len(b_items)):
                        m1 = b_items[i1]
                        m2 = b_items[i2]
                        pair_l = m1['l'] + m2['l']
                        pair_s = m1['s'] + m2['s']

                        for t in target_indices:
                            t_items = plc[t]
                            t_l, t_s = gpu_stats[t]['l'], gpu_stats[t]['s']
                            for j, m3 in enumerate(t_items):
                                if b_s - pair_s + m3['s'] >= GPU_MEM_SIZE: continue
                                if t_s - m3['s'] + pair_s >= GPU_MEM_SIZE: continue

                                nk_src = get_kvpr(b_l - pair_l + m3['l'], b_s - pair_s + m3['s'])
                                nk_tgt = get_kvpr(t_l - m3['l'] + pair_l, t_s - m3['s'] + pair_s)

                                if max(nk_src, nk_tgt) < cur_max - 1e-7:
                                    # Apply Swap 2-1
                                    it2 = plc[b].pop(i2)
                                    it1 = plc[b].pop(i1)
                                    it3 = plc[t].pop(j)
                                    plc[b].append(it3)
                                    plc[t].extend([it1, it2])

                                    pl = it1['l'] + it2['l']
                                    ps = it1['s'] + it2['s']
                                    update_stats(b, it3['l'] - pl, it3['s'] - ps)
                                    update_stats(t, pl - it3['l'], ps - it3['s'])
                                    improved = True
                                    break
                            if improved: break
                        if improved: break
                    if improved: break

        # Check Global Best
        cur_max, bottlenecks = evaluate_max_k()
        if cur_max < best_global_score:
            best_global_score = cur_max
            best_global_plc = [list(p) for p in plc]

        # Perturbation
        if not bottlenecks: break

        b = random.choice(bottlenecks)
        if not plc[b]: break

        # Eject 1 item to random feasible GPU
        idx = random.randrange(len(plc[b]))
        item = plc[b][idx]

        candidates = []
        for g in range(gpu_num):
            if g != b and gpu_stats[g]['s'] + item['s'] < GPU_MEM_SIZE:
                candidates.append(g)

        if candidates:
            t = random.choice(candidates)
            plc[b].pop(idx)
            plc[t].append(item)
            gpu_stats[b]['l'] -= item['l']; gpu_stats[b]['s'] -= item['s']
            gpu_stats[t]['l'] += item['l']; gpu_stats[t]['s'] += item['s']
        else:
            break

    # Final result
    result = {}
    for i in range(gpu_num):
        result[i] = [x['model'] for x in best_global_plc[i]]
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