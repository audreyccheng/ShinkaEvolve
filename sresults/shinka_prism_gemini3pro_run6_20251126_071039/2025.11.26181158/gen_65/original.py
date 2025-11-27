# EVOLVE-BLOCK-START
"""Model placement algorithm for minimizing maximum KV cache pressure across GPUs"""

import random
import time

GPU_MEM_SIZE = 80.0

def compute_model_placement(gpu_num, models):
    """
    Compute a model placement that minimizes the maximum KVPR across all GPUs.

    Algorithm:
    1. Randomized Binary Search Initialization (Multi-Start).
    2. Steepest Descent Local Search (Move, Swap1-1, Swap2-1).
    3. Ruins & Recreate Perturbation (LNS).
    """
    start_time = time.time()

    # --- 1. Data Preparation ---
    model_data = []
    for i, m in enumerate(models):
        model_data.append({
            'model': m,
            'l': m.req_rate / m.slo,
            's': m.model_size,
            'id': i
        })

    def get_kvpr(l, s):
        # Using a slightly larger epsilon for safety
        if s >= GPU_MEM_SIZE - 1e-7:
            return 1e15
        return l / (GPU_MEM_SIZE - s)

    # --- 2. Initialization (Multi-Start Binary Search) ---

    def solve_packing(target_k, seed=None):
        """
        Tries to pack models with constraint: L + K*S <= K*M.
        Returns a valid placement (list of lists) or None.
        """
        capacity = target_k * GPU_MEM_SIZE

        # Create items list
        items = list(model_data)

        # Sort key: Linearized weight w = l + K*s
        if seed is not None:
            rng = random.Random(seed)
            # Add noise to the sorting key to explore diverse packings
            items.sort(key=lambda x: (x['l'] + target_k * x['s']) * rng.uniform(0.9, 1.1), reverse=True)
        else:
            # Deterministic sorting
            items.sort(key=lambda x: x['l'] + target_k * x['s'], reverse=True)

        gpu_l = [0.0] * gpu_num
        gpu_s = [0.0] * gpu_num
        gpu_items = [[] for _ in range(gpu_num)]

        for item in items:
            w = item['l'] + target_k * item['s']

            # Best Fit Heuristic
            best_bin = -1
            min_rem = float('inf')

            for i in range(gpu_num):
                # Hard memory constraint
                if gpu_s[i] + item['s'] >= GPU_MEM_SIZE - 1e-6:
                    continue

                # Linearized capacity constraint
                curr_w = gpu_l[i] + target_k * gpu_s[i]
                if curr_w + w <= capacity + 1e-9:
                    rem = capacity - (curr_w + w)
                    if rem < min_rem:
                        min_rem = rem
                        best_bin = i

            if best_bin != -1:
                gpu_l[best_bin] += item['l']
                gpu_s[best_bin] += item['s']
                gpu_items[best_bin].append(item)
            else:
                return None # Failed to pack
        return gpu_items

    # Binary Search for the lowest feasible K
    low = 0.0
    high = 1.0

    # Exponential expansion for upper bound
    for _ in range(20):
        if solve_packing(high) is not None: break
        low = high
        high *= 2.0
    else:
        high = 1e9 # Fallback

    # Refine K
    for _ in range(20):
        mid = (low + high) / 2
        if solve_packing(mid) is not None:
            high = mid
        else:
            low = mid

    base_k = high

    # Multi-Start: Run multiple times with noise at the found K to find the "luckiest" start
    best_init_plc = None
    best_init_score = float('inf')

    # 1 Deterministic + 19 Randomized runs
    seeds = [None] + [random.randint(0, 100000) for _ in range(19)]

    for seed in seeds:
        # We use a slightly relaxed K (1.001x) for randomized runs to ensure feasibility isn't too brittle
        k_attempt = base_k if seed is None else base_k * 1.0001
        res = solve_packing(k_attempt, seed)

        if res:
            # Evaluate real Max KVPR
            current_max = 0
            for g in range(gpu_num):
                l = sum(x['l'] for x in res[g])
                s = sum(x['s'] for x in res[g])
                k_val = get_kvpr(l, s)
                if k_val > current_max: current_max = k_val

            if current_max < best_init_score:
                best_init_score = current_max
                best_init_plc = res

    if best_init_plc is None:
        # Should not happen if binary search worked, but fallback just in case
        best_init_plc = solve_packing(high * 2.0)
        if best_init_plc is None: return {}

    # --- 3. Iterated Local Search (ILS) ---

    # Convert to mutable structure
    plc = [list(best_init_plc[g]) for g in range(gpu_num)]

    # Maintain running stats
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

    # Run for a fixed time budget (0.8s allows for deep search without timeout)
    while time.time() - start_time < 0.8:

        # Steepest Descent Local Search
        while True:
            cur_max, bottlenecks = evaluate(gpu_stats)
            if cur_max < 1e-9: break

            best_move = None
            best_imp = 0.0

            # Explore moves from bottlenecks
            # Optimization: Check at most top 2 bottlenecks
            for b_idx in bottlenecks[:2]:
                b_l = gpu_stats[b_idx]['l']
                b_s = gpu_stats[b_idx]['s']
                src_items = plc[b_idx]

                # Identify valid targets (must be better than current max)
                targets = []
                for t in range(gpu_num):
                    if t == b_idx: continue
                    if get_kvpr(gpu_stats[t]['l'], gpu_stats[t]['s']) < cur_max - 1e-6:
                        targets.append(t)

                if not targets: continue

                # 1. Move (1 item)
                for i, item in enumerate(src_items):
                    for t in targets:
                        if gpu_stats[t]['s'] + item['s'] >= GPU_MEM_SIZE: continue

                        nk_src = get_kvpr(b_l - item['l'], b_s - item['s'])
                        nk_tgt = get_kvpr(gpu_stats[t]['l'] + item['l'], gpu_stats[t]['s'] + item['s'])

                        new_global = max(nk_src, nk_tgt)
                        if new_global < cur_max - 1e-7:
                            imp = cur_max - new_global
                            if imp > best_imp:
                                best_imp = imp
                                best_move = ('move', b_idx, i, t)

                # 2. Swap (1-1)
                for i, s_item in enumerate(src_items):
                    for t in targets:
                        tgt_items = plc[t]
                        for j, t_item in enumerate(tgt_items):
                            # Capacity check
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

                # 3. Swap (2-1) : 2 from bottleneck, 1 from target
                # Computationally expensive, so we limit checks
                if len(src_items) >= 2:
                    checks = 0
                    limit = 200 # Safety cap
                    for i1 in range(len(src_items)):
                        if checks > limit: break
                        for i2 in range(i1 + 1, len(src_items)):
                            m1 = src_items[i1]
                            m2 = src_items[i2]
                            pair_l = m1['l'] + m2['l']
                            pair_s = m1['s'] + m2['s']

                            for t in targets:
                                tgt_items = plc[t]
                                for j, m3 in enumerate(tgt_items):
                                    checks += 1
                                    if b_s - pair_s + m3['s'] >= GPU_MEM_SIZE: continue
                                    if gpu_stats[t]['s'] - m3['s'] + pair_s >= GPU_MEM_SIZE: continue

                                    nk_src = get_kvpr(b_l - pair_l + m3['l'], b_s - pair_s + m3['s'])
                                    nk_tgt = get_kvpr(gpu_stats[t]['l'] - m3['l'] + pair_l, gpu_stats[t]['s'] - m3['s'] + pair_s)

                                    new_global = max(nk_src, nk_tgt)
                                    if new_global < cur_max - 1e-7:
                                        imp = cur_max - new_global
                                        if imp > best_imp:
                                            best_imp = imp
                                            best_move = ('swap21', b_idx, i1, i2, t, j)

            # Apply Best Move
            if best_move:
                mtype = best_move[0]
                if mtype == 'move':
                    _, b, i, t = best_move
                    item = plc[b].pop(i)
                    plc[t].append(item)
                    gpu_stats[b]['l'] -= item['l']; gpu_stats[b]['s'] -= item['s']
                    gpu_stats[t]['l'] += item['l']; gpu_stats[t]['s'] += item['s']
                elif mtype == 'swap11':
                    _, b, i, t, j = best_move
                    it1 = plc[b][i]
                    it2 = plc[t][j]
                    plc[b][i] = it2
                    plc[t][j] = it1
                    dl = it2['l'] - it1['l']; ds = it2['s'] - it1['s']
                    gpu_stats[b]['l'] += dl; gpu_stats[b]['s'] += ds
                    gpu_stats[t]['l'] -= dl; gpu_stats[t]['s'] -= ds
                elif mtype == 'swap21':
                    _, b, i1, i2, t, j = best_move
                    # Pop higher index first
                    it2 = plc[b].pop(i2)
                    it1 = plc[b].pop(i1)
                    it3 = plc[t].pop(j)
                    plc[b].append(it3)
                    plc[t].extend([it1, it2])

                    pl = it1['l'] + it2['l']; ps = it1['s'] + it2['s']
                    gpu_stats[b]['l'] += it3['l'] - pl; gpu_stats[b]['s'] += it3['s'] - ps
                    gpu_stats[t]['l'] += pl - it3['l']; gpu_stats[t]['s'] += ps - it3['s']
            else:
                break # Local optima

        # Update Global Best
        cur_max, bottlenecks = evaluate(gpu_stats)
        if cur_max < global_best_score:
            global_best_score = cur_max
            global_best_plc = [list(p) for p in plc]

        # --- Perturbation (Ruins & Recreate) ---
        if not bottlenecks: break

        b_idx = random.choice(bottlenecks)
        if not plc[b_idx]: break

        # Remove 1-2 random items from bottleneck
        num_remove = min(len(plc[b_idx]), random.randint(1, 2))
        removed = []
        for _ in range(num_remove):
            idx = random.randrange(len(plc[b_idx]))
            item = plc[b_idx].pop(idx)
            removed.append(item)
            gpu_stats[b_idx]['l'] -= item['l']
            gpu_stats[b_idx]['s'] -= item['s']

        # Re-insert greedily into best feasible GPU
        success = True
        for item in removed:
            best_t = -1
            best_t_k = float('inf')

            # Check all other GPUs
            candidates = list(range(gpu_num))
            random.shuffle(candidates)
            for t in candidates:
                if t == b_idx: continue
                if gpu_stats[t]['s'] + item['s'] < GPU_MEM_SIZE:
                    k = get_kvpr(gpu_stats[t]['l'] + item['l'], gpu_stats[t]['s'] + item['s'])
                    if k < best_t_k:
                        best_t_k = k
                        best_t = t

            if best_t != -1:
                plc[best_t].append(item)
                gpu_stats[best_t]['l'] += item['l']
                gpu_stats[best_t]['s'] += item['s']
            else:
                success = False
                break

        if not success:
            # Revert to global best if perturbation failed
            plc = [list(p) for p in global_best_plc]
            gpu_stats = []
            for g in range(gpu_num):
                l = sum(x['l'] for x in plc[g])
                s = sum(x['s'] for x in plc[g])
                gpu_stats.append({'l': l, 's': s})

            # Fallback: simple random swap to break cycle
            b = random.randrange(gpu_num)
            if plc[b]:
                t = random.randrange(gpu_num)
                if t != b and plc[t]:
                    i = random.randrange(len(plc[b]))
                    j = random.randrange(len(plc[t]))
                    it1, it2 = plc[b][i], plc[t][j]
                    if (gpu_stats[b]['s'] - it1['s'] + it2['s'] < GPU_MEM_SIZE and
                        gpu_stats[t]['s'] - it2['s'] + it1['s'] < GPU_MEM_SIZE):
                        plc[b][i] = it2; plc[t][j] = it1
                        dl = it2['l'] - it1['l']; ds = it2['s'] - it1['s']
                        gpu_stats[b]['l'] += dl; gpu_stats[b]['s'] += ds
                        gpu_stats[t]['l'] -= dl; gpu_stats[t]['s'] -= ds

    # Format result
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