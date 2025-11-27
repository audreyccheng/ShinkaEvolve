# EVOLVE-BLOCK-START
"""Model placement algorithm for minimizing maximum KV cache pressure across GPUs"""

import random
import time

GPU_MEM_SIZE = 80.0

def compute_model_placement(gpu_num, models):
    """
    Compute a model placement that minimizes the maximum KVPR across all GPUs.

    Algorithm:
    1. Multi-Start Randomized Initialization:
       - Uses Binary Search to find the tightest feasible KVPR 'K'.
       - Generates multiple initial solutions by adding random noise to sorting weights (Linearized: Load + K*Size).
       - Selects the best start based on actual Max KVPR.
    2. Iterated Local Search (ILS):
       - Steepest Descent on bottleneck GPUs.
       - Neighborhoods: Move, Swap(1-1), Swap(2-1).
       - Targeted Perturbation: Ejects items from bottleneck and re-inserts them greedily.
    """
    start_time = time.time()

    # 1. Preprocessing
    model_data = []
    for i, m in enumerate(models):
        model_data.append({
            'model': m,
            'l': m.req_rate / m.slo,
            's': m.model_size,
            'id': i
        })

    def get_kvpr(l, s):
        if s >= GPU_MEM_SIZE - 1e-7: return 1e16
        return l / (GPU_MEM_SIZE - s)

    # 2. Initialization Logic
    def solve_packing(target_k, seed=None):
        capacity = target_k * GPU_MEM_SIZE
        items = list(model_data)

        # Sort Key: Linearized weight (L + K*S)
        # If seed provided, add multiplicative noise to diversify
        if seed is not None:
            rng = random.Random(seed)
            items.sort(key=lambda x: (x['l'] + target_k * x['s']) * rng.uniform(0.9, 1.1), reverse=True)
        else:
            items.sort(key=lambda x: x['l'] + target_k * x['s'], reverse=True)

        gpu_l = [0.0] * gpu_num
        gpu_s = [0.0] * gpu_num
        gpu_items = [[] for _ in range(gpu_num)]

        for item in items:
            w = item['l'] + target_k * item['s']
            best_bin = -1
            min_rem = float('inf')

            # Best Fit Decreasing
            for i in range(gpu_num):
                if gpu_s[i] + item['s'] >= GPU_MEM_SIZE - 1e-6: continue

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
                return None
        return gpu_items

    # 3. Binary Search for Baseline K
    low, high = 0.0, 1.0
    # Exponential search for upper bound
    for _ in range(20):
        if solve_packing(high) is not None: break
        low = high
        high *= 2.0
    else: high = 1e9

    # Binary Search refinement
    for _ in range(25):
        mid = (low + high) / 2
        if solve_packing(mid) is not None: high = mid
        else: low = mid
    base_k = high

    # 4. Randomized Restarts
    best_plc = None
    best_score = float('inf')

    # Run restarts near base_k to find a configuration with lower actual max KVPR
    # base_k is the theoretical limit where sum(w) <= Capacity, but actual KVPR might differ due to bin packing inefficiencies
    search_k = base_k * 1.001
    seeds = [None] + list(range(19)) # 20 total starts

    for seed in seeds:
        if time.time() - start_time > 0.4: break
        res = solve_packing(search_k, seed)
        if res:
            max_k = 0
            for g in range(gpu_num):
                l = sum(x['l'] for x in res[g])
                s = sum(x['s'] for x in res[g])
                k = get_kvpr(l, s)
                if k > max_k: max_k = k

            if max_k < best_score:
                best_score = max_k
                best_plc = res

    if best_plc is None:
        best_plc = solve_packing(base_k) or solve_packing(1e9)
        if best_plc is None: # Emergency fallback
             best_plc = [[] for _ in range(gpu_num)]
             for i, m in enumerate(model_data): best_plc[i%gpu_num].append(m)

    # Convert to mutable structures for Local Search
    plc = [list(best_plc[g]) for g in range(gpu_num)]
    gpu_stats = []
    for g in range(gpu_num):
        l = sum(x['l'] for x in plc[g])
        s = sum(x['s'] for x in plc[g])
        gpu_stats.append({'l': l, 's': s})

    global_best_plc = [list(p) for p in plc]
    global_best_score = best_score

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

    # 5. Iterated Local Search (ILS)
    while time.time() - start_time < 0.9:

        # Local Search (Steepest Descent)
        while True:
            cur_max, bottlenecks = evaluate(gpu_stats)
            if cur_max < 1e-9: break

            best_move = None
            best_imp = 0.0

            # Pruning: Only consider moves from worst bottlenecks to save time
            b_indices = bottlenecks[:2]

            # Identify valid targets (must be better than current max)
            targets = [t for t in range(gpu_num) if t not in bottlenecks]
            # Further filter targets
            valid_targets = [t for t in targets if get_kvpr(gpu_stats[t]['l'], gpu_stats[t]['s']) < cur_max]

            if not valid_targets: break # Cannot improve

            for b_idx in b_indices:
                b_l = gpu_stats[b_idx]['l']
                b_s = gpu_stats[b_idx]['s']
                src_items = plc[b_idx]

                # 1. Move
                for i, item in enumerate(src_items):
                    for t in valid_targets:
                        if gpu_stats[t]['s'] + item['s'] >= GPU_MEM_SIZE: continue

                        nk_src = get_kvpr(b_l - item['l'], b_s - item['s'])
                        nk_tgt = get_kvpr(gpu_stats[t]['l'] + item['l'], gpu_stats[t]['s'] + item['s'])

                        new_global = max(nk_src, nk_tgt)
                        if new_global < cur_max - 1e-8:
                            imp = cur_max - new_global
                            if imp > best_imp:
                                best_imp = imp
                                best_move = ('move', b_idx, i, t)

                # 2. Swap 1-1
                for i, s_item in enumerate(src_items):
                    for t in valid_targets:
                        tgt_items = plc[t]
                        for j, t_item in enumerate(tgt_items):
                            if b_s - s_item['s'] + t_item['s'] >= GPU_MEM_SIZE: continue
                            if gpu_stats[t]['s'] - t_item['s'] + s_item['s'] >= GPU_MEM_SIZE: continue

                            nk_src = get_kvpr(b_l - s_item['l'] + t_item['l'], b_s - s_item['s'] + t_item['s'])
                            nk_tgt = get_kvpr(gpu_stats[t]['l'] - t_item['l'] + s_item['l'], gpu_stats[t]['s'] - t_item['s'] + s_item['s'])

                            new_global = max(nk_src, nk_tgt)
                            if new_global < cur_max - 1e-8:
                                imp = cur_max - new_global
                                if imp > best_imp:
                                    best_imp = imp
                                    best_move = ('swap11', b_idx, i, t, j)

                # 3. Swap 2-1 (2 from bottleneck)
                if len(src_items) >= 2:
                    checks = 0
                    # Sort indices by size desc to prioritize larger removals which likely reduce pressure more
                    # This heuristic speeds up finding good swaps
                    sorted_indices = sorted(range(len(src_items)), key=lambda k: src_items[k]['s'], reverse=True)
                    MAX_CHECKS = 200

                    for idx1 in range(len(sorted_indices)):
                        i1 = sorted_indices[idx1]
                        if checks > MAX_CHECKS: break
                        for idx2 in range(idx1 + 1, len(sorted_indices)):
                            i2 = sorted_indices[idx2]
                            checks += 1
                            if checks > MAX_CHECKS: break

                            m1 = src_items[i1]
                            m2 = src_items[i2]
                            pair_l = m1['l'] + m2['l']
                            pair_s = m1['s'] + m2['s']

                            for t in valid_targets:
                                tgt_items = plc[t]
                                for j, m3 in enumerate(tgt_items):
                                    if b_s - pair_s + m3['s'] >= GPU_MEM_SIZE: continue
                                    if gpu_stats[t]['s'] - m3['s'] + pair_s >= GPU_MEM_SIZE: continue

                                    nk_src = get_kvpr(b_l - pair_l + m3['l'], b_s - pair_s + m3['s'])
                                    nk_tgt = get_kvpr(gpu_stats[t]['l'] - m3['l'] + pair_l, gpu_stats[t]['s'] - m3['s'] + pair_s)

                                    new_global = max(nk_src, nk_tgt)
                                    if new_global < cur_max - 1e-8:
                                        imp = cur_max - new_global
                                        if imp > best_imp:
                                            best_imp = imp
                                            best_move = ('swap21', b_idx, i1, i2, t, j)

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
                    it1, it2 = plc[b][i], plc[t][j]
                    plc[b][i], plc[t][j] = it2, it1
                    dl, ds = it2['l'] - it1['l'], it2['s'] - it1['s']
                    gpu_stats[b]['l'] += dl; gpu_stats[b]['s'] += ds
                    gpu_stats[t]['l'] -= dl; gpu_stats[t]['s'] -= ds
                elif mtype == 'swap21':
                    _, b, i1, i2, t, j = best_move
                    # Be careful with popping indices
                    idx1, idx2 = sorted((i1, i2), reverse=True)
                    m2 = plc[b].pop(idx1)
                    m1 = plc[b].pop(idx2)
                    m3 = plc[t].pop(j)
                    plc[b].append(m3)
                    plc[t].extend([m1, m2])

                    pl = m1['l'] + m2['l']; ps = m1['s'] + m2['s']
                    gpu_stats[b]['l'] += m3['l'] - pl; gpu_stats[b]['s'] += m3['s'] - ps
                    gpu_stats[t]['l'] += pl - m3['l']; gpu_stats[t]['s'] += ps - m3['s']
            else:
                break # Local Optima Reached

        # Update Global Best
        cur_max, bottlenecks = evaluate(gpu_stats)
        if cur_max < global_best_score:
            global_best_score = cur_max
            global_best_plc = [list(p) for p in plc]

        # Perturbation (Ruins & Recreate)
        if not bottlenecks: break

        b_idx = random.choice(bottlenecks)
        if not plc[b_idx]: break

        # Remove 1-2 random items
        num_remove = min(len(plc[b_idx]), random.randint(1, 2))
        removed_items = []
        for _ in range(num_remove):
            if not plc[b_idx]: break
            idx = random.randrange(len(plc[b_idx]))
            item = plc[b_idx].pop(idx)
            removed_items.append(item)
            gpu_stats[b_idx]['l'] -= item['l']
            gpu_stats[b_idx]['s'] -= item['s']

        # Re-insert greedily into best feasible GPU
        success = True
        for item in removed_items:
            best_t = -1
            best_k = float('inf')

            # Check all other GPUs
            candidates = list(range(gpu_num))
            random.shuffle(candidates)

            for t in candidates:
                if t == b_idx: continue
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
                # Put back
                plc[b_idx].append(item)
                gpu_stats[b_idx]['l'] += item['l']
                gpu_stats[b_idx]['s'] += item['s']
                success = False

    # Result
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