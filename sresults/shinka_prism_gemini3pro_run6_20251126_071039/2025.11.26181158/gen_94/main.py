# EVOLVE-BLOCK-START
import math

GPU_MEM_SIZE = 80  # GB

import random

def compute_model_placement(gpu_num, models):
    """
    Compute a model placement that minimizes the maximum KVPR across all GPUs.

    Algorithm: Multi-Start Perturbation-Based Iterated Local Search.
    1. Initialization: Randomized Bin Packing (Best Fit Decreasing) with Binary Search for K.
    2. Local Search: Steepest Descent on max KVPR using extensive neighborhoods:
       - Shift (1 -> 0)
       - Swap (1 <-> 1)
       - Swap (2 <-> 1)
       - Swap (2 <-> 2)
    3. Restarts: Run multiple times with randomized sorting to find better basins.
    """

    # Pre-calculate model properties
    model_data = []
    for i, m in enumerate(models):
        model_data.append({
            'model': m,
            'l': m.req_rate / m.slo,
            's': m.model_size,
            'id': i
        })

    def get_kvpr(l, s):
        if s >= GPU_MEM_SIZE - 1e-6: return 1e15
        return l / (GPU_MEM_SIZE - s)

    def solve_packing(target_k, mode='best_fit', randomize=False):
        capacity = target_k * GPU_MEM_SIZE

        # Base weights: L + K*S
        items = []
        for x in model_data:
            w = x['l'] + target_k * x['s']
            if randomize:
                w *= random.uniform(0.90, 1.10)
            items.append((w, x))

        # Sort descending by weight
        items.sort(key=lambda x: x[0], reverse=True)

        bins_l = [0.0] * gpu_num
        bins_s = [0.0] * gpu_num
        bins_items = [[] for _ in range(gpu_num)]

        for _, item in items:
            best_bin = -1

            # Setup metrics for comparison
            if mode == 'best_fit':
                best_val = float('inf') # Min slack
            elif mode == 'worst_fit':
                best_val = -1.0 # Max slack
            else: # min_kvpr
                best_val = float('inf') # Min resulting KVPR

            item_w = item['l'] + target_k * item['s']

            # Randomized start index
            indices = list(range(gpu_num))
            if randomize:
                random.shuffle(indices)

            for b in indices:
                # Physical Check
                if bins_s[b] + item['s'] >= GPU_MEM_SIZE - 1e-6:
                    continue

                # Linearized Constraint Check (soft check for min_kvpr but generally kept for feasibility)
                current_bin_w = bins_l[b] + target_k * bins_s[b]
                if current_bin_w + item_w > capacity + 1e-9:
                    continue

                if mode == 'best_fit':
                    slack = capacity - (current_bin_w + item_w)
                    if slack < best_val:
                        best_val = slack
                        best_bin = b
                elif mode == 'worst_fit':
                    slack = capacity - (current_bin_w + item_w)
                    if slack > best_val:
                        best_val = slack
                        best_bin = b
                else: # min_kvpr
                    new_l = bins_l[b] + item['l']
                    new_s = bins_s[b] + item['s']
                    kvpr = get_kvpr(new_l, new_s)
                    if kvpr < best_val:
                        best_val = kvpr
                        best_bin = b

            if best_bin != -1:
                bins_l[best_bin] += item['l']
                bins_s[best_bin] += item['s']
                bins_items[best_bin].append(item)
            else:
                return None

        return bins_items

    # --- Phase 1: Determine Baseline K ---
    # We use Best Fit for finding K as it is most efficient at packing
    low, high = 0.0, 1.0
    for _ in range(20):
        if solve_packing(high, mode='best_fit') is not None: break
        low = high
        high *= 2.0
    else: high = 1e9

    for _ in range(20):
        mid = (low + high) / 2
        if solve_packing(mid, mode='best_fit') is not None: high = mid
        else: low = mid
    base_k = high

    best_global_plc = None
    best_global_score = float('inf')

    # --- Phase 2: Multi-Start ILS ---
    strategies = ['best_fit', 'worst_fit', 'min_kvpr']

    # Run 20 restarts for diversity
    for restart in range(20):
        # Strategy Selection
        if restart == 0:
            mode = 'best_fit'
            k_factor = 1.0
            rnd = False
        else:
            mode = random.choice(strategies)
            k_factor = random.uniform(0.8, 1.3)
            rnd = True

        current_k = base_k * k_factor
        init_bins = solve_packing(current_k, mode=mode, randomize=rnd)

        # Fallback if aggressive parameters failed
        if init_bins is None:
            if restart == 0:
                init_bins = solve_packing(high * 1.01, mode='best_fit', randomize=False)
                if init_bins is None: continue
            else:
                continue

        # Convert to mutable state
        gpu_states = []
        for g in range(gpu_num):
            items = init_bins[g]
            g_l = sum(x['l'] for x in items)
            g_s = sum(x['s'] for x in items)
            gpu_states.append({'l': g_l, 's': g_s, 'items': list(items)})

        current_max = max(get_kvpr(g['l'], g['s']) for g in gpu_states)

        # Steepest Descent Local Search
        iter_limit = 200
        for _ in range(iter_limit):
            if current_max < 1e-9: break

            # Find bottleneck
            max_val = -1.0
            max_gpu = -1
            gpu_kvprs = []
            for g in range(gpu_num):
                val = get_kvpr(gpu_states[g]['l'], gpu_states[g]['s'])
                gpu_kvprs.append(val)
                if val > max_val:
                    max_val = val
                    max_gpu = g

            best_move = None
            best_gain = 0.0

            src = gpu_states[max_gpu]
            src_n = len(src['items'])

            # Function to eval move
            def eval_state(s_l, s_s, t_l, t_s):
                if s_s >= GPU_MEM_SIZE or t_s >= GPU_MEM_SIZE: return -1.0
                nk_s = get_kvpr(s_l, s_s)
                nk_t = get_kvpr(t_l, t_s)
                new_peak = max(nk_s, nk_t)
                if new_peak < current_max - 1e-9:
                    return current_max - new_peak
                return -1.0

            # Scan targets
            for t in range(gpu_num):
                if t == max_gpu: continue
                # Skip if target is already close to max (heuristic pruning)
                if gpu_kvprs[t] > current_max * 0.98: continue

                tgt = gpu_states[t]
                tgt_n = len(tgt['items'])

                # 1. Shift
                for i in range(src_n):
                    itm = src['items'][i]
                    gain = eval_state(src['l']-itm['l'], src['s']-itm['s'],
                                    tgt['l']+itm['l'], tgt['s']+itm['s'])
                    if gain > best_gain:
                        best_gain = gain
                        best_move = ('shift', i, t)

                # 2. Swap 1-1
                for i in range(src_n):
                    si = src['items'][i]
                    for j in range(tgt_n):
                        tj = tgt['items'][j]
                        gain = eval_state(
                            src['l']-si['l']+tj['l'], src['s']-si['s']+tj['s'],
                            tgt['l']-tj['l']+si['l'], tgt['s']-tj['s']+si['s']
                        )
                        if gain > best_gain:
                            best_gain = gain
                            best_move = ('swap11', i, t, j)

                # 3. Swap 2-1 (2 from src, 1 from tgt)
                if src_n >= 2:
                    for i1 in range(src_n):
                        for i2 in range(i1+1, src_n):
                            s1, s2 = src['items'][i1], src['items'][i2]
                            pl, ps = s1['l']+s2['l'], s1['s']+s2['s']
                            for j in range(tgt_n):
                                tj = tgt['items'][j]
                                gain = eval_state(
                                    src['l']-pl+tj['l'], src['s']-ps+tj['s'],
                                    tgt['l']-tj['l']+pl, tgt['s']-tj['s']+ps
                                )
                                if gain > best_gain:
                                    best_gain = gain
                                    best_move = ('swap21', i1, i2, t, j)

                # 4. Swap 2-2 (2 from src, 2 from tgt)
                if src_n >= 2 and tgt_n >= 2:
                     for i1 in range(src_n):
                        for i2 in range(i1+1, src_n):
                            s1, s2 = src['items'][i1], src['items'][i2]
                            sl, ss = s1['l']+s2['l'], s1['s']+s2['s']
                            for j1 in range(tgt_n):
                                for j2 in range(j1+1, tgt_n):
                                    t1, t2 = tgt['items'][j1], tgt['items'][j2]
                                    tl, ts = t1['l']+t2['l'], t1['s']+t2['s']

                                    gain = eval_state(
                                        src['l']-sl+tl, src['s']-ss+ts,
                                        tgt['l']-tl+sl, tgt['s']-ts+ss
                                    )
                                    if gain > best_gain:
                                        best_gain = gain
                                        best_move = ('swap22', i1, i2, t, j1, j2)

            if best_move:
                mtype = best_move[0]
                if mtype == 'shift':
                    _, i, t = best_move
                    itm = src['items'].pop(i)
                    tgt = gpu_states[t]
                    src['l']-=itm['l']; src['s']-=itm['s']
                    tgt['items'].append(itm)
                    tgt['l']+=itm['l']; tgt['s']+=itm['s']

                elif mtype == 'swap11':
                    _, i, t, j = best_move
                    tgt = gpu_states[t]
                    s_itm = src['items'][i]
                    t_itm = tgt['items'][j]
                    src['items'][i] = t_itm
                    tgt['items'][j] = s_itm

                    src['l'] += t_itm['l'] - s_itm['l']; src['s'] += t_itm['s'] - s_itm['s']
                    tgt['l'] += s_itm['l'] - t_itm['l']; tgt['s'] += s_itm['s'] - t_itm['s']

                elif mtype == 'swap21':
                    _, i1, i2, t, j = best_move
                    tgt = gpu_states[t]
                    # pop max index first
                    s2 = src['items'].pop(i2)
                    s1 = src['items'].pop(i1)
                    t1 = tgt['items'].pop(j)

                    src['items'].append(t1)
                    tgt['items'].extend([s1, s2])

                    # Full recalc simpler
                    src['l'] = sum(x['l'] for x in src['items']); src['s'] = sum(x['s'] for x in src['items'])
                    tgt['l'] = sum(x['l'] for x in tgt['items']); tgt['s'] = sum(x['s'] for x in tgt['items'])

                elif mtype == 'swap22':
                    _, i1, i2, t, j1, j2 = best_move
                    tgt = gpu_states[t]

                    s2 = src['items'].pop(i2)
                    s1 = src['items'].pop(i1)
                    t2 = tgt['items'].pop(j2)
                    t1 = tgt['items'].pop(j1)

                    src['items'].extend([t1, t2])
                    tgt['items'].extend([s1, s2])

                    src['l'] = sum(x['l'] for x in src['items']); src['s'] = sum(x['s'] for x in src['items'])
                    tgt['l'] = sum(x['l'] for x in tgt['items']); tgt['s'] = sum(x['s'] for x in tgt['items'])

                # Recompute current max
                current_max = max(get_kvpr(g['l'], g['s']) for g in gpu_states)
            else:
                break

        if current_max < best_global_score:
            best_global_score = current_max
            best_global_plc = {g: [x['model'] for x in gpu_states[g]['items']] for g in range(gpu_num)}

    if best_global_plc is None:
        raise ValueError("Could not find valid placement")

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