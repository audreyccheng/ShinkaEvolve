# EVOLVE-BLOCK-START
import math

GPU_MEM_SIZE = 80  # GB

def compute_model_placement(gpu_num, models):
    """
    Compute a model placement that minimizes the maximum KVPR across all GPUs.

    Algorithm: Multi-Start Randomized ILS with Heavy Local Search.

    1. Binary Search Baseline: Determine a 'base_k' using strict binary search.
    2. Multi-Start Loop (5 restarts):
       - Initialization: Randomized Bin Packing.
         Uses base_k but adds random noise to sorting weights to explore different configurations.
       - Local Search: Steepest Descent on max KVPR.
         Neighborhoods:
         - Shift (Move 1)
         - Swap 1-1
         - Swap 2-1 (2 from bottleneck, 1 from target)
         - Swap 2-2 (2 from bottleneck, 2 from target) - Critical for resolving complex deadlocks.
    """
    import random

    model_data = []
    for m in models:
        model_data.append({
            'model': m,
            'l': m.req_rate / m.slo,
            's': m.model_size
        })

    def get_kvpr(l, s):
        if s >= GPU_MEM_SIZE - 1e-6: return 1e15
        return l / (GPU_MEM_SIZE - s)

    # --- Phase 1: Determine Baseline K ---
    # Standard deterministic packing check
    def can_pack_deterministic(target_k):
        capacity = target_k * GPU_MEM_SIZE
        # Sort by linearized weight
        items = sorted(model_data, key=lambda x: x['l'] + target_k * x['s'], reverse=True)

        bins_l = [0.0] * gpu_num
        bins_s = [0.0] * gpu_num

        for item in items:
            best_idx = -1
            min_slack = float('inf')
            w = item['l'] + target_k * item['s']

            for i in range(gpu_num):
                if bins_s[i] + item['s'] >= GPU_MEM_SIZE - 1e-6: continue
                curr_w = bins_l[i] + target_k * bins_s[i]

                if curr_w + w <= capacity + 1e-9:
                    slack = capacity - (curr_w + w)
                    if slack < min_slack:
                        min_slack = slack
                        best_idx = i

            if best_idx != -1:
                bins_l[best_idx] += item['l']
                bins_s[best_idx] += item['s']
            else:
                return False
        return True

    low, high = 0.0, 1.0
    for _ in range(20):
        if can_pack_deterministic(high): break
        low = high; high *= 2.0
    else: high = 1e9

    for _ in range(25):
        mid = (low + high) / 2
        if can_pack_deterministic(mid): high = mid
        else: low = mid
    base_k = high

    # --- Phase 2: Multi-Start Local Search ---

    def solve_randomized_packing(target_k, randomize=False):
        capacity = target_k * GPU_MEM_SIZE
        # Create weighted items
        items = []
        for d in model_data:
            weight = d['l'] + target_k * d['s']
            if randomize:
                # Add +/- 5% noise
                weight *= random.uniform(0.95, 1.05)
            items.append((weight, d))

        items.sort(key=lambda x: x[0], reverse=True)

        gpu_state = [{'l': 0.0, 's': 0.0, 'items': []} for _ in range(gpu_num)]

        # Tie-breaking randomization
        indices = list(range(gpu_num))

        for _, item in items:
            best_idx = -1
            min_slack = float('inf')

            w_real = item['l'] + target_k * item['s']

            if randomize: random.shuffle(indices)

            for i in indices:
                st = gpu_state[i]
                if st['s'] + item['s'] >= GPU_MEM_SIZE - 1e-6: continue

                curr_w = st['l'] + target_k * st['s']
                if curr_w + w_real <= capacity + 1e-9:
                    slack = capacity - (curr_w + w_real)
                    if slack < min_slack:
                        min_slack = slack
                        best_idx = i

            if best_idx != -1:
                st = gpu_state[best_idx]
                st['l'] += item['l']
                st['s'] += item['s']
                st['items'].append(item)
            else:
                return None
        return gpu_state

    best_global_score = float('inf')
    best_global_state = None

    # Run restarts
    # 0: Deterministic (baseline)
    # 1-4: Randomized with slightly relaxed K to allow different packings
    for restart in range(5):
        k_factor = 1.0 if restart == 0 else 1.05
        init_state = solve_randomized_packing(base_k * k_factor, randomize=(restart > 0))

        if init_state is None:
            # If relaxed packing fails (rare if base_k is correct), try finding a looser K
            if restart == 0:
                init_state = solve_randomized_packing(base_k * 1.01, False)
            if init_state is None: continue

        # Local Search
        curr_state = init_state

        # Calculate initial max
        max_kvpr = 0.0
        for g in range(gpu_num):
            k = get_kvpr(curr_state[g]['l'], curr_state[g]['s'])
            if k > max_kvpr: max_kvpr = k

        for _ in range(100): # Iterations per restart
            if max_kvpr < 1e-9: break

            bottleneck = -1
            highest_k = -1.0

            # Identify bottleneck
            kvpr_map = []
            for g in range(gpu_num):
                val = get_kvpr(curr_state[g]['l'], curr_state[g]['s'])
                kvpr_map.append(val)
                if val > highest_k:
                    highest_k = val
                    bottleneck = g

            best_move = None # (type, improvement, args)
            best_imp = 0.0

            src = curr_state[bottleneck]

            # Helper to evaluate improvement
            def eval_imp(nl_src, ns_src, nl_tgt, ns_tgt):
                k1 = get_kvpr(nl_src, ns_src)
                k2 = get_kvpr(nl_tgt, ns_tgt)
                new_peak = max(k1, k2)
                if new_peak < highest_k - 1e-7:
                    return highest_k - new_peak
                return -1.0

            # 1. Shift
            for i, m in enumerate(src['items']):
                for t in range(gpu_num):
                    if t == bottleneck or kvpr_map[t] >= highest_k: continue
                    tgt = curr_state[t]
                    if tgt['s'] + m['s'] >= GPU_MEM_SIZE: continue

                    imp = eval_imp(
                        src['l'] - m['l'], src['s'] - m['s'],
                        tgt['l'] + m['l'], tgt['s'] + m['s']
                    )
                    if imp > best_imp:
                        best_imp = imp
                        best_move = ('shift', i, t)

            # 2. Swap 1-1
            for i, m1 in enumerate(src['items']):
                for t in range(gpu_num):
                    if t == bottleneck or kvpr_map[t] >= highest_k: continue
                    tgt = curr_state[t]
                    for j, m2 in enumerate(tgt['items']):
                        ns_s = src['s'] - m1['s'] + m2['s']
                        nt_s = tgt['s'] - m2['s'] + m1['s']
                        if ns_s >= GPU_MEM_SIZE or nt_s >= GPU_MEM_SIZE: continue

                        imp = eval_imp(
                            src['l'] - m1['l'] + m2['l'], ns_s,
                            tgt['l'] - m2['l'] + m1['l'], nt_s
                        )
                        if imp > best_imp:
                            best_imp = imp
                            best_move = ('swap11', i, t, j)

            # 3. Swap 2-1 (2 from Bottleneck)
            if len(src['items']) >= 2:
                n_src = len(src['items'])
                for i1 in range(n_src):
                    for i2 in range(i1+1, n_src):
                        m1, m2 = src['items'][i1], src['items'][i2]
                        pl, ps = m1['l']+m2['l'], m1['s']+m2['s']

                        for t in range(gpu_num):
                            if t == bottleneck or kvpr_map[t] >= highest_k: continue
                            tgt = curr_state[t]
                            for j, m3 in enumerate(tgt['items']):
                                ns_s = src['s'] - ps + m3['s']
                                nt_s = tgt['s'] - m3['s'] + ps
                                if ns_s >= GPU_MEM_SIZE or nt_s >= GPU_MEM_SIZE: continue

                                imp = eval_imp(
                                    src['l'] - pl + m3['l'], ns_s,
                                    tgt['l'] - m3['l'] + pl, nt_s
                                )
                                if imp > best_imp:
                                    best_imp = imp
                                    best_move = ('swap21', i1, i2, t, j)

            # 4. Swap 2-2 (2 from Bottleneck, 2 from Target)
            if len(src['items']) >= 2:
                n_src = len(src['items'])
                for i1 in range(n_src):
                    for i2 in range(i1+1, n_src):
                        m1, m2 = src['items'][i1], src['items'][i2]
                        pl_src, ps_src = m1['l']+m2['l'], m1['s']+m2['s']

                        for t in range(gpu_num):
                            if t == bottleneck or kvpr_map[t] >= highest_k: continue
                            tgt = curr_state[t]
                            if len(tgt['items']) < 2: continue

                            n_tgt = len(tgt['items'])
                            for j1 in range(n_tgt):
                                for j2 in range(j1+1, n_tgt):
                                    m3, m4 = tgt['items'][j1], tgt['items'][j2]
                                    pl_tgt, ps_tgt = m3['l']+m4['l'], m3['s']+m4['s']

                                    ns_s = src['s'] - ps_src + ps_tgt
                                    nt_s = tgt['s'] - ps_tgt + ps_src
                                    if ns_s >= GPU_MEM_SIZE or nt_s >= GPU_MEM_SIZE: continue

                                    imp = eval_imp(
                                        src['l'] - pl_src + pl_tgt, ns_s,
                                        tgt['l'] - pl_tgt + pl_src, nt_s
                                    )
                                    if imp > best_imp:
                                        best_imp = imp
                                        best_move = ('swap22', i1, i2, t, j1, j2)

            if best_move:
                mtype = best_move[0]
                if mtype == 'shift':
                    _, i, t = best_move
                    m = src['items'].pop(i)
                    tgt = curr_state[t]
                    tgt['items'].append(m)
                    src['l']-=m['l']; src['s']-=m['s']
                    tgt['l']+=m['l']; tgt['s']+=m['s']
                elif mtype == 'swap11':
                    _, i, t, j = best_move
                    tgt = curr_state[t]
                    m1, m2 = src['items'][i], tgt['items'][j]
                    src['items'][i] = m2
                    tgt['items'][j] = m1
                    diff_l = m2['l']-m1['l']; diff_s = m2['s']-m1['s']
                    src['l']+=diff_l; src['s']+=diff_s
                    tgt['l']-=diff_l; tgt['s']-=diff_s
                elif mtype == 'swap21':
                    _, i1, i2, t, j = best_move
                    tgt = curr_state[t]
                    # Pop largest index first
                    m2 = src['items'].pop(i2)
                    m1 = src['items'].pop(i1)
                    m3 = tgt['items'].pop(j)
                    src['items'].append(m3)
                    tgt['items'].extend([m1, m2])

                    pl, ps = m1['l']+m2['l'], m1['s']+m2['s']
                    src['l'] += m3['l'] - pl; src['s'] += m3['s'] - ps
                    tgt['l'] += pl - m3['l']; tgt['s'] += ps - m3['s']
                elif mtype == 'swap22':
                    _, i1, i2, t, j1, j2 = best_move
                    tgt = curr_state[t]
                    m_s2 = src['items'].pop(i2)
                    m_s1 = src['items'].pop(i1)
                    m_t2 = tgt['items'].pop(j2)
                    m_t1 = tgt['items'].pop(j1)

                    src['items'].extend([m_t1, m_t2])
                    tgt['items'].extend([m_s1, m_s2])

                    src['l'] = sum(x['l'] for x in src['items'])
                    src['s'] = sum(x['s'] for x in src['items'])
                    tgt['l'] = sum(x['l'] for x in tgt['items'])
                    tgt['s'] = sum(x['s'] for x in tgt['items'])

                # Update max_kvpr
                max_kvpr = 0.0
                for g in range(gpu_num):
                    k = get_kvpr(curr_state[g]['l'], curr_state[g]['s'])
                    if k > max_kvpr: max_kvpr = k
            else:
                break

        if max_kvpr < best_global_score:
            best_global_score = max_kvpr
            best_global_state = {g: [x['model'] for x in curr_state[g]['items']] for g in range(gpu_num)}

    if best_global_state is None:
        raise ValueError("No valid placement found.")

    return best_global_state
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