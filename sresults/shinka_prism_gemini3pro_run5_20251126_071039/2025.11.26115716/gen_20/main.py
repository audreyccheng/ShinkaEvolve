# EVOLVE-BLOCK-START
"""Model placement algorithm for minimizing maximum KV cache pressure across GPUs"""

GPU_MEM_SIZE = 80  # GB

def compute_model_placement(gpu_num, models):
    """
    Compute a model placement that minimizes the maximum KVPR across all GPUs.
    Combines Binary Search with Multi-Strategy Best-Fit packing and Local Search refinement.
    """

    # Precompute model data
    m_data = []
    for m in models:
        w = m.req_rate / m.slo
        s = m.model_size
        m_data.append({'w': w, 's': s, 'obj': m})

    # --- 1. Heuristic Initialization for Binary Search Bounds ---
    # We need a feasible upper bound.

    def try_pack_with_sort(sort_key):
        sorted_items = sorted(m_data, key=sort_key, reverse=True)
        placements = [[] for _ in range(gpu_num)]
        loads = [0.0] * gpu_num
        used = [0.0] * gpu_num

        for item in sorted_items:
            best_g = -1
            best_score = float('inf')

            for g in range(gpu_num):
                rem = GPU_MEM_SIZE - used[g] - item['s']
                if rem > 1e-6:
                    # Minimize resulting pressure
                    p = (loads[g] + item['w']) / rem
                    if p < best_score:
                        best_score = p
                        best_g = g

            if best_g == -1: return None, float('inf')
            placements[best_g].append(item['obj'])
            loads[best_g] += item['w']
            used[best_g] += item['s']

        max_p = 0.0
        for g in range(gpu_num):
            rem = GPU_MEM_SIZE - used[g]
            if rem <= 1e-6:
                if loads[g] > 0: return None, float('inf')
            else:
                max_p = max(max_p, loads[g]/rem)
        return placements, max_p

    # Try density sort (usually best)
    init_placement, upper_bound = try_pack_with_sort(lambda x: x['w'] / x['s'] if x['s'] > 0 else 0)

    # If fails, try size sort
    if init_placement is None:
        init_placement, upper_bound = try_pack_with_sort(lambda x: x['s'])

    # If still fails, use loose bound (assuming feasible solution exists)
    if init_placement is None:
        upper_bound = 1000.0

    # --- 2. Binary Search for Optimal K ---
    low = 0.0
    high = upper_bound
    final_placement = init_placement

    for _ in range(20):
        if high - low < 1e-4: break
        mid = (low + high) / 2.0

        feasible = False
        res_placement = None

        # Strategies: Virtual Size, Physical Size, Load, Density
        strategies = [
            lambda x: x['w'] + mid * x['s'],
            lambda x: x['s'],
            lambda x: x['w'],
            lambda x: x['w'] / x['s'] if x['s'] > 1e-6 else 0
        ]

        for key_func in strategies:
            # Best Fit Decreasing
            items_sorted = sorted(m_data, key=key_func, reverse=True)
            gpu_models = [[] for _ in range(gpu_num)]
            gpu_l = [0.0] * gpu_num
            gpu_u = [0.0] * gpu_num
            ok = True

            for item in items_sorted:
                best_g = -1
                min_residual = float('inf')

                # Check all bins
                for g in range(gpu_num):
                    # Hard mem check
                    if gpu_u[g] + item['s'] >= GPU_MEM_SIZE - 1e-6: continue

                    # Constraint check: (L + w) <= K * (C - (U + s))
                    # Transformed: L + w + K(U + s) <= KC
                    lhs = (gpu_l[g] + item['w']) + mid * (gpu_u[g] + item['s'])
                    rhs = mid * GPU_MEM_SIZE

                    if lhs <= rhs + 1e-7:
                        # Best Fit: minimize residual capacity of transformed bin
                        res = rhs - lhs
                        if res < min_residual:
                            min_residual = res
                            best_g = g

                if best_g != -1:
                    gpu_models[best_g].append(item['obj'])
                    gpu_l[best_g] += item['w']
                    gpu_u[best_g] += item['s']
                else:
                    ok = False
                    break

            if ok:
                feasible = True
                res_placement = {i: gpu_models[i] for i in range(gpu_num)}
                break

        if feasible:
            final_placement = res_placement
            high = mid
        else:
            low = mid

    if final_placement is None:
        raise ValueError("Could not find feasible placement")

    # --- 3. Local Search Refinement (Pairwise Rebalancing) ---
    current_placement = final_placement

    # Track gpu state: loads (l) and used memory (u)
    gpu_state = []
    for g in range(gpu_num):
        l = sum(m.req_rate / m.slo for m in current_placement[g])
        u = sum(m.model_size for m in current_placement[g])
        gpu_state.append({'l': l, 'u': u, 'items': list(current_placement[g])})

    def calc_pressure(l, u):
        rem = GPU_MEM_SIZE - u
        if rem <= 1e-6: return float('inf') if l > 1e-6 else 0.0
        return l / rem

    iter_limit = 200 # Increased iteration limit for faster but more frequent updates
    for _ in range(iter_limit):
        # Identify bottleneck
        max_p = -1.0
        bottleneck = -1
        for g in range(gpu_num):
            p = calc_pressure(gpu_state[g]['l'], gpu_state[g]['u'])
            if p > max_p:
                max_p = p
                bottleneck = g

        if bottleneck == -1: break

        improved = False

        # Try to rebalance bottleneck with another GPU
        # Sort partners by pressure (try emptiest first)
        partners = list(range(gpu_num))
        partners.sort(key=lambda g: calc_pressure(gpu_state[g]['l'], gpu_state[g]['u']))

        for partner in partners:
            if partner == bottleneck: continue

            # Combine items
            items_pool = gpu_state[bottleneck]['items'] + gpu_state[partner]['items']

            # Heuristics to repack 2 bins: Load, Size, Density
            sort_keys = [
                lambda m: m.req_rate / m.slo,
                lambda m: m.model_size,
                lambda m: (m.req_rate / m.slo) / m.model_size if m.model_size > 0 else 0
            ]

            best_local_sol = None
            # We strictly want to reduce the max pressure of this pair below global max_p
            best_pair_max = max_p

            for key in sort_keys:
                sorted_items = sorted(items_pool, key=key, reverse=True)

                # Greedy Best-Fit on 2 bins
                sub_l = [0.0, 0.0]
                sub_u = [0.0, 0.0]
                sub_bins = [[], []]
                possible = True

                for item in sorted_items:
                    w = item.req_rate / item.slo
                    s = item.model_size

                    best_b = -1
                    min_res_p = float('inf')

                    for b in [0, 1]:
                        rem = GPU_MEM_SIZE - sub_u[b] - s
                        if rem > 1e-6:
                            p = (sub_l[b] + w) / rem
                            if p < min_res_p:
                                min_res_p = p
                                best_b = b

                    if best_b == -1:
                        possible = False
                        break

                    sub_l[best_b] += w
                    sub_u[best_b] += s
                    sub_bins[best_b].append(item)

                if possible:
                    p0 = calc_pressure(sub_l[0], sub_u[0])
                    p1 = calc_pressure(sub_l[1], sub_u[1])
                    pair_max = max(p0, p1)

                    if pair_max < best_pair_max - 1e-6:
                        best_pair_max = pair_max
                        best_local_sol = (sub_l, sub_u, sub_bins)

            if best_local_sol:
                # Apply update
                sl, su, sbins = best_local_sol

                gpu_state[bottleneck]['l'] = sl[0]
                gpu_state[bottleneck]['u'] = su[0]
                gpu_state[bottleneck]['items'] = sbins[0]

                gpu_state[partner]['l'] = sl[1]
                gpu_state[partner]['u'] = su[1]
                gpu_state[partner]['items'] = sbins[1]

                current_placement[bottleneck] = sbins[0]
                current_placement[partner] = sbins[1]

                improved = True
                break

        if not improved: break

    return current_placement

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