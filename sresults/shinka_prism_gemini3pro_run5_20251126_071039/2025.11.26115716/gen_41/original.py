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

    # --- 3. Local Search Refinement with Variance Tie-Breaking ---
    current_placement = final_placement

    # Track gpu state
    gpu_loads = [sum(m.req_rate / m.slo for m in current_placement[g]) for g in range(gpu_num)]
    gpu_used = [sum(m.model_size for m in current_placement[g]) for g in range(gpu_num)]

    def get_pressure(l, u):
        rem = GPU_MEM_SIZE - u
        if rem <= 1e-6: return float('inf') if l > 1e-6 else 0.0
        return l / rem

    iter_limit = 500
    for _ in range(iter_limit):
        # Calculate current metrics
        pressures = [get_pressure(gpu_loads[g], gpu_used[g]) for g in range(gpu_num)]
        max_p = max(pressures)
        sum_sq_p = sum(p*p for p in pressures)

        # Identify bottleneck
        bottlenecks = [g for g in range(gpu_num) if pressures[g] >= max_p - 1e-6]
        if not bottlenecks: break
        bn = bottlenecks[0]

        best_move = None
        # move structure: (type, partner, idx_bn, idx_pt, new_bn_l, new_bn_u, new_pt_l, new_pt_u, new_max, new_sq)

        # Precompute sorted indices for max_others calculation
        sorted_p_indices = sorted(range(gpu_num), key=lambda i: pressures[i], reverse=True)

        for partner in range(gpu_num):
            if partner == bn: continue

            # Determine max_others for this (bn, partner) pair
            max_others = 0.0
            for k in range(min(3, len(sorted_p_indices))):
                idx = sorted_p_indices[k]
                if idx != bn and idx != partner:
                    max_others = pressures[idx]
                    break

            # --- Try Moving an item from BN to Partner ---
            for i, item in enumerate(current_placement[bn]):
                w, s = item.req_rate / item.slo, item.model_size
                if gpu_used[partner] + s >= GPU_MEM_SIZE - 1e-6: continue

                new_bn_l = gpu_loads[bn] - w
                new_bn_u = gpu_used[bn] - s
                new_pt_l = gpu_loads[partner] + w
                new_pt_u = gpu_used[partner] + s

                p_bn = get_pressure(new_bn_l, new_bn_u)
                p_pt = get_pressure(new_pt_l, new_pt_u)

                new_local_max = max(p_bn, p_pt)
                if new_local_max > max_p + 1e-9: continue

                new_global_max = max(max_others, new_local_max)
                new_global_sq = sum_sq_p - (pressures[bn]**2 + pressures[partner]**2) + (p_bn**2 + p_pt**2)

                is_better = False
                if new_global_max < max_p - 1e-9:
                    is_better = True
                elif new_global_max < max_p + 1e-9 and new_global_sq < sum_sq_p - 1e-9:
                    is_better = True

                if is_better:
                    if best_move is None or new_global_max < best_move[8] - 1e-9 or \
                       (abs(new_global_max - best_move[8]) < 1e-9 and new_global_sq < best_move[9]):
                        best_move = ('move', partner, i, None, new_bn_l, new_bn_u, new_pt_l, new_pt_u, new_global_max, new_global_sq)

            # --- Try Swapping items ---
            for i, m1 in enumerate(current_placement[bn]):
                w1, s1 = m1.req_rate / m1.slo, m1.model_size
                for j, m2 in enumerate(current_placement[partner]):
                    w2, s2 = m2.req_rate / m2.slo, m2.model_size

                    new_bn_u = gpu_used[bn] - s1 + s2
                    if new_bn_u >= GPU_MEM_SIZE - 1e-6: continue
                    new_pt_u = gpu_used[partner] - s2 + s1
                    if new_pt_u >= GPU_MEM_SIZE - 1e-6: continue

                    new_bn_l = gpu_loads[bn] - w1 + w2
                    new_pt_l = gpu_loads[partner] - w2 + w1

                    p_bn = get_pressure(new_bn_l, new_bn_u)
                    p_pt = get_pressure(new_pt_l, new_pt_u)

                    new_local_max = max(p_bn, p_pt)
                    if new_local_max > max_p + 1e-9: continue

                    new_global_max = max(max_others, new_local_max)
                    new_global_sq = sum_sq_p - (pressures[bn]**2 + pressures[partner]**2) + (p_bn**2 + p_pt**2)

                    is_better = False
                    if new_global_max < max_p - 1e-9:
                        is_better = True
                    elif new_global_max < max_p + 1e-9 and new_global_sq < sum_sq_p - 1e-9:
                        is_better = True

                    if is_better:
                         if best_move is None or new_global_max < best_move[8] - 1e-9 or \
                            (abs(new_global_max - best_move[8]) < 1e-9 and new_global_sq < best_move[9]):
                            best_move = ('swap', partner, i, j, new_bn_l, new_bn_u, new_pt_l, new_pt_u, new_global_max, new_global_sq)

        if best_move:
            m_type, partner, i, j, n_bl, n_bu, n_pl, n_pu, n_max, n_sq = best_move
            if m_type == 'move':
                item = current_placement[bn].pop(i)
                current_placement[partner].append(item)
            else:
                item1 = current_placement[bn][i]
                item2 = current_placement[partner][j]
                current_placement[bn][i] = item2
                current_placement[partner][j] = item1

            gpu_loads[bn] = n_bl
            gpu_used[bn] = n_bu
            gpu_loads[partner] = n_pl
            gpu_used[partner] = n_pu
        else:
            break

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