# EVOLVE-BLOCK-START
"""Model placement algorithm for minimizing maximum KV cache pressure across GPUs"""

import random

GPU_MEM_SIZE = 80  # GB

def compute_model_placement(gpu_num, models):
    """
    Compute a model placement that minimizes the maximum KVPR across all GPUs.
    Uses Binary Search for initialization and Variable Neighborhood Descent (VND)
    with Pairwise Ruin-and-Recreate for refinement.
    """

    # Fix random seed for reproducibility
    rng = random.Random(42)

    # Precompute model data for efficiency
    m_data = []
    for i, m in enumerate(models):
        w = m.req_rate / m.slo
        s = m.model_size
        m_data.append({
            'w': w,
            's': s,
            'obj': m,
            # Density is often the best heuristic for this specific problem
            'density': w/s if s > 0 else 0
        })

    # --- Part 1: Binary Search for Feasible K ---

    def solve_bin_packing(target_k, items):
        """
        Checks if placement is possible with max KVPR <= target_k.
        Constraint transformation:
        Load / (Cap - Used) <= K  <=>  Load + K*Used <= K*Cap
        Item effective size: w + K*s
        Bin capacity: K*Cap
        """
        # Strategies to sort items for Best Fit
        strategies = [
            lambda x: x['w'] + target_k * x['s'], # Effective cost
            lambda x: x['density'],               # Density
            lambda x: x['s'],                     # Physical size
            lambda x: x['w']                      # Load
        ]

        for key_func in strategies:
            sorted_items = sorted(items, key=key_func, reverse=True)

            bins_l = [0.0] * gpu_num
            bins_u = [0.0] * gpu_num
            bins_items = [[] for _ in range(gpu_num)]
            possible = True

            # Best Fit Decreasing
            for item in sorted_items:
                best_g = -1
                min_residual = float('inf')

                item_eff_size = item['w'] + target_k * item['s']
                max_eff_cap = target_k * GPU_MEM_SIZE

                for g in range(gpu_num):
                    # Hard memory limit (must be strictly less to avoid div by zero/inf pressure)
                    if bins_u[g] + item['s'] >= GPU_MEM_SIZE - 1e-6:
                        continue

                    # Target pressure limit check
                    current_eff_load = bins_l[g] + target_k * bins_u[g]

                    if current_eff_load + item_eff_size <= max_eff_cap + 1e-7:
                        # Minimize residual effective capacity
                        res = max_eff_cap - (current_eff_load + item_eff_size)
                        if res < min_residual:
                            min_residual = res
                            best_g = g

                if best_g != -1:
                    bins_l[best_g] += item['w']
                    bins_u[best_g] += item['s']
                    bins_items[best_g].append(item)
                else:
                    possible = False
                    break

            if possible:
                return bins_items

        return None

    # Determine bounds for Binary Search
    # Heuristic: Run a quick greedy pass to get a valid Upper Bound
    def get_initial_greedy(key_func):
        s_items = sorted(m_data, key=key_func, reverse=True)
        placements = [[] for _ in range(gpu_num)]
        bl = [0.0] * gpu_num
        bu = [0.0] * gpu_num

        for item in s_items:
            best_g = -1
            best_score = float('inf')

            for g in range(gpu_num):
                rem = GPU_MEM_SIZE - bu[g] - item['s']
                if rem > 1e-5:
                    # Minimize pressure increase
                    p = (bl[g] + item['w']) / rem
                    if p < best_score:
                        best_score = p
                        best_g = g

            if best_g == -1: return None, float('inf')
            placements[best_g].append(item)
            bl[best_g] += item['w']
            bu[best_g] += item['s']

        max_p = 0.0
        for g in range(gpu_num):
            rem = GPU_MEM_SIZE - bu[g]
            if rem <= 1e-6:
                if bl[g] > 0: max_p = float('inf')
            else:
                max_p = max(max_p, bl[g] / rem)
        return placements, max_p

    # Try getting an initial solution
    init_struct, upper_k = get_initial_greedy(lambda x: x['density'])
    if init_struct is None:
        init_struct, upper_k = get_initial_greedy(lambda x: x['s'])

    if init_struct is None:
        upper_k = 2000.0 # High fallback bound

    # Binary Search
    low = 0.0
    high = upper_k
    final_placement_struct = init_struct

    for _ in range(18): # Fixed iterations
        if high - low < 1e-4: break
        mid = (low + high) / 2.0
        res = solve_bin_packing(mid, m_data)
        if res:
            final_placement_struct = res
            high = mid
        else:
            low = mid

    if final_placement_struct is None:
        raise ValueError("Could not find feasible placement")

    # Convert struct to dictionary for refinement
    placement = {g: [x['obj'] for x in items] for g, items in enumerate(final_placement_struct)}

    # --- Part 2: Variable Neighborhood Descent (Local Search) ---

    def get_stats(placements_dict):
        stats = []
        max_p = -1.0
        bn_idx = -1

        for g in range(gpu_num):
            items = placements_dict[g]
            l = sum(m.req_rate / m.slo for m in items)
            u = sum(m.model_size for m in items)
            rem = GPU_MEM_SIZE - u

            p = float('inf')
            if rem > 1e-6:
                p = l / rem
            elif l == 0:
                p = 0.0

            if p > max_p:
                max_p = p
                bn_idx = g

            stats.append({'l': l, 'u': u, 'p': p, 'items': list(items), 'id': g})
        return max_p, bn_idx, stats

    # Iterative improvement
    iter_limit = 150
    for _ in range(iter_limit):
        curr_max, bn, g_stats = get_stats(placement)

        if bn == -1: break

        improved = False
        bn_items = g_stats[bn]['items']

        # Sort targets by pressure (prefer empty/low pressure GPUs)
        targets = sorted(range(gpu_num), key=lambda x: g_stats[x]['p'])

        # Neighborhood 1: Move Item
        # Try moving items from Bottleneck to others
        for m in bn_items:
            w = m.req_rate / m.slo
            s = m.model_size

            # Check source improvement
            src_rem = GPU_MEM_SIZE - (g_stats[bn]['u'] - s)
            if src_rem <= 1e-6: continue
            src_p = (g_stats[bn]['l'] - w) / src_rem
            if src_p >= curr_max - 1e-6: continue

            for dst in targets:
                if dst == bn: continue

                dst_stat = g_stats[dst]
                if dst_stat['u'] + s >= GPU_MEM_SIZE - 1e-6: continue

                dst_p = (dst_stat['l'] + w) / (GPU_MEM_SIZE - (dst_stat['u'] + s))

                # Check global improvement (both source and dest must be < curr_max)
                if dst_p < curr_max - 1e-6:
                    placement[bn].remove(m)
                    placement[dst].append(m)
                    improved = True
                    break
            if improved: break
        if improved: continue

        # Neighborhood 2: Swap Item
        # Try swapping items between Bottleneck and others
        for dst in targets:
            if dst == bn: continue

            dst_items = g_stats[dst]['items']
            for m_bn in bn_items:
                w1, s1 = m_bn.req_rate/m_bn.slo, m_bn.model_size

                for m_dst in dst_items:
                    w2, s2 = m_dst.req_rate/m_dst.slo, m_dst.model_size

                    # New Bottleneck State
                    bn_u = g_stats[bn]['u'] - s1 + s2
                    if bn_u >= GPU_MEM_SIZE - 1e-6: continue
                    bn_p = (g_stats[bn]['l'] - w1 + w2) / (GPU_MEM_SIZE - bn_u)

                    if bn_p >= curr_max - 1e-6: continue

                    # New Dest State
                    dst_u = g_stats[dst]['u'] - s2 + s1
                    if dst_u >= GPU_MEM_SIZE - 1e-6: continue
                    dst_p = (g_stats[dst]['l'] - w2 + w1) / (GPU_MEM_SIZE - dst_u)

                    if dst_p < curr_max - 1e-6:
                        placement[bn].remove(m_bn)
                        placement[bn].append(m_dst)
                        placement[dst].remove(m_dst)
                        placement[dst].append(m_bn)
                        improved = True
                        break
                if improved: break
            if improved: break
        if improved: continue

        # Neighborhood 3: Pairwise Ruin and Recreate (Shuffle)
        # Select Bottleneck and the Lowest Pressure GPU
        partner = targets[0] if targets[0] != bn else targets[1]

        pool = placement[bn] + placement[partner]
        pool_data = []
        for m in pool:
            pool_data.append({'w': m.req_rate/m.slo, 's': m.model_size, 'obj': m})

        # Strategies: standard sorts + random shuffles
        pair_strategies = [
            lambda x: x['w']/x['s'] if x['s'] > 0 else 0,
            lambda x: x['s'],
            'shuffle', 'shuffle', 'shuffle', 'shuffle'
        ]

        best_pair_max = curr_max
        best_cfg = None

        for strat in pair_strategies:
            if strat == 'shuffle':
                rng.shuffle(pool_data)
                curr_sorted = pool_data
            else:
                curr_sorted = sorted(pool_data, key=strat, reverse=True)

            # Simple Best Fit Minimizing Pressure on 2 bins
            l_bins_l = [0.0, 0.0]
            l_bins_u = [0.0, 0.0]
            l_bins_objs = [[], []]
            possible = True

            for item in curr_sorted:
                best_b = -1
                min_p = float('inf')

                for b in [0, 1]:
                    rem = GPU_MEM_SIZE - l_bins_u[b] - item['s']
                    if rem > 1e-6:
                        p = (l_bins_l[b] + item['w']) / rem
                        if p < min_p:
                            min_p = p
                            best_b = b

                if best_b == -1:
                    possible = False
                    break

                l_bins_l[best_b] += item['w']
                l_bins_u[best_b] += item['s']
                l_bins_objs[best_b].append(item['obj'])

            if possible:
                p0 = l_bins_l[0] / (GPU_MEM_SIZE - l_bins_u[0])
                p1 = l_bins_l[1] / (GPU_MEM_SIZE - l_bins_u[1])
                local_max = max(p0, p1)

                if local_max < best_pair_max - 1e-6:
                    best_pair_max = local_max
                    best_cfg = l_bins_objs

        if best_cfg:
            placement[bn] = best_cfg[0]
            placement[partner] = best_cfg[1]
            improved = True

        if not improved: break

    return placement
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