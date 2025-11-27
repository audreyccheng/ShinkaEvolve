# EVOLVE-BLOCK-START
"""Model placement algorithm for minimizing maximum KV cache pressure across GPUs using Binary Search with Multi-Strategy Packing and Local Search Refinement"""

GPU_MEM_SIZE = 80  # GB

def compute_model_placement(gpu_num, models):
    """
    Compute a model placement that minimizes the maximum KVPR across all GPUs.

    Algorithm:
    1. Binary Search on the target max KVPR.
    2. Feasibility check using multiple Bin Packing heuristics (FFD, BFD) on transformed item sizes.
    3. Post-processing local search (moves and swaps) to further reduce the max KVPR.

    Args:
        gpu_num: Number of GPUs
        models: List of models to place

    Returns:
        A placement of models to GPUs {gpu_id: [models]}
    """

    # 1. Quick validation
    total_size = sum(m.model_size for m in models)
    if total_size > gpu_num * GPU_MEM_SIZE:
        raise ValueError("Total model size exceeds total GPU memory capacity.")

    # Prepare items: (req_rate/slo, model_size, model_obj)
    # Store as dictionary/object for cleaner access
    items = []
    for m in models:
        items.append({
            'w': m.req_rate / m.slo,
            's': m.model_size,
            'm': m
        })

    # 2. Binary Search for optimal KVPR
    low = 0.0

    # Heuristic initialization for high bound
    total_w = sum(x['w'] for x in items)
    slack = gpu_num * GPU_MEM_SIZE - total_size
    high = 1.0
    if slack > 1e-4:
        # Lower bound estimate: total_load / total_slack
        # We multiply by 3.0 to be safe and cover fragmentation
        high = max(high, (total_w / slack) * 3.0)
    else:
        # Extremely tight fit
        high = 1000.0

    best_placement = None

    # Find a feasible upper bound first
    feasible_high = False
    for _ in range(20):
        feasible, placement = _check_feasibility_multi(gpu_num, items, high)
        if feasible:
            best_placement = placement
            feasible_high = True
            break
        high *= 2.0
        # If high is increasing rapidly, we can pull up low slightly to narrow range
        low = high / 4.0

    if not feasible_high:
        raise ValueError("Unable to place models. Constraints likely too tight.")

    # Binary Search
    # 25 iterations is sufficient for high precision
    for _ in range(25):
        mid = (low + high) / 2.0
        feasible, placement = _check_feasibility_multi(gpu_num, items, mid)
        if feasible:
            best_placement = placement
            high = mid
        else:
            low = mid

    # Convert list-based placement to dict format
    placement_map = {i: best_placement[i] for i in range(gpu_num)}

    # 3. Refine Placement with Local Search
    # This step tries to smooth out imbalances that the packing heuristic missed
    placement_map = _refine_placement(gpu_num, placement_map)

    return placement_map

def _check_feasibility_multi(gpu_num, items, target_kvpr):
    """
    Check if items can be packed into gpu_num bins given target_kvpr.
    Constraint: sum(w) / (C - sum(s)) <= target_kvpr
    Transformed: sum(w + target_kvpr * s) <= target_kvpr * C
    """
    virtual_cap = target_kvpr * GPU_MEM_SIZE

    # Pre-calculate virtual sizes
    # item: {'w': ..., 's': ..., 'm': ...}
    pack_items = []
    for item in items:
        v = item['w'] + target_kvpr * item['s']
        # Tuple: (virtual_size, physical_size, load, model_obj)
        pack_items.append((v, item['s'], item['w'], item['m']))

    # Strategy 1: FFD on Virtual Size (Standard)
    pack_items.sort(key=lambda x: x[0], reverse=True)
    res = _run_packing_ffd(gpu_num, pack_items, virtual_cap)
    if res: return True, res

    # Strategy 2: BFD on Virtual Size
    # (already sorted by virtual size)
    res = _run_packing_bfd(gpu_num, pack_items, virtual_cap)
    if res: return True, res

    # Strategy 3: FFD on Physical Size
    pack_items.sort(key=lambda x: x[1], reverse=True)
    res = _run_packing_ffd(gpu_num, pack_items, virtual_cap)
    if res: return True, res

    # Strategy 4: FFD on Load (w)
    pack_items.sort(key=lambda x: x[2], reverse=True)
    res = _run_packing_ffd(gpu_num, pack_items, virtual_cap)
    if res: return True, res

    return False, None

def _run_packing_ffd(gpu_num, items, virtual_cap):
    bins_v = [0.0] * gpu_num
    bins_p = [0.0] * gpu_num
    placement = [[] for _ in range(gpu_num)]

    for v, p, w, m in items:
        placed = False
        for i in range(gpu_num):
            if bins_v[i] + v <= virtual_cap + 1e-6 and bins_p[i] + p <= GPU_MEM_SIZE:
                bins_v[i] += v
                bins_p[i] += p
                placement[i].append(m)
                placed = True
                break
        if not placed:
            return None
    return placement

def _run_packing_bfd(gpu_num, items, virtual_cap):
    bins_v = [0.0] * gpu_num
    bins_p = [0.0] * gpu_num
    placement = [[] for _ in range(gpu_num)]

    for v, p, w, m in items:
        best_bin = -1
        min_rem_v = float('inf')

        for i in range(gpu_num):
            if bins_v[i] + v <= virtual_cap + 1e-6 and bins_p[i] + p <= GPU_MEM_SIZE:
                rem_v = virtual_cap - (bins_v[i] + v)
                if rem_v < min_rem_v:
                    min_rem_v = rem_v
                    best_bin = i

        if best_bin != -1:
            bins_v[best_bin] += v
            bins_p[best_bin] += p
            placement[best_bin].append(m)
        else:
            return None
    return placement

def _refine_placement(gpu_num, placement):
    """
    Local search to reduce max KVPR.
    Uses Steepest Descent (Best Improvement) strategy.
    """

    # Cache states: [size, load]
    gpu_states = []
    for i in range(gpu_num):
        s = sum(m.model_size for m in placement[i])
        w = sum(m.req_rate / m.slo for m in placement[i])
        gpu_states.append([s, w])

    # Helper to calculate KVPR
    def calc_kvpr(s, w):
        rem = GPU_MEM_SIZE - s
        if rem <= 1e-5: return float('inf')
        return w / rem

    # Iteration limit
    for _ in range(100):
        # Identify bottleneck
        max_kvpr = -1.0
        max_gpu = -1

        for i in range(gpu_num):
            k = calc_kvpr(gpu_states[i][0], gpu_states[i][1])
            if k > max_kvpr:
                max_kvpr = k
                max_gpu = i

        if max_gpu == -1 or max_kvpr == 0:
            break

        # Find best move or swap
        best_move = None # (type, improvement_magnitude, data)
        best_new_max = max_kvpr

        source_models = placement[max_gpu]

        # 1. Try Moving from max_gpu to others
        for m_idx, m in enumerate(source_models):
            m_s = m.model_size
            m_w = m.req_rate / m.slo

            for dest_gpu in range(gpu_num):
                if dest_gpu == max_gpu: continue

                # Check physical capacity
                if gpu_states[dest_gpu][0] + m_s > GPU_MEM_SIZE:
                    continue

                # New KVPRs
                ns_k = calc_kvpr(gpu_states[max_gpu][0] - m_s, gpu_states[max_gpu][1] - m_w)
                nd_k = calc_kvpr(gpu_states[dest_gpu][0] + m_s, gpu_states[dest_gpu][1] + m_w)

                local_max = max(ns_k, nd_k)
                if local_max < best_new_max - 1e-5:
                    best_new_max = local_max
                    best_move = ('move', m_idx, dest_gpu, m_s, m_w)

        # 2. Try Swapping (only if moves are not super effective or to find better)
        # To save time, we can limit swaps to when moves don't yield huge gains,
        # or just run them always. O(N*M) where N is models on bottleneck, M is total models.
        if True:
            for m1_idx, m1 in enumerate(source_models):
                m1_s = m1.model_size
                m1_w = m1.req_rate / m1.slo

                for other_gpu in range(gpu_num):
                    if other_gpu == max_gpu: continue

                    for m2_idx, m2 in enumerate(placement[other_gpu]):
                        m2_s = m2.model_size
                        m2_w = m2.req_rate / m2.slo

                        ns_s1 = gpu_states[max_gpu][0] - m1_s + m2_s
                        ns_s2 = gpu_states[other_gpu][0] - m2_s + m1_s

                        if ns_s1 > GPU_MEM_SIZE or ns_s2 > GPU_MEM_SIZE:
                            continue

                        k1 = calc_kvpr(ns_s1, gpu_states[max_gpu][1] - m1_w + m2_w)
                        k2 = calc_kvpr(ns_s2, gpu_states[other_gpu][1] - m2_w + m1_w)

                        local_max = max(k1, k2)
                        if local_max < best_new_max - 1e-5:
                            best_new_max = local_max
                            best_move = ('swap', m1_idx, other_gpu, m2_idx, m1_s, m1_w, m2_s, m2_w)

        if best_move:
            if best_move[0] == 'move':
                _, m_idx, dest_gpu, m_s, m_w = best_move
                m = placement[max_gpu].pop(m_idx)
                placement[dest_gpu].append(m)

                gpu_states[max_gpu][0] -= m_s
                gpu_states[max_gpu][1] -= m_w
                gpu_states[dest_gpu][0] += m_s
                gpu_states[dest_gpu][1] += m_w

            elif best_move[0] == 'swap':
                _, m1_idx, other_gpu, m2_idx, m1_s, m1_w, m2_s, m2_w = best_move
                m1 = placement[max_gpu][m1_idx]
                m2 = placement[other_gpu][m2_idx]

                placement[max_gpu][m1_idx] = m2
                placement[other_gpu][m2_idx] = m1

                gpu_states[max_gpu][0] = gpu_states[max_gpu][0] - m1_s + m2_s
                gpu_states[max_gpu][1] = gpu_states[max_gpu][1] - m1_w + m2_w
                gpu_states[other_gpu][0] = gpu_states[other_gpu][0] - m2_s + m1_s
                gpu_states[other_gpu][1] = gpu_states[other_gpu][1] - m2_w + m1_w
        else:
            break

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
