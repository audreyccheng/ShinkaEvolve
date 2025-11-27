# EVOLVE-BLOCK-START
"""Model placement algorithm for minimizing maximum KV cache pressure across GPUs using Binary Search with Multi-Strategy Packing and Local Search Refinement"""

import copy
import random

GPU_MEM_SIZE = 80.0  # GB

def compute_model_placement(gpu_num, models):
    """
    Minimizes max KVPR using Robust Binary Search with multiple packing heuristics
    followed by Iterated Local Search (Ruin & Recreate).
    """
    # 1. Validation
    total_size = sum(m.model_size for m in models)
    if total_size > gpu_num * GPU_MEM_SIZE:
        raise ValueError("Total model size exceeds total GPU memory capacity.")

    # Prepare items: (w, s, m)
    items = [{'w': m.req_rate / m.slo, 's': m.model_size, 'm': m} for m in models]

    # 2. Binary Search
    total_w = sum(x['w'] for x in items)
    slack = gpu_num * GPU_MEM_SIZE - total_size

    low = 0.0
    if slack < 1e-6:
        high = 10000.0
    else:
        avg_pressure = total_w / slack
        high = max(10.0, avg_pressure * 8.0)

    best_placement = None
    feasible_high = False

    # Find valid upper bound
    for _ in range(20):
        feasible, placement = _check_feasibility_robust(gpu_num, items, high)
        if feasible:
            best_placement = placement
            feasible_high = True
            break
        low = high
        high *= 2.0

    if not feasible_high:
        raise ValueError("Unable to place models. Constraints likely too tight.")

    # Binary Search
    for _ in range(30):
        mid = (low + high) / 2.0
        feasible, placement = _check_feasibility_robust(gpu_num, items, mid)
        if feasible:
            best_placement = placement
            high = mid
        else:
            low = mid

    placement_map = {i: best_placement[i] for i in range(gpu_num)}

    # 3. Iterated Local Search Refinement
    return _iterated_local_search(gpu_num, placement_map)

def _check_feasibility_robust(gpu_num, items, K):
    """
    Check feasibility using multiple sorting strategies and packing algorithms.
    """
    virtual_cap = K * GPU_MEM_SIZE
    # Augment items for sorting
    pack_items = []
    for x in items:
        v = x['w'] + K * x['s']
        d = x['w'] / (x['s'] + 1e-7)
        pack_items.append({'v': v, 's': x['s'], 'w': x['w'], 'd': d, 'm': x['m']})

    # Heuristics: (key_lambda, reverse)
    heuristics = [
        (lambda x: x['v'], True),  # Virtual Size Desc
        (lambda x: x['s'], True),  # Physical Size Desc
        (lambda x: x['d'], True),  # Density Desc
        (lambda x: x['w'], True),  # Load Desc
    ]

    for key_func, rev in heuristics:
        sorted_items = sorted(pack_items, key=key_func, reverse=rev)

        # Try Best Fit Decreasing (usually better for tight bins)
        if res := _pack_bfd(gpu_num, sorted_items, virtual_cap):
            return True, res

        # Try First Fit Decreasing
        if res := _pack_ffd(gpu_num, sorted_items, virtual_cap):
            return True, res

    return False, None

def _pack_ffd(gpu_num, items, virtual_cap):
    bins_v = [0.0] * gpu_num
    bins_p = [0.0] * gpu_num
    placement = [[] for _ in range(gpu_num)]

    for item in items:
        placed = False
        for i in range(gpu_num):
            if bins_p[i] + item['s'] <= GPU_MEM_SIZE and bins_v[i] + item['v'] <= virtual_cap + 1e-7:
                bins_p[i] += item['s']
                bins_v[i] += item['v']
                placement[i].append(item['m'])
                placed = True
                break
        if not placed: return None
    return placement

def _pack_bfd(gpu_num, items, virtual_cap):
    bins_v = [0.0] * gpu_num
    bins_p = [0.0] * gpu_num
    placement = [[] for _ in range(gpu_num)]

    for item in items:
        best_bin = -1
        min_rem = float('inf')

        for i in range(gpu_num):
            if bins_p[i] + item['s'] <= GPU_MEM_SIZE and bins_v[i] + item['v'] <= virtual_cap + 1e-7:
                rem = virtual_cap - (bins_v[i] + item['v'])
                if rem < min_rem:
                    min_rem = rem
                    best_bin = i

        if best_bin != -1:
            bins_p[best_bin] += item['s']
            bins_v[best_bin] += item['v']
            placement[best_bin].append(item['m'])
        else:
            return None
    return placement

def _iterated_local_search(gpu_num, placement):
    """
    Refines placement using Ruin and Recreate.
    """
    # Initialize state
    gpu_s = [sum(m.model_size for m in placement[i]) for i in range(gpu_num)]
    gpu_w = [sum(m.req_rate / m.slo for m in placement[i]) for i in range(gpu_num)]

    def get_k(i):
        rem = GPU_MEM_SIZE - gpu_s[i]
        if rem <= 1e-7: return 1e9
        return gpu_w[i] / rem

    best_max_k = max(get_k(i) for i in range(gpu_num))
    best_placement = copy.deepcopy(placement)

    max_steps = 300
    patience = 30
    no_improve = 0

    for _ in range(max_steps):
        # Identify bottleneck
        current_ks = [get_k(i) for i in range(gpu_num)]
        max_k = max(current_ks)

        # Check global improvement
        if max_k < best_max_k - 1e-6:
            best_max_k = max_k
            best_placement = copy.deepcopy(placement)
            no_improve = 0
        else:
            no_improve += 1

        # Ruin & Recreate
        # Select bottleneck and random others
        candidates = [i for i, k in enumerate(current_ks) if k > max_k * 0.99]
        if not candidates: candidates = [random.randint(0, gpu_num-1)]
        src = random.choice(candidates)

        ruin_set = {src}
        n_ruin = 2 + (1 if no_improve > patience else 0)

        others = list(range(gpu_num))
        random.shuffle(others)
        for o in others:
            if len(ruin_set) >= min(gpu_num, n_ruin): break
            if o != src: ruin_set.add(o)

        # Ruin
        removed_models = []
        backup = {}
        for idx in ruin_set:
            backup[idx] = (list(placement[idx]), gpu_s[idx], gpu_w[idx])
            removed_models.extend(placement[idx])
            placement[idx] = []
            gpu_s[idx] = 0.0
            gpu_w[idx] = 0.0

        # Recreate (Greedy Best Fit)
        # Sort by physical size desc to pack well
        removed_models.sort(key=lambda m: m.model_size, reverse=True)

        feasible = True
        for m in removed_models:
            best_idx = -1
            best_score = float('inf')

            m_s = m.model_size
            m_w = m.req_rate / m.slo

            for idx in ruin_set:
                if gpu_s[idx] + m_s <= GPU_MEM_SIZE:
                    # Score: minimizing resulting local K
                    rem = GPU_MEM_SIZE - (gpu_s[idx] + m_s)
                    k = (gpu_w[idx] + m_w) / rem if rem > 1e-7 else 1e9
                    if k < best_score:
                        best_score = k
                        best_idx = idx

            if best_idx != -1:
                placement[best_idx].append(m)
                gpu_s[best_idx] += m_s
                gpu_w[best_idx] += m_w
            else:
                feasible = False
                break

        # Acceptance
        accept = False
        if feasible:
            new_ks = [get_k(i) for i in range(gpu_num)]
            new_max = max(new_ks)
            if new_max < max_k - 1e-6:
                accept = True
            elif new_max < max_k + 1e-6:
                # Accept equal moves to traverse plateau
                accept = True
            elif no_improve > patience and random.random() < 0.2:
                # Random walk
                accept = True

        if not accept:
            # Revert
            for idx in ruin_set:
                placement[idx], gpu_s[idx], gpu_w[idx] = backup[idx]

    return best_placement
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