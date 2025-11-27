# EVOLVE-BLOCK-START
"""Model placement algorithm for minimizing maximum KV cache pressure using Multi-Strategy Packing and Iterated Local Search"""

import copy
import random

GPU_MEM_SIZE = 80  # GB

def compute_model_placement(gpu_num, models):
    """
    Minimizes max KVPR using Binary Search with multiple packing heuristics
    followed by Iterated Local Search for refinement.
    """
    total_size = sum(m.model_size for m in models)
    if total_size > gpu_num * GPU_MEM_SIZE:
        raise ValueError("Total model size exceeds total GPU memory capacity.")

    # Prepare items for packing: (w, s, m)
    # w = req_rate / slo
    items = [{'w': m.req_rate / m.slo, 's': m.model_size, 'm': m} for m in models]

    # Binary Search for optimal KVPR
    total_w = sum(x['w'] for x in items)
    slack = gpu_num * GPU_MEM_SIZE - total_size
    low = 0.0
    # Heuristic upper bound
    high = (total_w / slack) * 4.0 if slack > 1e-4 else 1000.0
    high = max(high, 10.0)

    best_packing_placement = None
    feasible_high = False

    # Find valid upper bound
    for _ in range(20):
        feasible, placement = _check_feasibility_multi(gpu_num, items, high)
        if feasible:
            best_packing_placement = placement
            feasible_high = True
            break
        low = high
        high *= 2.0

    if not feasible_high:
        raise ValueError("Unable to place models. Constraints likely too tight.")

    # Binary Search
    for _ in range(30):
        mid = (low + high) / 2.0
        feasible, placement = _check_feasibility_multi(gpu_num, items, mid)
        if feasible:
            best_packing_placement = placement
            high = mid
        else:
            low = mid

    # Convert list-based placement to dict format
    placement_map = {i: best_packing_placement[i] for i in range(gpu_num)}

    # Refinement using Large Neighborhood Search (Ruin and Recreate)
    final_placement = _lns_refinement(gpu_num, placement_map)

    return final_placement

def _check_feasibility_multi(gpu_num, items, K):
    """
    Check if items can be packed with target KVPR 'K' using multiple heuristics.
    """
    virtual_cap = K * GPU_MEM_SIZE

    # Prepare items with sorting keys
    pack_items = []
    for x in items:
        v = x['w'] + K * x['s']
        d = x['w'] / (x['s'] + 1e-7)
        pack_items.append({'v': v, 's': x['s'], 'w': x['w'], 'd': d, 'm': x['m']})

    # Heuristics: (Sort Key Lambda, Reverse)
    # 1. Virtual Size Desc (Standard)
    # 2. Physical Size Desc (Fit large items first)
    # 3. Density Desc (Fit high load/size items first)
    heuristics = [
        (lambda x: x['v'], True),
        (lambda x: x['s'], True),
        (lambda x: x['d'], True),
        (lambda x: x['w'], True),
    ]

    for key_func, rev in heuristics:
        sorted_items = sorted(pack_items, key=key_func, reverse=rev)
        # Use Best Fit Decreasing (BFD)
        if res := _pack_bfd(gpu_num, sorted_items, virtual_cap):
            return True, res

    return False, None

def _pack_bfd(gpu_num, items, virtual_cap):
    """Best Fit Decreasing Packing"""
    bins_v = [0.0] * gpu_num
    bins_p = [0.0] * gpu_num
    placement = [[] for _ in range(gpu_num)]

    for item in items:
        best_bin = -1
        min_rem_v = float('inf')

        for i in range(gpu_num):
            if bins_p[i] + item['s'] <= GPU_MEM_SIZE and bins_v[i] + item['v'] <= virtual_cap + 1e-7:
                # Minimize remaining virtual capacity
                rem = virtual_cap - (bins_v[i] + item['v'])
                if rem < min_rem_v:
                    min_rem_v = rem
                    best_bin = i

        if best_bin != -1:
            bins_p[best_bin] += item['s']
            bins_v[best_bin] += item['v']
            placement[best_bin].append(item['m'])
        else:
            return None
    return placement

def _lns_refinement(gpu_num, placement):
    """
    Refines placement using Ruin and Recreate (LNS).
    """
    gpu_s = [sum(m.model_size for m in placement[i]) for i in range(gpu_num)]
    gpu_w = [sum(m.req_rate / m.slo for m in placement[i]) for i in range(gpu_num)]

    def get_k(i):
        rem = GPU_MEM_SIZE - gpu_s[i]
        if rem <= 1e-7: return 1e9
        return gpu_w[i] / rem

    best_max_k = max(get_k(i) for i in range(gpu_num))
    best_sol = copy.deepcopy(placement)

    max_steps = 500
    patience = 40
    no_improve = 0

    for _ in range(max_steps):
        # Identify bottleneck
        current_ks = [get_k(i) for i in range(gpu_num)]
        max_k = max(current_ks)

        if max_k < best_max_k - 1e-6:
            best_max_k = max_k
            best_sol = copy.deepcopy(placement)
            no_improve = 0
        else:
            no_improve += 1

        # Select Ruin Targets
        # Bottleneck + Random
        candidates = [i for i, k in enumerate(current_ks) if k > max_k * 0.98]
        if not candidates: candidates = [random.randint(0, gpu_num-1)]
        src = random.choice(candidates)

        ruin_set = {src}
        # Dynamic ruin size
        n_ruin = 2
        if no_improve > patience // 2: n_ruin = 3
        if no_improve > patience: n_ruin = 4

        others = list(range(gpu_num))
        random.shuffle(others)
        for o in others:
            if len(ruin_set) >= min(gpu_num, n_ruin): break
            if o != src: ruin_set.add(o)

        # Ruin
        removed_models = []
        backup = {}
        for r in ruin_set:
            backup[r] = (list(placement[r]), gpu_s[r], gpu_w[r])
            removed_models.extend(placement[r])
            placement[r] = []
            gpu_s[r] = 0.0
            gpu_w[r] = 0.0

        # Recreate: Greedy Best Fit on Local K
        # Sort large models first to pack better
        removed_models.sort(key=lambda m: (m.model_size, m.req_rate/m.slo), reverse=True)

        feasible = True
        for m in removed_models:
            best_t = -1
            best_k = float('inf')

            for t in ruin_set:
                if gpu_s[t] + m.model_size <= GPU_MEM_SIZE:
                    rem = GPU_MEM_SIZE - (gpu_s[t] + m.model_size)
                    k = (gpu_w[t] + m.req_rate/m.slo) / rem if rem > 1e-7 else 1e9
                    if k < best_k:
                        best_k = k
                        best_t = t

            if best_t != -1:
                placement[best_t].append(m)
                gpu_s[best_t] += m.model_size
                gpu_w[best_t] += m.req_rate/m.slo
            else:
                feasible = False
                break

        if feasible:
            # Check acceptance
            new_max = max(get_k(i) for i in range(gpu_num))
            accept = False

            if new_max < max_k - 1e-6:
                accept = True
            elif new_max < max_k + 1e-6:
                # Accept neutral moves to traverse plateau
                accept = True
            elif no_improve > patience and random.random() < 0.1:
                # Escape local optima
                accept = True

            if not accept:
                # Revert
                for r in ruin_set:
                    placement[r], gpu_s[r], gpu_w[r] = backup[r]
        else:
            # Revert
            for r in ruin_set:
                placement[r], gpu_s[r], gpu_w[r] = backup[r]

    return best_sol
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
