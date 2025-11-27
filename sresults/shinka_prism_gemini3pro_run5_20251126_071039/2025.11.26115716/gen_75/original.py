# EVOLVE-BLOCK-START
"""Model placement algorithm for minimizing maximum KV cache pressure using Robust Binary Search Packing and Steepest Descent with Burst Kicks"""

import copy
import random

GPU_MEM_SIZE = 80.0  # GB

def compute_model_placement(gpu_num, models):
    """
    Minimizes max KVPR using Robust Binary Search Packing (BFD with multiple sorts)
    followed by Steepest Descent Iterated Local Search with Burst Kicks.
    """
    # 1. Validation and Setup
    total_size = sum(m.model_size for m in models)
    if total_size > gpu_num * GPU_MEM_SIZE:
        raise ValueError("Total model size exceeds total GPU memory capacity.")

    # Prepare items: (req_rate/slo, model_size, model_obj)
    items = [{'w': m.req_rate / m.slo, 's': m.model_size, 'm': m} for m in models]

    # 2. Binary Search for Initial Feasible Solution
    total_w = sum(x['w'] for x in items)
    slack = gpu_num * GPU_MEM_SIZE - total_size

    # Heuristic upper bound
    low = 0.0
    if slack < 1e-6:
        high = 1e5 # Tight constraints
    else:
        avg_k = total_w / slack
        high = max(10.0, avg_k * 10.0)

    best_placement = None
    feasible_high = False

    # Exponential search for valid high
    for _ in range(20):
        feasible, placement = _check_feasibility_robust(gpu_num, items, high)
        if feasible:
            best_placement = placement
            feasible_high = True
            break
        low = high
        high *= 2.0

    if not feasible_high:
        raise ValueError("Unable to place models even with high KVPR limit.")

    # Binary Search Refinement
    for _ in range(32):
        mid = (low + high) / 2.0
        feasible, placement = _check_feasibility_robust(gpu_num, items, mid)
        if feasible:
            best_placement = placement
            high = mid
        else:
            low = mid

    placement_map = {i: best_placement[i] for i in range(gpu_num)}

    # 3. Refinement: Steepest Descent with Burst Kicks
    return _steepest_descent_ils(gpu_num, placement_map)

def _check_feasibility_robust(gpu_num, items, K):
    """
    Checks feasibility using multiple sorting heuristics and Best-Fit Decreasing.
    """
    virtual_cap = K * GPU_MEM_SIZE

    # Precompute sort keys
    # (virtual_size, physical_size, load, density, model)
    pack_items = []
    for x in items:
        v = x['w'] + K * x['s']
        d = x['w'] / (x['s'] + 1e-9)
        pack_items.append({'v': v, 's': x['s'], 'w': x['w'], 'd': d, 'm': x['m']})

    # Heuristics: (key_lambda, reverse_bool)
    heuristics = [
        (lambda x: x['v'], True),  # Virtual Size Desc (Standard)
        (lambda x: x['s'], True),  # Physical Size Desc (Big items first)
        (lambda x: x['d'], True),  # Density Desc (High intensity small items first)
        (lambda x: x['w'], True),  # Load Desc
    ]

    for key_func, rev in heuristics:
        pack_items.sort(key=key_func, reverse=rev)
        if res := _pack_bfd(gpu_num, pack_items, virtual_cap):
            return True, res

    # Random shuffle retries for robustness
    # This helps when structured sorts get stuck in local optima
    rng_state = random.getstate()
    for _ in range(5):
        random.shuffle(pack_items)
        if res := _pack_bfd(gpu_num, pack_items, virtual_cap):
            random.setstate(rng_state)
            return True, res
    random.setstate(rng_state)

    return False, None

def _pack_bfd(gpu_num, items, virtual_cap):
    """
    Best Fit Decreasing packing: Minimize residual virtual capacity.
    """
    bins_v = [0.0] * gpu_num
    bins_p = [0.0] * gpu_num
    placement = [[] for _ in range(gpu_num)]

    for item in items:
        best_bin = -1
        min_rem_v = float('inf')

        for i in range(gpu_num):
            if bins_p[i] + item['s'] <= GPU_MEM_SIZE and bins_v[i] + item['v'] <= virtual_cap + 1e-7:
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

def _steepest_descent_ils(gpu_num, placement):
    """
    Refines placement using Steepest Descent and LNS (Ruin & Recreate).
    """
    # State tracking
    gpu_s = [sum(m.model_size for m in placement[i]) for i in range(gpu_num)]
    gpu_w = [sum(m.req_rate / m.slo for m in placement[i]) for i in range(gpu_num)]

    def get_k(i):
        rem = GPU_MEM_SIZE - gpu_s[i]
        if rem <= 1e-7: return 1e9
        return gpu_w[i] / rem

    # Initial State
    current_ks = [get_k(i) for i in range(gpu_num)]
    best_max_k = max(current_ks)
    best_sol = copy.deepcopy(placement)

    max_steps = 1000
    patience = 50
    no_improve = 0

    for _ in range(max_steps):
        # 1. Identify Bottleneck
        # Find all GPUs close to max to avoid cycling on just the first index
        max_k_val = max(current_ks)
        candidates = [i for i, k in enumerate(current_ks) if k > max_k_val - 1e-5]
        src = random.choice(candidates) if candidates else 0
        max_k = current_ks[src]

        # 2. Update Best
        global_max = max(current_ks)
        if global_max < best_max_k - 1e-7:
            best_max_k = global_max
            best_sol = copy.deepcopy(placement)
            no_improve = 0
        else:
            no_improve += 1

        # 3. LNS Ruin & Recreate (if stuck)
        if no_improve > patience:
            # Ruin: Select bottleneck 'src' + k random others
            ruin_set = {src}
            # Pick 3 random others
            others = list(range(gpu_num))
            random.shuffle(others)
            for o in others:
                if len(ruin_set) >= min(gpu_num, 4): break
                if o != src: ruin_set.add(o)

            ruin_indices = list(ruin_set)

            # Backup current state of ruin set
            backup_models = {i: list(placement[i]) for i in ruin_indices}
            backup_s = {i: gpu_s[i] for i in ruin_indices}
            backup_w = {i: gpu_w[i] for i in ruin_indices}

            # Collect models
            repack_models = []
            for i in ruin_indices:
                repack_models.extend(placement[i])
                placement[i] = []
                gpu_s[i] = 0.0
                gpu_w[i] = 0.0

            # Sort for packing (Size Descending is robust)
            repack_models.sort(key=lambda m: m.model_size, reverse=True)

            possible = True
            # Recreate: Best-Fit greedy on local K
            for m in repack_models:
                best_target = -1
                best_local_score = float('inf')

                for r_idx in ruin_indices:
                    if gpu_s[r_idx] + m.model_size <= GPU_MEM_SIZE:
                        # Hypothetical K
                        rem = GPU_MEM_SIZE - (gpu_s[r_idx] + m.model_size)
                        k = (gpu_w[r_idx] + m.req_rate/m.slo) / (rem + 1e-9)
                        if k < best_local_score:
                            best_local_score = k
                            best_target = r_idx

                if best_target != -1:
                    placement[best_target].append(m)
                    gpu_s[best_target] += m.model_size
                    gpu_w[best_target] += m.req_rate/m.slo
                else:
                    possible = False
                    break

            if possible:
                # Update Ks
                for i in ruin_indices:
                    current_ks[i] = get_k(i)
                no_improve = 0 # Reset regardless to explore this new valley
            else:
                # Revert
                for i in ruin_indices:
                    placement[i] = backup_models[i]
                    gpu_s[i] = backup_s[i]
                    gpu_w[i] = backup_w[i]
                    current_ks[i] = get_k(i)

            continue

        # 4. Steepest Descent Evaluation
        best_move = None
        src_models = placement[src]

        # A. Moves
        for i, m in enumerate(src_models):
            s, w = m.model_size, m.req_rate/m.slo
            for dst in range(gpu_num):
                if dst == src: continue
                if gpu_s[dst] + s > GPU_MEM_SIZE: continue

                nk_src = (gpu_w[src] - w) / (GPU_MEM_SIZE - (gpu_s[src] - s) + 1e-9)
                nk_dst = (gpu_w[dst] + w) / (GPU_MEM_SIZE - (gpu_s[dst] + s) + 1e-9)

                local_max = max(nk_src, nk_dst)
                # Allow equal peak moves if they improve variance (tie-breaking)
                if local_max > max_k + 1e-7: continue

                delta_sq = (nk_src**2 + nk_dst**2) - (current_ks[src]**2 + current_ks[dst]**2)
                metric = (local_max, delta_sq)

                if best_move is None or metric < best_move[1]:
                    best_move = (('move', i, dst, s, w), metric)

        # B. Swaps
        for i1, m1 in enumerate(src_models):
            s1, w1 = m1.model_size, m1.req_rate/m1.slo
            for dst in range(gpu_num):
                if dst == src: continue
                if current_ks[dst] > max_k * 0.95: continue

                for i2, m2 in enumerate(placement[dst]):
                    s2, w2 = m2.model_size, m2.req_rate/m2.slo

                    ns_src = gpu_s[src] - s1 + s2
                    ns_dst = gpu_s[dst] - s2 + s1
                    if ns_src > GPU_MEM_SIZE or ns_dst > GPU_MEM_SIZE: continue

                    nk_src = (gpu_w[src] - w1 + w2) / (GPU_MEM_SIZE - ns_src + 1e-9)
                    nk_dst = (gpu_w[dst] - w2 + w1) / (GPU_MEM_SIZE - ns_dst + 1e-9)

                    local_max = max(nk_src, nk_dst)
                    # Allow equal peak moves if they improve variance (tie-breaking)
                    if local_max > max_k + 1e-7: continue

                    delta_sq = (nk_src**2 + nk_dst**2) - (current_ks[src]**2 + current_ks[dst]**2)
                    metric = (local_max, delta_sq)

                    if best_move is None or metric < best_move[1]:
                        best_move = (('swap', i1, dst, i2, s1, w1, s2, w2), metric)

        if best_move:
            action, metric = best_move

            is_better = False
            if metric[0] < max_k - 1e-7:
                is_better = True
            elif metric[0] < max_k + 1e-7 and metric[1] < -1e-5:
                is_better = True

            if is_better:
                if action[0] == 'move':
                    _, i, dst, s, w = action
                    m = placement[src].pop(i)
                    placement[dst].append(m)
                    gpu_s[src] -= s; gpu_w[src] -= w
                    gpu_s[dst] += s; gpu_w[dst] += w
                else:
                    _, i1, dst, i2, s1, w1, s2, w2 = action
                    m1 = placement[src][i1]
                    m2 = placement[dst][i2]
                    placement[src][i1] = m2
                    placement[dst][i2] = m1
                    gpu_s[src] = gpu_s[src] - s1 + s2
                    gpu_w[src] = gpu_w[src] - w1 + w2
                    gpu_s[dst] = gpu_s[dst] - s2 + s1
                    gpu_w[dst] = gpu_w[dst] - w2 + w1

                current_ks[src] = get_k(src)
                current_ks[dst] = get_k(dst)
                no_improve = 0
            else:
                no_improve += 1
        else:
            no_improve += 1

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