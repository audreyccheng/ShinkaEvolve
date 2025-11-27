# EVOLVE-BLOCK-START
"""Model placement algorithm for minimizing maximum KV cache pressure using Robust Binary Search with BFD and Variance-Aware Local Search"""

import copy
import random
import math

GPU_MEM_SIZE = 80.0  # GB

def compute_model_placement(gpu_num, models):
    """
    Minimizes max KVPR using Robust Binary Search Packing (BFD/FFD)
    followed by Variance-Aware Iterated Local Search.
    """
    # 1. Validation
    total_size = sum(m.model_size for m in models)
    if total_size > gpu_num * GPU_MEM_SIZE:
        raise ValueError("Total model size exceeds total GPU memory capacity.")

    # Prepare items: (w, s, m)
    # w = req_rate / slo
    items = [{'w': m.req_rate / m.slo, 's': m.model_size, 'm': m} for m in models]

    # 2. Binary Search for Initial Solution
    total_w = sum(x['w'] for x in items)
    slack = gpu_num * GPU_MEM_SIZE - total_size

    # Heuristic bound initialization
    low = 0.0
    if slack < 1e-6:
        high = 10000.0
    else:
        avg_k = total_w / slack
        high = max(10.0, avg_k * 8.0)

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
        raise ValueError("Unable to place models. Constraints too tight.")

    # Binary Search
    for _ in range(32): # High precision
        mid = (low + high) / 2.0
        feasible, placement = _check_feasibility_robust(gpu_num, items, mid)
        if feasible:
            best_placement = placement
            high = mid
        else:
            low = mid

    # Convert list placement to dictionary map
    placement_map = {i: best_placement[i] for i in range(gpu_num)}

    # 3. Iterated Local Search Refinement
    final_placement = _variance_aware_ils(gpu_num, placement_map)

    return final_placement

def _check_feasibility_robust(gpu_num, items, K):
    """
    Checks feasibility using multiple sorting strategies and packing algorithms.
    """
    virtual_cap = K * GPU_MEM_SIZE

    # Create augmented items for sorting
    pack_items = []
    for x in items:
        v = x['w'] + K * x['s']
        d = x['w'] / (x['s'] + 1e-7)
        pack_items.append({'v': v, 's': x['s'], 'w': x['w'], 'd': d, 'm': x['m']})

    # Heuristics: (sort_key, reverse)
    heuristics = [
        (lambda x: x['v'], True),  # Virtual Size Desc
        (lambda x: x['s'], True),  # Physical Size Desc
        (lambda x: x['d'], True),  # Density Desc
    ]

    for key_func, rev in heuristics:
        sorted_items = sorted(pack_items, key=key_func, reverse=rev)

        # Try BFD (Best Fit Decreasing)
        if res := _pack_bfd(gpu_num, sorted_items, virtual_cap):
            return True, res

        # Try FFD (First Fit Decreasing) - fallback
        if res := _pack_ffd(gpu_num, sorted_items, virtual_cap):
            return True, res

    return False, None

def _pack_bfd(gpu_num, items, virtual_cap):
    """Best Fit Decreasing Packing: Minimize remaining virtual capacity."""
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

def _pack_ffd(gpu_num, items, virtual_cap):
    """First Fit Decreasing Packing."""
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

def _variance_aware_ils(gpu_num, placement):
    """
    Iterated Local Search with Best-Improvement and Variance Tie-Breaking.
    """
    # State tracking
    gpu_s = [sum(m.model_size for m in placement[i]) for i in range(gpu_num)]
    gpu_w = [sum(m.req_rate / m.slo for m in placement[i]) for i in range(gpu_num)]

    def get_k(idx):
        rem = GPU_MEM_SIZE - gpu_s[idx]
        if rem <= 1e-7: return 1e9
        return gpu_w[idx] / rem

    current_ks = [get_k(i) for i in range(gpu_num)]

    best_max_k = max(current_ks)
    best_sol = copy.deepcopy(placement)

    max_steps = 300
    patience = 30
    no_improve = 0

    for step in range(max_steps):
        # 1. Identify bottleneck
        max_k = -1.0
        src = -1
        sum_sq = 0.0
        for i in range(gpu_num):
            k = current_ks[i]
            if k > max_k:
                max_k = k
                src = i
            sum_sq += k*k

        # Check global best
        if max_k < best_max_k - 1e-7:
            best_max_k = max_k
            best_sol = copy.deepcopy(placement)
            no_improve = 0
        else:
            no_improve += 1

        # 2. Kick if stuck
        if no_improve > patience:
            # Burst Kick: 3 random moves
            for _ in range(3):
                s = random.randint(0, gpu_num - 1)
                if not placement[s]: continue
                d = random.randint(0, gpu_num - 1)
                if s == d: continue

                m_idx = random.randint(0, len(placement[s]) - 1)
                m = placement[s][m_idx]
                if gpu_s[d] + m.model_size <= GPU_MEM_SIZE:
                    placement[d].append(m)
                    placement[s].pop(m_idx)
                    gpu_s[d] += m.model_size
                    gpu_w[d] += m.req_rate/m.slo
                    gpu_s[s] -= m.model_size
                    gpu_w[s] -= m.req_rate/m.slo
                    current_ks[d] = get_k(d)
                    current_ks[s] = get_k(s)

            no_improve = 0 # Reset patience
            continue

        # 3. Best-Improvement Descent
        best_move = None
        best_eval = (max_k, sum_sq)

        models = placement[src]

        # Try Moves
        for i, m in enumerate(models):
            w, s = m.req_rate/m.slo, m.model_size
            for dst in range(gpu_num):
                if dst == src: continue
                if gpu_s[dst] + s > GPU_MEM_SIZE: continue

                # Predict
                rem_src = GPU_MEM_SIZE - (gpu_s[src] - s)
                nk_src = (gpu_w[src] - w) / rem_src if rem_src > 1e-7 else 1e9

                rem_dst = GPU_MEM_SIZE - (gpu_s[dst] + s)
                nk_dst = (gpu_w[dst] + w) / rem_dst if rem_dst > 1e-7 else 1e9

                local_peak = max(nk_src, nk_dst)
                if local_peak > max_k + 1e-7: continue

                # New variance
                delta_sq = (nk_src**2 + nk_dst**2) - (current_ks[src]**2 + current_ks[dst]**2)
                new_sum_sq = sum_sq + delta_sq

                # Acceptance
                if local_peak < best_eval[0] - 1e-7:
                    best_eval = (local_peak, new_sum_sq)
                    best_move = ('move', i, dst, s, w)
                elif abs(local_peak - best_eval[0]) < 1e-7:
                    if new_sum_sq < best_eval[1] - 1e-5:
                        best_eval = (local_peak, new_sum_sq)
                        best_move = ('move', i, dst, s, w)

        # Try Swaps
        if True:
            for i1, m1 in enumerate(models):
                w1, s1 = m1.req_rate/m1.slo, m1.model_size
                for dst in range(gpu_num):
                    if dst == src: continue
                    # Optimization: Skip if dst is too unloaded (moving won't help global peak much if swapped with small?)
                    # No, swapping might bring a large model back to src? No, we want to offload src.
                    # We want to swap a large item from src with a small item from dst?
                    # Let's just check all for correctness.

                    for i2, m2 in enumerate(placement[dst]):
                        w2, s2 = m2.req_rate/m2.slo, m2.model_size

                        ns_src = gpu_s[src] - s1 + s2
                        ns_dst = gpu_s[dst] - s2 + s1
                        if ns_src > GPU_MEM_SIZE or ns_dst > GPU_MEM_SIZE: continue

                        rem_src = GPU_MEM_SIZE - ns_src
                        nk_src = (gpu_w[src] - w1 + w2) / rem_src if rem_src > 1e-7 else 1e9

                        rem_dst = GPU_MEM_SIZE - ns_dst
                        nk_dst = (gpu_w[dst] - w2 + w1) / rem_dst if rem_dst > 1e-7 else 1e9

                        local_peak = max(nk_src, nk_dst)
                        if local_peak > max_k + 1e-7: continue

                        delta_sq = (nk_src**2 + nk_dst**2) - (current_ks[src]**2 + current_ks[dst]**2)
                        new_sum_sq = sum_sq + delta_sq

                        if local_peak < best_eval[0] - 1e-7:
                            best_eval = (local_peak, new_sum_sq)
                            best_move = ('swap', i1, dst, i2, s1, w1, s2, w2)
                        elif abs(local_peak - best_eval[0]) < 1e-7:
                            if new_sum_sq < best_eval[1] - 1e-5:
                                best_eval = (local_peak, new_sum_sq)
                                best_move = ('swap', i1, dst, i2, s1, w1, s2, w2)

        # Apply Best Move
        if best_move:
            if best_move[0] == 'move':
                _, i, dst, s, w = best_move
                m = placement[src].pop(i)
                placement[dst].append(m)
                gpu_s[src] -= s; gpu_w[src] -= w
                gpu_s[dst] += s; gpu_w[dst] += w
            elif best_move[0] == 'swap':
                _, i1, dst, i2, s1, w1, s2, w2 = best_move
                m1 = placement[src][i1]
                m2 = placement[dst][i2]
                placement[src][i1] = m2
                placement[dst][i2] = m1
                gpu_s[src] = gpu_s[src] - s1 + s2
                gpu_w[src] = gpu_w[src] - w1 + w2
                gpu_s[dst] = gpu_s[dst] - s2 + s1
                gpu_w[dst] = gpu_w[dst] - w2 + w1

            # Update cache
            current_ks[src] = get_k(src)
            current_ks[dst] = get_k(dst)
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