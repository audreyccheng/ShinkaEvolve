# EVOLVE-BLOCK-START
"""Model placement algorithm for minimizing maximum KV cache pressure using Robust Packing and Pressure Relief Local Search"""

import copy
import random

GPU_MEM_SIZE = 80  # GB

def compute_model_placement(gpu_num, models):
    """
    Minimizes max KVPR using Robust Binary Search Packing with Best-Fit Decreasing
    followed by Steepest Descent Iterated Local Search.
    """
    # 1. Validation and Setup
    total_size = sum(m.model_size for m in models)
    if total_size > gpu_num * GPU_MEM_SIZE:
        raise ValueError("Total model size exceeds total GPU memory capacity.")

    # Pre-calculate model weights for efficiency
    # item: (req_rate/slo, model_size, model_obj)
    items = [(m.req_rate / m.slo, m.model_size, m) for m in models]

    # 2. Binary Search for Initial Feasible Solution
    total_w = sum(x[0] for x in items)
    slack = gpu_num * GPU_MEM_SIZE - total_size

    if slack < 1e-5:
        low, high = 0.0, 1e6
    else:
        avg_k = total_w / slack
        low, high = avg_k * 0.5, avg_k * 5.0

    high = max(high, 100.0)

    best_placement = None
    feasible_high = False

    for _ in range(10):
        feasible, placement = _check_feasibility_robust(gpu_num, items, high)
        if feasible:
            best_placement = placement
            feasible_high = True
            break
        low = high
        high *= 2.0

    if not feasible_high:
        raise ValueError("Unable to place models even with high KVPR limit.")

    for _ in range(30):
        mid = (low + high) / 2.0
        feasible, placement = _check_feasibility_robust(gpu_num, items, mid)
        if feasible:
            best_placement = placement
            high = mid
        else:
            low = mid

    placement_map = {i: best_placement[i] for i in range(gpu_num)}

    # 3. Steepest Descent ILS with Burst Kicks
    return _steepest_descent_ils(gpu_num, placement_map)

def _check_feasibility_robust(gpu_num, items, K):
    """
    Checks feasibility using multiple sorting heuristics and Best-Fit Decreasing (BFD).
    """
    virtual_cap = K * GPU_MEM_SIZE
    pack_items = []
    for w, s, m in items:
        v = w + K * s
        pack_items.append((v, s, w, m))

    # Strategies: (key_lambda, reverse_bool)
    heuristics = [
        (lambda x: x[0], True), # Virtual Size Desc
        (lambda x: x[1], True), # Physical Size Desc
        (lambda x: x[2], True), # Load Desc
        (lambda x: x[2]/(x[1]+1e-7), True), # Density Desc
    ]

    for key, rev in heuristics:
        pack_items.sort(key=key, reverse=rev)
        if res := _pack_bfd(gpu_num, pack_items, virtual_cap):
            return True, res
    return False, None

def _pack_bfd(gpu_num, items, v_cap):
    """
    Best Fit Decreasing packing: Places item in bin with minimum sufficient residual capacity.
    """
    bins_v = [0.0] * gpu_num
    bins_p = [0.0] * gpu_num
    placement = [[] for _ in range(gpu_num)]

    for v, s, w, m in items:
        best_bin = -1
        min_rem_v = float('inf')

        for i in range(gpu_num):
            if bins_p[i] + s <= GPU_MEM_SIZE and bins_v[i] + v <= v_cap + 1e-7:
                rem = v_cap - (bins_v[i] + v)
                if rem < min_rem_v:
                    min_rem_v = rem
                    best_bin = i

        if best_bin != -1:
            bins_p[best_bin] += s
            bins_v[best_bin] += v
            placement[best_bin].append(m)
        else:
            return None
    return placement

def _steepest_descent_ils(gpu_num, placement):
    """
    Refines placement using Steepest Descent (Best Improvement) and Burst Kicks.
    """
    gpu_s = [sum(m.model_size for m in placement[i]) for i in range(gpu_num)]
    gpu_w = [sum(m.req_rate / m.slo for m in placement[i]) for i in range(gpu_num)]

    def get_k(idx):
        rem = GPU_MEM_SIZE - gpu_s[idx]
        if rem <= 1e-7: return 1e9
        return gpu_w[idx] / rem

    best_max_k = max(get_k(i) for i in range(gpu_num))
    best_sol = copy.deepcopy(placement)

    max_steps = 400
    patience = 20
    no_improve = 0

    for step in range(max_steps):
        # Identify bottleneck
        max_k = -1.0
        src = -1
        sum_sq = 0.0

        for i in range(gpu_num):
            k = get_k(i)
            if k > max_k:
                max_k = k
                src = i
            sum_sq += k*k

        if max_k < best_max_k - 1e-7:
            best_max_k = max_k
            best_sol = copy.deepcopy(placement)
            no_improve = 0
        else:
            no_improve += 1

        if no_improve > patience:
            # Burst Kick: Multiple random moves
            moves_made = 0
            for _ in range(5):
                for _ in range(10): # retry limit
                    s_idx = random.randint(0, gpu_num - 1)
                    if not placement[s_idx]: continue
                    d_idx = random.randint(0, gpu_num - 1)
                    if s_idx == d_idx: continue
                    m_idx = random.randint(0, len(placement[s_idx]) - 1)
                    m = placement[s_idx][m_idx]

                    if gpu_s[d_idx] + m.model_size <= GPU_MEM_SIZE:
                        placement[d_idx].append(m)
                        placement[s_idx].pop(m_idx)
                        gpu_s[d_idx] += m.model_size; gpu_w[d_idx] += m.req_rate/m.slo
                        gpu_s[s_idx] -= m.model_size; gpu_w[s_idx] -= m.req_rate/m.slo
                        moves_made += 1
                        break
            if moves_made > 0:
                no_improve = 0
            continue

        # Steepest Descent
        best_move = None # (type, data)
        best_improvement = -1e9
        best_var = sum_sq

        models = placement[src]

        # Check moves
        for i, m in enumerate(models):
            w, s = m.req_rate/m.slo, m.model_size
            for dst in range(gpu_num):
                if dst == src: continue
                if gpu_s[dst] + s > GPU_MEM_SIZE: continue

                rem_src = GPU_MEM_SIZE - (gpu_s[src] - s)
                nk_src = (gpu_w[src] - w) / rem_src if rem_src > 1e-7 else 1e9
                rem_dst = GPU_MEM_SIZE - (gpu_s[dst] + s)
                nk_dst = (gpu_w[dst] + w) / rem_dst if rem_dst > 1e-7 else 1e9

                new_peak = max(nk_src, nk_dst)

                # Check strict improvement or equal peak with better variance
                if new_peak < max_k - 1e-7:
                    imp = max_k - new_peak
                    if imp > best_improvement + 1e-7:
                        best_improvement = imp
                        best_move = ('move', i, dst, s, w)
                        # Estimate new var roughly? Or just ignore variance if peak improves significantly

                # Tie breaking with variance
                elif abs(new_peak - max_k) < 1e-7:
                    old_k_src = get_k(src)
                    old_k_dst = get_k(dst)
                    new_sum_sq = sum_sq - old_k_src**2 - old_k_dst**2 + nk_src**2 + nk_dst**2
                    if new_sum_sq < best_var - 1e-5:
                         best_var = new_sum_sq
                         best_move = ('move', i, dst, s, w)

        # Check swaps only if move didn't yield a huge improvement (optimization)
        if True:
            for i1, m1 in enumerate(models):
                w1, s1 = m1.req_rate/m1.slo, m1.model_size
                for dst in range(gpu_num):
                    if dst == src: continue
                    if get_k(dst) > max_k * 0.95: continue

                    for i2, m2 in enumerate(placement[dst]):
                        w2, s2 = m2.req_rate/m2.slo, m2.model_size
                        ns_src = gpu_s[src] - s1 + s2
                        ns_dst = gpu_s[dst] - s2 + s1
                        if ns_src > GPU_MEM_SIZE or ns_dst > GPU_MEM_SIZE: continue

                        rem_src = GPU_MEM_SIZE - ns_src
                        nk_src = (gpu_w[src] - w1 + w2) / rem_src if rem_src > 1e-7 else 1e9
                        rem_dst = GPU_MEM_SIZE - ns_dst
                        nk_dst = (gpu_w[dst] - w2 + w1) / rem_dst if rem_dst > 1e-7 else 1e9

                        new_peak = max(nk_src, nk_dst)

                        if new_peak < max_k - 1e-7:
                            imp = max_k - new_peak
                            if imp > best_improvement + 1e-7:
                                best_improvement = imp
                                best_move = ('swap', i1, dst, i2, s1, w1, s2, w2)
                        elif abs(new_peak - max_k) < 1e-7:
                            old_k_src = get_k(src)
                            old_k_dst = get_k(dst)
                            new_sum_sq = sum_sq - old_k_src**2 - old_k_dst**2 + nk_src**2 + nk_dst**2
                            if new_sum_sq < best_var - 1e-5:
                                best_var = new_sum_sq
                                best_move = ('swap', i1, dst, i2, s1, w1, s2, w2)

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
