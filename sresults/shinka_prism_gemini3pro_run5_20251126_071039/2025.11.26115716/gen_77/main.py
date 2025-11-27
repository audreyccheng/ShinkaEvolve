# EVOLVE-BLOCK-START
"""Model placement algorithm for minimizing maximum KV cache pressure using Binary Search with Best-Fit Decreasing and Variance-Smoothing Local Search"""

import copy
import random
import math

GPU_MEM_SIZE = 80.0  # GB

def compute_model_placement(gpu_num, models):
    """
    Minimizes max KVPR using Robust Binary Search (BFD Packing)
    followed by Variance-Smoothing Iterated Local Search.
    """
    # 1. Validation and Pre-computation
    total_size = sum(m.model_size for m in models)
    if total_size > gpu_num * GPU_MEM_SIZE:
        raise ValueError("Total model size exceeds total GPU memory capacity.")

    # Wrap models with cached properties for speed
    # Item: (w, s, density, model)
    items = []
    for m in models:
        w = m.req_rate / m.slo
        s = m.model_size
        d = w / (s + 1e-7)
        items.append({'w': w, 's': s, 'd': d, 'm': m})

    # 2. Binary Search for Initial Feasible Solution
    total_w = sum(x['w'] for x in items)
    slack = gpu_num * GPU_MEM_SIZE - total_size

    low = 0.0
    # Heuristic upper bound
    if slack < 1e-6:
        high = 10000.0
    else:
        avg_k = total_w / slack
        high = max(10.0, avg_k * 6.0)

    best_placement = None
    feasible_high = False

    # Exponential search for valid upper bound
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

    # Binary Search Refinement (30 iters -> high precision)
    for _ in range(30):
        mid = (low + high) / 2.0
        feasible, placement = _check_feasibility_robust(gpu_num, items, mid)
        if feasible:
            best_placement = placement
            high = mid
        else:
            low = mid

    # Convert to standard format
    placement_map = {i: best_placement[i] for i in range(gpu_num)}

    # 3. Variance-Smoothing Iterated Local Search
    return _variance_smoothing_ils(gpu_num, placement_map)

def _check_feasibility_robust(gpu_num, items, K):
    """
    Checks feasibility using Best-Fit and First-Fit Decreasing with multiple sorting keys.
    """
    virtual_cap = K * GPU_MEM_SIZE

    # Augment items with virtual size
    pack_items = []
    for x in items:
        v = x['w'] + K * x['s']
        pack_items.append({'v': v, 's': x['s'], 'w': x['w'], 'd': x['d'], 'm': x['m']})

    # 1. Deterministic Heuristics
    heuristics = [
        (lambda x: x['v'], True),  # Virtual Size Descending
        (lambda x: x['s'], True),  # Physical Size Descending
        (lambda x: x['w'], True),  # Load Descending
        (lambda x: x['d'], True),  # Density Descending
    ]

    for key_func, rev in heuristics:
        pack_items.sort(key=key_func, reverse=rev)
        # Try Best Fit Decreasing (usually superior for tight packing)
        if res := _pack_bfd(gpu_num, pack_items, virtual_cap):
            return True, res
        # Try First Fit Decreasing (adds diversity in bin assignment)
        if res := _pack_ffd(gpu_num, pack_items, virtual_cap):
            return True, res

    # 2. Randomized Heuristics (Fallback)
    temp_items = pack_items[:]
    for _ in range(4):
        random.shuffle(temp_items)
        if res := _pack_bfd(gpu_num, temp_items, virtual_cap):
            return True, res

    return False, None

def _pack_bfd(gpu_num, items, virtual_cap):
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

def _variance_smoothing_ils(gpu_num, placement):
    """
    Iterated Local Search with Variance Tie-Breaking and Burst Kicks (Moves & Swaps).
    """
    # State initialization
    gpu_s = [sum(m.model_size for m in placement[i]) for i in range(gpu_num)]
    gpu_w = [sum(m.req_rate / m.slo for m in placement[i]) for i in range(gpu_num)]

    def get_k(idx):
        rem = GPU_MEM_SIZE - gpu_s[idx]
        if rem <= 1e-7: return 1e9
        return gpu_w[idx] / rem

    best_max_k = max(get_k(i) for i in range(gpu_num))
    best_sol = copy.deepcopy(placement)

    max_steps = 1000
    patience = 20
    no_improve = 0

    # Pre-calculate initial sum of squares
    current_ks = [get_k(i) for i in range(gpu_num)]

    for _ in range(max_steps):
        # 1. Status Check
        max_k = -1.0
        src = -1
        sum_sq = 0.0

        for i in range(gpu_num):
            k = get_k(i)
            current_ks[i] = k
            if k > max_k:
                max_k = k
                src = i
            sum_sq += k*k

        # Update Global Best
        if max_k < best_max_k - 1e-6:
            best_max_k = max_k
            best_sol = copy.deepcopy(placement)
            no_improve = 0
        else:
            no_improve += 1

        # 2. Burst Kick (Escape Local Optima)
        if no_improve > patience:
            moves_done = 0
            # Perform a burst of perturbations
            burst_size = random.randint(3, 6)
            for _ in range(burst_size):
                # Bias source towards high-load GPUs
                candidates = [i for i in range(gpu_num) if current_ks[i] > max_k * 0.7]
                if not candidates: candidates = list(range(gpu_num))
                s_idx = random.choice(candidates)

                if not placement[s_idx]: continue

                d_idx = random.randint(0, gpu_num - 1)
                if s_idx == d_idx: continue

                # 50% Move, 50% Swap
                if random.random() < 0.5:
                    # Try Move
                    m_idx = random.randint(0, len(placement[s_idx]) - 1)
                    m = placement[s_idx][m_idx]

                    if gpu_s[d_idx] + m.model_size <= GPU_MEM_SIZE:
                        placement[d_idx].append(m)
                        placement[s_idx].pop(m_idx)
                        gpu_s[d_idx] += m.model_size; gpu_w[d_idx] += m.req_rate/m.slo
                        gpu_s[s_idx] -= m.model_size; gpu_w[s_idx] -= m.req_rate/m.slo
                        current_ks[s_idx] = get_k(s_idx)
                        current_ks[d_idx] = get_k(d_idx)
                        moves_done += 1
                else:
                    # Try Swap
                    if not placement[d_idx]: continue
                    m1_idx = random.randint(0, len(placement[s_idx]) - 1)
                    m1 = placement[s_idx][m1_idx]
                    m2_idx = random.randint(0, len(placement[d_idx]) - 1)
                    m2 = placement[d_idx][m2_idx]

                    ns_s = gpu_s[s_idx] - m1.model_size + m2.model_size
                    ns_d = gpu_s[d_idx] - m2.model_size + m1.model_size

                    if ns_s <= GPU_MEM_SIZE and ns_d <= GPU_MEM_SIZE:
                        placement[s_idx][m1_idx] = m2
                        placement[d_idx][m2_idx] = m1
                        gpu_s[s_idx] = ns_s; gpu_w[s_idx] = gpu_w[s_idx] - (m1.req_rate/m1.slo) + (m2.req_rate/m2.slo)
                        gpu_s[d_idx] = ns_d; gpu_w[d_idx] = gpu_w[d_idx] - (m2.req_rate/m2.slo) + (m1.req_rate/m1.slo)
                        current_ks[s_idx] = get_k(s_idx)
                        current_ks[d_idx] = get_k(d_idx)
                        moves_done += 1

            if moves_done > 0:
                # Reset patience partially
                no_improve = max(0, patience - 8)
            continue

        # 3. Steepest Descent with Variance Tie-Breaking
        # Evaluate Moves AND Swaps to find the absolute best step

        best_move = None
        best_imp_k = -1.0
        best_imp_var = -1.0

        models = placement[src]
        sorted_models_idx = sorted(range(len(models)), key=lambda i: models[i].req_rate/models[i].slo, reverse=True)

        # A. Evaluate Moves
        for i in sorted_models_idx:
            m = models[i]
            w, s = m.req_rate/m.slo, m.model_size

            for dst in range(gpu_num):
                if dst == src: continue
                if gpu_s[dst] + s > GPU_MEM_SIZE: continue

                rem_src = GPU_MEM_SIZE - (gpu_s[src] - s)
                nk_src = (gpu_w[src] - w) / rem_src if rem_src > 1e-7 else 1e9

                rem_dst = GPU_MEM_SIZE - (gpu_s[dst] + s)
                nk_dst = (gpu_w[dst] + w) / rem_dst if rem_dst > 1e-7 else 1e9

                new_peak = max(nk_src, nk_dst)
                diff_k = max_k - new_peak

                # Calculate variance change
                old_sq = current_ks[src]**2 + current_ks[dst]**2
                new_sq = nk_src**2 + nk_dst**2
                diff_var = old_sq - new_sq

                is_better = False

                if diff_k > 1e-6:
                    if diff_k > best_imp_k + 1e-7:
                        is_better = True
                    elif abs(diff_k - best_imp_k) < 1e-7:
                        # Tie on K improvement, check variance
                        if diff_var > best_imp_var + 1e-7:
                            is_better = True
                elif abs(diff_k) < 1e-6 and best_imp_k < 1e-6:
                    # Plateau move: only accept if variance improves and we haven't found a K-improving move
                    if diff_var > 1e-5 and diff_var > best_imp_var:
                        is_better = True

                if is_better:
                    best_imp_k = max(best_imp_k, diff_k)
                    best_imp_var = diff_var
                    best_move = ('move', i, dst, s, w)

        # B. Evaluate Swaps (Always check for better K improvement)
        for i1 in sorted_models_idx:
            m1 = models[i1]
            w1, s1 = m1.req_rate/m1.slo, m1.model_size

            for dst in range(gpu_num):
                if dst == src: continue

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
                    diff_k = max_k - new_peak

                    old_sq = current_ks[src]**2 + current_ks[dst]**2
                    new_sq = nk_src**2 + nk_dst**2
                    diff_var = old_sq - new_sq

                    is_better = False

                    if diff_k > 1e-6:
                        if diff_k > best_imp_k + 1e-7:
                            is_better = True
                        elif abs(diff_k - best_imp_k) < 1e-7:
                            if diff_var > best_imp_var + 1e-7:
                                is_better = True
                    elif abs(diff_k) < 1e-6 and best_imp_k < 1e-6:
                        if diff_var > 1e-5 and diff_var > best_imp_var:
                            is_better = True

                    if is_better:
                        best_imp_k = max(best_imp_k, diff_k)
                        best_imp_var = diff_var
                        best_move = ('swap', i1, dst, i2, s1, w1, s2, w2)

        # Execute Best Move
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

            # Reset patience slightly on successful variance reduction to allow climbing
            if best_imp_k < 1e-6:
                no_improve = max(0, patience - 5)
            else:
                no_improve = 0
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