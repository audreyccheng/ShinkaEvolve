# EVOLVE-BLOCK-START
"""Model placement algorithm for minimizing maximum KV cache pressure across GPUs using Binary Search with Multi-Strategy Packing and Local Search Refinement"""

import copy
import random

GPU_MEM_SIZE = 80.0  # GB

def compute_model_placement(gpu_num, models):
    """
    Minimizes max KVPR using Robust Binary Search with multiple packing heuristics
    followed by Steepest Descent Iterated Local Search (Descent + Ruin & Recreate).
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
    Refines placement using Steepest Descent with Ruin-and-Recreate (LNS) Kicks.
    """
    # Initialize state
    gpu_s = [sum(m.model_size for m in placement[i]) for i in range(gpu_num)]
    gpu_w = [sum(m.req_rate / m.slo for m in placement[i]) for i in range(gpu_num)]

    def get_k(i):
        rem = GPU_MEM_SIZE - gpu_s[i]
        if rem <= 1e-7: return 1e9
        return gpu_w[i] / rem

    current_ks = [get_k(i) for i in range(gpu_num)]
    best_max_k = max(current_ks)
    best_placement = copy.deepcopy(placement)

    max_steps = 600
    patience = 20
    no_improve = 0

    for step in range(max_steps):
        # 1. Identify bottleneck
        max_k = -1.0
        src = -1
        sorted_gpus = sorted(range(gpu_num), key=lambda x: current_ks[x])

        max_k = current_ks[sorted_gpus[-1]]
        src = sorted_gpus[-1]

        # Check global improvement
        if max_k < best_max_k - 1e-7:
            best_max_k = max_k
            best_placement = copy.deepcopy(placement)
            no_improve = 0
        else:
            no_improve += 1

        # 2. Ruin and Recreate Kick (if stuck)
        if no_improve > patience:
            # Select subset: Bottleneck + Random/Low-load
            subset = {src}
            # Pick 2 lowest loaded to help absorb load
            if gpu_num > 1: subset.add(sorted_gpus[0])
            if gpu_num > 2: subset.add(sorted_gpus[1])

            # Fill remaining slots with random
            target_size = min(gpu_num, random.randint(3, 5))
            while len(subset) < target_size:
                subset.add(random.randint(0, gpu_num - 1))

            subset = list(subset)

            # Ruin: Extract all models
            pool = []
            backup_state = {} # To revert if recreate fails badly (optional, but good for safety)

            for idx in subset:
                backup_state[idx] = (list(placement[idx]), gpu_s[idx], gpu_w[idx])
                pool.extend(placement[idx])
                placement[idx] = []
                gpu_s[idx] = 0.0
                gpu_w[idx] = 0.0
                current_ks[idx] = 0.0

            # Recreate: Sort and Pack
            # Sort by physical size descending (difficult items first), break ties with load
            pool.sort(key=lambda m: (m.model_size, m.req_rate/m.slo), reverse=True)

            valid_repack = True
            for m in pool:
                best_target = -1
                best_impact = float('inf') # Minimize resulting K

                for idx in subset:
                    if gpu_s[idx] + m.model_size <= GPU_MEM_SIZE:
                        # What would K be?
                        rem = GPU_MEM_SIZE - (gpu_s[idx] + m.model_size)
                        k = (gpu_w[idx] + m.req_rate/m.slo) / rem if rem > 1e-7 else 1e9
                        if k < best_impact:
                            best_impact = k
                            best_target = idx

                if best_target != -1:
                    placement[best_target].append(m)
                    gpu_s[best_target] += m.model_size
                    gpu_w[best_target] += m.req_rate/m.slo
                else:
                    valid_repack = False
                    break

            if valid_repack:
                # Update Ks
                for idx in subset:
                    current_ks[idx] = get_k(idx)
                # We always accept the kick to escape local optimum,
                # unless it causes a massive degradation (sanity check)
                if max(current_ks) > max_k * 1.5 and max(current_ks) > 1000:
                    # Revert if disastrous
                    for idx in subset:
                        placement[idx], gpu_s[idx], gpu_w[idx] = backup_state[idx]
                        current_ks[idx] = get_k(idx)
                else:
                    no_improve = 0 # Reset patience
            else:
                # Revert
                for idx in subset:
                    placement[idx], gpu_s[idx], gpu_w[idx] = backup_state[idx]
                    current_ks[idx] = get_k(idx)

            continue

        # 3. Steepest Descent (Greedy)
        best_move = None
        # Sort items in bottleneck by impact potential
        src_items = sorted(enumerate(placement[src]), key=lambda x: x[1].req_rate/x[1].slo, reverse=True)

        # A. Check Moves
        for idx, m in src_items:
            w, s = m.req_rate/m.slo, m.model_size
            for dst in range(gpu_num):
                if dst == src: continue
                if gpu_s[dst] + s > GPU_MEM_SIZE: continue

                rem_src = GPU_MEM_SIZE - (gpu_s[src] - s)
                nk_src = (gpu_w[src] - w) / rem_src if rem_src > 1e-7 else 1e9

                rem_dst = GPU_MEM_SIZE - (gpu_s[dst] + s)
                nk_dst = (gpu_w[dst] + w) / rem_dst if rem_dst > 1e-7 else 1e9

                new_peak = max(nk_src, nk_dst)
                if new_peak > max_k + 1e-7: continue

                peak_diff = max_k - new_peak
                # Variance reduction
                var_diff = (current_ks[src]**2 + current_ks[dst]**2) - (nk_src**2 + nk_dst**2)

                score = None
                if peak_diff > 1e-7: score = (1, peak_diff, var_diff)
                elif peak_diff > -1e-7 and var_diff > 1e-5: score = (0, 0, var_diff)

                if score and (best_move is None or score > best_move[0]):
                    best_move = (score, ('move', idx, dst, s, w))

        # B. Check Swaps (if Move not decisive)
        if best_move is None or best_move[0][0] == 0:
            for i1, m1 in src_items:
                w1, s1 = m1.req_rate/m1.slo, m1.model_size
                for dst in range(gpu_num):
                    if dst == src: continue
                    # Optimization: Skip high stress dst
                    if current_ks[dst] > max_k * 0.95: continue

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
                        if new_peak > max_k + 1e-7: continue

                        peak_diff = max_k - new_peak
                        var_diff = (current_ks[src]**2 + current_ks[dst]**2) - (nk_src**2 + nk_dst**2)

                        score = None
                        if peak_diff > 1e-7: score = (1, peak_diff, var_diff)
                        elif peak_diff > -1e-7 and var_diff > 1e-5: score = (0, 0, var_diff)

                        if score and (best_move is None or score > best_move[0]):
                            best_move = (score, ('swap', i1, dst, i2, s1, w1, s2, w2))

        # Apply Best Move
        if best_move:
            _, action = best_move
            if action[0] == 'move':
                _, idx, dst, s, w = action
                m = placement[src].pop(idx)
                placement[dst].append(m)
                gpu_s[src] -= s; gpu_w[src] -= w
                gpu_s[dst] += s; gpu_w[dst] += w
            elif action[0] == 'swap':
                _, i1, dst, i2, s1, w1, s2, w2 = action
                m1 = placement[src][i1]
                m2 = placement[dst][i2]
                placement[src][i1] = m2
                placement[dst][i2] = m1
                gpu_s[src] = gpu_s[src] - s1 + s2; gpu_w[src] = gpu_w[src] - w1 + w2
                gpu_s[dst] = gpu_s[dst] - s2 + s1; gpu_w[dst] = gpu_w[dst] - w2 + w1

            current_ks[src] = get_k(src)
            current_ks[dst] = get_k(dst)

            if max(current_ks) >= max_k - 1e-7:
                no_improve += 1
            else:
                no_improve = 0
        else:
            no_improve += 1

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