# EVOLVE-BLOCK-START
"""Model placement algorithm for minimizing maximum KV cache pressure using Robust Binary Search and Large Neighborhood Search (Ruin & Recreate)"""

import copy
import random
import math

GPU_MEM_SIZE = 80.0  # GB

def compute_model_placement(gpu_num, models):
    """
    Minimizes max KVPR using:
    1. Robust Binary Search with multiple packing heuristics.
    2. Large Neighborhood Search (LNS) using Ruin & Recreate strategy.

    Args:
        gpu_num: Number of GPUs
        models: List of models to place

    Returns:
        A placement of models to GPUs {gpu_id: [models]}
    """
    # 1. Validation and Pre-processing
    total_size = sum(m.model_size for m in models)
    if total_size > gpu_num * GPU_MEM_SIZE:
        raise ValueError("Total model size exceeds total GPU memory capacity.")

    # Prepare items for packing: (w, s, m)
    # w = req_rate / slo
    items = [{'w': m.req_rate / m.slo, 's': m.model_size, 'm': m} for m in models]

    # 2. Binary Search for Initial Feasible Solution
    total_w = sum(x['w'] for x in items)
    slack = gpu_num * GPU_MEM_SIZE - total_size

    low = 0.0
    # Heuristic upper bound
    if slack < 1e-6:
        high = 10000.0
    else:
        avg_pressure = total_w / slack
        high = max(10.0, avg_pressure * 6.0)

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
    # 32 iterations for high precision
    for _ in range(32):
        mid = (low + high) / 2.0
        feasible, placement = _check_feasibility_robust(gpu_num, items, mid)
        if feasible:
            best_placement = placement
            high = mid
        else:
            low = mid

    # Convert list placement to dictionary map
    placement_map = {i: best_placement[i] for i in range(gpu_num)}

    # 3. LNS / Ruin & Recreate Refinement
    final_placement = _lns_refinement(gpu_num, placement_map)

    return final_placement

def _check_feasibility_robust(gpu_num, items, K):
    """
    Checks feasibility using multiple sorting strategies and packing algorithms (FFD/BFD).
    Constraint: sum(w + K*s) <= K*Capacity
    """
    virtual_cap = K * GPU_MEM_SIZE

    # Create augmented items for sorting
    pack_items = []
    for x in items:
        # Virtual Size: v = w + K*s
        v = x['w'] + K * x['s']
        # Density: Load per unit size
        d = x['w'] / (x['s'] + 1e-7)
        pack_items.append({
            'v': v,
            's': x['s'],
            'w': x['w'],
            'd': d,
            'm': x['m']
        })

    # Heuristics: List of (sort_key_lambda, reverse_bool)
    heuristics = [
        (lambda x: x['v'], True),  # Virtual Desc
        (lambda x: x['s'], True),  # Physical Desc
        (lambda x: x['d'], True),  # Density Desc
        (lambda x: x['w'], True),  # Load Desc
    ]

    for key_func, rev in heuristics:
        sorted_items = sorted(pack_items, key=key_func, reverse=rev)

        # Try Best Fit Decreasing (BFD) - usually superior for tight packing
        if res := _pack_bfd(gpu_num, sorted_items, virtual_cap):
            return True, res

        # Try First Fit Decreasing (FFD)
        if res := _pack_ffd(gpu_num, sorted_items, virtual_cap):
            return True, res

    return False, None

def _pack_ffd(gpu_num, items, virtual_cap):
    """First Fit Decreasing Packing"""
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
    """Best Fit Decreasing Packing (Minimizing Residual Capacity)"""
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
    Refines placement using Steepest Descent (Move/Swap) and LNS Kicks.
    """
    # Initialize State
    gpu_s = [sum(m.model_size for m in placement[i]) for i in range(gpu_num)]
    gpu_w = [sum(m.req_rate / m.slo for m in placement[i]) for i in range(gpu_num)]

    def get_k(idx):
        rem = GPU_MEM_SIZE - gpu_s[idx]
        if rem <= 1e-7: return 1e9
        return gpu_w[idx] / rem

    current_ks = [get_k(i) for i in range(gpu_num)]
    best_max_k = max(current_ks)
    best_placement = copy.deepcopy(placement)

    max_steps = 1000
    patience = 50
    no_improve = 0

    for step in range(max_steps):
        # Identify bottleneck
        max_k = -1.0
        src = -1
        for i in range(gpu_num):
            if current_ks[i] > max_k:
                max_k = current_ks[i]
                src = i

        # Check global best
        if max_k < best_max_k - 1e-7:
            best_max_k = max_k
            best_placement = copy.deepcopy(placement)
            no_improve = 0
        else:
            no_improve += 1

        # 1. KICK (Ruin & Recreate) if stuck
        if no_improve > patience:
            subset_indices = {src}
            n_random = min(gpu_num - 1, random.randint(2, 4))
            others = list(range(gpu_num))
            random.shuffle(others)
            for o in others:
                if len(subset_indices) >= n_random + 1: break
                if o != src: subset_indices.add(o)
            subset_indices = list(subset_indices)

            # Backup
            backup_state = {}
            for idx in subset_indices:
                backup_state[idx] = (list(placement[idx]), gpu_s[idx], gpu_w[idx], current_ks[idx])

            # Collect models
            repack_models = []
            for idx in subset_indices:
                repack_models.extend(placement[idx])
                placement[idx] = []
                gpu_s[idx] = 0.0
                gpu_w[idx] = 0.0

            # Recreate with randomization to escape
            random.shuffle(repack_models)
            repack_models.sort(key=lambda m: (m.model_size, m.req_rate/m.slo), reverse=True)

            feasible = True
            for m in repack_models:
                best_idx = -1
                best_val = float('inf')
                for idx in subset_indices:
                    if gpu_s[idx] + m.model_size <= GPU_MEM_SIZE:
                        rem = GPU_MEM_SIZE - (gpu_s[idx] + m.model_size)
                        k = (gpu_w[idx] + m.req_rate/m.slo) / rem if rem > 1e-7 else 1e9
                        if k < best_val:
                            best_val = k
                            best_idx = idx
                if best_idx != -1:
                    placement[best_idx].append(m)
                    gpu_s[best_idx] += m.model_size
                    gpu_w[best_idx] += m.req_rate/m.slo
                else:
                    feasible = False
                    break

            if feasible:
                # Update state
                for idx in subset_indices:
                    current_ks[idx] = get_k(idx)
                no_improve = 0
            else:
                # Revert
                for idx in subset_indices:
                    placement[idx], gpu_s[idx], gpu_w[idx], current_ks[idx] = backup_state[idx]
            continue

        # 2. STEEPEST DESCENT (Moves & Swaps) on Bottleneck
        best_move = None # (gain, type, ...)

        # Sort items in bottleneck by weight descending
        src_items = sorted(enumerate(placement[src]), key=lambda x: x[1].req_rate/x[1].slo, reverse=True)

        # A. Try Moves
        for i, m in src_items:
            w, s = m.req_rate/m.slo, m.model_size
            for dst in range(gpu_num):
                if dst == src: continue
                if gpu_s[dst] + s > GPU_MEM_SIZE: continue

                # Pruning
                if current_ks[dst] > max_k * 0.95: continue

                rem_src = GPU_MEM_SIZE - (gpu_s[src] - s)
                nk_src = (gpu_w[src] - w) / rem_src if rem_src > 1e-7 else 1e9
                rem_dst = GPU_MEM_SIZE - (gpu_s[dst] + s)
                nk_dst = (gpu_w[dst] + w) / rem_dst if rem_dst > 1e-7 else 1e9

                new_peak = max(nk_src, nk_dst)

                if new_peak < max_k - 1e-7:
                    gain = max_k - new_peak
                    if best_move is None or gain > best_move[0]:
                        best_move = (gain, 'move', i, dst, s, w)

        # B. Try Swaps
        if best_move is None:
            for i1, m1 in src_items:
                w1, s1 = m1.req_rate/m1.slo, m1.model_size
                for dst in range(gpu_num):
                    if dst == src: continue
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
                        if new_peak < max_k - 1e-7:
                            gain = max_k - new_peak
                            if best_move is None or gain > best_move[0]:
                                best_move = (gain, 'swap', i1, dst, i2, s1, w1, s2, w2)

        # Apply Best Move
        if best_move:
            _, type_, *args = best_move
            if type_ == 'move':
                i, dst, s, w = args
                m = placement[src].pop(i)
                placement[dst].append(m)
                gpu_s[src] -= s; gpu_w[src] -= w
                gpu_s[dst] += s; gpu_w[dst] += w
            elif type_ == 'swap':
                i1, dst, i2, s1, w1, s2, w2 = args
                m1 = placement[src][i1]
                m2 = placement[dst][i2]
                placement[src][i1] = m2
                placement[dst][i2] = m1
                gpu_s[src] = gpu_s[src] - s1 + s2; gpu_w[src] = gpu_w[src] - w1 + w2
                gpu_s[dst] = gpu_s[dst] - s2 + s1; gpu_w[dst] = gpu_w[dst] - w2 + w1

            current_ks[src] = get_k(src)
            current_ks[dst] = get_k(dst)
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