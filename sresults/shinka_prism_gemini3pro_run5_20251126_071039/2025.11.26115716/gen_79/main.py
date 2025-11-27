# EVOLVE-BLOCK-START
"""Model placement algorithm for minimizing maximum KV cache pressure using Robust BFD Binary Search and Guided First-Improvement ILS"""

import copy
import random
import math

GPU_MEM_SIZE = 80.0  # GB

def compute_model_placement(gpu_num, models):
    """
    Minimizes max KVPR using:
    1. Robust Binary Search with Best-Fit Decreasing (BFD) Packing.
       - Includes deterministic heuristics and randomized shuffle fallbacks.
    2. Guided First-Improvement Iterated Local Search (ILS).
       - Focuses on bottleneck GPUs.
       - Sorts candidate items by pressure contribution (Guide).
       - Uses variance reduction for tie-breaking on plateaus.
       - Uses Burst Kicks to escape local optima.
    """
    # 1. Validation and Setup
    total_size = sum(m.model_size for m in models)
    if total_size > gpu_num * GPU_MEM_SIZE:
        raise ValueError("Total model size exceeds total GPU memory capacity.")

    # Prepare items: (req_rate/slo, model_size, model_obj)
    # item dictionary for easy access
    items = [{'w': m.req_rate / m.slo, 's': m.model_size, 'm': m} for m in models]

    # 2. Binary Search for Initial Feasible Solution
    total_w = sum(x['w'] for x in items)
    slack = gpu_num * GPU_MEM_SIZE - total_size

    low = 0.0
    if slack < 1e-6:
        high = 10000.0 # Wide range for tight constraints
    else:
        avg_pressure = total_w / slack
        high = max(10.0, avg_pressure * 8.0)

    best_placement = None
    feasible_high = False

    # Find valid upper bound (Heuristic Expansion)
    for _ in range(20):
        # Try finding feasible packing with current high
        feasible, placement = _check_feasibility_robust(gpu_num, items, high, use_random=True)
        if feasible:
            best_placement = placement
            feasible_high = True
            break
        low = high
        high *= 2.0

    if not feasible_high:
        raise ValueError("Unable to place models. Constraints likely too tight.")

    # Binary Search Refinement
    # 32 iterations gives high precision
    for _ in range(32):
        mid = (low + high) / 2.0
        feasible, placement = _check_feasibility_robust(gpu_num, items, mid, use_random=False)
        if feasible:
            best_placement = placement
            high = mid
        else:
            low = mid

    # Final check with random enabled at the found high to ensure we have the best layout for that K
    # (Sometimes random shuffle finds a packing that deterministic didn't at marginally lower K,
    # but here we just take the valid one we found)

    placement_map = {i: best_placement[i] for i in range(gpu_num)}

    # 3. Guided Local Search Refinement
    return _guided_ils(gpu_num, placement_map)

def _check_feasibility_robust(gpu_num, items, K, use_random=False):
    """
    Checks if items can be packed with max KVPR <= K.
    Uses Best-Fit Decreasing with multiple sort keys.
    """
    virtual_cap = K * GPU_MEM_SIZE

    # Precompute sort keys
    pack_items = []
    for x in items:
        v = x['w'] + K * x['s']
        d = x['w'] / (x['s'] + 1e-7)
        pack_items.append({
            'v': v,
            's': x['s'],
            'w': x['w'],
            'd': d,
            'm': x['m']
        })

    # Heuristics: (Sort Key, Reverse)
    heuristics = [
        (lambda x: x['v'], True),  # Virtual Size Desc (Standard)
        (lambda x: x['s'], True),  # Physical Size Desc
        (lambda x: x['d'], True),  # Density Desc (High pressure/size first)
        (lambda x: x['w'], True),  # Load Desc
    ]

    for key_func, rev in heuristics:
        sorted_items = sorted(pack_items, key=key_func, reverse=rev)
        if res := _pack_bfd(gpu_num, sorted_items, virtual_cap):
            return True, res

    # Random shuffle fallback
    if use_random:
        base_items = list(pack_items) # Copy
        for _ in range(5):
            random.shuffle(base_items)
            if res := _pack_bfd(gpu_num, base_items, virtual_cap):
                return True, res

    return False, None

def _pack_bfd(gpu_num, items, virtual_cap):
    """
    Best Fit Decreasing: Places item in the bin with smallest sufficient residual virtual capacity.
    """
    bins_v = [0.0] * gpu_num
    bins_p = [0.0] * gpu_num
    placement = [[] for _ in range(gpu_num)]

    for item in items:
        v, s, m = item['v'], item['s'], item['m']

        best_bin = -1
        min_rem_v = float('inf')

        for i in range(gpu_num):
            # Check constraints
            if bins_p[i] + s <= GPU_MEM_SIZE and bins_v[i] + v <= virtual_cap + 1e-7:
                rem = virtual_cap - (bins_v[i] + v)
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

def _guided_ils(gpu_num, placement):
    """
    Guided First-Improvement Local Search.
    - Sorts items on bottleneck GPU by 'w' descending to prioritize high-impact moves.
    - Accepts moves that reduce Max K or reduce Variance (if Max K is tied).
    - Uses Burst Kicks when stuck.
    """
    # State tracking
    gpu_s = [sum(m.model_size for m in placement[i]) for i in range(gpu_num)]
    gpu_w = [sum(m.req_rate / m.slo for m in placement[i]) for i in range(gpu_num)]

    def get_k(i):
        rem = GPU_MEM_SIZE - gpu_s[i]
        if rem <= 1e-7: return 1e9
        return gpu_w[i] / rem

    current_ks = [get_k(i) for i in range(gpu_num)]
    best_max_k = max(current_ks)
    best_sol = copy.deepcopy(placement)

    max_steps = 1000
    patience = 50
    no_improve = 0

    for step in range(max_steps):
        # 1. Status Update
        # Recalculate full stats (though we update incrementally, good to be safe or just use cached)
        # Using cached current_ks

        max_k = max(current_ks)

        # Check global best
        if max_k < best_max_k - 1e-7:
            best_max_k = max_k
            best_sol = copy.deepcopy(placement)
            no_improve = 0
        else:
            no_improve += 1

        # 2. Perturbation (Kick)
        if no_improve > patience:
            # Strategy A: Balance Pair (Targeted Redistribution)
            # Try to rebalance the bottleneck with the least loaded GPU
            # This allows escaping local optima where single moves fail

            # Find current bottleneck and least loaded
            sorted_gpus = sorted(range(gpu_num), key=lambda i: current_ks[i])
            src = sorted_gpus[-1] # Max K
            dst = sorted_gpus[0]  # Min K

            success = False
            if src != dst and placement[src]:
                # Attempt Redistribution
                pool = placement[src] + placement[dst]
                # Sort by weight desc (biggest pressure contributors first)
                pool.sort(key=lambda m: m.req_rate/m.slo, reverse=True)

                # Backup
                backup_src = (list(placement[src]), gpu_s[src], gpu_w[src])
                backup_dst = (list(placement[dst]), gpu_s[dst], gpu_w[dst])

                # Clear
                placement[src] = []
                placement[dst] = []
                gpu_s[src] = 0.0; gpu_w[src] = 0.0
                gpu_s[dst] = 0.0; gpu_w[dst] = 0.0

                valid_redist = True
                for m in pool:
                    # Try to place in src or dst to minimize peak K
                    s, w = m.model_size, m.req_rate/m.slo

                    can_src = (gpu_s[src] + s <= GPU_MEM_SIZE)
                    can_dst = (gpu_s[dst] + s <= GPU_MEM_SIZE)

                    if not can_src and not can_dst:
                        valid_redist = False; break

                    choice = None
                    if can_src and not can_dst:
                        choice = 'src'
                    elif can_dst and not can_src:
                        choice = 'dst'
                    else:
                        # Both valid, choose best fit
                        rem_src = GPU_MEM_SIZE - (gpu_s[src] + s)
                        k_src = (gpu_w[src] + w) / rem_src if rem_src > 1e-7 else 1e9

                        rem_dst = GPU_MEM_SIZE - (gpu_s[dst] + s)
                        k_dst = (gpu_w[dst] + w) / rem_dst if rem_dst > 1e-7 else 1e9

                        if k_src < k_dst:
                            choice = 'src'
                        else:
                            choice = 'dst'

                    if choice == 'src':
                        placement[src].append(m)
                        gpu_s[src] += s; gpu_w[src] += w
                    else:
                        placement[dst].append(m)
                        gpu_s[dst] += s; gpu_w[dst] += w

                # Update Ks
                current_ks[src] = get_k(src)
                current_ks[dst] = get_k(dst)

                if not valid_redist:
                    # Revert
                    placement[src], gpu_s[src], gpu_w[src] = backup_src
                    placement[dst], gpu_s[dst], gpu_w[dst] = backup_dst
                    current_ks[src] = get_k(src)
                    current_ks[dst] = get_k(dst)
                else:
                    success = True
                    no_improve = 0

            # Strategy B: Random Burst Kick (Fallback)
            if not success:
                kick_len = random.randint(3, 5)
                for _ in range(kick_len):
                    for _ in range(10):
                        s = random.randint(0, gpu_num - 1)
                        if not placement[s]: continue
                        d = random.randint(0, gpu_num - 1)
                        if s == d: continue

                        m_idx = random.randint(0, len(placement[s]) - 1)
                        m = placement[s][m_idx]

                        if gpu_s[d] + m.model_size <= GPU_MEM_SIZE:
                            placement[d].append(m)
                            placement[s].pop(m_idx)
                            gpu_s[s] -= m.model_size; gpu_w[s] -= m.req_rate/m.slo
                            gpu_s[d] += m.model_size; gpu_w[d] += m.req_rate/m.slo
                            current_ks[s] = get_k(s)
                            current_ks[d] = get_k(d)
                            break
                no_improve = 0
            continue

        # 3. Guided First-Improvement Descent
        # Identify bottleneck
        # Use a small tolerance to include tied bottlenecks
        bottlenecks = [i for i, k in enumerate(current_ks) if k > max_k - 1e-5]
        src = random.choice(bottlenecks)

        # Sort items by weight descending (Guide)
        # We need indices to pop correctly, so we store (index, model) then sort
        src_items = list(enumerate(placement[src]))
        # Sort by w descending
        src_items.sort(key=lambda x: x[1].req_rate/x[1].slo, reverse=True)

        improved = False

        for original_idx, m in src_items:
            # We need to find the current index of 'm' in placement[src] as popping changes indices
            if m not in placement[src]: continue

            s, w = m.model_size, m.req_rate/m.slo

            # A. Try Move
            for dst in range(gpu_num):
                if dst == src: continue
                if gpu_s[dst] + s > GPU_MEM_SIZE: continue

                # Check metrics
                rem_src = GPU_MEM_SIZE - (gpu_s[src] - s)
                nk_src = (gpu_w[src] - w) / rem_src if rem_src > 1e-7 else 1e9

                rem_dst = GPU_MEM_SIZE - (gpu_s[dst] + s)
                nk_dst = (gpu_w[dst] + w) / rem_dst if rem_dst > 1e-7 else 1e9

                new_peak = max(nk_src, nk_dst)

                # Condition: Peak reduced OR (Peak equal AND Variance reduced)
                # We also ensure we don't exceed current global max_k (locally)
                if new_peak < max_k - 1e-7:
                    # Good improvement
                    curr_idx = placement[src].index(m)
                    placement[src].pop(curr_idx)
                    placement[dst].append(m)
                    gpu_s[src] -= s; gpu_w[src] -= w
                    gpu_s[dst] += s; gpu_w[dst] += w
                    current_ks[src] = get_k(src)
                    current_ks[dst] = get_k(dst)
                    improved = True
                    break
                elif abs(new_peak - max_k) < 1e-7:
                    # Check variance
                    old_sq = current_ks[src]**2 + current_ks[dst]**2
                    new_sq = nk_src**2 + nk_dst**2
                    if new_sq < old_sq - 1e-5:
                        curr_idx = placement[src].index(m)
                        placement[src].pop(curr_idx)
                        placement[dst].append(m)
                        gpu_s[src] -= s; gpu_w[src] -= w
                        gpu_s[dst] += s; gpu_w[dst] += w
                        current_ks[src] = get_k(src)
                        current_ks[dst] = get_k(dst)
                        improved = True
                        break

            if improved: break

            # B. Try Swap
            # Only if Move failed
            for dst in range(gpu_num):
                if dst == src: continue
                # Skip if dst is also a bottleneck (swapping usually doesn't help unless sizes differ vastly)
                if current_ks[dst] > max_k - 1e-5: continue

                for m2 in placement[dst]:
                    s2, w2 = m2.model_size, m2.req_rate/m2.slo

                    ns_src = gpu_s[src] - s + s2
                    ns_dst = gpu_s[dst] - s2 + s

                    if ns_src > GPU_MEM_SIZE or ns_dst > GPU_MEM_SIZE: continue

                    rem_src = GPU_MEM_SIZE - ns_src
                    nk_src = (gpu_w[src] - w + w2) / rem_src if rem_src > 1e-7 else 1e9

                    rem_dst = GPU_MEM_SIZE - ns_dst
                    nk_dst = (gpu_w[dst] - w2 + w) / rem_dst if rem_dst > 1e-7 else 1e9

                    new_peak = max(nk_src, nk_dst)

                    if new_peak < max_k - 1e-7:
                        curr_idx = placement[src].index(m)
                        curr_idx2 = placement[dst].index(m2)
                        placement[src][curr_idx] = m2
                        placement[dst][curr_idx2] = m
                        gpu_s[src] = ns_src; gpu_w[src] += (w2 - w)
                        gpu_s[dst] = ns_dst; gpu_w[dst] += (w - w2)
                        current_ks[src] = get_k(src)
                        current_ks[dst] = get_k(dst)
                        improved = True
                        break
                    elif abs(new_peak - max_k) < 1e-7:
                        old_sq = current_ks[src]**2 + current_ks[dst]**2
                        new_sq = nk_src**2 + nk_dst**2
                        if new_sq < old_sq - 1e-5:
                            curr_idx = placement[src].index(m)
                            curr_idx2 = placement[dst].index(m2)
                            placement[src][curr_idx] = m2
                            placement[dst][curr_idx2] = m
                            gpu_s[src] = ns_src; gpu_w[src] += (w2 - w)
                            gpu_s[dst] = ns_dst; gpu_w[dst] += (w - w2)
                            current_ks[src] = get_k(src)
                            current_ks[dst] = get_k(dst)
                            improved = True
                            break
                if improved: break
            if improved: break

        # If no improvement found for ANY item in bottleneck GPU, we just loop again (and increment patience)
        # Eventually Burst Kick triggers

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
