# EVOLVE-BLOCK-START
"""Model placement algorithm for minimizing maximum KV cache pressure using BFD Binary Search and Best-Improvement LNS"""

import copy
import random
import math

GPU_MEM_SIZE = 80.0  # GB

def compute_model_placement(gpu_num, models):
    """
    Minimizes max KVPR using Binary Search with Best-Fit Decreasing Packing
    followed by Best-Improvement Local Search and Ruin-Recreate Perturbation.
    """
    # --- 1. Pre-processing ---
    total_size = sum(m.model_size for m in models)
    if total_size > gpu_num * GPU_MEM_SIZE:
        raise ValueError("Total model size exceeds total GPU memory capacity.")

    # Prepare items: (w, s, m)
    # w = req_rate / slo
    items = [{'w': m.req_rate / m.slo, 's': m.model_size, 'm': m} for m in models]

    # --- 2. Binary Search for Initial Solution ---
    total_w = sum(x['w'] for x in items)
    slack = gpu_num * GPU_MEM_SIZE - total_size

    low = 0.0
    # Heuristic upper bound
    if slack < 1e-5:
        high = 1000.0
    else:
        avg_pressure = total_w / slack
        high = max(10.0, avg_pressure * 6.0)

    best_placement = None
    feasible_high = False

    # Exponential search to find a valid upper bound
    for _ in range(20):
        feasible, placement = _check_feasibility(gpu_num, items, high)
        if feasible:
            best_placement = placement
            feasible_high = True
            break
        low = high
        high *= 2.0

    if not feasible_high:
        # Fallback: very high bound
        high = 1e9
        feasible, placement = _check_feasibility(gpu_num, items, high)
        if feasible:
            best_placement = placement
        else:
            raise ValueError("Unable to place models. Constraints likely too tight.")

    # Binary Search Refinement (32 iterations for precision)
    for _ in range(32):
        mid = (low + high) / 2.0
        feasible, placement = _check_feasibility(gpu_num, items, mid)
        if feasible:
            best_placement = placement
            high = mid
        else:
            low = mid

    # Convert list placement to dictionary map for ILS
    placement_map = {i: best_placement[i] for i in range(gpu_num)}

    # --- 3. Iterated Local Search ---
    return _best_improvement_ils(gpu_num, placement_map)

def _check_feasibility(gpu_num, items, K):
    """
    Checks feasibility if models can be packed such that for all GPUs:
    sum(w) / (Capacity - sum(s)) <= K
    Rearranging: sum(w + K*s) <= K * Capacity

    Uses Best-Fit Decreasing (BFD) with multiple sort keys.
    """
    virtual_cap = K * GPU_MEM_SIZE

    # Precompute pack items
    # item: (virtual_size, physical_size, weight, density, model)
    pack_items = []
    for x in items:
        v = x['w'] + K * x['s']
        d = x['w'] / (x['s'] + 1e-6)
        pack_items.append({'v': v, 's': x['s'], 'w': x['w'], 'd': d, 'm': x['m']})

    # Heuristics: Sort keys
    strategies = [
        (lambda x: x['v'], True),  # Virtual Size Desc
        (lambda x: x['s'], True),  # Physical Size Desc
        (lambda x: x['d'], True),  # Density Desc
        (lambda x: x['w'], True),  # Weight Desc
    ]

    for key_func, reverse in strategies:
        sorted_items = sorted(pack_items, key=key_func, reverse=reverse)

        # Best Fit Decreasing
        res = _pack_bfd(gpu_num, sorted_items, virtual_cap)
        if res: return True, res

    return False, None

def _pack_bfd(gpu_num, items, virtual_cap):
    """
    Best Fit Decreasing packing.
    Places item in the bin with minimum residual virtual capacity.
    """
    bins_v = [0.0] * gpu_num
    bins_p = [0.0] * gpu_num
    placement = [[] for _ in range(gpu_num)]

    for item in items:
        v, s, m = item['v'], item['s'], item['m']
        best_bin = -1
        min_rem = float('inf')

        for i in range(gpu_num):
            if bins_p[i] + s <= GPU_MEM_SIZE and bins_v[i] + v <= virtual_cap + 1e-7:
                rem = virtual_cap - (bins_v[i] + v)
                if rem < min_rem:
                    min_rem = rem
                    best_bin = i

        if best_bin != -1:
            bins_p[best_bin] += s
            bins_v[best_bin] += v
            placement[best_bin].append(m)
        else:
            return None

    return placement

def _best_improvement_ils(gpu_num, placement):
    """
    Refines placement using Best-Improvement Descent and Ruin-Recreate.
    """
    # State Cache
    gpu_s = [sum(m.model_size for m in placement[i]) for i in range(gpu_num)]
    gpu_w = [sum(m.req_rate / m.slo for m in placement[i]) for i in range(gpu_num)]

    def get_k(i):
        rem = GPU_MEM_SIZE - gpu_s[i]
        if rem <= 1e-7: return 1e9
        return gpu_w[i] / rem

    best_sol = copy.deepcopy(placement)

    # Initial calculation
    ks = [get_k(i) for i in range(gpu_num)]
    best_max_k = max(ks)

    max_steps = 300
    patience = 25
    no_improve = 0

    for step in range(max_steps):
        # Update metrics
        ks = [get_k(i) for i in range(gpu_num)]
        max_k = max(ks)
        sum_sq = sum(k*k for k in ks)

        # Identify bottleneck
        # Note: multiple GPUs might have same max_k, pick one
        src = ks.index(max_k)

        # Update Global Best
        if max_k < best_max_k - 1e-7:
            best_max_k = max_k
            best_sol = copy.deepcopy(placement)
            no_improve = 0
        else:
            no_improve += 1

        if no_improve <= patience:
            # --- Best-Improvement Descent ---
            # Search all moves/swaps involving 'src' to find the one that
            # maximizes improvement in Max K, breaking ties with Variance.

            best_move = None # (type, ...)
            best_move_imp = -1e9
            best_move_var_red = -1e9

            # 1. Try Moves (Source -> Dst)
            for m_idx, m in enumerate(placement[src]):
                s, w = m.model_size, m.req_rate/m.slo

                for dst in range(gpu_num):
                    if dst == src: continue
                    if gpu_s[dst] + s > GPU_MEM_SIZE: continue

                    # New States
                    rem_src = GPU_MEM_SIZE - (gpu_s[src] - s)
                    nk_src = (gpu_w[src] - w) / rem_src if rem_src > 1e-7 else 1e9

                    rem_dst = GPU_MEM_SIZE - (gpu_s[dst] + s)
                    nk_dst = (gpu_w[dst] + w) / rem_dst if rem_dst > 1e-7 else 1e9

                    new_max = max(nk_src, nk_dst)

                    # Improvement in Peak
                    # We are only interested if it lowers the global peak or
                    # keeps global peak same (if src was sole bottleneck) but reduces local pressure

                    # Actually, we need to compare against global max_k.
                    # If src is the unique bottleneck, reducing src helps.
                    # If there are other bottlenecks == max_k, reducing src won't reduce global max immediately,
                    # but variance reduction helps.

                    # Calculate improvement relative to max_k
                    # We accept move if new_local_max < max_k (solves bottleneck)
                    # OR if new_local_max == max_k and variance reduces (rare for move, common for swap)

                    if new_max > max_k + 1e-7: continue # Made things worse

                    # If this move makes the local pair < max_k, check if it's the best reduction
                    imp = max_k - new_max

                    # Variance reduction
                    old_sq = ks[src]**2 + ks[dst]**2
                    new_sq = nk_src**2 + nk_dst**2
                    var_red = old_sq - new_sq

                    if imp > best_move_imp + 1e-7:
                        best_move_imp = imp
                        best_move_var_red = var_red
                        best_move = ('move', m_idx, dst, s, w)
                    elif abs(imp - best_move_imp) < 1e-7:
                        if var_red > best_move_var_red + 1e-7:
                            best_move_var_red = var_red
                            best_move = ('move', m_idx, dst, s, w)

            # 2. Try Swaps
            # To save time, only check swaps if move didn't solve everything perfectly
            # or to find better variance solutions.
            # Optimization: Skip if dst is also near bottleneck

            for m1_idx, m1 in enumerate(placement[src]):
                s1, w1 = m1.model_size, m1.req_rate/m1.slo

                for dst in range(gpu_num):
                    if dst == src: continue
                    if ks[dst] > max_k * 0.99: continue

                    for m2_idx, m2 in enumerate(placement[dst]):
                        s2, w2 = m2.model_size, m2.req_rate/m2.slo

                        ns_src = gpu_s[src] - s1 + s2
                        ns_dst = gpu_s[dst] - s2 + s1

                        if ns_src > GPU_MEM_SIZE or ns_dst > GPU_MEM_SIZE: continue

                        rem_src = GPU_MEM_SIZE - ns_src
                        nk_src = (gpu_w[src] - w1 + w2) / rem_src if rem_src > 1e-7 else 1e9

                        rem_dst = GPU_MEM_SIZE - ns_dst
                        nk_dst = (gpu_w[dst] - w2 + w1) / rem_dst if rem_dst > 1e-7 else 1e9

                        new_max = max(nk_src, nk_dst)
                        if new_max > max_k + 1e-7: continue

                        imp = max_k - new_max
                        old_sq = ks[src]**2 + ks[dst]**2
                        new_sq = nk_src**2 + nk_dst**2
                        var_red = old_sq - new_sq

                        if imp > best_move_imp + 1e-7:
                            best_move_imp = imp
                            best_move_var_red = var_red
                            best_move = ('swap', m1_idx, dst, m2_idx, s1, w1, s2, w2)
                        elif abs(imp - best_move_imp) < 1e-7:
                            if var_red > best_move_var_red + 1e-7:
                                best_move_var_red = var_red
                                best_move = ('swap', m1_idx, dst, m2_idx, s1, w1, s2, w2)

            # Execute Best Move
            if best_move:
                if best_move[0] == 'move':
                    _, idx, dst, s, w = best_move
                    m = placement[src].pop(idx)
                    placement[dst].append(m)
                    gpu_s[src] -= s; gpu_w[src] -= w
                    gpu_s[dst] += s; gpu_w[dst] += w
                elif best_move[0] == 'swap':
                    _, idx1, dst, idx2, s1, w1, s2, w2 = best_move
                    m1 = placement[src][idx1]
                    m2 = placement[dst][idx2]
                    placement[src][idx1] = m2
                    placement[dst][idx2] = m1
                    gpu_s[src] = gpu_s[src] - s1 + s2
                    gpu_w[src] = gpu_w[src] - w1 + w2
                    gpu_s[dst] = gpu_s[dst] - s2 + s1
                    gpu_w[dst] = gpu_w[dst] - w2 + w1
            else:
                # No valid move found to improve bottleneck -> force perturbation
                no_improve = patience + 1

        else:
            # --- Ruin and Recreate (LNS) ---
            # Perturb the state to escape local optimum

            # Select targets: Bottleneck + random subset
            targets = {src}
            n_targets = min(gpu_num, random.randint(3, 5))

            # Prefer low-loaded GPUs as potential destinations for offloading
            sorted_by_k = sorted(range(gpu_num), key=lambda i: ks[i])
            # Add 1-2 lowest loaded
            for t in sorted_by_k[:2]:
                targets.add(t)

            # Fill rest with random
            candidates = list(range(gpu_num))
            random.shuffle(candidates)
            for c in candidates:
                if len(targets) >= n_targets: break
                targets.add(c)

            targets = list(targets)

            # Backup
            backup_state = {}
            for t in targets:
                backup_state[t] = (list(placement[t]), gpu_s[t], gpu_w[t])

            # Ruin
            pool = []
            for t in targets:
                pool.extend(placement[t])
                placement[t] = []
                gpu_s[t] = 0.0
                gpu_w[t] = 0.0

            # Recreate: Sort pool by density or size
            # Randomize sorting strategy for diversity
            if random.random() < 0.5:
                # Sort by Size Desc
                pool.sort(key=lambda m: m.model_size, reverse=True)
            else:
                # Sort by Density Desc
                pool.sort(key=lambda m: m.req_rate/m.slo, reverse=True)

            possible = True
            for m in pool:
                w, s = m.req_rate/m.slo, m.model_size
                best_t = -1
                best_local_k = float('inf')

                # Try to place to minimize peak K within targets
                for t in targets:
                    if gpu_s[t] + s <= GPU_MEM_SIZE:
                        rem = GPU_MEM_SIZE - (gpu_s[t] + s)
                        k = (gpu_w[t] + w) / rem if rem > 1e-7 else 1e9
                        if k < best_local_k:
                            best_local_k = k
                            best_t = t

                if best_t != -1:
                    placement[best_t].append(m)
                    gpu_s[best_t] += s
                    gpu_w[best_t] += w
                else:
                    possible = False
                    break

            if possible:
                # Check acceptance
                # Calculate new max over targets
                new_local_max = 0
                for t in targets:
                    k = get_k(t)
                    if k > new_local_max: new_local_max = k

                # Simple acceptance criteria: if improved or occasionally if not much worse
                # But since we are inside "stuck" logic, we generally accept to move away
                # unless it creates a disaster significantly worse than current global max
                if new_local_max < max_k + 1.0: # lenient acceptance
                     no_improve = max(0, patience - 5)
                else:
                    # Revert
                    for t in targets:
                        placement[t], gpu_s[t], gpu_w[t] = backup_state[t]
            else:
                # Revert
                for t in targets:
                    placement[t], gpu_s[t], gpu_w[t] = backup_state[t]

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
