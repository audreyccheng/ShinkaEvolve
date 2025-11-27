# EVOLVE-BLOCK-START
"""Model placement algorithm for minimizing maximum KV cache pressure across GPUs"""

import random
import math

GPU_MEM_SIZE = 80.0

def compute_model_placement(gpu_num, models):
    """
    Minimizes max KVPR using Binary Search packing followed by Iterated Local Search.

    Key innovations:
    1. Binary Search with Failure-Memory Packing to find a strong initial baseline.
    2. Two-stage Local Search Objective: Minimize (Max_KVPR, Sum_Squared_KVPR).
       This breaks plateaus by preferring balanced loads among non-bottleneck GPUs.
    3. Iterated Local Search (ILS) with 'Ruin & Recreate' perturbation to escape local optima.
    """

    # ---------------------------------------------------------
    # 0. Data Structuring
    # ---------------------------------------------------------
    m_data = []
    total_w = 0.0
    total_s = 0.0
    for i, m in enumerate(models):
        w = m.req_rate / m.slo
        s = m.model_size
        m_data.append({'w': w, 's': s, 'obj': m, 'id': i})
        total_w += w
        total_s += s

    # ---------------------------------------------------------
    # 1. Initialization via Binary Search (Feasibility Check)
    # ---------------------------------------------------------

    def solve_packing(target_k):
        """
        Tries to pack models such that for all GPUs: w_sum / (C - s_sum) <= target_k
        Equivalent to Bin Packing with items satisfying: w + target_k * s <= target_k * C
        """
        # We use a randomized approach with memory of failed items
        fail_counts = {i: 0 for i in range(len(m_data))}

        # Strategies: (Base Key Lambda, Reverse)
        strategies = [
            (lambda x: x['s'] + x['w']/target_k, True),              # Effective Size (Best Fit)
            (lambda x: x['s'], True),                                # Physical Size
            (lambda x: x['w'], True),                                # Weight
            (lambda x: x['w'] / (x['s'] + 1e-6), True),              # Density
            (lambda x: x['w'] / (GPU_MEM_SIZE - x['s'] + 1e-6), True)# Asymptotic Pressure
        ]

        for trial in range(30):
            # Select strategy
            base_key, reverse = strategies[trial % len(strategies)]

            # Noise factor
            noise = 0.02 + (0.005 * trial)

            def sort_key(idx):
                val = base_key(m_data[idx])
                # Random perturbation
                val *= random.uniform(1.0 - noise, 1.0 + noise)
                # Failure memory boost: Put hard items first
                val += fail_counts[idx] * 1e6
                return val

            indices = sorted(range(len(m_data)), key=sort_key, reverse=reverse)

            # Packing
            bins_s = [0.0] * gpu_num
            bins_w = [0.0] * gpu_num
            placement = [[] for _ in range(gpu_num)]
            possible = True
            first_fail = None

            for idx in indices:
                item = m_data[idx]
                w, s = item['w'], item['s']

                # We need a bin where:
                # 1. s_new <= C
                # 2. w_new <= K * (C - s_new)

                best_b = None
                min_slack = float('inf')

                for b in range(gpu_num):
                    if bins_s[b] + s > GPU_MEM_SIZE: continue

                    rem_s = GPU_MEM_SIZE - (bins_s[b] + s)
                    if rem_s < 0: rem_s = 0.0

                    max_allowed_w = target_k * rem_s
                    curr_w = bins_w[b] + w

                    if curr_w <= max_allowed_w + 1e-6:
                        # Feasible
                        slack = max_allowed_w - curr_w
                        if slack < min_slack:
                            min_slack = slack
                            best_b = b

                if best_b is None:
                    possible = False
                    first_fail = idx
                    break

                placement[best_b].append(idx)
                bins_s[best_b] += s
                bins_w[best_b] += w

            if possible:
                return {g: indices for g, indices in enumerate(placement)}
            else:
                if first_fail is not None:
                    fail_counts[first_fail] += 1

        return None

    # Determine Range
    rem_global = gpu_num * GPU_MEM_SIZE - total_s
    lb = total_w / rem_global if rem_global > 1e-6 else 0.0

    # Quick Upper Bound via simple greedy
    sorted_s = sorted(range(len(m_data)), key=lambda i: m_data[i]['s'], reverse=True)
    ub_bins = [{'w':0.0, 's':0.0} for _ in range(gpu_num)]
    ub_max = 0.0
    for idx in sorted_s:
        item = m_data[idx]
        best_b = -1
        min_p = float('inf')
        for b in range(gpu_num):
            if ub_bins[b]['s'] + item['s'] <= GPU_MEM_SIZE:
                rem = GPU_MEM_SIZE - (ub_bins[b]['s'] + item['s'])
                if rem < 1e-9:
                     p = float('inf') if (ub_bins[b]['w'] + item['w']) > 0 else 0.0
                else:
                     p = (ub_bins[b]['w'] + item['w']) / rem
                if p < min_p:
                    min_p = p
                    best_b = b
        if best_b != -1:
            ub_bins[best_b]['s'] += item['s']
            ub_bins[best_b]['w'] += item['w']
            ub_max = max(ub_max, min_p)
        else:
            ub_max = 5000.0 # Fallback
            break

    high = ub_max
    low = lb

    best_placement = None
    best_score = float('inf')

    # Binary Search
    for _ in range(16):
        mid = (low + high) / 2.0
        sol = solve_packing(mid)
        if sol:
            # Check real score
            real_max = 0.0
            for g_idxs in sol.values():
                w = sum(m_data[i]['w'] for i in g_idxs)
                s = sum(m_data[i]['s'] for i in g_idxs)
                rem = GPU_MEM_SIZE - s
                val = w/rem if rem > 1e-9 else float('inf')
                real_max = max(real_max, val)

            if real_max < best_score:
                best_score = real_max
                best_placement = sol
            high = mid
        else:
            low = mid

    if best_placement is None:
         raise ValueError("Could not find feasible placement.")

    # ---------------------------------------------------------
    # 2. Iterated Local Search (ILS)
    # ---------------------------------------------------------
    # State representation: list of lists of indices
    current_sol = [list(best_placement[g]) for g in range(gpu_num)]

    # Helper to get full stats
    def get_stats(solution):
        """Returns list of dicts {w, s, p} and global metrics"""
        stats = []
        max_p = 0.0
        sum_sq_p = 0.0
        for g in range(gpu_num):
            w = sum(m_data[i]['w'] for i in solution[g])
            s = sum(m_data[i]['s'] for i in solution[g])
            rem = GPU_MEM_SIZE - s
            p = w / rem if rem > 1e-9 else (float('inf') if w > 0 else 0.0)
            stats.append({'w': w, 's': s, 'p': p})
            max_p = max(max_p, p)
            sum_sq_p += p*p
        return stats, max_p, sum_sq_p

    curr_stats, curr_max, curr_ssq = get_stats(current_sol)

    # Store global best
    global_best_sol = [list(x) for x in current_sol]
    global_best_max = curr_max

    def local_search(sol, stats, max_p, ssq_p):
        """Greedy Hill Climbing with Lexicographical Objective"""
        improved = True
        while improved:
            improved = False

            # Identify bottleneck
            src_gpu = -1
            highest_p = -1.0
            for g in range(gpu_num):
                if stats[g]['p'] > highest_p:
                    highest_p = stats[g]['p']
                    src_gpu = g

            if src_gpu == -1 or highest_p < 1e-9: break

            # Candidates for destination
            # 1. MOVE
            for i_idx, m_idx in enumerate(sol[src_gpu]):
                item = m_data[m_idx]

                # Old Src
                src_w_old = stats[src_gpu]['w']
                src_s_old = stats[src_gpu]['s']
                src_p_old = stats[src_gpu]['p']

                # New Src
                src_s_new = src_s_old - item['s']
                src_w_new = src_w_old - item['w']
                rem = GPU_MEM_SIZE - src_s_new
                src_p_new = src_w_new/rem if rem > 1e-9 else float('inf')

                best_dst = None
                best_dst_tuple = (max_p, ssq_p) # (max, sum_sq)

                for dst in range(gpu_num):
                    if dst == src_gpu: continue
                    if stats[dst]['s'] + item['s'] > GPU_MEM_SIZE: continue

                    dst_w_old = stats[dst]['w']
                    dst_s_old = stats[dst]['s']
                    dst_p_old = stats[dst]['p']

                    dst_s_new = dst_s_old + item['s']
                    dst_w_new = dst_w_old + item['w']
                    rem_d = GPU_MEM_SIZE - dst_s_new
                    dst_p_new = dst_w_new/rem_d if rem_d > 1e-9 else float('inf')

                    # New global max check
                    local_pair_max = max(src_p_new, dst_p_new)

                    if local_pair_max > highest_p - 1e-6: continue

                    # Recalculate full metrics roughly
                    new_ssq = ssq_p - (src_p_old**2 + dst_p_old**2) + (src_p_new**2 + dst_p_new**2)

                    new_max = local_pair_max
                    for g in range(gpu_num):
                        if g != src_gpu and g != dst:
                            if stats[g]['p'] > new_max:
                                new_max = stats[g]['p']

                    new_tuple = (new_max, new_ssq)

                    if new_tuple < best_dst_tuple:
                        best_dst_tuple = new_tuple
                        best_dst = dst

                if best_dst is not None:
                    # Apply Move
                    sol[src_gpu].pop(i_idx)
                    sol[best_dst].append(m_idx)

                    m = item
                    stats[src_gpu] = {'w': src_w_new, 's': src_s_new, 'p': src_p_new}

                    d_w = stats[best_dst]['w'] + m['w']
                    d_s = stats[best_dst]['s'] + m['s']
                    rem_d = GPU_MEM_SIZE - d_s
                    d_p = d_w/rem_d if rem_d > 1e-9 else float('inf')
                    stats[best_dst] = {'w': d_w, 's': d_s, 'p': d_p}

                    max_p, ssq_p = best_dst_tuple
                    improved = True
                    break

            if improved: continue

            # 2. SWAP
            for s_idx, m_src_idx in enumerate(sol[src_gpu]):
                m_src = m_data[m_src_idx]

                best_swap_tuple = (max_p, ssq_p)
                best_swap_target = None

                src_p_old = stats[src_gpu]['p']

                for dst in range(gpu_num):
                    if dst == src_gpu: continue
                    dst_p_old = stats[dst]['p']

                    for d_idx, m_dst_idx in enumerate(sol[dst]):
                        m_dst = m_data[m_dst_idx]

                        # Capacity check
                        new_src_s = stats[src_gpu]['s'] - m_src['s'] + m_dst['s']
                        if new_src_s > GPU_MEM_SIZE: continue
                        new_dst_s = stats[dst]['s'] - m_dst['s'] + m_src['s']
                        if new_dst_s > GPU_MEM_SIZE: continue

                        # Pressure check
                        rem_src = GPU_MEM_SIZE - new_src_s
                        new_src_w = stats[src_gpu]['w'] - m_src['w'] + m_dst['w']
                        new_src_p = new_src_w / rem_src if rem_src > 1e-9 else float('inf')

                        rem_dst = GPU_MEM_SIZE - new_dst_s
                        new_dst_w = stats[dst]['w'] - m_dst['w'] + m_src['w']
                        new_dst_p = new_dst_w / rem_dst if rem_dst > 1e-9 else float('inf')

                        local_pair_max = max(new_src_p, new_dst_p)
                        if local_pair_max > highest_p - 1e-6: continue

                        new_ssq = ssq_p - (src_p_old**2 + dst_p_old**2) + (new_src_p**2 + new_dst_p**2)

                        new_max = local_pair_max
                        for g in range(gpu_num):
                            if g != src_gpu and g != dst:
                                if stats[g]['p'] > new_max:
                                    new_max = stats[g]['p']

                        new_tuple = (new_max, new_ssq)
                        if new_tuple < best_swap_tuple:
                            best_swap_tuple = new_tuple
                            best_swap_target = (dst, d_idx)

                if best_swap_target:
                    dst_gpu, d_list_idx = best_swap_target
                    m_dst_idx = sol[dst_gpu][d_list_idx]

                    # Apply Swap
                    sol[src_gpu][s_idx] = m_dst_idx
                    sol[dst_gpu][d_list_idx] = m_src_idx

                    # Update Stats
                    s_w = sum(m_data[i]['w'] for i in sol[src_gpu])
                    s_s = sum(m_data[i]['s'] for i in sol[src_gpu])
                    s_p = s_w/(GPU_MEM_SIZE-s_s) if (GPU_MEM_SIZE-s_s)>1e-9 else float('inf')
                    stats[src_gpu] = {'w': s_w, 's': s_s, 'p': s_p}

                    d_w = sum(m_data[i]['w'] for i in sol[dst_gpu])
                    d_s = sum(m_data[i]['s'] for i in sol[dst_gpu])
                    d_p = d_w/(GPU_MEM_SIZE-d_s) if (GPU_MEM_SIZE-d_s)>1e-9 else float('inf')
                    stats[dst_gpu] = {'w': d_w, 's': d_s, 'p': d_p}

                    max_p, ssq_p = best_swap_tuple
                    improved = True
                    break

            if improved: continue

        return sol, stats, max_p, ssq_p

    # Initial Local Search
    current_sol, curr_stats, curr_max, curr_ssq = local_search(current_sol, curr_stats, curr_max, curr_ssq)
    global_best_sol = [list(x) for x in current_sol]
    global_best_max = curr_max

    # ILS Loop (Perturbation)
    for _ in range(15):
        if global_best_max < 1e-9: break

        # Perturbation: Ruin and Recreate the bottleneck GPU
        bottleneck_gpu = -1
        max_p = -1.0
        for g in range(gpu_num):
            if curr_stats[g]['p'] > max_p:
                max_p = curr_stats[g]['p']
                bottleneck_gpu = g

        if bottleneck_gpu == -1: break

        # Copy state
        new_sol = [list(x) for x in current_sol]

        # Force Move: Eject 2 items from bottleneck to random feasible other GPUs
        if new_sol[bottleneck_gpu]:
            moves = 0
            indices_to_move = list(range(len(new_sol[bottleneck_gpu])))
            random.shuffle(indices_to_move)

            for idx_in_list in indices_to_move:
                m_idx = new_sol[bottleneck_gpu][idx_in_list]
                item = m_data[m_idx]

                targets = list(range(gpu_num))
                random.shuffle(targets)
                moved = False
                for t in targets:
                    if t == bottleneck_gpu: continue
                    s_curr = sum(m_data[i]['s'] for i in new_sol[t])
                    if s_curr + item['s'] <= GPU_MEM_SIZE:
                        new_sol[bottleneck_gpu][idx_in_list] = -1
                        new_sol[t].append(m_idx)
                        moved = True
                        break
                if moved:
                    moves += 1
                    if moves >= 2: break

            new_sol[bottleneck_gpu] = [x for x in new_sol[bottleneck_gpu] if x != -1]

        # Run Local Search on perturbed state
        new_stats, new_max, new_ssq = get_stats(new_sol)
        new_sol, new_stats, new_max, new_ssq = local_search(new_sol, new_stats, new_max, new_ssq)

        # Acceptance: Strict improvement on primary, or any improvement on secondary if primary is equal
        if new_max < global_best_max - 1e-6:
            global_best_max = new_max
            global_best_sol = [list(x) for x in new_sol]
            current_sol = new_sol
            curr_stats, curr_max, curr_ssq = new_stats, new_max, new_ssq
        elif abs(new_max - global_best_max) < 1e-6 and new_ssq < curr_ssq:
             current_sol = new_sol
             curr_stats, curr_max, curr_ssq = new_stats, new_max, new_ssq
        else:
             # Revert
             current_sol = [list(x) for x in global_best_sol]
             curr_stats, curr_max, curr_ssq = get_stats(current_sol)

    # ---------------------------------------------------------
    # 3. Output Formatting
    # ---------------------------------------------------------
    result = {}
    for g, idxs in enumerate(global_best_sol):
        result[g] = [m_data[i]['obj'] for i in idxs]

    return result
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