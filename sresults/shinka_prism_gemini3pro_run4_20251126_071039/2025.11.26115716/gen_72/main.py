# EVOLVE-BLOCK-START
"""Model placement algorithm for minimizing maximum KV cache pressure across GPUs"""

GPU_MEM_SIZE = 80  # GB

def compute_model_placement(gpu_num, models):
    """
    Compute a model placement that minimizes the maximum KVPR across all GPUs.
    Combines Greedy Heuristics, Binary Search (Bin Packing), and Local Search.
    """

    # Helper to calculate max KVPR of a placement
    def get_max_kvpr(placement):
        max_p = 0.0
        for assigned in placement.values():
            w = sum(m.req_rate / m.slo for m in assigned)
            s = sum(m.model_size for m in assigned)
            rem = GPU_MEM_SIZE - s
            if rem <= 1e-9:
                # If usage is exactly max memory (or close), and weight > 0, pressure is inf.
                # If weight is 0, it's 0. But models have weight.
                if w > 0: return float('inf')
                else: continue
            max_p = max(max_p, w / rem)
        return max_p

    best_placement = None
    best_score = float('inf')

    # ---------------------------------------------------------
    # 1. Greedy Heuristics Ensemble
    # ---------------------------------------------------------
    # Strategies:
    #   'min_result': Place on GPU that minimizes the resulting KVPR (Greedy Min-Max)
    #   'min_current': Place on GPU that has the lowest current KVPR (Load Balancing/Valley Filling)

    heuristics = [
        (lambda m: m.req_rate / m.slo, 'min_result'),
        (lambda m: m.req_rate / m.slo, 'min_current'),
        (lambda m: m.model_size, 'min_result'),
        (lambda m: (m.req_rate / m.slo) / (GPU_MEM_SIZE - m.model_size + 1e-6), 'min_result'),
    ]

    for key_fn, strategy in heuristics:
        sorted_models = sorted(models, key=key_fn, reverse=True)

        placement = {i: [] for i in range(gpu_num)}
        gpu_w = [0.0] * gpu_num
        gpu_s = [0.0] * gpu_num
        possible = True

        for model in sorted_models:
            w = model.req_rate / model.slo
            s = model.model_size

            best_idx = None
            best_val = float('inf')

            for i in range(gpu_num):
                if gpu_s[i] + s > GPU_MEM_SIZE: continue

                rem = GPU_MEM_SIZE - gpu_s[i]

                if strategy == 'min_result':
                    new_rem = rem - s
                    if new_rem > 1e-9:
                        val = (gpu_w[i] + w) / new_rem
                    else:
                        val = float('inf')
                else: # min_current
                    if rem > 1e-9:
                        val = gpu_w[i] / rem
                    else:
                        val = float('inf')

                if val < best_val:
                    best_val = val
                    best_idx = i
                elif val == best_val and best_idx is None:
                    best_idx = i

            if best_idx is None:
                possible = False
                break

            placement[best_idx].append(model)
            gpu_w[best_idx] += w
            gpu_s[best_idx] += s

        if possible:
            score = get_max_kvpr(placement)
            if score < best_score:
                best_score = score
                best_placement = placement

    # ---------------------------------------------------------
    # 2. Binary Search on Target KVPR (Transformation to Bin Packing)
    # ---------------------------------------------------------
    # Problem: Minimize K such that for all bins j: sum(w_i) / (C - sum(s_i)) <= K
    # Transformed to Bin Packing: v_i(K) = w_i + K*s_i <= K*C_j

    import random

    # Precompute data for speed and reuse
    m_data = []
    total_w = 0.0
    total_s = 0.0
    for i, m in enumerate(models):
        w = m.req_rate / m.slo
        s = m.model_size
        m_data.append({'w': w, 's': s, 'obj': m, 'id': i})
        total_w += w
        total_s += s

    def solve_bin_packing_for_k(target_k):
        """
        Attempts to pack models given target pressure K.
        Uses multiple sorting strategies to find a valid packing.
        Returns placement dict if feasible, else None.
        """
        # Strategies to sort items:
        # 1. Virtual Size: w + K*s (Standard BFD for this constraint)
        # 2. Physical Size: s
        # 3. Weight: w
        # 4. Stochastic Virtual Size: (w + K*s) * noise

        strategies = []
        # Deterministic strategies
        strategies.append(lambda x: x['w'] + target_k * x['s'])
        strategies.append(lambda x: x['s'])

        # Stochastic trials to break tie/local optima
        for _ in range(5):
            strategies.append(lambda x: (x['w'] + target_k * x['s']) * random.uniform(0.9, 1.1))

        for key_func in strategies:
            # Sort descending
            sorted_indices = sorted(range(len(m_data)), key=lambda i: key_func(m_data[i]), reverse=True)

            bins = [{'w': 0.0, 's': 0.0, 'idxs': []} for _ in range(gpu_num)]
            possible = True

            for idx in sorted_indices:
                item = m_data[idx]
                w, s = item['w'], item['s']

                best_bin = None
                min_slack = float('inf')

                # Best Fit Decreasing (Minimizing slack on the constraint)
                # Constraint: bin_w + w <= K * (C - bin_s - s)
                # Slack = K * (C - bin_s - s) - (bin_w + w)

                for b_idx in range(gpu_num):
                    b = bins[b_idx]

                    if b['s'] + s > GPU_MEM_SIZE: continue

                    rem_s = GPU_MEM_SIZE - (b['s'] + s)
                    rhs = target_k * rem_s
                    lhs = b['w'] + w

                    if lhs <= rhs + 1e-5:
                        # Feasible
                        slack = rhs - lhs
                        if slack < min_slack:
                            min_slack = slack
                            best_bin = b_idx

                if best_bin is None:
                    possible = False
                    break

                bins[best_bin]['idxs'].append(idx)
                bins[best_bin]['w'] += w
                bins[best_bin]['s'] += s

            if possible:
                # Convert back to object placement
                res = {}
                for g in range(gpu_num):
                    res[g] = [m_data[i]['obj'] for i in bins[g]['idxs']]
                return res
        return None

    rem_global = gpu_num * GPU_MEM_SIZE - total_s
    if rem_global > 1e-6:
        low = total_w / rem_global
        high = best_score if best_score != float('inf') else 1000.0

        if high > low + 1e-4:
            # Binary Search
            for _ in range(16):
                mid = (low + high) / 2.0

                placement = solve_bin_packing_for_k(mid)

                if placement:
                    # Update best known solution
                    curr_max = get_max_kvpr(placement)
                    if curr_max < best_score:
                        best_score = curr_max
                        best_placement = placement
                    high = mid
                else:
                    low = mid

    if best_placement is None:
        raise ValueError("Unable to place models on GPUs with available memory.")

    # ---------------------------------------------------------
    # 3. Iterated Local Search (ILS)
    # ---------------------------------------------------------
    # Refines solution by minimizing lexicographical objective: (Max KVPR, Sum of Squared KVPR)
    # Allows side-steps on Max KVPR to improve balance, preventing local optima.
    # Uses random perturbations when stuck.

    # Data setup
    curr_sol = {g: list(best_placement[g]) for g in range(gpu_num)}

    # State tracking
    g_stats = []
    for g in range(gpu_num):
        w = sum(m.req_rate / m.slo for m in curr_sol[g])
        s = sum(m.model_size for m in curr_sol[g])
        rem = GPU_MEM_SIZE - s
        p = w / rem if rem > 1e-9 else (float('inf') if w > 0 else 0.0)
        g_stats.append({'w': w, 's': s, 'p': p})

    def get_metrics(stats):
        """Returns (max_p, sum_sq_p)"""
        m_p = 0.0
        ssq = 0.0
        for st in stats:
            p = st['p']
            if p > m_p: m_p = p
            ssq += p * p
        return m_p, ssq

    curr_max, curr_ssq = get_metrics(g_stats)

    # Keep track of the absolute best solution found
    best_final_sol = {g: list(curr_sol[g]) for g in range(gpu_num)}
    best_final_max = curr_max

    no_improve_iters = 0
    max_iters = 300 # Computational budget

    for _ in range(max_iters):
        if curr_max < 1e-9: break

        # 3.1 Identification of Bottleneck
        # Select GPU with the maximum pressure
        candidates = [g for g in range(gpu_num) if abs(g_stats[g]['p'] - curr_max) < 1e-6]
        if not candidates: break

        # If stuck, maybe perturb one of the bottlenecks?
        src_gpu = random.choice(candidates)
        src_items = curr_sol[src_gpu]

        improved = False

        # 3.2 Move Operator
        # Try moving an item from bottleneck to another GPU
        move_indices = list(range(len(src_items)))
        random.shuffle(move_indices) # Randomize order

        for m_idx in move_indices:
            model = src_items[m_idx]
            w, s = model.req_rate / model.slo, model.model_size

            # Src State Update
            src_s = g_stats[src_gpu]['s'] - s
            src_w = g_stats[src_gpu]['w'] - w
            src_rem = GPU_MEM_SIZE - src_s
            src_p = src_w / src_rem if src_rem > 1e-9 else float('inf')

            # Check all destinations
            # Heuristic: Check GPUs with low pressure first?
            # Random order helps exploration
            dst_indices = list(range(gpu_num))
            random.shuffle(dst_indices)

            for dst in dst_indices:
                if dst == src_gpu: continue
                if g_stats[dst]['s'] + s > GPU_MEM_SIZE: continue

                # Dst State Update
                dst_s = g_stats[dst]['s'] + s
                dst_w = g_stats[dst]['w'] + w
                dst_rem = GPU_MEM_SIZE - dst_s
                dst_p = dst_w / dst_rem if dst_rem > 1e-9 else float('inf')

                # Check metrics
                # Calculate new global max and ssq efficiently
                # Delta SSQ: (src_p^2 + dst_p^2) - (old_src_p^2 + old_dst_p^2)
                # Global Max: max(others_max, src_p, dst_p)

                # Fast reject: if local max > curr_max, we definitely didn't improve global max
                if max(src_p, dst_p) > curr_max + 1e-9: continue

                old_pair_ssq = g_stats[src_gpu]['p']**2 + g_stats[dst]['p']**2
                new_pair_ssq = src_p**2 + dst_p**2
                new_ssq = curr_ssq - old_pair_ssq + new_pair_ssq

                # Recompute true max (others might be at curr_max)
                new_max = max(src_p, dst_p)
                # If new local max is less than curr_max, we might have reduced global max
                # But we need to verify if other GPUs hold the max
                if new_max < curr_max:
                    for g_chk in range(gpu_num):
                        if g_chk == src_gpu or g_chk == dst: continue
                        if g_stats[g_chk]['p'] > new_max:
                            new_max = g_stats[g_chk]['p']

                # Acceptance Criteria: Lexicographical (Max, SSQ)
                # 1. Strictly lower Max
                # 2. Same Max (within epsilon), strictly lower SSQ
                accept = False
                if new_max < curr_max - 1e-6:
                    accept = True
                elif abs(new_max - curr_max) < 1e-6 and new_ssq < curr_ssq - 1e-6:
                    accept = True

                if accept:
                    # Execute Move
                    curr_sol[src_gpu].pop(m_idx)
                    curr_sol[dst].append(model)
                    g_stats[src_gpu] = {'w': src_w, 's': src_s, 'p': src_p}
                    g_stats[dst] = {'w': dst_w, 's': dst_s, 'p': dst_p}
                    curr_max, curr_ssq = new_max, new_ssq
                    improved = True
                    break
            if improved: break

        if improved:
            no_improve_iters = 0
            if curr_max < best_final_max:
                best_final_max = curr_max
                best_final_sol = {g: list(curr_sol[g]) for g in range(gpu_num)}
            continue

        # 3.3 Swap Operator
        # If move didn't help, try swapping items
        for m_idx in move_indices: # Reuse shuffled indices
            m_src = src_items[m_idx]
            w_src, s_src = m_src.req_rate / m_src.slo, m_src.model_size

            dst_indices = list(range(gpu_num))
            random.shuffle(dst_indices)

            for dst in dst_indices:
                if dst == src_gpu: continue

                dst_items = curr_sol[dst]
                for d_idx, m_dst in enumerate(dst_items):
                    w_dst, s_dst = m_dst.req_rate / m_dst.slo, m_dst.model_size

                    # Capacity Check
                    n_src_s = g_stats[src_gpu]['s'] - s_src + s_dst
                    if n_src_s > GPU_MEM_SIZE: continue
                    n_dst_s = g_stats[dst]['s'] - s_dst + s_src
                    if n_dst_s > GPU_MEM_SIZE: continue

                    # Pressure Update
                    n_src_w = g_stats[src_gpu]['w'] - w_src + w_dst
                    n_src_rem = GPU_MEM_SIZE - n_src_s
                    n_src_p = n_src_w / n_src_rem if n_src_rem > 1e-9 else float('inf')

                    n_dst_w = g_stats[dst]['w'] - w_dst + w_src
                    n_dst_rem = GPU_MEM_SIZE - n_dst_s
                    n_dst_p = n_dst_w / n_dst_rem if n_dst_rem > 1e-9 else float('inf')

                    if max(n_src_p, n_dst_p) > curr_max + 1e-9: continue

                    # Metrics
                    old_pair_ssq = g_stats[src_gpu]['p']**2 + g_stats[dst]['p']**2
                    new_pair_ssq = n_src_p**2 + n_dst_p**2
                    new_ssq = curr_ssq - old_pair_ssq + new_pair_ssq

                    new_max = max(n_src_p, n_dst_p)
                    if new_max < curr_max:
                        for g_chk in range(gpu_num):
                            if g_chk == src_gpu or g_chk == dst: continue
                            if g_stats[g_chk]['p'] > new_max:
                                new_max = g_stats[g_chk]['p']

                    accept = False
                    if new_max < curr_max - 1e-6:
                        accept = True
                    elif abs(new_max - curr_max) < 1e-6 and new_ssq < curr_ssq - 1e-6:
                        accept = True

                    if accept:
                        # Execute Swap
                        curr_sol[src_gpu][m_idx] = m_dst
                        curr_sol[dst][d_idx] = m_src
                        g_stats[src_gpu] = {'w': n_src_w, 's': n_src_s, 'p': n_src_p}
                        g_stats[dst] = {'w': n_dst_w, 's': n_dst_s, 'p': n_dst_p}
                        curr_max, curr_ssq = new_max, new_ssq
                        improved = True
                        break
                if improved: break
            if improved: break

        if improved:
            no_improve_iters = 0
            if curr_max < best_final_max:
                best_final_max = curr_max
                best_final_sol = {g: list(curr_sol[g]) for g in range(gpu_num)}
            continue

        # 3.4 Perturbation (Kick)
        # If we are here, no greedy move/swap improved the solution
        no_improve_iters += 1
        if no_improve_iters > 5:
            # Randomly force a valid move from bottleneck
            # to shake up the configuration
            if src_items:
                m_idx = random.randrange(len(src_items))
                model = src_items[m_idx]

                targets = list(range(gpu_num))
                random.shuffle(targets)
                for t in targets:
                    if t == src_gpu: continue
                    if g_stats[t]['s'] + model.model_size <= GPU_MEM_SIZE:
                        # Force Move
                        curr_sol[src_gpu].pop(m_idx)
                        curr_sol[t].append(model)

                        # Recalculate stats completely for safety/simplicity
                        for g in [src_gpu, t]:
                            w = sum(m.req_rate / m.slo for m in curr_sol[g])
                            s = sum(m.model_size for m in curr_sol[g])
                            rem = GPU_MEM_SIZE - s
                            p = w / rem if rem > 1e-9 else (float('inf') if w > 0 else 0.0)
                            g_stats[g] = {'w': w, 's': s, 'p': p}

                        curr_max, curr_ssq = get_metrics(g_stats)
                        no_improve_iters = 0 # Reset counter after kick
                        break

    return best_final_sol

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