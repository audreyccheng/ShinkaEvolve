# EVOLVE-BLOCK-START
"""Model placement algorithm for minimizing maximum KV cache pressure across GPUs"""

GPU_MEM_SIZE = 80  # GB

def compute_model_placement(gpu_num, models):
    """
    Compute a model placement that minimizes the maximum KVPR across all GPUs.
    Combines Binary Search with Multi-Strategy Best-Fit packing and Local Search refinement.
    """

    # Precompute model data
    m_data = []
    for m in models:
        w = m.req_rate / m.slo
        s = m.model_size
        m_data.append({'w': w, 's': s, 'obj': m})

    # --- 1. Heuristic Initialization for Binary Search Bounds ---
    # We need a feasible upper bound.

    def try_pack_with_sort(sort_key):
        sorted_items = sorted(m_data, key=sort_key, reverse=True)
        placements = [[] for _ in range(gpu_num)]
        loads = [0.0] * gpu_num
        used = [0.0] * gpu_num

        for item in sorted_items:
            best_g = -1
            best_score = float('inf')

            for g in range(gpu_num):
                rem = GPU_MEM_SIZE - used[g] - item['s']
                if rem > 1e-6:
                    # Minimize resulting pressure
                    p = (loads[g] + item['w']) / rem
                    if p < best_score:
                        best_score = p
                        best_g = g

            if best_g == -1: return None, float('inf')
            placements[best_g].append(item['obj'])
            loads[best_g] += item['w']
            used[best_g] += item['s']

        max_p = 0.0
        for g in range(gpu_num):
            rem = GPU_MEM_SIZE - used[g]
            if rem <= 1e-6:
                if loads[g] > 0: return None, float('inf')
            else:
                max_p = max(max_p, loads[g]/rem)
        return placements, max_p

    # Try density sort (usually best)
    init_placement, upper_bound = try_pack_with_sort(lambda x: x['w'] / x['s'] if x['s'] > 0 else 0)

    # If fails, try size sort
    if init_placement is None:
        init_placement, upper_bound = try_pack_with_sort(lambda x: x['s'])

    # If still fails, use loose bound (assuming feasible solution exists)
    if init_placement is None:
        upper_bound = 1000.0

    # --- 2. Binary Search for Optimal K ---
    low = 0.0
    high = upper_bound
    final_placement = init_placement

    for _ in range(20):
        if high - low < 1e-4: break
        mid = (low + high) / 2.0

        feasible = False
        res_placement = None

        # Strategies: Virtual Size, Physical Size, Load, Density
        strategies = [
            lambda x: x['w'] + mid * x['s'],
            lambda x: x['s'],
            lambda x: x['w'],
            lambda x: x['w'] / x['s'] if x['s'] > 1e-6 else 0
        ]

        for key_func in strategies:
            # Best Fit Decreasing
            items_sorted = sorted(m_data, key=key_func, reverse=True)
            gpu_models = [[] for _ in range(gpu_num)]
            gpu_l = [0.0] * gpu_num
            gpu_u = [0.0] * gpu_num
            ok = True

            for item in items_sorted:
                best_g = -1
                min_residual = float('inf')

                # Check all bins
                for g in range(gpu_num):
                    # Hard mem check
                    if gpu_u[g] + item['s'] >= GPU_MEM_SIZE - 1e-6: continue

                    # Constraint check: (L + w) <= K * (C - (U + s))
                    # Transformed: L + w + K(U + s) <= KC
                    lhs = (gpu_l[g] + item['w']) + mid * (gpu_u[g] + item['s'])
                    rhs = mid * GPU_MEM_SIZE

                    if lhs <= rhs + 1e-7:
                        # Best Fit: minimize residual capacity of transformed bin
                        res = rhs - lhs
                        if res < min_residual:
                            min_residual = res
                            best_g = g

                if best_g != -1:
                    gpu_models[best_g].append(item['obj'])
                    gpu_l[best_g] += item['w']
                    gpu_u[best_g] += item['s']
                else:
                    ok = False
                    break

            if ok:
                feasible = True
                res_placement = {i: gpu_models[i] for i in range(gpu_num)}
                break

        if feasible:
            final_placement = res_placement
            high = mid
        else:
            low = mid

    if final_placement is None:
        raise ValueError("Could not find feasible placement")

    # --- 3. Iterated Local Search (Best-Improvement + Variance Tie-Breaking) ---
    import random

    current_placement = final_placement

    # Initialize state
    loads = [0.0] * gpu_num
    used = [0.0] * gpu_num
    for g in range(gpu_num):
        for m in current_placement[g]:
            loads[g] += m.req_rate / m.slo
            used[g] += m.model_size

    best_placement = {k: list(v) for k, v in current_placement.items()}

    def get_pressure(l, u):
        rem = GPU_MEM_SIZE - u
        if rem <= 1e-6: return float('inf') if l > 1e-6 else 0.0
        return l / rem

    # Initial metrics
    pressures = [get_pressure(loads[g], used[g]) for g in range(gpu_num)]
    best_max_p = max(pressures)

    # ILS Loop
    max_iters = 300 # Best-improvement is slower, reduce iters slightly
    for iteration in range(max_iters):
        # Current Global Metrics
        current_max_p = max(pressures)
        current_sum_sq = sum(p*p for p in pressures)

        # Update Global Best
        if current_max_p < best_max_p - 1e-7:
            best_max_p = current_max_p
            best_placement = {k: list(v) for k, v in current_placement.items()}

        # Identify bottleneck
        bottleneck = -1
        max_val = -1.0
        for g in range(gpu_num):
            if pressures[g] > max_val:
                max_val = pressures[g]
                bottleneck = g

        if bottleneck == -1: break # Should not happen unless empty

        # --- Best-Improvement Descent ---
        best_move = None
        # (move_type, partner, idx_bn, idx_pt, n_bl, n_bu, n_pl, n_pu, score_max, score_sq)

        # Precompute top pressures to quickly determine max excluding {bn, partner}
        # We need top 3 because bn is definitely in top 1 (or tied), partner might be top 2.
        sorted_indices = sorted(range(gpu_num), key=lambda i: pressures[i], reverse=True)
        top_indices = sorted_indices[:3]

        bn_items = current_placement[bottleneck]

        for partner in range(gpu_num):
            if partner == bottleneck: continue

            # Determine max pressure of "others" (excluding bn and partner)
            max_others = 0.0
            for idx in top_indices:
                if idx != bottleneck and idx != partner:
                    max_others = pressures[idx]
                    break

            # Base SumSq excluding bn and partner
            base_sq = current_sum_sq - (pressures[bottleneck]**2 + pressures[partner]**2)

            # 1. Try Moving item from Bottleneck -> Partner
            for i, m in enumerate(bn_items):
                w, s = m.req_rate / m.slo, m.model_size
                if used[partner] + s >= GPU_MEM_SIZE - 1e-6: continue

                n_bl = loads[bottleneck] - w
                n_bu = used[bottleneck] - s
                n_pl = loads[partner] + w
                n_pu = used[partner] + s

                p_b = get_pressure(n_bl, n_bu)
                p_p = get_pressure(n_pl, n_pu)

                new_max = max(max_others, p_b, p_p)

                # Filter: Don't allow strict degradation of max pressure
                if new_max > current_max_p + 1e-9: continue

                new_sq = base_sq + p_b**2 + p_p**2

                is_improvement = False
                if new_max < current_max_p - 1e-9:
                    is_improvement = True
                elif new_max < current_max_p + 1e-9 and new_sq < current_sum_sq - 1e-9:
                    is_improvement = True

                if is_improvement:
                    # Update best move
                    if best_move is None:
                        best_move = ('move', partner, i, -1, n_bl, n_bu, n_pl, n_pu, new_max, new_sq)
                    else:
                        bm_max, bm_sq = best_move[8], best_move[9]
                        if new_max < bm_max - 1e-9:
                            best_move = ('move', partner, i, -1, n_bl, n_bu, n_pl, n_pu, new_max, new_sq)
                        elif abs(new_max - bm_max) < 1e-9 and new_sq < bm_sq - 1e-9:
                            best_move = ('move', partner, i, -1, n_bl, n_bu, n_pl, n_pu, new_max, new_sq)

            # 2. Try Swapping items
            pt_items = current_placement[partner]
            for i, m1 in enumerate(bn_items):
                w1, s1 = m1.req_rate / m1.slo, m1.model_size
                for j, m2 in enumerate(pt_items):
                    w2, s2 = m2.req_rate / m2.slo, m2.model_size

                    n_bu = used[bottleneck] - s1 + s2
                    if n_bu >= GPU_MEM_SIZE - 1e-6: continue
                    n_pu = used[partner] - s2 + s1
                    if n_pu >= GPU_MEM_SIZE - 1e-6: continue

                    n_bl = loads[bottleneck] - w1 + w2
                    n_pl = loads[partner] - w2 + w1

                    p_b = get_pressure(n_bl, n_bu)
                    p_p = get_pressure(n_pl, n_pu)

                    new_max = max(max_others, p_b, p_p)

                    if new_max > current_max_p + 1e-9: continue

                    new_sq = base_sq + p_b**2 + p_p**2

                    is_improvement = False
                    if new_max < current_max_p - 1e-9:
                        is_improvement = True
                    elif new_max < current_max_p + 1e-9 and new_sq < current_sum_sq - 1e-9:
                        is_improvement = True

                    if is_improvement:
                        if best_move is None:
                            best_move = ('swap', partner, i, j, n_bl, n_bu, n_pl, n_pu, new_max, new_sq)
                        else:
                            bm_max, bm_sq = best_move[8], best_move[9]
                            if new_max < bm_max - 1e-9:
                                best_move = ('swap', partner, i, j, n_bl, n_bu, n_pl, n_pu, new_max, new_sq)
                            elif abs(new_max - bm_max) < 1e-9 and new_sq < bm_sq - 1e-9:
                                best_move = ('swap', partner, i, j, n_bl, n_bu, n_pl, n_pu, new_max, new_sq)

        # Apply Move
        if best_move:
            mtype, partner, i, j, nbl, nbu, npl, npu, _, _ = best_move
            if mtype == 'move':
                item = current_placement[bottleneck].pop(i)
                current_placement[partner].append(item)
            else:
                item1 = current_placement[bottleneck][i]
                item2 = current_placement[partner][j]
                current_placement[bottleneck][i] = item2
                current_placement[partner][j] = item1

            loads[bottleneck] = nbl
            used[bottleneck] = nbu
            loads[partner] = npl
            used[partner] = npu
            pressures[bottleneck] = get_pressure(loads[bottleneck], used[bottleneck])
            pressures[partner] = get_pressure(loads[partner], used[partner])
            continue

        # --- Perturbation (Ruin & Recreate) ---
        # Select victims: bottleneck + 2 random
        candidates = [g for g in range(gpu_num) if g != bottleneck]
        if not candidates: break

        victims = {bottleneck}
        victims.update(random.sample(candidates, min(2, len(candidates))))
        victim_list = list(victims)

        # Ruin
        removed_models = []
        for v in victim_list:
            removed_models.extend(current_placement[v])
            current_placement[v] = []
            loads[v] = 0.0
            used[v] = 0.0
            pressures[v] = 0.0

        # Recreate: Best-Fit Decreasing by Density with randomization
        # Add slight noise to density to explore different packings
        removed_models.sort(key=lambda x: ((x.req_rate/x.slo)/(x.model_size+1e-6)) * random.uniform(0.95, 1.05), reverse=True)

        feasible_repack = True
        for m in removed_models:
            w = m.req_rate / m.slo
            s = m.model_size
            best_v = -1
            min_p = float('inf')

            for v in victim_list:
                rem = GPU_MEM_SIZE - used[v] - s
                if rem > 1e-6:
                    p = (loads[v] + w) / rem
                    if p < min_p:
                        min_p = p
                        best_v = v

            # Fallback
            if best_v == -1:
                for v in victim_list:
                    if used[v] + s <= GPU_MEM_SIZE - 1e-6:
                        best_v = v
                        break

            if best_v != -1:
                current_placement[best_v].append(m)
                loads[best_v] += w
                used[best_v] += s
            else:
                feasible_repack = False
                break

        if feasible_repack:
            for v in victim_list:
                pressures[v] = get_pressure(loads[v], used[v])
        else:
            # Revert
            current_placement = {k: list(v) for k, v in best_placement.items()}
            # Recompute state
            loads = [0.0]*gpu_num
            used = [0.0]*gpu_num
            for g in range(gpu_num):
                for m in current_placement[g]:
                    loads[g] += m.req_rate / m.slo
                    used[g] += m.model_size
            pressures = [get_pressure(loads[g], used[g]) for g in range(gpu_num)]
            # If we are failing to pack in perturbation repeatedly, we might be stuck
            if iteration > max_iters * 0.9: break

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