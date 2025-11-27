# EVOLVE-BLOCK-START
"""Model placement algorithm for minimizing maximum KV cache pressure across GPUs"""

import random
import math

GPU_MEM_SIZE = 80  # GB

def compute_model_placement(gpu_num, models):
    """
    Compute a model placement that minimizes the maximum KVPR across all GPUs.
    Combines Binary Search with BFD packing and Large Neighborhood Search (LNS) refinement.
    """

    # --- Preprocessing ---
    m_data = []
    for m in models:
        w = m.req_rate / m.slo if m.slo > 0 else 0
        s = m.model_size
        # Density: pressure per unit size
        d = w / s if s > 1e-6 else 0
        m_data.append({'w': w, 's': s, 'd': d, 'obj': m})

    # --- Helper: Pressure Calculation ---
    def get_pressure(l, u):
        rem = GPU_MEM_SIZE - u
        if rem <= 1e-6:
            return float('inf') if l > 1e-6 else 0.0
        return l / rem

    # --- Phase 1: Binary Search for Initial Solution ---

    # Calculate initial bounds
    total_w = sum(x['w'] for x in m_data)
    total_s = sum(x['s'] for x in m_data)
    total_rem = GPU_MEM_SIZE * gpu_num - total_s
    # Lower bound based on total load spread over total remaining capacity
    low_bound = total_w / total_rem if total_rem > 1e-6 else 0.0

    # Heuristic packing function for initialization
    def pack_heuristic(sort_key_fn):
        sorted_items = sorted(m_data, key=sort_key_fn, reverse=True)
        placements = [[] for _ in range(gpu_num)]
        loads = [0.0] * gpu_num
        used = [0.0] * gpu_num

        for item in sorted_items:
            best_g = -1
            best_p = float('inf')

            # Minimize resultant pressure
            for g in range(gpu_num):
                if used[g] + item['s'] > GPU_MEM_SIZE - 1e-6: continue

                rem = GPU_MEM_SIZE - used[g] - item['s']
                if rem > 1e-6:
                    p = (loads[g] + item['w']) / rem
                else:
                    p = float('inf') if (loads[g] + item['w']) > 0 else 0

                if p < best_p:
                    best_p = p
                    best_g = g

            if best_g == -1: return None, float('inf')

            placements[best_g].append(item['obj'])
            loads[best_g] += item['w']
            used[best_g] += item['s']

        max_p = 0.0
        for g in range(gpu_num):
            max_p = max(max_p, get_pressure(loads[g], used[g]))
        return placements, max_p

    # Try density sort to get a good upper bound
    init_res, init_max = pack_heuristic(lambda x: x['d'])
    if init_res:
        best_bs_placement = init_res
        high_bound = init_max
    else:
        # Fallback upper bound
        high_bound = 1000.0
        best_bs_placement = None

    # Binary Search Loop
    low = low_bound
    high = high_bound
    if high > 999 and init_res is None: high = 10.0 # Heuristic adjustment if no initial found

    for _ in range(20):
        if high - low < 1e-4: break
        mid = (low + high) / 2.0

        feasible = False
        temp_placement = None

        # Strategies: Density, Virtual Size, Physical Size, Load
        sort_keys = [
            lambda x: x['d'],
            lambda x: x['w'] + mid * x['s'],
            lambda x: x['s'],
            lambda x: x['w']
        ]

        def try_packing(sorted_items):
            p_alloc = [[] for _ in range(gpu_num)]
            g_l = [0.0] * gpu_num
            g_u = [0.0] * gpu_num

            for item in sorted_items:
                best_g = -1
                min_slack = float('inf')

                for g in range(gpu_num):
                    if g_u[g] + item['s'] > GPU_MEM_SIZE - 1e-6: continue

                    rem_capacity = GPU_MEM_SIZE - g_u[g] - item['s']
                    max_load_allowed = mid * rem_capacity
                    current_proj_load = g_l[g] + item['w']

                    if current_proj_load <= max_load_allowed + 1e-7:
                        slack = max_load_allowed - current_proj_load
                        if slack < min_slack:
                            min_slack = slack
                            best_g = g

                if best_g != -1:
                    p_alloc[best_g].append(item['obj'])
                    g_l[best_g] += item['w']
                    g_u[best_g] += item['s']
                else:
                    return None
            return p_alloc

        for key in sort_keys:
            items_sorted = sorted(m_data, key=key, reverse=True)
            res = try_packing(items_sorted)
            if res:
                feasible = True
                temp_placement = res
                break

        # Randomized Fallback if deterministic strategies fail
        if not feasible:
            base_items = list(m_data)
            for _ in range(20):
                random.shuffle(base_items)
                res = try_packing(base_items)
                if res:
                    feasible = True
                    temp_placement = res
                    break

        if feasible:
            best_bs_placement = temp_placement
            high = mid
        else:
            low = mid

    if best_bs_placement is None:
        if init_res:
             best_bs_placement = init_res
        else:
             raise ValueError("No feasible placement found.")

    # --- Phase 2: Large Neighborhood Search (ILS) ---
    current_placement = {i: list(gpu) for i, gpu in enumerate(best_bs_placement)}

    loads = [0.0] * gpu_num
    used = [0.0] * gpu_num
    for g in range(gpu_num):
        for m in current_placement[g]:
            loads[g] += m.req_rate / m.slo
            used[g] += m.model_size

    pressures = [get_pressure(loads[g], used[g]) for g in range(gpu_num)]

    best_max_p = max(pressures)
    best_global_placement = {k: list(v) for k,v in current_placement.items()}

    # ILS Parameters
    iterations = 250

    for _ in range(iterations):
        # 1. Metrics & Global Best Update
        current_max = max(pressures)
        current_sq = sum(p*p for p in pressures)

        if current_max < best_max_p - 1e-8:
            best_max_p = current_max
            best_global_placement = {k: list(v) for k,v in current_placement.items()}

        # 2. Identify Bottleneck
        # Sort GPUs by pressure
        sorted_gpus = sorted(range(gpu_num), key=lambda g: pressures[g], reverse=True)
        bottleneck = sorted_gpus[0]

        # 3. Descent (Best-Improvement in Neighborhood)
        best_move = None
        # (type, partner, idx_bn, idx_pt, n_bl, n_bu, n_pl, n_pu, n_max, n_sq)

        # Precompute top pressures for fast max check
        # We need the max pressure of all GPUs excluding bottleneck and partner
        top_indices = sorted_gpus[:3]

        # Define neighborhood: Bottleneck <-> All Partners
        # Optimization: Filter partners? No, check all to find best relief.
        partners = [g for g in range(gpu_num) if g != bottleneck]

        bn_items = current_placement[bottleneck]

        for partner in partners:
            # Determine max_others
            max_others = 0.0
            for g_idx in top_indices:
                if g_idx != bottleneck and g_idx != partner:
                    max_others = pressures[g_idx]
                    break

            # Base sq sum excluding pair
            base_sq = current_sq - pressures[bottleneck]**2 - pressures[partner]**2

            # A. Try Moving Item: Bottleneck -> Partner
            for i, m in enumerate(bn_items):
                w, s = m.req_rate / m.slo, m.model_size

                if used[partner] + s > GPU_MEM_SIZE - 1e-6: continue

                n_bl = loads[bottleneck] - w
                n_bu = used[bottleneck] - s
                n_pl = loads[partner] + w
                n_pu = used[partner] + s

                p_b = get_pressure(n_bl, n_bu)
                p_p = get_pressure(n_pl, n_pu)

                new_max = max(max_others, p_b, p_p)

                # Strict degradation check
                if new_max > current_max + 1e-9: continue

                new_sq = base_sq + p_b**2 + p_p**2

                # Acceptance criteria: Better Max OR (Equal Max AND Better Variance)
                is_better = False
                if new_max < current_max - 1e-9: is_better = True
                elif new_max < current_max + 1e-9 and new_sq < current_sq - 1e-9: is_better = True

                if is_better:
                    if best_move is None:
                        best_move = ('move', partner, i, -1, n_bl, n_bu, n_pl, n_pu, new_max, new_sq)
                    else:
                        # Compare with best_move found so far
                        bm_max, bm_sq = best_move[8], best_move[9]
                        if new_max < bm_max - 1e-9:
                            best_move = ('move', partner, i, -1, n_bl, n_bu, n_pl, n_pu, new_max, new_sq)
                        elif abs(new_max - bm_max) < 1e-9 and new_sq < bm_sq - 1e-9:
                            best_move = ('move', partner, i, -1, n_bl, n_bu, n_pl, n_pu, new_max, new_sq)

            # B. Try Swapping: Bottleneck <-> Partner
            # Only try if we didn't find a very good move, or check anyway?
            # Swaps are expensive O(N*M). Do if partner not too full.
            pt_items = current_placement[partner]
            for i, m1 in enumerate(bn_items):
                w1, s1 = m1.req_rate / m1.slo, m1.model_size
                for j, m2 in enumerate(pt_items):
                    w2, s2 = m2.req_rate / m2.slo, m2.model_size

                    n_bu = used[bottleneck] - s1 + s2
                    if n_bu > GPU_MEM_SIZE - 1e-6: continue
                    n_pu = used[partner] - s2 + s1
                    if n_pu > GPU_MEM_SIZE - 1e-6: continue

                    n_bl = loads[bottleneck] - w1 + w2
                    n_pl = loads[partner] - w2 + w1

                    p_b = get_pressure(n_bl, n_bu)
                    p_p = get_pressure(n_pl, n_pu)

                    new_max = max(max_others, p_b, p_p)

                    if new_max > current_max + 1e-9: continue
                    new_sq = base_sq + p_b**2 + p_p**2

                    is_better = False
                    if new_max < current_max - 1e-9: is_better = True
                    elif new_max < current_max + 1e-9 and new_sq < current_sq - 1e-9: is_better = True

                    if is_better:
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
            type_, pt, i, j, nbl, nbu, npl, npu, _, _ = best_move
            if type_ == 'move':
                item = current_placement[bottleneck].pop(i)
                current_placement[pt].append(item)
            else:
                item1 = current_placement[bottleneck][i]
                item2 = current_placement[pt][j]
                current_placement[bottleneck][i] = item2
                current_placement[pt][j] = item1

            loads[bottleneck] = nbl
            used[bottleneck] = nbu
            loads[pt] = npl
            used[pt] = npu
            pressures[bottleneck] = get_pressure(loads[bottleneck], used[bottleneck])
            pressures[pt] = get_pressure(loads[pt], used[pt])

        else:
            # 4. Perturbation (Burst Kick) with Multiple Trials
            # If stuck, destroy packing of Bottleneck + k Random Partners
            k_partners = min(gpu_num - 1, 3)
            if k_partners == 0: break

            partner_cands = [g for g in range(gpu_num) if g != bottleneck]
            victims = [bottleneck] + random.sample(partner_cands, k_partners)

            # Extract items
            repack_items = []
            for v in victims:
                repack_items.extend(current_placement[v])
                current_placement[v] = []
                loads[v] = 0.0
                used[v] = 0.0
                pressures[v] = 0.0

            # Try multiple repacking attempts
            best_local_config = None
            best_local_max = float('inf')

            for _ in range(10): # 10 trials
                # Randomized Density Sort
                trial_items = list(repack_items)
                random.shuffle(trial_items)
                trial_items.sort(key=lambda x: (x.req_rate/x.slo)/(x.model_size+1e-6) * random.uniform(0.8, 1.2), reverse=True)

                l_loads = {v: 0.0 for v in victims}
                l_used = {v: 0.0 for v in victims}
                l_placement = {v: [] for v in victims}
                possible = True

                for item in trial_items:
                    w, s = item.req_rate / item.slo, item.model_size
                    best_v = -1
                    best_sc = float('inf')

                    # Best Fit (min pressure)
                    for v in victims:
                        if l_used[v] + s <= GPU_MEM_SIZE - 1e-6:
                            rem = GPU_MEM_SIZE - l_used[v] - s
                            if rem > 1e-6:
                                sc = (l_loads[v] + w) / rem
                            else:
                                sc = float('inf') if (l_loads[v] + w) > 0 else 0

                            if sc < best_sc:
                                best_sc = sc
                                best_v = v

                    # Fallback to Any Fit if Best Fit fails
                    if best_v == -1:
                        for v in victims:
                             if l_used[v] + s <= GPU_MEM_SIZE - 1e-6:
                                 best_v = v
                                 break

                    if best_v != -1:
                        l_placement[best_v].append(item)
                        l_loads[best_v] += w
                        l_used[best_v] += s
                    else:
                        possible = False
                        break

                if possible:
                    local_max = max(get_pressure(l_loads[v], l_used[v]) for v in victims)
                    if local_max < best_local_max:
                        best_local_max = local_max
                        best_local_config = (l_placement, l_loads, l_used)

            if best_local_config:
                # Apply best found local configuration
                l_place, l_l, l_u = best_local_config
                for v in victims:
                    current_placement[v] = l_place[v]
                    loads[v] = l_l[v]
                    used[v] = l_u[v]
                    pressures[v] = get_pressure(loads[v], used[v])
            else:
                # Revert
                current_placement = {k: list(v) for k, v in best_global_placement.items()}
                loads = [0.0]*gpu_num
                used = [0.0]*gpu_num
                for g in range(gpu_num):
                    for m in current_placement[g]:
                        loads[g] += m.req_rate / m.slo
                        used[g] += m.model_size
                pressures = [get_pressure(loads[g], used[g]) for g in range(gpu_num)]

    return best_global_placement

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