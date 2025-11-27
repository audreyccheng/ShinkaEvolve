# EVOLVE-BLOCK-START
"""Model placement algorithm for minimizing maximum KV cache pressure across GPUs"""

import random

GPU_MEM_SIZE = 80  # GB

def compute_model_placement(gpu_num, models):
    """
    Compute a model placement that minimizes the maximum KVPR across all GPUs.
    Combines Binary Search with Multi-Strategy packing and Smart-Perturbation ILS.
    """

    # Precompute model data
    m_data = []
    for m in models:
        w = m.req_rate / m.slo
        s = m.model_size
        m_data.append({'w': w, 's': s, 'obj': m})

    # --- 1. Heuristic Initialization ---
    def heuristic_pack(sort_key):
        sorted_items = sorted(m_data, key=sort_key, reverse=True)
        placements = [[] for _ in range(gpu_num)]
        loads = [0.0] * gpu_num
        used = [0.0] * gpu_num

        for item in sorted_items:
            best_g = -1
            best_p = float('inf')

            for g in range(gpu_num):
                rem = GPU_MEM_SIZE - used[g] - item['s']
                if rem > 1e-6:
                    p = (loads[g] + item['w']) / rem
                    if p < best_p:
                        best_p = p
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
    init_placement, upper_bound = heuristic_pack(lambda x: x['w'] / x['s'] if x['s'] > 0 else 0)
    if init_placement is None:
        init_placement, upper_bound = heuristic_pack(lambda x: x['s'])
    if init_placement is None:
        upper_bound = 1000.0

    # --- 2. Binary Search for Optimal K ---
    # Theoretical lower bound: Total Load / Total Free Capacity
    total_load = sum(d['w'] for d in m_data)
    total_size = sum(d['s'] for d in m_data)
    total_cap = gpu_num * GPU_MEM_SIZE
    low = total_load / (total_cap - total_size) if (total_cap - total_size) > 1e-6 else 0.0
    if low > upper_bound: low = 0.0

    high = upper_bound
    final_placement = init_placement

    # Helper for packing check
    def pack_check(items, k_target):
        # Best Fit check with transformed constraint
        # (L+w)/(C-(U+s)) <= K  => L+w+K(U+s) <= KC
        # LHS = (L+w) + K(U+s). RHS = K*C.
        # Minimize Residual: RHS - LHS

        p_gpu_models = [[] for _ in range(gpu_num)]
        p_gpu_l = [0.0] * gpu_num
        p_gpu_u = [0.0] * gpu_num

        for item in items:
            best_g = -1
            min_res = float('inf')

            w, s = item['w'], item['s']

            for g in range(gpu_num):
                if p_gpu_u[g] + s >= GPU_MEM_SIZE - 1e-6: continue

                lhs = (p_gpu_l[g] + w) + k_target * (p_gpu_u[g] + s)
                rhs = k_target * GPU_MEM_SIZE

                if lhs <= rhs + 1e-7:
                    res = rhs - lhs
                    if res < min_res:
                        min_res = res
                        best_g = g

            if best_g == -1: return None
            p_gpu_models[best_g].append(item['obj'])
            p_gpu_l[best_g] += w
            p_gpu_u[best_g] += s

        return {i: p_gpu_models[i] for i in range(gpu_num)}

    for _ in range(20):
        if high - low < 1e-4: break
        mid = (low + high) / 2.0

        feasible = False
        res_placement = None

        # 1. Deterministic Strategies (BFD)
        # Sort keys: Transformed Size, Physical Size, Load, Density
        keys = [
            lambda x: x['w'] + mid * x['s'],
            lambda x: x['s'],
            lambda x: x['w'],
            lambda x: x['w'] / x['s'] if x['s'] > 1e-6 else 0
        ]

        for key in keys:
            sorted_items = sorted(m_data, key=key, reverse=True)
            res = pack_check(sorted_items, mid)
            if res:
                feasible = True
                res_placement = res
                break

        # 2. Randomized Fallback (Random Order Best Fit)
        if not feasible:
            items_copy = list(m_data)
            for _ in range(30): # 30 trials to reduce false negatives
                random.shuffle(items_copy)
                res = pack_check(items_copy, mid)
                if res:
                    feasible = True
                    res_placement = res
                    break

        if feasible:
            final_placement = res_placement
            high = mid
        else:
            low = mid

    if final_placement is None:
        raise ValueError("No feasible placement found")

    # --- 3. Iterated Local Search ---
    current_placement = final_placement

    # State tracking
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

    pressures = [get_pressure(loads[g], used[g]) for g in range(gpu_num)]
    best_max_p = max(pressures)

    # ILS Parameters
    max_iters = 1000

    for iteration in range(max_iters):
        current_max_p = max(pressures)
        current_sum_sq = sum(p*p for p in pressures)

        if current_max_p < best_max_p - 1e-7:
            best_max_p = current_max_p
            best_placement = {k: list(v) for k, v in current_placement.items()}

        # Bottleneck
        bottleneck = -1
        max_val = -1.0
        for g in range(gpu_num):
            if pressures[g] > max_val:
                max_val = pressures[g]
                bottleneck = g

        if bottleneck == -1: break

        # --- Descent: Ordered First-Improvement ---
        found_improvement = False

        # Identify top pressures excluding bottleneck for efficient max checks
        other_indices = [g for g in range(gpu_num) if g != bottleneck]
        top_others = sorted(other_indices, key=lambda g: pressures[g], reverse=True)[:2]

        # Sort partners by pressure ascending (target least loaded first)
        sorted_partners = sorted(other_indices, key=lambda g: pressures[g])

        # Prepare bottleneck items with data: (index, w, s, obj)
        bn_items = []
        for i, m in enumerate(current_placement[bottleneck]):
            bn_items.append((i, m.req_rate/m.slo, m.model_size, m))
        # Sort by load descending (move heavy items first)
        bn_items.sort(key=lambda x: x[1], reverse=True)

        # 1. Try Moves
        for partner in sorted_partners:
            # Max pressure of rest of world
            if partner == top_others[0]:
                max_rest = pressures[top_others[1]] if len(top_others) > 1 else 0.0
            else:
                max_rest = pressures[top_others[0]] if len(top_others) > 0 else 0.0

            sq_base = current_sum_sq - (pressures[bottleneck]**2 + pressures[partner]**2)

            for idx, w, s, item_obj in bn_items:
                if used[partner] + s >= GPU_MEM_SIZE - 1e-6: continue

                n_bl = loads[bottleneck] - w
                n_bu = used[bottleneck] - s
                n_pl = loads[partner] + w
                n_pu = used[partner] + s

                pb = get_pressure(n_bl, n_bu)
                pp = get_pressure(n_pl, n_pu)

                nm = max(max_rest, pb, pp)

                if nm > current_max_p + 1e-9: continue

                # Acceptance
                better = False
                if nm < current_max_p - 1e-9:
                    better = True
                elif nm < current_max_p + 1e-9:
                    nsq = sq_base + pb**2 + pp**2
                    if nsq < current_sum_sq - 1e-9:
                        better = True

                if better:
                    current_placement[bottleneck].pop(idx)
                    current_placement[partner].append(item_obj)
                    loads[bottleneck], used[bottleneck] = n_bl, n_bu
                    loads[partner], used[partner] = n_pl, n_pu
                    pressures[bottleneck] = pb
                    pressures[partner] = pp
                    found_improvement = True
                    break
            if found_improvement: break

        # 2. Try Swaps (only if no move found)
        if not found_improvement:
            for partner in sorted_partners:
                if partner == top_others[0]:
                    max_rest = pressures[top_others[1]] if len(top_others) > 1 else 0.0
                else:
                    max_rest = pressures[top_others[0]] if len(top_others) > 0 else 0.0

                sq_base = current_sum_sq - (pressures[bottleneck]**2 + pressures[partner]**2)
                pt_items = current_placement[partner]

                for idx1, w1, s1, m1 in bn_items:
                    for idx2, m2 in enumerate(pt_items):
                        w2, s2 = m2.req_rate/m2.slo, m2.model_size

                        n_bu = used[bottleneck] - s1 + s2
                        if n_bu >= GPU_MEM_SIZE - 1e-6: continue
                        n_pu = used[partner] - s2 + s1
                        if n_pu >= GPU_MEM_SIZE - 1e-6: continue

                        n_bl = loads[bottleneck] - w1 + w2
                        n_pl = loads[partner] - w2 + w1

                        pb = get_pressure(n_bl, n_bu)
                        pp = get_pressure(n_pl, n_pu)

                        nm = max(max_rest, pb, pp)

                        if nm > current_max_p + 1e-9: continue

                        better = False
                        if nm < current_max_p - 1e-9:
                            better = True
                        elif nm < current_max_p + 1e-9:
                            nsq = sq_base + pb**2 + pp**2
                            if nsq < current_sum_sq - 1e-9:
                                better = True

                        if better:
                            current_placement[bottleneck][idx1] = m2
                            current_placement[partner][idx2] = m1
                            loads[bottleneck], used[bottleneck] = n_bl, n_bu
                            loads[partner], used[partner] = n_pl, n_pu
                            pressures[bottleneck] = pb
                            pressures[partner] = pp
                            found_improvement = True
                            break
                    if found_improvement: break
                if found_improvement: break

        if found_improvement: continue

        # --- Smart Perturbation (Multi-Strategy Burst Kick) ---
        # Select victims: Bottleneck + Randomly select 2-3 others
        # We favor least loaded but also allow random for diversity
        victims = {bottleneck}
        candidates = [g for g in range(gpu_num) if g != bottleneck]

        # Strategy: Pick 1 Least Loaded, then fill up to k random
        if candidates:
            # Find least loaded
            min_p = float('inf')
            min_g = -1
            for g in candidates:
                if pressures[g] < min_p:
                    min_p = pressures[g]
                    min_g = g
            if min_g != -1:
                victims.add(min_g)
                candidates.remove(min_g)

        # Add random victims up to target size (e.g. 4)
        target_victims = min(len(candidates) + len(victims), 4)
        while len(victims) < target_victims and candidates:
            c = random.choice(candidates)
            victims.add(c)
            candidates.remove(c)

        victim_list = list(victims)

        # Extract items
        repack_items = []
        for v in victim_list:
            repack_items.extend(current_placement[v])
            current_placement[v] = []
            loads[v] = 0.0
            used[v] = 0.0
            pressures[v] = 0.0

        # Try multiple strategies to repack victims
        best_local_config = None
        best_local_max = float('inf')

        # Repack Strategies with randomization: Density, Load, Size, Random
        for trial in range(20):
            iter_items = list(repack_items)
            random.shuffle(iter_items)

            mode = trial % 4
            if mode == 0: # Density with noise
                iter_items.sort(key=lambda x: ((x.req_rate/x.slo)/(x.model_size+1e-6)) * random.uniform(0.85, 1.15), reverse=True)
            elif mode == 1: # Load (w) with noise
                iter_items.sort(key=lambda x: (x.req_rate/x.slo) * random.uniform(0.9, 1.1), reverse=True)
            elif mode == 2: # Size (s) with noise
                iter_items.sort(key=lambda x: x.model_size * random.uniform(0.9, 1.1), reverse=True)

            l_loads = {v: 0.0 for v in victim_list}
            l_used = {v: 0.0 for v in victim_list}
            l_placement = {v: [] for v in victim_list}
            possible = True

            # Use Best Fit Decreasing logic on Pressure
            for item in iter_items:
                w, s = item.req_rate / item.slo, item.model_size
                best_v = -1
                best_score = float('inf')

                for v in victim_list:
                    # Capacity Check
                    if l_used[v] + s <= GPU_MEM_SIZE - 1e-6:
                        # Score: Resulting pressure
                        rem = GPU_MEM_SIZE - l_used[v] - s
                        # Check pressure if we add it
                        p = (l_loads[v] + w) / rem if rem > 1e-6 else (float('inf') if (l_loads[v] + w) > 0 else 0)

                        if p < best_score:
                            best_score = p
                            best_v = v

                if best_v != -1:
                    l_placement[best_v].append(item)
                    l_loads[best_v] += w
                    l_used[best_v] += s
                else:
                    possible = False
                    break

            if possible:
                # Calculate max pressure of this config
                local_max = 0.0
                for v in victim_list:
                    p = get_pressure(l_loads[v], l_used[v])
                    if p > local_max: local_max = p

                if local_max < best_local_max:
                    best_local_max = local_max
                    best_local_config = (l_placement, l_loads, l_used)

        if best_local_config:
            l_place, l_l, l_u = best_local_config
            for v in victim_list:
                current_placement[v] = l_place[v]
                loads[v] = l_l[v]
                used[v] = l_u[v]
                pressures[v] = get_pressure(loads[v], used[v])
        else:
            # Revert to global best
            current_placement = {k: list(v) for k, v in best_placement.items()}
            loads = [0.0]*gpu_num
            used = [0.0]*gpu_num
            for g in range(gpu_num):
                for m in current_placement[g]:
                    loads[g] += m.req_rate / m.slo
                    used[g] += m.model_size
            pressures = [get_pressure(loads[g], used[g]) for g in range(gpu_num)]

            if iteration > max_iters * 0.95: break

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