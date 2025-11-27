# EVOLVE-BLOCK-START
"""Model placement algorithm for minimizing maximum KV cache pressure across GPUs"""

import random
import math

GPU_MEM_SIZE = 80  # GB

def compute_model_placement(gpu_num, models):
    """
    Compute a model placement that minimizes the maximum KVPR across all GPUs.
    Architecture:
    1. Lower Bound Calculation
    2. Binary Search with Multi-Strategy Best-Fit Decreasing
    3. Iterated Local Search (Descent + Ruin & Recreate)
    """
    rng = random.Random(42)

    # 1. Data Preparation
    m_data = []
    total_w = 0.0
    total_s = 0.0
    for i, m in enumerate(models):
        w = m.req_rate / m.slo
        s = m.model_size
        d = w / s if s > 1e-6 else 0.0
        m_data.append({'w': w, 's': s, 'd': d, 'obj': m, 'id': i})
        total_w += w
        total_s += s

    # 2. Bound Estimation
    # Ideal average pressure
    rem_global = (gpu_num * GPU_MEM_SIZE) - total_s
    if rem_global <= 1e-6:
        low_bound = 0.0
    else:
        low_bound = total_w / rem_global

    # Upper bound heuristic
    def get_greedy_bound(key_func):
        sorted_items = sorted(m_data, key=key_func, reverse=True)
        # Simple Best Fit logic trying to minimize pressure increase
        l = [0.0]*gpu_num
        u = [0.0]*gpu_num

        for item in sorted_items:
            best_g = -1
            best_score = float('inf')

            for g in range(gpu_num):
                rem = GPU_MEM_SIZE - u[g] - item['s']
                if rem > 1e-5:
                    # Score: resulting pressure
                    p = (l[g] + item['w']) / rem
                    if p < best_score:
                        best_score = p
                        best_g = g

            if best_g == -1: return float('inf')
            l[best_g] += item['w']
            u[best_g] += item['s']

        return max((l[g]/(GPU_MEM_SIZE - u[g])) if (GPU_MEM_SIZE - u[g]) > 1e-6 else 0 for g in range(gpu_num))

    ub_d = get_greedy_bound(lambda x: x['d'])
    ub_s = get_greedy_bound(lambda x: x['s'])
    ub_w = get_greedy_bound(lambda x: x['w'])

    candidates = [b for b in [ub_d, ub_s, ub_w] if b != float('inf')]
    high_bound = min(candidates) if candidates else 2000.0

    # 3. Binary Search
    final_placement_lists = None

    # Check function
    def can_pack(target_k):
        # We need to pack all items such that (L+w)/(C-U-s) <= K
        # Strategies: Density, Effective Size (w + K*s), Size, Load
        strategies = [
            lambda x: x['w'] + target_k * x['s'],
            lambda x: x['d'],
            lambda x: x['s'],
            lambda x: x['w']
        ]

        # Deterministic passes
        for key in strategies:
            items_s = sorted(m_data, key=key, reverse=True)
            bins_l = [0.0]*gpu_num
            bins_u = [0.0]*gpu_num
            bins_items = [[] for _ in range(gpu_num)]
            possible = True

            for item in items_s:
                best_g = -1
                min_slack = float('inf')

                # Check all bins
                for g in range(gpu_num):
                    # Physical constraint
                    if bins_u[g] + item['s'] >= GPU_MEM_SIZE - 1e-6: continue

                    # Pressure constraint
                    rem = GPU_MEM_SIZE - bins_u[g] - item['s']
                    max_load = target_k * rem
                    new_load = bins_l[g] + item['w']

                    if new_load <= max_load + 1e-7:
                        # Best Fit: minimize slack (unused capacity relative to K)
                        slack = max_load - new_load
                        if slack < min_slack:
                            min_slack = slack
                            best_g = g

                if best_g != -1:
                    bins_l[best_g] += item['w']
                    bins_u[best_g] += item['s']
                    bins_items[best_g].append(item['obj'])
                else:
                    possible = False
                    break

            if possible:
                return bins_items

        # Randomized passes to reduce false negatives
        # This helps finding a feasible packing for tighter K bounds
        indices = list(range(len(m_data)))
        for _ in range(20):
            rng.shuffle(indices)
            bins_l = [0.0]*gpu_num
            bins_u = [0.0]*gpu_num
            bins_items = [[] for _ in range(gpu_num)]
            possible = True

            for idx in indices:
                item = m_data[idx]
                best_g = -1
                min_slack = float('inf')

                # Pre-extract values
                i_w, i_s = item['w'], item['s']

                for g in range(gpu_num):
                    if bins_u[g] + i_s >= GPU_MEM_SIZE - 1e-6: continue
                    rem = GPU_MEM_SIZE - bins_u[g] - i_s
                    max_load = target_k * rem
                    new_load = bins_l[g] + i_w

                    if new_load <= max_load + 1e-7:
                        # Best Fit: minimize slack (unused capacity relative to K)
                        slack = max_load - new_load
                        if slack < min_slack:
                            min_slack = slack
                            best_g = g

                if best_g != -1:
                    bins_l[best_g] += i_w
                    bins_u[best_g] += i_s
                    bins_items[best_g].append(item['obj'])
                else:
                    possible = False
                    break
            if possible:
                return bins_items

        return None

    # BS Loop
    best_bs_sol = None

    # Try packing with high bound first to ensure feasibility
    res = can_pack(high_bound)
    if res:
        best_bs_sol = res
    else:
        high_bound = 5000.0 # Fallback

    for _ in range(20):
        if high_bound - low_bound < 1e-4: break
        mid = (low_bound + high_bound) / 2.0
        res = can_pack(mid)
        if res:
            best_bs_sol = res
            high_bound = mid
        else:
            low_bound = mid

    if best_bs_sol is None:
        # Fallback naive distribution if optimization fails completely
        best_bs_sol = [[] for _ in range(gpu_num)]
        g = 0
        u = [0.0]*gpu_num
        for m in models:
            if u[g] + m.model_size < GPU_MEM_SIZE:
                best_bs_sol[g].append(m)
                u[g] += m.model_size
            else:
                g = (g + 1) % gpu_num
                best_bs_sol[g].append(m)
                u[g] += m.model_size

    # Convert to Dict
    current_placement = {i: list(l) for i, l in enumerate(best_bs_sol)}

    # 4. Iterated Local Search

    # State tracking
    loads = [0.0]*gpu_num
    used = [0.0]*gpu_num
    for g in range(gpu_num):
        for m in current_placement[g]:
            loads[g] += m.req_rate / m.slo
            used[g] += m.model_size

    def get_p(l, u):
        r = GPU_MEM_SIZE - u
        return l/r if r > 1e-6 else (float('inf') if l > 0 else 0.0)

    pressures = [get_p(loads[g], used[g]) for g in range(gpu_num)]
    best_max_p = max(pressures)
    best_placement = {k: list(v) for k, v in current_placement.items()}

    max_iters = 200

    for it in range(max_iters):
        # Update Global
        curr_max = max(pressures)
        curr_sq = sum(p*p for p in pressures)

        if curr_max < best_max_p - 1e-7:
            best_max_p = curr_max
            best_placement = {k: list(v) for k, v in current_placement.items()}

        # Bottleneck Identification
        bottleneck = -1
        max_val = -1.0
        for g in range(gpu_num):
            if pressures[g] > max_val:
                max_val = pressures[g]
                bottleneck = g

        if bottleneck == -1: break

        # --- Descent Strategy ---
        bn_items = current_placement[bottleneck]

        # Sort items by density: move high intensity items first
        bn_indices = sorted(range(len(bn_items)), key=lambda i: (bn_items[i].req_rate/bn_items[i].slo)/(bn_items[i].model_size+1e-6), reverse=True)

        # Sort partners by pressure: move to least loaded first
        partners = sorted([g for g in range(gpu_num) if g != bottleneck], key=lambda g: pressures[g])

        # Fast "Max Others" excluding bottleneck and partner
        sorted_p = sorted([(pressures[g], g) for g in range(gpu_num)], reverse=True)

        best_move = None
        # (type, pt, i_bn, i_pt, nbl, nbu, npl, npu, nm, nsq)

        for partner in partners:
            # Determine max_others
            max_others = 0.0
            for p_val, p_g in sorted_p:
                if p_g != bottleneck and p_g != partner:
                    max_others = p_val
                    break

            sq_base = curr_sq - pressures[bottleneck]**2 - pressures[partner]**2

            # 1. Move
            for i in bn_indices:
                m = bn_items[i]
                w, s = m.req_rate/m.slo, m.model_size

                if used[partner] + s >= GPU_MEM_SIZE - 1e-6: continue

                n_bl = loads[bottleneck] - w
                n_bu = used[bottleneck] - s
                n_pl = loads[partner] + w
                n_pu = used[partner] + s

                pb = get_p(n_bl, n_bu)
                pp = get_p(n_pl, n_pu)

                nm = max(max_others, pb, pp)

                if nm > curr_max + 1e-9: continue
                nsq = sq_base + pb**2 + pp**2

                is_better = False
                if nm < curr_max - 1e-9: is_better = True
                elif nm < curr_max + 1e-9 and nsq < curr_sq - 1e-9: is_better = True

                if is_better:
                    if best_move is None or nm < best_move[8] - 1e-9 or (abs(nm - best_move[8]) < 1e-9 and nsq < best_move[9]):
                        best_move = ('move', partner, i, -1, n_bl, n_bu, n_pl, n_pu, nm, nsq)

            # 2. Swap
            pt_items = current_placement[partner]
            for i in bn_indices:
                m1 = bn_items[i]
                w1, s1 = m1.req_rate/m1.slo, m1.model_size
                for j, m2 in enumerate(pt_items):
                    w2, s2 = m2.req_rate/m2.slo, m2.model_size

                    n_bu = used[bottleneck] - s1 + s2
                    if n_bu >= GPU_MEM_SIZE - 1e-6: continue
                    n_pu = used[partner] - s2 + s1
                    if n_pu >= GPU_MEM_SIZE - 1e-6: continue

                    n_bl = loads[bottleneck] - w1 + w2
                    n_pl = loads[partner] - w2 + w1

                    pb = get_p(n_bl, n_bu)
                    pp = get_p(n_pl, n_pu)

                    nm = max(max_others, pb, pp)
                    if nm > curr_max + 1e-9: continue
                    nsq = sq_base + pb**2 + pp**2

                    is_better = False
                    if nm < curr_max - 1e-9: is_better = True
                    elif nm < curr_max + 1e-9 and nsq < curr_sq - 1e-9: is_better = True

                    if is_better:
                        if best_move is None or nm < best_move[8] - 1e-9 or (abs(nm - best_move[8]) < 1e-9 and nsq < best_move[9]):
                            best_move = ('swap', partner, i, j, n_bl, n_bu, n_pl, n_pu, nm, nsq)

        # Apply Best Move
        if best_move:
            mtype, pt, i, j, nbl, nbu, npl, npu, _, _ = best_move
            if mtype == 'move':
                item = current_placement[bottleneck].pop(i)
                current_placement[pt].append(item)
            else:
                item1 = current_placement[bottleneck][i]
                item2 = current_placement[pt][j]
                current_placement[bottleneck][i] = item2
                current_placement[pt][j] = item1

            loads[bottleneck], used[bottleneck] = nbl, nbu
            loads[pt], used[pt] = npl, npu
            pressures[bottleneck] = get_p(nbl, nbu)
            pressures[pt] = get_p(npl, npu)
            continue

        # --- Ruin & Recreate (Perturbation) ---
        # Victims: Bottleneck + Least Loaded + Random
        victims = {bottleneck}

        # Find least loaded
        min_p = float('inf')
        min_g = -1
        for g in range(gpu_num):
            if g != bottleneck and pressures[g] < min_p:
                min_p = pressures[g]
                min_g = g
        if min_g != -1: victims.add(min_g)

        # Add random to ensure coverage
        cands = [g for g in range(gpu_num) if g not in victims]
        if cands: victims.update(rng.sample(cands, min(len(cands), 1)))

        victim_list = list(victims)

        # Extract items
        repack_items = []
        for v in victim_list:
            repack_items.extend(current_placement[v])
            current_placement[v] = []
            loads[v] = 0.0
            used[v] = 0.0
            pressures[v] = 0.0

        # Recreate Loop (Try a few randomized packings)
        best_local = None
        best_local_max = float('inf')

        # Strategies: Effective Size (with current K), Density, Size, Load (with noise)
        strategies = [
            lambda x: ((x.req_rate/x.slo) + best_max_p * x.model_size) * rng.uniform(0.85, 1.15),
            lambda x: ((x.req_rate/x.slo)/(x.model_size+1e-6)) * rng.uniform(0.85, 1.15),
            lambda x: x.model_size * rng.uniform(0.85, 1.15),
            lambda x: (x.req_rate/x.slo) * rng.uniform(0.85, 1.15)
        ]

        for i in range(12):
            iter_items = list(repack_items)
            iter_items.sort(key=strategies[i%4], reverse=True)

            l_loads = {v: 0.0 for v in victim_list}
            l_used = {v: 0.0 for v in victim_list}
            l_alloc = {v: [] for v in victim_list}
            possible = True

            for item in iter_items:
                w, s = item.req_rate/item.slo, item.model_size
                best_v = -1
                best_sc = float('inf')

                # Best Fit (Minimize pressure)
                for v in victim_list:
                    rem = GPU_MEM_SIZE - l_used[v] - s
                    if rem > 1e-6:
                        p = (l_loads[v] + w) / rem
                        if p < best_sc:
                            best_sc = p
                            best_v = v

                # Fallback
                if best_v == -1:
                    for v in victim_list:
                        if l_used[v] + s <= GPU_MEM_SIZE - 1e-6:
                            best_v = v
                            break

                if best_v != -1:
                    l_alloc[best_v].append(item)
                    l_loads[best_v] += w
                    l_used[best_v] += s
                else:
                    possible = False
                    break

            if possible:
                lm = 0.0
                for v in victim_list:
                     p = get_p(l_loads[v], l_used[v])
                     if p > lm: lm = p

                if lm < best_local_max:
                    best_local_max = lm
                    best_local = (l_alloc, l_loads, l_used)

        # Update or Revert
        if best_local:
            l_alloc, l_loads, l_used = best_local
            for v in victim_list:
                current_placement[v] = l_alloc[v]
                loads[v] = l_loads[v]
                used[v] = l_used[v]
                pressures[v] = get_p(loads[v], used[v])
        else:
            # Revert to global best
            current_placement = {k: list(v) for k, v in best_placement.items()}
            loads = [0.0]*gpu_num
            used = [0.0]*gpu_num
            for g in range(gpu_num):
                for m in current_placement[g]:
                    loads[g] += m.req_rate / m.slo
                    used[g] += m.model_size
            pressures = [get_p(loads[g], used[g]) for g in range(gpu_num)]
            if it > max_iters * 0.9: break

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