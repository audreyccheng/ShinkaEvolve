# EVOLVE-BLOCK-START
"""Model placement algorithm for minimizing maximum KV cache pressure across GPUs"""

import random

GPU_MEM_SIZE = 80  # GB

def compute_model_placement(gpu_num, models):
    """
    Compute a model placement that minimizes the maximum KVPR across all GPUs.
    Uses Binary Search for initialization and Variable Neighborhood Descent (VND)
    with Pairwise Ruin-and-Recreate for refinement.
    """

    # Fix random seed for reproducibility
    rng = random.Random(42)

    # Precompute model data for efficiency
    m_data = []
    for i, m in enumerate(models):
        w = m.req_rate / m.slo
        s = m.model_size
        m_data.append({
            'w': w,
            's': s,
            'obj': m,
            # Density is often the best heuristic for this specific problem
            'density': w/s if s > 0 else 0
        })

    # --- Part 1: Binary Search for Feasible K ---

    def solve_bin_packing(target_k, items):
        """
        Checks if placement is possible with max KVPR <= target_k.
        Constraint transformation:
        Load / (Cap - Used) <= K  <=>  Load + K*Used <= K*Cap
        Item effective size: w + K*s
        Bin capacity: K*Cap
        """
        # Strategies to sort items for Best Fit
        strategies = [
            lambda x: x['w'] + target_k * x['s'], # Effective cost
            lambda x: x['density'],               # Density
            lambda x: x['s'],                     # Physical size
            lambda x: x['w']                      # Load
        ]

        for key_func in strategies:
            sorted_items = sorted(items, key=key_func, reverse=True)

            bins_l = [0.0] * gpu_num
            bins_u = [0.0] * gpu_num
            bins_items = [[] for _ in range(gpu_num)]
            possible = True

            # Best Fit Decreasing
            for item in sorted_items:
                best_g = -1
                min_residual = float('inf')

                item_eff_size = item['w'] + target_k * item['s']
                max_eff_cap = target_k * GPU_MEM_SIZE

                for g in range(gpu_num):
                    # Hard memory limit (must be strictly less to avoid div by zero/inf pressure)
                    if bins_u[g] + item['s'] >= GPU_MEM_SIZE - 1e-6:
                        continue

                    # Target pressure limit check
                    current_eff_load = bins_l[g] + target_k * bins_u[g]

                    if current_eff_load + item_eff_size <= max_eff_cap + 1e-7:
                        # Minimize residual effective capacity
                        res = max_eff_cap - (current_eff_load + item_eff_size)
                        if res < min_residual:
                            min_residual = res
                            best_g = g

                if best_g != -1:
                    bins_l[best_g] += item['w']
                    bins_u[best_g] += item['s']
                    bins_items[best_g].append(item)
                else:
                    possible = False
                    break

            if possible:
                return bins_items

        return None

    # Determine bounds for Binary Search
    # Heuristic: Run a quick greedy pass to get a valid Upper Bound
    def get_initial_greedy(key_func):
        s_items = sorted(m_data, key=key_func, reverse=True)
        placements = [[] for _ in range(gpu_num)]
        bl = [0.0] * gpu_num
        bu = [0.0] * gpu_num

        for item in s_items:
            best_g = -1
            best_score = float('inf')

            for g in range(gpu_num):
                rem = GPU_MEM_SIZE - bu[g] - item['s']
                if rem > 1e-5:
                    # Minimize pressure increase
                    p = (bl[g] + item['w']) / rem
                    if p < best_score:
                        best_score = p
                        best_g = g

            if best_g == -1: return None, float('inf')
            placements[best_g].append(item)
            bl[best_g] += item['w']
            bu[best_g] += item['s']

        max_p = 0.0
        for g in range(gpu_num):
            rem = GPU_MEM_SIZE - bu[g]
            if rem <= 1e-6:
                if bl[g] > 0: max_p = float('inf')
            else:
                max_p = max(max_p, bl[g] / rem)
        return placements, max_p

    # Try getting an initial solution
    init_struct, upper_k = get_initial_greedy(lambda x: x['density'])
    if init_struct is None:
        init_struct, upper_k = get_initial_greedy(lambda x: x['s'])

    if init_struct is None:
        upper_k = 2000.0 # High fallback bound

    # Binary Search
    low = 0.0
    high = upper_k
    final_placement_struct = init_struct

    for _ in range(18): # Fixed iterations
        if high - low < 1e-4: break
        mid = (low + high) / 2.0
        res = solve_bin_packing(mid, m_data)
        if res:
            final_placement_struct = res
            high = mid
        else:
            low = mid

    if final_placement_struct is None:
        raise ValueError("Could not find feasible placement")

    # Convert struct to dictionary for refinement
    placement = {g: [x['obj'] for x in items] for g, items in enumerate(final_placement_struct)}

    # --- Part 2: Iterated Local Search (Refinement) ---

    current_placement = placement

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

    pressures = [get_pressure(loads[g], used[g]) for g in range(gpu_num)]
    best_max_p = max(pressures)

    # ILS Parameters
    max_iters = 250

    for iteration in range(max_iters):
        current_max_p = max(pressures)
        current_sum_sq = sum(p*p for p in pressures)

        if current_max_p < best_max_p - 1e-7:
            best_max_p = current_max_p
            best_placement = {k: list(v) for k, v in current_placement.items()}

        # Bottleneck identification
        bottleneck = -1
        max_val = -1.0
        for g in range(gpu_num):
            if pressures[g] > max_val:
                max_val = pressures[g]
                bottleneck = g

        if bottleneck == -1: break

        # --- Descent: Best Improvement ---
        best_move = None
        # (type, partner, idx_bn, idx_pt, nb_l, nb_u, np_l, np_u, max_p, sum_sq)

        bn_items = current_placement[bottleneck]

        # Sort pressures to get top few for efficient max checking
        sorted_p_indices = sorted(range(gpu_num), key=lambda i: pressures[i], reverse=True)

        for partner in range(gpu_num):
            if partner == bottleneck: continue

            # Efficient max_others calculation
            max_others = 0.0
            for idx in sorted_p_indices:
                if idx != bottleneck and idx != partner:
                    max_others = pressures[idx]
                    break

            sq_base = current_sum_sq - (pressures[bottleneck]**2 + pressures[partner]**2)

            # 1. Move BN -> Partner
            for i, m in enumerate(bn_items):
                w, s = m.req_rate/m.slo, m.model_size
                if used[partner] + s >= GPU_MEM_SIZE - 1e-6: continue

                n_bl = loads[bottleneck] - w
                n_bu = used[bottleneck] - s
                n_pl = loads[partner] + w
                n_pu = used[partner] + s

                pb = get_pressure(n_bl, n_bu)
                pp = get_pressure(n_pl, n_pu)

                nm = max(max_others, pb, pp)

                # Pruning
                if nm > current_max_p + 1e-9: continue

                nsq = sq_base + pb**2 + pp**2

                better = False
                if nm < current_max_p - 1e-9: better = True
                elif nm < current_max_p + 1e-9 and nsq < current_sum_sq - 1e-9: better = True

                if better:
                    if best_move is None:
                        best_move = ('move', partner, i, -1, n_bl, n_bu, n_pl, n_pu, nm, nsq)
                    else:
                        if nm < best_move[8] - 1e-9:
                            best_move = ('move', partner, i, -1, n_bl, n_bu, n_pl, n_pu, nm, nsq)
                        elif abs(nm - best_move[8]) < 1e-9 and nsq < best_move[9]:
                            best_move = ('move', partner, i, -1, n_bl, n_bu, n_pl, n_pu, nm, nsq)

            # 2. Swap BN <-> Partner
            pt_items = current_placement[partner]
            for i, m1 in enumerate(bn_items):
                w1, s1 = m1.req_rate/m1.slo, m1.model_size
                for j, m2 in enumerate(pt_items):
                    w2, s2 = m2.req_rate/m2.slo, m2.model_size

                    n_bu = used[bottleneck] - s1 + s2
                    if n_bu >= GPU_MEM_SIZE - 1e-6: continue
                    n_pu = used[partner] - s2 + s1
                    if n_pu >= GPU_MEM_SIZE - 1e-6: continue

                    n_bl = loads[bottleneck] - w1 + w2
                    n_pl = loads[partner] - w2 + w1

                    pb = get_pressure(n_bl, n_bu)
                    pp = get_pressure(n_pl, n_pu)

                    nm = max(max_others, pb, pp)
                    if nm > current_max_p + 1e-9: continue
                    nsq = sq_base + pb**2 + pp**2

                    better = False
                    if nm < current_max_p - 1e-9: better = True
                    elif nm < current_max_p + 1e-9 and nsq < current_sum_sq - 1e-9: better = True

                    if better:
                        if best_move is None:
                            best_move = ('swap', partner, i, j, n_bl, n_bu, n_pl, n_pu, nm, nsq)
                        else:
                            if nm < best_move[8] - 1e-9:
                                best_move = ('swap', partner, i, j, n_bl, n_bu, n_pl, n_pu, nm, nsq)
                            elif abs(nm - best_move[8]) < 1e-9 and nsq < best_move[9]:
                                best_move = ('swap', partner, i, j, n_bl, n_bu, n_pl, n_pu, nm, nsq)

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
            pressures[bottleneck] = get_pressure(nbl, nbu)
            pressures[pt] = get_pressure(npl, npu)
            continue

        # --- Smart Perturbation (Burst Kick) ---
        victims = {bottleneck}

        candidates = [g for g in range(gpu_num) if g not in victims]
        if candidates:
            # Pick up to 2 random partners
            k = min(len(candidates), 2)
            victims.update(rng.sample(candidates, k))

        victim_list = list(victims)

        # Extract items
        repack_items = []
        for v in victim_list:
            repack_items.extend(current_placement[v])
            current_placement[v] = []
            loads[v] = 0.0
            used[v] = 0.0
            pressures[v] = 0.0

        # Try multiple random greedy packings
        best_local_config = None
        best_local_max = float('inf')

        for _ in range(5):
            iter_items = list(repack_items)
            # Randomized Density Sort
            iter_items.sort(key=lambda x: ((x.req_rate/x.slo)/(x.model_size+1e-6)) * rng.uniform(0.8, 1.2), reverse=True)

            l_loads = {v: 0.0 for v in victim_list}
            l_used = {v: 0.0 for v in victim_list}
            l_placement = {v: [] for v in victim_list}
            possible = True

            for item in iter_items:
                w, s = item.req_rate / item.slo, item.model_size
                best_v = -1
                best_score = float('inf')

                # Best Fit minimizing pressure
                for v in victim_list:
                    rem = GPU_MEM_SIZE - l_used[v] - s
                    if rem > 1e-6:
                        p = (l_loads[v] + w) / rem
                        if p < best_score:
                            best_score = p
                            best_v = v

                if best_v == -1:
                    # Fallback
                    for v in victim_list:
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
                local_max = max(get_pressure(l_loads[v], l_used[v]) for v in victim_list)
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
            # Recompute state
            loads = [0.0]*gpu_num
            used = [0.0]*gpu_num
            for g in range(gpu_num):
                for m in current_placement[g]:
                    loads[g] += m.req_rate / m.slo
                    used[g] += m.model_size
            pressures = [get_pressure(loads[g], used[g]) for g in range(gpu_num)]

            # If failing repeatedly at perturbation, exit early
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