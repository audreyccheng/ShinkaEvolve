# EVOLVE-BLOCK-START
"""Model placement algorithm for minimizing maximum KV cache pressure across GPUs"""

import random
import math

GPU_MEM_SIZE = 80  # GB

def compute_model_placement(gpu_num, models):
    """
    Compute a model placement that minimizes the maximum KVPR across all GPUs.

    Architecture:
    1. Preprocessing: Calculate weights and sizes.
    2. Initialization: Binary Search for optimal Capacity Constrained K using BFD with multiple heuristics and randomization.
    3. Refinement: Iterated Local Search (ILS) using Lexicographical Descent (minimizing the sorted pressure vector).
    """

    # --- 1. Preprocessing ---
    m_data = []
    for m in models:
        w = m.req_rate / m.slo
        s = m.model_size
        m_data.append({'w': w, 's': s, 'obj': m})

    # --- 2. Binary Search for Initial Feasible Solution ---

    def attempt_pack(k_limit, max_trials=5):
        """
        Try to pack items such that (load + w) / (cap - used - s) <= k_limit.
        Transformed: load + w + k_limit * (used + s) <= k_limit * cap
        """
        # Heuristics:
        # 1. Decreasing Virtual Size: w + k*s
        # 2. Decreasing Physical Size: s
        # 3. Decreasing Load: w
        # 4. Decreasing Density: w/s

        heuristics = [
            lambda x: x['w'] + k_limit * x['s'],
            lambda x: x['s'],
            lambda x: x['w'],
            lambda x: x['w'] / x['s'] if x['s'] > 1e-6 else 0
        ]

        # Deterministic passes
        for h in heuristics:
            sorted_items = sorted(m_data, key=h, reverse=True)
            res = _pack_sequence(sorted_items, k_limit)
            if res: return res

        # Randomized passes: perturb one of the base heuristics
        # This provides more diversity than just perturbing size
        base_items = list(m_data)
        perturb_keys = [
            lambda x: x['s'],
            lambda x: x['w'],
            lambda x: x['w'] / (x['s'] + 1e-6)
        ]

        for _ in range(max_trials):
            # Pick a heuristic to perturb
            base_key = random.choice(perturb_keys)
            # Sort with multiplicative noise
            random.shuffle(base_items) # Shuffle ties
            base_items.sort(key=lambda x: base_key(x) * random.uniform(0.85, 1.15), reverse=True)

            res = _pack_sequence(base_items, k_limit)
            if res: return res

        return None

    def _pack_sequence(items, k):
        bins = [{'l': 0.0, 'u': 0.0, 'items': []} for _ in range(gpu_num)]
        limit = k * GPU_MEM_SIZE

        for item in items:
            w, s = item['w'], item['s']
            best_g = -1
            min_slack = float('inf')

            for g_idx, b in enumerate(bins):
                # Hard Constraint
                if b['u'] + s > GPU_MEM_SIZE - 1e-6: continue

                # Soft Constraint (Virtual Capacity)
                # (L + w) + K*(U + s) <= K*C
                lhs = (b['l'] + w) + k * (b['u'] + s)

                if lhs <= limit + 1e-7:
                    slack = limit - lhs
                    if slack < min_slack:
                        min_slack = slack
                        best_g = g_idx

            if best_g == -1: return None

            bins[best_g]['items'].append(item['obj'])
            bins[best_g]['l'] += w
            bins[best_g]['u'] += s

        return {i: b['items'] for i, b in enumerate(bins)}

    # Binary Search
    # Lower bound: Total Load / Total Free Space (Aggregate)
    total_w = sum(x['w'] for x in m_data)
    total_s = sum(x['s'] for x in m_data)
    rem_space = gpu_num * GPU_MEM_SIZE - total_s
    lb = total_w / rem_space if rem_space > 1e-6 else 0.0
    ub = 1000.0 # Conservative upper bound

    best_init_placement = None

    # 20 iterations is precise enough
    for _ in range(20):
        if ub - lb < 1e-4: break
        mid = (lb + ub) / 2.0

        res = attempt_pack(mid, max_trials=10) # Increase trials for better discovery
        if res:
            best_init_placement = res
            ub = mid
        else:
            lb = mid

    if best_init_placement is None:
        # Fallback: Just sort by density and pack Best-Fit Pressure
        sorted_items = sorted(m_data, key=lambda x: x['w']/x['s'] if x['s']>0 else 0, reverse=True)
        # Re-use packer with huge K to just find feasible placement
        best_init_placement = attempt_pack(1e6, max_trials=0)
        if best_init_placement is None:
             raise ValueError("Infeasible")

    # --- 3. Refinement: Lexicographical Descent ILS ---

    current_placement = {k: list(v) for k, v in best_init_placement.items()}

    # Helper to get current pressure
    def get_p(l, u):
        rem = GPU_MEM_SIZE - u
        if rem <= 1e-6: return float('inf') if l > 1e-6 else 0.0
        return l / rem

    # Reconstruct state
    loads = [0.0] * gpu_num
    used = [0.0] * gpu_num
    for g in range(gpu_num):
        for m in current_placement[g]:
            loads[g] += m.req_rate / m.slo
            used[g] += m.model_size

    # Calculate initial pressure vector
    pressures = [get_p(loads[g], used[g]) for g in range(gpu_num)]

    # Best found tracking (Lexicographical)
    # Stored as tuple of sorted pressures (descending)
    current_score = tuple(sorted(pressures, reverse=True))
    best_score = current_score
    best_placement_copy = {k: list(v) for k, v in current_placement.items()}

    # ILS Loop
    MAX_ITERS = 200

    for iteration in range(MAX_ITERS):
        # 3a. Descent Phase (Best Improvement)
        # Finds the move that minimizes the lexicographical pressure vector

        # Identify bottleneck
        bottleneck = -1
        max_p = -1.0
        for g in range(gpu_num):
            if pressures[g] > max_p:
                max_p = pressures[g]
                bottleneck = g

        if bottleneck == -1: break

        best_move = None
        # (type, partner, idx_bn, idx_pt, nbl, nbu, npl, npu, score)

        # Sort partners by pressure ascending (target least loaded first)
        partners = sorted([g for g in range(gpu_num) if g != bottleneck], key=lambda g: pressures[g])

        # Sort items on bottleneck by load descending (try moving heaviest items first)
        bn_items = current_placement[bottleneck]
        bn_indices = sorted(range(len(bn_items)), key=lambda k: bn_items[k].req_rate/bn_items[k].slo, reverse=True)

        for partner in partners:
            # 1. Move Item: Bottleneck -> Partner
            for i in bn_indices:
                m = bn_items[i]
                w, s = m.req_rate/m.slo, m.model_size
                if used[partner] + s > GPU_MEM_SIZE - 1e-6: continue

                nbl = loads[bottleneck] - w
                nbu = used[bottleneck] - s
                npl = loads[partner] + w
                npu = used[partner] + s

                pb = get_p(nbl, nbu)
                pp = get_p(npl, npu)

                # Heuristic Pruning: If new max of pair > current global max, likely bad
                if max(pb, pp) > current_score[0] + 1e-9: continue

                # Construct new pressure vector
                new_pressures = list(pressures)
                new_pressures[bottleneck] = pb
                new_pressures[partner] = pp
                new_score = tuple(sorted(new_pressures, reverse=True))

                if new_score < current_score:
                    if best_move is None or new_score < best_move[8]:
                        best_move = ('move', partner, i, -1, nbl, nbu, npl, npu, new_score)

            # 2. Swap Items
            pt_items = current_placement[partner]
            # Try swapping bottleneck items (heaviest first) with partner items
            for i in bn_indices:
                m1 = bn_items[i]
                w1, s1 = m1.req_rate/m1.slo, m1.model_size
                for j, m2 in enumerate(pt_items):
                    w2, s2 = m2.req_rate/m2.slo, m2.model_size

                    nbu = used[bottleneck] - s1 + s2
                    npu = used[partner] - s2 + s1
                    if nbu > GPU_MEM_SIZE - 1e-6 or npu > GPU_MEM_SIZE - 1e-6: continue

                    nbl = loads[bottleneck] - w1 + w2
                    npl = loads[partner] - w2 + w1

                    pb = get_p(nbl, nbu)
                    pp = get_p(npl, npu)

                    if max(pb, pp) > current_score[0] + 1e-9: continue

                    new_pressures = list(pressures)
                    new_pressures[bottleneck] = pb
                    new_pressures[partner] = pp
                    new_score = tuple(sorted(new_pressures, reverse=True))

                    if new_score < current_score:
                        if best_move is None or new_score < best_move[8]:
                            best_move = ('swap', partner, i, j, nbl, nbu, npl, npu, new_score)

        if best_move:
            # Apply Best Move
            mtype, pt, i, j, nbl, nbu, npl, npu, n_score = best_move
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
            current_score = n_score

            if current_score < best_score:
                best_score = current_score
                best_placement_copy = {k: list(v) for k, v in current_placement.items()}

        else:
            # 3b. Perturbation Phase (Smart Ruin & Recreate)
            # Find least loaded (min pressure)
            min_p = float('inf')
            min_g = -1
            for g in range(gpu_num):
                if g != bottleneck and pressures[g] < min_p:
                    min_p = pressures[g]
                    min_g = g

            victims = {bottleneck}
            if min_g != -1: victims.add(min_g)

            # Add random victim
            rem_cands = [g for g in range(gpu_num) if g not in victims]
            if rem_cands:
                victims.add(random.choice(rem_cands))

            victim_list = list(victims)
            repack_items = []
            for v in victim_list:
                repack_items.extend(current_placement[v])
                current_placement[v] = []
                loads[v] = 0.0
                used[v] = 0.0
                pressures[v] = 0.0

            # Randomized Re-pack
            random.shuffle(repack_items)
            repack_items.sort(key=lambda x: (x.req_rate/x.slo)/(x.model_size+1e-6) * random.uniform(0.9, 1.1), reverse=True)

            possible = True
            for item in repack_items:
                w, s = item.req_rate/item.slo, item.model_size
                best_v = -1
                best_res_p = float('inf')

                for v in victim_list:
                    rem = GPU_MEM_SIZE - used[v] - s
                    if rem > 1e-6:
                        p = (loads[v] + w) / rem
                        if p < best_res_p:
                            best_res_p = p
                            best_v = v

                if best_v == -1:
                    for v in victim_list:
                        if used[v] + s <= GPU_MEM_SIZE - 1e-6:
                            best_v = v
                            break

                if best_v != -1:
                    current_placement[best_v].append(item)
                    loads[best_v] += w
                    used[best_v] += s
                else:
                    possible = False
                    break

            if possible:
                for v in victim_list:
                    pressures[v] = get_p(loads[v], used[v])
                current_score = tuple(sorted(pressures, reverse=True))
                if current_score < best_score:
                    best_score = current_score
                    best_placement_copy = {k: list(v) for k, v in current_placement.items()}
            else:
                # Revert
                current_placement = {k: list(v) for k, v in best_placement_copy.items()}
                loads = [0.0]*gpu_num
                used = [0.0]*gpu_num
                for g in range(gpu_num):
                    for m in current_placement[g]:
                        loads[g] += m.req_rate / m.slo
                        used[g] += m.model_size
                pressures = [get_p(loads[g], used[g]) for g in range(gpu_num)]
                current_score = tuple(sorted(pressures, reverse=True))

                if iteration > MAX_ITERS * 0.8: break

    return best_placement_copy
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