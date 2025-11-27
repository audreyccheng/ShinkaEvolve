# EVOLVE-BLOCK-START
"""Model placement algorithm for minimizing maximum KV cache pressure across GPUs"""

import random

GPU_MEM_SIZE = 80  # GB

def compute_model_placement(gpu_num, models):
    """
    Compute a model placement that minimizes the maximum KVPR across all GPUs.

    Hybrid Approach:
    1. Preprocessing & Lower Bound: Calculates theoretical min K to narrow search.
    2. Feasibility Check (solve_check):
       - Uses 'Pack with Repair': Best-Fit Decreasing with repair (swaps).
       - Heuristics:
         - Deterministic: Linear Cost (W+K*S), Size (S), Weight (W), and Density (W/S).
         - Randomized: Mix of Noisy Sorting (perturbation) and Pure Shuffling (diversity).
    3. Binary Search: Finds optimal K.
    4. Local Optimization (Hill Climbing): Refines valid placements by moving/swapping
       items from the bottleneck GPU to reduce actual KVPR.
    """

    # --- 1. Preprocessing & Lower Bound ---
    items = []
    total_w = 0.0
    total_s = 0.0
    max_single_kvpr = 0.0

    for m in models:
        w = m.req_rate / m.slo
        s = m.model_size
        items.append({'model': m, 'w': w, 's': s})
        total_w += w
        total_s += s

        rem = GPU_MEM_SIZE - s
        if rem > 1e-9:
            max_single_kvpr = max(max_single_kvpr, w / rem)
        elif w > 1e-9:
            max_single_kvpr = float('inf')

    # Theoretical global lower bound
    total_mem_capacity = gpu_num * GPU_MEM_SIZE
    rem_global = total_mem_capacity - total_s

    lb = 0.0
    if rem_global > 1e-9:
        lb = total_w / rem_global

    # LB must be at least the max required by any single item
    lb = max(lb, max_single_kvpr)

    def get_max_kvpr(placement_list):
        """Calculates the actual maximum KVPR of a placement list."""
        mx = 0.0
        for p in placement_list:
            w = sum(x['w'] for x in p)
            s = sum(x['s'] for x in p)
            rem = GPU_MEM_SIZE - s
            if rem <= 1e-9:
                return float('inf') if w > 1e-9 else 0.0
            mx = max(mx, w / rem)
        return mx

    def local_optimize(placement_list, max_iter=200):
        """
        Plateau-aware Hill Climbing.
        Minimizes Max KVPR (Primary) and Sum-Squared KVPR (Secondary) to escape plateaus.
        """
        # Convert to state structure
        state = []
        for p in placement_list:
            w = sum(x['w'] for x in p)
            s = sum(x['s'] for x in p)
            rem = GPU_MEM_SIZE - s
            val = w / rem if rem > 1e-9 else (float('inf') if w > 1e-9 else 0.0)
            state.append({'w': w, 's': s, 'items': list(p), 'kvpr': val})

        def calc_global_metrics():
            mx = 0.0
            ss = 0.0
            for b in state:
                k = b['kvpr']
                if k > mx: mx = k
                ss += k * k
            return mx, ss

        def update_bin(idx):
            b = state[idx]
            b['w'] = sum(x['w'] for x in b['items'])
            b['s'] = sum(x['s'] for x in b['items'])
            rem = GPU_MEM_SIZE - b['s']
            b['kvpr'] = b['w'] / rem if rem > 1e-9 else (float('inf') if b['w'] > 1e-9 else 0.0)

        current_max, current_ss = calc_global_metrics()

        # Optimization loop
        for _ in range(max_iter):
            # Identify bottleneck
            if current_max <= 1e-9: break

            # Sort GPUs by load to find bottlenecks
            indices = sorted(range(gpu_num), key=lambda i: state[i]['kvpr'], reverse=True)
            src_idx = indices[0]

            # If there are multiple bottlenecks, we might pick one.
            src = state[src_idx]
            improved = False

            # 1. Try Move (Src -> Dst)
            for i, item in enumerate(src['items']):
                # Predict new Src state
                ns_rem = GPU_MEM_SIZE - (src['s'] - item['s'])
                if ns_rem <= 1e-9: continue
                ns_kvpr = (src['w'] - item['w']) / ns_rem

                # Pruning: If src doesn't improve enough to matter, skip
                # Actually, in plateau search, we might accept if max is same but ss improves.
                # But if src remains the unique bottleneck at same level, max won't change.
                # So we check if ns_kvpr < current_max OR (ns_kvpr == current_max implies check ss)

                for dst_idx in range(gpu_num):
                    if dst_idx == src_idx: continue
                    dst = state[dst_idx]

                    if dst['s'] + item['s'] > GPU_MEM_SIZE: continue

                    nd_rem = GPU_MEM_SIZE - (dst['s'] + item['s'])
                    if nd_rem <= 1e-9: continue
                    nd_kvpr = (dst['w'] + item['w']) / nd_rem

                    # New local states defined. Check Global Metrics.
                    # We can compute delta of Max and SS without full re-scan if we are careful,
                    # but with small gpu_num, full scan is fast enough.

                    # Apply temporary state change
                    old_src_kvpr = src['kvpr']
                    old_dst_kvpr = dst['kvpr']

                    src['kvpr'] = ns_kvpr
                    dst['kvpr'] = nd_kvpr

                    new_max = 0.0
                    new_ss = 0.0
                    for k in range(gpu_num):
                        val = state[k]['kvpr']
                        if val > new_max: new_max = val
                        new_ss += val * val

                    # Revert state change logic (just update back using stored values)
                    src['kvpr'] = old_src_kvpr
                    dst['kvpr'] = old_dst_kvpr

                    # Acceptance Criterion
                    if new_max < current_max - 1e-9:
                        accept = True
                    elif new_max < current_max + 1e-9: # Equal
                        accept = (new_ss < current_ss - 1e-7)
                    else:
                        accept = False

                    if accept:
                        # Apply Move
                        mov = src['items'].pop(i)
                        dst['items'].append(mov)
                        update_bin(src_idx)
                        update_bin(dst_idx)
                        current_max, current_ss = new_max, new_ss
                        improved = True
                        break
                if improved: break

            if improved: continue

            # 2. Try Swap (Src <-> Dst)
            for i, item1 in enumerate(src['items']):
                for dst_idx in indices[1:]: # Try swapping with less loaded GPUs
                    dst = state[dst_idx]

                    # Pruning: Don't swap with a GPU that is already high load unless it helps SS
                    if dst['kvpr'] > current_max * 0.99: continue

                    for j, item2 in enumerate(dst['items']):
                        ns_s = src['s'] - item1['s'] + item2['s']
                        nd_s = dst['s'] - item2['s'] + item1['s']
                        if ns_s > GPU_MEM_SIZE or nd_s > GPU_MEM_SIZE: continue

                        ns_rem = GPU_MEM_SIZE - ns_s
                        nd_rem = GPU_MEM_SIZE - nd_s
                        if ns_rem <= 1e-9 or nd_rem <= 1e-9: continue

                        ns_kvpr = (src['w'] - item1['w'] + item2['w']) / ns_rem
                        nd_kvpr = (dst['w'] - item2['w'] + item1['w']) / nd_rem

                        # Evaluate Move
                        old_src_kvpr = src['kvpr']
                        old_dst_kvpr = dst['kvpr']
                        src['kvpr'] = ns_kvpr
                        dst['kvpr'] = nd_kvpr

                        new_max = 0.0
                        new_ss = 0.0
                        for k in range(gpu_num):
                            val = state[k]['kvpr']
                            if val > new_max: new_max = val
                            new_ss += val * val

                        src['kvpr'] = old_src_kvpr
                        dst['kvpr'] = old_dst_kvpr

                        if new_max < current_max - 1e-9:
                            accept = True
                        elif new_max < current_max + 1e-9:
                            accept = (new_ss < current_ss - 1e-7)
                        else:
                            accept = False

                        if accept:
                            # Apply Swap
                            src['items'][i] = item2
                            dst['items'][j] = item1
                            update_bin(src_idx)
                            update_bin(dst_idx)
                            current_max, current_ss = new_max, new_ss
                            improved = True
                            break
                    if improved: break
                if improved: break

            if not improved: break

        return [b['items'] for b in state]

    def solve_check(k_target, attempt_limit=10):
        """
        Feasibility check for K.
        Returns BEST placement list found across all strategies, or None.
        """
        limit_val = k_target * GPU_MEM_SIZE
        best_of_check = None
        min_check_kvpr = float('inf')

        def pack(ordered_items):
            bins = [{'w': 0.0, 's': 0.0, 'items': []} for _ in range(gpu_num)]

            for item in ordered_items:
                item_lin = item['w'] + k_target * item['s']
                best_idx = -1
                best_fill = -1.0

                # Best Fit
                for i in range(gpu_num):
                    b = bins[i]
                    if b['s'] + item['s'] > GPU_MEM_SIZE: continue

                    current_lin = b['w'] + k_target * b['s']
                    if current_lin + item_lin > limit_val + 1e-5: continue

                    if current_lin > best_fill:
                        best_fill = current_lin
                        best_idx = i

                if best_idx != -1:
                    bins[best_idx]['items'].append(item)
                    bins[best_idx]['w'] += item['w']
                    bins[best_idx]['s'] += item['s']
                else:
                    # Repair: Swap
                    repaired = False
                    for i in range(gpu_num):
                        b = bins[i]
                        for v_idx, victim in enumerate(b['items']):
                            # Try replace victim in b with item
                            if b['s'] - victim['s'] + item['s'] > GPU_MEM_SIZE: continue
                            new_lin_b = (b['w'] - victim['w'] + item['w']) + k_target * (b['s'] - victim['s'] + item['s'])
                            if new_lin_b > limit_val + 1e-5: continue

                            # Try place victim elsewhere
                            victim_lin = victim['w'] + k_target * victim['s']
                            for k in range(gpu_num):
                                if i == k: continue
                                bk = bins[k]
                                if bk['s'] + victim['s'] > GPU_MEM_SIZE: continue
                                if (bk['w'] + k_target * bk['s']) + victim_lin <= limit_val + 1e-5:
                                    # Swap
                                    b['items'][v_idx] = item
                                    b['w'] += (item['w'] - victim['w'])
                                    b['s'] += (item['s'] - victim['s'])

                                    bk['items'].append(victim)
                                    bk['w'] += victim['w']
                                    bk['s'] += victim['s']
                                    repaired = True
                                    break
                            if repaired: break
                        if repaired: break
                    if not repaired: return None
            return [b['items'] for b in bins]

        def update_best(res):
            nonlocal best_of_check, min_check_kvpr
            if res:
                m = get_max_kvpr(res)
                if m < min_check_kvpr:
                    min_check_kvpr = m
                    best_of_check = res

        # 1. Deterministic Strategies
        update_best(pack(sorted(items, key=lambda x: x['w'] + k_target * x['s'], reverse=True)))
        update_best(pack(sorted(items, key=lambda x: x['s'], reverse=True)))
        update_best(pack(sorted(items, key=lambda x: x['w'], reverse=True)))
        update_best(pack(sorted(items, key=lambda x: x['w']/(x['s']+1e-9), reverse=True)))

        # 2. Randomized Strategies
        if attempt_limit > 0:
            rng = random.Random(42 + int(k_target))
            base_key = lambda x: x['w'] + k_target * x['s']
            indices = list(range(len(items)))

            for i in range(attempt_limit):
                if i % 2 == 0:
                    noisy_items = sorted(items, key=lambda x: base_key(x) * rng.uniform(0.9, 1.1), reverse=True)
                    update_best(pack(noisy_items))
                else:
                    rng.shuffle(indices)
                    update_best(pack([items[j] for j in indices]))

        return best_of_check

    # --- 3. Binary Search Loop ---
    high = 1e9

    # Initial Check
    init_res = solve_check(high, 0)
    if not init_res:
        init_res = solve_check(high, 50)
        if not init_res:
            raise ValueError("Unable to place models (insufficient total memory).")

    best_pl_list = local_optimize(init_res)
    high = min(high, get_max_kvpr(best_pl_list))
    low = lb

    for _ in range(25):
        if high - low < 1e-4: break

        mid = (low + high) / 2
        # Use more attempts as problem gets harder
        res = solve_check(mid, attempt_limit=15)

        if res:
            # We found a placement with KVPR <= mid (conceptually)
            # Actually pack() just ensures constraint.
            # Local optimize it to see if we can push it even lower
            res_opt = local_optimize(res, max_iter=100)
            actual_max = get_max_kvpr(res_opt)

            if actual_max < get_max_kvpr(best_pl_list):
                 best_pl_list = res_opt

            # If the best packing found by solve_check is significantly better than mid,
            # we can lower high more aggressively.
            high = min(mid, actual_max)
        else:
            low = mid

    best_pl_list = local_optimize(best_pl_list, max_iter=500)

    return {i: [x['model'] for x in p] for i, p in enumerate(best_pl_list)}
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