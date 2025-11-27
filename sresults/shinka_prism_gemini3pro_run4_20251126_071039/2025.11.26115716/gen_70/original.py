# EVOLVE-BLOCK-START
"""Model placement algorithm for minimizing maximum KV cache pressure across GPUs"""

import math
import random

GPU_MEM_SIZE = 80.0

def compute_model_placement(gpu_num, models):
    """
    Compute a model placement that minimizes the maximum KVPR across all GPUs.

    Strategy:
    1. Binary Search for the optimal target KVPR (K).
    2. Feasibility Check (Bin Packing):
       - Transforms the problem into checking if items fit in bins with capacity K*C
         and item sizes w + K*s.
       - Uses a Multi-Heuristic Greedy approach:
         - Tries multiple deterministic sort orders (Linearized Cost, Size, Weight, Density).
         - If deterministic fails, tries Randomized Greedy with perturbed sort keys.
    3. Local Search Optimization:
       - Refines any valid placement found using Steepest Descent Hill Climbing.
       - Uses both 'Move' (shift item to another GPU) and 'Swap' (exchange items) operators.

    Args:
        gpu_num: Number of GPUs
        models: List of models to place

    Returns:
        A placement of models to GPUs
    """

    # 1. Preprocess items
    items = []
    for i, m in enumerate(models):
        w = m.req_rate / m.slo
        s = m.model_size
        items.append({'m': m, 'w': w, 's': s})

    # Helper: Calculate KVPR given w and s
    def calc_kvpr(w, s):
        rem = GPU_MEM_SIZE - s
        if rem <= 1e-9:
            return float('inf') if w > 1e-9 else 0.0
        return w / rem

    def get_max_kvpr(placement):
        mx = 0.0
        for p in placement.values():
            w = sum(m.req_rate / m.slo for m in p)
            s = sum(m.model_size for m in p)
            mx = max(mx, calc_kvpr(w, s))
        return mx

    # 2. Local Search (Hill Climbing with Move & Swap)
    def local_optimize(placement):
        # Convert placement to mutable state
        state = []
        for i in range(gpu_num):
            p = placement[i]
            w = sum(m.req_rate / m.slo for m in p)
            s = sum(m.model_size for m in p)
            state.append({'w': w, 's': s, 'items': list(p)})

        # Optimization loop
        for _ in range(100):
            # Find bottleneck GPU
            max_k = -1.0
            src_idx = -1
            gpu_k = []

            for i, st in enumerate(state):
                k = calc_kvpr(st['w'], st['s'])
                gpu_k.append(k)
                if k > max_k:
                    max_k = k
                    src_idx = i

            if max_k <= 1e-9: break

            src = state[src_idx]
            improved = False

            # Operator 1: Move item from bottleneck to other
            for i, item in enumerate(src['items']):
                iw = item.req_rate / item.slo
                is_ = item.model_size

                # Check Src after removal
                ns_w = src['w'] - iw
                ns_s = src['s'] - is_
                ns_k = calc_kvpr(ns_w, ns_s)

                # Pruning: If src doesn't drop below max_k, it's not a complete fix,
                # but we proceed if it improves the global situation (steepest descent).
                # To be efficient, we check if max(ns_k, nd_k) < max_k

                for dst_idx in range(gpu_num):
                    if dst_idx == src_idx: continue
                    dst = state[dst_idx]

                    if dst['s'] + is_ > GPU_MEM_SIZE: continue

                    nd_w = dst['w'] + iw
                    nd_s = dst['s'] + is_
                    nd_k = calc_kvpr(nd_w, nd_s)

                    if max(ns_k, nd_k) < max_k - 1e-7:
                        # Apply Move
                        src['items'].pop(i)
                        src['w'], src['s'] = ns_w, ns_s
                        dst['items'].append(item)
                        dst['w'], dst['s'] = nd_w, nd_s
                        improved = True
                        break
                if improved: break

            if improved: continue

            # Operator 2: Swap item in bottleneck with item in other
            for i, item1 in enumerate(src['items']):
                iw1 = item1.req_rate / item1.slo
                is1 = item1.model_size

                for dst_idx in range(gpu_num):
                    if dst_idx == src_idx: continue
                    dst = state[dst_idx]

                    # Heuristic: Don't swap with a near-bottleneck GPU
                    if gpu_k[dst_idx] > max_k * 0.95: continue

                    for j, item2 in enumerate(dst['items']):
                        iw2 = item2.req_rate / item2.slo
                        is2 = item2.model_size

                        # New Src
                        ns_s = src['s'] - is1 + is2
                        if ns_s > GPU_MEM_SIZE: continue
                        ns_w = src['w'] - iw1 + iw2

                        # New Dst
                        nd_s = dst['s'] - is2 + is1
                        if nd_s > GPU_MEM_SIZE: continue
                        nd_w = dst['w'] - iw2 + iw1

                        ns_k = calc_kvpr(ns_w, ns_s)
                        nd_k = calc_kvpr(nd_w, nd_s)

                        if max(ns_k, nd_k) < max_k - 1e-7:
                            # Apply Swap
                            src['items'][i] = item2
                            src['w'], src['s'] = ns_w, ns_s
                            dst['items'][j] = item1
                            dst['w'], dst['s'] = nd_w, nd_s
                            improved = True
                            break
                    if improved: break
                if improved: break

            if not improved: break

        return {i: state[i]['items'] for i in range(gpu_num)}

    # 3. Feasibility Check (Multi-Heuristic Greedy)
    def check_placement(k_target):
        lin_cap = k_target * GPU_MEM_SIZE

        def try_pack(ordered_items):
            bins = [{'w': 0.0, 's': 0.0, 'items': []} for _ in range(gpu_num)]

            for item in ordered_items:
                w, s = item['w'], item['s']
                v_lin = w + k_target * s

                best_idx = -1
                best_fill = -1.0

                for i in range(gpu_num):
                    b = bins[i]
                    # Physical Check
                    if b['s'] + s > GPU_MEM_SIZE: continue

                    # KVPR Check: (current_w + w) + K*(current_s + s) <= K*C
                    # <=> current_lin + v_lin <= lin_cap
                    lin_load = b['w'] + k_target * b['s']
                    if lin_load + v_lin > lin_cap + 1e-7: continue

                    # Best Fit: Maximize current linearized load
                    if lin_load > best_fill:
                        best_fill = lin_load
                        best_idx = i

                if best_idx != -1:
                    bins[best_idx]['w'] += w
                    bins[best_idx]['s'] += s
                    bins[best_idx]['items'].append(item['m'])
                else:
                    return None
            return {i: bins[i]['items'] for i in range(gpu_num)}

        # A. Deterministic Sort Strategies
        keys = [
            lambda x: x['w'] + k_target * x['s'],   # Linearized Cost
            lambda x: x['s'],                       # Physical Size
            lambda x: x['w'],                       # Weight
            lambda x: x['w'] / (x['s'] + 1e-9)      # Density
        ]

        for key in keys:
            res = try_pack(sorted(items, key=key, reverse=True))
            if res: return res

        # B. Randomized Strategy
        # Perturb the Linearized Cost key
        rng = random.Random(42 + int(k_target * 100))
        base_key = lambda x: x['w'] + k_target * x['s']

        for _ in range(50):
            # Add noise to the key: key * uniform(0.9, 1.1)
            noisy_items = []
            for item in items:
                score = base_key(item) * rng.uniform(0.9, 1.1)
                noisy_items.append((score, item))

            noisy_items.sort(key=lambda x: x[0], reverse=True)
            res = try_pack([x[1] for x in noisy_items])
            if res: return res

        return None

    # 4. Binary Search Driver
    high = 1e9

    # Initial valid solution
    best_placement = check_placement(high)
    if best_placement is None:
        raise ValueError("Unable to place models on GPUs (insufficient total memory).")

    # Initial Optimization
    best_placement = local_optimize(best_placement)
    high = get_max_kvpr(best_placement)
    low = 0.0

    for _ in range(30):
        mid = (low + high) / 2
        res = check_placement(mid)
        if res:
            res = local_optimize(res)
            mx = get_max_kvpr(res)
            if mx < get_max_kvpr(best_placement):
                best_placement = res
            high = min(mid, mx)
        else:
            low = mid

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
