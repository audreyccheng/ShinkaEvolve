# EVOLVE-BLOCK-START
"""Model placement algorithm for minimizing maximum KV cache pressure across GPUs"""

import random

GPU_MEM_SIZE = 80.0

def compute_model_placement(gpu_num, models):
    """
    Compute a model placement that minimizes the maximum KVPR across all GPUs.

    Uses Binary Search on target KVPR (K).
    Inner check uses Best-Fit Decreasing with Adaptive Failure Prioritization.
    Post-processes with Local Search (Hill Climbing).
    """

    # Pre-process models
    items = []
    for i, m in enumerate(models):
        items.append({
            'model': m,
            'w': m.req_rate / m.slo,
            's': m.model_size,
            'id': i
        })

    def calc_kvpr(w, s):
        """Compute KVPR: w / (C - s)."""
        rem = GPU_MEM_SIZE - s
        if rem <= 1e-9:
            return float('inf') if w > 1e-9 else 0.0
        return w / rem

    def refine_placement(placement_list):
        """
        Hill climbing local search to reduce max KVPR.
        Input: list of lists of item dicts.
        Output: list of dicts representing GPU states.
        """
        # Initialize state
        state = []
        for p in placement_list:
            w = sum(x['w'] for x in p)
            s = sum(x['s'] for x in p)
            state.append({'w': w, 's': s, 'items': list(p)})

        # Optimize
        for _ in range(500):
            # Find bottleneck
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

            improved = False
            src = state[src_idx]

            # 1. Try Move (Src -> Dst)
            # Sort items by some heuristic? Maybe largest size first?
            # Iterating normally is usually fine for small N.
            for i, item in enumerate(src['items']):
                # Predicte source state after removal
                ns_w = src['w'] - item['w']
                ns_s = src['s'] - item['s']
                # KVPR generally improves on source, so we focus on dest

                # Check all destinations
                for dst_idx in range(gpu_num):
                    if dst_idx == src_idx: continue
                    dst = state[dst_idx]

                    if dst['s'] + item['s'] > GPU_MEM_SIZE: continue

                    nd_w = dst['w'] + item['w']
                    nd_s = dst['s'] + item['s']
                    nd_k = calc_kvpr(nd_w, nd_s)

                    # Acceptance criteria: Dest must stay below current bottleneck
                    if nd_k < max_k - 1e-9:
                        src['items'].pop(i)
                        src['w'], src['s'] = ns_w, ns_s
                        dst['items'].append(item)
                        dst['w'], dst['s'] = nd_w, nd_s
                        improved = True
                        break
                if improved: break

            if improved: continue

            # 2. Try Swap (Src <-> Dst)
            for i, item1 in enumerate(src['items']):
                for dst_idx in range(gpu_num):
                    if dst_idx == src_idx: continue
                    dst = state[dst_idx]

                    # Pruning: Don't swap with a GPU that is already near the limit
                    if gpu_k[dst_idx] > max_k * 0.95: continue

                    for j, item2 in enumerate(dst['items']):
                        # Check capacity
                        ns_s = src['s'] - item1['s'] + item2['s']
                        nd_s = dst['s'] - item2['s'] + item1['s']

                        if ns_s > GPU_MEM_SIZE or nd_s > GPU_MEM_SIZE: continue

                        ns_w = src['w'] - item1['w'] + item2['w']
                        nd_w = dst['w'] - item2['w'] + item1['w']

                        ns_k = calc_kvpr(ns_w, ns_s)
                        nd_k = calc_kvpr(nd_w, nd_s)

                        # Acceptance: Both must be better than current max
                        if ns_k < max_k - 1e-9 and nd_k < max_k - 1e-9:
                            src['items'][i] = item2
                            src['w'], src['s'] = ns_w, ns_s
                            dst['items'][j] = item1
                            dst['w'], dst['s'] = nd_w, nd_s
                            improved = True
                            break
                    if improved: break
                if improved: break

            if not improved: break

        return state

    def check_placement(k_target):
        limit_cost = k_target * GPU_MEM_SIZE

        def try_pack(ordered_items):
            # Returns (placement_list, failed_index)
            # Placement is list of lists of items

            # State: load (w), size (s), items
            # Optimization: track linearized load directly
            # lin_load = w + k*s
            bins = [{'lin': 0.0, 's': 0.0, 'w': 0.0, 'items': []} for _ in range(gpu_num)]

            for idx, item in enumerate(ordered_items):
                w, s = item['w'], item['s']
                cost = w + k_target * s

                best_idx = -1
                best_fill = -1.0

                for i in range(gpu_num):
                    b = bins[i]
                    if b['s'] + s > GPU_MEM_SIZE: continue

                    # Capacity Check
                    if b['lin'] + cost > limit_cost + 1e-7: continue

                    # Best Fit: Maximize current fill (linearized)
                    if b['lin'] > best_fill:
                        best_fill = b['lin']
                        best_idx = i

                if best_idx != -1:
                    bins[best_idx]['lin'] += cost
                    bins[best_idx]['s'] += s
                    bins[best_idx]['w'] += w
                    bins[best_idx]['items'].append(item)
                else:
                    return None, idx

            return [b['items'] for b in bins], -1

        # 1. Deterministic Heuristics
        strategies = [
            lambda x: x['w'] + k_target * x['s'],   # Linearized Cost
            lambda x: x['s'],                       # Size
            lambda x: x['w'],                       # Weight
            lambda x: x['w'] / (GPU_MEM_SIZE - x['s'] + 1e-9) # Standalone Pressure
        ]

        for key in strategies:
            items_sorted = sorted(items, key=key, reverse=True)
            res, _ = try_pack(items_sorted)
            if res: return res

        # 2. Adaptive Randomized Heuristics
        rng = random.Random(42 + int(k_target * 10))
        base_items = list(items)

        failed_items = set()

        for _ in range(50):
            # Partition items into failed (priority) and others
            prio_items = []
            other_items = []

            for it in base_items:
                if it['id'] in failed_items:
                    prio_items.append(it)
                else:
                    other_items.append(it)

            # Sort priority items by Size Descending (hardest to pack first)
            prio_items.sort(key=lambda x: x['s'], reverse=True)

            # Sort others by Perturbed Linearized Cost
            other_items.sort(key=lambda x: (x['w'] + k_target * x['s']) * rng.uniform(0.85, 1.15),
                             reverse=True)

            current_items = prio_items + other_items

            res, fail_idx = try_pack(current_items)
            if res: return res

            if fail_idx != -1:
                # Add failed item to set for prioritization
                failed_id = current_items[fail_idx]['id']
                failed_items.add(failed_id)

                # Prevent set from growing too large (avoid getting stuck)
                if len(failed_items) > 6:
                    # Reset strategy if too many failures accumulate
                    failed_items = {failed_id}

        return None

    # Binary Search
    high = 1e9

    # Quick check at high
    best_res = check_placement(high)
    if best_res is None:
        raise ValueError("Unable to place models on GPUs.")

    # Convert best_res to state for processing
    best_state = refine_placement(best_res)

    # Calculate upper bound
    mx = 0.0
    for st in best_state:
        mx = max(mx, calc_kvpr(st['w'], st['s']))
    high = mx
    low = 0.0

    for _ in range(25):
        if high - low < 1e-4: break
        mid = (low + high) / 2

        res = check_placement(mid)
        if res:
            refined = refine_placement(res)
            rmx = 0.0
            for st in refined:
                rmx = max(rmx, calc_kvpr(st['w'], st['s']))

            if rmx < high:
                best_state = refined
                high = rmx
            else:
                # Valid placement found at mid, so high can be at least mid
                high = min(mid, high)
        else:
            low = mid

    # Final pass
    best_state = refine_placement([st['items'] for st in best_state])

    return {i: [x['model'] for x in st['items']] for i, st in enumerate(best_state)}
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