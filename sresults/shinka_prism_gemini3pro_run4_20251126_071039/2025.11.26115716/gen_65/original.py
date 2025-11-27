# EVOLVE-BLOCK-START
"""Model placement algorithm for minimizing maximum KV cache pressure across GPUs"""

GPU_MEM_SIZE = 80  # GB

def compute_model_placement(gpu_num, models):
    """
    Compute a model placement that minimizes the maximum KVPR across all GPUs.

    Algorithm:
    1. Calculate Lower Bound (LB) for K.
    2. Binary Search for optimal K.
       - Feasibility Check: Beam Search packing.
         - Transforms capacity constraints to linearized form: w + K*s <= K*C.
         - Sorts items by linearized cost.
         - Explores packing states using Beam Search with adaptive width.
         - Heuristic: Best Fit (Sum of Squares of linearized loads).
    3. Steepest Descent Local Search.
       - Iteratively applies the single best move or swap that reduces the
         maximum KVPR across all GPUs until convergence.

    Args:
        gpu_num: Number of GPUs
        models: List of models to place

    Returns:
        A placement of models to GPUs
    """

    # 1. Preprocessing and Lower Bound
    items = []
    total_w = 0.0
    total_s = 0.0

    for m in models:
        w = m.req_rate / m.slo
        s = m.model_size
        items.append({'model': m, 'w': w, 's': s})
        total_w += w
        total_s += s

    # Global capacity lower bound
    rem_global = gpu_num * GPU_MEM_SIZE - total_s
    lb = 0.0
    if rem_global > 1e-9:
        lb = total_w / rem_global
    elif total_w > 1e-9:
        lb = 1e9 # Impossible

    # Single item constraint lower bound
    for x in items:
        rem = GPU_MEM_SIZE - x['s']
        if rem > 1e-9:
            val = x['w'] / rem
            if val > lb: lb = val

    def get_max_kvpr(placement_list):
        mx = 0.0
        for p in placement_list:
            w = sum(x['w'] for x in p)
            s = sum(x['s'] for x in p)
            rem = GPU_MEM_SIZE - s
            if rem <= 1e-9:
                if w > 1e-9: return float('inf')
                val = 0.0
            else:
                val = w / rem
            mx = max(mx, val)
        return mx

    def solve_check(k_target, beam_width=4):
        """
        Check feasibility for K using Beam Search.
        Constraint: Sum(w + K*s) <= K*C for each bin.
        """
        limit = k_target * GPU_MEM_SIZE

        # Prepare and sort items by linearized cost
        weighted = []
        for x in items:
            cost = x['w'] + k_target * x['s']
            if cost > limit + 1e-5: return None
            weighted.append((cost, x))

        # Sort descending (Best Fit Decreasing strategy)
        weighted.sort(key=lambda x: x[0], reverse=True)

        # Beam State: (score, loads_tuple, placement_list)
        # Score: Sum of squares of loads (preference for tight packing)
        # Loads: Tuple of linearized loads

        start_loads = tuple([0.0] * gpu_num)
        start_pl = tuple([[] for _ in range(gpu_num)])

        beam = [(0.0, start_loads, start_pl)]

        for cost, item in weighted:
            candidates = []
            seen_signatures = set()

            for score, loads, pl in beam:
                # Try placing in each bin
                # Optimization: Duplicate load handling (Symmetry breaking)
                tried_loads = set()

                for i in range(gpu_num):
                    current_l = loads[i]
                    if current_l in tried_loads: continue

                    if current_l + cost <= limit + 1e-5:
                        tried_loads.add(current_l)

                        new_loads_list = list(loads)
                        new_loads_list[i] += cost

                        # Signature for state merging: sorted loads
                        sig = tuple(sorted(new_loads_list))
                        if sig in seen_signatures: continue
                        seen_signatures.add(sig)

                        new_pl_list = list(pl)
                        new_pl_list[i] = pl[i] + [item]

                        # Heuristic: Maximize sum of squares (Best Fit)
                        new_score = sum(l*l for l in new_loads_list)

                        # Use negative score for sorting logic (if min-heap/sort default)
                        # Here we just store score and sort desc later
                        candidates.append((new_score, tuple(new_loads_list), tuple(new_pl_list)))

            if not candidates:
                return None

            # Select top beam_width
            candidates.sort(key=lambda x: x[0], reverse=True)
            beam = candidates[:beam_width]

        return beam[0][2]

    def local_optimize(placement_list):
        """Steepest Descent Hill Climbing"""

        # Initialize state
        state = []
        for p in placement_list:
            w = sum(x['w'] for x in p)
            s = sum(x['s'] for x in p)
            rem = GPU_MEM_SIZE - s
            val = w / rem if rem > 1e-9 else (float('inf') if w > 1e-9 else 0.0)
            state.append({'w': w, 's': s, 'items': list(p), 'val': val})

        # Helper to calc potential kvpr
        def calc_k(w, s):
            rem = GPU_MEM_SIZE - s
            if rem <= 1e-9: return float('inf') if w > 1e-9 else 0.0
            return w / rem

        for _ in range(100): # Limit iterations
            # Identify current max and bottleneck
            max_val = -1.0
            src_idx = -1
            for i, st in enumerate(state):
                if st['val'] > max_val:
                    max_val = st['val']
                    src_idx = i

            if max_val <= 1e-9: break

            src = state[src_idx]
            best_move = None
            best_improvement = 0.0

            # 1. Try Moves (Source -> Any Dest)
            for i, item in enumerate(src['items']):
                # Predict new src
                ns_w = src['w'] - item['w']
                ns_s = src['s'] - item['s']
                ns_val = calc_k(ns_w, ns_s)

                # Optimization: if src doesn't improve enough to matter, skip
                # (But we want steepest descent, so we check all that improve max_val)

                for dst_idx in range(gpu_num):
                    if dst_idx == src_idx: continue
                    dst = state[dst_idx]

                    if dst['s'] + item['s'] > GPU_MEM_SIZE: continue

                    nd_val = calc_k(dst['w'] + item['w'], dst['s'] + item['s'])

                    # The new system max will be at least max(ns_val, nd_val)
                    # We want to reduce max_val.
                    # Improvement is roughly max_val - max(ns_val, nd_val)
                    # Note: We assume other GPUs don't exceed max_val (they are <= max_val)

                    new_local_max = max(ns_val, nd_val)
                    if new_local_max < max_val:
                        imp = max_val - new_local_max
                        if imp > best_improvement:
                            best_improvement = imp
                            best_move = ('move', i, dst_idx, -1)

            # 2. Try Swaps (Source <-> Any Dest)
            # Only perform if we haven't found a very simple move?
            # Or always check to find BEST? Steepest descent says check all.

            for i, item1 in enumerate(src['items']):
                for dst_idx in range(gpu_num):
                    if dst_idx == src_idx: continue
                    dst = state[dst_idx]

                    # Heuristic pruning
                    if dst['val'] > max_val * 0.95: continue

                    for j, item2 in enumerate(dst['items']):
                        # Sizes
                        ns_s = src['s'] - item1['s'] + item2['s']
                        nd_s = dst['s'] - item2['s'] + item1['s']
                        if ns_s > GPU_MEM_SIZE or nd_s > GPU_MEM_SIZE: continue

                        ns_val = calc_k(src['w'] - item1['w'] + item2['w'], ns_s)
                        nd_val = calc_k(dst['w'] - item2['w'] + item1['w'], nd_s)

                        new_local_max = max(ns_val, nd_val)
                        if new_local_max < max_val:
                            imp = max_val - new_local_max
                            if imp > best_improvement:
                                best_improvement = imp
                                best_move = ('swap', i, dst_idx, j)

            # Apply best move
            if best_move:
                m_type, i, dst_idx, j = best_move
                dst = state[dst_idx]

                if m_type == 'move':
                    item = src['items'].pop(i)
                    dst['items'].append(item)
                else:
                    item1 = src['items'][i]
                    item2 = dst['items'][j]
                    src['items'][i] = item2
                    dst['items'][j] = item1

                # Update stats
                for b in [src, dst]:
                    b['w'] = sum(x['w'] for x in b['items'])
                    b['s'] = sum(x['s'] for x in b['items'])
                    b['val'] = calc_k(b['w'], b['s'])
            else:
                break

        return [p['items'] for p in state]

    # Binary Search
    high = 1e9

    # Initial Check (try simple greedy then wide beam)
    res = solve_check(high, beam_width=1)
    if not res:
        res = solve_check(high, beam_width=8)
        if not res:
            raise ValueError("Unable to place models on GPUs.")

    # Optimize initial solution to tighten bound
    res = local_optimize(res)
    best_pl = res
    high = min(high, get_max_kvpr(res))
    low = lb

    for _ in range(25):
        if high - low < 1e-4: break
        mid = (low + high) / 2

        # Adaptive Beam Width: Fast check then robust check
        r = solve_check(mid, beam_width=2)
        if not r:
            r = solve_check(mid, beam_width=8)

        if r:
            # Found valid placement for K=mid.
            # Can we do better? Run local optimization on this placement
            r = local_optimize(r)
            mx = get_max_kvpr(r)

            if mx < get_max_kvpr(best_pl):
                best_pl = r

            high = min(mid, mx)
        else:
            low = mid

    # Final Optimization pass
    best_pl = local_optimize(best_pl)

    # Format for return
    return {i: [x['model'] for x in p] for i, p in enumerate(best_pl)}
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