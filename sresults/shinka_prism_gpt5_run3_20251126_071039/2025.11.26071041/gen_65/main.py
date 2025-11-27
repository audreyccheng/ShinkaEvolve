# EVOLVE-BLOCK-START
"""Model placement algorithm for minimizing maximum KV cache pressure across GPUs"""

GPU_MEM_SIZE = 80  # GB

def compute_model_placement(gpu_num, models):
    """
    Compute a model placement that minimizes the maximum KVPR across all GPUs.

    Args:
        gpu_num: Number of GPUs
        models: List of models to place

    Returns:
        A placement of models to GPUs
    """
    if gpu_num <= 0:
        raise ValueError("gpu_num must be positive")

    # ---- Helpers ----
    def safe_div(num, den):
        if den <= 0:
            return float('inf') if num > 0 else 0.0
        return num / den

    def kvpr(numer, rem_mem):
        return safe_div(numer, rem_mem)

    # ---- Extract and validate model stats ----
    items = []
    total_mem = 0.0
    total_n = 0.0
    for m in models:
        ms = float(getattr(m, "model_size"))
        slo = float(getattr(m, "slo"))
        rr = float(getattr(m, "req_rate"))
        if ms < 0:
            raise ValueError("Model size must be non-negative")
        if ms > GPU_MEM_SIZE + 1e-9:
            raise ValueError(f"Model of size {ms} GB cannot fit into a single GPU of size {GPU_MEM_SIZE} GB")
        if slo <= 0:
            raise ValueError("Model SLO must be positive")
        n = rr / slo  # pressure numerator
        items.append((m, ms, n))
        total_mem += ms
        total_n += n

    if not items:
        return {g: [] for g in range(gpu_num)}

    total_capacity = gpu_num * GPU_MEM_SIZE
    if total_mem - total_capacity > 1e-9:
        raise ValueError("Total model memory exceeds total GPU memory")

    # ---- Scoring utilities ----
    def measured_max_kvpr(plc):
        vals = []
        for g in range(gpu_num):
            used_mem = sum(getattr(m, "model_size") for m in plc.get(g, []))
            numer = sum((getattr(m, "req_rate") / getattr(m, "slo")) for m in plc.get(g, []))
            vals.append(kvpr(numer, GPU_MEM_SIZE - used_mem))
        return max(vals) if vals else 0.0

    # ---- Greedy min-max placer (used for initial solution and for completing partial states) ----
    def greedy_minmax_place(base_mem, base_num, base_lists, remaining):
        # base_mem/base_num are modified (copied by caller if needed)
        mem = list(base_mem)
        num = list(base_num)
        plc_lists = [list(lst) for lst in base_lists]

        # Order remaining by danger score: high n/(S-s), then larger s
        def danger_key(it):
            _, ms, n = it
            return (safe_div(n, max(GPU_MEM_SIZE - ms, 1e-9)), ms, n)
        ordered = sorted(remaining, key=danger_key, reverse=True)

        # Precompute current global kvprs
        def current_kvprs():
            return [kvpr(num[g], GPU_MEM_SIZE - mem[g]) for g in range(gpu_num)]

        for mdl, ms, dn in ordered:
            best = None
            cur_k = current_kvprs()
            cur_max = max(cur_k) if cur_k else 0.0
            for g in range(gpu_num):
                if mem[g] + ms <= GPU_MEM_SIZE + 1e-12:
                    new_local = kvpr(num[g] + dn, GPU_MEM_SIZE - (mem[g] + ms))
                    new_max = new_local if new_local > cur_max else cur_max
                    # tie-break by local KVPR, then more remaining mem, then gpu id
                    key = (new_max, new_local, -(GPU_MEM_SIZE - (mem[g] + ms)), g)
                    if best is None or key < best[0]:
                        best = (key, g)
            if best is None:
                return None  # infeasible completion from this partial state
            g = best[1]
            plc_lists[g].append(mdl)
            mem[g] += ms
            num[g] += dn

        return plc_lists

    # ---- Initial greedy to seed an upper bound ----
    def initial_greedy_solution():
        base_mem = [0.0] * gpu_num
        base_num = [0.0] * gpu_num
        base_lists = [[] for _ in range(gpu_num)]
        plc_lists = greedy_minmax_place(base_mem, base_num, base_lists, items)
        if plc_lists is None:
            # Fallback very simple first-fit by size (shouldn't happen due to global capacity check)
            ordered = sorted(items, key=lambda it: it[1], reverse=True)
            mem = [0.0] * gpu_num
            lists = [[] for _ in range(gpu_num)]
            for mdl, ms, _ in ordered:
                placed = False
                for g in range(gpu_num):
                    if mem[g] + ms <= GPU_MEM_SIZE + 1e-12:
                        mem[g] += ms
                        lists[g].append(mdl)
                        placed = True
                        break
                if not placed:
                    return {g: [] for g in range(gpu_num)}
            return {g: lists[g] for g in range(gpu_num)}
        return {g: plc_lists[g] for g in range(gpu_num)}

    init_plc = initial_greedy_solution()
    best_plc = init_plc
    best_score = measured_max_kvpr(init_plc)

    # ---- Bounded Branch-and-Bound over top-P critical items ----

    # Rank items by danger score: high n/(S-s) + small memory slack bias
    def danger_score(it):
        _, ms, n = it
        return safe_div(n, max(GPU_MEM_SIZE - ms, 1e-9)) + 0.02 * (ms / GPU_MEM_SIZE)

    items_sorted = sorted(items, key=danger_score, reverse=True)
    # Choose P: small to keep search fast
    P = min(len(items_sorted), max(8, min(12, 2 * gpu_num)))

    topP = items_sorted[:P]
    rest_items = items_sorted[P:]

    # For pruning: precompute suffix max of indiv lower bound over remaining topP
    indiv_lb_suffix = [0.0] * (P + 1)
    cur = 0.0
    for i in range(P - 1, -1, -1):
        _, ms, n = topP[i]
        cur = max(cur, safe_div(n, max(GPU_MEM_SIZE - ms, 1e-9)))
        indiv_lb_suffix[i] = cur
    indiv_lb_suffix[P] = 0.0

    # Also a global indiv bound across all items (helps prune early)
    global_indiv_lb = max(safe_div(n, max(GPU_MEM_SIZE - ms, 1e-9)) for _, ms, n in items_sorted) if items_sorted else 0.0

    # Hall-type memory feasibility check on largest remaining sizes against residual capacities
    def hall_feasible(res_caps, rem_sizes, take_k=None):
        if not rem_sizes:
            return True
        cap_sorted = sorted(res_caps, reverse=True)
        sizes_sorted = sorted(rem_sizes, reverse=True)
        limit = min(len(cap_sorted), len(sizes_sorted))
        if take_k is not None:
            limit = min(limit, take_k)
        sum_caps = 0.0
        sum_sizes = 0.0
        for k in range(limit):
            sum_caps += cap_sorted[k]
            sum_sizes += sizes_sorted[k]
            if sum_sizes - sum_caps > 1e-9:
                return False
        return True

    # Prepare static arrays for DFS
    base_mem = [0.0] * gpu_num
    base_num = [0.0] * gpu_num
    base_lists = [[] for _ in range(gpu_num)]

    # DFS state
    best_local_score = best_score
    best_local_plc_lists = [list(init_plc.get(g, [])) for g in range(gpu_num)]

    # Node exploration cap to bound runtime
    node_budget = 60000
    nodes_visited = 0

    # Pre-collect sizes from rest for occasional Hall pruning (use the largest few)
    rest_sizes_sorted = sorted([ms for (_, ms, _) in rest_items], reverse=True)

    # DFS recursion
    def dfs(idx, mem, num, assigned_idxs):
        nonlocal best_local_score, best_local_plc_lists, nodes_visited
        if nodes_visited >= node_budget:
            return
        nodes_visited += 1

        # Current max KVPR
        cur_kvprs = [kvpr(num[g], GPU_MEM_SIZE - mem[g]) for g in range(gpu_num)]
        cur_max = max(cur_kvprs) if cur_kvprs else 0.0

        # Lower bounds to prune
        lb1 = max(cur_max, indiv_lb_suffix[idx], global_indiv_lb)
        if lb1 >= best_local_score - 1e-12:
            return

        # Residual capacities for Hall check
        res_caps = [GPU_MEM_SIZE - mem[g] for g in range(gpu_num)]
        # Remaining topP sizes from idx
        rem_top_sizes = [topP[j][1] for j in range(idx, P)]
        # Combine with head of rest largest (bounded) to tighten a bit
        q = min(len(rest_sizes_sorted), gpu_num * 2)
        rem_check_sizes = rem_top_sizes + rest_sizes_sorted[:q]
        if not hall_feasible(res_caps, rem_check_sizes, take_k=min(len(rem_check_sizes), gpu_num)):
            return

        if idx >= P:
            # Complete with greedy for the rest
            base_lists_now = [[] for _ in range(gpu_num)]
            for g in range(gpu_num):
                base_lists_now[g] = list(base_lists[g])  # base lists are empty here
            # Place the decided topP first
            for j in range(P):
                g = assigned_idxs[j]
                mdl, ms, dn = topP[j]
                base_lists_now[g].append(mdl)
                mem[g] += ms
                num[g] += dn
            # Greedy place the rest
            remaining = rest_items
            plc_lists = greedy_minmax_place(mem, num, base_lists_now, remaining)
            # Revert mem/num changes of topP assignments for safety
            for j in range(P - 1, -1, -1):
                g = assigned_idxs[j]
                _, ms, dn = topP[j]
                mem[g] -= ms
                num[g] -= dn
            if plc_lists is None:
                return
            # Evaluate
            plc_dict = {g: plc_lists[g] for g in range(gpu_num)}
            score = measured_max_kvpr(plc_dict)
            if score + 1e-12 < best_local_score:
                best_local_score = score
                best_local_plc_lists = plc_lists
            return

        # Choose next item
        mdl, ms, dn = topP[idx]

        # Try GPUs ordered by new local KVPR after placement
        candidates = []
        seen_sig = set()
        for g in range(gpu_num):
            if mem[g] + ms <= GPU_MEM_SIZE + 1e-12:
                # symmetry pruning for identical GPU states
                sig = (round(mem[g], 9), round(num[g], 9))
                if sig in seen_sig:
                    continue
                seen_sig.add(sig)
                new_local = kvpr(num[g] + dn, GPU_MEM_SIZE - (mem[g] + ms))
                candidates.append((new_local, g))
        candidates.sort(key=lambda x: (x[0], x[1]))

        # Explore
        for _, g in candidates:
            # Apply
            mem[g] += ms
            num[g] += dn
            assigned_idxs[idx] = g

            dfs(idx + 1, mem, num, assigned_idxs)

            # Revert
            assigned_idxs[idx] = -1
            mem[g] -= ms
            num[g] -= dn

    # Run DFS
    assigned_idxs = [-1] * P
    dfs(0, base_mem[:], base_num[:], assigned_idxs)

    # Decide best
    if best_local_plc_lists:
        best_plc = {g: best_local_plc_lists[g] for g in range(gpu_num)}
        # Ensure all GPUs present
        for g in range(gpu_num):
            best_plc.setdefault(g, [])
        return best_plc

    # Fallback (should not happen): initial greedy solution
    for g in range(gpu_num):
        init_plc.setdefault(g, [])
    return init_plc

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