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
        A placement of models to GPUs: dict {gpu_id: [model, ...]}
    """
    if gpu_num <= 0:
        raise ValueError("gpu_num must be positive")

    # ----------------- Helpers -----------------
    def safe_div(num, den):
        if den <= 0:
            return float('inf') if num > 0 else 0.0
        return num / den

    def measured_max_kvpr(placement):
        vals = []
        for g in range(gpu_num):
            used = sum(getattr(m, "model_size") for m in placement.get(g, []))
            numer = sum((getattr(m, "req_rate") / getattr(m, "slo")) for m in placement.get(g, []))
            vals.append(safe_div(numer, GPU_MEM_SIZE - used))
        return max(vals) if vals else 0.0

    # ----------------- Extract items -----------------
    items = []  # (idx, model, size, demand)
    total_mem = 0.0
    total_n = 0.0
    for idx, m in enumerate(models):
        ms = float(getattr(m, "model_size"))
        slo = float(getattr(m, "slo"))
        rr = float(getattr(m, "req_rate"))
        if ms < 0:
            raise ValueError("Model size must be non-negative")
        if ms > GPU_MEM_SIZE + 1e-9:
            raise ValueError(f"Model of size {ms} GB cannot fit into a single GPU of size {GPU_MEM_SIZE} GB")
        if slo <= 0:
            raise ValueError("Model SLO must be positive")
        n = rr / slo
        items.append((idx, m, ms, n))
        total_mem += ms
        total_n += n

    if not items:
        return {g: [] for g in range(gpu_num)}

    total_capacity_mem = gpu_num * GPU_MEM_SIZE
    if total_mem - total_capacity_mem > 1e-9:
        raise ValueError("Total model memory exceeds total GPU memory")

    # ----------------- Lower bounds on T -----------------
    indiv_lb = max(safe_div(n, GPU_MEM_SIZE - ms) for _, _, ms, n in items)
    global_lb = safe_div(total_n, max(total_capacity_mem - total_mem, 1e-9))

    # Light pair bound: for heavy pairs that cannot co-reside
    pair_lb = 0.0
    top = sorted(items, key=lambda it: it[2], reverse=True)[:min(len(items), 80)]
    for i in range(len(top)):
        mi, ni = top[i][2], top[i][3]
        for j in range(i + 1, len(top)):
            mj, nj = top[j][2], top[j][3]
            if mi + mj > GPU_MEM_SIZE + 1e-12:
                denom = 2.0 * GPU_MEM_SIZE - (mi + mj)
                pair_lb = max(pair_lb, safe_div(ni + nj, max(denom, 1e-9)))

    low_T = max(0.0, indiv_lb, global_lb, pair_lb)

    # Precompute per-item sizes and demands arrays for speed
    sizes = [it[2] for it in items]
    demands = [it[3] for it in items]

    # ----------------- Dual water-filling packing for fixed T -----------------
    def dual_pack(T, order_variant=0, iterations=7, repair_attempts=2, return_placement=False):
        cap = GPU_MEM_SIZE * T
        # Precompute weights w_i(T) = n + T*m
        weights = [demands[i] + T * sizes[i] for i in range(len(items))]

        # Build ordering
        if order_variant == 0:
            # By transformed weight desc, then demand, then size
            order = sorted(range(len(items)),
                           key=lambda i: (weights[i], demands[i], sizes[i]),
                           reverse=True)
        elif order_variant == 1:
            # By intrinsic-alone KVPR if alone: n/(S - m)
            order = sorted(range(len(items)),
                           key=lambda i: (safe_div(demands[i], max(GPU_MEM_SIZE - sizes[i], 1e-9)), sizes[i]),
                           reverse=True)
        else:
            # By size desc, then demand
            order = sorted(range(len(items)),
                           key=lambda i: (sizes[i], demands[i]),
                           reverse=True)

        # Dual variables (penalties) per GPU; start small to let costs differentiate
        lam = [0.0 for _ in range(gpu_num)]
        # Step size for subgradient updates (normalized)
        step = 0.85

        placement_idx = [[] for _ in range(gpu_num)]
        mem = [0.0] * gpu_num
        load = [0.0] * gpu_num  # transformed load sum w

        for it in range(iterations):
            # Reset assignment this pass
            for g in range(gpu_num):
                placement_idx[g].clear()
                mem[g] = 0.0
                load[g] = 0.0

            # Greedy assignment by current dual costs
            for i in order:
                ms = sizes[i]; wi = weights[i]
                # Collect feasible GPUs (memory)
                feas = []
                for g in range(gpu_num):
                    if mem[g] + ms <= GPU_MEM_SIZE + 1e-12:
                        # Cost combines dual penalty and tiny tie-breakers for load/memory balancing
                        tie_load = (load[g] + wi) / max(cap, 1e-9)
                        tie_mem = (mem[g] + ms) / GPU_MEM_SIZE
                        cost = lam[g] * wi + 1e-3 * tie_load + 1e-3 * tie_mem
                        feas.append((cost, g))
                if not feas:
                    return (False, None) if return_placement else False
                # Choose feasible GPU with minimum cost
                feas.sort(key=lambda x: (x[0], x[1]))
                gstar = feas[0][1]
                placement_idx[gstar].append(i)
                mem[gstar] += ms
                load[gstar] += wi

            # Subgradient update of duals
            any_over = False
            for g in range(gpu_num):
                excess = (load[g] - cap) / max(cap, 1e-9)
                if excess > 1e-12:
                    any_over = True
                lam[g] = max(0.0, lam[g] + step * excess)

            # Fast stop if all within capacity (transformed)
            if not any_over:
                break
            # Slightly decrease step over iterations
            step *= 0.9

        # Light repair: move items off overloaded GPUs to those with slack
        def try_repair():
            repaired = True
            attempts = 0
            while attempts < repair_attempts and repaired:
                repaired = False
                attempts += 1
                # Build list of overloaded GPUs
                over = [g for g in range(gpu_num) if load[g] > cap + 1e-9]
                if not over:
                    break
                # Precompute slack GPUs
                slack = [g for g in range(gpu_num) if load[g] < cap - 1e-12]
                if not slack:
                    break
                # Try moving smallest-weight items first from most overloaded
                over.sort(key=lambda g: load[g] - cap, reverse=True)
                for src in over:
                    if load[src] <= cap + 1e-9:
                        continue
                    # Sort items on src by increasing weight to move light ones first
                    src_items = sorted(placement_idx[src], key=lambda i: weights[i])
                    moved_any = False
                    for i in src_items:
                        ms = sizes[i]; wi = weights[i]
                        # Try slack GPUs; choose one with minimal resulting load
                        best = None
                        for tgt in slack:
                            if tgt == src:
                                continue
                            if mem[tgt] + ms > GPU_MEM_SIZE + 1e-12:
                                continue
                            if load[tgt] + wi > cap + 1e-12:
                                continue
                            key = (load[tgt] + wi, mem[tgt] + ms, tgt)
                            if best is None or key < best[0]:
                                best = (key, tgt)
                        if best is not None:
                            tgt = best[1]
                            # Apply move
                            placement_idx[src].remove(i)
                            placement_idx[tgt].append(i)
                            mem[src] -= ms; mem[tgt] += ms
                            load[src] -= wi; load[tgt] += wi
                            moved_any = True
                            if load[src] <= cap + 1e-12:
                                break
                    if moved_any:
                        repaired = True
            # Check final feasibility
            return all(load[g] <= cap + 1e-9 and mem[g] <= GPU_MEM_SIZE + 1e-9 for g in range(gpu_num))

        ok = all(load[g] <= cap + 1e-9 for g in range(gpu_num))
        if not ok:
            if not try_repair():
                return (False, None) if return_placement else False

        if return_placement:
            plc = {g: [items[i][1] for i in placement_idx[g]] for g in range(gpu_num)}
            # Ensure all GPUs present
            for g in range(gpu_num):
                plc.setdefault(g, [])
            return True, plc
        return True

    def try_any_dual(T, need_placement=False):
        feasibles = []
        for ov in (0, 1, 2):
            if need_placement:
                ok, plc = dual_pack(T, order_variant=ov, iterations=7, repair_attempts=2, return_placement=True)
                if ok:
                    feasibles.append(plc)
            else:
                if dual_pack(T, order_variant=ov, iterations=6, repair_attempts=2, return_placement=False):
                    return True
        if need_placement:
            return (len(feasibles) > 0), feasibles
        return False

    # ----------------- Search for minimal feasible T -----------------
    T = max(low_T, 1e-9)
    found = False
    for _ in range(50):
        if try_any_dual(T, need_placement=False):
            found = True
            break
        T *= 2.0
    if not found:
        raise ValueError("Unable to find a feasible packing for any KVPR threshold")

    low, high = low_T, T
    for _ in range(36):
        mid = (low + high) / 2.0
        if try_any_dual(mid, need_placement=False):
            high = mid
        else:
            low = mid

    # ----------------- Build candidates near optimal T -----------------
    candidates = []
    Ts = [high, high * 0.995, high * 1.005]
    for Tv in Ts:
        ok, plcs = try_any_dual(Tv, need_placement=True)
        if ok:
            candidates.extend(plcs)

    # Fallback: a simple greedy min-max candidate (memory-feasible)
    def greedy_minmax():
        # Order by pressure per GB: (n/m) then n
        order = sorted(items, key=lambda it: (safe_div(it[3], max(it[2], 1e-9)), it[3]), reverse=True)
        plc = {g: [] for g in range(gpu_num)}
        rem = [GPU_MEM_SIZE] * gpu_num
        numer = [0.0] * gpu_num

        def kv(n, rm): return safe_div(n, rm)

        for _, mdl, ms, n in order:
            best = None
            cur = [kv(numer[g], rem[g]) for g in range(gpu_num)]
            for g in range(gpu_num):
                if ms <= rem[g]:
                    new_local = kv(numer[g] + n, rem[g] - ms)
                    new_max = max(new_local, max(cur[k] for k in range(gpu_num) if k != g))
                    key = (new_max, new_local, -(rem[g] - ms), g)
                    if best is None or key < best[0]:
                        best = (key, g)
            if best is None:
                return None
            g = best[1]
            plc[g].append(mdl)
            numer[g] += n
            rem[g] -= ms
        return plc

    if not candidates:
        gu = greedy_minmax()
        if gu is not None:
            candidates.append(gu)

    if not candidates:
        # As a very last resort, produce any memory-feasible placement by best-fit decreasing
        ordered = sorted(items, key=lambda it: (it[2], it[3]), reverse=True)
        plc = {g: [] for g in range(gpu_num)}
        rem = [GPU_MEM_SIZE] * gpu_num
        for _, mdl, ms, _ in ordered:
            best_g = None
            best_res = float('inf')
            for g in range(gpu_num):
                if ms <= rem[g] and (rem[g] - ms) < best_res:
                    best_res = rem[g] - ms
                    best_g = g
            if best_g is None:
                raise ValueError("Unable to construct any feasible placement")
            plc[best_g].append(mdl)
            rem[best_g] -= ms
        candidates.append(plc)

    # ----------------- Pick best candidate and lightly refine -----------------
    best_plc = None
    best_score = float('inf')
    for plc in candidates:
        sc = measured_max_kvpr(plc)
        if sc < best_score:
            best_score = sc
            best_plc = plc

    # Lightweight moves from the worst GPU to reduce max KVPR
    def local_refine(plc, move_budget=60, eps=1e-12):
        per_g = {g: list(plc.get(g, [])) for g in range(gpu_num)}
        mem = [sum(getattr(m, "model_size") for m in per_g[g]) for g in range(gpu_num)]
        num = [sum((getattr(m, "req_rate") / getattr(m, "slo")) for m in per_g[g]) for g in range(gpu_num)]

        def kv_g(g, msum=None, nsum=None):
            mm = mem[g] if msum is None else msum
            nn = num[g] if nsum is None else nsum
            return safe_div(nn, GPU_MEM_SIZE - mm)

        def global_vals():
            vals = [kv_g(g) for g in range(gpu_num)]
            return max(vals), vals

        moves = 0
        while moves < move_budget:
            cur_max, vals = global_vals()
            worst = max(range(gpu_num), key=lambda g: vals[g])
            improved = False
            best_new_max = cur_max
            best_move = None

            # Try moving each model from worst to some target
            for mdl in list(per_g[worst]):
                ms = float(getattr(mdl, "model_size"))
                dn = float(getattr(mdl, "req_rate")) / float(getattr(mdl, "slo"))
                for tgt in range(gpu_num):
                    if tgt == worst:
                        continue
                    if mem[tgt] + ms > GPU_MEM_SIZE + 1e-12:
                        continue
                    src_mem = mem[worst] - ms
                    src_num = num[worst] - dn
                    tgt_mem = mem[tgt] + ms
                    tgt_num = num[tgt] + dn
                    src_k = kv_g(worst, src_mem, src_num)
                    tgt_k = kv_g(tgt, tgt_mem, tgt_num)
                    new_max = max(src_k, tgt_k)
                    for g in range(gpu_num):
                        if g != worst and g != tgt and vals[g] > new_max:
                            new_max = vals[g]
                    if new_max + eps < best_new_max:
                        best_new_max = new_max
                        best_move = (mdl, worst, tgt, ms, dn)
                        improved = True
            if not improved:
                break
            mdl, src, tgt, ms, dn = best_move
            per_g[src].remove(mdl)
            per_g[tgt].append(mdl)
            mem[src] -= ms; num[src] -= dn
            mem[tgt] += ms; num[tgt] += dn
            moves += 1

        return {g: per_g.get(g, []) for g in range(gpu_num)}

    best_plc = local_refine(best_plc, move_budget=60)

    # Ensure all GPUs present
    for g in range(gpu_num):
        best_plc.setdefault(g, [])

    return best_plc

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