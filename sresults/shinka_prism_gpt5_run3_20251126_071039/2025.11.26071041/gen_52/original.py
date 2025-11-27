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

    # Helpers
    def safe_div(num, den):
        if den <= 0:
            return float('inf') if num > 0 else 0.0
        return num / den

    def kvpr(numer, rem_mem):
        return safe_div(numer, rem_mem)

    # Extract and validate model stats
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
        n = rr / slo  # r/s
        items.append((m, ms, n))
        total_mem += ms
        total_n += n

    if not items:
        return {g: [] for g in range(gpu_num)}

    total_capacity = gpu_num * GPU_MEM_SIZE
    if total_mem - total_capacity > 1e-9:
        raise ValueError("Total model memory exceeds total GPU memory")

    # Measured max KVPR of a placement
    def measured_max_kvpr(plc):
        msum = [sum(getattr(m, "model_size") for m in plc.get(g, [])) for g in range(gpu_num)]
        nsum = [sum((getattr(m, "req_rate") / getattr(m, "slo")) for m in plc.get(g, [])) for g in range(gpu_num)]
        vals = [kvpr(nsum[g], GPU_MEM_SIZE - msum[g]) for g in range(gpu_num)]
        return max(vals) if vals else 0.0

    # Memory-oriented packing strategies (fall-back to ensure feasibility)
    def memory_pack(order="size_desc", strategy="dual"):
        # order: "size_desc" or "ratio_desc"
        # strategy:
        #   - "dual": first 30% largest use max-free, rest best-fit
        #   - "bestfit": classic best-fit decreasing
        #   - "maxfree": place on GPU with most remaining memory
        #   - "firstfit": first GPU that fits
        if order == "size_desc":
            ordered = sorted(items, key=lambda it: (it[1], it[2]), reverse=True)
        else:
            # ratio_desc by n per GB
            ordered = sorted(items, key=lambda it: (safe_div(it[2], max(it[1], 1e-9)), it[2]), reverse=True)

        placement = {g: [] for g in range(gpu_num)}
        rem = [GPU_MEM_SIZE] * gpu_num
        numer = [0.0] * gpu_num

        split_idx = max(0, int(0.3 * len(ordered))) if strategy == "dual" else 0
        for idx, (mdl, ms, dn) in enumerate(ordered):
            candidates = []
            for g in range(gpu_num):
                if ms <= rem[g]:
                    # For tie-breaking, consider post local kvpr
                    new_k = kvpr(numer[g] + dn, rem[g] - ms)
                    candidates.append((g, rem[g] - ms, new_k))

            if not candidates:
                return None  # failed to place

            chosen = None
            if strategy == "bestfit" or (strategy == "dual" and idx >= split_idx):
                # Best-fit: minimize residual; tie by smaller local kvpr; then gpu id
                candidates.sort(key=lambda x: (x[1], x[2], x[0]))
                chosen = candidates[0][0]
            elif strategy == "maxfree" or (strategy == "dual" and idx < split_idx):
                # Max-free: maximize residual; tie by smaller local kvpr; then gpu id
                candidates.sort(key=lambda x: (-x[1], x[2], x[0]))
                chosen = candidates[0][0]
            else:
                # First-fit
                chosen = min(candidates, key=lambda x: x[0])[0]

            placement[chosen].append(mdl)
            rem[chosen] -= ms
            numer[chosen] += dn

        return placement

    # Regret-based insertion tailored for min-max KVPR
    def regret_insertion():
        placement = {g: [] for g in range(gpu_num)}
        rem = [GPU_MEM_SIZE] * gpu_num
        numer = [0.0] * gpu_num

        unassigned = list(items)

        # Precompute top1/top2 of current kvprs for O(1) max-except calculations
        def top12(vals):
            top = (-1, -float('inf'))
            second = (-1, -float('inf'))
            for i, v in enumerate(vals):
                if v > top[1]:
                    second = top
                    top = (i, v)
                elif v > second[1]:
                    second = (i, v)
            return top, second

        # Iteratively insert models
        while unassigned:
            current_kvprs = [kvpr(numer[g], rem[g]) for g in range(gpu_num)]
            (top_idx, top_val), (sec_idx, sec_val) = top12(current_kvprs)

            best_model = None
            best_gpu = None
            best_new_max = float('inf')
            best_regret = -float('inf')

            # Evaluate regret for each model
            for (mdl, ms, dn) in unassigned:
                feasible = []
                for g in range(gpu_num):
                    if ms <= rem[g]:
                        new_local = kvpr(numer[g] + dn, rem[g] - ms)
                        base_other = top_val if g != top_idx else sec_val
                        new_max = new_local if new_local > base_other else base_other
                        feasible.append((g, new_max, new_local))
                if not feasible:
                    continue

                feasible.sort(key=lambda x: (x[1], x[2]))  # sort by new_max then local
                best = feasible[0]
                second = feasible[1] if len(feasible) > 1 else (None, float('inf'), float('inf'))
                regret = second[1] - best[1]  # larger regret => more critical

                # Choose the model with largest regret, then smaller best new_max
                if (regret > best_regret or
                    (regret == best_regret and best[1] < best_new_max)):
                    best_regret = regret
                    best_new_max = best[1]
                    best_model = (mdl, ms, dn)
                    best_gpu = best[0]

            if best_model is None:
                # No feasible candidate in this step; fail to allow fallback
                return None

            # Commit placement
            mdl, ms, dn = best_model
            placement[best_gpu].append(mdl)
            rem[best_gpu] -= ms
            numer[best_gpu] += dn
            unassigned.remove(best_model)

        return placement

    # Local search: move and swap to reduce global max KVPR
    def improve_local(plc, max_iters=4000, eps=1e-12):
        per_g = {g: list(plc.get(g, [])) for g in range(gpu_num)}
        mem = [sum(getattr(m, "model_size") for m in per_g[g]) for g in range(gpu_num)]
        num = [sum((getattr(m, "req_rate") / getattr(m, "slo")) for m in per_g[g]) for g in range(gpu_num)]

        def kvpr_g(g, msum=None, nsum=None):
            msum = mem[g] if msum is None else msum
            nsum = num[g] if nsum is None else nsum
            return kvpr(nsum, GPU_MEM_SIZE - msum)

        def global_max_vals():
            vals = [kvpr_g(g) for g in range(gpu_num)]
            return max(vals), vals

        it = 0
        while it < max_iters:
            it += 1
            cur_max, cur_vals = global_max_vals()
            worst = max(range(gpu_num), key=lambda g: cur_vals[g])
            improved = False
            best_move = None
            best_new_max = cur_max

            worst_models = list(per_g[worst])

            # Try single-item moves out of worst GPU
            for mdl in worst_models:
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

                    src_k = kvpr(src_num, GPU_MEM_SIZE - src_mem)
                    tgt_k = kvpr(tgt_num, GPU_MEM_SIZE - tgt_mem)

                    # New max among unchanged GPUs
                    new_max = max(src_k, tgt_k)
                    for g in range(gpu_num):
                        if g != worst and g != tgt:
                            if cur_vals[g] > new_max:
                                new_max = cur_vals[g]

                    if new_max + eps < best_new_max:
                        best_new_max = new_max
                        best_move = ("move", mdl, worst, tgt, ms, dn)
                        improved = True

            if improved:
                # Apply best single move
                _, mdl, src, tgt, ms, dn = best_move
                per_g[src].remove(mdl)
                per_g[tgt].append(mdl)
                mem[src] -= ms
                num[src] -= dn
                mem[tgt] += ms
                num[tgt] += dn
                continue  # iterate again

            # Try swaps between worst GPU and others (first improving swap)
            found_swap = False
            for mdl_a in worst_models:
                ms_a = float(getattr(mdl_a, "model_size"))
                dn_a = float(getattr(mdl_a, "req_rate")) / float(getattr(mdl_a, "slo"))
                for tgt in range(gpu_num):
                    if tgt == worst:
                        continue
                    for mdl_b in list(per_g[tgt]):
                        ms_b = float(getattr(mdl_b, "model_size"))
                        dn_b = float(getattr(mdl_b, "req_rate")) / float(getattr(mdl_b, "slo"))

                        # Memory feasibility after swap
                        if mem[worst] - ms_a + ms_b > GPU_MEM_SIZE + 1e-12:
                            continue
                        if mem[tgt] - ms_b + ms_a > GPU_MEM_SIZE + 1e-12:
                            continue

                        src_mem = mem[worst] - ms_a + ms_b
                        src_num = num[worst] - dn_a + dn_b
                        tgt_mem = mem[tgt] - ms_b + ms_a
                        tgt_num = num[tgt] - dn_b + dn_a

                        src_k = kvpr(src_num, GPU_MEM_SIZE - src_mem)
                        tgt_k = kvpr(tgt_num, GPU_MEM_SIZE - tgt_mem)

                        new_max = max(src_k, tgt_k)
                        for g in range(gpu_num):
                            if g != worst and g != tgt:
                                if cur_vals[g] > new_max:
                                    new_max = cur_vals[g]

                        if new_max + eps < cur_max:
                            # Apply first improving swap
                            per_g[worst].remove(mdl_a)
                            per_g[tgt].remove(mdl_b)
                            per_g[worst].append(mdl_b)
                            per_g[tgt].append(mdl_a)
                            mem[worst] = src_mem
                            num[worst] = src_num
                            mem[tgt] = tgt_mem
                            num[tgt] = tgt_num
                            found_swap = True
                            break
                    if found_swap:
                        break
                if found_swap:
                    break

            if found_swap:
                continue

            # No improving move or swap found; stop
            break

        # Targeted 2-opt swaps between the two worst GPUs (bounded)
        if gpu_num >= 2:
            for _ in range(12):
                cur_max, vals = global_max_vals()
                worst = max(range(gpu_num), key=lambda g: vals[g])
                # identify second-worst
                second = None
                best_val = -float('inf')
                for g in range(gpu_num):
                    if g == worst:
                        continue
                    if vals[g] > best_val:
                        best_val = vals[g]
                        second = g
                if second is None:
                    break

                improved = False
                best_new_max = cur_max
                best_swap = None

                for mdl_a in list(per_g[worst]):
                    ms_a = float(getattr(mdl_a, "model_size"))
                    dn_a = float(getattr(mdl_a, "req_rate")) / float(getattr(mdl_a, "slo"))
                    for mdl_b in list(per_g[second]):
                        ms_b = float(getattr(mdl_b, "model_size"))
                        dn_b = float(getattr(mdl_b, "req_rate")) / float(getattr(mdl_b, "slo"))

                        # Memory feasibility after swap
                        if mem[worst] - ms_a + ms_b > GPU_MEM_SIZE + 1e-12:
                            continue
                        if mem[second] - ms_b + ms_a > GPU_MEM_SIZE + 1e-12:
                            continue

                        w_mem = mem[worst] - ms_a + ms_b
                        w_num = num[worst] - dn_a + dn_b
                        s_mem = mem[second] - ms_b + ms_a
                        s_num = num[second] - dn_b + dn_a

                        w_k = kvpr(w_num, GPU_MEM_SIZE - w_mem)
                        s_k = kvpr(s_num, GPU_MEM_SIZE - s_mem)

                        new_max = max(w_k, s_k)
                        # other GPUs unchanged
                        for g in range(gpu_num):
                            if g != worst and g != second:
                                if kvpr_g(g) > new_max:
                                    new_max = kvpr_g(g)

                        if new_max + eps < best_new_max:
                            best_new_max = new_max
                            best_swap = (mdl_a, mdl_b, w_mem, w_num, s_mem, s_num)
                            improved = True

                if not improved:
                    break

                mdl_a, mdl_b, w_mem, w_num, s_mem, s_num = best_swap
                per_g[worst].remove(mdl_a)
                per_g[second].remove(mdl_b)
                per_g[worst].append(mdl_b)
                per_g[second].append(mdl_a)
                mem[worst] = w_mem
                num[worst] = w_num
                mem[second] = s_mem
                num[second] = s_num

        return {g: per_g.get(g, []) for g in range(gpu_num)}

    # Build multiple initial candidates

    # Parametric T-based transformed packing to directly minimize max KVPR
    def parametric_pack_candidates():
        # Lower bounds on T
        indiv_lb = max(safe_div(n, max(GPU_MEM_SIZE - ms, 1e-9)) for _, ms, n in items)
        global_lb = safe_div(total_n, max(total_capacity - total_mem, 1e-9))

        # Pair bound for heavy pairs that cannot co-reside
        pair_lb = 0.0
        if gpu_num >= 2 and len(items) >= 2:
            L = min(len(items), 120)
            heavy = sorted(items, key=lambda it: it[1], reverse=True)[:L]
            for i in range(len(heavy)):
                _, mi, ni = heavy[i]
                for j in range(i + 1, len(heavy)):
                    _, mj, nj = heavy[j]
                    if mi + mj > GPU_MEM_SIZE + 1e-12:
                        denom = 2.0 * GPU_MEM_SIZE - (mi + mj)
                        pair_lb = max(pair_lb, safe_div(ni + nj, max(denom, 1e-9)))

        # k-prefix bound (small k)
        kprefix_lb = 0.0
        if items:
            by_m = sorted(items, key=lambda it: it[1], reverse=True)
            max_k = min(gpu_num, 6)
            for k in range(1, max_k + 1):
                s_m = 0.0
                s_n = 0.0
                for (_, ms, n) in by_m:
                    s_m += ms
                    s_n += n
                    if s_m > (k - 1) * GPU_MEM_SIZE + 1e-12:
                        break
                denom = k * GPU_MEM_SIZE - s_m
                kprefix_lb = max(kprefix_lb, safe_div(s_n, max(denom, 1e-9)))

        low_T = max(0.0, indiv_lb, global_lb, pair_lb, kprefix_lb)
        avg_mem_frac = (total_mem / total_capacity) if total_capacity > 0 else 0.0

        # Try to pack at a given T with different orderings and a choice policy
        # ordering:
        #   0 -> by transformed weight w(T) = n + T*m
        #   1 -> by intrinsic KVPR n / (80 - m)
        #   2 -> by pressure per GB n / m
        # policy:
        #   "resid"  -> best-fit on transformed residual
        #   "minmax" -> place to minimize new global max KVPR (actual KVPR metric)
        def try_pack(T, ordering=0, policy="resid", return_placement=False):
            cap = GPU_MEM_SIZE * T
            eps = 1e-12
            if ordering == 0:
                ordered = sorted(items, key=lambda it: (it[2] + T * it[1], it[2], it[1]), reverse=True)
            elif ordering == 1:
                ordered = sorted(items, key=lambda it: (safe_div(it[2], max(GPU_MEM_SIZE - it[1], 1e-9)), it[1]), reverse=True)
            else:
                ordered = sorted(items, key=lambda it: (safe_div(it[2], max(it[1], 1e-9)), it[2]), reverse=True)

            m_sum = [0.0] * gpu_num
            n_sum = [0.0] * gpu_num
            used_cap = [0.0] * gpu_num
            plc = [[] for _ in range(gpu_num)]

            for mdl, ms, n in ordered:
                w = n + T * ms
                best_choice = None

                # Precompute current actual KVPRs for "minmax" policy
                cur_kvprs = [kvpr(n_sum[g], GPU_MEM_SIZE - m_sum[g]) for g in range(gpu_num)] if policy != "resid" else None

                for g in range(gpu_num):
                    if m_sum[g] + ms > GPU_MEM_SIZE + eps:
                        continue
                    resid = cap - (used_cap[g] + w)
                    if resid < -eps:
                        continue

                    if policy == "resid":
                        key = (resid, -(GPU_MEM_SIZE - (m_sum[g] + ms)), g)
                    elif policy == "minmax":
                        new_local = kvpr(n_sum[g] + n, GPU_MEM_SIZE - (m_sum[g] + ms))
                        # New global max if placed on g
                        new_max = new_local
                        for k in range(gpu_num):
                            if k != g and cur_kvprs[k] > new_max:
                                new_max = cur_kvprs[k]
                        # prefer smaller new_max, then smaller local, then residual, then more remaining mem, then id
                        key = (new_max, new_local, resid, -(GPU_MEM_SIZE - (m_sum[g] + ms)), g)
                    else:
                        # Hybrid policy: combine projected global KVPR, local KVPR, and memory imbalance
                        new_local = kvpr(n_sum[g] + n, GPU_MEM_SIZE - (m_sum[g] + ms))
                        new_max = new_local
                        for k in range(gpu_num):
                            if k != g and cur_kvprs[k] > new_max:
                                new_max = cur_kvprs[k]
                        mem_frac = (m_sum[g] + ms) / GPU_MEM_SIZE
                        K_after_norm = max(0.0, new_max) / max(T * GPU_MEM_SIZE, 1e-12)
                        kv_new_norm = new_local / max(T, 1e-12)
                        mem_imb = abs(mem_frac - avg_mem_frac)
                        alpha = 0.15
                        beta = 0.05
                        J = K_after_norm + alpha * kv_new_norm + beta * mem_imb
                        # prefer smaller J, then smaller new_max/local, then residual, then more remaining mem, then id
                        key = (J, new_max, new_local, resid, -(GPU_MEM_SIZE - (m_sum[g] + ms)), g)

                    if best_choice is None or key < best_choice[0]:
                        best_choice = (key, g)

                if best_choice is None:
                    return (False, None) if return_placement else False

                best_g = best_choice[1]
                plc[best_g].append(mdl)
                m_sum[best_g] += ms
                n_sum[best_g] += n
                used_cap[best_g] += w

            if return_placement:
                return True, {g: plc[g] for g in range(gpu_num)}
            return True

        # For feasibility checks during search (keep minimal to stay fast)
        def try_any(T, need_plc=False):
            if need_plc:
                feas = []
                ok0, p0 = try_pack(T, 0, "resid", True)
                ok1, p1 = try_pack(T, 1, "resid", True)
                if ok0: feas.append(p0)
                if ok1: feas.append(p1)
                return (len(feas) > 0), feas
            else:
                return try_pack(T, 0, "resid", False) or try_pack(T, 1, "resid", False)

        # Exponential search for an initial feasible T
        T = max(low_T, 1e-9)
        found = False
        for _ in range(50):
            if try_any(T, need_plc=False):
                found = True
                break
            T *= 2.0
        if not found:
            return []

        # Binary search to tighten T
        low, high = low_T, T
        for _ in range(40):
            mid = (low + high) / 2.0
            if try_any(mid, need_plc=False):
                high = mid
            else:
                low = mid

        # Build placements at near-optimal T across multiple orderings and a tiny T-neighborhood
        all_plcs = []
        Ts = [high, high * 0.995, high * 1.005, high * 0.99, high * 1.01]
        combos = [
            (0, "resid"), (1, "resid"), (2, "resid"),
            (0, "minmax"), (1, "minmax"), (2, "minmax"),
            (0, "hybrid"), (1, "hybrid"), (2, "hybrid")
        ]
        for Tv in Ts:
            for ov, pol in combos:
                ok, plc = try_pack(Tv, ov, pol, True)
                if ok:
                    all_plcs.append(plc)

        return all_plcs

    candidates = []

    # Add parametric T-based candidates first
    try:
        candidates.extend(parametric_pack_candidates())
    except Exception:
        # If any numerical oddity occurs, fall back to other strategies
        pass

    # Candidate A: regret-based insertion (KVPR-aware)
    plc_regret = regret_insertion()
    if plc_regret is not None:
        candidates.append(plc_regret)

    # Candidate B: memory-balanced dual strategy
    plc_dual = memory_pack(order="size_desc", strategy="dual")
    if plc_dual is not None:
        candidates.append(plc_dual)

    # Fallbacks to ensure a feasible start
    if not candidates:
        for strat in ("bestfit", "maxfree", "firstfit"):
            plc_try = memory_pack(order="size_desc", strategy=strat if strat != "firstfit" else "firstfit")
            if plc_try is not None:
                candidates.append(plc_try)
                break

    if not candidates:
        raise ValueError("Unable to construct any feasible placement")

    # Improve each candidate locally and pick the best by measured max KVPR
    improved_candidates = []
    for plc in candidates:
        improved_candidates.append(improve_local(plc))

    best_plc = None
    best_score = float('inf')
    for plc in improved_candidates:
        score = measured_max_kvpr(plc)
        if score < best_score:
            best_score = score
            best_plc = plc

    # Ensure all GPUs are present
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