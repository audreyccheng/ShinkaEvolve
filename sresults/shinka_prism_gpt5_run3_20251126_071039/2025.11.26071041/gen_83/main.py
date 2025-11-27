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

        # Triplet bound (lightweight over largest-by-size items)
        triplet_lb = 0.0
        if gpu_num >= 3 and len(items) >= 3:
            Ltr = min(len(items), 60)
            top_by_mem = sorted(items, key=lambda it: it[1], reverse=True)[:Ltr]
            for i in range(Ltr):
                mi, ni = top_by_mem[i][1], top_by_mem[i][2]
                for j in range(i + 1, min(Ltr, i + 1 + 8)):
                    mj, nj = top_by_mem[j][1], top_by_mem[j][2]
                    for k in range(j + 1, min(Ltr, j + 1 + 8)):
                        mk, nk = top_by_mem[k][1], top_by_mem[k][2]
                        total_m = mi + mj + mk
                        if total_m > 2.0 * GPU_MEM_SIZE + 1e-12:
                            denom = 3.0 * GPU_MEM_SIZE - total_m
                            triplet_lb = max(triplet_lb, safe_div(ni + nj + nk, max(denom, 1e-9)))

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

        low_T = max(0.0, indiv_lb, global_lb, pair_lb, triplet_lb, kprefix_lb)
        avg_mem_frac = (total_mem / total_capacity) if total_capacity > 0 else 0.0

        # Try to pack at a given T with different orderings and a choice policy
        # ordering:
        #   0 -> by transformed weight w(T) = n + T*m
        #   1 -> by intrinsic KVPR n / (80 - m)
        #   2 -> by pressure per GB n / m
        # policy:
        #   "resid"  -> best-fit on transformed residual
        #   "minmax" -> place to minimize new global max KVPR (actual KVPR metric)
        def try_pack(T, ordering=0, policy="resid", return_placement=False, jitter=0.0):
            cap = GPU_MEM_SIZE * T
            eps = 1e-12
            import math
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

            # Dynamic retuning thresholds
            N = len(ordered)
            thr1 = max(1, int(0.4 * N))
            thr2 = max(1, int(0.75 * N))
            updated1 = False
            updated2 = False

            i = 0
            while i < N:
                mdl, ms, n = ordered[i]
                w = n + T * ms
                best_choice = None

                # Precompute current actual KVPRs for "minmax"/"hybrid" policy
                cur_kvprs = [kvpr(n_sum[g], GPU_MEM_SIZE - m_sum[g]) for g in range(gpu_num)] if policy != "resid" else None

                for g in range(gpu_num):
                    if m_sum[g] + ms > GPU_MEM_SIZE + eps:
                        continue
                    resid = cap - (used_cap[g] + w)
                    if resid < -eps:
                        continue

                    # Tiny deterministic jitter to diversify tie-breaking; varies with jitter value
                    jbias = jitter * math.sin((g + 1) * 1.61803398875 + 1000.0 * jitter)

                    if policy == "resid":
                        key = (resid, -(GPU_MEM_SIZE - (m_sum[g] + ms)), jbias)
                    elif policy == "minmax":
                        new_local = kvpr(n_sum[g] + n, GPU_MEM_SIZE - (m_sum[g] + ms))
                        new_max = new_local
                        for k in range(gpu_num):
                            if k != g and cur_kvprs[k] > new_max:
                                new_max = cur_kvprs[k]
                        key = (new_max, new_local, resid, -(GPU_MEM_SIZE - (m_sum[g] + ms)), jbias)
                    else:
                        # Hybrid policy: combine projected global KVPR, local KVPR, and memory imbalance (dynamic target)
                        new_local = kvpr(n_sum[g] + n, GPU_MEM_SIZE - (m_sum[g] + ms))
                        new_max = new_local
                        for k in range(gpu_num):
                            if k != g and cur_kvprs[k] > new_max:
                                new_max = cur_kvprs[k]
                        # Dynamic average memory fraction given this item's size
                        total_m_after = sum(m_sum) + ms
                        avg_mem_frac_dyn = (total_m_after / (gpu_num * GPU_MEM_SIZE)) if gpu_num > 0 else 0.0
                        mem_frac = (m_sum[g] + ms) / GPU_MEM_SIZE
                        mem_imb = abs(mem_frac - avg_mem_frac_dyn)
                        # Adaptive alpha based on KVPR variance across GPUs
                        Tnorm = max(T, 1e-12)
                        mean_k = (sum(cur_kvprs) / gpu_num) if gpu_num > 0 else 0.0
                        var_k = (sum((x - mean_k) ** 2 for x in cur_kvprs) / gpu_num) if gpu_num > 0 else 0.0
                        alpha = 0.25 if var_k < 0.02 * (Tnorm ** 2) else 0.15
                        beta = 0.05
                        K_after_norm = new_max / Tnorm
                        kv_new_norm = new_local / Tnorm
                        J = K_after_norm + alpha * kv_new_norm + beta * mem_imb
                        key = (J, new_max, new_local, resid, -(GPU_MEM_SIZE - (m_sum[g] + ms)), jbias)

                    if best_choice is None or key < best_choice[0]:
                        best_choice = (key, g)

                if best_choice is None:
                    return (False, None) if return_placement else False

                best_g = best_choice[1]
                plc[best_g].append(mdl)
                m_sum[best_g] += ms
                n_sum[best_g] += n
                used_cap[best_g] += w

                i += 1

                # First adaptive retune around 40% placed: recompute residual lower bound and possibly reorder remainder
                if (not updated1) and (i >= thr1) and (i < N):
                    rem = ordered[i:]
                    if rem:
                        rem_ms = [it[1] for it in rem]
                        rem_ns = [it[2] for it in rem]
                        sum_m_rem = sum(rem_ms)
                        sum_n_rem = sum(rem_ns)
                        used_mem_total = sum(m_sum)

                        # Residual bounds
                        indiv_rem = max(safe_div(nv, max(GPU_MEM_SIZE - mv, 1e-9)) for mv, nv in zip(rem_ms, rem_ns)) if rem_ms else 0.0
                        global_rem = safe_div(sum_n_rem, max(total_capacity - used_mem_total - sum_m_rem, 1e-9))

                        # Lightweight pair bound on remainder
                        pair_rem = 0.0
                        Q = min(len(rem), 120)
                        heavy_rem = sorted(rem, key=lambda it: it[1], reverse=True)[:Q]
                        for a in range(len(heavy_rem)):
                            ma = heavy_rem[a][1]; na = heavy_rem[a][2]
                            for b in range(a + 1, len(heavy_rem)):
                                mb = heavy_rem[b][1]; nb = heavy_rem[b][2]
                                if ma + mb > GPU_MEM_SIZE + 1e-12:
                                    denom = 2.0 * GPU_MEM_SIZE - (ma + mb)
                                    pair_rem = max(pair_rem, safe_div(na + nb, max(denom, 1e-9)))

                        # Small k-prefix on remainder (k up to 4)
                        kpref_rem = 0.0
                        by_m_rem = sorted(rem, key=lambda it: it[1], reverse=True)
                        for kpf in range(1, min(gpu_num, 4) + 1):
                            sm = 0.0; sn = 0.0
                            for it in by_m_rem:
                                sm += it[1]; sn += it[2]
                                if sm > (kpf - 1) * GPU_MEM_SIZE + 1e-12:
                                    break
                            denom = kpf * GPU_MEM_SIZE - sm
                            kpref_rem = max(kpref_rem, safe_div(sn, max(denom, 1e-9)))

                        T_new = max(T, indiv_rem, global_rem, pair_rem, kpref_rem)
                        if T_new > T + 1e-12:
                            T = T_new
                            cap = GPU_MEM_SIZE * T
                            used_cap = [n_sum[g] + T * m_sum[g] for g in range(gpu_num)]
                            if ordering == 0:
                                rem_sorted = sorted(rem, key=lambda it: (it[2] + T * it[1], it[2], it[1]), reverse=True)
                                ordered[i:] = rem_sorted
                    updated1 = True

                # Second lightweight retune around 75% placed: update T and capacities without reordering
                if (not updated2) and (i >= thr2) and (i < N):
                    rem = ordered[i:]
                    if rem:
                        rem_ms = [it[1] for it in rem]
                        rem_ns = [it[2] for it in rem]
                        sum_m_rem = sum(rem_ms)
                        sum_n_rem = sum(rem_ns)
                        used_mem_total = sum(m_sum)
                        indiv_rem2 = max(safe_div(nv, max(GPU_MEM_SIZE - mv, 1e-9)) for mv, nv in zip(rem_ms, rem_ns)) if rem_ms else 0.0
                        global_rem2 = safe_div(sum_n_rem, max(total_capacity - used_mem_total - sum_m_rem, 1e-9))
                        T_new2 = max(T, indiv_rem2, global_rem2)
                        if T_new2 > T + 1e-12:
                            T = T_new2
                            cap = GPU_MEM_SIZE * T
                            used_cap = [n_sum[g] + T * m_sum[g] for g in range(gpu_num)]
                    updated2 = True

            if return_placement:
                return True, {g: plc[g] for g in range(gpu_num)}
            return True

        # For feasibility checks during search (broaden variants to find smaller feasible T)
        def try_any(T, need_plc=False):
            variants = [
                (0, "resid"), (1, "resid"), (2, "resid"),
                (0, "minmax"), (1, "minmax"),
                (0, "hybrid"), (1, "hybrid")
            ]
            if need_plc:
                feas = []
                for ov, pol in variants:
                    ok, p = try_pack(T, ov, pol, True)
                    if ok:
                        feas.append(p)
                return (len(feas) > 0), feas
            else:
                for ov, pol in variants:
                    if try_pack(T, ov, pol, False):
                        return True
                return False

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
        Ts = [high * 0.985, high * 0.99, high * 0.995, high, high * 1.005, high * 1.01]
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

        # Micro-restarts with tiny jitter at the best T to diversify ties and reduce max KVPR
        jitter_vals = [1e-6, 2e-6, 3e-6]
        for jit in jitter_vals:
            for ov, pol in [(0, "minmax"), (0, "hybrid"), (1, "minmax")]:
                okj, plcj = try_pack(high, ov, pol, True, jitter=jit)
                if okj:
                    all_plcs.append(plcj)

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

    # Candidate C: ratio-desc best-fit to diversify memory-rate density packing
    plc_ratio = memory_pack(order="ratio_desc", strategy="bestfit")
    if plc_ratio is not None:
        candidates.append(plc_ratio)

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

    # Select best by lexicographic (max, second, avg) KVPR to break ties robustly
    def score_tuple(plc):
        kvprs = []
        for g in range(gpu_num):
            used_mem = sum(getattr(m, "model_size") for m in plc.get(g, []))
            numer = sum((getattr(m, "req_rate") / getattr(m, "slo")) for m in plc.get(g, []))
            kvprs.append(kvpr(numer, GPU_MEM_SIZE - used_mem))
        if not kvprs:
            return (0.0, 0.0, 0.0)
        kvprs_sorted = sorted(kvprs, reverse=True)
        max_k = kvprs_sorted[0]
        second = kvprs_sorted[1] if len(kvprs_sorted) > 1 else kvprs_sorted[0]
        avg = sum(kvprs_sorted) / len(kvprs_sorted)
        return (max_k, second, avg)

    best_plc = None
    best_score = None
    for plc in improved_candidates:
        st = score_tuple(plc)
        if best_score is None or st < best_score:
            best_score = st
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