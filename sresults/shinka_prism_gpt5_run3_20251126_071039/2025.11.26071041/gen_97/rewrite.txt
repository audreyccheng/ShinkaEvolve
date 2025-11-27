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
        A placement of models to GPUs: dict[gpu_id] = list of models
    """
    if gpu_num <= 0:
        raise ValueError("gpu_num must be positive")

    # -------------------------
    # Helpers
    # -------------------------
    def safe_div(num, den):
        if den <= 0:
            return float('inf') if num > 0 else 0.0
        return num / den

    def kvpr(numer, rem_mem):
        # KV cache pressure (KVPR): sum(req_rate/slo) / remaining_mem
        return safe_div(numer, rem_mem)

    def measured_max_kvpr(plc):
        vals = []
        for g in range(gpu_num):
            msum = sum(float(getattr(m, "model_size")) for m in plc.get(g, []))
            nsum = sum(float(getattr(m, "req_rate")) / max(float(getattr(m, "slo")), 1e-12) for m in plc.get(g, []))
            vals.append(kvpr(nsum, GPU_MEM_SIZE - msum))
        return max(vals) if vals else 0.0

    # -------------------------
    # Extract and validate model stats; split into memory-only and others
    # -------------------------
    items_all = []
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
        # Allow slo <= 0 as memory-only items (n == 0)
        n = (rr / slo) if slo > 0 else 0.0
        items_all.append((m, ms, float(n)))
        total_mem += ms
        total_n += float(n)

    if not items_all:
        return {g: [] for g in range(gpu_num)}

    total_capacity = gpu_num * GPU_MEM_SIZE
    if total_mem - total_capacity > 1e-9:
        raise ValueError("Total model memory exceeds total GPU memory")

    # Partition: memory-only (n≈0) and positive-load items
    eps_n = 1e-12
    M0 = [(m, ms, n) for (m, ms, n) in items_all if n <= eps_n]
    Mp = [(m, ms, n) for (m, ms, n) in items_all if n > eps_n]

    # -------------------------
    # Pre-place memory-only models (M0): largest-first to GPUs with most free memory
    # -------------------------
    placement_base = {g: [] for g in range(gpu_num)}
    rem_mem = [GPU_MEM_SIZE] * gpu_num  # remaining memory after M0 placement

    if M0:
        M0_sorted = sorted(M0, key=lambda it: (it[1], id(it[0])), reverse=True)
        for mdl, ms, _ in M0_sorted:
            # choose GPU with maximum remaining memory; tie by fewest models; then id
            best = None
            for g in range(gpu_num):
                if ms <= rem_mem[g] + 1e-12:
                    key = (-rem_mem[g], len(placement_base[g]), g)  # maximize rem_mem -> minimize -rem
                    if best is None or key < best[0]:
                        best = (key, g)
            if best is None:
                # Shouldn't happen given earlier single-GPU size check and total capacity check
                # Fallback: place to the first that fits ignoring tiny eps
                for g in range(gpu_num):
                    if ms <= rem_mem[g] + 1e-9:
                        placement_base[g].append(mdl)
                        rem_mem[g] -= ms
                        break
                else:
                    raise ValueError("Unable to place memory-only model due to fragmentation")
            else:
                g = best[1]
                placement_base[g].append(mdl)
                rem_mem[g] -= ms

    # After M0, per-GPU residual capacity for M+:
    Sg = rem_mem[:]  # free memory per GPU for M+
    # Ensure all M+ items can fit in some GPU after M0
    if Mp and max(ms for (_, ms, _) in Mp) - (max(Sg) if Sg else 0.0) > 1e-9:
        # As a safeguard, if fragmentation occurred, skip M0 preplacement by reverting to all items in Mp
        # Rewind preplacement
        placement_base = {g: [] for g in range(gpu_num)}
        Sg = [GPU_MEM_SIZE] * gpu_num
        Mp = items_all[:]  # treat all as M+
        M0 = []

    # -------------------------
    # Bounds for T on M+ with current Sg
    # -------------------------
    sum_Sg = sum(Sg)
    sum_ms_Mp = sum(ms for (_, ms, _) in Mp)
    sum_n_Mp = sum(n for (_, _, n) in Mp)

    if sum_ms_Mp - sum_Sg > 1e-9:
        raise ValueError("Remaining models' total memory exceeds remaining GPU memory")

    # Individual bound: per-item local KVPR lower bound
    indiv_lb = 0.0
    if Mp:
        indiv_lb = max(safe_div(n, max(GPU_MEM_SIZE - ms, 1e-9)) for (_, ms, n) in Mp)
    # Global bound using remaining memory
    global_lb = safe_div(sum_n_Mp, max(sum_Sg - sum_ms_Mp, 1e-9)) if Mp else 0.0

    # Pair bound (lightweight, still conservative w.r.t. full S=80)
    pair_lb = 0.0
    if len(Mp) >= 2 and gpu_num >= 2:
        L = min(len(Mp), 120)
        heavy = sorted(Mp, key=lambda it: it[1], reverse=True)[:L]
        for i in range(L):
            mi, ni = heavy[i][1], heavy[i][2]
            for j in range(i + 1, L):
                mj, nj = heavy[j][1], heavy[j][2]
                if mi + mj > GPU_MEM_SIZE + 1e-12:
                    denom = 2.0 * GPU_MEM_SIZE - (mi + mj)
                    pair_lb = max(pair_lb, safe_div(ni + nj, max(denom, 1e-9)))

    # Triplet bound (very light)
    triplet_lb = 0.0
    if len(Mp) >= 3 and gpu_num >= 3:
        Ltr = min(len(Mp), 60)
        top_by_mem = sorted(Mp, key=lambda it: it[1], reverse=True)[:Ltr]
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

    # k-prefix bound (very small k)
    kprefix_lb = 0.0
    if Mp:
        by_m = sorted(Mp, key=lambda it: it[1], reverse=True)
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

    # -------------------------
    # Balanced-slack assignment with residual-aware retuning
    # -------------------------
    def assign_balanced_slack(T, ordering=0, jitter_coeff=0.0, do_retune=True):
        """
        Place M+ items using K-after slack equalization at level T given per-GPU free memory Sg.
        Returns placement dict (including pre-placed M0) or None if infeasible at this T.
        """
        if not Mp:
            # No M+ items; return base placement
            return {g: list(placement_base[g]) for g in range(gpu_num)}

        # Order items by selected heuristic
        def ordered_items(Tcur):
            if ordering == 0:
                return sorted(Mp, key=lambda it: (it[2] + Tcur * it[1], it[2], it[1]), reverse=True)
            elif ordering == 1:
                return sorted(Mp, key=lambda it: (safe_div(it[2], max(GPU_MEM_SIZE - it[1], 1e-9)), it[1]), reverse=True)
            else:  # pressure per GB
                return sorted(Mp, key=lambda it: (safe_div(it[2], max(it[1], 1e-12)), it[2]), reverse=True)

        items_seq = ordered_items(T)
        # per-GPU dynamic state for M+ (M0 already placed)
        m_sum = [0.0] * gpu_num
        n_sum = [0.0] * gpu_num
        used_cap = [0.0] * gpu_num  # transformed usage: n_sum + Tcur * m_sum
        cap = [T * Sg[g] for g in range(gpu_num)]
        plc = {g: list(placement_base[g]) for g in range(gpu_num)}

        # Jitter to diversify tie-breaks across restarts
        # Compare GPUs by (-K_after - jitter_coeff*g, local_kvpr, -rem_mem_after, gpu_id)
        Tcur = max(T, 1e-12)
        placed = 0
        N = len(items_seq)
        thr1 = max(1, int(0.4 * N))
        thr2 = max(thr1 + 1, int(0.75 * N))
        retuned_once = False
        retuned_twice = False

        i = 0
        while i < len(items_seq):
            mdl, ms, n = items_seq[i]
            best = None
            # evaluate candidate GPUs
            for g in range(gpu_num):
                if m_sum[g] + ms > Sg[g] + 1e-12:
                    continue
                w = n + Tcur * ms
                resid = cap[g] - (used_cap[g] + w)  # K_after in transformed space
                if resid < -1e-12:
                    continue
                rem_after = Sg[g] - (m_sum[g] + ms)
                # local kvpr after placement (denominator is total remaining mem on that GPU)
                local_k = kvpr(n_sum[g] + n, rem_after)
                key = (-resid - jitter_coeff * g, local_k, -rem_after, g)
                if best is None or key < best[0]:
                    best = (key, g, w, rem_after, local_k)

            if best is None:
                return None  # infeasible at this T

            g = best[1]
            w = best[2]
            plc[g].append(mdl)
            m_sum[g] += ms
            n_sum[g] += n
            used_cap[g] += w
            placed += 1
            items_seq.pop(i)  # placed current

            # Retune at ~40% if residual lower bound rises above current T
            if do_retune and (not retuned_once) and placed >= thr1:
                # Residual global bound with current state
                rem_mem_total = sum(Sg[g] - m_sum[g] for g in range(gpu_num))
                rem_ms_total = sum(ms for (_, ms, _) in items_seq)
                rem_n_total = sum(n for (_, _, n) in items_seq)
                denom = max(rem_mem_total - rem_ms_total, 1e-9)
                lb_res = rem_n_total / denom if items_seq else 0.0
                if lb_res > 1.02 * Tcur + 1e-12:
                    Tcur = lb_res
                    cap = [Tcur * Sg[g] for g in range(gpu_num)]
                    used_cap = [n_sum[g] + Tcur * m_sum[g] for g in range(gpu_num)]
                    # Reorder remaining using updated Tcur if ordering=0
                    if ordering == 0 and items_seq:
                        items_seq.sort(key=lambda it: (it[2] + Tcur * it[1], it[2], it[1]), reverse=True)
                    i = 0
                retuned_once = True
                continue

            # Optional light second retune near 75%
            if do_retune and retuned_once and (not retuned_twice) and placed >= thr2:
                rem_mem_total = sum(Sg[g] - m_sum[g] for g in range(gpu_num))
                rem_ms_total = sum(ms for (_, ms, _) in items_seq)
                rem_n_total = sum(n for (_, _, n) in items_seq)
                denom = max(rem_mem_total - rem_ms_total, 1e-9)
                lb_res = rem_n_total / denom if items_seq else 0.0
                if lb_res > Tcur + 1e-12:
                    Tcur = lb_res
                    cap = [Tcur * Sg[g] for g in range(gpu_num)]
                    used_cap = [n_sum[g] + Tcur * m_sum[g] for g in range(gpu_num)]
                retuned_twice = True
                # do not reorder to save time

        return plc

    # Quick K-aware pre-refinement: move up to two heavy items out of worst GPU at T
    def quick_prefine(plc, T):
        if not Mp:
            return plc
        # build per-gpu mem and n from full plc
        mem = [sum(float(getattr(m, "model_size")) for m in plc.get(g, [])) for g in range(gpu_num)]
        num = [sum(float(getattr(m, "req_rate")) / max(float(getattr(m, "slo")), 1e-12) for m in plc.get(g, [])) for g in range(gpu_num)]
        # compute KVPRs and worst gpu
        kvprs = [kvpr(num[g], GPU_MEM_SIZE - mem[g]) for g in range(gpu_num)]
        worst = max(range(gpu_num), key=lambda g: kvprs[g])

        # helper for transformed slack K_g = T*(S - used_mem_after_Mp?) - n; here S is remaining mem after all placed M0
        # But we don't know M0 per GPU; compute Sg_now as GPU_MEM_SIZE - mem0[g], where mem0[g] is memory of memory-only items
        # We can reconstruct mem0[g] by counting models in M0 set
        M0_ids = set(id(mdl) for (mdl, _, _) in M0)
        mem0 = [sum(float(getattr(m, "model_size")) for m in plc.get(g, []) if id(m) in M0_ids) for g in range(gpu_num)]
        Sg_now = [GPU_MEM_SIZE - mem0[g] for g in range(gpu_num)]

        # pick top-2 by weight w = n + T*ms from worst
        def dn_of(m): return float(getattr(m, "req_rate")) / max(float(getattr(m, "slo")), 1e-12)
        def sz_of(m): return float(getattr(m, "model_size"))
        def w_of(m): return dn_of(m) + T * sz_of(m)

        worst_list = list(plc.get(worst, []))
        if len(worst_list) == 0:
            return plc
        # filter to M+ candidates (exclude M0)
        worst_candidates = [m for m in worst_list if id(m) not in M0_ids]
        if not worst_candidates:
            return plc
        top_cand = sorted(worst_candidates, key=w_of, reverse=True)[:2]
        changed = False

        for mdl in top_cand:
            ms = sz_of(mdl)
            dn = dn_of(mdl)
            # try move to GPU with largest nonnegative K that fits
            best_tgt = None
            best_K = -float('inf')
            for g in range(gpu_num):
                if g == worst:
                    continue
                # memory feasibility
                if (mem[g] + ms) - GPU_MEM_SIZE > 1e-12:
                    continue
                # transformed slack after move
                # For M+ capacity, use Sg_now[g] as residual budget; current M+ memory on g is mem[g] - mem0[g]
                mg_plus = mem[g] - mem0[g]
                ng = num[g]
                K_after = T * (Sg_now[g] - (mg_plus + ms)) - (ng + dn)
                if K_after >= -1e-12 and K_after > best_K:
                    best_K = K_after
                    best_tgt = g
            if best_tgt is None:
                continue
            # evaluate impact on max KVPR
            src = worst
            new_mem_src = mem[src] - ms
            new_num_src = num[src] - dn
            new_mem_tgt = mem[best_tgt] + ms
            new_num_tgt = num[best_tgt] + dn
            src_k = kvpr(new_num_src, GPU_MEM_SIZE - new_mem_src)
            tgt_k = kvpr(new_num_tgt, GPU_MEM_SIZE - new_mem_tgt)
            new_max = max(src_k, tgt_k)
            for g in range(gpu_num):
                if g not in (src, best_tgt):
                    if kvprs[g] > new_max:
                        new_max = kvprs[g]
            if new_max + 1e-12 < max(kvprs):
                # apply move
                plc[src].remove(mdl)
                plc[best_tgt].append(mdl)
                mem[src] -= ms; num[src] -= dn
                mem[best_tgt] += ms; num[best_tgt] += dn
                kvprs = [kvpr(num[g], GPU_MEM_SIZE - mem[g]) for g in range(gpu_num)]
                worst = max(range(gpu_num), key=lambda g: kvprs[g])
                changed = True
        return plc if changed else plc

    # -------------------------
    # Feasibility-only search for minimal T
    # -------------------------
    def try_feasible(T):
        # Try a couple of orderings; accept if any succeeds
        for ov in (0, 1, 2):
            plc = assign_balanced_slack(T, ordering=ov, jitter_coeff=0.0, do_retune=True)
            if plc is not None:
                return True
        return False

    if not Mp:
        # Only memory-only models; return their placement
        for g in range(gpu_num):
            placement_base.setdefault(g, [])
        return placement_base

    # Exponential search to find first feasible T
    T = max(low_T, 1e-9)
    found = False
    for _ in range(50):
        if try_feasible(T):
            found = True
            break
        T *= 2.0
    if not found:
        # Fallback to simple memory packing on all items
        # (Should not happen under normal constraints)
        # Place largest-first on max-free
        ordered = sorted(items_all, key=lambda it: (it[1], it[2]), reverse=True)
        placement = {g: [] for g in range(gpu_num)}
        rem = [GPU_MEM_SIZE] * gpu_num
        for mdl, ms, _ in ordered:
            best = max(range(gpu_num), key=lambda g: (rem[g] - ms if rem[g] >= ms - 1e-12 else -float('inf'), -len(placement[g]), -g))
            if rem[best] + 1e-12 < ms:
                raise ValueError("Unable to construct any feasible placement")
            placement[best].append(mdl)
            rem[best] -= ms
        return placement

    # Feasibility-only binary search for smallest feasible T
    lo, hi = low_T, T
    binary_iters = 8
    for _ in range(binary_iters):
        mid = (lo + hi) / 2.0
        if try_feasible(mid):
            hi = mid
        else:
            lo = mid

    T_star = hi

    # -------------------------
    # Build T* neighborhood candidates with micro-restarts + quick pre-refine
    # -------------------------
    Ts = [T_star * 0.99, T_star, T_star * 1.01]
    candidates = []

    for Tv in Ts:
        # Micro-restarts with small jitter
        S_total = sum_Sg
        jitter_base = 1e-6 * max(Tv, 1e-12) * max(S_total, 1e-12)
        for r in range(4):  # micro_restarts
            jitter = jitter_base * (r + 1)
            plc = assign_balanced_slack(Tv, ordering=r % 3, jitter_coeff=jitter, do_retune=True)
            if plc is not None:
                plc = quick_prefine(plc, Tv)
                candidates.append(plc)

    # -------------------------
    # Fallback candidate generators (regret-based and memory packing)
    # -------------------------
    def regret_insertion(items):
        placement = {g: [] for g in range(gpu_num)}
        rem = [GPU_MEM_SIZE] * gpu_num
        numer = [0.0] * gpu_num
        unassigned = list(items)

        def top12(vals):
            top = (-1, -float('inf')); second = (-1, -float('inf'))
            for i, v in enumerate(vals):
                if v > top[1]:
                    second = top; top = (i, v)
                elif v > second[1]:
                    second = (i, v)
            return top, second

        while unassigned:
            current_kvprs = [kvpr(numer[g], rem[g]) for g in range(gpu_num)]
            (top_idx, top_val), (sec_idx, sec_val) = top12(current_kvprs)

            best_model = None
            best_gpu = None
            best_new_max = float('inf')
            best_regret = -float('inf')

            for (mdl, ms, dn) in unassigned:
                feas = []
                for g in range(gpu_num):
                    if ms <= rem[g] + 1e-12:
                        new_local = kvpr(numer[g] + dn, rem[g] - ms)
                        base_other = top_val if g != top_idx else sec_val
                        new_max = new_local if new_local > base_other else base_other
                        feas.append((g, new_max, new_local))
                if not feas:
                    return None
                feas.sort(key=lambda x: (x[1], x[2], x[0]))
                a = feas[0]
                b = feas[1] if len(feas) > 1 else (None, float('inf'), float('inf'))
                regret = b[1] - a[1]
                if (regret > best_regret) or (regret == best_regret and a[1] < best_new_max):
                    best_regret = regret
                    best_new_max = a[1]
                    best_model = (mdl, ms, dn)
                    best_gpu = a[0]

            if best_model is None:
                return None

            mdl, ms, dn = best_model
            placement[best_gpu].append(mdl)
            rem[best_gpu] -= ms
            numer[best_gpu] += dn
            unassigned.remove(best_model)

        return placement

    def memory_pack(items, order="size_desc", strategy="dual"):
        # order: "size_desc" or "ratio_desc"
        # strategy: "dual" (30% maxfree + bestfit), "bestfit", "maxfree", "firstfit"
        if order == "size_desc":
            ordered = sorted(items, key=lambda it: (it[1], it[2]), reverse=True)
        else:
            ordered = sorted(items, key=lambda it: (safe_div(it[2], max(it[1], 1e-9)), it[2]), reverse=True)

        placement = {g: [] for g in range(gpu_num)}
        rem = [GPU_MEM_SIZE] * gpu_num
        numer = [0.0] * gpu_num
        split_idx = max(0, int(0.3 * len(ordered))) if strategy == "dual" else 0

        for idx, (mdl, ms, dn) in enumerate(ordered):
            cands = []
            for g in range(gpu_num):
                if ms <= rem[g] + 1e-12:
                    new_k = kvpr(numer[g] + dn, rem[g] - ms)
                    cands.append((g, rem[g] - ms, new_k))
            if not cands:
                return None
            if strategy == "bestfit" or (strategy == "dual" and idx >= split_idx):
                cands.sort(key=lambda x: (x[1], x[2], x[0]))
                chosen = cands[0][0]
            elif strategy == "maxfree" or (strategy == "dual" and idx < split_idx):
                cands.sort(key=lambda x: (-x[1], x[2], x[0]))
                chosen = cands[0][0]
            else:
                chosen = min(cands, key=lambda x: x[0])[0]
            placement[chosen].append(mdl)
            rem[chosen] -= ms
            numer[chosen] += dn

        return placement

    if not candidates:
        # try regret-based on all items
        plc_regret = regret_insertion(items_all)
        if plc_regret is not None:
            candidates.append(plc_regret)
        plc_mem = memory_pack(items_all, order="size_desc", strategy="dual")
        if plc_mem is not None:
            candidates.append(plc_mem)
        plc_mem_ratio = memory_pack(items_all, order="ratio_desc", strategy="dual")
        if plc_mem_ratio is not None:
            candidates.append(plc_mem_ratio)
        if not candidates:
            for strat in ("bestfit", "maxfree", "firstfit"):
                plc_try = memory_pack(items_all, order="size_desc", strategy=strat)
                if plc_try is not None:
                    candidates.append(plc_try)
                    break

    if not candidates:
        raise ValueError("Unable to construct any feasible placement")

    # -------------------------
    # Local improvement: move/swap (kept simple and bounded)
    # -------------------------
    def improve_local(plc, max_iters=3000, eps=1e-12):
        per_g = {g: list(plc.get(g, [])) for g in range(gpu_num)}
        mem = [sum(float(getattr(m, "model_size")) for m in per_g[g]) for g in range(gpu_num)]
        num = [sum(float(getattr(m, "req_rate")) / max(float(getattr(m, "slo")), 1e-12) for m in per_g[g]) for g in range(gpu_num)]

        def kvpr_g(g, msum=None, nsum=None):
            msum = mem[g] if msum is None else msum
            nsum = num[g] if nsum is None else nsum
            return kvpr(nsum, GPU_MEM_SIZE - msum)

        it = 0
        while it < max_iters:
            it += 1
            vals = [kvpr_g(g) for g in range(gpu_num)]
            cur_max = max(vals)
            worst = max(range(gpu_num), key=lambda g: vals[g])
            improved = False
            best_new_max = cur_max
            best_move = None

            worst_models = list(per_g[worst])
            for mdl in worst_models:
                ms = float(getattr(mdl, "model_size"))
                dn = float(getattr(mdl, "req_rate")) / max(float(getattr(mdl, "slo")), 1e-12)
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
                    new_max = max(src_k, tgt_k)
                    for g in range(gpu_num):
                        if g != worst and g != tgt and vals[g] > new_max:
                            new_max = vals[g]
                    if new_max + eps < best_new_max:
                        best_new_max = new_max
                        best_move = ("move", mdl, worst, tgt, ms, dn)
                        improved = True
            if improved:
                _, mdl, src, tgt, ms, dn = best_move
                per_g[src].remove(mdl); per_g[tgt].append(mdl)
                mem[src] -= ms; num[src] -= dn
                mem[tgt] += ms; num[tgt] += dn
                continue

            # Try first improving swap between worst and others
            found_swap = False
            for mdl_a in worst_models:
                ms_a = float(getattr(mdl_a, "model_size"))
                dn_a = float(getattr(mdl_a, "req_rate")) / max(float(getattr(mdl_a, "slo")), 1e-12)
                for tgt in range(gpu_num):
                    if tgt == worst:
                        continue
                    for mdl_b in list(per_g[tgt]):
                        ms_b = float(getattr(mdl_b, "model_size"))
                        dn_b = float(getattr(mdl_b, "req_rate")) / max(float(getattr(mdl_b, "slo")), 1e-12)
                        if mem[worst] - ms_a + ms_b > GPU_MEM_SIZE + 1e-12: continue
                        if mem[tgt] - ms_b + ms_a > GPU_MEM_SIZE + 1e-12: continue
                        src_mem = mem[worst] - ms_a + ms_b
                        src_num = num[worst] - dn_a + dn_b
                        tgt_mem = mem[tgt] - ms_b + ms_a
                        tgt_num = num[tgt] - dn_b + dn_a
                        src_k = kvpr(src_num, GPU_MEM_SIZE - src_mem)
                        tgt_k = kvpr(tgt_num, GPU_MEM_SIZE - tgt_mem)
                        new_max = max(src_k, tgt_k)
                        for g in range(gpu_num):
                            if g != worst and g != tgt and kvpr_g(g) > new_max:
                                new_max = kvpr_g(g)
                        if new_max + eps < cur_max:
                            per_g[worst].remove(mdl_a)
                            per_g[tgt].remove(mdl_b)
                            per_g[worst].append(mdl_b)
                            per_g[tgt].append(mdl_a)
                            mem[worst] = src_mem; num[worst] = src_num
                            mem[tgt] = tgt_mem; num[tgt] = tgt_num
                            found_swap = True
                            break
                    if found_swap: break
                if found_swap: break
            if found_swap:
                continue
            break  # no improvement

        return {g: per_g.get(g, []) for g in range(gpu_num)}

    improved = [improve_local(plc) for plc in candidates]

    # Compare candidates by lexicographic KVPR vector (descending)
    def score_lex(plc):
        vals = []
        for g in range(gpu_num):
            msum = sum(float(getattr(m, "model_size")) for m in plc.get(g, []))
            nsum = sum(float(getattr(m, "req_rate")) / max(float(getattr(m, "slo")), 1e-12) for m in plc.get(g, []))
            vals.append(kvpr(nsum, GPU_MEM_SIZE - msum))
        if not vals:
            return (0.0,), 0.0
        vec = tuple(sorted(vals, reverse=True))
        return vec, vec[0]

    best_plc = None
    best_key = None
    for plc in improved:
        key = score_lex(plc)
        if best_key is None or key < best_key:
            best_key = key
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