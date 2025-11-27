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

    # ---------- Helpers ----------
    def safe_div(num, den):
        if den <= 0:
            return float('inf') if num > 0 else 0.0
        return num / den

    def kvpr_val(numer, rem_mem):
        return safe_div(numer, rem_mem)

    # ---------- Extract per-model stats ----------
    active_items = []      # (idx, model, ms, n) with n > 0
    memonly_items = []     # (idx, model, ms) with n == 0
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
        # Treat slo==0 or req_rate==0 as memory-only (no KV load)
        if slo < 0:
            raise ValueError("Model SLO must be non-negative")
        n = 0.0 if (slo == 0 or rr == 0) else (rr / slo)
        if n > 0:
            active_items.append((idx, m, ms, n))
            total_n += n
        else:
            memonly_items.append((idx, m, ms))
        total_mem += ms

    if not active_items and not memonly_items:
        return {gpu_id: [] for gpu_id in range(gpu_num)}

    total_capacity_mem = gpu_num * GPU_MEM_SIZE
    if total_mem - total_capacity_mem > 1e-9:
        raise ValueError("Total model memory exceeds total GPU memory")

    # ---------- Pre-place memory-only models (largest-size-first, max-free) ----------
    pre_plc = {g: [] for g in range(gpu_num)}
    pre_mem = [0.0] * gpu_num  # memory committed by memory-only models
    if memonly_items:
        memonly_sorted = sorted(memonly_items, key=lambda it: it[2], reverse=True)
        # Greedy: place each onto GPU with most remaining memory; fallback to best-fit if needed
        def try_place_memonly(order, strategy="maxfree"):
            local_plc = {g: [] for g in range(gpu_num)}
            rem = [GPU_MEM_SIZE] * gpu_num
            for _, mdl, ms in order:
                cand = []
                for g in range(gpu_num):
                    if ms <= rem[g]:
                        cand.append((g, rem[g] - ms))
                if not cand:
                    return None
                if strategy == "maxfree":
                    cand.sort(key=lambda x: (-x[1], x[0]))
                else:
                    cand.sort(key=lambda x: (x[1], x[0]))
                g = cand[0][0]
                local_plc[g].append(mdl)
                rem[g] -= ms
            return local_plc, [GPU_MEM_SIZE - r for r in rem]

        res = try_place_memonly(memonly_sorted, "maxfree")
        if res is None:
            res = try_place_memonly(memonly_sorted, "bestfit")
        if res is not None:
            plc0, used0 = res
            pre_plc = plc0
            pre_mem = used0
        else:
            # If greedy fails (rare), skip pre-placement and let main packer handle all
            pre_plc = {g: [] for g in range(gpu_num)}
            pre_mem = [0.0] * gpu_num
            # Merge memory-only items back to active with zero demand (safe)
            for idx, mdl, ms in memonly_items:
                active_items.append((idx, mdl, ms, 0.0))
            memonly_items = []

    # ---------- Lower bound on T (using active items only) ----------
    if active_items:
        indiv_lb = max(safe_div(n, GPU_MEM_SIZE - ms) for _, _, ms, n in active_items)
        global_lb = safe_div(total_n, max(total_capacity_mem - total_mem, 1e-9))
        # Pair bound: for heavy pairs that cannot co-reside (m_i + m_j > 80),
        # T >= (n_i + n_j) / (2*80 - (m_i + m_j))
        pair_lb = 0.0
        P = min(len(active_items), 120)
        heavy = sorted(active_items, key=lambda it: it[2], reverse=True)[:P]
        for i in range(len(heavy)):
            _, _, mi, ni = heavy[i]
            for j in range(i + 1, len(heavy)):
                _, _, mj, nj = heavy[j]
                if mi + mj > GPU_MEM_SIZE + 1e-12:
                    denom = 2 * GPU_MEM_SIZE - (mi + mj)
                    pair_lb = max(pair_lb, safe_div(ni + nj, max(denom, 1e-9)))
        # Triplet bound (light): check small neighborhoods among largest by size
        triplet_lb = 0.0
        L = min(len(active_items), 60)
        top_by_mem = sorted(active_items, key=lambda it: it[2], reverse=True)[:L]
        for i in range(L):
            mi = top_by_mem[i][2]; ni = top_by_mem[i][3]
            for j in range(i + 1, min(L, i + 1 + 8)):
                mj = top_by_mem[j][2]; nj = top_by_mem[j][3]
                for k in range(j + 1, min(L, j + 1 + 8)):
                    mk = top_by_mem[k][2]; nk = top_by_mem[k][3]
                    total_m = mi + mj + mk
                    if total_m > 2.0 * GPU_MEM_SIZE + 1e-12:
                        denom = 3.0 * GPU_MEM_SIZE - total_m
                        triplet_lb = max(triplet_lb, safe_div(ni + nj + nk, max(denom, 1e-9)))
        # k-bin prefix bound for k = 1..min(G,6)
        kprefix_lb = 0.0
        items_by_m = sorted(active_items, key=lambda it: it[2], reverse=True)
        for k in range(1, min(gpu_num, 6) + 1):
            sum_m = 0.0
            sum_n = 0.0
            for it in items_by_m:
                sum_m += it[2]
                sum_n += it[3]
                if sum_m > (k - 1) * GPU_MEM_SIZE + 1e-12:
                    break
            denom = k * GPU_MEM_SIZE - sum_m
            kprefix_lb = max(kprefix_lb, safe_div(sum_n, max(denom, 1e-9)))
        low_T = max(0.0, indiv_lb, global_lb, pair_lb, triplet_lb, kprefix_lb)
    else:
        # No active items -> KVPR is zero; any placement is fine
        placement = {g: list(pre_plc.get(g, [])) for g in range(gpu_num)}
        for g in range(gpu_num):
            placement.setdefault(g, [])
        return placement

    # ---------- Feasibility check and packing at T ----------
    # order_variant:
    #   0 -> by transformed weight w(T) = n + T*m
    #   1 -> by intrinsic-alone KVPR n / (80 - m)
    # policy:
    #   "resid"  -> best-fit on transformed residual
    #   "hybrid" -> J = K_after_norm + α·kv_new_norm + β·mem_imbalance
    def try_pack(T, order_variant=0, policy="hybrid", return_placement=False):
        cap = GPU_MEM_SIZE * T
        eps = 1e-12

        # Build sorted order on ACTIVE items only
        if order_variant == 0:
            ordered = sorted(active_items, key=lambda it: (it[3] + T * it[2], it[3], it[2]), reverse=True)
        else:
            ordered = sorted(active_items, key=lambda it: (safe_div(it[3], max(GPU_MEM_SIZE - it[2], 1e-9)), it[2]), reverse=True)

        # Per-GPU state (seeded with pre-placed memory-only models)
        n_sum = [0.0] * gpu_num
        m_sum = [float(pre_mem[g]) for g in range(gpu_num)]
        used_cap = [T * m_sum[g] for g in range(gpu_num)]  # equals n_sum + T * m_sum
        placement = [list(pre_plc.get(g, [])) for g in range(gpu_num)]

        # Two adaptive retunes: after ~40% (with reorder) and ~75% (light, no reorder)
        did_retune1 = False
        did_retune2 = False
        thr1 = max(1, int(0.40 * len(ordered)))
        thr2 = max(thr1 + 1, int(0.75 * len(ordered)))
        i = 0
        while i < len(ordered):
            _, mdl, ms, n = ordered[i]
            w = n + T * ms
            candidates = []

            # Precompute current KVPRs and balancing terms
            cur_kvprs = [kvpr_val(n_sum[g], GPU_MEM_SIZE - m_sum[g]) for g in range(gpu_num)]
            mean_k = sum(cur_kvprs) / gpu_num if gpu_num > 0 else 0.0
            var_k = sum((x - mean_k) ** 2 for x in cur_kvprs) / gpu_num if gpu_num > 0 else 0.0
            Tnorm = max(T, 1e-12)
            alpha = 0.15
            if var_k < 0.02 * (Tnorm ** 2):
                alpha = 0.25
            beta = 0.05

            total_m_after = sum(m_sum) + ms
            avg_mem_frac = total_m_after / (gpu_num * GPU_MEM_SIZE) if gpu_num > 0 else 0.0

            for g in range(gpu_num):
                if m_sum[g] + ms > GPU_MEM_SIZE + eps:
                    continue
                residual = cap - (used_cap[g] + w)
                if residual < -eps:
                    continue
                if policy == "resid":
                    candidates.append((g, residual, None))
                else:
                    new_n = n_sum[g] + n
                    new_m = m_sum[g] + ms
                    new_local = kvpr_val(new_n, GPU_MEM_SIZE - new_m)
                    new_max = new_local
                    for h in range(gpu_num):
                        if h != g and cur_kvprs[h] > new_max:
                            new_max = cur_kvprs[h]
                    mem_frac_after = new_m / GPU_MEM_SIZE
                    mem_imb = abs(mem_frac_after - avg_mem_frac)
                    J = (new_max / Tnorm) + alpha * (new_local / Tnorm) + beta * mem_imb
                    # key: (J, new_max, residual, more_remaining_mem, gpu_id)
                    key = (J, new_max, residual, -(GPU_MEM_SIZE - new_m), g)
                    candidates.append((g, key, None))

            if not candidates:
                return (False, None) if return_placement else False

            if policy == "resid":
                chosen = min(candidates, key=lambda c: (c[1], -(GPU_MEM_SIZE - (m_sum[c[0]] + ms)), c[0]))[0]
            else:
                chosen = min(candidates, key=lambda c: c[1])[0]

            # Place
            placement[chosen].append(mdl)
            n_sum[chosen] += n
            m_sum[chosen] += ms
            used_cap[chosen] += w
            i += 1

            # Adaptive retuning 1 (with reorder)
            if (not did_retune1) and (i >= thr1) and (i < len(ordered)):
                rem = ordered[i:]
                if rem:
                    rem_ms = [it[2] for it in rem]
                    rem_ns = [it[3] for it in rem]
                    sum_m_rem = sum(rem_ms)
                    sum_n_rem = sum(rem_ns)
                    used_mem_total = sum(m_sum)

                    indiv_rem = 0.0
                    for mv, nv in zip(rem_ms, rem_ns):
                        indiv_rem = max(indiv_rem, safe_div(nv, GPU_MEM_SIZE - mv))
                    global_rem = safe_div(sum_n_rem, max(gpu_num * GPU_MEM_SIZE - used_mem_total - sum_m_rem, 1e-9))

                    pair_rem = 0.0
                    Q = min(len(rem), 100)
                    heavy_rem = sorted(rem, key=lambda it: it[2], reverse=True)[:Q]
                    for a in range(len(heavy_rem)):
                        ma = heavy_rem[a][2]; na = heavy_rem[a][3]
                        for b in range(a + 1, len(heavy_rem)):
                            mb = heavy_rem[b][2]; nb = heavy_rem[b][3]
                            if ma + mb > GPU_MEM_SIZE + 1e-12:
                                denom = 2.0 * GPU_MEM_SIZE - (ma + mb)
                                pair_rem = max(pair_rem, safe_div(na + nb, max(denom, 1e-9)))

                    trip_rem = 0.0
                    L2 = min(len(rem), 36)
                    top_rem = sorted(rem, key=lambda it: it[2], reverse=True)[:L2]
                    for a in range(L2):
                        ma = top_rem[a][2]; na = top_rem[a][3]
                        for b in range(a + 1, min(L2, a + 1 + 6)):
                            mb = top_rem[b][2]; nb = top_rem[b][3]
                            for c in range(b + 1, min(L2, b + 1 + 6)):
                                mc = top_rem[c][2]; nc = top_rem[c][3]
                                tm = ma + mb + mc
                                if tm > 2.0 * GPU_MEM_SIZE + 1e-12:
                                    denom = 3.0 * GPU_MEM_SIZE - tm
                                    trip_rem = max(trip_rem, safe_div(na + nb + nc, max(denom, 1e-9)))

                    kpref_rem = 0.0
                    by_m_rem = sorted(rem, key=lambda it: it[2], reverse=True)
                    for kpf in range(1, min(gpu_num, 4) + 1):
                        sm = 0.0; sn = 0.0
                        for it in by_m_rem:
                            sm += it[2]; sn += it[3]
                            if sm > (kpf - 1) * GPU_MEM_SIZE + 1e-12:
                                break
                        denom = kpf * GPU_MEM_SIZE - sm
                        kpref_rem = max(kpref_rem, safe_div(sn, max(denom, 1e-9)))

                    T_new = max(T, indiv_rem, global_rem, pair_rem, trip_rem, kpref_rem)
                    if T_new > T + 1e-12:
                        T = T_new
                        cap = GPU_MEM_SIZE * T
                        used_cap = [n_sum[g] + T * m_sum[g] for g in range(gpu_num)]
                        # Reorder remaining based on the current order_variant
                        if order_variant == 0:
                            rem_sorted = sorted(rem, key=lambda it: (it[3] + T * it[2], it[3], it[2]), reverse=True)
                        else:
                            rem_sorted = sorted(rem, key=lambda it: (safe_div(it[3], max(GPU_MEM_SIZE - it[2], 1e-9)), it[2]), reverse=True)
                        ordered[i:] = rem_sorted
                did_retune1 = True

            # Adaptive retuning 2 (light, no reorder)
            if (not did_retune2) and (i >= thr2) and (i < len(ordered)):
                rem = ordered[i:]
                if rem:
                    sum_m_rem = sum(it[2] for it in rem)
                    sum_n_rem = sum(it[3] for it in rem)
                    used_mem_total = sum(m_sum)
                    global_rem = safe_div(sum_n_rem, max(gpu_num * GPU_MEM_SIZE - used_mem_total - sum_m_rem, 1e-9))
                    if global_rem > T + 1e-12:
                        T = global_rem
                        cap = GPU_MEM_SIZE * T
                        used_cap = [n_sum[g] + T * m_sum[g] for g in range(gpu_num)]
                did_retune2 = True

        if return_placement:
            return True, {g: placement[g] for g in range(gpu_num)}
        return True

    # For quick feasibility checks during search
    def try_pack_any(T, need_placement=False):
        variants = [(0, "hybrid"), (1, "hybrid"), (0, "resid")]
        feasibles = []
        for ov, pol in variants:
            if need_placement:
                ok, plc = try_pack(T, ov, pol, True)
                if ok:
                    feasibles.append(plc)
            else:
                if try_pack(T, ov, pol, False):
                    return True
        if need_placement:
            return (len(feasibles) > 0), feasibles
        return False

    # ---------- Exponential search for feasible T ----------
    T = max(low_T, 1e-9)
    found = False
    for _ in range(40):
        if try_pack_any(T, need_placement=False):
            found = True
            break
        # Early jump: if T is below static lower bound, jump up quickly, else double
        T = max(2.0 * T, low_T)
    if not found:
        raise ValueError("Unable to find a feasible packing for any KVPR threshold")

    # ---------- Binary search to minimize T ----------
    low, high = low_T, T
    for _ in range(32):
        mid = (low + high) / 2.0
        if mid <= 0:
            high = mid
            continue
        if try_pack_any(mid, need_placement=False):
            high = mid
        else:
            low = mid

    # ---------- Build candidates near T* and select the best ----------
    def measured_max_kvpr(plc):
        vals = []
        for g in range(gpu_num):
            used_mem = sum(getattr(m, "model_size") for m in plc.get(g, []))
            numer = sum((getattr(m, "req_rate") / getattr(m, "slo")) if getattr(m, "slo") != 0 else 0.0 for m in plc.get(g, []))
            vals.append(kvpr_val(numer, GPU_MEM_SIZE - used_mem))
        return max(vals) if vals else 0.0

    candidates = []
    for Tv in [high, high * 0.998, high * 1.002]:
        for ov, pol in [(0, "hybrid"), (1, "hybrid"), (0, "resid")]:
            ok, plc = try_pack(Tv, ov, pol, True)
            if ok:
                candidates.append(plc)

    if not candidates:
        # Fallback: at least one feasible must exist
        ok, plc = try_pack(high, 0, "hybrid", True)
        if not ok:
            raise ValueError("Feasible packing unexpectedly unavailable")
        candidates.append(plc)

    best_plc = None
    best_score = float('inf')
    for plc in candidates:
        score = measured_max_kvpr(plc)
        if score < best_score:
            best_score = score
            best_plc = plc

    # ---------- Lightweight local improvement ----------
    def local_improve(plc, max_moves=50, swap_budget=6, eps=1e-12):
        per_g = {g: list(plc.get(g, [])) for g in range(gpu_num)}
        mem = [sum(getattr(m, "model_size") for m in per_g[g]) for g in range(gpu_num)]
        num = [sum(((getattr(m, "req_rate") / getattr(m, "slo")) if getattr(m, "slo") != 0 else 0.0) for m in per_g[g]) for g in range(gpu_num)]

        def kv_g(g, msum=None, nsum=None):
            msum = mem[g] if msum is None else msum
            nsum = num[g] if nsum is None else nsum
            return kvpr_val(nsum, GPU_MEM_SIZE - msum)

        def global_vals():
            vals = [kv_g(g) for g in range(gpu_num)]
            return max(vals), vals

        # Single-item moves from worst GPU
        moves = 0
        while moves < max_moves:
            cur_max, vals = global_vals()
            worst = max(range(gpu_num), key=lambda g: vals[g])

            improved = False
            best_new_max = cur_max
            best_move = None

            for mdl in list(per_g[worst]):
                ms = float(getattr(mdl, "model_size"))
                dn = (float(getattr(mdl, "req_rate")) / float(getattr(mdl, "slo"))) if float(getattr(mdl, "slo")) != 0 else 0.0
                for tgt in range(gpu_num):
                    if tgt == worst:
                        continue
                    if mem[tgt] + ms > GPU_MEM_SIZE + 1e-12:
                        continue

                    src_mem = mem[worst] - ms
                    src_num = num[worst] - dn
                    tgt_mem = mem[tgt] + ms
                    tgt_num = num[tgt] + dn

                    src_k = kvpr_val(src_num, GPU_MEM_SIZE - src_mem)
                    tgt_k = kvpr_val(tgt_num, GPU_MEM_SIZE - tgt_mem)

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
            mem[src] -= ms
            num[src] -= dn
            mem[tgt] += ms
            num[tgt] += dn
            moves += 1

        # A few first-improving swaps involving the current worst GPU
        for _ in range(swap_budget):
            cur_max, vals = global_vals()
            worst = max(range(gpu_num), key=lambda g: vals[g])
            improved = False

            for mdl_a in list(per_g[worst]):
                ms_a = float(getattr(mdl_a, "model_size"))
                dn_a = (float(getattr(mdl_a, "req_rate")) / float(getattr(mdl_a, "slo"))) if float(getattr(mdl_a, "slo")) != 0 else 0.0
                for tgt in range(gpu_num):
                    if tgt == worst:
                        continue
                    for mdl_b in list(per_g[tgt]):
                        ms_b = float(getattr(mdl_b, "model_size"))
                        dn_b = (float(getattr(mdl_b, "req_rate")) / float(getattr(mdl_b, "slo"))) if float(getattr(mdl_b, "slo")) != 0 else 0.0

                        if mem[worst] - ms_a + ms_b > GPU_MEM_SIZE + 1e-12:
                            continue
                        if mem[tgt] - ms_b + ms_a > GPU_MEM_SIZE + 1e-12:
                            continue

                        src_mem = mem[worst] - ms_a + ms_b
                        src_num = num[worst] - dn_a + dn_b
                        tgt_mem = mem[tgt] - ms_b + ms_a
                        tgt_num = num[tgt] - dn_b + dn_a

                        src_k = kvpr_val(src_num, GPU_MEM_SIZE - src_mem)
                        tgt_k = kvpr_val(tgt_num, GPU_MEM_SIZE - tgt_mem)

                        new_max = max(src_k, tgt_k)
                        for g in range(gpu_num):
                            if g != worst and g != tgt and vals[g] > new_max:
                                new_max = vals[g]

                        if new_max + eps < cur_max:
                            per_g[worst].remove(mdl_a)
                            per_g[tgt].remove(mdl_b)
                            per_g[worst].append(mdl_b)
                            per_g[tgt].append(mdl_a)
                            mem[worst] = src_mem
                            num[worst] = src_num
                            mem[tgt] = tgt_mem
                            num[tgt] = tgt_num
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break

        return {g: per_g.get(g, []) for g in range(gpu_num)}

    best_plc = local_improve(best_plc, max_moves=40, swap_budget=6)

    # Ensure all GPUs are represented in dict keys
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