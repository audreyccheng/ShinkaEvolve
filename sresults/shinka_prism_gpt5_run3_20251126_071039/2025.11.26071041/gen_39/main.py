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
    # Safe KVPR
    def kvpr(R, rem_mem):
        if rem_mem <= 0:
            return float('inf')
        return R / rem_mem

    # Trivial cases
    empty = {i: [] for i in range(gpu_num)}
    if gpu_num <= 0 or not models:
        return empty

    S = GPU_MEM_SIZE
    n = len(models)

    # Extract arrays and basic checks
    sizes = []
    rates = []
    can_fit_gpu = []  # per-item mask of GPUs that can fit by size alone
    for m in models:
        s = float(m.model_size)
        if getattr(m, 'slo', 0) == 0:
            # If SLO is zero, KVPR would be undefined/infinite. Use a sane large surrogate to remain comparable.
            r = 1e9
        else:
            r = float(m.req_rate) / float(m.slo)
        sizes.append(s)
        rates.append(r)
        can_fit_gpu.append([s <= S + 1e-9] * gpu_num)

    # Ensure each model fits at least one GPU by size
    for i in range(n):
        if sizes[i] > S + 1e-9:
            raise ValueError(f"Model of size {sizes[i]} GB cannot fit into a single GPU of size {S} GB.")

    # Smooth objective helper
    def objective_and_stats(placement_dict):
        # Return (J, kvprs list, sumsR list, rem_mem list)
        sumsR = [0.0] * gpu_num
        rem = [S] * gpu_num
        for g in range(gpu_num):
            for m in placement_dict.get(g, []):
                rem[g] -= float(m.model_size)
                if getattr(m, 'slo', 0) != 0:
                    sumsR[g] += float(m.req_rate) / float(m.slo)
                else:
                    sumsR[g] += 1e9
        kvprs = [kvpr(sumsR[g], rem[g]) for g in range(gpu_num)]
        beta = 7.0
        J = 0.0
        for k in kvprs:
            if k == float('inf'):
                return float('inf'), kvprs, sumsR, rem
            J += pow(2.718281828459045, beta * k)
        return J, kvprs, sumsR, rem

    # Soft assignment via exponentiated-gradient descent on softmax potential
    def soft_assign(beta=7.0, iters=18, eta=0.15):
        # p[i][g] over allowed GPUs
        p = [[0.0] * gpu_num for _ in range(n)]
        # initialize uniformly over GPUs (by size feasibility)
        for i in range(n):
            avail = [g for g in range(gpu_num) if can_fit_gpu[i][g]]
            if not avail:
                raise ValueError(f"No GPU can fit model {i} by memory.")
            val = 1.0 / len(avail)
            for g in avail:
                p[i][g] = val

        eps = 1e-12
        for _ in range(max(1, iters)):
            # Compute current R_g, M_g
            Rg = [0.0] * gpu_num
            Mg = [0.0] * gpu_num
            for i in range(n):
                ri = rates[i]; si = sizes[i]
                for g in range(gpu_num):
                    if p[i][g] > 0:
                        Rg[g] += p[i][g] * ri
                        Mg[g] += p[i][g] * si
            rem = [S - Mg[g] for g in range(gpu_num)]
            # Compute per-GPU KVPR and exp(beta*kv)
            Jg = [0.0] * gpu_num
            for g in range(gpu_num):
                kv = kvpr(Rg[g], rem[g])
                if kv == float('inf'):
                    Jg[g] = float('inf')
                else:
                    Jg[g] = pow(2.718281828459045, beta * kv)

            # Gradients and multiplicative update
            for i in range(n):
                ri = rates[i]; si = sizes[i]
                # compute raw weights w_{i,g} ~ exp(-eta * grad)
                w = [0.0] * gpu_num
                w_sum = 0.0
                for g in range(gpu_num):
                    if not can_fit_gpu[i][g]:
                        w[g] = 0.0
                        continue
                    remg = rem[g]
                    # Avoid division by zero; if remg<=0, punish heavily
                    denom = (remg * remg) if remg > 1e-12 else (1e-12)
                    # d kvpr / d p_{i,g} = (ri*remg + Rg[g]*si) / remg^2
                    dkv = (ri * remg + Rg[g] * si) / denom
                    grad = beta * Jg[g] * dkv  # dJ/dp
                    # multiplicative weights: p *= exp(-eta * grad)
                    update = pow(2.718281828459045, -eta * grad)
                    w[g] = max(0.0, p[i][g] * update) if can_fit_gpu[i][g] else 0.0
                    w_sum += w[g]
                if w_sum <= eps:
                    # Reset to uniform over feasible GPUs to avoid collapse
                    feas = [g for g in range(gpu_num) if can_fit_gpu[i][g]]
                    val = 1.0 / len(feas)
                    for g in range(gpu_num):
                        p[i][g] = val if g in feas else 0.0
                else:
                    inv = 1.0 / w_sum
                    for g in range(gpu_num):
                        p[i][g] = w[g] * inv
        return p

    # Deterministic rounding guided by soft assignments and minimal J increase
    def round_from_p(p, beta=7.0):
        placement = {i: [] for i in range(gpu_num)}
        used_mem = [0.0] * gpu_num
        sum_R = [0.0] * gpu_num

        # Order items: more peaked assignment first, then by intrinsic pressure and size
        order = []
        for i in range(n):
            pmax = max(p[i]) if p[i] else 0.0
            intr = rates[i] / max(S - sizes[i], 1e-9)
            order.append((1.0 - pmax, -intr, -sizes[i], i))
        order.sort()

        def exp_sum(beta, kvprs):
            tot = 0.0
            for v in kvprs:
                if v == float('inf'):
                    return float('inf')
                tot += pow(2.718281828459045, beta * v)
            return tot

        for _, _, _, i in order:
            ri = rates[i]; si = sizes[i]
            # compute current per-GPU kvprs and objective
            kvprs_cur = [kvpr(sum_R[g], S - used_mem[g]) for g in range(gpu_num)]
            base = exp_sum(beta, kvprs_cur)

            best_g = None
            best_obj = float('inf')
            for g in range(gpu_num):
                if not can_fit_gpu[i][g]:
                    continue
                if used_mem[g] + si > S + 1e-9:
                    continue
                # simulate placement on g
                remg_new = S - (used_mem[g] + si)
                Rg_new = sum_R[g] + ri
                kv_g_new = kvpr(Rg_new, remg_new)
                # new objective = base - exp(beta*kv_cur) + exp(beta*kv_new)
                kv_g_cur = kvprs_cur[g]
                if kv_g_cur == float('inf'):
                    alt = float('inf')
                else:
                    alt = base - pow(2.718281828459045, beta * kv_g_cur) + pow(2.718281828459045, beta * kv_g_new)
                if alt < best_obj:
                    best_obj = alt
                    best_g = g

            if best_g is None:
                # If no GPU fits directly, select the GPU with most remaining memory and try a tiny repair
                g_try = max(range(gpu_num), key=lambda g: (S - used_mem[g]))
                moved = False
                # Try to free space on g_try by moving a single previously placed item to another GPU
                # capped small to remain fast
                candidates = list(placement[g_try])[:8]
                for m_to_move in candidates:
                    s_mv = float(m_to_move.model_size)
                    slo_mv = getattr(m_to_move, 'slo', 0)
                    r_mv = (float(m_to_move.req_rate) / float(slo_mv)) if slo_mv != 0 else 1e9
                    # look for a destination for m_to_move
                    for k in range(gpu_num):
                        if k == g_try:
                            continue
                        if used_mem[k] + s_mv <= S + 1e-9:
                            # simulate moving m_to_move to k, then place i on g_try
                            rem_src_new = S - (used_mem[g_try] - s_mv)
                            rem_k_new = S - (used_mem[k] + s_mv)
                            if rem_src_new <= 0 or rem_k_new <= 0:
                                continue
                            kvprs_tmp = []
                            for h in range(gpu_num):
                                if h == g_try:
                                    kvprs_tmp.append(kvpr(sum_R[h] - r_mv, rem_src_new))
                                elif h == k:
                                    kvprs_tmp.append(kvpr(sum_R[h] + r_mv, rem_k_new))
                                else:
                                    kvprs_tmp.append(kvpr(sum_R[h], S - used_mem[h]))
                            # Now check placing item i on g_try
                            rem_after_i = rem_src_new - si
                            if rem_after_i > 0:
                                kvprs_tmp[g_try] = kvpr((sum_R[g_try] - r_mv) + ri, rem_after_i)
                                # evaluate objective
                                J_tmp = 0.0
                                bad = False
                                for val in kvprs_tmp:
                                    if val == float('inf'):
                                        bad = True; break
                                    J_tmp += pow(2.718281828459045, beta * val)
                                if not bad:
                                    # Apply move+place
                                    placement[g_try].remove(m_to_move)
                                    placement[k].append(m_to_move)
                                    used_mem[g_try] -= s_mv
                                    used_mem[k] += s_mv
                                    sum_R[g_try] -= r_mv
                                    sum_R[k] += r_mv
                                    # place i
                                    placement[g_try].append(models[i])
                                    used_mem[g_try] += si
                                    sum_R[g_try] += ri
                                    moved = True
                                    break
                    if moved:
                        break
                if not moved:
                    # Fall back: put on GPU with max remaining memory ignoring objective if it fits post-repair not possible
                    g_fallback = None
                    rem_best = -1.0
                    for g in range(gpu_num):
                        remg = S - used_mem[g]
                        if si <= remg and remg > rem_best:
                            rem_best = remg
                            g_fallback = g
                    if g_fallback is None:
                        # As a safe final fallback: run a memory-first best-fit for all remaining items
                        return None
                    placement[g_fallback].append(models[i])
                    used_mem[g_fallback] += si
                    sum_R[g_fallback] += ri
            else:
                placement[best_g].append(models[i])
                used_mem[best_g] += si
                sum_R[best_g] += ri

        return placement

    # Greedy memory-only fallback minimizing local KVPR increase (used rarely)
    def greedy_fallback():
        placement = {i: [] for i in range(gpu_num)}
        rem_mem = [S] * gpu_num
        sum_R = [0.0] * gpu_num
        # Order by pressure, then size
        order = sorted(range(n), key=lambda i: (rates[i] / max(S - sizes[i], 1e-9), sizes[i]), reverse=True)
        for i in order:
            ri = rates[i]; si = sizes[i]
            best = None; best_val = float('inf')
            for g in range(gpu_num):
                if si <= rem_mem[g]:
                    rem_after = rem_mem[g] - si
                    if rem_after <= 0:
                        continue
                    val = kvpr(sum_R[g] + ri, rem_after)
                    if val < best_val:
                        best_val = val
                        best = g
            if best is None:
                raise ValueError(
                    f"Unable to place model of size {sizes[i]} GB on any GPU. Remaining per-GPU memory: {rem_mem}"
                )
            placement[best].append(models[i])
            rem_mem[best] -= si
            sum_R[best] += ri
        return placement

    # Local improvement using softmax potential and targeted moves/swaps
    def refine(placement, move_budget=20, swap_budget=10, beta=7.0):
        buckets = {g: list(placement.get(g, [])) for g in range(gpu_num)}
        used = [0.0] * gpu_num
        sums = [0.0] * gpu_num
        for g in range(gpu_num):
            for m in buckets[g]:
                used[g] += float(m.model_size)
                slo = getattr(m, 'slo', 0)
                if slo != 0:
                    sums[g] += float(m.req_rate) / float(slo)
                else:
                    sums[g] += 1e9

        def kvprs():
            return [kvpr(sums[g], S - used[g]) for g in range(gpu_num)]

        def soft_obj(kvlist):
            tot = 0.0
            for v in kvlist:
                if v == float('inf'):
                    return float('inf')
                tot += pow(2.718281828459045, beta * v)
            return tot

        it_moves = move_budget
        it_swaps = swap_budget
        while it_moves > 0 or it_swaps > 0:
            kv = kvprs()
            base = soft_obj(kv)
            if base == float('inf'):
                # If inf due to 0 rem with positive load, no feasible improvement under memory; break
                break
            improved = False

            # Try best improving single move
            best = None  # (delta, src, dst, mdl)
            src_idx = max(range(gpu_num), key=lambda g: kv[g])
            for mdl in list(buckets[src_idx]):
                s = float(mdl.model_size)
                slo = getattr(mdl, 'slo', 0)
                r = (float(mdl.req_rate) / float(slo)) if slo != 0 else 1e9
                # simulate removal from src
                rem_src = S - (used[src_idx] - s)
                if rem_src <= 0 and (sums[src_idx] - r) > 1e-12:
                    continue
                kv_src_new = kvpr(sums[src_idx] - r, rem_src) if rem_src > 0 else 0.0
                # Try destinations
                for dst in range(gpu_num):
                    if dst == src_idx:
                        continue
                    if used[dst] + s > S + 1e-9:
                        continue
                    rem_dst = S - (used[dst] + s)
                    if rem_dst <= 0 and (sums[dst] + r) > 1e-12:
                        continue
                    kv_dst_new = kvpr(sums[dst] + r, rem_dst) if rem_dst > 0 else 0.0
                    # compute new soft objective quickly
                    tmp_max = 0.0
                    total = 0.0
                    for g in range(gpu_num):
                        if g == src_idx:
                            v = kv_src_new
                        elif g == dst:
                            v = kv_dst_new
                        else:
                            v = kv[g]
                        if v == float('inf'):
                            total = float('inf'); break
                        total += pow(2.718281828459045, beta * v)
                        if v > tmp_max:
                            tmp_max = v
                    if total < base - 1e-12:
                        delta = base - total
                        if best is None or delta > best[0]:
                            best = (delta, src_idx, dst, mdl)
            if best is not None and it_moves > 0:
                _, sidx, didx, mdl = best
                s = float(mdl.model_size)
                slo = getattr(mdl, 'slo', 0)
                r = (float(mdl.req_rate) / float(slo)) if slo != 0 else 1e9
                buckets[sidx].remove(mdl); buckets[didx].append(mdl)
                used[sidx] -= s; used[didx] += s
                sums[sidx] -= r; sums[didx] += r
                it_moves -= 1
                improved = True
            else:
                # Try limited best swap
                best_swap = None  # (delta, src, dst, a, b)
                src = src_idx
                cap_a = min(10, len(buckets[src]))
                for a in list(buckets[src])[:cap_a]:
                    aS = float(a.model_size)
                    aslo = getattr(a, 'slo', 0)
                    aR = (float(a.req_rate) / float(aslo)) if aslo != 0 else 1e9
                    for dst in range(gpu_num):
                        if dst == src or not buckets[dst]:
                            continue
                        cap_b = min(10, len(buckets[dst]))
                        for b in list(buckets[dst])[:cap_b]:
                            bS = float(b.model_size)
                            bslo = getattr(b, 'slo', 0)
                            bR = (float(b.req_rate) / float(bslo)) if bslo != 0 else 1e9
                            # New mem after swap
                            mem_src_new = used[src] - aS + bS
                            mem_dst_new = used[dst] - bS + aS
                            if mem_src_new > S + 1e-9 or mem_dst_new > S + 1e-9:
                                continue
                            rem_src = S - mem_src_new
                            rem_dst = S - mem_dst_new
                            kv_src_new = kvpr(sums[src] - aR + bR, rem_src) if rem_src > 0 else (0.0 if (sums[src] - aR + bR) <= 1e-12 else float('inf'))
                            kv_dst_new = kvpr(sums[dst] - bR + aR, rem_dst) if rem_dst > 0 else (0.0 if (sums[dst] - bR + aR) <= 1e-12 else float('inf'))
                            total = 0.0
                            for g in range(gpu_num):
                                if g == src:
                                    v = kv_src_new
                                elif g == dst:
                                    v = kv_dst_new
                                else:
                                    v = kv[g]
                                if v == float('inf'):
                                    total = float('inf'); break
                                total += pow(2.718281828459045, beta * v)
                            if total < base - 1e-12:
                                delta = base - total
                                if best_swap is None or delta > best_swap[0]:
                                    best_swap = (delta, src, dst, a, b)
                if best_swap is not None and it_swaps > 0:
                    _, src, dst, a, b = best_swap
                    buckets[src].remove(a); buckets[src].append(b)
                    buckets[dst].remove(b); buckets[dst].append(a)
                    aS = float(a.model_size); bS = float(b.model_size)
                    aslo = getattr(a, 'slo', 0); bslo = getattr(b, 'slo', 0)
                    aR = (float(a.req_rate) / float(aslo)) if aslo != 0 else 1e9
                    bR = (float(b.req_rate) / float(bslo)) if bslo != 0 else 1e9
                    used[src] = used[src] - aS + bS
                    used[dst] = used[dst] - bS + aS
                    sums[src] = sums[src] - aR + bR
                    sums[dst] = sums[dst] - bR + aR
                    it_swaps -= 1
                    improved = True
            if not improved:
                break
        return buckets

    # Main pipeline
    # 1) soft assignment
    p = soft_assign(beta=7.0, iters=18, eta=0.12)
    # 2) rounding
    placement = round_from_p(p, beta=7.0)
    if placement is None:
        # Memory-repair fallback
        placement = greedy_fallback()

    # 3) local refinement
    refined = refine(placement, move_budget=20, swap_budget=10, beta=7.0)

    # Ensure memory safety
    for gid in range(gpu_num):
        used = sum(float(m.model_size) for m in refined.get(gid, []))
        if used - S > 1e-6:
            # If refinement broke memory (shouldn't), return pre-refined
            return placement

    return refined

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