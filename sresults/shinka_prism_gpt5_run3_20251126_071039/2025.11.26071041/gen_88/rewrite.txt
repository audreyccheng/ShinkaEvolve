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
    # Trivial guard
    placement_empty = {i: [] for i in range(gpu_num)}
    if gpu_num <= 0 or not models:
        return placement_empty

    S = GPU_MEM_SIZE

    # Basic KVPR helper
    def kvpr(R, rem_mem):
        if rem_mem <= 0:
            return float('inf')
        return R / rem_mem

    # Greedy fallback minimizing local max-KVPR increase (robust; treats slo==0 as memory-only)
    def greedy_minimax(models_all):
        ordered = sorted(
            models_all,
            key=lambda m: ((m.req_rate / m.slo) if getattr(m, "slo", 0) != 0 else 0.0, m.model_size),
            reverse=True
        )
        place = {i: [] for i in range(gpu_num)}
        used = [0.0] * gpu_num
        sumR = [0.0] * gpu_num
        for m in ordered:
            dR = (m.req_rate / m.slo) if getattr(m, "slo", 0) != 0 else 0.0
            sz = float(m.model_size)
            best, best_val = None, float('inf')
            for g in range(gpu_num):
                if used[g] + sz <= S:
                    rem = S - (used[g] + sz)
                    if rem <= 0 and dR > 0:
                        continue
                    val = kvpr(sumR[g] + dR, rem)
                    if val < best_val:
                        best_val, best = val, g
            if best is None:
                # No feasible memory capacity
                raise ValueError(f"Unable to place model of size {m.model_size} GB on any GPU.")
            place[best].append(m)
            used[best] += sz
            sumR[best] += dR
        return place

    # Phase 1: split models into memory-only (slo==0) and active (slo>0)
    mem_only = [m for m in models if getattr(m, "slo", 0) == 0]
    active = [m for m in models if getattr(m, "slo", 0) != 0]

    # Phase 2: pre-place memory-only models with largest-first, onto GPUs with most free memory
    def preplace_memory_only(mem_models):
        base = {i: [] for i in range(gpu_num)}
        used = [0.0] * gpu_num
        if not mem_models:
            return base, used
        mem_models = sorted(mem_models, key=lambda m: float(m.model_size), reverse=True)
        for m in mem_models:
            sz = float(m.model_size)
            # Choose GPU with most remaining memory (tie-break: fewest models)
            best, best_key = None, None
            for g in range(gpu_num):
                rem = S - used[g]
                if sz <= rem:
                    key = (rem, -len(base[g]))
                    if best is None or key > best_key:
                        best, best_key = g, key
            if best is None:
                raise ValueError("Memory-only model cannot fit on any GPU.")
            base[best].append(m)
            used[best] += sz
        return base, used

    base_assign, base_used = preplace_memory_only(mem_only)

    # Phase 3: prepare items for active models
    items = []
    tot_R, tot_size = 0.0, 0.0
    for m in active:
        slo = float(m.slo)
        dR = float(m.req_rate) / slo if slo != 0 else 0.0  # active implies slo>0
        sz = float(m.model_size)
        items.append({'obj': m, 'dR': dR, 'size': sz})
        tot_R += dR
        tot_size += sz

    # If no active models, return the memory-balanced base assignment
    if not items:
        return base_assign

    # Phase 4: lower bounds on T (KVPR upper bound)
    def compute_lower_bound():
        # Per-item bound: T >= dR / (S - s)
        lb_item = 0.0
        infeasible_single = False
        for it in items:
            dR, s = it['dR'], it['size']
            denom = S - s
            if denom <= 0:
                if dR > 0:
                    infeasible_single = True
                continue
            if dR > 0:
                lb_item = max(lb_item, dR / denom)

        # Global bound with pre-placed memory-only accounted
        free_total = gpu_num * S - sum(base_used)
        denom_global = free_total - tot_size
        if tot_R > 0 and denom_global <= 0:
            return float('inf'), True, infeasible_single
        lb_global = 0.0 if denom_global <= 0 or tot_R <= 0 else (tot_R / denom_global)

        # Pair bound: for pairs that cannot co-reside (s_i + s_j > S)
        lb_pair = 0.0
        P = min(len(items), 160)
        by_size = sorted(items, key=lambda x: x['size'], reverse=True)[:P]
        for i in range(len(by_size)):
            si, ri = by_size[i]['size'], by_size[i]['dR']
            for j in range(i + 1, len(by_size)):
                sj, rj = by_size[j]['size'], by_size[j]['dR']
                if si + sj > S:
                    denom = 2 * S - (si + sj)
                    if denom > 0:
                        lb_pair = max(lb_pair, (ri + rj) / denom)

        # k-prefix bound: k in [1..min(gpu_num,6)]
        lb_k = 0.0
        sorted_by_size = sorted(items, key=lambda x: x['size'], reverse=True)
        pref_s, pref_r = 0.0, 0.0
        prefix = []
        for it in sorted_by_size:
            pref_s += it['size']; pref_r += it['dR']
            prefix.append((pref_s, pref_r))
        for k in range(1, min(gpu_num, 6) + 1):
            threshold = (k - 1) * S
            idx = next((t for t, (ps, _) in enumerate(prefix) if ps > threshold), -1)
            if idx >= 0:
                ps, pr = prefix[idx]
                denom = k * S - ps
                if denom > 0 and pr > 0:
                    lb_k = max(lb_k, pr / denom)

        # Lightweight triplet bound over top-8 sizes
        lb_trip = 0.0
        top = by_size[:min(8, len(by_size))]
        for i in range(len(top)):
            si, ri = top[i]['size'], top[i]['dR']
            for j in range(i + 1, len(top)):
                sj, rj = top[j]['size'], top[j]['dR']
                for k in range(j + 1, len(top)):
                    sk, rk = top[k]['size'], top[k]['dR']
                    ssum = si + sj + sk
                    if ssum > 2 * S:
                        denom = 3 * S - ssum
                        if denom > 0:
                            lb_trip = max(lb_trip, (ri + rj + rk) / denom)

        lower = max(0.0, lb_item, lb_global, lb_pair, lb_k, lb_trip)
        return lower, False, infeasible_single

    lower, infeasible_global, infeasible_single = compute_lower_bound()
    if infeasible_single or infeasible_global or lower == float('inf'):
        # Safety net
        return greedy_minimax(models)

    # KVPR evaluator for any placement dict
    def eval_max_kvpr(placement):
        max_v = 0.0
        for g in range(gpu_num):
            bucket = placement.get(g, [])
            used, R = 0.0, 0.0
            for m in bucket:
                used += float(m.model_size)
                if getattr(m, "slo", 0) != 0:
                    R += float(m.req_rate) / float(m.slo)
            max_v = max(max_v, kvpr(R, S - used))
        return max_v

    # Phase 5: feasibility oracle with balanced-slack packing and residual retuning
    import random
    rng = random.Random(len(items) * 1009 + gpu_num * 9173)

    def assign_for_T(T, micro_restarts=3):
        if T < 0:
            return None
        # Build base state
        base_sumR = [0.0] * gpu_num  # memory-only has zero dR
        # Enriched items: w = dR + T*s
        enriched0 = [[(it['dR'] + T * it['size']), it['dR'], it['size'], it['obj']] for it in items]
        enriched0.sort(key=lambda x: x[0], reverse=True)

        best, best_val = None, float('inf')
        eps = 1e-9

        for r in range(max(1, micro_restarts)):
            # Initialize per-GPU states including pre-placed memory
            assign = {i: list(base_assign.get(i, [])) for i in range(gpu_num)}
            used = list(base_used)
            sumR = list(base_sumR)
            # Jittered slack to diversify ties
            jitter = 1e-6 * (T * S if T > 0 else 1.0)
            K = [T * S - (sumR[g] + T * used[g]) + jitter * g for g in range(gpu_num)]

            enriched = list(enriched0)

            # Placement routine with mid-run T retuning
            placed = 0
            cut = int(0.4 * len(enriched))
            def place_seq(seq, Tcur):
                nonlocal assign, used, sumR, K
                for w, dR, sz, m in seq:
                    options = []
                    for g in range(gpu_num):
                        if used[g] + sz <= S + eps and K[g] >= w - eps:
                            K_after = K[g] - w
                            rem = S - (used[g] + sz)
                            if rem <= 0 and dR > 0:
                                continue
                            kv_new = kvpr(sumR[g] + dR, max(rem, eps))
                            options.append((K_after, kv_new, g))
                    if not options:
                        return False, Tcur
                    options.sort(key=lambda t: (t[0], t[1]))  # tightest slack, then lower resulting kvpr
                    # Tiny deterministic tie-break
                    gsel = options[0][2]
                    assign[gsel].append(m)
                    used[gsel] += sz
                    sumR[gsel] += dR
                    K[gsel] -= w
                return True, Tcur

            # First chunk
            ok, Tcur = True, T
            if cut > 0:
                ok, Tcur = place_seq(enriched[:cut], Tcur)
                placed = cut if ok else 0
            if ok:
                # Residual-aware T retune
                rem_items = enriched[placed:]
                if rem_items:
                    rem_R = sum(x[1] for x in rem_items)
                    rem_size = sum(x[2] for x in rem_items)
                    free_sum = sum(max(0.0, S - used[g]) for g in range(gpu_num))
                    denom = free_sum - rem_size
                    if rem_R > 0 and denom > 0:
                        T_resid = rem_R / denom
                        if T_resid > Tcur * 1.02:
                            # Update K to reflect increased T
                            delta = T_resid - Tcur
                            for g in range(gpu_num):
                                K[g] += delta * (S - used[g])
                            Tcur = T_resid
                    # Rebuild remaining weights with Tcur and place them
                    rem_enriched = [[(x[1] + Tcur * x[2]), x[1], x[2], x[3]] for x in rem_items]
                    rem_enriched.sort(key=lambda x: x[0], reverse=True)
                    ok, Tcur = place_seq(rem_enriched, Tcur)

            if not ok:
                continue

            # Validate feasibility for Tcur
            valid = True
            for g in range(gpu_num):
                if used[g] - S > 1e-6:
                    valid = False; break
                rem = S - used[g]
                if rem <= 0 and sumR[g] > 0:
                    valid = False; break
                if Tcur > 0 and kvpr(sumR[g], max(rem, eps)) - Tcur > 1e-6:
                    valid = False; break
            if not valid:
                continue

            # Keep best by measured max KVPR
            val = 0.0
            for g in range(gpu_num):
                val = max(val, kvpr(sumR[g], S - used[g]))
            if val < best_val:
                best_val = val
                best = assign

        return best

    # Phase 6: search for T with true bracketing + polishing
    def first_feasible_T(start_T):
        T = max(0.0, start_T)
        # Jump to lower bound
        cand = assign_for_T(T)
        if cand is not None:
            return T, cand
        growth = 1.09
        for _ in range(40):
            T = (T * growth) if T > 0 else 1e-6
            cand = assign_for_T(T)
            if cand is not None:
                return T, cand
        return None, None

    T_hi, best_placement = first_feasible_T(lower)
    if best_placement is None:
        # Final safety net
        return greedy_minimax(models)

    # Binary search on feasibility between lower and T_hi
    lo, hi = max(0.0, lower), T_hi
    best_T, best_val = hi, eval_max_kvpr(best_placement)
    for _ in range(8):
        mid = (lo + hi) / 2.0
        cand = assign_for_T(mid)
        if cand is not None:
            hi = mid
            v = eval_max_kvpr(cand)
            if v < best_val:
                best_val = v
                best_placement = cand
                best_T = mid
        else:
            lo = mid

    # Compact polish near T*
    for mul in (0.99, 1.0, 1.01):
        T_probe = max(0.0, best_T * mul)
        cand = assign_for_T(T_probe)
        if cand is None:
            continue
        v = eval_max_kvpr(cand)
        if v < best_val:
            best_val = v
            best_placement = cand

    # Phase 7: small local refinement (moves only) to reduce max KVPR
    def local_refine(placement, iters=20):
        buckets = {g: list(placement.get(g, [])) for g in range(gpu_num)}
        used = [0.0] * gpu_num
        sumR = [0.0] * gpu_num
        for g in range(gpu_num):
            for m in buckets[g]:
                used[g] += float(m.model_size)
                if getattr(m, "slo", 0) != 0:
                    sumR[g] += float(m.req_rate) / float(m.slo)

        for _ in range(iters):
            kvprs = [kvpr(sumR[g], S - used[g]) for g in range(gpu_num)]
            cur_max = max(kvprs) if kvprs else 0.0
            src = max(range(gpu_num), key=lambda g: kvprs[g]) if kvprs else 0
            improved = False
            best_move = None  # (resulting_max, src, dst, mdl)
            for mdl in list(buckets[src]):
                dR = float(mdl.req_rate) / float(mdl.slo) if getattr(mdl, "slo", 0) != 0 else 0.0
                sz = float(mdl.model_size)
                R_src_new = sumR[src] - dR
                mem_src_new = used[src] - sz
                rem_src_new = S - mem_src_new
                if rem_src_new <= 0 and R_src_new > 0:
                    continue
                kv_src_new = kvpr(R_src_new, max(rem_src_new, 1e-9))
                for dst in range(gpu_num):
                    if dst == src: continue
                    if used[dst] + sz > S: continue
                    rem_dst_new = S - (used[dst] + sz)
                    if rem_dst_new <= 0 and dR > 0: continue
                    R_dst_new = sumR[dst] + dR
                    kv_dst_new = kvpr(R_dst_new, max(rem_dst_new, 1e-9))
                    resulting = max(kv_dst_new, kv_src_new)
                    for g in range(gpu_num):
                        if g == src or g == dst: continue
                        resulting = max(resulting, kvprs[g])
                    if resulting + 1e-12 < cur_max:
                        if best_move is None or resulting < best_move[0]:
                            best_move = (resulting, src, dst, mdl)
            if best_move is None:
                break
            _, s, d, m = best_move
            buckets[s].remove(m); buckets[d].append(m)
            dR = float(m.req_rate) / float(m.slo) if getattr(m, "slo", 0) != 0 else 0.0
            sz = float(m.model_size)
            sumR[s] -= dR; sumR[d] += dR
            used[s] -= sz; used[d] += sz
            improved = True
            if not improved:
                break
        return buckets

    refined = local_refine(best_placement, iters=16)

    # Compare with a greedy baseline candidate (built over base assignment)
    def greedy_over_base():
        # Place active models greedily while preserving base memory-only placement
        place = {g: list(base_assign.get(g, [])) for g in range(gpu_num)}
        used = [sum(float(m.model_size) for m in place[g]) for g in range(gpu_num)]
        sumR = [0.0] * gpu_num
        for g in range(gpu_num):
            for m in place[g]:
                if getattr(m, "slo", 0) != 0:
                    sumR[g] += float(m.req_rate) / float(m.slo)
        # Order by pressure dR/(S - s)
        def weight(m):
            s = float(m.model_size)
            dR = float(m.req_rate) / float(m.slo)
            denom = max(S - s, 1e-9)
            return dR / denom
        for m in sorted(active, key=weight, reverse=True):
            dR = float(m.req_rate) / float(m.slo); sz = float(m.model_size)
            best, best_val = None, float('inf')
            for g in range(gpu_num):
                if used[g] + sz <= S:
                    rem = S - (used[g] + sz)
                    if rem <= 0 and dR > 0: continue
                    val = kvpr(sumR[g] + dR, max(rem, 1e-9))
                    if val < best_val: best_val, best = val, g
            if best is None:
                return None
            place[best].append(m)
            used[best] += sz; sumR[best] += dR
        return place

    greedy_cand = greedy_over_base()
    best_final = refined
    best_score = eval_max_kvpr(best_final)
    if greedy_cand is not None:
        v = eval_max_kvpr(greedy_cand)
        if v < best_score:
            best_final, best_score = greedy_cand, v

    # Memory safety
    for g in range(gpu_num):
        if sum(float(m.model_size) for m in best_final.get(g, [])) - S > 1e-6:
            return refined

    return best_final

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