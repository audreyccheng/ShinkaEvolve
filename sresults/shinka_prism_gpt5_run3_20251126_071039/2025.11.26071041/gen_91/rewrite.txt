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
    # Trivial cases
    placement_empty = {i: [] for i in range(gpu_num)}
    if gpu_num <= 0 or not models:
        return placement_empty

    S = GPU_MEM_SIZE

    # Safe KVPR
    def kvpr(R, rem_mem):
        if rem_mem <= 0:
            return float('inf')
        return R / rem_mem

    # Baseline greedy: place to minimize resulting max KVPR increase; robust to slo==0
    def greedy_kvpr(models_all):
        order = sorted(
            models_all,
            key=lambda m: ((float(m.req_rate) / float(m.slo)) if getattr(m, "slo", 0) != 0 else 0.0, float(m.model_size)),
            reverse=True
        )
        placement = {i: [] for i in range(gpu_num)}
        used = [0.0] * gpu_num
        Rsum = [0.0] * gpu_num
        for m in order:
            dR = (float(m.req_rate) / float(m.slo)) if getattr(m, "slo", 0) != 0 else 0.0
            size = float(m.model_size)
            best = None
            best_val = float('inf')
            for g in range(gpu_num):
                if used[g] + size <= S:
                    rem = S - (used[g] + size)
                    if rem <= 0:
                        continue
                    val = kvpr(Rsum[g] + dR, rem)
                    # minimize resulting max across GPUs
                    worst = val
                    for j in range(gpu_num):
                        if j == g:
                            continue
                        t = kvpr(Rsum[j], S - used[j])
                        if t > worst:
                            worst = t
                    if worst < best_val:
                        best_val = worst
                        best = g
            if best is None:
                # If cannot place, raise, but upstream code always guards memory
                raise ValueError("Unable to place model due to memory capacity.")
            placement[best].append(m)
            used[best] += size
            Rsum[best] += dR
        return placement

    # Phase 1: pre-place memory-only models (slo==0), largest-first onto GPUs with most free mem
    def preplace_memory_only(all_models):
        base = {i: [] for i in range(gpu_num)}
        used0 = [0.0] * gpu_num
        mem_only = [m for m in all_models if getattr(m, "slo", 0) == 0]
        if not mem_only:
            return base, used0
        mem_only.sort(key=lambda m: float(m.model_size), reverse=True)
        for m in mem_only:
            sz = float(m.model_size)
            if sz > S + 1e-9:
                # Cannot fit anywhere; fall back to global greedy
                return None, None
            # GPU with most remaining memory; tie by fewest models
            candidates = [g for g in range(gpu_num) if used0[g] + sz <= S + 1e-9]
            if not candidates:
                return None, None
            candidates.sort(key=lambda g: (-(S - used0[g]), len(base[g])))
            gid = candidates[0]
            base[gid].append(m)
            used0[gid] += sz
        return base, used0

    base_assign, base_used = preplace_memory_only(models)
    if base_assign is None:
        # Fallback if preplacement fails
        return greedy_kvpr(models)

    # Phase 2: collect demanders (slo>0) and per-GPU residual capacities
    demanders = [m for m in models if getattr(m, "slo", 0) != 0]
    if not demanders:
        return base_assign

    items = []
    for m in demanders:
        slo = float(m.slo)
        dR = float(m.req_rate) / slo if slo != 0 else 0.0
        items.append({'obj': m, 'dR': dR, 'size': float(m.model_size)})

    Sg = [S - u for u in base_used]
    Sg_total = sum(Sg)
    Sg_max = max(Sg) if Sg else 0.0

    # If any demander cannot fit into any GPU memory-wise, fallback to greedy
    if any(it['size'] > Sg_max + 1e-9 for it in items):
        return greedy_kvpr(models)

    # Lower bounds for T using residual capacities
    def lower_bound_T():
        total_R = sum(it['dR'] for it in items)
        total_size = sum(it['size'] for it in items)

        # per-item bound
        lb1 = 0.0
        infeasible_single = False
        for it in items:
            denom = Sg_max - it['size']
            if denom <= 0:
                if it['dR'] > 0:
                    infeasible_single = True
                continue
            if it['dR'] > 0:
                lb1 = max(lb1, it['dR'] / denom)

        # global bound using ΣSg
        denom2 = Sg_total - total_size
        if denom2 <= 0 and total_R > 0:
            return float('inf'), True
        lb2 = 0.0 if denom2 <= 0 or total_R <= 0 else (total_R / denom2)

        # pair bound over top-by-size
        lb_pair = 0.0
        P = min(len(items), 120)
        by_size = sorted(items, key=lambda x: x['size'], reverse=True)[:P]
        for i in range(P):
            si, ri = by_size[i]['size'], by_size[i]['dR']
            for j in range(i + 1, P):
                sj, rj = by_size[j]['size'], by_size[j]['dR']
                if si + sj > Sg_max:
                    denom = 2 * Sg_max - (si + sj)
                    if denom > 0:
                        lb_pair = max(lb_pair, (ri + rj) / denom)

        # k-prefix bound for k up to 6 using Sg_max capacity proxy
        lb_k = 0.0
        sorted_by_s = sorted(items, key=lambda x: x['size'], reverse=True)
        prefix_size, prefix_rate = [], []
        cs = cr = 0.0
        for it in sorted_by_s:
            cs += it['size']; cr += it['dR']
            prefix_size.append(cs); prefix_rate.append(cr)
        for k in range(1, min(gpu_num, 6) + 1):
            threshold = (k - 1) * Sg_max
            idx = -1
            for t in range(len(prefix_size)):
                if prefix_size[t] > threshold:
                    idx = t
                    break
            if idx >= 0:
                denom = k * Sg_max - prefix_size[idx]
                if denom > 0 and prefix_rate[idx] > 0:
                    lb_k = max(lb_k, prefix_rate[idx] / denom)

        # lightweight triplet bound
        lb_trip = 0.0
        topK = by_size[:min(8, len(by_size))]
        for i in range(len(topK)):
            for j in range(i + 1, len(topK)):
                for k in range(j + 1, len(topK)):
                    ssum = topK[i]['size'] + topK[j]['size'] + topK[k]['size']
                    if ssum > 2 * Sg_max:
                        denom = 3 * Sg_max - ssum
                        if denom > 0:
                            rsum = topK[i]['dR'] + topK[j]['dR'] + topK[k]['dR']
                            lb_trip = max(lb_trip, rsum / denom)

        lower = max(0.0, lb1, lb2, lb_pair, lb_k, lb_trip)
        if infeasible_single:
            lower = max(lower, float('inf'))
        return lower, False

    lower, infeasible_global = lower_bound_T()
    if infeasible_global or lower == float('inf'):
        return greedy_kvpr(models)

    # Helper: evaluate KVPR vector and max
    def eval_kvpr_vector(placement):
        kv = []
        for g in range(gpu_num):
            used = sum(float(m.model_size) for m in placement.get(g, []))
            R = sum((float(m.req_rate) / float(m.slo)) for m in placement.get(g, []) if getattr(m, "slo", 0) != 0)
            kv.append(kvpr(R, S - used))
        return kv, (max(kv) if kv else 0.0)

    # Lexicographic compare on descending KVPR vectors
    def lex_better(vecA, vecB):
        a = sorted(vecA, reverse=True)
        b = sorted(vecB, reverse=True)
        return a < b  # lexicographic: smaller is better

    # Phase 3: feasibility packer for a given T on demanders only (respecting base capacities)
    from math import isfinite

    def pack_for_T(T, restarts=4, mid_retune=True):
        # Precompute base residual per GPU
        Sres = list(Sg)
        Sres_total = sum(Sres)
        if T < 0 or not isfinite(T):
            return None

        # Enriched items with initial weights; function to rebuild with T'
        def enrich_with_T(Tcur, sub=None):
            seq = items if sub is None else sub
            enriched = []
            for it in seq:
                w = it['dR'] + Tcur * it['size']
                enriched.append([w, it['dR'], it['size'], it['obj']])
            enriched.sort(key=lambda x: x[0], reverse=True)
            return enriched

        # Single attempt
        def attempt(seed_eps):
            enriched = enrich_with_T(T)
            assign = {i: list(base_assign.get(i, [])) for i in range(gpu_num)}
            used_add = [0.0] * gpu_num  # only demanders' memory
            Rsum = [0.0] * gpu_num      # only demanders' rates
            K = [T * Sres[g] + seed_eps * (g + 1) for g in range(gpu_num)]

            # Place a sequence with current T and K
            def place_seq(seq):
                nonlocal K
                for w, dR, sz, m in seq:
                    options = []
                    for g in range(gpu_num):
                        if used_add[g] + sz <= Sres[g] + 1e-9 and K[g] >= w - 1e-12:
                            K_after = K[g] - w
                            rem_after = S - (base_used[g] + used_add[g] + sz)
                            if rem_after <= 0:
                                continue
                            kv_new = kvpr(Rsum[g] + dR, rem_after)
                            mem_slack = Sres[g] - (used_add[g] + sz)
                            options.append((K_after, kv_new, -mem_slack, g))
                    if not options:
                        return False
                    # choose minimal K_after; tie by kv_new; then by larger mem slack
                    options.sort(key=lambda t: (t[0], t[1], t[2]))
                    _, _, _, gid = options[0]
                    assign[gid].append(m)
                    used_add[gid] += sz
                    Rsum[gid] += dR
                    K[gid] -= (dR + T * sz)
                return True

            # First chunk
            seq_all = list(enriched)
            cut = int(0.4 * len(seq_all))
            if cut > 0:
                if not place_seq(seq_all[:cut]):
                    return None
                # Optional mid-retune of T using residual bound
                if mid_retune:
                    rem_items = seq_all[cut:]
                    if rem_items:
                        rem_R = sum(x[1] for x in rem_items)
                        rem_sz = sum(x[2] for x in rem_items)
                        free_sum = sum((Sres[g] - used_add[g]) for g in range(gpu_num))
                        denom = free_sum - rem_sz
                        if denom > 0 and rem_R > 0:
                            T2 = rem_R / denom
                            if T2 > T * 1.02:
                                delta = T2 - T
                                # Update K for delta: K += delta*(Sres - used_add)
                                for g in range(gpu_num):
                                    K[g] += delta * (Sres[g] - used_add[g])
                                # Rebuild remaining with T2 and place
                                rem_enriched = enrich_with_T(T2, sub=[{'dR': x[1], 'size': x[2], 'obj': x[3]} for x in rem_items])
                                if not place_seq(rem_enriched):
                                    return None
                            else:
                                if not place_seq(rem_items):
                                    return None
                        else:
                            if not place_seq(rem_items):
                                return None
            else:
                if not place_seq(seq_all):
                    return None

            # Feasibility satisfied; return assignment
            return assign

        best = None
        best_vec = None
        # small deterministic jitter for diversity
        import random
        rng = random.Random(len(items) * 7919 + gpu_num * 1237)
        base_eps = 1e-9 * max(1.0, T * Sg_total)
        for r in range(max(1, restarts)):
            eps = base_eps * (1.0 + 0.1 * r + 0.01 * rng.random())
            cand = attempt(eps)
            if cand is not None:
                vec, _ = eval_kvpr_vector(cand)
                if best is None or lex_better(vec, best_vec):
                    best = cand
                    best_vec = vec
        return best

    # Find first feasible T via multiplicative sweep from lower bound
    def first_feasible_T(start):
        T = max(start, 1e-9)
        growth = 1.08
        for _ in range(48):
            cand = pack_for_T(T, restarts=3)
            if cand is not None:
                return T, cand
            T *= growth
        return None, None

    T_feas, place_T = first_feasible_T(lower)
    if place_T is None:
        # Safety net
        return greedy_kvpr(models)

    # Binary search to tighten T (feasibility oracle only)
    lo, hi = max(0.0, lower), T_feas
    best_place = place_T
    best_vec, best_max = eval_kvpr_vector(best_place)
    for _ in range(8):
        mid = (lo + hi) / 2.0
        cand = pack_for_T(mid, restarts=3)
        if cand is not None:
            hi = mid
            vec, _ = eval_kvpr_vector(cand)
            if lex_better(vec, best_vec):
                best_vec = vec
                best_place = cand
        else:
            lo = mid

    # Probe small neighborhood around T*
    T_star = hi
    probe = [max(lower, T_star * mul) for mul in (0.99, 1.0, 1.01)]
    for Ttry in probe:
        cand = pack_for_T(Ttry, restarts=4)
        if cand is not None:
            vec, _ = eval_kvpr_vector(cand)
            if lex_better(vec, best_vec):
                best_vec = vec
                best_place = cand

    # Cheap local pre-refinement: try moving one model from worst GPU
    def quick_move_refine(placement, attempts=2):
        # Build per-GPU stats
        buckets = {g: list(placement.get(g, [])) for g in range(gpu_num)}
        for _ in range(attempts):
            kvs, _ = eval_kvpr_vector(buckets)
            worst = max(range(gpu_num), key=lambda g: kvs[g])
            cur_max = kvs[worst]
            # Consider items on worst GPU sorted by size*dR surrogate
            cand_models = list(buckets[worst])
            if not cand_models:
                break
            def weight(m):
                dR = (float(m.req_rate) / float(m.slo)) if getattr(m, "slo", 0) != 0 else 0.0
                return dR + float(m.model_size)
            cand_models.sort(key=weight, reverse=True)
            best = None  # (result_max, src, dst, model)
            # Precompute current used and R per GPU
            used = [sum(float(m.model_size) for m in buckets[g]) for g in range(gpu_num)]
            Rs = [sum((float(m.req_rate) / float(m.slo)) for m in buckets[g] if getattr(m, "slo", 0) != 0) for g in range(gpu_num)]
            for m in cand_models[:min(2, len(cand_models))]:
                s = float(m.model_size)
                dR = (float(m.req_rate) / float(m.slo)) if getattr(m, "slo", 0) != 0 else 0.0
                for dst in range(gpu_num):
                    if dst == worst:
                        continue
                    if used[dst] + s > S + 1e-9:
                        continue
                    # simulate move
                    used_w_new = used[worst] - s
                    used_d_new = used[dst] + s
                    R_w_new = Rs[worst] - dR
                    R_d_new = Rs[dst] + dR
                    kv_w = kvpr(R_w_new, S - used_w_new)
                    kv_d = kvpr(R_d_new, S - used_d_new)
                    worst_new = max(kv_w, kv_d)
                    for g in range(gpu_num):
                        if g == worst or g == dst:
                            continue
                        val = kvpr(Rs[g], S - used[g])
                        if val > worst_new:
                            worst_new = val
                    if worst_new + 1e-12 < cur_max:
                        if best is None or worst_new < best[0]:
                            best = (worst_new, worst, dst, m)
            if best is None:
                break
            _, src, dst, m = best
            buckets[src].remove(m)
            buckets[dst].append(m)
        return buckets

    refined = quick_move_refine(best_place, attempts=2)

    # Also compute a refined greedy baseline for safety comparison
    greedy_base = greedy_kvpr(models)
    # Select the better result by lexicographic KVPR vector
    vec_ref, max_ref = eval_kvpr_vector(refined)
    vec_gre, max_gre = eval_kvpr_vector(greedy_base)
    final = refined if lex_better(vec_ref, vec_gre) else greedy_base

    # Final memory safety check
    for g in range(gpu_num):
        mem = sum(float(m.model_size) for m in final.get(g, []))
        if mem - S > 1e-6:
            return greedy_base
    return final

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