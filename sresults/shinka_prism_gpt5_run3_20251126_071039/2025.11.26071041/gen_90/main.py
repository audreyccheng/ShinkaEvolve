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
    # Basic checks and trivial return
    placement_empty = {i: [] for i in range(gpu_num)}
    if gpu_num <= 0 or not models:
        return placement_empty

    S = GPU_MEM_SIZE

    # Helper: KVPR safely
    def kvpr(R, rem_mem):
        if rem_mem <= 0:
            return float('inf')
        return R / rem_mem

    # Greedy fallback minimizing local max-KVPR increase (robust to slo==0)
    def _greedy_fallback_kvpr(gpu_num, models, S):
        # Sort by demand ratio descending; tie by size desc
        sorted_models = sorted(
            models,
            key=lambda m: ((m.req_rate / m.slo) if getattr(m, "slo", 0) != 0 else 0.0, m.model_size),
            reverse=True
        )
        placement = {i: [] for i in range(gpu_num)}
        rem_mem = [S] * gpu_num
        sum_R = [0.0] * gpu_num
        for m in sorted_models:
            dR_eff = (m.req_rate / m.slo) if getattr(m, "slo", 0) != 0 else 0.0
            size = float(m.model_size)
            best_gid = None
            best_val = float('inf')
            for gid in range(gpu_num):
                if size <= rem_mem[gid]:
                    rem = rem_mem[gid] - size
                    if rem <= 0:
                        continue
                    val = (sum_R[gid] + dR_eff) / rem
                    if val < best_val:
                        best_val = val
                        best_gid = gid
            if best_gid is None:
                raise ValueError(
                    f"Unable to place model of size {m.model_size} GB on any GPU. "
                    f"Remaining per-GPU memory: {rem_mem}"
                )
            placement[best_gid].append(m)
            sum_R[best_gid] += dR_eff
            rem_mem[best_gid] -= size
        return placement

    # Extract per-model attributes once
    items = []
    total_R = 0.0
    total_size = 0.0
    for m in models:
        slo = float(m.slo)
        dR = float(m.req_rate) / slo if slo != 0 else 0.0
        s = float(m.model_size)
        items.append({'obj': m, 'dR': dR, 'size': s})
        total_R += dR
        total_size += s

    # Lower bounds on optimal T
    def compute_lower_bound():
        # Per-item bound: T >= dR / (S - s)
        lb1 = 0.0
        infeasible_single = False
        for it in items:
            dR = it['dR']; s = it['size']
            denom = S - s
            if denom <= 0:
                if dR > 0 and dR != float('inf'):
                    infeasible_single = True
                elif dR == float('inf'):
                    infeasible_single = True
                continue
            if dR > 0 and dR != float('inf'):
                cand = dR / denom
                if cand > lb1:
                    lb1 = cand

        # Global bound: T >= total_R / (gpu_num*S - total_size)
        denom2 = gpu_num * S - total_size
        if denom2 <= 0 and total_R > 0:
            return float('inf'), True, infeasible_single
        lb2 = 0.0 if total_R <= 0 or denom2 <= 0 else (total_R / denom2)

        # Pair bound: for pairs that cannot co-reside (s_i + s_j > S)
        lb_pair = 0.0
        P = min(len(items), 200)
        by_size = sorted(items, key=lambda x: x['size'], reverse=True)[:P]
        for i in range(len(by_size)):
            si = by_size[i]['size']; ri = by_size[i]['dR']
            for j in range(i + 1, len(by_size)):
                sj = by_size[j]['size']; rj = by_size[j]['dR']
                if si + sj > S:
                    denom = 2 * S - (si + sj)
                    if denom > 0:
                        ri_f = 0.0 if ri == float('inf') else ri
                        rj_f = 0.0 if rj == float('inf') else rj
                        cand = (ri_f + rj_f) / denom
                        if cand > lb_pair:
                            lb_pair = cand

        # Lightweight triplet bound over top-by-size items
        lb_triplet = 0.0
        if len(by_size) >= 3:
            topK = min(8, len(by_size))
            top_items = by_size[:topK]
            for i in range(len(by_size)):
                si = by_size[i]['size']; ri = by_size[i]['dR']
                for j in range(i + 1, len(by_size)):
                    sj = by_size[j]['size']; rj = by_size[j]['dR']
                    for mk in top_items:
                        if mk is by_size[i] or mk is by_size[j]:
                            continue
                        sk = mk['size']; rk = mk['dR']
                        ssum = si + sj + sk
                        if ssum > 2 * S:
                            denom = 3 * S - ssum
                            if denom > 0:
                                ri_f = 0.0 if ri == float('inf') else ri
                                rj_f = 0.0 if rj == float('inf') else rj
                                rk_f = 0.0 if rk == float('inf') else rk
                                cand = (ri_f + rj_f + rk_f) / denom
                                if cand > lb_triplet:
                                    lb_triplet = cand

        # k-bin prefix bound for k in {1..min(gpu_num,6)}
        lb_k = 0.0
        sorted_by_size = sorted(items, key=lambda x: x['size'], reverse=True)
        prefix_sizes = []
        prefix_rates = []
        cs = 0.0; cr = 0.0
        for it in sorted_by_size:
            cs += it['size']; cr += (0.0 if it['dR'] == float('inf') else it['dR'])
            prefix_sizes.append(cs); prefix_rates.append(cr)
        for k in range(1, min(gpu_num, 6) + 1):
            threshold = (k - 1) * S
            idx = -1
            for t in range(len(prefix_sizes)):
                if prefix_sizes[t] > threshold:
                    idx = t
                    break
            if idx >= 0:
                numer = prefix_rates[idx]
                denom = k * S - prefix_sizes[idx]
                if denom > 0 and numer > 0:
                    cand = numer / denom
                    if cand > lb_k:
                        lb_k = cand

        lower = max(0.0, lb1, lb2, lb_pair, lb_triplet, lb_k)
        return lower, False, infeasible_single

    lower, infeasible_global, infeasible_single = compute_lower_bound()
    if infeasible_single or infeasible_global or lower == float('inf'):
        # Fallback: simple greedy minimizing local KVPR increase
        return _greedy_fallback_kvpr(gpu_num, models, S)

    # Deterministic small-random helper
    import random
    rng = random.Random(len(items) * 1009 + gpu_num * 9173)

    # Slack-equalization packer for a given T
    # K_g = T*S - (sumR_g + T*used_mem_g); placing (dR, s) consumes w = dR + T*s
    def assign_balanced_slack(T, order='w_desc', seed_H=0, choose_rule='tight', seeds=1):
        if T < 0:
            return None
        # Build enriched list
        enriched = []
        for it in items:
            dR = it['dR']; sz = it['size']
            w = dR + T * sz
            if w < 0:
                w = 0.0
            enriched.append([w, dR, sz, it['obj']])

        # Orderings
        if order == 'w_desc':
            enriched.sort(key=lambda x: x[0], reverse=True)
        elif order == 'intrinsic_desc':
            # dR/(S - sz)
            enriched.sort(key=lambda x: (x[1] / max(S - x[2], 1e-9)), reverse=True)
        elif order == 'size_desc':
            enriched.sort(key=lambda x: x[2], reverse=True)
        elif order == 'density_desc':
            enriched.sort(key=lambda x: (x[1] / (x[2] if x[2] > 0 else 1e-9)), reverse=True)
        else:
            enriched.sort(key=lambda x: x[0], reverse=True)

        best_assign = None
        best_val = float('inf')

        for _ in range(max(1, seeds)):
            # per-GPU states
            assign = {i: [] for i in range(gpu_num)}
            used_mem = [0.0] * gpu_num
            sum_R = [0.0] * gpu_num
            K = [T * S] * gpu_num  # KV slack

            # Seeding: spread top H intrinsic-pressure models using worst-fit on K
            H = int(seed_H) if seed_H else 0
            if H > 0 and len(enriched) > 0:
                intrinsic = sorted(enriched, key=lambda x: (x[1] / max(S - x[2], 1e-9)), reverse=True)
                seeds_list = intrinsic[:min(H, len(enriched))]
                # Remove seeds from enriched while preserving original sequence among the rest
                seed_ids = set(id(x) for x in seeds_list)
                remaining = [x for x in enriched if id(x) not in seed_ids]
                for w, dR, sz, m in seeds_list:
                    # Choose GPU with largest K that can fit both memory and slack
                    candidates = []
                    for gid in range(gpu_num):
                        if used_mem[gid] + sz <= S + 1e-9 and K[gid] >= w - 1e-9:
                            candidates.append(gid)
                    if not candidates:
                        assign = None
                        break
                    # Worst-fit on K; tie-break by larger memory slack, then lower resulting local kvpr
                    def key_fn(g):
                        rem = S - (used_mem[g] + sz)
                        kvnew = kvpr(sum_R[g] + dR, rem) if rem > 0 else float('inf')
                        return (-K[g], -(rem), kvnew)
                    candidates.sort(key=lambda g: key_fn(g))
                    chosen = candidates[0]
                    if rng.random() < 0.2 and len(candidates) > 1:
                        chosen = candidates[rng.randrange(min(2, len(candidates)))]
                    assign[chosen].append(m)
                    used_mem[chosen] += sz
                    sum_R[chosen] += dR
                    K[chosen] -= w
                if assign is None:
                    continue
                enriched = remaining

            # Main packing: equalize K by choosing GPU that leaves minimal nonnegative K'
            for w, dR, sz, m in enriched:
                options = []
                for gid in range(gpu_num):
                    if used_mem[gid] + sz <= S + 1e-9 and K[gid] >= w - 1e-9:
                        K_after = K[gid] - w
                        if choose_rule == 'min_kvpr':
                            rem = S - (used_mem[gid] + sz)
                            if rem <= 0:
                                continue
                            kv_new = kvpr(sum_R[gid] + dR, rem)
                            options.append((gid, K_after, kv_new))
                        else:
                            options.append((gid, K_after, None))
                if not options:
                    assign = None
                    break
                # Selection: prefer minimal K_after (tightest feasible), tie-break by min kvpr or larger mem slack
                if choose_rule == 'min_kvpr':
                    options.sort(key=lambda t: (t[1], t[2]))
                else:
                    options.sort(key=lambda t: t[1])
                chosen = options[0][0]
                # Tiny randomization among top-2 close in K_after
                if len(options) > 1:
                    bestK = options[0][1]
                    near = [op for op in options if abs(op[1] - bestK) <= max(1e-9, 0.001 * (T * S + 1e-9))]
                    if len(near) >= 2 and rng.random() < 0.25:
                        chosen = near[rng.randrange(min(2, len(near)))][0]

                assign[chosen].append(m)
                used_mem[chosen] += sz
                sum_R[chosen] += dR
                K[chosen] -= w

            if assign is None:
                continue

            # Validate constraints strictly
            ok = True
            for gid in range(gpu_num):
                if used_mem[gid] - S > 1e-6:
                    ok = False; break
                rem = S - used_mem[gid]
                if rem <= 0 and sum_R[gid] > 0:
                    ok = False; break
                if T > 0 and (sum_R[gid] / max(rem, 1e-12)) - T > 1e-6:
                    ok = False; break
            if not ok:
                continue

            # Keep best by measured max KVPR
            val = 0.0
            for gid in range(gpu_num):
                rem = S - used_mem[gid]
                val = max(val, kvpr(sum_R[gid], rem))
            if val < best_val:
                best_val = val
                best_assign = assign

        return best_assign

    # Find the first feasible T by multiplicative sweep from the lower bound
    def find_first_feasible_T(start_T):
        T = max(0.0, start_T)
        growth = 1.08
        max_steps = 40
        for _ in range(max_steps):
            for (order, choose_rule, seedH, seeds) in [
                ('w_desc', 'tight', min(4, max(1, gpu_num)), 1),
                ('intrinsic_desc', 'tight', min(3, max(1, gpu_num - 1)), 1),
                ('w_desc', 'min_kvpr', 0, 1),
            ]:
                cand = assign_balanced_slack(T, order=order, seed_H=seedH, choose_rule=choose_rule, seeds=1)
                if cand is not None:
                    return T, cand
            T = T * growth if T > 0 else 1e-6
        return None, None

    T_feas, placement_at_T = find_first_feasible_T(lower)
    if placement_at_T is None:
        # As a safety net
        return _greedy_fallback_kvpr(gpu_num, models, S)

    # Build a compact set of candidate T values around the first feasible T
    def eval_max_kvpr(placement_dict):
        max_v = 0.0
        for gid in range(gpu_num):
            bucket = placement_dict.get(gid, [])
            used = 0.0
            R = 0.0
            for m in bucket:
                used += float(m.model_size)
                R += float(m.req_rate / m.slo) if getattr(m, "slo", 0) != 0 else 0.0
            val = kvpr(R, S - used)
            if val > max_v:
                max_v = val
        return max_v

    candidates_T = []
    for mul in [0.975, 0.985, 0.99, 1.0, 1.005, 1.01, 1.02, 1.03]:
        val = max(lower, T_feas * mul)
        candidates_T.append(val)
    candidates_T.append(0.5 * (lower + T_feas))
    candidates_T = sorted(set(round(t, 12) for t in candidates_T))

    # Try a few packing variants per T and pick best by measured KVPR
    variants = [
        ('w_desc', 'tight', min(4, max(1, gpu_num)), 3),
        ('intrinsic_desc', 'tight', min(3, max(1, gpu_num - 1))),
        ('intrinsic_desc', 'min_kvpr', 0),
        ('density_desc', 'tight', 0),
        ('w_desc', 'min_kvpr', 0),
        ('size_desc', 'tight', 0),
    ]

    best_placement = placement_at_T
    best_val = eval_max_kvpr(best_placement)

    for T in candidates_T:
        for var in variants:
            # Unpack with defaults for backward compatibility tuple length
            if len(var) == 4:
                order, choose_rule, seedH, seeds = var
            else:
                order, choose_rule, seedH = var
                seeds = 1
            cand = assign_balanced_slack(T, order=order, seed_H=seedH, choose_rule=choose_rule, seeds=seeds)
            if cand is None:
                continue
            val = eval_max_kvpr(cand)
            if val < best_val:
                best_val = val
                best_placement = cand

    # Tighten T around the feasibility threshold with proper bracketing
    def _feasible_assign_for_T(T):
        best = None
        bestv = float('inf')
        for (order, choose_rule, seedH, seeds) in [
            ('w_desc', 'tight', min(4, max(1, gpu_num)), 1),
            ('intrinsic_desc', 'tight', min(3, max(1, gpu_num - 1)), 1),
            ('w_desc', 'min_kvpr', 0, 2),
            ('intrinsic_desc', 'min_kvpr', 0, 1),
        ]:
            cand = assign_balanced_slack(T, order=order, seed_H=seedH, choose_rule=choose_rule, seeds=seeds)
            if cand is not None:
                v = eval_max_kvpr(cand)
                if v < bestv:
                    bestv = v
                    best = cand
        return best

    # Bracket using the first feasible T found by the sweep
    lo, hi = max(0.0, lower), max(0.0, T_feas if T_feas is not None else lower)
    for _ in range(8):
        mid = (lo + hi) / 2.0
        cand = _feasible_assign_for_T(mid)
        if cand is not None:
            hi = mid
            v = eval_max_kvpr(cand)
            if v < best_val:
                best_val = v
                best_placement = cand
        else:
            lo = mid

    # Probe a narrow neighborhood around the tightened feasible T
    for mul in [0.99, 1.0, 1.01]:
        T_probe = max(lower, hi * mul)
        for (order, choose_rule, seedH, seeds) in [
            ('w_desc', 'tight', min(4, max(1, gpu_num)), 2),
            ('intrinsic_desc', 'tight', min(3, max(1, gpu_num - 1)), 1),
            ('w_desc', 'min_kvpr', 0, 2),
        ]:
            cand = assign_balanced_slack(T_probe, order=order, seed_H=seedH, choose_rule=choose_rule, seeds=seeds)
            if cand is not None:
                v = eval_max_kvpr(cand)
                if v < best_val:
                    best_val = v
                    best_placement = cand

    # Short bounded local search focusing on the most loaded GPU: moves then swaps
    def local_refine(placement, move_budget=20, swap_budget=10):
        buckets = {gid: list(placement.get(gid, [])) for gid in range(gpu_num)}
        used_mem = [0.0] * gpu_num
        sum_R = [0.0] * gpu_num
        for gid in range(gpu_num):
            for m in buckets[gid]:
                used_mem[gid] += float(m.model_size)
                sum_R[gid] += float(m.req_rate / m.slo) if m.slo != 0 else 0.0

        def current_kvprs():
            return [kvpr(sum_R[g], S - used_mem[g]) for g in range(gpu_num)]

        def apply_move(src, dst, mdl):
            dR = float(mdl.req_rate / mdl.slo) if mdl.slo != 0 else 0.0
            s = float(mdl.model_size)
            buckets[src].remove(mdl); buckets[dst].append(mdl)
            sum_R[src] -= dR; sum_R[dst] += dR
            used_mem[src] -= s; used_mem[dst] += s

        moves_left = move_budget
        swaps_left = swap_budget
        while moves_left > 0 or swaps_left > 0:
            kvprs = current_kvprs()
            cur_max = max(kvprs)
            max_gid = max(range(gpu_num), key=lambda g: kvprs[g])

            improved = False
            best_move = None  # (src,dst,mdl,resulting_max)
            # Single-item moves
            for mdl in list(buckets[max_gid]):
                dR = float(mdl.req_rate / mdl.slo) if mdl.slo != 0 else 0.0
                s = float(mdl.model_size)
                R_src_new = sum_R[max_gid] - dR
                mem_src_new = used_mem[max_gid] - s
                rem_src_new = S - mem_src_new
                if rem_src_new <= 0:
                    continue
                kv_src_new = kvpr(R_src_new, rem_src_new)
                for dst in range(gpu_num):
                    if dst == max_gid:
                        continue
                    if used_mem[dst] + s > S:
                        continue
                    rem_dst_new = S - (used_mem[dst] + s)
                    if rem_dst_new <= 0:
                        continue
                    R_dst_new = sum_R[dst] + dR
                    kv_dst_new = kvpr(R_dst_new, rem_dst_new)
                    resulting = kv_dst_new if kv_dst_new > kv_src_new else kv_src_new
                    for g in range(gpu_num):
                        if g == max_gid or g == dst:
                            continue
                        if kvprs[g] > resulting:
                            resulting = kvprs[g]
                    if resulting + 1e-12 < cur_max:
                        if best_move is None or resulting < best_move[3]:
                            best_move = (max_gid, dst, mdl, resulting)
            if best_move is not None and moves_left > 0:
                src, dst, mdl, _ = best_move
                apply_move(src, dst, mdl)
                moves_left -= 1
                improved = True
            else:
                # Try limited swaps
                best_swap = None  # (src,dst,a,b,resulting_max)
                if swaps_left > 0:
                    cap_a = min(10, len(buckets[max_gid]))
                    for a in list(buckets[max_gid])[:cap_a]:
                        aR = float(a.req_rate / a.slo) if a.slo != 0 else 0.0
                        aS = float(a.model_size)
                        for dst in range(gpu_num):
                            if dst == max_gid or not buckets[dst]:
                                continue
                            cap_b = min(10, len(buckets[dst]))
                            for b in list(buckets[dst])[:cap_b]:
                                bR = float(b.req_rate / b.slo) if b.slo != 0 else 0.0
                                bS = float(b.model_size)
                                mem_src_new = used_mem[max_gid] - aS + bS
                                mem_dst_new = used_mem[dst] - bS + aS
                                if mem_src_new > S or mem_dst_new > S:
                                    continue
                                rem_src = S - mem_src_new
                                rem_dst = S - mem_dst_new
                                if rem_src <= 0 or rem_dst <= 0:
                                    continue
                                R_src_new = sum_R[max_gid] - aR + bR
                                R_dst_new = sum_R[dst] - bR + aR
                                kv_src_new = kvpr(R_src_new, rem_src)
                                kv_dst_new = kvpr(R_dst_new, rem_dst)
                                resulting = kv_src_new if kv_src_new > kv_dst_new else kv_dst_new
                                for g in range(gpu_num):
                                    if g == max_gid or g == dst:
                                        continue
                                    val = kvpr(sum_R[g], S - used_mem[g])
                                    if val > resulting:
                                        resulting = val
                                if resulting + 1e-12 < cur_max:
                                    if best_swap is None or resulting < best_swap[4]:
                                        best_swap = (max_gid, dst, a, b, resulting)
                if best_swap is not None:
                    src, dst, a, b, _ = best_swap
                    buckets[src].remove(a); buckets[src].append(b)
                    buckets[dst].remove(b); buckets[dst].append(a)
                    aR = float(a.req_rate / a.slo) if a.slo != 0 else 0.0
                    bR = float(b.req_rate / b.slo) if b.slo != 0 else 0.0
                    aS = float(a.model_size); bS = float(b.model_size)
                    sum_R[src] = sum_R[src] - aR + bR
                    sum_R[dst] = sum_R[dst] - bR + aR
                    used_mem[src] = used_mem[src] - aS + bS
                    used_mem[dst] = used_mem[dst] - bS + aS
                    swaps_left -= 1
                    improved = True
            if not improved:
                break
        return buckets

    # Refinements and candidate selection
    refined = local_refine(best_placement, move_budget=18, swap_budget=8)
    greedy_baseline = _greedy_fallback_kvpr(gpu_num, models, S)
    greedy_refined = local_refine(greedy_baseline, move_budget=12, swap_budget=6)

    # Select the best by measured max KVPR
    candidates = [
        best_placement,
        refined,
        greedy_refined,
    ]
    best_final = candidates[0]
    best_score = eval_max_kvpr(best_final)
    for cand in candidates[1:]:
        val = eval_max_kvpr(cand)
        if val < best_score:
            best_score = val
            best_final = cand

    # Final check: memory safety
    for gid in range(gpu_num):
        mem = sum(float(m.model_size) for m in best_final.get(gid, []))
        if mem - S > 1e-6:
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