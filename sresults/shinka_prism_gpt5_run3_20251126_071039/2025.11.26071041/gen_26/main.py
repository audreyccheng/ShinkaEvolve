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
    S = GPU_MEM_SIZE

    # Safe KVPR
    def kvpr(R, rem_mem):
        if rem_mem <= 0:
            return float('inf')
        return R / rem_mem

    # Early return for trivial cases
    empty = {i: [] for i in range(gpu_num)}
    if gpu_num <= 0 or not models:
        return empty

    # Extract items: dR and size
    items = []
    total_R = 0.0
    total_size = 0.0
    for m in models:
        slo = getattr(m, 'slo', 0)
        if slo == 0:
            dR = float('inf')
        else:
            dR = float(m.req_rate) / float(slo)
        s = float(m.model_size)
        items.append({'obj': m, 'dR': dR, 'size': s})
        if dR != float('inf'):
            total_R += dR
        total_size += s

    # Greedy fallback minimizing local KVPR increase
    def _greedy_fallback_kvpr():
        sorted_models = sorted(models, key=lambda m: ((m.req_rate / m.slo) if getattr(m, 'slo', 0) != 0 else float('inf'),
                                                      m.model_size), reverse=True)
        placement = {i: [] for i in range(gpu_num)}
        rem_mem = [S] * gpu_num
        sum_R = [0.0] * gpu_num
        for m in sorted_models:
            dR = (m.req_rate / m.slo) if getattr(m, 'slo', 0) != 0 else float('inf')
            sz = float(m.model_size)
            best = None
            best_val = float('inf')
            for gid in range(gpu_num):
                if sz <= rem_mem[gid]:
                    rem_after = rem_mem[gid] - sz
                    if rem_after <= 0:
                        continue
                    cand = (sum_R[gid] + (0.0 if dR == float('inf') else dR)) / rem_after
                    if cand < best_val:
                        best_val = cand
                        best = gid
            if best is None:
                raise ValueError(
                    f"Unable to place model of size {m.model_size} GB on any GPU. "
                    f"Remaining per-GPU memory: {rem_mem}"
                )
            placement[best].append(m)
            if dR != float('inf'):
                sum_R[best] += dR
            rem_mem[best] -= sz
        return placement

    # Lower bounds on optimal T for a given set of items
    def compute_lower_bound(sub_items, residual_total_mem=None):
        lb1 = 0.0
        infeasible_single = False
        for it in sub_items:
            dR = it['dR']; sz = it['size']
            denom = S - sz
            if denom <= 0:
                if dR > 0:
                    infeasible_single = True
                continue
            if dR > 0:
                cand = dR / denom
                if cand > lb1:
                    lb1 = cand

        # Global bound with (optional) residual total memory
        denom2 = (residual_total_mem if residual_total_mem is not None else (gpu_num * S)) - sum(it['size'] for it in sub_items)
        totalR = sum((it['dR'] if it['dR'] != float('inf') else 0.0) for it in sub_items)
        if denom2 <= 0 and totalR > 0:
            return float('inf'), True, infeasible_single
        lb2 = 0.0 if totalR <= 0 or denom2 <= 0 else (totalR / denom2)

        # Pair bound over top-by-size
        lb_pair = 0.0
        P = min(len(sub_items), 200)
        by_size = sorted(sub_items, key=lambda x: x['size'], reverse=True)[:P]
        for i in range(len(by_size)):
            si = by_size[i]['size']; ri = by_size[i]['dR']
            for j in range(i + 1, len(by_size)):
                sj = by_size[j]['size']; rj = by_size[j]['dR']
                if si + sj > S:
                    denom = 2 * S - (si + sj)
                    if denom > 0:
                        cand = ((0.0 if ri == float('inf') else ri) + (0.0 if rj == float('inf') else rj)) / denom
                        if cand > lb_pair:
                            lb_pair = cand

        # Triplet bound (light): scan P2^2 pairs and top-10 size candidates for third
        lb_trip = 0.0
        P2 = min(len(sub_items), 80)
        by_size2 = sorted(sub_items, key=lambda x: x['size'], reverse=True)[:P2]
        top10 = by_size[:10] if len(by_size) >= 10 else by_size
        for i in range(len(by_size2)):
            si = by_size2[i]['size']; ri = by_size2[i]['dR']; ri = 0.0 if ri == float('inf') else ri
            for j in range(i + 1, len(by_size2)):
                sj = by_size2[j]['size']; rj = by_size2[j]['dR']; rj = 0.0 if rj == float('inf') else rj
                for k in range(len(top10)):
                    if top10[k] is by_size2[i] or top10[k] is by_size2[j]:
                        continue
                    sk = top10[k]['size']; rk = top10[k]['dR']; rk = 0.0 if rk == float('inf') else rk
                    if si + sj + sk > 2 * S:
                        denom = 3 * S - (si + sj + sk)
                        if denom > 0:
                            cand = (ri + rj + rk) / denom
                            if cand > lb_trip:
                                lb_trip = cand

        # k-prefix bound for k up to 6
        lb_k = 0.0
        sorted_by_size = sorted(sub_items, key=lambda x: x['size'], reverse=True)
        cs = 0.0; cr = 0.0
        prefix_sizes = []
        prefix_rates = []
        for it in sorted_by_size:
            cs += it['size']; cr += (0.0 if it['dR'] == float('inf') else it['dR'])
            prefix_sizes.append(cs); prefix_rates.append(cr)
        for k in range(1, min(gpu_num, 6) + 1):
            threshold = (k - 1) * S
            idx = -1
            for t in range(len(prefix_sizes)):
                if prefix_sizes[t] > threshold:
                    idx = t; break
            if idx >= 0:
                numer = prefix_rates[idx]
                denom = k * S - prefix_sizes[idx]
                if denom > 0 and numer > 0:
                    cand = numer / denom
                    if cand > lb_k:
                        lb_k = cand

        lower = max(0.0, lb1, lb2, lb_pair, lb_trip, lb_k)
        return lower, False, infeasible_single

    # Assignment via slack equalization for a given T with variants
    def assign_balanced_slack(T, order='w_desc', seed_H=0, choose_rule='tight', seeds=1, phase_fraction=0.45):
        import random
        rng = random.Random(len(items) * 971 + gpu_num * 101)
        if T < 0:
            return None

        # Prepare enriched items
        def build_enriched(T_local, base_list=None):
            src = items if base_list is None else base_list
            enriched_local = []
            for it in src:
                dR = it['dR']; sz = it['size']
                if dR == float('inf'):
                    return None
                w = dR + T_local * sz
                if w < 0: w = 0.0
                enriched_local.append([w, dR, sz, it['obj']])
            # Apply ordering
            if order == 'w_desc':
                enriched_local.sort(key=lambda x: x[0], reverse=True)
            elif order == 'intrinsic_desc':
                enriched_local.sort(key=lambda x: (x[1] / max(S - x[2], 1e-9)), reverse=True)
            elif order == 'size_desc':
                enriched_local.sort(key=lambda x: x[2], reverse=True)
            elif order == 'density_desc':
                enriched_local.sort(key=lambda x: (x[1] / (x[2] if x[2] > 0 else 1e-9)), reverse=True)
            else:
                enriched_local.sort(key=lambda x: x[0], reverse=True)
            return enriched_local

        enriched_master = build_enriched(T)
        if enriched_master is None:
            return None

        best_assign = None
        best_score = (float('inf'), float('inf'), float('inf'))  # (max, second, avg)

        for _ in range(max(1, seeds)):
            used_mem = [0.0] * gpu_num
            sum_R = [0.0] * gpu_num
            K = [T * S] * gpu_num
            assign = {i: [] for i in range(gpu_num)}
            enriched = list(enriched_master)

            # Seeding top H by intrinsic pressure with worst-fit on K
            H = int(seed_H) if seed_H else 0
            if H > 0 and enriched:
                intrinsic = sorted(enriched, key=lambda x: (x[1] / max(S - x[2], 1e-9)), reverse=True)[:H]
                # Remove seeds
                seed_set = set(id(x) for x in intrinsic)
                remaining = [x for x in enriched if id(x) not in seed_set]
                for w, dR, sz, m in intrinsic:
                    candidates = []
                    for gid in range(gpu_num):
                        if used_mem[gid] + sz <= S + 1e-9 and K[gid] >= w - 1e-9:
                            candidates.append(gid)
                    if not candidates:
                        assign = None
                        break
                    def tie_key(gid):
                        rem = S - (used_mem[gid] + sz)
                        kvnew = kvpr(sum_R[gid] + dR, rem) if rem > 0 else float('inf')
                        return (-K[gid], -(rem), kvnew)
                    candidates.sort(key=lambda g: tie_key(g))
                    chosen = candidates[0]
                    assign[chosen].append(m)
                    used_mem[chosen] += sz
                    sum_R[chosen] += dR
                    K[chosen] -= w
                if assign is None:
                    continue
                enriched = remaining

            # Two-phase: after placing a fraction, recompute LB on remaining and raise T if needed
            total_count = len(enriched)
            phase_cut = max(0, min(total_count, int(round(phase_fraction * total_count))))
            # Placement loop with optional dynamic T update
            placed = 0
            idx = 0
            while idx < len(enriched):
                w, dR, sz, m = enriched[idx]

                # Selection candidates
                options = []
                # For hybrid rule prepare normalization and alpha adapt
                if choose_rule == 'hybrid':
                    avg_mem_frac = (sum(used_mem) + 0.0) / (gpu_num * S) if gpu_num > 0 else 0.0
                    meanK = sum(K) / max(1, gpu_num)
                    varK = sum((Kg - meanK) ** 2 for Kg in K) / max(1, gpu_num)
                    alpha = 0.25 if (meanK > 0 and (varK / (meanK ** 2 + 1e-12)) < 0.05) else 0.15
                    beta = 0.05
                for gid in range(gpu_num):
                    if used_mem[gid] + sz <= S + 1e-9 and K[gid] >= w - 1e-9:
                        K_after = K[gid] - w
                        if choose_rule == 'tight':
                            options.append((gid, K_after, None, None))
                        elif choose_rule == 'min_kvpr':
                            rem = S - (used_mem[gid] + sz)
                            if rem <= 0:
                                continue
                            kv_new = kvpr(sum_R[gid] + dR, rem)
                            options.append((gid, K_after, kv_new, None))
                        else:  # hybrid
                            rem = S - (used_mem[gid] + sz)
                            if rem <= 0:
                                continue
                            kv_new = kvpr(sum_R[gid] + dR, rem)
                            K_after_norm = max(0.0, K_after) / max(T * S, 1e-12)
                            kv_new_norm = kv_new / max(T, 1e-12)
                            mem_imb = abs(((used_mem[gid] + sz) / S) - avg_mem_frac)
                            J = K_after_norm + alpha * kv_new_norm + beta * mem_imb
                            options.append((gid, K_after, kv_new, J))
                if not options:
                    assign = None
                    break

                # Choose GPU by rule
                chosen = None
                if choose_rule == 'tight':
                    options.sort(key=lambda t: (t[1]))
                    chosen = options[0][0]
                elif choose_rule == 'min_kvpr':
                    options.sort(key=lambda t: (t[1], t[2]))
                    chosen = options[0][0]
                else:
                    options.sort(key=lambda t: (t[3], t[2], t[1]))
                    chosen = options[0][0]

                # Commit
                assign[chosen].append(m)
                used_mem[chosen] += sz
                sum_R[chosen] += dR
                K[chosen] -= w
                placed += 1

                # Dynamic T update at phase cut
                if placed == phase_cut and idx + 1 < len(enriched):
                    # Recompute lower bound on remaining with residual memory
                    residual_total_mem = sum(S - used_mem[g] for g in range(gpu_num))
                    remaining_items = [{'obj': x[3], 'dR': x[1], 'size': x[2]} for x in enriched[idx+1:]]
                    lower_rem, infeas_g, infeas_s = compute_lower_bound(remaining_items, residual_total_mem=residual_total_mem)
                    if infeas_s or infeas_g:
                        assign = None
                        break
                    T_new = max(T, lower_rem)
                    if T_new > T + 1e-12:
                        # Update K to reflect the new T and rebuild remaining enriched with new weights
                        for g in range(gpu_num):
                            K[g] = T_new * S - (sum_R[g] + T_new * used_mem[g])
                        T = T_new
                        if order == 'w_desc':
                            # rebuild only the remaining tail
                            tail = []
                            for x in enriched[idx+1:]:
                                w2 = x[1] + T * x[2]
                                tail.append([w2, x[1], x[2], x[3]])
                            tail.sort(key=lambda x: x[0], reverse=True)
                            enriched = enriched[:idx+1] + tail
                idx += 1

            if assign is None:
                continue

            # Validate
            ok = True
            for g in range(gpu_num):
                if used_mem[g] - S > 1e-6:
                    ok = False; break
                rem = S - used_mem[g]
                if rem <= 0 and sum_R[g] > 0:
                    ok = False; break
            if not ok:
                continue

            # Score placement: lexicographic (max, second, avg)
            kvprs = []
            for g in range(gpu_num):
                kvprs.append(kvpr(sum_R[g], S - used_mem[g]))
            kvprs_sorted = sorted(kvprs, reverse=True)
            max_v = kvprs_sorted[0] if kvprs_sorted else 0.0
            sec_v = kvprs_sorted[1] if len(kvprs_sorted) > 1 else 0.0
            avg_v = sum(kvprs) / max(1, len(kvprs))
            score = (max_v, sec_v, avg_v)
            if score < best_score:
                best_score = score
                best_assign = assign

        return best_assign

    # Find first feasible T via multiplicative sweep starting from tight lower bound
    def find_first_feasible_T():
        lower, infeas_global, infeas_single = compute_lower_bound(items)
        if infeas_global or infeas_single or lower == float('inf'):
            return None, None

        growth = 1.07
        max_steps = 42
        T = max(0.0, lower)
        variants = [
            ('w_desc', 'tight', min(4, max(1, gpu_num)), 2),
            ('intrinsic_desc', 'tight', min(3, max(1, gpu_num - 1)), 2),
            ('w_desc', 'hybrid', min(3, max(1, gpu_num - 1)), 2),
        ]
        for _ in range(max_steps):
            for order, rule, seedH, seeds in variants:
                cand = assign_balanced_slack(T, order=order, seed_H=seedH, choose_rule=rule, seeds=seeds)
                if cand is not None:
                    return T, cand, lower
            T = T * growth if T > 0 else 1e-6
        return None, None, lower

    # Evaluate placement KVPRs
    def eval_kvprs(placement):
        used = [0.0] * gpu_num
        sums = [0.0] * gpu_num
        for gid in range(gpu_num):
            for m in placement.get(gid, []):
                used[gid] += float(m.model_size)
                slo = getattr(m, 'slo', 0)
                if slo != 0:
                    sums[gid] += float(m.req_rate) / float(slo)
        kvprs = [kvpr(sums[g], S - used[g]) for g in range(gpu_num)]
        return kvprs

    def lex_score(placement):
        kvprs = eval_kvprs(placement)
        ks = sorted(kvprs, reverse=True)
        max_v = ks[0] if ks else 0.0
        sec_v = ks[1] if len(ks) > 1 else 0.0
        avg_v = sum(kvprs) / max(1, len(kvprs))
        return (max_v, sec_v, avg_v)

    # Local refinement: bounded improving moves then limited swaps
    def local_refine(placement, move_budget=24, swap_budget=12):
        buckets = {gid: list(placement.get(gid, [])) for gid in range(gpu_num)}
        used_mem = [0.0] * gpu_num
        sum_R = [0.0] * gpu_num
        for gid in range(gpu_num):
            for m in buckets[gid]:
                used_mem[gid] += float(m.model_size)
                slo = getattr(m, 'slo', 0)
                if slo != 0:
                    sum_R[gid] += float(m.req_rate) / float(slo)

        def current_kvprs():
            return [kvpr(sum_R[g], S - used_mem[g]) for g in range(gpu_num)]

        def apply_move(src, dst, mdl, dR, s):
            buckets[src].remove(mdl); buckets[dst].append(mdl)
            sum_R[src] -= dR; sum_R[dst] += dR
            used_mem[src] -= s; used_mem[dst] += s

        eps = 1e-12
        moves_left = move_budget
        swaps_left = swap_budget
        while moves_left > 0 or swaps_left > 0:
            kvprs = current_kvprs()
            cur_max = max(kvprs) if kvprs else 0.0
            max_gid = max(range(gpu_num), key=lambda g: kvprs[g]) if gpu_num > 0 else 0
            improved = False

            # Try best single-item move from worst GPU
            best_move = None  # (src,dst,mdl,resulting_max)
            for mdl in list(buckets[max_gid]):
                s = float(mdl.model_size)
                slo = getattr(mdl, 'slo', 0)
                dR = (float(mdl.req_rate) / float(slo)) if slo != 0 else 0.0
                src_R_new = sum_R[max_gid] - dR
                src_mem_new = used_mem[max_gid] - s
                rem_src = S - src_mem_new
                if rem_src <= 0:
                    continue
                kv_src_new = kvpr(src_R_new, rem_src)
                for dst in range(gpu_num):
                    if dst == max_gid:
                        continue
                    if used_mem[dst] + s > S:
                        continue
                    rem_dst = S - (used_mem[dst] + s)
                    if rem_dst <= 0:
                        continue
                    dst_R_new = sum_R[dst] + dR
                    kv_dst_new = kvpr(dst_R_new, rem_dst)
                    resulting = max(kv_dst_new, kv_src_new)
                    for g in range(gpu_num):
                        if g == max_gid or g == dst:
                            continue
                        if kvprs[g] > resulting:
                            resulting = kvprs[g]
                    if resulting + eps < cur_max:
                        if best_move is None or resulting < best_move[3]:
                            best_move = (max_gid, dst, mdl, resulting)
            if best_move is not None and moves_left > 0:
                src, dst, mdl, _ = best_move
                s = float(mdl.model_size)
                slo = getattr(mdl, 'slo', 0)
                dR = (float(mdl.req_rate) / float(slo)) if slo != 0 else 0.0
                apply_move(src, dst, mdl, dR, s)
                moves_left -= 1
                improved = True
            else:
                # Try pairwise swap between worst and others
                best_swap = None  # (src,dst,a,b,resulting_max)
                src = max_gid
                cap_a = min(10, len(buckets[src]))
                for a in list(buckets[src])[:cap_a]:
                    aS = float(a.model_size)
                    aslo = getattr(a, 'slo', 0)
                    aR = (float(a.req_rate) / float(aslo)) if aslo != 0 else 0.0
                    for dst in range(gpu_num):
                        if dst == src or not buckets[dst]:
                            continue
                        cap_b = min(10, len(buckets[dst]))
                        for b in list(buckets[dst])[:cap_b]:
                            bS = float(b.model_size)
                            bslo = getattr(b, 'slo', 0)
                            bR = (float(b.req_rate) / float(bslo)) if bslo != 0 else 0.0
                            src_mem_new = used_mem[src] - aS + bS
                            dst_mem_new = used_mem[dst] - bS + aS
                            if src_mem_new > S or dst_mem_new > S:
                                continue
                            rem_src = S - src_mem_new
                            rem_dst = S - dst_mem_new
                            if rem_src <= 0 or rem_dst <= 0:
                                continue
                            src_R_new = sum_R[src] - aR + bR
                            dst_R_new = sum_R[dst] - bR + aR
                            kv_src_new = kvpr(src_R_new, rem_src)
                            kv_dst_new = kvpr(dst_R_new, rem_dst)
                            resulting = max(kv_src_new, kv_dst_new)
                            for g in range(gpu_num):
                                if g == src or g == dst:
                                    continue
                                if kvprs[g] > resulting:
                                    resulting = kvprs[g]
                            if resulting + eps < cur_max:
                                if best_swap is None or resulting < best_swap[4]:
                                    best_swap = (src, dst, a, b, resulting)
                if best_swap is not None and swaps_left > 0:
                    src, dst, a, b, _ = best_swap
                    buckets[src].remove(a); buckets[src].append(b)
                    buckets[dst].remove(b); buckets[dst].append(a)
                    aS = float(a.model_size); bS = float(b.model_size)
                    aslo = getattr(a, 'slo', 0); bslo = getattr(b, 'slo', 0)
                    aR = (float(a.req_rate) / float(aslo)) if aslo != 0 else 0.0
                    bR = (float(b.req_rate) / float(bslo)) if bslo != 0 else 0.0
                    used_mem[src] = used_mem[src] - aS + bS
                    used_mem[dst] = used_mem[dst] - bS + aS
                    sum_R[src] = sum_R[src] - aR + bR
                    sum_R[dst] = sum_R[dst] - bR + aR
                    swaps_left -= 1
                    improved = True

            if not improved:
                break
        return buckets

    # Main search
    T_feas, placement_at_T, lower = find_first_feasible_T()
    if placement_at_T is None:
        # Safety net
        return _greedy_fallback_kvpr()

    # Candidates of T around first feasible
    candidates_T = sorted(set([
        max(lower, T_feas * 0.985),
        max(lower, T_feas * 0.995),
        max(lower, T_feas),
        max(lower, T_feas * 1.005),
        max(lower, T_feas * 1.02),
        0.5 * (lower + T_feas),
    ]))

    # Try several packing variants per T and pick by lexicographic score
    variants = [
        ('w_desc', 'tight', min(4, max(1, gpu_num)), 2),
        ('intrinsic_desc', 'tight', min(3, max(1, gpu_num - 1)), 2),
        ('w_desc', 'hybrid', min(3, max(1, gpu_num - 1)), 2),
        ('density_desc', 'tight', 0, 1),
        ('size_desc', 'tight', 0, 1),
    ]

    best_placement = placement_at_T
    best_score = lex_score(best_placement)

    for T in candidates_T:
        for order, rule, seedH, seeds in variants:
            cand = assign_balanced_slack(T, order=order, seed_H=seedH, choose_rule=rule, seeds=seeds)
            if cand is None:
                continue
            sc = lex_score(cand)
            if sc < best_score:
                best_score = sc
                best_placement = cand

    # Local refinement
    refined = local_refine(best_placement, move_budget=24, swap_budget=12)
    # Ensure memory safety
    for gid in range(gpu_num):
        mem = sum(float(m.model_size) for m in refined.get(gid, []))
        if mem - S > 1e-6:
            return best_placement

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