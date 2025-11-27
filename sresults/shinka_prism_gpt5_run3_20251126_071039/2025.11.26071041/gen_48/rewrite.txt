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
        A placement (dict: gpu_id -> list of models) minimizing max KVPR
    """

    # Early returns
    placement = {i: [] for i in range(gpu_num)}
    if gpu_num <= 0 or not models:
        return placement

    S = GPU_MEM_SIZE

    # Safe KVPR
    def kvpr(sum_R, rem_mem):
        if rem_mem <= 0:
            return float('inf')
        return sum_R / rem_mem

    # Split models: memory-only (slo == 0) vs demand-bearing (slo > 0)
    mem_only = []
    demand_items = []
    for m in models:
        slo = getattr(m, 'slo', 0)
        if slo == 0:
            mem_only.append(m)
        else:
            demand_items.append(m)

    # Phase 1: Pre-pack memory-only models by pure memory balancing (largest-first)
    rem_mem = [S for _ in range(gpu_num)]
    if mem_only:
        # Place largest memory-only first to GPU with max remaining memory
        mem_only_sorted = sorted(mem_only, key=lambda x: float(getattr(x, 'model_size', 0.0)), reverse=True)
        for m in mem_only_sorted:
            size = float(getattr(m, 'model_size', 0.0))
            best_gid = max(range(gpu_num), key=lambda g: rem_mem[g])
            if size > rem_mem[best_gid] + 1e-12:
                # Try all GPUs (might be needed if max isn't enough but others are)
                best_gid = None
                best_free = -1.0
                for g in range(gpu_num):
                    if size <= rem_mem[g] and rem_mem[g] > best_free:
                        best_free = rem_mem[g]
                        best_gid = g
                if best_gid is None:
                    raise ValueError(
                        f"Unable to place memory-only model of size {size} GB. "
                        f"Remaining per-GPU memory: {rem_mem}"
                    )
            placement[best_gid].append(m)
            rem_mem[best_gid] -= size

    # If no demand-bearing models, return pre-packed placement
    if not demand_items:
        return placement

    # Build demand item tuples: (dR, size, obj)
    items = []
    total_R = 0.0
    total_size = 0.0
    for m in demand_items:
        slo = float(getattr(m, 'slo', 1.0))
        req = float(getattr(m, 'req_rate', 0.0))
        dR = req / slo if slo > 0 else 0.0  # demand-bearing set ensures slo > 0
        sz = float(getattr(m, 'model_size', 0.0))
        items.append((dR, sz, m))
        total_R += dR
        total_size += sz

    # Strong lower bounds on T based on residual capacities
    # Residual capacity vector after pre-placement
    S_eff = [max(0.0, rem_mem[g]) for g in range(gpu_num)]
    S_eff_sum = sum(S_eff)
    S_eff_max = max(S_eff) if S_eff else 0.0

    def compute_lower_bound():
        infeasible_single = False

        # Per-item bound: item must fit to some GPU; we use S_eff_max conservatively
        lb1 = 0.0
        for (dR, sz, _) in items:
            denom = S_eff_max - sz
            if denom <= 0:
                if dR > 0:
                    infeasible_single = True
                continue
            cand = dR / denom if dR > 0 else 0.0
            if cand > lb1:
                lb1 = cand

        # Global bound on residual instance
        denom2 = S_eff_sum - total_size
        if denom2 <= 0 and total_R > 1e-12:
            return float('inf'), True, infeasible_single
        lb2 = (total_R / denom2) if (denom2 > 0 and total_R > 0) else 0.0

        # Pair bound: si + sj > S_eff_max -> they cannot co-reside anywhere
        lb_pair = 0.0
        P = min(len(items), 200)
        by_size = sorted(items, key=lambda t: t[1], reverse=True)[:P]
        for i in range(len(by_size)):
            ri, si, _ = by_size[i][0], by_size[i][1], by_size[i][2]
            for j in range(i + 1, len(by_size)):
                rj, sj, _ = by_size[j][0], by_size[j][1], by_size[j][2]
                if si + sj > S_eff_max:
                    denom = 2 * S_eff_max - (si + sj)
                    if denom > 0:
                        cand = (ri + rj) / denom
                        if cand > lb_pair:
                            lb_pair = cand

        # Triplet bound: si + sj + sk > 2*S_eff_max
        lb_triplet = 0.0
        if len(by_size) >= 3:
            KTOP = min(8, len(by_size))
            topK = by_size[:KTOP]
            for i in range(len(by_size)):
                ri, si = by_size[i][0], by_size[i][1]
                for j in range(i + 1, len(by_size)):
                    rj, sj = by_size[j][0], by_size[j][1]
                    for mk in topK:
                        rk, sk = mk[0], mk[1]
                        ssum = si + sj + sk
                        if ssum > 2 * S_eff_max:
                            denom = 3 * S_eff_max - ssum
                            if denom > 0:
                                cand = (ri + rj + rk) / denom
                                if cand > lb_triplet:
                                    lb_triplet = cand

        # k-prefix bound (k up to min(gpu_num, 6))
        lb_k = 0.0
        sorted_by_size = sorted(items, key=lambda t: t[1], reverse=True)
        prefix_sizes = []
        prefix_rates = []
        cs = 0.0
        cr = 0.0
        for (dR, sz, _) in sorted_by_size:
            cs += sz
            cr += dR
            prefix_sizes.append(cs)
            prefix_rates.append(cr)
        for k in range(1, min(gpu_num, 6) + 1):
            threshold = (k - 1) * S_eff_max
            idx = -1
            for t in range(len(prefix_sizes)):
                if prefix_sizes[t] > threshold:
                    idx = t
                    break
            if idx >= 0:
                numer = prefix_rates[idx]
                denom = k * S_eff_max - prefix_sizes[idx]
                if denom > 0 and numer > 0:
                    cand = numer / denom
                    if cand > lb_k:
                        lb_k = cand

        lower = max(0.0, lb1, lb2, lb_pair, lb_triplet, lb_k)
        return lower, False, infeasible_single

    lower, infeas_global, infeas_single = compute_lower_bound()
    if infeas_single or infeas_global or lower == float('inf'):
        # Fallback: greedy lookahead on remaining items using current placement/rem_mem
        return _greedy_fallback_with_prepack(placement, rem_mem, items, S)

    # Deterministic small RNG
    import random
    rng = random.Random(1337 + 7 * len(items) + 3 * gpu_num)

    # Evaluate max KVPR of a placement (including pre-placed mem-only)
    def eval_max_kvpr(placement_dict):
        max_v = 0.0
        for gid in range(gpu_num):
            used = 0.0
            R = 0.0
            for m in placement_dict.get(gid, []):
                used += float(getattr(m, 'model_size', 0.0))
                slo = float(getattr(m, 'slo', 0.0))
                if slo > 0:
                    R += float(getattr(m, 'req_rate', 0.0)) / slo
            v = kvpr(R, S - used)
            if v > max_v:
                max_v = v
        return max_v

    # Slack-equalization packer with hybrid rule and adaptive T retune
    def assign_balanced_slack(T, order='w_desc', seed_H=0, choose_rule='hybrid'):
        if T < 0:
            return None

        # Build enriched list for demand items
        enriched = []
        for (dR, sz, m) in items:
            w = dR + T * sz
            if w < 0:
                w = 0.0
            enriched.append([w, dR, sz, m])

        # Various orderings
        def sort_enriched(arr, Tcur):
            if order == 'w_desc':
                arr.sort(key=lambda x: x[0], reverse=True)
            elif order == 'intrinsic_desc':
                # dR / (S_eff_max - sz)
                arr.sort(key=lambda x: (x[1] / max(S_eff_max - x[2], 1e-9)), reverse=True)
            elif order == 'size_desc':
                arr.sort(key=lambda x: x[2], reverse=True)
            elif order == 'density_desc':
                arr.sort(key=lambda x: (x[1] / (x[2] if x[2] > 0 else 1e-9)), reverse=True)
            else:
                arr.sort(key=lambda x: x[0], reverse=True)

        sort_enriched(enriched, T)

        # Initialize per-GPU states from prepack
        used_mem = [S - rem_mem[g] for g in range(gpu_num)]
        sum_R = [0.0 for _ in range(gpu_num)]
        K = [T * (S - used_mem[g]) for g in range(gpu_num)]  # initial slack after pre-mem-only
        assign = {g: list(placement.get(g, [])) for g in range(gpu_num)}  # copy, includes mem-only

        # Optional seeding: spread H highest intrinsic-pressure items
        H = int(seed_H) if seed_H else 0
        working = list(enriched)
        if H > 0 and working:
            intrinsic = sorted(working, key=lambda x: (x[1] / max(S_eff_max - x[2], 1e-9)), reverse=True)
            seeds_list = intrinsic[:min(H, len(working))]
            # remove seeds from list
            ids = set(id(x) for x in seeds_list)
            remaining = [x for x in working if id(x) not in ids]
            for w, dR, sz, m in seeds_list:
                candidates = []
                for gid in range(gpu_num):
                    if used_mem[gid] + sz <= S + 1e-9 and K[gid] >= w - 1e-9:
                        # choose worst-fit on slack with tie-breaks
                        rem_after = S - (used_mem[gid] + sz)
                        new_kv = kvpr(sum_R[gid] + dR, rem_after) if rem_after > 0 else float('inf')
                        candidates.append((gid, K[gid], rem_after, new_kv))
                if not candidates:
                    return None
                candidates.sort(key=lambda t: (-t[1], -t[2], t[3]))
                chosen = candidates[0][0]
                assign[chosen].append(m)
                used_mem[chosen] += sz
                sum_R[chosen] += dR
                K[chosen] -= w
            working = remaining

        # Helper to compute J score
        def score_choice(gid, w, dR, sz, Tcur, alpha=0.15, beta=0.05):
            rem_after = S - (used_mem[gid] + sz)
            if rem_after <= 0:
                return None
            kv_new = kvpr(sum_R[gid] + dR, rem_after)
            K_after = K[gid] - w
            cap = Tcur * S if Tcur > 0 else 1.0
            K_after_norm = max(0.0, K_after) / cap
            avg_mem_frac = (sum(used_mem) + sz) / (gpu_num * S) if S > 0 else 0.0
            mem_after_frac = (used_mem[gid] + sz) / S if S > 0 else 0.0
            mem_imb = abs(mem_after_frac - avg_mem_frac)
            kv_new_norm = kv_new / max(Tcur, 1e-12) if Tcur > 0 else kv_new
            J = K_after_norm + alpha * kv_new_norm + beta * mem_imb
            return (J, kv_new, K_after, rem_after)

        # Placement with adaptive T retuning
        def place_sequence(seq, Tcur, allow_retune=False):
            nonlocal used_mem, sum_R, K, assign
            for w, dR, sz, m in seq:
                options = []
                for gid in range(gpu_num):
                    if used_mem[gid] + sz <= S + 1e-9 and K[gid] >= w - 1e-9:
                        if choose_rule == 'hybrid':
                            sc = score_choice(gid, w, dR, sz, Tcur)
                            if sc is None:
                                continue
                            options.append((gid, sc))
                        elif choose_rule == 'min_kvpr':
                            rem_after = S - (used_mem[gid] + sz)
                            if rem_after <= 0:
                                continue
                            kv_new = kvpr(sum_R[gid] + dR, rem_after)
                            K_after = K[gid] - w
                            options.append((gid, (kv_new, kv_new, K_after, rem_after)))
                        else:
                            K_after = K[gid] - w
                            options.append((gid, (K_after, 0.0, K_after, S - (used_mem[gid] + sz))))
                if not options:
                    return False, Tcur
                # Select best option
                if choose_rule == 'hybrid':
                    options.sort(key=lambda t: (t[1][0], t[1][1], -t[1][3]))  # J, then kv_new, then larger rem_after
                elif choose_rule == 'min_kvpr':
                    options.sort(key=lambda t: (t[1][0], t[1][2]))  # kv_new then K_after
                else:
                    options.sort(key=lambda t: (t[1][2], -t[1][3]))  # K_after then mem
                chosen = options[0][0]
                assign[chosen].append(m)
                used_mem[chosen] += sz
                sum_R[chosen] += dR
                K[chosen] -= w
            return True, Tcur

        # Phase A: place ~40%, then retune T based on residual instance
        cut = int(0.4 * len(working))
        Tcur = T
        if cut > 0:
            ok, _ = place_sequence(working[:cut], Tcur, allow_retune=False)
            if not ok:
                return None
            rem_items = working[cut:]
            if rem_items:
                rem_total_R = sum(x[1] for x in rem_items)
                rem_total_size = sum(x[2] for x in rem_items)
                free_sum = sum(S - used_mem[g] for g in range(gpu_num))
                denom = free_sum - rem_total_size
                if denom > 0 and rem_total_R > 0:
                    T2 = max(Tcur, rem_total_R / denom)
                    if T2 > Tcur + 1e-12:
                        # Update slack baseline to new T2 while preserving current assignments
                        for g in range(gpu_num):
                            # K_new = T2 * (S - used_mem[g]) - (sum_R[g] + T2 * 0) but we only tracked K implicitly:
                            # Old K = Tcur*(S - used_mem[g]) - (sum_R[g])
                            # Delta = (T2 - Tcur)*(S - used_mem[g])
                            delta = (T2 - Tcur) * (S - used_mem[g])
                            K[g] += delta
                        Tcur = T2
                # Rebuild weights and resort remaining items
                rem_enriched = []
                for _, dR, sz, m in rem_items:
                    rem_enriched.append([dR + Tcur * sz, dR, sz, m])
                sort_enriched(rem_enriched, Tcur)
                ok, _ = place_sequence(rem_enriched, Tcur, allow_retune=True)
                if not ok:
                    return None
        else:
            ok, _ = place_sequence(working, Tcur, allow_retune=False)
            if not ok:
                return None

        # Optional light second retune near 75% (no reordering, just T update)
        T3 = Tcur
        filled = len(items)
        if filled > 0:
            # Compute residual bound again
            free_sum = sum(S - used_mem[g] for g in range(gpu_num))
            # No residual items left to place; skip

        # Validate KVPR <= Tcur and memory <= S
        for g in range(gpu_num):
            if used_mem[g] - S > 1e-6:
                return None
            rem = S - used_mem[g]
            if rem <= 0 and sum_R[g] > 1e-12:
                return None
            if rem > 0 and Tcur > 0:
                if (sum_R[g] / rem) - Tcur > 1e-6:
                    return None
        return assign

    # Find the first feasible T by multiplicative sweep from lower bound
    def find_first_feasible_T(start_T):
        T = max(0.0, start_T)
        growth = 1.08
        for _ in range(40):
            for (order, seedH, rule) in [
                ('w_desc', min(4, max(1, gpu_num)), 'hybrid'),
                ('intrinsic_desc', min(3, max(1, gpu_num - 1)), 'hybrid'),
                ('w_desc', 0, 'min_kvpr'),
            ]:
                cand = assign_balanced_slack(T, order=order, seed_H=seedH, choose_rule=rule)
                if cand is not None:
                    return T, cand
            T = T * growth if T > 0 else 1e-6
        return None, None

    T_feas, placement_T = find_first_feasible_T(lower)
    if placement_T is None:
        # As a safety net
        return _greedy_fallback_with_prepack(placement, rem_mem, items, S)

    # Probe a small neighborhood around T_feas
    candidate_T = sorted(set([
        max(lower, T_feas * 0.985),
        max(lower, T_feas * 0.99),
        max(lower, T_feas * 1.0),
        max(lower, T_feas * 1.005),
        max(lower, T_feas * 1.01),
        0.5 * (lower + T_feas),
    ]))

    best_place = placement_T
    best_val = eval_max_kvpr(best_place)

    variants = [
        ('w_desc', min(4, max(1, gpu_num)), 'hybrid'),
        ('intrinsic_desc', min(3, max(1, gpu_num - 1)), 'hybrid'),
        ('density_desc', 0, 'hybrid'),
        ('size_desc', 0, 'hybrid'),
        ('w_desc', 0, 'min_kvpr'),
    ]

    for T in candidate_T:
        for (order, seedH, rule) in variants:
            cand = assign_balanced_slack(T, order=order, seed_H=seedH, choose_rule=rule)
            if cand is None:
                continue
            val = eval_max_kvpr(cand)
            if val + 1e-12 < best_val:
                best_val = val
                best_place = cand

    # Local refinement: adjust only demand-bearing models; keep mem-only frozen
    def local_refine(placement_dict, move_budget=20, swap_budget=10):
        buckets = {g: list(placement_dict.get(g, [])) for g in range(gpu_num)}
        # Identify mem-only to freeze
        frozen = set(id(m) for m in mem_only)
        used_mem = [0.0] * gpu_num
        sum_R = [0.0] * gpu_num
        for g in range(gpu_num):
            for m in buckets[g]:
                sz = float(getattr(m, 'model_size', 0.0))
                slo = float(getattr(m, 'slo', 0.0))
                used_mem[g] += sz
                if slo > 0:
                    sum_R[g] += float(getattr(m, 'req_rate', 0.0)) / slo

        def all_kvprs():
            return [kvpr(sum_R[g], S - used_mem[g]) for g in range(gpu_num)]

        def can_move(m):
            return id(m) not in frozen and float(getattr(m, 'slo', 0.0)) > 0

        def apply_move(src, dst, m):
            dR = float(getattr(m, 'req_rate', 0.0)) / float(getattr(m, 'slo', 1.0))
            sz = float(getattr(m, 'model_size', 0.0))
            buckets[src].remove(m); buckets[dst].append(m)
            sum_R[src] -= dR; sum_R[dst] += dR
            used_mem[src] -= sz; used_mem[dst] += sz

        moves_left = move_budget
        swaps_left = swap_budget

        while moves_left > 0 or swaps_left > 0:
            kvprs = all_kvprs()
            cur_max = max(kvprs) if kvprs else 0.0
            max_gid = max(range(gpu_num), key=lambda g: kvprs[g]) if kvprs else 0

            improved = False
            best_move = None  # (src, dst, m, resulting_max)

            # Try demand-only moves from worst GPU
            for m in list(buckets[max_gid]):
                if not can_move(m):
                    continue
                dR = float(getattr(m, 'req_rate', 0.0)) / float(getattr(m, 'slo', 1.0))
                sz = float(getattr(m, 'model_size', 0.0))
                src_R_new = sum_R[max_gid] - dR
                src_mem_new = used_mem[max_gid] - sz
                src_kv_new = kvpr(src_R_new, S - src_mem_new) if (S - src_mem_new) > 0 else float('inf')
                for dst in range(gpu_num):
                    if dst == max_gid:
                        continue
                    if used_mem[dst] + sz > S + 1e-9:
                        continue
                    dst_R_new = sum_R[dst] + dR
                    dst_mem_new = used_mem[dst] + sz
                    dst_kv_new = kvpr(dst_R_new, S - dst_mem_new) if (S - dst_mem_new) > 0 else float('inf')
                    resulting = max(dst_kv_new, src_kv_new)
                    for g in range(gpu_num):
                        if g == max_gid or g == dst:
                            continue
                        if kvprs[g] > resulting:
                            resulting = kvprs[g]
                    if resulting + 1e-12 < cur_max:
                        if best_move is None or resulting < best_move[3]:
                            best_move = (max_gid, dst, m, resulting)
            if best_move is not None and moves_left > 0:
                s, d, m, _ = best_move
                apply_move(s, d, m)
                moves_left -= 1
                improved = True
            else:
                # Try limited swaps (demand-only)
                best_swap = None  # (src, dst, a, b, resulting_max)
                if swaps_left > 0:
                    cap_a = min(10, len(buckets[max_gid]))
                    for a in list(buckets[max_gid])[:cap_a]:
                        if not can_move(a):
                            continue
                        aR = float(getattr(a, 'req_rate', 0.0)) / float(getattr(a, 'slo', 1.0))
                        aS = float(getattr(a, 'model_size', 0.0))
                        for dst in range(gpu_num):
                            if dst == max_gid or not buckets[dst]:
                                continue
                            cap_b = min(10, len(buckets[dst]))
                            for b in list(buckets[dst])[:cap_b]:
                                if not can_move(b):
                                    continue
                                bR = float(getattr(b, 'req_rate', 0.0)) / float(getattr(b, 'slo', 1.0))
                                bS = float(getattr(b, 'model_size', 0.0))
                                mem_src_new = used_mem[max_gid] - aS + bS
                                mem_dst_new = used_mem[dst] - bS + aS
                                if mem_src_new > S + 1e-9 or mem_dst_new > S + 1e-9:
                                    continue
                                src_R_new = sum_R[max_gid] - aR + bR
                                dst_R_new = sum_R[dst] - bR + aR
                                kv_src_new = kvpr(src_R_new, S - mem_src_new) if (S - mem_src_new) > 0 else float('inf')
                                kv_dst_new = kvpr(dst_R_new, S - mem_dst_new) if (S - mem_dst_new) > 0 else float('inf')
                                resulting = max(kv_src_new, kv_dst_new)
                                for g in range(gpu_num):
                                    if g == max_gid or g == dst:
                                        continue
                                    if kvprs[g] > resulting:
                                        resulting = kvprs[g]
                                if resulting + 1e-12 < cur_max:
                                    if best_swap is None or resulting < best_swap[4]:
                                        best_swap = (max_gid, dst, a, b, resulting)
                if best_swap is not None:
                    src, dst, a, b, _ = best_swap
                    buckets[src].remove(a); buckets[src].append(b)
                    buckets[dst].remove(b); buckets[dst].append(a)
                    aR = float(getattr(a, 'req_rate', 0.0)) / float(getattr(a, 'slo', 1.0))
                    bR = float(getattr(b, 'req_rate', 0.0)) / float(getattr(b, 'slo', 1.0))
                    aS = float(getattr(a, 'model_size', 0.0))
                    bS = float(getattr(b, 'model_size', 0.0))
                    sum_R[src] = sum_R[src] - aR + bR
                    sum_R[dst] = sum_R[dst] - bR + aR
                    used_mem[src] = used_mem[src] - aS + bS
                    used_mem[dst] = used_mem[dst] - bS + aS
                    swaps_left -= 1
                    improved = True

            if not improved:
                break
        return buckets

    refined = local_refine(best_place, move_budget=20, swap_budget=10)

    # Memory safety check
    for g in range(gpu_num):
        mem = sum(float(getattr(m, 'model_size', 0.0)) for m in refined.get(g, []))
        if mem - S > 1e-6:
            return best_place

    return refined


def _greedy_fallback_with_prepack(pre_placement, rem_mem, items, S):
    """
    Simple greedy fallback: place remaining demand items by minimizing the
    projected max KVPR increase, respecting current pre-placement and memory.
    items: list of (dR, size, obj) for demand-bearing models
    """
    from copy import deepcopy
    placement = {g: list(pre_placement.get(g, [])) for g in range(len(rem_mem))}
    rem = list(rem_mem)
    sum_R = [0.0 for _ in range(len(rem))]
    # Compute current R from pre-placed (should be 0 for mem-only)
    for g in range(len(rem)):
        for m in placement[g]:
            slo = float(getattr(m, 'slo', 0.0))
            if slo > 0:
                sum_R[g] += float(getattr(m, 'req_rate', 0.0)) / slo

    def kvpr(R, rem_mem):
        if rem_mem <= 0:
            return float('inf')
        return R / rem_mem

    for dR, size, m in sorted(items, key=lambda x: (x[0] / max(S - x[1], 1e-9), x[1]), reverse=True):
        best_gid = None
        best_result = float('inf')
        best_new_k = float('inf')
        best_rem_after = -1.0
        current = [kvpr(sum_R[g], rem[g]) for g in range(len(rem))]
        for g in range(len(rem)):
            if size <= rem[g] + 1e-12:
                new_R = sum_R[g] + dR
                new_rem = rem[g] - size
                if new_rem <= 0:
                    continue
                new_k = kvpr(new_R, new_rem)
                projected_max = max(new_k, max(current[:g] + current[g+1:])) if len(current) > 1 else new_k
                if (projected_max < best_result or
                    (projected_max == best_result and (new_k < best_new_k or
                                                       (new_k == best_new_k and new_rem > best_rem_after)))):
                    best_result = projected_max
                    best_new_k = new_k
                    best_rem_after = new_rem
                    best_gid = g
        if best_gid is None:
            raise ValueError(
                f"Unable to place demand model of size {size} GB. Remaining per-GPU memory: {rem}"
            )
        placement[best_gid].append(m)
        sum_R[best_gid] += dR
        rem[best_gid] -= size

    return placement
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