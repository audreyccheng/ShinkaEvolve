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
        A placement of models to GPUs (dict: gpu_id -> list of model objects)
    """
    # Trivial cases
    placement_empty = {i: [] for i in range(gpu_num)}
    if gpu_num <= 0 or not models:
        return placement_empty

    S = GPU_MEM_SIZE

    # Safe KVPR computation
    def kvpr(R, rem_mem):
        if rem_mem <= 0:
            return float('inf')
        return R / rem_mem

    # -------------------------
    # Data preparation utility
    # -------------------------
    def to_items(ms):
        # Normalize slo==0 as zero effective demand to avoid infinities destabilizing packing
        items = []
        total_R = 0.0
        total_size = 0.0
        for m in ms:
            slo = float(getattr(m, "slo", 0.0))
            dR = (float(m.req_rate) / slo) if slo != 0.0 else 0.0
            sz = float(m.model_size)
            items.append({"obj": m, "dR": dR, "size": sz})
            total_R += dR
            total_size += sz
        return items, total_R, total_size

    items, total_R, total_size = to_items(models)

    # -------------------------
    # Lower bound computation
    # -------------------------
    def compute_lower_bound(items, gpu_num, S):
        # Per-item bound: dR/(S - s)
        lb_single = 0.0
        infeasible_single = False
        for it in items:
            dR = it["dR"]; s = it["size"]
            denom = S - s
            if denom <= 0:
                if dR > 0:
                    infeasible_single = True
                continue
            if dR > 0:
                cand = dR / denom
                if cand > lb_single:
                    lb_single = cand

        # Global bound from totals
        total_R = sum(it["dR"] for it in items)
        total_size = sum(it["size"] for it in items)
        denom2 = gpu_num * S - total_size
        if denom2 <= 0 and total_R > 0:
            return float("inf"), True, infeasible_single
        lb_global = 0.0 if (denom2 <= 0 or total_R <= 0) else (total_R / denom2)

        # Pair bound: for pairs that cannot co-reside
        lb_pair = 0.0
        P = min(len(items), 120)
        by_size = sorted(items, key=lambda x: x["size"], reverse=True)[:P]
        for i in range(len(by_size)):
            si = by_size[i]["size"]; ri = by_size[i]["dR"]
            for j in range(i + 1, len(by_size)):
                sj = by_size[j]["size"]; rj = by_size[j]["dR"]
                if si + sj > S:
                    denom = 2 * S - (si + sj)
                    if denom > 0:
                        cand = (ri + rj) / denom
                        if cand > lb_pair:
                            lb_pair = cand

        # Light triplet bound on a small subset (top by size)
        lb_triplet = 0.0
        if len(by_size) >= 3:
            topK = by_size[:8]
            for i in range(len(by_size)):
                si = by_size[i]["size"]; ri = by_size[i]["dR"]
                for j in range(i + 1, len(by_size)):
                    sj = by_size[j]["size"]; rj = by_size[j]["dR"]
                    for mk in topK:
                        if mk is by_size[i] or mk is by_size[j]:
                            continue
                        sk = mk["size"]; rk = mk["dR"]
                        ssum = si + sj + sk
                        if ssum > 2 * S:
                            denom = 3 * S - ssum
                            if denom > 0:
                                cand = (ri + rj + rk) / denom
                                if cand > lb_triplet:
                                    lb_triplet = cand

        # k-prefix bound for k up to 6
        lb_k = 0.0
        sorted_by_size = sorted(items, key=lambda it: it["size"], reverse=True)
        prefix_sizes, prefix_rates = [], []
        cs, cr = 0.0, 0.0
        for it in sorted_by_size:
            cs += it["size"]; cr += it["dR"]
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

        lower = max(0.0, lb_single, lb_global, lb_pair, lb_triplet, lb_k)
        return lower, False, infeasible_single

    lower, infeasible_global, infeasible_single = compute_lower_bound(items, gpu_num, S)
    if infeasible_global or infeasible_single or lower == float("inf"):
        # Fallback: greedy minimizing resulting max KVPR with capacity checks
        return _greedy_minmax(gpu_num, models, S)

    # -------------------------
    # Deterministic RNG
    # -------------------------
    import random
    rng = random.Random(len(items) * 65537 + gpu_num * 101)

    # -------------------------
    # Balanced-slack packer with two-phase T update
    # -------------------------
    # K_g = T*S - (sumR_g + T*used_mem_g), each item weight is w = dR + T*size.
    def assign_balanced_slack(T, order="w_desc", choose_rule="tight", seeds=1, seed_H=0):
        if T < 0:
            return None
        # Prepare enriched items
        enriched = []
        for it in items:
            dR = it["dR"]; sz = it["size"]
            w = dR + T * sz
            if w < 0:  # guard against negative due to numerical artifacts
                w = 0.0
            enriched.append([w, dR, sz, it["obj"]])

        # Ordering helper
        def sort_enriched(arr):
            if order == "w_desc":
                arr.sort(key=lambda x: x[0], reverse=True)
            elif order == "intrinsic_desc":
                arr.sort(key=lambda x: (x[1] / max(S - x[2], 1e-9)), reverse=True)
            elif order == "size_desc":
                arr.sort(key=lambda x: x[2], reverse=True)
            elif order == "density_desc":
                arr.sort(key=lambda x: (x[1] / max(x[2], 1e-9)), reverse=True)
            else:
                arr.sort(key=lambda x: x[0], reverse=True)

        sort_enriched(enriched)

        best_assign = None
        best_val = float("inf")

        for rep in range(max(1, seeds)):
            # Initialize per-GPU state
            used_mem = [0.0] * gpu_num
            sum_R = [0.0] * gpu_num
            K = [T * S] * gpu_num
            assign = {i: [] for i in range(gpu_num)}
            T_current = T

            # Optional seeding: spread H hardest items (by intrinsic pressure) using worst-fit on K
            H = int(seed_H) if seed_H else 0
            seq = list(enriched)
            if H > 0 and len(seq) > 0:
                intr_sorted = sorted(seq, key=lambda x: (x[1] / max(S - x[2], 1e-9)), reverse=True)
                seeds_list = intr_sorted[:min(H, len(seq))]
                seed_ids = set(id(x) for x in seeds_list)
                remainder = [x for x in seq if id(x) not in seed_ids]
                for w, dR, sz, m in seeds_list:
                    candidates = []
                    for g in range(gpu_num):
                        if used_mem[g] + sz <= S + 1e-9 and K[g] >= w - 1e-9:
                            candidates.append(g)
                    if not candidates:
                        assign = None
                        break
                    candidates.sort(key=lambda g: (-K[g], -(S - (used_mem[g] + sz))))
                    chosen = candidates[0]
                    assign[chosen].append(m)
                    used_mem[chosen] += sz
                    sum_R[chosen] += dR
                    K[chosen] -= w
                if assign is None:
                    continue
                seq = remainder
            else:
                seq = list(enriched)

            # Core placement with two-phase T update at 40% progress
            def place_sequence(seq_items):
                nonlocal T_current
                for w, dR, sz, m in seq_items:
                    options = []
                    for g in range(gpu_num):
                        if used_mem[g] + sz <= S + 1e-9 and K[g] >= w - 1e-9:
                            # Score choices
                            K_after = K[g] - w
                            if choose_rule in ("min_kvpr", "hybrid"):
                                rem_after = S - (used_mem[g] + sz)
                                if rem_after <= 0:
                                    continue
                                kv_new = kvpr(sum_R[g] + dR, rem_after)
                                if choose_rule == "hybrid":
                                    # Hybrid objective to stabilize residual KVPR and memory balance
                                    cap = max(T_current * S, 1e-12)
                                    K_after_norm = max(0.0, K_after) / cap
                                    avg_mem_frac = (sum(used_mem) + sz) / (gpu_num * S) if S > 0 else 0.0
                                    mem_after_frac = (used_mem[g] + sz) / S if S > 0 else 0.0
                                    mem_imbalance = abs(mem_after_frac - avg_mem_frac)
                                    alpha = 0.15
                                    beta = 0.05
                                    kv_new_norm = kv_new / max(T_current, 1e-12)
                                    hybrid_score = K_after_norm + alpha * kv_new_norm + beta * mem_imbalance
                                    options.append((g, K_after, kv_new, hybrid_score))
                                else:
                                    options.append((g, K_after, kv_new, None))
                            else:
                                options.append((g, K_after, None, None))
                    if not options:
                        return False
                    if choose_rule == "min_kvpr":
                        options.sort(key=lambda t: (t[1], t[2]))
                    elif choose_rule == "hybrid":
                        options.sort(key=lambda t: (t[1], t[3], t[2]))
                    else:
                        options.sort(key=lambda t: t[1])
                    chosen = options[0][0]
                    assign[chosen].append(m)
                    used_mem[chosen] += sz
                    sum_R[chosen] += dR
                    K[chosen] -= w
                return True

            seq_all = seq
            n = len(seq_all)
            cut = int(0.4 * n)
            ok = True
            if cut > 0:
                ok = place_sequence(seq_all[:cut])
                if not ok:
                    assign = None
                else:
                    # Recompute a residual lower bound using remaining items and residual capacities
                    rem_items = seq_all[cut:]
                    if rem_items:
                        rem_total_R = sum(x[1] for x in rem_items)
                        rem_total_size = sum(x[2] for x in rem_items)
                        free_total = sum(S - used_mem[g] for g in range(gpu_num))
                        denom = free_total - rem_total_size
                        if denom > 0 and rem_total_R > 0:
                            T2 = rem_total_R / denom
                            if T2 > T_current:
                                # Adjust per-GPU slack to reflect T increase
                                delta = T2 - T_current
                                for g in range(gpu_num):
                                    K[g] += delta * (S - used_mem[g])
                                T_current = T2
                        # Rebuild weights for remaining items and re-sort
                        rem_enriched = []
                        for _, dR, sz, m in rem_items:
                            rem_enriched.append([dR + T_current * sz, dR, sz, m])
                        sort_enriched(rem_enriched)
                        ok = place_sequence(rem_enriched)
                        if not ok:
                            assign = None
            else:
                ok = place_sequence(seq_all)
                if not ok:
                    assign = None

            if assign is None:
                continue

            # Validate constraints strictly
            feas = True
            for g in range(gpu_num):
                if used_mem[g] - S > 1e-6:
                    feas = False; break
                rem = S - used_mem[g]
                if rem < 0:
                    feas = False; break
                if rem == 0 and sum_R[g] > 0:
                    feas = False; break
                if T_current > 0 and (sum_R[g] / max(rem, 1e-12)) - T_current > 1e-6:
                    feas = False; break
            if not feas:
                continue

            # Score by actual max KVPR
            cur = 0.0
            for g in range(gpu_num):
                cur = max(cur, kvpr(sum_R[g], S - used_mem[g]))
            if cur < best_val:
                best_val = cur
                best_assign = {i: list(assign.get(i, [])) for i in range(gpu_num)}

        return best_assign

    # -------------------------
    # First-feasible T via multiplicative sweep
    # -------------------------
    def find_first_feasible_T(start_T):
        T = max(0.0, start_T)
        growth = 1.08
        for _ in range(44):
            for (order, choose_rule, H, seeds) in [
                ("w_desc", "tight", min(3, max(1, gpu_num)), 1),
                ("intrinsic_desc", "tight", min(2, max(1, gpu_num - 1)), 1),
                ("w_desc", "hybrid", min(2, max(1, gpu_num - 1)), 1),
                ("w_desc", "min_kvpr", 0, 1),
            ]:
                cand = assign_balanced_slack(T, order=order, choose_rule=choose_rule, seeds=seeds, seed_H=H)
                if cand is not None:
                    return T, cand
            T = T * growth if T > 0 else 1e-6
        return None, None

    # -------------------------
    # Evaluation util
    # -------------------------
    def eval_max_kvpr(placement):
        max_v = 0.0
        for gid in range(gpu_num):
            bucket = placement.get(gid, [])
            used = sum(float(m.model_size) for m in bucket)
            R = sum((float(m.req_rate) / float(m.slo)) if float(getattr(m, "slo", 0.0)) != 0.0 else 0.0 for m in bucket)
            max_v = max(max_v, kvpr(R, S - used))
        return max_v

    # -------------------------
    # Feasible seed placement and neighborhood exploration
    # -------------------------
    T_feas, placement_at_T = find_first_feasible_T(lower)
    if placement_at_T is None:
        return _greedy_minmax(gpu_num, models, S)

    best_placement = placement_at_T
    best_val = eval_max_kvpr(best_placement)

    # Neighbor T values (wider but small set)
    candidates_T = []
    for mul in [0.985, 0.99, 1.0, 1.005, 1.01, 1.015]:
        candidates_T.append(max(lower, T_feas * mul))
    candidates_T.append(0.5 * (lower + T_feas))
    candidates_T = sorted(set(round(t, 12) for t in candidates_T))

    variants = [
        ("w_desc", "tight", 2),
        ("intrinsic_desc", "tight", 1),
        ("w_desc", "hybrid", 2),
        ("density_desc", "tight", 1),
        ("w_desc", "min_kvpr", 1),
        ("size_desc", "tight", 1),
    ]

    for T in candidates_T:
        for order, rule, seeds in variants:
            cand = assign_balanced_slack(T, order=order, choose_rule=rule, seeds=seeds, seed_H=min(3, max(1, gpu_num - 1)))
            if cand is None:
                continue
            val = eval_max_kvpr(cand)
            if (val < best_val) or (val == best_val and rng.random() < 0.1):
                best_val = val
                best_placement = cand

    # -------------------------
    # Local refinement: moves, 2-opt swaps, and short eject chains
    # -------------------------
    def refine(placement, move_budget=18, swap_budget=12, eject_budget=6):
        buckets = {gid: list(placement.get(gid, [])) for gid in range(gpu_num)}
        used_mem = [0.0] * gpu_num
        sum_R = [0.0] * gpu_num
        for gid in range(gpu_num):
            for m in buckets[gid]:
                used_mem[gid] += float(m.model_size)
                slo = float(getattr(m, "slo", 0.0))
                sum_R[gid] += (float(m.req_rate) / slo) if slo != 0.0 else 0.0

        def kvprs_all():
            return [kvpr(sum_R[g], S - used_mem[g]) for g in range(gpu_num)]

        def apply_move(src, dst, mdl):
            dR = (float(mdl.req_rate) / float(mdl.slo)) if float(getattr(mdl, "slo", 0.0)) != 0.0 else 0.0
            s = float(mdl.model_size)
            buckets[src].remove(mdl); buckets[dst].append(mdl)
            sum_R[src] -= dR; sum_R[dst] += dR
            used_mem[src] -= s; used_mem[dst] += s

        # 1) Single-item moves from max-pressured GPU
        moves_left = move_budget
        swaps_left = swap_budget
        eject_left = eject_budget
        improved_any = True
        while (moves_left > 0 or swaps_left > 0 or eject_left > 0) and improved_any:
            improved_any = False
            kvprs = kvprs_all()
            cur_max = max(kvprs)
            worst = max(range(gpu_num), key=lambda g: kvprs[g])

            # Try best move
            best_move = None  # (src, dst, model, resulting)
            for mdl in list(buckets[worst]):
                dR = (float(mdl.req_rate) / float(mdl.slo)) if float(getattr(mdl, "slo", 0.0)) != 0.0 else 0.0
                s = float(mdl.model_size)
                R_src_new = sum_R[worst] - dR
                mem_src_new = used_mem[worst] - s
                rem_src = S - mem_src_new
                if rem_src <= 0:
                    continue
                kv_src_new = kvpr(R_src_new, rem_src)
                for dst in range(gpu_num):
                    if dst == worst:
                        continue
                    if used_mem[dst] + s > S:
                        continue
                    rem_dst = S - (used_mem[dst] + s)
                    if rem_dst <= 0:
                        continue
                    R_dst_new = sum_R[dst] + dR
                    kv_dst_new = kvpr(R_dst_new, rem_dst)
                    resulting = max(kv_dst_new, kv_src_new)
                    for g in range(gpu_num):
                        if g == worst or g == dst:
                            continue
                        resulting = max(resulting, kvprs[g])
                    if resulting + 1e-12 < cur_max:
                        if best_move is None or resulting < best_move[3]:
                            best_move = (worst, dst, mdl, resulting)
            if best_move is not None and moves_left > 0:
                apply_move(*best_move[:3])
                moves_left -= 1
                improved_any = True
                continue

            # 2) 2-opt swaps between two most pressured GPUs
            if swaps_left > 0:
                order = sorted(range(gpu_num), key=lambda g: kvprs[g], reverse=True)
                if len(order) >= 2:
                    a_gid, b_gid = order[0], order[1]
                    best_swap = None  # (src, dst, a, b, resulting)
                    cap_a = min(10, len(buckets[a_gid]))
                    cap_b = min(10, len(buckets[b_gid]))
                    for a in list(buckets[a_gid])[:cap_a]:
                        aR = (float(a.req_rate) / float(a.slo)) if float(getattr(a, "slo", 0.0)) != 0.0 else 0.0
                        aS = float(a.model_size)
                        for b in list(buckets[b_gid])[:cap_b]:
                            bR = (float(b.req_rate) / float(b.slo)) if float(getattr(b, "slo", 0.0)) != 0.0 else 0.0
                            bS = float(b.model_size)
                            mem_a_new = used_mem[a_gid] - aS + bS
                            mem_b_new = used_mem[b_gid] - bS + aS
                            if mem_a_new > S or mem_b_new > S:
                                continue
                            rem_a = S - mem_a_new; rem_b = S - mem_b_new
                            if rem_a <= 0 or rem_b <= 0:
                                continue
                            R_a_new = sum_R[a_gid] - aR + bR
                            R_b_new = sum_R[b_gid] - bR + aR
                            kv_a_new = kvpr(R_a_new, rem_a)
                            kv_b_new = kvpr(R_b_new, rem_b)
                            resulting = max(kv_a_new, kv_b_new)
                            for g in range(gpu_num):
                                if g == a_gid or g == b_gid:
                                    continue
                                resulting = max(resulting, kvprs[g])
                            if resulting + 1e-12 < cur_max:
                                if best_swap is None or resulting < best_swap[4]:
                                    best_swap = (a_gid, b_gid, a, b, resulting)
                    if best_swap is not None:
                        src, dst, a, b, _ = best_swap
                        # Apply swap
                        buckets[src].remove(a); buckets[dst].append(a)
                        buckets[dst].remove(b); buckets[src].append(b)
                        aR = (float(a.req_rate) / float(a.slo)) if float(getattr(a, "slo", 0.0)) != 0.0 else 0.0
                        bR = (float(b.req_rate) / float(b.slo)) if float(getattr(b, "slo", 0.0)) != 0.0 else 0.0
                        aS = float(a.model_size); bS = float(b.model_size)
                        sum_R[src] = sum_R[src] - aR + bR
                        sum_R[dst] = sum_R[dst] - bR + aR
                        used_mem[src] = used_mem[src] - aS + bS
                        used_mem[dst] = used_mem[dst] - bS + aS
                        swaps_left -= 1
                        improved_any = True
                        continue

            # 3) Short eject chains (length-2): free room on a destination to move a heavy model from worst
            if eject_left > 0:
                improved_chain = False
                for mdl in list(buckets[worst])[:8]:
                    dR_a = (float(mdl.req_rate) / float(mdl.slo)) if float(getattr(mdl, "slo", 0.0)) != 0.0 else 0.0
                    sA = float(mdl.model_size)
                    # Try each destination that is close but lacks memory
                    for dst in range(gpu_num):
                        if dst == worst:
                            continue
                        need = max(0.0, (used_mem[dst] + sA) - S)
                        if need <= 1e-9:
                            continue  # this is a normal move already handled
                        # Find a small model b in dst to eject to some k
                        best_chain = None  # (dst_model, k, resulting_max)
                        for b in list(buckets[dst])[:8]:
                            sB = float(b.model_size)
                            if sB + 1e-9 < need:
                                continue
                            dR_b = (float(b.req_rate) / float(b.slo)) if float(getattr(b, "slo", 0.0)) != 0.0 else 0.0
                            for k in range(gpu_num):
                                if k == dst or k == worst:
                                    continue
                                if used_mem[k] + sB > S:
                                    continue
                                # Simulate chain: dst->k for b, worst->dst for mdl
                                # Build new states' KVPR quickly
                                kvprs = kvprs_all()
                                # After dst ejects b
                                R_dst_tmp = sum_R[dst] - dR_b
                                mem_dst_tmp = used_mem[dst] - sB
                                # After worst sends mdl to dst
                                if mem_dst_tmp + sA > S:
                                    continue
                                R_worst_tmp = sum_R[worst] - dR_a
                                mem_worst_tmp = used_mem[worst] - sA
                                rem_dst = S - (mem_dst_tmp + sA)
                                rem_worst = S - mem_worst_tmp
                                if rem_dst <= 0 or rem_worst <= 0 or (S - (used_mem[k] + sB)) <= 0:
                                    continue
                                kv_dst_new = kvpr(R_dst_tmp + dR_a, rem_dst)
                                kv_worst_new = kvpr(R_worst_tmp, rem_worst)
                                kv_k_new = kvpr(sum_R[k] + dR_b, S - (used_mem[k] + sB))
                                resulting = max(kv_dst_new, kv_worst_new, kv_k_new)
                                for g in range(gpu_num):
                                    if g in (worst, dst, k):
                                        continue
                                    resulting = max(resulting, kvprs[g])
                                if resulting + 1e-12 < cur_max:
                                    if best_chain is None or resulting < best_chain[2]:
                                        best_chain = (b, k, resulting)
                        if best_chain is not None:
                            # Apply chain
                            b, k, _ = best_chain
                            # dst -> k: move b
                            buckets[dst].remove(b); buckets[k].append(b)
                            dR_b = (float(b.req_rate) / float(b.slo)) if float(getattr(b, "slo", 0.0)) != 0.0 else 0.0
                            sB = float(b.model_size)
                            sum_R[dst] -= dR_b; used_mem[dst] -= sB
                            sum_R[k] += dR_b; used_mem[k] += sB
                            # worst -> dst: move mdl
                            buckets[worst].remove(mdl); buckets[dst].append(mdl)
                            sum_R[worst] -= dR_a; used_mem[worst] -= sA
                            sum_R[dst] += dR_a; used_mem[dst] += sA
                            eject_left -= 1
                            improved_chain = True
                            break
                    if improved_chain:
                        break
                if improved_chain:
                    improved_any = True
                    continue

            # No improvement possible
            break

        return buckets

    refined = refine(best_placement, move_budget=18, swap_budget=12, eject_budget=6)

    # Final sanity: memory safety guard
    for gid in range(gpu_num):
        used = sum(float(m.model_size) for m in refined.get(gid, []))
        if used - S > 1e-6:
            return best_placement

    return refined


# -------------------------
# Greedy fallback (kept outside to avoid inner-reference issues)
# -------------------------
def _greedy_minmax(gpu_num, models, S):
    # Sort by high pressure weight then size
    def pressure_weight(m):
        denom = S - float(m.model_size)
        if denom <= 0:
            return float("inf")
        slo = float(getattr(m, "slo", 0.0))
        dR = (float(m.req_rate) / slo) if slo != 0.0 else 0.0
        return dR / denom

    sorted_models = sorted(models, key=lambda m: (pressure_weight(m), m.model_size), reverse=True)
    placement = {i: [] for i in range(gpu_num)}
    rem_mem = [S] * gpu_num
    sum_R = [0.0] * gpu_num

    for m in sorted_models:
        slo = float(getattr(m, "slo", 0.0))
        dR = (float(m.req_rate) / slo) if slo != 0.0 else 0.0
        size = float(m.model_size)
        best_gid = None
        best_res = float("inf")
        best_k = float("inf")
        best_rem = -1.0
        current_k = [ (sum_R[i] / rem_mem[i]) if rem_mem[i] > 0 else float('inf') for i in range(gpu_num) ]
        for gid in range(gpu_num):
            if size <= rem_mem[gid]:
                rem_after = rem_mem[gid] - size
                if rem_after <= 0:
                    continue
                k_new = (sum_R[gid] + dR) / rem_after
                # resulting global max after put
                res = k_new
                for j in range(gpu_num):
                    if j == gid:
                        continue
                    if current_k[j] > res:
                        res = current_k[j]
                if (res < best_res or
                    (res == best_res and k_new < best_k) or
                    (res == best_res and k_new == best_k and rem_after > best_rem)):
                    best_res = res; best_k = k_new; best_rem = rem_after; best_gid = gid
        if best_gid is None:
            raise ValueError(
                f"Unable to place model of size {m.model_size} GB on any GPU. "
                f"Remaining per-GPU memory: {rem_mem}"
            )
        placement[best_gid].append(m)
        rem_mem[best_gid] -= size
        sum_R[best_gid] += dR
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