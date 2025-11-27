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

    # Extract per-model attributes once
    items = []
    total_R = 0.0
    total_size = 0.0
    for m in models:
        dR = (m.req_rate / m.slo) if m.slo != 0 else float('inf')
        s = m.model_size
        items.append({
            'obj': m,
            'dR': float(dR),
            'size': float(s),
        })
        total_R += float(dR)
        total_size += float(s)

    # Helper: compute KVPR safely
    def kvpr(R, rem_mem):
        if rem_mem <= 0:
            return float('inf')
        return R / rem_mem

    # Compute tight lower bounds on optimal T
    def compute_lower_bound():
        # Per-item bound: T >= dR / (S - s)
        lb1 = 0.0
        infeasible_single = False
        for it in items:
            s = it['size']
            dR = it['dR']
            denom = S - s
            if denom <= 0:
                if dR > 0:
                    infeasible_single = True
                continue
            if dR > 0:
                cand = dR / denom
                if cand > lb1:
                    lb1 = cand

        # Global bound from totals: T >= total_R / (gpu_num*S - total_size)
        denom2 = gpu_num * S - total_size
        lb2 = 0.0
        if denom2 > 0 and total_R > 0:
            lb2 = total_R / denom2
        elif denom2 <= 0 and total_R > 0:
            # Not enough aggregate memory to leave any KV space; fall back later
            lb2 = float('inf')

        # Pair bound for large pairs that cannot co-reside (s_i + s_j > S)
        # Then at least two GPUs are needed: T >= (dR_i + dR_j) / (2S - (s_i + s_j))
        lb_pair = 0.0
        # Consider only top P by size to keep O(P^2) small if many items
        P = min(len(items), 200)
        by_size = sorted(items, key=lambda x: x['size'], reverse=True)[:P]
        for i in range(len(by_size)):
            si = by_size[i]['size']
            ri = by_size[i]['dR']
            for j in range(i + 1, len(by_size)):
                sj = by_size[j]['size']
                rj = by_size[j]['dR']
                if si + sj > S:
                    denom = 2 * S - (si + sj)
                    if denom > 0:
                        cand = (ri + rj) / denom
                        if cand > lb_pair:
                            lb_pair = cand

        # k-bin prefix bound for k in {1..min(gpu_num,4)}:
        # Sort by size desc, find shortest prefix whose total size > (k-1)*S, then
        # T >= sum(dR_prefix) / max(k*S - sum(size_prefix), eps)
        lb_k = 0.0
        max_k = min(gpu_num, 4)
        sorted_by_size = sorted(items, key=lambda x: x['size'], reverse=True)
        prefix_sizes = []
        prefix_rates = []
        cum_s = 0.0
        cum_r = 0.0
        for it in sorted_by_size:
            cum_s += it['size']
            cum_r += it['dR']
            prefix_sizes.append(cum_s)
            prefix_rates.append(cum_r)
        for k in range(1, max_k + 1):
            # Find shortest prefix with sum(size) > (k-1)*S
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

        lower = max(0.0, lb1, lb2, lb_pair, lb_k)
        return lower, (lb2 == float('inf')), infeasible_single

    lower, infeasible_global, infeasible_single = compute_lower_bound()
    if infeasible_single or infeasible_global:
        # Fallback simple greedy minimizes current KVPR, memory-feasible or error if impossible
        return _simple_kvpr_greedy(gpu_num, models, S)

    # Utility: evaluate placement and max KVPR
    def evaluate_max_kvpr(placement_dict):
        max_v = 0.0
        for gid in range(gpu_num):
            bucket = placement_dict.get(gid, [])
            used = 0.0
            R = 0.0
            for m in bucket:
                used += float(m.model_size)
                R += float(m.req_rate / m.slo)
            v = kvpr(R, S - used)
            if v > max_v:
                max_v = v
        return max_v

    # Pack for a given T using transformed capacity, with small seeding and a few variants
    def pack_for_T(T, variant='bfd', order='w_desc', seed_H=0, rng=None, choose_rule='best_fit'):
        # capacity in transformed space
        cap = T * S
        # Build item weights
        enriched = []
        for it in items:
            w = it['dR'] + T * it['size']
            if w < 0:
                w = 0.0
            enriched.append((w, it['dR'], it['size'], it['obj']))
        # Orderings
        if order == 'w_desc':
            enriched.sort(key=lambda x: x[0], reverse=True)
        elif order == 'density_desc':  # r/s per mem
            enriched.sort(key=lambda x: (x[1] / (x[2] if x[2] > 0 else 1e-9)), reverse=True)
        elif order == 'size_desc':
            enriched.sort(key=lambda x: x[2], reverse=True)
        else:
            enriched.sort(key=lambda x: x[0], reverse=True)

        assign = {i: [] for i in range(gpu_num)}
        used_w = [0.0] * gpu_num
        used_mem = [0.0] * gpu_num
        sum_R = [0.0] * gpu_num

        # Seeding: spread top H intrinsic-pressure models using worst-fit on transformed slack
        # intrinsic pressure score: dR / (S - size)
        H = int(seed_H) if seed_H else 0
        if H > 0 and len(enriched) > 0:
            intrinsic = sorted(enriched, key=lambda x: (x[1] / max(S - x[2], 1e-9)), reverse=True)
            seeds = intrinsic[:min(H, len(enriched))]
            # remove selected seeds from enriched (preserve ordering among the rest)
            seed_set = set(id(x) for x in seeds)
            remaining = [x for x in enriched if id(x) not in seed_set]
            # place seeds with worst-fit on transformed slack; break ties with min KVPR option
            for w, dR, sz, m in seeds:
                best_gid = None
                best_slack = -1.0
                best_kvpr = float('inf')
                for gid in range(gpu_num):
                    if used_w[gid] + w <= cap + 1e-9 and used_mem[gid] + sz <= S + 1e-9:
                        slack = cap - (used_w[gid] + w)
                        if choose_rule == 'min_kvpr':
                            rem = S - (used_mem[gid] + sz)
                            if rem <= 0:
                                continue
                            kvnew = (sum_R[gid] + dR) / rem
                            # Worst-fit by slack, tie-break by lower kvpr
                            if (slack > best_slack or
                               (slack == best_slack and kvnew < best_kvpr)):
                                best_slack = slack
                                best_kvpr = kvnew
                                best_gid = gid
                        else:
                            # Worst-fit by slack only
                            if slack > best_slack:
                                best_slack = slack
                                best_gid = gid
                if best_gid is None:
                    return None
                used_w[best_gid] += w
                used_mem[best_gid] += sz
                sum_R[best_gid] += dR
                assign[best_gid].append(m)
            enriched = remaining

        # Main packing
        for w, dR, sz, m in enriched:
            candidate_bins = []
            for gid in range(gpu_num):
                nw = used_w[gid] + w
                if nw <= cap + 1e-9 and used_mem[gid] + sz <= S + 1e-9:
                    if choose_rule == 'min_kvpr':
                        rem = S - (used_mem[gid] + sz)
                        if rem <= 0:
                            continue
                        kv_new = (sum_R[gid] + dR) / rem
                        candidate_bins.append((gid, kv_new, nw))
                    else:
                        candidate_bins.append((gid, None, nw))
            if not candidate_bins:
                return None
            # Select bin according to variant
            if choose_rule == 'min_kvpr':
                # Prefer minimal resulting per-GPU kvpr, tie-break by smallest nw
                candidate_bins.sort(key=lambda x: (x[1], x[2]))
                best_gid = candidate_bins[0][0]
            else:
                if variant == 'bfd':
                    # Best-fit: minimal nw
                    candidate_bins.sort(key=lambda x: x[2])
                    # Tiny random tie-breaking among top-2 if close
                    if rng is not None and len(candidate_bins) > 1:
                        nw0 = candidate_bins[0][2]
                        near = [c for c in candidate_bins if abs(c[2] - nw0) <= max(1e-9, 0.001 * (cap + 1e-9))]
                        if len(near) >= 2:
                            best_gid = near[rng.randrange(min(2, len(near)))][0]
                        else:
                            best_gid = candidate_bins[0][0]
                    else:
                        best_gid = candidate_bins[0][0]
                else:
                    # First-fit
                    best_gid = candidate_bins[0][0]
            used_w[best_gid] += w
            used_mem[best_gid] += sz
            sum_R[best_gid] += dR
            assign[best_gid].append(m)

        # Validate exact constraints
        for gid in range(gpu_num):
            if used_mem[gid] - S > 1e-6:
                return None
            rem = S - used_mem[gid]
            if rem <= 0 and sum_R[gid] > 0:
                return None
            if T > 0 and (sum_R[gid] / max(rem, 1e-12)) - T > 1e-6:
                return None
        return assign

    # Simple greedy fallback used when bounds indicate infeasibility or as last resort
    def _simple_kvpr_greedy(gpu_num, models, S):
        # Sort by dR descending
        sorted_models = sorted(models, key=lambda m: (m.req_rate / m.slo), reverse=True)
        placement = {i: [] for i in range(gpu_num)}
        rem_mem = [S] * gpu_num
        sum_R = [0.0] * gpu_num
        for m in sorted_models:
            dR = m.req_rate / m.slo
            best_gid = None
            best_val = float('inf')
            for gid in range(gpu_num):
                if m.model_size <= rem_mem[gid]:
                    rem = rem_mem[gid] - m.model_size
                    if rem <= 0 and dR > 0:
                        continue
                    val = (sum_R[gid] + dR) / max(rem, 1e-12) if rem > 0 else float('inf')
                    if val < best_val:
                        best_val = val
                        best_gid = gid
            if best_gid is None:
                # give a clear error like prior versions
                raise ValueError(
                    f"Unable to place model of size {m.model_size} GB on any GPU. "
                    f"Remaining per-GPU memory: {rem_mem}"
                )
            placement[best_gid].append(m)
            sum_R[best_gid] += dR
            rem_mem[best_gid] -= m.model_size
        return placement

    # Deterministic small-random helper
    import random
    rng = random.Random(len(items) * 1009 + gpu_num * 9173)

    # Find the first feasible T by multiplicative sweep from the lower bound
    def find_first_feasible_T(start_T):
        T = max(0.0, start_T)
        growth = 1.08
        max_steps = 40
        for _ in range(max_steps):
            for (variant, order, choose_rule, seedH) in [
                ('bfd', 'w_desc', 'best_fit', min(4, max(1, gpu_num))),
                ('bfd', 'w_desc', 'min_kvpr', min(3, max(1, gpu_num - 1))),
                ('ffd', 'w_desc', 'best_fit', 0),
            ]:
                cand = pack_for_T(T, variant=variant, order=order, seed_H=seedH, rng=rng, choose_rule=choose_rule)
                if cand is not None:
                    return T, cand
            T = T * growth if T > 0 else 1e-6
        return None, None

    T_feas, placement_at_T = find_first_feasible_T(lower)
    if placement_at_T is None:
        # Fall back to greedy if sweep fails
        return _simple_kvpr_greedy(gpu_num, models, S)

    # Build a compact set of candidate T values around the first feasible T
    candidates_T = []
    for mul in [0.995, 1.0, 1.005, 1.02]:
        val = max(lower, T_feas * mul)
        candidates_T.append(val)
    # Include midpoint between lower and T_feas
    candidates_T.append(0.5 * (lower + T_feas))
    # Deduplicate and sort
    candidates_T = sorted(set(round(t, 12) for t in candidates_T))

    # Try a few packing variants per T and pick best by measured KVPR
    best_placement = None
    best_val = float('inf')

    variants = [
        ('bfd', 'w_desc', 'best_fit', min(4, max(1, gpu_num))),      # BFD on weights with spread seeding
        ('bfd', 'w_desc', 'min_kvpr', min(3, max(1, gpu_num - 1))), # choose bin minimizing resulting per-GPU KVPR
        ('ffd', 'w_desc', 'best_fit', 0),                            # FFD baseline
        ('bfd', 'density_desc', 'best_fit', 0),                      # BFD by density
    ]

    for T in candidates_T:
        for variant, order, choose_rule, seedH in variants:
            cand = pack_for_T(T, variant=variant, order=order, seed_H=seedH, rng=rng, choose_rule=choose_rule)
            if cand is None:
                continue
            val = evaluate_max_kvpr(cand)
            if val < best_val:
                best_val = val
                best_placement = cand

    # As a safety net, consider the placement found during sweep
    if best_placement is None:
        best_placement = placement_at_T
        best_val = evaluate_max_kvpr(best_placement)

    # Short bounded local search focusing on the most loaded GPU (moves then swaps)
    def local_refine(placement, move_budget=20, swap_budget=10):
        # Build mutable per-GPU aggregates
        buckets = {gid: list(placement.get(gid, [])) for gid in range(gpu_num)}
        used_mem = [0.0] * gpu_num
        sum_R = [0.0] * gpu_num
        for gid in range(gpu_num):
            for m in buckets[gid]:
                used_mem[gid] += float(m.model_size)
                sum_R[gid] += float(m.req_rate / m.slo)

        def current_kvprs():
            return [kvpr(sum_R[g], S - used_mem[g]) for g in range(gpu_num)]

        # utility to apply a move
        def apply_move(src, dst, mdl):
            dR = float(mdl.req_rate / mdl.slo)
            s = float(mdl.model_size)
            buckets[src].remove(mdl)
            buckets[dst].append(mdl)
            sum_R[src] -= dR
            sum_R[dst] += dR
            used_mem[src] -= s
            used_mem[dst] += s

        # improvement loop
        moves_left = move_budget
        swaps_left = swap_budget
        while moves_left > 0 or swaps_left > 0:
            kvprs = current_kvprs()
            cur_max = max(kvprs)
            max_gid = max(range(gpu_num), key=lambda g: kvprs[g])

            improved = False
            best_move = None  # (src,dst,mdl,resulting_max)
            # Try moves
            for mdl in list(buckets[max_gid]):
                dR = float(mdl.req_rate / mdl.slo)
                s = float(mdl.model_size)
                # source after removal
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
                    # Compute resulting global max
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
                # Try swaps
                best_swap = None  # (src,dst,a,b,resulting_max)
                if swaps_left > 0:
                    # Limit search space
                    cap_a = min(10, len(buckets[max_gid]))
                    for a in list(buckets[max_gid])[:cap_a]:
                        aR = float(a.req_rate / a.slo)
                        aS = float(a.model_size)
                        for dst in range(gpu_num):
                            if dst == max_gid or not buckets[dst]:
                                continue
                            cap_b = min(10, len(buckets[dst]))
                            for b in list(buckets[dst])[:cap_b]:
                                bR = float(b.req_rate / b.slo)
                                bS = float(b.model_size)
                                # Check memory feasibility after swap
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
                    # Perform swap
                    buckets[src].remove(a); buckets[src].append(b)
                    buckets[dst].remove(b); buckets[dst].append(a)
                    aR = float(a.req_rate / a.slo); bR = float(b.req_rate / b.slo)
                    aS = float(a.model_size);       bS = float(b.model_size)
                    sum_R[src] = sum_R[src] - aR + bR
                    sum_R[dst] = sum_R[dst] - bR + aR
                    used_mem[src] = used_mem[src] - aS + bS
                    used_mem[dst] = used_mem[dst] - bS + aS
                    swaps_left -= 1
                    improved = True
            if not improved:
                break

        return buckets

    refined = local_refine(best_placement, move_budget=20, swap_budget=10)
    # Final safety validation: ensure memory-feasible; if not, return best_placement
    ok = True
    for gid in range(gpu_num):
        mem = sum(float(m.model_size) for m in refined.get(gid, []))
        if mem - S > 1e-6:
            ok = False
            break
    return refined if ok else best_placement

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
