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

    # KVPR helper: treat zero demand as zero pressure even if memory is full
    def kvpr(R, rem_mem):
        if R <= 0:
            return 0.0
        if rem_mem <= 0:
            return float('inf')
        return R / rem_mem

    # Memory-only pre-placement (slo==0)
    def preplace_memory_only(all_models):
        placement = {i: [] for i in range(gpu_num)}
        used = [0.0] * gpu_num
        mem_only = [m for m in all_models if getattr(m, "slo", 0) == 0]
        if mem_only:
            mem_only.sort(key=lambda m: float(m.model_size), reverse=True)
            for m in mem_only:
                size = float(m.model_size)
                # place on GPU with most remaining memory that can fit
                best = None
                best_rem = -1.0
                for gid in range(gpu_num):
                    rem = S - used[gid]
                    if size <= rem and rem > best_rem:
                        best_rem = rem
                        best = gid
                if best is None:
                    # Cannot place memory-only model -> infeasible
                    raise ValueError(
                        f"Unable to place memory-only model of size {m.model_size} GB on any GPU. "
                        f"Remaining per-GPU memory: {[S - u for u in used]}"
                    )
                placement[best].append(m)
                used[best] += size
        return placement, used

    base_place, base_used = preplace_memory_only(models)
    # Remaining models: slo>0
    rem_models = [m for m in models if getattr(m, "slo", 0) != 0]

    # Build greedy min-max placer on top of the pre-placement
    def greedy_place(order_models):
        placement = {gid: list(base_place.get(gid, [])) for gid in range(gpu_num)}
        rem_mem = [S - base_used[i] for i in range(gpu_num)]
        sum_R = [0.0 for _ in range(gpu_num)]  # total dR per GPU

        # initialize R with any demand-bearing already in base (should be none, but safe)
        for gid in range(gpu_num):
            for m0 in placement[gid]:
                slo0 = getattr(m0, "slo", 0.0)
                if slo0:
                    sum_R[gid] += float(m0.req_rate) / float(slo0)

        def current_kvprs():
            return [kvpr(sum_R[i], rem_mem[i]) for i in range(gpu_num)]

        for m in order_models:
            dR = float(m.req_rate) / float(m.slo)
            size = float(m.model_size)

            # Evaluate best GPU by minimizing resulting global max KVPR
            kvprs_before = current_kvprs()
            best_gid = None
            best_res_max = float('inf')
            best_local = float('inf')
            best_rem_after = -1.0

            for gid in range(gpu_num):
                if size <= rem_mem[gid]:
                    new_R = sum_R[gid] + dR
                    new_rem = rem_mem[gid] - size
                    new_k = kvpr(new_R, new_rem)
                    # resulting global max
                    res_max = new_k
                    for j in range(gpu_num):
                        if j == gid:
                            continue
                        if kvprs_before[j] > res_max:
                            res_max = kvprs_before[j]
                    # tie-break: minimize resulting max, then the GPU KVPR, then leave more rem
                    if (res_max < best_res_max or
                        (res_max == best_res_max and new_k < best_local) or
                        (res_max == best_res_max and new_k == best_local and new_rem > best_rem_after)):
                        best_res_max = res_max
                        best_local = new_k
                        best_rem_after = new_rem
                        best_gid = gid

            if best_gid is None:
                # Unplaceable with this ordering -> signal failure
                raise ValueError(
                    f"Unable to place model of size {m.model_size} GB; remaining per-GPU memory: {rem_mem}"
                )

            # Commit
            placement[best_gid].append(m)
            rem_mem[best_gid] -= size
            sum_R[best_gid] += dR

        return placement

    def eval_max_kvpr(placement):
        max_v = 0.0
        for gid in range(gpu_num):
            used = 0.0
            R = 0.0
            for m in placement.get(gid, []):
                used += float(m.model_size)
                slo = float(getattr(m, "slo", 0.0))
                if slo:
                    R += float(m.req_rate) / slo
            max_v = max(max_v, kvpr(R, S - used))
        return max_v

    # Build multiple deterministic orderings for initialization
    def pressure_weight(m):
        denom = S - float(m.model_size)
        if denom <= 0:
            return float('inf')
        return (float(m.req_rate) / float(m.slo)) / denom

    def demand(m):
        return float(m.req_rate) / float(m.slo)

    def density(m):
        sz = float(m.model_size) if float(m.model_size) > 0 else 1e-9
        return (float(m.req_rate) / float(m.slo)) / sz

    orderings = [
        lambda ms: sorted(ms, key=pressure_weight, reverse=True),
        lambda ms: sorted(ms, key=demand, reverse=True),
        lambda ms: sorted(ms, key=lambda m: float(m.model_size), reverse=True),
        lambda ms: sorted(ms, key=lambda m: float(m.model_size)),  # small first
        lambda ms: sorted(ms, key=density, reverse=True),
        lambda ms: sorted(ms, key=lambda m: (demand(m), -float(m.model_size)), reverse=True),
    ]

    best_placement = None
    best_score = float('inf')
    # Try each ordering; keep best feasible
    for make_order in orderings:
        try:
            ordered = make_order(rem_models)
            cand = greedy_place(ordered)
            score = eval_max_kvpr(cand)
            if score < best_score:
                best_score = score
                best_placement = cand
        except Exception:
            continue

    if best_placement is None:
        # Fallback: place all models greedily by size-desc
        ordered = sorted(rem_models, key=lambda m: float(m.model_size), reverse=True)
        best_placement = greedy_place(ordered)
        best_score = eval_max_kvpr(best_placement)

    # Large Neighborhood Search refinement with donation-first moves and size-focused swaps
    def lns_refine(placement, max_rounds=6, move_cap=24, swap_cap=12):
        # Copy mutable structures
        buckets = {gid: list(placement.get(gid, [])) for gid in range(gpu_num)}
        used = [0.0] * gpu_num
        Rsum = [0.0] * gpu_num
        for gid in range(gpu_num):
            for m in buckets[gid]:
                used[gid] += float(m.model_size)
                slo = float(getattr(m, "slo", 0.0))
                if slo:
                    Rsum[gid] += float(m.req_rate) / slo

        def kvprs_all():
            return [kvpr(Rsum[g], S - used[g]) for g in range(gpu_num)]

        def apply_move(src, dst, m):
            dR = float(m.req_rate) / float(m.slo)
            s = float(m.model_size)
            buckets[src].remove(m); buckets[dst].append(m)
            Rsum[src] -= dR; Rsum[dst] += dR
            used[src] -= s; used[dst] += s

        rounds = 0
        no_improve_passes = 0
        while rounds < max_rounds:
            rounds += 1
            improved_round = False
            kvs = kvprs_all()
            cur_max = max(kvs) if kvs else 0.0
            if gpu_num == 0:
                break
            worst = max(range(gpu_num), key=lambda g: kvs[g])
            # Two best recipient GPUs by current KVPR (lowest)
            best_recips = sorted(range(gpu_num), key=lambda g: kvs[g])[:min(2, gpu_num)]
            # Donation-first moves from worst
            moves_left = move_cap
            best_move = None  # (src,dst,mdl,resulting_max)
            # Candidate models: largest and highest-demand
            src_models = list(buckets[worst])
            src_models.sort(key=lambda mm: (float(mm.req_rate) / float(mm.slo), float(mm.model_size)), reverse=True)
            # Limit to top 12 by demand and top 12 by size
            top_by_demand = src_models[:12]
            top_by_size = sorted(list(buckets[worst]), key=lambda mm: float(mm.model_size), reverse=True)[:12]
            cand_set = list(dict.fromkeys(top_by_demand + top_by_size))

            for mdl in cand_set:
                if moves_left <= 0:
                    break
                dR = float(mdl.req_rate) / float(mdl.slo)
                s = float(mdl.model_size)
                # Source after move
                src_R_new = Rsum[worst] - dR
                src_used_new = used[worst] - s
                src_rem_new = S - src_used_new
                src_kv_new = kvpr(src_R_new, src_rem_new)
                for dst in best_recips:
                    if dst == worst:
                        continue
                    if used[dst] + s > S:
                        continue
                    dst_R_new = Rsum[dst] + dR
                    dst_used_new = used[dst] + s
                    dst_kv_new = kvpr(dst_R_new, S - dst_used_new)
                    # Resulting global max
                    resulting = src_kv_new if src_kv_new > dst_kv_new else dst_kv_new
                    for g in range(gpu_num):
                        if g == worst or g == dst:
                            continue
                        if kvs[g] > resulting:
                            resulting = kvs[g]
                    if resulting + 1e-12 < cur_max:
                        if best_move is None or resulting < best_move[3]:
                            best_move = (worst, dst, mdl, resulting)
                moves_left -= 1

            if best_move is not None:
                src, dst, mdl, _ = best_move
                apply_move(src, dst, mdl)
                improved_round = True
            else:
                # Try limited swaps between worst and others
                swaps_left = swap_cap
                best_swap = None  # (src,dst,a,b,resulting_max)
                src = worst
                # Cap candidates
                A = sorted(list(buckets[src]), key=lambda m: (float(m.req_rate)/float(m.slo), float(m.model_size)), reverse=True)[:12]
                for a in A:
                    if swaps_left <= 0:
                        break
                    aR = float(a.req_rate) / float(a.slo)
                    aS = float(a.model_size)
                    for dst in range(gpu_num):
                        if dst == src or not buckets[dst]:
                            continue
                        B = sorted(list(buckets[dst]), key=lambda m: (float(m.req_rate)/float(m.slo), float(m.model_size)), reverse=True)[:12]
                        for b in B:
                            if swaps_left <= 0:
                                break
                            bR = float(b.req_rate) / float(b.slo)
                            bS = float(b.model_size)
                            # Feasibility after swap
                            src_used_new = used[src] - aS + bS
                            dst_used_new = used[dst] - bS + aS
                            if src_used_new > S or dst_used_new > S:
                                swaps_left -= 1
                                continue
                            src_R_new = Rsum[src] - aR + bR
                            dst_R_new = Rsum[dst] - bR + aR
                            src_kv_new = kvpr(src_R_new, S - src_used_new)
                            dst_kv_new = kvpr(dst_R_new, S - dst_used_new)
                            resulting = src_kv_new if src_kv_new > dst_kv_new else dst_kv_new
                            for g in range(gpu_num):
                                if g == src or g == dst:
                                    continue
                                if kvs[g] > resulting:
                                    resulting = kvs[g]
                            if resulting + 1e-12 < cur_max:
                                if best_swap is None or resulting < best_swap[4]:
                                    best_swap = (src, dst, a, b, resulting)
                            swaps_left -= 1
                        if swaps_left <= 0:
                            break
                if best_swap is not None:
                    src, dst, a, b, _ = best_swap
                    # Apply swap
                    buckets[src].remove(a); buckets[src].append(b)
                    buckets[dst].remove(b); buckets[dst].append(a)
                    aR = float(a.req_rate) / float(a.slo)
                    bR = float(b.req_rate) / float(b.slo)
                    aS = float(a.model_size); bS = float(b.model_size)
                    Rsum[src] = Rsum[src] - aR + bR
                    Rsum[dst] = Rsum[dst] - bR + aR
                    used[src] = used[src] - aS + bS
                    used[dst] = used[dst] - bS + aS
                    improved_round = True
                else:
                    # Optional ruin-and-recreate when stuck: remove a few high-impact items from worst and reinsert
                    removed = []
                    if buckets[worst]:
                        # Score by decrease in worst KVPR if removed
                        scored = []
                        for m in buckets[worst]:
                            dR = float(m.req_rate) / float(m.slo)
                            s = float(m.model_size)
                            R_new = Rsum[worst] - dR
                            rem_new = S - (used[worst] - s)
                            new_k = kvpr(R_new, rem_new)
                            gain = kvs[worst] - new_k
                            scored.append((gain, m))
                        scored.sort(key=lambda t: t[0], reverse=True)
                        take = min(3, len(scored))
                        for i in range(take):
                            _, m = scored[i]
                            buckets[worst].remove(m)
                            Rsum[worst] -= float(m.req_rate) / float(m.slo)
                            used[worst] -= float(m.model_size)
                            removed.append(m)
                    # Reinsert removed greedily by minimizing resulting max
                    for m in removed:
                        dR = float(m.req_rate) / float(m.slo)
                        s = float(m.model_size)
                        # compute current kvprs
                        kvs_now = kvprs_all()
                        best_gid = None
                        best_res = float('inf')
                        best_local = float('inf')
                        best_rem_after = -1.0
                        for gid in range(gpu_num):
                            if used[gid] + s <= S:
                                new_R = Rsum[gid] + dR
                                new_rem = S - (used[gid] + s)
                                new_k = kvpr(new_R, new_rem)
                                res = new_k
                                for j in range(gpu_num):
                                    if j == gid:
                                        continue
                                    if kvs_now[j] > res:
                                        res = kvs_now[j]
                                if (res < best_res or
                                    (res == best_res and new_k < best_local) or
                                    (res == best_res and new_k == best_local and new_rem > best_rem_after)):
                                    best_res = res
                                    best_local = new_k
                                    best_rem_after = new_rem
                                    best_gid = gid
                        if best_gid is not None:
                            buckets[best_gid].append(m)
                            Rsum[best_gid] += dR
                            used[best_gid] += s
                    # Accept if improved
                    new_max = max(kvprs_all()) if gpu_num > 0 else 0.0
                    if new_max + 1e-12 < cur_max:
                        improved_round = True

            if improved_round:
                no_improve_passes = 0
            else:
                no_improve_passes += 1
                if no_improve_passes >= 2:
                    break  # two consecutive non-improving passes

        return buckets

    refined = lns_refine(best_placement, max_rounds=6, move_cap=24, swap_cap=12)

    # Final safety: ensure no GPU exceeds memory
    for gid in range(gpu_num):
        mem = sum(float(m.model_size) for m in refined.get(gid, []))
        if mem - S > 1e-6:
            # If ever violated, return best_placement (shouldn't happen)
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