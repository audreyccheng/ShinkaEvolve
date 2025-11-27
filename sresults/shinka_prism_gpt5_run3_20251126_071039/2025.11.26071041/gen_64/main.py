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

    # Helpers
    def eff_dR(m):
        # Treat slo==0 as memory-only (no KV demand)
        return (float(m.req_rate) / float(m.slo)) if getattr(m, "slo", 0) != 0 else 0.0

    def kvpr(R, rem_mem):
        if rem_mem <= 0:
            return float('inf')
        return R / rem_mem

    def current_kvprs(sum_R, used_mem):
        return [kvpr(sum_R[g], S - used_mem[g]) for g in range(gpu_num)]

    def eval_max_kvpr(placement):
        used_mem = [0.0] * gpu_num
        sum_R = [0.0] * gpu_num
        for gid in range(gpu_num):
            for m in placement.get(gid, []):
                used_mem[gid] += float(m.model_size)
                sum_R[gid] += eff_dR(m)
        kvprs = current_kvprs(sum_R, used_mem)
        return max(kvprs) if kvprs else 0.0

    # Greedy min-max assignment for arbitrary ordering (treat slo==0 as dR=0)
    def greedy_minmax_assign(ordered_models, init=None):
        placement = {i: [] for i in range(gpu_num)} if init is None else {i: list(init.get(i, [])) for i in range(gpu_num)}
        used_mem = [0.0] * gpu_num
        sum_R = [0.0] * gpu_num

        # initialize states if init provided
        if init is not None:
            for gid in range(gpu_num):
                for m in placement[gid]:
                    used_mem[gid] += float(m.model_size)
                    sum_R[gid] += eff_dR(m)

        for model in ordered_models:
            dR = eff_dR(model)
            size = float(model.model_size)
            # Compute current kvprs once
            curr_kvprs = current_kvprs(sum_R, used_mem)

            best_gid = None
            best_res_max = float('inf')
            best_local = float('inf')
            best_rem = -1.0

            for gid in range(gpu_num):
                if used_mem[gid] + size <= S + 1e-9:
                    new_R = sum_R[gid] + dR
                    new_used = used_mem[gid] + size
                    rem_after = S - new_used
                    if rem_after <= 0:
                        continue
                    new_k = kvpr(new_R, rem_after)
                    # resulting global max
                    resulting = new_k
                    for j in range(gpu_num):
                        if j == gid:
                            continue
                        if curr_kvprs[j] > resulting:
                            resulting = curr_kvprs[j]
                    # Tie breaks
                    if (resulting < best_res_max or
                        (resulting == best_res_max and new_k < best_local) or
                        (resulting == best_res_max and new_k == best_local and rem_after > best_rem)):
                        best_res_max = resulting
                        best_local = new_k
                        best_rem = rem_after
                        best_gid = gid

            if best_gid is None:
                return None  # infeasible for this ordering/init
            # commit
            placement[best_gid].append(model)
            used_mem[best_gid] += size
            sum_R[best_gid] += dR

        return placement

    # Pre-place memory-only models (slo==0) by pure memory balancing (largest first)
    mem_only = [m for m in models if getattr(m, "slo", 0) == 0]
    with_kv = [m for m in models if getattr(m, "slo", 0) != 0]

    # If there are memory-only models, place them first
    preplacement = {i: [] for i in range(gpu_num)}
    used_mem_pre = [0.0] * gpu_num
    sum_R_pre = [0.0] * gpu_num

    if mem_only:
        mem_only_sorted = sorted(mem_only, key=lambda m: float(m.model_size), reverse=True)
        for m in mem_only_sorted:
            size = float(m.model_size)
            # choose GPU with most remaining memory that can fit
            best_gid = None
            best_remain = -1.0
            for gid in range(gpu_num):
                if used_mem_pre[gid] + size <= S + 1e-9:
                    remain = S - (used_mem_pre[gid] + size)
                    if remain > best_remain:
                        best_remain = remain
                        best_gid = gid
            if best_gid is None:
                # Fallback to global greedy if preplacement blocks feasibility
                preplacement = None
                break
            preplacement[best_gid].append(m)
            used_mem_pre[best_gid] += size
            # no KV demand added for slo==0

    # Build initial placements for KV-demanding models
    initial_candidates = []

    if preplacement is not None:
        # Two different orderings for with_kv
        order1 = sorted(with_kv, key=lambda m: (eff_dR(m) / max(S - float(m.model_size), 1e-9)), reverse=True)
        order2 = sorted(with_kv, key=lambda m: float(m.model_size), reverse=True)
        for ordered in (order1, order2):
            cand = greedy_minmax_assign(ordered, init=preplacement)
            if cand is not None:
                initial_candidates.append(cand)

    # If no candidate built (e.g., preplacement failed), fall back to greedy over all models
    if not initial_candidates:
        # Greedy over all models treating slo==0 as 0 demand
        all_order1 = sorted(models, key=lambda m: (eff_dR(m) / max(S - float(m.model_size), 1e-9)), reverse=True)
        all_order2 = sorted(models, key=lambda m: float(m.model_size), reverse=True)
        best_cand = None
        best_val = float('inf')
        for ordered in (all_order1, all_order2):
            cand = greedy_minmax_assign(ordered, init=None)
            if cand is not None:
                val = eval_max_kvpr(cand)
                if val < best_val:
                    best_val = val
                    best_cand = cand
        if best_cand is None:
            # As a last resort, size-desc only (will raise by caller if impossible)
            best_cand = greedy_minmax_assign(sorted(models, key=lambda m: float(m.model_size), reverse=True), init=None)
        return best_cand

    # Pick the best initial candidate
    best_placement = min(initial_candidates, key=lambda p: eval_max_kvpr(p))
    best_score = eval_max_kvpr(best_placement)

    # Large Neighborhood Search (ruin-and-recreate) around worst GPUs
    def lns_improve(placement, rounds=16):
        # Build working state arrays
        def summarize(pl):
            used = [0.0] * gpu_num
            R = [0.0] * gpu_num
            for gid in range(gpu_num):
                for m in pl.get(gid, []):
                    used[gid] += float(m.model_size)
                    R[gid] += eff_dR(m)
            return used, R

        incumbent = {i: list(placement.get(i, [])) for i in range(gpu_num)}
        used_mem, sum_R = summarize(incumbent)
        inc_score = max(current_kvprs(sum_R, used_mem)) if gpu_num > 0 else 0.0

        for _ in range(rounds):
            kvprs = current_kvprs(sum_R, used_mem)
            if not kvprs:
                break
            worst = max(range(gpu_num), key=lambda g: kvprs[g])
            # Pick a second target if available
            second = None
            if gpu_num > 1:
                second = max([g for g in range(gpu_num) if g != worst], key=lambda g: kvprs[g]) if gpu_num > 1 else None

            # Build removal set from worst and second worst
            removal = []
            def select_from(gid, k):
                if gid is None or not incumbent[gid]:
                    return
                # choose union of top by dR and by size
                by_dR = sorted(incumbent[gid], key=lambda m: eff_dR(m), reverse=True)[:max(1, k//2)]
                by_sz = sorted(incumbent[gid], key=lambda m: float(m.model_size), reverse=True)[:k]
                sel = []
                seen = set()
                for m in by_dR + by_sz:
                    if id(m) not in seen:
                        seen.add(id(m)); sel.append(m)
                # truncate to k
                del sel[k:] if len(sel) > k else None
                removal.extend((gid, m) for m in sel)

            k1 = min(8, len(incumbent[worst]))
            select_from(worst, k1)
            if second is not None:
                k2 = min(4, len(incumbent[second]))
                select_from(second, k2)

            if not removal:
                break

            # Create candidate by removing selected models
            cand_place = {i: list(incumbent.get(i, [])) for i in range(gpu_num)}
            used_c = list(used_mem)
            sumR_c = list(sum_R)

            removed_models = []
            for gid, m in removal:
                if m in cand_place[gid]:
                    cand_place[gid].remove(m)
                    used_c[gid] -= float(m.model_size)
                    sumR_c[gid] -= eff_dR(m)
                    removed_models.append(m)

            # Repair: reinsert models by minimizing resulting global max KVPR
            # Order by pressure weight
            removed_models.sort(key=lambda m: (eff_dR(m) / max(S - float(m.model_size), 1e-9)), reverse=True)

            feasible = True
            for m in removed_models:
                dR = eff_dR(m)
                sz = float(m.model_size)
                curr_k = current_kvprs(sumR_c, used_c)
                best_gid = None
                best_res = float('inf')
                best_local = float('inf')
                best_rem = -1.0
                for gid in range(gpu_num):
                    if used_c[gid] + sz <= S + 1e-9:
                        new_used = used_c[gid] + sz
                        rem_after = S - new_used
                        if rem_after <= 0:
                            continue
                        new_R = sumR_c[gid] + dR
                        new_k = kvpr(new_R, rem_after)
                        # global resulting max
                        res = new_k
                        for j in range(gpu_num):
                            if j == gid:
                                continue
                            if curr_k[j] > res:
                                res = curr_k[j]
                        if (res < best_res or
                            (res == best_res and new_k < best_local) or
                            (res == best_res and new_k == best_local and rem_after > best_rem)):
                            best_res = res
                            best_local = new_k
                            best_rem = rem_after
                            best_gid = gid
                if best_gid is None:
                    feasible = False
                    break
                # commit reinsertion
                cand_place[best_gid].append(m)
                used_c[best_gid] += sz
                sumR_c[best_gid] += dR

            if not feasible:
                continue

            cand_k = current_kvprs(sumR_c, used_c)
            cand_score = max(cand_k) if cand_k else 0.0
            if cand_score + 1e-12 < inc_score:
                incumbent = cand_place
                used_mem, sum_R = used_c, sumR_c
                inc_score = cand_score

        return incumbent

    improved = lns_improve(best_placement, rounds=16)
    improved_score = eval_max_kvpr(improved)
    if improved_score + 1e-12 < best_score:
        best_placement = improved
        best_score = improved_score

    # Final safety: ensure memory limits are respected
    for gid in range(gpu_num):
        mem = sum(float(m.model_size) for m in best_placement.get(gid, []))
        if mem - S > 1e-6:
            # Fall back to global greedy if any overflow detected
            fallback = greedy_minmax_assign(sorted(models, key=lambda m: (eff_dR(m) / max(S - float(m.model_size), 1e-9)), reverse=True))
            if fallback is not None:
                return fallback
            # ultimate fallback: size-desc
            fallback2 = greedy_minmax_assign(sorted(models, key=lambda m: float(m.model_size), reverse=True))
            if fallback2 is not None:
                return fallback2
            break

    return best_placement

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