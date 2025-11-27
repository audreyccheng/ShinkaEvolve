# EVOLVE-BLOCK-START
"""Model placement algorithm for minimizing maximum KV cache pressure across GPUs
   Dual-driven water-filling rebalancing without T-parameter search.
"""

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
    # Trivial returns
    placement_empty = {i: [] for i in range(gpu_num)}
    if gpu_num <= 0 or not models:
        return placement_empty

    S = GPU_MEM_SIZE
    eps = 1e-12

    # KVPR helper
    def kvpr(R, rem_mem):
        if rem_mem <= 0:
            return float('inf')
        return R / rem_mem

    # Split models into memory-only (slo==0) and rate-contributing (slo>0)
    mem_only = []
    active = []
    for m in models:
        slo = float(getattr(m, "slo", 0.0))
        sz = float(getattr(m, "model_size", 0.0))
        if slo == 0.0:
            mem_only.append(m)
        else:
            dr = float(getattr(m, "req_rate", 0.0)) / slo
            active.append((m, dr, sz))

    # Place memory-only models first: largest size first onto GPU with most free memory
    placement = {i: [] for i in range(gpu_num)}
    used_mem = [0.0] * gpu_num
    sum_R = [0.0] * gpu_num

    mem_only_sorted = sorted(mem_only, key=lambda x: float(getattr(x, "model_size", 0.0)), reverse=True)
    for m in mem_only_sorted:
        sz = float(getattr(m, "model_size", 0.0))
        if sz > S + eps:
            # Cannot place a single model anywhere
            raise ValueError(f"Model size {sz} exceeds GPU memory {S}")
        # choose GPU with most remaining memory (tie: least items)
        best = None
        best_key = None
        for g in range(gpu_num):
            rem = S - used_mem[g]
            if sz <= rem + eps:
                key = (rem, -len(placement[g]))
                if best_key is None or key > best_key:
                    best_key = key
                    best = g
        if best is None:
            # No feasible placement due to memory
            raise ValueError("Unable to place memory-only model due to memory constraints")
        placement[best].append(m)
        used_mem[best] += sz
        # sum_R unchanged (no KV contribution)

    # If no active models, return memory-only placement
    if not active:
        return placement

    # Build convenience lists for active models
    # Element: (model, dR, size)
    active_sorted_seed = sorted(
        active,
        key=lambda t: (t[1] / max(S - t[2], 1e-9), t[1], t[2]),  # priority by intrinsic pressure
        reverse=True
    )

    # Initial greedy minimax placement for active models
    def greedy_minimax_place(seed_items):
        for (m, dr, sz) in seed_items:
            # choose GPU minimizing resulting max KVPR
            best_gpu = None
            best_result = float('inf')
            best_local = float('inf')
            for g in range(gpu_num):
                if used_mem[g] + sz > S + eps:
                    continue
                new_used = used_mem[g] + sz
                rem_g = S - new_used
                if rem_g <= 0:
                    continue
                new_R = sum_R[g] + dr
                new_k_g = kvpr(new_R, rem_g)
                # compute resulting max KVPR across GPUs
                resulting = new_k_g
                for h in range(gpu_num):
                    if h == g:
                        continue
                    other = kvpr(sum_R[h], S - used_mem[h])
                    if other > resulting:
                        resulting = other
                if (resulting < best_result or
                    (abs(resulting - best_result) <= 1e-12 and (new_k_g < best_local)) or
                    (abs(resulting - best_result) <= 1e-12 and abs(new_k_g - best_local) <= 1e-12 and (S - new_used) > (S - used_mem[best_gpu] - sz) if best_gpu is not None else True)):
                    best_result = resulting
                    best_local = new_k_g
                    best_gpu = g
            if best_gpu is None:
                # Fall back: place where it fits with least local KVPR
                best_gpu = None
                best_local = float('inf')
                for g in range(gpu_num):
                    if used_mem[g] + sz <= S + eps:
                        rem = S - (used_mem[g] + sz)
                        if rem > 0:
                            k = kvpr(sum_R[g] + dr, rem)
                            if k < best_local:
                                best_local = k
                                best_gpu = g
                if best_gpu is None:
                    raise ValueError("Unable to place active model due to memory.")
            placement[best_gpu].append(m)
            used_mem[best_gpu] += sz
            sum_R[best_gpu] += dr

    greedy_minimax_place(active_sorted_seed)

    # Evaluate current per-GPU KVPRs
    def per_gpu_kvprs(used_mem_list=None, sum_R_list=None):
        um = used_mem if used_mem_list is None else used_mem_list
        rr = sum_R if sum_R_list is None else sum_R_list
        arr = []
        for g in range(gpu_num):
            arr.append(kvpr(rr[g], S - um[g]))
        return arr

    # Dual sensitivities: α_g for memory, β_g for rate
    def dual_sensitivities():
        alpha = [0.0] * gpu_num
        beta = [0.0] * gpu_num
        for g in range(gpu_num):
            rem = max(eps, S - used_mem[g])
            beta[g] = 1.0 / rem
            alpha[g] = sum_R[g] / (rem * rem)
        return alpha, beta

    # Try moving a single model and return the improvement (delta in max KVPR)
    def try_move(model, dr, sz, src, dst):
        if src == dst:
            return None
        if used_mem[dst] + sz > S + eps:
            return None
        # simulate
        rem_src_new = S - (used_mem[src] - sz)
        rem_dst_new = S - (used_mem[dst] + sz)
        if rem_src_new <= 0 or rem_dst_new <= 0:
            return None
        kvprs_before = per_gpu_kvprs()
        cur_max = max(kvprs_before)
        # new kvprs only for src and dst
        new_src_k = kvpr(sum_R[src] - dr, rem_src_new)
        new_dst_k = kvpr(sum_R[dst] + dr, rem_dst_new)
        # resulting max
        res = new_src_k if new_src_k > new_dst_k else new_dst_k
        for g in range(gpu_num):
            if g == src or g == dst:
                continue
            if kvprs_before[g] > res:
                res = kvprs_before[g]
        return cur_max - res  # positive means improvement

    # Execute move
    def apply_move(model, dr, sz, src, dst):
        placement[src].remove(model)
        placement[dst].append(model)
        used_mem[src] -= sz
        used_mem[dst] += sz
        sum_R[src] -= dr
        sum_R[dst] += dr

    # Gather active item lookup: map model -> (dr, sz, gpu)
    def build_active_index():
        idx = {}
        for g in range(gpu_num):
            for m in placement[g]:
                slo = float(getattr(m, "slo", 0.0))
                if slo > 0:
                    dr = float(getattr(m, "req_rate", 0.0)) / slo
                    sz = float(getattr(m, "model_size", 0.0))
                    idx[m] = (dr, sz, g)
        return idx

    # Water-level surplus/deficit balancing
    def balance_surplus_deficit(max_iters=2, move_cap=20):
        it = 0
        moves = 0
        while it < max_iters and moves < move_cap:
            it += 1
            rems = [max(eps, S - used_mem[g]) for g in range(gpu_num)]
            total_R = sum(sum_R)
            total_rem = sum(rems)
            if total_rem <= eps:
                break
            R_hat = total_R / total_rem
            # deficit (positive means need more R)
            deficit = [R_hat * rems[g] - sum_R[g] for g in range(gpu_num)]
            donors = sorted([g for g in range(gpu_num) if deficit[g] < -1e-9],
                            key=lambda g: deficit[g])  # most negative first
            receivers = sorted([g for g in range(gpu_num) if deficit[g] > 1e-9],
                               key=lambda g: deficit[g], reverse=True)
            if not donors or not receivers:
                break

            improved = False
            kv_before = max(per_gpu_kvprs())
            # attempt to move one model per iteration
            for src in donors:
                # models on src sorted by "helpfulness" = dr*beta[src] + sz*alpha[src]
                alpha, beta = dual_sensitivities()
                cand_models = []
                for m in list(placement[src]):
                    slo = float(getattr(m, "slo", 0.0))
                    if slo <= 0:
                        continue
                    dr = float(getattr(m, "req_rate", 0.0)) / slo
                    sz = float(getattr(m, "model_size", 0.0))
                    score = dr * beta[src] + sz * alpha[src]
                    cand_models.append((score, m, dr, sz))
                cand_models.sort(key=lambda x: x[0], reverse=True)
                moved = False
                for _, m, dr, sz in cand_models:
                    # try receivers by largest deficit
                    for dst in receivers:
                        if used_mem[dst] + sz > S + eps:
                            continue
                        gain = try_move(m, dr, sz, src, dst)
                        if gain is not None and gain > 1e-12:
                            apply_move(m, dr, sz, src, dst)
                            moves += 1
                            improved = True
                            moved = True
                            break
                    if moved:
                        break
                if improved:
                    break
            if not improved:
                # no improving move found
                break

    # Dual-based targeted rebalancing from the most pressured GPU
    def dual_rebalance(passes=3, move_budget=24):
        for _ in range(passes):
            kvprs = per_gpu_kvprs()
            cur_max = max(kvprs)
            max_gid = max(range(gpu_num), key=lambda g: kvprs[g])
            alpha, beta = dual_sensitivities()
            # Rank models on the most loaded GPU by estimated incremental harm
            candidates = []
            for m in list(placement[max_gid]):
                slo = float(getattr(m, "slo", 0.0))
                if slo <= 0:
                    continue
                dr = float(getattr(m, "req_rate", 0.0)) / slo
                sz = float(getattr(m, "model_size", 0.0))
                harm = dr * beta[max_gid] + sz * alpha[max_gid]
                candidates.append((harm, m, dr, sz))
            if not candidates:
                break
            candidates.sort(key=lambda x: x[0], reverse=True)

            moved_any = False
            moves_left = move_budget
            for _, m, dr, sz in candidates:
                if moves_left <= 0:
                    break
                # choose destination by minimal incremental cost
                best_dst = None
                best_cost = float('inf')
                for g in range(gpu_num):
                    if g == max_gid:
                        continue
                    if used_mem[g] + sz > S + eps:
                        continue
                    cost = dr * beta[g] + sz * alpha[g]
                    if cost < best_cost:
                        best_cost = cost
                        best_dst = g
                if best_dst is None:
                    continue
                gain = try_move(m, dr, sz, max_gid, best_dst)
                if gain is not None and gain > 1e-12:
                    apply_move(m, dr, sz, max_gid, best_dst)
                    moved_any = True
                    moves_left -= 1
            if not moved_any:
                break

    # Limited swap improvement between hottest GPU and others
    def limited_swaps(swap_budget=8):
        swaps = 0
        while swaps < swap_budget:
            kvprs = per_gpu_kvprs()
            cur_max = max(kvprs)
            src = max(range(gpu_num), key=lambda g: kvprs[g])
            improved = False
            # limit consideration
            src_models = [m for m in placement[src] if float(getattr(m, "slo", 0.0)) > 0.0]
            src_models = sorted(
                src_models,
                key=lambda m: (float(getattr(m, "req_rate", 0.0)) / float(getattr(m, "slo", 1.0))),
                reverse=True
            )[:min(10, len(src_models))]

            for a in src_models:
                a_slo = float(getattr(a, "slo", 0.0))
                a_dr = float(getattr(a, "req_rate", 0.0)) / a_slo
                a_sz = float(getattr(a, "model_size", 0.0))
                for dst in range(gpu_num):
                    if dst == src or not placement[dst]:
                        continue
                    # try a few dst models
                    dst_models = [b for b in placement[dst] if float(getattr(b, "slo", 0.0)) > 0.0]
                    dst_models = sorted(
                        dst_models,
                        key=lambda b: (float(getattr(b, "req_rate", 0.0)) / float(getattr(b, "slo", 1.0)))
                    )[:min(8, len(dst_models))]

                    for b in dst_models:
                        b_slo = float(getattr(b, "slo", 0.0))
                        b_dr = float(getattr(b, "req_rate", 0.0)) / b_slo
                        b_sz = float(getattr(b, "model_size", 0.0))
                        # check memory feasibility after swap
                        mem_src_new = used_mem[src] - a_sz + b_sz
                        mem_dst_new = used_mem[dst] - b_sz + a_sz
                        if mem_src_new > S + eps or mem_dst_new > S + eps:
                            continue
                        rem_src = S - mem_src_new
                        rem_dst = S - mem_dst_new
                        if rem_src <= 0 or rem_dst <= 0:
                            continue
                        # compute resulting max kvpr
                        kvprs_before = per_gpu_kvprs()
                        new_src_k = kvpr(sum_R[src] - a_dr + b_dr, rem_src)
                        new_dst_k = kvpr(sum_R[dst] - b_dr + a_dr, rem_dst)
                        res = max(new_src_k, new_dst_k)
                        for g in range(gpu_num):
                            if g == src or g == dst:
                                continue
                            if kvprs_before[g] > res:
                                res = kvprs_before[g]
                        if res + 1e-12 < cur_max:
                            # apply swap
                            placement[src].remove(a); placement[src].append(b)
                            placement[dst].remove(b); placement[dst].append(a)
                            used_mem[src] = mem_src_new
                            used_mem[dst] = mem_dst_new
                            sum_R[src] = sum_R[src] - a_dr + b_dr
                            sum_R[dst] = sum_R[dst] - b_dr + a_dr
                            swaps += 1
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break

    # Run improvements
    dual_rebalance(passes=3, move_budget=24)
    balance_surplus_deficit(max_iters=2, move_cap=20)
    dual_rebalance(passes=2, move_budget=16)
    limited_swaps(swap_budget=6)

    # Final pass: single-item descent until no improvement
    def single_pass_descent(max_moves=20):
        moves = 0
        while moves < max_moves:
            kvprs = per_gpu_kvprs()
            cur_max = max(kvprs)
            src = max(range(gpu_num), key=lambda g: kvprs[g])
            best = None  # (gain, src, dst, model, dr, sz)
            for m in list(placement[src]):
                slo = float(getattr(m, "slo", 0.0))
                if slo <= 0:
                    continue
                dr = float(getattr(m, "req_rate", 0.0)) / slo
                sz = float(getattr(m, "model_size", 0.0))
                for dst in range(gpu_num):
                    if dst == src:
                        continue
                    g = try_move(m, dr, sz, src, dst)
                    if g is not None and g > 1e-12:
                        if best is None or g > best[0]:
                            best = (g, src, dst, m, dr, sz)
            if best is None:
                break
            _, s, d, m, dr, sz = best
            apply_move(m, dr, sz, s, d)
            moves += 1

    single_pass_descent(max_moves=20)

    # Safety: ensure memory is not exceeded
    for g in range(gpu_num):
        mem = sum(float(getattr(m, "model_size", 0.0)) for m in placement[g])
        if mem - S > 1e-6:
            # Should not happen due to checks; in case, roll back to a simpler feasible greedy
            return _greedy_fallback_kvpr(gpu_num, models, S)

    return placement


def _greedy_fallback_kvpr(gpu_num, models, S):
    """Simple greedy baseline minimizing the resulting max KVPR at each insertion."""
    def kvpr(R, rem_mem):
        if rem_mem <= 0:
            return float('inf')
        return R / rem_mem

    placement = {i: [] for i in range(gpu_num)}
    used_mem = [0.0] * gpu_num
    sum_R = [0.0] * gpu_num

    # sort by intrinsic pressure, tie by dR then size
    def model_key(m):
        slo = float(getattr(m, "slo", 0.0))
        sz = float(getattr(m, "model_size", 0.0))
        dr = (float(getattr(m, "req_rate", 0.0)) / slo) if slo > 0 else 0.0
        denom = max(1e-9, S - sz)
        return (dr / denom, dr, sz)

    ordered = sorted(models, key=model_key, reverse=True)
    for m in ordered:
        sz = float(getattr(m, "model_size", 0.0))
        slo = float(getattr(m, "slo", 0.0))
        dr = (float(getattr(m, "req_rate", 0.0)) / slo) if slo > 0 else 0.0
        best_gpu = None
        best_res = float('inf')
        for g in range(gpu_num):
            if used_mem[g] + sz <= S + 1e-12:
                new_used = used_mem[g] + sz
                rem = S - new_used
                if rem <= 0:
                    continue
                new_R = sum_R[g] + dr
                new_k = kvpr(new_R, rem)
                res = new_k
                for h in range(gpu_num):
                    if h == g:
                        continue
                    other = kvpr(sum_R[h], S - used_mem[h])
                    if other > res:
                        res = other
                if res < best_res:
                    best_res = res
                    best_gpu = g
        if best_gpu is None:
            raise ValueError("Unable to place model due to memory constraints.")
        placement[best_gpu].append(m)
        used_mem[best_gpu] += sz
        sum_R[best_gpu] += dr
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