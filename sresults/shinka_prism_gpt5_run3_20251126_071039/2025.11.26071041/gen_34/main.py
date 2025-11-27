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
    if gpu_num <= 0:
        raise ValueError("gpu_num must be positive")

    # ---------- Helpers ----------
    def safe_div(num, den):
        if den <= 0:
            return float('inf') if num > 0 else 0.0
        return num / den

    def kvpr_val(numer, rem_mem):
        return safe_div(numer, rem_mem)

    # ---------- Extract per-model stats ----------
    items = []
    total_mem = 0.0
    total_n = 0.0
    for idx, m in enumerate(models):
        ms = float(getattr(m, "model_size"))
        slo = float(getattr(m, "slo"))
        rr = float(getattr(m, "req_rate"))
        if ms < 0:
            raise ValueError("Model size must be non-negative")
        if ms > GPU_MEM_SIZE + 1e-9:
            raise ValueError(f"Model of size {ms} GB cannot fit into a single GPU of size {GPU_MEM_SIZE} GB")
        if slo <= 0:
            raise ValueError("Model SLO must be positive")
        n = rr / slo  # r_j / s_j
        items.append((idx, m, ms, n))
        total_mem += ms
        total_n += n

    if not items:
        return {gpu_id: [] for gpu_id in range(gpu_num)}

    total_capacity_mem = gpu_num * GPU_MEM_SIZE
    if total_mem - total_capacity_mem > 1e-9:
        raise ValueError("Total model memory exceeds total GPU memory")

    # ---------- Lower bound on T ----------
    indiv_lb = max(safe_div(n, GPU_MEM_SIZE - ms) for _, _, ms, n in items)
    global_lb = safe_div(total_n, max(total_capacity_mem - total_mem, 1e-9))

    # Pair bound for models that cannot co-reside on one GPU
    pair_lb = 0.0
    P = min(len(items), 200)
    heavy = sorted(items, key=lambda it: it[2], reverse=True)[:P]
    for i in range(len(heavy)):
        _, _, mi, ni = heavy[i]
        for j in range(i + 1, len(heavy)):
            _, _, mj, nj = heavy[j]
            if mi + mj > GPU_MEM_SIZE + 1e-12:
                denom = 2.0 * GPU_MEM_SIZE - (mi + mj)
                pair_lb = max(pair_lb, safe_div(ni + nj, max(denom, 1e-9)))

    # Small k-prefix bound
    kprefix_lb = 0.0
    items_by_m = sorted(items, key=lambda it: it[2], reverse=True)
    for k in range(1, min(gpu_num, 4) + 1):
        sum_m = 0.0
        sum_n = 0.0
        for it in items_by_m:
            sum_m += it[2]
            sum_n += it[3]
            if sum_m > (k - 1) * GPU_MEM_SIZE + 1e-12:
                break
        denom = k * GPU_MEM_SIZE - sum_m
        kprefix_lb = max(kprefix_lb, safe_div(sum_n, max(denom, 1e-9)))

    low_T = max(0.0, indiv_lb, global_lb, pair_lb, kprefix_lb)

    # ---------- Feasibility check and packing at T ----------
    # order_variant:
    #   0 -> by transformed weight w(T) = n + T*m
    #   1 -> by intrinsic-alone KVPR n / (80 - m)
    #   2 -> by pressure per GB n / m
    # policy:
    #   "resid"  -> best-fit on transformed residual
    #   "minmax" -> choose GPU minimizing new global max KVPR
    def try_pack(T, order_variant=0, policy="resid", return_placement=False):
        cap = GPU_MEM_SIZE * T
        eps = 1e-12

        # Build sorted order
        if order_variant == 0:
            ordered = sorted(items, key=lambda it: (it[3] + T * it[2], it[3], it[2]), reverse=True)
        elif order_variant == 1:
            ordered = sorted(items, key=lambda it: (safe_div(it[3], max(GPU_MEM_SIZE - it[2], 1e-9)), it[2]), reverse=True)
        else:
            ordered = sorted(items, key=lambda it: (safe_div(it[3], max(it[2], 1e-9)), it[3]), reverse=True)

        # Per-GPU state
        n_sum = [0.0] * gpu_num
        m_sum = [0.0] * gpu_num
        used_cap = [0.0] * gpu_num  # equals n_sum + T * m_sum
        placement = [[] for _ in range(gpu_num)]

        for _, mdl, ms, n in ordered:
            w = n + T * ms
            candidates = []
            # Precompute current kvprs
            cur_kvprs = [kvpr_val(n_sum[g], GPU_MEM_SIZE - m_sum[g]) for g in range(gpu_num)]

            for g in range(gpu_num):
                if m_sum[g] + ms > GPU_MEM_SIZE + eps:
                    continue
                residual = cap - (used_cap[g] + w)
                if residual < -eps:
                    continue
                if policy == "resid":
                    candidates.append((g, residual, None, None))
                else:
                    new_n = n_sum[g] + n
                    new_m = m_sum[g] + ms
                    new_local = kvpr_val(new_n, GPU_MEM_SIZE - new_m)
                    # New global max if placed on g
                    new_max = new_local
                    for k in range(gpu_num):
                        if k != g and cur_kvprs[k] > new_max:
                            new_max = cur_kvprs[k]
                    candidates.append((g, residual, new_max, new_local))

            if not candidates:
                return (False, None) if return_placement else False

            if policy == "resid":
                # Best-fit: minimize transformed residual; tie by more remaining mem then gpu id
                # Remaining mem after placement equals GPU_MEM_SIZE - (m_sum[g]+ms)
                chosen = min(
                    candidates,
                    key=lambda c: (c[1], -(GPU_MEM_SIZE - (m_sum[c[0]] + ms)), c[0])
                )[0]
            else:
                # Minimize global new max KVPR; tie-break by local KVPR then residual then more remaining mem then id
                chosen = min(
                    candidates,
                    key=lambda c: (c[2], c[3], c[1], -(GPU_MEM_SIZE - (m_sum[c[0]] + ms)), c[0])
                )[0]

            # Place
            placement[chosen].append(mdl)
            n_sum[chosen] += n
            m_sum[chosen] += ms
            used_cap[chosen] += w

        if return_placement:
            return True, {g: placement[g] for g in range(gpu_num)}
        return True

    # For quick feasibility checks during search
    def try_pack_any(T, need_placement=False):
        variants = [(0, "resid"), (1, "resid"), (0, "minmax")]
        feasibles = []
        for ov, pol in variants:
            if need_placement:
                ok, plc = try_pack(T, ov, pol, True)
                if ok:
                    feasibles.append(plc)
            else:
                if try_pack(T, ov, pol, False):
                    return True
        if need_placement:
            return (len(feasibles) > 0), feasibles
        return False

    # ---------- Exponential search for feasible T ----------
    T = max(low_T, 1e-9)
    found = False
    for _ in range(50):
        if try_pack_any(T, need_placement=False):
            found = True
            break
        T *= 2.0

    if not found:
        raise ValueError("Unable to find a feasible packing for any KVPR threshold")

    # ---------- Binary search to minimize T ----------
    low, high = low_T, T
    for _ in range(40):
        mid = (low + high) / 2.0
        if mid <= 0:
            high = mid
            continue
        if try_pack_any(mid, need_placement=False):
            high = mid
        else:
            low = mid

    # ---------- Build diverse candidates at near-optimal T ----------
    def measured_max_kvpr(plc):
        vals = []
        for g in range(gpu_num):
            used_mem = sum(getattr(m, "model_size") for m in plc.get(g, []))
            numer = sum((getattr(m, "req_rate") / getattr(m, "slo")) for m in plc.get(g, []))
            vals.append(kvpr_val(numer, GPU_MEM_SIZE - used_mem))
        return max(vals) if vals else 0.0

    # Additional greedy min-max candidate (ignores T; memory-feasible)
    def greedy_minmax_candidate():
        # Sort by pressure per GB, then by demand
        def key_m(m):
            dn = (m.req_rate / m.slo) if m.slo != 0 else float('inf')
            sz = m.model_size if m.model_size > 0 else 1e-9
            return (dn / sz, dn)
        ordered = sorted(models, key=key_m, reverse=True)

        plc = {g: [] for g in range(gpu_num)}
        rem = [GPU_MEM_SIZE] * gpu_num
        numer = [0.0] * gpu_num

        for m in ordered:
            ms = float(m.model_size)
            dn = float(m.req_rate) / float(m.slo)
            # Precompute current kvprs
            cur_k = [kvpr_val(numer[g], rem[g]) for g in range(gpu_num)]
            best = None
            for g in range(gpu_num):
                if ms <= rem[g]:
                    new_local = kvpr_val(numer[g] + dn, rem[g] - ms)
                    new_max = new_local
                    for k in range(gpu_num):
                        if k != g and cur_k[k] > new_max:
                            new_max = cur_k[k]
                    # tie-breaks: min new_max, then min local, then more remaining mem, then gpu id
                    key = (new_max, new_local, -(rem[g] - ms), g)
                    if best is None or key < best[0]:
                        best = (key, g)
            if best is None:
                return None
            g = best[1]
            plc[g].append(m)
            rem[g] -= ms
            numer[g] += dn
        return plc

    candidates = []

    # Evaluate multiple T variants near the optimal threshold
    Ts = [high, high * 0.995, high * 1.005]
    combos = [(0, "resid"), (1, "resid"), (2, "resid"), (0, "minmax"), (1, "minmax")]
    for Tv in Ts:
        for ov, pol in combos:
            ok, plc = try_pack(Tv, ov, pol, True)
            if ok:
                candidates.append(plc)

    # Add greedy candidate
    gu = greedy_minmax_candidate()
    if gu is not None:
        candidates.append(gu)

    if not candidates:
        # Fallback: at least one feasible must exist
        ok, plc = try_pack(high, 0, "resid", True)
        if not ok:
            raise ValueError("Feasible packing unexpectedly unavailable")
        candidates.append(plc)

    # ---------- Select best candidate by measured KVPR profile ----------
    def score_tuple(plc):
        kvprs = []
        for g in range(gpu_num):
            used_mem = sum(getattr(m, "model_size") for m in plc.get(g, []))
            numer = sum((getattr(m, "req_rate") / getattr(m, "slo")) for m in plc.get(g, []))
            kvprs.append(kvpr_val(numer, GPU_MEM_SIZE - used_mem))
        if not kvprs:
            return (0.0, 0.0, 0.0)
        kvprs.sort(reverse=True)
        first = kvprs[0]
        second = kvprs[1] if len(kvprs) > 1 else first
        avg = sum(kvprs) / len(kvprs)
        return (first, second, avg)

    best_plc = None
    best_score = None
    for plc in candidates:
        sc = score_tuple(plc)
        if best_score is None or sc < best_score:
            best_score = sc
            best_plc = plc

    # ---------- Lightweight local improvement ----------
    def local_improve(plc, max_moves=200, eps=1e-12):
        per_g = {g: list(plc.get(g, [])) for g in range(gpu_num)}
        mem = [sum(getattr(m, "model_size") for m in per_g[g]) for g in range(gpu_num)]
        num = [sum((getattr(m, "req_rate") / getattr(m, "slo")) for m in per_g[g]) for g in range(gpu_num)]

        def kv_g(g, msum=None, nsum=None):
            msum = mem[g] if msum is None else msum
            nsum = num[g] if nsum is None else nsum
            return kvpr_val(nsum, GPU_MEM_SIZE - msum)

        def global_vals():
            vals = [kv_g(g) for g in range(gpu_num)]
            return max(vals), vals

        moves = 0
        while moves < max_moves:
            cur_max, vals = global_vals()
            worst = max(range(gpu_num), key=lambda g: vals[g])

            improved = False
            best_new_max = cur_max
            best_move = None

            for mdl in list(per_g[worst]):
                ms = float(getattr(mdl, "model_size"))
                dn = float(getattr(mdl, "req_rate")) / float(getattr(mdl, "slo"))
                for tgt in range(gpu_num):
                    if tgt == worst:
                        continue
                    if mem[tgt] + ms > GPU_MEM_SIZE + 1e-12:
                        continue

                    src_mem = mem[worst] - ms
                    src_num = num[worst] - dn
                    tgt_mem = mem[tgt] + ms
                    tgt_num = num[tgt] + dn

                    src_k = kvpr_val(src_num, GPU_MEM_SIZE - src_mem)
                    tgt_k = kvpr_val(tgt_num, GPU_MEM_SIZE - tgt_mem)

                    new_max = max(src_k, tgt_k)
                    for g in range(gpu_num):
                        if g != worst and g != tgt:
                            if vals[g] > new_max:
                                new_max = vals[g]

                    if new_max + eps < best_new_max:
                        best_new_max = new_max
                        best_move = (mdl, worst, tgt, ms, dn)
                        improved = True

            if not improved:
                break

            mdl, src, tgt, ms, dn = best_move
            per_g[src].remove(mdl)
            per_g[tgt].append(mdl)
            mem[src] -= ms
            num[src] -= dn
            mem[tgt] += ms
            num[tgt] += dn
            moves += 1

        # First-improving swaps involving the current worst GPU (bounded)
        swap_budget = 8
        for _ in range(swap_budget):
            cur_max, vals = global_vals()
            worst = max(range(gpu_num), key=lambda g: vals[g])
            improved = False

            for mdl_a in list(per_g[worst]):
                ms_a = float(getattr(mdl_a, "model_size"))
                dn_a = float(getattr(mdl_a, "req_rate")) / float(getattr(mdl_a, "slo"))
                for tgt in range(gpu_num):
                    if tgt == worst:
                        continue
                    for mdl_b in list(per_g[tgt]):
                        ms_b = float(getattr(mdl_b, "model_size"))
                        dn_b = float(getattr(mdl_b, "req_rate")) / float(getattr(mdl_b, "slo"))

                        # Memory feasibility
                        if mem[worst] - ms_a + ms_b > GPU_MEM_SIZE + 1e-12:
                            continue
                        if mem[tgt] - ms_b + ms_a > GPU_MEM_SIZE + 1e-12:
                            continue

                        src_mem = mem[worst] - ms_a + ms_b
                        src_num = num[worst] - dn_a + dn_b
                        tgt_mem = mem[tgt] - ms_b + ms_a
                        tgt_num = num[tgt] - dn_b + dn_a

                        src_k = kvpr_val(src_num, GPU_MEM_SIZE - src_mem)
                        tgt_k = kvpr_val(tgt_num, GPU_MEM_SIZE - tgt_mem)

                        new_max = max(src_k, tgt_k)
                        for g in range(gpu_num):
                            if g != worst and g != tgt and vals[g] > new_max:
                                new_max = vals[g]

                        if new_max + eps < cur_max:
                            # Apply swap
                            per_g[worst].remove(mdl_a)
                            per_g[tgt].remove(mdl_b)
                            per_g[worst].append(mdl_b)
                            per_g[tgt].append(mdl_a)
                            mem[worst] = src_mem
                            num[worst] = src_num
                            mem[tgt] = tgt_mem
                            num[tgt] = tgt_num
                            improved = True
                            break
                    if improved:
                        break
                if improved:
                    break
            if not improved:
                break

        return {g: per_g.get(g, []) for g in range(gpu_num)}

    best_plc = local_improve(best_plc, max_moves=100)

    # Ensure all GPUs are represented in dict keys
    for g in range(gpu_num):
        best_plc.setdefault(g, [])

    return best_plc

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
