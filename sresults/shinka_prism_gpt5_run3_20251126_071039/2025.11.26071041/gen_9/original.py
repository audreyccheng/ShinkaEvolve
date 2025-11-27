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

    # Helper
    def safe_div(num, den):
        if den <= 0:
            return float('inf') if num > 0 else 0.0
        return num / den

    # Extract per-model stats, validate
    items = []
    total_mem = 0.0
    total_n = 0.0
    for idx, m in enumerate(models):
        ms = float(getattr(m, "model_size"))
        if ms < 0:
            raise ValueError("Model size must be non-negative")
        if ms > GPU_MEM_SIZE + 1e-9:
            raise ValueError(f"Model of size {ms} GB cannot fit into a single GPU of size {GPU_MEM_SIZE} GB")
        slo = float(getattr(m, "slo"))
        if slo <= 0:
            raise ValueError("Model SLO must be positive")
        n = float(getattr(m, "req_rate")) / slo  # r_j / s_j
        items.append((idx, m, ms, n))
        total_mem += ms
        total_n += n

    # Empty input -> empty placement
    if not items:
        return {gpu_id: [] for gpu_id in range(gpu_num)}

    # Capacity sanity
    total_capacity_mem = gpu_num * GPU_MEM_SIZE
    if total_mem - total_capacity_mem > 1e-9:
        raise ValueError("Total model memory exceeds total GPU memory")

    # Lower bound on T
    indiv_lb = max(safe_div(n, GPU_MEM_SIZE - ms) for _, _, ms, n in items)
    global_lb = safe_div(total_n, max(total_capacity_mem - total_mem, 1e-9))
    low_T = max(0.0, indiv_lb, global_lb)

    # Pack a given T using various orderings and a policy that minimizes global max KVPR
    def pack_once(T, order_variant):
        cap = GPU_MEM_SIZE * T  # transformed capacity per GPU
        # Build orderings
        if order_variant == 0:
            key_fn = lambda it: (it[3] + T * it[2], it[3], it[2])  # w(T), n, m
        elif order_variant == 1:
            # Pressure per GB: (n/m, n)
            key_fn = lambda it: ((it[3] / (it[2] if it[2] > 0 else 1e-9)), it[3])
        elif order_variant == 2:
            # Intrinsic KVPR if alone: n / (80 - m)
            key_fn = lambda it: (safe_div(it[3], max(GPU_MEM_SIZE - it[2], 1e-9)), it[2])
        elif order_variant == 3:
            # Larger models first, then higher n
            key_fn = lambda it: (it[2], it[3])
        else:
            # Higher demand first, then larger size
            key_fn = lambda it: (it[3], it[2])

        ordered = sorted(items, key=key_fn, reverse=True)

        # Per-GPU state
        n_sum = [0.0] * gpu_num
        m_sum = [0.0] * gpu_num
        used_cap = [0.0] * gpu_num
        place_lists = [[] for _ in range(gpu_num)]

        def gpu_kvpr(g):
            return safe_div(n_sum[g], GPU_MEM_SIZE - m_sum[g])

        for it in ordered:
            _, mdl, ms, n = it
            w = n + T * ms

            # Precompute current kvprs
            cur_kvprs = [gpu_kvpr(g) for g in range(gpu_num)]
            best_g = None
            best_new_max = float('inf')
            best_local = float('inf')
            best_rem = -1.0
            best_residual = float('inf')

            for g in range(gpu_num):
                # Memory and transformed capacity checks
                if m_sum[g] + ms > GPU_MEM_SIZE + 1e-12:
                    continue
                if used_cap[g] + w > cap + 1e-12:
                    continue
                new_n = n_sum[g] + n
                new_m = m_sum[g] + ms
                new_local = safe_div(new_n, GPU_MEM_SIZE - new_m)
                # New global max if placed on g
                new_max = new_local
                for k in range(gpu_num):
                    if k != g and cur_kvprs[k] > new_max:
                        new_max = cur_kvprs[k]
                # Tie-break: min new_max, then min local, then max remaining mem, then min transformed residual, then min gpu id
                new_rem_mem = GPU_MEM_SIZE - new_m
                residual = (cap - (used_cap[g] + w))
                if (new_max < best_new_max or
                    (new_max == best_new_max and new_local < best_local) or
                    (new_max == best_new_max and new_local == best_local and new_rem_mem > best_rem) or
                    (new_max == best_new_max and new_local == best_local and new_rem_mem == best_rem and residual < best_residual) or
                    (new_max == best_new_max and new_local == best_local and new_rem_mem == best_rem and residual == best_residual and (best_g is None or g < best_g))):
                    best_g = g
                    best_new_max = new_max
                    best_local = new_local
                    best_rem = new_rem_mem
                    best_residual = residual

            if best_g is None:
                return False, None

            # Place
            place_lists[best_g].append(mdl)
            n_sum[best_g] += n
            m_sum[best_g] += ms
            used_cap[best_g] += w

        return True, {g: place_lists[g] for g in range(gpu_num)}

    # Try multiple orderings for a given T
    def pack_with_variants(T, return_all=False):
        feasibles = []
        for ov in range(5):
            ok, plc = pack_once(T, ov)
            if ok:
                if return_all:
                    feasibles.append(plc)
                else:
                    return True, plc
        if return_all:
            return (len(feasibles) > 0), feasibles
        return False, None

    # Exponential search for initial feasible T
    T = max(low_T, 1e-9)
    feasible = False
    for _ in range(60):
        ok, _ = pack_with_variants(T)
        if ok:
            feasible = True
            break
        T *= 2.0

    if not feasible:
        raise ValueError("Unable to find a feasible packing for any KVPR threshold")

    # Binary search to minimize T
    low, high = low_T, T
    for _ in range(40):
        mid = (low + high) / 2.0
        if mid <= 0:
            high = mid
            continue
        ok, _ = pack_with_variants(mid)
        if ok:
            high = mid
        else:
            low = mid

    # Build candidates at near-optimal T and add a greedy unconstrained candidate
    def greedy_unconstrained():
        # Pressure per GB ordering from inspiration code
        def _pressure_per_gb(m):
            rs = (m.req_rate / m.slo) if m.slo != 0 else float('inf')
            size = m.model_size if m.model_size > 0 else 1e-9
            return (rs / size, rs)
        sorted_models = sorted(models, key=_pressure_per_gb, reverse=True)

        placement = {g: [] for g in range(gpu_num)}
        rem = [GPU_MEM_SIZE] * gpu_num
        numer = [0.0] * gpu_num

        def kvpr(nr, rm):
            return (nr / rm) if rm > 0 else float('inf')

        for model in sorted_models:
            delta = model.req_rate / model.slo
            best_idx = None
            best_max = float('inf')
            best_local = float('inf')
            best_rem = -1.0

            current_kvprs = [kvpr(numer[g], rem[g]) for g in range(gpu_num)]
            for g in range(gpu_num):
                if model.model_size <= rem[g]:
                    new_n = numer[g] + delta
                    new_rm = rem[g] - model.model_size
                    new_local = kvpr(new_n, new_rm)
                    new_max = new_local
                    for k in range(gpu_num):
                        if k != g and current_kvprs[k] > new_max:
                            new_max = current_kvprs[k]

                    if (new_max < best_max or
                        (new_max == best_max and new_local < best_local) or
                        (new_max == best_max and new_local == best_local and new_rm > best_rem) or
                        (new_max == best_max and new_local == best_local and new_rm == best_rem and (best_idx is None or g < best_idx))):
                        best_idx = g
                        best_max = new_max
                        best_local = new_local
                        best_rem = new_rm

            if best_idx is None:
                # No fit in memory; fail this candidate
                return None

            placement[best_idx].append(model)
            numer[best_idx] += delta
            rem[best_idx] -= model.model_size

        return placement

    ok_all, candidates = pack_with_variants(high, return_all=True)
    if not ok_all:
        # Should not happen, but fallback to greedy unconstrained
        gu = greedy_unconstrained()
        if gu is None:
            raise ValueError("Unable to construct any feasible placement")
        placement = gu
    else:
        # Add greedy unconstrained candidate for diversity
        gu = greedy_unconstrained()
        if gu is not None:
            candidates.append(gu)

        # Score candidates by measured max KVPR
        def measured_max_kvpr(plc):
            kvprs = []
            for g in range(gpu_num):
                used = sum(getattr(m, 'model_size') for m in plc.get(g, []))
                numer = sum((getattr(m, 'req_rate') / getattr(m, 'slo')) for m in plc.get(g, []))
                kvprs.append(safe_div(numer, GPU_MEM_SIZE - used))
            return max(kvprs) if kvprs else 0.0

        best_plc = None
        best_score = float('inf')
        for plc in candidates:
            score = measured_max_kvpr(plc)
            if score < best_score:
                best_score = score
                best_plc = plc
        placement = best_plc

    # Local improvement: move from worst GPU to reduce global max KVPR
    def local_improve(plc, max_iters=5000, eps=1e-12):
        # Initialize state
        per_g = {g: list(plc.get(g, [])) for g in range(gpu_num)}
        mem = [sum(getattr(m, 'model_size') for m in per_g[g]) for g in range(gpu_num)]
        num = [sum((getattr(m, 'req_rate') / getattr(m, 'slo')) for m in per_g[g]) for g in range(gpu_num)]

        def kvpr_g(g, msum=None, nsum=None):
            msum = mem[g] if msum is None else msum
            nsum = num[g] if nsum is None else nsum
            return safe_div(nsum, GPU_MEM_SIZE - msum)

        def global_max():
            vals = [kvpr_g(g) for g in range(gpu_num)]
            return (max(vals), vals)

        iters = 0
        while iters < max_iters:
            iters += 1
            cur_max, cur_vals = global_max()
            worst_g = max(range(gpu_num), key=lambda g: cur_vals[g])

            improved = False
            best_move = None
            best_new_max = cur_max

            # Try moving each model from worst_g to others
            for mdl in list(per_g[worst_g]):
                ms = float(getattr(mdl, 'model_size'))
                dn = float(getattr(mdl, 'req_rate')) / float(getattr(mdl, 'slo'))
                for tgt in range(gpu_num):
                    if tgt == worst_g:
                        continue
                    if mem[tgt] + ms > GPU_MEM_SIZE + 1e-12:
                        continue
                    # New per-GPU states
                    src_mem = mem[worst_g] - ms
                    src_num = num[worst_g] - dn
                    tgt_mem = mem[tgt] + ms
                    tgt_num = num[tgt] + dn

                    src_kv = kvpr_g(worst_g, src_mem, src_num)
                    tgt_kv = kvpr_g(tgt, tgt_mem, tgt_num)
                    new_max = tgt_kv
                    for g in range(gpu_num):
                        if g != worst_g and g != tgt:
                            if cur_vals[g] > new_max:
                                new_max = cur_vals[g]
                    if src_kv > new_max:
                        new_max = src_kv

                    if new_max + eps < best_new_max:
                        best_new_max = new_max
                        best_move = (mdl, worst_g, tgt, ms, dn)
                        improved = True

            if not improved:
                break

            # Apply best move
            mdl, src, tgt, ms, dn = best_move
            per_g[src].remove(mdl)
            per_g[tgt].append(mdl)
            mem[src] -= ms
            num[src] -= dn
            mem[tgt] += ms
            num[tgt] += dn

        return {g: per_g.get(g, []) for g in range(gpu_num)}

    placement = local_improve(placement)

    # Ensure all GPUs are present in dict keys
    for g in range(gpu_num):
        placement.setdefault(g, [])

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

