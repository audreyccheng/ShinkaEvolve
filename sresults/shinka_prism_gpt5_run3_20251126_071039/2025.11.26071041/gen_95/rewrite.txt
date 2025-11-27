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

    # Helpers
    def safe_div(num, den):
        if den <= 0:
            return float('inf') if num > 0 else 0.0
        return num / den

    def kvpr(numer, rem_mem):
        return safe_div(numer, rem_mem)

    # Extract and validate model stats
    items = []
    total_mem = 0.0
    total_n = 0.0
    for m in models:
        ms = float(getattr(m, "model_size"))
        slo = float(getattr(m, "slo"))
        rr = float(getattr(m, "req_rate"))
        if ms < 0:
            raise ValueError("Model size must be non-negative")
        if ms > GPU_MEM_SIZE + 1e-9:
            raise ValueError(f"Model of size {ms} GB cannot fit into a single GPU of size {GPU_MEM_SIZE} GB")
        if slo <= 0:
            raise ValueError("Model SLO must be positive")
        n = rr / slo  # r/s
        items.append((m, ms, n))
        total_mem += ms
        total_n += n

    if not items:
        return {g: [] for g in range(gpu_num)}

    total_capacity = gpu_num * GPU_MEM_SIZE
    if total_mem - total_capacity > 1e-9:
        raise ValueError("Total model memory exceeds total GPU memory")

    # Measured max KVPR of a placement
    def measured_max_kvpr(plc):
        msum = [sum(getattr(m, "model_size") for m in plc.get(g, [])) for g in range(gpu_num)]
        nsum = [sum((getattr(m, "req_rate") / getattr(m, "slo")) for m in plc.get(g, [])) for g in range(gpu_num)]
        vals = [kvpr(nsum[g], GPU_MEM_SIZE - msum[g]) for g in range(gpu_num)]
        return max(vals) if vals else 0.0

    # Core MWU-based feasibility oracle for a given T
    def mwu_feasible(T, order_kind="wdesc", seed_jitter=0):
        # Transformed weights
        w = []
        for (mdl, ms, n) in items:
            w.append(n + T * ms)

        # Item ordering strategies
        idxs = list(range(len(items)))
        if order_kind == "wdesc":
            idxs.sort(key=lambda i: (w[i], items[i][2], items[i][1]), reverse=True)
        elif order_kind == "kvpr_desc":
            idxs.sort(key=lambda i: (safe_div(items[i][2], max(GPU_MEM_SIZE - items[i][1], 1e-9)), items[i][1]), reverse=True)
        elif order_kind == "size_desc":
            idxs.sort(key=lambda i: (items[i][1], items[i][2]), reverse=True)
        else:
            idxs.sort(key=lambda i: (w[i], items[i][2], items[i][1]), reverse=True)

        # Dual variables for transformed capacity and raw memory
        # initialize with slight deterministic jitter for diversification
        base = 1.0 + 1e-6 * (1 + seed_jitter)
        lam = [base * (1.0 + 5e-7 * (g + 1) * (seed_jitter + 1)) for g in range(gpu_num)]
        the = [base * (1.0 + 3e-7 * (g + 3) * (seed_jitter + 2)) for g in range(gpu_num)]

        max_rounds = 32
        eta = 0.5

        best_over = float('inf')
        best_assignment = None
        best_loads = None

        for _ in range(max_rounds):
            # Fresh assignment for this round
            loads_w = [0.0] * gpu_num  # sum of w_i
            loads_m = [0.0] * gpu_num  # sum of m_i
            assign = [[] for _ in range(gpu_num)]

            # Greedy assignment under current dual prices
            for i in idxs:
                mdl, ms, n = items[i]
                wi = w[i]

                # cost per GPU: lam_g * w_i + the_g * m_i
                best_g = None
                best_cost = None
                for g in range(gpu_num):
                    cost = lam[g] * wi + the[g] * ms
                    if best_cost is None or cost < best_cost or (cost == best_cost and g < best_g):
                        best_cost = cost
                        best_g = g

                assign[best_g].append(i)
                loads_w[best_g] += wi
                loads_m[best_g] += ms

            # Check feasibility
            cap_w = T * GPU_MEM_SIZE
            feasible = True
            max_over = 0.0
            for g in range(gpu_num):
                over_w = max(0.0, loads_w[g] - cap_w)
                over_m = max(0.0, loads_m[g] - GPU_MEM_SIZE)
                if over_w > 1e-9 or over_m > 1e-9:
                    feasible = False
                # normalized overload
                ow = over_w / max(cap_w, 1e-9)
                om = over_m / max(GPU_MEM_SIZE, 1e-9)
                max_over = max(max_over, ow, om)

            if feasible:
                # Build placement dict from assign
                placement = {g: [items[i][0] for i in assign[g]] for g in range(gpu_num)}
                return True, placement

            # Track best (least overload) assignment in case we need to repair
            if max_over < best_over:
                best_over = max_over
                best_assignment = [list(lst) for lst in assign]
                best_loads = (list(loads_w), list(loads_m))

            # Update duals multiplicatively
            for g in range(gpu_num):
                ow = max(0.0, (loads_w[g] - cap_w) / max(cap_w, 1e-9))
                om = max(0.0, (loads_m[g] - GPU_MEM_SIZE) / GPU_MEM_SIZE)
                lam[g] *= (1.0 + eta * ow)
                the[g] *= (1.0 + eta * om)

            # Normalize to keep scales stable
            avg_lam = sum(lam) / gpu_num
            avg_the = sum(the) / gpu_num
            if avg_lam <= 0:
                avg_lam = 1.0
            if avg_the <= 0:
                avg_the = 1.0
            lam = [max(1e-12, x / avg_lam) for x in lam]
            the = [max(1e-12, x / avg_the) for x in the]

        # Lightweight repair on the best assignment found (if any)
        if best_assignment is None:
            return False, None

        assign = [list(lst) for lst in best_assignment]
        loads_w, loads_m = best_loads
        cap_w = T * GPU_MEM_SIZE

        # Greedy repair: move items out of the most overloaded GPU to the most underloaded GPUs
        # limit moves to keep it cheap
        def try_repair(max_moves=3):
            for _ in range(max_moves):
                # identify the worst GPU by normalized overload
                worst = None
                worst_score = -1.0
                for g in range(gpu_num):
                    ow = max(0.0, (loads_w[g] - cap_w) / max(cap_w, 1e-9))
                    om = max(0.0, (loads_m[g] - GPU_MEM_SIZE) / GPU_MEM_SIZE)
                    score = max(ow, om)
                    if score > worst_score:
                        worst_score = score
                        worst = g
                if worst_score <= 1e-12:
                    break  # already feasible

                improved = False
                # Consider moving the most impactful items first (by wi and then ms)
                cand_list = sorted(assign[worst], key=lambda i: (w[i], items[i][1]), reverse=True)
                for i in cand_list:
                    wi = w[i]; ms = items[i][1]
                    # Try to move to the GPU with most slack
                    best_tgt = None
                    best_gain = 0.0
                    for tgt in range(gpu_num):
                        if tgt == worst:
                            continue
                        new_w_src = loads_w[worst] - wi
                        new_m_src = loads_m[worst] - ms
                        new_w_tgt = loads_w[tgt] + wi
                        new_m_tgt = loads_m[tgt] + ms

                        # compute new worst normalized overload if we move (optimistic)
                        def over_score(wv, mv):
                            ow = max(0.0, (wv - cap_w) / max(cap_w, 1e-9))
                            om = max(0.0, (mv - GPU_MEM_SIZE) / GPU_MEM_SIZE)
                            return max(ow, om)

                        src_sc = over_score(new_w_src, new_m_src)
                        tgt_sc = over_score(new_w_tgt, new_m_tgt)
                        # rough target of the whole system's worst after move
                        # evaluate current others
                        worst_after = max(src_sc, tgt_sc)
                        for g in range(gpu_num):
                            if g != worst and g != tgt:
                                ow = max(0.0, (loads_w[g] - cap_w) / max(cap_w, 1e-9))
                                om = max(0.0, (loads_m[g] - GPU_MEM_SIZE) / GPU_MEM_SIZE)
                                worst_after = max(worst_after, ow, om)

                        gain = worst_score - worst_after
                        if gain > best_gain + 1e-15:
                            best_gain = gain
                            best_tgt = tgt

                    if best_tgt is not None and best_gain > 0:
                        # apply move
                        assign[worst].remove(i)
                        assign[best_tgt].append(i)
                        loads_w[worst] -= w[i]
                        loads_m[worst] -= items[i][1]
                        loads_w[best_tgt] += w[i]
                        loads_m[best_tgt] += items[i][1]
                        improved = True
                        break

                if not improved:
                    break  # no improving move found

            # final feasibility check
            ok = True
            for g in range(gpu_num):
                if loads_w[g] > cap_w + 1e-9 or loads_m[g] > GPU_MEM_SIZE + 1e-9:
                    ok = False
                    break
            return ok

        if try_repair(max_moves=4):
            placement = {g: [items[i][0] for i in assign[g]] for g in range(gpu_num)}
            return True, placement

        return False, None

    # Feasibility wrapper testing multiple restarts/orderings
    def feasible_any(T, need_plc=False):
        orders = ["wdesc", "kvpr_desc", "size_desc"]
        best_plc = None
        for sidx, ordk in enumerate(orders):
            ok, plc = mwu_feasible(T, ordk, seed_jitter=sidx)
            if ok:
                if need_plc:
                    # pick the one with smallest measured KVPR
                    if best_plc is None or measured_max_kvpr(plc) < measured_max_kvpr(best_plc):
                        best_plc = plc
                else:
                    return True, plc if need_plc else True
        if need_plc:
            return (best_plc is not None), best_plc
        return False, None

    # Lower bounds on T
    indiv_lb = max(safe_div(n, max(GPU_MEM_SIZE - ms, 1e-9)) for _, ms, n in items)
    global_den = max(gpu_num * GPU_MEM_SIZE - total_mem, 1e-9)
    global_lb = safe_div(total_n, global_den)
    # Lightweight pair bound over largest-by-size items
    pair_lb = 0.0
    if gpu_num >= 2 and len(items) >= 2:
        L = min(len(items), 80)
        heavy = sorted(items, key=lambda it: it[1], reverse=True)[:L]
        for i in range(len(heavy)):
            mi, ni = heavy[i][1], heavy[i][2]
            for j in range(i + 1, len(heavy)):
                mj, nj = heavy[j][1], heavy[j][2]
                if mi + mj > GPU_MEM_SIZE + 1e-12:
                    denom = 2.0 * GPU_MEM_SIZE - (mi + mj)
                    pair_lb = max(pair_lb, safe_div(ni + nj, max(denom, 1e-9)))
    low_T = max(0.0, indiv_lb, global_lb, pair_lb)

    # Exponential search for the first feasible T
    T = max(low_T, 1e-9)
    ok, plc = feasible_any(T, need_plc=True)
    tries = 0
    while not ok and tries < 50:
        T *= 2.0
        ok, plc = feasible_any(T, need_plc=True)
        tries += 1
    if not ok:
        # As a last resort, place by memory only (best-fit) – should be rare
        ordered = sorted(items, key=lambda it: it[1], reverse=True)
        placement = {g: [] for g in range(gpu_num)}
        rem = [GPU_MEM_SIZE] * gpu_num
        for mdl, ms, n in ordered:
            best = None
            for g in range(gpu_num):
                if ms <= rem[g]:
                    r = rem[g] - ms
                    if best is None or r < best[0] or (r == best[0] and g < best[1]):
                        best = (r, g)
            if best is None:
                raise ValueError("Unable to construct any feasible placement")
            g = best[1]
            placement[g].append(mdl)
            rem[g] -= ms
        # ensure keys
        for g in range(gpu_num):
            placement.setdefault(g, [])
        return placement

    # Binary search to tighten T to the smallest feasible
    lo, hi = low_T, T
    best_T = hi
    best_plc = plc
    for _ in range(28):
        mid = (lo + hi) / 2.0
        ok, plc_mid = feasible_any(mid, need_plc=True)
        if ok:
            hi = mid
            best_T = mid
            best_plc = plc_mid
        else:
            lo = mid

    # Probe a tiny neighborhood and run multi-order restarts; pick the best measured placement
    candidates = []
    for Tv in (best_T * 0.99, best_T, best_T * 1.01):
        ok, plc_v = feasible_any(Tv, need_plc=True)
        if ok and plc_v is not None:
            candidates.append(plc_v)

    if not candidates:
        candidates = [best_plc]

    # Choose the one with minimal measured max KVPR; tie-break by lexicographic KVPR vector
    def score_plc(plc):
        kvprs = []
        for g in range(gpu_num):
            used_m = sum(getattr(m, "model_size") for m in plc.get(g, []))
            numer = sum((getattr(m, "req_rate") / getattr(m, "slo")) for m in plc.get(g, []))
            kvprs.append(kvpr(numer, GPU_MEM_SIZE - used_m))
        if not kvprs:
            return (0.0, ())
        return (max(kvprs), tuple(sorted(kvprs, reverse=True)))

    best = None
    best_score = (float('inf'), ())
    for plc in candidates:
        sc = score_plc(plc)
        if sc < best_score:
            best_score = sc
            best = plc

    # Ensure all GPUs are present
    for g in range(gpu_num):
        best.setdefault(g, [])

    return best

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