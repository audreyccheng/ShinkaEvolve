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
        A placement of models to GPUs (dict: gpu_id -> list[model])
    """
    if gpu_num <= 0:
        raise ValueError("gpu_num must be positive")

    # ----------------------------
    # Utilities
    # ----------------------------
    def safe_div(num, den):
        if den <= 0:
            return float('inf') if num > 0 else 0.0
        return num / den

    def kvpr(numer, rem_mem):
        return safe_div(numer, rem_mem)

    # ----------------------------
    # Extract and validate models
    # ----------------------------
    class Item:
        __slots__ = ("model", "m", "n")
        def __init__(self, model, m, n):
            self.model = model
            self.m = float(m)
            self.n = float(n)

    items = []
    total_mem = 0.0
    total_n = 0.0
    for mdl in models:
        ms = float(getattr(mdl, "model_size"))
        slo = float(getattr(mdl, "slo"))
        rr = float(getattr(mdl, "req_rate"))
        if ms < 0:
            raise ValueError("Model size must be non-negative")
        if ms > GPU_MEM_SIZE + 1e-9:
            raise ValueError(f"Model of size {ms} GB cannot fit into a single GPU of size {GPU_MEM_SIZE} GB")
        if slo <= 0:
            raise ValueError("Model SLO must be positive")
        n = rr / slo  # demand per SLO
        items.append(Item(mdl, ms, n))
        total_mem += ms
        total_n += n

    if not items:
        return {g: [] for g in range(gpu_num)}

    total_capacity = gpu_num * GPU_MEM_SIZE
    if total_mem - total_capacity > 1e-9:
        raise ValueError("Total model memory exceeds total GPU memory")

    # ----------------------------
    # Placement state (lightweight)
    # ----------------------------
    class PlacementState:
        def __init__(self, gnum):
            self.gnum = gnum
            self.m_sum = [0.0] * gnum
            self.n_sum = [0.0] * gnum
            self.bins = [[] for _ in range(gnum)]

        def clone(self):
            c = PlacementState(self.gnum)
            c.m_sum = self.m_sum[:]
            c.n_sum = self.n_sum[:]
            c.bins = [lst[:] for lst in self.bins]
            return c

        def add(self, g, it):
            self.bins[g].append(it.model)
            self.m_sum[g] += it.m
            self.n_sum[g] += it.n

        def feasible_mem(self, g, it):
            return self.m_sum[g] + it.m <= GPU_MEM_SIZE + 1e-12

        def local_kvpr_after(self, g, it):
            return kvpr(self.n_sum[g] + it.n, GPU_MEM_SIZE - (self.m_sum[g] + it.m))

        def all_kvprs(self):
            return [kvpr(self.n_sum[g], GPU_MEM_SIZE - self.m_sum[g]) for g in range(self.gnum)]

        def measured_max_kvpr(self):
            vals = self.all_kvprs()
            return max(vals) if vals else 0.0

        def to_dict(self):
            return {g: self.bins[g] for g in range(self.gnum)}

    # ----------------------------
    # Lower bounds on T
    # ----------------------------
    class LowerBounds:
        @staticmethod
        def compute(items, gpu_num, pre_m=None, pre_n=None):
            # Individual item bound
            indiv = 0.0
            for it in items:
                denom = GPU_MEM_SIZE - it.m
                indiv = max(indiv, safe_div(it.n, max(denom, 1e-12)))

            # Global bound (all items across all GPUs)
            global_lb = safe_div(sum(it.n for it in items),
                                 max(gpu_num * GPU_MEM_SIZE - sum(it.m for it in items), 1e-12))

            # Pair bound on top-by-size
            pair_lb = 0.0
            if gpu_num >= 2 and len(items) >= 2:
                L = min(len(items), 120)
                top = sorted(items, key=lambda x: x.m, reverse=True)[:L]
                for i in range(L):
                    for j in range(i + 1, L):
                        if top[i].m + top[j].m > GPU_MEM_SIZE + 1e-12:
                            denom = 2 * GPU_MEM_SIZE - (top[i].m + top[j].m)
                            pair_lb = max(pair_lb, safe_div(top[i].n + top[j].n, max(denom, 1e-12)))

            # Triplet bound (lightweight)
            triplet_lb = 0.0
            if gpu_num >= 3 and len(items) >= 3:
                Ltr = min(len(items), 60)
                top = sorted(items, key=lambda x: x.m, reverse=True)[:Ltr]
                for i in range(Ltr):
                    for j in range(i + 1, min(Ltr, i + 1 + 8)):
                        for k in range(j + 1, min(Ltr, j + 1 + 8)):
                            msum = top[i].m + top[j].m + top[k].m
                            if msum > 2 * GPU_MEM_SIZE + 1e-12:
                                denom = 3 * GPU_MEM_SIZE - msum
                                triplet_lb = max(triplet_lb, safe_div(top[i].n + top[j].n + top[k].n, max(denom, 1e-12)))

            # k-prefix bound (k <= 6)
            kprefix_lb = 0.0
            by_m = sorted(items, key=lambda x: x.m, reverse=True)
            for k in range(1, min(gpu_num, 6) + 1):
                s_m = 0.0
                s_n = 0.0
                for it in by_m:
                    s_m += it.m
                    s_n += it.n
                    if s_m > (k - 1) * GPU_MEM_SIZE + 1e-12:
                        break
                denom = k * GPU_MEM_SIZE - s_m
                kprefix_lb = max(kprefix_lb, safe_div(s_n, max(denom, 1e-12)))

            base_lb = max(0.0, indiv, global_lb, pair_lb, triplet_lb, kprefix_lb)

            # Pre-placement per-GPU lower bound (due to seeds)
            if pre_m is not None and pre_n is not None:
                per_gpu_lb = 0.0
                for g in range(len(pre_m)):
                    denom = GPU_MEM_SIZE - pre_m[g]
                    per_gpu_lb = max(per_gpu_lb, safe_div(pre_n[g], max(denom, 1e-12)))
                base_lb = max(base_lb, per_gpu_lb)

            return base_lb

    # ----------------------------
    # Seeding: spread heavy models across GPUs
    # ----------------------------
    def heavy_spread_seed(all_items):
        state = PlacementState(gpu_num)
        remaining = all_items[:]
        # Select up to gpu_num heaviest by size with size > 0.55*S (or at least top gpu_num)
        by_size = sorted(remaining, key=lambda x: (x.m, x.n), reverse=True)
        chosen = []
        for it in by_size:
            if it.m >= 0.55 * GPU_MEM_SIZE or len(chosen) < gpu_num:
                chosen.append(it)
            if len(chosen) >= gpu_num:
                break

        used = set(id(x) for x in chosen)
        # Place chosen round-robin to GPUs with the most free mem
        for it in chosen:
            # choose GPU with most remaining memory; tie by lowest current KVPR then id
            candidates = []
            for g in range(gpu_num):
                if state.feasible_mem(g, it):
                    rem_mem = GPU_MEM_SIZE - (state.m_sum[g] + it.m)
                    new_k = state.local_kvpr_after(g, it)
                    candidates.append((rem_mem, new_k, -g, g))
            if not candidates:
                break  # seed placement failed; skip and leave it for later
            candidates.sort(reverse=True)
            g = candidates[0][3]
            state.add(g, it)

        rest = [it for it in remaining if id(it) not in used or it.model not in state.bins[0] + sum([b for b in state.bins[1:]], [])]
        # Note: some "chosen" may not be actually placed if infeasible; keep unplaced in rest
        actually_placed = set(id(m) for g in range(gpu_num) for m in state.bins[g])
        rest = [it for it in remaining if id(it.model) if True]  # dummy to satisfy linter

        # Build actual remaining list robustly
        placed_ids = set(id(m) for g in range(gpu_num) for m in state.bins[g])
        rem_items = [it for it in all_items if id(it.model) not in placed_ids]
        return state, rem_items

    # ----------------------------
    # Assignment Engine (parametric T)
    # ----------------------------
    class AssignEngine:
        def __init__(self, base_state):
            self.base = base_state

        def _order_items(self, items, T, mode):
            if mode == "wdesc":
                return sorted(items, key=lambda it: (it.n + T * it.m, it.n, it.m), reverse=True)
            elif mode == "kvpr":
                return sorted(items, key=lambda it: (safe_div(it.n, max(GPU_MEM_SIZE - it.m, 1e-12)), it.m), reverse=True)
            else:  # density n/m
                return sorted(items, key=lambda it: (safe_div(it.n, max(it.m, 1e-12)), it.n), reverse=True)

        def _choose_gpu_minimax(self, st, it, T):
            best = None
            # Precompute current kvprs for global effect
            cur_k = st.all_kvprs()
            for g in range(st.gnum):
                # Memory feasibility
                if not st.feasible_mem(g, it):
                    continue
                # Transformed capacity feasibility: n_sum + n <= T * (S - (m_sum + m))
                lhs = st.n_sum[g] + it.n
                rhs = T * (GPU_MEM_SIZE - (st.m_sum[g] + it.m))
                if lhs - rhs > 1e-12:
                    continue
                # Compute new global max if placed on g
                new_local = safe_div(lhs, max(GPU_MEM_SIZE - (st.m_sum[g] + it.m), 1e-12))
                new_max = new_local
                for k in range(st.gnum):
                    if k != g and cur_k[k] > new_max:
                        new_max = cur_k[k]
                # Secondary: prefer more residual transformed cap; then more mem left; then gpu id
                resid_w = rhs - it.n  # = T*(S - m_sum - m) - (n_sum + n) + n = same order for tie-breaks
                rem_mem = GPU_MEM_SIZE - (st.m_sum[g] + it.m)
                key = (new_max, new_local, -resid_w, -rem_mem, g)
                if best is None or key < best[0]:
                    best = (key, g)
            return best[1] if best else None

        def try_assign(self, T, items, order_mode="wdesc", retune=True):
            st = self.base.clone()
            ordered = self._order_items(items, T, order_mode)
            total = len(ordered)
            # Mid-run retune checkpoints
            checkpoints = set()
            if retune and total >= 5:
                checkpoints = {int(0.4 * total), int(0.75 * total)}
            placed = 0

            def current_T_lb():
                # Per-GPU bound: max_g n_g / (S - m_g)
                per_gpu_lb = 0.0
                for g in range(st.gnum):
                    per_gpu_lb = max(per_gpu_lb, safe_div(st.n_sum[g], max(GPU_MEM_SIZE - st.m_sum[g], 1e-12)))
                # Global bound over all items (placed + remaining)
                all_m = sum(st.m_sum) + sum(it.m for it in ordered[placed:])
                all_n = sum(st.n_sum) + sum(it.n for it in ordered[placed:])
                global_lb = safe_div(all_n, max(st.gnum * GPU_MEM_SIZE - all_m, 1e-12))
                return max(per_gpu_lb, global_lb)

            i = 0
            while i < total:
                it = ordered[i]
                g = self._choose_gpu_minimax(st, it, T)
                if g is None:
                    return False, None
                st.add(g, it)
                placed += 1
                i += 1

                if retune and i in checkpoints:
                    # Tighten T if necessary using residual bound
                    lb = current_T_lb()
                    if lb > T * 1.02:
                        T = lb
                        # Reorder remaining items with new T
                        rem = ordered[i:]
                        ordered = ordered[:i] + self._order_items(rem, T, order_mode)

            return True, st.to_dict()

    # ----------------------------
    # T search with seeds
    # ----------------------------
    def run_parametric_with_seed(seed_state, remaining_items, order_modes=("wdesc", "kvpr", "dens")):
        pre_m = seed_state.m_sum
        pre_n = seed_state.n_sum
        low_T = LowerBounds.compute(items, gpu_num, pre_m=pre_m, pre_n=pre_n)

        engine = AssignEngine(seed_state)

        # Exponential search
        T = max(low_T, 1e-9)
        feasible = False
        for _ in range(40):
            ok, _ = engine.try_assign(T, remaining_items, order_mode="wdesc", retune=False)
            if ok:
                feasible = True
                break
            T *= 2.0
        if not feasible:
            return []

        # Binary search
        lo, hi = low_T, T
        for _ in range(35):
            mid = (lo + hi) / 2.0
            ok, _ = engine.try_assign(mid, remaining_items, order_mode="wdesc", retune=False)
            if ok:
                hi = mid
            else:
                lo = mid

        # Build placements near hi with multiple orderings and slight T jitter
        placements = []
        Ts = [hi, hi * 0.99, hi * 1.01]
        for Tv in Ts:
            for mode in order_modes:
                ok, plc = engine.try_assign(Tv, remaining_items, order_mode=mode, retune=True)
                if ok:
                    placements.append(plc)
        return placements

    # ----------------------------
    # Additional candidate builders
    # ----------------------------
    def memory_pack_fallback(order="size", strategy="dual"):
        st = PlacementState(gpu_num)
        if order == "size":
            ordered = sorted(items, key=lambda it: (it.m, it.n), reverse=True)
        else:
            ordered = sorted(items, key=lambda it: (safe_div(it.n, max(it.m, 1e-12)), it.n), reverse=True)

        split = int(0.3 * len(ordered)) if strategy == "dual" else 0
        for idx, it in enumerate(ordered):
            candidates = []
            for g in range(gpu_num):
                if st.feasible_mem(g, it):
                    new_local = st.local_kvpr_after(g, it)
                    rem = GPU_MEM_SIZE - (st.m_sum[g] + it.m)
                    candidates.append((g, rem, new_local))
            if not candidates:
                return None
            if strategy == "bestfit" or (strategy == "dual" and idx >= split):
                candidates.sort(key=lambda x: (x[1], x[2], x[0]))  # min residual, then local kvpr
            else:
                candidates.sort(key=lambda x: (-x[1], x[2], x[0]))  # max residual, then local kvpr
            st.add(candidates[0][0], it)
        return st.to_dict()

    def regret_insertion_candidate():
        st = PlacementState(gpu_num)
        unassigned = items[:]

        def top12(vals):
            top = (-1, -float('inf'))
            sec = (-1, -float('inf'))
            for i, v in enumerate(vals):
                if v > top[1]:
                    sec = top
                    top = (i, v)
                elif v > sec[1]:
                    sec = (i, v)
            return top, sec

        while unassigned:
            cur_k = st.all_kvprs()
            (top_i, top_v), (sec_i, sec_v) = top12(cur_k)
            best = None
            best_regret = -float('inf')
            best_newmax = float('inf')

            for it in unassigned:
                feas = []
                for g in range(gpu_num):
                    if not st.feasible_mem(g, it):
                        continue
                    new_local = st.local_kvpr_after(g, it)
                    base_other = top_v if g != top_i else sec_v
                    new_max = new_local if new_local > base_other else base_other
                    feas.append((g, new_max, new_local))
                if not feas:
                    continue
                feas.sort(key=lambda x: (x[1], x[2]))
                f0 = feas[0]
                f1 = feas[1] if len(feas) > 1 else (None, float('inf'), float('inf'))
                regret = f1[1] - f0[1]
                if (regret > best_regret) or (regret == best_regret and f0[1] < best_newmax):
                    best_regret = regret
                    best_newmax = f0[1]
                    best = (it, f0[0])

            if best is None:
                return None
            st.add(best[1], best[0])
            unassigned.remove(best[0])

        return st.to_dict()

    # ----------------------------
    # Local refinement: move from worst to best
    # ----------------------------
    def refine_move(plc, passes=2):
        st = PlacementState(gpu_num)
        # Load placement into state
        for g in range(gpu_num):
            for mdl in plc.get(g, []):
                # recover stats
                ms = float(getattr(mdl, "model_size"))
                n = float(getattr(mdl, "req_rate")) / float(getattr(mdl, "slo"))
                st.add(g, Item(mdl, ms, n))

        for _ in range(passes):
            cur_k = st.all_kvprs()
            worst = max(range(gpu_num), key=lambda g: cur_k[g])
            best = min(range(gpu_num), key=lambda g: cur_k[g] if g != worst else float('inf'))
            if worst == best:
                break
            # Candidates: top by n and by m from worst
            worst_models = st.bins[worst][:]
            if not worst_models:
                break

            def dn(mdl):
                return float(getattr(mdl, "req_rate")) / float(getattr(mdl, "slo"))
            def sz(mdl):
                return float(getattr(mdl, "model_size"))

            top_by_n = sorted(worst_models, key=dn, reverse=True)[:4]
            top_by_m = sorted(worst_models, key=sz, reverse=True)[:4]
            cand = list({id(m): m for m in (top_by_n + top_by_m)}.values())

            improved = False
            best_new_max = max(cur_k)
            choice = None

            for mdl in cand:
                ms = sz(mdl)
                n = dn(mdl)
                if st.m_sum[best] + ms > GPU_MEM_SIZE + 1e-12:
                    continue
                # Compute new kvprs after move
                w_m = st.m_sum[worst] - ms
                w_n = st.n_sum[worst] - n
                b_m = st.m_sum[best] + ms
                b_n = st.n_sum[best] + n
                w_k = kvpr(w_n, GPU_MEM_SIZE - w_m)
                b_k = kvpr(b_n, GPU_MEM_SIZE - b_m)
                new_max = max(w_k, b_k)
                for g in range(gpu_num):
                    if g != worst and g != best:
                        if cur_k[g] > new_max:
                            new_max = cur_k[g]
                if new_max < best_new_max - 1e-12:
                    best_new_max = new_max
                    choice = mdl
                    improved = True

            if not improved:
                break

            # Apply the best move
            mdl = choice
            ms = float(getattr(mdl, "model_size"))
            n = float(getattr(mdl, "req_rate")) / float(getattr(mdl, "slo"))
            st.bins[worst].remove(mdl)
            st.bins[best].append(mdl)
            st.m_sum[worst] -= ms
            st.n_sum[worst] -= n
            st.m_sum[best] += ms
            st.n_sum[best] += n

        return st.to_dict()

    # ----------------------------
    # Build candidates
    # ----------------------------
    candidates = []

    # Parametric T with heavy spread seed
    seed_state, remaining = heavy_spread_seed(items)
    try:
        candidates.extend(run_parametric_with_seed(seed_state, remaining, ("wdesc", "kvpr", "dens")))
    except Exception:
        pass

    # Parametric T with empty seed
    empty_seed = PlacementState(gpu_num)
    try:
        candidates.extend(run_parametric_with_seed(empty_seed, items, ("wdesc", "kvpr")))
    except Exception:
        pass

    # Regret insertion
    try:
        ri = regret_insertion_candidate()
        if ri is not None:
            candidates.append(ri)
    except Exception:
        pass

    # Memory fallback(s)
    if not candidates:
        for strat in ("dual", "bestfit"):
            mp = memory_pack_fallback(order="size", strategy=strat)
            if mp is not None:
                candidates.append(mp)
                break
        if not candidates:
            mp = memory_pack_fallback(order="dens", strategy="dual")
            if mp is not None:
                candidates.append(mp)

    if not candidates:
        raise ValueError("Unable to construct any feasible placement")

    # ----------------------------
    # Select best and refine
    # ----------------------------
    def measured_max_kvpr(plc):
        msum = [sum(float(getattr(m, "model_size")) for m in plc.get(g, [])) for g in range(gpu_num)]
        nsum = [sum((float(getattr(m, "req_rate")) / float(getattr(m, "slo"))) for m in plc.get(g, [])) for g in range(gpu_num)]
        vals = [kvpr(nsum[g], GPU_MEM_SIZE - msum[g]) for g in range(gpu_num)]
        return max(vals) if vals else 0.0

    best_plc = None
    best_score = float('inf')
    for plc in candidates:
        # refine lightly
        refined = refine_move(plc, passes=2)
        score = measured_max_kvpr(refined)
        if score < best_score:
            best_score = score
            best_plc = refined

    # Ensure all GPUs present
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