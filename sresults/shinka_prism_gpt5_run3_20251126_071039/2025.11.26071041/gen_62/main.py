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

    # Utility to compute KVPR list and its max
    def kv_list(n_sum, m_sum):
        vals = []
        for g in range(gpu_num):
            vals.append(kvpr(n_sum[g], GPU_MEM_SIZE - m_sum[g]))
        return vals

    def top_two(vals):
        top_i, top_v = -1, -float('inf')
        sec_i, sec_v = -1, -float('inf')
        for i, v in enumerate(vals):
            if v > top_v:
                sec_i, sec_v = top_i, top_v
                top_i, top_v = i, v
            elif v > sec_v:
                sec_i, sec_v = i, v
        return (top_i, top_v), (sec_i, sec_v)

    # Difficulty ordering: place hard items first
    def difficulty_key(it):
        _, ms, n = it
        a = safe_div(n, max(GPU_MEM_SIZE - ms, 1e-12))
        b = safe_div(n, max(ms, 1e-12))
        return (max(a, b), n, ms)

    items_sorted = sorted(items, key=difficulty_key, reverse=True)

    # Beam search parameters
    BEAM_WIDTH = 8
    BRANCH_FACTOR = min(3, gpu_num)
    SAMPLE_LB_ITEMS = 8  # number of hardest remaining items to include in item-based bound

    # Node structure:
    # {
    #   'm': [per-gpu memory sum],
    #   'n': [per-gpu numer sum],
    #   'kv': [per-gpu kvpr],
    #   'plc': {gpu: [models]},
    #   'max': current max kvpr,
    #   'max2': second max kvpr,
    #   'idx_top': index of worst gpu,
    #   'bound': lower bound for completion from this node
    # }

    def compute_bound(node, remaining):
        # Current max on node
        cur_max = node['max']

        # Global remaining feasibility bound
        rem_total_m = sum(ms for _, ms, _ in remaining)
        rem_total_n = sum(n for _, _, n in remaining)
        total_free_mem_now = sum(GPU_MEM_SIZE - node['m'][g] for g in range(gpu_num))
        final_total_free = total_free_mem_now - rem_total_m
        if final_total_free <= 0:
            lb_global = float('inf') if (sum(node['n']) + rem_total_n) > 0 else 0.0
        else:
            lb_global = safe_div(sum(node['n']) + rem_total_n, final_total_free)

        # Quick necessary feasibility: largest remaining item must fit somewhere by memory
        if remaining:
            largest_ms = max(ms for _, ms, _ in remaining)
            # Check if any GPU has at least this much residual memory
            if all((GPU_MEM_SIZE - node['m'][g]) + 1e-12 < largest_ms for g in range(gpu_num)):
                return float('inf')

        # Item-based bound: for top few remaining items, compute minimal achievable local kvpr
        lb_item = 0.0
        if remaining:
            rem_sorted = sorted(remaining, key=difficulty_key, reverse=True)[:SAMPLE_LB_ITEMS]
            for _, ms, n in rem_sorted:
                best_local = float('inf')
                for g in range(gpu_num):
                    if node['m'][g] + ms <= GPU_MEM_SIZE + 1e-12:
                        new_local = kvpr(node['n'][g] + n, GPU_MEM_SIZE - (node['m'][g] + ms))
                        if new_local < best_local:
                            best_local = new_local
                if best_local == float('inf'):
                    return float('inf')
                if best_local > lb_item:
                    lb_item = best_local

        return max(cur_max, lb_global, lb_item)

    # Initialize root node
    root = {
        'm': [0.0] * gpu_num,
        'n': [0.0] * gpu_num,
        'kv': [0.0] * gpu_num,
        'plc': {g: [] for g in range(gpu_num)},
        'max': 0.0,
        'max2': 0.0,
        'idx_top': 0,
        'bound': 0.0
    }

    # Iteratively expand beam
    beam = [root]
    for t, (mdl, ms, n) in enumerate(items_sorted):
        new_beam = []
        # For each node, get top choices for this item
        for node in beam:
            # Build ranking of GPUs for this item by projected new global max KVPR
            choices = []
            (top_idx, top_val), (sec_idx, sec_val) = top_two(node['kv'])
            for g in range(gpu_num):
                if node['m'][g] + ms > GPU_MEM_SIZE + 1e-12:
                    continue
                new_local = kvpr(node['n'][g] + n, GPU_MEM_SIZE - (node['m'][g] + ms))
                base_other = top_val if g != top_idx else sec_val
                new_max = new_local if new_local > base_other else base_other
                rem_after = GPU_MEM_SIZE - (node['m'][g] + ms)
                # Rank by smaller new_max, then smaller local kvpr, then more remaining mem, then gpu id
                choices.append((new_max, new_local, -rem_after, g))
            if not choices:
                # This partial cannot place this item; drop
                continue
            choices.sort()
            # Expand only top K choices
            for rank in choices[:BRANCH_FACTOR]:
                _, _, _, g = rank
                # Create child node
                m_child = list(node['m'])
                n_child = list(node['n'])
                kv_child = list(node['kv'])
                plc_child = {k: list(v) for k, v in node['plc'].items()}

                m_child[g] += ms
                n_child[g] += n
                kv_child[g] = kvpr(n_child[g], GPU_MEM_SIZE - m_child[g])

                # Recompute top/second max quickly
                (topi, topv), (seci, secv) = top_two(kv_child)

                plc_child[g].append(mdl)
                child = {
                    'm': m_child,
                    'n': n_child,
                    'kv': kv_child,
                    'plc': plc_child,
                    'max': topv,
                    'max2': secv,
                    'idx_top': topi
                }
                # Compute lower bound for the remainder
                remaining = items_sorted[t+1:]
                child['bound'] = compute_bound(child, remaining)
                new_beam.append(child)

        if not new_beam:
            # If no expansions, fallback to a simple feasible memory-balanced placement
            # Greedy by size: place remaining items to the GPU with most free memory
            fallback = {g: list(beam[0]['plc'].get(g, [])) for g in range(gpu_num)}
            rem = [GPU_MEM_SIZE - sum(getattr(m, "model_size") for m in fallback[g]) for g in range(gpu_num)]
            numer = [sum((getattr(m, "req_rate") / getattr(m, "slo")) for m in fallback[g]) for g in range(gpu_num)]
            # Place current item + remaining
            pending = [items_sorted[t]] + items_sorted[t+1:]
            for md, ms2, n2 in pending:
                # choose GPU minimizing projected global max KVPR
                best = None
                for g in range(gpu_num):
                    if ms2 <= rem[g]:
                        new_local = kvpr(numer[g] + n2, rem[g] - ms2)
                        # compute max among others
                        others = []
                        for gg in range(gpu_num):
                            if gg == g:
                                others.append(new_local)
                            else:
                                others.append(kvpr(numer[gg], rem[gg]))
                        new_max = max(others)
                        key = (new_max, new_local, -(rem[g] - ms2), g)
                        if best is None or key < best[0]:
                            best = (key, g)
                if best is None:
                    # Truly infeasible
                    raise ValueError("Unable to construct any feasible placement")
                gsel = best[1]
                fallback[gsel].append(md)
                rem[gsel] -= ms2
                numer[gsel] += n2
            # Ensure all GPUs are present
            for g in range(gpu_num):
                fallback.setdefault(g, [])
            return fallback

        # Prune to beam width using bound then current max KVPR
        new_beam.sort(key=lambda nd: (nd['bound'], nd['max'], sum(nd['kv'])/gpu_num if gpu_num > 0 else 0.0))
        beam = new_beam[:BEAM_WIDTH]

    # Select best full placement from beam
    def measured_max_kvpr(plc):
        msum = [sum(getattr(m, "model_size") for m in plc.get(g, [])) for g in range(gpu_num)]
        nsum = [sum((getattr(m, "req_rate") / getattr(m, "slo")) for m in plc.get(g, [])) for g in range(gpu_num)]
        vals = [kvpr(nsum[g], GPU_MEM_SIZE - msum[g]) for g in range(gpu_num)]
        return max(vals) if vals else 0.0

    best_node = None
    best_score = float('inf')
    for nd in beam:
        score = measured_max_kvpr(nd['plc'])
        if score < best_score:
            best_score = score
            best_node = nd

    placement = {g: list(best_node['plc'].get(g, [])) for g in range(gpu_num)}

    # Light polishing: donation moves from worst to best GPUs (bounded)
    def polish(plc, move_budget=24):
        per_g = {g: list(plc.get(g, [])) for g in range(gpu_num)}
        mem = [sum(getattr(m, "model_size") for m in per_g[g]) for g in range(gpu_num)]
        num = [sum((getattr(m, "req_rate") / getattr(m, "slo")) for m in per_g[g]) for g in range(gpu_num)]

        def kv_g(g):
            return kvpr(num[g], GPU_MEM_SIZE - mem[g])

        for _ in range(move_budget):
            kvs = [kv_g(g) for g in range(gpu_num)]
            worst = max(range(gpu_num), key=lambda x: kvs[x])
            best = min(range(gpu_num), key=lambda x: kvs[x])
            cur_max = kvs[worst]
            improved = False
            candidate_models = sorted(per_g[worst], key=lambda m: (getattr(m, "model_size"), (getattr(m, "req_rate")/getattr(m, "slo"))), reverse=True)
            # Try moves to top-2 best GPUs by KVPR
            tgt_order = sorted(range(gpu_num), key=lambda x: kvs[x])[:min(2, gpu_num)]
            best_new_max = cur_max
            best_move = None
            for mdl in candidate_models[:min(12, len(candidate_models))]:
                ms = float(getattr(mdl, "model_size"))
                dn = float(getattr(mdl, "req_rate")) / float(getattr(mdl, "slo"))
                for tgt in tgt_order:
                    if tgt == worst:
                        continue
                    if mem[tgt] + ms > GPU_MEM_SIZE + 1e-12:
                        continue
                    # simulate move
                    src_mem = mem[worst] - ms
                    src_num = num[worst] - dn
                    tgt_mem = mem[tgt] + ms
                    tgt_num = num[tgt] + dn
                    src_k = kvpr(src_num, GPU_MEM_SIZE - src_mem)
                    tgt_k = kvpr(tgt_num, GPU_MEM_SIZE - tgt_mem)
                    new_max = max(src_k, tgt_k)
                    for g in range(gpu_num):
                        if g != worst and g != tgt:
                            kg = kv_g(g)
                            if kg > new_max:
                                new_max = kg
                    if new_max + 1e-12 < best_new_max:
                        best_new_max = new_max
                        best_move = (mdl, worst, tgt, ms, dn)
                        improved = True
            if not improved:
                break
            mdl, src, tgt, ms, dn = best_move
            per_g[src].remove(mdl)
            per_g[tgt].append(mdl)
            mem[src] -= ms; num[src] -= dn
            mem[tgt] += ms; num[tgt] += dn

        return {g: per_g.get(g, []) for g in range(gpu_num)}

    placement = polish(placement, move_budget=24)

    # Ensure all GPUs are present
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