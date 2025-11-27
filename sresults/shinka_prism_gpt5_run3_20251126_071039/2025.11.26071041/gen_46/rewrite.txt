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
    def safe_dR(m):
        slo = getattr(m, "slo", 0.0)
        if slo is None or slo == 0:
            return 0.0  # treat slo==0 as memory-only (no KV contribution)
        return float(m.req_rate) / float(slo)

    def kvpr(R, rem_mem):
        if rem_mem <= 0:
            return float('inf')
        return R / rem_mem

    # Partition models into memory-only (slo==0) and active (slo>0)
    mem_only = []
    active = []
    for m in models:
        dR = safe_dR(m)
        if dR == 0.0:
            mem_only.append(m)
        else:
            active.append(m)

    # Initialize placement state
    placement = {i: [] for i in range(gpu_num)}
    used_mem = [0.0] * gpu_num
    sum_R = [0.0] * gpu_num

    # Stage 1a: place memory-only by largest size first to GPUs with most remaining memory
    mem_only.sort(key=lambda m: float(m.model_size), reverse=True)
    for m in mem_only:
        size = float(m.model_size)
        # Choose GPU with largest remaining memory that fits
        best_gid = None
        best_rem = -1.0
        for gid in range(gpu_num):
            rem = S - used_mem[gid]
            if size <= rem and rem > best_rem:
                best_rem = rem
                best_gid = gid
        if best_gid is None:
            # Cannot fit even memory-only model
            raise ValueError(
                f"Unable to place memory-only model of size {m.model_size} GB. "
                f"Remaining per-GPU memory: {[S - um for um in used_mem]}"
            )
        placement[best_gid].append(m)
        used_mem[best_gid] += size
        # sum_R unchanged (dR=0)

    # Stage 1b: greedy global min-max assignment for active models
    # Order by "critical pressure" weight: dR / (S - size), tie by size desc
    def crit_weight(m):
        denom = S - float(m.model_size)
        if denom <= 0:
            return float('inf')
        return safe_dR(m) / denom

    active.sort(key=lambda m: (crit_weight(m), float(m.model_size)), reverse=True)

    def current_kvprs():
        return [kvpr(sum_R[g], S - used_mem[g]) for g in range(gpu_num)]

    for m in active:
        dR = safe_dR(m)
        size = float(m.model_size)
        kvprs_before = current_kvprs()

        best_gid = None
        best_resulting_max = float('inf')
        best_new_gpu_kvpr = float('inf')
        best_new_rem = -1.0

        for gid in range(gpu_num):
            if used_mem[gid] + size <= S:
                new_R = sum_R[gid] + dR
                new_rem = S - (used_mem[gid] + size)
                if new_rem <= 0:
                    continue
                new_gpu_kvpr = kvpr(new_R, new_rem)

                # Resulting global max KVPR if placed on gid
                resulting_max = new_gpu_kvpr
                for j in range(gpu_num):
                    if j == gid:
                        continue
                    if kvprs_before[j] > resulting_max:
                        resulting_max = kvprs_before[j]

                # Tie-break: minimize resulting_max, then this GPU's KVPR, then prefer more remaining memory
                if (resulting_max < best_resulting_max or
                    (resulting_max == best_resulting_max and new_gpu_kvpr < best_new_gpu_kvpr) or
                    (resulting_max == best_resulting_max and new_gpu_kvpr == best_new_gpu_kvpr and new_rem > best_new_rem)):
                    best_resulting_max = resulting_max
                    best_new_gpu_kvpr = new_gpu_kvpr
                    best_new_rem = new_rem
                    best_gid = gid

        if best_gid is None:
            raise ValueError(
                f"Unable to place active model of size {m.model_size} GB. "
                f"Remaining per-GPU memory: {[S - um for um in used_mem]}"
            )
        placement[best_gid].append(m)
        used_mem[best_gid] += size
        sum_R[best_gid] += dR

    # Stage 2: Local pressure-flow improvements + light simulated annealing
    import random
    rng = random.Random(len(models) * 10007 + gpu_num * 7919)

    def max_kvpr_and_arg():
        kvs = current_kvprs()
        if not kvs:
            return 0.0, 0
        max_val = kvs[0]
        arg = 0
        for i in range(1, gpu_num):
            if kvs[i] > max_val:
                max_val = kvs[i]
                arg = i
        return max_val, arg

    # Move application helpers
    def can_move(m, src, dst):
        s = float(m.model_size)
        return src != dst and (used_mem[dst] + s <= S)

    def apply_move(m, src, dst):
        # Update structures for moving model m
        s = float(m.model_size)
        r = safe_dR(m)
        placement[src].remove(m)
        placement[dst].append(m)
        used_mem[src] -= s
        used_mem[dst] += s
        sum_R[src] -= r
        sum_R[dst] += r

    def try_best_single_move_from(g_src, cur_max):
        # Try moving any model from g_src to reduce global max KVPR
        best = None  # (delta, src, dst, mdl)
        kvprs_now = current_kvprs()
        for mdl in list(placement[g_src]):
            s = float(mdl.model_size)
            r = safe_dR(mdl)
            # Source after removal
            src_new_R = sum_R[g_src] - r
            src_new_rem = S - (used_mem[g_src] - s)
            if src_new_rem <= 0:
                continue
            src_new_k = kvpr(src_new_R, src_new_rem)
            for dst in range(gpu_num):
                if not can_move(mdl, g_src, dst):
                    continue
                dst_new_R = sum_R[dst] + r
                dst_new_rem = S - (used_mem[dst] + s)
                if dst_new_rem <= 0:
                    continue
                dst_new_k = kvpr(dst_new_R, dst_new_rem)
                # Compute resulting global max
                resulting = dst_new_k if dst_new_k > src_new_k else src_new_k
                for j in range(gpu_num):
                    if j == g_src or j == dst:
                        continue
                    if kvprs_now[j] > resulting:
                        resulting = kvprs_now[j]
                delta = resulting - cur_max
                if delta < -1e-12:
                    if best is None or resulting < best[0]:
                        best = (resulting, g_src, dst, mdl)
        return best

    # Greedy hill-climb: only strictly improving moves
    move_budget = max(8, min(40, len(models) * 2))
    for _ in range(move_budget):
        cur_max, gmax = max_kvpr_and_arg()
        cand = try_best_single_move_from(gmax, cur_max)
        if cand is None:
            break
        _, src, dst, mdl = cand
        apply_move(mdl, src, dst)

    # Simulated annealing: allow occasional uphill to escape local minima
    def global_max_kvpr():
        return max(current_kvprs()) if gpu_num > 0 else 0.0

    best_snapshot = {i: list(placement[i]) for i in range(gpu_num)}
    best_used = list(used_mem)
    best_sumR = list(sum_R)
    best_val = global_max_kvpr()

    temp = best_val * 0.25 if best_val > 0 else 0.5
    cooling = 0.82
    outer = 10
    inner = max(10, min(40, len(models)))

    def rand_model_gpu():
        # Return a random (gpu, model) pair
        nonlocal rng
        # choose a non-empty gpu
        attempts = 0
        while attempts < 20:
            g = rng.randrange(gpu_num)
            if placement[g]:
                m = placement[g][rng.randrange(len(placement[g]))]
                return g, m
            attempts += 1
        # fallback: find first non-empty
        for g in range(gpu_num):
            if placement[g]:
                return g, placement[g][0]
        return 0, None

    for _ in range(outer):
        for __ in range(inner):
            cur_before = global_max_kvpr()

            # Randomly choose to attempt a move (70%) or a swap (30%)
            if rng.random() < 0.7:
                src, m = rand_model_gpu()
                if m is None:
                    continue
                dsts = list(range(gpu_num))
                rng.shuffle(dsts)
                accepted = False
                for dst in dsts:
                    if not can_move(m, src, dst):
                        continue
                    # Evaluate move
                    s = float(m.model_size); r = safe_dR(m)
                    # Temporarily apply
                    placement[src].remove(m); placement[dst].append(m)
                    used_mem[src] -= s; used_mem[dst] += s
                    sum_R[src] -= r; sum_R[dst] += r
                    cur_after = global_max_kvpr()
                    delta = cur_after - cur_before
                    # Metropolis acceptance
                    accept = (delta <= 0) or (rng.random() < pow(2.718281828, -(delta / max(temp, 1e-9))))
                    if accept:
                        accepted = True
                        if cur_after + 1e-12 < best_val:
                            best_val = cur_after
                            best_snapshot = {i: list(placement[i]) for i in range(gpu_num)}
                            best_used = list(used_mem)
                            best_sumR = list(sum_R)
                        break
                    else:
                        # Revert
                        placement[dst].remove(m); placement[src].append(m)
                        used_mem[src] += s; used_mem[dst] -= s
                        sum_R[src] += r; sum_R[dst] -= r
                # If not accepted anywhere, continue
                if not accepted:
                    # ensure state was not modified
                    continue
            else:
                # Try a swap between two random GPUs
                g1 = rng.randrange(gpu_num)
                g2 = rng.randrange(gpu_num)
                if g1 == g2 or not placement[g1] or not placement[g2]:
                    continue
                a = placement[g1][rng.randrange(len(placement[g1]))]
                b = placement[g2][rng.randrange(len(placement[g2]))]
                aS = float(a.model_size); bS = float(b.model_size)
                # Memory feasibility after swap
                if used_mem[g1] - aS + bS > S or used_mem[g2] - bS + aS > S:
                    continue
                aR = safe_dR(a); bR = safe_dR(b)
                # Apply swap
                placement[g1].remove(a); placement[g2].append(a)
                placement[g2].remove(b); placement[g1].append(b)
                used_mem[g1] = used_mem[g1] - aS + bS
                used_mem[g2] = used_mem[g2] - bS + aS
                sum_R[g1] = sum_R[g1] - aR + bR
                sum_R[g2] = sum_R[g2] - bR + aR

                cur_after = global_max_kvpr()
                delta = cur_after - cur_before
                accept = (delta <= 0) or (rng.random() < pow(2.718281828, -(delta / max(temp, 1e-9))))
                if accept:
                    if cur_after + 1e-12 < best_val:
                        best_val = cur_after
                        best_snapshot = {i: list(placement[i]) for i in range(gpu_num)}
                        best_used = list(used_mem)
                        best_sumR = list(sum_R)
                else:
                    # Revert swap
                    placement[g1].remove(b); placement[g2].append(b)
                    placement[g2].remove(a); placement[g1].append(a)
                    used_mem[g1] = used_mem[g1] + aS - bS
                    used_mem[g2] = used_mem[g2] + bS - aS
                    sum_R[g1] = sum_R[g1] + aR - bR
                    sum_R[g2] = sum_R[g2] + bR - aR

        temp *= cooling

    # Restore best snapshot if it is strictly better
    final_placement = {i: list(placement[i]) for i in range(gpu_num)}
    if best_val + 1e-12 < max(current_kvprs()):
        final_placement = best_snapshot

    # Final memory safety check
    for gid in range(gpu_num):
        mem = sum(float(m.model_size) for m in final_placement.get(gid, []))
        if mem - S > 1e-6:
            # fallback: return current placement (shouldn't happen)
            return {i: list(placement[i]) for i in range(gpu_num)}

    return final_placement

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