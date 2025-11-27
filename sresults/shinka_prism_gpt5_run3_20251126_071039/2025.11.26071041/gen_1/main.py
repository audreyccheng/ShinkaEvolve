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
    # Helper to compute KVPR for a GPU safely
    def kvpr(R, rem_mem):
        if rem_mem <= 0:
            return float('inf')
        return R / rem_mem

    # Early return for trivial cases
    placement = {gpu_id: [] for gpu_id in range(gpu_num)}
    if not models or gpu_num <= 0:
        return placement

    # Prepare per-GPU state
    rem_mem = [GPU_MEM_SIZE for _ in range(gpu_num)]  # remaining memory per GPU
    sum_r_over_s = [0.0 for _ in range(gpu_num)]      # sum of r_j / s_j per GPU

    # Sort models by "pressure weight": (r/s) / (GPU_MEM_SIZE - size)
    # Models that would cause higher KVPR on an empty GPU are placed first.
    def model_weight(m):
        denom = GPU_MEM_SIZE - m.model_size
        if denom <= 0:
            return float('inf')
        return (m.req_rate / m.slo) / denom

    sorted_models = sorted(models, key=model_weight, reverse=True)

    # Greedy min-max assignment using lookahead of resultant max KVPR
    for model in sorted_models:
        dR = model.req_rate / model.slo
        # Current KVPRs before placing this model
        current_kvprs = [kvpr(sum_r_over_s[i], rem_mem[i]) for i in range(gpu_num)]

        best_gpu = None
        best_resulting_max = float('inf')
        best_new_gpu_kvpr = float('inf')
        best_new_rem = -1

        for gid in range(gpu_num):
            # Must fit in memory
            if model.model_size <= rem_mem[gid]:
                new_R = sum_r_over_s[gid] + dR
                new_mem = rem_mem[gid] - model.model_size
                new_gpu_kvpr = kvpr(new_R, new_mem)

                # Compute resulting global max KVPR after placing on gid
                # Replace current_kvprs[gid] with new_gpu_kvpr and take max
                # This keeps the logic simple and robust for small gpu_num.
                resulting_max = new_gpu_kvpr
                for j in range(gpu_num):
                    if j == gid:
                        continue
                    if current_kvprs[j] > resulting_max:
                        resulting_max = current_kvprs[j]

                # Tie-breaking: minimize resulting max; then minimize this GPU's KVPR;
                # then prefer leaving more remaining memory.
                if (resulting_max < best_resulting_max or
                    (resulting_max == best_resulting_max and new_gpu_kvpr < best_new_gpu_kvpr) or
                    (resulting_max == best_resulting_max and new_gpu_kvpr == best_new_gpu_kvpr and new_mem > best_new_rem)):
                    best_resulting_max = resulting_max
                    best_new_gpu_kvpr = new_gpu_kvpr
                    best_new_rem = new_mem
                    best_gpu = gid

        if best_gpu is None:
            # No GPU can fit this model without exceeding memory
            raise ValueError(
                f"Unable to place model of size {model.model_size} GB on any GPU. "
                f"Remaining per-GPU memory: {rem_mem}"
            )

        # Commit placement
        placement[best_gpu].append(model)
        sum_r_over_s[best_gpu] += dR
        rem_mem[best_gpu] -= model.model_size

    # Local improvement: try moving models off the most pressured GPU if it reduces max KVPR
    def compute_all_kvprs():
        return [kvpr(sum_r_over_s[i], rem_mem[i]) for i in range(gpu_num)]

    # Limit the number of improvement iterations to keep it simple and fast
    max_iters = max(1, min(len(models), 5 * gpu_num))
    for _ in range(max_iters):
        kvprs = compute_all_kvprs()
        current_max = max(kvprs) if kvprs else 0.0
        if current_max == float('inf'):
            # Try to reduce infinity if possible
            pass
        max_gid = max(range(gpu_num), key=lambda i: kvprs[i] if i < len(kvprs) else -1.0)

        improved = False
        best_move = None  # (src, dst, model, resulting_max)

        # Try moving a single model from the max-pressure GPU to another GPU
        for mdl in list(placement[max_gid]):
            dR = mdl.req_rate / mdl.slo
            size = mdl.model_size

            # State after removing from source
            src_new_R = sum_r_over_s[max_gid] - dR
            src_new_mem = rem_mem[max_gid] + size
            src_new_kvpr = kvpr(src_new_R, src_new_mem)

            for dst in range(gpu_num):
                if dst == max_gid:
                    continue
                if size <= rem_mem[dst]:
                    dst_new_R = sum_r_over_s[dst] + dR
                    dst_new_mem = rem_mem[dst] - size
                    dst_new_kvpr = kvpr(dst_new_R, dst_new_mem)

                    # Compute resulting global max after move
                    # Start from current kvprs and update two GPUs
                    resulting_max = dst_new_kvpr
                    if src_new_kvpr > resulting_max:
                        resulting_max = src_new_kvpr
                    for j in range(gpu_num):
                        if j == max_gid or j == dst:
                            continue
                        if kvprs[j] > resulting_max:
                            resulting_max = kvprs[j]

                    if resulting_max + 1e-12 < current_max:  # strict improvement
                        # Tie-breakers: minimize resulting_max, then minimize dst kvpr, then maximize dst remaining mem
                        move_better = False
                        if best_move is None:
                            move_better = True
                        else:
                            _, _, _, best_res_max, best_dst_kvpr, best_dst_rem = best_move
                            if (resulting_max < best_res_max or
                                (resulting_max == best_res_max and dst_new_kvpr < best_dst_kvpr) or
                                (resulting_max == best_res_max and dst_new_kvpr == best_dst_kvpr and dst_new_mem > best_dst_rem)):
                                move_better = True
                        if move_better:
                            best_move = (max_gid, dst, mdl, resulting_max, dst_new_kvpr, dst_new_mem)

        if best_move is None:
            break  # no improving move found

        # Apply the best move
        src, dst, mdl, _, _, _ = best_move
        placement[src].remove(mdl)
        placement[dst].append(mdl)
        dR = mdl.req_rate / mdl.slo
        size = mdl.model_size
        sum_r_over_s[src] -= dR
        rem_mem[src] += size
        sum_r_over_s[dst] += dR
        rem_mem[dst] -= size
        improved = True

        if not improved:
            break

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

