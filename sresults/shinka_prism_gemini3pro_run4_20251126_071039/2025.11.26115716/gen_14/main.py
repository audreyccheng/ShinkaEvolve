# EVOLVE-BLOCK-START
"""Model placement algorithm for minimizing maximum KV cache pressure across GPUs"""

GPU_MEM_SIZE = 80  # GB

def compute_model_placement(gpu_num, models):
    """
    Compute a model placement that minimizes the maximum KVPR across all GPUs.
    Combines Greedy Heuristics, Binary Search (Bin Packing), and Local Search.
    """

    # Helper to calculate max KVPR of a placement
    def get_max_kvpr(placement):
        max_p = 0.0
        for assigned in placement.values():
            w = sum(m.req_rate / m.slo for m in assigned)
            s = sum(m.model_size for m in assigned)
            rem = GPU_MEM_SIZE - s
            if rem <= 1e-9:
                # If usage is exactly max memory (or close), and weight > 0, pressure is inf.
                # If weight is 0, it's 0. But models have weight.
                if w > 0: return float('inf')
                else: continue
            max_p = max(max_p, w / rem)
        return max_p

    best_placement = None
    best_score = float('inf')

    # ---------------------------------------------------------
    # 1. Greedy Heuristics Ensemble
    # ---------------------------------------------------------
    # Strategies:
    #   'min_result': Place on GPU that minimizes the resulting KVPR (Greedy Min-Max)
    #   'min_current': Place on GPU that has the lowest current KVPR (Load Balancing/Valley Filling)

    heuristics = [
        (lambda m: m.req_rate / m.slo, 'min_result'),
        (lambda m: m.req_rate / m.slo, 'min_current'),
        (lambda m: m.model_size, 'min_result'),
        (lambda m: (m.req_rate / m.slo) / (GPU_MEM_SIZE - m.model_size + 1e-6), 'min_result'),
    ]

    for key_fn, strategy in heuristics:
        sorted_models = sorted(models, key=key_fn, reverse=True)

        placement = {i: [] for i in range(gpu_num)}
        gpu_w = [0.0] * gpu_num
        gpu_s = [0.0] * gpu_num
        possible = True

        for model in sorted_models:
            w = model.req_rate / model.slo
            s = model.model_size

            best_idx = None
            best_val = float('inf')

            for i in range(gpu_num):
                if gpu_s[i] + s > GPU_MEM_SIZE: continue

                rem = GPU_MEM_SIZE - gpu_s[i]

                if strategy == 'min_result':
                    new_rem = rem - s
                    if new_rem > 1e-9:
                        val = (gpu_w[i] + w) / new_rem
                    else:
                        val = float('inf')
                else: # min_current
                    if rem > 1e-9:
                        val = gpu_w[i] / rem
                    else:
                        val = float('inf')

                if val < best_val:
                    best_val = val
                    best_idx = i
                elif val == best_val and best_idx is None:
                    best_idx = i

            if best_idx is None:
                possible = False
                break

            placement[best_idx].append(model)
            gpu_w[best_idx] += w
            gpu_s[best_idx] += s

        if possible:
            score = get_max_kvpr(placement)
            if score < best_score:
                best_score = score
                best_placement = placement

    # ---------------------------------------------------------
    # 2. Binary Search on Target KVPR (Transformation to Bin Packing)
    # ---------------------------------------------------------
    # We want to check if there exists a placement such that KVPR <= K for all GPUs.
    # Constraint: sum(w) / (C - sum(s)) <= K  <==>  sum(w) + K*sum(s) <= K*C
    # Let v_i(K) = w_i + K*s_i. We pack items of size v_i into bins of capacity K*C.

    # Lower bound: Perfect fluid balance
    total_w = sum(m.req_rate / m.slo for m in models)
    total_s = sum(m.model_size for m in models)
    rem_global = gpu_num * GPU_MEM_SIZE - total_s

    if rem_global > 1e-6:
        low = total_w / rem_global
        high = best_score if best_score != float('inf') else 1000.0

        # Binary Search loop
        # Only if the range is significant
        if high > low + 1e-4:
            for _ in range(20):
                mid = (low + high) / 2

                # Sort items by virtual size v_i(mid) descending
                # This heuristics works well for Bin Packing (Best Fit Decreasing)
                bs_models = sorted(models, key=lambda m: (m.req_rate/m.slo) + mid * m.model_size, reverse=True)

                temp_placement = {i: [] for i in range(gpu_num)}
                gpu_w = [0.0] * gpu_num
                gpu_s = [0.0] * gpu_num
                possible_k = True

                for model in bs_models:
                    w = model.req_rate / model.slo
                    s = model.model_size

                    best_idx = None
                    min_slack = float('inf')

                    # Best Fit Decreasing logic
                    for i in range(gpu_num):
                        if gpu_s[i] + s > GPU_MEM_SIZE: continue

                        # Check KVPR constraint: (W + w) <= mid * (C - S - s)
                        lhs = gpu_w[i] + w
                        rhs = mid * (GPU_MEM_SIZE - gpu_s[i] - s)

                        if lhs <= rhs + 1e-5:
                            slack = rhs - lhs
                            if slack < min_slack:
                                min_slack = slack
                                best_idx = i

                    if best_idx is None:
                        possible_k = False
                        break

                    temp_placement[best_idx].append(model)
                    gpu_w[best_idx] += w
                    gpu_s[best_idx] += s

                if possible_k:
                    # Found a valid placement with max KVPR <= mid
                    current_actual_max = get_max_kvpr(temp_placement)
                    if current_actual_max < best_score:
                        best_score = current_actual_max
                        best_placement = temp_placement
                    high = mid
                else:
                    low = mid

    if best_placement is None:
        raise ValueError("Unable to place models on GPUs with available memory.")

    # ---------------------------------------------------------
    # 3. Local Search Refinement
    # ---------------------------------------------------------
    # Iteratively move models from the bottleneck GPU to others to reduce peak
    for _ in range(50):
        # Find bottleneck GPU
        max_p = -1.0
        src_gpu = -1
        gpu_stats = []

        for i in range(gpu_num):
            assigned = best_placement[i]
            if not assigned:
                gpu_stats.append({'w': 0, 's': 0, 'p': 0})
                continue
            w = sum(m.req_rate / m.slo for m in assigned)
            s = sum(m.model_size for m in assigned)
            rem = GPU_MEM_SIZE - s
            p = w / rem if rem > 1e-9 else float('inf')
            gpu_stats.append({'w': w, 's': s, 'p': p})

            if p > max_p:
                max_p = p
                src_gpu = i

        if src_gpu == -1 or max_p < 1e-9: break

        improved = False
        src_models = best_placement[src_gpu]

        # Try to move one model
        for m_idx, model in enumerate(src_models):
            w = model.req_rate / model.slo
            s = model.model_size

            # Predict source pressure if moved
            # (src_w - w) / (src_rem + s)
            src_rem = GPU_MEM_SIZE - gpu_stats[src_gpu]['s']
            new_src_rem = src_rem + s
            new_src_p = (gpu_stats[src_gpu]['w'] - w) / new_src_rem

            best_dst = None
            # We want to move to a destination such that max(new_src_p, new_dst_p) < max_p
            # And ideally minimize new_dst_p

            for dst in range(gpu_num):
                if dst == src_gpu: continue
                if gpu_stats[dst]['s'] + s > GPU_MEM_SIZE: continue

                dst_rem = GPU_MEM_SIZE - gpu_stats[dst]['s']
                new_dst_rem = dst_rem - s
                if new_dst_rem <= 1e-9: continue # Avoid full GPU if w>0

                new_dst_p = (gpu_stats[dst]['w'] + w) / new_dst_rem

                if max(new_src_p, new_dst_p) < max_p - 1e-5:
                    best_dst = dst
                    break # First fit is enough for local search

            if best_dst is not None:
                # Apply move
                moved_model = src_models.pop(m_idx)
                best_placement[best_dst].append(moved_model)
                improved = True
                break

        if not improved: break

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