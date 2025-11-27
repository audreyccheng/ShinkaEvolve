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
    # Iteratively move or swap models to reduce the peak KVPR

    # Calculate initial states
    gpu_states = []
    current_kvpr = []
    for i in range(gpu_num):
        assigned = best_placement[i]
        w = sum(m.req_rate / m.slo for m in assigned)
        s = sum(m.model_size for m in assigned)
        gpu_states.append({'w': w, 's': s})
        rem = GPU_MEM_SIZE - s
        p = w / rem if rem > 1e-9 else float('inf')
        current_kvpr.append(p)

    for _ in range(100):
        # Find current global max pressure
        max_p = max(current_kvpr)
        if max_p < 1e-9: break

        # Identify bottleneck GPUs
        src_gpus = [i for i, p in enumerate(current_kvpr) if abs(p - max_p) < 1e-9]
        # Pick one to optimize
        src_gpu = src_gpus[0]
        src_models = best_placement[src_gpu]

        improved = False

        # 1. Try MOVE (src -> dst)
        for m_idx, model in enumerate(src_models):
            w = model.req_rate / model.slo
            s = model.model_size

            # Hypothetical Src State
            new_src_s = gpu_states[src_gpu]['s'] - s
            new_src_w = gpu_states[src_gpu]['w'] - w
            new_src_rem = GPU_MEM_SIZE - new_src_s
            new_src_p = new_src_w / new_src_rem if new_src_rem > 1e-9 else float('inf')

            for dst_gpu in range(gpu_num):
                if dst_gpu == src_gpu: continue

                if gpu_states[dst_gpu]['s'] + s > GPU_MEM_SIZE: continue

                new_dst_s = gpu_states[dst_gpu]['s'] + s
                new_dst_w = gpu_states[dst_gpu]['w'] + w
                new_dst_rem = GPU_MEM_SIZE - new_dst_s
                new_dst_p = new_dst_w / new_dst_rem if new_dst_rem > 1e-9 else float('inf')

                # Check if this move reduces the pressure of the bottleneck
                # and doesn't create a new bottleneck worse than current max_p
                if max(new_src_p, new_dst_p) < max_p - 1e-6:
                    # Apply Move
                    moved_model = src_models.pop(m_idx)
                    best_placement[dst_gpu].append(moved_model)

                    gpu_states[src_gpu] = {'w': new_src_w, 's': new_src_s}
                    gpu_states[dst_gpu] = {'w': new_dst_w, 's': new_dst_s}
                    current_kvpr[src_gpu] = new_src_p
                    current_kvpr[dst_gpu] = new_dst_p
                    improved = True
                    break
            if improved: break

        if improved: continue

        # 2. Try SWAP (src <-> dst)
        # Iterate over all models in src and all models in other GPUs
        for m_idx, m_src in enumerate(src_models):
            w_src = m_src.req_rate / m_src.slo
            s_src = m_src.model_size

            for dst_gpu in range(gpu_num):
                if dst_gpu == src_gpu: continue

                dst_models = best_placement[dst_gpu]
                for d_idx, m_dst in enumerate(dst_models):
                    w_dst = m_dst.req_rate / m_dst.slo
                    s_dst = m_dst.model_size

                    # Check Capacity
                    new_src_s = gpu_states[src_gpu]['s'] - s_src + s_dst
                    if new_src_s > GPU_MEM_SIZE: continue

                    new_dst_s = gpu_states[dst_gpu]['s'] - s_dst + s_src
                    if new_dst_s > GPU_MEM_SIZE: continue

                    # Check Pressure
                    new_src_w = gpu_states[src_gpu]['w'] - w_src + w_dst
                    new_src_rem = GPU_MEM_SIZE - new_src_s
                    new_src_p = new_src_w / new_src_rem if new_src_rem > 1e-9 else float('inf')

                    new_dst_w = gpu_states[dst_gpu]['w'] - w_dst + w_src
                    new_dst_rem = GPU_MEM_SIZE - new_dst_s
                    new_dst_p = new_dst_w / new_dst_rem if new_dst_rem > 1e-9 else float('inf')

                    if max(new_src_p, new_dst_p) < max_p - 1e-6:
                        # Apply Swap
                        mod_src = src_models.pop(m_idx)
                        mod_dst = dst_models.pop(d_idx)

                        src_models.append(mod_dst) # Put dst model into src list
                        dst_models.append(mod_src) # Put src model into dst list

                        gpu_states[src_gpu] = {'w': new_src_w, 's': new_src_s}
                        gpu_states[dst_gpu] = {'w': new_dst_w, 's': new_dst_s}
                        current_kvpr[src_gpu] = new_src_p
                        current_kvpr[dst_gpu] = new_dst_p
                        improved = True
                        break
                if improved: break
            if improved: break

        if not improved:
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