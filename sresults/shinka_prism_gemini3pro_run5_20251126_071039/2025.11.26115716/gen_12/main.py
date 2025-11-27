# EVOLVE-BLOCK-START
"""Model placement algorithm for minimizing maximum KV cache pressure across GPUs"""

GPU_MEM_SIZE = 80  # GB

def compute_model_placement(gpu_num, models):
    """
    Compute a model placement that minimizes the maximum KVPR across all GPUs.
    Uses Binary Search on the maximum pressure K, transforming the problem
    into a Bin Packing problem with item sizes w_j + K * s_j.
    """

    # Precompute model data
    m_data = []
    for m in models:
        m_data.append({
            'w': m.req_rate / m.slo,
            's': m.model_size,
            'obj': m
        })

    # Helper for greedy placement (used for initialization)
    def run_greedy(sort_key_func):
        sorted_items = sorted(m_data, key=sort_key_func, reverse=True)
        placements = {i: [] for i in range(gpu_num)}
        gpu_load = [0.0] * gpu_num
        gpu_used = [0.0] * gpu_num

        for item in sorted_items:
            best_gpu = -1
            best_score = float('inf')

            # Try to find best fit minimizing KVPR increase
            for g in range(gpu_num):
                rem = GPU_MEM_SIZE - (gpu_used[g] + item['s'])
                if rem > 0:
                    score = (gpu_load[g] + item['w']) / rem
                    if score < best_score:
                        best_score = score
                        best_gpu = g

            # Fallback to any valid fit if precise fit fails
            if best_gpu == -1:
                for g in range(gpu_num):
                    if gpu_used[g] + item['s'] < GPU_MEM_SIZE:
                         best_gpu = g
                         break

            if best_gpu == -1:
                return None, float('inf')

            placements[best_gpu].append(item['obj'])
            gpu_load[best_gpu] += item['w']
            gpu_used[best_gpu] += item['s']

        # Calculate max pressure
        max_p = 0.0
        for g in range(gpu_num):
            rem = GPU_MEM_SIZE - gpu_used[g]
            if rem <= 0:
                if gpu_load[g] > 0: max_p = float('inf')
            else:
                p = gpu_load[g] / rem
                if p > max_p: max_p = p
        return placements, max_p

    # 1. Get initial upper bound from a smart greedy approach
    # Sort by density (load/size) which is a strong heuristic
    best_placement, min_max_kvpr = run_greedy(lambda x: x['w'] / x['s'] if x['s'] > 0 else 0)

    # If density sort fails (e.g. fragmentation), try size sort
    if best_placement is None:
        best_placement, min_max_kvpr = run_greedy(lambda x: x['s'])

    if best_placement is None:
         raise ValueError("Unable to place models on GPUs.")

    # 2. Binary Search for optimal max pressure K
    # Range [0, greedy_result]
    low = 0.0
    high = min_max_kvpr

    # Optimization: limit iterations for speed
    for _ in range(20):
        if high - low < 1e-4:
            break

        mid = (low + high) / 2.0
        target_k = mid

        # Check feasibility with Best Fit Decreasing
        # Effective size v_j = w_j + K * s_j. Bin capacity C = K * M.
        # Sort items by effective size
        check_items = sorted(m_data, key=lambda x: x['w'] + target_k * x['s'], reverse=True)

        gpu_load = [0.0] * gpu_num
        gpu_used = [0.0] * gpu_num
        current_placement = {i: [] for i in range(gpu_num)}
        feasible = True

        for item in check_items:
            best_bin = -1
            min_residual = float('inf')
            item_cost = item['w'] + target_k * item['s']

            for g in range(gpu_num):
                # Hard memory constraint check (strictly < GPU_MEM_SIZE for valid KVPR)
                if gpu_used[g] + item['s'] >= GPU_MEM_SIZE:
                    continue

                # Transformed capacity constraint check
                # (load + w) + K(used + s) <= K*M
                current_cost = gpu_load[g] + target_k * gpu_used[g]

                if current_cost + item_cost <= target_k * GPU_MEM_SIZE + 1e-7:
                    # Best Fit: minimize residual capacity
                    res = (target_k * GPU_MEM_SIZE) - (current_cost + item_cost)
                    if res < min_residual:
                        min_residual = res
                        best_bin = g

            if best_bin != -1:
                current_placement[best_bin].append(item['obj'])
                gpu_load[best_bin] += item['w']
                gpu_used[best_bin] += item['s']
            else:
                feasible = False
                break

        if feasible:
            best_placement = current_placement
            high = mid
        else:
            low = mid

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