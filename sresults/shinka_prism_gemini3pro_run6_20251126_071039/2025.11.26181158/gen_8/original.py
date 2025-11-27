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

    # Precompute model data: l = req_rate/slo, s = model_size
    model_data = []
    for m in models:
        model_data.append({
            'model': m,
            'l': m.req_rate / m.slo,
            's': m.model_size
        })

    # Binary search for the optimal max KVPR (K)
    # Check if a KVPR 'K' is feasible by transforming to Bin Packing:
    # Constraint: sum(l) / (M - sum(s)) <= K
    #          => sum(l) + K * sum(s) <= K * M
    # Item weight: w = l + K * s
    # Bin capacity: C = K * M

    def can_pack(target_k):
        # Calculate weights for current K
        items = []
        for d in model_data:
            items.append((d['l'] + target_k * d['s'], d['model']))

        # Sort by weight descending (First Fit Decreasing)
        items.sort(key=lambda x: x[0], reverse=True)

        capacity = target_k * GPU_MEM_SIZE
        bins_load = [0.0] * gpu_num
        bins_models = [[] for _ in range(gpu_num)]

        for w, model in items:
            placed = False
            for i in range(gpu_num):
                if bins_load[i] + w <= capacity:
                    bins_load[i] += w
                    bins_models[i].append(model)
                    placed = True
                    break
            if not placed:
                return None
        return bins_models

    # Search range for K
    low = 0.0
    high = 1e12  # Large upper bound to cover tight memory cases
    best_placement = None

    # Check if feasible at all (at high K, memory constraints dominate)
    best_placement = can_pack(high)
    if best_placement is None:
        raise ValueError("Unable to fit models into GPU memory.")

    # Binary Search
    for _ in range(50):
        mid = (low + high) / 2
        res = can_pack(mid)
        if res is not None:
            best_placement = res
            high = mid
        else:
            low = mid

    # Format the result
    return {i: best_placement[i] for i in range(gpu_num)}

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
