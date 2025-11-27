# EVOLVE-BLOCK-START
"""Model placement algorithm for minimizing maximum KV cache pressure across GPUs"""

GPU_MEM_SIZE = 80  # GB

def compute_model_placement(gpu_num, models):
    """
    Compute a model placement that minimizes the maximum KVPR across all GPUs.

    Uses an ensemble of greedy heuristics. Each heuristic sorts models differently
    and then greedily assigns them to the GPU that minimizes the resulting KVPR.
    """

    # Sorting strategies to try:
    # 1. Weight (req/slo) primary, Size secondary
    # 2. Size primary, Weight secondary
    # 3. Isolated KVPR (Weight / Available_If_Alone)
    # 4. Density (Weight / Size)
    sorting_keys = [
        lambda m: (m.req_rate / m.slo, m.model_size),
        lambda m: (m.model_size, m.req_rate / m.slo),
        lambda m: (m.req_rate / m.slo) / (GPU_MEM_SIZE - m.model_size + 1e-6),
        lambda m: (m.req_rate / m.slo) / m.model_size
    ]

    best_placement = None
    best_max_kvpr = float('inf')

    # Try each sorting strategy
    for key_fn in sorting_keys:
        sorted_models = sorted(models, key=key_fn, reverse=True)

        # Per-pass state
        placement = {gpu_id: [] for gpu_id in range(gpu_num)}
        shared_kv = [GPU_MEM_SIZE for _ in range(gpu_num)]
        weighted_req_rate = [0.0 for _ in range(gpu_num)]
        possible = True

        # Greedy assignment
        for model in sorted_models:
            best_idx = None
            best_new_kvpr = float('inf')

            w = model.req_rate / model.slo
            s = model.model_size

            for gpu_id in range(gpu_num):
                if s <= shared_kv[gpu_id]:
                    # Calculate KVPR if we place model here
                    new_rem = shared_kv[gpu_id] - s
                    new_rate = weighted_req_rate[gpu_id] + w

                    if new_rem > 1e-6:
                        new_kvpr = new_rate / new_rem
                    else:
                        new_kvpr = float('inf')

                    if new_kvpr < best_new_kvpr:
                        best_new_kvpr = new_kvpr
                        best_idx = gpu_id

            if best_idx is None:
                possible = False
                break

            placement[best_idx].append(model)
            weighted_req_rate[best_idx] += w
            shared_kv[best_idx] -= s

        if possible:
            # Calculate global max KVPR for this placement
            current_max = 0.0
            for gpu_id in range(gpu_num):
                rem = shared_kv[gpu_id]
                rate = weighted_req_rate[gpu_id]
                if rem < GPU_MEM_SIZE: # If GPU is used
                    if rem > 1e-6:
                        current_max = max(current_max, rate / rem)
                    else:
                        current_max = float('inf')

            if current_max < best_max_kvpr:
                best_max_kvpr = current_max
                best_placement = placement

    if best_placement is None:
        raise ValueError("Unable to place models on GPUs with available memory.")

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