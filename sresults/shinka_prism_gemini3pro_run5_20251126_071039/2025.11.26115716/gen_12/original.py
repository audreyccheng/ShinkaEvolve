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

    # Greedy KVPR-minimizing placement
    # 1) Sort models by r_j / s_j descending (high load first).
    #    Secondary sort by model_size descending (large items first).
    sorted_models = sorted(models, key=lambda m: (m.req_rate / m.slo, m.model_size), reverse=True)

    # 2) Initialize per-GPU states
    placement = {gpu_id: [] for gpu_id in range(gpu_num)}
    shared_kv = [GPU_MEM_SIZE for _ in range(gpu_num)]  # remaining memory per GPU
    weighted_req_rate = [0.0 for _ in range(gpu_num)]   # sum of r_j / s_j per GPU

    # 3) Assign each model to the GPU that minimizes *resultant* KVPR
    for model in sorted_models:
        best_idx = None
        best_kvpr = float('inf')

        w_m = model.req_rate / model.slo
        z_m = model.model_size

        # Try to find a GPU where the model fits and leaves some memory (>0)
        for gpu_id in range(gpu_num):
            remaining_mem = shared_kv[gpu_id] - z_m
            if remaining_mem > 0:
                # Calculate projected KVPR: (current_load + new_load) / remaining_mem
                new_kvpr = (weighted_req_rate[gpu_id] + w_m) / remaining_mem
                if new_kvpr < best_kvpr:
                    best_kvpr = new_kvpr
                    best_idx = gpu_id

        # Fallback: if no GPU has >0 remaining memory, try exact fit (remaining == 0)
        if best_idx is None:
            for gpu_id in range(gpu_num):
                if shared_kv[gpu_id] >= z_m:
                    best_idx = gpu_id
                    break

        # Failure: if no GPU can fit, raise an error
        if best_idx is None:
            raise ValueError(
                f"Unable to place model of size {model.model_size} GB on any GPU. "
                f"Remaining per-GPU memory: {shared_kv}"
            )

        placement[best_idx].append(model)
        weighted_req_rate[best_idx] += w_m
        shared_kv[best_idx] -= z_m

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