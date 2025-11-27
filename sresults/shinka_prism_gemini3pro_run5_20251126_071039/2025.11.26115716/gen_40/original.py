# EVOLVE-BLOCK-START
"""Model placement algorithm for minimizing maximum KV cache pressure across GPUs using Binary Search and FFD Packing"""

GPU_MEM_SIZE = 80  # GB

def compute_model_placement(gpu_num, models):
    """
    Compute a model placement that minimizes the maximum KVPR across all GPUs.
    Uses binary search on the answer combined with First Fit Decreasing bin packing.

    Args:
        gpu_num: Number of GPUs
        models: List of models to place

    Returns:
        A placement of models to GPUs
    """

    # Check total physical size feasibility as a quick fail-safe
    if sum(m.model_size for m in models) > gpu_num * GPU_MEM_SIZE:
        raise ValueError("Total model size exceeds total GPU memory capacity.")

    # Prepare model data: (w, size, model_obj) where w = req_rate / slo
    model_data = []
    for m in models:
        model_data.append(((m.req_rate / m.slo), m.model_size, m))

    # Binary search for the minimum feasible max_kvpr (X).
    # Range of X is [0, inf).
    # We first find a feasible upper bound 'high'.

    high = 1.0
    # Heuristic initialization for high:
    # Estimate based on average load or use a safe doubling strategy.
    # Start with a heuristic guess to save iterations.
    total_w = sum(x[0] for x in model_data)
    total_slack = gpu_num * GPU_MEM_SIZE - sum(x[1] for x in model_data)
    if total_slack > 1e-6:
        high = max(high, (total_w / total_slack) * 2.0)
    else:
        high = 100.0

    # Find a valid upper bound
    best_placement = None
    feasible_high = False

    for _ in range(20):
        is_feasible, placement = _check_feasibility(gpu_num, model_data, high)
        if is_feasible:
            feasible_high = True
            best_placement = placement
            break
        high *= 2.0

    if not feasible_high:
        # If we cannot find a placement even with very high KVPR,
        # it likely implies physical constraints are tight or fragmentation is high.
        raise ValueError(f"Unable to place models. Physical constraints or fragmentation too high.")

    # Binary Search
    low = 0.0
    # 30 iterations is sufficient for high precision
    for _ in range(30):
        mid = (low + high) / 2.0
        is_feasible, placement = _check_feasibility(gpu_num, model_data, mid)
        if is_feasible:
            best_placement = placement
            high = mid
        else:
            low = mid

    # Format the output as expected {gpu_id: [models]}
    result = {}
    for i in range(gpu_num):
        result[i] = best_placement[i] if i < len(best_placement) else []

    return result

def _check_feasibility(gpu_num, model_data, target_kvpr):
    """
    Check if models can be packed into gpu_num bins given a target KVPR.
    Constraint: sum(w) / (Capacity - sum(size)) <= target_kvpr
    Rearranged: sum(w + target_kvpr * size) <= target_kvpr * Capacity
    """
    # Virtual Capacity
    cap = target_kvpr * GPU_MEM_SIZE

    # Calculate virtual sizes and sort for FFD
    # Item tuple: (virtual_size, physical_size, model_obj)
    # virtual_size = w + target_kvpr * size
    items = []
    for w, size, m in model_data:
        v_size = w + target_kvpr * size
        items.append((v_size, size, m))

    # Sort descending by virtual size (FFD heuristic)
    items.sort(key=lambda x: x[0], reverse=True)

    # Bins state
    bins_v_load = [0.0] * gpu_num
    bins_p_load = [0.0] * gpu_num
    placements = [[] for _ in range(gpu_num)]

    for v_size, p_size, m in items:
        placed = False
        for i in range(gpu_num):
            # Check virtual capacity constraint
            # Using a small epsilon for float comparison stability
            if bins_v_load[i] + v_size <= cap + 1e-7:
                # Check physical capacity constraint (hard limit)
                if bins_p_load[i] + p_size <= GPU_MEM_SIZE:
                    bins_v_load[i] += v_size
                    bins_p_load[i] += p_size
                    placements[i].append(m)
                    placed = True
                    break
        if not placed:
            return False, None

    return True, placements
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
