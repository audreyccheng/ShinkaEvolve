# EVOLVE-BLOCK-START
import math

GPU_MEM_SIZE = 80  # GB

def compute_model_placement(gpu_num, models):
    """
    Compute a model placement that minimizes the maximum KVPR across all GPUs.

    Algorithm:
    1. Calculate a theoretical lower bound for KVPR assuming fluid models.
    2. Binary Search for the optimal Max KVPR 'K'.
    3. The feasibility check 'can_pack(K)' attempts to fit models into GPUs
       subject to: sum(req/slo) + K * sum(size) <= K * MEM_SIZE.
       It uses Best Fit Decreasing with multiple sorting keys (Linearized Weight, Size, Load)
       to robustly find a packing in different constraint regimes.
    """

    # Precompute model characteristics
    # l = req_rate / slo, s = model_size
    model_data = []
    total_l = 0.0
    total_s = 0.0

    for m in models:
        l = m.req_rate / m.slo
        s = m.model_size
        model_data.append({
            'model': m,
            'l': l,
            's': s
        })
        total_l += l
        total_s += s

    # 1. Theoretical Lower Bound
    # If models were fluid, we could perfectly balance:
    # K_min = Total_L / (Total_Capacity - Total_S)
    remaining_mem_global = (gpu_num * GPU_MEM_SIZE) - total_s
    if remaining_mem_global <= 0:
         # This implies total model size > total GPU memory, strictly impossible.
         # We'll let the packing logic handle the failure or raise here.
         lower_bound_k = 0.0
    else:
         lower_bound_k = total_l / remaining_mem_global

    # 2. Feasibility Check with Multi-Strategy Best Fit
    def can_pack(target_k):
        capacity = target_k * GPU_MEM_SIZE

        # Define sorting strategies
        # Strategy A: Linearized Weight (L + K*S).
        # Adapts to K: High K -> sort by Size; Low K -> sort by Load.
        strat_weight = sorted(
            model_data,
            key=lambda x: x['l'] + target_k * x['s'],
            reverse=True
        )

        # Strategy B: Size Decreasing.
        # Good for tight memory constraints.
        strat_size = sorted(
            model_data,
            key=lambda x: x['s'],
            reverse=True
        )

        # Strategy C: Load Decreasing.
        # Good for loose memory but tight load constraints.
        strat_load = sorted(
            model_data,
            key=lambda x: x['l'],
            reverse=True
        )

        # Try each strategy until one works
        for items in [strat_weight, strat_size, strat_load]:
            bins_l = [0.0] * gpu_num
            bins_s = [0.0] * gpu_num
            bins_models = [[] for _ in range(gpu_num)]
            possible = True

            for item in items:
                # Best Fit: Choose valid bin with minimal remaining slack
                # Slack in linearized constraint: (Capacity) - (Bin_W + Item_W)
                best_idx = -1
                min_slack = float('inf')

                w_item = item['l'] + target_k * item['s']

                for i in range(gpu_num):
                    # Hard Memory Constraint
                    # Use slightly larger epsilon for safety against float precision
                    if bins_s[i] + item['s'] >= GPU_MEM_SIZE - 1e-6:
                        continue

                    # Linearized Capacity Constraint
                    w_bin = bins_l[i] + target_k * bins_s[i]
                    if w_bin + w_item <= capacity + 1e-9:
                        slack = capacity - (w_bin + w_item)
                        if slack < min_slack:
                            min_slack = slack
                            best_idx = i

                if best_idx != -1:
                    bins_l[best_idx] += item['l']
                    bins_s[best_idx] += item['s']
                    bins_models[best_idx].append(item['model'])
                else:
                    possible = False
                    break

            if possible:
                return bins_models

        return None

    # 3. Binary Search
    low = lower_bound_k
    high = max(lower_bound_k * 2, 1.0)
    best_placement = None

    # Exponential expansion to find valid upper bound
    # (Starting from a reasoned guess saves steps)
    found_high = False
    for _ in range(20):
        res = can_pack(high)
        if res is not None:
            best_placement = res
            found_high = True
            break
        low = high
        high *= 2.0

    if not found_high:
        # Last ditch attempt with massive K (effectively just memory packing)
        high = 1e12
        best_placement = can_pack(high)
        if best_placement is None:
            raise ValueError("Unable to fit models into GPU memory.")

    # Refine K with Binary Search
    for _ in range(40):
        mid = (low + high) / 2
        res = can_pack(mid)
        if res is not None:
            best_placement = res
            high = mid
        else:
            low = mid

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