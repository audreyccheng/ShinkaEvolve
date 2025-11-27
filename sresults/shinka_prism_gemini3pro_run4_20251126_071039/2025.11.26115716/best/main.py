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

    # Pre-process models to extract relevant metrics: weight (req/slo) and size
    items = []
    for m in models:
        items.append({
            'model': m,
            'w': m.req_rate / m.slo,
            's': m.model_size
        })

    def check_placement(k_target):
        import random
        """
        Determines if it is possible to place all models such that for every GPU:
        KVPR <= k_target.
        """

        def try_pack(item_order):
            placement = {i: [] for i in range(gpu_num)}
            gpu_state = [{'w': 0.0, 's': 0.0} for _ in range(gpu_num)]

            for item in item_order:
                best_idx = -1
                best_fill = -1.0

                for i in range(gpu_num):
                    new_s = gpu_state[i]['s'] + item['s']
                    if new_s > GPU_MEM_SIZE: continue

                    new_w = gpu_state[i]['w'] + item['w']
                    rem_mem = GPU_MEM_SIZE - new_s

                    # KVPR Check
                    if rem_mem <= 1e-9:
                        if k_target > 1e12: pass
                        elif new_w > 1e-9: continue
                    elif new_w > k_target * rem_mem + 1e-9:
                        continue

                    # Best Fit: Maximize w + k*s
                    current_fill = new_w + k_target * new_s
                    if current_fill > best_fill:
                        best_fill = current_fill
                        best_idx = i

                if best_idx != -1:
                    placement[best_idx].append(item['model'])
                    gpu_state[best_idx]['s'] += item['s']
                    gpu_state[best_idx]['w'] += item['w']
                else:
                    return None
            return placement

        # 1. Deterministic Strategies
        # Sort keys (descending)
        strategies = [
            lambda x: x['w'] + k_target * x['s'],
            lambda x: x['s'],
            lambda x: x['w'],
            lambda x: x['w'] / (x['s'] + 1e-9)
        ]

        for key in strategies:
            res = try_pack(sorted(items, key=key, reverse=True))
            if res: return res

        # 2. Randomized Strategies (Shuffle + Best Fit)
        rng = random.Random(42)
        indices = list(range(len(items)))
        for _ in range(50):
            rng.shuffle(indices)
            res = try_pack([items[i] for i in indices])
            if res: return res

        return None

    # Binary Search for the Minimum Maximum KVPR (K)

    # Initialization
    high = 1e9
    best_placement = check_placement(high)

    if best_placement is None:
        raise ValueError("Unable to place models on GPUs (insufficient total memory).")

    # Refine 'high' based on found solution
    current_max = 0.0
    for gpu_p in best_placement.values():
        w_sum = sum(m.req_rate / m.slo for m in gpu_p)
        s_sum = sum(m.model_size for m in gpu_p)
        rem = GPU_MEM_SIZE - s_sum
        if rem > 1e-9:
            current_max = max(current_max, w_sum / rem)
        elif w_sum > 0:
            current_max = high

    high = current_max
    low = 0.0

    # Binary Search Loop
    for _ in range(30):
        mid = (low + high) / 2
        result = check_placement(mid)
        if result is not None:
            best_placement = result

            # Refine high bound using the actual max KVPR of the valid solution
            actual_max = 0.0
            for gpu_p in best_placement.values():
                w_sum = sum(m.req_rate / m.slo for m in gpu_p)
                s_sum = sum(m.model_size for m in gpu_p)
                rem = GPU_MEM_SIZE - s_sum
                if rem > 1e-9:
                    val = w_sum / rem
                elif w_sum > 0:
                    val = float('inf')
                else:
                    val = 0.0
                if val > actual_max:
                    actual_max = val

            high = min(mid, actual_max)
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