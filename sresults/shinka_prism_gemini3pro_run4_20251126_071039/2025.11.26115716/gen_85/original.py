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
    import random

    # Pre-process models
    items = []
    for m in models:
        items.append({
            'model': m,
            'w': m.req_rate / m.slo,
            's': m.model_size
        })

    def get_placement_kvpr(placement):
        """Calculate max KVPR for a placement dict."""
        mx = 0.0
        for p in placement.values():
            w = sum(m.req_rate / m.slo for m in p)
            s = sum(m.model_size for m in p)
            rem = GPU_MEM_SIZE - s
            if rem <= 1e-9:
                val = float('inf') if w > 1e-9 else 0.0
            else:
                val = w / rem
            if val > mx: mx = val
        return mx

    def solve_packing(k_target, ordered_items):
        """Try to pack items into GPUs such that KVPR <= k_target."""
        placement = {i: [] for i in range(gpu_num)}
        gpu_state = [{'w': 0.0, 's': 0.0} for _ in range(gpu_num)]

        for item in ordered_items:
            best_idx = -1
            best_score = -1.0

            for i in range(gpu_num):
                # Capacity Check
                if gpu_state[i]['s'] + item['s'] > GPU_MEM_SIZE: continue

                new_w = gpu_state[i]['w'] + item['w']
                new_s = gpu_state[i]['s'] + item['s']
                rem = GPU_MEM_SIZE - new_s

                # KVPR Constraint Check
                if rem <= 1e-9:
                    if new_w > 1e-9: continue
                elif new_w > k_target * rem + 1e-7:
                    continue

                # Best Fit Heuristic: Maximize usage (linearized)
                # This packs bins tightly.
                score = new_w + k_target * new_s
                if score > best_score:
                    best_score = score
                    best_idx = i

            if best_idx != -1:
                placement[best_idx].append(item['model'])
                gpu_state[best_idx]['w'] += item['w']
                gpu_state[best_idx]['s'] += item['s']
            else:
                return None
        return placement

    def check_placement(k_target, attempts=10):
        # 1. Deterministic Strategies
        # Keys: Linearized Cost, Size, Weight
        strategies = [
            lambda x: x['w'] + k_target * x['s'],
            lambda x: x['s'],
            lambda x: x['w']
        ]

        for key_func in strategies:
            res = solve_packing(k_target, sorted(items, key=key_func, reverse=True))
            if res: return res

        # 2. Randomized Noisy Strategies
        if attempts > 0:
            rng = random.Random(42 + int(k_target))
            base_key = lambda x: x['w'] + k_target * x['s']
            for _ in range(attempts):
                # Add multiplicative noise
                noisy_items = sorted(items, key=lambda x: base_key(x) * rng.uniform(0.85, 1.15), reverse=True)
                res = solve_packing(k_target, noisy_items)
                if res: return res

        return None

    # Binary Search
    high = 1e9
    best_placement = check_placement(high, attempts=1)

    if best_placement is None:
        raise ValueError("Unable to place models on GPUs (insufficient total memory).")

    high = get_placement_kvpr(best_placement)
    low = 0.0

    for _ in range(25):
        mid = (low + high) / 2
        res = check_placement(mid, attempts=20)
        if res is not None:
            best_placement = res
            high = min(mid, get_placement_kvpr(res))
        else:
            low = mid

    # --- Local Search Optimization (Hill Climbing) ---
    # Convert to manageable state
    gpu_states = []
    for i in range(gpu_num):
        models_p = best_placement[i]
        w = sum(m.req_rate / m.slo for m in models_p)
        s = sum(m.model_size for m in models_p)
        gpu_states.append({'w': w, 's': s, 'models': list(models_p)})

    def calc_kvpr(w, s):
        rem = GPU_MEM_SIZE - s
        if rem <= 1e-9: return float('inf') if w > 1e-9 else 0.0
        return w / rem

    for _ in range(100):
        # Find bottleneck GPU
        max_k = -1.0
        src_idx = -1
        for i in range(gpu_num):
            k = calc_kvpr(gpu_states[i]['w'], gpu_states[i]['s'])
            if k > max_k:
                max_k = k
                src_idx = i

        if max_k <= 0: break

        improved = False
        src = gpu_states[src_idx]

        # Try moving a model from src to any dst
        for i, m in enumerate(src['models']):
            m_w = m.req_rate / m.slo
            m_s = m.model_size

            # Predict src after removal
            ns_w = src['w'] - m_w
            ns_s = src['s'] - m_s
            ns_k = calc_kvpr(ns_w, ns_s)

            # Optimization: Don't move if src doesn't improve enough to matter
            if ns_k >= max_k - 1e-9: continue

            for dst_idx in range(gpu_num):
                if dst_idx == src_idx: continue
                dst = gpu_states[dst_idx]

                if dst['s'] + m_s > GPU_MEM_SIZE: continue

                nd_k = calc_kvpr(dst['w'] + m_w, dst['s'] + m_s)

                if nd_k < max_k - 1e-9:
                    # Apply Move
                    src['models'].pop(i)
                    src['w'] = ns_w
                    src['s'] = ns_s

                    dst['models'].append(m)
                    dst['w'] += m_w
                    dst['s'] += m_s
                    improved = True
                    break
            if improved: break

        if improved: continue

        # Try Swapping
        # Swap model from src with model from dst
        for i, m1 in enumerate(src['models']):
            m1_w = m1.req_rate / m1.slo
            m1_s = m1.model_size

            for dst_idx in range(gpu_num):
                if dst_idx == src_idx: continue
                dst = gpu_states[dst_idx]

                # Heuristic: Don't swap with a GPU that is already near the bottleneck limit
                if calc_kvpr(dst['w'], dst['s']) > max_k * 0.95: continue

                for j, m2 in enumerate(dst['models']):
                    m2_w = m2.req_rate / m2.slo
                    m2_s = m2.model_size

                    # Check capacities
                    new_src_s = src['s'] - m1_s + m2_s
                    new_dst_s = dst['s'] - m2_s + m1_s

                    if new_src_s > GPU_MEM_SIZE or new_dst_s > GPU_MEM_SIZE: continue

                    new_src_w = src['w'] - m1_w + m2_w
                    new_dst_w = dst['w'] - m2_w + m1_w

                    nk_src = calc_kvpr(new_src_w, new_src_s)
                    nk_dst = calc_kvpr(new_dst_w, new_dst_s)

                    if max(nk_src, nk_dst) < max_k - 1e-9:
                        # Apply Swap
                        src['models'][i] = m2
                        src['w'] = new_src_w
                        src['s'] = new_src_s

                        dst['models'][j] = m1
                        dst['w'] = new_dst_w
                        dst['s'] = new_dst_s
                        improved = True
                        break
                if improved: break
            if improved: break

        if not improved: break

    return {i: gpu_states[i]['models'] for i in range(gpu_num)}

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
