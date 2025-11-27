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

    def get_max_kvpr(placement):
        """Calculate the actual maximum KVPR of a given placement."""
        current_max = 0.0
        for p in placement.values():
            w_sum = sum(m.req_rate / m.slo for m in p)
            s_sum = sum(m.model_size for m in p)
            rem = GPU_MEM_SIZE - s_sum
            if rem <= 1e-9:
                if w_sum > 1e-9: return float('inf')
                val = 0.0
            else:
                val = w_sum / rem
            if val > current_max:
                current_max = val
        return current_max

    def check_placement(k_target, attempt_limit=10):
        """
        Check if models can be placed with KVPR <= k_target using multiple heuristics.
        """

        def solve_packing(ordered_items):
            placement = {i: [] for i in range(gpu_num)}
            # Track states: weight, size
            gpu_state = [{'w': 0.0, 's': 0.0} for _ in range(gpu_num)]

            for item in ordered_items:
                best_idx = -1
                best_score = -1.0 # Best Fit score

                for i in range(gpu_num):
                    new_s = gpu_state[i]['s'] + item['s']
                    if new_s > GPU_MEM_SIZE: continue

                    new_w = gpu_state[i]['w'] + item['w']
                    rem = GPU_MEM_SIZE - new_s

                    # KVPR Check: w <= k * rem
                    if new_w > k_target * rem + 1e-7:
                        continue

                    # Best Fit Heuristic: Maximize linearized fill
                    # This packs items tightly w.r.t the constraint boundary
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

        # 1. Deterministic Strategies
        # Sort keys: (Linearized Cost, Size, Weight)
        strategies = [
            lambda x: x['w'] + k_target * x['s'],
            lambda x: x['s'],
            lambda x: x['w']
        ]

        for key_func in strategies:
            res = solve_packing(sorted(items, key=key_func, reverse=True))
            if res: return res

        # 2. Randomized Strategies
        rng = random.Random(42 + int(k_target)) # Seed dependent on K for variation
        indices = list(range(len(items)))
        for _ in range(attempt_limit):
            rng.shuffle(indices)
            res = solve_packing([items[i] for i in indices])
            if res: return res

        return None

    # Binary Search
    high = 1e9
    best_placement = check_placement(high, attempt_limit=1) # Quick check

    if best_placement is None:
        raise ValueError("Unable to place models on GPUs (insufficient total memory).")

    # Tighten initial bound
    high = get_max_kvpr(best_placement)
    low = 0.0

    # Binary Search Loop
    for _ in range(25):
        mid = (low + high) / 2
        # Use more attempts as we narrow down
        res = check_placement(mid, attempt_limit=20)
        if res is not None:
            best_placement = res
            high = min(mid, get_max_kvpr(res))
        else:
            low = mid

    # --- Local Search Optimization (Hill Climbing) ---
    # Try to reduce max KVPR by moving models from the bottleneck GPU

    # Initialize state
    gpu_states = []
    for i in range(gpu_num):
        w = sum(m.req_rate / m.slo for m in best_placement[i])
        s = sum(m.model_size for m in best_placement[i])
        gpu_states.append({'w': w, 's': s, 'models': list(best_placement[i])})

    def calc_kvpr(w, s):
        rem = GPU_MEM_SIZE - s
        if rem <= 1e-9:
             return float('inf') if w > 1e-9 else 0.0
        return w / rem

    for _ in range(150): # Iterations
        # Find bottleneck
        max_kvpr = -1.0
        src_idx = -1

        for i in range(gpu_num):
            k = calc_kvpr(gpu_states[i]['w'], gpu_states[i]['s'])
            if k > max_kvpr:
                max_kvpr = k
                src_idx = i

        if max_kvpr <= 0: break

        improved = False
        src = gpu_states[src_idx]

        # 1. Try Move
        for i, m in enumerate(src['models']):
            m_w = m.req_rate / m.slo
            m_s = m.model_size

            # Simulated removal
            ns_w = src['w'] - m_w
            ns_s = src['s'] - m_s
            ns_kvpr = calc_kvpr(ns_w, ns_s)

            # Only consider moving if source improves
            if ns_kvpr >= max_kvpr - 1e-9: continue

            for dst_idx in range(gpu_num):
                if dst_idx == src_idx: continue
                dst = gpu_states[dst_idx]

                # Check mem
                if dst['s'] + m_s > GPU_MEM_SIZE: continue

                nd_kvpr = calc_kvpr(dst['w'] + m_w, dst['s'] + m_s)

                if nd_kvpr < max_kvpr - 1e-9:
                    # Move is good
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

        # 2. Try Swap
        # Swap model m1 from src with m2 from dst
        for i, m1 in enumerate(src['models']):
            m1_w = m1.req_rate / m1.slo
            m1_s = m1.model_size

            for dst_idx in range(gpu_num):
                if dst_idx == src_idx: continue
                dst = gpu_states[dst_idx]

                # Heuristic: Don't swap with another bottleneck
                if calc_kvpr(dst['w'], dst['s']) > max_kvpr * 0.9: continue

                for j, m2 in enumerate(dst['models']):
                    m2_w = m2.req_rate / m2.slo
                    m2_s = m2.model_size

                    # New Src
                    ns_s = src['s'] - m1_s + m2_s
                    if ns_s > GPU_MEM_SIZE: continue
                    ns_w = src['w'] - m1_w + m2_w
                    ns_kvpr = calc_kvpr(ns_w, ns_s)

                    if ns_kvpr >= max_kvpr - 1e-9: continue

                    # New Dst
                    nd_s = dst['s'] - m2_s + m1_s
                    if nd_s > GPU_MEM_SIZE: continue
                    nd_w = dst['w'] - m2_w + m1_w
                    nd_kvpr = calc_kvpr(nd_w, nd_s)

                    if nd_kvpr < max_kvpr - 1e-9:
                        # Swap is good
                        src['models'][i] = m2
                        src['w'] = ns_w
                        src['s'] = ns_s

                        dst['models'][j] = m1
                        dst['w'] = nd_w
                        dst['s'] = nd_s
                        improved = True
                        break
                if improved: break
            if improved: break

        if not improved:
            break

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
