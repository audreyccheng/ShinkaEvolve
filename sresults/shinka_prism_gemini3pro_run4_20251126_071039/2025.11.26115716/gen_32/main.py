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
            lambda x: x['w'] / (x['s'] + 1e-9),
            lambda x: x['w'] / (GPU_MEM_SIZE - x['s'] + 1e-9)
        ]

        for key in strategies:
            res = try_pack(sorted(items, key=key, reverse=True))
            if res: return res

        # 2. Randomized Strategies (Noisy Heuristic)
        rng = random.Random(42 + int(k_target * 100))
        base_key = lambda x: x['w'] + k_target * x['s']

        for _ in range(50):
            # Sort with multiplicative noise on the key to explore near-optimal orderings
            noisy_items = sorted(items, key=lambda x: base_key(x) * rng.uniform(0.8, 1.2), reverse=True)
            res = try_pack(noisy_items)
            if res: return res

        return None

    # Helper: Local Search Optimization
    def local_optimize(placement):
        # Initialize state
        state = []
        for i in range(gpu_num):
            models_on_gpu = placement[i]
            w = sum(m.req_rate / m.slo for m in models_on_gpu)
            s = sum(m.model_size for m in models_on_gpu)
            rem = GPU_MEM_SIZE - s
            if rem <= 1e-9:
                val = float('inf') if w > 1e-9 else 0.0
            else:
                val = w / rem
            state.append({'w': w, 's': s, 'val': val, 'models': list(models_on_gpu)})

        def calc_val(w, s):
            rem = GPU_MEM_SIZE - s
            if rem <= 1e-9: return float('inf') if w > 1e-9 else 0.0
            return w / rem

        # Hill Climbing
        for _ in range(100):
            # Identify bottleneck
            max_val = -1.0
            src_idx = -1
            for i, st in enumerate(state):
                if st['val'] > max_val:
                    max_val = st['val']
                    src_idx = i

            if max_val <= 1e-9: break

            improved = False
            src = state[src_idx]

            # 1. Try Move
            for i, m in enumerate(src['models']):
                m_w = m.req_rate / m.slo
                m_s = m.model_size

                ns_s = src['s'] - m_s
                ns_w = src['w'] - m_w
                ns_val = calc_val(ns_w, ns_s)

                # Heuristic: only move if source actually improves significantly
                # or if we are just trying to get below max_val
                if ns_val >= max_val - 1e-9: continue

                best_dst_idx = -1
                best_dst_val = float('inf')

                for dst_idx in range(gpu_num):
                    if dst_idx == src_idx: continue
                    dst = state[dst_idx]

                    if dst['s'] + m_s > GPU_MEM_SIZE: continue

                    nd_val = calc_val(dst['w'] + m_w, dst['s'] + m_s)

                    if nd_val < max_val - 1e-9:
                        if nd_val < best_dst_val:
                            best_dst_val = nd_val
                            best_dst_idx = dst_idx

                if best_dst_idx != -1:
                    src['models'].pop(i)
                    src['w'], src['s'], src['val'] = ns_w, ns_s, ns_val

                    dst = state[best_dst_idx]
                    dst['models'].append(m)
                    dst['w'] += m_w
                    dst['s'] += m_s
                    dst['val'] = best_dst_val
                    improved = True
                    break

                if improved: break

            if improved: continue

            # 2. Try Swap
            for i, m1 in enumerate(src['models']):
                m1_w = m1.req_rate / m1.slo
                m1_s = m1.model_size

                for dst_idx in range(gpu_num):
                    if dst_idx == src_idx: continue
                    dst = state[dst_idx]
                    if dst['val'] > max_val * 0.95: continue

                    for j, m2 in enumerate(dst['models']):
                        m2_w = m2.req_rate / m2.slo
                        m2_s = m2.model_size

                        ns_s = src['s'] - m1_s + m2_s
                        nd_s = dst['s'] - m2_s + m1_s
                        if ns_s > GPU_MEM_SIZE or nd_s > GPU_MEM_SIZE: continue

                        ns_val = calc_val(src['w'] - m1_w + m2_w, ns_s)
                        nd_val = calc_val(dst['w'] - m2_w + m1_w, nd_s)

                        if max(ns_val, nd_val) < max_val - 1e-9:
                            # Swap
                            src['models'][i] = m2
                            src['w'] += (m2_w - m1_w)
                            src['s'] += (m2_s - m1_s)
                            src['val'] = ns_val

                            dst['models'][j] = m1
                            dst['w'] += (m1_w - m2_w)
                            dst['s'] += (m1_s - m2_s)
                            dst['val'] = nd_val
                            improved = True
                            break
                    if improved: break
                if improved: break

        return {i: state[i]['models'] for i in range(gpu_num)}

    def get_max_kvpr(placement):
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

    # Binary Search for the Minimum Maximum KVPR (K)

    # Initialization
    high = 1e9
    best_placement = check_placement(high)

    if best_placement is None:
        raise ValueError("Unable to place models on GPUs (insufficient total memory).")

    # Optimize initial
    best_placement = local_optimize(best_placement)
    high = get_max_kvpr(best_placement)
    low = 0.0

    # Binary Search Loop
    for _ in range(25):
        if high - low < 1e-4: break
        mid = (low + high) / 2
        result = check_placement(mid)
        if result is not None:
            # Try to improve the found placement to tighten the bound further
            refined_result = local_optimize(result)
            refined_max = get_max_kvpr(refined_result)

            if refined_max < get_max_kvpr(best_placement):
                best_placement = refined_result

            high = min(mid, refined_max)
        else:
            low = mid

    # Final polish
    best_placement = local_optimize(best_placement)
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