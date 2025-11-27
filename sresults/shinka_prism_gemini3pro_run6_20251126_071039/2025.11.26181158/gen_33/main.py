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

    # --- Phase 3: Steepest Descent Local Search ---
    # Convert to mutable state with precomputed values
    placement = []
    # Cache model info: (model_obj, l, s)
    for g in range(gpu_num):
        gpu_models = []
        for m in best_placement[g]:
            gpu_models.append((m, m.req_rate / m.slo, m.model_size))
        placement.append(gpu_models)

    # Track sums
    gpu_l = [sum(x[1] for x in p) for p in placement]
    gpu_s = [sum(x[2] for x in p) for p in placement]

    def get_kvpr(l, s):
        if s >= GPU_MEM_SIZE - 1e-6: return 1e15
        return l / (GPU_MEM_SIZE - s)

    for _ in range(100): # Limit iterations
        # Find bottleneck
        current_kvprs = [get_kvpr(gpu_l[g], gpu_s[g]) for g in range(gpu_num)]
        max_kvpr = max(current_kvprs)
        if max_kvpr < 1e-9: break

        # Identify bottleneck GPU
        b_idx = -1
        for g in range(gpu_num):
            if current_kvprs[g] >= max_kvpr - 1e-9:
                b_idx = g
                break

        best_move = None
        best_new_max = max_kvpr

        # Try Moves: b_idx -> t_idx
        for i, (m, m_l, m_s) in enumerate(placement[b_idx]):
            for t_idx in range(gpu_num):
                if t_idx == b_idx: continue
                # Capacity check
                if gpu_s[t_idx] + m_s >= GPU_MEM_SIZE - 1e-6: continue

                # Pruning: if target load is high, unlikely to help
                if current_kvprs[t_idx] >= max_kvpr: continue

                # Predict
                new_b_l = gpu_l[b_idx] - m_l
                new_b_s = gpu_s[b_idx] - m_s
                new_t_l = gpu_l[t_idx] + m_l
                new_t_s = gpu_s[t_idx] + m_s

                k_b = get_kvpr(new_b_l, new_b_s)
                k_t = get_kvpr(new_t_l, new_t_s)

                if max(k_b, k_t) < best_new_max - 1e-9:
                    best_new_max = max(k_b, k_t)
                    best_move = ('move', b_idx, i, t_idx)

        # Try Swaps: b_idx[i] <-> t_idx[j]
        for i, (m1, m1_l, m1_s) in enumerate(placement[b_idx]):
            for t_idx in range(gpu_num):
                if t_idx == b_idx: continue
                if current_kvprs[t_idx] >= max_kvpr: continue

                for j, (m2, m2_l, m2_s) in enumerate(placement[t_idx]):
                    # Capacity check
                    if gpu_s[b_idx] - m1_s + m2_s >= GPU_MEM_SIZE - 1e-6: continue
                    if gpu_s[t_idx] - m2_s + m1_s >= GPU_MEM_SIZE - 1e-6: continue

                    new_b_l = gpu_l[b_idx] - m1_l + m2_l
                    new_b_s = gpu_s[b_idx] - m1_s + m2_s
                    new_t_l = gpu_l[t_idx] - m2_l + m1_l
                    new_t_s = gpu_s[t_idx] - m2_s + m1_s

                    k_b = get_kvpr(new_b_l, new_b_s)
                    k_t = get_kvpr(new_t_l, new_t_s)

                    if max(k_b, k_t) < best_new_max - 1e-9:
                        best_new_max = max(k_b, k_t)
                        best_move = ('swap', b_idx, i, t_idx, j)

        if best_move:
            type_ = best_move[0]
            if type_ == 'move':
                _, s, si, t = best_move
                item = placement[s].pop(si)
                placement[t].append(item)

                # Update sums
                _, m_l, m_s = item
                gpu_l[s] -= m_l; gpu_s[s] -= m_s
                gpu_l[t] += m_l; gpu_s[t] += m_s

            elif type_ == 'swap':
                _, s, si, t, ti = best_move
                item1 = placement[s][si]
                item2 = placement[t][ti]
                placement[s][si] = item2
                placement[t][ti] = item1

                _, m1_l, m1_s = item1
                _, m2_l, m2_s = item2

                gpu_l[s] += (m2_l - m1_l); gpu_s[s] += (m2_s - m1_s)
                gpu_l[t] += (m1_l - m2_l); gpu_s[t] += (m1_s - m2_s)
        else:
            break

    # Reconstruct result format
    final_result = {}
    for i in range(gpu_num):
        final_result[i] = [x[0] for x in placement[i]]

    return final_result

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