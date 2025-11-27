# EVOLVE-BLOCK-START
"""Model placement algorithm for minimizing maximum KV cache pressure using Robust Binary Search and Simulated Annealing with Variance Penalty"""

import copy
import random
import math

GPU_MEM_SIZE = 80.0  # GB

def compute_model_placement(gpu_num, models):
    """
    Minimizes max KVPR using:
    1. Robust Binary Search with multiple packing heuristics (FFD/BFD on Virtual/Physical/Density).
    2. Simulated Annealing refinement with an energy function that penalizes variance.

    Args:
        gpu_num: Number of GPUs
        models: List of models to place

    Returns:
        A placement of models to GPUs {gpu_id: [models]}
    """
    # 1. Validation and Pre-processing
    total_size = sum(m.model_size for m in models)
    if total_size > gpu_num * GPU_MEM_SIZE:
        raise ValueError("Total model size exceeds total GPU memory capacity.")

    # Prepare items for packing: (w, s, m)
    # w = req_rate / slo
    items = [{'w': m.req_rate / m.slo, 's': m.model_size, 'm': m} for m in models]

    # 2. Binary Search for Initial Feasible Solution
    # Calculate range
    total_w = sum(x['w'] for x in items)
    slack = gpu_num * GPU_MEM_SIZE - total_size

    low = 0.0
    # Heuristic upper bound
    if slack < 1e-6:
        high = 10000.0 # Fallback for tight packing
    else:
        avg_pressure = total_w / slack
        high = max(10.0, avg_pressure * 8.0)

    best_placement = None
    feasible_high = False

    # Find valid upper bound
    for _ in range(20):
        feasible, placement = _check_feasibility_robust(gpu_num, items, high)
        if feasible:
            best_placement = placement
            feasible_high = True
            break
        low = high
        high *= 2.0

    if not feasible_high:
        raise ValueError("Unable to place models. Constraints likely too tight.")

    # Binary Search
    for _ in range(30):
        mid = (low + high) / 2.0
        feasible, placement = _check_feasibility_robust(gpu_num, items, mid)
        if feasible:
            best_placement = placement
            high = mid
        else:
            low = mid

    # Convert list placement to dictionary map
    placement_map = {i: best_placement[i] for i in range(gpu_num)}

    # 3. Large Neighborhood Search Refinement (Ruin & Recreate)
    final_placement = _large_neighborhood_search(gpu_num, placement_map)

    return final_placement

def _check_feasibility_robust(gpu_num, items, K):
    """
    Checks feasibility using multiple sorting strategies and packing algorithms (FFD/BFD).
    Constraint: sum(w + K*s) <= K*Capacity
    """
    virtual_cap = K * GPU_MEM_SIZE

    # Create augmented items for sorting
    pack_items = []
    for x in items:
        # Virtual Size: v = w + K*s
        v = x['w'] + K * x['s']
        # Density: Load per unit size
        density = x['w'] / (x['s'] + 1e-7)
        pack_items.append({
            'v': v,
            's': x['s'],
            'w': x['w'],
            'd': density,
            'm': x['m']
        })

    # Heuristics: List of (sort_key_lambda, reverse_bool)
    # 1. Virtual Size Descending (Standard)
    # 2. Physical Size Descending (Good for large models)
    # 3. Density Descending (Good for mixing high/low pressure models)
    # 4. Load Descending
    heuristics = [
        (lambda x: x['v'], True),
        (lambda x: x['s'], True),
        (lambda x: x['d'], True),
        (lambda x: x['w'], True),
    ]

    for key_func, rev in heuristics:
        sorted_items = sorted(pack_items, key=key_func, reverse=rev)

        # Try First Fit Decreasing (FFD)
        if res := _pack_ffd(gpu_num, sorted_items, virtual_cap):
            return True, res

        # Try Best Fit Decreasing (BFD)
        if res := _pack_bfd(gpu_num, sorted_items, virtual_cap):
            return True, res

    return False, None

def _pack_ffd(gpu_num, items, virtual_cap):
    """First Fit Decreasing Packing"""
    bins_v = [0.0] * gpu_num
    bins_p = [0.0] * gpu_num
    placement = [[] for _ in range(gpu_num)]

    for item in items:
        placed = False
        for i in range(gpu_num):
            if bins_p[i] + item['s'] <= GPU_MEM_SIZE and bins_v[i] + item['v'] <= virtual_cap + 1e-7:
                bins_p[i] += item['s']
                bins_v[i] += item['v']
                placement[i].append(item['m'])
                placed = True
                break
        if not placed: return None
    return placement

def _pack_bfd(gpu_num, items, virtual_cap):
    """Best Fit Decreasing Packing"""
    bins_v = [0.0] * gpu_num
    bins_p = [0.0] * gpu_num
    placement = [[] for _ in range(gpu_num)]

    for item in items:
        best_bin = -1
        min_rem_v = float('inf')

        for i in range(gpu_num):
            if bins_p[i] + item['s'] <= GPU_MEM_SIZE and bins_v[i] + item['v'] <= virtual_cap + 1e-7:
                # Minimize remaining virtual capacity (tightest fit)
                rem = virtual_cap - (bins_v[i] + item['v'])
                if rem < min_rem_v:
                    min_rem_v = rem
                    best_bin = i

        if best_bin != -1:
            bins_p[best_bin] += item['s']
            bins_v[best_bin] += item['v']
            placement[best_bin].append(item['m'])
        else:
            return None
    return placement

def _simulated_annealing_refinement(gpu_num, placement):
    """
    Refines placement using Simulated Annealing.
    Energy Function: E = Max(K) + Variance_Penalty.
    The variance penalty acts as a tie-breaker and landscape smoother.
    """
    # Initialize State Cache
    gpu_s = [sum(m.model_size for m in placement[i]) for i in range(gpu_num)]
    gpu_w = [sum(m.req_rate / m.slo for m in placement[i]) for i in range(gpu_num)]

    def get_k(s, w):
        rem = GPU_MEM_SIZE - s
        if rem <= 1e-7: return 1e9 # Penalty for full/overflow
        return w / rem

    current_ks = [get_k(gpu_s[i], gpu_w[i]) for i in range(gpu_num)]

    cur_max = max(current_ks)
    cur_sum_sq = sum(k*k for k in current_ks)

    best_max = cur_max
    # Use shallow copy for speed (models are objects, lists are structure)
    best_placement = {i: list(placement[i]) for i in range(gpu_num)}

    # SA Parameters
    T = max(1.0, cur_max * 0.1)
    alpha = 0.98
    steps = 1000

    for step in range(steps):
        # 1. Source Selection: Bias towards bottleneck
        # Identify bottleneck indices
        sorted_indices = sorted(range(gpu_num), key=lambda i: current_ks[i], reverse=True)

        # 50% chance bottleneck, 30% second bottleneck, 20% random
        r = random.random()
        if r < 0.5: src = sorted_indices[0]
        elif r < 0.8 and gpu_num > 1: src = sorted_indices[1]
        else: src = random.randint(0, gpu_num - 1)

        if not placement[src]: continue

        # 2. Move Generation: Move (70%) or Swap (30%)
        move_type = 'swap' if random.random() < 0.3 else 'move'
        accepted = False

        if move_type == 'move':
            m_idx = random.randint(0, len(placement[src])-1)
            m = placement[src][m_idx]
            dst = random.randint(0, gpu_num - 1)
            if src == dst: continue

            # Check Physical Feasibility
            if gpu_s[dst] + m.model_size <= GPU_MEM_SIZE:
                # Calculate new states
                new_s_src = gpu_s[src] - m.model_size
                new_w_src = gpu_w[src] - (m.req_rate/m.slo)
                new_k_src = get_k(new_s_src, new_w_src)

                new_s_dst = gpu_s[dst] + m.model_size
                new_w_dst = gpu_w[dst] + (m.req_rate/m.slo)
                new_k_dst = get_k(new_s_dst, new_w_dst)

                # Delta evaluation
                old_k_src = current_ks[src]
                old_k_dst = current_ks[dst]

                # Update temp
                current_ks[src] = new_k_src
                current_ks[dst] = new_k_dst
                new_max = max(current_ks)

                delta_max = new_max - cur_max

                # Sum of squares change
                new_sum_sq = cur_sum_sq - old_k_src**2 - old_k_dst**2 + new_k_src**2 + new_k_dst**2
                delta_sq = new_sum_sq - cur_sum_sq

                # Acceptance Logic
                if delta_max < -1e-7:
                    accepted = True
                elif delta_max < 1e-7:
                    # Tie-breaking with variance
                    if delta_sq < 0:
                        accepted = True
                    else:
                        # Allow slightly higher variance if temp is high
                        prob = math.exp(-delta_sq / (T * cur_max * 10))
                        if random.random() < prob: accepted = True
                else:
                    # Allow worsening max
                    prob = math.exp(-delta_max / T)
                    if random.random() < prob: accepted = True

                if accepted:
                    placement[dst].append(m)
                    placement[src].pop(m_idx)
                    gpu_s[src] = new_s_src; gpu_w[src] = new_w_src
                    gpu_s[dst] = new_s_dst; gpu_w[dst] = new_w_dst
                    cur_max = new_max
                    cur_sum_sq = new_sum_sq
                else:
                    # Revert temp
                    current_ks[src] = old_k_src
                    current_ks[dst] = old_k_dst

        elif move_type == 'swap':
            m1_idx = random.randint(0, len(placement[src])-1)
            m1 = placement[src][m1_idx]

            dst = random.randint(0, gpu_num - 1)
            if src == dst or not placement[dst]: continue

            m2_idx = random.randint(0, len(placement[dst])-1)
            m2 = placement[dst][m2_idx]

            new_s_src = gpu_s[src] - m1.model_size + m2.model_size
            new_s_dst = gpu_s[dst] - m2.model_size + m1.model_size

            if new_s_src <= GPU_MEM_SIZE and new_s_dst <= GPU_MEM_SIZE:
                w_change = (m2.req_rate/m2.slo) - (m1.req_rate/m1.slo)
                new_w_src = gpu_w[src] + w_change
                new_w_dst = gpu_w[dst] - w_change

                new_k_src = get_k(new_s_src, new_w_src)
                new_k_dst = get_k(new_s_dst, new_w_dst)

                old_k_src = current_ks[src]
                old_k_dst = current_ks[dst]

                current_ks[src] = new_k_src
                current_ks[dst] = new_k_dst
                new_max = max(current_ks)

                delta_max = new_max - cur_max
                new_sum_sq = cur_sum_sq - old_k_src**2 - old_k_dst**2 + new_k_src**2 + new_k_dst**2
                delta_sq = new_sum_sq - cur_sum_sq

                if delta_max < -1e-7:
                    accepted = True
                elif delta_max < 1e-7:
                    if delta_sq < 0:
                        accepted = True
                    else:
                        prob = math.exp(-delta_sq / (T * cur_max * 10))
                        if random.random() < prob: accepted = True
                else:
                    prob = math.exp(-delta_max / T)
                    if random.random() < prob: accepted = True

                if accepted:
                    placement[src][m1_idx] = m2
                    placement[dst][m2_idx] = m1
                    gpu_s[src] = new_s_src; gpu_w[src] = new_w_src
                    gpu_s[dst] = new_s_dst; gpu_w[dst] = new_w_dst
                    cur_max = new_max
                    cur_sum_sq = new_sum_sq
                else:
                    current_ks[src] = old_k_src
                    current_ks[dst] = old_k_dst

        # Update Global Best
        if cur_max < best_max - 1e-7:
            best_max = cur_max
            best_placement = {i: list(placement[i]) for i in range(gpu_num)}

        T *= alpha
        if T < 1e-4: break

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