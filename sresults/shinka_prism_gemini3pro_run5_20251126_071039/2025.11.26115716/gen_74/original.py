# EVOLVE-BLOCK-START
"""Model placement algorithm for minimizing maximum KV cache pressure using Randomized Binary Search Packing and Variance-Aware ILS"""

import copy
import random
import math

GPU_MEM_SIZE = 80.0  # GB

def compute_model_placement(gpu_num, models):
    """
    Minimizes max KVPR using:
    1. Randomized Binary Search with Best-Fit Decreasing Packing.
       - Tries multiple sorts + random shuffles to find tightest feasible K.
    2. Iterated Local Search (ILS).
       - Steepest Descent on (Max_K, Sum_Squared_K).
       - Burst Kicks to escape local optima.
    """
    # 1. Validation and Setup
    total_size = sum(m.model_size for m in models)
    if total_size > gpu_num * GPU_MEM_SIZE:
        raise ValueError("Total model size exceeds total GPU memory capacity.")

    # Prepare packing items
    # item: {'w': req/slo, 's': size, 'm': model}
    items = [{'w': m.req_rate / m.slo, 's': m.model_size, 'm': m} for m in models]

    # 2. Binary Search for Initial Feasible Solution
    total_w = sum(x['w'] for x in items)
    slack = gpu_num * GPU_MEM_SIZE - total_size

    low = 0.0
    # Heuristic upper bound
    if slack < 1e-6:
        high = 10000.0
    else:
        avg_pressure = total_w / slack
        high = max(10.0, avg_pressure * 8.0)

    best_placement = None
    feasible_high = False

    # Find valid upper bound
    for _ in range(20):
        # Use randomized trials to increase chance of finding feasible packing
        feasible, placement = _check_feasibility_randomized(gpu_num, items, high, trials=5)
        if feasible:
            best_placement = placement
            feasible_high = True
            break
        low = high
        high *= 2.0

    if not feasible_high:
        raise ValueError("Unable to place models. Constraints too tight.")

    # Binary Search Refinement
    for _ in range(32):
        mid = (low + high) / 2.0
        # Trials increase robustness of the check
        feasible, placement = _check_feasibility_randomized(gpu_num, items, mid, trials=3)
        if feasible:
            best_placement = placement
            high = mid
        else:
            low = mid

    placement_map = {i: best_placement[i] for i in range(gpu_num)}

    # 3. Refinement: Variance-Aware Steepest Descent ILS
    return _ils_descent_variance(gpu_num, placement_map)

def _check_feasibility_randomized(gpu_num, items, K, trials=0):
    """
    Checks if items can be packed into gpu_num bins such that for all bins:
    Sum(w) / (Cap - Sum(s)) <= K  <=>  Sum(w + K*s) <= K*Cap
    Uses BFD with deterministic sorts and random shuffles.
    """
    virtual_cap = K * GPU_MEM_SIZE
    
    # Create pack items with virtual size v
    pack_items = []
    for x in items:
        v = x['w'] + K * x['s']
        pack_items.append({'v': v, 's': x['s'], 'm': x['m']})
        
    # Strategy 1: Virtual Size Descending (Most effective for this inequality)
    pack_items.sort(key=lambda x: x['v'], reverse=True)
    if res := _pack_bfd(gpu_num, pack_items, virtual_cap):
        return True, res
        
    # Strategy 2: Physical Size Descending
    pack_items.sort(key=lambda x: x['s'], reverse=True)
    if res := _pack_bfd(gpu_num, pack_items, virtual_cap):
        return True, res

    # Strategy 3: Randomized Shuffles (to escape bad ordering)
    if trials > 0:
        indices = list(range(len(pack_items)))
        for _ in range(trials):
            random.shuffle(indices)
            shuffled_items = [pack_items[i] for i in indices]
            if res := _pack_bfd(gpu_num, shuffled_items, virtual_cap):
                return True, res

    return False, None

def _pack_bfd(gpu_num, items, virtual_cap):
    """
    Best Fit Decreasing for Virtual Capacity.
    Places item in the bin with smallest sufficient residual virtual capacity.
    """
    bins_v = [0.0] * gpu_num
    bins_p = [0.0] * gpu_num
    placement = [[] for _ in range(gpu_num)]
    
    for item in items:
        best_bin = -1
        min_rem_v = float('inf')
        
        for i in range(gpu_num):
            if bins_p[i] + item['s'] <= GPU_MEM_SIZE and bins_v[i] + item['v'] <= virtual_cap + 1e-7:
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

def _ils_descent_variance(gpu_num, placement):
    """
    Iterated Local Search using Steepest Descent.
    Objective: Minimize (Max_K, Sum_Sq_K).
    """
    # Initialize State
    gpu_s = [sum(m.model_size for m in placement[i]) for i in range(gpu_num)]
    gpu_w = [sum(m.req_rate / m.slo for m in placement[i]) for i in range(gpu_num)]
    
    def get_k(i):
        rem = GPU_MEM_SIZE - gpu_s[i]
        if rem <= 1e-7: return 1e9
        return gpu_w[i] / rem
        
    current_ks = [get_k(i) for i in range(gpu_num)]
    
    best_max_k = max(current_ks)
    best_sol = copy.deepcopy(placement)
    
    max_steps = 500
    patience = 30
    no_improve = 0
    
    step = 0
    while step < max_steps:
        step += 1
        
        # 1. Identify Bottleneck & Stats
        max_k = -1.0
        src = -1
        sum_sq = 0.0
        
        for i in range(gpu_num):
            k = current_ks[i]
            if k > max_k:
                max_k = k
                src = i
            sum_sq += k * k
            
        # Update Global Best
        if max_k < best_max_k - 1e-7:
            best_max_k = max_k
            best_sol = copy.deepcopy(placement)
            no_improve = 0
        else:
            no_improve += 1
            
        # 2. Kick if Stuck
        if no_improve > patience:
            # Burst Kick: 3-5 random moves
            kick_size = random.randint(3, 5)
            for _ in range(kick_size):
                for _ in range(10): # Retry for valid move
                    s = random.randint(0, gpu_num - 1)
                    if not placement[s]: continue
                    d = random.randint(0, gpu_num - 1)
                    if s == d: continue
                    
                    m_idx = random.randint(0, len(placement[s]) - 1)
                    m = placement[s][m_idx]
                    
                    if gpu_s[d] + m.model_size <= GPU_MEM_SIZE:
                        placement[d].append(m)
                        placement[s].pop(m_idx)
                        gpu_s[s] -= m.model_size; gpu_w[s] -= m.req_rate/m.slo
                        gpu_s[d] += m.model_size; gpu_w[d] += m.req_rate/m.slo
                        current_ks[s] = get_k(s)
                        current_ks[d] = get_k(d)
                        break
            no_improve = 0
            continue
            
        # 3. Steepest Descent
        # Find best move/swap from src that improves (Max_K, Sum_Sq) tuple
        best_move = None
        # Metric: (new_max_peak, new_sum_sq)
        best_metric = (max_k, sum_sq)
        
        src_models = placement[src]
        
        # A. Moves
        for i, m in enumerate(src_models):
            s, w = m.model_size, m.req_rate/m.slo
            for dst in range(gpu_num):
                if dst == src: continue
                if gpu_s[dst] + s > GPU_MEM_SIZE: continue
                
                # New values
                nk_src = (gpu_w[src] - w) / (GPU_MEM_SIZE - (gpu_s[src] - s) + 1e-9)
                nk_dst = (gpu_w[dst] + w) / (GPU_MEM_SIZE - (gpu_s[dst] + s) + 1e-9)
                
                new_peak = max(nk_src, nk_dst)
                
                # Optimization: Don't consider if new peak is strictly worse
                if new_peak > max_k + 1e-7: continue
                
                delta_sq = (nk_src**2 + nk_dst**2) - (current_ks[src]**2 + current_ks[dst]**2)
                new_sq = sum_sq + delta_sq
                
                metric = (new_peak, new_sq)
                
                # Lexicographical comparison: Minimize Peak, then SumSq
                if metric[0] < best_metric[0] - 1e-7:
                    best_metric = metric
                    best_move = ('move', i, dst, s, w)
                elif abs(metric[0] - best_metric[0]) < 1e-7:
                    if metric[1] < best_metric[1] - 1e-5:
                        best_metric = metric
                        best_move = ('move', i, dst, s, w)

        # B. Swaps
        # Only check swaps if we haven't found a "killer" move that reduces max_k massively
        # or if we are fine-tuning variance.
        for i1, m1 in enumerate(src_models):
            s1, w1 = m1.model_size, m1.req_rate/m1.slo
            for dst in range(gpu_num):
                if dst == src: continue
                # Skip swap with highly loaded GPU unless it's the 2nd bottleneck
                if current_ks[dst] > max_k * 0.98: continue
                
                for i2, m2 in enumerate(placement[dst]):
                    s2, w2 = m2.model_size, m2.req_rate/m2.slo
                    
                    ns_src = gpu_s[src] - s1 + s2
                    ns_dst = gpu_s[dst] - s2 + s1
                    if ns_src > GPU_MEM_SIZE or ns_dst > GPU_MEM_SIZE: continue
                    
                    nk_src = (gpu_w[src] - w1 + w2) / (GPU_MEM_SIZE - ns_src + 1e-9)
                    nk_dst = (gpu_w[dst] - w2 + w1) / (GPU_MEM_SIZE - ns_dst + 1e-9)
                    
                    new_peak = max(nk_src, nk_dst)
                    if new_peak > max_k + 1e-7: continue
                    
                    delta_sq = (nk_src**2 + nk_dst**2) - (current_ks[src]**2 + current_ks[dst]**2)
                    new_sq = sum_sq + delta_sq
                    metric = (new_peak, new_sq)
                    
                    if metric[0] < best_metric[0] - 1e-7:
                        best_metric = metric
                        best_move = ('swap', i1, dst, i2, s1, w1, s2, w2)
                    elif abs(metric[0] - best_metric[0]) < 1e-7:
                        if metric[1] < best_metric[1] - 1e-5:
                            best_metric = metric
                            best_move = ('swap', i1, dst, i2, s1, w1, s2, w2)

        # Execute Best Move
        if best_move:
            if best_move[0] == 'move':
                _, i, dst, s, w = best_move
                m = placement[src].pop(i)
                placement[dst].append(m)
                gpu_s[src] -= s; gpu_w[src] -= w
                gpu_s[dst] += s; gpu_w[dst] += w
            else:
                _, i1, dst, i2, s1, w1, s2, w2 = best_move
                m1 = placement[src][i1]
                m2 = placement[dst][i2]
                placement[src][i1] = m2
                placement[dst][i2] = m1
                gpu_s[src] = gpu_s[src] - s1 + s2
                gpu_w[src] = gpu_w[src] - w1 + w2
                gpu_s[dst] = gpu_s[dst] - s2 + s1
                gpu_w[dst] = gpu_w[dst] - w2 + w1
            
            # Update cache
            current_ks[src] = get_k(src)
            current_ks[dst] = get_k(dst)
            no_improve = 0
        else:
            no_improve += 1
            
    return best_sol
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
