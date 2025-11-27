# EVOLVE-BLOCK-START
"""Model placement algorithm for minimizing maximum KV cache pressure using Binary Search with BFD/FFD Packing and Simulated Annealing"""

import copy
import random
import math

GPU_MEM_SIZE = 80.0  # GB

def compute_model_placement(gpu_num, models):
    """
    Minimizes max KVPR using Binary Search with Multi-Strategy BFD/FFD packing 
    followed by Simulated Annealing refinement.
    
    Args:
        gpu_num: Number of GPUs
        models: List of models to place

    Returns:
        A placement of models to GPUs {gpu_id: [models]}
    """
    # 1. Validation and Setup
    total_size = sum(m.model_size for m in models)
    if total_size > gpu_num * GPU_MEM_SIZE:
        raise ValueError("Total model size exceeds total GPU memory capacity.")

    # Prepare items for packing: (w, s, m)
    # w = req_rate / slo
    items = [{'w': m.req_rate / m.slo, 's': m.model_size, 'm': m} for m in models]

    # 2. Binary Search for Initial Feasible Solution
    total_w = sum(x['w'] for x in items)
    slack = gpu_num * GPU_MEM_SIZE - total_size
    
    # Heuristic for upper bound
    if slack < 1e-6:
        high = 10000.0 # Arbitrary high for extremely tight cases
    else:
        avg_pressure = total_w / slack
        high = max(10.0, avg_pressure * 10.0)

    best_placement = None
    feasible_high = False
    
    # Find valid upper bound
    for _ in range(20):
        feasible, placement = _check_feasibility_multi(gpu_num, items, high)
        if feasible:
            best_placement = placement
            feasible_high = True
            break
        high *= 2.0
    
    if not feasible_high:
        raise ValueError("Unable to place models. Constraints too tight or fragmentation too high.")

    # Binary Search
    low = 0.0
    # 30 iterations for high precision
    for _ in range(30):
        mid = (low + high) / 2.0
        feasible, placement = _check_feasibility_multi(gpu_num, items, mid)
        if feasible:
            best_placement = placement
            high = mid
        else:
            low = mid
            
    # Convert list placement to dictionary map
    placement_map = {i: best_placement[i] for i in range(gpu_num)}
    
    # 3. Simulated Annealing Refinement
    final_placement = _simulated_annealing(gpu_num, placement_map)
    
    return final_placement

def _check_feasibility_multi(gpu_num, items, K):
    """
    Checks feasibility using FFD and BFD on multiple sort orders.
    Constraint: sum(w + K*s) <= K*Capacity
    """
    virtual_cap = K * GPU_MEM_SIZE
    
    # Create pack items dicts for easier access
    # virtual_size = w + K * s
    pack_items = []
    for x in items:
        v = x['w'] + K * x['s']
        pack_items.append({'v': v, 's': x['s'], 'm': x['m'], 'w': x['w']})
        
    # Heuristics:
    # 1. Virtual Size Descending
    # 2. Physical Size Descending
    # 3. Density (w/s) Descending
    
    sorters = [
        (lambda x: x['v'], True),               # Virtual Desc
        (lambda x: x['s'], True),               # Physical Desc
        (lambda x: x['w']/(x['s']+1e-6), True)  # Density Desc
    ]
    
    for key_func, reverse in sorters:
        sorted_items = sorted(pack_items, key=key_func, reverse=reverse)
        
        # Try FFD (First Fit Decreasing)
        res = _pack_ffd(gpu_num, sorted_items, virtual_cap)
        if res: return True, res
        
        # Try BFD (Best Fit Decreasing)
        res = _pack_bfd(gpu_num, sorted_items, virtual_cap)
        if res: return True, res
        
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

def _simulated_annealing(gpu_num, placement):
    """
    Refines placement using Simulated Annealing with variance-based tie-breaking.
    Uses shallow copies for performance.
    """
    # Initialize state caches
    gpu_s = [sum(m.model_size for m in placement[i]) for i in range(gpu_num)]
    gpu_w = [sum(m.req_rate/m.slo for m in placement[i]) for i in range(gpu_num)]
    
    def get_kvpr(i):
        rem = GPU_MEM_SIZE - gpu_s[i]
        if rem <= 1e-7: return 1e9
        return gpu_w[i] / rem
        
    # Pre-calculate initial metrics
    current_kvprs = [get_kvpr(i) for i in range(gpu_num)]
    max_k = max(current_kvprs)
    sum_sq_k = sum(k*k for k in current_kvprs)
    
    # Keep track of best solution
    best_max_k = max_k
    # Use shallow copy of lists, models are immutable refs
    best_placement = {i: list(placement[i]) for i in range(gpu_num)}
    
    # SA Parameters
    T = max_k * 0.05  # Start temperature at 5% of max pressure
    T_min = 1e-4
    alpha = 0.95      # Cooling rate
    steps = 600       # Iterations
    
    for step in range(steps):
        if T < T_min: break
        
        # 1. Select Source GPU
        # 70% chance to pick bottleneck, 30% random
        if random.random() < 0.7:
            src = max(range(gpu_num), key=lambda i: current_kvprs[i])
        else:
            src = random.randint(0, gpu_num - 1)
            
        if not placement[src]: 
            T *= alpha
            continue
            
        # 2. Select Move Type: 60% Move, 40% Swap
        move_type = 'move' if random.random() < 0.6 else 'swap'
        accepted = False
        
        if move_type == 'move':
            # Select model from source
            m_idx = random.randint(0, len(placement[src])-1)
            m = placement[src][m_idx]
            m_s = m.model_size
            m_w = m.req_rate / m.slo
            
            # Select destination
            dst = random.randint(0, gpu_num - 1)
            if src == dst: continue
            
            # Check physical constraints
            if gpu_s[dst] + m_s <= GPU_MEM_SIZE:
                # Calculate new states
                rem_src = GPU_MEM_SIZE - (gpu_s[src] - m_s)
                new_k_src = (gpu_w[src] - m_w) / rem_src if rem_src > 1e-7 else 1e9
                
                rem_dst = GPU_MEM_SIZE - (gpu_s[dst] + m_s)
                new_k_dst = (gpu_w[dst] + m_w) / rem_dst if rem_dst > 1e-7 else 1e9
                
                # Delta evaluation
                old_k_src = current_kvprs[src]
                old_k_dst = current_kvprs[dst]
                
                # Update temp list to find new global max
                current_kvprs[src] = new_k_src
                current_kvprs[dst] = new_k_dst
                new_max_k = max(current_kvprs)
                new_sum_sq_k = sum_sq_k - old_k_src**2 - old_k_dst**2 + new_k_src**2 + new_k_dst**2
                
                delta_max = new_max_k - max_k
                
                # Acceptance Logic
                if delta_max < -1e-6:
                    accepted = True
                elif delta_max < 1e-6:
                    # Tie-breaking: prefer lower variance
                    if new_sum_sq_k < sum_sq_k:
                        accepted = True
                    else:
                        prob = math.exp(-(new_sum_sq_k - sum_sq_k) / (T * max_k * 10))
                        if random.random() < prob: accepted = True
                else:
                    # Metropolis for worsening
                    prob = math.exp(-delta_max / T)
                    if random.random() < prob: accepted = True
                
                if accepted:
                    placement[dst].append(m)
                    placement[src].pop(m_idx)
                    gpu_s[src] -= m_s; gpu_w[src] -= m_w
                    gpu_s[dst] += m_s; gpu_w[dst] += m_w
                    max_k = new_max_k
                    sum_sq_k = new_sum_sq_k
                else:
                    # Revert temp list
                    current_kvprs[src] = old_k_src
                    current_kvprs[dst] = old_k_dst

        elif move_type == 'swap':
            # Select models
            m1_idx = random.randint(0, len(placement[src])-1)
            m1 = placement[src][m1_idx]
            
            dst = random.randint(0, gpu_num - 1)
            if src == dst or not placement[dst]: continue
            
            m2_idx = random.randint(0, len(placement[dst])-1)
            m2 = placement[dst][m2_idx]
            
            # Check capacity
            new_s_src = gpu_s[src] - m1.model_size + m2.model_size
            new_s_dst = gpu_s[dst] - m2.model_size + m1.model_size
            
            if new_s_src <= GPU_MEM_SIZE and new_s_dst <= GPU_MEM_SIZE:
                rem_src = GPU_MEM_SIZE - new_s_src
                new_w_src = gpu_w[src] - (m1.req_rate/m1.slo) + (m2.req_rate/m2.slo)
                new_k_src = new_w_src / rem_src if rem_src > 1e-7 else 1e9
                
                rem_dst = GPU_MEM_SIZE - new_s_dst
                new_w_dst = gpu_w[dst] - (m2.req_rate/m2.slo) + (m1.req_rate/m1.slo)
                new_k_dst = new_w_dst / rem_dst if rem_dst > 1e-7 else 1e9
                
                old_k_src = current_kvprs[src]
                old_k_dst = current_kvprs[dst]
                
                current_kvprs[src] = new_k_src
                current_kvprs[dst] = new_k_dst
                new_max_k = max(current_kvprs)
                new_sum_sq_k = sum_sq_k - old_k_src**2 - old_k_dst**2 + new_k_src**2 + new_k_dst**2
                
                delta_max = new_max_k - max_k
                
                if delta_max < -1e-6:
                    accepted = True
                elif delta_max < 1e-6:
                     if new_sum_sq_k < sum_sq_k:
                        accepted = True
                     else:
                        prob = math.exp(-(new_sum_sq_k - sum_sq_k) / (T * max_k * 10))
                        if random.random() < prob: accepted = True
                else:
                    prob = math.exp(-delta_max / T)
                    if random.random() < prob: accepted = True
                        
                if accepted:
                    placement[src][m1_idx] = m2
                    placement[dst][m2_idx] = m1
                    gpu_s[src] = new_s_src; gpu_w[src] = new_w_src
                    gpu_s[dst] = new_s_dst; gpu_w[dst] = new_w_dst
                    max_k = new_max_k
                    sum_sq_k = new_sum_sq_k
                else:
                    current_kvprs[src] = old_k_src
                    current_kvprs[dst] = old_k_dst
        
        # Update Global Best
        if max_k < best_max_k - 1e-6:
            best_max_k = max_k
            best_placement = {i: list(placement[i]) for i in range(gpu_num)}
            
        # Cooling
        T *= alpha
        
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

