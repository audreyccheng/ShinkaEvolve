# EVOLVE-BLOCK-START
"""Model placement algorithm for minimizing maximum KV cache pressure using Hybrid Binary Packing and Ruin-Recreate Local Search"""

import copy
import random

GPU_MEM_SIZE = 80.0  # GB

def compute_model_placement(gpu_num, models):
    """
    Minimizes max KVPR using:
    1. Binary Search with Robust Best-Fit Decreasing Packing.
    2. Iterated Local Search with Steepest Descent and Ruin-Recreate Kicks.
    """
    # 1. Validation and Setup
    total_size = sum(m.model_size for m in models)
    if total_size > gpu_num * GPU_MEM_SIZE:
        raise ValueError("Total model size exceeds total GPU memory capacity.")

    # Pre-calculate model weights
    # item: (w, s, m) -> w = req_rate/slo
    items = [(m.req_rate / m.slo, m.model_size, m) for m in models]

    # 2. Binary Search for Initial Feasible Solution
    # Determine bounds
    total_w = sum(x[0] for x in items)
    slack = gpu_num * GPU_MEM_SIZE - total_size
    
    low = 0.0
    if slack < 1e-5:
        high = 1000.0
    else:
        # Heuristic upper bound
        avg_pressure = total_w / slack
        high = max(10.0, avg_pressure * 6.0)

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
        raise ValueError("Unable to place models. Constraints too tight.")

    # Binary Search
    for _ in range(30):
        mid = (low + high) / 2.0
        feasible, placement = _check_feasibility_robust(gpu_num, items, mid)
        if feasible:
            best_placement = placement
            high = mid
        else:
            low = mid
            
    # Convert to map
    placement_map = {i: best_placement[i] for i in range(gpu_num)}
    
    # 3. Refinement: ILS with Ruin & Recreate
    return _ils_ruin_recreate(gpu_num, placement_map)

def _check_feasibility_robust(gpu_num, items, K):
    """
    Checks feasibility using multiple sorting heuristics and Best-Fit Decreasing.
    """
    virtual_cap = K * GPU_MEM_SIZE
    # Prepare items with virtual size
    # (virtual, physical, load, density, model)
    pack_items = []
    for w, s, m in items:
        v = w + K * s
        pack_items.append({'v': v, 's': s, 'w': w, 'm': m})
        
    # Heuristics: (key_lambda, reverse)
    heuristics = [
        (lambda x: x['v'], True),  # Virtual Size Desc
        (lambda x: x['s'], True),  # Physical Size Desc
        (lambda x: x['w'], True),  # Load Desc
        (lambda x: x['w']/(x['s']+1e-7), True), # Density Desc
    ]
    
    for key_func, rev in heuristics:
        pack_items.sort(key=key_func, reverse=rev)
        res = _pack_bfd(gpu_num, pack_items, virtual_cap)
        if res: return True, res
        
    return False, None

def _pack_bfd(gpu_num, items, virtual_cap):
    """
    Best Fit Decreasing packing based on Virtual Capacity.
    """
    bins_v = [0.0] * gpu_num
    bins_p = [0.0] * gpu_num
    placement = [[] for _ in range(gpu_num)]
    
    for item in items:
        best_bin = -1
        min_rem_v = float('inf')
        
        for i in range(gpu_num):
            if bins_p[i] + item['s'] <= GPU_MEM_SIZE and bins_v[i] + item['v'] <= virtual_cap + 1e-7:
                # Minimize residual virtual capacity
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

def _ils_ruin_recreate(gpu_num, placement):
    """
    Iterated Local Search with Steepest Descent and Randomized Ruin-Recreate.
    """
    # State tracking
    gpu_s = [sum(m.model_size for m in placement[i]) for i in range(gpu_num)]
    gpu_w = [sum(m.req_rate / m.slo for m in placement[i]) for i in range(gpu_num)]
    
    def get_k(idx):
        rem = GPU_MEM_SIZE - gpu_s[idx]
        if rem <= 1e-7: return 1e9
        return gpu_w[idx] / rem
        
    # Initial evaluation
    best_max_k = max(get_k(i) for i in range(gpu_num))
    best_sol = copy.deepcopy(placement)
    
    no_improve = 0
    patience = 20
    max_steps = 400
    
    for step in range(max_steps):
        # 1. Identify Bottleneck
        current_ks = [get_k(i) for i in range(gpu_num)]
        max_k = max(current_ks)
        src = current_ks.index(max_k)
        
        # Update Global Best
        if max_k < best_max_k - 1e-7:
            best_max_k = max_k
            best_sol = copy.deepcopy(placement)
            no_improve = 0
        else:
            no_improve += 1
            
        # 2. Strategy Selection
        if no_improve <= patience:
            # --- STEEPEST DESCENT (MOVES & SWAPS) ---
            best_move = None # (type, ...)
            best_delta = -1.0 # improvement in peak K
            
            src_models = placement[src]
            
            # A. Evaluate Moves from SRC
            for i, m in enumerate(src_models):
                s, w = m.model_size, m.req_rate/m.slo
                for dst in range(gpu_num):
                    if dst == src: continue
                    if gpu_s[dst] + s > GPU_MEM_SIZE: continue
                    
                    # New Ks
                    nk_src = (gpu_w[src] - w) / (GPU_MEM_SIZE - (gpu_s[src] - s) + 1e-9)
                    nk_dst = (gpu_w[dst] + w) / (GPU_MEM_SIZE - (gpu_s[dst] + s) + 1e-9)
                    
                    new_peak = max(nk_src, nk_dst)
                    if new_peak < max_k - 1e-7:
                        imp = max_k - new_peak
                        if imp > best_delta:
                            best_delta = imp
                            best_move = ('move', i, dst, s, w)
            
            # B. Evaluate Swaps with SRC
            # Optimization: Only check if delta is small or non-existent
            if best_delta < 0.5: 
                for i1, m1 in enumerate(src_models):
                    s1, w1 = m1.model_size, m1.req_rate/m1.slo
                    for dst in range(gpu_num):
                        if dst == src: continue
                        if current_ks[dst] > max_k * 0.95: continue # Skip if dst is also stressed
                        
                        for i2, m2 in enumerate(placement[dst]):
                            s2, w2 = m2.model_size, m2.req_rate/m2.slo
                            
                            ns_src = gpu_s[src] - s1 + s2
                            ns_dst = gpu_s[dst] - s2 + s1
                            if ns_src > GPU_MEM_SIZE or ns_dst > GPU_MEM_SIZE: continue
                            
                            nk_src = (gpu_w[src] - w1 + w2) / (GPU_MEM_SIZE - ns_src + 1e-9)
                            nk_dst = (gpu_w[dst] - w2 + w1) / (GPU_MEM_SIZE - ns_dst + 1e-9)
                            
                            new_peak = max(nk_src, nk_dst)
                            if new_peak < max_k - 1e-7:
                                imp = max_k - new_peak
                                if imp > best_delta:
                                    best_delta = imp
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
                no_improve = 0 
                continue

        # 3. KICK: Ruin & Recreate 
        # Select subset: Bottleneck + Min Load + Random
        subset = {src}
        min_k_idx = current_ks.index(min(current_ks))
        if min_k_idx != src: subset.add(min_k_idx)
        
        candidates = [x for x in range(gpu_num) if x not in subset]
        if candidates: subset.add(random.choice(candidates))
            
        subset = list(subset)
        
        # Backup
        backup_placement = {i: list(placement[i]) for i in subset}
        backup_s = {i: gpu_s[i] for i in subset}
        backup_w = {i: gpu_w[i] for i in subset}
        
        repack_models = []
        for i in subset:
            repack_models.extend(placement[i])
            placement[i] = []
            gpu_s[i] = 0.0
            gpu_w[i] = 0.0
            
        # Randomized Greedy Packing Trials
        best_repack_max = float('inf')
        best_repack_state = None
        
        for _ in range(20): # Trials
            random.shuffle(repack_models)
            trial_s = {i: 0.0 for i in subset}
            trial_w = {i: 0.0 for i in subset}
            trial_placement = {i: [] for i in subset}
            possible = True
            
            for m in repack_models:
                best_g = -1
                best_local_k = float('inf')
                
                for g in subset:
                    if trial_s[g] + m.model_size <= GPU_MEM_SIZE:
                        # Local K minimization
                        rem = GPU_MEM_SIZE - (trial_s[g] + m.model_size)
                        k = (trial_w[g] + m.req_rate/m.slo) / (rem + 1e-9)
                        if k < best_local_k:
                            best_local_k = k
                            best_g = g
                            
                if best_g != -1:
                    trial_placement[best_g].append(m)
                    trial_s[best_g] += m.model_size
                    trial_w[best_g] += m.req_rate/m.slo
                else:
                    possible = False
                    break
            
            if possible:
                local_max = max((trial_w[i] / (GPU_MEM_SIZE - trial_s[i] + 1e-9)) for i in subset)
                if local_max < best_repack_max:
                    best_repack_max = local_max
                    best_repack_state = (trial_placement, trial_s, trial_w)
        
        # Apply Repack if beneficial or probabalistic kick
        current_subset_max = max(current_ks[i] for i in subset)
        
        if best_repack_state and (best_repack_max < current_subset_max - 1e-7 or random.random() < 0.2):
            t_place, t_s, t_w = best_repack_state
            for i in subset:
                placement[i] = t_place[i]
                gpu_s[i] = t_s[i]
                gpu_w[i] = t_w[i]
            no_improve = max(0, patience - 5)
        else:
            # Revert
            for i in subset:
                placement[i] = backup_placement[i]
                gpu_s[i] = backup_s[i]
                gpu_w[i] = backup_w[i]
                
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
