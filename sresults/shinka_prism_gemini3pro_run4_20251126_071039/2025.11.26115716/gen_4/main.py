# EVOLVE-BLOCK-START
"""Model placement algorithm for minimizing maximum KV cache pressure across GPUs"""

GPU_MEM_SIZE = 80  # GB

def compute_model_placement(gpu_num, models):
    """
    Compute a model placement that minimizes the maximum KVPR across all GPUs.
    
    Combines two heuristics:
    1. Greedy Best-Fit: Places models on the GPU that minimizes immediate KVPR.
    2. Binary Search with Best-Fit-Decreasing: Optimizes a global KVPR constraint K
       using a bin packing heuristic with dynamic item sizes (w + K*s).
       
    Args:
        gpu_num: Number of GPUs
        models: List of models to place

    Returns:
        A placement of models to GPUs
    """
    
    # Pre-process models for faster access
    items = []
    for m in models:
        items.append({
            'model': m,
            'w': m.req_rate / m.slo,
            's': m.model_size
        })

    # --- Strategy 1: Greedy Constructive Heuristic ---
    # Sort by weight (req/slo) descending
    sorted_models_g = sorted(models, key=lambda m: (m.req_rate / m.slo), reverse=True)
    
    placement_greedy = {i: [] for i in range(gpu_num)}
    state_greedy = [{'w': 0.0, 's': 0.0} for _ in range(gpu_num)]
    greedy_possible = True
    
    for model in sorted_models_g:
        w = model.req_rate / model.slo
        s = model.model_size
        best_idx = None
        best_val = float('inf')
        
        for i in range(gpu_num):
            new_s = state_greedy[i]['s'] + s
            if new_s <= GPU_MEM_SIZE:
                new_w = state_greedy[i]['w'] + w
                rem = GPU_MEM_SIZE - new_s
                
                # Calculate resulting KVPR on this GPU
                if rem <= 1e-9:
                    if new_w > 1e-9: val = float('inf')
                    else: val = 0.0
                else:
                    val = new_w / rem
                
                if val < best_val:
                    best_val = val
                    best_idx = i
        
        if best_idx is None:
            greedy_possible = False
            break
        
        placement_greedy[best_idx].append(model)
        state_greedy[best_idx]['w'] += w
        state_greedy[best_idx]['s'] += s

    # --- Strategy 2: Binary Search with Best-Fit Decreasing ---
    
    def check_placement_bfd(k_target):
        # Sort by linearized size: w + k*s
        # High K -> Sort by Size. Low K -> Sort by Weight.
        sorted_items = sorted(items, key=lambda x: x['w'] + k_target * x['s'], reverse=True)
        
        placement = {i: [] for i in range(gpu_num)}
        # Track load components and linearized load
        gpu_load = [{'w': 0.0, 's': 0.0, 'lin': 0.0} for _ in range(gpu_num)]
        
        for item in sorted_items:
            w = item['w']
            s = item['s']
            item_lin = w + k_target * s
            
            best_idx = None
            best_lin_load = -1.0
            
            for i in range(gpu_num):
                # Hard memory constraint
                new_s = gpu_load[i]['s'] + s
                if new_s > GPU_MEM_SIZE:
                    continue
                
                # KVPR constraint: w_total / (C - s_total) <= k
                # Equivalent to: w_total <= k * (C - s_total)
                new_w = gpu_load[i]['w'] + w
                rem_mem = GPU_MEM_SIZE - new_s
                
                if rem_mem <= 1e-9:
                    # Avoid division by zero / infinite pressure
                    # Allow only if load is 0
                    if new_w > 1e-9:
                        continue
                elif new_w > k_target * rem_mem + 1e-7:
                    continue
                
                # Best Fit: Pick the GPU that is fullest (max current linear load)
                # This minimizes the remaining space in that bin
                if gpu_load[i]['lin'] > best_lin_load:
                    best_lin_load = gpu_load[i]['lin']
                    best_idx = i
            
            if best_idx is None:
                return None
            
            placement[best_idx].append(item['model'])
            gpu_load[best_idx]['s'] += s
            gpu_load[best_idx]['w'] += w
            gpu_load[best_idx]['lin'] += item_lin
            
        return placement

    # Helper to score a placement
    def calculate_max_kvpr(pl):
        if pl is None: return float('inf')
        max_p = 0.0
        for p_models in pl.values():
            w_sum = sum(m.req_rate / m.slo for m in p_models)
            s_sum = sum(m.model_size for m in p_models)
            rem = GPU_MEM_SIZE - s_sum
            if rem <= 1e-9:
                if w_sum > 1e-9: return float('inf')
                val = 0.0
            else:
                val = w_sum / rem
            if val > max_p: max_p = val
        return max_p

    # Run Binary Search
    placement_bs = None
    low = 0.0
    high = 1e9
    
    # Initialization
    init_pl = check_placement_bfd(high)
    if init_pl is None:
        # If even relaxed check fails, we can't fit models via this heuristic
        placement_bs = None 
    else:
        placement_bs = init_pl
        # Refine high bound to actual max KVPR found
        high = calculate_max_kvpr(init_pl)
        if high == float('inf'): high = 1e9
        
        # Search loop
        for _ in range(25):
            mid = (low + high) / 2
            res = check_placement_bfd(mid)
            if res is not None:
                placement_bs = res
                high = mid
            else:
                low = mid

    # --- Compare and Return ---
    score_greedy = calculate_max_kvpr(placement_greedy) if greedy_possible else float('inf')
    score_bs = calculate_max_kvpr(placement_bs)

    if score_greedy < score_bs:
        if placement_greedy is None:
             raise ValueError("Unable to place models on GPUs (insufficient memory).")
        return placement_greedy
    
    if placement_bs is None:
         raise ValueError("Unable to place models on GPUs (insufficient memory).")
         
    return placement_bs

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

