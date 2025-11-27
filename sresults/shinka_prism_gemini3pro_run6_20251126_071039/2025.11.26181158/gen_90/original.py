# EVOLVE-BLOCK-START
"""Model placement algorithm for minimizing maximum KV cache pressure across GPUs"""

GPU_MEM_SIZE = 80  # GB

def compute_model_placement(gpu_num, models):
    """
    Compute a model placement that minimizes the maximum KVPR across all GPUs.
    
    Algorithm:
    1. Binary Search on target KVPR 'K'.
       - Checks feasibility by solving a Bin Packing problem with Best Fit.
       - Uses multiple sorting heuristics (Linearized Weight, Size) to maximize packing success.
    2. Local Search Refinement.
       - Greedily moves models from the bottleneck GPU (max KVPR) to other GPUs
         to reduce the global maximum KVPR.
    """
    
    # Pre-calculate model properties
    model_data = []
    for i, m in enumerate(models):
        model_data.append({
            'model': m,
            'l': m.req_rate / m.slo,
            's': m.model_size
        })

    def solve_packing(target_k):
        """
        Attempts to place models into gpu_num bins given a target KVPR 'K'.
        Constraint per GPU: sum(L) + K * sum(S) <= K * M
        Returns placement dict if successful, None otherwise.
        """
        capacity = target_k * GPU_MEM_SIZE
        
        # Helper to try a specific packing order with Best Fit
        def try_best_fit(items):
            gpu_l = [0.0] * gpu_num
            gpu_s = [0.0] * gpu_num
            gpu_models = [[] for _ in range(gpu_num)]
            
            for item in items:
                best_idx = -1
                min_remaining = float('inf')
                
                w = item['l'] + target_k * item['s']
                
                for i in range(gpu_num):
                    # Hard memory constraint
                    if gpu_s[i] + item['s'] >= GPU_MEM_SIZE - 1e-6:
                        continue
                        
                    # Linearized KVPR constraint
                    curr_w = gpu_l[i] + target_k * gpu_s[i]
                    if curr_w + w <= capacity + 1e-9:
                        # Best Fit: Choose bin with minimum remaining linearized capacity
                        rem = capacity - (curr_w + w)
                        if rem < min_remaining:
                            min_remaining = rem
                            best_idx = i
                
                if best_idx != -1:
                    gpu_l[best_idx] += item['l']
                    gpu_s[best_idx] += item['s']
                    gpu_models[best_idx].append(item['model'])
                else:
                    return None
            return gpu_models

        # Strategy 1: Sort by Linearized Weight Descending
        # Effective for generic bin packing logic
        items_1 = sorted(model_data, key=lambda x: x['l'] + target_k * x['s'], reverse=True)
        res = try_best_fit(items_1)
        if res: return res
        
        # Strategy 2: Sort by Size Descending
        # Effective when memory is the primary constraint
        items_2 = sorted(model_data, key=lambda x: x['s'], reverse=True)
        res = try_best_fit(items_2)
        if res: return res
        
        return None

    # --- Phase 1: Binary Search for Optimal K ---
    
    # Find valid upper bound
    low = 0.0
    high = 1.0
    best_placement_list = None
    
    # Exponential search
    for _ in range(20):
        res = solve_packing(high)
        if res is not None:
            best_placement_list = res
            break
        low = high
        high *= 2.0
    else:
        # Fallback
        high = 1e9
        
    # Refine K
    for _ in range(25):
        mid = (low + high) / 2
        res = solve_packing(mid)
        if res is not None:
            best_placement_list = res
            high = mid
        else:
            low = mid
            
    # Ensure we have a placement
    if best_placement_list is None:
        best_placement_list = solve_packing(high)
        if best_placement_list is None:
            raise ValueError("Unable to place models even with infinite KVPR.")

    placement = {i: best_placement_list[i] for i in range(gpu_num)}

    # --- Phase 2: Local Search Refinement ---
    
    def calculate_gpu_kvpr(p_models):
        sl = sum(m.req_rate / m.slo for m in p_models)
        ss = sum(m.model_size for m in p_models)
        if ss >= GPU_MEM_SIZE: return float('inf')
        return sl / (GPU_MEM_SIZE - ss)

    # Greedily improve the worst GPU
    for _ in range(50):
        # Find current max KVPR GPU
        kvprs = {g: calculate_gpu_kvpr(placement[g]) for g in range(gpu_num)}
        max_gpu = max(kvprs, key=kvprs.get)
        max_val = kvprs[max_gpu]
        
        if max_val == 0: break # Optimal
        
        best_move = None
        best_improvement = 0.0
        
        # Try moving each model from max_gpu to any other GPU
        src_models = placement[max_gpu]
        
        for idx, model in enumerate(src_models):
            for tgt_gpu in range(gpu_num):
                if tgt_gpu == max_gpu: continue
                
                tgt_models = placement[tgt_gpu]
                
                # Check memory fit
                if sum(m.model_size for m in tgt_models) + model.model_size >= GPU_MEM_SIZE:
                    continue
                
                # Predict new KVPRs
                ns_l = sum(m.req_rate / m.slo for m in src_models) - (model.req_rate / model.slo)
                ns_s = sum(m.model_size for m in src_models) - model.model_size
                ns_kvpr = ns_l / (GPU_MEM_SIZE - ns_s) if ns_s < GPU_MEM_SIZE else float('inf')
                
                nt_l = sum(m.req_rate / m.slo for m in tgt_models) + (model.req_rate / model.slo)
                nt_s = sum(m.model_size for m in tgt_models) + model.model_size
                nt_kvpr = nt_l / (GPU_MEM_SIZE - nt_s) if nt_s < GPU_MEM_SIZE else float('inf')
                
                # We accept if the new local max is better than the old global max
                new_local_max = max(ns_kvpr, nt_kvpr)
                
                if new_local_max < max_val - 1e-5:
                    improv = max_val - new_local_max
                    if improv > best_improvement:
                        best_improvement = improv
                        best_move = (idx, tgt_gpu)
        
        if best_move:
            m_idx, t_gpu = best_move
            model_to_move = placement[max_gpu].pop(m_idx)
            placement[t_gpu].append(model_to_move)
        else:
            # No move from the bottleneck GPU improved the situation
            break
            
    return placement

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

