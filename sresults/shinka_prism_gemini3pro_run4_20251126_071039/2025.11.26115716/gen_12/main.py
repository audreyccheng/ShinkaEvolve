# EVOLVE-BLOCK-START
"""Model placement algorithm for minimizing maximum KV cache pressure across GPUs"""

import random

GPU_MEM_SIZE = 80  # GB

def compute_model_placement(gpu_num, models):
    """
    Compute a model placement that minimizes the maximum KVPR across all GPUs.
    
    Algorithm:
    1. Calculate Theoretical Lower Bound (LB) for K.
    2. Binary Search for optimal K between LB and High.
       - Feasibility Check: 'Noisy Best-Fit Decreasing'.
         - Checks if items fit in M bins with capacity constraints linearized by K.
         - Uses randomized sorting keys (perturbations) to retry if deterministic fit fails.
       - Updates Upper Bound aggressively using the actual max KVPR of valid solutions.
    3. Local Search Refinement.
       - Greedy Hill-Climbing (Moves and Swaps) to reduce the bottleneck GPU's pressure.

    Args:
        gpu_num: Number of GPUs
        models: List of models to place

    Returns:
        A placement of models to GPUs
    """

    # 1. Preprocessing
    items = []
    total_w = 0.0
    total_s = 0.0
    
    for m in models:
        w = m.req_rate / m.slo
        s = m.model_size
        items.append({'model': m, 'w': w, 's': s})
        total_w += w
        total_s += s

    # Theoretical Lower Bound: K >= Sum(w) / (M*C - Sum(s))
    rem_global = gpu_num * GPU_MEM_SIZE - total_s
    k_min = 0.0
    if rem_global > 1e-9:
        k_min = total_w / rem_global
    elif total_w > 1e-9:
        # If no global memory remains but we have load, it's impossible. 
        # But we let the search handle the failure.
        k_min = 1e9 

    def get_max_kvpr(placement_list):
        """Calculate actual max KVPR of a list-based placement."""
        max_p = 0.0
        for p in placement_list:
            w = sum(m.req_rate / m.slo for m in p)
            s = sum(m.model_size for m in p)
            rem = GPU_MEM_SIZE - s
            if rem <= 1e-9:
                if w > 1e-9: return float('inf')
                val = 0.0
            else:
                val = w / rem
            if val > max_p:
                max_p = val
        return max_p

    def pack(k_target, max_attempts=5):
        """
        Try to pack items into gpu_num bins such that for all j:
        sum(w) / (C - sum(s)) <= k_target
        Equivalently: sum(w + k_target*s) <= k_target * C
        """
        
        # Helper for a single packing attempt
        def try_order(ordered_items):
            # Track current usage
            # Linear Capacity = K * C
            # Item Linear Size = w + K * s
            # We use Best Fit Decreasing on Linear Size to minimize wasted Linear Capacity
            
            pl = [[] for _ in range(gpu_num)]
            g_w = [0.0] * gpu_num
            g_s = [0.0] * gpu_num
            
            for item in ordered_items:
                w = item['w']
                s = item['s']
                # Cost in the linearized domain
                cost = w + k_target * s
                
                best_idx = -1
                best_residue = float('inf')
                
                for i in range(gpu_num):
                    # Hard memory limit
                    if g_s[i] + s > GPU_MEM_SIZE:
                        continue
                    
                    # KVPR / Linear Capacity Limit
                    # Current load: g_w + K * g_s
                    # New load: (g_w + w) + K * (g_s + s)
                    current_lin_load = g_w[i] + k_target * g_s[i]
                    new_lin_load = current_lin_load + cost
                    limit = k_target * GPU_MEM_SIZE
                    
                    if new_lin_load > limit + 1e-5:
                        continue
                        
                    # Best Fit: Minimize remaining space in the linearized bin
                    # residue = limit - new_lin_load
                    # We want to minimize residue => maximize new_lin_load
                    residue = limit - new_lin_load
                    
                    if residue < best_residue:
                        best_residue = residue
                        best_idx = i
                
                if best_idx != -1:
                    pl[best_idx].append(item['model'])
                    g_w[best_idx] += w
                    g_s[best_idx] += s
                else:
                    return None
            return pl

        # 1. Deterministic Strategy
        # Sort by decreasing linear cost: w + k*s
        base_items = sorted(items, key=lambda x: x['w'] + k_target * x['s'], reverse=True)
        res = try_order(base_items)
        if res: return res
        
        # 2. Randomized Restarts (Noisy Sorting)
        if max_attempts > 0:
            # Seed based on k_target to ensure determinism for the same call
            rng = random.Random(hash(k_target))
            for _ in range(max_attempts):
                # Perturb the sort keys by +/- 10%
                # This keeps the general "heavy first" heuristic but changes local ordering
                noisy_items = sorted(items, key=lambda x: (x['w'] + k_target * x['s']) * rng.uniform(0.9, 1.1), reverse=True)
                res = try_order(noisy_items)
                if res: return res
                
        return None

    # 2. Binary Search
    best_pl_list = None
    
    # Initial Check at High Bound
    high = 1e9
    # Quick check
    if not pack(high, 0):
        # Detailed check
        if not pack(high, 20):
            raise ValueError("Unable to place models on GPUs (insufficient memory).")
    
    # If high check passed, we have a candidate. But pack(high) creates a packing valid for K=1e9.
    # We want to measure its actual K to tighten 'high'.
    # Re-run pack to capture the result
    best_pl_list = pack(high, 20)
    high = get_max_kvpr(best_pl_list)
    if high == float('inf'): high = 1e9
    
    low = k_min
    
    # Search Loop
    for _ in range(25):
        if high - low < 1e-4: break
        mid = (low + high) / 2
        
        # Try to pack with K = mid
        res = pack(mid, max_attempts=5)
        
        if res:
            best_pl_list = res
            actual_max = get_max_kvpr(res)
            # If we found a valid packing with actual max KVPR 'actual_max',
            # we know a solution exists for K = actual_max.
            # And by definition actual_max <= mid (since it fit in mid).
            # So we can safely lower high to actual_max.
            high = min(mid, actual_max)
        else:
            low = mid

    # Convert to dict format
    final_pl = {i: best_pl_list[i] for i in range(gpu_num)}

    # 3. Local Search Refinement
    # Optimize the placement to further reduce the bottleneck
    
    # State tracking
    states = []
    for i in range(gpu_num):
        ms = final_pl[i]
        w = sum(m.req_rate/m.slo for m in ms)
        s = sum(m.model_size for m in ms)
        rem = GPU_MEM_SIZE - s
        val = w/rem if rem > 1e-9 else (float('inf') if w > 1e-9 else 0.0)
        states.append({'w': w, 's': s, 'models': ms, 'kvpr': val})

    # Hill Climbing Loop
    for _ in range(150):
        # Identify bottleneck
        # Sort indices by KVPR desc
        sorted_gpus = sorted(range(gpu_num), key=lambda x: states[x]['kvpr'], reverse=True)
        src = sorted_gpus[0]
        curr_max = states[src]['kvpr']
        
        if curr_max <= 0: break
        
        improved = False
        
        # Strategy A: Move a model from Source to another GPU
        # We want to find a move that results in max(new_src_kvpr, new_dst_kvpr) < curr_max
        
        for m_idx, m in enumerate(states[src]['models']):
            m_w = m.req_rate / m.slo
            m_s = m.model_size
            
            # Predict Src after removal
            ns_s = states[src]['s'] - m_s
            ns_w = states[src]['w'] - m_w
            ns_rem = GPU_MEM_SIZE - ns_s
            ns_kvpr = ns_w/ns_rem if ns_rem > 1e-9 else 0.0
            
            if ns_kvpr >= curr_max: continue # Source not improved enough
            
            for dst in sorted_gpus[1:]:
                # Predict Dst after addition
                nd_s = states[dst]['s'] + m_s
                if nd_s > GPU_MEM_SIZE: continue
                
                nd_w = states[dst]['w'] + m_w
                nd_rem = GPU_MEM_SIZE - nd_s
                nd_kvpr = nd_w/nd_rem if nd_rem > 1e-9 else (float('inf') if nd_w > 1e-9 else 0.0)
                
                # Check if this move is globally better (locally for these 2 nodes)
                # Since 'dst' was <= curr_max, we just need to ensure new 'dst' < curr_max
                if nd_kvpr < curr_max:
                    # Apply Move
                    model = states[src]['models'].pop(m_idx)
                    states[dst]['models'].append(model)
                    
                    states[src]['s'] = ns_s
                    states[src]['w'] = ns_w
                    states[src]['kvpr'] = ns_kvpr
                    
                    states[dst]['s'] = nd_s
                    states[dst]['w'] = nd_w
                    states[dst]['kvpr'] = nd_kvpr
                    
                    improved = True
                    break
            if improved: break
            
        if improved: continue
        
        # Strategy B: Swap a model from Source with a model from another GPU
        for m1_idx, m1 in enumerate(states[src]['models']):
            m1_w = m1.req_rate / m1.slo
            m1_s = m1.model_size
            
            for dst in sorted_gpus[1:]:
                # Optimization: Don't swap with highly loaded GPUs
                if states[dst]['kvpr'] > curr_max * 0.95: continue
                
                for m2_idx, m2 in enumerate(states[dst]['models']):
                    m2_w = m2.req_rate / m2.slo
                    m2_s = m2.model_size
                    
                    # New Src
                    ns_s = states[src]['s'] - m1_s + m2_s
                    if ns_s > GPU_MEM_SIZE: continue
                    ns_w = states[src]['w'] - m1_w + m2_w
                    ns_rem = GPU_MEM_SIZE - ns_s
                    ns_kvpr = ns_w/ns_rem if ns_rem > 1e-9 else (float('inf') if ns_w > 1e-9 else 0.0)
                    
                    if ns_kvpr >= curr_max: continue

                    # New Dst
                    nd_s = states[dst]['s'] - m2_s + m1_s
                    if nd_s > GPU_MEM_SIZE: continue
                    nd_w = states[dst]['w'] - m2_w + m1_w
                    nd_rem = GPU_MEM_SIZE - nd_s
                    nd_kvpr = nd_w/nd_rem if nd_rem > 1e-9 else (float('inf') if nd_w > 1e-9 else 0.0)
                    
                    if nd_kvpr < curr_max:
                        # Apply Swap
                        model1 = states[src]['models'][m1_idx]
                        model2 = states[dst]['models'][m2_idx]
                        
                        states[src]['models'][m1_idx] = model2
                        states[dst]['models'][m2_idx] = model1
                        
                        states[src]['s'] = ns_s
                        states[src]['w'] = ns_w
                        states[src]['kvpr'] = ns_kvpr
                        
                        states[dst]['s'] = nd_s
                        states[dst]['w'] = nd_w
                        states[dst]['kvpr'] = nd_kvpr
                        
                        improved = True
                        break
                if improved: break
            if improved: break

    return {i: states[i]['models'] for i in range(gpu_num)}
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
