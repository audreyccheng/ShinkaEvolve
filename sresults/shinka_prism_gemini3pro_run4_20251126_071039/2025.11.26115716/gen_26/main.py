# EVOLVE-BLOCK-START
"""Model placement algorithm for minimizing maximum KV cache pressure across GPUs"""

import random

GPU_MEM_SIZE = 80  # GB

def compute_model_placement(gpu_num, models):
    """
    Compute a model placement that minimizes the maximum KVPR across all GPUs.
    
    Approach:
    1. Preprocessing: Calculate weight (req/slo) and size for all models.
    2. Binary Search: Find the minimum feasible max-KVPR (K) using a packing check.
       - The packing check uses Best-Fit Decreasing with linearized cost.
       - Includes deterministic strategies and limited randomized trials.
    3. Local Search: Starting from the best feasible placement, iteratively improve
       it by moving or swapping models to reduce the bottleneck GPU's pressure.
    """

    # 1. Preprocessing
    items = []
    for m in models:
        items.append({
            'model': m,
            'w': m.req_rate / m.slo,
            's': m.model_size
        })

    def calc_kvpr(w, s):
        """Calculate KVPR safely."""
        rem = GPU_MEM_SIZE - s
        if rem <= 1e-9:
            return float('inf') if w > 1e-9 else 0.0
        return w / rem

    def get_max_kvpr(placement_list):
        """Get max KVPR from a list-based placement state."""
        max_k = 0.0
        for p in placement_list:
            w = sum(x['w'] for x in p)
            s = sum(x['s'] for x in p)
            k = calc_kvpr(w, s)
            if k > max_k: max_k = k
        return max_k

    def solve_packing(k_target, attempts=1):
        """
        Attempt to place models such that KVPR <= k_target for all GPUs.
        Constraint: w <= k * (C - s)  <==>  w + k*s <= k*C
        """
        limit_val = k_target * GPU_MEM_SIZE
        
        # Strategies to generate ordering of items
        strategies = []
        # 1. Linear Cost: w + k*s (Matches the constraint boundary)
        strategies.append(lambda x: x['w'] + k_target * x['s'])
        # 2. Size (Classic bin packing, good for tight memory)
        strategies.append(lambda x: x['s'])
        # 3. Weight (Good for load balancing)
        strategies.append(lambda x: x['w'])

        def try_pack(ordered_items):
            # State: current w, s per GPU
            bins = [{'w': 0.0, 's': 0.0} for _ in range(gpu_num)]
            allocation = [[] for _ in range(gpu_num)] # List of lists
            
            for item in ordered_items:
                item_lin = item['w'] + k_target * item['s']
                
                best_idx = -1
                min_slack = float('inf') # Best Fit: Minimize remaining space
                
                for i in range(gpu_num):
                    b = bins[i]
                    # Hard Capacity Check
                    if b['s'] + item['s'] > GPU_MEM_SIZE: continue
                    
                    # KVPR / Linearized Check
                    # Current load + item load
                    new_lin = (b['w'] + item['w']) + k_target * (b['s'] + item['s'])
                    
                    if new_lin > limit_val + 1e-5: continue
                    
                    slack = limit_val - new_lin
                    if slack < min_slack:
                        min_slack = slack
                        best_idx = i
                
                if best_idx != -1:
                    allocation[best_idx].append(item)
                    bins[best_idx]['w'] += item['w']
                    bins[best_idx]['s'] += item['s']
                else:
                    return None # Failed to place this item
            return allocation

        # Run Deterministic Strategies
        for key_fn in strategies:
            # Sort Descending
            res = try_pack(sorted(items, key=key_fn, reverse=True))
            if res: return res
            
        # Run Randomized Strategies
        if attempts > 1:
            rng = random.Random(42 + int(k_target))
            # Base order: Linear cost
            for _ in range(attempts - 1):
                # Noisy sort: key * random noise
                noisy_items = sorted(items, key=lambda x: (x['w'] + k_target * x['s']) * rng.uniform(0.9, 1.1), reverse=True)
                res = try_pack(noisy_items)
                if res: return res
                
        return None

    # 2. Binary Search
    high = 1e9
    
    # Check feasibility
    best_placement_list = solve_packing(high, attempts=1)
    if not best_placement_list:
        raise ValueError("Unable to place models on GPUs (insufficient total memory).")
    
    high = get_max_kvpr(best_placement_list)
    low = 0.0
    
    for _ in range(25):
        if high - low < 1e-5: break
        mid = (low + high) / 2
        # Try to pack with target K=mid
        res = solve_packing(mid, attempts=10) # 10 attempts per check
        if res:
            best_placement_list = res
            high = min(mid, get_max_kvpr(res))
        else:
            low = mid

    # 3. Local Search (Hill Climbing)
    # Setup mutable state
    gpu_states = []
    for p in best_placement_list:
        w = sum(x['w'] for x in p)
        s = sum(x['s'] for x in p)
        gpu_states.append({
            'items': list(p),
            'w': w,
            's': s,
            'kvpr': calc_kvpr(w, s)
        })

    # Optimization Loop
    for _ in range(250): # Limit iterations
        # Find bottleneck GPU
        max_kvpr = -1.0
        src_idx = -1
        for i, st in enumerate(gpu_states):
            if st['kvpr'] > max_kvpr:
                max_kvpr = st['kvpr']
                src_idx = i
        
        if max_kvpr <= 1e-9: break
        
        improved = False
        src = gpu_states[src_idx]
        
        # Strategy A: Move item from Src -> Dst
        for i, item in enumerate(src['items']):
            # Predict Src removal
            ns_s = src['s'] - item['s']
            ns_w = src['w'] - item['w']
            ns_kvpr = calc_kvpr(ns_w, ns_s)
            
            # Optimization: If src doesn't improve significantly, try next item
            # But we must ensure src drops below max_kvpr to count as "solving" this bottleneck
            if ns_kvpr >= max_kvpr - 1e-9: continue
            
            for dst_idx, dst in enumerate(gpu_states):
                if dst_idx == src_idx: continue
                
                # Check mem
                nd_s = dst['s'] + item['s']
                if nd_s > GPU_MEM_SIZE: continue
                
                # Predict Dst addition
                nd_w = dst['w'] + item['w']
                nd_kvpr = calc_kvpr(nd_w, nd_s)
                
                # Verify improvement
                if nd_kvpr < max_kvpr - 1e-9:
                    # Apply Move
                    src['items'].pop(i)
                    src['w'], src['s'], src['kvpr'] = ns_w, ns_s, ns_kvpr
                    
                    dst['items'].append(item)
                    dst['w'], dst['s'], dst['kvpr'] = nd_w, nd_s, nd_kvpr
                    
                    improved = True
                    break
            if improved: break
            
        if improved: continue
        
        # Strategy B: Swap item in Src <-> item in Dst
        for i, item1 in enumerate(src['items']):
            for dst_idx, dst in enumerate(gpu_states):
                if dst_idx == src_idx: continue
                # Don't swap with another bottleneck
                if dst['kvpr'] >= max_kvpr - 1e-9: continue
                
                for j, item2 in enumerate(dst['items']):
                    # Check Capacities
                    ns_s = src['s'] - item1['s'] + item2['s']
                    nd_s = dst['s'] - item2['s'] + item1['s']
                    if ns_s > GPU_MEM_SIZE or nd_s > GPU_MEM_SIZE: continue
                    
                    # Check KVPRs
                    ns_w = src['w'] - item1['w'] + item2['w']
                    nd_w = dst['w'] - item2['w'] + item1['w']
                    
                    ns_kvpr = calc_kvpr(ns_w, ns_s)
                    nd_kvpr = calc_kvpr(nd_w, nd_s)
                    
                    # Both new pressures must be better than current max
                    if max(ns_kvpr, nd_kvpr) < max_kvpr - 1e-9:
                        # Apply Swap
                        src['items'][i] = item2
                        src['w'], src['s'], src['kvpr'] = ns_w, ns_s, ns_kvpr
                        
                        dst['items'][j] = item1
                        dst['w'], dst['s'], dst['kvpr'] = nd_w, nd_s, nd_kvpr
                        
                        improved = True
                        break
                if improved: break
            if improved: break
            
        if not improved: break

    # Format Output
    return {i: [x['model'] for x in gpu_states[i]['items']] for i in range(gpu_num)}
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

