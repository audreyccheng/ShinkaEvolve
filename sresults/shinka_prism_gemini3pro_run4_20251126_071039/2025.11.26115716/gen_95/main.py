# EVOLVE-BLOCK-START
"""Model placement algorithm for minimizing maximum KV cache pressure across GPUs"""

import math
import random

GPU_MEM_SIZE = 80.0

def compute_model_placement(gpu_num, models):
    """
    Compute a model placement that minimizes the maximum KVPR across all GPUs.
    
    Algorithm:
    1. Binary Search for optimal max KVPR (K).
    2. Feasibility Check with Failure-Driven Prioritization:
       - Uses Best Fit Decreasing on linearized cost w + K*s.
       - If packing fails, unplaced items are moved to the front (prioritized) and retried.
       - This adapts the sorting order to the specific 'hard' items of the dataset.
    3. Steepest Descent Hill Climbing:
       - Post-optimization that targets the bottleneck GPU.
       - Moves or swaps models to strictly reduce the maximum KVPR.
    """

    # 1. Preprocessing
    items = []
    for m in models:
        items.append({
            'model': m,
            'w': m.req_rate / m.slo,
            's': m.model_size
        })

    # Helper: Safe KVPR calculation
    def calc_kvpr(w, s):
        rem = GPU_MEM_SIZE - s
        if rem <= 1e-9:
            return 1e15 if w > 1e-9 else 0.0
        return w / rem

    def get_max_kvpr(placement):
        mx = 0.0
        for p in placement.values():
            w = sum(m.req_rate / m.slo for m in p)
            s = sum(m.model_size for m in p)
            mx = max(mx, calc_kvpr(w, s))
        return mx

    # 2. Optimization: Steepest Descent on Bottleneck
    def optimize_placement(placement):
        # Convert to mutable state
        state = []
        for i in range(gpu_num):
            p = placement[i]
            w = sum(m.req_rate / m.slo for m in p)
            s = sum(m.model_size for m in p)
            state.append({'w': w, 's': s, 'models': list(p)})

        # Cache KVPR values to avoid recomputation
        gpu_k = [calc_kvpr(st['w'], st['s']) for st in state]

        for _ in range(150): # Iteration limit
            # Find the global bottleneck
            max_val = -1.0
            src_idx = -1
            for i, k in enumerate(gpu_k):
                if k > max_val:
                    max_val = k
                    src_idx = i
            
            if max_val <= 1e-9: break
            
            src = state[src_idx]
            improved = False
            
            # Action: Move (Try to offload a model from bottleneck)
            for i, m in enumerate(src['models']):
                m_w = m.req_rate / m.slo
                m_s = m.model_size
                
                # State of Src after removal
                ns_w = src['w'] - m_w
                ns_s = src['s'] - m_s
                ns_k = calc_kvpr(ns_w, ns_s)
                
                # Pruning: Only proceed if src improves significantly
                if ns_k >= max_val - 1e-9: continue
                
                for dst_idx in range(gpu_num):
                    if dst_idx == src_idx: continue
                    dst = state[dst_idx]
                    
                    if dst['s'] + m_s > GPU_MEM_SIZE: continue
                    
                    nd_w = dst['w'] + m_w
                    nd_s = dst['s'] + m_s
                    nd_k = calc_kvpr(nd_w, nd_s)
                    
                    # Acceptance: The new local max between these two must be better than old global max
                    if max(ns_k, nd_k) < max_val - 1e-9:
                        # Apply Move
                        src['models'].pop(i)
                        src['w'], src['s'] = ns_w, ns_s
                        gpu_k[src_idx] = ns_k
                        
                        dst['models'].append(m)
                        dst['w'], dst['s'] = nd_w, nd_s
                        gpu_k[dst_idx] = nd_k
                        
                        improved = True
                        break
                if improved: break
            
            if improved: continue
            
            # Action: Swap (Exchange models to balance load)
            for i, m1 in enumerate(src['models']):
                m1_w = m1.req_rate / m1.slo
                m1_s = m1.model_size
                
                for dst_idx in range(gpu_num):
                    if dst_idx == src_idx: continue
                    if gpu_k[dst_idx] > max_val * 0.95: continue # Skip if dst is also stressed
                    
                    dst = state[dst_idx]
                    
                    for j, m2 in enumerate(dst['models']):
                        m2_w = m2.req_rate / m2.slo
                        m2_s = m2.model_size
                        
                        # New Src
                        ns_s = src['s'] - m1_s + m2_s
                        if ns_s > GPU_MEM_SIZE: continue
                        ns_w = src['w'] - m1_w + m2_w
                        
                        # New Dst
                        nd_s = dst['s'] - m2_s + m1_s
                        if nd_s > GPU_MEM_SIZE: continue
                        nd_w = dst['w'] - m2_w + m1_w
                        
                        ns_k = calc_kvpr(ns_w, ns_s)
                        nd_k = calc_kvpr(nd_w, nd_s)
                        
                        if max(ns_k, nd_k) < max_val - 1e-9:
                            # Apply Swap
                            src['models'][i] = m2
                            src['w'], src['s'] = ns_w, ns_s
                            gpu_k[src_idx] = ns_k
                            
                            dst['models'][j] = m1
                            dst['w'], dst['s'] = nd_w, nd_s
                            gpu_k[dst_idx] = nd_k
                            
                            improved = True
                            break
                    if improved: break
                if improved: break
            
            if not improved: break # Converged or local optimum
            
        return {i: state[i]['models'] for i in range(gpu_num)}

    # 3. Feasibility Check with Failure-Driven Retry
    def check_feasibility(k_target):
        
        def try_pack(ordered_items):
            # Best Fit Heuristic
            bins = [{'w': 0.0, 's': 0.0, 'items': []} for _ in range(gpu_num)]
            unplaced = []
            
            for item in ordered_items:
                w, s = item['w'], item['s']
                
                best_idx = -1
                best_score = -1.0
                
                for i in range(gpu_num):
                    b = bins[i]
                    if b['s'] + s > GPU_MEM_SIZE: continue
                    
                    # KVPR Constraint: w_new <= k * (C - s_new)
                    rem = GPU_MEM_SIZE - (b['s'] + s)
                    if rem <= 1e-9:
                        if (b['w'] + w) > 1e-9: continue
                    elif (b['w'] + w) > k_target * rem + 1e-7:
                        continue
                        
                    # Best Fit Score: Maximize Linearized Load
                    # This fills the "available capacity" defined by K most effectively
                    score = (b['w'] + w) + k_target * (b['s'] + s)
                    if score > best_score:
                        best_score = score
                        best_idx = i
                
                if best_idx != -1:
                    bins[best_idx]['w'] += w
                    bins[best_idx]['s'] += s
                    bins[best_idx]['items'].append(item['model'])
                else:
                    unplaced.append(item)
            
            if not unplaced:
                return {i: bins[i]['items'] for i in range(gpu_num)}, None
            return None, unplaced

        # Base Heuristic: Linearized Cost Descending
        base_key = lambda x: x['w'] + k_target * x['s']
        current_order = sorted(items, key=base_key, reverse=True)
        
        # Failure-Driven Loop
        rng = random.Random(42 + int(k_target))
        for _ in range(20): # Try up to 20 re-orderings
            res, unplaced = try_pack(current_order)
            if res: return res
            
            # Key Innovation: Prioritize failures
            # Move unplaced items to front, shuffling them to find a mutual fit
            rng.shuffle(unplaced)
            
            # Reconstruct order: Unplaced (Priority) + Placed (Relative Order Preserved)
            unplaced_ids = set(id(x) for x in unplaced)
            placed = [x for x in current_order if id(x) not in unplaced_ids]
            current_order = unplaced + placed
            
        return None

    # 4. Binary Search Driver
    high = 1e9
    
    # Initial valid solution
    best_placement = check_feasibility(high)
    if not best_placement:
        raise ValueError("Unable to place models on GPUs.")
    
    # Optimize initial solution
    best_placement = optimize_placement(best_placement)
    high = get_max_kvpr(best_placement)
    low = 0.0

    for _ in range(25):
        mid = (low + high) / 2
        res = check_feasibility(mid)
        if res:
            # Optimize feasible solution to see if it beats current best
            res = optimize_placement(res)
            mx = get_max_kvpr(res)
            
            if mx < get_max_kvpr(best_placement):
                best_placement = res
            
            high = min(mid, mx)
        else:
            low = mid

    # Final polish
    return optimize_placement(best_placement)
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
