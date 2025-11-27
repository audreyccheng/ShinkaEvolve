# EVOLVE-BLOCK-START
"""Model placement algorithm for minimizing maximum KV cache pressure across GPUs"""

import random

GPU_MEM_SIZE = 80  # GB

def compute_model_placement(gpu_num, models):
    """
    Compute a model placement that minimizes the maximum KVPR across all GPUs.
    
    Approach:
    1. Lower Bound Calculation: Estimate theoretical min K to narrow search space.
    2. Binary Search for K:
       - Uses a 'Pack with Repair' check function.
       - Heuristics: Best-Fit Decreasing on Linearized Cost (W + K*S), Size, and Weight.
       - Repair: If an item fails to fit, attempt to swap it with an item in a bin that allows both to fit.
    3. In-Loop Optimization: When a feasible K is found, optimize the placement (Hill Climbing)
       to find the *actual* max KVPR, allowing the binary search upper bound to be tightened aggressively.
    """

    # --- 1. Preprocessing & Lower Bound ---
    items = []
    total_w = 0.0
    total_s = 0.0
    max_single_kvpr = 0.0

    for m in models:
        w = m.req_rate / m.slo
        s = m.model_size
        items.append({'model': m, 'w': w, 's': s})
        total_w += w
        total_s += s
        
        rem = GPU_MEM_SIZE - s
        if rem > 1e-9:
            max_single_kvpr = max(max_single_kvpr, w / rem)
        elif w > 1e-9:
            max_single_kvpr = float('inf')

    # Theoretical global lower bound (perfect distribution)
    # Sum(w) / (Num_GPU * MEM - Sum(s)) <= K
    total_mem_capacity = gpu_num * GPU_MEM_SIZE
    rem_global = total_mem_capacity - total_s
    
    lb = 0.0
    if rem_global > 1e-9:
        lb = total_w / rem_global
    
    # K must be at least the max required by any single large item alone
    lb = max(lb, max_single_kvpr)

    def get_max_kvpr(placement_list):
        """Calculates the actual maximum KVPR of a placement list (list of item lists)."""
        mx = 0.0
        for p in placement_list:
            w = sum(x['w'] for x in p)
            s = sum(x['s'] for x in p)
            rem = GPU_MEM_SIZE - s
            if rem <= 1e-9:
                if w > 1e-9: return float('inf')
                val = 0.0
            else:
                val = w / rem
            mx = max(mx, val)
        return mx

    def local_optimize(placement_list):
        """
        Hill climbing to improve a valid placement.
        Tries to move or swap items from the bottleneck GPU to others.
        """
        # Convert to state structure for fast updates
        # bins: [{'w', 's', 'items', 'kvpr'}]
        state = []
        for p in placement_list:
            w = sum(x['w'] for x in p)
            s = sum(x['s'] for x in p)
            rem = GPU_MEM_SIZE - s
            val = w / rem if rem > 1e-9 else (float('inf') if w > 1e-9 else 0.0)
            state.append({'w': w, 's': s, 'items': list(p), 'kvpr': val})
            
        def update_state(idx):
            b = state[idx]
            b['w'] = sum(x['w'] for x in b['items'])
            b['s'] = sum(x['s'] for x in b['items'])
            rem = GPU_MEM_SIZE - b['s']
            b['kvpr'] = b['w'] / rem if rem > 1e-9 else (float('inf') if b['w'] > 1e-9 else 0.0)

        # Optimization loop
        for _ in range(150): # Cap iterations
            # Sort bins by KVPR descending
            indices = sorted(range(gpu_num), key=lambda i: state[i]['kvpr'], reverse=True)
            src_idx = indices[0]
            current_max = state[src_idx]['kvpr']
            
            if current_max <= 1e-9: break # Can't improve 0
            
            src = state[src_idx]
            improved = False
            
            # 1. Try Move (Src -> Dst)
            for i, item in enumerate(src['items']):
                # Predict new src KVPR
                ns_rem = GPU_MEM_SIZE - (src['s'] - item['s'])
                if ns_rem <= 1e-9: continue # Should not happen if it fit before
                ns_kvpr = (src['w'] - item['w']) / ns_rem
                
                # Pruning: Don't move if source doesn't improve enough or stays bottleneck
                if ns_kvpr >= current_max - 1e-9: continue

                for dst_idx in indices[1:]:
                    dst = state[dst_idx]
                    # Check fits
                    if dst['s'] + item['s'] > GPU_MEM_SIZE: continue
                    
                    nd_rem = GPU_MEM_SIZE - (dst['s'] + item['s'])
                    if nd_rem <= 1e-9: continue
                    nd_kvpr = (dst['w'] + item['w']) / nd_rem
                    
                    # Check if move improves overall state (bottleneck reduces)
                    if nd_kvpr < current_max - 1e-9:
                        # Do move
                        mov = src['items'].pop(i)
                        dst['items'].append(mov)
                        update_state(src_idx)
                        update_state(dst_idx)
                        improved = True
                        break
                if improved: break
            
            if improved: continue

            # 2. Try Swap (Src <-> Dst)
            for i, item1 in enumerate(src['items']):
                for dst_idx in indices[1:]:
                    dst = state[dst_idx]
                    # Pruning: Don't swap with a bin that is nearly as full
                    if dst['kvpr'] > current_max * 0.99: continue
                    
                    for j, item2 in enumerate(dst['items']):
                        # Check Capacity
                        ns_s = src['s'] - item1['s'] + item2['s']
                        nd_s = dst['s'] - item2['s'] + item1['s']
                        if ns_s > GPU_MEM_SIZE or nd_s > GPU_MEM_SIZE: continue
                        
                        # Check KVPR
                        ns_rem = GPU_MEM_SIZE - ns_s
                        nd_rem = GPU_MEM_SIZE - nd_s
                        if ns_rem <= 1e-9 or nd_rem <= 1e-9: continue
                        
                        ns_kvpr = (src['w'] - item1['w'] + item2['w']) / ns_rem
                        nd_kvpr = (dst['w'] - item2['w'] + item1['w']) / nd_rem
                        
                        new_local_max = max(ns_kvpr, nd_kvpr)
                        if new_local_max < current_max - 1e-9:
                            # Do swap
                            src['items'][i] = item2
                            dst['items'][j] = item1
                            update_state(src_idx)
                            update_state(dst_idx)
                            improved = True
                            break
                    if improved: break
                if improved: break
            
            if not improved: break
            
        return [b['items'] for b in state]

    def solve_check(k_target, attempt_limit=5):
        """
        Checks if placement is possible with max KVPR <= k_target.
        Returns placement list if success, None otherwise.
        """
        limit_val = k_target * GPU_MEM_SIZE
        
        def pack(ordered_items):
            bins = [{'w': 0.0, 's': 0.0, 'items': []} for _ in range(gpu_num)]
            
            for item in ordered_items:
                item_lin = item['w'] + k_target * item['s']
                
                best_idx = -1
                best_fill = -1.0
                
                # Try to place in existing bins
                for i in range(gpu_num):
                    b = bins[i]
                    if b['s'] + item['s'] > GPU_MEM_SIZE: continue
                    
                    # Linear constraint check: sum(w) + k*sum(s) <= k*MEM
                    current_lin = b['w'] + k_target * b['s']
                    if current_lin + item_lin > limit_val + 1e-5: continue
                    
                    # Best Fit Heuristic: Maximize fill (tight packing)
                    if current_lin > best_fill:
                        best_fill = current_lin
                        best_idx = i
                        
                if best_idx != -1:
                    bins[best_idx]['items'].append(item)
                    bins[best_idx]['w'] += item['w']
                    bins[best_idx]['s'] += item['s']
                else:
                    # Repair Strategy: Swap with a victim
                    repaired = False
                    for i in range(gpu_num):
                        b = bins[i]
                        for v_idx, victim in enumerate(b['items']):
                            # Proposed: item -> b, victim -> elsewhere
                            
                            # 1. Can item fit in b replacing victim?
                            if b['s'] - victim['s'] + item['s'] > GPU_MEM_SIZE: continue
                            new_lin_b = (b['w'] - victim['w'] + item['w']) + k_target * (b['s'] - victim['s'] + item['s'])
                            if new_lin_b > limit_val + 1e-5: continue
                            
                            # 2. Can victim fit elsewhere?
                            victim_lin = victim['w'] + k_target * victim['s']
                            for k in range(gpu_num):
                                if i == k: continue
                                bk = bins[k]
                                if bk['s'] + victim['s'] > GPU_MEM_SIZE: continue
                                if (bk['w'] + k_target * bk['s']) + victim_lin <= limit_val + 1e-5:
                                    # Execute Swap
                                    b['items'][v_idx] = item
                                    b['w'] += (item['w'] - victim['w'])
                                    b['s'] += (item['s'] - victim['s'])
                                    
                                    bk['items'].append(victim)
                                    bk['w'] += victim['w']
                                    bk['s'] += victim['s']
                                    repaired = True
                                    break
                            if repaired: break
                        if repaired: break
                    
                    if not repaired:
                        return None
                        
            return [b['items'] for b in bins]

        # 1. Deterministic Strategies
        # Sort by Linear Cost Descending (Primary for this constraint)
        res = pack(sorted(items, key=lambda x: x['w'] + k_target * x['s'], reverse=True))
        if res: return res
        
        # Sort by Size Descending (Standard Bin Packing)
        res = pack(sorted(items, key=lambda x: x['s'], reverse=True))
        if res: return res
        
        # Sort by Weight Descending (Rate intensive)
        res = pack(sorted(items, key=lambda x: x['w'], reverse=True))
        if res: return res

        # 2. Randomized Strategies (Noisy Sort)
        if attempt_limit > 0:
            rng = random.Random(42 + int(k_target))
            base_key = lambda x: x['w'] + k_target * x['s']
            for _ in range(attempt_limit):
                # Multiplicative noise
                noisy_items = sorted(items, key=lambda x: base_key(x) * rng.uniform(0.95, 1.05), reverse=True)
                res = pack(noisy_items)
                if res: return res
        
        return None

    # --- 2. Binary Search Loop ---
    
    # Initialize high bound
    high = 1e9
    
    # Check feasibility with loose constraint
    # Try deterministic first, then randomized if needed
    init_res = solve_check(high, 0)
    if not init_res:
        init_res = solve_check(high, 20)
        if not init_res:
            raise ValueError("Unable to place models (insufficient total memory).")
    
    # Initial Optimization
    best_pl_list = local_optimize(init_res)
    high = min(high, get_max_kvpr(best_pl_list))
    low = lb
    
    # Search
    for _ in range(25):
        if high - low < 1e-4: break
        
        mid = (low + high) / 2
        # Try to find placement for K=mid
        res = solve_check(mid, attempt_limit=5)
        
        if res:
            # Feasible. Optimize it to see how good it really is.
            res = local_optimize(res)
            actual_max = get_max_kvpr(res)
            
            # Store best
            if actual_max < get_max_kvpr(best_pl_list):
                 best_pl_list = res
            
            # Update high bound aggressively
            high = min(mid, actual_max)
        else:
            low = mid
            
    # Final cleanup optimization
    best_pl_list = local_optimize(best_pl_list)
    
    return {i: [x['model'] for x in p] for i, p in enumerate(best_pl_list)}
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
