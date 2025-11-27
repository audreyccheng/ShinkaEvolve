# EVOLVE-BLOCK-START
"""Model placement algorithm for minimizing maximum KV cache pressure across GPUs"""

import random
import math

GPU_MEM_SIZE = 80.0  # GB
EPS = 1e-9

def compute_model_placement(gpu_num, models):
    """
    Minimizes max KVPR using Binary Search with Adaptive Re-ordering and Local Search.
    
    Key Improvements:
    1. Adaptive Packing: If packing fails on a specific item, the algorithm 'learns' 
       by prioritizing that item in the next attempt (moving it to the front).
    2. Linearized Bin Packing: Treats the problem as packing items of size (w + K*s) 
       into capacity (K*C).
    3. Robust Local Search: Hill Climbing with 1-move and 1-swap neighborhoods to 
       refine the solution found by binary search.
    """

    # --- Preprocessing ---
    items = []
    total_w = 0.0
    total_s = 0.0
    max_single_kvpr = 0.0

    for i, m in enumerate(models):
        w = m.req_rate / m.slo
        s = m.model_size
        items.append({'model': m, 'w': w, 's': s, 'id': i})
        total_w += w
        total_s += s
        
        rem = GPU_MEM_SIZE - s
        if rem > EPS:
            max_single_kvpr = max(max_single_kvpr, w / rem)
        elif w > EPS:
            max_single_kvpr = float('inf')

    # Lower Bound Calculation
    # Global capacity constraint: Sum(w) / (NumGPUs * Cap - Sum(s)) <= K
    rem_global = (gpu_num * GPU_MEM_SIZE) - total_s
    lb = total_w / rem_global if rem_global > EPS else (float('inf') if total_w > EPS else 0.0)
    lb = max(lb, max_single_kvpr)

    def calc_kvpr(w, s):
        rem = GPU_MEM_SIZE - s
        if rem <= EPS:
            return float('inf') if w > EPS else 0.0
        return w / rem

    def get_max_kvpr(placement_list):
        mx = 0.0
        for p in placement_list:
            w = sum(x['w'] for x in p)
            s = sum(x['s'] for x in p)
            mx = max(mx, calc_kvpr(w, s))
        return mx

    # --- Packing Logic ---
    def try_pack(k_target, item_order):
        """
        Attempts to pack items into gpu_num bins with KVPR <= k_target.
        Returns (success_bool, placement_list, failed_item_index)
        """
        # Capacity limit for linearized cost: w + K*s <= K*C
        limit = k_target * GPU_MEM_SIZE
        
        # Bins state: list of [current_w, current_s, items_list]
        bins = [[0.0, 0.0, []] for _ in range(gpu_num)]
        
        for idx, item in enumerate(item_order):
            item_lin = item['w'] + k_target * item['s']
            
            best_bin = -1
            best_fill = -1.0
            
            # 1. Greedy Choice: Best Fit
            for b_idx in range(gpu_num):
                b = bins[b_idx]
                
                # Check physical size
                if b[1] + item['s'] > GPU_MEM_SIZE: continue
                
                # Check linearized capacity
                curr_lin = b[0] + k_target * b[1]
                if curr_lin + item_lin > limit + 1e-5: continue
                
                # Best Fit: Maximize current usage (minimize slack)
                if curr_lin > best_fill:
                    best_fill = curr_lin
                    best_bin = b_idx
            
            if best_bin != -1:
                # Place item
                bins[best_bin][0] += item['w']
                bins[best_bin][1] += item['s']
                bins[best_bin][2].append(item)
            else:
                # 2. Repair: Try to swap with a placed item
                # Find a victim in some bin that:
                # a) if removed, allows 'item' to fit
                # b) fits into another bin
                
                repaired = False
                for b_idx in range(gpu_num):
                    b = bins[b_idx]
                    
                    # Optimization: Sort victims? No, just iterate.
                    for v_i, victim in enumerate(b[2]):
                        # Check if item fits in b without victim
                        if b[1] - victim['s'] + item['s'] > GPU_MEM_SIZE: continue
                        
                        lin_b_new = (b[0] - victim['w'] + item['w']) + k_target * (b[1] - victim['s'] + item['s'])
                        if lin_b_new > limit + 1e-5: continue
                        
                        # Check if victim fits elsewhere
                        victim_lin = victim['w'] + k_target * victim['s']
                        
                        for target_b_idx in range(gpu_num):
                            if b_idx == target_b_idx: continue
                            tb = bins[target_b_idx]
                            
                            if tb[1] + victim['s'] > GPU_MEM_SIZE: continue
                            lin_tb_new = (tb[0] + k_target * tb[1]) + victim_lin
                            
                            if lin_tb_new <= limit + 1e-5:
                                # Execute Swap
                                # Remove victim from b, add item to b
                                b[2][v_i] = item
                                b[0] += (item['w'] - victim['w'])
                                b[1] += (item['s'] - victim['s'])
                                
                                # Add victim to tb
                                tb[2].append(victim)
                                tb[0] += victim['w']
                                tb[1] += victim['s']
                                
                                repaired = True
                                break
                        if repaired: break
                    if repaired: break
                
                if not repaired:
                    return False, None, idx

        return True, [b[2] for b in bins], -1

    def solve_check_adaptive(k_target):
        """
        Checks feasibility of k_target.
        Uses adaptive re-ordering: if packing fails, prioritize the failed item.
        """
        # Strategy 1: Linearized Cost Descending
        current_order = sorted(items, key=lambda x: x['w'] + k_target * x['s'], reverse=True)
        
        # Try up to 10 adaptive attempts
        for _ in range(10):
            success, placement, fail_idx = try_pack(k_target, current_order)
            if success:
                return placement
            
            # Adaptive step: Move failed item to front
            # This prioritizes the bottleneck item
            failed_item = current_order.pop(fail_idx)
            current_order.insert(0, failed_item)
            
        return None

    # --- Local Optimization (Hill Climbing) ---
    def optimize_placement(placement_list):
        # State: list of dicts for fast access
        state = []
        for p in placement_list:
            w = sum(x['w'] for x in p)
            s = sum(x['s'] for x in p)
            state.append({'w': w, 's': s, 'items': list(p), 'k': calc_kvpr(w, s)})
        
        # Iteratively improve
        max_iters = 100
        for _ in range(max_iters):
            # Sort bins by KVPR to find bottleneck
            state.sort(key=lambda x: x['k'], reverse=True)
            src = state[0]
            current_max = src['k']
            
            if current_max <= EPS: break
            
            best_move = None # (type, improvement, args)
            
            # Scan moves from src
            for i, item in enumerate(src['items']):
                # Predict src improvement
                ns_rem = GPU_MEM_SIZE - (src['s'] - item['s'])
                if ns_rem <= EPS: continue 
                ns_k = (src['w'] - item['w']) / ns_rem
                
                if ns_k >= current_max - EPS: continue
                
                for dst_idx in range(1, gpu_num):
                    dst = state[dst_idx]
                    if dst['s'] + item['s'] > GPU_MEM_SIZE: continue
                    
                    nd_k = calc_kvpr(dst['w'] + item['w'], dst['s'] + item['s'])
                    
                    new_max = max(ns_k, nd_k)
                    if new_max < current_max - EPS:
                        # Found valid move. Is it steepest? 
                        # We just take first good one (Greedy) or best?
                        # Greedy is faster and usually sufficient.
                        # Let's verify if we can do better than just "better than current".
                        # We accept the move immediately to pivot faster.
                        src['items'].pop(i)
                        dst['items'].append(item)
                        
                        # Update state
                        src['w'] -= item['w']; src['s'] -= item['s']
                        src['k'] = ns_k
                        dst['w'] += item['w']; dst['s'] += item['s']
                        dst['k'] = nd_k
                        
                        best_move = True
                        break
                if best_move: break
            
            if best_move: continue
            
            # Scan swaps from src
            for i, item1 in enumerate(src['items']):
                for dst_idx in range(1, gpu_num):
                    dst = state[dst_idx]
                    if dst['k'] > current_max * 0.95: continue
                    
                    for j, item2 in enumerate(dst['items']):
                        # Check size
                        ns_s = src['s'] - item1['s'] + item2['s']
                        nd_s = dst['s'] - item2['s'] + item1['s']
                        if ns_s > GPU_MEM_SIZE or nd_s > GPU_MEM_SIZE: continue
                        
                        # Check KVPR
                        ns_k = calc_kvpr(src['w'] - item1['w'] + item2['w'], ns_s)
                        nd_k = calc_kvpr(dst['w'] - item2['w'] + item1['w'], nd_s)
                        
                        new_max = max(ns_k, nd_k)
                        if new_max < current_max - EPS:
                            # Execute Swap
                            src['items'][i] = item2
                            dst['items'][j] = item1
                            
                            src['w'] = src['w'] - item1['w'] + item2['w']
                            src['s'] = ns_s
                            src['k'] = ns_k
                            
                            dst['w'] = dst['w'] - item2['w'] + item1['w']
                            dst['s'] = nd_s
                            dst['k'] = nd_k
                            
                            best_move = True
                            break
                    if best_move: break
                if best_move: break
            
            if not best_move: break
            
        return [b['items'] for b in state]

    # --- Binary Search ---
    high = 1e9
    
    # 1. Establish initial feasible solution
    # Try with high K
    res = solve_check_adaptive(high)
    if not res:
        raise ValueError("Unable to place models.")
    
    # Optimize to get tight bound
    res = optimize_placement(res)
    best_placement = res
    high = min(high, get_max_kvpr(res))
    low = lb
    
    # 2. Search
    for _ in range(25):
        if high - low < 1e-4: break
        
        mid = (low + high) / 2
        
        # Check feasibility
        mid_res = solve_check_adaptive(mid)
        
        if mid_res:
            # Found a solution. Optimize it.
            mid_res = optimize_placement(mid_res)
            mid_max = get_max_kvpr(mid_res)
            
            if mid_max < get_max_kvpr(best_placement):
                best_placement = mid_res
            
            # Tighten upper bound
            high = min(mid, mid_max)
        else:
            low = mid
            
    # Final Optimization to be sure
    best_placement = optimize_placement(best_placement)
    
    # Convert to output format
    return {i: [x['model'] for x in p] for i, p in enumerate(best_placement)}
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
