# EVOLVE-BLOCK-START
"""Model placement algorithm for minimizing maximum KV cache pressure across GPUs"""

import random

GPU_MEM_SIZE = 80  # GB

def compute_model_placement(gpu_num, models):
    """
    Compute a model placement that minimizes the maximum KVPR across all GPUs.
    Uses Binary Search with Multi-Heuristic Packing and Ejection Chain Local Search.
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
            max_single_kvpr = 1e9

    # Theoretical global lower bound
    lb = 0.0
    rem_global = gpu_num * GPU_MEM_SIZE - total_s
    if rem_global > 1e-9:
        lb = total_w / rem_global
    lb = max(lb, max_single_kvpr)

    # --- Helpers ---
    def calc_kvpr(w, s):
        rem = GPU_MEM_SIZE - s
        if rem <= 1e-9:
            return float('inf') if w > 1e-9 else 0.0
        return w / rem

    def get_max_kvpr(state):
        mx = 0.0
        for b in state:
            val = calc_kvpr(b['w'], b['s'])
            if val > mx: mx = val
        return mx

    # --- 2. Advanced Local Search (Ejection Chains) ---
    def hill_climb(state, max_iter=150):
        # state: list of dicts {'w', 's', 'items': []}
        
        for _ in range(max_iter):
            # Identify bottleneck
            # Compute KVPRs fresh
            for b in state:
                b['kvpr'] = calc_kvpr(b['w'], b['s'])
                
            indices = sorted(range(gpu_num), key=lambda i: state[i]['kvpr'], reverse=True)
            src_idx = indices[0]
            current_max = state[src_idx]['kvpr']
            
            if current_max <= 1e-9: break
            
            src = state[src_idx]
            improved = False
            
            # Iterate through items in bottleneck bin
            for i, item in enumerate(src['items']):
                # Proposed Src State
                src_ns_w = src['w'] - item['w']
                src_ns_s = src['s'] - item['s']
                src_ns_kvpr = calc_kvpr(src_ns_w, src_ns_s)
                
                # Heuristic: Only proceed if source improves
                if src_ns_kvpr >= current_max - 1e-9: continue
                
                # Try targets
                for dst_idx in indices[1:]:
                    dst = state[dst_idx]
                    
                    # A. Direct Move: Src -> Dst
                    if dst['s'] + item['s'] <= GPU_MEM_SIZE:
                        dst_ns_w = dst['w'] + item['w']
                        dst_ns_s = dst['s'] + item['s']
                        dst_ns_kvpr = calc_kvpr(dst_ns_w, dst_ns_s)
                        
                        if dst_ns_kvpr < current_max - 1e-9:
                            # Apply Move
                            src['items'].pop(i)
                            src['w'], src['s'], src['kvpr'] = src_ns_w, src_ns_s, src_ns_kvpr
                            dst['items'].append(item)
                            dst['w'], dst['s'], dst['kvpr'] = dst_ns_w, dst_ns_s, dst_ns_kvpr
                            improved = True
                            break
                    if improved: break
                    
                    # B. Swap / 3-Cycle: Src -> Dst, Dst -> ???
                    # We need to eject 'victim' from Dst to make room for 'item'
                    for j, victim in enumerate(dst['items']):
                        # Intermediate Dst state (Dst + item - victim)
                        inter_dst_s = dst['s'] + item['s'] - victim['s']
                        if inter_dst_s > GPU_MEM_SIZE: continue
                        
                        inter_dst_w = dst['w'] + item['w'] - victim['w']
                        inter_dst_kvpr = calc_kvpr(inter_dst_w, inter_dst_s)
                        
                        # Intermediate Dst must be valid and better than current_max
                        if inter_dst_kvpr >= current_max - 1e-9: continue
                        
                        # Now place victim
                        
                        # B1. Swap: Victim -> Src
                        if src_ns_s + victim['s'] <= GPU_MEM_SIZE:
                            swap_src_w = src_ns_w + victim['w']
                            swap_src_s = src_ns_s + victim['s']
                            swap_src_kvpr = calc_kvpr(swap_src_w, swap_src_s)
                            
                            if swap_src_kvpr < current_max - 1e-9:
                                # Apply Swap
                                src['items'][i] = victim
                                src['w'], src['s'], src['kvpr'] = swap_src_w, swap_src_s, swap_src_kvpr
                                
                                dst['items'][j] = item
                                dst['w'], dst['s'], dst['kvpr'] = inter_dst_w, inter_dst_s, inter_dst_kvpr
                                improved = True
                                break
                        
                        # B2. 3-Cycle: Victim -> Third Bin
                        for third_idx in indices[1:]:
                            if third_idx == dst_idx: continue
                            third = state[third_idx]
                            
                            if third['s'] + victim['s'] <= GPU_MEM_SIZE:
                                third_ns_w = third['w'] + victim['w']
                                third_ns_s = third['s'] + victim['s']
                                third_ns_kvpr = calc_kvpr(third_ns_w, third_ns_s)
                                
                                if third_ns_kvpr < current_max - 1e-9:
                                    # Apply 3-Cycle
                                    # 1. Remove item from Src
                                    src['items'].pop(i)
                                    src['w'], src['s'], src['kvpr'] = src_ns_w, src_ns_s, src_ns_kvpr
                                    
                                    # 2. Swap item into Dst, get victim out
                                    dst['items'][j] = item
                                    dst['w'], dst['s'], dst['kvpr'] = inter_dst_w, inter_dst_s, inter_dst_kvpr
                                    
                                    # 3. Add victim to Third
                                    third['items'].append(victim)
                                    third['w'], third['s'], third['kvpr'] = third_ns_w, third_ns_s, third_ns_kvpr
                                    
                                    improved = True
                                    break
                            if improved: break
                        if improved: break
                    if improved: break
                if improved: break
            
            if not improved: break
            
        return state

    # --- 3. Packing Logic with Repair ---
    def pack(k_target, strategies):
        limit_val = k_target * GPU_MEM_SIZE
        
        for key_fn in strategies:
            # Sort items
            ordered_items = sorted(items, key=key_fn, reverse=True)
            
            # Init Bins
            bins = [{'w': 0.0, 's': 0.0, 'items': []} for _ in range(gpu_num)]
            possible = True
            
            for item in ordered_items:
                item_lin = item['w'] + k_target * item['s']
                
                best_idx = -1
                best_slack = -1.0
                
                # Try Best Fit
                for i in range(gpu_num):
                    b = bins[i]
                    if b['s'] + item['s'] > GPU_MEM_SIZE: continue
                    
                    curr_lin = b['w'] + k_target * b['s']
                    if curr_lin + item_lin > limit_val + 1e-5: continue
                    
                    # Metric: Maximize current fill (Minimize slack)
                    if curr_lin > best_slack:
                        best_slack = curr_lin
                        best_idx = i
                
                if best_idx != -1:
                    bins[best_idx]['items'].append(item)
                    bins[best_idx]['w'] += item['w']
                    bins[best_idx]['s'] += item['s']
                else:
                    # Repair: Swap
                    repaired = False
                    for i in range(gpu_num):
                        b = bins[i]
                        for v_idx, victim in enumerate(b['items']):
                            # Can we put item in b replacing victim?
                            if b['s'] - victim['s'] + item['s'] > GPU_MEM_SIZE: continue
                            new_lin_b = (b['w'] - victim['w'] + item['w']) + k_target * (b['s'] - victim['s'] + item['s'])
                            if new_lin_b > limit_val + 1e-5: continue
                            
                            # Can victim go elsewhere?
                            victim_lin = victim['w'] + k_target * victim['s']
                            for k in range(gpu_num):
                                if i == k: continue
                                bk = bins[k]
                                if bk['s'] + victim['s'] > GPU_MEM_SIZE: continue
                                new_lin_k = (bk['w'] + k_target * bk['s']) + victim_lin
                                if new_lin_k <= limit_val + 1e-5:
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
                        possible = False
                        break
            
            if possible:
                return bins
        return None

    # --- 4. Main Algorithm Flow ---
    
    # Define strategy generator
    def get_strategies(k):
        return [
            lambda x: x['w'] + k * x['s'],       # Linearized Cost
            lambda x: x['s'],                    # Size
            lambda x: x['w'],                    # Weight
            lambda x: x['w'] / (x['s'] + 1e-9),  # Density
        ]
    
    # Initial Solution
    high = 1e9
    
    # Try deterministic packing
    init_state = pack(high, get_strategies(high))
    
    # If failed, try randomized
    if not init_state:
        rng = random.Random(42)
        # Noisy Linear
        strats = [lambda x: (x['w'] + high * x['s']) * rng.uniform(0.9, 1.1) for _ in range(20)]
        init_state = pack(high, strats)
        
    if not init_state:
        raise ValueError("Unable to place models.")
        
    # Optimize initial
    init_state = hill_climb(init_state)
    best_state = init_state
    high = get_max_kvpr(best_state)
    low = lb
    
    # Binary Search
    for _ in range(25):
        if high - low < 1e-4: break
        mid = (low + high) / 2
        
        # Determine strategies
        strats = get_strategies(mid)
        # Add a few noisy ones
        rng = random.Random(int(mid * 100))
        for _ in range(5):
            strats.append(lambda x: (x['w'] + mid * x['s']) * rng.uniform(0.9, 1.1))
            
        res = pack(mid, strats)
        
        if res:
            # Feasible K found.
            # Optimize to see true KVPR
            res = hill_climb(res, max_iter=50)
            res_max = get_max_kvpr(res)
            
            if res_max < get_max_kvpr(best_state):
                best_state = res
            
            high = min(mid, res_max)
        else:
            low = mid
            
    # Final Optimization
    best_state = hill_climb(best_state, max_iter=200)
    
    # Format Output
    return {i: [x['model'] for x in b['items']] for i in range(gpu_num)}
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