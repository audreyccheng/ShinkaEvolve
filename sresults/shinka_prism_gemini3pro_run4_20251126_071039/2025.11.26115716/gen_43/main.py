# EVOLVE-BLOCK-START
"""Model placement algorithm for minimizing maximum KV cache pressure across GPUs"""

import random

GPU_MEM_SIZE = 80  # GB

def compute_model_placement(gpu_num, models):
    """
    Compute a model placement that minimizes the maximum KVPR across all GPUs.
    
    Hybrid Approach:
    1. Preprocessing & Lower Bound: Calculates theoretical min K to narrow search.
    2. Feasibility Check (solve_check):
       - Uses 'Pack with Repair': Best-Fit Decreasing with repair (swaps).
       - Heuristics:
         - Deterministic: Linear Cost (W+K*S), Size (S), Weight (W), and Density (W/S).
         - Randomized: Mix of Noisy Sorting (perturbation) and Pure Shuffling (diversity).
    3. Binary Search: Finds optimal K.
    4. Local Optimization (Hill Climbing): Refines valid placements by moving/swapping 
       items from the bottleneck GPU to reduce actual KVPR.
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

    # Theoretical global lower bound
    total_mem_capacity = gpu_num * GPU_MEM_SIZE
    rem_global = total_mem_capacity - total_s
    
    lb = 0.0
    if rem_global > 1e-9:
        lb = total_w / rem_global
    
    # LB must be at least the max required by any single item
    lb = max(lb, max_single_kvpr)

    def get_max_kvpr(placement_list):
        """Calculates the actual maximum KVPR of a placement list."""
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
        Reduces the bottleneck GPU's KVPR by moving/swapping items.
        """
        # Convert to state structure
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
        for _ in range(150): 
            # Identify bottleneck
            indices = sorted(range(gpu_num), key=lambda i: state[i]['kvpr'], reverse=True)
            src_idx = indices[0]
            current_max = state[src_idx]['kvpr']
            
            if current_max <= 1e-9: break
            
            src = state[src_idx]
            improved = False
            
            # 1. Try Move (Src -> Dst)
            for i, item in enumerate(src['items']):
                # Pruning: Estimate improvement
                ns_rem = GPU_MEM_SIZE - (src['s'] - item['s'])
                if ns_rem <= 1e-9: continue 
                ns_kvpr = (src['w'] - item['w']) / ns_rem
                
                if ns_kvpr >= current_max - 1e-9: continue

                for dst_idx in indices[1:]:
                    dst = state[dst_idx]
                    if dst['s'] + item['s'] > GPU_MEM_SIZE: continue
                    
                    nd_rem = GPU_MEM_SIZE - (dst['s'] + item['s'])
                    if nd_rem <= 1e-9: continue
                    nd_kvpr = (dst['w'] + item['w']) / nd_rem
                    
                    if nd_kvpr < current_max - 1e-9:
                        # Move
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
                    if dst['kvpr'] > current_max * 0.95: continue # optimization
                    
                    for j, item2 in enumerate(dst['items']):
                        ns_s = src['s'] - item1['s'] + item2['s']
                        nd_s = dst['s'] - item2['s'] + item1['s']
                        if ns_s > GPU_MEM_SIZE or nd_s > GPU_MEM_SIZE: continue
                        
                        ns_rem = GPU_MEM_SIZE - ns_s
                        nd_rem = GPU_MEM_SIZE - nd_s
                        if ns_rem <= 1e-9 or nd_rem <= 1e-9: continue
                        
                        ns_kvpr = (src['w'] - item1['w'] + item2['w']) / ns_rem
                        nd_kvpr = (dst['w'] - item2['w'] + item1['w']) / nd_rem
                        
                        if max(ns_kvpr, nd_kvpr) < current_max - 1e-9:
                            # Swap
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
        Feasibility check for K.
        Returns placement list if feasible, else None.
        """
        limit_val = k_target * GPU_MEM_SIZE
        
        def pack(ordered_items):
            bins = [{'w': 0.0, 's': 0.0, 'items': []} for _ in range(gpu_num)]
            
            for item in ordered_items:
                item_lin = item['w'] + k_target * item['s']
                best_idx = -1
                best_fill = -1.0
                
                # Best Fit
                for i in range(gpu_num):
                    b = bins[i]
                    if b['s'] + item['s'] > GPU_MEM_SIZE: continue
                    
                    current_lin = b['w'] + k_target * b['s']
                    if current_lin + item_lin > limit_val + 1e-5: continue
                    
                    if current_lin > best_fill:
                        best_fill = current_lin
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
                            # Try replace victim in b with item
                            if b['s'] - victim['s'] + item['s'] > GPU_MEM_SIZE: continue
                            new_lin_b = (b['w'] - victim['w'] + item['w']) + k_target * (b['s'] - victim['s'] + item['s'])
                            if new_lin_b > limit_val + 1e-5: continue
                            
                            # Try place victim elsewhere
                            victim_lin = victim['w'] + k_target * victim['s']
                            for k in range(gpu_num):
                                if i == k: continue
                                bk = bins[k]
                                if bk['s'] + victim['s'] > GPU_MEM_SIZE: continue
                                if (bk['w'] + k_target * bk['s']) + victim_lin <= limit_val + 1e-5:
                                    # Swap
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
                    if not repaired: return None
            return [b['items'] for b in bins]

        # 1. Deterministic Strategies
        # Linear Cost
        res = pack(sorted(items, key=lambda x: x['w'] + k_target * x['s'], reverse=True))
        if res: return res
        # Size
        res = pack(sorted(items, key=lambda x: x['s'], reverse=True))
        if res: return res
        # Weight
        res = pack(sorted(items, key=lambda x: x['w'], reverse=True))
        if res: return res
        # Density (Weight/Size)
        res = pack(sorted(items, key=lambda x: x['w']/(x['s']+1e-9), reverse=True))
        if res: return res

        # 2. Randomized Strategies
        if attempt_limit > 0:
            rng = random.Random(42 + int(k_target))
            base_key = lambda x: x['w'] + k_target * x['s']
            indices = list(range(len(items)))
            
            for i in range(attempt_limit):
                if i % 2 == 0:
                    # Noisy Sort
                    noisy_items = sorted(items, key=lambda x: base_key(x) * rng.uniform(0.9, 1.1), reverse=True)
                    res = pack(noisy_items)
                else:
                    # Pure Shuffle
                    rng.shuffle(indices)
                    res = pack([items[j] for j in indices])
                    
                if res: return res
        return None

    # --- 3. Binary Search Loop ---
    high = 1e9
    
    # Initial Check
    init_res = solve_check(high, 0)
    if not init_res:
        init_res = solve_check(high, 25) # Slightly more attempts for robustness
        if not init_res:
            raise ValueError("Unable to place models (insufficient total memory).")
    
    best_pl_list = local_optimize(init_res)
    high = min(high, get_max_kvpr(best_pl_list))
    low = lb
    
    for _ in range(25):
        if high - low < 1e-4: break
        
        mid = (low + high) / 2
        # Use more attempts as the problem gets harder, but execution is fast
        res = solve_check(mid, attempt_limit=10)
        
        if res:
            res = local_optimize(res)
            actual_max = get_max_kvpr(res)
            
            if actual_max < get_max_kvpr(best_pl_list):
                 best_pl_list = res
            
            high = min(mid, actual_max)
        else:
            low = mid
            
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
