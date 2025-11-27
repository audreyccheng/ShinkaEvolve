# EVOLVE-BLOCK-START
"""Model placement algorithm for minimizing maximum KV cache pressure across GPUs"""

import math
import random

GPU_MEM_SIZE = 80.0

def compute_model_placement(gpu_num, models):
    """
    Compute a model placement that minimizes the maximum KVPR across all GPUs.
    
    Algorithm:
    1. Binary Search for optimal Max-KVPR (K).
    2. Feasibility Check (Hybrid):
       - Deterministic Greedy Best-Fit with multiple sort keys (Linearized, Size, Pressure).
       - Randomized Greedy with Failure-Driven Prioritization:
         If packing fails, unplaced items are moved to the front for the next attempt.
    3. Local Search (Steepest Descent):
       - Refines placement by moving/swapping items from the bottleneck GPU.
    """

    # 1. Preprocess items
    items = []
    for m in models:
        items.append({
            'model': m,
            'w': m.req_rate / m.slo,
            's': m.model_size,
            'id': id(m)
        })

    def calc_kvpr(w, s):
        """Calculate KVPR: w / (C - s)."""
        rem = GPU_MEM_SIZE - s
        if rem <= 1e-9:
            return float('inf') if w > 1e-9 else 0.0
        return w / rem

    def get_max_kvpr(placement):
        """Get max KVPR over all GPUs."""
        mx = 0.0
        for p in placement.values():
            w = sum(m.req_rate / m.slo for m in p)
            s = sum(m.model_size for m in p)
            mx = max(mx, calc_kvpr(w, s))
        return mx

    def solve_packing(k_target, ordered_items):
        """
        Attempt to pack items into GPUs with KVPR <= k_target.
        Returns (placement_dict, unplaced_list).
        """
        limit_lin = k_target * GPU_MEM_SIZE
        
        # Bins state: {'w', 's', 'items'}
        bins = [{'w': 0.0, 's': 0.0, 'items': []} for _ in range(gpu_num)]
        unplaced = []

        for item in ordered_items:
            w, s = item['w'], item['s']
            
            # Basic sanity check
            if s > GPU_MEM_SIZE:
                unplaced.append(item)
                continue

            cost_lin = w + k_target * s
            
            best_idx = -1
            best_score = -1.0 # Best Fit Score
            
            for i in range(gpu_num):
                b = bins[i]
                
                # 1. Physical Capacity Check
                if b['s'] + s > GPU_MEM_SIZE: continue
                
                # 2. Linearized KVPR Check: (w_bin + w) + K*(s_bin + s) <= K*C
                current_lin = b['w'] + k_target * b['s']
                if current_lin + cost_lin > limit_lin + 1e-7: continue
                
                # 3. Best Fit Heuristic
                # Maximize the linearized fill -> Minimize remaining space.
                # This keeps fragmentation low.
                score = current_lin + cost_lin
                if score > best_score:
                    best_score = score
                    best_idx = i
            
            if best_idx != -1:
                bins[best_idx]['w'] += w
                bins[best_idx]['s'] += s
                bins[best_idx]['items'].append(item['model'])
            else:
                unplaced.append(item)
        
        placement = {i: bins[i]['items'] for i in range(gpu_num)}
        return placement, unplaced

    def check_placement(k_target):
        # A. Deterministic Heuristics
        # Sort keys to try
        strategies = [
            lambda x: x['w'] + k_target * x['s'],            # Linearized Cost
            lambda x: x['s'],                                # Size
            lambda x: x['w'] / (GPU_MEM_SIZE - x['s'] + 1e-5) # Asymptotic Pressure
        ]
        
        for key in strategies:
            ordered = sorted(items, key=key, reverse=True)
            placement, unplaced = solve_packing(k_target, ordered)
            if not unplaced:
                return placement

        # B. Randomized Failure-Driven Heuristic
        # Use a RNG seeded by k_target for reproducibility
        rng = random.Random(42 + int(k_target * 100))
        
        # Initial list for randomized phase (Linearized Cost usually best base)
        current_items = sorted(items, key=lambda x: x['w'] + k_target * x['s'], reverse=True)
        
        attempts = 30
        for _ in range(attempts):
            # 1. Perturb weights with noise
            # Create a noisy key for sorting
            # This helps explore neighbors of the best deterministic strategy
            noisy_order = sorted(current_items, 
                               key=lambda x: (x['w'] + k_target * x['s']) * rng.uniform(0.85, 1.15), 
                               reverse=True)
            
            placement, unplaced = solve_packing(k_target, noisy_order)
            
            if not unplaced:
                return placement
            
            # 2. Failure-Driven Prioritization
            # If we failed, take the unplaced items and put them at the start of the next iteration's list.
            # We mix them with the placed items from this attempt.
            unplaced_ids = set(x['id'] for x in unplaced)
            placed_items = [x for x in noisy_order if x['id'] not in unplaced_ids]
            
            # Update current_items for next loop: Unplaced (Priority) + Placed
            # Note: We don't shuffle placed items aggressively here, relying on the noisy sort in next iter
            # But the 'unplaced' block effectively permutes the order significantly.
            current_items = unplaced + placed_items

        return None

    def local_optimize(placement):
        """Steepest Descent Hill Climbing to minimize max KVPR."""
        state = []
        for i in range(gpu_num):
            p = placement[i]
            w = sum(m.req_rate / m.slo for m in p)
            s = sum(m.model_size for m in p)
            state.append({'w': w, 's': s, 'items': list(p)})
            
        for _ in range(150): # Limit iterations
            # Find bottleneck
            max_k = -1.0
            src_idx = -1
            gpu_ks = []
            
            for i, st in enumerate(state):
                k = calc_kvpr(st['w'], st['s'])
                gpu_ks.append(k)
                if k > max_k:
                    max_k = k
                    src_idx = i
            
            if max_k <= 1e-9: break
            
            src = state[src_idx]
            improved = False
            
            # 1. Move Operation
            for i, item in enumerate(src['items']):
                iw = item.req_rate / item.slo
                is_ = item.model_size
                
                ns_w = src['w'] - iw
                ns_s = src['s'] - is_
                ns_k = calc_kvpr(ns_w, ns_s)
                
                # Optimization: Only consider moves that significantly help src
                if ns_k >= max_k - 1e-9: continue
                
                for dst_idx in range(gpu_num):
                    if dst_idx == src_idx: continue
                    dst = state[dst_idx]
                    
                    if dst['s'] + is_ > GPU_MEM_SIZE: continue
                    
                    nd_w = dst['w'] + iw
                    nd_s = dst['s'] + is_
                    nd_k = calc_kvpr(nd_w, nd_s)
                    
                    if max(ns_k, nd_k) < max_k - 1e-9:
                        # Apply Move
                        src['items'].pop(i)
                        src['w'], src['s'] = ns_w, ns_s
                        dst['items'].append(item)
                        dst['w'], dst['s'] = nd_w, nd_s
                        improved = True
                        break
                if improved: break
            
            if improved: continue
            
            # 2. Swap Operation
            for i, item1 in enumerate(src['items']):
                iw1, is1 = item1.req_rate / item1.slo, item1.model_size
                
                for dst_idx in range(gpu_num):
                    if dst_idx == src_idx: continue
                    dst = state[dst_idx]
                    
                    # Heuristic: Don't swap with a GPU that is nearly as bad
                    if gpu_ks[dst_idx] > max_k * 0.95: continue
                    
                    for j, item2 in enumerate(dst['items']):
                        iw2, is2 = item2.req_rate / item2.slo, item2.model_size
                        
                        ns_s = src['s'] - is1 + is2
                        nd_s = dst['s'] - is2 + is1
                        if ns_s > GPU_MEM_SIZE or nd_s > GPU_MEM_SIZE: continue
                        
                        ns_w = src['w'] - iw1 + iw2
                        nd_w = dst['w'] - iw2 + iw1
                        
                        ns_k = calc_kvpr(ns_w, ns_s)
                        nd_k = calc_kvpr(nd_w, nd_s)
                        
                        if max(ns_k, nd_k) < max_k - 1e-9:
                            # Apply Swap
                            src['items'][i] = item2
                            src['w'], src['s'] = ns_w, ns_s
                            dst['items'][j] = item1
                            dst['w'], dst['s'] = nd_w, nd_s
                            improved = True
                            break
                    if improved: break
                if improved: break
            
            if not improved: break
            
        return {i: state[i]['items'] for i in range(gpu_num)}

    # Binary Search
    high = 1e9
    
    # 1. Initial Feasibility Check
    best_placement = check_placement(high)
    if best_placement is None:
        raise ValueError("Unable to place models on GPUs.")
    
    # Optimize initial result to tighten bound immediately
    best_placement = local_optimize(best_placement)
    high = get_max_kvpr(best_placement)
    low = 0.0
    
    # 2. Binary Search Loop
    for _ in range(25):
        if high - low < 1e-4: break
        mid = (low + high) / 2
        
        res = check_placement(mid)
        if res:
            # If feasible, try to optimize it to see if we can get an even lower max_kvpr
            # This helps update 'high' more aggressively
            res = local_optimize(res)
            mx = get_max_kvpr(res)
            
            if mx < get_max_kvpr(best_placement):
                best_placement = res
            
            high = min(mid, mx)
        else:
            low = mid
            
    return local_optimize(best_placement)
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