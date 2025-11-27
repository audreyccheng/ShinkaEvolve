# EVOLVE-BLOCK-START
"""Model placement algorithm for minimizing maximum KV cache pressure across GPUs"""

import random
import math

GPU_MEM_SIZE = 80.0

def compute_model_placement(gpu_num, models):
    """
    Minimizes Maximum KV Cache Pressure (KVPR) using Binary Search for initial packing 
    followed by an aggressive Iterated Local Search (ILS) with Adaptive Ruin & Recreate.
    """
    
    # --- 1. Data Structure Optimization ---
    # Store model data as tuples for faster access: (w, s, density, original_obj)
    # w = req_rate / slo, s = model_size
    items = []
    total_w = 0.0
    total_s = 0.0
    for m in models:
        w = m.req_rate / m.slo
        s = m.model_size
        d = w / s if s > 1e-6 else 0.0
        items.append((w, s, d, m))
        total_w += w
        total_s += s
        
    # --- 2. Initial Solution via Binary Search ---
    # We treat the feasibility check for a given Max Pressure K as a 1D Bin Packing problem.
    # Virtual Size of item i: v_i(K) = w_i + K * s_i
    # Bin Capacity: V(K) = K * GPU_MEM_SIZE
    # We use Best-Fit Decreasing (BFD) on Virtual Sizes.
    
    def solve_packing(target_k):
        # We need to find a packing where for every bin: L + w <= K * (C - U - s)
        # Transformed: (L+w) + K*(U+s) <= K*C
        
        # Strategies to try for sorting
        strategies = [
            lambda x: x[0] + target_k * x[1], # Virtual Size (Theory optimal for fixed K)
            lambda x: x[2],                   # Density
            lambda x: x[0],                   # Load
            lambda x: x[1]                    # Size
        ]
        
        # Try deterministic strategies
        for key_func in strategies:
            sorted_items = sorted(items, key=key_func, reverse=True)
            
            bins_w = [0.0] * gpu_num
            bins_s = [0.0] * gpu_num
            bins_items = [[] for _ in range(gpu_num)]
            possible = True
            
            for w, s, d, m in sorted_items:
                best_g = -1
                min_slack = float('inf')
                
                for g in range(gpu_num):
                    # Physical Check
                    if bins_s[g] + s >= GPU_MEM_SIZE - 1e-6: continue
                    
                    # Pressure Check: (L + w) <= K * (C - U - s)
                    rem_phys = GPU_MEM_SIZE - bins_s[g] - s
                    max_allowed_w = target_k * rem_phys
                    new_w = bins_w[g] + w
                    
                    if new_w <= max_allowed_w + 1e-7:
                        # Best Fit: Minimize unused pressure capacity (slack)
                        slack = max_allowed_w - new_w
                        if slack < min_slack:
                            min_slack = slack
                            best_g = g
                
                if best_g != -1:
                    bins_w[best_g] += w
                    bins_s[best_g] += s
                    bins_items[best_g].append(m)
                else:
                    possible = False
                    break
            
            if possible:
                return bins_items
                
        return None

    # Binary Search
    # Lower bound: Average pressure
    low = max(0.0, total_w / (gpu_num * GPU_MEM_SIZE - total_s) if (gpu_num * GPU_MEM_SIZE - total_s) > 1 else 0.0)
    high = 2000.0 # Safe upper bound
    
    best_bs_placement = None
    
    # Check feasibility of high bound
    if solve_packing(high) is None:
        high = 10000.0
        if solve_packing(high) is None:
             # Very constrained, fallback to naive later
             best_bs_placement = None
    
    if best_bs_placement is None: # Only run BS if potentially feasible
        for _ in range(20):
            if high - low < 1e-4: break
            mid = (low + high) / 2.0
            res = solve_packing(mid)
            if res:
                best_bs_placement = res
                high = mid
            else:
                low = mid
            
    if best_bs_placement is None:
        # Fallback naive distribution
        best_bs_placement = [[] for _ in range(gpu_num)]
        for i, item in enumerate(items):
            best_bs_placement[i % gpu_num].append(item[3])

    # --- 3. Iterated Local Search with Adaptive Ruin & Recreate ---
    
    # Setup internal state using tuples (w, s, m)
    # Map original object id to tuple for fast recovery
    obj_to_item = {id(m): (w, s, m) for w, s, d, m in items}
    
    sol_state = [[] for _ in range(gpu_num)]
    g_w = [0.0] * gpu_num
    g_s = [0.0] * gpu_num
    
    for g in range(gpu_num):
        for m in best_bs_placement[g]:
            t = obj_to_item[id(m)]
            sol_state[g].append(t)
            g_w[g] += t[0]
            g_s[g] += t[1]
            
    def get_pressure(w, s):
        rem = GPU_MEM_SIZE - s
        if rem <= 1e-6: return 1e9 # Penalty
        return w / rem

    # Function to get metrics
    def get_metrics():
        max_p = -1.0
        sum_sq = 0.0
        bn = -1
        p_vals = []
        for g in range(gpu_num):
            p = get_pressure(g_w[g], g_s[g])
            p_vals.append(p)
            sum_sq += p*p
            if p > max_p:
                max_p = p
                bn = g
        return max_p, sum_sq, bn, p_vals

    best_max_p, _, _, _ = get_metrics()
    best_sol_state = [list(l) for l in sol_state] # Deep copy
    
    import random
    rng = random.Random(42)
    
    iterations = 500
    stagnation = 0
    
    for it in range(iterations):
        curr_max, curr_sq, bn, pressures = get_metrics()
        
        # Update Global Best
        if curr_max < best_max_p - 1e-7:
            best_max_p = curr_max
            best_sol_state = [list(l) for l in sol_state]
            stagnation = 0
        else:
            stagnation += 1
            
        # --- 1. Greedy Descent (Move only for speed) ---
        # Move item from Bottleneck to Best Partner
        
        # Sort partners: prioritize Low Pressure
        partners = sorted([g for g in range(gpu_num) if g != bn], key=lambda g: pressures[g])
        
        # Determine max pressure of "others" (excluding BN and Partner)
        # Fast approximate: use precomputed pressures
        sorted_p_indices = sorted(range(gpu_num), key=lambda g: pressures[g], reverse=True)
        
        best_move = None 
        
        # Try moving items from BN
        bn_items = sol_state[bn]
        # Sort items by weight descending
        bn_items_indices = sorted(range(len(bn_items)), key=lambda i: bn_items[i][0], reverse=True)
        
        found_improvement = False
        
        for pt in partners:
            # Max Others
            max_others = 0.0
            for idx in sorted_p_indices:
                if idx != bn and idx != pt:
                    max_others = pressures[idx]
                    break
            
            base_sq = curr_sq - pressures[bn]**2 - pressures[pt]**2
            
            for i in bn_items_indices:
                item = bn_items[i]
                w, s, _ = item
                
                if g_s[pt] + s >= GPU_MEM_SIZE - 1e-6: continue
                
                nw_bn = g_w[bn] - w
                ns_bn = g_s[bn] - s
                nw_pt = g_w[pt] + w
                ns_pt = g_s[pt] + s
                
                p_bn = get_pressure(nw_bn, ns_bn)
                p_pt = get_pressure(nw_pt, ns_pt)
                
                n_max = max(max_others, p_bn, p_pt)
                
                if n_max < curr_max - 1e-8:
                    # Strict Improvement
                    best_move = (pt, i, nw_bn, ns_bn, nw_pt, ns_pt)
                    found_improvement = True
                    break
                elif n_max < curr_max + 1e-8:
                     # Tie-break with Variance
                     n_sq = base_sq + p_bn**2 + p_pt**2
                     if n_sq < curr_sq - 1e-8:
                         if best_move is None:
                             best_move = (pt, i, nw_bn, ns_bn, nw_pt, ns_pt)
            
            if found_improvement: break
            
        if best_move:
            pt, i, nbw, nbs, npw, nps = best_move
            item = sol_state[bn].pop(i)
            sol_state[pt].append(item)
            g_w[bn], g_s[bn] = nbw, nbs
            g_w[pt], g_s[pt] = npw, nps
            continue # Descent success, repeat
            
        # --- 2. Adaptive Ruin & Recreate ---
        # If descent failed, use perturbation
        
        # Dynamic victim size
        n_victims = 2
        if stagnation > 20: n_victims = 3
        if stagnation > 50: n_victims = 4
        n_victims = min(n_victims, gpu_num)
        
        victims = {bn}
        cands = [g for g in range(gpu_num) if g != bn]
        if cands:
            victims.update(rng.sample(cands, min(len(cands), n_victims - 1)))
        victim_list = list(victims)
        
        # Extract
        repack_items = []
        for v in victim_list:
            repack_items.extend(sol_state[v])
            sol_state[v] = []
            g_w[v] = 0.0
            g_s[v] = 0.0
            
        # Recreate: Randomized BFD with Virtual Sizes
        best_local = None
        best_local_max = float('inf')
        
        # Target pressure to beat
        target_k = curr_max * 0.95
        
        # Multiple trials
        for _ in range(12):
            rng.shuffle(repack_items)
            # Sort by Virtual Size with noise
            repack_items.sort(key=lambda x: (x[0] + target_k * x[1]) * rng.uniform(0.9, 1.1), reverse=True)
            
            l_w = {v: 0.0 for v in victim_list}
            l_s = {v: 0.0 for v in victim_list}
            l_alloc = {v: [] for v in victim_list}
            possible = True
            
            for item in repack_items:
                w, s, _ = item
                best_v = -1
                min_slack = float('inf')
                
                # BFD logic for Target K
                for v in victim_list:
                    if l_s[v] + s >= GPU_MEM_SIZE - 1e-6: continue
                    rem = GPU_MEM_SIZE - l_s[v] - s
                    max_allowed = target_k * rem
                    new_val = l_w[v] + w
                    
                    if new_val <= max_allowed + 1e-7:
                        slack = max_allowed - new_val
                        if slack < min_slack:
                            min_slack = slack
                            best_v = v
                
                # Fallback: Best Fit on Pressure
                if best_v == -1:
                    best_p = float('inf')
                    for v in victim_list:
                        if l_s[v] + s < GPU_MEM_SIZE - 1e-6:
                            p = (l_w[v] + w) / (GPU_MEM_SIZE - l_s[v] - s)
                            if p < best_p:
                                best_p = p
                                best_v = v
                
                if best_v != -1:
                    l_alloc[best_v].append(item)
                    l_w[best_v] += w
                    l_s[best_v] += s
                else:
                    possible = False
                    break
            
            if possible:
                lm = max(get_pressure(l_w[v], l_s[v]) for v in victim_list)
                if lm < best_local_max:
                    best_local_max = lm
                    best_local = (l_alloc, l_w, l_s)

        # Decision: Apply if valid and not a huge degradation (Monotonic Basin Hopping-like)
        # We accept if it improves current OR matches it. 
        # Actually, strict improvement over current is hard if current is local min.
        # We accept if better than current OR with some probability?
        # Simpler: Accept if valid. If worse, the next descent phase will clean it up or we revert to global best later.
        # However, to avoid divergence, we only accept if it beats current max (strict descent).
        if best_local and best_local_max < curr_max - 1e-9:
            alloc, ws, ss = best_local
            for v in victim_list:
                sol_state[v] = alloc[v]
                g_w[v] = ws[v]
                g_s[v] = ss[v]
        else:
            # Revert to Global Best
            # This resets the search to the best known point, avoiding getting lost in worse regions
            sol_state = [list(l) for l in best_sol_state]
            g_w = [0.0]*gpu_num
            g_s = [0.0]*gpu_num
            for g in range(gpu_num):
                for w, s, m in sol_state[g]:
                    g_w[g] += w
                    g_s[g] += s

    # Final Output
    final_output = {g: [x[2] for x in best_sol_state[g]] for g in range(gpu_num)}
    return final_output
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