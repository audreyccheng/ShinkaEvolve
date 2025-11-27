# EVOLVE-BLOCK-START
"""Model placement algorithm for minimizing maximum KV cache pressure across GPUs"""

import random

GPU_MEM_SIZE = 80  # GB

def compute_model_placement(gpu_num, models):
    """
    Compute a model placement that minimizes the maximum KVPR across all GPUs.
    Combines Binary Search for Feasibility (solving BPP with target K) 
    and Lexicographical Descent Local Search with Smart Perturbation.
    """
    random.seed(42)

    # --- 1. Preprocessing ---
    m_data = []
    total_w = 0.0
    total_s = 0.0
    for m in models:
        w = m.req_rate / m.slo if m.slo > 0 else 0
        s = m.model_size
        m_data.append({'w': w, 's': s, 'obj': m})
        total_w += w
        total_s += s

    # --- 2. Binary Search for Initial Solution ---
    
    # We define the problem: Can we pack all items such that pressure <= K?
    # Constraint: (L + w) / (C - (U + s)) <= K
    # Transformed: (L + w) + K*(U + s) <= K*C
    # This is effectively a Bin Packing Problem with:
    #   Item Size_eff = w + K*s
    #   Bin Cap_eff = K*C
    
    def solve_bpp_feasibility(target_k):
        eff_capacity = target_k * GPU_MEM_SIZE
        
        # Heuristic Sort Keys for BFD
        strategies = [
            lambda x: x['w'] + target_k * x['s'], # Virtual Size (Exact constraint term)
            lambda x: x['s'],                     # Physical Size
            lambda x: x['w'] / x['s'] if x['s'] > 1e-6 else 0, # Density
            lambda x: x['w']                      # Workload
        ]
        
        for key_fn in strategies:
            sorted_items = sorted(m_data, key=key_fn, reverse=True)
            
            alloc = [[] for _ in range(gpu_num)]
            eff_loads = [0.0] * gpu_num # Track load + K*size
            phy_used = [0.0] * gpu_num
            possible = True
            
            for item in sorted_items:
                best_g = -1
                min_slack = float('inf')
                
                item_eff_size = item['w'] + target_k * item['s']
                
                for g in range(gpu_num):
                    # Hard memory constraint
                    if phy_used[g] + item['s'] > GPU_MEM_SIZE - 1e-6:
                        continue
                        
                    # Soft pressure constraint (Virtual Capacity)
                    if eff_loads[g] + item_eff_size <= eff_capacity + 1e-7:
                        # Best Fit: minimize slack
                        slack = eff_capacity - (eff_loads[g] + item_eff_size)
                        if slack < min_slack:
                            min_slack = slack
                            best_g = g
                
                if best_g != -1:
                    alloc[best_g].append(item['obj'])
                    eff_loads[best_g] += item_eff_size
                    phy_used[best_g] += item['s']
                else:
                    possible = False
                    break
            
            if possible:
                return alloc
        return None

    # Determine Binary Search Bounds
    rem_avg = max(1e-6, GPU_MEM_SIZE * gpu_num - total_s)
    lb = total_w / rem_avg
    ub = 2000.0 
    
    best_init = None
    
    for _ in range(20):
        if ub - lb < 1e-4: break
        mid = (lb + ub) / 2.0
        res = solve_bpp_feasibility(mid)
        if res:
            best_init = res
            ub = mid
        else:
            lb = mid
            
    # Fallback to simple Greedy if BS fails (rare)
    if best_init is None:
        best_init = [[] for _ in range(gpu_num)]
        # Sort by density
        s_items = sorted(m_data, key=lambda x: x['w']/x['s'] if x['s']>0 else 0, reverse=True)
        g_l = [0.0]*gpu_num
        g_u = [0.0]*gpu_num
        for item in s_items:
            best_g = -1
            best_sc = float('inf')
            for g in range(gpu_num):
                rem = GPU_MEM_SIZE - g_u[g] - item['s']
                if rem > 1e-6:
                    sc = (g_l[g] + item['w']) / rem
                    if sc < best_sc:
                        best_sc = sc
                        best_g = g
            if best_g != -1:
                best_init[best_g].append(item['obj'])
                g_l[best_g] += item['w']
                g_u[best_g] += item['s']
            else:
                # Force fit
                for g in range(gpu_num):
                     if g_u[g] + item['s'] <= GPU_MEM_SIZE - 1e-6:
                         best_init[g].append(item['obj'])
                         g_l[g] += item['w']
                         g_u[g] += item['s']
                         break

    # --- 3. Lexicographical Descent Local Search ---
    
    current_placement = {i: list(p) for i, p in enumerate(best_init)}
    
    # State tracking
    loads = [0.0] * gpu_num
    used = [0.0] * gpu_num
    for g in range(gpu_num):
        for m in current_placement[g]:
            loads[g] += m.req_rate / m.slo
            used[g] += m.model_size
            
    def get_pressure(l, u):
        rem = GPU_MEM_SIZE - u
        return l / rem if rem > 1e-6 else (float('inf') if l > 1e-9 else 0.0)
        
    pressures = [get_pressure(loads[g], used[g]) for g in range(gpu_num)]
    
    # Tuple of sorted pressures (descending) represents the "quality" vector
    current_sorted = tuple(sorted(pressures, reverse=True))
    best_sorted = current_sorted
    best_placement_copy = {k: list(v) for k, v in current_placement.items()}
    
    max_iters = 300
    
    for it in range(max_iters):
        # Identify Bottleneck (Max Pressure)
        # Use simple max search
        max_p = current_sorted[0]
        bottleneck = -1
        for g in range(gpu_num):
            if abs(pressures[g] - max_p) < 1e-9:
                bottleneck = g
                break
        
        best_move = None 
        # (type, partner, idx_bn, idx_pt, nl_b, nu_b, nl_p, nu_p, new_sorted_tuple)
        
        bn_items = current_placement[bottleneck]
        partners = [g for g in range(gpu_num) if g != bottleneck]
        
        # 1. Evaluate Moves: BN -> Partner
        for partner in partners:
            if used[partner] >= GPU_MEM_SIZE - 1e-6: continue
            
            for i, m in enumerate(bn_items):
                w, s = m.req_rate/m.slo, m.model_size
                if used[partner] + s > GPU_MEM_SIZE - 1e-6: continue
                
                nl_b = loads[bottleneck] - w
                nu_b = used[bottleneck] - s
                nl_p = loads[partner] + w
                nu_p = used[partner] + s
                
                p_b_new = get_pressure(nl_b, nu_b)
                p_p_new = get_pressure(nl_p, nu_p)
                
                # Pruning: if new local max >= global max, likely no improvement
                if max(p_b_new, p_p_new) > max_p + 1e-9: continue
                
                # Construct candidate sorted tuple
                # Optimized: We only change 2 values.
                # However, for simplicity and correctness in Python, reconstructing list is fine for N=small
                temp_p = list(pressures)
                temp_p[bottleneck] = p_b_new
                temp_p[partner] = p_p_new
                cand_sorted = tuple(sorted(temp_p, reverse=True))
                
                if cand_sorted < current_sorted:
                    if best_move is None or cand_sorted < best_move[8]:
                        best_move = ('move', partner, i, -1, nl_b, nu_b, nl_p, nu_p, cand_sorted)
                        
        # 2. Evaluate Swaps: BN <-> Partner
        for partner in partners:
             pt_items = current_placement[partner]
             for i, m1 in enumerate(bn_items):
                 w1, s1 = m1.req_rate/m1.slo, m1.model_size
                 for j, m2 in enumerate(pt_items):
                     w2, s2 = m2.req_rate/m2.slo, m2.model_size
                     
                     nu_b = used[bottleneck] - s1 + s2
                     nu_p = used[partner] - s2 + s1
                     if nu_b > GPU_MEM_SIZE - 1e-6 or nu_p > GPU_MEM_SIZE - 1e-6: continue
                     
                     nl_b = loads[bottleneck] - w1 + w2
                     nl_p = loads[partner] - w2 + w1
                     
                     p_b_new = get_pressure(nl_b, nu_b)
                     p_p_new = get_pressure(nl_p, nu_p)
                     
                     if max(p_b_new, p_p_new) > max_p + 1e-9: continue
                     
                     temp_p = list(pressures)
                     temp_p[bottleneck] = p_b_new
                     temp_p[partner] = p_p_new
                     cand_sorted = tuple(sorted(temp_p, reverse=True))
                     
                     if cand_sorted < current_sorted:
                         if best_move is None or cand_sorted < best_move[8]:
                             best_move = ('swap', partner, i, j, nl_b, nu_b, nl_p, nu_p, cand_sorted)

        if best_move:
            # Apply Move
            mtype, pt, i, j, nlb, nub, nlp, nup, n_sort = best_move
            if mtype == 'move':
                item = current_placement[bottleneck].pop(i)
                current_placement[pt].append(item)
            else:
                item1 = current_placement[bottleneck][i]
                item2 = current_placement[pt][j]
                current_placement[bottleneck][i] = item2
                current_placement[pt][j] = item1
            
            loads[bottleneck] = nlb
            used[bottleneck] = nub
            loads[pt] = nlp
            used[pt] = nup
            pressures[bottleneck] = get_pressure(nlb, nub)
            pressures[pt] = get_pressure(nlp, nup)
            current_sorted = n_sort
            
            if current_sorted < best_sorted:
                best_sorted = current_sorted
                best_placement_copy = {k: list(v) for k, v in current_placement.items()}
        else:
            # Smart Perturbation
            # Pick Bottleneck + Lowest Pressure Partner + Random Partner
            min_p_val = current_sorted[-1]
            cands = [g for g in range(gpu_num) if g != bottleneck and abs(pressures[g] - min_p_val) < 1e-9]
            target_low = cands[0] if cands else (bottleneck + 1) % gpu_num
            
            others = [g for g in range(gpu_num) if g != bottleneck and g != target_low]
            target_rand = random.choice(others) if others else target_low
            
            victims = list(set([bottleneck, target_low, target_rand]))
            
            # Extract
            repack_items = []
            for v in victims:
                repack_items.extend(current_placement[v])
                current_placement[v] = []
                loads[v] = 0.0
                used[v] = 0.0
                pressures[v] = 0.0
                
            # Repack with randomized heuristic (Best Fit with noise)
            best_local_alloc = None
            best_local_metrics = None # (loads, used)
            best_local_max = float('inf')
            
            for _ in range(10): # Mini-batch trials
                # Randomize order and sort criteria slightly
                iter_items = list(repack_items)
                random.shuffle(iter_items)
                # Sort by (w/s) * noise
                iter_items.sort(key=lambda x: (x.req_rate/x.slo)/(x.model_size+1e-6) * random.uniform(0.8, 1.2), reverse=True)
                
                l_alloc = {v: [] for v in victims}
                l_load = {v: 0.0 for v in victims}
                l_used = {v: 0.0 for v in victims}
                l_possible = True
                
                for item in iter_items:
                    w, s = item.req_rate/item.slo, item.model_size
                    best_v = -1
                    best_sc = float('inf')
                    
                    for v in victims:
                        if l_used[v] + s <= GPU_MEM_SIZE - 1e-6:
                            rem = GPU_MEM_SIZE - l_used[v] - s
                            p = (l_load[v] + w) / rem if rem > 1e-6 else float('inf')
                            if p < best_sc:
                                best_sc = p
                                best_v = v
                    
                    if best_v == -1:
                         # Try force fit any capable
                         for v in victims:
                             if l_used[v] + s <= GPU_MEM_SIZE - 1e-6:
                                 best_v = v
                                 break
                    
                    if best_v != -1:
                        l_alloc[best_v].append(item)
                        l_load[best_v] += w
                        l_used[best_v] += s
                    else:
                        l_possible = False
                        break
                
                if l_possible:
                    # check max pressure of this config
                    local_max = max(get_pressure(l_load[v], l_used[v]) for v in victims)
                    if local_max < best_local_max:
                        best_local_max = local_max
                        best_local_alloc = l_alloc
                        best_local_metrics = (l_load, l_used)
            
            if best_local_alloc:
                # Apply successful kick
                for v in victims:
                    current_placement[v] = best_local_alloc[v]
                    loads[v] = best_local_metrics[0][v]
                    used[v] = best_local_metrics[1][v]
                    pressures[v] = get_pressure(loads[v], used[v])
                
                # Update sorted tuple
                current_sorted = tuple(sorted(pressures, reverse=True))
            else:
                # Revert
                current_placement = {k: list(v) for k, v in best_placement_copy.items()}
                loads = [0.0]*gpu_num
                used = [0.0]*gpu_num
                for g in range(gpu_num):
                    for m in current_placement[g]:
                        loads[g] += m.req_rate/m.slo
                        used[g] += m.model_size
                pressures = [get_pressure(loads[g], used[g]) for g in range(gpu_num)]
                current_sorted = tuple(sorted(pressures, reverse=True))

    return best_placement_copy

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