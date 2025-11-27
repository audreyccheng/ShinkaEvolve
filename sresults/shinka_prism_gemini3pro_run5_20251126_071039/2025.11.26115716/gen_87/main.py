# EVOLVE-BLOCK-START
"""Model placement algorithm for minimizing maximum KV cache pressure across GPUs"""

import random
import math

GPU_MEM_SIZE = 80  # GB

def compute_model_placement(gpu_num, models):
    """
    Minimizes maximum KV cache pressure using Binary Search on effective capacity
    followed by Lexicographical Local Search optimization.
    """
    
    # 1. Setup and Preprocessing
    m_data = []
    for i, m in enumerate(models):
        w = m.req_rate / m.slo
        s = m.model_size
        # Primary sort key for heuristics: Density
        d = w / s if s > 1e-6 else 0
        m_data.append({'w': w, 's': s, 'd': d, 'obj': m, 'id': i})

    # Helper for pressure
    def calc_pressure(l, u):
        rem = GPU_MEM_SIZE - u
        if rem <= 1e-6:
            # If load > 0 and rem ~ 0, pressure is infinite. 
            # If load = 0 and rem ~ 0, pressure is 0.
            return float('inf') if l > 1e-7 else 0.0
        return l / rem

    # 2. Initial Solution & Bounds
    # We generate an initial solution using heuristics to get a tight upper bound
    best_placement = None
    min_max_p = float('inf')

    # Heuristic Packing strategies
    # Strategy: Sort items, then place in bin that minimizes local pressure increase
    # Keys: Density, Load, Size
    sort_keys = [
        lambda x: x['d'], 
        lambda x: x['w'], 
        lambda x: x['s']
    ]

    for key_fn in sort_keys:
        sorted_items = sorted(m_data, key=key_fn, reverse=True)
        current_alloc = [[] for _ in range(gpu_num)]
        cur_l = [0.0] * gpu_num
        cur_u = [0.0] * gpu_num
        possible = True
        
        for item in sorted_items:
            best_g = -1
            best_p = float('inf')
            
            # Find best bin (Min-Pressure Fit)
            for g in range(gpu_num):
                if cur_u[g] + item['s'] <= GPU_MEM_SIZE - 1e-6:
                    # Calculate potential pressure
                    p = calc_pressure(cur_l[g] + item['w'], cur_u[g] + item['s'])
                    if p < best_p:
                        best_p = p
                        best_g = g
            
            if best_g != -1:
                current_alloc[best_g].append(item['obj'])
                cur_l[best_g] += item['w']
                cur_u[best_g] += item['s']
            else:
                possible = False
                break
        
        if possible:
            # Evaluate global max pressure
            current_max = max(calc_pressure(cur_l[g], cur_u[g]) for g in range(gpu_num))
            if current_max < min_max_p:
                min_max_p = current_max
                best_placement = current_alloc

    # If heuristics failed (unlikely), fallback to broad search
    low_bound = 0.0
    high_bound = min_max_p if best_placement else 2000.0
    
    # 3. Binary Search for Optimal K (Min-Max Pressure)
    # Problem Transformation: Bin Packing with Item Size = w + K*s, Bin Cap = K*C
    # Constraint: (L+w)/(C-U-s) <= K  <=>  L+w + K(U+s) <= K*C
    
    bs_placement = best_placement # Fallback
    
    # Check function for BS
    def check_feasible(K):
        eff_cap = K * GPU_MEM_SIZE
        
        # Helper for BFD packing
        def try_pack(items_ordered):
            bins_eff = [0.0] * gpu_num # Effective load accumulator
            bins_u = [0.0] * gpu_num   # Physical usage check
            alloc = [[] for _ in range(gpu_num)]
            
            for item in items_ordered:
                eff_size = item['w'] + K * item['s']
                
                best_g = -1
                min_rem = float('inf')
                
                for g in range(gpu_num):
                    # Check physical constraint
                    if bins_u[g] + item['s'] > GPU_MEM_SIZE - 1e-6: continue
                    
                    # Check effective constraint: Load + Size <= Capacity
                    if bins_eff[g] + eff_size <= eff_cap + 1e-7:
                        # Best Fit: Minimize remaining capacity (tightest fit)
                        rem = eff_cap - (bins_eff[g] + eff_size)
                        if rem < min_rem:
                            min_rem = rem
                            best_g = g
                
                if best_g != -1:
                    bins_eff[best_g] += eff_size
                    bins_u[best_g] += item['s']
                    alloc[best_g].append(item['obj'])
                else:
                    return None
            return alloc

        # Strategy 1: Sort by Effective Size Descending (Standard BFD for transformed problem)
        # As K changes, the effective size weights w and s differently.
        items_by_eff = sorted(m_data, key=lambda x: x['w'] + K*x['s'], reverse=True)
        res = try_pack(items_by_eff)
        if res: return res
        
        # Strategy 2: Sort by Density (Heuristic backup)
        items_by_dens = sorted(m_data, key=lambda x: x['d'], reverse=True)
        res = try_pack(items_by_dens)
        if res: return res
        
        # Strategy 3: Randomized Shuffles (Handle edge cases where greedy fails)
        base_items = list(m_data)
        # Seed logic to make it deterministic for a fixed K, but varying across K if needed
        rng = random.Random(int(K * 100)) 
        for _ in range(5):
            rng.shuffle(base_items)
            res = try_pack(base_items)
            if res: return res
            
        return None

    # BS execution
    iterations = 20
    for _ in range(iterations):
        if high_bound - low_bound < 1e-4: break
        mid = (low_bound + high_bound) / 2.0
        
        res = check_feasible(mid)
        if res:
            bs_placement = res
            high_bound = mid
        else:
            low_bound = mid
            
    if bs_placement is None:
        raise ValueError("No feasible solution found.")
        
    # 4. Iterated Local Search (ILS) with Lexicographical Descent
    
    # State initialization
    current_p = {i: list(bs_placement[i]) for i in range(gpu_num)}
    
    loads = [0.0] * gpu_num
    used = [0.0] * gpu_num
    for g in range(gpu_num):
        for m in current_p[g]:
            loads[g] += m.req_rate / m.slo
            used[g] += m.model_size
            
    # Track pressures
    pressures = [calc_pressure(loads[g], used[g]) for g in range(gpu_num)]
    
    best_global_p = {k: list(v) for k,v in current_p.items()}
    best_global_max = max(pressures)
    
    max_ils_iters = 250
    
    for it in range(max_ils_iters):
        # Current metrics
        curr_max = max(pressures)
        
        # Update global
        if curr_max < best_global_max - 1e-7:
            best_global_max = curr_max
            best_global_p = {k: list(v) for k,v in current_p.items()}
            
        # Identify bottleneck
        g_indices = sorted(range(gpu_num), key=lambda g: pressures[g], reverse=True)
        bottleneck = g_indices[0]
        
        # --- Descent Phase (Moves/Swaps) ---
        improved = False
        
        # Top-K pressure values for fast comparison
        # We need the max pressure excluding the GPUs involved in move
        
        candidates = [g for g in range(gpu_num) if g != bottleneck]
        
        # Try Moving Bottleneck Item -> Candidate
        best_move = None 
        # (type, cand, item_idx, swap_idx, n_bn_l, n_bn_u, n_cand_l, n_cand_u, new_max)
        
        bn_items = current_p[bottleneck]
        
        for cand in candidates:
            # Max pressure among others
            max_others = 0.0
            for g in g_indices:
                if g != bottleneck and g != cand:
                    max_others = pressures[g]
                    break # Since g_indices is sorted
            
            # 1. Move
            for i, item in enumerate(bn_items):
                w, s = item.req_rate/item.slo, item.model_size
                
                if used[cand] + s > GPU_MEM_SIZE - 1e-6: continue
                
                n_bl = loads[bottleneck] - w
                n_bu = used[bottleneck] - s
                n_cl = loads[cand] + w
                n_cu = used[cand] + s
                
                pb = calc_pressure(n_bl, n_bu)
                pc = calc_pressure(n_cl, n_cu)
                
                n_max = max(max_others, pb, pc)
                
                # Check for strict improvement in max pressure
                if n_max < curr_max - 1e-8:
                    if best_move is None or n_max < best_move[8]:
                        best_move = ('move', cand, i, -1, n_bl, n_bu, n_cl, n_cu, n_max)
            
            # 2. Swap (Only if move didn't yield significant improvement or just to explore)
            cand_items = current_p[cand]
            for i, item1 in enumerate(bn_items):
                w1, s1 = item1.req_rate/item1.slo, item1.model_size
                for j, item2 in enumerate(cand_items):
                    w2, s2 = item2.req_rate/item2.slo, item2.model_size
                    
                    n_bu = used[bottleneck] - s1 + s2
                    if n_bu > GPU_MEM_SIZE - 1e-6: continue
                    n_cu = used[cand] - s2 + s1
                    if n_cu > GPU_MEM_SIZE - 1e-6: continue
                    
                    n_bl = loads[bottleneck] - w1 + w2
                    n_cl = loads[cand] - w2 + w1
                    
                    pb = calc_pressure(n_bl, n_bu)
                    pc = calc_pressure(n_cl, n_cu)
                    
                    n_max = max(max_others, pb, pc)
                    
                    if n_max < curr_max - 1e-8:
                        if best_move is None or n_max < best_move[8]:
                             best_move = ('swap', cand, i, j, n_bl, n_bu, n_cl, n_cu, n_max)

        if best_move:
            mtype, cand, i, j, nbl, nbu, ncl, ncu, _ = best_move
            if mtype == 'move':
                it_obj = current_p[bottleneck].pop(i)
                current_p[cand].append(it_obj)
            else:
                it1 = current_p[bottleneck][i]
                it2 = current_p[cand][j]
                current_p[bottleneck][i] = it2
                current_p[cand][j] = it1
            
            loads[bottleneck], used[bottleneck] = nbl, nbu
            loads[cand], used[cand] = ncl, ncu
            pressures[bottleneck] = calc_pressure(nbl, nbu)
            pressures[cand] = calc_pressure(ncl, ncu)
            improved = True
            
        # --- Perturbation Phase (Ruin & Recreate) ---
        # If descent failed to improve max pressure, we kick the system.
        if not improved:
            # Victims: Bottleneck + Least Loaded (to balance) + Random (to shuffle)
            victims = {bottleneck}
            if candidates:
                # Least loaded
                min_g = min(candidates, key=lambda g: pressures[g])
                victims.add(min_g)
                # Random
                rem_cands = [g for g in candidates if g not in victims]
                if rem_cands:
                    victims.update(random.sample(rem_cands, 1))
            
            v_list = list(victims)
            
            # Extract items
            repack_items = []
            for v in v_list:
                repack_items.extend(current_p[v])
                current_p[v] = []
                loads[v] = 0.0
                used[v] = 0.0
                pressures[v] = 0.0
            
            # Repack Strategies
            # Try to redistribute items among v_list to minimize local max pressure.
            # We use a greedy "Min-Pressure Fit" with randomized sorts.
            
            best_local_alloc = None
            best_local_max = float('inf')
            
            sub_sorts = [
                lambda x: x.req_rate/x.slo, # Load
                lambda x: (x.req_rate/x.slo)/(x.model_size+1e-6), # Density
                lambda x: x.model_size # Size
            ]
            
            # Try multiple randomized passes
            for k in range(12):
                if k < 3:
                    # Deterministic sorts with slight noise to break ties
                    items_iter = sorted(repack_items, key=lambda x: sub_sorts[k](x) * random.uniform(0.99, 1.01), reverse=True)
                else:
                    items_iter = list(repack_items)
                    random.shuffle(items_iter)
                
                # Temp state
                t_alloc = {v: [] for v in v_list}
                t_l = {v: 0.0 for v in v_list}
                t_u = {v: 0.0 for v in v_list}
                ok = True
                
                for item in items_iter:
                    w = item.req_rate/item.slo
                    s = item.model_size
                    
                    best_v = -1
                    min_p_incr = float('inf')
                    
                    # Try to put in bin with lowest resulting pressure (Greedy Balancing)
                    for v in v_list:
                        if t_u[v] + s <= GPU_MEM_SIZE - 1e-6:
                            p = calc_pressure(t_l[v] + w, t_u[v] + s)
                            if p < min_p_incr:
                                min_p_incr = p
                                best_v = v
                    
                    if best_v != -1:
                        t_alloc[best_v].append(item)
                        t_l[best_v] += w
                        t_u[best_v] += s
                    else:
                        ok = False
                        break
                
                if ok:
                    # Check local max pressure of this configuration
                    l_max = max(calc_pressure(t_l[v], t_u[v]) for v in v_list)
                    if l_max < best_local_max:
                        best_local_max = l_max
                        best_local_alloc = (t_alloc, t_l, t_u)
            
            if best_local_alloc:
                t_alloc, t_l, t_u = best_local_alloc
                for v in v_list:
                    current_p[v] = t_alloc[v]
                    loads[v] = t_l[v]
                    used[v] = t_u[v]
                    pressures[v] = calc_pressure(loads[v], used[v])
            else:
                # Failed to repack (very rare, implies super tight packing), revert to global best
                current_p = {k: list(v) for k,v in best_global_p.items()}
                # Recompute state
                loads = [0.0]*gpu_num
                used = [0.0]*gpu_num
                for g in range(gpu_num):
                    for m in current_p[g]:
                        loads[g] += m.req_rate / m.slo
                        used[g] += m.model_size
                pressures = [calc_pressure(loads[g], used[g]) for g in range(gpu_num)]
                
    return best_global_p
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