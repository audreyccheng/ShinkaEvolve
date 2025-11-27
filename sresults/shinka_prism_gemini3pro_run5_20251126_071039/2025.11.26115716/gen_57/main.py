# EVOLVE-BLOCK-START
"""Model placement algorithm for minimizing maximum KV cache pressure across GPUs"""

import random

GPU_MEM_SIZE = 80  # GB

def compute_model_placement(gpu_num, models):
    """
    Minimizes maximum KV cache pressure using Binary Search with BFD and 
    Guided Local Search with variance reduction.
    """
    
    # --- Data Preparation ---
    # w: Load (req/slo), s: Size, d: Density
    m_data = []
    for m in models:
        w = m.req_rate / m.slo
        s = m.model_size
        d = w / s if s > 1e-6 else 0
        m_data.append({'w': w, 's': s, 'd': d, 'obj': m})

    # --- Phase 1: Binary Search for Feasible K ---
    
    # Helper: Pack items into bins with max pressure constraint K
    # Constraint: (L + w) / (C - (U + s)) <= K  => (L + w) + K*(U + s) <= K*C
    # We use Best-Fit Decreasing on the 'virtual size' metric: w + K*s
    
    def check_feasible(k_target, items_sorted):
        # bins state: (current_L, current_U) represented by parallel arrays
        bin_l = [0.0] * gpu_num
        bin_u = [0.0] * gpu_num
        allocation = [[] for _ in range(gpu_num)]
        
        target_cap = k_target * GPU_MEM_SIZE
        
        for item in items_sorted:
            w, s = item['w'], item['s']
            
            best_g = -1
            min_slack = float('inf')
            
            for g in range(gpu_num):
                # Hard memory constraint check first
                if bin_u[g] + s > GPU_MEM_SIZE - 1e-6: continue
                
                # Pressure constraint (Transformed)
                lhs = (bin_l[g] + w) + k_target * (bin_u[g] + s)
                
                # Check if fits within K constraint
                if lhs <= target_cap + 1e-7:
                    # Best Fit: minimize slack (remaining capacity in terms of K)
                    slack = target_cap - lhs
                    if slack < min_slack:
                        min_slack = slack
                        best_g = g
            
            if best_g != -1:
                allocation[best_g].append(item['obj'])
                bin_l[best_g] += w
                bin_u[best_g] += s
            else:
                return None
                
        return allocation

    # Initial bounds
    total_w = sum(x['w'] for x in m_data)
    total_s = sum(x['s'] for x in m_data)
    total_rem = (gpu_num * GPU_MEM_SIZE) - total_s
    
    low = total_w / total_rem if total_rem > 1e-6 else 0.0
    high = 1000.0 # Safety fallback
    
    # Try a quick density pack to get a tighter upper bound
    def heuristic_density_pack():
        sorted_items = sorted(m_data, key=lambda x: x['d'], reverse=True)
        alloc = [[] for _ in range(gpu_num)]
        l = [0.0]*gpu_num
        u = [0.0]*gpu_num
        for item in sorted_items:
            best_g = -1
            best_p = float('inf')
            for g in range(gpu_num):
                rem = GPU_MEM_SIZE - u[g] - item['s']
                if rem > 1e-6:
                    p = (l[g] + item['w']) / rem
                    if p < best_p:
                        best_p = p
                        best_g = g
            if best_g == -1: return None, float('inf')
            alloc[best_g].append(item['obj'])
            l[best_g] += item['w']
            u[best_g] += item['s']
        
        max_p = 0.0
        for g in range(gpu_num):
            rem = GPU_MEM_SIZE - u[g]
            p = l[g]/rem if rem > 1e-6 else (float('inf') if l[g]>0 else 0)
            max_p = max(max_p, p)
        return alloc, max_p

    init_sol, init_k = heuristic_density_pack()
    if init_sol:
        high = init_k
        final_placement = init_sol
    else:
        final_placement = None

    # Binary Search Execution
    for _ in range(20):
        if high - low < 1e-4: break
        mid = (low + high) / 2.0
        
        feasible_alloc = None
        
        # Strategies:
        # 1. Virtual Size (w + Ks) - theoretically most aligned with constraint
        # 2. Physical Size (s) - good for memory packing
        # 3. Density (d) - good for balancing pressure
        strategies = [
            lambda x: x['w'] + mid * x['s'],
            lambda x: x['s'],
            lambda x: x['d']
        ]
        
        for key in strategies:
            items_sorted = sorted(m_data, key=key, reverse=True)
            res = check_feasible(mid, items_sorted)
            if res:
                feasible_alloc = res
                break
        
        # Stochastic retry if deterministic sort fails
        if not feasible_alloc:
            items_copy = list(m_data)
            for _ in range(5):
                random.shuffle(items_copy)
                res = check_feasible(mid, items_copy)
                if res:
                    feasible_alloc = res
                    break
        
        if feasible_alloc:
            high = mid
            final_placement = feasible_alloc
        else:
            low = mid

    if not final_placement:
        if init_sol: final_placement = init_sol
        else: 
            # Last ditch attempt with huge K to satisfy feasibility
            items_sorted = sorted(m_data, key=lambda x: x['s'], reverse=True)
            final_placement = check_feasible(1e9, items_sorted)
            if not final_placement:
                raise ValueError("No solution found")

    # --- Phase 2: Guided Local Search ---
    
    current_placement = final_placement
    
    # Initialize state
    loads = [0.0] * gpu_num
    used = [0.0] * gpu_num
    for g in range(gpu_num):
        for m in current_placement[g]:
            loads[g] += m.req_rate / m.slo
            used[g] += m.model_size
            
    def calc_pressure(l, u):
        rem = GPU_MEM_SIZE - u
        if rem <= 1e-6: return float('inf') if l > 1e-6 else 0.0
        return l / rem
        
    pressures = [calc_pressure(loads[g], used[g]) for g in range(gpu_num)]
    
    best_placement = {k: list(v) for k,v in current_placement.items()}
    best_max_p = max(pressures)
    
    # Optimization parameters
    max_iters = 300
    
    for it in range(max_iters):
        # 1. Metrics and Global Update
        current_sum_sq = sum(p*p for p in pressures)
        
        # Sort GPUs to find bottleneck
        gpus_sorted = sorted(range(gpu_num), key=lambda g: pressures[g], reverse=True)
        bottleneck = gpus_sorted[0]
        bn_p = pressures[bottleneck]
        
        if bn_p < best_max_p - 1e-8:
            best_max_p = bn_p
            best_placement = {k: list(v) for k,v in current_placement.items()}
        
        # 2. Guided Moves
        bn_items = current_placement[bottleneck]
        
        best_move = None 
        # (type, partner, bn_idx, pt_idx, nbl, nbu, npl, npu, nmax, nsq)
        
        for partner in range(gpu_num):
            if partner == bottleneck: continue
            
            # Identify max pressure of "other" GPUs for global max calculation
            if partner == gpus_sorted[1]:
                max_others = pressures[gpus_sorted[2]] if len(gpus_sorted) > 2 else 0.0
            else:
                max_others = pressures[gpus_sorted[1]]
                
            base_sq = current_sum_sq - (bn_p**2 + pressures[partner]**2)
            
            # Move Attempt
            for idx, m in enumerate(bn_items):
                w, s = m.req_rate/m.slo, m.model_size
                if used[partner] + s > GPU_MEM_SIZE - 1e-6: continue
                
                nbl = loads[bottleneck] - w
                nbu = used[bottleneck] - s
                npl = loads[partner] + w
                npu = used[partner] + s
                
                p_bn = calc_pressure(nbl, nbu)
                p_pt = calc_pressure(npl, npu)
                
                new_max = max(max_others, p_bn, p_pt)
                
                # Pruning: strict degradation
                if new_max > bn_p + 1e-9: continue
                
                new_sq = base_sq + p_bn**2 + p_pt**2
                
                is_better = False
                if new_max < bn_p - 1e-9: is_better = True
                elif new_max < bn_p + 1e-9 and new_sq < current_sum_sq - 1e-9: is_better = True
                
                if is_better:
                    if best_move is None or new_max < best_move[8] - 1e-9 or (abs(new_max - best_move[8]) < 1e-9 and new_sq < best_move[9]):
                        best_move = ('move', partner, idx, -1, nbl, nbu, npl, npu, new_max, new_sq)
            
            # Swap Attempt (if Move isn't perfect or to check improvement)
            pt_items = current_placement[partner]
            for idx1, m1 in enumerate(bn_items):
                w1, s1 = m1.req_rate/m1.slo, m1.model_size
                for idx2, m2 in enumerate(pt_items):
                    w2, s2 = m2.req_rate/m2.slo, m2.model_size
                    
                    nbu = used[bottleneck] - s1 + s2
                    if nbu > GPU_MEM_SIZE - 1e-6: continue
                    npu = used[partner] - s2 + s1
                    if npu > GPU_MEM_SIZE - 1e-6: continue
                    
                    nbl = loads[bottleneck] - w1 + w2
                    npl = loads[partner] - w2 + w1
                    
                    p_bn = calc_pressure(nbl, nbu)
                    p_pt = calc_pressure(npl, npu)
                    
                    new_max = max(max_others, p_bn, p_pt)
                    
                    if new_max > bn_p + 1e-9: continue
                    
                    new_sq = base_sq + p_bn**2 + p_pt**2
                    
                    is_better = False
                    if new_max < bn_p - 1e-9: is_better = True
                    elif new_max < bn_p + 1e-9 and new_sq < current_sum_sq - 1e-9: is_better = True
                    
                    if is_better:
                        if best_move is None or new_max < best_move[8] - 1e-9 or (abs(new_max - best_move[8]) < 1e-9 and new_sq < best_move[9]):
                            best_move = ('swap', partner, idx1, idx2, nbl, nbu, npl, npu, new_max, new_sq)
                            
        if best_move:
            mtype, pt, i, j, nbl, nbu, npl, npu, _, _ = best_move
            if mtype == 'move':
                # 'i' is the index in current_placement[bottleneck]
                item = current_placement[bottleneck].pop(i)
                current_placement[pt].append(item)
            else:
                item1 = current_placement[bottleneck][i]
                item2 = current_placement[pt][j]
                current_placement[bottleneck][i] = item2
                current_placement[pt][j] = item1
                
            loads[bottleneck], used[bottleneck] = nbl, nbu
            loads[pt], used[pt] = npl, npu
            pressures[bottleneck] = calc_pressure(nbl, nbu)
            pressures[pt] = calc_pressure(npl, npu)
            
        else:
            # 3. Perturbation (Ruin & Recreate)
            victims = {bottleneck}
            others = [g for g in range(gpu_num) if g != bottleneck]
            if not others: break
            
            # Select 2 random partners
            victims.update(random.sample(others, min(2, len(others))))
            v_list = list(victims)
            
            repack_models = []
            for v in v_list:
                repack_models.extend(current_placement[v])
                current_placement[v] = []
                loads[v] = 0.0
                used[v] = 0.0
                pressures[v] = 0.0
                
            # Sort by density with noise
            repack_models.sort(key=lambda x: (x.req_rate/x.slo)/(x.model_size+1e-6) * random.uniform(0.9, 1.1), reverse=True)
            
            success = True
            for m in repack_models:
                w, s = m.req_rate/m.slo, m.model_size
                best_v = -1
                best_p = float('inf')
                
                # Greedy fit
                for v in v_list:
                    rem = GPU_MEM_SIZE - used[v] - s
                    if rem > 1e-6:
                        p = (loads[v] + w) / rem
                        if p < best_p:
                            best_p = p
                            best_v = v
                
                # Fallback fit
                if best_v == -1:
                    for v in v_list:
                        if used[v] + s <= GPU_MEM_SIZE - 1e-6:
                            best_v = v
                            break
                            
                if best_v != -1:
                    current_placement[best_v].append(m)
                    loads[best_v] += w
                    used[best_v] += s
                else:
                    success = False
                    break
            
            if success:
                for v in v_list:
                    pressures[v] = calc_pressure(loads[v], used[v])
            else:
                # Revert
                current_placement = {k: list(v) for k,v in best_placement.items()}
                for g in range(gpu_num):
                    l, u = 0.0, 0.0
                    for m in current_placement[g]:
                        l += m.req_rate/m.slo
                        u += m.model_size
                    loads[g] = l
                    used[g] = u
                    pressures[g] = calc_pressure(l, u)
                if it > max_iters * 0.9: break

    return best_placement
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