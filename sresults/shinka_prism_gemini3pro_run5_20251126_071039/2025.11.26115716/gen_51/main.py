# EVOLVE-BLOCK-START
"""Model placement algorithm for minimizing maximum KV cache pressure across GPUs"""

import random

GPU_MEM_SIZE = 80  # GB

def compute_model_placement(gpu_num, models):
    """
    Minimizes max KVPR using Binary Search Initialization followed by 
    Guided First-Improvement Variable Neighborhood Descent.
    """
    rng = random.Random(42)

    # 1. Precompute Data
    m_data = []
    for i, m in enumerate(models):
        w = m.req_rate / m.slo
        s = m.model_size
        m_data.append({
            'w': w, 's': s, 'obj': m,
            'd': w/s if s > 1e-6 else 0,
            'id': i
        })

    # 2. Binary Search for Feasible K
    def check_feasible(k, items):
        # Strategies: Effective Size (w + k*s), Density, Physical Size
        strategies = [
            lambda x: x['w'] + k * x['s'],
            lambda x: x['d'],
            lambda x: x['s']
        ]
        
        for key_fn in strategies:
            sorted_items = sorted(items, key=key_fn, reverse=True)
            placement = [[] for _ in range(gpu_num)]
            g_l = [0.0] * gpu_num
            g_u = [0.0] * gpu_num
            possible = True
            
            for item in sorted_items:
                w, s = item['w'], item['s']
                best_g = -1
                min_slack = float('inf')
                
                eff_size = w + k * s
                max_cap = k * GPU_MEM_SIZE
                
                for g in range(gpu_num):
                    if g_u[g] + s >= GPU_MEM_SIZE - 1e-6: continue
                    
                    curr_eff = g_l[g] + k * g_u[g]
                    if curr_eff + eff_size <= max_cap + 1e-7:
                        slack = max_cap - (curr_eff + eff_size)
                        if slack < min_slack:
                            min_slack = slack
                            best_g = g
                
                if best_g != -1:
                    placement[best_g].append(item)
                    g_l[best_g] += w
                    g_u[best_g] += s
                else:
                    possible = False
                    break
            
            if possible:
                return placement
        return None

    # Heuristic Upper Bound
    def get_greedy_upper_bound():
        sorted_items = sorted(m_data, key=lambda x: x['d'], reverse=True)
        g_l = [0.0]*gpu_num
        g_u = [0.0]*gpu_num
        
        for item in sorted_items:
            best_g = -1
            best_p = float('inf')
            for g in range(gpu_num):
                rem = GPU_MEM_SIZE - g_u[g] - item['s']
                if rem > 1e-5:
                    p = (g_l[g] + item['w']) / rem
                    if p < best_p:
                        best_p = p
                        best_g = g
            if best_g == -1: return 2000.0
            g_l[best_g] += item['w']
            g_u[best_g] += item['s']
            
        max_p = 0.0
        for g in range(gpu_num):
            rem = GPU_MEM_SIZE - g_u[g]
            if rem <= 1e-6:
                if g_l[g] > 0: return 2000.0
            else:
                max_p = max(max_p, g_l[g]/rem)
        return max_p

    upper_bound = get_greedy_upper_bound()
    low, high = 0.0, upper_bound
    final_struct = None
    
    for _ in range(20):
        if high - low < 1e-4: break
        mid = (low + high) / 2.0
        res = check_feasible(mid, m_data)
        if res:
            final_struct = res
            high = mid
        else:
            low = mid
            
    if final_struct is None:
        final_struct = check_feasible(high + 10.0, m_data)
        if final_struct is None:
             raise ValueError("Could not find feasible placement")

    # 3. Guided Local Search
    placement = {g: [x['obj'] for x in items] for g, items in enumerate(final_struct)}
    g_loads = [sum(m.req_rate/m.slo for m in placement[g]) for g in range(gpu_num)]
    g_used = [sum(m.model_size for m in placement[g]) for g in range(gpu_num)]
    
    def get_p(l, u):
        rem = GPU_MEM_SIZE - u
        if rem <= 1e-6: return float('inf') if l > 1e-6 else 0.0
        return l / rem

    g_pressures = [get_p(g_loads[g], g_used[g]) for g in range(gpu_num)]
    best_placement = {k: list(v) for k, v in placement.items()}
    best_max_p = max(g_pressures)
    
    max_iters = 300
    
    for it in range(max_iters):
        curr_max = max(g_pressures)
        curr_sq = sum(p*p for p in g_pressures)
        
        if curr_max < best_max_p - 1e-8:
            best_max_p = curr_max
            best_placement = {k: list(v) for k, v in placement.items()}
        
        bottleneck = -1
        max_p_val = -1.0
        for g in range(gpu_num):
            if g_pressures[g] > max_p_val:
                max_p_val = g_pressures[g]
                bottleneck = g
        
        if max_p_val < 1e-6: break 
        
        # Sort partners: Emptiest (lowest pressure) first
        partners = sorted([g for g in range(gpu_num) if g != bottleneck], key=lambda x: g_pressures[x])
        
        # Sort bottleneck items: Largest Load first
        bn_items = []
        for i, m in enumerate(placement[bottleneck]):
            bn_items.append((i, m))
        bn_items.sort(key=lambda x: x[1].req_rate/x[1].slo, reverse=True)
        
        found_improvement = False
        
        # Precompute top pressures for fast "max_others"
        sorted_p = sorted(g_pressures, reverse=True)
        p_top1, p_top2 = sorted_p[0], sorted_p[1]
        
        for partner in partners:
            if (g_pressures[bottleneck] == p_top1 and g_pressures[partner] == p_top2) or \
               (g_pressures[bottleneck] == p_top2 and g_pressures[partner] == p_top1):
                max_others = sorted_p[2] if len(sorted_p) > 2 else 0.0
            elif g_pressures[bottleneck] == p_top1 or g_pressures[partner] == p_top1:
                max_others = p_top2
            else:
                max_others = p_top1
            
            sq_base = curr_sq - (g_pressures[bottleneck]**2 + g_pressures[partner]**2)
            
            # Move (Shift)
            for idx, m in bn_items:
                w, s = m.req_rate/m.slo, m.model_size
                
                if g_used[partner] + s >= GPU_MEM_SIZE - 1e-6: continue
                
                n_bl = g_loads[bottleneck] - w
                n_bu = g_used[bottleneck] - s
                n_pl = g_loads[partner] + w
                n_pu = g_used[partner] + s
                
                pb = get_p(n_bl, n_bu)
                pp = get_p(n_pl, n_pu)
                
                nm = max(max_others, pb, pp)
                
                if nm > curr_max + 1e-9: continue
                
                nsq = sq_base + pb**2 + pp**2
                
                is_better = False
                if nm < curr_max - 1e-9: is_better = True
                elif nm < curr_max + 1e-9 and nsq < curr_sq - 1e-9: is_better = True
                
                if is_better:
                    item = placement[bottleneck].pop(idx)
                    placement[partner].append(item)
                    g_loads[bottleneck], g_used[bottleneck] = n_bl, n_bu
                    g_loads[partner], g_used[partner] = n_pl, n_pu
                    g_pressures[bottleneck], g_pressures[partner] = pb, pp
                    found_improvement = True
                    break
            if found_improvement: break
            
            # Swap
            pt_items = placement[partner]
            for idx1, m1 in bn_items:
                w1, s1 = m1.req_rate/m1.slo, m1.model_size
                for idx2, m2 in enumerate(pt_items):
                    w2, s2 = m2.req_rate/m2.slo, m2.model_size
                    
                    n_bu = g_used[bottleneck] - s1 + s2
                    if n_bu >= GPU_MEM_SIZE - 1e-6: continue
                    n_pu = g_used[partner] - s2 + s1
                    if n_pu >= GPU_MEM_SIZE - 1e-6: continue
                    
                    n_bl = g_loads[bottleneck] - w1 + w2
                    n_pl = g_loads[partner] - w2 + w1
                    
                    pb = get_p(n_bl, n_bu)
                    pp = get_p(n_pl, n_pu)
                    
                    nm = max(max_others, pb, pp)
                    if nm > curr_max + 1e-9: continue
                    nsq = sq_base + pb**2 + pp**2
                    
                    is_better = False
                    if nm < curr_max - 1e-9: is_better = True
                    elif nm < curr_max + 1e-9 and nsq < curr_sq - 1e-9: is_better = True
                    
                    if is_better:
                        placement[bottleneck][idx1], placement[partner][idx2] = placement[partner][idx2], placement[bottleneck][idx1]
                        g_loads[bottleneck], g_used[bottleneck] = n_bl, n_bu
                        g_loads[partner], g_used[partner] = n_pl, n_pu
                        g_pressures[bottleneck], g_pressures[partner] = pb, pp
                        found_improvement = True
                        break
                if found_improvement: break
            if found_improvement: break
            
        if not found_improvement:
            # Burst Perturbation: Bottleneck + 2 Least Loaded
            k = min(len(partners), 2)
            victims = [bottleneck] + partners[:k]
            
            backup_state = []
            for v in victims:
                backup_state.append((v, list(placement[v]), g_loads[v], g_used[v], g_pressures[v]))
            
            repack_items = []
            for v in victims:
                repack_items.extend(placement[v])
                placement[v] = []
                g_loads[v] = 0.0
                g_used[v] = 0.0
            
            success_repack = False
            for _ in range(5):
                iter_items = list(repack_items)
                random.shuffle(iter_items)
                iter_items.sort(key=lambda x: ((x.req_rate/x.slo)/(x.model_size+1e-6)) * rng.uniform(0.85, 1.15), reverse=True)
                
                l_loads = {v: 0.0 for v in victims}
                l_used = {v: 0.0 for v in victims}
                l_alloc = {v: [] for v in victims}
                possible = True
                
                for m in iter_items:
                    w, s = m.req_rate/m.slo, m.model_size
                    best_v = -1
                    best_score = float('inf')
                    for v in victims:
                        rem = GPU_MEM_SIZE - l_used[v] - s
                        if rem > 1e-6:
                            p = (l_loads[v] + w) / rem
                            if p < best_score:
                                best_score = p
                                best_v = v
                    if best_v == -1:
                        for v in victims:
                            if l_used[v] + s <= GPU_MEM_SIZE - 1e-6:
                                best_v = v
                                break
                    if best_v != -1:
                        l_alloc[best_v].append(m)
                        l_loads[best_v] += w
                        l_used[best_v] += s
                    else:
                        possible = False
                        break
                
                if possible:
                    for v in victims:
                        placement[v] = l_alloc[v]
                        g_loads[v] = l_loads[v]
                        g_used[v] = l_used[v]
                        g_pressures[v] = get_p(g_loads[v], g_used[v])
                    success_repack = True
                    break
            
            if not success_repack:
                for v, items, l, u, p in backup_state:
                    placement[v] = list(items)
                    g_loads[v] = l
                    g_used[v] = u
                    g_pressures[v] = p
                if it > max_iters * 0.8: break
                
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