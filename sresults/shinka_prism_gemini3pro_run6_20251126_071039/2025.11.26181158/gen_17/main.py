# EVOLVE-BLOCK-START
"""Model placement algorithm for minimizing maximum KV cache pressure across GPUs"""

import random

GPU_MEM_SIZE = 80  # GB

def compute_model_placement(gpu_num, models):
    """
    Compute a model placement that minimizes the maximum KVPR across all GPUs.
    
    Combines Binary Search for initialization with Iterated Local Search (ILS)
    using Move, Swap(1-1), and Swap(2-1) neighborhoods.
    """
    
    # Pre-calculate model properties for efficiency
    model_data = []
    for m in models:
        model_data.append({
            'model': m,
            'l': m.req_rate / m.slo,
            's': m.model_size
        })

    # --- Phase 1: Construction (Binary Search) ---
    # We look for the smallest K such that all models fit with KVPR <= K.
    
    def solve_packing(target_k):
        """
        Tries to pack models such that sum(L)/(M-sum(S)) <= K.
        Linearized constraint: sum(L) + K*sum(S) <= K*M.
        """
        capacity = target_k * GPU_MEM_SIZE
        
        # Heuristics for sorting items
        # 1. Linearized Weight: balances L and S based on K
        s1 = sorted(model_data, key=lambda x: x['l'] + target_k * x['s'], reverse=True)
        # 2. Size: Good for memory-bound
        s2 = sorted(model_data, key=lambda x: x['s'], reverse=True)
        # 3. Load: Good for load-bound
        s3 = sorted(model_data, key=lambda x: x['l'], reverse=True)
        # 4. Density: Load/Size
        s4 = sorted(model_data, key=lambda x: x['l'] / x['s'] if x['s'] > 0 else 0, reverse=True)
        
        strategies = [s1, s2, s3, s4]
        
        for items in strategies:
            gpu_l = [0.0] * gpu_num
            gpu_s = [0.0] * gpu_num
            gpu_models = [[] for _ in range(gpu_num)]
            
            feasible = True
            for item in items:
                # Best Fit
                best_idx = -1
                min_rem = float('inf')
                
                w = item['l'] + target_k * item['s']
                
                for i in range(gpu_num):
                    if gpu_s[i] + item['s'] >= GPU_MEM_SIZE - 1e-6:
                        continue
                    
                    curr_w = gpu_l[i] + target_k * gpu_s[i]
                    if curr_w + w <= capacity + 1e-9:
                        # Remaining capacity in linearized terms
                        rem = capacity - (curr_w + w)
                        if rem < min_rem:
                            min_rem = rem
                            best_idx = i
                
                if best_idx != -1:
                    gpu_l[best_idx] += item['l']
                    gpu_s[best_idx] += item['s']
                    gpu_models[best_idx].append(item['model'])
                else:
                    feasible = False
                    break
            
            if feasible:
                return gpu_models
        return None

    # Binary search
    low = 0.0
    high = 1.0
    best_init = None
    
    # Exponential search for upper bound
    for _ in range(20):
        if solve_packing(high) is not None:
            break
        low = high
        high *= 2.0
    else:
        high = 1e9 # Should always fit if physically possible

    # Refine K
    for _ in range(30):
        mid = (low + high) / 2
        res = solve_packing(mid)
        if res:
            best_init = res
            high = mid
        else:
            low = mid
            
    if best_init is None:
        best_init = solve_packing(high)
        if best_init is None:
             raise ValueError("Unable to find a valid placement.")

    # Convert to mutable structure for Local Search
    placement = {i: best_init[i] for i in range(gpu_num)}

    # --- Phase 2: Iterated Local Search ---
    
    def get_kvpr(l, s):
        if s >= GPU_MEM_SIZE - 1e-7: return 1e15
        return l / (GPU_MEM_SIZE - s)

    def local_search(plc):
        # State: list of lists of models
        state = [list(plc[i]) for i in range(gpu_num)]
        # Track Load (l) and Size (s) for each GPU
        l_vec = [sum(m.req_rate/m.slo for m in state[g]) for g in range(gpu_num)]
        s_vec = [sum(m.model_size for m in state[g]) for g in range(gpu_num)]
        
        # Compute current max KVPR
        cur_max_k = max(get_kvpr(l_vec[g], s_vec[g]) for g in range(gpu_num))
        
        iterations = 0
        while iterations < 150: # Cap iterations
            iterations += 1
            if cur_max_k < 1e-9: break
            
            # Identify bottleneck GPU(s)
            candidates = [g for g in range(gpu_num) if get_kvpr(l_vec[g], s_vec[g]) >= cur_max_k - 1e-6]
            if not candidates: break
            
            bottleneck = candidates[0]
            src_l = l_vec[bottleneck]
            src_s = s_vec[bottleneck]
            
            best_move = None
            # We want to maximize the reduction of the bottleneck's pressure
            # without making another GPU worse than the current max.
            best_improvement = 0.0 
            
            # 1. Move
            for i, m in enumerate(state[bottleneck]):
                m_l = m.req_rate/m.slo
                m_s = m.model_size
                
                for tgt in range(gpu_num):
                    if tgt == bottleneck: continue
                    if get_kvpr(l_vec[tgt], s_vec[tgt]) >= cur_max_k: continue # Skip if target is bad
                    if s_vec[tgt] + m_s >= GPU_MEM_SIZE: continue
                    
                    new_src_k = get_kvpr(src_l - m_l, src_s - m_s)
                    new_tgt_k = get_kvpr(l_vec[tgt] + m_l, s_vec[tgt] + m_s)
                    
                    new_global = max(new_src_k, new_tgt_k)
                    if new_global < cur_max_k - 1e-7:
                        imp = cur_max_k - new_global
                        if imp > best_improvement:
                            best_improvement = imp
                            best_move = ('move', bottleneck, i, tgt)

            # 2. Swap (1-1)
            for i, m1 in enumerate(state[bottleneck]):
                m1_l = m1.req_rate/m1.slo
                m1_s = m1.model_size
                
                for tgt in range(gpu_num):
                    if tgt == bottleneck: continue
                    if get_kvpr(l_vec[tgt], s_vec[tgt]) >= cur_max_k: continue
                    
                    for j, m2 in enumerate(state[tgt]):
                        m2_l = m2.req_rate/m2.slo
                        m2_s = m2.model_size
                        
                        # Verify capacity
                        if src_s - m1_s + m2_s >= GPU_MEM_SIZE: continue
                        if s_vec[tgt] - m2_s + m1_s >= GPU_MEM_SIZE: continue
                        
                        new_src_k = get_kvpr(src_l - m1_l + m2_l, src_s - m1_s + m2_s)
                        new_tgt_k = get_kvpr(l_vec[tgt] - m2_l + m1_l, s_vec[tgt] - m2_s + m1_s)
                        
                        new_global = max(new_src_k, new_tgt_k)
                        if new_global < cur_max_k - 1e-7:
                            imp = cur_max_k - new_global
                            if imp > best_improvement:
                                best_improvement = imp
                                best_move = ('swap', bottleneck, i, tgt, j)

            # 3. Swap (2-1): 2 from bottleneck, 1 from target
            # Only check if we have enough models
            if len(state[bottleneck]) >= 2:
                for i1 in range(len(state[bottleneck])):
                    for i2 in range(i1 + 1, len(state[bottleneck])):
                        m1 = state[bottleneck][i1]
                        m2 = state[bottleneck][i2]
                        
                        pair_l = (m1.req_rate/m1.slo) + (m2.req_rate/m2.slo)
                        pair_s = m1.model_size + m2.model_size
                        
                        for tgt in range(gpu_num):
                            if tgt == bottleneck: continue
                            if get_kvpr(l_vec[tgt], s_vec[tgt]) >= cur_max_k: continue
                            
                            for j, m3 in enumerate(state[tgt]):
                                m3_l = m3.req_rate/m3.slo
                                m3_s = m3.model_size
                                
                                # Capacity
                                if src_s - pair_s + m3_s >= GPU_MEM_SIZE: continue
                                if s_vec[tgt] - m3_s + pair_s >= GPU_MEM_SIZE: continue
                                
                                new_src_k = get_kvpr(src_l - pair_l + m3_l, src_s - pair_s + m3_s)
                                new_tgt_k = get_kvpr(l_vec[tgt] - m3_l + pair_l, s_vec[tgt] - m3_s + pair_s)
                                
                                new_global = max(new_src_k, new_tgt_k)
                                if new_global < cur_max_k - 1e-7:
                                    imp = cur_max_k - new_global
                                    if imp > best_improvement:
                                        best_improvement = imp
                                        best_move = ('swap21', bottleneck, i1, i2, tgt, j)

            if best_move:
                mtype = best_move[0]
                if mtype == 'move':
                    _, b, i, t = best_move
                    m = state[b].pop(i)
                    state[t].append(m)
                    
                    l_vec[b] -= m.req_rate/m.slo
                    s_vec[b] -= m.model_size
                    l_vec[t] += m.req_rate/m.slo
                    s_vec[t] += m.model_size
                    
                elif mtype == 'swap':
                    _, b, i, t, j = best_move
                    m1 = state[b][i]
                    m2 = state[t][j]
                    state[b][i] = m2
                    state[t][j] = m1
                    
                    diff_l = (m2.req_rate/m2.slo) - (m1.req_rate/m1.slo)
                    diff_s = m2.model_size - m1.model_size
                    l_vec[b] += diff_l
                    s_vec[b] += diff_s
                    l_vec[t] -= diff_l
                    s_vec[t] -= diff_s
                    
                elif mtype == 'swap21':
                    _, b, i1, i2, t, j = best_move
                    # i1 < i2, pop larger index first
                    m2 = state[b].pop(i2)
                    m1 = state[b].pop(i1)
                    m3 = state[t].pop(j)
                    
                    state[b].append(m3)
                    state[t].append(m1)
                    state[t].append(m2)
                    
                    pair_l = (m1.req_rate/m1.slo) + (m2.req_rate/m2.slo)
                    pair_s = m1.model_size + m2.model_size
                    m3_l = m3.req_rate/m3.slo
                    m3_s = m3.model_size
                    
                    l_vec[b] = l_vec[b] - pair_l + m3_l
                    s_vec[b] = s_vec[b] - pair_s + m3_s
                    l_vec[t] = l_vec[t] - m3_l + pair_l
                    s_vec[t] = s_vec[t] - m3_s + pair_s
                
                # Recompute global max
                cur_max_k = max(get_kvpr(l_vec[g], s_vec[g]) for g in range(gpu_num))
            else:
                # Local optima reached
                break
        
        return {i: state[i] for i in range(gpu_num)}, cur_max_k

    # ILS Driver
    current_plc = placement
    current_score = 0
    # Calculate initial score
    init_stats = [get_kvpr(sum(m.req_rate/m.slo for m in current_plc[g]), 
                          sum(m.model_size for m in current_plc[g])) for g in range(gpu_num)]
    current_score = max(init_stats)
    
    best_plc = current_plc
    best_score = current_score
    
    # 5 Restarts with perturbation
    for _ in range(5):
        # Local Search
        refined_plc, refined_score = local_search(current_plc)
        
        if refined_score < best_score:
            best_score = refined_score
            best_plc = refined_plc
            
        current_plc = refined_plc
        current_score = refined_score
        
        # Perturbation: Move a model from bottleneck to a random feasible GPU
        # Find bottleneck in current solution
        stats = []
        max_k = -1
        b_gpu = -1
        for g in range(gpu_num):
            l = sum(m.req_rate/m.slo for m in current_plc[g])
            s = sum(m.model_size for m in current_plc[g])
            k = get_kvpr(l, s)
            stats.append({'l':l, 's':s, 'k':k})
            if k > max_k:
                max_k = k
                b_gpu = g
        
        if max_k == 0: break
        
        # Try to move a random model from bottleneck
        if b_gpu != -1 and current_plc[b_gpu]:
            # Convert to list for mutation
            mut_state = [list(current_plc[g]) for g in range(gpu_num)]
            
            # Pick a model to move
            idxs = list(range(len(mut_state[b_gpu])))
            random.shuffle(idxs)
            moved = False
            for idx in idxs:
                m = mut_state[b_gpu][idx]
                # Try random targets
                targets = list(range(gpu_num))
                random.shuffle(targets)
                for t in targets:
                    if t == b_gpu: continue
                    t_s = stats[t]['s']
                    if t_s + m.model_size < GPU_MEM_SIZE:
                        # Apply perturbation
                        mut_state[b_gpu].pop(idx)
                        mut_state[t].append(m)
                        moved = True
                        break
                if moved: break
            
            if moved:
                current_plc = {g: mut_state[g] for g in range(gpu_num)}
            else:
                # If cannot move, maybe try random swap? 
                # For now just continue, next local search might find something or just end.
                pass

    return best_plc

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