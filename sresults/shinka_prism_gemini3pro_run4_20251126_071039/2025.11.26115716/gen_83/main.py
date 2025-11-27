# EVOLVE-BLOCK-START
"""Model placement algorithm for minimizing maximum KV cache pressure across GPUs"""

import random
import math

GPU_MEM_SIZE = 80.0  # GB

def compute_model_placement(gpu_num, models):
    """
    Minimizes max KVPR using a hybrid approach:
    1. Ensemble of Greedy Heuristics (Best Fit / Worst Fit) for initial bounds.
    2. Binary Search on Pressure K using a feasibility oracle (Bin Packing).
       - The oracle uses adaptive stochastic search with failure memory.
       - Combines Best Fit and Worst Fit placement strategies.
    3. Aggressive Local Search to refine the best found placement.
    """

    # ------------------------------------------------------------------
    # 0. Data Preprocessing
    # ------------------------------------------------------------------
    m_data = []
    total_w = 0.0
    total_s = 0.0
    for i, m in enumerate(models):
        w = m.req_rate / m.slo
        s = m.model_size
        m_data.append({'w': w, 's': s, 'obj': m, 'id': i})
        total_w += w
        total_s += s

    # Calculate actual max KVPR for a given placement dictionary {gpu_idx: [item_indices]}
    def calc_score(placement_indices):
        max_p = 0.0
        for idxs in placement_indices.values():
            w_sum = sum(m_data[i]['w'] for i in idxs)
            s_sum = sum(m_data[i]['s'] for i in idxs)
            rem = GPU_MEM_SIZE - s_sum
            
            if rem <= 1e-9:
                if w_sum > 0: return float('inf')
                else: continue
            
            p = w_sum / rem
            if p > max_p: max_p = p
        return max_p

    best_sol_indices = None
    best_val = float('inf')

    # ------------------------------------------------------------------
    # 1. Phase I: Diverse Greedy Heuristics
    # ------------------------------------------------------------------
    # We use both Best Fit (minimize fragmentation) and Worst Fit (load balancing).
    
    # Sort keys: 
    # 1. Asymptotic Pressure: w / (C - s)
    # 2. Sum: w + s (heuristic proxy)
    # 3. Size: s
    # 4. Weight: w
    
    greedy_configs = [
        # (Key Function, Reverse, Placement Strategy)
        (lambda x: x['w']/(GPU_MEM_SIZE - x['s'] + 1e-6), True, 'best_fit'),
        (lambda x: x['w']/(GPU_MEM_SIZE - x['s'] + 1e-6), True, 'worst_fit'),
        (lambda x: x['s'], True, 'best_fit'),
        (lambda x: x['w'], True, 'worst_fit'),
    ]

    for key_fn, rev, strat in greedy_configs:
        sorted_indices = sorted(range(len(m_data)), key=lambda i: key_fn(m_data[i]), reverse=rev)
        
        bins = [{'w': 0.0, 's': 0.0, 'idxs': []} for _ in range(gpu_num)]
        possible = True
        
        for idx in sorted_indices:
            item = m_data[idx]
            
            # Metric: For greedy phase, we minimize the resulting local pressure of the target bin
            # Strategy determines tie-breaking or search order, but here we strictly look at the objective.
            
            candidate_bin = None
            min_resulting_p = float('inf')
            
            for b_idx in range(gpu_num):
                if bins[b_idx]['s'] + item['s'] > GPU_MEM_SIZE: continue
                new_s = bins[b_idx]['s'] + item['s']
                new_w = bins[b_idx]['w'] + item['w']
                rem = GPU_MEM_SIZE - new_s
                p = new_w / rem if rem > 1e-9 else (float('inf') if new_w > 0 else 0.0)
                
                # Logic: Find best bin according to strategy
                # For this specific KVPR problem, both strategies aim to minimize the resulting P,
                # but handle tie-breaking or secondary objectives differently.
                # Here we stick to a simple minimization loop but allow secondary logic.
                
                if p < min_resulting_p:
                    min_resulting_p = p
                    candidate_bin = b_idx
                elif strat == 'worst_fit' and abs(p - min_resulting_p) < 1e-6:
                    # Tie-break: if pressure is same (e.g. 0), pick one with MORE remaining space (WF)
                    # or LESS remaining space (BF)
                    cand_rem = GPU_MEM_SIZE - (bins[candidate_bin]['s'] + item['s'])
                    curr_rem_val = GPU_MEM_SIZE - new_s
                    if curr_rem_val > cand_rem:
                        candidate_bin = b_idx

            if candidate_bin is None:
                possible = False
                break
            
            bins[candidate_bin]['idxs'].append(idx)
            bins[candidate_bin]['w'] += item['w']
            bins[candidate_bin]['s'] += item['s']

        if possible:
            sol = {i: bins[i]['idxs'] for i in range(gpu_num)}
            score = calc_score(sol)
            if score < best_val:
                best_val = score
                best_sol_indices = sol

    # ------------------------------------------------------------------
    # 2. Phase II: Binary Search with Robust Feasibility Check
    # ------------------------------------------------------------------
    rem_global = gpu_num * GPU_MEM_SIZE - total_s
    lb = total_w / rem_global if rem_global > 1e-6 else 0.0
    if best_val < lb: lb = 0.0
    
    low = lb
    high = best_val if best_val != float('inf') else 5000.0

    # Feasibility Oracle
    def try_pack(target_k, effort_level=1.0):
        # target_k constraint: s + w/K <= C  (Effective size <= C)
        # We perform multiple stochastic passes.
        
        # Adaptive number of trials
        num_trials = int(30 * effort_level)
        if num_trials < 5: num_trials = 5
        
        fail_counts = {i: 0 for i in range(len(m_data))}
        
        # Deterministic First Pass: Sort by Effective Size Descending
        # Strategy rotation:
        # 0: Effective Size (s + w/K)
        # 1: Physical Size (s)
        # 2: Weight (w)
        # 3: Density (w/s)
        
        for attempt in range(num_trials):
            # 1. Determine Sorting Strategy
            strat = attempt % 4
            
            # 2. Determine Bin Selection Strategy (Best Fit vs Worst Fit)
            # Mostly Best Fit (tight packing), occasionally Worst Fit (load balancing)
            use_worst_fit = (attempt % 5 == 4) 
            
            # 3. Noise
            noise = 0.0
            if attempt >= 4:
                noise = 0.01 + (attempt * 0.003)
            
            def sort_key(idx):
                item = m_data[idx]
                if strat == 0:
                    val = item['s'] + item['w'] / target_k
                elif strat == 1:
                    val = item['s']
                elif strat == 2:
                    val = item['w']
                else:
                    val = item['w'] / (item['s'] + 1e-6)
                
                if noise > 0:
                    val *= random.uniform(1.0 - noise, 1.0 + noise)
                
                # Failure Memory Boost
                if fail_counts[idx] > 0:
                    val += fail_counts[idx] * 1e9
                return val
            
            indices = sorted(range(len(m_data)), key=sort_key, reverse=True)
            
            # Packing
            bins_eff = [0.0] * gpu_num
            bins_s = [0.0] * gpu_num
            placement = [[] for _ in range(gpu_num)]
            possible_iter = True
            first_fail_idx = None
            
            for idx in indices:
                item = m_data[idx]
                eff_size = item['s'] + item['w'] / target_k
                
                best_b = None
                best_metric = float('inf') if not use_worst_fit else -1.0
                
                for b in range(gpu_num):
                    if bins_s[b] + item['s'] > GPU_MEM_SIZE: continue
                    
                    if bins_eff[b] + eff_size <= GPU_MEM_SIZE + 1e-6:
                        rem_eff = GPU_MEM_SIZE - (bins_eff[b] + eff_size)
                        
                        if not use_worst_fit:
                            # Best Fit: Minimize remaining effective space
                            if rem_eff < best_metric:
                                best_metric = rem_eff
                                best_b = b
                        else:
                            # Worst Fit: Maximize remaining effective space
                            if rem_eff > best_metric:
                                best_metric = rem_eff
                                best_b = b
                
                if best_b is None:
                    possible_iter = False
                    first_fail_idx = idx
                    break
                
                placement[best_b].append(idx)
                bins_eff[best_b] += eff_size
                bins_s[best_b] += item['s']
            
            if possible_iter:
                return {i: placement[i] for i in range(gpu_num)}
            else:
                if first_fail_idx is not None:
                    fail_counts[first_fail_idx] += 1
        
        return None

    if high > low + 1e-4:
        # Binary Search
        for iteration in range(20):
            mid = (low + high) / 2.0
            
            # Increase effort as we narrow down or if iterations progress
            # Early iterations: loose bounds, check quickly.
            # Late iterations: tight bounds, check thoroughly.
            # Actually, the hardest checks are when K is near optimal.
            effort = 1.0 + (iteration * 0.1)
            
            sol = try_pack(mid, effort_level=effort)
            
            if sol:
                actual_score = calc_score(sol)
                if actual_score < best_val:
                    best_val = actual_score
                    best_sol_indices = sol
                high = mid
            else:
                low = mid

    if best_sol_indices is None:
        raise ValueError("Unable to place models on GPUs with available memory.")

    # ------------------------------------------------------------------
    # 3. Phase III: Extended Local Search
    # ------------------------------------------------------------------
    # Convert to mutable structure
    curr_sol = {g: list(idxs) for g, idxs in best_sol_indices.items()}
    
    # Precalculate stats
    g_stats = []
    for g in range(gpu_num):
        w = sum(m_data[i]['w'] for i in curr_sol[g])
        s = sum(m_data[i]['s'] for i in curr_sol[g])
        rem = GPU_MEM_SIZE - s
        p = w / rem if rem > 1e-9 else (float('inf') if w > 0 else 0.0)
        g_stats.append({'w': w, 's': s, 'p': p})

    # Optimization Loop
    # Increased iterations because we have time (prev exec 0.008s)
    max_ls_iter = 200
    
    for _ in range(max_ls_iter):
        # Identify bottleneck GPU
        max_p = -1.0
        src = -1
        for g in range(gpu_num):
            if g_stats[g]['p'] > max_p:
                max_p = g_stats[g]['p']
                src = g
        
        if src == -1 or max_p < 1e-9: break
        
        improved = False
        src_items = curr_sol[src]
        
        # 3.1 Move Operation (Reallocation)
        # Try to move an item from Bottleneck to ANY other GPU if it lowers global max pressure
        
        for i_idx, m_idx in enumerate(src_items):
            item = m_data[m_idx]
            
            # Hypothetical src state
            src_s_new = g_stats[src]['s'] - item['s']
            src_w_new = g_stats[src]['w'] - item['w']
            src_rem_new = GPU_MEM_SIZE - src_s_new
            src_p_new = src_w_new / src_rem_new if src_rem_new > 1e-9 else (float('inf') if src_w_new > 0 else 0.0)
            
            best_dst = None
            best_improvement = 0.0
            
            for dst in range(gpu_num):
                if dst == src: continue
                if g_stats[dst]['s'] + item['s'] > GPU_MEM_SIZE: continue
                
                dst_s_new = g_stats[dst]['s'] + item['s']
                dst_w_new = g_stats[dst]['w'] + item['w']
                dst_rem_new = GPU_MEM_SIZE - dst_s_new
                dst_p_new = dst_w_new / dst_rem_new if dst_rem_new > 1e-9 else (float('inf') if dst_w_new > 0 else 0.0)
                
                new_max = max(src_p_new, dst_p_new)
                current_improvement = max_p - new_max
                
                # Valid improvement if the new local max is less than global max
                # And we want to maximize the drop (greedy choice)
                if current_improvement > 1e-6:
                    if current_improvement > best_improvement:
                        best_improvement = current_improvement
                        best_dst = dst
            
            if best_dst is not None:
                # Execute Move
                curr_sol[src].pop(i_idx)
                curr_sol[best_dst].append(m_idx)
                
                g_stats[src] = {'w': src_w_new, 's': src_s_new, 'p': src_p_new}
                
                d_w = g_stats[best_dst]['w'] + item['w']
                d_s = g_stats[best_dst]['s'] + item['s']
                d_rem = GPU_MEM_SIZE - d_s
                d_p = d_w / d_rem if d_rem > 1e-9 else (float('inf') if d_w > 0 else 0.0)
                g_stats[best_dst] = {'w': d_w, 's': d_s, 'p': d_p}
                
                improved = True
                break
        
        if improved: continue
        
        # 3.2 Swap Operation
        # Swap item from Bottleneck with item from another GPU
        for s_list_i, s_idx in enumerate(src_items):
            s_item = m_data[s_idx]
            
            for dst in range(gpu_num):
                if dst == src: continue
                
                dst_items = curr_sol[dst]
                for d_list_i, d_idx in enumerate(dst_items):
                    d_item = m_data[d_idx]
                    
                    # Capacity check
                    new_src_s = g_stats[src]['s'] - s_item['s'] + d_item['s']
                    if new_src_s > GPU_MEM_SIZE: continue
                    
                    new_dst_s = g_stats[dst]['s'] - d_item['s'] + s_item['s']
                    if new_dst_s > GPU_MEM_SIZE: continue
                    
                    # Pressure check
                    new_src_w = g_stats[src]['w'] - s_item['w'] + d_item['w']
                    new_src_rem = GPU_MEM_SIZE - new_src_s
                    new_src_p = new_src_w / new_src_rem if new_src_rem > 1e-9 else float('inf')
                    
                    new_dst_w = g_stats[dst]['w'] - d_item['w'] + s_item['w']
                    new_dst_rem = GPU_MEM_SIZE - new_dst_s
                    new_dst_p = new_dst_w / new_dst_rem if new_dst_rem > 1e-9 else float('inf')
                    
                    new_max = max(new_src_p, new_dst_p)
                    
                    if new_max < max_p - 1e-6:
                        # Execute Swap
                        curr_sol[src][s_list_i] = d_idx
                        curr_sol[dst][d_list_i] = s_idx
                        
                        g_stats[src] = {'w': new_src_w, 's': new_src_s, 'p': new_src_p}
                        g_stats[dst] = {'w': new_dst_w, 's': new_dst_s, 'p': new_dst_p}
                        
                        improved = True
                        break
                if improved: break
            if improved: break
            
    # Final conversion
    result = {}
    for g, idxs in curr_sol.items():
        result[g] = [m_data[i]['obj'] for i in idxs]
    
    return result
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