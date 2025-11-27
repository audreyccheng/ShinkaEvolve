# EVOLVE-BLOCK-START
"""Model placement algorithm for minimizing maximum KV cache pressure across GPUs"""

import random

GPU_MEM_SIZE = 80.0  # GB

def compute_model_placement(gpu_num, models):
    """
    Compute a model placement that minimizes the maximum KVPR across all GPUs.
    
    Architecture:
    1. Greedy Heuristics: Establish valid upper bound using multiple deterministic strategies.
    2. Binary Search on Pressure (K):
       - Transforms optimization to feasibility problem: Can we pack with KVPR <= K?
       - Feasibility Check: Adaptive Stochastic Bin Packing.
         - Uses 'Effective Size' (s + w/K) for Best Fit Decreasing.
         - Adds noise to sort keys to explore solution space.
         - Implements 'Failure Memory': Items that fail to fit are prioritized in subsequent retries.
    3. Local Search: Hill climbing on the best found solution using Move and Swap operators.
    """

    # ------------------------------------------------------------------
    # 0. Data Preparation
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

    # Helper: Calculate Max KVPR for a solution (dict of indices)
    def calc_score(placement_indices):
        max_p = 0.0
        for idxs in placement_indices.values():
            w_sum = sum(m_data[i]['w'] for i in idxs)
            s_sum = sum(m_data[i]['s'] for i in idxs)
            rem = GPU_MEM_SIZE - s_sum
            
            if rem <= 1e-9:
                if w_sum > 0: return float('inf')
                else: continue # Empty or 0 weight
            
            p = w_sum / rem
            if p > max_p: max_p = p
        return max_p

    best_sol_indices = None
    best_val = float('inf')

    # ------------------------------------------------------------------
    # 1. Phase I: Greedy Heuristic Ensemble
    # ------------------------------------------------------------------
    # Quick deterministic passes to find a good initial solution/upper bound.
    greedy_strategies = [
        (lambda x: x['w']/(GPU_MEM_SIZE - x['s'] + 1e-6), 'min_peak'), # Asymptotic Pressure
        (lambda x: x['s'], 'min_peak'),                                # Physical Size
        (lambda x: x['w'], 'min_load'),                                # Weight (Load Balance)
        (lambda x: x['w']/(x['s'] + 1e-6), 'min_peak')                 # Density
    ]
    
    for key_fn, mode in greedy_strategies:
        sorted_indices = sorted(range(len(m_data)), key=lambda i: key_fn(m_data[i]), reverse=True)
        bins = [{'w':0.0, 's':0.0, 'idxs':[]} for _ in range(gpu_num)]
        possible = True
        
        for idx in sorted_indices:
            item = m_data[idx]
            best_bin = None
            best_metric = float('inf')
            
            for b_idx in range(gpu_num):
                if bins[b_idx]['s'] + item['s'] > GPU_MEM_SIZE: continue
                
                rem = GPU_MEM_SIZE - bins[b_idx]['s']
                
                if mode == 'min_peak':
                    # Look ahead: minimize resulting peak on this GPU
                    new_rem = rem - item['s']
                    if new_rem < 1e-9:
                        val = float('inf') if (bins[b_idx]['w'] + item['w']) > 0 else 0.0
                    else:
                        val = (bins[b_idx]['w'] + item['w']) / new_rem
                else:
                    # Load balancing: minimize current weight/pressure without lookahead
                    # Actually standard 'min_load' minimizes W.
                    val = bins[b_idx]['w']
                
                if val < best_metric:
                    best_metric = val
                    best_bin = b_idx
            
            if best_bin is None:
                possible = False
                break
            
            bins[best_bin]['idxs'].append(idx)
            bins[best_bin]['w'] += item['w']
            bins[best_bin]['s'] += item['s']
            
        if possible:
            sol = {i: bins[i]['idxs'] for i in range(gpu_num)}
            score = calc_score(sol)
            if score < best_val:
                best_val = score
                best_sol_indices = sol

    # ------------------------------------------------------------------
    # 2. Phase II: Binary Search with Adaptive Stochastic Packing
    # ------------------------------------------------------------------
    rem_global = gpu_num * GPU_MEM_SIZE - total_s
    # Theoretical lower bound
    lb = total_w / rem_global if rem_global > 1e-6 else best_val
    
    # Range setup
    low = lb
    high = best_val if best_val != float('inf') else 2000.0

    if high > low + 1e-4:
        
        def try_pack(target_k):
            """
            Attempts to pack items such that KVPR <= target_k.
            Uses randomized sorting with failure feedback.
            """
            # Track failure counts to prioritize difficult items
            fail_counts = {i: 0 for i in range(len(m_data))}
            
            # Strategies for base sorting
            # 1. Effective Size: s + w/K
            # 2. Physical Size: s
            strategies = [
                lambda idx: m_data[idx]['s'] + m_data[idx]['w'] / target_k,
                lambda idx: m_data[idx]['s']
            ]
            
            # Number of stochastic trials per K check
            num_trials = 30
            
            for attempt in range(num_trials):
                # Cycle strategies
                base_key_fn = strategies[attempt % len(strategies)]
                
                # Noise increases with attempts to break loops
                noise_scale = 0.02 + (0.005 * attempt)
                
                # Dynamic Sort Key
                def sort_key(idx):
                    val = base_key_fn(idx)
                    # Multiplicative noise
                    val *= random.uniform(1.0 - noise_scale, 1.0 + noise_scale)
                    # Failure boost: if item failed before, add huge value to put it first
                    val += fail_counts[idx] * 1e6
                    return val
                
                indices = sorted(range(len(m_data)), key=sort_key, reverse=True)
                
                # Bin Packing (Best Fit Decreasing on Effective Slack)
                bins_s = [0.0] * gpu_num
                bins_w = [0.0] * gpu_num
                placement = [[] for _ in range(gpu_num)]
                
                possible_iter = True
                first_fail_idx = None
                
                for idx in indices:
                    item = m_data[idx]
                    eff_size = item['s'] + item['w'] / target_k
                    
                    best_b = None
                    min_rem_eff = float('inf')
                    
                    for b in range(gpu_num):
                        # 1. Physical constraint
                        if bins_s[b] + item['s'] > GPU_MEM_SIZE: continue
                        
                        # 2. Pressure constraint: sum(w) / (C - sum(s)) <= K
                        # equivalent to: sum(w)/K + sum(s) <= C
                        # equivalent to: current_eff + item_eff <= C
                        curr_eff = bins_s[b] + bins_w[b] / target_k
                        
                        if curr_eff + eff_size <= GPU_MEM_SIZE + 1e-6:
                            # Valid bin. Minimize remaining effective space (Best Fit)
                            rem_eff = GPU_MEM_SIZE - (curr_eff + eff_size)
                            if rem_eff < min_rem_eff:
                                min_rem_eff = rem_eff
                                best_b = b
                    
                    if best_b is None:
                        possible_iter = False
                        first_fail_idx = idx
                        break
                    
                    placement[best_b].append(idx)
                    bins_s[best_b] += item['s']
                    bins_w[best_b] += item['w']
                
                if possible_iter:
                    return {i: placement[i] for i in range(gpu_num)}
                else:
                    if first_fail_idx is not None:
                        fail_counts[first_fail_idx] += 1
            
            return None

        # Binary Search Loop
        # 20 steps is enough precision
        for _ in range(20):
            mid = (low + high) / 2.0
            sol = try_pack(mid)
            
            if sol:
                # Found feasible configuration for K=mid.
                # Calculate actual max KVPR (it might be slightly less than mid)
                actual_score = calc_score(sol)
                if actual_score < best_val:
                    best_val = actual_score
                    best_sol_indices = sol
                # Try harder constraint
                high = mid
            else:
                # Cannot fit, relax constraint
                low = mid

    if best_sol_indices is None:
        raise ValueError("Unable to place models on GPUs with available memory.")

    # ------------------------------------------------------------------
    # 3. Phase III: Local Search Refinement
    # ------------------------------------------------------------------
    # Convert indices to a mutable structure for local search
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
    for _ in range(80):
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
        
        # 3.1 Move Operation
        for i_idx, m_idx in enumerate(src_items):
            item = m_data[m_idx]
            
            # Simulated removal from src
            src_s_new = g_stats[src]['s'] - item['s']
            src_w_new = g_stats[src]['w'] - item['w']
            src_rem_new = GPU_MEM_SIZE - src_s_new
            src_p_new = src_w_new / src_rem_new if src_rem_new > 1e-9 else float('inf')
            
            best_dst = None
            
            for dst in range(gpu_num):
                if dst == src: continue
                if g_stats[dst]['s'] + item['s'] > GPU_MEM_SIZE: continue
                
                # Simulated add to dst
                dst_s_new = g_stats[dst]['s'] + item['s']
                dst_w_new = g_stats[dst]['w'] + item['w']
                dst_rem_new = GPU_MEM_SIZE - dst_s_new
                dst_p_new = dst_w_new / dst_rem_new if dst_rem_new > 1e-9 else float('inf')
                
                # Check condition: reduce global max pressure
                # We need both affected nodes to be below current max_p (with epsilon)
                if max(src_p_new, dst_p_new) < max_p - 1e-5:
                    best_dst = dst
                    break # First Fit improvement
            
            if best_dst is not None:
                # Apply Move
                curr_sol[src].pop(i_idx)
                curr_sol[best_dst].append(m_idx)
                
                # Update stats
                g_stats[src] = {'w': src_w_new, 's': src_s_new, 'p': src_p_new}
                
                d_w = g_stats[best_dst]['w'] + item['w']
                d_s = g_stats[best_dst]['s'] + item['s']
                d_rem = GPU_MEM_SIZE - d_s
                d_p = d_w / d_rem if d_rem > 1e-9 else float('inf')
                g_stats[best_dst] = {'w': d_w, 's': d_s, 'p': d_p}
                
                improved = True
                break
        
        if improved: continue
        
        # 3.2 Swap Operation
        for s_list_i, s_idx in enumerate(src_items):
            s_item = m_data[s_idx]
            
            for dst in range(gpu_num):
                if dst == src: continue
                
                dst_items = curr_sol[dst]
                for d_list_i, d_idx in enumerate(dst_items):
                    d_item = m_data[d_idx]
                    
                    # Capacity check for Swap
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
                    
                    if max(new_src_p, new_dst_p) < max_p - 1e-5:
                        # Apply Swap
                        curr_sol[src][s_list_i] = d_idx
                        curr_sol[dst][d_list_i] = s_idx
                        
                        g_stats[src] = {'w': new_src_w, 's': new_src_s, 'p': new_src_p}
                        g_stats[dst] = {'w': new_dst_w, 's': new_dst_s, 'p': new_dst_p}
                        
                        improved = True
                        break
                if improved: break
            if improved: break
            
    # Final conversion to object list
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