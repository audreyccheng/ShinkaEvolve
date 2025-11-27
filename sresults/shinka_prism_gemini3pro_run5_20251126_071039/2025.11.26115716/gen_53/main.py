# EVOLVE-BLOCK-START
"""Model placement algorithm for minimizing maximum KV cache pressure using BFD Binary Search and Burst-Kicks ILS"""

import copy
import random
import math

GPU_MEM_SIZE = 80.0  # GB

def compute_model_placement(gpu_num, models):
    """
    Minimizes max KVPR using Robust Binary Search with Best-Fit Decreasing Packing
    followed by Steepest Descent Local Search with Multi-Step Burst Kicks.
    """
    # 1. Validation and Setup
    total_size = sum(m.model_size for m in models)
    if total_size > gpu_num * GPU_MEM_SIZE:
        raise ValueError("Total model size exceeds total GPU memory capacity.")

    # Prepare items for packing: (req_rate/slo, model_size, model_obj)
    # Storing as dictionary for easier attribute access
    items = [{'w': m.req_rate / m.slo, 's': m.model_size, 'm': m} for m in models]

    # 2. Binary Search for Initial Feasible Solution
    # Determine search range
    total_w = sum(x['w'] for x in items)
    slack = gpu_num * GPU_MEM_SIZE - total_size

    low = 0.0
    if slack < 1e-5:
        high = 1000.0
    else:
        avg_pressure = total_w / slack
        high = max(10.0, avg_pressure * 6.0)

    best_placement = None
    feasible_high = False

    # Find valid upper bound
    for _ in range(20):
        feasible, placement = _check_feasibility_robust(gpu_num, items, high)
        if feasible:
            best_placement = placement
            feasible_high = True
            break
        low = high
        high *= 2.0

    if not feasible_high:
        # Fallback for extreme cases
        high = 1e9
        feasible, placement = _check_feasibility_robust(gpu_num, items, high)
        if feasible:
            best_placement = placement
        else:
            raise ValueError("Unable to place models. Constraints likely too tight.")

    # Binary Search (32 iterations for precision)
    for _ in range(32):
        mid = (low + high) / 2.0
        feasible, placement = _check_feasibility_robust(gpu_num, items, mid)
        if feasible:
            best_placement = placement
            high = mid
        else:
            low = mid

    # Convert to placement map
    placement_map = {i: best_placement[i] for i in range(gpu_num)}

    # 3. Iterated Local Search (Steepest Descent + Burst Kicks)
    return _burst_local_search(gpu_num, placement_map)

def _check_feasibility_robust(gpu_num, items, K):
    """
    Checks if items can be packed into gpu_num bins with max KVPR <= K.
    Equation: sum(w) / (C - sum(s)) <= K  => sum(w + K*s) <= K*C
    Uses Best-Fit Decreasing with multiple sorting heuristics.
    """
    virtual_cap = K * GPU_MEM_SIZE
    
    # Precompute sort keys
    pack_items = []
    for x in items:
        v = x['w'] + K * x['s']
        d = x['w'] / (x['s'] + 1e-7)
        pack_items.append({
            'v': v, 
            's': x['s'], 
            'w': x['w'], 
            'd': d, 
            'm': x['m']
        })

    # Heuristics: (Sort Key, Reverse)
    heuristics = [
        (lambda x: x['v'], True),  # Virtual Size Desc
        (lambda x: x['s'], True),  # Physical Size Desc
        (lambda x: x['d'], True),  # Density Desc
        (lambda x: x['w'], True),  # Pressure contribution Desc
    ]

    for key_func, rev in heuristics:
        sorted_items = sorted(pack_items, key=key_func, reverse=rev)
        
        # Try Best Fit Decreasing
        if res := _pack_bfd(gpu_num, sorted_items, virtual_cap):
            return True, res
            
    return False, None

def _pack_bfd(gpu_num, items, virtual_cap):
    """
    Best Fit Decreasing: Place item in bin with minimum sufficient residual virtual capacity.
    """
    bins_v = [0.0] * gpu_num
    bins_p = [0.0] * gpu_num
    placement = [[] for _ in range(gpu_num)]

    for item in items:
        v, s, m = item['v'], item['s'], item['m']
        
        best_bin = -1
        min_rem_v = float('inf')

        for i in range(gpu_num):
            # Check constraints
            if bins_p[i] + s <= GPU_MEM_SIZE and bins_v[i] + v <= virtual_cap + 1e-7:
                rem = virtual_cap - (bins_v[i] + v)
                if rem < min_rem_v:
                    min_rem_v = rem
                    best_bin = i
        
        if best_bin != -1:
            bins_p[best_bin] += s
            bins_v[best_bin] += v
            placement[best_bin].append(m)
        else:
            return None
    return placement

def _burst_local_search(gpu_num, placement):
    """
    Refines placement using Steepest Descent Hill Climbing.
    Uses 'Burst Kicks' (sequence of random moves) to escape local optima.
    """
    # State tracking
    gpu_s = [sum(m.model_size for m in placement[i]) for i in range(gpu_num)]
    gpu_w = [sum(m.req_rate / m.slo for m in placement[i]) for i in range(gpu_num)]

    def get_k(i):
        rem = GPU_MEM_SIZE - gpu_s[i]
        if rem <= 1e-7: return 1e9
        return gpu_w[i] / rem

    best_sol = copy.deepcopy(placement)
    
    # Calculate initial max K
    ks = [get_k(i) for i in range(gpu_num)]
    best_max_k = max(ks)
    
    max_steps = 400
    patience = 30
    no_improve = 0

    for step in range(max_steps):
        # Current state analysis
        ks = [get_k(i) for i in range(gpu_num)]
        max_k = max(ks)
        
        # Check global improvement
        if max_k < best_max_k - 1e-7:
            best_max_k = max_k
            best_sol = copy.deepcopy(placement)
            no_improve = 0
        else:
            no_improve += 1

        # BURST KICK if stuck
        if no_improve > patience:
            # Perform a burst of random moves
            # We don't care about improving here, just changing state validly
            moves_executed = 0
            burst_limit = 4
            
            for _ in range(burst_limit * 3): # Try up to 3x limit to find valid moves
                if moves_executed >= burst_limit: break
                
                s_idx = random.randint(0, gpu_num - 1)
                if not placement[s_idx]: continue
                
                d_idx = random.randint(0, gpu_num - 1)
                if s_idx == d_idx: continue
                
                m_idx = random.randint(0, len(placement[s_idx]) - 1)
                m = placement[s_idx][m_idx]
                
                # Check feasibility
                if gpu_s[d_idx] + m.model_size <= GPU_MEM_SIZE:
                    # Move
                    placement[d_idx].append(m)
                    placement[s_idx].pop(m_idx)
                    gpu_s[d_idx] += m.model_size; gpu_w[d_idx] += m.req_rate/m.slo
                    gpu_s[s_idx] -= m.model_size; gpu_w[s_idx] -= m.req_rate/m.slo
                    moves_executed += 1
            
            if moves_executed > 0:
                no_improve = 0 # Reset patience
            continue

        # STEEPEST DESCENT
        # Focus on the bottleneck GPU(s)
        # Find the single best move/swap that reduces the bottleneck's K
        # or reduces variance if bottleneck K stays same (plateau traversal)
        
        # Identify bottleneck indices
        bottlenecks = [i for i, k in enumerate(ks) if k > max_k - 1e-5]
        src = random.choice(bottlenecks)
        
        best_move = None # ('move', idx, dst) or ('swap', idx1, dst, idx2)
        best_imp_k = -1e9
        best_imp_var = -1e9
        
        # 1. Evaluate Moves from Src
        for i, m in enumerate(placement[src]):
            s, w = m.model_size, m.req_rate/m.slo
            
            for dst in range(gpu_num):
                if dst == src: continue
                if gpu_s[dst] + s > GPU_MEM_SIZE: continue
                
                # Hypothetical State
                rem_src = GPU_MEM_SIZE - (gpu_s[src] - s)
                nk_src = (gpu_w[src] - w) / rem_src if rem_src > 1e-7 else 1e9
                
                rem_dst = GPU_MEM_SIZE - (gpu_s[dst] + s)
                nk_dst = (gpu_w[dst] + w) / rem_dst if rem_dst > 1e-7 else 1e9
                
                local_max = max(nk_src, nk_dst)
                
                # We want local_max <= max_k
                if local_max > max_k + 1e-7: continue
                
                imp_k = max_k - local_max
                # Variance improvement (sum of squares reduction)
                var_red = (ks[src]**2 + ks[dst]**2) - (nk_src**2 + nk_dst**2)
                
                if imp_k > best_imp_k + 1e-7:
                    best_imp_k = imp_k
                    best_imp_var = var_red
                    best_move = ('move', i, dst, s, w)
                elif abs(imp_k - best_imp_k) < 1e-7:
                    if var_red > best_imp_var + 1e-7:
                        best_imp_var = var_red
                        best_move = ('move', i, dst, s, w)

        # 2. Evaluate Swaps from Src
        # Only check swaps if we haven't found a dominant move
        if True: # Always check swaps for optimal descent
            for i1, m1 in enumerate(placement[src]):
                s1, w1 = m1.model_size, m1.req_rate/m1.slo
                
                for dst in range(gpu_num):
                    if dst == src: continue
                    # Optimization: skip swapping with other bottlenecks unless necessary
                    if ks[dst] > max_k - 1e-5: pass 

                    for i2, m2 in enumerate(placement[dst]):
                        s2, w2 = m2.model_size, m2.req_rate/m2.slo
                        
                        ns_src = gpu_s[src] - s1 + s2
                        ns_dst = gpu_s[dst] - s2 + s1
                        
                        if ns_src > GPU_MEM_SIZE or ns_dst > GPU_MEM_SIZE: continue
                        
                        rem_src = GPU_MEM_SIZE - ns_src
                        nk_src = (gpu_w[src] - w1 + w2) / rem_src if rem_src > 1e-7 else 1e9
                        
                        rem_dst = GPU_MEM_SIZE - ns_dst
                        nk_dst = (gpu_w[dst] - w2 + w1) / rem_dst if rem_dst > 1e-7 else 1e9
                        
                        local_max = max(nk_src, nk_dst)
                        if local_max > max_k + 1e-7: continue
                        
                        imp_k = max_k - local_max
                        var_red = (ks[src]**2 + ks[dst]**2) - (nk_src**2 + nk_dst**2)
                        
                        if imp_k > best_imp_k + 1e-7:
                            best_imp_k = imp_k
                            best_imp_var = var_red
                            best_move = ('swap', i1, dst, i2, s1, w1, s2, w2)
                        elif abs(imp_k - best_imp_k) < 1e-7:
                            if var_red > best_imp_var + 1e-7:
                                best_imp_var = var_red
                                best_move = ('swap', i1, dst, i2, s1, w1, s2, w2)

        # Apply Best Move
        if best_move:
            if best_move[0] == 'move':
                _, i, dst, s, w = best_move
                m = placement[src].pop(i)
                placement[dst].append(m)
                gpu_s[src] -= s; gpu_w[src] -= w
                gpu_s[dst] += s; gpu_w[dst] += w
            elif best_move[0] == 'swap':
                _, i1, dst, i2, s1, w1, s2, w2 = best_move
                m1 = placement[src][i1]
                m2 = placement[dst][i2]
                placement[src][i1] = m2
                placement[dst][i2] = m1
                gpu_s[src] = gpu_s[src] - s1 + s2
                gpu_w[src] = gpu_w[src] - w1 + w2
                gpu_s[dst] = gpu_s[dst] - s2 + s1
                gpu_w[dst] = gpu_w[dst] - w2 + w1
            
            # Reset patience only if we significantly improved the objective
            if best_imp_k > 1e-6:
                no_improve = 0
            # If we only reduced variance, we don't fully reset, effectively counting it half-step
            # but usually it finds a way down.
        else:
            no_improve += 1

    return best_sol
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

