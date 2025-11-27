# EVOLVE-BLOCK-START
"""Model placement algorithm for minimizing maximum KV cache pressure using Binary Search with BFD and Ruin-Recreate ILS"""

import copy
import random

GPU_MEM_SIZE = 80  # GB

def compute_model_placement(gpu_num, models):
    """
    Minimizes max KVPR using Binary Search with Best-Fit Decreasing Packing 
    followed by Ruin-and-Recreate Iterated Local Search.
    """
    # 1. Validation and Setup
    total_size = sum(m.model_size for m in models)
    if total_size > gpu_num * GPU_MEM_SIZE:
        raise ValueError("Total model size exceeds total GPU memory capacity.")
        
    # Prepare items for packing: (w, s, m)
    items = [{'w': m.req_rate / m.slo, 's': m.model_size, 'm': m} for m in models]
    
    # 2. Binary Search for Initial Feasible Solution
    total_w = sum(x['w'] for x in items)
    slack = gpu_num * GPU_MEM_SIZE - total_size
    
    # Heuristic Initialization
    high = 1000.0
    if slack > 1e-5:
        avg_k = total_w / slack
        high = max(10.0, avg_k * 4.0)
        
    best_placement = None
    feasible_high = False
    
    # Exponential search for valid upper bound
    for _ in range(20):
        feasible, placement = _check_feasibility_multi(gpu_num, items, high)
        if feasible:
            best_placement = placement
            feasible_high = True
            break
        high *= 2.0
    
    # If heuristic search fails, try a very large bound to guarantee solution if physically possible
    if not feasible_high:
        high = 1e9
        feasible, placement = _check_feasibility_multi(gpu_num, items, high)
        if feasible:
            best_placement = placement
        else:
            raise ValueError("Unable to place models. Physical constraints too tight.")
            
    low = 0.0
    # Binary Search Refinement
    for _ in range(30):
        mid = (low + high) / 2.0
        feasible, placement = _check_feasibility_multi(gpu_num, items, mid)
        if feasible:
            best_placement = placement
            high = mid
        else:
            low = mid
            
    # Convert list placement to dictionary map
    placement_map = {i: best_placement[i] for i in range(gpu_num)}
    
    # 3. Iterated Local Search with Ruin & Recreate
    return _ils_ruin_recreate(gpu_num, placement_map)

def _check_feasibility_multi(gpu_num, items, K):
    """
    Checks feasibility using multiple sorting heuristics including Best Fit Decreasing.
    """
    virtual_cap = K * GPU_MEM_SIZE
    # Create pack items: (virtual, physical, load, model)
    pack_items = []
    for x in items:
        v = x['w'] + K * x['s']
        pack_items.append((v, x['s'], x['w'], x['m']))
        
    # Strategy 1: FFD on Virtual Size
    pack_items.sort(key=lambda x: x[0], reverse=True)
    res = _pack_ffd(gpu_num, pack_items, virtual_cap)
    if res: return True, res
    
    # Strategy 2: BFD on Virtual Size
    # Uses the same sorted order (Virtual Size Descending)
    res = _pack_bfd(gpu_num, pack_items, virtual_cap)
    if res: return True, res
    
    # Strategy 3: FFD on Physical Size
    pack_items.sort(key=lambda x: x[1], reverse=True)
    res = _pack_ffd(gpu_num, pack_items, virtual_cap)
    if res: return True, res
    
    # Strategy 4: FFD on Load Density (w/s)
    pack_items.sort(key=lambda x: x[2]/(x[1]+1e-6), reverse=True)
    res = _pack_ffd(gpu_num, pack_items, virtual_cap)
    if res: return True, res
    
    return False, None

def _pack_ffd(gpu_num, items, v_cap):
    bins_v = [0.0] * gpu_num
    bins_p = [0.0] * gpu_num
    placement = [[] for _ in range(gpu_num)]
    
    for v, s, w, m in items:
        placed = False
        for i in range(gpu_num):
            if bins_p[i] + s <= GPU_MEM_SIZE and bins_v[i] + v <= v_cap + 1e-7:
                bins_p[i] += s
                bins_v[i] += v
                placement[i].append(m)
                placed = True
                break
        if not placed: return None
    return placement

def _pack_bfd(gpu_num, items, v_cap):
    """Best Fit Decreasing on Virtual Capacity."""
    bins_v = [0.0] * gpu_num
    bins_p = [0.0] * gpu_num
    placement = [[] for _ in range(gpu_num)]
    
    for v, s, w, m in items:
        best_i = -1
        min_rem = float('inf')
        
        for i in range(gpu_num):
            if bins_p[i] + s <= GPU_MEM_SIZE and bins_v[i] + v <= v_cap + 1e-7:
                rem = v_cap - bins_v[i] - v
                if rem < min_rem:
                    min_rem = rem
                    best_i = i
        
        if best_i != -1:
            bins_p[best_i] += s
            bins_v[best_i] += v
            placement[best_i].append(m)
        else:
            return None
    return placement

def _ils_ruin_recreate(gpu_num, placement):
    """
    Refines placement using Hill Climbing with Ruin-and-Recreate Perturbation.
    """
    # Initialize state
    gpu_s = [sum(m.model_size for m in placement[i]) for i in range(gpu_num)]
    gpu_w = [sum(m.req_rate / m.slo for m in placement[i]) for i in range(gpu_num)]
    
    def get_k(i):
        rem = GPU_MEM_SIZE - gpu_s[i]
        if rem <= 1e-7: return 1e9
        return gpu_w[i] / rem

    best_sol = copy.deepcopy(placement)
    best_max_k = max(get_k(i) for i in range(gpu_num))
    
    no_improve = 0
    max_steps = 400
    patience = 20
    
    for _ in range(max_steps):
        # 1. Metrics
        current_k = [get_k(i) for i in range(gpu_num)]
        max_k = max(current_k)
        src = current_k.index(max_k)
        
        # 2. Update Global Best
        if max_k < best_max_k - 1e-6:
            best_max_k = max_k
            best_sol = copy.deepcopy(placement)
            no_improve = 0
        else:
            no_improve += 1
            
        # 3. Strategy
        if no_improve <= patience:
            # Greedy Descent
            improved = False
            src_models = placement[src]
            
            # Phase A: Move
            for idx, m in enumerate(src_models):
                s, w = m.model_size, m.req_rate/m.slo
                for dst in range(gpu_num):
                    if dst == src: continue
                    if gpu_s[dst] + s > GPU_MEM_SIZE: continue
                    
                    # Check improvement
                    rem_src = GPU_MEM_SIZE - (gpu_s[src] - s)
                    nk_src = (gpu_w[src] - w) / rem_src if rem_src > 1e-7 else 1e9
                    
                    rem_dst = GPU_MEM_SIZE - (gpu_s[dst] + s)
                    nk_dst = (gpu_w[dst] + w) / rem_dst if rem_dst > 1e-7 else 1e9
                    
                    if max(nk_src, nk_dst) < max_k - 1e-6:
                        placement[dst].append(m)
                        placement[src].pop(idx)
                        gpu_s[src] -= s; gpu_w[src] -= w
                        gpu_s[dst] += s; gpu_w[dst] += w
                        improved = True
                        break
                if improved: break
            
            # Phase B: Swap
            if not improved:
                for i1, m1 in enumerate(src_models):
                    s1, w1 = m1.model_size, m1.req_rate/m1.slo
                    for dst in range(gpu_num):
                        if dst == src: continue
                        # Optimization: only swap with non-bottleneck GPUs
                        if current_k[dst] > max_k * 0.95: continue
                        
                        for i2, m2 in enumerate(placement[dst]):
                            s2, w2 = m2.model_size, m2.req_rate/m2.slo
                            
                            ns_src = gpu_s[src] - s1 + s2
                            ns_dst = gpu_s[dst] - s2 + s1
                            if ns_src > GPU_MEM_SIZE or ns_dst > GPU_MEM_SIZE: continue
                            
                            rem_src = GPU_MEM_SIZE - ns_src
                            nk_src = (gpu_w[src] - w1 + w2) / rem_src if rem_src > 1e-7 else 1e9
                            
                            rem_dst = GPU_MEM_SIZE - ns_dst
                            nk_dst = (gpu_w[dst] - w2 + w1) / rem_dst if rem_dst > 1e-7 else 1e9
                            
                            if max(nk_src, nk_dst) < max_k - 1e-6:
                                placement[src][i1] = m2
                                placement[dst][i2] = m1
                                gpu_s[src] = ns_src; gpu_w[src] += (w2 - w1)
                                gpu_s[dst] = ns_dst; gpu_w[dst] += (w1 - w2)
                                improved = True
                                break
                        if improved: break
                    if improved: break
                    
        else:
            # Phase C: Ruin and Recreate Perturbation
            # Select bottleneck GPU and 1 random other GPU (ideally low load)
            targets = [src]
            
            # Find a low load GPU
            min_k = min(current_k)
            min_gpu = current_k.index(min_k)
            if min_gpu != src:
                targets.append(min_gpu)
            else:
                # Pick random valid GPU
                candidates = [x for x in range(gpu_num) if x != src]
                if candidates:
                    targets.append(random.choice(candidates))
            
            if len(targets) > 1:
                # Ruin: Remove all items from these GPUs
                popped_models = []
                for t in targets:
                    popped_models.extend(placement[t])
                    placement[t] = []
                    gpu_s[t] = 0.0
                    gpu_w[t] = 0.0
                
                # Recreate: Sort models by descending weight and pack greedily 
                # to minimize max K among these targets
                popped_models.sort(key=lambda m: m.req_rate/m.slo, reverse=True)
                
                for m in popped_models:
                    w, s = m.req_rate/m.slo, m.model_size
                    best_t = -1
                    best_local_k = float('inf')
                    
                    # Try to place in one of the targets to minimize that target's K
                    for t in targets:
                        if gpu_s[t] + s <= GPU_MEM_SIZE:
                            rem = GPU_MEM_SIZE - (gpu_s[t] + s)
                            nk = (gpu_w[t] + w) / rem if rem > 1e-7 else 1e9
                            if nk < best_local_k:
                                best_local_k = nk
                                best_t = t
                    
                    if best_t != -1:
                        placement[best_t].append(m)
                        gpu_s[best_t] += s
                        gpu_w[best_t] += w
                    else:
                        # Fallback: simple first fit if heuristic optimal check fails
                        for t in targets:
                            if gpu_s[t] + s <= GPU_MEM_SIZE:
                                placement[t].append(m)
                                gpu_s[t] += s
                                gpu_w[t] += w
                                break
                
                # Reset patience
                no_improve = max(0, patience - 5)

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

