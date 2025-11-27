# EVOLVE-BLOCK-START
"""Model placement algorithm for minimizing maximum KV cache pressure using Multi-Strategy Packing and Iterated Local Search"""

import copy
import random

GPU_MEM_SIZE = 80  # GB

def compute_model_placement(gpu_num, models):
    """
    Minimizes max KVPR using Binary Search with 4 packing heuristics 
    followed by Iterated Local Search for refinement.
    """
    total_size = sum(m.model_size for m in models)
    if total_size > gpu_num * GPU_MEM_SIZE:
        raise ValueError("Total model size exceeds total GPU memory capacity.")

    # Prepare items: (w, s, m)
    # w = req_rate / slo
    items = [{'w': m.req_rate / m.slo, 's': m.model_size, 'm': m} for m in models]

    # Binary Search for optimal KVPR
    total_w = sum(x['w'] for x in items)
    slack = gpu_num * GPU_MEM_SIZE - total_size
    
    # Robust High Bound Initialization
    low = 0.0
    if slack > 1e-4:
        # Initial guess: Average Load / Average Slack per GPU
        # Multiplier to account for fragmentation
        high = (total_w / slack) * 4.0 
    else:
        high = 1000.0
    high = max(high, 10.0)

    best_packing_placement = None
    feasible_high = False
    
    # 1. Find valid upper bound
    for _ in range(20):
        feasible, placement = _check_feasibility_multi(gpu_num, items, high)
        if feasible:
            best_packing_placement = placement
            feasible_high = True
            break
        low = high
        high *= 2.0
        
    if not feasible_high:
        # Fallback to a very loose check or raise error
        feasible, placement = _check_feasibility_multi(gpu_num, items, 1e5)
        if feasible:
            best_packing_placement = placement
        else:
            raise ValueError("Unable to place models. Constraints likely too tight.")

    # 2. Binary Search Refinement
    for _ in range(30):
        mid = (low + high) / 2.0
        feasible, placement = _check_feasibility_multi(gpu_num, items, mid)
        if feasible:
            best_packing_placement = placement
            high = mid
        else:
            low = mid
            
    # Convert to placement map {gpu_id: [models]}
    placement_map = {i: best_packing_placement[i] for i in range(gpu_num)}
    
    # 3. Iterated Local Search
    # Refine the placement to reduce max KVPR further
    final_placement = _iterated_local_search(gpu_num, placement_map)
    
    return final_placement

def _check_feasibility_multi(gpu_num, items, K):
    """
    Check if items can be packed with target KVPR 'K' using multiple heuristics.
    virtual_size = w + K * s
    Capacity_virtual = K * GPU_MEM_SIZE
    """
    virtual_cap = K * GPU_MEM_SIZE
    
    # Create pack items: (virtual_size, physical_size, weight, model)
    pack_items = []
    for x in items:
        v = x['w'] + K * x['s']
        pack_items.append((v, x['s'], x['w'], x['m']))
    
    # Strategy 1: FFD on Virtual Size
    pack_items.sort(key=lambda x: x[0], reverse=True)
    res = _pack_ffd(gpu_num, pack_items, virtual_cap)
    if res: return True, res

    # Strategy 2: BFD on Virtual Size
    # Already sorted by virtual size
    res = _pack_bfd(gpu_num, pack_items, virtual_cap)
    if res: return True, res
    
    # Strategy 3: FFD on Physical Size
    pack_items.sort(key=lambda x: x[1], reverse=True)
    res = _pack_ffd(gpu_num, pack_items, virtual_cap)
    if res: return True, res

    # Strategy 4: FFD on Load (w)
    pack_items.sort(key=lambda x: x[2], reverse=True)
    res = _pack_ffd(gpu_num, pack_items, virtual_cap)
    if res: return True, res

    return False, None

def _pack_ffd(gpu_num, items, virtual_cap):
    bins_v = [0.0] * gpu_num
    bins_p = [0.0] * gpu_num
    placement = [[] for _ in range(gpu_num)]
    
    for v, p, w, m in items:
        placed = False
        for i in range(gpu_num):
            if bins_p[i] + p <= GPU_MEM_SIZE and bins_v[i] + v <= virtual_cap + 1e-7:
                bins_p[i] += p
                bins_v[i] += v
                placement[i].append(m)
                placed = True
                break
        if not placed: return None
    return placement

def _pack_bfd(gpu_num, items, virtual_cap):
    bins_v = [0.0] * gpu_num
    bins_p = [0.0] * gpu_num
    placement = [[] for _ in range(gpu_num)]
    
    for v, p, w, m in items:
        best_bin = -1
        min_rem_v = float('inf')
        
        for i in range(gpu_num):
            if bins_p[i] + p <= GPU_MEM_SIZE and bins_v[i] + v <= virtual_cap + 1e-7:
                rem = virtual_cap - (bins_v[i] + v)
                if rem < min_rem_v:
                    min_rem_v = rem
                    best_bin = i
        
        if best_bin != -1:
            bins_p[best_bin] += p
            bins_v[best_bin] += v
            placement[best_bin].append(m)
        else:
            return None
    return placement

def _iterated_local_search(gpu_num, placement):
    """
    Hill Climbing with Random Kicks to minimize Max KVPR.
    """
    # Initialize state
    gpu_s = [sum(m.model_size for m in placement[i]) for i in range(gpu_num)]
    gpu_w = [sum(m.req_rate / m.slo for m in placement[i]) for i in range(gpu_num)]
    
    def get_kvpr(idx):
        rem = GPU_MEM_SIZE - gpu_s[idx]
        if rem <= 1e-7: return 1e9 # Penalty for full/overflow
        return gpu_w[idx] / rem

    best_sol = copy.deepcopy(placement)
    best_max_k = max(get_kvpr(i) for i in range(gpu_num))
    
    no_improve = 0
    max_steps = 400
    patience = 50
    
    for step in range(max_steps):
        # 1. Identify bottleneck GPU
        max_k = -1.0
        src = -1
        for i in range(gpu_num):
            k = get_kvpr(i)
            if k > max_k:
                max_k = k
                src = i
        
        # 2. Check global improvement
        if max_k < best_max_k - 1e-6:
            best_max_k = max_k
            best_sol = copy.deepcopy(placement)
            no_improve = 0
        else:
            no_improve += 1

        # 3. Kick / Perturbation if stuck
        if no_improve > patience:
            # Try a few random moves to escape
            for _ in range(3): 
                s_rnd = random.randint(0, gpu_num-1)
                if not placement[s_rnd]: continue
                
                d_rnd = random.randint(0, gpu_num-1)
                if s_rnd == d_rnd: continue
                
                m_idx = random.randint(0, len(placement[s_rnd])-1)
                m = placement[s_rnd][m_idx]
                
                if gpu_s[d_rnd] + m.model_size <= GPU_MEM_SIZE:
                    # Execute Kick
                    placement[d_rnd].append(m)
                    placement[s_rnd].pop(m_idx)
                    gpu_s[d_rnd] += m.model_size
                    gpu_w[d_rnd] += m.req_rate/m.slo
                    gpu_s[s_rnd] -= m.model_size
                    gpu_w[s_rnd] -= m.req_rate/m.slo
                    no_improve = 0 
                    break
            continue

        # 4. Greedy Descent: Reduce bottleneck 'src'
        improved = False
        models = placement[src]
        
        # Phase A: Try Move
        for i, m in enumerate(models):
            w, s = m.req_rate/m.slo, m.model_size
            
            for dst in range(gpu_num):
                if dst == src: continue
                if gpu_s[dst] + s > GPU_MEM_SIZE: continue
                
                # Predict new KVPRs
                rem_src = GPU_MEM_SIZE - (gpu_s[src] - s)
                nk_src = (gpu_w[src] - w) / rem_src if rem_src > 1e-7 else 1e9
                
                rem_dst = GPU_MEM_SIZE - (gpu_s[dst] + s)
                nk_dst = (gpu_w[dst] + w) / rem_dst if rem_dst > 1e-7 else 1e9
                
                if max(nk_src, nk_dst) < max_k - 1e-6:
                    placement[dst].append(m)
                    placement[src].pop(i)
                    gpu_s[src] -= s; gpu_w[src] -= w
                    gpu_s[dst] += s; gpu_w[dst] += w
                    improved = True
                    break
            if improved: break
        
        if improved: continue

        # Phase B: Try Swap
        for i, m1 in enumerate(models):
            w1, s1 = m1.req_rate/m1.slo, m1.model_size
            
            for dst in range(gpu_num):
                if dst == src: continue
                
                for j, m2 in enumerate(placement[dst]):
                    w2, s2 = m2.req_rate/m2.slo, m2.model_size
                    
                    ns_src = gpu_s[src] - s1 + s2
                    ns_dst = gpu_s[dst] - s2 + s1
                    if ns_src > GPU_MEM_SIZE or ns_dst > GPU_MEM_SIZE: continue
                    
                    rem_src = GPU_MEM_SIZE - ns_src
                    nk_src = (gpu_w[src] - w1 + w2) / rem_src if rem_src > 1e-7 else 1e9
                    
                    rem_dst = GPU_MEM_SIZE - ns_dst
                    nk_dst = (gpu_w[dst] - w2 + w1) / rem_dst if rem_dst > 1e-7 else 1e9
                    
                    if max(nk_src, nk_dst) < max_k - 1e-6:
                        placement[src][i] = m2
                        placement[dst][j] = m1
                        gpu_s[src] = ns_src; gpu_w[src] += (w2 - w1)
                        gpu_s[dst] = ns_dst; gpu_w[dst] += (w1 - w2)
                        improved = True
                        break
                if improved: break
            if improved: break
            
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

