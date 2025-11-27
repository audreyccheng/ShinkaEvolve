# EVOLVE-BLOCK-START
"""Model placement algorithm for minimizing maximum KV cache pressure across GPUs"""

import math
import heapq

GPU_MEM_SIZE = 80.0  # GB

def compute_model_placement(gpu_num, models):
    """
    Compute a model placement that minimizes the maximum KVPR across all GPUs.
    
    Hybrid approach combining:
    1. Multi-strategy Beam Search with symmetry breaking for high-quality initialization.
    2. Binary Search with multi-heuristic Bin Packing (Effective Size, Physical Size, Weight)
       to push the maximum pressure down.
    3. Hill-climbing Local Search (Move & Swap) to refine the solution.
    """

    if not models:
        return {i: [] for i in range(gpu_num)}

    # 0. Preprocessing
    m_data = []
    total_w = 0.0
    total_s = 0.0
    for i, m in enumerate(models):
        w = m.req_rate / m.slo
        s = m.model_size
        m_data.append({
            'id': i,
            'w': w,
            's': s,
            'obj': m
        })
        total_w += w
        total_s += s

    def calculate_score(placement_indices):
        max_p = 0.0
        # Handle both list of lists and dict of lists
        if isinstance(placement_indices, dict):
            iterator = placement_indices.values()
        else:
            iterator = placement_indices

        for indices in iterator:
            cur_w = sum(m_data[idx]['w'] for idx in indices)
            cur_s = sum(m_data[idx]['s'] for idx in indices)
            rem = GPU_MEM_SIZE - cur_s
            
            if rem <= 1e-9:
                if cur_w > 0: return float('inf')
                else: continue
            
            p = cur_w / rem
            if p > max_p: max_p = p
        return max_p

    best_placement_indices = None
    best_max_kvpr = float('inf')

    # ---------------------------------------------------------
    # 1. Beam Search Construction
    # ---------------------------------------------------------
    # Explore placement space using different sorting heuristics.
    # Prune symmetric GPU states to keep the search space manageable.
    
    BEAM_WIDTH = 5  # Keeps search fast while exploring diversity
    
    # Sorting strategies to guide the construction
    strategies = [
        lambda x: x['w'] / (x['s'] + 1e-6),              # Density
        lambda x: x['w'],                                # Weight
        lambda x: x['s'],                                # Size
        lambda x: x['w'] / (GPU_MEM_SIZE - x['s'] + 1e-6)# Isolated Pressure
    ]

    for sort_key in strategies:
        sorted_indices = sorted(range(len(m_data)), key=lambda i: sort_key(m_data[i]), reverse=True)
        
        # State: (current_max_kvpr, gpu_states_tuple, placement_tuple)
        # gpu_states_tuple: tuple of (w, s)
        # placement_tuple: tuple of tuples of indices
        
        init_states = tuple([(0.0, 0.0)] * gpu_num)
        init_placement = tuple([() for _ in range(gpu_num)])
        beam = [(0.0, init_states, init_placement)]
        
        for m_idx in sorted_indices:
            item = m_data[m_idx]
            w, s = item['w'], item['s']
            
            candidates = []
            seen_configs = set()
            
            for score, states, pl in beam:
                # Try placing on each GPU
                for g in range(gpu_num):
                    gw, gs = states[g]
                    if gs + s > GPU_MEM_SIZE: continue
                    
                    new_gs = gs + s
                    new_gw = gw + w
                    
                    # Estimate new local pressure
                    rem = GPU_MEM_SIZE - new_gs
                    if rem <= 1e-9:
                        local_p = float('inf') if new_gw > 1e-9 else 0.0
                    else:
                        local_p = new_gw / rem
                    
                    new_score = max(score, local_p)
                    
                    # Update states
                    new_states_list = list(states)
                    new_states_list[g] = (new_gw, new_gs)
                    
                    # Symmetry Pruning: Sort states to canonicalize
                    canonical = tuple(sorted(new_states_list))
                    if canonical in seen_configs:
                        continue
                    seen_configs.add(canonical)
                    
                    new_pl_list = list(pl)
                    new_pl_list[g] = pl[g] + (m_idx,)
                    
                    candidates.append((new_score, tuple(new_states_list), tuple(new_pl_list)))
            
            if not candidates:
                beam = []
                break
            
            # Prune beam
            if len(candidates) > BEAM_WIDTH:
                beam = heapq.nsmallest(BEAM_WIDTH, candidates, key=lambda x: x[0])
            else:
                beam = candidates
        
        # Evaluate survivors
        for score, _, pl in beam:
            if score < best_max_kvpr:
                best_max_kvpr = score
                best_placement_indices = [list(x) for x in pl]

    # ---------------------------------------------------------
    # 2. Binary Search Refinement (Robust Bin Packing)
    # ---------------------------------------------------------
    # Attempt to find a placement with max KVPR <= K.
    # We use multiple heuristics to solve the feasibility check (Bin Packing).
    
    rem_global = gpu_num * GPU_MEM_SIZE - total_s
    lb = total_w / rem_global if rem_global > 1e-6 else best_max_kvpr
    
    high = best_max_kvpr if best_max_kvpr != float('inf') else 2000.0
    low = lb
    
    if high > low + 1e-4:
        for _ in range(16):
            mid = (low + high) / 2.0
            
            # Strategies for Feasibility Check:
            # 1. Effective Size: s + w/mid (derived from constraint w <= mid*(C-s))
            # 2. Physical Size: s
            # 3. Weight: w
            sort_strategies = [
                lambda x: x['s'] + x['w'] / mid,
                lambda x: x['s'],
                lambda x: x['w']
            ]
            
            found_solution = None
            
            for key_fn in sort_strategies:
                sorted_idx = sorted(range(len(m_data)), key=lambda i: key_fn(m_data[i]), reverse=True)
                
                temp_alloc = [[] for _ in range(gpu_num)]
                temp_states = [{'w': 0.0, 's': 0.0} for _ in range(gpu_num)]
                possible = True
                
                for idx in sorted_idx:
                    item = m_data[idx]
                    w, s = item['w'], item['s']
                    
                    best_g = None
                    min_slack = float('inf')
                    
                    # Best Fit Decreasing on Pressure Constraint
                    for g in range(gpu_num):
                        st = temp_states[g]
                        if st['s'] + s > GPU_MEM_SIZE: continue
                        
                        phys_rem = GPU_MEM_SIZE - (st['s'] + s)
                        if phys_rem < 0: phys_rem = 0.0
                        
                        # Constraint: current_w + w <= mid * phys_rem
                        max_allowed_w = mid * phys_rem
                        new_w = st['w'] + w
                        
                        if new_w <= max_allowed_w + 1e-7:
                            # Feasible. Minimize slack to pack tightly.
                            slack = max_allowed_w - new_w
                            if slack < min_slack:
                                min_slack = slack
                                best_g = g
                    
                    if best_g is None:
                        possible = False
                        break
                    
                    temp_alloc[best_g].append(idx)
                    temp_states[best_g]['w'] += w
                    temp_states[best_g]['s'] += s
                
                if possible:
                    found_solution = temp_alloc
                    break
            
            if found_solution:
                # Found a valid K. Store it and try smaller K.
                score = calculate_score(found_solution)
                if score < best_max_kvpr:
                    best_max_kvpr = score
                    best_placement_indices = found_solution
                high = mid
            else:
                low = mid

    if best_placement_indices is None:
        raise ValueError("Unable to place models on GPUs with available memory.")

    # ---------------------------------------------------------
    # 3. Local Search (Hill Climbing)
    # ---------------------------------------------------------
    # Refine the best solution found using Move and Swap operations.
    
    # Convert best_placement_indices to dict format {gpu_id: [indices]}
    if isinstance(best_placement_indices, dict):
         curr_map = {g: list(idxs) for g, idxs in best_placement_indices.items()}
    else:
         curr_map = {g: list(best_placement_indices[g]) for g in range(gpu_num)}
    
    def get_stats(g_idx):
        indices = curr_map[g_idx]
        w = sum(m_data[i]['w'] for i in indices)
        s = sum(m_data[i]['s'] for i in indices)
        rem = GPU_MEM_SIZE - s
        p = w / rem if rem > 1e-9 else (float('inf') if w > 0 else 0.0)
        return w, s, p
    
    g_stats = [get_stats(g) for g in range(gpu_num)]
    
    for _ in range(100):
        # Identify bottleneck GPU
        max_p = -1.0
        src_g = -1
        for g in range(gpu_num):
            if g_stats[g][2] > max_p:
                max_p = g_stats[g][2]
                src_g = g
        
        if src_g == -1 or max_p < 1e-9: break
        
        improved = False
        src_indices = curr_map[src_g]
        
        # 3.1 Try MOVE
        for i, m_idx in enumerate(src_indices):
            m = m_data[m_idx]
            
            # Predict Src stats
            s_rem = GPU_MEM_SIZE - (g_stats[src_g][1] - m['s'])
            s_w = g_stats[src_g][0] - m['w']
            s_p = s_w / s_rem if s_rem > 1e-9 else float('inf')
            
            best_dst = None
            
            for dst in range(gpu_num):
                if dst == src_g: continue
                d_w, d_s, d_p = g_stats[dst]
                
                if d_s + m['s'] > GPU_MEM_SIZE: continue
                
                d_rem = GPU_MEM_SIZE - (d_s + m['s'])
                d_w_new = d_w + m['w']
                d_p_new = d_w_new / d_rem if d_rem > 1e-9 else float('inf')
                
                if max(s_p, d_p_new) < max_p - 1e-6:
                    best_dst = dst
                    break 
            
            if best_dst is not None:
                curr_map[src_g].pop(i)
                curr_map[best_dst].append(m_idx)
                g_stats[src_g] = get_stats(src_g)
                g_stats[best_dst] = get_stats(best_dst)
                improved = True
                break
        
        if improved: continue
        
        # 3.2 Try SWAP
        for i, m1_idx in enumerate(src_indices):
            m1 = m_data[m1_idx]
            
            for dst in range(gpu_num):
                if dst == src_g: continue
                # Skip if dst is also near capacity to save cycles
                if g_stats[dst][2] > max_p - 0.5: continue
                
                dst_indices = curr_map[dst]
                for j, m2_idx in enumerate(dst_indices):
                    m2 = m_data[m2_idx]
                    
                    new_src_s = g_stats[src_g][1] - m1['s'] + m2['s']
                    new_dst_s = g_stats[dst][1] - m2['s'] + m1['s']
                    
                    if new_src_s > GPU_MEM_SIZE or new_dst_s > GPU_MEM_SIZE: continue
                    
                    new_src_w = g_stats[src_g][0] - m1['w'] + m2['w']
                    new_dst_w = g_stats[dst][0] - m2['w'] + m1['w']
                    
                    rem_src = GPU_MEM_SIZE - new_src_s
                    p_src = new_src_w / rem_src if rem_src > 1e-9 else float('inf')
                    
                    rem_dst = GPU_MEM_SIZE - new_dst_s
                    p_dst = new_dst_w / rem_dst if rem_dst > 1e-9 else float('inf')
                    
                    if max(p_src, p_dst) < max_p - 1e-6:
                        curr_map[src_g][i] = m2_idx
                        curr_map[dst][j] = m1_idx
                        g_stats[src_g] = get_stats(src_g)
                        g_stats[dst] = get_stats(dst)
                        improved = True
                        break
                if improved: break
            if improved: break
        
        if not improved: break

    # Final conversion
    result = {}
    for g, indices in curr_map.items():
        result[g] = [m_data[i]['obj'] for i in indices]
    
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