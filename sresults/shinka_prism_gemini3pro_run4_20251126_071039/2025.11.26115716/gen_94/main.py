# EVOLVE-BLOCK-START
"""Model placement algorithm for minimizing maximum KV cache pressure across GPUs"""

GPU_MEM_SIZE = 80.0

def compute_model_placement(gpu_num, models):
    """
    Compute a model placement that minimizes the maximum KVPR across all GPUs.
    
    Algorithm:
    1. Binary Search for optimal max KVPR (K).
    2. Feasibility Check (Linearized Beam Search):
       - Transforms the capacity/KVPR constraints into a single linearized dimension: cost = w + K*s.
       - Uses Beam Search with Best Fit heuristic (maximizing sum of squared loads) to find a valid packing.
       - Prunes symmetric states (identical bin loads) to explore the search space efficiently.
    3. Local Search (Steepest Descent):
       - Post-optimizes any feasible placement found.
       - Evaluates all possible Moves and Swaps to minimize the maximum KVPR.
       - Greedily applies the single best improvement at each step.
    """

    # Pre-process models for beam search
    items = []
    for m in models:
        items.append({
            'model': m,
            'w': m.req_rate / m.slo,
            's': m.model_size
        })

    def get_kvpr(w, s):
        """Calculate KVPR safely."""
        rem = GPU_MEM_SIZE - s
        if rem <= 1e-9:
            return float('inf') if w > 1e-9 else 0.0
        return w / rem

    def get_max_kvpr(placement):
        """Calculate global max KVPR for a placement dict."""
        mx = 0.0
        for p in placement.values():
            w = sum(m.req_rate / m.slo for m in p)
            s = sum(m.model_size for m in p)
            mx = max(mx, get_kvpr(w, s))
        return mx

    def optimize(placement):
        """
        Steepest Descent Hill Climbing.
        Iteratively applies the BEST move/swap that reduces the bottleneck.
        """
        # Convert to mutable state with cached sums
        gpu_states = []
        for i in range(gpu_num):
            p = placement[i]
            w = sum(m.req_rate / m.slo for m in p)
            s = sum(m.model_size for m in p)
            gpu_states.append({'w': w, 's': s, 'items': list(p)})

        # Limit iterations
        for _ in range(100):
            # Identify Bottleneck GPU
            current_max = -1.0
            src_idx = -1
            
            # Cache KVPRs to avoid recomputing for unchanged GPUs (optimization)
            gpu_kvprs = []
            for i in range(gpu_num):
                k = get_kvpr(gpu_states[i]['w'], gpu_states[i]['s'])
                gpu_kvprs.append(k)
                if k > current_max:
                    current_max = k
                    src_idx = i

            if current_max <= 1e-9: break
            
            src = gpu_states[src_idx]
            best_action = None
            # We want to find a state where the new max is minimized
            best_new_max = current_max 

            # 1. Evaluate Moves (Source -> Dest)
            for i, m in enumerate(src['items']):
                m_w = m.req_rate / m.slo
                m_s = m.model_size
                
                # Predict new Source state
                ns_w = src['w'] - m_w
                ns_s = src['s'] - m_s
                ns_k = get_kvpr(ns_w, ns_s)
                
                # Pruning: If removing doesn't help enough, skip
                if ns_k >= best_new_max - 1e-9: continue

                for dst_idx in range(gpu_num):
                    if dst_idx == src_idx: continue
                    dst = gpu_states[dst_idx]
                    
                    if dst['s'] + m_s > GPU_MEM_SIZE: continue
                    
                    nd_w = dst['w'] + m_w
                    nd_s = dst['s'] + m_s
                    nd_k = get_kvpr(nd_w, nd_s)
                    
                    # The limiting factor for this move is max(new_src, new_dst)
                    local_max = max(ns_k, nd_k)
                    
                    if local_max < best_new_max - 1e-9:
                        best_new_max = local_max
                        best_action = ('move', i, dst_idx, m_w, m_s)

            # 2. Evaluate Swaps (Source <-> Dest)
            for i, m1 in enumerate(src['items']):
                m1_w = m1.req_rate / m1.slo
                m1_s = m1.model_size
                
                for dst_idx in range(gpu_num):
                    if dst_idx == src_idx: continue
                    dst = gpu_states[dst_idx]
                    
                    # Heuristic: Skip if dst is already near bottleneck
                    if gpu_kvprs[dst_idx] > current_max * 0.95: continue
                    
                    for j, m2 in enumerate(dst['items']):
                        m2_w = m2.req_rate / m2.slo
                        m2_s = m2.model_size

                        # Size Check
                        ns_s = src['s'] - m1_s + m2_s
                        nd_s = dst['s'] - m2_s + m1_s
                        
                        if ns_s > GPU_MEM_SIZE or nd_s > GPU_MEM_SIZE: continue
                        
                        ns_w = src['w'] - m1_w + m2_w
                        nd_w = dst['w'] - m2_w + m1_w
                        
                        ns_k = get_kvpr(ns_w, ns_s)
                        nd_k = get_kvpr(nd_w, nd_s)
                        
                        local_max = max(ns_k, nd_k)
                        if local_max < best_new_max - 1e-9:
                            best_new_max = local_max
                            best_action = ('swap', i, dst_idx, j, m1_w, m1_s, m2_w, m2_s)

            # Apply Best Action
            if best_action:
                type_ = best_action[0]
                if type_ == 'move':
                    _, idx, dst_idx, m_w, m_s = best_action
                    dst = gpu_states[dst_idx]
                    
                    m = src['items'].pop(idx)
                    src['w'] -= m_w
                    src['s'] -= m_s
                    
                    dst['items'].append(m)
                    dst['w'] += m_w
                    dst['s'] += m_s
                else: # swap
                    _, idx1, dst_idx, idx2, m1_w, m1_s, m2_w, m2_s = best_action
                    dst = gpu_states[dst_idx]
                    
                    m1 = src['items'][idx1]
                    m2 = dst['items'][idx2]
                    
                    # Swap models
                    src['items'][idx1] = m2
                    dst['items'][idx2] = m1
                    
                    # Update metrics
                    src['w'] = src['w'] - m1_w + m2_w
                    src['s'] = src['s'] - m1_s + m2_s
                    dst['w'] = dst['w'] - m2_w + m1_w
                    dst['s'] = dst['s'] - m2_s + m1_s
            else:
                break # Local Optimum Reached

        return {i: gpu_states[i]['items'] for i in range(gpu_num)}

    def solve_beam(k_target, beam_width):
        """
        Beam Search to pack items satisfying w + k*s <= k*C.
        Constraint linearized as 1D Bin Packing.
        """
        cap = k_target * GPU_MEM_SIZE
        
        # Sort items by linearized cost (Best Fit Decreasing)
        # item cost = w + k*s
        weighted_items = []
        for x in items:
            cost = x['w'] + k_target * x['s']
            if cost > cap + 1e-5: return None
            weighted_items.append((cost, x))
        
        weighted_items.sort(key=lambda x: x[0], reverse=True)
        
        # State: tuple of current loads (floats)
        initial_loads = tuple([0.0] * gpu_num)
        
        # Beam: List of (score, loads_tuple, placement_list)
        # Score: Sum of squares of loads (favor uneven filling / Best Fit)
        beam = [(0.0, initial_loads, [[] for _ in range(gpu_num)])]
        
        for cost, item in weighted_items:
            candidates = []
            seen_signatures = set()
            
            for _, loads, placements in beam:
                # Try placing item in each bin
                
                # Optimization: Duplicate load handling
                # If bin 0 and bin 1 have same load, placing in either results in symmetric state.
                tried_loads = set()
                
                for i in range(gpu_num):
                    current_l = loads[i]
                    if current_l in tried_loads: continue
                    
                    # Check linearized capacity
                    if current_l + cost <= cap + 1e-7:
                        tried_loads.add(current_l)
                        
                        new_loads_list = list(loads)
                        new_loads_list[i] += cost
                        
                        # Symmetry Breaking: States with same load set are identical
                        sig = tuple(sorted(new_loads_list))
                        if sig in seen_signatures: continue
                        seen_signatures.add(sig)
                        
                        # Construct new placement
                        new_pl = [list(p) for p in placements]
                        new_pl[i].append(item['model'])
                        
                        # Heuristic Score: Sum of squares
                        new_score = sum(l*l for l in new_loads_list)
                        
                        candidates.append((new_score, tuple(new_loads_list), new_pl))
            
            if not candidates:
                return None
            
            # Select top K states
            candidates.sort(key=lambda x: x[0], reverse=True)
            beam = candidates[:beam_width]
            
        # Return best result from beam
        best_solution = beam[0][2]
        return {i: best_solution[i] for i in range(gpu_num)}

    def check_feasibility(k_target):
        # Adaptive Beam Width
        # Fast check
        res = solve_beam(k_target, beam_width=2)
        if res: return res
        # Robust check
        return solve_beam(k_target, beam_width=8)

    # Binary Search Driver
    high = 1e9
    
    # 1. Initial Feasibility
    best_placement = check_feasibility(high)
    if best_placement is None:
        raise ValueError("Unable to place models on GPUs.")

    # 2. Optimize Upper Bound
    best_placement = optimize(best_placement)
    high = get_max_kvpr(best_placement)
    low = 0.0
    
    # 3. Binary Search
    for _ in range(25):
        if high - low < 1e-4: break
        mid = (low + high) / 2
        
        res = check_feasibility(mid)
        if res:
            # Feasible at 'mid'. Optimize to potentially lower global max further
            res = optimize(res)
            current_max = get_max_kvpr(res)
            
            if current_max < get_max_kvpr(best_placement):
                best_placement = res
            
            high = min(mid, current_max)
        else:
            low = mid

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
