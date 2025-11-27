# EVOLVE-BLOCK-START
"""Model placement algorithm for minimizing maximum KV cache pressure across GPUs"""

import math

GPU_MEM_SIZE = 80.0

def compute_model_placement(gpu_num, models):
    """
    Compute a model placement that minimizes the maximum KVPR across all GPUs.
    
    Approach:
    1.  Transform the KVPR minimization problem into a series of feasibility checks for a target KVPR K.
        Constraint: sum(req/slo) / (Capacity - sum(size)) <= K
        Linearized: sum(req/slo + K * size) <= K * Capacity
    2.  Use Binary Search to find the minimum K.
    3.  Check feasibility using Beam Search (Width=5) with symmetry breaking (canonicalized bin states).
        Heuristic: Maximize sum of squared loads (Best-Fit preference).
    4.  Refine feasible solutions using local search (load balancing) to tighten binary search bounds faster.

    Args:
        gpu_num: Number of GPUs
        models: List of models to place

    Returns:
        A placement of models to GPUs
    """
    
    # 1. Preprocess items
    items = []
    total_w = 0.0
    total_s = 0.0
    for i, m in enumerate(models):
        w = m.req_rate / m.slo
        s = m.model_size
        items.append({'w': w, 's': s, 'model': m, 'id': i})
        total_w += w
        total_s += s

    # 2. Lower Bound for K
    # sum(w) / (Total_Cap - sum(s)) is a theoretical lower bound
    total_cap = gpu_num * GPU_MEM_SIZE
    rem_global = total_cap - total_s
    if rem_global <= 1e-9:
        k_min = 0.0
    else:
        k_min = total_w / rem_global

    def get_actual_kvpr(placement):
        """Calculate the actual maximum KVPR of a placement."""
        max_k = 0.0
        for p in placement.values():
            w_sum = sum(m.req_rate / m.slo for m in p)
            s_sum = sum(m.model_size for m in p)
            rem = GPU_MEM_SIZE - s_sum
            if rem <= 1e-9:
                # If memory is full, pressure is infinite if there is load, else 0
                if w_sum > 1e-9: return float('inf')
                val = 0.0
            else:
                val = w_sum / rem
            if val > max_k: max_k = val
        return max_k

    def check_feasibility(k_target):
        """
        Check if models can be packed with KVPR <= k_target using Beam Search.
        """
        # Linear capacity limit
        cap_lin = k_target * GPU_MEM_SIZE
        
        # Calculate item weights for this K: v_i = w_i + K * s_i
        item_data = []
        for it in items:
            v = it['w'] + k_target * it['s']
            # Optimization: If a single item is larger than bin capacity, impossible
            if v > cap_lin + 1e-7: return None 
            item_data.append((v, it))
            
        # Sort items descending by linear size (BFD heuristic)
        item_data.sort(key=lambda x: x[0], reverse=True)
        
        # Beam Search
        # State: (score, loads_tuple, placement_tuple)
        # We use canonicalized loads (sorted) to break symmetry.
        
        # Initial State: 0 load on all GPUs, empty placements
        init_loads = tuple([0.0] * gpu_num)
        init_pl = tuple([[] for _ in range(gpu_num)])
        
        beam = [(0.0, init_loads, init_pl)]
        beam_width = 5
        
        for v, it in item_data:
            next_beam = []
            seen_states = set()
            
            for score, loads, pl in beam:
                # To break symmetry, we track tried load values for this state
                tried_loads = set()
                
                for i in range(gpu_num):
                    current_l = loads[i]
                    
                    # Optimization: Only try one bin for each unique load value
                    if current_l in tried_loads:
                        continue
                        
                    # Check capacity
                    if current_l + v <= cap_lin + 1e-7:
                        tried_loads.add(current_l)
                        
                        # Create new state components
                        new_l = current_l + v
                        new_bin_pl = pl[i] + [it['model']]
                        
                        # Canonicalize: Sort bins by load to maintain unique state representation
                        # Combine load and placement to sort together
                        temp_state = []
                        for j in range(gpu_num):
                            if i == j:
                                temp_state.append((new_l, new_bin_pl))
                            else:
                                temp_state.append((loads[j], pl[j]))
                        
                        # Sort by load descending
                        temp_state.sort(key=lambda x: x[0], reverse=True)
                        
                        new_loads = tuple(x[0] for x in temp_state)
                        new_pl = tuple(x[1] for x in temp_state)
                        
                        if new_loads not in seen_states:
                            seen_states.add(new_loads)
                            # Heuristic: Maximize sum of squares (Best-Fit preference)
                            # This encourages filling bins tightly, leaving others empty
                            new_score = sum(l*l for l in new_loads)
                            next_beam.append((new_score, new_loads, new_pl))
            
            # If no valid states for this item, this path fails
            if not next_beam:
                return None
            
            # Prune beam: Keep top W states
            next_beam.sort(key=lambda x: x[0], reverse=True)
            beam = next_beam[:beam_width]
            
        # Return best placement found
        best_pl_tuple = beam[0][2]
        final_placement = {i: best_pl_tuple[i] for i in range(gpu_num)}
        return final_placement

    def optimize_placement(placement):
        """
        Local search to reduce max KVPR of a valid placement.
        Tries to move models from the highest pressure GPU to others.
        """
        for _ in range(30): # Limited iterations
            # Identify max KVPR GPU
            max_k = -1.0
            max_idx = -1
            gpu_stats = []
            
            for i in range(gpu_num):
                w = sum(m.req_rate / m.slo for m in placement[i])
                s = sum(m.model_size for m in placement[i])
                rem = GPU_MEM_SIZE - s
                if rem <= 1e-9:
                    k = float('inf') if w > 1e-9 else 0.0
                else:
                    k = w / rem
                gpu_stats.append({'k': k, 'w': w, 's': s, 'idx': i})
                if k > max_k:
                    max_k = k
                    max_idx = i
            
            if max_idx == -1 or max_k == 0: break
            
            improved = False
            src = gpu_stats[max_idx]
            
            # Try to move a model from bottleneck source to any destination
            src_models = placement[src['idx']]
            for m_idx, m in enumerate(src_models):
                mw = m.req_rate / m.slo
                ms = m.model_size
                
                # Predict src K after removal
                ns_rem = GPU_MEM_SIZE - (src['s'] - ms)
                if ns_rem <= 1e-9:
                    ns_k = float('inf') if (src['w'] - mw) > 1e-9 else 0.0
                else:
                    ns_k = (src['w'] - mw) / ns_rem
                
                # Optimization: if removing doesn't help enough, skip (optional)
                
                for dst in gpu_stats:
                    if dst['idx'] == src['idx']: continue
                    
                    # Check memory fit
                    if dst['s'] + ms > GPU_MEM_SIZE: continue
                    
                    # Predict dst K after addition
                    nd_rem = GPU_MEM_SIZE - (dst['s'] + ms)
                    if nd_rem <= 1e-9:
                        nd_k = float('inf') if (dst['w'] + mw) > 1e-9 else 0.0
                    else:
                        nd_k = (dst['w'] + mw) / nd_rem
                    
                    # Move is valid if both new pressures are strictly less than current global max
                    # (This ensures we are essentially 'lowering the water level')
                    if max(ns_k, nd_k) < max_k - 1e-9:
                        # Apply move
                        model = placement[src['idx']].pop(m_idx)
                        placement[dst['idx']].append(model)
                        improved = True
                        break
                if improved: break
            
            if not improved: break
            
        return placement

    # Main Binary Search Loop
    high = 1e9
    
    # 1. Initial Feasibility Check
    bs_res = check_feasibility(high)
    if not bs_res:
         raise ValueError("Unable to place models on GPUs (insufficient total memory).")
    
    # 2. Refine bounds
    # Optimization: tighten high bound immediately
    bs_res = optimize_placement(bs_res)
    best_placement = bs_res
    high = min(high, get_actual_kvpr(bs_res))
    low = k_min
    
    # 3. Search
    for _ in range(25):
        mid = (low + high) / 2
        res = check_feasibility(mid)
        if res:
            # Found a valid placement, optimize it to see if we can go lower
            opt_res = optimize_placement(res)
            best_placement = opt_res
            
            # The actual max KVPR of the optimized solution is a valid upper bound
            actual_k = get_actual_kvpr(opt_res)
            high = min(mid, actual_k)
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

