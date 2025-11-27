# EVOLVE-BLOCK-START
"""Model placement algorithm for minimizing maximum KV cache pressure using Hybrid BFD Packing and SA with Ruin & Recreate"""

import copy
import random
import math

GPU_MEM_SIZE = 80.0  # GB

def compute_model_placement(gpu_num, models):
    """
    Minimizes max KVPR using Binary Search with FFD/BFD packing heuristics
    followed by Simulated Annealing with Ruin & Recreate refinement.
    """
    # 1. Validation and Setup
    total_size = sum(m.model_size for m in models)
    if total_size > gpu_num * GPU_MEM_SIZE:
        raise ValueError("Total model size exceeds total GPU memory capacity.")
        
    # Prepare items: (req_rate/slo, model_size, model_obj)
    # Store as dictionary for cleaner access
    items = [{'w': m.req_rate / m.slo, 's': m.model_size, 'm': m} for m in models]
    
    # 2. Binary Search for Initial Solution
    # Determine bounds
    total_w = sum(x['w'] for x in items)
    slack = gpu_num * GPU_MEM_SIZE - total_size
    
    if slack < 1e-5:
        # Extremely tight
        low, high = 0.0, 10000.0
    else:
        avg_k = total_w / slack
        low, high = avg_k * 0.5, avg_k * 5.0
    high = max(high, 50.0)
    
    best_placement = None
    feasible_high = False
    
    # Find a valid upper bound first
    for _ in range(15):
        feasible, placement = _check_feasibility_robust(gpu_num, items, high)
        if feasible:
            best_placement = placement
            feasible_high = True
            break
        low = high
        high *= 2.0
        
    if not feasible_high:
        raise ValueError("Unable to place models. Constraints likely too tight.")
        
    # Binary Search
    for _ in range(25):
        mid = (low + high) / 2.0
        feasible, placement = _check_feasibility_robust(gpu_num, items, mid)
        if feasible:
            best_placement = placement
            high = mid
        else:
            low = mid
            
    # Convert list placement to dictionary map
    placement_map = {i: best_placement[i] for i in range(gpu_num)}
    
    # 3. Refinement: Simulated Annealing with Ruin & Recreate
    return _sa_ruin_recreate(gpu_num, placement_map)

def _check_feasibility_robust(gpu_num, items, K):
    """
    Checks if items can be packed with target KVPR 'K' using multiple heuristics.
    Includes both FFD and BFD strategies on Virtual Size, Physical Size, and Density.
    """
    virtual_cap = K * GPU_MEM_SIZE
    
    # Pack Items tuple: (virtual_size, physical_size, load, density, model)
    # virtual_size = w + K * s
    pack_items = []
    for x in items:
        v = x['w'] + K * x['s']
        d = x['w'] / (x['s'] + 1e-6)
        pack_items.append((v, x['s'], x['w'], d, x['m']))
        
    # Define strategies: (Sort Key Index, Reverse, Packing Function)
    # Key 0: Virtual Size
    # Key 1: Physical Size
    # Key 3: Density
    strategies = [
        (0, True, _pack_ffd), # Virtual Size FFD
        (0, True, _pack_bfd), # Virtual Size BFD
        (1, True, _pack_ffd), # Physical Size FFD
        (1, True, _pack_bfd), # Physical Size BFD
        (3, True, _pack_ffd), # Density FFD
    ]
    
    for key_idx, reverse, pack_func in strategies:
        # Sort items
        pack_items.sort(key=lambda x: x[key_idx], reverse=reverse)
        # Try packing
        res = pack_func(gpu_num, pack_items, virtual_cap)
        if res: return True, res
            
    return False, None

def _pack_ffd(gpu_num, items, v_cap):
    """First Fit Decreasing"""
    bins_v = [0.0] * gpu_num
    bins_p = [0.0] * gpu_num
    placement = [[] for _ in range(gpu_num)]
    
    for v, s, w, d, m in items:
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
    """Best Fit Decreasing"""
    bins_v = [0.0] * gpu_num
    bins_p = [0.0] * gpu_num
    placement = [[] for _ in range(gpu_num)]
    
    for v, s, w, d, m in items:
        best_bin = -1
        min_rem_v = float('inf')
        
        for i in range(gpu_num):
            if bins_p[i] + s <= GPU_MEM_SIZE and bins_v[i] + v <= v_cap + 1e-7:
                # Minimize remaining virtual capacity (tighter fit)
                rem = v_cap - (bins_v[i] + v)
                if rem < min_rem_v:
                    min_rem_v = rem
                    best_bin = i
        
        if best_bin != -1:
            bins_v[best_bin] += v
            bins_p[best_bin] += s
            placement[best_bin].append(m)
        else:
            return None
    return placement

def _sa_ruin_recreate(gpu_num, placement_map):
    """
    Refines placement using Simulated Annealing with Ruin & Recreate perturbation.
    """
    # State tracking structure: {gpu_id: {'models': [], 's': 0.0, 'w': 0.0}}
    # Initialize state
    state = {}
    for i in range(gpu_num):
        mods = placement_map[i]
        state[i] = {
            'models': list(mods),
            's': sum(m.model_size for m in mods),
            'w': sum(m.req_rate/m.slo for m in mods)
        }
        
    def get_max_k(st):
        max_k = 0.0
        idx = -1
        for i in range(gpu_num):
            rem = GPU_MEM_SIZE - st[i]['s']
            if rem <= 1e-7: 
                k = 1e9
            else:
                k = st[i]['w'] / rem
            if k > max_k:
                max_k = k
                idx = i
        return max_k, idx

    current_k, bottleneck_idx = get_max_k(state)
    best_state = copy.deepcopy(state)
    best_k = current_k
    
    # SA Parameters
    temp = current_k * 0.15
    alpha = 0.92
    iterations = 400
    
    for _ in range(iterations):
        # Create Neighbor
        # Deep copy is expensive, so we use a structural copy logic inside moves
        # For simplicity and correctness in this snippet, we use deepcopy of the state dict
        # Optimization: Manually copy the dict and the lists inside
        neighbor = {i: {'models': list(state[i]['models']), 's': state[i]['s'], 'w': state[i]['w']} for i in range(gpu_num)}
        
        nk, n_bottleneck = get_max_k(neighbor)
        
        # Move Selection
        r = random.random()
        move_success = False
        
        if r < 0.5: 
            # Strategy: Ruin and Recreate
            # Select bottleneck + random others
            targets = {n_bottleneck}
            others = list(range(gpu_num))
            random.shuffle(others)
            for t in others:
                if t not in targets:
                    targets.add(t)
                if len(targets) >= 3: break # Ruin 3 GPUs
            
            # Extract models
            removed = []
            for t in targets:
                removed.extend(neighbor[t]['models'])
                neighbor[t]['models'] = []
                neighbor[t]['s'] = 0.0
                neighbor[t]['w'] = 0.0
            
            # Sort for greedy repack: High Virtual Size first (approx with current K)
            # v = w + K*s. This balances load and size pressure.
            removed.sort(key=lambda m: (m.req_rate/m.slo) + current_k * m.model_size, reverse=True)
            
            feasible_repack = True
            for m in removed:
                best_dst = -1
                best_score = float('inf')
                m_w = m.req_rate/m.slo
                m_s = m.model_size
                
                # Try to place in any GPU
                for dst in range(gpu_num):
                    if neighbor[dst]['s'] + m_s <= GPU_MEM_SIZE:
                        # Score: Resulting local KVPR
                        rem = GPU_MEM_SIZE - (neighbor[dst]['s'] + m_s)
                        if rem < 1e-7: score = 1e9
                        else: score = (neighbor[dst]['w'] + m_w) / rem
                        
                        if score < best_score:
                            best_score = score
                            best_dst = dst
                
                if best_dst != -1:
                    neighbor[best_dst]['models'].append(m)
                    neighbor[best_dst]['s'] += m_s
                    neighbor[best_dst]['w'] += m_w
                else:
                    feasible_repack = False
                    break
            
            if feasible_repack:
                move_success = True
                
        else:
            # Strategy: Simple Move (Bottleneck -> Random Valid)
            if neighbor[n_bottleneck]['models']:
                m_idx = random.randint(0, len(neighbor[n_bottleneck]['models'])-1)
                m = neighbor[n_bottleneck]['models'][m_idx]
                
                # Try to find a valid destination
                destinations = list(range(gpu_num))
                random.shuffle(destinations)
                
                for dst in destinations:
                    if dst == n_bottleneck: continue
                    if neighbor[dst]['s'] + m.model_size <= GPU_MEM_SIZE:
                        # Perform Move
                        neighbor[n_bottleneck]['models'].pop(m_idx)
                        neighbor[n_bottleneck]['s'] -= m.model_size
                        neighbor[n_bottleneck]['w'] -= m.req_rate/m.slo
                        
                        neighbor[dst]['models'].append(m)
                        neighbor[dst]['s'] += m.model_size
                        neighbor[dst]['w'] += m.req_rate/m.slo
                        move_success = True
                        break
        
        if move_success:
            # Evaluate
            neighbor_k, _ = get_max_k(neighbor)
            delta = neighbor_k - current_k
            
            accept = False
            if delta < 0:
                accept = True
            else:
                prob = math.exp(-delta * 20.0 / (temp + 1e-9))
                if random.random() < prob:
                    accept = True
            
            if accept:
                state = neighbor
                current_k = neighbor_k
                if current_k < best_k:
                    best_k = current_k
                    best_state = copy.deepcopy(state)
        
        # Cool down
        temp *= alpha
        
    # Reconstruct format
    result = {i: best_state[i]['models'] for i in range(gpu_num)}
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

