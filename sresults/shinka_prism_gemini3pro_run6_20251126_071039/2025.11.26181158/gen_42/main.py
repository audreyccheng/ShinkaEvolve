# EVOLVE-BLOCK-START
import random
import math

GPU_MEM_SIZE = 80  # GB

def compute_model_placement(gpu_num, models):
    """
    Compute a model placement that minimizes the maximum KVPR across all GPUs.

    Combines Multi-Strategy Binary Search initialization with Multi-Start 
    Iterated Local Search using extensive neighborhoods (Shift, Swap11, Swap21, Swap22).
    """

    # 1. Data Preparation
    model_data = []
    for i, m in enumerate(models):
        model_data.append({
            'model': m,
            'l': m.req_rate / m.slo,
            's': m.model_size,
            'id': i
        })

    def get_kvpr(l, s):
        if s >= GPU_MEM_SIZE - 1e-6: return 1e15
        return l / (GPU_MEM_SIZE - s)

    # 2. Multi-Strategy Packing for Initialization
    def pack_models(target_k, sorting_strategy='weighted', randomize=False):
        capacity = target_k * GPU_MEM_SIZE
        
        # Prepare items with sorting key
        items = []
        for d in model_data:
            key = 0.0
            if sorting_strategy == 'weighted':
                # Linearized weight: L + K*S
                key = d['l'] + target_k * d['s']
            elif sorting_strategy == 'size':
                key = d['s']
            elif sorting_strategy == 'load':
                key = d['l']
            elif sorting_strategy == 'density':
                key = d['l'] / d['s'] if d['s'] > 0 else d['l'] * 1e6
            
            if randomize:
                key *= random.uniform(0.95, 1.05)
            items.append((key, d))
            
        # Sort descending by key
        items.sort(key=lambda x: x[0], reverse=True)
        
        gpu_l = [0.0] * gpu_num
        gpu_s = [0.0] * gpu_num
        gpu_items = [[] for _ in range(gpu_num)]
        
        # Tie-breaking for bins
        bin_indices = list(range(gpu_num))
        if randomize:
            random.shuffle(bin_indices)
            
        for _, item in items:
            best_idx = -1
            min_rem = float('inf')
            w_item = item['l'] + target_k * item['s']
            
            # Best Fit
            for i in bin_indices:
                if gpu_s[i] + item['s'] >= GPU_MEM_SIZE - 1e-6:
                    continue
                
                curr_w = gpu_l[i] + target_k * gpu_s[i]
                if curr_w + w_item <= capacity + 1e-9:
                    rem = capacity - (curr_w + w_item)
                    if rem < min_rem:
                        min_rem = rem
                        best_idx = i
            
            if best_idx != -1:
                gpu_l[best_idx] += item['l']
                gpu_s[best_idx] += item['s']
                gpu_items[best_idx].append(item)
            else:
                return None
        return gpu_items

    def solve_multistrategy(target_k):
        # Try multiple strategies to find ANY valid packing at target_k
        # This increases robustness of the binary search
        strategies = ['weighted', 'size', 'load']
        for strat in strategies:
            res = pack_models(target_k, sorting_strategy=strat, randomize=False)
            if res is not None:
                return res
        return None

    # 3. Binary Search for Baseline K
    low, high = 0.0, 1.0
    for _ in range(20):
        if solve_multistrategy(high) is not None: break
        low = high
        high *= 2.0
    else:
        high = 1e9 

    best_init_placement = None
    for _ in range(25):
        mid = (low + high) / 2
        res = solve_multistrategy(mid)
        if res:
            best_init_placement = res
            high = mid
        else:
            low = mid
            
    base_k = high
    if best_init_placement is None:
        best_init_placement = solve_multistrategy(base_k)
        if best_init_placement is None:
            # Should generally not happen unless physically impossible
            raise ValueError("Unable to fit models into GPU memory.")

    # 4. ILS / Local Search Phase
    best_global_plc = None
    best_global_score = float('inf')
    
    # Candidate generation
    # 1. The best result from Binary Search
    # 2. Randomized results using Weighted sort around base_k
    candidates = [best_init_placement]
    
    for _ in range(2): # Add randomized starts
        rnd = pack_models(base_k * 1.02, sorting_strategy='weighted', randomize=True)
        if rnd: candidates.append(rnd)
            
    # Run Local Search on candidates
    for start_sol in candidates:
        # State construction
        gpu_states = []
        for g in range(gpu_num):
            items = start_sol[g]
            gpu_states.append({
                'l': sum(x['l'] for x in items),
                's': sum(x['s'] for x in items),
                'items': list(items)
            })
            
        curr_max_k = max(get_kvpr(g['l'], g['s']) for g in gpu_states)
        
        # Steepest Descent Local Search
        for _ in range(150):
            if curr_max_k < 1e-9: break
            
            # Find bottleneck
            max_val = -1.0
            bottleneck = -1
            g_vals = []
            for g in range(gpu_num):
                v = get_kvpr(gpu_states[g]['l'], gpu_states[g]['s'])
                g_vals.append(v)
                if v > max_val:
                    max_val = v
                    bottleneck = g
                    
            best_move = None
            best_imp = 0.0
            
            src = gpu_states[bottleneck]
            
            def check_gain(sl, ss, tl, ts):
                if ss >= GPU_MEM_SIZE or ts >= GPU_MEM_SIZE: return -1.0
                n_max = max(get_kvpr(sl, ss), get_kvpr(tl, ts))
                if n_max < curr_max_k - 1e-7:
                    return curr_max_k - n_max
                return -1.0

            # Scan targets
            for t in range(gpu_num):
                if t == bottleneck: continue
                # Pruning: skip if target is already heavily loaded
                if g_vals[t] > curr_max_k * 0.98: continue
                
                tgt = gpu_states[t]
                
                # Shift
                for i, m in enumerate(src['items']):
                    gain = check_gain(src['l']-m['l'], src['s']-m['s'], tgt['l']+m['l'], tgt['s']+m['s'])
                    if gain > best_imp:
                        best_imp = gain
                        best_move = ('shift', i, t)
                        
                # Swap 1-1
                for i, m1 in enumerate(src['items']):
                    for j, m2 in enumerate(tgt['items']):
                        gain = check_gain(src['l']-m1['l']+m2['l'], src['s']-m1['s']+m2['s'],
                                          tgt['l']-m2['l']+m1['l'], tgt['s']-m2['s']+m1['s'])
                        if gain > best_imp:
                            best_imp = gain
                            best_move = ('swap11', i, t, j)
                            
                # Swap 2-1
                if len(src['items']) >= 2:
                    for i1 in range(len(src['items'])):
                        for i2 in range(i1+1, len(src['items'])):
                            m1, m2 = src['items'][i1], src['items'][i2]
                            pl, ps = m1['l']+m2['l'], m1['s']+m2['s']
                            for j, m3 in enumerate(tgt['items']):
                                gain = check_gain(src['l']-pl+m3['l'], src['s']-ps+m3['s'],
                                                  tgt['l']-m3['l']+pl, tgt['s']-m3['s']+ps)
                                if gain > best_imp:
                                    best_imp = gain
                                    best_move = ('swap21', i1, i2, t, j)

                # Swap 2-2
                if len(src['items']) >= 2 and len(tgt['items']) >= 2:
                    for i1 in range(len(src['items'])):
                        for i2 in range(i1+1, len(src['items'])):
                            m1, m2 = src['items'][i1], src['items'][i2]
                            sl, ss = m1['l']+m2['l'], m1['s']+m2['s']
                            for j1 in range(len(tgt['items'])):
                                for j2 in range(j1+1, len(tgt['items'])):
                                    m3, m4 = tgt['items'][j1], tgt['items'][j2]
                                    tl, ts = m3['l']+m4['l'], m3['s']+m4['s']
                                    gain = check_gain(src['l']-sl+tl, src['s']-ss+ts,
                                                      tgt['l']-tl+sl, tgt['s']-ts+ss)
                                    if gain > best_imp:
                                        best_imp = gain
                                        best_move = ('swap22', i1, i2, t, j1, j2)

            if best_move:
                mtype = best_move[0]
                if mtype == 'shift':
                    _, i, t = best_move
                    m = src['items'].pop(i)
                    tgt['items'].append(m)
                elif mtype == 'swap11':
                    _, i, t, j = best_move
                    src['items'][i], tgt['items'][j] = tgt['items'][j], src['items'][i]
                elif mtype == 'swap21':
                    _, i1, i2, t, j = best_move
                    # Pop higher index first to preserve lower index
                    m2 = src['items'].pop(i2)
                    m1 = src['items'].pop(i1)
                    m3 = tgt['items'].pop(j)
                    src['items'].append(m3)
                    tgt['items'].extend([m1, m2])
                elif mtype == 'swap22':
                    _, i1, i2, t, j1, j2 = best_move
                    m_s2 = src['items'].pop(i2)
                    m_s1 = src['items'].pop(i1)
                    m_t2 = tgt['items'].pop(j2)
                    m_t1 = tgt['items'].pop(j1)
                    src['items'].extend([m_t1, m_t2])
                    tgt['items'].extend([m_s1, m_s2])
                
                # Update cached sums
                src['l'] = sum(x['l'] for x in src['items']); src['s'] = sum(x['s'] for x in src['items'])
                tgt['l'] = sum(x['l'] for x in tgt['items']); tgt['s'] = sum(x['s'] for x in tgt['items'])
                
                curr_max_k = max(get_kvpr(g['l'], g['s']) for g in gpu_states)
            else:
                break
                
        if curr_max_k < best_global_score:
            best_global_score = curr_max_k
            best_global_plc = {g: [x['model'] for x in gpu_states[g]['items']] for g in range(gpu_num)}

    return best_global_plc
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