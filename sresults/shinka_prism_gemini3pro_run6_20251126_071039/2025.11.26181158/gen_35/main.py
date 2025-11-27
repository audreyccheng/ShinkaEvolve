# EVOLVE-BLOCK-START
"""Model placement algorithm for minimizing maximum KV cache pressure across GPUs"""

GPU_MEM_SIZE = 80.0  # GB
import math
import random

def compute_model_placement(gpu_num, models):
    """
    Algorithm:
    1. Multi-Start Initialization:
       - Linearized Bin Packing with Binary Search for optimal K.
       - Randomized variations around optimal K to explore the basin of attraction.
       - Greedy heuristics (Size, Load, Density) and a safe fallback.
    2. Iterated Local Search (ILS):
       - 'Best Improvement' Move strategy: Scans all possible moves from the bottleneck GPU 
         to find the one yielding the largest reduction in max KVPR.
       - Swap operators (1-1, 1-2, 2-1) to handle fragmentation.
       - 'Ruins and Recreate' Perturbation: Completely clears the bottleneck GPU and 
         redistributes models to escape deep local optima.
    """

    class GPUState:
        def __init__(self, gpu_id):
            self.id = gpu_id
            self.models = []
            self.load = 0.0
            self.used_mem = 0.0
            self._cached_kvpr = 0.0
            self._cached_rem = GPU_MEM_SIZE 

        def update_cache(self):
            self._cached_rem = GPU_MEM_SIZE - self.used_mem
            if self._cached_rem <= 1e-7:
                self._cached_kvpr = float('inf')
            else:
                self._cached_kvpr = self.load / self._cached_rem

        def can_fit(self, size):
            return self.used_mem + size <= GPU_MEM_SIZE

        def add(self, model):
            self.models.append(model)
            self.load += model.req_rate / model.slo
            self.used_mem += model.model_size
            self.update_cache()

        def remove(self, idx):
            model = self.models.pop(idx)
            self.load -= model.req_rate / model.slo
            self.used_mem -= model.model_size
            self.update_cache()
            return model
            
        def kvpr(self):
            return self._cached_kvpr

        def restore_model(self, idx, model):
            self.models.insert(idx, model)
            self.load += model.req_rate / model.slo
            self.used_mem += model.model_size
            self.update_cache()
            
        def copy_from(self, other):
            self.models = list(other.models)
            self.load = other.load
            self.used_mem = other.used_mem
            self._cached_kvpr = other._cached_kvpr
            self._cached_rem = other._cached_rem

    def get_vector(gpus):
        # Lexicographical comparison tuple
        return tuple(sorted((g.kvpr() for g in gpus), reverse=True))

    # -------------------------------------------------------------------------
    # 1. Initialization
    # -------------------------------------------------------------------------
    candidates = []

    def solve_linearized_bp(target_k, noise_level=0.0):
        bin_cap = target_k * GPU_MEM_SIZE
        items = []
        for m in models:
            # Weight = Load + K * Size
            base_w = (m.req_rate / m.slo) + target_k * m.model_size
            if noise_level > 0:
                base_w *= random.uniform(1.0 - noise_level, 1.0 + noise_level)
            items.append((base_w, m))
        
        # Best Fit Decreasing based on randomized weight
        items.sort(key=lambda x: x[0], reverse=True)
        
        bins = [GPUState(i) for i in range(gpu_num)]
        for w, m in items:
            best_idx = -1
            min_rem_linear = float('inf')
            
            for i in range(gpu_num):
                if not bins[i].can_fit(m.model_size): continue
                
                # Check linear capacity constraint
                # Use standard weight for constraint to respect the specific K
                std_w = (m.req_rate / m.slo) + target_k * m.model_size
                lin_usage = bins[i].load + target_k * bins[i].used_mem
                
                if lin_usage + std_w <= bin_cap:
                    rem = bin_cap - (lin_usage + std_w)
                    if rem < min_rem_linear:
                        min_rem_linear = rem
                        best_idx = i
            
            if best_idx != -1:
                bins[best_idx].add(m)
            else:
                return None
        return bins

    # Binary Search for K
    low, high = 0.0, 1000.0
    if solve_linearized_bp(high) is None: high = 1e9 
    
    best_k = high
    bs_res = None
    
    # 20 iterations to refine K
    for _ in range(20):
        mid = (low + high) / 2
        res = solve_linearized_bp(mid)
        if res:
            bs_res = res
            best_k = mid
            high = mid
        else:
            low = mid
            
    if bs_res: candidates.append(bs_res)

    # Randomized Multi-Start near Best K
    if bs_res and best_k < 1e8:
        for _ in range(10):
            res = solve_linearized_bp(best_k, noise_level=0.05)
            if res: candidates.append(res)
    
    # Greedy Strategies
    strategies = [
        ('size', lambda m: m.model_size),
        ('load', lambda m: m.req_rate / m.slo),
        ('density', lambda m: (m.req_rate / m.slo) / m.model_size if m.model_size > 1e-7 else 0)
    ]

    for _, key_fn in strategies:
        gpus = [GPUState(i) for i in range(gpu_num)]
        valid = True
        sorted_m = sorted(models, key=key_fn, reverse=True)
        for m in sorted_m:
            best_idx = -1
            best_val = float('inf')
            for i in range(gpu_num):
                if gpus[i].can_fit(m.model_size):
                    rem = GPU_MEM_SIZE - (gpus[i].used_mem + m.model_size)
                    val = (gpus[i].load + m.req_rate/m.slo) / rem if rem > 1e-7 else float('inf')
                    if val < best_val:
                        best_val = val
                        best_idx = i
            if best_idx != -1: gpus[best_idx].add(m)
            else:
                valid = False; break
        if valid: candidates.append(gpus)

    # Fallback: Pure Fit (Bin Packing on size)
    gpus_bf = [GPUState(i) for i in range(gpu_num)]
    valid_bf = True
    for m in sorted(models, key=lambda x: x.model_size, reverse=True):
        placed = False
        for i in range(gpu_num):
            if gpus_bf[i].can_fit(m.model_size):
                gpus_bf[i].add(m); placed = True; break
        if not placed: valid_bf = False; break
    if valid_bf: candidates.append(gpus_bf)

    if not candidates:
        raise ValueError("Models do not fit in GPU memory.")

    # Select best start
    current_gpus = min(candidates, key=lambda g: get_vector(g))
    current_vector = get_vector(current_gpus)
    
    best_gpus = [GPUState(i) for i in range(gpu_num)]
    for i in range(gpu_num): best_gpus[i].copy_from(current_gpus[i])
    best_vector = current_vector

    # -------------------------------------------------------------------------
    # 2. Iterated Local Search
    # -------------------------------------------------------------------------
    iter_cnt = 0
    max_iter = 150
    
    while iter_cnt < max_iter:
        improved_step = False
        
        sorted_gpus = sorted(current_gpus, key=lambda g: g.kvpr(), reverse=True)
        # Focus on the absolute bottleneck
        source = sorted_gpus[0]
        
        # --- Operator 1: Move (Steepest Descent / Best Improvement) ---
        # Look for the single best move from the bottleneck to any other GPU
        best_move_indices = None # (model_idx, dest_gpu)
        best_move_vec = current_vector

        for i, model in enumerate(source.models):
            for dest in current_gpus:
                if dest.id == source.id: continue
                if dest.can_fit(model.model_size):
                    # Try move
                    source.remove(i)
                    dest.add(model)
                    
                    new_vec = get_vector(current_gpus)
                    if new_vec < best_move_vec:
                        best_move_vec = new_vec
                        best_move_indices = (i, dest)
                    
                    # Revert
                    dest.remove(len(dest.models)-1)
                    source.restore_model(i, model)
        
        if best_move_indices:
            # Apply best move
            idx, dest_gpu = best_move_indices
            model = source.remove(idx)
            dest_gpu.add(model)
            current_vector = best_move_vec
            improved_step = True
            
            if current_vector < best_vector:
                best_vector = current_vector
                for k in range(gpu_num): best_gpus[k].copy_from(current_gpus[k])
            
            iter_cnt += 1
            continue # Restart loop to re-evaluate bottleneck

        # --- Operator 2: Swap 1-1 (First Improvement) ---
        sources = sorted_gpus[:2]
        destinations = sorted_gpus[::-1] 

        for s_gpu in sources:
            for i, m_a in enumerate(s_gpu.models):
                for d_gpu in destinations:
                    if d_gpu.id == s_gpu.id: continue
                    if d_gpu.kvpr() >= s_gpu.kvpr(): continue
                    
                    for j, m_b in enumerate(d_gpu.models):
                        s_mem = s_gpu.used_mem - m_a.model_size + m_b.model_size
                        d_mem = d_gpu.used_mem - m_b.model_size + m_a.model_size
                        if s_mem <= GPU_MEM_SIZE and d_mem <= GPU_MEM_SIZE:
                            s_gpu.remove(i)
                            d_gpu.remove(j)
                            s_gpu.add(m_b)
                            d_gpu.add(m_a)
                            
                            new_vec = get_vector(current_gpus)
                            if new_vec < current_vector:
                                current_vector = new_vec
                                improved_step = True
                                break
                            else:
                                d_gpu.remove(len(d_gpu.models)-1)
                                s_gpu.remove(len(s_gpu.models)-1)
                                d_gpu.restore_model(j, m_b)
                                s_gpu.restore_model(i, m_a)
                    if improved_step: break
                if improved_step: break
            if improved_step: break
        
        if improved_step:
            if current_vector < best_vector:
                best_vector = current_vector
                for k in range(gpu_num): best_gpus[k].copy_from(current_gpus[k])
            iter_cnt += 1
            continue

        # --- Operator 3: Swap 2-1 and 1-2 ---
        for s_gpu in sources[:1]: # Top bottleneck only
            # Try 1-2 (Source gives 1, Dest gives 2)
            for i, m_a in enumerate(s_gpu.models):
                for d_gpu in destinations:
                    if d_gpu.id == s_gpu.id: continue
                    if d_gpu.kvpr() >= s_gpu.kvpr(): continue
                    if len(d_gpu.models) < 2: continue
                    
                    n_d = len(d_gpu.models)
                    pair_found = False
                    for j1 in range(n_d):
                        for j2 in range(j1+1, n_d):
                            m_b1 = d_gpu.models[j1]
                            m_b2 = d_gpu.models[j2]
                            
                            s_mem = s_gpu.used_mem - m_a.model_size + m_b1.model_size + m_b2.model_size
                            d_mem = d_gpu.used_mem - m_b1.model_size - m_b2.model_size + m_a.model_size
                            
                            if s_mem <= GPU_MEM_SIZE and d_mem <= GPU_MEM_SIZE:
                                s_gpu.remove(i)
                                d_gpu.remove(j2)
                                d_gpu.remove(j1)
                                s_gpu.add(m_b1)
                                s_gpu.add(m_b2)
                                d_gpu.add(m_a)
                                
                                new_vec = get_vector(current_gpus)
                                if new_vec < current_vector:
                                    current_vector = new_vec
                                    improved_step = True
                                    pair_found = True
                                    break
                                else:
                                    d_gpu.remove(len(d_gpu.models)-1)
                                    s_gpu.remove(len(s_gpu.models)-1)
                                    s_gpu.remove(len(s_gpu.models)-1)
                                    d_gpu.restore_model(j1, m_b1)
                                    d_gpu.restore_model(j2, m_b2)
                        if pair_found: break
                    if pair_found: break
                if improved_step: break
            if improved_step: break

            # Try 2-1 (Source gives 2, Dest gives 1)
            if len(s_gpu.models) >= 2:
                n_s = len(s_gpu.models)
                pair_found = False
                for i1 in range(n_s):
                    for i2 in range(i1+1, n_s):
                        m_a1 = s_gpu.models[i1]
                        m_a2 = s_gpu.models[i2]
                        
                        for d_gpu in destinations:
                            if d_gpu.id == s_gpu.id: continue
                            if d_gpu.kvpr() >= s_gpu.kvpr(): continue
                            
                            for j, m_b in enumerate(d_gpu.models):
                                s_mem = s_gpu.used_mem - m_a1.model_size - m_a2.model_size + m_b.model_size
                                d_mem = d_gpu.used_mem - m_b.model_size + m_a1.model_size + m_a2.model_size
                                
                                if s_mem <= GPU_MEM_SIZE and d_mem <= GPU_MEM_SIZE:
                                    s_gpu.remove(i2)
                                    s_gpu.remove(i1)
                                    d_gpu.remove(j)
                                    s_gpu.add(m_b)
                                    d_gpu.add(m_a1)
                                    d_gpu.add(m_a2)
                                    
                                    new_vec = get_vector(current_gpus)
                                    if new_vec < current_vector:
                                        current_vector = new_vec
                                        improved_step = True
                                        pair_found = True
                                        break
                                    else:
                                        d_gpu.remove(len(d_gpu.models)-1)
                                        d_gpu.remove(len(d_gpu.models)-1)
                                        s_gpu.remove(len(s_gpu.models)-1)
                                        d_gpu.restore_model(j, m_b)
                                        s_gpu.restore_model(i1, m_a1)
                                        s_gpu.restore_model(i2, m_a2)
                            if pair_found: break
                        if pair_found: break
                    if pair_found: break
            if improved_step: break

        if improved_step:
            if current_vector < best_vector:
                best_vector = current_vector
                for k in range(gpu_num): best_gpus[k].copy_from(current_gpus[k])
            iter_cnt += 1
            continue

        # --- PERTURBATION: Ruins and Recreate ---
        iter_cnt += 1
        if iter_cnt > max_iter - 10: break
        
        worst_gpu = sorted_gpus[0]
        if not worst_gpu.models: break
        
        # Ruin: Remove all models from worst GPU
        displaced_models = list(worst_gpu.models)
        while worst_gpu.models:
            worst_gpu.remove(0)
            
        random.shuffle(displaced_models)
        
        # Recreate: Distribute to others if possible
        other_gpus = [g for g in current_gpus if g.id != worst_gpu.id]
        
        for m in displaced_models:
            # Try to place in other GPUs minimizing local pressure increase
            best_dest = None
            best_dest_val = float('inf')
            
            for g in other_gpus:
                if g.can_fit(m.model_size):
                    # Metric: New KVPR if added
                    rem = GPU_MEM_SIZE - (g.used_mem + m.model_size)
                    if rem > 1e-7:
                        val = (g.load + m.req_rate/m.slo) / rem
                        if val < best_dest_val:
                            best_dest_val = val
                            best_dest = g
            
            if best_dest:
                best_dest.add(m)
            else:
                # Must put back in source
                worst_gpu.add(m)
                
        current_vector = get_vector(current_gpus)

    return {g.id: g.models for g in best_gpus}
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