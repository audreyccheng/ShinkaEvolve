# EVOLVE-BLOCK-START
"""Model placement algorithm for minimizing maximum KV cache pressure across GPUs"""

GPU_MEM_SIZE = 80.0  # GB
import math
import random

def compute_model_placement(gpu_num, models):
    """
    Algorithm:
    1. Multi-Start Initialization:
       - Binary Search Linearization to find optimal K parameter.
       - Randomized instantiations of Linearized Bin Packing to generate diverse starting points.
       - Greedy heuristics (Size, Load) for robustness.
    2. Iterated Local Search (ILS):
       - Focuses on minimizing the vector of KVPRs (lexicographical optimization).
       - Operators:
         a) Steepest Descent Move: Scans all moves from bottleneck GPUs.
         b) Swap 2-1: Moves 2 small models out, brings 1 back.
         c) Swap 1-1: Standard swap.
         d) Swap 1-2: Moves 1 large model out, brings 2 back.
       - Perturbation (Ruins and Recreate):
         Empties the worst GPU and aggressively redistributes models to break local optima.
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
        # Lexicographical comparison vector
        return tuple(sorted((g.kvpr() for g in gpus), reverse=True))

    candidates = []

    # -------------------------------------------------------------------------
    # 1. Initialization
    # -------------------------------------------------------------------------
    
    # Linearized Bin Packing with Noise
    def solve_linearized_bp(target_k, noise=0.0):
        bin_cap = target_k * GPU_MEM_SIZE
        items = []
        for m in models:
            base_w = (m.req_rate / m.slo) + target_k * m.model_size
            w = base_w
            if noise > 0:
                w *= random.uniform(1.0 - noise, 1.0 + noise)
            items.append((w, m))
        items.sort(key=lambda x: x[0], reverse=True)
        
        bins = [GPUState(i) for i in range(gpu_num)]
        for w, m in items:
            best_idx = -1
            min_rem_linear = float('inf')
            for i in range(gpu_num):
                if not bins[i].can_fit(m.model_size): continue
                
                # Linear constraint check using deterministic values
                lin_use = bins[i].load + target_k * bins[i].used_mem
                item_lin = (m.req_rate / m.slo) + target_k * m.model_size
                
                if lin_use + item_lin <= bin_cap:
                    rem = bin_cap - (lin_use + item_lin)
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
    for _ in range(15):
        mid = (low + high) / 2
        if solve_linearized_bp(mid):
            best_k = mid
            high = mid
        else:
            low = mid
            
    # Generate candidates
    if best_k < 1e8:
        # Deterministic Best K
        res = solve_linearized_bp(best_k)
        if res: candidates.append(res)
        # Randomized Variations around Best K
        for _ in range(5):
            res = solve_linearized_bp(best_k, noise=0.05)
            if res: candidates.append(res)
            
    # Greedy Fallbacks
    strategies = [
        lambda m: m.model_size,
        lambda m: m.req_rate / m.slo
    ]
    for key_fn in strategies:
        gpus = [GPUState(i) for i in range(gpu_num)]
        valid = True
        for m in sorted(models, key=key_fn, reverse=True):
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
            else: valid = False; break
        if valid: candidates.append(gpus)

    # Fallback if nothing worked
    if not candidates:
        gpus = [GPUState(i) for i in range(gpu_num)]
        for m in sorted(models, key=lambda x: x.model_size, reverse=True):
            placed = False
            for i in range(gpu_num):
                if gpus[i].can_fit(m.model_size):
                    gpus[i].add(m); placed = True; break
            if not placed: raise ValueError("Models do not fit in GPU memory.")
        candidates.append(gpus)

    # Select Best Start
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
        iter_cnt += 1
        improved = False
        
        # Sort GPUs: worst first
        sorted_gpus = sorted(current_gpus, key=lambda g: g.kvpr(), reverse=True)
        sources = sorted_gpus[:4]
        destinations = sorted_gpus
        
        # --- Operator 1: Steepest Descent Move ---
        best_move = None
        best_move_gain = current_vector
        
        for src in sources:
            for i, model in enumerate(src.models):
                for dst in destinations:
                    if dst.id == src.id: continue
                    if dst.can_fit(model.model_size):
                        src.remove(i)
                        dst.add(model)
                        
                        vec = get_vector(current_gpus)
                        if vec < best_move_gain:
                            best_move_gain = vec
                            best_move = (src, i, dst, model)
                            
                        dst.remove(len(dst.models)-1)
                        src.restore_model(i, model)
        
        if best_move:
            src, idx, dst, model = best_move
            src.remove(idx)
            dst.add(model)
            current_vector = best_move_gain
            improved = True
            if current_vector < best_vector:
                best_vector = current_vector
                for k in range(gpu_num): best_gpus[k].copy_from(current_gpus[k])
        
        if improved: continue

        # --- Operator 2: Swap 2-1 (2 from Source, 1 from Dest) ---
        for src in sources[:2]:
            if len(src.models) < 2: continue
            for dst in destinations[::-1]:
                if dst.id == src.id or dst.kvpr() >= src.kvpr(): continue
                
                n_s = len(src.models)
                pair_found = False
                for i1 in range(n_s):
                    for i2 in range(i1+1, n_s):
                        m_a1 = src.models[i1]
                        m_a2 = src.models[i2]
                        for j, m_b in enumerate(dst.models):
                            # Capacity check
                            s_rem = GPU_MEM_SIZE - (src.used_mem - m_a1.model_size - m_a2.model_size + m_b.model_size)
                            d_rem = GPU_MEM_SIZE - (dst.used_mem - m_b.model_size + m_a1.model_size + m_a2.model_size)
                            if s_rem >= 0 and d_rem >= 0:
                                src.remove(i2); src.remove(i1)
                                dst.remove(j)
                                src.add(m_b)
                                dst.add(m_a1); dst.add(m_a2)
                                
                                vec = get_vector(current_gpus)
                                if vec < current_vector:
                                    current_vector = vec
                                    improved = True
                                    pair_found = True
                                    if current_vector < best_vector:
                                        best_vector = current_vector
                                        for k in range(gpu_num): best_gpus[k].copy_from(current_gpus[k])
                                    break
                                else:
                                    dst.remove(len(dst.models)-1); dst.remove(len(dst.models)-1)
                                    src.remove(len(src.models)-1)
                                    dst.restore_model(j, m_b)
                                    src.restore_model(i1, m_a1); src.restore_model(i2, m_a2)
                        if pair_found: break
                    if pair_found: break
                if improved: break
            if improved: break
        if improved: continue

        # --- Operator 3: Swap 1-1 ---
        for src in sources:
            for i, m_a in enumerate(src.models):
                for dst in destinations[::-1]:
                    if dst.id == src.id or dst.kvpr() >= src.kvpr(): continue
                    for j, m_b in enumerate(dst.models):
                        if (src.used_mem - m_a.model_size + m_b.model_size <= GPU_MEM_SIZE and 
                            dst.used_mem - m_b.model_size + m_a.model_size <= GPU_MEM_SIZE):
                            
                            src.remove(i); dst.remove(j)
                            src.add(m_b); dst.add(m_a)
                            
                            vec = get_vector(current_gpus)
                            if vec < current_vector:
                                current_vector = vec
                                improved = True
                                if current_vector < best_vector:
                                    best_vector = current_vector
                                    for k in range(gpu_num): best_gpus[k].copy_from(current_gpus[k])
                                break
                            else:
                                dst.remove(len(dst.models)-1); src.remove(len(src.models)-1)
                                dst.restore_model(j, m_b); src.restore_model(i, m_a)
                    if improved: break
                if improved: break
            if improved: break
        if improved: continue

        # --- Operator 4: Swap 1-2 (1 from Source, 2 from Dest) ---
        for src in sources[:2]:
            for i, m_a in enumerate(src.models):
                for dst in destinations[::-1]:
                    if dst.id == src.id or dst.kvpr() >= src.kvpr() or len(dst.models) < 2: continue
                    
                    pair_found = False
                    n_d = len(dst.models)
                    for j1 in range(n_d):
                        for j2 in range(j1+1, n_d):
                            m_b1 = dst.models[j1]
                            m_b2 = dst.models[j2]
                            
                            if (src.used_mem - m_a.model_size + m_b1.model_size + m_b2.model_size <= GPU_MEM_SIZE and
                                dst.used_mem - m_b1.model_size - m_b2.model_size + m_a.model_size <= GPU_MEM_SIZE):
                                
                                src.remove(i)
                                dst.remove(j2); dst.remove(j1)
                                src.add(m_b1); src.add(m_b2)
                                dst.add(m_a)
                                
                                vec = get_vector(current_gpus)
                                if vec < current_vector:
                                    current_vector = vec
                                    improved = True
                                    pair_found = True
                                    if current_vector < best_vector:
                                        best_vector = current_vector
                                        for k in range(gpu_num): best_gpus[k].copy_from(current_gpus[k])
                                    break
                                else:
                                    dst.remove(len(dst.models)-1)
                                    src.remove(len(src.models)-1); src.remove(len(src.models)-1)
                                    dst.restore_model(j1, m_b1); dst.restore_model(j2, m_b2)
                                    src.restore_model(i, m_a)
                        if pair_found: break
                    if pair_found: break
                if improved: break
            if improved: break
        if improved: continue

        # --- Perturbation: Ruins and Recreate ---
        if iter_cnt >= max_iter - 10: break
        
        worst = sorted_gpus[0]
        if not worst.models: break
        
        # Ruins: Completely empty the worst GPU
        removed_models = []
        while worst.models:
            removed_models.append(worst.remove(0))
            
        # Recreate: Try to distribute to other GPUs
        random.shuffle(removed_models)
        for m in removed_models:
            best_dest_idx = -1
            best_dest_val = float('inf')
            
            # Find Best Fit GPU among others
            for g in current_gpus:
                if g.id == worst.id: continue
                if g.can_fit(m.model_size):
                    rem = GPU_MEM_SIZE - (g.used_mem + m.model_size)
                    val = (g.load + m.req_rate/m.slo) / rem if rem > 1e-7 else float('inf')
                    if val < best_dest_val:
                        best_dest_val = val
                        best_dest_idx = g.id
            
            placed = False
            if best_dest_idx != -1:
                # Add to best found
                for g in current_gpus:
                    if g.id == best_dest_idx:
                        g.add(m)
                        placed = True
                        break
            
            if not placed:
                # If no fit elsewhere, put back in worst_gpu
                worst.add(m)
        
        current_vector = get_vector(current_gpus)
        if current_vector < best_vector:
            best_vector = current_vector
            for k in range(gpu_num): best_gpus[k].copy_from(current_gpus[k])

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
