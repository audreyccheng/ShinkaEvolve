# EVOLVE-BLOCK-START
"""Model placement algorithm for minimizing maximum KV cache pressure across GPUs"""

GPU_MEM_SIZE = 80.0  # GB
import math
import random

def compute_model_placement(gpu_num, models):
    """
    Algorithm:
    1. Robust Initialization:
       - Grid Search over Linearization Parameter K to find optimal balance between 
         Load Balancing (K=0) and Bin Packing (K=Large).
       - Deterministic Greedy strategies (Size, Load, Density) with KVPR minimization objective.
       - Reliability Safety Net: Randomized First-Fit with restarts if deterministic heuristics fail.
       
    2. Iterated Local Search (ILS):
       - Variable Neighborhood Descent focusing on Top Bottleneck GPUs.
       - Operators: Move, Swap 1-1, Swap 2-1 (Source gives 2).
       - Optimization: Prunes search space by filtering for feasible swaps and focusing on bottlenecks.
       
    3. Multi-GPU Perturbation:
       - Ruin and Recreate strategy that empties the top 2 bottleneck GPUs simultaneously.
       - Redstributes models using Best-Fit Decreasing to escape local optima.
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
        # Lexicographical vector: (max_kvpr, 2nd_max_kvpr, ...)
        return tuple(sorted((g.kvpr() for g in gpus), reverse=True))

    candidates = []

    # -------------------------------------------------------------------------
    # 1. Initialization
    # -------------------------------------------------------------------------

    # A. Grid Search Linearization (Balancing Packing vs Load)
    # K factors: 0 (Load Only) -> Large (Size Only)
    k_factors = [0.0, 0.5, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0, 100.0, 500.0]
    
    for k in k_factors:
        # Sort items by linearized weight
        # We also create a "score" for Best Fit based on this K
        items = []
        for m in models:
            w = (m.req_rate / m.slo) + k * m.model_size
            items.append((w, m))
        items.sort(key=lambda x: x[0], reverse=True)
        
        gpus = [GPUState(i) for i in range(gpu_num)]
        valid = True
        
        for _, m in items:
            best_idx = -1
            min_score = float('inf')
            
            for i in range(gpu_num):
                if gpus[i].can_fit(m.model_size):
                    # Linearized Load Minimization (Best Fit for this K)
                    # lin_load = real_load + k * used_mem
                    new_lin_load = (gpus[i].load + m.req_rate/m.slo) + k * (gpus[i].used_mem + m.model_size)
                    
                    if new_lin_load < min_score:
                        min_score = new_lin_load
                        best_idx = i
            
            if best_idx != -1:
                gpus[best_idx].add(m)
            else:
                valid = False
                break
        
        if valid:
            candidates.append(gpus)

    # B. Deterministic Greedy (Objective-Aware)
    # Strategies: Size, Load, Density
    strategies = [
        ('Size', lambda m: m.model_size),
        ('Load', lambda m: m.req_rate / m.slo),
        ('Density', lambda m: (m.req_rate / m.slo) / m.model_size if m.model_size > 0 else 0)
    ]
    
    for _, key_fn in strategies:
        gpus = [GPUState(i) for i in range(gpu_num)]
        valid = True
        # Sort desc
        for m in sorted(models, key=key_fn, reverse=True):
            best_idx = -1
            best_val = float('inf')
            # Minimize resulting KVPR
            for i in range(gpu_num):
                if gpus[i].can_fit(m.model_size):
                    rem = GPU_MEM_SIZE - (gpus[i].used_mem + m.model_size)
                    if rem > 1e-7:
                        val = (gpus[i].load + m.req_rate/m.slo) / rem
                        if val < best_val:
                            best_val = val
                            best_idx = i
            if best_idx != -1: gpus[best_idx].add(m)
            else: valid = False; break
        if valid: candidates.append(gpus)

    # C. Reliability Safety Net (Randomized Retry)
    # If standard heuristics failed, try random shuffles to find *any* valid placement
    retry_limit = 20
    retry_count = 0
    while (not candidates) and (retry_count < retry_limit):
        retry_count += 1
        gpus = [GPUState(i) for i in range(gpu_num)]
        m_shuffled = list(models)
        random.shuffle(m_shuffled)
        valid = True
        for m in m_shuffled:
            placed = False
            # First Fit
            for i in range(gpu_num):
                if gpus[i].can_fit(m.model_size):
                    gpus[i].add(m)
                    placed = True
                    break
            if not placed:
                valid = False
                break
        if valid: candidates.append(gpus)

    # Fallback if everything fails
    if not candidates:
        gpus = [GPUState(i) for i in range(gpu_num)]
        for m in sorted(models, key=lambda x: x.model_size, reverse=True):
            placed = False
            for i in range(gpu_num):
                if gpus[i].can_fit(m.model_size):
                    gpus[i].add(m); placed = True; break
            if not placed: raise ValueError("Models do not fit in GPU memory.")
        candidates.append(gpus)

    # Select best starting point
    current_gpus = min(candidates, key=lambda g: get_vector(g))
    current_vector = get_vector(current_gpus)

    best_gpus = [GPUState(i) for i in range(gpu_num)]
    for i in range(gpu_num): best_gpus[i].copy_from(current_gpus[i])
    best_vector = current_vector

    # -------------------------------------------------------------------------
    # 2. Iterated Local Search (ILS)
    # -------------------------------------------------------------------------
    iter_cnt = 0
    max_iter = 150

    while iter_cnt < max_iter:
        improved_step = False
        
        # Sort GPUs by KVPR
        sorted_gpus = sorted(current_gpus, key=lambda g: g.kvpr(), reverse=True)
        # Identify Bottlenecks
        sources = sorted_gpus[:3] 
        destinations = sorted_gpus[::-1]

        # --- Operator 1: Move (Best Improvement) ---
        best_move = None
        best_move_vec = current_vector

        for source in sources:
            for i, model in enumerate(source.models):
                for dest in destinations:
                    if dest.id == source.id: continue
                    if dest.can_fit(model.model_size):
                        source.remove(i)
                        dest.add(model)
                        
                        vec = get_vector(current_gpus)
                        if vec < best_move_vec:
                            best_move_vec = vec
                            best_move = (source, i, dest, model)
                        
                        dest.remove(len(dest.models)-1)
                        source.restore_model(i, model)
        
        if best_move:
            src, idx, dst, mdl = best_move
            src.remove(idx)
            dst.add(mdl)
            current_vector = best_move_vec
            improved_step = True
            if current_vector < best_vector:
                best_vector = current_vector
                for k in range(gpu_num): best_gpus[k].copy_from(current_gpus[k])
            iter_cnt += 1; continue

        # --- Operator 2: Swap 1-1 (Best Improvement) ---
        best_swap = None
        best_swap_vec = current_vector
        
        for source in sources:
            for i, m_a in enumerate(source.models):
                for dest in destinations:
                    if dest.id == source.id: continue
                    if dest.kvpr() >= source.kvpr(): continue 
                    
                    for j, m_b in enumerate(dest.models):
                        # Filter for likely good moves
                        if (source.used_mem - m_a.model_size + m_b.model_size <= GPU_MEM_SIZE) and \
                           (dest.used_mem - m_b.model_size + m_a.model_size <= GPU_MEM_SIZE):
                            
                            source.remove(i)
                            dest.remove(j)
                            source.add(m_b)
                            dest.add(m_a)
                            
                            vec = get_vector(current_gpus)
                            if vec < best_swap_vec:
                                best_swap_vec = vec
                                best_swap = (source, i, m_a, dest, j, m_b)
                                
                            dest.remove(len(dest.models)-1)
                            source.remove(len(source.models)-1)
                            dest.restore_model(j, m_b)
                            source.restore_model(i, m_a)
        
        if best_swap:
            src, i, ma, dst, j, mb = best_swap
            src.remove(i)
            dst.remove(j)
            src.add(mb)
            dst.add(ma)
            current_vector = best_swap_vec
            improved_step = True
            if current_vector < best_vector:
                best_vector = current_vector
                for k in range(gpu_num): best_gpus[k].copy_from(current_gpus[k])
            iter_cnt += 1; continue

        # --- Operator 3: Swap 2-1 (Source gives 2) ---
        # Only top 2 bottlenecks to control complexity
        for source in sources[:2]:
            if len(source.models) < 2: continue
            
            for i1 in range(len(source.models)):
                for i2 in range(i1+1, len(source.models)):
                    m_a1 = source.models[i1]
                    m_a2 = source.models[i2]
                    
                    for dest in destinations:
                        if dest.id == source.id: continue
                        if dest.kvpr() >= source.kvpr(): continue
                        
                        for j, m_b in enumerate(dest.models):
                            if (source.used_mem - m_a1.model_size - m_a2.model_size + m_b.model_size <= GPU_MEM_SIZE) and \
                               (dest.used_mem - m_b.model_size + m_a1.model_size + m_a2.model_size <= GPU_MEM_SIZE):
                                
                                source.remove(i2) # Order matters
                                source.remove(i1)
                                dest.remove(j)
                                source.add(m_b)
                                dest.add(m_a1)
                                dest.add(m_a2)
                                
                                vec = get_vector(current_gpus)
                                if vec < current_vector:
                                    current_vector = vec
                                    improved_step = True
                                    if current_vector < best_vector:
                                        best_vector = current_vector
                                        for k in range(gpu_num): best_gpus[k].copy_from(current_gpus[k])
                                    break
                                else:
                                    dest.remove(len(dest.models)-1)
                                    dest.remove(len(dest.models)-1)
                                    source.remove(len(source.models)-1)
                                    dest.restore_model(j, m_b)
                                    source.restore_model(i1, m_a1)
                                    source.restore_model(i2, m_a2)
                        if improved_step: break
                    if improved_step: break
                if improved_step: break
            if improved_step: break
        
        if improved_step:
            iter_cnt += 1
            continue

        # --- Perturbation: Multi-GPU Ruin & Recreate ---
        iter_cnt += 1
        if iter_cnt > max_iter - 10: break
        
        # Ruin top 2 bottlenecks to pool resources
        ruined_gpus = sorted_gpus[:2]
        floating_models = []
        for g in ruined_gpus:
            while g.models:
                floating_models.append(g.remove(0))
        
        # Recreate: Try to distribute floating models to any GPU
        # Sort floating models by Size (hardest to fit first)
        floating_models.sort(key=lambda m: m.model_size, reverse=True)
        
        success = True
        
        for m in floating_models:
            best_dest = None
            best_val = float('inf')
            
            # Check all GPUs
            for g in current_gpus:
                if g.can_fit(m.model_size):
                    rem = GPU_MEM_SIZE - (g.used_mem + m.model_size)
                    if rem > 1e-7:
                        val = (g.load + m.req_rate/m.slo) / rem
                        if val < best_val:
                            best_val = val
                            best_dest = g
            
            if best_dest:
                best_dest.add(m)
            else:
                success = False
                break
        
        if success:
            current_vector = get_vector(current_gpus)
            if current_vector < best_vector:
                best_vector = current_vector
                for k in range(gpu_num): best_gpus[k].copy_from(current_gpus[k])
        else:
            # Recreate failed (fragmentation), revert
            # Since we modified potentially many GPUs, full revert to best_vector is safest
            for k in range(gpu_num): current_gpus[k].copy_from(best_gpus[k])
            current_vector = best_vector
            
            # Apply a smaller perturbation to unstuck (Swap random pair between random GPUs)
            if gpu_num > 1:
                g1, g2 = random.sample(current_gpus, 2)
                if g1.models and g2.can_fit(g1.models[0].model_size):
                    m = g1.remove(0)
                    g2.add(m)
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