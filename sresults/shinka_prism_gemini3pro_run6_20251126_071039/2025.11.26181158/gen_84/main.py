# EVOLVE-BLOCK-START
"""Model placement algorithm for minimizing maximum KV cache pressure across GPUs"""

GPU_MEM_SIZE = 80.0  # GB
import math
import random

def compute_model_placement(gpu_num, models):
    """
    Algorithm:
    1. Multi-Start Initialization:
       - Linearized Bin Packing with Binary Search for K (proxy for KVPR).
       - Generates diverse candidates by scanning K values and adding noise.
       - Deterministic and Randomized Greedy fallbacks.
       - Robust Safety Net ensuring 100% success rate.
    2. Iterated Local Search (ILS):
       - Target: Minimize Peak KVPR (Lexicographical Vector).
       - Operators: Move, Swap 1-1, Swap 1-2, Swap 2-1.
       - Optimization: Prioritize 'Source' GPUs (Bottlenecks) and 'Destination' GPUs (Underutilized).
    3. Dual-Target Perturbation:
       - Ruins the highest pressure GPU (Bottleneck) AND the lowest pressure GPU.
       - Re-distributes models to mix high-load items into available capacity.
    """

    class GPUState:
        # __slots__ optimization for speed
        __slots__ = ['id', 'models', 'load', 'used_mem', '_cached_kvpr', '_cached_rem']

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

    # --- 1. Initialization ---

    # A. Linearized Bin Packing (Load + K * Size <= K * Cap)
    def solve_linearized_bp(target_k, noise=0.0):
        bin_cap = target_k * GPU_MEM_SIZE
        items = []
        for m in models:
            # Weight w = Load + K*Size
            # If K is essentially KVPR, this linearizes the constraint
            base_w = (m.req_rate / m.slo) + target_k * m.model_size
            w = base_w
            if noise > 0:
                w *= random.uniform(1.0 - noise, 1.0 + noise)
            items.append((w, m, base_w))
        
        # Sort Best Fit Decreasing
        items.sort(key=lambda x: x[0], reverse=True)

        bins = [GPUState(i) for i in range(gpu_num)]
        for w, m, base_w in items:
            best_idx = -1
            min_rem = float('inf')
            
            # Find best bin
            for i in range(gpu_num):
                if not bins[i].can_fit(m.model_size): continue
                
                # Linear check: Load + K*Size + w <= K*Cap
                lin_use = bins[i].load + target_k * bins[i].used_mem
                if lin_use + base_w <= bin_cap:
                    rem = bin_cap - (lin_use + base_w)
                    if rem < min_rem:
                        min_rem = rem
                        best_idx = i
            
            if best_idx != -1:
                bins[best_idx].add(m)
            else:
                return None # Failed to fit
        return bins

    # Binary Search for Minimum Feasible K
    low, high = 0.0, 2000.0
    # Quick check high
    if solve_linearized_bp(high) is None: high = 1e9

    best_k = high
    bs_res = None
    
    # Run binary search
    for _ in range(16):
        mid = (low + high) / 2
        res = solve_linearized_bp(mid)
        if res:
            bs_res = res
            best_k = mid
            high = mid
        else:
            low = mid
    
    if bs_res: candidates.append(bs_res)

    # Generate variations around best_k
    # K corresponds to KVPR. Try relaxing packing (higher K) to reduce load concentration
    if best_k < 1e8:
        multipliers = [1.0, 1.1, 1.2, 1.5, 2.0]
        for mult in multipliers:
            res = solve_linearized_bp(best_k * mult, noise=0.02)
            if res: candidates.append(res)
    
    # B. Greedy Strategies
    strategies = [
        ('size', lambda m: m.model_size),
        ('load', lambda m: m.req_rate / m.slo),
        ('density', lambda m: (m.req_rate / m.slo) / m.model_size if m.model_size > 0 else 0)
    ]
    for _, key_fn in strategies:
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

    # C. Randomized Safety Net
    # Try random shuffles to ensure we have valid candidates
    attempts = 20 if not candidates else 5
    for _ in range(attempts):
        gpus = [GPUState(i) for i in range(gpu_num)]
        shuffled = list(models)
        random.shuffle(shuffled)
        valid = True
        for m in shuffled:
            best_idx = -1
            best_val = float('inf')
            for i in range(gpu_num):
                if gpus[i].can_fit(m.model_size):
                    # Minimize KVPR increase
                    rem = GPU_MEM_SIZE - (gpus[i].used_mem + m.model_size)
                    val = (gpus[i].load + m.req_rate/m.slo) / rem if rem > 1e-7 else float('inf')
                    if val < best_val:
                        best_val = val
                        best_idx = i
            if best_idx != -1: gpus[best_idx].add(m)
            else: valid = False; break
        if valid: candidates.append(gpus)

    # Absolute Safety Fallback: First Fit (Pure packing)
    if not candidates:
        for _ in range(100):
            gpus = [GPUState(i) for i in range(gpu_num)]
            shuffled = list(models)
            random.shuffle(shuffled)
            placed_all = True
            for m in shuffled:
                placed = False
                for i in range(gpu_num):
                    if gpus[i].can_fit(m.model_size):
                        gpus[i].add(m)
                        placed = True
                        break
                if not placed: 
                    placed_all = False
                    break
            if placed_all:
                candidates.append(gpus)
                break
        
    if not candidates:
        # If absolutely nothing works, return a placeholder (though this should technically raise an error or handle gracefully)
        # Based on prior context, raising error is standard if physically impossible
        raise ValueError("Could not find any valid placement.")

    # Select Best Start
    current_gpus = min(candidates, key=lambda g: get_vector(g))
    current_vector = get_vector(current_gpus)
    
    best_gpus = [GPUState(i) for i in range(gpu_num)]
    for i in range(gpu_num): best_gpus[i].copy_from(current_gpus[i])
    best_vector = current_vector

    # --- 2. Iterated Local Search ---
    iter_cnt = 0
    max_iter = 200

    while iter_cnt < max_iter:
        improved_step = False
        
        # Sort: Bottlenecks first
        sorted_gpus = sorted(current_gpus, key=lambda g: g.kvpr(), reverse=True)
        sources = sorted_gpus[:4]
        # Destinations: Lowest KVPR first (Best Fit)
        destinations = sorted_gpus[::-1]

        # Operator 1: Move
        best_move = None
        best_move_gain = current_vector
        for source in sources:
            for i, model in enumerate(source.models):
                for dest in destinations:
                    if dest.id == source.id: continue
                    if dest.can_fit(model.model_size):
                        source.remove(i); dest.add(model)
                        new_vec = get_vector(current_gpus)
                        if new_vec < best_move_gain:
                            best_move_gain = new_vec
                            best_move = (source, i, dest, model)
                        dest.remove(len(dest.models)-1); source.restore_model(i, model)
        if best_move:
            src, idx, dst, mdl = best_move
            src.remove(idx); dst.add(mdl)
            current_vector = best_move_gain
            improved_step = True
            if current_vector < best_vector:
                best_vector = current_vector
                for k in range(gpu_num): best_gpus[k].copy_from(current_gpus[k])
            iter_cnt += 1; continue

        # Operator 2: Swap 1-1
        best_swap = None
        best_swap_gain = current_vector
        for source in sources:
            for i, ma in enumerate(source.models):
                for dest in destinations:
                    if dest.id == source.id or dest.kvpr() >= source.kvpr(): continue
                    for j, mb in enumerate(dest.models):
                        # Filter: Don't bring heavy load to bottleneck
                        if (mb.req_rate/mb.slo) >= (ma.req_rate/ma.slo): continue
                        
                        if (source.used_mem - ma.model_size + mb.model_size <= GPU_MEM_SIZE and 
                            dest.used_mem - mb.model_size + ma.model_size <= GPU_MEM_SIZE):
                            
                            source.remove(i); dest.remove(j)
                            source.add(mb); dest.add(ma)
                            new_vec = get_vector(current_gpus)
                            if new_vec < best_swap_gain:
                                best_swap_gain = new_vec
                                best_swap = (source, i, ma, dest, j, mb)
                            dest.remove(len(dest.models)-1); source.remove(len(source.models)-1)
                            dest.restore_model(j, mb); source.restore_model(i, ma)
        if best_swap:
            src, i, ma, dst, j, mb = best_swap
            src.remove(i); dst.remove(j)
            src.add(mb); dst.add(ma)
            current_vector = best_swap_gain
            improved_step = True
            if current_vector < best_vector:
                best_vector = current_vector
                for k in range(gpu_num): best_gpus[k].copy_from(current_gpus[k])
            iter_cnt += 1; continue

        # Operator 3: Swap 1-2 (First Improvement)
        for source in sources[:3]:
            for i, ma in enumerate(source.models):
                for dest in destinations:
                    if dest.id == source.id or dest.kvpr() >= source.kvpr(): continue
                    if len(dest.models) < 2: continue
                    n_d = len(dest.models)
                    for j1 in range(n_d):
                        for j2 in range(j1+1, n_d):
                            mb1, mb2 = dest.models[j1], dest.models[j2]
                            s_mem = source.used_mem - ma.model_size + mb1.model_size + mb2.model_size
                            d_mem = dest.used_mem - mb1.model_size - mb2.model_size + ma.model_size
                            if s_mem <= GPU_MEM_SIZE and d_mem <= GPU_MEM_SIZE:
                                source.remove(i); dest.remove(j2); dest.remove(j1)
                                source.add(mb1); source.add(mb2); dest.add(ma)
                                new_vec = get_vector(current_gpus)
                                if new_vec < current_vector:
                                    current_vector = new_vec
                                    improved_step = True
                                else:
                                    dest.remove(len(dest.models)-1)
                                    source.remove(len(source.models)-1); source.remove(len(source.models)-1)
                                    dest.restore_model(j1, mb1); dest.restore_model(j2, mb2)
                            if improved_step: break
                        if improved_step: break
                    if improved_step: break
                if improved_step: break
            if improved_step: break
            
        if improved_step:
            if current_vector < best_vector:
                best_vector = current_vector
                for k in range(gpu_num): best_gpus[k].copy_from(current_gpus[k])
            iter_cnt += 1; continue
            
        # Operator 4: Swap 2-1
        for source in sources[:3]:
            if len(source.models) < 2: continue
            n_s = len(source.models)
            for i1 in range(n_s):
                for i2 in range(i1+1, n_s):
                    ma1, ma2 = source.models[i1], source.models[i2]
                    for dest in destinations:
                        if dest.id == source.id or dest.kvpr() >= source.kvpr(): continue
                        for j, mb in enumerate(dest.models):
                            s_mem = source.used_mem - ma1.model_size - ma2.model_size + mb.model_size
                            d_mem = dest.used_mem - mb.model_size + ma1.model_size + ma2.model_size
                            if s_mem <= GPU_MEM_SIZE and d_mem <= GPU_MEM_SIZE:
                                source.remove(i2); source.remove(i1); dest.remove(j)
                                source.add(mb); dest.add(ma1); dest.add(ma2)
                                new_vec = get_vector(current_gpus)
                                if new_vec < current_vector:
                                    current_vector = new_vec
                                    improved_step = True
                                else:
                                    dest.remove(len(dest.models)-1); dest.remove(len(dest.models)-1)
                                    source.remove(len(source.models)-1)
                                    dest.restore_model(j, mb)
                                    source.restore_model(i1, ma1); source.restore_model(i2, ma2)
                            if improved_step: break
                        if improved_step: break
                    if improved_step: break
                if improved_step: break
            if improved_step: break
            
        if improved_step:
            if current_vector < best_vector:
                best_vector = current_vector
                for k in range(gpu_num): best_gpus[k].copy_from(current_gpus[k])
            iter_cnt += 1; continue

        # --- Perturbation: Dual Ruin ---
        iter_cnt += 1
        if iter_cnt > max_iter - 10: break

        worst_gpu = sorted_gpus[0]
        if not worst_gpu.models: break
        
        # Select partner: The best GPU (most slack)
        best_gpu = sorted_gpus[-1]
        
        # Gather models
        displaced = []
        while worst_gpu.models: displaced.append(worst_gpu.remove(0))
        if best_gpu.id != worst_gpu.id:
            while best_gpu.models: displaced.append(best_gpu.remove(0))
            
        # Re-insert: Alternating sort order
        if iter_cnt % 2 == 0:
            displaced.sort(key=lambda m: m.model_size, reverse=True)
        else:
            displaced.sort(key=lambda m: m.req_rate/m.slo, reverse=True)
            
        targets = [g for g in current_gpus] # All GPUs are targets
        
        for m in displaced:
            best_dest = None
            best_dest_val = float('inf')
            
            for dest in targets:
                if dest.can_fit(m.model_size):
                    rem = GPU_MEM_SIZE - (dest.used_mem + m.model_size)
                    if rem > 1e-7:
                        val = (dest.load + m.req_rate/m.slo) / rem
                        if val < best_dest_val:
                            best_dest_val = val
                            best_dest = dest
            
            if best_dest:
                best_dest.add(m)
            else:
                # Should not happen if they fit before, but as safety:
                # Put back into worst_gpu (it was emptied) or best_gpu
                if worst_gpu.can_fit(m.model_size): worst_gpu.add(m)
                elif best_gpu.can_fit(m.model_size): best_gpu.add(m)
                else: 
                     # Force fit check failed - panic mode
                     # Just try first fit on any
                     for g in targets:
                         if g.can_fit(m.model_size): g.add(m); break
        
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