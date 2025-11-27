# EVOLVE-BLOCK-START
"""Model placement algorithm for minimizing maximum KV cache pressure across GPUs"""

GPU_MEM_SIZE = 80.0  # GB
import math
import random

def compute_model_placement(gpu_num, models):
    """
    Algorithm:
    1. Multi-Start Initialization:
       - Linearized Bin Packing: Finds minimum feasible K, then scans [K, K*1.2] 
         to pick the start state with the best KVPR vector.
       - Greedy Strategies: Deterministic and Randomized strategies for diversity.
    2. Iterated Local Search (ILS):
       - Operators: Move, Swap 1-1, Swap 1-2, Swap 2-1.
       - Prioritizes reducing load on bottleneck GPUs.
    3. Dual-Ruin Perturbation:
       - Clears both the highest-load (bottleneck) and lowest-load (underutilized) GPUs.
       - Re-inserts models using alternating sorting strategies (Size vs Load).
       - This "breathing" mechanism helps move large/heavy items from bottlenecks 
         to spaces created on underutilized GPUs.
    """

    class GPUState:
        # Use slots for memory efficiency and slightly faster attribute access
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
        # Lexicographical minimization of this vector minimizes max_kvpr
        return tuple(sorted((g.kvpr() for g in gpus), reverse=True))

    candidates = []

    # -------------------------------------------------------------------------
    # 1. Initialization
    # -------------------------------------------------------------------------

    # A. Linearized Bin Packing
    # Solves: Minimize K s.t. Sum(Load + K*Size) <= K*Cap for each bin
    def solve_linearized_bp(target_k, noise=0.0):
        bin_cap = target_k * GPU_MEM_SIZE
        items = []
        for m in models:
            base_w = (m.req_rate / m.slo) + target_k * m.model_size
            w = base_w
            if noise > 0:
                w *= random.uniform(1.0 - noise, 1.0 + noise)
            items.append((w, m, base_w))
        
        # Best Fit Decreasing on linearized weight
        items.sort(key=lambda x: x[0], reverse=True)

        bins = [GPUState(i) for i in range(gpu_num)]
        for w, m, base_w in items:
            best_idx = -1
            min_rem = float('inf')
            for i in range(gpu_num):
                if not bins[i].can_fit(m.model_size): continue
                
                lin_use = bins[i].load + target_k * bins[i].used_mem
                # Linearized capacity check
                if lin_use + base_w <= bin_cap:
                    rem = bin_cap - (lin_use + base_w)
                    if rem < min_rem:
                        min_rem = rem
                        best_idx = i
            
            if best_idx != -1:
                bins[best_idx].add(m)
            else:
                return None
        return bins

    # Binary search for minimum feasible K
    low, high = 0.0, 2000.0
    if solve_linearized_bp(high) is None: high = 1e9

    min_k = high
    for _ in range(16):
        mid = (low + high) / 2
        if solve_linearized_bp(mid):
            min_k = mid
            high = mid
        else:
            low = mid

    # Grid Search near optimal K to find best vector
    if min_k < 1e8:
        # Check specific multipliers to balance tightness vs load distribution
        for mult in [1.0, 1.02, 1.05, 1.1, 1.2]:
            res = solve_linearized_bp(min_k * mult)
            if res: candidates.append(res)
        
        # Add noisy variations around min_k
        for _ in range(8):
            res = solve_linearized_bp(min_k * random.uniform(0.95, 1.1), noise=0.05)
            if res: candidates.append(res)

    # B. Greedy Strategies
    strategies = [
        ('size', lambda m: m.model_size),
        ('load', lambda m: m.req_rate / m.slo),
        ('density', lambda m: (m.req_rate / m.slo) / m.model_size if m.model_size > 1e-7 else 0)
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

    # C. Randomized Greedy
    for _ in range(5):
        gpus = [GPUState(i) for i in range(gpu_num)]
        shuffled = list(models)
        random.shuffle(shuffled)
        valid = True
        for m in shuffled:
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

    # Fallback (Safety Net)
    if not candidates:
        gpus = [GPUState(i) for i in range(gpu_num)]
        for m in sorted(models, key=lambda x: x.model_size, reverse=True):
            placed = False
            for i in range(gpu_num):
                if gpus[i].can_fit(m.model_size):
                    gpus[i].add(m); placed=True; break
            if not placed: raise ValueError("Models do not fit.")
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
    max_iter = 250 

    while iter_cnt < max_iter:
        improved_step = False
        sorted_gpus = sorted(current_gpus, key=lambda g: g.kvpr(), reverse=True)
        sources = sorted_gpus[:4] # Focus on top bottlenecks
        destinations = sorted_gpus[::-1] # Focus on least loaded

        # --- Operator 1: Move (Best Improvement) ---
        best_move = None
        best_move_gain = current_vector
        for source in sources:
            for i, model in enumerate(source.models):
                for dest in current_gpus: # Try all dests
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
            best_vector = current_vector
            for k in range(gpu_num): best_gpus[k].copy_from(current_gpus[k])
            improved_step = True
            iter_cnt += 1; continue

        # --- Operator 2: Swap 1-1 (Best Improvement) ---
        best_swap = None
        best_swap_gain = current_vector
        for source in sources:
            for i, ma in enumerate(source.models):
                for dest in destinations:
                    if dest.id == source.id or dest.kvpr() >= source.kvpr(): continue
                    for j, mb in enumerate(dest.models):
                        if (mb.req_rate/mb.slo) >= (ma.req_rate/ma.slo): continue
                        
                        s_mem = source.used_mem - ma.model_size + mb.model_size
                        d_mem = dest.used_mem - mb.model_size + ma.model_size
                        if s_mem <= GPU_MEM_SIZE and d_mem <= GPU_MEM_SIZE:
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
            best_vector = current_vector
            for k in range(gpu_num): best_gpus[k].copy_from(current_gpus[k])
            improved_step = True
            iter_cnt += 1; continue

        # --- Operator 3: Swap 1-2 (First Improvement) ---
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
                                    best_vector = current_vector
                                    for k in range(gpu_num): best_gpus[k].copy_from(current_gpus[k])
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
        if improved_step: iter_cnt += 1; continue

        # --- Operator 4: Swap 2-1 (First Improvement) ---
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
                                    best_vector = current_vector
                                    for k in range(gpu_num): best_gpus[k].copy_from(current_gpus[k])
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
        if improved_step: iter_cnt += 1; continue

        # --- Perturbation: Dual Ruin ---
        iter_cnt += 1
        if iter_cnt > max_iter - 10: break

        worst_gpu = sorted_gpus[0]
        if not worst_gpu.models: break
        
        # Determine partner for ruin (Lightest GPU)
        best_gpu = sorted_gpus[-1]
        if best_gpu.id == worst_gpu.id:
            if len(sorted_gpus) > 1: best_gpu = sorted_gpus[-2]
            else: best_gpu = None

        displaced = []
        # Clear worst
        while worst_gpu.models: displaced.append(worst_gpu.remove(0))
        # Clear best
        if best_gpu:
            while best_gpu.models: displaced.append(best_gpu.remove(0))
        
        # Alternate sorting for reconstruction
        if iter_cnt % 2 == 0:
            # Sort by Size (Packing)
            displaced.sort(key=lambda m: m.model_size, reverse=True)
        else:
            # Sort by Load (Balancing)
            displaced.sort(key=lambda m: m.req_rate/m.slo, reverse=True)

        # Re-insert anywhere (Best Fit minimizes local KVPR gain)
        all_targets = current_gpus 
        for m in displaced:
            best_dest = None
            best_dest_val = float('inf')
            
            for dest in all_targets:
                if dest.can_fit(m.model_size):
                    rem = GPU_MEM_SIZE - (dest.used_mem + m.model_size)
                    if rem > 1e-7:
                        # Minimize new KVPR of that GPU
                        val = (dest.load + m.req_rate/m.slo) / rem
                        if val < best_dest_val:
                            best_dest_val = val
                            best_dest = dest
            
            if best_dest:
                best_dest.add(m)
            else:
                # Fallback to worst/best (should fit as they came from there)
                if worst_gpu.can_fit(m.model_size): worst_gpu.add(m)
                elif best_gpu and best_gpu.can_fit(m.model_size): best_gpu.add(m)
                else: worst_gpu.add(m) # Last resort, might overflow if logic error, but generally safe

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