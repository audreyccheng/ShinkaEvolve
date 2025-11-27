# EVOLVE-BLOCK-START
"""Model placement algorithm for minimizing maximum KV cache pressure across GPUs"""

GPU_MEM_SIZE = 80.0  # GB
import math
import random

def compute_model_placement(gpu_num, models):
    """
    Hybrid ILS Placement Algorithm with Enhanced Operators:
    1. Multi-Start Initialization:
       - Binary Search Linearization (Weighted Bin Packing).
       - Greedy strategies (Size, Load, Density).
       - Randomized Greedy (Shuffled Best Fit) to explore diverse starting points.

    2. Iterated Local Search (ILS):
       - Move: Single model relocation.
       - Swap (1-1): Pairwise swap.
       - Swap (2-1): Two models from bottleneck for one from another (New).
       - Swap (1-2): One model from bottleneck for two from another.
       - Perturbation: Guided kick moving a model to a low-load GPU.
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
    # 1. Initialization Strategies
    # -------------------------------------------------------------------------

    # A. Randomized Binary Search Linearization
    def solve_linearized_bp(target_k, noise=0.0):
        bin_cap = target_k * GPU_MEM_SIZE
        items = []
        for m in models:
            base_w = (m.req_rate / m.slo) + target_k * m.model_size
            w = base_w
            if noise > 0:
                w *= random.uniform(1.0 - noise, 1.0 + noise)
            items.append((w, m, base_w))
        items.sort(key=lambda x: x[0], reverse=True)

        bins = [GPUState(i) for i in range(gpu_num)]
        for w, m, base_w in items:
            best_idx = -1
            min_rem = float('inf')
            for i in range(gpu_num):
                if not bins[i].can_fit(m.model_size): continue
                lin_usage = bins[i].load + target_k * bins[i].used_mem
                if lin_usage + base_w <= bin_cap:
                    rem = bin_cap - (lin_usage + base_w)
                    if rem < min_rem:
                        min_rem = rem
                        best_idx = i
            if best_idx != -1:
                bins[best_idx].add(m)
            else:
                return None
        return bins

    low, high = 0.0, 1000.0
    if solve_linearized_bp(high) is None: high = 1e9

    bs_res = None
    best_k = high
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

    # Randomized refinements around best_k
    for _ in range(15):
        k_noisy = best_k * random.uniform(0.9, 1.1)
        res_noise = solve_linearized_bp(k_noisy, noise=0.05)
        if res_noise: candidates.append(res_noise)

    # B. Deterministic Greedy
    sort_strategies = [
        ('size', lambda m: m.model_size),
        ('load', lambda m: m.req_rate / m.slo),
        ('density', lambda m: (m.req_rate / m.slo) / m.model_size if m.model_size > 1e-7 else float('inf'))
    ]

    for name, key_func in sort_strategies:
        gpus = [GPUState(i) for i in range(gpu_num)]
        sorted_models = sorted(models, key=key_func, reverse=True)
        valid = True
        for m in sorted_models:
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

    # C. Randomized Greedy (Explore diverse starts)
    for _ in range(3):
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

    # Fallback
    if not candidates:
        gpus = [GPUState(i) for i in range(gpu_num)]
        for m in sorted(models, key=lambda x: x.model_size, reverse=True):
            placed = False
            for i in range(gpu_num):
                if gpus[i].can_fit(m.model_size):
                    gpus[i].add(m); placed = True; break
            if not placed: raise ValueError("Models do not fit in GPU memory.")
        candidates.append(gpus)

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
        iter_cnt += 1

        sorted_gpus = sorted(current_gpus, key=lambda g: g.kvpr(), reverse=True)
        sources = sorted_gpus[:4]

        # --- Move ---
        for source in sources:
            for i, model in enumerate(source.models):
                for dest in current_gpus:
                    if dest.id == source.id: continue
                    if dest.can_fit(model.model_size):
                        source.remove(i)
                        dest.add(model)
                        new_vec = get_vector(current_gpus)
                        if new_vec < current_vector:
                            current_vector = new_vec
                            improved_step = True
                            if current_vector < best_vector:
                                best_vector = current_vector
                                for k in range(gpu_num): best_gpus[k].copy_from(current_gpus[k])
                            break
                        else:
                            dest.remove(len(dest.models)-1)
                            source.restore_model(i, model)
                if improved_step: break
            if improved_step: break
        if improved_step: continue

        # --- Swap 1-1 ---
        for source in sources:
            for i, m_a in enumerate(source.models):
                for dest in current_gpus:
                    if dest.id == source.id: continue
                    if dest.kvpr() >= source.kvpr(): continue
                    for j, m_b in enumerate(dest.models):
                        s_mem = source.used_mem - m_a.model_size + m_b.model_size
                        d_mem = dest.used_mem - m_b.model_size + m_a.model_size
                        if s_mem <= GPU_MEM_SIZE and d_mem <= GPU_MEM_SIZE:
                            source.remove(i)
                            dest.remove(j)
                            source.add(m_b)
                            dest.add(m_a)
                            new_vec = get_vector(current_gpus)
                            if new_vec < current_vector:
                                current_vector = new_vec
                                improved_step = True
                                if current_vector < best_vector:
                                    best_vector = current_vector
                                    for k in range(gpu_num): best_gpus[k].copy_from(current_gpus[k])
                                break
                            else:
                                dest.remove(len(dest.models)-1)
                                source.remove(len(source.models)-1)
                                dest.restore_model(j, m_b)
                                source.restore_model(i, m_a)
                    if improved_step: break
                if improved_step: break
        if improved_step: continue

        # --- Swap 2-1 (New) ---
        for source in sources[:2]:
            if len(source.models) < 2: continue
            for i1 in range(len(source.models)):
                for i2 in range(i1+1, len(source.models)):
                    m_a1 = source.models[i1]
                    m_a2 = source.models[i2]
                    for dest in current_gpus:
                        if dest.id == source.id: continue
                        if dest.kvpr() >= source.kvpr(): continue
                        for j, m_b in enumerate(dest.models):
                            s_mem = source.used_mem - m_a1.model_size - m_a2.model_size + m_b.model_size
                            d_mem = dest.used_mem - m_b.model_size + m_a1.model_size + m_a2.model_size
                            if s_mem <= GPU_MEM_SIZE and d_mem <= GPU_MEM_SIZE:
                                source.remove(i2) # Remove higher index first
                                source.remove(i1)
                                dest.remove(j)
                                source.add(m_b)
                                dest.add(m_a1)
                                dest.add(m_a2)
                                new_vec = get_vector(current_gpus)
                                if new_vec < current_vector:
                                    current_vector = new_vec
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
        if improved_step: continue

        # --- Swap 1-2 ---
        for source in sources[:2]:
            for i, m_a in enumerate(source.models):
                for dest in current_gpus:
                    if dest.id == source.id: continue
                    if dest.kvpr() >= source.kvpr(): continue
                    if len(dest.models) < 2: continue
                    n_d = len(dest.models)
                    pair_found = False
                    for j1 in range(n_d):
                        for j2 in range(j1+1, n_d):
                            m_b1 = dest.models[j1]
                            m_b2 = dest.models[j2]
                            s_mem = source.used_mem - m_a.model_size + m_b1.model_size + m_b2.model_size
                            d_mem = dest.used_mem - m_b1.model_size - m_b2.model_size + m_a.model_size
                            if s_mem <= GPU_MEM_SIZE and d_mem <= GPU_MEM_SIZE:
                                source.remove(i)
                                dest.remove(j2)
                                dest.remove(j1)
                                source.add(m_b1)
                                source.add(m_b2)
                                dest.add(m_a)
                                new_vec = get_vector(current_gpus)
                                if new_vec < current_vector:
                                    current_vector = new_vec
                                    improved_step = True
                                    pair_found = True
                                    if current_vector < best_vector:
                                        best_vector = current_vector
                                        for k in range(gpu_num): best_gpus[k].copy_from(current_gpus[k])
                                    break
                                else:
                                    dest.remove(len(dest.models)-1)
                                    source.remove(len(source.models)-1)
                                    source.remove(len(source.models)-1)
                                    dest.restore_model(j1, m_b1)
                                    dest.restore_model(j2, m_b2)
                        if pair_found: break
                    if pair_found: break
                if improved_step: break
        if improved_step: continue

        # --- Perturbation ---
        if iter_cnt > max_iter - 20: break
        worst_gpu = sorted_gpus[0]
        if not worst_gpu.models: break
        m_idx = random.randint(0, len(worst_gpu.models)-1)
        model_to_move = worst_gpu.models[m_idx]
        feasible_dests = [g for g in current_gpus if g.id != worst_gpu.id and g.can_fit(model_to_move.model_size)]
        if feasible_dests:
            # Guided perturbation: Prefer destination with low pressure
            feasible_dests.sort(key=lambda g: g.kvpr())
            dest = random.choice(feasible_dests[:3])
            worst_gpu.remove(m_idx)
            dest.add(model_to_move)
            current_vector = get_vector(current_gpus)
        else:
            break

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