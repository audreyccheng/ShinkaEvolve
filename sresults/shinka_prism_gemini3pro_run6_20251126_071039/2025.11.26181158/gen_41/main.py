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
    # 2. Iterated Local Search (Steepest Descent with Tabu)
    # -------------------------------------------------------------------------
    iter_cnt = 0
    max_iter = 150

    # Tabu list: stores ((model, source_gpu_id), expiry_iter)
    tabu_list = {}
    tabu_tenure = 5

    def is_tabu(model, gpu_id, iter_curr):
        return (model, gpu_id) in tabu_list and tabu_list[(model, gpu_id)] > iter_curr

    def add_tabu(model, gpu_id, iter_curr):
        tabu_list[(model, gpu_id)] = iter_curr + tabu_tenure

    while iter_cnt < max_iter:
        improved_step = False

        sorted_gpus = sorted(current_gpus, key=lambda g: g.kvpr(), reverse=True)
        sources = sorted_gpus[:4]
        destinations = sorted_gpus[::-1]

        # Global Steepest Descent: Find the BEST move across all operators
        best_op_gain = current_vector
        best_op_action = None # (type, data...)

        # 1. Operator: Move
        for src in sources:
            for i, m in enumerate(src.models):
                if is_tabu(m, src.id, iter_cnt): continue
                for dst in destinations:
                    if dst.id == src.id: continue
                    if dst.can_fit(m.model_size):
                        src.remove(i)
                        dst.add(m)

                        new_vec = get_vector(current_gpus)
                        if new_vec < best_op_gain:
                            best_op_gain = new_vec
                            best_op_action = ('move', src, i, dst, m)

                        dst.remove(len(dst.models)-1)
                        src.restore_model(i, m)

        # 2. Operator: Swap 1-1
        for src in sources:
            for i, m_a in enumerate(src.models):
                if is_tabu(m_a, src.id, iter_cnt): continue
                for dst in destinations:
                    if dst.id == src.id: continue
                    if dst.kvpr() >= src.kvpr(): continue

                    for j, m_b in enumerate(dst.models):
                        if is_tabu(m_b, dst.id, iter_cnt): continue

                        s_mem = src.used_mem - m_a.model_size + m_b.model_size
                        d_mem = dst.used_mem - m_b.model_size + m_a.model_size
                        if s_mem <= GPU_MEM_SIZE and d_mem <= GPU_MEM_SIZE:
                            src.remove(i)
                            dst.remove(j)
                            src.add(m_b)
                            dst.add(m_a)

                            new_vec = get_vector(current_gpus)
                            if new_vec < best_op_gain:
                                best_op_gain = new_vec
                                best_op_action = ('swap11', src, i, m_a, dst, j, m_b)

                            dst.remove(len(dst.models)-1)
                            src.remove(len(src.models)-1)
                            dst.restore_model(j, m_b)
                            src.restore_model(i, m_a)

        # 3. Operator: Swap 1-2 (Source gives 1, Dest gives 2)
        for src in sources:
            for i, m_a in enumerate(src.models):
                if is_tabu(m_a, src.id, iter_cnt): continue
                for dst in destinations:
                    if dst.id == src.id: continue
                    if dst.kvpr() >= src.kvpr(): continue
                    if len(dst.models) < 2: continue

                    n_d = len(dst.models)
                    for j1 in range(n_d):
                        for j2 in range(j1+1, n_d):
                            m_b1 = dst.models[j1]
                            m_b2 = dst.models[j2]
                            if is_tabu(m_b1, dst.id, iter_cnt) or is_tabu(m_b2, dst.id, iter_cnt): continue

                            s_mem = src.used_mem - m_a.model_size + m_b1.model_size + m_b2.model_size
                            d_mem = dst.used_mem - m_b1.model_size - m_b2.model_size + m_a.model_size

                            if s_mem <= GPU_MEM_SIZE and d_mem <= GPU_MEM_SIZE:
                                src.remove(i)
                                dst.remove(j2) # Higher index first
                                dst.remove(j1)
                                src.add(m_b1)
                                src.add(m_b2)
                                dst.add(m_a)

                                new_vec = get_vector(current_gpus)
                                if new_vec < best_op_gain:
                                    best_op_gain = new_vec
                                    best_op_action = ('swap12', src, i, m_a, dst, j1, m_b1, j2, m_b2)

                                dst.remove(len(dst.models)-1)
                                src.remove(len(src.models)-1)
                                src.remove(len(src.models)-1)
                                dst.restore_model(j1, m_b1)
                                dst.restore_model(j2, m_b2)
                                src.restore_model(i, m_a)

        # 4. Operator: Swap 2-1 (Source gives 2, Dest gives 1)
        for src in sources:
            if len(src.models) < 2: continue
            n_s = len(src.models)
            for i1 in range(n_s):
                for i2 in range(i1+1, n_s):
                    m_a1 = src.models[i1]
                    m_a2 = src.models[i2]
                    if is_tabu(m_a1, src.id, iter_cnt) or is_tabu(m_a2, src.id, iter_cnt): continue

                    for dst in destinations:
                        if dst.id == src.id: continue
                        if dst.kvpr() >= src.kvpr(): continue

                        for j, m_b in enumerate(dst.models):
                             if is_tabu(m_b, dst.id, iter_cnt): continue

                             s_mem = src.used_mem - m_a1.model_size - m_a2.model_size + m_b.model_size
                             d_mem = dst.used_mem - m_b.model_size + m_a1.model_size + m_a2.model_size

                             if s_mem <= GPU_MEM_SIZE and d_mem <= GPU_MEM_SIZE:
                                 src.remove(i2) # Remove higher index first
                                 src.remove(i1)
                                 dst.remove(j)
                                 src.add(m_b)
                                 dst.add(m_a1)
                                 dst.add(m_a2)

                                 new_vec = get_vector(current_gpus)
                                 if new_vec < best_op_gain:
                                    best_op_gain = new_vec
                                    best_op_action = ('swap21', src, i1, m_a1, i2, m_a2, dst, j, m_b)

                                 dst.remove(len(dst.models)-1)
                                 dst.remove(len(dst.models)-1)
                                 src.remove(len(src.models)-1)
                                 dst.restore_model(j, m_b)
                                 src.restore_model(i1, m_a1)
                                 src.restore_model(i2, m_a2)

        # Apply Best Action
        if best_op_action:
            type_op = best_op_action[0]
            improved_step = True
            current_vector = best_op_gain

            if type_op == 'move':
                _, src, i, dst, m = best_op_action
                src.remove(i)
                dst.add(m)
                add_tabu(m, dst.id, iter_cnt)

            elif type_op == 'swap11':
                _, src, i, m_a, dst, j, m_b = best_op_action
                src.remove(i)
                dst.remove(j)
                src.add(m_b)
                dst.add(m_a)
                add_tabu(m_a, dst.id, iter_cnt)
                add_tabu(m_b, src.id, iter_cnt)

            elif type_op == 'swap12':
                _, src, i, m_a, dst, j1, m_b1, j2, m_b2 = best_op_action
                src.remove(i)
                dst.remove(j2)
                dst.remove(j1)
                src.add(m_b1)
                src.add(m_b2)
                dst.add(m_a)
                add_tabu(m_a, dst.id, iter_cnt)
                add_tabu(m_b1, src.id, iter_cnt)
                add_tabu(m_b2, src.id, iter_cnt)

            elif type_op == 'swap21':
                _, src, i1, m_a1, i2, m_a2, dst, j, m_b = best_op_action
                src.remove(i2)
                src.remove(i1)
                dst.remove(j)
                src.add(m_b)
                dst.add(m_a1)
                dst.add(m_a2)
                add_tabu(m_b, src.id, iter_cnt)
                add_tabu(m_a1, dst.id, iter_cnt)
                add_tabu(m_a2, dst.id, iter_cnt)

            if current_vector < best_vector:
                best_vector = current_vector
                for k in range(gpu_num): best_gpus[k].copy_from(current_gpus[k])

            iter_cnt += 1
            continue

        # --- PERTURBATION: Guided Kick ---
        iter_cnt += 1
        if iter_cnt > max_iter - 10: break

        worst_gpu = sorted_gpus[0]
        if not worst_gpu.models: break

        m_idx = random.randint(0, len(worst_gpu.models)-1)
        model_to_move = worst_gpu.models[m_idx]

        candidates = [g for g in current_gpus if g.id != worst_gpu.id and g.can_fit(model_to_move.model_size)]
        if candidates:
            # Pick from top 3 least loaded
            candidates.sort(key=lambda g: g.kvpr())
            target = random.choice(candidates[:3]) if len(candidates) >= 3 else candidates[0]

            worst_gpu.remove(m_idx)
            target.add(model_to_move)
            current_vector = get_vector(current_gpus)
            add_tabu(model_to_move, target.id, iter_cnt)

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