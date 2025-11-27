# EVOLVE-BLOCK-START
"""Model placement algorithm for minimizing maximum KV cache pressure across GPUs"""

GPU_MEM_SIZE = 80.0  # GB
import math
import random

def compute_model_placement(gpu_num, models):
    """
    Algorithm:
    1. Binary Search Linearization: Solves a parameter-weighted Bin Packing problem
       to find a layout satisfying Sum(load + K*size) <= K*Cap.
    2. Iterated Local Search (ILS):
       - Variable Neighborhood Descent with Move, Swap(1-1), Swap(1-2).
       - Perturbation: If local search stagnates, forcefully moves a model from
         the bottleneck GPU to a random feasible GPU to escape local optima.
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

    # -------------------------------------------------------------------------
    # 1. Binary Search Initialization
    # -------------------------------------------------------------------------
    def solve_linearized_bin_packing(target_k):
        # Constraint: Sum(load) / (Cap - Sum(size)) <= K
        # <=> Sum(load) <= K * Cap - K * Sum(size)
        # <=> Sum(load + K * size) <= K * Cap
        bin_cap = target_k * GPU_MEM_SIZE
        items = []
        for m in models:
            w = (m.req_rate / m.slo) + target_k * m.model_size
            items.append((w, m))

        # Best Fit Decreasing
        items.sort(key=lambda x: x[0], reverse=True)
        bins = [GPUState(i) for i in range(gpu_num)]

        for w, m in items:
            best_idx = -1
            min_rem_linear = float('inf')

            for i in range(gpu_num):
                # Hard memory constraint
                if not bins[i].can_fit(m.model_size):
                    continue

                # Soft linearized constraint
                # Check if adding fits in the "linearized capacity"
                current_linear_usage = bins[i].load + target_k * bins[i].used_mem
                added_linear_usage = (m.req_rate / m.slo) + target_k * m.model_size

                if current_linear_usage + added_linear_usage <= bin_cap:
                    rem = bin_cap - (current_linear_usage + added_linear_usage)
                    if rem < min_rem_linear:
                        min_rem_linear = rem
                        best_idx = i

            if best_idx != -1:
                bins[best_idx].add(m)
            else:
                return None
        return bins

    # Binary search
    low = 0.0
    high = 1000.0
    if solve_linearized_bin_packing(high) is None: high = 1e9 # Relaxation

    best_init_gpus = None
    for _ in range(20):
        mid = (low + high) / 2
        res = solve_linearized_bin_packing(mid)
        if res is not None:
            best_init_gpus = res
            high = mid
        else:
            low = mid

    if best_init_gpus is None:
        # Fallback: Simple greedy
        best_init_gpus = [GPUState(i) for i in range(gpu_num)]
        for m in sorted(models, key=lambda x: x.model_size, reverse=True):
            placed = False
            for i in range(gpu_num):
                if best_init_gpus[i].can_fit(m.model_size):
                    best_init_gpus[i].add(m)
                    placed = True
                    break
            if not placed: raise ValueError("OOM")

    # -------------------------------------------------------------------------
    # 2. Local Search with Perturbation
    # -------------------------------------------------------------------------
    current_gpus = best_init_gpus

    def get_vector(gs):
        return tuple(sorted((g.kvpr() for g in gs), reverse=True))

    current_vector = get_vector(current_gpus)

    # Save best globally
    best_gpus = [GPUState(i) for i in range(gpu_num)]
    for i in range(gpu_num): best_gpus[i].copy_from(current_gpus[i])
    best_vector = current_vector

    iter_cnt = 0
    max_iter = 150

    while iter_cnt < max_iter:
        improved_step = False

        # Identify bottlenecks
        sorted_gpus = sorted(current_gpus, key=lambda g: g.kvpr(), reverse=True)
        sources = sorted_gpus[:4] # Top bottlenecks

        # --- MOVE ---
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
                            break
                        else:
                            dest.remove(len(dest.models)-1)
                            source.restore_model(i, model)
                if improved_step: break
            if improved_step: break

        if improved_step:
            if current_vector < best_vector:
                best_vector = current_vector
                for k in range(gpu_num): best_gpus[k].copy_from(current_gpus[k])
            iter_cnt += 1
            continue

        # --- SWAP 1-1 ---
        for source in sources:
            for i, m_a in enumerate(source.models):
                for dest in current_gpus:
                    if dest.id == source.id: continue
                    if dest.kvpr() >= source.kvpr(): continue

                    for j, m_b in enumerate(dest.models):
                        # Capacity check
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
                                break
                            else:
                                dest.remove(len(dest.models)-1)
                                source.remove(len(source.models)-1)
                                dest.restore_model(j, m_b)
                                source.restore_model(i, m_a)
                    if improved_step: break
                if improved_step: break
            if improved_step: break

        if improved_step:
            if current_vector < best_vector:
                best_vector = current_vector
                for k in range(gpu_num): best_gpus[k].copy_from(current_gpus[k])
            iter_cnt += 1
            continue

        # --- SWAP 1-2 --- (One from Source, Two from Dest)
        # Often Source has big model, Dest has fragmentation
        for source in sources[:2]:
            for i, m_a in enumerate(source.models):
                for dest in current_gpus:
                    if dest.id == source.id: continue
                    if dest.kvpr() >= source.kvpr(): continue
                    if len(dest.models) < 2: continue

                    n_d = len(dest.models)
                    pair_found = False
                    # Check pairs in dest
                    for j1 in range(n_d):
                        for j2 in range(j1 + 1, n_d):
                            m_b1 = dest.models[j1]
                            m_b2 = dest.models[j2]

                            s_mem = source.used_mem - m_a.model_size + m_b1.model_size + m_b2.model_size
                            d_mem = dest.used_mem - m_b1.model_size - m_b2.model_size + m_a.model_size

                            if s_mem <= GPU_MEM_SIZE and d_mem <= GPU_MEM_SIZE:
                                source.remove(i)
                                # Remove larger index first
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
                                    break
                                else:
                                    # Revert
                                    dest.remove(len(dest.models)-1) # m_a
                                    source.remove(len(source.models)-1) # m_b2
                                    source.remove(len(source.models)-1) # m_b1
                                    dest.restore_model(j1, m_b1)
                                    dest.restore_model(j2, m_b2)
                        if pair_found: break
                    if pair_found: break
                if improved_step: break
            if improved_step: break

        if improved_step:
            if current_vector < best_vector:
                best_vector = current_vector
                for k in range(gpu_num): best_gpus[k].copy_from(current_gpus[k])
            iter_cnt += 1
            continue

        # --- PERTURBATION ---
        # If we reach here, we are at a local optimum.
        # Apply a kick: Force move a model from the absolute worst GPU to a random feasible spot.
        # Then continue.

        iter_cnt += 1
        # Stop if near end to avoid returning a perturbed state that hasn't been fixed
        if iter_cnt > max_iter - 10: break

        worst_gpu = sorted_gpus[0]
        if not worst_gpu.models: break

        # Pick a random model from worst gpu
        m_idx = random.randint(0, len(worst_gpu.models)-1)
        model_to_move = worst_gpu.models[m_idx]

        # Find feasible destinations (exclude self)
        feasible_dests = [g for g in current_gpus if g.id != worst_gpu.id and g.can_fit(model_to_move.model_size)]

        if feasible_dests:
            dest = random.choice(feasible_dests)
            worst_gpu.remove(m_idx)
            dest.add(model_to_move)
            current_vector = get_vector(current_gpus)
            # We don't update best_vector here because the kick likely makes it worse.
        else:
            # If cannot move, break to avoid infinite loops
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