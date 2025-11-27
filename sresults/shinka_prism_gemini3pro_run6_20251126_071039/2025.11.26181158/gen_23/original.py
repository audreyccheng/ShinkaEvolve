# EVOLVE-BLOCK-START
"""Model placement algorithm for minimizing maximum KV cache pressure across GPUs"""

GPU_MEM_SIZE = 80  # GB

def compute_model_placement(gpu_num, models):
    """
    Compute a model placement that minimizes the maximum KVPR across all GPUs.

    Args:
        gpu_num: Number of GPUs
        models: List of models to place

    Returns:
        A placement of models to GPUs
    """

    # Helper class to manage GPU state and calculations
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

    # 1. Binary Search Initialization
    # Check if a target KVPR 'K' is feasible by treating it as a Bin Packing problem

    def check_feasibility(target_k):
        items = []
        bin_cap = target_k * GPU_MEM_SIZE

        # Create weighted items
        for m in models:
            # weight = load + K * size
            w = (m.req_rate / m.slo) + target_k * m.model_size
            items.append((w, m))

        # Sort by weight descending (Best Fit Decreasing heuristic)
        items.sort(key=lambda x: x[0], reverse=True)

        bins_weight = [0.0] * gpu_num
        bins_mem = [0.0] * gpu_num
        bins_models = [[] for _ in range(gpu_num)]

        for w, m in items:
            best_bin_idx = -1
            min_rem_cap = float('inf')

            for i in range(gpu_num):
                # Hard Constraint: Physical Memory
                if bins_mem[i] + m.model_size > GPU_MEM_SIZE:
                    continue

                # Soft Constraint: Linearized KVPR Capacity
                if bins_weight[i] + w <= bin_cap:
                    rem = bin_cap - (bins_weight[i] + w)
                    if rem < min_rem_cap:
                        min_rem_cap = rem
                        best_bin_idx = i

            if best_bin_idx != -1:
                bins_weight[best_bin_idx] += w
                bins_mem[best_bin_idx] += m.model_size
                bins_models[best_bin_idx].append(m)
            else:
                return None

        return bins_models

    # Binary search for optimal K
    low = 0.0
    high = 1000.0 # Heuristic upper bound

    # Verify upper bound
    if check_feasibility(high) is None:
        high = 1e9 # Try very loose bound effectively checking just memory

    best_init_placement = None

    for _ in range(30):
        mid = (low + high) / 2
        res = check_feasibility(mid)
        if res is not None:
            best_init_placement = res
            high = mid
        else:
            low = mid

    # Fallback if binary search fails (e.g. tight memory)
    if best_init_placement is None:
        best_init_placement = [[] for _ in range(gpu_num)]
        # Simple greedy fill
        s_models = sorted(models, key=lambda m: m.model_size, reverse=True)
        g_mem = [0.0] * gpu_num
        for m in s_models:
            placed = False
            for i in range(gpu_num):
                if g_mem[i] + m.model_size <= GPU_MEM_SIZE:
                    best_init_placement[i].append(m)
                    g_mem[i] += m.model_size
                    placed = True
                    break
            if not placed:
                raise ValueError("Models do not fit in GPU memory.")

    # 2. Local Search Refinement
    gpus = [GPUState(i) for i in range(gpu_num)]
    for i, m_list in enumerate(best_init_placement):
        for m in m_list:
            gpus[i].add(m)

    def get_vector_fast(current_gpus):
        # Returns tuple for lexicographical comparison
        return tuple(sorted((g.kvpr() for g in current_gpus), reverse=True))

    current_vector = get_vector_fast(gpus)

    loop_count = 0
    max_loops = 150

    while loop_count < max_loops:
        improved = False
        loop_count += 1

        # Sort GPUs by pressure to focus on bottlenecks
        sorted_gpus = sorted(gpus, key=lambda g: g.kvpr(), reverse=True)

        # Focus on the top bottleneck GPUs
        sources = sorted_gpus[:min(len(sorted_gpus), 4)]

        # --- Operator 1: Move ---
        for source in sources:
            for i, model in enumerate(source.models):
                for dest in gpus:
                    if dest.id == source.id: continue
                    if dest.can_fit(model.model_size):
                        source.remove(i)
                        dest.add(model)

                        new_vec = get_vector_fast(gpus)
                        if new_vec < current_vector:
                            current_vector = new_vec
                            improved = True
                            break
                        else:
                            dest.remove(len(dest.models)-1)
                            source.restore_model(i, model)
                if improved: break
            if improved: break

        if improved: continue

        # --- Operator 2: Swap (1-to-1) ---
        for source in sources:
            for i, m_a in enumerate(source.models):
                for dest in gpus:
                    if dest.id == source.id: continue
                    if dest.kvpr() >= source.kvpr(): continue

                    for j, m_b in enumerate(dest.models):
                        # Capacity Check
                        if source.used_mem - m_a.model_size + m_b.model_size <= GPU_MEM_SIZE and \
                           dest.used_mem - m_b.model_size + m_a.model_size <= GPU_MEM_SIZE:

                            source.remove(i)
                            dest.remove(j)
                            source.add(m_b)
                            dest.add(m_a)

                            new_vec = get_vector_fast(gpus)
                            if new_vec < current_vector:
                                current_vector = new_vec
                                improved = True
                                break
                            else:
                                # Revert
                                dest.remove(len(dest.models)-1)
                                source.remove(len(source.models)-1)
                                dest.restore_model(j, m_b)
                                source.restore_model(i, m_a)
                    if improved: break
                if improved: break
            if improved: break

        if improved: continue

        # --- Operator 3: Swap (1-to-2) ---
        # Swap 1 model from Source with 2 models from Dest
        # Helps when Source is stuck with big models and Dest has fragmentation
        for source in sources[:2]: # Limit to top bottlenecks for speed
            for i, m_a in enumerate(source.models):
                for dest in gpus:
                    if dest.id == source.id: continue
                    if dest.kvpr() >= source.kvpr(): continue
                    if len(dest.models) < 2: continue

                    n_dest = len(dest.models)
                    pair_found = False

                    # Try pairs in dest
                    for j1 in range(n_dest):
                        for j2 in range(j1 + 1, n_dest):
                            m_b1 = dest.models[j1]
                            m_b2 = dest.models[j2]

                            # Capacity Check
                            # Source: -m_a + m_b1 + m_b2
                            # Dest: -m_b1 - m_b2 + m_a
                            if source.used_mem - m_a.model_size + m_b1.model_size + m_b2.model_size <= GPU_MEM_SIZE and \
                               dest.used_mem - m_b1.model_size - m_b2.model_size + m_a.model_size <= GPU_MEM_SIZE:

                                source.remove(i)
                                # Remove larger index first to preserve smaller index
                                dest.remove(j2)
                                dest.remove(j1)

                                source.add(m_b1)
                                source.add(m_b2)
                                dest.add(m_a)

                                new_vec = get_vector_fast(gpus)
                                if new_vec < current_vector:
                                    current_vector = new_vec
                                    improved = True
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
                if improved: break
            if improved: break

        if not improved:
            break

    return {g.id: g.models for g in gpus}

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