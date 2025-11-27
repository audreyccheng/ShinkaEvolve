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

        def update_cache(self):
            rem = GPU_MEM_SIZE - self.used_mem
            if rem <= 1e-7:
                self._cached_kvpr = float('inf')
            else:
                self._cached_kvpr = self.load / rem

        def can_fit(self, model):
            return self.used_mem + model.model_size <= GPU_MEM_SIZE

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

    # Strategies for sorting models: Size Desc, Load Desc, Density Desc
    sort_keys = [
        lambda m: m.model_size,
        lambda m: m.req_rate / m.slo,
        lambda m: (m.req_rate / m.slo) / m.model_size if m.model_size > 0 else float('inf')
    ]

    best_gpus = None
    best_vector = None

    # 1. Multi-start Greedy Construction
    for key in sort_keys:
        # Sort models based on current strategy
        sorted_models = sorted(models, key=key, reverse=True)
        gpus = [GPUState(i) for i in range(gpu_num)]
        valid_strategy = True

        for model in sorted_models:
            best_idx = -1
            best_val = float('inf')

            # Place on GPU that minimizes its resulting KVPR
            # This is a 'Best Fit' heuristic for the objective function
            for i in range(gpu_num):
                g = gpus[i]
                if g.can_fit(model):
                    # Calculate hypothetical KVPR if we add this model
                    new_load = g.load + model.req_rate / model.slo
                    new_mem = g.used_mem + model.model_size
                    rem = GPU_MEM_SIZE - new_mem
                    val = float('inf')
                    if rem > 1e-7:
                        val = new_load / rem

                    if val < best_val:
                        best_val = val
                        best_idx = i

            if best_idx == -1:
                valid_strategy = False
                break
            gpus[best_idx].add(model)

        if valid_strategy:
            # Evaluate using lexicographical comparison of KVPR vectors
            # This prioritizes minimizing the max, then the second max, etc.
            current_vector = sorted([g.kvpr() for g in gpus], reverse=True)
            if best_vector is None or current_vector < best_vector:
                best_vector = current_vector
                best_gpus = gpus

    if best_gpus is None:
        raise ValueError("Unable to place models on GPUs with available memory.")

    # 2. Local Search Refinement
    gpus = best_gpus
    current_vector = best_vector

    def get_vector(current_gpus):
        return sorted([g.kvpr() for g in current_gpus], reverse=True)

    max_iter = 100
    for _ in range(max_iter):
        improved = False

        # Identify bottleneck GPUs (highest KVPR)
        sorted_gpus = sorted(gpus, key=lambda g: g.kvpr(), reverse=True)
        source = sorted_gpus[0]

        # Try Moving a model from source to any other GPU
        for i, model in enumerate(source.models):
            for dest in gpus:
                if dest.id == source.id: continue
                if dest.can_fit(model):
                    # Apply Move
                    source.remove(i)
                    dest.add(model)

                    new_vec = get_vector(gpus)
                    if new_vec < current_vector:
                        current_vector = new_vec
                        improved = True
                        break
                    else:
                        # Revert Move
                        dest.remove(len(dest.models)-1)
                        source.restore_model(i, model)
            if improved: break

        if improved: continue

        # Try Swapping a model from source with a model from another GPU
        for i, m_a in enumerate(source.models):
            for dest in gpus:
                if dest.id == source.id: continue
                for j, m_b in enumerate(dest.models):
                    # Check capacity for swap
                    s_rem = GPU_MEM_SIZE - (source.used_mem - m_a.model_size + m_b.model_size)
                    d_rem = GPU_MEM_SIZE - (dest.used_mem - m_b.model_size + m_a.model_size)

                    if s_rem >= 0 and d_rem >= 0:
                        # Apply Swap
                        source.remove(i)
                        dest.remove(j)
                        source.add(m_b)
                        dest.add(m_a)

                        new_vec = get_vector(gpus)
                        if new_vec < current_vector:
                            current_vector = new_vec
                            improved = True
                            break
                        else:
                            # Revert Swap
                            dest.remove(len(dest.models)-1)
                            source.remove(len(source.models)-1)
                            dest.restore_model(j, m_b)
                            source.restore_model(i, m_a)
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
