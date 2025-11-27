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

    # Import random for stochastic operations
    import random

    best_overall_gpus = None
    best_overall_vector = None

    def solve_greedy_and_local_search(sorted_models_input):
        # 1. Greedy Construction
        current_gpus = [GPUState(i) for i in range(gpu_num)]
        valid_strategy = True

        for model in sorted_models_input:
            best_idx = -1
            best_val = float('inf')

            # Place on GPU that minimizes its resulting KVPR
            for i in range(gpu_num):
                g = current_gpus[i]
                if g.can_fit(model):
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
            current_gpus[best_idx].add(model)

        if not valid_strategy:
            return None, None

        def get_vector(gs):
            return sorted([g.kvpr() for g in gs], reverse=True)

        current_vector = get_vector(current_gpus)

        # 2. Local Search Refinement
        max_iter = 100
        for _ in range(max_iter):
            improved = False

            # Identify bottleneck GPUs (sort by KVPR desc)
            sorted_gpus_by_kvpr = sorted(current_gpus, key=lambda g: g.kvpr(), reverse=True)

            # Optimization: Focus on moving load away from the most loaded GPUs
            # We check the top few GPUs, not just the first one, as lexicographical optimization benefits from fixing 2nd max too.
            sources_to_check = sorted_gpus_by_kvpr[:3] # Check top 3 bottleneck GPUs

            for source in sources_to_check:
                # Try Moving a model from source to any other GPU
                for i, model in enumerate(source.models):
                    for dest in current_gpus:
                        if dest.id == source.id: continue
                        if dest.can_fit(model):
                            source.remove(i)
                            dest.add(model)

                            new_vec = get_vector(current_gpus)
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

            # Try Swapping models between bottleneck GPUs and others
            for source in sources_to_check:
                for i, m_a in enumerate(source.models):
                    for dest in current_gpus:
                        if dest.id == source.id: continue
                        for j, m_b in enumerate(dest.models):
                            # Check capacity
                            s_rem = GPU_MEM_SIZE - (source.used_mem - m_a.model_size + m_b.model_size)
                            d_rem = GPU_MEM_SIZE - (dest.used_mem - m_b.model_size + m_a.model_size)

                            if s_rem >= 0 and d_rem >= 0:
                                source.remove(i)
                                dest.remove(j)
                                source.add(m_b)
                                dest.add(m_a)

                                new_vec = get_vector(current_gpus)
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

            if not improved:
                break

        return current_gpus, current_vector

    # Define various sorting strategies for initialization
    strategies = [
        lambda m: m.model_size, # Size Desc
        lambda m: m.req_rate / m.slo, # Load Desc
        lambda m: (m.req_rate / m.slo) / m.model_size if m.model_size > 0 else float('inf') # Density Desc
    ]

    # Run deterministic strategies
    for key in strategies:
        sorted_models = sorted(models, key=key, reverse=True)
        res_gpus, res_vec = solve_greedy_and_local_search(sorted_models)
        if res_gpus and (best_overall_vector is None or res_vec < best_overall_vector):
            best_overall_vector = res_vec
            best_overall_gpus = res_gpus

    # Run random shuffle strategies
    for _ in range(15):
        shuffled_models = list(models)
        random.shuffle(shuffled_models)
        res_gpus, res_vec = solve_greedy_and_local_search(shuffled_models)
        if res_gpus and (best_overall_vector is None or res_vec < best_overall_vector):
            best_overall_vector = res_vec
            best_overall_gpus = res_gpus

    if best_overall_gpus is None:
        raise ValueError("Unable to place models on GPUs with available memory.")

    return {g.id: g.models for g in best_overall_gpus}

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
