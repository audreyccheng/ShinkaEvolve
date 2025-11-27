# EVOLVE-BLOCK-START
"""Model placement algorithm for minimizing maximum KV cache pressure across GPUs"""

GPU_MEM_SIZE = 80  # GB

def compute_model_placement(gpu_num, models):
    """
    Compute a model placement that minimizes the maximum KVPR across all GPUs.
    
    Uses an ensemble of greedy heuristics:
    1. Sort by Weight (req/slo), minimize current KVPR (Load Balancing)
    2. Sort by Weight, minimize resulting KVPR (Peak Shaving)
    3. Sort by Isolated KVPR (Weight / Available_Mem_If_Alone), minimize resulting KVPR
    4. Sort by Size, minimize resulting KVPR (Bin Packing)
    
    Returns the placement from the heuristic that minimizes the global max KVPR.
    """

    # Helper to evaluate the quality of a complete placement
    def evaluate_placement(placement):
        max_p = 0.0
        for gpu_id, assigned in placement.items():
            total_w = sum(m.req_rate / m.slo for m in assigned)
            total_s = sum(m.model_size for m in assigned)
            rem = GPU_MEM_SIZE - total_s
            if rem <= 1e-9:
                return float('inf')
            max_p = max(max_p, total_w / rem)
        return max_p

    # Helper to generate a placement given a sorted list of models and a strategy
    def generate_placement(sorted_models, strategy):
        placement = {i: [] for i in range(gpu_num)}
        # Track state: w = sum(req/slo), s = sum(size)
        gpu_state = [{'w': 0.0, 's': 0.0} for _ in range(gpu_num)]
        
        for model in sorted_models:
            w = model.req_rate / model.slo
            s = model.model_size
            
            best_idx = None
            best_val = float('inf')
            
            # Find best GPU for this model
            for i in range(gpu_num):
                curr_s = gpu_state[i]['s']
                if curr_s + s > GPU_MEM_SIZE:
                    continue
                
                curr_w = gpu_state[i]['w']
                rem = GPU_MEM_SIZE - curr_s
                
                if strategy == 'min_current':
                    # Heuristic: Place on GPU with lowest CURRENT KVPR (Load Balance)
                    # Note: We prioritize validity (fits). Among valid, pick min current load.
                    if rem > 1e-9:
                        val = curr_w / rem
                    else:
                        val = float('inf')
                else: # 'min_result'
                    # Heuristic: Place on GPU that leads to lowest NEW KVPR (Greedy Min-Max)
                    new_rem = rem - s
                    if new_rem > 1e-9:
                        val = (curr_w + w) / new_rem
                    else:
                        val = float('inf')
                
                if val < best_val:
                    best_val = val
                    best_idx = i
                elif val == best_val and best_idx is None:
                    best_idx = i
            
            if best_idx is None:
                return None  # This heuristic failed to place a model
            
            placement[best_idx].append(model)
            gpu_state[best_idx]['w'] += w
            gpu_state[best_idx]['s'] += s
            
        return placement

    # Define heuristics: (Sorting Key Lambda, Placement Strategy)
    heuristics = [
        # 1. Baseline: High pressure first, fill valleys (Program 2 equivalent)
        (lambda m: m.req_rate / m.slo, 'min_current'),
        # 2. Lookahead: High pressure first, minimize peaks
        (lambda m: m.req_rate / m.slo, 'min_result'),
        # 3. Difficulty: High "Isolated KVPR" first. Handles large & heavy models early.
        (lambda m: (m.req_rate / m.slo) / (GPU_MEM_SIZE - m.model_size + 1e-6), 'min_result'),
        # 4. Packing: Large size first, minimize peaks
        (lambda m: m.model_size, 'min_result')
    ]

    best_placement = None
    best_max_kvpr = float('inf')

    # Run all heuristics and pick the winner
    for key_func, strat in heuristics:
        try:
            # Sort models (descending)
            sorted_m = sorted(models, key=key_func, reverse=True)
            placement = generate_placement(sorted_m, strat)
            
            if placement is not None:
                score = evaluate_placement(placement)
                if score < best_max_kvpr:
                    best_max_kvpr = score
                    best_placement = placement
        except Exception:
            continue

    if best_placement is None:
         # Fallback: Just try to place based on size if everything else failed unexpectedly,
         # or raise error if it's a capacity issue.
         raise ValueError("Unable to place models on GPUs with available memory.")

    return best_placement

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

