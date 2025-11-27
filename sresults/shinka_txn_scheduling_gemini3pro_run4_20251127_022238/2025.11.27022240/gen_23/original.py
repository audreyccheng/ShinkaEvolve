# EVOLVE-BLOCK-START
"""Transaction scheduling algorithm for optimizing makespan across multiple workloads"""

import time
import random
import sys
import os
import math

# Add the openevolve_examples directory to the path to import txn_simulator and workloads
# Find the repository root by looking for the openevolve_examples directory
def find_repo_root(start_path):
    """Find the repository root by looking for openevolve_examples directory."""
    current = os.path.abspath(start_path)
    # Search up the directory tree
    while current != os.path.dirname(current):  # Stop at filesystem root
        candidate = os.path.join(current, 'openevolve_examples', 'txn_scheduling')
        if os.path.exists(candidate):
            return current
        current = os.path.dirname(current)

    # If not found by searching up, try common locations relative to known paths
    # This handles when the program is copied to a results directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    possible_roots = [
        script_dir,  # Current directory
        os.path.dirname(script_dir),  # Parent
        os.path.dirname(os.path.dirname(script_dir)),  # Grandparent
        '/home/ubuntu/ShinkaEvolve',  # Absolute path fallback for Ubuntu
        '/Users/audreycc/Documents/Work/LLMTxn/ADRS-Exps/ShinkaEvolve',  # Absolute path fallback for macOS
    ]
    for root in possible_roots:
        candidate = os.path.join(root, 'openevolve_examples', 'txn_scheduling')
        if os.path.exists(candidate):
            return root

    raise RuntimeError(f"Could not find openevolve_examples directory. Searched from: {start_path}")

repo_root = find_repo_root(os.path.dirname(__file__))
sys.path.insert(0, os.path.join(repo_root, 'openevolve_examples', 'txn_scheduling'))

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3


def get_best_schedule(workload, num_seqs):
    """
    Get optimal schedule using Multi-Start Greedy Construction followed by Simulated Annealing.

    Algorithm:
    1.  **Length Analysis**: Pre-compute transaction operation counts for tie-breaking.
    2.  **Population Generation**: Generate `num_seqs` candidates.
        - Start randomly.
        - Greedy Step: Evaluate a random sample of `SAMPLE_SIZE` candidates.
        - Selection: Minimize `(makespan, -length)` (Best Fit Descending).
        - **Immediate Polish**: Apply full Adjacent Swap Descent to each greedy candidate.
          This ensures we are comparing local minima, not just rough constructions.
    3.  **Champion Selection**: Pick the best candidate from phase 2.
    4.  **Simulated Annealing (SA)**:
        - Refine the champion using a probabilistic acceptance criterion to escape local optima.
        - Moves:
          - Swap Adjacent (Small step).
          - Shift/Insert (Structural step, good for dependency resolving).
        - Cooling: Geometric decay.

    Args:
        workload: Workload object
        num_seqs: Number of independent random starts

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    import re

    # --- Pre-computation: Transaction Lengths ---
    txn_lens = {}
    try:
        for i in range(workload.num_txns):
            raw_txn = workload.txns[i]
            if isinstance(raw_txn, (list, tuple)):
                raw_txn = raw_txn[0]
            txn_str = str(raw_txn)
            ops = len(re.findall(r'[rw]-\d+', txn_str))
            txn_lens[i] = ops
    except Exception:
        for i in range(workload.num_txns):
            txn_lens[i] = 1

    SAMPLE_SIZE = 20
    candidates = []

    # --- Phase 1: Diverse Exploration (Generate & Optimize) ---
    for _ in range(num_seqs):
        # A. Greedy Construction
        remaining = list(range(workload.num_txns))
        start_txn = random.choice(remaining)
        current_seq = [start_txn]
        remaining.remove(start_txn)

        # Optimization: Track cost incrementally if possible, but simulation is complex.
        # We rely on simulation for accurate makespan.

        while remaining:
            # Sample candidates
            sample_size = min(len(remaining), SAMPLE_SIZE)
            # Pure random sampling (found to be more robust than "smart" sampling in previous gens)
            sample_cands = random.sample(remaining, sample_size)

            best_cand = -1
            # Minimize cost, Maximize length
            best_score = (float('inf'), 0)

            for t in sample_cands:
                test_seq = current_seq + [t]
                cost = workload.get_opt_seq_cost(test_seq)
                score = (cost, -txn_lens[t])

                if score < best_score:
                    best_score = score
                    best_cand = t

            current_seq.append(best_cand)
            remaining.remove(best_cand)

        # B. Fast Polish (Hill Climbing)
        # Bring candidate to a local optimum
        current_cost = workload.get_opt_seq_cost(current_seq)
        improved = True
        while improved:
            improved = False
            for i in range(len(current_seq) - 1):
                # Swap
                current_seq[i], current_seq[i+1] = current_seq[i+1], current_seq[i]
                new_cost = workload.get_opt_seq_cost(current_seq)

                if new_cost < current_cost:
                    current_cost = new_cost
                    improved = True
                else:
                    # Revert
                    current_seq[i], current_seq[i+1] = current_seq[i+1], current_seq[i]

        candidates.append((current_cost, current_seq))

    # --- Phase 2: Focused Exploitation (Simulated Annealing) ---
    # Pick the best seed
    candidates.sort(key=lambda x: x[0])
    best_cost, best_seq = candidates[0]

    # SA Parameters
    current_seq = list(best_seq)
    current_cost = best_cost

    T = 4.0               # Initial temperature (heuristic based on typical cost diffs)
    ALPHA = 0.95          # Cooling rate
    MIN_T = 0.1           # Stop temperature
    MAX_ITER = 600        # Safety break

    iter_count = 0
    while T > MIN_T and iter_count < MAX_ITER:
        iter_count += 1

        # 1. Generate Neighbor
        # 50% chance of simple Swap, 50% chance of Shift (Insertion)
        move_type = random.random()
        new_seq = list(current_seq)

        if move_type < 0.5:
            # Swap Adjacent
            if len(new_seq) < 2: continue
            idx = random.randint(0, len(new_seq) - 2)
            new_seq[idx], new_seq[idx+1] = new_seq[idx+1], new_seq[idx]
        else:
            # Shift (Insert)
            if len(new_seq) < 2: continue
            idx_from = random.randint(0, len(new_seq) - 1)
            idx_to = random.randint(0, len(new_seq) - 1)
            if idx_from == idx_to: continue
            txn = new_seq.pop(idx_from)
            new_seq.insert(idx_to, txn)

        # 2. Evaluate
        new_cost = workload.get_opt_seq_cost(new_seq)
        delta = new_cost - current_cost

        # 3. Acceptance Criterion (Metropolis)
        # If better (delta < 0), always accept.
        # If worse, accept with prob exp(-delta/T)
        if delta < 0 or random.random() < math.exp(-delta / T):
            current_seq = new_seq
            current_cost = new_cost

            # Update global best if found
            if current_cost < best_cost:
                best_cost = current_cost
                best_seq = list(current_seq)

        # Cool down
        T *= ALPHA

    return best_cost, best_seq


def get_random_costs():
    """
    Evaluate scheduling algorithm on three different workloads.

    Returns:
        Tuple of (total_makespan, list_of_schedules, execution_time)
    """
    start_time = time.time()
    # NUM_SEQS = 10 provides a good balance between exploration (multiple starts)
    # and exploitation (Simulated Annealing on the best candidate).
    NUM_SEQS = 10

    # Workload 1: Complex mixed read/write transactions
    workload = Workload(WORKLOAD_1)
    makespan1, schedule1 = get_best_schedule(workload, NUM_SEQS)
    cost1 = workload.get_opt_seq_cost(schedule1)

    # Workload 2: Simple read-then-write pattern
    workload2 = Workload(WORKLOAD_2)
    makespan2, schedule2 = get_best_schedule(workload2, NUM_SEQS)
    cost2 = workload2.get_opt_seq_cost(schedule2)

    # Workload 3: Minimal read/write operations
    workload3 = Workload(WORKLOAD_3)
    makespan3, schedule3 = get_best_schedule(workload3, NUM_SEQS)
    cost3 = workload3.get_opt_seq_cost(schedule3)

    total_makespan = cost1 + cost2 + cost3
    schedules = [schedule1, schedule2, schedule3]
    execution_time = time.time() - start_time

    return total_makespan, schedules, execution_time


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_scheduling():
    """Run the transaction scheduling algorithm for all workloads"""
    total_makespan, schedules, execution_time = get_random_costs()
    return total_makespan, schedules, execution_time


if __name__ == "__main__":
    total_makespan, schedules, execution_time = run_scheduling()
    print(f"Total makespan: {total_makespan}, Execution time: {execution_time:.4f}s")
    print(f"Individual workload costs: {[workload.get_opt_seq_cost(schedule) for workload, schedule in zip([Workload(WORKLOAD_1), Workload(WORKLOAD_2), Workload(WORKLOAD_3)], schedules)]}")