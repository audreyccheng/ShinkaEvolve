# EVOLVE-BLOCK-START
"""Transaction scheduling algorithm for optimizing makespan across multiple workloads"""

import time
import random
import sys
import os

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
    Get optimal schedule using repeated greedy construction with full local refinement,
    followed by Iterated Local Search (ILS) on the best candidate.

    Strategy:
    1. Generate `num_seqs` schedules. For each:
       - Construct using Randomized Greedy with lookahead.
       - Immediately apply full Adjacent Swap Descent (Local Search) to reach a local optimum.
         (Optimizing every candidate is crucial as greedy rank != local optimum rank).
    2. Select the global best from step 1.
    3. Apply Iterated Local Search (Kick + Descent) to the champion to escape local optima.

    Args:
        workload: Workload object
        num_seqs: Number of start points

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    SAMPLE_SIZE = 12
    ILS_ITERATIONS = 5     # Number of perturbation cycles
    KICK_STRENGTH = 4      # Number of random moves per kick

    candidates = []

    # Phase 1: Exploration (Generate & Refine)
    for _ in range(num_seqs):
        # A. Randomized Greedy Construction
        remaining = list(range(workload.num_txns))
        start_txn = random.choice(remaining)
        current_seq = [start_txn]
        remaining.remove(start_txn)

        # Track cost to avoid re-calculation
        current_makespan = workload.get_opt_seq_cost(current_seq)

        while remaining:
            # Sample subset of remaining transactions
            sample_size = min(len(remaining), SAMPLE_SIZE)
            sample_candidates = random.sample(remaining, sample_size)

            best_next = -1
            min_inc_cost = float('inf')

            for t in sample_candidates:
                # Evaluate appending t
                test_seq = current_seq + [t]
                cost = workload.get_opt_seq_cost(test_seq)

                if cost < min_inc_cost:
                    min_inc_cost = cost
                    best_next = t
                    # Optimization: Early exit if fits perfectly (no increase in makespan)
                    if cost <= current_makespan:
                        break

            current_seq.append(best_next)
            remaining.remove(best_next)
            current_makespan = min_inc_cost

        # B. Full Adjacent Swap Descent (Local Search)
        # Run until convergence for EVERY candidate
        improved = True
        while improved:
            improved = False
            for i in range(len(current_seq) - 1):
                # Try swap
                current_seq[i], current_seq[i+1] = current_seq[i+1], current_seq[i]
                new_cost = workload.get_opt_seq_cost(current_seq)

                if new_cost < current_makespan:
                    current_makespan = new_cost
                    improved = True
                else:
                    # Revert
                    current_seq[i], current_seq[i+1] = current_seq[i+1], current_seq[i]

        candidates.append((current_makespan, current_seq))

    # Phase 2: Exploitation (Iterated Local Search on Champion)
    candidates.sort(key=lambda x: x[0])
    best_cost, best_seq = candidates[0]

    # ILS Loop
    for _ in range(ILS_ITERATIONS):
        # Save current best state
        saved_seq = list(best_seq)

        # 1. Perturbation (Kick)
        # Apply multiple random insertions to shake up the schedule
        for _ in range(KICK_STRENGTH):
            if len(best_seq) < 2: break
            idx_from = random.randint(0, len(best_seq) - 1)
            idx_to = random.randint(0, len(best_seq) - 1)
            if idx_from != idx_to:
                txn = best_seq.pop(idx_from)
                best_seq.insert(idx_to, txn)

        # 2. Descent (Repair)
        # Optimize the perturbed schedule
        current_cost = workload.get_opt_seq_cost(best_seq)
        improved = True
        while improved:
            improved = False
            for i in range(len(best_seq) - 1):
                best_seq[i], best_seq[i+1] = best_seq[i+1], best_seq[i]
                new_cost = workload.get_opt_seq_cost(best_seq)
                if new_cost < current_cost:
                    current_cost = new_cost
                    improved = True
                else:
                    best_seq[i], best_seq[i+1] = best_seq[i+1], best_seq[i]

        # 3. Acceptance
        # If better, keep it. If worse, revert (Simple greedy acceptance)
        if current_cost < best_cost:
            best_cost = current_cost
        else:
            best_seq = saved_seq

    return best_cost, best_seq


def get_random_costs():
    """
    Evaluate scheduling algorithm on three different workloads.

    Returns:
        Tuple of (total_makespan, list_of_schedules, execution_time)
    """
    start_time = time.time()
    workload_size = 100

    # Workload 1: Complex mixed read/write transactions
    workload = Workload(WORKLOAD_1)
    makespan1, schedule1 = get_best_schedule(workload, 10)
    cost1 = workload.get_opt_seq_cost(schedule1)

    # Workload 2: Simple read-then-write pattern
    workload2 = Workload(WORKLOAD_2)
    makespan2, schedule2 = get_best_schedule(workload2, 10)
    cost2 = workload2.get_opt_seq_cost(schedule2)

    # Workload 3: Minimal read/write operations
    workload3 = Workload(WORKLOAD_3)
    makespan3, schedule3 = get_best_schedule(workload3, 10)
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