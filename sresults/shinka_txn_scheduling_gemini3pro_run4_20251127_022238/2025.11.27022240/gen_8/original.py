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
    Get optimal schedule using greedy cost sampling strategy.

    Args:
        workload: Workload object containing transaction data
        num_seqs: Number of sequences to sample for greedy selection

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    best_overall_cost = float('inf')
    best_schedule = []

    # Run multiple iterations of the greedy strategy
    for _ in range(num_seqs):
        # Random starting point
        start_txn = random.randint(0, workload.num_txns - 1)
        txn_seq = [start_txn]
        remaining_txns = list(range(workload.num_txns))
        remaining_txns.remove(start_txn)

        # Build sequence step by step
        for i in range(workload.num_txns - 1):
            min_cost = float('inf')
            min_txn = -1
            holdout_txns = []

            # Sample a subset of remaining transactions to test
            # Using 15 samples provides a good balance of exploration vs speed
            num_samples = 15
            current_samples = min(num_samples, len(remaining_txns))

            for j in range(current_samples):
                # Pick a random candidate from remaining
                idx = random.randint(0, len(remaining_txns) - 1)
                t = remaining_txns[idx]
                holdout_txns.append(remaining_txns.pop(idx))

                # Evaluate adding this candidate
                # Note: We pass a new list to avoid modifying txn_seq yet
                cost = workload.get_opt_seq_cost(txn_seq + [t])

                if cost < min_cost:
                    min_cost = cost
                    min_txn = t

            # Append the best candidate found
            txn_seq.append(min_txn)

            # Restore the other candidates to the remaining list
            holdout_txns.remove(min_txn)
            remaining_txns.extend(holdout_txns)

        # Evaluate final complete schedule
        overall_cost = workload.get_opt_seq_cost(txn_seq)
        if overall_cost < best_overall_cost:
            best_overall_cost = overall_cost
            best_schedule = txn_seq

    return best_overall_cost, best_schedule


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