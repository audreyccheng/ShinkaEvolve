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
    Get optimal schedule using beam search followed by extensive local search.

    Args:
        workload: Workload object containing transaction data
        num_seqs: Number of sequences to sample for greedy selection (used as beam width)

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    # Beam Search configuration
    beam_width = num_seqs
    candidates_per_node = 20

    # Initialize beam with unique random start transactions
    all_txns = list(range(workload.num_txns))
    num_starts = min(beam_width, len(all_txns))
    start_txns = random.sample(all_txns, num_starts)

    # Beam element: (cost, schedule, remaining_txns_list)
    beam = []
    for t in start_txns:
        rem = list(all_txns)
        rem.remove(t)
        cost = workload.get_opt_seq_cost([t])
        beam.append((cost, [t], rem))

    # Iteratively expand the beam
    target_length = workload.num_txns
    while len(beam[0][1]) < target_length:
        candidates = []
        for cost, sched, rem in beam:
            # Sample subset of remaining transactions
            n_sample = min(len(rem), candidates_per_node)
            next_txns = random.sample(rem, n_sample)

            for next_txn in next_txns:
                new_sched = sched + [next_txn]
                new_cost = workload.get_opt_seq_cost(new_sched)
                # Store needed info to reconstruct state (avoid deep copying rem yet)
                candidates.append((new_cost, new_sched, rem, next_txn))

        # Select top k candidates based on lowest cost
        # Shuffle first to break ties randomly
        random.shuffle(candidates)
        candidates.sort(key=lambda x: x[0])
        best_candidates = candidates[:beam_width]

        # Construct next beam
        new_beam = []
        for cost, new_sched, old_rem, added_txn in best_candidates:
            new_rem = list(old_rem)
            new_rem.remove(added_txn)
            new_beam.append((cost, new_sched, new_rem))
        beam = new_beam

    # Select best schedule from beam
    best_cost, best_schedule, _ = min(beam, key=lambda x: x[0])

    # Extended Local Search phase on the single best schedule
    # Uses both Move and Swap operators
    current_schedule = list(best_schedule)
    current_cost = best_cost

    # Iterations count balanced to fit within execution time budget
    # Saved comparisons from beam search allow for more local search
    iterations = 3000

    for _ in range(iterations):
        op = random.random()
        test_schedule = list(current_schedule)

        if op < 0.5:
            # Move operator: pick random txn and insert elsewhere
            idx_from = random.randint(0, len(test_schedule) - 1)
            txn = test_schedule.pop(idx_from)
            idx_to = random.randint(0, len(test_schedule)) # can insert at end
            test_schedule.insert(idx_to, txn)
        else:
            # Swap operator: swap two random positions
            idx1 = random.randint(0, len(test_schedule) - 1)
            idx2 = random.randint(0, len(test_schedule) - 1)
            if idx1 == idx2: continue
            test_schedule[idx1], test_schedule[idx2] = test_schedule[idx2], test_schedule[idx1]

        new_cost = workload.get_opt_seq_cost(test_schedule)

        if new_cost < current_cost:
            current_cost = new_cost
            current_schedule = test_schedule

    return current_cost, current_schedule


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