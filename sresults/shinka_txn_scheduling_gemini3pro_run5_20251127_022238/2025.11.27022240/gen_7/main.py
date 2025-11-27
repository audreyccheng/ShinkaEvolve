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
    Get optimal schedule using Exhaustive Beam Search construction followed by Simulated Annealing.

    Args:
        workload: Workload object containing transaction data
        num_seqs: Parameter affecting the computational budget

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    # Hyperparameters
    # Use budget for one high-quality construction instead of multiple restarts
    # Width 4 allows maintaining diversity while exhaustive search ensures local optimality
    BEAM_WIDTH = 4

    # Simulated Annealing Parameters
    # Increased iterations as we only run once
    SA_ITERATIONS = 4000
    SA_COOLING_RATE = 0.9985

    best_global_cost = float('inf')
    best_global_schedule = []

    # Cache for costs to avoid re-simulating identical schedules
    cost_cache = {}

    def get_cost_cached(seq):
        # Convert to tuple for hashing
        t_seq = tuple(seq)
        if t_seq in cost_cache:
            return cost_cache[t_seq]

        c = workload.get_opt_seq_cost(seq)
        cost_cache[t_seq] = c
        return c

    # --- PHASE 1: Exhaustive Beam Search Construction ---
    # Step 1: Initialize beam by evaluating ALL possible start transactions
    initial_candidates = []
    for t in range(workload.num_txns):
        seq = [t]
        cost = get_cost_cached(seq)
        initial_candidates.append((cost, seq, [x for x in range(workload.num_txns) if x != t]))

    # Sort and take top BEAM_WIDTH
    initial_candidates.sort(key=lambda x: x[0])
    beam = initial_candidates[:BEAM_WIDTH]

    # Iteratively build the schedule
    # We need to add (num_txns - 1) more transactions
    for _ in range(workload.num_txns - 1):
        next_beam_candidates = []

        for b_cost, b_seq, b_rem in beam:
            # Evaluate ALL remaining transactions
            # Exhaustive search ensures we pick the absolute best local moves
            for cand in b_rem:
                new_seq = b_seq + [cand]
                new_cost = get_cost_cached(new_seq)

                # Store candidate info
                next_beam_candidates.append((new_cost, new_seq, b_rem, cand))

        # Prune: Sort by cost and keep top BEAM_WIDTH
        # This keeps the best paths found so far
        next_beam_candidates.sort(key=lambda x: x[0])

        new_beam = []
        # Filter to ensure we don't pick duplicates (though unlikely with different parents)
        # and limit to BEAM_WIDTH
        for c_cost, c_seq, c_parent_rem, c_cand in next_beam_candidates:
            if len(new_beam) >= BEAM_WIDTH:
                break

            # Construct the new remaining list for the survivor
            new_rem = list(c_parent_rem)
            new_rem.remove(c_cand)
            new_beam.append((c_cost, c_seq, new_rem))

        beam = new_beam

    # Best result from Beam Search
    if not beam:
        # Fallback if beam empty (should not happen)
        return float('inf'), []

    current_schedule = beam[0][1]
    current_cost = beam[0][0]

    # --- PHASE 2: Simulated Annealing Refinement ---
    # Start temperature proportional to cost
    T = current_cost * 0.05

    best_run_schedule = list(current_schedule)
    best_run_cost = current_cost

    for k in range(SA_ITERATIONS):
        # Create neighbor by perturbing the schedule
        neighbor = list(current_schedule)
        idx1 = random.randint(0, len(neighbor) - 1)
        idx2 = random.randint(0, len(neighbor) - 1)

        if idx1 == idx2:
            continue

        # Randomly choose perturbation type
        if random.random() < 0.5:
            # Swap
            neighbor[idx1], neighbor[idx2] = neighbor[idx2], neighbor[idx1]
        else:
            # Insert (Move item at idx1 to idx2)
            val = neighbor.pop(idx1)
            neighbor.insert(idx2, val)

        new_cost = get_cost_cached(neighbor)
        delta = new_cost - current_cost

        # Acceptance Probability
        accept = False
        if delta < 0:
            accept = True
        elif T > 1e-9:
            # Metropolis criterion
            p = math.exp(-delta / T)
            if random.random() < p:
                accept = True

        if accept:
            current_schedule = neighbor
            current_cost = new_cost
            if current_cost < best_run_cost:
                best_run_cost = current_cost
                best_run_schedule = list(current_schedule)

        # Cool down
        T *= SA_COOLING_RATE
        if T < 0.1: T = 0.1

    return best_run_cost, best_run_schedule


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