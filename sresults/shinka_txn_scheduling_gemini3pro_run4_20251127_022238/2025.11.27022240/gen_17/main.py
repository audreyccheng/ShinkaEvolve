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
    Get optimal schedule using "Big Rocks" greedy strategy combined with 
    Simulated Annealing optimization.

    Args:
        workload: Workload object containing transaction data
        num_seqs: Number of sequences to sample for greedy selection

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    # Pre-calculate individual transaction costs (lengths) for tie-breaking
    txn_durations = {t: workload.get_opt_seq_cost([t]) for t in range(workload.num_txns)}
    # Sort transactions by duration descending to prioritize long transactions (Big Rocks)
    sorted_txns_by_len = sorted(range(workload.num_txns), key=lambda t: txn_durations[t], reverse=True)

    best_overall_cost = float('inf')
    best_schedule = []

    # Iteratively build solutions
    for i in range(num_seqs):
        # Random starting point
        start_txn = random.randint(0, workload.num_txns - 1)
        txn_seq = [start_txn]
        
        remaining_txns = set(range(workload.num_txns))
        remaining_txns.remove(start_txn)
        remaining_list = list(remaining_txns)

        # Build sequence greedy-style
        while remaining_list:
            candidates = set()

            # Strategy:
            # 1. First iteration: Check ALL candidates for maximum quality baseline.
            # 2. Other iterations: Hybrid sampling (Big Rocks + Random).
            
            if i == 0:
                candidates = set(remaining_list)
            else:
                # "Big Rocks" - prioritize longest available transactions
                added_rocks = 0
                target_rocks = 5 # Slightly increased from 4
                for t in sorted_txns_by_len:
                    if t in remaining_txns:
                        candidates.add(t)
                        added_rocks += 1
                        if added_rocks >= target_rocks:
                            break
                
                # Random sampling for diversity
                target_total = 25 # Increased from 20
                if len(remaining_list) <= target_total:
                    candidates.update(remaining_list)
                else:
                    while len(candidates) < target_total:
                        candidates.add(random.choice(remaining_list))

            # Select best candidate
            # Criteria: Minimize new makespan, Tie-break: Maximize transaction length
            best_txn = -1
            best_cost_tuple = (float('inf'), float('-inf')) # (makespan, -length)

            for t in candidates:
                cost = workload.get_opt_seq_cost(txn_seq + [t])
                # We want smallest cost, then largest length (so smallest negative length)
                cost_tuple = (cost, -txn_durations[t])

                if cost_tuple < best_cost_tuple:
                    best_cost_tuple = cost_tuple
                    best_txn = t

            # Append best found
            txn_seq.append(best_txn)
            remaining_txns.remove(best_txn)
            remaining_list.remove(best_txn)

        # Check total cost
        overall_cost = workload.get_opt_seq_cost(txn_seq)
        if overall_cost < best_overall_cost:
            best_overall_cost = overall_cost
            best_schedule = txn_seq

    # Optimization Phase: Simulated Annealing
    # Allows escaping local optima by accepting worse solutions with probability
    # Also explicitly allows "sideways moves" (delta=0) to traverse plateaus
    if best_schedule:
        current_schedule = list(best_schedule)
        current_cost = best_overall_cost

        # SA Hyperparameters
        temperature = 100.0
        cooling_rate = 0.95
        min_temperature = 0.5
        iters_per_temp = 50 

        while temperature > min_temperature:
            for _ in range(iters_per_temp):
                neighbor = list(current_schedule)

                # Propose a move: 70% Insert, 30% Swap
                # Insertion generally preserves relative ordering better
                if random.random() < 0.7:
                    # Insert Move
                    idx1 = random.randint(0, len(neighbor) - 1)
                    idx2 = random.randint(0, len(neighbor) - 1)
                    if idx1 != idx2:
                        item = neighbor.pop(idx1)
                        neighbor.insert(idx2, item)
                else:
                    # Swap Move
                    idx1, idx2 = random.sample(range(len(neighbor)), 2)
                    neighbor[idx1], neighbor[idx2] = neighbor[idx2], neighbor[idx1]

                # Evaluate
                new_cost = workload.get_opt_seq_cost(neighbor)
                delta = new_cost - current_cost

                # Acceptance criterion
                accept = False
                if delta <= 0:
                    # Always accept improvements and sideways moves
                    accept = True
                else:
                    # Metropolis acceptance for worse moves
                    try:
                        prob = math.exp(-delta / temperature)
                    except OverflowError:
                        prob = 0
                    if random.random() < prob:
                        accept = True

                if accept:
                    current_schedule = neighbor
                    current_cost = new_cost
                    if current_cost < best_overall_cost:
                        best_overall_cost = current_cost
                        best_schedule = list(current_schedule)

            temperature *= cooling_rate

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