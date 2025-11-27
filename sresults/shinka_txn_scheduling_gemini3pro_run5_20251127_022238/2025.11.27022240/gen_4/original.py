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
    Get optimal schedule using randomized greedy search with restarts and heuristics.

    Args:
        workload: Workload object containing transaction data
        num_seqs: Number of sequences to sample (restarts)

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    num_txns = workload.num_txns
    
    # Pre-calculate individual transaction costs (lengths) for heuristics
    # Accessing txn structure based on usage patterns in simulator: workload.txns[i][0][3]
    try:
        txn_costs = [(i, workload.txns[i][0][3]) for i in range(num_txns)]
    except (IndexError, TypeError, AttributeError):
        # Fallback if structure is different
        txn_costs = [(i, 1) for i in range(num_txns)]
        
    # Sort by cost descending (Longest Processing Time first heuristic)
    # This helps in prioritizing heavy transactions for evaluation
    txn_costs.sort(key=lambda x: x[1], reverse=True)
    sorted_txns = [x[0] for x in txn_costs]
    
    best_schedule = None
    min_makespan = float('inf')
    
    # Tuning parameters
    # SAMPLE_SIZE: Candidates to check at each step.
    # A blend of heuristic-selected and random candidates.
    SAMPLE_SIZE = 8
    
    # Perform multiple restarts to avoid local optima
    for _ in range(max(1, num_seqs)):
        # Start with a random transaction to ensure diversity across restarts
        start_txn = random.randint(0, num_txns - 1)
        current_seq = [start_txn]
        remaining = set(range(num_txns))
        remaining.remove(start_txn)
        
        # Greedy construction loop
        while remaining:
            candidates = set()
            
            # 1. Heuristic: Always consider the longest remaining transactions
            # This ensures we try to place "big rocks" effectively
            added_heuristic = 0
            for t in sorted_txns:
                if t in remaining:
                    candidates.add(t)
                    added_heuristic += 1
                    if added_heuristic >= 3: # Top 3 longest
                        break
            
            # 2. Random: Fill the rest of the sample size with random remaining transactions
            # This provides exploration to find non-obvious fits
            remaining_list = list(remaining)
            needed = SAMPLE_SIZE - len(candidates)
            
            if needed > 0 and remaining_list:
                if len(remaining_list) <= needed:
                    candidates.update(remaining_list)
                else:
                    candidates.update(random.sample(remaining_list, needed))
            
            # Evaluate candidates
            best_next_txn = -1
            best_next_cost = float('inf')
            
            if len(candidates) == 1:
                # Optimization: skip simulation if only one choice
                best_next_txn = list(candidates)[0]
            else:
                # Select the candidate that minimizes the makespan of the partial sequence
                for t in candidates:
                    trial_seq = current_seq + [t]
                    cost = workload.get_opt_seq_cost(trial_seq)
                    
                    if cost < best_next_cost:
                        best_next_cost = cost
                        best_next_txn = t
            
            current_seq.append(best_next_txn)
            remaining.remove(best_next_txn)
            
        # Evaluate final full sequence
        final_cost = workload.get_opt_seq_cost(current_seq)
        
        if final_cost < min_makespan:
            min_makespan = final_cost
            best_schedule = current_seq
            
    return min_makespan, best_schedule


def get_random_costs():
    """
    Evaluate scheduling algorithm on three different workloads.
    
    Returns:
        Tuple of (total_makespan, list_of_schedules, execution_time)
    """
    start_time = time.time()
    
    # Workload 1: Complex mixed read/write transactions
    workload1 = Workload(WORKLOAD_1)
    makespan1, schedule1 = get_best_schedule(workload1, 10)
    cost1 = workload1.get_opt_seq_cost(schedule1)

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
