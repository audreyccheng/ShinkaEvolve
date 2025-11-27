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
    Get optimal schedule using Advanced Greedy Construction and Hybrid Local Search.

    Combines 'Smart Sampling' (prioritizing longest transactions) with 'Best Fit Descending'
    tie-breaking, followed by a two-phase local search (Adjacent Swap + Random Shift) to
    minimize makespan.

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

    # Sort transactions by length (descending) for smart sampling
    sorted_txns = sorted(range(workload.num_txns), key=lambda t: txn_lens[t], reverse=True)

    # Heuristic Parameters
    SAMPLE_LONGEST = 5      # Explicitly check top K longest remaining
    SAMPLE_RANDOM = 10      # Plus M random candidates
    SHIFT_ATTEMPTS = 50     # Budget for insertion-based local search

    best_overall_cost = float('inf')
    best_overall_schedule = []

    for _ in range(num_seqs):
        # 1. Greedy Construction
        remaining = set(range(workload.num_txns))

        # Random start for diversity
        start_txn = random.choice(list(remaining))
        current_seq = [start_txn]
        remaining.remove(start_txn)

        while remaining:
            candidates = set()

            # A. Smart Sampling: Check longest available transactions
            # This ensures we try to pack "big rocks" if they fit well
            count = 0
            for t in sorted_txns:
                if t in remaining:
                    candidates.add(t)
                    count += 1
                    if count >= SAMPLE_LONGEST:
                        break

            # B. Random Sampling: Maintain diversity
            pool = list(remaining)
            if len(pool) > SAMPLE_RANDOM:
                candidates.update(random.sample(pool, SAMPLE_RANDOM))
            else:
                candidates.update(pool)

            # C. Evaluate Candidates
            # Objective: Minimize Cost, Break ties with Max Length (Best Fit Descending)
            best_candidate = -1
            best_score = (float('inf'), 0) # (cost, -length)

            for t in candidates:
                test_seq = current_seq + [t]
                cost = workload.get_opt_seq_cost(test_seq)

                score = (cost, -txn_lens[t])

                if score < best_score:
                    best_score = score
                    best_candidate = t

            current_seq.append(best_candidate)
            remaining.remove(best_candidate)

        # 2. Local Search Refinement
        current_cost = workload.get_opt_seq_cost(current_seq)

        # Phase A: Adjacent Swap Descent (Hill Climbing)
        # Fixes local ordering issues efficiently
        improved = True
        while improved:
            improved = False
            for i in range(len(current_seq) - 1):
                current_seq[i], current_seq[i+1] = current_seq[i+1], current_seq[i]
                new_cost = workload.get_opt_seq_cost(current_seq)

                if new_cost < current_cost:
                    current_cost = new_cost
                    improved = True
                else:
                    # Revert
                    current_seq[i], current_seq[i+1] = current_seq[i+1], current_seq[i]

        # Phase B: Shift Descent (Random Insertion)
        # Moves a transaction to a random new position to escape local optima
        # Effectively jumps over dependency barriers that swaps can't cross easily
        for _ in range(SHIFT_ATTEMPTS):
            if len(current_seq) < 2: break

            idx_from = random.randint(0, len(current_seq) - 1)
            idx_to = random.randint(0, len(current_seq) - 1)
            if idx_from == idx_to: continue

            # Apply Move
            txn = current_seq.pop(idx_from)
            current_seq.insert(idx_to, txn)

            new_cost = workload.get_opt_seq_cost(current_seq)

            if new_cost < current_cost:
                current_cost = new_cost
                # Keep change
            else:
                # Revert
                current_seq.pop(idx_to)
                current_seq.insert(idx_from, txn)

        # Update global best
        if current_cost < best_overall_cost:
            best_overall_cost = current_cost
            best_overall_schedule = list(current_seq)

    return best_overall_cost, best_overall_schedule


def get_random_costs():
    """
    Evaluate scheduling algorithm on three different workloads.

    Returns:
        Tuple of (total_makespan, list_of_schedules, execution_time)
    """
    start_time = time.time()
    # Increased number of sequences to improve exploration since ILS was removed
    NUM_SEQS = 12

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