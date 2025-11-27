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
    Get optimal schedule using greedy cost sampling strategy with LAHC refinement.
    Features:
    - Dynamic Big Rocks selection
    - Late Acceptance Hill Climbing (LAHC)
    - Block insertion operators

    Args:
        workload: Workload object containing transaction data
        num_seqs: Number of sequences to sample for greedy selection

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    # Precompute transaction durations
    txn_durations = {t: workload.get_opt_seq_cost([t]) for t in range(workload.num_txns)}
    # Sort transactions by duration descending
    sorted_txns_by_len = sorted(range(workload.num_txns), key=lambda t: txn_durations[t], reverse=True)

    candidates_pool = []

    # 1. Greedy Construction Phase
    for _ in range(num_seqs):
        start_txn = random.randint(0, workload.num_txns - 1)
        txn_seq = [start_txn]
        remaining_set = set(range(workload.num_txns))
        remaining_set.remove(start_txn)

        while remaining_set:
            candidates = set()

            # Dynamic Big Rocks: Pick transactions close to the longest remaining
            max_remaining_dur = 0
            # Identify max duration in remaining set (optimize by iterating sorted list)
            for t in sorted_txns_by_len:
                if t in remaining_set:
                    max_remaining_dur = txn_durations[t]
                    break

            threshold = max_remaining_dur * 0.90

            rock_count = 0
            for t in sorted_txns_by_len:
                if t in remaining_set:
                    if txn_durations[t] >= threshold:
                        candidates.add(t)
                        rock_count += 1
                        if rock_count >= 5:
                            break
                    else:
                        break # Sorted, so subsequent are smaller

            # Random Sampling
            if len(remaining_set) <= 20:
                candidates.update(remaining_set)
            else:
                # Efficient sampling from set
                # Convert set to list only if needed, or maintain shadow list.
                # For this problem size, list conversion is acceptable.
                temp_list = list(remaining_set)
                sample_size = min(len(temp_list), 20 - len(candidates))
                if sample_size > 0:
                    candidates.update(random.sample(temp_list, sample_size))

            # Select Best Candidate
            best_candidate = -1
            best_metric = (float('inf'), float('-inf')) # (cost, -duration)

            for t in candidates:
                # Evaluate cost of appending t
                cost = workload.get_opt_seq_cost(txn_seq + [t])

                # Metric: Lower cost is better; Higher duration (lower negative) is tie-breaker
                metric = (cost, -txn_durations[t])

                if metric < best_metric:
                    best_metric = metric
                    best_candidate = t

            txn_seq.append(best_candidate)
            remaining_set.remove(best_candidate)

        overall_cost = workload.get_opt_seq_cost(txn_seq)
        candidates_pool.append((overall_cost, txn_seq))

    # 2. Optimization Phase (LAHC)
    # Sort by cost and pick the best one to refine
    candidates_pool.sort(key=lambda x: x[0])
    best_overall_cost, best_schedule = candidates_pool[0]

    current_schedule = list(best_schedule)
    current_cost = best_overall_cost

    # LAHC Parameters
    L = 50
    history = [current_cost] * L

    # Budget
    num_optimizations = 2500

    for k in range(num_optimizations):
        neighbor = list(current_schedule)
        op_rnd = random.random()

        # Operators: 60% Single Insert, 20% Block Insert, 20% Swap
        if op_rnd < 0.60:
            # Single Insert
            idx1 = random.randint(0, len(neighbor) - 1)
            idx2 = random.randint(0, len(neighbor) - 1)
            if idx1 != idx2:
                item = neighbor.pop(idx1)
                neighbor.insert(idx2, item)

        elif op_rnd < 0.80:
            # Block Insert
            block_len = random.randint(2, 3)
            if len(neighbor) > block_len:
                start_idx = random.randint(0, len(neighbor) - block_len)
                # Create block
                block = neighbor[start_idx : start_idx + block_len]
                # Remove block
                del neighbor[start_idx : start_idx + block_len]
                # Insert block
                insert_idx = random.randint(0, len(neighbor))
                neighbor[insert_idx:insert_idx] = block
        else:
            # Swap
            idx1, idx2 = random.sample(range(len(neighbor)), 2)
            neighbor[idx1], neighbor[idx2] = neighbor[idx2], neighbor[idx1]

        new_cost = workload.get_opt_seq_cost(neighbor)

        # LAHC Acceptance Logic
        v = k % L
        if new_cost <= current_cost or new_cost <= history[v]:
            current_schedule = neighbor
            current_cost = new_cost
            if current_cost < best_overall_cost:
                best_overall_cost = current_cost
                best_schedule = list(current_schedule)

        # Update history
        history[v] = current_cost

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