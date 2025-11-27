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
    Get optimal schedule using Dynamic Big Rocks greedy strategy
    and Late Acceptance Hill Climbing (LAHC) refinement.

    Args:
        workload: Workload object containing transaction data
        num_seqs: Number of sequences to sample for greedy selection

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    # 1. Precompute transaction durations
    txn_durations = {t: workload.get_opt_seq_cost([t]) for t in range(workload.num_txns)}
    sorted_txns_by_len = sorted(range(workload.num_txns), key=lambda t: txn_durations[t], reverse=True)

    candidates_pool = []

    # 2. Greedy Construction Loop
    for i in range(num_seqs):
        # Initial transaction
        start_txn = random.randint(0, workload.num_txns - 1)
        txn_seq = [start_txn]
        remaining_txns = set(range(workload.num_txns))
        remaining_txns.remove(start_txn)

        while remaining_txns:
            candidates = set()

            # Dynamic Big Rocks Logic
            # Find max remaining duration
            max_rem_dur = 0
            for t in sorted_txns_by_len:
                if t in remaining_txns:
                    max_rem_dur = txn_durations[t]
                    break

            # Add big rocks (>= 85% of max duration)
            threshold = max_rem_dur * 0.85
            rocks_added = 0
            for t in sorted_txns_by_len:
                if t in remaining_txns:
                    if txn_durations[t] >= threshold:
                        candidates.add(t)
                        rocks_added += 1
                        if rocks_added >= 5:
                            break
                    else:
                        break # Sorted

            # Add random samples for diversity
            target_size = 20
            remaining_list = list(remaining_txns)

            if len(remaining_list) <= target_size:
                candidates.update(remaining_list)
            else:
                while len(candidates) < target_size:
                    candidates.add(random.choice(remaining_list))

            # Select best candidate
            # Criteria: Minimize Cost, Tie-break: Maximize Duration
            best_t = -1
            best_metric = (float('inf'), float('-inf'))

            for t in candidates:
                cost = workload.get_opt_seq_cost(txn_seq + [t])
                metric = (cost, -txn_durations[t])
                if metric < best_metric:
                    best_metric = metric
                    best_t = t

            txn_seq.append(best_t)
            remaining_txns.remove(best_t)

        total_cost = workload.get_opt_seq_cost(txn_seq)
        candidates_pool.append((total_cost, txn_seq))

    # 3. Multi-Candidate Selection
    candidates_pool.sort(key=lambda x: x[0])

    unique_candidates = []
    seen_costs = set()
    for c, seq in candidates_pool:
        if c not in seen_costs:
            unique_candidates.append((c, list(seq)))
            seen_costs.add(c)
        if len(unique_candidates) >= 3:
            break

    if not unique_candidates and candidates_pool:
        unique_candidates.append((candidates_pool[0][0], list(candidates_pool[0][1])))

    # 4. Refinement with LAHC
    def run_lahc_optimization(schedule, start_cost, steps):
        curr_seq = list(schedule)
        curr_cost = start_cost
        best_seq = list(schedule)
        best_cost = start_cost

        L = 50
        history = [curr_cost] * L

        for k in range(steps):
            op = random.random()
            neighbor = list(curr_seq)

            # Block Insert (30%) - Preserves subsequences
            if op < 0.30:
                block_len = random.randint(2, 4)
                if len(neighbor) > block_len:
                    start_idx = random.randint(0, len(neighbor) - block_len)
                    block = neighbor[start_idx : start_idx + block_len]
                    del neighbor[start_idx : start_idx + block_len]
                    insert_idx = random.randint(0, len(neighbor))
                    neighbor[insert_idx:insert_idx] = block

            # Single Insert (50%) - Fine tuning
            elif op < 0.80:
                idx1 = random.randint(0, len(neighbor) - 1)
                idx2 = random.randint(0, len(neighbor) - 1)
                if idx1 != idx2:
                    val = neighbor.pop(idx1)
                    neighbor.insert(idx2, val)

            # Swap (20%) - Exploration
            else:
                idx1, idx2 = random.sample(range(len(neighbor)), 2)
                neighbor[idx1], neighbor[idx2] = neighbor[idx2], neighbor[idx1]

            new_cost = workload.get_opt_seq_cost(neighbor)

            # LAHC Acceptance
            v = k % L
            accept = False
            if new_cost <= curr_cost or new_cost <= history[v]:
                accept = True

            if accept:
                curr_seq = neighbor
                curr_cost = new_cost
                if curr_cost < best_cost:
                    best_cost = curr_cost
                    best_seq = list(curr_seq)

            # Update history with current cost
            history[v] = curr_cost

        return best_cost, best_seq

    # Phase 4a: Preliminary Optimization (300 steps)
    refined_candidates = []
    for c, seq in unique_candidates:
        rc, rseq = run_lahc_optimization(seq, c, 300)
        refined_candidates.append((rc, rseq))

    # Phase 4b: Deep Optimization (Remaining Budget)
    refined_candidates.sort(key=lambda x: x[0])
    winner_cost, winner_seq = refined_candidates[0]

    final_cost, final_seq = run_lahc_optimization(winner_seq, winner_cost, 1500)

    return final_cost, final_seq


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