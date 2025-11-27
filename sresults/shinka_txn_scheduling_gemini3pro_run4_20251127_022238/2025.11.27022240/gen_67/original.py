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
    Get optimal schedule using Weighted Greedy Construction and LAHC Refinement.

    Args:
        workload: Workload object containing transaction data
        num_seqs: Number of sequences to sample for greedy selection

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    # 1. Pre-calculation
    # Compute transaction durations for heuristic guidance
    txn_durations = {t: workload.get_opt_seq_cost([t]) for t in range(workload.num_txns)}
    # Sort transactions by duration (descending) for Big Rocks lookup
    sorted_txns_by_len = sorted(range(workload.num_txns), key=lambda t: txn_durations[t], reverse=True)

    # Helper to generate a single greedy schedule
    def generate_greedy_schedule(iteration_idx):
        start_txn = random.randint(0, workload.num_txns - 1)
        schedule = [start_txn]
        remaining = set(range(workload.num_txns))
        remaining.remove(start_txn)

        while remaining:
            candidates = set()

            # A. Big Rocks Strategy
            # Always consider the top 5 longest available transactions
            rocks_added = 0
            for t in sorted_txns_by_len:
                if t in remaining:
                    candidates.add(t)
                    rocks_added += 1
                    if rocks_added >= 5:
                        break

            # B. Random Sampling / Full Scan
            if iteration_idx == 0:
                # Full scan for the first iteration (Baseline)
                candidates = remaining
            else:
                # Random sample for diversity
                # Pool size 20 is a good balance between speed and quality
                pool_list = list(remaining)
                sample_size = min(20, len(pool_list))
                if sample_size < len(pool_list):
                    candidates.update(random.sample(pool_list, sample_size))
                else:
                    candidates.update(pool_list)

            # C. Selection with Weighted Score
            # Score = NewMakespan - (Alpha * TxnDuration)
            # This heuristic allows a slight increase in makespan if it allows
            # scheduling a significantly longer transaction earlier.
            alpha = 0.3
            best_t = -1
            best_score = float('inf')

            for t in candidates:
                # Calculate makespan of the partial sequence
                # We reuse the Simulator logic here
                new_makespan = workload.get_opt_seq_cost(schedule + [t])

                # Weighted score
                score = new_makespan - (alpha * txn_durations[t])

                if score < best_score:
                    best_score = score
                    best_t = t
                elif score == best_score:
                    # Tie-break: Prefer longer duration (Standard Big Rocks)
                    if txn_durations[t] > txn_durations[best_t]:
                        best_t = t

            schedule.append(best_t)
            remaining.remove(best_t)

        total_cost = workload.get_opt_seq_cost(schedule)
        return total_cost, schedule

    # 2. Greedy Phase: Generate candidates
    candidates_pool = []
    for i in range(num_seqs):
        candidates_pool.append(generate_greedy_schedule(i))

    # 3. Candidate Selection
    # Pick top 3 distinct schedules to optimize
    candidates_pool.sort(key=lambda x: x[0])

    unique_candidates = []
    seen_hashes = set()

    for cost, seq in candidates_pool:
        # Use tuple of sequence as hash for uniqueness check
        seq_hash = tuple(seq)
        if seq_hash not in seen_hashes:
            unique_candidates.append((cost, list(seq)))
            seen_hashes.add(seq_hash)
        if len(unique_candidates) >= 3:
            break

    # Fallback if diversity is low
    if not unique_candidates and candidates_pool:
        unique_candidates = [(candidates_pool[0][0], list(candidates_pool[0][1]))]

    # 4. LAHC Optimization Helper
    def run_lahc(schedule, initial_cost, budget):
        current_seq = list(schedule)
        current_cost = initial_cost
        best_seq = list(schedule)
        best_cost = initial_cost

        # LAHC Parameters
        history_len = 75
        history = [initial_cost] * history_len

        for k in range(budget):
            neighbor = list(current_seq)
            op_rand = random.random()

            # Operators: 60% Insert, 30% Block Move, 10% Swap
            if op_rand < 0.60:
                # Single Insert
                idx1, idx2 = random.sample(range(len(neighbor)), 2)
                item = neighbor.pop(idx1)
                neighbor.insert(idx2, item)

            elif op_rand < 0.90:
                # Block Insert (Move chunk of 2-5 items)
                # Preserves local dependencies/structure
                bsize = random.randint(2, 5)
                if len(neighbor) > bsize:
                    start = random.randint(0, len(neighbor) - bsize)
                    block = neighbor[start : start + bsize]
                    del neighbor[start : start + bsize]
                    dest = random.randint(0, len(neighbor))
                    neighbor[dest : dest] = block
                else:
                    continue
            else:
                # Swap (Destructive but good for jumps)
                idx1, idx2 = random.sample(range(len(neighbor)), 2)
                neighbor[idx1], neighbor[idx2] = neighbor[idx2], neighbor[idx1]

            new_cost = workload.get_opt_seq_cost(neighbor)

            # Late Acceptance Logic
            h_idx = k % history_len
            if new_cost <= current_cost or new_cost <= history[h_idx]:
                current_seq = neighbor
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_seq = list(current_seq)

            history[h_idx] = current_cost

        return best_cost, best_seq

    # 5. Refinement Phase (Sprint)
    # Short optimization run on all top candidates
    refined_candidates = []
    for cost, seq in unique_candidates:
        rc, rs = run_lahc(seq, cost, 300)
        refined_candidates.append((rc, rs))

    # 6. Final Optimization (Marathon)
    # Long optimization run on the winner
    refined_candidates.sort(key=lambda x: x[0])
    winner_cost, winner_seq = refined_candidates[0]

    final_cost, final_seq = run_lahc(winner_seq, winner_cost, 1800)

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