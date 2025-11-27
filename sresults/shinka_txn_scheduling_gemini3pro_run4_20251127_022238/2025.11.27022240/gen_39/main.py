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

    Strategies:
    1. Weighted Greedy with Dynamic Big Rocks: Uses a scoring function `cost - alpha * duration`
       to opportunistically schedule long transactions early.
    2. Multi-Candidate Sprint: Selects top distinct candidates for a short optimization sprint.
    3. LAHC with Enhanced Operators: Late Acceptance Hill Climbing with mixed operators including
       variable-size block moves to preserve schedule structure.
    """
    # 1. Pre-calculation
    # Compute transaction durations for heuristic guidance
    txn_durations = {t: workload.get_opt_seq_cost([t]) for t in range(workload.num_txns)}
    # Sort transactions by duration (descending) for Big Rocks lookup
    sorted_txns_by_len = sorted(range(workload.num_txns), key=lambda t: txn_durations[t], reverse=True)

    candidates_pool = []

    # 2. Greedy Construction Phase
    for i in range(num_seqs):
        # Determine alpha for this iteration
        # Alpha > 0 encourages picking longer transactions even if they increase makespan slightly
        if i == 0:
            alpha = 0.0
            start_txn = sorted_txns_by_len[0] # Deterministic start with biggest rock
        else:
            alpha = random.uniform(0.1, 0.5)
            start_txn = random.randint(0, workload.num_txns - 1)

        txn_seq = [start_txn]
        remaining_txns = list(range(workload.num_txns))
        remaining_txns.remove(start_txn)
        remaining_set = set(remaining_txns)

        while remaining_txns:
            candidates = set()

            if i == 0:
                # Full scan for first sequence
                candidates = set(remaining_txns)
            else:
                # Dynamic Big Rocks Heuristic
                # Identify max duration in remaining
                max_dur = 0
                for t in sorted_txns_by_len:
                    if t in remaining_set:
                        max_dur = txn_durations[t]
                        break

                # Dynamic threshold: 90% of max remaining duration
                threshold = max_dur * 0.90
                rocks_added = 0
                max_rocks = 8

                for t in sorted_txns_by_len:
                    if t in remaining_set:
                        if txn_durations[t] >= threshold:
                            candidates.add(t)
                            rocks_added += 1
                            if rocks_added >= max_rocks:
                                break
                        else:
                            break

                # Fill rest with random samples
                target_total = 20
                if len(remaining_txns) <= target_total:
                    candidates.update(remaining_txns)
                else:
                    needed = target_total - len(candidates)
                    if needed > 0:
                        candidates.update(random.sample(remaining_txns, needed))

            # Selection: Minimize Weighted Score
            best_t = -1
            best_score = float('inf')

            for t in candidates:
                cost = workload.get_opt_seq_cost(txn_seq + [t])

                # Weighted Score: Cost penalized by fraction of duration
                # A higher duration reduces the score, making the candidate more attractive
                score = cost - (alpha * txn_durations[t])

                if score < best_score:
                    best_score = score
                    best_t = t
                elif score == best_score:
                    # Tie-break: prefer longer duration
                    if txn_durations[t] > txn_durations.get(best_t, 0):
                        best_t = t

            txn_seq.append(best_t)
            remaining_txns.remove(best_t)
            remaining_set.remove(best_t)

        total_cost = workload.get_opt_seq_cost(txn_seq)
        candidates_pool.append((total_cost, txn_seq))

    # 3. Multi-Candidate Selection
    # Sort by raw cost first
    candidates_pool.sort(key=lambda x: x[0])

    unique_candidates = []
    seen_hashes = set()

    # Select top 3 distinct schedules
    for cost, seq in candidates_pool:
        h = tuple(seq)
        if h not in seen_hashes:
            unique_candidates.append((cost, list(seq)))
            seen_hashes.add(h)
        if len(unique_candidates) >= 3:
            break

    if not unique_candidates and candidates_pool:
        unique_candidates = [(candidates_pool[0][0], list(candidates_pool[0][1]))]

    # Helper function for LAHC Optimization
    def run_lahc(schedule, start_cost, budget):
        current_sched = list(schedule)
        current_c = start_cost
        best_s = list(schedule)
        best_c = start_cost

        history_len = 50
        history = [current_c] * history_len

        for k in range(budget):
            op_rand = random.random()
            neighbor = list(current_sched)

            # Operators: 50% Single Insert, 30% Block Move, 20% Swap
            if op_rand < 0.50:
                # Single Insert
                idx1, idx2 = random.sample(range(len(neighbor)), 2)
                item = neighbor.pop(idx1)
                neighbor.insert(idx2, item)

            elif op_rand < 0.80:
                # Block Insert (Move contiguous block of 2-5 items)
                # Larger blocks preserve more local structure
                bsize = random.randint(2, 5)
                if len(neighbor) > bsize:
                    start_idx = random.randint(0, len(neighbor) - bsize)
                    block = neighbor[start_idx : start_idx + bsize]
                    del neighbor[start_idx : start_idx + bsize]
                    insert_idx = random.randint(0, len(neighbor))
                    neighbor[insert_idx:insert_idx] = block
                else:
                    continue
            else:
                # Swap
                idx1, idx2 = random.sample(range(len(neighbor)), 2)
                neighbor[idx1], neighbor[idx2] = neighbor[idx2], neighbor[idx1]

            new_c = workload.get_opt_seq_cost(neighbor)

            # LAHC Acceptance Logic
            v = k % history_len
            if new_c <= current_c or new_c <= history[v]:
                current_sched = neighbor
                current_c = new_c
                if current_c < best_c:
                    best_c = current_c
                    best_s = list(current_sched)

            history[v] = current_c

        return best_c, best_s

    # 4. Refinement Phase (Sprint & Marathon)

    # Phase 4a: Sprint (Short optimization on top candidates)
    refined_candidates = []
    for cost, seq in unique_candidates:
        imp_cost, imp_seq = run_lahc(seq, cost, 300)
        refined_candidates.append((imp_cost, imp_seq))

    # Phase 4b: Marathon (Deep optimization on the winner)
    refined_candidates.sort(key=lambda x: x[0])
    winner_cost, winner_seq = refined_candidates[0]

    # Use remaining computational budget
    final_cost, final_seq = run_lahc(winner_seq, winner_cost, 2200)

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