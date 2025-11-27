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
    Get optimal schedule using Diverse Weighted Greedy Construction, Funnel Refinement, and Enhanced LAHC.

    Strategies:
    1. Diversity-Driven Greedy: Randomizes alpha and Big Rocks threshold.
    2. Funnel Refinement: 3-stage process (Warmup -> Sprint -> Marathon).
    3. LAHC with Enhanced Operators: Includes larger block moves and stagnation kicks.
    """
    # 1. Pre-calculation
    # Compute transaction durations for heuristic guidance
    txn_durations = {t: workload.get_opt_seq_cost([t]) for t in range(workload.num_txns)}
    # Sort transactions by duration (descending) for Big Rocks lookup
    sorted_txns_by_len = sorted(range(workload.num_txns), key=lambda t: txn_durations[t], reverse=True)

    candidates_pool = []

    # 2. Greedy Construction Phase
    for i in range(num_seqs):
        # Diversity: Randomize parameters for each sequence
        if i == 0:
            # Baseline parameters for first run
            alpha = 0.4
            threshold_ratio = 0.90
            rock_limit = 6
        else:
            alpha = random.uniform(0.2, 0.6)
            threshold_ratio = random.uniform(0.85, 0.98)
            rock_limit = random.randint(2, 8)

        # Random start
        start_txn = random.randint(0, workload.num_txns - 1)
        txn_seq = [start_txn]

        remaining_txns = set(range(workload.num_txns))
        remaining_txns.remove(start_txn)

        while remaining_txns:
            candidates = set()

            if i == 0:
                # Full scan for first sequence
                candidates = remaining_txns
            else:
                # Dynamic Big Rocks Heuristic
                # Find max duration in remaining set
                max_dur = 0
                for t in sorted_txns_by_len:
                    if t in remaining_txns:
                        max_dur = txn_durations[t]
                        break

                threshold = max_dur * threshold_ratio

                # Add Big Rocks (transactions >= threshold)
                rocks_added = 0
                for t in sorted_txns_by_len:
                    if t in remaining_txns:
                        if txn_durations[t] >= threshold:
                            candidates.add(t)
                            rocks_added += 1
                            if rocks_added >= rock_limit:
                                break
                        else:
                            break

                # Add Random Samples
                if len(remaining_txns) > 20:
                    candidates.update(random.sample(list(remaining_txns), 15))
                else:
                    candidates.update(remaining_txns)

            # Weighted Selection
            # Score = Cost - (Alpha * Duration)
            best_t = -1
            best_score = float('inf')

            for t in candidates:
                cost = workload.get_opt_seq_cost(txn_seq + [t])
                score = cost - (alpha * txn_durations[t])

                if score < best_score:
                    best_score = score
                    best_t = t
                elif score == best_score:
                    if txn_durations[t] > txn_durations.get(best_t, 0):
                        best_t = t

            txn_seq.append(best_t)
            remaining_txns.remove(best_t)

        total_cost = workload.get_opt_seq_cost(txn_seq)
        candidates_pool.append((total_cost, txn_seq))

    # 3. Funnel Selection Setup
    candidates_pool.sort(key=lambda x: x[0])

    unique_candidates = []
    seen_hashes = set()
    for cost, seq in candidates_pool:
        h = tuple(seq)
        if h not in seen_hashes:
            unique_candidates.append((cost, list(seq)))
            seen_hashes.add(h)
        if len(unique_candidates) >= 5:
            break

    if not unique_candidates and candidates_pool:
        unique_candidates = [(candidates_pool[0][0], list(candidates_pool[0][1]))]

    # Helper: LAHC Optimization
    def run_lahc(schedule, start_cost, budget, use_kick=False):
        current_sched = list(schedule)
        current_cost = start_cost
        best_sched = list(schedule)
        best_cost = start_cost

        # Adaptive history length based on budget
        # Short history for short budgets to force convergence
        history_len = max(5, min(100, budget // 20))
        history = [start_cost] * history_len
        last_imp_idx = 0

        for k in range(budget):
            # Stagnation Kick (only enabled in Marathon phase)
            if use_kick and (k - last_imp_idx > 500):
                # Shuffle a random segment to escape local optima
                idx = random.randint(0, max(0, len(current_sched) - 15))
                seg_len = random.randint(10, 20)
                end = min(len(current_sched), idx + seg_len)
                segment = current_sched[idx:end]
                random.shuffle(segment)
                current_sched[idx:end] = segment

                # Reset history and costs
                current_cost = workload.get_opt_seq_cost(current_sched)
                history = [current_cost] * history_len
                last_imp_idx = k
                continue

            op_rand = random.random()
            neighbor = list(current_sched)

            # Operators: 45% Insert, 45% Block Insert (2-8 items), 10% Swap
            if op_rand < 0.45:
                # Single Insert
                idx1 = random.randint(0, len(neighbor) - 1)
                idx2 = random.randint(0, len(neighbor) - 1)
                if idx1 != idx2:
                    item = neighbor.pop(idx1)
                    neighbor.insert(idx2, item)
            elif op_rand < 0.90:
                # Block Insert
                bsize = random.randint(2, 8)
                if len(neighbor) > bsize:
                    start = random.randint(0, len(neighbor) - bsize)
                    block = neighbor[start : start + bsize]
                    del neighbor[start : start + bsize]
                    dest = random.randint(0, len(neighbor))
                    neighbor[dest : dest] = block
                else:
                    continue
            else:
                # Swap
                idx1, idx2 = random.sample(range(len(neighbor)), 2)
                neighbor[idx1], neighbor[idx2] = neighbor[idx2], neighbor[idx1]

            new_cost = workload.get_opt_seq_cost(neighbor)

            # LAHC Acceptance
            v = k % history_len
            if new_cost <= current_cost or new_cost <= history[v]:
                current_sched = neighbor
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_sched = list(current_sched)
                    last_imp_idx = k

            history[v] = current_cost

        return best_cost, best_sched

    # 4. Funnel Execution

    # Stage 1: Warmup (Top 5 -> Top 4) - Quick stabilization
    stage1_results = []
    for cost, seq in unique_candidates:
        c, s = run_lahc(seq, cost, 50, use_kick=False)
        stage1_results.append((c, s))

    stage1_results.sort(key=lambda x: x[0])
    # Keep top 4 instead of 3 to broaden Sprint phase
    top_sprint = stage1_results[:4]

    # Stage 2: Sprint (Top 4 -> Top 1) - Moderate optimization
    stage2_results = []
    for cost, seq in top_sprint:
        c, s = run_lahc(seq, cost, 300, use_kick=False)
        stage2_results.append((c, s))

    stage2_results.sort(key=lambda x: x[0])
    winner_cost, winner_seq = stage2_results[0]

    # Stage 3: Marathon (Winner) - Deep optimization with kicks
    final_cost, final_seq = run_lahc(winner_seq, winner_cost, 2500, use_kick=True)

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