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
    Get optimal schedule using Stratified Greedy Construction and Funnel LAHC Refinement.

    Strategies:
    1. Stratified Greedy: Iteration 0 uses tuned parameters (alpha=0.4). Subsequent iterations
       randomize alpha and thresholds to generate diverse starting points.
    2. Funnel Refinement: 3-stage optimization process (Filter -> Sprint -> Marathon) to 
       gradually narrow down candidates while allocating compute budget efficiently.
    3. Enhanced LAHC: Uses expanded Block Insert (2-8 items) and stagnation "Kicks" 
       to navigate the search space better.
    """
    # 1. Pre-calculation
    # Compute transaction durations for heuristic guidance
    txn_durations = {t: workload.get_opt_seq_cost([t]) for t in range(workload.num_txns)}
    # Sort transactions by duration (descending) for Big Rocks lookup
    sorted_txns_by_len = sorted(range(workload.num_txns), key=lambda t: txn_durations[t], reverse=True)

    # Helper: Late Acceptance Hill Climbing (LAHC)
    def run_lahc(schedule, start_cost, budget, enable_kick=False):
        current_sched = list(schedule)
        current_cost = start_cost
        best_sched = list(schedule)
        best_cost = start_cost

        history_len = 50
        history = [start_cost] * history_len
        
        # Stagnation tracking for Kicks
        steps_since_improvement = 0

        for k in range(budget):
            # Stagnation Kick Logic (Marathon Phase Only)
            if enable_kick and steps_since_improvement > 500:
                # Perturbation: Shuffle a random segment
                seg_len = random.randint(10, 20)
                start_k = random.randint(0, max(0, len(current_sched) - seg_len))
                segment = current_sched[start_k : start_k + seg_len]
                random.shuffle(segment)
                current_sched[start_k : start_k + seg_len] = segment
                
                # Reset state
                current_cost = workload.get_opt_seq_cost(current_sched)
                history = [current_cost] * history_len
                steps_since_improvement = 0
                continue

            op_rand = random.random()
            neighbor = list(current_sched)

            # Operators: 
            # 50% Insert (Standard)
            # 40% Block Insert (Expanded size 2-8)
            # 10% Swap
            if op_rand < 0.50:
                # Single Insert
                idx1 = random.randint(0, len(neighbor) - 1)
                idx2 = random.randint(0, len(neighbor) - 1)
                if idx1 != idx2:
                    item = neighbor.pop(idx1)
                    neighbor.insert(idx2, item)

            elif op_rand < 0.90:
                # Block Insert (Move contiguous block of 2-8 items)
                # Larger blocks preserve dependency chains better
                block_size = random.randint(2, 8)
                if len(neighbor) > block_size:
                    start_idx = random.randint(0, len(neighbor) - block_size)
                    block = neighbor[start_idx : start_idx + block_size]
                    del neighbor[start_idx : start_idx + block_size]
                    insert_idx = random.randint(0, len(neighbor))
                    neighbor[insert_idx:insert_idx] = block
                else:
                    continue
            else:
                # Swap
                idx1, idx2 = random.sample(range(len(neighbor)), 2)
                neighbor[idx1], neighbor[idx2] = neighbor[idx2], neighbor[idx1]

            new_cost = workload.get_opt_seq_cost(neighbor)

            # LAHC Acceptance Logic
            v = k % history_len
            if new_cost <= current_cost or new_cost <= history[v]:
                current_sched = neighbor
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_sched = list(current_sched)
                    steps_since_improvement = 0
            
            history[v] = current_cost
            steps_since_improvement += 1

        return best_cost, best_sched

    candidates_pool = []

    # 2. Greedy Construction Phase
    for i in range(num_seqs):
        # Stratified Parameters
        if i == 0:
            # Baseline: Historical best parameters
            alpha = 0.4
            threshold_ratio = 0.90
            start_random = False
        else:
            # Diversity: Random parameters
            alpha = random.uniform(0.3, 0.6)
            threshold_ratio = random.uniform(0.80, 0.95)
            start_random = True

        if start_random:
            start_txn = random.randint(0, workload.num_txns - 1)
        else:
            # Start with the largest rock
            start_txn = sorted_txns_by_len[0]

        txn_seq = [start_txn]
        remaining_txns = set(range(workload.num_txns))
        remaining_txns.remove(start_txn)

        while remaining_txns:
            candidates = set()

            if i == 0 and len(txn_seq) < 2:
                # For the very first sequence, do a broad scan initially
                candidates = remaining_txns
            else:
                # Dynamic Big Rocks Heuristic
                max_dur = 0
                for t in sorted_txns_by_len:
                    if t in remaining_txns:
                        max_dur = txn_durations[t]
                        break

                threshold = max_dur * threshold_ratio
                rocks_added = 0
                
                # Add Big Rocks
                for t in sorted_txns_by_len:
                    if t in remaining_txns:
                        if txn_durations[t] >= threshold:
                            candidates.add(t)
                            rocks_added += 1
                            if rocks_added >= 6:
                                break
                        else:
                            break

                # Add Random Samples
                if len(remaining_txns) > 20:
                    candidates.update(random.sample(list(remaining_txns), 15))
                else:
                    candidates.update(remaining_txns)

            # Weighted Selection
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

    # 3. Funnel Refinement Process
    
    # Stage 1: Filter (Short optimization on all candidates)
    # Allows "rough" candidates to smooth out before judging them
    stage1_candidates = []
    for cost, seq in candidates_pool:
        # Budget 50 is small but enough to fix trivial local flaws
        c, s = run_lahc(seq, cost, 50, enable_kick=False)
        stage1_candidates.append((c, s))
        
    # Selection: Keep top 3 unique
    stage1_candidates.sort(key=lambda x: x[0])
    unique_candidates = []
    seen_hashes = set()
    for cost, seq in stage1_candidates:
        h = tuple(seq)
        if h not in seen_hashes:
            unique_candidates.append((cost, list(seq)))
            seen_hashes.add(h)
        if len(unique_candidates) >= 3:
            break
            
    if not unique_candidates:
        unique_candidates = [stage1_candidates[0]]

    # Stage 2: Sprint (Medium optimization on top 3)
    stage2_candidates = []
    for cost, seq in unique_candidates:
        c, s = run_lahc(seq, cost, 300, enable_kick=False)
        stage2_candidates.append((c, s))

    # Stage 3: Marathon (Deep optimization on the winner)
    stage2_candidates.sort(key=lambda x: x[0])
    winner_cost, winner_seq = stage2_candidates[0]

    final_cost, final_seq = run_lahc(winner_seq, winner_cost, 2000, enable_kick=True)

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