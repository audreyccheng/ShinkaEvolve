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
    Get optimal schedule using Adaptive LAHC with Funnel Refinement.

    Strategies:
    1. Diversity-Driven Greedy: Randomized parameters for initial population.
    2. Adaptive Operators: Probabilities shift from Block Moves to Inserts over time.
    3. Dynamic History: LAHC history length shrinks to force convergence.
    4. Block Reversal: Specialized operator for dependency chains.
    """
    # 1. Pre-calculation
    txn_durations = {t: workload.get_opt_seq_cost([t]) for t in range(workload.num_txns)}
    sorted_txns_by_len = sorted(range(workload.num_txns), key=lambda t: txn_durations[t], reverse=True)

    candidates_pool = []

    # 2. Greedy Construction Phase with Diversity
    for i in range(num_seqs):
        # Randomize parameters to create diverse starting points
        if i == 0:
            alpha = 0.5
            rock_threshold_ratio = 0.90
            rock_limit = 5
        else:
            alpha = random.uniform(0.2, 0.7)
            rock_threshold_ratio = random.uniform(0.80, 0.98)
            rock_limit = random.randint(3, 8)

        # Start with random transaction
        start_txn = random.randint(0, workload.num_txns - 1)
        schedule = [start_txn]
        remaining = set(range(workload.num_txns))
        remaining.remove(start_txn)

        while remaining:
            candidates = set()

            # "Big Rocks" Strategy
            # Identify high duration transactions available
            max_remaining_dur = 0
            for t in sorted_txns_by_len:
                if t in remaining:
                    max_remaining_dur = txn_durations[t]
                    break
            
            threshold = max_remaining_dur * rock_threshold_ratio
            
            rocks_added = 0
            for t in sorted_txns_by_len:
                if t in remaining:
                    if txn_durations[t] >= threshold:
                        candidates.add(t)
                        rocks_added += 1
                        if rocks_added >= rock_limit:
                            break
                    else:
                        break # Optimization: sorted list

            # Random sampling filler
            remaining_list = list(remaining)
            if len(remaining_list) > 15:
                candidates.update(random.sample(remaining_list, 15))
            else:
                candidates.update(remaining)

            # Weighted Scoring
            best_t = -1
            best_score = float('inf')

            for t in candidates:
                cost = workload.get_opt_seq_cost(schedule + [t])
                # Penalty for duration is weighted by alpha
                # We want to minimize cost but favor long durations early (high negative score)
                score = cost - (alpha * txn_durations[t])

                if score < best_score:
                    best_score = score
                    best_t = t
                elif score == best_score:
                    if txn_durations[t] > txn_durations.get(best_t, 0):
                        best_t = t
            
            schedule.append(best_t)
            remaining.remove(best_t)

        cost = workload.get_opt_seq_cost(schedule)
        candidates_pool.append((cost, schedule))

    # 3. Deduplicate Candidates
    candidates_pool.sort(key=lambda x: x[0])
    unique_candidates = []
    seen = set()
    for c, s in candidates_pool:
        h = tuple(s)
        if h not in seen:
            unique_candidates.append((c, list(s)))
            seen.add(h)
            
    if not unique_candidates:
        unique_candidates = [(candidates_pool[0][0], list(candidates_pool[0][1]))]

    # 4. Adaptive LAHC Engine
    def run_adaptive_lahc(seq, start_cost, budget, stage='standard'):
        current_seq = list(seq)
        current_cost = start_cost
        best_seq = list(seq)
        best_cost = start_cost
        
        # Configuration based on stage
        if stage == 'marathon':
            base_history_len = 100
            enable_kick = True
            dynamic_history = True
        else:
            base_history_len = 30
            enable_kick = False
            dynamic_history = False
            
        history = [start_cost] * base_history_len
        last_imp_k = 0
        
        for k in range(budget):
            # Dynamic History Compression (Simulated Annealing effect)
            if dynamic_history:
                progress = k / budget
                # Decay from base_len down to 10
                current_h_len = max(10, int(base_history_len * (1.0 - 0.9 * progress)))
            else:
                current_h_len = base_history_len

            # Stagnation Kick
            if enable_kick and (k - last_imp_k > 400):
                # Shuffle a random segment
                slen = random.randint(15, 30)
                if len(current_seq) > slen:
                    sidx = random.randint(0, len(current_seq) - slen)
                    sub = current_seq[sidx:sidx+slen]
                    random.shuffle(sub)
                    current_seq[sidx:sidx+slen] = sub
                    current_cost = workload.get_opt_seq_cost(current_seq)
                    # Reset history to absorb the kick
                    history = [current_cost] * base_history_len
                    last_imp_k = k
                    continue

            # Adaptive Operator Selection
            progress = k / budget
            # Block moves start high probability, decay to 0
            # Inserts start lower, increase to 1
            p_block = 0.5 * (1.0 - progress) 
            p_reverse = 0.05
            p_swap = 0.1
            # p_insert is the remainder
            
            op_rand = random.random()
            neighbor = list(current_seq)
            n = len(neighbor)
            
            if op_rand < p_block:
                # Block Move (2 to 8 items)
                bsize = random.randint(2, 8)
                if n > bsize:
                    s = random.randint(0, n - bsize)
                    block = neighbor[s:s+bsize]
                    del neighbor[s:s+bsize]
                    d = random.randint(0, len(neighbor))
                    neighbor[d:d] = block
            
            elif op_rand < p_block + p_reverse:
                # Block Reverse (New Operator)
                # Helps untangle reverse dependency chains
                bsize = random.randint(3, 12)
                if n > bsize:
                    s = random.randint(0, n - bsize)
                    neighbor[s:s+bsize] = neighbor[s:s+bsize][::-1]
                    
            elif op_rand < p_block + p_reverse + p_swap:
                # Swap
                idx1, idx2 = random.sample(range(n), 2)
                neighbor[idx1], neighbor[idx2] = neighbor[idx2], neighbor[idx1]
                
            else:
                # Single Insert (Fine tuning)
                idx1 = random.randint(0, n - 1)
                idx2 = random.randint(0, n - 1)
                if idx1 != idx2:
                    item = neighbor.pop(idx1)
                    neighbor.insert(idx2, item)

            new_cost = workload.get_opt_seq_cost(neighbor)
            
            # LAHC Acceptance Logic
            # Use current_h_len for dynamic window
            v = k % current_h_len
            if new_cost <= current_cost or new_cost <= history[v]:
                current_seq = neighbor
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_seq = list(current_seq)
                    last_imp_k = k
            
            history[v] = current_cost
            
        return best_cost, best_seq

    # 5. Funnel Execution
    
    # Stage 1: Filter (All unique candidates) - Quick scan
    stage1_results = []
    for c, s in unique_candidates:
        rc, rs = run_adaptive_lahc(s, c, 50, stage='filter')
        stage1_results.append((rc, rs))
    
    stage1_results.sort(key=lambda x: x[0])
    
    # Stage 2: Sprint (Top 4) - Medium effort
    top_sprint = stage1_results[:4]
    stage2_results = []
    for c, s in top_sprint:
        rc, rs = run_adaptive_lahc(s, c, 400, stage='sprint')
        stage2_results.append((rc, rs))
        
    stage2_results.sort(key=lambda x: x[0])
    
    # Stage 3: Marathon (Winner) - Deep optimization
    winner_cost, winner_seq = stage2_results[0]
    final_cost, final_seq = run_adaptive_lahc(winner_seq, winner_cost, 2500, stage='marathon')
    
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