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
    Optimal scheduler using Adaptive Diversity Greedy, Multi-Scale LAHC, and Cooling Convergence.

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

    candidates_pool = []

    # 2. Diversity-Driven Greedy Construction
    for i in range(num_seqs):
        # Adaptive Parameters
        if i == 0:
            # Trusted Baseline
            alpha = 0.4
            threshold_ratio = 0.90
            stochastic_prob = 0.0
        else:
            # Exploratory Parameters
            alpha = random.uniform(0.2, 0.6)
            threshold_ratio = random.uniform(0.85, 0.95)
            stochastic_prob = 0.20  # 20% chance to pick sub-optimal neighbor for structure diversity

        # Initialization
        start_txn = random.randint(0, workload.num_txns - 1)
        schedule = [start_txn]
        remaining = set(range(workload.num_txns))
        remaining.remove(start_txn)

        while remaining:
            # Dynamic Threshold Calculation
            max_dur = 0
            for t in sorted_txns_by_len:
                if t in remaining:
                    max_dur = txn_durations[t]
                    break
            
            threshold = max_dur * threshold_ratio
            
            # Candidate Identification
            candidates = set()
            
            # A. Big Rocks (Largest remaining transactions)
            rocks_added = 0
            for t in sorted_txns_by_len:
                if t in remaining:
                    if txn_durations[t] >= threshold:
                        candidates.add(t)
                        rocks_added += 1
                        if rocks_added >= 5:
                            break
                    else:
                        break # Sorted, so no need to continue
            
            # B. Random Sampling (for diversity)
            rem_list = list(remaining)
            if len(rem_list) > 15:
                candidates.update(random.sample(rem_list, 15))
            else:
                candidates.update(rem_list)

            # C. Weighted Scoring
            scored_candidates = []
            for t in candidates:
                # Calculate cost if appended
                cost = workload.get_opt_seq_cost(schedule + [t])
                # Heuristic Score: Cost penalized by weighted duration
                score = cost - (alpha * txn_durations[t])
                # Tuple: (Score, Negative Duration for Tie-break, Txn ID)
                scored_candidates.append((score, -txn_durations[t], t))
            
            scored_candidates.sort()

            # D. Selection with Stochasticity
            selected_idx = 0
            if stochastic_prob > 0 and len(scored_candidates) > 2:
                if random.random() < stochastic_prob:
                    # Pick 2nd or 3rd best occasionally
                    selected_idx = random.randint(0, min(2, len(scored_candidates)-1))
            
            best_t = scored_candidates[selected_idx][2]
            schedule.append(best_t)
            remaining.remove(best_t)

        total_cost = workload.get_opt_seq_cost(schedule)
        candidates_pool.append((total_cost, schedule))

    # 3. Funnel Selection (Filter Unique)
    candidates_pool.sort(key=lambda x: x[0])
    
    unique_candidates = []
    seen = set()
    for cost, seq in candidates_pool:
        h = tuple(seq)
        if h not in seen:
            unique_candidates.append((cost, list(seq)))
            seen.add(h)
    
    if not unique_candidates:
        unique_candidates = [(candidates_pool[0][0], list(candidates_pool[0][1]))]

    # Helper: LAHC Optimization with Multi-Scale Mutation and Cooling
    def run_optimization(schedule, start_cost, budget, enable_kick=False, cooling=False):
        current_seq = list(schedule)
        current_cost = start_cost
        best_seq = list(schedule)
        best_cost = start_cost

        history_len = 50
        history = [start_cost] * history_len
        last_imp_idx = 0

        for k in range(budget):
            # Kick Mechanism: Perturb on stagnation
            if enable_kick and (k - last_imp_idx > 500):
                # Shuffle a random segment of moderate size
                seg_len = random.randint(10, 25)
                start_k = random.randint(0, max(0, len(current_seq) - seg_len))
                segment = current_seq[start_k : start_k + seg_len]
                random.shuffle(segment)
                current_seq[start_k : start_k + seg_len] = segment
                
                # Re-evaluate and reset history
                current_cost = workload.get_opt_seq_cost(current_seq)
                history = [current_cost] * history_len
                last_imp_idx = k
                continue

            # Multi-Scale Mutation
            neighbor = list(current_seq)
            op_rand = random.random()
            n = len(neighbor)

            if op_rand < 0.45:
                # Single Insert
                i, j = random.sample(range(n), 2)
                item = neighbor.pop(i)
                neighbor.insert(j, item)
            
            elif op_rand < 0.75:
                # Micro-Block Insert (Local polishing: size 2-4)
                bs = random.randint(2, 4)
                if n > bs:
                    i = random.randint(0, n - bs)
                    blk = neighbor[i : i+bs]
                    del neighbor[i : i+bs]
                    j = random.randint(0, len(neighbor))
                    neighbor[j : j] = blk
            
            elif op_rand < 0.90:
                # Macro-Block Insert (Chain relocation: size 5-8)
                bs = random.randint(5, 8)
                if n > bs:
                    i = random.randint(0, n - bs)
                    blk = neighbor[i : i+bs]
                    del neighbor[i : i+bs]
                    j = random.randint(0, len(neighbor))
                    neighbor[j : j] = blk
            
            else:
                # Swap
                i, j = random.sample(range(n), 2)
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]

            new_cost = workload.get_opt_seq_cost(neighbor)

            # Acceptance Logic
            accept = False
            
            # Cooling Phase: In last 20%, strictly accept only improvements (Hill Climbing)
            if cooling and k > budget * 0.8:
                if new_cost < current_cost:
                    accept = True
            else:
                # LAHC Phase
                v = k % history_len
                if new_cost <= current_cost or new_cost <= history[v]:
                    accept = True

            if accept:
                current_seq = neighbor
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_seq = list(current_seq)
                    last_imp_idx = k
            
            # Update history only if not in Cooling phase (or update with current to flush history)
            if not (cooling and k > budget * 0.8):
                history[k % history_len] = current_cost

        return best_cost, best_seq

    # 4. Funnel Execution
    
    # Sprint Phase: Top 4 candidates, medium budget
    sprint_candidates = unique_candidates[:4]
    sprint_results = []
    
    for c, s in sprint_candidates:
        sc, ss = run_optimization(s, c, 400, enable_kick=False, cooling=False)
        sprint_results.append((sc, ss))
        
    sprint_results.sort(key=lambda x: x[0])
    
    # Marathon Phase: Winner, large budget with Kick and Cooling
    winner_cost, winner_seq = sprint_results[0]
    final_cost, final_seq = run_optimization(
        winner_seq, 
        winner_cost, 
        3000, 
        enable_kick=True, 
        cooling=True
    )

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