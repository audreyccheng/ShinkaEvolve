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
    Get optimal schedule using Decaying Alpha Greedy construction and 
    refined LAHC with Multi-Scale Block moves and Final Descent.
    
    Key Improvements:
    1. Decaying Alpha: Prioritizes long transactions early (Big Rocks) and reduces bias later.
    2. Multi-Scale Operators: Micro-blocks for local fixes, Macro-blocks for structural changes.
    3. Final Descent: Switches from LAHC to Hill Climbing at the end to converge.
    4. Stagnation Kicks: Randomizes segments if stuck.
    """
    
    # --- 1. Pre-calculation ---
    # Compute duration for each transaction to identify "Big Rocks"
    txn_durations = {t: workload.get_opt_seq_cost([t]) for t in range(workload.num_txns)}
    # Sort for quick access to longest transactions
    sorted_txns = sorted(range(workload.num_txns), key=lambda t: txn_durations[t], reverse=True)
    
    candidates = []

    # --- 2. Greedy Construction Phase ---
    for i in range(num_seqs):
        # Stochastic Parameters for Diversity
        if i == 0:
            # Baseline configuration
            start_alpha = 0.6
            end_alpha = 0.1
            rock_threshold_ratio = 0.90
            start_txn = sorted_txns[0]
        else:
            # Randomized configuration
            start_alpha = random.uniform(0.5, 0.9) # Stronger bias for rocks
            end_alpha = random.uniform(0.0, 0.2)   # Neutral ending
            rock_threshold_ratio = random.uniform(0.80, 0.95)
            # Probabilistic start: usually a big rock, sometimes random
            if random.random() < 0.7:
                start_txn = sorted_txns[random.randint(0, min(5, workload.num_txns-1))]
            else:
                start_txn = random.randint(0, workload.num_txns - 1)

        schedule = [start_txn]
        remaining = set(range(workload.num_txns))
        remaining.remove(start_txn)

        while remaining:
            # Dynamic Threshold for "Big Rocks" in the remaining set
            # To avoid iterating full list, we just check top sorted until we find one in remaining
            max_dur = 0
            for t in sorted_txns:
                if t in remaining:
                    max_dur = txn_durations[t]
                    break
            
            threshold = max_dur * rock_threshold_ratio
            
            # Select Candidates
            candidate_pool = set()
            rocks_found = 0
            
            # Add Big Rocks
            for t in sorted_txns:
                if t in remaining:
                    if txn_durations[t] >= threshold:
                        candidate_pool.add(t)
                        rocks_found += 1
                        if rocks_found >= 6: # Heuristic limit
                            break
                    else:
                        break # Sorted list, none after this will match
            
            # Add Random Samples to fill pool
            # Ensure we have enough candidates for comparison
            target_size = 15
            if len(remaining) <= target_size:
                candidate_pool.update(remaining)
            else:
                needed = target_size - len(candidate_pool)
                if needed > 0:
                    candidate_pool.update(random.sample(list(remaining), needed))
            
            # Calculate Decaying Alpha
            # Linear decay based on progress: High alpha start -> Low alpha end
            progress = len(schedule) / workload.num_txns
            current_alpha = start_alpha - (progress * (start_alpha - end_alpha))
            
            # Select Best Candidate
            best_t = -1
            best_score = float('inf')
            
            for t in candidate_pool:
                # Calculate MakeSpan if we append t
                cost = workload.get_opt_seq_cost(schedule + [t])
                
                # Weighted Score: Cost - (Alpha * Duration)
                # Higher Alpha -> We forgive higher Cost if Duration is long
                score = cost - (current_alpha * txn_durations[t])
                
                if score < best_score:
                    best_score = score
                    best_t = t
                elif score == best_score:
                    # Tie-break: Longest first
                    if txn_durations[t] > txn_durations.get(best_t, 0):
                        best_t = t
            
            schedule.append(best_t)
            remaining.remove(best_t)
            
        initial_cost = workload.get_opt_seq_cost(schedule)
        candidates.append((initial_cost, schedule))

    # --- 3. Refinement Phase (Optimizer) ---
    
    def optimize_schedule(initial_seq, start_cost, budget, enable_kick=False):
        current_seq = list(initial_seq)
        current_cost = start_cost
        best_seq = list(initial_seq)
        best_cost = start_cost
        
        history_len = 50
        history = [start_cost] * history_len
        last_improvement_iter = 0
        
        for k in range(budget):
            # Final Descent: Last 20% of budget is pure Hill Climbing
            is_final_descent = k > (budget * 0.8)
            
            # Stagnation Kick
            # If enabled and no improvement for a long time, shuffle a segment
            if enable_kick and not is_final_descent and (k - last_improvement_iter > 400):
                # Pick a random segment
                seg_len = random.randint(10, 25)
                if len(current_seq) > seg_len:
                    idx = random.randint(0, len(current_seq) - seg_len)
                    segment = current_seq[idx : idx + seg_len]
                    random.shuffle(segment)
                    current_seq[idx : idx + seg_len] = segment
                    
                    # Reset state
                    current_cost = workload.get_opt_seq_cost(current_seq)
                    history = [current_cost] * history_len
                    last_improvement_iter = k
                    continue

            # Select Mutation Operator
            neighbor = list(current_seq)
            r = random.random()
            
            if r < 0.50:
                # Single Insert (Fine tuning)
                idx1, idx2 = random.sample(range(len(neighbor)), 2)
                item = neighbor.pop(idx1)
                neighbor.insert(idx2, item)
                
            elif r < 0.80:
                # Micro-Block Insert (Local structure move, size 2-4)
                bsize = random.randint(2, 4)
                if len(neighbor) > bsize:
                    start = random.randint(0, len(neighbor) - bsize)
                    block = neighbor[start : start + bsize]
                    del neighbor[start : start + bsize]
                    dest = random.randint(0, len(neighbor))
                    neighbor[dest : dest] = block
                else:
                    continue
                    
            elif r < 0.95:
                # Macro-Block Insert (Structural change, size 5-10)
                # Helps move large dependency chains
                bsize = random.randint(5, 10)
                if len(neighbor) > bsize:
                    start = random.randint(0, len(neighbor) - bsize)
                    block = neighbor[start : start + bsize]
                    del neighbor[start : start + bsize]
                    dest = random.randint(0, len(neighbor))
                    neighbor[dest : dest] = block
                else:
                    continue
            else:
                # Swap (Exploration)
                idx1, idx2 = random.sample(range(len(neighbor)), 2)
                neighbor[idx1], neighbor[idx2] = neighbor[idx2], neighbor[idx1]

            new_cost = workload.get_opt_seq_cost(neighbor)
            
            # Acceptance Logic
            accepted = False
            if is_final_descent:
                # Strict Hill Climbing
                if new_cost < current_cost:
                    accepted = True
            else:
                # Late Acceptance Hill Climbing
                v = k % history_len
                if new_cost <= current_cost or new_cost <= history[v]:
                    accepted = True
            
            if accepted:
                current_seq = neighbor
                current_cost = new_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_seq = list(current_seq)
                    last_improvement_iter = k
            
            if not is_final_descent:
                history[k % history_len] = current_cost
                
        return best_cost, best_seq

    # --- 4. Pipeline Execution ---
    
    # Sort candidates by greedy cost
    candidates.sort(key=lambda x: x[0])
    
    # Filter for uniqueness (Top 3 unique)
    unique_candidates = []
    seen_hashes = set()
    for c, s in candidates:
        h = tuple(s)
        if h not in seen_hashes:
            unique_candidates.append((c, s))
            seen_hashes.add(h)
        if len(unique_candidates) >= 3:
            break
            
    # Fallback
    if not unique_candidates and candidates:
        unique_candidates = [candidates[0]]

    # Sprint Phase: Quick optimization on top candidates
    sprint_results = []
    for c, s in unique_candidates:
        sc, ss = optimize_schedule(s, c, 300, enable_kick=False)
        sprint_results.append((sc, ss))
        
    sprint_results.sort(key=lambda x: x[0])
    winner_cost, winner_seq = sprint_results[0]
    
    # Marathon Phase: Deep optimization on the winner
    final_cost, final_seq = optimize_schedule(winner_seq, winner_cost, 2500, enable_kick=True)

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