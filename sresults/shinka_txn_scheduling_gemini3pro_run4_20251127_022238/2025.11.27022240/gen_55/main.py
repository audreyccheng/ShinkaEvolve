# EVOLVE-BLOCK-START
"""Transaction scheduling algorithm for optimizing makespan across multiple workloads"""

import time
import random
import sys
import os
import re
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
    Optimized Scheduler: Diverse Adaptive Greedy + Multi-Stage LAHC.
    
    Combines:
    1.  **Diverse Greedy Construction**: Varies heuristic parameters (threshold, alpha) per candidate 
        to explore different structural packings.
    2.  **Hybrid Pool Selection**: Combines 'Big Rocks' (threshold-based) with 'Top Absolute Longest' 
        (sorted-list based) to ensure critical transactions are prioritized.
    3.  **Multi-Stage Refinement**: Uses a Sprint (short parallel search) followed by a 
        Marathon (deep single search) using LAHC with shift and block-move operators.
    """
    
    # --- Pre-computation ---
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

    # Sorted list for absolute longest lookups
    sorted_txns = sorted(range(workload.num_txns), key=lambda i: txn_lens[i], reverse=True)

    # Configuration
    SAMPLE_SIZE = 12
    
    candidates = []

    # --- Phase 1: Diverse Adaptive Greedy ---
    for _ in range(num_seqs):
        # Diversity: Randomize parameters to widen search cone
        # Threshold: How close to max_len must a txn be to be a "Big Rock"?
        # Alpha: How much weight to give length vs cost?
        run_threshold = random.uniform(0.85, 0.98)
        run_alpha = random.uniform(0.0, 0.15)
        
        remaining = set(range(workload.num_txns))
        
        # Random start to break symmetry
        start_txn = random.choice(list(remaining))
        current_seq = [start_txn]
        remaining.remove(start_txn)

        while remaining:
            pool = []
            
            # 1. Dynamic Threshold Big Rocks
            # Get max remaining length
            rem_lens = [txn_lens[t] for t in remaining]
            max_rem_len = max(rem_lens) if rem_lens else 0
            len_threshold = max_rem_len * run_threshold
            
            # Add items meeting threshold
            threshold_rocks = [t for t in remaining if txn_lens[t] >= len_threshold]
            pool.extend(threshold_rocks)

            # 2. Absolute Longest Safety Net (from Inspiration)
            # Ensures we don't miss the absolute longest if threshold logic is jittery
            added_safety = 0
            for t in sorted_txns:
                if t in remaining:
                    pool.append(t)
                    added_safety += 1
                    if added_safety >= 3: # Top 3 available
                        break
            
            # 3. Random Fill
            pool = list(set(pool))
            needed = SAMPLE_SIZE - len(pool)
            if needed > 0:
                others = [t for t in remaining if t not in pool]
                if len(others) <= needed:
                    pool.extend(others)
                else:
                    pool.extend(random.sample(others, needed))
            
            # Deduplicate final pool
            pool = list(set(pool))
            
            # Selection
            best_cand = -1
            best_score = float('inf')
            
            for t in pool:
                cost = workload.get_opt_seq_cost(current_seq + [t])
                # Weighted score
                score = cost - (run_alpha * txn_lens[t])
                
                if score < best_score:
                    best_score = score
                    best_cand = t
            
            current_seq.append(best_cand)
            remaining.remove(best_cand)

        # Polish: Adjacent Swap Descent
        current_cost = workload.get_opt_seq_cost(current_seq)
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
                    current_seq[i], current_seq[i+1] = current_seq[i+1], current_seq[i]

        candidates.append((current_cost, current_seq))

    # --- Phase 2: Sprint (Multi-Candidate Refinement) ---
    candidates.sort(key=lambda x: x[0])
    
    # Select distinct candidates
    sprint_candidates = []
    seen_costs = set()
    for cost, seq in candidates:
        if cost not in seen_costs:
            sprint_candidates.append((cost, list(seq)))
            seen_costs.add(cost)
        if len(sprint_candidates) >= 3:
            break
            
    if not sprint_candidates:
        sprint_candidates = [candidates[0]]

    # LAHC Engine
    def run_lahc(start_seq, start_cost, iterations, history_len=50):
        curr_s = list(start_seq)
        curr_c = start_cost
        best_s = list(start_seq)
        best_c = start_cost
        
        history = [curr_c] * history_len
        
        for i in range(iterations):
            # Mutate
            neigh_s = list(curr_s)
            slen = len(neigh_s)
            r = random.random()
            
            if r < 0.5:
                # Single Shift (50%)
                if slen < 2: continue
                f, t = random.randint(0, slen-1), random.randint(0, slen-1)
                if f == t: continue
                item = neigh_s.pop(f)
                neigh_s.insert(t, item)
            elif r < 0.8:
                # Block Shift (30%) - Tuned size 2-5
                if slen < 6: continue
                bsize = random.randint(2, 5)
                f = random.randint(0, slen - bsize)
                block = neigh_s[f : f+bsize]
                del neigh_s[f : f+bsize]
                t = random.randint(0, len(neigh_s))
                neigh_s[t:t] = block
            else:
                # Swap (20%)
                if slen < 2: continue
                idx = random.randint(0, slen - 2)
                neigh_s[idx], neigh_s[idx+1] = neigh_s[idx+1], neigh_s[idx]
            
            neigh_c = workload.get_opt_seq_cost(neigh_s)
            
            h_idx = i % history_len
            if neigh_c <= curr_c or neigh_c <= history[h_idx]:
                curr_s = neigh_s
                curr_c = neigh_c
                if curr_c < best_c:
                    best_c = curr_c
                    best_s = list(curr_s)
            
            history[h_idx] = curr_c
            
        return best_c, best_s

    # Run Sprint
    sprint_results = []
    SPRINT_ITERS = 300 # Slightly increased from 250
    for cost, seq in sprint_candidates:
        res = run_lahc(seq, cost, SPRINT_ITERS)
        sprint_results.append(res)
        
    # --- Phase 3: Marathon (Deep Refinement) ---
    sprint_results.sort(key=lambda x: x[0])
    champion_cost, champion_seq = sprint_results[0]
    
    MARATHON_ITERS = 2500
    final_cost, final_seq = run_lahc(champion_seq, champion_cost, MARATHON_ITERS)

    return final_cost, final_seq


def get_random_costs():
    """
    Evaluate scheduling algorithm on three different workloads.

    Returns:
        Tuple of (total_makespan, list_of_schedules, execution_time)
    """
    start_time = time.time()
    NUM_SEQS = 10

    workload = Workload(WORKLOAD_1)
    makespan1, schedule1 = get_best_schedule(workload, NUM_SEQS)
    cost1 = workload.get_opt_seq_cost(schedule1)

    workload2 = Workload(WORKLOAD_2)
    makespan2, schedule2 = get_best_schedule(workload2, NUM_SEQS)
    cost2 = workload2.get_opt_seq_cost(schedule2)

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