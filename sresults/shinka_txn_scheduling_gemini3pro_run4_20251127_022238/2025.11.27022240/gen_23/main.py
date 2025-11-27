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
    Get optimal schedule using 'Big Rock' Greedy Construction and Late Acceptance Hill Climbing.
    
    Algorithm Design:
    1.  **Metric Analysis**: Pre-compute transaction operation counts. Sort transactions by length
        to identify "Big Rocks" (long transactions that limit concurrency).
    2.  **Greedy Phase (Population Generation)**:
        -   Construct `num_seqs` schedules.
        -   Selection Logic: At each step, sample `SAMPLE_BIG` of the longest remaining transactions
            and `SAMPLE_RAND` random ones.
        -   Cost Function: Minimize `(makespan, -length)`. This places long transactions as early
            as possible without increasing makespan (Best Fit Descending).
        -   **Immediate Polish**: Apply Adjacent Swap Descent to reach a local optimum immediately.
    3.  **Refinement Phase (Late Acceptance Hill Climbing)**:
        -   Apply LAHC to the best greedy candidate. LAHC accepts worsening moves if they are
            better than the state `L` iterations ago, allowing strictly controlled exploration
            without temperature parameters.
        -   **Operators**:
            -   Single Insertion (Shift): Relocates a transaction.
            -   **Block Insertion**: Moves a contiguous block (size 2-3) to a new location.
                This preserves optimized dependency chains while fixing global ordering.
            -   Swap: Minor local adjustments.

    Args:
        workload: Workload object
        num_seqs: Number of greedy start points

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
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
            
    # Sorted list of all transactions by length (descending)
    all_txns_by_len = sorted(range(workload.num_txns), key=lambda k: txn_lens[k], reverse=True)

    candidates = []
    
    # Constants
    SAMPLE_BIG = 3      # Ensure top 3 longest available are checked
    SAMPLE_RAND = 8     # Check 8 random others for holes
    LAHC_L = 50         # History length
    LAHC_ITERS = 2500   # Iteration budget
    
    # --- Phase 1: Diverse Greedy Construction ---
    for _ in range(num_seqs):
        remaining = set(range(workload.num_txns))
        
        # Random start to ensure diversity
        start_txn = random.choice(list(remaining))
        current_seq = [start_txn]
        remaining.remove(start_txn)
        
        while remaining:
            step_candidates = set()
            
            # 1. Add 'Big Rocks' (longest remaining)
            added = 0
            for t in all_txns_by_len:
                if t in remaining:
                    step_candidates.add(t)
                    added += 1
                    if added >= SAMPLE_BIG:
                        break
            
            # 2. Add Randoms
            if len(remaining) <= SAMPLE_RAND:
                step_candidates.update(remaining)
            else:
                # To avoid O(N) list conversion of large set every time, we can accept
                # that 'sample' might pick ones we already picked in Big Rocks, it's fine.
                # Just converting list(remaining) is fast enough for N=100.
                step_candidates.update(random.sample(list(remaining), SAMPLE_RAND))
            
            # 3. Evaluate
            best_cand = -1
            best_score = (float('inf'), 0)
            
            for t in step_candidates:
                # Heuristic: Min Cost, then Max Length (Best Fit Descending)
                cost = workload.get_opt_seq_cost(current_seq + [t])
                score = (cost, -txn_lens[t])
                
                if score < best_score:
                    best_score = score
                    best_cand = t
            
            current_seq.append(best_cand)
            remaining.remove(best_cand)
            
        # --- Phase 2: Immediate Polish (Hill Climbing) ---
        # Essential to bring the greedy guess to a local minimum
        curr_cost = workload.get_opt_seq_cost(current_seq)
        improved = True
        while improved:
            improved = False
            for i in range(len(current_seq) - 1):
                current_seq[i], current_seq[i+1] = current_seq[i+1], current_seq[i]
                new_cost = workload.get_opt_seq_cost(current_seq)
                if new_cost < curr_cost:
                    curr_cost = new_cost
                    improved = True
                else:
                    current_seq[i], current_seq[i+1] = current_seq[i+1], current_seq[i]
        
        candidates.append((curr_cost, current_seq))

    # --- Phase 3: Late Acceptance Hill Climbing (Refinement) ---
    candidates.sort(key=lambda x: x[0])
    best_cost, best_seq = candidates[0]
    
    current_seq = list(best_seq)
    current_cost = best_cost
    
    # Initialize History
    history = [current_cost] * LAHC_L
    
    for iteration in range(LAHC_ITERS):
        new_seq = list(current_seq)
        op_rand = random.random()
        
        # Operators
        if op_rand < 0.4:
            # Single Insertion (Shift)
            if len(new_seq) > 1:
                idx_from = random.randint(0, len(new_seq) - 1)
                idx_to = random.randint(0, len(new_seq) - 1)
                if idx_from != idx_to:
                    txn = new_seq.pop(idx_from)
                    new_seq.insert(idx_to, txn)
                    
        elif op_rand < 0.65:
            # Block Insertion (Move chunk of 2 or 3)
            # Preserves local dependencies while moving them globally
            if len(new_seq) > 3:
                blk_size = random.randint(2, 3)
                start_idx = random.randint(0, len(new_seq) - blk_size)
                # Extract
                block = new_seq[start_idx : start_idx + blk_size]
                del new_seq[start_idx : start_idx + blk_size]
                # Insert
                insert_idx = random.randint(0, len(new_seq))
                new_seq[insert_idx:insert_idx] = block
                
        else:
            # Swap Adjacent
            if len(new_seq) > 1:
                idx = random.randint(0, len(new_seq) - 2)
                new_seq[idx], new_seq[idx+1] = new_seq[idx+1], new_seq[idx]
        
        # Evaluate
        new_cost = workload.get_opt_seq_cost(new_seq)
        
        # LAHC Acceptance Logic
        v = iteration % LAHC_L
        if new_cost <= current_cost or new_cost <= history[v]:
            current_seq = new_seq
            current_cost = new_cost
            
            # Update Global Best
            if current_cost < best_cost:
                best_cost = current_cost
                best_seq = list(current_seq)
        
        # Update History
        # We record the cost of the *current* solution (accepted or not) into history
        history[v] = current_cost

    return best_cost, best_seq


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