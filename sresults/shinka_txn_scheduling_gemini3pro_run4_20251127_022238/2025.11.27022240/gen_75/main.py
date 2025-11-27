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
    Get optimal schedule using Adaptive Greedy and Multi-Scale LAHC with Cooling.

    Algorithm:
    1.  **Txn Analysis**: Pre-compute transaction lengths (ops count).
    2.  **Diverse Greedy Construction**:
        -   Generate `num_seqs` candidates.
        -   **Baseline & Exploration**: First candidate uses proven parameters; others use randomized 
            `alpha` (length weight) and `threshold` (pool filter) to sample different structural basins.
        -   **Adaptive Pool**: Select "Big Rocks" + Randoms.
        -   **Heuristic**: Minimize `cost - (alpha * length)`.
        -   **Polish**: Deterministic Swap Descent to clean up local inefficiencies immediately.
    3.  **Sprint Phase**:
        -   Select top 3 distinct candidates.
        -   Run short LAHC on each.
    4.  **Marathon Phase**:
        -   Select winner of Sprint.
        -   Run extended LAHC with **Multi-Scale Mutations** (Micro/Macro blocks) and **Terminal Cooling**.
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

    # --- LAHC Engine ---
    def run_lahc(start_seq, start_cost, iterations, history_len=50, cooling_ratio=0.0):
        """
        Late Acceptance Hill Climbing with Multi-Scale Mutation and Cooling.
        cooling_ratio: Fraction of iterations at the end to switch to strict Hill Climbing.
        """
        curr_s = list(start_seq)
        curr_c = start_cost
        best_s = list(start_seq)
        best_c = start_cost

        history = [curr_c] * history_len
        
        cooling_start = int(iterations * (1.0 - cooling_ratio))

        for i in range(iterations):
            # Mutation Strategy
            op = random.random()
            neigh_s = list(curr_s)
            slen = len(neigh_s)

            if op < 0.45: 
                # Single Shift (45%)
                if slen < 2: continue
                f = random.randint(0, slen-1)
                t = random.randint(0, slen-1)
                if f != t:
                    item = neigh_s.pop(f)
                    neigh_s.insert(t, item)
                    
            elif op < 0.70: 
                # Micro Block Shift (25%) - Local polish of dependencies
                if slen < 6: continue
                bsize = random.randint(2, 4)
                f = random.randint(0, slen-bsize)
                block = neigh_s[f:f+bsize]
                del neigh_s[f:f+bsize]
                t = random.randint(0, len(neigh_s))
                neigh_s[t:t] = block
                
            elif op < 0.80: 
                # Macro Block Shift (10%) - Structural reorganization
                if slen < 12: continue
                # Larger blocks to move entire sub-graphs of dependencies
                bsize = random.randint(5, 12)
                f = random.randint(0, slen-bsize)
                block = neigh_s[f:f+bsize]
                del neigh_s[f:f+bsize]
                t = random.randint(0, len(neigh_s))
                neigh_s[t:t] = block
                
            else: 
                # Swap (20%) - Very local fix
                if slen < 2: continue
                idx = random.randint(0, slen-2)
                neigh_s[idx], neigh_s[idx+1] = neigh_s[idx+1], neigh_s[idx]

            neigh_c = workload.get_opt_seq_cost(neigh_s)

            # Acceptance Logic
            is_cooling = (i >= cooling_start)
            
            if is_cooling:
                # Strict Hill Climbing (Cooling Phase)
                # Only accept improvements to lock in the minimum
                if neigh_c <= curr_c:
                    curr_s = neigh_s
                    curr_c = neigh_c
                    if curr_c < best_c:
                        best_c = curr_c
                        best_s = list(curr_s)
            else:
                # LAHC Standard
                h_idx = i % history_len
                if neigh_c <= curr_c or neigh_c <= history[h_idx]:
                    curr_s = neigh_s
                    curr_c = neigh_c
                    if curr_c < best_c:
                        best_c = curr_c
                        best_s = list(curr_s)
                
                history[h_idx] = curr_c

        return best_c, best_s

    # Configuration
    SAMPLE_SIZE = 12
    candidates = []

    # --- Phase 1: Diverse Greedy Construction ---
    for i in range(num_seqs):
        # Diversity Strategy:
        # Candidate 0: Proven baseline params
        # Others: Random exploration of Alpha (Packing weight) and Threshold (Big Rock filter)
        if i == 0:
            current_threshold_ratio = 0.90
            current_alpha = 0.05
        else:
            # Wider exploration: 
            # Alpha up to 0.3 allows strong packing preference
            # Threshold down to 0.8 allows looser "Big Rock" definition
            current_threshold_ratio = random.uniform(0.80, 0.98)
            current_alpha = random.uniform(0.0, 0.30)

        remaining = set(range(workload.num_txns))

        # Start with random transaction
        start_txn = random.choice(list(remaining))
        current_seq = [start_txn]
        remaining.remove(start_txn)

        while remaining:
            # Dynamic Pool Construction
            rem_lens = [txn_lens[t] for t in remaining]
            max_rem_len = max(rem_lens) if rem_lens else 0
            threshold = max_rem_len * current_threshold_ratio

            # Identify "Big Rocks"
            big_rocks = [t for t in remaining if txn_lens[t] >= threshold]
            
            # Fill pool
            pool = list(big_rocks)
            needed = SAMPLE_SIZE - len(pool)
            if needed > 0:
                others = [t for t in remaining if t not in big_rocks]
                if len(others) > needed:
                    pool.extend(random.sample(others, needed))
                else:
                    pool.extend(others)
            
            pool = list(set(pool))

            # Weighted Selection
            best_cand = -1
            best_score = float('inf')

            for t in pool:
                # Sim cost
                cost = workload.get_opt_seq_cost(current_seq + [t])
                # Heuristic: Cost minus benefit of packing a large item
                score = cost - (current_alpha * txn_lens[t])

                if score < best_score:
                    best_score = score
                    best_cand = t

            current_seq.append(best_cand)
            remaining.remove(best_cand)

        # Quick Polish: Deterministic Adjacent Swap Descent
        # Fast way to fix obvious greedy ordering errors before heavy optimization
        current_cost = workload.get_opt_seq_cost(current_seq)
        improved = True
        while improved:
            improved = False
            for j in range(len(current_seq) - 1):
                # Speculative swap
                current_seq[j], current_seq[j+1] = current_seq[j+1], current_seq[j]
                new_cost = workload.get_opt_seq_cost(current_seq)
                if new_cost < current_cost:
                    current_cost = new_cost
                    improved = True
                else:
                    # Revert
                    current_seq[j], current_seq[j+1] = current_seq[j+1], current_seq[j]

        candidates.append((current_cost, current_seq))

    # --- Phase 2: Sprint (Multi-Candidate Refinement) ---
    # Filter distinct starting points to maximize search utility
    candidates.sort(key=lambda x: x[0])
    
    unique_candidates = []
    seen_costs = set()
    for cost, seq in candidates:
        if cost not in seen_costs:
            unique_candidates.append((cost, list(seq)))
            seen_costs.add(cost)
        if len(unique_candidates) >= 3:
            break
            
    if not unique_candidates:
        unique_candidates = [candidates[0]]

    sprint_results = []
    SPRINT_ITERS = 300

    for cost, seq in unique_candidates:
        # Short run, no cooling needed
        res = run_lahc(seq, cost, SPRINT_ITERS, history_len=30, cooling_ratio=0.0)
        sprint_results.append(res)

    # --- Phase 3: Marathon Refinement (Global Optimization) ---
    sprint_results.sort(key=lambda x: x[0])
    champion_cost, champion_seq = sprint_results[0]

    MARATHON_ITERS = 3000
    # Cooling ratio 0.2 means last 600 iterations are pure Hill Climbing
    final_cost, final_seq = run_lahc(
        champion_seq, 
        champion_cost, 
        MARATHON_ITERS, 
        history_len=100, # Deeper history for main run
        cooling_ratio=0.2
    )

    return final_cost, final_seq

def get_random_costs():
    """Evaluate scheduling algorithm on three different workloads."""
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