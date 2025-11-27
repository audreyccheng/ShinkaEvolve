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
    Hybrid Scheduler: Adaptive Greedy + Polish + Advanced LAHC (Kick & Best-Fit).

    Algorithm:
    1.  **Txn Analysis**: Pre-compute transaction operation counts.
    2.  **Diverse Greedy Phase**:
        -   Generate `num_seqs` candidates.
        -   **Adaptive Pool**: "Big Rocks" + Randoms with randomized threshold/alpha.
        -   **Polish**: Deterministic Adjacent Swap Descent to fix immediate greedy errors.
    3.  **Sprint Phase**:
        -   Top 3 candidates.
        -   LAHC with **Sampled Best-Fit** mutation.
    4.  **Marathon Phase**:
        -   Champion candidate.
        -   Extended LAHC with **Stagnation Kick** and **Sampled Best-Fit**.
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

    # --- Advanced LAHC Engine ---
    def run_lahc(start_seq, start_cost, iterations, history_len=50, enable_kick=False, use_best_fit=False):
        curr_s = list(start_seq)
        curr_c = start_cost
        best_s = list(start_seq)
        best_c = start_cost

        history = [curr_c] * history_len
        last_imp_iter = 0

        for i in range(iterations):
            # 1. Stagnation Kick (Escape Deep Local Optima)
            # If enabled and no improvement for a long time, shuffle a segment
            if enable_kick and (i - last_imp_iter) > 500:
                slen = len(curr_s)
                if slen > 15:
                    seg_len = random.randint(5, 12)
                    idx = random.randint(0, slen - seg_len)
                    segment = curr_s[idx:idx+seg_len]
                    random.shuffle(segment)
                    curr_s[idx:idx+seg_len] = segment
                    
                    # Recalculate cost after kick
                    curr_c = workload.get_opt_seq_cost(curr_s)
                    # Reset history to accept the kick
                    history = [curr_c] * history_len
                    last_imp_iter = i
                    continue

            op = random.random()
            neigh_s = list(curr_s)
            slen = len(neigh_s)
            neigh_c = -1
            
            # 2. Sampled Best-Fit Insert (High Quality Mutation)
            # Expensive, so low probability. Tries K positions for an item.
            if use_best_fit and op < 0.04 and slen > 5:
                # Remove random item
                r_idx = random.randint(0, slen-1)
                item = neigh_s.pop(r_idx)
                
                # Sample K target positions
                K_SAMPLES = 5
                best_local_seq = None
                best_local_cost = float('inf')
                
                for _ in range(K_SAMPLES):
                    ins_idx = random.randint(0, len(neigh_s))
                    temp_s = list(neigh_s)
                    temp_s.insert(ins_idx, item)
                    c = workload.get_opt_seq_cost(temp_s)
                    if c < best_local_cost:
                        best_local_cost = c
                        best_local_seq = temp_s
                
                if best_local_seq:
                    neigh_s = best_local_seq
                    neigh_c = best_local_cost
                else:
                    neigh_s.insert(r_idx, item) # Revert
                    neigh_c = curr_c
            
            # 3. Standard Mutations
            elif op < 0.50: # Single Shift
                if slen < 2: continue
                f = random.randint(0, slen-1)
                t = random.randint(0, slen-1)
                if f != t:
                    item = neigh_s.pop(f)
                    neigh_s.insert(t, item)
            
            elif op < 0.85: # Block Shift
                if slen < 8: continue
                bsize = random.randint(2, 9) # Variable block size
                f = random.randint(0, slen-bsize)
                block = neigh_s[f:f+bsize]
                del neigh_s[f:f+bsize]
                t = random.randint(0, len(neigh_s))
                neigh_s[t:t] = block
            
            else: # Swap
                if slen < 2: continue
                idx = random.randint(0, slen-2)
                neigh_s[idx], neigh_s[idx+1] = neigh_s[idx+1], neigh_s[idx]

            # Calculate cost if not already done
            if neigh_c == -1:
                neigh_c = workload.get_opt_seq_cost(neigh_s)

            # LAHC Acceptance
            h_idx = i % history_len
            if neigh_c <= curr_c or neigh_c <= history[h_idx]:
                curr_s = neigh_s
                curr_c = neigh_c
                if curr_c < best_c:
                    best_c = curr_c
                    best_s = list(curr_s)
                    last_imp_iter = i

            history[h_idx] = curr_c

        return best_c, best_s

    # Configuration
    SAMPLE_SIZE = 12
    candidates = []

    # --- Phase 1: Diverse Greedy Construction ---
    for _ in range(num_seqs):
        # Parameters: Wide range to explore different structures
        current_threshold_ratio = random.uniform(0.80, 0.98)
        current_alpha = random.uniform(0.0, 0.10)

        remaining = set(range(workload.num_txns))

        # Random start
        start_txn = random.choice(list(remaining))
        current_seq = [start_txn]
        remaining.remove(start_txn)

        while remaining:
            # Dynamic "Big Rocks" threshold
            rem_lens = [txn_lens[t] for t in remaining]
            max_rem_len = max(rem_lens) if rem_lens else 0
            threshold = max_rem_len * current_threshold_ratio

            # Form pool
            big_rocks = [t for t in remaining if txn_lens[t] >= threshold]
            pool = list(big_rocks)

            needed = SAMPLE_SIZE - len(pool)
            if needed > 0:
                others = [t for t in remaining if t not in big_rocks]
                if len(others) > needed:
                    pool.extend(random.sample(others, needed))
                else:
                    pool.extend(others)

            pool = list(set(pool))

            best_cand = -1
            best_score = float('inf')

            # Evaluate Pool
            for t in pool:
                cost = workload.get_opt_seq_cost(current_seq + [t])
                score = cost - (current_alpha * txn_lens[t])

                if score < best_score:
                    best_score = score
                    best_cand = t

            current_seq.append(best_cand)
            remaining.remove(best_cand)

        # Polish: Deterministic Hill Climbing to fix local greedy artifacts
        # This acts as a fast, high-quality cleanup before LAHC
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

    # Top 3 Distinct
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
    SPRINT_ITERS = 250

    for cost, seq in unique_candidates:
        # Enable Best-Fit to find optimal slots quickly in the sprint
        res = run_lahc(seq, cost, SPRINT_ITERS, history_len=30, use_best_fit=True)
        sprint_results.append(res)

    # --- Phase 3: Marathon Refinement ---
    sprint_results.sort(key=lambda x: x[0])
    champion_cost, champion_seq = sprint_results[0]

    MARATHON_ITERS = 3000
    # Enable Kick and Best-Fit for deep search
    final_cost, final_seq = run_lahc(
        champion_seq, 
        champion_cost, 
        MARATHON_ITERS, 
        history_len=100, 
        enable_kick=True, 
        use_best_fit=True
    )

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