# EVOLVE-BLOCK-START
"""Transaction scheduling algorithm for optimizing makespan across multiple workloads"""

import time
import random
import sys
import os
import math
import re

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
    Get optimal schedule using Adaptive Big Rocks Greedy and Multi-Stage LAHC Refinement.

    Algorithm:
    1.  **Txn Analysis**: Compute transaction operation counts.
    2.  **Adaptive Greedy Phase**:
        -   Generate `num_seqs` candidates.
        -   **Adaptive Pool**: Select "Big Rocks" (txns within 85% of max remaining length) + Randoms.
        -   **Weighted Selection**: Minimize `cost - (alpha * length)`. Allows slight makespan increase
            to place large rocks early (Packing heuristic).
        -   **Polish**: Full Adjacent Swap Descent on each greedy candidate.
    3.  **Multi-Candidate Sprint**:
        -   Select top 3 distinct candidates from greedy phase.
        -   Run short Late Acceptance Hill Climbing (LAHC) on each (Sprint).
    4.  **Marathon Refinement**:
        -   Select winner of the Sprint.
        -   Run extended LAHC (Marathon).
        -   **Operators**:
            -   Single Shift (Insert): 50%
            -   Block Shift (Move 2-4 txns): 30% - Preserves dependency chains.
            -   Adjacent Swap: 20%

    Args:
        workload: Workload object
        num_seqs: Number of independent random starts

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

    # Heuristic Parameters
    SAMPLE_SIZE = 12
    BIG_ROCK_THRESHOLD = 0.85
    WEIGHT_ALPHA = 0.05

    candidates = []

    # --- Phase 1: Adaptive Greedy Construction ---
    for _ in range(num_seqs):
        remaining = set(range(workload.num_txns))

        # Random start
        start_txn = random.choice(list(remaining))
        current_seq = [start_txn]
        remaining.remove(start_txn)

        while remaining:
            # Determine dynamic "Big Rocks" threshold
            rem_lens = [txn_lens[t] for t in remaining]
            max_rem_len = max(rem_lens) if rem_lens else 0
            threshold = max_rem_len * BIG_ROCK_THRESHOLD

            # Form pool: Big Rocks + Randoms
            big_rocks = [t for t in remaining if txn_lens[t] >= threshold]

            pool = list(big_rocks)
            needed = SAMPLE_SIZE - len(pool)

            if needed > 0:
                others = [t for t in remaining if t not in big_rocks]
                if len(others) > needed:
                    pool.extend(random.sample(others, needed))
                else:
                    pool.extend(others)

            # Deduplicate just in case
            pool = list(set(pool))

            best_cand = -1
            best_score = float('inf')

            # Evaluate Pool
            for t in pool:
                cost = workload.get_opt_seq_cost(current_seq + [t])
                # Weighted score: allow worse cost if txn is long
                score = cost - (WEIGHT_ALPHA * txn_lens[t])

                if score < best_score:
                    best_score = score
                    best_cand = t

            current_seq.append(best_cand)
            remaining.remove(best_cand)

        # Quick Polish (Hill Climbing)
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

    # --- Phase 2: Multi-Candidate Sprint (Short LAHC) ---
    # Sort by cost and pick top 3 distinct schedules
    candidates.sort(key=lambda x: x[0])

    # Simple deduplication based on cost helps diversity
    unique_candidates = []
    seen_costs = set()
    for cost, seq in candidates:
        if cost not in seen_costs:
            unique_candidates.append((cost, list(seq)))
            seen_costs.add(cost)
        if len(unique_candidates) >= 3:
            break

    # Fallback if all costs same
    if not unique_candidates:
        unique_candidates = [candidates[0]]

    sprint_results = []

    # LAHC Helper
    def run_lahc(start_seq, start_cost, iterations, history_len=50):
        curr_s = list(start_seq)
        curr_c = start_cost
        best_s = list(start_seq)
        best_c = start_cost

        history = [curr_c] * history_len

        for i in range(iterations):
            # Mutate
            op = random.random()
            neigh_s = list(curr_s)
            slen = len(neigh_s)

            if op < 0.5: # Single Shift
                if slen < 2: continue
                f = random.randint(0, slen-1)
                t = random.randint(0, slen-1)
                if f==t: continue
                item = neigh_s.pop(f)
                neigh_s.insert(t, item)
            elif op < 0.8: # Block Shift
                if slen < 5: continue
                bsize = random.randint(2, 4)
                f = random.randint(0, slen-bsize)
                block = neigh_s[f:f+bsize]
                del neigh_s[f:f+bsize]
                t = random.randint(0, len(neigh_s))
                neigh_s[t:t] = block
            else: # Swap
                if slen < 2: continue
                idx = random.randint(0, slen-2)
                neigh_s[idx], neigh_s[idx+1] = neigh_s[idx+1], neigh_s[idx]

            neigh_c = workload.get_opt_seq_cost(neigh_s)

            # Acceptance
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
    SPRINT_ITERS = 200
    for cost, seq in unique_candidates:
        res = run_lahc(seq, cost, SPRINT_ITERS)
        sprint_results.append(res)

    # --- Phase 3: Marathon Refinement (Long LAHC) ---
    sprint_results.sort(key=lambda x: x[0])
    champion_cost, champion_seq = sprint_results[0]

    MARATHON_ITERS = 2000
    final_cost, final_seq = run_lahc(champion_seq, champion_cost, MARATHON_ITERS)

    return final_cost, final_seq


def get_random_costs():
    """
    Evaluate scheduling algorithm on three different workloads.

    Returns:
        Tuple of (total_makespan, list_of_schedules, execution_time)
    """
    start_time = time.time()
    # Number of initial candidates to generate
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