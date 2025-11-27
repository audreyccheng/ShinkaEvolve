# EVOLVE-BLOCK-START
"""Transaction scheduling algorithm for optimizing makespan across multiple workloads"""

import time
import random
import sys
import os
import re
import math
from collections import defaultdict

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
    Adaptive LNS Scheduler: Optimized Greedy Construction + LNS-Augmented LAHC.
    
    Algorithm:
    1.  **Txn Analysis**: Pre-compute transaction lengths.
    2.  **Diverse Greedy Phase**:
        -   Generates candidates using a "Packing Heuristic" (Cost - Alpha*Length).
        -   Varies Alpha and Pool Thresholds to explore different structural backbones.
        -   Includes a "Safety" Polish (Descent) on all candidates.
    3.  **Refinement Phase (Sprint & Marathon)**:
        -   **Sprint**: Runs short LAHC on top candidates to identify the most promising basin.
        -   **Marathon**: Runs deep LAHC on the champion.
        -   **LNS-Lite Mutation**: Occasionally removes a small set of items and greedily re-inserts them 
            at best-of-k positions. This repairs local sub-optimality better than random swaps.
        -   **Stagnation Kick**: Shuffles a segment if stuck for too long.
        -   **Cooldown**: Switches to strict Hill Climbing in the final 20% of iterations.
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

    # --- LNS-Augmented LAHC Engine ---
    def run_lahc(start_seq, start_cost, iterations, history_len=50, use_lns=False):
        curr_s = list(start_seq)
        curr_c = start_cost
        best_s = list(start_seq)
        best_c = start_cost

        history = [curr_c] * history_len
        last_imp_iter = 0

        for i in range(iterations):
            # Cooldown Phase: Strict Hill Climbing in last 20%
            is_cooldown = i > (iterations * 0.8)
            
            # Stagnation Kick (only if not cooling down)
            if not is_cooldown and (i - last_imp_iter > 600):
                # Shuffle a significant segment
                slen = len(curr_s)
                if slen > 15:
                    seg_len = random.randint(8, 16)
                    idx = random.randint(0, slen - seg_len)
                    segment = curr_s[idx : idx + seg_len]
                    random.shuffle(segment)
                    curr_s[idx : idx + seg_len] = segment
                    
                    curr_c = workload.get_opt_seq_cost(curr_s)
                    history = [curr_c] * history_len # Reset history
                    last_imp_iter = i
                    continue

            # Select Operator
            op = random.random()
            neigh_s = list(curr_s)
            slen = len(neigh_s)
            neigh_c = -1
            
            # 1. LNS-Lite (High Quality, Expensive) - 3%
            # Remove 2 items, re-insert at Best-of-5 positions
            if use_lns and op < 0.03 and slen > 10:
                # Select 2 distinct indices
                idxs = sorted(random.sample(range(slen), 2), reverse=True)
                items = [neigh_s.pop(idx) for idx in idxs]
                
                # Re-insert one by one
                for item in items:
                    best_local_seq = None
                    best_local_cost = float('inf')
                    # Sample 5 positions + current approximate position
                    candidates_pos = [random.randint(0, len(neigh_s)) for _ in range(5)]
                    
                    for pos in candidates_pos:
                        temp_s = list(neigh_s)
                        temp_s.insert(pos, item)
                        c = workload.get_opt_seq_cost(temp_s)
                        if c < best_local_cost:
                            best_local_cost = c
                            best_local_seq = temp_s
                    
                    # If we found a valid insertion, update
                    if best_local_seq:
                        neigh_s = best_local_seq
                    else:
                        # Fallback (shouldn't happen with random pos, but just in case)
                        neigh_s.append(item)
                
                neigh_c = best_local_cost # Already computed

            # 2. Block Shift (Structure Preserving) - 35%
            elif op < 0.38:
                if slen < 6: continue
                # Block size 2-6
                bsize = random.randint(2, 6)
                f = random.randint(0, slen - bsize)
                block = neigh_s[f : f+bsize]
                del neigh_s[f : f+bsize]
                t = random.randint(0, len(neigh_s))
                neigh_s[t:t] = block

            # 3. Single Shift (Fine Tuning) - 45%
            elif op < 0.83:
                if slen < 2: continue
                f = random.randint(0, slen-1)
                t = random.randint(0, slen-1)
                if f != t:
                    item = neigh_s.pop(f)
                    neigh_s.insert(t, item)
            
            # 4. Swap (Local) - 17%
            else:
                if slen < 2: continue
                idx = random.randint(0, slen-2)
                neigh_s[idx], neigh_s[idx+1] = neigh_s[idx+1], neigh_s[idx]

            # Calculate cost if not already done
            if neigh_c == -1:
                neigh_c = workload.get_opt_seq_cost(neigh_s)

            # Acceptance Logic
            accepted = False
            if is_cooldown:
                # Strict Descent
                if neigh_c < curr_c:
                    accepted = True
            else:
                # LAHC
                h_idx = i % history_len
                if neigh_c <= curr_c or neigh_c <= history[h_idx]:
                    accepted = True

            if accepted:
                curr_s = neigh_s
                curr_c = neigh_c
                if curr_c < best_c:
                    best_c = curr_c
                    best_s = list(curr_s)
                    last_imp_iter = i
            
            # Update history (only if not cooldown)
            if not is_cooldown:
                h_idx = i % history_len
                history[h_idx] = curr_c

        return best_c, best_s

    # --- Phase 1: Diverse Greedy Construction ---
    candidates = []
    
    # We use a mix of heuristic settings to populate the candidate pool
    # The 'Alpha' controls how much we prioritize length (Packing)
    # The 'Threshold' controls how picky we are about only taking big rocks
    
    for i in range(num_seqs):
        # Diversity Schedule
        if i == 0:
            # Baseline: Balanced
            alpha = 0.05
            threshold = 0.90
        elif i == 1:
            # Aggressive Packing
            alpha = 0.20
            threshold = 0.85
        elif i == 2:
            # Conservative (Minimize Cost Increase)
            alpha = 0.0
            threshold = 0.95
        else:
            # Random Exploration
            alpha = random.uniform(0.0, 0.25)
            threshold = random.uniform(0.80, 0.98)

        remaining = set(range(workload.num_txns))
        
        # Random start
        start_txn = random.choice(list(remaining))
        current_seq = [start_txn]
        remaining.remove(start_txn)

        SAMPLE_SIZE = 12

        while remaining:
            # 1. Identify "Big Rocks" tier
            rem_lens = [txn_lens[t] for t in remaining]
            max_rem_len = max(rem_lens) if rem_lens else 0
            limit = max_rem_len * threshold

            # Pool: Big Rocks + Random Fillers
            big_rocks = [t for t in remaining if txn_lens[t] >= limit]
            pool = list(big_rocks)
            
            # Fill pool to sample size if needed
            needed = SAMPLE_SIZE - len(pool)
            if needed > 0:
                others = [t for t in remaining if t not in big_rocks]
                if len(others) <= needed:
                    pool.extend(others)
                else:
                    pool.extend(random.sample(others, needed))
            
            pool = list(set(pool))
            
            best_cand = -1
            best_score = float('inf')
            
            # 2. Selection
            for t in pool:
                cost = workload.get_opt_seq_cost(current_seq + [t])
                # Score = NewCost - (Alpha * Length)
                # Lower is better.
                # If Alpha is high, we tolerate higher cost for longer txns.
                score = cost - (alpha * txn_lens[t])
                
                if score < best_score:
                    best_score = score
                    best_cand = t
            
            current_seq.append(best_cand)
            remaining.remove(best_cand)

        # Quick Polish (Descent)
        # Fixes obvious local ordering issues from greedy placement
        curr_cost = workload.get_opt_seq_cost(current_seq)
        improved = True
        while improved:
            improved = False
            for j in range(len(current_seq) - 1):
                # Try swap
                current_seq[j], current_seq[j+1] = current_seq[j+1], current_seq[j]
                nc = workload.get_opt_seq_cost(current_seq)
                if nc < curr_cost:
                    curr_cost = nc
                    improved = True
                else:
                    # Revert
                    current_seq[j], current_seq[j+1] = current_seq[j+1], current_seq[j]
        
        candidates.append((curr_cost, current_seq))

    # --- Phase 2: Sprint (Top 3) ---
    candidates.sort(key=lambda x: x[0])
    
    unique_candidates = []
    seen = set()
    for c, s in candidates:
        if c not in seen:
            unique_candidates.append((c, list(s)))
            seen.add(c)
        if len(unique_candidates) >= 3:
            break
            
    if not unique_candidates:
        unique_candidates = [candidates[0]]
        
    sprint_results = []
    # Short sprint, LNS enabled for high quality moves
    SPRINT_ITERS = 250
    for cost, seq in unique_candidates:
        # Use LNS here to quickly find good slots for misplaced items
        res = run_lahc(seq, cost, SPRINT_ITERS, history_len=30, use_lns=True)
        sprint_results.append(res)

    # --- Phase 3: Marathon (Champion) ---
    sprint_results.sort(key=lambda x: x[0])
    champ_cost, champ_seq = sprint_results[0]
    
    # Long run, larger history, LNS enabled
    MARATHON_ITERS = 3000
    final_cost, final_seq = run_lahc(champ_seq, champ_cost, MARATHON_ITERS, history_len=100, use_lns=True)

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