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
    Hybrid Scheduler combining Diverse Greedy with Decay and Multi-Scale LAHC.

    Improvements:
    - **Pool Size**: Increased to 20 for better local fit during construction.
    - **Decaying Alpha**: Linearly decays length-weight to zero to transition from "Packing" to "Fitting".
    - **Multi-Scale Mutation**: Includes Micro-Block (local repair), Macro-Block (structural change), 
      and Smart-Insert (greedy repair).
    - **Deep Optimization**: 4000 iterations in Marathon phase with Cooling and Stagnation Kicks.
    """

    # --- Pre-computation ---
    # Parse transaction lengths (approximate operation count)
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

    # --- Core Optimization Engine ---
    def run_lahc(start_seq, start_cost, iterations, history_len, enable_kick=False):
        """
        Late Acceptance Hill Climbing with Multi-Scale Mutations.
        """
        curr_s = list(start_seq)
        curr_c = start_cost
        best_s = list(start_seq)
        best_c = start_cost

        history = [curr_c] * history_len
        last_imp_iter = 0
        
        # Cooling starts at 80% of iterations
        cooling_start = int(iterations * 0.8)

        for i in range(iterations):
            is_cooling = (i >= cooling_start)

            # Stagnation Kick
            # If enabled, not in cooling, and no improvement for 600 iters
            if enable_kick and not is_cooling and (i - last_imp_iter > 600):
                slen = len(curr_s)
                if slen > 15:
                    # Shuffle a significant segment
                    seg_len = random.randint(10, 20) 
                    idx = random.randint(0, slen - seg_len)
                    segment = curr_s[idx : idx + seg_len]
                    random.shuffle(segment)
                    curr_s[idx : idx + seg_len] = segment
                    
                    curr_c = workload.get_opt_seq_cost(curr_s)
                    history = [curr_c] * history_len # Reset history
                    last_imp_iter = i
                    continue

            # Mutation Selection
            op = random.random()
            neigh_s = list(curr_s)
            slen = len(neigh_s)
            neigh_c = -1

            # 1. Smart Insert (5%) - Expensive but high quality
            # Takes an item and places it in the best of K random positions
            if op < 0.05:
                if slen < 5: continue
                idx = random.randint(0, slen-1)
                item = neigh_s.pop(idx)
                
                # Sample 3 positions
                best_loc_c = float('inf')
                best_loc_s = None
                
                # Heuristic: Try 3 random positions
                targets = [random.randint(0, len(neigh_s)) for _ in range(3)]
                
                for t_idx in targets:
                    tmp = list(neigh_s)
                    tmp.insert(t_idx, item)
                    c = workload.get_opt_seq_cost(tmp)
                    if c < best_loc_c:
                        best_loc_c = c
                        best_loc_s = tmp
                
                neigh_s = best_loc_s
                neigh_c = best_loc_c

            # 2. Micro Block Shift (25%) - Size 2-4
            elif op < 0.30:
                if slen < 6: continue
                bsize = random.randint(2, 4)
                f = random.randint(0, slen - bsize)
                block = neigh_s[f : f+bsize]
                del neigh_s[f : f+bsize]
                t = random.randint(0, len(neigh_s))
                neigh_s[t:t] = block

            # 3. Macro Block Shift (10%) - Size 6-12
            elif op < 0.40:
                if slen < 15: continue
                bsize = random.randint(6, 12)
                f = random.randint(0, slen - bsize)
                block = neigh_s[f : f+bsize]
                del neigh_s[f : f+bsize]
                t = random.randint(0, len(neigh_s))
                neigh_s[t:t] = block
            
            # 4. Single Shift (40%)
            elif op < 0.80:
                if slen < 2: continue
                f, t = random.randint(0, slen-1), random.randint(0, slen-1)
                if f != t:
                    item = neigh_s.pop(f)
                    neigh_s.insert(t, item)

            # 5. Swap (20%)
            else:
                if slen < 2: continue
                idx = random.randint(0, slen - 2)
                neigh_s[idx], neigh_s[idx+1] = neigh_s[idx+1], neigh_s[idx]

            # Calculate cost if not set
            if neigh_c == -1:
                neigh_c = workload.get_opt_seq_cost(neigh_s)

            # Acceptance Logic
            accepted = False
            if is_cooling:
                # Strict Hill Climbing
                if neigh_c < curr_c:
                    accepted = True
            else:
                # Late Acceptance
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
            
            if not is_cooling:
                history[i % history_len] = curr_c

        return best_c, best_s

    # --- Phase 1: Diverse Greedy Generation ---
    candidates = []
    
    # Increase pool size to find better fits
    POOL_SIZE = 20 

    for i in range(num_seqs):
        # Diversity Parameters
        if i == 0:
            # Anchor: Proven heuristic values
            start_alpha = 0.30
            threshold_ratio = 0.90
        else:
            # Exploration
            start_alpha = random.uniform(0.10, 0.50)
            threshold_ratio = random.uniform(0.85, 0.98)
            
        remaining = set(range(workload.num_txns))
        total_txns = workload.num_txns
        
        # Start random
        start_txn = random.choice(list(remaining))
        curr_seq = [start_txn]
        remaining.remove(start_txn)
        
        while remaining:
            # Decay Alpha: Linear decay to 0
            # Early: prioritizing Big Rocks. Late: Prioritizing Fit (Cost).
            progress = len(curr_seq) / total_txns
            curr_alpha = start_alpha * (1.0 - progress)
            
            # Filter Pool
            rem_lens = [txn_lens[t] for t in remaining]
            max_len = max(rem_lens) if rem_lens else 0
            threshold = max_len * threshold_ratio
            
            big_rocks = [t for t in remaining if txn_lens[t] >= threshold]
            pool = list(big_rocks)
            
            # Fill pool if needed
            needed = POOL_SIZE - len(pool)
            if needed > 0:
                others = [t for t in remaining if t not in big_rocks]
                if len(others) <= needed:
                    pool.extend(others)
                else:
                    pool.extend(random.sample(others, needed))
            
            pool = list(set(pool))
            
            # Select Best
            best_t = -1
            best_score = float('inf')
            
            for t in pool:
                cost = workload.get_opt_seq_cost(curr_seq + [t])
                score = cost - (curr_alpha * txn_lens[t])
                if score < best_score:
                    best_score = score
                    best_t = t
            
            curr_seq.append(best_t)
            remaining.remove(best_t)
            
        # Fast Polish (Descent)
        # Deterministic swap cleanup
        c_cost = workload.get_opt_seq_cost(curr_seq)
        improved = True
        while improved:
            improved = False
            for j in range(len(curr_seq)-1):
                # Swap
                curr_seq[j], curr_seq[j+1] = curr_seq[j+1], curr_seq[j]
                nc = workload.get_opt_seq_cost(curr_seq)
                if nc < c_cost:
                    c_cost = nc
                    improved = True
                else:
                    # Revert
                    curr_seq[j], curr_seq[j+1] = curr_seq[j+1], curr_seq[j]
                    
        candidates.append((c_cost, curr_seq))

    # --- Phase 2: Sprint (Filter) ---
    # Sort and pick unique
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
    # Short intense run
    SPRINT_ITERS = 400
    for c, s in unique_candidates:
        res = run_lahc(s, c, SPRINT_ITERS, history_len=50, enable_kick=False)
        sprint_results.append(res)
        
    # --- Phase 3: Marathon (Deep Optimization) ---
    sprint_results.sort(key=lambda x: x[0])
    champ_c, champ_s = sprint_results[0]
    
    # Extended run with all features
    MARATHON_ITERS = 4000
    final_c, final_s = run_lahc(champ_s, champ_c, MARATHON_ITERS, history_len=150, enable_kick=True)
    
    return final_c, final_s


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