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
    Hybrid Adaptive Scheduler with Hotness-Aware Greedy and Multi-Stage Refinement.

    Algorithm:
    1.  **Workload Analysis**: Computes 'Hotness' (conflict potential) and 'Length' for each transaction.
    2.  **Diverse Greedy Construction**:
        -   Constructs initial schedules using a utility function `Cost - (Alpha*Length + Beta*Hotness)`.
        -   Uses structured parameter diversity (Length-focused, Hotness-focused, Balanced, Random) to explore the landscape.
        -   Applies a "Big Rock" filter based on the combined utility metric.
    3.  **Refinement Funnel**:
        -   **Sprint**: Parallel LAHC on the top 3 distinct candidates.
        -   **Marathon**: Deep LAHC on the best candidate using advanced mutation operators (Inversion, Block Moves, Scramble Kicks).
    """

    # --- 1. Analysis & Pre-computation ---
    # Parse transactions to determine Length (ops count) and Hotness (access frequency of items)
    txn_lens = {}
    txn_hotness = {}
    
    key_pat = re.compile(r'[rw]-(\d+)')
    global_freq = defaultdict(int)
    parsed_txns = []
    
    # Pass 1: Global Frequency Counting
    for i in range(workload.num_txns):
        raw = workload.txns[i]
        if isinstance(raw, (list, tuple)): 
            raw = raw[0]
        t_str = str(raw)
        
        # Extract keys
        keys = [int(k) for k in key_pat.findall(t_str)]
        parsed_txns.append(keys)
        
        # Store Length
        op_count = len(keys)
        txn_lens[i] = op_count
        
        # Update Frequencies
        for k in keys:
            global_freq[k] += 1
            
    # Pass 2: Hotness Calculation & Normalization
    max_hot = 0
    max_len = 0
    
    for i in range(workload.num_txns):
        # Hotness = Sum of global frequencies of keys accessed by this txn
        # This identifies transactions that are likely to block or be blocked
        h = sum(global_freq[k] for k in parsed_txns[i])
        txn_hotness[i] = h
        
        if h > max_hot: max_hot = h
        if txn_lens[i] > max_len: max_len = txn_lens[i]
        
    # Avoid division by zero
    max_hot = max(max_hot, 1)
    max_len = max(max_len, 1)

    # Normalized getters for parameter mixing
    def get_norm_len(t): return txn_lens[t] / max_len
    def get_norm_hot(t): return txn_hotness[t] / max_hot

    # --- 2. Greedy Construction Phase ---
    candidates = []
    SAMPLE_SIZE = 16

    for i in range(num_seqs):
        # Structured Diversity Parameters
        # Alpha: Weight for Length (Packing heuristic)
        # Beta: Weight for Hotness (Conflict Resolution heuristic)
        # Threshold: Pool restriction stringency
        
        if i == 0:
            # Balanced Baseline
            alpha, beta, threshold = 0.4, 0.2, 0.90
        elif i == 1:
            # Length Aggressive (Packing heavy)
            alpha, beta, threshold = 0.8, 0.0, 0.85
        elif i == 2:
            # Hotness Aggressive (Conflict aware)
            alpha, beta, threshold = 0.0, 0.6, 0.90
        elif i == 3:
            # Conservative
            alpha, beta, threshold = 0.2, 0.1, 0.95
        else:
            # Randomized Exploration
            alpha = random.uniform(0.1, 0.7)
            beta = random.uniform(0.0, 0.5)
            threshold = random.uniform(0.80, 0.98)

        remaining = set(range(workload.num_txns))
        
        # Random start to break symmetry
        start_txn = random.choice(list(remaining))
        current_seq = [start_txn]
        remaining.remove(start_txn)

        while remaining:
            # 1. Identify "Big Rocks" based on Length AND Hotness
            # We compute a static priority for remaining items
            priorities = {t: (alpha * get_norm_len(t) + beta * get_norm_hot(t)) for t in remaining}
            max_p = max(priorities.values()) if priorities else 0
            
            # Pool: Items with priority >= threshold * max_priority
            if max_p > 0:
                pool = [t for t, p in priorities.items() if p >= max_p * threshold]
            else:
                pool = list(remaining)

            # Ensure pool size for diversity
            needed = SAMPLE_SIZE - len(pool)
            if needed > 0:
                others = [t for t in remaining if t not in pool]
                if len(others) <= needed:
                    pool.extend(others)
                else:
                    pool.extend(random.sample(others, needed))
            
            pool = list(set(pool))
            
            best_t = -1
            best_score = float('inf')
            
            # 2. Selection: Minimize (Dynamic Cost - Priority)
            for t in pool:
                # Simulator call
                cost = workload.get_opt_seq_cost(current_seq + [t])
                
                # Heuristic Score
                # We subtract weighted raw values to compete with cost magnitude
                # Scaling factors (2.0 and 0.5) adjust for typical range differences
                raw_len = txn_lens[t]
                raw_hot = txn_hotness[t]
                
                score = cost - (alpha * raw_len * 2.0) - (beta * raw_hot * 0.5)
                
                if score < best_score:
                    best_score = score
                    best_t = t
            
            current_seq.append(best_t)
            remaining.remove(best_t)

        # Quick Polish: Store result
        base_cost = workload.get_opt_seq_cost(current_seq)
        candidates.append((base_cost, current_seq))


    # --- 3. LAHC Refinement Engine ---
    def run_lahc(start_seq, start_cost, iterations, history_len=50, mode='standard'):
        curr_s = list(start_seq)
        curr_c = start_cost
        best_s = list(start_seq)
        best_c = start_cost

        history = [curr_c] * history_len
        last_imp_iter = 0

        for i in range(iterations):
            # Dynamic Kick for Marathon
            if mode == 'marathon' and (i - last_imp_iter > 500):
                # Destructive Kick: Shuffle a significant segment
                slen = len(curr_s)
                # Shuffle a random 10% to 25% chunk
                seg_len = random.randint(10, max(10, slen // 4))
                start_idx = random.randint(0, max(0, slen - seg_len))
                
                segment = curr_s[start_idx : start_idx + seg_len]
                random.shuffle(segment)
                curr_s[start_idx : start_idx + seg_len] = segment
                
                curr_c = workload.get_opt_seq_cost(curr_s)
                history = [curr_c] * history_len # Reset history
                last_imp_iter = i
                continue

            # Mutation Selection
            op = random.random()
            neigh_s = list(curr_s)
            slen = len(neigh_s)
            neigh_c = -1

            # 1. Sampled Insert (Exploitation) - 5%
            # Finds best position for a random item among K samples
            if op < 0.05:
                if slen < 5: continue
                idx = random.randint(0, slen-1)
                item = neigh_s.pop(idx)
                
                best_local_seq = None
                best_local_cost = float('inf')
                
                # Try 6 random spots
                for _ in range(6):
                    ins_idx = random.randint(0, len(neigh_s))
                    temp_s = list(neigh_s)
                    temp_s.insert(ins_idx, item)
                    c = workload.get_opt_seq_cost(temp_s)
                    if c < best_local_cost:
                        best_local_cost = c
                        best_local_seq = temp_s
                
                neigh_s = best_local_seq
                neigh_c = best_local_cost

            # 2. Block Move (Structure) - 30%
            # Moves a dependency chain
            elif op < 0.35:
                if slen < 5: continue
                bsize = random.randint(2, 8)
                f = random.randint(0, slen-bsize)
                block = neigh_s[f:f+bsize]
                del neigh_s[f:f+bsize]
                t = random.randint(0, len(neigh_s))
                neigh_s[t:t] = block

            # 3. Inversion (Reversal) - 10%
            # Flips a segment to reverse dependency order
            elif op < 0.45:
                if slen < 4: continue
                bsize = random.randint(3, 10)
                f = random.randint(0, slen-bsize)
                neigh_s[f:f+bsize] = reversed(neigh_s[f:f+bsize])

            # 4. Single Shift (Fine Tuning) - 40%
            elif op < 0.85:
                if slen < 2: continue
                f, t = random.randint(0, slen-1), random.randint(0, slen-1)
                if f != t:
                    item = neigh_s.pop(f)
                    neigh_s.insert(t, item)
            
            # 5. Swap (Local) - 15%
            else:
                if slen < 2: continue
                idx = random.randint(0, slen-2)
                neigh_s[idx], neigh_s[idx+1] = neigh_s[idx+1], neigh_s[idx]

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

    # --- 4. Sprint Phase (Top 3) ---
    # Sort by cost ascending
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
    SPRINT_ITERS = 250
    
    for cost, seq in unique_candidates:
        res = run_lahc(seq, cost, SPRINT_ITERS, history_len=50, mode='sprint')
        sprint_results.append(res)
        
    # --- 5. Marathon Phase (Top 1) ---
    sprint_results.sort(key=lambda x: x[0])
    champ_cost, champ_seq = sprint_results[0]
    
    MARATHON_ITERS = 3000
    final_cost, final_seq = run_lahc(champ_seq, champ_cost, MARATHON_ITERS, history_len=150, mode='marathon')
    
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