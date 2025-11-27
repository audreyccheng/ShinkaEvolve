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
    Get optimal schedule using Non-Linear Greedy Decay and Multi-Scale LAHC with Dynamic Compression.

    Algorithm:
    1.  **Txn Analysis**: Pre-compute transaction operation counts.
    2.  **Diverse Greedy Phase**:
        -   Generate `num_seqs` candidates.
        -   **Non-Linear Decay**: Alpha (weight for length) decays via a randomized power law.
            Varied `decay_power` allows different trade-off curves (concave vs convex).
        -   **Adaptive Construction**: Select from pool of large and random transactions.
        -   **Warmup**: Short LAHC.
    3.  **Sprint Phase**:
        -   Run LAHC on top 3 candidates.
    4.  **Marathon Phase**:
        -   Run extended LAHC on best candidate.
        -   **Advanced LAHC**:
            -   **Mutations**: Sampled Best-Fit, Block Reversal, Multi-scale Shifts, Swaps.
            -   **Dynamic History Compression**: Linearly reduces history length to 1 (Hill Climbing)
                over the course of the run, acting as a continuous cooling schedule.
            -   **Stagnation Kick**: Disruptive shuffle if stuck in local optimum.
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

    # LAHC Engine
    def run_lahc(start_seq, start_cost, iterations, history_len=50, enable_kick=False, compress_history=False):
        curr_s = list(start_seq)
        curr_c = start_cost
        best_s = list(start_seq)
        best_c = start_cost

        # Initialize history
        history = [curr_c] * history_len
        last_imp_iter = 0

        for i in range(iterations):
            # 1. Dynamic History Compression (Simulated Annealing-like behavior)
            if compress_history:
                # Linearly decrease history length from history_len down to 1
                remaining_ratio = 1.0 - (i / iterations)
                eff_h_len = max(1, int(history_len * remaining_ratio))
            else:
                eff_h_len = history_len

            # 2. Stagnation Kick
            # Trigger if no improvement for significant duration (and not in final cooling stages)
            if enable_kick and (eff_h_len > 5) and (i - last_imp_iter > 600):
                slen = len(curr_s)
                if slen > 15:
                    seg_len = random.randint(8, 16)
                    idx = random.randint(0, slen - seg_len)
                    segment = curr_s[idx:idx+seg_len]
                    random.shuffle(segment)
                    curr_s[idx:idx+seg_len] = segment

                    curr_c = workload.get_opt_seq_cost(curr_s)
                    # Reset history to absorb kick
                    history = [curr_c] * history_len
                    last_imp_iter = i
                    continue

            # 3. Mutation
            op = random.random()
            neigh_s = list(curr_s)
            slen = len(neigh_s)
            neigh_c = -1

            # A. Sampled Best-Fit (4%)
            if op < 0.04:
                if slen < 5: continue
                idx = random.randint(0, slen-1)
                item = neigh_s.pop(idx)

                # Small sample size for speed
                best_loc_c = float('inf')
                best_loc_s = None
                for _ in range(3):
                    ins = random.randint(0, len(neigh_s))
                    tmp = list(neigh_s)
                    tmp.insert(ins, item)
                    c = workload.get_opt_seq_cost(tmp)
                    if c < best_loc_c:
                        best_loc_c = c
                        best_loc_s = tmp
                neigh_s = best_loc_s
                neigh_c = best_loc_c

            # B. Block Reversal (5%) - NEW
            # Untangles backward dependency chains
            elif op < 0.09:
                if slen < 5: continue
                bsize = random.randint(3, 8)
                f = random.randint(0, slen-bsize)
                # Reverse the segment in place
                neigh_s[f:f+bsize] = neigh_s[f:f+bsize][::-1]

            # C. Micro-Block Shift (25%)
            elif op < 0.34:
                if slen < 5: continue
                bsize = random.randint(2, 4)
                f = random.randint(0, slen-bsize)
                block = neigh_s[f:f+bsize]
                del neigh_s[f:f+bsize]
                t = random.randint(0, len(neigh_s))
                neigh_s[t:t] = block

            # D. Macro-Block Shift (6%)
            elif op < 0.40:
                if slen < 12: continue
                bsize = random.randint(5, 10)
                f = random.randint(0, slen-bsize)
                block = neigh_s[f:f+bsize]
                del neigh_s[f:f+bsize]
                t = random.randint(0, len(neigh_s))
                neigh_s[t:t] = block

            # E. Single Shift (40%)
            elif op < 0.80:
                if slen < 2: continue
                f = random.randint(0, slen-1)
                t = random.randint(0, slen-1)
                if f != t:
                    item = neigh_s.pop(f)
                    neigh_s.insert(t, item)

            # F. Swap (20%)
            else:
                if slen < 2: continue
                idx = random.randint(0, slen-2)
                neigh_s[idx], neigh_s[idx+1] = neigh_s[idx+1], neigh_s[idx]

            if neigh_c == -1:
                neigh_c = workload.get_opt_seq_cost(neigh_s)

            # 4. Acceptance Logic with Effective History Length
            h_idx = i % eff_h_len
            # Check against current OR the specific history slot
            if neigh_c <= curr_c or neigh_c <= history[h_idx]:
                curr_s = neigh_s
                curr_c = neigh_c
                if curr_c < best_c:
                    best_c = curr_c
                    best_s = list(curr_s)
                    last_imp_iter = i

            # Update history (always update the slot we looked at)
            history[h_idx] = curr_c

        return best_c, best_s

    # Configuration
    SAMPLE_SIZE = 24
    candidates = []

    # --- Phase 1: Diverse Greedy Construction ---
    for _ in range(num_seqs):
        # Parameters
        current_threshold_ratio = random.uniform(0.85, 0.98)
        current_alpha = random.uniform(0.05, 0.35)

        # NEW: Randomize decay power (0.5=concave/slow start, 2.0=convex/fast start)
        decay_power = random.uniform(0.5, 2.0)

        remaining = set(range(workload.num_txns))
        total_txns = workload.num_txns

        start_txn = random.choice(list(remaining))
        current_seq = [start_txn]
        remaining.remove(start_txn)

        while remaining:
            # Non-Linear Decay
            # Progress 0 -> 1
            progress = len(current_seq) / total_txns

            # Decay factor goes from 1.0 down to 0.0
            decay_factor = (1.0 - progress) ** decay_power

            # Apply decay to alpha
            eff_alpha = current_alpha * decay_factor

            # Dynamic Pool Threshold
            rem_lens = [txn_lens[t] for t in remaining]
            max_rem_len = max(rem_lens) if rem_lens else 0
            threshold = max_rem_len * current_threshold_ratio

            # Form Pool
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
                score = cost - (eff_alpha * txn_lens[t])

                if score < best_score:
                    best_score = score
                    best_cand = t

            current_seq.append(best_cand)
            remaining.remove(best_cand)

        # Warmup: Short LAHC (no compression)
        base_cost = workload.get_opt_seq_cost(current_seq)
        warmup_cost, warmup_seq = run_lahc(current_seq, base_cost, 150)
        candidates.append((warmup_cost, warmup_seq))

    # --- Phase 2: Sprint (Multi-Candidate Refinement) ---
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
    SPRINT_ITERS = 400

    for cost, seq in unique_candidates:
        res = run_lahc(seq, cost, SPRINT_ITERS, history_len=50, compress_history=False)
        sprint_results.append(res)

    # --- Phase 3: Marathon Refinement (Long LAHC) ---
    sprint_results.sort(key=lambda x: x[0])
    champion_cost, champion_seq = sprint_results[0]

    MARATHON_ITERS = 3500
    # Enable Kick and Dynamic History Compression for Marathon
    # Initial history_len 150 decays to 1 over 3500 steps
    final_cost, final_seq = run_lahc(champion_seq, champion_cost, MARATHON_ITERS,
                                     history_len=150,
                                     enable_kick=True,
                                     compress_history=True)

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