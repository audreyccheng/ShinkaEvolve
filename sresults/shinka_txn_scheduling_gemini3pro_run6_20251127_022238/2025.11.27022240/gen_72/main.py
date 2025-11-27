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
    Get optimal schedule using Quartic-Weighted Beam Search and Extended ILS.

    Key Features:
    - Quartic Weighting (d^4): Aggressively biases sampling towards heavy transactions.
    - Extended Lookahead: Evaluates top-6 LPT candidates.
    - Extended Local Search: Includes small block moves in the descent phase.

    Args:
        workload: Workload object containing transaction data
        num_seqs: Used to scale the beam width (effort factor)

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    # --- Configuration ---
    # Fixed Beam Width for stability
    BEAM_WIDTH = int(max(15, num_seqs * 2.8))

    # Sampling
    SAMPLES_PER_NODE = 24

    # Lookahead: Increased to top 6 critical tasks
    LOOKAHEAD_LPT_CANDIDATES = 6

    # ILS
    ILS_CYCLES = 10

    num_txns = workload.num_txns

    # --- Precompute Heuristics ---
    txn_durations = {t: workload.get_opt_seq_cost([t]) for t in range(num_txns)}
    sorted_lpt = sorted(range(num_txns), key=lambda t: txn_durations[t], reverse=True)

    # --- Beam Search ---

    # Initialization
    start_candidates = set()
    start_candidates.update(sorted_lpt[:BEAM_WIDTH])
    # Add random starts to ensure diversity
    attempts = 0
    while len(start_candidates) < BEAM_WIDTH * 2 and attempts < num_txns:
        start_candidates.add(random.randint(0, num_txns - 1))
        attempts += 1

    beam = []
    for t in start_candidates:
        rem = set(range(num_txns))
        rem.remove(t)
        cost = txn_durations[t]
        beam.append({
            'cost': cost,
            'total_dur': cost,
            'seq': [t],
            'rem': rem,
            'score': 0
        })

    # Initial sort and prune
    beam.sort(key=lambda x: (x['score'], -x['total_dur']))
    beam = beam[:BEAM_WIDTH]

    # Main Beam Loop
    for _ in range(num_txns - 1):
        candidates = []

        for parent in beam:
            rem_list = list(parent['rem'])
            if not rem_list: continue

            # --- 1. Hybrid Sampling ---
            samples = set()

            # A. Deterministic LPT (Top 2)
            lpt_picked = 0
            for t in sorted_lpt:
                if t in parent['rem']:
                    samples.add(t)
                    lpt_picked += 1
                    if lpt_picked >= 2: break

            # B. Quartic Weighted Random
            needed = SAMPLES_PER_NODE - len(samples)
            if needed > 0:
                pool = [x for x in rem_list if x not in samples]
                if pool:
                    if len(pool) <= needed:
                        samples.update(pool)
                    else:
                        # Quartic weights (d^4) to aggressively favor heavy transactions
                        weights = [txn_durations[x]**4 for x in pool]
                        try:
                            # random.choices allows replacement, convert to set to dedupe
                            picks = set(random.choices(pool, weights=weights, k=needed * 2))
                            samples.update(picks)
                        except ValueError:
                            pass

                        # Backfill if duplicates reduced count
                        if len(samples) < SAMPLES_PER_NODE:
                            rem_needed = SAMPLES_PER_NODE - len(samples)
                            others = [x for x in pool if x not in samples]
                            if others:
                                samples.update(random.sample(others, min(len(others), rem_needed)))

            # --- 2. Base Evaluation ---
            for t in samples:
                new_seq = parent['seq'] + [t]
                new_cost = workload.get_opt_seq_cost(new_seq)
                new_total_dur = parent['total_dur'] + txn_durations[t]

                # Base Score: Density (Cost - TotalDuration)
                base_score = new_cost - new_total_dur

                candidates.append({
                    'cost': new_cost,
                    'total_dur': new_total_dur,
                    'seq': new_seq,
                    'rem': parent['rem'], # Ref
                    'added': t,
                    'base_score': base_score
                })

        # --- 3. Lookahead Selection ---
        candidates.sort(key=lambda x: (x['base_score'], -x['total_dur']))

        # Deduplicate states
        lookahead_pool = []
        seen_states = set()
        pool_limit = int(BEAM_WIDTH * 1.5)

        for cand in candidates:
            new_rem = cand['rem'].copy()
            new_rem.remove(cand['added'])
            state_key = frozenset(new_rem)

            if state_key not in seen_states:
                seen_states.add(state_key)
                cand['new_rem'] = new_rem
                lookahead_pool.append(cand)
                if len(lookahead_pool) >= pool_limit:
                    break

        # --- 4. Multi-Target Optimistic Lookahead ---
        for cand in lookahead_pool:
            rem = cand['new_rem']
            if not rem:
                cand['final_score'] = cand['base_score']
                continue

            # Identify top N LPT items in remainder
            targets = []
            count = 0
            for t in sorted_lpt:
                if t in rem:
                    targets.append(t)
                    count += 1
                    if count >= LOOKAHEAD_LPT_CANDIDATES: break

            if targets:
                # Can we fit any of the top items with minimal delay?
                best_la_score = float('inf')
                for next_t in targets:
                    la_seq = cand['seq'] + [next_t]
                    la_cost = workload.get_opt_seq_cost(la_seq)
                    la_total = cand['total_dur'] + txn_durations[next_t]
                    score = la_cost - la_total
                    if score < best_la_score:
                        best_la_score = score
                cand['final_score'] = best_la_score
            else:
                cand['final_score'] = cand['base_score']

        # --- 5. Final Beam Selection ---
        lookahead_pool.sort(key=lambda x: (x['final_score'], -x['total_dur']))

        new_beam = []
        # Enforce diversity (limit children per parent isn't strictly tracked here
        # but deduplication handles state diversity)
        for cand in lookahead_pool[:BEAM_WIDTH]:
            new_beam.append({
                'cost': cand['cost'],
                'total_dur': cand['total_dur'],
                'seq': cand['seq'],
                'rem': cand['new_rem']
            })

        if not new_beam: break
        beam = new_beam

    # Best Beam Result
    best_node = min(beam, key=lambda x: x['cost'])
    best_cost = best_node['cost']
    best_seq = best_node['seq']

    # --- Iterated Local Search (ILS) ---

    def run_local_search(seq, current_cost):
        """Extended Local Search: Swaps, Insertions, and Small Block Moves."""
        improved = True
        while improved:
            improved = False

            # 1. Swaps (Window 5)
            for i in range(len(seq) - 1):
                for offset in range(1, 6):
                    j = i + offset
                    if j >= len(seq): break
                    seq[i], seq[j] = seq[j], seq[i]
                    c = workload.get_opt_seq_cost(seq)
                    if c < current_cost:
                        current_cost = c
                        improved = True
                    else:
                        seq[i], seq[j] = seq[j], seq[i]
                if improved: break
            if improved: continue

            # 2. Insertions (Window 8)
            w_ins = 8
            for i in range(len(seq)):
                start = max(0, i - w_ins)
                end = min(len(seq), i + w_ins)
                if start >= end: continue
                curr = seq[i]
                temp = seq[:i] + seq[i+1:]

                # Try inserting 'curr' elsewhere in window
                for k in range(start, end):
                    if k == i: continue
                    cand = temp[:k] + [curr] + temp[k:]
                    c = workload.get_opt_seq_cost(cand)
                    if c < current_cost:
                        current_cost = c
                        seq = cand
                        improved = True
                        break
                if improved: break
            if improved: continue

            # 3. Small Block Moves (Block Size 2, Window 6)
            # Moves pairs of adjacent transactions.
            # This helps move coupled dependencies together.
            if len(seq) > 4:
                w_blk = 6
                for i in range(len(seq) - 1):
                    block = seq[i:i+2]
                    temp = seq[:i] + seq[i+2:]

                    start = max(0, i - w_blk)
                    end = min(len(temp) + 1, i + w_blk)

                    for k in range(start, end):
                        if abs(k - i) < 2: continue # Skip close positions
                        cand = temp[:k] + block + temp[k:]
                        c = workload.get_opt_seq_cost(cand)
                        if c < current_cost:
                            current_cost = c
                            seq = cand
                            improved = True
                            break
                    if improved: break

        return seq, current_cost

    def perturb(seq):
        """Structural perturbation."""
        n = len(seq)
        new_seq = seq[:]
        mode = random.random()

        if mode < 0.4:
            # Ruin and Recreate (LNS)
            k = random.randint(3, 7)
            removed = []
            if n > k:
                indices = sorted(random.sample(range(n), k), reverse=True)
                for idx in indices:
                    removed.append(new_seq.pop(idx))

                # Recreate: Insert heavy items first (Greedy Best-Fit)
                removed.sort(key=lambda t: txn_durations[t], reverse=True)
                for t in removed:
                    best_pos = -1
                    best_c = float('inf')
                    for pos in range(len(new_seq) + 1):
                        cand = new_seq[:pos] + [t] + new_seq[pos:]
                        c = workload.get_opt_seq_cost(cand)
                        if c < best_c:
                            best_c = c
                            best_pos = pos
                    new_seq.insert(best_pos, t)

        elif mode < 0.7:
            # Block Move (Random)
            block_size = random.randint(3, 8)
            if n > block_size:
                start = random.randint(0, n - block_size)
                block = new_seq[start : start + block_size]
                remaining = new_seq[:start] + new_seq[start + block_size:]
                insert_pos = random.randint(0, len(remaining))
                new_seq = remaining[:insert_pos] + block + remaining[insert_pos:]

        else:
            # Multi-Swap
            for _ in range(3):
                if n > 1:
                    i, j = random.sample(range(n), 2)
                    new_seq[i], new_seq[j] = new_seq[j], new_seq[i]

        return new_seq

    # 1. Initial Descent
    best_seq, best_cost = run_local_search(best_seq, best_cost)

    # 2. ILS Loop
    curr_seq = best_seq[:]
    curr_cost = best_cost

    for _ in range(ILS_CYCLES):
        p_seq = perturb(curr_seq)
        p_seq, p_cost = run_local_search(p_seq, workload.get_opt_seq_cost(p_seq))

        if p_cost < best_cost:
            best_cost = p_cost
            best_seq = p_seq
            curr_seq = p_seq
            curr_cost = p_cost
        elif p_cost < curr_cost:
            curr_seq = p_seq
            curr_cost = p_cost

    return best_cost, best_seq


def get_random_costs():
    """
    Evaluate scheduling algorithm on three different workloads.

    Returns:
        Tuple of (total_makespan, list_of_schedules, execution_time)
    """
    start_time = time.time()

    effort_level = 12

    # Workload 1: Complex mixed read/write transactions
    workload1 = Workload(WORKLOAD_1)
    makespan1, schedule1 = get_best_schedule(workload1, effort_level)

    # Workload 2: Simple read-then-write pattern
    workload2 = Workload(WORKLOAD_2)
    makespan2, schedule2 = get_best_schedule(workload2, effort_level)

    # Workload 3: Minimal read/write operations
    workload3 = Workload(WORKLOAD_3)
    makespan3, schedule3 = get_best_schedule(workload3, effort_level)

    total_makespan = makespan1 + makespan2 + makespan3
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