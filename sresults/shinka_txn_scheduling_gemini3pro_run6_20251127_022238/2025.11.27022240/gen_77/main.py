# EVOLVE-BLOCK-START
"""Transaction scheduling algorithm for optimizing makespan across multiple workloads"""

import time
import random
import sys
import os

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
    Get optimal schedule using Quartic Beam Search and Threshold-Acceptance ILS.

    Algorithm:
    1. Beam Search:
       - Expansion: Hybrid sampling with 3 Deterministic LPT anchors + Quartic Weighted Random.
       - Lookahead: Evaluates top-4 largest remaining transactions.
       - Heuristic: Minimizes "Density" (Makespan - SumOfDurations).
    2. Threshold-Acceptance ILS:
       - Ruin: Removes 5-8 random items.
       - Recreate: Greedy Best-Fit with LPT order.
       - Repair: Local search with Swaps, Block Moves, and Insertions.
       - Acceptance: Accepts slightly worse solutions (Threshold Acceptance) to escape local optima.

    Args:
        workload: Workload object containing transaction data
        num_seqs: Used to scale the beam width

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    # --- Parameters ---
    # Beam Search
    BEAM_WIDTH = int(max(16, num_seqs * 3.0)) # Increased width slightly
    SAMPLES_PER_NODE = 24
    MAX_CHILDREN = 4
    LPT_ANCHORS = 3 # Increased deterministic samples

    # Lookahead
    LOOKAHEAD_FACTOR = 2.0
    LOOKAHEAD_TARGETS = 4 # Optimized targets

    # ILS
    ILS_CYCLES = 10
    ILS_THRESHOLD_START = 0.02 # 2% degradation allowed initially

    num_txns = workload.num_txns

    # --- Precompute Heuristics ---
    txn_durations = {t: workload.get_opt_seq_cost([t]) for t in range(num_txns)}
    sorted_lpt = sorted(range(num_txns), key=lambda t: txn_durations[t], reverse=True)

    # --- Initialization ---
    start_candidates = set()
    start_candidates.update(sorted_lpt[:BEAM_WIDTH])
    attempts = 0
    while len(start_candidates) < BEAM_WIDTH * 2 and attempts < num_txns * 2:
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

    # Initial Prune
    beam.sort(key=lambda x: (x['score'], -x['total_dur']))
    beam = beam[:BEAM_WIDTH]

    # --- Beam Search Loop ---
    for _ in range(num_txns - 1):
        candidates = []

        for p_idx, parent in enumerate(beam):
            rem_list = list(parent['rem'])
            if not rem_list: continue

            # --- Hybrid Sampling ---
            samples = set()

            # 1. Deterministic LPT (Top N)
            lpt_count = 0
            for t in sorted_lpt:
                if t in parent['rem']:
                    samples.add(t)
                    lpt_count += 1
                    if lpt_count >= LPT_ANCHORS: break

            # 2. Quartic Weighted Random Sampling
            needed = SAMPLES_PER_NODE - len(samples)
            if needed > 0:
                pool = [x for x in rem_list if x not in samples]
                if pool:
                    if len(pool) <= needed:
                        samples.update(pool)
                    else:
                        # Quartic weights (d^4)
                        weights = [txn_durations[x]**4 for x in pool]
                        try:
                            chosen = random.choices(pool, weights=weights, k=needed * 2)
                            samples.update(chosen)
                        except ValueError:
                            pass
                        
                        # Backfill if duplicates reduced set size
                        if len(samples) < SAMPLES_PER_NODE:
                             rem_pool = [x for x in pool if x not in samples]
                             if rem_pool:
                                 needed_now = SAMPLES_PER_NODE - len(samples)
                                 samples.update(random.sample(rem_pool, min(len(rem_pool), needed_now)))

            # --- Base Evaluation ---
            for t in samples:
                new_seq = parent['seq'] + [t]
                new_cost = workload.get_opt_seq_cost(new_seq)
                new_total_dur = parent['total_dur'] + txn_durations[t]
                base_score = new_cost - new_total_dur

                candidates.append({
                    'cost': new_cost,
                    'total_dur': new_total_dur,
                    'seq': new_seq,
                    'rem': parent['rem'], 
                    'added': t,
                    'p_idx': p_idx,
                    'base_score': base_score
                })

        # --- Lookahead Phase ---
        candidates.sort(key=lambda x: (x['base_score'], -x['total_dur']))

        lookahead_pool = []
        seen_states = set()
        target_pool_size = int(BEAM_WIDTH * LOOKAHEAD_FACTOR)

        for cand in candidates:
            new_rem = cand['rem'].copy()
            new_rem.remove(cand['added'])
            state_key = frozenset(new_rem)

            if state_key not in seen_states:
                seen_states.add(state_key)
                cand['new_rem'] = new_rem
                lookahead_pool.append(cand)
                if len(lookahead_pool) >= target_pool_size:
                    break

        # Multi-Target Optimistic Lookahead
        for cand in lookahead_pool:
            rem_set = cand['new_rem']
            if not rem_set:
                cand['final_score'] = cand['base_score']
                continue

            targets = []
            count = 0
            for t in sorted_lpt:
                if t in rem_set:
                    targets.append(t)
                    count += 1
                    if count >= LOOKAHEAD_TARGETS: break

            if targets:
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

        # Final Selection
        lookahead_pool.sort(key=lambda x: (x['final_score'], -x['total_dur']))

        new_beam = []
        p_counts = {i: 0 for i in range(len(beam))}
        reserve = []

        for cand in lookahead_pool:
            p_idx = cand['p_idx']
            node = {
                'cost': cand['cost'],
                'total_dur': cand['total_dur'],
                'seq': cand['seq'],
                'rem': cand['new_rem']
            }

            if p_counts[p_idx] < MAX_CHILDREN:
                if len(new_beam) < BEAM_WIDTH:
                    new_beam.append(node)
                    p_counts[p_idx] += 1
                else:
                    pass
            else:
                reserve.append(node)

            if len(new_beam) >= BEAM_WIDTH and len(reserve) > BEAM_WIDTH:
                break

        if len(new_beam) < BEAM_WIDTH:
            for node in reserve:
                if len(new_beam) >= BEAM_WIDTH: break
                new_beam.append(node)

        if not new_beam:
            break
        beam = new_beam

    # Select best from beam
    best_result = min(beam, key=lambda x: x['cost'])
    best_cost = best_result['cost']
    best_seq = best_result['seq']

    # --- ILS Refinement Logic ---

    def refine(seq, cost):
        """Extended Local Search: Swaps, Block Moves, and Insertions."""
        improved = True
        while improved:
            improved = False

            # 1. Swaps (Window 6) - Increased window slightly
            for i in range(len(seq) - 1):
                for offset in range(1, 7):
                    j = i + offset
                    if j >= len(seq): break
                    seq[i], seq[j] = seq[j], seq[i]
                    c = workload.get_opt_seq_cost(seq)
                    if c < cost:
                        cost = c
                        improved = True
                    else:
                        seq[i], seq[j] = seq[j], seq[i] # Revert
                if improved: break
            if improved: continue

            # 2. Block Moves (Size 2, Window 8)
            w_block = 8
            if len(seq) > 5:
                for i in range(len(seq) - 1):
                    block = seq[i:i+2]
                    rem_seq = seq[:i] + seq[i+2:]
                    start = max(0, i - w_block)
                    end = min(len(rem_seq) + 1, i + w_block)
                    for k in range(start, end):
                        if abs(k - i) < 2: continue # Optimization: skip close moves
                        cand = rem_seq[:k] + block + rem_seq[k:]
                        c = workload.get_opt_seq_cost(cand)
                        if c < cost:
                            cost = c
                            seq = cand
                            improved = True
                            break
                    if improved: break
            if improved: continue

            # 3. Single Insertions (Window 10) - Increased window
            w_ins = 10
            for i in range(len(seq)):
                start = max(0, i - w_ins)
                end = min(len(seq), i + w_ins)
                if start >= end: continue
                
                curr = seq[i]
                temp = seq[:i] + seq[i+1:]
                for k in range(start, end):
                    if k == i: continue
                    cand = temp[:k] + [curr] + temp[k:]
                    c = workload.get_opt_seq_cost(cand)
                    if c < cost:
                        cost = c
                        seq = cand
                        improved = True
                        break
                if improved: break
        return seq, cost

    # Phase 1: Initial Descent
    best_seq, best_cost = refine(best_seq, best_cost)

    # Phase 2: Threshold-Acceptance ILS
    curr_seq = best_seq[:]
    curr_cost = best_cost

    for cycle in range(ILS_CYCLES):
        p_seq = curr_seq[:]

        # Ruin: Remove random 5-8 items
        if len(p_seq) > 8:
            ruin_size = random.randint(5, 8)
            # Ensure we don't remove more than we have
            ruin_size = min(ruin_size, len(p_seq) - 1)
            
            removed = []
            for _ in range(ruin_size):
                if not p_seq: break
                idx = random.randint(0, len(p_seq) - 1)
                removed.append(p_seq.pop(idx))

            # Recreate: Best-Fit greedy (LPT sorted)
            removed.sort(key=lambda t: txn_durations[t], reverse=True)

            for t in removed:
                best_pos = -1
                best_c = float('inf')
                # Try all valid positions
                for pos in range(len(p_seq) + 1):
                    cand = p_seq[:pos] + [t] + p_seq[pos:]
                    c = workload.get_opt_seq_cost(cand)
                    if c < best_c:
                        best_c = c
                        best_pos = pos

                if best_pos != -1:
                    p_seq.insert(best_pos, t)
                else:
                    p_seq.append(t)

        # Repair
        p_seq, p_cost = refine(p_seq, workload.get_opt_seq_cost(p_seq))

        # Threshold Acceptance
        # Allow acceptance of slightly worse solutions early in the search
        threshold_pct = ILS_THRESHOLD_START * (1.0 - (cycle / ILS_CYCLES))
        allowed_cost = curr_cost * (1.0 + threshold_pct)

        if p_cost < best_cost:
            # Always accept global best
            best_cost = p_cost
            best_seq = p_seq
            curr_seq = p_seq
            curr_cost = p_cost
        elif p_cost <= allowed_cost:
            # Accept if within threshold
            curr_seq = p_seq
            curr_cost = p_cost
        # Else: Reject, keep curr_seq (revert)

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