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
    Get optimal schedule using Quintic-Weighted Beam Search and Block-Move ILS.

    Algorithm:
    1. Beam Search:
       - Expansion: Hybrid sampling (Anchor Top-3 LPT + Quintic Weighted Random).
       - Weighting: d^5 weighting acts almost deterministically for distinct sizes, 
         prioritizing "Big Rocks" heavily while maintaining slight diversity.
       - Lookahead: Evaluates top-5 largest remaining transactions (Optimistic scoring).
    2. ILS (Ruin & Recreate):
       - Refinement: Swaps -> Block Moves (size 2-3) -> Insertions.
       - Perturbation: Random ruin of 4-8 items.
       - Re-insertion: Greedy LPT (Longest Processing Time).

    Args:
        workload: Workload object
        num_seqs: Scaling factor for beam width

    Returns:
        Tuple of (lowest makespan, schedule)
    """
    # --- Parameters ---
    BEAM_WIDTH = int(max(15, num_seqs * 2.8))
    SAMPLES_PER_NODE = 24
    MAX_CHILDREN = 4

    # Lookahead
    LOOKAHEAD_FACTOR = 2.0
    LOOKAHEAD_TARGETS = 5

    # ILS
    ILS_CYCLES = 12

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

            # 1. Deterministic LPT (Top 3)
            # Anchor the search with the absolute best local candidates
            lpt_count = 0
            for t in sorted_lpt:
                if t in parent['rem']:
                    samples.add(t)
                    lpt_count += 1
                    if lpt_count >= 3: break

            # 2. Quintic (d^5) Weighted Random
            needed = SAMPLES_PER_NODE - len(samples)
            if needed > 0:
                pool = [x for x in rem_list if x not in samples]
                if pool:
                    if len(pool) <= needed:
                        samples.update(pool)
                    else:
                        # Quintic weights: Extremely aggressive bias towards long transactions
                        weights = [txn_durations[x]**5 for x in pool]
                        try:
                            chosen = random.choices(pool, weights=weights, k=needed * 2)
                            samples.update(chosen)
                        except ValueError:
                            pass
                        
                        if len(samples) < SAMPLES_PER_NODE:
                             needed_now = SAMPLES_PER_NODE - len(samples)
                             rem_pool = [x for x in pool if x not in samples]
                             if rem_pool:
                                 samples.update(random.sample(rem_pool, min(len(rem_pool), needed_now)))

            # --- Base Evaluation ---
            for t in samples:
                new_seq = parent['seq'] + [t]
                new_cost = workload.get_opt_seq_cost(new_seq)
                new_total_dur = parent['total_dur'] + txn_durations[t]
                # Score: Work Density (Cost - TotalDuration)
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

        # Multi-Target Lookahead
        for cand in lookahead_pool:
            rem_set = cand['new_rem']
            if not rem_set:
                cand['final_score'] = cand['base_score']
                continue

            # Identify targets (Top 5 LPT)
            targets = []
            count = 0
            for t in sorted_lpt:
                if t in rem_set:
                    targets.append(t)
                    count += 1
                    if count >= LOOKAHEAD_TARGETS: break

            if targets:
                # "Optimistic" Lookahead: If we can fit ANY of the big rocks well next, this is a good path
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

        if not new_beam: break
        beam = new_beam

    # Best Result
    best_result = min(beam, key=lambda x: x['cost'])
    best_cost = best_result['cost']
    best_seq = best_result['seq']

    # --- ILS ---

    def refine(seq, cost):
        """Extended Local Search: Swaps -> Block Moves -> Insertions."""
        improved = True
        while improved:
            improved = False

            # 1. Swaps
            for i in range(len(seq) - 1):
                for offset in range(1, 6):
                    j = i + offset
                    if j >= len(seq): break
                    seq[i], seq[j] = seq[j], seq[i]
                    c = workload.get_opt_seq_cost(seq)
                    if c < cost:
                        cost = c
                        improved = True
                    else:
                        seq[i], seq[j] = seq[j], seq[i]
                if improved: break
            if improved: continue

            # 2. Block Moves (Size 2-3)
            # Moves small contiguous blocks to handle dependent chains
            if len(seq) > 5:
                w_block = 8
                block_sizes = [2, 3]
                for b_size in block_sizes:
                    for i in range(len(seq) - b_size):
                        block = seq[i : i+b_size]
                        rem_seq = seq[:i] + seq[i+b_size:]
                        
                        start = max(0, i - w_block)
                        end = min(len(rem_seq) + 1, i + w_block)
                        
                        found_better = False
                        for k in range(start, end):
                            # Skip original position approx
                            if abs(k - i) < 2: continue
                            
                            cand = rem_seq[:k] + block + rem_seq[k:]
                            c = workload.get_opt_seq_cost(cand)
                            if c < cost:
                                cost = c
                                seq = cand
                                improved = True
                                found_better = True
                                break
                        if found_better: break
                    if improved: break
            if improved: continue

            # 3. Insertions
            w_ins = 8
            for i in range(len(seq)):
                start = max(0, i - w_ins)
                end = min(len(seq), i + w_ins)
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

    # Initial Descent
    best_seq, best_cost = refine(best_seq, best_cost)

    curr_seq = best_seq[:]
    curr_cost = best_cost

    # Ruin & Recreate Loop
    for _ in range(ILS_CYCLES):
        p_seq = curr_seq[:]

        # Ruin
        if len(p_seq) > 10:
            # Remove 4 to 8 items
            ruin_size = random.randint(4, 8)
            removed = []
            for _ in range(ruin_size):
                if not p_seq: break
                idx = random.randint(0, len(p_seq) - 1)
                removed.append(p_seq.pop(idx))

            # Recreate: Best-Fit (Heavy First)
            # Sorting by duration ensures we place big rocks first
            removed.sort(key=lambda t: txn_durations[t], reverse=True)

            for t in removed:
                best_pos = -1
                best_c = float('inf')
                # Check all positions
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

        # Strict Acceptance (Greedy Descent)
        # Previous successful generations used simple greedy descent here, 
        # or very tight threshold. Let's stick to simple greedy to ensure convergence.
        if p_cost < best_cost:
            best_cost = p_cost
            best_seq = p_seq
            curr_seq = p_seq
            curr_cost = p_cost
        elif p_cost < curr_cost: # Accept neutral or better moves for diversity
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

    workload1 = Workload(WORKLOAD_1)
    makespan1, schedule1 = get_best_schedule(workload1, effort_level)

    workload2 = Workload(WORKLOAD_2)
    makespan2, schedule2 = get_best_schedule(workload2, effort_level)

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