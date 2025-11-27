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
    Get optimal schedule using Lookahead Beam Search with Stochastic Sampling and ILS.

    Key Features:
    1. Lookahead: Evaluates candidates not just on the immediate move, but on the
       feasibility of scheduling the next largest task (LPT).
    2. Weighted Sampling: Biases exploration towards longer transactions.
    3. Iterated Local Search: Uses perturbation to escape local optima after beam search.

    Args:
        workload: Workload object containing transaction data
        num_seqs: Used to scale the beam width

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    # --- Parameters ---
    BEAM_WIDTH = int(max(10, num_seqs * 2.5))
    SAMPLES_PER_NODE = 16
    LOOKAHEAD_FACTOR = 2  # Evaluate lookahead for Top N * Factor candidates

    num_txns = workload.num_txns

    # --- Precompute Heuristics ---
    txn_durations = {t: workload.get_opt_seq_cost([t]) for t in range(num_txns)}
    sorted_lpt = sorted(range(num_txns), key=lambda t: txn_durations[t], reverse=True)

    # --- Initialization ---
    start_candidates = set()
    start_candidates.update(sorted_lpt[:BEAM_WIDTH])

    # Fill with random if needed
    attempts = 0
    while len(start_candidates) < BEAM_WIDTH * 2 and attempts < num_txns * 2:
        start_candidates.add(random.randint(0, num_txns - 1))
        attempts += 1

    beam = []
    for t in start_candidates:
        rem = set(range(num_txns))
        rem.remove(t)
        cost = txn_durations[t]
        # Score: Cost - TotalDuration (Work Density). Lower is better.
        beam.append({
            'cost': cost,
            'total_dur': cost,
            'seq': [t],
            'rem': rem,
            'base_score': cost - cost
        })

    beam.sort(key=lambda x: (x['base_score'], -x['total_dur']))
    beam = beam[:BEAM_WIDTH]

    # --- Beam Search Loop ---
    for _ in range(num_txns - 1):
        candidates = []

        for parent in beam:
            rem_list = list(parent['rem'])
            if not rem_list: continue

            # --- Hybrid Sampling ---
            samples = set()

            # 1. Deterministic LPT (Top 3)
            lpt_added = 0
            for t in sorted_lpt:
                if t in parent['rem']:
                    samples.add(t)
                    lpt_added += 1
                    if lpt_added >= 3:
                        break

            # 2. Weighted Random Sampling
            needed = SAMPLES_PER_NODE - len(samples)
            if needed > 0:
                pool = [t for t in rem_list if t not in samples]
                if pool:
                    if len(pool) <= needed:
                        samples.update(pool)
                    else:
                        weights = [txn_durations[t] for t in pool]
                        for _ in range(needed * 2): # Oversample to handle collisions
                            if len(samples) >= SAMPLES_PER_NODE: break
                            pick = random.choices(pool, weights=weights, k=1)[0]
                            samples.add(pick)

            # --- Base Evaluation ---
            for t in samples:
                new_seq = parent['seq'] + [t]
                new_cost = workload.get_opt_seq_cost(new_seq)
                new_total_dur = parent['total_dur'] + txn_durations[t]

                # Base Score: Density
                base_score = new_cost - new_total_dur

                candidates.append({
                    'cost': new_cost,
                    'total_dur': new_total_dur,
                    'seq': new_seq,
                    'rem': parent['rem'], # Ref only
                    'added': t,
                    'base_score': base_score
                })

        # --- Lookahead Stage ---
        # 1. Filter candidates by base score to keep manageable set for lookahead
        candidates.sort(key=lambda x: (x['base_score'], -x['total_dur']))

        # Deduplicate states before lookahead
        unique_cands = []
        seen_states = set()
        for cand in candidates:
            # New rem set
            new_rem = cand['rem'].copy()
            new_rem.remove(cand['added'])

            # State key
            state_key = frozenset(new_rem)
            if state_key not in seen_states:
                seen_states.add(state_key)
                cand['new_rem'] = new_rem
                unique_cands.append(cand)

        # Limit pool for expensive lookahead
        lookahead_pool = unique_cands[:BEAM_WIDTH * LOOKAHEAD_FACTOR]

        # 2. Perform Lookahead
        # Tentatively add the largest remaining transaction to see if it fits well
        for cand in lookahead_pool:
            if not cand['new_rem']:
                cand['final_score'] = cand['base_score']
                continue

            # Find max duration item in remaining
            next_heavy = None
            for t in sorted_lpt:
                if t in cand['new_rem']:
                    next_heavy = t
                    break

            if next_heavy is not None:
                la_seq = cand['seq'] + [next_heavy]
                la_cost = workload.get_opt_seq_cost(la_seq)
                la_total = cand['total_dur'] + txn_durations[next_heavy]
                cand['final_score'] = la_cost - la_total
            else:
                cand['final_score'] = cand['base_score']

        # 3. Final Selection
        lookahead_pool.sort(key=lambda x: (x['final_score'], -x['total_dur']))

        new_beam = []
        for cand in lookahead_pool[:BEAM_WIDTH]:
            new_beam.append({
                'cost': cand['cost'],
                'total_dur': cand['total_dur'],
                'seq': cand['seq'],
                'rem': cand['new_rem'],
                'base_score': cand['base_score']
            })

        if not new_beam:
            break
        beam = new_beam

    # Select best
    best_result = min(beam, key=lambda x: x['cost'])
    best_cost = best_result['cost']
    best_seq = best_result['seq']

    # --- Iterated Local Search (ILS) ---
    def refine(seq, current_cost):
        """Greedy insertion refinement"""
        w_size = 8
        improved = True
        while improved:
            improved = False
            for i in range(len(seq)):
                # Define window
                start = max(0, i - w_size)
                end = min(len(seq), i + w_size)

                # Try moving seq[i] to positions in window
                txn = seq[i]
                temp = seq[:i] + seq[i+1:]

                for k in range(start, end): # position in temp
                    cand = temp[:k] + [txn] + temp[k:]
                    c = workload.get_opt_seq_cost(cand)
                    if c < current_cost:
                        current_cost = c
                        seq = cand
                        improved = True
                        break
                if improved: break
        return seq, current_cost

    # Initial Descent
    best_seq, best_cost = refine(best_seq, best_cost)

    # Perturbation (Kick)
    for _ in range(2): # Try 2 random kicks
        p_seq = best_seq[:]
        if len(p_seq) > 2:
            idx1, idx2 = random.sample(range(len(p_seq)), 2)
            p_seq[idx1], p_seq[idx2] = p_seq[idx2], p_seq[idx1]

            # Repair
            p_seq, p_cost = refine(p_seq, workload.get_opt_seq_cost(p_seq))

            if p_cost < best_cost:
                best_cost = p_cost
                best_seq = p_seq

    return best_cost, best_seq


def get_random_costs():
    """
    Evaluate scheduling algorithm on three different workloads.

    Returns:
        Tuple of (total_makespan, list_of_schedules, execution_time)
    """
    start_time = time.time()

    # We use a slightly higher num_seqs to enable a wider beam for better results
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