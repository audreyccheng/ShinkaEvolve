# EVOLVE-BLOCK-START
"""Transaction scheduling algorithm for optimizing makespan across multiple workloads"""

import time
import random
import math
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
    Hybrid Scheduling Algorithm: Adaptive Beam Search + Multi-Segment Ruin & Recreate.

    1. Constructive Phase:
       - Beam Search with Adaptive Gamma (1.5 -> 1.0) based on progress.
       - "Zero-Cost Bonus" for perfect parallelism candidates.

    2. Refinement Phase (ILS):
       - Multi-segment Ruin: Removes disjoint blocks to allow shuffling dependencies.
       - Recreate: Greedy Best-Fit insertion.
       - Local Search: Simulated Annealing for local polish.
    """
    num_txns = workload.num_txns

    # --- Precomputation ---
    txn_lengths = {}
    for i in range(num_txns):
        try:
            txn_lengths[i] = workload.txns[i][0][3]
        except (IndexError, TypeError, AttributeError):
            txn_lengths[i] = 1.0

    lpt_indices = sorted(txn_lengths.keys(), key=lambda k: txn_lengths[k], reverse=True)

    # --- 1. Constructive Phase: Adaptive Beam Search ---
    BEAM_WIDTH = max(8, int(num_seqs * 1.5))

    # Initial seeds
    seeds = set(lpt_indices[:BEAM_WIDTH])
    if len(seeds) < BEAM_WIDTH:
        rem = BEAM_WIDTH - len(seeds)
        pool = [x for x in range(num_txns) if x not in seeds]
        seeds.update(random.sample(pool, min(len(pool), rem)))

    beam = []
    # Initial Gamma
    gamma = 1.5

    for t in seeds:
        seq = [t]
        cost = workload.get_opt_seq_cost(seq)
        work = txn_lengths[t]
        score = cost - (gamma * work)
        rem = set(range(num_txns))
        rem.remove(t)
        beam.append({'seq': seq, 'cost': cost, 'score': score, 'work': work, 'rem': rem})

    beam.sort(key=lambda x: x['score'])
    beam = beam[:BEAM_WIDTH]

    for step in range(num_txns - 1):
        candidates = []

        # Adaptive Gamma Decay: 1.5 -> 1.0
        # step goes from 0 to num_txns-2. Progress roughly 0 to 1.
        progress = step / max(1, num_txns - 1)
        gamma = 1.5 - (0.5 * progress)

        for parent in beam:
            if not parent['rem']: continue

            # Smart Candidate Selection
            to_eval = set()
            rem_list = list(parent['rem'])

            # Top LPT
            added = 0
            for t in lpt_indices:
                if t in parent['rem']:
                    to_eval.add(t)
                    added += 1
                    if added >= 5: break

            # Random
            needed = 4
            if len(rem_list) > len(to_eval):
                pool = [x for x in rem_list if x not in to_eval]
                to_eval.update(random.sample(pool, min(len(pool), needed)))

            base_cost = parent['cost']

            for t in to_eval:
                new_seq = parent['seq'] + [t]
                new_cost = workload.get_opt_seq_cost(new_seq)
                new_work = parent['work'] + txn_lengths[t]

                # Base score
                new_score = new_cost - (gamma * new_work)

                # Zero-Cost Bonus: If cost doesn't increase, reward heavily
                if new_cost <= base_cost + 0.0001:
                    new_score -= (txn_lengths[t] * 3.0)

                new_rem = parent['rem'].copy()
                new_rem.remove(t)

                candidates.append({
                    'seq': new_seq,
                    'cost': new_cost,
                    'score': new_score,
                    'work': new_work,
                    'rem': new_rem
                })

        if not candidates: break

        candidates.sort(key=lambda x: x['score'])
        beam = candidates[:BEAM_WIDTH]

    beam.sort(key=lambda x: x['cost'])
    best_candidate = beam[0]

    current_schedule = best_candidate['seq']
    current_cost = best_candidate['cost']

    # --- 2. Refinement Phase: Multi-Segment Ruin & Recreate ---

    best_schedule = list(current_schedule)
    best_cost = current_cost

    # Only run refinement if workload is large enough
    if num_txns > 5:
        ILS_CYCLES = 4
        SA_STEPS = 50

        for cycle in range(ILS_CYCLES):
            work_seq = list(current_schedule)

            # Multi-Segment Ruin: Remove 2 separate small blocks
            removed_txns = []

            # Block 1
            if len(work_seq) > 2:
                bs1 = random.randint(1, max(2, int(num_txns * 0.1)))
                idx1 = random.randint(0, len(work_seq) - bs1)
                removed_txns.extend(work_seq[idx1 : idx1 + bs1])
                del work_seq[idx1 : idx1 + bs1]

            # Block 2
            if len(work_seq) > 2:
                bs2 = random.randint(1, max(2, int(num_txns * 0.1)))
                idx2 = random.randint(0, len(work_seq) - bs2)
                removed_txns.extend(work_seq[idx2 : idx2 + bs2])
                del work_seq[idx2 : idx2 + bs2]

            # Shuffle for re-insertion
            random.shuffle(removed_txns)

            # Recreate: Best-Fit
            for txn in removed_txns:
                best_pos = -1
                best_incr = float('inf')

                for pos in range(len(work_seq) + 1):
                    work_seq.insert(pos, txn)
                    c = workload.get_opt_seq_cost(work_seq)
                    if c < best_incr:
                        best_incr = c
                        best_pos = pos
                    del work_seq[pos]

                work_seq.insert(best_pos, txn)

            current_schedule = work_seq
            current_cost = best_incr

            if current_cost < best_cost:
                best_cost = current_cost
                best_schedule = list(current_schedule)
            else:
                # Restart chance from global best
                if random.random() < 0.2:
                     current_schedule = list(best_schedule)
                     current_cost = best_cost

            # Local Search (SA)
            temp = max(0.5, current_cost * 0.01)
            for _ in range(SA_STEPS):
                neighbor = list(current_schedule)
                op = random.random()
                if op < 0.5: # Swap
                    i, j = random.sample(range(len(neighbor)), 2)
                    neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                else: # Insert
                    i = random.randint(0, len(neighbor)-1)
                    val = neighbor.pop(i)
                    j = random.randint(0, len(neighbor))
                    neighbor.insert(j, val)

                n_cost = workload.get_opt_seq_cost(neighbor)
                delta = n_cost - current_cost

                if delta < 0 or (temp > 1e-4 and random.random() < math.exp(-delta/temp)):
                    current_schedule = neighbor
                    current_cost = n_cost
                    if current_cost < best_cost:
                        best_cost = current_cost
                        best_schedule = list(current_schedule)
                temp *= 0.9

    return best_cost, best_schedule


def get_random_costs():
    """
    Evaluate scheduling algorithm on three different workloads.

    Returns:
        Tuple of (total_makespan, list_of_schedules, execution_time)
    """
    start_time = time.time()

    # Workload 1: Complex mixed read/write transactions
    workload1 = Workload(WORKLOAD_1)
    makespan1, schedule1 = get_best_schedule(workload1, 10)
    cost1 = workload1.get_opt_seq_cost(schedule1)

    # Workload 2: Simple read-then-write pattern
    workload2 = Workload(WORKLOAD_2)
    makespan2, schedule2 = get_best_schedule(workload2, 10)
    cost2 = workload2.get_opt_seq_cost(schedule2)

    # Workload 3: Minimal read/write operations
    workload3 = Workload(WORKLOAD_3)
    makespan3, schedule3 = get_best_schedule(workload3, 10)
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