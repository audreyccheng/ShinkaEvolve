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
    Get optimal schedule using Beam Search with Work-Density scoring.

    The algorithm uses a beam search where candidates are evaluated not just by
    total makespan, but by a density metric (Makespan - Gamma * Work_Done).
    This encourages the scheduler to tackle 'expensive' (long) transactions early,
    avoiding the "Shortest Processing Time" bias of pure greedy makespan minimization.

    Args:
        workload: Workload object containing transaction data
        num_seqs: Hint for computational budget (used to scale beam width)

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    num_txns = workload.num_txns

    # 1. Metric Precomputation
    # Get raw costs (lengths) for density calculation
    raw_costs = {}
    for i in range(num_txns):
        try:
            # txns[i] -> (info_tuple, ops...)
            # info_tuple -> (id, type, ?, cost)
            raw_costs[i] = workload.txns[i][0][3]
        except (IndexError, TypeError, AttributeError):
            raw_costs[i] = 1.0

    # Sorted indices for LPT (Longest Processing Time) heuristics
    lpt_indices = sorted(raw_costs.keys(), key=lambda k: raw_costs[k], reverse=True)

    # 2. Search Parameters
    # Beam Width: Number of active schedules to track
    # Scaled slightly based on input hint, but pinned higher for quality
    BEAM_WIDTH = max(16, int(num_seqs * 2))

    # Expansion Params
    HEURISTIC_COUNT = 10  # Always check top 10 longest remaining
    RANDOM_COUNT = 6      # Check 6 random others

    # Scoring Param: Higher gamma prefers packing work early
    GAMMA = 1.3

    # 3. Initialization
    beam = []

    # Seed with diverse starts: Top LPTs + Randoms
    seeds = set(lpt_indices[:BEAM_WIDTH])
    seeds.update(random.sample(range(num_txns), min(num_txns, BEAM_WIDTH)))

    for t in seeds:
        seq = [t]
        cost = workload.get_opt_seq_cost(seq)
        raw_work = raw_costs[t]

        # Priority: Lower is better
        # We subtract work done to prioritize "heavy" partial schedules
        score = cost - (GAMMA * raw_work)

        rem = set(range(num_txns))
        rem.remove(t)

        beam.append({
            'seq': seq,
            'cost': cost,
            'score': score,
            'raw_work': raw_work,
            'rem': rem
        })

    # Initial prune
    beam.sort(key=lambda x: x['score'])
    beam = beam[:BEAM_WIDTH]

    # 4. Beam Search Loop
    for _ in range(num_txns - 1):
        candidates = []

        for item in beam:
            curr_rem = item['rem']
            if not curr_rem:
                continue

            # Selection Strategy
            to_try = set()

            # A) Heuristic LPT
            added = 0
            for t in lpt_indices:
                if t in curr_rem:
                    to_try.add(t)
                    added += 1
                    if added >= HEURISTIC_COUNT:
                        break

            # B) Random Sampling
            needed = RANDOM_COUNT
            curr_rem_list = list(curr_rem)

            if len(curr_rem_list) <= (len(to_try) + needed):
                 to_try.update(curr_rem_list)
            else:
                 # Efficiently sample without converting set to list again if possible
                 # Just sample from list and check existence
                 samples = random.sample(curr_rem_list, min(len(curr_rem_list), needed * 2))
                 for s in samples:
                     if len(to_try) >= (HEURISTIC_COUNT + needed):
                         break
                     to_try.add(s)

            # Evaluation
            base_seq = item['seq']
            base_work = item['raw_work']

            for t in to_try:
                new_seq = base_seq + [t]

                # Simulation Cost (Makespan)
                new_cost = workload.get_opt_seq_cost(new_seq)

                new_work = base_work + raw_costs[t]
                new_score = new_cost - (GAMMA * new_work)

                new_rem = curr_rem.copy()
                new_rem.remove(t)

                candidates.append({
                    'seq': new_seq,
                    'cost': new_cost, # True objective
                    'score': new_score, # Guidance objective
                    'raw_work': new_work,
                    'rem': new_rem
                })

        if not candidates:
            break

        # Pruning with Diversity
        # Sort by guided score
        candidates.sort(key=lambda x: x['score'])

        # Keep best K deterministically
        k_best = int(BEAM_WIDTH * 0.6)
        next_beam = candidates[:k_best]

        # Fill rest from a larger pool to maintain diversity
        remaining_slots = BEAM_WIDTH - len(next_beam)
        if remaining_slots > 0:
            pool = candidates[k_best : k_best * 5] # Look deeper
            if len(pool) <= remaining_slots:
                next_beam.extend(pool)
            else:
                next_beam.extend(random.sample(pool, remaining_slots))

        beam = next_beam

    # 5. Final Selection
    # Sort by actual Makespan (cost), not the guided score
    beam.sort(key=lambda x: x['cost'])
    best_result = beam[0]

    return best_result['cost'], best_result['seq']


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