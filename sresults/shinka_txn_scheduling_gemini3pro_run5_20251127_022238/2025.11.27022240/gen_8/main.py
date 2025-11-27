# EVOLVE-BLOCK-START
"""Transaction scheduling algorithm for optimizing makespan across multiple workloads"""

import time
import random
import sys
import os
import heapq

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
    Get optimal schedule using an improved Beam Search algorithm.
    Includes diversity preservation and LPT-biased heuristics.

    Args:
        workload: Workload object containing transaction data
        num_seqs: Used to derive beam width and search budget

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    num_txns = workload.num_txns

    # 1. Heuristic Setup
    try:
        # (index, length)
        txn_costs = [(i, workload.txns[i][0][3]) for i in range(num_txns)]
    except (IndexError, TypeError, AttributeError):
        txn_costs = [(i, 1) for i in range(num_txns)]

    cost_map = dict(txn_costs)

    # Sort by length descending (LPT)
    txn_costs.sort(key=lambda x: x[1], reverse=True)
    sorted_txn_indices = [x[0] for x in txn_costs]

    # 2. Search Parameters
    # Increased parameters for better exploration
    BEAM_WIDTH = max(8, int(num_seqs))
    SAMPLES_PER_NODE = 16
    HEURISTIC_COUNT = 6  # Increase heuristic bias

    # 3. Initialization
    # Initialize with top LPT transactions + some random ones to seed diversity
    start_pool = set(sorted_txn_indices[:BEAM_WIDTH])
    start_pool.update(random.sample(range(num_txns), min(num_txns, BEAM_WIDTH)))

    beam = []
    for start_node in start_pool:
        seq = [start_node]
        cost = workload.get_opt_seq_cost(seq)
        remaining = set(range(num_txns))
        remaining.remove(start_node)
        beam.append({
            'cost': cost,
            'seq': seq,
            'rem': remaining
        })

    # Keep best unique starts
    beam.sort(key=lambda x: x['cost'])
    beam = beam[:BEAM_WIDTH]

    # 4. Beam Search Loop
    for step in range(num_txns - 1):
        candidates_pool = []

        for path in beam:
            current_seq = path['seq']
            remaining = path['rem']

            if not remaining:
                continue

            # Candidate Generation
            next_candidates = set()

            # A) Top Longest Remaining
            added_heuristic = 0
            for t in sorted_txn_indices:
                if t in remaining:
                    next_candidates.add(t)
                    added_heuristic += 1
                    if added_heuristic >= HEURISTIC_COUNT:
                        break

            # B) Random Sampling
            needed = SAMPLES_PER_NODE - len(next_candidates)
            rem_list = list(remaining)
            if needed > 0 and rem_list:
                pool = [x for x in rem_list if x not in next_candidates]
                if len(pool) <= needed:
                    next_candidates.update(pool)
                else:
                    next_candidates.update(random.sample(pool, needed))

            # Evaluation
            for next_txn in next_candidates:
                new_seq = current_seq + [next_txn]
                new_cost = workload.get_opt_seq_cost(new_seq)

                new_rem = remaining.copy()
                new_rem.remove(next_txn)

                candidates_pool.append({
                    'cost': new_cost,
                    'seq': new_seq,
                    'rem': new_rem
                })

        if not candidates_pool:
            break

        candidates_pool.sort(key=lambda x: x['cost'])

        # Diversity Selection:
        # Take top K_BEST deterministically
        # Take remaining slots from a random sample of the top chunk to maintain diversity
        k_best = BEAM_WIDTH // 2
        k_random = BEAM_WIDTH - k_best

        new_beam = candidates_pool[:k_best]

        remaining_candidates = candidates_pool[k_best:min(len(candidates_pool), BEAM_WIDTH * 4)]
        if remaining_candidates and k_random > 0:
            if len(remaining_candidates) <= k_random:
                new_beam.extend(remaining_candidates)
            else:
                new_beam.extend(random.sample(remaining_candidates, k_random))

        beam = new_beam

    # 5. Final Result
    # Re-sort beam by pure cost
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