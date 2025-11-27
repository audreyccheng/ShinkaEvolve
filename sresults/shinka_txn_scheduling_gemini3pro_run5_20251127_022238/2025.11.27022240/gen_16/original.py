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
    Hybrid Scheduling Algorithm:
    1. Constructive Phase: Beam Search with Work-Density scoring to build a high-quality initial schedule.
    2. Refinement Phase: Simulated Annealing to perform local search (swaps/inserts) on the constructed schedule.

    Args:
        workload: Workload object containing transaction data
        num_seqs: Hint for computational budget

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """

    # --- 0. Helper & Precomputation ---
    num_txns = workload.num_txns

    # Extract transaction costs (lengths) for heuristics
    # Structure: workload.txns[i][0] is metadata, index 3 is usually length
    txn_lengths = {}
    for i in range(num_txns):
        try:
            txn_lengths[i] = workload.txns[i][0][3]
        except (IndexError, TypeError, AttributeError):
            txn_lengths[i] = 1.0

    # Sort indices by length descending (LPT - Longest Processing Time)
    lpt_sorted_indices = sorted(txn_lengths.keys(), key=lambda k: txn_lengths[k], reverse=True)

    # --- 1. Constructive Phase: Beam Search ---
    # Parameters
    # We use a narrower beam than pure beam search to save time for the annealing phase
    BEAM_WIDTH = 4
    SAMPLE_SIZE = 8       # Candidates to evaluate per expansion
    GAMMA = 1.5           # Work-density bias: favors schedules that complete "heavy" work early

    # Initial Beam Seeding
    # Start with a mix of longest transactions and random ones for diversity
    seeds = set(lpt_sorted_indices[:BEAM_WIDTH])
    if len(seeds) < BEAM_WIDTH:
        remaining_slots = BEAM_WIDTH - len(seeds)
        seeds.update(random.sample(range(num_txns), min(num_txns - len(seeds), remaining_slots)))

    beam = []
    for t in seeds:
        seq = [t]
        cost = workload.get_opt_seq_cost(seq)
        acc_work = txn_lengths[t]
        # Scoring Metric: Cost (Makespan) penalized by Work Done
        # Lower score is better. We subtract Work Done to "reward" progress.
        score = cost - (GAMMA * acc_work)

        remaining = set(range(num_txns))
        remaining.remove(t)

        beam.append({
            'seq': seq,
            'cost': cost,
            'score': score,
            'acc_work': acc_work,
            'rem': remaining
        })

    # Sort by score and trim to beam width
    beam.sort(key=lambda x: x['score'])
    beam = beam[:BEAM_WIDTH]

    # Construction Loop (Greedy Extension)
    for _ in range(num_txns - 1):
        candidates = []

        for parent in beam:
            rem_set = parent['rem']
            if not rem_set:
                continue

            # Candidate Selection: LPT + Random
            to_eval = set()

            # A) Add heuristic candidates (Longest remaining)
            added = 0
            for t in lpt_sorted_indices:
                if t in rem_set:
                    to_eval.add(t)
                    added += 1
                    if added >= (SAMPLE_SIZE // 2):
                        break

            # B) Add random candidates to maintain diversity
            needed = SAMPLE_SIZE - len(to_eval)
            rem_list = list(rem_set)
            if needed > 0 and rem_list:
                if len(rem_list) <= needed:
                    to_eval.update(rem_list)
                else:
                    to_eval.update(random.sample(rem_list, needed))

            # Evaluate candidates
            base_seq = parent['seq']
            base_work = parent['acc_work']

            for t in to_eval:
                new_seq = base_seq + [t]

                # Run simulation to get makespan
                new_cost = workload.get_opt_seq_cost(new_seq)

                new_work = base_work + txn_lengths[t]
                new_score = new_cost - (GAMMA * new_work)

                new_rem = rem_set.copy()
                new_rem.remove(t)

                candidates.append({
                    'seq': new_seq,
                    'cost': new_cost,
                    'score': new_score,
                    'acc_work': new_work,
                    'rem': new_rem
                })

        if not candidates:
            break

        # Beam Pruning: Sort by Work-Density Score
        candidates.sort(key=lambda x: x['score'])
        beam = candidates[:BEAM_WIDTH]

    # Select best result from construction phase based on actual Cost
    beam.sort(key=lambda x: x['cost'])
    best_candidate = beam[0]

    current_schedule = best_candidate['seq']
    current_cost = best_candidate['cost']

    # --- 2. Refinement Phase: Simulated Annealing ---
    # Attempt to improve the constructed schedule by local perturbation

    MAX_ITER = 600
    # Start temp relative to cost to allow some uphill moves initially
    INITIAL_TEMP = max(1.0, current_cost * 0.05)
    COOLING_RATE = 0.99

    best_schedule = list(current_schedule)
    best_cost = current_cost

    temp = INITIAL_TEMP

    for i in range(MAX_ITER):
        # Create neighbor
        neighbor = list(current_schedule)

        # Mutation Strategy: 50% Swap, 50% Insert (Block Move)
        mutation_type = random.random()
        idx1 = random.randint(0, num_txns - 1)

        if mutation_type < 0.5:
            # Swap two transactions
            idx2 = random.randint(0, num_txns - 1)
            while idx1 == idx2:
                idx2 = random.randint(0, num_txns - 1)
            neighbor[idx1], neighbor[idx2] = neighbor[idx2], neighbor[idx1]
        else:
            # Insert: Remove txn at idx1 and insert at idx2
            val = neighbor.pop(idx1)
            # Insert range is 0 to len(neighbor) inclusive (append)
            idx2 = random.randint(0, len(neighbor))
            neighbor.insert(idx2, val)

        # Evaluate neighbor
        neighbor_cost = workload.get_opt_seq_cost(neighbor)
        delta = neighbor_cost - current_cost

        # Metropolis Acceptance Criterion
        accept = False
        if delta < 0:
            accept = True
        else:
            # Probabilistic acceptance of worse solutions
            if temp > 0.001:
                prob = math.exp(-delta / temp)
                if random.random() < prob:
                    accept = True

        if accept:
            current_schedule = neighbor
            current_cost = neighbor_cost

            # Keep track of global best found
            if current_cost < best_cost:
                best_cost = current_cost
                best_schedule = list(current_schedule)

        # Cool down
        temp *= COOLING_RATE
        if temp < 0.001:
            break

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