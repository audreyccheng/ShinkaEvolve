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
    1. Constructive Phase: Wide Beam Search with diversity pruning to build a robust initial schedule.
    2. Refinement Phase: Simulated Annealing with advanced operators (Block Moves) to optimize clusters.

    Args:
        workload: Workload object containing transaction data
        num_seqs: Hint for computational budget

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """

    # --- 0. Helper & Precomputation ---
    num_txns = workload.num_txns

    # Extract transaction costs
    txn_lengths = {}
    for i in range(num_txns):
        try:
            txn_lengths[i] = workload.txns[i][0][3]
        except (IndexError, TypeError, AttributeError):
            txn_lengths[i] = 1.0

    lpt_sorted_indices = sorted(txn_lengths.keys(), key=lambda k: txn_lengths[k], reverse=True)

    # --- 1. Constructive Phase: Enhanced Beam Search ---
    # Restoring higher beam width and sampling from successful Generation 7
    BEAM_WIDTH = max(16, int(num_seqs * 2))
    SAMPLE_SIZE = 16
    GAMMA = 1.3

    # Initial Beam Seeding
    seeds = set(lpt_sorted_indices[:BEAM_WIDTH])
    if len(seeds) < BEAM_WIDTH:
        remaining_slots = BEAM_WIDTH - len(seeds)
        seeds.update(random.sample(range(num_txns), min(num_txns - len(seeds), remaining_slots)))

    beam = []
    for t in seeds:
        seq = [t]
        cost = workload.get_opt_seq_cost(seq)
        acc_work = txn_lengths[t]
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

    beam.sort(key=lambda x: x['score'])
    beam = beam[:BEAM_WIDTH]

    # Construction Loop
    for _ in range(num_txns - 1):
        candidates = []

        for parent in beam:
            rem_set = parent['rem']
            if not rem_set:
                continue

            to_eval = set()

            # A) Heuristic candidates
            added = 0
            for t in lpt_sorted_indices:
                if t in rem_set:
                    to_eval.add(t)
                    added += 1
                    if added >= 4: # Constant check of top remaining
                        break

            # B) Random candidates
            needed = SAMPLE_SIZE - len(to_eval)
            rem_list = list(rem_set)
            if needed > 0 and rem_list:
                if len(rem_list) <= needed:
                    to_eval.update(rem_list)
                else:
                    to_eval.update(random.sample(rem_list, needed))

            base_seq = parent['seq']
            base_work = parent['acc_work']

            for t in to_eval:
                new_seq = base_seq + [t]
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

        # Diversity-Aware Pruning
        candidates.sort(key=lambda x: x['score'])

        k_best = int(BEAM_WIDTH * 0.7)
        next_beam = candidates[:k_best]

        remaining_needed = BEAM_WIDTH - len(next_beam)
        if remaining_needed > 0:
            # Sample from a wider pool of good candidates to maintain diversity
            pool_size = min(len(candidates), BEAM_WIDTH * 3)
            pool = candidates[k_best : pool_size]
            if len(pool) <= remaining_needed:
                next_beam.extend(pool)
            else:
                next_beam.extend(random.sample(pool, remaining_needed))

        beam = next_beam

    # Select best from construction
    beam.sort(key=lambda x: x['cost'])
    best_candidate = beam[0]

    current_schedule = best_candidate['seq']
    current_cost = best_candidate['cost']

    # --- 2. Refinement Phase: SA with Block Moves ---

    MAX_ITER = 750 # Slightly increased iterations
    INITIAL_TEMP = max(1.0, current_cost * 0.05)
    COOLING_RATE = 0.99

    best_schedule = list(current_schedule)
    best_cost = current_cost

    temp = INITIAL_TEMP

    for i in range(MAX_ITER):
        neighbor = list(current_schedule)
        r = random.random()

        # Dynamic Operator Selection with Bias to target schedule tail
        if r < 0.25:
            # Standard Swap
            idx1, idx2 = random.sample(range(num_txns), 2)
            neighbor[idx1], neighbor[idx2] = neighbor[idx2], neighbor[idx1]
        elif r < 0.50:
            # Biased Insert: Move from second half to anywhere (Targeting Tail)
            idx1 = random.randint(num_txns // 2, num_txns - 1)
            val = neighbor.pop(idx1)
            idx2 = random.randint(0, num_txns - 1)
            neighbor.insert(idx2, val)
        elif r < 0.75:
            # Standard Insert
            idx1 = random.randint(0, num_txns - 1)
            val = neighbor.pop(idx1)
            idx2 = random.randint(0, num_txns - 1)
            neighbor.insert(idx2, val)
        else:
            # Block Move: Move a contiguous chunk
            if num_txns > 4:
                # Slightly larger blocks allowed
                block_size = random.randint(2, max(3, num_txns // 6))
                start_idx = random.randint(0, num_txns - block_size)

                # Extract block
                block = neighbor[start_idx : start_idx + block_size]
                del neighbor[start_idx : start_idx + block_size]

                # Insert block
                insert_idx = random.randint(0, len(neighbor))
                neighbor[insert_idx:insert_idx] = block
            else:
                # Fallback for very small workloads
                idx1, idx2 = random.sample(range(num_txns), 2)
                neighbor[idx1], neighbor[idx2] = neighbor[idx2], neighbor[idx1]

        # Evaluate neighbor
        neighbor_cost = workload.get_opt_seq_cost(neighbor)
        delta = neighbor_cost - current_cost

        accept = False
        if delta < 0:
            accept = True
        elif temp > 0.001:
            prob = math.exp(-delta / temp)
            if random.random() < prob:
                accept = True

        if accept:
            current_schedule = neighbor
            current_cost = neighbor_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_schedule = list(current_schedule)

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