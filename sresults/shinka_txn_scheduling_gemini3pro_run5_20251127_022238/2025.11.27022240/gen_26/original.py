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
    Hybrid Scheduling Algorithm: Work-Density Beam Search + Structural Refinement

    1. Constructive Phase: Uses a Beam Search guided by a "Work-Density" metric
       (Makespan - Gamma * CompletedWork). This encourages scheduling long/heavy
       transactions early to fill "gaps" in the schedule.

    2. Refinement Phase: Applies Simulated Annealing with a diverse set of operators
       (Swap, Insert, Block-Move, Block-Reverse) to escape local optima found
       by the constructive phase.

    Args:
        workload: Workload object
        num_seqs: Hint for computational budget

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """

    # --- 0. Setup & Precomputation ---
    num_txns = workload.num_txns

    # Extract transaction lengths (costs) for heuristics
    # workload.txns[i][0][3] is the estimated cost/length
    txn_lengths = {}
    for i in range(num_txns):
        try:
            txn_lengths[i] = workload.txns[i][0][3]
        except (IndexError, TypeError, AttributeError):
            txn_lengths[i] = 1.0

    # LPT (Longest Processing Time) sort for heuristic candidate selection
    lpt_indices = sorted(txn_lengths.keys(), key=lambda k: txn_lengths[k], reverse=True)

    # --- 1. Constructive Phase: Work-Density Beam Search ---

    # Parameters
    # Beam width: Controls breadth of search.
    # We use a moderate width to allow budget for the refinement phase.
    BEAM_WIDTH = max(8, int(num_seqs * 1.5))

    # Expansion: How many candidates to check per node
    HEURISTIC_SAMPLE = 6  # Top N longest remaining
    RANDOM_SAMPLE = 4     # M random remaining

    # Scoring: Controls bias towards completing work early
    # Higher gamma (>1.0) rewards schedules that finish heavy items quickly
    GAMMA = 1.4

    # Initialization
    # Seed beam with diverse starting transactions
    seeds = set(lpt_indices[:BEAM_WIDTH])
    if len(seeds) < BEAM_WIDTH:
        remaining_slots = BEAM_WIDTH - len(seeds)
        seeds.update(random.sample(range(num_txns), min(num_txns - len(seeds), remaining_slots)))

    beam = []
    for t in seeds:
        seq = [t]
        cost = workload.get_opt_seq_cost(seq)
        acc_work = txn_lengths[t]
        # Score calculation: Minimize (Cost - Gamma * Work)
        score = cost - (GAMMA * acc_work)

        rem = set(range(num_txns))
        rem.remove(t)

        beam.append({
            'seq': seq,
            'cost': cost,
            'score': score,
            'acc_work': acc_work,
            'rem': rem
        })

    # Sort and trim initial beam
    beam.sort(key=lambda x: x['score'])
    beam = beam[:BEAM_WIDTH]

    # Main Construction Loop
    for _ in range(num_txns - 1):
        candidates = []

        for parent in beam:
            rem_set = parent['rem']
            if not rem_set:
                continue

            to_eval = set()

            # A) Heuristic Candidates (LPT)
            added = 0
            for t in lpt_indices:
                if t in rem_set:
                    to_eval.add(t)
                    added += 1
                    if added >= HEURISTIC_SAMPLE:
                        break

            # B) Random Candidates (Diversity)
            # Sample directly from set if possible or convert to list
            rem_list = list(rem_set)
            needed = RANDOM_SAMPLE

            # Simple rejection sampling to avoid duplicates with heuristic
            attempts = 0
            while len(to_eval) < (added + needed) and attempts < needed * 3:
                attempts += 1
                if not rem_list: break
                t = random.choice(rem_list)
                to_eval.add(t)

            # If we still need more and haven't exhausted list, force add
            if len(to_eval) < (added + needed) and len(rem_list) > len(to_eval):
                 remaining_to_add = (added + needed) - len(to_eval)
                 pool = [x for x in rem_list if x not in to_eval]
                 to_eval.update(random.sample(pool, min(len(pool), remaining_to_add)))

            # Evaluate
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

        # Beam Selection
        candidates.sort(key=lambda x: x['score'])

        # Elitism: Keep best fraction
        k_best = int(BEAM_WIDTH * 0.6)
        next_beam = candidates[:k_best]

        # Diversity: Sample from the rest
        remaining_needed = BEAM_WIDTH - len(next_beam)
        if remaining_needed > 0:
            pool = candidates[k_best : k_best * 4] # Restrict pool to "good enough"
            if len(pool) <= remaining_needed:
                next_beam.extend(pool)
            else:
                next_beam.extend(random.sample(pool, remaining_needed))

        beam = next_beam

    # Select best candidate from construction
    beam.sort(key=lambda x: x['cost'])
    best_constructive = beam[0]

    current_schedule = best_constructive['seq']
    current_cost = best_constructive['cost']

    # --- 2. Refinement Phase: Simulated Annealing with Block Operators ---

    # SA Parameters
    # We use a fixed iteration count suitable for the problem scale
    MAX_ITER = 800
    INITIAL_TEMP = max(1.0, current_cost * 0.04)
    COOLING_RATE = 0.985

    best_schedule = list(current_schedule)
    best_cost = current_cost

    temp = INITIAL_TEMP

    for _ in range(MAX_ITER):
        neighbor = list(current_schedule)
        op = random.random()

        # Operator Selection Probabilities:
        # 0.0 - 0.2: Swap
        # 0.2 - 0.6: Insert (Shift) - very effective for scheduling
        # 0.6 - 0.9: Block Move - moves clusters
        # 0.9 - 1.0: Block Reverse - reorders conflicts locally

        if op < 0.2:
            # Swap
            i, j = random.sample(range(num_txns), 2)
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]

        elif op < 0.6:
            # Insert
            i = random.randint(0, num_txns - 1)
            val = neighbor.pop(i)
            j = random.randint(0, num_txns - 1) # Insert index (neighbor is shorter now)
            neighbor.insert(j, val)

        elif op < 0.9:
            # Block Move
            if num_txns > 5:
                # Random block size between 2 and 15% of total
                block_size = random.randint(2, max(3, int(num_txns * 0.15)))
                start_idx = random.randint(0, num_txns - block_size)

                block = neighbor[start_idx : start_idx + block_size]
                del neighbor[start_idx : start_idx + block_size]

                insert_idx = random.randint(0, len(neighbor))
                neighbor[insert_idx:insert_idx] = block
            else:
                # Fallback to swap
                i, j = random.sample(range(num_txns), 2)
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]

        else:
            # Block Reverse
            if num_txns > 4:
                block_size = random.randint(2, max(3, int(num_txns * 0.1)))
                start_idx = random.randint(0, num_txns - block_size)
                end_idx = start_idx + block_size
                neighbor[start_idx:end_idx] = reversed(neighbor[start_idx:end_idx])
            else:
                # Fallback
                i, j = random.sample(range(num_txns), 2)
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]

        # Evaluate
        neighbor_cost = workload.get_opt_seq_cost(neighbor)
        delta = neighbor_cost - current_cost

        accept = False
        if delta < 0:
            accept = True
        elif temp > 0.001:
            if random.random() < math.exp(-delta / temp):
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