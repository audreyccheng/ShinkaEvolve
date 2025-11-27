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
    Get optimal schedule using Exhaustive Beam Search followed by
    Iterated Local Search with Ruin-and-Recreate.

    Args:
        workload: Workload object containing transaction data
        num_seqs: Parameter affecting the computational budget

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    # --- Hyperparameters ---
    BEAM_WIDTH = 4

    # ILS / SA Parameters
    TOTAL_ITERATIONS = 4500
    SA_START_TEMP_RATIO = 0.05
    SA_COOLING_RATE = 0.998

    # Ruin-and-Recreate Parameters
    STAGNATION_LIMIT = 300  # Steps without improvement to trigger kick
    KICK_BLOCK_SIZE_MIN = 3
    KICK_BLOCK_SIZE_MAX = 8

    # --- Cost Cache ---
    cost_cache = {}

    def get_cost(seq):
        t_seq = tuple(seq)
        if t_seq in cost_cache:
            return cost_cache[t_seq]
        c = workload.get_opt_seq_cost(seq)
        cost_cache[t_seq] = c
        return c

    # --- PHASE 1: Exhaustive Beam Search Construction ---
    # Initialize beam
    candidates = []
    for t in range(workload.num_txns):
        seq = [t]
        cost = get_cost(seq)
        candidates.append({'cost': cost, 'seq': seq, 'rem': {x for x in range(workload.num_txns) if x != t}})

    candidates.sort(key=lambda x: x['cost'])
    beam = candidates[:BEAM_WIDTH]

    # Iteratively build schedule
    for _ in range(workload.num_txns - 1):
        next_candidates = []

        for node in beam:
            p_seq = node['seq']
            p_rem = node['rem']

            # Exhaustive expansion: Check ALL remaining transactions
            for cand in p_rem:
                new_seq = p_seq + [cand]
                new_cost = get_cost(new_seq)
                next_candidates.append((new_cost, new_seq, p_rem, cand))

        # Sort by cost
        next_candidates.sort(key=lambda x: x[0])

        new_beam = []
        for c_cost, c_seq, c_parent_rem, c_cand in next_candidates:
            if len(new_beam) >= BEAM_WIDTH:
                break

            # Create new remaining set
            new_rem = c_parent_rem.copy()
            new_rem.remove(c_cand)
            new_beam.append({'cost': c_cost, 'seq': c_seq, 'rem': new_rem})

        beam = new_beam

    if not beam:
        return float('inf'), []

    current_schedule = beam[0]['seq']
    current_cost = beam[0]['cost']

    # --- PHASE 2: ILS (SA + Ruin-and-Recreate) ---
    best_schedule = list(current_schedule)
    best_cost = current_cost

    T = current_cost * SA_START_TEMP_RATIO
    stagnation_counter = 0

    for i in range(TOTAL_ITERATIONS):
        # 1. Stagnation Check -> Ruin-and-Recreate (Kick)
        if stagnation_counter >= STAGNATION_LIMIT:
            # Ruin: Remove random block
            # Kick from the BEST solution found to intensify search around it
            kick_seq = list(best_schedule)
            n = len(kick_seq)

            b_size = random.randint(KICK_BLOCK_SIZE_MIN, KICK_BLOCK_SIZE_MAX)
            if n > b_size:
                start_idx = random.randint(0, n - b_size)
                removed_block = kick_seq[start_idx : start_idx + b_size]
                del kick_seq[start_idx : start_idx + b_size]

                # Recreate: Greedy Best-Fit Insertion
                # For each removed item, try all positions and pick best
                for item in removed_block:
                    best_pos = -1
                    min_c = float('inf')

                    # Try inserting at all possible positions
                    # This is costly but effective for repair
                    for pos in range(len(kick_seq) + 1):
                        kick_seq.insert(pos, item)
                        c = get_cost(kick_seq)
                        if c < min_c:
                            min_c = c
                            best_pos = pos
                        kick_seq.pop(pos)

                    kick_seq.insert(best_pos, item)

                # Update current state to the kicked state
                current_schedule = kick_seq
                current_cost = min_c

                # Update global best if lucky
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_schedule = list(current_schedule)

                # Reset
                stagnation_counter = 0
                # Reheat temperature slightly to allow local relaxation
                T = max(T, current_cost * 0.02)
                continue

        # 2. Normal Local Search Step (SA)
        neighbor = list(current_schedule)
        n = len(neighbor)
        r = random.random()

        # Operators
        if r < 0.4: # Insert
            idx1 = random.randint(0, n - 1)
            val = neighbor.pop(idx1)
            idx2 = random.randint(0, n - 1)
            neighbor.insert(idx2, val)
        elif r < 0.7: # Swap
            idx1, idx2 = random.sample(range(n), 2)
            neighbor[idx1], neighbor[idx2] = neighbor[idx2], neighbor[idx1]
        else: # Reverse
            idx1, idx2 = random.sample(range(n), 2)
            s, e = min(idx1, idx2), max(idx1, idx2)
            neighbor[s:e+1] = neighbor[s:e+1][::-1]

        new_cost = get_cost(neighbor)
        delta = new_cost - current_cost

        # Acceptance
        accept = False
        if delta < 0:
            accept = True
            if new_cost < best_cost:
                best_cost = new_cost
                best_schedule = list(neighbor)
                stagnation_counter = 0
            else:
                stagnation_counter += 1
        else:
            stagnation_counter += 1
            if T > 1e-9 and random.random() < math.exp(-delta / T):
                accept = True

        if accept:
            current_schedule = neighbor
            current_cost = new_cost

        # Cooling
        T *= SA_COOLING_RATE

    return best_cost, best_schedule


def get_random_costs():
    """
    Evaluate scheduling algorithm on three different workloads.

    Returns:
        Tuple of (total_makespan, list_of_schedules, execution_time)
    """
    start_time = time.time()
    workload_size = 100

    # Workload 1: Complex mixed read/write transactions
    workload = Workload(WORKLOAD_1)
    makespan1, schedule1 = get_best_schedule(workload, 10)
    cost1 = workload.get_opt_seq_cost(schedule1)

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