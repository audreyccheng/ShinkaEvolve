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
    Adaptive Simulated Annealing with Ruin-and-Recreate (ILS).

    Args:
        workload: Workload object containing transaction data
        num_seqs: Parameter affecting the computational budget

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    # --- Hyperparameters ---
    # Width 4 strikes the best balance between exploring structural diversity
    # and computational cost for the exhaustive expansion step.
    BEAM_WIDTH = 4

    # SA Parameters
    SA_ITERATIONS = 5000
    SA_COOLING_RATE = 0.998
    SA_START_TEMP_RATIO = 0.05

    # ILS / Kick Parameters
    # Trigger ruin-and-recreate when the search stagnates
    STAGNATION_LIMIT = 500
    KICK_BLOCK_SIZE_MIN = 3
    KICK_BLOCK_SIZE_MAX = 8

    # Adaptive Operator Weights
    OP_WEIGHTS = {'swap': 5.0, 'insert': 5.0, 'reverse': 2.0}
    OP_MIN_WEIGHT = 1.0
    OP_ADAPTATION_RATE = 0.1

    # --- Cost Cache ---
    cost_cache = {}

    def get_cost(seq):
        t_seq = tuple(seq)
        if t_seq in cost_cache:
            return cost_cache[t_seq]
        c = workload.get_opt_seq_cost(seq)
        cost_cache[t_seq] = c
        return c

    # --- Pre-calculation: Transaction Weights ---
    # Used for tie-breaking in Beam Search (prefer packing heavier txns for free)
    txn_weights = {}
    for t in range(workload.num_txns):
        txn_weights[t] = get_cost([t])

    # --- Phase 1: Density-Aware Beam Search ---
    # Initialize beam
    candidates = []
    for t in range(workload.num_txns):
        seq = [t]
        cost = get_cost(seq)
        # Score: minimize cost, then maximize weight (subtracted).
        # Factor 0.001 ensures it only breaks ties or near-ties.
        score = cost - (0.001 * txn_weights[t])
        candidates.append({
            'cost': cost,
            'score': score,
            'seq': seq,
            'rem': [x for x in range(workload.num_txns) if x != t]
        })

    candidates.sort(key=lambda x: x['score'])
    beam = candidates[:BEAM_WIDTH]

    # Iteratively build schedule
    for _ in range(workload.num_txns - 1):
        next_candidates = []

        for node in beam:
            b_seq = node['seq']
            b_rem = node['rem']

            for cand in b_rem:
                new_seq = b_seq + [cand]
                new_cost = get_cost(new_seq)

                # Density-Aware Scoring
                # If new_cost is same as old, we packed 'cand' for free.
                # We prefer packing larger 'cand'.
                score = new_cost - (0.001 * txn_weights[cand])

                next_candidates.append((score, new_cost, new_seq, b_rem, cand))

        # Select best global candidates by score
        next_candidates.sort(key=lambda x: x[0])

        new_beam = []
        for c_score, c_cost, c_seq, c_parent_rem, c_cand in next_candidates:
            if len(new_beam) >= BEAM_WIDTH:
                break

            new_rem = list(c_parent_rem)
            new_rem.remove(c_cand)
            new_beam.append({'cost': c_cost, 'seq': c_seq, 'rem': new_rem})

        beam = new_beam

    if not beam:
        return float('inf'), []

    current_schedule = beam[0]['seq']
    current_cost = beam[0]['cost']

    # --- Phase 2: Adaptive SA with Multi-Mode Ruin-and-Recreate ---
    best_schedule = list(current_schedule)
    best_cost = current_cost

    T = current_cost * SA_START_TEMP_RATIO
    ops = list(OP_WEIGHTS.keys())

    steps_since_improvement = 0

    for it in range(SA_ITERATIONS):
        # 1. Stagnation Check -> Ruin-and-Recreate (Kick)
        # Lower threshold to trigger intensification more often
        if steps_since_improvement > 150:
            # Multi-Mode Ruin
            kick_seq = list(best_schedule)
            n = len(kick_seq)
            removed_items = []

            mode = random.random()
            if mode < 0.5:
                # Mode A: Block Ruin (removes contiguity)
                if n > KICK_BLOCK_SIZE_MIN:
                    b_size = random.randint(KICK_BLOCK_SIZE_MIN, KICK_BLOCK_SIZE_MAX)
                    start = random.randint(0, n - b_size)
                    removed_items = kick_seq[start : start + b_size]
                    del kick_seq[start : start + b_size]
            else:
                # Mode B: Scatter Ruin (removes dependencies globally)
                # Pick K random indices to remove
                if n > KICK_BLOCK_SIZE_MIN:
                    num_to_remove = random.randint(KICK_BLOCK_SIZE_MIN, KICK_BLOCK_SIZE_MAX)
                    # Sample indices
                    indices = sorted(random.sample(range(n), num_to_remove), reverse=True)
                    for idx in indices:
                        removed_items.append(kick_seq.pop(idx))

            # Recreate: Greedy Best-Fit Insertion
            # Shuffle removed items to avoid insertion bias
            random.shuffle(removed_items)

            for item in removed_items:
                best_pos = -1
                min_c = float('inf')

                # Check all valid positions
                for i in range(len(kick_seq) + 1):
                    kick_seq.insert(i, item)
                    c = get_cost(kick_seq)
                    if c < min_c:
                        min_c = c
                        best_pos = i
                    kick_seq.pop(i)

                kick_seq.insert(best_pos, item)

            current_schedule = kick_seq
            current_cost = min_c

            # Check global best
            if current_cost < best_cost:
                best_cost = current_cost
                best_schedule = list(current_schedule)
                steps_since_improvement = 0
            else:
                # Reset anyway to explore the new basin
                steps_since_improvement = 0

            # Reheat Temperature
            T = max(T, current_cost * 0.05)
            continue

        # 2. Normal SA Step with Adaptive Operators
        # Select Operator
        total_w = sum(OP_WEIGHTS.values())
        r = random.uniform(0, total_w)
        cum = 0
        op = ops[0]
        for o in ops:
            cum += OP_WEIGHTS[o]
            if r <= cum:
                op = o
                break

        neighbor = list(current_schedule)
        n = len(neighbor)

        if op == 'swap':
            i, j = random.sample(range(n), 2)
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        elif op == 'insert':
            i = random.randint(0, n - 1)
            val = neighbor.pop(i)
            j = random.randint(0, n - 1)
            neighbor.insert(j, val)
        elif op == 'reverse':
            i, j = sorted(random.sample(range(n), 2))
            neighbor[i:j+1] = neighbor[i:j+1][::-1]

        new_cost = get_cost(neighbor)
        delta = new_cost - current_cost

        # Acceptance Criteria
        accept = False
        improved_global = False

        if delta < 0:
            accept = True
            if new_cost < best_cost:
                best_cost = new_cost
                best_schedule = list(neighbor)
                steps_since_improvement = 0
                improved_global = True
            else:
                steps_since_improvement += 1
        else:
            steps_since_improvement += 1
            if T > 1e-9 and random.random() < math.exp(-delta / T):
                accept = True

        if accept:
            current_schedule = neighbor
            current_cost = new_cost

        # 3. Adapt Weights
        reward = 0.1
        if improved_global:
            reward = 2.0
        elif accept and delta < 0:
            reward = 1.0

        OP_WEIGHTS[op] = (1 - OP_ADAPTATION_RATE) * OP_WEIGHTS[op] + OP_ADAPTATION_RATE * (OP_MIN_WEIGHT + reward)

        # 4. Cooling
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