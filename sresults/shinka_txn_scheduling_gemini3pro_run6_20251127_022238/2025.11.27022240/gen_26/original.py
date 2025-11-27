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

try:
    repo_root = find_repo_root(os.path.dirname(__file__))
    sys.path.insert(0, os.path.join(repo_root, 'openevolve_examples', 'txn_scheduling'))
except Exception as e:
    # Allow execution to proceed if modules are already in path or mock environment
    pass

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3


def get_best_schedule(workload, num_seqs):
    """
    Get optimal schedule using Dynamic Beam Search followed by Local Search.

    Method:
    1. Beam Search Construction:
       - Maintains multiple parallel partial schedules.
       - Uses 'Dynamic Beam Width': Starts wide to explore early structural choices,
         then narrows (tapers) as the schedule fills up.
       - Expansion uses weighted sampling (LPT heuristic) and tail optimization.
       - Selection uses a composite metric (Cost, -Duration) to prioritize "filling gaps"
         with larger transactions when costs are equal.
    2. Local Search Refinement:
       - Intensifies around the best found schedule using Shift (80%) and Swap (20%) operators.

    Args:
        workload: Workload object
        num_seqs: Determines the computational budget (base beam width)

    Returns:
        (lowest_makespan, schedule)
    """

    # 1. Metric Precomputation
    txn_metrics = {}
    try:
        for t in range(workload.num_txns):
            # Assumed structure: txns[t] -> list of sequences -> [0] -> [3] is duration
            txn_metrics[t] = workload.txns[t][0][3]
    except (IndexError, AttributeError, TypeError):
        for t in range(workload.num_txns):
            txn_metrics[t] = 1.0

    # 2. Beam Search Setup
    # Set base width roughly equal to num_seqs (e.g. 10)
    BASE_BEAM_WIDTH = max(4, int(num_seqs))

    # Beam State: (cost, schedule_list, remaining_indices_list)
    current_beam = [(0, [], list(range(workload.num_txns)))]

    # 3. Construction Loop
    num_txns = workload.num_txns
    for step in range(num_txns):
        candidates_pool = []

        # DYNAMIC BEAM WIDTH: Taper from 1.5x to 0.5x of base width
        # Allocates more search to early critical decisions
        progress = step / num_txns
        # Formula: width decays linearly
        width_factor = 1.5 - progress
        current_width = int(BASE_BEAM_WIDTH * width_factor)
        current_width = max(2, current_width)  # Ensure minimum width

        # Expand each partial schedule in the beam
        for parent_cost, parent_sched, parent_remaining in current_beam:
            next_candidates = set()

            # OPTIMIZATION: Exhaustive Tail
            if len(parent_remaining) <= 20:
                next_candidates.update(parent_remaining)
            else:
                # OPTIMIZATION: Weighted Sampling (LPT)
                weights = [txn_metrics[t] for t in parent_remaining]

                # Sample based on weights
                # k=6 provides good lookahead
                samples = random.choices(parent_remaining, weights=weights, k=6)
                next_candidates.update(samples)

                # Diversity: Pure random samples
                random_samples = random.sample(parent_remaining, min(len(parent_remaining), 2))
                next_candidates.update(random_samples)

            # Evaluate candidates
            for txn_idx in next_candidates:
                new_sched = parent_sched + [txn_idx]
                new_cost = workload.get_opt_seq_cost(new_sched)

                # Metric: (Cost, -Duration)
                # Primary: Minimize Makespan.
                # Secondary: Maximize Duration (LPT tie-breaking).
                sort_metric = (new_cost, -txn_metrics.get(txn_idx, 0))

                new_remaining = list(parent_remaining)
                new_remaining.remove(txn_idx)

                candidates_pool.append((sort_metric, new_sched, new_remaining))

        # Pruning: Keep best K candidates
        candidates_pool.sort(key=lambda x: x[0])

        # Create next beam
        current_beam = [ (x[0][0], x[1], x[2]) for x in candidates_pool[:current_width] ]

    # 4. Extract Best Schedule
    best_beam_state = current_beam[0]
    current_cost = best_beam_state[0]
    current_schedule = best_beam_state[1]

    # 5. Local Search Refinement
    search_steps = 800
    no_improv_limit = 150
    no_improv = 0

    for _ in range(search_steps):
        if no_improv >= no_improv_limit:
            break

        neighbor = list(current_schedule)

        # Operator: 80% Shift, 20% Swap
        if random.random() < 0.8:
            src = random.randint(0, len(neighbor) - 1)
            dst = random.randint(0, len(neighbor) - 1)
            if src == dst:
                continue
            txn = neighbor.pop(src)
            neighbor.insert(dst, txn)
        else:
            i = random.randint(0, len(neighbor) - 1)
            j = random.randint(0, len(neighbor) - 1)
            if i == j:
                continue
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]

        new_cost = workload.get_opt_seq_cost(neighbor)

        if new_cost < current_cost:
            current_cost = new_cost
            current_schedule = neighbor
            no_improv = 0
        else:
            no_improv += 1

    return current_cost, current_schedule


def get_random_costs():
    """Evaluate scheduling algorithm on three different workloads."""
    start_time = time.time()

    # Budget for greedy restarts
    num_seqs = 10

    # Process workloads
    workload1 = Workload(WORKLOAD_1)
    makespan1, schedule1 = get_best_schedule(workload1, num_seqs)

    workload2 = Workload(WORKLOAD_2)
    makespan2, schedule2 = get_best_schedule(workload2, num_seqs)

    workload3 = Workload(WORKLOAD_3)
    makespan3, schedule3 = get_best_schedule(workload3, num_seqs)

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