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
    Get optimal schedule using Conflict-Aware Tapered Beam Search and Local Search Refinement.

    Strategy:
    1. Pre-computation:
       - Calculates transaction durations.
       - Parses Read/Write sets to compute 'Conflict Weights' (measure of contention).
    2. Tapered Beam Search Construction:
       - Dynamic Beam Width: Starts wide (2x budget) to make better early decisions,
         tapers to narrow (0.5x budget) for the tail.
       - Candidate Selection:
         - Weighted sampling using (Duration + 0.1 * ConflictWeight) to target critical path items.
         - Exhaustive search for the tail (last 20 items).
       - Pruning: Ranks by (Makespan, -ConflictWeight, -Duration). Prioritizes resolving
         high-contention transactions early when costs are similar.
    3. Local Search Refinement:
       - Applies Shift (Insertion) and Swap operators to fine-tune the constructed schedule.

    Args:
        workload: Workload object
        num_seqs: Base parameter for Beam Width (computational budget)

    Returns:
        (lowest_makespan, schedule)
    """

    # --- 1. PRE-COMPUTATION & CONFLICT ANALYSIS ---
    num_txns = workload.num_txns
    txn_durations = {}
    txn_rw_sets = {}

    # Parse durations and R/W sets
    for t in range(num_txns):
        # Duration
        try:
            duration = workload.txns[t][0][3]
        except (IndexError, AttributeError, TypeError):
            duration = 1.0
        txn_durations[t] = duration

        # R/W Sets for Conflict Detection
        reads = set()
        writes = set()
        try:
            # ops string usually at index 1: "w-17 r-5 ..."
            ops_str = workload.txns[t][0][1]
            if isinstance(ops_str, str):
                for op in ops_str.split():
                    if '-' in op:
                        parts = op.split('-')
                        if len(parts) == 2:
                            op_type, key = parts
                            if op_type == 'r': reads.add(key)
                            elif op_type == 'w': writes.add(key)
        except:
            pass
        txn_rw_sets[t] = (reads, writes)

    # Compute Conflict Weights
    # Weight = Sum of durations of all other transactions that conflict with T
    txn_conflict_weights = {t: 0.0 for t in range(num_txns)}

    for i in range(num_txns):
        r1, w1 = txn_rw_sets[i]
        weight = 0.0
        for j in range(num_txns):
            if i == j: continue
            r2, w2 = txn_rw_sets[j]
            # Conflict if any overlap in R/W sets (W-W, W-R, R-W)
            if not w1.isdisjoint(w2) or not w1.isdisjoint(r2) or not r1.isdisjoint(w2):
                weight += txn_durations[j]
        txn_conflict_weights[i] = weight

    # --- 2. TAPERED BEAM SEARCH CONSTRUCTION ---

    # Beam Width Config
    base_width = max(4, int(num_seqs))
    start_width = int(base_width * 2.0)
    end_width = max(2, int(base_width * 0.5))

    # State: (cost, schedule_list, remaining_indices_list)
    current_beam = [(0, [], list(range(num_txns)))]

    for step in range(num_txns):
        # Calculate dynamic beam width (Linear Decay)
        progress = step / num_txns
        current_beam_width = int(start_width + (end_width - start_width) * progress)
        current_beam_width = max(2, current_beam_width)

        candidates_pool = []

        for parent_cost, parent_sched, parent_remaining in current_beam:
            next_candidates = set()

            # OPTIMIZATION: Exhaustive Tail
            if len(parent_remaining) <= 20:
                next_candidates.update(parent_remaining)
            else:
                # OPTIMIZATION: Conflict-Aware Weighted Sampling
                # Prioritize heavy items that also block others
                weights = []
                for t in parent_remaining:
                    # Heuristic combination
                    w = txn_durations[t] + 0.1 * txn_conflict_weights[t]
                    weights.append(w)

                # Sample k candidates
                k = 6
                samples = random.choices(parent_remaining, weights=weights, k=k)
                next_candidates.update(samples)

                # Diversity
                random_samples = random.sample(parent_remaining, min(len(parent_remaining), 2))
                next_candidates.update(random_samples)

            # Evaluate candidates
            for txn_idx in next_candidates:
                new_sched = parent_sched + [txn_idx]

                # Primary: Minimize Makespan
                new_cost = workload.get_opt_seq_cost(new_sched)

                # Secondary Sorting Metric: (Cost, -ConflictWeight, -Duration)
                # Tie-breaking favors high conflict/duration items to clear bottlenecks
                sort_metric = (
                    new_cost,
                    -txn_conflict_weights[txn_idx],
                    -txn_durations[txn_idx]
                )

                new_remaining = list(parent_remaining)
                new_remaining.remove(txn_idx)

                candidates_pool.append((sort_metric, new_sched, new_remaining))

        # Pruning
        candidates_pool.sort(key=lambda x: x[0])

        # Select best unique schedules
        # Note: x[0] is the sort metric tuple, x[0][0] is cost
        current_beam = [ (x[0][0], x[1], x[2]) for x in candidates_pool[:current_beam_width] ]

    # --- 3. EXTRACT & REFINE ---
    best_beam_state = current_beam[0]
    current_cost = best_beam_state[0]
    current_schedule = best_beam_state[1]

    # 5. Local Search Refinement
    # Apply Shift (mostly) and Swap operators to improve the result
    search_steps = 800
    no_improv_limit = 150
    no_improv = 0

    for _ in range(search_steps):
        if no_improv >= no_improv_limit:
            break

        neighbor = list(current_schedule)

        # Operator Selection: 80% Shift, 20% Swap
        if random.random() < 0.8:
            # Shift (Insert) Operator
            src = random.randint(0, len(neighbor) - 1)
            dst = random.randint(0, len(neighbor) - 1)
            if src == dst:
                continue
            txn = neighbor.pop(src)
            neighbor.insert(dst, txn)
        else:
            # Swap Operator
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

    # Beam width parameter passed as num_seqs
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