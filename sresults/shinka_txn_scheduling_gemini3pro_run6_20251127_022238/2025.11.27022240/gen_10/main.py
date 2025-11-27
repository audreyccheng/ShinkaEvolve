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
    Get optimal schedule using Adaptive Greedy Construction with Tie-Breaking and Hybrid Local Search.

    1. Adaptive Greedy: Builds schedules by selecting best next transaction.
       - Uses random sampling + heavy-item bias for candidates.
       - Switches to exhaustive search when few transactions remain (Exhaustive Tail).
       - Tie-breaks candidates with equal makespan by choosing the longest duration transaction (Best Fit).
    2. Local Search: Refines the best schedule using both Swap and Shift operators.

    Args:
        workload: Workload object containing transaction data
        num_seqs: Number of greedy iterations to perform

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """

    # Pre-calculate transaction costs for heuristic (Longest Processing Time)
    txn_costs = {}
    try:
        for t in range(workload.num_txns):
            txn_costs[t] = workload.txns[t][0][3]
    except (IndexError, AttributeError, TypeError):
        for t in range(workload.num_txns):
            txn_costs[t] = 1

    def generate_adaptive_schedule(sample_size, use_bias):
        """Generates a schedule using adaptive greedy strategy."""
        remaining = list(range(workload.num_txns))
        schedule = []

        # Start random
        start_idx = random.randint(0, len(remaining) - 1)
        schedule.append(remaining.pop(start_idx))

        exhaustive_threshold = 20

        while remaining:
            candidates = set()

            # Adaptive: Exhaustive search at the tail ensures tight packing
            if len(remaining) <= exhaustive_threshold:
                candidates.update(remaining)
            else:
                # Sampling
                k = min(len(remaining), sample_size)
                candidates.update(random.sample(remaining, k))

                # Heuristic bias: Check heavy transactions
                if use_bias:
                    bias_k = 4
                    heaviest = sorted(remaining, key=lambda x: txn_costs.get(x, 0), reverse=True)[:bias_k]
                    candidates.update(heaviest)

            # Evaluate candidates
            best_c = -1
            best_c_cost = float('inf')
            best_candidates_equal = []

            for c in candidates:
                test_seq = schedule + [c]
                cost = workload.get_opt_seq_cost(test_seq)

                if cost < best_c_cost:
                    best_c_cost = cost
                    best_candidates_equal = [c]
                elif cost == best_c_cost:
                    best_candidates_equal.append(c)

            # Tie-breaking: Pick heaviest among best to fill "shadow"
            if best_candidates_equal:
                best_c = max(best_candidates_equal, key=lambda x: txn_costs.get(x, 0))

            if best_c != -1:
                schedule.append(best_c)
                remaining.remove(best_c)
            else:
                # Fallback (rare)
                c = remaining.pop(0)
                schedule.append(c)

        return workload.get_opt_seq_cost(schedule), schedule

    best_overall_cost = float('inf')
    best_overall_schedule = []

    iterations = max(1, num_seqs)

    for i in range(iterations):
        # Diversity in first few iterations (random), then bias (heuristic)
        if i < 2:
            cost, sched = generate_adaptive_schedule(sample_size=8, use_bias=False)
        else:
            cost, sched = generate_adaptive_schedule(sample_size=4, use_bias=True)

        if cost < best_overall_cost:
            best_overall_cost = cost
            best_overall_schedule = sched

    # Local Search Phase
    current_schedule = list(best_overall_schedule)
    current_cost = best_overall_cost

    search_steps = 500
    no_improv_limit = 150
    no_improv = 0

    for _ in range(search_steps):
        if no_improv >= no_improv_limit:
            break

        neighbor = list(current_schedule)

        # Randomly choose operator: Swap or Shift
        # Shift is 70% likely as it preserves relative orderings better
        if random.random() < 0.3:
            # Swap
            i, j = random.sample(range(len(neighbor)), 2)
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        else:
            # Shift (Insert)
            src, dst = random.sample(range(len(neighbor)), 2)
            txn = neighbor.pop(src)
            neighbor.insert(dst, txn)

        new_cost = workload.get_opt_seq_cost(neighbor)

        if new_cost < current_cost:
            current_cost = new_cost
            current_schedule = neighbor
            no_improv = 0
        else:
            no_improv += 1

    return current_cost, current_schedule


def get_random_costs():
    """
    Evaluate scheduling algorithm on three different workloads.

    Returns:
        Tuple of (total_makespan, list_of_schedules, execution_time)
    """
    start_time = time.time()

    # Use 10 restarts for the multi-start greedy algorithm
    # This provides a good trade-off between execution time and schedule quality
    num_seqs = 10

    # Workload 1: Complex mixed read/write transactions
    workload1 = Workload(WORKLOAD_1)
    makespan1, schedule1 = get_best_schedule(workload1, num_seqs)

    # Workload 2: Simple read-then-write pattern
    workload2 = Workload(WORKLOAD_2)
    makespan2, schedule2 = get_best_schedule(workload2, num_seqs)

    # Workload 3: Minimal read/write operations
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