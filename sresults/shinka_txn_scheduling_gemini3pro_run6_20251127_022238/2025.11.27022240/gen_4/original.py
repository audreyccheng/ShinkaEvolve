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
    Get optimal schedule using a multi-start greedy strategy with heuristic bias.

    This approach runs multiple independent greedy constructions. It combines random
    sampling with a heuristic that prioritizes 'heavy' (long duration) transactions
    to find a schedule that minimizes makespan.

    Args:
        workload: Workload object containing transaction data
        num_seqs: Number of sequences to generate/sample (iterations)

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """

    # Pre-calculate transaction costs for heuristic (Longest Processing Time)
    # Structure assumption: workload.txns[t][0][3] is the duration/cost
    txn_costs = {}
    try:
        for t in range(workload.num_txns):
            # Accessing the cost of the first operation sequence of the transaction
            txn_costs[t] = workload.txns[t][0][3]
    except (IndexError, AttributeError, TypeError):
        # Fallback if structure differs
        for t in range(workload.num_txns):
            txn_costs[t] = 1

    def generate_schedule(sample_size, use_heavy_bias=False):
        """Generates a single schedule using randomized greedy construction."""
        remaining = list(range(workload.num_txns))
        schedule = []

        # Start with a random transaction to ensure diversity across restarts
        start_idx = random.randint(0, len(remaining) - 1)
        schedule.append(remaining.pop(start_idx))

        # Greedily build the rest of the schedule
        while remaining:
            candidates = set()

            # 1. Random sampling: Pick random candidates from remaining
            k = min(len(remaining), sample_size)
            candidates.update(random.sample(remaining, k))

            # 2. Heuristic bias: Also consider the heaviest remaining transactions
            # This helps to schedule long transactions when they fit best
            if use_heavy_bias:
                # Get top 2 heaviest transactions
                heaviest = sorted(remaining, key=lambda x: txn_costs.get(x, 0), reverse=True)[:2]
                candidates.update(heaviest)

            # Evaluate all unique candidates
            best_c = -1
            best_c_cost = float('inf')

            for c in candidates:
                # Calculate cost of appending this candidate
                # workload.get_opt_seq_cost returns the makespan of the sequence
                test_seq = schedule + [c]
                cost = workload.get_opt_seq_cost(test_seq)

                if cost < best_c_cost:
                    best_c_cost = cost
                    best_c = c

            if best_c != -1:
                schedule.append(best_c)
                remaining.remove(best_c)
            else:
                # Should technically not happen if candidates list is not empty
                c = remaining.pop(0)
                schedule.append(c)

        # Return final cost and schedule
        final_cost = workload.get_opt_seq_cost(schedule)
        return final_cost, schedule

    best_overall_cost = float('inf')
    best_overall_schedule = []

    # Run multiple iterations to explore the search space
    # num_seqs determines the budget (number of restarts)
    iterations = max(1, num_seqs)

    for i in range(iterations):
        # Iteration strategy:
        # First iteration: Use a wider random sample (6) without bias to get a robust baseline.
        # Subsequent iterations: Use a tighter sample (2) but inject heavy-item bias.
        # This combination (Randomized Greedy + LPT Heuristic) usually outperforms pure random greedy.

        if i == 0:
            cost, sched = generate_schedule(sample_size=6, use_heavy_bias=False)
        else:
            cost, sched = generate_schedule(sample_size=2, use_heavy_bias=True)

        if cost < best_overall_cost:
            best_overall_cost = cost
            best_overall_schedule = sched

    return best_overall_cost, best_overall_schedule


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