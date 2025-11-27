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
    Get optimal schedule using a hybrid strategy: Multi-start Greedy + Local Search.

    1. Multi-start Greedy: Generates candidate schedules using randomized greedy construction
       biased towards 'heavy' transactions (LPT heuristic).
    2. Local Search (Hill Climbing): Refines the best candidate found by shifting transactions
       to different positions to escape local optima and tighten the makespan.

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

    def generate_greedy_schedule(sample_size, bias_limit):
        """Generates a schedule using randomized greedy with optional heavy-item bias."""
        remaining = list(range(workload.num_txns))
        schedule = []

        # Start with a random transaction
        start_idx = random.randint(0, len(remaining) - 1)
        schedule.append(remaining.pop(start_idx))

        while remaining:
            candidates = set()

            # Random sampling
            k = min(len(remaining), sample_size)
            candidates.update(random.sample(remaining, k))

            # Heuristic bias: consider heaviest remaining transactions
            if bias_limit > 0:
                heaviest = sorted(remaining, key=lambda x: txn_costs.get(x, 0), reverse=True)[:bias_limit]
                candidates.update(heaviest)

            best_c = -1
            best_c_cost = float('inf')

            # Select best candidate based on immediate makespan
            for c in candidates:
                test_seq = schedule + [c]
                cost = workload.get_opt_seq_cost(test_seq)

                if cost < best_c_cost:
                    best_c_cost = cost
                    best_c = c

            if best_c != -1:
                schedule.append(best_c)
                remaining.remove(best_c)
            else:
                schedule.append(remaining.pop(0))

        return workload.get_opt_seq_cost(schedule), schedule

    best_overall_cost = float('inf')
    best_overall_schedule = []

    # Phase 1: Multi-start Greedy Construction
    iterations = max(1, num_seqs)

    for i in range(iterations):
        # First iteration: Exploration (random)
        # Others: Exploitation (heavy bias)
        if i == 0:
            cost, sched = generate_greedy_schedule(sample_size=6, bias_limit=0)
        else:
            cost, sched = generate_greedy_schedule(sample_size=3, bias_limit=3)

        if cost < best_overall_cost:
            best_overall_cost = cost
            best_overall_schedule = sched

    # Phase 2: Local Search Refinement (Hill Climbing with Shift)
    # Try to improve the best schedule by moving transactions
    current_schedule = list(best_overall_schedule)
    current_cost = best_overall_cost

    # Number of improvement attempts
    search_steps = 400
    no_improv_limit = 100
    no_improv = 0

    for _ in range(search_steps):
        if no_improv >= no_improv_limit:
            break

        # Shift operator: Move transaction from src to dst
        idx_src = random.randint(0, len(current_schedule) - 1)
        idx_dst = random.randint(0, len(current_schedule) - 1)

        if idx_src == idx_dst:
            continue

        neighbor = list(current_schedule)
        txn = neighbor.pop(idx_src)
        neighbor.insert(idx_dst, txn)

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