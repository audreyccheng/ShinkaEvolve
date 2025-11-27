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
    Find a low-makespan schedule using beam search with memoized cost
    evaluation and a local adjacent-swap refinement pass.

    Args:
        workload: Workload object containing transaction data
        num_seqs: Controls beam width (diversity of partial schedules)

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    n = workload.num_txns
    all_txns = list(range(n))

    # Simple memoization to avoid repeated cost computations for the same prefix
    cost_cache = {}
    def seq_cost(seq):
        key = tuple(seq)
        if key in cost_cache:
            return cost_cache[key]
        c = workload.get_opt_seq_cost(seq)
        cost_cache[key] = c
        return c

    # Beam parameters derived from num_seqs to control breadth and runtime
    beam_width = max(3, min(8, int(num_seqs)))  # keep a few strong candidates
    expand_all_threshold = 24  # when remaining txns are few, evaluate all
    sample_k = 24  # otherwise, sample a bounded subset for expansion

    # Initialize beam with the best singletons (evaluate exact cost for each start)
    init_candidates = []
    for t in all_txns:
        c = seq_cost([t])
        rem = set(all_txns)
        rem.remove(t)
        init_candidates.append((c, [t], rem))
    init_candidates.sort(key=lambda x: x[0])
    beam = init_candidates[:beam_width]

    # Beam search: grow sequences while keeping top-k by exact cost
    for _step in range(1, n):
        new_beam = []
        for cost_so_far, seq, rem in beam:
            if not rem:
                new_beam.append((cost_so_far, seq, rem))
                continue

            rem_list = list(rem)
            if len(rem_list) <= expand_all_threshold:
                candidates = rem_list
            else:
                candidates = random.sample(rem_list, min(sample_k, len(rem_list)))

            for t in candidates:
                new_seq = seq + [t]
                c = seq_cost(new_seq)
                new_rem = rem.copy()
                new_rem.remove(t)
                new_beam.append((c, new_seq, new_rem))

        # Select next beam: best unique endings to encourage diversity
        new_beam.sort(key=lambda x: x[0])
        next_beam = []
        seen_ends = set()
        for c, s, r in new_beam:
            end = s[-1]
            if end in seen_ends:
                continue
            seen_ends.add(end)
            next_beam.append((c, s, r))
            if len(next_beam) >= beam_width:
                break

        if not next_beam:
            # Fallback in degenerate cases
            next_beam = new_beam[:beam_width]

        beam = next_beam

    # Best complete sequence from the beam
    best_cost, best_seq, _ = min(beam, key=lambda x: x[0])

    # Local refinement: adjacent swap hill-climbing with first-improvement
    improved = True
    while improved:
        improved = False
        for i in range(n - 1):
            candidate = best_seq.copy()
            candidate[i], candidate[i + 1] = candidate[i + 1], candidate[i]
            c = seq_cost(candidate)
            if c < best_cost:
                best_cost = c
                best_seq = candidate
                improved = True
                break  # restart scan to capture cascading improvements

    return best_cost, best_seq


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