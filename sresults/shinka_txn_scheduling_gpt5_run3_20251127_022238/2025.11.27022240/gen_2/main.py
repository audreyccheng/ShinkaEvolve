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
    Beam-search-based scheduler with local improvements to minimize makespan.

    Args:
        workload: Workload object containing transaction data
        num_seqs: Controls beam width / exploration budget

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    # Memoized evaluator for partial sequences to reduce simulator calls
    cost_cache = {}

    def evaluate_seq(seq):
        key = tuple(seq)
        if key in cost_cache:
            return cost_cache[key]
        c = workload.get_opt_seq_cost(seq)
        cost_cache[key] = c
        return c

    def beam_search():
        n = workload.num_txns
        # Set beam width and branching based on workload size and input budget
        beam_width = max(5, int(num_seqs))  # typical values: 10-15
        # number of candidate extensions considered per partial
        cand_per_expand = max(6, min(15, n // 7 + 6))

        # Initialize beam with diverse random starts
        start_txns = list(range(n))
        random.shuffle(start_txns)
        start_txns = start_txns[:min(beam_width, n)]

        beam = []
        for t in start_txns:
            seq = [t]
            remaining = frozenset(set(range(n)) - {t})
            c = evaluate_seq(seq)
            beam.append((c, seq, remaining))

        # Expand until sequences are complete
        for _ in range(n - 1):
            next_beam = []
            seen = set()  # prevent duplicate seqs in beam
            for cost, seq, remaining in beam:
                rem_list = list(remaining)
                if not rem_list:
                    next_beam.append((cost, seq, remaining))
                    continue

                # Sample subset of candidates to expand
                if len(rem_list) <= cand_per_expand:
                    cand = rem_list
                else:
                    cand = random.sample(rem_list, cand_per_expand)

                for t in cand:
                    new_seq = seq + [t]
                    new_cost = evaluate_seq(new_seq)
                    rem_new = remaining - {t}
                    key = tuple(new_seq)
                    if key in seen:
                        continue
                    seen.add(key)
                    next_beam.append((new_cost, new_seq, rem_new))

            # Fallback in rare case sampling yielded nothing
            if not next_beam:
                # Expand deterministically with best single extension for each beam item
                for cost, seq, remaining in beam:
                    rem_list = list(remaining)
                    best_t = None
                    best_c = float('inf')
                    for t in rem_list:
                        c = evaluate_seq(seq + [t])
                        if c < best_c:
                            best_c = c
                            best_t = t
                    if best_t is not None:
                        new_seq = seq + [best_t]
                        next_beam.append((best_c, new_seq, remaining - {best_t}))

            # Keep top-k partial sequences
            next_beam.sort(key=lambda x: x[0])
            beam = next_beam[:beam_width]

        # Choose best complete sequence from beam
        beam.sort(key=lambda x: x[0])
        best_cost, best_seq, _ = beam[0]
        return best_cost, best_seq

    def local_improve(seq, base_cost=None):
        # Adjacent-swap hill climb and sampled insertion
        best_seq = list(seq)
        best_cost = evaluate_seq(best_seq) if base_cost is None else base_cost
        n = len(best_seq)

        # Adjacent swap passes
        max_passes = 2
        for _ in range(max_passes):
            improved = False
            for i in range(n - 1):
                candidate = best_seq[:]
                candidate[i], candidate[i + 1] = candidate[i + 1], candidate[i]
                c = evaluate_seq(candidate)
                if c < best_cost:
                    best_seq = candidate
                    best_cost = c
                    improved = True
            if not improved:
                break

        # Sampled insertion moves
        tries = min(200, 2 * n)
        for _ in range(tries):
            i, j = sorted(random.sample(range(n), 2))
            if i == j:
                continue
            candidate = best_seq[:]
            item = candidate.pop(j)
            candidate.insert(i, item)
            c = evaluate_seq(candidate)
            if c < best_cost:
                best_seq = candidate
                best_cost = c

        return best_cost, best_seq

    # Run beam search to get high-quality schedule, then refine locally
    beam_cost, beam_seq = beam_search()
    final_cost, final_seq = local_improve(beam_seq, base_cost=beam_cost)
    return final_cost, final_seq


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
    # Slightly larger beam for richer conflict structure
    makespan1, schedule1 = get_best_schedule(workload, 15)
    cost1 = workload.get_opt_seq_cost(schedule1)

    # Workload 2: Simple read-then-write pattern
    workload2 = Workload(WORKLOAD_2)
    makespan2, schedule2 = get_best_schedule(workload2, 12)
    cost2 = workload2.get_opt_seq_cost(schedule2)

    # Workload 3: Minimal read/write operations
    workload3 = Workload(WORKLOAD_3)
    makespan3, schedule3 = get_best_schedule(workload3, 12)
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