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
    Pairwise-tournament ranking + iterated local search (ILS) to minimize makespan.

    Args:
        workload: Workload object containing transaction data
        num_seqs: Exploration budget knob (affects refinement intensity)

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    rng = random.Random()
    rng.seed(time.time())

    # Memoized evaluator for sequences to reduce simulator calls
    cost_cache = {}

    def evaluate_seq(seq):
        key = tuple(seq)
        if key in cost_cache:
            return cost_cache[key]
        c = workload.get_opt_seq_cost(seq)
        cost_cache[key] = c
        return c

    n = workload.num_txns
    txns = list(range(n))

    # Phase 1: Build pairwise tournament from cheap simulator calls (length 1 and 2)
    # c1[i] = cost([i])
    c1 = [0] * n
    for i in range(n):
        c1[i] = evaluate_seq([i])

    # M[i][j] = cost([i, j]) for i != j, diagonal unused
    # This is O(n^2) short evaluations; captures read/write conflicts as they affect ordering.
    M = [[0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i == j:
                M[i][j] = c1[i]
            else:
                M[i][j] = evaluate_seq([i, j])

    # Preference weight w[i][j] = M[i][j] - M[j][i]; negative means i should precede j
    # Score s[i] = sum_j w[i][j]; smaller s[i] => better to place earlier
    s = [0] * n
    for i in range(n):
        total = 0
        row_i = M[i]
        for j in range(n):
            if i == j:
                continue
            total += (row_i[j] - M[j][i])
        s[i] = total

    # Initial order by tournament score
    order = list(range(n))
    order.sort(key=lambda x: (s[x], x))

    # Tournament comparator (cheap, uses precomputed M)
    def prefer_before(a, b):
        # True if placing a before b is no worse than b before a
        return M[a][b] <= M[b][a]

    # Cheap tournament-based local sorting: bubble-like passes under pairwise preference
    # Resolves many local inversions without calling the full simulator.
    max_tournament_passes = 4
    for _ in range(max_tournament_passes):
        improved = False
        for k in range(n - 1):
            a, b = order[k], order[k + 1]
            if not prefer_before(a, b):
                order[k], order[k + 1] = b, a
                improved = True
        if not improved:
            break

    # Phase 2: Real-cost refinement (ILS with modest budget)
    best_seq = order[:]
    best_cost = evaluate_seq(best_seq)

    # Adjacent swap pass using true cost
    max_true_passes = 1
    for _ in range(max_true_passes):
        improved = False
        for k in range(n - 1):
            cand = best_seq[:]
            cand[k], cand[k + 1] = cand[k + 1], cand[k]
            c = evaluate_seq(cand)
            if c < best_cost:
                best_seq = cand
                best_cost = c
                improved = True
        if not improved:
            break

    # Ruin-and-Recreate: remove a block and greedily reinsert using sampled positions (true cost)
    # Budget scales lightly with num_seqs
    block_size = max(6, min(20, n // 5))
    rr_tries = max(1, min(2, int(num_seqs // 10) + 1))

    for _ in range(rr_tries):
        if n <= block_size:
            break
        start = rng.randint(0, n - block_size)
        removed = best_seq[start:start + block_size]
        base = best_seq[:start] + best_seq[start + block_size:]

        # Reinsert each removed txn at a good position chosen from a small random sample of positions
        for t in rng.sample(removed, len(removed)):
            # Sample positions uniformly plus ensure ends are included
            positions = set()
            positions.add(0)
            positions.add(len(base))
            sample_k = min(6, len(base) + 1)
            while len(positions) < sample_k:
                positions.add(rng.randint(0, len(base)))
            best_pos = 0
            best_pos_cost = float('inf')
            for pos in positions:
                cand = base[:pos] + [t] + base[pos:]
                c = evaluate_seq(cand)
                if c < best_pos_cost:
                    best_pos_cost = c
                    best_pos = pos
            base.insert(best_pos, t)

        # Accept if improved
        cand_cost = evaluate_seq(base)
        if cand_cost < best_cost:
            best_seq = base
            best_cost = cand_cost

    # Final sampled insertion moves (true cost) with modest budget
    move_budget = min(80, n) + max(0, int(num_seqs // 2))
    for _ in range(move_budget):
        i, j = sorted(rng.sample(range(n), 2))
        cand = best_seq[:]
        item = cand.pop(j)
        cand.insert(i, item)
        c = evaluate_seq(cand)
        if c < best_cost:
            best_seq = cand
            best_cost = c

    return best_cost, best_seq


def get_random_costs():
    """
    Evaluate scheduling algorithm on three different workloads.

    Returns:
        Tuple of (total_makespan, list_of_schedules, execution_time)
    """
    start_time = time.time()

    # Workload 1: Complex mixed read/write transactions
    workload = Workload(WORKLOAD_1)
    # Allow a bit more budget for richer conflicts
    makespan1, schedule1 = get_best_schedule(workload, 16)
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