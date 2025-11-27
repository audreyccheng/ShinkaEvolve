# EVOLVE-BLOCK-START
"""Transaction scheduling algorithm for optimizing makespan across multiple workloads"""

import time
import random
import sys
import os

# Add the openevolve_examples directory to the path to import txn_simulator and workloads
# Find the repository root by looking for openevolve_examples directory
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
    Multi-start greedy with lookahead and Variable Neighborhood Search (VNS) including
    ruin-and-recreate reinsertion. Uses memoization to minimize simulator calls.

    Args:
        workload: Workload object containing transaction data
        num_seqs: Exploration budget; controls restarts and local search intensity

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    rng = random.Random()
    rng.seed(time.time())

    n = workload.num_txns

    # Global memoized evaluator for partial sequences to reduce simulator calls
    cost_cache = {}

    def eval_cost(seq):
        key = tuple(seq)
        if key in cost_cache:
            return cost_cache[key]
        c = workload.get_opt_seq_cost(seq)
        cost_cache[key] = c
        return c

    # Precompute singleton costs for better seeding
    singleton = [(t, eval_cost([t])) for t in range(n)]
    singleton.sort(key=lambda x: x[1])

    # Budgeting knobs derived from num_seqs and problem size
    # Limit restarts to avoid excessive runtime on large instances
    restarts = max(3, min(int(num_seqs), 8 if n >= 90 else 12))

    # Candidate sampling size during greedy construction when many remain
    sample_large = 12 if n >= 90 else 16
    sample_small_threshold = max(35, n // 3)  # below this, evaluate all remaining
    # Lookahead configuration
    lookahead_top = 3
    lookahead_samples_large = 4
    lookahead_samples_small = 6

    best_global_cost = float('inf')
    best_global_seq = None

    def greedy_build(seed_choice):
        # Seed selection
        if seed_choice == "best":
            k = min(10, n)
            t0 = rng.choice([t for t, _ in singleton[:k]])
        elif seed_choice == "median":
            k = min(30, n)
            t0 = rng.choice([t for t, _ in singleton[k // 2:k]])
        else:
            t0 = rng.randint(0, n - 1)

        seq = [t0]
        remaining = [t for t in range(n) if t != t0]

        # Greedy build with limited lookahead
        while remaining:
            R = len(remaining)

            # Candidate pool: either all remaining if small, else a random sample
            if R <= sample_small_threshold:
                cand_pool = remaining
            else:
                k = min(sample_large, R)
                cand_pool = rng.sample(remaining, k)

            # Evaluate immediate costs for candidate pool
            immediate = []
            base = seq
            for t in cand_pool:
                c = eval_cost(base + [t])
                immediate.append((t, c))
            immediate.sort(key=lambda x: x[1])

            # Take top few for lookahead
            L = min(lookahead_top, len(immediate))
            chosen_t = immediate[0][0]
            best_pair_eval = immediate[0][1]  # fallback if no lookahead improves

            for idx in range(L):
                t, immediate_c = immediate[idx]
                if R == 1:
                    la_cost = immediate_c
                else:
                    next_pool = [x for x in remaining if x != t]
                    if not next_pool:
                        la_cost = immediate_c
                    else:
                        # Sample next positions
                        if len(next_pool) <= lookahead_samples_small:
                            sampled_next = next_pool
                        else:
                            sample_cnt = lookahead_samples_large if R > sample_small_threshold else lookahead_samples_small
                            sampled_next = rng.sample(next_pool, sample_cnt)
                        la_cost = min(eval_cost(base + [t, u]) for u in sampled_next)
                if la_cost < best_pair_eval:
                    best_pair_eval = la_cost
                    chosen_t = t

            seq.append(chosen_t)
            remaining.remove(chosen_t)

        final_cost = eval_cost(seq)
        return final_cost, seq

    def local_improve(seq, base_cost=None):
        # Variable Neighborhood Search with:
        # - Adjacent swap passes
        # - Sampled insertion moves
        # - Ruin-and-recreate reinsertion
        best_seq = seq[:]
        best_cost = eval_cost(best_seq) if base_cost is None else base_cost
        nloc = len(best_seq)

        # 1) Adjacent swap hill-climb passes
        for _ in range(3):
            improved = False
            for i in range(nloc - 1):
                cand = best_seq[:]
                cand[i], cand[i + 1] = cand[i + 1], cand[i]
                c = eval_cost(cand)
                if c < best_cost:
                    best_seq, best_cost = cand, c
                    improved = True
            if not improved:
                break

        # 2) Sampled insertion (relocate) moves
        # Budget scales modestly with problem size and num_seqs
        relocate_attempts = min(2 * nloc, 180) + max(0, int(num_seqs // 3))
        for _ in range(relocate_attempts):
            i, j = rng.sample(range(nloc), 2)
            if i == j:
                continue
            cand = best_seq[:]
            item = cand.pop(j)
            cand.insert(i, item)
            c = eval_cost(cand)
            if c < best_cost:
                best_seq, best_cost = cand, c

        # 3) Ruin-and-Recreate (block removal, greedy reinsertion with sampled positions)
        # Keep this light to cap runtime
        if nloc > 12:
            block_size = max(5, min(18, nloc // 6))
            rr_tries = 1 if nloc >= 90 else 2
            for _ in range(rr_tries):
                start = rng.randint(0, nloc - block_size)
                removed = best_seq[start:start + block_size]
                base = best_seq[:start] + best_seq[start + block_size:]

                for t in rng.sample(removed, len(removed)):
                    # Sample positions including ends
                    pos_candidates = {0, len(base)}
                    sample_k = min(6, len(base) + 1)
                    while len(pos_candidates) < sample_k:
                        pos_candidates.add(rng.randint(0, len(base)))
                    best_pos = 0
                    best_pos_cost = float('inf')
                    for pos in pos_candidates:
                        cand = base[:pos] + [t] + base[pos:]
                        c = eval_cost(cand)
                        if c < best_pos_cost:
                            best_pos_cost = c
                            best_pos = pos
                    base.insert(best_pos, t)

                c = eval_cost(base)
                if c < best_cost:
                    best_seq, best_cost = base, c

        return best_cost, best_seq

    # Multi-start: diversify seeds between good singletons and randomness
    start_modes = ["best", "median", "random"]
    for r in range(restarts):
        mode = start_modes[r % len(start_modes)]
        g_cost, g_seq = greedy_build(mode)
        l_cost, l_seq = local_improve(g_seq, base_cost=g_cost)
        if l_cost < best_global_cost:
            best_global_cost, best_global_seq = l_cost, l_seq

    return best_global_cost, best_global_seq


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
    # Slightly larger exploration for richer conflict structure
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