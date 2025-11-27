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
    Enhanced multi-start greedy with incumbent pruning, adaptive lookahead,
    and targeted Variable Neighborhood Search (VNS) local improvement.

    Args:
        workload: Workload object containing transaction data
        num_seqs: Number of randomized restarts to attempt

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    n = workload.num_txns
    # Deterministic RNG per workload for stability
    rng = random.Random(1729 + n)

    best_cost = float('inf')
    best_seq = None

    # Memoization for sequence costs to avoid recomputation
    cost_cache = {}

    def seq_cost(seq):
        key = tuple(seq)
        if key in cost_cache:
            return cost_cache[key]
        c = workload.get_opt_seq_cost(seq)
        cost_cache[key] = c
        return c

    # Precompute singleton costs to seed better starting points
    singleton_costs = [(t, seq_cost([t])) for t in range(n)]
    singleton_costs.sort(key=lambda x: x[1])

    restarts = max(1, int(num_seqs))

    for r in range(restarts):
        # Diversify starts: bias to good singletons but keep randomness
        if r < min(5, n):
            k = min(10, n)
            t0 = rng.choice([t for t, _ in singleton_costs[:k]])
        else:
            t0 = rng.randint(0, n - 1)

        seq = [t0]
        remaining = [t for t in range(n) if t != t0]

        # Greedy build: evaluate all candidates and use adaptive lookahead to break ties
        step = 0
        while remaining:
            # Incumbent-based pruning: if partial already not better, abort this restart
            base_cost = seq_cost(seq)
            if best_cost < float('inf') and base_cost >= best_cost:
                # Abort current restart early; no completion can beat incumbent
                seq = None
                break

            cand_costs = []
            base = seq
            for t in remaining:
                c = seq_cost(base + [t])
                cand_costs.append((t, c))
            cand_costs.sort(key=lambda x: x[1])

            # Adaptive lookahead: slightly deeper in early steps
            L = min(4 if step < 10 else 3, len(cand_costs))
            top_cands = cand_costs[:L]

            chosen_t = top_cands[0][0]
            best_pair_cost = None
            # Dynamic next-sample size: smaller when many remain
            R = len(remaining)
            if n <= 60:
                base_la = 6
            else:
                base_la = 4
            lookahead_samples = base_la if R <= 60 else max(3, base_la - 1)

            for t, immediate_c in top_cands:
                if R == 1:
                    la_cost = immediate_c
                else:
                    next_pool = [x for x in remaining if x != t]
                    if len(next_pool) <= lookahead_samples:
                        sampled_next = next_pool
                    else:
                        sampled_next = rng.sample(next_pool, lookahead_samples)
                    la_cost = min(seq_cost(base + [t, u]) for u in sampled_next)
                if best_pair_cost is None or la_cost < best_pair_cost:
                    best_pair_cost = la_cost
                    chosen_t = t

            seq.append(chosen_t)
            remaining.remove(chosen_t)
            step += 1

        if seq is None:
            # Pruned restart
            continue

        # Local improvement phase
        current_cost = seq_cost(seq)

        # 1) Adjacent swap hill-climb passes
        for _ in range(2):
            any_improve = False
            for i in range(n - 1):
                seq[i], seq[i + 1] = seq[i + 1], seq[i]
                c = seq_cost(seq)
                if c < current_cost:
                    current_cost = c
                    any_improve = True
                else:
                    # revert if no improvement
                    seq[i], seq[i + 1] = seq[i + 1], seq[i]
            if not any_improve:
                break

        # 1b) Limited non-adjacent 2-opt swaps (sampled)
        swap_attempts = min(n, 60)
        for _ in range(swap_attempts):
            i, j = rng.sample(range(n), 2)
            if abs(i - j) <= 1:
                continue
            seq[i], seq[j] = seq[j], seq[i]
            c = seq_cost(seq)
            if c < current_cost:
                current_cost = c
            else:
                # revert
                seq[i], seq[j] = seq[j], seq[i]

        # 2) Random insertion improvements with accept-if-better
        attempts = min(150, 3 * n)
        for _ in range(attempts):
            i = rng.randint(0, n - 1)
            j = rng.randint(0, n - 1)
            if i == j:
                continue
            t = seq.pop(i)
            seq.insert(j, t)
            c = seq_cost(seq)
            if c < current_cost:
                current_cost = c
            else:
                # revert
                seq.pop(j)
                seq.insert(i, t)

        # 2b) Targeted relocate of high-marginal transactions
        # Compute prefix marginals to locate "spiky" contributors
        prefix_costs = [0] * n
        accum = 0
        for i in range(n):
            accum = seq_cost(seq[: i + 1])
            prefix_costs[i] = accum
        marg = [prefix_costs[0]] + [prefix_costs[i] - prefix_costs[i - 1] for i in range(1, n)]
        # Try moving top-3 marginals to sampled positions
        hot_positions = sorted(range(n), key=lambda idx: marg[idx], reverse=True)[:3]
        for pos in hot_positions:
            if pos >= len(seq):
                continue
            t = seq[pos]
            base = seq[:pos] + seq[pos + 1:]
            best_pos_idx = None
            best_pos_cost = current_cost
            # Try ends plus a few random positions
            positions_try = {0, len(base)}
            while len(positions_try) < 6:
                positions_try.add(rng.randint(0, len(base)))
            for j in positions_try:
                cand = base[:j] + [t] + base[j:]
                c = seq_cost(cand)
                if c < best_pos_cost:
                    best_pos_cost = c
                    best_pos_idx = j
            if best_pos_idx is not None and best_pos_cost < current_cost:
                seq = base[:best_pos_idx] + [t] + base[best_pos_idx:]
                current_cost = best_pos_cost

        # 3) Light ruin-and-recreate with greedy reinsertion
        if n > 12:
            block_size = max(5, min(18, n // 6))
            rr_tries = 1 if n >= 90 else 2
            for _ in range(rr_tries):
                start = rng.randint(0, n - block_size)
                removed = seq[start:start + block_size]
                base = seq[:start] + seq[start + block_size:]

                for t in rng.sample(removed, len(removed)):
                    # Try ends + random positions
                    pos_candidates = {0, len(base)}
                    while len(pos_candidates) < 6:
                        pos_candidates.add(rng.randint(0, len(base)))
                    best_pos = 0
                    best_pos_cost = float('inf')
                    for pos in pos_candidates:
                        cand = base[:pos] + [t] + base[pos:]
                        c = seq_cost(cand)
                        if c < best_pos_cost:
                            best_pos_cost = c
                            best_pos = pos
                    base.insert(best_pos, t)

                c = seq_cost(base)
                if c < current_cost:
                    seq = base
                    current_cost = c

        if current_cost < best_cost:
            best_cost = current_cost
            best_seq = seq[:]

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