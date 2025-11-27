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
    Lookahead-guided greedy with memoized costs, multi-start seeding, and
    lightweight local refinement to minimize makespan.
    """
    N = workload.num_txns
    start_time = time.time()
    # Per-workload time budget to keep runtime reasonable
    time_budget_sec = 0.6

    def time_left():
        return (time.time() - start_time) < time_budget_sec

    # Shared caches across construction and refinement
    cost_cache = {}
    ext_cache = {}

    def eval_cost(seq):
        key = tuple(seq)
        cached = cost_cache.get(key)
        if cached is not None:
            return cached
        c = workload.get_opt_seq_cost(seq)
        cost_cache[key] = c
        return c

    def eval_ext_cost(prefix_tuple, cand):
        key = (prefix_tuple, cand)
        cached = ext_cache.get(key)
        if cached is not None:
            return cached
        c = eval_cost(list(prefix_tuple) + [cand])
        ext_cache[key] = c
        return c

    # Warm up with singleton costs (useful as seeds and light LB)
    singleton_cost = {}
    for t in range(N):
        if not time_left():
            break
        singleton_cost[t] = eval_cost([t])

    def construct(seed=None):
        # Choose a seed; default to best singleton
        if seed is None:
            seed = min(range(N), key=lambda x: singleton_cost.get(x, float('inf')))
        seq = [seed]
        rem = set(range(N))
        rem.remove(seed)
        cur_cost = singleton_cost.get(seed, eval_cost([seed]))

        # Adaptive parameters
        prefilter = min(24, max(8, N // 8))   # number of candidates to sample/prefilter
        rcl = 5                                # restricted candidate list size
        lookahead_k = 4                        # next-step sample size

        while rem and time_left():
            rem_list = list(rem)
            prefix_tuple = tuple(seq)

            # Prefilter candidates by marginal cost (delta)
            if len(rem_list) > prefilter:
                pool = random.sample(rem_list, prefilter * 2)
            else:
                pool = rem_list

            scored = []
            for t in pool:
                ec = eval_ext_cost(prefix_tuple, t)
                scored.append((ec - cur_cost, ec, t))

            if not scored:
                # If time ran out, append arbitrary and continue
                t = rem_list[0]
                seq.append(t)
                rem.remove(t)
                cur_cost = eval_ext_cost(prefix_tuple, t)
                continue

            scored.sort(key=lambda x: x[0])
            top = scored[:min(rcl, len(scored))]

            # Two-step lookahead: for each top candidate, peek one more step
            best_t = None
            best_ec = None
            best_score = float('inf')
            for _, ec, t in top:
                new_prefix = tuple(seq + [t])
                new_rem = rem.copy()
                new_rem.remove(t)
                if not new_rem:
                    la_score = ec
                else:
                    la_list = list(new_rem)
                    k = min(lookahead_k, len(la_list))
                    la_sample = random.sample(la_list, k)
                    best_la = float('inf')
                    for nxt in la_sample:
                        c2 = eval_ext_cost(new_prefix, nxt)
                        if c2 < best_la:
                            best_la = c2
                    la_score = best_la
                # Rank by the tighter of immediate extension and lookahead
                score = min(ec, la_score)
                if score < best_score:
                    best_score = score
                    best_t = t
                    best_ec = ec

            seq.append(best_t)
            rem.remove(best_t)
            cur_cost = best_ec

        final_cost = eval_cost(seq)
        return final_cost, seq

    def local_improve(seq, cur_cost):
        best_seq = seq[:]
        best_cost = cur_cost

        # One pass of adjacent swaps
        for i in range(len(best_seq) - 1):
            if not time_left():
                break
            cand = best_seq[:]
            cand[i], cand[i + 1] = cand[i + 1], cand[i]
            c = eval_cost(cand)
            if c < best_cost:
                best_cost = c
                best_seq = cand

        # Limited random insertion moves
        trials = min(60, 2 * N)
        while trials > 0 and time_left():
            trials -= 1
            i, j = random.sample(range(len(best_seq)), 2)
            if i == j:
                continue
            cand = best_seq[:]
            val = cand.pop(i)
            cand.insert(j, val)
            c = eval_cost(cand)
            if c < best_cost:
                best_cost = c
                best_seq = cand

        return best_cost, best_seq

    # Multi-start: use a few best singletons + random seeds
    best_cost = float('inf')
    best_seq = None
    restarts = max(3, int(num_seqs))

    seeds = sorted(range(N), key=lambda t: singleton_cost.get(t, float('inf')))[:min(4, N)]
    # Add random seeds for diversity
    while len(seeds) < restarts and len(seeds) < N:
        t = random.randrange(N)
        if t not in seeds:
            seeds.append(t)

    for seed in seeds:
        if not time_left():
            break
        c, s = construct(seed)
        if time_left():
            c, s = local_improve(s, c)
        if c < best_cost:
            best_cost, best_seq = c, s

    if best_seq is None:
        # Fallback
        seq = list(range(N))
        random.shuffle(seq)
        best_seq = seq
        best_cost = eval_cost(seq)

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