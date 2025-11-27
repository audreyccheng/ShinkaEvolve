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
    Beam + marginal-cost construction with memoization and quick local refinement.

    Args:
        workload: Workload object containing transaction data
        num_seqs: Number of restarts to try (used as intensity parameter)

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    N = workload.num_txns
    start_time = time.time()
    # Keep budget conservative to preserve overall combined score
    time_budget_sec = 0.32 + min(0.18, 0.001 * max(0, N))

    def time_left():
        return (time.time() - start_time) < time_budget_sec

    # Shared caches per workload call
    cost_cache = {}
    delta_cache = {}

    def eval_cost(seq):
        key = tuple(seq)
        c = cost_cache.get(key)
        if c is None:
            c = workload.get_opt_seq_cost(seq)
            cost_cache[key] = c
        return c

    def ext_cost(prefix_tuple, cand):
        key = (prefix_tuple, cand)
        c = delta_cache.get(key)
        if c is None:
            c = eval_cost(list(prefix_tuple) + [cand])
            delta_cache[key] = c
        return c

    # Beam/branch parameters (adaptive with N)
    beam_width = min(20, max(6, N // 9 if N > 0 else 6))
    branch_factor = min(14, max(6, N // 12 if N > 0 else 6))
    lookahead_k = 3

    incumbent_cost = float('inf')
    incumbent_seq = None

    def greedy_finish(seq, rem_set):
        seq_out = list(seq)
        rem = set(rem_set)
        cur_cost = eval_cost(seq_out)
        while rem and time_left():
            prefix = tuple(seq_out)
            # Sampled candidate pool for speed
            if len(rem) > branch_factor:
                cand_pool = random.sample(list(rem), branch_factor)
            else:
                cand_pool = list(rem)
            best_t = None
            best_c = float('inf')
            for t in cand_pool:
                c = ext_cost(prefix, t)
                if c < best_c:
                    best_c = c
                    best_t = t
            if best_t is None:
                best_t = cand_pool[0]
                best_c = ext_cost(prefix, best_t)
            seq_out.append(best_t)
            rem.remove(best_t)
            cur_cost = best_c
            # Early prune if no better than incumbent
            if cur_cost >= incumbent_cost:
                break
        return cur_cost, seq_out

    def run_beam():
        nonlocal incumbent_cost, incumbent_seq
        all_txns = list(range(N))
        # Seed beam with best singletons from a random pool
        pool = random.sample(all_txns, min(len(all_txns), max(beam_width * 2, 8))) if N > 0 else []
        singles = []
        for t in pool:
            if not time_left():
                break
            c = eval_cost([t])
            singles.append((c, [t], set(x for x in all_txns if x != t)))
        if not singles:
            seq = all_txns[:]
            random.shuffle(seq)
            return eval_cost(seq), seq

        singles.sort(key=lambda x: x[0])
        beam = singles[:beam_width]

        steps = N - 1
        for _ in range(steps):
            if not time_left():
                break
            expanded = []
            for cost_so_far, seq, rem in beam:
                if not rem:
                    expanded.append((cost_so_far, seq, rem, cost_so_far))
                    continue
                if cost_so_far >= incumbent_cost:
                    continue
                rem_list = list(rem)
                # Pre-select a slightly larger pool and then cut to branch_factor
                if len(rem_list) > branch_factor * 3:
                    rem_list = random.sample(rem_list, branch_factor * 3)

                prefix = tuple(seq)
                scored = []
                for t in rem_list:
                    if not time_left():
                        break
                    ec = ext_cost(prefix, t)
                    scored.append((ec - cost_so_far, ec, t))
                if not scored:
                    continue
                scored.sort(key=lambda x: x[0])

                top = scored[:min(branch_factor, len(scored))]
                for _, ec, t in top:
                    new_seq = seq + [t]
                    new_rem = rem.copy()
                    new_rem.remove(t)

                    # Shallow lookahead: score by best next extension in a small sample
                    la_score = ec
                    if new_rem and time_left():
                        la_pool = list(new_rem)
                        la_sample = la_pool if len(la_pool) <= lookahead_k else random.sample(la_pool, lookahead_k)
                        best_la = float('inf')
                        new_prefix = tuple(new_seq)
                        for nx in la_sample:
                            c2 = ext_cost(new_prefix, nx)
                            if c2 < best_la:
                                best_la = c2
                        la_score = min(la_score, best_la)
                    expanded.append((ec, new_seq, new_rem, la_score))

            if not expanded:
                break

            # Keep top by lookahead score; carry forward actual prefix cost
            expanded.sort(key=lambda x: x[3])
            new_beam = []
            seen = set()
            for ec, s, r, _sc in expanded:
                key = tuple(s)
                if key in seen:
                    continue
                seen.add(key)
                new_beam.append((ec, s, r))
                if len(new_beam) >= beam_width:
                    break
            beam = new_beam

            # Periodically try to complete the best prefix to tighten incumbent
            if beam and time_left():
                cand = min(beam, key=lambda x: x[0])
                c_try, s_try = greedy_finish(cand[1], cand[2])
                if len(s_try) == N and c_try < incumbent_cost:
                    incumbent_cost, incumbent_seq = c_try, s_try

        # Greedily finish all prefixes; pick best
        best_cost_local = incumbent_cost
        best_seq_local = incumbent_seq
        for c, s, r in beam:
            if not time_left():
                break
            c2, s2 = greedy_finish(s, r)
            if len(s2) == N and c2 < best_cost_local:
                best_cost_local, best_seq_local = c2, s2

        if best_seq_local is None:
            seq = all_txns[:]
            random.shuffle(seq)
            return eval_cost(seq), seq
        return best_cost_local, best_seq_local

    global_best_cost = float('inf')
    global_best_seq = None
    restarts = max(1, int(num_seqs))

    for _ in range(restarts):
        if not time_left():
            break
        cost, seq = run_beam()

        # Lightweight local improvement: adjacent swaps and sampled insertions
        if time_left():
            improved = True
            while improved and time_left():
                improved = False
                for i in range(len(seq) - 1):
                    if not time_left():
                        break
                    cand = seq[:]
                    cand[i], cand[i + 1] = cand[i + 1], cand[i]
                    c = eval_cost(cand)
                    if c < cost:
                        seq, cost = cand, c
                        improved = True

        if time_left():
            trials = 40
            while trials > 0 and time_left():
                trials -= 1
                i = random.randrange(len(seq))
                j = random.randrange(len(seq))
                if i == j:
                    continue
                cand = seq[:]
                val = cand.pop(i)
                cand.insert(j, val)
                c = eval_cost(cand)
                if c < cost:
                    seq, cost = cand, c

        if cost < global_best_cost:
            global_best_cost, global_best_seq = cost, seq

    if global_best_seq is None:
        global_best_seq = list(range(N))
        random.shuffle(global_best_seq)
        global_best_cost = eval_cost(global_best_seq)

    return global_best_cost, global_best_seq


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