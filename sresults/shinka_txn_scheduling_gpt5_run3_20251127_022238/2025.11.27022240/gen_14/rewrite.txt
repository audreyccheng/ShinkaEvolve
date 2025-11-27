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
    A*-style marginal-cost beam with shared caches, lookahead, incumbent pruning,
    and bounded local refinement.

    Args:
        workload: Workload object containing transaction data
        num_seqs: Number of restarts (diversity-driven, deterministic)

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """

    N = workload.num_txns

    # Total time budget per workload to keep overall runtime in check
    # The rest of the code respects this budget.
    time_budget_sec = 0.60
    start_time = time.time()

    def time_left():
        return (time.time() - start_time) < time_budget_sec

    # Shared caches across all restarts
    cost_cache = {}
    delta_cache = {}

    def eval_cost(seq):
        key = tuple(seq)
        c = cost_cache.get(key)
        if c is not None:
            return c
        c = workload.get_opt_seq_cost(seq)
        cost_cache[key] = c
        return c

    def eval_ext_cost(prefix_tuple, cand):
        key = (prefix_tuple, cand)
        c = delta_cache.get(key)
        if c is not None:
            return c
        seq = list(prefix_tuple) + [cand]
        c = eval_cost(seq)
        delta_cache[key] = c
        return c

    # Precompute singleton costs as both seeding and lower bound data
    singleton_cost = {}
    for t in range(N):
        if not time_left():
            break
        singleton_cost[t] = eval_cost([t])

    # A cheap admissible-ish lower bound for remaining is max singleton among remaining
    def lower_bound_remaining(rem):
        if not rem:
            return 0
        return max(singleton_cost.get(i, 0) for i in rem)

    # A simple greedy finish using marginal-cost selections
    def greedy_finish(seq, rem_set, incumbent):
        seq_out = list(seq)
        rem = set(rem_set)
        cur_cost = eval_cost(seq_out)
        while rem and time_left():
            prefix_tuple = tuple(seq_out)
            # Sort by marginal delta; sample if large set
            rem_list = list(rem)
            k = min(len(rem_list), branch_factor)
            candidates = rem_list if len(rem_list) <= k else random.sample(rem_list, k)
            best_cand = None
            best_cost = float('inf')
            for t in candidates:
                c = eval_ext_cost(prefix_tuple, t)
                if c < best_cost:
                    best_cost = c
                    best_cand = t
            if best_cand is None:
                # fallback
                best_cand = rem_list[0]
                best_cost = eval_ext_cost(prefix_tuple, best_cand)
            seq_out.append(best_cand)
            rem.remove(best_cand)
            cur_cost = best_cost
            if cur_cost >= incumbent:
                # pruning in greedy completion
                break
        return cur_cost, seq_out

    # Local refinement: adjacent swaps + limited insertion moves
    def local_improve(seq, current_cost, incumbent):
        best_seq = list(seq)
        best_cost = current_cost

        # Adjacent swap pass
        for i in range(len(best_seq) - 1):
            if not time_left():
                break
            cand = best_seq[:]
            cand[i], cand[i + 1] = cand[i + 1], cand[i]
            c = eval_cost(cand)
            if c < best_cost:
                best_cost = c
                best_seq = cand
                if best_cost <= incumbent:
                    incumbent = best_cost

        # Limited insertion moves with accept-if-improves
        trials = 50
        no_improve = 0
        while trials > 0 and time_left() and no_improve < 8:
            trials -= 1
            i = random.randrange(len(best_seq))
            j = random.randrange(len(best_seq))
            if i == j:
                continue
            cand = best_seq[:]
            val = cand.pop(i)
            cand.insert(j, val)
            c = eval_cost(cand)
            if c < best_cost:
                best_cost = c
                best_seq = cand
                no_improve = 0
                if best_cost <= incumbent:
                    incumbent = best_cost
            else:
                no_improve += 1

        return best_cost, best_seq

    # Core A*-style marginal beam search
    def run_beam_search(seed_mode, restart_id):
        # seed_mode in {"cheapest", "expensive", "random"} controls initial seeds
        all_txns = list(range(N))

        # Beam/branch settings adaptive to N
        beam_w = min(40, max(8, N // 5))
        branch = min(18, max(6, N // 10))
        lookahead_k = 5

        # Expose to outer scope for greedy_finish
        nonlocal branch_factor
        branch_factor = branch

        # Incumbent best for pruning within this restart; use global incumbent across restarts via closure
        nonlocal incumbent_cost, incumbent_seq

        # Build initial seeds
        seeds = []
        if seed_mode == "cheapest":
            order = sorted(all_txns, key=lambda t: singleton_cost.get(t, float('inf')))
            seeds = order[:min(beam_w * 2, len(order))]
        elif seed_mode == "expensive":
            order = sorted(all_txns, key=lambda t: singleton_cost.get(t, -float('inf')), reverse=True)
            seeds = order[:min(beam_w * 2, len(order))]
        else:  # random
            pick = min(max(beam_w * 2, 8), len(all_txns))
            seeds = random.sample(all_txns, pick)

        # Evaluate seeds
        beam = []
        for t in seeds:
            if not time_left():
                break
            seq = [t]
            rem = set(all_txns)
            rem.remove(t)
            g = eval_cost(seq)
            # prune by incumbent
            if g >= incumbent_cost:
                continue
            h = lower_bound_remaining(rem)
            f = g + h
            beam.append((f, g, seq, rem))

        if not beam:
            # fallback to a single random seed if all pruned
            t = random.choice(all_txns)
            seq = [t]
            rem = set(all_txns)
            rem.remove(t)
            g = eval_cost(seq)
            beam = [(g + lower_bound_remaining(rem), g, seq, rem)]

        # Main expansion
        steps = N - 1
        for _ in range(steps):
            if not time_left():
                break
            new_beam = []
            # Sort by f to expand promising prefixes first
            beam.sort(key=lambda x: x[0])
            for f, g, seq, rem in beam:
                if not rem:
                    # Full sequence
                    if g < incumbent_cost:
                        incumbent_cost, incumbent_seq = g, seq[:]
                    new_beam.append((f, g, seq, rem))
                    continue
                if g >= incumbent_cost:
                    continue

                prefix_tuple = tuple(seq)
                rem_list = list(rem)

                # Score candidates by marginal delta; expand only top branch candidates
                deltas = []
                for cand in rem_list:
                    if not time_left():
                        break
                    ext_cost = eval_ext_cost(prefix_tuple, cand)
                    delta = ext_cost - g
                    deltas.append((delta, ext_cost, cand))
                if not deltas:
                    continue
                deltas.sort(key=lambda x: x[0])
                top = deltas[:min(branch, len(deltas))]

                # Two-step lookahead for final ranking on a small subset
                for _, ext_cost, cand in top:
                    if not time_left():
                        break
                    new_seq = seq + [cand]
                    new_rem = rem.copy()
                    new_rem.remove(cand)
                    # light lookahead: try a few best next deltas
                    la_score = ext_cost
                    if new_rem:
                        new_prefix_tuple = tuple(new_seq)
                        # Evaluate next-step deltas for a small sample of remaining
                        la_pool = list(new_rem)
                        if len(la_pool) > lookahead_k:
                            # choose by singleton to bias toward good next candidates
                            la_pool = sorted(la_pool, key=lambda t: singleton_cost.get(t, 0))[:lookahead_k]
                        best_next = float('inf')
                        for nxt in la_pool:
                            c2 = eval_ext_cost(new_prefix_tuple, nxt)
                            if c2 < best_next:
                                best_next = c2
                        la_score = best_next

                    g2 = ext_cost
                    if g2 >= incumbent_cost:
                        continue
                    h2 = lower_bound_remaining(new_rem)
                    # Use a blended priority: max of f and la_score as tie-breaker
                    f2 = g2 + h2
                    # Push entry storing both f2 and la_score for sorting
                    new_beam.append((max(f2, la_score), g2, new_seq, new_rem))

            if not new_beam:
                break

            # Keep top unique prefixes by f
            new_beam.sort(key=lambda x: x[0])
            unique = []
            seen = set()
            for entry in new_beam:
                key = tuple(entry[2])
                if key in seen:
                    continue
                seen.add(key)
                unique.append(entry)
                if len(unique) >= beam_w:
                    break
            beam = unique

            # Periodically greedily complete the current best prefix to tighten incumbent
            if beam and time_left():
                best_prefix = min(beam, key=lambda x: x[0])
                c_try, s_try = greedy_finish(best_prefix[2], best_prefix[3], incumbent_cost)
                if len(s_try) == N and c_try < incumbent_cost:
                    incumbent_cost, incumbent_seq = c_try, s_try

        # Finish all prefixes greedily; pick best
        best_cost = incumbent_cost
        best_seq_local = incumbent_seq
        for f, g, seq, rem in beam:
            if not time_left():
                break
            c, s = greedy_finish(seq, rem, incumbent_cost)
            if len(s) == N and c < best_cost:
                best_cost, best_seq_local = c, s

        # Fallback
        if best_seq_local is None:
            seq = all_txns[:]
            random.shuffle(seq)
            best_seq_local = seq
            best_cost = eval_cost(seq)

        return best_cost, best_seq_local

    # Global incumbent across restarts
    incumbent_cost = float('inf')
    incumbent_seq = None

    # Diversity-driven deterministic restarts
    restarts = max(1, int(num_seqs))
    # Structured seeding modes to diversify
    modes = ["cheapest", "random", "expensive"]

    # Deterministic seeding for reproducibility
    base_seed = 1729 + N
    for r in range(restarts):
        if not time_left():
            break
        random.seed(base_seed + 997 * r)
        mode = modes[r % len(modes)]
        # Vary beam/branch implicitly with randomness and mode; run
        cost_r, seq_r = run_beam_search(mode, r)
        # Local improvement
        if time_left():
            cost_r, seq_r = local_improve(seq_r, cost_r, incumbent_cost)
        if cost_r < incumbent_cost:
            incumbent_cost, incumbent_seq = cost_r, seq_r

    if incumbent_seq is None:
        # ultimate fallback: random
        incumbent_seq = list(range(N))
        random.shuffle(incumbent_seq)
        incumbent_cost = eval_cost(incumbent_seq)

    return incumbent_cost, incumbent_seq


def get_random_costs():
    """
    Evaluate scheduling algorithm on three different workloads.

    Returns:
        Tuple of (total_makespan, list_of_schedules, execution_time)
    """
    start_time = time.time()

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