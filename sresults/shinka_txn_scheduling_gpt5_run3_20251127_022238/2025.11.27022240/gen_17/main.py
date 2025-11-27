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
    Beam search with A*-style lower-bound pruning, marginal-cost ordering,
    shallow lookahead, shared memoized caches across restarts, multi-prefix
    greedy completion, and a stronger sliding-window local refinement.

    Args:
        workload: Workload object containing transaction data
        num_seqs: Number of random restarts for the beam search

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    N = workload.num_txns
    # Time budget per workload to balance quality and speed
    time_budget_sec = 0.55
    start_time = time.time()

    def time_left():
        return (time.time() - start_time) < time_budget_sec

    # Shared caches across all restarts
    cost_cache = {}
    ext_cache = {}

    def eval_seq_cost(seq):
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
        c = eval_seq_cost(list(prefix_tuple) + [cand])
        ext_cache[key] = c
        return c

    # Precompute singleton costs for lower bounds and seeding
    singleton_cost = {}
    for t in range(N):
        if not time_left():
            break
        singleton_cost[t] = eval_seq_cost([t])

    def max_singleton_rem(rem_set):
        if not rem_set:
            return 0
        # Compute max singleton among remaining; small overhead given beam widths
        m = 0
        for t in rem_set:
            c = singleton_cost.get(t)
            if c is None:
                c = eval_seq_cost([t])
                singleton_cost[t] = c
            if c > m:
                m = c
        return m

    def greedy_finish(seq, rem_set, branch_factor, incumbent=None):
        seq_out = list(seq)
        rem = set(rem_set)
        cur_cost = eval_seq_cost(seq_out)
        while rem and time_left():
            # Early prune if LB already worse than incumbent
            if incumbent is not None:
                lb_here = max(cur_cost, max_singleton_rem(rem))
                if lb_here >= incumbent:
                    break
            rem_list = list(rem)
            k = min(branch_factor, len(rem_list))
            cand_pool = random.sample(rem_list, k) if len(rem_list) > k else rem_list
            best_t = None
            best_c = float('inf')
            prefix_tuple = tuple(seq_out)
            for t in cand_pool:
                c = eval_ext_cost(prefix_tuple, t)
                if c < best_c:
                    best_c = c
                    best_t = t
            if best_t is None:
                best_t = rem_list[0]
                best_c = eval_ext_cost(prefix_tuple, best_t)
            seq_out.append(best_t)
            rem.remove(best_t)
            cur_cost = best_c
        if rem:
            # If time ran out or pruned, append the rest arbitrarily
            for t in list(rem):
                seq_out.append(t)
            cur_cost = eval_seq_cost(seq_out)
        return cur_cost, seq_out

    def run_beam_search():
        # Dynamically size the beam and branching according to problem size
        beam_width = min(max(8, N // 7), 40)
        branch_factor = min(max(8, N // 9), 28)
        lookahead_k = 3

        all_txns = list(range(N))
        # Precompute a few singleton costs to seed
        init_pool_size = min(len(all_txns), max(beam_width * 2, 8))
        init_candidates = random.sample(all_txns, init_pool_size) if init_pool_size > 0 else all_txns[:]
        beam = []
        for t in init_candidates:
            if not time_left():
                break
            seq = [t]
            rem = set(all_txns)
            rem.remove(t)
            cost = eval_seq_cost(seq)
            beam.append((cost, seq, rem))
        if not beam:
            seq = all_txns[:]
            random.shuffle(seq)
            return eval_seq_cost(seq), seq

        # Keep best initial seeds
        beam.sort(key=lambda x: x[0])
        beam = beam[:max(1, min(beam_width, len(beam)))]

        incumbent_cost = float('inf')
        incumbent_seq = None

        # Expand the beam until full sequences are built or time elapses
        steps = N - 1
        for _ in range(steps):
            if not time_left():
                break
            new_beam = []
            # Tighten incumbent early by greedily completing top prefixes
            if beam and time_left():
                # Try top-K by current prefix cost
                beam_sorted_for_finish = sorted(beam, key=lambda x: x[0])[:min(3, len(beam))]
                for cost_so_far, seq_b, rem_b in beam_sorted_for_finish:
                    if not time_left():
                        break
                    c_try, s_try = greedy_finish(seq_b, rem_b, branch_factor, incumbent_cost)
                    if len(s_try) == N and c_try < incumbent_cost:
                        incumbent_cost, incumbent_seq = c_try, s_try

            for cost_so_far, seq, rem in beam:
                # If no remaining, update incumbent and carry forward
                if not rem:
                    new_beam.append((cost_so_far, seq, rem))
                    if cost_so_far < incumbent_cost:
                        incumbent_cost, incumbent_seq = cost_so_far, seq[:]
                    continue

                # A*-style prefix lower bound prune
                if cost_so_far >= incumbent_cost:
                    continue
                lb_prefix = max(cost_so_far, max_singleton_rem(rem))
                if lb_prefix >= incumbent_cost:
                    continue

                rem_list = list(rem)
                # Create a candidate pool and sort by marginal delta
                pool_size = min(len(rem_list), branch_factor * 3)
                cand_pool = random.sample(rem_list, pool_size) if len(rem_list) > pool_size else rem_list

                scored = []
                prefix_tuple = tuple(seq)
                for cand in cand_pool:
                    if not time_left():
                        break
                    ext_cost = eval_ext_cost(prefix_tuple, cand)
                    delta = ext_cost - cost_so_far
                    scored.append((delta, ext_cost, cand))
                if not scored:
                    continue
                scored.sort(key=lambda x: x[0])
                top_cands = scored[:min(branch_factor, len(scored))]

                # Append candidates; use LB + shallow lookahead to rank
                for _, ext_cost, cand in top_cands:
                    if not time_left():
                        break
                    new_seq = seq + [cand]
                    new_rem = rem.copy()
                    new_rem.remove(cand)

                    # Child lower bound for pruning
                    lb_child = max(ext_cost, max_singleton_rem(new_rem))
                    if lb_child >= incumbent_cost:
                        continue

                    # Shallow lookahead ranking
                    rank_score = lb_child
                    if new_rem and time_left():
                        la_pool = list(new_rem)
                        la_sample = la_pool if len(la_pool) <= lookahead_k else random.sample(la_pool, lookahead_k)
                        best_la = float('inf')
                        new_prefix_tuple = tuple(new_seq)
                        for nxt in la_sample:
                            c2 = eval_ext_cost(new_prefix_tuple, nxt)
                            if c2 < best_la:
                                best_la = c2
                        # Use the better of LB and lookahead cost for ranking
                        rank_score = min(rank_score, best_la)

                    new_beam.append((ext_cost, new_seq, new_rem, rank_score))

            if not new_beam:
                break

            # Keep only the top beam_width unique prefixes by rank_score
            new_beam.sort(key=lambda x: x[3])
            unique = []
            seen = set()
            for entry in new_beam:
                key = tuple(entry[1])
                if key in seen:
                    continue
                seen.add(key)
                # store actual prefix cost ext_cost for further expansion
                unique.append((entry[0], entry[1], entry[2]))
                if len(unique) >= beam_width:
                    break
            beam = unique

        # Complete remaining prefixes greedily and pick the best
        best_cost = incumbent_cost
        best_seq_local = incumbent_seq
        for cost_so_far, seq, rem in beam:
            if not time_left():
                break
            c_fin, s_fin = greedy_finish(seq, rem, branch_factor, incumbent_cost)
            if len(s_fin) == N and c_fin < best_cost:
                best_cost, best_seq_local = c_fin, s_fin

        if best_seq_local is None:
            # Fallback to a random permutation
            seq = all_txns[:]
            random.shuffle(seq)
            best_seq_local = seq
            best_cost = eval_seq_cost(seq)
        return best_cost, best_seq_local

    def local_improve(seq, current_cost):
        # Adjacent swaps, sliding-window insertions, and limited random insertions
        best_seq = seq[:]
        best_cost = current_cost

        # Adjacent swap hill climbing (single pass)
        for i in range(len(best_seq) - 1):
            if not time_left():
                break
            cand = best_seq[:]
            cand[i], cand[i + 1] = cand[i + 1], cand[i]
            c = eval_seq_cost(cand)
            if c < best_cost:
                best_cost = c
                best_seq = cand

        # Sliding-window insertion refinement
        Nloc = len(best_seq)
        for w in (7, 9):
            if not time_left():
                break
            step = max(1, w // 2)
            start_idx = 0
            while start_idx < Nloc and time_left():
                end_idx = min(Nloc, start_idx + w)
                improved_window = True
                # Iterate until no improvement within this window or time runs out
                iter_guard = 0
                while improved_window and time_left() and iter_guard < 2:
                    improved_window = False
                    iter_guard += 1
                    for i in range(start_idx, end_idx):
                        if not time_left():
                            break
                        for j in range(start_idx, end_idx):
                            if i == j:
                                continue
                            cand = best_seq[:]
                            val = cand.pop(i)
                            cand.insert(j, val)
                            c = eval_seq_cost(cand)
                            if c < best_cost:
                                best_cost = c
                                best_seq = cand
                                # Update positions after change
                                Nloc = len(best_seq)
                                improved_window = True
                start_idx += step

        # Random insertion moves (bounded)
        trials = 50
        while trials > 0 and time_left():
            trials -= 1
            i, j = random.sample(range(len(best_seq)), 2)
            if i == j:
                continue
            cand = best_seq[:]
            val = cand.pop(i)
            cand.insert(j, val)
            c = eval_seq_cost(cand)
            if c < best_cost:
                best_cost = c
                best_seq = cand

        return best_cost, best_seq

    global_best_cost = float('inf')
    global_best_seq = None

    # Multiple random restarts for robustness, bounded by time
    restarts = max(1, int(num_seqs))
    r = 0
    while r < restarts and time_left():
        cost, seq = run_beam_search()
        if time_left():
            cost, seq = local_improve(seq, cost)
        if cost < global_best_cost:
            global_best_cost, global_best_seq = cost, seq
        r += 1

    if global_best_seq is None:
        # Fallback to a random permutation
        seq = list(range(N))
        random.shuffle(seq)
        global_best_seq = seq
        global_best_cost = eval_seq_cost(seq)

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