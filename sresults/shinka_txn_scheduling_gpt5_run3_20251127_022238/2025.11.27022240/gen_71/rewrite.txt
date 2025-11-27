# EVOLVE-BLOCK-START
"""Transaction scheduling algorithm for optimizing makespan across multiple workloads"""

import time
import random
import sys
import os
from collections import defaultdict

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
    Dominance-pruned stochastic beam search with memoized extension costs
    and shared prefix-dominance across restarts. Minimal greedy completion
    is used only for finalizing prefixes to keep runtime low.

    Args:
        workload: Workload object containing transaction data
        num_seqs: Number of restarts (bounded by time and capped to a small portfolio)

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    N = workload.num_txns
    start_time = time.time()
    # Tight time budget for better combined score; mildly larger for bigger workloads
    time_budget_sec = 0.54 if N >= 90 else 0.46

    def time_left():
        return (time.time() - start_time) < time_budget_sec

    # Deterministic RNG; perturb per restart
    base_rng = random.Random(1729 + 37 * N)

    # Caches shared across restarts to reduce recomputation
    cost_cache = {}
    ext_cache = {}

    def eval_seq_cost(seq):
        key = tuple(seq)
        c = cost_cache.get(key)
        if c is None:
            c = workload.get_opt_seq_cost(seq)
            cost_cache[key] = c
        return c

    def eval_ext_cost(prefix_tuple, cand):
        key = (prefix_tuple, cand)
        c = ext_cache.get(key)
        if c is None:
            c = eval_seq_cost(list(prefix_tuple) + [cand])
            ext_cache[key] = c
        return c

    all_txns = list(range(N))

    # Precompute singleton costs and ordering for seeding and a cheap LB
    singleton_cost = {}
    for t in all_txns:
        if not time_left():
            break
        singleton_cost[t] = eval_seq_cost([t])
    singles_sorted = sorted(all_txns, key=lambda t: singleton_cost.get(t, float('inf')))

    # Weak LB: current_cost bounded by max remaining singleton cost
    def lb_singleton(cur_cost, rem_set):
        if not rem_set:
            return cur_cost
        m = 0
        for t in rem_set:
            c = singleton_cost.get(t)
            if c is None:
                c = eval_seq_cost([t])
                singleton_cost[t] = c
            if c > m:
                m = c
        return max(cur_cost, m)

    # Greedy finalization from a prefix: pick the next that minimizes immediate extension cost
    # Sample-limited to keep it cheap.
    def greedy_complete(seq, rem_set, branch_k):
        seq_out = list(seq)
        rem = set(rem_set)
        cur_cost = eval_seq_cost(seq_out) if seq_out else 0
        while rem and time_left():
            rem_list = list(rem)
            if len(rem_list) <= branch_k:
                pool = rem_list
            else:
                pool = base_rng.sample(rem_list, branch_k)
            pt = tuple(seq_out)
            best_t, best_c = None, float('inf')
            for t in pool:
                c = eval_ext_cost(pt, t)
                if c < best_c:
                    best_c, best_t = c, t
            if best_t is None:
                # Time pressure fallback: append remaining and evaluate once
                seq_out.extend(rem)
                cur_cost = eval_seq_cost(seq_out)
                return cur_cost, seq_out
            seq_out.append(best_t)
            rem.remove(best_t)
            cur_cost = best_c
        if rem:
            seq_out.extend(list(rem))
            cur_cost = eval_seq_cost(seq_out)
        return cur_cost, seq_out

    # Global prefix-dominance shared across restarts
    prefix_dom = {}
    k_suffix = 3

    def dom_sig(rem_set, seq):
        tail = tuple(seq[-k_suffix:]) if len(seq) >= k_suffix else tuple(seq)
        return (frozenset(rem_set), tail)

    # Core single beam run
    def run_beam_once(rng, incumbent=float('inf')):
        if N <= 1:
            seq0 = all_txns[:]
            return eval_seq_cost(seq0), seq0

        # Moderate beam and branching to keep runtime low and stable
        beam_width = min(max(6, N // 9), 24)
        branch_factor = min(max(6, N // 10), 18)
        # Candidate sampling per parent
        sample_mult = 1  # evaluate exactly branch_factor random children per prefix

        # Initialize beam with best singletons and a little diversity
        init_pool_size = min(N, beam_width * 2)
        seed_pool = singles_sorted[:init_pool_size]
        # Add a few random seeds if available
        extras = [x for x in all_txns if x not in seed_pool]
        if extras:
            seed_pool += rng.sample(extras, min(4, len(extras)))
        # Deduplicate
        seen_seed = set()
        seeds = []
        for t in seed_pool:
            if t in seen_seed:
                continue
            seen_seed.add(t)
            seq = [t]
            rem = set(all_txns)
            rem.remove(t)
            c = eval_seq_cost(seq)
            seeds.append((c, seq, rem))

        if not seeds:
            seq = all_txns[:]
            rng.shuffle(seq)
            return eval_seq_cost(seq), seq

        seeds.sort(key=lambda x: x[0])
        beam = seeds[:beam_width]

        best_full_cost = incumbent
        best_full_seq = None

        steps = N - 1
        depth = 0
        greedy_probe_every = 0  # set >0 to enable occasional mid-run full greedy probes

        while depth < steps and time_left():
            depth += 1
            new_beam = []

            # Optional mid-run greedy probe to tighten incumbent (disabled to save time)
            if greedy_probe_every and depth % greedy_probe_every == 0:
                c_try, s_try = greedy_complete(beam[0][1], beam[0][2], branch_k=max(10, branch_factor))
                if c_try < best_full_cost:
                    best_full_cost, best_full_seq = c_try, s_try

            for cost_so_far, seq, rem in beam:
                if not rem:
                    # Full sequence already complete
                    if cost_so_far < best_full_cost:
                        best_full_cost, best_full_seq = cost_so_far, seq[:]
                    # Keep as a child so others can be pruned by incumbent
                    new_beam.append((cost_so_far, seq, rem, cost_so_far))
                    continue

                # Incumbent pruning on prefix and weak LB
                if cost_so_far >= best_full_cost:
                    continue
                if lb_singleton(cost_so_far, rem) >= best_full_cost:
                    continue

                rem_list = list(rem)
                # Select candidate set by random sampling; keep small, no heavy lookahead
                if len(rem_list) <= branch_factor:
                    candidates = rem_list
                else:
                    sample_size = min(len(rem_list), branch_factor * sample_mult)
                    candidates = rng.sample(rem_list, sample_size)

                pt = tuple(seq)
                for cand in candidates:
                    if not time_left():
                        break
                    new_cost = eval_ext_cost(pt, cand)
                    # Early prune by incumbent
                    if new_cost >= best_full_cost:
                        continue
                    new_seq = seq + [cand]
                    new_rem = rem.copy()
                    new_rem.remove(cand)

                    # Prefix dominance pruning
                    sig = dom_sig(new_rem, new_seq)
                    prev = prefix_dom.get(sig)
                    if prev is not None and new_cost >= prev:
                        continue
                    # Update dominance map
                    if prev is None or new_cost < prev:
                        prefix_dom[sig] = new_cost

                    new_beam.append((new_cost, new_seq, new_rem, new_cost))

            if not new_beam:
                break

            # Keep top unique prefixes by current prefix cost
            new_beam.sort(key=lambda x: x[3])
            next_beam = []
            seen = set()
            for _, s, r, score in new_beam:
                key = tuple(s)
                if key in seen:
                    continue
                seen.add(key)
                next_beam.append((score, s, r))
                if len(next_beam) >= beam_width:
                    break
            beam = next_beam

        # Finalization: greedily complete remaining top prefixes to update incumbent
        for cost_so_far, seq, rem in beam:
            if not time_left():
                break
            if not rem:
                if cost_so_far < best_full_cost:
                    best_full_cost, best_full_seq = cost_so_far, seq[:]
                continue
            c_fin, s_fin = greedy_complete(seq, rem, branch_k=max(10, branch_factor))
            if c_fin < best_full_cost:
                best_full_cost, best_full_seq = c_fin, s_fin

        # Safety fallback if still none
        if best_full_seq is None:
            seq = all_txns[:]
            rng.shuffle(seq)
            best_full_seq = seq
            best_full_cost = eval_seq_cost(seq)

        # Very cheap adjacent-swap pass (single sweep) to tighten cost slightly
        n = len(best_full_seq)
        for i in range(max(0, n - 12)):  # limit to reduce cost; swap early region mostly matters
            if not time_left():
                break
            cand = best_full_seq[:]
            cand[i], cand[i + 1] = cand[i + 1], cand[i]
            c = eval_seq_cost(cand)
            if c < best_full_cost:
                best_full_cost = c
                best_full_seq = cand

        return best_full_cost, best_full_seq

    # Multi-restart portfolio with shared caches and prefix-dom
    global_best_cost = float('inf')
    global_best_seq = None

    restarts = min(4, max(2, int(num_seqs)))
    for r in range(restarts):
        if not time_left():
            break
        rng = random.Random(base_rng.randint(1, 10**9) + 97 * r)
        c, s = run_beam_once(rng, incumbent=global_best_cost)
        if c < global_best_cost:
            global_best_cost, global_best_seq = c, s

    # Safety: ensure permutation validity
    if global_best_seq is None or len(global_best_seq) != N or len(set(global_best_seq)) != N:
        if global_best_seq is None:
            seq = list(range(N))
            random.shuffle(seq)
            global_best_seq = seq
        # Repair duplicates or missing
        seen = set()
        repaired = []
        for t in global_best_seq:
            if 0 <= t < N and t not in seen:
                repaired.append(t)
                seen.add(t)
        for t in range(N):
            if t not in seen:
                repaired.append(t)
        global_best_seq = repaired[:N]
        global_best_cost = eval_seq_cost(global_best_seq)

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