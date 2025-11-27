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
    A*-guided beam search with marginal-delta expansion and cache-aware local search.

    Args:
        workload: Workload object containing transaction data
        num_seqs: Exploration budget controlling beam width/branching and local search

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    n = workload.num_txns
    # Deterministic per-workload seed while still diverse across workloads
    seed = 1729 + n
    rng = random.Random(seed)

    # Shared caches across all restarts and phases
    cost_cache = {}
    delta_cache = {}

    def seq_cost(seq):
        key = tuple(seq)
        if key in cost_cache:
            return cost_cache[key]
        c = workload.get_opt_seq_cost(seq)
        cost_cache[key] = c
        return c

    def delta(prefix, t):
        key = (tuple(prefix), t)
        if key in delta_cache:
            return delta_cache[key]
        c0 = seq_cost(prefix)
        c1 = seq_cost(prefix + [t])
        d = c1 - c0
        delta_cache[key] = d
        return d

    # Precompute singleton costs and a conservative lower bound helper
    singleton_cost = [seq_cost([t]) for t in range(n)]
    min_singleton = min(singleton_cost) if singleton_cost else 0
    # Lower bound on remaining: at least the max of (singleton - min_singleton) across remaining
    def lb_remaining(remaining):
        if not remaining:
            return 0
        m = 0
        for t in remaining:
            x = singleton_cost[t] - min_singleton
            if x > m:
                m = x
        if m < 0:
            m = 0
        return m

    # Candidate sampling from remaining with diversity: best singletons + random + worst singletons
    def sample_candidates(remaining, k):
        R = list(remaining)
        if len(R) <= k:
            return R
        # ranks by singleton cost
        R_sorted = sorted(R, key=lambda t: singleton_cost[t])
        k1 = max(1, k // 3)
        k2 = max(1, k // 3)
        k3 = k - k1 - k2
        best_part = R_sorted[: min(k1, len(R_sorted))]
        worst_part = R_sorted[-min(k3, len(R_sorted)) :] if k3 > 0 else []
        remaining_pool = list(set(R) - set(best_part) - set(worst_part))
        rng.shuffle(remaining_pool)
        rand_part = remaining_pool[: min(k2, len(remaining_pool))]
        cand = list(dict.fromkeys(best_part + rand_part + worst_part))  # deduplicate preserve order
        if len(cand) < k:
            # top-up randomly
            extras = list(set(R) - set(cand))
            rng.shuffle(extras)
            cand += extras[: k - len(cand)]
        return cand

    # Greedy completion from a prefix using delta-guided selection
    def greedy_complete(prefix, remaining, sample_k):
        seq = prefix[:]
        rem = set(remaining)
        while rem:
            cand = sample_candidates(rem, sample_k)
            best_t = None
            best_d = float('inf')
            # Evaluate immediate deltas
            for t in cand:
                d = delta(seq, t)
                if d < best_d:
                    best_d = d
                    best_t = t
            if best_t is None:
                # Fallback: pick arbitrary
                best_t = rng.choice(list(rem))
            seq.append(best_t)
            rem.remove(best_t)
        return seq, seq_cost(seq)

    # Core beam A* search
    def run_beam_astar():
        # Config driven by num_seqs and problem size
        beam_width = max(10, min(28, int(1.6 * max(8, int(num_seqs)))))
        branch_factor = max(8, min(14, int(1.0 * max(8, int(num_seqs)))))
        candidate_sample = max(branch_factor * 2, 22)
        two_step_k = min(5, branch_factor)
        next_sample = 8 if n > 60 else 10
        # Periodically try greedy completion to get incumbent early
        refresh_every = 7

        # Seed modes: best singletons, worst singletons, random
        kseed = min(beam_width, n)
        idx_sorted = sorted(range(n), key=lambda t: singleton_cost[t])
        best_seeds = idx_sorted[: max(1, kseed // 3)]
        worst_seeds = idx_sorted[-max(1, kseed // 3) :]
        rem_pool = list(set(range(n)) - set(best_seeds) - set(worst_seeds))
        rng.shuffle(rem_pool)
        rand_seeds = rem_pool[: max(1, kseed - len(best_seeds) - len(worst_seeds))]
        seeds = list(dict.fromkeys(best_seeds + rand_seeds + worst_seeds))[:kseed]

        # Initial beam
        beam = []
        incumbent_cost = float('inf')
        incumbent_seq = None
        for t0 in seeds:
            seq0 = [t0]
            rem0 = set(range(n))
            rem0.remove(t0)
            g0 = seq_cost(seq0)
            h0 = lb_remaining(rem0)
            f0 = g0 + h0
            beam.append((f0, g0, seq0, frozenset(rem0)))
            # quick greedy completion from seeds to set early incumbent
            seq_comp, cost_comp = greedy_complete(seq0, rem0, sample_k=min(18, candidate_sample))
            if cost_comp < incumbent_cost:
                incumbent_cost, incumbent_seq = cost_comp, seq_comp

        depth = 1
        while depth < n and beam:
            next_beam = []
            seen = set()
            # Sort current beam by f then g to expand promising prefixes first
            beam.sort(key=lambda x: (x[0], x[1], len(x[2])))
            for f, g, seq, remaining in beam:
                # Prune with incumbent
                if g + lb_remaining(remaining) >= incumbent_cost:
                    continue
                rem_list = list(remaining)
                if not rem_list:
                    # complete sequence
                    if g < incumbent_cost:
                        incumbent_cost, incumbent_seq = g, seq
                    continue

                # Candidate sampling and immediate Δ
                cand_pool = sample_candidates(rem_list, min(candidate_sample, len(rem_list)))
                deltas = []
                for t in cand_pool:
                    d = delta(seq, t)
                    deltas.append((t, d))
                # order by marginal delta
                deltas.sort(key=lambda x: x[1])
                top_k = deltas[: min(branch_factor, len(deltas))]

                # two-step lookahead on best few
                ranked = []
                for idx, (t, d) in enumerate(top_k):
                    if idx < two_step_k and len(remaining) > 1:
                        # sample next candidates after t
                        next_rem = list(set(rem_list) - {t})
                        if len(next_rem) <= next_sample:
                            next_cands = next_rem
                        else:
                            next_cands = sample_candidates(next_rem, next_sample)
                        best_next = float('inf')
                        seq_t = seq + [t]
                        for u in next_cands:
                            du = delta(seq_t, u)
                            if du < best_next:
                                best_next = du
                        two_step_score = g + d + (best_next if best_next < float('inf') else 0.0)
                        ranked.append((two_step_score, t, d))
                    else:
                        ranked.append((g + d, t, d))

                ranked.sort(key=lambda x: x[0])
                # expand
                for _, t, d in ranked[: min(branch_factor, len(ranked))]:
                    new_seq = seq + [t]
                    new_g = g + d
                    new_remaining = frozenset(set(rem_list) - {t})
                    # prune with incumbent using lower bound on remaining
                    new_h = lb_remaining(new_remaining)
                    new_f = new_g + new_h
                    if new_f >= incumbent_cost:
                        continue
                    key = tuple(new_seq)
                    if key in seen:
                        continue
                    seen.add(key)
                    next_beam.append((new_f, new_g, new_seq, new_remaining))

                # occasional greedy completion to tighten incumbent
                if depth % refresh_every == 0 and next_beam:
                    # try from best current prefix
                    seq_g, rem_g = seq, set(rem_list)
                    seq_comp, cost_comp = greedy_complete(seq_g, rem_g, sample_k=min(16, candidate_sample))
                    if cost_comp < incumbent_cost:
                        incumbent_cost, incumbent_seq = cost_comp, seq_comp

            if not next_beam:
                # No better expansions; fall back to completing best prefix in beam
                # return incumbent if exists, else greedy completion from best current
                if incumbent_seq is not None:
                    return incumbent_cost, incumbent_seq
                # complete best beam prefix
                beam.sort(key=lambda x: (x[0], x[1]))
                _, g, seq, remaining = beam[0]
                seq_comp, cost_comp = greedy_complete(seq, set(remaining), sample_k=min(18, candidate_sample))
                return cost_comp, seq_comp

            # Keep top beam_width
            next_beam.sort(key=lambda x: (x[0], x[1]))
            beam = next_beam[:beam_width]
            depth += 1

        # If we exit loop, pick best between incumbent and beam completions
        if incumbent_seq is not None:
            return incumbent_cost, incumbent_seq
        if beam:
            beam.sort(key=lambda x: (x[0], x[1]))
            _, g, seq, remaining = beam[0]
            seq_comp, cost_comp = greedy_complete(seq, set(remaining), sample_k=min(18, candidate_sample))
            return cost_comp, seq_comp
        # Fallback: trivial order
        trivial = list(range(n))
        return seq_cost(trivial), trivial

    # Local search finisher using cached evaluations
    def local_search(seq, base_cost):
        best_seq = seq[:]
        best_cost = base_cost
        nloc = len(best_seq)

        # Adjacent swap passes
        no_improve_rounds = 0
        for _ in range(2):
            improved = False
            for i in range(nloc - 1):
                cand = best_seq[:]
                cand[i], cand[i + 1] = cand[i + 1], cand[i]
                c = seq_cost(cand)
                if c < best_cost:
                    best_seq, best_cost = cand, c
                    improved = True
            if not improved:
                no_improve_rounds += 1
            else:
                no_improve_rounds = 0
            if no_improve_rounds >= 2:
                break

        # Sampled relocation moves
        move_budget = min(80, nloc) + max(0, int(num_seqs // 2))
        tries = 0
        non_improve = 0
        while tries < move_budget and non_improve < 12:
            i, j = rng.sample(range(nloc), 2)
            if i == j:
                continue
            cand = best_seq[:]
            item = cand.pop(i)
            cand.insert(j, item)
            c = seq_cost(cand)
            if c < best_cost:
                best_seq, best_cost = cand, c
                non_improve = 0
            else:
                non_improve += 1
            tries += 1

        return best_cost, best_seq

    # Multi-restart with structured diversity; reuse caches across restarts
    restarts = 3
    incumbent_best = float('inf')
    incumbent_seq = None
    for _ in range(restarts):
        cost_b, seq_b = run_beam_astar()
        cost_l, seq_l = local_search(seq_b, cost_b)
        if cost_l < incumbent_best:
            incumbent_best, incumbent_seq = cost_l, seq_l

    return incumbent_best, incumbent_seq


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
    makespan1, schedule1 = get_best_schedule(workload, 16)
    cost1 = workload.get_opt_seq_cost(schedule1)

    # Workload 2: Simple read-then-write pattern
    workload2 = Workload(WORKLOAD_2)
    makespan2, schedule2 = get_best_schedule(workload2, 14)
    cost2 = workload2.get_opt_seq_cost(schedule2)

    # Workload 3: Minimal read/write operations
    workload3 = Workload(WORKLOAD_3)
    makespan3, schedule3 = get_best_schedule(workload3, 14)
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