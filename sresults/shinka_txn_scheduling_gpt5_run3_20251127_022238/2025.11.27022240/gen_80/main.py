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
    Hybrid buddy-guided beam search with memoized extension costs, greedy incumbent
    tightening, and lightweight VNS refinement to minimize makespan.

    - Beam search expands prefixes using a candidate pool biased by sampled buddies
      and low-singleton transactions. Ranking uses true extension costs.
    - Early and final greedy completions update an incumbent to prune expansions.
    - Multiple restarts for robustness.
    - A compact local search (adjacent swaps + relocations) improves the final schedule.

    Args:
        workload: Workload object containing transaction data
        num_seqs: Number of restarts to try (also bounded by time)

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    N = workload.num_txns
    rng = random.Random(1729 + 7 * N)

    start_time = time.time()
    # Keep execution fast for combined score; budget scales mildly with N
    time_budget_sec = 0.56 if N >= 90 else 0.48

    def time_left():
        return (time.time() - start_time) < time_budget_sec

    # Caches for true costs
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

    # Precompute singleton costs to guide candidates and provide weak LB
    singleton_cost = {}
    for t in all_txns:
        if not time_left():
            break
        singleton_cost[t] = eval_seq_cost([t])
    singles_sorted = sorted(all_txns, key=lambda t: singleton_cost.get(t, float('inf')))

    # Build small buddy lists by sampled pairwise extension deltas
    # pair_delta(a,b) = cost([a,b]) - cost([a])
    pair_delta_cache = {}
    def pair_delta(a, b):
        key = (a, b)
        v = pair_delta_cache.get(key)
        if v is not None:
            return v
        base = singleton_cost.get(a)
        if base is None:
            base = eval_seq_cost([a])
            singleton_cost[a] = base
        ec = eval_ext_cost((a,), b)
        d = ec - base
        pair_delta_cache[key] = d
        return d

    def build_buddies(max_buddies=8, sample_per_t=14):
        buddies = {t: [] for t in all_txns}
        # Candidate pool blending: top by singleton + random
        top_slice = singles_sorted[:min(22, max(8, N // 5))]
        for t in all_txns:
            if not time_left():
                break
            pool = [u for u in top_slice if u != t]
            others = [u for u in all_txns if u != t and u not in pool]
            if others:
                pool.extend(rng.sample(others, min(sample_per_t, len(others))))
            # Dedup
            seen = set()
            cand_list = []
            for u in pool:
                if u not in seen:
                    seen.add(u)
                    cand_list.append(u)
            scored = []
            for u in cand_list:
                if not time_left():
                    break
                scored.append((pair_delta(t, u), u))
            scored.sort(key=lambda x: x[0])  # lower marginal delta preferred
            buddies[t] = [u for _d, u in scored[:max_buddies]]
        return buddies

    buddies = build_buddies(max_buddies=8, sample_per_t=14)

    # Weak lower bound using max remaining singleton cost
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

    # Global prefix-dominance map to prune equivalent states across restarts
    # Keyed by (frozenset(remaining), suffix of recent k txns)
    prefix_dom = {}

    def dom_sig(rem_set, seq, k=3):
        tail = tuple(seq[-k:]) if len(seq) >= k else tuple(seq)
        return (frozenset(rem_set), tail)

    # Greedy completion from a prefix guided by buddies and extension costs
    def greedy_finish(prefix, rem_set, branch_factor=12, incumbent=None):
        seq = list(prefix)
        rem = set(rem_set)
        cur_cost = eval_seq_cost(seq) if seq else 0
        while rem and time_left():
            if incumbent is not None and lb_singleton(cur_cost, rem) >= incumbent:
                break
            rem_list = list(rem)
            # Build candidate pool: buddies of last + low-singleton + random fill
            pool = []
            if seq:
                last = seq[-1]
                for u in buddies.get(last, []):
                    if u in rem and u not in pool:
                        pool.append(u)
            low_single = sorted(rem_list, key=lambda t: singleton_cost.get(t, float('inf')))[:min(5, len(rem_list))]
            for u in low_single:
                if u not in pool:
                    pool.append(u)
            need = max(0, branch_factor - len(pool))
            if need > 0:
                others = [x for x in rem_list if x not in pool]
                if others:
                    pool.extend(rng.sample(others, min(need, len(others))))
            if not pool:
                pool = rem_list if len(rem_list) <= branch_factor else rng.sample(rem_list, branch_factor)

            prefix_tuple = tuple(seq)
            best_t, best_c = None, float('inf')
            for t in pool:
                c = eval_ext_cost(prefix_tuple, t)
                if c < best_c:
                    best_c, best_t = c, t
            if best_t is None:
                # Time exhausted or no pool; append arbitrary
                seq.extend(rem)
                cur_cost = eval_seq_cost(seq)
                return cur_cost, seq
            seq.append(best_t)
            rem.remove(best_t)
            cur_cost = best_c

        if rem:
            seq.extend(list(rem))
            cur_cost = eval_seq_cost(seq)
        return cur_cost, seq

    # Core beam search with incumbent tightening, guided candidate pools
    def run_beam_once():
        beam_width = min(28, max(6, N // 8))
        branch_factor = min(20, max(6, N // 10))
        lookahead_k = 3

        # Seed beam with best singletons and a few randoms for diversity
        seeds = []
        top_seeds = singles_sorted[:max(beam_width * 2, 6)]
        for t in top_seeds:
            seq = [t]
            rem = set(all_txns)
            rem.remove(t)
            c = eval_seq_cost(seq)
            seeds.append((c, seq, rem))
        others = [x for x in all_txns if x not in top_seeds]
        if others:
            for t in rng.sample(others, min(4, len(others))):
                seq = [t]
                rem = set(all_txns) - {t}
                c = eval_seq_cost(seq)
                seeds.append((c, seq, rem))

        seeds.sort(key=lambda x: x[0])
        beam = seeds[:beam_width] if seeds else []

        if not beam:
            seq = all_txns[:]
            rng.shuffle(seq)
            return eval_seq_cost(seq), seq

        incumbent_cost = float('inf')
        incumbent_seq = None

        # Early incumbent via greedy finish of best prefix
        if time_left():
            c_try, s_try = greedy_finish(beam[0][1], beam[0][2], branch_factor=max(8, branch_factor), incumbent=incumbent_cost)
            if len(s_try) == N and c_try < incumbent_cost:
                incumbent_cost, incumbent_seq = c_try, s_try

        steps = N - 1
        depth = 0
        for _ in range(steps):
            if not time_left():
                break
            depth += 1
            new_beam = []

            # Periodically tighten incumbent by completing top-k prefixes
            for (cost_so_far, seq, rem) in beam[:min(2, len(beam))]:
                if not time_left():
                    break
                c_try, s_try = greedy_finish(seq, rem, branch_factor=max(8, branch_factor), incumbent=incumbent_cost)
                if len(s_try) == N and c_try < incumbent_cost:
                    incumbent_cost, incumbent_seq = c_try, s_try

            for cost_so_far, seq, rem in beam:
                if not rem:
                    # full sequence
                    if cost_so_far < incumbent_cost:
                        incumbent_cost, incumbent_seq = cost_so_far, seq[:]
                    new_beam.append((cost_so_far, seq, rem, cost_so_far))
                    continue

                # incumbent pruning on prefix and simple LB
                if cost_so_far >= incumbent_cost:
                    continue
                if lb_singleton(cost_so_far, rem) >= incumbent_cost:
                    continue

                # Prefix-dominance pruning using global map
                sig = dom_sig(rem, seq, k=3 if depth < int(0.7 * N) else 2)
                prev_best = prefix_dom.get(sig)
                if prev_best is not None and cost_so_far >= prev_best:
                    continue
                # update dominance with the better cost
                if prev_best is None or cost_so_far < prev_best:
                    prefix_dom[sig] = cost_so_far

                rem_list = list(rem)
                # Build candidate pool: buddies of last + low-singleton + random fill (limit to 2*branch)
                cand_pool = []
                if seq:
                    last = seq[-1]
                    cand_pool.extend([x for x in buddies.get(last, []) if x in rem])
                low_s = sorted(rem_list, key=lambda t: singleton_cost.get(t, float('inf')))[:min(5, len(rem_list))]
                for u in low_s:
                    if u not in cand_pool:
                        cand_pool.append(u)
                need = max(0, 2 * branch_factor - len(cand_pool))
                if need > 0:
                    others = [x for x in rem_list if x not in cand_pool]
                    if others:
                        cand_pool.extend(rng.sample(others, min(need, len(others))))
                if not cand_pool:
                    cand_pool = rem_list if len(rem_list) <= 2 * branch_factor else rng.sample(rem_list, 2 * branch_factor)

                prefix_tuple = tuple(seq)
                scored = []
                best_immediate = float('inf')
                # First compute immediate extension costs
                tmp = []
                for t in cand_pool:
                    if not time_left():
                        break
                    ec = eval_ext_cost(prefix_tuple, t)
                    if ec >= incumbent_cost:
                        continue
                    tmp.append((ec, t))
                    if ec < best_immediate:
                        best_immediate = ec
                if not tmp:
                    continue
                # Rank primarily by extension cost; add slight bias for delta smoothness
                for ec, t in tmp:
                    delta = ec - cost_so_far
                    rank = (ec, delta)
                    scored.append((rank, ec, t))
                scored.sort(key=lambda x: (x[0][0], x[0][1]))
                top = scored[:min(branch_factor, len(scored))]

                kept_any = False
                # For top children, compute shallow lookahead and greedy probe to prune
                for idx, (_rank, ec, t) in enumerate(top):
                    new_seq = seq + [t]
                    new_rem = rem.copy()
                    new_rem.remove(t)

                    # child LB prune
                    if lb_singleton(ec, new_rem) >= incumbent_cost:
                        continue

                    # shallow lookahead from child
                    la_score = ec
                    if new_rem and time_left():
                        la_pool = [u for u in buddies.get(t, []) if u in new_rem]
                        if not la_pool:
                            la_pool = list(new_rem)
                        if len(la_pool) > lookahead_k:
                            la_pool = rng.sample(la_pool, lookahead_k)
                        new_prefix_tuple = tuple(new_seq)
                        for nxt in la_pool:
                            c2 = eval_ext_cost(new_prefix_tuple, nxt)
                            if c2 < la_score:
                                la_score = c2

                    # greedy probe for first few children to update/prune against incumbent
                    if idx < 2 and time_left():
                        g_cost, g_seq = greedy_finish(new_seq, new_rem, branch_factor=max(8, branch_factor), incumbent=incumbent_cost)
                        if len(g_seq) == N and g_cost < incumbent_cost:
                            incumbent_cost, incumbent_seq = g_cost, g_seq
                        la_score = min(la_score, g_cost)
                        if g_cost >= incumbent_cost:
                            # prune this child since its completion isn't competitive
                            # continue to next child but ensure at least one child is kept overall
                            pass

                    new_beam.append((ec, new_seq, new_rem, la_score))
                    kept_any = True

                # Ensure at least best immediate child is kept if all pruned
                if not kept_any and tmp:
                    ec0, t0 = min(tmp, key=lambda x: x[0])
                    new_seq = seq + [t0]
                    new_rem = rem.copy()
                    new_rem.remove(t0)
                    new_beam.append((ec0, new_seq, new_rem, ec0))

            if not new_beam:
                break

            # Keep top unique prefixes by lookahead-adjusted score
            new_beam.sort(key=lambda x: x[3])
            unique = []
            seen = set()
            for entry in new_beam:
                key = tuple(entry[1])
                if key in seen:
                    continue
                seen.add(key)
                unique.append((entry[0], entry[1], entry[2]))
                if len(unique) >= beam_width:
                    break
            beam = unique

        # Final greedy completion of remaining prefixes
        best_cost = incumbent_cost
        best_seq = incumbent_seq
        for cost_so_far, seq, rem in beam:
            if not time_left():
                break
            c_fin, s_fin = greedy_finish(seq, rem, branch_factor=max(8, branch_factor), incumbent=incumbent_cost)
            if len(s_fin) == N and c_fin < best_cost:
                best_cost, best_seq = c_fin, s_fin

        if best_seq is None:
            seq = all_txns[:]
            rng.shuffle(seq)
            best_seq = seq
            best_cost = eval_seq_cost(seq)
        return best_cost, best_seq

    # Lightweight local improvement: adjacent swaps and limited relocations
    def local_improve(seq, cur_cost):
        best_seq = seq[:]
        best_cost = cur_cost
        n = len(best_seq)

        # Two passes of adjacent swaps as hill climb
        for _ in range(2):
            improved = False
            for i in range(n - 1):
                if not time_left():
                    break
                cand = best_seq[:]
                cand[i], cand[i + 1] = cand[i + 1], cand[i]
                c = eval_seq_cost(cand)
                if c < best_cost:
                    best_cost = c
                    best_seq = cand
                    improved = True
            if not improved or not time_left():
                break

        # Helper: boundary preference margin using length-2 costs
        def pair_pref(a, b):
            return eval_seq_cost([a, b]) - eval_seq_cost([b, a])

        # Targeted relocations focusing near worst boundaries
        trials = 40
        while trials > 0 and time_left():
            trials -= 1
            # bias pick near worst adjacency
            worst_i = None
            worst_val = -float('inf')
            for i in range(n - 1):
                v = pair_pref(best_seq[i], best_seq[i + 1])
                if v > worst_val:
                    worst_val = v
                    worst_i = i
            if worst_i is None:
                i, j = rng.sample(range(n), 2)
            else:
                # choose i around worst_i or worst_i+1
                i = rng.choice([max(0, worst_i - 1), worst_i, min(n - 1, worst_i + 1)])
                j = rng.randrange(n)
            if i == j:
                continue
            cand = best_seq[:]
            v = cand.pop(i)
            cand.insert(j, v)
            c = eval_seq_cost(cand)
            if c < best_cost:
                best_cost = c
                best_seq = cand

        return best_cost, best_seq

    # Multi-restart within time budget
    restarts = max(2, min(5, int(num_seqs)))  # a few robust restarts
    global_best_cost = float('inf')
    global_best_seq = None
    r = 0
    while r < restarts and time_left():
        c, s = run_beam_once()
        if time_left():
            c, s = local_improve(s, c)
        if c < global_best_cost:
            global_best_cost, global_best_seq = c, s
        r += 1

    # Fallback safety: ensure permutation validity
    if global_best_seq is None or len(global_best_seq) != N or len(set(global_best_seq)) != N:
        seq = list(range(N))
        rng.shuffle(seq)
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