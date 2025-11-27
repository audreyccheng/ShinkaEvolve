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
    Conflict-aware beam search with shared prefix-dominance, shallow lookahead,
    and child-level greedy promotion. A final lightweight adjacent-swap pass
    locks in quick local improvements.

    Args:
        workload: Workload object containing transaction data
        num_seqs: Number of restarts/portfolio variants (time-bounded)

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    N = workload.num_txns
    rng = random.Random(1729 + 17 * N)

    # Time budget per workload; tuned to combined score runtime
    time_budget_sec = 0.62 if N >= 90 else 0.55
    start_time = time.time()

    def time_left():
        return (time.time() - start_time) < time_budget_sec

    all_txns = list(range(N))

    # Caches
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

    # Precompute singleton costs and ordering
    singleton_cost = {}
    for t in all_txns:
        if not time_left():
            break
        singleton_cost[t] = eval_seq_cost([t])
    singles_sorted = sorted(all_txns, key=lambda t: singleton_cost.get(t, float('inf')))

    # Pairwise marginal deltas for small buddy lists: d(a->b) = cost([a,b]) - cost([a])
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

    def build_buddies(max_buddies=8, sample_per_t=16):
        buds = {t: [] for t in all_txns}
        top_slice = singles_sorted[:min(22, max(8, N // 5))]
        for t in all_txns:
            if not time_left():
                break
            pool = [u for u in top_slice if u != t]
            others = [u for u in all_txns if u != t and u not in pool]
            if others:
                pool.extend(rng.sample(others, min(sample_per_t, len(others))))
            seen = set()
            cand_list = []
            for u in pool:
                if u not in seen and u != t:
                    seen.add(u)
                    cand_list.append(u)
            scored = []
            base = singleton_cost.get(t)
            pt = (t,)
            for u in cand_list:
                if not time_left():
                    break
                c2 = eval_ext_cost(pt, u)
                scored.append((c2 - base, u))
            scored.sort(key=lambda x: x[0])
            buds[t] = [u for _d, u in scored[:max_buddies]]
        return buds

    buddies = build_buddies(max_buddies=8, sample_per_t=14)

    # Lower bound: max remaining singleton vs current prefix cost
    def lb_singleton(cur_cost, rem_set):
        if not rem_set:
            return cur_cost
        m = 0.0
        for t in rem_set:
            c = singleton_cost.get(t)
            if c is None:
                c = eval_seq_cost([t])
                singleton_cost[t] = c
            if c > m:
                m = c
        return max(cur_cost, m)

    # Shared prefix-dominance across beam and greedy
    # Key: (frozenset(remaining), suffix)
    prefix_dom = {}

    def dom_key(rem_set, seq, depth):
        # suffix size: ≤3 early, ≤4 after 70% depth
        if depth < int(0.7 * N):
            k = 3
        else:
            k = 4
        tail = tuple(seq[-k:]) if len(seq) >= k else tuple(seq)
        return (frozenset(rem_set), tail)

    # Greedy completion with buddy guidance and early termination
    def greedy_complete(prefix, rem_set, incumbent=float('inf'), branch_k=12, depth0=0):
        seq = list(prefix)
        rem = set(rem_set)
        cur_cost = eval_seq_cost(seq) if seq else 0.0
        steps = 0
        while rem and time_left():
            steps += 1
            # lower bound prune
            if lb_singleton(cur_cost, rem) >= incumbent:
                break
            # prefix dominance prune
            sig = dom_key(rem, seq, depth0 + steps)
            prev = prefix_dom.get(sig)
            if prev is not None and cur_cost >= prev:
                break
            if prev is None or cur_cost < prev:
                prefix_dom[sig] = cur_cost

            rem_list = list(rem)
            pool = []
            last = seq[-1] if seq else None
            if last is not None:
                for u in buddies.get(last, []):
                    if u in rem and u not in pool:
                        pool.append(u)
            # add low-singleton
            low_single = sorted(rem_list, key=lambda t: singleton_cost.get(t, float('inf')))[:min(5, len(rem_list))]
            for u in low_single:
                if u not in pool:
                    pool.append(u)
            # random fill
            need = max(0, branch_k - len(pool))
            if need > 0:
                others = [x for x in rem_list if x not in pool]
                if others:
                    pool.extend(rng.sample(others, min(need, len(others))))
            if not pool:
                pool = rem_list

            pt = tuple(seq)
            best_c = float('inf')
            best_t = None
            for t in pool:
                c = eval_ext_cost(pt, t)
                if c < best_c:
                    best_c = c
                    best_t = t
            if best_t is None:
                # just append remaining if no time or pool
                seq.extend(rem)
                cur_cost = eval_seq_cost(seq)
                return cur_cost, seq
            seq.append(best_t)
            rem.remove(best_t)
            cur_cost = best_c

            # every ~9 steps, re-check dominance to tighten bounds
            if steps % 9 == 0:
                sig2 = dom_key(rem, seq, depth0 + steps)
                prev2 = prefix_dom.get(sig2)
                if prev2 is None or cur_cost < prev2:
                    prefix_dom[sig2] = cur_cost

        if rem:
            seq.extend(list(rem))
            cur_cost = eval_seq_cost(seq)
        return cur_cost, seq

    # Beam core with child-level greedy promotion and global dominance
    def run_beam(beam_width, branch_factor, lookahead_top, top_k_probes):
        # Seed beam with best singletons
        seeds = singles_sorted[:max(beam_width * 2, 8)]
        beam = []
        for t in seeds:
            if not time_left():
                break
            seq = [t]
            rem = set(all_txns)
            rem.remove(t)
            c = eval_seq_cost(seq)
            beam.append((c, seq, rem))
        if not beam:
            seq = all_txns[:]
            rng.shuffle(seq)
            return eval_seq_cost(seq), seq

        beam.sort(key=lambda x: x[0])
        beam = beam[:beam_width]

        incumbent_cost = float('inf')
        incumbent_seq = None

        # Early incumbent via greedy completion of best prefix
        if time_left():
            c_try, s_try = greedy_complete(beam[0][1], beam[0][2], incumbent=incumbent_cost, branch_k=max(8, branch_factor))
            if c_try < incumbent_cost:
                incumbent_cost, incumbent_seq = c_try, s_try

        steps_total = N - 1
        depth = 1
        while depth <= steps_total and time_left():
            new_entries = []
            # Occasionally tighten incumbent from top-2 prefixes
            for (cost_so_far, seq, rem) in beam[:min(2, len(beam))]:
                if not time_left():
                    break
                c_try, s_try = greedy_complete(seq, rem, incumbent=incumbent_cost, branch_k=max(8, branch_factor), depth0=depth)
                if c_try < incumbent_cost:
                    incumbent_cost, incumbent_seq = c_try, s_try

            for cost_so_far, seq, rem in beam:
                if not rem:
                    if cost_so_far < incumbent_cost:
                        incumbent_cost, incumbent_seq = cost_so_far, seq[:]
                    continue
                # incumbent + LB prune
                if cost_so_far >= incumbent_cost:
                    continue
                if lb_singleton(cost_so_far, rem) >= incumbent_cost:
                    continue

                # prefix dominance on parent
                sig_p = dom_key(rem, seq, depth)
                prev_p = prefix_dom.get(sig_p)
                if prev_p is not None and cost_so_far >= prev_p:
                    continue
                if prev_p is None or cost_so_far < prev_p:
                    prefix_dom[sig_p] = cost_so_far

                rem_list = list(rem)

                # Candidate pool: buddies of last + low-singleton + random fill up to 2*branch
                cand_pool = []
                last = seq[-1] if seq else None
                if last is not None:
                    for u in buddies.get(last, []):
                        if u in rem:
                            cand_pool.append(u)
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

                # Score by immediate ext cost and shallow lookahead
                pt = tuple(seq)
                tmp = []
                for cand in cand_pool:
                    if not time_left():
                        break
                    ec = eval_ext_cost(pt, cand)
                    tmp.append((ec, cand))
                if not tmp:
                    continue
                # shallow lookahead
                scored = []
                for ec, cand in tmp:
                    if ec >= incumbent_cost:
                        continue
                    new_rem = rem.copy()
                    new_rem.remove(cand)
                    new_pt = tuple(seq + [cand])
                    la_best = ec
                    # pick lookahead pool from buddies of cand or small random
                    la_pool = [v for v in buddies.get(cand, []) if v in new_rem]
                    if not la_pool:
                        la_pool = list(new_rem)
                    if len(la_pool) > lookahead_top:
                        la_pool = rng.sample(la_pool, lookahead_top)
                    for nxt in la_pool:
                        c2 = eval_ext_cost(new_pt, nxt)
                        if c2 < la_best:
                            la_best = c2
                    scored.append((ec, la_best, cand))
                if not scored:
                    continue
                scored.sort(key=lambda x: (x[0], x[1]))
                top = scored[:min(branch_factor, len(scored))]

                # Child expansion; greedy-complete top_k to promote incumbent and prune
                k_probe = 2 if depth < int(0.7 * N) else 1
                k_probe = min(k_probe, top_k_probes)
                for idx, (ec, la, cand) in enumerate(top):
                    new_seq = seq + [cand]
                    new_rem = rem.copy()
                    new_rem.remove(cand)

                    # Child LB prune
                    if lb_singleton(ec, new_rem) >= incumbent_cost:
                        continue

                    # Probe top few children
                    adj_score = la
                    if idx < k_probe and time_left():
                        g_cost, g_seq = greedy_complete(new_seq, new_rem, incumbent=incumbent_cost, branch_k=max(6, branch_factor // 2), depth0=depth + 1)
                        if g_cost < incumbent_cost:
                            incumbent_cost, incumbent_seq = g_cost, g_seq
                        adj_score = min(adj_score, g_cost)
                        # prune child if its greedy completion is not competitive
                        if g_cost >= incumbent_cost:
                            continue

                    # Child prefix dominance update
                    sig_c = dom_key(new_rem, new_seq, depth + 1)
                    prev_c = prefix_dom.get(sig_c)
                    if prev_c is not None and ec >= prev_c:
                        continue
                    if prev_c is None or ec < prev_c:
                        prefix_dom[sig_c] = ec

                    new_entries.append((ec, new_seq, new_rem, adj_score))

            if not new_entries:
                break

            # Select next beam: prioritize adjusted score, keep uniqueness
            new_entries.sort(key=lambda x: (x[3], x[0]))
            unique = []
            seen = set()
            for ec, s, r, adj in new_entries:
                key = tuple(s)
                if key in seen:
                    continue
                seen.add(key)
                unique.append((ec, s, r))
                if len(unique) >= beam_width:
                    break
            beam = unique
            depth += 1

        # Final greedy completion of frontier
        best_cost = incumbent_cost
        best_seq = incumbent_seq
        for ec, s, r in beam:
            if not time_left():
                break
            c_fin, s_fin = greedy_complete(s, r, incumbent=best_cost, branch_k=max(8, branch_factor), depth0=depth)
            if c_fin < best_cost:
                best_cost, best_seq = c_fin, s_fin

        if best_seq is None:
            seq = all_txns[:]
            rng.shuffle(seq)
            best_seq = seq
            best_cost = eval_seq_cost(seq)
        return best_cost, best_seq

    # Lightweight local improvement: single adjacent-swap pass
    def local_adjacent_pass(seq, cur_cost):
        best_seq = seq[:]
        best_cost = cur_cost
        n = len(best_seq)
        for i in range(n - 1):
            if not time_left():
                break
            cand = best_seq[:]
            cand[i], cand[i + 1] = cand[i + 1], cand[i]
            c = eval_seq_cost(cand)
            if c < best_cost:
                best_cost = c
                best_seq = cand
        return best_cost, best_seq

    # Portfolio: deterministic parameter variations
    portfolios = [
        # early broader beam, moderate branch
        {'beam': min(24, max(10, N // 6)), 'branch': min(16, max(10, N // 8)), 'lookahead_top': 4, 'top_k_probes': 2},
        # narrower beam, deeper branch
        {'beam': min(18, max(8, N // 7)), 'branch': min(18, max(12, N // 7)), 'lookahead_top': 3, 'top_k_probes': 2},
        # tight beam, tight branch, faster
        {'beam': min(14, max(8, N // 8)), 'branch': min(12, max(8, N // 9)), 'lookahead_top': 3, 'top_k_probes': 1},
    ]

    global_best_cost = float('inf')
    global_best_seq = None

    max_restarts = max(2, min(len(portfolios), int(num_seqs)))
    for r in range(max_restarts):
        if not time_left():
            break
        params = portfolios[r % len(portfolios)]
        c, s = run_beam(params['beam'], params['branch'], params['lookahead_top'], params['top_k_probes'])
        if time_left():
            c, s = local_adjacent_pass(s, c)
        if c < global_best_cost:
            global_best_cost, global_best_seq = c, s

    # Safety: ensure permutation validity
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