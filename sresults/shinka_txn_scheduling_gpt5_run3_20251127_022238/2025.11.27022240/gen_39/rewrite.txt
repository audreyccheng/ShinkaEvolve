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
    Dual-phase incumbent-seeded beam with prefix-dominance pruning,
    buddy-biased candidate generation, depth-adaptive lookahead,
    greedy completions, and a light VNS polish.
    """
    N = workload.num_txns
    start_time = time.time()
    # Balanced runtime budget; adapt mildly with N
    time_budget_sec = 0.42 + min(0.10, 0.0008 * max(0, N))

    rng = random.Random(1729 + N + int(num_seqs))

    def time_left():
        return (time.time() - start_time) < time_budget_sec

    # Shared caches across phases
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

    # Precompute singleton costs for seeding and lower bounds
    singleton_cost = {}
    for t in all_txns:
        if not time_left():
            break
        singleton_cost[t] = eval_seq_cost([t])

    # Build "buddy" lists using sampled pairwise two-step deltas
    def build_buddies(max_buddies=8):
        order_singleton = sorted(all_txns, key=lambda x: singleton_cost.get(x, float('inf')))
        buddies = {t: [] for t in all_txns}
        # Candidate pool size bounds
        pool_top = min(len(order_singleton), max(10, N // 5))
        pool_rand = min(max(10, N // 5), N)
        probe_budget = min(1200, N * 10)
        probes_used = 0
        for t in all_txns:
            if not time_left():
                break
            # Combine top by singleton and random
            pool = order_singleton[:pool_top]
            add = [x for x in all_txns if x != t]
            if add:
                pool.extend(rng.sample(add, min(pool_rand, len(add))))
            # Deduplicate and remove self
            seen = set([t])
            cand_pool = []
            for u in pool:
                if u not in seen:
                    seen.add(u)
                    cand_pool.append(u)
            # Score by true delta over singleton
            scored = []
            base = singleton_cost.get(t)
            pref = (t,)
            for u in cand_pool:
                if probes_used >= probe_budget or not time_left():
                    break
                c2 = eval_ext_cost(pref, u)
                scored.append((c2 - base, u))
                probes_used += 1
            scored.sort(key=lambda x: x[0])
            buddies[t] = [u for _d, u in scored[:max_buddies]]
        return buddies

    buddies = build_buddies(max_buddies=min(8, max(6, N // 18)))

    # Lower bound: max of current prefix cost and max singleton of remaining txns
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

    # Prefix-dominance map: (frozenset(rem), suffix_k tuple) -> min prefix cost
    best_state_global = {}

    def make_signature(rem_set, seq, k_suffix):
        if k_suffix <= 0:
            return (frozenset(rem_set), ())
        tail = tuple(seq[-k_suffix:]) if len(seq) >= k_suffix else tuple(seq)
        return (frozenset(rem_set), tail)

    # Greedy completion from a prefix using buddy-biased candidate pools
    def greedy_finish(seq, rem_set, branch_k=10, incumbent=None):
        seq_out = list(seq)
        rem = set(rem_set)
        cur_cost = eval_seq_cost(seq_out) if seq_out else 0
        while rem and time_left():
            if incumbent is not None and lb_singleton(cur_cost, rem) >= incumbent:
                break
            last = seq_out[-1] if seq_out else None
            rem_list = list(rem)
            cand_pool = []

            # Try buddies of last txn first
            if last is not None:
                for u in buddies.get(last, []):
                    if u in rem:
                        cand_pool.append(u)

            # Fill with random remainder
            need = max(0, branch_k - len(cand_pool))
            if need > 0:
                others = [x for x in rem_list if x not in cand_pool]
                if others:
                    cand_pool.extend(rng.sample(others, min(need, len(others))))

            if not cand_pool:
                cand_pool = rem_list if len(rem_list) <= branch_k else rng.sample(rem_list, branch_k)

            prefix_tuple = tuple(seq_out)
            best_t = None
            best_c = float('inf')
            for t in cand_pool:
                c = eval_ext_cost(prefix_tuple, t)
                if c < best_c:
                    best_c = c
                    best_t = t
            if best_t is None:
                # Append arbitrary remaining and evaluate once
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

    # Core beam runner
    def run_beam(beam_width, branch_factor, lookahead_top, k_suffix, incumbent_cost=float('inf')):
        # Seeds: best singletons by cost
        seeds = sorted(all_txns, key=lambda t: singleton_cost.get(t, float('inf')))
        seeds = seeds[:max(beam_width * 2, 8)]

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

        best_full_cost = incumbent_cost
        best_full_seq = None
        best_state_local = {}

        steps = N - 1
        depth = 0
        while depth < steps and time_left():
            depth += 1
            new_beam = []

            # Depth-adaptive suffix length to balance precision and merging
            if depth < int(0.4 * N):
                k_cur = k_suffix
            elif depth < int(0.75 * N):
                k_cur = max(2, k_suffix - 1)
            else:
                k_cur = max(2, k_suffix - 2)

            for cost_so_far, seq, rem in beam:
                if not rem:
                    if cost_so_far < best_full_cost:
                        best_full_cost, best_full_seq = cost_so_far, seq[:]
                    continue

                # Prune by incumbent
                if cost_so_far >= best_full_cost:
                    continue

                # Dominance prune (local and global)
                sig = make_signature(rem, seq, k_cur)
                prev_local = best_state_local.get(sig)
                if prev_local is not None and cost_so_far >= prev_local:
                    continue
                prev_global = best_state_global.get(sig)
                if prev_global is not None and cost_so_far >= prev_global:
                    continue
                best_state_local[sig] = min(cost_so_far, prev_local) if prev_local is not None else cost_so_far
                old = best_state_global.get(sig)
                if old is None or cost_so_far < old:
                    best_state_global[sig] = cost_so_far

                last = seq[-1]
                rem_list = list(rem)

                # Candidate pool: buddies of last + random
                cand_pool = []
                for u in buddies.get(last, []):
                    if u in rem:
                        cand_pool.append(u)
                need = max(0, branch_factor * 2 - len(cand_pool))
                if need > 0:
                    others = [x for x in rem_list if x not in cand_pool]
                    if others:
                        cand_pool.extend(rng.sample(others, min(need, len(others))))
                if not cand_pool:
                    cand_pool = rem_list if len(rem_list) <= branch_factor * 2 else rng.sample(rem_list, branch_factor * 2)

                # Score candidates by extension and shallow lookahead
                prefix_tuple = tuple(seq)
                scored = []
                for cand in cand_pool:
                    if not time_left():
                        break
                    ec = eval_ext_cost(prefix_tuple, cand)
                    # Phase-B compatible pruning by incumbent
                    if ec >= best_full_cost:
                        continue
                    la_best = ec
                    # Lookahead: try a few buddies/others after cand
                    new_rem_len = len(rem) - 1
                    if new_rem_len > 0 and lookahead_top > 0:
                        la_pool = []
                        for v in buddies.get(cand, []):
                            if v in rem and v != cand:
                                la_pool.append(v)
                        if not la_pool:
                            la_pool = [x for x in rem if x != cand]
                        if len(la_pool) > lookahead_top:
                            la_pool = rng.sample(la_pool, lookahead_top)
                        new_prefix_tuple = tuple(seq + [cand])
                        for nxt in la_pool:
                            c2 = eval_ext_cost(new_prefix_tuple, nxt)
                            if c2 < la_best:
                                la_best = c2
                    scored.append((ec - cost_so_far, la_best, ec, cand))

                if not scored:
                    continue
                scored.sort(key=lambda x: (x[0], x[1]))
                top = scored[:min(branch_factor, len(scored))]

                for _delta, la_score, ec, cand in top:
                    new_seq = seq + [cand]
                    new_rem = rem.copy()
                    new_rem.remove(cand)
                    # Lower bound prune
                    if lb_singleton(ec, new_rem) >= best_full_cost:
                        continue
                    new_beam.append((ec, new_seq, new_rem, la_score))

            if not new_beam:
                break

            # Rank next beam by lookahead score, keep unique prefixes
            new_beam.sort(key=lambda x: x[3])
            unique = []
            seen = set()
            for entry in new_beam:
                keyp = tuple(entry[1])
                if keyp in seen:
                    continue
                seen.add(keyp)
                unique.append((entry[0], entry[1], entry[2]))
                if len(unique) >= beam_width:
                    break
            beam = unique

            # Greedy complete top-2 prefixes to tighten incumbent
            if beam and time_left():
                upto = min(2, len(beam))
                for i in range(upto):
                    c_pref, s_pref, r_pref = beam[i]
                    c_try, s_try = greedy_finish(s_pref, r_pref, branch_k=max(8, N // 10), incumbent=best_full_cost)
                    if len(s_try) == N and c_try < best_full_cost:
                        best_full_cost, best_full_seq = c_try, s_try

        # Finalize: greedily finish remaining
        for c_pref, s_pref, r_pref in beam:
            if not time_left():
                break
            c_fin, s_fin = greedy_finish(s_pref, r_pref, branch_k=max(8, N // 10), incumbent=best_full_cost)
            if len(s_fin) == N and c_fin < best_full_cost:
                best_full_cost, best_full_seq = c_fin, s_fin

        if best_full_seq is None:
            seq = all_txns[:]
            rng.shuffle(seq)
            return eval_seq_cost(seq), seq
        return best_full_cost, best_full_seq

    # Portfolio parameters for two-phase beam
    params_A = {'beam': 16, 'branch': 12, 'lookahead_top': 4, 'k_suffix': 3}
    params_B = {'beam': 10, 'branch': 8,  'lookahead_top': 3, 'k_suffix': 3}

    global_best_cost = float('inf')
    global_best_seq = None

    # Multi-restart portfolio with shared caches; keep deterministic but diverse
    restarts = max(2, min(3, int(num_seqs)))
    for r in range(restarts):
        if not time_left():
            break
        # Phase A
        cA, sA = run_beam(params_A['beam'], params_A['branch'], params_A['lookahead_top'], params_A['k_suffix'], incumbent_cost=global_best_cost)
        incumbent = min(global_best_cost, cA)
        best_seq_phase = sA
        best_cost_phase = cA

        # Phase B with incumbent pruning
        if time_left():
            cB, sB = run_beam(params_B['beam'], params_B['branch'], params_B['lookahead_top'], params_B['k_suffix'], incumbent_cost=incumbent)
            if cB < best_cost_phase:
                best_cost_phase, best_seq_phase = cB, sB

        # Local VNS polish
        def local_improve(seq, cur_cost):
            best_seq = list(seq)
            best_cost = cur_cost

            # One adjacent swap pass
            for i in range(len(best_seq) - 1):
                if not time_left():
                    break
                cand = best_seq[:]
                cand[i], cand[i + 1] = cand[i + 1], cand[i]
                c = eval_seq_cost(cand)
                if c < best_cost:
                    best_cost = c
                    best_seq = cand

            # Targeted relocations (light)
            trials = 40
            n = len(best_seq)
            while trials > 0 and time_left():
                trials -= 1
                i = rng.randrange(n)
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

        if time_left():
            best_cost_phase, best_seq_phase = local_improve(best_seq_phase, best_cost_phase)

        if best_cost_phase < global_best_cost:
            global_best_cost, global_best_seq = best_cost_phase, best_seq_phase

        # Slightly perturb parameters next restart for diversity
        params_A = {
            'beam': max(10, params_A['beam'] - rng.choice([0, 2])),
            'branch': max(10, params_A['branch'] + rng.choice([-2, 0, 2])),
            'lookahead_top': max(3, params_A['lookahead_top'] + rng.choice([-1, 0, 1])),
            'k_suffix': params_A['k_suffix'],
        }
        params_B = {
            'beam': max(8, params_B['beam'] + rng.choice([-2, 0, 2])),
            'branch': max(6, params_B['branch'] + rng.choice([-1, 0, 1])),
            'lookahead_top': max(2, params_B['lookahead_top'] + rng.choice([-1, 0, 1])),
            'k_suffix': params_B['k_suffix'],
        }

    # Ensure a valid permutation and compute final cost
    if global_best_seq is None:
        global_best_seq = list(range(N))
        rng.shuffle(global_best_seq)
        global_best_cost = eval_seq_cost(global_best_seq)

    if len(global_best_seq) != N or len(set(global_best_seq)) != N:
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