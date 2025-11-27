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
    Beam search with memoized prefix/extension costs, A*-style pruning via
    singleton lower bounds, buddy-guided candidate pools, prefix-dominance
    pruning with suffix context, shallow lookahead, greedy completion to
    tighten an incumbent, and a lightweight local refinement pass.

    Args:
        workload: Workload object containing transaction data
        num_seqs: Number of restarts (upper bound, also time-bounded)

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    N = workload.num_txns
    start_time = time.time()
    # Slightly increased budget to leverage stronger pruning
    time_budget_sec = 0.50

    def time_left():
        return (time.time() - start_time) < time_budget_sec

    # Caches for partial and extension costs
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

    # Precompute singleton costs to guide lower bounds and seeding
    singleton_cost = {}
    for t in range(N):
        if not time_left():
            break
        singleton_cost[t] = eval_seq_cost([t])

    def max_singleton_rem(rem_set):
        if not rem_set:
            return 0
        m = 0
        for t in rem_set:
            c = singleton_cost.get(t)
            if c is None:
                c = eval_seq_cost([t])
                singleton_cost[t] = c
            if c > m:
                m = c
        return m

    # Build small buddy lists B[t] using true two-step extension deltas
    def build_buddies(max_buddies=8):
        all_txns = list(range(N))
        singles_sorted = sorted(all_txns, key=lambda x: singleton_cost.get(x, float('inf')))
        buddies = {t: [] for t in all_txns}
        for t in all_txns:
            if not time_left():
                break
            pool = []
            # mix of top-by-singleton and random for diversity
            top_slice = singles_sorted[:min(20, max(8, N // 6))]
            pool.extend([u for u in top_slice if u != t])
            others = [x for x in all_txns if x != t and x not in pool]
            if others:
                pool.extend(random.sample(others, min(max(10, N // 5), len(others))))
            # dedupe
            seen = set()
            pool_dedup = []
            for u in pool:
                if u not in seen:
                    seen.add(u)
                    pool_dedup.append(u)
            scored = []
            base = singleton_cost.get(t)
            pt = (t,)
            for u in pool_dedup:
                if not time_left():
                    break
                ec = eval_ext_cost(pt, u)
                scored.append((ec - base, u))
            scored.sort(key=lambda x: x[0])
            buddies[t] = [u for _d, u in scored[:max_buddies]]
        return buddies

    buddies = build_buddies(max_buddies=8)

    def greedy_finish(seq, rem_set, branch_factor, incumbent=None):
        seq_out = list(seq)
        rem = set(rem_set)
        cur_cost = eval_seq_cost(seq_out) if seq_out else 0
        while rem and time_left():
            if incumbent is not None:
                lb_here = max(cur_cost, max_singleton_rem(rem))
                if lb_here >= incumbent:
                    break
            rem_list = list(rem)
            # Prefer buddies of last if available
            last = seq_out[-1] if seq_out else None
            cand_pool = []
            if last is not None and last in buddies:
                cand_pool.extend([u for u in buddies[last] if u in rem])
            # fill with random remainder
            need = max(0, min(branch_factor, len(rem_list)) - len(cand_pool))
            if need > 0:
                others = [x for x in rem_list if x not in cand_pool]
                cand_pool.extend(random.sample(others, min(need, len(others))) if len(others) > need else others)
            if not cand_pool:
                cand_pool = rem_list if len(rem_list) <= branch_factor else random.sample(rem_list, branch_factor)
            best_t = None
            best_c = float('inf')
            prefix_tuple = tuple(seq_out)
            for t in cand_pool:
                c = eval_ext_cost(prefix_tuple, t)
                if c < best_c:
                    best_c = c
                    best_t = t
            if best_t is None:
                # Fallback if time exhausted
                for t in rem_list:
                    seq_out.append(t)
                return eval_seq_cost(seq_out), seq_out
            seq_out.append(best_t)
            rem.remove(best_t)
            cur_cost = best_c
        if rem:
            seq_out.extend(list(rem))
            cur_cost = eval_seq_cost(seq_out)
        return cur_cost, seq_out

    # Prefix-dominance: use global map to prune across depths/restarts
    best_state_global = {}

    def make_signature(rem_set, seq, k_suffix):
        if k_suffix <= 0:
            return (frozenset(rem_set), ())
        tail = tuple(seq[-k_suffix:]) if len(seq) >= k_suffix else tuple(seq)
        return (frozenset(rem_set), tail)

    def run_beam_once():
        all_txns = list(range(N))
        beam_width = min(18, max(8, N // 7))
        branch_factor = min(14, max(8, N // 10))
        lookahead_top = 3
        k_suffix_base = 3

        # Seed beam with best singletons
        seed_count = min(len(all_txns), max(beam_width * 2, 8))
        seeds = sorted(all_txns, key=lambda t: singleton_cost.get(t, float('inf')))[:seed_count]
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
            random.shuffle(seq)
            return eval_seq_cost(seq), seq

        beam.sort(key=lambda x: x[0])
        beam = beam[:min(beam_width, len(beam))]

        incumbent_cost = float('inf')
        incumbent_seq = None

        # Early incumbents: greedy finish best prefix and full singleton-ordered sequence
        if time_left():
            c_try, s_try = greedy_finish(beam[0][1], beam[0][2], branch_factor, incumbent=incumbent_cost)
            if len(s_try) == N and c_try < incumbent_cost:
                incumbent_cost, incumbent_seq = c_try, s_try
        if time_left():
            full_seed = sorted(all_txns, key=lambda t: singleton_cost.get(t, float('inf')))
            c_full = eval_seq_cost(full_seed)
            if c_full < incumbent_cost:
                incumbent_cost, incumbent_seq = c_full, full_seed

        steps = N - 1
        depth = 0
        best_state_local = {}

        for _ in range(steps):
            if not time_left():
                break
            depth += 1
            # Depth-adaptive suffix length
            if depth < int(0.35 * N):
                k_cur = max(3, k_suffix_base)
            elif depth < int(0.7 * N):
                k_cur = max(2, k_suffix_base - 1)
            else:
                k_cur = max(2, k_suffix_base - 2)

            new_beam = []

            for cost_so_far, seq, rem in beam:
                if not rem:
                    new_beam.append((cost_so_far, seq, rem, cost_so_far))
                    if cost_so_far < incumbent_cost:
                        incumbent_cost, incumbent_seq = cost_so_far, seq[:]
                    continue

                # Prune by incumbent and singleton LB
                if cost_so_far >= incumbent_cost:
                    continue
                lb_prefix = max(cost_so_far, max_singleton_rem(rem))
                if lb_prefix >= incumbent_cost:
                    continue

                # Prefix-dominance pruning
                sig = make_signature(rem, seq, k_cur)
                prev_local = best_state_local.get(sig)
                if prev_local is not None and cost_so_far >= prev_local:
                    continue
                prev_global = best_state_global.get(sig)
                if prev_global is not None and cost_so_far >= prev_global:
                    continue
                # update dominance maps
                best_state_local[sig] = cost_so_far
                if prev_global is None or cost_so_far < prev_global:
                    best_state_global[sig] = cost_so_far

                rem_list = list(rem)
                # Candidate pool: buddies of last + random supplement
                last = seq[-1]
                cand_pool = []
                if last in buddies:
                    cand_pool.extend([u for u in buddies[last] if u in rem])
                need = max(0, branch_factor * 2 - len(cand_pool))
                if need > 0:
                    others = [x for x in rem_list if x not in cand_pool]
                    if others:
                        cand_pool.extend(random.sample(others, min(need, len(others))))
                if not cand_pool:
                    cand_pool = rem_list if len(rem_list) <= branch_factor * 2 else random.sample(rem_list, branch_factor * 2)

                scored = []
                prefix_tuple = tuple(seq)
                for cand in cand_pool:
                    if not time_left():
                        break
                    ec = eval_ext_cost(prefix_tuple, cand)
                    delta = ec - cost_so_far
                    scored.append((delta, ec, cand))
                if not scored:
                    continue
                scored.sort(key=lambda x: x[0])
                top = scored[:min(branch_factor, len(scored))]

                # Expand children with buddy-biased lookahead and greedy probing
                probe_k = min(2, len(top))
                idx_child = 0
                for _, ec, cand in top:
                    if not time_left():
                        break
                    new_seq = seq + [cand]
                    new_rem = rem.copy()
                    new_rem.remove(cand)

                    # Child LB prune
                    lb_child = max(ec, max_singleton_rem(new_rem))
                    if lb_child >= incumbent_cost:
                        continue

                    # Shallow lookahead: prefer buddies of cand
                    rank_score = lb_child
                    if new_rem and time_left():
                        la_pool = []
                        if cand in buddies:
                            la_pool = [v for v in buddies[cand] if v in new_rem]
                        if not la_pool:
                            la_pool = list(new_rem)
                        if len(la_pool) > lookahead_top:
                            la_pool = random.sample(la_pool, lookahead_top)
                        new_prefix_tuple = tuple(new_seq)
                        best_la = float('inf')
                        for nxt in la_pool:
                            c2 = eval_ext_cost(new_prefix_tuple, nxt)
                            if c2 < best_la:
                                best_la = c2
                        rank_score = min(rank_score, best_la)

                    # Greedy probe on a few best children to tighten incumbent
                    if idx_child < probe_k and time_left():
                        g_cost, g_seq = greedy_finish(new_seq, new_rem, branch_factor, incumbent=incumbent_cost)
                        if len(g_seq) == N and g_cost < incumbent_cost:
                            incumbent_cost, incumbent_seq = g_cost, g_seq
                        rank_score = min(rank_score, g_cost)
                    idx_child += 1

                    new_beam.append((ec, new_seq, new_rem, rank_score))

            if not new_beam:
                break

            # Keep unique prefixes by best rank score
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

            # Tighten incumbent with greedy finish of top-2 prefixes
            if beam and time_left():
                for cost_so_far, seq, rem in beam[:min(2, len(beam))]:
                    c_try, s_try = greedy_finish(seq, rem, branch_factor, incumbent=incumbent_cost)
                    if len(s_try) == N and c_try < incumbent_cost:
                        incumbent_cost, incumbent_seq = c_try, s_try

        # Finalize: greedily finish remaining prefixes and pick best
        best_cost = incumbent_cost
        best_seq_local = incumbent_seq
        for cost_so_far, seq, rem in beam:
            if not time_left():
                break
            c_fin, s_fin = greedy_finish(seq, rem, branch_factor, incumbent=incumbent_cost)
            if len(s_fin) == N and c_fin < best_cost:
                best_cost, best_seq_local = c_fin, s_fin

        if best_seq_local is None:
            seq = all_txns[:]
            random.shuffle(seq)
            best_seq_local = seq
            best_cost = eval_seq_cost(seq)
        return best_cost, best_seq_local

    def local_improve(seq, current_cost):
        best_seq = seq[:]
        best_cost = current_cost

        # One pass of adjacent swaps
        for i in range(len(best_seq) - 1):
            if not time_left():
                break
            cand = best_seq[:]
            cand[i], cand[i + 1] = cand[i + 1], cand[i]
            c = eval_seq_cost(cand)
            if c < best_cost:
                best_cost = c
                best_seq = cand

        # Limited random insertions
        trials = 30
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

    # Multi-restart within time budget
    restarts = max(1, min(3, int(num_seqs)))
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

    # Safety: ensure a valid permutation
    if global_best_seq is None or len(global_best_seq) != N or len(set(global_best_seq)) != N:
        # Repair or fallback random
        if global_best_seq is None:
            seq = list(range(N))
            random.shuffle(seq)
            global_best_seq = seq
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