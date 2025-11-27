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
    Buddy-guided beam search with memoized prefix/extension costs,
    A*-style singleton lower bounds, seeded pairs, shallow lookahead,
    prefix-dominance pruning, greedy completion to tighten an incumbent,
    and lightweight local refinement (adjacent swaps + boundary repair + insertions).

    Args:
        workload: Workload object containing transaction data
        num_seqs: Number of restarts (upper bound, also time-bounded)

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    N = workload.num_txns
    start_time = time.time()
    # Balanced budget: enough exploration without hurting combined score
    time_budget_sec = 0.48

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

    # Pairwise marginal delta cache
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

    # Build small buddy list per txn using sampled candidates
    def build_buddies(max_buddies=6, sample_per_t=16):
        all_txns = list(range(N))
        # Sort by singleton to bias toward "cheaper" partners
        singles_sorted = sorted(all_txns, key=lambda x: singleton_cost.get(x, float('inf')))
        buddies = {t: [] for t in all_txns}
        for t in all_txns:
            if not time_left():
                break
            # Candidate pool: top by singleton plus random others
            pool = []
            top_slice = [u for u in singles_sorted[:min(18, max(6, N // 7))] if u != t]
            pool.extend(top_slice)
            others = [u for u in all_txns if u != t and u not in top_slice]
            if others:
                pool.extend(random.sample(others, min(sample_per_t, len(others))))
            # Dedup
            seen = set()
            cand_list = []
            for u in pool:
                if u in seen or u == t:
                    continue
                seen.add(u)
                cand_list.append(u)
            scored = []
            for u in cand_list:
                if not time_left():
                    break
                scored.append((pair_delta(t, u), u))
            scored.sort(key=lambda x: x[0])
            buddies[t] = [u for _d, u in scored[:max_buddies]]
        return buddies

    buddies = build_buddies(max_buddies=6, sample_per_t=16)

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

    # Prefix-dominance state: best cost seen for (frozenset(rem), last)
    prefix_best = {}

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
            # prioritize buddies of last
            last = seq_out[-1] if seq_out else None
            cand_pool = []
            if last is not None and last in buddies:
                for u in buddies[last]:
                    if u in rem:
                        cand_pool.append(u)
            # fill with randoms
            need = max(0, min(branch_factor, len(rem_list)) - len(cand_pool))
            if need > 0:
                others = [x for x in rem_list if x not in cand_pool]
                if len(others) > need:
                    cand_pool.extend(random.sample(others, need))
                else:
                    cand_pool.extend(others)
            if not cand_pool:
                cand_pool = rem_list if len(rem_list) <= branch_factor else random.sample(rem_list, branch_factor)

            prefix_tuple = tuple(seq_out)
            best_t = None
            best_c = float('inf')
            for t in cand_pool:
                c = eval_ext_cost(prefix_tuple, t)
                if c < best_c:
                    best_c = c
                    best_t = t
            if best_t is None:
                # Time exhausted; append arbitrarily
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

    def seed_initial_beam(beam_width):
        all_txns = list(range(N))
        seeds = []
        # Top singletons
        sorted_single = sorted(all_txns, key=lambda t: singleton_cost.get(t, float('inf')))
        for t in sorted_single[:max(beam_width, 6)]:
            if not time_left():
                break
            seq = [t]
            rem = set(all_txns)
            rem.remove(t)
            c = eval_seq_cost(seq)
            seeds.append((c, seq, rem))
        # Add some best pairs as length-2 seeds
        pair_quota = max(2, beam_width // 2)
        for t in sorted_single[:min(len(sorted_single), beam_width * 2)]:
            if not time_left():
                break
            if not buddies.get(t):
                continue
            u = buddies[t][0]
            seq = [t, u]
            rem = set(all_txns)
            if t in rem:
                rem.remove(t)
            if u in rem:
                rem.remove(u)
            c = eval_seq_cost(seq)
            seeds.append((c, seq, rem))
            if len(seeds) >= 3 * beam_width:
                break
        if not seeds:
            # Fallback
            seq = all_txns[:]
            random.shuffle(seq)
            return [(eval_seq_cost([seq[0]]), [seq[0]], set(all_txns) - {seq[0]})]
        seeds.sort(key=lambda x: x[0])
        # Keep only unique prefixes
        unique = []
        seen = set()
        for entry in seeds:
            key = tuple(entry[1])
            if key in seen:
                continue
            seen.add(key)
            unique.append(entry)
            if len(unique) >= beam_width:
                break
        return unique

    def run_beam_once():
        beam_width = min(22, max(8, N // 7))
        branch_factor = min(18, max(8, N // 9))
        lookahead_k = 3

        # Seed beam with best singletons and a few best pairs
        beam = seed_initial_beam(beam_width)
        if not beam:
            all_txns = list(range(N))
            seq = all_txns[:]
            random.shuffle(seq)
            return eval_seq_cost(seq), seq

        incumbent_cost = float('inf')
        incumbent_seq = None

        # Early incumbent by greedily finishing the best prefix
        if time_left():
            c_try, s_try = greedy_finish(beam[0][1], beam[0][2], branch_factor, incumbent=incumbent_cost)
            if len(s_try) == N and c_try < incumbent_cost:
                incumbent_cost, incumbent_seq = c_try, s_try

        steps = N - (len(beam[0][1]))
        depth = 0
        while depth < steps and time_left():
            depth += 1
            new_beam = []

            # Adjust dominance key granularity with depth: here last element only
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

                # Prefix-dominance by remaining set and last
                sig = (frozenset(rem), seq[-1])
                prev = prefix_best.get(sig)
                if prev is not None and cost_so_far >= prev:
                    continue
                # Update best seen for this signature
                if prev is None or cost_so_far < prev:
                    prefix_best[sig] = cost_so_far

                rem_list = list(rem)
                # Candidate pool guided by buddies of last + random supplements
                last = seq[-1]
                cand_pool = []
                if last in buddies:
                    for u in buddies[last]:
                        if u in rem:
                            cand_pool.append(u)
                # Supplement with randoms up to 2*branch
                need = max(0, branch_factor * 2 - len(cand_pool))
                if need > 0:
                    others = [x for x in rem_list if x not in cand_pool]
                    if others:
                        cand_pool.extend(random.sample(others, min(need, len(others))))
                if not cand_pool:
                    cand_pool = rem_list if len(rem_list) <= branch_factor * 2 else random.sample(rem_list, branch_factor * 2)

                prefix_tuple = tuple(seq)
                scored = []
                for cand in cand_pool:
                    if not time_left():
                        break
                    ec = eval_ext_cost(prefix_tuple, cand)
                    if ec >= incumbent_cost:
                        continue
                    # Shallow lookahead biased by buddies of cand
                    la_score = ec
                    if time_left():
                        new_rem = rem.copy()
                        if cand in new_rem:
                            new_rem.remove(cand)
                        la_pool = []
                        if cand in buddies:
                            for v in buddies[cand]:
                                if v in new_rem:
                                    la_pool.append(v)
                        if not la_pool:
                            la_pool = list(new_rem)
                        if la_pool:
                            la_sample = la_pool if len(la_pool) <= lookahead_k else random.sample(la_pool, lookahead_k)
                            new_prefix_tuple = tuple(seq + [cand])
                            best_la = float('inf')
                            for nxt in la_sample:
                                c2 = eval_ext_cost(new_prefix_tuple, nxt)
                                if c2 < best_la:
                                    best_la = c2
                            la_score = min(la_score, best_la)
                    scored.append((ec - cost_so_far, la_score, ec, cand))

                if not scored:
                    continue
                # rank by marginal delta, then lookahead score
                scored.sort(key=lambda x: (x[0], x[1]))
                top = scored[:min(branch_factor, len(scored))]

                for _delta, la_s, ec, cand in top:
                    new_seq = seq + [cand]
                    new_rem = rem.copy()
                    new_rem.remove(cand)
                    # Child LB prune
                    lb_child = max(ec, max_singleton_rem(new_rem))
                    if lb_child >= incumbent_cost:
                        continue
                    new_beam.append((ec, new_seq, new_rem, min(la_s, lb_child)))

            if not new_beam:
                break

            # Keep unique prefixes by best lookahead score
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

            # Tighten incumbent periodically
            if beam and time_left():
                best_prefix = min(beam, key=lambda x: x[0])
                c_try, s_try = greedy_finish(best_prefix[1], best_prefix[2], branch_factor, incumbent=incumbent_cost)
                if len(s_try) == N and c_try < incumbent_cost:
                    incumbent_cost, incumbent_seq = c_try, s_try

        # Finalize: greedily finish remaining prefixes
        best_cost = incumbent_cost
        best_seq_local = incumbent_seq
        for cost_so_far, seq, rem in beam:
            if not time_left():
                break
            c_fin, s_fin = greedy_finish(seq, rem, branch_factor, incumbent=incumbent_cost)
            if len(s_fin) == N and c_fin < best_cost:
                best_cost, best_seq_local = c_fin, s_fin

        if best_seq_local is None:
            all_txns = list(range(N))
            seq = all_txns[:]
            random.shuffle(seq)
            best_seq_local = seq
            best_cost = eval_seq_cost(seq)
        return best_cost, best_seq_local

    def local_improve(seq, current_cost):
        best_seq = seq[:]
        best_cost = current_cost

        # Two passes of adjacent swaps (hill climbing)
        for _ in range(2):
            improved = False
            for i in range(len(best_seq) - 1):
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

        # Single boundary repair around worst adjacency by pair delta
        if len(best_seq) >= 6 and time_left():
            n = len(best_seq)
            worst_idx = -1
            worst_val = -float('inf')
            for i in range(n - 1):
                if not time_left():
                    break
                v = pair_delta(best_seq[i], best_seq[i + 1])
                if v > worst_val:
                    worst_val = v
                    worst_idx = i
            if worst_idx >= 0 and time_left():
                # Try relocating the second element of the worst pair near best position
                j_elem = best_seq[worst_idx + 1]
                base = best_seq[:worst_idx + 1] + best_seq[worst_idx + 2:]
                best_local = best_cost
                best_pos = None
                # Evaluate a small window around original position plus a few random slots
                positions = list(range(max(0, worst_idx - 3), min(len(base) + 1, worst_idx + 5)))
                extra = set()
                limit = min(6, len(base) + 1)
                while len(extra) < limit and time_left():
                    extra.add(random.randrange(len(base) + 1))
                positions.extend([p for p in extra if p not in positions])
                for p in positions:
                    if not time_left():
                        break
                    cand = base[:]
                    cand.insert(p, j_elem)
                    c = eval_seq_cost(cand)
                    if c < best_local:
                        best_local = c
                        best_pos = p
                if best_pos is not None:
                    base.insert(best_pos, j_elem)
                    best_seq = base
                    best_cost = best_local

        # Limited random insertion moves
        trials = 25
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