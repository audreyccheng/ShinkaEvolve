# EVOLVE-BLOCK-START
"""Transaction scheduling algorithm for optimizing makespan across multiple workloads
Hybrid: regret-guided GRASP + regret-diverse beam + VND + LNS + elite path-relinking.
"""

import time
from functools import lru_cache
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
    Hybrid scheduler to minimize makespan:
    - GRASP (regret-guided best insertion)
    - Regret-diverse beam with shallow lookahead
    - VND local search (Or-opt + swaps + sampled pair swaps with DLB)
    - Ruin-and-recreate LNS
    - Elite pool with path relinking and ILS

    Args:
        workload: Workload object containing transaction data
        num_seqs: Search effort parameter

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    n = workload.num_txns
    all_txns = list(range(n))

    # ---------- Parameterization (adaptive) ----------
    small = n <= 50
    med = 50 < n <= 90
    large = n > 90

    # Construction
    rcl_alpha = 0.6
    cand_txn_sample = 10 if med or large else 12
    pos_sample_cap = 18 if large else (None if small else 22)
    EXHAUSTIVE_THRESHOLD = 20  # use exhaustive insertion positions when sequence is small
    RING_SPAN = 3  # +- window for focus-aware position sampling

    # Beam search
    beam_width = max(5, min(10, num_seqs + (2 if small else 0)))
    cand_per_state = min(26, max(10, n // (4 if large else 3)))
    lookahead_k = 6 if med or large else 8
    diversity_quota = max(1, int(0.25 * beam_width))  # fraction of beam from high-regret expansions

    # Seeds and elites
    seeds = max(3, min(6, num_seqs))
    elite_cap = max(3, min(5, 2 + num_seqs // 3))

    # Local search
    vnd_rounds = 2 if large else 3
    pair_swap_tries = max(60, n // 2)

    # LNS
    lns_rounds = 2 if med or large else 3
    lns_remove_frac = 0.08 if large else (0.1 if med else 0.12)
    lns_remove_min = 8 if med or large else 6
    lns_remove_max = 16 if med or large else 14

    # ILS iterations
    ils_iterations = max(2, min(4, num_seqs))

    # ---------- Cached cost evaluator ----------
    @lru_cache(maxsize=200_000)
    def _eval_cost_tuple(seq_tuple):
        return workload.get_opt_seq_cost(list(seq_tuple))

    def eval_cost(seq):
        return _eval_cost_tuple(tuple(seq))

    # ---------- Utility: sample positions for insertion ----------
    def sample_positions(seq_len):
        total = seq_len + 1
        if pos_sample_cap is None or total <= pos_sample_cap:
            return list(range(total))
        # Keep ends, sample interior
        interior = list(range(1, seq_len))
        k = max(2, min(pos_sample_cap - 2, len(interior)))
        chosen = set(random.sample(interior, k)) if interior else set()
        chosen.update({0, seq_len})
        return sorted(chosen)

    # ---------- Insertion helpers (deterministic sampling + memoization) ----------
    best_two_cache = {}

    def positions_signature(seq_len, exhaustive=False, focus_idx=None):
        total = seq_len + 1
        if exhaustive or pos_sample_cap is None or total <= pos_sample_cap:
            return ('all', seq_len)
        anchors = {0, seq_len, seq_len // 2, seq_len // 4, (3 * seq_len) // 4}
        if focus_idx is not None:
            for d in range(-RING_SPAN, RING_SPAN + 1):
                p = focus_idx + d
                if 0 <= p <= seq_len:
                    anchors.add(p)
        # deterministically fill evenly spaced positions up to cap
        need_cap = max(0, (pos_sample_cap - len(anchors)))
        if need_cap > 0:
            for i in range(1, need_cap + 1):
                pos = round(i * seq_len / (need_cap + 1))
                if 0 <= pos <= seq_len:
                    anchors.add(pos)
                if len(anchors) >= pos_sample_cap:
                    break
        return tuple(sorted(anchors))

    def positions_for_insertion(seq_len, exhaustive=False, focus_idx=None):
        sig = positions_signature(seq_len, exhaustive=exhaustive, focus_idx=focus_idx)
        if sig == ('all', seq_len):
            return list(range(seq_len + 1)), sig
        return list(sig), sig

    def best_two_insertions(seq, txn, focus_idx=None, exhaustive=None):
        if exhaustive is None:
            exhaustive = (len(seq) <= EXHAUSTIVE_THRESHOLD)
        positions, sig = positions_for_insertion(len(seq), exhaustive=exhaustive, focus_idx=focus_idx)
        all_sig = ('all', len(seq))
        all_key = (tuple(seq), txn, all_sig)
        if all_key in best_two_cache:
            return best_two_cache[all_key]
        key = (tuple(seq), txn, sig)
        cached = best_two_cache.get(key)
        if cached is not None:
            return cached
        best = (float('inf'), None)
        second = (float('inf'), None)
        for pos in positions:
            cand = seq[:]
            cand.insert(pos, txn)
            c = eval_cost(cand)
            if c < best[0]:
                second = best
                best = (c, pos)
            elif c < second[0]:
                second = (c, pos)
        if second[0] == float('inf'):
            second = best
        res = (best, second)
        best_two_cache[key] = res
        if sig == all_sig:
            best_two_cache[all_key] = res
        return res

    # ---------- GRASP constructor (regret-guided insertion) ----------
    def construct_regret_insertion(seed_txn=None, rcl_k=3):
        remaining = set(all_txns)
        seq = []
        if seed_txn is None:
            start_candidates = random.sample(all_txns, min(8, n))
            starter = min(start_candidates, key=lambda t: eval_cost([t]))
        else:
            starter = seed_txn
        seq.append(starter)
        remaining.remove(starter)

        # Try to choose a good second txn/position
        if remaining:
            k = min(6, len(remaining))
            cands = random.sample(list(remaining), k)
            best_pair = (float('inf'), None, None)
            for t in cands:
                for pos in [0, 1]:
                    cand = seq[:]
                    cand.insert(pos, t)
                    c = eval_cost(cand)
                    if c < best_pair[0]:
                        best_pair = (c, t, pos)
            if best_pair[1] is not None:
                seq.insert(best_pair[2], best_pair[1])
                remaining.remove(best_pair[1])

        while remaining:
            # Candidate transactions
            k_t = min(len(remaining), max(4, cand_txn_sample + random.randint(-2, 2)))
            cand_txns = random.sample(list(remaining), k_t)

            scored = []
            for t in cand_txns:
                (c1, p1), (c2, p2) = best_two_insertions(seq, t)
                regret = (c2 - c1) if c2 < float('inf') else (0.0)
                scored.append((c1, regret, t, p1))

            if not scored:
                t = random.choice(list(remaining))
                seq.append(t)
                remaining.remove(t)
                continue

            # Build RCL from best cost and high regret
            scored.sort(key=lambda x: x[0])  # by cost
            rcl_cost = scored[:max(1, min(rcl_k, len(scored)))]
            scored.sort(key=lambda x: (-x[1], x[0]))  # by regret desc
            rcl_regret = scored[:max(1, min(rcl_k, len(scored)))]

            pool = { (c, r, t, p) for (c, r, t, p) in (rcl_cost + rcl_regret) }
            pool = list(pool)

            # Weighted pick: favor low cost with probability alpha; else high regret
            if random.random() < rcl_alpha:
                pool.sort(key=lambda x: x[0])
            else:
                pool.sort(key=lambda x: (-x[1], x[0]))
            chosen_c, chosen_r, chosen_t, chosen_p = random.choice(pool[:max(1, min(3, len(pool)))])
            seq.insert(chosen_p, chosen_t)
            remaining.remove(chosen_t)

        return eval_cost(seq), seq

    # ---------- Beam search with lookahead and regret diversity ----------
    def beam_search():
        # Initialize with best singletons and 1 GRASP seed
        starters = [(eval_cost([t]), [t]) for t in all_txns]
        starters.sort(key=lambda x: x[0])
        init = starters[:min(len(starters), max(beam_width * 2, beam_width + 2))]

        c0, s0 = construct_regret_insertion()
        init.append((c0, s0))

        # Beam state: (cost, seq, remaining_set)
        beam = []
        seen = set()
        for c, seq in init:
            key = tuple(seq)
            if key in seen:
                continue
            seen.add(key)
            rem = frozenset(t for t in all_txns if t not in seq)
            beam.append((c, seq, rem))
            if len(beam) >= beam_width:
                break

        best_complete = (float('inf'), [])

        for depth in range(1, n + 1):
            if not beam:
                break
            next_pool = []
            layer_seen = set()
            suffix_seen = set()  # suffix diversity on last-2
            for c, seq, rem in beam:
                if not rem:
                    if c < best_complete[0]:
                        best_complete = (c, seq)
                    continue

                rem_list = list(rem)
                expand_list = rem_list if len(rem_list) <= cand_per_state else random.sample(rem_list, cand_per_state)

                scored = []
                for t in expand_list:
                    # Insert t at its best position for this prefix
                    (best1, pos1), (second1, _) = best_two_insertions(seq, t, exhaustive=(len(seq) <= EXHAUSTIVE_THRESHOLD))
                    c1 = best1
                    new_seq = seq[:]
                    new_seq.insert(pos1, t)

                    # lookahead: best-insertion among k sampled second steps
                    rem_after = [x for x in rem if x != t]
                    if rem_after:
                        k2 = min(lookahead_k, len(rem_after))
                        second = random.sample(rem_after, k2)
                        best_c2 = float('inf')
                        second_costs = []
                        for u in second:
                            (cu, _), _ = best_two_insertions(new_seq, u, exhaustive=(len(new_seq) <= EXHAUSTIVE_THRESHOLD))
                            second_costs.append(cu)
                            if cu < best_c2:
                                best_c2 = cu
                        spread = (max(second_costs) - min(second_costs)) if len(second_costs) >= 2 else 0.0
                        primary_regret = max(0.0, second1 - c1)
                        regret = 0.5 * spread + 0.5 * primary_regret
                        score = 0.7 * c1 + 0.3 * best_c2
                    else:
                        regret = max(0.0, second1 - c1)
                        score = c1

                    scored.append((score, c1, regret, t, pos1))

                if not scored:
                    continue

                scored.sort(key=lambda x: x[0])
                top_cost = scored[:max(1, min(len(scored), beam_width))]

                scored_by_regret = sorted(scored, key=lambda x: (-x[2], x[0]))
                top_regret = scored_by_regret[:min(diversity_quota, len(scored_by_regret))]

                cand_acts = top_cost + top_regret
                # Deduplicate by (txn, pos) to keep distinct placements
                uniq = {}
                for sc, c1, rg, t, pos in cand_acts:
                    k = (t, pos)
                    if k not in uniq or c1 < uniq[k][1]:
                        uniq[k] = (sc, c1, rg)
                for (t, pos), (sc, c1, rg) in uniq.items():
                    new_seq = seq[:]
                    new_seq.insert(pos, t)
                    new_rem = frozenset(x for x in rem if x != t)

                    if len(new_seq) >= 2:
                        sig = (new_seq[-2], new_seq[-1])
                    else:
                        sig = (None, new_seq[-1])

                    key = tuple(new_seq)
                    if key in layer_seen:
                        continue
                    if sig in suffix_seen:
                        continue

                    layer_seen.add(key)
                    suffix_seen.add(sig)
                    next_pool.append((c1, new_seq, new_rem))

            if not next_pool:
                break

            next_pool.sort(key=lambda x: x[0])
            pruned = []
            seen_prefixes = set()
            for c1, seq, rem in next_pool:
                key = tuple(seq)
                if key in seen_prefixes:
                    continue
                seen_prefixes.add(key)
                pruned.append((c1, seq, rem))
                if len(pruned) >= beam_width:
                    break
            beam = pruned

        for c, seq, rem in beam:
            if not rem and c < best_complete[0]:
                best_complete = (c, seq)

        if best_complete[1] and len(best_complete[1]) == n:
            return best_complete

        # Greedy completion from best partial
        if beam:
            c, seq, rem = min(beam, key=lambda x: x[0])
            cur = list(seq)
            rem_list = list(rem)
            while rem_list:
                best_t = None
                best_pos = 0
                best_c = float('inf')
                for t in rem_list:
                    (c2, p2), _ = best_two_insertions(cur, t, exhaustive=(len(cur) <= EXHAUSTIVE_THRESHOLD))
                    if c2 < best_c:
                        best_c = c2
                        best_t = t
                        best_pos = p2
                cur.insert(best_pos, best_t)
                rem_list.remove(best_t)
            return eval_cost(cur), cur

        # Fallback
        ident = list(range(n))
        return eval_cost(ident), ident

    # ---------- Local Search: VND with Or-opt, swaps, and sampled pair swaps ----------
    def vnd_local_search(seq, start_cost, max_rounds=vnd_rounds):
        best_seq = list(seq)
        best_cost = start_cost
        n_local = len(best_seq)
        if n_local <= 2:
            return best_cost, best_seq

        # DLB flags for indices; reset on improvements
        dont_look = [False] * n_local

        def or_opt_pass(block_len):
            nonlocal best_seq, best_cost, dont_look
            improved = False
            n_cur = len(best_seq)
            i = 0
            while i <= n_cur - block_len:
                if i < len(dont_look) and dont_look[i]:
                    i += 1
                    continue
                block = best_seq[i:i + block_len]
                base = best_seq[:i] + best_seq[i + block_len:]
                m = len(base) + 1
                positions = sample_positions(len(base))
                best_move_cost = best_cost
                best_pos = None
                for pos in positions:
                    cand = base[:]
                    cand[pos:pos] = block
                    c = eval_cost(cand)
                    if c < best_move_cost:
                        best_move_cost = c
                        best_pos = pos
                if best_pos is not None:
                    new_seq = base[:]
                    new_seq[best_pos:best_pos] = block
                    best_seq = new_seq
                    best_cost = best_move_cost
                    improved = True
                    dont_look = [False] * len(best_seq)
                else:
                    if i < len(dont_look):
                        dont_look[i] = True
                    i += 1
            return improved

        def adjacent_swap_pass():
            nonlocal best_seq, best_cost, dont_look
            improved = False
            i = 0
            while i < len(best_seq) - 1:
                cand = best_seq[:]
                cand[i], cand[i + 1] = cand[i + 1], cand[i]
                c = eval_cost(cand)
                if c < best_cost:
                    best_cost = c
                    best_seq = cand
                    improved = True
                    dont_look = [False] * len(best_seq)
                i += 1
            return improved

        def sampled_pair_swaps(tries=pair_swap_tries):
            nonlocal best_seq, best_cost, dont_look
            if len(best_seq) <= 4:
                return False
            improved = False
            best_delta = 0.0
            best_i = best_j = None
            L = len(best_seq)
            attempts = min(tries, max(20, L))
            for _ in range(attempts):
                i = random.randint(0, L - 1)
                j = random.randint(0, L - 1)
                if i == j or abs(i - j) <= 1:
                    continue
                cand = best_seq[:]
                cand[i], cand[j] = cand[j], cand[i]
                c = eval_cost(cand)
                delta = best_cost - c
                if delta > best_delta:
                    best_delta = delta
                    best_i, best_j = i, j
            if best_i is not None:
                cand = best_seq[:]
                cand[best_i], cand[best_j] = cand[best_j], cand[best_i]
                best_seq = cand
                best_cost = eval_cost(best_seq)
                improved = True
                dont_look = [False] * len(best_seq)
            return improved

        rounds = 0
        while rounds < max_rounds:
            rounds += 1
            any_improved = False
            for bl in (3, 2, 1):
                if or_opt_pass(bl):
                    any_improved = True
            if adjacent_swap_pass():
                any_improved = True
            if sampled_pair_swaps():
                any_improved = True
            if not any_improved:
                break

        return best_cost, best_seq

    # ---------- LNS (ruin-and-recreate) ----------
    def lns_ruin_and_recreate(seq, start_cost, rounds=lns_rounds):
        best_seq = list(seq)
        best_cost = start_cost

        for _ in range(rounds):
            current_seq = list(best_seq)
            n_cur = len(current_seq)
            if n_cur <= 6:
                break
            # number to remove
            k_remove = max(lns_remove_min, min(lns_remove_max, int(lns_remove_frac * n_cur)))
            k_remove = min(k_remove, n_cur - 2)

            remove_idx = set()

            # Contiguous segment
            if n_cur > k_remove:
                start = random.randint(0, max(0, n_cur - k_remove - 1))
                length = max(2, min(k_remove - 2, n_cur - start))
                for j in range(start, start + length):
                    if len(remove_idx) < k_remove:
                        remove_idx.add(j)

            # Scattered removals to fill
            while len(remove_idx) < k_remove:
                remove_idx.add(random.randint(0, n_cur - 1))
            remove_idx = sorted(remove_idx)

            removed = [current_seq[i] for i in remove_idx]
            base = [current_seq[i] for i in range(n_cur) if i not in remove_idx]

            # Regret-guided reinsertion
            seq_partial = list(base)
            remaining = list(removed)
            random.shuffle(remaining)
            while remaining:
                # sample some of the remaining to speed up
                k_t = min(len(remaining), max(4, cand_txn_sample // 2))
                cand_txns = random.sample(remaining, k_t)
                scored = []
                for t in cand_txns:
                    (c1, p1), (c2, p2) = best_two_insertions(seq_partial, t)
                    scored.append((c1, c2, t, p1))
                if not scored:
                    t = remaining.pop()
                    seq_partial.append(t)
                    continue
                # RCL by cost and regret
                scored.sort(key=lambda x: x[0])
                rcl_cost = scored[:max(1, min(2, len(scored)))]
                scored.sort(key=lambda x: (-(x[1] - x[0]), x[0]))
                rcl_regret = scored[:max(1, min(2, len(scored)))]
                pool = list({(c1, c2, t, p) for (c1, c2, t, p) in (rcl_cost + rcl_regret)})
                chosen = min(pool, key=lambda x: x[0])
                c1, c2, t, p = chosen
                seq_partial.insert(p, t)
                remaining.remove(t)

            c_new = eval_cost(seq_partial)
            if c_new < best_cost:
                best_cost = c_new
                best_seq = seq_partial

        return best_cost, best_seq

    # ---------- Path relinking between elite sequences ----------
    def path_relink(a_seq, b_seq):
        # transform a toward b; evaluate along the path
        target_pos = {t: i for i, t in enumerate(b_seq)}
        cur = list(a_seq)
        best_c = eval_cost(cur)
        best_s = list(cur)
        for i in range(len(cur)):
            desired = b_seq[i]
            j = cur.index(desired)
            if j == i:
                continue
            elem = cur.pop(j)
            cur.insert(i, elem)
            c = eval_cost(cur)
            if c < best_c:
                best_c = c
                best_s = list(cur)
        return best_c, best_s

    # ---------- Elite pool management ----------
    elites = []

    def add_elite(cost, seq):
        nonlocal elites
        if not seq or len(seq) != n:
            return
        key = tuple(seq)
        for c, s in elites:
            if tuple(s) == key:
                if cost < c:
                    elites.remove((c, s))
                    elites.append((cost, list(seq)))
                return
        elites.append((cost, list(seq)))
        elites.sort(key=lambda x: x[0])
        if len(elites) > elite_cap:
            elites = elites[:elite_cap]

    # ---------- Main search orchestration ----------
    best_cost = float('inf')
    best_seq = list(range(n))

    # Seed RNG for per-call diversity
    random.seed((n * 911 + num_seqs * 131 + 7) % (2**32 - 1))

    # 1) GRASP-style regret seeds
    for _ in range(seeds):
        c, s = construct_regret_insertion()
        c, s = vnd_local_search(s, c, max_rounds=1)
        add_elite(c, s)
        if c < best_cost:
            best_cost, best_seq = c, s

    # 2) Beam search seeds
    beam_runs = max(1, num_seqs // 3)
    for _ in range(beam_runs):
        c, s = beam_search()
        c, s = vnd_local_search(s, c, max_rounds=1)
        add_elite(c, s)
        if c < best_cost:
            best_cost, best_seq = c, s

    # 3) Recombine elites via path relinking
    if len(elites) >= 2:
        base_cost, base_seq = elites[0]
        for i in range(1, min(len(elites), 3)):
            _, other_seq = elites[i]
            c_rel, s_rel = path_relink(base_seq, other_seq)
            c_rel, s_rel = vnd_local_search(s_rel, c_rel, max_rounds=1)
            add_elite(c_rel, s_rel)
            if c_rel < best_cost:
                best_cost, best_seq = c_rel, s_rel

    # 4) Iterated LNS + VND from incumbent/elite
    incumbent_cost, incumbent_seq = best_cost, best_seq
    for _ in range(ils_iterations):
        # Diversify by picking an elite as a start occasionally
        if elites and random.random() < 0.4:
            start_c, start_s = random.choice(elites[:min(3, len(elites))])
        else:
            start_c, start_s = incumbent_cost, incumbent_seq

        c1, s1 = lns_ruin_and_recreate(start_s, start_c, rounds=lns_rounds)
        c2, s2 = vnd_local_search(s1, c1, max_rounds=vnd_rounds)
        if c2 < incumbent_cost:
            incumbent_cost, incumbent_seq = c2, s2
            add_elite(c2, s2)
        # Small adjacent polish
        c3, s3 = vnd_local_search(incumbent_seq, incumbent_cost, max_rounds=1)
        if c3 < incumbent_cost:
            incumbent_cost, incumbent_seq = c3, s3
            add_elite(c3, s3)

    best_cost, best_seq = incumbent_cost, incumbent_seq

    # Safety check
    assert len(best_seq) == n and len(set(best_seq)) == n, "Schedule must include each transaction exactly once"

    return best_cost, best_seq


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