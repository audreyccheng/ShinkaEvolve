# EVOLVE-BLOCK-START
"""Transaction scheduling algorithm for optimizing makespan across multiple workloads"""

import time
import random
import sys
import os
from collections import deque, defaultdict

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
    Hybrid constructor + beam + LNS + VND to minimize makespan.

    Args:
        workload: Workload object containing transaction data
        num_seqs: Search effort parameter (drives beam width, seeds, and refinement)

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    n = workload.num_txns
    txns_all = list(range(n))

    # ------- Parameterization (adaptive) -------
    small = n <= 40
    med = 40 < n <= 80
    large = n > 80

    # Construction
    rcl_alpha = 0.6  # RCL balance for cost vs regret
    cand_txn_sample = 10 if med or large else 12
    pos_sample_cap = 18 if large else (None if small else 24)

    # Beam search
    beam_width = max(5, min(10, num_seqs + (2 if small else 0)))
    cand_per_state = min(28, max(10, n // (4 if large else 3)))
    lookahead_k = 6 if med or large else 8
    diversity_quota = max(1, int(0.25 * beam_width))  # 25% for regret-diverse expansions

    # Seeds and elites
    seeds = max(3, min(6, num_seqs))
    elite_cap = 4

    # Local search
    ls_adj_passes = 2 if large else 3
    vnd_rounds = 2 if large else 3

    # LNS
    lns_rounds = 2 if med or large else 3
    lns_remove_frac = 0.08 if large else (0.1 if med else 0.12)
    lns_remove_min = 8 if med or large else 6
    lns_remove_max = 16 if med or large else 14

    # Iterated improvements
    ils_iterations = 2 if large else 3

    # ------- Cost evaluator with cache -------
    cost_cache = {}

    def eval_cost(prefix):
        key = tuple(prefix)
        c = cost_cache.get(key)
        if c is None:
            c = workload.get_opt_seq_cost(list(prefix))
            cost_cache[key] = c
        return c

    # ------- Utility: sample positions for insertion -------
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

    # ------- Insertion helpers (compute best and second-best insertion for a txn) -------
    def best_two_insertions(seq, txn):
        positions = sample_positions(len(seq))
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
        return best, second

    # ------- Regret-guided GRASP constructor -------
    def construct_regret_insertion(seed_txn=None, rcl_k=3):
        remaining = set(txns_all)
        seq = []
        if seed_txn is None:
            start_candidates = random.sample(txns_all, min(8, n))
            # choose best single starter by partial cost
            starter = min(start_candidates, key=lambda t: eval_cost([t]))
        else:
            starter = seed_txn
        seq.append(starter)
        remaining.remove(starter)

        # Optionally pick good second
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

            # Build RCL from both best cost and high regret
            scored.sort(key=lambda x: x[0])  # by best cost
            rcl_cost = scored[:max(1, min(rcl_k, len(scored)))]
            scored.sort(key=lambda x: (-x[1], x[0]))  # by regret desc, then cost
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

    # ------- Beam search with lookahead and regret diversity -------
    def beam_search():
        # Initialize with top singletons (and 1 GRASP seed)
        starters = [(eval_cost([t]), [t]) for t in txns_all]
        starters.sort(key=lambda x: x[0])
        init = starters[:min(len(starters), max(beam_width * 2, beam_width + 2))]

        # Add a GRASP seed for diversity
        c0, s0 = construct_regret_insertion()
        init.append((c0, s0))

        # Beam state: tuples (cost, seq, remaining_set)
        beam = []
        seen = set()
        for c, seq in init:
            key = tuple(seq)
            if key in seen:
                continue
            seen.add(key)
            rem = frozenset(txn for txn in txns_all if txn not in seq)
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
                # Sample candidates
                if len(rem_list) <= cand_per_state:
                    expand_list = rem_list
                else:
                    expand_list = random.sample(rem_list, cand_per_state)

                # Score expansions with lookahead and regret-based alternative
                scored = []
                for t in expand_list:
                    seq1 = seq + [t]
                    c1 = eval_cost(seq1)

                    # lookahead: best among k sampled second steps
                    rem_after = [x for x in rem if x != t]
                    if rem_after:
                        k2 = min(lookahead_k, len(rem_after))
                        second = random.sample(rem_after, k2)
                        best_c2 = float('inf')
                        second_costs = []
                        for u in second:
                            cu = eval_cost(seq1 + [u])
                            second_costs.append(cu)
                            if cu < best_c2:
                                best_c2 = cu
                        # regret approx: variance among second step costs
                        if len(second_costs) >= 2:
                            mx = max(second_costs)
                            mn = min(second_costs)
                            regret = mx - mn
                        else:
                            regret = 0.0
                        score = 0.7 * c1 + 0.3 * best_c2
                    else:
                        regret = 0.0
                        score = c1

                    scored.append((score, c1, regret, t))

                # Select by cost and regret diversity
                scored.sort(key=lambda x: x[0])  # by score
                top_cost = scored[:max(1, min(len(scored), beam_width))]
                # Add regret-heavy options
                scored_by_regret = sorted(scored, key=lambda x: (-x[2], x[0]))
                top_regret = scored_by_regret[:min(diversity_quota, len(scored_by_regret))]

                cand_acts = top_cost + top_regret
                # Dedup actions
                uniq = {}
                for sc, c1, rg, t in cand_acts:
                    if t not in uniq or c1 < uniq[t][1]:
                        uniq[t] = (sc, c1, rg)
                for t, (sc, c1, rg) in uniq.items():
                    new_seq = seq + [t]
                    new_rem = frozenset(x for x in rem if x != t)

                    # Suffix diversity: last-2 signature
                    if len(new_seq) >= 2:
                        sig = (new_seq[-2], new_seq[-1])
                    else:
                        sig = (None, new_seq[-1])

                    key = tuple(new_seq)
                    if key in layer_seen:
                        continue
                    if sig in suffix_seen:
                        # allow only if significantly better than previous signatures in pool
                        # We'll just skip to maintain diversity
                        continue

                    layer_seen.add(key)
                    suffix_seen.add(sig)
                    next_pool.append((c1, new_seq, new_rem))

            if not next_pool:
                break

            # Prune to beam width by cost; deduplicate by full prefix
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

        # Complete if not already
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
                best_c = float('inf')
                for t in rem_list:
                    c2 = eval_cost(cur + [t])
                    if c2 < best_c:
                        best_c = c2
                        best_t = t
                cur.append(best_t)
                rem_list.remove(best_t)
            return eval_cost(cur), cur

        # Fallback: identity
        ident = list(range(n))
        return eval_cost(ident), ident

    # ------- Local search: VND with Or-opt (1,2,3), Adjacent swaps, and DLB -------
    def vnd_local_search(seq, start_cost, max_rounds=vnd_rounds):
        best_seq = list(seq)
        best_cost = start_cost
        n_local = len(best_seq)
        if n_local <= 2:
            return best_cost, best_seq

        # DLB flags for indices; reset when neighborhood changes locally
        dont_look = [False] * n_local

        def or_opt_pass(block_len):
            nonlocal best_seq, best_cost, dont_look
            improved = False
            n_cur = len(best_seq)
            i = 0
            while i <= n_cur - block_len:
                if dont_look[i]:
                    i += 1
                    continue
                block = best_seq[i:i + block_len]
                base = best_seq[:i] + best_seq[i + block_len:]
                m = len(base) + 1
                best_move_cost = best_cost
                best_pos = None
                # Try inserting block at all positions except original
                for pos in range(m):
                    if pos == i:
                        continue
                    cand = base[:]
                    cand[pos:pos] = block
                    c = eval_cost(cand)
                    if c < best_move_cost:
                        best_move_cost = c
                        best_pos = pos
                if best_pos is not None:
                    # Apply improving move
                    new_seq = base[:]
                    new_seq[best_pos:best_pos] = block
                    best_seq = new_seq
                    best_cost = best_move_cost
                    improved = True
                    # Reset DLB around affected region
                    dont_look = [False] * len(best_seq)
                    # Stay on same i due to structural change
                else:
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

        # VND loop across neighborhoods
        rounds = 0
        while rounds < max_rounds:
            rounds += 1
            any_improved = False
            # Or-opt 1,2,3
            for bl in (1, 2, 3):
                if or_opt_pass(bl):
                    any_improved = True
            if adjacent_swap_pass():
                any_improved = True
            if not any_improved:
                break

        return best_cost, best_seq

    # ------- LNS (ruin and recreate) using regret-guided reinsertion -------
    def lns_ruin_and_recreate(seq, start_cost, rounds=lns_rounds):
        best_seq = list(seq)
        best_cost = start_cost

        for _ in range(rounds):
            current_seq = list(best_seq)
            n_cur = len(current_seq)
            if n_cur <= 6:
                break
            # Decide number to remove
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

            # Scattered removals to fill up to k_remove
            while len(remove_idx) < k_remove:
                remove_idx.add(random.randint(0, n_cur - 1))
            remove_idx = sorted(remove_idx)

            removed = [current_seq[i] for i in remove_idx]
            base = [current_seq[i] for i in range(n_cur) if i not in remove_idx]

            # Reinsert removed using regret-guided insertion
            seq_partial = list(base)
            remaining = list(removed)
            random.shuffle(remaining)
            while remaining:
                # sample some of the remaining to speed up
                k_t = min(len(remaining), max(4, cand_txn_sample // 2))
                cand_txns = random.sample(remaining, k_t)
                best_choice = None
                best_pos = None
                best_c = float('inf')
                best_second = float('inf')
                # evaluate best and second best for regret
                scored = []
                for t in cand_txns:
                    (c1, p1), (c2, p2) = best_two_insertions(seq_partial, t)
                    scored.append((c1, c2, t, p1))
                # RCL by cost and regret
                scored.sort(key=lambda x: x[0])
                rcl_cost = scored[:max(1, min(2, len(scored)))]
                scored.sort(key=lambda x: (-(x[1] - x[0]), x[0]))
                rcl_regret = scored[:max(1, min(2, len(scored)))]
                pool = list({(c1, c2, t, p) for (c1, c2, t, p) in (rcl_cost + rcl_regret)})
                # choose best among small pool
                chosen = min(pool, key=lambda x: x[0])
                c1, c2, t, p = chosen
                seq_partial.insert(p, t)
                remaining.remove(t)

            c_new = eval_cost(seq_partial)
            if c_new < best_cost:
                best_cost = c_new
                best_seq = seq_partial

        return best_cost, best_seq

    # ------- Path relinking between elite sequences -------
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
            # move element at j to i
            elem = cur.pop(j)
            cur.insert(i, elem)
            c = eval_cost(cur)
            if c < best_c:
                best_c = c
                best_s = list(cur)
        return best_c, best_s

    # ------- Elite pool management -------
    elites = []

    def add_elite(cost, seq):
        nonlocal elites
        if not seq or len(seq) != n:
            return
        # Avoid duplicates
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

    # ------- Main search orchestration -------
    best_cost = float('inf')
    best_seq = list(range(n))

    # Seed RNG lightly for diversity per call
    random.seed((n * 911 + num_seqs * 131 + 7) % (2**32 - 1))

    # 1) GRASP-style regret seeds
    for _ in range(seeds):
        c, s = construct_regret_insertion()
        c, s = vnd_local_search(s, c, max_rounds=1)
        add_elite(c, s)
        if c < best_cost:
            best_cost, best_seq = c, s

    # 2) Beam search seeds (couple of runs)
    beam_runs = max(1, num_seqs // 3)
    for _ in range(beam_runs):
        c, s = beam_search()
        c, s = vnd_local_search(s, c, max_rounds=1)
        add_elite(c, s)
        if c < best_cost:
            best_cost, best_seq = c, s

    # 3) Recombine best elites via path relinking
    if len(elites) >= 2:
        base_cost, base_seq = elites[0]
        for i in range(1, min(len(elites), 3)):
            _, other = elites[i]
            c_rel, s_rel = path_relink(base_seq, other)
            c_rel, s_rel = vnd_local_search(s_rel, c_rel, max_rounds=1)
            add_elite(c_rel, s_rel)
            if c_rel < best_cost:
                best_cost, best_seq = c_rel, s_rel

    # 4) Iterated LNS + VND from the current best/elite
    incumbent_cost, incumbent_seq = best_cost, best_seq
    for _ in range(ils_iterations):
        # Diversify by starting from one of elites occasionally
        if elites and random.random() < 0.4:
            start_c, start_s = random.choice(elites[:min(3, len(elites))])
        else:
            start_c, start_s = incumbent_cost, incumbent_seq

        c1, s1 = lns_ruin_and_recreate(start_s, start_c, rounds=lns_rounds)
        c2, s2 = vnd_local_search(s1, c1, max_rounds=vnd_rounds)
        if c2 < incumbent_cost:
            incumbent_cost, incumbent_seq = c2, s2
            add_elite(c2, s2)
        # small adjacent polish
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