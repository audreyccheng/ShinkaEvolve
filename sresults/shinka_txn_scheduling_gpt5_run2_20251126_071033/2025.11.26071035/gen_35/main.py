# EVOLVE-BLOCK-START
"""Transaction scheduling algorithm for optimizing makespan across multiple workloads
Crossover: GRASP + regret-diverse beam + VND (DLB) + LNS + path relinking with cached evaluations.
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
    Hybrid GRASP + beam + VND (DLB) + LNS + path relinking to minimize makespan.

    Args:
        workload: Workload object
        num_seqs: Search effort parameter (controls beam width, restarts, and refinement)

    Returns:
        (best_cost, best_schedule)
    """
    n = workload.num_txns
    all_txns = list(range(n))

    # Adaptive parameterization by problem size and effort
    small = n <= 50
    med = 50 < n <= 90
    large = n > 90

    # Construction parameters
    CAND_SAMPLE_BASE = 12 if small else (10 if med else 8)
    POS_SAMPLE_LIMIT = None if small else (18 if med else 15)
    JITTER = 2

    # Beam search parameters
    beam_width = max(5, min(10, (6 if small else 5) + num_seqs // (2 if small else 3)))
    cand_per_state = min(28, max(10, n // (3 if small else 4)))
    lookahead_k = 7 if small else 6
    beam_alpha = 0.7  # weight of current step vs lookahead
    diversity_quota = max(1, int(0.25 * beam_width))

    # Local search parameters
    MAX_DLB_CYCLES = 2
    MAX_ADJ_PASSES = 2
    # Relocations budget scaled by size and effort
    RELOC_TRIES = int((max(10, n // (2 if small else 3))) * max(1.0, min(2.0, 0.5 + 0.1 * max(1, num_seqs))))

    # LNS parameters
    lns_rounds = 2 if med or large else 3
    lns_remove_frac = 0.09 if large else (0.1 if med else 0.12)
    lns_remove_min = 8 if med or large else 6
    lns_remove_max = 18 if large else (16 if med else 14)

    # Multi-start / elites
    restarts = max(3, num_seqs)
    elite_cap = 4

    # Cached evaluator for partial prefixes
    @lru_cache(maxsize=200_000)
    def _eval_cost_tuple(seq_tuple):
        return workload.get_opt_seq_cost(list(seq_tuple))

    def eval_cost(seq):
        return _eval_cost_tuple(tuple(seq))

    # Utility: sample insertion positions
    def sample_positions(seq_len):
        total = seq_len + 1
        if POS_SAMPLE_LIMIT is None or total <= POS_SAMPLE_LIMIT:
            return list(range(total))
        # Keep ends, sample interior
        interior = list(range(1, seq_len))
        k = max(2, min(POS_SAMPLE_LIMIT - 2, len(interior)))
        chosen = set(random.sample(interior, k)) if interior else set()
        chosen.update({0, seq_len})
        return sorted(chosen)

    # Best and second-best insertion positions for a transaction
    def best_two_insertions(seq, txn):
        best_cost = float('inf')
        second = float('inf')
        best_pos = 0
        for pos in sample_positions(len(seq)):
            cand = seq[:]
            cand.insert(pos, txn)
            c = eval_cost(cand)
            if c < best_cost:
                second = best_cost
                best_cost = c
                best_pos = pos
            elif c < second:
                second = c
        if second == float('inf'):
            second = best_cost
        return (best_cost, best_pos), (second, None)

    # ---------------------- GRASP Constructor ----------------------
    def construct_grasp():
        remaining = set(all_txns)
        # pick a good starter from sampled candidates
        start_candidates = random.sample(all_txns, min(10, n))
        starter = min(start_candidates, key=lambda t: eval_cost([t]))
        seq = [starter]
        remaining.remove(starter)

        # good second placement via small RCL
        if remaining:
            k = min(8, len(remaining))
            pairs = []
            for t in random.sample(list(remaining), k):
                for pos in [0, 1]:
                    cand = seq[:]
                    cand.insert(pos, t)
                    c = eval_cost(cand)
                    pairs.append((c, t, pos))
            if pairs:
                pairs.sort(key=lambda x: x[0])
                rcl = pairs[:min(3, len(pairs))]
                c2, t2, p2 = random.choice(rcl)
                seq.insert(p2, t2)
                remaining.remove(t2)

        # iterative regret-guided insertion
        while remaining:
            # candidate txns (endgame widen)
            if len(remaining) <= max(8, 2 * CAND_SAMPLE_BASE):
                cand_txns = list(remaining)
            else:
                size = min(len(remaining), max(4, CAND_SAMPLE_BASE + random.randint(-JITTER, JITTER)))
                cand_txns = random.sample(list(remaining), size)

            scored = []
            for t in cand_txns:
                (c1, p1), (c2, _) = best_two_insertions(seq, t)
                regret = max(0.0, c2 - c1)
                scored.append((c1, regret, t, p1))
            if not scored:
                t = random.choice(list(remaining))
                seq.append(t)
                remaining.remove(t)
                continue

            # RCL primarily by best insertion cost, tie-break by regret
            scored.sort(key=lambda x: x[0])
            rcl_size = max(1, min(3, len(scored) // 2 if len(scored) > 3 else len(scored)))
            rcl = scored[:rcl_size]
            # 50% chance choose highest regret in RCL for robustness
            if random.random() < 0.5:
                chosen = max(rcl, key=lambda x: x[1])
            else:
                chosen = random.choice(rcl)
            c_step, _, t_step, pos_step = chosen
            seq.insert(pos_step, t_step)
            remaining.remove(t_step)

        return eval_cost(seq), seq

    # ---------------------- Beam Search ----------------------
    def suffix_sig(seq):
        if len(seq) >= 2:
            return (seq[-2], seq[-1])
        if len(seq) == 1:
            return (seq[-1],)
        return ()

    def beam_search():
        # Initialize with best singletons and one GRASP seed
        starters = [(eval_cost([t]), [t]) for t in all_txns]
        starters.sort(key=lambda x: x[0])
        init = starters[:min(len(starters), max(beam_width * 2, beam_width + 2))]

        gc, gs = construct_grasp()
        init.append((gc, gs))

        # State: (cost, prefix, remaining_tuple)
        beam = []
        used = set()
        for c, s in init:
            key = tuple(s)
            if key in used:
                continue
            used.add(key)
            rem = tuple(x for x in all_txns if x not in s)
            beam.append((c, s, rem))
            if len(beam) >= beam_width:
                break

        best_complete = (float('inf'), [])

        for _depth in range(1, n + 1):
            if not beam:
                break
            next_pool = []
            layer_seen = set()
            suffix_seen = set()
            for c_so_far, prefix, rem in beam:
                if not rem:
                    if c_so_far < best_complete[0]:
                        best_complete = (c_so_far, prefix)
                    continue
                rem_list = list(rem)
                # Candidate expansions
                expand = rem_list if len(rem_list) <= cand_per_state else random.sample(rem_list, cand_per_state)
                scored = []
                for t in expand:
                    seq1 = prefix + [t]
                    c1 = eval_cost(seq1)
                    # two-step lookahead
                    rem_after = [x for x in rem_list if x != t]
                    if rem_after:
                        k2 = min(lookahead_k, len(rem_after))
                        second = rem_after if k2 == len(rem_after) else random.sample(rem_after, k2)
                        best_c2 = float('inf')
                        for u in second:
                            cu = eval_cost(seq1 + [u])
                            if cu < best_c2:
                                best_c2 = cu
                        score = beam_alpha * c1 + (1.0 - beam_alpha) * best_c2
                        # simple regret proxy among two top second-steps
                        regret = max(0.0, best_c2 - c1)
                    else:
                        score = c1
                        regret = 0.0
                    scored.append((score, c1, regret, t))

                if not scored:
                    continue
                scored.sort(key=lambda x: x[0])
                top_cost = scored[:max(1, min(beam_width, len(scored)))]
                # add high-regret alternatives
                top_regret = sorted(scored, key=lambda x: (-x[2], x[0]))[:min(diversity_quota, len(scored))]
                cand_acts = top_cost + top_regret

                # Dedup by t keeping best c1
                uniq = {}
                for sc, c1, rg, t in cand_acts:
                    if (t not in uniq) or (c1 < uniq[t][0]):
                        uniq[t] = (c1, sc, rg)

                for t, (c1, sc, rg) in uniq.items():
                    new_seq = prefix + [t]
                    new_rem = tuple(x for x in rem if x != t)
                    sig = suffix_sig(new_seq)
                    key = tuple(new_seq)
                    if key in layer_seen or sig in suffix_seen:
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

        # If complete found
        for c, s, rem in beam:
            if not rem and c < best_complete[0]:
                best_complete = (c, s)
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

        # Fallback identity
        ident = list(range(n))
        return eval_cost(ident), ident

    # ---------------------- Local Search (VND with DLB) ----------------------
    def local_adjacent_swaps(seq, curr_cost, max_passes=MAX_ADJ_PASSES):
        best_seq = list(seq)
        best_cost = curr_cost
        passes = 0
        improved = True
        while improved and passes < max_passes:
            improved = False
            for i in range(len(best_seq) - 1):
                cand = best_seq[:]
                cand[i], cand[i + 1] = cand[i + 1], cand[i]
                c = eval_cost(cand)
                if c < best_cost:
                    best_seq, best_cost = cand, c
                    improved = True
            passes += 1
        return best_seq, best_cost

    def local_oropt_dlb(seq, curr_cost, cycles=MAX_DLB_CYCLES):
        best_seq = list(seq)
        best_cost = curr_cost
        n_local = len(best_seq)
        if n_local <= 2:
            return best_seq, best_cost

        dlb = [False] * n_local
        cycles_done = 0

        def positions_for(m):
            if POS_SAMPLE_LIMIT is None or m <= POS_SAMPLE_LIMIT:
                return list(range(m))
            interior = list(range(1, m - 1))
            k = min(POS_SAMPLE_LIMIT - 2, len(interior)) if interior else 0
            chosen = set(random.sample(interior, k)) if k > 0 else set()
            chosen.update({0, m - 1})
            return sorted(chosen)

        while cycles_done < cycles:
            improved_round = False
            for blk in (3, 2, 1):
                if blk > len(best_seq):
                    continue
                move_applied = False
                for i in range(0, len(best_seq) - blk + 1):
                    if i < len(dlb) and dlb[i]:
                        continue
                    block = best_seq[i:i + blk]
                    base = best_seq[:i] + best_seq[i + blk:]
                    m = len(base) + 1
                    best_delta = 0.0
                    best_pos = None
                    for pos in positions_for(m):
                        cand = base[:]
                        cand[pos:pos] = block
                        c = eval_cost(cand)
                        delta = best_cost - c
                        if delta > best_delta:
                            best_delta = delta
                            best_pos = pos
                    if best_pos is not None:
                        base[best_pos:best_pos] = block
                        best_seq = base
                        best_cost = eval_cost(best_seq)
                        dlb = [False] * len(best_seq)
                        improved_round = True
                        move_applied = True
                        break
                    else:
                        if i < len(dlb):
                            dlb[i] = True
                if move_applied:
                    break
            if not improved_round:
                cycles_done += 1
            else:
                cycles_done = 0
        return best_seq, best_cost

    def vnd_refine(seq, start_cost=None, rounds=1):
        c0 = eval_cost(seq) if start_cost is None else start_cost
        best_seq, best_cost = seq, c0
        for _ in range(max(1, rounds)):
            s, c = local_oropt_dlb(best_seq, best_cost, cycles=MAX_DLB_CYCLES)
            if c < best_cost:
                best_seq, best_cost = s, c
            s, c = local_adjacent_swaps(best_seq, best_cost, max_passes=1)
            if c < best_cost:
                best_seq, best_cost = s, c
        return best_seq, best_cost

    # ---------------------- LNS (ruin & recreate) ----------------------
    def lns_ruin_and_recreate(seq, start_cost, rounds=lns_rounds):
        best_seq = list(seq)
        best_cost = start_cost
        for _ in range(rounds):
            cur_seq = list(best_seq)
            L = len(cur_seq)
            if L <= 6:
                break
            # number to remove
            k_remove = max(lns_remove_min, min(lns_remove_max, int(lns_remove_frac * L)))
            k_remove = min(k_remove, L - 2)

            remove_idx = set()
            # contiguous block
            if L > k_remove:
                start = random.randint(0, max(0, L - k_remove - 1))
                length = max(2, min(k_remove - 2, L - start))
                for j in range(start, start + length):
                    if len(remove_idx) < k_remove:
                        remove_idx.add(j)
            # scattered fill
            while len(remove_idx) < k_remove:
                remove_idx.add(random.randint(0, L - 1))
            remove_idx = sorted(remove_idx)

            removed = [cur_seq[i] for i in remove_idx]
            base = [cur_seq[i] for i in range(L) if i not in remove_idx]

            # regret-guided reinsertion
            rebuilt = list(base)
            remaining = list(removed)
            random.shuffle(remaining)
            while remaining:
                sample = remaining if len(remaining) <= 8 else random.sample(remaining, 8)
                best_t = None
                best_p = 0
                best_c = float('inf')
                best_regret = -1.0
                for t in sample:
                    (c1, p1), (c2, _) = best_two_insertions(rebuilt, t)
                    regret = max(0.0, c2 - c1)
                    if c1 < best_c or (abs(c1 - best_c) < 1e-9 and regret > best_regret):
                        best_c, best_p, best_t, best_regret = c1, p1, t, regret
                if best_t is None:
                    best_t = remaining[0]
                    best_p = len(rebuilt)
                rebuilt.insert(best_p, best_t)
                remaining.remove(best_t)

            c_new = eval_cost(rebuilt)
            if c_new < best_cost:
                best_cost = c_new
                best_seq = rebuilt
        return best_cost, best_seq

    # ---------------------- Path Relinking ----------------------
    def path_relink(a_seq, b_seq):
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

    # ---------------------- Elite Pool ----------------------
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
                break
        else:
            elites.append((cost, list(seq)))
        elites.sort(key=lambda x: x[0])
        if len(elites) > elite_cap:
            elites[:] = elites[:elite_cap]

    # ---------------------- Orchestration ----------------------
    best_cost = float('inf')
    best_seq = list(range(n))

    # Seed RNG mildly for per-call diversity
    random.seed((n * 911 + num_seqs * 131 + 7) % (2**32 - 1))

    # 1) GRASP seeds
    grasp_seeds = max(2, min(6, restarts))
    for _ in range(grasp_seeds):
        c, s = construct_grasp()
        c, s = vnd_refine(s, c, rounds=1)
        add_elite(c, s)
        if c < best_cost:
            best_cost, best_seq = c, s

    # 2) Beam search seeds
    beam_runs = max(1, num_seqs // 3)
    for _ in range(beam_runs):
        c, s = beam_search()
        c, s = vnd_refine(s, c, rounds=1)
        add_elite(c, s)
        if c < best_cost:
            best_cost, best_seq = c, s

    # 3) Path relinking among elites
    if len(elites) >= 2:
        base_c, base_s = elites[0]
        for i in range(1, min(len(elites), 3)):
            _, other = elites[i]
            rc, rs = path_relink(base_s, other)
            rc, rs = vnd_refine(rs, rc, rounds=1)
            add_elite(rc, rs)
            if rc < best_cost:
                best_cost, best_seq = rc, rs

    # 4) Iterated LNS + VND improvement
    incumbent_c, incumbent_s = best_cost, best_seq
    ils_iterations = max(2, 2 if large else 3)
    for _ in range(ils_iterations):
        # diversify from elites or incumbent
        if elites and random.random() < 0.4:
            start_c, start_s = random.choice(elites[:min(3, len(elites))])
        else:
            start_c, start_s = incumbent_c, incumbent_s
        c1, s1 = lns_ruin_and_recreate(start_s, start_c, rounds=lns_rounds)
        c2, s2 = vnd_refine(s1, c1, rounds=2 if small else 1)
        if c2 < incumbent_c:
            incumbent_c, incumbent_s = c2, s2
            add_elite(c2, s2)
        # quick adjacent polish
        s3, c3 = local_adjacent_swaps(incumbent_s, incumbent_c, max_passes=1)
        if c3 < incumbent_c:
            incumbent_c, incumbent_s = c3, s3
            add_elite(c3, s3)

    best_cost, best_seq = incumbent_c, incumbent_s

    # Safety checks
    assert len(best_seq) == n and len(set(best_seq)) == n, "Schedule must include each txn exactly once"

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