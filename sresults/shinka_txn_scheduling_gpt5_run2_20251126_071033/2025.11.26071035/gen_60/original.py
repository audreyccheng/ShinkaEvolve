# EVOLVE-BLOCK-START
"""Transaction scheduling algorithm for optimizing makespan across multiple workloads
Redesigned with modular constructor + regret-diverse beam + VND (DLB) + LNS/ILS.
"""

import time
import random
import sys
import os
from functools import lru_cache
from math import ceil

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
    Hybrid constructor + regret-diverse beam + VND (DLB) + LNS/ILS.
    Returns (lowest makespan, schedule)
    """
    n = workload.num_txns
    all_txns = list(range(n))

    # Parameter configuration (adaptive by workload size and effort)
    small = n <= 50
    med = 50 < n <= 90
    large = n > 90

    # Construction / insertion sampling
    CAND_SAMPLE_BASE = 12 if small else 9
    POS_SAMPLE_LIMIT = None if small else 18  # evaluate all when small; otherwise stratify positions
    JITTER = 2

    # Beam/search parameters
    beam_width = max(5, min(11, (num_seqs // 2) + (3 if small else 1)))
    cand_per_state = min(28, max(10, n // (4 if large else 3)))
    lookahead_k = 6 if (med or large) else 8
    restarts = max(2, num_seqs // 2)
    elite_size = max(3, min(6, restarts))

    # Local search parameters
    MAX_DLB_CYCLES = 2
    RELOC_TRIES = max(10, n // (3 if large else 2))
    RELOC_TRIES = int(RELOC_TRIES * max(1.0, min(2.0, 0.7 + 0.1 * max(1, num_seqs))))

    # LNS parameters
    lns_rounds_base = 2 if (med or large) else 3

    # Centralized cached evaluation of partial prefixes
    @lru_cache(maxsize=250_000)
    def _eval_cost_tuple(seq_tuple):
        return workload.get_opt_seq_cost(list(seq_tuple))

    def eval_cost(seq):
        return _eval_cost_tuple(tuple(seq))

    # Best-two insertion cache reused across a build/LNS episode
    best_two_cache = {}

    # Helpers
    def positions_stratified(seq_len, cap=POS_SAMPLE_LIMIT, extra_random=8):
        """Deterministic stratified insertion positions with optional random interiors."""
        total = seq_len + 1
        if cap is None or total <= cap:
            return list(range(total))
        anchors = {0, seq_len, seq_len // 2, seq_len // 4, (3 * seq_len) // 4}
        anchors = {p for p in anchors if 0 <= p <= seq_len}
        interior = [p for p in range(1, seq_len) if p not in anchors]
        R = min(extra_random, len(interior))
        if R > 0:
            # Sampled interiors deterministic wrt current length by seeding temporarily
            sampled = set(random.sample(interior, R))
            anchors.update(sampled)
        return sorted(anchors)

    def best_insertion_for_txn(current_seq, txn):
        """Try inserting txn; return (best_cost, best_pos, second_best_cost), using cache and stratified positions."""
        key = (tuple(current_seq), txn)
        cached = best_two_cache.get(key)
        if cached is not None:
            return cached
        seq_len = len(current_seq)
        best_cost = float('inf')
        second_best = float('inf')
        best_pos = 0
        positions = positions_stratified(seq_len)
        for pos in positions:
            cand = current_seq.copy()
            cand.insert(pos, txn)
            cost = eval_cost(cand)
            if cost < best_cost:
                second_best = best_cost
                best_cost = cost
                best_pos = pos
            elif cost < second_best:
                second_best = cost
        if second_best == float('inf'):
            second_best = best_cost
        res = (best_cost, best_pos, second_best)
        best_two_cache[key] = res
        return res

    # ------------------------ Constructors ------------------------

    def construct_grasp():
        """Regret-guided randomized insertion (GRASP seed) using best-two caching."""
        remaining = all_txns.copy()
        # Start from a good starter among sampled candidates
        k_start = min(10, len(remaining))
        start_cands = random.sample(remaining, k_start)
        start = min(start_cands, key=lambda t: eval_cost([t]))
        seq = [start]
        remaining.remove(start)

        # Strong second placement (explore positions 0/1)
        if remaining:
            k2 = min(8, len(remaining))
            pairs = []
            for t in random.sample(remaining, k2):
                for pos in [0, 1]:
                    cand = seq.copy()
                    cand.insert(pos, t)
                    c = eval_cost(cand)
                    pairs.append((c, t, pos))
            if pairs:
                pairs.sort(key=lambda x: x[0])
                rcl = pairs[:min(3, len(pairs))]
                chosen_c, chosen_t, chosen_pos = random.choice(rcl)
                seq.insert(chosen_pos, chosen_t)
                remaining.remove(chosen_t)

        # Iteratively insert with regret-based RCL; reuse best-two cache
        while remaining:
            # Adapt candidate size; endgame widen
            if len(remaining) <= max(8, 2 * CAND_SAMPLE_BASE):
                cand_txns = list(remaining)
            else:
                size = min(len(remaining), max(5, CAND_SAMPLE_BASE + random.randint(-JITTER, JITTER)))
                cand_txns = random.sample(remaining, size)

            scored = []
            for t in cand_txns:
                c_best, pos_best, c_2nd = best_insertion_for_txn(seq, t)
                regret = max(0.0, c_2nd - c_best)
                scored.append((c_best, regret, t, pos_best))

            if not scored:
                t = random.choice(remaining)
                seq.append(t)
                remaining.remove(t)
                continue

            scored.sort(key=lambda x: x[0])
            # RCL: keep top by cost; choose inside by regret-biased roulette 60% of the time
            rcl_size = min(4, len(scored))
            rcl = scored[:rcl_size]
            if random.random() < 0.6:
                regrets = [max(1e-9, r) for (_, r, _, _) in rcl]
                idx = random.choices(range(len(rcl)), weights=regrets, k=1)[0]
                c_step, _, t_step, pos_step = rcl[idx]
            else:
                # greedy among RCL
                c_step, _, t_step, pos_step = rcl[0]
            seq.insert(pos_step, t_step)
            remaining.remove(t_step)

        return eval_cost(seq), seq

    def suffix_sig(seq):
        if not seq:
            return ()
        if len(seq) == 1:
            return (seq[-1],)
        return (seq[-2], seq[-1])

    def construct_beam():
        """Adaptive regret-diverse beam with two-step lookahead and dispersion-aware scoring."""
        # Initialize beam with best singletons and one GRASP seed
        starters = [(eval_cost([t]), [t]) for t in range(n)]
        starters.sort(key=lambda x: x[0])
        init_count = min(max(beam_width * 2, beam_width + 2), n)
        beam = []
        seen = set()
        for c, seq in starters[:init_count]:
            key = tuple(seq)
            if key in seen:
                continue
            seen.add(key)
            rem = tuple(x for x in range(n) if x != seq[0])
            beam.append((c, seq, rem))

        # Inject a GRASP seed
        gc, gs = construct_grasp()
        rem = tuple(x for x in range(n) if x not in set(gs))
        beam.append((gc, gs, rem))

        best_complete = (float('inf'), [])

        for depth in range(1, n + 1):
            if not beam:
                break
            next_candidates = []  # (score, regret_proxy, c1, seq, rem, sig)

            for c_so_far, prefix, remaining in beam:
                rem_list = list(remaining)
                if not rem_list:
                    if c_so_far < best_complete[0]:
                        best_complete = (c_so_far, prefix)
                    continue

                # Candidate expansions (endgame widen)
                endgame = len(rem_list) <= 2 * beam_width
                if endgame:
                    expand_list = rem_list
                else:
                    expand_list = random.sample(rem_list, min(cand_per_state, len(rem_list)))

                # First compute raw move stats to adapt alpha locally
                raw = []  # (t, c1, best_c2, span)
                for t in expand_list:
                    new_prefix = prefix + [t]
                    c1 = eval_cost(new_prefix)
                    rem_after = [x for x in rem_list if x != t]
                    best_c2 = c1
                    span = 0.0
                    if rem_after:
                        k2 = len(rem_after) if endgame and len(rem_after) <= 10 else min(lookahead_k, len(rem_after))
                        second = rem_after if k2 == len(rem_after) else random.sample(rem_after, k2)
                        vals = []
                        best_c2 = float('inf')
                        for u in second:
                            cu = eval_cost(new_prefix + [u])
                            vals.append(cu)
                            if cu < best_c2:
                                best_c2 = cu
                        if len(vals) >= 2:
                            span = max(vals) - min(vals)
                    raw.append((t, c1, best_c2, span, tuple(rem_after), new_prefix))

                if not raw:
                    continue

                spans = sorted(r[3] for r in raw)
                median_span = spans[len(spans) // 2] if spans else 0.0

                # Build scored local moves with adaptive mixing
                local = []
                for t, c1, best_c2, span, rem_after, new_prefix in raw:
                    a = 0.5 if span > median_span else 0.8  # dispersion-aware
                    score = a * c1 + (1.0 - a) * best_c2
                    regret_proxy = max(0.0, best_c2 - c1)
                    local.append((score, regret_proxy, c1, new_prefix, rem_after))

                    if len(new_prefix) == n and c1 < best_complete[0]:
                        best_complete = (c1, new_prefix)

                local.sort(key=lambda x: x[0])
                # keep more moves for endgame
                keep_local = local[:min(6 if endgame else 4, len(local))]
                # compute regret for the best vs second best
                best_score = local[0][0]
                second_score = local[1][0] if len(local) > 1 else best_score
                regret_layer = max(0.0, second_score - best_score)
                for idx, (score, rgp, c1, seq_cand, rem_cand) in enumerate(keep_local):
                    reg = regret_layer if idx == 0 else rgp * 0.1
                    next_candidates.append((score, reg, c1, seq_cand, tuple(rem_cand), suffix_sig(seq_cand)))

            if not next_candidates:
                break

            # Primary selection by score with suffix-2 diversity
            next_candidates.sort(key=lambda x: x[0])
            pruned = []
            used_seq = set()
            seen_suffix = set()
            primary_target = max(1, int((0.6 if len(next_candidates) < 3 * beam_width else 0.7) * beam_width))
            for score, reg, c1, seq_cand, rem_cand, sig in next_candidates:
                key = tuple(seq_cand)
                if key in used_seq or sig in seen_suffix:
                    continue
                pruned.append((c1, seq_cand, rem_cand))
                used_seq.add(key)
                seen_suffix.add(sig)
                if len(pruned) >= primary_target:
                    break

            # Regret-boosted fill with diversity
            if len(pruned) < beam_width:
                rem_pool = [x for x in next_candidates if tuple(x[3]) not in used_seq]
                rem_pool.sort(key=lambda x: (-x[1], x[0]))
                for score, reg, c1, seq_cand, rem_cand, sig in rem_pool:
                    key = tuple(seq_cand)
                    if key in used_seq or sig in seen_suffix:
                        continue
                    pruned.append((c1, seq_cand, rem_cand))
                    used_seq.add(key)
                    seen_suffix.add(sig)
                    if len(pruned) >= beam_width:
                        break

            # If still short, fill purely by score
            if len(pruned) < beam_width:
                for score, reg, c1, seq_cand, rem_cand, sig in next_candidates:
                    key = tuple(seq_cand)
                    if key in used_seq:
                        continue
                    pruned.append((c1, seq_cand, rem_cand))
                    used_seq.add(key)
                    if len(pruned) >= beam_width:
                        break

            beam = pruned

        if best_complete[1] and len(best_complete[1]) == n:
            return best_complete

        # Greedy completion from best beam state if not complete
        if beam:
            c, seq, rem = min(beam, key=lambda x: x[0])
            rem_list = list(rem)
            cur_seq = list(seq)
            while rem_list:
                best_t = None
                best_c = float('inf')
                for t in rem_list:
                    c2 = eval_cost(cur_seq + [t])
                    if c2 < best_c:
                        best_c = c2
                        best_t = t
                cur_seq.append(best_t)
                rem_list.remove(best_t)
            return eval_cost(cur_seq), cur_seq

        # Fallback
        identity = list(range(n))
        return eval_cost(identity), identity

    # ------------------------ Local Search (VND with DLB) ------------------------

    def local_adjacent_swaps(seq, curr_cost, max_passes=2):
        """Multiple passes of adjacent swap hill-climbing."""
        best_seq = list(seq)
        best_cost = curr_cost
        for _ in range(max_passes):
            improved = False
            for i in range(len(best_seq) - 1):
                cand = best_seq[:]
                cand[i], cand[i + 1] = cand[i + 1], cand[i]
                c = eval_cost(cand)
                if c < best_cost:
                    best_seq, best_cost = cand, c
                    improved = True
            if not improved:
                break
        return best_seq, best_cost

    def local_oropt_dlb(seq, curr_cost, pos_sample_limit=25, cycles=MAX_DLB_CYCLES):
        """
        Or-opt blocks (sizes 1..3) with don't-look-bits. Best-improvement per accepted move.
        """
        best_seq = list(seq)
        best_cost = curr_cost
        n_local = len(best_seq)
        if n_local <= 2:
            return best_seq, best_cost

        dlb = [False] * n_local
        cycles_done = 0

        def all_positions(m):
            # evaluate all for small; else anchors + random interiors
            if POS_SAMPLE_LIMIT is None or m <= (POS_SAMPLE_LIMIT or m) + 1:
                return list(range(m))
            k = min(pos_sample_limit, m - 1)
            interior = list(range(1, m - 1))
            sampled = set(random.sample(interior, k)) if interior else set()
            sampled.update({0, m - 1})
            return sorted(sampled)

        while cycles_done < cycles:
            improved_round = False
            # larger blocks first
            for blk in (3, 2, 1):
                if blk > len(best_seq):
                    continue
                move_applied = False
                # Iterate indices; use DLB to skip cold spots
                for i in range(0, len(best_seq) - blk + 1):
                    if i < len(dlb) and dlb[i]:
                        continue
                    block = best_seq[i:i + blk]
                    base = best_seq[:i] + best_seq[i + blk:]
                    m = len(base) + 1
                    positions = all_positions(m)
                    move_best_delta = 0.0
                    move_best_pos = None
                    for pos in positions:
                        cand = base[:]
                        cand[pos:pos] = block
                        c = eval_cost(cand)
                        delta = best_cost - c
                        if delta > move_best_delta:
                            move_best_delta = delta
                            move_best_pos = pos
                    if move_best_pos is not None:
                        base[move_best_pos:move_best_pos] = block
                        best_seq = base
                        best_cost = eval_cost(best_seq)
                        improved_round = True
                        move_applied = True
                        # Reset DLB around affected area
                        dlb = [False] * len(best_seq)
                        break
                    else:
                        if i < len(dlb):
                            dlb[i] = True
                if move_applied:
                    break
            if not improved_round:
                cycles_done += 1
            else:
                cycles_done = 0  # reset cycles on improvement
        return best_seq, best_cost

    def local_pair_swaps_dlb(seq, curr_cost, tries=120):
        """
        Sampled non-adjacent pair swaps with simple DLB. Apply best improving swap per pass.
        """
        best_seq = list(seq)
        best_cost = curr_cost
        L = len(best_seq)
        if L <= 3:
            return best_seq, best_cost
        dlb = [False] * L
        rounds = 0
        max_rounds = 2
        while rounds < max_rounds:
            rounds += 1
            improved = False
            best_delta = 0.0
            best_pair = None
            attempts = min(tries, max(60, L))
            for _ in range(attempts):
                i = random.randint(0, L - 1)
                j = random.randint(0, L - 1)
                if i == j or abs(i - j) <= 1:
                    continue
                if (i < len(dlb) and dlb[i]) and (j < len(dlb) and dlb[j]):
                    continue
                cand = best_seq[:]
                cand[i], cand[j] = cand[j], cand[i]
                c = eval_cost(cand)
                delta = best_cost - c
                if delta > best_delta:
                    best_delta = delta
                    best_pair = (i, j)
            if best_pair is not None and best_delta > 0:
                i, j = best_pair
                best_seq[i], best_seq[j] = best_seq[j], best_seq[i]
                best_cost = eval_cost(best_seq)
                improved = True
                dlb = [False] * len(best_seq)
            if not improved:
                break
        return best_seq, best_cost

    def local_segment_reversals_dlb(seq, curr_cost, tries=100):
        """
        Sampled 2-opt-style segment reversals under a light DLB scheme.
        """
        best_seq = list(seq)
        best_cost = curr_cost
        L = len(best_seq)
        if L <= 4:
            return best_seq, best_cost
        dlb = [False] * L
        rounds = 0
        max_rounds = 2
        while rounds < max_rounds:
            rounds += 1
            improved = False
            best_delta = 0.0
            best_move = None
            attempts = min(tries, max(50, L))
            for _ in range(attempts):
                i = random.randint(0, L - 3)
                j = random.randint(i + 2, min(L - 1, i + 12))  # cap reversal length
                if dlb[i] and dlb[j]:
                    continue
                cand = best_seq[:]
                cand[i:j + 1] = reversed(cand[i:j + 1])
                c = eval_cost(cand)
                delta = best_cost - c
                if delta > best_delta:
                    best_delta = delta
                    best_move = (i, j)
            if best_move is not None and best_delta > 0:
                i, j = best_move
                best_seq[i:j + 1] = reversed(best_seq[i:j + 1])
                best_cost = eval_cost(best_seq)
                improved = True
                dlb = [False] * len(best_seq)
            if not improved:
                break
        return best_seq, best_cost

    def vnd_refine(seq):
        """Variable Neighborhood Descent orchestration with DLB."""
        c0 = eval_cost(seq)
        s, c = local_oropt_dlb(seq, c0, cycles=MAX_DLB_CYCLES)
        s, c = local_adjacent_swaps(s, c, max_passes=2)
        s, c = local_pair_swaps_dlb(s, c, tries=max(80, n // 2))
        s, c = local_segment_reversals_dlb(s, c, tries=max(70, n // 2))
        # Final quick Or-opt + adjacent pass
        s, c = local_oropt_dlb(s, c, cycles=1)
        s, c = local_adjacent_swaps(s, c, max_passes=1)
        return s, c

    # ------------------------ Ruin-and-Recreate LNS ------------------------

    def sensitivity_scores(seq, sample_positions_per_idx=6):
        """
        Sensitivity of each position i: move txn at i to a set of positions, compute cost spread.
        Higher spread indicates high move influence; return list of (score, i).
        """
        L = len(seq)
        base_cost = eval_cost(seq)
        scores = []
        if L <= 6:
            return [(0.0, i) for i in range(L)]
        # Candidate insertion positions sampling helper
        def pos_samples(m):
            if m + 1 <= 18:
                return list(range(m + 1))
            anchors = {0, m, m // 2, m // 4, (3 * m) // 4}
            anchors = {p for p in anchors if 0 <= p <= m}
            interior = [p for p in range(1, m) if p not in anchors]
            R = min(max(2, sample_positions_per_idx - len(anchors)), len(interior))
            if R > 0:
                anchors.update(random.sample(interior, R))
            return sorted(anchors)

        for i in range(L):
            t = seq[i]
            base = seq[:i] + seq[i + 1:]
            m = len(base)
            positions = pos_samples(m)
            # limit computations
            if len(positions) > sample_positions_per_idx:
                # pick evenly spread
                step = max(1, len(positions) // sample_positions_per_idx)
                positions = positions[::step][:sample_positions_per_idx]
            vals = []
            for pos in positions:
                cand = base[:]
                cand.insert(pos, t)
                vals.append(eval_cost(cand))
            if vals:
                score = (max(vals) - min(vals)) if len(vals) > 1 else abs(vals[0] - base_cost)
            else:
                score = 0.0
            scores.append((score, i))
        scores.sort(reverse=True)
        return scores

    def ruin_and_recreate(seq, curr_cost, rounds=2, stagnation=0):
        """
        Remove a subset (sensitivity-driven + contiguous) and rebuild with regret-guided insertion.
        Escalate removal and allow two contiguous blocks after stagnation.
        """
        best_seq = list(seq)
        best_cost = curr_cost
        if len(best_seq) <= 6:
            return best_seq, best_cost

        for r in range(rounds):
            L = len(best_seq)
            if L < 6:
                break

            # Decide removal budget
            base_frac = 0.1 if large else (0.12 if med else 0.14)
            k = max(6, min(18, int(base_frac * L)))
            if stagnation >= 2:
                k = min(L - 2, int(k * 1.25))

            # Sensitivity-guided picks
            sens = sensitivity_scores(best_seq, sample_positions_per_idx=6)
            m = min(k, len(sens))
            high_sens_count = max(1, int(0.4 * m))
            high_sens_indices = [i for _, i in sens[:high_sens_count]]

            # One or two contiguous blocks
            remove_idx = set(high_sens_indices)
            blocks = 2 if stagnation >= 2 else 1
            remaining_budget = k - len(remove_idx)
            for _ in range(blocks):
                if remaining_budget <= 0:
                    break
                block_len = max(2, min(remaining_budget, max(2, L // 15)))
                start = random.randint(0, L - block_len)
                for j in range(start, start + block_len):
                    if len(remove_idx) < k:
                        remove_idx.add(j)
                remaining_budget = k - len(remove_idx)

            # Scattered to fill
            while len(remove_idx) < k:
                remove_idx.add(random.randint(0, L - 1))
            remove_idx = sorted(remove_idx)

            removed = [best_seq[i] for i in remove_idx]
            base = [best_seq[i] for i in range(L) if i not in remove_idx]

            # Regret-guided reinsertion using shared best-two cache
            cur = list(base)
            rem = list(removed)
            random.shuffle(rem)
            while rem:
                # evaluate a subset or all when small
                k_t = len(rem) if len(rem) <= 10 else 10
                cand_txns = random.sample(rem, k_t)
                scored = []
                for t in cand_txns:
                    c_best, pos_best, c_2nd = best_insertion_for_txn(cur, t)
                    regret = max(0.0, c_2nd - c_best)
                    scored.append((c_best, regret, t, pos_best))
                scored.sort(key=lambda x: (x[0], -x[1]))
                # choose best cost; tie break by regret
                c_best, regret, t, p = scored[0]
                cur.insert(p, t)
                rem.remove(t)

            c_new = eval_cost(cur)
            if c_new < best_cost:
                best_cost = c_new
                best_seq = cur

        return best_seq, best_cost

    # ------------------------ Elite management and path relinking ------------------------

    def add_elite(pool, cost, seq, cap=elite_size):
        """Maintain a small elite pool by cost and uniqueness (exact match)."""
        if not seq or len(seq) != n:
            return pool
        key = tuple(seq)
        for i, (c, s) in enumerate(pool):
            if tuple(s) == key:
                if cost < c:
                    pool[i] = (cost, list(seq))
                return pool
        pool.append((cost, list(seq)))
        pool.sort(key=lambda x: x[0])
        if len(pool) > cap:
            pool = pool[:cap]
        return pool

    def kendall_tau_like_distance(a, b):
        """Approximate displacement: sum absolute position differences."""
        pos_b = {t: i for i, t in enumerate(b)}
        return sum(abs(i - pos_b[t]) for i, t in enumerate(a))

    def block_aware_move_toward(cur, target, pos_target):
        """
        Move either a single item or the longest consecutive block already matching target order.
        Returns new sequence after one move.
        """
        L = len(cur)
        # Identify first mismatch index i
        for i in range(L):
            desired = target[i]
            if cur[i] != desired:
                break
        else:
            return list(cur)  # already equal

        # Find desired element position j in cur
        j = cur.index(desired)

        # Try to extend to a block [j, j+k) that matches a consecutive slice in target starting at i
        max_k = 1
        while j + max_k < L and i + max_k < L and cur[j + max_k] == target[i + max_k]:
            max_k += 1

        # Move the block to position i
        block = cur[j:j + max_k]
        base = cur[:j] + cur[j + max_k:]
        new_seq = base[:i] + block + base[i:]
        return new_seq

    def path_relink_bidirectional(a_seq, b_seq, micro_polish=True):
        """Bidirectional, block-aware path relinking with micro-polish."""
        best_c = eval_cost(a_seq)
        best_s = list(a_seq)

        def relink(from_seq, to_seq):
            nonlocal best_c, best_s
            cur = list(from_seq)
            pos_target = {t: i for i, t in enumerate(to_seq)}
            # iterate up to length moves
            for _ in range(len(cur)):
                if cur == to_seq:
                    break
                nxt = block_aware_move_toward(cur, to_seq, pos_target)
                if nxt == cur:
                    break
                cur = nxt
                c = eval_cost(cur)
                if c < best_c:
                    best_c = c
                    best_s = list(cur)
                if micro_polish:
                    # one-step Or-opt(1) polish
                    s1, c1 = local_oropt_dlb(cur, c, cycles=1)
                    if c1 < best_c:
                        best_c = c1
                        best_s = list(s1)
                    cur = s1

        relink(a_seq, b_seq)
        relink(b_seq, a_seq)
        return best_c, best_s

    # ------------------------ Multi-start + ILS orchestration ------------------------

    # Elite solutions store
    elites = []  # list of (cost, seq)
    best_global = (float('inf'), [])

    # Seed strategies: several GRASP seeds plus a beam result
    seed_count = max(3, restarts - 1)
    for _ in range(seed_count):
        # reset insertion cache per construction to keep memory bounded
        best_two_cache.clear()
        c, s = construct_grasp()
        elites = add_elite(elites, c, s, cap=elite_size)
    # Add beam seed
    best_two_cache.clear()
    c_beam, s_beam = construct_beam()
    elites = add_elite(elites, c_beam, s_beam, cap=elite_size)

    # Path relinking among top elites
    if len(elites) >= 2:
        base_cost, base_seq = elites[0]
        for i in range(1, min(len(elites), elite_size)):
            _, other_seq = elites[i]
            c_rel, s_rel = path_relink_bidirectional(base_seq, other_seq, micro_polish=True)
            elites = add_elite(elites, c_rel, s_rel, cap=elite_size)

    # Refine each elite with VND, then run a couple ILS LNS rounds
    refined = []
    for c, s in elites:
        s1, c1 = vnd_refine(s)
        # LNS escape and quick polish
        best_two_cache.clear()
        s2, c2 = ruin_and_recreate(s1, c1, rounds=lns_rounds_base, stagnation=0)
        if c2 < c1:
            s1, c1 = vnd_refine(s2)
        refined.append((c1, s1))

    refined.sort(key=lambda x: x[0])
    if refined:
        best_global = refined[0]

    # Iterated Local Search loops with stagnation-aware LNS
    ils_rounds = max(2, num_seqs)
    incumbent_seq = list(best_global[1]) if best_global[1] else None
    incumbent_cost = best_global[0]
    stagnation = 0

    for k_round in range(ils_rounds):
        if incumbent_seq is None:
            break
        # Stronger LNS if stagnated twice
        rounds = lns_rounds_base + (1 if stagnation >= 2 else 0)
        best_two_cache.clear()
        s_pert, c_pert = ruin_and_recreate(incumbent_seq, incumbent_cost, rounds=rounds, stagnation=stagnation)
        if c_pert < incumbent_cost:
            stagnation = 0
            s_loc, c_loc = vnd_refine(s_pert)
            if c_loc < incumbent_cost:
                incumbent_seq, incumbent_cost = s_loc, c_loc
                if c_loc < best_global[0]:
                    best_global = (c_loc, s_loc)
        else:
            stagnation += 1
            # Small diversification: apply a few random relocations then refine
            s_rand = list(incumbent_seq)
            tries = min(12, max(6, n // 25))
            for _ in range(tries):
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)
                if i == j:
                    continue
                t = s_rand.pop(i)
                if j > i:
                    j -= 1
                s_rand.insert(j, t)
            s_loc, c_loc = vnd_refine(s_rand)
            if c_loc < incumbent_cost:
                incumbent_seq, incumbent_cost = s_loc, c_loc
                if c_loc < best_global[0]:
                    best_global = (c_loc, s_loc)

    # Safety checks
    assert best_global[1] and len(best_global[1]) == n and len(set(best_global[1])) == n
    return best_global[0], best_global[1]


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