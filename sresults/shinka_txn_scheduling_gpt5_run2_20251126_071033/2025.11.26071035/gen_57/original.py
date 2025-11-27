# EVOLVE-BLOCK-START
"""Transaction scheduling algorithm for optimizing makespan across multiple workloads
Regret-dispersion beam with episode-wide best-two insertion caching, VND, and sensitivity-guided LNS.
"""

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
    Compute a low-makespan schedule using:
      - Regret-weighted GRASP construction with reusable best-two insertion caching,
      - Regret-dispersion beam search with suffix diversity and endgame widening,
      - Strong VND local search (Or-opt, pair swaps, segment reversal),
      - Sensitivity-guided LNS destroy/repair,
      - Bidirectional block-aware path relinking among elites.

    Args:
        workload: Workload object containing transaction data
        num_seqs: Search effort parameter (controls seeds, beam width, and refinement budgets)

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    n = workload.num_txns
    all_txns = list(range(n))

    # ------------------------ Adaptive parameters ------------------------
    small = n <= 50
    med = 50 < n <= 90
    large = n > 90

    # Construction / GRASP
    STARTER_SAMPLE = min(10, n)
    CAND_SAMPLE_BASE = 12 if small else (10 if med else 9)
    RCL_K = 3
    JITTER = 2

    # Position sampling for insertion
    POS_SAMPLE_CAP = None if small else (20 if med else 18)
    STRATIFIED_R = 10 if med else (12 if small else 8)

    # Beam search
    beam_width = max(6, min(12, (num_seqs // 2) + (3 if small else 0)))
    cand_per_state = min(28, max(12, n // (4 if large else 3)))
    lookahead_k = 6 if med or large else 8
    diversity_quota = max(1, int(0.35 * beam_width))  # a bit more diversity
    endgame_widen = 2

    # Local search (VND)
    VND_MAX_ROUNDS = 2 if large else 3
    MAX_ADJ_PASSES = 2 if large else 3
    PAIR_SWAP_TRIES = max(100, n // 2)
    SEGMENT_REV_TRIES = max(100, n // 2)

    # LNS
    LNS_ROUNDS = 2 if med or large else 3
    LNS_BASE_REMOVE_FRAC = 0.08 if large else (0.1 if med else 0.12)
    LNS_REMOVE_MIN = 8 if med or large else 6
    LNS_REMOVE_MAX = 18 if med or large else 14

    # ILS
    ILS_ITERS = max(2, min(4, num_seqs))

    # Elites
    ELITE_CAP = max(4, min(6, 2 + num_seqs // 2))

    # Seed RNG per call for reproducible-but-diverse runs
    random.seed((n * 1109 + num_seqs * 911 + 13) % (2**32 - 1))

    # ------------------------ Episode-wide caches ------------------------
    class EpisodeContext:
        def __init__(self, workload, n):
            self.workload = workload
            self.n = n
            self.cost_cache = {}
            # best-two cache: (tuple(seq), txn) -> (best_cost, best_pos, second_cost)
            self.best2_cache = {}

        def eval_cost(self, seq):
            key = tuple(seq)
            c = self.cost_cache.get(key)
            if c is None:
                c = self.workload.get_opt_seq_cost(list(seq))
                self.cost_cache[key] = c
            return c

        def positions_for_insertion(self, seq_len, exhaustive=False):
            total_slots = seq_len + 1
            if exhaustive or POS_SAMPLE_CAP is None or total_slots <= POS_SAMPLE_CAP:
                return list(range(total_slots))
            # Stratified anchors: ends + quartiles + median
            anchors = {0, seq_len}
            anchors.add(seq_len // 2)
            anchors.add(seq_len // 4)
            anchors.add((3 * seq_len) // 4)
            anchors = [p for p in sorted(anchors) if 0 <= p <= seq_len]
            interior = [i for i in range(1, seq_len) if i not in anchors]
            k = max(0, min(STRATIFIED_R, len(interior)))
            sampled = random.sample(interior, k) if k > 0 else []
            pos = sorted(set(anchors).union(sampled))
            return pos

        def best_two_insertions(self, seq, txn):
            """
            Return (best_cost, best_pos, second_cost) for inserting txn into seq.
            Use exhaustive positions when the base is small or when few remain.
            """
            key = (tuple(seq), txn)
            cached = self.best2_cache.get(key)
            if cached is not None:
                return cached

            seq_len = len(seq)
            # Exhaustive when small or late stage to avoid sampling noise
            exhaustive = (seq_len <= 20)
            positions = self.positions_for_insertion(seq_len, exhaustive=exhaustive)

            best_c = float('inf')
            best_p = 0
            second_c = float('inf')
            for p in positions:
                cand = seq[:]
                cand.insert(p, txn)
                c = self.eval_cost(cand)
                if c < best_c:
                    second_c = best_c
                    best_c = c
                    best_p = p
                elif c < second_c:
                    second_c = c
            if second_c == float('inf'):
                second_c = best_c
            self.best2_cache[key] = (best_c, best_p, second_c)
            return self.best2_cache[key]

    ctx = EpisodeContext(workload, n)

    # ------------------------ Elite pool with suffix-3 diversity ------------------------
    elites = []  # list of (cost, seq)

    def elite_suffix3(seq):
        if len(seq) < 3:
            return tuple(seq)
        return tuple(seq[-3:])

    def add_elite(cost, seq):
        nonlocal elites
        if not seq or len(seq) != n:
            return
        key = tuple(seq)
        for i, (c, s) in enumerate(elites):
            if tuple(s) == key:
                if cost < c:
                    elites[i] = (cost, list(seq))
                return
        # ensure suffix-3 diversity
        sfx = elite_suffix3(seq)
        replaced = False
        for i, (c, s) in enumerate(elites):
            if elite_suffix3(s) == sfx and cost < c:
                elites[i] = (cost, list(seq))
                replaced = True
                break
        if not replaced:
            elites.append((cost, list(seq)))
        elites.sort(key=lambda x: x[0])
        if len(elites) > ELITE_CAP:
            elites = elites[:ELITE_CAP]

    # ------------------------ GRASP constructor (regret-weighted) ------------------------
    def select_starter():
        cands = random.sample(all_txns, STARTER_SAMPLE) if STARTER_SAMPLE < n else all_txns
        best_t, best_c = None, float('inf')
        for t in cands:
            c = ctx.eval_cost([t])
            if c < best_c:
                best_c, best_t = c, t
        return best_t if best_t is not None else random.randint(0, n - 1)

    def construct_grasp_regret():
        remaining = set(all_txns)
        seq = []
        s0 = select_starter()
        seq.append(s0)
        remaining.remove(s0)

        # second placement: exhaustive over a few candidates and both positions
        if remaining:
            k2 = min(8, len(remaining))
            pairs = []
            for t in random.sample(list(remaining), k2):
                for p in [0, 1]:
                    cand = seq[:]
                    cand.insert(p, t)
                    c = ctx.eval_cost(cand)
                    pairs.append((c, t, p))
            if pairs:
                pairs.sort(key=lambda x: x[0])
                rcl = pairs[:min(3, len(pairs))]
                _, t2, p2 = random.choice(rcl)
                seq.insert(p2, t2)
                remaining.remove(t2)

        while remaining:
            # choose candidate set
            if len(remaining) <= max(8, 2 * CAND_SAMPLE_BASE):
                cand_txns = list(remaining)
            else:
                sz = min(len(remaining), max(4, CAND_SAMPLE_BASE + random.randint(-JITTER, JITTER)))
                cand_txns = random.sample(list(remaining), sz)

            scored = []
            for t in cand_txns:
                c1, p1, c2 = ctx.best_two_insertions(seq, t)
                regret = max(0.0, c2 - c1)
                scored.append((c1, regret, t, p1))
            if not scored:
                # fallback
                t = random.choice(list(remaining))
                seq.append(t)
                remaining.remove(t)
                continue
            # make RCL mixing best cost and regret
            scored.sort(key=lambda x: x[0])
            rcl = scored[:min(len(scored), max(3, RCL_K))]
            if random.random() < 0.6:
                # emphasize regret among good ones
                choice = max(rcl, key=lambda x: (x[1], -x[0]))
            else:
                choice = random.choice(rcl)
            _, _, t_pick, pos_pick = choice
            seq.insert(pos_pick, t_pick)
            remaining.remove(t_pick)

        return ctx.eval_cost(seq), seq

    # ------------------------ Regret-dispersion beam search ------------------------
    def beam_search():
        # initialize with strong singletons and one GRASP seed
        starters = [(ctx.eval_cost([t]), [t]) for t in all_txns]
        starters.sort(key=lambda x: x[0])
        init = starters[:min(len(starters), max(beam_width * 2, beam_width + 2))]
        c0, s0 = construct_grasp_regret()
        init.append((c0, s0))

        beam = []
        used = set()
        for c, seq in init:
            k = tuple(seq)
            if k in used:
                continue
            used.add(k)
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
            suffix_seen = set()
            for c_so_far, prefix, rem in beam:
                if not rem:
                    if c_so_far < best_complete[0]:
                        best_complete = (c_so_far, prefix)
                    continue
                rem_list = list(rem)
                if len(rem_list) <= cand_per_state:
                    expand_list = rem_list
                else:
                    expand_list = random.sample(rem_list, cand_per_state)

                # compute immediate + lookahead with regret dispersion
                scored = []
                spans = []
                for t in expand_list:
                    new_prefix = prefix + [t]
                    c1 = ctx.eval_cost(new_prefix)
                    rem_after = [x for x in rem_list if x != t]
                    best_c2 = c1
                    second_costs = []
                    if rem_after:
                        k2 = min(lookahead_k, len(rem_after))
                        second = random.sample(rem_after, k2)
                        best_c2 = float('inf')
                        for u in second:
                            cu = ctx.eval_cost(new_prefix + [u])
                            second_costs.append(cu)
                            if cu < best_c2:
                                best_c2 = cu
                    span = (max(second_costs) - min(second_costs)) if len(second_costs) >= 2 else 0.0
                    spans.append(span)
                    scored.append((c1, best_c2, span, new_prefix, frozenset(rem_after)))

                if not scored:
                    continue

                # adaptive alpha based on median span in this state's expansions
                spans_sorted = sorted(spans)
                median_span = spans_sorted[len(spans_sorted) // 2] if spans_sorted else 0.0

                local = []
                for c1, best_c2, span, seq2, rem2 in scored:
                    alpha = 0.5 if span > median_span else 0.8
                    score = alpha * c1 + (1.0 - alpha) * best_c2
                    local.append((score, c1, span, seq2, rem2))

                # choose top by score, plus top by span (diversity on regret)
                local.sort(key=lambda x: x[0])
                top_cost = local[:max(1, min(beam_width, len(local)))]
                local_by_span = sorted(local, key=lambda x: (-x[2], x[0]))
                top_regret = local_by_span[:min(diversity_quota, len(local_by_span))]
                cand = top_cost + top_regret

                for score, c1, span, seq2, rem2 in cand:
                    key = tuple(seq2)
                    if key in layer_seen:
                        continue
                    if len(seq2) >= 3:
                        sig = (seq2[-3], seq2[-2], seq2[-1])
                    elif len(seq2) == 2:
                        sig = (None, seq2[-2], seq2[-1])
                    else:
                        sig = (None, None, seq2[-1])
                    if sig in suffix_seen:
                        continue
                    layer_seen.add(key)
                    suffix_seen.add(sig)
                    next_pool.append((c1, seq2, rem2))

            if not next_pool:
                break

            next_pool.sort(key=lambda x: x[0])
            pruned = []
            seen_prefixes = set()
            # widen in endgame if few remain
            local_bw = beam_width
            if next_pool:
                min_rem = min(len(r) for _, _, r in next_pool)
                if min_rem <= endgame_widen * beam_width:
                    local_bw = min(len(next_pool), beam_width + 2)
            for c1, seq, rem in next_pool:
                key = tuple(seq)
                if key in seen_prefixes:
                    continue
                seen_prefixes.add(key)
                pruned.append((c1, seq, rem))
                if len(pruned) >= local_bw:
                    break
            beam = pruned

        # return best complete if any
        for c, seq, rem in beam:
            if not rem and c < best_complete[0]:
                best_complete = (c, seq)
        if best_complete[1] and len(best_complete[1]) == n:
            return best_complete

        # otherwise greedy complete from best partial
        if beam:
            c, seq, rem = min(beam, key=lambda x: x[0])
            cur = list(seq)
            rem_list = list(rem)
            while rem_list:
                best_t, best_c = None, float('inf')
                for t in rem_list:
                    c2 = ctx.eval_cost(cur + [t])
                    if c2 < best_c:
                        best_c, best_t = c2, t
                cur.append(best_t)
                rem_list.remove(best_t)
            return ctx.eval_cost(cur), cur

        # fallback
        ident = list(range(n))
        return ctx.eval_cost(ident), ident

    # ------------------------ Local Search (VND) ------------------------
    def vnd_local_search(seq, start_cost, max_rounds=VND_MAX_ROUNDS):
        best_seq = list(seq)
        best_cost = start_cost

        def or_opt_pass(k):
            nonlocal best_seq, best_cost
            L = len(best_seq)
            if L <= k:
                return False
            improved = False
            dlb = [False] * (L - k + 1)  # don't look bits per start index
            i = 0
            while i <= len(best_seq) - k:
                if dlb[i]:
                    i += 1
                    continue
                block = best_seq[i:i + k]
                base = best_seq[:i] + best_seq[i + k:]
                positions = ctx.positions_for_insertion(len(base), exhaustive=(len(base) <= 20))
                move_best_c = best_cost
                move_best_p = None
                for p in positions:
                    cand = base[:]
                    cand[p:p] = block
                    c = ctx.eval_cost(cand)
                    if c < move_best_c:
                        move_best_c = c
                        move_best_p = p
                if move_best_p is not None and move_best_c + 1e-9 < best_cost:
                    new_seq = base[:]
                    new_seq[move_best_p:move_best_p] = block
                    best_seq = new_seq
                    best_cost = move_best_c
                    improved = True
                    # reset scan after improvement
                    L = len(best_seq)
                    dlb = [False] * max(1, L - k + 1)
                    i = 0
                else:
                    dlb[i] = True
                    i += 1
            return improved

        def adjacent_swap_pass(max_passes=MAX_ADJ_PASSES):
            nonlocal best_seq, best_cost
            improved_any = False
            passes = 0
            while passes < max_passes:
                improved = False
                for i in range(len(best_seq) - 1):
                    cand = best_seq[:]
                    cand[i], cand[i + 1] = cand[i + 1], cand[i]
                    c = ctx.eval_cost(cand)
                    if c < best_cost:
                        best_seq = cand
                        best_cost = c
                        improved = True
                improved_any = improved_any or improved
                if not improved:
                    break
                passes += 1
            return improved_any

        def sampled_pair_swaps(tries=PAIR_SWAP_TRIES):
            nonlocal best_seq, best_cost
            L = len(best_seq)
            if L <= 4:
                return False
            improved = False
            best_move = None
            best_delta = 0.0
            attempts = min(tries, max(50, L))
            for _ in range(attempts):
                i = random.randint(0, L - 1)
                j = random.randint(0, L - 1)
                if i == j or abs(i - j) <= 1:
                    continue
                cand = best_seq[:]
                cand[i], cand[j] = cand[j], cand[i]
                c = ctx.eval_cost(cand)
                delta = best_cost - c
                if delta > best_delta:
                    best_delta = delta
                    best_move = (i, j, c)
            if best_move is not None:
                i, j, c = best_move
                cand = best_seq[:]
                cand[i], cand[j] = cand[j], cand[i]
                best_seq = cand
                best_cost = c
                improved = True
            return improved

        def sampled_segment_reversal(tries=SEGMENT_REV_TRIES):
            nonlocal best_seq, best_cost
            L = len(best_seq)
            if L <= 5:
                return False
            improved = False
            best_move = None
            best_delta = 0.0
            attempts = min(tries, max(50, L))
            for _ in range(attempts):
                i = random.randint(0, L - 3)
                j = random.randint(i + 2, min(L - 1, i + 12))
                cand = best_seq[:]
                cand[i:j + 1] = reversed(cand[i:j + 1])
                c = ctx.eval_cost(cand)
                delta = best_cost - c
                if delta > best_delta:
                    best_delta = delta
                    best_move = (i, j, c)
            if best_move is not None:
                i, j, c = best_move
                cand = best_seq[:]
                cand[i:j + 1] = reversed(cand[i:j + 1])
                best_seq = cand
                best_cost = c
                improved = True
            return improved

        rounds = 0
        while rounds < max_rounds:
            rounds += 1
            any_improved = False
            for k in (3, 2, 1):
                if or_opt_pass(k):
                    any_improved = True
            if adjacent_swap_pass(max_passes=1):
                any_improved = True
            if sampled_pair_swaps():
                any_improved = True
            if sampled_segment_reversal():
                any_improved = True
            if not any_improved:
                break

        return best_seq, best_cost

    # ------------------------ LNS: Sensitivity-guided destroy/repair ------------------------
    def lns_ruin_and_repair(seq, start_cost, rounds=LNS_ROUNDS):
        best_seq = list(seq)
        best_cost = start_cost
        stagnation = 0

        for _ in range(max(1, rounds)):
            cur = list(best_seq)
            L = len(cur)
            if L <= 6:
                break

            # Sensitivity scoring: variance/span across sampled reinsertion positions
            K = min(20, L)
            P = 6
            sampled_idx = random.sample(range(L), K)
            sensitivity = []
            for i in sampled_idx:
                t = cur[i]
                base = cur[:i] + cur[i + 1:]
                positions = ctx.positions_for_insertion(len(base), exhaustive=False)
                # deterministic anchors + a few random slots
                pick = set()
                pick.update({0, len(base)})
                if len(base) >= 2:
                    pick.add(len(base) // 2)
                if len(base) >= 4:
                    pick.add(len(base) // 4)
                    pick.add((3 * len(base)) // 4)
                while len(pick) < P and len(pick) < len(positions):
                    pick.add(random.choice(positions))
                pos_list = sorted(pick)
                costs = []
                for p in pos_list:
                    cand = base[:]
                    cand.insert(p, t)
                    costs.append(ctx.eval_cost(cand))
                span = (max(costs) - min(costs)) if costs else 0.0
                sensitivity.append((span, i))
            sensitivity.sort(reverse=True)

            # Removal budget with stagnation escalation
            frac = LNS_BASE_REMOVE_FRAC * (1.25 if stagnation >= 2 else 1.0)
            k_remove = max(LNS_REMOVE_MIN, min(LNS_REMOVE_MAX, int(frac * L)))
            k_remove = min(k_remove, L - 2)

            remove_idx_set = set(i for _, i in sensitivity[:max(1, int(0.4 * k_remove))])

            # Add one contiguous block for the rest
            remaining_needed = k_remove - len(remove_idx_set)
            if remaining_needed > 0:
                start = random.randint(0, max(0, L - remaining_needed))
                for j in range(start, start + remaining_needed):
                    remove_idx_set.add(j)
            remove_idx = sorted(remove_idx_set)

            removed = [cur[i] for i in remove_idx]
            base = [cur[i] for i in range(L) if i not in remove_idx]

            # Regret-guided repair using best-two cache
            rebuilt = list(base)
            pool = list(removed)
            random.shuffle(pool)
            while pool:
                # All removed when small endgame or small base; else sample
                exhaustive = (len(pool) <= 2 * beam_width or len(rebuilt) <= 20)
                cand_txns = list(pool) if exhaustive else random.sample(pool, min(len(pool), max(4, CAND_SAMPLE_BASE // 2)))
                scored = []
                for t in cand_txns:
                    c1, p1, c2 = ctx.best_two_insertions(rebuilt, t)
                    scored.append((c1, c2, t, p1))
                scored.sort(key=lambda x: x[0])
                if not scored:
                    t = pool.pop()
                    rebuilt.append(t)
                    continue
                # pick by highest regret among top few by best cost
                top = scored[:min(3, len(scored))]
                top.sort(key=lambda x: (-(x[1] - x[0]), x[0]))
                c1, c2, t_pick, pos_pick = top[0]
                rebuilt.insert(pos_pick, t_pick)
                pool.remove(t_pick)

            c_new = ctx.eval_cost(rebuilt)
            if c_new + 1e-9 < best_cost:
                best_cost = c_new
                best_seq = rebuilt
                stagnation = 0
            else:
                stagnation += 1

            # quick VND polish
            best_seq, best_cost = vnd_local_search(best_seq, best_cost, max_rounds=1)

        return best_seq, best_cost

    # ------------------------ Path relinking (bidirectional, block-aware) ------------------------
    def path_relink(a_seq, b_seq, keep_intermediate=2):
        def relink_one(src, tgt):
            target_pos = {t: i for i, t in enumerate(tgt)}
            cur = list(src)
            best_c = ctx.eval_cost(cur)
            best_s = list(cur)
            candidates = []

            i = 0
            while i < len(cur):
                want = tgt[i]
                j = cur.index(want)
                if j == i:
                    i += 1
                    continue
                # Move the longest block that already matches the target order
                block_len = 1
                while j + block_len < len(cur) and i + block_len < len(tgt) and cur[j + block_len] == tgt[i + block_len]:
                    block_len += 1
                block = cur[j:j + block_len]
                base = cur[:j] + cur[j + block_len:]
                new_seq = base[:i] + block + base[i:]
                c = ctx.eval_cost(new_seq)
                cur = new_seq
                if c < best_c:
                    best_c, best_s = c, list(cur)
                candidates.append((c, list(cur)))
                i += block_len

            candidates.sort(key=lambda x: x[0])
            for c, s in candidates[:min(keep_intermediate, len(candidates))]:
                s2, c2 = vnd_local_search(s, c, max_rounds=1)
                if c2 < best_c:
                    best_c, best_s = c2, s2
            return best_c, best_s

        c1, s1 = relink_one(a_seq, b_seq)
        c2, s2 = relink_one(b_seq, a_seq)
        return (c1, s1) if c1 <= c2 else (c2, s2)

    # ------------------------ Search orchestration ------------------------
    best_cost = float('inf')
    best_seq = list(range(n))

    # 1) GRASP seeds + quick polish
    seeds = max(3, min(6, num_seqs))
    for _ in range(seeds):
        c, s = construct_grasp_regret()
        s, c = vnd_local_search(s, c, max_rounds=1)
        add_elite(c, s)
        if c < best_cost:
            best_cost, best_seq = c, s

    # 2) Beam search seeds
    beam_runs = max(1, num_seqs // 3)
    for _ in range(beam_runs):
        c, s = beam_search()
        s, c = vnd_local_search(s, c, max_rounds=1)
        add_elite(c, s)
        if c < best_cost:
            best_cost, best_seq = c, s

    # 3) Path relinking among top elites
    if len(elites) >= 2:
        base_cost, base_seq = elites[0]
        for i in range(1, min(len(elites), ELITE_CAP)):
            _, seq2 = elites[i]
            c_rel, s_rel = path_relink(base_seq, seq2, keep_intermediate=2)
            s_rel, c_rel = vnd_local_search(s_rel, c_rel, max_rounds=1)
            add_elite(c_rel, s_rel)
            if c_rel < best_cost:
                best_cost, best_seq = c_rel, s_rel

    # 4) Iterated LNS + VND from incumbent or elites
    incumbent_cost, incumbent_seq = best_cost, best_seq
    for _ in range(ILS_ITERS):
        if elites and random.random() < 0.5:
            start_c, start_s = random.choice(elites[:min(3, len(elites))])
        else:
            start_c, start_s = incumbent_cost, incumbent_seq

        s1, c1 = lns_ruin_and_repair(start_s, start_c, rounds=LNS_ROUNDS)
        s2, c2 = vnd_local_search(s1, c1, max_rounds=VND_MAX_ROUNDS)
        if c2 < incumbent_cost:
            incumbent_cost, incumbent_seq = c2, s2
            add_elite(c2, s2)
        # quick adjacent polish
        s3, c3 = vnd_local_search(incumbent_seq, incumbent_cost, max_rounds=1)
        if c3 < incumbent_cost:
            incumbent_cost, incumbent_seq = c3, s3
            add_elite(c3, s3)

    best_cost, best_seq = incumbent_cost, incumbent_seq

    # Safety checks
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