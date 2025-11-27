# EVOLVE-BLOCK-START
"""Transaction scheduling algorithm for optimizing makespan across multiple workloads
Planner-based redesign with deterministic best-two memo, regret-diverse beam, VND (DLB), and sensitivity LNS.
"""

import time
import random
import sys
import os
from math import inf, sqrt

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
      - Deterministic position-sampled regret-insertion (GRASP) with reusable best-two memoization,
      - Insertion-based dispersion-adaptive beam with suffix diversity and endgame widening,
      - Strong VND (Or-opt, non-adjacent swaps, sampled 2-opt under DLB),
      - Sensitivity-guided LNS with stagnation escalation and quick polish,
      - Bidirectional, block-aware path relinking among elites.

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
    POS_SAMPLE_CAP = None if small else (22 if med else 18)
    RING_RADIUS = 3
    EXHAUSTIVE_THRESHOLD = 20  # all positions when seq_len <= this

    # Beam search
    beam_width = max(6, min(12, (num_seqs // 2) + (3 if small else 1)))
    cand_per_state = min(30, max(12, n // (4 if large else 3)))
    lookahead_k = 6 if (med or large) else 8
    diversity_quota_ratio = 0.35
    endgame_widen_by = 2
    suffix_div_k_early = 2
    suffix_div_k_late = 3

    # Local search (VND)
    VND_MAX_ROUNDS = 2 if large else 3
    MAX_ADJ_PASSES = 2 if large else 3
    PAIR_SWAP_TRIES = min(220, max(80, n))
    TWO_OPT_TRIES = min(220, max(80, n))

    # LNS
    LNS_ROUNDS = 2 if med or large else 3
    LNS_REMOVE_FRAC = 0.08 if large else (0.11 if med else 0.13)
    LNS_REMOVE_MIN = 8 if med or large else 6
    LNS_REMOVE_MAX = 20 if med or large else 16

    # ILS
    ILS_ITERS = max(2, min(5, num_seqs))

    # Elites
    ELITE_CAP = max(4, min(6, 2 + num_seqs // 2))

    # Endgame exact enumeration (branch-and-bound)
    BNB_K = 9
    BNB_TIME_BUDGET = 0.35

    # Seed RNG per call for reproducible-but-diverse runs
    random.seed((n * 11939 + num_seqs * 911 + 29) % (2**32 - 1))

    # ------------------------ Episode context with shared caches ------------------------
    class EpisodeContext:
        def __init__(self, workload, n):
            self.workload = workload
            self.n = n
            self.cost_cache = {}  # tuple(seq) -> cost
            # best-two cache: (tuple(seq), txn, pos_sig) -> (best_cost, best_pos, second_cost)
            # Special 'all' signature used when evaluating all positions
            self.best2_cache = {}

        def eval_cost(self, seq):
            key = tuple(seq)
            c = self.cost_cache.get(key)
            if c is None:
                c = self.workload.get_opt_seq_cost(list(seq))
                self.cost_cache[key] = c
            return c

        def pos_samples(self, seq_len, cap=POS_SAMPLE_CAP, focus_idx=None, ring=RING_RADIUS, exhaustive=False):
            """
            Deterministic stratified positions:
              - Always include ends, quartiles, and median.
              - Optionally include a tight ring around focus_idx (±ring).
              - Fill interiors with a low-discrepancy sequence based on golden ratio.
            """
            total_slots = seq_len + 1
            if exhaustive or cap is None or total_slots <= cap:
                pos = list(range(total_slots))
                return pos, ('all', seq_len)  # signature for caching

            anchors = {0, seq_len, seq_len // 2, seq_len // 4, (3 * seq_len) // 4}
            if focus_idx is not None:
                for d in range(-ring, ring + 1):
                    p = focus_idx + d
                    if 0 <= p <= seq_len:
                        anchors.add(p)
            anchors = [p for p in sorted(anchors) if 0 <= p <= seq_len]
            # Low-discrepancy interiors
            interiors = [i for i in range(1, seq_len) if i not in anchors]
            need = max(0, min(cap - len(anchors), len(interiors)))
            phi = (1 + 5**0.5) / 2.0
            step = phi - 1  # ~0.618
            picks = []
            x = 0.0
            for _ in range(need):
                x = (x + step) % 1.0
                idx = int(x * len(interiors))
                picks.append(interiors[idx])
            pos = sorted(set(anchors).union(picks))
            pos_sig = tuple(pos)  # signature used in cache key
            return pos, pos_sig

        def best_two_insertions(self, seq, txn, focus_idx=None, force_all=False):
            """
            Return (best_cost, best_pos, second_cost) for inserting txn into seq.
            Use deterministic positions; reuse results across modules via (seq, txn, pos_sig) cache.
            Prefer 'all' signature if already computed.
            """
            seq_len = len(seq)
            exhaustive = force_all or (seq_len <= EXHAUSTIVE_THRESHOLD)
            # Prefer 'all' if it exists
            all_sig = ('all', seq_len)
            all_key = (tuple(seq), txn, all_sig)
            if all_key in self.best2_cache:
                return self.best2_cache[all_key]

            positions, pos_sig = self.pos_samples(seq_len, focus_idx=focus_idx, exhaustive=exhaustive)
            key = (tuple(seq), txn, pos_sig)
            cached = self.best2_cache.get(key)
            if cached is not None:
                return cached

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
            res = (best_c, best_p, second_c)
            self.best2_cache[key] = res
            # If exhaustive (all positions) were evaluated, also store under 'all'
            if pos_sig == all_sig:
                self.best2_cache[all_key] = res
            return res

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
        sfx = elite_suffix3(seq)
        # replace exact duplicates if better
        for i, (c, s) in enumerate(elites):
            if tuple(s) == key:
                if cost < c:
                    elites[i] = (cost, list(seq))
                elites.sort(key=lambda x: x[0])
                if len(elites) > ELITE_CAP:
                    elites = elites[:ELITE_CAP]
                return
        # enforce suffix-3 diversity
        for i, (c, s) in enumerate(elites):
            if elite_suffix3(s) == sfx and cost < c:
                elites[i] = (cost, list(seq))
                elites.sort(key=lambda x: x[0])
                if len(elites) > ELITE_CAP:
                    elites = elites[:ELITE_CAP]
                return
        elites.append((cost, list(seq)))
        elites.sort(key=lambda x: x[0])
        if len(elites) > ELITE_CAP:
            elites = elites[:ELITE_CAP]

    # ------------------------ Constructors ------------------------
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
        seq = [select_starter()]
        remaining.remove(seq[0])

        # strong second (positions 0/1)
        if remaining:
            k2 = min(8, len(remaining))
            pairs = []
            for t in random.sample(list(remaining), k2):
                for pos in [0, 1]:
                    cand = seq[:]
                    cand.insert(pos, t)
                    c = ctx.eval_cost(cand)
                    pairs.append((c, t, pos))
            if pairs:
                pairs.sort(key=lambda x: x[0])
                c2, t2, p2 = random.choice(pairs[:min(3, len(pairs))])
                seq.insert(p2, t2)
                remaining.remove(t2)

        while remaining:
            # candidate transaction sample
            size = len(remaining)
            cand_size = size if size <= (2 * CAND_SAMPLE_BASE) else min(size, max(5, CAND_SAMPLE_BASE + random.randint(-JITTER, JITTER)))
            cand_txns = random.sample(list(remaining), cand_size)

            scored = []
            force_all = (len(seq) <= EXHAUSTIVE_THRESHOLD) or (len(remaining) <= 2 * beam_width)
            for t in cand_txns:
                c1, p1, c2 = ctx.best_two_insertions(seq, t, force_all=force_all)
                scored.append((c1, max(0.0, c2 - c1), t, p1))

            scored.sort(key=lambda x: x[0])
            rcl = scored[:min(max(3, RCL_K), len(scored))]
            # favor high regret among good options with probability 0.6
            if random.random() < 0.6:
                rcl.sort(key=lambda x: (-x[1], x[0]))
            chosen = rcl[0] if random.random() < 0.6 else random.choice(rcl)
            _, _, t_star, p_star = chosen
            seq.insert(p_star, t_star)
            remaining.remove(t_star)

        return ctx.eval_cost(seq), seq

    # ------------------------ Beam search with insertion-based expansions ------------------------
    # Endgame exact completion via bounded branch-and-bound
    def bnb_complete(prefix, remaining, best_bound=float('inf')):
        start_time = time.time()
        best_cost = best_bound
        best_seq = None
        rem_list = list(remaining)

        # small transposition keyed by (prefix suffix, sorted remaining)
        memo = {}

        def dfs(cur, rem):
            nonlocal best_cost, best_seq
            # Time pruning
            if time.time() - start_time > BNB_TIME_BUDGET:
                return
            if not rem:
                c = ctx.eval_cost(cur)
                if c < best_cost:
                    best_cost = c
                    best_seq = list(cur)
                return
            # Prefix cost bound
            c_pref = ctx.eval_cost(cur)
            if c_pref >= best_cost:
                return
            key = (tuple(cur[-3:]), tuple(sorted(rem)))
            prev = memo.get(key)
            # Prune if we've already reached this subproblem with <= cost
            if prev is not None and c_pref >= prev - 1e-12:
                return
            memo[key] = c_pref
            # Order by immediate extension cost for faster pruning
            ordered = sorted(rem, key=lambda t: ctx.eval_cost(cur + [t]))
            for t in ordered:
                c1 = ctx.eval_cost(cur + [t])
                if c1 >= best_cost:
                    continue
                nxt = list(cur)
                nxt.append(t)
                rem_next = [x for x in rem if x != t]
                dfs(nxt, rem_next)
                if time.time() - start_time > BNB_TIME_BUDGET:
                    break

        dfs(list(prefix), rem_list)
        if best_seq is None:
            # Fallback: greedy best-insertion completion
            cur = list(prefix)
            rem = list(remaining)
            while rem:
                best_t, best_pos, best_c = None, 0, float('inf')
                for t in rem:
                    c2, p2, _ = ctx.best_two_insertions(cur, t, force_all=(len(cur) <= EXHAUSTIVE_THRESHOLD))
                    if c2 < best_c:
                        best_c, best_t, best_pos = c2, t, p2
                cur.insert(best_pos, best_t)
                rem.remove(best_t)
            return ctx.eval_cost(cur), cur
        return best_cost, best_seq

    def beam_search():
        bw = beam_width
        diversity_quota = max(1, int(diversity_quota_ratio * bw))

        # initialize with top singletons and one GRASP seed
        starters = [(ctx.eval_cost([t]), [t]) for t in all_txns]
        starters.sort(key=lambda x: x[0])
        init_pool = starters[:min(len(starters), max(bw * 2, bw + 2))]
        c0, s0 = construct_grasp_regret()
        init_pool.append((c0, s0))

        beam = []
        used = set()
        for c, seq in init_pool:
            k = tuple(seq)
            if k in used:
                continue
            used.add(k)
            rem = frozenset(t for t in all_txns if t not in seq)
            beam.append((c, seq, rem))
            if len(beam) >= bw:
                break

        best_complete = (float('inf'), [])

        depth = 1
        while beam and depth <= n:
            next_pool = []
            layer_seen = set()
            # Determine endgame and suffix diversity length
            min_remaining = min((len(rem) for (_, _, rem) in beam), default=n)
            endgame = min_remaining <= 2 * bw
            suffix_k = suffix_div_k_late if endgame else suffix_div_k_early
            suffix_seen = set()

            for c_so_far, prefix, rem in beam:
                if not rem:
                    if c_so_far < best_complete[0]:
                        best_complete = (c_so_far, prefix)
                    continue

                # Endgame exact enumeration if remaining is small
                if len(rem) <= BNB_K:
                    c_b, s_b = bnb_complete(prefix, list(rem), best_complete[0])
                    if c_b < best_complete[0]:
                        best_complete = (c_b, s_b)
                    # Push completed sequence to next layer for uniform handling
                    next_pool.append((c_b, s_b, frozenset()))
                    continue

                rem_list = list(rem)
                expand_list = rem_list if len(rem_list) <= cand_per_state else random.sample(rem_list, cand_per_state)

                scored_local = []
                spans = []
                for t in expand_list:
                    # Best insertion of t into current prefix (not just append)
                    c1, p1, c1_second = ctx.best_two_insertions(prefix, t, force_all=(len(prefix) <= EXHAUSTIVE_THRESHOLD))
                    seq1 = prefix[:]
                    seq1.insert(p1, t)
                    # Lookahead: best-of-k insertion for the next item
                    rem_after = [x for x in rem_list if x != t]
                    best_c2 = c1
                    second_costs = []
                    if rem_after:
                        k2 = len(rem_after) if (endgame and len(rem_after) <= 6) else min(lookahead_k, len(rem_after))
                        second = rem_after if k2 == len(rem_after) else random.sample(rem_after, k2)
                        best_c2 = float('inf')
                        for u in second:
                            cu, pu, cu2 = ctx.best_two_insertions(seq1, u, force_all=(len(seq1) <= EXHAUSTIVE_THRESHOLD))
                            second_costs.append(cu)
                            if cu < best_c2:
                                best_c2 = cu
                    span = (max(second_costs) - min(second_costs)) if len(second_costs) >= 2 else 0.0
                    spans.append(span)
                    scored_local.append((c1, best_c2, span, seq1, frozenset(rem_after)))

                if not scored_local:
                    continue

                med_span = sorted(spans)[len(spans) // 2] if spans else 0.0
                ranked = []
                for c1, c2, span, seq2, rem2 in scored_local:
                    alpha = 0.5 if span > med_span else 0.8
                    score = alpha * c1 + (1.0 - alpha) * c2
                    regret_signal = max(0.0, c2 - c1)
                    ranked.append((score, regret_signal, c1, seq2, rem2))

                ranked.sort(key=lambda x: x[0])
                top_cost = ranked[:max(1, min(bw, len(ranked)))]
                top_regret = sorted(ranked, key=lambda x: (-x[1], x[0]))[:min(diversity_quota, len(ranked))]

                for score, rg, c1, seq2, rem2 in top_cost + top_regret:
                    key = tuple(seq2)
                    if key in layer_seen:
                        continue
                    # suffix-k diversity
                    if suffix_k >= 1:
                        sig = tuple(seq2[-suffix_k:]) if len(seq2) >= suffix_k else tuple(seq2)
                        if sig in suffix_seen:
                            continue
                        suffix_seen.add(sig)
                    layer_seen.add(key)
                    next_pool.append((c1, seq2, rem2))

            if not next_pool:
                break

            next_pool.sort(key=lambda x: x[0])
            pruned = []
            seen_prefixes = set()

            local_bw = bw
            min_rem = min((len(r) for (_, _, r) in next_pool), default=n)
            if min_rem <= endgame_widen_by * bw:
                local_bw = min(len(next_pool), bw + endgame_widen_by)

            for c1, seq, rem in next_pool:
                key = tuple(seq)
                if key in seen_prefixes:
                    continue
                seen_prefixes.add(key)
                pruned.append((c1, seq, rem))
                if len(pruned) >= local_bw:
                    break

            beam = pruned
            depth += 1

        # return best complete if any
        for c, seq, rem in beam:
            if not rem and c < best_complete[0]:
                best_complete = (c, seq)
        if best_complete[1] and len(best_complete[1]) == n:
            return best_complete

        # otherwise greedy complete from best partial using best-insertion or BnB if small
        if beam:
            c, seq, rem = min(beam, key=lambda x: x[0])
            if len(rem) <= BNB_K:
                return bnb_complete(seq, list(rem), float('inf'))
            cur = list(seq)
            rem_list = list(rem)
            while rem_list:
                best_t, best_pos, best_c = None, 0, float('inf')
                for t in rem_list:
                    c2, p2, _ = ctx.best_two_insertions(cur, t, force_all=(len(cur) <= EXHAUSTIVE_THRESHOLD))
                    if c2 < best_c:
                        best_c = c2
                        best_t = t
                        best_pos = p2
                cur.insert(best_pos, best_t)
                rem_list.remove(best_t)
            return ctx.eval_cost(cur), cur

        # fallback
        ident = list(range(n))
        return ctx.eval_cost(ident), ident

    # ------------------------ Local Search (VND with DLB) ------------------------
    def or_opt_pass(seq, start_cost, k, pos_cap=POS_SAMPLE_CAP):
        best_seq = list(seq)
        best_cost = start_cost
        L = len(best_seq)
        if L <= k:
            return best_seq, best_cost, False
        improved = False
        i = 0
        while i <= len(best_seq) - k:
            block = best_seq[i:i + k]
            base = best_seq[:i] + best_seq[i + k:]
            m = len(base)
            # Use deterministic positions (reuse pos signature)
            positions, _sig = ctx.pos_samples(m, cap=pos_cap, focus_idx=i, ring=2, exhaustive=(m + 1 <= EXHAUSTIVE_THRESHOLD))
            move_best_c = best_cost
            move_best_p = None
            for p in positions:
                cand = base[:]
                cand[p:p] = block
                c = ctx.eval_cost(cand)
                if c < move_best_c:
                    move_best_c = c
                    move_best_p = p
            if move_best_p is not None and move_best_c + 1e-12 < best_cost:
                new_seq = base[:]
                new_seq[move_best_p:move_best_p] = block
                best_seq = new_seq
                best_cost = move_best_c
                improved = True
                i = 0  # restart scan after structural change
            else:
                i += 1
        return best_seq, best_cost, improved

    def adjacent_swaps_pass(seq, start_cost, max_passes=MAX_ADJ_PASSES):
        best_seq = list(seq)
        best_cost = start_cost
        improved_any = False
        for _ in range(max_passes):
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
        return best_seq, best_cost, improved_any

    def sampled_pair_swaps_dlb(seq, start_cost, tries=PAIR_SWAP_TRIES):
        best_seq = list(seq)
        best_cost = start_cost
        L = len(best_seq)
        if L <= 3:
            return best_seq, best_cost, False
        improved = False
        best_move = None
        best_delta = 0.0
        attempts = min(tries, max(60, L))
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
        return best_seq, best_cost, improved

    def sampled_two_opt_reversal(seq, start_cost, tries=TWO_OPT_TRIES):
        best_seq = list(seq)
        best_cost = start_cost
        L = len(best_seq)
        if L <= 5:
            return best_seq, best_cost, False
        improved = False
        best_move = None
        best_delta = 0.0
        attempts = min(tries, max(60, L))
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
        return best_seq, best_cost, improved

    def vnd_local_search(seq, start_cost, max_rounds=VND_MAX_ROUNDS):
        best_seq = list(seq)
        best_cost = start_cost
        rounds = 0
        while rounds < max_rounds:
            rounds += 1
            any_imp = False
            for k in (3, 2, 1):
                s, c, imp = or_opt_pass(best_seq, best_cost, k)
                if imp and c < best_cost:
                    best_seq, best_cost = s, c
                    any_imp = True
            s, c, imp = adjacent_swaps_pass(best_seq, best_cost, max_passes=1)
            if imp and c < best_cost:
                best_seq, best_cost = s, c
                any_imp = True
            s, c, imp = sampled_pair_swaps_dlb(best_seq, best_cost)
            if imp and c < best_cost:
                best_seq, best_cost = s, c
                any_imp = True
            s, c, imp = sampled_two_opt_reversal(best_seq, best_cost)
            if imp and c < best_cost:
                best_seq, best_cost = s, c
                any_imp = True
            if not any_imp:
                break
        return best_seq, best_cost

    # ------------------------ LNS: Sensitivity-guided destroy/repair ------------------------
    def sensitivity_rank(seq, P=6):
        L = len(seq)
        scores = []
        if L <= 6:
            return [(0.0, i) for i in range(L)]
        for i in range(L):
            t = seq[i]
            base = seq[:i] + seq[i + 1:]
            m = len(base)
            positions, _sig = ctx.pos_samples(m, cap=12, focus_idx=min(i, m), ring=2, exhaustive=(m + 1 <= 12))
            # limit to P positions, take spread-out subset
            if len(positions) > P:
                step = max(1, len(positions) // P)
                positions = positions[::step][:P]
            vals = []
            for p in positions:
                cand = base[:]
                cand.insert(p, t)
                vals.append(ctx.eval_cost(cand))
            if len(vals) >= 2:
                s = max(vals) - min(vals)
            else:
                s = 0.0
            scores.append((s, i))
        scores.sort(reverse=True)
        return scores

    def lns_ruin_and_repair(seq, start_cost, rounds=LNS_ROUNDS):
        best_seq = list(seq)
        best_cost = start_cost
        stagnation = 0

        for _ in range(max(1, rounds)):
            cur = list(best_seq)
            L = len(cur)
            if L <= 6:
                break

            # Decide removal size with escalation
            factor = 1.0 + (0.25 * (stagnation // 2))
            m = int(min(LNS_REMOVE_MAX, max(LNS_REMOVE_MIN, LNS_REMOVE_FRAC * L * factor)))
            m = min(m, L - 2)

            sens = sensitivity_rank(cur, P=6)
            high_sens = max(1, int(0.4 * m))
            remove_idx = set(i for _, i in sens[:high_sens])

            # Add contiguous block(s)
            remaining = m - len(remove_idx)
            blocks = 2 if stagnation >= 2 else 1
            for _b in range(blocks):
                if remaining <= 0:
                    break
                blk_len = max(2, min(remaining, max(2, L // 12)))
                start = random.randint(0, max(0, L - blk_len))
                for j in range(start, start + blk_len):
                    remove_idx.add(j)
                    if len(remove_idx) >= m:
                        break
                remaining = m - len(remove_idx)

            # Fill if short
            while len(remove_idx) < m:
                remove_idx.add(random.randint(0, L - 1))

            remove_idx = sorted(remove_idx)
            removed = [cur[i] for i in remove_idx]
            base = [cur[i] for i in range(L) if i not in remove_idx]

            # Regret-guided repair using shared best-two cache
            rebuilt = list(base)
            pool = list(removed)
            # Use original positions as focus targets to reduce disruption
            orig_pos = {cur[i]: i for i in remove_idx}
            # deterministic reinsertion order: by sensitivity (higher first)
            pool_sens = {i: s for s, i in sens}
            pool.sort(key=lambda t: -pool_sens.get(orig_pos.get(t, 0), 0.0))
            while pool:
                # evaluate all when small; else subset
                exhaustive = (len(rebuilt) <= EXHAUSTIVE_THRESHOLD) or (len(pool) <= 2 * beam_width)
                cand_txns = pool if len(pool) <= 10 else random.sample(pool, 10)
                scored = []
                for t in cand_txns:
                    focus = min(orig_pos.get(t, len(rebuilt)), len(rebuilt))
                    c1, p1, c2 = ctx.best_two_insertions(rebuilt, t, focus_idx=focus, force_all=exhaustive)
                    scored.append((c1, c2, t, p1))
                if not scored:
                    t = pool.pop()
                    rebuilt.append(t)
                    continue
                # choose highest regret among top-3 by best cost
                scored.sort(key=lambda x: x[0])
                top = scored[:min(3, len(scored))]
                top.sort(key=lambda x: (-(x[1] - x[0]), x[0]))
                c1, c2, t_pick, p_pick = top[0]
                rebuilt.insert(p_pick, t_pick)
                pool.remove(t_pick)

            c_new = ctx.eval_cost(rebuilt)
            if c_new + 1e-12 < best_cost:
                best_cost = c_new
                best_seq = rebuilt
                stagnation = 0
                # quick micro-polish
                best_seq, best_cost = or_opt_pass(best_seq, best_cost, 1)[0:2]
            else:
                stagnation += 1

        return best_seq, best_cost

    # ------------------------ Path relinking (bidirectional, block-aware) ------------------------
    def path_relink(a_seq, b_seq, micro_polish=True):
        def relink_one(src, tgt):
            cur = list(src)
            best_c = ctx.eval_cost(cur)
            best_s = list(cur)
            i = 0
            while i < len(cur):
                want = tgt[i]
                j = cur.index(want)
                if j == i:
                    i += 1
                    continue
                # move longest block already matching the target order starting at i
                k = 1
                while j + k < len(cur) and i + k < len(tgt) and cur[j + k] == tgt[i + k]:
                    k += 1
                block = cur[j:j + k]
                base = cur[:j] + cur[j + k:]
                new_seq = base[:i] + block + base[i:]
                c = ctx.eval_cost(new_seq)
                cur = new_seq
                if c < best_c:
                    best_c, best_s = c, list(cur)
                if micro_polish:
                    s1, c1, _ = or_opt_pass(cur, c, 1)
                    if c1 < best_c:
                        best_c, best_s = c1, list(s1)
                    cur = s1
                i += k
            return best_c, best_s

        c1, s1 = relink_one(a_seq, b_seq)
        c2, s2 = relink_one(b_seq, a_seq)
        return (c1, s1) if c1 <= c2 else (c2, s2)

    # ------------------------ Final exhaustive 1-block reinsertion polish ------------------------
    def full_or_opt1_polish(seq, start_cost, passes=1):
        best_seq = list(seq)
        best_cost = start_cost
        L = len(best_seq)
        if L <= 2:
            return best_seq, best_cost
        for _ in range(max(1, passes)):
            improved = False
            move_best_cost = best_cost
            move_apply = None
            for i in range(L):
                t = best_seq[i]
                base = best_seq[:i] + best_seq[i + 1:]
                for pos in range(len(base) + 1):
                    if pos == i:
                        continue
                    cand = base[:]
                    cand.insert(pos, t)
                    c = ctx.eval_cost(cand)
                    if c < move_best_cost:
                        move_best_cost = c
                        move_apply = (i, pos, cand)
            if move_apply is not None and move_best_cost + 1e-12 < best_cost:
                _, _, new_seq = move_apply
                best_seq = new_seq
                best_cost = move_best_cost
                L = len(best_seq)
                improved = True
            if not improved:
                break
        return best_seq, best_cost

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

    # 2) Beam search seeds (multiple runs for diversity)
    beam_runs = max(1, num_seqs // 3)
    for _ in range(beam_runs):
        c, s = beam_search()
        s, c = vnd_local_search(s, c, max_rounds=1)
        add_elite(c, s)
        if c < best_cost:
            best_cost, best_seq = c, s

    # 3) Path relinking among elites
    if len(elites) >= 2:
        base_cost, base_seq = elites[0]
        for i in range(1, min(len(elites), ELITE_CAP)):
            _, other = elites[i]
            c_rel, s_rel = path_relink(base_seq, other, micro_polish=True)
            s_rel, c_rel = vnd_local_search(s_rel, c_rel, max_rounds=1)
            add_elite(c_rel, s_rel)
            if c_rel < best_cost:
                best_cost, best_seq = c_rel, s_rel

    # 4) LNS + VND (iterated)
    incumbent_cost, incumbent_seq = best_cost, best_seq
    for _ in range(ILS_ITERS):
        s1, c1 = lns_ruin_and_repair(incumbent_seq, incumbent_cost, rounds=LNS_ROUNDS)
        s2, c2 = vnd_local_search(s1, c1, max_rounds=VND_MAX_ROUNDS)
        if c2 < incumbent_cost:
            incumbent_cost, incumbent_seq = c2, s2
            add_elite(c2, s2)
        # quick polish
        s3, c3 = vnd_local_search(incumbent_seq, incumbent_cost, max_rounds=1)
        if c3 < incumbent_cost:
            incumbent_cost, incumbent_seq = c3, s3
            add_elite(c3, s3)

    best_cost, best_seq = incumbent_cost, incumbent_seq
    # Final exhaustive 1-block polish for last gains
    best_seq, best_cost = full_or_opt1_polish(best_seq, best_cost, passes=1)

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