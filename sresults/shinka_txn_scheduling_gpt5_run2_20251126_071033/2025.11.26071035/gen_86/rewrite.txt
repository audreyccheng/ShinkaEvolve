# EVOLVE-BLOCK-START
"""Transaction scheduling algorithm for optimizing makespan across multiple workloads
Unified best-two LRU memo + regret-adaptive beam + bounded BnB endgame + VND Or-opt (block-aware) + sensitivity+marginal LNS + bidirectional relinking.
"""

import time
import random
import sys
import os
from collections import OrderedDict, defaultdict, deque

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
    Hybrid constructor + regret-adaptive beam + bounded endgame BnB + VND + LNS + relinking.

    Args:
        workload: Workload object containing transaction data
        num_seqs: Search effort parameter (controls seeds, beam width, and refinement budgets)

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    n = workload.num_txns
    all_txns = list(range(n))

    # ---------------- Adaptive parameters ----------------
    small = n <= 50
    med = 50 < n <= 90
    large = n > 90

    # Construction / GRASP
    STARTER_SAMPLE = min(10, n)
    CAND_SAMPLE_BASE = 12 if small else (10 if med else 9)
    RCL_K = 3
    JITTER = 2

    # Positions sampling + memoization signature
    POS_SAMPLE_CAP = None if small else (22 if med else 18)
    RING_SPAN = 3
    EXHAUSTIVE_THRESHOLD = 20  # evaluate all positions when current seq length <= this

    # Beam search
    beam_width = max(6, min(12, (num_seqs // 2) + (3 if small else 1)))
    cand_per_state = min(30, max(12, n // (4 if large else 3)))
    lookahead_k = 6 if (med or large) else 8
    diversity_quota_ratio = 0.35
    endgame_widen_by = 2

    # Endgame bounded BnB
    BNB_K = 9  # enumerate when remaining <= K
    BNB_NODE_CAP = 3000
    TT_CAP = 20000  # transposition table bound cache

    # Local search (VND)
    VND_MAX_ROUNDS = 2 if large else 3
    MAX_ADJ_PASSES = 2 if large else 3
    PAIR_SWAP_TRIES = min(220, max(100, n))
    TWO_OPT_TRIES = min(220, max(100, n))

    # LNS
    LNS_ROUNDS = 2 if med or large else 3
    LNS_REMOVE_FRAC = 0.08 if large else (0.11 if med else 0.13)
    LNS_REMOVE_MIN = 8 if med or large else 6
    LNS_REMOVE_MAX = 20 if med or large else 16

    # Iterations
    ILS_ITERS = max(2, min(5, num_seqs))

    # Elites
    ELITE_CAP = max(4, min(6, 2 + num_seqs // 2))

    # Seed RNG per call for reproducible-but-diverse runs
    random.seed((n * 1315423911 + num_seqs * 2654435761 + 37) % (2**32 - 1))

    # ---------------- Episode context with shared caches ----------------
    class EpisodeContext:
        def __init__(self, workload, n):
            self.workload = workload
            self.n = n
            self.cost_cache = {}  # tuple(seq) -> cost
            # Unified best-two: key = (tuple(seq), kind, obj, pos_sig)
            # kind is 'txn' or 'block'; obj is txn int or tuple(block)
            # Value = (best_cost, best_pos, second_cost)
            self.best_two_cache = OrderedDict()
            self.best_two_cap = 40000  # LRU capacity
            # Deterministic position memo: signature -> list of positions
            self.pos_sig_cache = {}

        def eval_cost(self, seq):
            key = tuple(seq)
            c = self.cost_cache.get(key)
            if c is None:
                c = self.workload.get_opt_seq_cost(list(seq))
                self.cost_cache[key] = c
            return c

        def _lru_put(self, key, value):
            self.best_two_cache[key] = value
            self.best_two_cache.move_to_end(key, last=True)
            if len(self.best_two_cache) > self.best_two_cap:
                self.best_two_cache.popitem(last=False)

        def positions_signature(self, seq_len, exhaustive=False, focus_idx=None):
            total = seq_len + 1
            if exhaustive or POS_SAMPLE_CAP is None or total <= POS_SAMPLE_CAP:
                return ('all', seq_len)
            anchors = {0, seq_len, seq_len // 2, seq_len // 4, (3 * seq_len) // 4}
            if focus_idx is not None:
                for d in range(-RING_SPAN, RING_SPAN + 1):
                    p = focus_idx + d
                    if 0 <= p <= seq_len:
                        anchors.add(p)
            anchors = {p for p in anchors if 0 <= p <= seq_len}
            # Low-discrepancy fill up to cap using golden ratio sequence
            need = max(0, POS_SAMPLE_CAP - len(anchors))
            interior = [i for i in range(1, seq_len) if i not in anchors]
            picks = set()
            if need > 0 and interior:
                phi = (5 ** 0.5 - 1) / 2.0  # ~0.618
                x = 0.0
                for _ in range(min(need, len(interior))):
                    x = (x + phi) % 1.0
                    idx = int(x * len(interior))
                    picks.add(interior[idx])
            pos_list = tuple(sorted(anchors.union(picks)))
            return pos_list

        def positions_for_insertion(self, seq_len, exhaustive=False, focus_idx=None):
            sig = self.positions_signature(seq_len, exhaustive=exhaustive, focus_idx=focus_idx)
            if isinstance(sig, tuple) and len(sig) == 2 and sig[0] == 'all':
                return list(range(seq_len + 1)), sig
            # Cache actual lists for non-all signatures to avoid recomputation
            cached = self.pos_sig_cache.get((seq_len, sig))
            if cached is None:
                cached = list(sig)
                self.pos_sig_cache[(seq_len, sig)] = cached
            return cached, sig

        def best_two_insertions(self, seq, txn, focus_idx=None, exhaustive=None):
            # Default: force all positions when small seq, otherwise stratified
            if exhaustive is None:
                exhaustive = (len(seq) <= EXHAUSTIVE_THRESHOLD)
            positions, sig = self.positions_for_insertion(len(seq), exhaustive=exhaustive, focus_idx=focus_idx)
            key_all = (tuple(seq), 'txn', txn, ('all', len(seq)))
            if sig == ('all', len(seq)):
                cached = self.best_two_cache.get(key_all)
                if cached is not None:
                    self.best_two_cache.move_to_end(key_all, last=True)
                    return cached
            key = (tuple(seq), 'txn', txn, sig)
            cached = self.best_two_cache.get(key)
            if cached is not None:
                self.best_two_cache.move_to_end(key, last=True)
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
            self._lru_put(key, res)
            if sig == ('all', len(seq)):
                self._lru_put(key_all, res)
            return res

        def best_two_block(self, seq, block, focus_idx=None, exhaustive=None):
            # Evaluate inserting a contiguous block (list of txns) into seq
            if exhaustive is None:
                exhaustive = (len(seq) <= EXHAUSTIVE_THRESHOLD)
            block_t = tuple(block)
            positions, sig = self.positions_for_insertion(len(seq), exhaustive=exhaustive, focus_idx=focus_idx)
            key_all = (tuple(seq), 'block', block_t, ('all', len(seq)))
            if sig == ('all', len(seq)):
                cached = self.best_two_cache.get(key_all)
                if cached is not None:
                    self.best_two_cache.move_to_end(key_all, last=True)
                    return cached
            key = (tuple(seq), 'block', block_t, sig)
            cached = self.best_two_cache.get(key)
            if cached is not None:
                self.best_two_cache.move_to_end(key, last=True)
                return cached
            best_c = float('inf')
            best_p = 0
            second_c = float('inf')
            for p in positions:
                cand = seq[:]
                cand[p:p] = block
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
            self._lru_put(key, res)
            if sig == ('all', len(seq)):
                self._lru_put(key_all, res)
            return res

    ctx = EpisodeContext(workload, n)

    # ---------------- Elite pool with suffix-3 diversity ----------------
    elites = []  # list of (cost, seq)

    def elite_suffix3(seq):
        return tuple(seq[-3:]) if len(seq) >= 3 else tuple(seq)

    def add_elite(cost, seq):
        nonlocal elites
        if not seq or len(seq) != n:
            return
        key = tuple(seq)
        # Replace exact match if better
        for i, (c, s) in enumerate(elites):
            if tuple(s) == key:
                if cost < c:
                    elites[i] = (cost, list(seq))
                elites.sort(key=lambda x: x[0])
                if len(elites) > ELITE_CAP:
                    elites = elites[:ELITE_CAP]
                return
        # Enforce suffix-3 diversity: replace if same suffix but better
        sfx = elite_suffix3(seq)
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

    # ---------------- GRASP constructor (regret-guided) ----------------
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

        # Strong second (positions 0/1)
        if remaining:
            k2 = min(8, len(remaining))
            trials = []
            for t in random.sample(list(remaining), k2):
                for p in [0, 1]:
                    cand = seq[:]
                    cand.insert(p, t)
                    c = ctx.eval_cost(cand)
                    trials.append((c, t, p))
            if trials:
                trials.sort(key=lambda x: x[0])
                pick = random.choice(trials[:min(3, len(trials))])
                _, t2, p2 = pick
                seq.insert(p2, t2)
                remaining.remove(t2)

        while remaining:
            size = len(remaining)
            cand_size = size if size <= (2 * CAND_SAMPLE_BASE) else min(size, max(5, CAND_SAMPLE_BASE + random.randint(-JITTER, JITTER)))
            cand_txns = random.sample(list(remaining), cand_size)

            force_all = (len(seq) <= EXHAUSTIVE_THRESHOLD) or (len(remaining) <= 2 * beam_width)
            scored = []
            for t in cand_txns:
                c1, p1, c2 = ctx.best_two_insertions(seq, t, exhaustive=force_all)
                scored.append((c1, max(0.0, c2 - c1), t, p1))

            scored.sort(key=lambda x: x[0])
            rcl = scored[:min(max(3, RCL_K), len(scored))]
            # favor high regret among good options with 60% probability
            if random.random() < 0.6:
                rcl.sort(key=lambda x: (-x[1], x[0]))
            chosen = rcl[0] if random.random() < 0.6 else random.choice(rcl)
            _, _, t_star, p_star = chosen
            seq.insert(p_star, t_star)
            remaining.remove(t_star)

        return ctx.eval_cost(seq), seq

    # ---------------- Regret-adaptive beam search ----------------
    def beam_search():
        bw = beam_width
        diversity_quota = max(1, int(diversity_quota_ratio * bw))

        # initialize with top singletons + one GRASP seed
        starters = [(ctx.eval_cost([t]), [t]) for t in all_txns]
        starters.sort(key=lambda x: x[0])
        init = starters[:min(len(starters), max(bw * 2, bw + 2))]
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
            if len(beam) >= bw:
                break

        best_complete = (float('inf'), [])

        for _depth in range(1, n + 1):
            if not beam:
                break
            next_pool = []
            layer_seen = set()
            suffix_seen = set()
            min_remaining = min((len(rem) for _, _, rem in beam), default=n)
            endgame = min_remaining <= 2 * bw

            for c_so_far, prefix, rem in beam:
                if not rem:
                    if c_so_far < best_complete[0]:
                        best_complete = (c_so_far, prefix)
                    continue

                rem_list = list(rem)
                expand_list = rem_list if len(rem_list) <= cand_per_state else random.sample(rem_list, cand_per_state)

                scored = []
                second_vals = []
                for t in expand_list:
                    new_prefix = prefix + [t]
                    c1 = ctx.eval_cost(new_prefix)
                    rem_after = [x for x in rem_list if x != t]
                    best_c2 = c1
                    samples = []
                    if rem_after:
                        k2 = len(rem_after) if (endgame and len(rem_after) <= 8) else min(lookahead_k, len(rem_after))
                        second = rem_after if k2 == len(rem_after) else random.sample(rem_after, k2)
                        best_c2 = float('inf')
                        for u in second:
                            cu = ctx.eval_cost(new_prefix + [u])
                            samples.append(cu)
                            if cu < best_c2:
                                best_c2 = cu
                    disp = (max(samples) - min(samples)) if len(samples) >= 2 else 0.0
                    second_vals.append(best_c2)
                    scored.append((c1, best_c2, disp, new_prefix, frozenset(rem_after)))

                if not scored:
                    continue

                # dispersion-aware scoring alpha
                if second_vals:
                    vmin, vmax = min(second_vals), max(second_vals)
                    dispersion = (vmax - vmin) if vmax >= vmin else 0.0
                else:
                    dispersion = 0.0

                local = []
                for c1, c2, disp, seq2, rem2 in scored:
                    alpha = 0.5 if disp > dispersion * 0.6 else 0.8
                    score = alpha * c1 + (1.0 - alpha) * c2
                    regret_proxy = max(0.0, c2 - c1)
                    local.append((score, c1, regret_proxy, seq2, rem2))

                    if len(seq2) == n and c1 < best_complete[0]:
                        best_complete = (c1, seq2)

                local.sort(key=lambda x: x[0])
                top_cost = local[:max(1, min(bw, len(local)))]
                top_regret = sorted(local, key=lambda x: (-x[2], x[0]))[:min(diversity_quota, len(local))]

                for score, c1, rg, seq2, rem2 in (top_cost + top_regret):
                    key = tuple(seq2)
                    if key in layer_seen:
                        continue
                    sig = tuple(seq2[-3:]) if len(seq2) >= 3 else tuple(seq2)
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
            local_bw = bw
            min_rem = min((len(r) for _, _, r in next_pool), default=n)
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

        # if found complete
        for c, seq, rem in beam:
            if not rem and c < best_complete[0]:
                best_complete = (c, seq)
        if best_complete[1] and len(best_complete[1]) == n:
            return best_complete

        # otherwise return best partial
        if beam:
            return min(beam, key=lambda x: x[0])[0:3]  # cost, seq, rem (partial)
        # fallback
        ident = list(range(n))
        return ctx.eval_cost(ident), ident, frozenset()

    # ---------------- Endgame BnB enumeration for last K ----------------
    def endgame_bnb(prefix, rem_set, incumbent_cost):
        rem = list(rem_set)
        best_cost = float('inf')
        best_seq = None
        nodes = 0
        # small transposition with LRU-like behavior
        TT = OrderedDict()

        def tt_put(key, val):
            TT[key] = val
            TT.move_to_end(key, last=True)
            if len(TT) > TT_CAP:
                TT.popitem(last=False)

        def dfs(cur_prefix, remaining):
            nonlocal best_cost, best_seq, nodes
            if nodes >= BNB_NODE_CAP:
                return
            nodes += 1
            if not remaining:
                c = ctx.eval_cost(cur_prefix)
                if c < best_cost:
                    best_cost = c
                    best_seq = list(cur_prefix)
                return
            # Bound with current best/ incumbent
            c_partial = ctx.eval_cost(cur_prefix)
            if c_partial >= best_cost or c_partial >= incumbent_cost:
                return
            # Transposition bound
            key = (frozenset(remaining), tuple(cur_prefix[-3:]) if len(cur_prefix) >= 3 else tuple(cur_prefix))
            val = TT.get(key)
            if val is not None and c_partial >= val - 1e-9:
                return
            tt_put(key, c_partial)

            # Order branches by regret proxy: choose next txn minimizing c(cur+[t]) and with high (second-best - best) gap
            scored = []
            for t in remaining:
                c1 = ctx.eval_cost(cur_prefix + [t])
                scored.append((c1, t))
            scored.sort(key=lambda x: x[0])
            # Explore best-first to prune quickly
            for c1, t in scored:
                if c1 >= best_cost or c1 >= incumbent_cost:
                    continue
                new_pref = cur_prefix + [t]
                rem_next = [x for x in remaining if x != t]
                dfs(new_pref, rem_next)

        dfs(list(prefix), rem)
        return best_cost, best_seq if best_seq is not None else (float('inf'), None)

    # ---------------- Local Search (VND with block-aware reinsertion) ----------------
    def vnd_local_search(seq, start_cost, max_rounds=VND_MAX_ROUNDS):
        best_seq = list(seq)
        best_cost = start_cost

        def or_opt_block(k):
            nonlocal best_seq, best_cost
            L = len(best_seq)
            if L <= k:
                return False
            improved = False
            i = 0
            while i <= len(best_seq) - k:
                block = best_seq[i:i + k]
                base = best_seq[:i] + best_seq[i + k:]
                # block-aware best-two using unified memo
                move_best_c, move_best_p, _ = ctx.best_two_block(base, block, focus_idx=i, exhaustive=(len(base) <= EXHAUSTIVE_THRESHOLD))
                if move_best_c + 1e-12 < best_cost:
                    new_seq = base[:]
                    new_seq[move_best_p:move_best_p] = block
                    best_seq = new_seq
                    best_cost = move_best_c
                    improved = True
                    i = 0  # restart scan after improvement
                else:
                    i += 1
            return improved

        def adjacent_swap_pass():
            nonlocal best_seq, best_cost
            improved = False
            for i in range(len(best_seq) - 1):
                cand = best_seq[:]
                cand[i], cand[i + 1] = cand[i + 1], cand[i]
                c = ctx.eval_cost(cand)
                if c < best_cost:
                    best_seq = cand
                    best_cost = c
                    improved = True
            return improved

        def sampled_pair_swaps(tries=PAIR_SWAP_TRIES):
            nonlocal best_seq, best_cost
            L = len(best_seq)
            if L <= 4:
                return False
            improved = False
            best_move = None
            best_delta = 0.0
            attempts = min(tries, max(80, L))
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

        def sampled_two_opt(tries=TWO_OPT_TRIES):
            nonlocal best_seq, best_cost
            L = len(best_seq)
            if L <= 6:
                return False
            improved = False
            best_move = None
            best_delta = 0.0
            attempts = min(tries, max(80, L))
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
                if or_opt_block(k):
                    any_improved = True
            if adjacent_swap_pass():
                any_improved = True
            if sampled_pair_swaps():
                any_improved = True
            if sampled_two_opt():
                any_improved = True
            if not any_improved:
                break

        return best_seq, best_cost

    # ---------------- LNS: Sensitivity + marginal destroy, memo-repair ----------------
    def lns_ruin_and_repair(seq, start_cost, rounds=LNS_ROUNDS):
        best_seq = list(seq)
        best_cost = start_cost
        stagnation = 0

        for _ in range(max(1, rounds)):
            cur = list(best_seq)
            L = len(cur)
            if L <= 6:
                break

            # Sensitivity (span) over P positions + marginal remove benefit
            K = min(24, L)
            P = 6
            sampled_idx = random.sample(range(L), K)
            metrics = []
            for i in sampled_idx:
                t = cur[i]
                base = cur[:i] + cur[i + 1:]
                positions, _ = ctx.positions_for_insertion(len(base), exhaustive=False, focus_idx=i)
                # pick up to P anchors
                picks = [0, len(base)]
                if len(base) >= 2:
                    picks.append(len(base) // 2)
                if len(base) >= 4:
                    picks.extend([len(base) // 4, (3 * len(base)) // 4])
                # fill from signature list
                for p in positions:
                    if p not in picks:
                        picks.append(p)
                    if len(picks) >= P:
                        break
                costs = []
                for p in picks[:P]:
                    cand = base[:]
                    cand.insert(p, t)
                    costs.append(ctx.eval_cost(cand))
                span = (max(costs) - min(costs)) if len(costs) >= 2 else 0.0
                # marginal benefit via leave-one-out
                cost_wo = ctx.eval_cost(base)
                marginal = max(0.0, best_cost - cost_wo)
                # combined score
                metrics.append((0.6 * span + 0.4 * marginal, i))
            metrics.sort(reverse=True)

            # Removal budget with escalation
            frac = LNS_REMOVE_FRAC * (1.25 if stagnation >= 2 else 1.0)
            k_remove = max(LNS_REMOVE_MIN, min(LNS_REMOVE_MAX, int(frac * L)))
            k_remove = min(k_remove, L - 2)

            remove_idx_set = set(i for _, i in metrics[:max(1, int(0.5 * k_remove))])

            # add contiguous block(s)
            need = k_remove - len(remove_idx_set)
            blocks = 2 if stagnation >= 2 else 1
            for _b in range(blocks):
                if need <= 0:
                    break
                blk_len = max(2, min(need, max(2, L // 12)))
                start = random.randint(0, max(0, L - blk_len))
                for j in range(start, start + blk_len):
                    if len(remove_idx_set) >= k_remove:
                        break
                    remove_idx_set.add(j)
                need = k_remove - len(remove_idx_set)
            while len(remove_idx_set) < k_remove:
                remove_idx_set.add(random.randint(0, L - 1))
            remove_idx = sorted(remove_idx_set)

            removed = [cur[i] for i in remove_idx]
            base_seq = [cur[i] for i in range(L) if i not in remove_idx]

            # Repair: regret-guided using best-two memo
            rebuilt = list(base_seq)
            pool = list(removed)
            while pool:
                exhaustive = (len(rebuilt) <= EXHAUSTIVE_THRESHOLD) or (len(pool) <= 2 * beam_width)
                cand_txns = pool if len(pool) <= 10 else random.sample(pool, 10)
                scored = []
                for t in cand_txns:
                    c1, p1, c2 = ctx.best_two_insertions(rebuilt, t, focus_idx=None, exhaustive=exhaustive)
                    scored.append((c1, c2, t, p1))
                if scored:
                    scored.sort(key=lambda x: x[0])
                    top = scored[:min(3, len(scored))]
                    top.sort(key=lambda x: (-(x[1] - x[0]), x[0]))
                    c1, c2, t_pick, p_pick = top[0]
                    rebuilt.insert(p_pick, t_pick)
                    pool.remove(t_pick)
                else:
                    t = pool.pop()
                    rebuilt.append(t)

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

    # ---------------- Path relinking (bidirectional, block-aware) ----------------
    def path_relink(a_seq, b_seq, keep_intermediate=2):
        def relink_one(src, tgt):
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
                # move the longest consecutive block matching target order
                k = 1
                while j + k < len(cur) and i + k < len(tgt) and cur[j + k] == tgt[i + k]:
                    k += 1
                block = cur[j:j + k]
                base = cur[:j] + cur[j + k:]
                new_seq = base[:i] + block + base[i:]
                c = ctx.eval_cost(new_seq)
                cur = new_seq
                candidates.append((c, list(cur)))
                if c < best_c:
                    best_c, best_s = c, list(cur)
                i += k
            candidates.sort(key=lambda x: x[0])
            for c, s in candidates[:min(keep_intermediate, len(candidates))]:
                s2, c2 = vnd_local_search(s, c, max_rounds=1)
                if c2 < best_c:
                    best_c, best_s = c2, s2
            return best_c, best_s

        c1, s1 = relink_one(a_seq, b_seq)
        c2, s2 = relink_one(b_seq, a_seq)
        return (c1, s1) if c1 <= c2 else (c2, s2)

    # ---------------- Orchestration ----------------
    best_cost = float('inf')
    best_seq = list(range(n))

    # 1) GRASP seeds + quick VND polish
    seeds = max(3, min(6, num_seqs))
    for _ in range(seeds):
        c, s = construct_grasp_regret()
        s, c = vnd_local_search(s, c, max_rounds=1)
        add_elite(c, s)
        if c < best_cost:
            best_cost, best_seq = c, s

    # 2) Beam search seeds; if partial, run bounded BnB to finish
    beam_runs = max(1, num_seqs // 3)
    for _ in range(beam_runs):
        res = beam_search()
        if len(res) == 2:
            c, s = res
        else:
            c_partial, prefix, rem = res
            if isinstance(rem, frozenset) and len(rem) <= BNB_K:
                bnb_c, bnb_s = endgame_bnb(prefix, rem, incumbent_cost=best_cost)
                if bnb_s is not None and bnb_c < float('inf'):
                    c, s = bnb_c, bnb_s
                else:
                    # greedy completion if BnB did not return
                    cur = list(prefix)
                    rem_list = list(rem)
                    while rem_list:
                        best_t, best_cand = None, float('inf')
                        for t in rem_list:
                            c2 = ctx.eval_cost(cur + [t])
                            if c2 < best_cand:
                                best_cand, best_t = c2, t
                        cur.append(best_t); rem_list.remove(best_t)
                    c, s = ctx.eval_cost(cur), cur
            else:
                # greedy completion
                cur = list(prefix)
                rem_list = list(rem) if isinstance(rem, frozenset) else []
                while rem_list:
                    best_t, best_cand = None, float('inf')
                    for t in rem_list:
                        c2 = ctx.eval_cost(cur + [t])
                        if c2 < best_cand:
                            best_cand, best_t = c2, t
                    cur.append(best_t); rem_list.remove(best_t)
                c, s = ctx.eval_cost(cur), cur

        s, c = vnd_local_search(s, c, max_rounds=1)
        add_elite(c, s)
        if c < best_cost:
            best_cost, best_seq = c, s

    # 3) Path relinking among elites
    if len(elites) >= 2:
        base_cost, base_seq = elites[0]
        for i in range(1, min(len(elites), ELITE_CAP)):
            _, other = elites[i]
            c_rel, s_rel = path_relink(base_seq, other, keep_intermediate=2)
            s_rel, c_rel = vnd_local_search(s_rel, c_rel, max_rounds=1)
            add_elite(c_rel, s_rel)
            if c_rel < best_cost:
                best_cost, best_seq = c_rel, s_rel

    # 4) Iterated LNS + VND from incumbent or elites, with endgame BnB finish if possible
    incumbent_cost, incumbent_seq = best_cost, best_seq
    for _ in range(ILS_ITERS):
        if elites and random.random() < 0.5:
            start_c, start_s = random.choice(elites[:min(3, len(elites))])
        else:
            start_c, start_s = incumbent_cost, incumbent_seq
        s1, c1 = lns_ruin_and_repair(start_s, start_c, rounds=LNS_ROUNDS)
        # Endgame BnB on suffix if helpful: reorder last <= K elements while keeping prefix fixed
        if len(s1) >= BNB_K + 1:
            prefix = s1[:-BNB_K]
            suffix = s1[-BNB_K:]
            bnb_c, bnb_s = endgame_bnb(prefix, frozenset(suffix), incumbent_cost=min(best_cost, c1))
            if bnb_s is not None and bnb_c < c1:
                s1, c1 = bnb_s, bnb_c
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