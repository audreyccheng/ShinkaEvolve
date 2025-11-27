# EVOLVE-BLOCK-START
"""Transaction scheduling algorithm for optimizing makespan across multiple workloads"""

import time
import random
import sys
import os
from collections import deque

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
    Find a low-makespan schedule using:
    - Regret-adaptive insertion beam with 1-step lookahead and suffix diversity
    - Unified best-two block insertion cache reused across all neighborhoods
    - Bounded endgame BnB with transposition table for exact/near-exact completion
    - Strong VND with don’t-look bits, Or-opt(1/2/3), sampled swaps, segment reversals
    - LNS destroy&repair with best-two guided reinsertion
    - Bidirectional path relinking using best-two guided relocations and quick LNS polish
    """
    n = workload.num_txns
    all_txns = list(range(n))

    # ----------------------------
    # Global memoization for sequence costs
    # ----------------------------
    cost_cache = {}
    def seq_cost(seq):
        k = tuple(seq)
        c = cost_cache.get(k)
        if c is not None:
            return c
        c = workload.get_opt_seq_cost(seq)
        cost_cache[k] = c
        return c

    # ----------------------------
    # Best-two cache for block insertion with deterministic pos policies
    # ----------------------------
    # Cache keyed by (tuple(base_seq), tuple(block), tuple(pos_list))
    best_two_block_cache = {}
    # Lightweight LRU of keys to bound memory
    best_two_block_keys = deque()
    best_two_block_cap = 40000

    # Position policy cache if we need to reuse pos_list from the same signature
    pos_policy_cache = {}
    pos_policy_keys = deque()
    pos_policy_cap = 20000

    # Parameters (adaptive)
    elite_size = max(3, min(7, 2 + (num_seqs // 3)))
    seed_elite_singletons = max(3, min(7, int(num_seqs)))      # top singleton seeds
    seed_random_additional = max(1, min(4, (num_seqs + 1) // 3))

    # Beam & sampling
    base_beam_width = max(4, min(8, 2 + num_seqs))
    suffix_div_k = 3  # stronger suffix diversity in late stages
    k_txn_sample = min(20, max(8, 2 + int(1.7 * num_seqs)))
    k_pos_sample = min(12, max(7, 3 + int(1.2 * num_seqs)))
    rem_all_threshold = 14
    endgame_all_pos_threshold = max(6, min(12, 2 + num_seqs))
    lookahead_txn_k = max(4, min(8, 2 + num_seqs // 2))

    # Local search
    vnd_dont_look_radius = 2
    two_opt_samples = min(240, max(60, 2 * n))
    pair_swap_samples = min(400, max(80, 3 * n))

    # ILS
    ils_rounds = max(2, min(6, 1 + num_seqs // 3))
    perturb_swap_count = max(2, min(6, 2 + num_seqs // 3))
    perturb_block_len = max(3, min(10, 3 + num_seqs // 2))

    # LNS
    lns_iters = max(2, min(7, 2 + num_seqs // 2))
    destroy_frac_range = (0.08, 0.18)

    # Endgame BnB
    endgame_K_base = max(6, min(10, endgame_all_pos_threshold + 2))
    bnb_node_cap = 3000

    # ----------------------------
    # Deterministic position policy for reuse across phases
    # ----------------------------
    def build_pos_list(seq_suffix, L, block_tuple, use_all, focus_idx, cap):
        if use_all or L <= 12:
            pos_list = list(range(L + 1))
            pos_sig = ('all', L)
            return pos_list, pos_sig

        # Deterministic sampling: anchors + near-focus + seeded interior
        cap_eff = cap if cap is not None else k_pos_sample
        anchors = {0, L, L // 2, (L * 1) // 4, (L * 3) // 4}
        if focus_idx is not None:
            for d in (-3, -2, -1, 0, 1, 2, 3):
                p = focus_idx + d
                if 0 <= p <= L:
                    anchors.add(p)

        seed = (tuple(seq_suffix), block_tuple, L, cap_eff)
        # Use a stable 32-bit hash for reproducibility but Python version independent
        rng = random.Random(hash(seed) & 0xffffffff)
        while len(anchors) < min(cap_eff + 5, L + 1):
            anchors.add(rng.randint(0, L))
        pos_list = sorted(anchors)
        pos_sig = ('det', L, cap_eff, tuple(sorted(anchors)))  # stable signature for caching
        return pos_list, pos_sig

    def get_pos_list_cached(seq_suffix, L, block_tuple, use_all, focus_idx, cap):
        key = (tuple(seq_suffix), L, block_tuple, bool(use_all), focus_idx if focus_idx is not None else -1, cap if cap is not None else -1)
        if key in pos_policy_cache:
            return pos_policy_cache[key]
        pos_list, pos_sig = build_pos_list(seq_suffix, L, block_tuple, use_all, focus_idx, cap)
        pos_policy_cache[key] = (pos_list, pos_sig)
        pos_policy_keys.append(key)
        if len(pos_policy_cache) > pos_policy_cap:
            for _ in range(min(1000, len(pos_policy_keys))):
                old = pos_policy_keys.popleft()
                pos_policy_cache.pop(old, None)
        return pos_list, pos_sig

    # ----------------------------
    # Unified best-two block insertion with cache
    # ----------------------------
    def best_two_block(base_seq, block, use_all_pos=False, focus_idx=None, k_positions=None, exclude_positions=None):
        """
        Compute best and second-best insertion of a block (list of txns) into base_seq.
        Returns (best_cost, best_pos, second_cost).
        - use_all_pos: evaluate all positions or deterministic sample otherwise
        - focus_idx: bias around an index
        - exclude_positions: set of positions to skip (e.g., original index to avoid no-op)
        """
        L = len(base_seq)
        block_tuple = tuple(block)
        seq_suffix = base_seq[-min(10, L):]
        pos_list, pos_sig = get_pos_list_cached(seq_suffix, L, block_tuple, use_all_pos, focus_idx, k_positions)
        cache_key = (tuple(base_seq), block_tuple, pos_sig)
        cached = best_two_block_cache.get(cache_key)
        if cached is not None and (exclude_positions is None or not exclude_positions):
            return cached

        best = (float('inf'), None)
        second = float('inf')
        excl = set(exclude_positions) if exclude_positions else set()
        for p in pos_list:
            if p in excl:
                continue
            cand = base_seq[:p] + block + base_seq[p:]
            c = seq_cost(cand)
            if c < best[0]:
                second = best[0]
                best = (c, p)
            elif c < second:
                second = c

        res = (best[0], best[1], second)
        if not exclude_positions:
            best_two_block_cache[cache_key] = res
            best_two_block_keys.append(cache_key)
            if len(best_two_block_cache) > best_two_block_cap:
                for _ in range(min(2000, len(best_two_block_keys))):
                    old = best_two_block_keys.popleft()
                    best_two_block_cache.pop(old, None)
        return res

    def best_two_insertion(base_seq, t, use_all_pos=False, focus_idx=None, k_positions=None, exclude_positions=None):
        return best_two_block(base_seq, [t], use_all_pos, focus_idx, k_positions, exclude_positions)

    # ----------------------------
    # Endgame exact/near-exact completion with bounded BnB
    # ----------------------------
    endgame_memo = {}
    def endgame_optimal_completion(prefix_seq, rem_set):
        """
        Branch-and-bound search over remaining txns (|R| <= K).
        Transposition by (frozenset(R), prefix_suffix_3) -> best known cost to prune duplicates.
        Node expansion capped; if exceeded, return best seen so far or greedy completion.
        """
        R = list(rem_set)
        K = len(R)
        # trivial
        if K == 0:
            c0 = seq_cost(prefix_seq)
            return c0, prefix_seq[:]

        # Order by higher regret first (using full positions) then lower best insertion cost
        order = []
        for t in R:
            b, p, s2 = best_two_insertion(prefix_seq, t, use_all_pos=True)
            rg = (s2 - b) if s2 < float('inf') else 0.0
            order.append((-rg, b, t))
        order.sort()

        best_cost = float('inf')
        best_seq = None
        nodes = 0

        transpo = {}  # local transposition: key -> best partial cost observed
        prefix_tail = tuple(prefix_seq[-3:]) if len(prefix_seq) >= 3 else tuple(prefix_seq)

        def dfs(seq, remaining, cur_bound):
            nonlocal best_cost, best_seq, nodes
            nodes += 1
            if nodes > bnb_node_cap:
                return
            c_prefix = seq_cost(seq)
            if c_prefix >= best_cost:
                return
            key = (frozenset(remaining), tuple(seq[-3:]) if len(seq) >= 3 else tuple(seq))
            prev_best = transpo.get(key)
            if prev_best is not None and c_prefix >= prev_best:
                return
            transpo[key] = c_prefix

            if not remaining:
                if c_prefix < best_cost:
                    best_cost = c_prefix
                    best_seq = seq[:]
                return

            # Candidate ordering: higher regret first, then lower best cost
            cand = []
            for _rg, _b, t in order:
                if t not in remaining:
                    continue
                b, p, s2 = best_two_insertion(seq, t, use_all_pos=True)
                rg = (s2 - b) if s2 < float('inf') else 0.0
                cand.append((-rg, b, t, p))
            cand.sort()
            for _neg_rg, bcost, t, p in cand:
                if nodes > bnb_node_cap:
                    break
                if bcost >= best_cost:
                    continue
                new_seq = seq[:p] + [t] + seq[p:]
                new_rem = set(remaining)
                new_rem.remove(t)
                dfs(new_seq, new_rem, best_cost)

        # seed with prefix
        dfs(prefix_seq[:], set(R), best_cost)

        if best_seq is None:
            # Fallback to greedy completion using best-two insertions
            seq_rep = prefix_seq[:]
            rem = set(R)
            while rem:
                picks = []
                for t in list(rem):
                    b, p, s2 = best_two_insertion(seq_rep, t, use_all_pos=True)
                    picks.append((b, t, p))
                picks.sort()
                b, t, p = picks[0]
                seq_rep = seq_rep[:p] + [t] + seq_rep[p:]
                rem.remove(t)
            best_cost = seq_cost(seq_rep)
            best_seq = seq_rep
        return best_cost, best_seq

    # ----------------------------
    # Construction: regret-adaptive beam with 1-step lookahead
    # ----------------------------
    def suffix_sig(seq):
        if not seq:
            return (None,)
        k = min(len(seq), suffix_div_k)
        return tuple(seq[-k:])

    def build_from_seed(seed_t):
        seq0 = [seed_t]
        rem0 = set(all_txns)
        rem0.remove(seed_t)
        beam = [(seq0, rem0, seq_cost(seq0))]

        while True:
            if all(len(rem) == 0 for _, rem, _ in beam):
                break

            expansions = []
            seen_seqs = set()
            for seq, rem, _c in beam:
                if not rem:
                    key = tuple(seq)
                    if key not in seen_seqs:
                        seen_seqs.add(key)
                        expansions.append((seq, rem, seq_cost(seq), 0.0, seq))
                    continue

                # Candidate txns
                if len(rem) <= rem_all_threshold:
                    cand_txns = list(rem)
                else:
                    cand_txns = random.sample(list(rem), min(k_txn_sample, len(rem)))

                # Position policy
                use_all = (len(seq) <= 18) or (len(rem) <= endgame_all_pos_threshold)
                for t in cand_txns:
                    b, p, s2 = best_two_insertion(seq, t, use_all_pos=use_all)
                    reg = (s2 - b) if s2 < float('inf') else 0.0
                    new_seq = seq[:p] + [t] + seq[p:]
                    new_rem = rem.copy()
                    new_rem.remove(t)

                    # 1-step lookahead
                    if new_rem:
                        if len(new_rem) <= lookahead_txn_k:
                            cand2 = list(new_rem)
                        else:
                            cand2 = random.sample(list(new_rem), lookahead_txn_k)
                        best_c2 = float('inf')
                        use_all2 = len(new_seq) <= 18
                        for u in cand2:
                            c2, _, _ = best_two_insertion(new_seq, u, use_all_pos=use_all2)
                            if c2 < best_c2:
                                best_c2 = c2
                    else:
                        best_c2 = b

                    expansions.append((new_seq, new_rem, b, reg, best_c2))

            if not expansions:
                break

            # Adaptive blend
            scs = [e[4] for e in expansions]
            regs = [e[3] for e in expansions]
            if scs and regs:
                spread = (max(scs) - min(scs))
                med_reg = sorted(regs)[len(regs) // 2]
                alpha = 0.5 if spread > med_reg else 0.8
            else:
                alpha = 0.8

            scored = []
            for s, r, b, reg, c2 in expansions:
                score = alpha * b + (1.0 - alpha) * c2
                scored.append((score, b, -reg, s, r))

            # Beam width adaptation near endgame
            rem_sizes = [len(r) for _, _, _, _, r in expansions]
            min_rem = min(rem_sizes) if rem_sizes else 0
            k_beam = base_beam_width + 2 if min_rem <= 2 * base_beam_width else base_beam_width

            scored.sort(key=lambda x: (x[0], x[1], x[2]))
            next_beam = []
            seen_suffix = set()
            for score, b, nreg, s, r in scored:
                sig = suffix_sig(s)
                if sig in seen_suffix:
                    continue
                seen_suffix.add(sig)
                next_beam.append((s, r, b))
                if len(next_beam) >= k_beam:
                    break

            if not next_beam:
                # Fallback: keep best
                next_beam = [(s, r, b) for _, b, _, s, r in scored[:k_beam]]

            beam = next_beam

        # If incomplete sequences remain (rare), finish with endgame BnB if small enough
        best_seq = None
        best_cost = float('inf')
        for seq, rem, c in beam:
            if not rem:
                cc = c
                ss = seq
            else:
                if len(rem) <= endgame_K_base:
                    cc, ss = endgame_optimal_completion(seq, rem)
                else:
                    seq_complete = seq + sorted(list(rem))
                    cc = seq_cost(seq_complete)
                    ss = seq_complete
            if cc < best_cost:
                best_cost = cc
                best_seq = ss
        return best_seq

    # ----------------------------
    # Local refinement: VND with don’t-look bits, Or-opt(1..3), swaps, 2-opt
    # ----------------------------
    def vnd_refine(seq):
        best_seq = seq[:]
        best_cost = seq_cost(best_seq)

        # Don’t-look bits per index (reset around improved segments)
        dont_look = [False] * n

        def reset_bits_around(i, radius=vnd_dont_look_radius):
            lo = max(0, i - radius)
            hi = min(n - 1, i + radius)
            for k in range(lo, hi + 1):
                dont_look[k] = False

        improved = True
        while improved:
            improved = False

            # 1) Or-opt blocks (3, 2, 1) with first improvement acceptance
            for blk in (3, 2, 1):
                L = len(best_seq)
                i = 0
                while i <= L - blk:
                    if dont_look[i]:
                        i += 1
                        continue
                    block = best_seq[i:i + blk]
                    base = best_seq[:i] + best_seq[i + blk:]
                    exclude = {i}
                    use_all = len(base) <= 20
                    b, p, s2 = best_two_block(base, block, use_all_pos=use_all, focus_idx=i, exclude_positions=exclude)
                    if p is None:
                        i += 1
                        continue
                    cand = base[:p] + block + base[p:]
                    if cand == best_seq:
                        i += 1
                        continue
                    c = seq_cost(cand)
                    if c < best_cost:
                        best_cost = c
                        best_seq = cand
                        improved = True
                        # Reset don’t-look around affected region
                        reset_bits_around(max(i, p))
                        # Restart this neighborhood
                        L = len(best_seq)
                        i = 0
                        continue
                    else:
                        dont_look[i] = True
                        i += 1
                if improved:
                    break
            if improved:
                continue

            # 2) Adjacent swaps (first-improvement)
            i = 0
            while i < n - 1:
                if dont_look[i]:
                    i += 1
                    continue
                cand = best_seq[:]
                cand[i], cand[i + 1] = cand[i + 1], cand[i]
                c = seq_cost(cand)
                if c < best_cost:
                    best_cost = c
                    best_seq = cand
                    improved = True
                    reset_bits_around(i)
                    i = 0
                else:
                    dont_look[i] = True
                    i += 1
            if improved:
                continue

            # 3) Sampled non-adjacent swaps
            for _ in range(pair_swap_samples):
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)
                if i == j or abs(i - j) == 1:
                    continue
                if i > j:
                    i, j = j, i
                cand = best_seq[:]
                cand[i], cand[j] = cand[j], cand[i]
                c = seq_cost(cand)
                if c < best_cost:
                    best_cost = c
                    best_seq = cand
                    improved = True
                    reset_bits_around((i + j) // 2)
                    break
            if improved:
                continue

            # 4) 2-opt segment reversals (true segment reverse)
            for _ in range(two_opt_samples):
                i = random.randint(0, n - 2)
                j = random.randint(i + 2, n - 1)  # ensure at least length 2 segment
                cand = best_seq[:i] + best_seq[i:j + 1][::-1] + best_seq[j + 1:]
                c = seq_cost(cand)
                if c < best_cost:
                    best_cost = c
                    best_seq = cand
                    improved = True
                    reset_bits_around((i + j) // 2)
                    break

        return best_cost, best_seq

    # ----------------------------
    # Perturbation for ILS
    # ----------------------------
    def perturb(seq):
        s = seq[:]
        mode = random.random()
        if mode < 0.5:
            # random swaps
            for _ in range(perturb_swap_count):
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)
                if i != j:
                    s[i], s[j] = s[j], s[i]
        else:
            # block shuffle/reinsert
            if n > perturb_block_len + 2:
                start = random.randint(0, n - perturb_block_len - 1)
                block = s[start:start + perturb_block_len]
                base = s[:start] + s[start + perturb_block_len:]
                # insert block at best position
                use_all = len(base) <= 20
                b, p, _ = best_two_block(base, block, use_all_pos=use_all, focus_idx=start)
                if p is None:
                    s = base + block
                else:
                    s = base[:p] + block + base[p:]
        return s

    # ----------------------------
    # LNS destroy-and-repair using best-two; endgame exact if small
    # ----------------------------
    def lns_attempt(seq):
        cur = seq[:]
        frac = random.uniform(*destroy_frac_range)
        m = max(4, min(n // 2, int(frac * n)))
        # Remove either contiguous block or random subset
        if random.random() < 0.5:
            start = random.randint(0, n - m)
            remove_idxs = set(range(start, start + m))
        else:
            remove_idxs = set(random.sample(range(n), m))
        removed = [cur[i] for i in sorted(remove_idxs)]
        remaining = [cur[i] for i in range(n) if i not in remove_idxs]

        seq_rep = remaining[:]
        rem_list = removed[:]

        if len(rem_list) <= endgame_K_base:
            c_end, s_end = endgame_optimal_completion(seq_rep, set(rem_list))
            c_final, s_final = vnd_refine(s_end)
            return c_final, s_final

        # Greedy regret-guided reinsertion
        while rem_list:
            if len(rem_list) > k_txn_sample:
                cand_txns = random.sample(rem_list, k_txn_sample)
            else:
                cand_txns = rem_list[:]
            use_all = (len(seq_rep) <= 18) or (len(rem_list) <= endgame_all_pos_threshold)
            best_overall = (float('inf'), None, None)  # cost, t, p
            best_regret = (float('-inf'), None, None)  # regret, t, p
            for t in cand_txns:
                b, p, s2 = best_two_insertion(seq_rep, t, use_all_pos=use_all)
                rg = (s2 - b) if s2 < float('inf') else 0.0
                if b < best_overall[0]:
                    best_overall = (b, t, p)
                if rg > best_regret[0]:
                    best_regret = (rg, t, p)
            # pick regret if helpful else best
            chosen = best_regret if best_regret[1] is not None and best_regret[0] > 0 else best_overall
            t = chosen[1] if chosen[1] is not None else random.choice(rem_list)
            p = chosen[2] if chosen[2] is not None else len(seq_rep)
            seq_rep = seq_rep[:p] + [t] + seq_rep[p:]
            rem_list.remove(t)

        return vnd_refine(seq_rep)

    # ----------------------------
    # Path relinking (bidirectional) with best-two guided relocations
    # ----------------------------
    def path_relink(source_seq, target_seq, max_moves=None):
        if max_moves is None:
            max_moves = max(10, min(16, n // 6))
        pos_in_target = {t: i for i, t in enumerate(target_seq)}
        s = source_seq[:]
        best_c = seq_cost(s)
        best_s = s[:]

        # Order items by largest displacement
        displacement = sorted([(abs(i - pos_in_target[s[i]]), i) for i in range(n)], reverse=True)
        moves = 0
        for _, idx in displacement:
            if moves >= max_moves:
                break
            item = s[idx]
            desired = pos_in_target[item]
            base = s[:idx] + s[idx + 1:]
            # Pick best position guided by desired as focus
            b, p, s2 = best_two_insertion(base, item, use_all_pos=(len(base) <= 18), focus_idx=desired)
            if p is None:
                continue
            cand = base[:p] + [item] + base[p:]
            c = seq_cost(cand)
            if c < best_c:
                best_c = c
                best_s = cand
                s = cand
                moves += 1
        return best_c, best_s

    # ----------------------------
    # Seeding and elites
    # ----------------------------
    singleton_scores = []
    for t in all_txns:
        singleton_scores.append((seq_cost([t]), t))
    singleton_scores.sort(key=lambda x: x[0])

    seed_txns = [t for _, t in singleton_scores[:seed_elite_singletons]]
    remaining_for_seeds = [t for t in all_txns if t not in seed_txns]
    if remaining_for_seeds and seed_random_additional > 0:
        extra = random.sample(remaining_for_seeds, min(seed_random_additional, len(remaining_for_seeds)))
        seed_txns.extend(extra)

    elite = []  # list of (cost, seq)
    def add_elite(c, s):
        nonlocal elite
        elite.append((c, s))
        elite.sort(key=lambda x: x[0])
        # keep unique sequences by suffix diversity
        uniq = []
        seen_sig = set()
        for c1, s1 in elite:
            sig = suffix_sig(s1)
            if sig in seen_sig:
                continue
            seen_sig.add(sig)
            uniq.append((c1, s1))
            if len(uniq) >= elite_size:
                break
        elite = uniq

    # Build from seeds
    best_overall_cost = float('inf')
    best_overall_seq = None
    for seed in seed_txns:
        seq0 = build_from_seed(seed)
        c1, s1 = vnd_refine(seq0)
        add_elite(c1, s1)
        if c1 < best_overall_cost:
            best_overall_cost, best_overall_seq = c1, s1

    if best_overall_seq is None:
        base = all_txns[:]
        random.shuffle(base)
        best_overall_seq = base
        best_overall_cost = seq_cost(base)
        add_elite(best_overall_cost, best_overall_seq)

    # Iterated Local Search
    cur_cost, cur_seq = best_overall_cost, best_overall_seq[:]
    for _ in range(ils_rounds):
        pseq = perturb(cur_seq)
        c2, s2 = vnd_refine(pseq)
        add_elite(c2, s2)
        if c2 < cur_cost:
            cur_cost, cur_seq = c2, s2
            if c2 < best_overall_cost:
                best_overall_cost, best_overall_seq = c2, s2

    # LNS attempts
    for _ in range(lns_iters):
        c3, s3 = lns_attempt(best_overall_seq)
        add_elite(c3, s3)
        if c3 < best_overall_cost:
            best_overall_cost, best_overall_seq = c3, s3

    # Path relinking among elites with quick polish
    if len(elite) >= 2:
        base_seq = best_overall_seq[:]
        partners = elite[1:min(len(elite), elite_size)]
        for c_e, s_e in partners:
            pr1_c, pr1_s = path_relink(base_seq, s_e)
            pr2_c, pr2_s = path_relink(s_e, base_seq)
            if pr1_c <= pr2_c:
                cand_c, cand_s = pr1_c, pr1_s
            else:
                cand_c, cand_s = pr2_c, pr2_s
            if cand_c < best_overall_cost:
                # Quick LNS polish removing a small biased fraction
                c_pol, s_pol = lns_attempt(cand_s)
                if c_pol < best_overall_cost:
                    best_overall_cost, best_overall_seq = c_pol, s_pol
                else:
                    best_overall_cost, best_overall_seq = cand_c, cand_s

    return best_overall_cost, best_overall_seq


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