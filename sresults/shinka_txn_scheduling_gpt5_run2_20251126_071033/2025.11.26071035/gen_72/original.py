# EVOLVE-BLOCK-START
"""Transaction scheduling algorithm for optimizing makespan across multiple workloads"""

import time
import random
import sys
import os
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
    Hybrid regret-guided construction + adaptive beam + sensitivity LNS + VND with DLB + elite relinking.
    Minimize makespan of the workload.

    Args:
        workload: Workload object containing transaction data
        num_seqs: Search effort parameter (used to scale beam width, restarts and local effort)

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    n = workload.num_txns
    all_txns = list(range(n))

    # ---------------- Parameterization (adaptive by size/effort) ----------------
    small, med, large = (n <= 50), (50 < n <= 80), (n > 80)

    # Position sampling caps (for insertion evaluation)
    pos_sample_cap = None if small else (18 if med else 16)
    rand_pos_R = 6 if small else (6 if med else 5)
    exhaustive_threshold = 20  # evaluate all positions when seq <= this or endgame

    # Beam parameters
    local_beam_width = max(5, min(10, (num_seqs + (2 if small else 0))))
    cand_per_state = min(28, max(10, n // (4 if large else 3)))
    lookahead_k = 6 if (med or large) else 8
    high_regret_quota_ratio = 0.3
    diversity_suffix_k = 2  # increase to 3 in endgame

    # Construction (GRASP)
    CAND_SAMPLE_BASE = 12 if small else 9
    STARTER_SAMPLE = min(10, n)
    RCL_K = 3
    JITTER = 2
    rcl_alpha = 0.6  # cost vs regret choice balance

    # Local search
    vnd_rounds = 2 if large else 3
    two_opt_samples = min(200, max(80, n))
    swap_samples = min(200, max(80, n))
    oropt_pos_cap = 24 if (med or large) else None

    # LNS
    lns_rounds_base = 2 if med or large else 3
    lns_remove_frac = 0.1 if large else (0.12 if med else 0.14)
    lns_remove_min = 8 if (med or large) else 6
    lns_remove_max = 18 if (med or large) else 16

    # Iterated improvements (ILS)
    ils_iterations = 2 if large else 3

    # Elite pool
    elite_cap = 5

    # ---------------- Cached cost evaluator ----------------
    cost_cache = {}

    def eval_cost(prefix):
        key = tuple(prefix)
        c = cost_cache.get(key)
        if c is None:
            c = workload.get_opt_seq_cost(list(prefix))
            cost_cache[key] = c
        return c

    # ---------------- Stratified position sampling ----------------
    def stratified_positions(seq_len, cap=pos_sample_cap, R=rand_pos_R, exhaustive=False):
        total = seq_len + 1
        if exhaustive or cap is None or total <= cap:
            return list(range(total))
        anchors = {0, seq_len, seq_len // 2, seq_len // 4, (3 * seq_len) // 4}
        anchors = {p for p in anchors if 0 <= p <= seq_len}
        interior = [p for p in range(1, seq_len) if p not in anchors]
        r = min(R, len(interior))
        if r > 0:
            anchors.update(random.sample(interior, r))
        return sorted(anchors)

    # ---------------- Best-two insertion with per-episode cache ----------------
    # best_two_cache[(mode, tuple(seq), txn)] -> (best_cost, best_pos, second_cost)
    def make_best_two_cache():
        return {}

    def best_two_insertions(seq, txn, bt_cache, exhaustive=False):
        mode = 'all' if exhaustive else 'strat'
        key = (mode, tuple(seq), txn)
        cached = bt_cache.get(key)
        if cached is not None:
            return cached
        positions = stratified_positions(len(seq), exhaustive=exhaustive)
        best_c, best_p = float('inf'), None
        second_c = float('inf')
        for pos in positions:
            cand = seq[:]
            cand.insert(pos, txn)
            c = eval_cost(cand)
            if c < best_c:
                second_c = best_c
                best_c, best_p = c, pos
            elif c < second_c:
                second_c = c
        if second_c == float('inf'):
            second_c = best_c
        res = (best_c, best_p, second_c)
        bt_cache[key] = res
        return res

    # ---------------- GRASP constructor with regret-weighted insertion ----------------
    def select_best_starter():
        cands = random.sample(all_txns, STARTER_SAMPLE) if STARTER_SAMPLE < n else all_txns
        best_t, best_c = None, float('inf')
        for t in cands:
            c = eval_cost([t])
            if c < best_c:
                best_c, best_t = c, t
        return best_t if best_t is not None else random.randint(0, n - 1)

    def construct_regret_insertion():
        bt_cache = make_best_two_cache()
        remaining = set(all_txns)
        seq = [select_best_starter()]
        remaining.remove(seq[0])

        # strong second
        if remaining:
            k = min(6, len(remaining))
            pairs = []
            for t in random.sample(list(remaining), k):
                for pos in [0, 1]:
                    cand = seq[:]
                    cand.insert(pos, t)
                    c = eval_cost(cand)
                    pairs.append((c, t, pos))
            if pairs:
                pairs.sort(key=lambda x: x[0])
                choice = random.choice(pairs[:min(3, len(pairs))])
                _, t, p = choice
                seq.insert(p, t)
                remaining.remove(t)

        while remaining:
            # adaptively sample transactions
            size = len(remaining)
            cand_size = size if size <= (2 * CAND_SAMPLE_BASE) else min(size, max(4, CAND_SAMPLE_BASE + random.randint(-JITTER, JITTER)))
            cand_txns = random.sample(list(remaining), cand_size)

            scored = []
            exhaustive = (len(seq) <= exhaustive_threshold) or (len(remaining) <= 2 * local_beam_width)
            for t in cand_txns:
                best_c, best_p, second_c = best_two_insertions(seq, t, bt_cache, exhaustive=exhaustive)
                regret = max(0.0, second_c - best_c)
                scored.append((best_c, regret, t, best_p))

            scored.sort(key=lambda x: x[0])
            rcl = scored[:min(max(3, RCL_K), len(scored))]
            if random.random() < rcl_alpha:
                # mostly choose low cost among RCL
                rcl.sort(key=lambda x: x[0])
            else:
                # sometimes emphasize higher regret within RCL
                rcl.sort(key=lambda x: (-x[1], x[0]))
            chosen = rcl[0] if random.random() < 0.6 else random.choice(rcl)
            _, _, t_star, p_star = chosen
            seq.insert(p_star, t_star)
            remaining.remove(t_star)

        return eval_cost(seq), seq

    # ---------------- Beam search with adaptive lookahead and diversity ----------------
    def beam_search():
        bw = local_beam_width
        diversity_quota = max(1, int(high_regret_quota_ratio * bw))

        # starters: top singletons + 1 GRASP seed
        starters = [(eval_cost([t]), [t]) for t in all_txns]
        starters.sort(key=lambda x: x[0])
        init_pool = starters[:min(len(starters), max(bw * 2, bw + 2))]
        c0, s0 = construct_regret_insertion()
        init_pool.append((c0, s0))

        beam = []
        seen = set()
        for c, seq in init_pool:
            key = tuple(seq)
            if key in seen:
                continue
            seen.add(key)
            rem = frozenset(t for t in all_txns if t not in seq)
            beam.append((c, seq, rem))
            if len(beam) >= bw:
                break

        best_complete = (float('inf'), [])

        for depth in range(1, n + 1):
            if not beam:
                break
            next_pool = []
            suffix_seen = set()
            endgame = False
            min_remaining = min((len(rem) for (_, _, rem) in beam), default=n)
            if min_remaining <= 2 * bw:
                endgame = True

            for c_so_far, seq, rem in beam:
                if not rem:
                    if c_so_far < best_complete[0]:
                        best_complete = (c_so_far, seq)
                    continue

                rem_list = list(rem)
                # expansion list
                expand_list = rem_list if len(rem_list) <= cand_per_state else random.sample(rem_list, cand_per_state)

                local_moves = []
                spans = []
                # compute immediate + second-step stats
                for t in expand_list:
                    seq1 = seq + [t]
                    c1 = eval_cost(seq1)
                    rem_after = [x for x in rem_list if x != t]

                    best_c2 = c1
                    second_costs = []
                    if rem_after:
                        k2 = len(rem_after) if endgame or len(rem_after) <= 6 else min(lookahead_k, len(rem_after))
                        choices2 = rem_after if k2 == len(rem_after) else random.sample(rem_after, k2)
                        best_c2 = float('inf')
                        for u in choices2:
                            cu = eval_cost(seq1 + [u])
                            second_costs.append(cu)
                            if cu < best_c2:
                                best_c2 = cu
                    span = (max(second_costs) - min(second_costs)) if len(second_costs) >= 2 else 0.0
                    spans.append(span)
                    local_moves.append((c1, best_c2, span, seq1, frozenset(rem_after)))

                if not local_moves:
                    continue

                # adaptive alpha by dispersion
                spans_sorted = sorted(spans)
                median_span = spans_sorted[len(spans_sorted) // 2] if spans_sorted else 0.0
                local_scored = []
                for c1, c2, span, seq1, rem_after in local_moves:
                    a = 0.5 if span > median_span else 0.8  # more lookahead under higher dispersion
                    score = a * c1 + (1.0 - a) * c2
                    local_scored.append((score, c1, span, seq1, rem_after))

                local_scored.sort(key=lambda x: x[0])
                # compute regret on top-2
                best_score = local_scored[0][0]
                second_score = local_scored[1][0] if len(local_scored) > 1 else best_score
                regret = max(0.0, second_score - best_score)

                keep_k = (bw + 2) if endgame else min(6, len(local_scored))
                kept = local_scored[:keep_k]
                for idx, (score, c1, span, seq1, rem_after) in enumerate(kept):
                    # suffix diversity signature
                    if endgame:
                        sig = tuple(seq1[-3:]) if len(seq1) >= 3 else tuple(seq1)
                    else:
                        sig = tuple(seq1[-2:]) if len(seq1) >= 2 else tuple(seq1)
                    # store: (priority score, regret for first option, cost c1, prefix, remaining, suffix sig)
                    next_pool.append((score, regret if idx == 0 else 0.0, c1, seq1, rem_after, sig))
                    if len(seq1) == n and c1 < best_complete[0]:
                        best_complete = (c1, seq1)

            if not next_pool:
                break

            # selection with diversity and regret quota
            next_pool.sort(key=lambda x: x[0])  # by score
            pruned = []
            used_prefix = set()
            used_suffix = set()
            # primary by score
            target_primary = max(1, int((0.6 if endgame else 0.7) * bw))
            for score, rg, c1, seq1, rem1, sig in next_pool:
                key = tuple(seq1)
                if key in used_prefix or sig in used_suffix:
                    continue
                pruned.append((c1, seq1, rem1))
                used_prefix.add(key)
                used_suffix.add(sig)
                if len(pruned) >= target_primary:
                    break
            # regret-boosted fill
            if len(pruned) < bw:
                pool2 = [x for x in next_pool if tuple(x[3]) not in used_prefix]
                pool2.sort(key=lambda x: (-x[1], x[0]))
                quota = min(bw - len(pruned), max(1, int(high_regret_quota_ratio * bw)))
                added = 0
                for score, rg, c1, seq1, rem1, sig in pool2:
                    key = tuple(seq1)
                    if key in used_prefix or sig in used_suffix:
                        continue
                    pruned.append((c1, seq1, rem1))
                    used_prefix.add(key)
                    used_suffix.add(sig)
                    added += 1
                    if added >= quota or len(pruned) >= bw:
                        break
            # final fill if needed
            if len(pruned) < bw:
                for score, rg, c1, seq1, rem1, sig in next_pool:
                    key = tuple(seq1)
                    if key in used_prefix:
                        continue
                    pruned.append((c1, seq1, rem1))
                    used_prefix.add(key)
                    if len(pruned) >= bw:
                        break

            beam = pruned

        # finalize
        for c, seq, rem in beam:
            if not rem and c < best_complete[0]:
                best_complete = (c, seq)
        if best_complete[1] and len(best_complete[1]) == n:
            return best_complete

        if beam:
            # greedy completion
            c, seq, rem = min(beam, key=lambda x: x[0])
            cur = list(seq)
            rem_list = list(rem)
            while rem_list:
                best_t, best_c = None, float('inf')
                for t in rem_list:
                    c2 = eval_cost(cur + [t])
                    if c2 < best_c:
                        best_c, best_t = c2, t
                cur.append(best_t)
                rem_list.remove(best_t)
            return eval_cost(cur), cur

        ident = list(range(n))
        return eval_cost(ident), ident

    # ---------------- Local search: Or-opt, swaps, 2-opt with DLB ----------------
    def or_opt_pass(seq, start_cost, block_len, pos_cap=oropt_pos_cap):
        best_seq = list(seq)
        best_cost = start_cost
        L = len(best_seq)
        if L <= block_len:
            return best_cost, best_seq, False
        improved_any = False
        i = 0
        while i <= L - block_len:
            block = best_seq[i:i + block_len]
            base = best_seq[:i] + best_seq[i + block_len:]
            m = len(base) + 1
            exhaustive = (pos_cap is None) or (m <= (pos_cap + 1))
            positions = list(range(m)) if exhaustive else stratified_positions(len(base), cap=pos_cap, R=rand_pos_R, exhaustive=False)
            move_best_c, move_best_pos = best_cost, None
            for pos in positions:
                if pos == i:
                    continue
                cand = base[:]
                cand[pos:pos] = block
                c = eval_cost(cand)
                if c < move_best_c:
                    move_best_c, move_best_pos = c, pos
            if move_best_pos is not None and move_best_c + 1e-12 < best_cost:
                new_seq = base[:]
                new_seq[move_best_pos:move_best_pos] = block
                best_seq = new_seq
                best_cost = move_best_c
                improved_any = True
                L = len(best_seq)
                i = 0  # restart
            else:
                i += 1
        return best_cost, best_seq, improved_any

    def sampled_pair_swaps_dlb(seq, start_cost, tries=swap_samples):
        best_seq = list(seq)
        best_cost = start_cost
        L = len(best_seq)
        if L <= 3:
            return best_cost, best_seq, False
        improved_any = False
        dont_look = [False] * L
        attempts = 0
        while attempts < tries:
            i = random.randint(0, L - 1)
            if dont_look[i]:
                attempts += 1
                continue
            j = random.randint(0, L - 1)
            if i == j or abs(i - j) <= 1:
                attempts += 1
                continue
            cand = best_seq[:]
            cand[i], cand[j] = cand[j], cand[i]
            c = eval_cost(cand)
            if c < best_cost:
                best_cost, best_seq = c, cand
                improved_any = True
                dont_look = [False] * len(best_seq)
                L = len(best_seq)
                attempts = 0
            else:
                dont_look[i] = True
                attempts += 1
        return best_cost, best_seq, improved_any

    def sampled_two_opt_reversal_dlb(seq, start_cost, tries=two_opt_samples):
        best_seq = list(seq)
        best_cost = start_cost
        L = len(best_seq)
        if L <= 4:
            return best_cost, best_seq, False
        improved_any = False
        dont_look = [False] * L
        attempts = 0
        while attempts < tries:
            i = random.randint(0, L - 3)
            if dont_look[i]:
                attempts += 1
                continue
            j = random.randint(i + 2, min(L - 1, i + 12))  # cap segment length
            cand = best_seq[:]
            cand[i:j + 1] = reversed(cand[i:j + 1])
            c = eval_cost(cand)
            if c < best_cost:
                best_cost, best_seq = c, cand
                improved_any = True
                dont_look = [False] * len(best_seq)
                L = len(best_seq)
                attempts = 0
            else:
                dont_look[i] = True
                attempts += 1
        return best_cost, best_seq, improved_any

    def adjacent_swaps_pass(seq, start_cost, max_passes=2):
        best_seq = list(seq)
        best_cost = start_cost
        improved_any = False
        for _ in range(max_passes):
            improved = False
            for i in range(len(best_seq) - 1):
                cand = best_seq[:]
                cand[i], cand[i + 1] = cand[i + 1], cand[i]
                c = eval_cost(cand)
                if c < best_cost:
                    best_cost, best_seq = c, cand
                    improved = True
                    improved_any = True
            if not improved:
                break
        return best_cost, best_seq, improved_any

    def vnd_local_search(seq, start_cost, rounds=vnd_rounds):
        best_seq = list(seq)
        best_cost = start_cost
        for _ in range(rounds):
            changed = False
            # Or-opt blocks 3,2,1
            for k in (3, 2, 1):
                best_cost, best_seq, imp = or_opt_pass(best_seq, best_cost, k)
                changed = changed or imp
            # Adjacent swaps
            best_cost, best_seq, imp = adjacent_swaps_pass(best_seq, best_cost, max_passes=1)
            changed = changed or imp
            # Non-adjacent swaps (sampled) with DLB
            best_cost, best_seq, imp = sampled_pair_swaps_dlb(best_seq, best_cost, tries=swap_samples)
            changed = changed or imp
            # 2-opt segment reversals (sampled) with DLB
            best_cost, best_seq, imp = sampled_two_opt_reversal_dlb(best_seq, best_cost, tries=two_opt_samples)
            changed = changed or imp
            if not changed:
                break
        return best_cost, best_seq

    # ---------------- LNS: Sensitivity-guided destroy + regret repair ----------------
    def lns_ruin_recreate(seq, start_cost, rounds=lns_rounds_base):
        best_seq = list(seq)
        best_cost = start_cost
        fails = 0
        bt_cache = make_best_two_cache()  # reuse across LNS rounds

        def choose_removal_indices(cur_seq, m):
            L = len(cur_seq)
            if L <= m:
                return set(range(L))
            # Sensitivity on K sampled indices
            K = min(20, L)
            sampled_idx = random.sample(range(L), K)
            # For each idx, try moving to P positions and compute variance in cost
            P = 6
            scores = []
            base_cost = eval_cost(cur_seq)
            for idx in sampled_idx:
                t = cur_seq[idx]
                base = cur_seq[:idx] + cur_seq[idx + 1:]
                mpos = len(base) + 1
                positions = stratified_positions(len(base), cap=10, R=4, exhaustive=(mpos <= 12))
                if len(positions) > P:
                    # pick P representative positions
                    anchors = [0, len(base) // 4, len(base) // 2, (3 * len(base)) // 4, len(base)]
                    choices = sorted(set(p for p in anchors if 0 <= p <= len(base)))
                    need = max(0, P - len(choices))
                    interiors = [p for p in positions if p not in choices]
                    if need > 0 and interiors:
                        choices += random.sample(interiors, min(need, len(interiors)))
                    positions = choices
                vals = []
                for pos in positions[:P]:
                    cand = base[:]
                    cand.insert(pos, t)
                    vals.append(eval_cost(cand))
                if vals:
                    avg = sum(vals) / len(vals)
                    var = sum((x - avg) ** 2 for x in vals) / len(vals)
                else:
                    var = 0.0
                scores.append((var, idx))
            scores.sort(key=lambda x: -x[0])
            take_sensitive = min(len(scores), ceil(0.4 * m))
            sensitive_set = set(idx for _, idx in scores[:take_sensitive])

            # Add one contiguous block for the rest
            remaining = m - len(sensitive_set)
            chosen = set(sensitive_set)
            if remaining > 0:
                block_len = max(2, min(remaining, max(2, L // 12)))
                start = random.randint(0, max(0, L - block_len))
                for j in range(start, start + block_len):
                    if len(chosen) >= m:
                        break
                    chosen.add(j)
            # If short, fill with random extras
            while len(chosen) < m:
                chosen.add(random.randint(0, L - 1))
            return set(sorted(chosen))

        for rr in range(max(1, rounds)):
            L = len(best_seq)
            if L < 6:
                break
            # escalate removal if fails accumulate
            factor = 1.0 + (0.25 * (fails // 2))  # after 2 fails, +25%
            m = int(min(lns_remove_max, max(lns_remove_min, lns_remove_frac * L * factor)))
            m = min(m, L - 2)
            remove_idx = choose_removal_indices(best_seq, m)
            removed = [best_seq[i] for i in sorted(remove_idx)]
            base = [best_seq[i] for i in range(L) if i not in remove_idx]

            # repair by regret-best-two with per-round cache reuse
            rebuilt = list(base)
            remaining = list(removed)
            while remaining:
                # evaluate all or subset
                k_t = len(remaining) if len(remaining) <= 10 else 10
                cand_txns = remaining if len(remaining) <= k_t else random.sample(remaining, k_t)
                scored = []
                exhaustive = (len(rebuilt) <= exhaustive_threshold) or (len(remaining) <= 2 * local_beam_width)
                for t in cand_txns:
                    best_c, best_p, second_c = best_two_insertions(rebuilt, t, bt_cache, exhaustive=exhaustive)
                    regret = max(0.0, second_c - best_c)
                    scored.append((best_c, regret, t, best_p))
                if not scored:
                    t = remaining.pop()
                    rebuilt.append(t)
                    continue
                scored.sort(key=lambda x: x[0])
                rcl_cost = scored[:min(3, len(scored))]
                scored.sort(key=lambda x: (-x[1], x[0]))
                rcl_regret = scored[:min(3, len(scored))]
                pool = list({(c, r, t, p) for (c, r, t, p) in (rcl_cost + rcl_regret)})
                # prefer cost, but keep some regret emphasis
                if random.random() < 0.6:
                    pool.sort(key=lambda x: x[0])
                else:
                    pool.sort(key=lambda x: (-x[1], x[0]))
                c_ins, r_ins, t_ins, p_ins = pool[0]
                rebuilt.insert(p_ins, t_ins)
                remaining.remove(t_ins)

            c_new = eval_cost(rebuilt)
            if c_new < best_cost:
                best_cost, best_seq = c_new, rebuilt
                fails = 0
                # quick polish
                best_cost, best_seq = or_opt_pass(best_seq, best_cost, 1)[0:2]
            else:
                fails += 1

        return best_cost, best_seq

    # ---------------- Elite pool and path relinking ----------------
    elites = []  # list of (cost, seq, suffix3)
    def suffix3(seq):
        return tuple(seq[-3:]) if len(seq) >= 3 else tuple(seq)

    def add_elite(cost, seq):
        nonlocal elites
        key = tuple(seq)
        suf = suffix3(seq)
        # enforce suffix-3 uniqueness
        for i, (c, s, su) in enumerate(elites):
            if tuple(s) == key:
                if cost < c:
                    elites[i] = (cost, list(seq), suffix3(seq))
                return
            if su == suf:
                # keep better suffix twin
                if cost < c:
                    elites[i] = (cost, list(seq), suffix3(seq))
                return
        elites.append((cost, list(seq), suffix3(seq)))
        elites.sort(key=lambda x: x[0])
        if len(elites) > elite_cap:
            elites = elites[:elite_cap]

    def block_aware_path_relink(a_seq, b_seq):
        # Transform a toward b, moving the longest matching block each step where possible
        target_pos = {t: i for i, t in enumerate(b_seq)}
        cur = list(a_seq)
        best_c = eval_cost(cur)
        best_s = list(cur)

        def longest_block_to_place(cur, i):
            # desire block starting at position i in target
            if i >= len(cur):
                return None
            desired_first = b_seq[i]
            if cur[i] == desired_first:
                return None
            # find where desired_first currently is
            j = cur.index(desired_first)
            # extend block length k such that cur[j:j+k] equals b_seq[i:i+k]
            k = 1
            while (j + k < len(cur)) and (i + k < len(b_seq)) and cur[j + k] == b_seq[i + k]:
                k += 1
            return (j, i, k)

        steps = 0
        while steps < len(cur):
            moved = False
            # try block move first
            for i in range(len(cur)):
                blk = longest_block_to_place(cur, i)
                if blk is None:
                    continue
                j, i_target, k = blk
                block = cur[j:j + k]
                base = cur[:j] + cur[j + k:]
                cand = base[:]
                cand[i_target:i_target] = block
                c = eval_cost(cand)
                # fallback: also test single move of desired element only
                # (choose the better between block and single)
                j_single = cur.index(b_seq[i])
                base_single = cur[:j_single] + cur[j_single + 1:]
                cand_single = base_single[:]
                cand_single.insert(i, b_seq[i])
                c_single = eval_cost(cand_single)
                if c_single < c:
                    cand, c = cand_single, c_single
                cur = cand
                steps += 1
                if c < best_c:
                    best_c, best_s = c, list(cur)
                # micro polish: one Or-opt(1) pass
                best_c, best_s = or_opt_pass(best_s, best_c, 1)[0:2]
                cur = list(best_s)
                moved = True
                break
            if not moved:
                # if no block, move single out-of-place element
                improved_any = False
                for i in range(len(cur)):
                    if cur[i] != b_seq[i]:
                        j = cur.index(b_seq[i])
                        elem = cur.pop(j)
                        cur.insert(i, elem)
                        c = eval_cost(cur)
                        steps += 1
                        if c < best_c:
                            best_c, best_s = c, list(cur)
                        best_c, best_s = or_opt_pass(best_s, best_c, 1)[0:2]
                        cur = list(best_s)
                        improved_any = True
                        break
                if not improved_any:
                    break
        return best_c, best_s

    # ---------------- Orchestration ----------------
    random.seed((n * 911 + num_seqs * 131 + 7) % (2**32 - 1))
    best_cost = float('inf')
    best_seq = list(range(n))

    # 1) GRASP seeds + VND polish
    seeds = max(3, min(6, num_seqs))
    for _ in range(seeds):
        c, s = construct_regret_insertion()
        c, s = vnd_local_search(s, c)
        add_elite(c, s)
        if c < best_cost:
            best_cost, best_seq = c, s

    # 2) Beam search seeds
    beam_runs = max(1, num_seqs // 3)
    for _ in range(beam_runs):
        c, s = beam_search()
        c, s = vnd_local_search(s, c)
        add_elite(c, s)
        if c < best_cost:
            best_cost, best_seq = c, s

    # 3) Bidirectional path relinking among elites with micro VND
    if len(elites) >= 2:
        for i in range(min(len(elites), elite_cap)):
            for j in range(i + 1, min(len(elites), elite_cap)):
                _, a, _ = elites[i]
                _, b, _ = elites[j]
                # A -> B
                c_rel, s_rel = block_aware_path_relink(a, b)
                c_rel, s_rel = vnd_local_search(s_rel, c_rel, rounds=1)
                add_elite(c_rel, s_rel)
                if c_rel < best_cost:
                    best_cost, best_seq = c_rel, s_rel
                # B -> A
                c_rel, s_rel = block_aware_path_relink(b, a)
                c_rel, s_rel = vnd_local_search(s_rel, c_rel, rounds=1)
                add_elite(c_rel, s_rel)
                if c_rel < best_cost:
                    best_cost, best_seq = c_rel, s_rel

    # 4) Iterated LNS + VND from incumbent/elite
    incumbent_cost, incumbent_seq = best_cost, best_seq
    for it in range(ils_iterations):
        start_c, start_s = (incumbent_cost, incumbent_seq)
        if elites and random.random() < 0.5:
            start_c, start_s, _ = random.choice(elites[:min(len(elites), elite_cap)])

        c1, s1 = lns_ruin_recreate(start_s, start_c, rounds=lns_rounds_base)
        c2, s2 = vnd_local_search(s1, c1, rounds=vnd_rounds)
        if c2 < incumbent_cost:
            incumbent_cost, incumbent_seq = c2, s2
            add_elite(c2, s2)
        # small polish
        c3, s3 = or_opt_pass(incumbent_seq, incumbent_cost, 1)[0:2]
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