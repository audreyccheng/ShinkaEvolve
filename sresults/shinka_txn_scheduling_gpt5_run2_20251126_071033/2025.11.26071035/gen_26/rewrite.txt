# EVOLVE-BLOCK-START
"""Transaction scheduling algorithm for optimizing makespan across multiple workloads"""

import time
import random
import sys
import os
from statistics import median

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
    Minimize makespan via:
      - regret-guided insertion beam with adaptive lookahead and diversity,
      - stratified position sampling with memoized best-two evaluation,
      - VND local search with Or-opt(1..3), adj swaps, sampled swaps and reversals,
      - sensitivity-guided LNS with escalation,
      - bidirectional block-aware path relinking among elites.

    Args:
        workload: Workload object containing transaction data
        num_seqs: Controls search breadth (seeds, beam width, sampling sizes)

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    n = workload.num_txns
    all_txns = list(range(n))

    # Cost memoization for sequences
    cost_cache = {}
    def seq_cost(seq):
        key = tuple(seq)
        if key in cost_cache:
            return cost_cache[key]
        c = workload.get_opt_seq_cost(seq)
        cost_cache[key] = c
        return c

    # Cache for best and second-best insertion of t into seq
    best_two_cache = {}
    best_two_cache_limit = 12000

    def cache_prune_if_needed():
        if len(best_two_cache) > best_two_cache_limit:
            best_two_cache.clear()

    # Parameters (adaptive)
    # Seeding and elites
    elite_size = max(4, min(8, 2 + num_seqs // 2))
    seed_elite_singletons = max(3, min(8, int(num_seqs)))
    seed_random_additional = max(1, min(4, int((num_seqs + 1) // 3)))

    # Beam and lookahead
    beam_width_base = max(3, min(6, 2 + num_seqs // 2))
    diversity_suffix_len_base = 2
    regret_diversity_quota_base = max(1, beam_width_base // 4)  # ~25%

    k_txn_sample = min(16, max(8, 2 + int(1.5 * num_seqs)))     # txns considered per insertion
    k_pos_sample = min(10, max(6, 2 + int(1.2 * num_seqs)))     # positions to sample
    rem_all_threshold = 14                                       # when few remain, consider all txns

    # Lookahead blend parameters
    alpha_low = 0.8   # mostly immediate cost
    alpha_high = 0.5  # balanced with lookahead
    second_k_txn_sample = 6

    # Local search (VND)
    or_opt_block_sizes = (1, 2, 3)
    pair_swap_samples = min(300, max(80, 3 * n))
    segment_rev_samples = min(180, max(50, 2 * n))

    # Iterated improvement / perturbations
    ils_rounds = max(2, min(5, 1 + num_seqs // 4))
    perturb_swap_count = max(2, min(6, 2 + num_seqs // 3))
    perturb_block_len = max(3, min(10, 3 + num_seqs // 2))

    # LNS
    lns_iters = max(3, min(8, 2 + num_seqs // 2))
    destroy_frac_range = (0.08, 0.18)
    lns_regret_prob = 0.6
    lns_no_improve_escalate_after = 2

    # Path relinking
    pr_max_moves = max(8, min(16, n // 6))

    # Stratified insertion positions (deterministic order)
    def position_samples(seq_len, cap=None):
        if seq_len <= 1:
            return [0, seq_len]
        if seq_len <= 8:
            return list(range(seq_len + 1))
        k = cap if cap is not None else k_pos_sample
        pos_set = set()
        # anchors and quantiles
        anchors = [0, seq_len, seq_len // 2, seq_len // 4, (3 * seq_len) // 4]
        for p in anchors:
            if 0 <= p <= seq_len:
                pos_set.add(p)
        # add evenly spaced points if capacity allows
        if k > len(pos_set):
            step = max(1, seq_len // max(2, k // 2))
            for p in range(0, seq_len + 1, step):
                pos_set.add(p)
                if len(pos_set) >= k:
                    break
        # final guard: add a few randoms if still short
        while len(pos_set) < k:
            pos_set.add(random.randint(0, seq_len))
        return sorted(pos_set)

    def evaluate_best_two_positions(base_seq, t, pos_list):
        """Return (best_cost, best_pos, second_best_cost) for inserting t into base_seq over pos_list."""
        key = (tuple(base_seq), t)
        if key in best_two_cache:
            return best_two_cache[key]
        best = (float('inf'), None)
        second = float('inf')
        for p in pos_list:
            cand = base_seq[:p] + [t] + base_seq[p:]
            c = seq_cost(cand)
            if c < best[0]:
                second = best[0]
                best = (c, p)
            elif c < second:
                second = c
        res = (best[0], best[1], second)
        best_two_cache[key] = res
        cache_prune_if_needed()
        return res

    # Regret-guided insertion beam with adaptive lookahead and diversity
    def build_from_seed(seed_t):
        seq0 = [seed_t]
        rem0 = set(all_txns)
        rem0.remove(seed_t)
        beam = [(seq0, rem0, seq_cost(seq0))]

        while True:
            # If all states are complete, stop
            if all(len(rem) == 0 for _, rem, _ in beam):
                break

            # Adaptive endgame controls
            remaining_sizes = [len(rem) for _, rem, _ in beam]
            rmin = min(remaining_sizes) if remaining_sizes else 0
            beam_width = beam_width_base + (2 if rmin <= 2 * beam_width_base else 0)
            diversity_suffix_len = diversity_suffix_len_base + (1 if rmin <= 2 * beam_width else 0)
            regret_diversity_quota = max(1, int((0.4 if rmin <= 2 * beam_width else 0.25) * beam_width))
            k_main = max(1, beam_width - regret_diversity_quota)

            expansions = []
            layer_regrets = []
            layer_c2s = []

            for seq, rem, base_cost in beam:
                if not rem:
                    expansions.append((seq, rem, seq_cost(seq), 0.0, None, seq_cost(seq)))
                    continue

                # Candidate transactions to insert next
                if len(rem) <= rem_all_threshold:
                    cand_txns = list(rem)
                else:
                    cand_txns = random.sample(list(rem), min(k_txn_sample, len(rem)))

                pos_list = position_samples(len(seq))

                # For each candidate txn, find best and second-best insertion positions and 2-step lookahead
                for t in cand_txns:
                    best_c, best_p, second_c = evaluate_best_two_positions(seq, t, pos_list)
                    new_seq = seq[:best_p] + [t] + seq[best_p:]
                    new_rem = rem.copy()
                    new_rem.remove(t)
                    # Lookahead: sample next txns and positions to estimate c2
                    c2_best = float('inf')
                    if new_rem:
                        if len(new_rem) <= second_k_txn_sample:
                            next_txns = list(new_rem)
                        else:
                            next_txns = random.sample(list(new_rem), second_k_txn_sample)
                        pos_list2 = position_samples(len(new_seq))
                        for u in next_txns:
                            c_u, p_u, _ = evaluate_best_two_positions(new_seq, u, pos_list2)
                            if c_u < c2_best:
                                c2_best = c_u
                    else:
                        c2_best = best_c
                    regret = (second_c - best_c) if second_c < float('inf') else 0.0
                    expansions.append((new_seq, new_rem, best_c, regret, t, c2_best))
                    layer_regrets.append(regret)
                    layer_c2s.append(c2_best)

            if not expansions:
                break

            # Choose lookahead blend weight α for this layer
            spread_c2 = (max(layer_c2s) - min(layer_c2s)) if layer_c2s else 0.0
            tau = median(layer_regrets) if layer_regrets else 0.0
            alpha = alpha_high if spread_c2 > tau else alpha_low

            # Rank expansions by blended lookahead score primarily, then by cost and regret
            def score_exp(e):
                _, _, c1, r, _, c2 = e
                return alpha * c1 + (1.0 - alpha) * c2

            sorted_by_score = sorted(expansions, key=lambda e: (score_exp(e), e[2], -e[3]))
            sorted_by_regret = sorted(expansions, key=lambda e: (-e[3], e[2]))

            # Next beam: mix top-by-score with a quota of high-regret candidates, ensuring suffix diversity
            next_beam = []
            seen_suffix = set()
            seen_seq = set()

            def suffix_sig(s):
                if not s:
                    return (None,)
                if len(s) >= diversity_suffix_len:
                    return tuple(s[-diversity_suffix_len:])
                return tuple([None] * (diversity_suffix_len - len(s)) + s)

            # Fill main portion by blended score
            for seq, rem, cost, regret, t, c2 in sorted_by_score:
                tup = tuple(seq)
                suf = suffix_sig(seq)
                if tup in seen_seq or suf in seen_suffix:
                    continue
                seen_seq.add(tup)
                seen_suffix.add(suf)
                next_beam.append((seq, rem, cost))
                if len(next_beam) >= k_main:
                    break

            # Fill diversity portion by regret
            for seq, rem, cost, regret, t, c2 in sorted_by_regret:
                if len(next_beam) >= beam_width:
                    break
                tup = tuple(seq)
                suf = suffix_sig(seq)
                if tup in seen_seq or suf in seen_suffix:
                    continue
                seen_seq.add(tup)
                seen_suffix.add(suf)
                next_beam.append((seq, rem, cost))

            if not next_beam:
                # Fallback: keep best unique expansions by blended score
                seen = set()
                tmp = []
                for seq, rem, cost, regret, t, c2 in sorted_by_score:
                    tup = tuple(seq)
                    if tup in seen:
                        continue
                    seen.add(tup)
                    tmp.append((seq, rem, cost))
                    if len(tmp) >= beam_width:
                        break
                next_beam = tmp if tmp else [(seq, rem, cost) for seq, rem, cost, _, _, _ in expansions[:beam_width]]

            beam = next_beam

        # Choose the best complete sequence from the beam
        best_seq = None
        best_cost = float('inf')
        for seq, rem, cost in beam:
            if rem:
                seq_complete = seq[:]
                append_rest = sorted(list(rem))
                seq_complete.extend(append_rest)
                cost = seq_cost(seq_complete)
                if cost < best_cost:
                    best_cost = cost
                    best_seq = seq_complete
            else:
                if cost < best_cost:
                    best_cost = cost
                    best_seq = seq

        return best_seq

    # Local refinement: VND with light DLB
    def local_refine(seq):
        best_seq = seq[:]
        best_cost = seq_cost(best_seq)

        # dont-look bits keyed by index starts
        dont_look = [False] * n

        def mark_region(i, width=3):
            lo = max(0, i - width)
            hi = min(n - 1, i + width)
            for k in range(lo, hi + 1):
                dont_look[k] = False

        def try_oropt_block(cur_seq, cur_cost, block_size):
            L = len(cur_seq)
            improved_any = False
            for i in range(0, L - block_size + 1):
                if dont_look[i]:
                    continue
                block = cur_seq[i:i + block_size]
                base = cur_seq[:i] + cur_seq[i + block_size:]
                positions = position_samples(len(base))
                local_improved = False
                for p in positions:
                    cand = base[:p] + block + base[p:]
                    c = seq_cost(cand)
                    if c < cur_cost:
                        cur_seq = cand
                        cur_cost = c
                        improved_any = True
                        local_improved = True
                        mark_region(p)
                        break
                if not local_improved:
                    dont_look[i] = True
                else:
                    # restart scanning after improvement
                    return cur_seq, cur_cost, True
            return cur_seq, cur_cost, improved_any

        def try_adjacent_swap(cur_seq, cur_cost):
            for i in range(n - 1):
                cand = cur_seq[:]
                cand[i], cand[i + 1] = cand[i + 1], cand[i]
                c = seq_cost(cand)
                if c < cur_cost:
                    mark_region(i)
                    return cand, c, True
            return cur_seq, cur_cost, False

        def try_pair_swaps(cur_seq, cur_cost):
            # Sampled non-adjacent swaps
            for _ in range(pair_swap_samples):
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)
                if i == j:
                    continue
                if abs(i - j) == 1:
                    continue
                if i > j:
                    i, j = j, i
                cand = cur_seq[:]
                cand[i], cand[j] = cand[j], cand[i]
                c = seq_cost(cand)
                if c < cur_cost:
                    mark_region((i + j) // 2)
                    return cand, c, True
            return cur_seq, cur_cost, False

        def try_segment_reverse(cur_seq, cur_cost):
            for _ in range(segment_rev_samples):
                i = random.randint(0, n - 2)
                j = random.randint(i + 2, n - 1) if i + 2 < n else None
                if j is None:
                    continue
                cand = cur_seq[:i] + cur_seq[i:j + 1][::-1] + cur_seq[j + 1:]
                c = seq_cost(cand)
                if c < cur_cost:
                    mark_region((i + j) // 2)
                    return cand, c, True
            return cur_seq, cur_cost, False

        improved = True
        while improved:
            improved = False
            # Or-opt blocks: sizes 1, 2, 3 with DLB
            for k in or_opt_block_sizes:
                best_seq, best_cost, did = try_oropt_block(best_seq, best_cost, k)
                if did:
                    improved = True
                    break
            if improved:
                continue
            # Adjacent swaps
            best_seq, best_cost, did = try_adjacent_swap(best_seq, best_cost)
            if did:
                improved = True
                continue
            # Sampled non-adjacent swaps
            best_seq, best_cost, did = try_pair_swaps(best_seq, best_cost)
            if did:
                improved = True
                continue
            # Segment reversals
            best_seq, best_cost, did = try_segment_reverse(best_seq, best_cost)
            if did:
                improved = True
                continue

        return best_cost, best_seq

    # Perturbation for ILS
    def perturb(seq):
        s = seq[:]
        mode = random.random()
        if mode < 0.5:
            # Random swaps
            for _ in range(perturb_swap_count):
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)
                if i != j:
                    s[i], s[j] = s[j], s[i]
        else:
            # Block relocation
            if n > perturb_block_len + 2:
                start = random.randint(0, n - perturb_block_len - 1)
                block = s[start:start + perturb_block_len]
                del s[start:start + perturb_block_len]
                insert_pos = random.randint(0, len(s))
                s = s[:insert_pos] + block + s[insert_pos:]
        return s

    # Sensitivity-guided LNS
    def lns_attempt(seq, no_improve_rounds):
        cur = seq[:]
        # Choose destroy size, escalate if stagnated
        frac = random.uniform(*destroy_frac_range)
        if no_improve_rounds >= lns_no_improve_escalate_after:
            frac *= 1.25
        m = max(4, min(n // 2, int(frac * n)))

        # Sensitivity scoring
        K = min(20, n)
        P = 6
        sampled_idxs = sorted(random.sample(range(n), K)) if n > 0 else []
        sens_scores = []
        for idx in sampled_idxs:
            item = cur[idx]
            base = cur[:idx] + cur[idx + 1:]
            pos_list = position_samples(len(base))
            if len(pos_list) > P:
                # stratified subsample
                step = max(1, len(pos_list) // P)
                pos_list_s = [pos_list[i] for i in range(0, len(pos_list), step)][:P]
            else:
                pos_list_s = pos_list
            vals = []
            for p in pos_list_s:
                cand = base[:p] + [item] + base[p:]
                vals.append(seq_cost(cand))
            if vals:
                mean = sum(vals) / len(vals)
                var = sum((x - mean) ** 2 for x in vals) / len(vals)
                sens_scores.append((var, idx))
        sens_scores.sort(reverse=True)

        # Removal set: one or two contiguous blocks + top sensitivity indices
        remove_idxs = set()
        # primary block
        block_len = max(3, int(0.4 * m))
        start = random.randint(0, n - block_len) if n > block_len else 0
        for i in range(start, min(n, start + block_len)):
            remove_idxs.add(i)

        # optional second block on escalation
        if no_improve_rounds >= lns_no_improve_escalate_after and len(remove_idxs) < m:
            block_len2 = max(2, int(0.25 * m))
            start2 = random.randint(0, n - block_len2) if n > block_len2 else 0
            for i in range(start2, min(n, start2 + block_len2)):
                remove_idxs.add(i)

        # fill remaining using high-sensitivity indices
        for _, idx in sens_scores:
            if len(remove_idxs) >= m:
                break
            remove_idxs.add(idx)

        # If still short (small n), add random indices
        while len(remove_idxs) < m:
            remove_idxs.add(random.randint(0, n - 1))
        remove_idxs = sorted(remove_idxs)[:m]

        remove_set = set(remove_idxs)
        removed = [cur[i] for i in remove_idxs]
        remaining = [cur[i] for i in range(n) if i not in remove_set]

        # Repair using regret-aware insertion
        seq_rep = remaining[:]
        rem_set = removed[:]
        while rem_set:
            if len(rem_set) > k_txn_sample:
                cand_txns = random.sample(rem_set, k_txn_sample)
            else:
                cand_txns = rem_set[:]

            best_overall = (float('inf'), None, None)  # cost, txn, pos
            best_by_regret = (float('-inf'), None, None)  # regret, txn, pos

            pos_list = position_samples(len(seq_rep))
            for t in cand_txns:
                best_c, best_p, second_c = evaluate_best_two_positions(seq_rep, t, pos_list)
                regret = (second_c - best_c) if second_c < float('inf') else 0.0
                if best_c < best_overall[0]:
                    best_overall = (best_c, t, best_p)
                if regret > best_by_regret[0]:
                    best_by_regret = (regret, t, best_p)

            pick_regret = (random.random() < lns_regret_prob)
            chosen = best_by_regret if pick_regret and best_by_regret[1] is not None else best_overall
            t = chosen[1]
            p = chosen[2] if chosen[2] is not None else len(seq_rep)
            if t is None:
                t = random.choice(rem_set)
                p = len(seq_rep)
            seq_rep = seq_rep[:p] + [t] + seq_rep[p:]
            rem_set.remove(t)

        c_rep, s_rep = local_refine(seq_rep)
        return c_rep, s_rep

    # Path Relinking: bidirectional, block-aware, with quick polish
    def path_relink_bidirectional(A, B, max_moves=pr_max_moves):
        def relink(source, target):
            pos_in_source = {t: i for i, t in enumerate(source)}
            s = source[:]
            best_c = seq_cost(s)
            best_s = s[:]
            moves = 0
            i = 0
            while i < n and moves < max_moves:
                if s[i] == target[i]:
                    i += 1
                    continue
                # find block [j, j+L) in s that matches target[i:i+L]
                j = s.index(target[i])
                L = 1
                while i + L < n and j + L < n and s[j + L] == target[i + L]:
                    L += 1
                block = s[j:j + L]
                base = s[:j] + s[j + L:]
                cand = base[:i] + block + base[i:]
                # quick Or-opt(1) on cand: move one element (block of size 1) greedily if helps
                c_cand = seq_cost(cand)
                if len(cand) >= 2:
                    for k in range(max(0, i - 2), min(len(cand) - 1, i + 2)):
                        item = cand[k]
                        base2 = cand[:k] + cand[k + 1:]
                        positions = position_samples(len(base2), cap=6)
                        for p in positions:
                            cand2 = base2[:p] + [item] + base2[p:]
                            c2 = seq_cost(cand2)
                            if c2 < c_cand:
                                cand = cand2
                                c_cand = c2
                                break
                s = cand
                if c_cand < best_c:
                    best_c = c_cand
                    best_s = cand[:]
                moves += 1
                i += L
            return best_c, best_s

        c1, s1 = relink(A, B)
        c2, s2 = relink(B, A)
        if c1 <= c2:
            return c1, s1
        return c2, s2

    # Elite management
    elite = []  # list of (cost, seq)

    def add_elite(c, s):
        nonlocal elite
        elite.append((c, s[:]))
        elite.sort(key=lambda x: x[0])
        # maintain diversity by last-3 suffix uniqueness
        uniq = []
        seen_suffix = set()
        for c1, s1 in elite:
            suf = tuple(s1[-3:]) if len(s1) >= 3 else tuple(s1)
            if suf in seen_suffix:
                continue
            seen_suffix.add(suf)
            uniq.append((c1, s1))
            if len(uniq) >= elite_size:
                break
        elite = uniq

    # Seeding: all singletons + random seeds
    singleton_scores = []
    for t in all_txns:
        singleton_scores.append((seq_cost([t]), t))
    singleton_scores.sort(key=lambda x: x[0])
    seed_txns = [t for _, t in singleton_scores[:seed_elite_singletons]]
    remaining_txns = [t for t in all_txns if t not in seed_txns]
    if remaining_txns and seed_random_additional > 0:
        seed_txns.extend(random.sample(remaining_txns, min(seed_random_additional, len(remaining_txns))))

    # Build from seeds, refine, and populate elites
    for seed in seed_txns:
        seq0 = build_from_seed(seed)
        c1, s1 = local_refine(seq0)
        add_elite(c1, s1)

    # Fallback if no elite
    if not elite:
        base = all_txns[:]
        random.shuffle(base)
        elite = [(seq_cost(base), base)]

    best_overall_cost, best_overall_seq = elite[0]

    # Iterated local search
    cur_cost, cur_seq = best_overall_cost, best_overall_seq[:]
    for _ in range(ils_rounds):
        pert = perturb(cur_seq)
        c2, s2 = local_refine(pert)
        if c2 < cur_cost:
            cur_cost, cur_seq = c2, s2
            add_elite(c2, s2)
            if c2 < best_overall_cost:
                best_overall_cost, best_overall_seq = c2, s2

    # LNS with escalation on stagnation
    no_improve = 0
    for _ in range(lns_iters):
        c3, s3 = lns_attempt(best_overall_seq, no_improve)
        if c3 < best_overall_cost:
            best_overall_cost, best_overall_seq = c3, s3
            add_elite(c3, s3)
            no_improve = 0
        else:
            no_improve += 1

    # Path relinking among elites
    if len(elite) >= 2:
        base_cost, base_seq = best_overall_cost, best_overall_seq[:]
        partners = elite[1:min(len(elite), elite_size)]
        for c_t, s_t in partners:
            pr_c, pr_s = path_relink_bidirectional(base_seq, s_t, max_moves=pr_max_moves)
            if pr_c < best_overall_cost:
                best_overall_cost, best_overall_seq = pr_c, pr_s

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