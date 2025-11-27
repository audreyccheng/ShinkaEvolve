# EVOLVE-BLOCK-START
"""Transaction scheduling algorithm for optimizing makespan across multiple workloads"""

import time
import random
import sys
import os
from math import sqrt, ceil

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
    Find a low-makespan schedule using regret-based insertion beam search with
    best-two insertion memoization and deterministic position sampling,
    followed by VND with don't-look bits, iterated local search, sensitivity-guided LNS,
    and elite path relinking.

    Args:
        workload: Workload object containing transaction data
        num_seqs: Controls search breadth (seeds, beam width, sampling sizes)

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    n = workload.num_txns
    all_txns = list(range(n))

    # Memoization to avoid repeated cost computations for the same sequence
    cost_cache = {}
    def seq_cost(seq):
        key = tuple(seq)
        if key in cost_cache:
            return cost_cache[key]
        c = workload.get_opt_seq_cost(seq)
        cost_cache[key] = c
        return c

    # ----------------------------
    # Parameters (adaptive)
    # ----------------------------
    # Elite and seeds
    elite_size = max(4, min(7, 2 + num_seqs // 2))
    seed_elite_singletons = max(3, min(8, int(0.7 * num_seqs)))
    seed_random_additional = max(1, min(5, int((num_seqs + 1) // 3)))

    # Construction parameters
    build_beam_width_init = max(4, min(8, 2 + num_seqs // 2))
    endgame_beam_bonus = 2
    regret_high_ratio = 0.35
    diversity_suffix_k_base = 2
    endgame_suffix_k = 3

    # Candidate sampling sizes
    k_txn_sample = min(16, max(8, 2 + int(1.5 * num_seqs)))  # txns per insertion
    k_pos_sample_base = min(10, max(6, 2 + int(1.2 * num_seqs)))
    pos_ring_radius = 3
    all_pos_threshold = 12  # when seq small or few remaining, use all positions

    # Local search parameters (VND)
    ls_adj_rounds_max = 2
    swap_samples = min(400, max(100, 3 * n))
    two_opt_samples = min(200, max(60, 2 * n))

    # Iterated local search (ILS) / perturbation
    ils_rounds = max(2, min(6, 1 + num_seqs // 3))
    perturb_swap_count = max(2, min(6, 2 + num_seqs // 3))
    perturb_block_len = max(3, min(10, 3 + num_seqs // 2))

    # Large Neighborhood Search (LNS)
    lns_iters = max(3, min(7, 2 + num_seqs // 3))
    destroy_frac_range = (0.10, 0.20)
    lns_stagnation_escalation = 2  # after these failures, increase destroy size and allow double-block
    sensitivity_positions = 6  # positions to test per item to score sensitivity

    # Path relinking
    pr_moves = max(10, min(18, n // 6))

    # ----------------------------
    # Best-two insertion memoization
    # ----------------------------
    # Cache results per (tuple(seq), t, tuple(pos_list))
    best_two_cache = {}

    def evaluate_best_two_positions(base_seq, t, pos_list):
        """Return (best_cost, best_pos, second_best_cost) for inserting t into base_seq over pos_list."""
        key = (tuple(base_seq), t, tuple(pos_list))
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
        return res

    # ----------------------------
    # Deterministic, stratified position sampling
    # ----------------------------
    def deterministic_positions(seq_len, focus_idx=None, cap=None, seed_tuple=None):
        """Deterministic, stratified position samples including anchors, quarters, and a near-focus ring."""
        if seq_len <= 1:
            return [0, seq_len]
        # All positions when small for accuracy
        if seq_len <= all_pos_threshold:
            return list(range(seq_len + 1))
        k = cap if cap is not None else k_pos_sample_base
        pos_set = {0, seq_len, seq_len // 2, seq_len // 4, (3 * seq_len) // 4}
        if focus_idx is not None:
            for d in range(-pos_ring_radius, pos_ring_radius + 1):
                p = focus_idx + d
                if 0 <= p <= seq_len:
                    pos_set.add(p)
        # Low-discrepancy-like deterministic fill using a hashed RNG
        rnd = random.Random((hash(seed_tuple if seed_tuple is not None else (seq_len, focus_idx)) & 0xffffffff))
        tries = min(k, seq_len + 1)
        for _ in range(tries):
            pos_set.add(rnd.randint(0, seq_len))
        return sorted(pos_set)

    # ----------------------------
    # Construction (regret-insertion with adaptive beam and diversity)
    # ----------------------------
    def regret_insertion_build(seed_t=None):
        if seed_t is None:
            seed_t = random.randint(0, n - 1)

        # Beam holds tuples: (seq, rem_set, cost)
        seq0 = [seed_t]
        rem0 = set(all_txns)
        rem0.remove(seed_t)
        beam = [(seq0, rem0, seq_cost(seq0))]

        diversity_suffix_k = diversity_suffix_k_base

        while True:
            if all(len(rem) == 0 for _, rem, _ in beam):
                break

            expansions = []
            seen_seqs = set()

            # Decide if we are in endgame (few remaining overall across beam)
            rem_counts = [len(rem) for _, rem, _ in beam]
            min_rem = min(rem_counts) if rem_counts else 0
            endgame = (min_rem <= max(6, build_beam_width_init))
            local_beam_width = build_beam_width_init + (endgame_beam_bonus if endgame else 0)
            diversity_suffix_k = endgame_suffix_k if endgame else diversity_suffix_k_base

            for seq, rem, _ in beam:
                if not rem:
                    key = tuple(seq)
                    if key not in seen_seqs:
                        seen_seqs.add(key)
                        expansions.append((seq, rem, seq_cost(seq), 0.0, None))
                    continue

                # Candidate transactions
                if len(rem) <= max(12, local_beam_width * 2):
                    cand_txns = list(rem)
                else:
                    cand_txns = random.sample(list(rem), min(k_txn_sample, len(rem)))

                # Position policy (all vs deterministic sampled)
                use_all_pos = endgame or (len(seq) <= all_pos_threshold)
                for t in cand_txns:
                    if use_all_pos:
                        pos_list = list(range(len(seq) + 1))
                    else:
                        seed_tuple = (tuple(seq[-min(10, len(seq)):]), t, len(seq))
                        pos_list = deterministic_positions(len(seq), focus_idx=None, cap=k_pos_sample_base, seed_tuple=seed_tuple)

                    best_c, best_p, second_c = evaluate_best_two_positions(seq, t, pos_list)
                    new_seq = seq[:best_p] + [t] + seq[best_p:]
                    new_rem = rem.copy()
                    new_rem.remove(t)
                    regret = (second_c - best_c) if second_c < float('inf') else 0.0
                    key = tuple(new_seq)
                    if key in seen_seqs:
                        continue
                    seen_seqs.add(key)
                    expansions.append((new_seq, new_rem, best_c, regret, t))

            if not expansions:
                break

            # Compute regret dispersion to adapt selection pressure
            regrets = [e[3] for e in expansions]
            if regrets:
                r_min, r_max = min(regrets), max(regrets)
                dispersion = r_max - r_min
            else:
                dispersion = 0.0

            # Rank expansions with adaptive weighting
            # When dispersion large -> emphasize regret more; else prioritize cost
            if dispersion > (sum(regrets) / len(regrets) if regrets else 0.0):
                # 50/50 weighting
                expansions.sort(key=lambda x: (0.5 * x[2] - 0.5 * x[3], x[2]))
                regret_quota = max(1, int(regret_high_ratio * local_beam_width))
            else:
                expansions.sort(key=lambda x: (x[2], -x[3]))
                regret_quota = max(1, max(1, int(0.25 * local_beam_width)))

            # Diversity-aware beam selection
            next_beam = []
            seen_suffix = set()

            def suffix_sig(s):
                k = min(len(s), diversity_suffix_k)
                return tuple(s[-k:]) if k > 0 else ()

            # Base fill by cost-oriented order
            i = 0
            while len(next_beam) < max(1, local_beam_width - regret_quota) and i < len(expansions):
                s, r, c, reg, _t = expansions[i]
                sig = suffix_sig(s)
                if sig not in seen_suffix:
                    seen_suffix.add(sig)
                    next_beam.append((s, r, c))
                i += 1

            # Add high-regret candidates for exploration
            by_regret = sorted(expansions, key=lambda x: (-x[3], x[2]))
            j = 0
            while len(next_beam) < local_beam_width and j < len(by_regret):
                s, r, c, reg, _t = by_regret[j]
                sig = suffix_sig(s)
                if sig not in seen_suffix:
                    seen_suffix.add(sig)
                    next_beam.append((s, r, c))
                j += 1

            if not next_beam:
                # Fallback: take top-k by cost
                next_beam = [(s, r, c) for s, r, c, _, _ in expansions[:local_beam_width]]

            beam = next_beam

        # Choose best complete
        best_seq = None
        best_cost = float('inf')
        for seq, rem, cost in beam:
            if rem:
                seq_complete = seq[:] + sorted(list(rem))
                c = seq_cost(seq_complete)
                if c < best_cost:
                    best_cost = c
                    best_seq = seq_complete
            else:
                if cost < best_cost:
                    best_cost = cost
                    best_seq = seq
        return best_seq

    # ----------------------------
    # Local refinement: VND with don't-look bits, Or-opt (1..3), swaps, 2-opt
    # ----------------------------
    def local_refine(seq):
        best_seq = seq[:]
        best_cost = seq_cost(best_seq)

        # don't-look bits per index
        dont_look = [0] * n
        dl_epoch = 1

        def mark_reset_around(i, j=None):
            # reset don't-look bits around modified positions
            r = 3
            L = len(dont_look)
            for idx in range(max(0, i - r), min(L, i + r + 1)):
                dont_look[idx] = 0
            if j is not None:
                for idx in range(max(0, j - r), min(L, j + r + 1)):
                    dont_look[idx] = 0

        def reinsertion_pass(cur_seq, cur_cost, k_block=1):
            L = len(cur_seq)
            improved = False
            # Best-improvement per outer sweep
            best_local_delta = 0.0
            best_local = None
            for i in range(0, L - k_block + 1):
                if dont_look[i] >= dl_epoch:
                    continue
                block = cur_seq[i:i + k_block]
                base = cur_seq[:i] + cur_seq[i + k_block:]
                # Deterministic positions biased around i
                pos_list = deterministic_positions(len(base), focus_idx=i, cap=k_pos_sample_base, seed_tuple=(k_block, i, len(base)))
                # Evaluate
                for p in pos_list:
                    if p == i:
                        continue
                    cand = base[:p] + block + base[p:]
                    c = seq_cost(cand)
                    delta = c - cur_cost
                    if delta < best_local_delta:
                        best_local_delta = delta
                        best_local = (cand, c, i, p, k_block)
            if best_local is not None:
                cand, c, i, p, kb = best_local
                mark_reset_around(i)
                return c, cand, True
            # If no move improved, mark all scanned as don't-look
            for i in range(0, L - k_block + 1):
                dont_look[i] = dl_epoch
            return cur_cost, cur_seq, improved

        def adjacent_swap_pass(cur_seq, cur_cost):
            L = len(cur_seq)
            for i in range(L - 1):
                if dont_look[i] >= dl_epoch:
                    continue
                cand = cur_seq[:]
                cand[i], cand[i + 1] = cand[i + 1], cand[i]
                c = seq_cost(cand)
                if c < cur_cost:
                    mark_reset_around(i, i + 1)
                    return c, cand, True
                else:
                    dont_look[i] = dl_epoch
            return cur_cost, cur_seq, False

        def sampled_swaps_pass(cur_seq, cur_cost):
            L = len(cur_seq)
            improved = False
            for _ in range(swap_samples):
                i = random.randint(0, L - 1)
                j = random.randint(0, L - 1)
                if i == j or abs(i - j) == 1:
                    continue
                if dont_look[i] >= dl_epoch and dont_look[j] >= dl_epoch:
                    continue
                if i > j:
                    i, j = j, i
                cand = cur_seq[:]
                cand[i], cand[j] = cand[j], cand[i]
                c = seq_cost(cand)
                if c < cur_cost:
                    mark_reset_around(i, j)
                    return c, cand, True
            return cur_cost, cur_seq, improved

        def sampled_2opt_pass(cur_seq, cur_cost):
            L = len(cur_seq)
            for _ in range(two_opt_samples):
                i = random.randint(0, L - 2)
                j = random.randint(i + 2, L - 1)
                if dont_look[i] >= dl_epoch and dont_look[j] >= dl_epoch:
                    continue
                cand = cur_seq[:i] + cur_seq[i:j + 1][::-1] + cur_seq[j + 1:]
                c = seq_cost(cand)
                if c < cur_cost:
                    mark_reset_around(i, j)
                    return c, cand, True
            return cur_cost, cur_seq, False

        improved_outer = True
        adj_rounds = 0
        while improved_outer:
            improved_outer = False

            # Or-opt reinsertion blocks in descending size (3,2,1) for big moves early
            for kb in (3, 2, 1):
                c2, s2, changed = reinsertion_pass(best_seq, best_cost, k_block=kb)
                if changed:
                    best_seq, best_cost = s2, c2
                    improved_outer = True
                    break
            if improved_outer:
                continue

            # Adjacent swaps (limited rounds)
            if adj_rounds < ls_adj_rounds_max:
                c2, s2, changed = adjacent_swap_pass(best_seq, best_cost)
                if changed:
                    best_seq, best_cost = s2, c2
                    improved_outer = True
                    continue
                adj_rounds += 1

            # Sampled non-adjacent swaps
            c2, s2, changed = sampled_swaps_pass(best_seq, best_cost)
            if changed:
                best_seq, best_cost = s2, c2
                improved_outer = True
                continue

            # Sampled 2-opt segment reversals
            c2, s2, changed = sampled_2opt_pass(best_seq, best_cost)
            if changed:
                best_seq, best_cost = s2, c2
                improved_outer = True
                continue

        return best_cost, best_seq

    # ----------------------------
    # Perturbation methods for ILS
    # ----------------------------
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
            # Block shuffle
            if n > perturb_block_len + 2:
                start = random.randint(0, n - perturb_block_len - 1)
                block = s[start:start + perturb_block_len]
                del s[start:start + perturb_block_len]
                insert_pos = random.randint(0, len(s))
                s = s[:insert_pos] + block + s[insert_pos:]
        return s

    # ----------------------------
    # Sensitivity-guided LNS destroy-and-repair with escalation
    # ----------------------------
    def score_sensitivity(seq):
        # Return list of (score, idx). Higher score = higher sensitivity.
        L = len(seq)
        scores = []
        base_cost = seq_cost(seq)
        for i in range(L):
            item = seq[i]
            base = seq[:i] + seq[i + 1:]
            pos_list = deterministic_positions(len(base), focus_idx=i, cap=sensitivity_positions, seed_tuple=('sens', i, item))
            deltas = []
            for p in pos_list:
                cand = base[:p] + [item] + base[p:]
                c = seq_cost(cand)
                deltas.append(c - base_cost)
            if deltas:
                mean_delta = sum(deltas) / len(deltas)
                var = sum((d - mean_delta) ** 2 for d in deltas) / len(deltas)
                score = (var + abs(mean_delta))  # combine dispersion and shift
            else:
                score = 0.0
            scores.append((score, i))
        scores.sort(reverse=True)
        return scores

    def quick_polish(seq):
        # Single light pass: reinsertion k=1 with deterministic positions
        L = len(seq)
        best_seq = seq[:]
        best_cost = seq_cost(best_seq)
        for i in range(L):
            item = best_seq[i]
            base = best_seq[:i] + best_seq[i + 1:]
            pos_list = deterministic_positions(len(base), focus_idx=i, cap=k_pos_sample_base, seed_tuple=('qp', i, item))
            for p in pos_list:
                cand = base[:p] + [item] + base[p:]
                if cand == best_seq:
                    continue
                c = seq_cost(cand)
                if c < best_cost:
                    best_cost = c
                    best_seq = cand
                    break
        return best_cost, best_seq

    def lns_attempt(seq, escalation=False):
        cur = seq[:]
        L = len(cur)
        frac = random.uniform(*destroy_frac_range)
        if escalation:
            frac = min(0.35, frac + 0.10)
        m = max(4, min(L // 2, int(frac * L)))

        # Sensitivity-guided indices
        sens = score_sensitivity(cur)
        top_k = ceil(0.4 * m)
        remove_high = [idx for _, idx in sens[:min(top_k, L)]]

        remaining_indices = [i for i in range(L) if i not in set(remove_high)]
        # Mix in a contiguous block
        if not remaining_indices:
            remaining_indices = list(range(L))
        start = random.randint(0, max(0, L - (m - len(remove_high))))
        remove_block = list(range(start, min(L, start + (m - len(remove_high)))))
        remove_idxs = sorted(set(remove_high + remove_block))[:m]

        removed = [cur[i] for i in remove_idxs]
        remaining = [cur[i] for i in range(L) if i not in set(remove_idxs)]

        # Repair using regret insertion with memoized best-two
        seq_rep = remaining[:]
        rem_set = removed[:]
        while rem_set:
            if len(rem_set) > k_txn_sample:
                cand_txns = random.sample(rem_set, k_txn_sample)
            else:
                cand_txns = rem_set[:]

            best_overall = (float('inf'), None, None)  # cost, txn, new_seq
            best_by_regret = (float('-inf'), None, None)

            for t in cand_txns:
                # Deterministic positions per t
                pos_list = deterministic_positions(len(seq_rep), focus_idx=None, cap=k_pos_sample_base, seed_tuple=('lns', len(seq_rep), t))
                best_c, best_p, second_c = evaluate_best_two_positions(seq_rep, t, pos_list)
                new_seq = seq_rep[:best_p] + [t] + seq_rep[best_p:]
                if best_c < best_overall[0]:
                    best_overall = (best_c, t, new_seq)
                regret = second_c - best_c if second_c < float('inf') else 0.0
                if regret > best_by_regret[0]:
                    best_by_regret = (regret, t, new_seq)

            # Pick by regret when it provides diversity; else pure best
            chosen = best_by_regret if best_by_regret[1] is not None and best_by_regret[0] > 0 else best_overall
            if chosen[1] is None:
                t = random.choice(rem_set)
                seq_rep = seq_rep + [t]
                rem_set.remove(t)
            else:
                seq_rep = chosen[2]
                rem_set.remove(chosen[1])

        # Quick polish then full local refine
        c_q, s_q = quick_polish(seq_rep)
        c_rep, s_rep = local_refine(s_q)
        return c_rep, s_rep

    # ----------------------------
    # Path Relinking (bidirectional, block-aware) with micro-polish
    # ----------------------------
    def path_relink(A, B, max_moves=12):
        # Move source towards target; at each step, place either the item or a matching block
        pos_in_B = {t: i for i, t in enumerate(B)}
        s = A[:]
        best_c = seq_cost(s)
        best_s = s[:]

        moves = 0
        # order items by displacement
        displacement = [(abs(i - pos_in_B[s[i]]), i) for i in range(n)]
        displacement.sort(reverse=True)

        while moves < max_moves and displacement:
            _, idx = displacement.pop(0)
            if idx >= len(s):
                continue
            item = s[idx]
            desired = pos_in_B[item]

            if desired == idx:
                continue

            # Try to move the longest block already matching B starting at idx
            block_len = 1
            while idx + block_len <= len(s) and desired + block_len <= len(B) and s[idx + block_len - 1] == B[desired + block_len - 1]:
                block_len += 1
            block_len -= 1
            # Move block
            block = s[idx:idx + block_len] if block_len > 0 else [item]
            base = s[:idx] + s[idx + (block_len if block_len > 0 else 1):]
            insert_pos = min(desired, len(base))
            cand = base[:insert_pos] + block + base[insert_pos:]
            # Micro-polish with single reinsertion around the modified area
            c = seq_cost(cand)
            if c < best_c:
                best_c, best_s = c, cand[:]
            s = cand
            moves += 1

        # final light polish
        c2, s2 = quick_polish(best_s)
        return (c2, s2) if c2 < best_c else (best_c, best_s)

    # ----------------------------
    # Elite management
    # ----------------------------
    elite = []  # list of (cost, seq)
    def add_elite(c, s):
        nonlocal elite
        elite.append((c, s))
        elite.sort(key=lambda x: x[0])
        # enforce suffix-3 uniqueness
        uniq = []
        seen_sig = set()
        for c1, s1 in elite:
            sig = tuple(s1[-3:]) if len(s1) >= 3 else tuple(s1)
            if sig in seen_sig:
                continue
            seen_sig.add(sig)
            uniq.append((c1, s1))
            if len(uniq) >= elite_size:
                break
        elite = uniq

    # ----------------------------
    # Seeding
    # ----------------------------
    singleton_scores = []
    for t in all_txns:
        singleton_scores.append((seq_cost([t]), t))
    singleton_scores.sort(key=lambda x: x[0])

    seed_txns = [t for _, t in singleton_scores[:seed_elite_singletons]]
    remaining = [t for t in all_txns if t not in seed_txns]
    if remaining and seed_random_additional > 0:
        extra = random.sample(remaining, min(seed_random_additional, len(remaining)))
        seed_txns.extend(extra)

    best_overall_cost = float('inf')
    best_overall_seq = None

    for seed in seed_txns:
        seq0 = regret_insertion_build(seed)
        c1, s1 = local_refine(seq0)
        add_elite(c1, s1)
        if c1 < best_overall_cost:
            best_overall_cost, best_overall_seq = c1, s1

    # Fallback if needed
    if best_overall_seq is None:
        best_overall_seq = all_txns[:]
        random.shuffle(best_overall_seq)
        best_overall_cost = seq_cost(best_overall_seq)
        add_elite(best_overall_cost, best_overall_seq)

    # ----------------------------
    # Iterated Local Search (ILS)
    # ----------------------------
    cur_cost, cur_seq = best_overall_cost, best_overall_seq[:]
    for _ in range(ils_rounds):
        pert = perturb(cur_seq)
        c2, s2 = local_refine(pert)
        if c2 < cur_cost:
            cur_cost, cur_seq = c2, s2
            add_elite(c2, s2)
            if c2 < best_overall_cost:
                best_overall_cost, best_overall_seq = c2, s2

    # ----------------------------
    # LNS destroy-and-repair with escalation on stagnation
    # ----------------------------
    stagnation = 0
    for _ in range(lns_iters):
        escalate = (stagnation >= lns_stagnation_escalation)
        c3, s3 = lns_attempt(best_overall_seq, escalation=escalate)
        if c3 < best_overall_cost:
            best_overall_cost, best_overall_seq = c3, s3
            add_elite(c3, s3)
            stagnation = 0
        else:
            stagnation += 1

    # ----------------------------
    # Path Relinking among elites (bidirectional)
    # ----------------------------
    if len(elite) >= 2:
        base_cost, base_seq = best_overall_cost, best_overall_seq
        partners = elite[1:min(len(elite), elite_size)]
        for c_t, s_t in partners:
            pr1_c, pr1_s = path_relink(base_seq, s_t, max_moves=pr_moves)
            if pr1_c < best_overall_cost:
                best_overall_cost, best_overall_seq = pr1_c, pr1_s
            pr2_c, pr2_s = path_relink(s_t, base_seq, max_moves=pr_moves)
            if pr2_c < best_overall_cost:
                best_overall_cost, best_overall_seq = pr2_c, pr2_s

    return best_overall_cost, best_overall_seq


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