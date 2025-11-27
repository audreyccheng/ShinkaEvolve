# EVOLVE-BLOCK-START
"""Transaction scheduling algorithm for optimizing makespan across multiple workloads"""

import time
import random
import sys
import os
from collections import OrderedDict

# Add the openevolve_examples directory to the path to import txn_simulator and workloads
# Find the repository root by looking for the openevolve_examples directory
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
    adaptive diversification and best-two LRU memoization, followed by strong
    local refinement (VND with don't-look bits), sensitivity-guided LNS, and
    bidirectional path relinking over a small elite pool.

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
    # Parameters (adaptive/tuned)
    # ----------------------------
    elite_size = max(3, min(6, 2 + num_seqs // 3))
    elite_suffix_k = 3  # elite diversity by suffix-3

    seed_elite_singletons = max(2, min(6, int(num_seqs)))
    seed_random_additional = max(1, min(4, int((num_seqs + 1) // 3)))

    # Candidate sampling sizes
    k_txn_sample = min(16, max(8, 2 + int(1.5 * num_seqs)))   # txns per insertion attempt
    k_pos_sample = min(10, max(6, 2 + int(1.2 * num_seqs)))   # positions per insertion
    rem_all_threshold = 14

    # Beam construction parameters
    base_beam_width = max(3, min(6, 2 + num_seqs // 2))
    diversity_suffix_k = 2
    endgame_all_pos_threshold = max(6, min(12, num_seqs))

    # Local search parameters
    ls_adj_rounds_max = 2
    two_opt_pairs = min(200, max(40, n))     # sampled 2-opt pairs (segment reversals)
    swap_samples = min(400, max(60, 4 * n))  # sampled non-adjacent swaps

    # Iterated local search (ILS) / perturbation
    ils_rounds = max(2, min(5, 1 + num_seqs // 4))
    perturb_swap_count = max(2, min(6, 2 + num_seqs // 3))
    perturb_block_len = max(3, min(10, 3 + num_seqs // 2))

    # LNS parameters
    lns_iters = max(2, min(6, 2 + num_seqs // 3))
    destroy_frac_range = (0.08, 0.18)
    sensitivity_K = min(20, n)
    sensitivity_pos_P = 6
    sensitivity_pick_ratio = 0.4

    # Path relinking
    pr_max_moves = max(8, min(12, n // 8))

    # ----------------------------
    # Best-two insertion cache (LRU)
    # ----------------------------
    best_two_cache_cap = 20000
    best_two_cache = OrderedDict()

    def bt_cache_get(key):
        if key in best_two_cache:
            val = best_two_cache.pop(key)
            best_two_cache[key] = val
            return val
        return None

    def bt_cache_put(key, val):
        best_two_cache[key] = val
        if len(best_two_cache) > best_two_cache_cap:
            best_two_cache.popitem(last=False)

    # ----------------------------
    # Position sampling helpers (deterministic)
    # ----------------------------
    def deterministic_positions(L, use_all=False, focus_idx=None, k_positions=None, seed_tuple=None):
        if use_all or L <= 1:
            pos_list = list(range(L + 1))
            pos_sig = ('all', L)
            return pos_list, pos_sig
        if L <= 12:
            pos_list = list(range(L + 1))
            pos_sig = ('all', L)
            return pos_list, pos_sig
        k = k_positions if k_positions is not None else k_pos_sample
        pos_set = {0, L, L // 2, (L * 1) // 4, (L * 3) // 4}
        if focus_idx is not None:
            for d in (-3, -2, -1, 0, 1, 2, 3):
                p = focus_idx + d
                if 0 <= p <= L:
                    pos_set.add(p)
        st = seed_tuple if seed_tuple is not None else (L, focus_idx if focus_idx is not None else -1, k)
        rng = random.Random(hash(st) & 0xffffffff)
        for _ in range(min(k, L + 1)):
            pos_set.add(rng.randint(0, L))
        pos_list = sorted(pos_set)
        pos_sig = ('s', L, tuple(pos_list))
        return pos_list, pos_sig

    def evaluate_best_two_positions(base_seq, t, pos_list):
        """Return (best_cost, best_pos, second_best_cost)"""
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
        return best[0], best[1], second

    def best_two_insertion(base_seq, t, use_all_pos=False, focus_idx=None, k_positions=None):
        L = len(base_seq)
        seed_tuple = (tuple(base_seq[-min(10, L):]), t, L)
        pos_list, pos_sig = deterministic_positions(L, use_all=use_all_pos, focus_idx=focus_idx, k_positions=k_positions, seed_tuple=seed_tuple)
        key = (tuple(base_seq), t, pos_sig)
        cached = bt_cache_get(key)
        if cached is not None:
            return cached
        res = evaluate_best_two_positions(base_seq, t, pos_list)
        bt_cache_put(key, res)
        return res

    # ----------------------------
    # Construction: adaptive regret-diversified beam with best-two memo
    # ----------------------------
    def regret_insertion_build(seed_t):
        # Beam holds tuples: (seq, rem_set, cost)
        seq0 = [seed_t]
        rem0 = set(all_txns)
        rem0.remove(seed_t)
        beam = [(seq0, rem0, seq_cost(seq0))]

        while True:
            if all(len(rem) == 0 for _, rem, _ in beam):
                break

            expansions = []
            for seq, rem, _ in beam:
                if not rem:
                    expansions.append((seq, rem, seq_cost(seq), 0.0, float('inf')))
                    continue

                # Candidate transactions: sample then expand by top-k best and top-k regret union
                if len(rem) <= rem_all_threshold:
                    sample_txns = list(rem)
                else:
                    sample_txns = random.sample(list(rem), min(k_txn_sample, len(rem)))

                # Evaluate quick best-two to compute best cost and regret for filtering
                prelim = []
                use_all = (len(rem) <= endgame_all_pos_threshold) or (len(seq) <= 18)
                for t in sample_txns:
                    best_c, best_p, second_c = best_two_insertion(seq, t, use_all_pos=use_all)
                    regret = (second_c - best_c) if second_c < float('inf') else 0.0
                    prelim.append((best_c, regret, t, best_p, second_c))
                if not prelim:
                    continue

                prelim.sort(key=lambda x: x[0])  # by best cost
                k_keep = min(8, len(prelim))
                top_by_cost = prelim[:k_keep]

                prelim.sort(key=lambda x: (-x[1], x[0]))  # by regret desc then cost
                top_by_regret = prelim[:k_keep]

                chosen = {}
                for bc, rg, t, bp, sc in top_by_cost + top_by_regret:
                    # Build new state
                    new_seq = seq[:bp] + [t] + seq[bp:]
                    new_rem = rem.copy()
                    if t in new_rem:
                        new_rem.remove(t)
                    chosen_key = tuple(new_seq)
                    # Store best among duplicates by cost and regret
                    if chosen_key not in chosen or bc < chosen[chosen_key][2] or (bc == chosen[chosen_key][2] and rg > chosen[chosen_key][3]):
                        chosen[chosen_key] = (new_seq, new_rem, bc, rg, sc)

                # Final expansions from chosen
                for v in chosen.values():
                    expansions.append(v)  # (new_seq, new_rem, best_c, regret, second_c)

            if not expansions:
                break

            # Adaptive blend between immediate best and second-best by dispersion
            second_vals = [e[4] for e in expansions if e[4] < float('inf')]
            if second_vals:
                s_min, s_max = min(second_vals), max(second_vals)
                spread = s_max - s_min
                median_sc = sorted(second_vals)[len(second_vals) // 2]
                high_dispersion = spread > median_sc
                alpha, beta = (0.5, 0.5) if high_dispersion else (0.8, 0.2)
                regret_quota_ratio = 0.5 if high_dispersion else 0.3
            else:
                alpha, beta = 0.8, 0.2
                regret_quota_ratio = 0.3

            # Prepare scores
            scored = []
            for s, r, bc, rg, sc in expansions:
                score = alpha * bc + beta * (sc if sc < float('inf') else bc)
                scored.append((score, s, r, bc, rg))

            # Beam width and diversity adapt in endgame
            sample_rem_len = min((len(r) for _, s, r, _, _ in scored), default=0)
            local_beam_width = base_beam_width + 2 if sample_rem_len <= 2 * base_beam_width else base_beam_width
            div_k = 3 if sample_rem_len <= 2 * base_beam_width else diversity_suffix_k

            # Rank by score, then best cost, then regret
            scored.sort(key=lambda x: (x[0], x[3], -x[4]))

            # Next beam selection with diversity and high-regret quota
            next_beam = []
            seen_sig = set()

            def sig(seq):
                if not seq:
                    return ()
                k = min(len(seq), div_k)
                return tuple(seq[-k:])

            base_quota = max(1, int(local_beam_width * (1.0 - regret_quota_ratio)))
            reg_quota = max(0, local_beam_width - base_quota)

            # Fill by best score first
            i = 0
            while len(next_beam) < base_quota and i < len(scored):
                _, s, r, bc, rg = scored[i]
                i += 1
                if sig(s) in seen_sig:
                    continue
                seen_sig.add(sig(s))
                next_beam.append((s, r, bc))

            # Fill by high regret
            j = 0
            scored_by_regret = sorted(scored, key=lambda x: (-x[4], x[3], x[0]))
            while len(next_beam) < base_quota + reg_quota and j < len(scored_by_regret):
                _, s, r, bc, rg = scored_by_regret[j]
                j += 1
                if sig(s) in seen_sig:
                    continue
                seen_sig.add(sig(s))
                next_beam.append((s, r, bc))

            # Fill remaining by score
            k = 0
            while len(next_beam) < local_beam_width and k < len(scored):
                _, s, r, bc, rg = scored[k]
                k += 1
                if sig(s) in seen_sig:
                    continue
                seen_sig.add(sig(s))
                next_beam.append((s, r, bc))

            if not next_beam:
                next_beam = [(s, r, bc) for _, s, r, bc, _ in scored[:local_beam_width]]

            beam = next_beam

        # Select best complete sequence
        best_seq = None
        best_cost = float('inf')
        for seq, rem, cost in beam:
            if rem:
                seq_complete = seq + sorted(list(rem))
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
    # Local refinement: VND with don't-look bits and sampled 2-opt/swaps
    # ----------------------------
    def local_refine(seq):
        best_seq = seq[:]
        best_cost = seq_cost(best_seq)

        def try_adjacent_swap(cur_seq, cur_cost):
            for i in range(n - 1):
                cand = cur_seq[:]
                cand[i], cand[i + 1] = cand[i + 1], cand[i]
                c = seq_cost(cand)
                if c < cur_cost:
                    return cand, c, True, (i, i + 1)
            return cur_seq, cur_cost, False, None

        def reinsertion_pass(cur_seq, cur_cost):
            # Or-opt k=1,2,3 first-improvement using best-two cache
            for k_block in (1, 2, 3):
                L = len(cur_seq)
                for i in range(L - k_block + 1):
                    block = cur_seq[i:i + k_block]
                    base = cur_seq[:i] + cur_seq[i + k_block:]
                    use_all = len(base) <= 20
                    # Evaluate reinsertion position using best-two for the block (insert as a unit)
                    # Fall back to sampled positions
                    pos_list, _ = deterministic_positions(len(base), use_all=use_all, focus_idx=i, k_positions=k_pos_sample, seed_tuple=(tuple(base[-min(8, len(base)):]), tuple(block), len(base)))
                    best_c = float('inf')
                    best_p = None
                    for p in pos_list:
                        cand = base[:p] + block + base[p:]
                        c = seq_cost(cand)
                        if c < best_c:
                            best_c = c
                            best_p = p
                    if best_c < cur_cost:
                        cand = base[:best_p] + block + base[best_p:]
                        return cand, best_c, True, (i, best_p)
            return cur_seq, cur_cost, False, None

        def non_adj_swaps(cur_seq, cur_cost, dont_look):
            improved = False
            s = cur_seq[:]
            c = cur_cost
            trials = 0
            while trials < swap_samples:
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)
                trials += 1
                if i == j or abs(i - j) == 1:
                    continue
                if dont_look[i] and dont_look[j]:
                    continue
                if i > j:
                    i, j = j, i
                cand = s[:]
                cand[i], cand[j] = cand[j], cand[i]
                cc = seq_cost(cand)
                if cc < c:
                    s, c = cand, cc
                    improved = True
                    # reset around i, j
                    for d in range(max(0, i - 2), min(n, i + 3)):
                        dont_look[d] = False
                    for d in range(max(0, j - 2), min(n, j + 3)):
                        dont_look[d] = False
            return s, c, improved

        def two_opt_segments(cur_seq, cur_cost, dont_look):
            improved = False
            s = cur_seq[:]
            c = cur_cost
            for _ in range(two_opt_pairs):
                i = random.randint(0, n - 3)
                j = random.randint(i + 2, n - 1)
                if dont_look[i] and dont_look[j]:
                    continue
                cand = s[:i] + s[i:j + 1][::-1] + s[j + 1:]
                cc = seq_cost(cand)
                if cc < c:
                    s, c = cand, cc
                    improved = True
                    for d in range(max(0, i - 2), min(n, i + 3)):
                        dont_look[d] = False
                    for d in range(max(0, j - 2), min(n, j + 3)):
                        dont_look[d] = False
            return s, c, improved

        # VND loop with don't-look bits
        dont_look = [False] * n
        adj_rounds = 0
        improved_outer = True
        while improved_outer:
            improved_outer = False

            # Adjacent swap pass
            if adj_rounds < ls_adj_rounds_max:
                s2, c2, did, pos = try_adjacent_swap(best_seq, best_cost)
                if did:
                    best_seq, best_cost = s2, c2
                    improved_outer = True
                    if pos is not None:
                        i, j = pos
                        for d in range(max(0, i - 2), min(n, j + 3)):
                            dont_look[d] = False
                    continue
                adj_rounds += 1

            # Or-opt reinsertion blocks
            s2, c2, did, pos = reinsertion_pass(best_seq, best_cost)
            if did:
                best_seq, best_cost = s2, c2
                improved_outer = True
                if pos is not None:
                    i, j = pos
                    for d in range(max(0, min(i, j) - 2), min(n, max(i, j) + 3)):
                        dont_look[d] = False
                continue

            # Sampled non-adjacent swaps
            best_seq, best_cost, did = non_adj_swaps(best_seq, best_cost, dont_look)
            if did:
                improved_outer = True
                continue

            # True 2-opt segment reversals
            best_seq, best_cost, did = two_opt_segments(best_seq, best_cost, dont_look)
            if did:
                improved_outer = True
                continue

        return best_cost, best_seq

    # ----------------------------
    # LNS destroy-and-repair (sensitivity-guided) with quick polish
    # ----------------------------
    def lns_attempt(seq, prev_failures=0):
        cur = seq[:]
        frac = random.uniform(*destroy_frac_range)
        base_m = max(4, min(n // 2, int(frac * n)))
        # escalation on failures
        m = int(base_m * (1.25 if prev_failures >= 2 else 1.0))
        m = max(4, min(n // 2, m))

        # Sensitivity scoring on K indices
        K = min(sensitivity_K, n)
        if K > 0:
            cand_indices = sorted(random.sample(range(n), K))
            sens = []
            original_cost = seq_cost(cur)
            for idx in cand_indices:
                item = cur[idx]
                base = cur[:idx] + cur[idx + 1:]
                # Deterministic P positions
                pos_list, _ = deterministic_positions(len(base), use_all=(len(base) <= 12), focus_idx=idx, k_positions=sensitivity_pos_P, seed_tuple=(tuple(base[-min(8, len(base)):]), item, len(base)))
                vals = []
                for p in pos_list:
                    cand = base[:p] + [item] + base[p:]
                    vals.append(seq_cost(cand))
                if vals:
                    mean_v = sum(vals) / len(vals)
                    var = sum((v - mean_v) ** 2 for v in vals) / len(vals)
                    sens_score = var ** 0.5  # std dev
                else:
                    sens_score = 0.0
                sens.append((sens_score, idx))
            sens.sort(reverse=True)
            top_count = max(1, int(sensitivity_pick_ratio * m))
            remove_idxs = [idx for _, idx in sens[:top_count]]
        else:
            remove_idxs = []

        # Fill remainder with 1–2 contiguous blocks and randoms if needed
        remaining_to_pick = m - len(remove_idxs)
        blocks = 2 if prev_failures >= 2 else 1
        for _ in range(blocks):
            if remaining_to_pick <= 0:
                break
            span = max(2, min(remaining_to_pick, max(2, m // (blocks + 1))))
            start = random.randint(0, n - span)
            block = list(range(start, start + span))
            for b in block:
                if b not in remove_idxs and len(remove_idxs) < m:
                    remove_idxs.append(b)
            remaining_to_pick = m - len(remove_idxs)
        # If still short, fill random distinct indices
        all_idxs = set(range(n))
        rem_need = m - len(remove_idxs)
        if rem_need > 0:
            pool = list(all_idxs - set(remove_idxs))
            extra = random.sample(pool, min(rem_need, len(pool)))
            remove_idxs.extend(extra)

        remove_idxs = sorted(set(remove_idxs))[:m]
        remove_set = set(remove_idxs)
        removed = [cur[i] for i in remove_idxs]
        remaining = [cur[i] for i in range(n) if i not in remove_set]

        # Repair using regret-best-two insertion with cache
        seq_rep = remaining[:]
        rem_set = removed[:]
        while rem_set:
            cand_txns = rem_set if len(rem_set) <= k_txn_sample else random.sample(rem_set, k_txn_sample)
            best_overall = (float('inf'), None, None)  # cost, txn, new_seq
            best_regret = (float('-inf'), None, None)
            use_all = (len(seq_rep) <= 20) or (len(rem_set) <= endgame_all_pos_threshold)
            for t in cand_txns:
                best_c, best_p, second_c = best_two_insertion(seq_rep, t, use_all_pos=use_all)
                new_seq = seq_rep[:best_p] + [t] + seq_rep[best_p:]
                if best_c < best_overall[0]:
                    best_overall = (best_c, t, new_seq)
                regret = (second_c - best_c) if second_c < float('inf') else 0.0
                if regret > best_regret[0]:
                    best_regret = (regret, t, new_seq)
            chosen = best_regret if best_regret[1] is not None and best_regret[0] > 0 else best_overall
            t = chosen[1] if chosen[1] is not None else random.choice(rem_set)
            seq_rep = chosen[2] if chosen[2] is not None else (seq_rep + [t])
            rem_set.remove(t)

        # Quick polish: a light reinsertion then Or-opt(1)
        def quick_polish(s):
            base_cost = seq_cost(s)
            # Single reinsertion pass
            L = len(s)
            improved = True
            best_s, best_c = s[:], base_cost
            while improved:
                improved = False
                for i in range(L):
                    item = best_s[i]
                    base = best_s[:i] + best_s[i + 1:]
                    use_all = (len(base) <= 16)
                    bc, bp, _ = best_two_insertion(base, item, use_all_pos=use_all, focus_idx=i)
                    cand = base[:bp] + [item] + base[bp:]
                    if bc < best_c:
                        best_s, best_c = cand, bc
                        improved = True
                        break
            return best_c, best_s

        c_rep, s_rep = quick_polish(seq_rep)
        c_rep2, s_rep2 = local_refine(s_rep)
        if c_rep2 < c_rep:
            return c_rep2, s_rep2
        return c_rep, s_rep

    # ----------------------------
    # Path Relinking: bidirectional, block-aware with quick polish
    # ----------------------------
    def path_relink(A, B, max_moves=12):
        pos_in_B = {t: i for i, t in enumerate(B)}
        s = A[:]
        best_c = seq_cost(s)
        best_s = s[:]
        moves = 0

        def longest_increasing_run(seq):
            # Return (start, end) of the longest run whose positions in target are increasing by 1
            best = (0, 0)
            i = 0
            while i < n:
                j = i
                while j + 1 < n and pos_in_B[seq[j + 1]] == pos_in_B[seq[j]] + 1:
                    j += 1
                if (j - i) > (best[1] - best[0]):
                    best = (i, j)
                i = j + 1
            return best

        while moves < max_moves:
            # Identify the most displaced index
            disp = [(abs(i - pos_in_B[s[i]]), i) for i in range(n)]
            disp.sort(reverse=True)
            improved = False
            for _, idx in disp:
                if moves >= max_moves:
                    break
                item = s[idx]
                desired = pos_in_B[item]
                if desired == idx:
                    continue
                base = s[:idx] + s[idx + 1:]
                desired = max(0, min(desired, len(base)))
                cand1 = base[:desired] + [item] + base[desired:]
                c1 = seq_cost(cand1)

                # Try block move using longest increasing run containing idx
                l, r = longest_increasing_run(s)
                c2 = float('inf')
                cand2 = None
                if l <= idx <= r and r - l >= 1:
                    block = s[l:r + 1]
                    base2 = s[:l] + s[r + 1:]
                    ins = max(0, min(pos_in_B[block[0]], len(base2)))
                    cand2 = base2[:ins] + block + base2[ins:]
                    c2 = seq_cost(cand2)

                if c1 <= c2 and c1 < best_c:
                    s = cand1
                    best_c = c1
                    best_s = s[:]
                    moves += 1
                    # quick local tweak
                    qc, qs = local_refine(best_s)
                    if qc < best_c:
                        best_c, best_s = qc, qs
                        s = best_s[:]
                    improved = True
                    break
                elif cand2 is not None and c2 < best_c:
                    s = cand2
                    best_c = c2
                    best_s = s[:]
                    moves += 1
                    qc, qs = local_refine(best_s)
                    if qc < best_c:
                        best_c, best_s = qc, qs
                        s = best_s[:]
                    improved = True
                    break
            if not improved:
                break
        return best_c, best_s

    # ----------------------------
    # Seeding and elite management
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

    elite = []  # list of (cost, seq)
    def add_elite(c, s):
        nonlocal elite
        elite.append((c, s))
        elite.sort(key=lambda x: x[0])
        # enforce suffix-3 diversity
        uniq = []
        seen = set()
        for c1, s1 in elite:
            sig = tuple(s1[-elite_suffix_k:]) if len(s1) >= elite_suffix_k else tuple(s1)
            if sig in seen:
                continue
            seen.add(sig)
            uniq.append((c1, s1))
            if len(uniq) >= elite_size:
                break
        elite = uniq

    # Build from seeds
    for seed in seed_txns:
        seq0 = regret_insertion_build(seed)
        c1, s1 = local_refine(seq0)
        add_elite(c1, s1)

    # Fallback if no elite
    if not elite:
        base = all_txns[:]
        random.shuffle(base)
        elite = [(seq_cost(base), base)]

    best_overall_cost, best_overall_seq = elite[0]

    # ----------------------------
    # Iterated Local Search (ILS)
    # ----------------------------
    cur_cost, cur_seq = best_overall_cost, best_overall_seq[:]
    for _ in range(ils_rounds):
        # Perturb: segment reversal + random swaps
        pert = cur_seq[:]
        if n > 3:
            i = random.randint(0, n - 2)
            j = random.randint(i + 1, n - 1)
            pert = pert[:i] + pert[i:j + 1][::-1] + pert[j + 1:]
        for _ in range(perturb_swap_count):
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i != j:
                pert[i], pert[j] = pert[j], pert[i]
        c2, s2 = local_refine(pert)
        if c2 < cur_cost:
            cur_cost, cur_seq = c2, s2
            add_elite(c2, s2)
            if c2 < best_overall_cost:
                best_overall_cost, best_overall_seq = c2, s2

    # ----------------------------
    # LNS attempts with sensitivity and escalation
    # ----------------------------
    failures = 0
    for _ in range(lns_iters):
        c3, s3 = lns_attempt(best_overall_seq, prev_failures=failures)
        if c3 < best_overall_cost:
            best_overall_cost, best_overall_seq = c3, s3
            add_elite(c3, s3)
            failures = 0
        else:
            failures += 1

    # ----------------------------
    # Path Relinking among elites (bidirectional, block-aware)
    # ----------------------------
    if len(elite) >= 2:
        partners = elite[1:min(len(elite), elite_size)]
        base_cost, base_seq = best_overall_cost, best_overall_seq[:]
        for c_t, s_t in partners:
            pr1_c, pr1_s = path_relink(base_seq, s_t, max_moves=pr_max_moves)
            pr2_c, pr2_s = path_relink(s_t, base_seq, max_moves=pr_max_moves)
            if pr1_c < best_overall_cost:
                best_overall_cost, best_overall_seq = pr1_c, pr1_s
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