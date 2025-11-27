# EVOLVE-BLOCK-START
"""Transaction scheduling algorithm for optimizing makespan across multiple workloads"""

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
    Find a low-makespan schedule using regret-based best-insertion construction
    with memoized cost evaluation, exact endgame BnB, strong VND, ILS/LNS, and elite path relinking.

    Args:
        workload: Workload object containing transaction data
        num_seqs: Controls search breadth (number of seeds, sampling sizes)

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    n = workload.num_txns
    all_txns = list(range(n))

    # ----------------------------
    # Global memoization
    # ----------------------------
    cost_cache = {}
    def seq_cost(seq):
        key = tuple(seq)
        if key in cost_cache:
            return cost_cache[key]
        c = workload.get_opt_seq_cost(seq)
        cost_cache[key] = c
        return c

    # best-two memo across all phases: key=(tuple(base), ('T',t) or ('B',block_tuple), pos_sig)
    best_two_cache = {}

    # ----------------------------
    # Parameters (adaptive)
    # ----------------------------
    elite_size = max(3, min(8, 2 + num_seqs // 2))
    seed_elite_singletons = max(2, min(8, int(num_seqs)))
    seed_random_additional = max(1, min(4, int((num_seqs + 1) // 3)))

    # Construction sampling
    k_txn_sample = min(16, max(8, 2 + int(1.4 * num_seqs)))  # txns per insertion
    k_pos_sample = min(12, max(6, 2 + int(1.2 * num_seqs)))  # positions per insertion
    build_beam_width = max(3, min(8, 2 + num_seqs // 2))
    beam_div_suffix = 2
    rem_all_threshold = 14
    endgame_all_pos_threshold = max(6, min(12, num_seqs))
    # 1-step lookahead
    k_look_txn = max(4, min(8, 2 + num_seqs // 2))

    # Beam regret quota adaptive
    high_regret_ratio = 0.3

    # Local search parameters
    ls_adj_rounds_max = 2
    two_opt_trials = min(220, max(60, n))   # segment reversals
    reinsertion_pos_factor = 1.0

    # Iterated local search (ILS) / perturbation
    ils_rounds = max(2, min(5, 1 + num_seqs // 4))
    perturb_swap_count = max(2, min(6, 2 + num_seqs // 3))
    perturb_block_len = max(3, min(10, 3 + num_seqs // 2))

    # Large Neighborhood Search (LNS)
    lns_iters = max(2, min(6, 2 + num_seqs // 3))
    destroy_frac_range = (0.08, 0.18)
    regret_prob = 0.65

    # Endgame exact completion (bounded BnB)
    endgame_enum_K = max(6, min(9, endgame_all_pos_threshold + 3))
    endgame_node_cap = 3000

    # ----------------------------
    # Helpers for position sampling and best-two memo
    # ----------------------------
    def all_positions_sig(L):
        return ('ALL', L)

    def pos_signature(L, pos_list):
        # Stable signature for reusing best-two results across neighborhoods
        s = tuple(sorted(set(pos_list)))
        if len(s) == L + 1 and all(p == i for i, p in enumerate(s)):
            return all_positions_sig(L)
        if len(s) <= 14:
            return (L, s)
        # compress large sets by head+tail
        return (L, s[:7] + s[-7:])

    def deterministically_seeded_rng(seq_suffix, t_or_block, L):
        seed = (tuple(seq_suffix), t_or_block, L)
        return random.Random(hash(seed) & 0xffffffff)

    def position_samples(seq_len, base_seq=None, t_or_block=None, focus_idx=None, k_positions=None, use_all=False):
        """Sample insertion positions; include anchors and positions near focus_idx if provided.
        Deterministically seeded by sequence suffix and item/block to increase memo reuse."""
        if seq_len <= 1:
            return [0, seq_len]
        if use_all or seq_len <= 12:
            return list(range(seq_len + 1))
        k = k_positions if k_positions is not None else k_pos_sample
        pos_set = {0, seq_len, seq_len // 2, (seq_len * 1) // 4, (seq_len * 3) // 4}
        if focus_idx is not None:
            for d in (-3, -2, -1, 0, 1, 2, 3):
                p = focus_idx + d
                if 0 <= p <= seq_len:
                    pos_set.add(p)
        # Deterministic interior samples
        suffix = base_seq[-min(10, len(base_seq)):] if base_seq else []
        rng = deterministically_seeded_rng(suffix, t_or_block, seq_len)
        for _ in range(min(k, seq_len + 1)):
            pos_set.add(rng.randint(0, seq_len))
        return sorted(pos_set)

    def evaluate_best_two_positions(base_seq, payload, pos_list, is_block=False):
        """Return (best_cost, best_pos, second_best_cost) for inserting payload into base_seq over pos_list."""
        L = len(base_seq)
        sig = pos_signature(L, pos_list)
        key = (tuple(base_seq), ('B', tuple(payload)) if is_block else ('T', payload), sig)
        if key in best_two_cache:
            return best_two_cache[key]
        best = (float('inf'), None)
        second = float('inf')
        if is_block:
            block = list(payload)
            for p in pos_list:
                cand = base_seq[:p] + block + base_seq[p:]
                c = seq_cost(cand)
                if c < best[0]:
                    second = best[0]
                    best = (c, p)
                elif c < second:
                    second = c
        else:
            t = payload
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

    def best_two_insertion(base_seq, t, use_all_pos=False, focus_idx=None, k_positions=None):
        L = len(base_seq)
        pos_list = position_samples(L, base_seq, ('T', t), focus_idx, k_positions, use_all_pos)
        return evaluate_best_two_positions(base_seq, t, pos_list, is_block=False)

    def best_two_block(base_seq, block, use_all_pos=False, focus_idx=None, k_positions=None):
        L = len(base_seq)
        pos_list = position_samples(L, base_seq, ('B', tuple(block)), focus_idx, k_positions, use_all_pos)
        return evaluate_best_two_positions(base_seq, block, pos_list, is_block=True)

    # ----------------------------
    # Endgame exact completion (branch-and-bound)
    # ----------------------------
    def endgame_optimal_completion(prefix_seq, rem_set):
        """
        Given prefix_seq and small remainder set (|rem_set|<=K), return (best_cost,best_seq)
        using DFS branch-and-bound with a small transposition table and node cap.
        """
        if not rem_set:
            return seq_cost(prefix_seq), prefix_seq[:]
        best_c = float('inf')
        best_s = None
        TT = {}
        nodes = 0

        def suffix_key(seq, k=3):
            m = min(k, len(seq))
            return tuple(seq[-m:]) if m > 0 else ()

        def dfs(seq, rem):
            nonlocal best_c, best_s, nodes
            if nodes >= endgame_node_cap:
                return
            nodes += 1
            c_prefix = seq_cost(seq)
            if c_prefix >= best_c:
                return
            key = (frozenset(rem), suffix_key(seq, 3))
            prev = TT.get(key)
            if prev is not None and prev <= c_prefix:
                return
            TT[key] = c_prefix
            if not rem:
                best_c = c_prefix
                best_s = seq[:]
                return
            # Order children by high regret first then low best cost
            order = []
            for t in rem:
                b, p, s2 = best_two_insertion(seq, t, use_all_pos=True)
                regret = (s2 - b) if s2 < float('inf') else 0.0
                order.append((-regret, b, t, p))
            order.sort()
            for _, bcost, t, p in order:
                if bcost >= best_c:
                    continue
                new_seq = seq[:p] + [t] + seq[p:]
                new_rem = rem.copy()
                new_rem.remove(t)
                dfs(new_seq, new_rem)

        dfs(prefix_seq[:], set(rem_set))
        if best_s is None:
            seq_complete = prefix_seq[:] + sorted(list(rem_set))
            best_c = seq_cost(seq_complete)
            best_s = seq_complete
        return best_c, best_s

    # ----------------------------
    # Construction: regret beam with lookahead and exact endgame
    # ----------------------------
    def regret_insertion_build(seed_t=None):
        """Construct a schedule using regret-guided insertion, 1-step lookahead, and suffix diversity."""
        if seed_t is None:
            seed_t = random.randint(0, n - 1)

        seq0 = [seed_t]
        rem0 = set(all_txns)
        rem0.remove(seed_t)
        beam = [(seq0, rem0, seq_cost(seq0))]

        while True:
            if all(len(rem) == 0 for _, rem, _ in beam):
                break

            expansions = []
            for seq, rem, base_cost in beam:
                if not rem:
                    expansions.append((seq, rem, base_cost, 0.0, base_cost))
                    continue

                if len(rem) <= rem_all_threshold:
                    cand_txns = list(rem)
                else:
                    cand_txns = random.sample(list(rem), min(k_txn_sample, len(rem)))

                use_all = (len(seq) <= 18) or (len(rem) <= rem_all_threshold)
                for t in cand_txns:
                    best_c, best_p, second_c = best_two_insertion(seq, t, use_all_pos=use_all)
                    new_seq = seq[:best_p] + [t] + seq[best_p:]
                    new_rem = rem.copy()
                    new_rem.remove(t)
                    # 1-step lookahead: try inserting a few next candidates
                    if new_rem:
                        cand2 = list(new_rem) if len(new_rem) <= k_look_txn else random.sample(list(new_rem), k_look_txn)
                        best_c2 = float('inf')
                        for u in cand2:
                            c2, _, _ = best_two_insertion(new_seq, u, use_all_pos=(len(new_seq) <= 18))
                            if c2 < best_c2:
                                best_c2 = c2
                    else:
                        best_c2 = best_c
                    regret = (second_c - best_c) if second_c < float('inf') else 0.0
                    expansions.append((new_seq, new_rem, best_c, regret, best_c2))

            if not expansions:
                break

            # Adaptive blend based on lookahead dispersion
            lookahead_vals = [e[4] for e in expansions]
            if lookahead_vals:
                spread = max(lookahead_vals) - min(lookahead_vals)
                median_sc = sorted(lookahead_vals)[len(lookahead_vals) // 2]
                alpha = 0.5 if spread > median_sc else 0.8
            else:
                alpha = 0.8

            scored = []
            for seq, rem, best_c, regret, best_c2 in expansions:
                score = alpha * best_c + (1.0 - alpha) * best_c2
                scored.append((score, seq, rem, best_c, regret))

            # Beam width and regret quota adaptation near end
            min_rem = min((len(r) for _, r, _, _, _ in expansions), default=0)
            bw = build_beam_width + 2 if min_rem <= 2 * build_beam_width else build_beam_width
            regret_ratio = 0.5 if min_rem <= rem_all_threshold else high_regret_ratio
            regret_slots = max(1, int(bw * regret_ratio))
            cost_slots = max(1, bw - regret_slots)

            # Rank and select with suffix diversity
            scored.sort(key=lambda x: (x[0], x[3], -x[4]))  # blended score, best cost, high regret
            by_cost = scored[:]
            by_regret = sorted(scored, key=lambda x: (-x[4], x[3], x[0]))

            next_beam = []
            seen_sig = set()

            def suffix_sig(s):
                return tuple(s[-beam_div_suffix:]) if len(s) >= beam_div_suffix else tuple(s)

            i = 0
            while len(next_beam) < cost_slots and i < len(by_cost):
                _, s, r, c, _ = by_cost[i]
                sig = suffix_sig(s)
                if sig not in seen_sig:
                    seen_sig.add(sig)
                    next_beam.append((s, r, c))
                i += 1
            j = 0
            while len(next_beam) < cost_slots + regret_slots and j < len(by_regret):
                _, s, r, c, _ = by_regret[j]
                sig = suffix_sig(s)
                if sig not in seen_sig:
                    seen_sig.add(sig)
                    next_beam.append((s, r, c))
                j += 1
            k = 0
            while len(next_beam) < bw and k < len(by_cost):
                _, s, r, c, _ = by_cost[k]
                sig = suffix_sig(s)
                if sig not in seen_sig:
                    seen_sig.add(sig)
                    next_beam.append((s, r, c))
                k += 1

            if not next_beam:
                next_beam = [(s, r, c) for _, s, r, c, _ in scored[:bw]]

            beam = next_beam

        # Choose best completion; solve exactly when small remainder exists
        best_seq = None
        best_cost = float('inf')
        for seq, rem, cost in beam:
            if not rem:
                c, s = cost, seq
            else:
                if len(rem) <= endgame_enum_K:
                    c, s = endgame_optimal_completion(seq, rem)
                else:
                    seq_complete = seq + sorted(list(rem))
                    c, s = seq_cost(seq_complete), seq_complete
            if c < best_cost:
                best_cost = c
                best_seq = s
        return best_seq

    # ----------------------------
    # Local refinement: VND with best-two–guided moves and don't-look bits
    # ----------------------------
    def local_refine(seq):
        best_seq = seq[:]
        best_cost = seq_cost(best_seq)
        dont_look = [0] * n

        def try_reinsertion(cur_seq, cur_cost):
            L = len(cur_seq)
            kpos = max(6, int(reinsertion_pos_factor * k_pos_sample))
            for i in range(L):
                if dont_look[i]:
                    continue
                item = cur_seq[i]
                base = cur_seq[:i] + cur_seq[i + 1:]
                use_all = len(base) <= 20
                b, p, _ = best_two_block(base, [item], use_all_pos=use_all, focus_idx=i, k_positions=kpos)
                cand = base[:p] + [item] + base[p:]
                if cand == cur_seq:
                    dont_look[i] = 1
                    continue
                if b < cur_cost:
                    # reset bits around affected region
                    for j in range(max(0, i - 2), min(L, i + 3)):
                        dont_look[j] = 0
                    return True, b, cand
                else:
                    dont_look[i] = 1
            return False, cur_cost, cur_seq

        def try_or_opt(cur_seq, cur_cost, k_block):
            L = len(cur_seq)
            for i in range(L - k_block + 1):
                block = cur_seq[i:i + k_block]
                base = cur_seq[:i] + cur_seq[i + k_block:]
                use_all = len(base) <= 20
                b, p, _ = best_two_block(base, block, use_all_pos=use_all, focus_idx=i, k_positions=k_pos_sample)
                cand = base[:p] + block + base[p:]
                if cand == cur_seq:
                    continue
                if b < cur_cost:
                    return True, b, cand
            return False, cur_cost, cur_seq

        def adjacent_swap_pass(cur_seq, cur_cost):
            L = len(cur_seq)
            for i in range(L - 1):
                cand = cur_seq[:]
                cand[i], cand[i + 1] = cand[i + 1], cand[i]
                c = seq_cost(cand)
                if c < cur_cost:
                    # reset bits near i
                    for j in range(max(0, i - 2), min(L, i + 3)):
                        dont_look[j] = 0
                    return True, c, cand
            return False, cur_cost, cur_seq

        def two_opt_segment_reverse(cur_seq, cur_cost):
            L = len(cur_seq)
            trials = two_opt_trials
            for _ in range(trials):
                i = random.randint(0, L - 2)
                j = random.randint(i + 2, L - 1) if i + 2 < L else None
                if j is None:
                    continue
                cand = cur_seq[:i] + cur_seq[i:j + 1][::-1] + cur_seq[j + 1:]
                if cand == cur_seq:
                    continue
                c = seq_cost(cand)
                if c < cur_cost:
                    # reset bits near i..j
                    for idx in range(max(0, i - 2), min(L, j + 3)):
                        if idx < len(dont_look):
                            dont_look[idx] = 0
                    return True, c, cand
            return False, cur_cost, cur_seq

        improved_outer = True
        adj_rounds = 0
        while improved_outer:
            improved_outer = False

            # Or-opt 1 (reinsertion)
            changed, best_cost, best_seq = try_reinsertion(best_seq, best_cost)
            if changed:
                improved_outer = True
                continue

            # Or-opt 2
            changed, best_cost, best_seq = try_or_opt(best_seq, best_cost, 2)
            if changed:
                improved_outer = True
                continue

            # Or-opt 3
            changed, best_cost, best_seq = try_or_opt(best_seq, best_cost, 3)
            if changed:
                improved_outer = True
                continue

            # Adjacent swap pass (limited)
            if adj_rounds < ls_adj_rounds_max:
                changed, best_cost, best_seq = adjacent_swap_pass(best_seq, best_cost)
                adj_rounds += 1
                if changed:
                    improved_outer = True
                    continue

            # 2-opt segment reversals
            changed, best_cost, best_seq = two_opt_segment_reverse(best_seq, best_cost)
            if changed:
                improved_outer = True

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
            # Block relocation
            if n > perturb_block_len + 2:
                start = random.randint(0, n - perturb_block_len - 1)
                block = s[start:start + perturb_block_len]
                del s[start:start + perturb_block_len]
                insert_pos = random.randint(0, len(s))
                s = s[:insert_pos] + block + s[insert_pos:]
        return s

    # ----------------------------
    # LNS destroy-and-repair
    # ----------------------------
    def lns_attempt(seq):
        cur = seq[:]
        # Destroy size
        frac = random.uniform(*destroy_frac_range)
        m = max(4, min(n // 2, int(frac * n)))
        # Mix random subset and contiguous block
        if random.random() < 0.5:
            remove_idxs = sorted(random.sample(range(n), m))
        else:
            start = random.randint(0, n - m)
            remove_idxs = list(range(start, start + m))
        remove_set = set(remove_idxs)
        removed = [cur[i] for i in remove_idxs]
        remaining = [cur[i] for i in range(n) if i not in remove_set]

        # Repair via best-two regret insertion
        seq_rep = remaining[:]
        rem = removed[:]
        # If small, solve exact completion with current prefix
        if len(rem) <= endgame_enum_K:
            c_end, s_end = endgame_optimal_completion(seq_rep, set(rem))
            c_ref, s_ref = local_refine(s_end)
            return c_ref, s_ref

        while rem:
            cand_txns = rem if len(rem) <= k_txn_sample else random.sample(rem, k_txn_sample)
            best_overall = (float('inf'), None, None)  # cost, txn, pos
            best_by_regret = (float('-inf'), None, None)
            use_all = (len(seq_rep) <= 18) or (len(rem) <= endgame_all_pos_threshold)
            for t in cand_txns:
                b, p, s2 = best_two_insertion(seq_rep, t, use_all_pos=use_all)
                regret = (s2 - b) if s2 < float('inf') else 0.0
                if b < best_overall[0]:
                    best_overall = (b, t, p)
                if regret > best_by_regret[0]:
                    best_by_regret = (regret, t, p)
            pick_regret = (random.random() < regret_prob)
            chosen = best_by_regret if pick_regret and best_by_regret[1] is not None else best_overall
            t = chosen[1] if chosen[1] is not None else random.choice(rem)
            p = chosen[2] if chosen[2] is not None else len(seq_rep)
            seq_rep = seq_rep[:p] + [t] + seq_rep[p:]
            rem.remove(t)

        c_rep, s_rep = local_refine(seq_rep)
        return c_rep, s_rep

    # ----------------------------
    # Path Relinking: bidirectional with best-two guided moves
    # ----------------------------
    def path_relink(source_seq, target_seq, max_moves=12):
        pos_in_target = {t: i for i, t in enumerate(target_seq)}
        s = source_seq[:]
        best_c = seq_cost(s)
        best_s = s[:]
        moves = 0
        displacement = [(abs(i - pos_in_target.get(s[i], i)), i) for i in range(n)]
        displacement.sort(reverse=True)
        for _, idx in displacement:
            if moves >= max_moves:
                break
            item = s[idx]
            desired = pos_in_target.get(item, idx)
            if desired == idx:
                continue
            base = s[:idx] + s[idx + 1:]
            use_all = len(base) <= 20
            b, p, _ = best_two_block(base, [item], use_all_pos=use_all, focus_idx=desired, k_positions=k_pos_sample)
            cand = base[:p] + [item] + base[p:]
            c = seq_cost(cand)
            if c < best_c:
                best_c = c
                best_s = cand[:]
                s = cand
                moves += 1
        return best_c, best_s

    # ----------------------------
    # Seeding and elite pool
    # ----------------------------
    singleton_scores = [(seq_cost([t]), t) for t in all_txns]
    singleton_scores.sort(key=lambda x: x[0])

    seed_txns = [t for _, t in singleton_scores[:seed_elite_singletons]]
    remaining = [t for t in all_txns if t not in seed_txns]
    if remaining and seed_random_additional > 0:
        seed_txns += random.sample(remaining, min(seed_random_additional, len(remaining)))

    elite = []  # list of (cost, seq)

    def add_elite(c, s):
        nonlocal elite
        elite.append((c, s))
        elite.sort(key=lambda x: x[0])
        # Keep unique by suffix-2 to enforce diversity
        uniq = []
        seen_sig = set()
        for c1, s1 in elite:
            sig = tuple(s1[-2:]) if len(s1) >= 2 else tuple(s1)
            if sig in seen_sig:
                continue
            seen_sig.add(sig)
            uniq.append((c1, s1))
            if len(uniq) >= elite_size:
                break
        elite = uniq

    # Build from seeds and refine
    for seed in seed_txns:
        seq0 = regret_insertion_build(seed)
        c1, s1 = local_refine(seq0)
        add_elite(c1, s1)

    # Fallback if needed
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
        pert = perturb(cur_seq)
        c2, s2 = local_refine(pert)
        if c2 < cur_cost:
            cur_cost, cur_seq = c2, s2
            add_elite(c2, s2)
            if c2 < best_overall_cost:
                best_overall_cost, best_overall_seq = c2, s2

    # ----------------------------
    # LNS destroy-and-repair
    # ----------------------------
    for _ in range(lns_iters):
        c3, s3 = lns_attempt(best_overall_seq)
        if c3 < best_overall_cost:
            best_overall_cost, best_overall_seq = c3, s3
            add_elite(c3, s3)

    # ----------------------------
    # Path Relinking among elites
    # ----------------------------
    if len(elite) >= 2:
        base_seq = best_overall_seq
        partners = elite[1:min(len(elite), elite_size)]
        for c_t, s_t in partners:
            pr1_c, pr1_s = path_relink(base_seq, s_t, max_moves=max(8, min(16, n // 6)))
            pr2_c, pr2_s = path_relink(s_t, base_seq, max_moves=max(8, min(16, n // 6)))
            # choose better direction
            if pr1_c <= pr2_c:
                cand_c, cand_s = pr1_c, pr1_s
            else:
                cand_c, cand_s = pr2_c, pr2_s
            if cand_c < best_overall_cost:
                # quick polish
                q_c, q_s = local_refine(cand_s)
                if q_c < best_overall_cost:
                    best_overall_cost, best_overall_seq = q_c, q_s
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