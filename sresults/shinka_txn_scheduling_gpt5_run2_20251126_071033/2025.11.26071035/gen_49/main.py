# EVOLVE-BLOCK-START
"""Transaction scheduling algorithm for optimizing makespan across multiple workloads
Regret-diverse GRASP + adaptive beam lookahead + VND (DLB, 2-opt, pair-swaps) + sensitivity-guided LNS + relinking.
"""

import time
from functools import lru_cache
import random
import statistics
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
    Hybrid constructor + adaptive regret-diverse beam + VND + sensitivity-guided LNS + relinking.

    Args:
        workload: Workload object containing transaction data
        num_seqs: Search effort parameter (drives beam width, seeds, and refinement)

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    n = workload.num_txns
    txns_all = list(range(n))

    # ------- Parameterization (adaptive) -------
    small = n <= 40
    med = 40 < n <= 80
    large = n > 80

    # Construction
    rcl_alpha = 0.65  # pick low cost vs high regret in RCL
    cand_txn_sample = 10 if med or large else 12
    pos_sample_cap = 18 if large else (None if small else 24)

    # Beam search
    base_bw = max(5, min(10, num_seqs + (2 if small else 0)))
    beam_width = base_bw
    cand_per_state = min(28, max(10, n // (4 if large else 3)))
    lookahead_k = 6 if med or large else 8
    diversity_quota = max(1, int(0.25 * beam_width))  # 25% initially

    # Seeds and elites
    seeds = max(3, min(6, num_seqs))
    elite_cap = 5

    # Local search
    ls_adj_passes = 2 if large else 3
    vnd_rounds = 2 if large else 3
    pair_swap_samples = min(200, max(60, n))
    two_opt_samples = min(200, max(60, n))

    # LNS
    lns_rounds = 2 if med or large else 3
    lns_remove_frac = 0.08 if large else (0.1 if med else 0.12)
    lns_remove_min = 8 if med or large else 6
    lns_remove_max = 18 if med or large else 16
    lns_sensitivity_K = min(20, n)
    lns_sensitivity_P = 6
    lns_high_sens_share = 0.4  # 40% of removals guided by sensitivity

    # Iterated improvements
    ils_iterations = 2 if large else 3

    # ------- Cost evaluator with cache -------
    @lru_cache(maxsize=250_000)
    def _eval_cost_tuple(seq_tuple):
        return workload.get_opt_seq_cost(list(seq_tuple))

    def eval_cost(prefix):
        return _eval_cost_tuple(tuple(prefix))

    # ------- Utility: stratified sample positions for insertion -------
    def sample_positions(seq_len):
        total = seq_len + 1
        if pos_sample_cap is None or total <= pos_sample_cap:
            return list(range(total))
        # Fixed anchors: ends, mid, quartiles
        anchors = {0, seq_len, seq_len // 2, seq_len // 4, (3 * seq_len) // 4}
        anchors = {p for p in anchors if 0 <= p <= seq_len}
        need = max(2, pos_sample_cap - len(anchors))
        interior = [p for p in range(1, seq_len) if p not in anchors]
        if need > 0 and interior:
            if need >= len(interior):
                extra = interior
            else:
                extra = random.sample(interior, need)
            anchors.update(extra)
        return sorted(anchors)

    # ------- Insertion helpers with per-episode cache -------
    def make_best_two_cache():
        return {}

    def best_two_insertions(seq, txn, cache):
        key = (tuple(seq), txn)
        hit = cache.get(key)
        if hit is not None:
            return hit
        positions = sample_positions(len(seq))
        best = (float('inf'), None)
        second = (float('inf'), None)
        for pos in positions:
            cand = seq[:]
            cand.insert(pos, txn)
            c = eval_cost(cand)
            if c < best[0]:
                second = best
                best = (c, pos)
            elif c < second[0]:
                second = (c, pos)
        if second[0] == float('inf'):
            second = best
        cache[key] = (best, second)
        return best, second

    # ------- Regret-guided GRASP constructor -------
    def construct_regret_insertion(seed_txn=None, rcl_k=3):
        cache = make_best_two_cache()
        remaining = set(txns_all)
        seq = []
        if seed_txn is None:
            start_candidates = random.sample(txns_all, min(8, n))
            starter = min(start_candidates, key=lambda t: eval_cost([t]))
        else:
            starter = seed_txn
        seq.append(starter)
        remaining.remove(starter)

        # Optional second
        if remaining:
            k = min(6, len(remaining))
            cands = random.sample(list(remaining), k)
            best_pair = (float('inf'), None, None)
            for t in cands:
                for pos in [0, 1]:
                    cand = seq[:]
                    cand.insert(pos, t)
                    c = eval_cost(cand)
                    if c < best_pair[0]:
                        best_pair = (c, t, pos)
            if best_pair[1] is not None:
                seq.insert(best_pair[2], best_pair[1])
                remaining.remove(best_pair[1])

        while remaining:
            # Candidate transactions
            k_t = min(len(remaining), max(4, cand_txn_sample + random.randint(-2, 2)))
            cand_txns = random.sample(list(remaining), k_t)

            scored = []
            for t in cand_txns:
                (c1, p1), (c2, p2) = best_two_insertions(seq, t, cache)
                regret = (c2 - c1) if c2 < float('inf') else (0.0)
                scored.append((c1, regret, t, p1))

            # Build RCL from both best cost and high regret
            scored.sort(key=lambda x: x[0])
            rcl_cost = scored[:max(1, min(rcl_k, len(scored)))]
            scored.sort(key=lambda x: (-x[1], x[0]))
            rcl_regret = scored[:max(1, min(rcl_k, len(scored)))]

            pool = { (c, r, t, p) for (c, r, t, p) in (rcl_cost + rcl_regret) }
            pool = list(pool)
            # Weighted pick: favor low cost with probability alpha; else high regret
            if random.random() < rcl_alpha:
                pool.sort(key=lambda x: x[0])
            else:
                pool.sort(key=lambda x: (-x[1], x[0]))
            chosen_c, chosen_r, chosen_t, chosen_p = random.choice(pool[:max(1, min(3, len(pool)))])
            seq.insert(chosen_p, chosen_t)
            remaining.remove(chosen_t)

        return eval_cost(seq), seq

    # ------- Beam search with regret-aware lookahead and suffix diversity -------
    def beam_search():
        nonlocal beam_width, diversity_quota
        # Initialize with top singletons (and 1 GRASP seed)
        starters = [(eval_cost([t]), [t]) for t in txns_all]
        starters.sort(key=lambda x: x[0])
        init = starters[:min(len(starters), max(beam_width * 2, beam_width + 2))]

        # Add a GRASP seed for diversity
        c0, s0 = construct_regret_insertion()
        init.append((c0, s0))

        # Beam state: tuples (cost, seq, remaining_set)
        beam = []
        seen = set()
        for c, seq in init:
            key = tuple(seq)
            if key in seen:
                continue
            seen.add(key)
            rem = frozenset(txn for txn in txns_all if txn not in seq)
            beam.append((c, seq, rem))
            if len(beam) >= beam_width:
                break

        best_complete = (float('inf'), [])

        for depth in range(1, n + 1):
            if not beam:
                break
            # Endgame widening adjustments
            min_remaining = min((len(rem) for _, _, rem in beam), default=n)
            if min_remaining <= 2 * beam_width:
                beam_width = base_bw + 2
                diversity_quota = max(1, int(0.40 * beam_width))  # raise to 40%

            next_pool = []
            layer_seen = set()
            suffix_best = {}  # suffix-2 -> best c1 tracked for allowing strictly better duplicates
            # Collect regret dispersion to adjust lookahead blending
            layer_regrets = []

            per_state_scored = []
            for c, seq, rem in beam:
                if not rem:
                    if c < best_complete[0]:
                        best_complete = (c, seq)
                    continue

                rem_list = list(rem)
                # Sample candidates
                if len(rem_list) <= cand_per_state:
                    expand_list = rem_list
                else:
                    expand_list = random.sample(rem_list, cand_per_state)

                # Score expansions with lookahead and compute regret dispersion info
                local_scored = []
                for t in expand_list:
                    seq1 = seq + [t]
                    c1 = eval_cost(seq1)

                    # lookahead: best among k sampled second steps
                    rem_after = [x for x in rem if x != t]
                    best_c2 = float('inf')
                    regret_range = 0.0
                    if rem_after:
                        k2 = min(lookahead_k, len(rem_after))
                        second = random.sample(rem_after, k2)
                        second_costs = []
                        for u in second:
                            cu = eval_cost(seq1 + [u])
                            second_costs.append(cu)
                            if cu < best_c2:
                                best_c2 = cu
                        if len(second_costs) >= 2:
                            regret_range = max(second_costs) - min(second_costs)
                    local_scored.append((t, c1, best_c2, regret_range))
                    if regret_range > 0:
                        layer_regrets.append(regret_range)

                per_state_scored.append((seq, rem, local_scored))

            # Determine layer regret threshold (median)
            tau = statistics.median(layer_regrets) if layer_regrets else 0.0

            # Build next layer
            for seq, rem, local_scored in per_state_scored:
                scored = []
                for (t, c1, best_c2, rgr) in local_scored:
                    if rgr > tau:
                        score = 0.5 * c1 + 0.5 * best_c2
                    else:
                        score = 0.8 * c1 + 0.2 * best_c2
                    scored.append((score, c1, rgr, t))

                # Select by cost and regret diversity
                scored.sort(key=lambda x: x[0])  # by blended score
                top_cost = scored[:max(1, min(len(scored), beam_width))]
                # Add regret-heavy options
                scored_by_regret = sorted(scored, key=lambda x: (-x[2], x[0]))
                top_regret = scored_by_regret[:min(diversity_quota, len(scored_by_regret))]

                cand_acts = top_cost + top_regret
                # Dedup actions
                uniq = {}
                for sc, c1, rg, t in cand_acts:
                    if t not in uniq or c1 < uniq[t][1]:
                        uniq[t] = (sc, c1, rg)
                for t, (sc, c1, rg) in uniq.items():
                    new_seq = seq + [t]
                    new_rem = frozenset(x for x in rem if x != t)

                    # Suffix diversity: last-2 signature
                    if len(new_seq) >= 2:
                        sig = (new_seq[-2], new_seq[-1])
                    else:
                        sig = (None, new_seq[-1])

                    key = tuple(new_seq)
                    if key in layer_seen:
                        continue
                    # Allow one per suffix unless a strictly better c1 appears; admit if new or better
                    prev_best = suffix_best.get(sig)
                    if prev_best is not None and c1 >= prev_best - 1e-9:
                        continue

                    layer_seen.add(key)
                    suffix_best[sig] = c1 if prev_best is None or c1 < prev_best else prev_best
                    next_pool.append((c1, new_seq, new_rem))

            if not next_pool:
                break

            # Prune to beam width by cost; deduplicate by full prefix
            next_pool.sort(key=lambda x: x[0])
            pruned = []
            seen_prefixes = set()
            for c1, seq, rem in next_pool:
                key = tuple(seq)
                if key in seen_prefixes:
                    continue
                seen_prefixes.add(key)
                pruned.append((c1, seq, rem))
                if len(pruned) >= beam_width:
                    break
            beam = pruned

        # Complete if not already
        for c, seq, rem in beam:
            if not rem and c < best_complete[0]:
                best_complete = (c, seq)

        if best_complete[1] and len(best_complete[1]) == n:
            return best_complete

        # Greedy completion from best partial using regret-guided best insertion
        if beam:
            c, seq, rem = min(beam, key=lambda x: x[0])
            seq_partial = list(seq)
            remaining = list(rem)
            re_cache = make_best_two_cache()
            while remaining:
                k_t = min(len(remaining), max(6, cand_txn_sample))
                cand_txns = remaining if len(remaining) <= k_t else random.sample(remaining, k_t)
                scored = []
                for t in cand_txns:
                    (c1, p1), (c2, p2) = best_two_insertions(seq_partial, t, re_cache)
                    regret = (c2 - c1) if c2 < float('inf') else 0.0
                    scored.append((c1, regret, t, p1))
                if not scored:
                    t = remaining.pop()
                    seq_partial.append(t)
                    continue
                # Build small pool by best cost and highest regret
                scored.sort(key=lambda x: x[0])
                rcl_cost = scored[:min(3, len(scored))]
                scored.sort(key=lambda x: (-x[1], x[0]))
                rcl_regret = scored[:min(2, len(scored))]
                pool = list({(c1, rg, t, p) for (c1, rg, t, p) in (rcl_cost + rcl_regret)})
                # Choose the best from the pool by cost, tie-break by regret
                pool.sort(key=lambda x: (x[0], -x[1]))
                chosen_c, chosen_rg, chosen_t, chosen_p = pool[0]
                seq_partial.insert(chosen_p, chosen_t)
                remaining.remove(chosen_t)
            final_cost = eval_cost(seq_partial)
            return final_cost, seq_partial

        # Fallback: identity
        ident = list(range(n))
        return eval_cost(ident), ident

    # ------- Local search: VND with Or-opt (1..3), Adjacent swaps, 2-opt, pair swaps (DLB where applicable) -------
    def vnd_local_search(seq, start_cost, max_rounds=vnd_rounds):
        best_seq = list(seq)
        best_cost = start_cost
        n_local = len(best_seq)
        if n_local <= 2:
            return best_cost, best_seq

        # DLB flags for indices; reset when neighborhood changes locally
        dont_look = [False] * n_local

        def or_opt_pass(block_len):
            nonlocal best_seq, best_cost, dont_look
            improved = False
            n_cur = len(best_seq)
            i = 0
            while i <= n_cur - block_len:
                if i < len(dont_look) and dont_look[i]:
                    i += 1
                    continue
                block = best_seq[i:i + block_len]
                base = best_seq[:i] + best_seq[i + block_len:]
                m = len(base) + 1
                # stratified positions
                if pos_sample_cap is None or m <= (pos_sample_cap or m):
                    positions = list(range(m))
                else:
                    anchors = {0, m - 1, m // 2, m // 4, (3 * m) // 4}
                    anchors = {p for p in anchors if 0 <= p < m}
                    need = max(2, min(pos_sample_cap - len(anchors), max(0, m - len(anchors))))
                    interior = [p for p in range(1, m - 1) if p not in anchors]
                    if need > 0 and interior:
                        anchors.update(random.sample(interior, min(need, len(interior))))
                    positions = sorted(anchors)
                best_move_cost = best_cost
                best_pos = None
                for pos in positions:
                    if pos == i:
                        continue
                    cand = base[:]
                    cand[pos:pos] = block
                    c = eval_cost(cand)
                    if c < best_move_cost:
                        best_move_cost = c
                        best_pos = pos
                if best_pos is not None:
                    new_seq = base[:]
                    new_seq[best_pos:best_pos] = block
                    best_seq = new_seq
                    best_cost = best_move_cost
                    improved = True
                    dont_look = [False] * len(best_seq)
                else:
                    if i < len(dont_look):
                        dont_look[i] = True
                    i += 1
            return improved

        def adjacent_swap_pass():
            nonlocal best_seq, best_cost, dont_look
            improved = False
            i = 0
            while i < len(best_seq) - 1:
                cand = best_seq[:]
                cand[i], cand[i + 1] = cand[i + 1], cand[i]
                c = eval_cost(cand)
                if c < best_cost:
                    best_cost = c
                    best_seq = cand
                    improved = True
                    dont_look = [False] * len(best_seq)
                i += 1
            return improved

        def sampled_pair_swaps(tries=pair_swap_samples):
            nonlocal best_seq, best_cost, dont_look
            improved = False
            ncur = len(best_seq)
            if ncur <= 3:
                return False
            best_delta = 0.0
            best_ij = None
            for _ in range(tries):
                i = random.randint(0, ncur - 1)
                j = random.randint(0, ncur - 1)
                if i == j or abs(i - j) <= 1:
                    continue
                cand = best_seq[:]
                cand[i], cand[j] = cand[j], cand[i]
                c = eval_cost(cand)
                delta = best_cost - c
                if delta > best_delta:
                    best_delta = delta
                    best_ij = (i, j, cand, c)
            if best_ij is not None:
                _, _, cand, c = best_ij
                best_seq = cand
                best_cost = c
                dont_look = [False] * len(best_seq)
                improved = True
            return improved

        def sampled_two_opt(tries=two_opt_samples):
            nonlocal best_seq, best_cost, dont_look
            improved = False
            ncur = len(best_seq)
            if ncur <= 4:
                return False
            best_delta = 0.0
            best_pair = None
            for _ in range(tries):
                i = random.randint(0, ncur - 3)
                j = random.randint(i + 2, ncur - 1)
                cand = best_seq[:i] + list(reversed(best_seq[i:j + 1])) + best_seq[j + 1:]
                c = eval_cost(cand)
                delta = best_cost - c
                if delta > best_delta:
                    best_delta = delta
                    best_pair = (i, j, cand, c)
            if best_pair is not None:
                _, _, cand, c = best_pair
                best_seq = cand
                best_cost = c
                dont_look = [False] * len(best_seq)
                improved = True
            return improved

        # VND loop across neighborhoods
        rounds = 0
        while rounds < max_rounds:
            rounds += 1
            any_improved = False
            # Or-opt 1,2,3
            for bl in (3, 2, 1):
                if or_opt_pass(bl):
                    any_improved = True
            if adjacent_swap_pass():
                any_improved = True
            if sampled_pair_swaps():
                any_improved = True
            if sampled_two_opt():
                any_improved = True
            if not any_improved:
                break

        return best_cost, best_seq

    # ------- LNS (ruin and recreate) using sensitivity-guided removals and regret reinsertion -------
    def lns_ruin_and_recreate(seq, start_cost, rounds=lns_rounds):
        best_seq = list(seq)
        best_cost = start_cost
        no_improve = 0

        for r in range(rounds):
            current_seq = list(best_seq)
            n_cur = len(current_seq)
            if n_cur <= 6:
                break
            # Decide number to remove
            base_k = max(lns_remove_min, min(lns_remove_max, int(lns_remove_frac * n_cur)))
            # Escalate after stagnation
            k_remove = int(base_k * (1.25 if no_improve >= 2 else 1.0))
            k_remove = min(k_remove, n_cur - 2)

            remove_idx = set()

            # One (or two) contiguous segments
            blocks = 2 if (no_improve >= 2 and n_cur > 12) else 1
            remaining_quota = k_remove
            for b in range(blocks):
                if remaining_quota <= 2:
                    break
                max_len = max(2, remaining_quota // (2 if blocks == 2 else 1))
                length = random.randint(2, max_len)
                start = random.randint(0, max(0, n_cur - length))
                for j in range(start, start + length):
                    if len(remove_idx) < k_remove:
                        remove_idx.add(j)
                remaining_quota = k_remove - len(remove_idx)

            # Sensitivity-guided scattered removals
            # Compute sensitivity for K sampled indices not already chosen
            cand_indices = [i for i in range(n_cur) if i not in remove_idx]
            if cand_indices:
                sampled_idx = cand_indices if len(cand_indices) <= lns_sensitivity_K else random.sample(cand_indices, lns_sensitivity_K)
            else:
                sampled_idx = []
            sensitivities = []
            for i in sampled_idx:
                t = current_seq[i]
                base = current_seq[:i] + current_seq[i + 1:]
                positions = sample_positions(len(base))
                # sample P positions
                if len(positions) > lns_sensitivity_P:
                    positions = random.sample(positions, lns_sensitivity_P)
                costs = []
                for pos in positions:
                    cand = base[:]
                    cand.insert(pos, t)
                    costs.append(eval_cost(cand))
                if len(costs) >= 2:
                    v = statistics.pvariance(costs)
                elif costs:
                    v = 0.0
                else:
                    v = 0.0
                sensitivities.append((v, i))
            sensitivities.sort(reverse=True, key=lambda x: x[0])

            k_high = int(lns_high_sens_share * k_remove)
            for _, idx in sensitivities[:k_high]:
                if len(remove_idx) < k_remove:
                    remove_idx.add(idx)

            # Fill remaining with random
            while len(remove_idx) < k_remove:
                remove_idx.add(random.randint(0, n_cur - 1))

            remove_idx = sorted(remove_idx)

            removed = [current_seq[i] for i in remove_idx]
            base = [current_seq[i] for i in range(n_cur) if i not in remove_idx]

            # Reinsert removed using regret-guided insertion
            seq_partial = list(base)
            remaining = list(removed)
            random.shuffle(remaining)

            # Local cache for reinsertion
            re_cache = make_best_two_cache()

            while remaining:
                # sample some of the remaining to speed up
                k_t = min(len(remaining), max(4, cand_txn_sample // 2))
                cand_txns = random.sample(remaining, k_t)
                scored = []
                for t in cand_txns:
                    (c1, p1), (c2, p2) = best_two_insertions(seq_partial, t, re_cache)
                    regret = (c2 - c1) if c2 < float('inf') else 0.0
                    scored.append((c1, regret, t, p1))
                # RCL by cost and regret
                scored.sort(key=lambda x: x[0])
                rcl_cost = scored[:max(1, min(2, len(scored)))]
                scored.sort(key=lambda x: (-x[1], x[0]))
                rcl_regret = scored[:max(1, min(2, len(scored)))]

                pool = list({(c1, rg, t, p) for (c1, rg, t, p) in (rcl_cost + rcl_regret)})
                # choose best among small pool
                chosen = min(pool, key=lambda x: x[0])
                c1, rg, t, p = chosen
                seq_partial.insert(p, t)
                remaining.remove(t)

            c_new = eval_cost(seq_partial)
            if c_new < best_cost:
                best_cost = c_new
                best_seq = seq_partial
                no_improve = 0
            else:
                no_improve += 1

        return best_cost, best_seq

    # ------- Path relinking between elite sequences (bidirectional, block-aware) -------
    def quick_oropt1(seq):
        # single Or-opt(1) pass for quick polish
        best_seq = list(seq)
        best_cost = eval_cost(best_seq)
        i = 0
        while i < len(best_seq):
            t = best_seq[i]
            base = best_seq[:i] + best_seq[i + 1:]
            positions = sample_positions(len(base))
            move_best_cost = best_cost
            move_best_pos = None
            for pos in positions:
                cand = base[:]
                cand.insert(pos, t)
                c = eval_cost(cand)
                if c < move_best_cost:
                    move_best_cost = c
                    move_best_pos = pos
            if move_best_pos is not None:
                best_seq = base
                best_seq.insert(move_best_pos, t)
                best_cost = move_best_cost
            else:
                i += 1
        return best_cost, best_seq

    def longest_matching_block(cur, target, start_idx):
        # From target start_idx, find desired element and prefer moving the longest matching block at once.
        desired = target[start_idx]
        if desired not in cur:
            return start_idx, start_idx
        j = cur.index(desired)
        # Extend block while consecutive order matches target
        i_cur = j
        i_tar = start_idx
        while i_cur < len(cur) and i_tar < len(target) and cur[i_cur] == target[i_tar]:
            i_cur += 1
            i_tar += 1
        return j, i_cur - 1  # [j..end_block] in cur

    def path_relink(a_seq, b_seq, direction="A2B"):
        # Move a toward b or b toward a
        src = list(a_seq)
        dst = list(b_seq)
        cur = list(src)
        best_c = eval_cost(cur)
        best_s = list(cur)
        for i in range(len(cur)):
            # choose block movement to match target at position i
            start_j, end_j = longest_matching_block(cur, dst, i)
            if start_j == i:
                continue
            block = cur[start_j:end_j + 1]
            # remove block
            rest = cur[:start_j] + cur[end_j + 1:]
            # insert block at position i
            cand = rest[:i] + block + rest[i:]
            c = eval_cost(cand)
            if c < best_c:
                best_c = c
                best_s = list(cand)
            # quick polish
            qc, qs = quick_oropt1(cand)
            if qc < best_c:
                best_c = qc
                best_s = qs
            cur = cand
        return best_c, best_s

    # ------- Elite pool management -------
    elites = []

    def add_elite(cost, seq):
        nonlocal elites
        if not seq or len(seq) != n:
            return
        key = tuple(seq)
        for c, s in elites:
            if tuple(s) == key:
                if cost < c:
                    elites.remove((c, s))
                    elites.append((cost, list(seq)))
                return
        # enforce last-3 suffix diversity
        suffix3 = tuple(seq[-3:]) if len(seq) >= 3 else tuple(seq)
        for c, s in elites:
            suf = tuple(s[-3:]) if len(s) >= 3 else tuple(s)
            if suf == suffix3:
                # If duplicate suffix, only add if strictly better
                if cost < c:
                    elites.remove((c, s))
                    elites.append((cost, list(seq)))
                break
        else:
            elites.append((cost, list(seq)))
        elites.sort(key=lambda x: x[0])
        if len(elites) > elite_cap:
            elites = elites[:elite_cap]

    # ------- Main search orchestration -------
    best_cost = float('inf')
    best_seq = list(range(n))

    # Seed RNG lightly for diversity per call
    random.seed((n * 911 + num_seqs * 131 + 7) % (2**32 - 1))

    # 1) GRASP-style regret seeds
    for _ in range(seeds):
        c, s = construct_regret_insertion()
        c, s = vnd_local_search(s, c, max_rounds=1)
        add_elite(c, s)
        if c < best_cost:
            best_cost, best_seq = c, s

    # 2) Beam search seeds (couple of runs)
    beam_runs = max(1, num_seqs // 3)
    for _ in range(beam_runs):
        c, s = beam_search()
        c, s = vnd_local_search(s, c, max_rounds=1)
        add_elite(c, s)
        if c < best_cost:
            best_cost, best_seq = c, s

    # 3) Bidirectional path relinking among elites
    if len(elites) >= 2:
        base_cost, base_seq = elites[0]
        for i in range(1, min(len(elites), 4)):
            c_other, other = elites[i]
            c_rel1, s_rel1 = path_relink(base_seq, other, direction="A2B")
            c_rel2, s_rel2 = path_relink(other, base_seq, direction="B2A")
            for c_rel, s_rel in [(c_rel1, s_rel1), (c_rel2, s_rel2)]:
                c_rel, s_rel = vnd_local_search(s_rel, c_rel, max_rounds=1)
                add_elite(c_rel, s_rel)
                if c_rel < best_cost:
                    best_cost, best_seq = c_rel, s_rel

    # 4) Iterated LNS + VND from the current best/elite
    incumbent_cost, incumbent_seq = best_cost, best_seq
    for _ in range(ils_iterations):
        # Diversify by starting from one of elites occasionally
        if elites and random.random() < 0.5:
            start_c, start_s = random.choice(elites[:min(3, len(elites))])
        else:
            start_c, start_s = incumbent_cost, incumbent_seq

        c1, s1 = lns_ruin_and_recreate(start_s, start_c, rounds=lns_rounds)
        c2, s2 = vnd_local_search(s1, c1, max_rounds=vnd_rounds)
        if c2 < incumbent_cost:
            incumbent_cost, incumbent_seq = c2, s2
            add_elite(c2, s2)
        # small adjacent+or-opt polish
        c3, s3 = vnd_local_search(incumbent_seq, incumbent_cost, max_rounds=1)
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