# EVOLVE-BLOCK-START
"""Transaction scheduling algorithm for optimizing makespan across multiple workloads"""

import time
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
    Find a low-makespan schedule using deterministic-stratified regret insertion with
    best-two caching, adaptive beam selection by regret dispersion, strong VND
    (Or-opt + DLB swaps + true 2-opt), sensitivity-guided LNS with light polish,
    and bidirectional, block-aware path relinking among elites.

    Args:
        workload: Workload object containing transaction data
        num_seqs: Controls search breadth (number of seeds, sampling sizes)

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

    # Best-two insertion cache reused across build and LNS episodes
    # key: (tuple(seq), txn, mode) where mode in ('ALL', 'STRAT')
    best_two_cache = {}

    # ----------------------------
    # Parameters (adaptive)
    # ----------------------------
    elite_size = max(4, min(8, 3 + num_seqs // 2))
    seed_elite_singletons = max(3, min(8, int(1.2 * num_seqs)))
    seed_random_additional = max(1, min(4, int((num_seqs + 1) // 3)))

    # Candidate sampling sizes
    k_txn_sample = min(18, max(8, 2 + int(1.5 * num_seqs)))  # txns per insertion
    base_R_positions = min(8, max(4, 1 + int(1.2 * num_seqs)))  # deterministic interior samples
    rem_all_threshold = 16                                      # when few remain, consider all txns

    # Regret selection probability (exploration vs exploitation)
    regret_prob = 0.62

    # Beam for constructive insertion (adaptive)
    local_beam_width_base = max(3, min(7, 2 + num_seqs // 2))
    high_regret_quota_ratio_base = 0.3
    diversity_suffix_k_base = 2
    endgame_all_pos_threshold = max(6, min(14, 2 + num_seqs))   # when few remain, try all positions

    # Local search parameters
    ls_adj_rounds_max = 2
    two_opt_trials = min(220, max(60, 4 * n))                   # sampled 2-opt pairs
    swap_trials = min(400, max(80, 4 * n))                      # sampled non-adjacent swaps
    reinsertion_focus_span = 3                                  # neighborhood focus window
    reinsertion_positions_cap = 10

    # Iterated local search (ILS) / perturbation
    ils_rounds = max(2, min(6, 1 + num_seqs // 3))
    perturb_swap_count = max(2, min(6, 2 + num_seqs // 3))
    perturb_block_len = max(3, min(10, 3 + num_seqs // 2))

    # Large Neighborhood Search (LNS)
    lns_iters = max(3, min(7, 2 + num_seqs // 2))
    destroy_frac_range = (0.10, 0.22)
    sensitivity_K = lambda m: max(8, min(20, m))                # sampled indices for sensitivity
    sensitivity_P = 6                                           # positions per sensitivity index
    stagnation_escalate_after = 2
    escalate_factor = 1.25

    # ----------------------------
    # Helpers
    # ----------------------------
    def deterministic_stratified_positions(seq_len, R=None):
        """Deterministic stratified insertion positions: anchors + fixed interior samples."""
        if seq_len <= 1:
            return [0, seq_len]
        if seq_len <= 12:
            return list(range(seq_len + 1))
        R = base_R_positions if R is None else R
        anchors = {0, seq_len, seq_len // 2, seq_len // 4, (3 * seq_len) // 4}
        # Deterministic pseudo-random interior positions
        rng = random.Random(911 * (seq_len + 17) + 1337)
        for _ in range(R):
            anchors.add(rng.randint(0, seq_len))
        return sorted(anchors)

    def evaluate_best_two_positions_cached(base_seq, t, use_all):
        """Return (best_cost, best_pos, second_best_cost) using deterministic position sets and cache."""
        seq_key = tuple(base_seq)
        mode = 'ALL' if use_all else 'STRAT'
        key = (seq_key, t, mode)
        if key in best_two_cache:
            return best_two_cache[key]
        if use_all or len(base_seq) <= 12:
            pos_list = list(range(len(base_seq) + 1))
        else:
            pos_list = deterministic_stratified_positions(len(base_seq))
        best_c = float('inf')
        best_p = 0
        second_c = float('inf')
        for p in pos_list:
            cand = base_seq[:p] + [t] + base_seq[p:]
            c = seq_cost(cand)
            if c < best_c:
                second_c = best_c
                best_c = c
                best_p = p
            elif c < second_c:
                second_c = c
        res = (best_c, best_p, second_c)
        best_two_cache[key] = res
        return res

    def diversity_sig(seq, k):
        k = min(k, len(seq))
        if k <= 0:
            return ()
        return tuple(seq[-k:])

    # ----------------------------
    # Construction (regret-insertion with adaptive beam and dispersion scoring)
    # ----------------------------
    def regret_insertion_build(seed_t=None):
        if seed_t is None:
            seed_t = random.randint(0, n - 1)

        beam_width = local_beam_width_base
        regret_quota_ratio = high_regret_quota_ratio_base
        suffix_k = diversity_suffix_k_base

        seq0 = [seed_t]
        rem0 = set(all_txns)
        rem0.remove(seed_t)
        beam = [(seq0, rem0, seq_cost(seq0))]

        # Early shaping: prepend/append best second element
        if rem0:
            expansions = []
            cand_txns = list(rem0)
            if len(cand_txns) > k_txn_sample:
                cand_txns = random.sample(cand_txns, k_txn_sample)
            for t in cand_txns:
                for new_seq in ([t] + seq0, seq0 + [t]):
                    c = seq_cost(new_seq)
                    new_rem = set(rem0)
                    new_rem.remove(t)
                    expansions.append((new_seq, new_rem, c, 0.0, t, c))  # score placeholder includes second=c
            # Select initial beam by cost with suffix diversity
            expansions.sort(key=lambda x: x[2])
            next_beam = []
            seen_sig = set()
            for s, r, c, _, _t, _sec in expansions:
                sig = diversity_sig(s, suffix_k)
                if sig in seen_sig:
                    continue
                seen_sig.add(sig)
                next_beam.append((s, r, c))
                if len(next_beam) >= beam_width:
                    break
            beam = next_beam if next_beam else beam

        # Grow sequences
        while True:
            if all(len(rem) == 0 for _, rem, _ in beam):
                break

            # Endgame adaptation
            min_rem = min(len(rem) for _, rem, _ in beam)
            endgame = (min_rem <= 2 * beam_width)
            if endgame:
                beam_width = min(beam_width + 2, local_beam_width_base + 3)
                regret_quota_ratio = 0.4
                suffix_k = 3

            expansions = []
            seen_seqs = set()

            for seq, rem, _ in beam:
                if not rem:
                    key = tuple(seq)
                    if key not in seen_seqs:
                        seen_seqs.add(key)
                        expansions.append((seq, rem, seq_cost(seq), 0.0, None, seq_cost(seq)))
                    continue

                # Candidate transactions
                if len(rem) <= rem_all_threshold or endgame:
                    cand_txns = list(rem)
                else:
                    cand_txns = random.sample(list(rem), min(k_txn_sample, len(rem)))

                use_all_pos = endgame or (len(rem) <= endgame_all_pos_threshold) or (len(seq) <= 18)

                for t in cand_txns:
                    best_c, best_p, second_c = evaluate_best_two_positions_cached(seq, t, use_all_pos)
                    new_seq = seq[:best_p] + [t] + seq[best_p:]
                    new_rem = set(rem)
                    new_rem.remove(t)
                    key = tuple(new_seq)
                    regret = (second_c - best_c) if second_c < float('inf') else 0.0
                    if key in seen_seqs:
                        continue
                    seen_seqs.add(key)
                    expansions.append((new_seq, new_rem, best_c, regret, t, second_c))

            if not expansions:
                break

            # Regret dispersion-based scoring
            second_vals = [sec for *_r, sec in expansions]
            if second_vals:
                max_second = max(second_vals)
                min_second = min(second_vals)
                regrets = [reg for *_x, reg, _t, sec in [(e[2], e[3], e[4], e[5]) for e in expansions]]
                med_regret = statistics.median(regrets) if regrets else 0.0
                dispersion_high = (max_second - min_second) > med_regret
            else:
                dispersion_high = False
            alpha = 0.5 if dispersion_high else 0.8

            # Compute composite score and sort
            scored = []
            for s, r, best_c, reg, t, second_c in expansions:
                score = alpha * best_c + (1 - alpha) * (second_c if second_c < float('inf') else best_c)
                scored.append((score, best_c, -reg, s, r))

            scored.sort(key=lambda x: (x[0], x[1], x[2]))

            # Next beam with diversity and regret quota
            next_beam = []
            seen_sig = set()
            base_quota = max(1, int(beam_width * (1.0 - regret_quota_ratio)))
            regret_quota = max(0, beam_width - base_quota)

            # Fill base quota by composite score
            idx = 0
            while len(next_beam) < base_quota and idx < len(scored):
                _, best_c, _neg_reg, s, r = scored[idx]
                sig = diversity_sig(s, suffix_k)
                if sig not in seen_sig:
                    seen_sig.add(sig)
                    next_beam.append((s, r, best_c))
                idx += 1

            # Fill regret quota by highest regret among remaining
            # Build list sorted by regret desc then best_c
            by_regret = sorted(expansions, key=lambda e: (-e[3], e[2]))
            jdx = 0
            while len(next_beam) < base_quota + regret_quota and jdx < len(by_regret):
                s, r, best_c, reg, _t, _sec = by_regret[jdx]
                sig = diversity_sig(s, suffix_k)
                if sig not in seen_sig:
                    seen_sig.add(sig)
                    next_beam.append((s, r, best_c))
                jdx += 1

            # If still short, fill by composite score
            kdx = 0
            while len(next_beam) < beam_width and kdx < len(scored):
                _, best_c, _nreg, s, r = scored[kdx]
                sig = diversity_sig(s, suffix_k)
                if sig not in seen_sig:
                    seen_sig.add(sig)
                    next_beam.append((s, r, best_c))
                kdx += 1

            if not next_beam:
                # Fallback by best cost
                expansions.sort(key=lambda x: x[2])
                next_beam = [(s, r, c) for s, r, c, _reg, _t, _sec in expansions[:beam_width]]

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
    # Local refinement: VND with Or-opt (1..3), DLB swaps, and true sampled 2-opt
    # ----------------------------
    def local_refine(seq):
        best_seq = seq[:]
        best_cost = seq_cost(best_seq)

        def try_reinsertion(cur_seq, cur_cost):
            L = len(cur_seq)
            for i in range(L):
                item = cur_seq[i]
                base = cur_seq[:i] + cur_seq[i + 1:]
                if len(base) <= 20:
                    positions = list(range(len(base) + 1))
                else:
                    # focus around original position + deterministic positions
                    positions = deterministic_stratified_positions(len(base))
                    # also add neighborhood around i
                    for d in range(-reinsertion_focus_span, reinsertion_focus_span + 1):
                        p = i + d
                        if 0 <= p <= len(base):
                            positions.append(p)
                    # cap to avoid explosion
                    if len(positions) > reinsertion_positions_cap + 5:
                        positions = sorted(set(positions))[:reinsertion_positions_cap + 5]
                for p in positions:
                    cand = base[:p] + [item] + base[p:]
                    if cand == cur_seq:
                        continue
                    c = seq_cost(cand)
                    if c < cur_cost:
                        return True, c, cand
            return False, cur_cost, cur_seq

        def try_or_opt(cur_seq, cur_cost, k):
            L = len(cur_seq)
            if L <= k:
                return False, cur_cost, cur_seq
            for i in range(L - k + 1):
                block = cur_seq[i:i + k]
                base = cur_seq[:i] + cur_seq[i + k:]
                if len(base) <= 20:
                    positions = list(range(len(base) + 1))
                else:
                    positions = deterministic_stratified_positions(len(base))
                for p in positions:
                    if p == i:
                        continue
                    cand = base[:p] + block + base[p:]
                    if cand == cur_seq:
                        continue
                    c = seq_cost(cand)
                    if c < cur_cost:
                        return True, c, cand
            return False, cur_cost, cur_seq

        def try_swaps_DLB(cur_seq, cur_cost):
            L = len(cur_seq)
            dlb = [False] * L
            improved_any = False
            no_change_iters = 0
            max_iters = 2  # passes
            passes = 0
            while passes < max_iters:
                changed_in_pass = False
                for i in range(L):
                    if dlb[i]:
                        continue
                    improved_i = False
                    trials = 0
                    while trials < 1 + (swap_trials // L):
                        j = random.randint(0, L - 1)
                        if j == i or abs(j - i) == 1:
                            trials += 1
                            continue
                        cand = cur_seq[:]
                        cand[i], cand[j] = cand[j], cand[i]
                        c = seq_cost(cand)
                        trials += 1
                        if c < cur_cost:
                            cur_seq = cand
                            cur_cost = c
                            dlb[i] = False
                            if j < len(dlb):
                                dlb[j] = False
                            improved_i = True
                            improved_any = True
                            changed_in_pass = True
                            break
                    if not improved_i:
                        dlb[i] = True
                if not changed_in_pass:
                    no_change_iters += 1
                    if no_change_iters >= 1:
                        break
                passes += 1
            return improved_any, cur_cost, cur_seq

        def try_2opt_segments(cur_seq, cur_cost):
            L = len(cur_seq)
            tried = 0
            improved = False
            while tried < two_opt_trials:
                i = random.randint(0, L - 2)
                j = random.randint(i + 2, L - 1) if i + 2 < L else None
                tried += 1
                if j is None:
                    continue
                if j == i + 1:
                    continue
                cand = cur_seq[:i] + cur_seq[i:j + 1][::-1] + cur_seq[j + 1:]
                c = seq_cost(cand)
                if c < cur_cost:
                    cur_cost = c
                    cur_seq = cand
                    improved = True
                    # continue exploring within budget
            return improved, cur_cost, cur_seq

        improved_outer = True
        adj_rounds = 0

        while improved_outer:
            improved_outer = False

            # Or-opt blocks (3,2,1) typically strong
            for blk in (3, 2, 1):
                changed, nc, ns = try_or_opt(best_seq, best_cost, blk)
                if changed:
                    best_seq, best_cost = ns, nc
                    improved_outer = True
                    break
            if improved_outer:
                continue

            # Reinsertion k=1
            changed, nc, ns = try_reinsertion(best_seq, best_cost)
            if changed:
                best_seq, best_cost = ns, nc
                improved_outer = True
                continue

            # Adjacent swap pass (limited rounds)
            if adj_rounds < ls_adj_rounds_max:
                improved = True
                while improved:
                    improved = False
                    for i in range(n - 1):
                        cand = best_seq[:]
                        cand[i], cand[i + 1] = cand[i + 1], cand[i]
                        c = seq_cost(cand)
                        if c < best_cost:
                            best_cost = c
                            best_seq = cand
                            improved = True
                            improved_outer = True
                            break
                adj_rounds += 1
                if improved_outer:
                    continue

            # Non-adjacent swaps under simple DLB
            changed, nc, ns = try_swaps_DLB(best_seq, best_cost)
            if changed:
                best_seq, best_cost = ns, nc
                improved_outer = True
                continue

            # True sampled 2-opt via segment reversal
            changed, nc, ns = try_2opt_segments(best_seq, best_cost)
            if changed:
                best_seq, best_cost = ns, nc
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
                if i == j:
                    continue
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
    # LNS: sensitivity-guided destroy and regret-based repair with quick polish
    # ----------------------------
    def compute_move_sensitivity(seq, indices, P):
        """Return list of (score, idx) where score is variance of costs when moving txn at idx."""
        scores = []
        L = len(seq)
        for idx in indices:
            item = seq[idx]
            base = seq[:idx] + seq[idx + 1:]
            # deterministic 6 positions
            if len(base) <= 12:
                positions = list(range(len(base) + 1))
            else:
                positions = deterministic_stratified_positions(len(base), R=P)
            vals = []
            for p in positions:
                cand = base[:p] + [item] + base[p:]
                vals.append(seq_cost(cand))
                if len(vals) >= P:
                    break
            if len(vals) >= 2:
                var = statistics.pvariance(vals)
            else:
                var = 0.0
            scores.append((var, idx))
        scores.sort(reverse=True)
        return scores

    def lns_attempt(seq, escalate=False):
        cur = seq[:]
        base_frac = random.uniform(*destroy_frac_range)
        frac = min(0.5, base_frac * (escalate_factor if escalate else 1.0))
        m = max(4, min(n // 2, int(frac * n)))
        # Sensitivity-guided picks
        K = sensitivity_K(m)
        sample_indices = random.sample(range(n), min(K, n))
        sens = compute_move_sensitivity(cur, sample_indices, sensitivity_P)
        pick_cnt = max(1, int(0.4 * m))
        sens_idxs = [idx for _score, idx in sens[:pick_cnt]]

        # Remaining by contiguous block(s)
        rem_cnt = m - len(sens_idxs)
        remove_idxs = set(sens_idxs)
        if rem_cnt > 0:
            if escalate and rem_cnt >= 2:
                # two blocks
                b1 = rem_cnt // 2
                b2 = rem_cnt - b1
                start1 = random.randint(0, n - b1)
                start2 = random.randint(0, n - b2)
                for i in range(start1, start1 + b1):
                    remove_idxs.add(i)
                for i in range(start2, start2 + b2):
                    remove_idxs.add(i)
            else:
                start = random.randint(0, n - rem_cnt)
                for i in range(start, start + rem_cnt):
                    remove_idxs.add(i)

        remove_idxs = sorted(remove_idxs)
        removed = [cur[i] for i in remove_idxs]
        remaining = [cur[i] for i in range(n) if i not in set(remove_idxs)]

        # Repair using regret insertion with cached best-two
        seq_rep = remaining[:]
        rem_list = removed[:]
        while rem_list:
            cand_txns = rem_list if len(rem_list) <= k_txn_sample else random.sample(rem_list, k_txn_sample)
            use_all_pos = (len(rem_list) <= 2 * local_beam_width_base) or (len(seq_rep) <= 18)
            best_overall = (float('inf'), None, None)  # cost, txn, new_seq
            best_regret = (float('-inf'), None, None)
            for t in cand_txns:
                best_c, best_p, second_c = evaluate_best_two_positions_cached(seq_rep, t, use_all_pos)
                new_seq = seq_rep[:best_p] + [t] + seq_rep[best_p:]
                if best_c < best_overall[0]:
                    best_overall = (best_c, t, new_seq)
                regret = (second_c - best_c) if second_c < float('inf') else 0.0
                if regret > best_regret[0]:
                    best_regret = (regret, t, new_seq)
            pick_regret = (random.random() < regret_prob)
            chosen = best_regret if pick_regret and best_regret[1] is not None else best_overall
            if chosen[1] is None:
                # Fallback
                t = random.choice(rem_list)
                seq_rep = seq_rep + [t]
                rem_list.remove(t)
            else:
                seq_rep = chosen[2]
                rem_list.remove(chosen[1])

        # Quick polish: single reinsertion + one 2-opt sample
        c_rep, s_rep = local_refine_light(seq_rep)
        return c_rep, s_rep

    def local_refine_light(seq):
        """A quick polish: one reinsertion sweep and a single sampled 2-opt pass."""
        best_seq = seq[:]
        best_cost = seq_cost(best_seq)

        # Reinsertion first-improvement
        L = len(best_seq)
        for i in range(L):
            item = best_seq[i]
            base = best_seq[:i] + best_seq[i + 1:]
            positions = deterministic_stratified_positions(len(base))
            for p in positions:
                cand = base[:p] + [item] + base[p:]
                if cand == best_seq:
                    continue
                c = seq_cost(cand)
                if c < best_cost:
                    best_seq, best_cost = cand, c
                    break

        # One sampled 2-opt pass
        trials = min(100, max(30, n))
        for _ in range(trials):
            i = random.randint(0, n - 2)
            j = random.randint(i + 2, n - 1) if i + 2 < n else None
            if j is None:
                continue
            cand = best_seq[:i] + best_seq[i:j + 1][::-1] + best_seq[j + 1:]
            c = seq_cost(cand)
            if c < best_cost:
                best_seq, best_cost = cand, c

        return best_cost, best_seq

    # ----------------------------
    # Path Relinking: bidirectional, block-aware with micro-polish
    # ----------------------------
    def block_aware_move_towards(source_seq, target_seq, max_steps):
        """Move source towards target by single-item or longest block align; return best along path."""
        pos_in_target = {t: i for i, t in enumerate(target_seq)}
        s = source_seq[:]
        best_c = seq_cost(s)
        best_s = s[:]
        steps = 0

        def micro_polish(seq):
            # very light Or-opt(1) around last change
            L = len(seq)
            rng_positions = deterministic_stratified_positions(L)
            # sample two items to reinsert quickly
            for _ in range(2):
                i = random.randint(0, L - 1)
                item = seq[i]
                base = seq[:i] + seq[i + 1:]
                best_c_local = float('inf')
                best_seq_local = None
                for p in rng_positions[:6]:
                    cand = base[:p] + [item] + base[p:]
                    c = seq_cost(cand)
                    if c < best_c_local:
                        best_c_local = c
                        best_seq_local = cand
                if best_seq_local is not None and best_c_local < seq_cost(seq):
                    seq = best_seq_local
            return seq

        while steps < max_steps:
            # Find item with largest displacement
            disp_list = []
            for i, x in enumerate(s):
                j = pos_in_target[x]
                disp_list.append((abs(i - j), i, j))
            disp_list.sort(reverse=True)
            if not disp_list or disp_list[0][0] == 0:
                break
            _, i, j_target = disp_list[0]

            # Candidate 1: move single item to desired index
            item = s[i]
            base = s[:i] + s[i + 1:]
            j_clamped = max(0, min(j_target, len(base)))
            cand1 = base[:j_clamped] + [item] + base[j_clamped:]
            c1 = seq_cost(cand1)

            # Candidate 2: longest consecutive block from i that matches target adjacency
            # Grow block [i..k] such that target positions are consecutive increasing
            k = i
            while k + 1 < len(s):
                next_item = s[k + 1]
                if pos_in_target[next_item] == pos_in_target[s[k]] + 1:
                    k += 1
                else:
                    break
            block = s[i:k + 1]
            base2 = s[:i] + s[k + 1:]
            desired_start = pos_in_target[block[0]]
            desired_start = max(0, min(desired_start, len(base2)))
            cand2 = base2[:desired_start] + block + base2[desired_start:]
            c2 = seq_cost(cand2)

            if c1 <= c2:
                s = cand1
                cur_c = c1
            else:
                s = cand2
                cur_c = c2

            # Micro polish
            s = micro_polish(s)
            cur_c = seq_cost(s)

            if cur_c < best_c:
                best_c = cur_c
                best_s = s[:]
            steps += 1

        return best_c, best_s

    # ----------------------------
    # Seeding and elite management
    # ----------------------------
    # Evaluate all singletons; pick elite seeds + some random distinct seeds
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
        # keep sequences unique by suffix-3 to maintain diversity
        uniq = []
        seen_sig = set()
        for c1, s1 in elite:
            sig = diversity_sig(s1, 3)
            if sig in seen_sig:
                continue
            seen_sig.add(sig)
            uniq.append((c1, s1))
            if len(uniq) >= elite_size:
                break
        elite = uniq

    # Build and refine from seeds
    for seed in seed_txns:
        # Reset best-two cache per build episode for cleanliness
        best_two_cache.clear()
        seq0 = regret_insertion_build(seed)
        c1, s1 = local_refine(seq0)
        add_elite(c1, s1)

    # If no elite (shouldn't happen), fallback
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
    # LNS destroy-and-repair with escalation on stagnation
    # ----------------------------
    no_improve = 0
    for _ in range(lns_iters):
        best_two_cache.clear()
        c3, s3 = lns_attempt(best_overall_seq, escalate=(no_improve >= stagnation_escalate_after))
        if c3 < best_overall_cost:
            best_overall_cost, best_overall_seq = c3, s3
            add_elite(c3, s3)
            no_improve = 0
        else:
            no_improve += 1

    # ----------------------------
    # Path Relinking among elites (bidirectional, block-aware)
    # ----------------------------
    if len(elite) >= 2:
        base_cost, base_seq = best_overall_cost, best_overall_seq
        partners = elite[1:min(len(elite), elite_size)]
        max_steps = max(8, min(14, n // 6))
        for c_t, s_t in partners:
            # A -> B
            pr_c1, pr_s1 = block_aware_move_towards(base_seq, s_t, max_steps=max_steps)
            if pr_c1 < best_overall_cost:
                best_overall_cost, best_overall_seq = pr_c1, pr_s1
            # B -> A
            pr_c2, pr_s2 = block_aware_move_towards(s_t, base_seq, max_steps=max_steps)
            if pr_c2 < best_overall_cost:
                best_overall_cost, best_overall_seq = pr_c2, pr_s2

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