# EVOLVE-BLOCK-START
"""Transaction scheduling algorithm for optimizing makespan across multiple workloads"""

import time
import random
import sys
import os

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
    Get near-optimal schedule using cached beam search with two-step lookahead and
    strong local refinement (reinsertion + adjacent swaps).

    Args:
        workload: Workload object containing transaction data
        num_seqs: Search effort parameter (used to scale beam width and restarts)

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    n = workload.num_txns

    # Cost cache for partial prefixes to reduce repeated evaluations
    cost_cache = {}

    def eval_cost(prefix):
        """Evaluate and cache the cost for a prefix sequence."""
        key = tuple(prefix)
        if key in cost_cache:
            return cost_cache[key]
        c = workload.get_opt_seq_cost(list(prefix))
        cost_cache[key] = c
        return c

    def local_adjacent_refine(seq, max_passes=2):
        """Local improvement via adjacent swaps (strict improvement)."""
        current = list(seq)
        best_c = eval_cost(current)
        passes = 0
        improved = True
        while improved and passes < max_passes:
            improved = False
            for i in range(len(current) - 1):
                cand = current[:]
                cand[i], cand[i + 1] = cand[i + 1], cand[i]
                c = eval_cost(cand)
                if c < best_c:
                    current, best_c = cand, c
                    improved = True
            passes += 1
        return best_c, current

    def local_reinsert_refine(seq, max_passes=2, pos_sample_limit=25):
        """
        Or-opt reinsertion local search: move contiguous blocks (sizes 1..3) to the best position.
        Evaluates all positions for small n; otherwise samples interior while keeping ends.
        Best-improving acceptance with restarts per improvement.
        """
        current = list(seq)
        best_c = eval_cost(current)
        if len(current) <= 2:
            return best_c, current
        passes = 0
        while passes < max_passes:
            improved = False
            # Try larger blocks first to escape local minima
            for blk in (3, 2, 1):
                if blk > len(current):
                    continue
                indices = list(range(0, len(current) - blk + 1))
                random.shuffle(indices)
                moved = False
                for i in indices:
                    # Remove block [i : i+blk)
                    left = current[:i]
                    right = current[i + blk:]
                    base = left + right
                    block = current[i:i + blk]
                    m = len(base) + 1  # insertion slots
                    # Determine candidate positions
                    if m <= pos_sample_limit + 1:
                        positions = list(range(m))
                    else:
                        k = min(pos_sample_limit, m - 1)
                        interior = list(range(1, m - 1))
                        sampled = set(random.sample(interior, k)) if interior else set()
                        sampled.update({0, m - 1})
                        positions = sorted(sampled)
                    move_best_c = float('inf')
                    move_best_pos = None
                    for pos in positions:
                        cand = base[:]
                        cand[pos:pos] = block
                        c = eval_cost(cand)
                        if c < move_best_c:
                            move_best_c = c
                            move_best_pos = pos
                    if move_best_pos is not None and move_best_c + 1e-9 < best_c:
                        base[move_best_pos:move_best_pos] = block
                        current = base
                        best_c = move_best_c
                        improved = True
                        moved = True
                        break  # restart after an improving move
                if moved:
                    break
            passes += 1
            if not improved:
                break
        return best_c, current

    def local_pair_swaps_sampled(seq, curr_cost, tries=100):
        """
        Sampled non-adjacent pair swaps; accept improving swaps and reset attempts on improvement.
        """
        best_seq = list(seq)
        best_cost = curr_cost
        n_local = len(best_seq)
        if n_local <= 3:
            return best_cost, best_seq
        attempts = 0
        while attempts < tries:
            i = random.randint(0, n_local - 1)
            j = random.randint(0, n_local - 1)
            if i == j or abs(i - j) <= 1:
                attempts += 1
                continue
            cand = best_seq[:]
            cand[i], cand[j] = cand[j], cand[i]
            c = eval_cost(cand)
            if c < best_cost:
                best_cost = c
                best_seq = cand
                attempts = 0  # reset on improvement
            else:
                attempts += 1
        return best_cost, best_seq

    def local_segment_reversals_sampled(seq, curr_cost, tries=80):
        """
        Sampled 2-opt-like segment reversals; accept improving reversals and reset on improvement.
        """
        best_seq = list(seq)
        best_cost = curr_cost
        n_local = len(best_seq)
        if n_local <= 4:
            return best_cost, best_seq
        attempts = 0
        while attempts < tries:
            i = random.randint(0, n_local - 2)
            j = random.randint(i + 1, min(n_local - 1, i + 10))  # cap reversal length
            if j - i < 2:
                attempts += 1
                continue
            cand = best_seq[:]
            cand[i:j + 1] = reversed(cand[i:j + 1])
            c = eval_cost(cand)
            if c < best_cost:
                best_cost = c
                best_seq = cand
                attempts = 0
            else:
                attempts += 1
        return best_cost, best_seq

    def lns_ruin_and_recreate(seq, curr_cost, rounds=2):
        """
        Ruin-and-recreate using regret-guided best-two insertion repair.
        - Remove a contiguous block plus scattered items (sized ~12% of sequence, capped).
        - Reinsert removed transactions using best-two insertion and a small per-repair cache.
        - Accept if improved; repeat for given rounds.
        Returns (best_cost, best_seq).
        """
        best_seq = list(seq)
        best_cost = curr_cost
        if len(best_seq) <= 6:
            return best_cost, best_seq

        for _ in range(max(1, rounds)):
            L = len(best_seq)
            # Determine number to remove: 12% of L, within [6, 16], keep at least two elements
            k_remove = max(6, min(16, int(0.12 * L)))
            k_remove = min(k_remove, L - 2)

            # Select indices: one contiguous block plus scattered extras
            remove_idx = set()
            if L > k_remove:
                contig_len = max(2, min(k_remove - 2, max(2, L // 15)))
                start = random.randint(0, L - contig_len)
                for j in range(start, start + contig_len):
                    if len(remove_idx) < k_remove:
                        remove_idx.add(j)
            while len(remove_idx) < k_remove:
                remove_idx.add(random.randint(0, L - 1))
            remove_idx = sorted(remove_idx)

            removed = [best_seq[i] for i in remove_idx]
            base = [best_seq[i] for i in range(L) if i not in remove_idx]

            # Per-repair cache for best-two insertion evaluations
            bt_cache = {}

            def sample_positions_repair(m):
                """Anchor-based position sampling with random interiors; all positions if small."""
                if m + 1 <= 20:
                    return list(range(m + 1))
                anchors = {0, m, m // 2, m // 4, (3 * m) // 4}
                anchors = {p for p in anchors if 0 <= p <= m}
                interior = [p for p in range(1, m) if p not in anchors]
                R = min(10, len(interior))
                if R > 0:
                    anchors.update(random.sample(interior, R))
                return sorted(anchors)

            def best_two_for(seq_partial, t):
                """Return ((best_cost, best_pos), (second_cost, second_pos)) for inserting t."""
                key = (tuple(seq_partial), t)
                v = bt_cache.get(key)
                if v is not None:
                    return v
                positions = sample_positions_repair(len(seq_partial))
                best = (float('inf'), None)
                second = (float('inf'), None)
                for pos in positions:
                    cand = seq_partial[:]
                    cand.insert(pos, t)
                    c = eval_cost(cand)
                    if c < best[0]:
                        second = best
                        best = (c, pos)
                    elif c < second[0]:
                        second = (c, pos)
                bt_cache[key] = (best, second)
                return best, second

            # Regret-guided reinsertion
            seq_partial = list(base)
            remaining = list(removed)
            random.shuffle(remaining)
            while remaining:
                # Evaluate a small candidate subset, or all if small remaining
                k_t = len(remaining) if len(remaining) <= 8 else 8
                cand_txns = random.sample(remaining, k_t)
                scored = []
                for t in cand_txns:
                    (c1, p1), (c2, p2) = best_two_for(seq_partial, t)
                    regret = (c2 - c1) if c2 < float('inf') else 0.0
                    scored.append((c1, regret, t, p1))
                # Build RCL pool by cost and regret
                scored.sort(key=lambda x: x[0])
                rcl_cost = scored[:min(3, len(scored))]
                scored.sort(key=lambda x: (-x[1], x[0]))
                rcl_regret = scored[:min(2, len(scored))]
                pool = list({(c, r, t, p) for (c, r, t, p) in (rcl_cost + rcl_regret)})

                # Choose: 60% lowest best-cost, else highest regret
                if random.random() < 0.6:
                    pool.sort(key=lambda x: x[0])
                    chosen_c, chosen_r, chosen_t, chosen_p = pool[0]
                else:
                    pool.sort(key=lambda x: (-x[1], x[0]))
                    chosen_c, chosen_r, chosen_t, chosen_p = pool[0]

                seq_partial.insert(chosen_p, chosen_t)
                remaining.remove(chosen_t)

            c_new = eval_cost(seq_partial)
            if c_new < best_cost:
                best_cost = c_new
                best_seq = seq_partial

        return best_cost, best_seq

    def construct_grasp_seed():
        """
        Randomized best-insertion construction to seed the beam with a good initial sequence.
        """
        remaining = list(range(n))
        start = random.randint(0, n - 1)
        seq = [start]
        remaining.remove(start)

        def sample_positions(seq_len):
            if seq_len + 1 <= 20:
                return list(range(seq_len + 1))
            # sample interior, keep ends
            k = 12
            interior = list(range(1, seq_len))
            choose = set(random.sample(interior, min(k, len(interior)))) if interior else set()
            choose.update({0, seq_len})
            return sorted(choose)

        while remaining:
            # Candidate txns to consider
            k_txn = min(8, len(remaining))
            cand_txns = random.sample(remaining, k_txn)
            best_cost = float('inf')
            best_choice = None
            best_pos = 0
            for t in cand_txns:
                positions = sample_positions(len(seq))
                for pos in positions:
                    cand = seq[:]
                    cand.insert(pos, t)
                    c = eval_cost(cand)
                    if c < best_cost:
                        best_cost = c
                        best_choice = t
                        best_pos = pos
            if best_choice is None:
                # Fallback: append a random remaining txn
                t = random.choice(remaining)
                seq.append(t)
                remaining.remove(t)
            else:
                seq.insert(best_pos, best_choice)
                remaining.remove(best_choice)
        return eval_cost(seq), seq

    def beam_search(beam_width, cand_per_state, lookahead_k=5, alpha=0.7):
        """
        Beam search with two-step lookahead, suffix-2 diversity, and regret-weighted selection.

        beam_width: number of partial sequences to keep at each depth
        cand_per_state: number of candidates to expand per beam state when remaining is large
        lookahead_k: number of second-step candidates to sample for scoring
        alpha: weight on immediate cost vs lookahead (0..1)
        """
        # Evaluate all single-transaction starters and pick top starters
        starters = []
        for t in range(n):
            c = eval_cost([t])
            starters.append((c, [t]))
        starters.sort(key=lambda x: x[0])

        init_count = min(max(beam_width * 2, beam_width), n)
        init_beam = []
        used_prefixes = set()
        for c, seq in starters[:init_count]:
            key = tuple(seq)
            if key in used_prefixes:
                continue
            used_prefixes.add(key)
            rem = tuple(x for x in range(n) if x != seq[0])
            init_beam.append((c, seq, rem))

        # Add a GRASP-style seed to diversify
        grasp_c, grasp_seq = construct_grasp_seed()
        rem = tuple(x for x in range(n) if x not in set(grasp_seq))
        if len(grasp_seq) == n:
            init_beam.append((grasp_c, grasp_seq, ()))
        else:
            init_beam.append((grasp_c, grasp_seq, rem))

        best_complete = (float('inf'), [])
        beam = init_beam

        # Helper for suffix-2 signature
        def suffix_sig(seq):
            if not seq:
                return ()
            if len(seq) == 1:
                return (seq[-1],)
            return (seq[-2], seq[-1])

        # Progressively grow prefixes
        for depth in range(1, n + 1):
            next_candidates = []  # entries: (score, regret, c1, seq, rem, sig)
            min_remaining = n
            for cost_so_far, prefix, remaining in beam:
                rem_list = list(remaining)
                # track minimum remaining across this layer for endgame decision
                if len(rem_list) < min_remaining:
                    min_remaining = len(rem_list)

                if not rem_list:
                    # Completed
                    if cost_so_far < best_complete[0]:
                        best_complete = (cost_so_far, prefix)
                    # Keep as candidate (no further expansion)
                    next_candidates.append((cost_so_far, 0.0, cost_so_far, prefix, remaining, suffix_sig(prefix)))
                    continue

                # Decide candidate set size (endgame widening)
                if len(rem_list) <= min(cand_per_state, 16):
                    expand_list = rem_list
                else:
                    k = min(cand_per_state, len(rem_list))
                    expand_list = random.sample(rem_list, k)

                # Expand and score by immediate + adaptive lookahead
                local = []
                # Collect raw expansion stats first to compute regret span-driven alpha
                raw_moves = []  # (c1, best_c2, span, new_prefix, rem_after)
                for t in expand_list:
                    new_prefix = prefix + [t]
                    c1 = eval_cost(new_prefix)
                    rem_after = [x for x in rem_list if x != t]

                    best_c2 = c1
                    span = 0.0
                    if rem_after:
                        # Sample second-step candidates; in local endgame, evaluate all
                        endgame_local = (len(rem_list) <= 2 * beam_width)
                        k2 = len(rem_after) if endgame_local or len(rem_after) <= 6 else min(lookahead_k, len(rem_after))
                        second = rem_after if k2 == len(rem_after) else random.sample(rem_after, k2)
                        second_costs = []
                        best_c2 = float('inf')
                        for u in second:
                            c2 = eval_cost(new_prefix + [u])
                            second_costs.append(c2)
                            if c2 < best_c2:
                                best_c2 = c2
                        if second_costs:
                            span = max(second_costs) - min(second_costs)

                    if len(new_prefix) == n and c1 < best_complete[0]:
                        best_complete = (c1, new_prefix)
                    raw_moves.append((c1, best_c2, span, new_prefix, tuple(rem_after)))

                # Compute median span to adapt alpha within this state
                if raw_moves:
                    spans = sorted([s for _, _, s, _, _ in raw_moves])
                    mid = len(spans) // 2
                    median_span = spans[mid] if spans else 0.0
                    # Build local scored list with adaptive mixing
                    for c1, best_c2, span, new_prefix, rem_after in raw_moves:
                        # If high dispersion among second-step costs, rely more on lookahead
                        if span > median_span:
                            a = 0.5
                        else:
                            a = 0.8
                        score = a * c1 + (1.0 - a) * best_c2
                        local.append((score, c1, new_prefix, rem_after))

                if not local:
                    continue
                local.sort(key=lambda x: x[0])
                # compute regret for the best local choice
                best_score = local[0][0]
                second_score = local[1][0] if len(local) > 1 else best_score
                regret = max(0.0, second_score - best_score)
                # Keep a few of the best local moves for global selection
                # Widen retention in endgame for this state
                keep_k = 6 if len(rem_list) <= 2 * beam_width else 4
                keep_local = local[:min(keep_k, len(local))]
                for idx, (score, c1, seq_cand, rem_cand) in enumerate(keep_local):
                    reg = regret if idx == 0 else 0.0
                    next_candidates.append((score, reg, c1, seq_cand, rem_cand, suffix_sig(seq_cand)))

            if not next_candidates:
                break

            # Primary selection by score with suffix-2 diversity
            next_candidates.sort(key=lambda x: x[0])
            pruned = []
            used_seq = set()
            seen_suffix = set()
            endgame = (min_remaining <= 2 * beam_width)
            primary_target = max(1, int((0.6 if endgame else 0.7) * beam_width))
            for score, reg, c1, seq_cand, rem_cand, sig in next_candidates:
                key = tuple(seq_cand)
                if sig in seen_suffix or key in used_seq:
                    continue
                pruned.append((c1, seq_cand, rem_cand))
                used_seq.add(key)
                seen_suffix.add(sig)
                if len(pruned) >= primary_target:
                    break

            # Regret-boosted fill with diversity
            if len(pruned) < beam_width:
                remaining = [x for x in next_candidates if tuple(x[3]) not in used_seq]
                remaining.sort(key=lambda x: (-x[1], x[0]))  # high regret first, then better score
                for score, reg, c1, seq_cand, rem_cand, sig in remaining:
                    key = tuple(seq_cand)
                    if key in used_seq or sig in seen_suffix:
                        continue
                    pruned.append((c1, seq_cand, rem_cand))
                    used_seq.add(key)
                    seen_suffix.add(sig)
                    if len(pruned) >= beam_width:
                        break

            # If still short, fill by best score regardless of suffix
            if len(pruned) < beam_width:
                for score, reg, c1, seq_cand, rem_cand, sig in next_candidates:
                    key = tuple(seq_cand)
                    if key in used_seq:
                        continue
                    pruned.append((c1, seq_cand, rem_cand))
                    used_seq.add(key)
                    if len(pruned) >= beam_width:
                        break

            beam = pruned

        # If we didn't complete during loop, finalize from beam candidates greedily
        for c, seq, rem in beam:
            if len(seq) == n and c < best_complete[0]:
                best_complete = (c, seq)

        if best_complete[1] and len(best_complete[1]) == n:
            return best_complete

        if beam:
            # Greedy completion with best-position insertion (not only append)
            c, seq, rem = min(beam, key=lambda x: x[0])
            rem_list = list(rem)
            cur_seq = list(seq)
            while rem_list:
                best_t = None
                best_pos = 0
                best_ext_cost = float('inf')
                m = len(cur_seq)
                # position candidates: all for small m, anchored sampling otherwise
                def completion_positions(m):
                    if m + 1 <= 30:
                        return list(range(m + 1))
                    anchors = {0, m, m // 2, m // 4, (3 * m) // 4}
                    anchors = {p for p in anchors if 0 <= p <= m}
                    interior = [p for p in range(1, m) if p not in anchors]
                    k = min(20, len(interior))
                    if k > 0:
                        anchors.update(random.sample(interior, k))
                    return sorted(anchors)
                pos_list = completion_positions(m)
                for t in rem_list:
                    for pos in pos_list:
                        cand = cur_seq[:]
                        cand.insert(pos, t)
                        c2 = eval_cost(cand)
                        if c2 < best_ext_cost:
                            best_ext_cost = c2
                            best_t = t
                            best_pos = pos
                # apply best insertion
                cur_seq.insert(best_pos, best_t)
                rem_list.remove(best_t)
            final_cost = eval_cost(cur_seq)
            return final_cost, cur_seq

        # Absolute fallback: identity sequence
        identity = list(range(n))
        return eval_cost(identity), identity

    # Configure search parameters based on problem size and provided num_seqs "effort" hint
    # Keep runtime practical while improving quality over the baseline.
    if n > 30:
        beam_width = max(4, min(8, num_seqs))  # slightly wider beam for larger problems
        cand_per_state = min(24, max(10, n // 4))
    else:
        beam_width = max(4, min(10, num_seqs + 3))
        cand_per_state = min(20, max(8, n // 3))
    restarts = max(2, num_seqs // 2)  # more restarts for diversity

    best_overall_cost = float('inf')
    best_overall_seq = []

    # Multiple randomized restarts with slight seeding perturbations
    for r in range(restarts):
        random.seed((n * 131 + num_seqs * 17 + r * 911 + random.randint(0, 1_000_000)) % (2**32 - 1))

        c, seq = beam_search(beam_width=beam_width, cand_per_state=cand_per_state, lookahead_k=5, alpha=0.7)

        # Strong local refinements: reinsertion -> adjacent swaps -> sampled pair swaps -> segment reversals
        c1, s1 = local_reinsert_refine(seq, max_passes=2, pos_sample_limit=25)
        c2, s2 = local_adjacent_refine(s1, max_passes=2)
        c3, s3 = local_pair_swaps_sampled(s2, c2, tries=max(60, n // 2))
        c4, s4 = local_segment_reversals_sampled(s3, c3, tries=max(50, n // 2))
        # Quick polish
        c5, s5 = local_reinsert_refine(s4, max_passes=1, pos_sample_limit=20)
        c6, s6 = local_adjacent_refine(s5, max_passes=1)

        # Adopt best among refined variants
        best_local_cost = c
        best_local_seq = seq
        for cand_cost, cand_seq in [(c1, s1), (c2, s2), (c3, s3), (c4, s4), (c5, s5), (c6, s6)]:
            if cand_cost < best_local_cost:
                best_local_cost, best_local_seq = cand_cost, cand_seq

        # Regret-guided LNS to escape local minima, then quick polish
        lns_rounds = 3 if n <= 60 else 2
        c_lns, s_lns = lns_ruin_and_recreate(best_local_seq, best_local_cost, rounds=lns_rounds)
        if c_lns < best_local_cost:
            # Quick polish after reconstruction
            c_pol1, s_pol1 = local_reinsert_refine(s_lns, max_passes=1, pos_sample_limit=20)
            c_pol2, s_pol2 = local_adjacent_refine(s_pol1, max_passes=1)
            # Choose best among reconstructed and polished
            c, seq = c_lns, s_lns
            for cand_cost, cand_seq in [(c_pol1, s_pol1), (c_pol2, s_pol2)]:
                if cand_cost < c:
                    c, seq = cand_cost, cand_seq
        else:
            c, seq = best_local_cost, best_local_seq

        if c < best_overall_cost:
            best_overall_cost, best_overall_seq = c, seq

    # Safety check
    assert len(best_overall_seq) == n and len(set(best_overall_seq)) == n, "Schedule must include each transaction exactly once"

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