# EVOLVE-BLOCK-START
"""Transaction scheduling algorithm for optimizing makespan across multiple workloads"""

import time
import random
import sys
import os
import itertools

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
    Tournament-guided beam + greedy with buddy lookahead, shared prefix dominance,
    and boundary-focused LNS for improved makespan.

    Args:
        workload: Workload object containing transaction data
        num_seqs: Controls exploration budget

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    rng = random.Random()
    rng.seed(time.time_ns() ^ os.getpid())

    n = workload.num_txns

    # Global cache for sequence costs
    cost_cache = {}

    def eval_seq(seq):
        key = tuple(seq)
        if key in cost_cache:
            return cost_cache[key]
        c = workload.get_opt_seq_cost(seq)
        cost_cache[key] = c
        return c

    # Precompute singleton and pairwise costs to expose conflict structure
    c1 = [eval_seq([i]) for i in range(n)]
    M = [[0] * n for _ in range(n)]
    for i in range(n):
        Mi = M[i]
        for j in range(n):
            Mi[j] = c1[i] if i == j else eval_seq([i, j])

    # Preference margins W: negative suggests i before j
    W = [[0] * n for _ in range(n)]
    for i in range(n):
        Wi = W[i]
        Mi = M[i]
        for j in range(n):
            Wi[j] = 0 if i == j else (Mi[j] - M[j][i])

    # Tournament order (lower score -> earlier)
    s = [0] * n
    for i in range(n):
        s[i] = sum(W[i][j] for j in range(n) if j != i)
    tournament_order = list(range(n))
    tournament_order.sort(key=lambda x: (s[x], x))

    def prefer_before(a, b):
        return M[a][b] <= M[b][a]

    def tournament_bubble_pass(seq, passes=2):
        arr = seq[:]
        for _ in range(passes):
            improved = False
            for k in range(len(arr) - 1):
                a, b = arr[k], arr[k + 1]
                if not prefer_before(a, b):
                    arr[k], arr[k + 1] = b, a
                    improved = True
            if not improved:
                break
        return arr

    # Buddy lists: top partners by small M[t][u]
    buddy_k = 8 if n >= 90 else 6
    buddies = []
    for t in range(n):
        order = sorted((u for u in range(n) if u != t), key=lambda u: M[t][u])
        buddies.append(order[:buddy_k])

    def preselect_by_tournament(prefix, remaining, k, recent_k=4):
        if not remaining:
            return []
        recents = prefix[-recent_k:] if recent_k > 0 else []
        scored = []
        for t in remaining:
            sc = 0
            for x in recents:
                sc += W[x][t]
            scored.append((sc, t))
        scored.sort(key=lambda z: (z[0], z[1]))
        return [t for _, t in scored[:k]]

    def recent_k_for_depth(d):
        frac = d / max(1, n - 1)
        return 5 if frac < 0.33 else (4 if frac < 0.66 else 3)

    # Shared prefix-dominance cache across restarts: (frozenset(remaining), suffix<=3) -> best seen cost
    prefix_dom = {}

    # Greedy completion from a prefix; used by beam search to close prefixes
    def greedy_complete(prefix):
        rem = [t for t in range(n) if t not in prefix]
        cur = prefix[:]
        while rem:
            t = min(rem, key=lambda u: eval_seq(cur + [u]))
            cur.append(t)
            rem.remove(t)
        return cur, eval_seq(cur)

    # Two-phase beam-seeded prefix explorer with buddy/tournament lookahead and dominance pruning
    def beam_seed(inc=None):
        beam_width_A = max(10, min(16, int(num_seqs) + 2))
        beam_width_B = max(8, min(12, int(num_seqs)))
        cand_per_expand_A = 12
        cand_per_expand_B = 8
        lookahead_top_A = 4
        lookahead_top_B = 3
        next_k_A = 6
        next_k_B = 5
        depth_limit = max(3, int(0.4 * n))

        # Starts: tournament-best, a good singleton, then randoms
        starts = []
        starts.append(tournament_order[0])
        topk = min(10, n)
        singles_sorted = sorted(range(n), key=lambda t: c1[t])[:topk]
        starts.append(rng.choice(singles_sorted))
        remcands = [t for t in range(n) if t not in starts]
        rng.shuffle(remcands)
        initial_width = max(beam_width_A, beam_width_B)
        starts.extend(remcands[:max(0, initial_width - len(starts))])

        beam = []
        dom = {}  # local prefix-dominance
        seen_init = set()
        for t in starts:
            seq = [t]
            rem = frozenset(set(range(n)) - {t})
            cost = eval_seq(seq)
            key = (tuple(seq), rem)
            if key in seen_init:
                continue
            seen_init.add(key)
            sig = (rem, tuple(seq[-1:]))
            prev = dom.get(sig)
            if prev is None or cost < prev:
                dom[sig] = cost
                beam.append((cost, seq, rem))

        incumbent_cost = inc if inc is not None else float('inf')
        incumbent_seq = None

        for d in range(n - 1):
            phaseA = d < depth_limit
            beam_width = beam_width_A if phaseA else beam_width_B
            cand_per_expand = cand_per_expand_A if phaseA else cand_per_expand_B
            lookahead_top = lookahead_top_A if phaseA else lookahead_top_B
            next_k = next_k_A if phaseA else next_k_B
            recent_k = recent_k_for_depth(d)
            suffix_k = 3

            # Tighten incumbent by greedily completing top beam items
            if beam:
                beam_sorted = sorted(beam, key=lambda x: x[0])
                for bc, bseq, _ in beam_sorted[:2]:
                    if bc < incumbent_cost:
                        full, fc = greedy_complete(bseq)
                        if fc < incumbent_cost:
                            incumbent_cost = fc
                            incumbent_seq = full

            next_beam = []
            local_seen = set()

            for cost, seq, rem in beam:
                if (incumbent_cost < float('inf')) and (cost >= incumbent_cost):
                    continue
                rem_list = list(rem)
                if not rem_list:
                    next_beam.append((cost, seq, rem))
                    continue

                # Tournament-guided preselection
                if len(rem_list) > cand_per_expand * 2:
                    pre = preselect_by_tournament(seq, rem_list, cand_per_expand * 2, recent_k=recent_k)
                else:
                    pre = rem_list

                # Immediate costs
                imm = []
                for t in pre:
                    c_im = eval_seq(seq + [t])
                    if (not phaseA) and (incumbent_cost < float('inf')) and (c_im >= incumbent_cost):
                        continue
                    imm.append((t, c_im))
                if not imm:
                    continue
                imm.sort(key=lambda z: z[1])

                # Buddy/tournament lookahead on top-L
                L = min(lookahead_top, len(imm))
                scored = []
                for t, imc in imm[:L]:
                    nexts = [u for u in rem_list if u != t]
                    if not nexts:
                        la = imc
                    else:
                        buddy_pref = [u for u in buddies[t] if u in nexts][:next_k]
                        if len(buddy_pref) < next_k:
                            extra = preselect_by_tournament(seq + [t], [u for u in nexts if u not in buddy_pref], next_k - len(buddy_pref), recent_k=3)
                            pool = buddy_pref + extra
                        else:
                            pool = buddy_pref
                        if not pool:
                            pool = nexts[:min(next_k, len(nexts))]
                        la = min(eval_seq(seq + [t, u]) for u in pool)
                    scored.append((t, min(imc, la)))

                # Add immediate-best for diversity
                diversity = min(max(2, cand_per_expand // 3), len(imm))
                for t, imc in imm[:diversity]:
                    scored.append((t, imc))

                # Deduplicate and expand best-k
                uniq = {}
                for t, m in scored:
                    if (t not in uniq) or (m < uniq[t]):
                        uniq[t] = m
                items = sorted(uniq.items(), key=lambda z: z[1])
                take = min(cand_per_expand, len(items))
                for t, _ in items[:take]:
                    new_seq = seq + [t]
                    new_cost = eval_seq(new_seq)
                    if (incumbent_cost < float('inf')) and (new_cost >= incumbent_cost):
                        continue
                    new_rem = rem - {t}
                    sig = (new_rem, tuple(new_seq[-suffix_k:]) if len(new_seq) >= suffix_k else tuple(new_seq))
                    prev = dom.get(sig)
                    if (prev is not None) and (new_cost >= prev):
                        continue
                    dom[sig] = new_cost
                    key = (tuple(new_seq), new_rem)
                    if key in local_seen:
                        continue
                    local_seen.add(key)
                    next_beam.append((new_cost, new_seq, new_rem))

            if not next_beam:
                break
            next_beam.sort(key=lambda x: x[0])
            beam = next_beam[:beam_width]

        if not beam:
            # Fallback: random singleton completion
            t = rng.randint(0, n - 1)
            full, fc = greedy_complete([t])
            return full, fc

        beam.sort(key=lambda x: x[0])
        _, bseq, _ = beam[0]
        full, fc = greedy_complete(bseq)
        if (incumbent_seq is not None) and (fc >= incumbent_cost):
            return incumbent_seq, incumbent_cost
        return full, fc

    # ----- Greedy constructor with tournament-guided pool and buddy lookahead -----
    def build_initial_sequence_with_pruning(incumbent_cost_ref):
        seq = []
        # Seed starting txn: choose among tournament-best and good singletons
        candidate_seeds = set()
        candidate_seeds.add(tournament_order[0])
        topk = min(10, n)
        good_singles = sorted(range(n), key=lambda t: c1[t])[:topk]
        candidate_seeds.update(good_singles)
        start = rng.choice(list(candidate_seeds))
        seq.append(start)
        remaining = [t for t in range(n) if t != start]

        step = 0
        while remaining:
            base_cost = eval_seq(seq)

            # Incumbent pruning
            if (incumbent_cost_ref[0] < float('inf')) and (base_cost >= incumbent_cost_ref[0]):
                return None  # prune

            # Prefix-dominance pruning
            sig_cur = (frozenset(remaining), tuple(seq[-3:]) if len(seq) >= 3 else tuple(seq))
            prev_cur = prefix_dom.get(sig_cur)
            if (prev_cur is not None) and (base_cost >= prev_cur):
                return None
            if (prev_cur is None) or (base_cost < prev_cur):
                prefix_dom[sig_cur] = base_cost

            # Periodic greedy completion to tighten incumbent
            if step % 10 == 0 and remaining:
                full, fc = greedy_complete(seq)
                if fc < incumbent_cost_ref[0]:
                    incumbent_cost_ref[0] = fc
                    incumbent_cost_ref[1] = full[:]

            # Tournament-guided preselection pool
            R = len(remaining)
            pool_size = 16 if n >= 90 else 12
            k_pool = min(pool_size, R) if R > pool_size else R
            cand_pool = preselect_by_tournament(seq, list(remaining), k_pool, recent_k=recent_k_for_depth(len(seq)))
            if not cand_pool:
                cand_pool = list(remaining)

            # Evaluate immediate costs within pool
            imm = [(t, eval_seq(seq + [t])) for t in cand_pool]
            imm.sort(key=lambda x: x[1])

            # Adaptive lookahead with buddies
            L = min(4 if step < 10 else 3, len(imm))
            chosen_t = imm[0][0]
            best_metric = imm[0][1]

            for t, immediate_c in imm[:L]:
                rest = [x for x in remaining if x != t]
                if not rest:
                    metric = immediate_c
                else:
                    need = 6 if n > 60 else 7
                    buddy_pref = [u for u in buddies[t] if u in rest]
                    la_pool = buddy_pref[:min(need, len(rest))]
                    if len(la_pool) < min(need, len(rest)):
                        extra = preselect_by_tournament(seq + [t], [u for u in rest if u not in la_pool], min(need - len(la_pool), len(rest) - len(la_pool)), recent_k=3)
                        la_pool.extend(extra)
                    if not la_pool:
                        la_pool = rest[:min(need, len(rest))]
                    metric = min(eval_seq(seq + [t, u]) for u in la_pool)
                if metric < best_metric:
                    best_metric = metric
                    chosen_t = t

            # Update prefix-dominance with child
            new_cost = eval_seq(seq + [chosen_t])
            child_rem = set(remaining)
            child_rem.remove(chosen_t)
            sig_child = (frozenset(child_rem), tuple((seq + [chosen_t])[-3:]) if len(seq) + 1 >= 3 else tuple(seq + [chosen_t]))
            prev_child = prefix_dom.get(sig_child)
            if (prev_child is None) or (new_cost < prev_child):
                prefix_dom[sig_child] = new_cost

            seq.append(chosen_t)
            remaining.remove(chosen_t)
            step += 1

        return seq

    # Local adjacent swap hill-climb
    def adjacent_pass(seq, current_cost):
        improved = True
        n_local = len(seq)
        while improved:
            improved = False
            for i in range(n_local - 1):
                if seq[i] == seq[i + 1]:
                    continue
                cand = seq[:]
                cand[i], cand[i + 1] = cand[i + 1], cand[i]
                c = eval_seq(cand)
                if c < current_cost:
                    seq = cand
                    current_cost = c
                    improved = True
        return seq, current_cost

    # Compute prefix costs and marginal contributions for a sequence
    def prefix_marginals(seq):
        prefix_costs = [0] * len(seq)
        c = 0
        for i in range(len(seq)):
            c = eval_seq(seq[: i + 1])
            prefix_costs[i] = c
        marg = [prefix_costs[0]] + [prefix_costs[i] - prefix_costs[i - 1] for i in range(1, len(seq))]
        return prefix_costs, marg

    def worst_violation_boundaries(seq, topm=3):
        pairs = []
        for i in range(len(seq) - 1):
            a, b = seq[i], seq[i + 1]
            pen = M[a][b] - M[b][a]  # positive when a before b is worse
            if pen > 0:
                pairs.append((pen, i))
        pairs.sort(reverse=True)
        return [i for _, i in pairs[:topm]]

    # LNS: pick hot windows by marginal sum and boundary violations; reorder via perms
    def lns_improve(seq, base_cost, budget_factor):
        best_seq = seq[:]
        best_cost = base_cost
        n_local = len(best_seq)

        # Iterations scale mildly with budget_factor and n
        iters = max(4, min(10, 2 + int(budget_factor) + n_local // 40))
        max_k = 7 if n_local >= 40 else 6

        for it in range(iters):
            # Cheap tournament-based cleanup
            tb = tournament_bubble_pass(best_seq, passes=2)
            cb = eval_seq(tb)
            if cb < best_cost:
                best_seq, best_cost = tb, cb

            # Compute marginals to locate hot regions
            _, marg = prefix_marginals(best_seq)

            # Choose window size
            k = max_k if it < 2 else rng.randint(4, max_k)
            if n_local <= k:
                break

            # Find top windows by marginal sum (sliding window)
            sums = []
            window_sum = sum(marg[0:k])
            sums.append((0, window_sum))
            for s_ in range(1, n_local - k + 1):
                window_sum += marg[s_ + k - 1] - marg[s_ - 1]
                sums.append((s_, window_sum))
            sums.sort(key=lambda x: -x[1])

            # Also consider windows around worst violated adjacencies
            viol_starts = worst_violation_boundaries(best_seq, topm=3)
            candidate_starts = []
            for v in viol_starts:
                start = max(0, min(v - (k // 2), n_local - k))
                candidate_starts.append(start)
            candidate_starts.extend([start for start, _ in sums[:2]])
            # Deduplicate while preserving order
            seen_starts = set()
            ordered_starts = []
            for st in candidate_starts:
                if st not in seen_starts:
                    seen_starts.add(st)
                    ordered_starts.append(st)

            tried_any = False
            for start in ordered_starts:
                block = best_seq[start : start + k]
                base = best_seq[:start] + best_seq[start + k :]

                # Determine permutation budget
                factorial = 1
                for i_ in range(2, k + 1):
                    factorial *= i_
                # Cap permutations to keep time in check
                if k <= 6:
                    perm_budget = min(720, factorial)
                else:
                    perm_budget = 2000  # sample for k=7

                perm_best_seq = None
                perm_best_cost = best_cost

                if factorial <= perm_budget:
                    for p in itertools.permutations(block):
                        cand_seq = base[:start] + list(p) + base[start:]
                        c = eval_seq(cand_seq)
                        if c < perm_best_cost:
                            perm_best_cost = c
                            perm_best_seq = cand_seq
                else:
                    seenp = set()
                    attempts = 0
                    while attempts < perm_budget:
                        p = tuple(rng.sample(block, k))
                        if p in seenp:
                            continue
                        seenp.add(p)
                        cand_seq = base[:start] + list(p) + base[start:]
                        c = eval_seq(cand_seq)
                        if c < perm_best_cost:
                            perm_best_cost = c
                            perm_best_seq = cand_seq
                        attempts += 1

                if perm_best_seq is not None and perm_best_cost < best_cost:
                    best_seq = perm_best_seq
                    best_cost = perm_best_cost
                    tried_any = True
                    break  # accept first improving window

            # Targeted relocate moves for top-blame transactions
            _, marg = prefix_marginals(best_seq)
            positions = sorted(range(n_local), key=lambda i: marg[i], reverse=True)[:3]
            for pos in positions:
                if pos >= len(best_seq):
                    continue
                t = best_seq[pos]
                base = best_seq[:pos] + best_seq[pos + 1 :]
                best_pos_cost = best_cost
                best_pos_idx = None
                positions_try = set([0, len(base)])
                for _ in range(8):
                    positions_try.add(rng.randint(0, len(base)))
                for j in positions_try:
                    cand = base[:j] + [t] + base[j:]
                    c = eval_seq(cand)
                    if c < best_pos_cost:
                        best_pos_cost = c
                        best_pos_idx = j
                if best_pos_idx is not None and best_pos_cost < best_cost:
                    best_seq = base[:best_pos_idx] + [t] + base[best_pos_idx:]
                    best_cost = best_pos_cost

            # Additional non-local neighborhoods

            # Block-swap two short windows around worst violated adjacencies
            if n_local >= 12:
                viols = worst_violation_boundaries(best_seq, topm=2)
                if viols:
                    k1 = 3 if n_local < 40 else 4
                    k2 = k1
                    s1 = max(0, min(viols[0] - k1 // 2, n_local - k1))
                    # Candidate second starts
                    s2_candidates = [max(0, min((viols[-1] if len(viols) > 1 else viols[0]) + k1, n_local - k2))]
                    for _ in range(2):
                        s2_candidates.append(rng.randint(0, n_local - k2))
                    improved_swap = False
                    for s2 in s2_candidates:
                        if abs(s2 - s1) < k1:  # avoid overlap
                            continue
                        block1 = best_seq[s1:s1 + k1]
                        block2 = best_seq[s2:s2 + k2]
                        if s1 < s2:
                            cand = best_seq[:s1] + block2 + best_seq[s1 + k1:s2] + block1 + best_seq[s2 + k2:]
                        else:
                            cand = best_seq[:s2] + block1 + best_seq[s2 + k2:s1] + block2 + best_seq[s1 + k1:]
                        c = eval_seq(cand)
                        if c < best_cost:
                            best_cost = c
                            best_seq = cand
                            improved_swap = True
                            break
                    if improved_swap:
                        continue

            # Block reinsert: remove a hot block (by marginal sum) and reinsert at sampled positions
            if n_local >= 10:
                kbr = 4 if n_local < 50 else 5
                sums = []
                window_sum = sum(marg[0:kbr])
                sums.append((0, window_sum))
                for s in range(1, n_local - kbr + 1):
                    window_sum += marg[s + kbr - 1] - marg[s - 1]
                    sums.append((s, window_sum))
                sums.sort(key=lambda x: -x[1])
                start = sums[0][0]
                block = best_seq[start:start + kbr]
                base = best_seq[:start] + best_seq[start + kbr:]
                positions_try = {0, len(base)}
                while len(positions_try) < 6:
                    positions_try.add(rng.randint(0, len(base)))
                best_ins = None
                best_ins_cost = best_cost
                for pos in positions_try:
                    cand = base[:pos] + block + base[pos:]
                    c = eval_seq(cand)
                    if c < best_ins_cost:
                        best_ins_cost = c
                        best_ins = cand
                if best_ins is not None and best_ins_cost < best_cost:
                    best_seq = best_ins
                    best_cost = best_ins_cost
                    continue

            if not tried_any:
                continue

        return best_seq, best_cost

    # Multi-start: beam-seeded plus greedy restarts with pruning; pick best as LNS seed
    seed_best_cost = float('inf')
    seed_best_seq = None

    # 0) Beam seed once
    bseq, bcost = beam_seed()
    bseq, bcost = adjacent_pass(bseq, bcost)
    seed_best_seq, seed_best_cost = bseq, bcost

    # 1) Greedy restarts with shared prefix dominance and incumbent pruning
    restarts = max(1, int(num_seqs))
    # Use list as mutable ref for incumbent within builder
    inc_ref = [seed_best_cost, seed_best_seq[:] if seed_best_seq is not None else None]
    for _ in range(restarts):
        seq0 = build_initial_sequence_with_pruning(inc_ref)
        if seq0 is None:
            continue
        cost0 = eval_seq(seq0)
        # quick local cleanup
        seq0, cost0 = adjacent_pass(seq0, cost0)
        if cost0 < seed_best_cost:
            seed_best_cost = cost0
            seed_best_seq = seq0[:]
            inc_ref[0] = seed_best_cost
            inc_ref[1] = seed_best_seq[:]

    # LNS improvement on best seed
    final_seq, final_cost = lns_improve(seed_best_seq, seed_best_cost, budget_factor=max(1, int(num_seqs)))

    # Small non-adjacent 2-opt pass
    two_opt_trials = min(60, n)
    for _ in range(two_opt_trials):
        i, j = rng.sample(range(n), 2)
        if abs(i - j) <= 1:
            continue
        cand = final_seq[:]
        cand[i], cand[j] = cand[j], cand[i]
        c = eval_seq(cand)
        if c < final_cost:
            final_seq = cand
            final_cost = c

    # Final light random insertion moves
    move_budget = min(60, n)
    for _ in range(move_budget):
        i, j = rng.sample(range(n), 2)
        if i == j:
            continue
        cand = final_seq[:]
        item = cand.pop(i)
        cand.insert(j, item)
        c = eval_seq(cand)
        if c < final_cost:
            final_seq = cand
            final_cost = c

    return final_cost, final_seq


def get_random_costs():
    """
    Evaluate scheduling algorithm on three different workloads.

    Returns:
        Tuple of (total_makespan, list_of_schedules, execution_time)
    """
    start_time = time.time()

    # Workload 1: Complex mixed read/write transactions
    workload = Workload(WORKLOAD_1)
    makespan1, schedule1 = get_best_schedule(workload, 12)
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