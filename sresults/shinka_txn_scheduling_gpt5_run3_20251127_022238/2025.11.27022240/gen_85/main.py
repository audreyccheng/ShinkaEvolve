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
    Portfolio beam+greedy with unified prefix-dominance and strengthened LNS.

    Args:
        workload: Workload object containing transaction data
        num_seqs: Exploration budget (~beam width, restarts, and LNS effort)

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    n = workload.num_txns
    rng = random.Random(1729 + n)  # deterministic per workload

    # ----------------- Evaluator with shared memoization -----------------
    cost_cache = {}

    def seq_cost(seq):
        key = tuple(seq)
        if key in cost_cache:
            return cost_cache[key]
        c = workload.get_opt_seq_cost(seq)
        cost_cache[key] = c
        return c

    # ----------------- Model precomputation (pairwise/tournament) -----------------
    c1 = [seq_cost([i]) for i in range(n)]

    M = [[0] * n for _ in range(n)]
    for i in range(n):
        Mi = M[i]
        for j in range(n):
            Mi[j] = c1[i] if i == j else seq_cost([i, j])

    # W[i][j] = M[i][j] - M[j][i] (positive: i before j is worse)
    W = [[0] * n for _ in range(n)]
    for i in range(n):
        Wi = W[i]
        Mi = M[i]
        for j in range(n):
            Wi[j] = 0 if i == j else (Mi[j] - M[j][i])

    # Tournament ranking (smaller sum => earlier)
    s = [sum(W[i][j] for j in range(n) if j != i) for i in range(n)]
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

    # Buddy lists by small M[i][j]
    buddy_k = 8 if n >= 90 else 6
    buddies = []
    for t in range(n):
        order = sorted((u for u in range(n) if u != t), key=lambda u: M[t][u])
        buddies.append(order[:buddy_k])

    # Anti-buddy gating threshold (75th percentile of W[row])
    def q75_row(vals):
        if not vals:
            return 0
        v = sorted(vals)
        idx = int(0.75 * (len(v) - 1))
        return v[idx]

    qcut = [q75_row([W[i][j] for j in range(n) if j != i]) for i in range(n)]

    def preselect_by_tournament(prefix, remaining, k, recent_k=4):
        if not remaining or k <= 0:
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

    def recent_k_for_depth(depth):
        frac = depth / max(1, n - 1)
        return 5 if frac < 0.33 else (4 if frac < 0.66 else 3)

    # Shared prefix-dominance across builders: key = (frozenset(remaining), suffix<=k)
    prefix_dom = {}

    # Greedy-complete a prefix, given remaining set
    def greedy_complete(prefix, remaining_fset=None):
        if remaining_fset is None:
            remaining = [t for t in range(n) if t not in prefix]
        else:
            remaining = list(remaining_fset)
        cur = list(prefix)
        while remaining:
            t = min(remaining, key=lambda u: seq_cost(cur + [u]))
            cur.append(t)
            remaining.remove(t)
        return cur, seq_cost(cur)

    # ----------------- Beam-seeded prefix explorer -----------------
    def beam_seed(beam_width, cand_per_expand, lookahead_top, next_k, incumbent_cost=float('inf')):
        # Diverse starts: tournament-best, a good singleton, then random
        starts = []
        starts.append(tournament_order[0])
        tops = sorted(range(n), key=lambda t: c1[t])[:min(10, n)]
        starts.append(rng.choice(tops))
        remcands = [t for t in range(n) if t not in starts]
        rng.shuffle(remcands)
        starts.extend(remcands[:max(0, beam_width - len(starts))])

        # Initialize beam with shared prefix dominance
        beam = []
        seen_init = set()
        for t in starts:
            seq = [t]
            rem = frozenset(set(range(n)) - {t})
            c = seq_cost(seq)
            key = (tuple(seq), rem)
            if key in seen_init:
                continue
            seen_init.add(key)
            sig = (rem, tuple(seq[-1:]))
            prev = prefix_dom.get(sig)
            if prev is None or c < prev:
                prefix_dom[sig] = c
                beam.append((c, seq, rem))

        if not beam:
            # fallback to random seed completion
            t = rng.randint(0, n - 1)
            full, fc = greedy_complete([t])
            return full, fc

        # Depth expansion with adaptive suffix length and incumbent promotion
        depth = 1
        best_inc_seq = None
        best_inc_cost = incumbent_cost

        depth_limit = max(3, int(0.45 * n))
        while depth < min(depth_limit, n) and beam:
            next_beam = []
            local_seen = set()
            suffix_k = 3 if (depth / max(1, n)) < 0.7 else 4
            # Opportunistic incumbent tightening on top beam items
            beam_sorted = sorted(beam, key=lambda x: x[0])[:2]
            for bc, bseq, brem in beam_sorted:
                if bc < best_inc_cost:
                    full, fc = greedy_complete(bseq, brem)
                    if fc < best_inc_cost:
                        best_inc_cost = fc
                        best_inc_seq = full

            for cost, seq, rem in beam:
                if best_inc_cost < float('inf') and cost >= best_inc_cost:
                    continue
                rem_list = list(rem)
                if not rem_list:
                    next_beam.append((cost, seq, rem))
                    continue

                # Preselect candidates
                recent_k = recent_k_for_depth(depth)
                pre = preselect_by_tournament(seq, rem_list, cand_per_expand * 2, recent_k=recent_k) if len(rem_list) > cand_per_expand * 2 else rem_list

                # Immediate costs
                imm = []
                for t in pre:
                    c_im = seq_cost(seq + [t])
                    if best_inc_cost < float('inf') and c_im >= best_inc_cost:
                        continue
                    imm.append((t, c_im))
                if not imm:
                    continue
                imm.sort(key=lambda z: z[1])
                best_immediate = imm[0][1]

                # Lookahead on top-L with anti-buddy guard
                L = min(lookahead_top, len(imm))
                scored = []
                last = seq[-1] if seq else None
                for t, imc in imm[:L]:
                    if last is not None and W[last][t] > 0 and W[last][t] >= qcut[last] and imc > best_immediate * 1.01:
                        continue
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
                        la = min(seq_cost(seq + [t, u]) for u in pool)
                    scored.append((t, min(imc, la)))

                # Add some immediate-best for diversity
                diversity = min(max(2, cand_per_expand // 3), len(imm))
                for t, imc in imm[:diversity]:
                    if last is not None and W[last][t] > 0 and W[last][t] >= qcut[last] and imc > best_immediate * 1.01:
                        continue
                    scored.append((t, imc))

                # Unique and rank
                uniq = {}
                for t, m in scored:
                    if (t not in uniq) or (m < uniq[t]):
                        uniq[t] = m
                items = sorted(uniq.items(), key=lambda z: z[1])

                # Expand: greedy-complete top K children per parent; prune against incumbent
                per_parent_greedy_K = 2 if depth < int(0.6 * n) else 1
                taken = 0
                for t, _ in items[:cand_per_expand]:
                    new_seq = seq + [t]
                    new_cost = seq_cost(new_seq)
                    if best_inc_cost < float('inf') and new_cost >= best_inc_cost:
                        continue
                    new_rem = rem - {t}
                    sig = (new_rem, tuple(new_seq[-suffix_k:]))
                    prev = prefix_dom.get(sig)
                    if (prev is not None) and (new_cost >= prev):
                        continue

                    # Greedy probe
                    if taken < per_parent_greedy_K:
                        full, fc = greedy_complete(new_seq, new_rem)
                        if fc < best_inc_cost:
                            best_inc_cost = fc
                            best_inc_seq = full
                        # prune non-promising child
                        if fc >= best_inc_cost:
                            taken += 1
                            continue
                        taken += 1

                    prefix_dom[sig] = new_cost
                    key = (tuple(new_seq), new_rem)
                    if key in local_seen:
                        continue
                    local_seen.add(key)
                    next_beam.append((new_cost, new_seq, new_rem))

            if not next_beam:
                break
            next_beam.sort(key=lambda x: x[0])
            beam = next_beam[:beam_width]
            depth += 1

        # Finalize best
        if not beam:
            if best_inc_seq is not None:
                return best_inc_seq, best_inc_cost
            t = rng.randint(0, n - 1)
            full, fc = greedy_complete([t])
            return full, fc

        beam.sort(key=lambda x: x[0])
        _, bseq, brem = beam[0]
        full, fc = greedy_complete(bseq, brem)
        if best_inc_seq is not None and best_inc_cost < fc:
            return best_inc_seq, best_inc_cost
        return full, fc

    # ----------------- Greedy builder with unified dominance -----------------
    def greedy_build(incumbent_cost=float('inf')):
        # Seed from tournament or best singleton
        if n == 0:
            return [], 0
        pool = set(tournament_order[:min(10, n)])
        pool.update(sorted(range(n), key=lambda t: c1[t])[:min(10, n)])
        t0 = rng.choice(list(pool)) if pool else rng.randint(0, n - 1)

        seq = [t0]
        remaining = set(range(n))
        remaining.remove(t0)

        step = 0
        while remaining:
            base_cost = seq_cost(seq)
            if incumbent_cost < float('inf') and base_cost >= incumbent_cost:
                # Early abort: pessimistic branch
                return None, None

            # Update prefix dominance
            sig_cur = (frozenset(remaining), tuple(seq[-3:]) if len(seq) >= 3 else tuple(seq))
            prev_cur = prefix_dom.get(sig_cur)
            if prev_cur is not None and base_cost >= prev_cur:
                return None, None
            if prev_cur is None or base_cost < prev_cur:
                prefix_dom[sig_cur] = base_cost

            # Periodically greedy-complete to tighten incumbent
            if step % 9 == 0:
                full, fc = greedy_complete(seq, remaining)
                if fc < incumbent_cost:
                    incumbent_cost = fc

            # Candidate pool with tournament guidance
            R = len(remaining)
            pool_size = 12 if n < 90 else 16
            k_pool = min(pool_size, R)
            cand_pool = preselect_by_tournament(seq, list(remaining), k_pool, recent_k=recent_k_for_depth(len(seq))) or list(remaining)

            imm = [(t, seq_cost(seq + [t])) for t in cand_pool]
            imm.sort(key=lambda x: x[1])
            if not imm:
                t = rng.choice(list(remaining))
                seq.append(t)
                remaining.remove(t)
                step += 1
                continue

            # Adaptive lookahead + anti-buddy
            L = min(4 if step < 10 else 3, len(imm))
            chosen_t = None
            best_metric = float('inf')
            last = seq[-1] if seq else None
            best_immediate = imm[0][1]
            for t, immediate_c in imm[:L]:
                if last is not None and W[last][t] > 0 and W[last][t] >= qcut[last] and immediate_c > best_immediate * 1.01:
                    continue
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
                    metric = min(seq_cost(seq + [t, u]) for u in la_pool)
                if metric < best_metric:
                    best_metric = metric
                    chosen_t = t

            if chosen_t is None:
                chosen_t = imm[0][0]

            # Child dominance update
            child_cost = seq_cost(seq + [chosen_t])
            child_rem = frozenset(remaining - {chosen_t})
            sig_child = (child_rem, tuple((seq + [chosen_t])[-3:]) if len(seq) + 1 >= 3 else tuple(seq + [chosen_t]))
            prev_child = prefix_dom.get(sig_child)
            if prev_child is None or child_cost < prev_child:
                prefix_dom[sig_child] = child_cost

            seq.append(chosen_t)
            remaining.remove(chosen_t)
            step += 1

        return seq, seq_cost(seq)

    # ----------------- Local search (LNS and neighborhoods) -----------------
    def prefix_marginals(seq):
        prefix_costs = [0] * len(seq)
        c = 0
        for i in range(len(seq)):
            c = seq_cost(seq[: i + 1])
            prefix_costs[i] = c
        marg = [prefix_costs[0]] + [prefix_costs[i] - prefix_costs[i - 1] for i in range(1, len(seq))]
        return prefix_costs, marg

    def worst_violation_boundaries(seq, topm=3):
        pairs = []
        for i in range(len(seq) - 1):
            a, b = seq[i], seq[i + 1]
            pen = M[a][b] - M[b][a]
            if pen > 0:
                pairs.append((pen, i))
        pairs.sort(reverse=True)
        return [i for _, i in pairs[:topm]]

    def lns_improve(seq, base_cost, heavy=True, budget_factor=1):
        best_seq = seq[:]
        best_cost = base_cost
        n_local = len(best_seq)
        if n_local <= 1:
            return best_seq, best_cost

        # Anchored window permutations near violations and marginal-hot regions
        iters = max(4, min(10, 2 + int(budget_factor) + (n_local // 45)))
        max_k = 7 if n_local >= 40 else 6

        for it in range(iters):
            # Cheap cleanup
            tb = tournament_bubble_pass(best_seq, passes=2)
            cb = seq_cost(tb)
            if cb < best_cost:
                best_seq, best_cost = tb, cb

            _, marg = prefix_marginals(best_seq)

            # Choose window size
            k = max_k if (heavy and it < 2) else (rng.randint(4, max_k) if n_local > 6 else n_local)
            if n_local <= k:
                break

            sums = []
            window_sum = sum(marg[0:k])
            sums.append((0, window_sum))
            for s_ in range(1, n_local - k + 1):
                window_sum += marg[s_ + k - 1] - marg[s_ - 1]
                sums.append((s_, window_sum))
            sums.sort(key=lambda x: -x[1])

            viol_starts = worst_violation_boundaries(best_seq, topm=3)
            candidate_starts = []
            for v in viol_starts:
                candidate_starts.append(max(0, min(v - (k // 2), n_local - k)))
            candidate_starts.extend([start for start, _ in sums[:2]])

            # Deduplicate preserving order
            seen_starts = set()
            ordered_starts = []
            for st in candidate_starts:
                if st not in seen_starts:
                    seen_starts.add(st)
                    ordered_starts.append(st)

            # Try anchored windows, accept first improving
            perm_budget = 720 if k <= 6 else 2000
            for start in ordered_starts:
                block = best_seq[start:start + k]
                base = best_seq[:start] + best_seq[start + k:]
                factorial = 1
                for i_ in range(2, k + 1):
                    factorial *= i_
                perm_best_seq = None
                perm_best_cost = best_cost
                if factorial <= perm_budget:
                    for p in itertools.permutations(block):
                        cand = base[:start] + list(p) + base[start:]
                        c = seq_cost(cand)
                        if c < perm_best_cost:
                            perm_best_cost = c
                            perm_best_seq = cand
                else:
                    tried = set()
                    attempts = 0
                    # Score a few permutations by ΔW surrogate for warm-start; then sample
                    while attempts < perm_budget:
                        p = tuple(rng.sample(block, k))
                        if p in tried:
                            continue
                        tried.add(p)
                        cand = base[:start] + list(p) + base[start:]
                        c = seq_cost(cand)
                        if c < perm_best_cost:
                            perm_best_cost = c
                            perm_best_seq = cand
                        attempts += 1

                if perm_best_seq is not None and perm_best_cost < best_cost:
                    best_seq, best_cost = perm_best_seq, perm_best_cost
                    # lock in adjacency afterwards
                    best_seq = tournament_bubble_pass(best_seq, passes=1)
                    best_cost = seq_cost(best_seq)
                    break

            # Block-swap between two disjoint short windows around worst adjacencies (ΔW gated)
            if heavy and n_local >= 12:
                viols = worst_violation_boundaries(best_seq, topm=2)
                if viols:
                    k1 = 3 if n_local < 40 else 4
                    k2 = k1
                    s1 = max(0, min(viols[0] - k1 // 2, n_local - k1))
                    s2_candidates = [max(0, min((viols[-1] if len(viols) > 1 else viols[0]) + k1, n_local - k2))]
                    for _ in range(2):
                        s2_candidates.append(rng.randint(0, n_local - k2))
                    # Evaluate only the top by ΔW, accept first improvement
                    eval_list = []
                    for s2 in s2_candidates:
                        if abs(s2 - s1) < max(k1, k2):
                            continue
                        # ΔW surrogate: boundary penalties of swapping blocks
                        aL = best_seq[s1 - 1] if s1 > 0 else None
                        aR = best_seq[s1 + k1] if s1 + k1 < n_local else None
                        bL = best_seq[s2 - 1] if s2 > 0 else None
                        bR = best_seq[s2 + k2] if s2 + k2 < n_local else None
                        score = 0
                        if aL is not None:
                            score += max(0, W[aL][best_seq[s1]])
                        if aR is not None:
                            score += max(0, W[best_seq[s1 + k1 - 1]][aR])
                        if bL is not None:
                            score += max(0, W[bL][best_seq[s2]])
                        if bR is not None:
                            score += max(0, W[best_seq[s2 + k2 - 1]][bR])
                        eval_list.append((score, s2))
                    eval_list.sort(key=lambda z: z[0])
                    for _, s2 in eval_list[:3]:
                        block1 = best_seq[s1:s1 + k1]
                        block2 = best_seq[s2:s2 + k2]
                        if s1 < s2:
                            cand = best_seq[:s1] + block2 + best_seq[s1 + k1:s2] + block1 + best_seq[s2 + k2:]
                        else:
                            cand = best_seq[:s2] + block1 + best_seq[s2 + k2:s1] + block2 + best_seq[s1 + k1:]
                        c = seq_cost(cand)
                        if c < best_cost:
                            best_seq, best_cost = cand, c
                            break

            # Block reinsert guided by ΔW local context; accept first improving
            if n_local >= 10:
                kbr = 4 if n_local < 50 else 5
                # pick hottest block by marginal sum
                sums = []
                window_sum = sum(marg[0:kbr])
                sums.append((0, window_sum))
                for s_ in range(1, n_local - kbr + 1):
                    window_sum += marg[s_ + kbr - 1] - marg[s_ - 1]
                    sums.append((s_, window_sum))
                sums.sort(key=lambda x: -x[1])
                start = sums[0][0]
                block = best_seq[start:start + kbr]
                base = best_seq[:start] + best_seq[start + kbr:]
                positions = list(range(len(base) + 1))
                sur_scores = []
                for pos in positions:
                    left = base[pos - 1] if pos > 0 else None
                    right = base[pos] if pos < len(base) else None
                    score = 0
                    if left is not None:
                        score += max(0, W[left][block[0]])
                    if right is not None:
                        score += max(0, W[block[-1]][right])
                    sur_scores.append((score, pos))
                sur_scores.sort(key=lambda x: x[0])
                eval_positions = [p for _, p in sur_scores[:min(6, len(sur_scores))]]
                tried = set(eval_positions)
                while len(eval_positions) < min(8, len(positions)):
                    j = rng.randint(0, len(base))
                    if j not in tried:
                        tried.add(j)
                        eval_positions.append(j)
                for pos in eval_positions:
                    cand = base[:pos] + block + base[pos:]
                    c = seq_cost(cand)
                    if c < best_cost:
                        best_seq, best_cost = cand, c
                        break

            # "Bridge" move: take block around b and reinsert before a
            if heavy and n_local >= 9:
                viols_all = []
                for i in range(n_local - 1):
                    a, b = best_seq[i], best_seq[i + 1]
                    pen = M[a][b] - M[b][a]
                    viols_all.append((pen, i))
                viols_all.sort(reverse=True)
                if viols_all and viols_all[0][0] > 0:
                    i = viols_all[0][1]
                    a, b = best_seq[i], best_seq[i + 1]
                    # pick a block around b
                    kblk = 3 if n_local < 40 else 4
                    start_b = max(0, min(i + 1 - kblk // 2, n_local - kblk))
                    blk = best_seq[start_b:start_b + kblk]
                    base = best_seq[:start_b] + best_seq[start_b + kblk:]
                    # Try reinserting just before 'a' position with optional reverse
                    pos_before_a = max(0, min(i, len(base)))
                    candidates = []
                    candidates.append((blk, pos_before_a))
                    candidates.append((list(reversed(blk)), pos_before_a))
                    # Try a few neighboring positions
                    for delta in (-1, 1):
                        p = max(0, min(pos_before_a + delta, len(base)))
                        candidates.append((blk, p))
                    # Evaluate small set
                    for bblk, pos in candidates[:4]:
                        cand = base[:pos] + bblk + base[pos:]
                        c = seq_cost(cand)
                        if c < best_cost:
                            best_seq, best_cost = cand, c
                            # lock adjacencies
                            best_seq = tournament_bubble_pass(best_seq, passes=1)
                            best_cost = seq_cost(best_seq)
                            break

        # Final local tweaks
        # Small adjacent passes
        improved = True
        while improved:
            improved = False
            for i in range(len(best_seq) - 1):
                cand = best_seq[:]
                cand[i], cand[i + 1] = cand[i + 1], cand[i]
                c = seq_cost(cand)
                if c < best_cost:
                    best_seq, best_cost = cand, c
                    improved = True

        # Sampled 2-opt and random insertions
        two_opt_trials = min(60, n_local)
        for _ in range(two_opt_trials):
            i, j = rng.sample(range(n_local), 2)
            if abs(i - j) <= 1:
                continue
            cand = best_seq[:]
            cand[i], cand[j] = cand[j], cand[i]
            c = seq_cost(cand)
            if c < best_cost:
                best_seq, best_cost = cand, c

        move_budget = min(60, n_local)
        for _ in range(move_budget):
            i, j = rng.sample(range(n_local), 2)
            if i == j:
                continue
            cand = best_seq[:]
            item = cand.pop(i)
            cand.insert(j, item)
            c = seq_cost(cand)
            if c < best_cost:
                best_seq, best_cost = cand, c

        return best_seq, best_cost

    # ----------------- Portfolio orchestration -----------------
    incumbent_cost = float('inf')
    incumbent_seq = None

    seeds = []

    # 1) Deterministic portfolio beam configurations (shared caches and dominance)
    portfolio = [
        # (beam_width, cand_per_expand, lookahead_top, next_k)
        (max(16, int(num_seqs)), 12, 4, 6),
        (max(12, int(num_seqs) - 2), 12, 3, 5),
        (max(10, int(num_seqs) - 4), 10, 3, 4),
    ]
    for bw, cpe, la, nk in portfolio:
        seq_b, cost_b = beam_seed(bw, cpe, la, nk, incumbent_cost)
        seeds.append((cost_b, seq_b))
        if cost_b < incumbent_cost:
            incumbent_cost, incumbent_seq = cost_b, seq_b

    # 2) Tournament-guided greedy seeds (2 restarts)
    greedy_restarts = 2
    for _ in range(greedy_restarts):
        seq_g, cost_g = greedy_build(incumbent_cost)
        if seq_g is not None:
            seeds.append((cost_g, seq_g))
            if cost_g < incumbent_cost:
                incumbent_cost, incumbent_seq = cost_g, seq_g

    # Select top-k seeds for LNS
    seeds.sort(key=lambda x: x[0])
    if not seeds:
        # fallback
        default_seq = list(range(n))
        rng.shuffle(default_seq)
        return seq_cost(default_seq), default_seq

    topk = min(3, len(seeds))
    final_best_cost = float('inf')
    final_best_seq = None
    for idx, (sc, ss) in enumerate(seeds[:topk]):
        heavy = idx < 2  # apply heavier LNS on top-2 seeds
        seq_i, cost_i = lns_improve(ss, sc, heavy=heavy, budget_factor=max(1, int(num_seqs)))
        # quick cleanup
        seq_i = tournament_bubble_pass(seq_i, passes=1)
        cost_i = seq_cost(seq_i)
        if cost_i < final_best_cost:
            final_best_cost, final_best_seq = cost_i, seq_i

    # Consider remaining seeds with lighter local tweaks
    for sc, ss in seeds[topk:]:
        seq_i, cost_i = lns_improve(ss, sc, heavy=False, budget_factor=max(1, int(num_seqs // 2)))
        if cost_i < final_best_cost:
            final_best_cost, final_best_seq = cost_i, seq_i

    return final_best_cost, final_best_seq


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