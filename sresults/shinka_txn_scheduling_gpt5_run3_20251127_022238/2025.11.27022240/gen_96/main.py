# EVOLVE-BLOCK-START
"""Transaction scheduling algorithm for optimizing makespan across multiple workloads"""

import time
import random
import sys
import os
import itertools

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
    Enhanced multi-start greedy with incumbent pruning, adaptive lookahead,
    a beam-seeded prefix explorer, and boundary-focused LNS for improved makespan.

    Args:
        workload: Workload object containing transaction data
        num_seqs: Number of randomized restarts / beam width budget

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    n = workload.num_txns
    rng = random.Random(1729 + n)

    # Memoization for sequence costs to avoid recomputation
    cost_cache = {}

    def seq_cost(seq):
        key = tuple(seq)
        if key in cost_cache:
            return cost_cache[key]
        c = workload.get_opt_seq_cost(seq)
        cost_cache[key] = c
        return c

    # Precompute singleton and pairwise costs and tournament structure
    c1 = [seq_cost([i]) for i in range(n)]
    M = [[0] * n for _ in range(n)]
    for i in range(n):
        Mi = M[i]
        for j in range(n):
            Mi[j] = c1[i] if i == j else seq_cost([i, j])

    W = [[0] * n for _ in range(n)]
    for i in range(n):
        Wi = W[i]
        Mi = M[i]
        for j in range(n):
            Wi[j] = 0 if i == j else (Mi[j] - M[j][i])

    s = [0] * n
    for i in range(n):
        s[i] = sum(W[i][j] for j in range(n) if j != i)
    tournament_order = list(range(n))
    tournament_order.sort(key=lambda x: (s[x], x))

    # Anti-buddy quartile cutoff per txn (for filtering strongly disfavored followers)
    qcut = [0] * n
    for i in range(n):
        vals = [W[i][j] for j in range(n) if j != i]
        if vals:
            vals.sort()
            idx = int(0.75 * (len(vals) - 1))
            idx = max(0, min(idx, len(vals) - 1))
            qcut[i] = vals[idx]
        else:
            qcut[i] = 0

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

    # Buddy list per txn by small M[i][j]
    buddy_k = 8 if n >= 90 else 6
    buddies = []
    for t in range(n):
        order = sorted((u for u in range(n) if u != t), key=lambda u: M[t][u])
        buddies.append(order[:buddy_k])

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

    def recent_k_for_depth(d):
        frac = d / max(1, n - 1)
        return 5 if frac < 0.33 else (4 if frac < 0.66 else 3)

    # Cheap greedy completion used to tighten incumbent during construction
    def greedy_complete(seq, remaining):
        base = seq[:]
        rem = list(remaining)
        while rem:
            t = min(rem, key=lambda u: seq_cost(base + [u]))
            base.append(t)
            rem.remove(t)
        return base, seq_cost(base)

    # Greedy completion from a prefix (compute remaining internally)
    def greedy_complete_from(seq):
        rem = [t for t in range(n) if t not in seq]
        cur = seq[:]
        while rem:
            t = min(rem, key=lambda u: seq_cost(cur + [u]))
            cur.append(t)
            rem.remove(t)
        return cur, seq_cost(cur)

    # Prefix-dominance map shared across restarts: (frozenset(remaining), suffix<=3) -> best cost
    prefix_dom = {}

    # Two-phase beam seed with incumbent-pruned expansion and lookahead
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

        starts = []
        starts.append(tournament_order[0])
        topk = min(10, n)
        good_singletons = sorted(range(n), key=lambda t: c1[t])[:topk]
        starts.append(rng.choice(good_singletons))
        remcands = [t for t in range(n) if t not in starts]
        rng.shuffle(remcands)
        initial_width = max(beam_width_A, beam_width_B)
        starts.extend(remcands[:max(0, initial_width - len(starts))])

        beam = []
        used = set()
        for t in starts:
            seq = [t]
            rem = frozenset(set(range(n)) - {t})
            cost = seq_cost(seq)
            key = (tuple(seq), rem)
            if key in used:
                continue
            used.add(key)
            # shared prefix dominance on initial nodes
            sig0 = (rem, tuple(seq[-1:]))
            prev0 = prefix_dom.get(sig0)
            if (prev0 is None) or (cost < prev0):
                prefix_dom[sig0] = cost
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
            frac = d / max(1, n - 1)
            suffix_k = 3 if frac < 0.7 else 4
            greedy_probe_k = 2 if frac < 0.7 else 1

            # Tighten incumbent by greedy-completing top beam items
            if beam:
                beam_sorted = sorted(beam, key=lambda x: x[0])
                for bc, bseq, brem in beam_sorted[:2]:
                    if bc < incumbent_cost:
                        full, fc = greedy_complete(bseq, list(brem))
                        if fc < incumbent_cost:
                            incumbent_cost = fc
                            incumbent_seq = full

            next_beam = []
            seen = set()

            for cost, seq, rem in beam:
                if incumbent_cost < float('inf') and cost >= incumbent_cost:
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

                # Immediate costs for candidates
                imm = []
                best_immediate = float('inf')
                last = seq[-1] if seq else None
                for t in pre:
                    c_im = seq_cost(seq + [t])
                    if (not phaseA) and (incumbent_cost < float('inf')) and (c_im >= incumbent_cost):
                        continue
                    # anti-buddy gating against last placed
                    if last is not None and W[last][t] > 0 and W[last][t] >= qcut[last]:
                        # allow only if near-best immediate
                        best_immediate = min(best_immediate, c_im)
                        imm.append((t, c_im))
                    else:
                        best_immediate = min(best_immediate, c_im)
                        imm.append((t, c_im))
                if not imm:
                    continue
                imm.sort(key=lambda z: z[1])

                # Buddy/tournament lookahead for top-L with anti-buddy gating
                L = min(lookahead_top, len(imm))
                scored = []
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

                # Deduplicate candidates and expand best-k
                uniq = {}
                for t, m in scored:
                    if (t not in uniq) or (m < uniq[t]):
                        uniq[t] = m
                items = sorted(uniq.items(), key=lambda z: z[1])
                take = min(cand_per_expand, len(items))

                # Probe greedy completion on top-K children by lookahead and prune
                probe_children = [items[i][0] for i in range(min(greedy_probe_k, len(items)))]
                child_block = set(probe_children)

                for idx, (t, _) in enumerate(items[:take]):
                    new_seq = seq + [t]
                    new_cost = seq_cost(new_seq)
                    if (incumbent_cost < float('inf')) and (new_cost >= incumbent_cost):
                        continue
                    new_rem = rem - {t}
                    sig = (new_rem, tuple(new_seq[-suffix_k:]) if len(new_seq) >= suffix_k else tuple(new_seq))
                    prev = prefix_dom.get(sig)
                    if (prev is not None) and (new_cost >= prev):
                        continue

                    # Greedy promotion for top children
                    if t in child_block:
                        full, fc = greedy_complete(new_seq, list(new_rem))
                        if fc < incumbent_cost:
                            incumbent_cost = fc
                            incumbent_seq = full
                        # prune child if completion not strictly improving incumbent
                        if fc >= incumbent_cost:
                            continue

                    prefix_dom[sig] = new_cost
                    key2 = (tuple(new_seq), new_rem)
                    if key2 in seen:
                        continue
                    seen.add(key2)
                    next_beam.append((new_cost, new_seq, new_rem))

            if not next_beam:
                break
            next_beam.sort(key=lambda x: x[0])
            beam = next_beam[:beam_width]

        if not beam:
            # Fallback to random singleton seed completion
            t = rng.randint(0, n - 1)
            full, fc = greedy_complete_from([t])
            return full, fc

        beam.sort(key=lambda x: x[0])
        _, bseq, brem = beam[0]
        full, fc = greedy_complete(bseq, list(brem))
        if (incumbent_seq is not None) and (fc >= incumbent_cost):
            return incumbent_seq, incumbent_cost
        return full, fc

    # Local search helpers
    def adjacent_pass(seq, current_cost):
        improved = True
        n_local = len(seq)
        best_seq = seq[:]
        best_cost = current_cost
        while improved:
            improved = False
            for i in range(n_local - 1):
                cand = best_seq[:]
                cand[i], cand[i + 1] = cand[i + 1], cand[i]
                c = seq_cost(cand)
                if c < best_cost:
                    best_seq = cand
                    best_cost = c
                    improved = True
        return best_seq, best_cost

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
            pen = M[a][b] - M[b][a]  # positive when a before b is worse
            if pen > 0:
                pairs.append((pen, i))
        pairs.sort(reverse=True)
        return [i for _, i in pairs[:topm]]

    # Boundary-focused LNS with hot-window permutations and targeted relocates
    def lns_improve(seq, base_cost, budget_factor):
        best_seq = seq[:]
        best_cost = base_cost
        n_local = len(best_seq)

        iters = max(4, min(10, 2 + int(budget_factor) + n_local // 40))
        max_k = 7 if n_local >= 40 else 6

        for it in range(iters):
            # Tournament bubble cheap cleanup
            tb = tournament_bubble_pass(best_seq, passes=2)
            cb = seq_cost(tb)
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
                        c = seq_cost(cand_seq)
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
                        c = seq_cost(cand_seq)
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
                    c = seq_cost(cand)
                    if c < best_pos_cost:
                        best_pos_cost = c
                        best_pos_idx = j
                if best_pos_idx is not None and best_pos_cost < best_cost:
                    best_seq = base[:best_pos_idx] + [t] + base[best_pos_idx:]
                    best_cost = best_pos_cost

            # Anchored "bridge" move around worst violated adjacency using ΔW surrogate
            if n_local >= 8:
                viols_idx = worst_violation_boundaries(best_seq, topm=2)
                improved_bridge = False
                for v in viols_idx:
                    if v < 0 or v + 1 >= len(best_seq):
                        continue
                    a = best_seq[v]
                    b = best_seq[v + 1]
                    # Try short blocks anchored at b moved before a
                    for size in (3, 4, 5):
                        startR = v + 1
                        endR = min(len(best_seq), startR + size)
                        if endR - startR < 2:
                            continue
                        block = best_seq[startR:endR]
                        base = best_seq[:startR] + best_seq[endR:]
                        try:
                            a_idx = base.index(a)
                        except ValueError:
                            continue
                        # candidate positions just before a (a_idx) and slightly earlier
                        pos_cands = {max(0, a_idx - 2), max(0, a_idx - 1), a_idx}
                        # Rank positions by ΔW surrogate
                        scored = []
                        for pos in pos_cands:
                            left = base[pos - 1] if pos > 0 else None
                            right = base[pos] if pos < len(base) else None
                            score = 0
                            if left is not None:
                                score += max(0, W[left][block[0]])
                            if right is not None:
                                score += max(0, W[block[-1]][right])
                            scored.append((score, pos))
                        scored.sort(key=lambda x: x[0])
                        for _, pos in scored:
                            cand = base[:pos] + block + base[pos:]
                            c = seq_cost(cand)
                            if c < best_cost:
                                best_seq = cand
                                best_cost = c
                                n_local = len(best_seq)
                                improved_bridge = True
                                break
                        if improved_bridge:
                            break
                    if improved_bridge:
                        break
                if improved_bridge:
                    continue

            if not tried_any:
                continue

        return best_seq, best_cost

    # Use beam seed once to set a strong incumbent for pruning and seeding LNS
    seed_best_cost = float('inf')
    seed_best_seq = None
    try:
        bseq, bcost = beam_seed()
        # quick local cleanup on beam seed
        bseq_adj, bcost_adj = adjacent_pass(bseq, bcost)
        if bcost_adj < bcost:
            bseq, bcost = bseq_adj, bcost_adj
        seed_best_seq, seed_best_cost = bseq[:], bcost
    except Exception:
        pass

    # Greedy restarts (construction only) with tournament-guided pools and buddy lookahead
    restarts = max(1, int(num_seqs))
    for r in range(restarts):
        # Diversify starts: tournament-best for first, then mix good singletons and randoms
        if r == 0:
            t0 = tournament_order[0]
        else:
            pool = set(sorted(range(n), key=lambda t: c1[t])[:min(10, n)])
            pool.update(tournament_order[:min(10, n)])
            if len(pool) < 3:
                pool.update(range(n))
            t0 = rng.choice(list(pool))

        seq = [t0]
        remaining = [t for t in range(n) if t != t0]

        step = 0
        while remaining:
            base_cost = seq_cost(seq)
            # Incumbent-based pruning
            if seed_best_cost < float('inf') and base_cost >= seed_best_cost:
                seq = None
                break

            # Prefix-dominance pruning on current prefix
            sig_cur = (frozenset(remaining), tuple(seq[-3:]) if len(seq) >= 3 else tuple(seq))
            prev_cur = prefix_dom.get(sig_cur)
            if prev_cur is not None and base_cost >= prev_cur:
                seq = None
                break
            if prev_cur is None or base_cost < prev_cur:
                prefix_dom[sig_cur] = base_cost

            # Tournament-guided preselection pool
            R = len(remaining)
            pool_size = 16 if n >= 90 else 12
            k_pool = min(pool_size, R) if R > pool_size else R
            cand_pool = preselect_by_tournament(seq, list(remaining), k_pool, recent_k=recent_k_for_depth(len(seq)))
            if not cand_pool:
                cand_pool = list(remaining)

            # Evaluate immediate costs within pool
            imm = [(t, seq_cost(seq + [t])) for t in cand_pool]
            imm.sort(key=lambda x: x[1])

            # Adaptive lookahead with buddies + anti-buddy gating
            L = min(4 if step < 10 else 3, len(imm))
            chosen_t = imm[0][0]
            best_metric = imm[0][1]
            last = seq[-1] if seq else None
            best_immediate = imm[0][1]

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
                    metric = min(seq_cost(seq + [t, u]) for u in la_pool)

                # anti-buddy: if last->t is strongly bad and not near best, skip
                if last is not None and W[last][t] > 0 and W[last][t] >= qcut[last] and immediate_c > best_immediate * 1.01:
                    continue

                if metric < best_metric:
                    best_metric = metric
                    chosen_t = t

            # Update prefix-dominance with child
            new_cost = seq_cost(seq + [chosen_t])
            child_rem = set(remaining)
            child_rem.remove(chosen_t)
            sig_child = (frozenset(child_rem), tuple((seq + [chosen_t])[-3:]) if len(seq) + 1 >= 3 else tuple(seq + [chosen_t]))
            prev_child = prefix_dom.get(sig_child)
            if prev_child is None or new_cost < prev_child:
                prefix_dom[sig_child] = new_cost

            seq.append(chosen_t)
            remaining.remove(chosen_t)
            step += 1

        if seq is None:
            continue

        # Quick adjacent cleanup on each constructed seed
        scost = seq_cost(seq)
        seq, scost = adjacent_pass(seq, scost)
        if scost < seed_best_cost:
            seed_best_cost = scost
            seed_best_seq = seq[:]

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
        c = seq_cost(cand)
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
        c = seq_cost(cand)
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
    workload_size = 100

    # Workload 1: Complex mixed read/write transactions
    workload = Workload(WORKLOAD_1)
    # Slightly larger exploration for richer conflict structure
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