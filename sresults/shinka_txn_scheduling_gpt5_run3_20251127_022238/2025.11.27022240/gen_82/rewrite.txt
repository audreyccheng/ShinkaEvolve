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
    Portfolio: beam-seeded and greedy builders with unified pruning, then LNS.
    Deterministic across runs for a given workload size.

    Args:
        workload: Workload object containing transaction data
        num_seqs: Controls exploration budget (beam width, local budgets)

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    n = workload.num_txns

    # Deterministic RNGs for 3 restarts
    def make_rng(seed_offset):
        return random.Random(1729 + n * 31 + seed_offset)

    # Global memoization for sequence costs
    cost_cache = {}

    def seq_cost(seq):
        key = tuple(seq)
        if key in cost_cache:
            return cost_cache[key]
        c = workload.get_opt_seq_cost(seq)
        cost_cache[key] = c
        return c

    # Precompute singleton and pairwise costs to expose conflicts
    c1 = [seq_cost([i]) for i in range(n)]
    M = [[0] * n for _ in range(n)]
    for i in range(n):
        Mi = M[i]
        for j in range(n):
            Mi[j] = c1[i] if i == j else seq_cost([i, j])

    # Margin matrix: W[i][j] > 0 means i before j is worse than j before i
    W = [[0] * n for _ in range(n)]
    for i in range(n):
        Wi = W[i]
        Mi = M[i]
        for j in range(n):
            Wi[j] = 0 if i == j else (Mi[j] - M[j][i])

    # Tournament score: lower is better (place earlier)
    s = [sum(W[i][j] for j in range(n) if j != i) for i in range(n)]
    tournament_order = list(range(n))
    tournament_order.sort(key=lambda x: (s[x], x))

    # Buddy list by best pairwise followers
    buddy_k = 8 if n >= 90 else 6
    buddies = []
    for t in range(n):
        order = sorted((u for u in range(n) if u != t), key=lambda u: M[t][u])
        buddies.append(order[:buddy_k])

    # Anti-buddy quantile per txn (75th percentile of W row)
    qcut = [0] * n
    for i in range(n):
        vals = [W[i][j] for j in range(n) if j != i]
        if vals:
            vals.sort()
            idx = int(0.75 * (len(vals) - 1))
            qcut[i] = vals[max(0, min(idx, len(vals) - 1))]
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

    def recent_k_for_depth(depth):
        frac = depth / max(1, n - 1)
        return 5 if frac < 0.33 else (4 if frac < 0.66 else 3)

    def la_params_for_depth(depth):
        frac = depth / max(1, n - 1)
        if frac < 0.33:
            return 4, 6  # lookahead_top, next_k
        elif frac < 0.66:
            return 3, 5
        else:
            return 2, 4

    def suffix_len_for_depth(depth):
        frac = depth / max(1, n - 1)
        return 3 if frac < 0.7 else 4

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

    def greedy_complete_from(prefix, rem_set=None):
        # Greedy completion: append txn minimizing current seq completion cost
        cur = prefix[:]
        if rem_set is None:
            rem = [t for t in range(n) if t not in cur]
        else:
            rem = list(rem_set)
        while rem:
            t = min(rem, key=lambda u: seq_cost(cur + [u]))
            cur.append(t)
            rem.remove(t)
        return cur, seq_cost(cur)

    # Shared prefix dominance: (frozenset(remaining), suffix) -> best cost
    prefix_dom = {}

    # Global incumbent across portfolio runs
    INC_SEQ = None
    INC_COST = float('inf')

    # Beam-seeded constructor
    def beam_seed(rng, beam_width, cand_per_expand, depth_frac_limit):
        nonlocal INC_SEQ, INC_COST
        depth_limit = max(3, int(depth_frac_limit * n))
        # Start seeds: tournament-best, a good singleton, then random fill
        starts = [tournament_order[0]]
        topk = min(10, n)
        singles_sorted = sorted(range(n), key=lambda t: c1[t])[:topk]
        starts.append(rng.choice(singles_sorted))
        remcands = [t for t in range(n) if t not in starts]
        rng.shuffle(remcands)
        starts.extend(remcands[:max(0, beam_width - len(starts))])

        beam = []
        seen = set()
        for t in starts:
            seq = [t]
            rem = frozenset(set(range(n)) - {t})
            c = seq_cost(seq)
            sig = (rem, tuple(seq[-1:]))
            prev = prefix_dom.get(sig)
            if prev is not None and c >= prev:
                continue
            prefix_dom[sig] = c
            key = (tuple(seq), rem)
            if key in seen:
                continue
            seen.add(key)
            beam.append((c, seq, rem))

        depth = 1
        while depth < min(depth_limit, n) and beam:
            recent_k = recent_k_for_depth(depth)
            lookahead_top, next_k = la_params_for_depth(depth)
            suffix_k = suffix_len_for_depth(depth)
            K_children_probe = 2 if depth / n < 0.7 else 1

            next_beam = []
            expanded = 0
            for cost, seq, rem in sorted(beam, key=lambda x: x[0]):
                if INC_COST < float('inf') and cost >= INC_COST:
                    continue
                rem_list = list(rem)
                if not rem_list:
                    next_beam.append((cost, seq, rem))
                    continue

                # Tournament-guided preselection
                pre = preselect_by_tournament(seq, rem_list, min(cand_per_expand * 2, len(rem_list)), recent_k=recent_k)
                if not pre:
                    pre = rem_list

                # Immediate costs
                imm = []
                best_immediate = float('inf')
                for t in pre:
                    c_im = seq_cost(seq + [t])
                    imm.append((t, c_im))
                    if c_im < best_immediate:
                        best_immediate = c_im
                imm.sort(key=lambda z: z[1])

                # Score candidates with one-step lookahead, anti-buddy guarded
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

                # Add diversity: a few immediate best
                diversity = min(max(2, cand_per_expand // 3), len(imm))
                for t, imc in imm[:diversity]:
                    if last is not None and W[last][t] > 0 and W[last][t] >= qcut[last] and imc > best_immediate * 1.01:
                        continue
                    scored.append((t, imc))

                # Unique and expand best-k
                uniq = {}
                for t, m in scored:
                    if (t not in uniq) or (m < uniq[t]):
                        uniq[t] = m
                items = sorted(uniq.items(), key=lambda z: z[1])
                take = min(cand_per_expand, len(items))
                probe_taken = 0
                for t, _ in items[:take]:
                    new_seq = seq + [t]
                    new_rem = rem - {t}
                    new_cost = seq_cost(new_seq)
                    if INC_COST < float('inf') and new_cost >= INC_COST:
                        continue
                    sig = (new_rem, tuple(new_seq[-suffix_k:]) if len(new_seq) >= suffix_k else tuple(new_seq))
                    prev = prefix_dom.get(sig)
                    if (prev is not None) and (new_cost >= prev):
                        continue

                    # Incumbent promotion at expansion: greedy-complete top children
                    if probe_taken < K_children_probe:
                        full, fc = greedy_complete_from(new_seq, new_rem)
                        if fc < INC_COST:
                            INC_COST = fc
                            INC_SEQ = full
                        # Prune child if completed cost not better than incumbent
                        if fc >= INC_COST:
                            probe_taken += 1
                            continue
                        probe_taken += 1

                    prefix_dom[sig] = new_cost
                    next_beam.append((new_cost, new_seq, new_rem))
                    expanded += 1

            if not next_beam:
                break
            next_beam.sort(key=lambda x: x[0])
            beam = next_beam[:beam_width]
            depth += 1

            # Occasionally tighten incumbent by completing best beam prefix
            if beam:
                bc, bseq, brem = beam[0]
                if bc < INC_COST:
                    full, fc = greedy_complete_from(bseq, brem)
                    if fc < INC_COST:
                        INC_COST = fc
                        INC_SEQ = full

        if not beam:
            # fallback: complete from a good singleton
            t0 = tournament_order[0]
            full, fc = greedy_complete_from([t0])
            return full, fc
        beam.sort(key=lambda x: x[0])
        _, bseq, brem = beam[0]
        full, fc = greedy_complete_from(bseq, brem)
        if INC_SEQ is not None and INC_COST < fc:
            return INC_SEQ[:], INC_COST
        return full, fc

    # Greedy builder with adaptive lookahead and unified pruning
    def greedy_build(rng):
        nonlocal INC_SEQ, INC_COST
        # Choose start among tournament-best and good singletons
        pool = set()
        pool.add(tournament_order[0])
        topk = min(10, n)
        pool.update(sorted(range(n), key=lambda t: c1[t])[:topk])
        t0 = rng.choice(list(pool))
        seq = [t0]
        remaining = set(range(n)) - {t0}

        step = 0
        while remaining:
            base_cost = seq_cost(seq)
            if INC_COST < float('inf') and base_cost >= INC_COST:
                # Early terminate via greedy completion to tighten bound
                full, fc = greedy_complete_from(seq, remaining)
                if fc < INC_COST:
                    INC_COST = fc
                    INC_SEQ = full
                return None  # abort this greedy build

            # Prefix dominance check on current prefix
            sig_cur = (frozenset(remaining), tuple(seq[-3:]) if len(seq) >= 3 else tuple(seq))
            prev_cur = prefix_dom.get(sig_cur)
            if prev_cur is not None and base_cost >= prev_cur:
                return None
            if prev_cur is None or base_cost < prev_cur:
                prefix_dom[sig_cur] = base_cost

            recent_k = recent_k_for_depth(step)
            lookahead_top, next_k = la_params_for_depth(step)

            R = len(remaining)
            pool_size = 14 if n >= 90 else 12
            k_pool = min(pool_size * 2, R)
            cand_pool = preselect_by_tournament(seq, list(remaining), k_pool, recent_k=recent_k) or list(remaining)

            # Immediate costs
            imm = [(t, seq_cost(seq + [t])) for t in cand_pool]
            imm.sort(key=lambda x: x[1])
            if not imm:
                # degenerate, pick any
                t = remaining.pop()
                seq.append(t)
                step += 1
                continue

            best_t = imm[0][0]
            best_metric = imm[0][1]
            last = seq[-1] if seq else None
            L = min(lookahead_top, len(imm))
            for t, imc in imm[:L]:
                if last is not None and W[last][t] > 0 and W[last][t] >= qcut[last] and imc > best_metric * 1.01:
                    continue
                nexts = [u for u in remaining if u != t]
                if not nexts:
                    metric = imc
                else:
                    buddy_pref = [u for u in buddies[t] if u in nexts][:next_k]
                    if len(buddy_pref) < next_k:
                        extra = preselect_by_tournament(seq + [t], [u for u in nexts if u not in buddy_pref], next_k - len(buddy_pref), recent_k=3)
                        buddy_pref += extra
                    pool_la = buddy_pref or nexts[:min(next_k, len(nexts))]
                    metric = min(seq_cost(seq + [t, u]) for u in pool_la)
                if metric < best_metric:
                    best_metric = metric
                    best_t = t

            # Update dominance for child
            child = seq + [best_t]
            child_rem = remaining - {best_t}
            child_cost = seq_cost(child)
            sig_child = (frozenset(child_rem), tuple(child[-3:]) if len(child) >= 3 else tuple(child))
            prev_child = prefix_dom.get(sig_child)
            if prev_child is None or child_cost < prev_child:
                prefix_dom[sig_child] = child_cost

            seq.append(best_t)
            remaining.remove(best_t)
            step += 1

            # Periodically promote incumbent via greedy completion
            if step % 8 == 0:
                full, fc = greedy_complete_from(seq, remaining)
                if fc < INC_COST:
                    INC_COST = fc
                    INC_SEQ = full
                if fc >= INC_COST:
                    return None

        return seq

    # LNS improvement suite
    def lns_improve(rng, seq, base_cost, rounds=6):
        best_seq = seq[:]
        best_cost = base_cost
        n_local = len(best_seq)

        def prefix_marginals(sq):
            prefix_costs = [0] * len(sq)
            c = 0
            for i in range(len(sq)):
                c = seq_cost(sq[: i + 1])
                prefix_costs[i] = c
            marg = [prefix_costs[0]] + [prefix_costs[i] - prefix_costs[i - 1] for i in range(1, len(sq))]
            return prefix_costs, marg

        def worst_violation_indices(sq, topm=3):
            pairs = []
            for i in range(len(sq) - 1):
                a, b = sq[i], sq[i + 1]
                pen = M[a][b] - M[b][a]
                if pen > 0:
                    pairs.append((pen, i))
            pairs.sort(reverse=True)
            return [i for _, i in pairs[:topm]]

        eval_cap_per_round = 700  # cap to avoid blowups
        for rd in range(rounds):
            evals = 0

            # Cheap cleanup
            tb = tournament_bubble_pass(best_seq, passes=2)
            ctb = seq_cost(tb)
            evals += 1
            if ctb < best_cost:
                best_seq, best_cost = tb, ctb

            # Anchored windows around worst violations
            viols = worst_violation_indices(best_seq, topm=3)
            if viols:
                k = 7 if n_local >= 50 else 6
                for v in viols:
                    if evals > eval_cap_per_round:
                        break
                    start = max(0, min(v - k // 2, n_local - k))
                    if start < 0 or start + k > n_local:
                        continue
                    block = best_seq[start:start + k]
                    base = best_seq[:start] + best_seq[start + k:]
                    # Two anchored variants: fix left boundary, permute interior; fix right boundary
                    interior = block[1:-1]
                    # Sample permutations with budget
                    perm_budget = 2000 if k == 7 else min(720, len(list(itertools.permutations(interior))) if len(interior) <= 6 else 720)
                    tried = set()
                    improved = False
                    attempts = 0
                    while attempts < perm_budget and evals <= eval_cap_per_round:
                        # alternate left-anchored and right-anchored generation
                        if attempts % 2 == 0 and interior:
                            p_int = tuple(rng.sample(interior, len(interior)))
                            if ('L', p_int) in tried:
                                attempts += 1
                                continue
                            tried.add(('L', p_int))
                            cand_block = [block[0]] + list(p_int) + [block[-1]]
                        else:
                            p_full = tuple(rng.sample(block, len(block)))
                            if ('F', p_full) in tried:
                                attempts += 1
                                continue
                            tried.add(('F', p_full))
                            cand_block = list(p_full)
                        cand = base[:start] + cand_block + base[start:]
                        c = seq_cost(cand)
                        evals += 1
                        if c < best_cost:
                            best_seq, best_cost = cand, c
                            improved = True
                            break
                        attempts += 1
                    if improved:
                        # lock-in small cleanup
                        best_seq = tournament_bubble_pass(best_seq, passes=1)
                        best_cost = seq_cost(best_seq)
                        evals += 1
                        break

            # Block-swap between two hot windows (ΔW gated)
            if evals <= eval_cap_per_round and n_local >= 12:
                kswap = 4 if n_local >= 40 else 3
                viols = worst_violation_indices(best_seq, topm=2)
                if viols:
                    s1 = max(0, min(viols[0] - kswap // 2, n_local - kswap))
                    s2_candidates = [max(0, min((viols[1] if len(viols) > 1 else viols[0]) + kswap, n_local - kswap))]
                    for _ in range(2):
                        s2_candidates.append(rng.randint(0, n_local - kswap))
                    for s2 in s2_candidates:
                        if abs(s2 - s1) < kswap:
                            continue
                        block1 = best_seq[s1:s1 + kswap]
                        block2 = best_seq[s2:s2 + kswap]
                        if s1 < s2:
                            cand = best_seq[:s1] + block2 + best_seq[s1 + kswap:s2] + block1 + best_seq[s2 + kswap:]
                        else:
                            cand = best_seq[:s2] + block1 + best_seq[s2 + kswap:s1] + block2 + best_seq[s1 + kswap:]
                        c = seq_cost(cand)
                        evals += 1
                        if c < best_cost:
                            best_seq, best_cost = cand, c
                            # quick bubble after swap
                            best_seq = tournament_bubble_pass(best_seq, passes=1)
                            best_cost = seq_cost(best_seq)
                            evals += 1
                            break

            # Block reinsert with ΔW surrogate ranking
            if evals <= eval_cap_per_round and n_local >= 10:
                _, marg = prefix_marginals(best_seq)
                kbr = 5 if n_local >= 50 else 4
                # Choose hottest window by marginal sum
                best_sum = None
                best_start = 0
                window_sum = sum(marg[0:kbr])
                best_sum = window_sum
                for s_ in range(1, n_local - kbr + 1):
                    window_sum += marg[s_ + kbr - 1] - marg[s_ - 1]
                    if window_sum > best_sum:
                        best_sum = window_sum
                        best_start = s_
                start = best_start
                block = best_seq[start:start + kbr]
                base = best_seq[:start] + best_seq[start + kbr:]
                # Surrogate for positions
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
                eval_positions = [p for _, p in sur_scores[:max(5, len(sur_scores) // 3)]]
                # add small random fraction
                rnd_extra = set()
                while len(rnd_extra) < min(2, max(0, len(positions) - len(eval_positions))):
                    rnd_extra.add(rng.randint(0, len(base)))
                eval_positions.extend(rnd_extra)
                improved = False
                for pos in eval_positions:
                    if evals > eval_cap_per_round:
                        break
                    cand = base[:pos] + block + base[pos:]
                    c = seq_cost(cand)
                    evals += 1
                    if c < best_cost:
                        best_seq, best_cost = cand, c
                        improved = True
                        break
                if improved:
                    best_seq = tournament_bubble_pass(best_seq, passes=1)
                    best_cost = seq_cost(best_seq)
                    evals += 1

            # 2.5-opt bridge: move block around follower before its worst predecessor
            if evals <= eval_cap_per_round and n_local >= 9:
                # Pick worst adjacency (a,b)
                worst = None
                worst_val = 0
                for i in range(n_local - 1):
                    a, b = best_seq[i], best_seq[i + 1]
                    pen = M[a][b] - M[b][a]
                    if pen > worst_val:
                        worst_val = pen
                        worst = (i, a, b)
                if worst and worst_val > 0:
                    i, a, b = worst
                    # block centered on b
                    block_size = 4 if n_local >= 40 else 3
                    j = i + 1
                    left = max(0, j - block_size // 2)
                    right = min(n_local, left + block_size)
                    left = max(0, right - block_size)
                    block = best_seq[left:right]
                    base = best_seq[:left] + best_seq[right:]
                    # candidate target position: just before a
                    target_pos = max(0, base.index(a))  # index of a
                    # Try both orientations
                    for bl in [block, list(reversed(block))]:
                        cand = base[:target_pos] + bl + base[target_pos:]
                        c = seq_cost(cand)
                        evals += 1
                        if c < best_cost:
                            best_seq, best_cost = cand, c
                            best_seq = tournament_bubble_pass(best_seq, passes=1)
                            best_cost = seq_cost(best_seq)
                            evals += 1
                            break

        return best_seq, best_cost

    # Portfolio orchestrator
    # Parameter sets for 3 deterministic restarts (beam_width, cand_per_expand, depth_limit_frac)
    portfolio_params = [
        (16, 12, 0.45),
        (12, 14, 0.40),
        (10, 10, 0.35),
    ]

    seeds = []
    # Build seeds via beam and greedy under deterministic RNGs; maintain shared prefix_dom/INC
    for r, params in enumerate(portfolio_params):
        rng = make_rng(r)
        bw, cpe, dlf = params
        # Beam seed
        try:
            bseq, bcost = beam_seed(rng, bw, cpe, dlf)
        except Exception:
            # fallback to simple greedy
            bseq, bcost = greedy_complete_from([tournament_order[0]])
        # local light cleanup
        bseq = tournament_bubble_pass(bseq, passes=2)
        bcost = seq_cost(bseq)
        seeds.append((bcost, bseq))

        # Greedy restart
        gseq = greedy_build(rng)
        if gseq is not None:
            gseq = tournament_bubble_pass(gseq, passes=2)
            gcost = seq_cost(gseq)
            seeds.append((gcost, gseq))

    # Pick best two seeds for full LNS; light cleanup on others
    seeds.sort(key=lambda x: x[0])
    best_cost, best_seq = seeds[0][0], seeds[0][1]
    finalists = seeds[:2]

    for cost0, seq0 in finalists:
        # Adjacent pass first
        seqA = seq0[:]
        cur_cost = seq_cost(seqA)
        improved = True
        while improved:
            improved = False
            for i in range(n - 1):
                if seqA[i] == seqA[i + 1]:
                    continue
                cand = seqA[:]
                cand[i], cand[i + 1] = cand[i + 1], cand[i]
                c = seq_cost(cand)
                if c < cur_cost:
                    seqA, cur_cost = cand, c
                    improved = True
        # LNS improvement
        seqB, costB = lns_improve(make_rng(hash(tuple(seqA)) % (10**6)), seqA, cur_cost, rounds=max(5, min(8, 2 + n // 30)))
        if costB < best_cost:
            best_cost, best_seq = costB, seqB

    # Finishing touches: sampled 2-opt and insertions with small budgets
    rng_finish = make_rng(999)
    two_opt_trials = min(60, n)
    for _ in range(two_opt_trials):
        i, j = rng_finish.sample(range(n), 2)
        if abs(i - j) <= 1:
            continue
        cand = best_seq[:]
        cand[i], cand[j] = cand[j], cand[i]
        c = seq_cost(cand)
        if c < best_cost:
            best_seq, best_cost = cand, c

    move_budget = min(60, n)
    for _ in range(move_budget):
        i, j = rng_finish.sample(range(n), 2)
        if i == j:
            continue
        cand = best_seq[:]
        item = cand.pop(i)
        cand.insert(j, item)
        c = seq_cost(cand)
        if c < best_cost:
            best_seq, best_cost = cand, c

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