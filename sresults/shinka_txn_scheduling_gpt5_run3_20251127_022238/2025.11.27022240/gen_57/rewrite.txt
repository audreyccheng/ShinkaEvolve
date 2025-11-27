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
    Two-phase beam + adaptive greedy with global dominance pruning and strengthened LNS.

    Args:
        workload: Workload object containing transaction data
        num_seqs: Exploration budget; used to scale beam widths and attempts

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    n = workload.num_txns

    # Shared RNG base; each restart gets a deterministic shift
    base_seed = 1729 + n * 31

    # Global memo for partial sequence cost
    cost_cache = {}

    def eval_seq(seq):
        key = tuple(seq)
        if key in cost_cache:
            return cost_cache[key]
        c = workload.get_opt_seq_cost(seq)
        cost_cache[key] = c
        return c

    # Precompute singleton and pairwise costs (exposes conflict structure)
    c1 = [eval_seq([i]) for i in range(n)]
    M = [[0] * n for _ in range(n)]
    for i in range(n):
        Mi = M[i]
        for j in range(n):
            Mi[j] = c1[i] if i == j else eval_seq([i, j])

    # Preference margins (W[a][b] positive => a before b is worse than b before a)
    W = [[0] * n for _ in range(n)]
    for i in range(n):
        Wi = W[i]
        Mi = M[i]
        for j in range(n):
            Wi[j] = 0 if i == j else (Mi[j] - M[j][i])

    # Tournament order (lower aggregate margin score tends to be earlier)
    s = [sum(W[i][j] for j in range(n) if j != i) for i in range(n)]
    tournament_order = list(range(n))
    tournament_order.sort(key=lambda x: (s[x], x))

    # Buddy lists: top neighbors by small M[t][u]
    buddy_k = 8 if n >= 90 else 6
    buddies = []
    for t in range(n):
        order = sorted((u for u in range(n) if u != t), key=lambda u: M[t][u])
        buddies.append(order[:buddy_k])

    # Global prefix-dominance map across builders and restarts
    # Key: (frozenset(remaining), suffix<=3 tuple) -> best known prefix cost
    global_dom = {}

    # Global incumbent
    incumbent_cost = float('inf')
    incumbent_seq = None

    def recent_k_for_depth(d):
        frac = d / max(1, n - 1)
        return 5 if frac < 0.25 else (4 if frac < 0.7 else 3)

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

    def anti_buddy_deprioritize(last, candidates, best_metric, thresh_frac=0.25, slack=0.01):
        # Push strongly disfavored transactions to the back unless within slack of best_metric
        if last is None or not candidates:
            return candidates
        row = W[last][:]
        positives = sorted([row[t] for t in range(n) if t != last and row[t] > 0])
        if not positives:
            return candidates
        idx = max(0, int((1 - thresh_frac) * len(positives)) - 1)
        cutoff = positives[idx]
        safe, risky = [], []
        for t, m in candidates:
            if W[last][t] > cutoff and m > (1.0 + slack) * best_metric:
                risky.append((t, m))
            else:
                safe.append((t, m))
        return safe + risky

    # Greedy completion routine (tournament-guided pool + buddy lookahead)
    def greedy_complete(prefix):
        seq = prefix[:]
        remaining = [t for t in range(n) if t not in seq]
        while remaining:
            R = len(remaining)
            pool_k = min(14, R) if R > 28 else R
            pool = preselect_by_tournament(seq, remaining, pool_k, recent_k=recent_k_for_depth(len(seq)))
            if not pool:
                pool = remaining[:]
            imm = [(t, eval_seq(seq + [t])) for t in pool]
            imm.sort(key=lambda z: z[1])
            L = min(3, len(imm))
            best_t, best_metric = imm[0]
            last = seq[-1] if seq else None
            cand_la = []
            for t, imc in imm[:L]:
                nxt = [u for u in remaining if u != t]
                if not nxt:
                    la = imc
                else:
                    need = 6 if n > 60 else 7
                    bp = [u for u in buddies[t] if u in nxt][:need]
                    if len(bp) < need:
                        extra = preselect_by_tournament(seq + [t], [u for u in nxt if u not in bp], need - len(bp), recent_k=3)
                        pool2 = bp + extra
                    else:
                        pool2 = bp
                    if not pool2:
                        pool2 = nxt[:min(need, len(nxt))]
                    la = min(eval_seq(seq + [t, u]) for u in pool2)
                cand_la.append((t, min(imc, la)))
            # Anti-buddy filtering
            cand_la.sort(key=lambda z: z[1])
            cand_la = anti_buddy_deprioritize(last, cand_la, cand_la[0][1] if cand_la else float('inf'))
            choose = cand_la[0][0] if cand_la else imm[0][0]
            seq.append(choose)
            remaining.remove(choose)
        return seq, eval_seq(seq)

    # Beam search (two-phase) returning a strong seed
    def beam_seed(rng, param_override=None):
        nonlocal incumbent_cost, incumbent_seq

        # Defaults per recommendations
        beam_A = 16
        beam_B = 10
        cand_A = 12
        cand_B = 8
        lookahead_top_A = 4
        lookahead_top_B = 3
        next_k_A = 6
        next_k_B = 5
        depth_limit = max(3, int(0.4 * n))

        if param_override:
            beam_A = param_override.get("beam_A", beam_A)
            beam_B = param_override.get("beam_B", beam_B)
            cand_A = param_override.get("cand_A", cand_A)
            cand_B = param_override.get("cand_B", cand_B)
            lookahead_top_A = param_override.get("lookahead_top_A", lookahead_top_A)
            lookahead_top_B = param_override.get("lookahead_top_B", lookahead_top_B)
            next_k_A = param_override.get("next_k_A", next_k_A)
            next_k_B = param_override.get("next_k_B", next_k_B)

        # Diverse starts: tournament best + good singleton + randoms
        starts = []
        starts.append(tournament_order[0])
        good_singletons = sorted(range(n), key=lambda t: c1[t])[:min(10, n)]
        starts.append(rng.choice(good_singletons))
        remain = [t for t in range(n) if t not in starts]
        rng.shuffle(remain)
        initial_width = max(beam_A, beam_B)
        starts.extend(remain[:max(0, initial_width - len(starts))])

        # Initialize beam
        beam = []
        seen = set()
        for t in starts:
            seq = [t]
            rem = frozenset(set(range(n)) - {t})
            cost = eval_seq(seq)
            key = (tuple(seq), rem)
            if key in seen:
                continue
            seen.add(key)
            beam.append((cost, seq, rem))

        # Local dominance map reused across phases, but also consult global_dom
        loc_dom = {}

        depth = 1
        while depth < n and beam:
            phaseA = depth < depth_limit
            beam_width = beam_A if phaseA else beam_B
            cand_per_expand = cand_A if phaseA else cand_B
            lookahead_top = lookahead_top_A if phaseA else lookahead_top_B
            next_k = next_k_A if phaseA else next_k_B
            suffix_k = 3
            rk = recent_k_for_depth(depth)

            # Try to tighten incumbent by greedy completion of top prefixes
            beam_sorted = sorted(beam, key=lambda x: x[0])
            for bc, bseq, brem in beam_sorted[:2]:
                if bc < incumbent_cost:
                    full, fc = greedy_complete(bseq)
                    if fc < incumbent_cost:
                        incumbent_cost = fc
                        incumbent_seq = full

            next_beam = []
            local_seen = set()

            for base_cost, seq, rem in beam:
                if incumbent_cost < float('inf') and base_cost >= incumbent_cost:
                    continue
                rem_list = list(rem)
                if not rem_list:
                    next_beam.append((base_cost, seq, rem))
                    continue

                # Tournament-guided preselection
                if len(rem_list) > cand_per_expand * 2:
                    pre = preselect_by_tournament(seq, rem_list, cand_per_expand * 2, recent_k=rk)
                else:
                    pre = rem_list

                # Evaluate immediate costs
                imm = [(t, eval_seq(seq + [t])) for t in pre]
                imm.sort(key=lambda z: z[1])

                L = min(lookahead_top, len(imm))
                scored = []
                last = seq[-1] if seq else None
                for t, imc in imm[:L]:
                    nxts = [u for u in rem_list if u != t]
                    if not nxts:
                        la = imc
                    else:
                        bp = [u for u in buddies[t] if u in nxts][:next_k]
                        if len(bp) < next_k:
                            extra = preselect_by_tournament(seq + [t], [u for u in nxts if u not in bp], next_k - len(bp), recent_k=3)
                            pool = bp + extra
                        else:
                            pool = bp
                        if not pool:
                            pool = nxts[:min(next_k, len(nxts))]
                        la = min(eval_seq(seq + [t, u]) for u in pool)
                    scored.append((t, min(imc, la)))
                # Add diversity by immediate-best
                diversity = min(max(2, cand_per_expand // 3), len(imm))
                for t, imc in imm[:diversity]:
                    scored.append((t, imc))

                # Sort and anti-buddy filter
                scored.sort(key=lambda z: z[1])
                if scored:
                    scored = anti_buddy_deprioritize(last, scored, scored[0][1])

                # Expand top candidates with dominance and incumbent pruning
                best_child_for_parent = None
                expanded = 0
                for t, _metric in scored:
                    new_seq = seq + [t]
                    new_cost = eval_seq(new_seq)
                    if incumbent_cost < float('inf') and new_cost >= incumbent_cost:
                        continue
                    new_rem = rem - {t}
                    sig = (new_rem, tuple(new_seq[-suffix_k:]) if len(new_seq) >= suffix_k else tuple(new_seq))
                    prev = loc_dom.get(sig)
                    prev_g = global_dom.get(sig)
                    if (prev is not None and new_cost >= prev) or (prev_g is not None and new_cost >= prev_g):
                        continue
                    loc_dom[sig] = new_cost
                    global_dom[sig] = min(new_cost, global_dom.get(sig, float('inf')))
                    key = (tuple(new_seq), new_rem)
                    if key in local_seen:
                        continue
                    local_seen.add(key)

                    next_beam.append((new_cost, new_seq, new_rem))
                    expanded += 1
                    if best_child_for_parent is None or new_cost < best_child_for_parent[0]:
                        best_child_for_parent = (new_cost, new_seq, new_rem)
                    if expanded >= cand_per_expand:
                        break

                # Always retain the best child found (if any) for this parent
                if best_child_for_parent is not None:
                    next_beam.append(best_child_for_parent)

            if not next_beam:
                break

            # Sort next_beam and greedy-complete top few to update incumbent and prune
            next_beam.sort(key=lambda x: x[0])
            pruned_beam = []
            checked = 0
            for bc, bseq, brem in next_beam:
                if checked < 3:  # greedily complete top 2-3
                    full, fc = greedy_complete(bseq)
                    if fc < incumbent_cost:
                        incumbent_cost = fc
                        incumbent_seq = full
                    checked += 1
                # prune if prefix already worse than incumbent
                if incumbent_cost < float('inf') and bc >= incumbent_cost:
                    continue
                pruned_beam.append((bc, bseq, brem))

            beam = pruned_beam[: (beam_A if phaseA else beam_B)]
            depth += 1

        if not beam:
            # fallback: greedy from tournament best
            start = tournament_order[0]
            full, fc = greedy_complete([start])
            return full, fc

        beam.sort(key=lambda x: x[0])
        _, bseq, _ = beam[0]
        full, fc = greedy_complete(bseq)
        if incumbent_seq is not None and incumbent_cost < fc:
            return incumbent_seq, incumbent_cost
        return full, fc

    # Greedy builder that uses global dominance map and periodic greedy-complete
    def build_initial_sequence(rng, inc_ref, param_override=None):
        pool_size = 16 if n >= 90 else 12
        reevaluate_every = 8
        seq = []
        # Seed from tournament-best or good singleton
        k = min(10, n)
        candidates = set([tournament_order[0]])
        candidates.update(sorted(range(n), key=lambda t: c1[t])[:k])
        seq.append(rng.choice(list(candidates)))
        remaining = set(range(n))
        remaining.remove(seq[0])

        step = 0
        while remaining:
            base_cost = eval_seq(seq)
            if inc_ref[0] < float('inf') and base_cost >= inc_ref[0]:
                # early stop with greedy completion for pruning
                full, fc = greedy_complete(seq)
                if fc < inc_ref[0]:
                    inc_ref[0] = fc
                    inc_ref[1] = full
                # append immediate best to finish quickly
                rest = list(remaining)
                rest.sort(key=lambda t: eval_seq(seq + [t]))
                seq.extend(rest)
                remaining.clear()
                break

            # Global prefix dominance
            sig_cur = (frozenset(remaining), tuple(seq[-3:]) if len(seq) >= 3 else tuple(seq))
            prev = global_dom.get(sig_cur)
            if prev is not None and base_cost >= prev:
                # Abort this build; dominated
                break
            global_dom[sig_cur] = min(base_cost, prev if prev is not None else float('inf'))

            if step % reevaluate_every == 0:
                pool = preselect_by_tournament(seq, list(remaining), min(pool_size, len(remaining)), recent_k=recent_k_for_depth(len(seq)))
                if not pool:
                    pool = list(remaining)
            else:
                pool = list(remaining)

            # Evaluate pool with lookahead
            imm = [(t, eval_seq(seq + [t])) for t in pool]
            imm.sort(key=lambda z: z[1])
            L = min(4 if step < int(0.25 * n) else 3, len(imm))
            last = seq[-1] if seq else None
            cand_la = []
            for t, imc in imm[:L]:
                nxt = [u for u in remaining if u != t]
                if not nxt:
                    la = imc
                else:
                    need = 6 if n > 60 else 7
                    bp = [u for u in buddies[t] if u in nxt][:need]
                    if len(bp) < need:
                        extra = preselect_by_tournament(seq + [t], [u for u in nxt if u not in bp], need - len(bp), recent_k=3)
                        pool2 = bp + extra
                    else:
                        pool2 = bp
                    if not pool2:
                        pool2 = nxt[:min(need, len(nxt))]
                    la = min(eval_seq(seq + [t, u]) for u in pool2)
                cand_la.append((t, min(imc, la)))
            cand_la.sort(key=lambda z: z[1])
            cand_la = anti_buddy_deprioritize(last, cand_la, cand_la[0][1] if cand_la else float('inf'))

            choose = cand_la[0][0] if cand_la else imm[0][0]
            seq.append(choose)
            remaining.remove(choose)

            # Periodically greedy-complete to raise incumbent
            if step % 10 == 0 and remaining:
                full, fc = greedy_complete(seq)
                if fc < inc_ref[0]:
                    inc_ref[0] = fc
                    inc_ref[1] = full
            step += 1

        return seq, eval_seq(seq)

    # LNS utilities
    def boundary_penalty(a, b):
        return max(0, M[a][b] - M[b][a])

    def worst_violation_indices(seq, topm=3):
        pairs = []
        for i in range(len(seq) - 1):
            a, b = seq[i], seq[i + 1]
            pen = boundary_penalty(a, b)
            if pen > 0:
                pairs.append((pen, i))
        pairs.sort(reverse=True)
        return [i for _, i in pairs[:topm]]

    def prefix_marginals(seq):
        prefix_costs = [0] * len(seq)
        c = 0
        for i in range(len(seq)):
            c = eval_seq(seq[: i + 1])
            prefix_costs[i] = c
        marg = [prefix_costs[0]] + [prefix_costs[i] - prefix_costs[i - 1] for i in range(1, len(seq))]
        return prefix_costs, marg

    # ΔW surrogate for block operation evaluation: sum of boundary penalties
    def surrogate_block_cost(seq, start, end):
        # cost contributed by boundaries around [start:end]
        cost = 0
        if start - 1 >= 0:
            cost += boundary_penalty(seq[start - 1], seq[start])
        if end < len(seq):
            cost += boundary_penalty(seq[end - 1], seq[end])
        if start - 1 >= 0 and end < len(seq):
            # penalty if block removed; used as baseline
            cost += boundary_penalty(seq[start - 1], seq[end])
        return cost

    def lns_improve(rng, seq, base_cost, round_budget=700):
        best_seq = seq[:]
        best_cost = base_cost
        nloc = len(best_seq)
        if nloc <= 2:
            return best_seq, best_cost

        # Iterations adapt to size
        rounds = 3 + max(1, nloc // 50)

        for _ in range(rounds):
            budget_left = round_budget

            # 1) Boundary-focused window permutations
            k_win = 6 if nloc >= 40 else 5
            viols = worst_violation_indices(best_seq, topm=3)
            for v in viols:
                if budget_left <= 0:
                    break
                start = max(0, min(v - k_win // 2, nloc - k_win))
                block = best_seq[start:start + k_win]
                base = best_seq[:start] + best_seq[start + k_win:]
                best_blk_seq = None
                best_blk_cost = best_cost
                # Try a mixture of systematic and random perms within budget
                attempts = 0
                tried = set()
                full = math_factorial_cap(k_win, 720)
                exhaustive = full[1]
                perm_budget = min(240, budget_left)
                if exhaustive:
                    for p in itertools.permutations(block):
                        cand = base[:start] + list(p) + base[start:]
                        c = eval_seq(cand)
                        budget_left -= 1
                        if c < best_blk_cost:
                            best_blk_cost = c
                            best_blk_seq = cand
                        if budget_left <= 0:
                            break
                else:
                    while attempts < perm_budget and budget_left > 0:
                        p = tuple(rng.sample(block, len(block)))
                        if p in tried:
                            continue
                        tried.add(p)
                        cand = base[:start] + list(p) + base[start:]
                        c = eval_seq(cand)
                        budget_left -= 1
                        attempts += 1
                        if c < best_blk_cost:
                            best_blk_cost = c
                            best_blk_seq = cand
                if best_blk_seq is not None and best_blk_cost < best_cost:
                    best_seq, best_cost = best_blk_seq, best_blk_cost
                    nloc = len(best_seq)
                    if budget_left <= 0:
                        break

            if budget_left <= 0:
                break

            # 2) Block-swap and block-reinsert moves ranked by surrogate
            # Determine "hot" centers from violations and marginal spikes
            _, marg = prefix_marginals(best_seq)
            hot_positions = set(worst_violation_indices(best_seq, topm=3))
            hot_positions.update(sorted(range(nloc), key=lambda i: marg[i], reverse=True)[:3])
            block_sizes = [3, 4, 5, 6] if nloc >= 30 else [3, 4, 5]

            candidates = []

            # Build swap candidates (two blocks)
            for sz in block_sizes:
                for c in list(hot_positions):
                    i = max(0, min(c - sz // 2, nloc - sz))
                    j = i + sz
                    if j > nloc or i >= j:
                        continue
                    for k in block_sizes:
                        a = j + 1
                        if a + k > nloc:
                            continue
                        # surrogate ranking: sum penalties around both blocks
                        surr = surrogate_block_cost(best_seq, i, j) + surrogate_block_cost(best_seq, a, a + k)
                        candidates.append(("swap", (i, j, a, a + k), surr))

            # Build reinsert candidates
            for sz in block_sizes:
                for c in list(hot_positions):
                    i = max(0, min(c - sz // 2, nloc - sz))
                    j = i + sz
                    if j > nloc or i >= j:
                        continue
                    # try reinserting at few promising anchors
                    anchors = [0, j, nloc - 1]
                    anchors.extend([max(0, i - 5), min(nloc, j + 5)])
                    anchors = list(sorted(set([a for a in anchors if not (i <= a < j)])))
                    surr = surrogate_block_cost(best_seq, i, j)
                    for a in anchors:
                        candidates.append(("reinsert", (i, j, a), surr))

            # Rank by surrogate (descending surr => likely worse boundary => more gain if fixed)
            candidates.sort(key=lambda x: -x[2])
            if candidates:
                # Evaluate top 40% + 10% random
                topN = max(1, int(0.4 * len(candidates)))
                eval_set = set(range(min(topN, len(candidates))))
                extra = max(0, int(0.1 * len(candidates)))
                while len(eval_set) < min(topN + extra, len(candidates)):
                    eval_set.add(rng.randint(0, len(candidates) - 1))
                improved = False
                for idx in list(sorted(eval_set)):
                    if budget_left <= 0:
                        break
                    kind, params, _s = candidates[idx]
                    if kind == "swap":
                        i, j, a, b = params
                        block1 = best_seq[i:j]
                        block2 = best_seq[a:b]
                        cand = best_seq[:i] + block2 + best_seq[j:a] + block1 + best_seq[b:]
                    else:  # reinsert
                        i, j, a = params
                        block = best_seq[i:j]
                        base = best_seq[:i] + best_seq[j:]
                        a = max(0, min(a, len(base)))
                        cand = base[:a] + block + base[a:]
                    c = eval_seq(cand)
                    budget_left -= 1
                    if c < best_cost:
                        best_seq, best_cost = cand, c
                        nloc = len(best_seq)
                        improved = True
                        break
                if improved and budget_left <= 0:
                    break

            if budget_left <= 0:
                break

            # 3) Targeted relocations of hot single transactions
            positions = sorted(range(nloc), key=lambda i: marg[i], reverse=True)[:3]
            for pos in positions:
                if budget_left <= 0 or pos >= len(best_seq):
                    break
                t = best_seq[pos]
                base = best_seq[:pos] + best_seq[pos + 1 :]
                positions_try = {0, len(base)}
                for _ in range(6):
                    positions_try.add(rng.randint(0, len(base)))
                best_pos_idx = None
                best_pos_cost = best_cost
                for j in positions_try:
                    cand = base[:j] + [t] + base[j:]
                    c = eval_seq(cand)
                    budget_left -= 1
                    if c < best_pos_cost:
                        best_pos_cost = c
                        best_pos_idx = j
                    if budget_left <= 0:
                        break
                if best_pos_idx is not None and best_pos_cost < best_cost:
                    best_seq = base[:best_pos_idx] + [t] + base[best_pos_idx:]
                    best_cost = best_pos_cost

        return best_seq, best_cost

    def math_factorial_cap(k, cap):
        # Returns (value, is_exact_exhaustive_possible)
        val = 1
        for i in range(2, k + 1):
            val *= i
            if val > cap:
                return (val, False)
        return (val, True)

    # Portfolio of 3 restarts with param perturbations (shared caches)
    profiles = [
        {"beam_A": 16, "beam_B": 10, "cand_A": 12, "cand_B": 8, "lookahead_top_A": 4, "lookahead_top_B": 3, "next_k_A": 6, "next_k_B": 5},
        {"beam_A": max(12, min(16, int(num_seqs))), "beam_B": 12, "cand_A": 14, "cand_B": 8, "lookahead_top_A": 3, "lookahead_top_B": 3, "next_k_A": 5, "next_k_B": 5},
        {"beam_A": 10, "beam_B": 10, "cand_A": 10, "cand_B": 10, "lookahead_top_A": 3, "lookahead_top_B": 3, "next_k_A": 4, "next_k_B": 4},
    ]

    best_overall_cost = float('inf')
    best_overall_seq = None

    for r in range(3):
        rng = random.Random(base_seed + r)
        inc_ref = [incumbent_cost, incumbent_seq]

        # 1) Beam seed per profile
        bseq, bcost = beam_seed(rng, param_override=profiles[r])

        # 2) Greedy seed
        gseq, gcost = build_initial_sequence(rng, inc_ref)

        seeds = [(bcost, bseq), (gcost, gseq)]
        # Add a quick tournament singleton greedy as third seed
        t0 = tournament_order[0] if r == 0 else rng.choice(tournament_order[:min(10, n)])
        sseq, scost = greedy_complete([t0])
        seeds.append((scost, sseq))

        seeds.sort(key=lambda z: z[0])
        # Tighten incumbent globally
        if seeds[0][0] < incumbent_cost:
            incumbent_cost, incumbent_seq = seeds[0]
        # LNS on top two seeds
        improved = []
        for i in range(min(2, len(seeds))):
            s_cost, s_seq = seeds[i]
            lns_round_budget = 700 if n >= 60 else 500
            seq_improved, cost_improved = lns_improve(rng, s_seq, s_cost, round_budget=lns_round_budget)
            improved.append((cost_improved, seq_improved))

        improved.sort(key=lambda z: z[0])
        if improved and improved[0][0] < best_overall_cost:
            best_overall_cost, best_overall_seq = improved[0]

    # Final light random insertion and 2-opt polish
    if best_overall_seq is None:
        # Fallback: greedy from tournament best
        best_overall_seq, best_overall_cost = greedy_complete([tournament_order[0]])
    else:
        rng = random.Random(base_seed + 97)
        seq = best_overall_seq[:]
        cost = best_overall_cost
        two_opt_trials = min(60, n)
        for _ in range(two_opt_trials):
            i, j = rng.sample(range(n), 2)
            if abs(i - j) <= 1:
                continue
            cand = seq[:]
            cand[i], cand[j] = cand[j], cand[i]
            c = eval_seq(cand)
            if c < cost:
                seq, cost = cand, c
        move_budget = min(60, n)
        for _ in range(move_budget):
            i, j = rng.sample(range(n), 2)
            if i == j:
                continue
            cand = seq[:]
            item = cand.pop(i)
            cand.insert(j, item)
            c = eval_seq(cand)
            if c < cost:
                seq, cost = cand, c
        best_overall_seq, best_overall_cost = seq, cost

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
    makespan1, schedule1 = get_best_schedule(workload, 12)
    cost1 = workload.get_opt_seq_cost(schedule1)

    # Workload 2: Simple read-then-write pattern
    workload2 = Workload(WORKLOAD_2)
    makespan2, schedule2 = get_best_schedule(workload2, 12)
    cost2 = workload2.get_opt_seq_cost(schedule2)

    # Workload 3: Minimal read/write operations
    workload3 = Workload(WORKLOAD_3)
    makespan3, schedule3 = get_best_schedule(workload3, 12)
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