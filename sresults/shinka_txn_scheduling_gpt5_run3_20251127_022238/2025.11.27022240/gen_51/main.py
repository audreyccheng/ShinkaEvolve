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
    Portfolio two-phase beam with global prefix-dominance, adaptive lookahead
    with anti-buddy filtering, greedy completions, and ΔW-gated LNS refinement.

    Args:
        workload: Workload object containing transaction data
        num_seqs: Controls beam width / exploration budget

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    n = workload.num_txns
    # Deterministic RNG per workload for stability with slight jitter from num_seqs
    rng = random.Random(1729 + n * 31 + int(num_seqs) * 17)

    # Memoized evaluator for partial sequences to reduce simulator calls
    cost_cache = {}

    def evaluate_seq(seq):
        key = tuple(seq)
        if key in cost_cache:
            return cost_cache[key]
        c = workload.get_opt_seq_cost(seq)
        cost_cache[key] = c
        return c

    txns = list(range(n))

    # Precompute singleton and pairwise costs to capture conflict structure cheaply
    c1 = [0] * n
    for i in range(n):
        c1[i] = evaluate_seq([i])

    M = [[0] * n for _ in range(n)]  # M[i][j] = cost([i, j])
    for i in range(n):
        Mi = M[i]
        for j in range(n):
            if i == j:
                Mi[j] = c1[i]
            else:
                Mi[j] = evaluate_seq([i, j])

    # Preference margins: W[i][j] < 0 suggests i before j
    W = [[0] * n for _ in range(n)]
    for i in range(n):
        Wi = W[i]
        Mi = M[i]
        for j in range(n):
            if i == j:
                Wi[j] = 0
            else:
                Wi[j] = Mi[j] - M[j][i]

    # Tournament scores: smaller => earlier
    s = [0] * n
    for i in range(n):
        s[i] = sum(W[i][j] for j in range(n) if j != i)

    tournament_order = list(range(n))
    tournament_order.sort(key=lambda x: (s[x], x))

    # Buddy lists: for each t, top partners by small M[t][u]
    buddy_k = 8 if n >= 90 else 6
    buddies = []
    for t in range(n):
        order = sorted((u for u in range(n) if u != t), key=lambda u: M[t][u])
        buddies.append(order[:buddy_k])

    # Strong dislike thresholds for anti-buddy filter (top quartile of positive W[row])
    strong_dislike = [[False] * n for _ in range(n)]
    for x in range(n):
        pos = sorted([W[x][j] for j in range(n) if j != x and W[x][j] > 0])
        if pos:
            qidx = max(0, int(0.75 * (len(pos) - 1)))
            thr = pos[qidx]
            for t in range(n):
                strong_dislike[x][t] = W[x][t] >= thr and W[x][t] > 0

    def prefer_before(a, b):
        # Cheap indication if a before b is no worse than b before a
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

    # Depth-adaptive recent_k
    def recent_k_for_depth(d):
        frac = d / max(1, n - 1)
        if frac < 0.33:
            return 5
        elif frac < 0.66:
            return 4
        else:
            return 3

    # Global prefix-dominance map shared across constructors and beams
    # Key: (frozenset(remaining), tuple(suffix[-k:])), Value: best_cost at this signature
    dom = {}

    def dominated_update(remaining_fset, suffix_tuple, cost):
        prev = dom.get((remaining_fset, suffix_tuple))
        if prev is not None and cost >= prev:
            return True
        dom[(remaining_fset, suffix_tuple)] = cost
        return False

    # Greedy completion routine guided by tournament + buddies with shallow lookahead
    def greedy_complete(seq, rem):
        base = seq[:]
        remaining = list(rem)
        while remaining:
            R = len(remaining)
            # Tournament-guided preselection
            k_pool = min(14, R) if R > 28 else R
            cand_pool = preselect_by_tournament(base, remaining, k_pool, recent_k=recent_k_for_depth(len(base)))
            if not cand_pool:
                cand_pool = remaining[:]

            # immediate costs
            imm = []
            for t in cand_pool:
                c_im = evaluate_seq(base + [t])
                imm.append((t, c_im))
            imm.sort(key=lambda x: x[1])

            # anti-buddy gating reference
            last = base[-1] if base else None
            best_imm = imm[0][1] if imm else float('inf')

            L = min(3, len(imm))
            best_t = imm[0][0]
            best_metric = imm[0][1]
            for t, immediate_c in imm[:L]:
                # Anti-buddy: deprioritize strong dislikes unless it's ≥1% better than current best
                if last is not None and strong_dislike[last][t] and immediate_c > best_imm * 0.99:
                    continue
                rest = [x for x in remaining if x != t]
                if not rest:
                    la = immediate_c
                else:
                    need = 5 if R > 35 else 6
                    buddy_pref = [u for u in buddies[t] if u in rest][:need]
                    if len(buddy_pref) < need:
                        extra = preselect_by_tournament(base + [t], [u for u in rest if u not in buddy_pref], need - len(buddy_pref), recent_k=3)
                        la_pool = buddy_pref + extra
                    else:
                        la_pool = buddy_pref
                    if not la_pool:
                        la_pool = rest[:min(need, len(rest))]
                    la = min(evaluate_seq(base + [t, u]) for u in la_pool)
                metric = min(immediate_c, la)
                if metric < best_metric:
                    best_metric = metric
                    best_t = t

            base.append(best_t)
            remaining.remove(best_t)

            # Global dominance pruning early exits
            rem_fset = frozenset(remaining)
            suffix_k = 3 if len(base) > int(0.6 * n) else 4
            suffix = tuple(base[-suffix_k:]) if len(base) >= suffix_k else tuple(base)
            if dominated_update(rem_fset, suffix, evaluate_seq(base)):
                # If dominated, greedily append remaining by immediate best
                rest = remaining[:]
                rest.sort(key=lambda t: evaluate_seq(base + [t]))
                base.extend(rest)
                remaining.clear()

        return base, evaluate_seq(base)

    def beam_search_two_phase_with_params(beam_A, cand_A, lookahead_top_A, next_k_A, beam_B, cand_B, lookahead_top_B, next_k_B, recent_k_seed):
        split_depth = max(3, int(0.4 * n))

        # Initialize beam with diverse seeds
        starts = []
        starts.append(tournament_order[0])
        # top-10 singletons
        good_singletons = sorted(range(n), key=lambda t: c1[t])[:min(10, n)]
        starts.append(rng.choice(good_singletons))
        # fill remainder with random distinct
        remaining_candidates = [t for t in range(n) if t not in starts]
        rng.shuffle(remaining_candidates)
        initial_width = max(beam_A, beam_B)
        starts.extend(remaining_candidates[:max(0, initial_width - len(starts))])

        beam = []
        used = set()
        for t in starts:
            seq = [t]
            remaining = frozenset(set(txns) - {t})
            c = evaluate_seq(seq)
            # dominance seed
            suffix = tuple(seq)
            dominated_update(remaining, suffix, c)
            key = (tuple(seq), remaining)
            if key in used:
                continue
            used.add(key)
            beam.append((c, seq, remaining))

        incumbent_cost = float('inf')
        incumbent_seq = None

        # Beam loop across all depths
        for d in range(n - 1):
            phaseA = d < split_depth
            beam_width = beam_A if phaseA else beam_B
            cand_per_expand = cand_A if phaseA else cand_B
            lookahead_top = lookahead_top_A if phaseA else lookahead_top_B
            lookahead_next_k = next_k_A if phaseA else next_k_B
            recent_k = recent_k_seed if recent_k_seed else recent_k_for_depth(d)
            suffix_k = 4 if phaseA else 3

            # Periodic greedy completions on top prefixes to tighten incumbent
            beam_sorted = sorted(beam, key=lambda x: x[0])
            for c_prefix, seq_prefix, rem_prefix in beam_sorted[:min(2, len(beam_sorted))]:
                if c_prefix < incumbent_cost:
                    full_seq, full_cost = greedy_complete(seq_prefix, list(rem_prefix))
                    if full_cost < incumbent_cost:
                        incumbent_cost = full_cost
                        incumbent_seq = full_seq

            next_beam = []
            seen = set()

            # Expand each partial sequence
            for cost, seq, remaining in beam:
                # Safe prune by incumbent
                if incumbent_cost < float('inf') and cost >= incumbent_cost:
                    continue

                rem_list = list(remaining)
                if not rem_list:
                    # already full
                    next_beam.append((cost, seq, remaining))
                    continue

                # Tournament-guided preselection
                if len(rem_list) > cand_per_expand * 2:
                    pre = preselect_by_tournament(seq, rem_list, cand_per_expand * 2, recent_k=recent_k)
                else:
                    pre = rem_list

                # Evaluate immediate cost of candidates
                imm = []
                last = seq[-1] if seq else None
                best_immediate = float('inf')
                for t in pre:
                    c_im = evaluate_seq(seq + [t])
                    if c_im < best_immediate:
                        best_immediate = c_im
                    imm.append((t, c_im))

                if not imm:
                    continue
                imm.sort(key=lambda x: x[1])

                # Lookahead over top few with anti-buddy filtering
                L = min(lookahead_top, len(imm))
                scored_ext = []
                for t, immediate_c in imm[:L]:
                    # Anti-buddy: skip strongly disliked unless it's strong immediate gain
                    if last is not None and strong_dislike[last][t] and immediate_c > best_immediate * 0.99:
                        continue

                    next_pool_all = [x for x in rem_list if x != t]
                    if not next_pool_all:
                        la_cost = immediate_c
                    else:
                        la_pref_buddies = [u for u in buddies[t] if u in next_pool_all]
                        need = min(lookahead_next_k, len(next_pool_all))
                        if len(la_pref_buddies) >= need:
                            la_pool = la_pref_buddies[:need]
                        else:
                            extra = preselect_by_tournament(seq + [t], [u for u in next_pool_all if u not in la_pref_buddies], need - len(la_pref_buddies), recent_k=3)
                            la_pool = la_pref_buddies + extra
                            if not la_pool:
                                la_pool = next_pool_all[:need]
                        la_cost = min(evaluate_seq(seq + [t, u]) for u in la_pool)
                    metric = min(immediate_c, la_cost)
                    scored_ext.append((t, metric))

                # Also add a few immediate-best without lookahead to maintain diversity
                diversity_take = min(max(2, cand_per_expand // 3), len(imm))
                for t, c_im in imm[:diversity_take]:
                    scored_ext.append((t, c_im))

                # Deduplicate by candidate t and keep best-k expansions for this parent
                unique = {}
                for t, m in scored_ext:
                    if (t not in unique) or (m < unique[t]):
                        unique[t] = m
                items = sorted(unique.items(), key=lambda z: z[1])

                # In Phase B, evaluate greedy completions for top-4 children to prune safely
                child_bounds = {}
                if not phaseA and items:
                    bound_candidates = [t for t, _ in items[:min(4, len(items))]]
                    for t in bound_candidates:
                        new_seq = seq + [t]
                        rem_new = remaining - {t}
                        full_seq, full_cost = greedy_complete(new_seq, list(rem_new))
                        child_bounds[t] = (full_cost, full_seq)

                expand_k = min(cand_per_expand, len(items))
                kept_any = False
                best_child_local = None
                best_child_cost = float('inf')

                for t, _metric in items[:expand_k]:
                    new_seq = seq + [t]
                    new_cost = evaluate_seq(new_seq)
                    rem_new = remaining - {t}

                    # Safe prune by incumbent
                    if (incumbent_cost < float('inf')) and (new_cost >= incumbent_cost):
                        # keep candidate for guaranteed progress if it's the best local
                        if new_cost < best_child_cost:
                            best_child_cost = new_cost
                            best_child_local = (new_cost, new_seq, rem_new)
                        continue

                    # Safe prune by full-seq child bound if available
                    if not phaseA and (t in child_bounds):
                        fc, fseq = child_bounds[t]
                        if fc >= incumbent_cost:
                            # keep best local child even if pruned
                            if new_cost < best_child_cost:
                                best_child_cost = new_cost
                                best_child_local = (new_cost, new_seq, rem_new)
                            continue
                        # Update incumbent if child full bound is better
                        if fc < incumbent_cost:
                            incumbent_cost = fc
                            incumbent_seq = fseq

                    # Prefix-dominance signature
                    sig_suffix = tuple(new_seq[-suffix_k:]) if len(new_seq) >= suffix_k else tuple(new_seq)
                    if dominated_update(rem_new, sig_suffix, new_cost):
                        # dominated: skip
                        if new_cost < best_child_cost:
                            best_child_cost = new_cost
                            best_child_local = (new_cost, new_seq, rem_new)
                        continue

                    key = (tuple(new_seq), rem_new)
                    if key in seen:
                        continue
                    seen.add(key)
                    next_beam.append((new_cost, new_seq, rem_new))
                    kept_any = True

                # Ensure at least best child stays (guarantee progress)
                if not kept_any and best_child_local is not None:
                    next_beam.append(best_child_local)

            # Fallback if all were pruned
            if not next_beam:
                temp_next = []
                for cost, seq, remaining in beam:
                    rem_list = list(remaining)
                    if not rem_list:
                        temp_next.append((cost, seq, remaining))
                        continue
                    best_t = None
                    best_c = float('inf')
                    for t in rem_list:
                        c = evaluate_seq(seq + [t])
                        if c < best_c:
                            best_c = c
                            best_t = t
                    if best_t is not None:
                        new_seq = seq + [best_t]
                        rem_new = remaining - {best_t}
                        temp_next.append((best_c, new_seq, rem_new))
                next_beam = temp_next

            # Keep top beam_width partial sequences
            next_beam.sort(key=lambda x: x[0])
            beam = next_beam[:beam_width]

        # Finalize best from beam; ensure full sequence coverage
        beam.sort(key=lambda x: x[0])
        best_cost, best_seq, rem = beam[0]
        if len(best_seq) < n:
            best_seq, best_cost = greedy_complete(best_seq, list(rem))

        # Compare against incumbent
        if incumbent_cost < best_cost:
            return incumbent_cost, incumbent_seq
        return best_cost, best_seq

    # ΔW-gated LNS utilities
    def boundary_penalties(seq):
        # penalty for each adjacency i: between seq[i] and seq[i+1]
        pens = []
        for i in range(len(seq) - 1):
            a, b = seq[i], seq[i + 1]
            pens.append(max(0, M[a][b] - M[b][a]))
        return pens

    def lns_improve(seq, base_cost):
        best_seq = seq[:]
        best_cost = base_cost
        nloc = len(best_seq)

        # Quick adjacent hill climb to clean up before heavy LNS
        for _ in range(2):
            improved = False
            for i in range(nloc - 1):
                cand = best_seq[:]
                cand[i], cand[i + 1] = cand[i + 1], cand[i]
                c = evaluate_seq(cand)
                if c < best_cost:
                    best_seq, best_cost = cand, c
                    improved = True
            if not improved:
                break

        rounds = 2 if nloc < 80 else 1
        for _round in range(rounds):
            pens = boundary_penalties(best_seq)
            # pick worst 2 violated boundaries
            worst = sorted([(p, i) for i, p in enumerate(pens)], reverse=True)[:2]
            hot_centers = [i for _, i in worst] if worst else []

            eval_budget = 600 if nloc < 90 else 750
            evaluated = 0

            # Block sizes
            min_k, max_k = 3, 6

            candidates = []

            # Generate block-swap candidates around hot boundaries
            for center in hot_centers:
                k1 = rng.randint(min_k, max_k)
                k2 = rng.randint(min_k, max_k)
                s1 = max(0, min(center - k1 // 2, nloc - k1))
                # second block away from first
                offset = rng.randint(k1 + 1, min(nloc - k2, s1 + k1 + 8)) if nloc > k1 + k2 + 2 else s1 + k1
                s2 = max(0, min(offset, nloc - k2))
                if s1 == s2:
                    continue
                # Build candidate by swapping blocks
                def swap_blocks(arr, s1, k1, s2, k2):
                    if s1 > s2:
                        s1, s2 = s2, s1
                        k1, k2 = k2, k1
                    a = arr[:s1]
                    b1 = arr[s1:s1 + k1]
                    mid = arr[s1 + k1:s2]
                    b2 = arr[s2:s2 + k2]
                    c = arr[s2 + k2:]
                    return a + b2 + mid + b1 + c
                cand = swap_blocks(best_seq, s1, k1, s2, k2)
                # Surrogate: sum of penalties around edited region +-2
                affected = set(range(max(0, s1 - 2), min(nloc - 1, s1 + k1 + 2))) | set(range(max(0, s2 - 2), min(nloc - 1, s2 + k2 + 2)))
                def local_penalty(seq_local, idxs):
                    tot = 0
                    for i in idxs:
                        if i < 0 or i >= len(seq_local) - 1:
                            continue
                        a, b = seq_local[i], seq_local[i + 1]
                        tot += max(0, M[a][b] - M[b][a])
                    return tot
                surr = local_penalty(cand, affected)
                candidates.append((surr, cand))

            # Generate block-reinsert candidates
            for center in hot_centers:
                k = rng.randint(min_k + 1, max_k + 1)
                start = max(0, min(center - k // 2, nloc - k))
                block = best_seq[start:start + k]
                base = best_seq[:start] + best_seq[start + k:]
                # try top-3 positions: ends + one near center
                positions = [0, len(base), max(0, min(len(base), start - 1))]
                for pos in positions:
                    cand = base[:pos] + block + base[pos:]
                    affected = set(range(max(0, pos - 3), min(len(cand) - 1, pos + k + 3)))
                    surr = 0
                    for i in affected:
                        a, b = cand[i], cand[i + 1]
                        surr += max(0, M[a][b] - M[b][a])
                    candidates.append((surr, cand))

            # Rank by surrogate and pick top 40% plus 10% random
            if candidates:
                candidates.sort(key=lambda x: x[0])
                take_top = int(0.4 * len(candidates))
                take_top = max(1, take_top)
                picked = candidates[:take_top]
                # add random 10%
                extra = max(1, int(0.1 * len(candidates)))
                rnd_idx = rng.sample(range(len(candidates)), min(extra, len(candidates)))
                for idx in rnd_idx:
                    picked.append(candidates[idx])

                # Evaluate selected candidates
                for _, cand_seq in picked:
                    if evaluated >= eval_budget:
                        break
                    c = evaluate_seq(cand_seq)
                    evaluated += 1
                    if c < best_cost:
                        best_seq, best_cost = cand_seq, c

            # Sampled relocate moves around hot regions
            tries = min(200, 2 * nloc)
            for _ in range(tries):
                i, j = rng.sample(range(nloc), 2)
                if i == j or abs(i - j) <= 1:
                    continue
                cand = best_seq[:]
                item = cand.pop(i)
                cand.insert(j, item)
                c = evaluate_seq(cand)
                if c < best_cost:
                    best_seq, best_cost = cand, c

            # Sampled 2-opt swaps
            swap_attempts = min(nloc, 80)
            for _ in range(swap_attempts):
                i, j = rng.sample(range(nloc), 2)
                if abs(i - j) <= 1:
                    continue
                cand = best_seq[:]
                cand[i], cand[j] = cand[j], cand[i]
                c = evaluate_seq(cand)
                if c < best_cost:
                    best_seq, best_cost = cand, c

        # Final adjacent clean-up
        for _ in range(2):
            improved = False
            for i in range(nloc - 1):
                cand = best_seq[:]
                cand[i], cand[i + 1] = cand[i + 1], cand[i]
                c = evaluate_seq(cand)
                if c < best_cost:
                    best_seq, best_cost = cand, c
                    improved = True
            if not improved:
                break

        return best_seq, best_cost

    # Portfolio of parameterizations; share caches and dominance map
    portfolio = [
        # (beam_A, cand_A, lookahead_top_A, next_k_A, beam_B, cand_B, lookahead_top_B, next_k_B, recent_k_seed)
        (16, 12, 4, 6, 10, 8, 3, 5, 5),
        (12, 14, 3, 5, 10, 8, 3, 5, 4),
        (10, 10, 3, 4, 10, 8, 2, 4, 3),
    ]

    # Run beams and collect candidates
    beam_results = []
    for params in portfolio:
        bc, bs = beam_search_two_phase_with_params(*params)
        beam_results.append((bc, bs))

    # Choose top-2 for refinement
    beam_results.sort(key=lambda x: x[0])
    top_for_lns = beam_results[:min(2, len(beam_results))]

    best_cost = float('inf')
    best_seq = None

    # Run LNS on top two seeds
    for bc, bs in top_for_lns:
        seq1, cost1 = lns_improve(bs, bc)
        if cost1 < best_cost:
            best_cost, best_seq = cost1, seq1

    # If LNS didn't run (edge), fall back to best beam
    if best_seq is None:
        best_cost, best_seq = beam_results[0]

    return best_cost, best_seq


def get_random_costs():
    """
    Evaluate scheduling algorithm on three different workloads.

    Returns:
        Tuple of (total_makespan, list_of_schedules, execution_time)
    """
    start_time = time.time()

    # Workload 1: Complex mixed read/write transactions
    workload = Workload(WORKLOAD_1)
    # Slightly larger exploration for richer conflict structure
    makespan1, schedule1 = get_best_schedule(workload, 18)
    cost1 = workload.get_opt_seq_cost(schedule1)

    # Workload 2: Simple read-then-write pattern
    workload2 = Workload(WORKLOAD_2)
    makespan2, schedule2 = get_best_schedule(workload2, 14)
    cost2 = workload2.get_opt_seq_cost(schedule2)

    # Workload 3: Minimal read/write operations
    workload3 = Workload(WORKLOAD_3)
    makespan3, schedule3 = get_best_schedule(workload3, 14)
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