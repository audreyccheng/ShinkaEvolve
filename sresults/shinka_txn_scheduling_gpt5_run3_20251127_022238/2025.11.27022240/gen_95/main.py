# EVOLVE-BLOCK-START
"""Transaction scheduling algorithm for optimizing makespan across multiple workloads"""

import time
import random
import sys
import os
from collections import defaultdict

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
    Buddy-guided, prefix-dominance beam search with greedy promotion and
    conflict-focused LNS refinement to minimize makespan.

    Args:
        workload: Workload object containing transaction data
        num_seqs: Number of random restarts (used as intensity; also time-bounded)

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    N = workload.num_txns
    start_time = time.time()
    # Budget tuned for combined score; slightly adaptive by N
    time_budget_sec = 0.56 if N >= 90 else 0.50

    def time_left():
        return (time.time() - start_time) < time_budget_sec

    rng = random.Random(1729 + 31 * N)

    # Shared caches across all phases/restarts
    cost_cache = {}
    ext_cache = {}

    def eval_seq_cost(seq):
        key = tuple(seq)
        cached = cost_cache.get(key)
        if cached is not None:
            return cached
        c = workload.get_opt_seq_cost(seq)
        cost_cache[key] = c
        return c

    def eval_ext_cost(prefix_tuple, cand):
        key = (prefix_tuple, cand)
        cached = ext_cache.get(key)
        if cached is not None:
            return cached
        c = eval_seq_cost(list(prefix_tuple) + [cand])
        ext_cache[key] = c
        return c

    all_txns = list(range(N))

    # Precompute singleton costs to seed and for lower bounds
    singleton_cost = {}
    for t in all_txns:
        if not time_left():
            break
        singleton_cost[t] = eval_seq_cost([t])
    singles_sorted = sorted(all_txns, key=lambda x: singleton_cost.get(x, float('inf')))

    # Pairwise marginal extension delta: pair_cost(a,b) = cost([a,b]) - cost([a])
    pair_cost_cache = {}
    def pair_cost(a, b):
        key = (a, b)
        v = pair_cost_cache.get(key)
        if v is not None:
            return v
        base = singleton_cost.get(a)
        if base is None:
            base = eval_seq_cost([a])
            singleton_cost[a] = base
        ec = eval_ext_cost((a,), b)
        d = ec - base
        pair_cost_cache[key] = d
        return d

    # Symmetric preference W[a][b] = cost([a,b]) - cost([b,a]); computed on-demand
    pair_pref_cache = {}
    def pair_pref(a, b):
        key = (a, b)
        v = pair_pref_cache.get(key)
        if v is not None:
            return v
        ab = eval_seq_cost([a, b])
        ba = eval_seq_cost([b, a])
        d = ab - ba
        pair_pref_cache[key] = d
        pair_pref_cache[(b, a)] = -d
        return d

    # Build buddy lists B[t]: top partners minimizing cost([t,u]) - cost([t])
    # Also derive per-'last' anti-buddy threshold as 75th percentile of positive deltas
    def build_buddies(max_buddies=8):
        buds = {t: [] for t in all_txns}
        anti_thresh = {t: float('inf') for t in all_txns}
        top_slice = singles_sorted[:min(20, max(8, N // 6))]
        for t in all_txns:
            if not time_left():
                break
            pool = list(top_slice)
            others = [u for u in all_txns if u != t and u not in pool]
            if others:
                pool.extend(rng.sample(others, min(24, max(10, N // 5))))
            # Dedup and remove t
            seen = set()
            cand = []
            for u in pool:
                if u == t or u in seen:
                    continue
                seen.add(u)
                cand.append(u)
            scored = []
            pos = []
            base = singleton_cost[t]
            pt = (t,)
            for u in cand:
                c2 = eval_ext_cost(pt, u)
                delta = c2 - base
                scored.append((delta, u))
                if delta > 0:
                    pos.append(delta)
            scored.sort(key=lambda x: x[0])
            buds[t] = [u for _d, u in scored[:max_buddies]]
            if pos:
                pos.sort()
                idx = int(0.75 * (len(pos) - 1))
                anti_thresh[t] = pos[idx]
        return buds, anti_thresh

    buddies, anti_buddy_thresh = build_buddies(max_buddies=8)

    def is_antibuddy(last, cand):
        if last is None:
            return False
        thr = anti_buddy_thresh.get(last, float('inf'))
        if thr == float('inf'):
            return False
        pc = pair_cost(last, cand)
        return pc > 0 and pc >= thr

    # Global prefix-dominance shared across restarts: (frozenset(remaining), suffix) -> best prefix cost
    prefix_dom = {}

    def dom_sig(rem_set, seq, k):
        if k <= 0:
            return (frozenset(rem_set), ())
        tail = tuple(seq[-k:]) if len(seq) >= k else tuple(seq)
        return (frozenset(rem_set), tail)

    # Lower bound using max remaining singleton cost
    def lb_singleton(cur_cost, rem_set):
        if not rem_set:
            return cur_cost
        m = 0.0
        for t in rem_set:
            c = singleton_cost.get(t)
            if c is None:
                c = eval_seq_cost([t])
                singleton_cost[t] = c
            if c > m:
                m = c
        return max(cur_cost, m)

    # Conflict-aware greedy completion with anti-buddy gating and prefix-dom updates
    def greedy_complete(seq, rem_set, branch_k=12, incumbent=None, k_suffix=3):
        seq_out = list(seq)
        rem = set(rem_set)
        cur_cost = eval_seq_cost(seq_out) if seq_out else 0.0
        while rem and time_left():
            # Update dominance with current prefix
            ds = dom_sig(rem, seq_out, k_suffix)
            prev = prefix_dom.get(ds)
            if prev is None or cur_cost < prev:
                prefix_dom[ds] = cur_cost

            # Incumbent prune
            if incumbent is not None and lb_singleton(cur_cost, rem) >= incumbent:
                break

            last = seq_out[-1] if seq_out else None
            rem_list = list(rem)

            cand_pool = []
            # buddies of last
            if last is not None:
                for u in buddies.get(last, []):
                    if u in rem and u not in cand_pool:
                        cand_pool.append(u)
            # low-singletons
            low_single = sorted(rem_list, key=lambda t: singleton_cost.get(t, float('inf')))[:min(4, len(rem_list))]
            for u in low_single:
                if u not in cand_pool:
                    cand_pool.append(u)
            # fill random
            need = max(0, branch_k - len(cand_pool))
            others = [x for x in rem_list if x not in cand_pool]
            if need > 0 and others:
                cand_pool.extend(rng.sample(others, min(need, len(others))))
            if not cand_pool:
                cand_pool = rem_list

            pt = tuple(seq_out)
            scored = []
            best_immediate = float('inf')
            for t in cand_pool:
                c = eval_ext_cost(pt, t)
                scored.append((c, t))
                if c < best_immediate:
                    best_immediate = c
            if not scored:
                # exhausted time; append remaining arbitrarily
                seq_out.extend(rem)
                cur_cost = eval_seq_cost(seq_out)
                return cur_cost, seq_out

            # Anti-buddy: skip strongly disfavored unless within 1% of best immediate
            tol = best_immediate * 1.01
            filtered = []
            for c, t in scored:
                if is_antibuddy(last, t) and c > tol:
                    continue
                filtered.append((c, t))
            if not filtered:
                filtered = scored
            filtered.sort(key=lambda x: x[0])
            best_c, best_t = filtered[0]

            # Extend
            seq_out.append(best_t)
            rem.remove(best_t)
            cur_cost = best_c

        if rem:
            seq_out.extend(list(rem))
            cur_cost = eval_seq_cost(seq_out)
        return cur_cost, seq_out

    # Core beam runner with unified prefix-dominance and child-level greedy promotion
    def run_beam(params, incumbent_cost=float('inf')):
        beam_width = params['beam']
        branch_factor = params['branch']
        lookahead_top = params['lookahead_top']
        next_k = params['next_k']

        # Seed: best singletons + a few randoms
        seeds = singles_sorted[:max(beam_width * 2, 8)]
        others = [x for x in all_txns if x not in seeds]
        if others:
            seeds.extend(rng.sample(others, min(6, len(others))))
        beam = []
        for t in seeds:
            if not time_left():
                break
            seq = [t]
            rem = set(all_txns); rem.remove(t)
            c = eval_seq_cost(seq)
            beam.append((c, seq, rem))
        if not beam:
            seq = all_txns[:]; rng.shuffle(seq)
            return eval_seq_cost(seq), seq

        beam.sort(key=lambda x: x[0])
        beam = beam[:beam_width]

        best_full_cost = incumbent_cost
        best_full_seq = None

        steps = N - 1
        depth = 0
        while depth < steps and time_left():
            depth += 1
            new_beam = []

            # Depth-adaptive dominance suffix length
            k_cur = 3 if depth / max(1, N) < 0.7 else 4

            # Complete top beam items occasionally
            for cost_so_far, seq, rem in beam[:min(2, len(beam))]:
                if not time_left():
                    break
                c_try, s_try = greedy_complete(seq, rem, branch_k=max(8, N // 10), incumbent=best_full_cost, k_suffix=k_cur)
                if len(s_try) == N and c_try < best_full_cost:
                    best_full_cost, best_full_seq = c_try, s_try

            for cost_so_far, seq, rem in beam:
                if not rem:
                    if cost_so_far < best_full_cost:
                        best_full_cost, best_full_seq = cost_so_far, seq[:]
                    continue

                # Incumbent prune
                if cost_so_far >= best_full_cost:
                    continue

                # Prefix dominance at parent
                sig_parent = dom_sig(rem, seq, k_cur)
                prev = prefix_dom.get(sig_parent)
                if prev is not None and cost_so_far >= prev:
                    continue
                # Update dominance
                prefix_dom[sig_parent] = cost_so_far if prev is None else min(prev, cost_so_far)

                # Candidate pool: buddies of last + low-singletons + random fill up to 2*branch
                last = seq[-1]
                rem_list = list(rem)
                cand_pool = []
                for u in buddies.get(last, []):
                    if u in rem and u not in cand_pool:
                        cand_pool.append(u)
                # add a few low singletons
                low_s = sorted(rem_list, key=lambda t: singleton_cost.get(t, float('inf')))[:min(5, len(rem_list))]
                for u in low_s:
                    if u not in cand_pool:
                        cand_pool.append(u)
                # random fill
                need = max(0, 2 * branch_factor - len(cand_pool))
                if need > 0:
                    others = [x for x in rem_list if x not in cand_pool]
                    if others:
                        cand_pool.extend(rng.sample(others, min(need, len(others))))
                if not cand_pool:
                    cand_pool = rem_list if len(rem_list) <= 2 * branch_factor else rng.sample(rem_list, 2 * branch_factor)

                # Immediate extension costs and anti-buddy gating
                pt = tuple(seq)
                tmp = []
                best_immediate = float('inf')
                for cand in cand_pool:
                    if not time_left():
                        break
                    ec = eval_ext_cost(pt, cand)
                    tmp.append((ec, cand))
                    if ec < best_immediate:
                        best_immediate = ec

                scored = []
                tol = best_immediate * 1.01
                for ec, cand in tmp:
                    if ec >= best_full_cost:
                        continue
                    if is_antibuddy(last, cand) and ec > tol:
                        continue
                    # Shallow lookahead: buddies of cand
                    la_best = ec
                    if rem and time_left():
                        new_rem = rem.copy(); new_rem.remove(cand)
                        la_pool = [v for v in buddies.get(cand, []) if v in new_rem]
                        if not la_pool:
                            la_pool = list(new_rem)
                        if len(la_pool) > lookahead_top:
                            la_pool = rng.sample(la_pool, lookahead_top)
                        new_pt = tuple(seq + [cand])
                        for nxt in la_pool:
                            c2 = eval_ext_cost(new_pt, nxt)
                            if c2 < la_best:
                                la_best = c2
                    scored.append((ec - cost_so_far, la_best, ec, cand))

                if not scored:
                    continue

                scored.sort(key=lambda x: (x[0], x[1]))
                top = scored[:min(branch_factor, len(scored))]

                # Child-level greedy promotion: probe K children and prune by incumbent
                K = 2 if depth / max(1, N) < 0.7 else 1
                promoted = 0
                for idx, (_delta, la, ec, cand) in enumerate(top):
                    new_seq = seq + [cand]
                    new_rem = rem.copy(); new_rem.remove(cand)
                    # Prefix dominance at child
                    sig_child = dom_sig(new_rem, new_seq, k_cur)
                    prevc = prefix_dom.get(sig_child)
                    if prevc is not None and ec >= prevc:
                        continue
                    prefix_dom[sig_child] = ec if prevc is None else min(prevc, ec)

                    # Lower bound prune
                    if lb_singleton(ec, new_rem) >= best_full_cost:
                        continue

                    adj_la = la
                    if promoted < K and time_left():
                        g_cost, g_seq = greedy_complete(new_seq, new_rem, branch_k=max(6, N // 12), incumbent=best_full_cost, k_suffix=k_cur)
                        promoted += 1
                        if len(g_seq) == N and g_cost < best_full_cost:
                            best_full_cost, best_full_seq = g_cost, g_seq
                        adj_la = min(adj_la, g_cost)
                        # prune if its greedy completion already not improving
                        if g_cost >= best_full_cost:
                            continue

                    new_beam.append((ec, new_seq, new_rem, adj_la))

            if not new_beam:
                break

            # Select next beam by adjusted lookahead score; keep unique prefixes
            new_beam.sort(key=lambda x: x[3])
            unique = []
            seen = set()
            for ec, s, r, la in new_beam:
                key = tuple(s)
                if key in seen:
                    continue
                seen.add(key)
                unique.append((ec, s, r))
                if len(unique) >= beam_width:
                    break
            beam = unique

            # Periodically finalize top-2
            if beam and time_left():
                for ec, s, r in beam[:min(2, len(beam))]:
                    c_try, s_try = greedy_complete(s, r, branch_k=max(8, N // 10), incumbent=best_full_cost, k_suffix=k_cur)
                    if len(s_try) == N and c_try < best_full_cost:
                        best_full_cost, best_full_seq = c_try, s_try

        # Greedily finish remaining prefixes
        for ec, s, r in beam:
            if not time_left():
                break
            c_fin, s_fin = greedy_complete(s, r, branch_k=max(8, N // 10), incumbent=best_full_cost, k_suffix=4)
            if len(s_fin) == N and c_fin < best_full_cost:
                best_full_cost, best_full_seq = c_fin, s_fin

        if best_full_seq is None:
            seq = all_txns[:]; rng.shuffle(seq)
            best_full_seq = seq
            best_full_cost = eval_seq_cost(seq)
        return best_full_cost, best_full_seq

    # Helpers for LNS
    def W_pref(a, b):
        return pair_pref(a, b)

    def worst_adjacencies(seq, topk=6):
        viols = []
        for i in range(len(seq) - 1):
            a, b = seq[i], seq[i + 1]
            m = W_pref(a, b)  # positive prefers b before a -> bad adjacency
            viols.append((m, i))
        viols.sort(key=lambda x: -x[0])
        return viols[:topk]

    def conflict_targeted_25opt(seq, cur_cost, budget=70):
        best_seq = seq[:]
        best_cost = cur_cost
        n = len(seq)
        if n < 6:
            return best_cost, best_seq
        viols = worst_adjacencies(seq, topk=min(20, max(4, n // 8)))
        # Build candidates j far from worst i
        cands = []
        for v, i in viols:
            if i < 0 or i + 1 >= n:
                continue
            a, b = seq[i], seq[i + 1]
            for _ in range(3):
                j = rng.randrange(0, n - 1)
                if abs(j - i) <= 2:
                    continue
                cands.append((v + W_pref(seq[j], seq[j + 1]), i, j))
        if not cands:
            return best_cost, best_seq
        cands.sort(key=lambda x: -x[0])
        k_eval = max(3, int(0.4 * len(cands)))
        pick = cands[:k_eval]
        # add 10% random
        extra = max(1, len(cands) // 10)
        pick_ids = set((i, j) for _, i, j in pick)
        for _ in range(extra):
            _, i, j = rng.choice(cands)
            if (i, j) not in pick_ids:
                pick.append((0.0, i, j))
                pick_ids.add((i, j))

        evals = 0
        for _, i, j in pick:
            if evals >= budget or not time_left():
                break
            # 2-opt style swap endpoints
            cand = best_seq[:]
            cand[i], cand[j] = cand[j], cand[i]
            c = eval_seq_cost(cand); evals += 1
            if c < best_cost:
                best_cost, best_seq = c, cand
                continue
            if evals >= budget or not time_left():
                break
            # 2.5-opt: move small block around b before a (bridge move variant)
            s = min(4, max(3, n // 40))
            start = max(0, min(i + 1, n - s))
            block = best_seq[start:start + s]
            remain = best_seq[:start] + best_seq[start + s:]
            # choose insert pos before index i
            pos = max(0, min(i - 1, len(remain)))
            cand2 = remain[:]
            if rng.random() < 0.5:
                block_use = block[::-1]
            else:
                block_use = block
            for off, x in enumerate(block_use):
                cand2.insert(pos + off, x)
            c2 = eval_seq_cost(cand2); evals += 1
            if c2 < best_cost:
                best_cost, best_seq = c2, cand2
        return best_cost, best_seq

    def anchored_window_reinsert(seq, cur_cost, k=7, eval_cap=320):
        # Fix one boundary element near worst violation; permute and reinsert interior greedily
        best_seq = seq[:]
        best_cost = cur_cost
        n = len(seq)
        if n < k:
            return best_cost, best_seq
        viols = worst_adjacencies(seq, topk=3)
        evals = 0
        for _, idx in viols:
            if not time_left():
                break
            # Build window around idx: [s, e)
            s = max(0, min(idx - (k // 2), n - k))
            e = s + k
            window = best_seq[s:e]
            if len(window) < 3:
                continue
            # Two anchor modes: fix left or right
            for anchor_left in (True, False):
                if evals >= eval_cap or not time_left():
                    break
                if anchor_left:
                    anchor = window[0]
                    interior = window[1:]
                else:
                    anchor = window[-1]
                    interior = window[:-1]
                # Greedy construct interior order by minimizing pair_pref contributions w.r.t anchor and neighbors
                # Create a candidate ranking list; sample limited permutations via randomized greedy
                trials = 10
                for _ in range(trials):
                    if evals >= eval_cap or not time_left():
                        break
                    avail = interior[:]
                    rng.shuffle(avail)
                    build = [anchor] if anchor_left else []
                    while avail and time_left():
                        # choose next with best marginal W score to last placed
                        last = build[-1] if build else None
                        scored = []
                        for u in avail:
                            score = 0.0
                            if last is not None:
                                score += W_pref(last, u)
                            # small lookahead: favor u with low singleton
                            score += 0.1 * singleton_cost.get(u, 0.0)
                            scored.append((score, u))
                        scored.sort(key=lambda x: x[0])
                        u = scored[0][1]
                        build.append(u)
                        avail.remove(u)
                    if not anchor_left:
                        build.append(anchor)
                    # Reinsert built window back into sequence at s..e
                    cand = best_seq[:s] + build + best_seq[e:]
                    c = eval_seq_cost(cand); evals += 1
                    if c < best_cost:
                        best_cost, best_seq = c, cand
        return best_cost, best_seq

    # LNS refinement: adjacent swaps, boundary repair, block swaps, anchored reinsert, and conflict-2.5opt
    def lns_improve(seq, cur_cost):
        best_seq = seq[:]
        best_cost = cur_cost
        n = len(best_seq)

        if n <= 2 or not time_left():
            return best_cost, best_seq

        # Tournament bubble pass around worst adjacencies (2 passes)
        for _ in range(2):
            improved = False
            # prioritize indices near worst violations
            viols = worst_adjacencies(best_seq, topk=min(10, max(4, n // 12)))
            idxs = sorted({i for _v, i in viols})
            if not idxs:
                idxs = list(range(n - 1))
            for i in idxs:
                if i < 0 or i + 1 >= n:
                    continue
                if not time_left():
                    break
                cand = best_seq[:]
                cand[i], cand[i + 1] = cand[i + 1], cand[i]
                c = eval_seq_cost(cand)
                if c < best_cost:
                    best_cost, best_seq = c, cand
                    improved = True
            if not improved or not time_left():
                break

        if not time_left():
            return best_cost, best_seq

        # Anchored window reinsert around worst boundary
        best_cost, best_seq = anchored_window_reinsert(best_seq, best_cost, k=min(7, max(5, n // 25)), eval_cap=320 if N >= 90 else 240)
        if not time_left():
            return best_cost, best_seq

        # Block-swap neighborhood around top-2 worst adjacencies (limited tries)
        n = len(best_seq)
        if n >= 8 and time_left():
            worst = worst_adjacencies(best_seq, topk=4)
            tries = 0
            for a in range(min(2, len(worst))):
                if not time_left() or tries >= 4:
                    break
                for b in range(a + 1, min(4, len(worst))):
                    if not time_left() or tries >= 4:
                        break
                    i = worst[a][1]; j = worst[b][1]
                    block = min(6, max(3, n // 40))
                    si = max(0, min(i - block // 2, n - block))
                    sj = max(0, min(j - block // 2, n - block))
                    if abs(si - sj) < block:
                        continue
                    cand = best_seq[:]
                    if si > sj:
                        si, sj = sj, si
                    block_i = cand[si:si + block]
                    block_j = cand[sj:sj + block]
                    mid = cand[si + block:sj]
                    cand2 = cand[:si] + block_j + mid + block_i + cand[sj + block:]
                    c = eval_seq_cost(cand2); tries += 1
                    if c < best_cost:
                        best_cost, best_seq = c, cand2

        if not time_left():
            return best_cost, best_seq

        # Conflict-targeted 2.5-opt sampling
        best_cost, best_seq = conflict_targeted_25opt(best_seq, best_cost, budget=80 if N >= 90 else 60)

        # Final cheap adjacent pass
        if time_left():
            for i in range(len(best_seq) - 1):
                if not time_left():
                    break
                cand = best_seq[:]
                cand[i], cand[i + 1] = cand[i + 1], cand[i]
                c = eval_seq_cost(cand)
                if c < best_cost:
                    best_cost, best_seq = c, cand

        return best_cost, best_seq

    # Portfolio parameter settings (deterministic), shared caches and prefix_dom across restarts
    portfolios = [
        {'beam': 16, 'branch': 12, 'lookahead_top': 4, 'next_k': 6},
        {'beam': 12, 'branch': 14, 'lookahead_top': 3, 'next_k': 5},
        {'beam': 10, 'branch': 10, 'lookahead_top': 2, 'next_k': 4},
    ]

    global_best_cost = float('inf')
    global_best_seq = None

    # Deterministic portfolio restarts within budget
    max_restarts = max(2, min(len(portfolios), int(num_seqs)))
    for r in range(max_restarts):
        if not time_left():
            break
        params = portfolios[r % len(portfolios)]
        c_seed, s_seed = run_beam(params, incumbent_cost=global_best_cost)
        # Apply LNS only to best two seeds overall
        if c_seed < global_best_cost:
            global_best_cost, global_best_seq = c_seed, s_seed

    # Select top-2 seeds by greedy polish from prefix_dom hints (if available)
    candidates = []
    if global_best_seq is not None:
        candidates.append((global_best_cost, global_best_seq))
    # Build an additional candidate by greedy from the best singleton
    if time_left() and singles_sorted:
        c2, s2 = greedy_complete([singles_sorted[0]], set(x for x in all_txns if x != singles_sorted[0]),
                                 branch_k=max(8, N // 10), incumbent=global_best_cost, k_suffix=3)
        candidates.append((c2, s2))

    # Run LNS on up to top-2 candidates
    candidates.sort(key=lambda x: x[0])
    for i, (c0, s0) in enumerate(candidates[:min(2, len(candidates))]):
        if not time_left():
            break
        c1, s1 = lns_improve(s0, c0)
        if c1 < global_best_cost:
            global_best_cost, global_best_seq = c1, s1

    # Safety: ensure permutation validity
    if global_best_seq is None or len(global_best_seq) != N or len(set(global_best_seq)) != N:
        if global_best_seq is None:
            seq = list(range(N))
            rng.shuffle(seq)
            global_best_seq = seq
        # Repair permutation
        seen = set()
        repaired = []
        for t in global_best_seq:
            if 0 <= t < N and t not in seen:
                repaired.append(t)
                seen.add(t)
        for t in range(N):
            if t not in seen:
                repaired.append(t)
        global_best_seq = repaired[:N]
        global_best_cost = eval_seq_cost(global_best_seq)

    return global_best_cost, global_best_seq


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