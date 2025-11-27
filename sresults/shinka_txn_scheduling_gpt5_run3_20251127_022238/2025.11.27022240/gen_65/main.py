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
    Tournament-guided construction with shared prefix-dominance pruning and ΔW-gated LNS.

    Inputs:
        workload: Workload object
        num_seqs: number of restarts (upper bound; time-bounded)

    Outputs:
        (best_makespan, best_schedule)
    """
    N = workload.num_txns
    rng = random.Random(1729 + 31 * N)

    start_time = time.time()
    # Balance quality/runtime for combined score
    base_budget = 0.60 if N >= 90 else 0.52
    time_budget_sec = base_budget

    def time_left():
        return (time.time() - start_time) < time_budget_sec

    # Shared caches across all phases/restarts
    cost_cache = {}
    ext_cache = {}
    prefix_dom = {}  # (frozenset(remaining), tuple(tail)) -> best_prefix_cost

    def eval_seq_cost(seq):
        key = tuple(seq)
        c = cost_cache.get(key)
        if c is None:
            c = workload.get_opt_seq_cost(seq)
            cost_cache[key] = c
        return c

    def eval_ext_cost(prefix_tuple, cand):
        key = (prefix_tuple, cand)
        c = ext_cache.get(key)
        if c is None:
            c = eval_seq_cost(list(prefix_tuple) + [cand])
            ext_cache[key] = c
        return c

    all_txns = list(range(N))

    # Precompute singleton costs
    singleton_cost = {}
    for t in all_txns:
        if not time_left():
            break
        singleton_cost[t] = eval_seq_cost([t])
    singles_sorted = sorted(all_txns, key=lambda t: singleton_cost.get(t, float('inf')))

    # Build pairwise tournament W[a][b] = cost([a,b]) - cost([b,a])
    # negative -> prefer a before b
    W = defaultdict(dict)
    abs_edges = []

    # Sampling strategy: focus on low-singletons and some random coverage
    global_pool = singles_sorted[:min(N, max(18, N // 3))]
    extra = [x for x in all_txns if x not in global_pool]
    if extra:
        global_pool.extend(rng.sample(extra, min(len(extra), max(18, N // 3))))
    global_pool = list(dict.fromkeys(global_pool))

    max_pair_probes = min(1600, max(700, N * 12))
    probes = 0
    for a in all_txns:
        if not time_left() or probes >= max_pair_probes:
            break
        # choose a limited set of b for each a
        candidates = [x for x in global_pool if x != a]
        rng.shuffle(candidates)
        candidates = candidates[:min(14, max(8, N // 10))]
        for b in candidates:
            if probes >= max_pair_probes or not time_left():
                break
            if b in W.get(a, {}):
                continue
            cab = eval_seq_cost([a, b])
            cba = eval_seq_cost([b, a])
            w = cab - cba
            W[a][b] = w
            W[b][a] = -w
            abs_edges.append(abs(w))
            probes += 2

    # Compute buddies using tournament edges (prefer small W[a][b])
    def build_buddies(max_buddies=8):
        buddies = {t: [] for t in all_txns}
        for t in all_txns:
            row = W.get(t, {})
            if row:
                scored = sorted(row.items(), key=lambda kv: kv[1])  # smaller is better
                buddies[t] = [u for u, val in scored[:max_buddies]]
            else:
                buddies[t] = [u for u in singles_sorted if u != t][:max_buddies]
        return buddies

    buddies = build_buddies(max_buddies=8)

    # Tournament-based initial order: compute score s[t] = sum_u -W[t][u] over known edges
    # Intuition: higher s means t tends to be before others (more negative W[t][u])
    def tournament_rank():
        s = {t: 0.0 for t in all_txns}
        for a in all_txns:
            row = W.get(a, {})
            if not row:
                continue
            acc = 0.0
            for b, w in row.items():
                acc += -w
            s[a] = acc
        # Initial order by score
        order = sorted(all_txns, key=lambda x: (s.get(x, 0.0), singleton_cost.get(x, float('inf'))), reverse=True)
        # One pass of adjacent improvements using true cost for robustness
        best = order[:]
        best_c = eval_seq_cost(best)
        improved = True
        iters = 0
        while improved and iters < 1 and time_left():
            improved = False
            iters += 1
            for i in range(N - 1):
                if not time_left():
                    break
                cand = best[:]
                cand[i], cand[i + 1] = cand[i + 1], cand[i]
                c = eval_seq_cost(cand)
                if c < best_c:
                    best, best_c, improved = cand, c, True
        return best, best_c

    seed_order, seed_cost = tournament_rank()

    # Lower bound using max remaining singleton
    def lb_singleton(cur_cost, rem_set):
        if not rem_set:
            return cur_cost
        m = 0
        for t in rem_set:
            c = singleton_cost.get(t)
            if c is None:
                c = eval_seq_cost([t])
                singleton_cost[t] = c
            if c > m:
                m = c
        return max(cur_cost, m)

    # Shared signature for prefix dominance
    def make_signature(rem_set, seq, k_suffix):
        tail = tuple(seq[-k_suffix:]) if len(seq) >= k_suffix else tuple(seq)
        return (frozenset(rem_set), tail)

    # Greedy completion with buddies + anti-buddy W filter
    def greedy_complete(prefix, remaining, params, incumbent=None):
        seq = list(prefix)
        rem = set(remaining)
        cur_cost = eval_seq_cost(seq) if seq else 0
        k_suffix = params.get('k_suffix', 3)
        recent_k = params.get('recent_k', 4)

        while rem and time_left():
            # dominance pruning on prefix
            sig = make_signature(rem, seq, k_suffix)
            best_seen = prefix_dom.get(sig)
            if best_seen is not None and cur_cost >= best_seen:
                break
            if best_seen is None or cur_cost < best_seen:
                prefix_dom[sig] = cur_cost

            if incumbent is not None and lb_singleton(cur_cost, rem) >= incumbent:
                break

            last = seq[-1] if seq else None
            rem_list = list(rem)

            # Adaptive candidate pool
            depth = len(seq)
            if depth < N // 3:
                lookahead_top = params.get('lookahead_top', 4)
                next_k = params.get('next_k', 6)
            elif depth < (2 * N) // 3:
                lookahead_top = max(2, params.get('lookahead_top', 4) - 1)
                next_k = max(4, params.get('next_k', 6) - 1)
            else:
                lookahead_top = max(2, params.get('lookahead_top', 4) - 2)
                next_k = max(3, params.get('next_k', 6) - 2)

            pool = []
            if last is not None:
                # Use buddies; filter anti-buddies by W
                for u in buddies.get(last, []):
                    if u in rem:
                        pool.append(u)
                # anti-buddy: skip strongly bad unless near-best by singleton
                bad = [u for u in rem if last in W and u in W[last] and W[last][u] > 0]
                # keep only a few worst to avoid heavy filtering
                if bad:
                    bad = sorted(bad, key=lambda u: W[last][u], reverse=True)[:max(3, len(bad) // 6)]
                    # remove bad from pool unless near-best singleton
                    pool = [u for u in pool if u not in bad or singleton_cost[u] <= sorted(singleton_cost[x] for x in rem)[min(2, len(rem) - 1)] + 1e-9]

            # add low-singletons
            low_single = sorted(rem_list, key=lambda t: singleton_cost.get(t, float('inf')))[:min(5, len(rem_list))]
            for u in low_single:
                if u not in pool:
                    pool.append(u)
            # fill random
            need = max(0, next_k * 2 - len(pool))
            if need > 0:
                others = [x for x in rem_list if x not in pool]
                if others:
                    pool.extend(rng.sample(others, min(need, len(others))))
            if not pool:
                pool = rem_list

            # score candidates by extension cost plus shallow lookahead
            pt = tuple(seq)
            scored = []
            best_immediate = float('inf')
            for t in pool:
                if not time_left():
                    break
                ec = eval_ext_cost(pt, t)
                if ec < best_immediate:
                    best_immediate = ec
                la_best = ec
                if rem and time_left():
                    new_rem = rem.copy()
                    new_rem.remove(t)
                    la_pool = [u for u in buddies.get(t, []) if u in new_rem]
                    if not la_pool:
                        la_pool = list(new_rem)
                    if len(la_pool) > lookahead_top:
                        la_pool = rng.sample(la_pool, lookahead_top)
                    new_pt = tuple(seq + [t])
                    for nxt in la_pool:
                        c2 = eval_ext_cost(new_pt, nxt)
                        if c2 < la_best:
                            la_best = c2
                scored.append((la_best, ec, t))
            if not scored:
                # time up; append remaining
                seq.extend(rem)
                cur_cost = eval_seq_cost(seq)
                return cur_cost, seq

            # anti-buddy gating: if W[last][t] strongly positive (>quartile), penalize unless ec within 1% of best
            if last is not None and last in W:
                row = W[last]
                if row:
                    vals = list(row.values())
                    if vals:
                        thresh = sorted(vals)[int(0.75 * (len(vals) - 1))] if len(vals) > 1 else max(vals)
                    else:
                        thresh = None
                    best_ec = min(x[1] for x in scored)
                    scored2 = []
                    for la, ec, t in scored:
                        if t in row and row[t] > (thresh if thresh is not None else 0) and ec > 1.01 * best_ec:
                            # penalize
                            la = la + abs(row[t]) * 0.25
                        scored2.append((la, ec, t))
                    scored = scored2

            scored.sort(key=lambda x: (x[0], x[1]))
            chosen_la, chosen_ec, chosen_t = scored[0]

            # place chosen
            seq.append(chosen_t)
            rem.remove(chosen_t)
            cur_cost = chosen_ec

        if rem:
            seq.extend(list(rem))
            cur_cost = eval_seq_cost(seq)
        return cur_cost, seq

    # Beam seed constructor with prefix-dominance reuse and incumbent-aware greedy child pruning
    def beam_seed(initial_hint, params, incumbent):
        beam = []
        beam_w = params['beam']
        cand_per_expand = params['cand_per_expand']
        k_suffix = params.get('k_suffix', 3)

        # Seed from initial hint and top singleton variants
        seeds = []
        if initial_hint:
            t0 = initial_hint[0]
            seq = [t0]
            rem = set(all_txns) - {t0}
            c = eval_seq_cost(seq)
            seeds.append((c, seq, rem))
        # add more seeds from top singletons
        seeds2 = singles_sorted[:max(beam_w, 6)]
        for t in seeds2:
            seq = [t]
            rem = set(all_txns) - {t}
            c = eval_seq_cost(seq)
            seeds.append((c, seq, rem))
        # dedup by prefix
        seen = set()
        uniq = []
        for c, s, r in seeds:
            key = tuple(s)
            if key in seen:
                continue
            seen.add(key)
            # dominance store
            sig = make_signature(r, s, k_suffix)
            prev = prefix_dom.get(sig)
            if prev is None or c < prev:
                prefix_dom[sig] = c
            uniq.append((c, s, r))
        uniq.sort(key=lambda x: x[0])
        beam = uniq[:beam_w] if uniq else []

        inc_cost = incumbent if incumbent is not None else float('inf')
        inc_seq = None

        # Try to tighten incumbent from the best seed
        if beam and time_left():
            cc, ss = greedy_complete(beam[0][1], beam[0][2], params, inc_cost)
            if len(ss) == N and cc < inc_cost:
                inc_cost, inc_seq = cc, ss

        depth = 1
        while beam and depth < N and time_left():
            new_beam = []
            for cost_so_far, seq, rem in beam:
                if not rem:
                    # complete
                    if cost_so_far < inc_cost:
                        inc_cost, inc_seq = cost_so_far, seq[:]
                    continue
                # prune by incumbent and dominance
                if cost_so_far >= inc_cost:
                    continue
                if lb_singleton(cost_so_far, rem) >= inc_cost:
                    continue
                sig_cur = make_signature(rem, seq, k_suffix)
                prev = prefix_dom.get(sig_cur)
                if prev is not None and cost_so_far >= prev:
                    continue
                if prev is None or cost_so_far < prev:
                    prefix_dom[sig_cur] = cost_so_far

                # Candidate pool around W buddies and near the hinted order
                last = seq[-1]
                rem_list = list(rem)
                pool = []
                # buddies of last
                for u in buddies.get(last, []):
                    if u in rem and u not in pool:
                        pool.append(u)
                # take next by hint order
                if initial_hint:
                    for u in initial_hint:
                        if u in rem and u not in pool:
                            pool.append(u)
                            if len(pool) >= cand_per_expand * 2:
                                break
                # fill with low singleton and random
                if len(pool) < cand_per_expand * 2:
                    low = sorted(rem_list, key=lambda t: singleton_cost.get(t, float('inf')))[:min(6, len(rem_list))]
                    for u in low:
                        if u not in pool:
                            pool.append(u)
                if len(pool) < cand_per_expand * 2:
                    others = [x for x in rem_list if x not in pool]
                    if others:
                        pool.extend(rng.sample(others, min(cand_per_expand * 2 - len(pool), len(others))))

                # Score pool by extension and shallow lookahead
                pt = tuple(seq)
                scored = []
                for t in pool:
                    if not time_left():
                        break
                    ec = eval_ext_cost(pt, t)
                    if ec >= inc_cost:
                        continue
                    # child lb
                    nrem = rem.copy()
                    nrem.remove(t)
                    lb = lb_singleton(ec, nrem)
                    if lb >= inc_cost:
                        continue
                    # anti-buddy filter: skip if strongly disfavored by W vs best ec
                    row = W.get(last, {})
                    la_best = ec
                    if nrem and time_left():
                        la_pool = [u for u in buddies.get(t, []) if u in nrem] or list(nrem)
                        la_pool = la_pool[:params.get('lookahead_top', 4)]
                        npt = tuple(seq + [t])
                        for nxt in la_pool:
                            c2 = eval_ext_cost(npt, nxt)
                            if c2 < la_best:
                                la_best = c2
                    scored.append(((la_best, ec - cost_so_far), ec, t, nrem))
                if not scored:
                    continue
                scored.sort(key=lambda x: (x[0][0], x[0][1]))
                top = scored[:min(cand_per_expand, len(scored))]

                # For top-K children, run greedy completion to update/prune by incumbent
                Kprobe = params.get('probe_k', 3)
                kept_any = False
                for idx, (_rank, ec, t, nrem) in enumerate(top):
                    child_seq = seq + [t]
                    # dominance check for child
                    sig_ch = make_signature(nrem, child_seq, k_suffix)
                    prev = prefix_dom.get(sig_ch)
                    if prev is not None and ec >= prev:
                        continue
                    # Probe selectively
                    if idx < Kprobe and time_left():
                        gc, gs = greedy_complete(child_seq, nrem, params, inc_cost)
                        if len(gs) == N and gc < inc_cost:
                            inc_cost, inc_seq = gc, gs
                        # If probe already worse than incumbent, skip adding
                        if gc >= inc_cost:
                            continue
                        rank_score = min(ec, gc)
                    else:
                        rank_score = ec
                    # record dominance
                    if prev is None or ec < prev:
                        prefix_dom[sig_ch] = ec
                    new_beam.append((ec, child_seq, nrem, rank_score))
                    kept_any = True

                if not kept_any:
                    # As a safety, keep the single best child to avoid beam starvation
                    (_rank, ec, t, nrem) = scored[0]
                    child_seq = seq + [t]
                    sig_ch = make_signature(nrem, child_seq, k_suffix)
                    prev = prefix_dom.get(sig_ch)
                    if prev is None or ec < prev:
                        prefix_dom[sig_ch] = ec
                    new_beam.append((ec, child_seq, nrem, ec))

            if not new_beam:
                break
            # select next beam by rank_score
            new_beam.sort(key=lambda x: x[3])
            unique = []
            seen = set()
            for ec, s, r, rank in new_beam:
                key = tuple(s)
                if key in seen:
                    continue
                seen.add(key)
                unique.append((ec, s, r))
                if len(unique) >= beam_w:
                    break
            beam = unique
            depth += 1

        # Finalize by greedy completion of each remaining prefix
        best_cost = inc_cost
        best_seq = inc_seq
        for c_pref, s_pref, r_pref in beam:
            if not time_left():
                break
            c_fin, s_fin = greedy_complete(s_pref, r_pref, params, best_cost)
            if len(s_fin) == N and c_fin < best_cost:
                best_cost, best_seq = c_fin, s_fin

        if best_seq is None:
            seq = all_txns[:]
            rng.shuffle(seq)
            best_seq = seq
            best_cost = eval_seq_cost(seq)
        return best_cost, best_seq

    # ΔW surrogate utilities for LNS neighborhoods
    def edge_penalty(a, b):
        # Positive W[a][b] penalizes placing a before b
        return max(0.0, W.get(a, {}).get(b, 0.0))

    def seq_surrogate_cost(seq):
        # Sum of penalties on adjacent edges (cheap)
        s = 0.0
        for i in range(len(seq) - 1):
            s += edge_penalty(seq[i], seq[i + 1])
        return s

    # LNS neighborhoods with ΔW gating and ranked evaluation
    def lns_improve(seq, cur_cost, params):
        best_seq = seq[:]
        best_cost = cur_cost
        n = len(best_seq)
        if n < 6 or not time_left():
            return best_cost, best_seq

        # Identify hot adjacencies by W
        def hot_indices(k=4):
            idx = []
            for i in range(n - 1):
                idx.append((edge_penalty(best_seq[i], best_seq[i + 1]), i))
            idx.sort(reverse=True, key=lambda x: x[0])
            return [i for _, i in idx[:min(k, len(idx))]]

        # Allow limited number of true evaluations
        eval_cap = params.get('lns_eval_cap', 700)
        eval_used = 0

        # Window permutations around hot indices
        for center in hot_indices(k=max(4, n // 20)):
            if not time_left():
                break
            w = min(6, max(4, n // 30))
            start = max(0, min(center - w // 2, n - w))
            window = best_seq[start:start + w]
            remain_prefix = best_seq[:start]
            remain_suffix = best_seq[start + w:]
            # Generate a few permutations biased by ΔW
            candidates = []

            # Try simple reversals and rotations
            candidates.append(window[::-1])
            if w >= 5:
                candidates.append(window[1:] + window[:1])
                candidates.append(window[-1:] + window[:-1])

            # Pick two random shuffles
            for _ in range(2):
                ww = window[:]
                rng.shuffle(ww)
                candidates.append(ww)

            # Rank by surrogate: build full seq and compute ΔW cost
            scored = []
            for cand_win in candidates:
                cand = remain_prefix + cand_win + remain_suffix
                surr = seq_surrogate_cost(cand)
                scored.append((surr, cand))
            scored.sort(key=lambda x: x[0])
            # Evaluate only top fraction plus a random one
            take = max(1, int(0.4 * len(scored)))
            eval_set = scored[:take]
            if len(scored) > take:
                eval_set.append(rng.choice(scored[take:]))

            for _surr, cand in eval_set:
                if not time_left() or eval_used >= eval_cap:
                    break
                c = eval_seq_cost(cand)
                eval_used += 1
                if c < best_cost:
                    best_cost, best_seq = c, cand[:]

        # Block swap between top-2 hot blocks
        hot = hot_indices(k=6)
        if len(hot) >= 2 and time_left():
            block = min(6, max(3, n // 40))
            pairs = []
            for i in range(len(hot)):
                for j in range(i + 1, len(hot)):
                    si = max(0, min(hot[i] - block // 2, n - block))
                    sj = max(0, min(hot[j] - block // 2, n - block))
                    if abs(si - sj) < block:
                        continue
                    pairs.append((si, sj))
            rng.shuffle(pairs)
            pairs = pairs[:3]
            for si, sj in pairs:
                if not time_left() or eval_used >= eval_cap:
                    break
                cand = best_seq[:]
                if si > sj:
                    si, sj = sj, si
                A = cand[si:si + block]
                B = cand[sj:sj + block]
                mid = cand[si + block:sj]
                cand2 = cand[:si] + B + mid + A + cand[sj + block:]
                # ΔW gating
                if seq_surrogate_cost(cand2) > seq_surrogate_cost(best_seq) * 1.05:
                    continue
                c = eval_seq_cost(cand2)
                eval_used += 1
                if c < best_cost:
                    best_cost, best_seq = c, cand2[:]

        # Block reinsert: remove hot block and reinsert at top-ΔW positions
        if time_left():
            center = hot[0] if hot else n // 2
            b = min(5, max(3, n // 50))
            s = max(0, min(center - b // 2, n - b))
            block_items = best_seq[s:s + b]
            remain = best_seq[:s] + best_seq[s + b:]
            positions = list(range(0, len(remain) + 1))
            # Score positions by ΔW by checking adjacencies around insertion
            pos_scored = []
            for p in positions:
                # Build a small local context to approximate ΔW delta
                left = remain[p - 1] if p - 1 >= 0 else None
                right = remain[p] if p < len(remain) else None
                penalty = 0.0
                if left is not None:
                    penalty += edge_penalty(left, block_items[0])
                for i in range(len(block_items) - 1):
                    penalty += edge_penalty(block_items[i], block_items[i + 1])
                if right is not None:
                    penalty += edge_penalty(block_items[-1], right)
                pos_scored.append((penalty, p))
            pos_scored.sort(key=lambda x: x[0])
            top_positions = [p for _sc, p in pos_scored[:max(3, len(pos_scored) // 6)]]
            # Evaluate only a few
            for p in top_positions:
                if not time_left() or eval_used >= eval_cap:
                    break
                cand = remain[:]
                for i, x in enumerate(block_items):
                    cand.insert(p + i, x)
                c = eval_seq_cost(cand)
                eval_used += 1
                if c < best_cost:
                    best_cost, best_seq = c, cand[:]

        return best_cost, best_seq

    # Portfolio parameters (deterministic restarts with shared caches)
    portfolios = [
        {'beam': 16, 'cand_per_expand': 12, 'lookahead_top': 4, 'probe_k': 3, 'recent_k': 5, 'k_suffix': 3, 'lns_eval_cap': 700},
        {'beam': 12, 'cand_per_expand': 14, 'lookahead_top': 3, 'probe_k': 3, 'recent_k': 4, 'k_suffix': 3, 'lns_eval_cap': 700},
        {'beam': 10, 'cand_per_expand': 10, 'lookahead_top': 3, 'probe_k': 2, 'recent_k': 3, 'k_suffix': 4, 'lns_eval_cap': 600},
    ]

    global_best_cost = float('inf')
    global_best_seq = None

    # Deterministic restart portfolio
    max_restarts = min(len(portfolios), max(2, int(num_seqs)))
    # Prepare a couple of initial hints from tournament order and singleton order
    hints = [seed_order, singles_sorted[:], list(range(N))]
    for r in range(max_restarts):
        if not time_left():
            break
        params = portfolios[r]
        hint = hints[min(r, len(hints) - 1)]

        # Build seed with beam + greedy probes
        cA, sA = beam_seed(hint, params, global_best_cost)

        # Selective LNS on this seed
        if time_left():
            cA, sA = lns_improve(sA, cA, params)

        # Optional second seed: greedy completion from first two of hint
        if time_left() and len(hint) >= 2:
            cB, sB = greedy_complete(hint[:2], set(all_txns) - set(hint[:2]), params, incumbent=cA)
            if cB < cA and time_left():
                cB, sB = lns_improve(sB, cB, params)
            if cB < cA:
                cA, sA = cB, sB

        if cA < global_best_cost:
            global_best_cost, global_best_seq = cA, sA

    # Fallback safety: ensure a valid permutation
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