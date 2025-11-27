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
    Beam + greedy + ΔW-LNS with unified prefix-dominance and anti-buddy filtering.

    Args:
        workload: Workload object containing transaction data
        num_seqs: Number of random restarts (used as an upper bound; time-bounded)

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    N = workload.num_txns
    start_time = time.time()

    # Time budget tuned for combined score
    time_budget_sec = 0.58
    rng = random.Random(1729 + 31 * N)

    def time_left():
        return (time.time() - start_time) < time_budget_sec

    # Caches for costs
    cost_cache = {}
    ext_cache = {}

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

    # Sample pairwise preferences W[a][b] = cost([a,b]) - cost([b,a])
    W = defaultdict(dict)
    abs_edges = []
    max_pair_probes = min(900, max(450, N * 9))
    probes = 0

    # Build a biased pool towards low singletons, plus random
    bias_pool = singles_sorted[:min(N, max(16, N // 3))]
    if N > 0:
        bias_pool += rng.sample(all_txns, min(N, max(16, N // 3)))
    bias_pool = list(dict.fromkeys(bias_pool))

    for a in all_txns:
        if not time_left() or probes >= max_pair_probes:
            break
        # choose a few peers
        peers = [x for x in bias_pool if x != a]
        rng.shuffle(peers)
        peers = peers[:min(12, max(8, N // 10))]
        for b in peers:
            if not time_left() or probes >= max_pair_probes:
                break
            if b in W[a]:
                continue
            cab = eval_seq_cost([a, b])
            cba = eval_seq_cost([b, a])
            d = cab - cba
            W[a][b] = d
            W[b][a] = -d
            abs_edges.append(abs(d))
            probes += 2

    # Anti-buddy thresholds: 75th percentile of positive W[a][b]
    anti_buddy_thresh = {}
    for t in all_txns:
        row = W.get(t, {})
        pos = [v for v in row.values() if v > 0]
        if not pos:
            anti_buddy_thresh[t] = float('inf')
        else:
            pos.sort()
            idx = int(0.75 * (len(pos) - 1))
            anti_buddy_thresh[t] = pos[idx]

    # Build buddies by extension delta cost([t,u]) - cost([t])
    pair_delta_cache = {}

    def pair_delta(a, b):
        key = (a, b)
        v = pair_delta_cache.get(key)
        if v is not None:
            return v
        base = singleton_cost.get(a)
        if base is None:
            base = eval_seq_cost([a])
            singleton_cost[a] = base
        ec = eval_ext_cost((a,), b)
        d = ec - base
        pair_delta_cache[key] = d
        return d

    def build_buddies(max_buddies=8, sample_per_t=14):
        buds = {t: [] for t in all_txns}
        top_slice = singles_sorted[:min(22, max(8, N // 5))]
        for t in all_txns:
            if not time_left():
                break
            pool = [u for u in top_slice if u != t]
            others = [u for u in all_txns if u != t and u not in pool]
            if others:
                pool.extend(rng.sample(others, min(sample_per_t, len(others))))
            # Dedup
            seen = set()
            cand_list = []
            for u in pool:
                if u in seen or u == t:
                    continue
                seen.add(u)
                cand_list.append(u)
            scored = []
            base = singleton_cost[t]
            for u in cand_list:
                if not time_left():
                    break
                c2 = eval_ext_cost((t,), u)
                scored.append((c2 - base, u))
            scored.sort(key=lambda x: x[0])
            buds[t] = [u for _d, u in scored[:max_buddies]]
        return buds

    buddies = build_buddies(max_buddies=8, sample_per_t=14)

    # Prefix-dominance cache shared globally
    prefix_dom = {}

    def dom_sig(rem_set, seq, k=3):
        if k <= 0:
            return (frozenset(rem_set), ())
        tail = tuple(seq[-k:]) if len(seq) >= k else tuple(seq)
        return (frozenset(rem_set), tail)

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

    def is_antibuddy(last, cand):
        if last is None:
            return False
        thr = anti_buddy_thresh.get(last, float('inf'))
        if thr == float('inf'):
            return False
        # Use W if available, else fallback to pair_delta
        v = W.get(last, {}).get(cand, None)
        if v is None:
            pd = pair_delta(last, cand)
            return pd > 0 and pd >= thr
        return v > 0 and v >= thr

    # Greedy completion guided by buddies and extension costs; uses prefix-dom and anti-buddy
    def greedy_complete(seq, rem_set, branch_k=12, incumbent=None, k_suffix=3):
        seq_out = list(seq)
        rem = set(rem_set)
        cur_cost = eval_seq_cost(seq_out) if seq_out else 0.0
        steps = 0
        while rem and time_left():
            steps += 1
            # Dominate check on current prefix
            sig_cur = dom_sig(rem, seq_out, k=k_suffix)
            prev = prefix_dom.get(sig_cur)
            if prev is not None and cur_cost >= prev - 1e-9:
                break
            if prev is None or cur_cost < prev:
                prefix_dom[sig_cur] = cur_cost

            if incumbent is not None and lb_singleton(cur_cost, rem) >= incumbent:
                break

            rem_list = list(rem)
            last = seq_out[-1] if seq_out else None
            pool = []
            if last is not None:
                pool.extend([u for u in buddies.get(last, []) if u in rem])
            # add top few low singletons
            low_single = sorted(rem_list, key=lambda t: singleton_cost.get(t, float('inf')))[:min(4, len(rem_list))]
            for u in low_single:
                if u not in pool:
                    pool.append(u)
            # fill random
            need = max(0, branch_k - len(pool))
            if need > 0:
                others = [x for x in rem_list if x not in pool]
                if others:
                    pool.extend(rng.sample(others, min(need, len(others))))
            if not pool:
                pool = rem_list if len(rem_list) <= branch_k else rng.sample(rem_list, branch_k)

            pt = tuple(seq_out)
            tmp = []
            best_immediate = float('inf')
            for t in pool:
                c = eval_ext_cost(pt, t)
                tmp.append((c, t))
                if c < best_immediate:
                    best_immediate = c
            # anti-buddy filtering with 1% tolerance
            filtered = []
            tol = best_immediate * 1.01 if best_immediate < float('inf') else float('inf')
            for c, t in tmp:
                if last is not None and is_antibuddy(last, t) and c > tol:
                    continue
                filtered.append((c, t))
            if not filtered:
                filtered = tmp

            filtered.sort(key=lambda x: x[0])
            best_c, best_t = filtered[0]
            seq_out.append(best_t)
            rem.remove(best_t)
            cur_cost = best_c

            # Periodically update prefix_dom to enable cross-pruning
            if steps % 9 == 0:
                sig_mid = dom_sig(rem, seq_out, k=k_suffix)
                prev = prefix_dom.get(sig_mid)
                if prev is None or cur_cost < prev:
                    prefix_dom[sig_mid] = cur_cost

        if rem:
            seq_fin = seq_out + list(rem)
            cur_cost = eval_seq_cost(seq_fin)
            return cur_cost, seq_fin
        return cur_cost, seq_out

    # Beam search with unified prefix-dominance, depth-adaptive lookahead, anti-buddy, and incumbent-aware greedy probes
    def run_beam(params, incumbent=float('inf')):
        beam_width = params['beam']
        branch_factor = params['branch']

        # depth-adaptive parameter function
        def depth_params(depth):
            # early / mid / late
            if depth < int(0.4 * N):
                return params['lookahead_top'], params['next_k'], 3
            elif depth < int(0.75 * N):
                return max(2, params['lookahead_top'] - 1), max(3, params['next_k'] - 1), 3
            else:
                return max(2, params['lookahead_top'] - 2), max(2, params['next_k'] - 1), 2

        # Seed beam from best singletons and a few random for diversity
        seeds = singles_sorted[:max(beam_width * 2, 8)]
        others = [x for x in all_txns if x not in seeds]
        rng.shuffle(others)
        seeds += others[:min(4, len(others))]
        beam = []
        for t in seeds:
            if not time_left():
                break
            seq = [t]
            rem = set(all_txns)
            rem.remove(t)
            c = eval_seq_cost(seq)
            beam.append((c, seq, rem))
            # record dom for singleton
            prefix_dom[dom_sig(rem, seq, k=3)] = min(prefix_dom.get(dom_sig(rem, seq, k=3), float('inf')), c)
        beam.sort(key=lambda x: x[0])
        beam = beam[:beam_width]

        best_full_cost = incumbent
        best_full_seq = None

        # Early greedy completion on top prefix
        if beam and time_left():
            c_try, s_try = greedy_complete(beam[0][1], beam[0][2], branch_k=max(8, N // 12), incumbent=best_full_cost, k_suffix=3)
            if c_try < best_full_cost:
                best_full_cost, best_full_seq = c_try, s_try

        steps = N - 1
        depth = 0
        while depth < steps and time_left():
            lookahead_top, next_k, k_suf = depth_params(depth)
            depth += 1
            new_beam = []

            # Periodically promote incumbent by greedy completing top-2 prefixes
            for cost_so_far, seq, rem in beam[:min(2, len(beam))]:
                if not time_left():
                    break
                c_try, s_try = greedy_complete(seq, rem, branch_k=max(8, N // 10), incumbent=best_full_cost, k_suffix=k_suf)
                if c_try < best_full_cost:
                    best_full_cost, best_full_seq = c_try, s_try

            for cost_so_far, seq, rem in beam:
                if not rem:
                    if cost_so_far < best_full_cost:
                        best_full_cost, best_full_seq = cost_so_far, seq[:]
                    continue

                # Incumbent prune
                if cost_so_far >= best_full_cost:
                    continue

                # Prefix-dominance prune
                sig = dom_sig(rem, seq, k=k_suf)
                prev = prefix_dom.get(sig)
                if prev is not None and cost_so_far >= prev - 1e-9:
                    continue
                if prev is None or cost_so_far < prev:
                    prefix_dom[sig] = cost_so_far

                last = seq[-1]
                rem_list = list(rem)
                cand_pool = []
                # buddies of last
                if last in buddies:
                    cand_pool.extend([u for u in buddies[last] if u in rem])
                # random fill to 2*branch
                need = max(0, 2 * branch_factor - len(cand_pool))
                if need > 0:
                    others = [x for x in rem_list if x not in cand_pool]
                    if others:
                        cand_pool.extend(rng.sample(others, min(need, len(others))))
                if not cand_pool:
                    cand_pool = rem_list if len(rem_list) <= 2 * branch_factor else rng.sample(rem_list, 2 * branch_factor)

                # Score immediate ext cost and shallow lookahead; anti-buddy filtering
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

                # Filter anti-buddies unless within 1%
                tol = best_immediate * 1.01 if best_immediate < float('inf') else float('inf')
                scored = []
                for ec, cand in tmp:
                    if ec >= best_full_cost:
                        continue
                    if last is not None and is_antibuddy(last, cand) and ec > tol:
                        continue
                    la_best = ec
                    if rem and time_left():
                        new_rem = rem.copy()
                        if cand in new_rem:
                            new_rem.remove(cand)
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
                    # ensure at least best-immediate considered
                    if tmp:
                        ec, cand = min(tmp, key=lambda x: x[0])
                        new_seq = seq + [cand]
                        new_rem = rem.copy()
                        new_rem.remove(cand)
                        new_beam.append((ec, new_seq, new_rem, ec))
                    continue

                scored.sort(key=lambda x: (x[0], x[1]))
                top = scored[:min(branch_factor, len(scored))]

                kept_any = False
                for idx, (_delta, la_score, ec, cand) in enumerate(top):
                    new_seq = seq + [cand]
                    new_rem = rem.copy()
                    new_rem.remove(cand)

                    # Child LB prune
                    if lb_singleton(ec, new_rem) >= best_full_cost:
                        continue

                    # Child dominated?
                    sig_ch = dom_sig(new_rem, new_seq, k=k_suf)
                    prev_ch = prefix_dom.get(sig_ch)
                    if prev_ch is not None and ec >= prev_ch - 1e-9:
                        continue
                    if prev_ch is None or ec < prev_ch:
                        prefix_dom[sig_ch] = ec

                    # Incumbent-aware greedy completion for first next_k children
                    adj_la = la_score
                    if idx < next_k and time_left():
                        g_cost, g_seq = greedy_complete(new_seq, new_rem, branch_k=max(6, N // 12), incumbent=best_full_cost, k_suffix=k_suf)
                        if g_cost < best_full_cost:
                            best_full_cost, best_full_seq = g_cost, g_seq
                        adj_la = min(adj_la, g_cost)
                        # Early prune if even completion ≥ incumbent (handled by adj_la heuristic implicitly)

                    new_beam.append((ec, new_seq, new_rem, adj_la))
                    kept_any = True

                if not kept_any and tmp:
                    # Guarantee at least one child
                    ec, cand = min(tmp, key=lambda x: x[0])
                    new_seq = seq + [cand]
                    new_rem = rem.copy()
                    new_rem.remove(cand)
                    new_beam.append((ec, new_seq, new_rem, ec))

            if not new_beam:
                break

            # Select next beam by adjusted lookahead score, keep unique prefixes
            new_beam.sort(key=lambda x: x[3])
            unique = []
            seen = set()
            for ec, s, r, score in new_beam:
                key = tuple(s)
                if key in seen:
                    continue
                seen.add(key)
                unique.append((ec, s, r))
                if len(unique) >= beam_width:
                    break
            beam = unique

        # Final greedy finish for remaining prefixes
        for ec, s, r in beam:
            if not time_left():
                break
            c_fin, s_fin = greedy_complete(s, r, branch_k=max(8, N // 10), incumbent=best_full_cost, k_suffix=3)
            if c_fin < best_full_cost:
                best_full_cost, best_full_seq = c_fin, s_fin

        if best_full_seq is None:
            seq = all_txns[:]
            rng.shuffle(seq)
            best_full_seq = seq
            best_full_cost = eval_seq_cost(seq)
        return best_full_cost, best_full_seq

    # ΔW-gated LNS
    def pref(a, b):
        # adjacency preference surrogate; positive means b->a preferred (a->b is bad)
        if b in W.get(a, {}):
            return W[a][b]
        # fall back to symmetric from pair_delta
        ab = eval_seq_cost([a, b])
        ba = eval_seq_cost([b, a])
        d = ab - ba
        W[a][b] = d
        W[b][a] = -d
        return d

    def worst_adjacencies(seq, topk=3):
        n = len(seq)
        viols = []
        for i in range(n - 1):
            a, b = seq[i], seq[i + 1]
            viols.append((pref(a, b), i))
        viols.sort(key=lambda x: x[0], reverse=True)
        return viols[:topk]

    def deltaW_for_insert(seq, block, pos, recent_k=3):
        # Surrogate change due to inserting block at position pos using boundaries
        n = len(seq)
        left_idx = pos - 1
        right_idx = pos
        score = 0.0
        if 0 <= left_idx < n:
            left = seq[left_idx]
            score += pref(left, block[0])
            for k in range(2, recent_k + 1):
                if left_idx - (k - 1) >= 0:
                    ctx = seq[left_idx - (k - 1)]
                    score += 0.25 * pref(ctx, block[0])
        if 0 <= right_idx < n:
            right = seq[right_idx]
            score += pref(block[-1], right)
            for k in range(2, recent_k + 1):
                if right_idx + (k - 1) < n:
                    ctx = seq[right_idx + (k - 1)]
                    score += 0.25 * pref(block[-1], ctx)
        return score

    def lns_improve(seq, cur_cost):
        best_seq = seq[:]
        best_cost = cur_cost
        n = len(best_seq)
        if n < 6 or not time_left():
            return best_cost, best_seq

        # Cheap bubble pass
        for _ in range(2):
            improved = False
            for i in range(n - 1):
                if not time_left():
                    break
                cand = best_seq[:]
                cand[i], cand[i + 1] = cand[i + 1], cand[i]
                c = eval_seq_cost(cand)
                if c < best_cost:
                    best_cost = c
                    best_seq = cand
                    improved = True
            if not improved or not time_left():
                break

        # Determine evaluation cap
        eval_cap = 800 if N >= 90 else 650
        evals = 0

        # Block-swap centered on worst adjacencies (two blocks 3–6)
        viols = worst_adjacencies(best_seq, topk=2)
        centers = [i for _, i in viols] if viols else [n // 3, 2 * n // 3]
        for a_idx in range(len(centers)):
            for b_idx in range(a_idx + 1, len(centers)):
                if evals >= eval_cap or not time_left():
                    break
                i = centers[a_idx]
                j = centers[b_idx]
                blk = min(6, max(3, n // 30))
                si = max(0, min(i - blk // 2, n - blk))
                sj = max(0, min(j - blk // 2, n - blk))
                if abs(si - sj) < blk:
                    continue
                cand = best_seq[:]
                if si > sj:
                    si, sj = sj, si
                block_i = cand[si:si + blk]
                block_j = cand[sj:sj + blk]
                mid = cand[si + blk:sj]
                cand2 = cand[:si] + block_j + mid + block_i + cand[sj + blk:]
                c = eval_seq_cost(cand2)
                evals += 1
                if c < best_cost:
                    best_cost = c
                    best_seq = cand2

        if evals >= eval_cap or not time_left():
            return best_cost, best_seq

        # Block reinsert: remove a hot block and reinsert at top positions by ΔW
        blk = min(6, max(4, n // 28))
        center = centers[0] if centers else n // 2
        start = max(0, min(center - blk // 2, n - blk))
        block = best_seq[start:start + blk]
        remain = best_seq[:start] + best_seq[start + blk:]
        pos_candidates = list(range(0, len(remain) + 1))
        scored = []
        for p in pos_candidates:
            scored.append((deltaW_for_insert(remain, block, p, recent_k=3), p))
        scored.sort(key=lambda x: -x[0])
        k_eval = max(3, int(0.4 * len(scored)))
        top_positions = [p for _s, p in scored[:k_eval]]

        # Add ~10% random positions
        extra = max(1, len(scored) // 10)
        rand_positions = rng.sample(pos_candidates, min(extra, len(pos_candidates)))
        for p in rand_positions:
            if p not in top_positions:
                top_positions.append(p)

        for p in top_positions:
            if evals >= eval_cap or not time_left():
                break
            cand = remain[:]
            for off, x in enumerate(block):
                cand.insert(p + off, x)
            c = eval_seq_cost(cand)
            evals += 1
            if c < best_cost:
                best_cost = c
                best_seq = cand

        # Final cheap adjacent pass
        if time_left():
            for i in range(n - 1):
                if not time_left():
                    break
                cand = best_seq[:]
                cand[i], cand[i + 1] = cand[i + 1], cand[i]
                c = eval_seq_cost(cand)
                if c < best_cost:
                    best_cost = c
                    best_seq = cand

        return best_cost, best_seq

    # Portfolio of parameter settings
    portfolios = [
        {'beam': 16, 'branch': 12, 'lookahead_top': 4, 'next_k': 6},
        {'beam': 12, 'branch': 14, 'lookahead_top': 3, 'next_k': 5},
        {'beam': 10, 'branch': 10, 'lookahead_top': 3, 'next_k': 4},
    ]

    global_best_cost = float('inf')
    global_best_seq = None

    # Deterministic restarts within budget
    max_restarts = max(2, min(len(portfolios), int(num_seqs)))
    seeds = []
    for r in range(max_restarts):
        if not time_left():
            break
        params = portfolios[r % len(portfolios)]
        c_s, s_s = run_beam(params, incumbent=global_best_cost)
        seeds.append((c_s, s_s))
        if c_s < global_best_cost:
            global_best_cost, global_best_seq = c_s, s_s

    # Apply LNS to the top-2 seeds
    seeds.sort(key=lambda x: x[0])
    for i, (c0, s0) in enumerate(seeds[:min(2, len(seeds))]):
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
        # repair
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