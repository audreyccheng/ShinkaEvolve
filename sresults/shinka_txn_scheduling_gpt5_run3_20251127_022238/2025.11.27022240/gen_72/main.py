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
    Incumbent-aware beam search with global prefix dominance, anti-buddy filtering,
    depth-adaptive lookahead, and ΔW-gated LNS refinement to minimize makespan.

    Args:
        workload: Workload object containing transaction data
        num_seqs: Number of restarts (used to choose portfolio size; also time bounded)

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    N = workload.num_txns
    # Deterministic base RNG for reproducibility across runs
    base_seed = 1729 + 31 * N
    start_time = time.time()
    # Time budget tuned for combined score across three workloads
    time_budget_sec = 0.58 if N >= 90 else 0.50

    def time_left():
        return (time.time() - start_time) < time_budget_sec

    # Global shared caches (across restarts)
    cost_cache = {}
    ext_cache = {}
    # Prefix-dominance across entire search: (frozenset(rem), tail[-k:]) -> best cost
    prefix_dom = {}

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

    # Sampled pairwise preferences W[a][b] = cost([a,b]) - cost([b,a])
    # and pair-extension delta D[a][b] = cost([a,b]) - cost([a])
    W = defaultdict(dict)
    D = defaultdict(dict)
    abs_edges = []

    def sample_pairwise(rng, max_pair_probes):
        probes = 0
        # Candidate pool biased by low singletons and random
        bias = singles_sorted[:min(N, max(16, N // 3))]
        if N > 0:
            bias += rng.sample(all_txns, min(N, max(16, N // 3)))
        bias = list(dict.fromkeys(bias))
        for a in all_txns:
            if not time_left():
                break
            peers = [x for x in bias if x != a]
            rng.shuffle(peers)
            peers = peers[:min(12, max(8, N // 10))]
            for b in peers:
                if probes >= max_pair_probes or not time_left():
                    break
                if b in W[a]:
                    continue
                cab = eval_seq_cost([a, b])
                cba = eval_seq_cost([b, a])
                wab = cab - cba
                W[a][b] = wab
                W[b][a] = -wab
                abs_edges.append(abs(wab))
                # Extension deltas
                dab = cab - singleton_cost[a]
                dba = cba - singleton_cost[b]
                D[a][b] = dab
                D[b][a] = dba
                probes += 2

    sample_pairwise(random.Random(base_seed ^ 0x9e3779b1), max_pair_probes=min(1200, max(600, N * 10)))

    # Anti-buddy threshold (per last): 75th percentile of positive W values
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

    def is_antibuddy(last, cand):
        if last is None:
            return False
        val = W.get(last, {}).get(cand, 0.0)
        thr = anti_buddy_thresh.get(last, float('inf'))
        return val > 0 and val >= thr

    # Buddies: top neighbors minimizing D[last][cand]
    def build_buddies(max_buddies=8, sample_per_t=16, rng=None):
        if rng is None:
            rng = random
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
                if u not in seen and u != t:
                    seen.add(u)
                    cand_list.append(u)
            scored = []
            for u in cand_list:
                # If D missing, compute via ext cost
                du = D.get(t, {}).get(u)
                if du is None:
                    du = eval_ext_cost((t,), u) - singleton_cost[t]
                    D[t][u] = du
                scored.append((du, u))
            scored.sort(key=lambda x: x[0])
            buds[t] = [u for _d, u in scored[:max_buddies]]
        return buds

    buddies = build_buddies(max_buddies=8, sample_per_t=min(16, max(8, N // 10)), rng=random.Random(base_seed ^ 0xC0FFEE))

    # Lower bound using max remaining singleton
    def lb_singleton(cur_cost, rem_set):
        if not rem_set:
            return cur_cost
        return max(cur_cost, max(singleton_cost[t] for t in rem_set))

    # Global incumbent shared across restarts
    incumbent_cost = float('inf')
    incumbent_seq = None

    # Greedy completion with anti-buddy filter and small lookahead
    def greedy_complete(seq, rem_set, branch_k=12, incumbent=None, rng=None):
        if rng is None:
            rng = random
        seq_out = list(seq)
        rem = set(rem_set)
        cur_cost = eval_seq_cost(seq_out) if seq_out else 0.0
        steps = 0
        while rem and time_left():
            steps += 1
            if incumbent is not None and lb_singleton(cur_cost, rem) >= incumbent:
                break
            rem_list = list(rem)
            pool = []
            last = seq_out[-1] if seq_out else None
            if last is not None:
                pool.extend([u for u in buddies.get(last, []) if u in rem])
            # add low-singletons
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
                pool = rem_list

            pt = tuple(seq_out)
            scored = []
            best_immediate = float('inf')
            tmp = []
            for t in pool:
                c = eval_ext_cost(pt, t)
                tmp.append((c, t))
                if c < best_immediate:
                    best_immediate = c
            # anti-buddy unless within 1%
            for c, t in tmp:
                if last is not None and is_antibuddy(last, t) and c > best_immediate * 1.01:
                    continue
                # shallow one-step lookahead
                la = c
                new_rem = rem.copy()
                new_rem.remove(t)
                if new_rem:
                    la_pool = [v for v in buddies.get(t, []) if v in new_rem]
                    if not la_pool:
                        la_pool = list(new_rem)
                    if len(la_pool) > 2:
                        la_pool = rng.sample(la_pool, 2)
                    new_pt = tuple(seq_out + [t])
                    for nxt in la_pool:
                        c2 = eval_ext_cost(new_pt, nxt)
                        if c2 < la:
                            la = c2
                scored.append((la, c, t))
            if not scored:
                scored = [(c, c, t) for (c, t) in tmp]
            scored.sort(key=lambda x: (x[0], x[1]))
            _, best_c, best_t = scored[0]
            seq_out.append(best_t)
            rem.remove(best_t)
            cur_cost = best_c

            # Periodically store prefix dominance to tighten later pruning
            if steps % 9 == 0:
                sig = (frozenset(rem), tuple(seq_out[-3:] if len(seq_out) >= 3 else seq_out))
                prev = prefix_dom.get(sig)
                if prev is None or cur_cost < prev:
                    prefix_dom[sig] = cur_cost

        if rem:
            seq_out.extend(list(rem))
            cur_cost = eval_seq_cost(seq_out)
        return cur_cost, seq_out

    # Helper for prefix signature
    def dom_sig(rem_set, seq, k=3):
        return (frozenset(rem_set), tuple(seq[-k:] if len(seq) >= k else seq))

    # Beam search with unified prefix-dominance and incumbent-aware greedy child probes
    def beam_seed(params, rng, incumbent_cost, incumbent_seq, prefix_dom):
        beam_width = params['beam']
        cand_per_expand = params['cand_per_expand']
        lookahead_top = params['lookahead_top']
        next_k = params['next_k']

        # Initialize beam with best singleton starters and a few randoms
        seeds = singles_sorted[:max(beam_width * 2, 8)]
        extra = [t for t in all_txns if t not in seeds]
        rng.shuffle(extra)
        seeds += extra[:min(4, len(extra))]
        beam = []
        for t in seeds:
            seq = [t]
            rem = set(all_txns)
            rem.remove(t)
            c = eval_seq_cost(seq)
            beam.append((c, seq, rem))
            # seed dominance
            s = dom_sig(rem, seq, k=3)
            prev = prefix_dom.get(s)
            if prev is None or c < prev:
                prefix_dom[s] = c
        beam.sort(key=lambda x: x[0])
        beam = beam[:beam_width]

        best_cost = incumbent_cost
        best_seq = incumbent_seq

        # Early tighten via greedy completion of best 1-2 prefixes
        for cost_so_far, seq, rem in beam[:min(2, len(beam))]:
            if not time_left():
                break
            c_try, s_try = greedy_complete(seq, rem, branch_k=max(8, cand_per_expand), incumbent=best_cost, rng=rng)
            if len(s_try) == N and c_try < best_cost:
                best_cost, best_seq = c_try, s_try

        # Depth adaptive knobs
        def depth_cfg(depth):
            if depth < int(0.4 * N):
                return (lookahead_top, next_k, 3, 5)
            elif depth < int(0.75 * N):
                return (max(2, lookahead_top - 1), max(2, next_k - 1), 3, 4)
            else:
                return (max(2, lookahead_top - 2), max(1, next_k - 1), 2, 3)

        steps = N - 1
        depth = 0
        while depth < steps and time_left():
            depth += 1
            la_top, probe_k, k_suffix, recent_k = depth_cfg(depth)
            new_beam = []

            # Periodically promote incumbent by completing top-2 prefixes
            for cost_so_far, seq, rem in beam[:min(2, len(beam))]:
                if not time_left():
                    break
                c_try, s_try = greedy_complete(seq, rem, branch_k=max(8, cand_per_expand), incumbent=best_cost, rng=rng)
                if len(s_try) == N and c_try < best_cost:
                    best_cost, best_seq = c_try, s_try

            for cost_so_far, seq, rem in beam:
                if not rem:
                    if cost_so_far < best_cost:
                        best_cost, best_seq = cost_so_far, seq[:]
                    continue

                # Prune by incumbent and LB
                if cost_so_far >= best_cost:
                    continue
                if lb_singleton(cost_so_far, rem) >= best_cost:
                    continue

                # Prefix-dominance pruning
                sig = dom_sig(rem, seq, k=k_suffix)
                prev = prefix_dom.get(sig)
                if prev is not None and cost_so_far >= prev:
                    continue
                if prev is None or cost_so_far < prev:
                    prefix_dom[sig] = cost_so_far

                rem_list = list(rem)
                # Candidate pool: buddies of last + low-singleton + random; apply anti-buddy filter later
                cand_pool = []
                last = seq[-1]
                cand_pool.extend([x for x in buddies.get(last, []) if x in rem])
                # add top low-singletons
                low_s = sorted(rem_list, key=lambda t: singleton_cost.get(t, float('inf')))[:min(5, len(rem_list))]
                for u in low_s:
                    if u not in cand_pool:
                        cand_pool.append(u)
                # add random to reach 2*cand_per_expand
                need = max(0, 2 * cand_per_expand - len(cand_pool))
                if need > 0:
                    others = [x for x in rem_list if x not in cand_pool]
                    if others:
                        cand_pool.extend(rng.sample(others, min(need, len(others))))
                if not cand_pool:
                    cand_pool = rem_list if len(rem_list) <= 2 * cand_per_expand else rng.sample(rem_list, 2 * cand_per_expand)

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

                # score with shallow lookahead; anti-buddy gate unless within 1%
                scored = []
                for ec, cand in tmp:
                    if is_antibuddy(last, cand) and ec > best_immediate * 1.01:
                        continue
                    new_rem = rem - {cand}
                    if not new_rem:
                        la = ec
                    else:
                        la_pool = [v for v in buddies.get(cand, []) if v in new_rem]
                        if not la_pool:
                            la_pool = list(new_rem)
                        if len(la_pool) > la_top:
                            la_pool = rng.sample(la_pool, la_top)
                        new_pt = tuple(seq + [cand])
                        la = ec
                        for nxt in la_pool:
                            c2 = eval_ext_cost(new_pt, nxt)
                            if c2 < la:
                                la = c2
                    # Rank tuple: (lookahead, immediate delta)
                    scored.append(((la, ec - cost_so_far), ec, cand))

                if not scored:
                    continue
                scored.sort(key=lambda x: (x[0][0], x[0][1]))
                top_children = scored[:min(cand_per_expand, len(scored))]

                # For top-K children, run greedy completion; prune if completion >= incumbent
                kept = 0
                for idx, (rk, ec, cand) in enumerate(top_children):
                    new_seq = seq + [cand]
                    new_rem = rem - {cand}

                    # Child LB prune
                    if lb_singleton(ec, new_rem) >= best_cost:
                        continue

                    rank_score = rk[0]
                    if idx < probe_k and time_left():
                        g_cost, g_seq = greedy_complete(new_seq, new_rem, branch_k=max(6, N // 12), incumbent=best_cost, rng=rng)
                        if len(g_seq) == N and g_cost < best_cost:
                            best_cost, best_seq = g_cost, g_seq
                        if g_cost >= best_cost:
                            # prune this child
                            continue
                        rank_score = min(rank_score, g_cost)

                    # Child-specific dominance
                    csig = dom_sig(new_rem, new_seq, k=k_suffix)
                    prev_c = prefix_dom.get(csig)
                    if prev_c is not None and ec >= prev_c:
                        continue
                    if prev_c is None or ec < prev_c:
                        prefix_dom[csig] = ec

                    new_beam.append((ec, new_seq, new_rem, rank_score))
                    kept += 1

                # Always keep at least the single best immediate child to avoid empty beams
                if kept == 0 and top_children:
                    _, ec, cand = top_children[0]
                    new_seq = seq + [cand]
                    new_rem = rem - {cand}
                    rank_score = ec
                    new_beam.append((ec, new_seq, new_rem, rank_score))

            if not new_beam:
                break
            # Select next beam by rank score, ensuring unique prefixes
            new_beam.sort(key=lambda x: x[3])
            unique = []
            seen = set()
            for ec, s, r, rs in new_beam:
                key = tuple(s)
                if key in seen:
                    continue
                seen.add(key)
                unique.append((ec, s, r))
                if len(unique) >= beam_width:
                    break
            beam = unique

        # Final greedy completion on surviving prefixes
        for cost_so_far, seq, rem in beam:
            if not time_left():
                break
            c_fin, s_fin = greedy_complete(seq, rem, branch_k=max(8, cand_per_expand), incumbent=best_cost, rng=rng)
            if len(s_fin) == N and c_fin < best_cost:
                best_cost, best_seq = c_fin, s_fin

        if best_seq is None:
            seq = all_txns[:]
            rng.shuffle(seq)
            best_seq = seq
            best_cost = eval_seq_cost(seq)
        return best_cost, best_seq

    # ΔW surrogate helpers for LNS
    def pref(a, b):
        return W.get(a, {}).get(b, 0.0)

    def worst_adjacencies(seq, topk=2):
        viols = []
        for i in range(len(seq) - 1):
            a, b = seq[i], seq[i + 1]
            m = pref(a, b)
            viols.append((m, i))
        viols.sort(key=lambda x: -x[0])
        return viols[:topk]

    def deltaW_block_insert(rem_seq, block, pos, recent_k=3):
        # Score boundary impact around insertion position using W with short context
        n = len(rem_seq)
        s = 0.0
        left_idx = pos - 1
        right_idx = pos
        if 0 <= left_idx < n:
            left = rem_seq[left_idx]
            s += pref(left, block[0])
            for k in range(2, recent_k + 1):
                if left_idx - (k - 1) >= 0:
                    ctx = rem_seq[left_idx - (k - 1)]
                    s += 0.25 * pref(ctx, block[0])
        if 0 <= right_idx < n:
            right = rem_seq[right_idx]
            s += pref(block[-1], right)
            for k in range(2, recent_k + 1):
                if right_idx + (k - 1) < n:
                    ctx = rem_seq[right_idx + (k - 1)]
                    s += 0.25 * pref(block[-1], ctx)
        # inner-block ignored (constant across positions)
        return s

    # LNS improvement with ΔW-gated block-swap and block-reinsert, capped evaluations
    def lns_improve(seq, cur_cost, rng, eval_cap=720):
        best_seq = seq[:]
        best_cost = cur_cost
        n = len(best_seq)
        if n < 6 or not time_left():
            return best_cost, best_seq

        evals = 0

        # Cheap adjacent bubble pass
        for _ in range(2):
            improved = False
            for i in range(n - 1):
                if not time_left():
                    break
                cand = best_seq[:]
                cand[i], cand[i + 1] = cand[i + 1], cand[i]
                c = eval_seq_cost(cand)
                evals += 1
                if c < best_cost:
                    best_cost = c
                    best_seq = cand
                    improved = True
                if evals >= eval_cap:
                    break
            if not improved or evals >= eval_cap or not time_left():
                break

        if evals >= eval_cap or not time_left():
            return best_cost, best_seq

        # Neighborhood A: block-swap between two disjoint hot blocks around worst adjacencies
        viols = worst_adjacencies(best_seq, topk=2)
        centers = [i for _, i in viols] if viols else [n // 3, 2 * n // 3]
        block_lo = 3
        block_hi = min(6, max(3, n // 20))
        cand_moves = []
        for i_idx in range(len(centers)):
            for j_idx in range(i_idx + 1, len(centers)):
                ci = centers[i_idx]
                cj = centers[j_idx]
                for bsz in range(block_lo, block_hi + 1):
                    si = max(0, min(ci - bsz // 2, n - bsz))
                    sj = max(0, min(cj - bsz // 2, n - bsz))
                    if abs(si - sj) < bsz:
                        continue
                    # Surrogate gate: sum of boundary W for both blocks (higher is worse)
                    surr = 0.0
                    if si - 1 >= 0:
                        surr += pref(best_seq[si - 1], best_seq[si])
                    if si + bsz < n:
                        surr += pref(best_seq[si + bsz - 1], best_seq[si + bsz])
                    if sj - 1 >= 0:
                        surr += pref(best_seq[sj - 1], best_seq[sj])
                    if sj + bsz < n:
                        surr += pref(best_seq[sj + bsz - 1], best_seq[sj + bsz])
                    cand_moves.append((surr, si, sj, bsz))
        # Evaluate only top 40% plus 10% random
        if cand_moves:
            cand_moves.sort(key=lambda x: -x[0])
            topk = max(1, int(0.4 * len(cand_moves)))
            eval_set = cand_moves[:topk]
            tail = cand_moves[topk:]
            if tail:
                eval_set += rng.sample(tail, min(max(1, len(cand_moves) // 10), len(tail)))
            for _s, si, sj, bsz in eval_set:
                if evals >= eval_cap or not time_left():
                    break
                cand = best_seq[:]
                if si > sj:
                    si, sj = sj, si
                block_i = cand[si:si + bsz]
                block_j = cand[sj:sj + bsz]
                mid = cand[si + bsz:sj]
                cand2 = cand[:si] + block_j + mid + block_i + cand[sj + bsz:]
                c = eval_seq_cost(cand2)
                evals += 1
                if c < best_cost:
                    best_cost = c
                    best_seq = cand2

        if evals >= eval_cap or not time_left():
            return best_cost, best_seq

        # Neighborhood B: remove a hot block and reinsert at best ΔW-ranked positions
        if n >= 6:
            viols = worst_adjacencies(best_seq, topk=1)
            center = viols[0][1] if viols else n // 2
            bsz = min(6, max(4, n // 28))
            start = max(0, min(center - bsz // 2, n - bsz))
            block = best_seq[start:start + bsz]
            remain = best_seq[:start] + best_seq[start + bsz:]
            pos_candidates = list(range(len(remain) + 1))
            scored = []
            for p in pos_candidates:
                scored.append((deltaW_block_insert(remain, block, p, recent_k=3), p))
            scored.sort(key=lambda x: -x[0])  # try worst ΔW first
            topk = max(3, int(0.4 * len(scored)))
            eval_positions = [p for _s, p in scored[:topk]]
            # +10% random
            extra = max(1, len(scored) // 10)
            eval_positions += random.sample(pos_candidates, min(extra, len(pos_candidates)))
            seenp = set()
            for p in eval_positions:
                if p in seenp or evals >= eval_cap or not time_left():
                    continue
                seenp.add(p)
                cand = remain[:]
                for k, v in enumerate(block):
                    cand.insert(p + k, v)
                c = eval_seq_cost(cand)
                evals += 1
                if c < best_cost:
                    best_cost = c
                    best_seq = cand

        # Final cheap adjacent pass
        for i in range(n - 1):
            if evals >= eval_cap or not time_left():
                break
            cand = best_seq[:]
            cand[i], cand[i + 1] = cand[i + 1], cand[i]
            c = eval_seq_cost(cand)
            evals += 1
            if c < best_cost:
                best_cost = c
                best_seq = cand

        return best_cost, best_seq

    # Restart portfolio (deterministic), shared caches and prefix_dom
    portfolios = [
        {'beam': 16, 'cand_per_expand': 12, 'lookahead_top': 4, 'next_k': 6},
        {'beam': 12, 'cand_per_expand': 14, 'lookahead_top': 3, 'next_k': 5},
        {'beam': 10, 'cand_per_expand': 10, 'lookahead_top': 3, 'next_k': 4},
    ]
    max_restarts = min(len(portfolios), max(2, int(num_seqs)))

    global_best_cost = float('inf')
    global_best_seq = None
    candidates = []

    for r in range(max_restarts):
        if not time_left():
            break
        rng = random.Random(base_seed + r * 31 + 7)
        c_seed, s_seed = beam_seed(portfolios[r], rng, incumbent_cost, incumbent_seq, prefix_dom)
        candidates.append((c_seed, s_seed))
        if c_seed < global_best_cost:
            global_best_cost, global_best_seq = c_seed, s_seed
        # Promote to incumbent for subsequent restarts
        incumbent_cost, incumbent_seq = global_best_cost, global_best_seq

    # Apply LNS to top-2 seeds only
    candidates.sort(key=lambda x: x[0])
    for i, (c0, s0) in enumerate(candidates[:min(2, len(candidates))]):
        if not time_left():
            break
        rng = random.Random(base_seed ^ (i * 131 + 911))
        cap = 800 if N >= 90 else 600
        c1, s1 = lns_improve(s0, c0, rng, eval_cap=cap)
        if c1 < global_best_cost:
            global_best_cost, global_best_seq = c1, s1

    # Safety: ensure a valid permutation
    if global_best_seq is None or len(global_best_seq) != N or len(set(global_best_seq)) != N:
        seq = list(range(N))
        random.Random(base_seed ^ 0xABCDEF).shuffle(seq)
        global_best_seq = seq
        global_best_cost = eval_seq_cost(seq)

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