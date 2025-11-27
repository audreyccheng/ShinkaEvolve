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
    Beam+LNS portfolio with global prefix-dominance, conflict-aware greedy,
    depth-adaptive lookahead and anti-buddy filtering to minimize makespan.

    Args:
        workload: Workload object containing transaction data
        num_seqs: Number of random restarts (used as intensity; actual restarts are time-bounded)

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    N = workload.num_txns
    # Budget tuned for combined score; larger N gets a bit more time
    time_budget_sec = 0.58 if N >= 90 else 0.50
    start_time = time.time()

    def time_left():
        return (time.time() - start_time) < time_budget_sec

    rng = random.Random(1729 + 31 * N)

    # Shared caches across all restarts
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

    # Precompute singleton costs
    singleton_cost = {}
    for t in all_txns:
        if not time_left():
            break
        singleton_cost[t] = eval_seq_cost([t])
    singles_sorted = sorted(all_txns, key=lambda t: singleton_cost.get(t, float('inf')))

    # Sampled pairwise preferences W[a][b] = cost([a,b]) - cost([b,a])
    W = defaultdict(dict)
    abs_edges = []
    max_pair_probes = min(1200, max(600, N * 10))
    probes_done = 0

    # Build a candidate pool biased by low singletons and diversified by random
    pool_bias = singles_sorted[:min(N, max(16, N // 3))]
    pool_bias += rng.sample(all_txns, min(N, max(16, N // 3))) if N > 0 else []
    pool_bias = list(dict.fromkeys(pool_bias))

    for a in all_txns:
        if not time_left():
            break
        # choose a few peers
        peers = [x for x in pool_bias if x != a]
        rng.shuffle(peers)
        peers = peers[:min(12, max(8, N // 10))]
        for b in peers:
            if probes_done >= max_pair_probes or not time_left():
                break
            if b in W[a]:
                continue
            cab = eval_seq_cost([a, b])
            cba = eval_seq_cost([b, a])
            d = cab - cba
            W[a][b] = d
            W[b][a] = -d
            abs_edges.append(abs(d))
            probes_done += 2

    # Anti-buddy threshold per last: top quartile of positive costs
    anti_buddy_thresh = {}
    for t in all_txns:
        row = W.get(t, {})
        pos_vals = [v for v in row.values() if v > 0]
        if not pos_vals:
            anti_buddy_thresh[t] = float('inf')  # no filtering
        else:
            pos_vals.sort()
            # 75th percentile
            idx = int(0.75 * (len(pos_vals) - 1))
            anti_buddy_thresh[t] = pos_vals[idx]

    # Build buddies: top neighbors minimizing delta cost cost([t,u]) - cost([t])
    def build_buddies(max_buddies=8, sample_per_t=16):
        buds = {t: [] for t in all_txns}
        for t in all_txns:
            if not time_left():
                break
            base = singleton_cost[t]
            cand_pool = singles_sorted[:min(20, max(6, N // 6))]
            others = [u for u in all_txns if u != t and u not in cand_pool]
            if others:
                cand_pool = cand_pool + rng.sample(others, min(sample_per_t, len(others)))
            seen = set()
            scored = []
            for u in cand_pool:
                if u == t or u in seen:
                    continue
                seen.add(u)
                ec = eval_ext_cost((t,), u)  # cost([t,u])
                scored.append((ec - base, u))
            scored.sort(key=lambda x: x[0])
            buds[t] = [u for _d, u in scored[:max_buddies]]
        return buds

    buddies = build_buddies(max_buddies=8, sample_per_t=min(16, max(8, N // 10)))

    # Global prefix-dominance: (frozenset(remaining), suffix tuple) -> best prefix cost
    prefix_dom = {}

    def dom_sig(rem_set, seq, k=3):
        tail = tuple(seq[-k:]) if len(seq) >= k else tuple(seq)
        return (frozenset(rem_set), tail)

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

    # Anti-buddy gate
    def is_antibuddy(last, cand):
        if last is None:
            return False
        row = W.get(last, {})
        v = row.get(cand, 0.0)
        thr = anti_buddy_thresh.get(last, float('inf'))
        return v > 0 and v >= thr

    # Conflict-aware greedy completion with anti-buddy filtering
    def greedy_complete(seq, rem_set, branch_k=12, incumbent=None, depth_ctx=0):
        seq_out = list(seq)
        rem = set(rem_set)
        cur_cost = eval_seq_cost(seq_out) if seq_out else 0.0
        steps = 0
        while rem and time_left():
            steps += 1
            # incumbent-based lower bound prune
            if incumbent is not None and lb_singleton(cur_cost, rem) >= incumbent:
                break
            last = seq_out[-1] if seq_out else None
            rem_list = list(rem)

            cand_pool = []
            # buddies of last
            if last is not None:
                for u in buddies.get(last, []):
                    if u in rem:
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
            best_t = None
            best_c = float('inf')
            # Evaluate immediate ext costs
            scored = []
            for t in cand_pool:
                c = eval_ext_cost(pt, t)
                scored.append((c, t))
            scored.sort(key=lambda x: x[0])
            # Anti-buddy filter: skip strongly disfavored unless within 1%
            filtered = []
            if last is not None:
                if scored:
                    thresh = scored[0][0] * 1.01
                else:
                    thresh = float('inf')
                for c, t in scored:
                    if is_antibuddy(last, t) and c > thresh:
                        continue
                    filtered.append((c, t))
            else:
                filtered = scored

            if not filtered:
                filtered = scored

            best_c, best_t = filtered[0]
            seq_out.append(best_t)
            rem.remove(best_t)
            cur_cost = best_c

            # Occasionally hoist prefix dominance and try completion to update incumbent
            if steps % 9 == 0:
                s = dom_sig(rem, seq_out, k=3)
                prev = prefix_dom.get(s)
                if prev is None or cur_cost < prev:
                    prefix_dom[s] = cur_cost

        if rem:
            seq_out.extend(list(rem))
            cur_cost = eval_seq_cost(seq_out)
        return cur_cost, seq_out

    # Beam builder with prefix-dominance pruning and incumbent-aware greedy probes
    def beam_seed(params, incumbent=float('inf')):
        beam_width = params['beam']
        branch_factor = params['branch']
        k_suffix = params.get('k_suffix', 3)

        # Depth-adaptive lookahead parameters
        def depth_params(depth):
            if depth < int(0.4 * N):
                return params['lookahead_top'], params['next_k'], max(3, k_suffix)
            elif depth < int(0.75 * N):
                return max(2, params['lookahead_top'] - 1), max(3, params['next_k'] - 1), max(2, k_suffix - 1)
            else:
                return max(2, params['lookahead_top'] - 2), max(2, params['next_k'] - 1), max(2, k_suffix - 1)

        # Initialize with best singletons and a few randoms
        seed_pool = singles_sorted[:max(beam_width * 2, 8)]
        extra = min(6, max(0, N - len(seed_pool)))
        if extra > 0:
            add = [x for x in all_txns if x not in seed_pool]
            if add:
                seed_pool.extend(rng.sample(add, min(extra, len(add))))

        beam = []
        for t in seed_pool:
            if not time_left():
                break
            seq = [t]
            rem = set(all_txns)
            rem.remove(t)
            c = eval_seq_cost(seq)
            beam.append((c, seq, rem))
        if not beam:
            seq = all_txns[:]
            rng.shuffle(seq)
            return eval_seq_cost(seq), seq

        beam.sort(key=lambda x: x[0])
        beam = beam[:beam_width]

        best_full_cost = incumbent
        best_full_seq = None

        steps = N - 1
        depth = 0
        while depth < steps and time_left():
            lookahead_top, next_k, k_cur = depth_params(depth)
            depth += 1
            new_beam = []

            for cost_so_far, seq, rem in beam:
                if not rem:
                    if cost_so_far < best_full_cost:
                        best_full_cost, best_full_seq = cost_so_far, seq[:]
                    continue

                # incumbent pruning
                if cost_so_far >= best_full_cost:
                    continue

                # prefix dominance
                sig = dom_sig(rem, seq, k=k_cur)
                prev = prefix_dom.get(sig)
                if prev is not None and cost_so_far >= prev:
                    continue
                # update dom
                prefix_dom[sig] = cost_so_far if prev is None else min(prev, cost_so_far)

                last = seq[-1]
                rem_list = list(rem)

                # Candidate pool: buddies of last + random sample
                cand_pool = []
                for u in buddies.get(last, []):
                    if u in rem:
                        cand_pool.append(u)
                need = max(0, branch_factor * 2 - len(cand_pool))
                if need > 0:
                    others = [x for x in rem_list if x not in cand_pool]
                    if others:
                        cand_pool.extend(rng.sample(others, min(need, len(others))))
                if not cand_pool:
                    cand_pool = rem_list if len(rem_list) <= branch_factor * 2 else rng.sample(rem_list, branch_factor * 2)

                # Score immediate and shallow lookahead with anti-buddy filter
                pt = tuple(seq)
                scored = []
                # First pass to get best immediate for anti-buddy tolerance
                best_immediate = float('inf')
                tmp = []
                for cand in cand_pool:
                    if not time_left():
                        break
                    ec = eval_ext_cost(pt, cand)
                    tmp.append((ec, cand))
                    if ec < best_immediate:
                        best_immediate = ec

                # Apply anti-buddy gating
                for ec, cand in tmp:
                    if is_antibuddy(last, cand) and ec > best_immediate * 1.01:
                        continue
                    # Lookahead candidates from buddies of cand or random few
                    la = ec
                    if rem and time_left():
                        new_rem = rem.copy()
                        new_rem.remove(cand)
                        la_pool = [v for v in buddies.get(cand, []) if v in new_rem]
                        if not la_pool:
                            la_pool = list(new_rem)
                        if len(la_pool) > lookahead_top:
                            la_pool = rng.sample(la_pool, lookahead_top)
                        new_pt = tuple(seq + [cand])
                        for nxt in la_pool:
                            c2 = eval_ext_cost(new_pt, nxt)
                            if c2 < la:
                                la = c2
                    scored.append((ec, la, cand))

                if not scored:
                    continue
                scored.sort(key=lambda x: (x[0], x[1]))
                top = scored[:min(branch_factor, len(scored))]

                # Expand children; greedy probe top next_k
                kept_any = False
                for idx, (ec, la, cand) in enumerate(top):
                    new_seq = seq + [cand]
                    new_rem = rem.copy()
                    new_rem.remove(cand)

                    # LB prune
                    if lb_singleton(ec, new_rem) >= best_full_cost:
                        continue

                    # Greedy probe for first next_k children
                    adj_la = la
                    if idx < next_k and time_left():
                        g_cost, g_seq = greedy_complete(new_seq, new_rem, branch_k=max(6, N // 12), incumbent=best_full_cost, depth_ctx=depth)
                        if len(g_seq) == N and g_cost < best_full_cost:
                            best_full_cost, best_full_seq = g_cost, g_seq
                        adj_la = min(adj_la, g_cost)

                        # Prune child if probe is already worse than incumbent
                        if g_cost >= best_full_cost:
                            # still consider at least one candidate overall
                            pass

                    new_beam.append((ec, new_seq, new_rem, adj_la))
                    kept_any = True

                # Always ensure at least the single best child is kept
                if not kept_any:
                    ec, la, cand = min(scored, key=lambda x: (x[0], x[1]))
                    new_seq = seq + [cand]
                    new_rem = rem.copy()
                    new_rem.remove(cand)
                    new_beam.append((ec, new_seq, new_rem, la))

            if not new_beam:
                break

            # Select next beam by adjusted lookahead score
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

            # Periodically greedily complete top-2 prefixes to tighten incumbent
            if beam and time_left():
                for ec, s, r in beam[:min(2, len(beam))]:
                    c_try, s_try = greedy_complete(s, r, branch_k=max(8, N // 10), incumbent=best_full_cost, depth_ctx=depth)
                    if len(s_try) == N and c_try < best_full_cost:
                        best_full_cost, best_full_seq = c_try, s_try

        # Finalize remaining prefixes greedily
        for ec, s, r in beam:
            if not time_left():
                break
            c_fin, s_fin = greedy_complete(s, r, branch_k=max(8, N // 10), incumbent=best_full_cost)
            if len(s_fin) == N and c_fin < best_full_cost:
                best_full_cost, best_full_seq = c_fin, s_fin

        if best_full_seq is None:
            seq = all_txns[:]
            rng.shuffle(seq)
            best_full_seq = seq
            best_full_cost = eval_seq_cost(seq)
        return best_full_cost, best_full_seq

    # ΔW surrogate helpers for LNS
    def pref(a, b):
        return W.get(a, {}).get(b, 0.0)

    def worst_adjacencies(seq, topk=3):
        viols = []
        for i in range(len(seq) - 1):
            a, b = seq[i], seq[i + 1]
            m = pref(a, b)  # positive: prefer b before a -> bad adjacency
            viols.append((m, i))
        viols.sort(key=lambda x: -x[0])
        return viols[:topk]

    def deltaW_for_insert(seq, block, pos, recent_k=3):
        # Surrogate score for inserting entire block at pos using boundaries with up to recent_k context
        n = len(seq)
        left_idx = pos - 1
        right_idx = pos
        score = 0.0
        # left boundary: seq[left] -> block[0]
        if 0 <= left_idx < n:
            left = seq[left_idx]
            score += pref(left, block[0])
            # context left-2 ... left-(recent_k)
            for k in range(2, recent_k + 1):
                if left_idx - (k - 1) >= 0:
                    ctx = seq[left_idx - (k - 1)]
                    score += 0.25 * pref(ctx, block[0])
        # right boundary: block[-1] -> seq[right]
        if 0 <= right_idx < n:
            right = seq[right_idx]
            score += pref(block[-1], right)
            for k in range(2, recent_k + 1):
                if right_idx + (k - 1) < n:
                    ctx = seq[right_idx + (k - 1)]
                    score += 0.25 * pref(block[-1], ctx)
        # intra-block is ignored (assumed fixed)
        return score

    # LNS refinement: block-swap and block-reinsert neighborhoods
    def lns_improve(seq, cur_cost, eval_cap=720):
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
                if evals >= eval_cap or not time_left():
                    break
            if not improved or evals >= eval_cap or not time_left():
                break

        if evals >= eval_cap or not time_left():
            return best_cost, best_seq

        # Identify two worst adjacencies
        viols = worst_adjacencies(best_seq, topk=2)
        centers = [i for _, i in viols] if viols else [n // 3, 2 * n // 3]

        # Block-swap between two blocks centered on worst adjacencies
        for a_idx in range(len(centers)):
            for b_idx in range(a_idx + 1, len(centers)):
                if evals >= eval_cap or not time_left():
                    break
                i = centers[a_idx]
                j = centers[b_idx]
                block = min(6, max(3, n // 30))
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
                c = eval_seq_cost(cand2)
                evals += 1
                if c < best_cost:
                    best_cost = c
                    best_seq = cand2
                if evals >= eval_cap or not time_left():
                    break

        if evals >= eval_cap or not time_left():
            return best_cost, best_seq

        # Block reinsert around a hot region
        block_size = min(6, max(4, n // 28))
        # Choose center: worst boundary or middle
        center = centers[0] if centers else n // 2
        start = max(0, min(center - block_size // 2, n - block_size))
        block = best_seq[start:start + block_size]
        remain = best_seq[:start] + best_seq[start + block_size:]

        # Rank positions by ΔW surrogate
        pos_candidates = list(range(0, len(remain) + 1))
        scored_pos = []
        for p in pos_candidates:
            scored_pos.append((deltaW_for_insert(remain, block, p, recent_k=3), p))
        scored_pos.sort(key=lambda x: -x[0])  # higher ΔW is worse; try fixing first
        k_eval = max(3, int(0.4 * len(scored_pos)))
        top_positions = [p for _s, p in scored_pos[:k_eval]]

        # Add ~10% random positions for diversity
        extra = max(1, len(scored_pos) // 10)
        rand_positions = rng.sample(pos_candidates, min(extra, len(pos_candidates)))
        for p in rand_positions:
            if p not in top_positions:
                top_positions.append(p)

        best_local = best_cost
        best_local_seq = None
        for p in top_positions:
            if evals >= eval_cap or not time_left():
                break
            cand = remain[:]
            for off, x in enumerate(block):
                cand.insert(p + off, x)
            c = eval_seq_cost(cand)
            evals += 1
            if c < best_local:
                best_local = c
                best_local_seq = cand
        if best_local_seq is not None and best_local < best_cost:
            best_cost, best_seq = best_local, best_local_seq

        # Final cheap adjacent pass
        if time_left() and evals < eval_cap:
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
                if evals >= eval_cap:
                    break

        return best_cost, best_seq

    # Portfolio parameter settings (deterministic)
    portfolios = [
        {'beam': 16, 'branch': 12, 'lookahead_top': 4, 'next_k': 6, 'k_suffix': 3},
        {'beam': 12, 'branch': 14, 'lookahead_top': 3, 'next_k': 5, 'k_suffix': 3},
        {'beam': 10, 'branch': 10, 'lookahead_top': 3, 'next_k': 4, 'k_suffix': 3},
    ]

    global_best_cost = float('inf')
    global_best_seq = None

    # Evaluate multiple restarts within the budget
    max_restarts = min(len(portfolios), max(2, int(num_seqs)))
    candidates = []
    for r in range(max_restarts):
        if not time_left():
            break
        params = portfolios[r % len(portfolios)]
        c_seed, s_seed = beam_seed(params, incumbent=global_best_cost)
        candidates.append((c_seed, s_seed))
        if c_seed < global_best_cost:
            global_best_cost, global_best_seq = c_seed, s_seed

    # Apply LNS only to top-2 candidates to save time
    candidates.sort(key=lambda x: x[0])
    for i, (c0, s0) in enumerate(candidates[:min(2, len(candidates))]):
        if not time_left():
            break
        c1, s1 = lns_improve(s0, c0, eval_cap=720 if N >= 90 else 600)
        if c1 < global_best_cost:
            global_best_cost, global_best_seq = c1, s1

    # Safety: ensure validity
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