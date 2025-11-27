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
    Beam search with shared prefix-dominance and incumbent-aware greedy completion,
    anti-buddy filtering from sampled pairwise preferences, and ΔW-gated LNS refinement.

    Args:
        workload: Workload object containing transaction data
        num_seqs: Number of restarts (used as intensity; actual count is time-bounded)

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    N = workload.num_txns
    start_time = time.time()
    # Tuned for combined score (quality + speed)
    time_budget_sec = 0.62 if N >= 90 else 0.55

    def time_left():
        return (time.time() - start_time) < time_budget_sec

    rng = random.Random(1729 + 31 * max(1, N))

    # Shared caches across the entire run
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

    # Singleton costs for LB and seeding
    singleton_cost = {}
    for t in all_txns:
        if not time_left():
            break
        singleton_cost[t] = eval_seq_cost([t])
    singles_sorted = sorted(all_txns, key=lambda t: singleton_cost.get(t, float('inf')))

    # Sampled pairwise preferences W[a][b] = cost([a,b]) - cost([b,a])
    W = defaultdict(dict)
    # Probe budget scaled with N
    max_pair_probes = min(1400, max(600, N * 12))
    probes_done = 0

    # Candidate pool to bias toward informative pairs
    bias_pool = singles_sorted[:min(N, max(20, N // 3))]
    if N > 0:
        bias_pool = list(dict.fromkeys(bias_pool + rng.sample(all_txns, min(N, max(20, N // 3)))))

    for a in all_txns:
        if not time_left():
            break
        peers = [x for x in bias_pool if x != a]
        rng.shuffle(peers)
        peers = peers[:min(14, max(8, N // 8))]
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
            probes_done += 2

    # Anti-buddy threshold: 75th percentile of positive W[a][*] per a
    anti_buddy_thresh = {}
    for a in all_txns:
        row = W.get(a, {})
        pos = [v for v in row.values() if v > 0]
        if not pos:
            anti_buddy_thresh[a] = float('inf')
        else:
            pos.sort()
            idx = int(0.75 * (len(pos) - 1))
            anti_buddy_thresh[a] = pos[idx]

    # Buddies list per txn: top neighbors minimizing extension delta
    pair_ext_cache = {}
    def pair_ext_cost(a, b):
        key = (a, b)
        v = pair_ext_cache.get(key)
        if v is not None:
            return v
        base = singleton_cost[a]
        ec = eval_ext_cost((a,), b)
        d = ec - base
        pair_ext_cache[key] = d
        return d

    def build_buddies(max_buddies=8, sample_per_t=16):
        buds = {t: [] for t in all_txns}
        for t in all_txns:
            if not time_left():
                break
            base = singleton_cost[t]
            pool = [u for u in singles_sorted[:min(24, max(8, N // 5))] if u != t]
            others = [u for u in all_txns if u != t and u not in pool]
            if others:
                pool.extend(rng.sample(others, min(sample_per_t, len(others))))
            seen = set()
            scored = []
            for u in pool:
                if u in seen or u == t:
                    continue
                seen.add(u)
                ec = eval_ext_cost((t,), u)
                scored.append((ec - base, u))
            scored.sort(key=lambda x: x[0])
            buds[t] = [u for _d, u in scored[:max_buddies]]
        return buds

    buddies = build_buddies(max_buddies=8, sample_per_t=min(16, max(8, N // 10)))

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

    # Prefix-dominance cache shared globally: (frozenset(rem), tail tuple) -> best cost
    prefix_dom = {}

    def dom_sig(rem_set, seq, k=3):
        tail = tuple(seq[-k:]) if len(seq) >= k else tuple(seq)
        return (frozenset(rem_set), tail)

    def is_antibuddy(last, cand):
        if last is None:
            return False
        v = W.get(last, {}).get(cand, 0.0)
        thr = anti_buddy_thresh.get(last, float('inf'))
        return v > 0 and v >= thr

    # Greedy completion with anti-buddy gating and prefix-dominance updates
    def greedy_complete(prefix, rem_set, branch_k=12, incumbent=None, recent_k=3):
        seq = list(prefix)
        rem = set(rem_set)
        cur_cost = eval_seq_cost(seq) if seq else 0.0
        steps = 0
        while rem and time_left():
            steps += 1
            # Dominance prune
            s = dom_sig(rem, seq, k=recent_k)
            prev = prefix_dom.get(s)
            if prev is not None and cur_cost >= prev:
                break
            prefix_dom[s] = cur_cost if prev is None else min(prev, cur_cost)

            # Incumbent-based LB
            if incumbent is not None and lb_singleton(cur_cost, rem) >= incumbent:
                break

            rem_list = list(rem)
            pool = []
            if seq:
                last = seq[-1]
                for u in buddies.get(last, []):
                    if u in rem and u not in pool:
                        pool.append(u)
            # add low singletons
            low_single = sorted(rem_list, key=lambda t: singleton_cost.get(t, float('inf')))[:min(5, len(rem_list))]
            for u in low_single:
                if u not in pool:
                    pool.append(u)
            # fill with randoms
            need = max(0, branch_k - len(pool))
            if need > 0:
                others = [x for x in rem_list if x not in pool]
                if others:
                    pool.extend(rng.sample(others, min(need, len(others))))
            if not pool:
                pool = rem_list

            pt = tuple(seq)
            scored = []
            best_immediate = float('inf')
            for t in pool:
                c = eval_ext_cost(pt, t)
                scored.append((c, t))
                if c < best_immediate:
                    best_immediate = c
            # Anti-buddy gate: skip clearly bad anti-buddies unless within 1%
            filtered = []
            last = seq[-1] if seq else None
            for c, t in scored:
                if last is not None and is_antibuddy(last, t) and c > best_immediate * 1.01:
                    continue
                filtered.append((c, t))
            if not filtered:
                filtered = scored
            filtered.sort(key=lambda x: x[0])
            best_c, best_t = filtered[0]
            seq.append(best_t)
            rem.remove(best_t)
            cur_cost = best_c

            # Opportunistically hoist dominance every 9 steps
            if steps % 9 == 0:
                s = dom_sig(rem, seq, k=recent_k)
                prev = prefix_dom.get(s)
                if prev is None or cur_cost < prev:
                    prefix_dom[s] = cur_cost

        if rem:
            # append remainder arbitrarily and evaluate true cost
            seq.extend(list(rem))
            cur_cost = eval_seq_cost(seq)
        return cur_cost, seq

    # Beam search with incumbent-aware greedy probes and global prefix dominance
    def beam_seed(params, incumbent=float('inf')):
        beam = params['beam']
        branch = params['branch']
        lookahead_top = params['lookahead_top']
        next_k = params['next_k']
        recent_k = params['recent_k']

        # Seed with best singletons and a few random
        seed_pool = singles_sorted[:max(beam * 2, 8)]
        extras = [x for x in all_txns if x not in seed_pool]
        if extras:
            seed_pool.extend(rng.sample(extras, min(4, len(extras))))
        nodes = []
        for t in seed_pool:
            if not time_left():
                break
            seq = [t]
            rem = set(all_txns)
            rem.remove(t)
            c = eval_seq_cost(seq)
            nodes.append((c, seq, rem))
        nodes.sort(key=lambda x: x[0])
        nodes = nodes[:beam] if nodes else []

        if not nodes:
            seq = all_txns[:]
            rng.shuffle(seq)
            return eval_seq_cost(seq), seq

        best_full = incumbent
        best_seq = None

        # Early greedy on best prefix
        if time_left():
            cg, sg = greedy_complete(nodes[0][1], nodes[0][2], branch_k=max(8, branch), incumbent=best_full, recent_k=recent_k)
            if len(sg) == N and cg < best_full:
                best_full, best_seq = cg, sg

        steps = N - 1
        depth = 0
        while depth < steps and time_left():
            depth += 1
            new_nodes = []

            # Depth-adaptive lookahead and suffix for dominance
            if depth < int(0.4 * N):
                la_top = lookahead_top
                k_suffix = max(3, recent_k)
            elif depth < int(0.75 * N):
                la_top = max(2, lookahead_top - 1)
                k_suffix = max(2, recent_k - 1)
            else:
                la_top = max(2, lookahead_top - 2)
                k_suffix = max(2, recent_k - 1)

            # Tighten incumbent occasionally
            for c_so_far, seq, rem in nodes[:min(2, len(nodes))]:
                if not time_left():
                    break
                cg, sg = greedy_complete(seq, rem, branch_k=max(8, branch), incumbent=best_full, recent_k=recent_k)
                if len(sg) == N and cg < best_full:
                    best_full, best_seq = cg, sg

            for c_so_far, seq, rem in nodes:
                if not rem:
                    if c_so_far < best_full:
                        best_full, best_seq = c_so_far, seq[:]
                    continue
                # incumbent prune
                if c_so_far >= best_full:
                    continue
                # prefix dominance prune
                s = dom_sig(rem, seq, k=k_suffix)
                prev = prefix_dom.get(s)
                if prev is not None and c_so_far >= prev:
                    continue
                prefix_dom[s] = c_so_far if prev is None else min(prev, c_so_far)

                rem_list = list(rem)
                # Build candidate pool: buddies of last + random fill
                cand_pool = []
                last = seq[-1]
                for u in buddies.get(last, []):
                    if u in rem and u not in cand_pool:
                        cand_pool.append(u)
                need = max(0, branch * 2 - len(cand_pool))
                if need > 0:
                    others = [x for x in rem_list if x not in cand_pool]
                    if others:
                        cand_pool.extend(rng.sample(others, min(need, len(others))))
                if not cand_pool:
                    cand_pool = rem_list if len(rem_list) <= branch * 2 else rng.sample(rem_list, branch * 2)

                pt = tuple(seq)
                scored = []
                best_immediate = float('inf')
                temp = []
                for t in cand_pool:
                    if not time_left():
                        break
                    ec = eval_ext_cost(pt, t)
                    temp.append((ec, t))
                    if ec < best_immediate:
                        best_immediate = ec
                # Anti-buddy gating
                for ec, t in temp:
                    if is_antibuddy(last, t) and ec > best_immediate * 1.01:
                        continue
                    la = ec
                    # shallow lookahead
                    if rem and time_left():
                        new_rem = rem.copy()
                        new_rem.remove(t)
                        la_pool = [v for v in buddies.get(t, []) if v in new_rem]
                        if not la_pool:
                            la_pool = list(new_rem)
                        if len(la_pool) > la_top:
                            la_pool = rng.sample(la_pool, la_top)
                        new_pt = tuple(seq + [t])
                        for nxt in la_pool:
                            c2 = eval_ext_cost(new_pt, nxt)
                            if c2 < la:
                                la = c2
                    scored.append((ec, la, t))
                if not scored:
                    continue
                scored.sort(key=lambda x: (x[0], x[1]))
                top = scored[:min(branch, len(scored))]

                kept_any = False
                # Incumbent-aware greedy completion on top next_k children
                for idx, (ec, la, t) in enumerate(top):
                    new_seq = seq + [t]
                    new_rem = rem.copy()
                    new_rem.remove(t)
                    # Child LB prune
                    if lb_singleton(ec, new_rem) >= best_full:
                        continue
                    adj_la = la
                    if idx < next_k and time_left():
                        cg, sg = greedy_complete(new_seq, new_rem, branch_k=max(8, branch), incumbent=best_full, recent_k=recent_k)
                        if len(sg) == N and cg < best_full:
                            best_full, best_seq = cg, sg
                        adj_la = min(adj_la, cg)
                        # prune this child if its full completion is not better than incumbent
                        if cg >= best_full:
                            # still possibly keep best child below to avoid empty beam
                            pass
                    new_nodes.append((ec, new_seq, new_rem, adj_la))
                    kept_any = True

                # Always keep at least the single best child
                if not kept_any:
                    ec, la, t = min(scored, key=lambda x: (x[0], x[1]))
                    new_seq = seq + [t]
                    new_rem = rem.copy()
                    new_rem.remove(t)
                    new_nodes.append((ec, new_seq, new_rem, la))

            if not new_nodes:
                break

            # Select next beam using adjusted score; keep unique prefixes
            new_nodes.sort(key=lambda x: x[3])
            unique = []
            seen = set()
            for ec, s, r, la in new_nodes:
                key = tuple(s)
                if key in seen:
                    continue
                seen.add(key)
                unique.append((ec, s, r))
                if len(unique) >= beam:
                    break
            nodes = unique

        # Finish remaining prefixes greedily
        for ec, s, r in nodes:
            if not time_left():
                break
            cg, sg = greedy_complete(s, r, branch_k=max(8, branch), incumbent=best_full, recent_k=recent_k)
            if len(sg) == N and cg < best_full:
                best_full, best_seq = cg, sg

        if best_seq is None:
            seq = all_txns[:]
            rng.shuffle(seq)
            best_seq = seq
            best_full = eval_seq_cost(seq)
        return best_full, best_seq

    # ΔW surrogate helpers for LNS
    def pref(a, b):
        return W.get(a, {}).get(b, 0.0)

    def worst_adjacencies(seq, topk=2):
        viols = []
        for i in range(len(seq) - 1):
            a, b = seq[i], seq[i + 1]
            m = pref(a, b)  # bad if positive (prefer b before a)
            viols.append((m, i))
        viols.sort(key=lambda x: -x[0])
        return viols[:topk]

    def deltaW_for_insert(seq, block, pos, recent_k=3):
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

    # LNS: block-swap and block-reinsert with evaluation cap
    def lns_improve(seq, cur_cost, eval_cap=650):
        best_seq = seq[:]
        best_cost = cur_cost
        n = len(best_seq)
        if n < 6 or not time_left():
            return best_cost, best_seq

        evals = 0

        # Cheap adjacent pass before
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

        if evals >= eval_cap or not time_left():
            return best_cost, best_seq

        # Block-swap centered on worst adjacencies
        viols = worst_adjacencies(best_seq, topk=2)
        centers = [i for _, i in viols] if viols else [n // 3, 2 * n // 3]
        block_size = min(6, max(3, n // 30))
        for a_idx in range(len(centers)):
            for b_idx in range(a_idx + 1, len(centers)):
                if evals >= eval_cap or not time_left():
                    break
                i = centers[a_idx]
                j = centers[b_idx]
                si = max(0, min(i - block_size // 2, n - block_size))
                sj = max(0, min(j - block_size // 2, n - block_size))
                if abs(si - sj) < block_size:
                    continue
                cand = best_seq[:]
                if si > sj:
                    si, sj = sj, si
                block_i = cand[si:si + block_size]
                block_j = cand[sj:sj + block_size]
                mid = cand[si + block_size:sj]
                cand2 = cand[:si] + block_j + mid + block_i + cand[sj + block_size:]
                c = eval_seq_cost(cand2)
                evals += 1
                if c < best_cost:
                    best_cost = c
                    best_seq = cand2

        if evals >= eval_cap or not time_left():
            return best_cost, best_seq

        # Block reinsert at ΔW-ranked positions
        n = len(best_seq)
        center = centers[0] if centers else n // 2
        bsize = min(6, max(4, n // 28))
        start = max(0, min(center - bsize // 2, n - bsize))
        block = best_seq[start:start + bsize]
        remain = best_seq[:start] + best_seq[start + bsize:]

        positions = list(range(0, len(remain) + 1))
        scored_pos = [(deltaW_for_insert(remain, block, p, recent_k=3), p) for p in positions]
        scored_pos.sort(key=lambda x: -x[0])  # higher ΔW worse -> prioritize
        k_eval = max(3, int(0.4 * len(scored_pos)))
        top_positions = [p for _s, p in scored_pos[:k_eval]]
        # add ~10% random
        extra = max(1, len(positions) // 10)
        rand_positions = rng.sample(positions, min(extra, len(positions)))
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

        # Final adjacent pass
        if time_left() and evals < eval_cap:
            for i in range(len(best_seq) - 1):
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

    # Portfolio restarts (deterministic) with shared caches
    portfolios = [
        {'beam': 16, 'branch': 12, 'lookahead_top': 4, 'next_k': 4, 'recent_k': 5},
        {'beam': 12, 'branch': 14, 'lookahead_top': 3, 'next_k': 5, 'recent_k': 4},
        {'beam': 10, 'branch': 10, 'lookahead_top': 3, 'next_k': 4, 'recent_k': 3},
    ]

    incumbent_cost = float('inf')
    incumbent_seq = None

    max_restarts = min(len(portfolios), max(2, int(num_seqs)))
    # Quick greedy baseline to set an initial incumbent
    if time_left() and N > 0:
        seed = [singles_sorted[0]]
        rem = set(all_txns) - {seed[0]}
        c0, s0 = greedy_complete(seed, rem, branch_k=12, incumbent=incumbent_cost, recent_k=3)
        if len(s0) == N and c0 < incumbent_cost:
            incumbent_cost, incumbent_seq = c0, s0

    for r in range(max_restarts):
        if not time_left():
            break
        params = portfolios[r]
        c_seed, s_seed = beam_seed(params, incumbent=incumbent_cost)
        # Cheap polish before LNS
        if time_left():
            c_ln, s_ln = lns_improve(s_seed, c_seed, eval_cap=700 if N >= 90 else 600)
        else:
            c_ln, s_ln = c_seed, s_seed
        if c_ln < incumbent_cost:
            incumbent_cost, incumbent_seq = c_ln, s_ln

    # Safety: ensure a valid permutation
    if incumbent_seq is None:
        seq = list(range(N))
        rng.shuffle(seq)
        incumbent_seq = seq
        incumbent_cost = eval_seq_cost(seq)

    if len(incumbent_seq) != N or len(set(incumbent_seq)) != N:
        seen = set()
        repaired = []
        for t in incumbent_seq:
            if 0 <= t < N and t not in seen:
                repaired.append(t)
                seen.add(t)
        for t in range(N):
            if t not in seen:
                repaired.append(t)
        incumbent_seq = repaired[:N]
        incumbent_cost = eval_seq_cost(incumbent_seq)

    return incumbent_cost, incumbent_seq


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