# EVOLVE-BLOCK-START
"""Transaction scheduling algorithm for optimizing makespan across multiple workloads"""

import time
import random
import sys
import os
from collections import defaultdict, deque

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
    Two-phase buddy-guided beam search with prefix-dominance pruning,
    shallow adaptive lookahead, incumbent-based pruning, and VNS refinement.

    Args:
        workload: Workload object containing transaction data
        num_seqs: Number of random restarts (used as an upper bound; also time-bounded)

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    N = workload.num_txns
    start_time = time.time()
    # Budget chosen to balance quality and runtime for combined score
    base_budget = 0.58
    time_budget_sec = base_budget
    rng = random.Random(1729 + 31 * N)

    def time_left():
        return (time.time() - start_time) < time_budget_sec

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

    # Build buddy lists B[t]: top partners minimizing cost([t,u]).
    # Keep small lists (6–8) for speed/focus. Sample candidates to bound cost.
    def build_buddies(max_buddies=8):
        buddies = {t: [] for t in all_txns}
        anti_thresh = {t: float('inf') for t in all_txns}
        # Candidate pools per t
        singles_sorted = sorted(all_txns, key=lambda x: singleton_cost.get(x, float('inf')))
        for t in all_txns:
            if not time_left():
                break
            # Candidate sample: mix of top by singleton and random
            cand_pool = []
            top_slice = singles_sorted[:min(20, max(8, N // 6))]
            cand_pool.extend(top_slice)
            if N > 1:
                extra = min(24, N - 1)
                cand_pool.extend(rng.sample([x for x in all_txns if x != t], min(extra, max(10, N // 5))))
            # Deduplicate and remove t
            pool = []
            seen_l = set()
            for u in cand_pool:
                if u == t or u in seen_l:
                    continue
                seen_l.add(u)
                pool.append(u)
            scored = []
            pos_deltas = []
            base = singleton_cost.get(t)
            prefix_tuple = (t,)  # singleton prefix
            for u in pool:
                if not time_left():
                    break
                c2 = eval_ext_cost(prefix_tuple, u)
                delta = c2 - base
                scored.append((delta, u))  # delta over singleton
                if delta > 0:
                    pos_deltas.append(delta)
            scored.sort(key=lambda x: x[0])
            buddies[t] = [u for _d, u in scored[:max_buddies]]
            # Anti-buddy threshold as 75th percentile of positive deltas (if any)
            if pos_deltas:
                pos_deltas.sort()
                idx = int(0.75 * (len(pos_deltas) - 1))
                anti_thresh[t] = pos_deltas[idx]
        return buddies, anti_thresh

    buddies, anti_buddy_thresh = build_buddies(max_buddies=8)

    # Pairwise adjacency surrogate using true two-step marginal delta
    pair_cost_cache = {}
    def pair_cost(a, b):
        key = (a, b)
        c = pair_cost_cache.get(key)
        if c is not None:
            return c
        base = singleton_cost.get(a)
        if base is None:
            base = eval_seq_cost([a])
            singleton_cost[a] = base
        ec = eval_ext_cost((a,), b)
        delta = ec - base
        pair_cost_cache[key] = delta
        return delta

    # Symmetric pairwise preference: negative prefers a->b, positive prefers b->a
    def pair_pref(a, b):
        ab = eval_ext_cost((a,), b)
        ba = eval_ext_cost((b,), a)
        return ab - ba

    # Anti-buddy gate using learned threshold per 'last'
    def is_antibuddy(last, cand):
        if last is None:
            return False
        thr = anti_buddy_thresh.get(last, float('inf'))
        if thr == float('inf'):
            return False
        pc = pair_cost(last, cand)
        return pc > 0 and pc >= thr

    # Global prefix-dominance map shared across restarts/phases
    best_state_global = {}

    # Lower bound using max remaining singleton cost
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

    # Greedy completion guided by buddies and extension costs
    def greedy_finish(seq, rem_set, branch_k=10, incumbent=None):
        seq_out = list(seq)
        rem = set(rem_set)
        cur_cost = eval_seq_cost(seq_out) if seq_out else 0
        while rem and time_left():
            if incumbent is not None:
                if lb_singleton(cur_cost, rem) >= incumbent:
                    break
            last = seq_out[-1] if seq_out else None
            rem_list = list(rem)
            cand_pool = []

            # Prefer buddies of last
            if last is not None and last in buddies:
                for u in buddies[last]:
                    if u in rem:
                        cand_pool.append(u)

            # Also include a few low-singleton txns
            low_single = sorted(rem_list, key=lambda t: singleton_cost.get(t, float('inf')))[:min(5, len(rem_list))]
            for u in low_single:
                if u not in cand_pool:
                    cand_pool.append(u)

            # Fill with random sample for diversity
            need = max(0, branch_k - len(cand_pool))
            if need > 0:
                others = [x for x in rem_list if x not in cand_pool]
                if len(others) > need:
                    cand_pool.extend(rng.sample(others, need))
                else:
                    cand_pool.extend(others)

            if not cand_pool:
                cand_pool = rem_list if len(rem_list) <= branch_k else rng.sample(rem_list, branch_k)

            # Update global prefix-dominance for this greedy prefix state
            sig_g = make_signature(rem, seq_out, 3)
            prev_g = best_state_global.get(sig_g)
            if prev_g is None or cur_cost < prev_g:
                best_state_global[sig_g] = cur_cost

            prefix_tuple = tuple(seq_out)
            # Score immediate extensions
            scored = []
            best_immediate = float('inf')
            for t in cand_pool:
                c = eval_ext_cost(prefix_tuple, t)
                scored.append((c, t))
                if c < best_immediate:
                    best_immediate = c
            # Anti-buddy filtering: depth-adaptive tolerance (more lenient early, stricter late)
            filtered = []
            last_txn = seq_out[-1] if seq_out else None
            depth_ratio = (len(seq_out) / max(1, N))
            if depth_ratio < 0.33:
                tol_factor = 1.012
            elif depth_ratio < 0.66:
                tol_factor = 1.008
            else:
                tol_factor = 1.004
            tol = best_immediate * tol_factor if best_immediate < float('inf') else float('inf')
            for c, t in scored:
                if is_antibuddy(last_txn, t) and c > tol:
                    continue
                filtered.append((c, t))
            if not filtered:
                filtered = scored

            # Pick best candidate after filtering
            filtered.sort(key=lambda x: x[0])
            best_c, best_t = filtered[0]
            if best_t is None:
                # Time exhausted; append remaining arbitrarily
                seq_out.extend(rem)
                cur_cost = eval_seq_cost(seq_out)
                return cur_cost, seq_out
            seq_out.append(best_t)
            rem.remove(best_t)
            cur_cost = best_c
        if rem:
            seq_out.extend(list(rem))
            cur_cost = eval_seq_cost(seq_out)
        return cur_cost, seq_out

    # Prefix-dominance pruning state
    def make_signature(rem_set, seq, k_suffix):
        if k_suffix <= 0:
            return (frozenset(rem_set), ())
        tail = tuple(seq[-k_suffix:]) if len(seq) >= k_suffix else tuple(seq)
        return (frozenset(rem_set), tail)

    # Core beam runner with params and incumbent-based pruning
    def run_beam(params, incumbent_cost=float('inf'), k_suffix=3):
        beam_width = params['beam']
        branch_factor = params['branch']
        lookahead_top = params['lookahead_top']
        next_k = params['next_k']

        all_txns_local = all_txns
        # Seed: best singletons by cost + a few strong pairs
        seeds = sorted(all_txns_local, key=lambda t: singleton_cost.get(t, float('inf')))
        seeds = seeds[:max(beam_width * 2, 8)]
        beam = []
        for t in seeds:
            if not time_left():
                break
            seq = [t]
            rem = set(all_txns_local)
            rem.remove(t)
            c = eval_seq_cost(seq)
            beam.append((c, seq, rem))
        # Add pair seeds from top singletons to capture early conflicts
        if time_left() and len(seeds) >= 2:
            pair_candidates = seeds[:min(12, len(seeds))]
            pair_evals = []
            for i in range(len(pair_candidates)):
                if not time_left():
                    break
                a = pair_candidates[i]
                for j in range(i + 1, len(pair_candidates)):
                    if not time_left():
                        break
                    b = pair_candidates[j]
                    c2 = eval_seq_cost([a, b])
                    pair_evals.append((c2, [a, b]))
                    c3 = eval_seq_cost([b, a])
                    pair_evals.append((c3, [b, a]))
            pair_evals.sort(key=lambda x: x[0])
            for c2, pseq in pair_evals[:min(beam_width, len(pair_evals))]:
                rem = set(all_txns_local)
                for x in pseq:
                    if x in rem:
                        rem.remove(x)
                beam.append((c2, pseq, rem))
        if not beam:
            seq = all_txns_local[:]
            rng.shuffle(seq)
            return eval_seq_cost(seq), seq, incumbent_cost, None

        beam.sort(key=lambda x: x[0])
        beam = beam[:beam_width]

        # Prefix-dominance map (local) and track best full
        best_state = {}

        best_full_cost = incumbent_cost
        best_full_seq = None

        steps = N - 1
        depth = 0
        while depth < steps and time_left():
            depth += 1
            new_beam = []

            # Depth-adaptive suffix length for dominance
            # Keep more context early; shrink later to merge states
            if depth < int(0.35 * N):
                k_cur = max(3, k_suffix)
            elif depth < int(0.7 * N):
                k_cur = max(2, k_suffix - 1)
            else:
                k_cur = max(2, k_suffix - 2)

            for cost_so_far, seq, rem in beam:
                if not rem:
                    # complete sequence
                    if cost_so_far < best_full_cost:
                        best_full_cost, best_full_seq = cost_so_far, seq[:]
                    continue

                # incumbent pruning
                if cost_so_far >= best_full_cost:
                    continue

                # prefix-dominance pruning (local and global)
                sig = make_signature(rem, seq, k_cur)
                prev_local = best_state.get(sig)
                if prev_local is not None and cost_so_far >= prev_local:
                    continue
                prev_global = best_state_global.get(sig)
                if prev_global is not None and cost_so_far >= prev_global:
                    continue
                # Update both
                best_state[sig] = cost_so_far
                # Keep minimum cost in global map
                cur_best = best_state_global.get(sig)
                if cur_best is None or cost_so_far < cur_best:
                    best_state_global[sig] = cost_so_far

                # Candidate pool: buddies of last + random sample
                last = seq[-1]
                rem_list = list(rem)
                cand_pool = []
                if last in buddies:
                    for u in buddies[last]:
                        if u in rem:
                            cand_pool.append(u)
                # supplement with random to reach 2*branch
                need = max(0, branch_factor * 2 - len(cand_pool))
                if need > 0:
                    others = [x for x in rem_list if x not in cand_pool]
                    add = min(len(others), need)
                    if add > 0:
                        cand_pool.extend(rng.sample(others, add))

                if not cand_pool:
                    cand_pool = rem_list if len(rem_list) <= branch_factor * 2 else rng.sample(rem_list, branch_factor * 2)

                # Score by marginal delta and shallow lookahead with anti-buddy gating
                prefix_tuple = tuple(seq)
                # First pass: compute immediate extension costs and best
                tmp = []
                best_immediate = float('inf')
                for cand in cand_pool:
                    if not time_left():
                        break
                    ec = eval_ext_cost(prefix_tuple, cand)
                    tmp.append((ec, cand))
                    if ec < best_immediate:
                        best_immediate = ec

                # Second pass: anti-buddy filter and lookahead (depth-adaptive tolerance)
                scored = []
                depth_ratio = (len(seq) / max(1, N))
                if depth_ratio < 0.33:
                    tol_factor = 1.012
                elif depth_ratio < 0.66:
                    tol_factor = 1.008
                else:
                    tol_factor = 1.004
                tol = best_immediate * tol_factor if best_immediate < float('inf') else float('inf')
                for ec, cand in tmp:
                    if ec >= best_full_cost:
                        continue
                    if is_antibuddy(last, cand) and ec > tol:
                        continue
                    # shallow lookahead: prefer buddy-next of cand first
                    la_best = ec
                    if rem and time_left():
                        new_rem = rem.copy()
                        if cand in new_rem:
                            new_rem.remove(cand)
                        la_pool = []
                        if cand in buddies:
                            for v in buddies[cand]:
                                if v in new_rem:
                                    la_pool.append(v)
                        if not la_pool:
                            la_pool = list(new_rem)
                        # sample top 'lookahead_top'
                        if len(la_pool) > lookahead_top:
                            la_pool = rng.sample(la_pool, lookahead_top)
                        new_prefix_tuple = tuple(seq + [cand])
                        for nxt in la_pool:
                            c2 = eval_ext_cost(new_prefix_tuple, nxt)
                            if c2 < la_best:
                                la_best = c2
                    scored.append((ec - cost_so_far, la_best, ec, cand))

                if not scored:
                    continue
                # rank by marginal delta then lookahead score
                scored.sort(key=lambda x: (x[0], x[1]))
                top = scored[:min(branch_factor, len(scored))]
                # Probe a few best children by greedily completing them to tighten incumbent
                probe_k = min(params.get('next_k', 4), len(top))
                idx_child = 0
                for _delta, la_score, ec, cand in top:
                    new_seq = seq + [cand]
                    new_rem = rem.copy()
                    new_rem.remove(cand)
                    # Child signature dominance check before expensive probes
                    sig_child = make_signature(new_rem, new_seq, k_cur)
                    prev_local_c = best_state.get(sig_child)
                    if prev_local_c is not None and ec >= prev_local_c:
                        continue
                    prev_global_c = best_state_global.get(sig_child)
                    if prev_global_c is not None and ec >= prev_global_c:
                        continue
                    # Update dominance with child's extension cost
                    best_state[sig_child] = ec if prev_local_c is None else min(prev_local_c, ec)
                    cur_best = best_state_global.get(sig_child)
                    if cur_best is None or ec < cur_best:
                        best_state_global[sig_child] = ec
                    # Child LB pruning
                    if lb_singleton(ec, new_rem) >= best_full_cost:
                        continue
                    # Greedy probe for first few children to update incumbent and refine ranking
                    if idx_child < probe_k and time_left():
                        g_cost, g_seq = greedy_finish(new_seq, new_rem, branch_k=max(6, N // 12), incumbent=best_full_cost)
                        if len(g_seq) == N and g_cost < best_full_cost:
                            best_full_cost, best_full_seq = g_cost, g_seq
                        # Strengthen dominance with probe result
                        prev_local_probe = best_state.get(sig_child)
                        if prev_local_probe is None or g_cost < prev_local_probe:
                            best_state[sig_child] = g_cost
                        prev_global_probe = best_state_global.get(sig_child)
                        if prev_global_probe is None or g_cost < prev_global_probe:
                            best_state_global[sig_child] = g_cost
                        la_score = min(la_score, g_cost)
                    idx_child += 1
                    # Prune child if its adjusted lookahead is not better than the incumbent
                    if la_score >= best_full_cost:
                        continue
                    new_beam.append((ec, new_seq, new_rem, la_score))

            if not new_beam:
                break

            # Select next beam by lookahead score; keep unique prefixes
            new_beam.sort(key=lambda x: x[3])
            unique = []
            seen = set()
            for entry in new_beam:
                key = tuple(entry[1])
                if key in seen:
                    continue
                seen.add(key)
                unique.append((entry[0], entry[1], entry[2]))
                if len(unique) >= beam_width:
                    break
            beam = unique

            # Periodically greedily complete top-2 prefixes
            if beam and time_left():
                for c_pref, s_pref, r_pref in beam[:min(2, len(beam))]:
                    c_try, s_try = greedy_finish(s_pref, r_pref, branch_k=max(8, N // 10), incumbent=best_full_cost)
                    if len(s_try) == N and c_try < best_full_cost:
                        best_full_cost, best_full_seq = c_try, s_try

        # Finalize: greedily finish remaining prefixes
        for c_pref, s_pref, r_pref in beam:
            if not time_left():
                break
            c_fin, s_fin = greedy_finish(s_pref, r_pref, branch_k=max(8, N // 10), incumbent=best_full_cost)
            if len(s_fin) == N and c_fin < best_full_cost:
                best_full_cost, best_full_seq = c_fin, s_fin

        if best_full_seq is None:
            seq = all_txns_local[:]
            rng.shuffle(seq)
            best_full_seq = seq
            best_full_cost = eval_seq_cost(seq)
        return best_full_cost, best_full_seq, best_full_cost, beam[:]

    # Lightweight VNS refinement: adjacent swaps and relocations
    def local_improve(seq, current_cost):
        best_seq = seq[:]
        best_cost = current_cost

        # Multiple adjacent-swap passes until no improvement or time
        for _ in range(2):
            improved = False
            for i in range(len(best_seq) - 1):
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

        # Boundary-focused ruin-and-recreate around the worst adjacency
        def boundary_repair_once(cur_seq, cur_cost):
            n = len(cur_seq)
            if n < 6 or not time_left():
                return cur_cost, cur_seq
            # Find worst adjacency by surrogate pair_cost
            worst_idx = -1
            worst_val = -float('inf')
            for i in range(n - 1):
                if not time_left():
                    break
                v = pair_pref(cur_seq[i], cur_seq[i + 1])
                if v > worst_val:
                    worst_val = v
                    worst_idx = i
            if worst_idx < 0:
                return cur_cost, cur_seq
            # Remove a small block centered at worst_idx
            block_size = min(5, max(3, n // 30))
            start = max(0, min(worst_idx - block_size // 2, n - block_size))
            block = cur_seq[start:start + block_size]
            remaining = cur_seq[:start] + cur_seq[start + block_size:]
            seq_build = remaining[:]
            # Greedily reinsert block elements at best positions using true cost
            for x in block:
                if not time_left():
                    break
                positions = list(range(max(0, start - 3), min(len(seq_build) + 1, start + 4)))
                # add a few random positions for diversity
                extra = set()
                limit = min(8, len(seq_build) + 1)
                while len(extra) < limit and time_left():
                    extra.add(rng.randrange(len(seq_build) + 1))
                for p in extra:
                    if p not in positions:
                        positions.append(p)
                best_local_cost = float('inf')
                best_pos = 0
                for p in positions:
                    cand = seq_build[:]
                    cand.insert(p, x)
                    c = eval_seq_cost(cand)
                    if c < best_local_cost:
                        best_local_cost = c
                        best_pos = p
                seq_build.insert(best_pos, x)
            if len(seq_build) == n:
                c_new = eval_seq_cost(seq_build)
                if c_new < cur_cost:
                    return c_new, seq_build
            return cur_cost, cur_seq

        # Attempt a couple of boundary repairs
        for _ in range(2):
            if not time_left():
                break
            c_try, s_try = boundary_repair_once(best_seq, best_cost)
            if c_try < best_cost:
                best_cost, best_seq = c_try, s_try

        # Block-swap neighborhood around top-2 worst adjacencies (limited tries)
        if time_left():
            n = len(best_seq)
            if n >= 8:
                worst = []
                for i in range(n - 1):
                    if not time_left():
                        break
                    worst.append((pair_pref(best_seq[i], best_seq[i + 1]), i))
                worst.sort(key=lambda x: x[0], reverse=True)
                tries = 0
                for a in range(min(2, len(worst))):
                    if not time_left():
                        break
                    for b in range(a + 1, min(4, len(worst))):
                        if not time_left():
                            break
                        i = worst[a][1]
                        j = worst[b][1]
                        block = min(6, max(3, n // 40))
                        si = max(0, min(i - block // 2, n - block))
                        sj = max(0, min(j - block // 2, n - block))
                        # ensure non-overlap
                        if abs(si - sj) < block:
                            continue
                        cand = best_seq[:]
                        # ensure si < sj
                        if si > sj:
                            si, sj = sj, si
                        block_i = cand[si:si + block]
                        block_j = cand[sj:sj + block]
                        mid = cand[si + block:sj]
                        cand2 = cand[:si] + block_j + mid + block_i + cand[sj + block:]
                        c = eval_seq_cost(cand2)
                        tries += 1
                        if c < best_cost:
                            best_cost = c
                            best_seq = cand2
                    if tries >= 4:
                        break

        # Anchored block reinsert around worst boundary (move entire block to best positions)
        if time_left():
            n = len(best_seq)
            if n >= 7:
                # identify worst boundary again
                worst_idx = -1
                worst_val = -float('inf')
                for i in range(n - 1):
                    if not time_left():
                        break
                    v = pair_pref(best_seq[i], best_seq[i + 1])
                    if v > worst_val:
                        worst_val = v
                        worst_idx = i
                if worst_idx >= 0:
                    block_size = min(6, max(4, n // 28))
                    start = max(0, min(worst_idx + 1 - block_size // 2, n - block_size))
                    block = best_seq[start:start + block_size]
                    remain = best_seq[:start] + best_seq[start + block_size:]
                    # candidate positions near the boundary and a few random
                    positions = list(range(max(0, worst_idx - 3), min(len(remain) + 1, worst_idx + 4)))
                    extra = set()
                    limit = min(6, len(remain) + 1)
                    while len(extra) < limit and time_left():
                        extra.add(rng.randrange(len(remain) + 1))
                    for p in extra:
                        if p not in positions:
                            positions.append(p)
                    best_local = best_cost
                    best_candidate = None
                    for p in positions:
                        cand = remain[:]
                        for off, x in enumerate(block):
                            cand.insert(p + off, x)
                        c = eval_seq_cost(cand)
                        if c < best_local:
                            best_local = c
                            best_candidate = cand
                    if best_candidate is not None and best_local < best_cost:
                        best_cost, best_seq = best_local, best_candidate

        # Targeted relocations with small window (reduced trials, keep diversity)
        trials = 45
        n = len(best_seq)
        while trials > 0 and time_left():
            trials -= 1
            i = rng.randrange(n)
            j = rng.randrange(n)
            if i == j:
                continue
            cand = best_seq[:]
            val = cand.pop(i)
            cand.insert(j, val)
            c = eval_seq_cost(cand)
            if c < best_cost:
                best_cost = c
                best_seq = cand

        return best_cost, best_seq

    # Portfolio of parameter settings (deterministic seeds)
    portfolios = [
        {'beam': 16, 'branch': 12, 'lookahead_top': 4, 'next_k': 6, 'k_suffix': 3},
        {'beam': 12, 'branch': 14, 'lookahead_top': 3, 'next_k': 5, 'k_suffix': 3},
        {'beam': 10, 'branch': 10, 'lookahead_top': 3, 'next_k': 4, 'k_suffix': 4},
    ]

    global_best_cost = float('inf')
    global_best_seq = None

    # Early greedy baseline from empty prefix to tighten incumbent and pruning
    if time_left():
        c0, s0 = greedy_finish([], set(all_txns), branch_k=max(10, N // 8), incumbent=float('inf'))
        if c0 < global_best_cost:
            global_best_cost, global_best_seq = c0, s0

    max_restarts = max(2, min(len(portfolios), int(num_seqs)))
    for r in range(max_restarts):
        if not time_left():
            break
        # Use natural randomness for diversity across restarts

        # Phase A: broader beam to ~40% depth implicitly via parameter breadth
        params_A = portfolios[r % len(portfolios)].copy()
        cA, sA, incumbent_cost, _beam_frontier = run_beam(params_A, incumbent_cost=global_best_cost, k_suffix=params_A['k_suffix'])

        # Phase B: tighter beam with incumbent pruning
        if time_left():
            params_B = {'beam': max(8, params_A['beam'] - 4),
                        'branch': max(8, params_A['branch'] - 4),
                        'lookahead_top': max(2, params_A['lookahead_top'] - 1),
                        'next_k': max(4, params_A['next_k'] - 1),
                        'k_suffix': params_A['k_suffix']}
            cB, sB, _, _ = run_beam(params_B, incumbent_cost=min(global_best_cost, incumbent_cost), k_suffix=params_B['k_suffix'])
            if cB < cA:
                cA, sA = cB, sB

        # Local search refinement
        if time_left():
            cA, sA = local_improve(sA, cA)

        if cA < global_best_cost:
            global_best_cost, global_best_seq = cA, sA

    # Safety: ensure permutation validity
    if global_best_seq is None or len(global_best_seq) != N or len(set(global_best_seq)) != N:
        if global_best_seq is None:
            seq = list(range(N))
            random.shuffle(seq)
            global_best_seq = seq
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