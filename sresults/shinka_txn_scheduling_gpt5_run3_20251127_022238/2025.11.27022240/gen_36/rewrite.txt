# EVOLVE-BLOCK-START
"""Transaction scheduling algorithm for optimizing makespan across multiple workloads"""

import time
import random
import sys
import os

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
    Two-phase tournament-guided beam search with prefix-dominance pruning,
    adaptive lookahead, greedy completions, and VNS refinement.

    Args:
        workload: Workload object containing transaction data
        num_seqs: Controls beam width / exploration budget

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    n = workload.num_txns
    # Deterministic RNG per workload for stability with slight jitter from num_seqs
    rng = random.Random(1729 + n * 7 + int(num_seqs) * 13)

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
            imm = [(t, evaluate_seq(base + [t])) for t in cand_pool]
            imm.sort(key=lambda x: x[1])
            L = min(3, len(imm))
            best_t = imm[0][0]
            best_metric = imm[0][1]
            for t, immediate_c in imm[:L]:
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
        return base, evaluate_seq(base)

    # Two-phase beam search with incumbent and prefix-dominance pruning
    def beam_search_two_phase():
        # Phase parameters
        beam_A = max(12, min(16, int(num_seqs) + 4))  # broader
        beam_B = max(8, min(12, int(num_seqs)))       # tighter
        cand_A = 12
        cand_B = 8
        lookahead_top_A = 4
        lookahead_top_B = 3
        next_k_A = 6
        next_k_B = 5
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
            key = (tuple(seq), remaining)
            if key in used:
                continue
            used.add(key)
            beam.append((c, seq, remaining))

        incumbent_cost = float('inf')
        incumbent_seq = None

        # Prefix-dominance map
        best_sig_cost = {}

        # Beam loop across all depths
        for d in range(n - 1):
            phaseA = d < split_depth
            beam_width = beam_A if phaseA else beam_B
            cand_per_expand = cand_A if phaseA else cand_B
            lookahead_top = lookahead_top_A if phaseA else lookahead_top_B
            lookahead_next_k = next_k_A if phaseA else next_k_B
            recent_k = recent_k_for_depth(d)
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
                for t in pre:
                    c_im = evaluate_seq(seq + [t])
                    # Safe prune children by incumbent in Phase B
                    if (not phaseA) and (incumbent_cost < float('inf')) and (c_im >= incumbent_cost):
                        continue
                    imm.append((t, c_im))
                if not imm:
                    # no candidates left after pruning; fallback to best immediate without prune
                    for t in pre[:min(cand_per_expand, len(pre))]:
                        imm.append((t, evaluate_seq(seq + [t])))

                if not imm:
                    continue

                imm.sort(key=lambda x: x[1])

                # Lookahead over top few
                L = min(lookahead_top, len(imm))
                scored_ext = []
                for t, immediate_c in imm[:L]:
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
                expand_k = min(cand_per_expand, len(items))
                for t, _metric in items[:expand_k]:
                    new_seq = seq + [t]
                    new_cost = evaluate_seq(new_seq)
                    # Safe prune by incumbent
                    if (incumbent_cost < float('inf')) and (new_cost >= incumbent_cost):
                        continue
                    rem_new = remaining - {t}
                    # Prefix-dominance signature
                    sig = (rem_new, tuple(new_seq[-suffix_k:]) if len(new_seq) >= suffix_k else tuple(new_seq))
                    prev = best_sig_cost.get(sig)
                    if (prev is not None) and (new_cost >= prev):
                        continue
                    best_sig_cost[sig] = new_cost

                    key = (tuple(new_seq), rem_new)
                    if key in seen:
                        continue
                    seen.add(key)
                    next_beam.append((new_cost, new_seq, rem_new))

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

    def local_improve(seq, base_cost=None):
        # Robust VNS: tournament cleanup, adjacent swaps, relocations, limited swaps, ruin-and-recreate
        best_seq = list(seq)
        best_cost = evaluate_seq(best_seq) if base_cost is None else base_cost
        nloc = len(best_seq)

        # 0) Tournament-based cheap cleanup
        cand0 = tournament_bubble_pass(best_seq, passes=3)
        c0 = evaluate_seq(cand0)
        if c0 < best_cost:
            best_seq, best_cost = cand0, c0

        # 1) Adjacent swap hill-climb passes
        for _ in range(3):
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

        # 1b) Limited sampled 2-opt (non-adjacent swap)
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

        # 2) Sampled insertion (relocate) moves
        relocate_attempts = min(2 * nloc, 200) + max(0, int(num_seqs // 3))
        for _ in range(relocate_attempts):
            i, j = rng.sample(range(nloc), 2)
            if i == j:
                continue
            cand = best_seq[:]
            item = cand.pop(j)
            cand.insert(i, item)
            c = evaluate_seq(cand)
            if c < best_cost:
                best_seq, best_cost = cand, c

        # 3) Boundary-focused light ruin-and-recreate with greedy reinsertion
        if nloc > 12:
            # find worst violated adjacency indices by W
            def worst_violation_boundaries(seq_local, topm=3):
                pairs = []
                for ii in range(len(seq_local) - 1):
                    a, b = seq_local[ii], seq_local[ii + 1]
                    pen = M[a][b] - M[b][a]  # positive when a before b is worse
                    if pen > 0:
                        pairs.append((pen, ii))
                pairs.sort(reverse=True)
                return [i for _, i in pairs[:topm]]

            rr_tries = 2 if nloc < 90 else 1
            for _ in range(rr_tries):
                viol = worst_violation_boundaries(best_seq, topm=2)
                if viol:
                    center = viol[0] + 1
                    block_size = max(5, min(18, nloc // 6))
                    start = max(0, min(center - block_size // 2, nloc - block_size))
                else:
                    block_size = max(5, min(18, nloc // 6))
                    start = rng.randint(0, nloc - block_size)
                removed = best_seq[start:start + block_size]
                base = best_seq[:start] + best_seq[start + block_size:]

                for t in rng.sample(removed, len(removed)):
                    positions = {0, len(base)}
                    sample_k = min(6, len(base) + 1)
                    while len(positions) < sample_k:
                        positions.add(rng.randint(0, len(base)))
                    best_pos = 0
                    best_pos_cost = float('inf')
                    for pos in positions:
                        cand = base[:pos] + [t] + base[pos:]
                        c = evaluate_seq(cand)
                        if c < best_pos_cost:
                            best_pos_cost = c
                            best_pos = pos
                    base.insert(best_pos, t)

                c = evaluate_seq(base)
                if c < best_cost:
                    best_seq, best_cost = base, c

        return best_cost, best_seq

    # Run two-phase beam search to get high-quality schedule, then refine locally
    beam_cost, beam_seq = beam_search_two_phase()
    final_cost, final_seq = local_improve(beam_seq, base_cost=beam_cost)
    return final_cost, final_seq


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