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
    Tournament-guided multi-restart beam search with incumbent pruning, buddy-guided lookahead,
    periodic greedy completions, and VNS local improvement.

    Args:
        workload: Workload object containing transaction data
        num_seqs: Controls beam width / exploration budget

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    # Deterministic RNG seeded by workload size for stability
    n = workload.num_txns
    rng = random.Random(1729 + n)

    # Shared memoized evaluator for sequences to reduce simulator calls across all phases
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

    # Buddy lists: for each t, pick transactions u that pair best with t immediately after
    K_BUDDY = min(8, max(1, n - 1))
    buddies = [[] for _ in range(n)]
    for t in range(n):
        cand = [(W[t][u], u) for u in range(n) if u != t]
        cand.sort(key=lambda z: (z[0], z[1]))  # most negative W first (strong t before u)
        buddies[t] = [u for _, u in cand[:K_BUDDY]]

    # Global incumbent to prune expansions and guide greedy completions
    best_full_cost = float('inf')
    best_full_seq = None

    def greedy_complete(prefix, remaining_set):
        # Cheap greedy completion guided by tournament preferences
        seq = prefix[:]
        remaining = set(remaining_set)
        while remaining:
            rem_list = list(remaining)
            k = min(10, len(rem_list))
            pre = preselect_by_tournament(seq, rem_list, k)
            if not pre:
                pre = rem_list
            # pick the candidate minimizing immediate true cost
            best_t = None
            best_c = float('inf')
            for t in pre:
                c = evaluate_seq(seq + [t])
                if c < best_c:
                    best_c = c
                    best_t = t
            seq.append(best_t)
            remaining.remove(best_t)
        # light cleanup and evaluate
        seq = tournament_bubble_pass(seq, passes=1)
        return evaluate_seq(seq), seq

    # Beam-search variant with incumbent-based pruning and periodic greedy completions
    def beam_variant(beam_width, cand_per_expand, lookahead_top, lookahead_next_k):
        nonlocal best_full_cost, best_full_seq

        # Diverse starts: tournament-best, a good singleton, and randoms
        starts = []
        starts.append(tournament_order[0])
        topk = min(10, n)
        good_singletons = sorted(range(n), key=lambda t: c1[t])[:topk]
        starts.append(rng.choice(good_singletons))
        remaining_candidates = [t for t in range(n) if t not in starts]
        rng.shuffle(remaining_candidates)
        starts.extend(remaining_candidates[:max(0, beam_width - len(starts))])

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

        # Periodic greedy completions to tighten incumbent early
        def tighten_with_greedy(beam_list, take=2):
            nonlocal best_full_cost, best_full_seq
            if not beam_list:
                return
            beam_list_sorted = sorted(beam_list, key=lambda x: x[0])
            for idx in range(min(take, len(beam_list_sorted))):
                _, seq_pref, rem = beam_list_sorted[idx]
                if best_full_cost < float('inf') and evaluate_seq(seq_pref) >= best_full_cost:
                    continue
                g_cost, g_seq = greedy_complete(seq_pref, rem)
                if g_cost < best_full_cost:
                    best_full_cost, best_full_seq = g_cost, g_seq

        tighten_with_greedy(beam, take=1)

        for depth in range(n - 1):
            next_beam = []
            seen = set()

            for cost, seq, remaining in beam:
                # Incumbent pruning on partial
                if best_full_cost < float('inf') and cost >= best_full_cost:
                    continue

                rem_list = list(remaining)
                if not rem_list:
                    next_beam.append((cost, seq, remaining))
                    continue

                # Tournament-guided preselection
                if len(rem_list) > cand_per_expand * 2:
                    pre = preselect_by_tournament(seq, rem_list, cand_per_expand * 2)
                else:
                    pre = rem_list

                # Evaluate immediate cost of candidates
                imm = []
                for t in pre:
                    c_im = evaluate_seq(seq + [t])
                    imm.append((t, c_im))
                imm.sort(key=lambda x: x[1])

                # Lookahead over top few, guided by buddies then tournament
                L = min(lookahead_top, len(imm))
                scored_ext = []
                for t, immediate_c in imm[:L]:
                    next_pool_all = [x for x in rem_list if x != t]
                    if not next_pool_all:
                        la_cost = immediate_c
                    else:
                        B = [u for u in buddies[t] if u in next_pool_all]
                        if B:
                            # refine B by tournament preference near the current prefix
                            la_pref = preselect_by_tournament(seq + [t], B, min(lookahead_next_k, len(B)))
                        else:
                            la_pref = preselect_by_tournament(seq + [t], next_pool_all, min(lookahead_next_k, len(next_pool_all)))
                        if not la_pref:
                            la_pref = next_pool_all
                        la_cost = min(evaluate_seq(seq + [t, u]) for u in la_pref)
                    scored_ext.append((t, min(la_cost, immediate_c)))

                # Also add a few immediate-best without lookahead to maintain diversity
                diversity_take = min(max(2, cand_per_expand // 3), len(imm))
                for t, c_im in imm[:diversity_take]:
                    scored_ext.append((t, c_im))

                # Deduplicate and keep best-k expansions for this parent
                unique = {}
                for t, m in scored_ext:
                    if (t not in unique) or (m < unique[t]):
                        unique[t] = m
                items = sorted(unique.items(), key=lambda z: z[1])
                expand_k = min(cand_per_expand, len(items))
                for t, _metric in items[:expand_k]:
                    new_seq = seq + [t]
                    new_cost = evaluate_seq(new_seq)
                    # Prune by incumbent
                    if best_full_cost < float('inf') and new_cost >= best_full_cost:
                        continue
                    rem_new = remaining - {t}
                    key = (tuple(new_seq), rem_new)
                    if key in seen:
                        continue
                    seen.add(key)
                    next_beam.append((new_cost, new_seq, rem_new))

            # Fallback if sampling yielded nothing
            if not next_beam:
                for cost, seq, remaining in beam:
                    rem_list = list(remaining)
                    best_t = None
                    best_c = float('inf')
                    for t in rem_list:
                        c = evaluate_seq(seq + [t])
                        if c < best_c:
                            best_c = c
                            best_t = t
                    if best_t is not None:
                        new_seq = seq + [best_t]
                        if not (best_full_cost < float('inf') and best_c >= best_full_cost):
                            next_beam.append((best_c, new_seq, remaining - {best_t}))

            # Keep top beam_width partial sequences
            next_beam.sort(key=lambda x: x[0])
            beam = next_beam[:beam_width]

            # Periodically tighten incumbent with greedy completions
            if (depth + 1) % 5 == 0 or depth >= n - 3:
                tighten_with_greedy(beam, take=2)

        # Choose best complete (or near-complete) sequence from beam
        beam.sort(key=lambda x: x[0])
        best_cost, best_seq, _ = beam[0]
        # If we found a better full incumbent, use it
        if best_full_seq is not None and best_full_cost < best_cost:
            return best_full_cost, best_full_seq
        return best_cost, best_seq

    # Try multiple beam configurations (deterministic variants) and keep the best
    # The shared cache and incumbent amplify gains across restarts
    beam_variants = []
    base_bw = max(6, min(16, int(num_seqs)))
    # Construct 3 restarts with different trade-offs
    beam_variants.append((base_bw, max(8, min(16, n // 7 + 6)), 3, 6))
    beam_variants.append((max(8, base_bw - 2), max(10, min(18, n // 6 + 6)), 3, 5))
    beam_variants.append((max(6, base_bw // 2), max(14, min(20, n // 5 + 6)), 2, 6))

    best_beam_cost = float('inf')
    best_beam_seq = None
    for (bw, cpe, lt, ln) in beam_variants:
        c, s = beam_variant(bw, cpe, lt, ln)
        if c < best_beam_cost:
            best_beam_cost, best_beam_seq = c, s

    def local_improve(seq, base_cost=None):
        # Robust VNS: tournament cleanup, targeted boundary fixes, adjacent swaps,
        # relocations, limited swaps, ruin-and-recreate
        best_seq = list(seq)
        best_cost = evaluate_seq(best_seq) if base_cost is None else base_cost
        nloc = len(best_seq)

        # 0) Tournament-based cheap cleanup
        cand0 = tournament_bubble_pass(best_seq, passes=2)
        c0 = evaluate_seq(cand0)
        if c0 < best_cost:
            best_seq, best_cost = cand0, c0

        # 0b) Targeted fixes on most violated adjacent boundaries
        if nloc >= 2:
            violations = []
            for i in range(nloc - 1):
                a, b = best_seq[i], best_seq[i + 1]
                margin = M[a][b] - M[b][a]
                if margin > 0:
                    violations.append((margin, i))
            violations.sort(reverse=True)
            for _, i in violations[:3]:
                a, b = best_seq[i], best_seq[i + 1]
                # Try swap (adjacent)
                cand = best_seq[:]
                cand[i], cand[i + 1] = cand[i + 1], cand[i]
                c = evaluate_seq(cand)
                if c < best_cost:
                    best_seq, best_cost = cand, c
                    continue
                # Try pulling b backward further by 1-2 positions
                for back in (1, 2):
                    j = max(0, i - back)
                    cand2 = best_seq[:]
                    item = cand2.pop(i + 1)
                    cand2.insert(j, item)
                    c2 = evaluate_seq(cand2)
                    if c2 < best_cost:
                        best_seq, best_cost = cand2, c2
                        break

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
        relocate_attempts = min(2 * nloc, 180) + max(0, int(num_seqs // 3))
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

        # 3) Ruin-and-Recreate (block removal, greedy reinsertion with sampled positions)
        if nloc > 12:
            block_size = max(5, min(18, nloc // 6))
            rr_tries = 1 if nloc >= 90 else 2
            for _ in range(rr_tries):
                start = rng.randint(0, nloc - block_size)
                removed = best_seq[start:start + block_size]
                base = best_seq[:start] + best_seq[start + block_size:]

                for t in rng.sample(removed, len(removed)):
                    # Sample positions including ends
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

    # Run multiple beam variants to get a strong schedule, then refine locally
    beam_cost, beam_seq = best_beam_cost, best_beam_seq
    final_cost, final_seq = local_improve(beam_seq, base_cost=beam_cost)
    return final_cost, final_seq


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
    # Slightly larger beam for richer conflict structure
    makespan1, schedule1 = get_best_schedule(workload, 16)
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