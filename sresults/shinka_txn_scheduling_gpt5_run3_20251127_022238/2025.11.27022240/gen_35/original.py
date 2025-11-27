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
    Tournament-guided greedy with buddy lookahead and boundary-focused LNS.

    Args:
        workload: Workload object containing transaction data
        num_seqs: Controls exploration budget

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    rng = random.Random()
    rng.seed(time.time_ns() ^ os.getpid())

    n = workload.num_txns

    # Global cache for sequence costs
    cost_cache = {}

    def eval_seq(seq):
        key = tuple(seq)
        if key in cost_cache:
            return cost_cache[key]
        c = workload.get_opt_seq_cost(seq)
        cost_cache[key] = c
        return c

    # Precompute singleton and pairwise costs to expose conflict structure
    c1 = [eval_seq([i]) for i in range(n)]
    M = [[0] * n for _ in range(n)]
    for i in range(n):
        Mi = M[i]
        for j in range(n):
            Mi[j] = c1[i] if i == j else eval_seq([i, j])

    # Preference margins W: negative suggests i before j
    W = [[0] * n for _ in range(n)]
    for i in range(n):
        Wi = W[i]
        Mi = M[i]
        for j in range(n):
            Wi[j] = 0 if i == j else (Mi[j] - M[j][i])

    # Tournament order (lower score -> earlier)
    s = [0] * n
    for i in range(n):
        s[i] = sum(W[i][j] for j in range(n) if j != i)
    tournament_order = list(range(n))
    tournament_order.sort(key=lambda x: (s[x], x))

    def prefer_before(a, b):
        return M[a][b] <= M[b][a]

    # Buddy lists: top partners by small M[t][u]
    buddy_k = 8 if n >= 90 else 6
    buddies = []
    for t in range(n):
        order = sorted((u for u in range(n) if u != t), key=lambda u: M[t][u])
        buddies.append(order[:buddy_k])

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

    # ----- Greedy constructor with tournament-guided pool and buddy lookahead -----
    def build_initial_sequence():
        pool_size = max(10, min(16, 3 + n // 8))
        reevaluate_every = 10  # refresh pool globally at intervals

        # Seed starting txn: choose among tournament-best and good singletons
        candidate_seeds = set()
        candidate_seeds.add(tournament_order[0])
        topk = min(10, n)
        for t, _ in sorted(enumerate(c1), key=lambda x: x[1])[:topk]:
            candidate_seeds.add(t)
        if len(candidate_seeds) < 3:
            candidate_seeds.update(rng.sample(range(n), min(3, n)))
        best_seed = min(candidate_seeds, key=lambda s: eval_seq([s]))

        seq = [best_seed]
        remaining = set(range(n))
        remaining.remove(best_seed)

        pool = set()

        def refresh_pool_full():
            nonlocal pool
            if len(remaining) <= pool_size * 2:
                cand_list = [(t, eval_seq(seq + [t])) for t in remaining]
                cand_list.sort(key=lambda x: x[1])
                pool = set([t for t, _ in cand_list[:pool_size]])
            else:
                # Tournament-guided preselection using W against recent prefix
                pre = preselect_by_tournament(seq, list(remaining), min(pool_size * 2, len(remaining)))
                cand_list = [(t, eval_seq(seq + [t])) for t in pre]
                cand_list.sort(key=lambda x: x[1])
                pool = set([t for t, _ in cand_list[:pool_size]])

        def top_pool_candidates():
            cand_list = [(t, eval_seq(seq + [t])) for t in list(pool)]
            cand_list.sort(key=lambda x: x[1])
            return cand_list

        # Initialize pool
        refresh_pool_full()

        step = 0
        while remaining:
            # Incumbent pruning against best seed so far (if available)
            try:
                if seed_best_cost < float('inf'):
                    base_cost = eval_seq(seq)
                    if base_cost >= seed_best_cost:
                        # Complete cheaply by immediate best to terminate early
                        rest = list(remaining)
                        rest.sort(key=lambda t: eval_seq(seq + [t]))
                        seq.extend(rest)
                        remaining.clear()
                        break
            except NameError:
                # seed_best_cost not yet defined in first call
                pass

            if step % reevaluate_every == 0:
                refresh_pool_full()
            else:
                while len(pool) < pool_size and remaining:
                    choices = list(remaining - pool)
                    if not choices:
                        break
                    # bias by tournament preference for diversity with guidance
                    add = preselect_by_tournament(seq, choices, 1) or [rng.choice(choices)]
                    pool.add(add[0])

            cand_list = top_pool_candidates()
            if not cand_list:
                pick = rng.choice(list(remaining))
                seq.append(pick)
                remaining.remove(pick)
                step += 1
                continue

            # Lookahead over top few with buddy-prioritized second step
            L = min(4, len(cand_list))
            best_t = cand_list[0][0]
            best_metric = cand_list[0][1]
            for t, immediate_c in cand_list[:L]:
                nexts = list(remaining - {t})
                if not nexts:
                    metric = immediate_c
                else:
                    # build la_pool = buddies ∩ nexts, then fill by tournament preference
                    buddy_pref = [u for u in buddies[t] if u in nexts]
                    need = 6 if n > 60 else 8
                    la_pool = buddy_pref[:need]
                    if len(la_pool) < need:
                        extra = preselect_by_tournament(seq + [t], [u for u in nexts if u not in la_pool], need - len(la_pool), recent_k=3)
                        la_pool.extend(extra)
                    if not la_pool:
                        la_pool = nexts[:min(need, len(nexts))]
                    metric = min(eval_seq(seq + [t, u]) for u in la_pool)
                if metric < best_metric:
                    best_metric = metric
                    best_t = t

            seq.append(best_t)
            remaining.remove(best_t)
            if best_t in pool:
                pool.remove(best_t)
            step += 1

        return seq

    # Local adjacent swap hill-climb
    def adjacent_pass(seq, current_cost):
        improved = True
        n_local = len(seq)
        while improved:
            improved = False
            for i in range(n_local - 1):
                if seq[i] == seq[i + 1]:
                    continue
                cand = seq[:]
                cand[i], cand[i + 1] = cand[i + 1], cand[i]
                c = eval_seq(cand)
                if c < current_cost:
                    seq = cand
                    current_cost = c
                    improved = True
        return seq, current_cost

    # Compute prefix costs and marginal contributions for a sequence
    def prefix_marginals(seq):
        prefix_costs = [0] * len(seq)
        c = 0
        for i in range(len(seq)):
            c = eval_seq(seq[: i + 1])
            prefix_costs[i] = c
        marg = [prefix_costs[0]] + [prefix_costs[i] - prefix_costs[i - 1] for i in range(1, len(seq))]
        return prefix_costs, marg

    def worst_violation_boundaries(seq, topm=3):
        pairs = []
        for i in range(len(seq) - 1):
            a, b = seq[i], seq[i + 1]
            pen = M[a][b] - M[b][a]  # positive when a before b is worse
            if pen > 0:
                pairs.append((pen, i))
        pairs.sort(reverse=True)
        return [i for _, i in pairs[:topm]]

    # LNS: pick hot windows by marginal sum and boundary violations; reorder via perms
    def lns_improve(seq, base_cost, budget_factor):
        best_seq = seq[:]
        best_cost = base_cost
        n_local = len(best_seq)

        # Iterations scale mildly with budget_factor and n
        iters = max(4, min(10, 2 + int(budget_factor) + n_local // 40))
        max_k = 7 if n_local >= 40 else 6

        for it in range(iters):
            # Compute marginals to locate hot regions
            _, marg = prefix_marginals(best_seq)

            # Choose window size
            k = max_k if it < 2 else rng.randint(4, max_k)
            if n_local <= k:
                break

            # Find top windows by marginal sum (sliding window)
            sums = []
            window_sum = sum(marg[0:k])
            sums.append((0, window_sum))
            for s in range(1, n_local - k + 1):
                window_sum += marg[s + k - 1] - marg[s - 1]
                sums.append((s, window_sum))
            sums.sort(key=lambda x: -x[1])

            # Also consider windows around worst violated adjacencies
            viol_starts = worst_violation_boundaries(best_seq, topm=3)
            candidate_starts = []
            for v in viol_starts:
                start = max(0, min(v - (k // 2), n_local - k))
                candidate_starts.append(start)
            candidate_starts.extend([start for start, _ in sums[:2]])
            # Deduplicate while preserving order
            seen_starts = set()
            ordered_starts = []
            for st in candidate_starts:
                if st not in seen_starts:
                    seen_starts.add(st)
                    ordered_starts.append(st)

            tried_any = False
            for start in ordered_starts:
                block = best_seq[start : start + k]
                base = best_seq[:start] + best_seq[start + k :]

                # Determine permutation budget
                factorial = 1
                for i in range(2, k + 1):
                    factorial *= i
                # Cap permutations to keep time in check
                if k <= 6:
                    perm_budget = min(720, factorial)
                else:
                    perm_budget = 2000  # sample for k=7

                perm_best_seq = None
                perm_best_cost = best_cost

                if factorial <= perm_budget:
                    for p in itertools.permutations(block):
                        cand_seq = base[:start] + list(p) + base[start:]
                        c = eval_seq(cand_seq)
                        if c < perm_best_cost:
                            perm_best_cost = c
                            perm_best_seq = cand_seq
                else:
                    seenp = set()
                    attempts = 0
                    while attempts < perm_budget:
                        p = tuple(rng.sample(block, k))
                        if p in seenp:
                            continue
                        seenp.add(p)
                        cand_seq = base[:start] + list(p) + base[start:]
                        c = eval_seq(cand_seq)
                        if c < perm_best_cost:
                            perm_best_cost = c
                            perm_best_seq = cand_seq
                        attempts += 1

                if perm_best_seq is not None and perm_best_cost < best_cost:
                    best_seq = perm_best_seq
                    best_cost = perm_best_cost
                    tried_any = True
                    break  # accept first improving window

            # Targeted relocate moves for top-blame transactions
            _, marg = prefix_marginals(best_seq)
            positions = sorted(range(n_local), key=lambda i: marg[i], reverse=True)[:3]
            for pos in positions:
                if pos >= len(best_seq):
                    continue
                t = best_seq[pos]
                base = best_seq[:pos] + best_seq[pos + 1 :]
                best_pos_cost = best_cost
                best_pos_idx = None
                positions_try = set([0, len(base)])
                for _ in range(8):
                    positions_try.add(rng.randint(0, len(base)))
                for j in positions_try:
                    cand = base[:j] + [t] + base[j:]
                    c = eval_seq(cand)
                    if c < best_pos_cost:
                        best_pos_cost = c
                        best_pos_idx = j
                if best_pos_idx is not None and best_pos_cost < best_cost:
                    best_seq = base[:best_pos_idx] + [t] + base[best_pos_idx:]
                    best_cost = best_pos_cost

            if not tried_any:
                continue

        return best_seq, best_cost

    # Multi-start: build a few diverse initial sequences and pick the best as LNS seed
    starts = max(2, min(5, int(num_seqs // 3) + 2))
    seed_best_cost = float('inf')
    seed_best_seq = None
    for _ in range(starts):
        seq0 = build_initial_sequence()
        cost0 = eval_seq(seq0)
        # quick local cleanup
        seq0, cost0 = adjacent_pass(seq0, cost0)
        if cost0 < seed_best_cost:
            seed_best_cost = cost0
            seed_best_seq = seq0

    # LNS improvement on best seed
    final_seq, final_cost = lns_improve(seed_best_seq, seed_best_cost, budget_factor=max(1, int(num_seqs)))

    # Small non-adjacent 2-opt pass
    two_opt_trials = min(60, n)
    for _ in range(two_opt_trials):
        i, j = rng.sample(range(n), 2)
        if abs(i - j) <= 1:
            continue
        cand = final_seq[:]
        cand[i], cand[j] = cand[j], cand[i]
        c = eval_seq(cand)
        if c < final_cost:
            final_seq = cand
            final_cost = c

    # Final light random insertion moves
    move_budget = min(60, n)
    for _ in range(move_budget):
        i, j = rng.sample(range(n), 2)
        if i == j:
            continue
        cand = final_seq[:]
        item = cand.pop(i)
        cand.insert(j, item)
        c = eval_seq(cand)
        if c < final_cost:
            final_seq = cand
            final_cost = c

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
    makespan1, schedule1 = get_best_schedule(workload, 12)
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