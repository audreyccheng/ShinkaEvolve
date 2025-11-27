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
    Tournament-guided greedy with buddy-list lookahead and VNS refinement.
    Uses pairwise preferences (W) to minimize conflict-induced delays and
    memoization to reduce simulator calls.

    Args:
        workload: Workload object containing transaction data
        num_seqs: Exploration budget; controls restarts and local search intensity

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    # Deterministic RNG per workload for stable results
    n = workload.num_txns
    rng = random.Random(1729 + n)

    # Global memoized evaluator for partial sequences to reduce simulator calls
    cost_cache = {}

    def eval_cost(seq):
        key = tuple(seq)
        if key in cost_cache:
            return cost_cache[key]
        c = workload.get_opt_seq_cost(seq)
        cost_cache[key] = c
        return c

    # Precompute singleton and pairwise costs to capture conflict structure
    c1 = [eval_cost([i]) for i in range(n)]
    M = [[0] * n for _ in range(n)]  # M[i][j] = cost([i, j])
    for i in range(n):
        Mi = M[i]
        for j in range(n):
            Mi[j] = c1[i] if i == j else eval_cost([i, j])

    # Preference margins: W[i][j] < 0 suggests i before j
    W = [[0] * n for _ in range(n)]
    for i in range(n):
        Wi = W[i]
        Mi = M[i]
        for j in range(n):
            Wi[j] = 0 if i == j else (Mi[j] - M[j][i])

    # Tournament scores: smaller => earlier
    s = [0] * n
    for i in range(n):
        s[i] = sum(W[i][j] for j in range(n) if j != i)

    tournament_order = list(range(n))
    tournament_order.sort(key=lambda x: (s[x], x))

    def prefer_before(a, b):
        # True if placing a before b is no worse than b before a
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

    # Precompute singletons for alternative seeds
    singleton = [(t, c1[t]) for t in range(n)]
    singleton.sort(key=lambda x: x[1])

    # Buddy lists: for each t, top partners by small M[t][u]
    buddy_k = 8 if n >= 90 else 6
    buddies = []
    for t in range(n):
        order = sorted((u for u in range(n) if u != t), key=lambda u: M[t][u])
        buddies.append(order[:buddy_k])

    # Budgeting knobs derived from num_seqs and problem size
    restarts = max(4, min(int(num_seqs) + 2, 10 if n >= 90 else 14))

    # Candidate selection and lookahead settings
    cand_pool_large = 12 if n >= 90 else 16
    small_threshold = max(35, n // 3)
    lookahead_top = 3
    lookahead_samples_small = 6
    lookahead_samples_large = 4

    best_global_cost = float('inf')
    best_global_seq = None

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

    def greedy_build(seed_choice):
        # Seed selection
        if seed_choice == "tournament":
            t0 = tournament_order[0]
        elif seed_choice == "best":
            k = min(10, n)
            t0 = rng.choice([t for t, _ in singleton[:k]])
        elif seed_choice == "median":
            k = min(30, n)
            t0 = rng.choice([t for t, _ in singleton[k // 2:k]])
        else:
            t0 = rng.randint(0, n - 1)

        seq = [t0]
        remaining = [t for t in range(n) if t != t0]

        # Greedy build with tournament-guided preselection and buddy lookahead
        step = 0
        while remaining:
            # Incumbent-based pruning: stop this restart if already worse
            base_cost = eval_cost(seq)
            if best_global_cost < float('inf') and base_cost >= best_global_cost:
                return float('inf'), None

            R = len(remaining)

            # Candidate pool: all remaining if small, else tournament-guided preselection
            if R <= small_threshold:
                cand_pool = remaining[:]
            else:
                k = min(cand_pool_large, R)
                cand_pool = preselect_by_tournament(seq, remaining, k)

            # Evaluate immediate costs for candidate pool
            immediate = []
            base = seq
            for t in cand_pool:
                c = eval_cost(base + [t])
                immediate.append((t, c))
            immediate.sort(key=lambda x: x[1])

            # Take top few for lookahead using buddy lists
            L = min(lookahead_top, len(immediate))
            chosen_t = immediate[0][0]
            best_pair_eval = immediate[0][1]

            for idx in range(L):
                t, immediate_c = immediate[idx]
                if R == 1:
                    la_cost = immediate_c
                else:
                    next_pool_all = [x for x in remaining if x != t]
                    if not next_pool_all:
                        la_cost = immediate_c
                    else:
                        # Prioritize buddies, then tournament-preferred others
                        buddy_pref = [u for u in buddies[t] if u in next_pool_all]
                        need = lookahead_samples_large if R > small_threshold else lookahead_samples_small
                        if len(buddy_pref) < need:
                            extra_needed = need - len(buddy_pref)
                            extra = preselect_by_tournament(base + [t], [u for u in next_pool_all if u not in buddy_pref], extra_needed, recent_k=3)
                            la_pool = buddy_pref + extra
                        else:
                            la_pool = buddy_pref[:need]
                        if not la_pool:
                            la_pool = next_pool_all[:min(need, len(next_pool_all))]
                        la_cost = min(eval_cost(base + [t, u]) for u in la_pool)
                if la_cost < best_pair_eval:
                    best_pair_eval = la_cost
                    chosen_t = t

            seq.append(chosen_t)
            remaining.remove(chosen_t)
            step += 1

        # Cheap tournament cleanup
        seq_t = tournament_bubble_pass(seq, passes=2)
        c_seq = eval_cost(seq)
        c_seq_t = eval_cost(seq_t)
        return (c_seq_t, seq_t) if c_seq_t < c_seq else (c_seq, seq)

    def local_improve(seq, base_cost=None):
        # VNS refinement:
        # - Tournament bubble pass
        # - Adjacent swaps
        # - Limited 2-opt (non-adjacent) swaps
        # - Sampled insertion (relocate)
        # - Ruin-and-recreate
        best_seq = seq[:]
        best_cost = eval_cost(best_seq) if base_cost is None else base_cost
        nloc = len(best_seq)

        # Tournament cleanup
        cand0 = tournament_bubble_pass(best_seq, passes=2)
        c0 = eval_cost(cand0)
        if c0 < best_cost:
            best_seq, best_cost = cand0, c0

        # 1) Adjacent swap hill-climb passes
        for _ in range(3):
            improved = False
            for i in range(nloc - 1):
                cand = best_seq[:]
                cand[i], cand[i + 1] = cand[i + 1], cand[i]
                c = eval_cost(cand)
                if c < best_cost:
                    best_seq, best_cost = cand, c
                    improved = True
            if not improved:
                break

        # 1b) Limited non-adjacent 2-opt swaps
        swap_attempts = min(nloc, 80)
        for _ in range(swap_attempts):
            i, j = rng.sample(range(nloc), 2)
            if abs(i - j) <= 1:
                continue
            cand = best_seq[:]
            cand[i], cand[j] = cand[j], cand[i]
            c = eval_cost(cand)
            if c < best_cost:
                best_seq, best_cost = cand, c

        # Helper: compute prefix marginals
        def prefix_marginals(seq_local):
            prefix_costs = [0] * len(seq_local)
            c = 0
            for idx in range(len(seq_local)):
                c = eval_cost(seq_local[: idx + 1])
                prefix_costs[idx] = c
            marg = [prefix_costs[0]] + [prefix_costs[i] - prefix_costs[i - 1] for i in range(1, len(seq_local))]
            return marg

        # 2) Sampled insertion (relocate) moves
        relocate_attempts = min(2 * nloc, 180) + max(0, int(num_seqs // 3))
        for _ in range(relocate_attempts):
            i, j = rng.sample(range(nloc), 2)
            if i == j:
                continue
            cand = best_seq[:]
            item = cand.pop(j)
            cand.insert(i, item)
            c = eval_cost(cand)
            if c < best_cost:
                best_seq, best_cost = cand, c

        # 2b) Targeted relocate for top-marginal positions
        if nloc >= 6:
            marg = prefix_marginals(best_seq)
            hot_positions = sorted(range(nloc), key=lambda idx: marg[idx], reverse=True)[:3]
            for pos in hot_positions:
                if pos >= len(best_seq):
                    continue
                t = best_seq[pos]
                base = best_seq[:pos] + best_seq[pos + 1:]
                best_pos_cost = best_cost
                best_pos_idx = None
                positions_try = {0, len(base)}
                while len(positions_try) < 6:
                    positions_try.add(rng.randint(0, len(base)))
                for j in positions_try:
                    cand = base[:j] + [t] + base[j:]
                    c = eval_cost(cand)
                    if c < best_pos_cost:
                        best_pos_cost = c
                        best_pos_idx = j
                if best_pos_idx is not None and best_pos_cost < best_cost:
                    best_seq = base[:best_pos_idx] + [t] + base[best_pos_idx:]
                    best_cost = best_pos_cost

        # 3) Light ruin-and-recreate with greedy reinsertion
        if nloc > 12:
            block_size = max(5, min(18, nloc // 6))
            rr_tries = 1 if nloc >= 90 else 2
            for _ in range(rr_tries):
                start = rng.randint(0, nloc - block_size)
                removed = best_seq[start:start + block_size]
                base = best_seq[:start] + best_seq[start + block_size:]

                for t in rng.sample(removed, len(removed)):
                    pos_candidates = {0, len(base)}
                    sample_k = min(6, len(base) + 1)
                    while len(pos_candidates) < sample_k:
                        pos_candidates.add(rng.randint(0, len(base)))
                    best_pos = 0
                    best_pos_cost = float('inf')
                    for pos in pos_candidates:
                        cand = base[:pos] + [t] + base[pos:]
                        c = eval_cost(cand)
                        if c < best_pos_cost:
                            best_pos_cost = c
                            best_pos = pos
                    base.insert(best_pos, t)

                c = eval_cost(base)
                if c < best_cost:
                    best_seq, best_cost = base, c

        return best_cost, best_seq

    # Multi-start: diversify seeds between tournament, good singletons and randomness
    start_modes = ["tournament", "best", "median", "random"]
    for r in range(restarts):
        mode = start_modes[r % len(start_modes)]
        g_cost, g_seq = greedy_build(mode)
        if g_seq is None:
            continue
        l_cost, l_seq = local_improve(g_seq, base_cost=g_cost)
        if l_cost < best_global_cost:
            best_global_cost, best_global_seq = l_cost, l_seq

    return best_global_cost, best_global_seq


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
    # Slightly larger exploration for richer conflict structure
    makespan1, schedule1 = get_best_schedule(workload, 15)
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