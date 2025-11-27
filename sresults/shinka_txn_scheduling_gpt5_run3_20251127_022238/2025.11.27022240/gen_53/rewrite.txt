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
    FAS-guided global ranking + true-cost greedy build + local search and simulated annealing.

    Args:
        workload: Workload object containing transaction data
        num_seqs: Number of restarts/portfolio variants

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    N = workload.num_txns
    start_time = time.time()
    # Balanced budget to maintain good combined score
    time_budget_sec = 0.65

    def time_left():
        return (time.time() - start_time) < time_budget_sec

    rng = random.Random(1729 + 31 * N)

    # Cost oracle with memoization for sequences and prefix-extensions
    cost_cache = {}
    ext_cache = {}

    def eval_seq_cost(seq):
        key = tuple(seq)
        c = cost_cache.get(key)
        if c is not None:
            return c
        c = workload.get_opt_seq_cost(seq)
        cost_cache[key] = c
        return c

    def eval_ext_cost(prefix_tuple, cand):
        key = (prefix_tuple, cand)
        c = ext_cache.get(key)
        if c is not None:
            return c
        c = eval_seq_cost(list(prefix_tuple) + [cand])
        ext_cache[key] = c
        return c

    # Pairwise preference sampling: P[i][j] = c[j,i] - c[i,j] (positive => i before j is better)
    # Sparse representation
    P = defaultdict(dict)
    pair_cost_cache = {}

    def pair_cost(i, j):
        key = (i, j)
        v = pair_cost_cache.get(key)
        if v is not None:
            return v
        c = eval_seq_cost([i, j])
        pair_cost_cache[key] = c
        return c

    def sample_pairs():
        # Budget for pairwise probing
        # Aim ~O(N * k) samples; adapt k by N
        k = min(18, max(10, N // 8))
        max_pairs = min(1800, max(600, N * k))
        done = 0
        all_txns = list(range(N))
        # Bias sampling toward neighbors via random buckets
        while done < max_pairs and time_left():
            i = rng.randrange(N)
            # pick few opponents
            opps = set()
            # sample some deterministic near indices and randoms
            while len(opps) < min(k, N - 1) and time_left():
                j = rng.randrange(N)
                if j != i:
                    opps.add(j)
            for j in opps:
                if not time_left():
                    break
                if j in P.get(i, {}):
                    continue
                cij = pair_cost(i, j)
                cji = pair_cost(j, i)
                w = cji - cij  # positive: i before j
                P[i][j] = w
                P[j][i] = -w
                done += 2
                if done >= max_pairs or not time_left():
                    break

    sample_pairs()

    # Compute net scores from P (sum of outgoing weights)
    scores = [0.0] * N
    for i in range(N):
        s = 0.0
        row = P.get(i, {})
        for _, w in row.items():
            s += w
        scores[i] = s

    # Greedy insertion that minimizes surrogate P-violations
    def insertion_penalty(L, x, pos):
        # Penalty added when inserting x at position pos in list L
        # For y before pos (y before x after insertion): violation if P[x][y] > 0
        # For y at/after pos (x before y): violation if P[y][x] > 0
        pen = 0.0
        pref_x = P.get(x, {})
        for idx, y in enumerate(L):
            if idx < pos:
                w = pref_x.get(y)
                if w is not None and w > 0:
                    pen += w
            else:
                w = P.get(y, {}).get(x)
                if w is not None and w > 0:
                    pen += w
        return pen

    def build_order_by_fas(scores):
        order = []
        # Process items by descending score (most dominant first)
        items = list(range(N))
        items.sort(key=lambda i: -scores[i])
        for x in items:
            # find best position by minimal surrogate penalty
            best_pos = 0
            best_pen = float('inf')
            # Try a limited set of positions: ends + middle + around neighbors to keep fast
            candidates = set([0, len(order)])
            if len(order) > 0:
                mid = len(order) // 2
                candidates.add(mid)
                candidates.add(max(0, mid - 1))
                candidates.add(min(len(order), mid + 1))
                # try a few random positions
                for _ in range(min(4, len(order))):
                    candidates.add(rng.randrange(len(order) + 1))
            for pos in candidates:
                pen = insertion_penalty(order, x, pos)
                if pen < best_pen:
                    best_pen = pen
                    best_pos = pos
            order.insert(best_pos, x)
        return order

    fas_order = build_order_by_fas(scores)

    # Optional quick P-violation bubble improvements
    def fast_pairwise_bubble(seq, rounds=1):
        s = seq[:]
        n = len(s)
        for _ in range(rounds):
            improved = False
            for i in range(n - 1):
                a, b = s[i], s[i + 1]
                # If preference suggests b should be after a (i before j is good), check violation
                # Swap if P[b][a] is strongly positive (prefer b before a), which our order violates
                w = P.get(b, {}).get(a, 0.0)
                if w > 0:
                    s[i], s[i + 1] = s[i + 1], s[i]
                    improved = True
            if not improved:
                break
        return s

    fas_order = fast_pairwise_bubble(fas_order, rounds=1)

    # True-cost greedy builder guided by FAS order
    def build_seq_true_cost(order, window=8):
        rem = set(order)
        seq = []
        cur_cost = 0
        rank = {t: idx for idx, t in enumerate(order)}
        while rem and time_left():
            # pick next among top-ranked remaining (window)
            pool = sorted(rem, key=lambda t: rank[t])[:min(len(rem), window)]
            if not pool:
                pool = list(rem)
            prefix_tuple = tuple(seq)
            best_t = None
            best_c = float('inf')
            for t in pool:
                c = eval_ext_cost(prefix_tuple, t)
                if c < best_c:
                    best_c = c
                    best_t = t
            if best_t is None:
                # fallback arbitrary
                best_t = next(iter(rem))
                best_c = eval_ext_cost(prefix_tuple, best_t)
            seq.append(best_t)
            rem.remove(best_t)
            cur_cost = best_c
        if rem:
            # append rest arbitrarily if time ran out
            seq.extend(list(rem))
            cur_cost = eval_seq_cost(seq)
        else:
            cur_cost = eval_seq_cost(seq)
        return cur_cost, seq

    # Local refinements with true cost
    def local_refine(seq, start_cost, time_frac=0.22):
        # Allocate a fraction of remaining time
        end_t = time.time() + max(0.0, time_budget_sec * time_frac)
        def tleft():
            return time.time() < end_t and time_left()

        best_seq = seq[:]
        best_cost = start_cost
        n = len(best_seq)

        # Pass 1: Adjacent swap hill-climb
        passes = 0
        while tleft() and passes < 2:
            improved = False
            for i in range(n - 1):
                if not tleft():
                    break
                cand = best_seq[:]
                cand[i], cand[i + 1] = cand[i + 1], cand[i]
                c = eval_seq_cost(cand)
                if c < best_cost:
                    best_seq, best_cost = cand, c
                    improved = True
            passes += 1
            if not improved:
                break

        # Pass 2: Targeted relocations around worst P-violations
        if tleft():
            # identify top-k violating adjacencies by P
            viols = []
            for i in range(n - 1):
                w = P.get(best_seq[i + 1], {}).get(best_seq[i], 0.0)
                if w > 0:
                    viols.append((w, i))
            viols.sort(key=lambda x: -x[0])
            focus_idx = [i for _, i in viols[:min(8, len(viols))]]
            tries = 0
            max_tries = 60
            while tleft() and tries < max_tries:
                tries += 1
                if focus_idx and rng.random() < 0.7:
                    i = rng.choice(focus_idx) + (0 if rng.random() < 0.5 else 1)
                    i = max(0, min(n - 1, i))
                else:
                    i = rng.randrange(n)
                j = rng.randrange(n)
                if i == j:
                    continue
                cand = best_seq[:]
                val = cand.pop(i)
                cand.insert(j, val)
                c = eval_seq_cost(cand)
                if c < best_cost:
                    best_seq, best_cost = cand, c

        return best_cost, best_seq

    # Simulated annealing to escape local minima
    def simulated_annealing(seq, start_cost):
        best_seq = seq[:]
        best_cost = start_cost
        cur_seq = seq[:]
        cur_cost = start_cost
        n = len(cur_seq)

        # Estimate initial temperature from random moves
        deltas = []
        samples = min(25, max(10, n // 6))
        for _ in range(samples):
            i, j = rng.randrange(n), rng.randrange(n)
            if i == j:
                continue
            cand = cur_seq[:]
            if rng.random() < 0.5:
                # swap
                cand[i], cand[j] = cand[j], cand[i]
            else:
                # insert
                v = cand.pop(i)
                cand.insert(j, v)
            c = eval_seq_cost(cand)
            d = c - cur_cost
            if d > 0:
                deltas.append(d)
        T0 = (sum(deltas) / len(deltas)) if deltas else 1.0
        if T0 <= 1e-6:
            T0 = 1.0
        T = T0

        # Iteration budget bounded by time
        iter_limit = min(800, max(250, 10 * n))
        it = 0
        while time_left() and it < iter_limit:
            it += 1
            # cooling schedule
            alpha = 0.985
            if it % 20 == 0:
                T *= alpha

            cand = cur_seq[:]
            move = rng.random()
            if move < 0.5:
                # insertion
                i, j = rng.randrange(n), rng.randrange(n)
                if i != j:
                    v = cand.pop(i)
                    cand.insert(j, v)
            elif move < 0.8:
                # swap
                i, j = rng.randrange(n), rng.randrange(n)
                cand[i], cand[j] = cand[j], cand[i]
            else:
                # small 2-opt reversal
                a, b = sorted(rng.sample(range(n), 2))
                if b - a >= 3:
                    seg = cand[a:b]
                    seg.reverse()
                    cand = cand[:a] + seg + cand[b:]

            c = eval_seq_cost(cand)
            d = c - cur_cost
            accept = False
            if d <= 0:
                accept = True
            else:
                # Metropolis criterion
                if rng.random() < pow(2.718281828, -d / max(1e-9, T)):
                    accept = True
            if accept:
                cur_seq, cur_cost = cand, c
                if cur_cost < best_cost:
                    best_seq, best_cost = cur_seq[:], cur_cost

        return best_cost, best_seq

    # Portfolio of starts: FAS order, reversed FAS, random
    candidates = []

    if time_left():
        c1, s1 = build_seq_true_cost(fas_order, window=min(10, max(6, N // 10)))
        candidates.append((c1, s1))

    if time_left():
        fas_rev = fas_order[::-1]
        c2, s2 = build_seq_true_cost(fas_rev, window=min(10, max(6, N // 10)))
        candidates.append((c2, s2))

    if time_left():
        rnd = list(range(N))
        rng.shuffle(rnd)
        c3, s3 = build_seq_true_cost(rnd, window=min(10, max(6, N // 10)))
        candidates.append((c3, s3))

    # Keep the best two seeds for refinement
    candidates.sort(key=lambda x: x[0])
    candidates = candidates[:min(2, len(candidates))]

    global_best_cost = float('inf')
    global_best_seq = None

    for cost0, seq0 in candidates:
        if not time_left():
            break
        # Local deterministic refinement
        cost1, seq1 = local_refine(seq0, cost0, time_frac=0.18)
        # SA refinement
        if time_left():
            cost2, seq2 = simulated_annealing(seq1, cost1)
        else:
            cost2, seq2 = cost1, seq1
        if cost2 < global_best_cost:
            global_best_cost, global_best_seq = cost2, seq2

    # Safety: ensure permutation validity
    if global_best_seq is None or len(global_best_seq) != N or len(set(global_best_seq)) != N:
        if global_best_seq is None:
            seq = list(range(N))
            rng.shuffle(seq)
            global_best_seq = seq
        # Repair duplicates/missing
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