# EVOLVE-BLOCK-START
"""Transaction scheduling algorithm for optimizing makespan across multiple workloads
Conflict-aware divide-and-merge with DP interleaving and light polishing.
"""

import time
import random
import sys
import os
from collections import defaultdict, deque
from functools import lru_cache

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
    Conflict-aware divide-and-merge scheduling with DP interleaving.

    Args:
        workload: Workload object containing transaction data
        num_seqs: Search effort parameter (scales clustering tightness and local beam)

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    n = workload.num_txns
    txns = list(range(n))

    # Seed RNG deterministically per call with slight variability
    random.seed((n * 911 + num_seqs * 131 + 17) % (2**32 - 1))

    # ---------------------- Cost evaluator with cache ----------------------
    @lru_cache(maxsize=400_000)
    def _eval_cost_tuple(seq_tuple):
        return workload.get_opt_seq_cost(list(seq_tuple))

    def eval_cost(seq):
        return _eval_cost_tuple(tuple(seq))

    # ---------------------- Conflict graph estimation ----------------------
    # Use only short-prefix cost probes; no access to workload internals required.
    # weight(i,j) ~ max( increase placing j after i, increase placing i after j ), floored at 0
    # For large n, sample partners to keep evaluation budget in check.

    # Precompute singleton costs
    singleton_cost = [eval_cost([i]) for i in txns]

    # Determine estimation regime
    full_pairs = n <= 80
    sample_per_txn = 28 if n > 120 else (32 if n > 80 else None)

    # Conflict weights (undirected, non-negative) and directional preference scores
    # pref_dir[i][j] > 0 means "prefer i before j" by that margin
    adj = [defaultdict(float) for _ in range(n)]
    pref_dir = [defaultdict(float) for _ in range(n)]

    if full_pairs:
        for i in range(n):
            for j in range(i + 1, n):
                c_ij = eval_cost([i, j])
                c_ji = eval_cost([j, i])
                d_ij = c_ij - singleton_cost[i]
                d_ji = c_ji - singleton_cost[j]
                w = max(d_ij, d_ji, 0.0)
                if w > 0.0:
                    adj[i][j] += w
                    adj[j][i] += w
                pref = d_ji - d_ij  # positive => prefer i before j
                if abs(pref) > 0.0:
                    pref_dir[i][j] += pref
                    pref_dir[j][i] -= pref
    else:
        # Sampled estimation for large n
        for i in range(n):
            partners = list(range(n))
            partners.remove(i)
            k = min(sample_per_txn, len(partners))
            sample = random.sample(partners, k) if k and k < len(partners) else partners
            for j in sample:
                if j < i:
                    # avoid recomputing if already seen from (j,i) in its sample
                    continue
                c_ij = eval_cost([i, j])
                c_ji = eval_cost([j, i])
                d_ij = c_ij - singleton_cost[i]
                d_ji = c_ji - singleton_cost[j]
                w = max(d_ij, d_ji, 0.0)
                if w > 0.0:
                    adj[i][j] += w
                    adj[j][i] += w
                pref = d_ji - d_ij
                if abs(pref) > 0.0:
                    pref_dir[i][j] += pref
                    pref_dir[j][i] -= pref

    degree = [sum(adj[i].values()) for i in range(n)]

    # ---------------------- Greedy conflict-aware clustering ----------------------
    # Build compact clusters around high-degree seeds, growing by maximal internal gain.
    if n <= 50:
        max_cluster = 12
    elif n <= 90:
        max_cluster = 14
    else:
        max_cluster = 16

    # Allow a bit larger clusters if num_seqs is high
    max_cluster = min(max_cluster + (1 if num_seqs >= 10 else 0), max_cluster + 2)

    unassigned = set(txns)
    clusters = []

    while unassigned:
        # Pick the highest-degree unassigned vertex as a seed
        seed = max(unassigned, key=lambda x: degree[x])
        cluster = [seed]
        unassigned.remove(seed)

        # Grow by adding the node with the largest sum of weights to current cluster
        while len(cluster) < max_cluster and unassigned:
            best_u = None
            best_gain = 0.0
            # Candidate pool: neighbors of cluster or any unassigned if sparse
            cand_pool = set()
            for c in cluster:
                cand_pool.update(set(adj[c].keys()))
            cand_pool = cand_pool.intersection(unassigned)
            if not cand_pool:
                # Fallback: consider all unassigned but with tiny gain
                cand_pool = set(unassigned)

            for u in cand_pool:
                gain = sum(adj[u][c] for c in cluster if c in adj[u])
                if gain > best_gain:
                    best_gain = gain
                    best_u = u

            # Add only if it contributes; allow minor additions if cluster is too small
            if best_u is None or (best_gain <= 0 and len(cluster) >= max(3, max_cluster // 2)):
                break
            cluster.append(best_u)
            unassigned.remove(best_u)

        clusters.append(cluster)

    # ---------------------- Order each cluster locally ----------------------
    # Use a small beam search restricted to cluster elements to get a robust local order.

    def cluster_order_beam(cluster):
        # Initial seed: choose best singleton prefix by global prefix cost
        starters = [(eval_cost([t]), [t]) for t in cluster]
        starters.sort(key=lambda x: x[0])
        beam = starters[:min(4, len(starters))]

        # Beam params (small, cluster-limited)
        B = min(80, max(30, 12 + 6 * num_seqs))
        lookahead_k = 5
        alpha = 0.65

        remaining_all = set(cluster)
        for _ in range(1, len(cluster) + 1):
            next_pool = []
            seen = set()
            for c_pref, prefix in beam:
                rem = list(remaining_all.difference(prefix))
                if not rem:
                    next_pool.append((c_pref, prefix))
                    continue

                # Expand each remaining candidate (endgame widen)
                expand = rem if len(rem) <= 10 else random.sample(rem, 10)
                scored = []
                for t in expand:
                    p1 = prefix + [t]
                    c1 = eval_cost(p1)
                    # lookahead
                    rem_after = [x for x in rem if x != t]
                    if rem_after:
                        k2 = min(lookahead_k, len(rem_after))
                        sample2 = rem_after if len(rem_after) <= k2 else random.sample(rem_after, k2)
                        best_c2 = float('inf')
                        for u in sample2:
                            cu = eval_cost(p1 + [u])
                            if cu < best_c2:
                                best_c2 = cu
                        score = alpha * c1 + (1 - alpha) * best_c2
                    else:
                        score = c1
                    scored.append((score, c1, p1))

                scored.sort(key=lambda x: x[0])
                # Keep top few per parent to encourage diversity
                keep_k = min(5, len(scored))
                for k in range(keep_k):
                    c1 = scored[k][1]
                    p1 = scored[k][2]
                    key = tuple(p1)
                    if key in seen:
                        continue
                    seen.add(key)
                    next_pool.append((c1, p1))

            if not next_pool:
                break
            next_pool.sort(key=lambda x: x[0])
            beam = next_pool[:min(B, len(next_pool))]

        if not beam:
            # Fallback: simple greedy insertion constrained to cluster
            seq = []
            rem = list(cluster)
            while rem:
                best_t = None
                best_c = float('inf')
                for t in rem:
                    c = eval_cost(seq + [t])
                    if c < best_c:
                        best_c = c
                        best_t = t
                seq.append(best_t)
                rem.remove(best_t)
            return seq

        best = min(beam, key=lambda x: x[0])[1]
        return best

    cluster_orders = [cluster_order_beam(c) if len(c) > 1 else c[:] for c in clusters]

    # ---------------------- DP merge: exact interleaving of two sequences ----------------------
    def merge_two_orders_dp(a, b):
        # dp[i][j] = (cost, seq_list) for best interleaving of a[:i] and b[:j]
        la, lb = len(a), len(b)
        # Using list of dicts to keep memory modest
        dp_cost = [[float('inf')] * (lb + 1) for _ in range(la + 1)]
        dp_seq = [[None] * (lb + 1) for _ in range(la + 1)]

        dp_cost[0][0] = 0.0
        dp_seq[0][0] = []

        # Initialize first row/column
        for i in range(1, la + 1):
            seq = dp_seq[i - 1][0] + [a[i - 1]]
            dp_seq[i][0] = seq
            dp_cost[i][0] = eval_cost(seq)
        for j in range(1, lb + 1):
            seq = dp_seq[0][j - 1] + [b[j - 1]]
            dp_seq[0][j] = seq
            dp_cost[0][j] = eval_cost(seq)

        # Fill DP
        for i in range(1, la + 1):
            for j in range(1, lb + 1):
                # Option 1: take next from a
                seq1 = dp_seq[i - 1][j] + [a[i - 1]]
                c1 = eval_cost(seq1)
                # Option 2: take next from b
                seq2 = dp_seq[i][j - 1] + [b[j - 1]]
                c2 = eval_cost(seq2)

                if c1 <= c2:
                    dp_cost[i][j] = c1
                    dp_seq[i][j] = seq1
                else:
                    dp_cost[i][j] = c2
                    dp_seq[i][j] = seq2

        return dp_seq[la][lb], dp_cost[la][lb]

    # ---------------------- Select merge order across clusters ----------------------
    # Start with the most internally contentious cluster (highest internal degree sum)
    def internal_weight(cluster):
        return sum(degree[t] for t in cluster)

    # Total cross-weight of cluster to a set of nodes
    def cross_weight(cluster, node_set):
        s = 0.0
        for u in cluster:
            for v in node_set:
                s += adj[u].get(v, 0.0)
        return s

    # Merge clusters iteratively, trying the best few by cross-conflict at each step
    remaining_idx = list(range(len(cluster_orders)))
    remaining_idx.sort(key=lambda idx: internal_weight(clusters[idx]), reverse=True)

    merged_seq = cluster_orders[remaining_idx[0]][:]
    merged_cost = eval_cost(merged_seq)
    merged_set = set(merged_seq)
    used = {remaining_idx[0]}

    # How many candidates to try at each merge step (lookahead breadth)
    try_top_k = 2 if len(cluster_orders) > 4 else len(cluster_orders)

    while len(used) < len(cluster_orders):
        candidates = []
        for idx in remaining_idx:
            if idx in used:
                continue
            cw = cross_weight(clusters[idx], merged_set)
            candidates.append((cw, idx))
        # Pick top few by cross weight (most interacting with current merged set)
        candidates.sort(key=lambda x: x[0], reverse=True)
        consider = [idx for _, idx in candidates[:max(1, try_top_k)]]

        best_merge_cost = float('inf')
        best_merge_seq = None
        best_idx = None

        for idx in consider:
            seq_cand, cost_cand = merge_two_orders_dp(merged_seq, cluster_orders[idx])
            if cost_cand < best_merge_cost:
                best_merge_cost = cost_cand
                best_merge_seq = seq_cand
                best_idx = idx

        if best_idx is None:
            # Fallback: just append the remaining cluster with DP to ensure feasibility
            best_idx = next(idx for idx in remaining_idx if idx not in used)
            best_merge_seq, best_merge_cost = merge_two_orders_dp(merged_seq, cluster_orders[best_idx])

        merged_seq = best_merge_seq
        merged_cost = best_merge_cost
        merged_set.update(clusters[best_idx])
        used.add(best_idx)

    # ---------------------- Light polishing (small Or-opt + adjacent swaps) ----------------------
    def local_adjacent_swaps(seq, curr_cost, max_passes=2):
        best_seq = list(seq)
        best_cost = curr_cost
        for _ in range(max_passes):
            improved = False
            for i in range(len(best_seq) - 1):
                cand = best_seq[:]
                cand[i], cand[i + 1] = cand[i + 1], cand[i]
                c = eval_cost(cand)
                if c < best_cost:
                    best_cost = c
                    best_seq = cand
                    improved = True
            if not improved:
                break
        return best_seq, best_cost

    def local_oropt_small(seq, curr_cost, block_sizes=(1, 2)):
        best_seq = list(seq)
        best_cost = curr_cost
        n_local = len(best_seq)
        for blk in block_sizes:
            if blk >= n_local:
                continue
            improved = True
            while improved:
                improved = False
                for i in range(0, len(best_seq) - blk + 1):
                    block = best_seq[i:i + blk]
                    base = best_seq[:i] + best_seq[i + blk:]
                    m = len(base) + 1
                    # Try a few positions; all around the original and ends
                    positions = set([0, m - 1])
                    if m > 4:
                        positions.update({max(0, i - 1), min(m - 1, i + 1), m // 2})
                    for pos in sorted(positions):
                        cand = base[:]
                        cand[pos:pos] = block
                        c = eval_cost(cand)
                        if c < best_cost:
                            best_cost = c
                            best_seq = cand
                            improved = True
                            break
                    if improved:
                        break
        return best_seq, best_cost

    seq_final, cost_final = local_oropt_small(merged_seq, merged_cost, block_sizes=(2, 1))
    seq_final, cost_final = local_adjacent_swaps(seq_final, cost_final, max_passes=2)

    # Safety checks
    assert len(seq_final) == n and len(set(seq_final)) == n, "Schedule must include each transaction exactly once"

    return cost_final, seq_final


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