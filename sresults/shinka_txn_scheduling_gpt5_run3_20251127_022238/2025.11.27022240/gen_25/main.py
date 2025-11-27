# EVOLVE-BLOCK-START
"""Transaction scheduling algorithm for optimizing makespan across multiple workloads"""

import time
import random
import sys
import os
import math
from collections import deque, defaultdict

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
    Cluster-guided Monte Carlo Tree Search (MCTS) with UCT selection, W/Buddy preselection,
    and incumbent pruning. Rollouts use cheap W-guided completions; only the final schedule
    is evaluated by the simulator per simulation, drastically reducing expensive calls.

    Args:
        workload: Workload object containing transaction data
        num_seqs: Exploration budget controlling iterations and restarts

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    n = workload.num_txns
    rng = random.Random(1729 + n)  # deterministic per workload

    # Global memoized evaluator for partial and full sequences
    cost_cache = {}

    def evaluate_seq(seq):
        key = tuple(seq)
        if key in cost_cache:
            return cost_cache[key]
        c = workload.get_opt_seq_cost(seq)
        cost_cache[key] = c
        return c

    # Precompute singleton and pairwise costs (M), preference margins (W), and tournament scores (s)
    c1 = [evaluate_seq([i]) for i in range(n)]
    M = [[0] * n for _ in range(n)]
    for i in range(n):
        Mi = M[i]
        for j in range(n):
            if i == j:
                Mi[j] = c1[i]
            else:
                Mi[j] = evaluate_seq([i, j])

    W = [[0] * n for _ in range(n)]
    for i in range(n):
        Wi = W[i]
        Mi = M[i]
        for j in range(n):
            if i == j:
                Wi[j] = 0
            else:
                Wi[j] = Mi[j] - M[j][i]

    s = [0] * n
    for i in range(n):
        s[i] = sum(W[i][j] for j in range(n) if j != i)

    tournament_order = list(range(n))
    tournament_order.sort(key=lambda x: (s[x], x))

    def prefer_before(a, b):
        # True if placing a before b is no worse than b before a
        return M[a][b] <= M[b][a]

    def tournament_bubble_pass(seq, passes=2):
        # Cheap W/M-based cleanup without simulator calls
        arr = list(seq)
        for _ in range(passes):
            improved = False
            for k in range(len(arr) - 1):
                a, b = arr[k], arr[k + 1]
                if not prefer_before(a, b):
                    arr[k], arr[k + 1] = arr[k + 1], arr[k]
                    improved = True
            if not improved:
                break
        return arr

    # Build conflict clusters using strong W edges (undirected graph at high |W|)
    abs_vals = [abs(W[i][j]) for i in range(n) for j in range(n) if i != j]
    tau = 0
    if abs_vals:
        abs_vals.sort()
        # 70th percentile threshold
        idx = int(0.70 * (len(abs_vals) - 1))
        tau = abs_vals[idx]
    # Build adjacency list
    adj = [[] for _ in range(n)]
    if tau > 0:
        for i in range(n):
            for j in range(i + 1, n):
                if abs(W[i][j]) >= tau:
                    adj[i].append(j)
                    adj[j].append(i)

    # Extract connected components
    visited = [False] * n
    clusters = []
    for i in range(n):
        if not visited[i]:
            comp = []
            dq = deque([i])
            visited[i] = True
            while dq:
                u = dq.popleft()
                comp.append(u)
                for v in adj[u]:
                    if not visited[v]:
                        visited[v] = True
                        dq.append(v)
            clusters.append(comp)

    # Order clusters by aggregate tournament score; order within cluster by tournament score
    cluster_scores = [(idx, sum(s[t] for t in comp)) for idx, comp in enumerate(clusters)]
    cluster_scores.sort(key=lambda x: x[1])
    ordered_clusters = []
    for idx, _score in cluster_scores:
        comp = clusters[idx]
        comp.sort(key=lambda t: (s[t], t))
        ordered_clusters.append(comp)

    # Produce a cluster-respecting initial schedule as an incumbent
    cluster_seq = []
    for comp in ordered_clusters:
        cluster_seq.extend(comp)
    # Also consider pure tournament order as a second incumbent candidate
    tournament_seq = tournament_order[:]

    incumbent_seq = tournament_bubble_pass(cluster_seq, passes=2)
    incumbent_cost = evaluate_seq(incumbent_seq)
    t_seq_cost = evaluate_seq(tournament_seq)
    if t_seq_cost < incumbent_cost:
        incumbent_seq = tournament_seq
        incumbent_cost = t_seq_cost

    # Build buddy lists from pair synergy (smaller sum of M[i][j]+M[j][i] is better)
    buddy_k = min(8, max(3, n // 20 + 4))
    buddies = [[] for _ in range(n)]
    for i in range(n):
        cand = [(M[i][j] + M[j][i], j) for j in range(n) if j != i]
        cand.sort(key=lambda z: (z[0], z[1]))
        buddies[i] = [j for _, j in cand[:buddy_k]]

    # Preselection by W: score candidates based on recent prefix
    def preselect_by_W(prefix, remaining, k, recent_k=4):
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

    # Cluster rank for rollout bias
    cluster_of = {}
    for cid, comp in enumerate(ordered_clusters):
        for v in comp:
            cluster_of[v] = cid
    cluster_rank = {cid: ri for ri, (cid, _) in enumerate([(i, 0) for i in range(len(ordered_clusters))])}

    # Rollout policy: W/buddy/cluster-guided completion; no simulator calls until the end
    def rollout_complete(prefix, remaining):
        seq = list(prefix)
        rem = set(remaining)
        # Choose cluster order bias: prefer to finish earliest remaining cluster
        while rem:
            recents = seq[-4:] if len(seq) >= 4 else seq
            # Identify earliest cluster among remaining
            earliest_cluster = None
            min_rank = 10**9
            for v in rem:
                cr = cluster_of.get(v, 0)
                if cr < min_rank:
                    min_rank = cr
                    earliest_cluster = cr
            # Score each candidate by W-margin vs recents + small bonuses
            best_t = None
            best_score = float('inf')
            last = seq[-1] if seq else None
            for t in rem:
                sc = 0
                for x in recents:
                    sc += W[x][t]
                # Bias to stay within earliest cluster
                if cluster_of.get(t, 0) == earliest_cluster:
                    sc -= 0.15 * tau
                # Buddy bias
                if last is not None and t in buddies[last]:
                    sc -= 0.10 * tau
                if sc < best_score:
                    best_score = sc
                    best_t = t
            seq.append(best_t)
            rem.remove(best_t)
        # Cheap ordering cleanup with tournament-style bubble based on M/W only
        seq = tournament_bubble_pass(seq, passes=2)
        return seq

    # Node for MCTS
    class Node:
        __slots__ = ("prefix", "remaining", "N", "Wsum", "children", "untried", "lb")

        def __init__(self, prefix, remaining):
            self.prefix = tuple(prefix)
            self.remaining = frozenset(remaining)
            self.N = 0
            self.Wsum = 0.0  # accumulated reward
            self.children = {}  # move t -> Node
            # Lower bound from partial sequence (monotone). Used for pruning.
            self.lb = evaluate_seq(list(self.prefix))
            # Prepare untried moves using W-preselection and buddies
            if self.remaining:
                cand = list(self.remaining)
                k = min(16, len(cand))
                pre = preselect_by_W(list(self.prefix), cand, k, recent_k=4)
                if self.prefix:
                    last = self.prefix[-1]
                    for b in buddies[last]:
                        if b in self.remaining and b not in pre:
                            pre.append(b)
                            if len(pre) >= k:
                                break
                # Ensure diversity
                if len(pre) < k:
                    rng.shuffle(cand)
                    for t in cand:
                        if t not in pre:
                            pre.append(t)
                            if len(pre) >= k:
                                break
                self.untried = pre
            else:
                self.untried = []

        def ucb_child(self, explore_c, rng):
            # Choose child maximizing UCB1 on reward (higher is better since reward = -cost)
            logN = math.log(self.N + 1.0)
            best = None
            best_val = -1e18
            for t, ch in self.children.items():
                if ch.N == 0:
                    ucb = float('inf')
                else:
                    mean = ch.Wsum / ch.N
                    ucb = mean + explore_c * math.sqrt(logN / ch.N)
                # tiny noise to break ties deterministically by RNG
                val = ucb + 1e-9 * rng.random()
                if val > best_val:
                    best_val = val
                    best = ch
            return best

    # Lightweight adjacent-improvement pass using simulator (bounded budget)
    def adjacent_improve(seq, base_cost):
        best_seq = list(seq)
        best_cost = base_cost
        nloc = len(best_seq)
        improved = True
        passes = 2
        for _ in range(passes):
            improved = False
            for i in range(nloc - 1):
                a, b = best_seq[i], best_seq[i + 1]
                if prefer_before(a, b):
                    continue
                cand = best_seq[:]
                cand[i], cand[i + 1] = cand[i + 1], cand[i]
                c = evaluate_seq(cand)
                if c < best_cost:
                    best_cost = c
                    best_seq = cand
                    improved = True
            if not improved:
                break
        return best_seq, best_cost

    # MCTS core
    def mcts_search(iterations, explore_c, incumbent_cost, incumbent_seq):
        best_cost = incumbent_cost
        best_seq = incumbent_seq[:]
        root = Node([], set(range(n)))

        for _ in range(iterations):
            node = root
            path = [node]

            # Selection
            while node.remaining and not node.untried:
                # Incumbent pruning
                if node.lb >= best_cost:
                    break
                next_node = node.ucb_child(explore_c, rng)
                if next_node is None:
                    break
                node = next_node
                path.append(node)

            # Expansion
            if node.remaining and node.untried and node.lb < best_cost:
                t = node.untried.pop(0)
                new_prefix = list(node.prefix) + [t]
                new_remaining = set(node.remaining)
                new_remaining.remove(t)
                child = Node(new_prefix, new_remaining)
                node.children[t] = child
                node = child
                path.append(node)

            # Rollout or pruning outcome
            if node.lb >= best_cost:
                # Guaranteed not to beat incumbent; assign incumbent reward
                rollout_cost = best_cost
            else:
                full_seq = node.prefix if not node.remaining else rollout_complete(node.prefix, node.remaining)
                # Evaluate final sequence cost
                rollout_cost = evaluate_seq(list(full_seq))
                # Update incumbent
                if rollout_cost < best_cost:
                    best_cost = rollout_cost
                    best_seq = list(full_seq)

            # Backpropagate reward (higher better): reward = -cost
            reward = -rollout_cost
            for nd in path:
                nd.N += 1
                nd.Wsum += reward

        return best_cost, best_seq

    # Iteration budget and restarts with shared cache
    # Balanced to limit runtime yet allow good search; scaled by num_seqs and n
    base_iters = min(20 * n, 2400)
    extra = max(0, int(num_seqs) * 60)
    iterations = base_iters + extra
    # 2–3 deterministic restarts with varying exploration constants
    restarts = 3 if n <= 110 else 2
    explore_constants = [0.9, 1.3, 1.7][:restarts]

    global_best_cost = incumbent_cost
    global_best_seq = incumbent_seq[:]

    for c_exp in explore_constants:
        # Periodically tighten incumbent by a greedy W-guided completion from a short prefix
        # Seed with cluster order prefix of length k and greedy complete once
        # This is cheap and can update incumbent quickly
        k_seed = min(6, n)
        if k_seed > 0:
            prefix_seed = cluster_seq[:k_seed]
            rem = [t for t in range(n) if t not in prefix_seed]
            quick_seq = rollout_complete(prefix_seed, rem)
            quick_cost = evaluate_seq(quick_seq)
            if quick_cost < global_best_cost:
                global_best_cost = quick_cost
                global_best_seq = quick_seq

        best_cost, best_seq = mcts_search(iterations, c_exp, global_best_cost, global_best_seq)
        if best_cost < global_best_cost:
            global_best_cost = best_cost
            global_best_seq = best_seq

    # Final tiny adjacent improvement to polish
    final_seq, final_cost = adjacent_improve(global_best_seq, global_best_cost)

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