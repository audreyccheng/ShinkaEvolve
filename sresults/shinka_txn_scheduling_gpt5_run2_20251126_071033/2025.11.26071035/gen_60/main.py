# EVOLVE-BLOCK-START
"""Transaction scheduling algorithm for optimizing makespan across multiple workloads
MCTS with UCT selection, progressive widening, regret-aware greedy rollouts, and light polishing.
"""

import time
import random
import sys
import os
from collections import defaultdict, deque
from functools import lru_cache
from math import sqrt, log

# Add the openevolve_examples directory to the path to import txn_simulator and workloads
# Find the repository root by looking for the openevolve_examples directory
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
    Monte Carlo Tree Search with UCT over transaction order prefixes.
    Progressive widening and regret-aware greedy rollouts complete schedules.
    A light polishing step refines the best found solution.

    Args:
        workload: Workload object
        num_seqs: Relative search effort
    Returns:
        (best_makespan, best_schedule_list)
    """
    n = workload.num_txns
    all_txns = list(range(n))
    all_set = set(all_txns)

    # Global RNG seed for controlled diversity across runs
    random.seed((n * 2654435761 + num_seqs * 1315423911) % (2**32 - 1))

    # Cached evaluation of partial prefixes to minimize simulator calls
    @lru_cache(maxsize=400_000)
    def eval_cost_tuple(prefix_tuple):
        return workload.get_opt_seq_cost(list(prefix_tuple))

    def eval_cost(seq):
        return eval_cost_tuple(tuple(seq))

    # Node structure for MCTS
    class Node:
        __slots__ = ("prefix", "rem_set", "N", "W", "children", "cand_rank", "widen_k")
        def __init__(self, prefix, rem_set):
            self.prefix = tuple(prefix)
            self.rem_set = frozenset(rem_set)
            self.N = 0          # visit count
            self.W = 0.0        # cumulative reward (negative makespan)
            self.children = {}  # txn -> Node
            self.cand_rank = None  # ranked candidate list cache (txn order)
            self.widen_k = 0    # current progressive widening limit

    # Transposition table for nodes by prefix
    node_table = {}

    def get_node(prefix):
        key = tuple(prefix)
        node = node_table.get(key)
        if node is None:
            rem_set = all_set.difference(prefix)
            node = Node(prefix, rem_set)
            node_table[key] = node
        return node

    # Candidate ranking for progressive widening at a node
    def ensure_ranked_candidates(node, sample_cap=32):
        if node.cand_rank is not None:
            return
        rem = list(node.rem_set)
        # Rank by immediate partial cost of appending txn at end of prefix
        # Use sampling when many remain
        if len(rem) > sample_cap:
            rem_sample = random.sample(rem, sample_cap)
        else:
            rem_sample = rem
        scored = []
        base_prefix = list(node.prefix)
        for t in rem_sample:
            c = eval_cost(base_prefix + [t])
            scored.append((c, t))
        scored.sort(key=lambda x: x[0])
        # Place all non-sampled txns after the scored set, preserving some randomness
        others = [t for t in rem if t not in {t for _, t in scored}]
        random.shuffle(others)
        node.cand_rank = [t for _, t in scored] + others
        node.widen_k = 0

    # Progressive widening control
    def next_candidates(node, alpha=0.5, base=2, max_new=6):
        """
        Return a slice of candidate txns permitted by progressive widening.
        K grows roughly like base + N^alpha, capped by rem size.
        """
        ensure_ranked_candidates(node)
        K_target = min(len(node.cand_rank), base + int(node.N ** alpha))
        # Grow only a limited number per call to stabilize UCT stats
        K_target = min(K_target, node.widen_k + max_new)
        if K_target > node.widen_k:
            node.widen_k = K_target
        return node.cand_rank[:node.widen_k]

    # Rollout policy: greedily append txns by lowest incremental partial cost among a sampled subset,
    # with small randomness and cached evals reused across steps.
    def rollout_from(prefix, rem_set, sample_k=16, jitter=3):
        cur = list(prefix)
        rem = list(rem_set)
        # Small table for prefix+[t] costs within this rollout step
        while rem:
            if len(rem) <= sample_k + jitter:
                cand = list(rem)
            else:
                cand = random.sample(rem, sample_k + random.randint(0, jitter))
            best_t = None
            best_c = float('inf')
            for t in cand:
                c = eval_cost(cur + [t])
                if c < best_c:
                    best_c = c
                    best_t = t
            if best_t is None:
                # fallback
                best_t = rem[0]
            cur.append(best_t)
            rem.remove(best_t)
        return cur, eval_cost(cur)

    # UCT parameters
    # Exploration constant tuned to typical scale of negative makespans (rewards)
    C_UCB = 1.8

    # MCTS main loop
    def mcts_search(budget_iters, time_cap_sec=None):
        best_cost = float('inf')
        best_seq = list(range(n))
        start_time = time.time() if time_cap_sec is not None else None

        root = get_node(())
        baseline_reward = -eval_cost([])  # baseline reward reference for stability

        for it in range(budget_iters):
            if time_cap_sec is not None and (time.time() - start_time) > time_cap_sec:
                break

            # SELECTION
            path = []
            node = root
            path.append(node)
            # Traverse until leaf or not fully widened
            while True:
                # If node is terminal (no remaining), stop selection
                if not node.rem_set:
                    break
                # Progressive widening: fetch allowed candidates
                candidates = next_candidates(node)
                # If there are unexpanded actions, prefer expanding one randomly among top few
                unexpanded = [t for t in candidates if t not in node.children]
                if unexpanded:
                    # Bias toward top by immediate score with slight randomness
                    pick = unexpanded[0] if random.random() < 0.6 else random.choice(unexpanded[:max(1, min(4, len(unexpanded)))])
                    # EXPANSION
                    new_prefix = list(node.prefix) + [pick]
                    child = get_node(new_prefix)
                    node.children[pick] = child
                    node = child
                    path.append(node)
                    break
                else:
                    # All candidates expanded: choose child by UCT among them
                    # If progressive widening is too tight and no candidates, widen more
                    if not candidates:
                        # Force widen
                        node.widen_k = min(len(node.rem_set), node.widen_k + 1)
                        candidates = next_candidates(node)
                        if not candidates:
                            break
                    # Select best UCT
                    best_score = -1e18
                    best_child = None
                    best_txn = None
                    parent_N = max(1, node.N)
                    for t in candidates:
                        child = node.children.get(t)
                        if child is None:
                            continue
                        if child.N == 0:
                            uct = 1e9  # force exploration
                        else:
                            q = child.W / child.N
                            uct = q + C_UCB * sqrt(log(parent_N) / child.N)
                        if uct > best_score:
                            best_score = uct
                            best_child = child
                            best_txn = t
                    if best_child is None:
                        break
                    node = best_child
                    path.append(node)

            # SIMULATION (rollout)
            # From current node's prefix, complete schedule greedily with sampling
            final_seq, final_cost = rollout_from(node.prefix, node.rem_set)

            # BACKUP
            reward = -final_cost  # higher is better
            for p in path:
                p.N += 1
                p.W += reward

            if final_cost < best_cost:
                best_cost = final_cost
                best_seq = final_seq

        return best_cost, best_seq

    # Post-MCTS lightweight polishing: Or-opt(1,2) reinsertion and a few sampled swaps
    # Best-two insertion cache to speed reinsertion
    best_two_cache = {}

    def positions_stratified(seq_len, cap=20):
        total = seq_len + 1
        if total <= cap:
            return list(range(total))
        anchors = {0, seq_len, seq_len // 2, seq_len // 4, (3 * seq_len) // 4}
        anchors = [p for p in sorted(anchors) if 0 <= p <= seq_len]
        # sample a few interiors
        interior = [i for i in range(1, seq_len) if i not in anchors]
        k = min(8, len(interior))
        if k > 0:
            anchors = sorted(set(anchors).union(random.sample(interior, k)))
        return anchors

    def best_two_insertions(seq, txn):
        key = (tuple(seq), txn)
        cached = best_two_cache.get(key)
        if cached is not None:
            return cached
        positions = positions_stratified(len(seq), cap=22 if n > 50 else 1000)
        best = (float('inf'), None)
        second = (float('inf'), None)
        for pos in positions:
            cand = seq[:]
            cand.insert(pos, txn)
            c = eval_cost(cand)
            if c < best[0]:
                second = best
                best = (c, pos)
            elif c < second[0]:
                second = (c, pos)
        if second[0] == float('inf'):
            second = best
        best_two_cache[key] = (best, second)
        return best, second

    def or_opt_reinsertion(seq, start_cost, block_sizes=(2, 1), passes=2):
        best_seq = list(seq)
        best_cost = start_cost
        for _ in range(passes):
            improved = False
            for k in block_sizes:
                if k > len(best_seq):
                    continue
                i = 0
                while i <= len(best_seq) - k:
                    block = best_seq[i:i + k]
                    base = best_seq[:i] + best_seq[i + k:]
                    move_best_c = best_cost
                    move_best_p = None
                    positions = positions_stratified(len(base), cap=24 if n > 50 else 1000)
                    for p in positions:
                        cand = base[:]
                        cand[p:p] = block
                        c = eval_cost(cand)
                        if c < move_best_c:
                            move_best_c = c
                            move_best_p = p
                    if move_best_p is not None:
                        base[move_best_p:move_best_p] = block
                        best_seq = base
                        best_cost = move_best_c
                        improved = True
                        # do not increment i to re-evaluate after structural change
                    else:
                        i += 1
            if not improved:
                break
        return best_seq, best_cost

    def sampled_pair_swaps(seq, start_cost, tries):
        best_seq = list(seq)
        best_cost = start_cost
        L = len(best_seq)
        if L <= 3:
            return best_seq, best_cost
        best_delta = 0.0
        best_pair = None
        for _ in range(tries):
            i = random.randint(0, L - 1)
            j = random.randint(0, L - 1)
            if i == j or abs(i - j) <= 1:
                continue
            cand = best_seq[:]
            cand[i], cand[j] = cand[j], cand[i]
            c = eval_cost(cand)
            delta = best_cost - c
            if delta > best_delta:
                best_delta = delta
                best_pair = (i, j)
        if best_pair is not None and best_delta > 0:
            i, j = best_pair
            best_seq[i], best_seq[j] = best_seq[j], best_seq[i]
            best_cost = eval_cost(best_seq)
        return best_seq, best_cost

    # Configure MCTS effort based on n and num_seqs
    # Use more iterations for larger n but cap to keep runtime practical
    if n <= 40:
        iters = 800 + 30 * num_seqs
        time_cap = None
    elif n <= 80:
        iters = 1400 + 40 * num_seqs
        time_cap = None
    else:
        iters = 2200 + 50 * num_seqs
        time_cap = None

    # Split into a few restarts to diversify tree roots slightly (via RNG in candidate ranking)
    restarts = max(2, min(5, 1 + num_seqs // 3))
    iters_per = max(200, iters // restarts)

    global_best_cost = float('inf')
    global_best_seq = list(range(n))

    for r in range(restarts):
        # light RNG nudge per restart
        random.seed(((n * 911) ^ (num_seqs * 131) ^ (r * 524287)) % (2**32 - 1))
        cost, seq = mcts_search(budget_iters=iters_per, time_cap_sec=time_cap)
        if cost < global_best_cost:
            global_best_cost, global_best_seq = cost, seq

    # Final light polishing
    best_seq = list(global_best_seq)
    best_cost = eval_cost(best_seq)
    # Or-opt(2,1) passes
    best_seq, best_cost = or_opt_reinsertion(best_seq, best_cost, block_sizes=(2, 1), passes=2)
    # A few sampled non-adjacent swaps
    tries = min(250, max(60, n))
    best_seq, best_cost = sampled_pair_swaps(best_seq, best_cost, tries=tries)
    # One more quick Or-opt(1)
    best_seq, best_cost = or_opt_reinsertion(best_seq, best_cost, block_sizes=(1,), passes=1)

    # Safety checks
    assert len(best_seq) == n and len(set(best_seq)) == n, "Schedule must include each transaction exactly once"

    return best_cost, best_seq


def get_random_costs():
    """
    Evaluate scheduling algorithm on three different workloads.

    Returns:
        Tuple of (total_makespan, list_of_schedules, execution_time)
    """
    start_time = time.time()

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