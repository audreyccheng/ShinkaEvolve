# EVOLVE-BLOCK-START
"""Transaction scheduling algorithm for optimizing makespan across multiple workloads"""

import time
import random
import sys
import os
import math

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
    UCT-based MCTS over partial schedules with progressive widening and VND local improvement.

    Args:
        workload: Workload object containing transaction data
        num_seqs: Budget scalar to control search effort

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    n = workload.num_txns
    rng = random.Random()
    # Tie exploration to problem size via iteration budget
    iter_budget = max(500, int((3.5 if n >= 80 else 5.0) * n * max(8, int(num_seqs))))
    exploration_c = 1.15

    # Cost cache for partial sequence evaluation
    cost_cache = {}

    def eval_cost(seq):
        key = tuple(seq)
        if key in cost_cache:
            return cost_cache[key]
        c = workload.get_opt_seq_cost(seq)
        cost_cache[key] = c
        return c

    best_cost = float('inf')
    best_seq = list(range(n))  # placeholder

    class Node:
        __slots__ = ("seq", "remaining", "N", "W", "children", "expanded_actions", "partial_cost")

        def __init__(self, seq, remaining):
            self.seq = tuple(seq)
            self.remaining = frozenset(remaining)
            self.N = 0
            self.W = 0.0  # cumulative reward (negative makespan)
            self.children = {}  # action -> Node
            self.expanded_actions = set()  # track which actions were expanded
            self.partial_cost = eval_cost(seq)  # cost of this prefix

        def terminal(self):
            return len(self.remaining) == 0

        def widen_limit(self):
            # Progressive widening limit increases with visits
            # Keep small at first; grow roughly with sqrt(N)
            return min(len(self.remaining), 1 + int(max(1, math.sqrt(self.N + 1))))

    # Transposition table keyed by (seq, remaining) to share nodes
    TT = {}

    def get_node(seq, remaining):
        key = (tuple(seq), frozenset(remaining))
        node = TT.get(key)
        if node is None:
            node = Node(seq, remaining)
            TT[key] = node
        return node

    def select(node):
        # Traverse down using UCT until a node eligible for expansion or terminal
        path = [node]
        while True:
            current = path[-1]
            if current.terminal():
                return path
            # Progressive widening: expand if room available
            if len(current.expanded_actions) < current.widen_limit():
                return path
            # Otherwise select child with max UCB
            best = None
            best_ucb = -float('inf')
            logN = math.log(current.N + 1.0)
            for a, child in current.children.items():
                if child.N == 0:
                    ucb = float('inf')
                else:
                    mean = child.W / child.N
                    ucb = mean + exploration_c * math.sqrt(logN / child.N)
                if ucb > best_ucb:
                    best_ucb = ucb
                    best = child
            if best is None:
                return path
            path.append(best)

    def expand(node):
        # Choose an unexpanded action to add; bias by immediate cost on a sampled subset
        available = list(node.remaining - node.expanded_actions)
        if not available:
            return node  # nothing to expand
        # Sample candidate actions to score
        sample_k = min(len(available), 8 if n <= 60 else 12)
        cand = available if len(available) <= sample_k else rng.sample(available, sample_k)

        # Pick by minimal immediate partial cost
        best_a = None
        best_c = float('inf')
        base_seq = list(node.seq)
        for a in cand:
            c = eval_cost(base_seq + [a])
            if c < best_c:
                best_c = c
                best_a = a

        if best_a is None:  # fallback
            best_a = rng.choice(available)

        new_seq = list(node.seq) + [best_a]
        new_remaining = set(node.remaining)
        new_remaining.remove(best_a)
        child = get_node(new_seq, new_remaining)
        node.children[best_a] = child
        node.expanded_actions.add(best_a)
        return child

    def rollout_from(node):
        nonlocal best_cost, best_seq
        # If this prefix already worse than best known, abort
        if node.partial_cost >= best_cost:
            return best_cost  # discourage

        seq = list(node.seq)
        remaining = list(node.remaining)
        partial_cost = node.partial_cost

        # Greedy rollout with limited lookahead; prune if hopeless
        while remaining:
            # Early cutoff if already worse than best
            if partial_cost >= best_cost:
                return best_cost

            # Sample candidate set
            cand_k = min(len(remaining), 10 if n <= 60 else 14)
            cand = remaining if len(remaining) <= cand_k else rng.sample(remaining, cand_k)

            # Score candidates by immediate and 1-step lookahead
            chosen = None
            chosen_score = float('inf')
            la_samples = 3 if len(remaining) > 6 else len(remaining) - 1
            if la_samples < 0:
                la_samples = 0

            for t in cand:
                c1 = eval_cost(seq + [t])
                if c1 >= chosen_score:
                    continue
                if la_samples > 0 and len(remaining) > 1:
                    pool = [x for x in remaining if x != t]
                    pool_samp = pool if len(pool) <= la_samples else rng.sample(pool, la_samples)
                    c2 = min(eval_cost(seq + [t, u]) for u in pool_samp)
                    score = c2
                else:
                    score = c1
                if score < chosen_score:
                    chosen_score = score
                    chosen = t

            if chosen is None:
                chosen = rng.choice(remaining)

            seq.append(chosen)
            remaining.remove(chosen)
            partial_cost = eval_cost(seq)

        # Complete schedule cost
        full_cost = partial_cost
        if full_cost < best_cost:
            best_cost = full_cost
            best_seq = seq[:]
        return full_cost

    def backpropagate(path, final_cost):
        R = -float(final_cost)  # reward is negative cost
        for node in path:
            node.N += 1
            node.W += R

    # Run MCTS iterations
    root = get_node([], set(range(n)))
    # Seed best by a quick greedy completion to help pruning
    _ = rollout_from(root)

    for _ in range(iter_budget):
        path = select(root)
        leaf = path[-1]
        if not leaf.terminal():
            leaf = expand(leaf)
            path.append(leaf)
        # Simulate from leaf
        final_cost = rollout_from(leaf)
        backpropagate(path, final_cost)

    # Extract best sequence by traversing most visited children
    seq = []
    current = root
    remaining = set(range(n))
    while remaining:
        if not current.children:
            # Finish greedily if tree shallow here
            # Reuse rollout policy deterministically
            # Build one step at a time using immediate cost
            cand = list(remaining)
            best_t = None
            best_c = float('inf')
            for t in cand:
                c = eval_cost(seq + [t])
                if c < best_c:
                    best_c = c
                    best_t = t
            seq.append(best_t)
            remaining.remove(best_t)
            current = get_node(seq, remaining)
            continue

        # Choose child with highest visit count
        best_child = None
        best_N = -1
        best_a = None
        for a, child in current.children.items():
            if child.N > best_N:
                best_N = child.N
                best_child = child
                best_a = a
        if best_a is None:
            # Fallback random
            best_a = rng.choice(list(remaining))
            seq.append(best_a)
            remaining.remove(best_a)
            current = get_node(seq, remaining)
        else:
            seq.append(best_a)
            remaining.remove(best_a)
            current = best_child

    seq_cost = eval_cost(seq)
    if seq_cost < best_cost:
        best_cost, best_seq = seq_cost, seq

    # Variable Neighborhood Descent (local improvement)
    def vnd_improve(seq):
        best_seq_local = list(seq)
        best_c = eval_cost(best_seq_local)
        nloc = len(best_seq_local)

        # 1) Adjacent swaps until no improvement
        improved = True
        while improved:
            improved = False
            for i in range(nloc - 1):
                s = best_seq_local
                s[i], s[i + 1] = s[i + 1], s[i]
                c = eval_cost(s)
                if c < best_c:
                    best_c = c
                    improved = True
                else:
                    # revert
                    s[i], s[i + 1] = s[i + 1], s[i]

        # 2) Randomized insertions (O(n)) attempts
        attempts = min(5 * nloc, 400 + 2 * nloc)
        for _ in range(attempts):
            i = rng.randrange(nloc)
            j = rng.randrange(nloc)
            if i == j:
                continue
            s = best_seq_local
            x = s.pop(i)
            s.insert(j, x)
            c = eval_cost(s)
            if c < best_c:
                best_c = c
            else:
                # revert
                x2 = s.pop(j)
                s.insert(i, x2)

        # 3) Small block relocation (length 2 or 3)
        block_attempts = min(3 * nloc, 300)
        for _ in range(block_attempts):
            L = 2 if rng.random() < 0.7 else 3
            if nloc <= L:
                break
            i = rng.randrange(0, nloc - L + 1)
            j = rng.randrange(0, nloc - L + 1)
            if j >= i and j <= i + L - 1:
                continue  # avoid trivial overlap
            s = best_seq_local
            block = s[i:i + L]
            del s[i:i + L]
            # adjust insertion index if needed
            if j > i:
                j = j - L
            s[j:j] = block
            c = eval_cost(s)
            if c < best_c:
                best_c = c
                nloc = len(s)
            else:
                # revert
                del s[j:j + L]
                s[i:i] = block

        return best_c, best_seq_local

    final_cost, final_seq = vnd_improve(best_seq)
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