# EVOLVE-BLOCK-START
"""Transaction scheduling algorithm for optimizing makespan across multiple workloads
MCTS with progressive widening, cache-backed evaluations, greedy rollouts, and light polish.
"""

import time
import random
import sys
import os
from math import sqrt, log
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
    MCTS with progressive widening to minimize makespan.

    Args:
        workload: Workload object containing transaction data
        num_seqs: Search effort parameter (drives MCTS iterations and light polish)

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    n = workload.num_txns
    all_txns = list(range(n))

    # ---------------- Parameters ----------------
    small = n <= 50
    med = 50 < n <= 90
    large = n > 90

    # MCTS iteration budget (scales with num_seqs and size)
    base_iters = 1400 if small else (2200 if med else 3000)
    per_seq_boost = 260 if small else (300 if med else 340)
    ITERATIONS = min(9000, base_iters + per_seq_boost * max(1, num_seqs))

    # Rollout policy
    ROLLOUT_TOP_K = 3 if small else 4
    ROLLOUT_EPS = 0.22  # exploration probability to pick among top k
    ROLLOUT_LOOKAHEAD_K = 0  # keep 0 for speed in rollout

    # Expansion scoring
    EXP_LOOKAHEAD_K = 6 if (med or large) else 8
    SCORE_ALPHA = 0.7  # weight on immediate after-append cost vs. lookahead

    # UCT / progressive widening
    UCT_C = 0.9  # exploration constant (rewards normalized to ~[0,1])
    PW_K = 2.5   # widening factor: limit ~ 1 + PW_K*sqrt(N)
    PW_ALPHA = 0.5

    # Light polish passes after rollout and at the end
    POLISH_ADJ_PASSES = 1
    POLISH_OROPT1_PASSES = 1

    # RNG seed for reproducible diversity
    random.seed((n * 93911 + num_seqs * 131071 + 17) % (2**32 - 1))

    # ---------------- Cost evaluator with cache ----------------
    cost_cache = {}

    def eval_cost(seq):
        key = tuple(seq)
        c = cost_cache.get(key)
        if c is None:
            c = workload.get_opt_seq_cost(list(seq))
            cost_cache[key] = c
        return c

    # ---------------- Helper: greedy rollout completion ----------------
    def greedy_complete(seq_prefix, rem_set, eps=ROLLOUT_EPS, top_k=ROLLOUT_TOP_K):
        seq = list(seq_prefix)
        rem = set(rem_set)
        while rem:
            # Evaluate append cost for each candidate
            best_list = []
            for t in rem:
                c1 = eval_cost(seq + [t])
                best_list.append((c1, t))
            best_list.sort(key=lambda x: x[0])
            # epsilon-greedy pick among top_k
            if len(best_list) > 1 and random.random() < eps:
                pick_idx = random.randint(0, min(top_k - 1, len(best_list) - 1))
            else:
                pick_idx = 0
            _, t_star = best_list[pick_idx]
            seq.append(t_star)
            rem.remove(t_star)
        return seq, eval_cost(seq)

    # ---------------- Light polish operators ----------------
    def adjacent_swap_polish(seq, passes=POLISH_ADJ_PASSES):
        best_seq = list(seq)
        best_cost = eval_cost(best_seq)
        for _ in range(max(0, passes)):
            improved = False
            for i in range(len(best_seq) - 1):
                cand = best_seq[:]
                cand[i], cand[i + 1] = cand[i + 1], cand[i]
                c = eval_cost(cand)
                if c + 1e-12 < best_cost:
                    best_seq, best_cost = cand, c
                    improved = True
            if not improved:
                break
        return best_seq, best_cost

    def oropt1_polish(seq, passes=POLISH_OROPT1_PASSES):
        best_seq = list(seq)
        best_cost = eval_cost(best_seq)
        for _ in range(max(0, passes)):
            improved = False
            L = len(best_seq)
            for i in range(L):
                t = best_seq[i]
                base = best_seq[:i] + best_seq[i + 1:]
                # Try reinserting at ends and median for speed
                positions = [0, len(base), len(base) // 2]
                seen = set()
                for p in positions:
                    if p in seen:
                        continue
                    seen.add(p)
                    cand = base[:]
                    cand.insert(p, t)
                    c = eval_cost(cand)
                    if c + 1e-12 < best_cost:
                        best_seq, best_cost = cand, c
                        improved = True
                        break
                if improved:
                    break
            if not improved:
                break
        return best_seq, best_cost

    # ---------------- Expansion scoring helper ----------------
    def score_action(seq, rem_list, t, lookahead_k=EXP_LOOKAHEAD_K):
        """Score appending transaction t to seq, using one-step lookahead sample."""
        seq1 = seq + [t]
        c1 = eval_cost(seq1)
        rem_after = [x for x in rem_list if x != t]
        if not rem_after or lookahead_k <= 0:
            return c1
        k2 = min(lookahead_k, len(rem_after))
        sampled = rem_after if len(rem_after) <= k2 else random.sample(rem_after, k2)
        best_c2 = float('inf')
        for u in sampled:
            cu = eval_cost(seq1 + [u])
            if cu < best_c2:
                best_c2 = cu
        return SCORE_ALPHA * c1 + (1.0 - SCORE_ALPHA) * best_c2

    # ---------------- MCTS Node ----------------
    class Node:
        __slots__ = ('seq', 'rem', 'parent', 'children', 'visits', 'value_sum',
                     'best_cost', 'best_seq', 'action_scores', 'expanded_actions')
        def __init__(self, seq, rem, parent=None):
            self.seq = list(seq)
            self.rem = frozenset(rem)
            self.parent = parent
            self.children = {}  # action t -> Node
            self.visits = 0
            self.value_sum = 0.0  # sum of normalized rewards
            self.best_cost = float('inf')
            self.best_seq = None
            self.action_scores = None  # cached list of (score, t)
            self.expanded_actions = set()

        def expand_limit(self):
            return 1 + int(PW_K * (self.visits ** PW_ALPHA))

        def unexpanded_candidates(self):
            rem_actions = [t for t in self.rem if t not in self.expanded_actions]
            if not rem_actions:
                return []
            # Lazy score computation and caching
            if self.action_scores is None:
                rem_list = list(self.rem)
                scored = []
                for t in rem_list:
                    s = score_action(self.seq, rem_list, t, lookahead_k=EXP_LOOKAHEAD_K)
                    scored.append((s, t))
                scored.sort(key=lambda x: x[0])
                self.action_scores = scored
            # Pick top candidates not yet expanded
            candidates = []
            for s, t in self.action_scores:
                if t in self.expanded_actions:
                    continue
                candidates.append((s, t))
            return candidates

        def best_child_ucb(self):
            """Select child maximizing UCB with normalized rewards."""
            best = None
            best_score = -1e100
            for t, child in self.children.items():
                # Average reward
                q = (child.value_sum / max(1, child.visits))
                u = UCT_C * sqrt(max(0.0, log(max(1, self.visits)) / max(1, child.visits)))
                sc = q + u
                if sc > best_score:
                    best_score = sc
                    best = child
            return best

    # ---------------- Baseline for reward normalization ----------------
    greedy_seq, greedy_cost = greedy_complete([], set(all_txns), eps=0.0, top_k=1)
    # normalized reward: (baseline - cost)/baseline in [~0,1]
    baseline_cost = max(1.0, float(greedy_cost))

    # ---------------- MCTS main loop ----------------
    root = Node([], set(all_txns))
    global_best_cost = float('inf')
    global_best_seq = list(range(n))

    for it in range(ITERATIONS):
        # SELECTION with progressive widening
        node = root
        while True:
            if not node.rem:
                break  # already complete
            can_expand_more = len(node.children) < min(len(node.rem), node.expand_limit())
            if can_expand_more:
                break
            # Otherwise select best child by UCB
            if not node.children:
                break
            nxt = node.best_child_ucb()
            if nxt is None:
                break
            node = nxt

        # EXPANSION (if possible)
        if node.rem:
            candidates = node.unexpanded_candidates()
            if candidates:
                # Expand the most promising unexpanded action
                _, t_pick = candidates[0]
                new_seq = node.seq + [t_pick]
                new_rem = set(node.rem)
                new_rem.remove(t_pick)
                child = Node(new_seq, new_rem, parent=node)
                node.children[t_pick] = child
                node.expanded_actions.add(t_pick)
                node = child

        # ROLLOUT from node
        completed_seq, rollout_cost = greedy_complete(node.seq, set(node.rem), eps=ROLLOUT_EPS, top_k=ROLLOUT_TOP_K)
        # Light polish
        if POLISH_ADJ_PASSES > 0:
            completed_seq, rollout_cost = adjacent_swap_polish(completed_seq, passes=POLISH_ADJ_PASSES)
        if POLISH_OROPT1_PASSES > 0:
            completed_seq, rollout_cost = oropt1_polish(completed_seq, passes=POLISH_OROPT1_PASSES)

        # Update global best
        if rollout_cost + 1e-12 < global_best_cost:
            global_best_cost = rollout_cost
            global_best_seq = list(completed_seq)

        # BACKPROPAGATE normalized reward
        reward = (baseline_cost - rollout_cost) / baseline_cost
        cur = node
        while cur is not None:
            cur.visits += 1
            cur.value_sum += reward
            if rollout_cost + 1e-12 < cur.best_cost:
                cur.best_cost = rollout_cost
                cur.best_seq = list(completed_seq)
            cur = cur.parent

    # Final best sequence and cost
    best_seq = global_best_seq
    best_cost = global_best_cost

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