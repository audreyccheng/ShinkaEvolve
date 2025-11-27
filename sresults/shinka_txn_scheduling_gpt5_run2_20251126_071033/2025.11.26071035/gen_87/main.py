# EVOLVE-BLOCK-START
"""Transaction scheduling algorithm for optimizing makespan across multiple workloads
MCTS + bounded branch-and-bound endgame + suffix exact optimize + light local polish.
"""

import time
import random
import sys
import os
from math import sqrt, inf, log, exp

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
    Monte Carlo Tree Search planner with endgame exact optimization to minimize makespan.

    Args:
        workload: Workload object containing transaction data
        num_seqs: Search effort parameter (drives MCTS iterations and refine budgets)

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    n = workload.num_txns
    all_txns = list(range(n))
    if n == 0:
        return 0, []

    # ------------------------ Adaptive parameters ------------------------
    small = n <= 50
    med = 50 < n <= 90
    large = n > 90

    # MCTS iterations and rollout/sample settings
    ENDGAME_K = 9  # branch-and-bound exact finish threshold
    MAX_BNB_NODES = 3000  # cap for DFS nodes expanded
    # Iteration budget scales with n and num_seqs
    MCTS_ITERS = max(1200, min(6500, int(22 * n + 200 * num_seqs)))
    CP_UCT = 1.25  # exploration term
    PRIOR_TEMP = 0.10  # softmax temperature on prior (lower => sharper)
    # Progressive widening parameters
    PW_BASE = 2
    PW_ALPHA = 0.5  # allowed = PW_BASE + floor((N) ** PW_ALPHA)
    # Rollout sampling
    ROLLOUT_K = 10 if med or large else 12
    EPSILON_TOPK = 0.2  # epsilon pick random among top-2 during rollout

    # Suffix optimization postprocessing
    SUFFIX_K = 9
    SUFFIX_BNB_NODES = 3500

    # Light local polish parameters
    POLISH_ADJ_PASSES = 2 if large else 3
    OR_OPT1_SPARSITY = 8  # evaluate every s-th position to be lighter

    # Seed RNG per call for reproducible-but-diverse runs
    random.seed((n * 2654435761 + num_seqs * 11939 + 97) % (2**32 - 1))

    # ------------------------ Cached evaluator ------------------------
    cost_cache = {}

    def eval_cost(seq):
        key = tuple(seq)
        c = cost_cache.get(key)
        if c is None:
            c = workload.get_opt_seq_cost(list(seq))
            cost_cache[key] = c
        return c

    # ------------------------ Branch-and-bound exact finish ------------------------
    def exact_finish(prefix, rem_set, ub, node_cap=MAX_BNB_NODES):
        """
        Optimal completion for small remaining set using DFS branch-and-bound.
        Prefix is fixed; returns (best_tail, best_cost).
        """
        rem = list(rem_set)
        best_cost = ub
        best_tail = None
        nodes = 0

        # transposition table: (frozenset(rem_left), last2_tuple) -> best lower bound seen
        tt = {}

        last2 = tuple(prefix[-2:]) if len(prefix) >= 2 else tuple(prefix)
        base_lb = eval_cost(prefix)

        def dfs(cur_tail, remaining, last2_sig):
            nonlocal best_cost, best_tail, nodes
            if nodes >= node_cap:
                return
            # current lower bound
            cur_seq = prefix + cur_tail
            lb = eval_cost(cur_seq)
            if lb >= best_cost:
                return
            key = (frozenset(remaining), last2_sig)
            prev_lb = tt.get(key)
            if prev_lb is not None and lb >= prev_lb - 1e-12:
                return
            tt[key] = lb
            if not remaining:
                # full completion
                final_c = lb
                if final_c < best_cost:
                    best_cost = final_c
                    best_tail = list(cur_tail)
                return
            nodes += 1
            # order next candidates by immediate append cost
            scored = []
            for t in remaining:
                c1 = eval_cost(cur_seq + [t])
                scored.append((c1, t))
            scored.sort(key=lambda x: x[0])
            for c1, t in scored:
                nxt_tail = cur_tail + [t]
                nxt_remaining = [x for x in remaining if x != t]
                nxt_last2 = (last2_sig + (t,))[-2:]
                dfs(nxt_tail, nxt_remaining, nxt_last2)
                if nodes >= node_cap:
                    break

        dfs([], rem, last2)
        if best_tail is None:
            # fallback greedy
            cur = list(prefix)
            remaining = list(rem_set)
            while remaining:
                best_t = min(remaining, key=lambda t: eval_cost(cur + [t]))
                cur.append(best_t)
                remaining.remove(best_t)
            return cur[len(prefix):], eval_cost(cur)
        return best_tail, best_cost

    # ------------------------ MCTS Node ------------------------
    class Node:
        __slots__ = ('prefix', 'rem', 'N', 'child_stats', 'children', 'pool', 'scored', 'prior_probs')

        def __init__(self, prefix, rem):
            self.prefix = tuple(prefix)
            self.rem = frozenset(rem)
            self.N = 0
            # child_stats[action] = [N_sa, W_sum, best_reward]
            self.child_stats = {}
            # children[action] = child_key (tuple new_prefix)
            self.children = {}
            # Progressive widening pools
            self.pool = set(rem)  # unexplored actions
            self.scored = []      # list of (score, action) where score is eval_cost(prefix+[action])
            self.prior_probs = {} # action -> prior prob among scored top-K

        def key(self):
            return self.prefix

        def allowed_children(self):
            return PW_BASE + int(self.N ** PW_ALPHA)

        def ensure_top_k(self, k):
            # Grow scored list until having at least k items or pool exhausted
            while len(self.scored) < min(k, len(self.pool) + len(self.scored)):
                to_eval = min(8, len(self.pool))
                if to_eval <= 0:
                    break
                sample = random.sample(list(self.pool), to_eval)
                for a in sample:
                    self.pool.remove(a)
                    score = eval_cost(list(self.prefix) + [a])
                    self.scored.append((score, a))
                self.scored.sort(key=lambda x: x[0])
            # compute prior probs among current top-k
            consider = [a for _, a in self.scored[:k]]
            if consider:
                # softmax over negative score for preference
                scores = [s for s, a in self.scored[:k]]
                # normalize with subtract min for stability
                m = min(scores)
                logits = [-(s - m) / max(1e-6, PRIOR_TEMP) for s in scores]
                maxlog = max(logits)
                exps = [exp(z - maxlog) for z in logits]
                Z = sum(exps) + 1e-12
                for a, e in zip(consider, exps):
                    self.prior_probs[a] = e / Z

        def considered_actions(self):
            k = self.allowed_children()
            self.ensure_top_k(k)
            return [a for _, a in self.scored[:k]]

        def select_action_ucb(self):
            acts = self.considered_actions()
            if not acts:
                return None
            # Find action with max PUCT score
            best_a = None
            best_score = -1e18
            for a in acts:
                stats = self.child_stats.get(a)
                if stats is None:
                    nsa = 0
                    q = 0.0
                else:
                    nsa = stats[0]
                    q = stats[1] / max(1, nsa)
                p = self.prior_probs.get(a, 1.0 / max(1, len(acts)))
                u = CP_UCT * p * sqrt(max(1, self.N)) / (1 + nsa)
                val = q + u
                if val > best_score:
                    best_score = val
                    best_a = a
            return best_a

    # ------------------------ MCTS core ------------------------
    nodes = {}

    def get_node(prefix, rem):
        key = tuple(prefix)
        nd = nodes.get(key)
        if nd is None:
            nd = Node(prefix, rem)
            nodes[key] = nd
        return nd

    best_global_cost = float('inf')
    best_global_seq = None

    def greedy_completion(prefix, rem):
        cur = list(prefix)
        remaining = set(rem)
        while remaining:
            # choose best next append by cost
            t = min(remaining, key=lambda x: eval_cost(cur + [x]))
            cur.append(t)
            remaining.remove(t)
        return cur, eval_cost(cur)

    def rollout_from(prefix, rem):
        # epsilon-greedy rollout with sampled candidates
        cur = list(prefix)
        rem_list = list(rem)
        while rem_list:
            if len(rem_list) <= ROLLOUT_K:
                cand = rem_list
            else:
                cand = random.sample(rem_list, ROLLOUT_K)
            scored = [(eval_cost(cur + [t]), t) for t in cand]
            scored.sort(key=lambda x: x[0])
            if random.random() < EPSILON_TOPK and len(scored) >= 2:
                _, t = random.choice(scored[:2])
            else:
                _, t = scored[0]
            cur.append(t)
            rem_list.remove(t)
        return cur, eval_cost(cur)

    # Main MCTS loop
    for it in range(MCTS_ITERS):
        prefix = []
        rem = set(all_txns)
        path = []  # list of (node, action)
        reward = None

        # Selection & expansion
        while True:
            node = get_node(prefix, rem)
            node.N += 1
            # Terminal?
            if not rem:
                full_cost = eval_cost(prefix)
                reward = -full_cost
                if full_cost < best_global_cost:
                    best_global_cost, best_global_seq = full_cost, list(prefix)
                break
            a = node.select_action_ucb()
            if a is None:
                # should not happen unless empty rem
                break
            # Expand if unseen child
            if a not in node.children:
                new_prefix = prefix + [a]
                new_rem = set(rem)
                new_rem.remove(a)
                node.children[a] = tuple(new_prefix)
                path.append((node, a))
                # Endgame exact if small
                if len(new_rem) <= ENDGAME_K:
                    # quick bound by greedy completion
                    ub_seq, ub_cost = greedy_completion(new_prefix, new_rem)
                    tail, opt_cost = exact_finish(new_prefix, new_rem, ub_cost, node_cap=MAX_BNB_NODES)
                    full_seq = new_prefix + tail
                    final_cost = opt_cost
                else:
                    # rollout
                    full_seq, final_cost = rollout_from(new_prefix, new_rem)
                reward = -final_cost
                if final_cost < best_global_cost:
                    best_global_cost, best_global_seq = final_cost, list(full_seq)
                break
            else:
                # move down the tree; record path and continue
                path.append((node, a))
                prefix = prefix + [a]
                rem = set(x for x in rem if x != a)
                continue

        # Backpropagate
        if reward is not None:
            for node, a in path:
                st = node.child_stats.get(a)
                if st is None:
                    node.child_stats[a] = [1, reward, reward]
                else:
                    st[0] += 1
                    st[1] += reward
                    if reward > st[2]:
                        st[2] = reward

    # Extract best sequence from tree policy
    def extract_sequence_from_tree():
        prefix = []
        rem = set(all_txns)
        while rem:
            node = nodes.get(tuple(prefix))
            if node is None:
                break
            acts = node.considered_actions()
            if not acts:
                break
            # pick action with best observed reward; fall back to prior
            best_a = None
            best_r = -1e18
            for a in acts:
                st = node.child_stats.get(a)
                r = st[2] if st is not None else -eval_cost(prefix + [a])
                if r > best_r:
                    best_r = r
                    best_a = a
            if best_a is None:
                best_a = min(rem, key=lambda t: eval_cost(prefix + [t]))
            prefix.append(best_a)
            rem.remove(best_a)
        if rem:
            # complete greedily
            seq, cost = greedy_completion(prefix, rem)
            return seq, cost
        return prefix, eval_cost(prefix)

    if best_global_seq is None:
        seq_tree, cost_tree = extract_sequence_from_tree()
        best_global_seq, best_global_cost = seq_tree, cost_tree

    # ------------------------ Suffix exact optimization and light polish ------------------------
    def suffix_exact_optimize(seq, K=SUFFIX_K, cap=SUFFIX_BNB_NODES):
        if K <= 1 or len(seq) <= 2:
            return seq, eval_cost(seq)
        K = min(K, len(seq))
        prefix = seq[:-K]
        suffix_set = set(seq[-K:])
        tail, opt_cost = exact_finish(prefix, suffix_set, ub=eval_cost(seq), node_cap=cap)
        if tail is None:
            return seq, eval_cost(seq)
        new_seq = prefix + tail
        return new_seq, opt_cost

    def or_opt1_pass(seq, start_cost, stride=OR_OPT1_SPARSITY):
        best_seq = list(seq)
        best_cost = start_cost
        L = len(best_seq)
        for i in range(0, L, max(1, stride)):
            t = best_seq[i]
            base = best_seq[:i] + best_seq[i + 1:]
            m = len(base) + 1
            # try a sparse grid of positions: ends + mid + quartiles
            positions = {0, m - 1, m // 2, m // 4, (3 * m) // 4}
            positions = [p for p in sorted(positions) if 0 <= p < m and p != i]
            # add two neighbors around original index if possible
            for d in (-2, -1, 1, 2):
                p = i + d
                if 0 <= p < m:
                    positions.append(p)
            seen = set()
            move_best = best_cost
            move_pos = None
            for p in positions:
                if p in seen:
                    continue
                seen.add(p)
                cand = base[:]
                cand.insert(p, t)
                c = eval_cost(cand)
                if c < move_best:
                    move_best = c
                    move_pos = p
            if move_pos is not None and move_best + 1e-12 < best_cost:
                new_seq = base[:]
                new_seq.insert(move_pos, t)
                best_seq, best_cost = new_seq, move_best
        return best_seq, best_cost

    def adjacent_swaps(seq, start_cost, max_passes=POLISH_ADJ_PASSES):
        best_seq = list(seq)
        best_cost = start_cost
        for _ in range(max_passes):
            improved = False
            for i in range(len(best_seq) - 1):
                cand = best_seq[:]
                cand[i], cand[i + 1] = cand[i + 1], cand[i]
                c = eval_cost(cand)
                if c < best_cost:
                    best_seq, best_cost = cand, c
                    improved = True
            if not improved:
                break
        return best_seq, best_cost

    # Apply suffix exact optimization then local polish
    seq_opt, cost_opt = suffix_exact_optimize(best_global_seq, K=SUFFIX_K, cap=SUFFIX_BNB_NODES)
    seq_opt, cost_opt = or_opt1_pass(seq_opt, cost_opt, stride=OR_OPT1_SPARSITY)
    seq_opt, cost_opt = adjacent_swaps(seq_opt, cost_opt, max_passes=POLISH_ADJ_PASSES)

    # Safety checks
    assert len(seq_opt) == n and len(set(seq_opt)) == n, "Schedule must include each transaction exactly once"

    return cost_opt, seq_opt


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