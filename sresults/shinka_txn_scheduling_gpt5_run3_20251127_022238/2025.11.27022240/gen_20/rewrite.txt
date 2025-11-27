# EVOLVE-BLOCK-START
"""Transaction scheduling algorithm for optimizing makespan across multiple workloads"""

import time
import random
import sys
import os
from math import sqrt, log

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
    Monte Carlo Tree Search (MCTS) constructor with conflict-aware rollouts,
    followed by Large Neighborhood Search (LNS) ruin-and-recreate refinement.
    Shared memoized caches are used across all phases to minimize simulator calls.

    Args:
        workload: Workload object containing transaction data
        num_seqs: Intensity parameter; used to scale search effort slightly

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    N = workload.num_txns

    # Overall per-workload time budget (seconds)
    # Scaled lightly by num_seqs but bounded to keep runtime reasonable
    base_budget = 0.70
    time_budget_sec = min(1.00, base_budget + 0.03 * max(0, int(num_seqs) - 5))
    start_time = time.time()

    def time_left():
        return (time.time() - start_time) < time_budget_sec

    # ----------------------
    # Shared memoized caches
    # ----------------------
    cost_cache = {}
    ext_cache = {}   # (prefix_tuple, cand) -> cost
    pair_cost_cache = {}  # (i, j) -> cost([i, j])
    singleton_cost = {}

    def eval_cost(seq):
        key = tuple(seq)
        c = cost_cache.get(key)
        if c is not None:
            return c
        c = workload.get_opt_seq_cost(seq)
        cost_cache[key] = c
        return c

    def eval_ext(prefix_tuple, cand):
        key = (prefix_tuple, cand)
        c = ext_cache.get(key)
        if c is not None:
            return c
        c = eval_cost(list(prefix_tuple) + [cand])
        ext_cache[key] = c
        return c

    def eval_pair(i, j):
        key = (i, j)
        c = pair_cost_cache.get(key)
        if c is not None:
            return c
        c = eval_cost([i, j])
        pair_cost_cache[key] = c
        return c

    # Precompute singleton costs
    for t in range(N):
        if not time_left():
            break
        singleton_cost[t] = eval_cost([t])

    # ----------------------
    # Conflict estimation
    # ----------------------
    # Sampled symmetric conflict weight between i and j approximated as
    # min(cost([i, j]), cost([j, i])) - max(singleton[i], singleton[j])
    # Larger means more interference if they are adjacent.
    conflict_weight = {}
    def add_conflict(i, j):
        if i == j:
            return
        a = singleton_cost.get(i, eval_cost([i]))
        b = singleton_cost.get(j, eval_cost([j]))
        cij = eval_pair(i, j)
        cji = eval_pair(j, i)
        w = min(cij, cji) - max(a, b)
        if w < 0:
            w = 0.0
        conflict_weight[(i, j)] = w
        conflict_weight[(j, i)] = w

    # Limit estimation time
    if time_left():
        # per txn, sample k others
        sample_k = min(14, max(6, N // 10))
        all_txns = list(range(N))
        for i in all_txns:
            if not time_left():
                break
            others = [x for x in all_txns if x != i]
            k = min(sample_k, len(others))
            if k > 0:
                for j in random.sample(others, k):
                    if not time_left():
                        break
                    if (i, j) not in conflict_weight:
                        add_conflict(i, j)

    def pair_penalty(a, b):
        # Fallback to zero if not sampled; this keeps LNS conservative
        return conflict_weight.get((a, b), 0.0)

    # ----------------------
    # Rollout policy (used by MCTS)
    # ----------------------
    def greedy_rollout(prefix, remaining_set, sample_branch=10, epsilon=0.05):
        seq = list(prefix)
        rem = set(remaining_set)
        cur_cost = eval_cost(seq)
        while rem and time_left():
            rem_list = list(rem)
            # Epsilon-greedy exploration in rollout
            if random.random() < epsilon:
                t = random.choice(rem_list)
                seq.append(t)
                rem.remove(t)
                cur_cost = eval_cost(seq)
                continue
            k = min(sample_branch, len(rem_list))
            cand_pool = rem_list if len(rem_list) <= k else random.sample(rem_list, k)
            best_t = None
            best_val = float('inf')
            pfx = tuple(seq)
            for t in cand_pool:
                c = eval_ext(pfx, t)
                # Use slight penalty by expected conflicts with remaining (bias away from "hot" txns)
                # Estimate future penalty as average pair penalty vs a small sample of rem
                if len(rem) > 1:
                    samp = random.sample(rem - {t}, min(4, len(rem) - 1))
                    conf_bias = sum(pair_penalty(t, u) for u in samp) / max(1, len(samp))
                else:
                    conf_bias = 0.0
                val = c + 0.15 * conf_bias
                if val < best_val:
                    best_val = val
                    best_t = t
            if best_t is None:
                best_t = rem_list[0]
                best_val = eval_ext(pfx, best_t)
            seq.append(best_t)
            rem.remove(best_t)
            cur_cost = best_val if isinstance(best_val, (int, float)) else eval_cost(seq)
        return eval_cost(seq), seq

    # ----------------------
    # MCTS constructor
    # ----------------------
    class MCTSNode:
        __slots__ = ("prefix", "rem", "children", "untried", "visits", "value_sum", "best_seen_cost", "best_seen_seq")
        def __init__(self, prefix, rem):
            self.prefix = tuple(prefix)
            self.rem = frozenset(rem)
            self.children = {}  # action t -> node
            self.untried = set(rem)  # actions yet to expand
            self.visits = 0
            self.value_sum = 0.0  # sum of rollout costs (we minimize)
            self.best_seen_cost = float('inf')
            self.best_seen_seq = None

    node_table = {}

    def get_node(prefix, rem_set):
        key = (tuple(prefix), frozenset(rem_set))
        n = node_table.get(key)
        if n is None:
            n = MCTSNode(prefix, rem_set)
            node_table[key] = n
        return n

    def mcts_search(time_budget_frac=0.55):
        # Reserve fraction of total time for MCTS
        end_time = start_time + time_budget_sec * time_budget_frac
        # UCB exploration constant; tuned to balance exploration vs exploitation
        C = 1.25

        # Root initialization
        all_txns = list(range(N))
        root = get_node([], set(all_txns))

        global_best_cost = float('inf')
        global_best_seq = None

        # Warm-start: try a couple of greedy starts from different seeds
        if time_left():
            seeds = random.sample(all_txns, min(6, N))
            for s in seeds:
                if not time_left() or (time.time() > end_time):
                    break
                c0, seq0 = greedy_rollout([s], set(all_txns) - {s}, sample_branch=12)
                if c0 < global_best_cost:
                    global_best_cost, global_best_seq = c0, seq0

        iterations = 0
        while time_left() and (time.time() < end_time):
            iterations += 1
            node = root
            path = [node]
            # Selection
            while node.untried == set() and node.children and node.rem:
                # UCB on children: maximize -mean_cost + C * sqrt(log(Np)/Nc)
                best_score = -1e18
                best_action = None
                for a, child in node.children.items():
                    if child.visits == 0:
                        score = 1e9  # force exploration
                    else:
                        mean = child.value_sum / child.visits
                        score = (-mean) + C * sqrt(max(0.0, log(max(1.0, node.visits)) / child.visits))
                    if score > best_score:
                        best_score = score
                        best_action = a
                if best_action is None:
                    break
                # Move to child
                new_prefix = list(node.prefix) + [best_action]
                new_rem = set(node.rem)
                if best_action in new_rem:
                    new_rem.remove(best_action)
                node = get_node(new_prefix, new_rem)
                path.append(node)

            # Expansion
            if node.rem and node.untried:
                # Choose action to expand: prioritize low marginal ext cost among a sample
                cand_pool = list(node.untried)
                if len(cand_pool) > 12:
                    cand_pool = random.sample(cand_pool, 12)
                pfx = tuple(node.prefix)
                best_a = None
                best_val = float('inf')
                for a in cand_pool:
                    c = eval_ext(pfx, a)
                    # tie-break by singleton cost a bit
                    val = c + 0.05 * singleton_cost.get(a, eval_cost([a]))
                    if val < best_val:
                        best_val = val
                        best_a = a
                if best_a is None:
                    best_a = random.choice(list(node.untried))
                node.untried.discard(best_a)
                new_prefix = list(node.prefix) + [best_a]
                new_rem = set(node.rem)
                if best_a in new_rem:
                    new_rem.remove(best_a)
                child = get_node(new_prefix, new_rem)
                node.children[best_a] = child
                node = child
                path.append(node)

            # Rollout
            rem_set = set(node.rem)
            if rem_set:
                rollout_branch = 10 if len(rem_set) > 30 else 14
                cost_final, seq_final = greedy_rollout(list(node.prefix), rem_set, sample_branch=rollout_branch)
            else:
                cost_final = eval_cost(list(node.prefix))
                seq_final = list(node.prefix)

            # Backpropagation
            for nd in path:
                nd.visits += 1
                nd.value_sum += cost_final
                if cost_final < nd.best_seen_cost:
                    nd.best_seen_cost = cost_final
                    nd.best_seen_seq = seq_final

            if cost_final < global_best_cost:
                global_best_cost, global_best_seq = cost_final, seq_final

        # Fallback
        if global_best_seq is None:
            seq = list(range(N))
            random.shuffle(seq)
            global_best_seq = seq
            global_best_cost = eval_cost(seq)
        return global_best_cost, global_best_seq

    # ----------------------
    # LNS Refinement
    # ----------------------
    def lns_refine(seq, start_cost, time_budget_frac=0.35):
        end_time = start_time + time_budget_sec * (1.0)  # up to total budget
        best_seq = list(seq)
        best_cost = start_cost

        if N <= 2 or not time_left():
            return best_cost, best_seq

        # Helper: compute adjacency penalties along sequence (approx)
        def adjacency_penalties(s):
            pens = []
            for i in range(len(s) - 1):
                a, b = s[i], s[i + 1]
                pens.append((pair_penalty(a, b), i))
            pens.sort(key=lambda x: -x[0])
            return pens

        def reinsert_sequence(base_seq, removed):
            # Greedy reinsertion of removed items into base_seq using sampled positions
            seq_local = list(base_seq)
            for t in removed:
                if not time_left() or time.time() > end_time:
                    break
                best_c = float('inf')
                best_pos = None
                # Candidate positions: endpoints + a few random positions + near high-penalty edges
                positions = {0, len(seq_local)}
                if len(seq_local) > 1:
                    positions.update(random.sample(range(1, len(seq_local)), min(8, len(seq_local) - 1)))
                # Try to bias by adjacency penalties: try around a few worst edges
                ap = adjacency_penalties(seq_local)[:4]
                for _, idx in ap:
                    positions.add(idx + 1)
                for pos in positions:
                    cand = seq_local[:]
                    cand.insert(pos, t)
                    c = eval_cost(cand)
                    if c < best_c:
                        best_c = c
                        best_pos = pos
                if best_pos is None:
                    best_pos = len(seq_local)
                seq_local.insert(best_pos, t)
            return eval_cost(seq_local), seq_local

        # Run multiple LNS iterations until out of time
        iterations = 0
        while time_left() and (time.time() < end_time):
            iterations += 1
            cur_seq = list(best_seq)
            cur_cost = best_cost
            # Choose removal size
            rem_size = max(4, min(18, N // 8))
            # Strategy selection
            strat_roll = random.random()
            removed_idx = set()

            if strat_roll < 0.40:
                # Conflict-guided removal: remove transactions around top-K high-penalty edges
                ap = adjacency_penalties(cur_seq)
                k_edges = min(max(2, rem_size // 3), len(ap))
                chosen_edges = ap[:k_edges]
                for _, idx in chosen_edges:
                    removed_idx.add(idx)
                    removed_idx.add(idx + 1)
                # Top up randomly to reach target size
                while len(removed_idx) < rem_size:
                    removed_idx.add(random.randrange(len(cur_seq)))
            elif strat_roll < 0.75:
                # Random removal
                candidates = list(range(len(cur_seq)))
                removed_idx = set(random.sample(candidates, min(rem_size, len(candidates))))
            else:
                # Contiguous block removal
                if len(cur_seq) > rem_size:
                    start = random.randrange(0, len(cur_seq) - rem_size + 1)
                    removed_idx = set(range(start, start + rem_size))
                else:
                    removed_idx = set(range(len(cur_seq)))

            removed_idx = sorted(list(removed_idx))
            removed_items = [cur_seq[i] for i in removed_idx]
            base_seq = [cur_seq[i] for i in range(len(cur_seq)) if i not in removed_idx]

            # Reinsert
            new_cost, new_seq = reinsert_sequence(base_seq, removed_items)

            # Accept if improved
            if new_cost < best_cost:
                best_cost, best_seq = new_cost, new_seq
            else:
                # Occasionally accept equal-cost to escape plateaus
                if new_cost == best_cost and random.random() < 0.1:
                    best_cost, best_seq = new_cost, new_seq

            # Small adjacent-swap polish on occasion
            if iterations % 3 == 0 and time_left():
                for i in range(min(8, len(best_seq) - 1)):
                    idx = random.randrange(0, len(best_seq) - 1)
                    cand = best_seq[:]
                    cand[idx], cand[idx + 1] = cand[idx + 1], cand[idx]
                    c = eval_cost(cand)
                    if c < best_cost:
                        best_cost, best_seq = c, cand

        return best_cost, best_seq

    # ----------------------
    # Orchestrate phases
    # ----------------------
    # Use MCTS to construct a strong schedule, then refine via LNS
    mcts_cost, mcts_seq = mcts_search(time_budget_frac=0.55)
    final_cost, final_seq = lns_refine(mcts_seq, mcts_cost, time_budget_frac=0.35)

    # Fallback if something failed
    if not final_seq or len(final_seq) != N:
        seq = list(range(N))
        random.shuffle(seq)
        final_seq = seq
        final_cost = eval_cost(seq)

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