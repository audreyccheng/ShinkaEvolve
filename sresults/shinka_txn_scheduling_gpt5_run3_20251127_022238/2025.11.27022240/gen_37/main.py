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
    UCT-based Monte Carlo Tree Search scheduler with:
    - Progressive widening and RAVE adjacency guidance
    - Memoized prefix/extension costs
    - Greedy rollouts with incumbent pruning
    - Light 2-opt-style refinement
    """
    N = workload.num_txns
    start_time = time.time()

    # Balanced time budget per workload
    # Slightly adaptive with N but capped to keep combined score strong
    time_budget_sec = 0.55 + min(0.07, 0.001 * max(0, N - 60))

    rng = random.Random(314159 + N + int(num_seqs))

    def time_left():
        return (time.time() - start_time) < time_budget_sec

    # Bit utilities for fast set ops on remaining txns
    BIT = [1 << i for i in range(N)]
    ALL_MASK = (1 << N) - 1

    def bits_iter(mask):
        while mask:
            lb = mask & -mask
            i = (lb.bit_length() - 1)
            yield i
            mask ^= lb

    # Cached cost computations
    cost_cache = {}
    ext_cache = {}

    def eval_seq_cost(seq):
        key = tuple(seq)
        c = cost_cache.get(key)
        if c is None:
            c = workload.get_opt_seq_cost(seq)
            cost_cache[key] = c
        return c

    def eval_ext_cost(prefix_tuple, cand):
        key = (prefix_tuple, cand)
        c = ext_cache.get(key)
        if c is None:
            c = eval_seq_cost(list(prefix_tuple) + [cand])
            ext_cache[key] = c
        return c

    # Precompute singleton costs
    singleton_cost = [0] * N
    for t in range(N):
        if not time_left():
            break
        singleton_cost[t] = eval_seq_cost([t])

    # Build small buddy lists for each txn using sampled pairwise ext costs
    buddies = [[] for _ in range(N)]
    def build_buddies():
        # Candidate pool per txn: mix of globally low singleton and random
        order_singleton = sorted(range(N), key=lambda t: singleton_cost[t])
        max_buddies = min(8, max(6, N // 18))
        max_pool = min(N, 24)
        probe_budget = min(900, N * 10)
        probes_used = 0
        for t in range(N):
            if not time_left():
                return
            pool = []
            # Top by singleton
            pool.extend(order_singleton[:min(max_pool, len(order_singleton))])
            # Some randoms
            if N > max_pool:
                pool.extend(rng.sample(range(N), min(N - 1, max_pool)))
            # Deduplicate and remove self
            seen = set()
            cand_pool = []
            for u in pool:
                if u == t or u in seen:
                    continue
                seen.add(u)
                cand_pool.append(u)
            # Score by true two-step cost delta
            scored = []
            base = singleton_cost[t]
            pref = (t,)
            for u in cand_pool:
                if probes_used >= probe_budget or not time_left():
                    break
                c2 = eval_ext_cost(pref, u)
                scored.append((c2 - base, u))
                probes_used += 1
            scored.sort(key=lambda x: x[0])
            buddies[t] = [u for _d, u in scored[:max_buddies]]
    build_buddies()

    # RAVE-like adjacency stats: reward per observed pair (a -> b)
    # Reward is negative makespan so that higher is better
    rave_sum = defaultdict(float)
    rave_cnt = defaultdict(int)

    # Global incumbent to prune bad branches quickly
    incumbent_cost = float('inf')
    incumbent_seq = None

    # Node storage with transpositions: key = (remaining_mask, suffix_tuple)
    # Each node: dict with fields
    #   'visits', 'value' (sum rewards), 'children' (a->child_key),
    #   'edge_stats' (a-> [n,w]), 'cand_order' (list of actions sorted), 'opened' (int: how many allowed)
    nodes = {}

    suffix_k = 3

    def make_key(rem_mask, suffix):
        if len(suffix) > suffix_k:
            suffix = suffix[-suffix_k:]
        return (rem_mask, tuple(suffix))

    def get_node(key):
        node = nodes.get(key)
        if node is None:
            node = {
                'visits': 0,
                'value': 0.0,
                'children': {},
                'edge_stats': {},   # a -> [n, w]
                'cand_order': None, # cached candidate action order
                'opened': 0,        # number of actions opened under progressive widening
            }
            nodes[key] = node
        return node

    def allowed_children_count(visits):
        # Progressive widening: increases slowly with visits
        # ensure at least 2; up to ~20 for enough breadth
        base = 2 + int(1.6 * (visits ** 0.5))
        return max(2, min(20, base))

    # Candidate generator for a node: sort actions by extension cost from current prefix
    def gen_candidates(prefix, rem_mask):
        last = prefix[-1] if prefix else None
        rem_list = list(bits_iter(rem_mask))
        if not rem_list:
            return []

        pool = []
        # Bias: buddies of last come first
        if last is not None:
            for u in buddies[last]:
                if (rem_mask & BIT[u]) != 0:
                    pool.append(u)
        # Add top low-singleton from remaining
        if len(rem_list) > 0:
            low_single = sorted(rem_list, key=lambda t: singleton_cost[t])[:min(6, len(rem_list))]
            pool.extend(low_single)

        # Fill with randoms up to pool size cap
        pool_cap = min(24, max(12, N // 5))
        if len(pool) < min(pool_cap, len(rem_list)):
            remain = [t for t in rem_list if t not in pool]
            if remain:
                pool.extend(rng.sample(remain, min(pool_cap - len(pool), len(remain))))

        # Dedup
        pool = list(dict.fromkeys(pool))
        # Rank by true extension cost
        prefix_tuple = tuple(prefix)
        scored = []
        for t in pool:
            if not time_left():
                break
            ec = eval_ext_cost(prefix_tuple, t)
            if ec >= incumbent_cost:
                continue
            scored.append((ec, t))
        if not scored:
            # fallback: evaluate a couple randoms
            for t in rem_list[:min(4, len(rem_list))]:
                ec = eval_ext_cost(prefix_tuple, t)
                scored.append((ec, t))
        scored.sort(key=lambda x: x[0])
        return [t for _c, t in scored]

    # Greedy rollout completion with incumbent pruning
    def rollout_greedy(prefix, rem_mask):
        seq = list(prefix)
        cur_mask = rem_mask
        while cur_mask and time_left():
            # Candidate pool
            cand = gen_candidates(seq, cur_mask)
            if not cand:
                # if budget-thin, take any remaining (should be rare)
                # choose by lowest singleton heuristic
                rem_list = list(bits_iter(cur_mask))
                rem_list.sort(key=lambda t: singleton_cost[t])
                t = rem_list[0]
            else:
                t = cand[0]
            seq.append(t)
            cur_mask &= ~BIT[t]
        cost = eval_seq_cost(seq)
        return cost, seq

    # UCT selection score with RAVE blend
    def uct_rave_score(parent_node, last, action_t, child_key):
        # Edge stats from parent for this action
        n_edge, w_edge = parent_node['edge_stats'].get(action_t, [0, 0.0])
        q_edge = (w_edge / n_edge) if n_edge > 0 else 0.0

        # Child stats if available
        child = nodes.get(child_key)
        if child and child['visits'] > 0:
            q_child = child['value'] / child['visits']
            # prefer child estimate if better populated
            if child['visits'] > n_edge:
                q_edge = q_child

        # RAVE statistics for adjacency (last -> action_t)
        if last is not None:
            rs = rave_sum.get((last, action_t), 0.0)
            rc = rave_cnt.get((last, action_t), 0)
            q_rave = (rs / rc) if rc > 0 else 0.0
        else:
            q_rave = 0.0
            rc = 0

        # Blend
        # n0 controls RAVE influence on low-visit edges
        n0 = 50.0
        beta = n0 / (n_edge + n0) if n_edge >= 0 else 1.0
        q_blend = (1.0 - beta) * q_edge + beta * q_rave

        # UCB exploration term
        Nv = max(1, parent_node['visits'])
        ne = max(1, n_edge)
        c = 1.45
        ucb = q_blend + c * (Nv ** 0.5) / (ne ** 0.5)
        return ucb

    # Main MCTS loop
    # Root state: prefix=[], rem_mask=ALL_MASK, suffix=()
    root_key = make_key(ALL_MASK, ())
    get_node(root_key)

    # Establish a baseline incumbent via one greedy full rollout from root
    if time_left():
        base_cost, base_seq = rollout_greedy([], ALL_MASK)
        incumbent_cost = base_cost
        incumbent_seq = base_seq

    # Iterative simulations within budget
    while time_left():
        # Selection
        path = []
        prefix = []
        rem_mask = ALL_MASK
        suffix = []
        cur_key = make_key(rem_mask, suffix)
        node = get_node(cur_key)

        # Keep selecting down the tree until we can expand a new action
        expanded = False
        while True:
            if rem_mask == 0:
                break
            # Generate candidates if not cached
            if node['cand_order'] is None:
                node['cand_order'] = gen_candidates(prefix, rem_mask)
                node['opened'] = 0

            # Progressive widening: allow only a subset of children based on visits
            allow = allowed_children_count(node['visits'])
            if node['opened'] < min(allow, len(node['cand_order'])):
                # Expand the next best not-yet-opened action
                a = node['cand_order'][node['opened']]
                node['opened'] += 1
                # Take action a
                prefix.append(a)
                rem_mask &= ~BIT[a]
                last = suffix[-1] if suffix else None
                suffix = (suffix + [a]) if len(suffix) < suffix_k else (suffix + [a])[1:]
                child_key = make_key(rem_mask, suffix)
                child = get_node(child_key)
                node['children'][a] = child_key
                # Record edge path
                if a not in node['edge_stats']:
                    node['edge_stats'][a] = [0, 0.0]
                path.append((node, a, last))
                node = child
                expanded = True
                break
            else:
                # Choose among opened actions using UCT+RAVE
                opened_actions = node['cand_order'][:node['opened']] if node['cand_order'] else []
                if not opened_actions:
                    # No candidates; stop selection
                    break
                last = suffix[-1] if suffix else None
                best_a = None
                best_score = -float('inf')
                for a in opened_actions:
                    child_key = node['children'].get(a)
                    if child_key is None:
                        # Might be unopened mapping in rare race; skip
                        continue
                    score = uct_rave_score(node, last, a, child_key)
                    if score > best_score:
                        best_score = score
                        best_a = a
                if best_a is None:
                    break
                # Step to chosen child
                prefix.append(best_a)
                rem_mask &= ~BIT[best_a]
                suffix = (suffix + [best_a]) if len(suffix) < suffix_k else (suffix + [best_a])[1:]
                child_key = node['children'][best_a]
                child = get_node(child_key)
                path.append((node, best_a, last))
                node = child
                # Continue loop; expansion will occur when widening opens new actions
                continue

        # Rollout from current prefix
        cost, full_seq = rollout_greedy(prefix, rem_mask)
        reward = -float(cost)

        # Update incumbent
        if cost < incumbent_cost:
            incumbent_cost = cost
            incumbent_seq = full_seq

        # Backpropagation along the path
        for nd, action_t, last in path:
            nd['visits'] += 1
            nd['value'] += reward
            n_edge, w_edge = nd['edge_stats'].get(action_t, [0, 0.0])
            n_edge += 1
            w_edge += reward
            nd['edge_stats'][action_t] = [n_edge, w_edge]
            # Update RAVE for adjacency (last -> action_t)
            if last is not None:
                key = (last, action_t)
                rave_sum[key] += reward
                rave_cnt[key] += 1

        # Also update stats at leaf node
        node['visits'] += 1
        node['value'] += reward

    # Light 2-opt-style refinement on incumbent
    def polish(seq, cur_cost, tries=50):
        best_seq = list(seq)
        best_cost = cur_cost
        n = len(best_seq)
        if n <= 2:
            return best_cost, best_seq
        # Adjacent swap pass
        for i in range(n - 1):
            if not time_left():
                break
            cand = best_seq[:]
            cand[i], cand[i + 1] = cand[i + 1], cand[i]
            c = eval_seq_cost(cand)
            if c < best_cost:
                best_cost = c
                best_seq = cand
        # Limited relocate moves
        k = tries
        while k > 0 and time_left():
            k -= 1
            i, j = rng.sample(range(n), 2)
            if i == j:
                continue
            cand = best_seq[:]
            v = cand.pop(i)
            cand.insert(j, v)
            c = eval_seq_cost(cand)
            if c < best_cost:
                best_cost = c
                best_seq = cand
        return best_cost, best_seq

    if incumbent_seq is None:
        seq = list(range(N))
        rng.shuffle(seq)
        incumbent_seq = seq
        incumbent_cost = eval_seq_cost(seq)
    else:
        # Ensure validity and polish
        if len(incumbent_seq) != N or len(set(incumbent_seq)) != N:
            # Repair permutation
            seen = set()
            repaired = []
            for t in incumbent_seq:
                if 0 <= t < N and t not in seen:
                    repaired.append(t)
                    seen.add(t)
            for t in range(N):
                if t not in seen:
                    repaired.append(t)
            incumbent_seq = repaired[:N]
            incumbent_cost = eval_seq_cost(incumbent_seq)
        if time_left():
            incumbent_cost, incumbent_seq = polish(incumbent_seq, incumbent_cost, tries=40)

    return incumbent_cost, incumbent_seq


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