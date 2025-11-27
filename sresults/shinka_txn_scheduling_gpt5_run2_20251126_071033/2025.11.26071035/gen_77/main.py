# EVOLVE-BLOCK-START
"""Transaction scheduling algorithm for optimizing makespan across multiple workloads
PUCT-style MCTS with insertion actions + exact endgame enumeration + VND polish.
"""

import time
import random
import sys
import os
from math import sqrt, inf, exp
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
    PUCT-style MCTS with best-insertion actions, exact endgame enumeration, and VND polish.

    Args:
        workload: Workload object containing transaction data
        num_seqs: Search effort parameter (controls MCTS iterations and polish depth)

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    n = workload.num_txns
    all_txns = list(range(n))

    # ---------------- Parameters (adaptive by problem size) ----------------
    small = n <= 40
    med = 40 < n <= 90
    large = n > 90

    # Cost/Insertion evaluation
    POS_SAMPLE_CAP = None if small else (22 if med else 18)  # None -> evaluate all positions
    EXHAUSTIVE_POS_THRESHOLD = 20  # evaluate all positions when seq length <= this

    # MCTS core
    base_cpuct = 1.5 if small else (1.2 if med else 1.0)
    iter_per_txn = 12 if small else (9 if med else 7)
    MCTS_ITERS = max(400, min(3000, iter_per_txn * n + num_seqs * 40))
    MCTS_TIME_CAP = None  # optional time cap (seconds); keep None for deterministic iteration cap
    PRIOR_TEMPERATURE = 0.25  # softmax temperature over negative best-insertion costs

    # Endgame exact enumeration
    ENDGAME_K = 9 if not large else 8  # switch to DFS exact when |remaining| <= K

    # Light local search
    VND_ROUNDS = 2 if large else 3
    LS_ADJ_PASSES = 2 if large else 3
    REV_SAMPLES = min(200, max(80, n))

    # RNG seed for reproducibility but workload-diverse behavior
    random.seed((n * 1315423911 + num_seqs * 2654435761 + 17) % (2**32 - 1))

    # ---------------- Shared episode caches ----------------
    # Cost cache for partial prefixes (LRU bounded for memory safety)
    @lru_cache(maxsize=300_000)
    def _eval_cost_tuple(seq_tuple):
        return workload.get_opt_seq_cost(list(seq_tuple))

    def eval_cost(seq_list):
        return _eval_cost_tuple(tuple(seq_list))

    # Deterministic position sampling keyed by (seq_len)
    pos_signature_cache = {}

    def positions_for_insertion(seq_len, exhaustive=False):
        total = seq_len + 1
        if exhaustive or POS_SAMPLE_CAP is None or total <= POS_SAMPLE_CAP:
            return list(range(total))
        sig = pos_signature_cache.get(seq_len)
        if sig is not None:
            return sig
        anchors = {0, seq_len, seq_len // 2, seq_len // 4, (3 * seq_len) // 4}
        anchors = {p for p in anchors if 0 <= p <= seq_len}
        need = max(2, POS_SAMPLE_CAP - len(anchors))
        # Evenly spaced interior
        for i in range(1, need + 1):
            pos = round(i * seq_len / (need + 1))
            anchors.add(max(0, min(seq_len, pos)))
        sig = sorted(anchors)
        pos_signature_cache[seq_len] = sig
        return sig

    # Best-two insertion cache: (tuple(seq), txn, sig_tag) -> (best_cost, best_pos, second_cost)
    best_two_cache = {}

    def best_two_insertions(seq, txn):
        seq_t = tuple(seq)
        exhaustive = (len(seq) <= EXHAUSTIVE_POS_THRESHOLD)
        sig_tag = ('all', len(seq)) if (POS_SAMPLE_CAP is None or exhaustive) else ('sig', len(seq))
        key = (seq_t, txn, sig_tag)
        hit = best_two_cache.get(key)
        if hit is not None:
            return hit
        positions = positions_for_insertion(len(seq), exhaustive=exhaustive)
        best_c, best_p = float('inf'), 0
        second_c = float('inf')
        for p in positions:
            cand = seq[:]
            cand.insert(p, txn)
            c = eval_cost(cand)
            if c < best_c:
                second_c = best_c
                best_c, best_p = c, p
            elif c < second_c:
                second_c = c
        if second_c == float('inf'):
            second_c = best_c
        res = (best_c, best_p, second_c)
        best_two_cache[key] = res
        return res

    # ---------------- Endgame exact enumeration with memoized BnB ----------------
    endgame_cache = {}

    def endgame_optimal(prefix, remaining_fset, global_bound):
        """
        Compute optimal completion for small remaining set via DFS + memo + pruning.
        Returns (best_cost, best_seq).
        """
        key = (tuple(prefix), remaining_fset)
        if key in endgame_cache:
            return endgame_cache[key]

        # Monotonic bound: makespan cannot be less than current prefix cost
        base_cost = eval_cost(prefix)
        if global_bound is not None and base_cost >= global_bound - 1e-12:
            return (inf, None)

        if not remaining_fset:
            endgame_cache[key] = (base_cost, list(prefix))
            return endgame_cache[key]

        best_cost = inf
        best_seq = None
        remaining = list(remaining_fset)

        # Order candidates by cheapest immediate insertion to help pruning
        candidates = []
        for t in remaining:
            c1, p1, _ = best_two_insertions(prefix, t)
            candidates.append((c1, t, p1))
        candidates.sort(key=lambda x: x[0])

        for c1, t, p1 in candidates:
            # pruning by current best cost
            if global_bound is not None and c1 >= global_bound - 1e-12:
                continue
            new_seq = prefix[:]
            new_seq.insert(p1, t)
            new_rem = frozenset(x for x in remaining_fset if x != t)
            c_child, s_child = endgame_optimal(new_seq, new_rem, global_bound if best_cost == inf else min(best_cost, global_bound))
            if c_child < best_cost:
                best_cost, best_seq = c_child, s_child
                if global_bound is not None and best_cost <= global_bound - 1e-12:
                    global_bound = best_cost
        endgame_cache[key] = (best_cost, best_seq)
        return endgame_cache[key]

    # ---------------- Rollout policy (greedy best-insertion with regret tie-break) ----------------
    def greedy_completion(prefix, remaining_set):
        seq = list(prefix)
        rem = set(remaining_set)
        while rem:
            # evaluate best insertion for a sample (or all when small)
            cand_txns = rem if len(rem) <= 12 else set(random.sample(list(rem), 12))
            best_t, best_p, best_c, best_regret = None, 0, float('inf'), -1.0
            for t in cand_txns:
                c1, p1, c2 = best_two_insertions(seq, t)
                regret = max(0.0, c2 - c1)
                if c1 + 1e-12 < best_c or (abs(c1 - best_c) <= 1e-12 and regret > best_regret):
                    best_c, best_t, best_p, best_regret = c1, t, p1, regret
            if best_t is None:
                # fallback: pick any remaining and append
                t = rem.pop()
                seq.append(t)
            else:
                seq.insert(best_p, best_t)
                rem.remove(best_t)
        return eval_cost(seq), seq

    # ---------------- MCTS with PUCT ----------------
    class Node:
        __slots__ = ('seq', 'cost', 'N', 'W', 'children', 'priors', 'best_pos', 'remaining', 'expanded')
        def __init__(self, seq):
            self.seq = list(seq)
            self.cost = eval_cost(self.seq)
            self.N = 0
            self.W = 0.0   # cumulative reward (negative cost)
            self.children = {}   # txn -> Node
            self.priors = {}     # txn -> prior probability
            self.best_pos = {}   # txn -> best insertion position at this node
            self.remaining = None
            self.expanded = False

        def compute_remaining(self):
            if self.remaining is None:
                in_seq = set(self.seq)
                self.remaining = [t for t in all_txns if t not in in_seq]
            return self.remaining

        def is_terminal(self):
            return len(self.seq) == n

        def ensure_priors(self, temperature=PRIOR_TEMPERATURE):
            if self.expanded:
                return
            rem_list = self.compute_remaining()
            if not rem_list:
                self.priors = {}
                self.best_pos = {}
                self.expanded = True
                return
            # Compute best-insertion costs as action-values for priors
            vals = []
            for t in rem_list:
                c1, p1, _ = best_two_insertions(self.seq, t)
                vals.append((t, c1, p1))
            # Softmax over negative costs for priors (lower cost -> higher prob)
            min_c = min(c for _, c, _ in vals)
            # stabilize by subtracting min
            logits = []
            for t, c, p in vals:
                logits.append((t, - (c - min_c) / max(1e-9, temperature)))
            # compute probabilities
            exps = [(t, exp(v)) for (t, v) in logits]
            Z = sum(v for _, v in exps) or 1.0
            self.priors = {t: v / Z for (t, v) in exps}
            self.best_pos = {t: p for (t, _, p) in vals}
            self.expanded = True

        def select_action(self, cpuct=base_cpuct):
            """
            Return the txn with maximum PUCT value and its child (creating it if needed).
            """
            self.ensure_priors()
            rem = self.compute_remaining()
            if not rem:
                return None, None
            sqrt_N = sqrt(self.N + 1e-9)
            best_val = -float('inf')
            best_txn = None
            best_child = None
            for t in rem:
                prior = self.priors.get(t, 0.0)
                child = self.children.get(t)
                if child is None or child.N == 0:
                    q = 0.0  # optimistic unknown
                    n_child = 0
                else:
                    q = child.W / child.N
                    n_child = child.N
                u = q + cpuct * prior * (sqrt_N / (1.0 + n_child))
                if u > best_val:
                    best_val = u
                    best_txn = t
                    best_child = child
            # Create child if absent
            if best_child is None:
                new_seq = self.seq[:]
                pos = self.best_pos.get(best_txn, len(new_seq))
                new_seq.insert(pos, best_txn)
                best_child = Node(new_seq)
                self.children[best_txn] = best_child
            return best_txn, best_child

    # Initialize root with a strong heuristic start (best single + good second by insertion)
    def initial_seed():
        # choose the best single starter by cost
        start_candidates = random.sample(all_txns, min(10, n)) if n > 10 else all_txns
        starter = min(start_candidates, key=lambda t: eval_cost([t])) if start_candidates else 0
        seq = [starter]
        # add a strong second
        remaining = [t for t in all_txns if t != starter]
        if remaining:
            trials = []
            for t in random.sample(remaining, min(8, len(remaining))):
                for pos in [0, 1]:
                    cand = seq[:]
                    cand.insert(pos, t)
                    trials.append((eval_cost(cand), t, pos))
            if trials:
                trials.sort(key=lambda x: x[0])
                _, t2, p2 = trials[0]
                seq.insert(p2, t2)
        return Node(seq)

    root = initial_seed()
    best_global_cost = eval_cost(root.seq)
    best_global_seq = list(root.seq)

    # Fast baseline as an incumbent bound: greedy full construction from root with endgame exact
    def complete_with_endgame(prefix):
        rem = [t for t in all_txns if t not in set(prefix)]
        if len(rem) <= ENDGAME_K:
            c, s = endgame_optimal(prefix, frozenset(rem), best_global_cost)
            return c, s
        # else greedy to reduce to <=K then exact
        seq = list(prefix)
        remaining = set(rem)
        while len(remaining) > ENDGAME_K:
            # greedily choose best insertion among a small sample
            cand_txns = remaining if len(remaining) <= 14 else set(random.sample(list(remaining), 14))
            best_t, best_p, best_c = None, 0, float('inf')
            for t in cand_txns:
                c1, p1, _ = best_two_insertions(seq, t)
                if c1 < best_c:
                    best_c, best_t, best_p = c1, t, p1
            if best_t is None:
                t = remaining.pop()
                seq.append(t)
            else:
                seq.insert(best_p, best_t)
                remaining.remove(best_t)
        c_final, s_final = endgame_optimal(seq, frozenset(remaining), best_global_cost)
        return c_final, s_final

    # ---------------- Run MCTS iterations ----------------
    start_time = time.time()
    for it in range(MCTS_ITERS):
        if MCTS_TIME_CAP is not None and (time.time() - start_time) >= MCTS_TIME_CAP:
            break

        # SELECTION/EXPANSION
        path = []
        node = root
        path.append(node)
        while not node.is_terminal():
            t, child = node.select_action(cpuct=base_cpuct)
            if t is None:
                break
            path.append(child)
            node = child
            # Expand only one new node per iteration (standard MCTS)
            if child.N == 0:
                break

        leaf = node

        # ROLLOUT / EVALUATION
        if leaf.is_terminal():
            final_cost = leaf.cost
            rollout_seq = list(leaf.seq)
        else:
            # Mixed policy: exact endgame when close, else greedy + exact tail
            final_cost, rollout_seq = complete_with_endgame(leaf.seq)

        # Update global incumbent
        if final_cost < best_global_cost:
            best_global_cost = final_cost
            best_global_seq = rollout_seq

        # BACKPROPAGATION (reward = -cost)
        reward = -final_cost
        for nd in path:
            nd.N += 1
            nd.W += reward

        # Small exploration decay for cpuct over time
        if (it + 1) % max(50, n // 2 + 1) == 0:
            base_cpuct = max(0.6, base_cpuct * 0.98)  # anneal exploration slightly

    # ---------------- Derive schedule from MCTS visit counts ----------------
    # Follow most visited actions from root; fall back to greedy for unseen branches
    seq_build = list(root.seq)
    while len(seq_build) < n:
        # Recreate node for current prefix
        node_key = tuple(seq_build)
        # Build a temporary node to access children if not in tree
        # Use the same prior + selection to choose next
        tm_node = Node(seq_build)
        tm_node.ensure_priors()
        remaining = tm_node.compute_remaining()
        # If we have stats from root's descendants, use N counts; else use priors
        best_action = None
        best_score = -float('inf')
        for t in remaining:
            pos = tm_node.best_pos.get(t, len(seq_build))
            child_seq = seq_build[:]
            child_seq.insert(pos, t)
            # check if this child exists via root-based exploration (we don't have a global node table),
            # so proxy score via prior and immediate cost
            c1 = eval_cost(child_seq)
            prior = tm_node.priors.get(t, 0.0)
            # Lower cost, higher prior -> higher score
            score = -c1 + 0.5 * prior * n
            if score > best_score:
                best_score = score
                best_action = (t, pos)
        if best_action is None:
            # fallback: greedy best-insertion
            best_t, best_p, best_c = None, 0, float('inf')
            for t in remaining:
                c1, p1, _ = best_two_insertions(seq_build, t)
                if c1 < best_c:
                    best_c, best_t, best_p = c1, t, p1
            if best_t is None:
                # absolute fallback
                best_t, best_p = remaining[0], len(seq_build)
        else:
            best_t, best_p = best_action
        seq_build.insert(best_p, best_t)

    mcts_cost = eval_cost(seq_build)
    if mcts_cost < best_global_cost:
        best_global_cost, best_global_seq = mcts_cost, seq_build

    # ---------------- Local Search polish (light VND) ----------------
    def adjacent_swap_pass(seq, start_cost, passes=LS_ADJ_PASSES):
        best_seq = list(seq)
        best_cost = start_cost
        for _ in range(max(1, passes)):
            improved = False
            for i in range(len(best_seq) - 1):
                cand = best_seq[:]
                cand[i], cand[i + 1] = cand[i + 1], cand[i]
                c = eval_cost(cand)
                if c + 1e-12 < best_cost:
                    best_cost, best_seq, improved = c, cand, True
            if not improved:
                break
        return best_seq, best_cost

    def or_opt_pass(seq, start_cost, k):
        best_seq = list(seq)
        best_cost = start_cost
        L = len(best_seq)
        if L <= k:
            return best_seq, best_cost
        improved = True
        while improved:
            improved = False
            move_best_c = best_cost
            move = None
            for i in range(0, L - k + 1):
                block = best_seq[i:i + k]
                base = best_seq[:i] + best_seq[i + k:]
                positions = positions_for_insertion(len(base), exhaustive=(len(base) <= EXHAUSTIVE_POS_THRESHOLD))
                for p in positions:
                    if k == 1 and p == i:
                        continue
                    cand = base[:]
                    cand[p:p] = block
                    c = eval_cost(cand)
                    if c + 1e-12 < move_best_c:
                        move_best_c = c
                        move = (i, p, list(cand))
            if move is not None:
                _, _, cand = move
                best_seq = cand
                best_cost = move_best_c
                L = len(best_seq)
                improved = True
        return best_seq, best_cost

    def sampled_segment_reversal(seq, start_cost, samples=REV_SAMPLES):
        best_seq = list(seq)
        best_cost = start_cost
        L = len(best_seq)
        if L <= 5:
            return best_seq, best_cost
        best_delta = 0.0
        best_move = None
        for _ in range(samples):
            i = random.randint(0, L - 3)
            j = random.randint(i + 2, min(L - 1, i + 12))
            cand = best_seq[:]
            cand[i:j + 1] = reversed(cand[i:j + 1])
            c = eval_cost(cand)
            delta = best_cost - c
            if delta > best_delta:
                best_delta = delta
                best_move = (i, j, c, cand)
        if best_move is not None:
            _, _, c, cand = best_move
            best_seq, best_cost = cand, c
        return best_seq, best_cost

    def vnd_polish(seq, cost, rounds=VND_ROUNDS):
        best_seq, best_cost = list(seq), cost
        for _ in range(max(1, rounds)):
            s, c = or_opt_pass(best_seq, best_cost, 2)
            if c < best_cost:
                best_seq, best_cost = s, c
            s, c = or_opt_pass(best_seq, best_cost, 1)
            if c < best_cost:
                best_seq, best_cost = s, c
            s, c = adjacent_swap_pass(best_seq, best_cost, passes=1)
            if c < best_cost:
                best_seq, best_cost = s, c
            s, c = sampled_segment_reversal(best_seq, best_cost, samples=REV_SAMPLES // 2)
            if c < best_cost:
                best_seq, best_cost = s, c
        return best_seq, best_cost

    best_global_seq, best_global_cost = vnd_polish(best_global_seq, best_global_cost)

    # Safety checks
    assert len(best_global_seq) == n and len(set(best_global_seq)) == n, "Schedule must include each transaction exactly once"

    return best_global_cost, best_global_seq


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