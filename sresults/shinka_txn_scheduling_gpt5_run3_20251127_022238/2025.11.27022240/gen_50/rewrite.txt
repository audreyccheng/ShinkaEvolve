# EVOLVE-BLOCK-START
"""Transaction scheduling algorithm for optimizing makespan across multiple workloads"""

import time
import random
import sys
import os
import itertools
from math import sqrt, log, exp


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
    MCTS with PUCT selection, progressive widening, incumbent pruning,
    and ΔW-gated LNS post-optimization.
    """
    n = workload.num_txns
    rng = random.Random(1729 + 17 * n + int(num_seqs) * 3)

    # Global memoized evaluator
    cost_cache = {}

    def eval_seq(seq):
        key = tuple(seq)
        if key in cost_cache:
            return cost_cache[key]
        c = workload.get_opt_seq_cost(seq)
        cost_cache[key] = c
        return c

    # Precompute singleton and pairwise costs
    c1 = [eval_seq([i]) for i in range(n)]
    M = [[0] * n for _ in range(n)]
    for i in range(n):
        Mi = M[i]
        for j in range(n):
            Mi[j] = c1[i] if i == j else eval_seq([i, j])

    # Preference margins W[i][j] = M[i][j] - M[j][i]; positive -> i before j is worse
    W = [[0] * n for _ in range(n)]
    for i in range(n):
        Wi = W[i]
        for j in range(n):
            Wi[j] = 0 if i == j else (M[i][j] - M[j][i])

    # Tournament ranking (lower score -> earlier)
    s = [0] * n
    for i in range(n):
        s[i] = sum(W[i][j] for j in range(n) if j != i)
    tournament_order = list(range(n))
    tournament_order.sort(key=lambda x: (s[x], x))

    # Buddy lists: top partners by small M[t][u]
    buddy_k = 8 if n >= 90 else 6
    buddies = []
    for t in range(n):
        order = sorted((u for u in range(n) if u != t), key=lambda u: M[t][u])
        buddies.append(order[:buddy_k])

    def preselect_by_tournament(prefix, remaining, k, recent_k=4):
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

    # Greedy rollout policy with shallow lookahead and buddy preference
    def greedy_rollout(prefix, remaining):
        seq = list(prefix)
        rem = list(remaining)
        while rem:
            R = len(rem)
            # Tournament-guided preselection from last few recents
            k_pool = R if R <= 32 else min(16, R)
            cand_pool = preselect_by_tournament(seq, rem, k_pool, recent_k=4) if R > 16 else rem[:]
            if not cand_pool:
                cand_pool = rem[:]
            # immediate costs
            imm = [(t, eval_seq(seq + [t])) for t in cand_pool]
            imm.sort(key=lambda x: x[1])
            # Limited lookahead on top-3
            L = min(3, len(imm))
            best_t, best_metric = imm[0][0], imm[0][1]
            for t, ic in imm[:L]:
                rest = [u for u in rem if u != t]
                if not rest:
                    metric = ic
                else:
                    need = 6 if R <= 40 else 4
                    la_b = [u for u in buddies[t] if u in rest][:need]
                    if len(la_b) < need:
                        extra = preselect_by_tournament(seq + [t], [u for u in rest if u not in la_b], need - len(la_b), recent_k=3)
                        la_pool = la_b + extra
                    else:
                        la_pool = la_b
                    if not la_pool:
                        la_pool = rest[:min(need, len(rest))]
                    metric = min(eval_seq(seq + [t, u]) for u in la_pool)
                if metric < best_metric:
                    best_metric = metric
                    best_t = t
            seq.append(best_t)
            rem.remove(best_t)
        return seq, eval_seq(seq)

    # Baseline using greedy rollout from scratch for stable MCTS rewards
    baseline_seq, baseline_cost = greedy_rollout([], list(range(n)))

    # Prefix dominance map shared across MCTS to prune
    dom = {}  # key: (frozenset(remaining), suffix<=3) -> best partial cost seen

    # MCTS parameters
    sims_budget = max(1600, min(5200, 1000 + 30 * n + 100 * int(num_seqs)))
    cpuct = 1.2
    tau = 0.03  # prior softmax temperature
    alpha_pw = 0.5
    c_pw = 1.6

    best_global_cost = baseline_cost
    best_global_seq = baseline_seq

    class Node:
        __slots__ = ("seq", "rem", "N", "Wsum", "children", "priors", "action_order", "partial_cost")

        def __init__(self, seq_tuple, rem_fset):
            self.seq = seq_tuple
            self.rem = rem_fset
            self.N = 0
            self.Wsum = 0.0  # sum of rewards (baseline - rollout_cost), maximize
            self.children = {}  # t -> Node
            # Precompute priors and action_order
            prefix = list(self.seq)
            remaining = list(self.rem)
            self.partial_cost = eval_seq(prefix)
            if not remaining:
                self.priors = {}
                self.action_order = []
                return

            # Candidate generation: tournament-guided when large
            if len(remaining) > 36:
                pre = preselect_by_tournament(prefix, remaining, min(18, len(remaining)), recent_k=4)
                cand = pre if pre else remaining
            else:
                cand = remaining

            # Immediate costs for priors
            imms = []
            for t in cand:
                c = eval_seq(prefix + [t])
                imms.append((t, c))
            imms.sort(key=lambda z: z[1])

            if imms:
                minc = imms[0][1]
                maxc = imms[-1][1]
                span = max(1e-6, maxc - minc)
                # Prior: softmax over neg-normalized cost
                scores = [(t, exp(-(c - minc) / (tau * span))) for t, c in imms]
                Z = sum(w for _, w in scores) or 1.0
                self.priors = {t: (w / Z) for t, w in scores}
                self.action_order = [t for t, _ in imms]  # low immediate cost first
            else:
                self.priors = {}
                self.action_order = remaining

    root = Node(tuple(), frozenset(range(n)))

    def progressive_widening_limit(N):
        return max(1, int(c_pw * (N ** alpha_pw)))

    def puct_select(node):
        # Select child maximizing Q + U
        best = None
        best_score = -1e18
        sqrtN = sqrt(max(1, node.N))
        for t, child in node.children.items():
            prior = node.priors.get(t, 1e-6)
            Q = (child.Wsum / max(1, child.N))  # average reward
            U = cpuct * prior * (sqrtN / (1 + child.N))
            score = Q + U
            if score > best_score:
                best_score = score
                best = (t, child)
        return best

    # Main MCTS loop
    start_time = time.time()
    time_budget_s = 1.8 if n >= 90 else 1.2  # modest time guard

    for sim in range(sims_budget):
        if time.time() - start_time > time_budget_s:
            break

        node = root
        path = [node]
        actions_taken = []

        # Selection + progressive widening + pruning
        while True:
            if not node.rem:
                break

            # Incumbent prune at node
            if best_global_cost < float('inf') and node.partial_cost >= best_global_cost:
                break

            # Progressive widening
            pw_lim = progressive_widening_limit(node.N)
            expanded = False
            if len(node.children) < min(pw_lim, len(node.action_order)):
                # Try to expand the next best unexpanded action
                for t in node.action_order:
                    if t in node.children:
                        continue
                    # Prune child by incumbent and dominance
                    new_seq = list(node.seq) + [t]
                    new_rem = node.rem - {t}
                    new_cost = eval_seq(new_seq)

                    # Prefix dominance signature
                    sig = (new_rem, tuple(new_seq[-3:]) if len(new_seq) >= 3 else tuple(new_seq))
                    prev = dom.get(sig)
                    if (prev is not None) and (new_cost >= prev):
                        # dominated
                        continue
                    dom[sig] = new_cost

                    # Incumbent prune
                    if best_global_cost < float('inf') and new_cost >= best_global_cost:
                        continue

                    child = Node(tuple(new_seq), new_rem)
                    node.children[t] = child
                    node = child
                    path.append(node)
                    actions_taken.append(t)
                    expanded = True
                    break

            if expanded:
                break  # go to rollout from first expanded node

            # If cannot expand, select an existing child via PUCT
            if node.children:
                tsel, child = puct_select(node)
                node = child
                path.append(node)
                actions_taken.append(tsel)
            else:
                # Dead-end due to pruning; stop and treat as terminal for backprop
                break

        # Rollout from current node (or complete if terminal)
        if not node.rem:
            full_seq = list(node.seq)
            full_cost = eval_seq(full_seq)
        else:
            full_seq, full_cost = greedy_rollout(list(node.seq), list(node.rem))

        # Update best incumbent
        if full_cost < best_global_cost:
            best_global_cost = full_cost
            best_global_seq = full_seq

        # Reward: improvement over baseline (maximize)
        reward = baseline_cost - full_cost

        # Backpropagate
        for nd in path:
            nd.N += 1
            nd.Wsum += reward

    # Derive sequence from tree (visits path), then complete greedy if needed
    seq_path = []
    node = root
    while node and node.children:
        # choose child with max visits
        best_t = None
        best_N = -1
        for t, ch in node.children.items():
            if ch.N > best_N:
                best_N = ch.N
                best_t = t
        if best_t is None:
            break
        seq_path.append(best_t)
        node = node.children[best_t]

    # Ensure full sequence
    remaining = [t for t in range(n) if t not in seq_path]
    final_seq, final_cost = greedy_rollout(seq_path, remaining)

    # If MCTS incumbent is better, take it
    if best_global_cost < final_cost:
        final_seq, final_cost = best_global_seq, best_global_cost

    # -------------------- ΔW-gated LNS Refinement --------------------
    def adj_penalty(a, b):
        return max(0, W[a][b])  # positive means a->b is disfavored

    def worst_adjacencies(seq, topm=6):
        lst = []
        for i in range(len(seq) - 1):
            a, b = seq[i], seq[i + 1]
            pen = adj_penalty(a, b)
            if pen > 0:
                lst.append((pen, i))
        lst.sort(reverse=True)
        return [i for _, i in lst[:topm]]

    def surrogate_delta_for_swap(seq, i, j):
        # Estimate change using adjacency margins around i and j boundaries
        nloc = len(seq)
        def neigh_pairs(idx, val=None):
            pairs = []
            x = seq[idx] if val is None else val
            # left neighbor
            if idx - 1 >= 0:
                l = seq[idx - 1]
                pairs.append((l, x))
            # right neighbor
            if idx + 1 < nloc:
                r = seq[idx + 1]
                pairs.append((x, r))
            return pairs

        def pen_pairs(pairs):
            return sum(adj_penalty(a, b) for a, b in pairs)

        base = pen_pairs(neigh_pairs(i)) + pen_pairs(neigh_pairs(j))
        # after swap: place seq[j] at i and seq[i] at j
        ai, aj = seq[i], seq[j]
        new_i_pairs = []
        if i - 1 >= 0:
            new_i_pairs.append((seq[i - 1], aj))
        if i + 1 < nloc:
            # be careful if neighbor is j; we approximate with current order
            right = aj if i + 1 == j else seq[i + 1]
            new_i_pairs.append((aj, right))
        new_j_pairs = []
        if j - 1 >= 0:
            left = ai if j - 1 == i else seq[j - 1]
            new_j_pairs.append((left, ai))
        if j + 1 < nloc:
            new_j_pairs.append((ai, seq[j + 1]))
        newp = pen_pairs(new_i_pairs) + pen_pairs(new_j_pairs)
        return base - newp  # positive => likely improvement

    def lns_refine(seq, base_cost, rounds=2, eval_budget=700):
        best_seq = seq[:]
        best_cost = base_cost
        nloc = len(best_seq)

        for rd in range(rounds):
            budget_left = eval_budget

            # 1) Block-swap around worst adjacencies
            viols = worst_adjacencies(best_seq, topm=4)
            proposals = []
            for v_idx in viols[:2]:
                # choose partner boundary far from v_idx among next worst
                partners = [p for p in viols if abs(p - v_idx) > 5]
                if not partners:
                    continue
                p_idx = partners[0]
                # block sizes 3..6
                for sz in range(3, min(6, max(3, nloc // 10)) + 1):
                    i_start = max(0, v_idx - sz // 2)
                    i_end = min(nloc, i_start + sz)
                    if i_end - i_start < 3:
                        continue
                    j_start = max(0, p_idx - sz // 2)
                    j_end = min(nloc, j_start + sz)
                    if j_end - j_start < 3:
                        continue
                    if not (i_end <= j_start or j_end <= i_start):
                        continue  # avoid overlap
                    # surrogate improvement: reduce penalties around edges
                    # approximate by sum of boundary penalties before/after swap (cheap)
                    proposals.append(("swap", (i_start, i_end, j_start, j_end)))
            # Deduplicate
            seen_sw = set()
            filtered = []
            for tag, p in proposals:
                if p in seen_sw:
                    continue
                seen_sw.add(p)
                filtered.append((tag, p))
            # Rank proposals by simple ΔW heuristic: evaluate a cheap boundary-based score
            ranked = []
            for tag, (a0, a1, b0, b1) in filtered:
                # boundaries (a0-1,a0), (a1-1,a1), (b0-1,b0), (b1-1,b1)
                def pen_at_boundary(idx_left, idx_right):
                    if 0 <= idx_left < nloc and 0 <= idx_right < nloc and idx_left < idx_right:
                        return adj_penalty(best_seq[idx_left], best_seq[idx_right])
                    return 0
                base_pen = pen_at_boundary(a0 - 1, a0) + pen_at_boundary(a1 - 1, a1) + pen_at_boundary(b0 - 1, b0) + pen_at_boundary(b1 - 1, b1)
                # optimistic improvement score
                score = base_pen
                ranked.append((score, (a0, a1, b0, b1)))
            ranked.sort(reverse=True)

            # Evaluate top-k proposals
            take = min(len(ranked), max(6, eval_budget // 40))
            for _, (a0, a1, b0, b1) in ranked[:take]:
                if budget_left <= 0:
                    break
                blockA = best_seq[a0:a1]
                blockB = best_seq[b0:b1]
                cand = best_seq[:a0] + blockB + best_seq[a1:b0] + blockA + best_seq[b1:]
                c = eval_seq(cand)
                budget_left -= 1
                if c < best_cost:
                    best_seq, best_cost = cand, c

            # 2) Block reinsert of a hot block
            viols = worst_adjacencies(best_seq, topm=4)
            if viols:
                center = viols[0] + 1
            else:
                center = nloc // 2
            sz = max(4, min(6, nloc // 12))
            start = max(0, min(center - sz // 2, nloc - sz))
            block = best_seq[start:start + sz]
            base = best_seq[:start] + best_seq[start + sz:]
            # predict top-3 insertion positions by ΔW surrogate: try ends + few around worst adjacencies in base
            base_n = len(base)
            candidates_pos = {0, base_n}
            worst_in_base = worst_adjacencies(base, topm=3)
            for w in worst_in_base:
                candidates_pos.add(w)
                candidates_pos.add(min(base_n, w + 1))
            while len(candidates_pos) < 8:
                candidates_pos.add(rng.randint(0, base_n))
            cand_scores = []
            for pos in candidates_pos:
                # evaluate surrogate score: boundaries after inserting
                left_pen = adj_penalty(base[pos - 1], block[0]) if pos - 1 >= 0 else 0
                right_pen = adj_penalty(block[-1], base[pos]) if pos < base_n else 0
                score = -(left_pen + right_pen)
                cand_scores.append((score, pos))
            cand_scores.sort()
            # Evaluate top few
            for _, pos in cand_scores[:min(6, len(cand_scores))]:
                if budget_left <= 0:
                    break
                cand = base[:pos] + block + base[pos:]
                c = eval_seq(cand)
                budget_left -= 1
                if c < best_cost:
                    best_seq, best_cost = cand, c
                    # update base and related indices for next moves
                    base = cand[:]
                    base_n = len(base)

            # 3) A small set of non-adjacent swaps chosen by ΔW improvement
            trials = min(60, max(30, nloc // 2))
            # pre-rank by surrogate delta around random pairs biased by distance
            pairs = []
            for _ in range(trials):
                i = rng.randint(0, nloc - 1)
                j = rng.randint(0, nloc - 1)
                if i == j or abs(i - j) <= 1:
                    continue
                if i > j:
                    i, j = j, i
                delta = surrogate_delta_for_swap(best_seq, i, j)
                pairs.append((delta, i, j))
            pairs.sort(reverse=True)
            take = min(len(pairs), 20)
            for k in range(take):
                if budget_left <= 0:
                    break
                _, i, j = pairs[k]
                cand = best_seq[:]
                cand[i], cand[j] = cand[j], cand[i]
                c = eval_seq(cand)
                budget_left -= 1
                if c < best_cost:
                    best_seq, best_cost = cand, c

        # Final light adjacent pass
        improved = True
        while improved:
            improved = False
            for i in range(len(best_seq) - 1):
                cand = best_seq[:]
                cand[i], cand[i + 1] = cand[i + 1], cand[i]
                c = eval_seq(cand)
                if c < best_cost:
                    best_seq, best_cost = cand, c
                    improved = True
        return best_seq, best_cost

    final_seq, final_cost = lns_refine(final_seq, final_cost, rounds=2, eval_budget=600)

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
    # Slightly larger budget for the richest workload
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