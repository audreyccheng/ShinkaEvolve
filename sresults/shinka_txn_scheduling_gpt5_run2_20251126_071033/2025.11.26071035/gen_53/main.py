# EVOLVE-BLOCK-START
"""Transaction scheduling algorithm for optimizing makespan across multiple workloads
Pairwise Rank-Centrality global ordering + Kemeny-guided refinement + light LNS.
"""

import time
import random
import sys
import os
from math import inf

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
    Pairwise-preference global ordering + Kemeny-guided refinements to minimize makespan.

    Args:
        workload: Workload object containing transaction data
        num_seqs: Search effort parameter (scales refinement and LNS intensity)

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    n = workload.num_txns
    txns = list(range(n))

    # Seed for reproducibility with some diversity
    random.seed((n * 7349 + num_seqs * 271 + 17) % (2**32 - 1))

    # Effort scaling
    small = n <= 50
    med = 50 < n <= 90
    large = n > 90

    # Local search budgets
    adj_passes = 2 if (med or large) else 3
    oropt_rounds = 2 if large else 3
    guided_moves = max(15, int(0.8 * n)) if large else max(10, int(1.1 * n))
    lns_rounds = 2 if (med or large) else 3
    lns_remove_frac = 0.10 if large else (0.12 if med else 0.14)
    lns_remove_min = 8 if (med or large) else 6
    lns_remove_max = 18 if (med or large) else 16

    # Restarts (diverse seeds from different global rankers)
    restarts = max(2, min(6, 3 + num_seqs // 3))

    # Cached cost evaluations for any prefix/sequence
    cost_cache = {}
    def eval_cost(seq):
        key = tuple(seq)
        c = cost_cache.get(key)
        if c is None:
            c = workload.get_opt_seq_cost(list(seq))
            cost_cache[key] = c
        return c

    # Build pairwise margin matrix m[i][j] = cost([i,j]) - cost([j,i])
    # Negative m[i][j] means i should precede j (beneficial).
    def build_pairwise_margins():
        m = [[0.0] * n for _ in range(n)]
        # Pre-evaluate singletons for potential cache hits; not strictly necessary
        for i in range(n):
            eval_cost([i])
        for i in range(n):
            for j in range(i + 1, n):
                c_ij = eval_cost([i, j])
                c_ji = eval_cost([j, i])
                margin = c_ij - c_ji
                m[i][j] = margin
                m[j][i] = -margin
        return m

    margins = build_pairwise_margins()

    # Borda-like score: s[i] = -sum_j m[i][j]; more negative margins (wins) => larger score
    def borda_scores(m):
        scores = [0.0] * n
        for i in range(n):
            s = 0.0
            row = m[i]
            for j in range(n):
                if i == j:
                    continue
                s += -row[j]
            scores[i] = s
        return scores

    borda = borda_scores(margins)

    # Copeland wins (unweighted): count of j where m[i][j] < 0
    def copeland_wins(m):
        wins = [0] * n
        for i in range(n):
            cnt = 0
            for j in range(n):
                if i == j: continue
                if m[i][j] < 0:
                    cnt += 1
            wins[i] = cnt
        return wins

    copeland = copeland_wins(margins)

    # Rank Centrality via power iteration; transitions from i to j ~ L[i][j] where L[i][j] = max(0, m[i][j])
    def rank_centrality(m):
        L = [[0.0] * n for _ in range(n)]
        for i in range(n):
            for j in range(n):
                if i == j: continue
                val = m[i][j]
                if val > 0:
                    L[i][j] = val
        # Build row-stochastic P with damping to ensure ergodicity
        damping = 0.15
        P = [[0.0] * n for _ in range(n)]
        for i in range(n):
            row_sum = sum(L[i][j] for j in range(n))
            if row_sum <= 1e-12:
                # If i does not "lose" to anyone, use uniform teleport row
                for j in range(n):
                    P[i][j] = 1.0 / n
            else:
                for j in range(n):
                    P[i][j] = L[i][j] / row_sum
        # Apply damping
        for i in range(n):
            for j in range(n):
                P[i][j] = (1.0 - damping) * P[i][j] + damping * (1.0 / n)
        # Power iteration
        pi = [1.0 / n] * n
        tmp = [0.0] * n
        for _ in range(200):
            for j in range(n):
                tmp[j] = 0.0
            for i in range(n):
                pi_i = pi[i]
                row = P[i]
                for j in range(n):
                    tmp[j] += pi_i * row[j]
            # Normalize
            s = sum(tmp)
            if s > 0:
                for j in range(n):
                    tmp[j] /= s
            # Check convergence
            diff = sum(abs(tmp[j] - pi[j]) for j in range(n))
            pi, tmp = tmp, pi
            if diff < 1e-10:
                break
        return pi

    pi = rank_centrality(margins)

    # Multiple initial orders
    def order_from_scores(scores, reverse=True):
        idx = list(range(n))
        idx.sort(key=lambda i: scores[i], reverse=reverse)
        return idx

    def greedy_tournament_order(m):
        # Build order by repeatedly choosing node with highest net wins weighted by margin magnitude
        remaining = set(range(n))
        order = []
        while remaining:
            best = None
            best_score = -inf
            for i in remaining:
                s = 0.0
                for j in remaining:
                    if i == j: continue
                    # Net win: negative margin is a "win"
                    s += (-m[i][j])
                if s > best_score:
                    best_score = s
                    best = i
            order.append(best)
            remaining.remove(best)
        return order

    order_pi = order_from_scores(pi, reverse=True)
    order_borda = order_from_scores(borda, reverse=True)
    order_copeland = order_from_scores(copeland, reverse=True)
    order_greedy = greedy_tournament_order(margins)

    # Blended order from rank positions of pi and borda
    def blended(pi_scores, b_scores, alpha=0.6):
        # Rank indices
        r_pi = {t: i for i, t in enumerate(order_from_scores(pi_scores, reverse=True))}
        r_bo = {t: i for i, t in enumerate(order_from_scores(b_scores, reverse=True))}
        items = list(range(n))
        items.sort(key=lambda t: alpha * r_pi[t] + (1 - alpha) * r_bo[t])
        return items

    order_blend = blended(pi, borda, alpha=0.65)

    # Initial candidate orders (and a couple of perturbations)
    initial_orders = []
    for o in [order_pi, order_borda, order_copeland, order_greedy, order_blend]:
        initial_orders.append(list(o))
        # small controlled perturbation to diversify
        if n >= 8:
            p = list(o)
            # swap two items with close Borda ranks
            i = random.randint(0, n - 4)
            p[i], p[i + 2] = p[i + 2], p[i]
            initial_orders.append(p)

    # Deduplicate initial orders
    seen_init = set()
    unique_initial_orders = []
    for o in initial_orders:
        key = tuple(o)
        if key not in seen_init:
            seen_init.add(key)
            unique_initial_orders.append(o)

    # Pairwise penalty of an order (Kemeny-like surrogate)
    def pairwise_penalty(order):
        # sum over i<j of max(0, m[a][b]) where a=order[i], b=order[j]
        s = 0.0
        for i in range(n - 1):
            a = order[i]
            row = margins[a]
            for j in range(i + 1, n):
                b = order[j]
                val = row[b]
                if val > 0:
                    s += val
        return s

    # Adjacent swap pass guided by margins, verified by true cost
    def adjacent_swap_refine(seq, passes=adj_passes):
        best_seq = list(seq)
        best_cost = eval_cost(best_seq)
        for _ in range(passes):
            improved = False
            i = 0
            while i < len(best_seq) - 1:
                a, b = best_seq[i], best_seq[i + 1]
                # If pairwise margin suggests swap (a before b is bad), try it
                if margins[a][b] > 0:
                    cand = list(best_seq)
                    cand[i], cand[i + 1] = cand[i + 1], cand[i]
                    c = eval_cost(cand)
                    if c + 1e-12 < best_cost:
                        best_cost = c
                        best_seq = cand
                        improved = True
                        # do not increment i to allow cascading effects
                        continue
                i += 1
            if not improved:
                break
        return best_seq, best_cost

    # Or-opt block relocation (k=1..3)
    def or_opt_block(seq, start_cost, k, pos_cap=None):
        best_seq = list(seq)
        best_cost = start_cost
        L = len(best_seq)
        if L <= k:
            return best_seq, best_cost, False
        changed = False
        i = 0
        while i <= L - k:
            block = best_seq[i:i + k]
            base = best_seq[:i] + best_seq[i + k:]
            m_slots = len(base) + 1
            # Candidate positions: ends + around Borda target of first block item
            positions = set([0, m_slots - 1])
            # guide by Borda target
            target_idx = {t: idx for idx, t in enumerate(order_borda)}
            t0 = block[0]
            tpos = target_idx.get(t0, min(i, m_slots - 1))
            for d in (-3, -2, -1, 0, 1, 2, 3):
                p = min(max(0, tpos + d), m_slots - 1)
                positions.add(p)
            # If allowed, add a few midpoints
            if pos_cap is None or len(positions) < pos_cap:
                positions.update({m_slots // 2})
            move_best_cost = best_cost
            best_pos = None
            for pos in sorted(positions):
                if pos == i:
                    continue
                cand = list(base)
                cand[pos:pos] = block
                c = eval_cost(cand)
                if c < move_best_cost:
                    move_best_cost = c
                    best_pos = pos
            if best_pos is not None and move_best_cost + 1e-12 < best_cost:
                new_seq = list(base)
                new_seq[best_pos:best_pos] = block
                best_seq = new_seq
                best_cost = move_best_cost
                L = len(best_seq)
                changed = True
                i = 0
            else:
                i += 1
        return best_seq, best_cost, changed

    def vnd_refine(seq, start_cost):
        best_seq = list(seq)
        best_cost = start_cost
        rounds = oropt_rounds
        for _ in range(rounds):
            changed = False
            for k in (3, 2, 1):
                s, c, imp = or_opt_block(best_seq, best_cost, k, pos_cap=20 if (med or large) else None)
                if imp and c < best_cost:
                    best_seq, best_cost = s, c
                    changed = True
            s, c = adjacent_swap_refine(best_seq, passes=1)
            if c < best_cost:
                best_seq, best_cost = s, c
                changed = True
            if not changed:
                break
        return best_seq, best_cost

    # Kemeny-guided targeted reinsertions toward rank targets
    def kemeny_guided_moves(seq, start_cost, moves=guided_moves):
        # target ranks by blended (pi+borda) and borda alone
        target_blend = {t: i for i, t in enumerate(order_blend)}
        target_borda = {t: i for i, t in enumerate(order_borda)}

        best_seq = list(seq)
        best_cost = start_cost
        for _ in range(moves):
            # score sensitivity by pairwise penalty contribution and displacement from targets
            L = len(best_seq)
            penalties = []
            pos_of = {t: i for i, t in enumerate(best_seq)}
            for i in range(L):
                t = best_seq[i]
                # penalty contribution: pairs (left,t) and (t,right)
                pen = 0.0
                for j in range(0, i):
                    a = best_seq[j]
                    val = margins[a][t]
                    if val > 0: pen += val
                for j in range(i + 1, L):
                    b = best_seq[j]
                    val = margins[t][b]
                    if val > 0: pen += val
                disp = abs(i - target_blend.get(t, i)) + 0.5 * abs(i - target_borda.get(t, i))
                score = pen + 0.5 * disp
                penalties.append((score, i, t))
            penalties.sort(reverse=True)  # highest first

            improved_any = False
            # Try top-k sensitive items
            top_k = min(5, len(penalties))
            for _, i, t in penalties[:top_k]:
                base = best_seq[:i] + best_seq[i + 1:]
                m_slots = len(base) + 1
                candidate_positions = set()
                tb = target_borda.get(t, i)
                tl = target_blend.get(t, i)
                # sample around both targets
                for d in range(-4, 5):
                    candidate_positions.add(min(max(0, tb + d), m_slots - 1))
                    candidate_positions.add(min(max(0, tl + d), m_slots - 1))
                # also current neighborhood and ends
                candidate_positions.update({0, m_slots - 1, min(i, m_slots - 1), max(0, i - 1), min(m_slots - 1, i + 1)})
                best_local_c = inf
                best_local_p = None
                for pos in sorted(candidate_positions):
                    cand = list(base)
                    cand.insert(pos, t)
                    c = eval_cost(cand)
                    if c < best_local_c:
                        best_local_c = c
                        best_local_p = pos
                if best_local_p is not None and best_local_c + 1e-12 < best_cost:
                    new_seq = list(base)
                    new_seq.insert(best_local_p, t)
                    best_seq = new_seq
                    best_cost = best_local_c
                    improved_any = True
                    break
            if not improved_any:
                break
        return best_seq, best_cost

    # LNS: remove high-penalty items and reinsert near targets
    def lns_ruin_recreate(seq, start_cost, rounds=lns_rounds):
        best_seq = list(seq)
        best_cost = start_cost

        target_blend = {t: i for i, t in enumerate(order_blend)}
        target_borda = {t: i for i, t in enumerate(order_borda)}

        def penalty_contrib(order):
            L = len(order)
            contrib = [0.0] * L
            for i in range(L):
                t = order[i]
                s = 0.0
                for j in range(0, i):
                    a = order[j]
                    val = margins[a][t]
                    if val > 0: s += val
                for j in range(i + 1, L):
                    b = order[j]
                    val = margins[t][b]
                    if val > 0: s += val
                contrib[i] = s
            return contrib

        for _ in range(rounds):
            L = len(best_seq)
            if L < 6:
                break
            # decide removal size
            m = int(max(lns_remove_min, min(lns_remove_max, lns_remove_frac * L)))
            m = min(m, L - 2)

            contrib = penalty_contrib(best_seq)
            idx_sorted = sorted(range(L), key=lambda i: contrib[i], reverse=True)
            remove_idx = set(idx_sorted[:max(2, m // 2)])

            # fill rest with a contiguous chunk to create a hole
            if len(remove_idx) < m:
                chunk_len = max(2, min(m - len(remove_idx), max(2, L // 12)))
                start = random.randint(0, L - chunk_len)
                for j in range(start, start + chunk_len):
                    if len(remove_idx) >= m:
                        break
                    remove_idx.add(j)

            removed = [best_seq[i] for i in sorted(remove_idx)]
            base = [best_seq[i] for i in range(L) if i not in remove_idx]

            # reinsert removed guided by target positions and true cost
            rebuilt = list(base)
            for t in removed:
                m_slots = len(rebuilt) + 1
                candidate_positions = set([0, m_slots - 1, m_slots // 2])
                tb = target_borda.get(t, m_slots // 2)
                tl = target_blend.get(t, m_slots // 2)
                for d in range(-3, 4):
                    candidate_positions.add(min(max(0, tb + d), m_slots - 1))
                    candidate_positions.add(min(max(0, tl + d), m_slots - 1))
                best_c = inf
                best_p = None
                for pos in sorted(candidate_positions):
                    cand = list(rebuilt)
                    cand.insert(pos, t)
                    c = eval_cost(cand)
                    if c < best_c:
                        best_c = c
                        best_p = pos
                if best_p is None:
                    rebuilt.append(t)
                else:
                    rebuilt.insert(best_p, t)

            c_new = eval_cost(rebuilt)
            if c_new + 1e-12 < best_cost:
                best_cost = c_new
                best_seq = rebuilt

        return best_seq, best_cost

    # Compose full refinement pipeline for one initial order
    def refine_pipeline(order):
        c0 = eval_cost(order)
        s1, c1 = adjacent_swap_refine(order, passes=adj_passes)
        s2, c2 = vnd_refine(s1, c1)
        s3, c3 = kemeny_guided_moves(s2, c2, moves=guided_moves)
        # Light LNS
        s4, c4 = lns_ruin_recreate(s3, c3, rounds=lns_rounds)
        # Final quick polish
        s5, c5 = vnd_refine(s4, c4)
        return s5, c5

    # Try multiple seeds and pick best
    best_seq = None
    best_cost = inf

    # Limit number of seeds processed based on restarts
    seeds = unique_initial_orders[:max(3, min(len(unique_initial_orders), 2 + restarts))]
    for seed in seeds:
        s, c = refine_pipeline(seed)
        if c < best_cost:
            best_cost, best_seq = c, s

    # Safety check
    assert best_seq is not None
    assert len(best_seq) == n and len(set(best_seq)) == n

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