# EVOLVE-BLOCK-START
"""Transaction scheduling algorithm for optimizing makespan across multiple workloads"""

import time
import random
import sys
import os
import math

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
    Optimize makespan using a graph-driven tournament ranking and sensitivity-guided LNS.
    1) Build a weighted pairwise preference tournament from exact 2-length costs.
    2) Generate multiple initial schedules via Borda/PageRank scores and randomized QuickSort.
    3) Polish with VND (Or-opt 1..3, 2-opt segment reversals, sampled non-adjacent swaps) with DLB and caching.
    4) Apply sensitivity-guided LNS with pairwise-informed repair and quick polish.

    Args:
        workload: Workload object containing transaction data
        num_seqs: Controls breadth (number of seeds, sampling sizes, LNS rounds)

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    n = workload.num_txns
    all_txns = list(range(n))

    # Global memoization for exact cost of sequences
    cost_cache = {}
    def seq_cost(seq):
        key = tuple(seq)
        if key in cost_cache:
            return cost_cache[key]
        c = workload.get_opt_seq_cost(seq)
        cost_cache[key] = c
        return c

    # Parameters (adaptive to num_seqs and n)
    # Seed portfolio sizes
    quicksort_seeds = max(4, min(8, 2 + num_seqs))
    noise_seeds = max(2, min(4, 1 + num_seqs // 3))
    elite_keep = max(4, min(8, 3 + num_seqs // 2))

    # Local search parameters
    ls_max_rounds = 2
    two_opt_samples = min(220, max(80, n * 2))
    swap_samples = min(300, max(80, n * 3))
    reinsertion_positions_base = 8

    # LNS parameters
    lns_iters = max(3, min(7, 2 + num_seqs // 2))
    destroy_frac_range = (0.10, 0.22)
    sensitivity_sample_pairs = min(40, max(16, n // 2))  # sampled counterpart checks per item
    repair_top_positions = 4  # candidate positions by pairwise signal to verify with exact cost

    # ----------------------------
    # Build weighted pairwise tournament from exact costs
    # margin[i][j] = cost([j,i]) - cost([i,j]); positive => i before j preferred
    # ----------------------------
    margin = [[0.0 for _ in range(n)] for __ in range(n)]
    if n > 0:
        for i in range(n):
            # self margins are zero
            margin[i][i] = 0.0
        for i in range(n):
            for j in range(i + 1, n):
                cij = seq_cost([i, j])
                cji = seq_cost([j, i])
                m = cji - cij
                margin[i][j] = m
                margin[j][i] = -m

    # Borda-like score: sum of margins vs everyone
    borda = [0.0 for _ in range(n)]
    for i in range(n):
        s = 0.0
        mi = margin[i]
        for j in range(n):
            s += mi[j]
        borda[i] = s

    # PageRank-like score over preference graph (edges from i to j with weight=max(margin[i][j],0))
    def pagerank_scores(alpha=0.85, iters=40):
        # Build outgoing weights
        out_w = [0.0] * n
        G = [[] for _ in range(n)]  # (neighbor, weight)
        for i in range(n):
            wsum = 0.0
            for j in range(n):
                if i == j:
                    continue
                w = margin[i][j]
                if w > 0:
                    G[i].append((j, w))
                    wsum += w
            out_w[i] = wsum if wsum > 0 else 0.0
        pr = [1.0 / n for _ in range(n)]
        teleport = (1.0 - alpha) / n if n > 0 else 0.0
        for _ in range(iters):
            new_pr = [teleport for __ in range(n)]
            for i in range(n):
                if out_w[i] <= 0.0:
                    # Distribute uniformly if no outgoing positive preferences
                    share = alpha * pr[i] / n
                    for j in range(n):
                        new_pr[j] += share
                else:
                    for j, w in G[i]:
                        new_pr[j] += alpha * pr[i] * (w / out_w[i])
            pr = new_pr
        return pr

    pr_scores = pagerank_scores() if n > 0 else []

    # Randomized QuickSort guided by margins (Ailon et al. noisy sorting)
    def tour_quicksort(items):
        if len(items) <= 1:
            return items
        pivot = random.choice(items)
        left = []
        right = []
        # tie-breaking weights: compare margin, if tie, use borda diff or random
        for x in items:
            if x == pivot:
                continue
            m = margin[x][pivot]
            if m > 1e-9 or (abs(m) <= 1e-9 and (borda[x] > borda[pivot] or (abs(borda[x] - borda[pivot]) <= 1e-9 and random.random() < 0.5))):
                left.append(x)
            else:
                right.append(x)
        return tour_quicksort(left) + [pivot] + tour_quicksort(right)

    # Seed generators
    def seed_borda():
        return sorted(all_txns, key=lambda t: (-borda[t], t))

    def seed_pagerank():
        if not pr_scores:
            return seed_borda()
        idxs = list(range(n))
        return sorted(idxs, key=lambda t: (-pr_scores[t], -borda[t], t))

    def seed_quicksort():
        items = all_txns[:]
        return tour_quicksort(items)

    def slight_noise(seq, swaps= max(3, n // 20)):
        s = seq[:]
        for _ in range(swaps):
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i != j:
                s[i], s[j] = s[j], s[i]
        return s

    # ----------------------------
    # Local search (VND) with DLB and best-two reinsertion caching
    # ----------------------------
    def local_refine(seq, max_rounds=ls_max_rounds):
        best_seq = seq[:]
        best_cost = seq_cost(best_seq)

        # Reinsertion best-two cache per episode: key=(tuple(base), item)
        best_two_cache = {}

        def eval_best_two_positions(base_seq, item, positions):
            key = (tuple(base_seq), item)
            if key in best_two_cache:
                return best_two_cache[key]
            best_c = float('inf')
            second_c = float('inf')
            best_p = None
            for p in positions:
                cand = base_seq[:p] + [item] + base_seq[p:]
                c = seq_cost(cand)
                if c < best_c:
                    second_c = best_c
                    best_c = c
                    best_p = p
                elif c < second_c:
                    second_c = c
            best_two_cache[key] = (best_c, best_p, second_c)
            return best_two_cache[key]

        # position sampling helper
        def pos_samples(L, focus=None, k=reinsertion_positions_base):
            if L <= 18:
                return list(range(L + 1))
            anchors = {0, L, L // 2}
            if focus is not None:
                for d in range(-4, 5):
                    p = focus + d
                    if 0 <= p <= L:
                        anchors.add(p)
            # deterministic spread
            for frac in [1/4, 3/4]:
                anchors.add(int(L * frac))
            # random extras
            for _ in range(min(k, L + 1)):
                anchors.add(random.randint(0, L))
            return sorted(anchors)

        # Don't-look bits for indices (helps skip stable areas)
        dlb = [False] * n

        rounds = 0
        improved_global = True
        while improved_global and rounds < max_rounds:
            improved_global = False
            rounds += 1

            # 1) Reinsertion (Or-opt 1..3), best-improvement passes with DLB
            for blk in (1, 2, 3):
                improved = True
                while improved:
                    improved = False
                    i = 0
                    while i <= len(best_seq) - blk:
                        if dlb[i]:
                            i += 1
                            continue
                        block = best_seq[i:i + blk]
                        base = best_seq[:i] + best_seq[i + blk:]
                        positions = pos_samples(len(base), focus=i, k=reinsertion_positions_base + 2)
                        # avoid no-op at original place
                        positions = [p for p in positions if p != i]
                        best_c, best_p, _ = eval_best_two_positions(base, None if blk != 1 else block[0], positions) if False else (None, None, None)
                        # Evaluate explicitly for block moves
                        best_found_cost = float('inf')
                        best_found_seq = None
                        for p in positions:
                            cand = base[:p] + block + base[p:]
                            if cand == best_seq:
                                continue
                            c = seq_cost(cand)
                            if c < best_found_cost:
                                best_found_cost = c
                                best_found_seq = cand
                        if best_found_seq is not None and best_found_cost < best_cost:
                            best_seq = best_found_seq
                            best_cost = best_found_cost
                            improved = True
                            improved_global = True
                            # reset DLB near changed region
                            for kidx in range(max(0, i - 2), min(len(dlb), i + blk + 2)):
                                if kidx < len(dlb):
                                    dlb[kidx] = False
                            # restart scan
                            i = 0
                            continue
                        else:
                            dlb[i] = True
                            i += 1
                if improved_global:
                    # restart outer if improvement found in this neighborhood
                    continue

            # 2) Adjacent swap steepest-like with early exit
            i = 0
            while i < len(best_seq) - 1:
                cand = best_seq[:]
                cand[i], cand[i + 1] = cand[i + 1], cand[i]
                c = seq_cost(cand)
                if c < best_cost:
                    best_seq = cand
                    best_cost = c
                    improved_global = True
                    # reset DLB around swap region
                    for kidx in range(max(0, i - 2), min(len(dlb), i + 3)):
                        dlb[kidx] = False
                    i = max(0, i - 2)
                    continue
                i += 1
            if improved_global:
                continue

            # 3) Sampled non-adjacent swaps (first-improvement)
            for _ in range(swap_samples):
                i = random.randint(0, n - 2)
                j = random.randint(i + 1, n - 1)
                if j == i + 1:
                    continue
                cand = best_seq[:]
                cand[i], cand[j] = cand[j], cand[i]
                c = seq_cost(cand)
                if c < best_cost:
                    best_seq = cand
                    best_cost = c
                    improved_global = True
                    break
            if improved_global:
                continue

            # 4) True 2-opt: reverse a random segment (first-improvement)
            for _ in range(two_opt_samples):
                i = random.randint(0, n - 2)
                j = random.randint(i + 2, n - 1)
                cand = best_seq[:i] + best_seq[i:j + 1][::-1] + best_seq[j + 1:]
                c = seq_cost(cand)
                if c < best_cost:
                    best_seq = cand
                    best_cost = c
                    improved_global = True
                    break

        return best_cost, best_seq

    # ----------------------------
    # Sensitivity-guided LNS
    # ----------------------------
    def pairwise_local_sensitivity(seq, sample_pairs=sensitivity_sample_pairs):
        # Estimate how "unstable" each position is, based on local pairwise violations and sampled negatives
        L = len(seq)
        sens = [0.0] * L
        if L == 0:
            return sens
        # Pre-sample opponents for each item
        for i in range(L):
            item = seq[i]
            val = 0.0
            if i > 0:
                left = seq[i - 1]
                # if margin[left][item] < 0: left ideally should be after item -> violation
                val += max(0.0, -margin[left][item])
            if i < L - 1:
                right = seq[i + 1]
                # if margin[item][right] < 0: item ideally should be after right -> violation
                val += max(0.0, -margin[item][right])
            # Few sampled counterparts across the sequence to capture broader conflicts
            if L > 3:
                sample_idx = random.sample(range(L), min(sample_pairs, L))
                for j in sample_idx:
                    if j == i:
                        continue
                    other = seq[j]
                    # penalty if current relative order contradicts margin
                    if i < j:
                        if margin[item][other] < 0:
                            val += -margin[item][other] * 0.05
                    else:
                        if margin[other][item] < 0:
                            val += -margin[other][item] * 0.05
            sens[i] = val
        return sens

    def lns_attempt(seq):
        L = len(seq)
        if L <= 4:
            return seq_cost(seq), seq[:]
        # Compute sensitivities
        sens = pairwise_local_sensitivity(seq)
        # Determine how many to remove
        frac = random.uniform(*destroy_frac_range)
        m = max(6, min(L // 2, int(frac * L)))
        # pick top-m*0.6 sensitive plus random/contiguous for the rest
        k_top = int(0.6 * m)
        idx_sorted = sorted(range(L), key=lambda i: (-sens[i], i))
        to_remove = set(idx_sorted[:k_top])
        # mix with contiguous block or random remainder
        rem_left = m - len(to_remove)
        if rem_left > 0:
            if random.random() < 0.5 and L - m > 0:
                start = random.randint(0, L - rem_left)
                for i in range(start, start + rem_left):
                    to_remove.add(i)
            else:
                while len(to_remove) < m:
                    to_remove.add(random.randint(0, L - 1))
        removed = [seq[i] for i in sorted(to_remove)]
        remaining = [seq[i] for i in range(L) if i not in to_remove]

        # Repair: pairwise-guided candidate positions filtered by exact cost check
        rebuilt = remaining[:]
        for t in removed:
            # compute pairwise local score over positions: lower is better
            pos_scores = []
            for p in range(len(rebuilt) + 1):
                left = rebuilt[p - 1] if p > 0 else None
                right = rebuilt[p] if p < len(rebuilt) else None
                score = 0.0
                if left is not None:
                    # penalty if left should be after t
                    score += max(0.0, -margin[left][t])
                if right is not None:
                    # penalty if t should be after right
                    score += max(0.0, -margin[t][right])
                # very light lookahead around neighbors two away
                if p > 1:
                    left2 = rebuilt[p - 2]
                    score += max(0.0, -margin[left2][t]) * 0.25
                if p + 1 < len(rebuilt):
                    right2 = rebuilt[p + 1]
                    score += max(0.0, -margin[t][right2]) * 0.25
                pos_scores.append((score, p))
            # choose top-k by pairwise score and verify by exact cost
            pos_scores.sort(key=lambda x: (x[0], x[1]))
            candidates = [p for _, p in pos_scores[:repair_top_positions]]
            best_c = float('inf')
            best_pos = candidates[0] if candidates else len(rebuilt)
            for p in candidates:
                cand = rebuilt[:p] + [t] + rebuilt[p:]
                c = seq_cost(cand)
                if c < best_c:
                    best_c = c
                    best_pos = p
            rebuilt = rebuilt[:best_pos] + [t] + rebuilt[best_pos:]

        # Quick polish
        c_pol, s_pol = local_refine(rebuilt, max_rounds=1)
        return c_pol, s_pol

    # ----------------------------
    # Seed portfolio construction and evaluation
    # ----------------------------
    seeds = []

    sb = seed_borda()
    seeds.append(sb)
    seeds.append(list(reversed(sb)))

    sp = seed_pagerank()
    seeds.append(sp)

    for _ in range(quicksort_seeds):
        seeds.append(seed_quicksort())

    for _ in range(noise_seeds):
        seeds.append(slight_noise(sb))

    # Deduplicate seeds
    seed_unique = []
    seen = set()
    for s in seeds:
        t = tuple(s)
        if t in seen:
            continue
        seen.add(t)
        seed_unique.append(s)

    # Evaluate seeds and polish
    elite = []  # list of (cost, seq)
    def add_elite(c, s):
        elite.append((c, s))
        elite.sort(key=lambda x: x[0])
        # keep uniqueness by suffix-3 signature to preserve diversity
        uniq = []
        seen_sig = set()
        for c1, s1 in elite:
            sig = tuple(s1[-3:]) if len(s1) >= 3 else tuple(s1)
            if sig in seen_sig:
                continue
            seen_sig.add(sig)
            uniq.append((c1, s1))
            if len(uniq) >= elite_keep:
                break
        elite[:] = uniq

    for s in seed_unique:
        c0 = seq_cost(s)
        c1, s1 = local_refine(s)
        add_elite(c1, s1)

    if not elite:
        base = all_txns[:]
        random.shuffle(base)
        elite = [(seq_cost(base), base)]

    best_cost, best_seq = elite[0]

    # ----------------------------
    # LNS rounds from elites
    # ----------------------------
    for _ in range(lns_iters):
        # try from current best and a random elite
        starts = [best_seq]
        if len(elite) > 1 and random.random() < 0.6:
            starts.append(random.choice(elite[1:])[1])
        for start_seq in starts:
            c2, s2 = lns_attempt(start_seq)
            if c2 < best_cost:
                best_cost, best_seq = c2, s2
                add_elite(c2, s2)

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