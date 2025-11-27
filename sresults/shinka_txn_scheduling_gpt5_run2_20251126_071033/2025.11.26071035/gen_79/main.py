# EVOLVE-BLOCK-START
"""Transaction scheduling algorithm for optimizing makespan across multiple workloads
Memetic GA with order-based crossover + bounded BnB endgame + light VND.
"""

import time
import random
import sys
import os
from collections import OrderedDict

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
    Memetic GA with order-based crossover + bounded endgame BnB + light VND.

    Args:
        workload: Workload object containing transaction data
        num_seqs: Search effort parameter (controls population size, generations)

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    n = workload.num_txns
    all_txns = list(range(n))

    # -------------------- Adaptive parameters --------------------
    small = n <= 50
    med = 50 < n <= 90
    large = n > 90

    # GA population and generations
    base_pop = 14 if small else (18 if med else 22)
    pop_size = max(12, min(28, base_pop + max(0, num_seqs - 6)))
    generations = 18 if small else (20 if med else 22)
    generations = min(28, generations + max(0, num_seqs - 8))
    tournament_k = 3

    # Mutation and LS budgets
    mutation_rate = 0.25 if large else 0.20
    swap_mut_rate = 0.55  # otherwise relocation/scramble
    vnd_rounds = 1 if large else 2
    adj_passes = 1 if large else 2
    two_opt_samples = min(180, max(70, n))
    pair_swap_samples = min(180, max(70, n))

    # Endgame BnB
    tail_K = 8 if med or large else 9
    node_cap = 3500 if med or large else 3000

    # Elite management
    elite_keep = max(3, min(6, pop_size // 3))
    elite_pool_cap = max(6, min(10, 2 + num_seqs))

    # Position sampling for Or-opt(1) checks in VND
    POS_SAMPLE_CAP = None if small else (22 if med else 18)

    # RNG seed for reproducible-but-diverse runs
    random.seed((n * 2654435761 + num_seqs * 11400714819323198485) % (2**64 - 1))

    # -------------------- Cached cost evaluator --------------------
    cost_cache = {}

    def eval_cost(seq):
        key = tuple(seq)
        c = cost_cache.get(key)
        if c is None:
            c = workload.get_opt_seq_cost(list(seq))
            cost_cache[key] = c
        return c

    # -------------------- Helper: stratified positions (for Or-opt(1)) --------------------
    def stratified_positions(m):
        total = m + 1
        if POS_SAMPLE_CAP is None or total <= POS_SAMPLE_CAP:
            return list(range(total))
        k = POS_SAMPLE_CAP
        anchors = set([0, m, m // 2, m // 4, (3 * m) // 4])
        denom = max(1, k - len(anchors))
        for i in range(denom):
            p = round((i + 0.5) * m / denom)
            anchors.add(int(max(0, min(m, p))))
        return sorted(anchors)

    # -------------------- Initialization (diverse seeds) --------------------
    # Best-insertion (deterministic positions) used for a subset of seeds
    def greedy_best_insertion(seed=None):
        rem = list(all_txns)
        seq = []
        if seed is None:
            # pick best starter among a small candidate set
            k = min(10, n)
            cand = random.sample(rem, k) if k < n else rem
            starter = min(cand, key=lambda t: eval_cost([t]))
        else:
            starter = seed
        seq.append(starter)
        rem.remove(starter)
        # strong second by trying 0/1
        if rem:
            k2 = min(8, len(rem))
            trials = []
            for t in random.sample(rem, k2):
                for p in [0, 1]:
                    c = eval_cost(seq[:p] + [t] + seq[p:])
                    trials.append((c, t, p))
            if trials:
                trials.sort(key=lambda x: x[0])
                _, t2, p2 = random.choice(trials[:min(3, len(trials))])
                seq.insert(p2, t2)
                rem.remove(t2)
        while rem:
            best_t, best_c, best_p = None, float('inf'), 0
            # sample txns to speed-up on large instances
            sample = rem if len(rem) <= 14 else random.sample(rem, 14)
            for t in sample:
                m = len(seq)
                positions = stratified_positions(m)
                for p in positions:
                    c = eval_cost(seq[:p] + [t] + seq[p:])
                    if c < best_c:
                        best_c, best_t, best_p = c, t, p
            if best_t is None:
                best_t = rem.pop()
                seq.append(best_t)
            else:
                seq.insert(best_p, best_t)
                rem.remove(best_t)
        return seq

    def random_seed():
        s = list(all_txns)
        random.shuffle(s)
        return s

    # -------------------- Order-based crossovers --------------------
    def ox_crossover(p1, p2):
        """Order Crossover (OX): copy segment from p1, fill with p2 order."""
        L = len(p1)
        a, b = sorted(random.sample(range(L), 2))
        child = [None] * L
        in_seg = set(p1[a:b + 1])
        # copy segment
        for i in range(a, b + 1):
            child[i] = p1[i]
        # fill with p2 order
        idx = (b + 1) % L
        for x in p2:
            if x in in_seg:
                continue
            while child[idx] is not None:
                idx = (idx + 1) % L
            child[idx] = x
        return child

    def pos_crossover(p1, p2):
        """Position-based crossover: keep random positions from p1, fill rest by p2 order."""
        L = len(p1)
        k = max(1, L // (7 if L > 70 else 5))
        keep_idx = set(random.sample(range(L), k))
        child = [None] * L
        taken = set()
        for i in keep_idx:
            child[i] = p1[i]
            taken.add(p1[i])
        fill = [x for x in p2 if x not in taken]
        it = iter(fill)
        for i in range(L):
            if child[i] is None:
                child[i] = next(it)
        return child

    def crossover(p1, p2):
        return ox_crossover(p1, p2) if random.random() < 0.6 else pos_crossover(p1, p2)

    # -------------------- Mutations --------------------
    def mutate(seq):
        L = len(seq)
        if L <= 2:
            return seq
        s = list(seq)
        if random.random() < swap_mut_rate:
            # Random pair swap
            i, j = random.sample(range(L), 2)
            s[i], s[j] = s[j], s[i]
        else:
            r = random.random()
            if r < 0.5:
                # Relocation of single item
                i, j = random.sample(range(L), 2)
                t = s.pop(i)
                if j > i:
                    j -= 1
                s.insert(j, t)
            else:
                # Scramble a small segment
                i, j = sorted(random.sample(range(L), 2))
                if j - i > 1:
                    mid = s[i:j]
                    random.shuffle(mid)
                    s[i:j] = mid
        return s

    # -------------------- Local search (light VND) --------------------
    def adjacent_swaps_pass(seq, start_cost, max_passes=adj_passes):
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

    def or_opt_pass(seq, start_cost, k=1):
        best_seq = list(seq)
        best_cost = start_cost
        L = len(best_seq)
        if L <= k:
            return best_seq, best_cost
        improved = True
        while improved:
            improved = False
            i = 0
            while i <= len(best_seq) - k:
                block = best_seq[i:i + k]
                base = best_seq[:i] + best_seq[i + k:]
                positions = stratified_positions(len(base))
                move_best = (best_cost, None)
                for p in positions:
                    if k == 1 and p == i:
                        continue
                    cand = base[:]
                    cand[p:p] = block
                    c = eval_cost(cand)
                    if c < move_best[0]:
                        move_best = (c, p)
                if move_best[1] is not None and move_best[0] + 1e-12 < best_cost:
                    p = move_best[1]
                    new_seq = base[:]
                    new_seq[p:p] = block
                    best_seq, best_cost = new_seq, move_best[0]
                    improved = True
                    i = 0  # restart scan after improvement
                else:
                    i += 1
        return best_seq, best_cost

    def sampled_pair_swaps(seq, start_cost, tries=pair_swap_samples):
        best_seq = list(seq)
        best_cost = start_cost
        L = len(best_seq)
        if L <= 3:
            return best_seq, best_cost
        best_delta = 0.0
        best_move = None
        attempts = min(tries, max(60, L))
        for _ in range(attempts):
            i = random.randint(0, L - 1)
            j = random.randint(0, L - 1)
            if i == j or abs(i - j) <= 1:
                continue
            cand = best_seq[:]
            cand[i], cand[j] = cand[j], cand[i]
            c = eval_cost(cand)
            delta = start_cost - c
            if delta > best_delta:
                best_delta = delta
                best_move = (i, j, c)
        if best_move is not None:
            i, j, c = best_move
            cand = best_seq[:]
            cand[i], cand[j] = cand[j], cand[i]
            best_seq, best_cost = cand, c
        return best_seq, best_cost

    def sampled_two_opt_reversal(seq, start_cost, tries=two_opt_samples):
        best_seq = list(seq)
        best_cost = start_cost
        L = len(best_seq)
        if L <= 5:
            return best_seq, best_cost
        best_delta = 0.0
        best_move = None
        attempts = min(tries, max(60, L))
        for _ in range(attempts):
            i = random.randint(0, L - 3)
            j = random.randint(i + 2, min(L - 1, i + 12))
            cand = best_seq[:]
            cand[i:j + 1] = reversed(cand[i:j + 1])
            c = eval_cost(cand)
            delta = start_cost - c
            if delta > best_delta:
                best_delta = delta
                best_move = (i, j, c)
        if best_move is not None:
            i, j, c = best_move
            cand = best_seq[:]
            cand[i:j + 1] = reversed(cand[i:j + 1])
            best_seq, best_cost = cand, c
        return best_seq, best_cost

    def vnd_refine(seq):
        c0 = eval_cost(seq)
        s, c = or_opt_pass(seq, c0, k=2)
        s, c = or_opt_pass(s, c, k=1)
        s, c = adjacent_swaps_pass(s, c, max_passes=adj_passes)
        s, c = sampled_pair_swaps(s, c, tries=pair_swap_samples)
        s, c = sampled_two_opt_reversal(s, c, tries=two_opt_samples)
        # final quick Or-opt(1)
        s, c = or_opt_pass(s, c, k=1)
        return s, c

    # -------------------- Endgame BnB for last K transactions --------------------
    def bnb_optimize_tail(seq, K=tail_K, nodes_limit=node_cap):
        if K <= 0 or K >= len(seq):
            return seq, eval_cost(seq)
        p = len(seq) - K
        fixed = seq[:p]
        tail_set = seq[p:]
        rem = list(tail_set)
        best_c = float('inf')
        best_tail = tail_set[:]
        base_cost = eval_cost(fixed)

        # Transposition table: maps (frozenset(rem), last3) -> best known cost bound
        tt = {}
        nodes = [0]

        last3 = tuple(fixed[-3:]) if len(fixed) >= 3 else tuple(fixed)

        def dfs(prefix, remaining, last_key):
            nonlocal best_c, best_tail
            if nodes[0] >= nodes_limit:
                return
            if not remaining:
                c = eval_cost(prefix)
                if c < best_c:
                    best_c = c
                    best_tail = prefix[p:]
                return
            # pruning by transposition
            key = (frozenset(remaining), last_key)
            cur_cost = eval_cost(prefix)
            prev_best = tt.get(key)
            if prev_best is not None and cur_cost >= prev_best - 1e-12:
                return
            tt[key] = cur_cost if prev_best is None else min(prev_best, cur_cost)

            # bound by current best
            if cur_cost >= best_c - 1e-12:
                return

            # order remaining by greedy next best insertion cost into current prefix
            scored = []
            for t in remaining:
                c_next = eval_cost(prefix + [t])
                scored.append((c_next, t))
            scored.sort(key=lambda x: x[0])

            for c_next, t in scored:
                if nodes[0] >= nodes_limit:
                    break
                if c_next >= best_c - 1e-12:
                    # further insertions will hardly improve
                    continue
                nodes[0] += 1
                new_prefix = prefix + [t]
                new_remaining = [x for x in remaining if x != t]
                new_last = tuple(new_prefix[-3:]) if len(new_prefix) >= 3 else tuple(new_prefix)
                dfs(new_prefix, new_remaining, new_last)

        dfs(fixed, rem, last3)
        new_seq = fixed + best_tail
        return new_seq, best_c

    # -------------------- Elite management and consensus rank --------------------
    elites = []  # list of (cost, seq)

    def add_elite(cost, seq):
        nonlocal elites
        if not seq or len(seq) != n:
            return
        key = tuple(seq)
        for i, (c, s) in enumerate(elites):
            if tuple(s) == key:
                if cost < c:
                    elites[i] = (cost, list(seq))
                break
        else:
            elites.append((cost, list(seq)))
        elites.sort(key=lambda x: x[0])
        if len(elites) > elite_pool_cap:
            elites = elites[:elite_pool_cap]

    def consensus_parent():
        if not elites:
            return None
        # Borda averaging of positions to form a consensus permutation
        pos_sum = {t: 0.0 for t in all_txns}
        for rank_weight, (c, s) in enumerate(elites, start=1):
            # better elites get slightly more weight
            w = (len(elites) + 1 - rank_weight)
            for idx, t in enumerate(s):
                pos_sum[t] += w * idx
        ranked = sorted(all_txns, key=lambda t: pos_sum[t])
        return ranked

    # -------------------- GA lifecycle --------------------
    population = []

    # Initialize: half greedy insertions, half random
    greedy_count = max(4, pop_size // 2)
    for _ in range(greedy_count):
        s = greedy_best_insertion()
        c = eval_cost(s)
        population.append((c, s))
        add_elite(c, s)
    while len(population) < pop_size:
        s = random_seed()
        c = eval_cost(s)
        population.append((c, s))
        add_elite(c, s)

    population.sort(key=lambda x: x[0])
    best_cost, best_seq = population[0]

    def tournament_select(pop, k=tournament_k):
        cand = random.sample(pop, k)
        cand.sort(key=lambda x: x[0])
        return cand[0][1]

    for gen in range(generations):
        new_population = []

        # Elitism: carry over top elites
        keep_elites = min(elite_keep, len(population))
        for i in range(keep_elites):
            new_population.append(population[i])

        # Inject a consensus parent occasionally
        consensus = consensus_parent()
        if consensus is not None:
            c = eval_cost(consensus)
            new_population.append((c, consensus))
            add_elite(c, consensus)

        # Produce offspring
        while len(new_population) < pop_size:
            p1 = tournament_select(population)
            # diversify parent2 selection
            if consensus is not None and random.random() < 0.25:
                p2 = consensus
            else:
                p2 = tournament_select(population)
            child = crossover(p1, p2)
            if random.random() < mutation_rate:
                child = mutate(child)
            # Local search refine
            child, c_child = vnd_refine(child)

            # BnB endgame refine on last-K
            if tail_K > 0:
                child, c_child = bnb_optimize_tail(child, K=min(tail_K, n // 2), nodes_limit=node_cap)

            new_population.append((c_child, child))
            add_elite(c_child, child)

        # Select next generation
        new_population.sort(key=lambda x: x[0])

        # Maintain uniqueness by suffix-2 signature to preserve diversity
        seen_sfx = set()
        filtered = []
        for c, s in new_population:
            sig = tuple(s[-2:]) if len(s) >= 2 else tuple(s)
            if sig in seen_sfx:
                continue
            seen_sfx.add(sig)
            filtered.append((c, s))
            if len(filtered) >= pop_size:
                break

        population = filtered
        if population[0][0] < best_cost:
            best_cost, best_seq = population[0]

        # Light stagnation escape: shuffle and refine a random elite
        if gen % max(4, (8 if large else 6)) == 0 and elites:
            _, es = random.choice(elites[:min(3, len(elites))])
            s = list(es)
            # apply a few random relocations
            tries = min(10, max(6, n // 25))
            for _ in range(tries):
                i, j = random.randint(0, n - 1), random.randint(0, n - 1)
                if i == j:
                    continue
                t = s.pop(i)
                if j > i:
                    j -= 1
                s.insert(j, t)
            s, c = vnd_refine(s)
            if tail_K > 0:
                s, c = bnb_optimize_tail(s, K=min(tail_K, n // 2), nodes_limit=node_cap)
            population.append((c, s))
            add_elite(c, s)
            population.sort(key=lambda x: x[0])
            population = population[:pop_size]
            if population[0][0] < best_cost:
                best_cost, best_seq = population[0]

    # Final polish: one more endgame BnB + light VND
    best_seq, best_cost = bnb_optimize_tail(best_seq, K=min(tail_K + 1, n // 2), nodes_limit=node_cap)
    best_seq, best_cost = vnd_refine(best_seq)

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