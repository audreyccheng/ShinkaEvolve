# EVOLVE-BLOCK-START
"""Transaction scheduling algorithm for optimizing makespan across multiple workloads"""

import time
import random
import sys
import os
from collections import defaultdict, deque

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
    Clustered, deterministic multi-restart beam search + VNS local refinement.
    - Sampled pairwise preference matrix to form conflict clusters and buddy lists
    - Deterministic multi-restart beam search with shared memoization and pruning
    - Greedy completions to set incumbents early
    - VNS with surrogate prefiltering for swaps/relocates/block moves
    """
    N = workload.num_txns

    # Deterministic RNG based on problem size to stabilize quality
    rng = random.Random(1729 + N)

    start_time = time.time()
    # Balanced time budget per workload; scales mildly with N
    base_budget = 0.50
    time_budget_sec = base_budget + min(0.12, 0.001 * max(0, N))

    def time_left():
        return (time.time() - start_time) < time_budget_sec

    # Shared caches across all phases/restarts
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
            # Share inner cache with seq cost
            c = eval_seq_cost(list(prefix_tuple) + [cand])
            ext_cache[key] = c
        return c

    all_txns = list(range(N))

    # Precompute singleton costs (used for seeding and a weak LB)
    singleton_cost = {}
    for t in all_txns:
        if not time_left():
            break
        singleton_cost[t] = eval_seq_cost([t])

    # Sampled pairwise preferences to guide ordering and clustering
    # P[i][j] = cost([i,j]) - cost([j,i]); negative => i before j is better
    P = defaultdict(dict)
    abs_edges = []

    # Build buddy lists: for each i, choose K buddies (low singleton and random)
    K_buddy = min(8, max(5, N // 20))
    singles_sorted = sorted(all_txns, key=lambda t: singleton_cost.get(t, float('inf')))
    # form candidate pools mixing top singletons and randoms
    def buddy_candidates(i):
        pool = []
        # Strong candidates from top singletons
        pool.extend(singles_sorted[:min(N, 2 * K_buddy)])
        # Random diversity
        if N > 2 * K_buddy:
            pool.extend(rng.sample(all_txns, min(N - 2 * K_buddy, 2 * K_buddy)))
        # Unique and remove self
        seen = set()
        out = []
        for x in pool:
            if x == i or x in seen:
                continue
            seen.add(x)
            out.append(x)
        return out

    buddies = {i: [] for i in all_txns}
    # Limit total pairwise probes to control runtime
    max_pair_probes = min(N * 16, 1100)
    probes = 0
    for i in all_txns:
        if not time_left():
            break
        cands = buddy_candidates(i)
        rng.shuffle(cands)
        cands = cands[:K_buddy]
        row = []
        for j in cands:
            if probes >= max_pair_probes or not time_left():
                break
            # Compute P[i][j] and P[j][i]
            cij = eval_seq_cost([i, j])
            cji = eval_seq_cost([j, i])
            p = cij - cji
            P[i][j] = p
            P[j][i] = -p
            abs_edges.append(abs(p))
            # Preference strength to rank buddies: more negative is stronger preference i before j
            row.append((p, j))
            probes += 2  # two evals per pair accounted in eval_seq_cost cache
        row.sort(key=lambda x: x[0])  # negative first
        buddies[i] = [j for _, j in row]

    # Build conflict clusters via connected components on strong preference edges
    def build_clusters():
        if not abs_edges:
            return [all_txns[:]]
        # Threshold at ~70th percentile of |P| to keep only strong edges
        sorted_abs = sorted(abs_edges)
        idx = int(0.70 * (len(sorted_abs) - 1))
        tau = sorted_abs[idx] if sorted_abs else 0
        # Graph adjacency
        adj = defaultdict(list)
        for i, row in P.items():
            for j, val in row.items():
                if abs(val) >= tau and i != j:
                    adj[i].append(j)
                    adj[j].append(i)
        # Extract components with BFS
        seen = set()
        comps = []
        for v in all_txns:
            if v in seen:
                continue
            q = deque([v])
            seen.add(v)
            comp = [v]
            while q:
                u = q.popleft()
                for w in adj.get(u, []):
                    if w not in seen:
                        seen.add(w)
                        q.append(w)
                        comp.append(w)
            comps.append(comp)
        return comps

    clusters = build_clusters()

    # Within a cluster, produce an order using weighted tournament scores from P
    def order_within_cluster(nodes):
        # Score(v) = sum over u in nodes, -P[v][u] (so negative P contributes positively)
        scores = []
        for v in nodes:
            s = 0.0
            row = P.get(v, {})
            for u in nodes:
                if u == v:
                    continue
                if u in row:
                    s += -row[u]
            scores.append((s, v))
        scores.sort(reverse=True, key=lambda x: x[0])
        return [v for _, v in scores]

    # Produce a cluster-respecting initial sequence seed
    def cluster_seed():
        if not clusters or len(clusters) == 1:
            return order_within_cluster(clusters[0]) if clusters else all_txns[:]
        # Order clusters by min singleton cost anchor
        cluster_keys = []
        for comp in clusters:
            best_anchor = min(comp, key=lambda t: singleton_cost.get(t, float('inf')))
            key_cost = singleton_cost.get(best_anchor, float('inf'))
            cluster_keys.append((key_cost, comp))
        cluster_keys.sort(key=lambda x: x[0])
        seq = []
        for _, comp in cluster_keys:
            seq.extend(order_within_cluster(comp))
        return seq

    # Greedy finish from a prefix using true extension cost and buddy-biased candidate pool
    def greedy_finish(prefix, rem_set, branch_factor):
        seq = list(prefix)
        rem = set(rem_set)
        while rem and time_left():
            prefix_tuple = tuple(seq)
            rem_list = list(rem)
            # Build candidate pool from buddies of last element + low singleton + random
            pool = []
            if seq:
                last = seq[-1]
                pool.extend([x for x in buddies.get(last, []) if x in rem])
            # Add top few low singleton txns from rem
            low_single = sorted(rem_list, key=lambda t: singleton_cost.get(t, float('inf')))[:min(4, len(rem_list))]
            pool.extend(low_single)
            # Fill random
            if len(pool) < branch_factor:
                need = min(branch_factor - len(pool), max(0, len(rem_list) - len(pool)))
                if need > 0:
                    # sample from remaining
                    remain = [t for t in rem_list if t not in pool]
                    if remain:
                        pool.extend(rng.sample(remain, min(need, len(remain))))
            # Deduplicate
            pool = list(dict.fromkeys(pool))
            # Score by extension cost
            best_t = None
            best_c = float('inf')
            for t in pool:
                c = eval_ext_cost(prefix_tuple, t)
                if c < best_c:
                    best_c = c
                    best_t = t
            if best_t is None:
                # fallback
                best_t = rem_list[0]
                best_c = eval_ext_cost(prefix_tuple, best_t)
            seq.append(best_t)
            rem.remove(best_t)
        if rem:
            seq.extend(list(rem))
        return eval_seq_cost(seq), seq

    # Deterministic multi-restart beam search with shared caches
    def run_beam(beam_width, cand_expand):
        all_set = set(all_txns)
        # Seed beam with best singletons plus cluster-based prefix
        seeds = []

        # cluster seed
        cseed = cluster_seed()
        if cseed:
            seeds.append([cseed[0]])

        # top singleton seeds
        topk = min(6, max(3, N // 20))
        seeds.extend([[t] for t in singles_sorted[:topk]])

        # a few random seeds deterministically shuffled
        others = [t for t in all_txns if [t] not in seeds]
        rng.shuffle(others)
        for t in others[:max(2, N // 30)]:
            seeds.append([t])

        # Initialize beam entries: (prefix_cost, seq, rem)
        beam = []
        for s in seeds:
            rem = all_set - set(s)
            beam.append((eval_seq_cost(s), s, rem))

        if not beam:
            seq = all_txns[:]
            rng.shuffle(seq)
            return eval_seq_cost(seq), seq

        beam.sort(key=lambda x: x[0])
        beam = beam[:beam_width]

        incumbent = float('inf')
        inc_seq = None

        steps = N - 1
        for _ in range(steps):
            if not time_left():
                break
            new_beam = []

            # Early tightening: greedily complete best K prefixes
            for (cost_so_far, seq, rem) in beam[:min(2, len(beam))]:
                if not time_left():
                    break
                c_try, s_try = greedy_finish(seq, rem, branch_factor=max(8, cand_expand))
                if len(s_try) == N and c_try < incumbent:
                    incumbent, inc_seq = c_try, s_try

            for cost_so_far, seq, rem in beam:
                if not rem:
                    if cost_so_far < incumbent:
                        incumbent, inc_seq = cost_so_far, seq[:]
                    new_beam.append((cost_so_far, seq, rem, cost_so_far))
                    continue
                # prune dominated prefix
                if cost_so_far >= incumbent:
                    continue

                rem_list = list(rem)

                # Candidate pool guided by buddies and low singletons
                cand_pool = []
                if seq:
                    last = seq[-1]
                    cand_pool.extend([x for x in buddies.get(last, []) if x in rem])
                low_s = sorted(rem_list, key=lambda t: singleton_cost.get(t, float('inf')))[:min(5, len(rem_list))]
                cand_pool.extend(low_s)
                # add random fill
                if len(cand_pool) < cand_expand:
                    need = cand_expand - len(cand_pool)
                    remain = [t for t in rem_list if t not in cand_pool]
                    if remain:
                        cand_pool.extend(rng.sample(remain, min(need, len(remain))))
                # Dedup
                cand_pool = list(dict.fromkeys(cand_pool))

                prefix_tuple = tuple(seq)
                scored = []
                for t in cand_pool:
                    if not time_left():
                        break
                    ec = eval_ext_cost(prefix_tuple, t)
                    if ec >= incumbent:
                        # prune child against incumbent
                        continue
                    delta = ec - cost_so_far
                    # Simple rank score: immediate cost and small LB using best singleton in rem\{t}
                    rem2 = rem - {t}
                    lb = 0
                    if rem2:
                        # weak lower bound: max of existing ec and min singleton add-on
                        lb = min(singleton_cost[u] for u in rem2)
                    rank = ec + 0.02 * lb + 0.001 * delta
                    scored.append((rank, ec, t))
                if not scored:
                    continue
                scored.sort(key=lambda x: x[0])
                top = scored[:min(beam_width, len(scored))]

                for rank, ec, t in top:
                    new_seq = seq + [t]
                    new_rem = rem.copy()
                    new_rem.remove(t)
                    new_beam.append((ec, new_seq, new_rem, rank))

            if not new_beam:
                break

            # Keep top unique prefixes by rank
            new_beam.sort(key=lambda x: x[3])
            unique = []
            seen = set()
            for entry in new_beam:
                key = tuple(entry[1])
                if key in seen:
                    continue
                seen.add(key)
                unique.append((entry[0], entry[1], entry[2]))
                if len(unique) >= beam_width:
                    break
            beam = unique

        # Final greedy completion
        best_cost = incumbent
        best_seq = inc_seq
        for cost_so_far, seq, rem in beam:
            if not time_left():
                break
            c_fin, s_fin = greedy_finish(seq, rem, branch_factor=max(8, cand_expand))
            if len(s_fin) == N and c_fin < best_cost:
                best_cost, best_seq = c_fin, s_fin

        if best_seq is None:
            seq = all_txns[:]
            rng.shuffle(seq)
            best_seq = seq
            best_cost = eval_seq_cost(seq)
        return best_cost, best_seq

    # Surrogate for VNS: sum of pairwise preference margins around boundaries
    # margin(k) for boundary between seq[k] and seq[k+1] is P[seq[k]][seq[k+1]] (negative preferred)
    # We compute unknown P on-demand using cached evals.
    def pair_pref(a, b):
        # Returns P[a][b], compute if unknown
        if b in P.get(a, {}):
            return P[a][b]
        # Compute using cached costs
        cab = eval_seq_cost([a, b])
        cba = eval_seq_cost([b, a])
        p = cab - cba
        P[a][b] = p
        P[b][a] = -p
        return p

    def seq_margin_sum(seq):
        s = 0.0
        for k in range(len(seq) - 1):
            s += pair_pref(seq[k], seq[k + 1])
        return s

    # Estimate surrogate delta for a relocate or swap by only evaluating affected boundaries
    def surrogate_delta(seq, move):
        # move: ("swap", i, j) or ("reloc", i, j) or ("block", i, j, w)
        n = len(seq)
        old_pairs = []
        new_pairs = []

        def add_pairs(indexes):
            idxs = sorted(set([i for i in indexes if 0 <= i < n - 1]))
            for k in idxs:
                old_pairs.append((seq[k], seq[k + 1]))

        if move[0] == "swap":
            _, i, j = move
            if i == j:
                return 0.0
            a, b = min(i, j), max(i, j)
            add_pairs([a - 1, a, b - 1, b])
            # construct new neighbors around a,b only
            cand = list(seq)
            cand[i], cand[j] = cand[j], cand[i]
            nn = []
            for k in [a - 1, a, b - 1, b]:
                if 0 <= k < n - 1:
                    nn.append((cand[k], cand[k + 1]))
            new_pairs.extend(nn)
        elif move[0] == "reloc":
            _, i, j = move
            if i == j:
                return 0.0
            a, b = min(i, j), max(i, j)
            add_pairs([i - 1, i, j - 1, j])
            cand = list(seq)
            v = cand.pop(i)
            cand.insert(j, v)
            nn = []
            for k in [min(i, j) - 1, min(i, j), max(i, j) - 1, max(i, j)]:
                if 0 <= k < n - 1:
                    nn.append((cand[k], cand[k + 1]))
            new_pairs.extend(nn)
        else:  # "block" move
            _, i, j, w = move
            i2 = min(i, i + w - 1)
            j2 = j
            idxs = [i2 - 1, i2, j2 - 1, j2, i - 1, i]
            add_pairs(idxs)
            cand = list(seq)
            block = cand[i:i + w]
            del cand[i:i + w]
            if j > i:
                j = j - w
            for k, v in enumerate(block):
                cand.insert(j + k, v)
            nn = []
            for k in range(min(idxs), max(idxs) + 1):
                if 0 <= k < n - 1:
                    nn.append((cand[k], cand[k + 1]))
            new_pairs.extend(nn)

        # Compute surrogate change for these pairs
        def sum_pairs(pairs):
            return sum(pair_pref(a, b) for (a, b) in pairs)

        return sum_pairs(new_pairs) - sum_pairs(old_pairs)

    def vns_local_refine(seq, cur_cost):
        best_seq = seq[:]
        best_cost = cur_cost
        n = len(best_seq)
        if n <= 2 or not time_left():
            return best_cost, best_seq

        # Precompute current surrogate to guide
        _ = seq_margin_sum(best_seq)

        # Neighborhood phases: swaps, relocations, block moves
        phases = [
            ("swap", 120),
            ("reloc", 140),
            ("block", 80)
        ]

        for phase, budget in phases:
            if not time_left():
                break
            # Candidate move generation
            candidates = []
            # Bias around high-violation boundaries
            margins = []
            for k in range(n - 1):
                margins.append((abs(pair_pref(best_seq[k], best_seq[k + 1])), k))
            margins.sort(reverse=True, key=lambda x: x[0])
            focus = [k for _, k in margins[:max(8, n // 10)]]

            tries = 0
            while tries < budget and time_left():
                tries += 1
                if phase == "swap":
                    # Choose indices near high-violation boundaries
                    if focus and rng.random() < 0.7:
                        k = rng.choice(focus)
                        i = k
                        j = min(n - 1, k + 1 + rng.randint(0, min(6, n - k - 1)))
                    else:
                        i, j = sorted(rng.sample(range(n), 2))
                    move = ("swap", i, j)
                elif phase == "reloc":
                    if focus and rng.random() < 0.7:
                        k = rng.choice(focus)
                        i = k if rng.random() < 0.5 else k + 1
                        j = rng.randint(0, n - 1)
                    else:
                        i, j = rng.sample(range(n), 2)
                    move = ("reloc", i, j)
                else:  # block
                    if n < 5:
                        continue
                    w = rng.randint(2, min(4, n // 5 + 2))
                    i = rng.randint(0, n - w)
                    j = rng.randint(0, n - (w if rng.random() < 0.5 else 1))
                    move = ("block", i, j, w)

                sd = surrogate_delta(best_seq, move)
                # Negative surrogate delta suggests improvement (more preferred order)
                candidates.append((sd, move))

            if not candidates:
                continue
            # Keep top 35% surrogate-ranked plus a small random tail
            candidates.sort(key=lambda x: x[0])
            topk = max(1, int(0.35 * len(candidates)))
            eval_set = [m for _, m in candidates[:topk]]
            # Diversity tail
            tail = candidates[topk:]
            if tail:
                eval_set.extend([m for _, m in rng.sample(tail, min(len(tail), max(3, len(candidates) // 12)))])

            improved = False
            for move in eval_set:
                if not time_left():
                    break
                if move[0] == "swap":
                    _, i, j = move
                    cand = best_seq[:]
                    cand[i], cand[j] = cand[j], cand[i]
                elif move[0] == "reloc":
                    _, i, j = move
                    cand = best_seq[:]
                    v = cand.pop(i)
                    cand.insert(j, v)
                else:
                    _, i, j, w = move
                    cand = best_seq[:]
                    block = cand[i:i + w]
                    del cand[i:i + w]
                    if j > i:
                        j = j - w
                    for k, v in enumerate(block):
                        cand.insert(j + k, v)
                c = eval_seq_cost(cand)
                if c < best_cost:
                    best_cost = c
                    best_seq = cand
                    improved = True
            if improved:
                # Recompute margins for the next phase iteration
                _ = seq_margin_sum(best_seq)

        return best_cost, best_seq

    # Global search orchestration
    global_best_cost = float('inf')
    global_best_seq = None

    # Deterministic parameter sets for restarts
    param_sets = [
        (16, 12),
        (12, 16),
        (8, 20)
    ]

    # Try a cluster-first greedy schedule to set a baseline incumbent
    if time_left():
        cs = cluster_seed()
        c0 = eval_seq_cost(cs)
        if c0 < global_best_cost:
            global_best_cost, global_best_seq = c0, cs

    # Run beam restarts
    for (bw, br) in param_sets:
        if not time_left():
            break
        c, s = run_beam(bw, br)
        if c < global_best_cost:
            global_best_cost, global_best_seq = c, s

    # Local VNS refinement
    if time_left():
        c2, s2 = vns_local_refine(global_best_seq, global_best_cost)
        if c2 < global_best_cost:
            global_best_cost, global_best_seq = c2, s2

    # Fallback safety: ensure permutation
    if global_best_seq is None or len(global_best_seq) != N or len(set(global_best_seq)) != N:
        seq = list(range(N))
        rng.shuffle(seq)
        global_best_seq = seq
        global_best_cost = eval_seq_cost(seq)

    return global_best_cost, global_best_seq


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