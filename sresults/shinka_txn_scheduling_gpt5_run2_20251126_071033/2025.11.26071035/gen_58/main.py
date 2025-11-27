# EVOLVE-BLOCK-START
"""Transaction scheduling algorithm for optimizing makespan across multiple workloads
Crossover: GRASP-style randomized best-insertion + cached evaluations + Beam + VND + LNS + Elites.
"""

import time
import random
import sys
import os

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
    Hybrid GRASP + Beam + VND + LNS with elites and path relinking to minimize makespan.

    Args:
        workload: Workload object containing transaction data
        num_seqs: Number of randomized restarts (search effort)

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    n = workload.num_txns
    all_txns = list(range(n))

    # -------- Adaptive parameters (merged from best historical variants) --------
    small = n <= 50
    med = 50 < n <= 80
    large = n > 80

    # Construction / GRASP
    CAND_SAMPLE_BASE = 12 if small else 9
    STARTER_SAMPLE = min(10, n)
    RCL_K = 3
    JITTER = 2
    rcl_alpha = 0.6  # cost vs regret balance when picking from RCL

    # Position sampling for insertion evaluations
    POS_SAMPLE_LIMIT = None if small else (15 if med else 15)

    # Beam search
    beam_width = max(5, min(10, (num_seqs // 2) + (2 if small else 0)))
    cand_per_state = min(28, max(10, n // (4 if large else 3)))
    lookahead_k = 6 if med or large else 8
    diversity_quota = max(1, int(0.25 * beam_width))
    alpha = 0.7  # score = alpha*c1 + (1-alpha)*best lookahead cost

    # Local search (VND)
    MAX_LS_PASSES = 2 if large else 3
    RELOC_TRIES = max(12, n // (3 if large else 2))
    RELOC_TRIES = int(RELOC_TRIES * max(1.0, min(2.0, 0.5 + 0.1 * max(1, num_seqs))))

    # LNS
    lns_rounds = 2 if med or large else 3

    # Iterated improvements
    ils_iterations = 2 if large else 3

    # Elites
    elite_cap = 4
    elites = []  # list of (cost, seq)

    # -------- Cached evaluator --------
    cost_cache = {}

    def eval_cost(prefix):
        """Evaluate and cache the cost for a (possibly partial) prefix sequence."""
        key = tuple(prefix)
        c = cost_cache.get(key)
        if c is None:
            c = workload.get_opt_seq_cost(list(prefix))
            cost_cache[key] = c
        return c

    # -------- Position sampling (deterministic stratified) --------
    def sample_positions(seq_len):
        """
        Deterministic stratified positions: always include ends and a few anchors.
        Keeps cache-friendly behavior and avoids excessive evaluations on long sequences.
        """
        total = seq_len + 1
        if POS_SAMPLE_LIMIT is None or total <= POS_SAMPLE_LIMIT:
            return list(range(total))
        # Evenly spaced anchors, include ends
        anchors = {0, seq_len, seq_len // 2, seq_len // 4, (3 * seq_len) // 4}
        anchors = {max(0, min(seq_len, p)) for p in anchors}
        # Add a few interior evenly spaced positions up to POS_SAMPLE_LIMIT
        needed = max(0, POS_SAMPLE_LIMIT - len(anchors))
        for i in range(1, needed + 1):
            pos = round(i * seq_len / (needed + 1))
            anchors.add(max(0, min(seq_len, pos)))
        return sorted(anchors)

    # Cache best and second-best insertion results per (seq, txn)
    best_two_cache = {}

    def best_two_insertions(seq, txn):
        """
        Compute best and second-best insertion cost/pos for regret-based selection.
        Uses deterministic position sampling and caching.
        Returns: (best_cost, best_pos, second_best_cost)
        """
        key = (tuple(seq), txn)
        cached = best_two_cache.get(key)
        if cached is not None:
            return cached
        positions = sample_positions(len(seq))
        best_c, best_p = float('inf'), 0
        second_c = float('inf')
        for pos in positions:
            cand = seq[:]
            cand.insert(pos, txn)
            c = eval_cost(cand)
            if c < best_c:
                second_c = best_c
                best_c, best_p = c, pos
            elif c < second_c:
                second_c = c
        if second_c == float('inf'):
            second_c = best_c
        res = (best_c, best_p, second_c)
        best_two_cache[key] = res
        return res

    # -------- GRASP constructor (regret-guided insertion) --------
    def select_best_starter():
        """Pick a robust starting transaction by sampling and choosing best by prefix cost."""
        candidates = random.sample(all_txns, STARTER_SAMPLE) if STARTER_SAMPLE < n else all_txns
        best_t, best_c = None, float('inf')
        for t in candidates:
            c = eval_cost([t])
            if c < best_c:
                best_c, best_t = c, t
        return best_t if best_t is not None else random.randint(0, n - 1)

    def construct_grasp():
        """
        Regret-guided randomized best-insertion construction with RCL.
        Returns (cost, seq)
        """
        remaining = set(all_txns)
        seq = []
        start = select_best_starter()
        seq.append(start)
        remaining.remove(start)

        # Strong second placement
        if remaining:
            pairs = []
            for t in random.sample(list(remaining), min(8, len(remaining))):
                for pos in [0, 1]:
                    cand = seq[:]
                    cand.insert(pos, t)
                    c = eval_cost(cand)
                    pairs.append((c, t, pos))
            if pairs:
                pairs.sort(key=lambda x: x[0])
                chosen = random.choice(pairs[:min(3, len(pairs))])
                _, t2, p2 = chosen
                seq.insert(p2, t2)
                remaining.remove(t2)

        # Iterative regret-aware insertions
        while remaining:
            # Candidate txns
            if len(remaining) <= max(8, 2 * CAND_SAMPLE_BASE):
                cand_txns = list(remaining)
            else:
                size = min(len(remaining), max(4, CAND_SAMPLE_BASE + random.randint(-JITTER, JITTER)))
                cand_txns = random.sample(list(remaining), size)

            scored = []
            for t in cand_txns:
                c1, p1, c2 = best_two_insertions(seq, t)
                regret = max(0.0, c2 - c1)
                scored.append((c1, regret, t, p1))

            if not scored:
                t = random.choice(list(remaining))
                seq.append(t)
                remaining.remove(t)
                continue

            scored.sort(key=lambda x: x[0])  # by best cost
            rcl_size = min(max(3, RCL_K), len(scored))
            rcl = scored[:rcl_size]

            # Choose among RCL: bias to cost vs regret with rcl_alpha
            if random.random() < rcl_alpha:
                rcl.sort(key=lambda x: x[0])
            else:
                rcl.sort(key=lambda x: (-x[1], x[0]))
            chosen = rcl[0] if random.random() < 0.6 else random.choice(rcl)
            _, _, t_star, p_star = chosen
            seq.insert(p_star, t_star)
            remaining.remove(t_star)

        return eval_cost(seq), seq

    # -------- Beam search with lookahead and suffix diversity --------
    def beam_search():
        """Regret-diverse beam search with two-step lookahead and suffix-2 diversity."""
        # Initialize with best singletons and one GRASP seed
        starters = [(eval_cost([t]), [t]) for t in all_txns]
        starters.sort(key=lambda x: x[0])
        init = starters[:min(len(starters), max(beam_width * 2, beam_width + 2))]
        gc, gs = construct_grasp()
        init.append((gc, gs))

        # Beam state: (cost, seq, remaining_set)
        beam = []
        used = set()
        for c, seq in init:
            key = tuple(seq)
            if key in used:
                continue
            used.add(key)
            rem = frozenset(t for t in all_txns if t not in seq)
            beam.append((c, seq, rem))
            if len(beam) >= beam_width:
                break

        best_complete = (float('inf'), [])

        for _depth in range(1, n + 1):
            if not beam:
                break
            next_pool = []
            layer_seen = set()
            suffix_seen = set()  # suffix diversity on last-2
            for c_so_far, prefix, rem in beam:
                if not rem:
                    if c_so_far < best_complete[0]:
                        best_complete = (c_so_far, prefix)
                    continue

                rem_list = list(rem)
                # Expand a subset of remaining
                expand_list = rem_list if len(rem_list) <= cand_per_state else random.sample(rem_list, cand_per_state)

                scored = []
                for t in expand_list:
                    new_prefix = prefix + [t]
                    c1 = eval_cost(new_prefix)

                    # lookahead: approximate best next extension
                    rem_after = [x for x in rem_list if x != t]
                    if rem_after:
                        k2 = min(lookahead_k, len(rem_after))
                        second = random.sample(rem_after, k2)
                        best_c2 = float('inf')
                        second_costs = []
                        for u in second:
                            cu = eval_cost(new_prefix + [u])
                            second_costs.append(cu)
                            if cu < best_c2:
                                best_c2 = cu
                        score = alpha * c1 + (1.0 - alpha) * best_c2
                        # regret proxy via dispersion
                        regret_proxy = (max(second_costs) - min(second_costs)) if len(second_costs) >= 2 else (best_c2 - c1)
                    else:
                        score = c1
                        regret_proxy = 0.0

                    scored.append((score, c1, regret_proxy, new_prefix, frozenset(rem_after)))

                if not scored:
                    continue

                # Select expansions by score and add a few regret-diverse ones
                scored.sort(key=lambda x: x[0])
                top_cost = scored[:max(1, min(len(scored), beam_width))]
                scored_by_regret = sorted(scored, key=lambda x: (-x[2], x[0]))
                top_regret = scored_by_regret[:min(diversity_quota, len(scored_by_regret))]
                cand_acts = top_cost + top_regret

                # Deduplicate and enforce suffix-2 diversity
                for score, c1, rg, seq_cand, rem_cand in cand_acts:
                    key = tuple(seq_cand)
                    if key in layer_seen:
                        continue
                    if len(seq_cand) >= 2:
                        sig = (seq_cand[-2], seq_cand[-1])
                    else:
                        sig = (None, seq_cand[-1])
                    if sig in suffix_seen:
                        continue
                    layer_seen.add(key)
                    suffix_seen.add(sig)
                    next_pool.append((c1, seq_cand, rem_cand))

            if not next_pool:
                break

            # Prune to beam width by cost; deduplicate by full prefix
            next_pool.sort(key=lambda x: x[0])
            pruned = []
            seen_prefixes = set()
            for c1, seq, rem in next_pool:
                key = tuple(seq)
                if key in seen_prefixes:
                    continue
                seen_prefixes.add(key)
                pruned.append((c1, seq, rem))
                if len(pruned) >= beam_width:
                    break
            beam = pruned

        # Return best complete if found
        for c, seq, rem in beam:
            if not rem and c < best_complete[0]:
                best_complete = (c, seq)
        if best_complete[1] and len(best_complete[1]) == n:
            return best_complete

        # Greedy completion from best partial
        if beam:
            c, seq, rem = min(beam, key=lambda x: x[0])
            cur = list(seq)
            rem_list = list(rem)
            while rem_list:
                best_t = None
                best_c = float('inf')
                for t in rem_list:
                    c2 = eval_cost(cur + [t])
                    if c2 < best_c:
                        best_c = c2
                        best_t = t
                cur.append(best_t)
                rem_list.remove(best_t)
            return eval_cost(cur), cur

        # Fallback
        ident = list(range(n))
        return eval_cost(ident), ident

    # -------- Local Search (VND) --------
    def local_search_adjacent_swaps(seq, curr_cost, max_passes=MAX_LS_PASSES):
        """Multiple passes of adjacent swap hill-climbing."""
        improved = True
        passes = 0
        best_seq = list(seq)
        best_cost = curr_cost

        while improved and passes < max_passes:
            improved = False
            passes += 1
            for i in range(len(best_seq) - 1):
                cand = best_seq[:]
                cand[i], cand[i + 1] = cand[i + 1], cand[i]
                c = eval_cost(cand)
                if c < best_cost:
                    best_cost = c
                    best_seq = cand
                    improved = True
        return best_seq, best_cost

    def local_search_relocations(seq, curr_cost, tries=RELOC_TRIES):
        """
        Random relocation: remove at index i and reinsert at j if improves.
        Accepts improving moves under a limited budget; resets trials on improvement.
        """
        best_seq = list(seq)
        best_cost = curr_cost
        if len(best_seq) <= 2:
            return best_seq, best_cost

        trials = 0
        while trials < tries:
            i = random.randint(0, len(best_seq) - 1)
            j = random.randint(0, len(best_seq) - 1)
            if i == j:
                trials += 1
                continue
            cand = best_seq[:]
            t = cand.pop(i)
            if j > i:
                j -= 1
            cand.insert(j, t)
            c = eval_cost(cand)
            if c < best_cost:
                best_cost = c
                best_seq = cand
                trials = 0
            else:
                trials += 1
        return best_seq, best_cost

    def local_search_pair_swaps(seq, curr_cost, tries=None):
        """
        Sampled non-adjacent pair swaps: explore up to 'tries' random pairs and apply the best improving swap.
        Complements Or-opt and adjacent swaps to resolve distant conflicts.
        """
        best_seq = list(seq)
        best_cost = curr_cost
        L = len(best_seq)
        if L <= 3:
            return best_seq, best_cost
        if tries is None:
            tries = min(200, max(60, n))

        best_delta = 0.0
        best_pair = None
        for _ in range(tries):
            i = random.randint(0, L - 1)
            j = random.randint(0, L - 1)
            if i == j or abs(i - j) <= 1:
                continue
            cand = best_seq[:]
            cand[i], cand[j] = cand[j], cand[i]
            c = eval_cost(cand)
            delta = best_cost - c
            if delta > best_delta:
                best_delta = delta
                best_pair = (i, j)
        if best_pair is not None and best_delta > 0:
            i, j = best_pair
            best_seq[i], best_seq[j] = best_seq[j], best_seq[i]
            best_cost = eval_cost(best_seq)
        return best_seq, best_cost

    def or_opt_block(seq, curr_cost, k):
        """
        Or-opt move with block size k: relocate any contiguous block of length k to its best position.
        Performs best-improving passes until no improvement.
        """
        best_seq = list(seq)
        best_cost = curr_cost
        if len(best_seq) <= k:
            return best_seq, best_cost

        improved = True
        while improved:
            improved = False
            move_best_cost = best_cost
            move = None  # (i, pos)
            L = len(best_seq)
            for i in range(0, L - k + 1):
                block = best_seq[i:i + k]
                base = best_seq[:i] + best_seq[i + k:]
                positions = sample_positions(len(base))
                for pos in positions:
                    cand = base[:]
                    cand[pos:pos] = block
                    c = eval_cost(cand)
                    if c < move_best_cost:
                        move_best_cost = c
                        move = (i, pos)
            if move is not None:
                i, pos = move
                block = best_seq[i:i + k]
                base = best_seq[:i] + best_seq[i + k:]
                new_seq = base[:]
                new_seq[pos:pos] = block
                best_seq = new_seq
                best_cost = move_best_cost
                improved = True
        return best_seq, best_cost

    def vnd_local_search(seq, curr_cost):
        """
        Variable Neighborhood Descent:
        Or-opt blocks k=3,2,1 (best-improving), then adjacent swaps, sampled pair swaps, then relocations.
        Repeat cycle until no further improvement.
        """
        best_seq = list(seq)
        best_cost = curr_cost
        changed = True
        while changed:
            changed = False
            for k in (3, 2, 1):
                s, c = or_opt_block(best_seq, best_cost, k)
                if c < best_cost:
                    best_seq, best_cost = s, c
                    changed = True
            s, c = local_search_adjacent_swaps(best_seq, best_cost, max_passes=1)
            if c < best_cost:
                best_seq, best_cost = s, c
                changed = True
            s, c = local_search_pair_swaps(best_seq, best_cost, tries=min(200, max(60, n)))
            if c < best_cost:
                best_seq, best_cost = s, c
                changed = True
            s, c = local_search_relocations(best_seq, best_cost, tries=max(8, RELOC_TRIES // 2))
            if c < best_cost:
                best_seq, best_cost = s, c
                changed = True
        return best_seq, best_cost

    # -------- LNS: Ruin-and-Recreate with regret-guided reinsertion --------
    def lns_ruin_recreate(seq, curr_cost, rounds=lns_rounds):
        """
        Remove a contiguous block (plus a few random extras) and rebuild with regret-guided best-two insertions.
        """
        best_seq = list(seq)
        best_cost = curr_cost
        for _ in range(max(1, rounds)):
            L = len(best_seq)
            if L < 6:
                break
            # Choose block to remove
            base_len = max(4, L // 10)
            block_len = min(L - 2, base_len + random.randint(0, 3))
            start = random.randint(0, L - block_len)
            removed = best_seq[start:start + block_len]
            skeleton = best_seq[:start] + best_seq[start + block_len:]
            # Optionally remove a few extras for diversification
            extra_count = min(3, max(0, L // 30))
            extras_idx = sorted(random.sample(range(len(skeleton)), extra_count)) if extra_count and len(skeleton) > extra_count else []
            extras = []
            offset = 0
            for idx in extras_idx:
                idx_adj = idx - offset
                extras.append(skeleton.pop(idx_adj))
                offset += 1
            to_insert = removed + extras

            # Regret-guided reinsertion with small candidate sampling per step
            rebuilt = list(skeleton)
            remaining = list(to_insert)
            while remaining:
                k_t = min(len(remaining), max(4, len(remaining) // 2))
                cand_txns = remaining if len(remaining) <= k_t else random.sample(remaining, k_t)
                scored = []
                for t in cand_txns:
                    best_c, best_p, second_c = best_two_insertions(rebuilt, t)
                    regret = max(0.0, second_c - best_c)
                    scored.append((regret, best_c, best_p, t))
                if not scored:
                    t = remaining.pop()
                    rebuilt.append(t)
                    continue
                # Choose highest regret; tie-break on best cost
                scored.sort(key=lambda x: (-x[0], x[1]))
                _, _, best_p, t = scored[0]
                rebuilt.insert(best_p, t)
                remaining.remove(t)

            c_final = eval_cost(rebuilt)
            if c_final < best_cost:
                best_cost = c_final
                best_seq = rebuilt
        return best_seq, best_cost

    # -------- Elite management and path relinking --------
    def add_elite(cost, seq):
        nonlocal elites
        if not seq or len(seq) != n:
            return
        key = tuple(seq)
        for i, (c, s) in enumerate(elites):
            if tuple(s) == key:
                if cost < c:
                    elites[i] = (cost, list(seq))
                return
        elites.append((cost, list(seq)))
        elites.sort(key=lambda x: x[0])
        if len(elites) > elite_cap:
            elites = elites[:elite_cap]

    def path_relink(a_seq, b_seq):
        """Transform a toward b, evaluating along the path; return best encountered."""
        target_pos = {t: i for i, t in enumerate(b_seq)}
        cur = list(a_seq)
        best_c = eval_cost(cur)
        best_s = list(cur)
        for i in range(len(cur)):
            desired = b_seq[i]
            j = cur.index(desired)
            if j == i:
                continue
            elem = cur.pop(j)
            cur.insert(i, elem)
            c = eval_cost(cur)
            if c < best_c:
                best_c = c
                best_s = list(cur)
        return best_c, best_s

    # -------- Search Orchestration --------
    best_cost = float('inf')
    best_seq = list(range(n))

    # Seed RNG lightly per call for diversity
    random.seed((n * 911 + num_seqs * 131 + 7) % (2**32 - 1))

    # 1) GRASP seeds with quick VND polish
    seeds = max(3, min(6, num_seqs))
    for _ in range(seeds):
        c0, s0 = construct_grasp()
        s0, c0 = vnd_local_search(s0, c0)
        add_elite(c0, s0)
        if c0 < best_cost:
            best_cost, best_seq = c0, s0

    # 2) Beam search seeds (1-2 runs)
    beam_runs = max(1, num_seqs // 3)
    for _ in range(beam_runs):
        cb, sb = beam_search()
        sb, cb = vnd_local_search(sb, cb)
        add_elite(cb, sb)
        if cb < best_cost:
            best_cost, best_seq = cb, sb

    # 3) Path relinking between top elites and refine
    if len(elites) >= 2:
        base_cost, base_seq = elites[0]
        for i in range(1, min(len(elites), elite_cap)):
            _, other = elites[i]
            c_rel, s_rel = path_relink(base_seq, other)
            s_rel, c_rel = vnd_local_search(s_rel, c_rel)
            add_elite(c_rel, s_rel)
            if c_rel < best_cost:
                best_cost, best_seq = c_rel, s_rel

    # 4) Iterated LNS + VND from incumbent/elite to escape local minima
    incumbent_cost, incumbent_seq = best_cost, best_seq
    for _ in range(ils_iterations):
        # Diversify by starting from one of elites occasionally
        if elites and random.random() < 0.4:
            start_c, start_s = random.choice(elites[:min(3, len(elites))])
        else:
            start_c, start_s = incumbent_cost, incumbent_seq

        s1, c1 = lns_ruin_recreate(start_s, start_c, rounds=lns_rounds)
        s2, c2 = vnd_local_search(s1, c1)
        if c2 < incumbent_cost:
            incumbent_cost, incumbent_seq = c2, s2
            add_elite(c2, s2)
        # small adjacent polish
        s3, c3 = local_search_adjacent_swaps(incumbent_seq, incumbent_cost, max_passes=1)
        if c3 < incumbent_cost:
            incumbent_cost, incumbent_seq = c3, s3
            add_elite(c3, s3)

    best_cost, best_seq = incumbent_cost, incumbent_seq

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