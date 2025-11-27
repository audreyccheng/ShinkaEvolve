# EVOLVE-BLOCK-START
"""Transaction scheduling algorithm for optimizing makespan across multiple workloads"""

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
    Find a low-makespan schedule using regret-based best-insertion construction
    with memoized cost evaluation, followed by multi-neighborhood local refinement,
    iterated local search, large-neighborhood destroy/repair, and elite path relinking.

    Args:
        workload: Workload object containing transaction data
        num_seqs: Controls search breadth (number of seeds, sampling sizes)

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    n = workload.num_txns
    all_txns = list(range(n))

    # Memoization to avoid repeated cost computations for the same sequence
    cost_cache = {}
    def seq_cost(seq):
        key = tuple(seq)
        if key in cost_cache:
            return cost_cache[key]
        c = workload.get_opt_seq_cost(seq)
        cost_cache[key] = c
        return c

    # ----------------------------
    # Parameters (adaptive)
    # ----------------------------
    elite_size = max(3, min(6, 2 + num_seqs // 3))
    seed_elite_singletons = max(2, min(6, int(num_seqs)))
    seed_random_additional = max(1, min(4, int((num_seqs + 1) // 3)))

    # Candidate sampling sizes
    k_txn_sample = min(16, max(8, 2 + int(1.5 * num_seqs)))  # txns per insertion
    k_pos_sample = min(10, max(6, 2 + int(1.2 * num_seqs)))  # positions per insertion
    rem_all_threshold = 14                                    # when few remain, consider all txns

    # Regret selection probability (exploration vs exploitation)
    regret_prob = 0.65

    # Beam for constructive insertion
    local_beam_width = max(3, min(6, 2 + num_seqs // 2))
    high_regret_quota_ratio = 0.3
    diversity_suffix_k = 2
    endgame_all_pos_threshold = max(6, min(12, num_seqs))     # when few remain, try all positions

    # Local search parameters
    ls_adj_rounds_max = 2
    two_opt_trials = min(40, max(12, n // 3))
    reinsertion_pos_factor = 1.0  # multiplier for k_pos_sample

    # Iterated local search (ILS) / perturbation
    ils_rounds = max(2, min(5, 1 + num_seqs // 4))
    perturb_swap_count = max(2, min(6, 2 + num_seqs // 3))
    perturb_block_len = max(3, min(10, 3 + num_seqs // 2))

    # Large Neighborhood Search (LNS)
    lns_iters = max(2, min(6, 2 + num_seqs // 3))
    destroy_frac_range = (0.08, 0.18)

    # ----------------------------
    # Helpers
    # ----------------------------
    def position_samples(seq_len, focus_idx=None, k_positions=None):
        """Sample insertion positions; include anchors and positions near focus_idx if provided.
        When the sequence is small, evaluate all positions for accuracy."""
        if seq_len <= 1:
            return [0, seq_len]
        # For short sequences, evaluate all positions deterministically
        if seq_len <= 12:
            return list(range(seq_len + 1))
        k = k_positions if k_positions is not None else k_pos_sample
        pos_set = {0, seq_len, seq_len // 2}
        # Bias positions near the focus index (if provided)
        if focus_idx is not None:
            for d in (-3, -2, -1, 0, 1, 2, 3):
                p = focus_idx + d
                if 0 <= p <= seq_len:
                    pos_set.add(p)
        # Add random internal positions
        for _ in range(min(k, seq_len + 1)):
            pos_set.add(random.randint(0, seq_len))
        return sorted(pos_set)

    def evaluate_best_two_positions(base_seq, t, pos_list):
        """Return (best_cost, best_pos, second_best_cost) for inserting t into base_seq over pos_list."""
        best = (float('inf'), None)
        second = float('inf')
        for p in pos_list:
            cand = base_seq[:p] + [t] + base_seq[p:]
            c = seq_cost(cand)
            if c < best[0]:
                second = best[0]
                best = (c, p)
            elif c < second:
                second = c
        return best[0], best[1], second

    # ----------------------------
    # Construction (regret-insertion with small beam and diversity)
    # ----------------------------
    def regret_insertion_build(seed_t=None):
        """Construct a schedule using regret-based insertion with a small beam and diversity."""
        if seed_t is None:
            seed_t = random.randint(0, n - 1)

        # Beam holds tuples: (seq, rem_set, cost)
        seq0 = [seed_t]
        rem0 = set(all_txns)
        rem0.remove(seed_t)
        beam = [(seq0, rem0, seq_cost(seq0))]

        # Add a second element by prepend/append across beam to shape early order
        expansions = []
        for seq, rem, _ in beam:
            if not rem:
                expansions.append((seq, rem, seq_cost(seq), 0.0))
                continue
            cand_txns = list(rem)
            if len(cand_txns) > k_txn_sample:
                cand_txns = random.sample(cand_txns, k_txn_sample)
            for t in cand_txns:
                for new_seq in ([t] + seq, seq + [t]):
                    c = seq_cost(new_seq)
                    new_rem = rem.copy()
                    new_rem.remove(t)
                    expansions.append((new_seq, new_rem, c, 0.0))
        # Select initial beam by cost with suffix diversity
        expansions.sort(key=lambda x: x[2])
        next_beam = []
        seen_sig = set()
        for s, r, c, _ in expansions:
            sig = tuple(s[-diversity_suffix_k:]) if len(s) >= diversity_suffix_k else tuple(s)
            if sig in seen_sig:
                continue
            seen_sig.add(sig)
            next_beam.append((s, r, c))
            if len(next_beam) >= local_beam_width:
                break
        if next_beam:
            beam = next_beam

        # Grow sequences
        while True:
            if all(len(rem) == 0 for _, rem, _ in beam):
                break

            expansions = []
            seen_seqs = set()
            for seq, rem, _ in beam:
                if not rem:
                    key = tuple(seq)
                    if key not in seen_seqs:
                        seen_seqs.add(key)
                        expansions.append((seq, rem, seq_cost(seq), 0.0, None))
                    continue

                # Candidate transactions
                if len(rem) <= rem_all_threshold:
                    cand_txns = list(rem)
                else:
                    cand_txns = random.sample(list(rem), min(k_txn_sample, len(rem)))

                # Positions policy (all vs sampled)
                use_all_pos = (len(rem) <= endgame_all_pos_threshold) or (len(seq) <= 18)
                pos_list = list(range(len(seq) + 1)) if use_all_pos else position_samples(len(seq))

                for t in cand_txns:
                    best_c, best_p, second_c = evaluate_best_two_positions(seq, t, pos_list)
                    new_seq = seq[:best_p] + [t] + seq[best_p:]
                    new_rem = rem.copy()
                    new_rem.remove(t)
                    regret = second_c - best_c if second_c < float('inf') else 0.0
                    key = tuple(new_seq)
                    if key in seen_seqs:
                        continue
                    seen_seqs.add(key)
                    expansions.append((new_seq, new_rem, best_c, regret, t))

            if not expansions:
                break

            # Rank expansions
            by_cost = sorted(expansions, key=lambda x: (x[2], -x[3]))
            by_regret = sorted(expansions, key=lambda x: (-x[3], x[2]))

            # Select next beam with diversity and regret quota
            base_quota = max(1, int(local_beam_width * (1.0 - high_regret_quota_ratio)))
            regret_quota = max(0, local_beam_width - base_quota)
            next_beam = []
            seen_sig = set()

            def try_add(entry):
                s, r, c, reg, _t = entry
                sig = tuple(s[-diversity_suffix_k:]) if len(s) >= diversity_suffix_k else tuple(s)
                if sig in seen_sig:
                    return False
                seen_sig.add(sig)
                next_beam.append((s, r, c))
                return True

            i = 0
            while len(next_beam) < base_quota and i < len(by_cost):
                try_add(by_cost[i])
                i += 1
            j = 0
            while len(next_beam) < base_quota + regret_quota and j < len(by_regret):
                try_add(by_regret[j])
                j += 1
            k = 0
            while len(next_beam) < local_beam_width and k < len(by_cost):
                try_add(by_cost[k])
                k += 1

            if not next_beam:
                next_beam = [(s, r, c) for s, r, c, _, _ in by_cost[:local_beam_width]]

            beam = next_beam

        # Choose best complete
        best_seq = None
        best_cost = float('inf')
        for seq, rem, cost in beam:
            if rem:
                seq_complete = seq[:] + sorted(list(rem))
                c = seq_cost(seq_complete)
                if c < best_cost:
                    best_cost = c
                    best_seq = seq_complete
            else:
                if cost < best_cost:
                    best_cost = cost
                    best_seq = seq

        return best_seq

    # ----------------------------
    # Local refinement: VND with adjacent swaps, Or-opt (1,2,3), and sampled 2-opt
    # ----------------------------
    def local_refine(seq):
        best_seq = seq[:]
        best_cost = seq_cost(best_seq)
        improved_outer = True

        adj_rounds = 0

        def two_opt_sample_pass(cur_seq, cur_cost):
            tried = 0
            while tried < two_opt_trials:
                i = random.randint(0, n - 2)
                j = random.randint(i + 2, n - 1) if i + 2 < n else None
                tried += 1
                if j is None:
                    continue
                cand = cur_seq[:]
                cand[i], cand[j] = cand[j], cand[i]
                c = seq_cost(cand)
                if c < cur_cost:
                    return True, c, cand
            return False, cur_cost, cur_seq

        def try_reinsertion(cur_seq, cur_cost):
            k_positions = max(6, int(reinsertion_pos_factor * k_pos_sample))
            for i in range(n):
                item = cur_seq[i]
                base = cur_seq[:i] + cur_seq[i + 1:]
                positions = position_samples(len(base), focus_idx=i, k_positions=k_positions) if len(base) > 20 else list(range(len(base) + 1))
                for p in positions:
                    cand = base[:p] + [item] + base[p:]
                    if cand == cur_seq:
                        continue
                    c = seq_cost(cand)
                    if c < cur_cost:
                        return True, c, cand
            return False, cur_cost, cur_seq

        def try_or_opt(cur_seq, cur_cost, k):
            L = len(cur_seq)
            for i in range(L - k + 1):
                block = cur_seq[i:i + k]
                base = cur_seq[:i] + cur_seq[i + k:]
                positions = position_samples(len(base), focus_idx=i, k_positions=k_pos_sample) if len(base) > 20 else list(range(len(base) + 1))
                for p in positions:
                    if p == i:
                        continue
                    cand = base[:p] + block + base[p:]
                    if cand == cur_seq:
                        continue
                    c = seq_cost(cand)
                    if c < cur_cost:
                        return True, c, cand
            return False, cur_cost, cur_seq

        while improved_outer:
            improved_outer = False

            # Adjacent swap pass (limited rounds)
            if adj_rounds < ls_adj_rounds_max:
                improved = True
                while improved:
                    improved = False
                    for i in range(n - 1):
                        cand = best_seq[:]
                        cand[i], cand[i + 1] = cand[i + 1], cand[i]
                        c = seq_cost(cand)
                        if c < best_cost:
                            best_cost = c
                            best_seq = cand
                            improved = True
                            improved_outer = True
                            break
                adj_rounds += 1
                if improved_outer:
                    continue

            # Or-opt k=1 (single reinsertion)
            changed, new_cost, new_seq = try_reinsertion(best_seq, best_cost)
            if changed:
                best_seq, best_cost = new_seq, new_cost
                improved_outer = True
                continue

            # Or-opt k=2
            changed, new_cost, new_seq = try_or_opt(best_seq, best_cost, 2)
            if changed:
                best_seq, best_cost = new_seq, new_cost
                improved_outer = True
                continue

            # Or-opt k=3
            changed, new_cost, new_seq = try_or_opt(best_seq, best_cost, 3)
            if changed:
                best_seq, best_cost = new_seq, new_cost
                improved_outer = True
                continue

            # Sampled 2-opt pass (non-adjacent swaps)
            changed, new_cost, new_seq = two_opt_sample_pass(best_seq, best_cost)
            if changed:
                best_seq = new_seq
                best_cost = new_cost
                improved_outer = True

        return best_cost, best_seq

    # ----------------------------
    # Perturbation methods for ILS
    # ----------------------------
    def perturb(seq):
        s = seq[:]
        mode = random.random()
        if mode < 0.5:
            # Random swaps
            for _ in range(perturb_swap_count):
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)
                if i == j:
                    continue
                s[i], s[j] = s[j], s[i]
        else:
            # Block shuffle
            if n > perturb_block_len + 2:
                start = random.randint(0, n - perturb_block_len - 1)
                block = s[start:start + perturb_block_len]
                del s[start:start + perturb_block_len]
                insert_pos = random.randint(0, len(s))
                s = s[:insert_pos] + block + s[insert_pos:]
        return s

    # ----------------------------
    # LNS destroy-and-repair
    # ----------------------------
    def lns_attempt(seq):
        cur = seq[:]
        # Choose destroy size
        frac = random.uniform(*destroy_frac_range)
        m = max(4, min(n // 2, int(frac * n)))
        # Mix of random removal and contiguous block removal
        if random.random() < 0.5:
            remove_idxs = sorted(random.sample(range(n), m))
        else:
            start = random.randint(0, n - m)
            remove_idxs = list(range(start, start + m))
        remove_set = set(remove_idxs)
        removed = [cur[i] for i in remove_idxs]
        remaining = [cur[i] for i in range(n) if i not in remove_set]
        # Repair using regret insertion
        seq_rep = remaining[:]
        rem_set = removed[:]
        while rem_set:
            if len(rem_set) > k_txn_sample:
                cand_txns = random.sample(rem_set, k_txn_sample)
            else:
                cand_txns = rem_set[:]

            best_overall = (float('inf'), None, None)  # cost, txn, new_seq
            best_by_regret = (float('-inf'), None, None)

            for t in cand_txns:
                pos_list = position_samples(len(seq_rep))
                best_c, best_p, second_c = evaluate_best_two_positions(seq_rep, t, pos_list)
                # Track pure best
                if best_c < best_overall[0]:
                    new_seq = seq_rep[:best_p] + [t] + seq_rep[best_p:]
                    best_overall = (best_c, t, new_seq)
                # Track regret
                regret = second_c - best_c if second_c < float('inf') else 0.0
                if regret > best_by_regret[0]:
                    new_seq_r = seq_rep[:best_p] + [t] + seq_rep[best_p:]
                    best_by_regret = (regret, t, new_seq_r)

            pick_regret = (random.random() < regret_prob)
            chosen = best_by_regret if pick_regret and best_by_regret[1] is not None else best_overall
            if chosen[1] is None:
                t = random.choice(rem_set)
                seq_rep = seq_rep + [t]
                rem_set.remove(t)
            else:
                seq_rep = chosen[2]
                rem_set.remove(chosen[1])

        c_rep, s_rep = local_refine(seq_rep)
        return c_rep, s_rep

    # ----------------------------
    # Path Relinking: move current solution towards target elite by aligning positions
    # ----------------------------
    def path_relink(source_seq, target_seq, max_moves=12):
        pos_in_target = {t: i for i, t in enumerate(target_seq)}
        s = source_seq[:]
        best_c = seq_cost(s)
        best_s = s[:]
        moves = 0
        # Choose items with largest position displacement
        displacement = [(abs(i - pos_in_target[s[i]]), i) for i in range(n)]
        displacement.sort(reverse=True)
        for _, idx in displacement:
            if moves >= max_moves:
                break
            item = s[idx]
            desired = pos_in_target[item]
            if desired == idx:
                continue
            base = s[:idx] + s[idx + 1:]
            # Insert at desired (bounded within current length)
            desired = max(0, min(desired, len(base)))
            cand = base[:desired] + [item] + base[desired:]
            c = seq_cost(cand)
            if c < best_c:
                best_c = c
                best_s = cand[:]
                s = cand
                moves += 1
        if best_c < float('inf'):
            return best_c, best_s
        return seq_cost(source_seq), source_seq

    # ----------------------------
    # Seeding and elite management
    # ----------------------------
    singleton_scores = []
    for t in all_txns:
        singleton_scores.append((seq_cost([t]), t))
    singleton_scores.sort(key=lambda x: x[0])

    seed_txns = [t for _, t in singleton_scores[:seed_elite_singletons]]
    remaining = [t for t in all_txns if t not in seed_txns]
    if remaining and seed_random_additional > 0:
        extra = random.sample(remaining, min(seed_random_additional, len(remaining)))
        seed_txns.extend(extra)

    elite = []  # list of (cost, seq)
    def add_elite(c, s):
        nonlocal elite
        elite.append((c, s))
        elite.sort(key=lambda x: x[0])
        # keep unique sequences by suffix of length 2 to add diversity
        uniq = []
        seen_sig = set()
        for c1, s1 in elite:
            sig = tuple(s1[-2:]) if len(s1) >= 2 else tuple(s1)
            if sig in seen_sig:
                continue
            seen_sig.add(sig)
            uniq.append((c1, s1))
            if len(uniq) >= elite_size:
                break
        elite = uniq

    # Build and refine from seeds
    for seed in seed_txns:
        seq0 = regret_insertion_build(seed)
        c1, s1 = local_refine(seq0)
        add_elite(c1, s1)

    # If no elite (shouldn't happen), fallback
    if not elite:
        base = all_txns[:]
        random.shuffle(base)
        elite = [(seq_cost(base), base)]

    best_overall_cost, best_overall_seq = elite[0]

    # ----------------------------
    # Iterated Local Search (ILS)
    # ----------------------------
    cur_cost, cur_seq = best_overall_cost, best_overall_seq[:]
    for _ in range(ils_rounds):
        pert = perturb(cur_seq)
        c2, s2 = local_refine(pert)
        if c2 < cur_cost:
            cur_cost, cur_seq = c2, s2
            add_elite(c2, s2)
            if c2 < best_overall_cost:
                best_overall_cost, best_overall_seq = c2, s2

    # ----------------------------
    # LNS destroy-and-repair
    # ----------------------------
    for _ in range(lns_iters):
        c3, s3 = lns_attempt(best_overall_seq)
        if c3 < best_overall_cost:
            best_overall_cost, best_overall_seq = c3, s3
            add_elite(c3, s3)

    # ----------------------------
    # Path Relinking among elites
    # ----------------------------
    if len(elite) >= 2:
        base_cost, base_seq = best_overall_cost, best_overall_seq
        partners = elite[1:min(len(elite), elite_size)]
        for c_t, s_t in partners:
            pr_c, pr_s = path_relink(base_seq, s_t, max_moves=max(8, min(12, n // 8)))
            if pr_c < best_overall_cost:
                best_overall_cost, best_overall_seq = pr_c, pr_s

    return best_overall_cost, best_overall_seq


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