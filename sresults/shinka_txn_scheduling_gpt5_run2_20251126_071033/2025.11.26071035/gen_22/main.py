# EVOLVE-BLOCK-START
"""Transaction scheduling algorithm for optimizing makespan across multiple workloads"""

import time
import random
import sys
import os

# Add the openevolve_examples directory to the path to import txn_simulator and workloads
# Find the repository root by looking for the openevolve_examples directory
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
    Find a low-makespan schedule using regret-guided insertion beam search
    with memoized cost evaluation, followed by strong local refinement
    (adjacent swaps + reinsertion + sampled pair swaps + segment reversals)
    and a light iterated local search with structured perturbations.

    Args:
        workload: Workload object containing transaction data
        num_seqs: Controls search breadth (seeds, beam width, sampling sizes)

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

    # Parameterization (tuned for quality/runtime balance)
    elite_seeds = max(2, min(6, int(num_seqs)))           # number of elite singleton starts
    random_seeds = max(1, min(3, int(num_seqs) // 4))     # random additional starts
    # Candidate sampling sizes
    k_txn_sample = min(12, max(6, 2 + int(num_seqs)))     # txns to consider per insertion
    k_pos_sample = 6                                       # insertion positions to sample
    rem_all_threshold = 12                                 # when few remain, consider all txns
    # Local insertion beam
    local_beam_width = max(2, min(4, int(num_seqs) // 3))  # keep multiple good partials
    # Beam diversity controls
    diversity_suffix_len = 2
    regret_diversity_quota = max(1, local_beam_width // 4)  # 25% of beam for high-regret expansions
    # Iterated local search parameters
    perturbations = max(2, min(4, int(num_seqs) // 4))
    random_swap_trials = 3
    # Large Neighborhood Search (LNS)
    lns_iters = max(2, min(6, 1 + num_seqs // 3))
    destroy_frac_range = (0.08, 0.18)
    lns_regret_prob = 0.6

    # Helper to generate insertion position samples (always include ends)
    def position_samples(seq_len):
        if seq_len <= 1:
            return [0, seq_len]
        pos_set = {0, seq_len, seq_len // 2}
        # Add a few random internal positions
        for _ in range(min(k_pos_sample, seq_len + 1)):
            pos_set.add(random.randint(0, seq_len))
        # Keep deterministic ordering for caching benefits
        return sorted(pos_set)

    # Construct a schedule from a seed using regret-guided insertion beam
    def build_from_seed(seed_t):
        # Beam holds tuples: (seq, rem_set, cost)
        seq0 = [seed_t]
        rem0 = set(all_txns)
        rem0.remove(seed_t)
        beam = [(seq0, rem0, seq_cost(seq0))]

        while True:
            # If all states are complete, stop
            if all(len(rem) == 0 for _, rem, _ in beam):
                break

            expansions = []
            for seq, rem, base_cost in beam:
                if not rem:
                    # Carry forward completed sequences unchanged
                    expansions.append((seq, rem, seq_cost(seq), 0.0, None))
                    continue

                # Candidate transactions to insert next
                if len(rem) <= rem_all_threshold:
                    cand_txns = list(rem)
                else:
                    cand_txns = random.sample(list(rem), min(k_txn_sample, len(rem)))

                positions = position_samples(len(seq))

                # For each candidate txn, find best and second-best insertion positions (regret)
                for t in cand_txns:
                    best_cost = float('inf')
                    second_best = float('inf')
                    best_seq = None
                    for p in positions:
                        new_seq = seq[:p] + [t] + seq[p:]
                        c = seq_cost(new_seq)
                        if c < best_cost:
                            second_best = best_cost
                            best_cost = c
                            best_seq = new_seq
                        elif c < second_best:
                            second_best = c
                    regret = (second_best - best_cost) if second_best < float('inf') else 0.0
                    new_rem = rem.copy()
                    new_rem.remove(t)
                    expansions.append((best_seq, new_rem, best_cost, regret, t))

            if not expansions:
                break

            # Build ranked lists for selection: by cost and by regret
            sorted_by_cost = sorted(expansions, key=lambda x: (x[2], -x[3]))
            sorted_by_regret = sorted(expansions, key=lambda x: (-x[3], x[2]))

            # Next beam: mix top-by-cost with a small quota of high-regret candidates.
            next_beam = []
            seen_suffix = set()
            seen_seq = set()
            k_main = max(1, local_beam_width - regret_diversity_quota)

            def suffix_sig(s):
                if not s:
                    return (None,)
                if len(s) >= diversity_suffix_len:
                    return tuple(s[-diversity_suffix_len:])
                # pad with None for short sequences
                return tuple([None] * (diversity_suffix_len - len(s)) + s)

            # Fill main portion by cost
            for seq, rem, cost, regret, t in sorted_by_cost:
                tup = tuple(seq)
                suf = suffix_sig(seq)
                if tup in seen_seq or suf in seen_suffix:
                    continue
                seen_seq.add(tup)
                seen_suffix.add(suf)
                next_beam.append((seq, rem, cost))
                if len(next_beam) >= k_main:
                    break

            # Fill diversity portion by regret
            for seq, rem, cost, regret, t in sorted_by_regret:
                if len(next_beam) >= local_beam_width:
                    break
                tup = tuple(seq)
                suf = suffix_sig(seq)
                if tup in seen_seq or suf in seen_suffix:
                    continue
                seen_seq.add(tup)
                seen_suffix.add(suf)
                next_beam.append((seq, rem, cost))

            if not next_beam:
                # Fallback: keep best unique expansions
                seen = set()
                tmp = []
                for seq, rem, cost, _, _ in sorted_by_cost:
                    tup = tuple(seq)
                    if tup in seen:
                        continue
                    seen.add(tup)
                    tmp.append((seq, rem, cost))
                    if len(tmp) >= local_beam_width:
                        break
                next_beam = tmp if tmp else [(seq, rem, cost) for seq, rem, cost, _, _ in expansions[:local_beam_width]]

            beam = next_beam

        # Choose the best complete sequence from the beam
        best_seq = None
        best_cost = float('inf')
        for seq, rem, cost in beam:
            if rem:
                # Should not happen if loop completes cleanly, but guard anyway
                seq_complete = seq[:]
                # Append remaining arbitrarily (deterministic order) then evaluate
                append_rest = sorted(list(rem))
                seq_complete.extend(append_rest)
                cost = seq_cost(seq_complete)
                if cost < best_cost:
                    best_cost = cost
                    best_seq = seq_complete
            else:
                if cost < best_cost:
                    best_cost = cost
                    best_seq = seq

        return best_seq

    # Local refinement: Or-opt (1,2,3), adjacent swaps, sampled pair swaps, segment reversals
    def local_refine(seq):
        best_seq = seq[:]
        best_cost = seq_cost(best_seq)

        def try_adjacent_swap(cur_seq, cur_cost):
            for i in range(n - 1):
                cand = cur_seq[:]
                cand[i], cand[i + 1] = cand[i + 1], cand[i]
                c = seq_cost(cand)
                if c < cur_cost:
                    return cand, c, True
            return cur_seq, cur_cost, False

        def try_oropt(cur_seq, cur_cost, block_size):
            L = len(cur_seq)
            for i in range(0, L - block_size + 1):
                block = cur_seq[i:i + block_size]
                base = cur_seq[:i] + cur_seq[i + block_size:]
                positions = position_samples(len(base))
                for p in positions:
                    cand = base[:p] + block + base[p:]
                    c = seq_cost(cand)
                    if c < cur_cost:
                        return cand, c, True
            return cur_seq, cur_cost, False

        def try_pair_swaps(cur_seq, cur_cost):
            # Sampled non-adjacent pair swaps
            samples = min(300, max(60, 3 * n))
            for _ in range(samples):
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)
                if i == j:
                    continue
                if i > j:
                    i, j = j, i
                if j == i + 1:
                    continue  # handled by adjacent swaps
                cand = cur_seq[:]
                cand[i], cand[j] = cand[j], cand[i]
                c = seq_cost(cand)
                if c < cur_cost:
                    return cand, c, True
            return cur_seq, cur_cost, False

        def try_segment_reverse(cur_seq, cur_cost):
            # Reverse a random segment (2-opt style)
            samples = min(150, max(40, 2 * n))
            for _ in range(samples):
                i = random.randint(0, n - 2)
                j = random.randint(i + 1, n - 1)
                if j == i + 1:
                    continue
                cand = cur_seq[:i] + cur_seq[i:j + 1][::-1] + cur_seq[j + 1:]
                c = seq_cost(cand)
                if c < cur_cost:
                    return cand, c, True
            return cur_seq, cur_cost, False

        improved = True
        while improved:
            improved = False

            # Or-opt blocks: sizes 1, 2, 3
            for k in (1, 2, 3):
                best_seq, best_cost, did = try_oropt(best_seq, best_cost, k)
                if did:
                    improved = True
                    break
            if improved:
                continue

            # Adjacent swaps
            best_seq, best_cost, did = try_adjacent_swap(best_seq, best_cost)
            if did:
                improved = True
                continue

            # Sampled non-adjacent swaps
            best_seq, best_cost, did = try_pair_swaps(best_seq, best_cost)
            if did:
                improved = True
                continue

            # Segment reversals
            best_seq, best_cost, did = try_segment_reverse(best_seq, best_cost)
            if did:
                improved = True
                continue

        return best_cost, best_seq

    # LNS destroy-and-repair: remove a subset and rebuild via regret-guided insertion
    def lns_attempt(seq):
        cur = seq[:]
        frac = random.uniform(*destroy_frac_range)
        m = max(4, min(n // 2, int(frac * n)))
        # Choose removal indices: random subset or contiguous block
        if random.random() < 0.5:
            remove_idxs = sorted(random.sample(range(n), m))
        else:
            start = random.randint(0, n - m)
            remove_idxs = list(range(start, start + m))
        remove_set = set(remove_idxs)
        removed = [cur[i] for i in remove_idxs]
        remaining = [cur[i] for i in range(n) if i not in remove_set]

        seq_rep = remaining[:]
        rem_set = removed[:]
        while rem_set:
            if len(rem_set) > k_txn_sample:
                cand_txns = random.sample(rem_set, k_txn_sample)
            else:
                cand_txns = rem_set[:]
            best_overall = (float('inf'), None, None)  # cost, txn, pos
            best_by_regret = (float('-inf'), None, None)  # regret, txn, pos

            pos_list = position_samples(len(seq_rep))
            for t in cand_txns:
                best_c = float('inf')
                second_c = float('inf')
                best_p = 0
                for p in pos_list:
                    c = seq_cost(seq_rep[:p] + [t] + seq_rep[p:])
                    if c < best_c:
                        second_c = best_c
                        best_c = c
                        best_p = p
                    elif c < second_c:
                        second_c = c
                regret = (second_c - best_c) if second_c < float('inf') else 0.0
                if best_c < best_overall[0]:
                    best_overall = (best_c, t, best_p)
                if regret > best_by_regret[0]:
                    best_by_regret = (regret, t, best_p)

            pick_regret = (random.random() < lns_regret_prob)
            chosen = best_by_regret if pick_regret and best_by_regret[1] is not None else best_overall
            t = chosen[1]
            p = chosen[2] if chosen[2] is not None else len(seq_rep)
            if t is None:
                t = random.choice(rem_set)
                p = len(seq_rep)
            seq_rep = seq_rep[:p] + [t] + seq_rep[p:]
            rem_set.remove(t)

        c_rep, s_rep = local_refine(seq_rep)
        return c_rep, s_rep

    # Seed selection: evaluate all singletons, take elite + some random seeds
    singleton_scores = []
    for t in all_txns:
        singleton_scores.append((seq_cost([t]), t))
    singleton_scores.sort(key=lambda x: x[0])

    seed_txns = [t for _, t in singleton_scores[:elite_seeds]]
    # Add random distinct seeds
    remaining = [t for t in all_txns if t not in seed_txns]
    if remaining and random_seeds > 0:
        extra = random.sample(remaining, min(random_seeds, len(remaining)))
        seed_txns.extend(extra)

    # Build schedules from seeds and keep the best
    best_overall_cost = float('inf')
    best_overall_seq = None

    for seed in seed_txns:
        seq0 = build_from_seed(seed)
        # Local refinement
        c1, s1 = local_refine(seq0)
        if c1 < best_overall_cost:
            best_overall_cost = c1
            best_overall_seq = s1

    # Iterated local search: structured perturbations and re-refinement
    if best_overall_seq is None:
        # Fallback: random permutation if everything failed
        best_overall_seq = all_txns[:]
        random.shuffle(best_overall_seq)
        best_overall_cost = seq_cost(best_overall_seq)

    for _ in range(perturbations):
        pert = best_overall_seq[:]
        # Apply a random segment reversal to escape local minima
        if n > 3:
            i = random.randint(0, n - 2)
            j = random.randint(i + 1, n - 1)
            pert = pert[:i] + pert[i:j + 1][::-1] + pert[j + 1:]
        # Plus a few random swaps
        for _trial in range(random_swap_trials):
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i != j:
                pert[i], pert[j] = pert[j], pert[i]
        c2, s2 = local_refine(pert)
        if c2 < best_overall_cost:
            best_overall_cost = c2
            best_overall_seq = s2

    # LNS destroy-and-repair attempts to further improve the best solution
    for _ in range(lns_iters):
        c3, s3 = lns_attempt(best_overall_seq)
        if c3 < best_overall_cost:
            best_overall_cost = c3
            best_overall_seq = s3

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