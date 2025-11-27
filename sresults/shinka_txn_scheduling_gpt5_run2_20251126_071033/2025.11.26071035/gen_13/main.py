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
    Minimize makespan using regret-guided insertion beam search (with diversity
    and high-regret quotas), strong VND local search with DLB, ruin-and-recreate
    LNS, and light path relinking.

    Args:
        workload: Workload object containing transaction data
        num_seqs: Controls search breadth (seeds, beam width, sampling sizes)

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    n = workload.num_txns
    all_txns = list(range(n))

    # -------------------------
    # Cost memoization
    # -------------------------
    cost_cache = {}
    def seq_cost(seq):
        key = tuple(seq)
        if key in cost_cache:
            return cost_cache[key]
        c = workload.get_opt_seq_cost(seq)
        cost_cache[key] = c
        return c

    # -------------------------
    # Parameters (tunable)
    # -------------------------
    # Seeds and GRASP
    elite_seeds = max(2, min(6, int(num_seqs)))
    random_seeds = max(1, min(3, int(num_seqs) // 4))
    grasp_seeds = 2
    grasp_alpha = 0.30  # pick randomly among top alpha fraction of candidates

    # Beam search
    beam_width_base = max(4, min(6, int(num_seqs)))
    regret_quota_frac = 0.30  # fraction of beam reserved for high-regret choices
    diversity_suffix_len = 2  # signature length for diversity
    rem_all_threshold = 12
    k_txn_sample = min(14, max(8, 2 + int(num_seqs)))
    k_pos_sample = 6
    endgame_widen_factor = 2  # widen beam by +2 when few remain
    endgame_threshold_factor = 2  # when rem <= 2*beam width, intensify

    # Local search (VND) and DLB
    dlb_max_noimprove = 2  # skip indices with no improvement for this many consecutive passes
    pair_swap_samples = lambda n: min(500, max(80, 5 * n))
    seg_reverse_samples = lambda n: min(250, max(60, 3 * n))

    # Ruin & Recreate LNS
    lns_rounds = min(3, 1 + n // 60)
    k_remove_frac = 0.08  # 8% remove
    k_remove_bounds = (8, 15)
    small_rebuild_beam = 3  # small beam for reinserting removed items

    # Iterated local search perturbations (light)
    ils_perturbations = max(2, min(4, int(num_seqs) // 4))
    random_swap_trials = 3

    # -------------------------
    # Helpers: positions and regret evaluation
    # -------------------------
    def position_samples(seq_len, intensify=False):
        # Always include ends and center; add random internal positions
        if seq_len <= 1:
            return [0, seq_len]
        base = {0, seq_len, seq_len // 2, max(0, seq_len // 3), min(seq_len, (2 * seq_len) // 3)}
        extra = k_pos_sample + (2 if intensify else 0)
        for _ in range(min(extra, seq_len + 1)):
            base.add(random.randint(0, seq_len))
        return sorted(base)

    def best_two_insertion(seq, t, positions):
        best_cost = float('inf')
        second_best = float('inf')
        best_seq = None
        # Evaluate inserting t at each candidate position
        for p in positions:
            new_seq = seq[:p] + [t] + seq[p:]
            c = seq_cost(new_seq)
            if c < best_cost:
                second_best = best_cost
                best_cost = c
                best_seq = new_seq
            elif c < second_best:
                second_best = c
        return best_seq, best_cost, second_best

    def suffix_signature(seq):
        L = len(seq)
        if L == 0:
            return ()
        if L == 1 or diversity_suffix_len == 1:
            return (seq[-1],)
        return tuple(seq[max(0, L - diversity_suffix_len):])

    # -------------------------
    # Beam expansion from a partial state
    # -------------------------
    def expand_beam(beam_states):
        # beam_states: list of (seq, rem_set)
        expansions = []
        seen_seqs = set()  # exact sequence duplicates
        for seq, rem in beam_states:
            if not rem:
                key = tuple(seq)
                if key in seen_seqs:
                    continue
                seen_seqs.add(key)
                expansions.append((seq, rem, seq_cost(seq), 0.0, None))
                continue

            rem_list = list(rem)
            # Adaptive intensification
            beam_width = beam_width_base
            if len(rem) <= endgame_threshold_factor * beam_width_base:
                beam_width = beam_width_base + endgame_widen_factor
            if len(rem_list) <= rem_all_threshold:
                cand_txns = rem_list
                intensify = True
            else:
                cand_txns = random.sample(rem_list, min(k_txn_sample, len(rem_list)))
                intensify = False

            positions = position_samples(len(seq), intensify=intensify)
            for t in cand_txns:
                best_seq, best_c, second_c = best_two_insertion(seq, t, positions)
                regret = (second_c - best_c) if second_c < float('inf') else 0.0
                new_rem = rem.copy()
                new_rem.remove(t)
                key = tuple(best_seq)
                if key in seen_seqs:
                    # keep the better one if duplicate arises
                    continue
                seen_seqs.add(key)
                expansions.append((best_seq, new_rem, best_c, regret, t))

        if not expansions:
            return []

        # Rank primarily by cost; also prepare regret ranking list
        by_cost = sorted(expansions, key=lambda x: x[2])
        by_regret = sorted(expansions, key=lambda x: (-x[3], x[2]))

        # Diversity selection using suffix signature of last-2 elements
        next_beam = []
        seen_signatures = set()
        # beam size adapt
        beam_target = beam_width_base
        # If any state had small remainder, widen a little
        if any(len(rem) <= endgame_threshold_factor * beam_width_base for _, rem in beam_states):
            beam_target = beam_width_base + endgame_widen_factor

        main_quota = max(1, int((1.0 - regret_quota_frac) * beam_target))
        regret_quota = max(1, beam_target - main_quota)

        # Fill by cost first
        for seq, rem, c, r, t in by_cost:
            sig = suffix_signature(seq)
            if sig in seen_signatures:
                continue
            seen_signatures.add(sig)
            next_beam.append((seq, rem))
            if len(next_beam) >= main_quota:
                break

        # Then fill regret quota
        for seq, rem, c, r, t in by_regret:
            if len(next_beam) >= beam_target:
                break
            sig = suffix_signature(seq)
            if sig in seen_signatures:
                continue
            seen_signatures.add(sig)
            next_beam.append((seq, rem))

        # Fallback if empty
        if not next_beam:
            # drop diversity constraint
            for seq, rem, c, r, t in by_cost[:beam_target]:
                next_beam.append((seq, rem))

        return next_beam

    # -------------------------
    # Build full schedule from seed using regret-beam construction
    # -------------------------
    def build_from_seed(seed_t):
        seq0 = [seed_t]
        rem0 = set(all_txns)
        rem0.remove(seed_t)
        beam = [(seq0, rem0)]

        # Optionally add a second element quickly (prepend/append best of samples)
        if rem0:
            cand_txns = list(rem0)
            if len(cand_txns) > k_txn_sample:
                cand_txns = random.sample(cand_txns, k_txn_sample)
            best_seq2 = None
            best_c2 = float('inf')
            best_t2 = None
            for t in cand_txns:
                for variant in ([t] + beam[0][0], beam[0][0] + [t]):
                    c = seq_cost(variant)
                    if c < best_c2:
                        best_c2 = c
                        best_seq2 = variant
                        best_t2 = t
            if best_seq2 is not None:
                rem1 = set(rem0)
                rem1.remove(best_t2)
                beam = [(best_seq2, rem1)]

        # Grow beam until complete
        while True:
            if all(len(rem) == 0 for _, rem in beam):
                break
            beam = expand_beam(beam)
            if not beam:
                break

        # Choose final best by cost
        best_seq = None
        best_cost = float('inf')
        for seq, rem in beam:
            if rem:
                # Append remaining in deterministic order
                seq_complete = seq + sorted(list(rem))
                c = seq_cost(seq_complete)
            else:
                seq_complete = seq
                c = seq_cost(seq_complete)
            if c < best_cost:
                best_cost = c
                best_seq = seq_complete

        return best_seq

    # -------------------------
    # GRASP constructive seed
    # -------------------------
    def grasp_construct():
        seq = []
        rem = set(all_txns)
        while rem:
            # Sample candidate txns
            if len(rem) <= rem_all_threshold:
                cand_txns = list(rem)
                intensify = True
            else:
                cand_txns = random.sample(list(rem), min(k_txn_sample, len(rem)))
                intensify = False

            candidates = []
            positions = position_samples(len(seq), intensify=intensify)
            for t in cand_txns:
                best_seq, best_c, _ = best_two_insertion(seq, t, positions)
                candidates.append((best_c, best_seq, t))
            if not candidates:
                # append any
                t = rem.pop()
                seq = seq + [t]
                continue
            candidates.sort(key=lambda x: x[0])
            top_m = max(1, int(grasp_alpha * len(candidates)))
            chosen_idx = random.randint(0, top_m - 1)
            _, seq, t = candidates[chosen_idx]
            rem.remove(t)
        return seq

    # -------------------------
    # Local refinement: VND + DLB
    # -------------------------
    def local_refine(seq):
        best_seq = seq[:]
        best_cost = seq_cost(best_seq)

        # DLB structure: track indices that recently failed to improve
        dont_look = [0] * n

        def try_or_opt(k):
            nonlocal best_seq, best_cost, dont_look
            L = len(best_seq)
            improved = False
            i = 0
            while i <= L - k:
                if dont_look[i] >= dlb_max_noimprove:
                    i += 1
                    continue
                block = best_seq[i:i + k]
                base = best_seq[:i] + best_seq[i + k:]
                # Generate candidate positions: ends, neighbors, middle, plus a few random
                base_positions = {0, len(base), max(0, i - 1), min(len(base), i + 1), len(base) // 2}
                for _ in range(min(4 + k, len(base) + 1)):
                    base_positions.add(random.randint(0, len(base)))
                improved_here = False
                for p in sorted(base_positions):
                    if p == i:  # same place (after removal) not meaningful
                        continue
                    cand = base[:p] + block + base[p:]
                    c = seq_cost(cand)
                    if c < best_cost:
                        best_cost = c
                        best_seq = cand
                        # Reset DLB around affected region
                        dont_look[max(0, i - 2):min(n, i + k + 3)] = [0] * len(dont_look[max(0, i - 2):min(n, i + k + 3)])
                        improved = True
                        improved_here = True
                        # Restart pass for this k (first-improvement in VND)
                        L = len(best_seq)
                        i = max(0, min(i, L - k))  # adjust index if needed
                        break
                if not improved_here:
                    dont_look[i] += 1
                    i += 1
                else:
                    # restart scanning from beginning for stability
                    i = 0
            return improved

        def try_adjacent_swaps():
            nonlocal best_seq, best_cost, dont_look
            L = len(best_seq)
            for i in range(L - 1):
                cand = best_seq[:]
                cand[i], cand[i + 1] = cand[i + 1], cand[i]
                c = seq_cost(cand)
                if c < best_cost:
                    best_cost = c
                    best_seq = cand
                    dont_look[max(0, i - 2):min(n, i + 3)] = [0] * len(dont_look[max(0, i - 2):min(n, i + 3)])
                    return True
            return False

        def try_pair_swaps_sampled():
            nonlocal best_seq, best_cost
            L = len(best_seq)
            samples = pair_swap_samples(L)
            for _ in range(samples):
                i = random.randint(0, L - 1)
                j = random.randint(0, L - 1)
                if abs(i - j) <= 1:
                    continue
                if i > j:
                    i, j = j, i
                cand = best_seq[:]
                cand[i], cand[j] = cand[j], cand[i]
                c = seq_cost(cand)
                if c < best_cost:
                    best_cost = c
                    best_seq = cand
                    return True
            return False

        def try_segment_reverse_sampled():
            nonlocal best_seq, best_cost
            L = len(best_seq)
            samples = seg_reverse_samples(L)
            for _ in range(samples):
                i = random.randint(0, L - 2)
                j = random.randint(i + 2, L - 1)
                cand = best_seq[:i] + best_seq[i:j + 1][::-1] + best_seq[j + 1:]
                c = seq_cost(cand)
                if c < best_cost:
                    best_cost = c
                    best_seq = cand
                    return True
            return False

        improved_any = True
        while improved_any:
            improved_any = False
            # VND order: Or-opt-1, Or-opt-2, Or-opt-3, Adjacent swaps, Pair swaps, Segment reverse
            if try_or_opt(1):
                improved_any = True
                continue
            if try_or_opt(2):
                improved_any = True
                continue
            if try_or_opt(3):
                improved_any = True
                continue
            if try_adjacent_swaps():
                improved_any = True
                continue
            if try_pair_swaps_sampled():
                improved_any = True
                continue
            if try_segment_reverse_sampled():
                improved_any = True
                continue

        return best_cost, best_seq

    # -------------------------
    # Ruin-and-recreate LNS (regret-guided reinsertion with small beam)
    # -------------------------
    def lns_ruin_recreate(base_seq):
        best_seq = base_seq[:]
        best_cost = seq_cost(best_seq)
        for _ in range(lns_rounds):
            L = len(best_seq)
            k_remove = max(k_remove_bounds[0], min(k_remove_bounds[1], int(k_remove_frac * L)))
            # Build removal set: contiguous chunk + random picks
            remove_idx = set()
            if L > 3:
                start = random.randint(0, L - 2)
                chunk_len = max(2, min(k_remove // 2, L - start))
                for i in range(start, start + chunk_len):
                    remove_idx.add(i)
            while len(remove_idx) < k_remove:
                remove_idx.add(random.randint(0, L - 1))
            remove_idx = sorted(remove_idx)
            removed = [best_seq[i] for i in remove_idx]
            remaining = [best_seq[i] for i in range(L) if i not in remove_idx]

            # Reinsert removed set with small regret-beam
            partial_beam = [(remaining[:], set(removed))]
            def expand_partial_beam(states):
                exps = []
                for seq, remset in states:
                    if not remset:
                        exps.append((seq, remset, seq_cost(seq), 0.0))
                        continue
                    cand_txns = list(remset)
                    positions = position_samples(len(seq), intensify=True)
                    for t in cand_txns:
                        best_seq_ins, best_c, second_c = best_two_insertion(seq, t, positions)
                        regret = (second_c - best_c) if second_c < float('inf') else 0.0
                        new_rem = set(remset)
                        new_rem.remove(t)
                        exps.append((best_seq_ins, new_rem, best_c, regret))
                if not exps:
                    return []
                exps.sort(key=lambda x: (x[2], -x[3]))
                # keep unique by suffix signature
                next_states = []
                seen_sigs = set()
                for seq, rem, c, r in exps:
                    sig = suffix_signature(seq)
                    if sig in seen_sigs:
                        continue
                    seen_sigs.add(sig)
                    next_states.append((seq, rem))
                    if len(next_states) >= small_rebuild_beam:
                        break
                return next_states

            while True:
                if all(len(rem) == 0 for _, rem in partial_beam):
                    break
                partial_beam = expand_partial_beam(partial_beam)
                if not partial_beam:
                    break

            # take best reconstructed
            best_rebuilt = None
            best_rebuilt_cost = float('inf')
            for seq, rem in partial_beam:
                if rem:
                    seq_c = seq + sorted(list(rem))
                else:
                    seq_c = seq
                c = seq_cost(seq_c)
                if c < best_rebuilt_cost:
                    best_rebuilt_cost = c
                    best_rebuilt = seq_c

            if best_rebuilt is None:
                continue

            c_ref, s_ref = local_refine(best_rebuilt)
            if c_ref < best_cost:
                best_cost = c_ref
                best_seq = s_ref
        return best_cost, best_seq

    # -------------------------
    # Path relinking between two sequences (light)
    # -------------------------
    def path_relink(a_seq, b_seq, max_steps=None):
        if max_steps is None:
            max_steps = max(10, n // 8)
        cur = a_seq[:]
        best_c = seq_cost(cur)
        best_s = cur[:]
        pos_in_cur = {v: i for i, v in enumerate(cur)}
        steps = 0
        for i in range(n):
            if steps >= max_steps:
                break
            if cur[i] == b_seq[i]:
                continue
            target = b_seq[i]
            j = pos_in_cur[target]
            # Move target from j to i (reinsertion)
            base = cur[:j] + cur[j + 1:]
            cand = base[:i] + [target] + base[i:]
            c = seq_cost(cand)
            cur = cand
            # rebuild positions
            pos_in_cur = {v: k for k, v in enumerate(cur)}
            steps += 1
            if c < best_c:
                best_c = c
                best_s = cur[:]
        # refine the best found on the path
        c_ref, s_ref = local_refine(best_s)
        return c_ref, s_ref

    # -------------------------
    # Seed selection and construction
    # -------------------------
    singleton_scores = []
    for t in all_txns:
        singleton_scores.append((seq_cost([t]), t))
    singleton_scores.sort(key=lambda x: x[0])

    seed_txns = [t for _, t in singleton_scores[:elite_seeds]]
    # Random distinct seeds
    remaining = [t for t in all_txns if t not in seed_txns]
    if remaining and random_seeds > 0:
        extra = random.sample(remaining, min(random_seeds, len(remaining)))
        seed_txns.extend(extra)

    # Add GRASP seeds (constructed whole sequences, we'll derive their "seed" as first element)
    grasp_sequences = []
    for _ in range(grasp_seeds):
        gs = grasp_construct()
        grasp_sequences.append(gs)

    # Build schedules from seeds
    candidates = []
    for seed in seed_txns:
        seq0 = build_from_seed(seed)
        c1, s1 = local_refine(seq0)
        candidates.append((c1, s1))
    # Include GRASP sequences
    for gs in grasp_sequences:
        c1, s1 = local_refine(gs)
        candidates.append((c1, s1))

    # Keep top-k elites for path relinking
    candidates.sort(key=lambda x: x[0])
    best_overall_cost, best_overall_seq = candidates[0]
    elites_for_pr = [s for _, s in candidates[:2]]  # top-2 for path relinking

    # Path relinking between top-2 if available
    if len(elites_for_pr) >= 2:
        c_pr, s_pr = path_relink(elites_for_pr[0], elites_for_pr[1])
        if c_pr < best_overall_cost:
            best_overall_cost, best_overall_seq = c_pr, s_pr

    # Light ILS perturbations with local refine
    for _ in range(ils_perturbations):
        pert = best_overall_seq[:]
        for _trial in range(random_swap_trials):
            i = random.randint(0, n - 1)
            j = random.randint(0, n - 1)
            if i != j:
                pert[i], pert[j] = pert[j], pert[i]
        c2, s2 = local_refine(pert)
        if c2 < best_overall_cost:
            best_overall_cost = c2
            best_overall_seq = s2

    # Ruin-and-recreate LNS
    c_lns, s_lns = lns_ruin_recreate(best_overall_seq)
    if c_lns < best_overall_cost:
        best_overall_cost, best_overall_seq = c_lns, s_lns

    # Final return
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