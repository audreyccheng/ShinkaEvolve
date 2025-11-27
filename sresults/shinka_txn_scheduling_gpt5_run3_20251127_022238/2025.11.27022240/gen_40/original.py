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
    GRASP + micro-beam hybrid with memoized marginal-cost evaluation:
    - Randomized greedy construction using true extension costs (RCL selection)
    - Pair- and singleton-seeded multi-starts
    - Shared caches for partial and extension costs
    - Lightweight local refinement (adjacent swaps, segment reverse, insertions)
    """

    N = workload.num_txns
    start_time = time.time()
    # Keep budget conservative to improve combined score (quality vs runtime)
    base_budget = 0.36
    # Allow a tiny bump with N to stabilize for larger workloads
    time_budget_sec = base_budget + min(0.10, 0.0009 * max(0, N))

    def time_left():
        return (time.time() - start_time) < time_budget_sec

    # Shared caches
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
            c = eval_seq_cost(list(prefix_tuple) + [cand])
            ext_cache[key] = c
        return c

    all_txns = list(range(N))

    # Precompute singleton costs to seed and guide
    singleton_cost = {}
    for t in all_txns:
        if not time_left():
            break
        singleton_cost[t] = eval_seq_cost([t])

    # Helper: greedy completion from a prefix using RCL on true extension costs
    def greedy_from_prefix(prefix):
        seq = list(prefix)
        rem = set(x for x in all_txns if x not in seq)
        # current cost of prefix
        cur_cost = eval_seq_cost(seq) if seq else 0

        # Adaptive sampling parameters
        while rem and time_left():
            rem_list = list(rem)
            # Candidate sampling to cap compute
            # Slightly larger at start, smaller later
            depth = len(seq)
            if len(rem_list) > 0:
                if depth < N // 3:
                    k_pool = min(len(rem_list), max(10, N // 8))
                elif depth < (2 * N) // 3:
                    k_pool = min(len(rem_list), max(8, N // 10))
                else:
                    k_pool = min(len(rem_list), max(6, N // 12))
            else:
                k_pool = 0

            cand_pool = rem_list if len(rem_list) <= k_pool else random.sample(rem_list, k_pool)

            # Evaluate extension costs
            prefix_tuple = tuple(seq)
            scored = []
            for t in cand_pool:
                if not time_left():
                    break
                ec = eval_ext_cost(prefix_tuple, t)
                scored.append((ec, t))
            if not scored:
                # time expired; append arbitrary order
                seq.extend(rem)
                rem.clear()
                break

            scored.sort(key=lambda x: x[0])

            # RCL selection among top fraction
            # Wider RCL early, narrower later
            if depth < N // 3:
                rcl_frac = 0.35
            elif depth < (2 * N) // 3:
                rcl_frac = 0.25
            else:
                rcl_frac = 0.18

            rcl_size = max(1, int(rcl_frac * len(scored)))
            pick_idx = random.randrange(rcl_size)
            chosen_cost, chosen_t = scored[pick_idx]

            seq.append(chosen_t)
            rem.remove(chosen_t)
            cur_cost = chosen_cost

        # If time ran out mid-way, ensure full sequence
        if rem:
            for t in list(rem):
                seq.append(t)
            cur_cost = eval_seq_cost(seq)

        return cur_cost, seq

    # Create seeds: top singletons and a few strong pairs
    def build_seeds():
        seeds = []

        # Top-k singletons by cost
        singles_sorted = sorted(all_txns, key=lambda t: singleton_cost.get(t, float('inf')))
        top_k_single = singles_sorted[:min(8, max(4, N // 12))]
        for t in top_k_single:
            seeds.append([t])

        # Random singleton seeds for diversity
        rand_k = min(6, max(3, N // 20))
        if len(all_txns) > 0:
            seeds.extend([[t] for t in random.sample(all_txns, min(len(all_txns), rand_k))])

        # Best pairs from a small candidate set
        pair_candidates = singles_sorted[:min(16, max(6, N // 8))] if singles_sorted else []
        eval_pairs = []
        for i in range(len(pair_candidates)):
            if not time_left():
                break
            for j in range(len(pair_candidates)):
                if i == j or not time_left():
                    continue
                a = pair_candidates[i]
                b = pair_candidates[j]
                # Evaluate [a,b] extension cost
                ec = eval_seq_cost([a, b])
                eval_pairs.append((ec, [a, b]))
                # Keep this pool bounded
                if len(eval_pairs) >= 80:
                    break
            if len(eval_pairs) >= 80:
                break
        eval_pairs.sort(key=lambda x: x[0])
        seeds.extend([p for _c, p in eval_pairs[:min(8, len(eval_pairs))]])

        return seeds

    # Local refinement: adjacent swaps, 2-opt style reverse, and insertions
    def local_improve(seq, cost):
        best_seq = seq[:]
        best_cost = cost

        # Pass 1: adjacent swaps
        for _ in range(2):
            improved = False
            for i in range(len(best_seq) - 1):
                if not time_left():
                    break
                cand = best_seq[:]
                cand[i], cand[i + 1] = cand[i + 1], cand[i]
                c = eval_seq_cost(cand)
                if c < best_cost:
                    best_cost = c
                    best_seq = cand
                    improved = True
            if not improved or not time_left():
                break

        # Pass 2: limited 2-opt segment reversals
        trials = 25
        while trials > 0 and time_left():
            trials -= 1
            i = random.randrange(len(best_seq))
            j = random.randrange(len(best_seq))
            if abs(i - j) <= 1:
                continue
            if i > j:
                i, j = j, i
            cand = best_seq[:]
            cand[i:j+1] = reversed(cand[i:j+1])
            c = eval_seq_cost(cand)
            if c < best_cost:
                best_cost = c
                best_seq = cand

        # Pass 3: random insertions
        trials = 40
        while trials > 0 and time_left():
            trials -= 1
            i = random.randrange(len(best_seq))
            j = random.randrange(len(best_seq))
            if i == j:
                continue
            cand = best_seq[:]
            val = cand.pop(i)
            cand.insert(j, val)
            c = eval_seq_cost(cand)
            if c < best_cost:
                best_cost = c
                best_seq = cand

        return best_cost, best_seq

    # Micro-beam around a prefix: try a handful of best single-step choices and greedily finish
    def micro_beam(prefix, rem_set, beam_w=6, branch=10):
        if not rem_set:
            c = eval_seq_cost(prefix)
            return c, list(prefix)
        # Score candidates by extension cost
        prefix_tuple = tuple(prefix)
        rem_list = list(rem_set)
        if len(rem_list) > branch:
            cand_pool = random.sample(rem_list, branch)
        else:
            cand_pool = rem_list

        scored = []
        for t in cand_pool:
            if not time_left():
                break
            ec = eval_ext_cost(prefix_tuple, t)
            scored.append((ec, t))
        if not scored:
            # time ran out
            seq = list(prefix) + list(rem_set)
            return eval_seq_cost(seq), seq

        scored.sort(key=lambda x: x[0])
        top = scored[:min(beam_w, len(scored))]

        best_c = float('inf')
        best_s = None
        for ec, t in top:
            if not time_left():
                break
            new_prefix = list(prefix) + [t]
            new_rem = set(rem_set)
            new_rem.remove(t)
            c_try, s_try = greedy_from_prefix(new_prefix)
            if c_try < best_c:
                best_c, best_s = c_try, s_try
        if best_s is None:
            seq = list(prefix) + list(rem_set)
            return eval_seq_cost(seq), seq
        return best_c, best_s

    # Main multi-start GRASP with time bound
    global_best_cost = float('inf')
    global_best_seq = None

    seeds = build_seeds()
    # Ensure at least one seed
    if not seeds:
        seeds = [[]]

    # Number of restarts bounded by num_seqs and time
    max_restarts = max(6, int(num_seqs))
    r = 0
    while r < max_restarts and time_left():
        # Choose a seed
        if r < len(seeds):
            prefix = seeds[r]
        else:
            # Random short prefix for diversity
            k = random.randint(0, min(2, N))
            prefix = random.sample(all_txns, k) if k > 0 else []

        # Build greedily from prefix
        c, s = greedy_from_prefix(prefix)

        # Micro-beam intensification starting from the first few positions of s
        if time_left():
            # Use first m positions as a strong prefix
            m = min(2, len(s))
            c2, s2 = micro_beam(s[:m], set(x for x in all_txns if x not in s[:m]), beam_w=5, branch=min(10, max(6, N // 10)))
            if c2 < c:
                c, s = c2, s2

        # Local search refinement
        if time_left():
            c, s = local_improve(s, c)

        if c < global_best_cost:
            global_best_cost, global_best_seq = c, s

        r += 1

    if global_best_seq is None:
        global_best_seq = list(range(N))
        random.shuffle(global_best_seq)
        global_best_cost = eval_seq_cost(global_best_seq)

    # Final assert: permutation
    # (avoid raising in production; trust greedy to generate permutation)
    # but ensure we return a valid sequence of all txns
    if len(global_best_seq) != N or len(set(global_best_seq)) != N:
        # As a safety fallback, repair: create a permutation
        seen = set()
        repaired = []
        for t in global_best_seq:
            if t not in seen and 0 <= t < N:
                repaired.append(t)
                seen.add(t)
        for t in all_txns:
            if t not in seen:
                repaired.append(t)
        global_best_seq = repaired[:N]
        global_best_cost = eval_seq_cost(global_best_seq)

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