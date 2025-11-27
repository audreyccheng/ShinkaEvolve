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
    Hybrid Beam + GRASP constructor with Iterated Local Search refinement.

    Args:
        workload: Workload object containing transaction data
        num_seqs: Number of random restarts / constructions to attempt

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """

    # Parameters
    N = workload.num_txns
    # Construction parameters (adaptive with N)
    beam_width = min(48, max(8, N // 6))
    branch_factor = min(32, max(8, N // 8))
    lookahead_prob = 0.6
    lookahead_k = 4

    grasp_top_k = 6
    # Fraction of remaining txns to consider per step (capped by branch_factor)
    greedy_sample_frac = 0.5

    # Local improvement parameters
    swap_passes = 2
    rand_swap_trials = 120
    insertion_trials = 60

    # Time budget per workload (seconds)
    time_budget_sec = 0.35

    # Multiple constructions/restarts
    restarts = max(1, int(num_seqs))

    start_time = time.time()

    # Shared memoized cost cache across all phases in this workload call
    cost_cache = {}

    def time_left():
        return (time.time() - start_time) < time_budget_sec

    def eval_cost(seq):
        key = tuple(seq)
        cached = cost_cache.get(key)
        if cached is not None:
            return cached
        c = workload.get_opt_seq_cost(seq)
        cost_cache[key] = c
        return c

    # Beam search with shallow lookahead to reduce myopia
    def run_beam_search():
        all_txns = list(range(N))
        # Seed beam with a random subset of singletons evaluated by cost
        init_pool_size = min(len(all_txns), max(beam_width * 2, 8))
        init_candidates = random.sample(all_txns, init_pool_size) if init_pool_size > 0 else all_txns[:]
        beam = []
        for t in init_candidates:
            if not time_left():
                break
            seq = [t]
            rem = set(all_txns)
            rem.remove(t)
            c = eval_cost(seq)
            beam.append((c, seq, rem))
        if not beam:
            # Fallback: trivial order
            return eval_cost(all_txns), all_txns

        beam.sort(key=lambda x: x[0])
        beam = beam[:min(beam_width, len(beam))]

        remaining_steps = N - 1
        for step in range(remaining_steps):
            if not time_left():
                break
            new_beam = []
            for cost_so_far, seq, rem in beam:
                if not rem:
                    new_beam.append((cost_so_far, seq, rem))
                    continue

                rem_list = list(rem)
                # Adaptive branching: larger early, smaller later
                max_cands = min(branch_factor, len(rem_list))
                if len(rem_list) > max_cands:
                    candidates = random.sample(rem_list, max_cands)
                else:
                    candidates = rem_list

                for cand in candidates:
                    if not time_left():
                        break
                    new_seq = seq + [cand]
                    new_rem = rem.copy()
                    new_rem.remove(cand)
                    base_cost = eval_cost(new_seq)

                    # Shallow lookahead: evaluate best next among a small random sample
                    lookahead_eval = base_cost
                    if new_rem and random.random() < lookahead_prob:
                        la_pool = list(new_rem)
                        la_k = min(lookahead_k, len(la_pool))
                        la_sample = random.sample(la_pool, la_k)
                        best_la = float('inf')
                        for nxt in la_sample:
                            if not time_left():
                                break
                            la_cost = eval_cost(new_seq + [nxt])
                            if la_cost < best_la:
                                best_la = la_cost
                        lookahead_eval = best_la

                    # We score by (lookahead_eval) but carry forward base_cost
                    new_beam.append((lookahead_eval, new_seq, new_rem, base_cost))

            if not new_beam:
                break

            # Keep only the top beam_width unique prefixes by lookahead-evaluated cost
            new_beam.sort(key=lambda x: x[0])
            unique = []
            seen = set()
            for entry in new_beam:
                key = tuple(entry[1])
                if key in seen:
                    continue
                seen.add(key)
                # Store actual base_cost in the kept tuple to continue expansion
                actual_cost = entry[3] if len(entry) > 3 else entry[0]
                unique.append((actual_cost, entry[1], entry[2]))
                if len(unique) >= beam_width:
                    break
            beam = unique

        # Complete sequences in beam may not be full due to time budget; finish greedily if needed
        best_cost = float('inf')
        best_seq = None
        for cost_so_far, seq, rem in beam:
            if not time_left():
                break
            final_seq, final_cost = seq, cost_so_far
            if rem:
                # Greedy finish by minimal marginal addition among remaining
                rem_set = set(rem)
                while rem_set and time_left():
                    rem_list = list(rem_set)
                    # Sampled set for speed near the end
                    max_cands = min(branch_factor, len(rem_list))
                    if len(rem_list) > max_cands:
                        candidates = random.sample(rem_list, max_cands)
                    else:
                        candidates = rem_list
                    best_c = float('inf')
                    best_t = None
                    for t in candidates:
                        c = eval_cost(final_seq + [t])
                        if c < best_c:
                            best_c = c
                            best_t = t
                    if best_t is None:
                        # fallback pick
                        best_t = candidates[0]
                        best_c = eval_cost(final_seq + [best_t])
                    final_seq = final_seq + [best_t]
                    rem_set.remove(best_t)
                    final_cost = best_c
            if final_cost < best_cost:
                best_cost, best_seq = final_cost, final_seq

        # Fallback if beam empty
        if best_seq is None:
            best_seq = all_txns[:]
            random.shuffle(best_seq)
            best_cost = eval_cost(best_seq)
        return best_cost, best_seq

    # GRASP constructive heuristic
    def run_grasp():
        all_txns = list(range(N))
        seq = []
        rem = set(all_txns)
        current_cost = None  # we use absolute cost of prefix; delta computed via new_cost - current_cost

        # Starting point: pick a good seed by sampling a few singletons
        seed_pool = random.sample(all_txns, min(12, len(all_txns)))
        seed_best = None
        seed_best_cost = float('inf')
        for t in seed_pool:
            if not time_left():
                break
            c = eval_cost([t])
            if c < seed_best_cost:
                seed_best_cost = c
                seed_best = t
        if seed_best is None:
            # fallback
            seed_best = random.choice(all_txns)
            seed_best_cost = eval_cost([seed_best])

        seq.append(seed_best)
        rem.remove(seed_best)
        current_cost = seed_best_cost

        while rem and time_left():
            rem_list = list(rem)
            # Adaptive candidate sampling
            dynamic_branch = min(
                branch_factor,
                max(6, int(len(rem_list) * greedy_sample_frac))
            )
            if len(rem_list) > dynamic_branch:
                candidates = random.sample(rem_list, dynamic_branch)
            else:
                candidates = rem_list

            scored = []
            for t in candidates:
                if not time_left():
                    break
                c = eval_cost(seq + [t])
                scored.append((c, t))
            if not scored:
                # pick arbitrary if time ran out
                t = candidates[0] if candidates else random.choice(list(rem))
                seq.append(t)
                rem.remove(t)
                current_cost = eval_cost(seq)
                continue

            scored.sort(key=lambda x: x[0])
            top_k = min(grasp_top_k, len(scored))
            chosen_cost, chosen_t = random.choice(scored[:top_k])
            seq.append(chosen_t)
            rem.remove(chosen_t)
            current_cost = chosen_cost

        # If time ended with remaining, append arbitrarily
        if rem:
            seq.extend(list(rem))
        final_cost = eval_cost(seq)
        return final_cost, seq

    # Local improvement: adjacent swaps, random swaps, and sampled insertion moves
    def local_improve(seq, current_cost):
        best_seq = seq[:]
        best_cost = current_cost

        # Adjacent swap passes
        for _ in range(swap_passes):
            improved = False
            for i in range(len(best_seq) - 1):
                if not time_left():
                    break
                cand = best_seq[:]
                cand[i], cand[i + 1] = cand[i + 1], cand[i]
                c = eval_cost(cand)
                if c < best_cost:
                    best_cost = c
                    best_seq = cand
                    improved = True
            if not improved or not time_left():
                break

        # Random two-swap hill climbing
        trials = rand_swap_trials
        while trials > 0 and time_left():
            trials -= 1
            i, j = random.sample(range(len(best_seq)), 2)
            if i == j:
                continue
            if abs(i - j) <= 1:
                continue
            cand = best_seq[:]
            cand[i], cand[j] = cand[j], cand[i]
            c = eval_cost(cand)
            if c < best_cost:
                best_cost = c
                best_seq = cand

        # Random insertion moves
        trials = insertion_trials
        while trials > 0 and time_left():
            trials -= 1
            i, j = random.sample(range(len(best_seq)), 2)
            if i == j:
                continue
            cand = best_seq[:]
            val = cand.pop(i)
            cand.insert(j, val)
            c = eval_cost(cand)
            if c < best_cost:
                best_cost = c
                best_seq = cand

        return best_cost, best_seq

    global_best_cost = float('inf')
    global_best_seq = None

    # Alternate between beam and GRASP constructions across restarts
    for r in range(restarts):
        if not time_left():
            break
        if r % 2 == 0:
            cost, seq = run_beam_search()
        else:
            cost, seq = run_grasp()

        # Local improvement
        if time_left():
            cost, seq = local_improve(seq, cost)

        if cost < global_best_cost:
            global_best_cost, global_best_seq = cost, seq

    # Fallback if nothing computed
    if global_best_seq is None:
        global_best_seq = list(range(N))
        random.shuffle(global_best_seq)
        global_best_cost = eval_cost(global_best_seq)

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