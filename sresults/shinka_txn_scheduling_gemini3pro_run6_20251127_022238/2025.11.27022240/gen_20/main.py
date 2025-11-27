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
    Get optimal schedule using Work-Density Beam Search with Ends-Out Sampling.

    Strategy:
    1. Beam Search prioritizing 'Work Density' (Cost - TotalDuration).
       This heuristic favors schedules that pack more work into the same makespan.
    2. Ends-Out Sampling: Samples LPT and SPT items to manage critical paths and gaps.
    3. State Deduplication to maximize beam diversity.

    Args:
        workload: Workload object containing transaction data
        num_seqs: Used to scale the beam width

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    # --- Beam Search Parameters ---
    # Increased width to allow the density heuristic to work (it benefits from looking ahead)
    BEAM_WIDTH = int(max(12, num_seqs * 2.0))
    SAMPLES_PER_NODE = 24

    num_txns = workload.num_txns

    # --- Precompute Heuristics ---
    txn_durations = {t: workload.get_opt_seq_cost([t]) for t in range(num_txns)}

    # Sorted lists for Ends-Out Sampling
    sorted_desc = sorted(range(num_txns), key=lambda t: txn_durations[t], reverse=True) # Longest first
    sorted_asc = sorted(range(num_txns), key=lambda t: txn_durations[t])  # Shortest first

    # --- Initialization ---
    # Seed beam with longest txns (bottlenecks) and randoms
    start_candidates = set()
    start_candidates.update(sorted_desc[:BEAM_WIDTH])

    # Fill with random if needed
    while len(start_candidates) < BEAM_WIDTH * 2:
        start_candidates.add(random.randint(0, num_txns - 1))

    beam = []
    for t in start_candidates:
        rem = set(range(num_txns))
        rem.remove(t)
        beam.append({
            'cost': txn_durations[t],
            'total_dur': txn_durations[t],
            'seq': [t],
            'rem': rem
        })

    # Prune initial beam
    beam.sort(key=lambda x: (x['cost'] - x['total_dur'], -x['total_dur']))
    beam = beam[:BEAM_WIDTH]

    # --- Main Beam Search Loop ---
    for _ in range(num_txns - 1):
        candidates = []

        for parent in beam:
            rem_list = list(parent['rem'])
            rem_set = parent['rem']

            if len(rem_list) <= SAMPLES_PER_NODE:
                samples = rem_list
            else:
                samples = set()

                # 1. Longest Remaining (LPT)
                count = 0
                for t in sorted_desc:
                    if t in rem_set:
                        samples.add(t)
                        count += 1
                        if count >= 3: break

                # 2. Shortest Remaining (SPT)
                count = 0
                for t in sorted_asc:
                    if t in rem_set:
                        samples.add(t)
                        count += 1
                        if count >= 2: break

                # 3. Random Exploration
                needed = SAMPLES_PER_NODE - len(samples)
                if needed > 0:
                    pool = [t for t in rem_list if t not in samples]
                    if len(pool) <= needed:
                        samples.update(pool)
                    else:
                        samples.update(random.sample(pool, needed))

            # --- Evaluation ---
            for t in samples:
                new_seq = parent['seq'] + [t]
                new_cost = workload.get_opt_seq_cost(new_seq)
                new_total_dur = parent['total_dur'] + txn_durations[t]

                # Priority: Work Density
                # Score = Makespan - SerialDuration.
                # Minimizing this favors schedules that have completed more work
                # (high SerialDuration) relative to their makespan.
                score = new_cost - new_total_dur

                candidates.append({
                    'priority': (score, new_cost),
                    'cost': new_cost,
                    'total_dur': new_total_dur,
                    'seq': new_seq,
                    'rem': parent['rem'],
                    'added': t
                })

        # --- Selection with Deduplication ---
        candidates.sort(key=lambda x: x['priority'])

        new_beam = []
        seen_states = set()

        for cand in candidates:
            if len(new_beam) >= BEAM_WIDTH:
                break

            new_rem = cand['rem'].copy()
            new_rem.remove(cand['added'])

            # State Identity: Sorted tuple of remaining transaction IDs
            state_key = tuple(sorted(list(new_rem)))

            if state_key not in seen_states:
                seen_states.add(state_key)
                new_beam.append({
                    'cost': cand['cost'],
                    'total_dur': cand['total_dur'],
                    'seq': cand['seq'],
                    'rem': new_rem
                })

        if not new_beam:
            break
        beam = new_beam

    # --- Best Schedule Selection ---
    best_result = min(beam, key=lambda x: x['cost'])
    best_cost = best_result['cost']
    best_seq = best_result['seq']

    # --- Hybrid Local Search Refinement ---

    # 1. Fast Adjacent Swaps
    for _ in range(3):
        improved = False
        for i in range(num_txns - 1):
            best_seq[i], best_seq[i+1] = best_seq[i+1], best_seq[i]
            c = workload.get_opt_seq_cost(best_seq)
            if c < best_cost:
                best_cost = c
                improved = True
            else:
                best_seq[i], best_seq[i+1] = best_seq[i+1], best_seq[i]
        if not improved:
            break

    # 2. Windowed Insertion (Reduced window for speed)
    window = 5
    for i in range(num_txns):
        curr_txn = best_seq[i]
        start = max(0, i - window)
        end = min(num_txns, i + window)

        for j in range(start, end):
            if i == j: continue

            if i < j:
                cand_seq = best_seq[:i] + best_seq[i+1:j+1] + [curr_txn] + best_seq[j+1:]
            else:
                cand_seq = best_seq[:j] + [curr_txn] + best_seq[j:i] + best_seq[i+1:]

            c = workload.get_opt_seq_cost(cand_seq)
            if c < best_cost:
                best_cost = c
                best_seq = cand_seq
                break

    return best_cost, best_seq


def get_random_costs():
    """
    Evaluate scheduling algorithm on three different workloads.

    Returns:
        Tuple of (total_makespan, list_of_schedules, execution_time)
    """
    start_time = time.time()

    # Workload 1: Complex mixed read/write transactions
    workload1 = Workload(WORKLOAD_1)
    makespan1, schedule1 = get_best_schedule(workload1, 10)

    # Workload 2: Simple read-then-write pattern
    workload2 = Workload(WORKLOAD_2)
    makespan2, schedule2 = get_best_schedule(workload2, 10)

    # Workload 3: Minimal read/write operations
    workload3 = Workload(WORKLOAD_3)
    makespan3, schedule3 = get_best_schedule(workload3, 10)

    total_makespan = makespan1 + makespan2 + makespan3
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