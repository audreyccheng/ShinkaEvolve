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
    Get optimal schedule using Hybrid Deduplicated Beam Search with Weighted Sampling.

    Args:
        workload: Workload object containing transaction data
        num_seqs: Used to scale the beam width

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    # --- Beam Search Parameters ---
    # Width: Maximize exploration while fitting in time budget
    BEAM_WIDTH = int(max(10, num_seqs * 1.5))
    # Samples: Candidates to evaluate per beam node
    SAMPLES_PER_NODE = 24

    num_txns = workload.num_txns

    # --- Precompute Heuristics ---
    # Duration: Cost of running txn in isolation.
    # Used for LPT (Longest Processing Time) heuristics and tie-breaking.
    txn_durations = {t: workload.get_opt_seq_cost([t]) for t in range(num_txns)}
    sorted_by_duration = sorted(range(num_txns), key=lambda t: txn_durations[t], reverse=True)

    # --- Initialization ---
    # Seed beam with a mix of "Heavy" (LPT) starts and Random starts.
    # We explicitly preserve LPT starts even if they have high initial cost (duration)
    # because they often form the backbone of efficient schedules.
    start_candidates = []

    # 1. Top LPT starts (60% of beam)
    num_lpt = int(BEAM_WIDTH * 0.6)
    start_candidates.extend(sorted_by_duration[:num_lpt])

    # 2. Random starts for diversity (40% of beam)
    remaining_pool = [t for t in range(num_txns) if t not in start_candidates]
    needed = BEAM_WIDTH - len(start_candidates)
    if remaining_pool and needed > 0:
        start_candidates.extend(random.sample(remaining_pool, min(len(remaining_pool), needed)))

    beam = []
    for t in start_candidates:
        rem = set(range(num_txns))
        rem.remove(t)
        beam.append({
            'cost': txn_durations[t],
            'seq': [t],
            'rem': rem
        })

    # --- Main Beam Search Loop ---
    for _ in range(num_txns - 1):
        candidates = []

        for parent in beam:
            rem_list = list(parent['rem'])

            # --- Weighted Sampling Strategy ---
            samples = set()

            # 1. Deterministic LPT: Always include the top 2 longest remaining transactions.
            # This ensures we don't miss scheduling "big rocks" early.
            lpt_count = 0
            for t in sorted_by_duration:
                if t in parent['rem']:
                    samples.add(t)
                    lpt_count += 1
                    if lpt_count >= 2:
                        break

            # 2. Weighted Random: Fill remaining sample slots based on duration weights.
            # Longer transactions have higher probability of being sampled.
            needed = SAMPLES_PER_NODE - len(samples)
            if needed > 0 and len(rem_list) > len(samples):
                pool = [t for t in rem_list if t not in samples]
                if len(pool) <= needed:
                    samples.update(pool)
                else:
                    weights = [txn_durations[t] for t in pool]
                    # Loop to ensure we get enough unique samples
                    for _ in range(needed * 2):
                        if len(samples) >= SAMPLES_PER_NODE:
                            break
                        # random.choices is efficient for weighted sampling
                        pick = random.choices(pool, weights=weights, k=1)[0]
                        samples.add(pick)

            # --- Evaluation ---
            for t in samples:
                new_seq = parent['seq'] + [t]
                new_cost = workload.get_opt_seq_cost(new_seq)

                # Priority Tuple:
                # 1. Minimize Total Cost (Makespan)
                # 2. Maximize Added Duration (LPT tie-breaker: prefer filling with heavy items)
                priority = (new_cost, -txn_durations[t])

                candidates.append({
                    'priority': priority,
                    'cost': new_cost,
                    'seq': new_seq,
                    'rem': parent['rem'], # Reference only, copied if selected
                    'added': t
                })

        # --- Selection with Deduplication ---
        candidates.sort(key=lambda x: x['priority'])

        new_beam = []
        seen_states = set()

        for cand in candidates:
            if len(new_beam) >= BEAM_WIDTH:
                break

            # Create new remaining set
            new_rem = cand['rem'].copy()
            new_rem.remove(cand['added'])

            # State Identity: Sorted tuple of remaining transaction IDs.
            # This deduplicates paths that arrive at the same set of completed work.
            # Since candidates are sorted by priority (cost), we keep the best path to this state.
            state_key = tuple(sorted(list(new_rem)))

            if state_key not in seen_states:
                seen_states.add(state_key)
                new_beam.append({
                    'cost': cand['cost'],
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

    # --- Windowed Insertion Refinement ---
    # Move transactions within a local window to fix greedy ordering mistakes.
    # Insertion (pick and move) is more effective than swap.
    WINDOW = 6
    PASSES = 2

    for _ in range(PASSES):
        improved = False
        for i in range(num_txns):
            curr_txn = best_seq[i]

            # Define search window around current position
            start = max(0, i - WINDOW)
            end = min(num_txns, i + WINDOW)

            for j in range(start, end):
                if i == j: continue

                # Construct candidate: Remove i, insert at j
                if i < j:
                    cand_seq = best_seq[:i] + best_seq[i+1:j+1] + [curr_txn] + best_seq[j+1:]
                else:
                    cand_seq = best_seq[:j] + [curr_txn] + best_seq[j:i] + best_seq[i+1:]

                cand_cost = workload.get_opt_seq_cost(cand_seq)

                if cand_cost < best_cost:
                    best_cost = cand_cost
                    best_seq = cand_seq
                    improved = True
                    break # Greedy: take first improvement and restart scan for this item

            if improved:
                break # Restart full scan to propagate dependencies

        if not improved:
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