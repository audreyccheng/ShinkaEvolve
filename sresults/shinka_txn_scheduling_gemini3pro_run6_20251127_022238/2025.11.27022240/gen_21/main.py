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
    Get optimal schedule using Deduplicated Beam Search with Ends-Out Sampling and Windowed Refinement.

    Args:
        workload: Workload object containing transaction data
        num_seqs: Used to determine the beam width (effort level)

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    # Beam Search Parameters
    # Wider beam for better exploration coverage
    BEAM_WIDTH = int(max(12, num_seqs * 2.0))
    SAMPLES_PER_NODE = 24

    num_txns = workload.num_txns

    # Precompute transaction durations
    txn_durations = {t: workload.get_opt_seq_cost([t]) for t in range(num_txns)}

    # Initialization
    # Seed beam with LPT (Longest Processing Time) to secure bottlenecks
    sorted_by_duration = sorted(range(num_txns), key=lambda t: txn_durations[t], reverse=True)

    start_candidates = set()
    start_candidates.update(sorted_by_duration[:BEAM_WIDTH])

    # Fill remaining slots with random starts for diversity
    while len(start_candidates) < BEAM_WIDTH * 1.5:
        start_candidates.add(random.randint(0, num_txns - 1))

    beam = []
    for t in list(start_candidates)[:int(BEAM_WIDTH * 1.5)]:
        rem = set(range(num_txns))
        rem.remove(t)
        beam.append({
            'cost': txn_durations[t],
            'seq': [t],
            'rem': rem
        })

    # Prune initial beam to WIDTH
    beam.sort(key=lambda x: (x['cost'], -txn_durations[x['seq'][0]]))
    beam = beam[:BEAM_WIDTH]

    # Beam Search Loop
    for _ in range(num_txns - 1):
        candidates = []

        for parent in beam:
            rem_list = list(parent['rem'])

            # --- Ends-Out Sampling Strategy ---
            if len(rem_list) <= SAMPLES_PER_NODE:
                samples = rem_list
            else:
                samples = set()
                # 1. LPT: Top longest txns (Critical path reduction)
                samples.update(sorted(rem_list, key=lambda t: txn_durations[t], reverse=True)[:2])

                # 2. SPT: Top shortest txns (Gap filling / Latency hiding)
                samples.update(sorted(rem_list, key=lambda t: txn_durations[t])[:1])

                # 3. Random: Explore remaining space
                needed = SAMPLES_PER_NODE - len(samples)
                pool = [t for t in rem_list if t not in samples]
                if needed > 0 and pool:
                    samples.update(random.sample(pool, min(len(pool), needed)))

            # --- Evaluation ---
            for t in samples:
                new_seq = parent['seq'] + [t]
                new_cost = workload.get_opt_seq_cost(new_seq)

                # Priority: (Cost, -Duration)
                # Minimize Cost. Tie-break: prefer scheduling heavy items (Max Duration)
                priority = (new_cost, -txn_durations[t])

                candidates.append({
                    'priority': priority,
                    'cost': new_cost,
                    'seq': new_seq,
                    'rem': parent['rem'], # Reference
                    'added': t
                })

        # Sort candidates
        candidates.sort(key=lambda x: x['priority'])

        # --- Deduplication & Selection ---
        new_beam = []
        seen_states = set()

        for cand in candidates:
            if len(new_beam) >= BEAM_WIDTH:
                break

            new_rem = cand['rem'].copy()
            new_rem.remove(cand['added'])

            # State Deduplication: Key = Frozen set of remaining items
            state_key = frozenset(new_rem)

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

    # Select best from beam
    best_result = min(beam, key=lambda x: x['cost'])
    best_cost = best_result['cost']
    best_seq = best_result['seq']

    # --- Local Search Refinement: Windowed Insertion ---
    # Shifting an element is stronger than adjacent swaps for fixing ordering.
    WINDOW = 5
    MAX_PASSES = 2

    for _ in range(MAX_PASSES):
        improved = False
        for i in range(num_txns):
            # Try moving element at i to valid positions in window [i-W, i+W]
            val = best_seq[i]

            start_j = max(0, i - WINDOW)
            end_j = min(num_txns, i + WINDOW + 1)

            for j in range(start_j, end_j):
                if j == i: continue

                # Construct candidate sequence with insertion
                if j < i:
                    cand_seq = best_seq[:j] + [val] + best_seq[j:i] + best_seq[i+1:]
                else: # j > i
                    cand_seq = best_seq[:i] + best_seq[i+1:j+1] + [val] + best_seq[j+1:]

                new_c = workload.get_opt_seq_cost(cand_seq)
                if new_c < best_cost:
                    best_cost = new_c
                    best_seq = cand_seq
                    improved = True
                    # Greedy first improvement
                    break

            if improved:
                # Restart scan to ripple effects of the change
                break

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