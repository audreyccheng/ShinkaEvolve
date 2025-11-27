# EVOLVE-BLOCK-START
"""Transaction scheduling algorithm for optimizing makespan across multiple workloads"""

import time
import random
import sys
import os
import math

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

try:
    repo_root = find_repo_root(os.path.dirname(__file__))
    sys.path.insert(0, os.path.join(repo_root, 'openevolve_examples', 'txn_scheduling'))
except Exception as e:
    # Allow execution to proceed if modules are already in path or mock environment
    pass

from txn_simulator import Workload
from workloads import WORKLOAD_1, WORKLOAD_2, WORKLOAD_3


def get_best_schedule(workload, num_seqs):
    """
    Get optimal schedule using Conflict-Aware Tapered Beam Search and Sampled Insertion Refinement.

    Key Logic:
    1. Pre-computation: detailed conflict analysis (R/W sets) to assign contention scores.
    2. Construction: Tapered Beam Search.
       - Beam width decreases linearly to shift computational budget to early, critical decisions.
       - Candidate selection biases towards high-duration and high-conflict transactions.
       - Scoring prefers lower makespan, tie-breaking with higher conflict resolution potential.
    3. Refinement: Local search that samples multiple insertion points to find steepest descent.

    Args:
        workload: Workload object
        num_seqs: Budget parameter used to scale beam width

    Returns:
        (lowest_makespan, schedule)
    """

    num_txns = workload.num_txns

    # --- 1. PRE-COMPUTATION & CONFLICT ANALYSIS ---

    # Extract duration and operation sets
    txn_durations = {}
    txn_rw_sets = {}

    for t in range(num_txns):
        # Default duration
        duration = 1.0
        try:
            # workload.txns[t][0] -> [id, ops_string, ..., duration]
            # We try to get duration from index 3 as observed in prior working code
            duration = workload.txns[t][0][3]
        except:
            pass
        txn_durations[t] = duration

        # Parse Read/Write sets for conflict detection
        reads = set()
        writes = set()
        try:
            # ops string usually at index 1
            ops_str = workload.txns[t][0][1]
            if isinstance(ops_str, str):
                for op in ops_str.split():
                    if '-' in op:
                        parts = op.split('-')
                        if len(parts) == 2:
                            op_type, key = parts
                            if op_type == 'r': reads.add(key)
                            elif op_type == 'w': writes.add(key)
        except:
            # If parsing fails, sets remain empty, conflict score will be 0 (fallback to duration)
            pass
        txn_rw_sets[t] = (reads, writes)

    # Compute Conflict Weights
    # Conflict Weight = Sum of durations of all transactions that conflict with T
    # This identifies "high contention" nodes that should be scheduled intelligently.
    txn_conflict_weights = {t: 0.0 for t in range(num_txns)}

    # O(N^2) naive check is fine for N=100 (~10k ops)
    for i in range(num_txns):
        r1, w1 = txn_rw_sets[i]
        weight = 0.0
        for j in range(num_txns):
            if i == j: continue
            r2, w2 = txn_rw_sets[j]

            # Conflict conditions: W-W, W-R, R-W
            # Intersection check
            if not w1.isdisjoint(w2) or not w1.isdisjoint(r2) or not r1.isdisjoint(w2):
                weight += txn_durations[j]

        txn_conflict_weights[i] = weight

    # --- 2. TAPERED BEAM SEARCH CONSTRUCTION ---

    # Dynamic Beam Width
    # Start with ~2x budget, end with ~0.5x budget
    # Base width derived from num_seqs
    base_width = max(4, int(num_seqs))
    start_width = int(base_width * 2.0)
    end_width = max(2, int(base_width * 0.5))

    # Initial State: (cost, schedule_list, remaining_set_list)
    beam = [(0, [], list(range(num_txns)))]

    for step in range(num_txns):
        # Calculate current beam width (Linear Taper)
        progress = step / num_txns
        current_beam_width = int(start_width + (end_width - start_width) * progress)
        current_beam_width = max(2, current_beam_width)

        candidates_pool = []

        # Expand each partial schedule in the beam
        for p_cost, p_sched, p_remain in beam:

            # Determine subset of next candidates to evaluate
            next_candidates = set()

            # Tail Optimization: Exhaustive search when few items remain
            if len(p_remain) <= 20:
                next_candidates.update(p_remain)
            else:
                # Weighted Sampling
                # Weight = Duration + (Factor * ConflictWeight)
                # We want to clear high-conflict heavy items early
                weights = []
                for t in p_remain:
                    w = txn_durations[t] + 0.1 * txn_conflict_weights[t]
                    weights.append(w)

                # Sample
                k_samples = 6
                samples = random.choices(p_remain, weights=weights, k=k_samples)
                next_candidates.update(samples)

                # Add randoms for diversity
                next_candidates.update(random.sample(p_remain, min(len(p_remain), 2)))

            # Evaluate candidates
            for cand in next_candidates:
                new_sched = p_sched + [cand]

                # 1. Primary Objective: Minimize Makespan
                cost = workload.get_opt_seq_cost(new_sched)

                # 2. Secondary Objectives for Ranking/Pruning
                # If costs are equal, which state is better?
                # The one that scheduled a "troublesome" transaction (High Conflict/Duration)
                # is likely in a better position for the future (less blocking remaining).
                # Score: (Cost, -ConflictWeight, -Duration)
                metric = (cost, -txn_conflict_weights[cand], -txn_durations[cand])

                # Create new state
                new_remain = list(p_remain)
                new_remain.remove(cand)

                candidates_pool.append((metric, new_sched, new_remain))

        # Pruning
        # Sort by metric
        candidates_pool.sort(key=lambda x: x[0])

        # Select top K unique schedules (based on content, though metric sort handles logic)
        # We just take top K
        beam = [(x[0][0], x[1], x[2]) for x in candidates_pool[:current_beam_width]]

    # --- 3. LOCAL SEARCH REFINEMENT (SAMPLED INSERTION) ---

    best_state = beam[0]
    current_cost = best_state[0]
    current_schedule = best_state[1]

    # We use a "Sampled Insertion" strategy which is more aggressive than simple random shift
    search_steps = 600
    no_improv_limit = 100
    no_improv = 0

    for _ in range(search_steps):
        if no_improv >= no_improv_limit:
            break

        # Strategy:
        # 1. Pick a random transaction to move
        # 2. Try inserting it at 'k' random positions + maybe start/end
        # 3. Pick the best position

        src_idx = random.randint(0, num_txns - 1)
        txn = current_schedule[src_idx]

        # Create a base schedule without the txn
        base_sched = current_schedule[:src_idx] + current_schedule[src_idx+1:]

        # Determine trial positions
        # Always try a few specific spots + randoms
        trial_positions = set()
        trial_positions.add(random.randint(0, len(base_sched)))
        trial_positions.add(random.randint(0, len(base_sched)))
        trial_positions.add(random.randint(0, len(base_sched)))

        # Local window around original position often good
        start_win = max(0, src_idx - 5)
        end_win = min(len(base_sched), src_idx + 5)
        trial_positions.add(random.randint(start_win, end_win))

        best_neighbor_cost = float('inf')
        best_neighbor_sched = None

        for pos in trial_positions:
            test_sched = list(base_sched)
            test_sched.insert(pos, txn)

            c = workload.get_opt_seq_cost(test_sched)
            if c < best_neighbor_cost:
                best_neighbor_cost = c
                best_neighbor_sched = test_sched

        # Accept if improvement
        if best_neighbor_cost < current_cost:
            current_cost = best_neighbor_cost
            current_schedule = best_neighbor_sched
            no_improv = 0
        else:
            # Occasional Swap for perturbation (low probability)
            if random.random() < 0.1:
                idx1 = random.randint(0, num_txns - 1)
                idx2 = random.randint(0, num_txns - 1)
                if idx1 != idx2:
                    test_sched = list(current_schedule)
                    test_sched[idx1], test_sched[idx2] = test_sched[idx2], test_sched[idx1]
                    c = workload.get_opt_seq_cost(test_sched)
                    if c < current_cost:
                        current_cost = c
                        current_schedule = test_sched
                        no_improv = 0
                        continue

            no_improv += 1

    return current_cost, current_schedule


def get_random_costs():
    """Evaluate scheduling algorithm on three different workloads."""
    start_time = time.time()

    # Parameter for beam search budget
    num_seqs = 10

    # Process workloads
    workload1 = Workload(WORKLOAD_1)
    makespan1, schedule1 = get_best_schedule(workload1, num_seqs)

    workload2 = Workload(WORKLOAD_2)
    makespan2, schedule2 = get_best_schedule(workload2, num_seqs)

    workload3 = Workload(WORKLOAD_3)
    makespan3, schedule3 = get_best_schedule(workload3, num_seqs)

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