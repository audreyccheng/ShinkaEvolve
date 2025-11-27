# EVOLVE-BLOCK-START
"""Transaction scheduling algorithm for optimizing makespan across multiple workloads"""

import time
import random
import math
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
    Hybrid Scheduling Algorithm: Tiered-Efficiency Beam Search + LPT-Polish ILS.

    1. Constructive Phase (Beam Search):
       - Uses 'Work-Density' metric (Cost - Gamma * Work).
       - 'Tiered Efficiency Bonus':
         - Tier 1: Massive bonus for Zero-Cost insertion (perfect parallelism).
         - Tier 2: Quadratic bonus for High Efficiency (>50% latency hiding).
       - Tie-Breaking: Implicitly favors LPT via the Work component in score.

    2. Refinement Phase (ILS):
       - Multi-Mode Ruin: Tail (Critical Path), Block (Locality), Disjoint (Global).
       - Recreate: LPT-Ordered Best-Fit. Large items are inserted first to establish structure.
       - Deterministic Polish (Gap Repair):
         - Iteratively removes and re-inserts every transaction.
         - Critical improvement: Processes transactions in LPT order (Big rocks first).
         - Critical improvement: 'Early Exit' if a position results in no makespan increase relative
           to the schedule without the transaction (perfect packing), stopping the scan immediately.

    Args:
        workload: Workload object
        num_seqs: Hint for computational budget

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """

    # --- 0. Setup & Precomputation ---
    num_txns = workload.num_txns

    # Extract transaction lengths for heuristics
    txn_lengths = {}
    for i in range(num_txns):
        try:
            # txns[i][0] is the first op, [3] is metadata/length
            txn_lengths[i] = workload.txns[i][0][3]
        except (IndexError, TypeError, AttributeError):
            txn_lengths[i] = 1.0

    # Sort indices by length (Longest Processing Time first)
    lpt_indices = sorted(txn_lengths.keys(), key=lambda k: txn_lengths[k], reverse=True)

    # --- 1. Constructive Phase: Beam Search with Tiered Efficiency ---

    # Dynamic Beam Width
    BEAM_WIDTH = max(16, int(num_seqs * 2.5))

    # Adaptive Gamma: Higher start favors parallel packing, lower end favors tight fitting
    GAMMA_START = 2.0
    GAMMA_END = 1.0

    # Seed generation: Top LPT + Random filler
    seeds = set(lpt_indices[:BEAM_WIDTH])
    if len(seeds) < BEAM_WIDTH:
        rem_slots = BEAM_WIDTH - len(seeds)
        rem_pool = list(set(range(num_txns)) - seeds)
        seeds.update(random.sample(rem_pool, min(len(rem_pool), rem_slots)))

    beam = []
    for t in seeds:
        seq = [t]
        cost = workload.get_opt_seq_cost(seq)
        acc_work = txn_lengths[t]
        # Initial score
        score = cost - (GAMMA_START * acc_work)

        rem = set(range(num_txns))
        rem.remove(t)
        beam.append({
            'seq': seq,
            'cost': cost,
            'score': score,
            'work': acc_work,
            'rem': rem
        })

    beam.sort(key=lambda x: x['score'])
    beam = beam[:BEAM_WIDTH]

    # Construction Loop
    for step in range(num_txns - 1):
        # Decay Gamma linearly
        progress = (step + 1) / max(1, num_txns - 1)
        current_gamma = GAMMA_START - (progress * (GAMMA_START - GAMMA_END))

        candidates = []
        for parent in beam:
            if not parent['rem']: continue

            # Smart Candidate Selection
            to_eval = set()
            rem_list = list(parent['rem'])

            # 1. Top LPT candidates in remaining set
            added_lpt = 0
            for t in lpt_indices:
                if t in parent['rem']:
                    to_eval.add(t)
                    added_lpt += 1
                    if added_lpt >= 5: break # Check top 5 available LPT

            # 2. Random candidates for diversity
            needed = 5
            if len(rem_list) > len(to_eval):
                pool = [x for x in rem_list if x not in to_eval]
                to_eval.update(random.sample(pool, min(len(pool), needed)))

            parent_cost = parent['cost']
            parent_work = parent['work']
            parent_seq = parent['seq']

            for t in to_eval:
                new_seq = parent_seq + [t]
                new_cost = workload.get_opt_seq_cost(new_seq)
                new_work = parent_work + txn_lengths[t]

                # Base Score: Cost - Work Density
                new_score = new_cost - (current_gamma * new_work)

                # Tiered Efficiency Bonus
                delta_cost = new_cost - parent_cost
                t_len = txn_lengths[t]

                efficiency = 0.0
                if t_len > 1e-9:
                    efficiency = max(0.0, (t_len - delta_cost) / t_len)

                # Tier 1: Perfect Parallelism (Zero Cost Increase)
                if delta_cost <= 1e-6:
                    new_score -= (t_len * 3.0)
                # Tier 2: High Partial Efficiency
                elif efficiency > 0.5:
                    # Quadratic scaling favors 0.9 over 0.6 significantly
                    new_score -= (t_len * 1.5 * (efficiency ** 2))

                new_rem = parent['rem'].copy()
                new_rem.remove(t)

                candidates.append({
                    'seq': new_seq,
                    'cost': new_cost,
                    'score': new_score,
                    'work': new_work,
                    'rem': new_rem
                })

        if not candidates: break

        # Selection: Sort by score (lowest is best)
        candidates.sort(key=lambda x: x['score'])

        # Elitism + Diversity
        k_best = int(BEAM_WIDTH * 0.6)
        next_beam = candidates[:k_best]

        # Fill remainder from a larger pool to preserve diversity
        if len(candidates) > k_best:
            rem_needed = BEAM_WIDTH - len(next_beam)
            pool = candidates[k_best : min(len(candidates), BEAM_WIDTH * 3)]
            if len(pool) <= rem_needed:
                next_beam.extend(pool)
            else:
                next_beam.extend(random.sample(pool, rem_needed))

        beam = next_beam

    # Best from construction
    beam.sort(key=lambda x: x['cost'])
    current_schedule = beam[0]['seq']
    current_cost = beam[0]['cost']

    # --- 2. Refinement Phase: ILS with LPT-Polish ---

    best_schedule = list(current_schedule)
    best_cost = current_cost

    # Adjust cycles based on problem size
    ILS_CYCLES = 6
    if num_txns < 20: ILS_CYCLES = 3

    for cycle in range(ILS_CYCLES):

        # A. Restart Strategy
        # If we've drifted to a worse solution, probabilistically restart from best
        if cycle > 0 and current_cost > best_cost:
            if random.random() < 0.4:
                current_schedule = list(best_schedule)
                current_cost = best_cost

        # B. Multi-Mode Ruin
        work_seq = list(current_schedule)
        removed_txns = []

        mode = random.random()

        if mode < 0.30:
            # Mode 1: Tail Ruin (Break critical path at the end)
            bs = max(4, int(num_txns * 0.25))
            if len(work_seq) > bs:
                start = len(work_seq) - bs
                removed_txns = work_seq[start:]
                del work_seq[start:]

        elif mode < 0.65:
            # Mode 2: Random Block Ruin (Locality)
            bs = max(2, int(num_txns * 0.20))
            if len(work_seq) > bs:
                start = random.randint(0, len(work_seq) - bs)
                removed_txns = work_seq[start : start + bs]
                del work_seq[start : start + bs]

        else:
            # Mode 3: Disjoint Blocks (Global shuffle)
            total_rem = max(4, int(num_txns * 0.20))
            b1 = total_rem // 2
            b2 = total_rem - b1

            if len(work_seq) > b1:
                s1 = random.randint(0, len(work_seq) - b1)
                removed_txns.extend(work_seq[s1 : s1 + b1])
                del work_seq[s1 : s1 + b1]
            if len(work_seq) > b2:
                s2 = random.randint(0, len(work_seq) - b2)
                removed_txns.extend(work_seq[s2 : s2 + b2])
                del work_seq[s2 : s2 + b2]

        # C. Recreate: LPT-First Greedy Best-Fit
        # Sort removed transactions by length descending
        removed_txns.sort(key=lambda t: txn_lengths.get(t, 0), reverse=True)

        for txn in removed_txns:
            best_pos = -1
            best_incr = float('inf')

            # Find best insertion point
            for pos in range(len(work_seq) + 1):
                work_seq.insert(pos, txn)
                c = workload.get_opt_seq_cost(work_seq)

                # Check strict inequality to prefer earlier positions on ties
                if c < best_incr:
                    best_incr = c
                    best_pos = pos

                del work_seq[pos]

            work_seq.insert(best_pos, txn)

        current_schedule = work_seq
        current_cost = best_incr

        # Save if improved
        if current_cost < best_cost:
            best_cost = current_cost
            best_schedule = list(current_schedule)

        # D. Polish Phase: Deterministic LPT-Ordered Gap Repair
        # Systematically try to improve the schedule by moving transactions.
        # Run if the solution is promising.

        should_polish = (current_cost <= best_cost * 1.05) or (random.random() < 0.3)

        if should_polish and num_txns > 1:
            improved = True
            passes = 0
            MAX_PASSES = 1 # Expensive operation, limit passes

            while improved and passes < MAX_PASSES:
                improved = False
                passes += 1

                # Order to check: LPT (Largest items first)
                # This ensures big blocks are settled in optimal spots before small items fill gaps
                txns_to_check = sorted(current_schedule, key=lambda t: txn_lengths.get(t, 0), reverse=True)

                for txn in txns_to_check:
                    try:
                        current_idx = current_schedule.index(txn)
                    except ValueError: continue

                    # Remove transaction
                    del current_schedule[current_idx]

                    # Cost without this transaction (baseline for this step)
                    base_val = workload.get_opt_seq_cost(current_schedule)

                    best_pos = -1
                    best_val = float('inf')

                    # Scan for best position
                    for pos in range(len(current_schedule) + 1):
                        current_schedule.insert(pos, txn)
                        c = workload.get_opt_seq_cost(current_schedule)
                        current_schedule.pop(pos) # Backtrack

                        if c < best_val:
                            best_val = c
                            best_pos = pos

                        # Early Exit Optimization:
                        # If the new cost equals the baseline cost (schedule without txn),
                        # the transaction is perfectly hidden. We can't do better than "free".
                        # Stop scanning and accept this position (Pack Left).
                        if abs(c - base_val) < 1e-9:
                            break

                    # Re-insert at best position
                    current_schedule.insert(best_pos, txn)

                    # Check if this move improved the global best
                    if best_val < current_cost - 1e-6:
                        current_cost = best_val
                        improved = True
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_schedule = list(current_schedule)

    return best_cost, best_schedule


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
    cost1 = workload1.get_opt_seq_cost(schedule1)

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