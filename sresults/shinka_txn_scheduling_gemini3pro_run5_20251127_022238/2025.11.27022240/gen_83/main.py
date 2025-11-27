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
    Hybrid Scheduling Algorithm: Adaptive Beam Search + Hybrid Convergent Polish ILS.

    1. Constructive Phase (Adaptive Beam Search):
       - Work-Density Metric: Score = Cost - (Gamma * Work).
       - Gamma Decay: 1.8 -> 0.8. Favor parallelism early, packing late.
       - Efficiency Bonus: Quadratically rewards moves where cost increase < txn length.
       - Candidate Generation: Top 6 LPT + Random.

    2. Refinement Phase (ILS):
       - Multi-Mode Ruin: Tail, Single Block, Disjoint Blocks, Random Scatter.
         - Scales with stagnation.
       - Recreate: LPT-Ordered Greedy Best-Fit.
       - Hybrid Convergent Polish:
         - Runs multiple passes (up to 15) to settle the schedule.
         - Alternates processing order: Primarily LPT (Big Rocks) for packing efficiency,
           periodically (every 3rd pass) Reverse-Schedule order to target makespan tail.
         - Uses 'Early Exit' optimization (delta=0) for speed.

    Args:
        workload: Workload object
        num_seqs: Hint for computational budget

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """

    # --- 0. Precomputation ---
    num_txns = workload.num_txns

    # Extract transaction lengths for heuristics
    txn_lengths = {}
    for i in range(num_txns):
        try:
            # txns[i][0][3] is the length in the simulator
            txn_lengths[i] = workload.txns[i][0][3]
        except (IndexError, TypeError, AttributeError):
            txn_lengths[i] = 1.0

    # Sort indices by length (Longest Processing Time first)
    lpt_indices = sorted(txn_lengths.keys(), key=lambda k: txn_lengths[k], reverse=True)

    # --- 1. Constructive Phase: Adaptive Beam Search ---

    BEAM_WIDTH = max(16, int(num_seqs * 2.5))
    GAMMA_START = 1.8
    GAMMA_END = 0.8  # Allow slightly more packing pressure at the end

    # Seed beam with top LPT and some randoms
    seeds = set(lpt_indices[:BEAM_WIDTH])
    if len(seeds) < BEAM_WIDTH:
        rem = list(set(range(num_txns)) - seeds)
        seeds.update(random.sample(rem, min(len(rem), BEAM_WIDTH - len(seeds))))

    beam = []
    for t in seeds:
        seq = [t]
        cost = workload.get_opt_seq_cost(seq)
        work = txn_lengths[t]
        score = cost - (GAMMA_START * work)
        beam.append({
            'seq': seq,
            'cost': cost,
            'work': work,
            'score': score,
            'rem': set(range(num_txns)) - {t}
        })

    beam.sort(key=lambda x: x['score'])
    beam = beam[:BEAM_WIDTH]

    # Construction Loop
    for step in range(num_txns - 1):
        # Linear Gamma Decay
        progress = (step + 1) / max(1, num_txns - 1)
        current_gamma = GAMMA_START - (progress * (GAMMA_START - GAMMA_END))

        candidates = []
        for parent in beam:
            rem_list = list(parent['rem'])
            if not rem_list: continue

            # Candidate Selection: Smart LPT + Random
            to_eval = set()

            # 1. Top LPT available
            lpt_count = 0
            for t in lpt_indices:
                if t in parent['rem']:
                    to_eval.add(t)
                    lpt_count += 1
                    if lpt_count >= 6: break 

            # 2. Random diversity
            if len(rem_list) > len(to_eval):
                pool = [x for x in rem_list if x not in to_eval]
                count = min(len(pool), 6)
                to_eval.update(random.sample(pool, count))

            parent_cost = parent['cost']
            parent_work = parent['work']

            for t in to_eval:
                new_seq = parent['seq'] + [t]
                new_cost = workload.get_opt_seq_cost(new_seq)
                new_work = parent_work + txn_lengths[t]

                # Base Score
                new_score = new_cost - (current_gamma * new_work)

                # Continuous Efficiency Bonus
                delta = new_cost - parent_cost
                t_len = txn_lengths[t]

                if t_len > 1e-6:
                    # efficiency 1.0 = perfect parallel (delta=0)
                    # efficiency 0.0 = sequential (delta=t_len)
                    efficiency = max(0.0, (t_len - delta) / t_len)

                    # Quadratic bonus for efficiency > 1%
                    if efficiency > 0.01:
                        bonus = t_len * 3.0 * (efficiency ** 2)
                        new_score -= bonus

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

        # Selection
        candidates.sort(key=lambda x: x['score'])

        # Elitism & Diversity
        k_best = int(BEAM_WIDTH * 0.6)
        next_beam = candidates[:k_best]

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

    # --- 2. Refinement Phase: Convergent Gap-Repair ILS ---

    best_schedule = list(current_schedule)
    best_cost = current_cost
    
    last_improved_cycle = -1
    
    # Adjust cycles for computational budget
    ILS_CYCLES = 6
    if num_txns < 20: ILS_CYCLES = 3
    if num_txns > 80: ILS_CYCLES = 4 

    for cycle in range(ILS_CYCLES):

        # A. Restart Strategy
        if cycle > 0 and current_cost > best_cost:
            stagnation = cycle - last_improved_cycle
            # Restart if stagnating deeply or random chance
            if stagnation > 2 and random.random() < 0.5:
                current_schedule = list(best_schedule)
                current_cost = best_cost

        # B. Multi-Mode Ruin
        stagnation = cycle - last_improved_cycle
        ruin_factor = 1.0 + (0.3 * min(3, stagnation))
        
        work_seq = list(current_schedule)
        removed_txns = []

        r_mode = random.random()

        if r_mode < 0.25:
            # Mode 1: Tail Ruin
            base_size = max(4, int(num_txns * 0.25))
            bs = min(len(work_seq), int(base_size * ruin_factor))
            if len(work_seq) > bs:
                start = len(work_seq) - bs
                removed_txns = work_seq[start:]
                del work_seq[start:]

        elif r_mode < 0.55:
            # Mode 2: Single Large Block
            base_size = max(2, int(num_txns * 0.2))
            bs = min(len(work_seq), int(base_size * ruin_factor))
            if len(work_seq) > bs:
                start = random.randint(0, len(work_seq) - bs)
                removed_txns = work_seq[start : start + bs]
                del work_seq[start : start + bs]

        elif r_mode < 0.85:
            # Mode 3: Two Disjoint Blocks
            base_total = max(4, int(num_txns * 0.20))
            total_rem = min(len(work_seq), int(base_total * ruin_factor))
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

        else:
            # Mode 4: Random Scatter
            base_cnt = max(3, int(num_txns * 0.15))
            cnt = min(len(work_seq), int(base_cnt * ruin_factor))
            if cnt > 0:
                indices = sorted(random.sample(range(len(work_seq)), cnt), reverse=True)
                for idx in indices:
                    removed_txns.append(work_seq.pop(idx))

        # Sort removed transactions by LPT (Big Rocks Principle)
        removed_txns.sort(key=lambda t: txn_lengths.get(t, 0), reverse=True)

        # C. Recreate (Greedy Best-Fit)
        for txn in removed_txns:
            best_pos = -1
            best_incr = float('inf')

            for pos in range(len(work_seq) + 1):
                work_seq.insert(pos, txn)
                c = workload.get_opt_seq_cost(work_seq)
                
                # Strict inequality ensures Left-Packing (first best position)
                if c < best_incr:
                    best_incr = c
                    best_pos = pos
                
                del work_seq[pos]

            work_seq.insert(best_pos, txn)

        current_schedule = work_seq
        current_cost = best_incr

        # Update best if strictly better
        if current_cost < best_cost:
            best_cost = current_cost
            best_schedule = list(current_schedule)
            last_improved_cycle = cycle

        # D. Hybrid Convergent Polish (Gap Repair)
        # Scan entire schedule, try to move every transaction to its optimal position.
        
        should_polish = (current_cost <= best_cost * 1.05) or (random.random() < 0.25)

        if should_polish and num_txns > 1:
            improved = True
            passes = 0
            # Higher pass limit for convergence
            MAX_PASSES = 10 
            if num_txns < 40: MAX_PASSES = 15

            while improved and passes < MAX_PASSES:
                improved = False
                passes += 1

                # Order Selection:
                # Primarily LPT to settle Big Rocks.
                # Every 3rd pass, try Reverse Order to optimize Tail/Makespan.
                if passes % 3 == 0:
                    txns_to_check = list(reversed(current_schedule))
                else:
                    txns_to_check = sorted(current_schedule, key=lambda t: txn_lengths.get(t, 0), reverse=True)

                for txn in txns_to_check:
                    try:
                        current_idx = current_schedule.index(txn)
                    except ValueError: continue

                    # Remove
                    del current_schedule[current_idx]
                    
                    # Baseline cost without txn
                    base_val = workload.get_opt_seq_cost(current_schedule)

                    best_pos = -1
                    best_val = float('inf')

                    # Scan all positions
                    for pos in range(len(current_schedule) + 1):
                        current_schedule.insert(pos, txn)
                        c = workload.get_opt_seq_cost(current_schedule)
                        current_schedule.pop(pos) # Backtrack

                        if c < best_val:
                            best_val = c
                            best_pos = pos

                        # Early Exit Optimization:
                        # If delta is 0 (cost == base_val), perfect hole found.
                        if abs(c - base_val) < 1e-9:
                            best_val = c
                            best_pos = pos
                            break

                    # Re-insert at best position
                    current_schedule.insert(best_pos, txn)

                    # Check for improvement
                    if best_val < current_cost - 1e-6:
                        current_cost = best_val
                        improved = True
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_schedule = list(current_schedule)
                            last_improved_cycle = cycle

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