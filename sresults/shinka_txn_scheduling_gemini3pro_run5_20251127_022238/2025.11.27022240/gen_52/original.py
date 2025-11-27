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
    Hybrid Scheduling Algorithm: Zero-Cost Bonus Beam Search + Multi-Mode ILS.

    1. Constructive Phase (Beam Search):
       - Uses 'Work-Density' metric (Cost - Gamma * Work).
       - Features 'Zero-Cost Bonus': Heavily rewards candidates that add work 
         without increasing makespan (perfect parallelism).
       - Maintains diversity by sampling random candidates alongside LPT heuristic.

    2. Refinement Phase (ILS + SA):
       - Multi-Mode Ruin: Randomly selects between removing one large contiguous block
         or two smaller dispersed blocks to vary perturbation scale.
       - Recreate: Greedy Best-Fit insertion to repair schedule.
       - Local Search: Simulated Annealing with higher probability for Insertion (Shift).

    Args:
        workload: Workload object
        num_seqs: Hint for computational budget

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """

    # --- 0. Setup & Precomputation ---
    num_txns = workload.num_txns

    # Extract transaction lengths
    txn_lengths = {}
    for i in range(num_txns):
        try:
            txn_lengths[i] = workload.txns[i][0][3]
        except (IndexError, TypeError, AttributeError):
            txn_lengths[i] = 1.0

    # LPT sort
    lpt_indices = sorted(txn_lengths.keys(), key=lambda k: txn_lengths[k], reverse=True)

    # --- 1. Constructive Phase: Beam Search with Parallelism Bonus ---
    
    # Parameters
    BEAM_WIDTH = max(16, int(num_seqs * 2.2))
    GAMMA = 1.5  # High bias for packing work early
    
    # Seed beam
    seeds = set(lpt_indices[:BEAM_WIDTH])
    if len(seeds) < BEAM_WIDTH:
        rem_slots = BEAM_WIDTH - len(seeds)
        seeds.update(random.sample(range(num_txns), min(num_txns - len(seeds), rem_slots)))

    beam = []
    for t in seeds:
        seq = [t]
        cost = workload.get_opt_seq_cost(seq)
        acc_work = txn_lengths[t]
        score = cost - (GAMMA * acc_work)
        rem = set(range(num_txns))
        rem.remove(t)
        beam.append({'seq': seq, 'cost': cost, 'score': score, 'work': acc_work, 'rem': rem})

    beam.sort(key=lambda x: x['score'])
    beam = beam[:BEAM_WIDTH]

    # Construction Loop
    for _ in range(num_txns - 1):
        candidates = []
        for parent in beam:
            if not parent['rem']: continue

            # Candidate Selection: Top LPT + Random
            to_eval = set()
            rem_list = list(parent['rem'])

            # 1. Top LPT candidates
            added = 0
            for t in lpt_indices:
                if t in parent['rem']:
                    to_eval.add(t)
                    added += 1
                    if added >= 6: break

            # 2. Random candidates (Diversity)
            needed = 6
            if len(rem_list) > len(to_eval):
                pool = [x for x in rem_list if x not in to_eval]
                to_eval.update(random.sample(pool, min(len(pool), needed)))

            parent_cost = parent['cost']
            parent_work = parent['work']

            for t in to_eval:
                new_seq = parent['seq'] + [t]
                new_cost = workload.get_opt_seq_cost(new_seq)
                new_work = parent_work + txn_lengths[t]
                
                # Base Score calculation
                new_score = new_cost - (GAMMA * new_work)

                # Zero-Cost Bonus:
                # If adding transaction creates near-zero makespan increase,
                # it implies perfect parallelism. Reward this significantly.
                if new_cost <= parent_cost + 1e-6:
                    new_score -= (txn_lengths[t] * 2.5)

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

        # Diversity-Preserving Selection
        candidates.sort(key=lambda x: x['score'])
        
        # Elitism: Keep best 60%
        k_best = int(BEAM_WIDTH * 0.6)
        next_beam = candidates[:k_best]

        # Diversity: Fill rest from top 300% to ensure we don't tunnel vision
        if len(candidates) > k_best:
            rem_needed = BEAM_WIDTH - len(next_beam)
            pool = candidates[k_best : min(len(candidates), BEAM_WIDTH * 3)]
            if len(pool) <= rem_needed:
                next_beam.extend(pool)
            else:
                next_beam.extend(random.sample(pool, rem_needed))

        beam = next_beam

    # Select best from construction
    beam.sort(key=lambda x: x['cost'])
    best_candidate = beam[0]
    
    current_schedule = best_candidate['seq']
    current_cost = best_candidate['cost']

    # --- 2. Refinement Phase: Multi-Mode ILS ---

    best_schedule = list(current_schedule)
    best_cost = current_cost

    ILS_CYCLES = 8
    if num_txns < 20: ILS_CYCLES = 3
    
    SA_STEPS = 150

    for cycle in range(ILS_CYCLES):
        # A. Restart Strategy
        # Periodically restart from global best to explore its neighborhood deeper
        if cycle > 0 and current_cost > best_cost:
            if random.random() < 0.6:
                current_schedule = list(best_schedule)
                current_cost = best_cost

        # B. Ruin Strategy (Multi-Mode)
        work_seq = list(current_schedule)
        removed_txns = []
        
        mode = random.random()
        
        # Mode 1: Single Large Block (Focus on local reordering of a window)
        if mode < 0.7:
            block_size = max(2, int(num_txns * random.uniform(0.15, 0.25)))
            if num_txns > block_size:
                start_idx = random.randint(0, num_txns - block_size)
                removed_txns = work_seq[start_idx : start_idx + block_size]
                del work_seq[start_idx : start_idx + block_size]
        
        # Mode 2: Two Smaller Dispersed Blocks (Focus on global shuffling)
        else:
            total_remove = max(4, int(num_txns * 0.15))
            b1 = total_remove // 2
            b2 = total_remove - b1
            
            # First block
            if len(work_seq) > b1:
                idx1 = random.randint(0, len(work_seq) - b1)
                removed_txns.extend(work_seq[idx1 : idx1 + b1])
                del work_seq[idx1 : idx1 + b1]
            
            # Second block
            if len(work_seq) > b2:
                idx2 = random.randint(0, len(work_seq) - b2)
                removed_txns.extend(work_seq[idx2 : idx2 + b2])
                del work_seq[idx2 : idx2 + b2]

        random.shuffle(removed_txns)

        # C. Recreate Strategy (Greedy Best-Fit)
        for txn in removed_txns:
            best_pos = -1
            best_incr = float('inf')

            # Check all positions
            for pos in range(len(work_seq) + 1):
                work_seq.insert(pos, txn)
                cost = workload.get_opt_seq_cost(work_seq)
                if cost < best_incr:
                    best_incr = cost
                    best_pos = pos
                del work_seq[pos]

            work_seq.insert(best_pos, txn)

        current_schedule = work_seq
        current_cost = best_incr
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_schedule = list(current_schedule)

        # D. Local Search (Simulated Annealing)
        temp = max(1.0, current_cost * 0.05)
        cooling = 0.95
        
        for _ in range(SA_STEPS):
            neighbor = list(current_schedule)
            r = random.random()
            
            # Prioritize Insert (Shift) as it is the most effective operator for scheduling
            if r < 0.50: 
                # Insert
                i = random.randint(0, num_txns - 1)
                val = neighbor.pop(i)
                j = random.randint(0, num_txns - 1)
                neighbor.insert(j, val)
            elif r < 0.80: 
                # Swap
                i, j = random.sample(range(num_txns), 2)
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            else: 
                # Block Reverse
                if num_txns > 3:
                    bs = random.randint(2, 6)
                    si = random.randint(0, num_txns - bs)
                    neighbor[si:si+bs] = reversed(neighbor[si:si+bs])

            n_cost = workload.get_opt_seq_cost(neighbor)
            delta = n_cost - current_cost

            accept = False
            if delta < 0:
                accept = True
            elif temp > 0.001 and random.random() < math.exp(-delta / temp):
                accept = True

            if accept:
                current_schedule = neighbor
                current_cost = n_cost
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_schedule = list(current_schedule)
            
            temp *= cooling

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