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
    Hybrid Scheduling Algorithm: Adaptive Beam Search + Deep Convergent Polish ILS.

    1. Constructive Phase (Beam Search):
       - Work-Density Metric with continuous efficiency bonus.
       - Parameters tuned for parallelism early, packing late.
    
    2. Refinement Phase (ILS):
       - Multi-Mode Ruin (Tail, Block, Disjoint).
       - Recreate: LPT-First Best-Fit.
       - Deep Convergent Polish:
         - Iteratively removes and re-inserts transactions in LPT order.
         - Left-Packing Bias: Explicitly prefers the earliest position that minimizes makespan.
         - Convergence: Continues passes as long as schedule structure changes (transactions moving left),
           not just when cost improves. This compacts the schedule to open future gaps.
         - Efficiency: Uses 'Early Exit' (perfect fit detection) to speed up scanning.

    Args:
        workload: Workload object
        num_seqs: Hint for computational budget

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """

    # --- 0. Precomputation ---
    num_txns = workload.num_txns
    
    # Extract transaction lengths
    txn_lengths = {}
    for i in range(num_txns):
        try:
            txn_lengths[i] = workload.txns[i][0][3]
        except (IndexError, TypeError, AttributeError):
            txn_lengths[i] = 1.0

    lpt_indices = sorted(txn_lengths.keys(), key=lambda k: txn_lengths[k], reverse=True)

    # --- 1. Constructive Phase: Adaptive Beam Search ---
    
    # Parameters
    BEAM_WIDTH = max(16, int(num_seqs * 2.5))
    GAMMA_START = 1.9
    GAMMA_END = 1.0

    # Seed generation
    seeds = set(lpt_indices[:BEAM_WIDTH])
    if len(seeds) < BEAM_WIDTH:
        rem_slots = BEAM_WIDTH - len(seeds)
        rem_pool = list(set(range(num_txns)) - seeds)
        seeds.update(random.sample(rem_pool, min(len(rem_pool), rem_slots)))

    beam = []
    for t in seeds:
        seq = [t]
        cost = workload.get_opt_seq_cost(seq)
        work = txn_lengths[t]
        score = cost - (GAMMA_START * work)
        beam.append({
            'seq': seq, 'cost': cost, 'work': work, 'score': score, 
            'rem': set(range(num_txns)) - {t}
        })

    beam.sort(key=lambda x: x['score'])
    beam = beam[:BEAM_WIDTH]

    for step in range(num_txns - 1):
        progress = (step + 1) / max(1, num_txns - 1)
        current_gamma = GAMMA_START - (progress * (GAMMA_START - GAMMA_END))

        candidates = []
        for parent in beam:
            if not parent['rem']: continue

            # Candidate selection
            to_eval = set()
            
            # Top LPT
            added = 0
            for t in lpt_indices:
                if t in parent['rem']:
                    to_eval.add(t)
                    added += 1
                    if added >= 5: break
            
            # Random filler
            rem_list = list(parent['rem'])
            if len(rem_list) > len(to_eval):
                pool = [x for x in rem_list if x not in to_eval]
                to_eval.update(random.sample(pool, min(len(pool), 5)))

            p_cost = parent['cost']
            p_work = parent['work']

            for t in to_eval:
                new_seq = parent['seq'] + [t]
                new_cost = workload.get_opt_seq_cost(new_seq)
                new_work = p_work + txn_lengths[t]
                
                # Base score
                new_score = new_cost - (current_gamma * new_work)
                
                # Efficiency Bonus
                delta = new_cost - p_cost
                t_len = txn_lengths[t]
                if t_len > 1e-6:
                    efficiency = max(0.0, (t_len - delta) / t_len)
                    if efficiency > 0.01:
                        # Quadratic reward for parallelism
                        new_score -= (t_len * 3.0 * (efficiency ** 2))

                new_rem = parent['rem'].copy()
                new_rem.remove(t)
                
                candidates.append({
                    'seq': new_seq, 'cost': new_cost, 'score': new_score, 
                    'work': new_work, 'rem': new_rem
                })

        if not candidates: break
        
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

    # --- 2. Refinement Phase: ILS with Deep Polish ---

    best_schedule = list(current_schedule)
    best_cost = current_cost
    last_improved_cycle = -1

    ILS_CYCLES = 6
    if num_txns < 20: ILS_CYCLES = 3

    for cycle in range(ILS_CYCLES):
        
        # A. Restart logic
        if cycle > 0 and current_cost > best_cost:
            stagnation = cycle - last_improved_cycle
            if stagnation > 2 or random.random() < 0.3:
                current_schedule = list(best_schedule)
                current_cost = best_cost

        # B. Ruin
        ruin_factor = 1.0 + (0.2 * max(0, cycle - last_improved_cycle))
        work_seq = list(current_schedule)
        removed = []
        
        r_mode = random.random()
        if r_mode < 0.25: # Tail
            bs = min(len(work_seq), int(max(4, num_txns * 0.25) * ruin_factor))
            if len(work_seq) > bs:
                removed = work_seq[len(work_seq)-bs:]
                del work_seq[len(work_seq)-bs:]
        elif r_mode < 0.55: # Block
            bs = min(len(work_seq), int(max(2, num_txns * 0.20) * ruin_factor))
            if len(work_seq) > bs:
                start = random.randint(0, len(work_seq) - bs)
                removed = work_seq[start:start+bs]
                del work_seq[start:start+bs]
        elif r_mode < 0.85: # Disjoint
            total = min(len(work_seq), int(max(4, num_txns * 0.20) * ruin_factor))
            b1 = total // 2
            b2 = total - b1
            if len(work_seq) > b1:
                s1 = random.randint(0, len(work_seq) - b1)
                removed.extend(work_seq[s1:s1+b1])
                del work_seq[s1:s1+b1]
            if len(work_seq) > b2:
                s2 = random.randint(0, len(work_seq) - b2)
                removed.extend(work_seq[s2:s2+b2])
                del work_seq[s2:s2+b2]
        else: # Scatter
            cnt = min(len(work_seq), int(max(3, num_txns * 0.15) * ruin_factor))
            if cnt > 0:
                indices = sorted(random.sample(range(len(work_seq)), cnt), reverse=True)
                for i in indices: removed.append(work_seq.pop(i))
        
        # C. Recreate (LPT)
        removed.sort(key=lambda t: txn_lengths.get(t, 0), reverse=True)
        for txn in removed:
            best_pos = -1
            best_incr = float('inf')
            for pos in range(len(work_seq) + 1):
                work_seq.insert(pos, txn)
                c = workload.get_opt_seq_cost(work_seq)
                if c < best_incr: # Prefer Left (first min)
                    best_incr = c
                    best_pos = pos
                del work_seq[pos]
            work_seq.insert(best_pos, txn)
        
        current_schedule = work_seq
        current_cost = best_incr
        
        if current_cost < best_cost:
            best_cost = current_cost
            best_schedule = list(current_schedule)
            last_improved_cycle = cycle

        # D. Deep Convergent Polish
        # Only polish promising solutions to save time
        if current_cost <= best_cost * 1.05 or random.random() < 0.2:
            polish_active = True
            passes = 0
            MAX_PASSES = 12 if num_txns > 50 else 20
            
            while polish_active and passes < MAX_PASSES:
                polish_active = False
                passes += 1
                
                # Sort txns by length (LPT) to place big rocks first
                txns_to_check = sorted(current_schedule, key=lambda t: txn_lengths.get(t, 0), reverse=True)
                
                for txn in txns_to_check:
                    try:
                        curr_idx = current_schedule.index(txn)
                    except ValueError: continue
                    
                    # Remove txn
                    del current_schedule[curr_idx]
                    base_val = workload.get_opt_seq_cost(current_schedule)
                    
                    best_pos = -1
                    best_val = float('inf')
                    
                    # Scan for best position
                    for pos in range(len(current_schedule) + 1):
                        current_schedule.insert(pos, txn)
                        c = workload.get_opt_seq_cost(current_schedule)
                        current_schedule.pop(pos)
                        
                        # Check improvement
                        if c < best_val:
                            best_val = c
                            best_pos = pos
                        
                        # Early Exit: Perfect Fit
                        # Also serves as Left-Packing enforcement since we break at first perfect slot
                        if abs(c - base_val) < 1e-9:
                            best_val = c
                            best_pos = pos
                            break
                    
                    # Re-insert
                    current_schedule.insert(best_pos, txn)
                    
                    # Check what happened
                    # 1. Cost Improvement
                    if best_val < current_cost - 1e-6:
                        current_cost = best_val
                        polish_active = True
                        if current_cost < best_cost:
                            best_cost = current_cost
                            best_schedule = list(current_schedule)
                            last_improved_cycle = cycle
                    
                    # 2. Structural Improvement (Left Packing)
                    # If we moved the txn to a strictly earlier index with equal cost,
                    # we compacted the schedule. This is worth another pass.
                    elif best_pos < curr_idx and best_val <= current_cost + 1e-9:
                        polish_active = True
                        
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