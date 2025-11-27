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
    Hybrid Scheduling Algorithm: Diversity-Enforced Beam Search + ILS/VND

    1. Constructive Phase (Beam Search):
       - Uses 'Work-Density' metric (Cost - Gamma*Work) to bias towards completing heavy work early.
       - Enforces 'Structural Diversity': Filters beam candidates to ensure the schedule tails (last 2 txns)
         are distinct. This prevents the beam from saturating with minor variations of one path.

    2. Refinement Phase (ILS + Stochastic VND):
       - Iterated Local Search with 'Ruin and Recreate' perturbation.
       - Stochastic Variable Neighborhood Descent (VND) for local optimization.
         - Sequentially applies Insert, Swap, and Block-Reverse operators.
         - Uses strided scanning for insertions to optimize computational budget.

    Args:
        workload: Workload object
        num_seqs: Hint for computational budget

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """

    # --- 0. Precomputation ---
    num_txns = workload.num_txns
    
    # Extract transaction weights (proxy for processing time)
    txn_weights = {}
    for i in range(num_txns):
        try:
            # txns[i] -> (metadata, ops...). metadata[3] is length/cost
            txn_weights[i] = workload.txns[i][0][3]
        except (IndexError, TypeError, AttributeError):
            txn_weights[i] = 1.0
            
    # LPT sort for heuristics
    lpt_indices = sorted(txn_weights.keys(), key=lambda k: txn_weights[k], reverse=True)

    # --- 1. Diversity-Enforced Beam Search ---
    
    BEAM_WIDTH = 8
    GAMMA = 1.45  # Slightly higher work bias
    
    # Initialization
    seeds = lpt_indices[:BEAM_WIDTH]
    # Ensure full beam if txn count allows
    if len(seeds) < BEAM_WIDTH:
        rem = list(set(range(num_txns)) - set(seeds))
        seeds.extend(random.sample(rem, min(len(rem), BEAM_WIDTH - len(seeds))))
        
    beam = []
    for t in seeds:
        seq = [t]
        cost = workload.get_opt_seq_cost(seq)
        work = txn_weights[t]
        score = cost - (GAMMA * work)
        beam.append({
            'seq': seq,
            'cost': cost,
            'score': score,
            'work': work,
            'rem': set(range(num_txns)) - {t}
        })
        
    beam.sort(key=lambda x: x['score'])
    
    # Construction Loop
    for _ in range(num_txns - 1):
        candidates = []
        for parent in beam:
            rem = list(parent['rem'])
            if not rem: continue
            
            # Candidate Selection: Top LPT + Random
            to_eval = set()
            
            # Top LPT
            added = 0
            for t in lpt_indices:
                if t in parent['rem']:
                    to_eval.add(t)
                    added += 1
                    if added >= 5: break
            
            # Randoms
            if len(rem) > len(to_eval):
                pool = [x for x in rem if x not in to_eval]
                to_eval.update(random.sample(pool, min(len(pool), 4)))
            
            # Expansion
            base_seq = parent['seq']
            base_work = parent['work']
            
            for t in to_eval:
                new_seq = base_seq + [t]
                new_cost = workload.get_opt_seq_cost(new_seq)
                new_work = base_work + txn_weights[t]
                new_score = new_cost - (GAMMA * new_work)
                
                candidates.append({
                    'seq': new_seq,
                    'cost': new_cost,
                    'score': new_score,
                    'work': new_work,
                    'rem': parent['rem'] - {t}
                })
        
        if not candidates: break
        
        # Diversity Filtered Selection
        # We want to pick the best candidates, but avoid picking those that end identically
        candidates.sort(key=lambda x: x['score'])
        
        next_beam = []
        seen_tails = set()
        
        for cand in candidates:
            if len(next_beam) >= BEAM_WIDTH: break
            
            # Use last 2 transactions as a "structure signature"
            tail = tuple(cand['seq'][-2:])
            if tail in seen_tails:
                continue # Skip to maintain diversity
            
            seen_tails.add(tail)
            next_beam.append(cand)
            
        # Backfill if we were too strict
        if len(next_beam) < BEAM_WIDTH:
            for cand in candidates:
                if len(next_beam) >= BEAM_WIDTH: break
                tail = tuple(cand['seq'][-2:])
                if tail in seen_tails:
                    next_beam.append(cand) # Add duplicate tail if necessary
        
        beam = next_beam

    # Select best from beam
    beam.sort(key=lambda x: x['cost'])
    best_candidate = beam[0]
    
    best_schedule = list(best_candidate['seq'])
    best_cost = best_candidate['cost']

    # --- 2. ILS with Stochastic VND ---
    
    # Budget
    MAX_EVALS = 2500
    evals = 0
    
    while evals < MAX_EVALS:
        
        # A. Ruin and Recreate (Perturbation)
        # Always restart perturbation from the global best to find new basins
        current_seq = list(best_schedule)
        
        # Ruin
        if num_txns > 6:
            # Remove 10-20% chunk
            block_size = random.randint(3, max(4, int(num_txns * 0.2)))
            start_idx = random.randint(0, num_txns - block_size)
            
            removed_txns = current_seq[start_idx : start_idx + block_size]
            del current_seq[start_idx : start_idx + block_size]
            
            # Recreate: Best-Fit Insertion
            random.shuffle(removed_txns)
            
            for txn in removed_txns:
                best_pos = -1
                best_incr = float('inf')
                
                # Optimization: Scan strided positions + boundaries
                # This reduces N^2 complexity to N/k * N
                seq_len = len(current_seq)
                if seq_len > 40:
                    step = 3
                else:
                    step = 1
                    
                check_indices = list(range(0, seq_len + 1, step))
                # Ensure critical boundaries are checked
                if seq_len not in check_indices: check_indices.append(seq_len)
                
                for pos in check_indices:
                    current_seq.insert(pos, txn)
                    c = workload.get_opt_seq_cost(current_seq)
                    evals += 1
                    
                    if c < best_incr:
                        best_incr = c
                        best_pos = pos
                    
                    del current_seq[pos]
                    if evals >= MAX_EVALS: break
                
                if evals >= MAX_EVALS: break
                current_seq.insert(best_pos, txn)
                
            if evals >= MAX_EVALS: break
            
            # Check if reconstruction beat best
            final_rec_cost = workload.get_opt_seq_cost(current_seq)
            if final_rec_cost < best_cost:
                best_cost = final_rec_cost
                best_schedule = list(current_seq)
        else:
            # Fallback for tiny workloads
            random.shuffle(current_seq)
            final_rec_cost = workload.get_opt_seq_cost(current_seq)
        
        # B. Stochastic VND (Local Descent)
        # Apply operators sequentially: Insert -> Swap -> Reverse
        # We use a greedy "First Improvement" strategy
        
        vnd_seq = list(current_seq)
        vnd_cost = final_rec_cost
        
        # 1. Insertion (Shift)
        # Moves a transaction to a new place. Good for dependency ordering.
        improved = True
        while improved and evals < MAX_EVALS:
            improved = False
            for _ in range(25): # Limit attempts
                idx1 = random.randint(0, len(vnd_seq)-1)
                val = vnd_seq.pop(idx1)
                idx2 = random.randint(0, len(vnd_seq)) # Insert allows len
                vnd_seq.insert(idx2, val)
                
                c = workload.get_opt_seq_cost(vnd_seq)
                evals += 1
                
                if c < vnd_cost:
                    vnd_cost = c
                    if c < best_cost:
                        best_cost = c
                        best_schedule = list(vnd_seq)
                    improved = True
                    break # Restart loop on improvement
                else:
                    # Revert
                    del vnd_seq[idx2]
                    vnd_seq.insert(idx1, val)
                if evals >= MAX_EVALS: break

        # 2. Swaps
        # Good for localized fixes.
        improved = True
        while improved and evals < MAX_EVALS:
            improved = False
            for _ in range(25):
                i, j = random.sample(range(len(vnd_seq)), 2)
                vnd_seq[i], vnd_seq[j] = vnd_seq[j], vnd_seq[i]
                
                c = workload.get_opt_seq_cost(vnd_seq)
                evals += 1
                
                if c < vnd_cost:
                    vnd_cost = c
                    if c < best_cost:
                        best_cost = c
                        best_schedule = list(vnd_seq)
                    improved = True
                    break
                else:
                    vnd_seq[i], vnd_seq[j] = vnd_seq[j], vnd_seq[i] # Revert
                if evals >= MAX_EVALS: break
                
        # 3. Block Reversal
        # Good for clustered conflicts.
        if evals < MAX_EVALS:
            for _ in range(10):
                bs = random.randint(2, 6)
                if len(vnd_seq) > bs:
                    si = random.randint(0, len(vnd_seq) - bs)
                    # Reverse in place
                    vnd_seq[si:si+bs] = reversed(vnd_seq[si:si+bs])
                    
                    c = workload.get_opt_seq_cost(vnd_seq)
                    evals += 1
                    
                    if c < vnd_cost:
                        vnd_cost = c
                        if c < best_cost:
                            best_cost = c
                            best_schedule = list(vnd_seq)
                    else:
                        vnd_seq[si:si+bs] = reversed(vnd_seq[si:si+bs]) # Revert
                if evals >= MAX_EVALS: break

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