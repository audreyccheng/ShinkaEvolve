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
    Optimized Transaction Scheduler:
    1. Fast Beam Search with Work-Density Heuristic
    2. Iterated Local Search with Efficient Ruin-and-Recreate (Sampling)

    Args:
        workload: Workload object containing transaction data
        num_seqs: Hint for computational budget

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """

    # --- 0. Precomputation ---
    num_txns = workload.num_txns
    
    # Extract transaction costs for heuristics
    txn_lengths = {}
    for i in range(num_txns):
        try:
            # txns[i][0][3] is often the cost in this simulator
            txn_lengths[i] = workload.txns[i][0][3]
        except (IndexError, TypeError, AttributeError):
            txn_lengths[i] = 1.0

    lpt_sorted = sorted(txn_lengths.keys(), key=lambda k: txn_lengths[k], reverse=True)

    # --- 1. Phase I: Fast Beam Search ---
    # Construct an initial good solution. We keep the beam narrow to save budget for ILS.
    BEAM_WIDTH = 5
    CANDIDATES_TO_CHECK = 8 # Top 4 LPT + 4 Random
    GAMMA = 1.5 # High bias for work density

    # Initialize Beam
    seeds = lpt_sorted[:BEAM_WIDTH]
    # Fill if needed
    if len(seeds) < BEAM_WIDTH:
        rem_seeds = [x for x in range(num_txns) if x not in seeds]
        seeds.extend(random.sample(rem_seeds, min(len(rem_seeds), BEAM_WIDTH - len(seeds))))
    
    beam = []
    for t in seeds:
        seq = [t]
        cost = workload.get_opt_seq_cost(seq)
        acc = txn_lengths[t]
        score = cost - (GAMMA * acc)
        rem = set(range(num_txns))
        rem.remove(t)
        beam.append({
            'seq': seq, 'cost': cost, 'score': score, 
            'acc': acc, 'rem': rem
        })
    
    beam.sort(key=lambda x: x['score'])
    
    # Beam Construction
    for _ in range(num_txns - 1):
        candidates = []
        for parent in beam:
            rem = parent['rem']
            if not rem: continue
            
            to_try = set()
            # A. Heuristic: Top LPT available
            added_lpt = 0
            for t in lpt_sorted:
                if t in rem:
                    to_try.add(t)
                    added_lpt += 1
                    if added_lpt >= 4: break
            
            # B. Random Diversity
            needed = CANDIDATES_TO_CHECK - len(to_try)
            if needed > 0 and len(rem) > len(to_try):
                # Efficient sampling
                pool = [x for x in list(rem) if x not in to_try]
                if pool:
                    to_try.update(random.sample(pool, min(len(pool), needed)))
            
            # Evaluate
            base_seq = parent['seq']
            base_acc = parent['acc']
            
            for t in to_try:
                n_seq = base_seq + [t]
                n_cost = workload.get_opt_seq_cost(n_seq)
                n_acc = base_acc + txn_lengths[t]
                n_score = n_cost - (GAMMA * n_acc)
                
                n_rem = rem.copy()
                n_rem.remove(t)
                
                candidates.append({
                    'seq': n_seq, 'cost': n_cost, 'score': n_score,
                    'acc': n_acc, 'rem': n_rem
                })
        
        if not candidates: break
        
        candidates.sort(key=lambda x: x['score'])
        beam = candidates[:BEAM_WIDTH]
        
    # Pick best from beam
    beam.sort(key=lambda x: x['cost'])
    best_beam_sol = beam[0]
    
    curr_seq = list(best_beam_sol['seq'])
    curr_cost = best_beam_sol['cost']
    
    best_seq = list(curr_seq)
    best_cost = curr_cost
    
    # --- 2. Phase II: Iterated Local Search (ILS) ---
    # Budget: limit by evaluations to control runtime
    # Gen 17 uses ~25k evals effectively. We aim for ~4000 high-quality ones per workload.
    MAX_EVALS = 4000
    evals_performed = 0
    
    while evals_performed < MAX_EVALS:
        
        # A. Ruin (Perturbation)
        # Decide start point: 30% chance to revert to global best (intensification)
        # 70% chance to continue from current (diversification)
        if random.random() < 0.3:
            work_seq = list(best_seq)
        else:
            work_seq = list(curr_seq)
            
        # Remove a random block
        if num_txns > 5:
            block_size = random.randint(2, max(3, int(num_txns * 0.25)))
            start_idx = random.randint(0, len(work_seq) - block_size)
            
            removed_txns = work_seq[start_idx : start_idx + block_size]
            del work_seq[start_idx : start_idx + block_size]
            
            # B. Recreate (Repair)
            # Shuffle removed items and re-insert using best-fit with sampling
            random.shuffle(removed_txns)
            
            for txn in removed_txns:
                best_pos = -1
                best_incr = float('inf')
                
                # Sample positions to check:
                # 1. Start, End
                check_indices = {0, len(work_seq)}
                
                # 2. Approx original vicinity (scaled)
                orig_rel_pos = start_idx / num_txns
                approx_idx = int(orig_rel_pos * len(work_seq))
                for offset in range(-2, 3):
                    p = approx_idx + offset
                    if 0 <= p <= len(work_seq):
                        check_indices.add(p)
                
                # 3. Random spots
                for _ in range(6):
                    check_indices.add(random.randint(0, len(work_seq)))
                
                sorted_checks = sorted(list(check_indices))
                
                # Evaluate positions
                for pos in sorted_checks:
                    work_seq.insert(pos, txn)
                    c = workload.get_opt_seq_cost(work_seq)
                    evals_performed += 1
                    
                    if c < best_incr:
                        best_incr = c
                        best_pos = pos
                    
                    del work_seq[pos] # Backtrack
                    if evals_performed >= MAX_EVALS: break
                
                if evals_performed >= MAX_EVALS:
                    # Emergency insert to keep schedule valid
                    work_seq.insert(best_pos if best_pos != -1 else 0, txn)
                    break
                    
                work_seq.insert(best_pos, txn)
        
        # C. Local Search (Stochastic Descent)
        # Attempt to improve the reconstructed schedule
        # Strategy: "Best of K" random moves
        ls_improved = True
        ls_iter = 0
        current_work_cost = workload.get_opt_seq_cost(work_seq) # Update cost after recreate
        
        while ls_improved and ls_iter < 10 and evals_performed < MAX_EVALS:
            ls_improved = False
            ls_iter += 1
            
            best_neigh_cost = current_work_cost
            best_neigh_seq = None
            
            # Check 12 random neighbors
            for _ in range(12):
                neigh = list(work_seq)
                r = random.random()
                
                if r < 0.4: # Swap
                    i, j = random.sample(range(len(neigh)), 2)
                    neigh[i], neigh[j] = neigh[j], neigh[i]
                elif r < 0.8: # Insert
                    i = random.randint(0, len(neigh)-1)
                    val = neigh.pop(i)
                    j = random.randint(0, len(neigh))
                    neigh.insert(j, val)
                else: # Block Reverse
                    if len(neigh) > 3:
                        bs = random.randint(2, 6)
                        si = random.randint(0, len(neigh) - bs)
                        neigh[si:si+bs] = reversed(neigh[si:si+bs])
                
                nc = workload.get_opt_seq_cost(neigh)
                evals_performed += 1
                
                if nc < best_neigh_cost:
                    best_neigh_cost = nc
                    best_neigh_seq = neigh
                    
                if evals_performed >= MAX_EVALS: break
            
            if best_neigh_seq:
                work_seq = best_neigh_seq
                current_work_cost = best_neigh_cost
                ls_improved = True
                
        # D. Acceptance
        curr_seq = list(work_seq)
        curr_cost = current_work_cost
        
        if curr_cost < best_cost:
            best_cost = curr_cost
            best_seq = list(curr_seq)
            
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