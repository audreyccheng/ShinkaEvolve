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
    Algorithmic Approach:
    1. Constructive Phase: Beam Search with Adaptive Gamma & Continuous Efficiency Bonus.
       - Metric: Cost - (Gamma * Work) - EfficiencyBonus
       - Gamma: Decays linearly to shift focus from packing to makespan.
       - Bonus: Continuous reward for low-cost insertions (parallelism), not just zero-cost.
    
    2. Refinement Phase: Iterated Local Search with Deterministic Polish.
       - Ruin: Multi-block removal to perturb schedule.
       - Recreate: Greedy best-fit insertion.
       - Polish: Deterministic 'Best-Insertion' pass on a subset of transactions. 
         Replaces stochastic local search with a systematic optimization of transaction placement.
         
    Args:
        workload: Workload object
        num_seqs: Hint for computational budget

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    
    # 0. Setup & Precomputation
    num_txns = workload.num_txns
    
    # Extract lengths for heuristics
    txn_lengths = {}
    for i in range(num_txns):
        try:
            # First op usually contains length at index 3
            txn_lengths[i] = workload.txns[i][0][3]
        except (IndexError, TypeError, AttributeError):
            txn_lengths[i] = 1.0
            
    lpt_indices = sorted(txn_lengths.keys(), key=lambda k: txn_lengths[k], reverse=True)
    
    # 1. Beam Search Construction
    BEAM_WIDTH = max(16, int(num_seqs * 2.2))
    
    # Initial Beam
    # Seed with top LPT transactions to ensure heavy items are handled early
    seeds = set(lpt_indices[:BEAM_WIDTH])
    if len(seeds) < BEAM_WIDTH:
        rem_slots = BEAM_WIDTH - len(seeds)
        seeds.update(random.sample(range(num_txns), min(num_txns - len(seeds), rem_slots)))
        
    beam = []
    # Start Gamma high to encourage work density
    start_gamma = 1.6
    
    for t in seeds:
        seq = [t]
        cost = workload.get_opt_seq_cost(seq)
        acc_work = txn_lengths[t]
        score = cost - (start_gamma * acc_work)
        beam.append({
            'seq': seq,
            'cost': cost,
            'score': score,
            'work': acc_work,
            'rem': set(range(num_txns)) - {t}
        })
        
    beam.sort(key=lambda x: x['score'])
    beam = beam[:BEAM_WIDTH]
    
    # Construction Loop
    for step in range(num_txns - 1):
        # Gamma Decay: 1.6 -> 1.0
        # Slowly transition from "pack as much work as possible" to "minimize makespan increase"
        progress = step / max(1, num_txns - 1)
        curr_gamma = 1.6 - (0.6 * progress)
        
        candidates = []
        for parent in beam:
            if not parent['rem']: continue
            
            # Candidate selection: Mix of Heuristic (LPT) and Random
            to_eval = set()
            rem_list = list(parent['rem'])
            
            # Top LPT available
            added_lpt = 0
            for t in lpt_indices:
                if t in parent['rem']:
                    to_eval.add(t)
                    added_lpt += 1
                    if added_lpt >= 5: break
            
            # Random samples for diversity
            needed = 5
            pool = [x for x in rem_list if x not in to_eval]
            if len(pool) > 0:
                count = min(len(pool), needed)
                to_eval.update(random.sample(pool, count))
                
            p_cost = parent['cost']
            p_work = parent['work']
            
            for t in to_eval:
                new_seq = parent['seq'] + [t]
                new_cost = workload.get_opt_seq_cost(new_seq)
                new_work = p_work + txn_lengths[t]
                
                delta_cost = new_cost - p_cost
                t_len = txn_lengths[t]
                
                # Base Score calculation (Minimize this)
                base_score = new_cost - (curr_gamma * new_work)
                
                # Continuous Efficiency Bonus
                # Instead of a binary zero-cost check, we use a continuous gradient.
                # If delta_cost is significantly smaller than the transaction length,
                # it means we are hiding work (parallelism).
                bonus = 0.0
                threshold = 0.4 * t_len
                if delta_cost < threshold:
                    # Linearly scale bonus: Max when delta=0, Zero when delta=threshold
                    quality = 1.0 - (delta_cost / max(1e-9, threshold))
                    bonus = 2.5 * t_len * quality
                
                final_score = base_score - bonus
                
                new_rem = parent['rem'].copy()
                new_rem.remove(t)
                
                candidates.append({
                    'seq': new_seq,
                    'cost': new_cost,
                    'score': final_score,
                    'work': new_work,
                    'rem': new_rem
                })
        
        if not candidates: break
        
        # Diversity Selection
        candidates.sort(key=lambda x: x['score'])
        
        # Keep top fraction purely by score
        k_elite = int(BEAM_WIDTH * 0.6)
        next_beam = candidates[:k_elite]
        
        # Sample remaining from top 3x pool to maintain diversity
        if len(candidates) > k_elite:
            pool = candidates[k_elite : min(len(candidates), BEAM_WIDTH * 3)]
            needed = BEAM_WIDTH - len(next_beam)
            if len(pool) <= needed:
                next_beam.extend(pool)
            else:
                next_beam.extend(random.sample(pool, needed))
        
        beam = next_beam

    # Best from construction
    beam.sort(key=lambda x: x['cost'])
    current_schedule = beam[0]['seq']
    current_cost = beam[0]['cost']
    
    # 2. Refinement Phase: ILS with Deterministic Polish
    
    best_schedule = list(current_schedule)
    best_cost = current_cost
    
    # Adjust cycles based on workload size
    ILS_CYCLES = 5 if num_txns > 20 else 3
    
    for cycle in range(ILS_CYCLES):
        # A. Ruin (Multi-block)
        work_seq = list(current_schedule)
        removed = []
        
        # Bias towards removing 2 disjoint blocks to allow shuffling (70%)
        # vs 1 contiguous block for local reordering (30%)
        if random.random() < 0.7 and num_txns > 10:
            # 2 blocks
            total_remove = max(4, int(num_txns * 0.15))
            b1 = total_remove // 2
            b2 = total_remove - b1
            
            # Block 1
            if len(work_seq) > b1:
                i1 = random.randint(0, len(work_seq) - b1)
                removed.extend(work_seq[i1 : i1 + b1])
                del work_seq[i1 : i1 + b1]
            
            # Block 2
            if len(work_seq) > b2:
                i2 = random.randint(0, len(work_seq) - b2)
                removed.extend(work_seq[i2 : i2 + b2])
                del work_seq[i2 : i2 + b2]
        else:
            # 1 block
            bs = max(2, int(num_txns * 0.20))
            if len(work_seq) > bs:
                si = random.randint(0, len(work_seq) - bs)
                removed.extend(work_seq[si : si + bs])
                del work_seq[si : si + bs]
                
        random.shuffle(removed)
        
        # B. Recreate (Greedy Best Fit)
        for txn in removed:
            best_pos = -1
            best_incr = float('inf')
            
            for p in range(len(work_seq) + 1):
                work_seq.insert(p, txn)
                c = workload.get_opt_seq_cost(work_seq)
                if c < best_incr:
                    best_incr = c
                    best_pos = p
                del work_seq[p]
            
            work_seq.insert(best_pos, txn)
            
        current_schedule = work_seq
        current_cost = best_incr
        
        # Check against best
        real_cost = workload.get_opt_seq_cost(current_schedule)
        if real_cost < best_cost:
            best_cost = real_cost
            best_schedule = list(current_schedule)
            current_cost = real_cost
        
        # C. Deterministic Polish (Sampled Best Insertion)
        # Replacing stochastic SA with a deterministic pass improves convergence.
        # We perform a "remove and re-insert at best position" operation on a subset of transactions.
        
        # Sample size: 35% of transactions (or min 10) to keep runtime low while optimizing structure
        polish_size = max(10, int(num_txns * 0.35))
        # Safely sample indices
        if len(current_schedule) > 0:
            polish_indices = random.sample(range(len(current_schedule)), min(len(current_schedule), polish_size))
            
            # Identify transactions (by value) to polish
            polish_txns = [current_schedule[i] for i in polish_indices]
            
            improved_polish = False
            
            for txn in polish_txns:
                # Find current index
                try:
                    curr_idx = current_schedule.index(txn)
                except ValueError:
                    continue 
                    
                # Temporarily remove
                current_schedule.pop(curr_idx)
                
                # Scan all positions for best insertion
                best_pos = -1
                min_c = float('inf')
                
                for p in range(len(current_schedule) + 1):
                    current_schedule.insert(p, txn)
                    c = workload.get_opt_seq_cost(current_schedule)
                    if c < min_c:
                        min_c = c
                        best_pos = p
                    current_schedule.pop(p)
                
                # Insert at best found position
                current_schedule.insert(best_pos, txn)
                
                # Track improvement
                if min_c < current_cost:
                    current_cost = min_c
                    improved_polish = True
            
            if improved_polish:
                if current_cost < best_cost:
                    best_cost = current_cost
                    best_schedule = list(current_schedule)
                
        # D. Restart (occasional)
        # If we drift too far into high cost without improvement, reset to best
        if current_cost > best_cost and random.random() < 0.2:
            current_schedule = list(best_schedule)
            current_cost = best_cost

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