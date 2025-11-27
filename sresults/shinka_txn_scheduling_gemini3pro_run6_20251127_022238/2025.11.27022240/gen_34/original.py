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
    Get optimal schedule using Conflict-Aware Beam Search and Deterministic Windowed Re-insertion.
    
    Args:
        workload: Workload object
        num_seqs: Budget parameter (influences beam width)

    Returns:
        (lowest_makespan, schedule)
    """
    
    num_txns = workload.num_txns

    # --- 1. METRIC PRECOMPUTATION ---
    # Parse transactions to get durations and read/write sets
    txn_durations = {}
    txn_rw_sets = {}
    
    for t in range(num_txns):
        # Extract Duration
        duration = 1.0
        try:
            # txns structure: [ [id, ops_str, ..., duration], ... ]
            duration = workload.txns[t][0][3]
        except:
            pass
        txn_durations[t] = duration
        
        # Extract R/W Sets
        reads = set()
        writes = set()
        try:
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
            pass
        txn_rw_sets[t] = (reads, writes)

    # Compute Conflict Weights
    # Conflict Weight = Sum of durations of all other transactions that conflict with T
    # High conflict weight means T is "contentious" and should be handled carefully (likely early).
    txn_priorities = {}
    
    for i in range(num_txns):
        r1, w1 = txn_rw_sets[i]
        conflict_weight = 0.0
        
        # Check against all others
        for j in range(num_txns):
            if i == j: continue
            r2, w2 = txn_rw_sets[j]
            
            # Intersection logic: (W1 & W2) or (W1 & R2) or (R1 & W2)
            if not w1.isdisjoint(w2) or not w1.isdisjoint(r2) or not r1.isdisjoint(w2):
                conflict_weight += txn_durations[j]
        
        # Priority: Duration is important, but Conflict is also important.
        # We combine them. A long transaction with many conflicts is a critical path candidate.
        txn_priorities[i] = txn_durations[i] + (0.5 * conflict_weight)

    # --- 2. BEAM SEARCH CONSTRUCTION ---
    
    BEAM_WIDTH = max(8, int(num_seqs * 1.2))
    
    # Beam State: (cost, schedule_list, remaining_list)
    beam = [(0, [], list(range(num_txns)))]
    
    for _ in range(num_txns):
        candidates_pool = []
        
        for p_cost, p_sched, p_remain in beam:
            
            # Candidate Selection
            # 1. Always consider top priority items (LPT/Most Conflicting)
            # 2. Random sampling for diversity
            
            # Sort remaining by priority descending
            sorted_remain = sorted(p_remain, key=lambda x: txn_priorities[x], reverse=True)
            
            next_candidates = set()
            
            # Take top K heaviest
            k_heavy = min(len(sorted_remain), 5)
            next_candidates.update(sorted_remain[:k_heavy])
            
            # Take random samples
            if len(sorted_remain) > k_heavy:
                next_candidates.update(random.sample(sorted_remain[k_heavy:], min(3, len(sorted_remain)-k_heavy)))
            
            # Evaluate
            for cand in next_candidates:
                new_sched = p_sched + [cand]
                cost = workload.get_opt_seq_cost(new_sched)
                
                # Zero-Cost heuristic: If cost didn't increase, this is a perfect fit.
                # Give it a massive bonus in the sorting metric.
                cost_increase = cost - p_cost
                
                # Metric: (Cost, -Priority)
                # We want minimal cost. Tie-break with higher priority (schedule heavy stuff early).
                metric = (cost, -txn_priorities[cand])
                
                # Bonus for 0 increase (effectively reduces cost in sorting)
                if cost_increase <= 1e-9:
                    metric = (cost - 0.1, -txn_priorities[cand]) # Artificial boost
                
                new_remain = list(p_remain)
                new_remain.remove(cand)
                
                candidates_pool.append((metric, new_sched, new_remain))
        
        # Pruning
        candidates_pool.sort(key=lambda x: x[0])
        beam = [(x[0][0], x[1], x[2]) for x in candidates_pool[:BEAM_WIDTH]]

    # Extract best
    best_state = beam[0]
    current_cost = best_state[0]
    current_schedule = best_state[1]
    
    # --- 3. REFINEMENT: WINDOWED RE-INSERTION ---
    # Iteratively remove each transaction and find the best insertion point within a local window.
    # This is more robust than random sampling.
    
    # Number of passes over the entire schedule
    num_passes = 2
    
    for _ in range(num_passes):
        improved_in_pass = False
        
        # We iterate through transactions.
        # Shuffling helps prevent getting stuck in cycle dependencies based on original index order.
        txns_to_check = list(current_schedule)
        # Sort by priority? Or random? Random is usually safer for local search to avoid bias loops.
        random.shuffle(txns_to_check) 
        
        for txn in txns_to_check:
            # Find current position
            curr_idx = current_schedule.index(txn)
            
            # Remove txn
            # Slicing is O(N), acceptable for N=100
            temp_schedule = current_schedule[:curr_idx] + current_schedule[curr_idx+1:]
            
            # Define Search Window
            # We don't need to check the whole array, just a neighborhood.
            # However, conflicts can be far apart. A window of +/- 12 covers significant range (~25% of N=100)
            window_radius = 12
            start_pos = max(0, curr_idx - window_radius)
            end_pos = min(len(temp_schedule) + 1, curr_idx + window_radius + 1)
            
            best_pos = curr_idx
            best_local_cost = current_cost
            
            # Check all positions in window
            for pos in range(start_pos, end_pos):
                # Construct candidate
                # Optimization: In a real heavy loop we'd be careful with allocs, 
                # but Python lists are optimized enough for this scale.
                cand_sched = temp_schedule[:pos] + [txn] + temp_schedule[pos:]
                
                c = workload.get_opt_seq_cost(cand_sched)
                
                if c < best_local_cost:
                    best_local_cost = c
                    best_pos = pos
                elif c == best_local_cost:
                    # Tie-breaker: If cost is same, prefer position closer to original?
                    # Or random? No, keep stable.
                    pass
            
            # Update if we found a better spot
            if best_local_cost < current_cost:
                current_cost = best_local_cost
                # Reconstruct schedule with new pos
                current_schedule = temp_schedule[:best_pos] + [txn] + temp_schedule[best_pos:]
                improved_in_pass = True
        
        if not improved_in_pass:
            break

    # --- 4. FINAL PERTURBATION CHECK ---
    # Quick micro-optimization: Swap adjacent pairs if it helps
    # This catches small ordering errors like A-B vs B-A
    for i in range(num_txns - 1):
        # Swap i and i+1
        current_schedule[i], current_schedule[i+1] = current_schedule[i+1], current_schedule[i]
        c = workload.get_opt_seq_cost(current_schedule)
        if c < current_cost:
            current_cost = c
        else:
            # Swap back
            current_schedule[i], current_schedule[i+1] = current_schedule[i+1], current_schedule[i]

    return current_cost, current_schedule


def get_random_costs():
    """Evaluate scheduling algorithm on three different workloads."""
    start_time = time.time()
    
    # Beam budget
    num_seqs = 10
    
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