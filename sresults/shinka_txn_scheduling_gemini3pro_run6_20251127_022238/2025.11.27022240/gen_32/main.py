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
    Get optimal schedule using Lookahead Beam Search and Iterated Local Search.
    
    Args:
        workload: Workload object
        num_seqs: Budget parameter used to scale beam width

    Returns:
        (lowest_makespan, schedule)
    """
    
    num_txns = workload.num_txns
    
    # --- 1. METRIC PRECOMPUTATION ---
    txn_durations = {}
    txn_rw = {}
    
    # Parse Workload
    for t in range(num_txns):
        # Duration
        try:
            d = workload.txns[t][0][3]
        except:
            d = 1.0
        txn_durations[t] = d
        
        # Read/Write Sets
        reads = set()
        writes = set()
        try:
            ops_str = workload.txns[t][0][1]
            if isinstance(ops_str, str):
                for op in ops_str.split():
                    if '-' in op:
                        type_, key = op.split('-')
                        if type_ == 'r': reads.add(key)
                        elif type_ == 'w': writes.add(key)
        except:
            pass
        txn_rw[t] = (reads, writes)

    # Compute Conflict Volume (Blocking Potential)
    txn_conflict_vol = {t: 0.0 for t in range(num_txns)}
    
    for i in range(num_txns):
        r1, w1 = txn_rw[i]
        vol = 0.0
        for j in range(num_txns):
            if i == j: continue
            r2, w2 = txn_rw[j]
            # Conflict: W1 n (W2 u R2) != 0 OR R1 n W2 != 0
            if not w1.isdisjoint(w2) or not w1.isdisjoint(r2) or not r1.isdisjoint(w2):
                vol += txn_durations[j]
        txn_conflict_vol[i] = vol

    # Compute Static Priority
    # High Priority = Long Duration + High Conflict Volume
    max_vol = max(txn_conflict_vol.values()) if txn_conflict_vol else 1.0
    max_dur = max(txn_durations.values()) if txn_durations else 1.0
    if max_vol == 0: max_vol = 1.0
    
    txn_priority = {}
    for t in range(num_txns):
        # Balance Duration and Conflict
        # Conflict is slightly more critical for schedule packing
        p = 0.5 * (txn_durations[t] / max_dur) + 0.5 * (txn_conflict_vol[t] / max_vol)
        txn_priority[t] = p

    # --- 2. LOOKAHEAD BEAM SEARCH ---
    
    # Beam settings
    base_width = max(5, int(num_seqs))
    start_width = int(base_width * 1.5)
    end_width = max(2, int(base_width * 0.6))
    
    # Initial Beam: (cost, schedule, remaining_list)
    beam = [(0, [], list(range(num_txns)))]
    
    for step in range(num_txns):
        # Tapered Width
        progress = step / num_txns
        width = int(start_width + (end_width - start_width) * progress)
        width = max(2, width)
        
        candidates_pool = []
        
        for p_cost, p_sched, p_remain in beam:
            # Candidate Selection
            next_candidates = set()
            
            if len(p_remain) <= 20:
                # Tail Optimization: Check all
                next_candidates.update(p_remain)
            else:
                # Weighted Sampling
                weights = [txn_priority[x] for x in p_remain]
                # Pick top 2 deterministic (Greedy)
                sorted_rem = sorted(zip(p_remain, weights), key=lambda x: x[1], reverse=True)
                next_candidates.add(sorted_rem[0][0])
                if len(sorted_rem) > 1: next_candidates.add(sorted_rem[1][0])
                
                # Sample others
                samples = random.choices(p_remain, weights=weights, k=5)
                next_candidates.update(samples)
            
            # Identify the single highest priority item remaining (for lookahead)
            # We do this once per parent state to save time? 
            # Actually, relative to 'c', 'next_best' is max(remaining - {c}).
            # Pre-calculate sorted remaining for efficiency
            sorted_remain_base = sorted(p_remain, key=lambda x: txn_priority[x], reverse=True)
            
            for cand in next_candidates:
                # Immediate Step
                new_sched = p_sched + [cand]
                
                # Lookahead Step (Selective)
                # Only do lookahead if we aren't at the very end
                cost_metric = 0
                if len(p_remain) > 1:
                    # Find 'next_best' item that isn't 'cand'
                    next_best = sorted_remain_base[0] if sorted_remain_base[0] != cand else sorted_remain_base[1]
                    
                    # Tentative schedule with lookahead
                    lookahead_sched = new_sched + [next_best]
                    cost_metric = workload.get_opt_seq_cost(lookahead_sched)
                else:
                    cost_metric = workload.get_opt_seq_cost(new_sched)
                
                # Secondary metric: actual current cost (if lookahead costs are tied)
                # But to save calls, we might skip calculating exact current cost if lookahead is distinct?
                # Let's just use lookahead cost as primary.
                
                # Tuple: (LookaheadCost, -ConflictVol, -Duration)
                metric = (cost_metric, -txn_conflict_vol[cand], -txn_durations[cand])
                
                new_remain = list(p_remain)
                new_remain.remove(cand)
                
                candidates_pool.append((metric, new_sched, new_remain))
        
        # Prune
        candidates_pool.sort(key=lambda x: x[0])
        # Reconstruct beam (using lookahead cost as the 'cost' carried forward is inaccurate, 
        # but we only use 'cost' in beam for nothing? Wait.
        # The beam loop doesn't use p_cost for calculation, only to pass it along.
        # But we should probably carry the ACTUAL cost if we wanted to use it.
        # However, re-calculating actual cost for the chosen few is cheap.
        
        selected = candidates_pool[:width]
        beam = []
        for m, s, r in selected:
            # We don't strictly need the accurate cost for the next step's selection logic
            # as we re-calculate everything. Just pass m[0] or 0.
            beam.append((m[0], s, r))

    # Best from Beam
    # Note: The cost in beam is the Lookahead cost (Cost of S+1). 
    # The actual schedule 's' has length N. Lookahead on last step is just Cost(S).
    best_state = beam[0]
    current_schedule = best_state[1]
    current_cost = workload.get_opt_seq_cost(current_schedule)

    # --- 3. ITERATED LOCAL SEARCH (ILS) ---
    
    best_global_cost = current_cost
    best_global_schedule = list(current_schedule)
    
    # ILS Parameters
    max_ils_iter = 3
    if num_txns < 50: max_ils_iter = 5
    
    # We work on 'current_schedule'.
    
    for iteration in range(max_ils_iter):
        # A. Local Search (Windowed Insertion)
        improved = True
        while improved:
            improved = False
            # Randomize order of checking to avoid bias
            check_order = list(range(num_txns))
            random.shuffle(check_order)
            
            for i in check_order:
                txn = current_schedule[i]
                
                # Window definition
                window_size = 12
                start = max(0, i - window_size)
                end = min(num_txns, i + window_size + 1)
                
                # Try moving txn to all positions in window
                best_pos = i
                best_val = current_cost
                
                # Remove txn once
                temp_sched = current_schedule[:i] + current_schedule[i+1:]
                
                for pos in range(start, end):
                    # Construct candidate
                    cand = temp_sched[:pos] + [txn] + temp_sched[pos:]
                    c = workload.get_opt_seq_cost(cand)
                    if c < best_val:
                        best_val = c
                        best_pos = pos
                
                if best_val < current_cost:
                    current_cost = best_val
                    current_schedule = current_schedule[:i] + current_schedule[i+1:] # Remove from old
                    current_schedule.insert(best_pos, txn) # Insert at new
                    improved = True
        
        # Check against global best
        if current_cost < best_global_cost:
            best_global_cost = current_cost
            best_global_schedule = list(current_schedule)
        
        # B. Perturbation (Kick)
        # Don't perturb on last iteration
        if iteration < max_ils_iter - 1:
            # Restart from global best to ensure we don't drift into bad areas?
            # Standard ILS perturbs the local optima.
            # Let's perturb the BEST found so far to exploit it.
            current_schedule = list(best_global_schedule)
            current_cost = best_global_cost
            
            # Apply 3 random swaps
            for _ in range(3):
                idx1 = random.randint(0, num_txns - 1)
                idx2 = random.randint(0, num_txns - 1)
                if idx1 != idx2:
                    current_schedule[idx1], current_schedule[idx2] = current_schedule[idx2], current_schedule[idx1]
            
            # Or move a block
            if random.random() < 0.5:
                # Move a block of 4 items
                blk_size = 4
                if num_txns > blk_size:
                    src = random.randint(0, num_txns - blk_size)
                    block = current_schedule[src:src+blk_size]
                    del current_schedule[src:src+blk_size]
                    dst = random.randint(0, len(current_schedule))
                    current_schedule[dst:dst] = block
            
            current_cost = workload.get_opt_seq_cost(current_schedule)

    return best_global_cost, best_global_schedule


def get_random_costs():
    """Evaluate scheduling algorithm on three different workloads."""
    start_time = time.time()
    
    # Budget parameter
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
