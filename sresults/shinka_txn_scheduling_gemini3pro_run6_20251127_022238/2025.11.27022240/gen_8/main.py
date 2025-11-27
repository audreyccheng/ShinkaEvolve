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
    Get optimal schedule using Smart Beam Search followed by Local Search refinement.
    
    Args:
        workload: Workload object containing transaction data
        num_seqs: Used to scale the beam width

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    # --- Parameters ---
    # Beam width: Number of parallel paths to explore
    BEAM_WIDTH = max(4, num_seqs) 
    
    # Sampling: How many candidates to check per node.
    # Higher = better decisions but slower.
    SAMPLES_PER_NODE = 20
    
    # Local Search: Iterations to refine the final result
    LOCAL_SEARCH_ITERS = 30

    num_txns = workload.num_txns
    
    # --- Precompute Heuristics ---
    # Calculate standalone duration for each transaction.
    # Used for sampling heuristics (LPT/SPT) and tie-breaking.
    txn_durations = {}
    for t in range(num_txns):
        txn_durations[t] = workload.get_opt_seq_cost([t])
        
    # Sorted lists for heuristics
    # Sort by duration descending
    txns_by_duration_desc = sorted(range(num_txns), key=lambda t: txn_durations[t], reverse=True)

    # --- Beam Search ---
    
    # Initial Beam: Start with single transactions
    # We pick the top K longest transactions as starts, plus some randoms
    start_candidates = set()
    start_candidates.update(txns_by_duration_desc[:BEAM_WIDTH]) # Try starting with heavy hitters
    while len(start_candidates) < BEAM_WIDTH * 2:
        start_candidates.add(random.randint(0, num_txns - 1))
    
    beam = []
    for t in list(start_candidates)[:BEAM_WIDTH]:
        rem = set(range(num_txns))
        rem.remove(t)
        beam.append({
            'cost': txn_durations[t],
            'seq': [t],
            'rem': rem
        })

    # Construct schedules
    for step in range(num_txns - 1):
        candidates = []
        
        for p_idx, parent in enumerate(beam):
            parent_rem = list(parent['rem'])
            
            # Smart Sampling of Candidates
            if len(parent_rem) <= SAMPLES_PER_NODE:
                samples = parent_rem
            else:
                samples = set()
                # 1. Heuristic: Always consider the longest remaining transaction
                # (Scheduling bottleneck early often helps)
                curr_longest = -1
                max_dur = -1
                for t in parent_rem:
                    if txn_durations[t] > max_dur:
                        max_dur = txn_durations[t]
                        curr_longest = t
                if curr_longest != -1:
                    samples.add(curr_longest)
                
                # 2. Heuristic: Consider shortest (good for filling gaps)
                curr_shortest = -1
                min_dur = float('inf')
                for t in parent_rem:
                    if txn_durations[t] < min_dur:
                        min_dur = txn_durations[t]
                        curr_shortest = t
                if curr_shortest != -1:
                    samples.add(curr_shortest)
                    
                # 3. Random Fill
                while len(samples) < SAMPLES_PER_NODE:
                    samples.add(random.choice(parent_rem))
                
                samples = list(samples)

            # Evaluate candidates
            for t in samples:
                new_seq = parent['seq'] + [t]
                cost = workload.get_opt_seq_cost(new_seq)
                
                # Priority: Minimize Cost, Break ties with Max Duration (prefer filling with large items)
                priority = (cost, -txn_durations[t])
                
                candidates.append({
                    'priority': priority,
                    'cost': cost,
                    'seq': new_seq,
                    'added': t,
                    'p_idx': p_idx
                })
        
        # Select best candidates for next beam
        candidates.sort(key=lambda x: x['priority'])
        
        new_beam = []
        parent_usage = {i: 0 for i in range(len(beam))}
        
        # Diversity constraint: Don't let one parent dominate the beam
        # Max children per parent = BEAM_WIDTH / 2 (approx)
        max_children = max(2, BEAM_WIDTH // 2)
        
        for cand in candidates:
            if len(new_beam) >= BEAM_WIDTH:
                break
            
            if parent_usage[cand['p_idx']] < max_children:
                new_rem = beam[cand['p_idx']]['rem'].copy()
                new_rem.remove(cand['added'])
                new_beam.append({
                    'cost': cand['cost'],
                    'seq': cand['seq'],
                    'rem': new_rem
                })
                parent_usage[cand['p_idx']] += 1
                
        # Fill rest if needed (relax constraint)
        if len(new_beam) < BEAM_WIDTH:
            for cand in candidates:
                if len(new_beam) >= BEAM_WIDTH:
                    break
                # Check if this exact seq is already in (via p_idx check implicitly, but different branch)
                # Just check if we skipped it due to parent usage
                if parent_usage[cand['p_idx']] >= max_children:
                     new_rem = beam[cand['p_idx']]['rem'].copy()
                     new_rem.remove(cand['added'])
                     new_beam.append({
                        'cost': cand['cost'],
                        'seq': cand['seq'],
                        'rem': new_rem
                    })

        beam = new_beam

    # Get best from Beam
    best_beam_node = min(beam, key=lambda x: x['cost'])
    best_schedule = best_beam_node['seq']
    min_makespan = best_beam_node['cost']

    # --- Local Search Refinement ---
    # Try to improve the schedule by moving transactions around
    # This helps escape local optima from the greedy construction
    
    current_seq = list(best_schedule)
    current_cost = min_makespan
    
    for _ in range(LOCAL_SEARCH_ITERS):
        # Pick a random transaction to move
        idx_from = random.randint(0, num_txns - 1)
        val = current_seq.pop(idx_from)
        
        # Pick a random new position
        idx_to = random.randint(0, num_txns - 1) # Insert acts before index, so 0..len is valid
        current_seq.insert(idx_to, val)
        
        new_cost = workload.get_opt_seq_cost(current_seq)
        
        if new_cost < current_cost:
            current_cost = new_cost
            # Keep the change
        else:
            # Revert the change
            current_seq.pop(idx_to)
            current_seq.insert(idx_from, val)

    return current_cost, current_seq


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

    # Workload 2: Simple read-then-write pattern
    workload2 = Workload(WORKLOAD_2)
    makespan2, schedule2 = get_best_schedule(workload2, 10)

    # Workload 3: Minimal read/write operations
    workload3 = Workload(WORKLOAD_3)
    makespan3, schedule3 = get_best_schedule(workload3, 10)
    
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