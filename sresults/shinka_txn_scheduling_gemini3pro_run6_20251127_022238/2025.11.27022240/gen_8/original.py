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
    Get optimal schedule using Beam Search with heuristic tie-breaking.
    
    Args:
        workload: Workload object containing transaction data
        num_seqs: Used to determine the beam width (effort level)

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    # Beam Search Parameters
    # Beam width determines how many parallel paths we explore. 
    # Using max(5, num_seqs) ensures reasonable exploration even if num_seqs is low.
    BEAM_WIDTH = max(5, num_seqs)
    SAMPLES_PER_NODE = 16          # Number of candidates to evaluate per beam node
    MAX_CHILDREN = 3               # Diversity constraint: max paths from one parent
    
    num_txns = workload.num_txns
    
    # Precompute transaction durations (proxy: cost of running txn alone)
    # Used for tie-breaking: if costs are equal, prefer longer transactions to fill gaps.
    txn_durations = {}
    for t in range(num_txns):
        txn_durations[t] = workload.get_opt_seq_cost([t])

    # Initialize Beam with distinct random starts
    start_candidates = list(range(num_txns))
    random.shuffle(start_candidates)
    
    # Beam Item: {'cost': float, 'seq': list, 'rem': set}
    beam = []
    
    # Initialize the beam with single-transaction sequences
    for t in start_candidates[:BEAM_WIDTH]:
        seq = [t]
        cost = txn_durations[t]
        rem = set(range(num_txns))
        rem.remove(t)
        beam.append({'cost': cost, 'seq': seq, 'rem': rem})
    
    # Beam Search Construction Loop
    # Grow sequences one transaction at a time
    for _ in range(num_txns - 1):
        candidates = []
        
        # Expand each node in the current beam
        for p_idx, parent in enumerate(beam):
            parent_rem_list = list(parent['rem'])
            
            # Determine candidates to sample
            if len(parent_rem_list) <= SAMPLES_PER_NODE:
                samples = parent_rem_list
            else:
                samples = random.sample(parent_rem_list, SAMPLES_PER_NODE)
            
            for t in samples:
                new_seq = parent['seq'] + [t]
                
                # Evaluate cost (makespan) - this is the expensive step
                new_cost = workload.get_opt_seq_cost(new_seq)
                
                # Priority Tuple: (Primary: Cost, Secondary: -Duration)
                # We want minimal cost. If costs equal, we want maximal duration (min -Duration)
                priority = (new_cost, -txn_durations[t])
                
                candidates.append({
                    'priority': priority,
                    'cost': new_cost,
                    'seq': new_seq,
                    'added': t,
                    'p_idx': p_idx
                })
        
        # Sort all candidates by priority (lowest cost, then longest txn)
        candidates.sort(key=lambda x: x['priority'])
        
        # Select next beam with diversity constraints
        new_beam = []
        parent_usage = {i: 0 for i in range(len(beam))}
        reserve = []
        
        # First pass: fill beam respecting MAX_CHILDREN to maintain path diversity
        for cand in candidates:
            if len(new_beam) >= BEAM_WIDTH:
                break
                
            p_idx = cand['p_idx']
            if parent_usage[p_idx] < MAX_CHILDREN:
                # Add to beam
                new_rem = beam[p_idx]['rem'].copy()
                new_rem.remove(cand['added'])
                new_beam.append({'cost': cand['cost'], 'seq': cand['seq'], 'rem': new_rem})
                parent_usage[p_idx] += 1
            else:
                reserve.append(cand)
        
        # Second pass: if beam not full, fill from reserve (relax constraint)
        if len(new_beam) < BEAM_WIDTH:
            for cand in reserve:
                if len(new_beam) >= BEAM_WIDTH:
                    break
                p_idx = cand['p_idx']
                new_rem = beam[p_idx]['rem'].copy()
                new_rem.remove(cand['added'])
                new_beam.append({'cost': cand['cost'], 'seq': cand['seq'], 'rem': new_rem})
        
        if not new_beam:
            break
            
        beam = new_beam

    # Find best schedule in final beam
    best_result = min(beam, key=lambda x: x['cost'])
    
    return best_result['cost'], best_result['seq']


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