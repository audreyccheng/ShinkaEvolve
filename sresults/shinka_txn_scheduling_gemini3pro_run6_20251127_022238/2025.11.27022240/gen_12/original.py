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
    Get optimal schedule using Adaptive Greedy Construction followed by Hybrid Local Search.
    
    Strategy:
    1. Generate 'num_seqs' candidate schedules using a randomized greedy approach that 
       switches to exhaustive search near the end of the sequence.
    2. Pick the best candidate.
    3. Refine it using a local search with both Swap and Shift operators.

    Args:
        workload: Workload object containing transaction data
        num_seqs: Number of initial greedy candidates to generate

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    
    # Pre-calculate transaction costs for heuristic (Longest Processing Time)
    txn_costs = {}
    try:
        for t in range(workload.num_txns):
            txn_costs[t] = workload.txns[t][0][3]
    except (IndexError, AttributeError, TypeError):
        for t in range(workload.num_txns):
            txn_costs[t] = 1

    def generate_adaptive_greedy_schedule():
        """
        Generates a schedule. Uses sampling for the bulk of the schedule
        and exhaustive search for the tail to ensure tight packing.
        """
        remaining = list(range(workload.num_txns))
        schedule = []
        
        # Start with a random transaction for diversity
        start_idx = random.randint(0, len(remaining) - 1)
        schedule.append(remaining.pop(start_idx))
        
        while remaining:
            candidates = []
            
            # ADAPTIVE LOGIC:
            # If few transactions remain, checking all of them is cheap and valuable
            # to minimize gaps at the end of the schedule.
            exhaustive_threshold = 15
            
            if len(remaining) <= exhaustive_threshold:
                candidates = list(remaining)
            else:
                # Random sampling
                sample_k = 6
                candidates = random.sample(remaining, min(len(remaining), sample_k))
                
                # Heuristic bias: Add heaviest remaining transactions
                # These are "big rocks" that are harder to place later
                bias_k = 4
                heaviest = sorted(remaining, key=lambda x: txn_costs.get(x, 0), reverse=True)[:bias_k]
                for h in heaviest:
                    if h not in candidates:
                        candidates.append(h)
            
            # Select best candidate based on immediate makespan
            best_c = -1
            best_c_cost = float('inf')
            
            for c in candidates:
                # Appending to a list and calculating cost
                # Note: get_opt_seq_cost computes total makespan of the sequence
                test_seq = schedule + [c]
                cost = workload.get_opt_seq_cost(test_seq)
                
                if cost < best_c_cost:
                    best_c_cost = cost
                    best_c = c
            
            if best_c != -1:
                schedule.append(best_c)
                remaining.remove(best_c)
            else:
                # Fallback (should not happen with correct logic)
                c = remaining.pop(0)
                schedule.append(c)
                
        return workload.get_opt_seq_cost(schedule), schedule

    # Phase 1: Filter - Generate multiple candidates and keep the best
    candidates = []
    iterations = max(1, num_seqs)
    
    for _ in range(iterations):
        cost, sched = generate_adaptive_greedy_schedule()
        candidates.append((cost, sched))
    
    # Sort by cost ascending
    candidates.sort(key=lambda x: x[0])
    best_cost, best_schedule = candidates[0]

    # Phase 2: Intensify - Hybrid Local Search on the best candidate
    # Use a mix of Shift (Insert) and Swap operators
    
    current_schedule = list(best_schedule)
    current_cost = best_cost
    
    search_steps = 500  # Budget for local search
    no_improv_limit = 150
    no_improv = 0
    
    for _ in range(search_steps):
        if no_improv >= no_improv_limit:
            break
            
        neighbor = list(current_schedule)
        
        # 50% chance to Shift, 50% chance to Swap
        if random.random() < 0.5:
            # Shift Operator: Move transaction from src to dst
            src = random.randint(0, len(neighbor) - 1)
            dst = random.randint(0, len(neighbor) - 1)
            if src == dst:
                continue
            txn = neighbor.pop(src)
            neighbor.insert(dst, txn)
        else:
            # Swap Operator: Exchange two transactions
            i = random.randint(0, len(neighbor) - 1)
            j = random.randint(0, len(neighbor) - 1)
            if i == j:
                continue
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            
        new_cost = workload.get_opt_seq_cost(neighbor)
        
        # Strict improvement to converge to local optimum
        if new_cost < current_cost:
            current_cost = new_cost
            current_schedule = neighbor
            no_improv = 0
        else:
            no_improv += 1
            
    return current_cost, current_schedule


def get_random_costs():
    """
    Evaluate scheduling algorithm on three different workloads.
    
    Returns:
        Tuple of (total_makespan, list_of_schedules, execution_time)
    """
    start_time = time.time()
    
    # Number of initial greedy candidates to generate
    # 10 provides a good diversity of starting points
    num_seqs = 10
    
    # Workload 1: Complex mixed read/write transactions
    workload1 = Workload(WORKLOAD_1)
    makespan1, schedule1 = get_best_schedule(workload1, num_seqs)

    # Workload 2: Simple read-then-write pattern
    workload2 = Workload(WORKLOAD_2)
    makespan2, schedule2 = get_best_schedule(workload2, num_seqs)

    # Workload 3: Minimal read/write operations
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
