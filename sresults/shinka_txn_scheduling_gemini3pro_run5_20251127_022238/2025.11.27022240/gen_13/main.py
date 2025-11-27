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
    Get optimal schedule using Diversity-Aware Exhaustive Beam Search followed by 
    Simulated Annealing with Block Moves and Reheating.

    Args:
        workload: Workload object containing transaction data
        num_seqs: Parameter affecting computational budget

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    # --- Hyperparameters ---
    # Beam Search
    BEAM_WIDTH = max(4, int(num_seqs * 0.6))
    MAX_CHILDREN_PER_PARENT = 2 # Diversity constraint

    # Simulated Annealing
    SA_ITERATIONS = 5000
    SA_COOLING_RATE = 0.995
    SA_REHEAT_THRESHOLD = 300
    
    # Cost Cache
    cost_cache = {}

    def get_cost(seq):
        t_seq = tuple(seq)
        if t_seq in cost_cache:
            return cost_cache[t_seq]
        c = workload.get_opt_seq_cost(seq)
        cost_cache[t_seq] = c
        return c

    # --- Phase 1: Exhaustive Beam Search with Diversity ---
    # Initialize beam
    initial_candidates = []
    for t in range(workload.num_txns):
        seq = [t]
        cost = get_cost(seq)
        initial_candidates.append({
            'cost': cost,
            'seq': seq,
            'rem': set(range(workload.num_txns)) - {t}
        })
    
    # Prune initial
    initial_candidates.sort(key=lambda x: x['cost'])
    beam = initial_candidates[:BEAM_WIDTH]

    # Construction loop
    for _ in range(workload.num_txns - 1):
        all_candidates = []

        # Exhaustively expand each beam node
        for parent_idx, node in enumerate(beam):
            parent_seq = node['seq']
            parent_rem = node['rem']

            for t in parent_rem:
                new_seq = parent_seq + [t]
                cost = get_cost(new_seq)
                all_candidates.append({
                    'cost': cost,
                    'seq': new_seq,
                    'rem_item': t,
                    'parent_rem': parent_rem,
                    'parent_idx': parent_idx
                })

        # Sort by cost
        all_candidates.sort(key=lambda x: x['cost'])

        # Select next beam with diversity constraint
        new_beam = []
        parent_counts = {} # Track children per parent
        added_signatures = set()

        for cand in all_candidates:
            if len(new_beam) >= BEAM_WIDTH:
                break
            
            p_idx = cand['parent_idx']
            count = parent_counts.get(p_idx, 0)
            
            if count < MAX_CHILDREN_PER_PARENT:
                new_rem = cand['parent_rem'].copy()
                new_rem.remove(cand['rem_item'])
                
                new_beam.append({
                    'cost': cand['cost'],
                    'seq': cand['seq'],
                    'rem': new_rem
                })
                parent_counts[p_idx] = count + 1
                added_signatures.add(tuple(cand['seq']))

        # Fallback: if we didn't fill beam due to constraints, fill with next best valid
        if len(new_beam) < BEAM_WIDTH:
            for cand in all_candidates:
                if len(new_beam) >= BEAM_WIDTH:
                    break
                
                sig = tuple(cand['seq'])
                if sig not in added_signatures:
                    new_rem = cand['parent_rem'].copy()
                    new_rem.remove(cand['rem_item'])
                    new_beam.append({
                        'cost': cand['cost'],
                        'seq': cand['seq'],
                        'rem': new_rem
                    })
                    added_signatures.add(sig)

        beam = new_beam

    if not beam:
        return float('inf'), []

    current_schedule = beam[0]['seq']
    current_cost = beam[0]['cost']

    # --- Phase 2: Simulated Annealing with Block Moves ---
    best_schedule = list(current_schedule)
    best_cost = current_cost

    T_max = current_cost * 0.1
    T = T_max
    T_min = 0.001
    stagnant_steps = 0

    for _ in range(SA_ITERATIONS):
        neighbor = list(current_schedule)
        n = len(neighbor)
        
        # Operator Selection
        op = random.random()
        
        if op < 0.35:
            # Insert (Single Move) - Fine tuning
            i = random.randint(0, n - 1)
            val = neighbor.pop(i)
            j = random.randint(0, n - 1)
            neighbor.insert(j, val)
            
        elif op < 0.65:
            # Block Move - Structural change
            # Move a block of size 2-6
            b_size = random.randint(2, 6)
            if n > b_size:
                i = random.randint(0, n - b_size)
                block = neighbor[i:i+b_size]
                del neighbor[i:i+b_size]
                j = random.randint(0, len(neighbor)) # new length
                neighbor[j:j] = block
                
        elif op < 0.90:
            # Swap - Local exchange
            i, j = random.randint(0, n-1), random.randint(0, n-1)
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            
        else:
            # Reverse - Inversion
            i, j = random.randint(0, n-1), random.randint(0, n-1)
            start, end = min(i, j), max(i, j)
            neighbor[start:end+1] = neighbor[start:end+1][::-1]

        new_cost = get_cost(neighbor)
        delta = new_cost - current_cost

        # Acceptance
        accept = False
        if delta < 0:
            accept = True
        elif T > 1e-9:
            if random.random() < math.exp(-delta / T):
                accept = True

        if accept:
            current_schedule = neighbor
            current_cost = new_cost
            if current_cost < best_cost:
                best_cost = current_cost
                best_schedule = list(current_schedule)
                stagnant_steps = 0
            else:
                stagnant_steps += 1
        else:
            stagnant_steps += 1

        # Cooling
        T *= SA_COOLING_RATE

        # Reheating
        if stagnant_steps > SA_REHEAT_THRESHOLD:
            T = T_max * 0.4
            stagnant_steps = 0
            # Diversity injection: restart from best or just heat up current
            if random.random() < 0.5:
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
    workload_size = 100

    # Workload 1: Complex mixed read/write transactions
    workload = Workload(WORKLOAD_1)
    makespan1, schedule1 = get_best_schedule(workload, 10)
    cost1 = workload.get_opt_seq_cost(schedule1)

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