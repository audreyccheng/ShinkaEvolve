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
    Get optimal schedule using Quadratic Efficiency Beam Search, 
    Adaptive SA with Hybrid Ruin, and Deep Alternating Polish.

    Args:
        workload: Workload object containing transaction data
        num_seqs: Parameter affecting the computational budget

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    # --- Hyperparameters ---
    BEAM_WIDTH = 6
    
    # SA Parameters
    # Moderate iterations to leave budget for extensive polishing
    SA_ITERATIONS = 2000
    SA_COOLING_RATE = 0.998
    SA_START_TEMP_RATIO = 0.05
    
    # Ruin Parameters
    STAGNATION_LIMIT = 150
    RUIN_BLOCK_MIN = 2
    RUIN_BLOCK_MAX = 6
    
    # Polish Parameters
    # High pass count to ensure convergence
    POLISH_MAX_PASSES = 30
    
    # --- Cost Cache ---
    cost_cache = {}

    def get_cost(seq):
        t_seq = tuple(seq)
        if t_seq in cost_cache:
            return cost_cache[t_seq]
        c = workload.get_opt_seq_cost(seq)
        cost_cache[t_seq] = c
        return c

    # --- Pre-calculation: Transaction Weights ---
    txn_weights = {}
    for t in range(workload.num_txns):
        txn_weights[t] = get_cost([t])

    # --- Phase 1: Quadratic Efficiency Beam Search ---
    # Prioritizes placing transactions that "hide" their latency well (high efficiency).
    # Gamma decays to shift focus from local packing efficiency to global makespan minimization.
    GAMMA_START = 4.0
    GAMMA_END = 0.5
    
    candidates = []
    for t in range(workload.num_txns):
        seq = [t]
        c = get_cost(seq)
        w = txn_weights[t]
        # Initial score: Prefer heavier anchors (high w) which reduce score more
        score = c - (0.01 * w)
        candidates.append({
            'cost': c, 
            'score': score, 
            'seq': seq, 
            'rem': {x for x in range(workload.num_txns) if x != t}
        })
        
    candidates.sort(key=lambda x: x['score'])
    beam = candidates[:BEAM_WIDTH]
    
    total_txns = workload.num_txns
    
    for _ in range(total_txns - 1):
        next_candidates = []
        
        # Dynamic Gamma Calculation
        current_len = len(beam[0]['seq'])
        progress = current_len / total_txns
        gamma = GAMMA_START * (1.0 - progress) + GAMMA_END * progress
        
        for node in beam:
            p_seq = node['seq']
            p_rem = node['rem']
            p_cost = node['cost']
            
            for cand in p_rem:
                new_seq = p_seq + [cand]
                new_cost = get_cost(new_seq)
                
                delta = new_cost - p_cost
                w = txn_weights[cand]
                
                # Efficiency Calculation: 
                # Fraction of transaction weight absorbed by parallelism.
                # If delta (cost increase) is 0, efficiency is 1.0 (fully hidden).
                # If delta >= weight, efficiency is 0.0.
                efficiency = 0.0
                if w > 1e-9:
                    eff_val = w - delta
                    if eff_val > 0:
                        efficiency = eff_val / w
                
                # Quadratic Bonus: Strongly rewards high efficiency (perfect packing)
                bonus = gamma * w * (efficiency ** 2)
                score = new_cost - bonus
                
                next_candidates.append((score, new_cost, new_seq, p_rem, cand))
        
        # Pruning
        next_candidates.sort(key=lambda x: x[0])
        
        new_beam = []
        for c_score, c_cost, c_seq, c_parent_rem, c_cand in next_candidates:
            if len(new_beam) >= BEAM_WIDTH:
                break
            new_rem = c_parent_rem.copy()
            new_rem.remove(c_cand)
            new_beam.append({'cost': c_cost, 'seq': c_seq, 'rem': new_rem})
            
        beam = new_beam

    if not beam:
        return float('inf'), []

    current_schedule = beam[0]['seq']
    current_cost = beam[0]['cost']

    # --- Phase 2: Adaptive SA with Hybrid Ruin ---
    best_schedule = list(current_schedule)
    best_cost = current_cost
    
    OP_WEIGHTS = {'swap': 2.0, 'insert': 8.0, 'block_move': 4.0, 'reverse': 1.0}
    OP_MIN = 0.5
    OP_ADAPT = 0.1
    
    T = current_cost * SA_START_TEMP_RATIO
    steps_since_imp = 0
    ops = list(OP_WEIGHTS.keys())
    
    for it in range(SA_ITERATIONS):
        # 1. Hybrid Ruin-and-Recreate on Stagnation
        if steps_since_imp > STAGNATION_LIMIT:
            kick_seq = list(best_schedule)
            n = len(kick_seq)
            removed = []
            
            # 50% chance of Block Ruin (moves chunks), 50% Scatter Ruin (breaks chains)
            if random.random() < 0.5:
                # Multi-Block Ruin
                for _ in range(2):
                    if len(kick_seq) > RUIN_BLOCK_MIN:
                        sz = random.randint(RUIN_BLOCK_MIN, min(len(kick_seq), RUIN_BLOCK_MAX))
                        start = random.randint(0, len(kick_seq) - sz)
                        removed.extend(kick_seq[start : start+sz])
                        del kick_seq[start : start+sz]
            else:
                # Scatter Ruin
                if n > RUIN_BLOCK_MIN:
                    cnt = random.randint(3, 8)
                    idxs = sorted(random.sample(range(n), cnt), reverse=True)
                    for idx in idxs:
                        removed.append(kick_seq.pop(idx))
                        
            random.shuffle(removed)
            
            # Recreate: Greedy Best-Fit
            for item in removed:
                best_pos = -1
                min_c = float('inf')
                for i in range(len(kick_seq)+1):
                    kick_seq.insert(i, item)
                    c = get_cost(kick_seq)
                    if c < min_c:
                        min_c = c
                        best_pos = i
                    kick_seq.pop(i)
                kick_seq.insert(best_pos, item)
            
            current_schedule = kick_seq
            current_cost = min_c
            
            if current_cost < best_cost:
                best_cost = current_cost
                best_schedule = list(current_schedule)
                steps_since_imp = 0
            else:
                steps_since_imp = 0 # Reset to explore from new state
            
            T = max(T, current_cost * 0.05)
            continue
            
        # 2. Adaptive Operator Selection
        total_w = sum(OP_WEIGHTS.values())
        r = random.uniform(0, total_w)
        cum = 0
        op = ops[0]
        for o in ops:
            cum += OP_WEIGHTS[o]
            if r <= cum:
                op = o
                break
                
        neighbor = list(current_schedule)
        n = len(neighbor)
        
        if op == 'swap':
            i, j = random.sample(range(n), 2)
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        elif op == 'insert':
            i = random.randint(0, n-1)
            val = neighbor.pop(i)
            j = random.randint(0, n-1)
            neighbor.insert(j, val)
        elif op == 'reverse':
            i, j = sorted(random.sample(range(n), 2))
            neighbor[i:j+1] = neighbor[i:j+1][::-1]
        elif op == 'block_move':
            b = random.randint(2, 6)
            if n > b:
                i = random.randint(0, n-b)
                block = neighbor[i:i+b]
                del neighbor[i:i+b]
                j = random.randint(0, len(neighbor))
                neighbor[j:j] = block
        
        new_cost = get_cost(neighbor)
        delta = new_cost - current_cost
        
        # 3. Acceptance Criteria
        accept = False
        if delta < 0:
            accept = True
            if new_cost < best_cost:
                best_cost = new_cost
                best_schedule = list(neighbor)
                steps_since_imp = 0
            else:
                steps_since_imp += 1
        else:
            steps_since_imp += 1
            if T > 1e-9 and random.random() < math.exp(-delta/T):
                accept = True
                
        if accept:
            current_schedule = neighbor
            current_cost = new_cost
            
        # 4. Adapt Weights
        reward = 0.1
        if delta < 0:
            reward = 1.0 if new_cost >= best_cost else 2.0
        OP_WEIGHTS[op] = (1 - OP_ADAPT) * OP_WEIGHTS[op] + OP_ADAPT * (OP_MIN + reward)
        T *= SA_COOLING_RATE

    # --- Phase 3: Deep Alternating Polish (LPT/SPT) ---
    # Alternates between sorting by Longest Processing Time (to fit big rocks)
    # and Shortest Processing Time (to fill gaps with sand).
    
    txns_lpt = sorted(range(workload.num_txns), key=lambda x: txn_weights[x], reverse=True)
    txns_spt = sorted(range(workload.num_txns), key=lambda x: txn_weights[x], reverse=False)
    
    polish_seq = list(best_schedule)
    current_polish_cost = best_cost
    
    for pass_idx in range(POLISH_MAX_PASSES):
        search_active = False
        
        # Alternate sorting order
        ordering = txns_lpt if pass_idx % 2 == 0 else txns_spt
        
        for item in ordering:
            try:
                curr_idx = polish_seq.index(item)
            except ValueError:
                continue
                
            polish_seq.pop(curr_idx)
            
            best_pos = -1
            min_c = float('inf')
            
            # Exhaustive scan for best position.
            # Strict inequality (<) enforces Left-Packing Bias (chooses earliest slot).
            for j in range(len(polish_seq) + 1):
                polish_seq.insert(j, item)
                c = get_cost(polish_seq)
                if c < min_c:
                    min_c = c
                    best_pos = j
                polish_seq.pop(j)
                
            polish_seq.insert(best_pos, item)
            
            # Continue if we improved cost OR shifted structure (plateau traversal)
            if min_c < current_polish_cost:
                current_polish_cost = min_c
                search_active = True
            elif min_c == current_polish_cost and best_pos != curr_idx:
                search_active = True
                
        if current_polish_cost < best_cost:
            best_cost = current_polish_cost
            best_schedule = list(polish_seq)
            
        if not search_active:
            break
            
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