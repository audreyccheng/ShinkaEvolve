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
    Get optimal schedule using Prospector Beam Search and Scramble-Enhanced ILS.
    
    Innovations:
    1. Prospector Beam Search: Uses a wider lookahead probe set (Top-3) to validate 
       candidate transactions, ensuring they don't block multiple critical paths.
    2. Scramble Perturbation: A novel ILS kick that randomizes the order within 
       contiguous blocks (Block Scramble), effective for resolving ordering 
       deadlocks within conflict clusters.
    3. Balanced Urgency: Weighted 0.6 Duration / 0.4 Conflict to slightly favor 
       clearing long tasks while respecting dependencies.

    Args:
        workload: Workload object
        num_seqs: Budget parameter

    Returns:
        (lowest_makespan, schedule)
    """

    num_txns = workload.num_txns

    # --- 1. METRIC PRECOMPUTATION ---
    txn_durations = {}
    txn_rw_sets = {}

    for t in range(num_txns):
        # Duration
        try:
            d = workload.txns[t][0][3]
        except:
            d = 1.0
        txn_durations[t] = d

        # R/W Sets
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

    # Compute Conflict Volume
    txn_conflict_vol = {t: 0.0 for t in range(num_txns)}

    # Check conflicts (O(N^2))
    if num_txns < 2000:
        for i in range(num_txns):
            r1, w1 = txn_rw_sets[i]
            vol = 0.0
            for j in range(num_txns):
                if i == j: continue
                r2, w2 = txn_rw_sets[j]
                # Conflict: (W1 n (W2 u R2)) or (R1 n W2)
                if not w1.isdisjoint(w2) or not w1.isdisjoint(r2) or not r1.isdisjoint(w2):
                    vol += txn_durations[j]
            txn_conflict_vol[i] = vol

    # Urgency Score
    max_vol = max(txn_conflict_vol.values()) if txn_conflict_vol else 1.0
    if max_vol == 0: max_vol = 1.0
    max_dur = max(txn_durations.values()) if txn_durations else 1.0

    txn_urgency = {}
    for t in range(num_txns):
        # 60% Duration, 40% Conflict Volume
        # Unlocking large conflicts is good, but clearing long tasks is critical.
        u = 0.6 * (txn_durations[t] / max_dur) + 0.4 * (txn_conflict_vol[t] / max_vol)
        txn_urgency[t] = u

    # --- 2. PROSPECTOR BEAM SEARCH ---
    
    # Beam settings
    BEAM_WIDTH = max(6, int(num_seqs)) # Slightly wider beam
    
    # State: (immediate_cost, schedule, remaining)
    beam = [(0, [], list(range(num_txns)))]
    
    for _ in range(num_txns):
        candidates_pool = []
        
        for p_cost, p_sched, p_remain in beam:
            
            # Sort remaining by urgency
            sorted_remain = sorted(p_remain, key=lambda x: txn_urgency[x], reverse=True)
            
            candidates_to_eval = set()
            
            # A. Deterministic High-Priority (Top 2)
            candidates_to_eval.update(sorted_remain[:2])
            
            # B. Stochastic Sampling (Top-Tier)
            # Sample from top 50% or top 15 items to ensure quality
            pool_size = len(sorted_remain)
            if pool_size > 2:
                # Focus on the upper echelon
                sample_pool = sorted_remain[2 : 2 + min(15, pool_size)] 
                if sample_pool:
                    # Cubic weights for aggressive bias
                    weights = [txn_urgency[x]**3 for x in sample_pool]
                    k = min(4, len(sample_pool))
                    candidates_to_eval.update(random.choices(sample_pool, weights=weights, k=k))
            
            # C. Pure Random (Escape)
            if pool_size > 20:
                candidates_to_eval.update(random.sample(sorted_remain[2:], 1))
            
            # Evaluate
            for cand in candidates_to_eval:
                sched_c = p_sched + [cand]
                cost_c = workload.get_opt_seq_cost(sched_c)
                
                remain_c = list(p_remain)
                remain_c.remove(cand)
                
                # LOOKAHEAD PROSPECTING
                # Check top-3 urgent items in the remaining set
                lookahead_val = cost_c
                
                probes = []
                p_count = 0
                # Scan sorted_remain to find top 3 urgent that are not cand
                for r in sorted_remain:
                    if r == cand: continue
                    probes.append(r)
                    p_count += 1
                    if p_count >= 3: break
                
                if probes:
                    # Prospect: can any of these follow efficiently?
                    best_probe_cost = float('inf')
                    for p in probes:
                        pc = workload.get_opt_seq_cost(sched_c + [p])
                        if pc < best_probe_cost:
                            best_probe_cost = pc
                    
                    if best_probe_cost != float('inf'):
                        lookahead_val = best_probe_cost

                # Metric: (Lookahead, Immediate, -Urgency)
                score = (lookahead_val, cost_c, -txn_urgency[cand])
                candidates_pool.append((score, sched_c, remain_c))
        
        # Pruning
        candidates_pool.sort(key=lambda x: x[0])
        # Next beam: (immediate_cost, sched, remain)
        # x[0] is (lookahead, immediate, urgency). We keep immediate cost for next iter base.
        beam = [(x[0][1], x[1], x[2]) for x in candidates_pool[:BEAM_WIDTH]]
    
    best_state = beam[0]
    current_cost = best_state[0]
    current_schedule = best_state[1]

    # --- 3. SCRAMBLE-ENHANCED ILS ---
    
    def local_search_descent(sched, start_c, passes=1, window=15):
        s = list(sched)
        c = start_c
        
        for _ in range(passes):
            improved = False
            # Randomize order of checking to avoid bias
            indices = list(range(len(s)))
            random.shuffle(indices)
            
            for i in indices:
                item = s[i]
                temp = s[:i] + s[i+1:]
                
                # Windowed search
                start = max(0, i - window)
                end = min(len(temp), i + window)
                
                best_p = -1
                best_v = c
                
                # Optimization: Check current position (implied by temp insertion) 
                # effectively first or naturally in loop
                
                for p in range(start, end + 1):
                    cand = temp[:p] + [item] + temp[p:]
                    # Only compute if different from current state (approx check)
                    # Actually get_opt_seq_cost is fast enough
                    val = workload.get_opt_seq_cost(cand)
                    if val < best_v:
                        best_v = val
                        best_p = p
                
                if best_p != -1:
                    s = temp[:best_p] + [item] + temp[best_p:]
                    c = best_v
                    improved = True
                    
            if not improved: break
        return c, s

    # 1. Initial Deep Descent
    current_cost, current_schedule = local_search_descent(current_schedule, current_cost, passes=2)

    # 2. Perturbation Loop
    # With num_seqs=10, we can afford ~5-6 kicks
    num_kicks = 6
    
    for k in range(num_kicks):
        neighbor = list(current_schedule)
        r = random.random()
        
        # Adaptive Strategy:
        # Mix of Block Scramble, Block Move, and Swap
        
        if r < 0.35:
            # Block Scramble (Novelty)
            # Pick a block and shuffle it in place. 
            # Helps resolve internal ordering of a conflict cluster.
            if len(neighbor) > 5:
                bs = random.randint(4, 10)
                start = random.randint(0, len(neighbor) - bs)
                segment = neighbor[start : start+bs]
                random.shuffle(segment)
                neighbor[start : start+bs] = segment
                
        elif r < 0.70:
            # Block Move
            # Relocate a sequence of transactions
            if len(neighbor) > 5:
                bs = random.randint(3, 8)
                src = random.randint(0, len(neighbor) - bs)
                block = neighbor[src : src+bs]
                del neighbor[src : src+bs]
                dst = random.randint(0, len(neighbor))
                neighbor[dst:dst] = block
                
        else:
            # Multi-Swap (Fallback)
            for _ in range(3):
                i, j = random.randint(0, num_txns-1), random.randint(0, num_txns-1)
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        
        kick_cost = workload.get_opt_seq_cost(neighbor)
        
        # Repair (Shallow)
        new_cost, new_sched = local_search_descent(neighbor, kick_cost, passes=1)
        
        if new_cost < current_cost:
            current_cost = new_cost
            current_schedule = new_sched

    return current_cost, current_schedule


def get_random_costs():
    """Evaluate scheduling algorithm on three different workloads."""
    start_time = time.time()
    
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