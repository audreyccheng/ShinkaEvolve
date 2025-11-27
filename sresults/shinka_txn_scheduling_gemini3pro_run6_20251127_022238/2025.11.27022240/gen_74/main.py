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
    Get optimal schedule using Quartic-Weighted Prospector Beam Search and 
    Adaptive Scramble/Ruin ILS.

    Innovations:
    1. Quartic Prospector Beam: Uses aggressive d^4 urgency weighting to sample candidates,
       validated by a Top-4 Lookahead that ensures critical future tasks are not blocked.
    2. Adaptive Scramble/Ruin ILS: Combines three perturbation strategies based on stagnation:
       - Block Scramble: Randomizes a dense cluster (from Current).
       - Block Move: Relocates a chunk (Standard).
       - Ruin & Recreate: Removes and intelligently re-inserts items to escape optima (from Inspiration).

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

    if num_txns < 2500:
        for i in range(num_txns):
            r1, w1 = txn_rw_sets[i]
            vol = 0.0
            for j in range(num_txns):
                if i == j: continue
                r2, w2 = txn_rw_sets[j]
                if not w1.isdisjoint(w2) or not w1.isdisjoint(r2) or not r1.isdisjoint(w2):
                    vol += txn_durations[j]
            txn_conflict_vol[i] = vol

    # Urgency Score
    max_vol = max(txn_conflict_vol.values()) if txn_conflict_vol else 1.0
    if max_vol == 0: max_vol = 1.0
    max_dur = max(txn_durations.values()) if txn_durations else 1.0

    txn_urgency = {}
    for t in range(num_txns):
        # Balanced Approach: 0.5 Duration + 0.5 Conflict Volume
        u = 0.5 * (txn_durations[t] / max_dur) + 0.5 * (txn_conflict_vol[t] / max_vol)
        txn_urgency[t] = u

    # --- 2. QUARTIC PROSPECTOR BEAM SEARCH ---
    
    BEAM_WIDTH = max(6, int(num_seqs))
    
    beam = [(0, [], list(range(num_txns)))]
    
    for _ in range(num_txns):
        candidates_pool = []
        
        for p_cost, p_sched, p_remain in beam:
            
            # Sort remaining by urgency
            sorted_remain = sorted(p_remain, key=lambda x: txn_urgency[x], reverse=True)
            
            candidates_to_eval = set()
            
            # A. Deterministic Top 2
            candidates_to_eval.update(sorted_remain[:2])
            
            # B. Quartic Stochastic Sampling
            if len(sorted_remain) > 2:
                pool = sorted_remain[2:]
                # Weights = urgency^4 (Strong bias)
                weights = [txn_urgency[x]**4 for x in pool]
                
                # Sample up to 4 items
                k_samples = min(4, len(pool))
                if k_samples > 0:
                    candidates_to_eval.update(random.choices(pool, weights=weights, k=k_samples))
            
            # C. Exploration
            if len(sorted_remain) > 20:
                candidates_to_eval.update(random.sample(sorted_remain, 1))
            
            # Evaluate with Prospector Lookahead
            for cand in candidates_to_eval:
                sched_c = p_sched + [cand]
                cost_c = workload.get_opt_seq_cost(sched_c)
                
                # Lookahead: Can any of the top-4 urgent items follow?
                lookahead_metric = cost_c
                probes = []
                count = 0
                for r in sorted_remain:
                    if r != cand:
                        probes.append(r)
                        count += 1
                    if count >= 4: break
                
                if probes:
                    probe_costs = []
                    for p in probes:
                        probe_costs.append(workload.get_opt_seq_cost(sched_c + [p]))
                    if probe_costs:
                        lookahead_metric = min(probe_costs)
                
                # Metric: (LookaheadCost, ImmediateCost, -Urgency)
                score = (lookahead_metric, cost_c, -txn_urgency[cand])
                
                rem_c = list(p_remain)
                rem_c.remove(cand)
                
                candidates_pool.append((score, sched_c, rem_c))
        
        # Pruning
        candidates_pool.sort(key=lambda x: x[0])
        # Beam keeps (immediate_cost, sched, remain)
        beam = [(x[0][1], x[1], x[2]) for x in candidates_pool[:BEAM_WIDTH]]
    
    best_state = beam[0]
    current_cost = best_state[0]
    current_schedule = best_state[1]

    # --- 3. ADAPTIVE SCRAMBLE/RUIN ILS ---
    
    def local_descent(sched, start_c, passes=1, window=12):
        s = list(sched)
        c = start_c
        for _ in range(passes):
            improved = False
            indices = list(range(len(s)))
            random.shuffle(indices)
            for i in indices:
                item = s[i]
                temp = s[:i] + s[i+1:]
                start = max(0, i - window)
                end = min(len(temp), i + window)
                best_p = -1
                best_v = c
                
                for p in range(start, end + 1):
                    cand = temp[:p] + [item] + temp[p:]
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

    # Initial Polish
    current_cost, current_schedule = local_descent(current_schedule, current_cost, passes=2)
    
    stagnation = 0
    num_kicks = 6
    
    for _ in range(num_kicks):
        neighbor = list(current_schedule)
        
        # Strategy Selector
        # Stagnation 0: Mix of Scramble / Block Move
        # Stagnation >= 1: Ruin & Recreate (Heavy)
        
        strategy = 'mix'
        if stagnation >= 1:
            strategy = 'ruin'
            
        if strategy == 'mix':
            r = random.random()
            if r < 0.4:
                # Block Scramble
                if len(neighbor) > 6:
                    bs = random.randint(4, 10)
                    start = random.randint(0, len(neighbor) - bs)
                    chunk = neighbor[start : start+bs]
                    random.shuffle(chunk)
                    neighbor[start : start+bs] = chunk
            elif r < 0.8:
                # Block Move
                if len(neighbor) > 6:
                    bs = random.randint(4, 8)
                    src = random.randint(0, len(neighbor) - bs)
                    block = neighbor[src : src+bs]
                    del neighbor[src : src+bs]
                    dst = random.randint(0, len(neighbor))
                    neighbor[dst:dst] = block
            else:
                # Swap
                for _ in range(3):
                    i, j = random.randint(0, num_txns-1), random.randint(0, num_txns-1)
                    neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
                    
        elif strategy == 'ruin':
            # Ruin & Recreate
            num_remove = random.randint(6, 12)
            removed = []
            for _ in range(num_remove):
                if not neighbor: break
                idx = random.randint(0, len(neighbor)-1)
                removed.append(neighbor.pop(idx))
            
            # Re-insert heavy first
            removed.sort(key=lambda x: txn_urgency[x], reverse=True)
            for item in removed:
                best_p = -1
                best_v = float('inf')
                
                # Coarse search
                step = 1
                if len(neighbor) > 100: step = 2
                
                for p in range(0, len(neighbor)+1, step):
                    cand = neighbor[:p] + [item] + neighbor[p:]
                    val = workload.get_opt_seq_cost(cand)
                    if val < best_v:
                        best_v = val
                        best_p = p
                
                # Refine
                if step > 1 and best_p != -1:
                    for p in range(max(0, best_p-2), min(len(neighbor), best_p+3)):
                        cand = neighbor[:p] + [item] + neighbor[p:]
                        val = workload.get_opt_seq_cost(cand)
                        if val < best_v:
                            best_v = val
                            best_p = p
                            
                if best_p != -1:
                    neighbor.insert(best_p, item)
                else:
                    neighbor.append(item)
        
        # Repair
        kick_cost = workload.get_opt_seq_cost(neighbor)
        new_cost, new_sched = local_descent(neighbor, kick_cost, passes=1)
        
        if new_cost < current_cost:
            current_cost = new_cost
            current_schedule = new_sched
            stagnation = 0
        else:
            stagnation += 1

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