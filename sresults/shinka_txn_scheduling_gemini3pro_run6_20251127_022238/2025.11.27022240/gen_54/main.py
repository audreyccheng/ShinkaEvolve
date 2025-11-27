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
    Get optimal schedule using Tapered Beam Search and Multi-Phase Refinement.

    Algorithm Steps:
    1. Metric Calculation: Compute Urgency based on Duration and Conflict Volume.
    2. Construction: Tapered Beam Search (Wide start, Narrow end) to explore early critical paths.
    3. Refinement 1: Global Critical Reinsertion (Optimally place top bottlenecks).
    4. Refinement 2: Iterated Local Search (Kick + Windowed Descent).

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

    # Compute Conflict Volume (Sum of durations of conflicting transactions)
    txn_conflict_vol = {t: 0.0 for t in range(num_txns)}

    # Optimize for larger workloads by skipping if N is too large, but typically N < 1000
    if num_txns < 2000:
        for i in range(num_txns):
            r1, w1 = txn_rw_sets[i]
            vol = 0.0
            for j in range(num_txns):
                if i == j: continue
                r2, w2 = txn_rw_sets[j]
                # Conflict Check
                if not w1.isdisjoint(w2) or not w1.isdisjoint(r2) or not r1.isdisjoint(w2):
                    vol += txn_durations[j]
            txn_conflict_vol[i] = vol

    # Normalized Urgency
    max_vol = max(txn_conflict_vol.values()) if txn_conflict_vol else 1.0
    max_dur = max(txn_durations.values()) if txn_durations else 1.0
    if max_vol == 0: max_vol = 1.0

    txn_urgency = {}
    for t in range(num_txns):
        # Weight: Duration + 0.8 * Conflict Volume
        # Slightly favor Conflict Volume compared to 0.5 to better resolve bottlenecks
        u = (txn_durations[t] / max_dur) + 0.8 * (txn_conflict_vol[t] / max_vol)
        txn_urgency[t] = u

    # --- 2. TAPERED BEAM SEARCH ---
    # Start wide to catch diverse roots, narrow down to focus budget
    base_width = max(5, int(num_seqs))
    start_width = int(base_width * 2.0)
    end_width = max(2, int(base_width * 0.6))

    # Beam State: (cost, schedule_list, remaining_list)
    beam = [(0, [], list(range(num_txns)))]

    for step in range(num_txns):
        # Calculate current width based on progress
        progress = step / num_txns
        width = int(start_width + (end_width - start_width) * progress)
        width = max(2, width)

        candidates_pool = []

        for p_cost, p_sched, p_remain in beam:
            
            # Sort remaining by Urgency
            sorted_remain = sorted(p_remain, key=lambda x: txn_urgency[x], reverse=True)
            
            # Candidate Selection:
            next_candidates = set()
            
            # 1. Top Greedy
            next_candidates.update(sorted_remain[:2])
            
            # 2. Weighted Random (Soft Greedy)
            if len(sorted_remain) > 2:
                pool = sorted_remain[2:]
                weights = [txn_urgency[x] for x in pool]
                if pool:
                    samples = random.choices(pool, weights=weights, k=min(3, len(pool)))
                    next_candidates.update(samples)
            
            # 3. Pure Random (Diversity)
            if len(sorted_remain) > 10:
                next_candidates.update(random.sample(sorted_remain, 1))

            # Evaluate
            for cand in next_candidates:
                new_sched = p_sched + [cand]
                cost = workload.get_opt_seq_cost(new_sched)

                # Pruning Metric: (Cost, -Urgency)
                # Primary: Low Cost. Tie-break: High Urgency (clearing heavy items is good)
                metric = (cost, -txn_urgency[cand])
                
                new_remain = list(p_remain)
                new_remain.remove(cand)
                
                candidates_pool.append((metric, new_sched, new_remain))

        # Select best states
        candidates_pool.sort(key=lambda x: x[0])
        beam = [(x[0][0], x[1], x[2]) for x in candidates_pool[:width]]

    best_state = beam[0]
    current_cost = best_state[0]
    current_schedule = best_state[1]

    # --- 3. REFINEMENT Phase 1: GLOBAL CRITICAL REINSERTION ---
    # Identify top bottlenecks and try to place them optimally in the constructed schedule.
    # This fixes early mistakes where high-urgency items were placed too late or too early.
    
    num_critical = min(8, num_txns)
    critical_items = sorted(range(num_txns), key=lambda x: txn_urgency[x], reverse=True)[:num_critical]

    for t_crit in critical_items:
        if t_crit not in current_schedule: continue
        curr_idx = current_schedule.index(t_crit)
        
        # Remove
        temp_sched = current_schedule[:curr_idx] + current_schedule[curr_idx+1:]
        
        best_pos = -1
        best_val = current_cost
        
        # Global Search
        for p in range(len(temp_sched) + 1):
            cand = temp_sched[:p] + [t_crit] + temp_sched[p:]
            c = workload.get_opt_seq_cost(cand)
            if c < best_val:
                best_val = c
                best_pos = p
        
        if best_pos != -1:
            current_schedule = temp_sched[:best_pos] + [t_crit] + temp_sched[best_pos:]
            current_cost = best_val

    # --- 4. REFINEMENT Phase 2: ITERATED LOCAL SEARCH (ILS) ---

    def local_descent(sched, base_cost, passes=1):
        """Windowed insertion descent."""
        curr_s = list(sched)
        curr_c = base_cost
        improved = True
        p = 0
        
        while improved and p < passes:
            improved = False
            p += 1
            
            indices = list(range(len(curr_s)))
            random.shuffle(indices)
            
            for i in indices:
                txn = curr_s[i]
                temp = curr_s[:i] + curr_s[i+1:]
                
                # Window size
                window = 10
                start = max(0, i - window)
                end = min(len(temp), i + window)
                
                best_p = -1
                best_v = curr_c
                
                for pos in range(start, end + 1):
                    cand = temp[:pos] + [txn] + temp[pos:]
                    val = workload.get_opt_seq_cost(cand)
                    if val < best_v:
                        best_v = val
                        best_p = pos
                
                if best_p != -1:
                    curr_s = temp[:best_p] + [txn] + temp[best_p:]
                    curr_c = best_v
                    improved = True
        
        return curr_c, curr_s

    # Initial Descent
    current_cost, current_schedule = local_descent(current_schedule, current_cost, passes=2)

    # Kick & Repair Loop
    num_kicks = 4
    for _ in range(num_kicks):
        # Kick
        neighbor = list(current_schedule)
        for _ in range(3):
            i, j = random.randint(0, num_txns-1), random.randint(0, num_txns-1)
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            
        kick_cost = workload.get_opt_seq_cost(neighbor)
        
        # Repair
        repair_cost, repair_sched = local_descent(neighbor, kick_cost, passes=1)
        
        # Accept
        if repair_cost < current_cost:
            current_cost = repair_cost
            current_schedule = repair_sched

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