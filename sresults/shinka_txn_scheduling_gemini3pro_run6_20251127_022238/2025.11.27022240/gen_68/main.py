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
    Get optimal schedule using Quartic-Weighted Beam Search with State Deduplication
    and Smart Block-Move ILS.

    Innovations:
    1. Quartic Weighting (d^4): Exponentially biases stochastic sampling towards
       critical tasks to ensure "big rocks" are packed first.
    2. State Deduplication: Filters the beam to ensure only the best schedule for a 
       given set of completed transactions is kept. This maximizes beam diversity 
       by preventing it from filling up with permutations of the same subset.
    3. Deep Lookahead (Top-5): Validates candidates by checking compatibility with
       the top 5 remaining urgent items.
    4. Smart Block Move: In ILS, when moving a block of transactions, we scan for 
       the optimal insertion point rather than picking a random destination.

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
                # Conflict Check
                if not w1.isdisjoint(w2) or not w1.isdisjoint(r2) or not r1.isdisjoint(w2):
                    vol += txn_durations[j]
            txn_conflict_vol[i] = vol

    # Urgency Score
    max_vol = max(txn_conflict_vol.values()) if txn_conflict_vol else 1.0
    if max_vol == 0: max_vol = 1.0
    max_dur = max(txn_durations.values()) if txn_durations else 1.0

    txn_urgency = {}
    for t in range(num_txns):
        # 50/50 balance of duration and conflict impact
        u = 0.5 * (txn_durations[t] / max_dur) + 0.5 * (txn_conflict_vol[t] / max_vol)
        txn_urgency[t] = u

    # --- 2. BEAM SEARCH WITH DEDUPLICATION ---

    BEAM_WIDTH = max(5, int(num_seqs))

    # Beam State: (immediate_cost, schedule, remaining_list)
    beam = [(0, [], list(range(num_txns)))]

    for _ in range(num_txns):
        candidates_pool = []

        for p_cost, p_sched, p_remain in beam:

            # Sort remaining by Urgency
            sorted_remain = sorted(p_remain, key=lambda x: txn_urgency[x], reverse=True)

            to_evaluate = set()

            # A. Deterministic Top 2
            to_evaluate.update(sorted_remain[:2])

            # B. Quartic Weighted Stochastic Sampling
            if len(sorted_remain) > 2:
                pool = sorted_remain[2:20] # Focus on top chunk
                # Urgency^4
                weights = [txn_urgency[x]**4 for x in pool]
                if pool:
                    k_samples = min(4, len(pool))
                    to_evaluate.update(random.choices(pool, weights=weights, k=k_samples))

            # C. Exploration
            if len(sorted_remain) > 15:
                to_evaluate.update(random.sample(sorted_remain, 1))

            # Evaluate
            for cand in to_evaluate:
                curr_sched = p_sched + [cand]
                curr_cost = workload.get_opt_seq_cost(curr_sched)

                # Lookahead: Check Top-5 urgent items
                la_metric = curr_cost
                probes = []
                count = 0
                for r in sorted_remain:
                    if r != cand:
                        probes.append(r)
                        count += 1
                    if count >= 5: break
                
                if probes:
                    probe_costs = []
                    for p in probes:
                        probe_costs.append(workload.get_opt_seq_cost(curr_sched + [p]))
                    
                    if probe_costs:
                        # Optimistic lookahead: can we fit at least one efficiently?
                        la_metric = min(probe_costs)

                # Score Tuple: (LookaheadMetric, ImmediateCost, -Urgency)
                score = (la_metric, curr_cost, -txn_urgency[cand])
                
                rem_c = list(p_remain)
                rem_c.remove(cand)
                
                candidates_pool.append((score, curr_sched, rem_c))

        # Sort all candidates by score
        candidates_pool.sort(key=lambda x: x[0])
        
        # Construct next beam with State Deduplication
        # Two states are "equivalent" if they have scheduled the same set of transactions.
        # We keep the one with the lowest cost.
        next_beam = []
        seen_states = set()
        
        for cand in candidates_pool:
            if len(next_beam) >= BEAM_WIDTH:
                break
            
            # State identifier: frozenset of scheduled items
            # Optimization: since all candidates at this depth have same length, 
            # set equality is sufficient.
            state_key = frozenset(cand[1])
            
            if state_key not in seen_states:
                seen_states.add(state_key)
                # Store (immediate_cost, sched, remain)
                next_beam.append((cand[0][1], cand[1], cand[2]))
        
        beam = next_beam

    best_state = beam[0]
    current_cost = best_state[0]
    current_schedule = best_state[1]

    # --- 3. STAGNATION-AWARE SMART ILS ---

    def local_descent(sched, current_c, passes=1, window_size=10):
        s = list(sched)
        c = current_c
        
        for _ in range(passes):
            improved = False
            indices = list(range(len(s)))
            random.shuffle(indices)
            
            for i in indices:
                item = s[i]
                temp = s[:i] + s[i+1:]
                
                start = max(0, i - window_size)
                end = min(len(temp), i + window_size)
                
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

    # Adaptive Loop
    stagnation = 0
    max_kicks = 5
    
    for _ in range(max_kicks):
        neighbor = list(current_schedule)
        
        # Strategy selection based on stagnation
        if stagnation >= 2:
            # Ruin & Recreate (Heavy)
            # Remove random items
            k_remove = random.randint(5, 12)
            removed = []
            for _ in range(k_remove):
                if not neighbor: break
                idx = random.randint(0, len(neighbor)-1)
                removed.append(neighbor.pop(idx))
            
            # Re-insert heavy items first using scan
            removed.sort(key=lambda x: txn_urgency[x], reverse=True)
            for item in removed:
                best_p = -1
                best_v = float('inf')
                step = 2 if len(neighbor) > 100 else 1
                for p in range(0, len(neighbor)+1, step):
                    cand = neighbor[:p] + [item] + neighbor[p:]
                    val = workload.get_opt_seq_cost(cand)
                    if val < best_v:
                        best_v = val
                        best_p = p
                
                if best_p != -1:
                    neighbor.insert(best_p, item)
                else:
                    neighbor.append(item)
                    
        else:
            # Smart Block Move (Medium)
            # Pick a block, find best place for it
            if len(neighbor) > 10:
                bs = random.randint(4, 8)
                src = random.randint(0, len(neighbor) - bs)
                block = neighbor[src : src+bs]
                del neighbor[src : src+bs]
                
                best_p = -1
                best_v = float('inf')
                
                # Scan for block insertion
                step = 2 if len(neighbor) > 100 else 1
                for p in range(0, len(neighbor)+1, step):
                    # List slicing to insert block
                    cand = neighbor[:p] + block + neighbor[p:]
                    val = workload.get_opt_seq_cost(cand)
                    if val < best_v:
                        best_v = val
                        best_p = p
                
                if best_p != -1:
                    neighbor[best_p:best_p] = block
            else:
                # Fallback to swap
                for _ in range(3):
                    i, j = random.randint(0, num_txns-1), random.randint(0, num_txns-1)
                    neighbor[i], neighbor[j] = neighbor[j], neighbor[i]

        # Repair
        kick_cost = workload.get_opt_seq_cost(neighbor)
        new_cost, new_sched = local_descent(neighbor, kick_cost, passes=1)
        
        # Acceptance
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