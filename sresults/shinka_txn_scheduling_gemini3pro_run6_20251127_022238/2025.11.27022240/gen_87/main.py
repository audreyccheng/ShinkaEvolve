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
    Get optimal schedule using Quartic-Weighted Beam Search and Threshold ILS.
    
    Strategies:
    - Quartic (d^4) Sampling: Aggressively prioritizes long transactions.
    - Deep Lookahead: Evaluates top-5 LPT targets to minimize future idle time.
    - Threshold Acceptance ILS: Accepts slightly degraded solutions (0.5%) to escape local optima.
    - Aggressive Ruin: Removes 6-10 items to break complex dependencies.
    """
    
    # --- Configuration ---
    # Beam Search
    BEAM_WIDTH = int(max(16, num_seqs * 3.2))
    SAMPLES_PER_NODE = 32
    MAX_CHILDREN_PER_NODE = 5
    
    # Lookahead
    LOOKAHEAD_TARGETS = 5
    LOOKAHEAD_POOL_FACTOR = 2.0
    
    # ILS
    ILS_CYCLES = 10
    THRESHOLD_ALPHA = 0.005 # 0.5% threshold for acceptance
    RUIN_MIN = 6
    RUIN_MAX = 10

    num_txns = workload.num_txns

    # --- Precompute Heuristics ---
    # Cost of a single transaction is its duration proxy
    txn_durations = {t: workload.get_opt_seq_cost([t]) for t in range(num_txns)}
    # Sort transactions by duration (LPT)
    sorted_lpt = sorted(range(num_txns), key=lambda t: txn_durations[t], reverse=True)

    # --- Beam Search ---

    # Initialize Beam with diverse start points
    start_candidates = set()
    start_candidates.update(sorted_lpt[:BEAM_WIDTH])
    attempts = 0
    while len(start_candidates) < BEAM_WIDTH * 2 and attempts < num_txns:
        start_candidates.add(random.randint(0, num_txns - 1))
        attempts += 1

    beam = []
    for t in start_candidates:
        rem = set(range(num_txns))
        rem.remove(t)
        cost = txn_durations[t]
        beam.append({
            'cost': cost,
            'total_dur': cost,
            'seq': [t],
            'rem': rem,
            'score': 0
        })

    # Initial Prune
    beam.sort(key=lambda x: (x['score'], -x['total_dur']))
    beam = beam[:BEAM_WIDTH]

    # Main Loop
    for _ in range(num_txns - 1):
        candidates = []

        for p_idx, parent in enumerate(beam):
            rem_list = list(parent['rem'])
            if not rem_list: continue

            # --- 1. Hybrid Sampling ---
            samples = set()

            # A. Deterministic Anchors (Top 5 LPT)
            # Ensures we don't miss obvious critical path steps
            lpt_count = 0
            for t in sorted_lpt:
                if t in parent['rem']:
                    samples.add(t)
                    lpt_count += 1
                    if lpt_count >= 5: break

            # B. Quartic Weighted Random Sampling
            needed = SAMPLES_PER_NODE - len(samples)
            if needed > 0:
                pool = [x for x in rem_list if x not in samples]
                if pool:
                    if len(pool) <= needed:
                        samples.update(pool)
                    else:
                        # Quartic weights (d^4)
                        weights = [txn_durations[x]**4 for x in pool]
                        try:
                            # Use random.choices for weighted sampling with replacement
                            # Oversample to account for collisions
                            chosen = random.choices(pool, weights=weights, k=needed * 2)
                            samples.update(chosen)
                        except ValueError:
                            pass
                        
                        # Fill up if needed
                        if len(samples) < SAMPLES_PER_NODE:
                            rem_pool = [x for x in pool if x not in samples]
                            if rem_pool:
                                k = min(len(rem_pool), SAMPLES_PER_NODE - len(samples))
                                samples.update(random.sample(rem_pool, k))

            # --- 2. Base Evaluation ---
            for t in samples:
                new_seq = parent['seq'] + [t]
                new_cost = workload.get_opt_seq_cost(new_seq)
                new_total_dur = parent['total_dur'] + txn_durations[t]
                
                # Base Score: Density (Cost - Sum of Durations)
                # Proxy for idle time
                base_score = new_cost - new_total_dur
                
                candidates.append({
                    'cost': new_cost,
                    'total_dur': new_total_dur,
                    'seq': new_seq,
                    'rem': parent['rem'], # Ref
                    'added': t,
                    'p_idx': p_idx,
                    'base_score': base_score
                })

        # --- 3. Lookahead Selection ---
        # Sort by base score to pick promising candidates for expensive lookahead
        candidates.sort(key=lambda x: (x['base_score'], -x['total_dur']))
        
        # Deduplicate states
        lookahead_pool = []
        seen_states = set()
        target_pool_size = int(BEAM_WIDTH * LOOKAHEAD_POOL_FACTOR)
        
        for cand in candidates:
            new_rem = cand['rem'].copy()
            new_rem.remove(cand['added'])
            state_key = frozenset(new_rem)
            
            if state_key not in seen_states:
                seen_states.add(state_key)
                cand['new_rem'] = new_rem
                lookahead_pool.append(cand)
                if len(lookahead_pool) >= target_pool_size:
                    break
        
        # --- 4. Multi-Target Optimistic Lookahead ---
        for cand in lookahead_pool:
            rem = cand['new_rem']
            if not rem:
                cand['final_score'] = cand['base_score']
                continue
            
            # Identify Top-N LPT targets
            targets = []
            count = 0
            for t in sorted_lpt:
                if t in rem:
                    targets.append(t)
                    count += 1
                    if count >= LOOKAHEAD_TARGETS: break
            
            if targets:
                # Optimistic Scoring: Can we fit ANY of the critical items well?
                best_la_score = float('inf')
                for next_t in targets:
                    la_seq = cand['seq'] + [next_t]
                    la_cost = workload.get_opt_seq_cost(la_seq)
                    la_total = cand['total_dur'] + txn_durations[next_t]
                    score = la_cost - la_total
                    if score < best_la_score:
                        best_la_score = score
                cand['final_score'] = best_la_score
            else:
                cand['final_score'] = cand['base_score']

        # --- 5. Final Beam Selection ---
        lookahead_pool.sort(key=lambda x: (x['final_score'], -x['total_dur']))
        
        new_beam = []
        p_counts = {i: 0 for i in range(len(beam))}
        reserve = []
        
        for cand in lookahead_pool:
            p_idx = cand['p_idx']
            node = {
                'cost': cand['cost'],
                'total_dur': cand['total_dur'],
                'seq': cand['seq'],
                'rem': cand['new_rem']
            }
            
            # Diversity check
            if p_counts[p_idx] < MAX_CHILDREN_PER_NODE:
                if len(new_beam) < BEAM_WIDTH:
                    new_beam.append(node)
                    p_counts[p_idx] += 1
                else:
                    pass
            else:
                reserve.append(node)
                
            if len(new_beam) >= BEAM_WIDTH and len(reserve) > BEAM_WIDTH:
                break
        
        # Fill from reserve if beam not full
        if len(new_beam) < BEAM_WIDTH:
            for node in reserve:
                if len(new_beam) >= BEAM_WIDTH: break
                new_beam.append(node)
        
        if not new_beam: break
        beam = new_beam

    # Best Beam Result
    best_result = min(beam, key=lambda x: x['cost'])
    best_cost = best_result['cost']
    best_seq = best_result['seq']

    # --- Iterated Local Search (ILS) ---

    def refine(seq, cost):
        """Robust Local Search: Swaps, Insertions, Block Moves."""
        improved = True
        while improved:
            improved = False
            
            # 1. Swaps (Window 5)
            for i in range(len(seq) - 1):
                for offset in range(1, 6):
                    j = i + offset
                    if j >= len(seq): break
                    
                    seq[i], seq[j] = seq[j], seq[i]
                    c = workload.get_opt_seq_cost(seq)
                    if c < cost:
                        cost = c
                        improved = True
                    else:
                        seq[i], seq[j] = seq[j], seq[i] # Revert
                if improved: break
            if improved: continue

            # 2. Insertions (Window 8)
            w_ins = 8
            for i in range(len(seq)):
                start = max(0, i - w_ins)
                end = min(len(seq), i + w_ins)
                
                curr = seq[i]
                temp = seq[:i] + seq[i+1:]
                
                for k in range(start, end):
                    if k == i: continue
                    cand = temp[:k] + [curr] + temp[k:]
                    c = workload.get_opt_seq_cost(cand)
                    if c < cost:
                        cost = c
                        seq = cand
                        improved = True
                        break
                if improved: break
            if improved: continue

            # 3. Block Moves (Size 2, Window 6)
            if len(seq) > 4:
                w_blk = 6
                for i in range(len(seq) - 1):
                    block = seq[i:i+2]
                    rem_seq = seq[:i] + seq[i+2:]
                    
                    start = max(0, i - w_blk)
                    end = min(len(rem_seq) + 1, i + w_blk)
                    
                    for k in range(start, end):
                        if abs(k - i) < 2: continue
                        cand = rem_seq[:k] + block + rem_seq[k:]
                        c = workload.get_opt_seq_cost(cand)
                        if c < cost:
                            cost = c
                            seq = cand
                            improved = True
                            break
                    if improved: break
        return seq, cost

    # Phase 1: Initial Descent
    best_seq, best_cost = refine(best_seq, best_cost)

    # Phase 2: Threshold Acceptance Loop
    curr_seq = best_seq[:]
    curr_cost = best_cost
    
    for _ in range(ILS_CYCLES):
        p_seq = curr_seq[:]
        
        # --- Ruin ---
        if len(p_seq) > 10:
            ruin_size = random.randint(RUIN_MIN, RUIN_MAX)
            removed = []
            for _ in range(ruin_size):
                if not p_seq: break
                idx = random.randint(0, len(p_seq) - 1)
                removed.append(p_seq.pop(idx))
            
            # --- Recreate (Greedy Best-Fit, Heavy First) ---
            removed.sort(key=lambda t: txn_durations[t], reverse=True)
            
            for t in removed:
                best_pos = -1
                best_c = float('inf')
                # Try all positions
                for pos in range(len(p_seq) + 1):
                    cand = p_seq[:pos] + [t] + p_seq[pos:]
                    c = workload.get_opt_seq_cost(cand)
                    if c < best_c:
                        best_c = c
                        best_pos = pos
                
                if best_pos != -1:
                    p_seq.insert(best_pos, t)
                else:
                    p_seq.append(t)
        
        # --- Repair ---
        p_seq, p_cost = refine(p_seq, workload.get_opt_seq_cost(p_seq))
        
        # --- Threshold Acceptance ---
        if p_cost < best_cost:
            best_cost = p_cost
            best_seq = p_seq
            curr_seq = p_seq
            curr_cost = p_cost
        # Accept if within threshold to explore neighbors
        elif p_cost < curr_cost * (1.0 + THRESHOLD_ALPHA):
            curr_seq = p_seq
            curr_cost = p_cost

    return best_cost, best_seq


def get_random_costs():
    """
    Evaluate scheduling algorithm on three different workloads.

    Returns:
        Tuple of (total_makespan, list_of_schedules, execution_time)
    """
    start_time = time.time()

    # Increase effort level for wider beam search
    effort_level = 12

    # Workload 1: Complex mixed read/write transactions
    workload1 = Workload(WORKLOAD_1)
    makespan1, schedule1 = get_best_schedule(workload1, effort_level)

    # Workload 2: Simple read-then-write pattern
    workload2 = Workload(WORKLOAD_2)
    makespan2, schedule2 = get_best_schedule(workload2, effort_level)

    # Workload 3: Minimal read/write operations
    workload3 = Workload(WORKLOAD_3)
    makespan3, schedule3 = get_best_schedule(workload3, effort_level)

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