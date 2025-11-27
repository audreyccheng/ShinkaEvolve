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
    Get optimal schedule using Lookahead Beam Search and Multi-Cycle ILS.

    Args:
        workload: Workload object containing transaction data
        num_seqs: Used to scale the beam width

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    # --- Parameters ---
    BEAM_WIDTH = int(max(10, num_seqs * 2.5))
    SAMPLES_PER_NODE = 20
    MAX_CHILDREN = 3
    ILS_CYCLES = 6  # Number of perturbation-repair iterations

    # Lookahead pool size (subset of candidates to apply expensive lookahead to)
    LOOKAHEAD_POOL_SIZE = int(BEAM_WIDTH * 1.5)

    num_txns = workload.num_txns

    # --- Precompute Heuristics ---
    txn_durations = {t: workload.get_opt_seq_cost([t]) for t in range(num_txns)}
    sorted_lpt = sorted(range(num_txns), key=lambda t: txn_durations[t], reverse=True)

    # --- Initialization ---
    start_candidates = set()
    start_candidates.update(sorted_lpt[:BEAM_WIDTH])
    # Fill remaining with random to ensure diversity
    attempts = 0
    while len(start_candidates) < BEAM_WIDTH * 2 and attempts < num_txns * 2:
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
            'score': cost - cost # 0
        })

    # Initial Prune
    beam.sort(key=lambda x: (x['score'], -x['total_dur']))
    beam = beam[:BEAM_WIDTH]

    # --- Beam Search Loop ---
    for _ in range(num_txns - 1):
        candidates = []

        for p_idx, parent in enumerate(beam):
            rem_list = list(parent['rem'])
            if not rem_list: continue

            # --- Hybrid Sampling ---
            samples = set()

            # 1. Deterministic LPT (Top 2)
            lpt_count = 0
            for t in sorted_lpt:
                if t in parent['rem']:
                    samples.add(t)
                    lpt_count += 1
                    if lpt_count >= 2:
                        break

            # 2. Weighted Random Sampling
            needed = SAMPLES_PER_NODE - len(samples)
            if needed > 0:
                pool = [x for x in rem_list if x not in samples]
                if pool:
                    if len(pool) <= needed:
                        samples.update(pool)
                    else:
                        # Weight by duration
                        weights = [txn_durations[x] for x in pool]
                        
                        picks = set()
                        # Oversample to account for potential duplicates in choices
                        try:
                            chosen = random.choices(pool, weights=weights, k=needed * 2)
                            samples.update(chosen)
                        except ValueError:
                            # Fallback if weights sum to 0 (unlikely)
                            samples.update(random.sample(pool, min(len(pool), needed)))
                            
                        # Fill if still needed
                        if len(samples) < SAMPLES_PER_NODE:
                             needed_now = SAMPLES_PER_NODE - len(samples)
                             rem_pool = [x for x in pool if x not in samples]
                             if rem_pool:
                                 samples.update(random.sample(rem_pool, min(len(rem_pool), needed_now)))

            # --- Base Evaluation ---
            for t in samples:
                new_seq = parent['seq'] + [t]
                new_cost = workload.get_opt_seq_cost(new_seq)
                new_total_dur = parent['total_dur'] + txn_durations[t]
                
                # Base Score: Density (Cost - SumOfDurations)
                # Lower is better (implies higher parallelism)
                base_score = new_cost - new_total_dur

                candidates.append({
                    'cost': new_cost,
                    'total_dur': new_total_dur,
                    'seq': new_seq,
                    'rem': parent['rem'], # Reference only
                    'added': t,
                    'p_idx': p_idx,
                    'base_score': base_score
                })

        # --- Lookahead Phase ---
        # 1. Filter promising candidates by base score
        candidates.sort(key=lambda x: (x['base_score'], -x['total_dur']))
        
        # 2. Deduplicate states to avoid redundant lookahead computations
        lookahead_candidates = []
        seen_states = set()
        
        for cand in candidates:
            # Create the future state key
            new_rem = cand['rem'].copy()
            new_rem.remove(cand['added'])
            state_key = frozenset(new_rem)
            
            if state_key not in seen_states:
                seen_states.add(state_key)
                cand['new_rem'] = new_rem
                lookahead_candidates.append(cand)
                if len(lookahead_candidates) >= LOOKAHEAD_POOL_SIZE:
                    break
        
        # 3. Apply One-Step Lookahead (LPT)
        for cand in lookahead_candidates:
            rem_set = cand['new_rem']
            if not rem_set:
                cand['final_score'] = cand['base_score']
                continue

            # Find next largest transaction available in this branch
            next_t = None
            for t in sorted_lpt:
                if t in rem_set:
                    next_t = t
                    break
            
            if next_t is not None:
                # Tentatively schedule it
                la_seq = cand['seq'] + [next_t]
                la_cost = workload.get_opt_seq_cost(la_seq)
                la_total = cand['total_dur'] + txn_durations[next_t]
                
                # Update score based on lookahead result
                cand['final_score'] = la_cost - la_total
            else:
                cand['final_score'] = cand['base_score']

        # 4. Final Selection
        lookahead_candidates.sort(key=lambda x: (x['final_score'], -x['total_dur']))

        new_beam = []
        p_counts = {i: 0 for i in range(len(beam))}
        reserve = []

        for cand in lookahead_candidates:
            p_idx = cand['p_idx']
            node = {
                'cost': cand['cost'],
                'total_dur': cand['total_dur'],
                'seq': cand['seq'],
                'rem': cand['new_rem']
            }
            
            # Enforce diversity (max children per parent)
            if p_counts[p_idx] < MAX_CHILDREN:
                if len(new_beam) < BEAM_WIDTH:
                    new_beam.append(node)
                    p_counts[p_idx] += 1
                else:
                    # Beam full
                    pass
            else:
                reserve.append(node)
            
            if len(new_beam) >= BEAM_WIDTH and len(reserve) > BEAM_WIDTH:
                break
        
        # Fill from reserve if needed
        if len(new_beam) < BEAM_WIDTH:
            for node in reserve:
                if len(new_beam) >= BEAM_WIDTH: break
                new_beam.append(node)

        if not new_beam:
            break
        beam = new_beam

    # Extract best from beam
    best_result = min(beam, key=lambda x: x['cost'])
    best_cost = best_result['cost']
    best_seq = best_result['seq']

    # --- Multi-Cycle Iterated Local Search (ILS) ---
    
    def run_local_search(seq, current_cost):
        """
        Greedy refinement using Swaps and Insertions.
        Uses First-Improvement strategy within small windows for speed.
        """
        improved = True
        while improved:
            improved = False
            
            # 1. Swaps (Window 5)
            # Good for fixing local ordering of conflicting items
            for i in range(len(seq) - 1):
                for offset in range(1, 6):
                    j = i + offset
                    if j >= len(seq): break
                    
                    # Swap
                    seq[i], seq[j] = seq[j], seq[i]
                    c = workload.get_opt_seq_cost(seq)
                    if c < current_cost:
                        current_cost = c
                        improved = True
                    else:
                        seq[i], seq[j] = seq[j], seq[i] # Revert
                if improved: break 

            if improved: continue

            # 2. Insertions (Window 8)
            # Good for moving a transaction past a block of conflicts
            w_ins = 8
            for i in range(len(seq)):
                start = max(0, i - w_ins)
                end = min(len(seq), i + w_ins)
                if start >= end: continue
                
                curr = seq[i]
                temp = seq[:i] + seq[i+1:]
                
                found_better = False
                for k in range(start, end):
                    cand = temp[:k] + [curr] + temp[k:]
                    c = workload.get_opt_seq_cost(cand)
                    if c < current_cost:
                        current_cost = c
                        seq = cand
                        found_better = True
                        break
                
                if found_better:
                    improved = True
                    break
                    
        return seq, current_cost

    # Phase 1: Initial Descent
    best_seq, best_cost = run_local_search(best_seq, best_cost)

    # Phase 2: Iterated Perturbation & Repair
    curr_seq = best_seq[:]
    curr_cost = best_cost

    for i in range(ILS_CYCLES):
        # Perturb (Kick)
        # Vary kick strength: every 3rd cycle do a double swap
        num_swaps = 2 if (i % 3 == 0) else 1
        
        p_seq = curr_seq[:]
        if len(p_seq) > 4:
            for _ in range(num_swaps):
                idx1, idx2 = random.sample(range(len(p_seq)), 2)
                p_seq[idx1], p_seq[idx2] = p_seq[idx2], p_seq[idx1]
        
        # Repair
        p_seq, p_cost = run_local_search(p_seq, workload.get_opt_seq_cost(p_seq))
        
        # Acceptance Criterion (Simple Greedy)
        # We accept if better than globally best, or better than current base
        # Here we track global best
        if p_cost < best_cost:
            best_cost = p_cost
            best_seq = p_seq
            curr_seq = p_seq # Move to new basin
            curr_cost = p_cost
        elif p_cost < curr_cost:
            # Accept improvement over current even if not global best (unlikely in greedy ILS but logical)
            curr_seq = p_seq
            curr_cost = p_cost
        # Else: Reject and loop again from curr_seq (implicitly, as we use curr_seq for next kick)

    return best_cost, best_seq


def get_random_costs():
    """
    Evaluate scheduling algorithm on three different workloads.

    Returns:
        Tuple of (total_makespan, list_of_schedules, execution_time)
    """
    start_time = time.time()

    # Effort level scales beam width
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