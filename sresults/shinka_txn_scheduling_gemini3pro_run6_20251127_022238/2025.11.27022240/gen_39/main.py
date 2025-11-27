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
    Get optimal schedule using Robust Lookahead Beam Search with Multi-Start ILS.

    Key Features:
    1. Robust Lookahead: Checks impact on Top-2 LPT items to avoid blocking secondary critical paths.
    2. Squared Weighted Sampling: Aggressively prioritizes heavy items in the random pool.
    3. Multi-Start ILS: Runs multiple perturbation-repair cycles with variable kick strength.

    Args:
        workload: Workload object containing transaction data
        num_seqs: Used to scale the beam width

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    # --- Parameters ---
    BEAM_WIDTH = int(max(10, num_seqs * 3.0))
    SAMPLES_PER_NODE = 24
    MAX_CHILDREN = 5
    
    # Lookahead: Check this many top candidates
    # We restrict this pool to balance the cost of doing double-lookahead checks
    LOOKAHEAD_POOL_SIZE = BEAM_WIDTH 
    
    ILS_ITERATIONS = 4

    num_txns = workload.num_txns

    # --- Precompute Heuristics ---
    txn_durations = {t: workload.get_opt_seq_cost([t]) for t in range(num_txns)}
    sorted_lpt = sorted(range(num_txns), key=lambda t: txn_durations[t], reverse=True)

    # --- Initialization ---
    start_candidates = set()
    start_candidates.update(sorted_lpt[:BEAM_WIDTH])
    
    # Fill remainder with random
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
            'score': cost - cost # 0
        })
    
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

            # 1. Deterministic LPT (Top 3)
            lpt_count = 0
            for t in sorted_lpt:
                if t in parent['rem']:
                    samples.add(t)
                    lpt_count += 1
                    if lpt_count >= 3:
                        break

            # 2. Weighted Random Sampling (Squared weights)
            needed = SAMPLES_PER_NODE - len(samples)
            if needed > 0:
                pool = [x for x in rem_list if x not in samples]
                if pool:
                    if len(pool) <= needed:
                        samples.update(pool)
                    else:
                        # Square weights to bias strongly towards long transactions
                        weights = [txn_durations[x]**2 for x in pool]
                        
                        picks = set()
                        for _ in range(needed * 2): # Oversample
                            if len(picks) >= needed: break
                            pick = random.choices(pool, weights=weights, k=1)[0]
                            picks.add(pick)
                        samples.update(picks)
                        
                        # Fallback if choices didn't give enough uniques
                        if len(samples) < SAMPLES_PER_NODE:
                            rem_needed = SAMPLES_PER_NODE - len(samples)
                            others = [x for x in pool if x not in samples]
                            if others:
                                samples.update(random.sample(others, min(len(others), rem_needed)))

            # --- Base Evaluation ---
            for t in samples:
                new_seq = parent['seq'] + [t]
                new_cost = workload.get_opt_seq_cost(new_seq)
                new_total_dur = parent['total_dur'] + txn_durations[t]
                
                # Base Score: Density (Lower is better)
                score = new_cost - new_total_dur

                candidates.append({
                    'cost': new_cost,
                    'total_dur': new_total_dur,
                    'seq': new_seq,
                    'rem': parent['rem'], # Ref
                    'added': t,
                    'p_idx': p_idx,
                    'base_score': score
                })

        # --- Lookahead Stage ---
        # Sort by base score to pick promising candidates for expensive lookahead
        candidates.sort(key=lambda x: (x['base_score'], -x['total_dur']))
        
        # Deduplicate states
        unique_cands = []
        seen_states = set()
        
        for cand in candidates:
            new_rem = cand['rem'].copy()
            new_rem.remove(cand['added'])
            state_key = frozenset(new_rem)
            
            if state_key not in seen_states:
                seen_states.add(state_key)
                cand['new_rem'] = new_rem
                unique_cands.append(cand)
        
        # Limit pool
        lookahead_pool = unique_cands[:LOOKAHEAD_POOL_SIZE]

        # Perform Lookahead (Top 2 Check)
        for cand in lookahead_pool:
            rem_set = cand['new_rem']
            if not rem_set:
                cand['final_score'] = cand['base_score']
                continue

            # Identify Top 2 LPT items in remaining set
            check_list = []
            lpt_iter = iter(sorted_lpt)
            while len(check_list) < 2:
                try:
                    t = next(lpt_iter)
                    if t in rem_set:
                        check_list.append(t)
                except StopIteration:
                    break
            
            if not check_list:
                cand['final_score'] = cand['base_score']
                continue

            # Evaluate cost if we append these critical items
            la_scores = []
            for next_t in check_list:
                la_seq = cand['seq'] + [next_t]
                la_cost = workload.get_opt_seq_cost(la_seq)
                la_total = cand['total_dur'] + txn_durations[next_t]
                la_scores.append(la_cost - la_total)
            
            # Pessimistic Score: Take the MAX (worst) score.
            # If the current candidate blocks *either* of the top 2 critical items,
            # it will result in a high cost for that branch, penalizing the candidate.
            cand['final_score'] = max(la_scores)

        # --- Final Selection ---
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
                'rem': cand['new_rem'],
                'score': cand['final_score'] # Pass forward for sorting next gen
            }
            
            if p_counts[p_idx] < MAX_CHILDREN:
                if len(new_beam) < BEAM_WIDTH:
                    new_beam.append(node)
                    p_counts[p_idx] += 1
                else:
                    pass
            else:
                reserve.append(node)
            
            if len(new_beam) >= BEAM_WIDTH and len(reserve) > BEAM_WIDTH:
                break
        
        if len(new_beam) < BEAM_WIDTH:
            for node in reserve:
                if len(new_beam) >= BEAM_WIDTH: break
                new_beam.append(node)

        if not new_beam:
            break
        beam = new_beam

    # Select best
    best_result = min(beam, key=lambda x: x['cost'])
    best_cost = best_result['cost']
    best_seq = best_result['seq']

    # --- Iterated Local Search (ILS) ---
    
    def run_local_search(seq, current_cost):
        """Greedy insertion refinement"""
        w_ins = 8
        improved = True
        while improved:
            improved = False
            for i in range(len(seq)):
                # Windowed insertion
                start = max(0, i - w_ins)
                end = min(len(seq), i + w_ins)
                
                curr = seq[i]
                temp = seq[:i] + seq[i+1:]
                
                for k in range(start, end):
                    cand = temp[:k] + [curr] + temp[k:]
                    c = workload.get_opt_seq_cost(cand)
                    if c < current_cost:
                        current_cost = c
                        seq = cand
                        improved = True
                        break
                if improved: break
        return seq, current_cost

    # 1. Initial Descent
    best_seq, best_cost = run_local_search(best_seq, best_cost)

    # 2. Multi-Start Perturbation
    for _ in range(ILS_ITERATIONS):
        p_seq = best_seq[:]
        if len(p_seq) < 4: break
        
        # Variable Kick: 1 or 2 swaps
        num_swaps = random.choice([1, 2])
        for _ in range(num_swaps):
            idx1, idx2 = random.sample(range(len(p_seq)), 2)
            p_seq[idx1], p_seq[idx2] = p_seq[idx2], p_seq[idx1]
            
        # Repair
        p_cost = workload.get_opt_seq_cost(p_seq)
        p_seq, p_cost = run_local_search(p_seq, p_cost)
        
        if p_cost < best_cost:
            best_cost = p_cost
            best_seq = p_seq

    return best_cost, best_seq


def get_random_costs():
    """
    Evaluate scheduling algorithm on three different workloads.

    Returns:
        Tuple of (total_makespan, list_of_schedules, execution_time)
    """
    start_time = time.time()

    # Effort level 12 to enable sufficient beam width
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