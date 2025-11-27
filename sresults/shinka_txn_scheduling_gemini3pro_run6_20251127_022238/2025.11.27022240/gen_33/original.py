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
    Get optimal schedule using Parallelism-Guided Beam Search with Diversity Control.

    Algorithm:
    1. Beam Search driven by 'Work Density' (Makespan - TotalDuration).
       - Promotes schedules that achieve high parallelism early (low makespan relative to work done).
    2. Diversity Control: Limits the number of descendants from a single parent to prevent beam collapse.
    3. Hybrid Sampling: Mixes Deterministic LPT (for critical paths) and Random Sampling (for exploration).
    4. Enhanced Local Search: Wide-window swaps and insertions to refine the structure.

    Args:
        workload: Workload object containing transaction data
        num_seqs: Used to scale the beam width

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    # --- Configuration ---
    # Width: Trade-off between exploration and speed. 
    # Slightly wider than minimal to allow the density heuristic to work effectivey.
    BEAM_WIDTH = int(max(12, num_seqs * 2.5))
    SAMPLES_PER_NODE = 24
    MAX_CHILDREN = 4  # Limit descendants per parent to enforce diversity

    num_txns = workload.num_txns

    # --- Precomputation ---
    # Individual transaction costs (durations) used for heuristics
    txn_durations = {t: workload.get_opt_seq_cost([t]) for t in range(num_txns)}

    # LPT Sorted List for deterministic sampling
    sorted_by_duration = sorted(range(num_txns), key=lambda t: txn_durations[t], reverse=True)

    # --- Initialization ---
    start_candidates = set()
    # Seed with top LPT items (potential anchors)
    start_candidates.update(sorted_by_duration[:BEAM_WIDTH])
    # Fill remaining initial slots with random transactions
    while len(start_candidates) < BEAM_WIDTH * 2:
        start_candidates.add(random.randint(0, num_txns - 1))

    beam = []
    for t in start_candidates:
        rem = set(range(num_txns))
        rem.remove(t)
        # Score Calculation: Cost - TotalDuration
        # Lower is better. Indicates amount of "serial overhead".
        cost = txn_durations[t]
        total_dur = txn_durations[t]
        score = cost - total_dur 
        beam.append({
            'cost': cost,
            'total_dur': total_dur,
            'score': score,
            'seq': [t],
            'rem': rem
        })

    # Prune initial beam based on density heuristic
    beam.sort(key=lambda x: (x['score'], -x['total_dur']))
    beam = beam[:BEAM_WIDTH]

    # --- Beam Search Loop ---
    for _ in range(num_txns - 1):
        candidates = []

        for p_idx, parent in enumerate(beam):
            rem_list = list(parent['rem'])
            
            # --- Hybrid Sampling ---
            samples = set()

            # 1. Deterministic LPT (Top 3)
            # Ensures we always consider scheduling the largest remaining items
            lpt_added = 0
            for t in sorted_by_duration:
                if t in parent['rem']:
                    samples.add(t)
                    lpt_added += 1
                    if lpt_added >= 3:
                        break

            # 2. Random Sampling (Fill remainder)
            needed = SAMPLES_PER_NODE - len(samples)
            if needed > 0:
                pool = [x for x in rem_list if x not in samples]
                if len(pool) <= needed:
                    samples.update(pool)
                else:
                    samples.update(random.sample(pool, needed))

            # --- Evaluation ---
            for t in samples:
                new_seq = parent['seq'] + [t]
                new_cost = workload.get_opt_seq_cost(new_seq)
                new_total_dur = parent['total_dur'] + txn_durations[t]

                # Heuristic: Minimize (Makespan - TotalWork)
                # This explicitly targets parallelism.
                # If transactions run in parallel, Cost < TotalWork, Score becomes negative.
                score = new_cost - new_total_dur

                candidates.append({
                    'priority': (score, -new_total_dur), # Primary: Density, Secondary: Prefer more work
                    'cost': new_cost,
                    'total_dur': new_total_dur,
                    'seq': new_seq,
                    'rem': parent['rem'], # Ref only
                    'added': t,
                    'p_idx': p_idx
                })

        # Sort candidates by priority (best first)
        candidates.sort(key=lambda x: x['priority'])

        # --- Selection with Deduplication and Diversity ---
        new_beam = []
        seen_states = set()
        parent_counts = {i: 0 for i in range(len(beam))}
        reserve = []

        for cand in candidates:
            # Deduplication: Check if state (set of remaining txns) is unique
            new_rem = cand['rem'].copy()
            new_rem.remove(cand['added'])
            
            state_key = frozenset(new_rem)
            if state_key in seen_states:
                continue
            seen_states.add(state_key)

            # Node creation
            node = {
                'cost': cand['cost'],
                'total_dur': cand['total_dur'],
                'score': cand['priority'][0],
                'seq': cand['seq'],
                'rem': new_rem
            }

            # Diversity Check
            p_idx = cand['p_idx']
            if parent_counts[p_idx] < MAX_CHILDREN:
                if len(new_beam) < BEAM_WIDTH:
                    new_beam.append(node)
                    parent_counts[p_idx] += 1
                else:
                    # Beam is full, and we've satisfied diversity with better nodes
                    pass
            else:
                # Parent has contributed enough; put in reserve
                reserve.append(node)
            
            # Optimization: Stop if we have enough diverse candidates
            # Note: We continue to collect reserve candidates just in case
            if len(new_beam) >= BEAM_WIDTH and len(reserve) > BEAM_WIDTH:
                break

        # Fill from reserve if beam is not full (relax diversity constraint)
        if len(new_beam) < BEAM_WIDTH:
            for node in reserve:
                if len(new_beam) >= BEAM_WIDTH:
                    break
                new_beam.append(node)

        if not new_beam:
            break
        beam = new_beam

    # --- Result Selection ---
    best_result = min(beam, key=lambda x: x['cost'])
    best_cost = best_result['cost']
    best_seq = best_result['seq']

    # --- Local Search Refinement ---
    
    # Phase 1: Windowed Swaps (Structure cleanup)
    # Window size 6 covers slightly more than immediate neighbors
    SWAP_WINDOW = 6
    for _ in range(2):
        improved = False
        for i in range(num_txns - 1):
            for offset in range(1, SWAP_WINDOW + 1):
                j = i + offset
                if j >= num_txns: break

                best_seq[i], best_seq[j] = best_seq[j], best_seq[i]
                c = workload.get_opt_seq_cost(best_seq)

                if c < best_cost:
                    best_cost = c
                    improved = True
                else:
                    best_seq[i], best_seq[j] = best_seq[j], best_seq[i]
        if not improved:
            break

    # Phase 2: Windowed Insertions (Deep optimization)
    # Try moving each transaction to a better spot within a larger window
    INSERT_WINDOW = 10
    for i in range(num_txns):
        curr_txn = best_seq[i]
        
        # Create base sequence without the current transaction
        temp_seq = best_seq[:i] + best_seq[i+1:]
        
        # Determine range of positions to try
        start_k = max(0, i - INSERT_WINDOW)
        end_k = min(len(temp_seq), i + INSERT_WINDOW)
        
        for k in range(start_k, end_k + 1):
            # Construct candidate: insert curr_txn at position k in temp_seq
            cand_seq = temp_seq[:k] + [curr_txn] + temp_seq[k:]
            
            c = workload.get_opt_seq_cost(cand_seq)
            if c < best_cost:
                best_cost = c
                best_seq = cand_seq
                # Greedy: Update and move to next transaction 'i'
                # (Note: indices shift, but simplistic iteration is sufficient for refinement)
                break

    return best_cost, best_seq


def get_random_costs():
    """
    Evaluate scheduling algorithm on three different workloads.

    Returns:
        Tuple of (total_makespan, list_of_schedules, execution_time)
    """
    start_time = time.time()

    effort_level = 10

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