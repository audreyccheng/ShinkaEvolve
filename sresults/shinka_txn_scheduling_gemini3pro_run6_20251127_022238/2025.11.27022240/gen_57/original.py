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
    Get optimal schedule using Adaptive Beam Search and Block-Move ILS.

    Args:
        workload: Workload object containing transaction data
        num_seqs: Used to scale the beam width (effort factor)

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    # --- Configuration ---
    # Adaptive Beam Width
    BEAM_WIDTH_START = int(max(15, num_seqs * 2.5))
    BEAM_WIDTH_END = int(max(8, num_seqs * 1.5))

    # Sampling
    SAMPLES_PER_NODE = 16

    # Lookahead
    # Check this many LPT candidates in lookahead phase to determine score
    LOOKAHEAD_LPT_CANDIDATES = 2

    # ILS
    ILS_CYCLES = 8

    num_txns = workload.num_txns

    # --- Precompute Heuristics ---
    # Cost of individual transactions (proxy for duration/complexity)
    txn_durations = {t: workload.get_opt_seq_cost([t]) for t in range(num_txns)}
    # Sorted list of transactions by duration (LPT)
    sorted_lpt = sorted(range(num_txns), key=lambda t: txn_durations[t], reverse=True)

    # --- Beam Search ---

    # Initialization
    start_candidates = set()
    # Anchor with heaviest transactions
    start_candidates.update(sorted_lpt[:BEAM_WIDTH_START])
    # Add random starts
    while len(start_candidates) < BEAM_WIDTH_START * 2:
        start_candidates.add(random.randint(0, num_txns - 1))

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
            'score': 0 # cost - total_dur
        })

    # Initial sort and prune
    beam.sort(key=lambda x: (x['score'], -x['total_dur']))
    beam = beam[:BEAM_WIDTH_START]

    # Main Loop
    for step in range(num_txns - 1):
        # Calculate current beam width (linear interpolation)
        progress = step / num_txns
        current_width = int(BEAM_WIDTH_START + (BEAM_WIDTH_END - BEAM_WIDTH_START) * progress)

        candidates = []

        for parent in beam:
            rem_list = list(parent['rem'])
            if not rem_list: continue

            # --- 1. Hybrid Sampling ---
            samples = set()

            # A. Deterministic LPT (Top 3)
            # Pick longest available tasks to schedule early
            lpt_picked = 0
            for t in sorted_lpt:
                if t in parent['rem']:
                    samples.add(t)
                    lpt_picked += 1
                    if lpt_picked >= 3:
                        break

            # B. Weighted Random
            needed = SAMPLES_PER_NODE - len(samples)
            if needed > 0:
                pool = [x for x in rem_list if x not in samples]
                if pool:
                    if len(pool) <= needed:
                        samples.update(pool)
                    else:
                        weights = [txn_durations[x] for x in pool]
                        # Oversample to hit unique targets
                        picks = random.choices(pool, weights=weights, k=needed + 2)
                        samples.update(picks)

                        # Fill randomly if still short
                        if len(samples) < SAMPLES_PER_NODE:
                            rem_needed = SAMPLES_PER_NODE - len(samples)
                            others = [x for x in pool if x not in samples]
                            if others:
                                samples.update(random.sample(others, min(len(others), rem_needed)))

            # --- 2. Base Evaluation ---
            for t in samples:
                new_seq = parent['seq'] + [t]
                new_cost = workload.get_opt_seq_cost(new_seq)
                new_total_dur = parent['total_dur'] + txn_durations[t]

                # Base Score: Density (Lower is better)
                base_score = new_cost - new_total_dur

                candidates.append({
                    'cost': new_cost,
                    'total_dur': new_total_dur,
                    'seq': new_seq,
                    'rem': parent['rem'], # Ref
                    'added': t,
                    'base_score': base_score
                })

        # --- 3. Lookahead Selection ---
        # Sort by base score to pick candidates for lookahead
        candidates.sort(key=lambda x: (x['base_score'], -x['total_dur']))

        # Deduplicate states for lookahead pool
        # We process more candidates than the beam width to find hidden gems
        lookahead_pool = []
        seen_states = set()
        pool_limit = int(current_width * 1.5)

        for cand in candidates:
            new_rem = cand['rem'].copy()
            new_rem.remove(cand['added'])
            state_key = frozenset(new_rem)

            if state_key not in seen_states:
                seen_states.add(state_key)
                cand['new_rem'] = new_rem
                lookahead_pool.append(cand)
                if len(lookahead_pool) >= pool_limit:
                    break

        # --- 4. Multi-Candidate Lookahead ---
        for cand in lookahead_pool:
            rem = cand['new_rem']
            if not rem:
                cand['final_score'] = cand['base_score']
                continue

            # Find top N LPT items in remainder
            next_lpts = []
            for t in sorted_lpt:
                if t in rem:
                    next_lpts.append(t)
                    if len(next_lpts) >= LOOKAHEAD_LPT_CANDIDATES:
                        break

            if next_lpts:
                # Optimistic Lookahead:
                # "Can I schedule *any* of the important heavy tasks next efficiently?"
                best_la_score = float('inf')

                for next_t in next_lpts:
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
        for cand in lookahead_pool[:current_width]:
            new_beam.append({
                'cost': cand['cost'],
                'total_dur': cand['total_dur'],
                'seq': cand['seq'],
                'rem': cand['new_rem']
            })

        if not new_beam: break
        beam = new_beam

    # Best Beam Result
    best_node = min(beam, key=lambda x: x['cost'])
    best_cost = best_node['cost']
    best_seq = best_node['seq']

    # --- Iterated Local Search (ILS) ---

    def run_local_search(seq, current_cost):
        """Greedy refinement: Swaps and Insertions."""
        improved = True
        while improved:
            improved = False

            # Swaps (Window 6)
            for i in range(len(seq) - 1):
                for offset in range(1, 7):
                    j = i + offset
                    if j >= len(seq): break

                    seq[i], seq[j] = seq[j], seq[i]
                    c = workload.get_opt_seq_cost(seq)
                    if c < current_cost:
                        current_cost = c
                        improved = True
                    else:
                        seq[i], seq[j] = seq[j], seq[i]
                if improved: break

            if improved: continue

            # Insertions (Window 10)
            w_ins = 10
            for i in range(len(seq)):
                start = max(0, i - w_ins)
                end = min(len(seq), i + w_ins)
                if start >= end: continue

                curr = seq[i]
                temp = seq[:i] + seq[i+1:]

                best_pos = -1
                best_c = current_cost

                # Check positions
                for k in range(start, end):
                    if k == i: continue # Same pos
                    # Construct candidate (optimization: partial copy)
                    cand = temp[:k] + [curr] + temp[k:]
                    c = workload.get_opt_seq_cost(cand)
                    if c < best_c:
                        best_c = c
                        best_pos = k

                if best_pos != -1:
                    # Apply move
                    # Reconstruct correctly based on index logic
                    if best_pos < i:
                        seq = seq[:best_pos] + [curr] + seq[best_pos:i] + seq[i+1:]
                    else:
                         # item was at i, removed (indices shift down), insert at best_pos
                         # Wait, temp is already seq without i.
                         seq = temp[:best_pos] + [curr] + temp[best_pos:]

                    current_cost = best_c
                    improved = True
                    break # Restart loop

        return seq, current_cost

    def perturb(seq):
        """Apply structural perturbation."""
        n = len(seq)
        new_seq = seq[:]
        mode = random.random()

        if mode < 0.4:
            # Block Move
            # Move a chunk of transactions to a new location
            # Good for fixing phase issues
            block_size = random.randint(3, 6)
            if n > block_size:
                start = random.randint(0, n - block_size)
                block = new_seq[start : start + block_size]
                remaining = new_seq[:start] + new_seq[start + block_size:]
                insert_pos = random.randint(0, len(remaining))
                new_seq = remaining[:insert_pos] + block + remaining[insert_pos:]

        elif mode < 0.7:
            # Multi-Swap
            for _ in range(3):
                i, j = random.sample(range(n), 2)
                new_seq[i], new_seq[j] = new_seq[j], new_seq[i]

        else:
            # Shuffle Segment
            l = random.randint(5, 10)
            if n > l:
                s = random.randint(0, n - l)
                sub = new_seq[s : s+l]
                random.shuffle(sub)
                new_seq[s : s+l] = sub

        return new_seq

    # 1. Initial Descent
    best_seq, best_cost = run_local_search(best_seq, best_cost)

    # 2. ILS Loop
    curr_seq = best_seq[:]
    curr_cost = best_cost

    for _ in range(ILS_CYCLES):
        p_seq = perturb(curr_seq)
        p_seq, p_cost = run_local_search(p_seq, workload.get_opt_seq_cost(p_seq))

        # Acceptance: Better than global best OR better than current
        if p_cost < best_cost:
            best_cost = p_cost
            best_seq = p_seq
            curr_seq = p_seq
            curr_cost = p_cost
        elif p_cost < curr_cost:
            curr_seq = p_seq
            curr_cost = p_cost
        else:
            # Small probability to accept worse? No, stick to greedy for now given constraints
            pass

    return best_cost, best_seq


def get_random_costs():
    """
    Evaluate scheduling algorithm on three different workloads.

    Returns:
        Tuple of (total_makespan, list_of_schedules, execution_time)
    """
    start_time = time.time()

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