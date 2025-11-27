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
    Get optimal schedule using Lookahead Beam Search with Iterated Local Search.

    Combines:
    1. Beam Search with 1-Step Lookahead (avoids myopic greedy choices).
    2. Hybrid Sampling (LPT + Weighted Random) for robust exploration.
    3. Iterated Local Search (Descent + Perturbation + Repair) to refine results.

    Args:
        workload: Workload object containing transaction data
        num_seqs: Used to scale the beam width

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    # --- Parameters ---
    BEAM_WIDTH = int(max(15, num_seqs * 3.0))
    SAMPLES_PER_NODE = 24
    MAX_CHILDREN = 3
    # Lookahead: Evaluate lookahead score for top N candidates
    # N = Beam Width * Factor. Higher = more accurate but slower.
    LOOKAHEAD_FACTOR = 1.5

    num_txns = workload.num_txns

    # --- Precompute Heuristics ---
    txn_durations = {t: workload.get_opt_seq_cost([t]) for t in range(num_txns)}
    sorted_lpt = sorted(range(num_txns), key=lambda t: txn_durations[t], reverse=True)

    # --- Initialization ---
    start_candidates = set()
    start_candidates.update(sorted_lpt[:BEAM_WIDTH])
    # Fill remaining with random starts
    while len(start_candidates) < BEAM_WIDTH * 2 and len(start_candidates) < num_txns:
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
            'score': 0 # cost - total_dur, 0 for single item
        })

    # Initial Prune
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
            lpt_count = 0
            for t in sorted_lpt:
                if t in parent['rem']:
                    samples.add(t)
                    lpt_count += 1
                    if lpt_count >= 3: break

            # 2. Weighted Random Sampling
            needed = SAMPLES_PER_NODE - len(samples)
            if needed > 0:
                pool = [x for x in rem_list if x not in samples]
                if pool:
                    if len(pool) <= needed:
                        samples.update(pool)
                    else:
                        weights = [txn_durations[x] for x in pool]
                        # Efficient approximate sampling
                        # Oversample slightly to account for collisions
                        picks = random.choices(pool, weights=weights, k=needed * 2)
                        samples.update(picks)

                        # Fallback if choices didn't yield enough unique items
                        if len(samples) < SAMPLES_PER_NODE:
                            remaining = [x for x in pool if x not in samples]
                            if remaining:
                                k = min(len(remaining), SAMPLES_PER_NODE - len(samples))
                                samples.update(random.sample(remaining, k))

            # --- Base Evaluation ---
            for t in samples:
                new_seq = parent['seq'] + [t]
                new_cost = workload.get_opt_seq_cost(new_seq)
                new_total_dur = parent['total_dur'] + txn_durations[t]

                # Base Score: Work Density (Makespan - TotalDuration)
                base_score = new_cost - new_total_dur

                candidates.append({
                    'cost': new_cost,
                    'total_dur': new_total_dur,
                    'seq': new_seq,
                    'rem': parent['rem'], # Ref only
                    'added': t,
                    'p_idx': p_idx,
                    'base_score': base_score
                })

        # --- Lookahead Phase ---
        # 1. Pre-filter by base score to limit lookahead cost
        candidates.sort(key=lambda x: (x['base_score'], -x['total_dur']))

        # 2. Select deduplicated pool for lookahead
        lookahead_pool = []
        seen_states = set()
        target_pool_size = int(BEAM_WIDTH * LOOKAHEAD_FACTOR)

        for cand in candidates:
            # Deduplicate state
            new_rem = cand['rem'].copy()
            new_rem.remove(cand['added'])
            state_key = frozenset(new_rem)

            if state_key not in seen_states:
                seen_states.add(state_key)
                cand['new_rem'] = new_rem
                lookahead_pool.append(cand)
                if len(lookahead_pool) >= target_pool_size:
                    break

        # 3. Compute Robust Lookahead Score
        for cand in lookahead_pool:
            rem = cand['new_rem']
            if not rem:
                cand['final_score'] = cand['base_score']
                continue

            # Identify top critical transactions (LPT) in remaining set
            # We check the top 2 to ensure we don't block secondary critical paths
            check_candidates = []
            for t in sorted_lpt:
                if t in rem:
                    check_candidates.append(t)
                    if len(check_candidates) >= 2:
                        break

            if check_candidates:
                la_scores = []
                for next_t in check_candidates:
                    la_seq = cand['seq'] + [next_t]
                    la_cost = workload.get_opt_seq_cost(la_seq)
                    la_total_dur = cand['total_dur'] + txn_durations[next_t]
                    # Score = Idle time introduced
                    la_scores.append(la_cost - la_total_dur)

                # Pessimistic: assume the worst outcome among critical paths
                cand['final_score'] = max(la_scores)
            else:
                cand['final_score'] = cand['base_score']

        # 4. Final Selection (Sort by Lookahead Score)
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
                'score': cand['final_score']
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

        # Fill from reserve
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
    def refine(seq, cost):
        """Improve sequence using swaps and insertions"""
        # Swaps
        w_swap = 5
        improved = True
        while improved:
            improved = False
            for i in range(len(seq) - 1):
                for offset in range(1, w_swap + 1):
                    j = i + offset
                    if j >= len(seq): break

                    seq[i], seq[j] = seq[j], seq[i]
                    c = workload.get_opt_seq_cost(seq)
                    if c < cost:
                        cost = c
                        improved = True
                    else:
                        seq[i], seq[j] = seq[j], seq[i]
            if improved: break # Limit iterations for speed

        # Insertions
        w_ins = 8
        for i in range(len(seq)):
            curr = seq[i]
            temp = seq[:i] + seq[i+1:]
            start = max(0, i - w_ins)
            end = min(len(seq), i + w_ins)

            for k in range(start, end):
                cand = temp[:k] + [curr] + temp[k:]
                c = workload.get_opt_seq_cost(cand)
                if c < cost:
                    cost = c
                    seq = cand
                    break # Greedy first improvement
        return seq, cost

    # 1. Initial Descent
    best_seq, best_cost = refine(best_seq, best_cost)

    # 2. Perturbation Loop (Multi-Swap Kick)
    for _ in range(3):
        p_seq = best_seq[:]
        if len(p_seq) > 5:
            # Stronger Kick: Swap 3 random pairs
            for _ in range(3):
                i1, i2 = random.sample(range(len(p_seq)), 2)
                p_seq[i1], p_seq[i2] = p_seq[i2], p_seq[i1]

            # Repair
            p_seq, p_cost = refine(p_seq, workload.get_opt_seq_cost(p_seq))

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

    # Scale effort with sequence budget
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