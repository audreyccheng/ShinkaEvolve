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
    Get optimal schedule using Lookahead Beam Search and Multi-Phase ILS.

    Strategy:
    1. Calculate Urgency based on Duration and Conflict Volume.
    2. Construct initial schedule with Beam Search + Lookahead (Cost(S + c + NextBest)).
    3. Refine by globally re-inserting top critical items.
    4. Optimize with ILS using Block Moves and Windowed Descent.

    Args:
        workload: Workload object
        num_seqs: Budget parameter (controls beam width)

    Returns:
        (lowest_makespan, schedule)
    """

    num_txns = workload.num_txns

    # --- 1. METRIC PRECOMPUTATION ---
    txn_data = [] # Stores (duration, read_set, write_set)

    for t in range(num_txns):
        # Duration
        d = 1.0
        try:
            d = workload.txns[t][0][3]
        except:
            pass

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
                            type_, key = parts
                            if type_ == 'r': reads.add(key)
                            elif type_ == 'w': writes.add(key)
        except:
            pass
        txn_data.append((d, reads, writes))

    # Conflict Volume: Sum of durations of conflicting transactions
    conflict_vol = [0.0] * num_txns
    for i in range(num_txns):
        d1, r1, w1 = txn_data[i]
        vol = 0.0
        for j in range(num_txns):
            if i == j: continue
            d2, r2, w2 = txn_data[j]
            # Conflict Condition:
            # (W1 n W2) or (W1 n R2) or (R1 n W2)
            if not w1.isdisjoint(w2) or not w1.isdisjoint(r2) or not r1.isdisjoint(w2):
                vol += d2
        conflict_vol[i] = vol

    # Urgency Score
    # Normalize for combination
    max_vol = max(conflict_vol) if conflict_vol else 1.0
    max_dur = max(t[0] for t in txn_data) if txn_data else 1.0
    if max_vol == 0: max_vol = 1.0

    urgency = {}
    for t in range(num_txns):
        # Weighted mix: 1.0 Duration, 0.7 Conflict Volume (Based on historical best)
        # Duration is slightly prioritized to clear long jobs.
        u = 1.0 * (txn_data[t][0] / max_dur) + 0.7 * (conflict_vol[t] / max_vol)
        urgency[t] = u

    # --- 2. TAPERED LOOKAHEAD BEAM SEARCH ---

    # Dynamic Beam Width: Start wide to explore, taper down to exploit
    base_width = max(5, int(num_seqs))
    start_width = int(base_width * 1.5)
    min_width = max(2, int(base_width * 0.5))

    # Beam State: (cost_heuristic, schedule_list, remaining_list)
    beam = [(0, [], list(range(num_txns)))]

    for step in range(num_txns):
        # Calculate current width
        width = int(start_width - (step / num_txns) * (start_width - min_width))
        width = max(min_width, width)

        candidates_pool = []

        for _, p_sched, p_remain in beam:

            # Sort remaining by Urgency
            sorted_remain = sorted(p_remain, key=lambda x: urgency[x], reverse=True)

            # Candidate Selection
            # 1. Top K Deterministic
            to_check = set(sorted_remain[:3])

            # 2. Weighted Random
            if len(sorted_remain) > 3:
                pool = sorted_remain[3:]
                weights = [urgency[x] for x in pool]
                samples = random.choices(pool, weights=weights, k=min(2, len(pool)))
                to_check.update(samples)

            # Lookahead Setup
            best_urgent = sorted_remain[0]
            second_best = sorted_remain[1] if len(sorted_remain) > 1 else None

            for c in to_check:
                new_sched = p_sched + [c]

                # Lookahead Metric: Cost(S + c + next_best)
                probe = best_urgent if c != best_urgent else second_best

                if probe is not None:
                    metric_cost = workload.get_opt_seq_cost(new_sched + [probe])
                else:
                    metric_cost = workload.get_opt_seq_cost(new_sched)

                # Metric: (LookaheadCost, -Urgency)
                metric = (metric_cost, -urgency[c])

                new_remain = list(p_remain)
                new_remain.remove(c)

                candidates_pool.append((metric, new_sched, new_remain))

        # Pruning
        candidates_pool.sort(key=lambda x: x[0])
        beam = [(x[0][0], x[1], x[2]) for x in candidates_pool[:width]]

    # Best Schedule from Beam
    best_state = beam[0]
    current_schedule = best_state[1]
    current_cost = workload.get_opt_seq_cost(current_schedule)

    # --- 3. VARIABLE NEIGHBORHOOD DESCENT (VND) & ILS ---

    def run_vnd(sched, base_cost):
        """
        Variable Neighborhood Descent:
        Alternates between Windowed Insertion (Global) and Adjacent Swap (Local).
        """
        curr_s = list(sched)
        curr_c = base_cost

        # Limit VND loops to prevent timeout
        for _ in range(2):
            improved_in_loop = False

            # Neighborhood 1: Windowed Insertion
            # Shuffling check order avoids directional bias
            check_order = list(range(len(curr_s)))
            random.shuffle(check_order)

            for i in check_order:
                item = curr_s[i]
                temp = curr_s[:i] + curr_s[i+1:]

                window = 12
                start = max(0, i - window)
                end = min(len(temp), i + window)

                best_p = -1
                best_val = curr_c

                for p in range(start, end + 1):
                    cand = temp[:p] + [item] + temp[p:]
                    val = workload.get_opt_seq_cost(cand)
                    if val < best_val:
                        best_val = val
                        best_p = p

                if best_p != -1:
                    curr_s = temp[:best_p] + [item] + temp[best_p:]
                    curr_c = best_val
                    improved_in_loop = True

            # Neighborhood 2: Adjacent Swaps
            # Fast scan to fix local ordering
            swap_improved = False
            for i in range(len(curr_s) - 1):
                # Speculative swap
                curr_s[i], curr_s[i+1] = curr_s[i+1], curr_s[i]
                val = workload.get_opt_seq_cost(curr_s)
                if val < curr_c:
                    curr_c = val
                    swap_improved = True
                    improved_in_loop = True
                else:
                    # Revert
                    curr_s[i], curr_s[i+1] = curr_s[i+1], curr_s[i]

            if not improved_in_loop:
                break

        return curr_c, curr_s

    # Initial Deep Descent
    current_cost, current_schedule = run_vnd(current_schedule, current_cost)

    # ILS Loop
    num_kicks = 3

    for _ in range(num_kicks):
        neighbor = list(current_schedule)

        # Perturbation: Block Move or Multi-Swap
        if random.random() < 0.5:
            # Block Move
            if num_txns > 6:
                bs = random.randint(4, 8)
                src = random.randint(0, num_txns - bs)
                block = neighbor[src : src+bs]
                del neighbor[src : src+bs]
                dst = random.randint(0, len(neighbor))
                neighbor[dst:dst] = block
        else:
            # Multi-Swap
            for _ in range(3):
                i, j = random.randint(0, num_txns-1), random.randint(0, num_txns-1)
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]

        # Repair
        kick_cost = workload.get_opt_seq_cost(neighbor)
        repaired_cost, repaired_sched = run_vnd(neighbor, kick_cost)

        # Acceptance
        if repaired_cost < current_cost:
            current_cost = repaired_cost
            current_schedule = repaired_sched

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