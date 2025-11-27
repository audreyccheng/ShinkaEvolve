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
    Get optimal schedule using Volume-Weighted Beam Search and Multi-Cycle ILS.

    Innovations:
    1. Conflict Volume: Measures conflict impact by duration of conflicting tasks, not just count.
    2. Beam Search: Uses Urgency (Duration + Conflict Vol) to guide expansion and pruning.
    3. ILS: Structured loop of Perturbation (Kick) and Repair (Windowed Insertion Descent).

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
    # Sum of durations of all transactions that conflict with T
    txn_conflict_vol = {t: 0.0 for t in range(num_txns)}

    # Only compute if N is reasonable to avoid O(N^2) overhead on huge sets
    if num_txns < 1000:
        for i in range(num_txns):
            r1, w1 = txn_rw_sets[i]
            vol = 0.0
            for j in range(num_txns):
                if i == j: continue
                r2, w2 = txn_rw_sets[j]
                # Conflict: (W1 n (W2 u R2)) or (R1 n W2)
                if not w1.isdisjoint(w2) or not w1.isdisjoint(r2) or not r1.isdisjoint(w2):
                    vol += txn_durations[j]
            txn_conflict_vol[i] = vol

    # Urgency Score: Normalize and combine
    max_vol = max(txn_conflict_vol.values()) if txn_conflict_vol else 1.0
    if max_vol == 0: max_vol = 1.0
    max_dur = max(txn_durations.values()) if txn_durations else 1.0

    txn_urgency = {}
    for t in range(num_txns):
        # Weight formula: Duration + 0.5 * ConflictVolume
        # We want long items early, and high conflict items early.
        # Normalization helps balance the two components.
        u = (txn_durations[t] / max_dur) + 0.5 * (txn_conflict_vol[t] / max_vol)
        txn_urgency[t] = u

    # --- 2. BEAM SEARCH ---

    BEAM_WIDTH = max(5, int(num_seqs))

    # Beam State: (cost, schedule_list, remaining_list)
    beam = [(0, [], list(range(num_txns)))]

    for _ in range(num_txns):
        candidates_pool = []

        for p_cost, p_sched, p_remain in beam:

            # Expansion Strategy:
            # Sort remaining by urgency
            sorted_remain = sorted(p_remain, key=lambda x: txn_urgency[x], reverse=True)

            next_candidates = set()

            # Top 2 Deterministic
            next_candidates.update(sorted_remain[:2])

            # Weighted Sample 3
            if len(sorted_remain) > 2:
                pool = sorted_remain[2:]
                weights = [txn_urgency[x] for x in pool]
                if pool:
                    samples = random.choices(pool, weights=weights, k=min(3, len(pool)))
                    next_candidates.update(samples)

            # Pure Random 1
            if len(sorted_remain) > 10:
                next_candidates.update(random.sample(sorted_remain, 1))

            # Evaluate
            for c in next_candidates:
                new_sched = p_sched + [c]
                cost = workload.get_opt_seq_cost(new_sched)

                # Pruning Metric: (Cost, -Urgency)
                metric = (cost, -txn_urgency[c])

                new_remain = list(p_remain)
                new_remain.remove(c)

                candidates_pool.append((metric, new_sched, new_remain))

        # Select best
        candidates_pool.sort(key=lambda x: x[0])
        beam = [(x[0][0], x[1], x[2]) for x in candidates_pool[:BEAM_WIDTH]]

    best_state = beam[0]
    current_cost = best_state[0]
    current_schedule = best_state[1]

    # --- 3. ITERATED LOCAL SEARCH (ILS) ---

    def windowed_descent(sched, current_c, passes=1, window_size=10):
        """
        Attempts to improve schedule by removing each txn and re-inserting
        it at the best position within a local window.
        """
        best_s = list(sched)
        best_c = current_c

        for _ in range(passes):
            improved = False
            # Randomize order to avoid bias
            indices = list(range(len(best_s)))
            random.shuffle(indices)

            for i in indices:
                txn = best_s[i]

                # Create base by removing txn
                temp_s = best_s[:i] + best_s[i+1:]

                # Define window
                start = max(0, i - window_size)
                end = min(len(temp_s), i + window_size)

                # Scan window
                local_best_pos = -1
                local_min_c = best_c

                for pos in range(start, end + 1):
                    cand = temp_s[:pos] + [txn] + temp_s[pos:]
                    c = workload.get_opt_seq_cost(cand)
                    if c < local_min_c:
                        local_min_c = c
                        local_best_pos = pos

                if local_best_pos != -1:
                    best_s = temp_s[:local_best_pos] + [txn] + temp_s[local_best_pos:]
                    best_c = local_min_c
                    improved = True

            if not improved:
                break

        return best_c, best_s

    # Initial Descent
    current_cost, current_schedule = windowed_descent(current_schedule, current_cost, passes=2)

    # ILS Loop
    # Number of kicks/restarts
    num_kicks = 4

    for _ in range(num_kicks):
        # 1. Perturb (Kick)
        candidate_schedule = list(current_schedule)

        # Swap 3 random pairs
        for _ in range(3):
            idx1, idx2 = random.randint(0, num_txns-1), random.randint(0, num_txns-1)
            candidate_schedule[idx1], candidate_schedule[idx2] = candidate_schedule[idx2], candidate_schedule[idx1]

        kick_cost = workload.get_opt_seq_cost(candidate_schedule)

        # 2. Repair (Descent)
        # Use single pass for speed in loop
        new_cost, new_sched = windowed_descent(candidate_schedule, kick_cost, passes=1)

        # 3. Accept
        if new_cost < current_cost:
            current_cost = new_cost
            current_schedule = new_sched

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