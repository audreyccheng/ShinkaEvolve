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
    Get optimal schedule using Conflict-Volume Lookahead Beam Search and ILS.

    Innovations:
    1. Metric: Conflict Volume (sum of durations of conflicting transactions).
    2. Construction: Beam Search with Lookahead. We evaluate placing a transaction not just
       by immediate cost, but by the cost after also placing the highest-risk remaining transaction.
       This helps avoid painting the schedule into a corner.
    3. Refinement: Iterated Local Search (ILS) with a mix of swap and reinsertion moves.

    Args:
        workload: Workload object
        num_seqs: Budget parameter (controls beam width)

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

    # Urgency Score
    max_vol = max(txn_conflict_vol.values()) if txn_conflict_vol else 1.0
    max_dur = max(txn_durations.values()) if txn_durations else 1.0

    txn_urgency = {}
    for t in range(num_txns):
        # 0.6 Conflict Volume, 0.4 Duration
        # Higher volume means harder to schedule later
        score = 0.6 * (txn_conflict_vol[t] / max_vol) + 0.4 * (txn_durations[t] / max_dur)
        txn_urgency[t] = score

    # --- 2. LOOKAHEAD BEAM SEARCH ---

    BEAM_WIDTH = max(4, int(num_seqs))

    # Beam State: (cost, schedule_list, remaining_list)
    beam = [(0, [], list(range(num_txns)))]

    for step in range(num_txns):
        candidates_pool = []

        for p_cost, p_sched, p_remain in beam:
            # Candidate Selection
            # Sort remaining by urgency
            sorted_remain = sorted(p_remain, key=lambda x: txn_urgency[x], reverse=True)

            # Select subset to evaluate
            to_evaluate = set()

            # Top 3 urgent (Greedy)
            to_evaluate.update(sorted_remain[:3])

            # Weighted sampling from rest (Stochastic)
            if len(sorted_remain) > 3:
                rest = sorted_remain[3:]
                weights = [txn_urgency[x] for x in rest]
                # Sample a few
                try:
                    samples = random.choices(rest, weights=weights, k=min(3, len(rest)))
                    to_evaluate.update(samples)
                except ValueError:
                    pass

            # Lookahead Evaluation
            for cand in to_evaluate:
                # Immediate
                sched_c = p_sched + [cand]
                cost_c = workload.get_opt_seq_cost(sched_c)

                # Lookahead
                # What if we schedule the *next* most urgent item immediately after?
                remain_c = [x for x in p_remain if x != cand]

                metric_val = cost_c

                if remain_c:
                    # Find highest urgency in remaining
                    # This acts as a proxy for "blocking probability"
                    next_heavy = max(remain_c, key=lambda x: txn_urgency[x])

                    sched_lookahead = sched_c + [next_heavy]
                    cost_lookahead = workload.get_opt_seq_cost(sched_lookahead)

                    metric_val = cost_lookahead

                # Metric tuple: (LookaheadCost, ImmediateCost, -Urgency)
                candidates_pool.append(((metric_val, cost_c, -txn_urgency[cand]), sched_c, remain_c))

        # Pruning
        candidates_pool.sort(key=lambda x: x[0])

        # Next beam
        # Note: we store (immediate_cost, sched, remain) in beam,
        # but sort by lookahead metric
        beam = []
        for cand in candidates_pool:
            metric, sched, remain = cand
            if len(beam) < BEAM_WIDTH:
                beam.append((metric[1], sched, remain))
            else:
                break

    best_state = beam[0]
    current_cost = best_state[0]
    current_schedule = best_state[1]

    # --- 3. ITERATED LOCAL SEARCH (ILS) ---

    def run_local_search(sched, start_cost):
        """Windowed re-insertion and swap."""
        best_s = list(sched)
        best_c = start_cost

        improved = True
        while improved:
            improved = False

            # 1. Windowed Re-insertion
            for i in range(len(best_s)):
                txn = best_s[i]
                # Try moving txn to [i-window, i+window]
                window = 8
                low = max(0, i - window)
                high = min(len(best_s), i + window + 1)

                # Remove
                temp = best_s[:i] + best_s[i+1:]

                for pos in range(low, high):
                    if pos == i: continue # approx
                    cand = temp[:pos] + [txn] + temp[pos:]
                    c = workload.get_opt_seq_cost(cand)
                    if c < best_c:
                        best_c = c
                        best_s = cand
                        improved = True
                        break # Take first improvement
                if improved: break

            if improved: continue

            # 2. Adjacent Swaps (Fine tuning)
            for i in range(len(best_s) - 1):
                best_s[i], best_s[i+1] = best_s[i+1], best_s[i]
                c = workload.get_opt_seq_cost(best_s)
                if c < best_c:
                    best_c = c
                    improved = True
                    break
                else:
                    # Revert
                    best_s[i], best_s[i+1] = best_s[i+1], best_s[i]

        return best_c, best_s

    # Initial Local Search
    current_cost, current_schedule = run_local_search(current_schedule, current_cost)

    # ILS Iterations
    # Perturb -> Search -> Accept
    max_iter = 5
    for _ in range(max_iter):
        neighbor = list(current_schedule)

        # Perturbation: Block Swap or Multi-Swap
        if random.random() < 0.5:
            # Block move
            blk_size = random.randint(3, 8)
            src = random.randint(0, len(neighbor) - blk_size)
            block = neighbor[src : src+blk_size]
            del neighbor[src : src+blk_size]
            dst = random.randint(0, len(neighbor))
            neighbor[dst:dst] = block
        else:
            # Random swaps
            for _ in range(4):
                i, j = random.randint(0, len(neighbor)-1), random.randint(0, len(neighbor)-1)
                neighbor[i], neighbor[j] = neighbor[j], neighbor[i]

        # Repair
        neigh_cost = workload.get_opt_seq_cost(neighbor)
        new_cost, new_sched = run_local_search(neighbor, neigh_cost)

        # Accept if better
        if new_cost < current_cost:
            current_cost = new_cost
            current_schedule = new_sched

    return current_cost, current_schedule


def get_random_costs():
    """Evaluate scheduling algorithm on three different workloads."""
    start_time = time.time()

    # Parameter for beam search budget
    num_seqs = 10

    # Process workloads
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