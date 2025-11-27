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
    Get optimal schedule using Efficiency-Aware Beam Search followed by
    Adaptive SA with Hybrid Ruin-and-Recreate and Deterministic Polish.

    Args:
        workload: Workload object containing transaction data
        num_seqs: Parameter affecting the computational budget

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    # --- Hyperparameters ---
    # Width 8 provides better structural coverage for complex dependency chains
    BEAM_WIDTH = 8

    # SA Parameters
    # High iterations to thoroughly explore the neighborhood of the beam result
    SA_ITERATIONS = 4000
    SA_COOLING_RATE = 0.9985
    SA_START_TEMP_RATIO = 0.05

    # Ruin-and-Recreate Parameters
    STAGNATION_LIMIT = 200 # Aggressive restart to escape local optima
    RUIN_BLOCK_SIZE_MIN = 2
    RUIN_BLOCK_SIZE_MAX = 6

    # Adaptive Operator Weights
    OP_WEIGHTS = {
        'swap': 2.0,
        'insert': 8.0,
        'block_move': 4.0,
        'reverse': 1.0
    }
    OP_MIN_WEIGHT = 0.5
    OP_ADAPTATION_RATE = 0.1

    # --- Cost Cache ---
    cost_cache = {}

    def get_cost(seq):
        t_seq = tuple(seq)
        if t_seq in cost_cache:
            return cost_cache[t_seq]
        c = workload.get_opt_seq_cost(seq)
        cost_cache[t_seq] = c
        return c

    # --- Pre-calculation: Transaction Weights ---
    # Used for efficiency calculation and LPT sorting
    txn_weights = {}
    for t in range(workload.num_txns):
        txn_weights[t] = get_cost([t])

    # --- Phase 1: Efficiency-Aware Beam Search ---
    # Gamma decays from high (prioritizing parallelism/efficiency) to low (minimizing global cost)
    # This helps pack complementary transactions early when flexibility is high.
    GAMMA_START = 1.5
    GAMMA_END = 0.5

    candidates = []
    for t in range(workload.num_txns):
        seq = [t]
        cost = get_cost(seq)
        w = txn_weights[t]
        # Initial score: maximize weighted potential
        score = cost - (GAMMA_START * w)
        candidates.append({
            'score': score, 
            'cost': cost, 
            'seq': seq, 
            'rem': {x for x in range(workload.num_txns) if x != t}
        })

    candidates.sort(key=lambda x: x['score'])
    beam = candidates[:BEAM_WIDTH]

    total_txns = workload.num_txns

    for _ in range(total_txns - 1):
        next_candidates = []
        
        # Calculate dynamic gamma based on schedule completeness
        # Use first beam node to estimate progress (all nodes have same length in this step)
        current_len = len(beam[0]['seq'])
        progress = current_len / total_txns
        gamma = GAMMA_START * (1.0 - progress) + GAMMA_END * progress

        for node in beam:
            b_seq = node['seq']
            b_rem = node['rem']
            b_cost = node['cost']

            for cand in b_rem:
                new_seq = b_seq + [cand]
                new_cost = get_cost(new_seq)
                
                delta = new_cost - b_cost
                w = txn_weights[cand]

                # Efficiency Metric:
                # Calculate how much of the transaction's standalone cost is "hidden" by parallelism.
                # hidden = weight - cost_increase. 
                # If delta >= weight, hidden is 0 (or negative, clamped to 0).
                hidden = max(0.0, w - delta)
                
                # Score: Minimize cost, Maximize hidden latency (weighted by gamma)
                score = new_cost - (gamma * hidden)
                
                next_candidates.append((score, new_cost, new_seq, b_rem, cand))

        # Greedy pruning
        next_candidates.sort(key=lambda x: x[0])

        new_beam = []
        for c_score, c_cost, c_seq, c_parent_rem, c_cand in next_candidates:
            if len(new_beam) >= BEAM_WIDTH:
                break

            new_rem = c_parent_rem.copy()
            new_rem.remove(c_cand)
            new_beam.append({'cost': c_cost, 'seq': c_seq, 'rem': new_rem})

        beam = new_beam

    if not beam:
        return float('inf'), []

    current_schedule = beam[0]['seq']
    current_cost = beam[0]['cost']

    # --- Phase 2: Adaptive SA with Hybrid Ruin-and-Recreate ---
    best_schedule = list(current_schedule)
    best_cost = current_cost

    T = current_cost * SA_START_TEMP_RATIO
    ops = list(OP_WEIGHTS.keys())
    steps_since_imp = 0

    for it in range(SA_ITERATIONS):
        # 1. Ruin-and-Recreate (Kick) on Stagnation
        if steps_since_imp > STAGNATION_LIMIT:
            kick_seq = list(best_schedule)
            n = len(kick_seq)
            removed_items = []

            # Hybrid Ruin Strategy:
            # 50%: Multi-Segment Ruin (blocks)
            # 50%: Scatter Ruin (random items)
            strategy = random.random()

            if strategy < 0.5:
                # Multi-Segment (2 blocks)
                for _ in range(2):
                    if len(kick_seq) > RUIN_BLOCK_SIZE_MIN:
                        sz = random.randint(RUIN_BLOCK_SIZE_MIN, min(len(kick_seq), RUIN_BLOCK_SIZE_MAX))
                        start = random.randint(0, len(kick_seq) - sz)
                        removed_items.extend(kick_seq[start : start+sz])
                        del kick_seq[start : start+sz]
            else:
                # Scatter Ruin
                if n > RUIN_BLOCK_SIZE_MIN:
                    num_to_remove = random.randint(3, 8)
                    indices = sorted(random.sample(range(n), num_to_remove), reverse=True)
                    for idx in indices:
                        removed_items.append(kick_seq.pop(idx))

            # Recreate: Greedy Best-Fit
            random.shuffle(removed_items)

            for item in removed_items:
                best_pos = -1
                min_c = float('inf')

                # Try all positions
                for i in range(len(kick_seq) + 1):
                    kick_seq.insert(i, item)
                    c = get_cost(kick_seq)
                    if c < min_c:
                        min_c = c
                        best_pos = i
                    kick_seq.pop(i)

                kick_seq.insert(best_pos, item)

            current_schedule = kick_seq
            current_cost = min_c

            if current_cost < best_cost:
                best_cost = current_cost
                best_schedule = list(current_schedule)
                steps_since_imp = 0
            else:
                steps_since_imp = 0

            T = max(T, current_cost * 0.05)
            continue

        # 2. Operator Selection
        total_w = sum(OP_WEIGHTS.values())
        r = random.uniform(0, total_w)
        cum = 0
        op = ops[0]
        for o in ops:
            cum += OP_WEIGHTS[o]
            if r <= cum:
                op = o
                break

        neighbor = list(current_schedule)
        n = len(neighbor)

        if op == 'swap':
            i, j = random.sample(range(n), 2)
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
        elif op == 'insert':
            i = random.randint(0, n - 1)
            val = neighbor.pop(i)
            j = random.randint(0, n - 1)
            neighbor.insert(j, val)
        elif op == 'reverse':
            i, j = sorted(random.sample(range(n), 2))
            neighbor[i:j+1] = neighbor[i:j+1][::-1]
        elif op == 'block_move':
            b_size = random.randint(2, 6)
            if n > b_size:
                i = random.randint(0, n - b_size)
                block = neighbor[i:i+b_size]
                del neighbor[i:i+b_size]
                j = random.randint(0, len(neighbor))
                neighbor[j:j] = block

        new_cost = get_cost(neighbor)
        delta = new_cost - current_cost

        # 3. Acceptance
        accept = False
        if delta < 0:
            accept = True
            if new_cost < best_cost:
                best_cost = new_cost
                best_schedule = list(neighbor)
                steps_since_imp = 0
            else:
                steps_since_imp += 1
        else:
            steps_since_imp += 1
            if T > 1e-9 and random.random() < math.exp(-delta / T):
                accept = True

        if accept:
            current_schedule = neighbor
            current_cost = new_cost

        # 4. Adapt Weights
        reward = 0.1
        if delta < 0:
            reward = 1.0 if new_cost >= best_cost else 2.0

        OP_WEIGHTS[op] = (1 - OP_ADAPTATION_RATE) * OP_WEIGHTS[op] + OP_ADAPTATION_RATE * (OP_MIN_WEIGHT + reward)

        # 5. Cooling
        T *= SA_COOLING_RATE

    # --- Phase 3: Deterministic Polish (LPT Ordered) ---
    # Sort transactions by weight descending (Longest Processing Time)
    sorted_txns = sorted(range(workload.num_txns), key=lambda x: txn_weights[x], reverse=True)

    polish_seq = list(best_schedule)
    current_polish_cost = best_cost

    # Run multiple passes to allow schedule to settle
    MAX_PASSES = 10
    for pass_idx in range(MAX_PASSES):
        search_active = False # Tracks if we should continue passes (change or improvement)

        for item in sorted_txns:
            # Find current position and remove
            try:
                curr_idx = polish_seq.index(item)
            except ValueError:
                continue

            polish_seq.pop(curr_idx)

            best_pos = -1
            min_c = float('inf')

            # Exhaustive scan for best position
            # Left-Packing Bias: Strict inequality (<) prefers earlier positions 
            # effectively compacting the schedule when costs are equal.
            for j in range(len(polish_seq) + 1):
                polish_seq.insert(j, item)
                c = get_cost(polish_seq)
                if c < min_c:
                    min_c = c
                    best_pos = j
                polish_seq.pop(j)

            # Insert at best position
            polish_seq.insert(best_pos, item)

            # Check if schedule improved or changed structure
            if min_c < current_polish_cost:
                current_polish_cost = min_c
                search_active = True
            elif best_pos != curr_idx:
                # Structural change even if cost is same - allows traversing plateaus
                search_active = True

        if current_polish_cost < best_cost:
            best_cost = current_polish_cost
            best_schedule = list(polish_seq)

        if not search_active:
            break

    return best_cost, best_schedule


def get_random_costs():
    """
    Evaluate scheduling algorithm on three different workloads.

    Returns:
        Tuple of (total_makespan, list_of_schedules, execution_time)
    """
    start_time = time.time()
    workload_size = 100

    # Workload 1: Complex mixed read/write transactions
    workload = Workload(WORKLOAD_1)
    makespan1, schedule1 = get_best_schedule(workload, 10)
    cost1 = workload.get_opt_seq_cost(schedule1)

    # Workload 2: Simple read-then-write pattern
    workload2 = Workload(WORKLOAD_2)
    makespan2, schedule2 = get_best_schedule(workload2, 10)
    cost2 = workload2.get_opt_seq_cost(schedule2)

    # Workload 3: Minimal read/write operations
    workload3 = Workload(WORKLOAD_3)
    makespan3, schedule3 = get_best_schedule(workload3, 10)
    cost3 = workload3.get_opt_seq_cost(schedule3)

    total_makespan = cost1 + cost2 + cost3
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