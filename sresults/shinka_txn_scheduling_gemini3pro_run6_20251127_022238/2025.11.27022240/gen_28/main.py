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
    Get optimal schedule using Conflict-Aware Tapered Beam Search and Heavy-Item Refinement.
    
    Args:
        workload: Workload object
        num_seqs: Budget parameter used to scale beam width

    Returns:
        (lowest_makespan, schedule)
    """
    
    num_txns = workload.num_txns
    
    # --- 1. METRIC PRECOMPUTATION ---
    # We analyze conflicts to prioritize "difficult" transactions.
    
    txn_durations = {}
    txn_rw = {}
    
    # Extract data
    for t in range(num_txns):
        # Duration
        try:
            d = workload.txns[t][0][3]
        except:
            d = 1.0
        txn_durations[t] = d
        
        # Read/Write Sets
        reads = set()
        writes = set()
        try:
            ops_str = workload.txns[t][0][1]
            if isinstance(ops_str, str):
                for op in ops_str.split():
                    if '-' in op:
                        type_, key = op.split('-')
                        if type_ == 'r': reads.add(key)
                        elif type_ == 'w': writes.add(key)
        except:
            pass
        txn_rw[t] = (reads, writes)

    # Compute Conflict Volume
    # Volume = Sum of durations of all conflicting transactions.
    # This metric identifies transactions that cause the most "blocking".
    txn_conflict_vol = {t: 0.0 for t in range(num_txns)}
    
    # O(N^2) conflict check
    for i in range(num_txns):
        r1, w1 = txn_rw[i]
        vol = 0.0
        for j in range(num_txns):
            if i == j: continue
            r2, w2 = txn_rw[j]
            # Conflict Check: W1 overlaps W2 or R2, OR R1 overlaps W2
            if not w1.isdisjoint(w2) or not w1.isdisjoint(r2) or not r1.isdisjoint(w2):
                vol += txn_durations[j]
        txn_conflict_vol[i] = vol

    # Combined Heuristic for Candidate Selection
    # We want to greedily pick items that are High Conflict Volume AND Long Duration.
    # Normalize for combination.
    max_vol = max(txn_conflict_vol.values()) if txn_conflict_vol else 1.0
    max_dur = max(txn_durations.values()) if txn_durations else 1.0
    if max_vol == 0: max_vol = 1.0
    
    txn_heuristic_score = {}
    for t in range(num_txns):
        # 0.6 weight on conflict, 0.4 on duration.
        # Resolving conflicts is slightly more important than just fitting long items.
        score = 0.6 * (txn_conflict_vol[t] / max_vol) + 0.4 * (txn_durations[t] / max_dur)
        txn_heuristic_score[t] = score

    # --- 2. TAPERED BEAM SEARCH ---
    
    # Width setup: Start wider to establish good roots, taper to focus computation.
    base_width = max(8, int(num_seqs))
    start_width = int(base_width * 1.5)
    end_width = max(3, int(base_width * 0.5))
    
    # Beam State: (cost, schedule_list, remaining_list)
    beam = [(0, [], list(range(num_txns)))]
    
    for step in range(num_txns):
        # Calculate current beam width based on progress
        progress = step / num_txns
        width = int(start_width + (end_width - start_width) * progress)
        width = max(2, width)
        
        candidates_pool = []
        
        for p_cost, p_sched, p_remain in beam:
            next_candidates = set()
            
            # Strategy 1: Exhaustive Tail
            # If few items left, check all to pack perfectly.
            if len(p_remain) <= 20:
                next_candidates.update(p_remain)
            else:
                # Strategy 2: Heuristic Top-K
                # Sort remaining by heuristic score
                sorted_rem = sorted(p_remain, key=lambda x: txn_heuristic_score[x], reverse=True)
                
                # Pick top 3 (Deterministic Greedy)
                next_candidates.update(sorted_rem[:3])
                
                # Strategy 3: Weighted Sampling
                # Sample from the rest to avoid local optima
                rest = sorted_rem[3:]
                if rest:
                    weights = [txn_heuristic_score[x] for x in rest]
                    # Sample 4 items
                    samples = random.choices(rest, weights=weights, k=min(4, len(rest)))
                    next_candidates.update(samples)
                    
                    # Strategy 4: Pure Random (Low probability fallback)
                    next_candidates.update(random.sample(rest, min(2, len(rest))))
            
            # Evaluate Candidates
            for cand in next_candidates:
                new_sched = p_sched + [cand]
                cost = workload.get_opt_seq_cost(new_sched)
                
                # Pruning Metric:
                # 1. Minimize Cost (Makespan)
                # 2. Tie-breaker: Maximize Conflict Volume of added txn (cleared a hard item)
                # 3. Tie-breaker: Maximize Duration of added txn
                metric = (cost, -txn_conflict_vol[cand], -txn_durations[cand])
                
                new_remain = list(p_remain)
                new_remain.remove(cand)
                
                candidates_pool.append((metric, new_sched, new_remain))
        
        # Sort and Prune
        candidates_pool.sort(key=lambda x: x[0])
        
        # Extract top 'width' states
        # Deduplication based on cost check? No, straightforward slicing is standard for beam search
        beam = [(x[0][0], x[1], x[2]) for x in candidates_pool[:width]]

    # Extract best result from beam
    best_state = beam[0]
    current_cost = best_state[0]
    current_schedule = best_state[1]

    # --- 3. REFINEMENT: HEAVY ITEM RE-INSERTION ---
    # Targeted optimization for the most problematic transactions.
    # We take the top K "heaviest" items and try to find their optimal spot in the schedule.
    
    heavy_items = sorted(range(num_txns), key=lambda x: txn_heuristic_score[x], reverse=True)[:15]
    
    for item in heavy_items:
        try:
            curr_idx = current_schedule.index(item)
        except ValueError:
            continue
            
        # Remove item
        temp_sched = list(current_schedule)
        temp_sched.pop(curr_idx)
        
        best_pos_cost = current_cost
        best_pos_sched = None
        
        # Search Window: Local area around original position is most promising
        # but large shifts can sometimes fix major blockages.
        # We use a window +/- 15
        low = max(0, curr_idx - 15)
        high = min(len(temp_sched), curr_idx + 15)
        
        for pos in range(low, high + 1):
            test_sched = list(temp_sched)
            test_sched.insert(pos, item)
            
            c = workload.get_opt_seq_cost(test_sched)
            
            if c < best_pos_cost:
                best_pos_cost = c
                best_pos_sched = test_sched
        
        if best_pos_sched:
            current_cost = best_pos_cost
            current_schedule = best_pos_sched

    # --- 4. REFINEMENT: STOCHASTIC SHIFT/SWAP ---
    # Standard hill climbing to catch any remaining small optimizations
    search_steps = 500
    no_improv_limit = 100
    no_improv = 0
    
    for _ in range(search_steps):
        if no_improv >= no_improv_limit:
            break
            
        neighbor = list(current_schedule)
        
        # 80% Shift, 20% Swap
        if random.random() < 0.8:
            src = random.randint(0, num_txns - 1)
            dst = random.randint(0, num_txns - 1)
            if src == dst: continue
            
            val = neighbor.pop(src)
            neighbor.insert(dst, val)
        else:
            i = random.randint(0, num_txns - 1)
            j = random.randint(0, num_txns - 1)
            if i == j: continue
            neighbor[i], neighbor[j] = neighbor[j], neighbor[i]
            
        new_cost = workload.get_opt_seq_cost(neighbor)
        
        if new_cost < current_cost:
            current_cost = new_cost
            current_schedule = neighbor
            no_improv = 0
        else:
            no_improv += 1

    return current_cost, current_schedule


def get_random_costs():
    """Evaluate scheduling algorithm on three different workloads."""
    start_time = time.time()
    
    # Budget parameter
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
