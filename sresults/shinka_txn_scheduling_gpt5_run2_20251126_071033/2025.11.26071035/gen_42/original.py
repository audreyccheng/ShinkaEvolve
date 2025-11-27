# EVOLVE-BLOCK-START
"""Transaction scheduling algorithm for optimizing makespan across multiple workloads
Crossover: GRASP-style randomized best-insertion + cached evaluations + VND local search.
"""

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
    GRASP + cached evaluation + VND local search to minimize makespan.

    Args:
        workload: Workload object containing transaction data
        num_seqs: Number of randomized restarts (search effort)

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    n = workload.num_txns
    all_txns = list(range(n))

    # Parameter configuration (adaptive by workload size)
    if n <= 50:
        CAND_SAMPLE_BASE = 10     # candidate txn per step
        POS_SAMPLE_LIMIT = None   # evaluate all insertion positions
        MAX_LS_PASSES = 3         # local search passes for adjacent swaps
        RELOC_TRIES = max(10, n)  # relocation trials for VND
        STARTER_SAMPLE = min(8, n)
        RCL_K = 3
    else:
        CAND_SAMPLE_BASE = 8
        POS_SAMPLE_LIMIT = 15     # cap insertion positions evaluated
        MAX_LS_PASSES = 2
        RELOC_TRIES = max(12, n // 2)
        STARTER_SAMPLE = min(10, n)
        RCL_K = 3

    JITTER = 2  # small random variation in candidate set size

    # Cost cache for partial prefixes to reduce repeated evaluations
    cost_cache = {}

    def eval_cost(prefix):
        """Evaluate and cache the cost for a (possibly partial) prefix sequence."""
        key = tuple(prefix)
        if key in cost_cache:
            return cost_cache[key]
        c = workload.get_opt_seq_cost(list(prefix))
        cost_cache[key] = c
        return c

    def sample_positions(seq_len):
        """Return list of positions [0..seq_len] where insertion can occur, with optional sampling."""
        if POS_SAMPLE_LIMIT is None or seq_len + 1 <= (POS_SAMPLE_LIMIT or (seq_len + 1)):
            return list(range(seq_len + 1))
        # Sample positions but always include ends to preserve global structure
        num_to_sample = max(2, min(POS_SAMPLE_LIMIT, seq_len + 1))
        mandatory = {0, seq_len}
        interior = list(range(1, seq_len))
        if len(interior) <= num_to_sample - 2:
            chosen = set(interior)
        else:
            chosen = set(random.sample(interior, num_to_sample - 2))
        chosen.update(mandatory)
        return sorted(chosen)

    def best_insertion_for_txn(current_seq, txn):
        """
        Try inserting txn into multiple positions in current_seq.
        Returns (best_cost, best_pos, second_best_cost) to enable regret-based selection.
        """
        seq_len = len(current_seq)
        best_cost = float('inf')
        second_best = float('inf')
        best_pos = 0
        positions = sample_positions(seq_len)
        # Evaluate each possible position, track the two best costs
        for pos in positions:
            cand = current_seq.copy()
            cand.insert(pos, txn)
            cost = eval_cost(cand)
            if cost < best_cost:
                second_best = best_cost
                best_cost = cost
                best_pos = pos
            elif cost < second_best:
                second_best = cost
        # In case we only had one position, second_best may remain inf; normalize
        if second_best == float('inf'):
            second_best = best_cost
        return best_cost, best_pos, second_best

    def select_best_starter():
        """
        Pick a robust starting transaction by sampling a subset and choosing best by prefix cost.
        """
        candidates = random.sample(all_txns, STARTER_SAMPLE) if STARTER_SAMPLE < n else all_txns
        best_t = None
        best_c = float('inf')
        for t in candidates:
            c = eval_cost([t])
            if c < best_c:
                best_c = c
                best_t = t
        return best_t if best_t is not None else random.randint(0, n - 1)

    def construct_sequence():
        """
        GRASP randomized best-insertion construction with regret-weighted RCL.
        """
        remaining = all_txns.copy()
        start_txn = select_best_starter()
        seq = [start_txn]
        remaining.remove(start_txn)
        curr_cost = eval_cost(seq)

        # Try to find a good second transaction by testing a small candidate set thoroughly
        if remaining:
            k = min(6, len(remaining))
            second_cands = random.sample(remaining, k)
            best_pair_cost = float('inf')
            best_txn = None
            best_pos = 1  # only positions 0 or 1 possible
            for t in second_cands:
                for pos in [0, 1]:
                    cand = seq.copy()
                    cand.insert(pos, t)
                    cost = eval_cost(cand)
                    if cost < best_pair_cost:
                        best_pair_cost = cost
                        best_txn = t
                        best_pos = pos
            if best_txn is not None:
                seq.insert(best_pos, best_txn)
                remaining.remove(best_txn)
                curr_cost = best_pair_cost

        # Build the rest using sampled candidates and best insertion positions + regret-weighted RCL selection
        while remaining:
            # Adaptive candidate sample size with slight randomness
            dynamic_base = CAND_SAMPLE_BASE
            cand_size = min(len(remaining), max(3, dynamic_base + random.randint(-JITTER, JITTER)))
            cand_txns = random.sample(remaining, cand_size)

            step_evals = []
            for t in cand_txns:
                cost_t, pos_t, second_cost_t = best_insertion_for_txn(seq, t)
                step_evals.append((cost_t, t, pos_t, second_cost_t))

            # Sort by best insertion cost
            step_evals.sort(key=lambda x: x[0])

            # Build RCL: include a few best candidates, then choose weighted by regret
            rcl_size = min(max(3, RCL_K), len(step_evals))
            rcl = step_evals[:rcl_size]

            # Regret = (second_best_cost - best_cost). Prefer higher regret among good candidates.
            regrets = [max(0.0, s2 - s1) for (s1, _, _, s2) in rcl]
            total_regret = sum(regrets)

            if total_regret > 0 and random.random() < 0.7:
                # Pick by regret weights among RCL to emphasize critical placements
                choices = [item for item in rcl]
                weights = regrets
                chosen = random.choices(choices, weights=weights, k=1)[0]
                best_step_cost, best_txn, best_pos, _ = chosen
            else:
                # Greedy fallback among RCL
                best_step_cost, best_txn, best_pos, _ = rcl[0]

            seq.insert(best_pos, best_txn)
            curr_cost = best_step_cost
            remaining.remove(best_txn)

        return seq, curr_cost

    def local_search_adjacent_swaps(seq, curr_cost, max_passes=MAX_LS_PASSES):
        """
        Perform multiple passes of adjacent swap hill-climbing.
        """
        improved = True
        passes = 0
        best_seq = seq
        best_cost = curr_cost

        while improved and passes < max_passes:
            improved = False
            passes += 1
            i = 0
            while i < len(best_seq) - 1:
                cand = best_seq.copy()
                cand[i], cand[i + 1] = cand[i + 1], cand[i]
                c = eval_cost(cand)
                if c < best_cost:
                    best_cost = c
                    best_seq = cand
                    improved = True
                i += 1
        return best_seq, best_cost

    def local_search_relocations(seq, curr_cost, tries=RELOC_TRIES):
        """
        Random relocation: remove at index i and reinsert at j if improves.
        Accepts first improving move and continues until no improvement across budgeted tries.
        """
        best_seq = list(seq)
        best_cost = curr_cost

        if len(best_seq) <= 2:
            return best_seq, best_cost

        trials = 0
        while trials < tries:
            i = random.randint(0, len(best_seq) - 1)
            j = random.randint(0, len(best_seq) - 1)
            if i == j:
                trials += 1
                continue
            cand = best_seq.copy()
            t = cand.pop(i)
            if j > i:
                j -= 1
            cand.insert(j, t)
            c = eval_cost(cand)
            if c < best_cost:
                best_cost = c
                best_seq = cand
                trials = 0  # reset upon improvement
            else:
                trials += 1
        return best_seq, best_cost

    def or_opt_block(seq, curr_cost, k):
        """
        Or-opt move with block size k: relocate any contiguous block of length k to its best position.
        Performs best-improving passes until no improvement.
        """
        best_seq = list(seq)
        best_cost = curr_cost
        if len(best_seq) <= k:
            return best_seq, best_cost

        improved = True
        while improved:
            improved = False
            move_best_cost = best_cost
            move = None  # (i, pos)
            L = len(best_seq)
            for i in range(0, L - k + 1):
                block = best_seq[i:i + k]
                base = best_seq[:i] + best_seq[i + k:]
                positions = sample_positions(len(base))
                for pos in positions:
                    cand = base[:]
                    cand[pos:pos] = block
                    c = eval_cost(cand)
                    if c < move_best_cost:
                        move_best_cost = c
                        move = (i, pos)
            if move is not None:
                i, pos = move
                block = best_seq[i:i + k]
                base = best_seq[:i] + best_seq[i + k:]
                new_seq = base[:]
                new_seq[pos:pos] = block
                best_seq = new_seq
                best_cost = move_best_cost
                improved = True
        return best_seq, best_cost

    def vnd_local_search(seq, curr_cost):
        """
        Variable Neighborhood Descent:
        Or-opt blocks k=3,2,1 (best-improving), then adjacent swaps, then light relocations.
        Repeat cycle until no further improvement.
        """
        best_seq = list(seq)
        best_cost = curr_cost
        changed = True
        while changed:
            changed = False
            for k in (3, 2, 1):
                s, c = or_opt_block(best_seq, best_cost, k)
                if c < best_cost:
                    best_seq, best_cost = s, c
                    changed = True
            s, c = local_search_adjacent_swaps(best_seq, best_cost, max_passes=1)
            if c < best_cost:
                best_seq, best_cost = s, c
                changed = True
            s, c = local_search_relocations(best_seq, best_cost, tries=max(8, RELOC_TRIES // 2))
            if c < best_cost:
                best_seq, best_cost = s, c
                changed = True
        return best_seq, best_cost

    def lns_ruin_recreate(seq, curr_cost, rounds=2):
        """
        Ruin-and-recreate LNS: remove a contiguous block (plus a few random extras) and reinsert greedily.
        Accept only improving reconstructions; repeat for a few rounds.
        """
        best_seq = list(seq)
        best_cost = curr_cost
        for _ in range(max(1, rounds)):
            L = len(best_seq)
            if L < 6:
                break
            # Choose block to remove
            base_len = max(4, L // 10)
            block_len = min(L - 2, base_len + random.randint(0, 3))
            start = random.randint(0, L - block_len)
            removed = best_seq[start:start + block_len]
            skeleton = best_seq[:start] + best_seq[start + block_len:]
            # Optionally remove a few extras for diversification
            extra_count = min(3, max(0, L // 30))
            extras_idx = sorted(random.sample(range(len(skeleton)), extra_count)) if extra_count and len(skeleton) > extra_count else []
            extras = []
            offset = 0
            for idx in extras_idx:
                idx_adj = idx - offset
                extras.append(skeleton.pop(idx_adj))
                offset += 1
            to_insert = removed + extras

            # Reinsert removed items using best insertion positions
            rebuilt = list(skeleton)
            for t in to_insert:
                best_c = float('inf')
                best_p = 0
                positions = sample_positions(len(rebuilt))
                for pos in positions:
                    cand = rebuilt[:]
                    cand.insert(pos, t)
                    c = eval_cost(cand)
                    if c < best_c:
                        best_c = c
                        best_p = pos
                rebuilt.insert(best_p, t)

            c_final = eval_cost(rebuilt)
            if c_final < best_cost:
                best_cost = c_final
                best_seq = rebuilt
        return best_seq, best_cost

    # Multi-start GRASP with VND + LNS refinement
    restarts = max(1, num_seqs)
    global_best_seq = None
    global_best_cost = float('inf')

    for r in range(restarts):
        # Diverse construction
        seq, cost = construct_sequence()
        # Strong local search via VND (Or-opt blocks + swaps + relocations)
        seq, cost = vnd_local_search(seq, cost)
        # Ruin-and-recreate to escape local minima, then polish again if improved
        seq_rr, cost_rr = lns_ruin_recreate(seq, cost, rounds=2 if n > 50 else 3)
        if cost_rr < cost:
            seq, cost = vnd_local_search(seq_rr, cost_rr)
        else:
            # quick polish
            seq, cost = local_search_adjacent_swaps(seq, cost, 1)

        if cost < global_best_cost:
            global_best_cost = cost
            global_best_seq = seq

    # Safety checks
    assert global_best_seq is not None
    assert len(global_best_seq) == n
    assert len(set(global_best_seq)) == n

    return global_best_cost, global_best_seq


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