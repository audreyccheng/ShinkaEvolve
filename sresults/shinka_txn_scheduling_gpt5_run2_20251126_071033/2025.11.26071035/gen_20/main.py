# EVOLVE-BLOCK-START
"""Transaction scheduling algorithm for optimizing makespan across multiple workloads
Enhanced with GRASP-style randomized best-insertion and local search refinement.
"""

import time
from functools import lru_cache
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
    GRASP-style scheduler: multi-start randomized best-insertion + local search.

    Args:
        workload: Workload object containing transaction data
        num_seqs: Number of randomized restarts

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
        RELOC_TRIES = max(8, n // 2)
    else:
        CAND_SAMPLE_BASE = 8
        POS_SAMPLE_LIMIT = 15     # cap insertion positions evaluated
        MAX_LS_PASSES = 2
        RELOC_TRIES = max(12, n // 3)

    JITTER = 2  # small random variation in candidate set size

    @lru_cache(maxsize=50000)
    def _eval_cost_tuple(seq_tuple):
        # Cached evaluation for sequences (or prefixes) represented as tuples
        return workload.get_opt_seq_cost(list(seq_tuple))

    def eval_cost(seq):
        # Wrapper accepting list; converts to tuple key for caching
        return _eval_cost_tuple(tuple(seq))

    def sample_positions(seq_len):
        # Return list of positions [0..seq_len] where insertion can occur
        if POS_SAMPLE_LIMIT is None or seq_len + 1 <= (POS_SAMPLE_LIMIT or (seq_len + 1)):
            return list(range(seq_len + 1))
        # Sample positions but always include ends to preserve global structure
        num_to_sample = max(2, min(POS_SAMPLE_LIMIT, seq_len + 1))
        mandatory = {0, seq_len}
        # Available interior positions
        interior = list(range(1, seq_len))
        if len(interior) <= num_to_sample - 2:
            chosen = set(interior)
        else:
            chosen = set(random.sample(interior, num_to_sample - 2))
        chosen.update(mandatory)
        return sorted(chosen)

    def best_insertion_for_txn(current_seq, txn, current_best):
        """
        Try inserting txn into multiple positions in current_seq.
        Returns (best_cost, best_pos, second_best_cost). current_best is kept for potential pruning.
        """
        seq_len = len(current_seq)
        best_cost = float('inf')
        second_best = float('inf')
        best_pos = 0
        positions = sample_positions(seq_len)
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
        if second_best == float('inf'):
            second_best = best_cost
        return best_cost, best_pos, second_best

    def construct_sequence():
        """
        Randomized best-insertion construction.
        """
        # Seed with a random start transaction; quickly choose a good second insertion
        remaining = all_txns.copy()
        random_start = random.randint(0, n - 1)
        seq = [random_start]
        remaining.remove(random_start)
        curr_cost = eval_cost(seq)

        # Optionally add a second txn by testing a small candidate set thoroughly
        if remaining:
            k = min(6, len(remaining))
            candidates = random.sample(remaining, k)
            best_pair_cost = float('inf')
            best_txn = None
            best_pos = 1  # only positions 0 or 1 possible
            for t in candidates:
                # Try both positions in [0,1]
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

        # Build the rest using sampled candidates and best insertion positions
        while remaining:
            # Adaptive candidate sample size
            # Focus more candidates when many remain; taper off later
            dynamic_base = CAND_SAMPLE_BASE
            # small randomness to diversify
            cand_size = min(len(remaining), max(3, dynamic_base + random.randint(-JITTER, JITTER)))
            # In the endgame, evaluate all remaining to avoid myopia
            if len(remaining) <= dynamic_base + 2:
                candidates = list(remaining)
            else:
                candidates = random.sample(remaining, cand_size)

            # Score candidates by best insertion, along with regret (second-best - best)
            scored = []
            for t in candidates:
                cost_t, pos_t, second_best = best_insertion_for_txn(seq, t, float('inf'))
                regret = max(0.0, second_best - cost_t)
                scored.append((cost_t, regret, t, pos_t))

            # Choose with regret-biased RCL among top by cost
            if scored:
                scored.sort(key=lambda x: x[0])  # ascending by cost
                rcl_size = max(1, min(3, len(scored)//2 if len(scored) > 3 else len(scored)))
                rcl = scored[:rcl_size]
                # With probability 0.6, pick the highest regret in RCL; else random among RCL
                if random.random() < 0.6:
                    best_step_cost, _, best_txn, best_pos = max(rcl, key=lambda x: x[1])
                else:
                    best_step_cost, _, best_txn, best_pos = random.choice(rcl)

                seq.insert(best_pos, best_txn)
                curr_cost = best_step_cost
                remaining.remove(best_txn)
            else:
                # Fallback: append a random remaining txn
                t = random.choice(remaining)
                seq.append(t)
                curr_cost = eval_cost(seq)
                remaining.remove(t)

        return seq, curr_cost

    def local_search_adjacent_swaps(seq, curr_cost):
        """
        Perform multiple passes of adjacent swap hill-climbing.
        """
        improved = True
        passes = 0
        best_seq = seq
        best_cost = curr_cost

        while improved and passes < MAX_LS_PASSES:
            improved = False
            passes += 1
            i = 0
            # One pass left-to-right
            while i < len(best_seq) - 1:
                cand = best_seq.copy()
                cand[i], cand[i + 1] = cand[i + 1], cand[i]
                c = eval_cost(cand)
                if c < best_cost:
                    best_cost = c
                    best_seq = cand
                    improved = True
                    # After improvement, we can continue locally; do not skip ahead
                i += 1
        return best_seq, best_cost

    def local_search_oropt(seq, curr_cost, max_rounds=1):
        """
        Or-opt block reinsertion (block sizes 1..3), best-improvement per round.
        """
        best_seq = list(seq)
        best_cost = curr_cost
        if len(best_seq) <= 2:
            return best_seq, best_cost

        rounds = 0
        while rounds < max_rounds:
            rounds += 1
            improved = False
            # Try larger blocks first
            for blk in (3, 2, 1):
                if blk > len(best_seq):
                    continue
                move_best_delta = 0.0
                move_best_i = None
                move_best_pos = None
                move_best_block = None
                # Enumerate block start indices
                for i in range(0, len(best_seq) - blk + 1):
                    block = best_seq[i:i + blk]
                    base = best_seq[:i] + best_seq[i + blk:]
                    # Candidate insertion positions (respect ends; sample if long)
                    m = len(base) + 1
                    if POS_SAMPLE_LIMIT is None or m <= (POS_SAMPLE_LIMIT or m) + 1:
                        positions = list(range(m))
                    else:
                        k = min(POS_SAMPLE_LIMIT, m - 1)
                        interior = list(range(1, m - 1))
                        sampled = set(random.sample(interior, k)) if interior else set()
                        sampled.update({0, m - 1})
                        positions = sorted(sampled)
                    for pos in positions:
                        cand = base[:]
                        cand[pos:pos] = block
                        c = eval_cost(cand)
                        if c < best_cost - 1e-12:
                            # Best-improvement tracking
                            delta = best_cost - c
                            if delta > move_best_delta:
                                move_best_delta = delta
                                move_best_i = i
                                move_best_pos = pos
                                move_best_block = block
                if move_best_i is not None:
                    # Apply best move for this block size
                    base = best_seq[:move_best_i] + best_seq[move_best_i + len(move_best_block):]
                    base[move_best_pos:move_best_pos] = move_best_block
                    best_seq = base
                    best_cost = eval_cost(best_seq)
                    improved = True
                    break  # restart at largest block after improvement
            if not improved:
                break

        return best_seq, best_cost

    def local_search_relocations(seq, curr_cost):
        """
        Random relocation: remove at index i and reinsert at j if improves.
        """
        best_seq = seq
        best_cost = curr_cost
        trials = 0
        # Early exit if trivial
        if len(seq) <= 2:
            return best_seq, best_cost

        while trials < RELOC_TRIES:
            i = random.randint(0, len(best_seq) - 1)
            j = random.randint(0, len(best_seq) - 1)
            if i == j:
                trials += 1
                continue
            cand = best_seq.copy()
            t = cand.pop(i)
            # Adjust j if removal shifts indices
            if j > i:
                j -= 1
            cand.insert(j, t)
            c = eval_cost(cand)
            if c < best_cost:
                best_cost = c
                best_seq = cand
                # On improvement, reset trials to keep exploring around new incumbent
                trials = 0
            else:
                trials += 1
        return best_seq, best_cost

    def ruin_and_recreate(seq, curr_cost, rounds=2):
        """
        Ruin-and-recreate LNS: remove k transactions (contiguous + random) and greedily reinsert.
        """
        best_seq = list(seq)
        best_cost = curr_cost
        if len(best_seq) <= 4:
            return best_seq, best_cost

        for _ in range(rounds):
            k = min(max(8, int(0.1 * n)), len(best_seq) - 1)
            # Select a contiguous segment (half of k) and a few random removals
            k_contig = max(2, k // 2)
            start = random.randint(0, len(best_seq) - k_contig)
            remove_idx = set(range(start, start + k_contig))
            # Add random indices
            while len(remove_idx) < k:
                remove_idx.add(random.randint(0, len(best_seq) - 1))
            remove_idx = sorted(remove_idx)

            removed = [best_seq[i] for i in remove_idx]
            base = [best_seq[i] for i in range(len(best_seq)) if i not in remove_idx]

            # Greedy reinsertion of removed items using best insertion positions
            cur = list(base)
            for t in removed:
                ins_cost, ins_pos, _ = best_insertion_for_txn(cur, t, float('inf'))
                cur.insert(ins_pos, t)
            c_new = eval_cost(cur)
            if c_new < best_cost:
                best_cost = c_new
                best_seq = cur

        return best_seq, best_cost

    # Multi-start GRASP
    global_best_seq = None
    global_best_cost = float('inf')

    for _ in range(max(1, num_seqs)):
        seq, cost = construct_sequence()
        # Local search refinement: Or-opt blocks, adjacent swaps, then relocations
        seq, cost = local_search_oropt(seq, cost, max_rounds=1)
        seq, cost = local_search_adjacent_swaps(seq, cost)
        seq, cost = local_search_relocations(seq, cost)
        # Ruin-and-recreate with quick polishing
        seq2, cost2 = ruin_and_recreate(seq, cost, rounds=2)
        if cost2 < cost:
            seq, cost = seq2, cost2
            seq, cost = local_search_adjacent_swaps(seq, cost)
            seq, cost = local_search_oropt(seq, cost, max_rounds=1)

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