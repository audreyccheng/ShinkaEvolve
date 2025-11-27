# EVOLVE-BLOCK-START
"""Transaction scheduling algorithm for optimizing makespan across multiple workloads
Enhanced with GRASP-style randomized best-insertion and local search refinement.
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

    # Scale relocation tries mildly with search effort to deepen local search
    RELOC_TRIES = int(RELOC_TRIES * max(1.0, min(2.0, 0.5 + 0.1 * max(1, num_seqs))))

    JITTER = 2  # small random variation in candidate set size

    # Memoized evaluation for partial prefixes to avoid redundant simulator calls
    cost_cache = {}
    def eval_cost(seq):
        key = tuple(seq)
        if key in cost_cache:
            return cost_cache[key]
        c = workload.get_opt_seq_cost(list(seq))
        cost_cache[key] = c
        return c

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

    def best_insertion_for_txn(current_seq, txn):
        """
        Try inserting txn into multiple positions in current_seq.
        Returns (best_cost, best_pos, regret), where regret = second_best_cost - best_cost.
        """
        seq_len = len(current_seq)
        best_cost = float('inf')
        second_best = float('inf')
        best_pos = 0
        positions = sample_positions(seq_len)
        for pos in positions:
            # Build candidate sequence with insertion
            cand = current_seq.copy()
            cand.insert(pos, txn)
            cost = eval_cost(cand)
            if cost < best_cost:
                second_best = best_cost
                best_cost = cost
                best_pos = pos
            elif cost < second_best:
                second_best = cost
        regret = (second_best - best_cost) if second_best < float('inf') else 0.0
        return best_cost, best_pos, regret

    def construct_sequence():
        """
        Randomized best-insertion construction with regret-guided RCL.
        """
        remaining = all_txns.copy()

        # Choose a good starting transaction by sampling a few and picking lowest prefix cost
        k_start = min(8, len(remaining)) if len(remaining) > 0 else 0
        if k_start > 0:
            start_cands = random.sample(remaining, k_start)
            best_t = min(start_cands, key=lambda t: eval_cost([t]))
            seq = [best_t]
            remaining.remove(best_t)
        else:
            seq = []

        curr_cost = eval_cost(seq) if seq else 0.0

        # Optionally add a second txn by testing a small candidate set thoroughly with RCL
        if remaining:
            k = min(6, len(remaining))
            candidates = random.sample(remaining, k)
            pair_eval = []
            for t in candidates:
                for pos in [0, 1]:
                    cand = seq.copy()
                    cand.insert(pos, t)
                    cost = eval_cost(cand)
                    pair_eval.append((cost, t, pos))
            if pair_eval:
                pair_eval.sort(key=lambda x: x[0])
                rcl_size = min(3, len(pair_eval))
                chosen_cost, chosen_txn, chosen_pos = random.choice(pair_eval[:rcl_size])
                seq.insert(chosen_pos, chosen_txn)
                remaining.remove(chosen_txn)
                curr_cost = chosen_cost

        # Build the rest using sampled candidates and best insertion positions + regret-based RCL
        while remaining:
            # Adaptive candidate sample size with slight randomness and endgame widening
            dynamic_base = CAND_SAMPLE_BASE
            if len(remaining) <= max(8, 2 * dynamic_base):
                candidates = list(remaining)
            else:
                cand_size = min(len(remaining), max(3, dynamic_base + random.randint(-JITTER, JITTER)))
                candidates = random.sample(remaining, cand_size)

            step_evals = []
            for t in candidates:
                cost_t, pos_t, regret_t = best_insertion_for_txn(seq, t)
                step_evals.append((cost_t, t, pos_t, regret_t))

            if not step_evals:
                # Fallback: append a random remaining txn
                t = random.choice(remaining)
                seq.append(t)
                curr_cost = eval_cost(seq)
                remaining.remove(t)
                continue

            # Primary ranking by cost; small chance to pick high-regret from RCL for diversification
            step_evals.sort(key=lambda x: x[0])
            rcl_size = max(1, min(3, len(step_evals) // 2))
            rcl = step_evals[:rcl_size]
            choose_regret = random.random() < 0.3
            if choose_regret:
                rcl_by_regret = sorted(rcl, key=lambda x: x[3], reverse=True)
                chosen_cost, chosen_txn, chosen_pos, _ = rcl_by_regret[0]
            else:
                chosen_cost, chosen_txn, chosen_pos, _ = random.choice(rcl)

            seq.insert(chosen_pos, chosen_txn)
            curr_cost = chosen_cost
            remaining.remove(chosen_txn)

        return seq, curr_cost

    def local_search_adjacent_swaps(seq, curr_cost):
        """
        Perform multiple passes of adjacent swap hill-climbing.
        """
        improved = True
        passes = 0
        best_seq = list(seq)
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
                i += 1
        return best_seq, best_cost

    def local_search_best_reinsert(seq, curr_cost, max_rounds=2):
        """
        Deterministic best-improving reinsertion (Or-opt-1): for each index,
        reinsert txn at the position that yields the best cost if it improves.
        """
        best_seq = list(seq)
        best_cost = curr_cost
        rounds = 0

        if len(best_seq) <= 2:
            return best_seq, best_cost

        while rounds < max_rounds:
            improved = False
            rounds += 1
            idx = 0
            while idx < len(best_seq):
                t = best_seq[idx]
                base = best_seq[:idx] + best_seq[idx + 1:]
                positions = sample_positions(len(base))
                move_best_cost = best_cost
                move_best_pos = None
                for pos in positions:
                    cand = base.copy()
                    cand.insert(pos, t)
                    c = eval_cost(cand)
                    if c < move_best_cost:
                        move_best_cost = c
                        move_best_pos = pos
                if move_best_pos is not None:
                    best_seq = base
                    best_seq.insert(move_best_pos, t)
                    best_cost = move_best_cost
                    improved = True
                    # stay at same idx neighborhood after change
                else:
                    idx += 1
            if not improved:
                break
        return best_seq, best_cost

    def local_search_relocations(seq, curr_cost):
        """
        Random relocation: remove at index i and reinsert at j if improves.
        """
        best_seq = list(seq)
        best_cost = curr_cost
        trials = 0
        # Early exit if trivial
        if len(best_seq) <= 2:
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

    def or_opt_block(seq, curr_cost, block_len, max_rounds=1):
        """
        Best-improving Or-opt move with block length 'block_len'.
        Relocates a contiguous block to another position if it reduces cost.
        """
        best_seq = list(seq)
        best_cost = curr_cost
        if len(best_seq) <= block_len:
            return best_seq, best_cost

        rounds = 0
        while rounds < max_rounds:
            rounds += 1
            improved = False
            move = None  # (i, pos, delta)
            move_best_delta = 0.0
            L = len(best_seq)
            for i in range(0, L - block_len + 1):
                block = best_seq[i:i + block_len]
                base = best_seq[:i] + best_seq[i + block_len:]
                positions = sample_positions(len(base))
                for pos in positions:
                    cand = base[:]
                    cand[pos:pos] = block
                    c = eval_cost(cand)
                    if c < best_cost:
                        delta = best_cost - c
                        if delta > move_best_delta:
                            move_best_delta = delta
                            move = (i, pos)
            if move is not None:
                i, pos = move
                block = best_seq[i:i + block_len]
                base = best_seq[:i] + best_seq[i + block_len:]
                new_seq = base[:]
                new_seq[pos:pos] = block
                best_seq = new_seq
                best_cost = eval_cost(best_seq)
                improved = True
            if not improved:
                break
        return best_seq, best_cost

    def vnd_local_search(seq, curr_cost, rounds=2):
        """
        Variable Neighborhood Descent:
        - Or-opt blocks k=3,2 (best-improving)
        - Or-opt-1 (deterministic best reinsertion)
        - Adjacent swaps
        - Random relocations
        Repeat up to 'rounds' cycles or until no improvement.
        """
        best_seq = list(seq)
        best_cost = curr_cost
        for _ in range(max(1, rounds)):
            changed = False
            for k in (3, 2):
                s, c = or_opt_block(best_seq, best_cost, k, max_rounds=1)
                if c < best_cost:
                    best_seq, best_cost = s, c
                    changed = True
            s, c = local_search_best_reinsert(best_seq, best_cost, max_rounds=1)
            if c < best_cost:
                best_seq, best_cost = s, c
                changed = True
            s, c = local_search_adjacent_swaps(best_seq, best_cost, max_passes=1)
            if c < best_cost:
                best_seq, best_cost = s, c
                changed = True
            s, c = local_search_relocations(best_seq, best_cost)
            if c < best_cost:
                best_seq, best_cost = s, c
                changed = True
            if not changed:
                break
        return best_seq, best_cost

    def ruin_and_recreate(seq, curr_cost, rounds=2):
        """
        Ruin-and-recreate LNS: remove a contiguous block plus a few random items,
        then greedily reinsert by best insertion positions.
        """
        best_seq = list(seq)
        best_cost = curr_cost
        if len(best_seq) <= 6:
            return best_seq, best_cost

        for _ in range(max(1, rounds)):
            L = len(best_seq)
            k_total = min(L - 2, max(6, int(0.1 * L)))
            k_contig = max(2, k_total // 2)
            start = random.randint(0, L - k_contig)
            remove_idx = set(range(start, start + k_contig))
            while len(remove_idx) < k_total:
                remove_idx.add(random.randint(0, L - 1))
            remove_idx = sorted(remove_idx)
            removed = [best_seq[i] for i in remove_idx]
            base = [best_seq[i] for i in range(L) if i not in remove_idx]

            # Greedy reinsertion using cached evals
            rebuilt = list(base)
            for t in removed:
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

            c_new = eval_cost(rebuilt)
            if c_new < best_cost:
                best_cost = c_new
                best_seq = rebuilt

        return best_seq, best_cost

    def path_relink(a_seq, b_seq):
        """
        Transform sequence a toward b, evaluating intermediate schedules and keeping the best.
        Move the required element into position i at each step.
        """
        cur = list(a_seq)
        best_c = eval_cost(cur)
        best_s = list(cur)
        for i in range(len(cur)):
            desired = b_seq[i]
            j = cur.index(desired)
            if j == i:
                continue
            elem = cur.pop(j)
            cur.insert(i, elem)
            c = eval_cost(cur)
            if c < best_c:
                best_c = c
                best_s = list(cur)
        return best_s, best_c

    # Multi-start GRASP with VND + LNS + Path Relinking
    global_best_seq = None
    global_best_cost = float('inf')
    elites = []
    elite_cap = 4

    def add_elite(cost, seq):
        # Keep a small pool of best distinct schedules
        key = tuple(seq)
        for i, (c, s) in enumerate(elites):
            if tuple(s) == key:
                if cost < c:
                    elites[i] = (cost, list(seq))
                break
        else:
            elites.append((cost, list(seq)))
        elites.sort(key=lambda x: x[0])
        if len(elites) > elite_cap:
            del elites[elite_cap:]

    for r in range(max(1, num_seqs)):
        seq, cost = construct_sequence()
        seq, cost = vnd_local_search(seq, cost, rounds=2)
        seq_rr, cost_rr = ruin_and_recreate(seq, cost, rounds=2 if n > 50 else 3)
        if cost_rr < cost:
            seq, cost = vnd_local_search(seq_rr, cost_rr, rounds=2)
        else:
            # quick polish
            seq, cost = local_search_adjacent_swaps(seq, cost, max_passes=1)

        add_elite(cost, seq)
        if cost < global_best_cost:
            global_best_cost = cost
            global_best_seq = seq

    # Path relinking among elites for recombination
    if len(elites) >= 2:
        base_cost, base_seq = elites[0]
        for _, other_seq in elites[1:min(len(elites), 3)]:
            pr_seq, pr_cost = path_relink(base_seq, other_seq)
            pr_seq, pr_cost = vnd_local_search(pr_seq, pr_cost, rounds=2)
            add_elite(pr_cost, pr_seq)
            if pr_cost < global_best_cost:
                global_best_cost = pr_cost
                global_best_seq = pr_seq

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