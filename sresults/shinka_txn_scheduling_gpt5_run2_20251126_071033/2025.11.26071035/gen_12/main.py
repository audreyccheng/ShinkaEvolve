# EVOLVE-BLOCK-START
"""Transaction scheduling algorithm for optimizing makespan across multiple workloads
Conflict-regret LNS: cached lookahead construction + VND + ruin-and-recreate with regret insertion.
"""

import time
import random
import sys
import os

# Add the openevolve_examples directory to the path to import txn_simulator and workloads
# Find the repository root by looking for openevolve_examples directory
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
    Conflict-regret LNS scheduler: multi-start lookahead construction,
    followed by VND and ruin-and-recreate with regret-based reinsertion.

    Args:
        workload: Workload object containing transaction data
        num_seqs: Search effort parameter (controls restarts and LNS iterations)

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    n = workload.num_txns
    all_txns = list(range(n))

    # Adaptive parameters
    small = n <= 50
    # Construction parameters
    LOOKAHEAD_ALPHA = 0.7
    CAND_TXN_K = 10 if small else 8
    CAND_POS_LIMIT = None if small else 16  # positions to sample per insertion during construction
    SECOND_STEP_K = 6 if small else 5

    # Local search parameters
    ADJ_PASSES = 2 if small else 2
    TWO_OPT_BUDGET = n * (3 if small else 2)  # sampled non-adjacent swaps
    REINSERT_PASSES = 2 if small else 2
    REINSERT_POS_LIMIT = None if small else 24

    # LNS parameters
    BASE_RUIN_FRAC = 0.12 if small else 0.15
    EXTRA_RUIN = 0.06 if small else 0.05
    RCL_K = 3  # small RCL during construction for diversification
    LNS_ITERS = max(5, min(14, (num_seqs or 1) + (2 if not small else 1)))

    # Restarts (multi-start constructions)
    RESTARTS = max(2, (num_seqs or 1))

    # Global evaluation cache to reduce repeated simulator calls
    cost_cache = {}

    def eval_cost(seq):
        """Evaluate full or partial sequence with caching."""
        key = tuple(seq)
        c = cost_cache.get(key)
        if c is not None:
            return c
        c = workload.get_opt_seq_cost(list(seq))
        cost_cache[key] = c
        return c

    def sample_positions(seq_len, limit=None):
        """Positions [0..seq_len], optionally sampled (always keep ends)."""
        if limit is None or seq_len + 1 <= limit:
            return list(range(seq_len + 1))
        m = seq_len + 1
        k = max(2, min(limit, m))
        mandatory = {0, seq_len}
        interior = list(range(1, seq_len))
        if not interior:
            return [0, seq_len]
        choose = set(interior if len(interior) <= k - 2 else random.sample(interior, k - 2))
        choose.update(mandatory)
        return sorted(choose)

    def best_insertion_cost(seq, txn, pos_limit=None):
        """Return (min_cost, best_pos) by inserting txn into seq with sampled positions."""
        positions = sample_positions(len(seq), pos_limit)
        best_c = float('inf')
        best_p = 0
        for p in positions:
            cand = seq[:]
            cand.insert(p, txn)
            c = eval_cost(cand)
            if c < best_c:
                best_c = c
                best_p = p
        return best_c, best_p

    def construct_lookahead(seed=None):
        """
        Lookahead construction:
        - Choose candidates among remaining txns.
        - For each candidate, evaluate best insertion position and a shallow second-step lookahead.
        - Use RCL among top candidates to diversify slightly.
        """
        remaining = all_txns[:]
        if seed is None:
            start = random.randint(0, n - 1)
        else:
            start = seed
        seq = [start]
        remaining.remove(start)

        # Optionally pick a solid 2nd txn
        if remaining:
            k = min(6, len(remaining))
            cands = random.sample(remaining, k)
            best_pair = (float('inf'), None, 1)
            for t in cands:
                for p in [0, 1]:
                    cand = seq[:]
                    cand.insert(p, t)
                    c = eval_cost(cand)
                    if c < best_pair[0]:
                        best_pair = (c, t, p)
            if best_pair[1] is not None:
                seq.insert(best_pair[2], best_pair[1])
                remaining.remove(best_pair[1])

        while remaining:
            k_txn = min(CAND_TXN_K, len(remaining))
            cand_txns = random.sample(remaining, k_txn)
            scored = []
            for t in cand_txns:
                c1, pos = best_insertion_cost(seq, t, CAND_POS_LIMIT)
                # two-step lookahead over a sample of remaining-after
                rem_after = [x for x in remaining if x != t]
                if rem_after:
                    k2 = min(SECOND_STEP_K, len(rem_after))
                    second = random.sample(rem_after, k2)
                    best_c2 = float('inf')
                    for u in second:
                        # Insert u optimally after placing t
                        cand_seq = seq[:]
                        cand_seq.insert(pos, t)
                        c2, _ = best_insertion_cost(cand_seq, u, CAND_POS_LIMIT)
                        if c2 < best_c2:
                            best_c2 = c2
                    score = LOOKAHEAD_ALPHA * c1 + (1.0 - LOOKAHEAD_ALPHA) * best_c2
                else:
                    score = c1
                scored.append((score, c1, t, pos))
            scored.sort(key=lambda x: x[0])
            rcl_size = min(RCL_K, len(scored))
            choice = scored[random.randint(0, rcl_size - 1)]
            _, c1, t_sel, pos_sel = choice
            seq.insert(pos_sel, t_sel)
            remaining.remove(t_sel)
        return seq, eval_cost(seq))

    def construct_rcl_best_insertion():
        """Simpler GRASP/RCL construction without lookahead for diversification."""
        remaining = all_txns[:]
        start = random.randint(0, n - 1)
        seq = [start]
        remaining.remove(start)
        while remaining:
            k_txn = min(CAND_TXN_K, len(remaining))
            cand_txns = random.sample(remaining, k_txn)
            options = []
            for t in cand_txns:
                c, p = best_insertion_cost(seq, t, CAND_POS_LIMIT)
                options.append((c, t, p))
            options.sort(key=lambda x: x[0])
            r = min(RCL_K, len(options))
            pick = options[random.randint(0, r - 1)]
            c, t, p = pick
            seq.insert(p, t)
            remaining.remove(t)
        return seq, eval_cost(seq))

    def local_adjacent_swaps(seq, cost, passes=ADJ_PASSES):
        best_seq = list(seq)
        best_cost = cost
        for _ in range(passes):
            improved = False
            for i in range(len(best_seq) - 1):
                cand = best_seq[:]
                cand[i], cand[i + 1] = cand[i + 1], cand[i]
                c = eval_cost(cand)
                if c < best_cost:
                    best_cost = c
                    best_seq = cand
                    improved = True
            if not improved:
                break
        return best_seq, best_cost

    def local_two_opt(seq, cost, budget=TWO_OPT_BUDGET):
        """Sampled non-adjacent swaps (2-opt-like) to escape local minima."""
        best_seq = list(seq)
        best_cost = cost
        n_local = len(best_seq)
        tries = 0
        while tries < budget:
            i = random.randint(0, n_local - 1)
            j = random.randint(0, n_local - 1)
            if abs(i - j) <= 1:
                tries += 1
                continue
            if i > j:
                i, j = j, i
            cand = best_seq[:]
            cand[i], cand[j] = cand[j], cand[i]
            c = eval_cost(cand)
            if c < best_cost:
                best_cost = c
                best_seq = cand
                # allow further improvements around new incumbent
                tries = 0
            else:
                tries += 1
        return best_seq, best_cost

    def local_reinsert(seq, cost, passes=REINSERT_PASSES, pos_limit=REINSERT_POS_LIMIT):
        """1-move relocation local search with first improvement per pass."""
        best_seq = list(seq)
        best_cost = cost
        for _ in range(passes):
            improved = False
            indices = list(range(len(best_seq)))
            random.shuffle(indices)
            for i in indices:
                base = best_seq[:]
                t = base.pop(i)
                positions = sample_positions(len(base), pos_limit)
                # Try best position for this t
                best_local = None
                best_local_cost = float('inf')
                for p in positions:
                    cand = base[:]
                    cand.insert(p, t)
                    c = eval_cost(cand)
                    if c < best_local_cost:
                        best_local_cost = c
                        best_local = (p, cand)
                if best_local_cost + 1e-9 < best_cost:
                    best_cost = best_local_cost
                    best_seq = best_local[1]
                    improved = True
                    break
            if not improved:
                break
        return best_seq, best_cost

    def ruin_and_recreate(seq, cost, iter_idx):
        """Remove a fraction of transactions and reinsert via regret-2 heuristic."""
        n_local = len(seq)
        frac = BASE_RUIN_FRAC + (EXTRA_RUIN if (iter_idx % 3 == 2) else 0.0)
        k_remove = max(2, int(n_local * frac))
        # Select unique indices to remove
        idxs = sorted(random.sample(range(n_local), k_remove))
        removed = [seq[i] for i in idxs]
        # Build base by deleting from end to keep indices stable
        base = seq[:]
        for i in reversed(idxs):
            base.pop(i)
        # Regret-2 insertion loop
        pool = set(removed)
        while pool:
            candidates = []
            for t in list(pool):
                positions = sample_positions(len(base), REINSERT_POS_LIMIT)
                scores = []
                for p in positions:
                    cand = base[:]
                    cand.insert(p, t)
                    c = eval_cost(cand)
                    scores.append((c, p))
                scores.sort(key=lambda x: x[0])
                best_c, best_p = scores[0]
                second_c = scores[1][0] if len(scores) > 1 else best_c
                regret = second_c - best_c
                candidates.append((regret, best_c, best_p, t))
            # Choose the txn with max regret (most critical placement)
            candidates.sort(key=lambda x: (-x[0], x[1]))
            _, best_c, best_p, t_sel = candidates[0]
            base.insert(best_p, t_sel)
            pool.remove(t_sel)
        final_seq = base
        final_cost = eval_cost(final_seq)
        return final_seq, final_cost

    def VND(seq, cost):
        """Variable Neighborhood Descent: reinsert -> two-opt -> adjacent swaps."""
        s, c = local_reinsert(seq, cost)
        s, c = local_two_opt(s, c)
        s, c = local_adjacent_swaps(s, c)
        return s, c

    # Multi-start construction + LNS
    best_seq = None
    best_cost = float('inf')

    for r in range(RESTARTS):
        # Diversify seed choice
        if r % 3 == 0:
            seq, c = construct_lookahead()
        elif r % 3 == 1:
            # Seed with a good starter (min prefix cost among sample)
            sample = random.sample(all_txns, min(10, n))
            start = min(sample, key=lambda t: eval_cost([t]))
            seq, c = construct_lookahead(seed=start)
        else:
            seq, c = construct_rcl_best_insertion()

        # Initial local improvement
        seq, c = VND(seq, c)

        # Iterated LNS with regret reinsertion
        for it in range(LNS_ITERS):
            new_seq, new_cost = ruin_and_recreate(seq, c, it)
            new_seq, new_cost = VND(new_seq, new_cost)
            if new_cost + 1e-9 < c:
                seq, c = new_seq, new_cost
            else:
                # small perturbation to escape plateaus: random 2-swap
                i, j = random.sample(range(n), 2)
                pert = seq[:]
                if i > j: i, j = j, i
                pert[i], pert[j] = pert[j], pert[i]
                pc = eval_cost(pert)
                if pc < c:
                    seq, c = pert, pc

        if c < best_cost:
            best_cost, best_seq = c, seq

    # Safety checks
    assert best_seq is not None
    assert len(best_seq) == n and len(set(best_seq)) == n

    return best_cost, best_seq


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
