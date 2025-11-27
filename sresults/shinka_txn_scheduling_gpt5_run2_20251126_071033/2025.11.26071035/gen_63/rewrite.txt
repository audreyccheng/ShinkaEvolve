# EVOLVE-BLOCK-START
"""Transaction scheduling algorithm for optimizing makespan across multiple workloads
Regret-diverse GRASP + insertion-based beam lookahead + VND (Or-opt, swaps) + regret LNS.
"""

import time
from functools import lru_cache
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
    Hybrid GRASP + insertion-beam + VND + regret-guided LNS to minimize makespan.

    Args:
        workload: Workload object containing transaction data
        num_seqs: Search effort parameter (drives seeds, beam width, and refinement)

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    n = workload.num_txns
    txns_all = list(range(n))

    # -------- Adaptive parameters --------
    small = n <= 50
    med = 50 < n <= 90
    large = n > 90

    # Construction / GRASP
    STARTER_SAMPLE = min(10, n)
    CAND_SAMPLE_BASE = 12 if small else (10 if med else 9)
    RCL_K = 3
    JITTER = 2
    RCL_COST_PROB = 0.6  # prob to bias cost over regret inside RCL

    # Position sampling for insertion evaluations
    POS_SAMPLE_CAP = None if small else (22 if med else 18)  # evaluate all when small

    # Beam search (insertion-based)
    beam_width = max(6, min(12, (num_seqs // 2) + (3 if small else 0)))
    cand_per_state = min(28, max(12, n // (4 if large else 3)))
    lookahead_k = 6 if med or large else 8
    diversity_quota = max(1, int(0.3 * beam_width))  # add regret-diverse options
    alpha_blend = 0.7  # score = alpha*c1 + (1-alpha)*best_c2

    # Local search (VND)
    VND_MAX_ROUNDS = 2 if large else 3
    LS_ADJ_PASSES = 2 if large else 3
    PAIR_SWAP_SAMPLES = min(200, max(80, n))
    SEGMENT_REV_SAMPLES = min(200, max(80, n))

    # LNS
    LNS_ROUNDS = 2 if med or large else 3
    LNS_REMOVE_FRAC = 0.08 if large else (0.10 if med else 0.12)
    LNS_REMOVE_MIN = 8 if med or large else 6
    LNS_REMOVE_MAX = 18 if med or large else 14

    # Iterated improvements
    ILS_ITERS = max(2, min(4, num_seqs))

    # Seed RNG lightly for diversity per call
    random.seed((n * 1103 + num_seqs * 911 + 7) % (2**32 - 1))

    # -------- Cached evaluator --------
    @lru_cache(maxsize=250_000)
    def _eval_cost_tuple(seq_tuple):
        return workload.get_opt_seq_cost(list(seq_tuple))

    def eval_cost(prefix):
        return _eval_cost_tuple(tuple(prefix))

    # -------- Episode context: deterministic position sampling + best-two cache --------
    class EpisodeContext:
        def __init__(self):
            # key = (tuple(seq), txn, pos_sig) -> (best_cost, best_pos, second_best_cost)
            self.best_two_cache = {}

        def positions_signature(self, seq_len, exhaustive=False):
            total = seq_len + 1
            if exhaustive or POS_SAMPLE_CAP is None or total <= POS_SAMPLE_CAP:
                return ('all', seq_len)
            anchors = {0, seq_len, seq_len // 2, seq_len // 4, (3 * seq_len) // 4}
            anchors = {p for p in anchors if 0 <= p <= seq_len}
            # fill evenly to POS_SAMPLE_CAP deterministically
            need = max(0, POS_SAMPLE_CAP - len(anchors))
            if need > 0 and seq_len > 1:
                for i in range(1, need + 1):
                    pos = round(i * seq_len / (need + 1))
                    anchors.add(max(0, min(seq_len, pos)))
            return tuple(sorted(anchors))

        def positions_for_insertion(self, seq_len, exhaustive=False):
            sig = self.positions_signature(seq_len, exhaustive=exhaustive)
            if sig == ('all', seq_len):
                return list(range(seq_len + 1)), sig
            return list(sig), sig

        def best_two_insertions(self, seq, txn, exhaustive=False):
            positions, sig = self.positions_for_insertion(len(seq), exhaustive=exhaustive)
            key = (tuple(seq), txn, sig)
            hit = self.best_two_cache.get(key)
            if hit is not None:
                return hit
            best_c, best_p = float('inf'), 0
            second_c = float('inf')
            for p in positions:
                cand = seq[:]
                cand.insert(p, txn)
                c = eval_cost(cand)
                if c < best_c:
                    second_c = best_c
                    best_c, best_p = c, p
                elif c < second_c:
                    second_c = c
            if second_c == float('inf'):
                second_c = best_c
            res = (best_c, best_p, second_c)
            self.best_two_cache[key] = res
            # If exhaustive evaluation was used, also store under 'all' signature
            if sig == ('all', len(seq)):
                self.best_two_cache[(tuple(seq), txn, ('all', len(seq)))] = res
            return res

    ctx = EpisodeContext()

    # -------- GRASP constructor (regret-guided) --------
    def select_best_starter():
        cands = random.sample(txns_all, STARTER_SAMPLE) if STARTER_SAMPLE < n else txns_all
        best_t, best_c = None, float('inf')
        for t in cands:
            c = eval_cost([t])
            if c < best_c:
                best_c, best_t = c, t
        return best_t if best_t is not None else random.randint(0, n - 1)

    def construct_regret_insertion():
        remaining = set(txns_all)
        seq = [select_best_starter()]
        remaining.remove(seq[0])

        # Strong second: test both positions for a few candidates
        if remaining:
            k2 = min(8, len(remaining))
            pairs = []
            for t in random.sample(list(remaining), k2):
                for pos in [0, 1]:
                    cand = seq[:]
                    cand.insert(pos, t)
                    c = eval_cost(cand)
                    pairs.append((c, t, pos))
            if pairs:
                pairs.sort(key=lambda x: x[0])
                choice = random.choice(pairs[:min(3, len(pairs))])
                _, t2, p2 = choice
                seq.insert(p2, t2)
                remaining.remove(t2)

        while remaining:
            # Candidate transaction subset
            size = len(remaining)
            cand_size = size if size <= max(8, 2 * CAND_SAMPLE_BASE) else min(size, max(4, CAND_SAMPLE_BASE + random.randint(-JITTER, JITTER)))
            cand_txns = random.sample(list(remaining), cand_size)

            scored = []
            exhaustive_ins = (len(seq) <= 20) or (len(remaining) <= 2 * beam_width)
            for t in cand_txns:
                c1, p1, c2 = ctx.best_two_insertions(seq, t, exhaustive=exhaustive_ins)
                regret = max(0.0, c2 - c1)
                scored.append((c1, regret, t, p1))
            if not scored:
                t = random.choice(list(remaining))
                seq.append(t)
                remaining.remove(t)
                continue

            # Build RCL from both best cost and high regret
            scored.sort(key=lambda x: x[0])
            rcl_cost = scored[:min(RCL_K, len(scored))]
            scored.sort(key=lambda x: (-x[1], x[0]))
            rcl_regret = scored[:min(RCL_K, len(scored))]
            pool = list({(c, r, t, p) for (c, r, t, p) in (rcl_cost + rcl_regret)})

            # Choose within pool with bias
            if random.random() < RCL_COST_PROB:
                pool.sort(key=lambda x: x[0])
            else:
                pool.sort(key=lambda x: (-x[1], x[0]))
            chosen_c, _, t_pick, pos_pick = random.choice(pool[:min(3, len(pool))])
            seq.insert(pos_pick, t_pick)
            remaining.remove(t_pick)

        return eval_cost(seq), seq

    # -------- Insertion-based beam search with lookahead and suffix diversity --------
    def beam_search():
        # Initialize with top singletons and one GRASP seed
        starters = [(eval_cost([t]), [t]) for t in txns_all]
        starters.sort(key=lambda x: x[0])
        init = starters[:min(len(starters), max(beam_width * 2, beam_width + 2))]
        c0, s0 = construct_regret_insertion()
        init.append((c0, s0))

        beam = []
        seen = set()
        for c, seq in init:
            key = tuple(seq)
            if key in seen:
                continue
            seen.add(key)
            rem = frozenset(t for t in txns_all if t not in seq)
            beam.append((c, seq, rem))
            if len(beam) >= beam_width:
                break

        best_complete = (float('inf'), [])

        for _depth in range(1, n + 1):
            if not beam:
                break
            next_pool = []
            layer_seen = set()
            suffix_seen = set()  # suffix-2 diversity signature
            for c_so_far, prefix, rem in beam:
                if not rem:
                    if c_so_far < best_complete[0]:
                        best_complete = (c_so_far, prefix)
                    continue
                rem_list = list(rem)
                expand_list = rem_list if len(rem_list) <= cand_per_state else random.sample(rem_list, cand_per_state)

                scored = []
                for t in expand_list:
                    # insert t at its best position into current prefix
                    c1, p1, c1_second = ctx.best_two_insertions(prefix, t, exhaustive=(len(prefix) <= 20))
                    new_prefix = prefix[:]
                    new_prefix.insert(p1, t)
                    # lookahead one more insertion
                    rem_after = [x for x in rem_list if x != t]
                    best_c2 = c1
                    disp = 0.0
                    if rem_after:
                        k2 = min(lookahead_k, len(rem_after))
                        second = random.sample(rem_after, k2)
                        costs2 = []
                        for u in second:
                            cu, pu, _ = ctx.best_two_insertions(new_prefix, u, exhaustive=False)
                            costs2.append(cu)
                            if cu < best_c2:
                                best_c2 = cu
                        if len(costs2) >= 2:
                            disp = max(costs2) - min(costs2)
                        else:
                            disp = max(0.0, best_c2 - c1)
                    score = alpha_blend * c1 + (1.0 - alpha_blend) * best_c2
                    # primary regret from best-two at current step
                    r_primary = max(0.0, c1_second - c1)
                    total_regret = 0.5 * r_primary + 0.5 * disp
                    scored.append((score, c1, total_regret, t, p1, frozenset(x for x in rem if x != t), new_prefix))

                # pick expansions by score and regret diversity
                scored.sort(key=lambda x: x[0])
                top_cost = scored[:max(1, min(beam_width, len(scored)))]
                scored_by_regret = sorted(scored, key=lambda x: (-x[2], x[0]))
                top_regret = scored_by_regret[:min(diversity_quota, len(scored_by_regret))]
                for sc, c1, rg, t, pos, new_rem, new_seq in (top_cost + top_regret):
                    key = tuple(new_seq)
                    if key in layer_seen:
                        continue
                    sig = (new_seq[-2], new_seq[-1]) if len(new_seq) >= 2 else (None, new_seq[-1])
                    if sig in suffix_seen:
                        continue
                    layer_seen.add(key)
                    suffix_seen.add(sig)
                    next_pool.append((c1, new_seq, new_rem))

            if not next_pool:
                break
            next_pool.sort(key=lambda x: x[0])
            pruned, seen_prefixes = [], set()
            for c1, seq, rem in next_pool:
                k = tuple(seq)
                if k in seen_prefixes:
                    continue
                seen_prefixes.add(k)
                pruned.append((c1, seq, rem))
                if len(pruned) >= beam_width:
                    break
            beam = pruned

        for c, seq, rem in beam:
            if not rem and c < best_complete[0]:
                best_complete = (c, seq)

        if best_complete[1] and len(best_complete[1]) == n:
            return best_complete

        # Greedy completion from best partial by best-insertion
        if beam:
            c, seq, rem = min(beam, key=lambda x: x[0])
            cur = list(seq)
            rem_list = list(rem)
            while rem_list:
                best_t, best_pos, best_c = None, 0, float('inf')
                for t in rem_list:
                    c2, p2, _ = ctx.best_two_insertions(cur, t, exhaustive=False)
                    if c2 < best_c:
                        best_c, best_t, best_pos = c2, t, p2
                cur.insert(best_pos, best_t)
                rem_list.remove(best_t)
            return eval_cost(cur), cur

        # Fallback: identity
        ident = list(range(n))
        return eval_cost(ident), ident

    # -------- Local Search (VND) --------
    def vnd_local_search(seq, start_cost, max_rounds=VND_MAX_ROUNDS):
        best_seq = list(seq)
        best_cost = start_cost

        def or_opt_pass(k):
            nonlocal best_seq, best_cost
            L = len(best_seq)
            if L <= k:
                return False
            improved = False
            i = 0
            while i <= len(best_seq) - k:
                block = best_seq[i:i + k]
                base = best_seq[:i] + best_seq[i + k:]
                positions, _ = ctx.positions_for_insertion(len(base), exhaustive=(len(base) <= 20))
                move_best_c = best_cost
                move_best_p = None
                for p in positions:
                    # avoid equivalent reinsertion at original place for k==1
                    if k == 1 and p == i:
                        continue
                    cand = base[:]
                    cand[p:p] = block
                    c = eval_cost(cand)
                    if c < move_best_c:
                        move_best_c, move_best_p = c, p
                if move_best_p is not None and move_best_c + 1e-12 < best_cost:
                    new_seq = base[:]
                    new_seq[move_best_p:move_best_p] = block
                    best_seq, best_cost = new_seq, move_best_c
                    improved = True
                    i = 0  # restart scan
                else:
                    i += 1
            return improved

        def adjacent_swap_pass(max_passes=LS_ADJ_PASSES):
            nonlocal best_seq, best_cost
            improved_any = False
            for _ in range(max_passes):
                improved = False
                for i in range(len(best_seq) - 1):
                    cand = best_seq[:]
                    cand[i], cand[i + 1] = cand[i + 1], cand[i]
                    c = eval_cost(cand)
                    if c < best_cost:
                        best_seq, best_cost = cand, c
                        improved = True
                improved_any = improved_any or improved
                if not improved:
                    break
            return improved_any

        def sampled_pair_swaps(tries=PAIR_SWAP_SAMPLES):
            nonlocal best_seq, best_cost
            L = len(best_seq)
            if L <= 3:
                return False
            improved = False
            best_delta = 0.0
            best_move = None
            attempts = tries
            while attempts > 0:
                i = random.randint(0, L - 1)
                j = random.randint(0, L - 1)
                if i == j or abs(i - j) <= 1:
                    attempts -= 1
                    continue
                cand = best_seq[:]
                cand[i], cand[j] = cand[j], cand[i]
                c = eval_cost(cand)
                delta = best_cost - c
                if delta > best_delta:
                    best_delta = delta
                    best_move = (i, j, c, cand)
                attempts -= 1
            if best_move is not None:
                _, _, c, cand = best_move
                best_seq, best_cost = cand, c
                improved = True
            return improved

        def sampled_segment_reversal(tries=SEGMENT_REV_SAMPLES):
            nonlocal best_seq, best_cost
            L = len(best_seq)
            if L <= 5:
                return False
            improved = False
            best_delta = 0.0
            best_move = None
            attempts = tries
            while attempts > 0:
                i = random.randint(0, L - 3)
                j = random.randint(i + 2, min(L - 1, i + 12))
                cand = best_seq[:]
                cand[i:j + 1] = reversed(cand[i:j + 1])
                c = eval_cost(cand)
                delta = best_cost - c
                if delta > best_delta:
                    best_delta = delta
                    best_move = (i, j, c, cand)
                attempts -= 1
            if best_move is not None:
                _, _, c, cand = best_move
                best_seq, best_cost = cand, c
                improved = True
            return improved

        rounds = 0
        while rounds < max_rounds:
            rounds += 1
            any_improved = False
            for k in (3, 2, 1):
                if or_opt_pass(k):
                    any_improved = True
            if adjacent_swap_pass():
                any_improved = True
            if sampled_pair_swaps():
                any_improved = True
            if sampled_segment_reversal():
                any_improved = True
            if not any_improved:
                break
        return best_cost, best_seq

    # -------- LNS: Ruin-and-Recreate with regret-guided reinsertion --------
    def lns_ruin_recreate(seq, start_cost, rounds=LNS_ROUNDS):
        best_seq = list(seq)
        best_cost = start_cost
        for _ in range(max(1, rounds)):
            L = len(best_seq)
            if L <= 6:
                break
            # choose a block to remove
            base_len = max(4, L // 10)
            block_len = min(L - 2, base_len + random.randint(0, 3))
            start = random.randint(0, L - block_len)
            removed = best_seq[start:start + block_len]
            skeleton = best_seq[:start] + best_seq[start + block_len:]
            # optionally remove a few extras
            extra_count = min(3, max(0, L // 30))
            if extra_count and len(skeleton) > extra_count:
                extras_idx = sorted(random.sample(range(len(skeleton)), extra_count))
                offset = 0
                extras = []
                for idx in extras_idx:
                    idx_adj = idx - offset
                    extras.append(skeleton.pop(idx_adj))
                    offset += 1
                removed += extras

            # reinsert removed using regret-guided best-two insertions
            rebuilt = list(skeleton)
            remain = list(removed)
            while remain:
                # sample some to speed up
                k_t = min(len(remain), max(4, CAND_SAMPLE_BASE // 2))
                cand_txns = remain if len(remain) <= k_t else random.sample(remain, k_t)
                scored = []
                for t in cand_txns:
                    c1, p1, c2 = ctx.best_two_insertions(rebuilt, t, exhaustive=(len(rebuilt) <= 20 or len(remain) <= beam_width))
                    scored.append((c1, c2, t, p1))
                scored.sort(key=lambda x: x[0])
                # pick among top few by highest regret then best cost
                top = scored[:min(3, len(scored))]
                top.sort(key=lambda x: (-(x[1] - x[0]), x[0]))
                c1, c2, t_pick, pos_pick = top[0]
                rebuilt.insert(pos_pick, t_pick)
                remain.remove(t_pick)

            c_new = eval_cost(rebuilt)
            if c_new < best_cost:
                best_cost, best_seq = c_new, rebuilt
        return best_cost, best_seq

    # -------- Final Or-opt(1) polish (exhaustive) --------
    def full_oropt1_polish(seq, start_cost, passes=1):
        best_seq = list(seq)
        best_cost = start_cost
        L = len(best_seq)
        if L <= 2:
            return best_cost, best_seq
        for _ in range(max(1, passes)):
            improved = False
            move_best = (best_cost, None)  # (cost, new_seq)
            for i in range(L):
                t = best_seq[i]
                base = best_seq[:i] + best_seq[i + 1:]
                for p in range(len(base) + 1):
                    if p == i:
                        continue
                    cand = base[:]
                    cand.insert(p, t)
                    c = eval_cost(cand)
                    if c < move_best[0]:
                        move_best = (c, cand)
            if move_best[1] is not None and move_best[0] + 1e-12 < best_cost:
                best_cost, best_seq = move_best
                L = len(best_seq)
                improved = True
            if not improved:
                break
        return best_cost, best_seq

    # -------- Search orchestration --------
    best_cost = float('inf')
    best_seq = list(range(n))

    # 1) Multi-start GRASP seeds + quick VND polish
    seeds = max(3, min(6, num_seqs))
    for _ in range(seeds):
        c, s = construct_regret_insertion()
        c, s = vnd_local_search(s, c, max_rounds=1)
        if c < best_cost:
            best_cost, best_seq = c, s

    # 2) One or two insertion-based beam runs for diversification
    beam_runs = max(1, num_seqs // 3)
    for _ in range(beam_runs):
        c, s = beam_search()
        c, s = vnd_local_search(s, c, max_rounds=1)
        if c < best_cost:
            best_cost, best_seq = c, s

    # 3) Iterated LNS + VND cycles to escape local minima
    incumbent_cost, incumbent_seq = best_cost, best_seq
    for _ in range(ILS_ITERS):
        c1, s1 = lns_ruin_recreate(incumbent_seq, incumbent_cost, rounds=LNS_ROUNDS)
        c2, s2 = vnd_local_search(s1, c1, max_rounds=VND_MAX_ROUNDS)
        if c2 < incumbent_cost:
            incumbent_cost, incumbent_seq = c2, s2
        # quick adjacent pass
        c3, s3 = vnd_local_search(incumbent_seq, incumbent_cost, max_rounds=1)
        if c3 < incumbent_cost:
            incumbent_cost, incumbent_seq = c3, s3

    best_cost, best_seq = incumbent_cost, incumbent_seq
    # final exhaustive 1-block reinsertion polish
    best_cost, best_seq = full_oropt1_polish(best_seq, best_cost, passes=1)

    # Safety checks
    assert len(best_seq) == n and len(set(best_seq)) == n, "Schedule must include each transaction exactly once"

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