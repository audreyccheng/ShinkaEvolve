# EVOLVE-BLOCK-START
"""Transaction scheduling algorithm for optimizing makespan across multiple workloads"""

import time
import random
import sys
import os
from collections import OrderedDict

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
    Find a low-makespan schedule using an adaptive regret-beam constructor with
    best-two insertion caching, an endgame branch-and-bound, strong local search,
    ruin-and-repair LNS, and elite path relinking.

    Args:
        workload: Workload object containing transaction data
        num_seqs: Controls search breadth (seeds, beam width, sampling sizes)

    Returns:
        Tuple of (lowest makespan, corresponding schedule)
    """
    n = workload.num_txns
    all_txns = list(range(n))

    # ----------------------------
    # Oracles and caches
    # ----------------------------
    class CostOracle:
        def __init__(self, workload):
            self.workload = workload
            self.cache = {}

        def cost(self, seq):
            key = tuple(seq)
            c = self.cache.get(key)
            if c is not None:
                return c
            c = self.workload.get_opt_seq_cost(seq)
            self.cache[key] = c
            return c

    class LRU:
        def __init__(self, capacity):
            self.capacity = capacity
            self.od = OrderedDict()

        def get(self, key):
            if key in self.od:
                val = self.od.pop(key)
                self.od[key] = val
                return val
            return None

        def set(self, key, val):
            if key in self.od:
                self.od.pop(key)
            elif len(self.od) >= self.capacity:
                self.od.popitem(last=False)
            self.od[key] = val

    class PositionPolicy:
        def __init__(self, k_pos_sample):
            self.k_pos_sample = k_pos_sample

        def all_positions(self, L):
            return list(range(L + 1))

        def sampled_positions(self, L, focus_idx=None, k_override=None, seed=None):
            if L <= 1:
                return [0, L]
            if L <= 12:
                return self.all_positions(L)
            k = k_override if k_override is not None else self.k_pos_sample
            pos_set = {0, L, L // 2, (L * 1) // 4, (L * 3) // 4}
            if focus_idx is not None:
                for d in (-3, -2, -1, 0, 1, 2, 3):
                    p = focus_idx + d
                    if 0 <= p <= L:
                        pos_set.add(p)
            rng = random.Random(seed) if seed is not None else random
            for _ in range(min(k, L + 1)):
                pos_set.add(rng.randint(0, L))
            return sorted(pos_set)

        def policy_sig(self, L, use_all):
            return ('ALL', L) if use_all or L <= 12 else ('SMP', L, self.k_pos_sample)

    class InsertionOracle:
        def __init__(self, cost_oracle, pos_policy, lru_cap=20000):
            self.cost = cost_oracle.cost
            self.pos = pos_policy
            self.cache_best_two = LRU(lru_cap)         # (seq_tuple, t, policy_sig) -> (best_c, best_p, second_c)
            self.cache_block_two = LRU(lru_cap // 2)   # (base_tuple, block_tuple, policy_sig) -> (best_c, best_p, second_c)

        def evaluate_best_two_positions(self, base_seq, t, pos_list):
            best_c, best_p = float('inf'), None
            second_c = float('inf')
            for p in pos_list:
                cand = base_seq[:p] + [t] + base_seq[p:]
                c = self.cost(cand)
                if c < best_c:
                    second_c = best_c
                    best_c, best_p = c, p
                elif c < second_c:
                    second_c = c
            return best_c, best_p, second_c

        def best_two(self, base_seq, t, use_all=False, focus_idx=None, k_override=None):
            L = len(base_seq)
            key = (tuple(base_seq), t, self.pos.policy_sig(L, use_all))
            cached = self.cache_best_two.get(key)
            if cached is not None:
                return cached
            pos_list = self.pos.all_positions(L) if use_all or L <= 12 else \
                self.pos.sampled_positions(L, focus_idx=focus_idx, k_override=k_override,
                                           seed=(hash((tuple(base_seq[-min(10, L):]), t, L)) & 0xffffffff))
            res = self.evaluate_best_two_positions(base_seq, t, pos_list)
            self.cache_best_two.set(key, res)
            return res

        def evaluate_best_two_block(self, base_seq, block, pos_list):
            best_c, best_p = float('inf'), None
            second_c = float('inf')
            for p in pos_list:
                cand = base_seq[:p] + block + base_seq[p:]
                c = self.cost(cand)
                if c < best_c:
                    second_c = best_c
                    best_c, best_p = c, p
                elif c < second_c:
                    second_c = c
            return best_c, best_p, second_c

        def best_two_block(self, base_seq, block, use_all=False, focus_idx=None, k_override=None):
            L = len(base_seq)
            key = (tuple(base_seq), tuple(block), self.pos.policy_sig(L, use_all))
            cached = self.cache_block_two.get(key)
            if cached is not None:
                return cached
            pos_list = self.pos.all_positions(L) if use_all or L <= 12 else \
                self.pos.sampled_positions(L, focus_idx=focus_idx, k_override=k_override,
                                           seed=(hash((tuple(base_seq[-min(10, L):]), tuple(block), L)) & 0xffffffff))
            res = self.evaluate_best_two_block(base_seq, block, pos_list)
            self.cache_block_two.set(key, res)
            return res

    cost_oracle = CostOracle(workload)

    # ----------------------------
    # Hyper-parameters (adaptive)
    # ----------------------------
    elite_size = max(3, min(6, 2 + num_seqs // 3))
    seed_elite_singletons = max(2, min(6, int(num_seqs)))
    seed_random_additional = max(1, min(4, int((num_seqs + 1) // 3)))

    k_txn_sample = min(16, max(8, 2 + int(1.5 * num_seqs)))
    k_pos_sample = min(10, max(6, 2 + int(1.2 * num_seqs)))
    rem_all_threshold = 14

    regret_prob = 0.65

    build_beam_width = max(3, min(6, 2 + num_seqs // 2))
    endgame_all_pos_threshold = max(6, min(12, num_seqs))
    beam_suffix_k = 2

    # Endgame exact enumeration threshold
    endgame_K = max(7, min(9, 4 + num_seqs // 2))

    # Local search parameters
    ls_adj_rounds_max = 2
    two_opt_trials = min(40, max(12, n // 3))
    reinsertion_pos_factor = 1.0

    # ILS / perturbation
    ils_rounds = max(2, min(5, 1 + num_seqs // 4))
    perturb_swap_count = max(2, min(6, 2 + num_seqs // 3))
    perturb_block_len = max(3, min(10, 3 + num_seqs // 2))

    # LNS
    lns_iters = max(2, min(6, 2 + num_seqs // 3))
    destroy_frac_range = (0.08, 0.18)

    pos_policy = PositionPolicy(k_pos_sample)
    ins_oracle = InsertionOracle(cost_oracle, pos_policy, lru_cap=20000)

    # ----------------------------
    # Builders and search modules
    # ----------------------------
    def suffix_sig(seq, k=beam_suffix_k):
        if not seq:
            return (None,)
        return tuple(seq[-min(len(seq), k):])

    class EndgameEnumerator:
        def __init__(self):
            self.memo = {}  # (tuple(seq), frozenset(rem)) -> (best_cost, best_seq)

        def solve(self, seq, rem, global_best=float('inf')):
            key = (tuple(seq), frozenset(rem))
            cached = self.memo.get(key)
            if cached is not None:
                return cached

            base_cost = cost_oracle.cost(seq)
            # Bound: costs are monotone nondecreasing with more txns
            if base_cost >= global_best:
                return (float('inf'), None)

            if not rem:
                self.memo[key] = (base_cost, seq)
                return (base_cost, seq)

            # Order by high regret to prune earlier
            L = len(seq)
            pos_list = list(range(L + 1))
            txn_scores = []
            for t in rem:
                best_c, best_p, second_c = ins_oracle.evaluate_best_two_positions(seq, t, pos_list)
                regret = (second_c - best_c) if second_c < float('inf') else 0.0
                txn_scores.append((regret, t))
            txn_scores.sort(reverse=True)

            best_sol = (float('inf'), None)
            for _, t in txn_scores:
                for p in pos_list:
                    cand = seq[:p] + [t] + seq[p:]
                    c = cost_oracle.cost(cand)
                    # Prune branch that already worse than current best
                    if c >= min(global_best, best_sol[0]):
                        continue
                    sub_rem = set(rem)
                    sub_rem.remove(t)
                    c2, s2 = self.solve(cand, sub_rem, global_best=min(global_best, best_sol[0]))
                    if c2 < best_sol[0]:
                        best_sol = (c2, s2)
            self.memo[key] = best_sol
            return best_sol

    class BeamConstructor:
        def build(self, seed_t):
            seq0 = [seed_t]
            rem0 = set(all_txns)
            rem0.remove(seed_t)
            beam = [(seq0, rem0, cost_oracle.cost(seq0))]

            endgame = EndgameEnumerator()

            while True:
                if all(len(rem) == 0 for _, rem, _ in beam):
                    break

                expansions = []
                seen_seq = set()
                for seq, rem, base_cost in beam:
                    if not rem:
                        tup = tuple(seq)
                        if tup not in seen_seq:
                            seen_seq.add(tup)
                            expansions.append((seq, rem, base_cost, 0.0))
                        continue

                    # Endgame exact enumeration
                    if len(rem) <= endgame_K:
                        c_end, s_end = endgame.solve(seq, rem, global_best=float('inf'))
                        tup = tuple(s_end)
                        if tup not in seen_seq:
                            seen_seq.add(tup)
                            expansions.append((s_end, set(), c_end, 0.0))
                        # Do not expand this node further
                        continue

                    # Candidate transactions to insert next
                    if len(rem) <= rem_all_threshold:
                        cand_txns = list(rem)
                    else:
                        cand_txns = random.sample(list(rem), min(k_txn_sample, len(rem)))

                    # Position policy (all vs sampled)
                    use_all = (len(rem) <= endgame_all_pos_threshold) or (len(seq) <= 18)
                    # For regret dispersion metrics
                    local_regs = []

                    for t in cand_txns:
                        best_c, best_p, second_c = ins_oracle.best_two(seq, t, use_all=use_all)
                        regret = (second_c - best_c) if second_c < float('inf') else 0.0
                        local_regs.append(regret)
                        new_seq = seq[:best_p] + [t] + seq[best_p:]
                        new_rem = rem.copy()
                        new_rem.remove(t)
                        tup = tuple(new_seq)
                        if tup in seen_seq:
                            continue
                        seen_seq.add(tup)
                        expansions.append((new_seq, new_rem, best_c, regret))

                if not expansions:
                    break

                # Dispersion-aware regret quota
                regs = [e[3] for e in expansions]
                if regs:
                    spread = (max(regs) - min(regs))
                    median = sorted(regs)[len(regs) // 2]
                    high_regret_quota_ratio = 0.5 if spread > median else 0.2
                else:
                    high_regret_quota_ratio = 0.3

                # Rank expansions
                by_cost = sorted(expansions, key=lambda x: (x[2], -x[3]))
                by_regret = sorted(expansions, key=lambda x: (-x[3], x[2]))

                base_quota = max(1, int(build_beam_width * (1.0 - high_regret_quota_ratio)))
                regret_quota = max(0, build_beam_width - base_quota)
                next_beam = []
                seen_sig = set()

                def try_add(entry):
                    s, r, c, reg = entry
                    sig = suffix_sig(s)
                    if sig in seen_sig:
                        return False
                    seen_sig.add(sig)
                    next_beam.append((s, r, c))
                    return True

                i = 0
                while len(next_beam) < base_quota and i < len(by_cost):
                    try_add(by_cost[i])
                    i += 1
                j = 0
                while len(next_beam) < base_quota + regret_quota and j < len(by_regret):
                    try_add(by_regret[j])
                    j += 1
                k = 0
                while len(next_beam) < build_beam_width and k < len(by_cost):
                    try_add(by_cost[k])
                    k += 1

                if not next_beam:
                    next_beam = [(s, r, c) for s, r, c, _ in by_cost[:build_beam_width]]

                beam = next_beam

            # Choose best complete
            best_seq = None
            best_cost = float('inf')
            for seq, rem, cost in beam:
                if rem:
                    seq_c = seq + sorted(list(rem))
                    c = cost_oracle.cost(seq_c)
                    if c < best_cost:
                        best_cost = c
                        best_seq = seq_c
                else:
                    if cost < best_cost:
                        best_cost = cost
                        best_seq = seq
            return best_seq

    class LocalSearch:
        def __init__(self):
            self.dont_look = [0] * n

        def reset_dl(self):
            for i in range(n):
                self.dont_look[i] = 0

        def vnd(self, seq):
            best_seq = seq[:]
            best_cost = cost_oracle.cost(best_seq)
            adj_rounds = 0

            def try_adjacent(cur_seq, cur_cost):
                improved = False
                for i in range(n - 1):
                    if self.dont_look[i]:
                        continue
                    cand = cur_seq[:]
                    cand[i], cand[i + 1] = cand[i + 1], cand[i]
                    c = cost_oracle.cost(cand)
                    if c < cur_cost:
                        self.dont_look[i] = 0
                        # reset neighbors to explore around improvement
                        if i > 0:
                            self.dont_look[i - 1] = 0
                        if i + 1 < n - 1:
                            self.dont_look[i + 1] = 0
                        return True, c, cand
                    else:
                        self.dont_look[i] = 1
                return improved, cur_cost, cur_seq

            def try_reinsertion(cur_seq, cur_cost):
                L = len(cur_seq)
                k_positions = max(6, int(reinsertion_pos_factor * k_pos_sample))
                for i in range(L):
                    if self.dont_look[i]:
                        continue
                    item = cur_seq[i]
                    base = cur_seq[:i] + cur_seq[i + 1:]
                    use_all = len(base) <= 20
                    # Use best-two to get candidates quickly
                    best_c, best_p, second_c = ins_oracle.best_two(base, item, use_all=use_all, focus_idx=i, k_override=k_positions)
                    # First try the best position
                    cand = base[:best_p] + [item] + base[best_p:]
                    if cand != cur_seq:
                        c = cost_oracle.cost(cand)
                        if c < cur_cost:
                            self.dont_look[i] = 0
                            return True, c, cand
                    # Try second best occasionally
                    if second_c < float('inf'):
                        # estimate second best position by exploring neighbors
                        pos_list = pos_policy.all_positions(len(base)) if use_all else pos_policy.sampled_positions(len(base), focus_idx=i, k_override=k_positions)
                        # evaluate to find the second best pos explicitly
                        _, _, _ = 0, 0, 0  # placeholder no-op
                        # We already have second cost, but not its pos; sample a few to seek better
                        for p in pos_list:
                            if p == best_p:
                                continue
                            cand2 = base[:p] + [item] + base[p:]
                            c2 = cost_oracle.cost(cand2)
                            if c2 < cur_cost:
                                self.dont_look[i] = 0
                                return True, c2, cand2
                    self.dont_look[i] = 1
                return False, cur_cost, cur_seq

            def try_oropt_k(cur_seq, cur_cost, kblk):
                L = len(cur_seq)
                k_positions = max(6, int(reinsertion_pos_factor * k_pos_sample))
                for i in range(L - kblk + 1):
                    block = cur_seq[i:i + kblk]
                    base = cur_seq[:i] + cur_seq[i + kblk:]
                    use_all = len(base) <= 20
                    best_c, best_p, second_c = ins_oracle.best_two_block(base, block, use_all=use_all, focus_idx=i, k_override=k_positions)
                    cand = base[:best_p] + block + base[best_p:]
                    if cand != cur_seq:
                        c = cost_oracle.cost(cand)
                        if c < cur_cost:
                            # reset local don't-look in block neighborhood
                            for j in range(max(0, i - 1), min(n - 1, i + kblk + 1)):
                                self.dont_look[j] = 0
                            return True, c, cand
                return False, cur_cost, cur_seq

            def try_two_opt(cur_seq, cur_cost):
                tried = 0
                while tried < two_opt_trials:
                    i = random.randint(0, n - 2)
                    j = random.randint(i + 2, n - 1)
                    tried += 1
                    cand = cur_seq[:i] + cur_seq[i:j + 1][::-1] + cur_seq[j + 1:]
                    c = cost_oracle.cost(cand)
                    if c < cur_cost:
                        # reset don't-look in affected range
                        for k in range(i, min(n - 1, j + 1)):
                            self.dont_look[k] = 0
                        return True, c, cand
                return False, cur_cost, cur_seq

            improved = True
            while improved:
                improved = False

                # Or-opt blocks 3 -> 2 -> 1
                for kblk in (3, 2, 1):
                    ch, nc, ns = try_oropt_k(best_seq, best_cost, kblk)
                    if ch:
                        best_seq, best_cost = ns, nc
                        improved = True
                        continue
                if improved:
                    continue

                # Adjacent swaps (limited rounds)
                if adj_rounds < ls_adj_rounds_max:
                    ch, nc, ns = try_adjacent(best_seq, best_cost)
                    if ch:
                        best_seq, best_cost = ns, nc
                        improved = True
                        continue
                    adj_rounds += 1

                # Reinsertion (Or-opt 1) with cached best-two
                ch, nc, ns = try_reinsertion(best_seq, best_cost)
                if ch:
                    best_seq, best_cost = ns, nc
                    improved = True
                    continue

                # Sampled 2-opt segment reversals
                ch, nc, ns = try_two_opt(best_seq, best_cost)
                if ch:
                    best_seq, best_cost = ns, nc
                    improved = True
                    continue

            return best_cost, best_seq

    def perturb(seq):
        s = seq[:]
        mode = random.random()
        if mode < 0.5:
            for _ in range(perturb_swap_count):
                i = random.randint(0, n - 1)
                j = random.randint(0, n - 1)
                if i != j:
                    s[i], s[j] = s[j], s[i]
        else:
            if n > perturb_block_len + 2:
                start = random.randint(0, n - perturb_block_len - 1)
                block = s[start:start + perturb_block_len]
                del s[start:start + perturb_block_len]
                insert_pos = random.randint(0, len(s))
                s = s[:insert_pos] + block + s[insert_pos:]
        return s

    def lns_attempt(seq):
        cur = seq[:]
        frac = random.uniform(*destroy_frac_range)
        m = max(4, min(n // 2, int(frac * n)))
        if random.random() < 0.5:
            remove_idxs = sorted(random.sample(range(n), m))
        else:
            start = random.randint(0, n - m)
            remove_idxs = list(range(start, start + m))
        remove_set = set(remove_idxs)
        removed = [cur[i] for i in remove_idxs]
        remaining = [cur[i] for i in range(n) if i not in remove_set]

        seq_rep = remaining[:]
        rem_set = removed[:]
        while rem_set:
            cand_txns = rem_set if len(rem_set) <= k_txn_sample else random.sample(rem_set, k_txn_sample)
            best_overall = (float('inf'), None, None)  # cost, txn, new_seq
            best_by_regret = (float('-inf'), None, None)
            use_all = (len(rem_set) <= endgame_all_pos_threshold) or (len(seq_rep) <= 18)

            for t in cand_txns:
                best_c, best_p, second_c = ins_oracle.best_two(seq_rep, t, use_all=use_all)
                c_regret = (second_c - best_c) if second_c < float('inf') else 0.0
                new_seq = seq_rep[:best_p] + [t] + seq_rep[best_p:]
                if best_c < best_overall[0]:
                    best_overall = (best_c, t, new_seq)
                if c_regret > best_by_regret[0]:
                    best_by_regret = (c_regret, t, new_seq)

            chosen = best_by_regret if best_by_regret[1] is not None and random.random() < regret_prob else best_overall
            if chosen[1] is None:
                t = random.choice(rem_set)
                seq_rep = seq_rep + [t]
                rem_set.remove(t)
            else:
                seq_rep = chosen[2]
                rem_set.remove(chosen[1])

        ls = LocalSearch()
        c_rep, s_rep = ls.vnd(seq_rep)
        return c_rep, s_rep

    def path_relink(source_seq, target_seq, max_moves=12):
        pos_in_target = {t: i for i, t in enumerate(target_seq)}
        s = source_seq[:]
        best_c = cost_oracle.cost(s)
        best_s = s[:]
        moves = 0
        disp = [(abs(i - pos_in_target[s[i]]), i) for i in range(n)]
        disp.sort(reverse=True)
        for _, idx in disp:
            if moves >= max_moves:
                break
            item = s[idx]
            desired = pos_in_target[item]
            if desired == idx:
                continue
            base = s[:idx] + s[idx + 1:]
            desired = max(0, min(desired, len(base)))
            # Use block insertion oracle if possible (block size 1 simplifies to best_two)
            best_cand = base[:desired] + [item] + base[desired:]
            c = cost_oracle.cost(best_cand)
            if c < best_c:
                best_c = c
                best_s = best_cand[:]
                s = best_cand
                moves += 1
        return best_c, best_s

    # ----------------------------
    # Seeding and elite management
    # ----------------------------
    singleton_scores = [(cost_oracle.cost([t]), t) for t in all_txns]
    singleton_scores.sort(key=lambda x: x[0])

    seed_txns = [t for _, t in singleton_scores[:seed_elite_singletons]]
    remaining = [t for t in all_txns if t not in seed_txns]
    if remaining and seed_random_additional > 0:
        seed_txns.extend(random.sample(remaining, min(seed_random_additional, len(remaining))))

    elite = []  # list of (cost, seq)
    def add_elite(c, s):
        nonlocal elite
        elite.append((c, s))
        elite.sort(key=lambda x: x[0])
        uniq = []
        seen_sig = set()
        seen_full = set()
        for c1, s1 in elite:
            sig = suffix_sig(s1, k=2)
            full = tuple(s1)
            if full in seen_full or sig in seen_sig:
                continue
            seen_full.add(full)
            seen_sig.add(sig)
            uniq.append((c1, s1))
            if len(uniq) >= elite_size:
                break
        elite = uniq

    constructor = BeamConstructor()

    for seed in seed_txns:
        seq0 = constructor.build(seed)
        ls = LocalSearch()
        c1, s1 = ls.vnd(seq0)
        add_elite(c1, s1)

    if not elite:
        base = all_txns[:]
        random.shuffle(base)
        elite = [(cost_oracle.cost(base), base)]

    best_overall_cost, best_overall_seq = elite[0]

    # ----------------------------
    # Iterated Local Search (ILS)
    # ----------------------------
    cur_cost, cur_seq = best_overall_cost, best_overall_seq[:]
    for _ in range(ils_rounds):
        pert = perturb(cur_seq)
        ls = LocalSearch()
        c2, s2 = ls.vnd(pert)
        if c2 < cur_cost:
            cur_cost, cur_seq = c2, s2
            add_elite(c2, s2)
            if c2 < best_overall_cost:
                best_overall_cost, best_overall_seq = c2, s2

    # ----------------------------
    # LNS destroy-and-repair
    # ----------------------------
    for _ in range(lns_iters):
        c3, s3 = lns_attempt(best_overall_seq)
        if c3 < best_overall_cost:
            best_overall_cost, best_overall_seq = c3, s3
            add_elite(c3, s3)

    # ----------------------------
    # Path Relinking among elites
    # ----------------------------
    if len(elite) >= 2:
        base_seq = best_overall_seq[:]
        partners = elite[1:min(len(elite), elite_size)]
        for c_t, s_t in partners:
            pr_c, pr_s = path_relink(base_seq, s_t, max_moves=max(8, min(12, n // 8)))
            if pr_c < best_overall_cost:
                # Quick polish
                ls = LocalSearch()
                lc, ls_seq = ls.vnd(pr_s)
                if lc < best_overall_cost:
                    best_overall_cost, best_overall_seq = lc, ls_seq
                else:
                    best_overall_cost, best_overall_seq = pr_c, pr_s

    return best_overall_cost, best_overall_seq


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