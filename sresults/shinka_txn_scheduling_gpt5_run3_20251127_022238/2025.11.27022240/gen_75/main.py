# EVOLVE-BLOCK-START
"""Transaction scheduling algorithm for optimizing makespan across multiple workloads"""

import time
import random
import sys
import os
import itertools
from collections import defaultdict

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


class Scheduler:
    def __init__(self, workload, num_seqs):
        self.workload = workload
        self.n = workload.num_txns
        # Deterministic RNG per workload size plus budget for stable, reproducible search
        self.rng = random.Random(1729 + self.n * 31 + int(num_seqs) * 7)

        # Shared caches and structures
        self.cost_cache = {}
        self.M = None
        self.W = None
        self.c1 = None
        self.tournament_order = None
        self.buddies = None

        # Anti-buddy thresholds per last txn (top quartile of positive W margins)
        self.pos_threshold = None

        # Unified prefix dominance across all builders
        self.prefix_dom = {}

        # Global incumbent shared across builders and local search
        self.inc_cost = float('inf')
        self.inc_seq = None

        # Portfolio parameter sets
        self.portfolio = [
            # (beam_width, cand_per_expand, lookahead_top, next_k)
            (16, 12, 4, 6),
            (12, 14, 3, 5),
            (10, 10, 3, 4),
        ]

    # ---------- Evaluator and precompute ----------
    def eval_seq(self, seq):
        key = tuple(seq)
        if key in self.cost_cache:
            return self.cost_cache[key]
        c = self.workload.get_opt_seq_cost(seq)
        self.cost_cache[key] = c
        return c

    def precompute(self):
        n = self.n
        # Singletons
        self.c1 = [self.eval_seq([i]) for i in range(n)]
        # Pairwise costs M[i][j] = cost of [i, j]
        self.M = [[0] * n for _ in range(n)]
        for i in range(n):
            Mi = self.M[i]
            for j in range(n):
                Mi[j] = self.c1[i] if i == j else self.eval_seq([i, j])

        # Preference margins W[i][j] = M[i][j] - M[j][i] (negative suggests i before j)
        self.W = [[0] * n for _ in range(n)]
        for i in range(n):
            Wi = self.W[i]
            Mi = self.M[i]
            for j in range(n):
                Wi[j] = 0 if i == j else (Mi[j] - self.M[j][i])

        # Tournament order (lower score -> earlier)
        s = [0] * n
        for i in range(n):
            s[i] = sum(self.W[i][j] for j in range(n) if j != i)
        self.tournament_order = list(range(n))
        self.tournament_order.sort(key=lambda x: (s[x], x))

        # Buddy lists: top partners by small M[t][u]
        buddy_k = 8 if n >= 90 else 6
        self.buddies = []
        for t in range(n):
            order = sorted((u for u in range(n) if u != t), key=lambda u: self.M[t][u])
            self.buddies.append(order[:buddy_k])

        # Anti-buddy: per txn x, compute threshold for "strongly disfavored" child t via positive W[x][t]
        self.pos_threshold = [0] * n
        for x in range(n):
            vals = [self.W[x][t] for t in range(n) if t != x and self.W[x][t] > 0]
            if not vals:
                self.pos_threshold[x] = 0
            else:
                vals.sort()
                idx = int(0.75 * (len(vals) - 1))
                self.pos_threshold[x] = vals[idx]

    # ---------- Helpers ----------
    def prefer_before(self, a, b):
        return self.M[a][b] <= self.M[b][a]

    def tournament_bubble_pass(self, seq, passes=2):
        arr = list(seq)
        for _ in range(passes):
            improved = False
            for k in range(len(arr) - 1):
                a, b = arr[k], arr[k + 1]
                if not self.prefer_before(a, b):
                    arr[k], arr[k + 1] = b, a
                    improved = True
            if not improved:
                break
        return arr

    def greedy_complete(self, prefix, remaining=None):
        if remaining is None:
            rem = [t for t in range(self.n) if t not in prefix]
        else:
            rem = list(remaining)
        cur = list(prefix)
        while rem:
            t = min(rem, key=lambda u: self.eval_seq(cur + [u]))
            cur.append(t)
            rem.remove(t)
        return cur, self.eval_seq(cur)

    def preselect_by_tournament(self, prefix, remaining, k, recent_k=4):
        if not remaining or k <= 0:
            return []
        recents = prefix[-recent_k:] if recent_k > 0 else []
        scored = []
        for t in remaining:
            sc = 0
            for x in recents:
                sc += self.W[x][t]
            scored.append((sc, t))
        scored.sort(key=lambda z: (z[0], z[1]))
        return [t for _, t in scored[:k]]

    def recent_k_for_depth(self, d):
        frac = d / max(1, self.n - 1)
        return 5 if frac < 0.33 else (4 if frac < 0.66 else 3)

    # Unified prefix dominance: skip if base_cost >= best stored for signature
    def dom_ok_and_update(self, remaining_fset, seq, base_cost):
        suffix_k = 3
        sig = (remaining_fset, tuple(seq[-suffix_k:]) if len(seq) >= suffix_k else tuple(seq))
        prev = self.prefix_dom.get(sig)
        if prev is not None and base_cost >= prev:
            return False
        if prev is None or base_cost < prev:
            self.prefix_dom[sig] = base_cost
        return True

    # ---------- Builders ----------
    def greedy_build(self, params, start_mode="auto"):
        n = self.n
        rng = self.rng

        # Seed selection
        if start_mode == "tournament":
            t0 = self.tournament_order[0]
        elif start_mode == "best":
            topk = min(10, n)
            top_singletons = sorted(range(n), key=lambda t: self.c1[t])[:topk]
            t0 = rng.choice(top_singletons)
        elif start_mode == "random":
            t0 = rng.randint(0, n - 1)
        else:
            # auto: mix tournament and best singles
            pool = set(self.tournament_order[:min(8, n)])
            pool.update(sorted(range(n), key=lambda t: self.c1[t])[:min(8, n)])
            t0 = rng.choice(list(pool))

        seq = [t0]
        remaining = set(range(n)) - {t0}
        step = 0
        preselect_pool_size = 16 if n >= 90 else 12

        while remaining:
            base_cost = self.eval_seq(seq)
            # Incumbent pruning
            if self.inc_cost < float('inf') and base_cost >= self.inc_cost:
                return None, float('inf')

            # Prefix dominance prune/update
            if not self.dom_ok_and_update(frozenset(remaining), seq, base_cost):
                return None, float('inf')

            # Incumbent tightening via periodic greedy completion
            if step % 9 == 0:
                full, fc = self.greedy_complete(seq, remaining)
                if fc < self.inc_cost:
                    self.inc_cost, self.inc_seq = fc, full

            R = len(remaining)
            # Candidate pool by tournament-preselection
            if R <= preselect_pool_size:
                cand_pool = list(remaining)
            else:
                cand_pool = self.preselect_by_tournament(seq, list(remaining), preselect_pool_size, recent_k=self.recent_k_for_depth(len(seq)))

            # Evaluate immediate costs
            imm = [(t, self.eval_seq(seq + [t])) for t in cand_pool]
            imm.sort(key=lambda x: x[1])

            # Anti-buddy filtering relative to last placed
            last = seq[-1]
            pos_thr = self.pos_threshold[last]
            best_immediate = imm[0][1] if imm else float('inf')

            # Adaptive lookahead depth
            lookahead_top, next_k = params["lookahead_top"], params["next_k"]
            L = min(lookahead_top, len(imm))
            chosen_t = imm[0][0] if imm else rng.choice(list(remaining))
            best_metric = best_immediate

            for idx in range(L):
                t, immediate_c = imm[idx]
                # Anti-buddy skip unless within 1% of best immediate
                if self.W[last][t] > pos_thr and immediate_c > best_immediate * 1.01:
                    continue

                nexts = [u for u in remaining if u != t]
                if not nexts:
                    metric = immediate_c
                else:
                    buddy_pref = [u for u in self.buddies[t] if u in nexts][:next_k]
                    if len(buddy_pref) < next_k:
                        extra = self.preselect_by_tournament(seq + [t], [u for u in nexts if u not in buddy_pref],
                                                            next_k - len(buddy_pref), recent_k=3)
                        pool = buddy_pref + extra
                    else:
                        pool = buddy_pref
                    if not pool:
                        pool = nexts[:min(next_k, len(nexts))]
                    metric = min(self.eval_seq(seq + [t, u]) for u in pool)
                if metric < best_metric:
                    best_metric = metric
                    chosen_t = t

            # Extend
            seq.append(chosen_t)
            remaining.remove(chosen_t)
            step += 1

        c = self.eval_seq(seq)
        return seq, c

    def beam_build(self, params):
        n = self.n
        rng = self.rng

        beam_width = params["beam_width"]
        cand_per_expand = params["cand_per_expand"]
        lookahead_top = params["lookahead_top"]
        next_k = params["next_k"]
        depth_limit = max(3, int(0.4 * n))

        # Initialize starts: tournament-best, a good singleton, then randoms
        starts = [self.tournament_order[0]]
        good_singletons = sorted(range(n), key=lambda t: self.c1[t])[:min(10, n)]
        if good_singletons:
            starts.append(rng.choice(good_singletons))
        remcands = [t for t in range(n) if t not in starts]
        rng.shuffle(remcands)
        starts.extend(remcands[:max(0, beam_width - len(starts))])

        beam = []
        seen_init = set()
        for t in starts:
            seq = [t]
            rem = frozenset(set(range(n)) - {t})
            cost = self.eval_seq(seq)
            key = (tuple(seq), rem)
            if key in seen_init:
                continue
            seen_init.add(key)
            # dominance
            if self.dom_ok_and_update(rem, seq, cost):
                beam.append((cost, seq, rem))

        depth = 1
        while depth < min(depth_limit, n) and beam:
            # Incumbent tightening on top items
            beam_sorted = sorted(beam, key=lambda x: x[0])
            for bc, bseq, brem in beam_sorted[:2]:
                if bc < self.inc_cost:
                    full, fc = self.greedy_complete(bseq, brem)
                    if fc < self.inc_cost:
                        self.inc_cost, self.inc_seq = fc, full

            next_beam = []
            local_seen = set()

            for base_cost, seq, rem in beam:
                if self.inc_cost < float('inf') and base_cost >= self.inc_cost:
                    continue

                rem_list = list(rem)
                if not rem_list:
                    next_beam.append((base_cost, seq, rem))
                    continue

                recent_k = self.recent_k_for_depth(len(seq))
                # Tournament preselection
                pre = rem_list if len(rem_list) <= cand_per_expand * 2 else self.preselect_by_tournament(seq, rem_list, cand_per_expand * 2, recent_k=recent_k)

                # Immediate costs and anti-buddy filter
                last = seq[-1]
                pos_thr = self.pos_threshold[last]
                imm = []
                for t in pre:
                    c_im = self.eval_seq(seq + [t])
                    if self.inc_cost < float('inf') and c_im >= self.inc_cost:
                        continue
                    imm.append((t, c_im))
                if not imm:
                    continue
                imm.sort(key=lambda z: z[1])
                best_immediate = imm[0][1]

                # Score top children with lookahead
                L = min(lookahead_top, len(imm))
                scored = []
                for t, imc in imm[:L]:
                    # Anti-buddy skip unless near-best immediate
                    if self.W[last][t] > pos_thr and imc > best_immediate * 1.01:
                        continue
                    nexts = [u for u in rem_list if u != t]
                    if not nexts:
                        la = imc
                    else:
                        buddy_pref = [u for u in self.buddies[t] if u in nexts][:next_k]
                        if len(buddy_pref) < next_k:
                            extra = self.preselect_by_tournament(seq + [t], [u for u in nexts if u not in buddy_pref],
                                                                 next_k - len(buddy_pref), recent_k=3)
                            pool = buddy_pref + extra
                        else:
                            pool = buddy_pref
                        if not pool:
                            pool = nexts[:min(next_k, len(nexts))]
                        la = min(self.eval_seq(seq + [t, u]) for u in pool)
                    scored.append((t, min(imc, la)))

                # Add a few immediate best for diversity
                diversity = min(max(2, cand_per_expand // 3), len(imm))
                for t, imc in imm[:diversity]:
                    scored.append((t, imc))

                # Unique best children and expand top-k
                uniq = {}
                for t, m in scored:
                    if (t not in uniq) or (m < uniq[t]):
                        uniq[t] = m
                items = sorted(uniq.items(), key=lambda z: z[1])
                take = min(cand_per_expand, len(items))
                for t, _ in items[:take]:
                    new_seq = seq + [t]
                    new_cost = self.eval_seq(new_seq)
                    if self.inc_cost < float('inf') and new_cost >= self.inc_cost:
                        continue
                    new_rem = rem - {t}
                    if not self.dom_ok_and_update(new_rem, new_seq, new_cost):
                        continue
                    key = (tuple(new_seq), new_rem)
                    if key in local_seen:
                        continue
                    local_seen.add(key)

                    # Incumbent-aware greedy completion at child expansion
                    # Only on top 2 children per parent by immediate metric
                    if len(items) <= 2 or t in [items[0][0], items[1][0]]:
                        full, fc = self.greedy_complete(new_seq, new_rem)
                        if fc < self.inc_cost:
                            self.inc_cost, self.inc_seq = fc, full
                        # Prune if child base is already not competitive
                        if self.inc_cost < float('inf') and new_cost >= self.inc_cost:
                            continue

                    next_beam.append((new_cost, new_seq, new_rem))

            if not next_beam:
                break
            next_beam.sort(key=lambda x: x[0])
            beam = next_beam[:beam_width]
            depth += 1

        if not beam:
            # Fallback: greedy-complete from a random singleton
            t = rng.randint(0, n - 1)
            return self.greedy_complete([t])

        beam.sort(key=lambda x: x[0])
        _, bseq, brem = beam[0]
        return self.greedy_complete(bseq, brem)

    # ---------- Local search / LNS ----------
    def prefix_marginals(self, seq):
        prefix_costs = [0] * len(seq)
        c = 0
        for i in range(len(seq)):
            c = self.eval_seq(seq[: i + 1])
            prefix_costs[i] = c
        marg = [prefix_costs[0]] + [prefix_costs[i] - prefix_costs[i - 1] for i in range(1, len(seq))]
        return prefix_costs, marg

    def worst_violation_boundaries(self, seq, topm=3):
        pairs = []
        for i in range(len(seq) - 1):
            a, b = seq[i], seq[i + 1]
            pen = self.M[a][b] - self.M[b][a]  # positive when a before b is worse
            if pen > 0:
                pairs.append((pen, i))
        pairs.sort(reverse=True)
        return [i for _, i in pairs[:topm]]

    def adjacency_surrogate(self, seq, start, end):
        # Sum W margins across affected adjacency pairs [start-1..end]
        s = 0
        for i in range(max(0, start - 1), min(len(seq) - 1, end) + 1):
            a, b = seq[i], seq[i + 1]
            s += max(0, self.M[a][b] - self.M[b][a])
        return s

    def adjacent_pass(self, seq, current_cost):
        best_seq = list(seq)
        best_cost = current_cost
        improved = True
        n_local = len(best_seq)
        while improved:
            improved = False
            for i in range(n_local - 1):
                cand = best_seq[:]
                cand[i], cand[i + 1] = cand[i + 1], cand[i]
                c = self.eval_seq(cand)
                if c < best_cost:
                    best_seq, best_cost = cand, c
                    improved = True
        return best_seq, best_cost

    def lns_improve(self, seq, base_cost, budget_factor):
        rng = self.rng
        best_seq = seq[:]
        best_cost = base_cost
        n_local = len(best_seq)

        iters = max(4, min(10, 2 + int(budget_factor) + n_local // 40))
        max_k = 7 if n_local >= 40 else 6
        # Cap total expensive evaluations per round
        eval_cap_per_round = 700

        for it in range(iters):
            used_evals = 0

            # Cheap tournament cleanup
            tb = self.tournament_bubble_pass(best_seq, passes=2)
            cb = self.eval_seq(tb)
            used_evals += 1
            if cb < best_cost:
                best_seq, best_cost = tb, cb

            # Compute marginals and hotspot windows
            _, marg = self.prefix_marginals(best_seq)

            # Choose window size
            k = max_k if it < 2 else rng.randint(4, max_k)
            if n_local <= k:
                break

            # Sliding window sums
            sums = []
            window_sum = sum(marg[0:k])
            sums.append((0, window_sum))
            for s in range(1, n_local - k + 1):
                window_sum += marg[s + k - 1] - marg[s - 1]
                sums.append((s, window_sum))
            sums.sort(key=lambda x: -x[1])

            # Violation-boundary starts
            viol_starts = self.worst_violation_boundaries(best_seq, topm=3)
            candidate_starts = []
            for v in viol_starts:
                start = max(0, min(v - (k // 2), n_local - k))
                candidate_starts.append(start)
            candidate_starts.extend([start for start, _ in sums[:2]])

            # Deduplicate starts
            seen_starts = set()
            ordered_starts = []
            for st in candidate_starts:
                if st not in seen_starts:
                    seen_starts.add(st)
                    ordered_starts.append(st)

            # Window permutation with ΔW-gated evaluation
            for start in ordered_starts:
                if used_evals >= eval_cap_per_round:
                    break
                block = best_seq[start:start + k]
                base = best_seq[:start] + best_seq[start + k:]

                # Prepare candidate permutations budget
                factorial = 1
                for i2 in range(2, k + 1):
                    factorial *= i2
                perm_budget = min(720, factorial) if k <= 6 else 2000

                # Surrogate rank by local adjacency penalty
                # We generate a sample set then pick top by surrogate
                cand_perms = []
                if factorial <= perm_budget:
                    iterable = itertools.permutations(block)
                    # We will rank all by surrogate, but compute surrogate cheaply
                    for p in iterable:
                        perm = list(p)
                        cand = base[:start] + perm + base[start:]
                        sur = self.adjacency_surrogate(cand, start, start + k)
                        cand_perms.append((sur, perm))
                else:
                    seenp = set()
                    attempts = 0
                    while attempts < perm_budget:
                        p = tuple(rng.sample(block, k))
                        if p in seenp:
                            continue
                        seenp.add(p)
                        perm = list(p)
                        cand = base[:start] + perm + base[start:]
                        sur = self.adjacency_surrogate(cand, start, start + k)
                        cand_perms.append((sur, perm))
                        attempts += 1

                cand_perms.sort(key=lambda x: x[0])
                # Evaluate top 40% by surrogate + 10% random
                top_count = max(1, int(0.4 * len(cand_perms)))
                eval_set_idx = set(range(top_count))
                extra = max(0, int(0.1 * len(cand_perms)))
                while len(eval_set_idx) < top_count + extra:
                    eval_set_idx.add(rng.randint(0, len(cand_perms) - 1))

                improved_here = False
                for idx in sorted(list(eval_set_idx)):
                    if used_evals >= eval_cap_per_round:
                        break
                    perm = cand_perms[idx][1]
                    cand_seq = base[:start] + perm + base[start:]
                    c = self.eval_seq(cand_seq)
                    used_evals += 1
                    if c < best_cost:
                        best_seq, best_cost = cand_seq, c
                        improved_here = True
                        break
                if improved_here:
                    break  # Accept first improving window

            if used_evals >= eval_cap_per_round:
                continue

            # Block-swap neighborhood centered on worst adjacencies
            if n_local >= 12:
                viols = self.worst_violation_boundaries(best_seq, topm=2)
                if viols:
                    k1 = 4 if n_local >= 40 else 3
                    k2 = k1
                    s1 = max(0, min(viols[0] - k1 // 2, n_local - k1))
                    # choose candidate s2 positions
                    s2_candidates = [max(0, min((viols[-1] if len(viols) > 1 else viols[0]) + k1, n_local - k2))]
                    for _ in range(2):
                        s2_candidates.append(rng.randint(0, n_local - k2))
                    evaluated = 0
                    for s2 in s2_candidates:
                        if abs(s2 - s1) < k1:
                            continue
                        block1 = best_seq[s1:s1 + k1]
                        block2 = best_seq[s2:s2 + k2]
                        if s1 < s2:
                            cand = best_seq[:s1] + block2 + best_seq[s1 + k1:s2] + block1 + best_seq[s2 + k2:]
                        else:
                            cand = best_seq[:s2] + block1 + best_seq[s2 + k2:s1] + block2 + best_seq[s1 + k1:]
                        # ΔW surrogate to rank
                        sur = self.adjacency_surrogate(cand, min(s1, s2) - 1, max(s1 + k1, s2 + k2))
                        # Evaluate only if surrogate is promising
                        if evaluated < 4 or sur <= self.adjacency_surrogate(best_seq, min(s1, s2) - 1, max(s1 + k1, s2 + k2)):
                            c = self.eval_seq(cand)
                            used_evals += 1
                            evaluated += 1
                            if c < best_cost:
                                best_seq, best_cost = cand, c
                                break

            if used_evals >= eval_cap_per_round:
                continue

            # Block reinsert neighborhood
            if n_local >= 10:
                kbr = 5 if n_local >= 50 else 4
                # pick hot block by marginal sum
                sums = []
                window_sum = sum(marg[0:kbr])
                sums.append((0, window_sum))
                for s in range(1, n_local - kbr + 1):
                    window_sum += marg[s + kbr - 1] - marg[s - 1]
                    sums.append((s, window_sum))
                sums.sort(key=lambda x: -x[1])
                start = sums[0][0]
                block = best_seq[start:start + kbr]
                base = best_seq[:start] + best_seq[start + kbr:]
                # Candidate insertion positions ranked by ΔW surrogate
                pos_candidates = list(range(0, len(base) + 1))
                rank = []
                for pos in pos_candidates:
                    tmp = base[:pos] + block + base[pos:]
                    sur = self.adjacency_surrogate(tmp, pos - 1, pos + kbr)
                    rank.append((sur, pos))
                rank.sort(key=lambda x: x[0])
                # Evaluate top few plus a couple randoms
                eval_pos = [p for _, p in rank[:6]]
                added = set(eval_pos)
                while len(eval_pos) < 8 and len(added) < len(pos_candidates):
                    candp = rng.choice(pos_candidates)
                    if candp not in added:
                        eval_pos.append(candp)
                        added.add(candp)
                for pos in eval_pos:
                    cand = base[:pos] + block + base[pos:]
                    c = self.eval_seq(cand)
                    used_evals += 1
                    if c < best_cost:
                        best_seq, best_cost = cand, c
                        break

        return best_seq, best_cost

    # ---------- Orchestrator ----------
    def run(self, num_seqs):
        self.precompute()

        # Portfolio restarts share caches and dominance, deterministic seeds per portfolio index
        seeds = []

        for r, (bw, cpe, la, nk) in enumerate(self.portfolio):
            params = {
                "beam_width": bw,
                "cand_per_expand": cpe,
                "lookahead_top": la,
                "next_k": nk,
            }
            # Beam seed
            b_seq, b_cost = self.beam_build(params)
            # Light clean
            b_seq_t = self.tournament_bubble_pass(b_seq, passes=2)
            b_cost_t = self.eval_seq(b_seq_t)
            if b_cost_t < b_cost:
                b_seq, b_cost = b_seq_t, b_cost_t
            seeds.append((b_cost, b_seq))

            # Greedy seeds with different start modes
            for mode in ["tournament", "best", "random"]:
                g_seq, g_cost = self.greedy_build(params, start_mode=mode)
                if g_seq is None:
                    continue
                # Light adjacent cleanup
                g_seq, g_cost = self.adjacent_pass(g_seq, g_cost)
                seeds.append((g_cost, g_seq))

        # Promote incumbent from seeds
        seeds.sort(key=lambda x: x[0])
        if seeds and seeds[0][0] < self.inc_cost:
            self.inc_cost, self.inc_seq = seeds[0]

        # Apply full LNS only on top-2 seeds; others get quick passes
        final_best_cost = self.inc_cost
        final_best_seq = self.inc_seq[:]
        top_k_lns = min(2, len(seeds))
        for idx, (sc, ss) in enumerate(seeds):
            if idx < top_k_lns:
                l_seq, l_cost = self.lns_improve(ss, sc, budget_factor=max(1, int(num_seqs)))
                if l_cost < final_best_cost:
                    final_best_cost, final_best_seq = l_cost, l_seq
            else:
                # Quick adjacent + tournament cleanup
                s1, c1 = self.adjacent_pass(ss, sc)
                s1 = self.tournament_bubble_pass(s1, passes=2)
                c1 = self.eval_seq(s1)
                if c1 < final_best_cost:
                    final_best_cost, final_best_seq = c1, s1

        # Final small polish: sampled 2-opt and random insertion
        n = self.n
        two_opt_trials = min(60, n)
        for _ in range(two_opt_trials):
            i, j = self.rng.sample(range(n), 2)
            if abs(i - j) <= 1:
                continue
            cand = final_best_seq[:]
            cand[i], cand[j] = cand[j], cand[i]
            c = self.eval_seq(cand)
            if c < final_best_cost:
                final_best_seq, final_best_cost = cand, c

        move_budget = min(60, n)
        for _ in range(move_budget):
            i, j = self.rng.sample(range(n), 2)
            if i == j:
                continue
            cand = final_best_seq[:]
            item = cand.pop(i)
            cand.insert(j, item)
            c = self.eval_seq(cand)
            if c < final_best_cost:
                final_best_seq, final_best_cost = cand, c

        return final_best_cost, final_best_seq


def get_best_schedule(workload, num_seqs):
    """
    Incumbent-aware portfolio search combining beam and greedy builders
    with ΔW-gated LNS to minimize makespan.
    """
    scheduler = Scheduler(workload, num_seqs)
    return scheduler.run(num_seqs)


def get_random_costs():
    """
    Evaluate scheduling algorithm on three different workloads.

    Returns:
        Tuple of (total_makespan, list_of_schedules, execution_time)
    """
    start_time = time.time()

    # Workload 1: Complex mixed read/write transactions
    workload = Workload(WORKLOAD_1)
    makespan1, schedule1 = get_best_schedule(workload, 12)
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