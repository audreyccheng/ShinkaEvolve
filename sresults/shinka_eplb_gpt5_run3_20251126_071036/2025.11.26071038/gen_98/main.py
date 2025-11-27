# EVOLVE-BLOCK-START
"""
Expert parallelism load balancer (EPLB) for vLLM.

Structural redesign:
- ReplicaAllocator: water-filling with adaptive, peak-aware hybrid tail and guarded fix-up.
- CapacityPacker: unified packer for both group->node and GPU placement with bounded refinement,
  diversity-aware tie-breaking (GPU layer only), and adaptive iteration.
- HierarchicalPlanner: orchestrates group packing, replication, and GPU packing with clean
  index transforms and inverses.

Inputs/outputs remain identical to the original API.
"""

import torch
from collections import defaultdict
from typing import Optional, Tuple


# -------------------------- Small utilities --------------------------

def _inverse_rows(perm: torch.Tensor) -> torch.Tensor:
    """
    Compute inverse permutation row-wise.
    perm: [R, N] with values in [0, N)
    Returns inv: [R, N] such that inv[r, perm[r, j]] = j
    """
    R, N = perm.shape
    inv = torch.empty_like(perm)
    inv.scatter_(1, perm, torch.arange(N, dtype=torch.int64, device=perm.device).expand(R, N))
    return inv


def _build_blocked_map(pack_idx: torch.Tensor, rank_in_pack: torch.Tensor, block_size: int) -> torch.Tensor:
    """
    Convert pack assignment + rank to contiguous blocked indices:
      blocked = pack_idx * block_size + rank_in_pack
    """
    return pack_idx * block_size + rank_in_pack


# -------------------------- Replica allocator --------------------------

class ReplicaAllocator:
    """
    Row-wise water-filling replica allocator with:
    - Binary search scale
    - Bulk D'Hondt fill
    - Adaptive, peak-aware hybrid tail comparing D'Hondt and Sainte–Laguë at each step
    - Bounded 2-step donor->receiver fix-up targeting peak reduction
    """

    @staticmethod
    def _counts_row(w: torch.Tensor, target_total: int) -> torch.Tensor:
        """
        Compute integer replica counts c_i >= 1 that approximately minimize max w_i / c_i
        subject to sum c_i == target_total.
        w: [E], float CPU
        """
        num_log = w.numel()
        assert target_total >= num_log
        if target_total == num_log:
            return torch.ones(num_log, dtype=torch.int64, device=w.device)

        maxw = float(w.max().item()) if num_log > 0 else 0.0
        if maxw == 0.0:
            # All-zero load: uniform distribution
            counts = torch.ones(num_log, dtype=torch.int64, device=w.device)
            extras = target_total - num_log
            if extras > 0:
                q, r = divmod(extras, num_log)
                if q:
                    counts += q
                if r:
                    counts[:r] += 1
            return counts

        # Binary search for scale hi s.t. sum ceil(w/hi) <= target_total
        lo, hi = 0.0, max(maxw, 1.0)
        for _ in range(40):
            mid = 0.5 * (lo + hi)
            c = torch.ceil(w / mid).to(torch.int64)
            c = torch.maximum(c, torch.ones_like(c))
            s = int(c.sum().item())
            if s <= target_total:
                hi = mid
            else:
                lo = mid

        counts = torch.ceil(w / hi).to(torch.int64)
        counts = torch.maximum(counts, torch.ones_like(counts))
        s = int(counts.sum().item())
        extras = target_total - s
        if extras <= 0:
            return counts

        # Bulk D'Hondt fill except for an adaptive tail sized by current imbalance
        num_redundant = max(0, target_total - num_log)
        counts_f = counts.to(w.dtype)
        avg = w / counts_f
        mean_avg = float(w.sum().item() / max(1, target_total))
        cur_peak = float(avg.max().item())
        # clamp(cur_peak/mean(avg) − 1, 0, 1)
        ratio = 0.0
        if mean_avg > 0:
            ratio = max(0.0, min(1.0, cur_peak / mean_avg - 1.0))
        tail = min(extras, max(1, int(round(0.1 * num_redundant * ratio))))
        bulk = extras - tail

        # Fast bulk fill via topk of D'Hondt scores
        while bulk > 0:
            k = min(bulk, num_log)
            scores = w / counts.to(w.dtype)
            topk_idx = torch.topk(scores, k=k, largest=True).indices
            counts[topk_idx] += 1
            bulk -= k

        if tail > 0:
            # Hybrid tail: at each step pick between D'Hondt and Sainte–Laguë to minimize
            # (new_peak, new_second_peak, receiver_count) deterministically
            for _ in range(tail):
                counts_f = counts.to(w.dtype)
                avg = w / counts_f
                if num_log >= 2:
                    vals, idxs = torch.topk(avg, k=2)
                    peak1_val = float(vals[0].item())
                    peak1_idx = int(idxs[0].item())
                    peak2_val = float(vals[1].item())
                else:
                    peak1_val = float(avg[0].item())
                    peak1_idx = 0
                    peak2_val = float("-inf")

                # D'Hondt: argmax w/c
                d_pick = int(torch.argmax(avg).item())
                # Sainte–Laguë: argmax w/(2c+1)
                denom = (2 * counts + 1).to(w.dtype)
                s_scores = w / denom
                s_pick = int(torch.argmax(s_scores).item())

                best_choice = None
                best_key = None  # (new_peak, new_second_peak, receiver_count)
                # Evaluate both candidates; small constant-time work
                for pick in (d_pick, s_pick):
                    if best_choice is not None and pick == best_choice:
                        continue
                    new_avg = float((w[pick] / (counts_f[pick] + 1.0)).item())
                    if pick == peak1_idx:
                        new_peak = max(new_avg, peak2_val)
                        new_second = min(new_avg, peak2_val)
                    else:
                        new_peak = max(peak1_val, new_avg)
                        new_second = peak1_val if new_avg >= peak1_val else max(new_avg, peak2_val)
                    cand = (new_peak, new_second, int(counts[pick].item()))
                    if best_key is None or cand < best_key:
                        best_key = cand
                        best_choice = pick
                counts[best_choice] += 1

        return counts

    @staticmethod
    def _fixup_counts(w: torch.Tensor, counts: torch.Tensor, max_iters: int = 2) -> torch.Tensor:
        """
        Up to 2 donor->receiver moves. Donors: top-2 avg with count>1, receivers: bottom-2 avg.
        Accept a move only if it strictly reduces the peak; tie-break with second peak, then donor post-move avg.
        """
        num_log = w.numel()
        if num_log <= 1:
            return counts
        for _ in range(max_iters):
            c_safe = torch.clamp(counts, min=1)
            avg = w / c_safe.to(w.dtype)
            cur_peak = float(avg.max().item())
            can_donate = (counts > 1)
            if not bool(can_donate.any()):
                break
            k_d = int(min(2, int(can_donate.sum().item())))
            avg_mask = avg.clone()
            avg_mask[~can_donate] = float("-inf")
            donors = torch.topk(avg_mask, k=k_d).indices.tolist()
            receivers = torch.topk(-avg, k=int(min(2, num_log))).indices.tolist()

            best_pair = None
            best_key = None
            eps = 1e-9
            for d in donors:
                for r in receivers:
                    if d == r or counts[d] <= 1:
                        continue
                    c_try = counts.clone()
                    c_try[d] -= 1
                    c_try[r] += 1
                    avg_try = w / c_try.to(w.dtype)
                    new_peak = float(avg_try.max().item())
                    if new_peak + eps >= cur_peak:
                        continue
                    k2 = min(2, num_log)
                    top2_vals = torch.topk(avg_try, k=k2).values
                    new_second = float(top2_vals[1].item()) if top2_vals.numel() >= 2 else float("-inf")
                    donor_post = float((w[d] / float(counts[d] - 1)).item()) if counts[d] > 1 else float("inf")
                    cand_key = (new_peak, new_second, donor_post)
                    if best_key is None or cand_key < best_key:
                        best_key = cand_key
                        best_pair = (d, r)
            if best_pair is None:
                break
            d, r = best_pair
            counts[d] -= 1
            counts[r] += 1
        return counts

    def allocate(self, weight: torch.Tensor, num_phy: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        weight: [R, E] CPU float
        Returns:
          phy2log: [R, num_phy]
          rank:    [R, num_phy]
          logcnt:  [R, E]
        """
        R, E = weight.shape
        device = weight.device
        exp_ids = torch.arange(E, dtype=torch.int64, device=device)

        phy2log_rows = []
        rank_rows = []
        logcnt_rows = []

        for i in range(R):
            w = weight[i]
            counts = self._counts_row(w, num_phy)
            counts = self._fixup_counts(w, counts, max_iters=2)
            logcnt_rows.append(counts)

            # Build contiguous phy2log and rank
            phy2log_i = torch.repeat_interleave(exp_ids, counts)
            starts = torch.cumsum(counts, dim=0) - counts
            arange_phy = torch.arange(num_phy, dtype=torch.int64, device=device)
            rank_i = arange_phy - torch.repeat_interleave(starts, counts)

            phy2log_rows.append(phy2log_i)
            rank_rows.append(rank_i)

        phy2log = torch.stack(phy2log_rows, dim=0)
        rank = torch.stack(rank_rows, dim=0)
        logcnt = torch.stack(logcnt_rows, dim=0)
        return phy2log, rank, logcnt


# -------------------------- Capacity packer --------------------------

class CapacityPacker:
    """
    Unified capacity-constrained packer with optional label awareness.
    - Greedy LPT with min-load and capacity guard (tie-break by fewer items)
    - Optional diversity-aware tie-breaking (minimize label repeats) for GPU layer
    - Bounded refinement: search a tiny set of promising 1x1 swaps; fallback 2x2 when needed
    - Adaptive refinement depth based on residual imbalance; early exit if no improvement
    """

    def __init__(self, num_packs: int, with_diversity: bool):
        self.num_packs = num_packs
        self.with_diversity = with_diversity

    def _pack_row(
        self,
        row_w: torch.Tensor,
        cap: int,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        n_items = row_w.numel()
        device = row_w.device
        num_packs = self.num_packs

        # Trivial capacity case
        if cap == 1 or num_packs == 1:
            idx = torch.arange(n_items, dtype=torch.int64, device=device)
            return idx % num_packs, torch.zeros(n_items, dtype=torch.int64, device=device)

        # Pre-sort items by weight descending
        sorted_idx = row_w.sort(descending=True).indices.tolist()

        loads = [0.0] * num_packs
        counts = [0] * num_packs
        pack_groups = [[] for _ in range(num_packs)]

        # Label book-keeping only if requested
        label_counts = [defaultdict(int) for _ in range(num_packs)] if (self.with_diversity and labels is not None) else None
        assigned = [False] * n_items

        pack_index_row = torch.full((n_items,), -1, dtype=torch.int64, device=device)
        rank_row = torch.full_like(pack_index_row, -1)

        # Optional fast path when labels are unique
        all_unique = False
        if label_counts is not None:
            lbl_sorted, _ = torch.sort(labels)
            all_unique = (torch.unique_consecutive(lbl_sorted).numel() == labels.numel())

        if label_counts is None or all_unique:
            # Standard greedy placement
            for g in sorted_idx:
                wv = float(row_w[g].item())
                best_p = None
                best_load = None
                best_cnt = None
                for p in range(num_packs):
                    if counts[p] >= cap:
                        continue
                    if (best_load is None or loads[p] < best_load or
                        (abs(loads[p] - (best_load if best_load is not None else 0.0)) <= 1e-12 and
                         (best_cnt is None or counts[p] < best_cnt))):
                        best_p = p
                        best_load = loads[p]
                        best_cnt = counts[p]
                pack_index_row[g] = best_p
                rank_row[g] = counts[best_p]
                counts[best_p] += 1
                loads[best_p] += wv
                pack_groups[best_p].append(g)
        else:
            # Phase 1: label spreading (one heaviest per label)
            label_items = defaultdict(list)
            for g in sorted_idx:
                lab = int(labels[g].item())
                label_items[lab].append(g)
            label_order = sorted(label_items.keys(),
                                 key=lambda lab: float(row_w[label_items[lab][0]].item()),
                                 reverse=True)

            for lab in label_order:
                if counts.count(cap) == num_packs:
                    break
                g = None
                for cand in label_items[lab]:
                    if not assigned[cand]:
                        g = cand
                        break
                if g is None:
                    continue
                wv = float(row_w[g].item())
                best_p = None
                best_load = None
                best_cnt = None
                # Prefer packs without this label
                for p in range(num_packs):
                    if counts[p] >= cap:
                        continue
                    if label_counts[p].get(lab, 0) > 0:
                        continue
                    if (best_load is None or loads[p] < best_load or
                        (abs(loads[p] - (best_load if best_load is not None else 0.0)) <= 1e-12 and counts[p] < (best_cnt if best_cnt is not None else 1 << 30))):
                        best_p = p
                        best_load = loads[p]
                        best_cnt = counts[p]
                # Fallback to any pack with capacity
                if best_p is None:
                    for p in range(num_packs):
                        if counts[p] >= cap:
                            continue
                        if (best_load is None or loads[p] < best_load or
                            (abs(loads[p] - best_load) <= 1e-12 and counts[p] < best_cnt)):
                            best_p = p
                            best_load = loads[p]
                            best_cnt = counts[p]

                if best_p is not None:
                    pack_index_row[g] = best_p
                    rank_row[g] = counts[best_p]
                    counts[best_p] += 1
                    loads[best_p] += wv
                    label_counts[best_p][lab] += 1
                    pack_groups[best_p].append(g)
                    assigned[g] = True

            # Phase 2: greedy fill with diversity tie-breaking
            eps = 1e-6 * float(row_w.mean().item() if row_w.numel() > 0 else 1.0)
            for g in sorted_idx:
                if assigned[g]:
                    continue
                lab = int(labels[g].item())
                wv = float(row_w[g].item())
                best_p = None
                best_base = None
                best_rep = None
                best_cnt = None
                for p in range(num_packs):
                    if counts[p] >= cap:
                        continue
                    base = loads[p] + wv
                    rep = label_counts[p].get(lab, 0)
                    if best_p is None:
                        best_p, best_base, best_rep, best_cnt = p, base, rep, counts[p]
                        continue
                    if base + eps < best_base:
                        best_p, best_base, best_rep, best_cnt = p, base, rep, counts[p]
                    elif abs(base - best_base) <= eps:
                        if rep < best_rep or (rep == best_rep and counts[p] < best_cnt):
                            best_p, best_base, best_rep, best_cnt = p, base, rep, counts[p]

                pack_index_row[g] = best_p
                rank_row[g] = counts[best_p]
                counts[best_p] += 1
                loads[best_p] += wv
                label_counts[best_p][lab] += 1
                pack_groups[best_p].append(g)
                assigned[g] = True

        # Bounded refinement: tiny local search among top-2 heavy packs and top-2 light packs
        def _refine_once() -> bool:
            order_asc = sorted(range(num_packs), key=lambda p: loads[p])
            order_desc = list(reversed(order_asc))
            h1 = order_desc[0]
            l1 = order_asc[0] if order_asc[0] != h1 else (order_asc[1] if len(order_asc) > 1 else order_asc[0])
            pairs = [(h1, l1)]
            if len(order_asc) >= 3:
                l2 = next(p for p in order_asc if p != h1 and p != l1)
                pairs.append((h1, l2))
            if len(order_desc) >= 3:
                h2 = next(p for p in order_desc if p != h1)
                pairs.append((h2, l1))

            cur_max = max(loads)

            best1 = None  # (new_peak, new_imb, penalty, d, r, ai, bi, a_item, b_item, wa, wb, la, lb)
            for (d, r) in pairs:
                if d == r:
                    continue
                if not pack_groups[d] or not pack_groups[r]:
                    continue
                if loads[d] <= loads[r]:
                    continue
                d_idx_t = torch.tensor(pack_groups[d], dtype=torch.int64, device=device)
                r_idx_t = torch.tensor(pack_groups[r], dtype=torch.int64, device=device)
                d_w = row_w[d_idx_t]
                r_w = row_w[r_idx_t]
                kd = min(2, d_w.numel())
                kr = min(2, r_w.numel())
                if kd == 0 or kr == 0:
                    continue
                d_top = torch.topk(d_w, kd).indices.tolist()
                r_bot = torch.topk(r_w, kr, largest=False).indices.tolist()

                other_max = max([loads[p] for p in range(num_packs) if p != d and p != r], default=float("-inf"))
                other_min = min([loads[p] for p in range(num_packs) if p != d and p != r], default=float("inf"))

                for ai in d_top:
                    wa = float(d_w[ai].item())
                    a_item = int(d_idx_t[ai].item())
                    la = int(labels[a_item].item()) if labels is not None else -1
                    for bi in r_bot:
                        wb = float(r_w[bi].item())
                        b_item = int(r_idx_t[bi].item())
                        lb = int(labels[b_item].item()) if labels is not None else -1
                        new_d = loads[d] - wa + wb
                        new_r = loads[r] - wb + wa
                        new_peak = max(new_d, new_r, other_max)
                        new_bottom = min(new_d, new_r, other_min)
                        new_imb = new_peak - new_bottom
                        penalty = 0
                        if label_counts is not None:
                            penalty += 1 if label_counts[d].get(lb, 0) > 0 else 0
                            penalty += 1 if label_counts[r].get(la, 0) > 0 else 0
                        cand = (new_peak, new_imb, penalty, d, r, ai, bi, a_item, b_item, wa, wb, la, lb)
                        if best1 is None:
                            best1 = cand
                        else:
                            if (cand[0] + 1e-9 < best1[0] or
                                (abs(cand[0] - best1[0]) <= 1e-9 and (cand[1] + 1e-9 < best1[1] or
                                 (abs(cand[1] - best1[1]) <= 1e-9 and cand[2] < best1[2])))):
                                best1 = cand

            # Optional 2x2 fallback when helpful on the (heaviest, lightest) pair
            best2 = None
            if pack_groups[h1] and pack_groups[l1]:
                h_idx_t = torch.tensor(pack_groups[h1], dtype=torch.int64, device=device)
                l_idx_t = torch.tensor(pack_groups[l1], dtype=torch.int64, device=device)
                h_w = row_w[h_idx_t]
                l_w = row_w[l_idx_t]
                if h_w.numel() >= 2 and l_w.numel() >= 2:
                    ai, aj = torch.topk(h_w, 2).indices.tolist()
                    bi, bj = torch.topk(l_w, 2, largest=False).indices.tolist()
                    wa_sum = float(h_w[ai].item() + h_w[aj].item())
                    wb_sum = float(l_w[bi].item() + l_w[bj].item())
                    a_items = (int(h_idx_t[ai].item()), int(h_idx_t[aj].item()))
                    b_items = (int(l_idx_t[bi].item()), int(l_idx_t[bj].item()))
                    other_max = max([loads[p] for p in range(num_packs) if p != h1 and p != l1], default=float("-inf"))
                    other_min = min([loads[p] for p in range(num_packs) if p != h1 and p != l1], default=float("inf"))
                    new_h = loads[h1] - wa_sum + wb_sum
                    new_l = loads[l1] - wb_sum + wa_sum
                    new_peak2 = max(new_h, new_l, other_max)
                    new_bottom2 = min(new_h, new_l, other_min)
                    new_imb2 = new_peak2 - new_bottom2
                    penalty2 = 0
                    if label_counts is not None:
                        la_i = int(labels[a_items[0]].item())
                        la_j = int(labels[a_items[1]].item())
                        lb_i = int(labels[b_items[0]].item())
                        lb_j = int(labels[b_items[1]].item())
                        penalty2 += (1 if label_counts[h1].get(lb_i, 0) > 0 else 0)
                        penalty2 += (1 if label_counts[h1].get(lb_j, 0) > 0 else 0)
                        penalty2 += (1 if label_counts[l1].get(la_i, 0) > 0 else 0)
                        penalty2 += (1 if label_counts[l1].get(la_j, 0) > 0 else 0)
                    best2 = (new_peak2, new_imb2, h1, l1, ai, aj, bi, bj, a_items, b_items, wa_sum, wb_sum, penalty2)

            applied = False
            if best1 is not None and best1[0] + 1e-9 < cur_max:
                _, _, _, d, r, ai, bi, a_item, b_item, wa, wb, la, lb = best1
                loads[d] = loads[d] - wa + wb
                loads[r] = loads[r] - wb + wa
                pack_groups[d][ai] = b_item
                pack_groups[r][bi] = a_item
                pack_index_row[a_item] = r
                pack_index_row[b_item] = d
                for rr, gg in enumerate(pack_groups[d]):
                    rank_row[gg] = rr
                for rr, gg in enumerate(pack_groups[r]):
                    rank_row[gg] = rr
                if label_counts is not None:
                    label_counts[d][la] -= 1
                    if label_counts[d][la] == 0:
                        del label_counts[d][la]
                    label_counts[d][lb] = label_counts[d].get(lb, 0) + 1
                    label_counts[r][lb] -= 1
                    if label_counts[r][lb] == 0:
                        del label_counts[r][lb]
                    label_counts[r][la] = label_counts[r].get(la, 0) + 1
                applied = True
            elif best2 is not None and (best1 is None or best2[0] + 1e-9 < (best1[0] if best1 else cur_max)):
                new_peak2, _, h_sel, l_sel, ai, aj, bi, bj, a_items, b_items, wa_sum, wb_sum, _ = best2
                if new_peak2 + 1e-9 < cur_max:
                    loads[h_sel] = loads[h_sel] - wa_sum + wb_sum
                    loads[l_sel] = loads[l_sel] - wb_sum + wa_sum
                    pack_groups[h_sel][ai] = b_items[0]
                    pack_groups[h_sel][aj] = b_items[1]
                    pack_groups[l_sel][bi] = a_items[0]
                    pack_groups[l_sel][bj] = a_items[1]
                    pack_index_row[a_items[0]] = l_sel
                    pack_index_row[a_items[1]] = l_sel
                    pack_index_row[b_items[0]] = h_sel
                    pack_index_row[b_items[1]] = h_sel
                    for rr, gg in enumerate(pack_groups[h_sel]):
                        rank_row[gg] = rr
                    for rr, gg in enumerate(pack_groups[l_sel]):
                        rank_row[gg] = rr
                    applied = True
            return applied

        # Adaptive refinement depth with early-exit
        mean_load = sum(loads) / max(1, num_packs)
        delta = max(loads) - min(loads)
        rel = (delta / max(mean_load, 1e-12)) if mean_load > 0 else 0.0
        # GPU layer tends to benefit from extra pass at higher residual; groups layer will set low rel
        if rel > 0.12:
            refine_steps = 3
        elif rel > 0.03 if self.with_diversity else rel > 0.05:
            refine_steps = 2
        else:
            refine_steps = 1

        for _ in range(refine_steps):
            changed = _refine_once()
            if not changed:
                break

        return pack_index_row, rank_row

    def pack(
        self,
        weights: torch.Tensor,
        num_items_per_pack: int,
        labels: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        weights: [R, N], labels: [R, N] or None
        Returns pack_index [R, N], rank_in_pack [R, N]
        """
        R, N = weights.shape
        assert N % self.num_packs == 0
        cap = N // self.num_packs

        pack_index = torch.full((R, N), -1, dtype=torch.int64, device=weights.device)
        rank_in_pack = torch.full_like(pack_index, -1)

        for i in range(R):
            row_labels = labels[i] if (labels is not None) else None
            pi, ri = self._pack_row(weights[i], cap, row_labels)
            pack_index[i] = pi
            rank_in_pack[i] = ri
        return pack_index, rank_in_pack


# -------------------------- Hierarchical planner --------------------------

def _pack_groups_to_nodes(tokens_per_group: torch.Tensor, num_nodes: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Light capacity packing from groups -> nodes using CapacityPacker without diversity.
    Adds a small refinement to reduce inter-node imbalance.
    Returns (pack_index, rank_in_pack)
    """
    packer = CapacityPacker(num_packs=num_nodes, with_diversity=False)
    # groups_per_node is implicit via N / num_nodes inside packer
    pack_index, rank_in_pack = packer.pack(tokens_per_group, tokens_per_group.shape[-1] // num_nodes, labels=None)
    return pack_index, rank_in_pack


def rebalance_experts_hierarchical(
    weight: torch.Tensor,
    num_physical_experts: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
):
    """
    Parameters:
        weight: [num_moe_layers, num_logical_experts]
        num_physical_experts: total physical experts (replicas)
        num_groups: expert groups
        num_nodes: server nodes
        num_gpus: total GPUs, must be a multiple of num_nodes

    Returns:
        physical_to_logical_map: [layers, num_physical_experts]
        logical_to_physical_map: [layers, num_logical_experts, X]
        logical_count: [layers, num_logical_experts]
    """
    num_layers, num_logical_experts = weight.shape
    assert num_logical_experts % num_groups == 0
    group_size = num_logical_experts // num_groups
    assert num_groups % num_nodes == 0
    groups_per_node = num_groups // num_nodes
    assert num_gpus % num_nodes == 0
    assert num_physical_experts % num_gpus == 0
    phy_experts_per_gpu = num_physical_experts // num_gpus

    # Step 1: pack groups to nodes
    tokens_per_group = weight.unflatten(-1, (num_groups, group_size)).sum(-1)  # [L, G]
    group_pack_index, group_rank_in_pack = _pack_groups_to_nodes(tokens_per_group, num_nodes)

    # Build meta-logical (mlog) layout inside nodes (contiguous groups per node)
    # log2mlog maps from global logical to per-node logical ordering
    log2mlog = (((group_pack_index * groups_per_node + group_rank_in_pack) * group_size).unsqueeze(-1) +
                torch.arange(group_size, dtype=torch.int64, device=weight.device)).flatten(-2)
    mlog2log = _inverse_rows(log2mlog)

    # Step 2: replicate within nodes using ReplicaAllocator
    tokens_per_mlog = weight.gather(-1, mlog2log).view(-1, num_logical_experts // num_nodes)
    allocator = ReplicaAllocator()
    phy2mlog, phyrank, mlogcnt = allocator.allocate(tokens_per_mlog, num_physical_experts // num_nodes)

    # Step 3: pack physical experts to GPUs inside nodes with diversity awareness
    # Each physical expert weight is its per-replica load: tokens_per_mlog / mlogcnt at phy2mlog indices
    tokens_per_phy = (tokens_per_mlog / mlogcnt).gather(-1, phy2mlog)
    gpus_per_node = num_gpus // num_nodes
    gpu_packer = CapacityPacker(num_packs=gpus_per_node, with_diversity=True)
    pack_index, rank_in_pack = gpu_packer.pack(tokens_per_phy, tokens_per_phy.shape[-1] // gpus_per_node, labels=phy2mlog)

    # Convert to global physical indices inside node
    phy2pphy = _build_blocked_map(pack_index, rank_in_pack, phy_experts_per_gpu)
    pphy2phy = _inverse_rows(phy2pphy)

    # Back to global logical indexing
    pphy2mlog = phy2mlog.gather(-1, pphy2phy)  # [L * N, phy_per_node]
    pphy2mlog = (pphy2mlog.view(num_layers, num_nodes, -1) +
                 torch.arange(0, num_logical_experts, num_logical_experts // num_nodes, device=weight.device).view(1, -1, 1)).flatten(-2)
    pphy2log = mlog2log.gather(-1, pphy2mlog)
    pphyrank = phyrank.gather(-1, pphy2phy).view(num_layers, -1)
    logcnt = mlogcnt.view(num_layers, -1).gather(-1, log2mlog)
    return pphy2log, pphyrank, logcnt


def rebalance_experts(
    weight: torch.Tensor,
    num_replicas: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Entry point for expert-parallelism load balancer.

    Parameters:
        weight: [layers, num_logical_experts], float-like
        num_replicas: number of physical experts, multiple of num_gpus
        num_groups: number of expert groups
        num_nodes: server nodes (fast intra-node interconnect)
        num_gpus: number of GPUs, multiple of num_nodes

    Returns:
        physical_to_logical_map: [layers, num_replicas]
        logical_to_physical_map: [layers, num_logical_experts, X]
        expert_count: [layers, num_logical_experts]
    """
    num_layers, num_logical_experts = weight.shape
    weight = weight.float().cpu()

    if num_groups % num_nodes == 0:
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, num_groups, num_nodes, num_gpus
        )
    else:
        # Fallback: treat entire space as a single node
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, 1, 1, num_gpus
        )

    # Build logical->physical scatter buffer
    num_redundant_experts = num_replicas - num_logical_experts
    maxlogcnt = num_redundant_experts + 1
    log2phy: torch.Tensor = torch.full(
        (num_layers, num_logical_experts, maxlogcnt),
        -1,
        dtype=torch.int64,
        device=logcnt.device,
    )
    # scatter physical indices into logical->physical map
    # Index formula: offset = phy2log * maxlogcnt + phyrank
    log2phy.view(num_layers, -1).scatter_(
        -1,
        phy2log * maxlogcnt + phyrank,
        torch.arange(num_replicas, dtype=torch.int64, device=log2phy.device).expand(num_layers, -1),
    )
    return phy2log, log2phy, logcnt


# EVOLVE-BLOCK-END


# This part remains fixed (not evolved)
def run_eplb(weight: torch.Tensor, num_replicas: int, num_groups: int,
             num_nodes: int, num_gpus: int):
    """Run the expert parallelism load balancer"""
    phy2log, log2phy, logcnt = rebalance_experts(
        weight, num_replicas, num_groups, num_nodes, num_gpus
    )
    return phy2log, log2phy, logcnt


__all__ = ["rebalance_experts", "run_eplb"]