# EVOLVE-BLOCK-START
"""
Expert parallelism load balancer (EPLB) for vLLM.

This module implements the core rearrangement algorithm.

This version uses:
- Peak-aware hybrid water-filling replica allocation with an adaptive tail and
  per-step D’Hondt vs Sainte–Laguë chooser to minimize the max per-replica load.
- A small, guarded donor→receiver fix-up to further smooth peaks.
- Diversity-aware GPU packing with label-spreading seed and bounded refinement.
"""

import torch
from collections import defaultdict


def balanced_packing(weight: torch.Tensor,
                     num_packs: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pack n weighted objects to m packs, such that each bin contains exactly
    n/m objects and the weights of all packs are as balanced as possible.

    Parameters:
        weight: [X, n], the weight of each item
        num_packs: number of packs

    Returns:
        pack_index: [X, n], the pack index of each item
        rank_in_pack: [X, n], the rank of the item in the pack
    """
    num_layers, num_groups = weight.shape
    assert num_groups % num_packs == 0
    groups_per_pack = num_groups // num_packs

    if groups_per_pack == 1:
        pack_index = torch.arange(num_groups,
                                  dtype=torch.int64,
                                  device=weight.device).expand(num_layers, num_groups)
        rank_in_pack = torch.zeros_like(pack_index, dtype=torch.int64)
        return pack_index, rank_in_pack

    # Longest-processing-time greedy with capacity constraints and count-aware tie-breaking
    indices = weight.float().sort(-1, descending=True).indices.cpu()
    pack_index = torch.full((num_layers, num_groups),
                             fill_value=-1,
                             dtype=torch.int64,
                             device="cpu")
    rank_in_pack = torch.full_like(pack_index, fill_value=-1)
    for i in range(num_layers):
        pack_loads = [0.0] * num_packs
        pack_counts = [0] * num_packs
        for g in indices[i].tolist():
            # choose pack with capacity and minimal load; tie-break by fewer items
            best_p = None
            best_load = None
            best_cnt = None
            for p in range(num_packs):
                if pack_counts[p] >= groups_per_pack:
                    continue
                if (best_load is None or pack_loads[p] < best_load or
                    (abs(pack_loads[p] - (best_load if best_load is not None else 0.0)) <= 1e-12 and
                     (best_cnt is None or pack_counts[p] < best_cnt))):
                    best_p = p
                    best_load = pack_loads[p]
                    best_cnt = pack_counts[p]
            pack_index[i, g] = best_p
            rank_in_pack[i, g] = pack_counts[best_p]
            pack_counts[best_p] += 1
            pack_loads[best_p] += float(weight[i, g].item())
    return pack_index, rank_in_pack


def _waterfill_counts_row(w: torch.Tensor, target_total: int) -> torch.Tensor:
    """
    Compute integer replica counts c_i >= 1 that approximately minimize max_i w_i / c_i
    subject to sum c_i == target_total using water-filling + greedy fill.

    Parameters:
        w: [num_log], float tensor (on CPU)
        target_total: int, total replicas to allocate

    Returns:
        counts: [num_log], int64
    """
    num_log = w.numel()
    assert target_total >= num_log  # at least one per expert

    if target_total == num_log:
        return torch.ones(num_log, dtype=torch.int64, device=w.device)

    maxw = float(w.max().item()) if num_log > 0 else 0.0
    # Handle all-zero quickly
    if maxw == 0.0:
        counts = torch.ones(num_log, dtype=torch.int64, device=w.device)
        extras = target_total - num_log
        if extras > 0 and num_log > 0:
            base_add = extras // num_log
            rem = extras % num_log
            if base_add > 0:
                counts += base_add
            if rem > 0:
                counts[:rem] += 1
        return counts

    # Binary search T such that sum ceil(w_i / T) <= target_total, with c_i >= 1
    lo = 0.0
    hi = max(maxw, 1.0)
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        c = torch.ceil(w / mid).to(torch.int64)
        c = torch.maximum(c, torch.ones_like(c))
        s = int(c.sum().item())
        if s <= target_total:
            hi = mid
        else:
            lo = mid

    # Base counts from hi guarantee <= target_total
    counts = torch.ceil(w / hi).to(torch.int64)
    counts = torch.maximum(counts, torch.ones_like(counts))
    s = int(counts.sum().item())

    extras = target_total - s
    if extras <= 0:
        return counts

    # Greedy water-filling with peak-aware hybrid tail
    num_redundant = max(0, target_total - num_log)
    counts_f = counts.to(w.dtype)
    avg = w / counts_f
    mean_avg = float(w.sum().item() / max(1, target_total))
    cur_peak = float(avg.max().item())
    ratio = (cur_peak / max(mean_avg, 1e-12) - 1.0) if mean_avg > 0 else 0.0
    ratio = max(0.0, min(1.0, ratio))
    tail = min(extras, max(1, int(round(0.1 * num_redundant * ratio))))
    bulk = extras - tail

    # Fast batched fill for bulk (D'Hondt)
    while bulk > 0:
        k = min(bulk, num_log)
        scores = w / counts.to(w.dtype)
        topk_idx = torch.topk(scores, k=k, largest=True).indices
        counts[topk_idx] += 1
        bulk -= k

    # Adaptive tail: at each step evaluate D’Hondt vs Sainte–Laguë and pick the one
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

        # D'Hondt pick (argmax w/c == argmax avg)
        d_pick = int(torch.argmax(avg).item())
        # Sainte–Laguë pick (argmax w/(2c+1))
        denom = (2 * counts + 1).to(w.dtype)
        s_scores = w / denom
        s_pick = int(torch.argmax(s_scores).item())

        best_choice = None
        best_key = None  # (new_peak, new_second, receiver_cnt, -new_avg)
        for pick in (d_pick, s_pick):
            if best_choice is not None and pick == best_choice:
                continue
            new_avg_chosen = float((w[pick] / (counts_f[pick] + 1.0)).item())
            # Predict new peak and second-highest after this assignment
            if pick == peak1_idx:
                new_peak = max(new_avg_chosen, peak2_val)
                new_second = min(new_avg_chosen, peak2_val)
            else:
                new_peak = max(peak1_val, new_avg_chosen)
                new_second = (peak1_val if new_avg_chosen >= peak1_val else max(new_avg_chosen, peak2_val))
            receiver_cnt = int(counts[pick].item())
            key = (new_peak, new_second, receiver_cnt, -new_avg_chosen)
            if best_key is None or key < best_key:
                best_key = key
                best_choice = pick

        counts[best_choice] += 1

    return counts


def replicate_experts_waterfill(
    weight: torch.Tensor,
    num_phy: int
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Water-filling replication to minimize the maximum per-replica load,
    with a small iterative donor→receiver fix-up to reduce peak average load.

    Parameters:
        weight: [X, num_log] (CPU float)
        num_phy: total number of replicas

    Returns:
        phy2log: [X, num_phy], logical expert id for each physical expert
        rank:    [X, num_phy], replica rank per logical expert
        logcnt:  [X, num_log], replica counts per logical expert
    """
    n, num_log = weight.shape
    assert num_phy >= num_log
    device = weight.device

    phy2log_list = []
    rank_list = []
    logcnt_list = []

    exp_ids = torch.arange(num_log, dtype=torch.int64, device=device)
    for i in range(n):
        w = weight[i]  # [num_log], float CPU
        counts = _waterfill_counts_row(w, num_phy)  # int64

        # Iterative small fix-up: up to 2 best donor->receiver moves (peak-aware with second-order tie-breakers)
        if num_log > 1:
            max_iters = 2
            for _ in range(max_iters):
                counts_safe = torch.clamp(counts, min=1)
                avg = w / counts_safe.to(w.dtype)
                cur_peak = float(avg.max().item())

                # donors: top-2 by avg with count>1
                can_donate = (counts > 1)
                if not bool(can_donate.any()):
                    break
                kd = int(min(2, int(can_donate.sum().item())))
                avg_mask = avg.clone()
                avg_mask[~can_donate] = float("-inf")
                donors = torch.topk(avg_mask, k=kd).indices.tolist()

                # receivers: bottom-2 by avg
                kr = int(min(2, num_log))
                receivers = torch.topk(-avg, k=kr).indices.tolist()

                best_pair = None
                best_key = None  # (new_peak, new_second_peak, donor_post_avg)
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
                        # second-highest after the move
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

        logcnt_list.append(counts)

        # Build phy2log and rank (contiguous blocks per logical expert)
        phy2log_i = torch.repeat_interleave(exp_ids, counts)
        # ranks: 0..count-1 for each expert
        starts = torch.cumsum(counts, dim=0) - counts
        arange_phy = torch.arange(num_phy, dtype=torch.int64, device=device)
        rank_i = arange_phy - torch.repeat_interleave(starts, counts)

        phy2log_list.append(phy2log_i)
        rank_list.append(rank_i)

    phy2log = torch.stack(phy2log_list, dim=0)
    rank = torch.stack(rank_list, dim=0)
    logcnt = torch.stack(logcnt_list, dim=0)
    return phy2log, rank, logcnt


def pack_diverse_heap(
    weights: torch.Tensor,
    labels: torch.Tensor,
    num_packs: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Diversity-aware greedy packing with exact capacity per pack:
    - Fast path when labels are unique (no duplicate labels).
    - Seed phase: for duplicated labels, place one heaviest item per label first,
      spreading across packs that don't yet contain that label to avoid hotspots.
    - Greedy fill with projected-load tie-broken by label repeat and count.
    - Tiny bounded refinement (1x1 prioritized, 2x2 fallback) to reduce global max.
    """
    num_layers, n_items = weights.shape
    assert n_items % num_packs == 0
    cap = n_items // num_packs

    if cap == 1 or num_packs == 1:
        # Each pack gets exactly one item
        idx = torch.arange(n_items, dtype=torch.int64, device=weights.device)
        pack_index = (idx % num_packs).expand(num_layers, n_items).clone()
        rank_in_pack = torch.zeros_like(pack_index, dtype=torch.int64)
        return pack_index, rank_in_pack

    pack_index = torch.full((num_layers, n_items), -1, dtype=torch.int64, device=weights.device)
    rank_in_pack = torch.full_like(pack_index, -1)

    sorted_idx_all = weights.sort(dim=-1, descending=True).indices  # [X, n]

    for i in range(num_layers):
        row_w = weights[i]
        row_labels = labels[i]
        sorted_idx = sorted_idx_all[i].tolist()

        loads = [0.0] * num_packs
        counts = [0] * num_packs
        label_counts = [defaultdict(int) for _ in range(num_packs)]
        pack_groups = [[] for _ in range(num_packs)]
        assigned = [False] * n_items

        # Duplicate label fast check
        lbl_sorted, _ = torch.sort(row_labels)
        all_unique = (torch.unique_consecutive(lbl_sorted).numel() == row_labels.numel())

        if all_unique:
            # Fast greedy: choose pack with minimal load (tie: fewer items)
            for g in sorted_idx:
                wv = float(row_w[g].item())
                best_p = None
                best_load = None
                best_cnt = None
                for p in range(num_packs):
                    if counts[p] >= cap:
                        continue
                    if (best_load is None or loads[p] < best_load or
                        (abs(loads[p] - best_load) <= 1e-12 and counts[p] < best_cnt)):
                        best_p = p
                        best_load = loads[p]
                        best_cnt = counts[p]
                pack_index[i, g] = best_p
                rank_in_pack[i, g] = counts[best_p]
                counts[best_p] += 1
                loads[best_p] += wv
                pack_groups[best_p].append(g)
        else:
            # Phase 1: label-spreading seed (one heaviest item per label)
            label_items = defaultdict(list)  # lab -> [g_heaviest_first...]
            for g in sorted_idx:
                lab = int(row_labels[g].item())
                label_items[lab].append(g)

            # Sort labels by their top item's weight (heaviest first)
            label_order = sorted(
                label_items.keys(),
                key=lambda lab: float(row_w[label_items[lab][0]].item()),
                reverse=True,
            )

            # Place one item per label, preferring packs without that label and min load
            for lab in label_order:
                if counts.count(cap) == num_packs:
                    break  # all packs full
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
                        (abs(loads[p] - best_load) <= 1e-12 and counts[p] < best_cnt)):
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
                    pack_index[i, g] = best_p
                    rank_in_pack[i, g] = counts[best_p]
                    counts[best_p] += 1
                    loads[best_p] += wv
                    label_counts[best_p][lab] += 1
                    pack_groups[best_p].append(g)
                    assigned[g] = True

            # Phase 2: diversity-aware greedy fill for remaining items
            eps = 1e-6 * float(row_w.mean().item() if row_w.numel() > 0 else 1.0)
            for g in sorted_idx:
                if assigned[g]:
                    continue
                lab = int(row_labels[g].item())
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
                        best_p = p
                        best_base = base
                        best_rep = rep
                        best_cnt = counts[p]
                        continue

                    if base + eps < best_base:
                        best_p = p
                        best_base = base
                        best_rep = rep
                        best_cnt = counts[p]
                    elif abs(base - best_base) <= eps:
                        if rep < best_rep or (rep == best_rep and counts[p] < best_cnt):
                            best_p = p
                            best_base = base
                            best_rep = rep
                            best_cnt = counts[p]

                pack_index[i, g] = best_p
                rank_in_pack[i, g] = counts[best_p]
                counts[best_p] += 1
                loads[best_p] += wv
                label_counts[best_p][lab] += 1
                pack_groups[best_p].append(g)
                assigned[g] = True

        # Bounded refinement to reduce global max
        if num_packs >= 2:
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
                    if d == r or not pack_groups[d] or not pack_groups[r] or loads[d] <= loads[r]:
                        continue
                    d_idx_tensor = torch.tensor(pack_groups[d], dtype=torch.int64, device=row_w.device)
                    r_idx_tensor = torch.tensor(pack_groups[r], dtype=torch.int64, device=row_w.device)
                    d_w = row_w[d_idx_tensor]
                    r_w = row_w[r_idx_tensor]
                    kd = min(2, d_w.numel())
                    kr = min(2, r_w.numel())
                    if kd == 0 or kr == 0:
                        continue
                    d_top = torch.topk(d_w, kd).indices.tolist()
                    r_bot = torch.topk(r_w, kr, largest=False).indices.tolist()

                    other_max = max([loads[p] for p in range(num_packs) if p != d and p != r], default=float("-inf"))
                    other_min = min([loads[p] for p in range(num_packs) if p != d and p != r], default=float("inf"))

                    # Build label counts maps (small) for the two packs
                    lcnt_d = defaultdict(int)
                    lcnt_r = defaultdict(int)
                    for gg in pack_groups[d]:
                        lcnt_d[int(row_labels[gg].item())] += 1
                    for gg in pack_groups[r]:
                        lcnt_r[int(row_labels[gg].item())] += 1

                    for ai in d_top:
                        wa = float(d_w[ai].item())
                        a_item = int(d_idx_tensor[ai].item())
                        la = int(row_labels[a_item].item())
                        for bi in r_bot:
                            wb = float(r_w[bi].item())
                            b_item = int(r_idx_tensor[bi].item())
                            lb = int(row_labels[b_item].item())
                            new_d = loads[d] - wa + wb
                            new_r = loads[r] - wb + wa
                            new_peak = max(new_d, new_r, other_max)
                            new_bottom = min(new_d, new_r, other_min)
                            new_imb = new_peak - new_bottom
                            penalty = (1 if lcnt_d.get(lb, 0) > 0 else 0) + (1 if lcnt_r.get(la, 0) > 0 else 0)
                            cand = (new_peak, new_imb, penalty, d, r, ai, bi, a_item, b_item, wa, wb, la, lb)
                            if best1 is None:
                                best1 = cand
                            else:
                                if (cand[0] + 1e-9 < best1[0] or
                                    (abs(cand[0] - best1[0]) <= 1e-9 and (cand[1] + 1e-9 < best1[1] or
                                     (abs(cand[1] - best1[1]) <= 1e-9 and cand[2] < best1[2])))):
                                    best1 = cand

                # Optional 2x2 bounded swap only for (heaviest↔lightest)
                best2 = None
                if pack_groups[h1] and pack_groups[l1]:
                    h_idx_tensor = torch.tensor(pack_groups[h1], dtype=torch.int64, device=row_w.device)
                    l_idx_tensor = torch.tensor(pack_groups[l1], dtype=torch.int64, device=row_w.device)
                    h_w = row_w[h_idx_tensor]
                    l_w = row_w[l_idx_tensor]
                    if h_w.numel() >= 2 and l_w.numel() >= 2:
                        ai, aj = torch.topk(h_w, 2).indices.tolist()
                        bi, bj = torch.topk(l_w, 2, largest=False).indices.tolist()
                        wa_sum = float(h_w[ai].item() + h_w[aj].item())
                        wb_sum = float(l_w[bi].item() + l_w[bj].item())
                        a_items = (int(h_idx_tensor[ai].item()), int(h_idx_tensor[aj].item()))
                        b_items = (int(l_idx_tensor[bi].item()), int(l_idx_tensor[bj].item()))
                        other_max = max([loads[p] for p in range(num_packs) if p != h1 and p != l1], default=float("-inf"))
                        other_min = min([loads[p] for p in range(num_packs) if p != h1 and p != l1], default=float("inf"))
                        new_h = loads[h1] - wa_sum + wb_sum
                        new_l = loads[l1] - wb_sum + wa_sum
                        new_peak2 = max(new_h, new_l, other_max)
                        new_bottom2 = min(new_h, new_l, other_min)
                        new_imb2 = new_peak2 - new_bottom2
                        # light label penalty check for ties
                        penalty2 = 0
                        la_i = int(row_labels[a_items[0]].item())
                        la_j = int(row_labels[a_items[1]].item())
                        lb_i = int(row_labels[b_items[0]].item())
                        lb_j = int(row_labels[b_items[1]].item())
                        penalty2 += 1 if any(int(row_labels[g].item()) == lb_i for g in pack_groups[h1]) else 0
                        penalty2 += 1 if any(int(row_labels[g].item()) == lb_j for g in pack_groups[h1]) else 0
                        penalty2 += 1 if any(int(row_labels[g].item()) == la_i for g in pack_groups[l1]) else 0
                        penalty2 += 1 if any(int(row_labels[g].item()) == la_j for g in pack_groups[l1]) else 0
                        best2 = (new_peak2, new_imb2, penalty2, h1, l1, ai, aj, bi, bj, a_items, b_items, wa_sum, wb_sum)

                applied = False
                if best1 is not None and best1[0] + 1e-9 < cur_max:
                    new_peak, new_imb, _, d, r, ai, bi, a_item, b_item, wa, wb, la, lb = best1
                    loads[d] = loads[d] - wa + wb
                    loads[r] = loads[r] - wb + wa
                    # swap membership
                    pack_groups[d][ai] = b_item
                    pack_groups[r][bi] = a_item
                    # update indices
                    pack_index[i, a_item] = r
                    pack_index[i, b_item] = d
                    # update ranks (only two packs touched)
                    for rr, gg in enumerate(pack_groups[d]):
                        rank_in_pack[i, gg] = rr
                    for rr, gg in enumerate(pack_groups[r]):
                        rank_in_pack[i, gg] = rr
                    applied = True
                elif best2 is not None:
                    new_peak2, _, _, h_sel, l_sel, ai, aj, bi, bj, a_items, b_items, wa_sum, wb_sum = best2
                    if new_peak2 + 1e-9 < cur_max:
                        loads[h_sel] = loads[h_sel] - wa_sum + wb_sum
                        loads[l_sel] = loads[l_sel] - wb_sum + wa_sum
                        # swap membership
                        pack_groups[h_sel][ai] = b_items[0]
                        pack_groups[h_sel][aj] = b_items[1]
                        pack_groups[l_sel][bi] = a_items[0]
                        pack_groups[l_sel][bj] = a_items[1]
                        # update indices
                        pack_index[i, a_items[0]] = l_sel
                        pack_index[i, a_items[1]] = l_sel
                        pack_index[i, b_items[0]] = h_sel
                        pack_index[i, b_items[1]] = h_sel
                        # update ranks
                        for rr, gg in enumerate(pack_groups[h_sel]):
                            rank_in_pack[i, gg] = rr
                        for rr, gg in enumerate(pack_groups[l_sel]):
                            rank_in_pack[i, gg] = rr
                        applied = True
                return applied

            # Adaptive refinement steps with early exit
            mean_load = sum(loads) / max(1, num_packs)
            delta = max(loads) - min(loads)
            rel = (delta / max(mean_load, 1e-12)) if mean_load > 0 else 0.0
            if rel > 0.12:
                refine_steps = 3
            elif rel > 0.02:
                refine_steps = 2
            else:
                refine_steps = 1
            for _ in range(refine_steps):
                if not _refine_once():
                    break

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
        num_physical_experts: number of physical experts after replication
        num_groups: number of expert groups
        num_nodes: number of server nodes, where the intra-node network
        (e.g, NVLink) is faster
        num_gpus: number of GPUs, must be a multiple of `num_nodes`

    Returns:
        physical_to_logical_map: [num_moe_layers, num_physical_experts]
        logical_to_physical_map: [num_moe_layers, num_logical_experts, X]
        logical_count: [num_moe_layers, num_logical_experts]
    """
    num_layers, num_logical_experts = weight.shape
    assert num_logical_experts % num_groups == 0
    group_size = num_logical_experts // num_groups
    assert num_groups % num_nodes == 0
    groups_per_node = num_groups // num_nodes
    assert num_gpus % num_nodes == 0
    assert num_physical_experts % num_gpus == 0
    phy_experts_per_gpu = num_physical_experts // num_gpus

    def inverse(perm: torch.Tensor) -> torch.Tensor:
        inv = torch.empty_like(perm)
        inv.scatter_(
            1,
            perm,
            torch.arange(perm.size(1), dtype=torch.int64,
                         device=perm.device).expand(perm.shape),
        )
        return inv

    # Step 1: pack groups to nodes (capacity-aware LPT with count-aware tiebreak)
    tokens_per_group = weight.unflatten(-1, (num_groups, group_size)).sum(-1)
    group_pack_index, group_rank_in_pack = balanced_packing(
        tokens_per_group, num_nodes)
    log2mlog = (((group_pack_index * groups_per_node + group_rank_in_pack) *
                 group_size).unsqueeze(-1) +
                torch.arange(group_size,
                             dtype=torch.int64,
                             device=group_pack_index.device)).flatten(-2)
    mlog2log = inverse(log2mlog)

    # Step 2: replicate within nodes using peak-aware water-filling
    tokens_per_mlog = weight.gather(-1, mlog2log).view(
        -1, num_logical_experts // num_nodes)

    phy2mlog, phyrank, mlogcnt = replicate_experts_waterfill(
        tokens_per_mlog, num_physical_experts // num_nodes
    )

    # Step 3: pack physical experts to GPUs in each node with diversity-aware heap
    tokens_per_phy = (tokens_per_mlog / mlogcnt).gather(-1, phy2mlog)
    gpus_per_node = num_gpus // num_nodes
    pack_index, rank_in_pack = pack_diverse_heap(
        tokens_per_phy, phy2mlog, gpus_per_node
    )
    phy2pphy = pack_index * phy_experts_per_gpu + rank_in_pack
    pphy2phy = inverse(phy2pphy)

    pphy2mlog = phy2mlog.gather(-1, pphy2phy)  # [num_layers * num_nodes, num_phy_per_node]
    # convert back to global logical indexing
    pphy2mlog = (pphy2mlog.view(num_layers, num_nodes, -1) + torch.arange(
        0,
        num_logical_experts,
        num_logical_experts // num_nodes,
        device=group_rank_in_pack.device,
    ).view(1, -1, 1)).flatten(-2)
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
        weight: [layers, num_logical_experts], the load statistics for all
            logical experts
        num_replicas: number of physical experts, must be a multiple of
            `num_gpus`
        num_groups: number of expert groups
        num_nodes: number of server nodes, where the intra-node network
            (e.g, NVLink) is faster
        num_gpus: number of GPUs, must be a multiple of `num_nodes`

    Returns:
        physical_to_logical_map: [layers, num_replicas], the expert index of
            each replica
        logical_to_physical_map: [layers, num_logical_experts, X], the replica
            indices for each expert
        expert_count: [layers, num_logical_experts], number of physical
            replicas for each logical expert
    """
    num_layers, num_logical_experts = weight.shape
    weight = weight.float().cpu()
    if num_groups % num_nodes == 0:
        # hierarchical policy
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, num_groups, num_nodes, num_gpus)
    else:
        # global policy (treat as single node)
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, 1, 1, num_gpus)

    num_redundant_experts = num_replicas - num_logical_experts
    maxlogcnt = num_redundant_experts + 1
    log2phy: torch.Tensor = torch.full(
        (num_layers, num_logical_experts, maxlogcnt),
        -1,
        dtype=torch.int64,
        device=logcnt.device,
    )
    # scatter physical indices into logical->physical map
    log2phy.view(num_layers, -1).scatter_(
        -1,
        phy2log * maxlogcnt + phyrank,
        torch.arange(num_replicas, dtype=torch.int64,
                     device=log2phy.device).expand(num_layers, -1),
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