# EVOLVE-BLOCK-START
"""
Expert parallelism load balancer (EPLB) for vLLM.

This module implements the core rearrangement algorithm.

The rearrangement algorithm is adapted from
[DeepSeek EPLB](https://github.com/deepseek-ai/eplb).

Please find at [#12](https://github.com/deepseek-ai/EPLB/issues/12) an example
on how the EPLB algorithm works.
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
        pack_index = torch.arange(weight.size(-1),
                                  dtype=torch.int64,
                                  device=weight.device).expand(weight.shape)
        rank_in_pack = torch.zeros_like(weight, dtype=torch.int64)
        return pack_index, rank_in_pack

    indices = weight.float().sort(-1, descending=True).indices.cpu()
    pack_index = torch.full_like(weight,
                                 fill_value=-1,
                                 dtype=torch.int64,
                                 device="cpu")
    rank_in_pack = torch.full_like(pack_index, fill_value=-1)
    for i in range(num_layers):
        pack_weights = [0] * num_packs
        pack_items = [0] * num_packs
        for group in indices[i]:
            pack = min(
                (i
                 for i in range(num_packs) if pack_items[i] < groups_per_pack),
                key=pack_weights.__getitem__,
            )
            assert pack_items[pack] < groups_per_pack
            pack_index[i, group] = pack
            rank_in_pack[i, group] = pack_items[pack]
            pack_weights[pack] += weight[i, group]
            pack_items[pack] += 1
    return pack_index, rank_in_pack


def balanced_packing_diverse(weight: torch.Tensor,
                             label: torch.Tensor,
                             num_packs: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Balanced packing with diversity: prefer spreading items with the same label
    across different packs to reduce hotspotting when replicas of the same
    logical expert are placed on the same GPU. After the greedy assignment,
    apply a single bounded refinement swap between the heaviest pack and the
    best of the two lightest packs to further reduce peak/imbalance.
    """
    num_layers, num_items = weight.shape
    assert num_items % num_packs == 0
    items_per_pack = num_items // num_packs

    # Trivial case: one item per pack, fallback to standard balanced packing
    if items_per_pack == 1 or num_packs == 1:
        return balanced_packing(weight, num_packs)

    device = weight.device
    w = weight.float()
    labels = label.to(dtype=torch.int64, device=device)

    # Pre-sort indices by descending weights per row
    sorted_indices = w.sort(dim=-1, descending=True).indices

    pack_index = torch.full_like(weight,
                                 fill_value=-1,
                                 dtype=torch.int64,
                                 device=device)
    rank_in_pack = torch.full_like(pack_index, fill_value=-1)

    for i in range(num_layers):
        row_sorted = sorted_indices[i].tolist()
        row_w = w[i]
        row_labels = labels[i]

        # Detect duplicates quickly
        has_dup = torch.unique(row_labels).numel() != row_labels.numel()

        # Greedy assignment
        pack_loads = [0.0] * num_packs
        pack_counts = [0] * num_packs
        label_counts = [defaultdict(int) for _ in range(num_packs)] if has_dup else None

        if not has_dup:
            for group in row_sorted:
                wv = float(row_w[group].item())
                # choose among packs with capacity the one with min load; tie-break by fewer items
                best_p = None
                best_load = None
                best_cnt = None
                for p in range(num_packs):
                    if pack_counts[p] >= items_per_pack:
                        continue
                    if (best_load is None or pack_loads[p] < best_load or
                        (abs(pack_loads[p] - best_load) <= 1e-12 and pack_counts[p] < best_cnt)):
                        best_p = p
                        best_load = pack_loads[p]
                        best_cnt = pack_counts[p]
                pack_index[i, group] = best_p
                rank_in_pack[i, group] = pack_counts[best_p]
                pack_counts[best_p] += 1
                pack_loads[best_p] += wv
        else:
            # Diversity-aware greedy: minimize (projected load, repeats, count)
            eps = 1e-6 * float(row_w.mean().item() if row_w.numel() > 0 else 1.0)
            for group in row_sorted:
                lab = int(row_labels[group].item())
                wv = float(row_w[group].item())
                best_p = None
                best_base = None
                best_rep = None
                best_cnt = None
                for p in range(num_packs):
                    if pack_counts[p] >= items_per_pack:
                        continue
                    base = pack_loads[p] + wv
                    rep = label_counts[p].get(lab, 0)
                    if best_p is None:
                        best_p = p
                        best_base = base
                        best_rep = rep
                        best_cnt = pack_counts[p]
                        continue
                    if base + eps < best_base:
                        best_p = p
                        best_base = base
                        best_rep = rep
                        best_cnt = pack_counts[p]
                    elif abs(base - best_base) <= eps:
                        if rep < best_rep or (rep == best_rep and pack_counts[p] < best_cnt):
                            best_p = p
                            best_base = base
                            best_rep = rep
                            best_cnt = pack_counts[p]
                pack_index[i, group] = best_p
                rank_in_pack[i, group] = pack_counts[best_p]
                pack_counts[best_p] += 1
                pack_loads[best_p] += wv
                label_counts[best_p][lab] += 1  # type: ignore[index]

        # Micro refinement: broaden 1x1 search donors={heaviest, second-heaviest}
        # and receivers={lightest, second-lightest, third-lightest} (excluding donors).
        # Evaluate top-2 vs bottom-2 items and apply the best swap if it strictly reduces global peak
        # (ties by imbalance, then label-duplicate penalty). Only one swap is applied.
        if num_packs >= 2:
            cur_max = max(pack_loads)
            cur_min = min(pack_loads)
            if cur_max > cur_min:
                # Build pack membership lists for this row
                pack_groups = [[] for _ in range(num_packs)]
                for g in range(num_items):
                    p = int(pack_index[i, g])
                    pack_groups[p].append(g)

                order_asc = sorted(range(num_packs), key=lambda k: pack_loads[k])
                order_desc = list(reversed(order_asc))
                donors = []
                if order_desc:
                    donors.append(order_desc[0])
                    if len(order_desc) > 1 and order_desc[1] != order_desc[0]:
                        donors.append(order_desc[1])

                receivers = []
                for p in order_asc:
                    if p not in donors:
                        receivers.append(p)
                    if len(receivers) >= 3:
                        break

                if donors and receivers:
                    # Optional label histograms for penalty when duplicates exist
                    label_hist = None
                    if has_dup:
                        label_hist = [defaultdict(int) for _ in range(num_packs)]
                        for p in range(num_packs):
                            for g in pack_groups[p]:
                                label_hist[p][int(row_labels[g].item())] += 1

                    cur_imb = cur_max - cur_min
                    best = None
                    best_key = None  # (new_peak, new_imb, penalty)

                    # Precompute a per-pack tensor view for fast lookup
                    for d in donors:
                        if not pack_groups[d]:
                            continue
                        d_idx_tensor = torch.tensor(pack_groups[d], dtype=torch.int64, device=row_w.device)
                        d_w = row_w[d_idx_tensor]
                        kd = min(2, d_w.numel())
                        if kd == 0:
                            continue
                        d_top = torch.topk(d_w, kd).indices.tolist()

                        for r in receivers:
                            if not pack_groups[r]:
                                continue
                            if pack_loads[d] <= pack_loads[r]:
                                continue
                            r_idx_tensor = torch.tensor(pack_groups[r], dtype=torch.int64, device=row_w.device)
                            r_w = row_w[r_idx_tensor]
                            kr = min(2, r_w.numel())
                            if kr == 0:
                                continue
                            r_bot = torch.topk(r_w, kr, largest=False).indices.tolist()

                            # Precompute other packs' extrema
                            other_max = max([pack_loads[p] for p in range(num_packs) if p != d and p != r],
                                            default=float("-inf"))
                            other_min = min([pack_loads[p] for p in range(num_packs) if p != d and p != r],
                                            default=float("inf"))

                            for ai in d_top:
                                a_item = int(d_idx_tensor[ai].item())
                                wa = float(d_w[ai].item())
                                la = int(row_labels[a_item].item())
                                for bi in r_bot:
                                    b_item = int(r_idx_tensor[bi].item())
                                    wb = float(r_w[bi].item())
                                    lb = int(row_labels[b_item].item())

                                    new_d = pack_loads[d] - wa + wb
                                    new_r = pack_loads[r] - wb + wa
                                    new_peak = max(new_d, new_r, other_max)
                                    new_bottom = min(new_d, new_r, other_min)
                                    new_imb = new_peak - new_bottom
                                    penalty = 0
                                    if has_dup and label_hist is not None:
                                        penalty += 1 if label_hist[d].get(lb, 0) > 0 else 0
                                        penalty += 1 if label_hist[r].get(la, 0) > 0 else 0
                                    cand_key = (new_peak, new_imb, penalty)
                                    if best_key is None or cand_key < best_key:
                                        best_key = cand_key
                                        best = (d, r, ai, bi, a_item, b_item, wa, wb)

                    if best is not None:
                        new_peak, new_imb, _ = best_key  # type: ignore[misc]
                        # Apply only if it strictly reduces the global peak;
                        # if peak ties, require a strict imbalance improvement
                        if (new_peak + 1e-12 < cur_max) or (abs(new_peak - cur_max) <= 1e-12 and new_imb + 1e-12 < cur_imb):
                            d_sel, r_sel, ai, bi, a_item, b_item, wa, wb = best  # type: ignore[misc]
                            # Update loads
                            pack_loads[d_sel] = pack_loads[d_sel] - wa + wb
                            pack_loads[r_sel] = pack_loads[r_sel] - wb + wa
                            # Swap membership and indices
                            pack_groups[d_sel][ai] = b_item
                            pack_groups[r_sel][bi] = a_item
                            pack_index[i, a_item] = r_sel
                            pack_index[i, b_item] = d_sel
                            # Update ranks for affected packs
                            for rpos, g in enumerate(pack_groups[d_sel]):
                                rank_in_pack[i, g] = rpos
                            for rpos, g in enumerate(pack_groups[r_sel]):
                                rank_in_pack[i, g] = rpos

    return pack_index, rank_in_pack


def _waterfill_counts_row(w: torch.Tensor, target_total: int) -> torch.Tensor:
    """
    Compute integer replica counts c_i >= 1 that approximately minimize max_i w_i / c_i
    subject to sum c_i == target_total using water-filling + greedy fill.
    w: 1D float tensor (CPU)
    """
    num_log = w.numel()
    assert target_total >= num_log
    if target_total == num_log:
        return torch.ones(num_log, dtype=torch.int64, device=w.device)

    maxw = float(w.max().item()) if num_log > 0 else 0.0
    # All-zero quick path: distribute evenly
    if maxw == 0.0:
        base = target_total // num_log
        rem = target_total % num_log
        counts = torch.full((num_log,), base, dtype=torch.int64, device=w.device)
        if base == 0:
            counts[:] = 1
            extras = target_total - num_log
            if extras > 0:
                counts[:extras] += 1
        else:
            if rem > 0:
                counts[:rem] += 1
        return counts

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
    counts = torch.ceil(w / hi).to(torch.int64)
    counts = torch.maximum(counts, torch.ones_like(counts))
    s = int(counts.sum().item())

    # Peak-aware hybrid tail allocation:
    # - Do a fast bulk D'Hondt-like fill for the first extras - T seats
    # - For the last T seats, choose between D’Hondt and Sainte–Laguë picks
    #   by predicting the new peak average and selecting the lower one.
    extras = target_total - s
    if extras <= 0:
        return counts

    # Tail length: 10% of total redundant seats, capped by remaining extras
    total_redundant = target_total - num_log
    tail_T = max(1, int(round(0.1 * max(0, total_redundant))))
    tail = min(tail_T, extras)
    bulk = extras - tail

    # Fast bulk: add in k-chunks by highest w / c
    while bulk > 0:
        k = min(bulk, num_log)
        score = w / counts.to(w.dtype)
        idx = torch.argsort(score, descending=True)[:k]
        counts[idx] += 1
        bulk -= k

    # Recompute remaining as tail
    remaining = target_total - int(counts.sum().item())
    if remaining <= 0:
        return counts

    # Tail: hybrid D’Hondt vs Sainte–Laguë, peak-aware
    for _ in range(remaining):
        if num_log == 1:
            counts[0] += 1
            continue

        cfloat = counts.to(w.dtype)
        avg = w / torch.clamp(cfloat, min=1)

        # D’Hondt and Sainte–Laguë candidate indices
        d_scores = w / (cfloat + 1.0)
        s_scores = w / (2.0 * cfloat + 1.0)
        d_idx = int(torch.argmax(d_scores).item())
        s_idx = int(torch.argmax(s_scores).item())

        # Current top-2 averages to quickly get "others' max"
        topk_vals, topk_idx = torch.topk(avg, k=min(2, num_log))
        cur_max_val = float(topk_vals[0].item())
        cur_max_idx = int(topk_idx[0].item())
        cur_second_val = float(topk_vals[1].item()) if topk_vals.numel() >= 2 else float("-inf")

        best_idx = None
        best_key = None  # (new_peak, proxy_second, -expert_weight) to bias fixes to heavy experts on ties
        for cand in {d_idx, s_idx}:
            new_avg_cand = float((w[cand] / (cfloat[cand] + 1.0)).item())
            others_max = cur_second_val if cand == cur_max_idx else cur_max_val
            new_peak = max(others_max, new_avg_cand)
            proxy_second = others_max  # good proxy for the new second-highest avg
            tie_bias = -float(w[cand].item())
            key = (new_peak, proxy_second, tie_bias)
            if best_key is None or key < best_key:
                best_key = key
                best_idx = cand

        counts[best_idx] += 1

    return counts


def replicate_experts(
        weight: torch.Tensor,
        num_phy: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Replicate `num_log` experts to `num_phy` replicas using a water-filling
    allocation to minimize the maximum per-replica load, with a single
    donor→receiver fix-up for discrete smoothing.

    Parameters:
        weight: [X, num_log] (float CPU tensor)
        num_phy: total number of experts after replication

    Returns:
        phy2log: [X, num_phy], logical expert id of each physical expert
        rank: [X, num_phy], the replica rank
        logcnt: [X, num_log], number of replicas for each logical expert
    """
    n, num_log = weight.shape
    assert num_phy >= num_log
    device = weight.device

    phy_rows = []
    rank_rows = []
    cnt_rows = []

    exp_ids = torch.arange(num_log, dtype=torch.int64, device=device)
    for i in range(n):
        w = weight[i].float()
        counts = _waterfill_counts_row(w, num_phy)

        # Guarded donor->receiver fix-up over a bounded 2x2 candidate set.
        # Donors: top-2 by avg with count>1; Receivers: bottom-2 by avg.
        # Apply at most one move only if it strictly reduces the peak average;
        # tie-break by the new second-highest average.
        if num_log > 1:
            avg = w / counts.to(w.dtype)
            cur_peak = float(avg.max().item())
            can_donate = (counts > 1)
            if bool(can_donate.any()):
                avg_mask = avg.clone()
                avg_mask[~can_donate] = float("-inf")
                kd = int(min(2, int(can_donate.sum().item())))
                donors = torch.topk(avg_mask, k=kd).indices.tolist() if kd > 0 else []
                kr = int(min(2, num_log))
                receivers = torch.topk(-avg, k=kr).indices.tolist() if kr > 0 else []

                best = None  # (new_peak, new_second, d, r, c_try)
                for d in donors:
                    for r in receivers:
                        if d == r:
                            continue
                        c_try = counts.clone()
                        c_try[d] -= 1
                        c_try[r] += 1
                        avg_try = w / c_try.to(w.dtype)
                        new_peak = float(avg_try.max().item())
                        if new_peak + 1e-12 < cur_peak:
                            if num_log >= 2:
                                topk = torch.topk(avg_try, k=min(2, num_log)).values
                                new_second = float(topk[-1].item()) if topk.numel() >= 2 else float("-inf")
                            else:
                                new_second = float("-inf")
                            cand = (new_peak, new_second, int(d), int(r), c_try)
                            if best is None or cand < best:
                                best = cand
                if best is not None:
                    counts = best[4]

        cnt_rows.append(counts)

        # Build phy2log and rank
        phy2log_i = torch.repeat_interleave(exp_ids, counts)
        starts = torch.cumsum(counts, dim=0) - counts
        arange_phy = torch.arange(num_phy, dtype=torch.int64, device=device)
        rank_i = arange_phy - torch.repeat_interleave(starts, counts)

        phy_rows.append(phy2log_i)
        rank_rows.append(rank_i)

    phy2log = torch.stack(phy_rows, dim=0)
    rank = torch.stack(rank_rows, dim=0)
    logcnt = torch.stack(cnt_rows, dim=0)
    return phy2log, rank, logcnt


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

    # Step 1: pack groups to nodes
    tokens_per_group = weight.unflatten(-1, (num_groups, group_size)).sum(-1)
    group_pack_index, group_rank_in_pack = balanced_packing(
        tokens_per_group, num_nodes)
    log2mlog = (((group_pack_index * groups_per_node + group_rank_in_pack) *
                 group_size).unsqueeze(-1) +
                torch.arange(group_size,
                             dtype=torch.int64,
                             device=group_pack_index.device)).flatten(-2)
    mlog2log = inverse(log2mlog)

    # Step 2: construct redundant experts within nodes
    # [num_layers * num_nodes, num_logical_experts // num_nodes]
    tokens_per_mlog = weight.gather(-1, mlog2log).view(
        -1, num_logical_experts // num_nodes)
    phy2mlog, phyrank, mlogcnt = replicate_experts(
        tokens_per_mlog, num_physical_experts // num_nodes)

    # Step 3: pack physical_experts to GPUs
    # [num_layers * num_nodes, num_physical_experts // num_nodes]
    tokens_per_phy = (tokens_per_mlog / mlogcnt).gather(-1, phy2mlog)
    # Use diversity-aware packing to spread replicas of the same logical expert
    # across GPUs within a node, reducing hotspotting and improving balance.
    pack_index, rank_in_pack = balanced_packing_diverse(
        tokens_per_phy, phy2mlog, num_gpus // num_nodes)
    phy2pphy = pack_index * phy_experts_per_gpu + rank_in_pack
    pphy2phy = inverse(phy2pphy)

    pphy2mlog = phy2mlog.gather(
        -1, pphy2phy)  # [num_layers * num_nodes, num_log_per_nodes]
    pphy2mlog = (pphy2mlog.view(num_layers, num_nodes, -1) + torch.arange(
        0,
        num_logical_experts,
        num_logical_experts // num_nodes,
        device=group_pack_index.device,
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
        # use hierarchical load-balance policy
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, num_groups, num_nodes, num_gpus)
    else:
        # use global load-balance policy
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