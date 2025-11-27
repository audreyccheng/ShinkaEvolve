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
        pack_weights = [0.0] * num_packs
        pack_items = [0] * num_packs
        for g in indices[i].tolist():
            # choose pack with capacity and minimal load
            best_p = None
            best_load = None
            best_cnt = None
            for p in range(num_packs):
                if pack_items[p] >= groups_per_pack:
                    continue
                if (best_load is None or pack_weights[p] < best_load or
                    (abs(pack_weights[p] - (best_load if best_load is not None else 0.0)) <= 1e-12 and pack_items[p] < (best_cnt if best_cnt is not None else 1 << 30))):
                    best_p = p
                    best_load = pack_weights[p]
                    best_cnt = pack_items[p]
            pack_index[i, g] = best_p
            rank_in_pack[i, g] = pack_items[best_p]
            pack_items[best_p] += 1
            pack_weights[best_p] += float(weight[i, g].item())
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

    extras = target_total - s
    while extras > 0:
        k = min(extras, num_log)
        scores = w / counts.to(w.dtype)
        topk_idx = torch.argsort(scores, descending=True)[:k]
        counts[topk_idx] += 1
        extras -= k
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
        w = weight[i]  # [num_log]
        counts = _waterfill_counts_row(w, num_phy)

        # Up to 2 donor->receiver moves among top-2 donors and bottom-2 receivers
        if num_log > 1:
            max_iters = 2
            for _ in range(max_iters):
                counts_safe = torch.clamp(counts, min=1)
                avg = w / counts_safe.to(w.dtype)
                cur_peak = float(avg.max().item())

                can_donate = (counts > 1)
                if not bool(can_donate.any()):
                    break
                kd = int(min(2, int(can_donate.sum().item())))
                kr = int(min(2, num_log))

                avg_mask = avg.clone()
                avg_mask[~can_donate] = float("-inf")
                donors = torch.topk(avg_mask, k=kd).indices.tolist()
                receivers = torch.topk(-avg, k=kr).indices.tolist()

                best_pair = None
                best_peak = cur_peak
                for d in donors:
                    for r in receivers:
                        if d == r:
                            continue
                        c_try = counts.clone()
                        c_try[d] -= 1
                        c_try[r] += 1
                        peak = float((w / c_try.to(w.dtype)).max().item())
                        if peak + 1e-9 < best_peak:
                            best_peak = peak
                            best_pair = (d, r)
                if best_pair is None:
                    break
                d, r = best_pair
                counts[d] -= 1
                counts[r] += 1

        logcnt_list.append(counts)

        # Build phy2log and rank (contiguous blocks per logical expert)
        phy2log_i = torch.repeat_interleave(exp_ids, counts)
        starts = torch.cumsum(counts, dim=0) - counts
        arange_phy = torch.arange(num_phy, dtype=torch.int64, device=device)
        rank_i = arange_phy - torch.repeat_interleave(starts, counts)

        phy2log_list.append(phy2log_i)
        rank_list.append(rank_i)

    phy2log = torch.stack(phy2log_list, dim=0)
    rank = torch.stack(rank_list, dim=0)
    logcnt = torch.stack(logcnt_list, dim=0)
    return phy2log, rank, logcnt


def balanced_packing_diverse(weight: torch.Tensor,
                             label: torch.Tensor,
                             num_packs: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Balanced packing with diversity: prefer spreading items with the same label
    across different packs to reduce hotspotting when replicas of the same
    logical expert are placed on the same GPU.

    Algorithm:
    - Fast path when labels are unique.
    - Seed: place one heaviest item per label to packs without that label,
      minimizing projected load.
    - Greedy fill: projected-load objective; tie-break by fewer repeats, then
      fewer items.
    - Bounded refinement: evaluate swaps among a tiny set of heavy/light pairs
      and apply if it strictly reduces the global peak with minimal new duplicates.

    Parameters:
        weight: [X, n], the weight of each item
        label:  [X, n], an integer label for each item (e.g., logical expert id)
        num_packs: number of packs

    Returns:
        pack_index: [X, n], the pack index of each item
        rank_in_pack: [X, n], the rank of the item in the pack
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

        # Fast duplicate check
        lbl_sorted, _ = torch.sort(row_labels)
        all_unique = (torch.unique_consecutive(lbl_sorted).numel() == row_labels.numel())

        loads = [0.0] * num_packs
        counts = [0] * num_packs
        pack_groups = [[] for _ in range(num_packs)]

        if all_unique:
            # No duplicate labels; standard greedy is faster
            for g in row_sorted:
                wv = float(row_w[g].item())
                best_p = None
                best_load = None
                best_cnt = None
                for p in range(num_packs):
                    if counts[p] >= items_per_pack:
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
            # Seed phase: place one heaviest item per label spreading labels
            label_items = defaultdict(list)
            for g in row_sorted:
                lab = int(row_labels[g].item())
                label_items[lab].append(g)
            label_order = sorted(
                label_items.keys(),
                key=lambda lab: float(row_w[label_items[lab][0]].item()),
                reverse=True,
            )

            label_counts = [defaultdict(int) for _ in range(num_packs)]
            assigned = [False] * num_items

            for lab in label_order:
                # pick heaviest unassigned item for this label
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
                    if counts[p] >= items_per_pack or label_counts[p].get(lab, 0) > 0:
                        continue
                    if (best_load is None or loads[p] < best_load or
                        (abs(loads[p] - best_load) <= 1e-12 and counts[p] < best_cnt)):
                        best_p = p
                        best_load = loads[p]
                        best_cnt = counts[p]
                # Fallback if all have this label
                if best_p is None:
                    for p in range(num_packs):
                        if counts[p] >= items_per_pack:
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

            # Greedy fill for remaining items with diversity-aware tie-breaking
            eps = 1e-6 * float(row_w.mean().item() if row_w.numel() > 0 else 1.0)
            for g in row_sorted:
                if assigned[g]:
                    continue
                lab = int(row_labels[g].item())
                wv = float(row_w[g].item())

                best_p = None
                best_base = None
                best_rep = None
                best_cnt = None
                for p in range(num_packs):
                    if counts[p] >= items_per_pack:
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

            # Bounded refinement: evaluate (h1↔l1), (h1↔l2), (h2↔l1) with label-aware penalty
            if num_packs >= 2:
                def _refine_once() -> bool:
                    order_asc = sorted(range(num_packs), key=lambda p: loads[p])
                    order_desc = list(reversed(order_asc))
                    pairs = []
                    if order_desc:
                        h1 = order_desc[0]
                        pairs.append((h1, order_asc[0]))
                        if len(order_asc) >= 2:
                            pairs.append((h1, order_asc[1]))
                    if len(order_desc) >= 2:
                        h2 = order_desc[1]
                        pairs.append((h2, order_asc[0]))

                    cur_max = max(loads)
                    best = None  # (new_peak, new_imb, penalty, d, r, ai, bi, a_item, b_item, wa, wb)

                    # Precompute other max/min quickly per pair
                    for (d, r) in pairs:
                        if d == r or not pack_groups[d] or not pack_groups[r] or loads[d] <= loads[r]:
                            continue
                        d_idx = torch.tensor(pack_groups[d], dtype=torch.int64, device=row_w.device)
                        r_idx = torch.tensor(pack_groups[r], dtype=torch.int64, device=row_w.device)
                        d_w = row_w[d_idx]
                        r_w = row_w[r_idx]
                        kd = min(2, d_w.numel())
                        kr = min(2, r_w.numel())
                        if kd == 0 or kr == 0:
                            continue
                        d_top = torch.topk(d_w, kd).indices.tolist()
                        r_bot = torch.topk(r_w, kr, largest=False).indices.tolist()

                        other_max = max([loads[p] for p in range(num_packs) if p != d and p != r], default=float("-inf"))
                        other_min = min([loads[p] for p in range(num_packs) if p != d and p != r], default=float("inf"))

                        # Build label counts for the two packs
                        lcnt_d = defaultdict(int)
                        lcnt_r = defaultdict(int)
                        for gg in pack_groups[d]:
                            lcnt_d[int(row_labels[gg].item())] += 1
                        for gg in pack_groups[r]:
                            lcnt_r[int(row_labels[gg].item())] += 1

                        for ai in d_top:
                            wa = float(d_w[ai].item())
                            a_item = int(d_idx[ai].item())
                            la = int(row_labels[a_item].item())
                            for bi in r_bot:
                                wb = float(r_w[bi].item())
                                b_item = int(r_idx[bi].item())
                                lb = int(row_labels[b_item].item())
                                new_d = loads[d] - wa + wb
                                new_r = loads[r] - wb + wa
                                new_peak = max(new_d, new_r, other_max)
                                new_bottom = min(new_d, new_r, other_min)
                                new_imb = new_peak - new_bottom
                                penalty = (1 if lcnt_d.get(lb, 0) > 0 else 0) + (1 if lcnt_r.get(la, 0) > 0 else 0)
                                cand = (new_peak, new_imb, penalty, d, r, ai, bi, a_item, b_item, wa, wb)
                                if best is None:
                                    best = cand
                                else:
                                    if (cand[0] + 1e-9 < best[0] or
                                        (abs(cand[0] - best[0]) <= 1e-9 and (cand[1] + 1e-9 < best[1] or
                                         (abs(cand[1] - best[1]) <= 1e-9 and cand[2] < best[2])))):
                                        best = cand

                    if best is None or best[0] + 1e-9 >= cur_max:
                        return False
                    # apply swap
                    _, _, _, d, r, ai, bi, a_item, b_item, wa, wb = best
                    loads[d] = loads[d] - wa + wb
                    loads[r] = loads[r] - wb + wa
                    pack_groups[d][ai] = b_item
                    pack_groups[r][bi] = a_item
                    pack_index[i, a_item] = r
                    pack_index[i, b_item] = d
                    for rr, gg in enumerate(pack_groups[d]):
                        rank_in_pack[i, gg] = rr
                    for rr, gg in enumerate(pack_groups[r]):
                        rank_in_pack[i, gg] = rr
                    return True

                # Do up to two passes if imbalance is notable
                mean_load = sum(loads) / max(1, num_packs)
                delta = max(loads) - min(loads)
                steps = 2 if (mean_load > 0 and delta / max(mean_load, 1e-12) > 0.12) else 1
                for _ in range(steps):
                    if not _refine_once():
                        break

    return pack_index, rank_in_pack


def replicate_experts(
        weight: torch.Tensor,
        num_phy: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Legacy greedy replica construction kept for API compatibility if needed,
    but not used in the hierarchical pipeline. Retained to preserve exports.
    """
    n, num_log = weight.shape
    num_redundant = num_phy - num_log
    assert num_redundant >= 0
    device = weight.device
    phy2log = torch.arange(num_phy, dtype=torch.int64,
                           device=device).repeat(n, 1)
    rank = torch.zeros(n, num_phy, dtype=torch.int64, device=device)
    logcnt = torch.ones(n, num_log, dtype=torch.int64, device=device)
    arangen = torch.arange(n, dtype=torch.int64, device=device)
    for i in range(num_log, num_phy):
        redundant_indices = (weight / logcnt).max(dim=-1).indices
        phy2log[:, i] = redundant_indices
        rank[:, i] = logcnt[arangen, redundant_indices]
        logcnt[arangen, redundant_indices] += 1
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

    # Step 2: construct redundant experts within nodes using water-filling
    tokens_per_mlog = weight.gather(-1, mlog2log).view(
        -1, num_logical_experts // num_nodes)
    phy2mlog, phyrank, mlogcnt = replicate_experts_waterfill(
        tokens_per_mlog, num_physical_experts // num_nodes)

    # Step 3: pack physical_experts to GPUs with diversity-aware packing
    tokens_per_phy = (tokens_per_mlog / mlogcnt).gather(-1, phy2mlog)
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