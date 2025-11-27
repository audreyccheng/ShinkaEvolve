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
    logical expert are placed on the same GPU.

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

    # Ensure CPU tensors for Python-side loops
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

        # Quick duplicate check to avoid overhead when not needed
        seen = set()
        has_dup = False
        for idx in range(num_items):
            l = int(row_labels[idx].item())
            if l in seen:
                has_dup = True
                break
            seen.add(l)

        if not has_dup:
            # No duplicate labels; standard greedy suffices and is faster
            pack_loads = [0.0] * num_packs
            pack_counts = [0] * num_packs
            for group in row_sorted:
                # choose among packs with capacity the one with min load
                best_p = min(
                    (p for p in range(num_packs) if pack_counts[p] < items_per_pack),
                    key=lambda p: pack_loads[p],
                )
                pack_index[i, group] = best_p
                rank_in_pack[i, group] = pack_counts[best_p]
                pack_counts[best_p] += 1
                pack_loads[best_p] += float(row_w[group].item())
            continue

        # Diversity-aware greedy
        pack_loads = [0.0] * num_packs
        pack_counts = [0] * num_packs
        label_counts = [defaultdict(int) for _ in range(num_packs)]

        for group in row_sorted:
            lab = int(row_labels[group].item())
            # Select pack preferring fewer repetitions of label, then lower load, then fewer items
            best_p = None
            best_key = None
            for p in range(num_packs):
                if pack_counts[p] >= items_per_pack:
                    continue
                rep = label_counts[p].get(lab, 0)
                key = (rep, pack_loads[p], pack_counts[p])
                if best_key is None or key < best_key:
                    best_key = key
                    best_p = p
            # Assign
            pack_index[i, group] = best_p
            rank_in_pack[i, group] = pack_counts[best_p]
            pack_counts[best_p] += 1
            pack_loads[best_p] += float(row_w[group].item())
            label_counts[best_p][lab] += 1

    return pack_index, rank_in_pack


def replicate_experts(
        weight: torch.Tensor,
        num_phy: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Replicate `num_log` experts to `num_phy` replicas, such that the maximum
    load of all replicas is minimized.

    Parameters:
        weight: [X, num_log]
        num_phy: total number of experts after replication

    Returns:
        phy2log: [X, num_phy], logical expert id of each physical expert
        rank: [X, num_phy], the replica rank
        logcnt: [X, num_log], number of replicas for each logical expert
    """
    n, num_log = weight.shape
    num_redundant = num_phy - num_log
    assert num_redundant >= 0
    device = weight.device

    if num_redundant == 0:
        # Trivial mapping: one physical per logical
        phy2log = torch.arange(num_log, dtype=torch.int64,
                               device=device).repeat(n, 1)
        rank = torch.zeros(n, num_phy, dtype=torch.int64, device=device)
        logcnt = torch.ones(n, num_log, dtype=torch.int64, device=device)
        return phy2log, rank, logcnt

    # Proportional apportionment (Hamilton's method / largest remainder)
    w = weight.float()
    sums = w.sum(dim=-1, keepdim=True)  # [n,1]
    logcnt = torch.ones(n, num_log, dtype=torch.int64, device=device)

    # Rows with nonzero total weight
    nonzero_mask = (sums.squeeze(-1) > 0)
    if nonzero_mask.any():
        nz_rows = nonzero_mask.nonzero(as_tuple=False).squeeze(-1)
        w_nz = w[nz_rows]
        sums_nz = sums[nz_rows]
        quotas = (w_nz / sums_nz) * float(num_redundant)  # desired additional counts
        add_floor = torch.floor(quotas).to(torch.int64)  # [k, num_log]
        # remaining replicas to distribute per row
        remain = num_redundant - add_floor.sum(dim=-1)   # [k]
        frac = (quotas - add_floor.float())              # [k, num_log]
        logcnt[nz_rows] += add_floor

        if remain.numel() > 0:
            # Assign remaining by largest fractional parts per row
            # Sort indices of fractional parts descending
            order = torch.argsort(frac, dim=-1, descending=True)
            # extras to add 1 at the top 'remain[i]' positions per row
            for idx_row in range(order.size(0)):
                r = int(remain[idx_row].item())
                if r <= 0:
                    continue
                cols = order[idx_row, :r]
                logcnt[nz_rows[idx_row], cols] += 1

    # Rows with zero total weight: distribute evenly
    if (~nonzero_mask).any():
        z_rows = (~nonzero_mask).nonzero(as_tuple=False).squeeze(-1)
        base_add = num_redundant // num_log
        rem = num_redundant % num_log
        if base_add > 0:
            logcnt[z_rows] += base_add
        if rem > 0:
            # Deterministic assignment to the first 'rem' experts
            first_idx = torch.arange(rem, device=device)
            logcnt[z_rows.unsqueeze(-1), first_idx] += 1

    # Build mappings per row without Python-item loops over experts
    phy2log_list = []
    rank_list = []
    exp_ids = torch.arange(num_log, dtype=torch.int64, device=device)
    arange_phy = torch.arange(num_phy, dtype=torch.int64, device=device)

    for i in range(n):
        counts = logcnt[i]  # [num_log], int64
        # Repeat expert ids according to their replica counts
        phy2log_i = torch.repeat_interleave(exp_ids, counts)
        # ranks: 0..count-1 for each expert, in the repeated order
        starts = torch.cumsum(counts, dim=0) - counts
        rank_i = arange_phy - torch.repeat_interleave(starts, counts)
        phy2log_list.append(phy2log_i)
        rank_list.append(rank_i)

    phy2log = torch.stack(phy2log_list, dim=0)
    rank = torch.stack(rank_list, dim=0)
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
