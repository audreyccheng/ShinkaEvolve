# EVOLVE-BLOCK-START
"""
Expert parallelism load balancer (EPLB) for vLLM.

This module implements the core rearrangement algorithm.

This version uses:
- Water-filling replica allocation to minimize the maximum per-replica load.
- Diversity-aware heap-based packing for GPU placement within nodes.
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

    # Longest-processing-time greedy with capacity constraints
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
            # choose pack with capacity and minimal load
            best_p = None
            best_load = None
            for p in range(num_packs):
                if pack_counts[p] >= groups_per_pack:
                    continue
                if best_load is None or pack_loads[p] < best_load:
                    best_load = pack_loads[p]
                    best_p = p
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
    # Binary search T such that sum max(1, ceil(w_i / T)) <= target_total
    lo = 0.0
    hi = max(maxw, 1.0)
    # Handle all-zero quickly
    if maxw == 0.0:
        counts = torch.ones(num_log, dtype=torch.int64, device=w.device)
        extras = target_total - num_log
        if extras > 0:
            base_add = extras // num_log
            rem = extras % num_log
            if base_add > 0:
                counts += base_add
            if rem > 0:
                counts[:rem] += 1
        return counts

    for _ in range(40):
        mid = 0.5 * (lo + hi)
        # counts_i = max(1, ceil(w_i / mid))
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

    # Greedy water-filling for remaining extras
    extras = target_total - s
    while extras > 0:
        k = min(extras, num_log)
        # Select top-k by current w_i / c_i
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
    Water-filling replication to minimize the maximum per-replica load.

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
    Diversity-aware heap-like greedy packing with exact capacity per pack.

    Parameters:
        weights: [X, n], float CPU
        labels:  [X, n], int64 CPU
        num_packs: number of packs

    Returns:
        pack_index: [X, n], assigned pack id per item
        rank_in_pack: [X, n], position within pack
    """
    num_layers, n_items = weights.shape
    assert n_items % num_packs == 0
    cap = n_items // num_packs

    if cap == 1:
        # Each pack gets exactly one item
        # Assign by descending weight but any mapping works; keep contiguous
        # pack_index = item_id % num_packs
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

        for g in sorted_idx:
            lab = int(row_labels[g].item())
            wv = float(row_w[g].item())

            best_p = None
            best_key = None
            for p in range(num_packs):
                if counts[p] >= cap:
                    continue
                rep = label_counts[p].get(lab, 0)
                key = (rep, loads[p], counts[p])
                if best_key is None or key < best_key:
                    best_key = key
                    best_p = p

            pack_index[i, g] = best_p
            rank_in_pack[i, g] = counts[best_p]
            counts[best_p] += 1
            loads[best_p] += wv
            label_counts[best_p][lab] += 1

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

    # Step 1: pack groups to nodes (capacity-aware LPT)
    tokens_per_group = weight.unflatten(-1, (num_groups, group_size)).sum(-1)
    group_pack_index, group_rank_in_pack = balanced_packing(
        tokens_per_group, num_nodes)
    log2mlog = (((group_pack_index * groups_per_node + group_rank_in_pack) *
                 group_size).unsqueeze(-1) +
                torch.arange(group_size,
                             dtype=torch.int64,
                             device=group_pack_index.device)).flatten(-2)
    mlog2log = inverse(log2mlog)

    # Step 2: replicate within nodes using water-filling
    # Reorder to meta-logical within nodes: [num_layers * num_nodes, num_logical_experts // num_nodes]
    tokens_per_mlog = weight.gather(-1, mlog2log).view(
        -1, num_logical_experts // num_nodes)

    phy2mlog, phyrank, mlogcnt = replicate_experts_waterfill(
        tokens_per_mlog, num_physical_experts // num_nodes
    )

    # Step 3: pack physical experts to GPUs in each node with diversity-aware heap
    tokens_per_phy = (tokens_per_mlog / mlogcnt).gather(-1, phy2mlog)
    # each row corresponds to a (layer, node)
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
