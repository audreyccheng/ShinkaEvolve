# EVOLVE-BLOCK-START
"""
Expert parallelism load balancer (EPLB) for vLLM.

This module implements the core rearrangement algorithm using
Chunked Sorted Greedy packing and Binary Search based replication.
"""

import torch


def balanced_packing(weight: torch.Tensor,
                     num_packs: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pack n weighted objects to m packs using a Folded Chunked Sorted Greedy strategy.

    Items are processed in "double chunks" of size 2*m. In each double chunk,
    items are paired (heaviest + lightest) to form m balanced pairs, which are
    then assigned to packs greedily (heaviest pair -> lightest pack).
    This smoothing reduces load variance significantly compared to single-item greedy.

    Parameters:
        weight: [layers, n], the weight of each item
        num_packs: number of packs

    Returns:
        pack_index: [layers, n], the pack index of each item
        rank_in_pack: [layers, n], the rank of the item in the pack
    """
    num_layers, num_items = weight.shape
    device = weight.device

    sorted_weights, sorted_indices = weight.sort(dim=-1, descending=True)

    pack_loads = torch.zeros(num_layers, num_packs, device=device, dtype=weight.dtype)
    aligned_pack_ids = torch.empty(num_layers, num_items, device=device, dtype=torch.int64)
    aligned_ranks = torch.empty(num_layers, num_items, device=device, dtype=torch.int64)

    # Process in chunks of 2 * num_packs to enable folding
    double_chunk_size = 2 * num_packs
    num_double_chunks = num_items // double_chunk_size

    for k in range(num_double_chunks):
        start = k * double_chunk_size
        end = start + double_chunk_size

        # Chunk weights: [Layers, 2*M]
        chunk_weights = sorted_weights[:, start:end]

        # Pair indices: 0 with 2M-1, 1 with 2M-2, ...
        idx_low = torch.arange(num_packs, device=device)
        idx_high = torch.arange(double_chunk_size - 1, num_packs - 1, -1, device=device)

        # Compute pair weights
        pair_weights = chunk_weights[:, idx_low] + chunk_weights[:, idx_high]

        # Identify assignment order: Heaviest pair -> Lightest pack
        _, sorted_bin_indices = pack_loads.sort(dim=-1) # Lightest pack at 0
        _, pairs_order = pair_weights.sort(dim=-1, descending=True) # Heaviest pair at 0

        # We assign sorted_bin_indices[i] to pairs_order[i]
        # Map: Pair j gets Pack assigned_packs[j]
        # assigned_packs[pairs_order] = sorted_bin_indices
        assigned_packs = torch.empty_like(sorted_bin_indices)
        assigned_packs.scatter_(1, pairs_order, sorted_bin_indices)

        # Assign to items
        # Low item (j) and High item (2M-1-j) get the same pack
        aligned_pack_ids[:, start + idx_low] = assigned_packs
        aligned_pack_ids[:, start + idx_high] = assigned_packs

        # Assign ranks (2 items added per pack)
        aligned_ranks[:, start + idx_low] = 2 * k
        aligned_ranks[:, start + idx_high] = 2 * k + 1

        # Update loads
        pack_loads.scatter_add_(1, assigned_packs, pair_weights)

    # Handle remainder chunk (size M, if any)
    remainder_start = num_double_chunks * double_chunk_size
    if remainder_start < num_items:
        start = remainder_start
        end = num_items # Should be start + num_packs

        chunk_weights = sorted_weights[:, start:end]
        _, sorted_bin_indices = pack_loads.sort(dim=-1)

        aligned_pack_ids[:, start:end] = sorted_bin_indices
        aligned_ranks[:, start:end] = 2 * num_double_chunks

        pack_loads.scatter_add_(1, sorted_bin_indices, chunk_weights)

    # Scatter back to original order
    pack_index = torch.empty_like(aligned_pack_ids)
    rank_in_pack = torch.empty_like(aligned_ranks)

    pack_index.scatter_(1, sorted_indices, aligned_pack_ids)
    rank_in_pack.scatter_(1, sorted_indices, aligned_ranks)

    return pack_index, rank_in_pack


def replicate_experts(
        weight: torch.Tensor,
        num_phy: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Replicate experts using Binary Search on Max Load followed by Greedy Refinement.

    Finds a capacity threshold T such that sum(ceil(w/T)) is close to num_phy,
    then adjusts counts to exactly num_phy minimizing the max load density.
    """
    num_layers, num_log = weight.shape
    device = weight.device

    # Trivial case
    if num_phy == num_log:
        phy2log = torch.arange(num_log, device=device).expand(num_layers, -1)
        rank = torch.zeros(num_layers, num_phy, dtype=torch.int64, device=device)
        logcnt = torch.ones(num_layers, num_log, dtype=torch.int64, device=device)
        return phy2log, rank, logcnt

    # Binary Search for optimal max load threshold
    # Lower bound: average load per physical slot
    low = weight.sum(dim=-1, keepdim=True) / num_phy
    # Upper bound: max weight (since min replica count is 1)
    high = weight.max(dim=-1, keepdim=True).values

    # Ensure low > 0
    low = torch.max(low, torch.tensor(1e-6, device=device))

    # 15 iterations provide sufficient precision for integer allocation
    for _ in range(15):
        mid = (low + high) * 0.5
        # Calculate required replicas for this threshold
        counts = torch.ceil(weight / mid)
        total = counts.sum(dim=-1, keepdim=True)

        # If we fit within num_phy, try to lower the threshold (tighten)
        mask = total <= num_phy
        high = torch.where(mask, mid, high)
        low = torch.where(mask, low, mid)

    # Initial counts using the feasible threshold
    logcnt = torch.ceil(weight / high).long().clamp(min=1)

    # Correct sum to equal num_phy
    current_sum = logcnt.sum(dim=-1)
    diff = num_phy - current_sum

    # Handle under-allocation (sum < num_phy): Add to experts with highest load density
    max_diff = int(diff.max().item())
    if max_diff > 0:
        rows = torch.arange(num_layers, device=device)
        for _ in range(max_diff):
            active = current_sum < num_phy
            if not active.any(): break

            # Density = weight / count
            density = weight / logcnt.float()

            # Pick expert with max density
            target_idx = density.argmax(dim=-1)

            # Update active rows
            mask_indices = rows[active]
            mask_targets = target_idx[active]

            logcnt.index_put_((mask_indices, mask_targets),
                              torch.tensor(1, device=device, dtype=torch.int64),
                              accumulate=True)
            current_sum[active] += 1

    # Handle over-allocation (sum > num_phy): Remove from experts with lowest cost
    # Cost = New Load = weight / (count - 1). We want minimum new load.
    min_diff = int(diff.min().item())
    if min_diff < 0:
        rows = torch.arange(num_layers, device=device)
        for _ in range(abs(min_diff)):
            active = current_sum > num_phy
            if not active.any(): break

            # Only consider experts with > 1 replica
            valid = logcnt > 1
            cost = weight / (logcnt - 1).float()
            cost[~valid] = float('inf')

            target_idx = cost.argmin(dim=-1)

            mask_indices = rows[active]
            mask_targets = target_idx[active]

            logcnt.index_put_((mask_indices, mask_targets),
                              torch.tensor(-1, device=device, dtype=torch.int64),
                              accumulate=True)
            current_sum[active] -= 1

    # Construct physical to logical map
    flat_log_ids = torch.arange(num_log, device=device).repeat(num_layers)
    flat_counts = logcnt.flatten()

    flat_phy2log = torch.repeat_interleave(flat_log_ids, flat_counts)

    # Reshape
    # If sizes mismatch due to weird edge cases, we safeguard, though logic implies exact match
    target_size = num_layers * num_phy
    if flat_phy2log.numel() != target_size:
        if flat_phy2log.numel() < target_size:
            flat_phy2log = torch.cat([flat_phy2log, torch.zeros(target_size - flat_phy2log.numel(), device=device, dtype=torch.long)])
        else:
            flat_phy2log = flat_phy2log[:target_size]

    phy2log = flat_phy2log.view(num_layers, num_phy)

    # Calculate ranks
    # Rank is the 0-based index of a replica for a specific logical expert
    offsets = torch.zeros_like(logcnt)
    offsets[:, 1:] = logcnt[:, :-1].cumsum(dim=1)

    mapped_offsets = offsets.gather(1, phy2log)
    phy_indices = torch.arange(num_phy, device=device).expand(num_layers, -1)

    rank = phy_indices - mapped_offsets

    return phy2log, rank, logcnt


def rebalance_experts_hierarchical(
    weight: torch.Tensor,
    num_physical_experts: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
):
    """
    Hierarchical rebalancing using the optimized packing and replication strategies.
    """
    num_layers, num_logical_experts = weight.shape
    group_size = num_logical_experts // num_groups
    groups_per_node = num_groups // num_nodes
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

    # Step 1: Pack groups to nodes
    tokens_per_group = weight.unflatten(-1, (num_groups, group_size)).sum(-1)
    group_pack_index, group_rank_in_pack = balanced_packing(
        tokens_per_group, num_nodes)

    log2mlog = (((group_pack_index * groups_per_node + group_rank_in_pack) *
                 group_size).unsqueeze(-1) +
                torch.arange(group_size,
                             dtype=torch.int64,
                             device=group_pack_index.device)).flatten(-2)
    mlog2log = inverse(log2mlog)

    # Step 2: Replicate experts within nodes
    tokens_per_mlog = weight.gather(-1, mlog2log).view(
        -1, num_logical_experts // num_nodes)
    phy2mlog, phyrank, mlogcnt = replicate_experts(
        tokens_per_mlog, num_physical_experts // num_nodes)

    # Step 3: Pack physical experts to GPUs
    # Load per replica is approximated as total_weight / count
    tokens_per_phy = (tokens_per_mlog / mlogcnt).gather(-1, phy2mlog)
    pack_index, rank_in_pack = balanced_packing(tokens_per_phy,
                                                num_gpus // num_nodes)

    phy2pphy = pack_index * phy_experts_per_gpu + rank_in_pack
    pphy2phy = inverse(phy2pphy)

    # Map back to original logical IDs
    pphy2mlog = phy2mlog.gather(-1, pphy2phy)

    node_offsets = torch.arange(
        0,
        num_logical_experts,
        num_logical_experts // num_nodes,
        device=weight.device,
    ).view(1, -1, 1)

    pphy2mlog_restored = (pphy2mlog.view(num_layers, num_nodes, -1) + node_offsets).flatten(-2)

    pphy2log = mlog2log.gather(-1, pphy2mlog_restored)
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
    """
    num_layers, num_logical_experts = weight.shape
    # Ensure CPU computation for algorithmic stability and memory
    weight = weight.float().cpu()

    if num_groups % num_nodes == 0:
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, num_groups, num_nodes, num_gpus)
    else:
        # Fallback to global policy if groups not divisible by nodes
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, 1, 1, num_gpus)

    # Construct log2phy map [layers, logical_experts, max_replicas]
    max_replicas = int(logcnt.max().item())

    log2phy = torch.full(
        (num_layers, num_logical_experts, max_replicas),
        -1,
        dtype=torch.int64,
        device=logcnt.device,
    )

    # Optimized scatter for log2phy
    # Flat indices calculation
    flat_layer_idx = torch.arange(num_layers, device=logcnt.device).unsqueeze(-1).expand(-1, num_replicas).flatten()
    flat_log_idx = phy2log.flatten()
    flat_rank_idx = phyrank.flatten()
    flat_phy_ids = torch.arange(num_replicas, dtype=torch.int64, device=logcnt.device).expand(num_layers, -1).flatten()

    # Map to flat index of log2phy
    # Index = layer * (num_logical * max_rep) + logical * max_rep + rank
    flat_indices = (flat_layer_idx * num_logical_experts * max_replicas) + \
                   (flat_log_idx * max_replicas) + \
                   flat_rank_idx

    # Scatter
    log2phy.view(-1).scatter_(0, flat_indices, flat_phy_ids)

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
