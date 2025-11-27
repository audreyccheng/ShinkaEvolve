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


def balanced_packing(weight: torch.Tensor,
                     num_packs: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pack n weighted objects to m packs, such that each bin contains exactly
    n/m objects and the weights of all packs are as balanced as possible.

    Uses Randomized Restarts with Parallel Greedy Assignment and Vectorized
    Refinement to escape local optima.

    Parameters:
        weight: [X, n], the weight of each item
        num_packs: number of packs

    Returns:
        pack_index: [X, n], the pack index of each item
        rank_in_pack: [X, n], the rank of the item in the pack
    """
    num_layers, num_groups = weight.shape
    device = weight.device
    assert num_groups % num_packs == 0
    groups_per_pack = num_groups // num_packs

    if groups_per_pack == 1:
        pack_index = torch.arange(weight.size(-1),
                                  dtype=torch.int64,
                                  device=device).expand(weight.shape)
        rank_in_pack = torch.zeros_like(weight, dtype=torch.int64)
        return pack_index, rank_in_pack

    # Number of parallel restarts (1 deterministic + N-1 randomized)
    num_restarts = 8

    # Expand problem for parallel execution: [num_layers * num_restarts, num_groups]
    # Replicate input weights
    weight_expanded = weight.repeat_interleave(num_restarts, dim=0)

    # Generate perturbations for restarts 1..N-1
    weight_view = weight_expanded.view(num_layers, num_restarts, num_groups)

    # We use multiplicative noise to preserve 0s and scale
    if num_restarts > 1:
        # Create noise for all but the first restart in each group
        noise = torch.rand(num_layers, num_restarts - 1, num_groups, device=device) * 0.05  # 5% noise
        perturbed_slice = weight_view[:, 1:, :].clone()
        perturbed_slice *= (1.0 + noise)

        # Combine deterministic (0-th) and perturbed
        # We need sorting indices based on perturbed weights, but we accumulate original weights
        sorting_weights = torch.cat([weight_view[:, 0:1, :], perturbed_slice], dim=1).reshape(-1, num_groups)
    else:
        sorting_weights = weight_expanded

    # Sort weights (LPT heuristic) based on perturbed values
    sorted_indices = sorting_weights.argsort(dim=-1, descending=True)

    # Gather original weights in the sorted order for accumulation
    w_sorted = weight_expanded.gather(1, sorted_indices)

    batch_size = weight_expanded.shape[0]

    # --- 1. Vectorized Greedy Assignment ---

    # State tracking
    pack_weights = torch.zeros(batch_size, num_packs, device=device)
    pack_counts = torch.zeros(batch_size, num_packs, dtype=torch.int64, device=device)

    # To store results in sorted order (temporarily)
    pack_idx_sorted = torch.empty(batch_size, num_groups, dtype=torch.int64, device=device)
    rank_sorted = torch.empty(batch_size, num_groups, dtype=torch.int64, device=device)

    row_indices = torch.arange(batch_size, device=device)

    for i in range(num_groups):
        w = w_sorted[:, i]

        # Mask valid packs (count < limit)
        valid_mask = pack_counts < groups_per_pack

        # Select pack with min weight among valid ones
        # Add large penalty to invalid packs
        candidate_weights = pack_weights.clone()
        candidate_weights[~valid_mask] = float('inf')

        chosen_pack = torch.argmin(candidate_weights, dim=1) # [B]
        chosen_idx = chosen_pack.unsqueeze(1) # [B, 1]

        # Update state
        pack_weights.scatter_add_(1, chosen_idx, w.unsqueeze(1))

        # Record rank
        current_counts = pack_counts.gather(1, chosen_idx).squeeze(1)
        rank_sorted[:, i] = current_counts

        # Increment count
        pack_counts.scatter_add_(1, chosen_idx, torch.ones(batch_size, 1, dtype=torch.int64, device=device))

        # Record pack
        pack_idx_sorted[:, i] = chosen_pack

    # --- 2. Vectorized Refinement (Pairwise Swap) ---

    # Reconstruct pack contents for easy swapping access: [B, P, G]
    pack_contents = torch.zeros(batch_size, num_packs, groups_per_pack, device=device)
    # Store original indices to reconstruct final map: [B, P, G]
    pack_item_ids = torch.zeros(batch_size, num_packs, groups_per_pack, dtype=torch.int64, device=device)

    flat_batch = row_indices.unsqueeze(1).expand(-1, num_groups).flatten()
    flat_pack = pack_idx_sorted.flatten()
    flat_rank = rank_sorted.flatten()

    pack_contents.index_put_((flat_batch, flat_pack, flat_rank), w_sorted.flatten())
    pack_item_ids.index_put_((flat_batch, flat_pack, flat_rank), sorted_indices.flatten())

    # Refinement iterations (Max vs Min pack)
    for _ in range(20):
        # Current weights
        p_weights = pack_contents.sum(dim=-1)

        max_val, max_pack = p_weights.max(dim=1)
        min_val, min_pack = p_weights.min(dim=1)
        diff = max_val - min_val

        active_mask = diff > 1e-4
        if not active_mask.any():
            break

        # Extract items from max and min packs: [B, G]
        items_max = pack_contents[row_indices, max_pack]
        items_min = pack_contents[row_indices, min_pack]

        # Compute delta for all pairs: w_max - w_min
        # [B, G, G]
        delta = items_max.unsqueeze(2) - items_min.unsqueeze(1)

        # Improvement: minimize |new_diff| where new_diff = diff - 2*delta
        # Maximize gain = diff - |diff - 2*delta|
        diff_view = diff.view(-1, 1, 1)
        gain = diff_view - (diff_view - 2 * delta).abs()

        # Find best swap
        best_gain, best_idx_flat = gain.view(batch_size, -1).max(dim=1)

        # Determine who swaps
        do_swap = (best_gain > 1e-5) & active_mask

        if not do_swap.any():
            break

        swap_indices = best_idx_flat[do_swap]
        batch_indices_swap = row_indices[do_swap]

        idx_max = swap_indices // groups_per_pack
        idx_min = swap_indices % groups_per_pack

        p_max_swap = max_pack[do_swap]
        p_min_swap = min_pack[do_swap]

        # Execute swaps
        val_max = pack_contents[batch_indices_swap, p_max_swap, idx_max]
        val_min = pack_contents[batch_indices_swap, p_min_swap, idx_min]

        pack_contents[batch_indices_swap, p_max_swap, idx_max] = val_min
        pack_contents[batch_indices_swap, p_min_swap, idx_min] = val_max

        id_max = pack_item_ids[batch_indices_swap, p_max_swap, idx_max]
        id_min = pack_item_ids[batch_indices_swap, p_min_swap, idx_min]

        pack_item_ids[batch_indices_swap, p_max_swap, idx_max] = id_min
        pack_item_ids[batch_indices_swap, p_min_swap, idx_min] = id_max

    # --- 3. Select Best Restart ---

    final_weights = pack_contents.sum(dim=-1)
    imbalance = final_weights.max(dim=1).values - final_weights.min(dim=1).values
    imbalance = imbalance.view(num_layers, num_restarts)

    best_restart_idx = imbalance.argmin(dim=1) # [L]

    # Gather best results
    # Indices in the batch dimension
    best_batch_indices = torch.arange(num_layers, device=device) * num_restarts + best_restart_idx
    final_item_ids = pack_item_ids[best_batch_indices] # [L, P, G]

    # Convert to output format: [L, N] -> pack_index, rank
    pack_index = torch.empty(num_layers, num_groups, dtype=torch.int64, device=device)
    rank_in_pack = torch.empty(num_layers, num_groups, dtype=torch.int64, device=device)

    # Helper grids
    grid_packs = torch.arange(num_packs, device=device).view(1, num_packs, 1).expand(num_layers, -1, groups_per_pack)
    grid_ranks = torch.arange(groups_per_pack, device=device).view(1, 1, groups_per_pack).expand(num_layers, num_packs, -1)

    flat_item_ids = final_item_ids.view(num_layers, -1)
    flat_packs = grid_packs.reshape(num_layers, -1)
    flat_ranks = grid_ranks.reshape(num_layers, -1)

    pack_index.scatter_(1, flat_item_ids, flat_packs)
    rank_in_pack.scatter_(1, flat_item_ids, flat_ranks)

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
    phy2log = torch.arange(num_phy, dtype=torch.int64,
                           device=device).repeat(n, 1)
    rank = torch.zeros(n, num_phy, dtype=torch.int64, device=device)
    logcnt = torch.ones(n, num_log, dtype=torch.int64, device=device)
    arangen = torch.arange(n, dtype=torch.int64, device=device)

    # Pre-compute scores. Score = weight / count. Initially count is 1.
    current_scores = weight.float() / logcnt.float()

    for i in range(num_log, num_phy):
        redundant_indices = current_scores.argmax(dim=-1)
        phy2log[:, i] = redundant_indices
        rank[:, i] = logcnt[arangen, redundant_indices]

        # Update logcnt
        logcnt[arangen, redundant_indices] += 1

        # Incrementally update scores for modified experts
        new_cnt = logcnt[arangen, redundant_indices].float()
        chosen_weight = weight[arangen, redundant_indices].float()
        current_scores[arangen, redundant_indices] = chosen_weight / new_cnt

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
    pack_index, rank_in_pack = balanced_packing(tokens_per_phy,
                                                num_gpus // num_nodes)
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
    weight = weight.float()
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