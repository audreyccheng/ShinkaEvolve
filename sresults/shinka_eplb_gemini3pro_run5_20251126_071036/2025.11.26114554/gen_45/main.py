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
    
    Uses massively parallel randomized greedy restarts (Parallel LPT) followed
    by vectorized pairwise refinement to find an optimal distribution.

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

    # Number of parallel restarts.
    # 2048 effectively utilizes GPU parallelism to explore the solution space.
    num_restarts = 2048

    # Expand weights: [Layers, Restarts, Groups]
    weight_expanded = weight.unsqueeze(1).expand(-1, num_restarts, -1).clone()

    # 1. Perturbation (Randomize Sort Order)
    # Restart 0: Deterministic (original weights)
    # Restarts 1..N: Multiplicative noise.
    # We use a linear schedule of noise from 0.01 (1%) to 1.0 (100%) to 
    # cover 'stable' LPT to 'randomized' Greedy.
    if num_restarts > 1:
        noise_scales = torch.linspace(0.01, 1.0, steps=num_restarts - 1, device=device)
        noise = torch.rand_like(weight_expanded[:, 1:]) * noise_scales.view(1, -1, 1)
        weight_expanded[:, 1:] *= (1.0 + noise)
        
    # Flatten for processing: [Batch, Groups] where Batch = Layers * Restarts
    flat_weight_perturbed = weight_expanded.reshape(-1, num_groups)
    
    # Use original weights for actual accumulation (repeated for each restart)
    flat_weight_original = weight.unsqueeze(1).expand(-1, num_restarts, -1).reshape(-1, num_groups)

    # Sort indices based on perturbed weights (Descending LPT)
    sorted_indices = flat_weight_perturbed.argsort(dim=-1, descending=True)
    
    # Gather actual weights in the sorted order
    w_sorted = flat_weight_original.gather(1, sorted_indices)

    batch_size = flat_weight_perturbed.shape[0]
    
    # 2. Vectorized Greedy Packing
    # State tracking: [Batch, Packs]
    pack_weights = torch.zeros(batch_size, num_packs, device=device)
    pack_counts = torch.zeros(batch_size, num_packs, dtype=torch.int64, device=device)

    # Outputs in sorted order: [Batch, Groups]
    pack_idx_sorted = torch.zeros(batch_size, num_groups, dtype=torch.int64, device=device)
    rank_sorted = torch.zeros(batch_size, num_groups, dtype=torch.int64, device=device)

    # Pre-allocate large value for masking
    inf_tensor = torch.full((batch_size, num_packs), float('inf'), device=device)
    row_indices = torch.arange(batch_size, device=device)

    # Iterate through items (columns)
    for i in range(num_groups):
        w_curr = w_sorted[:, i]

        # Identify valid packs (count < limit)
        valid_mask = pack_counts < groups_per_pack
        
        # Select pack with minimum weight among valid packs
        # In-place masking to avoid allocation
        costs = torch.where(valid_mask, pack_weights, inf_tensor)
        chosen_pack = costs.argmin(dim=1)
        
        # Gather current rank for the chosen pack
        # Equivalent to pack_counts[row, chosen_pack]
        ranks = pack_counts.gather(1, chosen_pack.unsqueeze(1)).squeeze(1)
        
        # Update state
        # pack_weights[row, chosen] += w
        pack_weights.scatter_add_(1, chosen_pack.unsqueeze(1), w_curr.unsqueeze(1))
        
        # pack_counts[row, chosen] += 1
        pack_counts.scatter_add_(1, chosen_pack.unsqueeze(1), torch.ones_like(chosen_pack.unsqueeze(1), dtype=torch.int64))
        
        # Record assignment
        pack_idx_sorted[:, i] = chosen_pack
        rank_sorted[:, i] = ranks

    # 3. Refinement (Vectorized Top-K / Bot-K Swap)
    # Prepare structures for fast swap evaluation: [Batch, Packs, Groups_Per_Pack]
    pack_contents = torch.zeros(batch_size, num_packs, groups_per_pack, device=device)
    pack_orig_ids = torch.zeros(batch_size, num_packs, groups_per_pack, dtype=torch.int64, device=device)

    # Scatter into structured format
    flat_batch_idx = row_indices.unsqueeze(1).expand(-1, num_groups).flatten()
    flat_pack = pack_idx_sorted.flatten()
    flat_rank = rank_sorted.flatten()
    
    pack_contents.index_put_((flat_batch_idx, flat_pack, flat_rank), w_sorted.flatten())
    pack_orig_ids.index_put_((flat_batch_idx, flat_pack, flat_rank), sorted_indices.flatten())

    # K determines the search breadth (Top-K vs Bot-K)
    K = min(num_packs, 4)

    # Iterative Refinement
    for _ in range(50):
        # Current pack weights
        p_weights = pack_contents.sum(dim=2) # [B, P]

        # Identify Top-K and Bottom-K packs
        # We need indices to gather content
        vals_top, idx_top = torch.topk(p_weights, k=K, largest=True)
        vals_bot, idx_bot = torch.topk(p_weights, k=K, largest=False)

        # Convergence check on primary objective
        diff = vals_top[:, 0] - vals_bot[:, 0]
        active_mask = diff > 1e-4
        if not active_mask.any():
            break

        # Gather items from Top-K and Bot-K packs
        # [B, K, G]
        idx_top_exp = idx_top.unsqueeze(2).expand(-1, -1, groups_per_pack)
        idx_bot_exp = idx_bot.unsqueeze(2).expand(-1, -1, groups_per_pack)

        items_top = pack_contents.gather(1, idx_top_exp)
        items_bot = pack_contents.gather(1, idx_bot_exp)

        # Compute Delta for all pairs: w_top - w_bot
        # Shape: [B, K, G, K, G]
        delta = items_top.unsqueeze(3).unsqueeze(4) - items_bot.unsqueeze(1).unsqueeze(2)

        # Compute L2 Gain
        # Gain = (W_T^2 + W_B^2) - ((W_T - d)^2 + (W_B + d)^2) = 2*d*(W_T - W_B - d)
        W_T = vals_top.unsqueeze(2).unsqueeze(3).unsqueeze(4)
        W_B = vals_bot.unsqueeze(1).unsqueeze(2).unsqueeze(4)
        
        # Positive gain means reduction in sum of squares
        gain = 2 * delta * ((W_T - W_B) - delta)

        # Mask invalid swaps:
        # 1. Must be different packs (if K is large enough to overlap)
        p_top_id = idx_top.unsqueeze(2).unsqueeze(3).unsqueeze(4)
        p_bot_id = idx_bot.unsqueeze(1).unsqueeze(2).unsqueeze(4)
        valid_pair = p_top_id != p_bot_id
        
        # 2. Gain must be positive
        gain_mask = (gain > 1e-6) & valid_pair
        
        # Invalidate non-beneficial moves
        gain = torch.where(gain_mask, gain, -float('inf'))

        # Find best swap per batch
        gain_flat = gain.view(batch_size, -1)
        best_gain, best_flat_idx = gain_flat.max(dim=1)

        do_swap = (best_gain > 1e-6) & active_mask
        if not do_swap.any():
            break

        # Execute swaps for active batches
        batch_active = torch.where(do_swap)[0]
        idx_tuple = best_flat_idx[batch_active]

        # Decode indices
        KG = K * groups_per_pack
        idx_pair_top = idx_tuple // KG
        idx_pair_bot = idx_tuple % KG

        k_t = idx_pair_top // groups_per_pack
        g_t = idx_pair_top % groups_per_pack
        k_b = idx_pair_bot // groups_per_pack
        g_b = idx_pair_bot % groups_per_pack

        # Retrieve pack indices
        p_top = idx_top[batch_active, k_t]
        p_bot = idx_bot[batch_active, k_b]

        # Perform swap in contents and IDs
        val_top = pack_contents[batch_active, p_top, g_t]
        val_bot = pack_contents[batch_active, p_bot, g_b]
        
        pack_contents[batch_active, p_top, g_t] = val_bot
        pack_contents[batch_active, p_bot, g_b] = val_top
        
        id_top = pack_orig_ids[batch_active, p_top, g_t]
        id_bot = pack_orig_ids[batch_active, p_bot, g_b]
        
        pack_orig_ids[batch_active, p_top, g_t] = id_bot
        pack_orig_ids[batch_active, p_bot, g_b] = id_top

    # 4. Select Best Restart
    # Metric: Max Load - Min Load (Minimize Imbalance)
    final_pack_weights = pack_contents.sum(dim=2)
    imbalance = final_pack_weights.max(dim=1).values - final_pack_weights.min(dim=1).values
    
    # Reshape to [Layers, Restarts]
    imbalance = imbalance.view(num_layers, num_restarts)
    best_restart_idx = imbalance.argmin(dim=1)

    # Gather best results
    select_batch_idx = torch.arange(num_layers, device=device) * num_restarts + best_restart_idx
    best_ids = pack_orig_ids[select_batch_idx] # [L, P, G]

    # 5. Reconstruct Output
    pack_index = torch.empty(num_layers, num_groups, dtype=torch.int64, device=device)
    rank_in_pack = torch.empty(num_layers, num_groups, dtype=torch.int64, device=device)

    flat_best_ids = best_ids.view(num_layers, -1)

    grid_packs = torch.arange(num_packs, device=device).view(1, -1, 1).expand(num_layers, -1, groups_per_pack).reshape(num_layers, -1)
    grid_ranks = torch.arange(groups_per_pack, device=device).view(1, 1, -1).expand(num_layers, num_packs, -1).reshape(num_layers, -1)

    pack_index.scatter_(1, flat_best_ids, grid_packs)
    rank_in_pack.scatter_(1, flat_best_ids, grid_ranks)

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
    # Pre-compute scores to avoid redundant division
    current_scores = weight.float() / logcnt.float()

    for i in range(num_log, num_phy):
        redundant_indices = current_scores.argmax(dim=-1)
        phy2log[:, i] = redundant_indices
        rank[:, i] = logcnt[arangen, redundant_indices]

        # Update logcnt
        logcnt[arangen, redundant_indices] += 1

        # Incrementally update scores only for modified experts
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