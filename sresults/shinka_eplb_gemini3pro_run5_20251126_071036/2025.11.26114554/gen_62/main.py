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

    # Run multiple random restarts in parallel
    num_restarts = 12

    # Expand weight for restarts: [num_layers, num_restarts, num_groups]
    weight_expanded = weight.unsqueeze(1).expand(-1, num_restarts, -1).clone()

    # Perturb weights with varied noise levels to randomize sorting order
    # Use a linear spread of noise scales from 0.01 to 0.15
    if num_restarts > 1:
        noise_scales = torch.linspace(0.01,
                                      0.15,
                                      steps=num_restarts - 1,
                                      device=device)
        noise = torch.rand_like(weight_expanded[:, 1:]) * noise_scales.view(
            1, -1, 1)
        weight_expanded[:, 1:] *= (1.0 + noise)

    flat_weight = weight_expanded.reshape(-1, num_groups)
    # Use original weights for accumulation (repeated for each restart)
    original_w_flat = weight.unsqueeze(1).expand(-1, num_restarts, -1).reshape(
        -1, num_groups)

    # Sort descending based on perturbed weights
    sorted_indices = flat_weight.argsort(dim=-1, descending=True)
    sorted_w = original_w_flat.gather(1, sorted_indices)

    batch_size = flat_weight.shape[0]
    row_indices = torch.arange(batch_size, device=device)

    # Tracking state
    pack_weights = torch.zeros(batch_size, num_packs, device=device)
    pack_counts = torch.zeros(batch_size,
                              num_packs,
                              dtype=torch.int64,
                              device=device)

    pack_index_sorted = torch.zeros(batch_size,
                                    num_groups,
                                    dtype=torch.int64,
                                    device=device)
    rank_in_pack_sorted = torch.zeros(batch_size,
                                      num_groups,
                                      dtype=torch.int64,
                                      device=device)

    inf_tensor = torch.full((batch_size, num_packs),
                            float('inf'),
                            device=device)

    # Vectorized Greedy Packing
    for i in range(num_groups):
        w = sorted_w[:, i]

        valid_mask = pack_counts < groups_per_pack
        # Use torch.where to avoid cloning large tensors repeatedly
        candidate_weights = torch.where(valid_mask, pack_weights, inf_tensor)

        chosen_pack = candidate_weights.argmin(dim=1)

        pack_weights[row_indices, chosen_pack] += w
        rank_in_pack_sorted[:, i] = pack_counts[row_indices, chosen_pack]
        pack_counts[row_indices, chosen_pack] += 1
        pack_index_sorted[:, i] = chosen_pack

    # Refinement: All-Pairs Swap minimizing L2 norm (Sum of Squared Weights)
    # Construct pack_contents for easy access: [Batch, num_packs, groups_per_pack]
    pack_contents = torch.zeros(batch_size,
                                num_packs,
                                groups_per_pack,
                                device=device)
    pack_item_ids = torch.zeros(batch_size,
                                num_packs,
                                groups_per_pack,
                                dtype=torch.int64,
                                device=device)

    flat_indices = row_indices.unsqueeze(1).expand(-1, num_groups).flatten()
    flat_pack_idx = pack_index_sorted.flatten()
    flat_rank_idx = rank_in_pack_sorted.flatten()

    pack_contents.index_put_((flat_indices, flat_pack_idx, flat_rank_idx),
                             sorted_w.flatten())
    pack_item_ids.index_put_((flat_indices, flat_pack_idx, flat_rank_idx),
                             sorted_indices.flatten())

    for _ in range(30):
        current_pack_weights = pack_contents.sum(dim=2)  # [B, P]

        # Check convergence
        max_w = current_pack_weights.max(dim=1).values
        min_w = current_pack_weights.min(dim=1).values
        diff_max_min = max_w - min_w
        active_mask = diff_max_min > 1e-4
        if not active_mask.any():
            break

        # Compute Weight Difference between all pairs of packs: W_u - W_v
        # [B, P, 1] - [B, 1, P] -> [B, P, P]
        diff_matrix = current_pack_weights.unsqueeze(2) - current_pack_weights.unsqueeze(1)

        # Compute Item Difference between all pairs of items from all pairs of packs
        # items: [B, P, G]
        # [B, P, G, 1, 1] - [B, 1, 1, P, G] -> [B, P, G, P, G]
        # We perform subtraction. delta = item_u - item_v
        delta = pack_contents.unsqueeze(3).unsqueeze(4) - pack_contents.unsqueeze(1).unsqueeze(2)

        # Compute Gain: 2 * delta * (diff - delta)
        # diff_matrix needs expansion to [B, P, 1, P, 1]
        target = diff_matrix.unsqueeze(2).unsqueeze(4)
        gain = 2 * delta * (target - delta)

        # Mask invalid swaps:
        # 1. Gain must be positive
        # 2. If p_u == p_v, diff=0, gain = -2*delta^2 <= 0. So diagonal is handled.
        gain_mask = (gain > 1e-6)

        # Optimization: We can zero out gains where mask is false to avoid NaN issues if any
        # though we init with valid math. Just setting to -inf for argmax.
        gain[~gain_mask] = -float('inf')

        # Find best swap per batch
        # Flatten the last 4 dims: P*G*P*G
        gain_flat = gain.view(batch_size, -1)
        best_gain, best_flat_idx = gain_flat.max(dim=1)

        do_swap = (best_gain > 1e-6) & active_mask
        if not do_swap.any():
            break

        # Decode indices
        batch_active = row_indices[do_swap]
        idx_tuple = best_flat_idx[do_swap]

        PG = num_packs * groups_per_pack
        G = groups_per_pack

        # indices in flattened [P, G, P, G] space
        # idx_tuple = ((p_u * G + g_u) * P + p_v) * G + g_v

        g_v = idx_tuple % G
        rem = idx_tuple // G
        p_v = rem % num_packs
        rem = rem // num_packs
        g_u = rem % G
        p_u = rem // G

        # Perform swap
        val_u = pack_contents[batch_active, p_u, g_u]
        val_v = pack_contents[batch_active, p_v, g_v]

        pack_contents[batch_active, p_u, g_u] = val_v
        pack_contents[batch_active, p_v, g_v] = val_u

        id_u = pack_item_ids[batch_active, p_u, g_u]
        id_v = pack_item_ids[batch_active, p_v, g_v]

        pack_item_ids[batch_active, p_u, g_u] = id_v
        pack_item_ids[batch_active, p_v, g_v] = id_u

    # Select best restart per layer
    final_pack_weights = pack_contents.sum(dim=2)
    imbalance = final_pack_weights.max(
        dim=1).values - final_pack_weights.min(dim=1).values
    imbalance = imbalance.view(num_layers, num_restarts)

    best_restart_idx = imbalance.argmin(dim=1)

    best_batch_idx = torch.arange(
        num_layers, device=device) * num_restarts + best_restart_idx
    best_item_ids = pack_item_ids[best_batch_idx]  # [L, P, G]

    # Scatter back to output format
    pack_index = torch.empty(num_layers,
                             num_groups,
                             dtype=torch.int64,
                             device=device)
    rank_in_pack = torch.empty(num_layers,
                               num_groups,
                               dtype=torch.int64,
                               device=device)

    flat_item_ids = best_item_ids.view(num_layers, -1)
    grid_packs = torch.arange(num_packs, device=device).view(
        1, -1, 1).expand(num_layers, -1,
                         groups_per_pack).reshape(num_layers, -1)
    grid_ranks = torch.arange(groups_per_pack, device=device).view(
        1, 1, -1).expand(num_layers, num_packs, -1).reshape(num_layers, -1)

    pack_index.scatter_(1, flat_item_ids, grid_packs)
    rank_in_pack.scatter_(1, flat_item_ids, grid_ranks)

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