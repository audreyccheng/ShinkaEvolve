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

    Uses Greedy LPT initialization followed by an advanced L2-gain based
    local search refinement considering top-K heavy/light swaps.

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

    # 1. Greedy LPT Initialization
    # Sort weights descending
    sorted_weight, sorted_indices = weight.float().sort(dim=-1, descending=True)

    # State tracking:
    # pack_contents: [L, P, G] stores weights of items in each slot
    # pack_item_ids: [L, P, G] stores original indices of items in each slot
    pack_contents = torch.zeros(num_layers, num_packs, groups_per_pack, device=device)
    pack_item_ids = torch.zeros(num_layers, num_packs, groups_per_pack, dtype=torch.int64, device=device)

    pack_weights = torch.zeros(num_layers, num_packs, device=device)
    pack_counts = torch.zeros(num_layers, num_packs, dtype=torch.int64, device=device)

    row_indices = torch.arange(num_layers, device=device)

    # Assign items one by one
    for i in range(num_groups):
        w = sorted_weight[:, i]
        original_idx = sorted_indices[:, i]

        # Mask valid packs (not full)
        valid_mask = pack_counts < groups_per_pack

        # Find pack with min weight among valid ones
        candidate_weights = pack_weights.clone()
        candidate_weights[~valid_mask] = float('inf')

        chosen_pack = torch.argmin(candidate_weights, dim=1) # [L]

        # Get rank (current count)
        chosen_rank = pack_counts[row_indices, chosen_pack] # [L]

        # Update State
        pack_weights[row_indices, chosen_pack] += w
        pack_counts[row_indices, chosen_pack] += 1

        # Store item info
        pack_contents[row_indices, chosen_pack, chosen_rank] = w
        pack_item_ids[row_indices, chosen_pack, chosen_rank] = original_idx

    # 2. Refinement Loop (L2 Optimization)
    num_iters = 20
    # K determines how many heavy/light packs we consider for swapping.
    K = 4
    if num_packs < 2 * K:
        K = num_packs // 2

    if K > 0:
        for _ in range(num_iters):
            # Sort packs by weight to find heaviest and lightest
            # We recompute pack_weights from contents to stay consistent
            current_pack_weights = pack_contents.sum(dim=-1)

            # Get indices of packs sorted by weight
            sorted_pack_indices = current_pack_weights.argsort(dim=1)

            # Select K heavy and K light packs
            # heavy_packs indices: [L, K] (heaviest at end)
            heavy_packs = sorted_pack_indices[:, -K:]
            light_packs = sorted_pack_indices[:, :K]

            w_packs_heavy = torch.gather(current_pack_weights, 1, heavy_packs)
            w_packs_light = torch.gather(current_pack_weights, 1, light_packs)

            # Pack Diff: [L, K, K]
            # Difference between heavy pack i and light pack j
            # P = W_H - W_L
            pack_diff = w_packs_heavy.unsqueeze(2) - w_packs_light.unsqueeze(1)

            # Gather Items: [L, K, G]
            idx_heavy_expanded = heavy_packs.unsqueeze(2).expand(-1, -1, groups_per_pack)
            idx_light_expanded = light_packs.unsqueeze(2).expand(-1, -1, groups_per_pack)

            items_heavy = torch.gather(pack_contents, 1, idx_heavy_expanded)
            items_light = torch.gather(pack_contents, 1, idx_light_expanded)

            # Compute Delta for all pairs of items between these packs
            # delta = weight_item_heavy - weight_item_light
            # Shape: [L, K (heavy), G, K (light), G]
            delta = items_heavy.view(num_layers, K, groups_per_pack, 1, 1) - \
                    items_light.view(num_layers, 1, 1, K, groups_per_pack)

            # Improvement Gain (L2 Objective)
            # Minimize Sum(Squares). Maximize Gain = 2 * delta * (PackDiff - delta)
            pd_view = pack_diff.view(num_layers, K, 1, K, 1)
            gain = 2 * delta * (pd_view - delta)

            # Find best swap
            gain_flat = gain.view(num_layers, -1)
            best_gain, best_idx_flat = gain_flat.max(dim=1)

            # Threshold for improvement
            active_mask = best_gain > 1e-6
            if not active_mask.any():
                break

            # Perform Swaps for layers with improvement
            l_idx = torch.where(active_mask)[0]
            b_idx = best_idx_flat[l_idx]

            # Decode flattened indices
            # Structure: K_heavy * G * K_light * G
            total_G = groups_per_pack
            total_KG = K * total_G

            idx_item_l = b_idx % total_G
            b_idx = b_idx // total_G
            idx_pack_l = b_idx % K
            b_idx = b_idx // K
            idx_item_h = b_idx % total_G
            idx_pack_h = b_idx // total_G

            # Get actual pack indices
            p_h = heavy_packs[l_idx, idx_pack_h]
            p_l = light_packs[l_idx, idx_pack_l]

            # Swap values in pack_contents
            val_h = pack_contents[l_idx, p_h, idx_item_h]
            val_l = pack_contents[l_idx, p_l, idx_item_l]

            pack_contents[l_idx, p_h, idx_item_h] = val_l
            pack_contents[l_idx, p_l, idx_item_l] = val_h

            # Swap IDs in pack_item_ids
            id_h = pack_item_ids[l_idx, p_h, idx_item_h]
            id_l = pack_item_ids[l_idx, p_l, idx_item_l]

            pack_item_ids[l_idx, p_h, idx_item_h] = id_l
            pack_item_ids[l_idx, p_l, idx_item_l] = id_h

    # 3. Reconstruction
    pack_index = torch.empty(num_layers, num_groups, dtype=torch.int64, device=device)
    rank_in_pack = torch.empty(num_layers, num_groups, dtype=torch.int64, device=device)

    # Generate grid of Pack IDs and Ranks corresponding to the shape of pack_item_ids
    # pack_ids grid: same for all layers
    grid_packs = torch.arange(num_packs, device=device).view(1, num_packs, 1).expand(num_layers, -1, groups_per_pack).reshape(num_layers, -1)
    grid_ranks = torch.arange(groups_per_pack, device=device).view(1, 1, groups_per_pack).expand(num_layers, num_packs, -1).reshape(num_layers, -1)

    # Flat item IDs tells us which item is at which position
    flat_ids = pack_item_ids.view(num_layers, -1)

    # Scatter results back to output tensors
    # pack_index[layer, item_id] = pack_id
    pack_index.scatter_(1, flat_ids, grid_packs)
    rank_in_pack.scatter_(1, flat_ids, grid_ranks)

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