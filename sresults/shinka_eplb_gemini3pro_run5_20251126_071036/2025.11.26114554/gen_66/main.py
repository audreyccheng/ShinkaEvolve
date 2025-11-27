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

    # Massive Parallel Restarts with Pruning
    # Increase to 4096 to maximize probability of finding good initial seeds
    num_restarts = 4096
    num_refined = 64

    # Expand weight for restarts: [num_layers, num_restarts, num_groups]
    weight_expanded = weight.unsqueeze(1).expand(-1, num_restarts, -1).clone()

    # Apply noise scales linearly from 0.0 to 0.20
    # This creates a diversity of sorting orders
    if num_restarts > 1:
        noise_scales = torch.linspace(0.01,
                                      0.20,
                                      steps=num_restarts - 1,
                                      device=device)
        noise = torch.rand_like(weight_expanded[:, 1:]) * noise_scales.view(
            1, -1, 1)
        weight_expanded[:, 1:] *= (1.0 + noise)

    flat_weight = weight_expanded.reshape(-1, num_groups)
    # Use original weights for accumulation
    original_w_flat = weight.unsqueeze(1).expand(-1, num_restarts, -1).reshape(
        -1, num_groups)

    # Sort descending based on perturbed weights
    sorted_indices = flat_weight.argsort(dim=-1, descending=True)
    sorted_w = original_w_flat.gather(1, sorted_indices)

    batch_size = flat_weight.shape[0]
    row_indices = torch.arange(batch_size, device=device)

    # Optimization: Transpose sorted_w to [Groups, Batch] for coalesced access
    sorted_w_t = sorted_w.t().contiguous()

    # State tracking
    pack_weights = torch.zeros(batch_size, num_packs, device=device)
    pack_counts = torch.zeros(batch_size,
                              num_packs,
                              dtype=torch.int64,
                              device=device)

    # Store results transposed [Groups, Batch] to allow coalesced writes
    pack_index_sorted_t = torch.zeros(num_groups,
                                      batch_size,
                                      dtype=torch.int64,
                                      device=device)
    rank_in_pack_sorted_t = torch.zeros(num_groups,
                                        batch_size,
                                        dtype=torch.int64,
                                        device=device)

    inf_tensor = torch.full((batch_size, num_packs),
                            float('inf'),
                            device=device)

    # Vectorized Greedy Packing
    for i in range(num_groups):
        w = sorted_w_t[i]

        valid_mask = pack_counts < groups_per_pack
        candidate_weights = torch.where(valid_mask, pack_weights, inf_tensor)
        chosen_pack = candidate_weights.argmin(dim=1)

        # Scatter add to update pack weights
        pack_weights[row_indices, chosen_pack] += w

        # Store rank (current count)
        rank_in_pack_sorted_t[i] = pack_counts[row_indices, chosen_pack]

        # Increment count
        pack_counts[row_indices, chosen_pack] += 1

        # Store assignment
        pack_index_sorted_t[i] = chosen_pack

    # Pruning: Select top-N candidates based on initial imbalance
    imbalance = pack_weights.max(dim=1).values - pack_weights.min(dim=1).values
    imbalance = imbalance.view(num_layers, num_restarts)

    _, best_restart_indices = imbalance.topk(num_refined, dim=1, largest=False)

    # Gather indices for the selected best restarts
    layer_offsets = (torch.arange(num_layers, device=device) *
                     num_restarts).unsqueeze(1)
    selected_flat_indices = (layer_offsets + best_restart_indices).flatten()

    # Subset the data for refinement
    # Gather columns from transposed structures
    sel_pack_idx = pack_index_sorted_t[:, selected_flat_indices].t()
    sel_rank_idx = rank_in_pack_sorted_t[:, selected_flat_indices].t()
    sel_sorted_w = sorted_w[selected_flat_indices]
    sel_sorted_idx = sorted_indices[selected_flat_indices]

    refined_batch_size = selected_flat_indices.size(0)
    ref_row_indices = torch.arange(refined_batch_size, device=device)

    # Construct pack_contents for refinement on the pruned set
    pack_contents = torch.zeros(refined_batch_size,
                                num_packs,
                                groups_per_pack,
                                device=device)
    pack_item_ids = torch.zeros(refined_batch_size,
                                num_packs,
                                groups_per_pack,
                                dtype=torch.int64,
                                device=device)

    flat_indices = ref_row_indices.unsqueeze(1).expand(-1,
                                                       num_groups).flatten()
    flat_pack_idx = sel_pack_idx.flatten()
    flat_rank_idx = sel_rank_idx.flatten()

    pack_contents.index_put_((flat_indices, flat_pack_idx, flat_rank_idx),
                             sel_sorted_w.flatten())
    pack_item_ids.index_put_((flat_indices, flat_pack_idx, flat_rank_idx),
                             sel_sorted_idx.flatten())

    # Refinement: Top-K / Bottom-K Swap with Hybrid 1-item and 2-item Moves
    K = min(num_packs, 4)  # Use smaller K to manage memory

    # Pre-compute for 2-item swaps
    can_do_pair_swap = (groups_per_pack >= 2) and (groups_per_pack <= 32)
    if can_do_pair_swap:
        pair_i, pair_j = torch.triu_indices(groups_per_pack,
                                            groups_per_pack,
                                            1,
                                            device=device)
        num_pairs = pair_i.shape[0]

    for _ in range(50):
        current_pack_weights = pack_contents.sum(dim=2)  # [B, P]

        vals_top, idx_top = torch.topk(current_pack_weights, k=K, largest=True)
        vals_bot, idx_bot = torch.topk(current_pack_weights,
                                       k=K,
                                       largest=False)

        diff = vals_top[:, 0] - vals_bot[:, 0]
        active_mask = diff > 1e-4
        if not active_mask.any():
            break

        # --- 1. Evaluate 1-Item Swaps ---
        gather_top = idx_top.unsqueeze(2).expand(-1, -1, groups_per_pack)
        gather_bot = idx_bot.unsqueeze(2).expand(-1, -1, groups_per_pack)

        items_top = pack_contents.gather(1, gather_top)  # [B, K, G]
        items_bot = pack_contents.gather(1, gather_bot)  # [B, K, G]

        # Delta: [B, K, G, K, G]
        delta_1 = items_top.unsqueeze(3).unsqueeze(4) - items_bot.unsqueeze(
            1).unsqueeze(2)

        W_A = vals_top.unsqueeze(2).unsqueeze(3).unsqueeze(4)
        W_B = vals_bot.unsqueeze(1).unsqueeze(2).unsqueeze(4)
        diff_packs_1 = W_A - W_B

        gain_1 = 2 * delta_1 * (diff_packs_1 - delta_1)

        p_top_exp = idx_top.unsqueeze(2).unsqueeze(3).unsqueeze(4)
        p_bot_exp = idx_bot.unsqueeze(1).unsqueeze(2).unsqueeze(4)
        valid_pair = p_top_exp != p_bot_exp

        gain_1[~valid_pair] = -float('inf')
        gain_1_flat = gain_1.view(refined_batch_size, -1)
        best_gain_1, best_idx_1 = gain_1_flat.max(dim=1)

        # --- 2. Evaluate 2-Item Swaps (Top-1 vs Bottom-1) ---
        best_gain_2 = torch.full_like(best_gain_1, -float('inf'))
        best_idx_2 = torch.zeros_like(best_idx_1)

        if can_do_pair_swap:
            # We only check 2-item swaps between best top and best bottom
            # to save memory/compute.
            t1_items = pack_contents[ref_row_indices, idx_top[:, 0]]  # [B, G]
            b1_items = pack_contents[ref_row_indices, idx_bot[:, 0]]  # [B, G]

            # [B, NumPairs]
            t1_pairs = t1_items[:, pair_i] + t1_items[:, pair_j]
            b1_pairs = b1_items[:, pair_i] + b1_items[:, pair_j]

            # [B, NumPairs, NumPairs]
            delta_2 = t1_pairs.unsqueeze(2) - b1_pairs.unsqueeze(1)
            diff_2 = (vals_top[:, 0] - vals_bot[:, 0]).view(-1, 1, 1)

            gain_2 = 2 * delta_2 * (diff_2 - delta_2)

            # Mask invalid gains
            gain_2_flat = gain_2.view(refined_batch_size, -1)
            best_gain_2, best_idx_2 = gain_2_flat.max(dim=1)

        # --- 3. Execute Best Swap ---
        use_2item = (best_gain_2 > best_gain_1) & (best_gain_2 > 1e-6) & can_do_pair_swap
        do_swap = (torch.max(best_gain_1, best_gain_2) > 1e-6) & active_mask

        if not do_swap.any():
            break

        # Execute 1-item swaps
        mask_1 = do_swap & (~use_2item)
        if mask_1.any():
            b_1 = ref_row_indices[mask_1]
            idx_tuple = best_idx_1[mask_1]

            KG = K * groups_per_pack
            idx_pair_top = idx_tuple // KG
            idx_pair_bot = idx_tuple % KG

            k_t = idx_pair_top // groups_per_pack
            g_t = idx_pair_top % groups_per_pack
            k_b = idx_pair_bot // groups_per_pack
            g_b = idx_pair_bot % groups_per_pack

            p_top = idx_top[mask_1, k_t]
            p_bot = idx_bot[mask_1, k_b]

            val_top = pack_contents[b_1, p_top, g_t]
            val_bot = pack_contents[b_1, p_bot, g_b]

            pack_contents[b_1, p_top, g_t] = val_bot
            pack_contents[b_1, p_bot, g_b] = val_top

            id_top = pack_item_ids[b_1, p_top, g_t]
            id_bot = pack_item_ids[b_1, p_bot, g_b]

            pack_item_ids[b_1, p_top, g_t] = id_bot
            pack_item_ids[b_1, p_bot, g_b] = id_top

        # Execute 2-item swaps
        mask_2 = do_swap & use_2item
        if mask_2.any():
            b_2 = ref_row_indices[mask_2]
            idx_tuple_2 = best_idx_2[mask_2]

            idx_pair_t = idx_tuple_2 // num_pairs
            idx_pair_b = idx_tuple_2 % num_pairs

            p_top_2 = idx_top[mask_2, 0]
            p_bot_2 = idx_bot[mask_2, 0]

            t_i = pair_i[idx_pair_t]
            t_j = pair_j[idx_pair_t]
            b_k = pair_i[idx_pair_b]
            b_l = pair_j[idx_pair_b]

            # Get values
            val_t_i = pack_contents[b_2, p_top_2, t_i]
            val_t_j = pack_contents[b_2, p_top_2, t_j]
            val_b_k = pack_contents[b_2, p_bot_2, b_k]
            val_b_l = pack_contents[b_2, p_bot_2, b_l]

            # Perform 2-item swap
            pack_contents[b_2, p_top_2, t_i] = val_b_k
            pack_contents[b_2, p_top_2, t_j] = val_b_l
            pack_contents[b_2, p_bot_2, b_k] = val_t_i
            pack_contents[b_2, p_bot_2, b_l] = val_t_j

            id_t_i = pack_item_ids[b_2, p_top_2, t_i]
            id_t_j = pack_item_ids[b_2, p_top_2, t_j]
            id_b_k = pack_item_ids[b_2, p_bot_2, b_k]
            id_b_l = pack_item_ids[b_2, p_bot_2, b_l]

            pack_item_ids[b_2, p_top_2, t_i] = id_b_k
            pack_item_ids[b_2, p_top_2, t_j] = id_b_l
            pack_item_ids[b_2, p_bot_2, b_k] = id_t_i
            pack_item_ids[b_2, p_bot_2, b_l] = id_t_j

    # Final Selection from Refined Set
    final_pack_weights = pack_contents.sum(dim=2)
    final_imbalance = final_pack_weights.max(
        dim=1).values - final_pack_weights.min(dim=1).values
    final_imbalance = final_imbalance.view(num_layers, num_refined)

    best_idx_in_refined = final_imbalance.argmin(dim=1)
    offsets = torch.arange(num_layers, device=device) * num_refined
    best_flat_idx = offsets + best_idx_in_refined

    best_item_ids = pack_item_ids[best_flat_idx]

    # Reconstruct Output
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