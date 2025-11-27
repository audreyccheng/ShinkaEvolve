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

    # Configuration
    num_restarts = 4096
    num_refined = 64

    # --- 1. Parallel Randomized Initialization ---

    # Expand weights: [Layers, Restarts, Groups]
    weight_expanded = weight.unsqueeze(1).expand(-1, num_restarts, -1).clone()

    # Apply multiplicative noise (linear scale 0.01 -> 0.20)
    if num_restarts > 1:
        noise_scales = torch.linspace(0.01, 0.20, steps=num_restarts - 1, device=device)
        noise = torch.rand_like(weight_expanded[:, 1:]) * noise_scales.view(1, -1, 1)
        weight_expanded[:, 1:] *= (1.0 + noise)

    flat_weight_perturbed = weight_expanded.reshape(-1, num_groups)

    # Use original weights for actual summing to avoid drift
    flat_weight_original = weight.unsqueeze(1).expand(-1, num_restarts, -1).reshape(-1, num_groups)

    # Sort based on perturbed weights (LPT)
    sorted_indices = flat_weight_perturbed.argsort(dim=-1, descending=True)
    sorted_w = flat_weight_original.gather(1, sorted_indices)

    batch_size = flat_weight_original.shape[0]

    # --- 2. Vectorized Greedy Packing ---

    # Transpose for coalesced memory access during loop: [Groups, Batch]
    sorted_w_t = sorted_w.t().contiguous()

    pack_weights = torch.zeros(batch_size, num_packs, device=device)
    pack_counts = torch.zeros(batch_size, num_packs, dtype=torch.int64, device=device)

    # Store decisions in temporary arrays [Groups, Batch]
    # We store the PACK index assigned to the i-th sorted item
    pack_decisions_t = torch.zeros(num_groups, batch_size, dtype=torch.int64, device=device)
    rank_decisions_t = torch.zeros(num_groups, batch_size, dtype=torch.int64, device=device)

    row_indices = torch.arange(batch_size, device=device)
    inf_tensor = torch.full((batch_size, num_packs), float('inf'), device=device)

    for i in range(num_groups):
        w = sorted_w_t[i]

        valid_mask = pack_counts < groups_per_pack
        # Choose min weight pack among valid ones
        costs = torch.where(valid_mask, pack_weights, inf_tensor)
        chosen_pack = costs.argmin(dim=1)

        # Update
        pack_weights[row_indices, chosen_pack] += w
        rank_decisions_t[i] = pack_counts[row_indices, chosen_pack]
        pack_counts[row_indices, chosen_pack] += 1
        pack_decisions_t[i] = chosen_pack

    # --- 3. Pruning ---

    imbalance = pack_weights.max(dim=1).values - pack_weights.min(dim=1).values
    imbalance = imbalance.view(num_layers, num_restarts)

    # Select top candidates
    _, best_restart_indices = imbalance.topk(num_refined, dim=1, largest=False)

    # Flatten selection indices
    layer_offsets = (torch.arange(num_layers, device=device) * num_restarts).unsqueeze(1)
    selected_flat = (layer_offsets + best_restart_indices).flatten()

    refined_batch_size = selected_flat.size(0)
    ref_row_indices = torch.arange(refined_batch_size, device=device)

    # Gather data for refinement
    # [Batch, Groups] <- [Groups, Batch]
    sel_pack_idx = pack_decisions_t[:, selected_flat].t()
    sel_rank_idx = rank_decisions_t[:, selected_flat].t()
    sel_sorted_w = sorted_w[selected_flat]
    sel_sorted_idx = sorted_indices[selected_flat]

    # Structure for refinement: [Batch, Packs, GroupsPerPack]
    pack_contents = torch.zeros(refined_batch_size, num_packs, groups_per_pack, device=device)
    pack_item_ids = torch.zeros(refined_batch_size, num_packs, groups_per_pack, dtype=torch.int64, device=device)

    flat_b = ref_row_indices.unsqueeze(1).expand(-1, num_groups).flatten()
    flat_p = sel_pack_idx.flatten()
    flat_r = sel_rank_idx.flatten()

    pack_contents.index_put_((flat_b, flat_p, flat_r), sel_sorted_w.flatten())
    pack_item_ids.index_put_((flat_b, flat_p, flat_r), sel_sorted_idx.flatten())

    # --- 4. Two-Stage Refinement ---

    # Pre-calculate pair indices for 2-for-2 swap (if applicable)
    can_do_2for2 = (groups_per_pack >= 2) and (groups_per_pack <= 32)
    if can_do_2for2:
        triu_r, triu_c = torch.triu_indices(groups_per_pack, groups_per_pack, offset=1, device=device)

    K_swap = min(num_packs, 4)

    for _ in range(30):
        # Update weights
        current_weights = pack_contents.sum(dim=2)

        # --- Phase 1: 1-for-1 Swaps (Top-K vs Bottom-K) ---
        vals_top, idx_top = torch.topk(current_weights, k=K_swap, largest=True)
        vals_bot, idx_bot = torch.topk(current_weights, k=K_swap, largest=False)

        diff = vals_top[:, 0] - vals_bot[:, 0]
        active_mask = diff > 1e-4
        if not active_mask.any():
            break

        # Extract items [B, K, G]
        items_top = pack_contents.gather(1, idx_top.unsqueeze(2).expand(-1, -1, groups_per_pack))
        items_bot = pack_contents.gather(1, idx_bot.unsqueeze(2).expand(-1, -1, groups_per_pack))

        # 1-for-1 Delta: [B, K_top, G, K_bot, G]
        # Compare every item in Top-K packs with every item in Bottom-K packs
        delta_1 = items_top.unsqueeze(3).unsqueeze(4) - items_bot.unsqueeze(1).unsqueeze(2)

        # Target reduction
        diff_packs = vals_top.unsqueeze(2).unsqueeze(3).unsqueeze(4) - vals_bot.unsqueeze(1).unsqueeze(2).unsqueeze(4)
        gain_1 = 2 * delta_1 * (diff_packs - delta_1)

        # Mask invalid pairs (same pack or non-positive gain)
        p_top_exp = idx_top.unsqueeze(2).unsqueeze(3).unsqueeze(4)
        p_bot_exp = idx_bot.unsqueeze(1).unsqueeze(2).unsqueeze(4)
        gain_1[(p_top_exp == p_bot_exp) | (gain_1 < 1e-6)] = -1.0

        best_gain_1, best_idx_1 = gain_1.view(refined_batch_size, -1).max(dim=1)

        # --- Phase 2: 2-for-2 Swaps (Max vs Min) ---
        # Only check Max vs Min to save compute

        best_gain_2 = torch.full_like(best_gain_1, -1.0)
        best_idx_2 = torch.zeros_like(best_idx_1)

        if can_do_2for2:
            p_max = idx_top[:, 0]
            p_min = idx_bot[:, 0]
            val_max = vals_top[:, 0]
            val_min = vals_bot[:, 0]

            # Gather items [B, G]
            items_max = pack_contents[ref_row_indices, p_max]
            items_min = pack_contents[ref_row_indices, p_min]

            # Compute pair sums [B, N_pairs]
            pair_sum_max = items_max[:, triu_r] + items_max[:, triu_c]
            pair_sum_min = items_min[:, triu_r] + items_min[:, triu_c]

            # Delta [B, N_p_max, N_p_min]
            delta_2 = pair_sum_max.unsqueeze(2) - pair_sum_min.unsqueeze(1)

            diff_2 = (val_max - val_min).view(-1, 1, 1)

            # Maximize improvement
            gain_2_vals = diff_2 - (diff_2 - 2 * delta_2).abs()
            gain_2_vals[delta_2 <= 0] = -1.0

            best_gain_2, best_idx_2 = gain_2_vals.view(refined_batch_size, -1).max(dim=1)

        # --- Decision ---
        use_2 = (best_gain_2 > best_gain_1) & (best_gain_2 > 1e-6)
        use_1 = (~use_2) & (best_gain_1 > 1e-6)

        do_any = (use_1 | use_2) & active_mask
        if not do_any.any():
            break

        # Execute 1-for-1
        if use_1.any():
            b_1 = ref_row_indices[use_1]
            idx_flat = best_idx_1[use_1]

            KG = K_swap * groups_per_pack
            k_t = (idx_flat // (KG * groups_per_pack))
            rem = idx_flat % (KG * groups_per_pack)
            g_t = rem // (KG)
            rem2 = rem % (KG)
            k_b = rem2 // groups_per_pack
            g_b = rem2 % groups_per_pack

            p_t = idx_top[b_1, k_t]
            p_b = idx_bot[b_1, k_b]

            v_t = pack_contents[b_1, p_t, g_t]
            v_b = pack_contents[b_1, p_b, g_b]
            pack_contents[b_1, p_t, g_t] = v_b
            pack_contents[b_1, p_b, g_b] = v_t

            i_t = pack_item_ids[b_1, p_t, g_t]
            i_b = pack_item_ids[b_1, p_b, g_b]
            pack_item_ids[b_1, p_t, g_t] = i_b
            pack_item_ids[b_1, p_b, g_b] = i_t

        # Execute 2-for-2
        if use_2.any():
            b_2 = ref_row_indices[use_2]
            idx_flat = best_idx_2[use_2]

            num_pairs = triu_r.size(0)
            idx_pair_max = idx_flat // num_pairs
            idx_pair_min = idx_flat % num_pairs

            u1 = triu_r[idx_pair_max]
            u2 = triu_c[idx_pair_max]
            v1 = triu_r[idx_pair_min]
            v2 = triu_c[idx_pair_min]

            pm = idx_top[b_2, 0]
            pn = idx_bot[b_2, 0]

            # Swap items u1 <-> v1
            val_u1 = pack_contents[b_2, pm, u1]
            val_v1 = pack_contents[b_2, pn, v1]
            pack_contents[b_2, pm, u1] = val_v1
            pack_contents[b_2, pn, v1] = val_u1

            id_u1 = pack_item_ids[b_2, pm, u1]
            id_v1 = pack_item_ids[b_2, pn, v1]
            pack_item_ids[b_2, pm, u1] = id_v1
            pack_item_ids[b_2, pn, v1] = id_u1

            # Swap items u2 <-> v2
            val_u2 = pack_contents[b_2, pm, u2]
            val_v2 = pack_contents[b_2, pn, v2]
            pack_contents[b_2, pm, u2] = val_v2
            pack_contents[b_2, pn, v2] = val_u2

            id_u2 = pack_item_ids[b_2, pm, u2]
            id_v2 = pack_item_ids[b_2, pn, v2]
            pack_item_ids[b_2, pm, u2] = id_v2
            pack_item_ids[b_2, pn, v2] = id_u2

    # --- 5. Final Selection ---

    final_weights = pack_contents.sum(dim=2)
    final_imbalance = final_weights.max(dim=1).values - final_weights.min(dim=1).values
    final_imbalance = final_imbalance.view(num_layers, num_refined)

    best_idx_in_refined = final_imbalance.argmin(dim=1)
    offsets = torch.arange(num_layers, device=device) * num_refined
    best_flat_idx = offsets + best_idx_in_refined

    best_item_ids = pack_item_ids[best_flat_idx]

    # Reconstruct Output
    pack_index = torch.empty(num_layers, num_groups, dtype=torch.int64, device=device)
    rank_in_pack = torch.empty(num_layers, num_groups, dtype=torch.int64, device=device)

    flat_item_ids = best_item_ids.view(num_layers, -1)
    grid_packs = torch.arange(num_packs, device=device).view(1, -1, 1).expand(num_layers, -1, groups_per_pack).reshape(num_layers, -1)
    grid_ranks = torch.arange(groups_per_pack, device=device).view(1, 1, -1).expand(num_layers, num_packs, -1).reshape(num_layers, -1)

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