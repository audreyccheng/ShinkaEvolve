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

    # Randomized Restarts with 2-Swap Refinement
    # Reduce restarts to 256 to budget for expensive 2-item swaps
    num_restarts = 256
    num_refined = 8  # Only refine the very best candidates per layer

    # 1. Expand and Perturb Weights
    weight_expanded = weight.unsqueeze(1).expand(-1, num_restarts, -1).clone()
    noise_scales = torch.linspace(0.0, 0.20, steps=num_restarts, device=device)
    noise = torch.rand_like(weight_expanded) * noise_scales.view(1, -1, 1)
    weight_expanded *= (1.0 + noise)

    flat_weight = weight_expanded.reshape(-1, num_groups)
    original_w_flat = weight.unsqueeze(1).expand(-1, num_restarts, -1).reshape(
        -1, num_groups)

    # Sort descending
    sorted_indices = flat_weight.argsort(dim=-1, descending=True)
    sorted_w = original_w_flat.gather(1, sorted_indices)

    batch_size = flat_weight.shape[0]
    row_indices = torch.arange(batch_size, device=device)

    # 2. Vectorized Greedy Packing
    pack_weights = torch.zeros(batch_size, num_packs, device=device)
    pack_counts = torch.zeros(batch_size,
                              num_packs,
                              dtype=torch.int64,
                              device=device)
    # Track results
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

    for i in range(num_groups):
        w = sorted_w[:, i]
        valid_mask = pack_counts < groups_per_pack
        candidate_weights = torch.where(valid_mask, pack_weights, inf_tensor)
        chosen_pack = candidate_weights.argmin(dim=1)

        pack_weights[row_indices, chosen_pack] += w
        rank_in_pack_sorted[:, i] = pack_counts[row_indices, chosen_pack]
        pack_counts[row_indices, chosen_pack] += 1
        pack_index_sorted[:, i] = chosen_pack

    # 3. Pruning
    imbalance = pack_weights.max(dim=1).values - pack_weights.min(dim=1).values
    imbalance = imbalance.view(num_layers, num_restarts)
    _, best_restart_indices = imbalance.topk(num_refined, dim=1, largest=False)

    layer_offsets = (torch.arange(num_layers, device=device) *
                     num_restarts).unsqueeze(1)
    selected_flat_indices = (layer_offsets + best_restart_indices).flatten()

    sel_sorted_w = sorted_w[selected_flat_indices]
    sel_sorted_idx = sorted_indices[selected_flat_indices]
    sel_pack_idx = pack_index_sorted[selected_flat_indices]
    sel_rank_idx = rank_in_pack_sorted[selected_flat_indices]

    refined_batch_size = selected_flat_indices.size(0)
    ref_row_indices = torch.arange(refined_batch_size, device=device)

    # Fill pack contents for refinement
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

    # 4. Advanced Refinement (1-Swap and 2-Swap)
    use_2swap = (groups_per_pack >= 2) and (groups_per_pack <= 24)
    if use_2swap:
        triu_r, triu_c = torch.triu_indices(groups_per_pack,
                                            groups_per_pack,
                                            offset=1,
                                            device=device)
        num_pairs = triu_r.size(0)

    for _ in range(50):
        # Identify heaviest and lightest packs
        p_weights = pack_contents.sum(dim=2)
        val_max, idx_max = p_weights.max(dim=1)
        val_min, idx_min = p_weights.min(dim=1)

        diff = val_max - val_min
        active_mask = diff > 1e-4
        if not active_mask.any():
            break

        # Gather items
        items_max = pack_contents[ref_row_indices, idx_max]  # [B, G]
        items_min = pack_contents[ref_row_indices, idx_min]  # [B, G]

        # --- 1-Swap Logic ---
        # delta = max[i] - min[j]
        delta_1 = items_max.unsqueeze(2) - items_min.unsqueeze(1)  # [B, G, G]

        # We want to minimize |diff - 2*delta|.
        # Improvement = diff - |diff - 2*delta|
        target = diff.view(-1, 1, 1)
        imp_1 = target - (target - 2 * delta_1).abs()

        # Valid if delta > 0 (strictly moves weight from max to min)
        mask_1 = delta_1 > 0
        imp_1[~mask_1] = -float('inf')

        best_imp_1, best_idx_1 = imp_1.view(refined_batch_size, -1).max(dim=1)

        best_imp_total = best_imp_1
        best_type = torch.zeros(refined_batch_size,
                                dtype=torch.int64,
                                device=device)  # 0 for 1-swap

        # --- 2-Swap Logic ---
        if use_2swap:
            # Pairs: [B, num_pairs]
            pair_sum_max = items_max[:, triu_r] + items_max[:, triu_c]
            pair_sum_min = items_min[:, triu_r] + items_min[:, triu_c]

            delta_2 = pair_sum_max.unsqueeze(2) - pair_sum_min.unsqueeze(
                1)  # [B, P, P]

            imp_2 = target - (target - 2 * delta_2).abs()
            mask_2 = delta_2 > 0
            imp_2[~mask_2] = -float('inf')

            best_imp_2, best_idx_2 = imp_2.view(refined_batch_size,
                                                -1).max(dim=1)

            # Compare with 1-swap
            better_2 = best_imp_2 > best_imp_total
            best_imp_total = torch.where(better_2, best_imp_2, best_imp_total)
            best_type = torch.where(better_2, torch.tensor(1, device=device),
                                    best_type)

        # Check threshold
        do_swap = (best_imp_total > 1e-6) & active_mask
        if not do_swap.any():
            break

        # Apply Swaps
        # --- Apply 1-Swaps ---
        mask_apply_1 = do_swap & (best_type == 0)
        if mask_apply_1.any():
            b_idx = ref_row_indices[mask_apply_1]
            flat = best_idx_1[mask_apply_1]

            u = flat // groups_per_pack
            v = flat % groups_per_pack

            p_h = idx_max[mask_apply_1]
            p_l = idx_min[mask_apply_1]

            val_h = pack_contents[b_idx, p_h, u]
            val_l = pack_contents[b_idx, p_l, v]
            pack_contents[b_idx, p_h, u] = val_l
            pack_contents[b_idx, p_l, v] = val_h

            id_h = pack_item_ids[b_idx, p_h, u]
            id_l = pack_item_ids[b_idx, p_l, v]
            pack_item_ids[b_idx, p_h, u] = id_l
            pack_item_ids[b_idx, p_l, v] = id_h

        # --- Apply 2-Swaps ---
        if use_2swap:
            mask_apply_2 = do_swap & (best_type == 1)
            if mask_apply_2.any():
                b_idx = ref_row_indices[mask_apply_2]
                flat = best_idx_2[mask_apply_2]

                idx_pair_max = flat // num_pairs
                idx_pair_min = flat % num_pairs

                u1 = triu_r[idx_pair_max]
                u2 = triu_c[idx_pair_max]
                v1 = triu_r[idx_pair_min]
                v2 = triu_c[idx_pair_min]

                p_h = idx_max[mask_apply_2]
                p_l = idx_min[mask_apply_2]

                # Swap u1 <-> v1
                val_u1 = pack_contents[b_idx, p_h, u1]
                val_v1 = pack_contents[b_idx, p_l, v1]
                pack_contents[b_idx, p_h, u1] = val_v1
                pack_contents[b_idx, p_l, v1] = val_u1

                id_u1 = pack_item_ids[b_idx, p_h, u1]
                id_v1 = pack_item_ids[b_idx, p_l, v1]
                pack_item_ids[b_idx, p_h, u1] = id_v1
                pack_item_ids[b_idx, p_l, v1] = id_u1

                # Swap u2 <-> v2
                val_u2 = pack_contents[b_idx, p_h, u2]
                val_v2 = pack_contents[b_idx, p_l, v2]
                pack_contents[b_idx, p_h, u2] = val_v2
                pack_contents[b_idx, p_l, v2] = val_u2

                id_u2 = pack_item_ids[b_idx, p_h, u2]
                id_v2 = pack_item_ids[b_idx, p_l, v2]
                pack_item_ids[b_idx, p_h, u2] = id_v2
                pack_item_ids[b_idx, p_l, v2] = id_u2

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