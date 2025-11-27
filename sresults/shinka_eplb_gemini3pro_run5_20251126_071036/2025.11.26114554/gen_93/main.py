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
    num_restarts = 8
    
    # Expand for parallel restarts
    # [L, R, G]
    weight_expanded = weight.unsqueeze(1).expand(-1, num_restarts, -1).clone()
    
    # Perturb weights for diversity (Restart 0 is deterministic)
    if num_restarts > 1:
        # Linear noise schedule from 1% to 15%
        noise_scales = torch.linspace(0.01, 0.15, steps=num_restarts - 1, device=device)
        noise = torch.rand_like(weight_expanded[:, 1:]) * noise_scales.view(1, -1, 1)
        weight_expanded[:, 1:] *= (1.0 + noise)

    # Flatten for processing: [Batch, G] where Batch = L * R
    batch_size = num_layers * num_restarts
    flat_perturbed = weight_expanded.reshape(batch_size, num_groups)
    flat_original = weight.unsqueeze(1).expand(-1, num_restarts, -1).reshape(batch_size, num_groups)
    
    # Sort LPT
    sorted_indices = flat_perturbed.argsort(dim=-1, descending=True)
    sorted_w = flat_original.gather(1, sorted_indices)
    
    row_indices = torch.arange(batch_size, device=device)

    # --- Step 1: Vectorized Greedy Initialization ---
    pack_weights = torch.zeros(batch_size, num_packs, device=device)
    pack_counts = torch.zeros(batch_size, num_packs, dtype=torch.int64, device=device)
    
    # Storage for results in sorted order
    pack_index_sorted = torch.zeros(batch_size, num_groups, dtype=torch.int64, device=device)
    rank_in_pack_sorted = torch.zeros(batch_size, num_groups, dtype=torch.int64, device=device)
    
    inf_tensor = torch.full((batch_size, num_packs), float('inf'), device=device)

    for i in range(num_groups):
        w = sorted_w[:, i]
        
        # Mask full packs
        valid_mask = pack_counts < groups_per_pack
        candidate_weights = torch.where(valid_mask, pack_weights, inf_tensor)
        
        chosen_pack = candidate_weights.argmin(dim=1)
        
        # Update
        pack_weights[row_indices, chosen_pack] += w
        rank_in_pack_sorted[:, i] = pack_counts[row_indices, chosen_pack]
        pack_counts[row_indices, chosen_pack] += 1
        pack_index_sorted[:, i] = chosen_pack

    # --- Step 2: Refinement Setup ---
    # Construct pack_contents: [Batch, Packs, GroupsPerPack]
    pack_contents = torch.zeros(batch_size, num_packs, groups_per_pack, device=device)
    pack_item_ids = torch.zeros(batch_size, num_packs, groups_per_pack, dtype=torch.int64, device=device)
    
    flat_b = row_indices.unsqueeze(1).expand(-1, num_groups).flatten()
    flat_p = pack_index_sorted.flatten()
    flat_r = rank_in_pack_sorted.flatten()
    
    pack_contents.index_put_((flat_b, flat_p, flat_r), sorted_w.flatten())
    pack_item_ids.index_put_((flat_b, flat_p, flat_r), sorted_indices.flatten())

    # --- Refinement Helper Functions ---
    
    def run_1_swap_pass(p_contents, p_ids, iterations=10, K=4):
        """Standard 1-for-1 swap between Top-K and Bottom-K packs."""
        if K > num_packs // 2: K = num_packs // 2
        if K == 0: return p_contents, p_ids

        for _ in range(iterations):
            p_weights = p_contents.sum(dim=2)
            vals_top, idx_top = torch.topk(p_weights, k=K, largest=True)
            vals_bot, idx_bot = torch.topk(p_weights, k=K, largest=False)
            
            diff = vals_top[:, 0] - vals_bot[:, 0]
            active_mask = diff > 1e-4
            if not active_mask.any(): break
            
            # Gather items: [B, K, G]
            idx_top_exp = idx_top.unsqueeze(2).expand(-1, -1, groups_per_pack)
            idx_bot_exp = idx_bot.unsqueeze(2).expand(-1, -1, groups_per_pack)
            
            items_top = p_contents.gather(1, idx_top_exp)
            items_bot = p_contents.gather(1, idx_bot_exp)
            
            # Delta [B, K, G, K, G]
            delta = items_top.unsqueeze(3).unsqueeze(4) - items_bot.unsqueeze(1).unsqueeze(2)
            
            # Gain L2
            W_A = vals_top.unsqueeze(2).unsqueeze(3).unsqueeze(4)
            W_B = vals_bot.unsqueeze(1).unsqueeze(2).unsqueeze(4)
            gain = 2 * delta * ((W_A - W_B) - delta)
            
            # Mask invalid
            p_top_id = idx_top.unsqueeze(2).unsqueeze(3).unsqueeze(4)
            p_bot_id = idx_bot.unsqueeze(1).unsqueeze(2).unsqueeze(4)
            mask = (gain > 1e-5) & (p_top_id != p_bot_id) & active_mask.view(-1, 1, 1, 1, 1)
            
            gain[~mask] = -float('inf')
            
            best_gain, best_flat = gain.view(batch_size, -1).max(dim=1)
            do_swap = best_gain > -float('inf')
            if not do_swap.any(): break
            
            # Perform swap on active batches
            b_idx = row_indices[do_swap]
            flat_idx = best_flat[do_swap]
            
            # Decode indices
            KG = K * groups_per_pack
            i_top_flat = flat_idx // KG
            i_bot_flat = flat_idx % KG
            
            k_t, g_t = i_top_flat // groups_per_pack, i_top_flat % groups_per_pack
            k_b, g_b = i_bot_flat // groups_per_pack, i_bot_flat % groups_per_pack
            
            p_t = idx_top[b_idx, k_t]
            p_b = idx_bot[b_idx, k_b]
            
            # Swap values
            v_t = p_contents[b_idx, p_t, g_t]
            v_b = p_contents[b_idx, p_b, g_b]
            p_contents[b_idx, p_t, g_t] = v_b
            p_contents[b_idx, p_b, g_b] = v_t
            
            # Swap IDs
            id_t = p_ids[b_idx, p_t, g_t]
            id_b = p_ids[b_idx, p_b, g_b]
            p_ids[b_idx, p_t, g_t] = id_b
            p_ids[b_idx, p_b, g_b] = id_t
            
        return p_contents, p_ids

    def run_2_swap_pass(p_contents, p_ids, iterations=5):
        """2-for-2 swap between Max and Min pack.
           Only enabled for reasonably small group sizes to keep memory checks."""
        # Precompute pair indices: [2, NumPairs]
        G = groups_per_pack
        u_idx, v_idx = torch.triu_indices(G, G, offset=1, device=device)
        num_pairs = u_idx.shape[0]
        
        for _ in range(iterations):
            p_weights = p_contents.sum(dim=2)
            # Just take absolute max and min (K=1) to save compute
            val_max, p_max = p_weights.max(dim=1)
            val_min, p_min = p_weights.min(dim=1)
            
            diff = val_max - val_min
            active_mask = diff > 1e-4
            if not active_mask.any(): break
            
            # Gather items: [B, G]
            items_max = p_contents[row_indices, p_max, :]
            items_min = p_contents[row_indices, p_min, :]
            
            # Compute Pair Sums: [B, NumPairs]
            pair_sum_max = items_max[:, u_idx] + items_max[:, v_idx]
            pair_sum_min = items_min[:, u_idx] + items_min[:, v_idx]
            
            # Delta: [B, NumPairs, NumPairs]
            # (Pair from Max) - (Pair from Min)
            delta = pair_sum_max.unsqueeze(2) - pair_sum_min.unsqueeze(1)
            
            # Gain
            target = diff.view(-1, 1, 1)
            # Maximize reduction: diff^2 - (diff-2*delta)^2 approx -> maximize delta * (diff - delta)
            # Or strictly: new_diff = |diff - 2*delta|. We want new_diff < diff.
            # Improvement = diff - |diff - 2*delta|
            change = (target - 2 * delta).abs()
            improvement = target - change
            
            mask = (improvement > 1e-5) & active_mask.view(-1, 1, 1)
            improvement[~mask] = -float('inf')
            
            best_imp, best_flat = improvement.view(batch_size, -1).max(dim=1)
            do_swap = best_imp > -float('inf')
            
            if not do_swap.any(): break
            
            b_idx = row_indices[do_swap]
            flat_idx = best_flat[do_swap]
            
            # Decode
            pair_idx_max = flat_idx // num_pairs
            pair_idx_min = flat_idx % num_pairs
            
            # Indices in the group (0..G-1)
            g_max_1 = u_idx[pair_idx_max]
            g_max_2 = v_idx[pair_idx_max]
            g_min_1 = u_idx[pair_idx_min]
            g_min_2 = v_idx[pair_idx_min]
            
            p_m = p_max[do_swap]
            p_l = p_min[do_swap]
            
            # Swap 1
            v_m1 = p_contents[b_idx, p_m, g_max_1]
            v_l1 = p_contents[b_idx, p_l, g_min_1]
            p_contents[b_idx, p_m, g_max_1] = v_l1
            p_contents[b_idx, p_l, g_min_1] = v_m1
            
            id_m1 = p_ids[b_idx, p_m, g_max_1]
            id_l1 = p_ids[b_idx, p_l, g_min_1]
            p_ids[b_idx, p_m, g_max_1] = id_l1
            p_ids[b_idx, p_l, g_min_1] = id_m1
            
            # Swap 2
            v_m2 = p_contents[b_idx, p_m, g_max_2]
            v_l2 = p_contents[b_idx, p_l, g_min_2]
            p_contents[b_idx, p_m, g_max_2] = v_l2
            p_contents[b_idx, p_l, g_min_2] = v_m2

            id_m2 = p_ids[b_idx, p_m, g_max_2]
            id_l2 = p_ids[b_idx, p_l, g_min_2]
            p_ids[b_idx, p_m, g_max_2] = id_l2
            p_ids[b_idx, p_l, g_min_2] = id_m2
            
        return p_contents, p_ids

    # --- Step 3: Execution Strategy ---
    
    # Phase 1: Fast 1-for-1 swaps to clear low hanging fruit
    pack_contents, pack_item_ids = run_1_swap_pass(pack_contents, pack_item_ids, iterations=15, K=4)
    
    # Phase 2: 2-for-2 swaps (Computationally heavier, helps structural imbalances)
    # Only run if group size is small enough to keep pair tensor manageable (e.g., <= 32)
    # 32*31/2 = 496 pairs. 496^2 = 246k. Batch 400 -> 100M elements. Feasible.
    if groups_per_pack > 1 and groups_per_pack <= 32:
        pack_contents, pack_item_ids = run_2_swap_pass(pack_contents, pack_item_ids, iterations=8)
        
        # Phase 3: Cleanup 1-for-1 swaps after structure change
        pack_contents, pack_item_ids = run_1_swap_pass(pack_contents, pack_item_ids, iterations=5, K=4)

    # --- Step 4: Selection and Reconstruction ---
    
    # Evaluate final imbalance
    final_weights = pack_contents.sum(dim=2)
    imbalance = final_weights.max(dim=1).values - final_weights.min(dim=1).values
    imbalance = imbalance.view(num_layers, num_restarts)
    
    # Select best restart
    best_restart_idx = imbalance.argmin(dim=1)
    best_batch_idx = torch.arange(num_layers, device=device) * num_restarts + best_restart_idx
    
    best_item_ids = pack_item_ids[best_batch_idx] # [L, P, G]
    
    # Scatter back
    pack_index = torch.empty(num_layers, num_groups, dtype=torch.int64, device=device)
    rank_in_pack = torch.empty(num_layers, num_groups, dtype=torch.int64, device=device)
    
    flat_item_ids = best_item_ids.view(num_layers, -1)
    
    # Create grids
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