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

    # --- Phase 1: Parallel Randomized Greedy LPT ---
    # Use many restarts to explore the landscape globally
    num_restarts = 256
    
    # Expand weights: [Layers, Restarts, Groups]
    w_expanded = weight.unsqueeze(1).expand(-1, num_restarts, -1)
    
    # Perturb weights for sorting to induce different greedy choices
    # Keep restart 0 deterministic
    noise = torch.rand(num_layers, num_restarts - 1, num_groups, device=device) * 0.05
    w_sorting = torch.empty(num_layers, num_restarts, num_groups, device=device)
    w_sorting[:, 0, :] = w_expanded[:, 0, :]
    w_sorting[:, 1:, :] = w_expanded[:, 1:, :] * (1.0 + noise)
    
    # Sort indices based on perturbed weights
    sorted_indices = w_sorting.argsort(dim=-1, descending=True)
    w_sorted = w_expanded.gather(2, sorted_indices)
    
    # Allocate state tensors
    pack_weights = torch.zeros(num_layers, num_restarts, num_packs, device=device)
    pack_counts = torch.zeros(num_layers, num_restarts, num_packs, dtype=torch.int64, device=device)
    pack_idx_sorted = torch.empty(num_layers, num_restarts, num_groups, dtype=torch.int64, device=device)
    rank_sorted = torch.empty(num_layers, num_restarts, num_groups, dtype=torch.int64, device=device)
    
    # Vectorized Greedy Loop
    for i in range(num_groups):
        w_curr = w_sorted[:, :, i] # [L, R]
        
        # Mask valid packs
        valid_mask = pack_counts < groups_per_pack
        costs = pack_weights.clone()
        costs[~valid_mask] = float('inf')
        
        # Select best pack
        chosen_pack = costs.argmin(dim=2) # [L, R]
        chosen_pack_u = chosen_pack.unsqueeze(2)
        
        # Update state
        pack_weights.scatter_add_(2, chosen_pack_u, w_curr.unsqueeze(2))
        ranks = pack_counts.gather(2, chosen_pack_u).squeeze(2)
        rank_sorted[:, :, i] = ranks
        pack_counts.scatter_add_(2, chosen_pack_u, torch.ones_like(chosen_pack_u))
        pack_idx_sorted[:, :, i] = chosen_pack

    # --- Phase 2: Coarse Refinement (Max-Min 1-Swap) ---
    # Flatten batch dimensions for efficiency
    B = num_layers * num_restarts
    pack_contents = torch.zeros(B, num_packs, groups_per_pack, device=device)
    pack_orig_ids = torch.zeros(B, num_packs, groups_per_pack, dtype=torch.int64, device=device)
    
    flat_pack = pack_idx_sorted.view(B, num_groups)
    flat_rank = rank_sorted.view(B, num_groups)
    flat_w = w_sorted.view(B, num_groups)
    flat_ids = sorted_indices.view(B, num_groups)
    batch_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, num_groups)
    
    # Scatter to structured format
    pack_contents.index_put_((batch_idx.flatten(), flat_pack.flatten(), flat_rank.flatten()), flat_w.flatten())
    pack_orig_ids.index_put_((batch_idx.flatten(), flat_pack.flatten(), flat_rank.flatten()), flat_ids.flatten())
    
    # 20 iterations of basic Max-Min refinement on all candidates
    for _ in range(20):
        p_weights = pack_contents.sum(dim=2)
        val_max, idx_max = p_weights.max(dim=1)
        val_min, idx_min = p_weights.min(dim=1)
        diff = val_max - val_min
        
        if not (diff > 1e-4).any(): break
        
        idx_max_exp = idx_max.view(-1, 1, 1).expand(-1, 1, groups_per_pack)
        idx_min_exp = idx_min.view(-1, 1, 1).expand(-1, 1, groups_per_pack)
        items_max = pack_contents.gather(1, idx_max_exp).squeeze(1)
        items_min = pack_contents.gather(1, idx_min_exp).squeeze(1)
        
        delta = items_max.unsqueeze(2) - items_min.unsqueeze(1)
        target = diff.view(-1, 1, 1)
        improvement = target - (target - 2 * delta).abs()
        
        best_imp, best_idx = improvement.view(B, -1).max(dim=1)
        do_swap = (best_imp > 1e-6)
        
        if not do_swap.any(): break
        
        batch_ids = torch.where(do_swap)[0]
        swap_indices = best_idx[do_swap]
        
        p_mx = idx_max[batch_ids]
        p_mn = idx_min[batch_ids]
        u = swap_indices // groups_per_pack
        v = swap_indices % groups_per_pack
        
        val_u = pack_contents[batch_ids, p_mx, u]
        val_v = pack_contents[batch_ids, p_mn, v]
        pack_contents[batch_ids, p_mx, u] = val_v
        pack_contents[batch_ids, p_mn, v] = val_u
        
        id_u = pack_orig_ids[batch_ids, p_mx, u]
        id_v = pack_orig_ids[batch_ids, p_mn, v]
        pack_orig_ids[batch_ids, p_mx, u] = id_v
        pack_orig_ids[batch_ids, p_mn, v] = id_u
        
    # --- Phase 3: Selection of Best Restart ---
    final_weights = pack_contents.sum(dim=2)
    imbalance = final_weights.max(dim=1).values - final_weights.min(dim=1).values
    imbalance = imbalance.view(num_layers, num_restarts)
    best_restart_idx = imbalance.argmin(dim=1)
    
    select_batch_idx = torch.arange(num_layers, device=device) * num_restarts + best_restart_idx
    best_contents = pack_contents[select_batch_idx] # [L, P, G]
    best_ids = pack_orig_ids[select_batch_idx]
    
    # --- Phase 4: Deep Refinement (Variance-Minimizing All-Pairs + Max-Min 2-Swap) ---
    # Operate only on the best candidate per layer
    
    # Precompute indices for 2-swaps if needed
    has_2swap = groups_per_pack >= 2
    if has_2swap:
        r_idx, c_idx = torch.triu_indices(groups_per_pack, groups_per_pack, offset=1, device=device)
        num_pairs = r_idx.size(0)
    
    for _ in range(50):
        # Current Pack Weights [L, P]
        p_weights = best_contents.sum(dim=2)
        
        # --- Sub-step A: All-Pairs 1-Swap (Minimize Variance) ---
        # Evaluate swapping any item from any pack i to any pack j
        # Objective: Minimize Sum of Squared Loads
        # Gain = 2*delta*(L_i - L_j) - 2*delta^2, where delta = w_u - w_v
        
        # Diff Matrix [L, P, P] = L_i - L_j
        diff_matrix = p_weights.unsqueeze(2) - p_weights.unsqueeze(1)
        
        # Only consider pairs where L_i > L_j + epsilon
        mask_pairs = diff_matrix > 1e-4
        
        found_improvement = False
        
        if mask_pairs.any():
            # Delta [L, P, P, G, G]
            # w[i, g_i] - w[j, g_j]
            # w_i: items in pack i
            w_i = best_contents.unsqueeze(2).unsqueeze(4) # [L, P, 1, G, 1]
            w_j = best_contents.unsqueeze(1).unsqueeze(3) # [L, 1, P, 1, G]
            
            delta = w_i - w_j
            
            # Gain calculation
            # term1: [L, P, P, G, G] broadcasting diff_matrix [L, P, P, 1, 1]
            # Note: We only care about positive gain.
            term1 = 2 * delta * diff_matrix.unsqueeze(3).unsqueeze(4)
            term2 = 2 * delta.pow(2)
            gain = term1 - term2
            
            # Mask invalid swaps
            # delta > 0: enforces moving weight from heavier to lighter pack relative to items
            # mask_pairs: ensures pack i is heavier than pack j
            valid_mask = (delta > 0) & (gain > 1e-6) & mask_pairs.unsqueeze(3).unsqueeze(4)
            
            # Find global best swap for each layer
            # Flatten P, P, G, G -> [L, -1]
            gain_flat = gain.view(num_layers, -1)
            best_gain, best_idx_flat = gain_flat.max(dim=1)
            
            should_swap = best_gain > 1e-6
            
            if should_swap.any():
                found_improvement = True
                
                # Decode Indices
                l_ids = torch.where(should_swap)[0]
                idx_flat = best_idx_flat[l_ids]
                
                # Decode: G, G, P, P
                G = groups_per_pack
                idx_g_j = idx_flat % G
                rem = idx_flat // G
                idx_g_i = rem % G
                rem = rem // G
                idx_p_j = rem % num_packs
                idx_p_i = rem // num_packs
                
                # Perform Swap
                # i is source (heavier), j is dest (lighter)
                val_u = best_contents[l_ids, idx_p_i, idx_g_i]
                val_v = best_contents[l_ids, idx_p_j, idx_g_j]
                
                best_contents[l_ids, idx_p_i, idx_g_i] = val_v
                best_contents[l_ids, idx_p_j, idx_g_j] = val_u
                
                id_u = best_ids[l_ids, idx_p_i, idx_g_i]
                id_v = best_ids[l_ids, idx_p_j, idx_g_j]
                
                best_ids[l_ids, idx_p_i, idx_g_i] = id_v
                best_ids[l_ids, idx_p_j, idx_g_j] = id_u
        
        if found_improvement:
            continue
            
        # --- Sub-step B: Max-Min 2-Swap ---
        # Only if 1-swap didn't improve things
        if not has_2swap:
            break
            
        val_max, idx_max = p_weights.max(dim=1)
        val_min, idx_min = p_weights.min(dim=1)
        diff = val_max - val_min
        
        if not (diff > 1e-4).any():
            break
            
        # Gather items [L, G]
        items_max = best_contents[torch.arange(num_layers, device=device), idx_max]
        items_min = best_contents[torch.arange(num_layers, device=device), idx_min]
        
        # Pair sums [L, NumPairs]
        pair_sum_max = items_max[:, r_idx] + items_max[:, c_idx]
        pair_sum_min = items_min[:, r_idx] + items_min[:, c_idx]
        
        delta = pair_sum_max.unsqueeze(2) - pair_sum_min.unsqueeze(1)
        target = diff.view(-1, 1, 1)
        improvement = target - (target - 2 * delta).abs()
        
        best_imp, best_idx_flat_2 = improvement.view(num_layers, -1).max(dim=1)
        should_swap_2 = (best_imp > 1e-6)
        
        if not should_swap_2.any():
            break
            
        # Perform 2-swaps
        l_ids = torch.where(should_swap_2)[0]
        flat_indices_2 = best_idx_flat_2[l_ids]
        
        pair_max_idx = flat_indices_2 // num_pairs
        pair_min_idx = flat_indices_2 % num_pairs
        
        u1, u2 = r_idx[pair_max_idx], c_idx[pair_max_idx]
        v1, v2 = r_idx[pair_min_idx], c_idx[pair_min_idx]
        
        p_mx = idx_max[l_ids]
        p_mn = idx_min[l_ids]
        
        # Swap values
        val_u1 = best_contents[l_ids, p_mx, u1]
        val_u2 = best_contents[l_ids, p_mx, u2]
        val_v1 = best_contents[l_ids, p_mn, v1]
        val_v2 = best_contents[l_ids, p_mn, v2]
        
        best_contents[l_ids, p_mx, u1] = val_v1
        best_contents[l_ids, p_mx, u2] = val_v2
        best_contents[l_ids, p_mn, v1] = val_u1
        best_contents[l_ids, p_mn, v2] = val_u2
        
        # Swap IDs
        id_u1 = best_ids[l_ids, p_mx, u1]
        id_u2 = best_ids[l_ids, p_mx, u2]
        id_v1 = best_ids[l_ids, p_mn, v1]
        id_v2 = best_ids[l_ids, p_mn, v2]
        
        best_ids[l_ids, p_mx, u1] = id_v1
        best_ids[l_ids, p_mx, u2] = id_v2
        best_ids[l_ids, p_mn, v1] = id_u1
        best_ids[l_ids, p_mn, v2] = id_u2

    # --- Phase 5: Reconstruction ---
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