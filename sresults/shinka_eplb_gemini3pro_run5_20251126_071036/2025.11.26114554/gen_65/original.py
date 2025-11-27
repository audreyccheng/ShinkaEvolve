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
    
    Uses Parallel Randomized Greedy LPT initialization followed by 
    Top-K Pairwise Refinement.

    Parameters:
        weight: [X, n], the weight of each item
        num_packs: number of packs

    Returns:
        pack_index: [X, n], the pack index of each item
        rank_in_pack: [X, n], the rank of the item in the pack
    """
    num_layers, num_groups = weight.shape
    device = weight.device
    
    # 1. Parameter Validation and Trivial Case
    assert num_groups % num_packs == 0
    groups_per_pack = num_groups // num_packs

    if groups_per_pack == 1:
        # Trivial mapping: items just map sequentially to packs
        pack_index = torch.arange(weight.size(-1),
                                  dtype=torch.int64,
                                  device=device).expand(weight.shape) % num_packs
        rank_in_pack = torch.zeros_like(weight, dtype=torch.int64)
        return pack_index, rank_in_pack

    # 2. Parallel Random Greedy Initialization
    # We use 128 restarts to explore the solution space.
    num_restarts = 128
    
    # Expand weights [L, R, G]
    w_exp = weight.float().unsqueeze(1).expand(-1, num_restarts, -1)
    
    # Perturb weights for sorting (Randomized LPT)
    # Restart 0 is deterministic (noise=0), others have 5% noise
    noise = torch.rand_like(w_exp) * 0.05
    noise[:, 0, :] = 0.0
    
    # Sort indices based on noisy weights
    w_noisy = w_exp * (1 + noise)
    sorted_indices = w_noisy.argsort(dim=-1, descending=True)
    
    # Gather actual weights [L, R, G] in the sorted order
    w_sorted = w_exp.gather(2, sorted_indices)
    
    # Allocations for greedy state
    pack_weights = torch.zeros(num_layers, num_restarts, num_packs, device=device)
    pack_counts = torch.zeros(num_layers, num_restarts, num_packs, dtype=torch.int64, device=device)
    
    # Result holders (in sorted order)
    pack_idx_sorted = torch.zeros(num_layers, num_restarts, num_groups, dtype=torch.int64, device=device)
    rank_sorted = torch.zeros(num_layers, num_restarts, num_groups, dtype=torch.int64, device=device)
    
    # Vectorized Greedy Packing
    for i in range(num_groups):
        w_curr = w_sorted[:, :, i]
        
        # Identify valid packs (count < limit)
        mask_valid = pack_counts < groups_per_pack
        
        # Select pack with minimum weight among valid packs
        curr_costs = pack_weights.clone()
        curr_costs[~mask_valid] = float('inf')
        
        chosen_pack = curr_costs.argmin(dim=2) # [L, R]
        chosen_pack_exp = chosen_pack.unsqueeze(2)
        
        # Get rank (current count)
        curr_rank = pack_counts.gather(2, chosen_pack_exp).squeeze(2)
        
        # Update State
        pack_weights.scatter_add_(2, chosen_pack_exp, w_curr.unsqueeze(2))
        pack_counts.scatter_add_(2, chosen_pack_exp, torch.ones_like(chosen_pack_exp, dtype=torch.int64))
        
        # Record Assignment
        pack_idx_sorted[:, :, i] = chosen_pack
        rank_sorted[:, :, i] = curr_rank

    # 3. Candidate Selection
    # Calculate imbalance range (max - min)
    rng = pack_weights.max(dim=2).values - pack_weights.min(dim=2).values
    
    # Select Top 4 candidates per layer to refine
    K_CAND = 4
    if num_restarts < K_CAND:
        K_CAND = num_restarts
        
    best_cand_indices = rng.topk(K_CAND, dim=1, largest=False).indices # [L, K]
    
    # Gather candidates into a flattened batch B = L * K_CAND
    B = num_layers * K_CAND
    
    def gather_batch(tensor):
        # tensor: [L, R, ...]
        dims = list(tensor.shape)
        dims[1] = K_CAND
        # Create gather indices
        gather_idx = best_cand_indices.view([num_layers, K_CAND] + [1]*(tensor.ndim-2))
        gather_idx = gather_idx.expand(dims)
        return tensor.gather(1, gather_idx).reshape(B, *dims[2:])

    b_w_sorted = gather_batch(w_sorted)
    b_indices = gather_batch(sorted_indices)
    b_pack_idx = gather_batch(pack_idx_sorted)
    b_rank = gather_batch(rank_sorted)
    
    # Construct mutable workspace for refinement
    # pack_contents: [B, Packs, ItemsPerPack]
    pack_contents = torch.zeros(B, num_packs, groups_per_pack, device=device)
    pack_ids = torch.zeros(B, num_packs, groups_per_pack, dtype=torch.int64, device=device)
    
    b_seq = torch.arange(B, device=device).unsqueeze(1).expand(-1, num_groups).reshape(-1)
    
    # Scatter data into structured format
    pack_contents.index_put_((b_seq, b_pack_idx.flatten(), b_rank.flatten()), b_w_sorted.flatten())
    pack_ids.index_put_((b_seq, b_pack_idx.flatten(), b_rank.flatten()), b_indices.flatten())

    # 4. Top-K vs Bottom-K Refinement
    # We check swaps between the Top-K heaviest and Bottom-K lightest packs.
    # This avoids getting stuck if the absolute max/min pair has no valid swaps.
    SEARCH_K = min(4, num_packs)
    
    for _ in range(50):
        # Compute pack weights
        pw = pack_contents.sum(dim=2) # [B, P]
        
        # Sort packs by weight
        sorted_p_idx = pw.argsort(dim=1)
        
        # Identify heavy and light packs indices
        heavy_idx = sorted_p_idx[:, -SEARCH_K:] # [B, K]
        light_idx = sorted_p_idx[:, :SEARCH_K]  # [B, K]
        
        # Global convergence check
        glob_max = pw.max(dim=1).values
        glob_min = pw.min(dim=1).values
        if (glob_max - glob_min).max() < 1e-4:
            break
            
        # Extract weights and items for comparison
        # Heavy Weights: [B, K, 1, 1, 1] (Broadcasting dims)
        w_h = pw.gather(1, heavy_idx).view(B, SEARCH_K, 1, 1, 1)
        # Light Weights: [B, 1, K, 1, 1]
        w_l = pw.gather(1, light_idx).view(B, 1, SEARCH_K, 1, 1)
        
        # Heavy Items: [B, K, 1, G, 1]
        idx_h_exp = heavy_idx.unsqueeze(2).expand(-1, -1, groups_per_pack)
        items_h = pack_contents.gather(1, idx_h_exp).view(B, SEARCH_K, 1, groups_per_pack, 1)
        
        # Light Items: [B, 1, K, 1, G]
        idx_l_exp = light_idx.unsqueeze(2).expand(-1, -1, groups_per_pack)
        items_l = pack_contents.gather(1, idx_l_exp).view(B, 1, SEARCH_K, 1, groups_per_pack)
        
        # Compute Gains
        # Current Difference between specific pairs
        diff = w_h - w_l # [B, K, K, 1, 1]
        
        # Delta if we swap item h with item l
        delta = items_h - items_l # [B, K, K, G, G]
        
        # Objective: Maximize improvement in the difference between these packs
        # Improvement = |OldDiff| - |OldDiff - 2*Delta|
        # Since we know OldDiff > 0 (Heavy - Light), and we want reduction:
        improvement = diff - (diff - 2 * delta).abs()
        
        # Find best swap across all K*K pairs and G*G items
        imp_flat = improvement.view(B, -1)
        best_imp, best_flat_idx = imp_flat.max(dim=1)
        
        # Check if improvement is significant
        do_swap = best_imp > 1e-5
        if not do_swap.any():
            break
            
        # Decode flattened indices to perform the swap
        valid = torch.where(do_swap)[0]
        flat = best_flat_idx[valid]
        
        G = groups_per_pack
        l_item = flat % G
        flat //= G
        h_item = flat % G
        flat //= G
        l_k = flat % SEARCH_K
        h_k = flat // SEARCH_K
        
        # Retrieve actual pack indices from heavy/light arrays
        real_p_h = heavy_idx[valid, h_k]
        real_p_l = light_idx[valid, l_k]
        
        # Perform the swap on contents and IDs
        val_h = pack_contents[valid, real_p_h, h_item]
        val_l = pack_contents[valid, real_p_l, l_item]
        pack_contents[valid, real_p_h, h_item] = val_l
        pack_contents[valid, real_p_l, l_item] = val_h
        
        id_h = pack_ids[valid, real_p_h, h_item]
        id_l = pack_ids[valid, real_p_l, l_item]
        pack_ids[valid, real_p_h, h_item] = id_l
        pack_ids[valid, real_p_l, l_item] = id_h

    # 5. Final Selection
    # Select best from the refined candidates
    final_pw = pack_contents.sum(dim=2)
    final_rng = final_pw.max(dim=1).values - final_pw.min(dim=1).values
    final_rng = final_rng.view(num_layers, K_CAND)
    
    best_k = final_rng.argmin(dim=1)
    
    # Gather best item IDs
    gather_off = torch.arange(num_layers, device=device) * K_CAND + best_k
    best_ids = pack_ids[gather_off] # [L, Packs, G]
    
    # Flatten to [L, N] for scatter
    flat_ids = best_ids.view(num_layers, -1)
    
    # Reconstruct Output Tensors
    # We scatter pack indices and ranks back to their original item positions
    res_pack = torch.zeros(num_layers, num_groups, dtype=torch.int64, device=device)
    res_rank = torch.zeros(num_layers, num_groups, dtype=torch.int64, device=device)
    
    src_pack = torch.arange(num_packs, device=device).view(1, -1, 1).expand(num_layers, -1, groups_per_pack).reshape(num_layers, -1)
    src_rank = torch.arange(groups_per_pack, device=device).view(1, 1, -1).expand(num_layers, num_packs, -1).reshape(num_layers, -1)
    
    res_pack.scatter_(1, flat_ids, src_pack)
    res_rank.scatter_(1, flat_ids, src_rank)
    
    return res_pack, res_rank


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