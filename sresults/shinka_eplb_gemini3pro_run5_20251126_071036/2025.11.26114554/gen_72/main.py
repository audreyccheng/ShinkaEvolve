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

    Uses Parallel Randomized Greedy Initialization followed by
    Randomized Pairwise Diffusion Refinement.

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

    # Trivial case
    if groups_per_pack == 1:
        pack_index = torch.arange(weight.size(-1),
                                  dtype=torch.int64,
                                  device=device).expand(weight.shape)
        rank_in_pack = torch.zeros_like(weight, dtype=torch.int64)
        return pack_index, rank_in_pack

    # Hyperparameters
    # We use 64 parallel restarts to explore the solution space.
    num_restarts = 64
    num_diffusion_steps = 20

    # Expand weights: [Layers, Restarts, Groups]
    # w_real stores the actual weights for summation/evaluation
    w_real = weight.unsqueeze(1).expand(-1, num_restarts, -1)

    # w_noisy stores perturbed weights for decision making (breaking ties/local optima)
    # Restart 0 is deterministic (original weights)
    noise = torch.rand(num_layers, num_restarts - 1, num_groups, device=device) * 0.05
    w_noisy = torch.empty(num_layers, num_restarts, num_groups, device=device)
    w_noisy[:, 0, :] = w_real[:, 0, :]
    w_noisy[:, 1:, :] = w_real[:, 1:, :] * (1.0 + noise)

    # --- Step 1: Parallel Greedy LPT Initialization ---
    
    # Sort items by perturbed weights (Descending)
    # sorted_indices: [L, R, G]
    sorted_indices = w_noisy.argsort(dim=-1, descending=True)
    
    # Gather real weights in sorted order
    # w_sorted: [L, R, G]
    w_sorted = w_real.gather(2, sorted_indices)

    # State tensors: [Layers, Restarts, Packs]
    pack_weights = torch.zeros(num_layers, num_restarts, num_packs, device=device)
    pack_counts = torch.zeros(num_layers, num_restarts, num_packs, dtype=torch.int64, device=device)
    
    # We will reconstruct the assignment map directly into pack_contents
    # pack_contents: [Layers, Restarts, Packs, Slots] -> Item Index (Original 0..G-1)
    # We need intermediate storage to capture where each sorted item goes.
    
    # pack_assignment_sorted: [L, R, G] -> Pack Index
    # rank_assignment_sorted: [L, R, G] -> Rank Index
    pack_assignment_sorted = torch.zeros(num_layers, num_restarts, num_groups, dtype=torch.int64, device=device)
    rank_assignment_sorted = torch.zeros(num_layers, num_restarts, num_groups, dtype=torch.int64, device=device)

    # Greedy Allocation Loop
    for i in range(num_groups):
        # Weight of the i-th heaviest item across all restarts
        w_curr = w_sorted[:, :, i] # [L, R]
        
        # Mask valid packs (count < capacity)
        valid_mask = pack_counts < groups_per_pack # [L, R, P]
        
        # Select pack with min weight among valid ones
        costs = pack_weights.clone()
        costs[~valid_mask] = float('inf')
        
        chosen_pack = costs.argmin(dim=2) # [L, R]
        
        # Prepare indices for update
        chosen_pack_uns = chosen_pack.unsqueeze(2) # [L, R, 1]
        
        # Update weights
        pack_weights.scatter_add_(2, chosen_pack_uns, w_curr.unsqueeze(2))
        
        # Record assignment
        current_counts = pack_counts.gather(2, chosen_pack_uns).squeeze(2)
        rank_assignment_sorted[:, :, i] = current_counts
        pack_assignment_sorted[:, :, i] = chosen_pack
        
        # Increment counts
        pack_counts.scatter_add_(2, chosen_pack_uns, torch.ones_like(chosen_pack_uns, dtype=torch.int64))

    # Construct Pack Contents (The state for Diffusion)
    # This tensor holds the original Item IDs (0..G-1)
    pack_contents = torch.zeros(num_layers, num_restarts, num_packs, groups_per_pack, dtype=torch.int64, device=device)
    
    # We scatter 'sorted_indices' into 'pack_contents' at the assigned positions.
    # Dimensions: L, R, P, S
    
    flat_l = torch.arange(num_layers, device=device).view(-1, 1, 1).expand(-1, num_restarts, num_groups).flatten()
    flat_r = torch.arange(num_restarts, device=device).view(1, -1, 1).expand(num_layers, -1, num_groups).flatten()
    flat_p = pack_assignment_sorted.flatten()
    flat_s = rank_assignment_sorted.flatten()
    
    flat_item_ids = sorted_indices.flatten()
    
    pack_contents.index_put_((flat_l, flat_r, flat_p, flat_s), flat_item_ids)

    # --- Step 2: Diffusion Refinement Loop ---
    
    num_pairs = num_packs // 2
    
    if num_pairs > 0:
        for step in range(num_diffusion_steps):
            # 2.1 Generate Random Pairings
            perm = torch.randperm(num_packs, device=device)
            p1_indices = perm[:num_pairs]      # [Pairs]
            p2_indices = perm[num_pairs:2*num_pairs] # [Pairs]
            
            # 2.2 Gather Items and Weights for pairs
            # Access: [L, R, P, S]
            # We select P indices
            
            # items_1: [L, R, Pairs, S]
            items_1 = pack_contents[:, :, p1_indices, :]
            items_2 = pack_contents[:, :, p2_indices, :]
            
            # Gather weights. We need to look up w_real [L, R, G] using items [L, R, Pairs, S]
            
            flat_items_1 = items_1.view(num_layers, num_restarts, -1)
            flat_items_2 = items_2.view(num_layers, num_restarts, -1)
            
            w_1 = w_real.gather(2, flat_items_1).view(num_layers, num_restarts, num_pairs, groups_per_pack)
            w_2 = w_real.gather(2, flat_items_2).view(num_layers, num_restarts, num_pairs, groups_per_pack)
            
            # 2.3 Merge and Sort
            combined_items = torch.cat([items_1, items_2], dim=3) # [L, R, Pairs, 2*S]
            combined_w = torch.cat([w_1, w_2], dim=3)
            
            # Sort combined items descending by weight
            c_w_sorted, sort_order = combined_w.sort(dim=3, descending=True)
            c_items_sorted = combined_items.gather(3, sort_order)
            
            # 2.4 Partition Greedy (Mini-LPT)
            # Two bins (0 and 1) representing p1 and p2. Capacity groups_per_pack.
            
            bin_w = torch.zeros(num_layers, num_restarts, num_pairs, 2, device=device)
            bin_c = torch.zeros(num_layers, num_restarts, num_pairs, 2, dtype=torch.int64, device=device)
            
            # Buffers to store where each item goes
            # dest_bin: 0 or 1. dest_slot: 0..S-1.
            dest_bin = torch.empty(num_layers, num_restarts, num_pairs, 2*groups_per_pack, dtype=torch.int64, device=device)
            dest_slot = torch.empty(num_layers, num_restarts, num_pairs, 2*groups_per_pack, dtype=torch.int64, device=device)
            
            for k in range(2 * groups_per_pack):
                w_k = c_w_sorted[:, :, :, k]
                
                # Check valid moves
                fits_0 = bin_c[:, :, :, 0] < groups_per_pack
                fits_1 = bin_c[:, :, :, 1] < groups_per_pack
                
                # Greedy choice: if both fit, pick lighter bin.
                prefer_0 = bin_w[:, :, :, 0] <= bin_w[:, :, :, 1]
                pick_0 = (fits_0 & fits_1 & prefer_0) | (fits_0 & (~fits_1))
                
                choice = (~pick_0).long() # 0 or 1
                
                # Update weights
                w_add_0 = torch.where(pick_0, w_k, torch.tensor(0.0, device=device))
                w_add_1 = torch.where(~pick_0, w_k, torch.tensor(0.0, device=device))
                bin_w[:, :, :, 0] += w_add_0
                bin_w[:, :, :, 1] += w_add_1
                
                # Store
                dest_bin[:, :, :, k] = choice
                
                # Get current count for slot index
                # choice is [L, R, Pairs]. Expand to use as index.
                slot_idx = bin_c.gather(3, choice.unsqueeze(3)).squeeze(3)
                dest_slot[:, :, :, k] = slot_idx
                
                # Update counts
                bin_c.scatter_add_(3, choice.unsqueeze(3), torch.ones_like(choice.unsqueeze(3)))
                
            # 2.5 Scatter back to pack_contents
            # We map bin 0 -> p1, bin 1 -> p2
            p1_exp = p1_indices.view(1, 1, -1, 1).expand(num_layers, num_restarts, -1, 2*groups_per_pack)
            p2_exp = p2_indices.view(1, 1, -1, 1).expand(num_layers, num_restarts, -1, 2*groups_per_pack)
            
            final_p = torch.where(dest_bin == 0, p1_exp, p2_exp)
            final_s = dest_slot
            
            # Construct indices for scatter
            # We need to scatter c_items_sorted [L, R, Pairs, 2S] into pack_contents [L, R, Packs, S]
            
            l_idx = torch.arange(num_layers, device=device).view(-1, 1, 1, 1).expand(-1, num_restarts, num_pairs, 2*groups_per_pack)
            r_idx = torch.arange(num_restarts, device=device).view(1, -1, 1, 1).expand(num_layers, -1, num_pairs, 2*groups_per_pack)
            
            pack_contents.index_put_(
                (l_idx.flatten(), r_idx.flatten(), final_p.flatten(), final_s.flatten()),
                c_items_sorted.flatten()
            )

    # --- Step 3: Selection ---
    
    # Calculate weights for all packs in all restarts
    # Flatten items to gather weights
    flat_contents = pack_contents.view(num_layers, num_restarts, -1)
    pack_item_weights = w_real.gather(2, flat_contents)
    pack_sums = pack_item_weights.view(num_layers, num_restarts, num_packs, groups_per_pack).sum(dim=3) # [L, R, P]
    
    # Calculate imbalance: Max - Min
    imbalance = pack_sums.max(dim=2).values - pack_sums.min(dim=2).values # [L, R]
    
    # Select best restart per layer
    best_restart_idx = imbalance.argmin(dim=1) # [L]
    
    # --- Step 4: Output Generation ---
    
    # Extract the best configuration
    # best_contents: [L, P, S]
    # We use gather on the Restart dimension
    
    # Expand best_restart_idx for gathering: [L, 1, P, S]
    gather_idx = best_restart_idx.view(num_layers, 1, 1, 1).expand(-1, 1, num_packs, groups_per_pack)
    best_contents = pack_contents.gather(1, gather_idx).squeeze(1) # [L, P, S]
    
    # Map back to pack_index and rank_in_pack
    # best_contents contains Item IDs.
    # pack_index[layer, item_id] = pack_id
    # rank_in_pack[layer, item_id] = slot_id
    
    pack_index = torch.empty(num_layers, num_groups, dtype=torch.int64, device=device)
    rank_in_pack = torch.empty(num_layers, num_groups, dtype=torch.int64, device=device)
    
    # Grids of P and S indices
    p_grid = torch.arange(num_packs, device=device).view(1, -1, 1).expand(num_layers, -1, groups_per_pack)
    s_grid = torch.arange(groups_per_pack, device=device).view(1, 1, -1).expand(num_layers, num_packs, -1)
    
    # Scatter to output
    flat_indices = best_contents.view(num_layers, -1)
    flat_p_vals = p_grid.reshape(num_layers, -1)
    flat_s_vals = s_grid.reshape(num_layers, -1)
    
    pack_index.scatter_(1, flat_indices, flat_p_vals)
    rank_in_pack.scatter_(1, flat_indices, flat_s_vals)
    
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
