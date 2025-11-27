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
    
    Uses Parallel Greedy LPT + 1-Swap and 2-Swap Refinement.

    Parameters:
        weight: [Batch, n], the weight of each item
        num_packs: number of packs

    Returns:
        pack_index: [Batch, n], the pack index of each item
        rank_in_pack: [Batch, n], the rank of the item in the pack
    """
    batch_size, num_groups = weight.shape
    device = weight.device
    assert num_groups % num_packs == 0
    groups_per_pack = num_groups // num_packs

    if groups_per_pack == 1:
        pack_index = torch.arange(num_packs,
                                  dtype=torch.int64,
                                  device=device).repeat(batch_size, 1)
        rank_in_pack = torch.zeros_like(weight, dtype=torch.int64)
        return pack_index, rank_in_pack

    # 1. Greedy LPT Initialization
    sorted_weight, sorted_indices = weight.float().sort(dim=-1, descending=True)
    
    pack_weights = torch.zeros(batch_size, num_packs, device=device)
    pack_counts = torch.zeros(batch_size, num_packs, dtype=torch.int64, device=device)
    
    pack_idx_sorted = torch.empty(batch_size, num_groups, dtype=torch.int64, device=device)
    rank_sorted = torch.empty(batch_size, num_groups, dtype=torch.int64, device=device)
    
    row_indices = torch.arange(batch_size, device=device)
    
    # Vectorized assignment loop
    for i in range(num_groups):
        w = sorted_weight[:, i]
        
        # Mask valid packs (count < limit)
        valid_mask = pack_counts < groups_per_pack
        
        # Select pack with min weight among valid packs
        cand_w = pack_weights.clone()
        cand_w[~valid_mask] = float('inf')
        
        chosen_pack = torch.argmin(cand_w, dim=1)
        chosen_idx = chosen_pack.unsqueeze(1)
        
        # Update weights
        pack_weights.scatter_add_(1, chosen_idx, w.unsqueeze(1))
        
        # Store rank and pack index
        current_rank = pack_counts.gather(1, chosen_idx).squeeze(1)
        rank_sorted[:, i] = current_rank
        pack_idx_sorted[:, i] = chosen_pack
        
        # Update counts
        pack_counts.scatter_add_(1, chosen_idx, torch.ones(batch_size, 1, dtype=torch.int64, device=device))

    # 2. Refinement Setup
    # Reconstruct packs: [B, P, G]
    pack_contents = torch.zeros(batch_size, num_packs, groups_per_pack, device=device)
    pack_item_ids = torch.zeros(batch_size, num_packs, groups_per_pack, dtype=torch.int64, device=device)
    
    flat_b = row_indices.unsqueeze(1).expand(-1, num_groups).flatten()
    flat_p = pack_idx_sorted.flatten()
    flat_r = rank_sorted.flatten()
    
    pack_contents.index_put_((flat_b, flat_p, flat_r), sorted_weight.flatten())
    pack_item_ids.index_put_((flat_b, flat_p, flat_r), sorted_indices.flatten())
    
    # Setup 2-Swap indices if feasible (G <= 16)
    enable_2swap = (groups_per_pack <= 16)
    num_pairs = 0
    pair_idx_i = None
    pair_idx_j = None
    
    if enable_2swap:
        # Generate pairs (i, j) with i < j
        idx_g = torch.arange(groups_per_pack, device=device)
        grid_i, grid_j = torch.meshgrid(idx_g, idx_g, indexing='ij')
        mask = grid_i < grid_j
        pair_idx_i = grid_i[mask]
        pair_idx_j = grid_j[mask]
        num_pairs = pair_idx_i.numel()
        if num_pairs == 0:
            enable_2swap = False

    # Refinement Loop
    for _ in range(20):
        # Recalculate pack weights
        p_weights = pack_contents.sum(dim=-1) # [B, P]
        
        max_val, max_pack = p_weights.max(dim=1)
        min_val, min_pack = p_weights.min(dim=1)
        
        diff = max_val - min_val
        active_mask = diff > 1e-4
        if not active_mask.any():
            break
            
        # --- 1-Swap Candidate Calculation ---
        # Get items from max and min packs: [B, G]
        items_max = pack_contents[row_indices, max_pack]
        items_min = pack_contents[row_indices, min_pack]
        
        # Delta: [B, G, G] (max_item - min_item)
        delta_1 = items_max.unsqueeze(2) - items_min.unsqueeze(1)
        
        # Gain = diff - |diff - 2*delta|
        # Valid if delta > 0 (strictly reducing max pack weight)
        target = diff.view(-1, 1, 1)
        gain_1 = target - (target - 2 * delta_1).abs()
        gain_1 = torch.where(delta_1 > 0, gain_1, -1.0)
        
        best_gain_1, best_idx_1 = gain_1.view(batch_size, -1).max(dim=1)
        
        # --- 2-Swap Candidate Calculation ---
        best_gain_2 = torch.full_like(best_gain_1, -1.0)
        best_idx_2 = torch.zeros_like(best_idx_1)
        
        if enable_2swap:
            # Pair sums: [B, NumPairs]
            p_sum_max = items_max[:, pair_idx_i] + items_max[:, pair_idx_j]
            p_sum_min = items_min[:, pair_idx_i] + items_min[:, pair_idx_j]
            
            # Delta: [B, Pairs, Pairs]
            delta_2 = p_sum_max.unsqueeze(2) - p_sum_min.unsqueeze(1)
            
            target_2 = diff.view(-1, 1, 1)
            gain_2 = target_2 - (target_2 - 2 * delta_2).abs()
            gain_2 = torch.where(delta_2 > 0, gain_2, -1.0)
            
            best_gain_2, best_idx_2 = gain_2.view(batch_size, -1).max(dim=1)
            
        # --- Choose Best Move ---
        # Prioritize 2-swap if it offers better gain
        use_2swap = (best_gain_2 > best_gain_1) & (best_gain_2 > 1e-5) & active_mask
        use_1swap = (~use_2swap) & (best_gain_1 > 1e-5) & active_mask
        
        if not (use_1swap.any() or use_2swap.any()):
            break
            
        # Apply 1-Swap
        if use_1swap.any():
            b = row_indices[use_1swap]
            idx = best_idx_1[use_1swap]
            p_h = max_pack[use_1swap]
            p_l = min_pack[use_1swap]
            
            i_h = idx // groups_per_pack
            i_l = idx % groups_per_pack
            
            # Swap values
            v_h = pack_contents[b, p_h, i_h]
            v_l = pack_contents[b, p_l, i_l]
            pack_contents[b, p_h, i_h] = v_l
            pack_contents[b, p_l, i_l] = v_h
            
            # Swap IDs
            id_h = pack_item_ids[b, p_h, i_h]
            id_l = pack_item_ids[b, p_l, i_l]
            pack_item_ids[b, p_h, i_h] = id_l
            pack_item_ids[b, p_l, i_l] = id_h
            
        # Apply 2-Swap
        if use_2swap.any():
            b = row_indices[use_2swap]
            idx = best_idx_2[use_2swap]
            p_h = max_pack[use_2swap]
            p_l = min_pack[use_2swap]
            
            idx_pair_h = idx // num_pairs
            idx_pair_l = idx % num_pairs
            
            # Indices of items to swap
            h1, h2 = pair_idx_i[idx_pair_h], pair_idx_j[idx_pair_h]
            l1, l2 = pair_idx_i[idx_pair_l], pair_idx_j[idx_pair_l]
            
            # Swap pair 1
            v_h1 = pack_contents[b, p_h, h1]
            v_l1 = pack_contents[b, p_l, l1]
            pack_contents[b, p_h, h1] = v_l1
            pack_contents[b, p_l, l1] = v_h1
            
            id_h1 = pack_item_ids[b, p_h, h1]
            id_l1 = pack_item_ids[b, p_l, l1]
            pack_item_ids[b, p_h, h1] = id_l1
            pack_item_ids[b, p_l, l1] = id_h1
            
            # Swap pair 2
            v_h2 = pack_contents[b, p_h, h2]
            v_l2 = pack_contents[b, p_l, l2]
            pack_contents[b, p_h, h2] = v_l2
            pack_contents[b, p_l, l2] = v_h2
            
            id_h2 = pack_item_ids[b, p_h, h2]
            id_l2 = pack_item_ids[b, p_l, l2]
            pack_item_ids[b, p_h, h2] = id_l2
            pack_item_ids[b, p_l, l2] = id_h2

    # Reconstruction
    pack_index = torch.empty(batch_size, num_groups, dtype=torch.int64, device=device)
    rank_in_pack = torch.empty(batch_size, num_groups, dtype=torch.int64, device=device)
    
    flat_ids = pack_item_ids.view(batch_size, -1)
    
    grid_packs = torch.arange(num_packs, device=device).view(1, num_packs, 1).expand(batch_size, -1, groups_per_pack).reshape(batch_size, -1)
    grid_ranks = torch.arange(groups_per_pack, device=device).view(1, 1, groups_per_pack).expand(batch_size, num_packs, -1).reshape(batch_size, -1)
    
    pack_index.scatter_(1, flat_ids, grid_packs)
    rank_in_pack.scatter_(1, flat_ids, grid_ranks)
    
    return pack_index, rank_in_pack


def replicate_experts(
        weight: torch.Tensor,
        num_phy: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Replicate `num_log` experts to `num_phy` replicas.
    Vectorized over batch dimension.
    """
    batch_size, num_log = weight.shape
    device = weight.device
    
    phy2log = torch.arange(num_phy, dtype=torch.int64, device=device).expand(batch_size, -1).clone()
    rank = torch.zeros(batch_size, num_phy, dtype=torch.int64, device=device)
    logcnt = torch.ones(batch_size, num_log, dtype=torch.int64, device=device)
    row_indices = torch.arange(batch_size, device=device)
    
    current_scores = weight.float()

    for i in range(num_log, num_phy):
        redundant_indices = current_scores.argmax(dim=-1)
        phy2log[:, i] = redundant_indices
        rank[:, i] = logcnt[row_indices, redundant_indices]

        # Update counts
        logcnt[row_indices, redundant_indices] += 1

        # Update scores efficiently
        new_cnt = logcnt[row_indices, redundant_indices].float()
        chosen_weight = weight[row_indices, redundant_indices].float()
        current_scores[row_indices, redundant_indices] = chosen_weight / new_cnt

    return phy2log, rank, logcnt


def rebalance_experts_hierarchical(
    weight: torch.Tensor,
    num_physical_experts: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
):
    # Standard hierarchical flow, but adapted for batched input
    num_layers, num_logical_experts = weight.shape
    group_size = num_logical_experts // num_groups
    groups_per_node = num_groups // num_nodes
    phy_experts_per_gpu = num_physical_experts // num_gpus

    def inverse(perm: torch.Tensor) -> torch.Tensor:
        inv = torch.empty_like(perm)
        inv.scatter_(1, perm, torch.arange(perm.size(1), dtype=torch.int64, device=perm.device).expand(perm.shape))
        return inv

    # Step 1: Pack Groups -> Nodes
    # weight: [B, L]
    tokens_per_group = weight.unflatten(-1, (num_groups, group_size)).sum(-1) # [B, G]
    group_pack_index, group_rank_in_pack = balanced_packing(tokens_per_group, num_nodes)
    
    # Calc map
    base = (group_pack_index * groups_per_node + group_rank_in_pack) * group_size
    offset = torch.arange(group_size, dtype=torch.int64, device=weight.device)
    log2mlog = (base.unsqueeze(-1) + offset).flatten(-2)
    mlog2log = inverse(log2mlog)

    # Step 2: Replicate within Nodes
    # Gather weights [B, L] -> [B, L]
    tokens_per_mlog = weight.gather(-1, mlog2log)
    
    # Flatten batch and nodes: [B * Nodes, LogPerNode]
    log_per_node = num_logical_experts // num_nodes
    phy_per_node = num_physical_experts // num_nodes
    tokens_flat = tokens_per_mlog.view(-1, log_per_node)
    
    phy2mlog_flat, phyrank_flat, mlogcnt_flat = replicate_experts(tokens_flat, phy_per_node)
    
    # Step 3: Pack Replicas -> GPUs
    tokens_per_phy = (tokens_flat / mlogcnt_flat).gather(-1, phy2mlog_flat)
    
    gpu_per_node = num_gpus // num_nodes
    pack_index_flat, rank_in_pack_flat = balanced_packing(tokens_per_phy, gpu_per_node)
    
    # Reconstruct
    phy2pphy_flat = pack_index_flat * phy_experts_per_gpu + rank_in_pack_flat
    pphy2phy_flat = inverse(phy2pphy_flat)
    
    # Restore dimensions
    # phy2mlog: [B, Nodes, PhyPerNode]
    phy2mlog = phy2mlog_flat.view(num_layers, num_nodes, -1)
    pphy2phy = pphy2phy_flat.view(num_layers, num_nodes, -1)
    mlogcnt = mlogcnt_flat.view(num_layers, -1)
    phyrank = phyrank_flat.view(num_layers, num_nodes, -1)
    
    # Global Mapping
    pphy2mlog_local = phy2mlog.gather(2, pphy2phy)
    node_offset = (torch.arange(num_nodes, device=weight.device) * log_per_node).view(1, num_nodes, 1)
    pphy2mlog_global = (pphy2mlog_local + node_offset).flatten(1)
    
    pphy2log = mlog2log.gather(1, pphy2mlog_global)
    pphyrank_global = phyrank.gather(2, pphy2phy).flatten(1)
    
    logcnt_global = torch.empty_like(mlogcnt)
    logcnt_global.scatter_(1, mlog2log, mlogcnt)
    
    return pphy2log, pphyrank_global, logcnt_global


def rebalance_experts(
    weight: torch.Tensor,
    num_replicas: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Entry point with Parallel Randomized Restarts.
    """
    num_layers, num_logical_experts = weight.shape
    device = weight.device
    
    num_restarts = 16 
    
    # 1. Expand inputs
    weight_expanded = weight.float().repeat_interleave(num_restarts, dim=0)
    
    # 2. Add noise to restarts (excluding first one per group)
    if num_restarts > 1:
        noise = torch.rand(num_layers, num_restarts - 1, num_logical_experts, device=device) * 0.05
        
        weight_view = weight_expanded.view(num_layers, num_restarts, -1)
        perturbed = weight_view[:, 1:, :].clone()
        perturbed *= (1.0 + noise)
        
        # We construct a tensor for solving that has noise, but we eval on original
        weight_solving = weight_view.clone()
        weight_solving[:, 1:, :] = perturbed
        weight_solving = weight_solving.reshape(-1, num_logical_experts)
    else:
        weight_solving = weight_expanded
        
    # 3. Solve
    if num_groups % num_nodes == 0:
        phy2log_cand, phyrank_cand, logcnt_cand = rebalance_experts_hierarchical(
            weight_solving, num_replicas, num_groups, num_nodes, num_gpus)
    else:
        phy2log_cand, phyrank_cand, logcnt_cand = rebalance_experts_hierarchical(
            weight_solving, num_replicas, 1, 1, num_gpus)
            
    # 4. Evaluate candidates on ORIGINAL weights
    # Map physical experts back to loads
    # phy2log_cand: [B, NumPhy]. 
    # Experts are ordered by GPU slots: [GPU0...GPU0, GPU1...GPU1, ...]
    
    experts_per_gpu = num_replicas // num_gpus
    
    # Get original weights mapped
    assigned_weights = weight_expanded.gather(1, phy2log_cand)
    counts = logcnt_cand.gather(1, phy2log_cand)
    
    replica_loads = assigned_weights / counts
    
    # Sum by GPU
    gpu_loads = replica_loads.view(-1, num_gpus, experts_per_gpu).sum(dim=-1) # [B, NumGPUs]
    
    # Max load per batch
    max_loads = gpu_loads.max(dim=1).values # [B]
    
    # Reshape to find best restart
    scores = max_loads.view(num_layers, num_restarts)
    best_idx = scores.argmin(dim=1) # [Layers]
    
    # 5. Gather best
    best_batch_idx = torch.arange(num_layers, device=device) * num_restarts + best_idx
    
    phy2log = phy2log_cand[best_batch_idx]
    logcnt = logcnt_cand[best_batch_idx]
    phyrank = phyrank_cand[best_batch_idx]
    
    # 6. Construct log2phy
    num_redundant_experts = num_replicas - num_logical_experts
    maxlogcnt = num_redundant_experts + 1
    
    log2phy = torch.full(
        (num_layers, num_logical_experts, maxlogcnt),
        -1,
        dtype=torch.int64,
        device=device,
    )
    
    scatter_idx = phy2log * maxlogcnt + phyrank
    src = torch.arange(num_replicas, dtype=torch.int64, device=device).expand(num_layers, -1)
    
    log2phy.view(num_layers, -1).scatter_(-1, scatter_idx, src)
    
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