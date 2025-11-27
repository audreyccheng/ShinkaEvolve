# EVOLVE-BLOCK-START
"""
Expert parallelism load balancer (EPLB) for vLLM.

This module implements the core rearrangement algorithm using a
Hybrid Strategy that combines Folded Chunked Greedy and
Parallel Randomized Greedy with Local Search Refinement.
"""

import torch


def _vectorized_greedy_packing(weights: torch.Tensor,
                               num_packs: int,
                               capacity: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Vectorized Greedy Packing Kernel.
    Assigns items to packs to minimize immediate load.
    
    Args:
        weights: [Batch, Items] (sorted/permuted)
        num_packs: int
        capacity: int
        
    Returns:
        pack_ids: [Batch, Items]
        ranks: [Batch, Items]
        pack_loads: [Batch, num_packs]
    """
    batch_size, num_items = weights.shape
    device = weights.device

    # State tracking
    pack_loads = torch.zeros(batch_size, num_packs, device=device, dtype=weights.dtype)
    pack_counts = torch.zeros(batch_size, num_packs, device=device, dtype=torch.int64)
    
    pack_ids = torch.empty(batch_size, num_items, device=device, dtype=torch.int64)
    ranks = torch.empty(batch_size, num_items, device=device, dtype=torch.int64)

    batch_indices = torch.arange(batch_size, device=device)
    inf_tensor = torch.tensor(float('inf'), device=device, dtype=weights.dtype)

    # Greedy allocation loop
    for i in range(num_items):
        w = weights[:, i]
        
        # Mask valid packs (count < capacity)
        valid_mask = pack_counts < capacity
        
        # Add infinity to invalid packs
        # [Batch, Packs]
        temp_loads = torch.where(valid_mask, pack_loads, inf_tensor)
        
        # Choose pack with minimum current load
        chosen_packs = temp_loads.argmin(dim=1)

        # Record assignment
        pack_ids[:, i] = chosen_packs
        ranks[:, i] = pack_counts[batch_indices, chosen_packs]

        # Update state
        pack_counts[batch_indices, chosen_packs] += 1
        pack_loads[batch_indices, chosen_packs] += w

    return pack_ids, ranks, pack_loads


def _refine_packing_swap(weights: torch.Tensor,
                         pack_ids: torch.Tensor,
                         pack_loads: torch.Tensor,
                         ranks: torch.Tensor,
                         num_iters: int = 5) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Refines packing by swapping items between the heaviest and lightest packs.
    """
    batch_size, num_items = weights.shape
    device = weights.device
    batch_indices = torch.arange(batch_size, device=device)

    for _ in range(num_iters):
        # Identify heaviest and lightest packs
        max_load, max_pack_idx = pack_loads.max(dim=1)
        min_load, min_pack_idx = pack_loads.min(dim=1)

        current_imbalance = max_load - min_load

        # Mask items belonging to these packs
        # [Batch, Items]
        is_in_max = (pack_ids == max_pack_idx.unsqueeze(1))
        is_in_min = (pack_ids == min_pack_idx.unsqueeze(1))
        
        # We want to swap item i (from max) with item j (from min)
        # New Max Load = L_max - w_i + w_j
        # New Min Load = L_min - w_j + w_i
        # New Imbalance = |(L_max - L_min) - 2(w_i - w_j)|
        # Target reduction: 2(w_i - w_j) approx (L_max - L_min)
        
        # Compute weight differences matrix: w_i - w_j
        w_diff = weights.unsqueeze(2) - weights.unsqueeze(1)
        
        # Calculate projected new imbalance for every pair
        # mask[b, i, j] = is_in_max[b, i] & is_in_min[b, j]
        mask = is_in_max.unsqueeze(2) & is_in_min.unsqueeze(1)
        
        # New diff metric: |current_imbalance - 2 * (w_i - w_j)|
        projected_imbalance = torch.abs(current_imbalance.view(-1, 1, 1) - 2 * w_diff)
        
        # Apply mask
        projected_imbalance = torch.where(mask, projected_imbalance, torch.tensor(float('inf'), device=device))
        
        # Find best swap per batch
        flat_imbalance = projected_imbalance.view(batch_size, -1)
        best_new_imbalance, best_indices = flat_imbalance.min(dim=1)
        
        # Only apply if strict improvement
        improvement_mask = best_new_imbalance < current_imbalance
        
        if not improvement_mask.any():
            break
            
        active_batch_idx = batch_indices[improvement_mask]
        active_flat_idx = best_indices[improvement_mask]
        
        # Decode indices
        i_idx = active_flat_idx // num_items
        j_idx = active_flat_idx % num_items
        
        # Perform updates
        p_max = max_pack_idx[active_batch_idx]
        p_min = min_pack_idx[active_batch_idx]
        
        w_i = weights[active_batch_idx, i_idx]
        w_j = weights[active_batch_idx, j_idx]
        
        # Swap assignments
        pack_ids[active_batch_idx, i_idx] = p_min
        pack_ids[active_batch_idx, j_idx] = p_max
        
        # Swap ranks
        r_i = ranks[active_batch_idx, i_idx]
        r_j = ranks[active_batch_idx, j_idx]
        ranks[active_batch_idx, i_idx] = r_j
        ranks[active_batch_idx, j_idx] = r_i
        
        # Update loads
        delta = w_i - w_j
        pack_loads[active_batch_idx, p_max] -= delta
        pack_loads[active_batch_idx, p_min] += delta
        
    return pack_ids, ranks, pack_loads


def balanced_packing(weight: torch.Tensor,
                     num_packs: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pack n weighted objects to m packs using a Hybrid Ensemble Strategy.
    """
    num_layers, num_items = weight.shape
    device = weight.device
    capacity = num_items // num_packs

    # --- Strategy 1: Folded Chunked Sorted Greedy ---
    sorted_weights, sorted_indices = weight.sort(dim=-1, descending=True)
    
    loads1 = torch.zeros(num_layers, num_packs, device=device, dtype=weight.dtype)
    ids1 = torch.empty(num_layers, num_items, device=device, dtype=torch.int64)
    ranks1 = torch.empty(num_layers, num_items, device=device, dtype=torch.int64)

    double_chunk_size = 2 * num_packs
    num_double_chunks = num_items // double_chunk_size

    for k in range(num_double_chunks):
        start = k * double_chunk_size
        end = start + double_chunk_size
        
        chunk_weights = sorted_weights[:, start:end]
        
        idx_low = torch.arange(num_packs, device=device)
        idx_high = torch.arange(double_chunk_size - 1, num_packs - 1, -1, device=device)
        
        pair_weights = chunk_weights[:, idx_low] + chunk_weights[:, idx_high]
        
        _, sorted_bin_indices = loads1.sort(dim=-1)
        _, pairs_order = pair_weights.sort(dim=-1, descending=True)
        
        assigned_packs = torch.empty_like(sorted_bin_indices)
        assigned_packs.scatter_(1, pairs_order, sorted_bin_indices)
        
        ids1[:, start + idx_low] = assigned_packs
        ids1[:, start + idx_high] = assigned_packs
        ranks1[:, start + idx_low] = 2 * k
        ranks1[:, start + idx_high] = 2 * k + 1
        
        loads1.scatter_add_(1, assigned_packs, pair_weights)

    remainder_start = num_double_chunks * double_chunk_size
    if remainder_start < num_items:
        start = remainder_start
        end = num_items
        chunk_weights = sorted_weights[:, start:end]
        _, sorted_bin_indices = loads1.sort(dim=-1)
        ids1[:, start:end] = sorted_bin_indices
        ranks1[:, start:end] = 2 * num_double_chunks
        loads1.scatter_add_(1, sorted_bin_indices, chunk_weights)
        
    ids1_final = torch.empty_like(ids1)
    ranks1_final = torch.empty_like(ranks1)
    ids1_final.scatter_(1, sorted_indices, ids1)
    ranks1_final.scatter_(1, sorted_indices, ranks1)

    # --- Strategy 2: Parallel Randomized Greedy + Refinement ---
    num_candidates = 128
    
    # Generate noisy weights: LPT base + Random Noise
    noise = torch.rand(num_layers, num_candidates - 1, num_items, device=device) * 0.4 + 0.8
    ones = torch.ones(num_layers, 1, num_items, device=device)
    scales = torch.cat([ones, noise], dim=1)
    
    perturbed_weights = weight.unsqueeze(1) * scales
    
    # Sort perturbed weights
    cand_sorted_weights, cand_sorted_idx = perturbed_weights.sort(dim=-1, descending=True)
    
    # Retrieve original weights in permuted order for accurate load calculation
    orig_expanded = weight.unsqueeze(1).expand(-1, num_candidates, -1)
    aligned_weights = orig_expanded.gather(2, cand_sorted_idx)
    
    # Flatten for batch processing
    flat_weights = aligned_weights.view(-1, num_items)
    
    # Greedy Packing
    flat_ids, flat_ranks, flat_loads = _vectorized_greedy_packing(flat_weights, num_packs, capacity)
    
    # Refinement (Swap)
    flat_ids, flat_ranks, flat_loads = _refine_packing_swap(flat_weights, flat_ids, flat_loads, flat_ranks, num_iters=3)
    
    # --- Selection ---
    loads2_all = flat_loads.view(num_layers, num_candidates, num_packs)
    imbalance2_all = loads2_all.max(dim=-1).values - loads2_all.min(dim=-1).values
    
    best_cand_idx = imbalance2_all.argmin(dim=1) # [Layers]
    
    batch_indices = torch.arange(num_layers, device=device)
    
    ids2_all = flat_ids.view(num_layers, num_candidates, num_items)
    ranks2_all = flat_ranks.view(num_layers, num_candidates, num_items)
    indices2_all = cand_sorted_idx 
    
    ids2 = ids2_all[batch_indices, best_cand_idx]
    ranks2 = ranks2_all[batch_indices, best_cand_idx]
    indices2 = indices2_all[batch_indices, best_cand_idx]
    
    ids2_restored = torch.empty_like(ids1)
    ranks2_restored = torch.empty_like(ranks1)
    
    ids2_restored.scatter_(1, indices2, ids2)
    ranks2_restored.scatter_(1, indices2, ranks2)
    
    # Recalculate loads for Strat 2
    loads2 = torch.zeros_like(loads1)
    loads2.scatter_add_(1, ids2_restored, weight)
    
    # Final Selection
    imbalance1 = loads1.max(dim=-1).values - loads1.min(dim=-1).values
    imbalance2 = loads2.max(dim=-1).values - loads2.min(dim=-1).values
    
    use_strategy2 = (imbalance2 < imbalance1).unsqueeze(-1)
    
    final_ids = torch.where(use_strategy2, ids2_restored, ids1_final)
    final_ranks = torch.where(use_strategy2, ranks2_restored, ranks1_final)
    
    return final_ids, final_ranks


def replicate_experts(
        weight: torch.Tensor,
        num_phy: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Replicate experts using Binary Search on Max Load followed by Greedy Refinement.
    """
    num_layers, num_log = weight.shape
    device = weight.device

    if num_phy == num_log:
        phy2log = torch.arange(num_log, device=device).expand(num_layers, -1)
        rank = torch.zeros(num_layers, num_phy, dtype=torch.int64, device=device)
        logcnt = torch.ones(num_layers, num_log, dtype=torch.int64, device=device)
        return phy2log, rank, logcnt

    low = weight.sum(dim=-1, keepdim=True) / num_phy
    high = weight.max(dim=-1, keepdim=True).values
    low = torch.max(low, torch.tensor(1e-6, device=device))

    for _ in range(15):
        mid = (low + high) * 0.5
        counts = torch.ceil(weight / mid)
        total = counts.sum(dim=-1, keepdim=True)
        mask = total <= num_phy
        high = torch.where(mask, mid, high)
        low = torch.where(mask, low, mid)

    logcnt = torch.ceil(weight / high).long().clamp(min=1)

    current_sum = logcnt.sum(dim=-1)
    diff = num_phy - current_sum

    max_diff = int(diff.max().item())
    if max_diff > 0:
        rows = torch.arange(num_layers, device=device)
        for _ in range(max_diff):
            active = current_sum < num_phy
            if not active.any(): break
            
            density = weight / logcnt.float()
            target_idx = density.argmax(dim=-1)
            
            active_rows = rows[active]
            active_targets = target_idx[active]
            
            logcnt.index_put_((active_rows, active_targets), 
                              torch.tensor(1, device=device, dtype=torch.int64), 
                              accumulate=True)
            current_sum[active] += 1

    min_diff = int(diff.min().item())
    if min_diff < 0:
        rows = torch.arange(num_layers, device=device)
        for _ in range(abs(min_diff)):
            active = current_sum > num_phy
            if not active.any(): break
            
            valid = logcnt > 1
            cost = weight / (logcnt - 1).float()
            cost[~valid] = float('inf')
            
            target_idx = cost.argmin(dim=-1)
            
            active_rows = rows[active]
            active_targets = target_idx[active]
            
            logcnt.index_put_((active_rows, active_targets), 
                              torch.tensor(-1, device=device, dtype=torch.int64), 
                              accumulate=True)
            current_sum[active] -= 1

    flat_log_ids = torch.arange(num_log, device=device).repeat(num_layers)
    flat_counts = logcnt.flatten()
    flat_phy2log = torch.repeat_interleave(flat_log_ids, flat_counts)
    
    target_size = num_layers * num_phy
    if flat_phy2log.numel() != target_size:
        if flat_phy2log.numel() < target_size:
            flat_phy2log = torch.cat([flat_phy2log, torch.zeros(target_size - flat_phy2log.numel(), device=device, dtype=torch.long)])
        else:
            flat_phy2log = flat_phy2log[:target_size]
            
    phy2log = flat_phy2log.view(num_layers, num_phy)
    
    offsets = torch.zeros_like(logcnt)
    offsets[:, 1:] = logcnt[:, :-1].cumsum(dim=1)
    mapped_offsets = offsets.gather(1, phy2log)
    phy_indices = torch.arange(num_phy, device=device).expand(num_layers, -1)
    rank = phy_indices - mapped_offsets

    return phy2log, rank, logcnt


def rebalance_experts_hierarchical(
    weight: torch.Tensor,
    num_physical_experts: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
):
    """
    Hierarchical rebalancing.
    """
    num_layers, num_logical_experts = weight.shape
    group_size = num_logical_experts // num_groups
    groups_per_node = num_groups // num_nodes
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

    # 1. Groups -> Nodes
    tokens_per_group = weight.unflatten(-1, (num_groups, group_size)).sum(-1)
    group_pack_index, group_rank_in_pack = balanced_packing(
        tokens_per_group, num_nodes)
        
    log2mlog = (((group_pack_index * groups_per_node + group_rank_in_pack) *
                 group_size).unsqueeze(-1) +
                torch.arange(group_size,
                             dtype=torch.int64,
                             device=group_pack_index.device)).flatten(-2)
    mlog2log = inverse(log2mlog)

    # 2. Replicate within nodes
    tokens_per_mlog = weight.gather(-1, mlog2log).view(
        -1, num_logical_experts // num_nodes)
    phy2mlog, phyrank, mlogcnt = replicate_experts(
        tokens_per_mlog, num_physical_experts // num_nodes)

    # 3. Physical -> GPUs
    tokens_per_phy = (tokens_per_mlog / mlogcnt).gather(-1, phy2mlog)
    pack_index, rank_in_pack = balanced_packing(tokens_per_phy,
                                                num_gpus // num_nodes)
    
    phy2pphy = pack_index * phy_experts_per_gpu + rank_in_pack
    pphy2phy = inverse(phy2pphy)

    pphy2mlog = phy2mlog.gather(-1, pphy2phy)
    
    node_offsets = torch.arange(
        0,
        num_logical_experts,
        num_logical_experts // num_nodes,
        device=weight.device,
    ).view(1, -1, 1)
    
    pphy2mlog_restored = (pphy2mlog.view(num_layers, num_nodes, -1) + node_offsets).flatten(-2)
    
    pphy2log = mlog2log.gather(-1, pphy2mlog_restored)
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
    Entry point.
    """
    num_layers, num_logical_experts = weight.shape
    weight = weight.float().cpu()
    
    if num_groups % num_nodes == 0:
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, num_groups, num_nodes, num_gpus)
    else:
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, 1, 1, num_gpus)
            
    max_replicas = int(logcnt.max().item())
    
    log2phy = torch.full(
        (num_layers, num_logical_experts, max_replicas),
        -1,
        dtype=torch.int64,
        device=logcnt.device,
    )
    
    flat_layer_idx = torch.arange(num_layers, device=logcnt.device).unsqueeze(-1).expand(-1, num_replicas).flatten()
    flat_log_idx = phy2log.flatten()
    flat_rank_idx = phyrank.flatten()
    flat_phy_ids = torch.arange(num_replicas, dtype=torch.int64, device=logcnt.device).expand(num_layers, -1).flatten()
    
    indices = (flat_layer_idx * num_logical_experts * max_replicas) + \
              (flat_log_idx * max_replicas) + \
              flat_rank_idx
              
    log2phy.view(-1).scatter_(0, indices, flat_phy_ids)
    
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
