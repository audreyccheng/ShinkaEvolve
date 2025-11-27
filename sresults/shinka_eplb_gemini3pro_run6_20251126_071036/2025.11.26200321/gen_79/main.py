# EVOLVE-BLOCK-START
"""
Expert parallelism load balancer (EPLB) for vLLM.

This module implements the core rearrangement algorithm using a
Hybrid Stochastic Ensemble strategy with Move-then-Swap Refinement.
"""

import torch


def _refine_packing_swap(weight: torch.Tensor,
                         pack_index: torch.Tensor,
                         rank_in_pack: torch.Tensor,
                         num_packs: int,
                         num_iters: int = 10) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Iterative Max-Any Swap refinement.
    Attempts to swap items between the max-load pack and any other pack to reduce max load.
    """
    batch_size, num_items = weight.shape
    device = weight.device
    batch_range = torch.arange(batch_size, device=device)

    # Compute loads
    pack_loads = torch.zeros(batch_size, num_packs, device=device, dtype=weight.dtype)
    pack_loads.scatter_add_(1, pack_index, weight)

    # Pre-compute weight differences: w[b, i] - w[b, j]
    # [B, N, N]
    w_diff = weight.unsqueeze(2) - weight.unsqueeze(1) 

    for _ in range(num_iters):
        max_load, max_idx = pack_loads.max(dim=1)
        
        # Identify items in the max pack
        is_max = (pack_index == max_idx.unsqueeze(1)) # [B, N]
        
        # For every item j (potential swap partner), get its pack and that pack's load
        p_j = pack_index # [B, N]
        l_j = pack_loads.gather(1, p_j) # [B, N]
        
        # Gap = L_max - L_pack(j)
        gap = max_load.unsqueeze(1) - l_j # [B, N]
        
        # We want to swap item i (from Max) with item j (from Other)
        # Gain = min(w_i - w_j, gap - (w_i - w_j))
        # This formula maximizes the reduction of the max load while ensuring the target pack doesn't exceed old max.
        
        delta = w_diff # [B, N, N] -> w[i] - w[j]
        
        # Valid: i in Max, j NOT in Max
        # We also implicitly handle the case where j is in Max (valid_mask is False)
        valid_mask = is_max.unsqueeze(2) & (~is_max.unsqueeze(1))
        
        gap_expanded = gap.unsqueeze(1) # [B, 1, N]
        gain = torch.min(delta, gap_expanded - delta)
        
        # Filter strictly positive gains
        mask = valid_mask & (delta > 1e-5) & (gain > 1e-5)
        
        gain = torch.where(mask, gain, torch.tensor(-1.0, device=device))
        
        # Find best swap per batch
        flat_gain = gain.view(batch_size, -1)
        best_val, best_flat = flat_gain.max(dim=1)
        
        # Stop if no improvement
        if not (best_val > 1e-5).any():
            break
            
        improve = best_val > 1e-5
        active_batch = batch_range[improve]
        active_flat = best_flat[improve]
        
        idx_i = active_flat // num_items
        idx_j = active_flat % num_items
        
        p_max = max_idx[active_batch]
        p_other = pack_index[active_batch, idx_j]
        
        # Apply swap
        d = weight[active_batch, idx_i] - weight[active_batch, idx_j]
        pack_loads[active_batch, p_max] -= d
        pack_loads[active_batch, p_other] += d
        
        pack_index[active_batch, idx_i] = p_other
        pack_index[active_batch, idx_j] = p_max
        
        # Swap ranks
        r_i = rank_in_pack[active_batch, idx_i]
        r_j = rank_in_pack[active_batch, idx_j]
        rank_in_pack[active_batch, idx_i] = r_j
        rank_in_pack[active_batch, idx_j] = r_i
        
    return pack_index, rank_in_pack


def _vectorized_greedy_packing(weight: torch.Tensor,
                               num_packs: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Vectorized Greedy packing kernel.
    """
    batch_size, num_items = weight.shape
    device = weight.device
    capacity = num_items // num_packs
    
    pack_loads = torch.zeros(batch_size, num_packs, device=device, dtype=weight.dtype)
    pack_counts = torch.zeros(batch_size, num_packs, device=device, dtype=torch.int64)
    
    pack_ids = torch.empty(batch_size, num_items, device=device, dtype=torch.int64)
    ranks = torch.empty(batch_size, num_items, device=device, dtype=torch.int64)
    
    batch_range = torch.arange(batch_size, device=device)
    inf = torch.tensor(float('inf'), device=device, dtype=weight.dtype)
    
    for i in range(num_items):
        w = weight[:, i]
        valid = pack_counts < capacity
        loads = torch.where(valid, pack_loads, inf)
        chosen = loads.argmin(dim=1)
        
        pack_ids[:, i] = chosen
        ranks[:, i] = pack_counts[batch_range, chosen]
        
        pack_loads[batch_range, chosen] += w
        pack_counts[batch_range, chosen] += 1
        
    return pack_ids, ranks, pack_loads


def balanced_packing(weight: torch.Tensor,
                     num_packs: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Hybrid Stochastic Ensemble Strategy.
    Generates diverse candidates using LPT, ZigZag, Random, and Noisy Weights.
    Selects best candidates and refines them.
    """
    num_layers, num_items = weight.shape
    device = weight.device
    
    # Configuration
    num_candidates = 128
    
    # 1. Candidate Generation
    
    # Base LPT
    lpt_val, lpt_idx = weight.sort(dim=-1, descending=True)
    
    # ZigZag
    zigzag_perm = torch.empty(num_items, device=device, dtype=torch.long)
    half = (num_items + 1) // 2
    arange = torch.arange(num_items, device=device)
    zigzag_perm[0::2] = arange[:half]
    zigzag_perm[1::2] = arange[half:].flip(0)
    zigzag_idx = lpt_idx.gather(1, zigzag_perm.view(1, -1).expand(num_layers, -1))
    
    # Random Permutations (30 candidates)
    num_rand = 30
    rand_idx = torch.rand(num_layers, num_rand, num_items, device=device).argsort(dim=-1)
    
    # Noisy Weights (96 candidates)
    # Varying noise levels to explore different neighborhoods
    num_noisy = 96
    
    # Low noise (preserve LPT mostly): 0.01 - 0.1
    noise_low = torch.rand(num_layers, num_noisy // 2, num_items, device=device) * 0.1
    # High noise (more exploration): 0.1 - 0.5
    noise_high = torch.rand(num_layers, num_noisy - (num_noisy // 2), num_items, device=device) * 0.4 + 0.1
    
    noise = torch.cat([noise_low, noise_high], dim=1) + 1.0
    
    noisy_weight = weight.unsqueeze(1) * noise
    _, noisy_idx = noisy_weight.sort(dim=-1, descending=True)
    
    # Combine Indices
    all_indices = torch.cat([
        lpt_idx.unsqueeze(1),
        zigzag_idx.unsqueeze(1),
        rand_idx,
        noisy_idx
    ], dim=1)
    
    # Gather weights [Layers, Candidates, Items]
    expanded_weight = weight.unsqueeze(1).expand(-1, num_candidates, -1)
    ordered_weights = expanded_weight.gather(2, all_indices)
    
    # Flatten for Kernel
    flat_weights = ordered_weights.view(-1, num_items)
    
    # 2. Greedy Packing
    flat_ids, flat_ranks, flat_loads = _vectorized_greedy_packing(flat_weights, num_packs)
    
    # 3. Top-K Selection
    loads = flat_loads.view(num_layers, num_candidates, num_packs)
    imbalance = loads.max(dim=-1).values - loads.min(dim=-1).values
    
    k = 8
    best_vals, best_k_idx = imbalance.topk(k, dim=1, largest=False)
    
    # Flatten Selection Indices
    layer_offsets = (torch.arange(num_layers, device=device) * num_candidates).unsqueeze(1)
    flat_select = (best_k_idx + layer_offsets).flatten()
    
    refined_weights = flat_weights[flat_select]
    refined_ids = flat_ids[flat_select]
    refined_ranks = flat_ranks[flat_select]
    
    # 4. Refinement
    refined_ids, refined_ranks = _refine_packing_swap(refined_weights, refined_ids, refined_ranks, num_packs, num_iters=10)
    
    # 5. Final Selection
    # Recalculate loads
    refined_loads = torch.zeros(num_layers * k, num_packs, device=device, dtype=weight.dtype)
    refined_loads.scatter_add_(1, refined_ids, refined_weights)
    
    final_imbalance = refined_loads.max(dim=1).values - refined_loads.min(dim=1).values
    final_imbalance = final_imbalance.view(num_layers, k)
    
    best_in_k = final_imbalance.argmin(dim=1)
    
    # 6. Scatter Back
    # Find the original permutation for the winner
    winner_cand_idx = best_k_idx.gather(1, best_in_k.unsqueeze(1)).squeeze(1)
    
    idx_view = winner_cand_idx.view(num_layers, 1, 1).expand(-1, 1, num_items)
    final_sorted_idx = all_indices.gather(1, idx_view).squeeze(1)
    
    # Get winner's packing assignment
    winner_flat_idx = (torch.arange(num_layers, device=device) * k) + best_in_k
    final_ids_aligned = refined_ids[winner_flat_idx]
    final_ranks_aligned = refined_ranks[winner_flat_idx]
    
    pack_index = torch.empty(num_layers, num_items, device=device, dtype=torch.int64)
    rank_in_pack = torch.empty(num_layers, num_items, device=device, dtype=torch.int64)
    
    pack_index.scatter_(1, final_sorted_idx, final_ids_aligned)
    rank_in_pack.scatter_(1, final_sorted_idx, final_ranks_aligned)
    
    return pack_index, rank_in_pack


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

    # Binary Search
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

    # Correct sum
    current_sum = logcnt.sum(dim=-1)
    diff = num_phy - current_sum

    # Under-allocation: Add to max density
    max_diff = int(diff.max().item())
    if max_diff > 0:
        rows = torch.arange(num_layers, device=device)
        for _ in range(max_diff):
            active = current_sum < num_phy
            if not active.any(): break
            
            density = weight / logcnt.float()
            density[~active] = -1.0
            
            target_idx = density.argmax(dim=-1)
            
            active_rows = rows[active]
            active_targets = target_idx[active]
            
            logcnt.index_put_((active_rows, active_targets), 
                              torch.tensor(1, device=device, dtype=torch.int64), 
                              accumulate=True)
            current_sum[active] += 1

    # Over-allocation: Remove from min cost
    min_diff = int(diff.min().item())
    if min_diff < 0:
        rows = torch.arange(num_layers, device=device)
        for _ in range(abs(min_diff)):
            active = current_sum > num_phy
            if not active.any(): break
            
            valid = logcnt > 1
            cost = weight / (logcnt - 1).float()
            cost[~valid] = float('inf')
            cost[~active] = float('inf')
            
            target_idx = cost.argmin(dim=-1)
            
            active_rows = rows[active]
            active_targets = target_idx[active]
            
            logcnt.index_put_((active_rows, active_targets), 
                              torch.tensor(-1, device=device, dtype=torch.int64), 
                              accumulate=True)
            current_sum[active] -= 1

    # Construct maps
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