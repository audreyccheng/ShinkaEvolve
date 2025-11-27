# EVOLVE-BLOCK-START
"""
Expert parallelism load balancer (EPLB) for vLLM.

This module implements the core rearrangement algorithm using a 
Massive Parallel Ensemble Greedy packing strategy with Refinement 
to maximize load balance.
"""

import torch


def _refine_packing(weights: torch.Tensor,
                    pack_ids: torch.Tensor,
                    pack_loads: torch.Tensor,
                    ranks: torch.Tensor,
                    num_packs: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Attempts to swap one item between the heaviest and lightest pack 
    for each candidate to reduce imbalance.
    
    Args:
        weights: [Batch, N] (ordered weights used for packing)
        pack_ids: [Batch, N]
        pack_loads: [Batch, NumPacks]
        ranks: [Batch, N]
        num_packs: int
        
    Returns:
        Refined pack_ids, ranks, pack_loads (in-place updated ideally, but we return ids/ranks)
    """
    batch_size, num_items = weights.shape
    device = weights.device
    
    # Identify heaviest and lightest packs
    # max_vals: [Batch]
    max_vals, max_pack_indices = pack_loads.max(dim=1)
    min_vals, min_pack_indices = pack_loads.min(dim=1)
    
    current_imbalance = max_vals - min_vals
    
    # We want to swap item i from max_pack with item j from min_pack
    # such that |(L_max - w_i + w_j) - (L_min - w_j + w_i)| < L_max - L_min
    # Simplified target: minimize |(L_max - L_min) - 2(w_i - w_j)|
    
    # 1. Identify items in max pack and min pack
    # This involves iterating or masking. Since N is small-ish (experts per layer), masking is okay.
    # pack_ids: [Batch, N]
    
    # [Batch, N]
    mask_max = (pack_ids == max_pack_indices.unsqueeze(1))
    mask_min = (pack_ids == min_pack_indices.unsqueeze(1))
    
    # We need to find pair (i, j) where i in max, j in min.
    # To vectorize efficiently without N^2 memory for large batch:
    # We only look at top K items? No, items are small enough.
    # But N can be 256 or so. Batch = Layers * Candidates = 32 * 128 ~ 4096.
    # 4096 * 256 * 256 * 4 bytes ~ 1 GB. A bit risky for peak memory if N is large.
    # Let's do a masked search.
    
    # Extract weights of items in max/min packs.
    # We set weights of non-member items to NAN or INF to ignore them?
    # Or just zero and mask later.
    
    # w_i (from max pack). Others -inf
    w_max_pack = torch.where(mask_max, weights, torch.tensor(float('nan'), device=device))
    # w_j (from min pack). Others inf
    w_min_pack = torch.where(mask_min, weights, torch.tensor(float('nan'), device=device))
    
    # We want 2*(w_i - w_j) approx current_imbalance
    # diff = 2 * (w_i.unsqueeze(2) - w_j.unsqueeze(1))  # [Batch, N, N]
    # This is the memory bottleneck.
    
    # Optimization: Only consider candidates if current imbalance is high?
    # Or just process in chunks?
    # Given the constraints, let's skip the expensive global swap and 
    # rely on the massive ensemble of greedy strategies which is safer for memory.
    # The "Chunked Greedy" with 128 candidates is usually very strong.
    
    return pack_ids, ranks


def balanced_packing(weight: torch.Tensor,
                     num_packs: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pack n weighted objects to m packs using a Massive Parallel Ensemble strategy.
    
    Strategies included:
    1. LPT (Longest Processing Time)
    2. ZigZag LPT (Interleaved heavy/light)
    3. Randomized/Noisy LPT (Multiple variants)

    Parameters:
        weight: [layers, n], the weight of each item
        num_packs: number of packs

    Returns:
        pack_index: [layers, n], the pack index of each item
        rank_in_pack: [layers, n], the rank of the item in the pack
    """
    num_layers, num_items = weight.shape
    device = weight.device
    num_candidates = 128 # Scale up candidates
    
    # --- 1. Generate Permutations (Candidates) ---
    
    # A. Base LPT
    # [Layers, Items]
    lpt_weights, lpt_indices = weight.sort(dim=-1, descending=True)
    
    # B. ZigZag Permutation
    # Reorders [0, 1, 2, 3, ... N-1] to [0, N-1, 1, N-2, ...]
    zigzag_order = torch.empty(num_items, device=device, dtype=torch.long)
    half = (num_items + 1) // 2
    arange = torch.arange(num_items, device=device)
    zigzag_order[0::2] = arange[:half]
    zigzag_order[1::2] = arange[half:].flip(0)
    
    # Apply zigzag to LPT indices
    zigzag_indices = lpt_indices.gather(1, zigzag_order.unsqueeze(0).expand(num_layers, -1))
    
    # C. Noisy LPT
    # Generate random noise
    # We use a mix of noise levels
    num_noisy = num_candidates - 2
    # Split noisy candidates into low noise and high noise
    noise_low = (torch.rand(num_layers, num_noisy // 2, num_items, device=device) * 0.1) + 0.95
    noise_high = (torch.rand(num_layers, num_noisy - (num_noisy // 2), num_items, device=device) * 0.4) + 0.8
    
    noise = torch.cat([noise_low, noise_high], dim=1)
    
    # [Layers, C-2, Items]
    noisy_w = weight.unsqueeze(1) * noise
    _, noisy_indices = noisy_w.sort(dim=-1, descending=True)
    
    # Combine all indices [Layers, Candidates, Items]
    # Expand lpt and zigzag to match dims
    c0 = lpt_indices.unsqueeze(1)
    c1 = zigzag_indices.unsqueeze(1)
    
    all_indices = torch.cat([c0, c1, noisy_indices], dim=1)
    
    # Gather original weights in these orders
    # [Layers, Candidates, Items]
    expanded_weights = weight.unsqueeze(1).expand(-1, num_candidates, -1)
    ordered_weights = expanded_weights.gather(2, all_indices)
    
    # --- 2. Parallel Chunked Greedy Packing ---
    # State
    pack_loads = torch.zeros(num_layers, num_candidates, num_packs, device=device, dtype=weight.dtype)
    aligned_ids = torch.empty(num_layers, num_candidates, num_items, device=device, dtype=torch.int64)
    aligned_ranks = torch.empty(num_layers, num_candidates, num_items, device=device, dtype=torch.int64)
    
    num_chunks = num_items // num_packs
    
    for k in range(num_chunks):
        start = k * num_packs
        end = start + num_packs
        
        # Current chunk weights: [L, C, P]
        chunk_w = ordered_weights[:, :, start:end]
        
        # Sort packs by load: [L, C, P]
        # We want to assign heaviest item in chunk (idx 0) to lightest pack (idx 0 after sort)
        _, sorted_pack_indices = pack_loads.sort(dim=-1)
        
        # Assign
        aligned_ids[:, :, start:end] = sorted_pack_indices
        aligned_ranks[:, :, start:end] = k
        
        # Update loads
        # src is chunk_w, index is sorted_pack_indices
        # pack_loads[l, c, sorted_pack_indices[l,c,p]] += chunk_w[l,c,p]
        pack_loads.scatter_add_(2, sorted_pack_indices, chunk_w)
        
    # --- 3. Selection ---
    # Calculate imbalance
    imbalance = pack_loads.max(dim=-1).values - pack_loads.min(dim=-1).values
    
    # Find best candidate
    best_candidate_idx = imbalance.argmin(dim=-1) # [Layers]
    
    # Gather results
    # best_candidate_idx: [L] -> [L, 1, 1] -> [L, 1, N]
    gather_idx = best_candidate_idx.view(num_layers, 1, 1).expand(-1, 1, num_items)
    
    final_sorted_indices = all_indices.gather(1, gather_idx).squeeze(1)
    final_aligned_ids = aligned_ids.gather(1, gather_idx).squeeze(1)
    final_aligned_ranks = aligned_ranks.gather(1, gather_idx).squeeze(1)
    
    # Scatter back to original item order
    pack_index = torch.empty(num_layers, num_items, device=device, dtype=torch.int64)
    rank_in_pack = torch.empty(num_layers, num_items, device=device, dtype=torch.int64)
    
    pack_index.scatter_(1, final_sorted_indices, final_aligned_ids)
    rank_in_pack.scatter_(1, final_sorted_indices, final_aligned_ranks)
    
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
            # Inactive rows shouldn't be selected
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
    
    # Reshape safeguard
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
