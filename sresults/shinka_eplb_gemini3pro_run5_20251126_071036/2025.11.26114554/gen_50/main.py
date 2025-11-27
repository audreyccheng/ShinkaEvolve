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

def _refine_packing_batched(weights: torch.Tensor,
                            pack_indices: torch.Tensor,
                            pack_weights: torch.Tensor,
                            num_packs: int,
                            groups_per_pack: int,
                            max_iters: int = 50) -> torch.Tensor:
    """
    Refines a batch of packings by iteratively swapping items to minimize load variance.
    
    Args:
        weights: [B, num_groups] Weights of items.
        pack_indices: [B, num_packs, groups_per_pack] Item IDs assigned to packs.
        pack_weights: [B, num_packs] Current weights of packs.
        
    Returns:
        Refined pack_indices.
    """
    B = weights.shape[0]
    device = weights.device
    num_items = num_packs * groups_per_pack
    
    # Pre-compute mapping from flat item position to pack ID
    # [num_items] e.g. [0,0,1,1,2,2...]
    item_to_pack = torch.arange(num_packs, device=device).repeat_interleave(groups_per_pack)
    
    # Mask for self-swaps (same pack)
    # [1, N, N]
    same_pack_mask = (item_to_pack.unsqueeze(0) == item_to_pack.unsqueeze(1)).unsqueeze(0)
    
    batch_idx = torch.arange(B, device=device)
    
    for _ in range(max_iters):
        # 1. Flatten indices: [B, num_items]
        flat_indices = pack_indices.view(B, -1)
        
        # 2. Gather item weights: [B, num_items]
        w_current = torch.gather(weights, 1, flat_indices)
        
        # 3. Gather pack loads for each item position: [B, num_items]
        # We need L[b, item_pos] = pack_weights[b, item_to_pack[item_pos]]
        L_current = pack_weights[:, item_to_pack]
        
        # 4. Calculate pairwise Delta matrix: D[b, i, j] = w_i - w_j
        # We broadcast: [B, N, 1] - [B, 1, N]
        D = w_current.unsqueeze(2) - w_current.unsqueeze(1)
        
        # 5. Calculate Load Diff: L_diff[b, i, j] = L_i - L_j
        L_diff = L_current.unsqueeze(2) - L_current.unsqueeze(1)
        
        # 6. Cost Change (Variance): 2 * D * (D + L_diff)
        # We want to minimize this.
        cost = 2 * D * (D + L_diff)
        
        # 7. Apply mask (ignore same-pack swaps)
        cost.masked_fill_(same_pack_mask, float('inf'))
        
        # 8. Find best swap per batch
        # Flatten last two dims: [B, N*N]
        cost_flat = cost.view(B, -1)
        min_val, min_idx = torch.min(cost_flat, dim=1)
        
        # 9. Filter active batches (those with improvement)
        active_mask = min_val < -1e-6
        if not active_mask.any():
            break
            
        active_batches = batch_idx[active_mask]
        best_swaps = min_idx[active_mask]
        
        # 10. Decode indices
        idx_i = best_swaps // num_items
        idx_j = best_swaps % num_items
        
        pid_i = item_to_pack[idx_i]
        pid_j = item_to_pack[idx_j]
        
        # 11. Execute Swap
        # We manipulate the flat view of pack_indices directly
        
        # Get item IDs
        item_i = flat_indices[active_batches, idx_i]
        item_j = flat_indices[active_batches, idx_j]
        
        # Swap in indices tensor
        flat_indices[active_batches, idx_i] = item_j
        flat_indices[active_batches, idx_j] = item_i
        
        # Persist changes to pack_indices view (flat_indices is a view, so modifying it in-place works)
        # pack_indices.view(B, -1)[:] = flat_indices
        
        # 12. Update pack weights
        d_val = D[active_batches, idx_i, idx_j]
        pack_weights[active_batches, pid_i] += d_val
        pack_weights[active_batches, pid_j] -= d_val

    return pack_indices


def balanced_packing(weight: torch.Tensor,
                     num_packs: int,
                     num_candidates: int = 128) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pack items to packs with load balancing using Parallel Randomized Greedy + Batched Refinement.
    """
    num_layers, num_groups = weight.shape
    device = weight.device
    groups_per_pack = num_groups // num_packs

    if groups_per_pack == 1:
        pack_index = torch.arange(num_groups, dtype=torch.int64,
                                  device=device).expand(num_layers, -1)
        rank_in_pack = torch.zeros_like(weight, dtype=torch.int64)
        return pack_index, rank_in_pack

    # 1. Expand Candidates
    # We will process [num_layers * num_candidates] independent problems
    B = num_layers * num_candidates
    
    # Replicate weights
    # [B, G]
    weights_flat = weight.repeat_interleave(num_candidates, dim=0)
    
    # 2. Generate Sort Keys (Randomization)
    # Strategy 1: LPT with Noise (75% of candidates)
    # Strategy 2: Random Permutation (25% of candidates)
    
    # Create noise
    noise = torch.rand_like(weights_flat)
    
    # For first 75%, noise is small multiplicative perturbation (LPT)
    split = int(B * 0.75)
    
    # Magnitude of noise: 1%
    lpt_keys = weights_flat[:split] * (1.0 + 0.01 * noise[:split])
    
    # For last 25%, keys are purely random (Random Greedy)
    random_keys = noise[split:]
    
    sort_keys = torch.cat([lpt_keys, random_keys], dim=0)
    
    # Sort items by keys (Descending)
    sorted_indices = torch.argsort(sort_keys, dim=1, descending=True)
    
    # Gather actual weights in sorted order
    sorted_weights = torch.gather(weights_flat, 1, sorted_indices)
    
    # 3. Vectorized Greedy Construction
    # We assign items one by one to the valid pack with min load
    
    pack_weights = torch.zeros(B, num_packs, device=device, dtype=weight.dtype)
    pack_counts = torch.zeros(B, num_packs, device=device, dtype=torch.int64)
    # [B, P, G]
    pack_assignments = torch.zeros(B, num_packs, groups_per_pack, device=device, dtype=torch.int64)
    
    batch_idx = torch.arange(B, device=device)
    inf_val = torch.tensor(float('inf'), device=device)
    
    for i in range(num_groups):
        # Current item for each batch
        w_item = sorted_weights[:, i]       # [B]
        id_item = sorted_indices[:, i]      # [B]
        
        # Mask full packs
        is_full = pack_counts >= groups_per_pack
        
        # Choose min load pack among valid ones
        # Add infinity to full packs
        masked_weights = pack_weights.clone()
        masked_weights[is_full] = inf_val
        
        best_pack = torch.argmin(masked_weights, dim=1) # [B]
        
        # Assign
        # Find slot index
        slots = pack_counts[batch_idx, best_pack]
        
        pack_assignments[batch_idx, best_pack, slots] = id_item
        pack_weights[batch_idx, best_pack] += w_item
        pack_counts[batch_idx, best_pack] += 1
        
    # 4. Batched Refinement
    # Run vectorized swap refinement on all candidates
    pack_assignments = _refine_packing_batched(
        weights_flat, pack_assignments, pack_weights,
        num_packs, groups_per_pack, max_iters=50
    )
    
    # 5. Selection
    # Calculate max load for each candidate
    # pack_weights is updated in place during refinement
    max_loads, _ = pack_weights.max(dim=1) # [B]
    
    # Reshape to [Layers, Candidates]
    max_loads = max_loads.view(num_layers, num_candidates)
    
    # Find best candidate index for each layer
    best_indices = torch.argmin(max_loads, dim=1) # [L]
    
    # Extract best assignments
    # We need to map [L] best_indices to [B] indices
    offsets = torch.arange(num_layers, device=device) * num_candidates
    final_indices = offsets + best_indices
    
    best_assignments = pack_assignments[final_indices] # [L, P, G]
    
    # 6. Reconstruct Outputs
    pack_index = torch.empty((num_layers, num_groups), device=device, dtype=torch.int64)
    rank_in_pack = torch.empty((num_layers, num_groups), device=device, dtype=torch.int64)
    
    # Flatten assignments: [L, P*G]
    flat_assignments = best_assignments.view(num_layers, -1)
    
    # Grids for scatter
    # [P*G]
    p_grid = torch.arange(num_packs, device=device).repeat_interleave(groups_per_pack)
    r_grid = torch.arange(groups_per_pack, device=device).repeat(num_packs)
    
    # Expand to [L, P*G]
    p_grid = p_grid.unsqueeze(0).expand(num_layers, -1)
    r_grid = r_grid.unsqueeze(0).expand(num_layers, -1)
    
    # Scatter
    # pack_index[l, item_id] = p_id
    pack_index.scatter_(1, flat_assignments, p_grid)
    rank_in_pack.scatter_(1, flat_assignments, r_grid)
    
    return pack_index, rank_in_pack


def replicate_experts(
        weight: torch.Tensor,
        num_phy: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Replicate experts to `num_phy` replicas, minimizing the maximum load.
    Uses a vectorized greedy approach.
    """
    n, num_log = weight.shape
    device = weight.device
    
    # Initialize: Each expert gets at least 1 replica
    logcnt = torch.ones((n, num_log), dtype=torch.int64, device=device)
    
    # Greedily assign remaining replicas
    # Vectorized loop over number of replicas to add
    for _ in range(num_log, num_phy):
        scores = weight / logcnt
        indices = torch.argmax(scores, dim=-1)
        logcnt[torch.arange(n, device=device), indices] += 1
        
    phy2log = torch.zeros((n, num_phy), dtype=torch.int64, device=device)
    rank = torch.zeros((n, num_phy), dtype=torch.int64, device=device)
    
    for i in range(n):
        counts = logcnt[i]
        l_ids = torch.repeat_interleave(
            torch.arange(num_log, device=device), counts
        )
        phy2log[i] = l_ids
        
        curr = 0
        for idx in range(num_log):
            c = counts[idx].item()
            rank[i, curr:curr+c] = torch.arange(c, device=device)
            curr += c
            
    return phy2log, rank, logcnt


def rebalance_experts_hierarchical(
    weight: torch.Tensor,
    num_physical_experts: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
):
    """
    Hierarchical rebalancing: Groups->Nodes, then Replicas->GPUs.
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
    tokens_per_mlog = weight.gather(-1, mlog2log).view(
        -1, num_logical_experts // num_nodes)
    phy2mlog, phyrank, mlogcnt = replicate_experts(
        tokens_per_mlog, num_physical_experts // num_nodes)

    # Step 3: pack physical_experts to GPUs
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
    """
    num_layers, num_logical_experts = weight.shape
    weight = weight.float().cpu()
    
    # Dispatch policy
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
    
    # Construct reverse mapping log2phy
    log2phy: torch.Tensor = torch.full(
        (num_layers, num_logical_experts, maxlogcnt),
        -1,
        dtype=torch.int64,
        device=logcnt.device,
    )
    
    # Efficient scatter
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