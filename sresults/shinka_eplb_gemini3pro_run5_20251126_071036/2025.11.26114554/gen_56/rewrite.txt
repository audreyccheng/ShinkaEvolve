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


def _refine_packing_1swap(weights: torch.Tensor,
                          pack_indices: torch.Tensor,
                          pack_weights: torch.Tensor,
                          num_packs: int,
                          groups_per_pack: int,
                          max_iters: int = 50) -> torch.Tensor:
    """
    Batched all-pairs 1-item swap refinement.
    Minimizes load variance by checking all possible single swaps between packs.
    
    Args:
        weights: [B, num_groups] Weights of items.
        pack_indices: [B, num_packs, groups_per_pack] Current assignment of items.
        pack_weights: [B, num_packs] Current weights of packs.
        
    Returns:
        pack_indices: Refined assignments.
    """
    B = weights.shape[0]
    device = weights.device
    
    # Pre-compute flattened pack IDs: [MG] -> [0, 0, ..., 1, 1, ...]
    flat_pack_ids = torch.arange(num_packs, device=device).repeat_interleave(groups_per_pack)
    
    # Mask to prevent swapping items within the same pack
    # [MG, MG]
    same_pack_mask = flat_pack_ids.unsqueeze(0) == flat_pack_ids.unsqueeze(1)
    
    batch_idx = torch.arange(B, device=device)
    num_items = num_packs * groups_per_pack
    
    for _ in range(max_iters):
        # Flatten pack_indices to [B, MG] for easier indexing
        flat_indices = pack_indices.view(B, -1)
        
        # Gather item weights: [B, MG]
        w = weights.gather(1, flat_indices)
        
        # Gather pack loads for each item slot: [B, MG]
        # pack_weights is [B, M], we index with [MG]
        l = pack_weights[:, flat_pack_ids]
        
        # Compute Delta matrix (Weight Diff): w_i - w_j
        # [B, MG, MG]
        diff_w = w.unsqueeze(2) - w.unsqueeze(1)
        
        # Compute Load Diff: Load(pack(j)) - Load(pack(i))
        # [B, MG, MG]
        diff_l = l.unsqueeze(1) - l.unsqueeze(2)
        
        # Change in variance = 2 * diff_w * (diff_l + diff_w)
        # We want to find negative changes (variance reduction)
        change = diff_w * (diff_l + diff_w)
        
        # Apply mask to ignore same-pack swaps
        change.masked_fill_(same_pack_mask, float('inf'))
        
        # Find best swap for each batch
        # [B, MG*MG] -> [B]
        min_val, flat_arg = torch.min(change.view(B, -1), dim=1)
        
        # Convergence check: stop if improvement is negligible
        active = min_val < -1e-5
        if not active.any():
            break
            
        # Select active batches
        active_b = batch_idx[active]
        best_idx = flat_arg[active]
        
        # Decode indices
        idx_i = best_idx // num_items
        idx_j = best_idx % num_items
        
        # Execute swap on flat view (propagates to pack_indices)
        val_i = flat_indices[active_b, idx_i]
        val_j = flat_indices[active_b, idx_j]
        
        flat_indices[active_b, idx_i] = val_j
        flat_indices[active_b, idx_j] = val_i
        
        # Update pack weights
        pid_i = flat_pack_ids[idx_i]
        pid_j = flat_pack_ids[idx_j]
        
        dw = diff_w[active_b, idx_i, idx_j]
        
        pack_weights[active_b, pid_i] -= dw
        pack_weights[active_b, pid_j] += dw
        
    return pack_indices


def _refine_packing_2swap(weights: torch.Tensor,
                          pack_indices: torch.Tensor,
                          pack_weights: torch.Tensor,
                          num_packs: int,
                          groups_per_pack: int,
                          max_iters: int = 10) -> torch.Tensor:
    """
    Batched 2-item swap refinement between the Max-Load pack and Min-Load pack.
    Attempts to swap 2 items from Max with 2 items from Min to reduce the Gap.
    Only runs if groups_per_pack is small (<= 32) to limit memory usage.
    """
    if groups_per_pack > 32 or groups_per_pack < 2:
        return pack_indices
        
    B = weights.shape[0]
    device = weights.device
    batch_idx = torch.arange(B, device=device)
    
    # Mask for diagonal pairs (i == j) within a pack
    # [G, G]
    diag_mask = torch.eye(groups_per_pack, device=device, dtype=torch.bool)
    
    for _ in range(max_iters):
        # Identify Max and Min packs
        max_load, max_pid = torch.max(pack_weights, dim=1) # [B]
        min_load, min_pid = torch.min(pack_weights, dim=1) # [B]
        
        gap = max_load - min_load
        
        # Gather items from Max and Min packs
        # pack_indices: [B, M, G]
        items_max = pack_indices[batch_idx, max_pid, :] # [B, G]
        items_min = pack_indices[batch_idx, min_pid, :] # [B, G]
        
        # Gather weights
        w_max = weights.gather(1, items_max) # [B, G]
        w_min = weights.gather(1, items_min) # [B, G]
        
        # Compute Pair Sums
        # [B, G, G]
        sum_max = w_max.unsqueeze(2) + w_max.unsqueeze(1)
        sum_min = w_min.unsqueeze(2) + w_min.unsqueeze(1)
        
        # Calculate Delta: sum_max[i,j] - sum_min[u,v]
        # [B, G, G, G, G]
        delta = sum_max.unsqueeze(3).unsqueeze(4) - sum_min.unsqueeze(1).unsqueeze(2)
        
        # Metric: New Gap = |gap - 2*delta|
        # We want to minimize this new gap.
        new_gaps = torch.abs(gap.view(B, 1, 1, 1, 1) - 2 * delta)
        
        # Mask invalid pairs (diagonal elements where i==j or u==v)
        mask_max = diag_mask.view(1, groups_per_pack, groups_per_pack, 1, 1)
        mask_min = diag_mask.view(1, 1, 1, groups_per_pack, groups_per_pack)
        
        new_gaps.masked_fill_(mask_max, float('inf'))
        new_gaps.masked_fill_(mask_min, float('inf'))
        
        # Find minimum gap
        min_gap_flat, idx_flat = torch.min(new_gaps.view(B, -1), dim=1)
        
        # Check if we improve the gap
        improved = min_gap_flat < (gap - 1e-5)
        if not improved.any():
            break
            
        active = batch_idx[improved]
        idx = idx_flat[improved]
        
        # Decode indices
        G = groups_per_pack
        G2 = G * G
        G3 = G * G * G
        
        # idx breakdown from flattened [G, G, G, G]
        i = idx // G3
        rem = idx % G3
        j = rem // G2
        rem = rem % G2
        u = rem // G
        v = rem % G
        
        # Get global pack IDs for updates
        p_max = max_pid[active]
        p_min = min_pid[active]
        
        # Items to swap
        val_max_i = pack_indices[active, p_max, i]
        val_max_j = pack_indices[active, p_max, j]
        val_min_u = pack_indices[active, p_min, u]
        val_min_v = pack_indices[active, p_min, v]
        
        # Execute Swap
        pack_indices[active, p_max, i] = val_min_u
        pack_indices[active, p_max, j] = val_min_v
        pack_indices[active, p_min, u] = val_max_i
        pack_indices[active, p_min, v] = val_max_j
        
        # Update weights
        d_val = delta[active, i, j, u, v]
        pack_weights[active, p_max] -= d_val
        pack_weights[active, p_min] += d_val
        
    return pack_indices


def balanced_packing(weight: torch.Tensor,
                     num_packs: int,
                     num_restarts: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pack n weighted objects to m packs.
    Uses Randomized Greedy LPT initialization followed by 
    1-Swap (Global) and 2-Swap (Max-Min) refinements.

    Parameters:
        weight: [X, n], the weight of each item
        num_packs: number of packs
        num_restarts: number of parallel candidates to evaluate (batch size)

    Returns:
        pack_index: [X, n]
        rank_in_pack: [X, n]
    """
    num_layers, num_groups = weight.shape
    device = weight.device
    assert num_groups % num_packs == 0
    groups_per_pack = num_groups // num_packs

    # Optimization for trivial case
    if groups_per_pack == 1:
        pack_index = torch.arange(num_groups, dtype=torch.int64,
                                  device=device).expand(num_layers, -1)
        rank_in_pack = torch.zeros_like(weight, dtype=torch.int64)
        return pack_index, rank_in_pack

    # Expand weights for candidates: [L, K, G]
    B = num_layers * num_restarts
    
    # 1. Randomized Greedy Initialization
    # Base candidates
    w_expanded = weight.unsqueeze(1).expand(-1, num_restarts, -1).reshape(B, num_groups)
    
    # Add noise to sort keys to explore different greedy orders
    # Candidate 0 in each layer group is kept pure (no noise) via noise masking
    noise = torch.rand_like(w_expanded) * 0.05 # 5% noise
    # Set first candidate of each layer to 0 noise (Pure LPT)
    # Reshape to [L, K, G] to zero out K=0
    noise.view(num_layers, num_restarts, num_groups)[:, 0, :] = 0.0
    
    sort_keys = w_expanded * (1.0 + noise)
    
    # Sort descending
    sorted_indices = torch.argsort(sort_keys, dim=-1, descending=True)
    
    # Gather actual weights in sorted order
    sorted_weights = torch.gather(w_expanded, 1, sorted_indices)
    
    # Perform Greedy Packing
    pack_weights = torch.zeros((B, num_packs), dtype=weight.dtype, device=device)
    pack_counts = torch.zeros((B, num_packs), dtype=torch.int64, device=device)
    pack_assignment = torch.zeros((B, num_packs, groups_per_pack), dtype=torch.int64, device=device)
    
    batch_idx = torch.arange(B, device=device)
    inf_val = float('inf')
    
    for j in range(num_groups):
        w = sorted_weights[:, j]
        item_idx = sorted_indices[:, j]
        
        # Mask full packs
        is_full = (pack_counts >= groups_per_pack)
        masked_weights = torch.where(is_full, inf_val, pack_weights)
        
        # Choose pack with min weight
        best_pack = torch.argmin(masked_weights, dim=1)
        
        # Assign
        slots = pack_counts[batch_idx, best_pack]
        pack_assignment[batch_idx, best_pack, slots] = item_idx
        pack_weights[batch_idx, best_pack] += w
        pack_counts[batch_idx, best_pack] += 1
        
    # 2. Refinement Stage 1: Global 1-Swap (Variance Descent)
    pack_assignment = _refine_packing_1swap(
        w_expanded, pack_assignment, pack_weights, 
        num_packs, groups_per_pack, max_iters=50
    )
    
    # 3. Refinement Stage 2: Targeted 2-Swap (Max-Min Reduction)
    pack_assignment = _refine_packing_2swap(
        w_expanded, pack_assignment, pack_weights,
        num_packs, groups_per_pack, max_iters=10
    )
    
    # 4. Selection
    # Calculate Max Load for every candidate
    max_loads, _ = torch.max(pack_weights, dim=1)
    max_loads = max_loads.view(num_layers, num_restarts)
    
    # Find best candidate index for each layer
    best_k = torch.argmin(max_loads, dim=1) # [L]
    
    # Select best assignment
    # Global batch indices
    offsets = torch.arange(num_layers, device=device) * num_restarts
    best_indices = offsets + best_k
    
    best_assignment = pack_assignment[best_indices] # [L, M, G]
    
    # 5. Output Construction
    pack_index = torch.empty((num_layers, num_groups), dtype=torch.int64, device=device)
    rank_in_pack = torch.empty((num_layers, num_groups), dtype=torch.int64, device=device)
    
    flat_assignment = best_assignment.view(num_layers, -1)
    
    # Grids for scattering
    # pack_ids: [0,0,..,1,1,..]
    p_ids = torch.arange(num_packs, device=device).unsqueeze(1).expand(-1, groups_per_pack).reshape(1, -1).expand(num_layers, -1)
    # rank_ids: [0,1,..,0,1,..]
    r_ids = torch.arange(groups_per_pack, device=device).unsqueeze(0).expand(num_packs, -1).reshape(1, -1).expand(num_layers, -1)
    
    # Scatter using the item indices in flat_assignment as positions
    pack_index.scatter_(1, flat_assignment, p_ids)
    rank_in_pack.scatter_(1, flat_assignment, r_ids)
    
    return pack_index, rank_in_pack


def replicate_experts(
        weight: torch.Tensor,
        num_phy: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Replicate experts to `num_phy` replicas, minimizing the maximum load.
    Vectorized greedy approach.
    """
    n, num_log = weight.shape
    device = weight.device

    # Initialize: Each expert gets at least 1 replica
    logcnt = torch.ones((n, num_log), dtype=torch.int64, device=device)
    
    # Greedily assign remaining replicas
    for _ in range(num_log, num_phy):
        scores = weight / logcnt
        indices = torch.argmax(scores, dim=-1)
        rows = torch.arange(n, device=device)
        logcnt[rows, indices] += 1

    phy2log = torch.zeros((n, num_phy), dtype=torch.int64, device=device)
    rank = torch.zeros((n, num_phy), dtype=torch.int64, device=device)

    # Reconstruct mappings
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
        -1, pphy2phy) 
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
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, num_groups, num_nodes, num_gpus)
    else:
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