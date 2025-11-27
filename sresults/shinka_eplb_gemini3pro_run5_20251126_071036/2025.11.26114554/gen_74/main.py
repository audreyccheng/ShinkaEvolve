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
    Batched refinement using pairwise variance minimization.
    Minimizes sum of squared loads by swapping items between any two packs.
    """
    B, M, G = pack_indices.shape
    device = weights.device
    N = M * G

    # Pre-compute pack identifiers for every item slot in the flattened view
    # [0, 0, ..., 1, 1, ...]
    item_to_pack_id = torch.arange(num_packs, device=device).unsqueeze(1).expand(-1, groups_per_pack).reshape(-1)

    # Mask to prevent swapping within same pack [N, N]
    same_pack_mask = item_to_pack_id.unsqueeze(0) == item_to_pack_id.unsqueeze(1)

    batch_arange = torch.arange(B, device=device)

    # Flatten pack_indices for easy access: [B, N]
    flat_pack_indices = pack_indices.view(B, -1)

    for _ in range(max_iters):
        # Get weights: w[b, slot] = weights[b, flat_pack_indices[b, slot]]
        w_current = weights.gather(1, flat_pack_indices) # [B, N]

        # Load of the pack each item currently belongs to
        # l_current[b, slot] = pack_weights[b, item_to_pack_id[slot]]
        l_current = pack_weights[:, item_to_pack_id] # [B, N]

        # Delta matrix D[b, i, j] = w[b, i] - w[b, j] (weight moved from i's pack to j's pack)
        # We consider swapping item at slot i with item at slot j
        D = w_current.unsqueeze(2) - w_current.unsqueeze(1) # [B, N, N]

        # Load Diff matrix L_diff[b, i, j] = L[b, pack(j)] - L[b, pack(i)]
        L_diff = l_current.unsqueeze(1) - l_current.unsqueeze(2) # [B, N, N]

        # Change in variance (sum of squares)
        # Change = 2 * D * (L_diff + D)
        change = 2 * D * (L_diff + D)

        # Apply mask
        change.masked_fill_(same_pack_mask, float('inf'))

        # Find best swap per batch
        min_val, flat_idx = torch.min(change.view(B, -1), dim=1) # [B]

        # Check convergence
        update_mask = min_val < -1e-6
        if not update_mask.any():
            break

        # Indices to update
        active_batches = batch_arange[update_mask]
        active_indices = flat_idx[update_mask]

        idx_i = active_indices // N
        idx_j = active_indices % N

        pid_i = item_to_pack_id[idx_i]
        pid_j = item_to_pack_id[idx_j]

        # Execute swap in flat_pack_indices
        item_i = flat_pack_indices[active_batches, idx_i]
        item_j = flat_pack_indices[active_batches, idx_j]

        flat_pack_indices[active_batches, idx_i] = item_j
        flat_pack_indices[active_batches, idx_j] = item_i

        # Update weights
        delta = D[active_batches, idx_i, idx_j]
        pack_weights[active_batches, pid_i] -= delta
        pack_weights[active_batches, pid_j] += delta

    return pack_indices


def balanced_packing(weight: torch.Tensor,
                     num_packs: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pack n weighted objects to m packs using Iterative Re-weighting with Parallel Greedy + Refinement.
    """
    num_layers, num_groups = weight.shape
    device = weight.device
    assert num_groups % num_packs == 0
    groups_per_pack = num_groups // num_packs

    # Trivial case
    if groups_per_pack == 1:
        pack_index = torch.arange(num_groups, dtype=torch.int64,
                                  device=device).expand(num_layers, -1)
        rank_in_pack = torch.zeros_like(weight, dtype=torch.int64)
        return pack_index, rank_in_pack

    # Config
    num_iters = 3
    num_candidates = 64
    B = num_layers * num_candidates
    
    # State tracking
    best_pack_index = torch.zeros_like(weight, dtype=torch.int64)
    best_rank_in_pack = torch.zeros_like(weight, dtype=torch.int64)
    best_max_loads = torch.full((num_layers,), float('inf'), device=device)
    
    # Virtual weights for LPT ordering
    virtual_weight = weight.clone().float()
    
    # Pre-allocate for batch operations
    batch_idx = torch.arange(B, device=device)
    inf_tensor = torch.tensor(float('inf'), device=device)
    
    # Grid for scatter reconstruction
    grid_p = torch.arange(num_packs, device=device).view(1, -1, 1).expand(num_layers, -1, groups_per_pack)
    grid_r = torch.arange(groups_per_pack, device=device).view(1, 1, -1).expand(num_layers, num_packs, -1)
    
    for i in range(num_iters):
        # 1. Prepare Batch
        # Expand weights: [L, K, G] -> [B, G]
        real_weights_expanded = weight.unsqueeze(1).expand(-1, num_candidates, -1).reshape(B, num_groups)
        virtual_weights_expanded = virtual_weight.unsqueeze(1).expand(-1, num_candidates, -1).reshape(B, num_groups)
        
        # Add Noise to virtual weights
        # Decay noise over iterations
        noise_scale = 0.05 * (1.0 - i / num_iters)
        noise = torch.rand_like(virtual_weights_expanded) * noise_scale
        # Keep first candidate pure (no noise) for stability
        # Reshape to [L, K, G] to mask first candidate
        noise.view(num_layers, num_candidates, num_groups)[:, 0, :] = 0.0
        
        sort_keys = virtual_weights_expanded * (1.0 + noise)
        sorted_indices = torch.argsort(sort_keys, dim=1, descending=True) # [B, G]
        
        # Gather real weights in sorted order
        sorted_real_weights = torch.gather(real_weights_expanded, 1, sorted_indices)
        
        # 2. Greedy LPT
        pack_weights = torch.zeros((B, num_packs), dtype=weight.dtype, device=device)
        pack_counts = torch.zeros((B, num_packs), dtype=torch.int64, device=device)
        pack_indices = torch.zeros((B, num_packs, groups_per_pack), dtype=torch.int64, device=device)
        
        for j in range(num_groups):
            w_vals = sorted_real_weights[:, j]
            item_ids = sorted_indices[:, j]
            
            # Mask full packs
            is_full = pack_counts >= groups_per_pack
            cand_weights = torch.where(is_full, inf_tensor, pack_weights)
            
            # Select min
            best_pack = torch.argmin(cand_weights, dim=1)
            
            # Assign
            slots = pack_counts[batch_idx, best_pack]
            pack_indices[batch_idx, best_pack, slots] = item_ids
            pack_weights[batch_idx, best_pack] += w_vals
            pack_counts[batch_idx, best_pack] += 1
            
        # 3. Refine
        pack_indices = _refine_packing_batched(
            real_weights_expanded,
            pack_indices,
            pack_weights,
            num_packs,
            groups_per_pack,
            max_iters=20 if i < num_iters - 1 else 50
        )
        
        # 4. Evaluate & Update Best
        max_loads, max_pids_batch = torch.max(pack_weights, dim=1) # [B]
        
        # Reshape to [L, K]
        max_loads_reshaped = max_loads.view(num_layers, num_candidates)
        
        # Find best candidate for each layer
        iter_best_vals, iter_best_k = torch.min(max_loads_reshaped, dim=1) # [L]
        
        # Compare with global best
        improved_mask = iter_best_vals < best_max_loads
        
        if improved_mask.any():
            best_max_loads[improved_mask] = iter_best_vals[improved_mask]
            
            # Extract assignments
            # Indices in batch: (layer_idx * K) + best_k
            layer_indices = torch.arange(num_layers, device=device)
            best_batch_indices = layer_indices * num_candidates + iter_best_k
            
            # Get best pack_indices [L, M, G]
            best_assignments_iter = pack_indices[best_batch_indices]
            
            # Scatter to output format
            # Flat assignment [L, M*G]
            flat_ass = best_assignments_iter.view(num_layers, -1)
            
            # Create full temp outputs
            temp_p_idx = torch.empty_like(best_pack_index)
            temp_r_idx = torch.empty_like(best_rank_in_pack)
            temp_p_idx.scatter_(1, flat_ass, grid_p.view(num_layers, -1))
            temp_r_idx.scatter_(1, flat_ass, grid_r.view(num_layers, -1))
            
            best_pack_index = torch.where(improved_mask.unsqueeze(1), temp_p_idx, best_pack_index)
            best_rank_in_pack = torch.where(improved_mask.unsqueeze(1), temp_r_idx, best_rank_in_pack)
            
        # 5. Feedback Loop (Re-weighting)
        if i < num_iters - 1:
            # We want to boost items in the heaviest pack of the *current iteration's best*
            # Even if it didn't beat global best, it's the gradient we follow.
            
            # Get max_pids for the best candidate of this iteration
            layer_indices = torch.arange(num_layers, device=device)
            best_batch_indices = layer_indices * num_candidates + iter_best_k
            best_max_pids = max_pids_batch[best_batch_indices] # [L]
            
            # Get items in those packs: [L, G]
            # pack_indices: [B, M, G]
            best_assignments_iter = pack_indices[best_batch_indices] # [L, M, G]
            
            # Gather items from max pack
            # [L, 1, G]
            gather_idx = best_max_pids.view(num_layers, 1, 1).expand(-1, 1, groups_per_pack)
            max_items = torch.gather(best_assignments_iter, 1, gather_idx).squeeze(1) # [L, G]
            
            # Boost weights
            mult = torch.ones_like(virtual_weight)
            # Scatter 1.1
            src = torch.full_like(max_items, 1.1, dtype=virtual_weight.dtype)
            mult.scatter_(1, max_items, src)
            
            virtual_weight = virtual_weight * mult

    return best_pack_index, best_rank_in_pack


def replicate_experts(
        weight: torch.Tensor,
        num_phy: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Replicate experts to `num_phy` replicas, minimizing the maximum load.
    
    Uses a vectorized greedy approach (Jefferson/D'Hondt method) which is
    optimal for the min-max load objective with discrete allocations.
    """
    n, num_log = weight.shape
    device = weight.device
    
    # Initialize: Each expert gets at least 1 replica
    logcnt = torch.ones((n, num_log), dtype=torch.int64, device=device)
    
    # Greedily assign remaining replicas
    for _ in range(num_log, num_phy):
        # Score is current load per replica
        scores = weight / logcnt
        # Find expert with max score in each layer
        indices = torch.argmax(scores, dim=-1)
        # Add replica
        rows = torch.arange(n, device=device)
        logcnt[rows, indices] += 1

    # Reconstruction of the mapping tables from counts
    phy2log = torch.zeros((n, num_phy), dtype=torch.int64, device=device)
    rank = torch.zeros((n, num_phy), dtype=torch.int64, device=device)

    for i in range(n):
        counts = logcnt[i]
        
        # Create logical IDs
        l_ids = torch.repeat_interleave(
            torch.arange(num_log, device=device), counts
        )
        phy2log[i] = l_ids
        
        # Create ranks
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
    # Sum weights within each group
    tokens_per_group = weight.unflatten(-1, (num_groups, group_size)).sum(-1)
    
    # Use improved balanced packing
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
    
    # Use improved balanced packing
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