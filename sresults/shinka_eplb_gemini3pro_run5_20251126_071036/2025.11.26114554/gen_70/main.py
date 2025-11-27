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


def _refine_variance(weights: torch.Tensor,
                     pack_indices: torch.Tensor,
                     pack_weights: torch.Tensor,
                     num_packs: int,
                     groups_per_pack: int,
                     max_iters: int = 10) -> None:
    """
    Refines packing by minimizing load variance (L2 norm) via all-pairs swapping.
    Operates in-place on pack_indices and pack_weights.
    """
    B = weights.shape[0]
    device = weights.device
    N = num_packs * groups_per_pack
    
    # Pre-compute pack id for each slot [0,0,..,1,1,..]
    # [N]
    slot_to_pid = torch.arange(num_packs, device=device).repeat_interleave(groups_per_pack)
    
    # Mask for same-pack swaps
    same_pack = slot_to_pid.unsqueeze(0) == slot_to_pid.unsqueeze(1) # [N, N]
    
    batch_idx = torch.arange(B, device=device)
    
    for _ in range(max_iters):
        # Flatten structure
        flat_indices = pack_indices.view(B, N)
        
        # Gather weights: [B, N]
        w = weights.gather(1, flat_indices)
        
        # Gather pack loads for each item slot: [B, N]
        l = pack_weights.gather(1, slot_to_pid.expand(B, -1))
        
        # Compute D = w[u] - w[v] (weight moved from u to v)
        # We want to swap item at u with item at v
        w_diff = w.unsqueeze(2) - w.unsqueeze(1) # [B, N, N]
        l_diff = l.unsqueeze(1) - l.unsqueeze(2) # [B, N, N] (L(v) - L(u))
        
        # Change in sum of squares = 2 * d * (L_v - L_u + d)
        # We want to minimize this.
        delta_cost = 2 * w_diff * (l_diff + w_diff)
        
        # Apply mask (prevent same pack swaps)
        delta_cost.masked_fill_(same_pack, float('inf'))
        
        # Find minimum change per batch
        flat_cost = delta_cost.view(B, -1)
        min_val, flat_idx = torch.min(flat_cost, dim=1)
        
        # Threshold for improvement
        active = min_val < -1e-4
        if not active.any():
            break
            
        # Execute swaps for active batches
        active_batches = batch_idx[active]
        idx_pair = flat_idx[active]
        
        u = idx_pair // N
        v = idx_pair % N
        
        # Swap indices in the flattened view
        item_u = flat_indices[active_batches, u]
        item_v = flat_indices[active_batches, v]
        
        # Update flat indices (view of pack_indices)
        # We need to manually update pack_indices because basic indexing on view might not propagate 
        # if using fancy indexing on the LHS. However, here we update flat_indices which is a view.
        # To be safe, we compute 3D indices.
        p_u, g_u = u // groups_per_pack, u % groups_per_pack
        p_v, g_v = v // groups_per_pack, v % groups_per_pack
        
        pack_indices[active_batches, p_u, g_u] = item_v
        pack_indices[active_batches, p_v, g_v] = item_u
        
        # Update weights
        pid_u = slot_to_pid[u]
        pid_v = slot_to_pid[v]
        
        d_val = w_diff[active_batches, u, v]
        
        pack_weights[active_batches, pid_u] -= d_val
        pack_weights[active_batches, pid_v] += d_val


def _refine_minmax(weights: torch.Tensor,
                   pack_indices: torch.Tensor,
                   pack_weights: torch.Tensor,
                   max_iters: int = 20) -> None:
    """
    Refines packing by reducing Max Load (L_inf norm).
    Targeted swaps from Max Pack to others. In-place.
    """
    B, M, G = pack_indices.shape
    device = weights.device
    N = M * G
    batch_idx = torch.arange(B, device=device)
    
    # Precompute slot grid
    slot_to_pid = torch.arange(M, device=device).repeat_interleave(G)

    for _ in range(max_iters):
        # Identify max pack
        max_load, max_pid = torch.max(pack_weights, dim=1) # [B]
        
        # Gather items in max pack: [B, G]
        items_max = pack_indices[batch_idx, max_pid] 
        w_max = weights.gather(1, items_max) # [B, G]
        
        # Gather all items for comparison: [B, N]
        flat_indices = pack_indices.view(B, N)
        w_all = weights.gather(1, flat_indices)
        
        # Deltas: w_max[i] - w_all[j]
        # [B, G, N]
        deltas = w_max.unsqueeze(2) - w_all.unsqueeze(1)
        
        # Prospective Loads
        # New Max = Max - delta
        # New Other = Other + delta
        l_all = pack_weights.gather(1, slot_to_pid.expand(B, -1)) # [B, N]
        
        new_max = max_load.view(B, 1, 1) - deltas
        new_other = l_all.unsqueeze(1) + deltas
        
        # Objective: max(new_max, new_other)
        obj = torch.max(new_max, new_other)
        
        # Mask where j is in max_pid
        is_max = slot_to_pid.unsqueeze(0) == max_pid.unsqueeze(1) # [B, N]
        mask = is_max.unsqueeze(1).expand(-1, G, -1)
        obj.masked_fill_(mask, float('inf'))
        
        # Find best swap
        flat_obj = obj.view(B, -1)
        best_val, best_idx_flat = torch.min(flat_obj, dim=1)
        
        # Check improvement
        improve = best_val < (max_load - 1e-5)
        if not improve.any():
            break
            
        active = batch_idx[improve]
            
        # Decode indices
        idx_flat = best_idx_flat[improve]
        i = idx_flat // N # Index within Max Pack
        j = idx_flat % N  # Index within All Items
        
        pid_max_active = max_pid[active]
        pid_other_active = slot_to_pid[j]
        
        d_val = deltas[active, i, j]
        
        # Update weights
        pack_weights[active, pid_max_active] -= d_val
        pack_weights[active, pid_other_active] += d_val
        
        # Update indices
        # Calculate flat index of item i
        flat_offset_i = pid_max_active * G + i
        
        val_i = flat_indices[active, flat_offset_i]
        val_j = flat_indices[active, j]
        
        # Flattened view allows write-through
        flat_indices[active, flat_offset_i] = val_j
        flat_indices[active, j] = val_i


def balanced_packing(weight: torch.Tensor,
                     num_packs: int,
                     num_candidates: int = 64,
                     num_rounds: int = 2) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pack n weighted objects to m packs using Multi-Round Parallel Randomized Greedy + Refinement.
    Uses iterative re-weighting to escape local minima.
    """
    num_layers, num_groups = weight.shape
    device = weight.device
    assert num_groups % num_packs == 0
    groups_per_pack = num_groups // num_packs

    if groups_per_pack == 1:
        pack_index = torch.arange(num_groups, dtype=torch.int64,
                                  device=device).expand(num_layers, -1)
        rank_in_pack = torch.zeros_like(weight, dtype=torch.int64)
        return pack_index, rank_in_pack

    # State tracking
    best_assignment = torch.zeros((num_layers, num_packs, groups_per_pack), dtype=torch.int64, device=device)
    best_max_loads = torch.full((num_layers,), float('inf'), device=device)
    
    # Virtual weights for re-weighting (starts as real weights)
    virtual_weight = weight.clone().float()
    
    # Batch size B = L * K
    B = num_layers * num_candidates
    batch_idx = torch.arange(B, device=device)
    inf_val = torch.tensor(float('inf'), device=device)
    
    for round_idx in range(num_rounds):
        # 1. Prepare Inputs
        # Expand real weights: [B, N]
        w_real = weight.unsqueeze(1).expand(-1, num_candidates, -1).reshape(B, num_groups)
        
        # Expand virtual weights: [B, N]
        w_virt = virtual_weight.unsqueeze(1).expand(-1, num_candidates, -1).reshape(B, num_groups)
        
        # 2. Add Noise to Virtual Weights (Randomized Priorities)
        noise = torch.rand_like(w_virt) * 0.02 # 2% noise
        # Keep the first candidate pure (greedy baseline)
        noise.view(num_layers, num_candidates, num_groups)[:, 0, :] = 0.0
        
        sort_keys = w_virt * (1.0 + noise)
        
        # 3. Sort based on perturbed virtual weights
        sorted_indices = torch.argsort(sort_keys, dim=1, descending=True)
        # Gather REAL weights for accurate load tracking
        sorted_weights = torch.gather(w_real, 1, sorted_indices)
        
        # 4. Vectorized Greedy Packing
        pack_weights = torch.zeros(B, num_packs, device=device, dtype=weight.dtype)
        pack_counts = torch.zeros(B, num_packs, device=device, dtype=torch.int64)
        pack_indices = torch.zeros(B, num_packs, groups_per_pack, device=device, dtype=torch.int64)
        
        for i in range(num_groups):
            w = sorted_weights[:, i]
            idx = sorted_indices[:, i]
            
            is_full = pack_counts >= groups_per_pack
            cand_weights = torch.where(is_full, inf_val, pack_weights)
            best_pack = torch.argmin(cand_weights, dim=1)
            
            slots = pack_counts[batch_idx, best_pack]
            pack_indices[batch_idx, best_pack, slots] = idx
            pack_weights[batch_idx, best_pack] += w
            pack_counts[batch_idx, best_pack] += 1
            
        # 5. Hybrid Refinement (using REAL weights)
        # A. Global Variance Reduction
        _refine_variance(w_real, pack_indices, pack_weights, num_packs, groups_per_pack, max_iters=5)
        # B. Targeted Min-Max Reduction
        _refine_minmax(w_real, pack_indices, pack_weights, max_iters=10)
        
        # 6. Select Best Candidate Per Layer
        pw_reshaped = pack_weights.view(num_layers, num_candidates, num_packs)
        max_loads = torch.max(pw_reshaped, dim=2).values # [L, K]
        
        round_best_vals, round_best_k = torch.min(max_loads, dim=1) # [L]
        
        improved = round_best_vals < best_max_loads
        if improved.any():
            best_max_loads = torch.where(improved, round_best_vals, best_max_loads)
            
            pi_reshaped = pack_indices.view(num_layers, num_candidates, num_packs, groups_per_pack)
            round_best_assignments = pi_reshaped[torch.arange(num_layers, device=device), round_best_k]
            
            mask_exp = improved.view(num_layers, 1, 1).expand(-1, num_packs, groups_per_pack)
            best_assignment = torch.where(mask_exp, round_best_assignments, best_assignment)
            
        # 7. Re-weighting Strategy
        if round_idx < num_rounds - 1:
            # Identify the heaviest pack in the CURRENT BEST assignment
            # Calculate loads for best assignment
            best_flat = best_assignment.view(num_layers, -1)
            w_best = weight.gather(1, best_flat).view(num_layers, num_packs, groups_per_pack)
            w_best_sums = w_best.sum(dim=2) # [L, M]
            _, max_pids = torch.max(w_best_sums, dim=1) # [L]
            
            # Get items in max pack: [L, G]
            max_items = best_assignment[torch.arange(num_layers, device=device), max_pids]
            
            # Boost virtual weights for these bottleneck items
            # This makes them "heavier" in the priority sort, forcing them to be handled earlier
            mult_tensor = torch.ones_like(virtual_weight)
            mult_tensor.scatter_(1, max_items, 1.05) # 5% boost
            virtual_weight *= mult_tensor

    # Final Map Construction
    pack_index = torch.zeros_like(weight, dtype=torch.int64)
    rank_in_pack = torch.zeros_like(weight, dtype=torch.int64)
    
    flat_assignment = best_assignment.view(num_layers, -1)
    
    p_ids = torch.arange(num_packs, device=device).unsqueeze(1).expand(-1, groups_per_pack).reshape(1, -1).expand(num_layers, -1)
    r_ids = torch.arange(groups_per_pack, device=device).unsqueeze(0).expand(num_packs, -1).reshape(1, -1).expand(num_layers, -1)
    
    pack_index.scatter_(1, flat_assignment, p_ids)
    rank_in_pack.scatter_(1, flat_assignment, r_ids)

    return pack_index, rank_in_pack


def replicate_experts(
        weight: torch.Tensor,
        num_phy: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Replicate experts to `num_phy` replicas, minimizing the maximum load.
    Uses a vectorized greedy approach (Jefferson/D'Hondt method).
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

    # Reconstruction of the mapping tables from counts
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