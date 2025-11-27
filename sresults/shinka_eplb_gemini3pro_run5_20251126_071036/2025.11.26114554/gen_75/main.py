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


def _refine_packing_minmax(weights: torch.Tensor,
                           pack_indices: torch.Tensor,
                           pack_weights: torch.Tensor,
                           num_packs: int,
                           groups_per_pack: int,
                           max_iters: int = 20) -> torch.Tensor:
    """
    Batched vectorized refinement minimizing max load by swapping items
    between the heaviest pack and any other pack.
    
    Args:
        weights: [B, num_groups] Real weights
        pack_indices: [B, num_packs, groups_per_pack]
        pack_weights: [B, num_packs]
    """
    B = weights.shape[0]
    device = weights.device
    
    for _ in range(max_iters):
        # 1. Identify max load pack
        max_vals, max_pids = torch.max(pack_weights, dim=1) # [B]
        
        # 2. Gather items in max packs
        # max_pids: [B] -> [B, 1, 1] -> [B, 1, G]
        gather_indices = max_pids.view(B, 1, 1).expand(B, 1, groups_per_pack)
        # pack_indices: [B, M, G]
        items_max = torch.gather(pack_indices, 1, gather_indices).squeeze(1) # [B, G]
        w_max = torch.gather(weights, 1, items_max) # [B, G]
        
        # 3. Gather weights of ALL items
        # flatten pack_indices: [B, M*G]
        flat_pack_indices = pack_indices.view(B, -1)
        w_all = torch.gather(weights, 1, flat_pack_indices).view(B, num_packs, groups_per_pack)
        
        # 4. Compute Deltas: swap i (max) with j (other)
        # delta = w_max[i] - w_all[k, j]
        # [B, G, 1, 1] - [B, 1, M, G] -> [B, G, M, G]
        deltas = w_max.unsqueeze(2).unsqueeze(3) - w_all.unsqueeze(1)
        
        # 5. New Loads
        # New Max = Max - delta
        # New Other = Load[k] + delta
        new_max_load = max_vals.view(B, 1, 1, 1) - deltas
        new_other_load = pack_weights.view(B, 1, num_packs, 1) + deltas
        
        # Objective: Max of the two changed packs
        # We assume other packs don't exceed these two (heuristic for speed)
        objs = torch.max(new_max_load, new_other_load)
        
        # 6. Mask invalid swaps (same pack)
        # k == max_pid
        mask_k = (torch.arange(num_packs, device=device).unsqueeze(0) == max_pids.unsqueeze(1)) # [B, M]
        mask = mask_k.view(B, 1, num_packs, 1).expand(-1, groups_per_pack, -1, groups_per_pack)
        objs.masked_fill_(mask, float('inf'))
        
        # 7. Find Best Swap
        flat_objs = objs.view(B, -1)
        min_obj, best_idx_flat = torch.min(flat_objs, dim=1)
        
        # 8. Update Check
        improve = min_obj < (max_vals - 1e-5)
        if not improve.any():
            break
            
        active = improve
        batch_active = torch.where(active)[0]
        
        if len(batch_active) == 0:
            break
            
        # Decode indices
        # idx into [G, M, G]
        best_flat_act = best_idx_flat[active]
        M = num_packs
        G = groups_per_pack
        
        # i (max_item), k (other_pack), j (other_item)
        i_idx = best_flat_act // (M * G)
        rem = best_flat_act % (M * G)
        k_idx = rem // G
        j_idx = rem % G
        
        p_max = max_pids[active]
        p_other = k_idx
        
        # Swap Items
        val_max = pack_indices[active, p_max, i_idx]
        val_other = pack_indices[active, p_other, j_idx]
        
        pack_indices[active, p_max, i_idx] = val_other
        pack_indices[active, p_other, j_idx] = val_max
        
        # Update Loads
        d_val = deltas.view(B, -1)[batch_active, best_flat_act]
        
        pack_weights[batch_active, p_max] -= d_val
        pack_weights[batch_active, p_other] += d_val
        
    return pack_indices


def balanced_packing(weight: torch.Tensor,
                     num_packs: int,
                     num_candidates: int = 64,
                     num_reweight_iters: int = 5) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pack n weighted objects to m packs using Parallel Randomized Greedy LPT 
    with Iterative Re-weighting and Min-Max Refinement.
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

    # 1. Setup Parallel Batches
    # B = Total number of independent packing trials
    B = num_layers * num_candidates
    
    # Expand real weights: [L, 1, G] -> [L, K, G] -> [B, G]
    real_weights_expanded = weight.unsqueeze(1).expand(-1, num_candidates, -1).reshape(B, num_groups)
    
    # Initialize Virtual Weights with slight noise for tie-breaking
    # [B, G]
    noise = torch.rand_like(real_weights_expanded) * 1e-4
    virtual_weights = real_weights_expanded + noise
    
    # Track best solution found so far across iterations
    best_max_loads = torch.full((B,), float('inf'), device=device)
    best_pack_indices = torch.zeros((B, num_packs, groups_per_pack), dtype=torch.int64, device=device)
    
    # Pre-allocate buffers for greedy loop
    batch_idx = torch.arange(B, device=device)
    inf_tensor = torch.tensor(float('inf'), device=device)
    
    # 2. Iterative Re-weighting Loop
    for _ in range(num_reweight_iters):
        # A. Greedy Packing based on Virtual Weights
        # Sort based on current virtual (re-weighted) values
        sorted_indices = torch.argsort(virtual_weights, dim=1, descending=True) # [B, G]
        
        pack_weights = torch.zeros((B, num_packs), dtype=weight.dtype, device=device)
        pack_counts = torch.zeros((B, num_packs), dtype=torch.int64, device=device)
        pack_indices = torch.zeros((B, num_packs, groups_per_pack), dtype=torch.int64, device=device)
        
        # Gather real weights in sorted order: [B, G]
        w_sorted = torch.gather(real_weights_expanded, 1, sorted_indices)
        
        # Vectorized Packing Loop
        for j in range(num_groups):
            w = w_sorted[:, j]
            idx = sorted_indices[:, j]
            
            # Find min load among valid packs
            is_full = (pack_counts >= groups_per_pack)
            cand_w = torch.where(is_full, inf_tensor, pack_weights)
            
            best_p = torch.argmin(cand_w, dim=1)
            
            slots = pack_counts[batch_idx, best_p]
            pack_indices[batch_idx, best_p, slots] = idx
            pack_weights[batch_idx, best_p] += w
            pack_counts[batch_idx, best_p] += 1
            
        # B. Refinement (using Real Weights)
        # Local search to fix sub-optimal greedy choices
        pack_indices = _refine_packing_minmax(
            real_weights_expanded, pack_indices, pack_weights,
            num_packs, groups_per_pack, max_iters=15
        )
        
        # C. Update Best & Virtual Weights
        curr_max, curr_pids = torch.max(pack_weights, dim=1)
        
        # Check if this iteration produced a better result for any batch
        improved = curr_max < best_max_loads
        if improved.any():
            best_max_loads = torch.where(improved, curr_max, best_max_loads)
            # Expand mask to [B, M, G]
            mask_idx = improved.view(B, 1, 1).expand(-1, num_packs, groups_per_pack)
            best_pack_indices = torch.where(mask_idx, pack_indices, best_pack_indices)
            
        # Reweight items in max pack for next iteration
        # This increases the priority of items that ended up in the heaviest pack
        gather_idx = curr_pids.view(B, 1, 1).expand(-1, 1, groups_per_pack)
        items_in_max = torch.gather(pack_indices, 1, gather_idx).squeeze(1)
        
        vals = torch.gather(virtual_weights, 1, items_in_max)
        vals *= 1.05 # 5% boost
        virtual_weights.scatter_(1, items_in_max, vals)

    # 3. Final Selection (Best of Candidates)
    # best_max_loads: [B] -> [L, K]
    b_reshaped = best_max_loads.view(num_layers, num_candidates)
    best_k = torch.argmin(b_reshaped, dim=1)
    
    offsets = torch.arange(num_layers, device=device) * num_candidates
    final_indices = offsets + best_k
    
    final_assignment = best_pack_indices[final_indices] # [L, M, G]
    
    # 4. Construct Output
    pack_index = torch.empty_like(weight, dtype=torch.int64)
    rank_in_pack = torch.empty_like(weight, dtype=torch.int64)
    
    flat_assignment = final_assignment.view(num_layers, -1)
    
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
    Vectorized Jefferson/D'Hondt method.
    """
    n, num_log = weight.shape
    device = weight.device
    
    logcnt = torch.ones((n, num_log), dtype=torch.int64, device=device)
    
    for _ in range(num_log, num_phy):
        scores = weight / logcnt
        indices = torch.argmax(scores, dim=-1)
        rows = torch.arange(n, device=device)
        logcnt[rows, indices] += 1

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