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

    Uses Hybrid Initialization (LPT+Noise, Interleaved, Random) and 
    Deep Refinement (4-Way Snake Rebalance + Max-Any Swap).

    Parameters:
        weight: [layers, n], the weight of each item
        num_packs: number of packs

    Returns:
        pack_index: [layers, n], the pack index of each item
        rank_in_pack: [layers, n], the rank of the item in the pack
    """
    num_layers, num_groups = weight.shape
    device = weight.device
    assert num_groups % num_packs == 0
    groups_per_pack = num_groups // num_packs

    # Trivial case
    if groups_per_pack == 1:
        pack_index = torch.arange(num_packs, dtype=torch.int64, device=device).expand(num_layers, -1)
        rank_in_pack = torch.zeros_like(pack_index)
        return pack_index, rank_in_pack

    # --- 1. Candidate Generation ---
    # 256 candidates for broad search
    num_candidates = 256
    num_problems = num_layers * num_candidates
    w_expanded = weight.repeat_interleave(num_candidates, dim=0) # [LC, N]

    # Strategies:
    # 0: Pure LPT
    # 1-127: Randomized LPT (varying noise)
    # 128-191: Interleaved (Heavy-Light)
    # 192-255: Random / High Noise
    
    # Noise scales
    scales = torch.linspace(0, 0.4, 128, device=device)
    # Pad for other strategies
    scales = torch.cat([scales, torch.zeros(128, device=device)])
    
    noise_scale = scales.repeat(num_layers).view(-1, 1)
    noise = torch.rand_like(w_expanded) * w_expanded * noise_scale
    
    # Pure LPT for Cand 0
    noise.view(num_layers, num_candidates, num_groups)[:, 0, :] = 0
    
    sort_keys = w_expanded + noise
    _, sorted_indices = sort_keys.sort(dim=-1, descending=True)
    
    # Strategy B: Interleaved (128-191)
    cand_ids = torch.arange(num_candidates, device=device).repeat_interleave(num_layers)
    mask_interleave = (cand_ids >= 128) & (cand_ids < 192)
    
    if mask_interleave.any():
        perms = torch.empty(num_groups, dtype=torch.long, device=device)
        perms[0::2] = torch.arange((num_groups + 1) // 2, device=device)
        perms[1::2] = torch.arange(num_groups - 1, (num_groups + 1) // 2 - 1, -1, device=device)
        subset = sorted_indices[mask_interleave]
        sorted_indices[mask_interleave] = subset[:, perms]

    # Strategy C: Random (192-255)
    mask_random = (cand_ids >= 192)
    if mask_random.any():
        rand_keys = torch.rand(mask_random.sum(), num_groups, device=device)
        _, rand_idx = rand_keys.sort(dim=-1)
        sorted_indices[mask_random] = rand_idx

    sorted_weight = torch.gather(w_expanded, 1, sorted_indices)

    # --- 2. Vectorized Greedy Packing ---
    pack_weights = torch.zeros(num_problems, num_packs, device=device, dtype=weight.dtype)
    pack_counts = torch.zeros(num_problems, num_packs, device=device, dtype=torch.int64)
    assignments = torch.zeros(num_problems, num_groups, dtype=torch.int64, device=device)
    
    inf_val = float('inf')

    for i in range(num_groups):
        w_item = sorted_weight[:, i:i+1]
        is_full = (pack_counts >= groups_per_pack)
        costs = pack_weights.clone()
        costs[is_full] = inf_val
        chosen_pack = costs.argmin(dim=1, keepdim=True)
        
        assignments[:, i:i+1] = chosen_pack
        pack_weights.scatter_add_(1, chosen_pack, w_item)
        pack_counts.scatter_add_(1, chosen_pack, torch.ones_like(chosen_pack))

    # Convert to Pack Contents
    _, sort_by_pack_order = assignments.sort(dim=1, stable=True)
    pack_contents = sort_by_pack_order.view(num_problems, num_packs, groups_per_pack)
    K = groups_per_pack
    
    # --- 3. Refinement: 4-Way Snake Rebalance ---
    # Select Top-2 Heavy and Bottom-2 Light packs.
    # Pool items, sort, and distribute using Snake pattern (0,1,2,3,3,2,1,0...)
    # Only applicable if num_packs >= 4.
    
    if num_packs >= 4:
        # Snake Pattern Indices
        # For 4 packs with K items each -> 4K items total
        # Pattern length 8: 0,1,2,3,3,2,1,0
        indices_4k = torch.arange(4 * K, device=device)
        snake_map = torch.empty(4 * K, dtype=torch.long, device=device)
        
        # 0, 1, 2, 3
        snake_map[0::8] = 0
        snake_map[1::8] = 1
        snake_map[2::8] = 2
        snake_map[3::8] = 3
        # 3, 2, 1, 0
        snake_map[4::8] = 3
        snake_map[5::8] = 2
        snake_map[6::8] = 1
        snake_map[7::8] = 0
        
        # Masks for redistribution
        mask_0 = (snake_map == 0)
        mask_1 = (snake_map == 1)
        mask_2 = (snake_map == 2)
        mask_3 = (snake_map == 3)
        
        num_iters_snake = 20
        
        for _ in range(num_iters_snake):
            # Calc Loads
            flat_contents = pack_contents.view(num_problems, -1)
            curr_w = torch.gather(sorted_weight, 1, flat_contents).view(num_problems, num_packs, K)
            pack_sums = curr_w.sum(dim=2)
            
            # Sort packs by load to find Top 2 and Bottom 2
            # Use sort instead of topk/min for stability and getting indices easily
            _, p_sorted = pack_sums.sort(dim=1, descending=True)
            
            # Indices of packs involved
            p_0 = p_sorted[:, 0:1]   # Max
            p_1 = p_sorted[:, 1:2]   # 2nd Max
            p_2 = p_sorted[:, -2:-1] # 2nd Min
            p_3 = p_sorted[:, -1:]   # Min
            
            # Gather Items from these 4 packs
            gather_idx = torch.cat([p_0, p_1, p_2, p_3], dim=1).unsqueeze(2).expand(-1, -1, K) # [LC, 4, K]
            
            # Indices in sorted_weight
            involved_indices = torch.gather(pack_contents, 1, gather_idx) # [LC, 4, K]
            involved_indices_flat = involved_indices.view(num_problems, 4 * K)
            
            # Weights
            w_involved = torch.gather(sorted_weight, 1, involved_indices_flat)
            
            # Sort Pooled Items
            _, sort_pooled = w_involved.sort(dim=1, descending=True)
            sorted_pooled_indices = torch.gather(involved_indices_flat, 1, sort_pooled)
            
            # Distribute via Snake Map
            new_p0 = sorted_pooled_indices[:, mask_0]
            new_p1 = sorted_pooled_indices[:, mask_1]
            new_p2 = sorted_pooled_indices[:, mask_2]
            new_p3 = sorted_pooled_indices[:, mask_3]
            
            # Write back
            # We scatter back into pack_contents using the p_x indices
            p_0_idx = p_0.expand(-1, K)
            p_1_idx = p_1.expand(-1, K)
            p_2_idx = p_2.expand(-1, K)
            p_3_idx = p_3.expand(-1, K)

            pack_contents.scatter_(1, p_0_idx.unsqueeze(2), new_p0.unsqueeze(1))
            pack_contents.scatter_(1, p_1_idx.unsqueeze(2), new_p1.unsqueeze(1))
            pack_contents.scatter_(1, p_2_idx.unsqueeze(2), new_p2.unsqueeze(1))
            pack_contents.scatter_(1, p_3_idx.unsqueeze(2), new_p3.unsqueeze(1))
            
    else:
        # Fallback to 2-way ABBA if num_packs < 4
        arange_2k = torch.arange(2 * K, device=device)
        mask_b = (arange_2k % 4 == 1) | (arange_2k % 4 == 2)
        idx_a = torch.nonzero(~mask_b).squeeze()
        idx_b = torch.nonzero(mask_b).squeeze()
        
        for _ in range(20):
            flat_contents = pack_contents.view(num_problems, -1)
            curr_w = torch.gather(sorted_weight, 1, flat_contents).view(num_problems, num_packs, K)
            pack_sums = curr_w.sum(dim=2)
            
            _, p_max = pack_sums.max(dim=1)
            _, p_min = pack_sums.min(dim=1)
            
            gather_max = p_max.view(-1, 1, 1).expand(-1, 1, K)
            gather_min = p_min.view(-1, 1, 1).expand(-1, 1, K)
            
            indices_max = torch.gather(pack_contents, 1, gather_max).squeeze(1)
            indices_min = torch.gather(pack_contents, 1, gather_min).squeeze(1)
            
            w_max_items = torch.gather(sorted_weight, 1, indices_max)
            w_min_items = torch.gather(sorted_weight, 1, indices_min)
            
            merged_indices = torch.cat([indices_max, indices_min], dim=1)
            merged_weights = torch.cat([w_max_items, w_min_items], dim=1)
            
            _, sort_idx = merged_weights.sort(dim=1, descending=True)
            sorted_merged_indices = torch.gather(merged_indices, 1, sort_idx)
            
            batch_idx = torch.arange(num_problems, device=device)
            pack_contents[batch_idx, p_max, :] = sorted_merged_indices[:, idx_a]
            pack_contents[batch_idx, p_min, :] = sorted_merged_indices[:, idx_b]

    # --- 4. Refinement: Max-Any Swap ---
    # 15 iters
    for _ in range(15):
        flat_contents = pack_contents.view(num_problems, -1)
        curr_w = torch.gather(sorted_weight, 1, flat_contents).view(num_problems, num_packs, K)
        pack_sums = curr_w.sum(dim=2)
        
        val_max, p_max = pack_sums.max(dim=1)
        
        gather_max = p_max.view(-1, 1, 1).expand(-1, 1, K)
        w_max = torch.gather(curr_w, 1, gather_max)
        
        diffs = w_max.unsqueeze(3) - curr_w.unsqueeze(2)
        
        new_max_val = val_max.view(-1, 1, 1, 1) - diffs
        new_other_val = pack_sums.view(num_problems, num_packs, 1, 1) + diffs
        
        new_pair_max = torch.max(new_max_val, new_other_val)
        improvement = val_max.view(-1, 1, 1, 1) - new_pair_max
        
        mask_self = (torch.arange(num_packs, device=device).view(1, -1) == p_max.view(-1, 1))
        mask_self = mask_self.view(num_problems, num_packs, 1, 1)
        
        valid = (diffs > 0) & (improvement > 1e-6) & (~mask_self)
        scores = torch.where(valid, improvement, torch.tensor(float('-inf'), device=device))
        
        best_imp, best_idx = scores.view(num_problems, -1).max(dim=1)
        
        if not (best_imp > 0).any(): break
        
        idx_do = torch.nonzero(best_imp > 0).squeeze(1)
        if len(idx_do) == 0: break
        
        sel_idx = best_idx[idx_do]
        
        K2 = K * K
        p_target = sel_idx // K2
        rem = sel_idx % K2
        k_max = rem // K
        k_target = rem % K
        
        p_max_do = p_max[idx_do]
        
        val_max_idx = pack_contents[idx_do, p_max_do, k_max]
        val_target_idx = pack_contents[idx_do, p_target, k_target]
        
        pack_contents[idx_do, p_max_do, k_max] = val_target_idx
        pack_contents[idx_do, p_target, k_target] = val_max_idx

    # --- 5. Selection with L2 Tie-Breaking ---
    flat_contents = pack_contents.view(num_problems, -1)
    final_w = torch.gather(sorted_weight, 1, flat_contents).view(num_problems, num_packs, K)
    
    pack_loads = final_w.sum(dim=2) # [LC, M]
    max_loads = pack_loads.max(dim=1).values # [LC]
    l2_norms = (pack_loads ** 2).sum(dim=1) # [LC]
    
    # Reshape to [Layers, Candidates]
    max_loads = max_loads.view(num_layers, num_candidates)
    l2_norms = l2_norms.view(num_layers, num_candidates)
    
    # Robust Selection:
    # 1. Find Min MaxLoad per layer
    min_max = max_loads.min(dim=1, keepdim=True).values
    
    # 2. Identify candidates within tolerance of Min MaxLoad
    # Since weights are floats, use small epsilon relative to magnitude
    is_optimal = (max_loads <= min_max * 1.000001 + 1e-6)
    
    # 3. Mask L2 norms of non-optimal candidates
    # We want to minimize L2 among optimals.
    # Set non-optimal L2 to infinity.
    masked_l2 = torch.where(is_optimal, l2_norms, torch.tensor(float('inf'), device=device))
    
    # 4. Pick candidate with min L2
    best_cand = masked_l2.argmin(dim=1)
    
    # Gather Result
    best_global_idx = torch.arange(num_layers, device=device) * num_candidates + best_cand
    
    best_contents = pack_contents[best_global_idx]
    best_sorted_idx = sorted_indices[best_global_idx]
    
    flat_best = best_contents.view(num_layers, -1)
    original_idx = torch.gather(best_sorted_idx, 1, flat_best)
    
    pack_index = torch.empty(num_layers, num_groups, dtype=torch.long, device=device)
    rank_in_pack = torch.empty(num_layers, num_groups, dtype=torch.long, device=device)
    
    p_ids = torch.arange(num_packs, device=device).view(1, num_packs, 1).expand(num_layers, -1, K).reshape(num_layers, -1)
    r_ids = torch.arange(K, device=device).view(1, 1, K).expand(num_layers, num_packs, -1).reshape(num_layers, -1)
    
    pack_index.scatter_(1, original_idx, p_ids)
    rank_in_pack.scatter_(1, original_idx, r_ids)
    
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

    # Initialize with 1 replica per expert
    phy2log = torch.arange(num_phy, dtype=torch.int64, device=device).repeat(n, 1)
    rank = torch.zeros(n, num_phy, dtype=torch.int64, device=device)
    logcnt = torch.ones(n, num_log, dtype=torch.int64, device=device)
    arangen = torch.arange(n, dtype=torch.int64, device=device)

    # Greedily add replicas to the expert with highest current load per replica
    for i in range(num_log, num_phy):
        # Metric: current load per replica = weight / count
        redundant_indices = (weight / logcnt).max(dim=-1).indices

        phy2log[:, i] = redundant_indices
        rank[:, i] = logcnt[arangen, redundant_indices]
        logcnt[arangen, redundant_indices] += 1

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
    # Ensure weight is float for calculations. Keep on original device for speed.
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
