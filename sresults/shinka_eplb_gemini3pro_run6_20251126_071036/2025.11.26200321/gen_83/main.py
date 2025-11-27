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

    Uses Diverse Initialization followed by Multi-Pack Pooling (Snake Redistrib)
    and Max-Any Swap refinement.

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
    # Generate 256 candidates to explore search space
    num_candidates = 256
    num_problems = num_layers * num_candidates
    w_expanded = weight.repeat_interleave(num_candidates, dim=0) # [LC, N]

    # Strategy configuration
    # 0-63:    LPT + Low Noise (Refined Greedy)
    # 64-127:  LPT + High Noise (Exploration)
    # 128-191: Interleaved (Heavy-Light-Heavy-Light)
    # 192-255: Random / Very High Noise
    
    scales = torch.linspace(0, 0.4, 128, device=device)
    scales = torch.cat([scales, torch.zeros(64, device=device), torch.ones(64, device=device) * 100.0])
    
    noise_scale = scales.repeat(num_layers).view(-1, 1)
    noise = torch.rand_like(w_expanded) * w_expanded * noise_scale
    
    # Ensure Candidate 0 is Baseline LPT
    noise.view(num_layers, num_candidates, num_groups)[:, 0, :] = 0
    
    sort_keys = w_expanded + noise
    _, sorted_indices = sort_keys.sort(dim=-1, descending=True)
    
    # Apply Interleaved Strategy (128-191)
    # Permutes the sorted indices to be [0, N-1, 1, N-2, ...]
    cand_ids = torch.arange(num_candidates, device=device).repeat_interleave(num_layers)
    mask_interleave = (cand_ids >= 128) & (cand_ids < 192)
    
    if mask_interleave.any():
        perm = torch.empty(num_groups, dtype=torch.long, device=device)
        perm[0::2] = torch.arange((num_groups + 1) // 2, device=device)
        perm[1::2] = torch.arange(num_groups - 1, (num_groups + 1) // 2 - 1, -1, device=device)
        
        subset = sorted_indices[mask_interleave]
        sorted_indices[mask_interleave] = subset[:, perm]
        
    sorted_weight = torch.gather(w_expanded, 1, sorted_indices)

    # --- 2. Vectorized Greedy Packing ---
    pack_weights = torch.zeros(num_problems, num_packs, device=device, dtype=weight.dtype)
    pack_counts = torch.zeros(num_problems, num_packs, device=device, dtype=torch.int64)
    assignments = torch.zeros(num_problems, num_groups, dtype=torch.int64, device=device)
    
    inf_val = float('inf')
    
    # Simple Greedy
    for i in range(num_groups):
        w_item = sorted_weight[:, i:i+1]
        
        is_full = (pack_counts >= groups_per_pack)
        costs = pack_weights.clone()
        costs[is_full] = inf_val
        
        chosen_pack = costs.argmin(dim=1, keepdim=True)
        assignments[:, i:i+1] = chosen_pack
        pack_weights.scatter_add_(1, chosen_pack, w_item)
        pack_counts.scatter_add_(1, chosen_pack, torch.ones_like(chosen_pack))
        
    # Organize into [Batch, Pack, Group]
    _, sort_by_pack = assignments.sort(dim=1, stable=True)
    pack_contents = sort_by_pack.view(num_problems, num_packs, groups_per_pack)
    
    K = groups_per_pack
    
    # --- 3. Refinement: Multi-Pack Destruct-Reconstruct (Snake) ---
    # Pool items from Top-2 Heavy and Top-2 Light packs.
    # Redistribute using a Snake pattern (0,1,2,3,3,2,1,0) to balance.
    
    num_pool = 4
    if num_packs < num_pool:
        num_pool = num_packs
        
    # Precompute distribution pattern
    # Cycle: 0, 1, ... M-1, M-1, ... 0
    cycle = torch.cat([torch.arange(num_pool, device=device), 
                       torch.arange(num_pool - 1, -1, -1, device=device)])
    
    total_items = num_pool * K
    pattern_ids = torch.arange(total_items, device=device)
    item_to_pack_map = cycle[pattern_ids % len(cycle)]
    
    # Get permutation to group items by target pack
    _, distrib_perm = item_to_pack_map.sort(stable=True)
    
    num_iters_pool = 20
    
    for _ in range(num_iters_pool):
        # 1. Identify Target Packs
        flat_c = pack_contents.view(num_problems, -1)
        curr_w = torch.gather(sorted_weight, 1, flat_c).view(num_problems, num_packs, K)
        p_sums = curr_w.sum(dim=2)
        
        # Add noise to break ties
        noisy_sums = p_sums + torch.rand_like(p_sums) * 1e-6
        _, p_sorted = noisy_sums.sort(dim=1, descending=True)
        
        # Select: Top 2 Heavy + Top 2 Light
        if num_packs >= 4:
            p_sel = torch.cat([p_sorted[:, :2], p_sorted[:, -2:]], dim=1) # [Batch, 4]
        else:
            p_sel = p_sorted # [Batch, M]
            
        # 2. Gather Items
        gather_idx = p_sel.unsqueeze(2).expand(-1, -1, K)
        # indices in sorted_weight
        items_idx = torch.gather(pack_contents, 1, gather_idx).view(num_problems, -1)
        items_w = torch.gather(sorted_weight, 1, items_idx)
        
        # 3. Sort Pooled Items
        _, sort_items = items_w.sort(dim=1, descending=True)
        sorted_items_idx = torch.gather(items_idx, 1, sort_items) # [Batch, 4K]
        
        # 4. Redistribute via Pattern
        # Rearrange sorted items according to distrib_perm to group by target pack 0..3
        grouped_items = sorted_items_idx[:, distrib_perm]
        
        # Reshape to [Batch, 4, K]
        new_contents = grouped_items.view(num_problems, num_pool, K)
        
        # 5. Scatter back to Pack Contents
        # new_contents[:, j, :] belongs to pack p_sel[:, j]
        for j in range(num_pool):
            # We must use advanced indexing for batch+pack
            # pack_contents[batch, p_sel[:, j], :] = new_contents[:, j, :]
            # Since p_sel varies per batch row, we flatten or loop.
            # Looping over 4 is cheap.
            
            # Construct flat indices
            batch_idx = torch.arange(num_problems, device=device)
            pack_idx = p_sel[:, j]
            pack_contents[batch_idx, pack_idx, :] = new_contents[:, j, :]
            
    # --- 4. Refinement: Max-Any Swap Polish ---
    # Standard refinement for fine-tuning the Max Pack
    
    num_iters_swap = 20
    for _ in range(num_iters_swap):
        flat_c = pack_contents.view(num_problems, -1)
        curr_w = torch.gather(sorted_weight, 1, flat_c).view(num_problems, num_packs, K)
        p_sums = curr_w.sum(dim=2)
        
        val_max, idx_max = p_sums.max(dim=1)
        
        # Get Max Pack Items
        gather_max = idx_max.view(-1, 1, 1).expand(-1, 1, K)
        w_max = torch.gather(curr_w, 1, gather_max) # [B, 1, K]
        
        # Diffs: [B, 1, K, 1] - [B, M, 1, K]
        diffs = w_max.unsqueeze(3) - curr_w.unsqueeze(2)
        
        # New Max vs New Other
        new_pair_max = torch.max(val_max.view(-1, 1, 1, 1) - diffs,
                                 p_sums.view(num_problems, num_packs, 1, 1) + diffs)
        
        improvement = val_max.view(-1, 1, 1, 1) - new_pair_max
        
        mask_self = (torch.arange(num_packs, device=device).view(1, -1) == idx_max.view(-1, 1))
        mask_self = mask_self.view(num_problems, num_packs, 1, 1)
        
        valid = (diffs > 0) & (improvement > 1e-6) & (~mask_self)
        
        improvement = torch.where(valid, improvement, torch.tensor(float('-inf'), device=device))
        
        flat_imp = improvement.view(num_problems, -1)
        best_imp, best_idx = flat_imp.max(dim=1)
        
        if not (best_imp > 0).any():
            break
            
        active = torch.nonzero(best_imp > 0).squeeze(1)
        sel_idx = best_idx[active]
        
        K2 = K * K
        p_target = sel_idx // K2
        rem = sel_idx % K2
        k_max = rem // K
        k_target = rem % K
        
        p_mx = idx_max[active]
        
        # Swap
        v1 = pack_contents[active, p_mx, k_max]
        v2 = pack_contents[active, p_target, k_target]
        
        pack_contents[active, p_mx, k_max] = v2
        pack_contents[active, p_target, k_target] = v1

    # --- 5. Final Selection ---
    flat_c = pack_contents.view(num_problems, -1)
    final_w = torch.gather(sorted_weight, 1, flat_c).view(num_problems, num_packs, K)
    
    max_loads = final_w.sum(dim=2).max(dim=1).values
    max_loads = max_loads.view(num_layers, num_candidates)
    
    best_cand = max_loads.argmin(dim=1)
    best_offset = torch.arange(num_layers, device=device) * num_candidates + best_cand
    
    best_contents = pack_contents[best_offset]
    best_sorted_idx = sorted_indices[best_offset]
    
    # Map back
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
