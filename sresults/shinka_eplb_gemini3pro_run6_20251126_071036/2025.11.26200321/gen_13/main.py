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
    
    Uses a Hybrid Initialization (ZigZag + Greedy) followed by a vectorized 
    Max-Any Swap local search refinement.

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

    # Sort weights descending: [L, N]
    sorted_weight, sorted_indices = weight.sort(dim=-1, descending=True)

    # --- Strategy 1: ZigZag Init ---
    # Deterministic pattern 0..M-1, M-1..0 repeated
    pattern = torch.cat([
        torch.arange(num_packs, device=device),
        torch.arange(num_packs - 1, -1, -1, device=device)
    ])
    num_patterns = (num_groups + len(pattern) - 1) // len(pattern)
    zigzag_assignments = pattern.repeat(num_patterns)[:num_groups] # [N]
    zigzag_pack_index = zigzag_assignments.unsqueeze(0).expand(num_layers, -1) # [L, N]

    # Calculate ZigZag max loads
    zigzag_loads = torch.zeros(num_layers, num_packs, device=device, dtype=weight.dtype)
    zigzag_loads.scatter_add_(1, zigzag_pack_index, sorted_weight)
    zigzag_max, _ = zigzag_loads.max(dim=1)

    # --- Strategy 2: Constrained Greedy LPT Init ---
    # Only run greedy if N is reasonable to avoid excessive Python loop overhead
    # For very large N, the loop overhead might outweigh benefits, but N=num_groups is usually small (e.g. 256)
    if num_groups <= 512:
        greedy_pack_index = torch.zeros(num_layers, num_groups, dtype=torch.int64, device=device)
        current_loads = torch.zeros(num_layers, num_packs, dtype=weight.dtype, device=device)
        current_counts = torch.zeros(num_layers, num_packs, dtype=torch.int64, device=device)
        
        # Iterate through items (already sorted descending)
        for i in range(num_groups):
            w = sorted_weight[:, i:i+1] # [L, 1]
            
            # Mask full packs
            is_full = current_counts >= groups_per_pack
            masked_loads = current_loads.clone()
            masked_loads[is_full] = float('inf')
            
            # Select pack with min weight among valid
            chosen_pack = masked_loads.argmin(dim=1, keepdim=True) # [L, 1]
            
            greedy_pack_index[:, i:i+1] = chosen_pack
            current_loads.scatter_add_(1, chosen_pack, w)
            current_counts.scatter_add_(1, chosen_pack, torch.ones_like(chosen_pack))
            
        greedy_max, _ = current_loads.max(dim=1)
        
        # Select Best Init per layer
        use_greedy = greedy_max < zigzag_max
        mask_greedy = use_greedy.unsqueeze(1).expand(-1, num_groups)
        current_pack_index = torch.where(mask_greedy, greedy_pack_index, zigzag_pack_index)
    else:
        current_pack_index = zigzag_pack_index

    # --- Local Search: Max-Any Swap ---
    num_iters = 20
    
    for _ in range(num_iters):
        # 1. Compute Pack Loads
        pack_loads = torch.zeros(num_layers, num_packs, device=device, dtype=weight.dtype)
        pack_loads.scatter_add_(1, current_pack_index, sorted_weight)
        
        # 2. Identify Max Pack
        max_load, max_pack_idx = pack_loads.max(dim=1) # [L], [L]
        
        # 3. Organize items by pack to easily find items in MaxPack
        # Sort current_pack_index to group items by pack
        # argsort is [L, N]. indices[l, 0..K-1] are items in pack 0, etc.
        # But we need to know which pack corresponds to which block.
        # current_pack_index values are not necessarily ordered 0..M. 
        # Actually argsort groups 0s, then 1s, then 2s. 
        # So item_indices[l, m*K : (m+1)*K] are the items in pack m.
        item_indices_by_pack = current_pack_index.argsort(dim=1, stable=True) # [L, N]
        grouped_items = item_indices_by_pack.view(num_layers, num_packs, groups_per_pack) # [L, M, K]
        
        # Gather items belonging to MaxPack: [L, K]
        # We need to select the block 'm' corresponding to max_pack_idx[l]
        # max_pack_idx: [L] -> expand to [L, 1, K]
        gather_idx = max_pack_idx.view(num_layers, 1, 1).expand(-1, 1, groups_per_pack)
        items_in_max = torch.gather(grouped_items, 1, gather_idx).squeeze(1) # [L, K]
        
        # Get weights of items in MaxPack: [L, K]
        weights_in_max = torch.gather(sorted_weight, 1, items_in_max)
        
        # 4. Compute Swap Diffs [L, K, N]
        # We consider swapping each item 'u' in MaxPack with any item 'v' in ANY pack.
        # diff = w_u - w_v
        # new_max = max_load - diff
        # new_other = load_other + diff
        # We want to minimize max(new_max, new_other).
        
        # [L, K, 1] - [L, 1, N] -> [L, K, N]
        diffs = weights_in_max.unsqueeze(2) - sorted_weight.unsqueeze(1)
        
        # Filter valid swaps:
        # 1. v must NOT be in MaxPack.
        is_v_in_max = (current_pack_index == max_pack_idx.unsqueeze(1)) # [L, N]
        valid_mask = ~is_v_in_max.unsqueeze(1) # [L, 1, N] broadcasts to [L, K, N]
        
        # 2. diff > 0 (Must reduce max pack)
        valid_mask &= (diffs > 0)
        
        if not valid_mask.any():
            break
            
        # Target load for 'other' packs
        # load_v depends on which pack v is in.
        # Get pack of each v: [L, N]
        pack_of_v = current_pack_index
        # Get load of pack_of_v: [L, N]
        load_of_pack_v = torch.gather(pack_loads, 1, pack_of_v)
        
        # Expand to [L, K, N]
        load_other_expanded = load_of_pack_v.unsqueeze(1)
        max_load_expanded = max_load.view(num_layers, 1, 1)
        
        # Calculate new max of the pair
        # new_max_load_self = max_load - diff
        # new_other_load = load_other + diff
        # optimization_metric = max(new_max_load_self, new_other_load)
        
        metric = torch.max(max_load_expanded - diffs, load_other_expanded + diffs)
        
        # Improvement = Old Max - New Pair Max
        improvement = max_load_expanded - metric
        
        # Apply mask
        improvement = torch.where(valid_mask, improvement, torch.tensor(float('-inf'), device=device))
        
        # Find best swap per layer
        best_imp_flat, best_idx_flat = improvement.view(num_layers, -1).max(dim=1)
        
        # Check if improvement is positive
        do_swap = best_imp_flat > 1e-6
        
        if not do_swap.any():
            break
            
        # Execute swaps
        l_idx = torch.nonzero(do_swap).squeeze(1)
        
        # Decode indices
        # flat index in range [0, K*N)
        idx_u_in_max = (best_idx_flat[l_idx] // num_groups) # Index 0..K-1
        idx_v_global = (best_idx_flat[l_idx] % num_groups)  # Index 0..N-1
        
        # Get actual item index for u
        u_global = items_in_max[l_idx, idx_u_in_max]
        v_global = idx_v_global
        
        # Get packs
        p_u = max_pack_idx[l_idx]
        p_v = pack_of_v[l_idx, v_global]
        
        # Perform swap in current_pack_index
        current_pack_index[l_idx, u_global] = p_v
        current_pack_index[l_idx, v_global] = p_u

    # --- Construct Output ---
    # Map back to original item indices
    pack_index = torch.empty_like(current_pack_index)
    pack_index.scatter_(1, sorted_indices, current_pack_index)

    # Compute ranks
    # Sort items by pack (stable sort keeps heaviest first)
    pack_sort_idx = current_pack_index.argsort(dim=1, stable=True)
    
    # Ranks pattern: 0, 1, ..., K-1 repeated M times
    ranks_pattern = torch.arange(groups_per_pack, device=device).repeat(num_packs).expand(num_layers, -1)
    
    # Map ranks to sorted_pack positions
    sorted_ranks = torch.empty_like(ranks_pattern)
    sorted_ranks.scatter_(1, pack_sort_idx, ranks_pattern)
    
    # Map back to original item order
    rank_in_pack = torch.empty_like(pack_index)
    rank_in_pack.scatter_(1, sorted_indices, sorted_ranks)

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
        # Find expert with max metric
        metrics = weight / logcnt
        redundant_indices = metrics.max(dim=-1).indices # [N]
        
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
    # [num_layers * num_nodes, num_logical_experts // num_nodes]
    tokens_per_mlog = weight.gather(-1, mlog2log).view(
        -1, num_logical_experts // num_nodes)
        
    phy2mlog, phyrank, mlogcnt = replicate_experts(
        tokens_per_mlog, num_physical_experts // num_nodes)

    # Step 3: pack physical_experts to GPUs
    # [num_layers * num_nodes, num_physical_experts // num_nodes]
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
    # NOTE: We keep weight on its original device to allow GPU acceleration.
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
    
    # Scatter to create the reverse map
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
