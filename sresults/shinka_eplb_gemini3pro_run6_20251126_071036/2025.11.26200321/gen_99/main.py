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

    Uses a Zoom-In Strategy with Snake Initialization, L2 Tie-Breaking, 
    and Top-4 Snake Refinement.

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

    # --- 1. Candidate Generation (Exploration) ---
    num_candidates = 256
    num_problems = num_layers * num_candidates
    w_expanded = weight.repeat_interleave(num_candidates, dim=0) # [LC, N]
    cand_ids = torch.arange(num_candidates, device=device).repeat_interleave(num_layers)

    # Strategy A: Randomized LPT (0-127)
    # Variable noise scales to explore different greedy choices
    scales = torch.linspace(0, 0.4, 128, device=device)
    scales = torch.cat([scales, torch.zeros(128, device=device)]) # Pad
    noise_scale = scales.repeat(num_layers).view(-1, 1)
    
    # Add noise
    noise = torch.rand_like(w_expanded) * w_expanded * noise_scale
    # Ensure pure LPT for candidate 0 of each layer
    noise.view(num_layers, num_candidates, num_groups)[:, 0, :] = 0
    
    sort_keys = w_expanded + noise
    _, sorted_indices = sort_keys.sort(dim=-1, descending=True)

    # Strategy B: Interleaved (128-191)
    # Permute indices to (Heavy, Light, Heavy, Light...)
    mask_interleave = (cand_ids >= 128) & (cand_ids < 192)
    if mask_interleave.any():
        perms = torch.empty(num_groups, dtype=torch.long, device=device)
        perms[0::2] = torch.arange((num_groups + 1) // 2, device=device)
        perms[1::2] = torch.arange(num_groups - 1, (num_groups + 1) // 2 - 1, -1, device=device)
        subset = sorted_indices[mask_interleave]
        sorted_indices[mask_interleave] = subset[:, perms]

    # Strategy C: Snake Initialization (192-255)
    # Handled in the greedy loop via masking. These candidates stay strictly LPT sorted.
    mask_snake = (cand_ids >= 192)

    sorted_weight = torch.gather(w_expanded, 1, sorted_indices)

    # Vectorized Greedy Packing
    pack_weights = torch.zeros(num_problems, num_packs, device=device, dtype=weight.dtype)
    pack_counts = torch.zeros(num_problems, num_packs, device=device, dtype=torch.int64)
    assignments = torch.zeros(num_problems, num_groups, dtype=torch.int64, device=device)

    inf_val = float('inf')
    
    # Snake setup
    do_snake = mask_snake.any()

    for i in range(num_groups):
        w_item = sorted_weight[:, i:i+1]
        
        costs = pack_weights.clone()
        
        # Apply Snake Forcing for first 2*M items
        # Forces largest items into distinct packs (Round Robin) then reverses (Snake)
        if do_snake and i < 2 * num_packs:
            target = i if i < num_packs else (2 * num_packs - 1 - i)
            
            # Apply to snake candidates: set costs of non-target packs to inf, target to -inf
            # This forces the greedy argmin to pick 'target'
            costs[mask_snake, :] = inf_val
            costs[mask_snake, target] = -inf_val

        # Mask full packs
        is_full = (pack_counts >= groups_per_pack)
        costs[is_full] = inf_val
        
        chosen_pack = costs.argmin(dim=1, keepdim=True)
        assignments[:, i:i+1] = chosen_pack
        pack_weights.scatter_add_(1, chosen_pack, w_item)
        pack_counts.scatter_add_(1, chosen_pack, torch.ones_like(chosen_pack))

    # --- 2. Zoom-In Selection (L2 Tie-Breaking) ---
    _, sort_by_pack = assignments.sort(dim=1, stable=True)
    pack_contents = sort_by_pack.view(num_problems, num_packs, groups_per_pack)
    K = groups_per_pack

    # Calculate Metrics
    flat_c = pack_contents.view(num_problems, -1)
    curr_w = torch.gather(sorted_weight, 1, flat_c).view(num_problems, num_packs, K)
    pack_sums = curr_w.sum(dim=2)
    
    max_loads = pack_sums.max(dim=1).values.view(num_layers, num_candidates)
    l2_norms = (pack_sums ** 2).sum(dim=1).view(num_layers, num_candidates)
    
    # L2 Tie-Breaking Score
    # Primary: Max Load, Secondary: L2 Norm (normalized magnitude)
    max_min = max_loads.min(dim=1, keepdim=True).values
    norm_scale = l2_norms.max(dim=1, keepdim=True).values + 1e-6
    score = max_loads + (l2_norms / norm_scale) * 1e-4 * max_min
    
    best_cand_idx = score.argmin(dim=1)
    global_best_idx = torch.arange(num_layers, device=device) * num_candidates + best_cand_idx
    
    # Replicate Best Candidates to 64 replicas each
    num_replicas = 64
    num_zoom = num_layers * num_replicas
    
    best_indices = sorted_indices[global_best_idx]
    best_contents = pack_contents[global_best_idx]
    best_weights = sorted_weight[global_best_idx]
    
    sorted_indices = best_indices.repeat_interleave(num_replicas, dim=0)
    pack_contents = best_contents.repeat_interleave(num_replicas, dim=0)
    sorted_weight = best_weights.repeat_interleave(num_replicas, dim=0)
    
    # --- 3. Mutation: Pairwise ABBA ---
    # Apply to 5 random pairs for replicas (skip 0th replica to preserve best greedy)
    mask_mutate = torch.ones(num_zoom, dtype=torch.bool, device=device)
    mask_mutate.view(num_layers, num_replicas)[:, 0] = False
    idx_mutate = torch.nonzero(mask_mutate).squeeze()
    
    if len(idx_mutate) > 0:
        for _ in range(5):
            # Select two distinct random packs per candidate
            p1 = torch.randint(0, num_packs, (len(idx_mutate),), device=device)
            p2 = torch.randint(0, num_packs, (len(idx_mutate),), device=device)
            p2 = torch.where(p1 == p2, (p2 + 1) % num_packs, p2)
            
            # Gather indices
            g1 = p1.view(-1, 1, 1).expand(-1, 1, K)
            g2 = p2.view(-1, 1, 1).expand(-1, 1, K)
            idx1 = torch.gather(pack_contents[idx_mutate], 1, g1).squeeze(1)
            idx2 = torch.gather(pack_contents[idx_mutate], 1, g2).squeeze(1)
            
            # Gather weights
            w1 = torch.gather(sorted_weight[idx_mutate], 1, idx1)
            w2 = torch.gather(sorted_weight[idx_mutate], 1, idx2)
            
            # Merge and ABBA Rebalance
            merged_w = torch.cat([w1, w2], dim=1)
            merged_idx = torch.cat([idx1, idx2], dim=1)
            _, s_idx = merged_w.sort(dim=1, descending=True)
            s_merged = torch.gather(merged_idx, 1, s_idx)
            
            # Mask for 2K items (A-B-B-A...)
            arange_2k = torch.arange(2 * K, device=device)
            mask_b = (arange_2k % 4 == 1) | (arange_2k % 4 == 2)
            
            pack_contents[idx_mutate, p1, :] = s_merged[:, ~mask_b]
            pack_contents[idx_mutate, p2, :] = s_merged[:, mask_b]

    # --- 4. Refinement ---
    # Interleave Top-4 Snake Balancing and Max-Any Swaps
    
    # Pre-calculate Top-4 Snake Pattern Indices
    use_top4 = (num_packs >= 4)
    if use_top4:
        pat = torch.tensor([0, 1, 2, 3, 3, 2, 1, 0], device=device)
        full_pat = pat.repeat((4 * K + 7) // 8)[:4 * K]
        idx_p0 = torch.nonzero(full_pat == 0).squeeze()
        idx_p1 = torch.nonzero(full_pat == 1).squeeze()
        idx_p2 = torch.nonzero(full_pat == 2).squeeze()
        idx_p3 = torch.nonzero(full_pat == 3).squeeze()
    
    # 20 Iterations
    for it in range(20):
        flat_c = pack_contents.view(num_zoom, -1)
        curr_w = torch.gather(sorted_weight, 1, flat_c).view(num_zoom, num_packs, K)
        pack_sums = curr_w.sum(dim=2)
        
        # Strategy Switching
        if use_top4 and (it % 2 == 0):
            # --- Top-4 Snake Balancing ---
            # Identify Top 2 and Bottom 2 packs
            sorted_packs = pack_sums.argsort(dim=1)
            p_min1 = sorted_packs[:, 0]
            p_min2 = sorted_packs[:, 1]
            p_max2 = sorted_packs[:, -2]
            p_max1 = sorted_packs[:, -1]
            
            # Gather all items from these 4 packs
            g_list = [p.view(-1, 1, 1).expand(-1, 1, K) for p in [p_min1, p_min2, p_max2, p_max1]]
            idx_list = [torch.gather(pack_contents, 1, g).squeeze(1) for g in g_list]
            w_list = [torch.gather(sorted_weight, 1, idx) for idx in idx_list]
            
            merged_idx = torch.cat(idx_list, dim=1)
            merged_w = torch.cat(w_list, dim=1)
            
            # Sort all 4K items
            _, s_idx = merged_w.sort(dim=1, descending=True)
            s_merged = torch.gather(merged_idx, 1, s_idx)
            
            # Redistribute using Snake pattern: 0->Min1, 1->Min2, 2->Max2, 3->Max1...
            batch = torch.arange(num_zoom, device=device)
            pack_contents[batch, p_min1, :] = s_merged[:, idx_p0]
            pack_contents[batch, p_min2, :] = s_merged[:, idx_p1]
            pack_contents[batch, p_max2, :] = s_merged[:, idx_p2]
            pack_contents[batch, p_max1, :] = s_merged[:, idx_p3]
            
        else:
            # --- Max-Any Swap (Plateau Surfing) ---
            val_max, p_max = pack_sums.max(dim=1)
            gather_max = p_max.view(-1, 1, 1).expand(-1, 1, K)
            w_max = torch.gather(curr_w, 1, gather_max)
            
            # Calculate potential swaps
            diffs = w_max.unsqueeze(3) - curr_w.unsqueeze(2)
            
            new_max = val_max.view(-1, 1, 1, 1) - diffs
            new_other = pack_sums.view(num_zoom, num_packs, 1, 1) + diffs
            
            new_pair_max = torch.max(new_max, new_other)
            improvement = val_max.view(-1, 1, 1, 1) - new_pair_max
            
            mask_self = (torch.arange(num_packs, device=device).view(1, -1) == p_max.view(-1, 1))
            mask_self = mask_self.view(num_zoom, num_packs, 1, 1)
            
            # Allow zero improvement (plateau surfing) if weight is moved out of max pack (diffs > 0)
            valid = (diffs > 0) & (improvement > -1e-6) & (~mask_self)
            
            scores = torch.where(valid, improvement, torch.tensor(float('-inf'), device=device))
            
            best_imp, best_idx = scores.view(num_zoom, -1).max(dim=1)
            
            # Execute Swaps
            do_swap = (best_imp > float('-inf'))
            idx_do = torch.nonzero(do_swap).squeeze(1)
            
            if len(idx_do) > 0:
                sel_idx = best_idx[idx_do]
                p_max_do = p_max[idx_do]
                
                K2 = K * K
                p_target = sel_idx // K2
                rem = sel_idx % K2
                k_max = rem // K
                k_target = rem % K
                
                v1 = pack_contents[idx_do, p_max_do, k_max]
                v2 = pack_contents[idx_do, p_target, k_target]
                
                pack_contents[idx_do, p_max_do, k_max] = v2
                pack_contents[idx_do, p_target, k_target] = v1

    # --- 5. Final Output ---
    flat_c = pack_contents.view(num_zoom, -1)
    final_w = torch.gather(sorted_weight, 1, flat_c).view(num_zoom, num_packs, K)
    final_max = final_w.sum(dim=2).max(dim=1).values
    
    # Pick best replica
    final_max = final_max.view(num_layers, num_replicas)
    best_cand = final_max.argmin(dim=1)
    
    best_offset = torch.arange(num_layers, device=device) * num_replicas + best_cand
    
    best_pack_c = pack_contents[best_offset]
    best_sort_w_idx = sorted_indices[best_offset]
    
    flat_best = best_pack_c.view(num_layers, -1)
    original_idx = torch.gather(best_sort_w_idx, 1, flat_best)
    
    # Map back
    pack_index = torch.empty(num_layers, num_groups, dtype=torch.long, device=device)
    rank_in_pack = torch.empty(num_layers, num_groups, dtype=torch.long, device=device)
    
    p_pat = torch.arange(num_packs, device=device).view(1, num_packs, 1).expand(num_layers, -1, groups_per_pack).reshape(num_layers, -1)
    r_pat = torch.arange(groups_per_pack, device=device).view(1, 1, groups_per_pack).expand(num_layers, num_packs, -1).reshape(num_layers, -1)
    
    pack_index.scatter_(1, original_idx, p_pat)
    rank_in_pack.scatter_(1, original_idx, r_pat)
    
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
        # Finding the expert where adding a replica reduces the max load the most
        # is equivalent to finding the expert with the current maximum load per replica.
        metrics = weight / logcnt
        redundant_indices = metrics.max(dim=-1).indices

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