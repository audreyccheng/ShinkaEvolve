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

    Uses Hybrid Initialization (LPT+Noise, Interleaved, Random) with Pack Offsets,
    followed by Destruct-Reconstruct (ABBA) + Max-Any Swap refinement.

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
    # Massive search space exploration with 256 candidates
    num_candidates = 256
    num_problems = num_layers * num_candidates
    w_expanded = weight.repeat_interleave(num_candidates, dim=0) # [LC, N]

    # --- Strategy Setup ---
    # 0-127:   Randomized LPT (Descending Sort + Noise)
    # 128-191: Interleaved (Heavy, Light, Heavy, Light...)
    # 192-255: Random Permutation
    
    # Generate noise scales for LPT
    # 0..63: Low noise (refined LPT)
    # 64..127: High noise (fuzzy LPT)
    scales = torch.linspace(0, 0.4, 128, device=device)
    # Pad for other strategies
    scales = torch.cat([scales, torch.zeros(128, device=device)])
    
    noise_scale = scales.repeat(num_layers).view(-1, 1)
    noise = torch.rand_like(w_expanded) * w_expanded * noise_scale
    
    # Ensure Candidate 0 is pure LPT (Baseline)
    noise.view(num_layers, num_candidates, num_groups)[:, 0, :] = 0
    
    sort_keys = w_expanded + noise
    _, sorted_indices = sort_keys.sort(dim=-1, descending=True)
    
    # Apply Strategy B: Interleaved (128-191)
    cand_ids = torch.arange(num_candidates, device=device).repeat_interleave(num_layers)
    mask_interleave = (cand_ids >= 128) & (cand_ids < 192)
    
    if mask_interleave.any():
        perms = torch.empty(num_groups, dtype=torch.long, device=device)
        # Even indices take from start (Heavy), Odd from end (Light)
        perms[0::2] = torch.arange((num_groups + 1) // 2, device=device)
        perms[1::2] = torch.arange(num_groups - 1, (num_groups + 1) // 2 - 1, -1, device=device)
        
        subset = sorted_indices[mask_interleave]
        sorted_indices[mask_interleave] = subset[:, perms]

    # Apply Strategy C: Random (192-255) is implicitly handled by high noise or random keys, 
    # but here we rely on the high noise sort or specific shuffle. 
    # Let's add explicit shuffle for the last group.
    mask_random = (cand_ids >= 192)
    if mask_random.any():
        rand_keys = torch.rand(mask_random.sum(), num_groups, device=device)
        _, rand_idx = rand_keys.sort(dim=-1)
        sorted_indices[mask_random] = rand_idx

    # Gather weights in processing order
    sorted_weight = torch.gather(w_expanded, 1, sorted_indices)

    # --- 2. Vectorized Greedy Packing with Diversity Offsets ---
    # We apply initial "Pack Offsets" to force diversity in bin selection.
    # This prevents the greedy heuristic from filling packs in the exact same order for similar sortings.
    
    pack_weights = torch.zeros(num_problems, num_packs, device=device, dtype=weight.dtype)
    pack_counts = torch.zeros(num_problems, num_packs, device=device, dtype=torch.int64)
    assignments = torch.zeros(num_problems, num_groups, dtype=torch.int64, device=device)
    
    # Generate Offsets
    # Apply offsets to: High Noise LPT (64-127) and Random (192-255)
    # Keep Pure/Low Noise LPT (0-63) and Interleaved (128-191) clean for stability.
    mask_offset = (cand_ids >= 64) & (cand_ids < 128) | (cand_ids >= 192)
    
    if mask_offset.any():
        avg_w = w_expanded.mean(dim=1, keepdim=True)
        # Random offsets up to 1.0x average weight
        offsets = torch.rand(num_problems, num_packs, device=device, dtype=weight.dtype) * avg_w
        # Zero out offsets for non-selected candidates
        offsets[~mask_offset] = 0
        pack_weights = offsets.clone()
    
    inf_val = float('inf')

    for i in range(num_groups):
        w_item = sorted_weight[:, i:i+1] # [LC, 1]
        
        # Mask full packs
        is_full = (pack_counts >= groups_per_pack)
        costs = pack_weights.clone()
        costs[is_full] = inf_val
        
        # Greedy Choice: Pack with min weight (including offset)
        chosen_pack = costs.argmin(dim=1, keepdim=True)
        
        assignments[:, i:i+1] = chosen_pack
        pack_weights.scatter_add_(1, chosen_pack, w_item)
        pack_counts.scatter_add_(1, chosen_pack, torch.ones_like(chosen_pack))

    # --- Prepare Data for Refinement ---
    # Convert assignments to pack_contents [Batch, Packs, K]
    # This structure relies on indices into `sorted_weight`
    _, sort_by_pack_order = assignments.sort(dim=1, stable=True)
    pack_contents = sort_by_pack_order.view(num_problems, num_packs, groups_per_pack)
    
    K = groups_per_pack
    
    # --- 3. Refinement: Destruct-Reconstruct (ABBA Pattern) ---
    # 30 Iterations
    # Select Max and Min packs, pool items, redistribute using ABBA.
    
    arange_2k = torch.arange(2 * K, device=device)
    mask_b = (arange_2k % 4 == 1) | (arange_2k % 4 == 2)
    idx_a = torch.nonzero(~mask_b).squeeze()
    idx_b = torch.nonzero(mask_b).squeeze()
    
    num_iters_abba = 30
    
    for _ in range(num_iters_abba):
        flat_contents = pack_contents.view(num_problems, -1)
        curr_w = torch.gather(sorted_weight, 1, flat_contents).view(num_problems, num_packs, K)
        pack_sums = curr_w.sum(dim=2)
        
        # Identify Max and Min Packs (with tie-break noise)
        noisy_sums = pack_sums + torch.rand_like(pack_sums) * 1e-6
        _, p_max = noisy_sums.max(dim=1)
        _, p_min = noisy_sums.min(dim=1)
        
        # Extract Items
        gather_max = p_max.view(-1, 1, 1).expand(-1, 1, K)
        gather_min = p_min.view(-1, 1, 1).expand(-1, 1, K)
        
        indices_max = torch.gather(pack_contents, 1, gather_max).squeeze(1)
        indices_min = torch.gather(pack_contents, 1, gather_min).squeeze(1)
        
        w_max_items = torch.gather(sorted_weight, 1, indices_max)
        w_min_items = torch.gather(sorted_weight, 1, indices_min)
        
        # Pool and Sort
        merged_indices = torch.cat([indices_max, indices_min], dim=1) # [LC, 2K]
        merged_weights = torch.cat([w_max_items, w_min_items], dim=1)
        
        _, sort_idx = merged_weights.sort(dim=1, descending=True)
        sorted_merged_indices = torch.gather(merged_indices, 1, sort_idx)
        
        # Redistribute
        new_indices_max = sorted_merged_indices[:, idx_a]
        new_indices_min = sorted_merged_indices[:, idx_b]
        
        # Update
        batch_idx = torch.arange(num_problems, device=device)
        pack_contents[batch_idx, p_max, :] = new_indices_max
        pack_contents[batch_idx, p_min, :] = new_indices_min

    # --- 4. Refinement: Max-Any Swap ---
    # 15 Iterations
    # Swap items from Max Pack with ANY other pack to reduce global max.
    
    num_iters_swap = 15
    for _ in range(num_iters_swap):
        flat_contents = pack_contents.view(num_problems, -1)
        curr_w = torch.gather(sorted_weight, 1, flat_contents).view(num_problems, num_packs, K)
        pack_sums = curr_w.sum(dim=2)
        
        val_max, p_max = pack_sums.max(dim=1)
        
        # Max Items
        gather_max = p_max.view(-1, 1, 1).expand(-1, 1, K)
        w_max = torch.gather(curr_w, 1, gather_max) # [LC, 1, K]
        
        # Diffs: [LC, 1, K, 1] - [LC, M, 1, K]
        diffs = w_max.unsqueeze(3) - curr_w.unsqueeze(2)
        
        # New Loads
        new_max_val = val_max.view(-1, 1, 1, 1) - diffs
        new_other_val = pack_sums.view(num_problems, num_packs, 1, 1) + diffs
        
        new_pair_max = torch.max(new_max_val, new_other_val)
        improvement = val_max.view(-1, 1, 1, 1) - new_pair_max
        
        # Validity
        mask_self = (torch.arange(num_packs, device=device).view(1, -1) == p_max.view(-1, 1))
        mask_self = mask_self.view(num_problems, num_packs, 1, 1)
        
        valid = (diffs > 0) & (improvement > 1e-6) & (~mask_self)
        
        scores = torch.where(valid, improvement, torch.tensor(float('-inf'), device=device))
        
        # Find best swap
        flat_scores = scores.view(num_problems, -1)
        best_imp, best_idx = flat_scores.max(dim=1)
        
        if not (best_imp > 0).any():
            break
            
        # Execute
        idx_do = torch.nonzero(best_imp > 0).squeeze(1)
        if len(idx_do) == 0: break
        
        sel_idx = best_idx[idx_do]
        
        # Decode
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

    # --- 5. Final Selection ---
    flat_contents = pack_contents.view(num_problems, -1)
    final_w = torch.gather(sorted_weight, 1, flat_contents).view(num_problems, num_packs, K)
    final_max = final_w.sum(dim=2).max(dim=1).values
    
    final_max = final_max.view(num_layers, num_candidates)
    best_cand = final_max.argmin(dim=1)
    
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