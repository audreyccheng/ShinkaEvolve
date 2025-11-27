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

    Uses a Hybrid Parallel Initialization (LPT, Random, Folded-LPT) followed by
    a Two-Phase Local Search (L2 Smoothing + Min-Max Peak Shaving).

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

    # --- Initialization Strategy ---
    # 128 Parallel Candidates per Layer
    # 0-63:   Perturbed LPT (Greedy)
    # 64-95:  Randomized
    # 96-127: Folded LPT (Pairs largest with smallest)
    num_candidates = 128
    idx_lpt_end = 64
    idx_rnd_end = 96
    
    num_total_problems = num_layers * num_candidates
    
    # Expand Weights
    w_expanded = weight.repeat_interleave(num_candidates, dim=0) # [LC, N]
    
    # 1. Generate Sort Keys
    # LPT: Low noise (0.0 to 0.15)
    noise_lpt = torch.rand(num_layers * idx_lpt_end, num_groups, device=device) * (w_expanded[:num_layers*idx_lpt_end] * 0.15)
    # Force pure LPT for first candidate in block for baseline guarantee
    noise_lpt.view(num_layers, idx_lpt_end, num_groups)[:, 0, :] = 0
    
    # Rnd: High noise (random sort)
    noise_rnd = torch.rand(num_layers * (idx_rnd_end - idx_lpt_end), num_groups, device=device) * 1e6
    
    # Folded: Pure weights (no noise) -> Deterministic LPT sort initially
    noise_fold = torch.zeros(num_layers * (num_candidates - idx_rnd_end), num_groups, device=device)
    
    # Combine
    noise = torch.cat([noise_lpt, noise_rnd, noise_fold], dim=0)
    sort_keys = w_expanded + noise
    
    # 2. Sort
    _, sorted_indices = sort_keys.sort(dim=-1, descending=True)
    sorted_weight = torch.gather(w_expanded, 1, sorted_indices)
    
    # 3. Apply "Folded" Permutation to the last group
    # Folded pattern: 0, N-1, 1, N-2, 2, ...
    # This rearranges the sorted items to pair heavy/light items sequentially
    start_fold = num_layers * idx_rnd_end
    if start_fold < num_total_problems:
        # Precompute fold indices
        fold_pattern = torch.empty(num_groups, dtype=torch.long, device=device)
        half = (num_groups + 1) // 2
        fold_pattern[0::2] = torch.arange(half, device=device)
        fold_pattern[1::2] = torch.arange(num_groups - 1, half - 1, -1, device=device)
        
        # Apply to sorted_weight and sorted_indices for the folded section
        sorted_weight[start_fold:] = sorted_weight[start_fold:][:, fold_pattern]
        sorted_indices[start_fold:] = sorted_indices[start_fold:][:, fold_pattern]

    # 4. Vectorized Greedy Packing
    pack_weights = torch.zeros(num_total_problems, num_packs, device=device, dtype=weight.dtype)
    pack_counts = torch.zeros(num_total_problems, num_packs, device=device, dtype=torch.int64)
    sorted_pack_index = torch.zeros_like(sorted_indices)
    
    # Standard Greedy loop
    for i in range(num_groups):
        w_item = sorted_weight[:, i:i+1] # [LC, 1]
        
        # Mask full packs
        is_full = (pack_counts >= groups_per_pack)
        candidates = pack_weights.clone()
        candidates[is_full] = float('inf')
        
        chosen_pack = candidates.argmin(dim=1, keepdim=True)
        
        sorted_pack_index[:, i:i+1] = chosen_pack
        pack_weights.scatter_add_(1, chosen_pack, w_item)
        pack_counts.scatter_add_(1, chosen_pack, torch.ones_like(chosen_pack))

    # --- Phase 2: Hybrid Local Search ---
    # Construct structured representation
    # Sort items by assigned pack to group them
    _, pack_sort_idx = sorted_pack_index.sort(dim=1, stable=True)
    pack_contents = pack_sort_idx.view(num_total_problems, num_packs, groups_per_pack)
    K = groups_per_pack
    
    # Define optimization kernel
    def run_swap_pass(iterations, objective_mode):
        """
        objective_mode='l2_maxmin': Min(SumSquares) via Max-Min Swap
        objective_mode='max_any': Min(MaxLoad) via Max-Any Swap
        """
        for _ in range(iterations):
            # Recompute Weights
            flat_contents = pack_contents.view(num_total_problems, -1)
            curr_w = torch.gather(sorted_weight, 1, flat_contents).view(num_total_problems, num_packs, K)
            pack_sums = curr_w.sum(dim=2) # [LC, M]
            
            val_max, idx_max = pack_sums.max(dim=1) # [LC]
            val_min, idx_min = pack_sums.min(dim=1) # [LC]
            
            # Gather Max Items: [LC, 1, K]
            gather_max = idx_max.view(-1, 1, 1).expand(-1, 1, K)
            w_max_items = torch.gather(curr_w, 1, gather_max) 
            
            if objective_mode == 'l2_maxmin':
                # Target: Min Pack only
                # Gather Min Items: [LC, 1, K]
                gather_min = idx_min.view(-1, 1, 1).expand(-1, 1, K)
                w_target_items = torch.gather(curr_w, 1, gather_min) # [LC, 1, K]
                
                # Expand for broadcasting [LC, 1, K(max), K(target)]
                w_src = w_max_items.view(num_total_problems, 1, K, 1)
                w_dst = w_target_items.view(num_total_problems, 1, 1, K)
                
            else: # max_any
                # Target: All Packs
                # w_target_items: [LC, M, K] -> [LC, M, 1, K]
                w_target_items = curr_w
                target_sums = pack_sums.view(num_total_problems, num_packs, 1, 1)
                
                w_src = w_max_items.view(num_total_problems, 1, K, 1)
                w_dst = w_target_items.view(num_total_problems, num_packs, 1, K)
            
            # Compute Diffs: src - dst
            diffs = w_src - w_dst 
            
            # Calculate Improvement
            if objective_mode == 'l2_maxmin':
                # Minimize Sum Squares: DeltaCost ~ 2 * diff * (SumMin - SumMax + diff)
                # Improvement = -DeltaCost
                sum_diff = (val_min - val_max).view(-1, 1, 1, 1)
                change = 2 * diffs * (sum_diff + diffs)
                improvement = -change
                
                valid_mask = (improvement > 1e-6)
                if num_packs > 1:
                    mask_self = (idx_max == idx_min).view(-1, 1, 1, 1)
                    valid_mask = valid_mask & (~mask_self)
                    
            else: # max_any
                # Minimize Max Load
                current_max = val_max.view(-1, 1, 1, 1)
                new_pair_max = torch.max(current_max - diffs, target_sums + diffs)
                improvement = current_max - new_pair_max
                
                # Validity: strictly move weight out of max (diff > 0)
                mask_self = (torch.arange(num_packs, device=device).view(1, -1) == idx_max.view(-1, 1))
                mask_self = mask_self.view(num_total_problems, num_packs, 1, 1)
                valid_mask = (diffs > 0) & (improvement > 1e-6) & (~mask_self)
            
            # Select Best
            scores = torch.where(valid_mask, improvement, torch.tensor(float('-inf'), device=device))
            flat_scores = scores.view(num_total_problems, -1)
            best_imp, flat_idx = flat_scores.max(dim=1)
            
            if not (best_imp > 0).any():
                return
                
            # Execute Swaps
            valid = torch.nonzero(best_imp > 0).squeeze(1)
            f_idx = flat_idx[valid]
            
            # Decode indices
            K2 = K * K
            p_src = idx_max[valid]
            
            if objective_mode == 'l2_maxmin':
                p_dst = idx_min[valid]
                idx_src = f_idx // K
                idx_dst = f_idx % K
            else:
                p_dst = f_idx // K2
                rem = f_idx % K2
                idx_src = rem // K
                idx_dst = rem % K
            
            # Perform Swap in pack_contents
            v_src = pack_contents[valid, p_src, idx_src]
            v_dst = pack_contents[valid, p_dst, idx_dst]
            
            pack_contents[valid, p_src, idx_src] = v_dst
            pack_contents[valid, p_dst, idx_dst] = v_src

    # Schedule: 
    # 1. 10 iterations of L2 Smoothing (Centers the distribution, avoiding local minima)
    # 2. 10 iterations of Min-Max (Strictly shaves peaks)
    run_swap_pass(10, 'l2_maxmin')
    run_swap_pass(10, 'max_any')
    
    # --- Final Selection ---
    flat_contents = pack_contents.view(num_total_problems, -1)
    final_items = torch.gather(sorted_weight, 1, flat_contents).view(num_total_problems, num_packs, K)
    # Use Max Load for final selection
    final_max_loads = final_items.sum(dim=2).max(dim=1).values # [LC]
    
    reshaped_loads = final_max_loads.view(num_layers, num_candidates)
    best_cand = reshaped_loads.argmin(dim=1)
    
    best_indices = torch.arange(num_layers, device=device) * num_candidates + best_cand
    
    best_contents = pack_contents[best_indices]
    best_sorted_indices = sorted_indices[best_indices]
    
    flat_best = best_contents.view(num_layers, -1)
    original_idx = torch.gather(best_sorted_indices, 1, flat_best)
    
    # Output construction
    pack_ids = torch.arange(num_packs, device=device).view(1, num_packs, 1).expand(num_layers, -1, groups_per_pack).reshape(num_layers, -1)
    rank_ids = torch.arange(groups_per_pack, device=device).view(1, 1, groups_per_pack).expand(num_layers, num_packs, -1).reshape(num_layers, -1)
    
    pack_index = torch.empty_like(pack_ids)
    rank_in_pack = torch.empty_like(rank_ids)
    
    pack_index.scatter_(1, original_idx, pack_ids)
    rank_in_pack.scatter_(1, original_idx, rank_ids)
    
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
        # Find which expert has the max load per replica
        # metric = weight / count
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
    # Sum weights within each group
    tokens_per_group = weight.unflatten(-1, (num_groups, group_size)).sum(-1)

    # Use improved packing
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
    # Each physical expert has weight approx (total_weight / num_replicas)
    tokens_per_phy = (tokens_per_mlog / mlogcnt).gather(-1, phy2mlog)

    # Use improved packing
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
    weight = weight.float()

    if num_groups % num_nodes == 0:
        # use hierarchical load-balance policy
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, num_groups, num_nodes, num_gpus)
    else:
        # use global load-balance policy
        # Treating as if 1 huge group per layer, so packing step 1 is trivial
        # But here logic passes num_groups=1, so group_size=all experts.
        # Step 1 packs 1 item to 1 node? No, step 1 uses num_nodes=1.
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

    # Create the reverse map
    # phy2log * maxlogcnt + phyrank gives a unique index for (expert, replica_id)
    # We scatter the physical index (0..num_replicas) into this location
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