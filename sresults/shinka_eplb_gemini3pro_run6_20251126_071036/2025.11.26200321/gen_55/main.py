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
import itertools
import math

def balanced_packing(weight: torch.Tensor,
                     num_packs: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pack n weighted objects to m packs, such that each bin contains exactly
    n/m objects and the weights of all packs are as balanced as possible.

    Uses a Parallel Hybrid Initialization (LPT and Interleaved-LPT with Noise)
    followed by a Vectorized Local Search (Max-Any Swap and Block Swap).

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

    # --- Configuration ---
    # 128 candidates: 64 use Standard LPT, 64 use Interleaved LPT
    num_candidates = 128
    half_cand = num_candidates // 2
    num_problems = num_layers * num_candidates

    # --- Phase 1: Expansion and Sorting ---
    # [L*C, N]
    w_expanded = weight.repeat_interleave(num_candidates, dim=0)

    # Noise injection for diversity
    # Scale varies from 0.0 to 0.4 across candidates to mix greedy and random
    scales = torch.linspace(0, 0.4, num_candidates, device=device)
    noise_scale = scales.repeat(num_layers).view(-1, 1)

    noise = torch.rand_like(w_expanded) * w_expanded * noise_scale
    
    # Preserve pure deterministic LPT for the first candidate in each block
    # (Candidate 0 and Candidate 64 for each layer)
    noise.view(num_layers, num_candidates, num_groups)[:, 0, :] = 0
    noise.view(num_layers, num_candidates, num_groups)[:, half_cand, :] = 0

    sort_keys = w_expanded + noise
    _, sorted_indices = sort_keys.sort(dim=-1, descending=True)
    sorted_weight = torch.gather(w_expanded, 1, sorted_indices)
    
    # --- Phase 1b: Interleaved Permutation ---
    # For the second half of candidates, apply Interleaved ordering
    # (Heavy, Light, Heavy, Light...)
    # This forces the greedy packer to pair large items with small items early.
    
    # Create permutation indices [0, N-1, 1, N-2, ...]
    arange = torch.arange(num_groups, device=device)
    interleaved_perm = torch.empty_like(arange)
    half_n = (num_groups + 1) // 2
    interleaved_perm[0::2] = arange[:half_n]
    interleaved_perm[1::2] = arange.flip(0)[:(num_groups - half_n)]
    
    # Apply to second half of each layer's candidates
    # Reshape to address candidates easily
    # sorted_weight: [L, C, N]
    sw_view = sorted_weight.view(num_layers, num_candidates, num_groups)
    si_view = sorted_indices.view(num_layers, num_candidates, num_groups)
    
    # Select second half: [L, C/2, N]
    sw_half = sw_view[:, half_cand:, :]
    si_half = si_view[:, half_cand:, :]
    
    # Permute
    sw_view[:, half_cand:, :] = torch.gather(sw_half, 2, interleaved_perm.view(1, 1, -1).expand(num_layers, half_cand, -1))
    si_view[:, half_cand:, :] = torch.gather(si_half, 2, interleaved_perm.view(1, 1, -1).expand(num_layers, half_cand, -1))
    
    # Flatten back
    sorted_weight = sw_view.view(num_problems, num_groups)
    sorted_indices = si_view.view(num_problems, num_groups)

    # --- Phase 2: Vectorized Greedy Assignment ---
    pack_weights = torch.zeros(num_problems, num_packs, device=device, dtype=weight.dtype)
    pack_counts = torch.zeros(num_problems, num_packs, device=device, dtype=torch.int64)
    sorted_pack_index = torch.zeros_like(sorted_indices)
    
    inf_val = torch.tensor(float('inf'), device=device)

    for i in range(num_groups):
        w_item = sorted_weight[:, i:i+1] # [LC, 1]

        # Mask full packs
        is_full = (pack_counts >= groups_per_pack)
        candidates = torch.where(is_full, inf_val, pack_weights)

        # Choose pack with min weight
        chosen_pack = candidates.argmin(dim=1, keepdim=True)

        sorted_pack_index[:, i:i+1] = chosen_pack
        pack_weights.scatter_add_(1, chosen_pack, w_item)
        pack_counts.scatter_add_(1, chosen_pack, torch.ones_like(chosen_pack))

    # --- Phase 3: Vectorized Local Search ---
    # Construct structured pack contents: [LC, M, K]
    # Stable sort by pack index to group items
    _, pack_sort_idx = sorted_pack_index.sort(dim=1, stable=True)
    pack_contents = pack_sort_idx.view(num_problems, num_packs, groups_per_pack)
    
    K = groups_per_pack

    # -- Stage 3a: 1-Item Swap (Max Pack vs Any Pack) --
    num_iters_1 = 20
    for _ in range(num_iters_1):
        # 1. Weights
        flat_contents = pack_contents.view(num_problems, -1)
        curr_items = torch.gather(sorted_weight, 1, flat_contents).view(num_problems, num_packs, K)
        
        pack_sums = curr_items.sum(dim=2) # [LC, M]
        val_max, idx_max = pack_sums.max(dim=1) # [LC]
        
        # 2. Max Pack Items
        gather_max = idx_max.view(-1, 1, 1).expand(-1, 1, K)
        max_items = torch.gather(curr_items, 1, gather_max).squeeze(1) # [LC, K]
        
        # 3. Diffs (Max Items vs All Other Items)
        # [LC, 1, K, 1] - [LC, M, 1, K] -> [LC, M, K, K]
        diffs = max_items.view(num_problems, 1, K, 1) - curr_items.view(num_problems, num_packs, 1, K)
        
        # 4. Improvement
        val_max_exp = val_max.view(num_problems, 1, 1, 1)
        pack_sums_exp = pack_sums.view(num_problems, num_packs, 1, 1)
        
        # New Pair Max = max(LoadMax - diff, LoadOther + diff)
        new_pair_max = torch.max(val_max_exp - diffs, pack_sums_exp + diffs)
        improvement = val_max_exp - new_pair_max
        
        # 5. Mask
        mask_self = (torch.arange(num_packs, device=device).view(1, -1) == idx_max.view(-1, 1))
        mask_self = mask_self.view(num_problems, num_packs, 1, 1)
        
        valid = (diffs > 0) & (improvement > 1e-6) & (~mask_self)
        scores = torch.where(valid, improvement, torch.tensor(float('-inf'), device=device))
        
        # 6. Best Swap
        flat_scores = scores.view(num_problems, -1)
        best_imp, flat_idx = flat_scores.max(dim=1)
        
        if not (best_imp > float('-inf')).any():
            break
            
        # 7. Apply
        valid_batch = torch.nonzero(best_imp > float('-inf')).squeeze(1)
        sel_idx = flat_idx[valid_batch]
        
        K2 = K * K
        p_other = sel_idx // K2
        rem = sel_idx % K2
        idx_in_max = rem // K
        idx_in_other = rem % K
        
        p_max = idx_max[valid_batch]
        
        # Swap indices in pack_contents
        v_max = pack_contents[valid_batch, p_max, idx_in_max]
        v_other = pack_contents[valid_batch, p_other, idx_in_other]
        
        pack_contents[valid_batch, p_max, idx_in_max] = v_other
        pack_contents[valid_batch, p_other, idx_in_other] = v_max

    # -- Stage 3b: 2-Item Swap (Max Pack vs Any Pack) --
    # Only if K is small to keep complexity managed
    if K >= 2 and K <= 16:
        # Precompute pairs indices
        pairs = torch.tensor(list(itertools.combinations(range(K), 2)), device=device, dtype=torch.long)
        num_pairs = pairs.size(0)
        
        num_iters_2 = 5
        for _ in range(num_iters_2):
            flat_contents = pack_contents.view(num_problems, -1)
            curr_items = torch.gather(sorted_weight, 1, flat_contents).view(num_problems, num_packs, K)
            pack_sums = curr_items.sum(dim=2)
            
            val_max, idx_max = pack_sums.max(dim=1)
            
            gather_max = idx_max.view(-1, 1, 1).expand(-1, 1, K)
            max_items = torch.gather(curr_items, 1, gather_max).squeeze(1)
            
            # Sum pairs: [LC, NP]
            max_pairs = max_items[:, pairs[:, 0]] + max_items[:, pairs[:, 1]]
            # [LC, M, NP]
            curr_pairs = curr_items[:, :, pairs[:, 0]] + curr_items[:, :, pairs[:, 1]]
            
            diffs = max_pairs.view(num_problems, 1, num_pairs, 1) - curr_pairs.view(num_problems, num_packs, 1, num_pairs)
            
            val_max_exp = val_max.view(num_problems, 1, 1, 1)
            pack_sums_exp = pack_sums.view(num_problems, num_packs, 1, 1)
            
            new_pair_max = torch.max(val_max_exp - diffs, pack_sums_exp + diffs)
            improvement = val_max_exp - new_pair_max
            
            mask_self = (torch.arange(num_packs, device=device).view(1, -1) == idx_max.view(-1, 1))
            mask_self = mask_self.view(num_problems, num_packs, 1, 1)
            
            valid = (diffs > 0) & (improvement > 1e-6) & (~mask_self)
            scores = torch.where(valid, improvement, torch.tensor(float('-inf'), device=device))
            
            flat_scores = scores.view(num_problems, -1)
            best_imp, flat_idx = flat_scores.max(dim=1)
            
            if not (best_imp > float('-inf')).any():
                break
                
            valid_batch = torch.nonzero(best_imp > float('-inf')).squeeze(1)
            sel_idx = flat_idx[valid_batch]
            
            NP2 = num_pairs * num_pairs
            p_other = sel_idx // NP2
            rem = sel_idx % NP2
            pidx_max = rem // num_pairs
            pidx_other = rem % num_pairs
            
            p_max = idx_max[valid_batch]
            
            idx_pair_max = pairs[pidx_max]
            idx_pair_other = pairs[pidx_other]
            
            # Swap
            for k in range(2):
                im = idx_pair_max[:, k]
                io = idx_pair_other[:, k]
                vm = pack_contents[valid_batch, p_max, im]
                vo = pack_contents[valid_batch, p_other, io]
                pack_contents[valid_batch, p_max, im] = vo
                pack_contents[valid_batch, p_other, io] = vm

    # --- Phase 4: Selection ---
    flat_contents = pack_contents.view(num_problems, -1)
    final_items = torch.gather(sorted_weight, 1, flat_contents).view(num_problems, num_packs, K)
    final_max_load = final_items.sum(dim=2).max(dim=1).values
    
    final_max_load = final_max_load.view(num_layers, num_candidates)
    best_cand = final_max_load.argmin(dim=1)
    
    best_indices = torch.arange(num_layers, device=device) * num_candidates + best_cand
    best_contents = pack_contents[best_indices] # [L, M, K]
    best_sorted_idx = sorted_indices[best_indices] # [L, N]
    
    flat_best = best_contents.view(num_layers, -1)
    original_idx = torch.gather(best_sorted_idx, 1, flat_best)
    
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