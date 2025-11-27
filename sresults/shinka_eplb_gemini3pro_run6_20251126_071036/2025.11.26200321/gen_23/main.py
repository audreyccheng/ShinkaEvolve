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
import math

def balanced_packing(weight: torch.Tensor,
                     num_packs: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pack n weighted objects to m packs, such that each bin contains exactly
    n/m objects and the weights of all packs are as balanced as possible.

    Uses a Massive Parallel Randomized Greedy LPT initialization followed by
    a Vectorized Swap-based local search refinement.

    Parameters:
        weight: [X, n], the weight of each item
        num_packs: number of packs

    Returns:
        pack_index: [X, n], the pack index of each item
        rank_in_pack: [X, n], the rank of the item in the pack
    """
    num_layers, num_groups = weight.shape
    device = weight.device
    assert num_groups % num_packs == 0
    groups_per_pack = num_groups // num_packs

    if groups_per_pack == 1:
        pack_index = torch.arange(num_groups, dtype=torch.int64, device=device).expand(num_layers, num_groups)
        rank_in_pack = torch.zeros_like(pack_index)
        return pack_index, rank_in_pack

    # Massive Parallelism
    num_candidates = 64
    num_problems = num_layers * num_candidates

    # Expand weights [L*C, N]
    w_expanded = weight.repeat_interleave(num_candidates, dim=0)

    # Noise Injection for Diversity (0.0 to 0.2 range)
    # Candidate 0 of each layer has 0 noise (Pure LPT)
    scales = torch.linspace(0, 0.2, num_candidates, device=device)
    noise_scale = scales.repeat(num_layers).unsqueeze(1)

    noise = torch.rand_like(w_expanded) * w_expanded * noise_scale
    # Enforce pure LPT for the first candidate
    noise.view(num_layers, num_candidates, num_groups)[:, 0, :] = 0

    sort_keys = w_expanded + noise
    _, sorted_indices = sort_keys.sort(dim=-1, descending=True)
    sorted_weight = torch.gather(w_expanded, 1, sorted_indices)

    # --- Phase 1: Parallel Greedy LPT ---
    pack_weights = torch.zeros(num_problems, num_packs, device=device, dtype=weight.dtype)
    pack_counts = torch.zeros(num_problems, num_packs, device=device, dtype=torch.int64)
    sorted_pack_index = torch.zeros_like(sorted_indices)

    # Iterate over items (columns)
    # Vectorized across all batch problems
    for i in range(num_groups):
        w_item = sorted_weight[:, i:i+1] # [LC, 1]

        # Mask full packs
        # We add infinity to weights of full packs so argmin skips them
        is_full = (pack_counts >= groups_per_pack)
        candidates = pack_weights.clone()
        candidates[is_full] = float('inf')

        # Choose pack with min weight
        chosen_pack = candidates.argmin(dim=1, keepdim=True) # [LC, 1]

        # Assign
        sorted_pack_index[:, i:i+1] = chosen_pack

        # Update
        pack_weights.scatter_add_(1, chosen_pack, w_item)
        pack_counts.scatter_add_(1, chosen_pack, torch.ones_like(chosen_pack))

    # --- Phase 2: Vectorized Swap Refinement ---
    num_iters = 20
    for _ in range(num_iters):
        # Re-calculate max/min packs
        # pack_weights is already current from greedy loop, but to be safe/clean in loop:
        # (Optimized: we could maintain pack_weights incrementally, but re-calc avoids drift)
        pack_weights.fill_(0)
        pack_weights.scatter_add_(1, sorted_pack_index, sorted_weight)

        max_val, max_pack = pack_weights.max(dim=1) # [LC]
        min_val, min_pack = pack_weights.min(dim=1) # [LC]

        # Identify items
        # We want to swap item from max pack with item from min pack.
        # Mask for items in max pack: [LC, N]
        mask_max = (sorted_pack_index == max_pack.unsqueeze(1))
        mask_min = (sorted_pack_index == min_pack.unsqueeze(1))

        # diffs = w_i - w_j
        diffs = sorted_weight.unsqueeze(2) - sorted_weight.unsqueeze(1) # [LC, N, N]

        # Valid mask: i in max, j in min
        valid = mask_max.unsqueeze(2) & mask_min.unsqueeze(1)

        # Goal: Minimize max(new_max, new_min)
        # current max is max_val. current min is min_val.
        # new_max = max_val - (w_i - w_j) = max_val - diff
        # new_min = min_val + (w_i - w_j) = min_val + diff

        mv = max_val.view(-1, 1, 1)
        mnv = min_val.view(-1, 1, 1)

        new_max = mv - diffs
        new_min = mnv + diffs
        new_peak = torch.max(new_max, new_min)
        improvement = mv - new_peak

        # Filter valid
        improvement = torch.where(valid, improvement, torch.tensor(float('-inf'), device=device))

        # Find best
        best_imp_flat, best_idx_flat = improvement.view(num_problems, -1).max(dim=1)

        should_swap = best_imp_flat > 1e-6
        if not should_swap.any():
            break

        # Execute Swaps
        # Decode indices: best_idx_flat is i * N + j
        active_batch = torch.nonzero(should_swap).squeeze(1)
        flat_idx = best_idx_flat[active_batch]

        idx_i = flat_idx // num_groups
        idx_j = flat_idx % num_groups

        p_max_active = max_pack[active_batch]
        p_min_active = min_pack[active_batch]

        # Swap pack assignments
        sorted_pack_index[active_batch, idx_i] = p_min_active
        sorted_pack_index[active_batch, idx_j] = p_max_active

    # --- Phase 3: Select Best Candidate ---
    # Calc max loads
    pack_weights.fill_(0)
    pack_weights.scatter_add_(1, sorted_pack_index, sorted_weight)
    final_max, _ = pack_weights.max(dim=1)

    # Reshape [L, C]
    final_max = final_max.view(num_layers, num_candidates)
    best_cand = final_max.argmin(dim=1)

    # Gather best
    batch_indices = torch.arange(num_layers, device=device) * num_candidates + best_cand

    best_sorted_pack_index = sorted_pack_index[batch_indices] # [L, N]
    best_sorted_indices = sorted_indices[batch_indices]       # [L, N]

    # Map back to original
    pack_index = torch.empty_like(best_sorted_pack_index)
    pack_index.scatter_(1, best_sorted_indices, best_sorted_pack_index)

    # Compute rank_in_pack
    # Sort items by pack ID to group them
    _, pack_sort_idx = best_sorted_pack_index.sort(dim=1, stable=True)

    # Create ranks pattern [0, 1, ... k-1, 0, 1... k-1]
    ranks_pattern = torch.arange(groups_per_pack, device=device).repeat(num_packs).expand(num_layers, -1)

    # Scatter ranks to the position of the items
    sorted_ranks = torch.empty_like(ranks_pattern)
    sorted_ranks.scatter_(1, pack_sort_idx, ranks_pattern)

    # Now sorted_ranks[i] corresponds to item sorted_indices[i]
    rank_in_pack = torch.empty_like(pack_index)
    rank_in_pack.scatter_(1, best_sorted_indices, sorted_ranks)

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
    # This loop runs K times where K is number of redundant slots.
    # For common MoE configs, K is comparable to num_log.
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
    weight = weight.float().cpu()

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