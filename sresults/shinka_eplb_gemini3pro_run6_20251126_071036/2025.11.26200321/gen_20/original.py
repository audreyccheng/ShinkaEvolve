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

    Uses a Vectorized Greedy LPT initialization followed by a Batched Swap-based
    local search refinement on GPU.

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

    # Trivial case
    if groups_per_pack == 1:
        pack_index = torch.arange(num_groups, dtype=torch.int64, device=device).expand(num_layers, num_groups)
        rank_in_pack = torch.zeros_like(pack_index)
        return pack_index, rank_in_pack

    # --- Phase 1: Vectorized LPT Initialization ---
    # Sort items descending [L, N]
    sorted_weight, sorted_indices = weight.sort(dim=-1, descending=True)

    # pack_weights: [L, M]
    pack_weights = torch.zeros(num_layers, num_packs, device=device, dtype=weight.dtype)
    pack_counts = torch.zeros(num_layers, num_packs, device=device, dtype=torch.int64)

    sorted_pack_index = torch.zeros_like(sorted_indices)

    for i in range(num_groups):
        w_item = sorted_weight[:, i:i+1] # [L, 1]

        # Mask full packs by adding infinity to their weights so they aren't chosen
        is_full = (pack_counts >= groups_per_pack)
        masked_w = pack_weights.clone()
        masked_w[is_full] = float('inf')

        # Choose pack with min weight among valid ones
        chosen_pack = masked_w.argmin(dim=1, keepdim=True) # [L, 1]

        # Assign
        sorted_pack_index[:, i:i+1] = chosen_pack

        # Update
        pack_weights.scatter_add_(1, chosen_pack, w_item)
        pack_counts.scatter_add_(1, chosen_pack, torch.ones_like(chosen_pack))

    # --- Phase 2: Vectorized Swap Refinement ---
    num_iters = 50
    for _ in range(num_iters):
        # 1. Find max pack and recompute weights to be safe
        pack_weights.fill_(0)
        pack_weights.scatter_add_(1, sorted_pack_index, sorted_weight)

        max_vals, max_packs = pack_weights.max(dim=1) # [L], [L]

        # 2. Candidate Swaps: item i in max_pack, item j not in max_pack
        # Mask for items in max pack: [L, N]
        in_max = (sorted_pack_index == max_packs.unsqueeze(1))

        # Diff matrix: diff = w_i - w_j. [L, N, N]
        # Memory-efficient calc: we only need diffs where i is in max pack and j is not.
        # But for full vectorization, calculating all pairs is often faster if N is small.
        diffs = sorted_weight.unsqueeze(2) - sorted_weight.unsqueeze(1)

        # Validity mask: i in max, j not in max, diff > 0
        valid_mask = in_max.unsqueeze(2) & (~in_max.unsqueeze(1))
        valid_mask &= (diffs > 0)

        if not valid_mask.any():
            break

        # Get weight of pack containing j
        p_j = sorted_pack_index # [L, N]
        w_packs_j = torch.gather(pack_weights, 1, p_j) # [L, N]
        w_target = w_packs_j.unsqueeze(1) # [L, 1, N]

        # Improvement score = min(diff, M - T - diff)
        # We want to maximize this score.
        M = max_vals.view(-1, 1, 1)
        score = torch.min(diffs, M - w_target - diffs)

        # Apply mask
        score = torch.where(valid_mask, score, torch.tensor(float('-inf'), device=device))

        # Find best swap per layer
        best_score_flat, best_idx_flat = score.view(num_layers, -1).max(dim=1)

        # Filter improvements (epsilon 1e-6)
        do_swap = best_score_flat > 1e-6

        if not do_swap.any():
            break

        # Decode indices
        idx_i = best_idx_flat // num_groups
        idx_j = best_idx_flat % num_groups

        # Update sorted_pack_index for layers that swap
        l_idx = torch.nonzero(do_swap).squeeze(1)

        i_idx = idx_i[l_idx]
        j_idx = idx_j[l_idx]

        p_i = max_packs[l_idx]
        p_j_val = sorted_pack_index[l_idx, j_idx]

        sorted_pack_index[l_idx, i_idx] = p_j_val
        sorted_pack_index[l_idx, j_idx] = p_i

    # --- Construct Output ---
    # Map back to original indices
    pack_index = torch.empty_like(sorted_pack_index)
    pack_index.scatter_(1, sorted_indices, sorted_pack_index)

    # Construct rank_in_pack
    # Sort items by pack (stable sort keeps heavier items first)
    pack_sort_idx = sorted_pack_index.argsort(dim=1, stable=True)

    # Ranks pattern: 0, 1, ..., k-1 repeated M times
    ranks_pattern = torch.arange(groups_per_pack, device=device).repeat(num_packs).expand(num_layers, -1)

    # Map ranks to sorted_pack positions
    # We need to assign ranks based on the position in pack_sort_idx
    # pack_sort_idx[i, k] tells us which item is at position k in the sorted list of packs.
    # The item at position k belongs to the pack pack_sort_idx[i, k] maps to? No.
    # sorted_pack_index[i, pack_sort_idx[i, k]] is the pack ID.
    # Since we sorted by pack ID, the pack IDs appear in order 0,0,0, 1,1,1 ...
    # So ranks_pattern corresponds exactly to the items ordered by pack_sort_idx.
    
    # We want rank_in_pack[layer, item_idx] = rank
    # item_idx = sorted_indices[layer, pack_sort_idx[layer, k]]?
    # No, let's step back.
    # pack_index maps original_item_idx -> pack_id.
    
    # We need rank_in_pack.
    # Let's use the 'pack_sort_idx' which sorts the 'sorted_pack_index' (which corresponds to 'sorted_weight' items).
    # pack_sort_idx[l, 0] is the index (into sorted_weight items) of the first item in pack 0 (or whatever min pack is).
    # Actually, argsort on pack IDs puts items of pack 0 first, then pack 1, etc.
    # Within pack 0, the order is determined by stable sort. Since input was 'sorted_pack_index', the indices are 0..N-1.
    # 0..N-1 corresponds to items sorted by weight descending.
    # So stable sort preserves weight order: heaviest items in pack 0 come first.
    
    # So, for the k-th item in the sorted list (index `idx = pack_sort_idx[l, k]`), 
    # its rank is `ranks_pattern[l, k]`.
    # This item corresponds to `sorted_indices[l, idx]` in the original array.
    
    sorted_ranks = torch.empty_like(ranks_pattern)
    sorted_ranks.scatter_(1, pack_sort_idx, ranks_pattern)
    # Now sorted_ranks[l, i] is the rank of the i-th heaviest item.
    
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
