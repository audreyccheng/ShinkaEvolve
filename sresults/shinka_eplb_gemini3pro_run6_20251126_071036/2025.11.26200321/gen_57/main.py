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

    Uses a Massive Parallel Randomized Greedy LPT initialization followed by a
    Vectorized Max-Any Swap local search refinement on GPU.

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
    num_candidates = 64
    num_iters = 20
    batch_size = num_layers * num_candidates

    # --- Phase 1: Massive Parallel Initialization ---
    # Expand problem: [L*C, N]
    w_expanded = weight.repeat_interleave(num_candidates, dim=0)

    # Generate noise for Randomized LPT
    scales = torch.linspace(0, 0.1, num_candidates, device=device)
    noise_scale = scales.repeat(num_layers).unsqueeze(1)
    noise = torch.rand_like(w_expanded) * w_expanded * noise_scale

    # Calculate sort keys
    sort_keys = w_expanded + noise
    _, sorted_indices = sort_keys.sort(dim=-1, descending=True)

    # Interleaved Strategy (Heavy-Light-Heavy...) for subset of candidates
    # Apply to last 16 candidates
    if num_candidates >= 16:
        # Create interleaved permutation: [0, N-1, 1, N-2, ...]
        left = torch.arange((num_groups + 1) // 2, device=device)
        right = torch.arange(num_groups - 1, (num_groups + 1) // 2 - 1, -1, device=device)
        interleaved_perm = torch.empty(num_groups, dtype=torch.long, device=device)
        interleaved_perm[0::2] = left
        interleaved_perm[1::2] = right[:len(interleaved_perm[1::2])]

        mask = torch.zeros(num_candidates, dtype=torch.bool, device=device)
        mask[-16:] = True
        mask_batch = mask.repeat(num_layers)
        sorted_indices[mask_batch] = sorted_indices[mask_batch][:, interleaved_perm]

    # Gather weights in sorted order
    sorted_w = torch.gather(w_expanded, 1, sorted_indices)

    # Vectorized Greedy Assignment
    # We assign items one by one to the least loaded pack
    pack_weights = torch.zeros(batch_size, num_packs, device=device, dtype=weight.dtype)
    pack_counts = torch.zeros(batch_size, num_packs, device=device, dtype=torch.int64)
    assigned_pack = torch.zeros_like(sorted_indices)

    for i in range(num_groups):
        w_item = sorted_w[:, i:i+1] # [LC, 1]

        # Mask full packs
        is_full = (pack_counts >= groups_per_pack)
        candidates = pack_weights.clone()
        candidates[is_full] = float('inf')

        # Choose min weight pack
        chosen_pack = candidates.argmin(dim=1, keepdim=True)

        assigned_pack[:, i:i+1] = chosen_pack
        pack_weights.scatter_add_(1, chosen_pack, w_item)
        pack_counts.scatter_add_(1, chosen_pack, torch.ones_like(chosen_pack))

    # --- Phase 2: Vectorized Max-Any Swap Refinement ---
    # Construct pack_contents for swapping
    # Sort assigned_pack to group items by pack: 0,0,0, 1,1,1...
    _, pack_order = assigned_pack.sort(dim=1, stable=True)

    # pack_contents_idx points to the item index in 'sorted_indices'
    pack_contents_idx = pack_order.view(batch_size, num_packs, groups_per_pack)

    # Current item weights in packs: [Batch, M, K]
    curr_items = torch.gather(sorted_w, 1, pack_contents_idx.view(batch_size, -1)).view(batch_size, num_packs, groups_per_pack)

    for _ in range(num_iters):
        # 1. Identify Max Pack
        pack_sums = curr_items.sum(dim=2)
        val_max, idx_max = pack_sums.max(dim=1)

        # 2. Gather items from Max Pack: [Batch, K]
        max_items = torch.gather(curr_items, 1, idx_max.view(-1, 1, 1).expand(-1, 1, groups_per_pack)).squeeze(1)

        # 3. Compute Swap Improvement vs All Other Packs
        # Swap item u (max pack) with v (other pack)
        # diff = w_u - w_v
        # Valid if: val_other + diff < val_max  AND diff > 0

        w_u = max_items.unsqueeze(1).unsqueeze(3) # [B, 1, K, 1]
        w_v = curr_items.unsqueeze(2)             # [B, M, 1, K]
        diffs = w_u - w_v                         # [B, M, K, K]

        # Improvement = val_max - max(new_max, new_other)
        #             = val_max - max(val_max - diff, val_other + diff)

        val_max_exp = val_max.view(-1, 1, 1, 1)
        val_other_exp = pack_sums.view(batch_size, num_packs, 1, 1)

        new_pair_max = torch.max(val_max_exp - diffs, val_other_exp + diffs)
        improvement = val_max_exp - new_pair_max

        # Masks
        mask_self = (torch.arange(num_packs, device=device).view(1, -1, 1, 1) == idx_max.view(-1, 1, 1, 1))
        valid_mask = (~mask_self) & (improvement > 1e-6)

        scores = torch.where(valid_mask, improvement, torch.tensor(float('-inf'), device=device))

        # Best swap
        best_imp, flat_idx = scores.view(batch_size, -1).max(dim=1)

        should_swap = best_imp > 1e-6
        if not should_swap.any():
            break

        # Execute Swap
        active = torch.nonzero(should_swap).squeeze(1)

        sel_flat = flat_idx[active]
        sel_idx_max = idx_max[active]

        K = groups_per_pack
        K2 = K * K
        sel_pack_other = sel_flat // K2
        rem = sel_flat % K2
        sel_k_u = rem // K
        sel_k_v = rem % K

        # Swap in curr_items
        val_u = curr_items[active, sel_idx_max, sel_k_u].clone()
        val_v = curr_items[active, sel_pack_other, sel_k_v].clone()
        curr_items[active, sel_idx_max, sel_k_u] = val_v
        curr_items[active, sel_pack_other, sel_k_v] = val_u

        # Swap in pack_contents_idx
        idx_u = pack_contents_idx[active, sel_idx_max, sel_k_u].clone()
        idx_v = pack_contents_idx[active, sel_pack_other, sel_k_v].clone()
        pack_contents_idx[active, sel_idx_max, sel_k_u] = idx_v
        pack_contents_idx[active, sel_pack_other, sel_k_v] = idx_u

    # --- Phase 3: Selection ---
    final_max = curr_items.sum(dim=2).max(dim=1).values
    final_max = final_max.view(num_layers, num_candidates)
    best_cand = final_max.argmin(dim=1)

    best_indices = torch.arange(num_layers, device=device) * num_candidates + best_cand

    # Retrieve best permutation
    best_contents_idx = pack_contents_idx[best_indices] # [L, M, K]
    best_sorted_indices = sorted_indices[best_indices]  # [L, N]

    # Map back to original indices
    # best_contents_idx[l, m, k] is index into best_sorted_indices
    flat_ptr = best_contents_idx.view(num_layers, -1)
    original_idx = torch.gather(best_sorted_indices, 1, flat_ptr)

    # Generate Output Tensors
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
