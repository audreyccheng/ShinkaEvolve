# EVOLVE-BLOCK-START
"""
Expert parallelism load balancer (EPLB) for vLLM.

This module implements the core rearrangement algorithm using
Chunked Sorted Greedy packing and Binary Search based replication.
"""

import torch


def _refine_packing(weight: torch.Tensor,
                    pack_index: torch.Tensor,
                    rank_in_pack: torch.Tensor,
                    num_packs: int,
                    num_iters: int = 10) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Iterative swap refinement (Max-Any) to minimize load imbalance.
    Attempts swaps between the max-load pack and *any* other pack to reduce max load.
    Vectorized over batch dimension.
    """
    batch_size, num_items = weight.shape
    device = weight.device

    # Compute initial loads
    pack_loads = torch.zeros(batch_size, num_packs, device=device, dtype=weight.dtype)
    pack_loads.scatter_add_(1, pack_index, weight)

    batch_range = torch.arange(batch_size, device=device)

    # Precompute w_diff: w[b, i] - w[b, j]
    # [B, N, 1] - [B, 1, N] = [B, N, N]
    w_diff = weight.unsqueeze(2) - weight.unsqueeze(1)

    for _ in range(num_iters):
        max_load, max_idx = pack_loads.max(dim=1)

        # Logic: Find pair (i, j) where i in MaxPack, j in OtherPack
        # Such that swapping i and j reduces MaxLoad without creating a new peak higher than old MaxLoad.
        # Target Gain = L_max - max(L_max - (w_i - w_j), L_other + (w_i - w_j))
        #             = min(w_i - w_j, (L_max - L_other) - (w_i - w_j))
        # Let delta = w_i - w_j. Let gap = L_max - L_other.
        # Gain = min(delta, gap - delta)
        # We need delta > 0 and gain > 0.

        # Masks
        # is_max[b, i]
        is_max = (pack_index == max_idx.unsqueeze(1))

        # Get pack index and load for every item j
        pack_of_j = pack_index  # [B, N]
        load_of_j = pack_loads.gather(1, pack_of_j) # [B, N]

        # gap[b, j] = max_load - load(pack_of_j)
        gap = max_load.unsqueeze(1) - load_of_j

        # delta[b, i, j] = w[i] - w[j]
        delta = w_diff

        # valid_ij: i in Max, j NOT in Max
        # [B, N, 1] & [B, 1, N]
        valid_ij = is_max.unsqueeze(2) & (~is_max.unsqueeze(1))

        # Gain calculation [B, N, N]
        # gap is [B, 1, N]
        gap_expanded = gap.unsqueeze(1)
        gain = torch.min(delta, gap_expanded - delta)

        # Apply validity mask
        # We need delta > 0 for reduction of max pack
        # We need gain > epsilon
        mask = valid_ij & (delta > 1e-6) & (gain > 1e-6)

        # Find max gain
        gain = torch.where(mask, gain, torch.tensor(-1.0, device=device))

        flat_gain = gain.view(batch_size, -1)
        best_gain_val, best_flat_idx = flat_gain.max(dim=1)

        # Check improvement
        improve_mask = best_gain_val > 1e-6
        if not improve_mask.any():
            break

        # Apply swaps
        active_batch = batch_range[improve_mask]
        active_flat = best_flat_idx[improve_mask]

        idx_i = active_flat // num_items
        idx_j = active_flat % num_items

        p_max = max_idx[active_batch]
        p_other = pack_index[active_batch, idx_j]

        w_i = weight[active_batch, idx_i]
        w_j = weight[active_batch, idx_j]
        d = w_i - w_j

        # Update loads
        # Note: p_max and p_other are distinct by definition of valid_ij
        pack_loads[active_batch, p_max] -= d
        pack_loads[active_batch, p_other] += d

        # Update indices
        pack_index[active_batch, idx_i] = p_other
        pack_index[active_batch, idx_j] = p_max

        # Update ranks
        r_i = rank_in_pack[active_batch, idx_i]
        r_j = rank_in_pack[active_batch, idx_j]
        rank_in_pack[active_batch, idx_i] = r_j
        rank_in_pack[active_batch, idx_j] = r_i

    return pack_index, rank_in_pack


def balanced_packing(weight: torch.Tensor,
                     num_packs: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pack n weighted objects to m packs using a Massive Parallel Ensemble strategy.

    Includes Parallel Top-K Refinement.
    """
    num_layers, num_items = weight.shape
    device = weight.device
    num_candidates = 128
    capacity = num_items // num_packs

    # 1. Generate Candidates
    lpt_val, lpt_idx = weight.sort(dim=-1, descending=True)

    # ZigZag
    zigzag_perm = torch.empty(num_items, device=device, dtype=torch.long)
    half = (num_items + 1) // 2
    arange = torch.arange(num_items, device=device)
    zigzag_perm[0::2] = arange[:half]
    zigzag_perm[1::2] = arange[half:].flip(0)

    # Candidate Indices
    c0_idx = lpt_idx.unsqueeze(1)
    c1_idx = lpt_idx.gather(1, zigzag_perm.view(1, -1).expand(num_layers, -1)).unsqueeze(1)

    # Random Shuffles
    num_random = 32
    rand_perm = torch.rand(num_layers, num_random, num_items, device=device).argsort(dim=-1)

    # Noisy LPT
    num_noisy = num_candidates - 2 - num_random
    noise = torch.rand(num_layers, num_noisy, num_items, device=device) * 0.4 + 0.8
    noisy_weights = weight.unsqueeze(1) * noise
    _, c_noisy_idx = noisy_weights.sort(dim=-1, descending=True)

    all_indices = torch.cat([c0_idx, c1_idx, rand_perm, c_noisy_idx], dim=1)

    # Gather weights
    expanded_weight = weight.unsqueeze(1).expand(-1, num_candidates, -1)
    ordered_weights = expanded_weight.gather(2, all_indices)

    # Flatten
    batch_size = num_layers * num_candidates
    flat_weights = ordered_weights.view(batch_size, num_items)

    # 2. Vectorized Greedy Packing
    pack_loads = torch.zeros(batch_size, num_packs, device=device, dtype=weight.dtype)
    pack_counts = torch.zeros(batch_size, num_packs, device=device, dtype=torch.int64)
    flat_assigned_packs = torch.empty(batch_size, num_items, device=device, dtype=torch.int64)
    flat_assigned_ranks = torch.empty(batch_size, num_items, device=device, dtype=torch.int64)

    batch_range = torch.arange(batch_size, device=device)
    inf = torch.tensor(float('inf'), device=device, dtype=weight.dtype)

    for i in range(num_items):
        w = flat_weights[:, i]
        valid_mask = pack_counts < capacity
        temp_loads = torch.where(valid_mask, pack_loads, inf)
        chosen_pack = temp_loads.argmin(dim=1)

        flat_assigned_packs[:, i] = chosen_pack
        flat_assigned_ranks[:, i] = pack_counts[batch_range, chosen_pack]

        pack_loads[batch_range, chosen_pack] += w
        pack_counts[batch_range, chosen_pack] += 1

    # 3. Top-K Selection
    loads = pack_loads.view(num_layers, num_candidates, num_packs)
    imbalance = loads.max(dim=-1).values - loads.min(dim=-1).values

    k = 8
    _, best_k_idx = imbalance.topk(k, dim=1, largest=False) # [L, K]

    # Flatten selection
    layer_offsets = (torch.arange(num_layers, device=device) * num_candidates).unsqueeze(1)
    flat_selected_indices = (best_k_idx + layer_offsets).flatten() # [L*K]

    refined_weights = flat_weights[flat_selected_indices]
    refined_packs = flat_assigned_packs[flat_selected_indices]
    refined_ranks = flat_assigned_ranks[flat_selected_indices]

    # 4. Refinement on Top-K
    refined_packs, refined_ranks = _refine_packing(refined_weights, refined_packs, refined_ranks, num_packs, num_iters=10)

    # 5. Final Selection
    refined_loads = torch.zeros(num_layers * k, num_packs, device=device, dtype=weight.dtype)
    refined_loads.scatter_add_(1, refined_packs, refined_weights)

    refined_imbalance = refined_loads.max(dim=1).values - refined_loads.min(dim=1).values
    refined_imbalance = refined_imbalance.view(num_layers, k)

    best_in_k = refined_imbalance.argmin(dim=1) # [L]

    # 6. Scatter Back
    winner_cand_idx = best_k_idx.gather(1, best_in_k.unsqueeze(1)).squeeze(1) # [L]

    winner_flat_idx = (torch.arange(num_layers, device=device) * k) + best_in_k
    final_packs_aligned = refined_packs[winner_flat_idx]
    final_ranks_aligned = refined_ranks[winner_flat_idx]

    idx_view = winner_cand_idx.view(num_layers, 1, 1).expand(-1, 1, num_items)
    final_perm = all_indices.gather(1, idx_view).squeeze(1)

    pack_index = torch.empty(num_layers, num_items, device=device, dtype=torch.int64)
    rank_in_pack = torch.empty(num_layers, num_items, device=device, dtype=torch.int64)

    pack_index.scatter_(1, final_perm, final_packs_aligned)
    rank_in_pack.scatter_(1, final_perm, final_ranks_aligned)

    return pack_index, rank_in_pack


def replicate_experts(
        weight: torch.Tensor,
        num_phy: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Replicate experts using Binary Search on Max Load followed by Greedy Refinement.

    Finds a capacity threshold T such that sum(ceil(w/T)) is close to num_phy,
    then adjusts counts to exactly num_phy minimizing the max load density.
    """
    num_layers, num_log = weight.shape
    device = weight.device

    # Trivial case
    if num_phy == num_log:
        phy2log = torch.arange(num_log, device=device).expand(num_layers, -1)
        rank = torch.zeros(num_layers, num_phy, dtype=torch.int64, device=device)
        logcnt = torch.ones(num_layers, num_log, dtype=torch.int64, device=device)
        return phy2log, rank, logcnt

    # Binary Search for optimal max load threshold
    # Lower bound: average load per physical slot
    low = weight.sum(dim=-1, keepdim=True) / num_phy
    # Upper bound: max weight (since min replica count is 1)
    high = weight.max(dim=-1, keepdim=True).values

    # Ensure low > 0
    low = torch.max(low, torch.tensor(1e-6, device=device))

    # 15 iterations provide sufficient precision for integer allocation
    for _ in range(15):
        mid = (low + high) * 0.5
        # Calculate required replicas for this threshold
        counts = torch.ceil(weight / mid)
        total = counts.sum(dim=-1, keepdim=True)

        # If we fit within num_phy, try to lower the threshold (tighten)
        mask = total <= num_phy
        high = torch.where(mask, mid, high)
        low = torch.where(mask, low, mid)

    # Initial counts using the feasible threshold
    logcnt = torch.ceil(weight / high).long().clamp(min=1)

    # Correct sum to equal num_phy
    current_sum = logcnt.sum(dim=-1)
    diff = num_phy - current_sum

    # Handle under-allocation (sum < num_phy): Add to experts with highest load density
    max_diff = int(diff.max().item())
    if max_diff > 0:
        rows = torch.arange(num_layers, device=device)
        for _ in range(max_diff):
            active = current_sum < num_phy
            if not active.any(): break

            # Density = weight / count
            density = weight / logcnt.float()

            # Pick expert with max density
            target_idx = density.argmax(dim=-1)

            # Update active rows
            mask_indices = rows[active]
            mask_targets = target_idx[active]

            logcnt.index_put_((mask_indices, mask_targets),
                              torch.tensor(1, device=device, dtype=torch.int64),
                              accumulate=True)
            current_sum[active] += 1

    # Handle over-allocation (sum > num_phy): Remove from experts with lowest cost
    # Cost = New Load = weight / (count - 1). We want minimum new load.
    min_diff = int(diff.min().item())
    if min_diff < 0:
        rows = torch.arange(num_layers, device=device)
        for _ in range(abs(min_diff)):
            active = current_sum > num_phy
            if not active.any(): break

            # Only consider experts with > 1 replica
            valid = logcnt > 1
            cost = weight / (logcnt - 1).float()
            cost[~valid] = float('inf')

            target_idx = cost.argmin(dim=-1)

            mask_indices = rows[active]
            mask_targets = target_idx[active]

            logcnt.index_put_((mask_indices, mask_targets),
                              torch.tensor(-1, device=device, dtype=torch.int64),
                              accumulate=True)
            current_sum[active] -= 1

    # Construct physical to logical map
    flat_log_ids = torch.arange(num_log, device=device).repeat(num_layers)
    flat_counts = logcnt.flatten()

    flat_phy2log = torch.repeat_interleave(flat_log_ids, flat_counts)

    # Reshape
    # If sizes mismatch due to weird edge cases, we safeguard, though logic implies exact match
    target_size = num_layers * num_phy
    if flat_phy2log.numel() != target_size:
        if flat_phy2log.numel() < target_size:
            flat_phy2log = torch.cat([flat_phy2log, torch.zeros(target_size - flat_phy2log.numel(), device=device, dtype=torch.long)])
        else:
            flat_phy2log = flat_phy2log[:target_size]

    phy2log = flat_phy2log.view(num_layers, num_phy)

    # Calculate ranks
    # Rank is the 0-based index of a replica for a specific logical expert
    offsets = torch.zeros_like(logcnt)
    offsets[:, 1:] = logcnt[:, :-1].cumsum(dim=1)

    mapped_offsets = offsets.gather(1, phy2log)
    phy_indices = torch.arange(num_phy, device=device).expand(num_layers, -1)

    rank = phy_indices - mapped_offsets

    return phy2log, rank, logcnt


def rebalance_experts_hierarchical(
    weight: torch.Tensor,
    num_physical_experts: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
):
    """
    Hierarchical rebalancing using the optimized packing and replication strategies.
    """
    num_layers, num_logical_experts = weight.shape
    group_size = num_logical_experts // num_groups
    groups_per_node = num_groups // num_nodes
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

    # Step 1: Pack groups to nodes
    tokens_per_group = weight.unflatten(-1, (num_groups, group_size)).sum(-1)
    group_pack_index, group_rank_in_pack = balanced_packing(
        tokens_per_group, num_nodes)

    log2mlog = (((group_pack_index * groups_per_node + group_rank_in_pack) *
                 group_size).unsqueeze(-1) +
                torch.arange(group_size,
                             dtype=torch.int64,
                             device=group_pack_index.device)).flatten(-2)
    mlog2log = inverse(log2mlog)

    # Step 2: Replicate experts within nodes
    tokens_per_mlog = weight.gather(-1, mlog2log).view(
        -1, num_logical_experts // num_nodes)
    phy2mlog, phyrank, mlogcnt = replicate_experts(
        tokens_per_mlog, num_physical_experts // num_nodes)

    # Step 3: Pack physical experts to GPUs
    # Load per replica is approximated as total_weight / count
    tokens_per_phy = (tokens_per_mlog / mlogcnt).gather(-1, phy2mlog)
    pack_index, rank_in_pack = balanced_packing(tokens_per_phy,
                                                num_gpus // num_nodes)

    phy2pphy = pack_index * phy_experts_per_gpu + rank_in_pack
    pphy2phy = inverse(phy2pphy)

    # Map back to original logical IDs
    pphy2mlog = phy2mlog.gather(-1, pphy2phy)

    node_offsets = torch.arange(
        0,
        num_logical_experts,
        num_logical_experts // num_nodes,
        device=weight.device,
    ).view(1, -1, 1)

    pphy2mlog_restored = (pphy2mlog.view(num_layers, num_nodes, -1) + node_offsets).flatten(-2)

    pphy2log = mlog2log.gather(-1, pphy2mlog_restored)
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
    # Ensure CPU computation for algorithmic stability and memory
    weight = weight.float().cpu()

    if num_groups % num_nodes == 0:
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, num_groups, num_nodes, num_gpus)
    else:
        # Fallback to global policy if groups not divisible by nodes
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, 1, 1, num_gpus)

    # Construct log2phy map [layers, logical_experts, max_replicas]
    max_replicas = int(logcnt.max().item())

    log2phy = torch.full(
        (num_layers, num_logical_experts, max_replicas),
        -1,
        dtype=torch.int64,
        device=logcnt.device,
    )

    # Optimized scatter for log2phy
    # Flat indices calculation
    flat_layer_idx = torch.arange(num_layers, device=logcnt.device).unsqueeze(-1).expand(-1, num_replicas).flatten()
    flat_log_idx = phy2log.flatten()
    flat_rank_idx = phyrank.flatten()
    flat_phy_ids = torch.arange(num_replicas, dtype=torch.int64, device=logcnt.device).expand(num_layers, -1).flatten()

    # Map to flat index of log2phy
    # Index = layer * (num_logical * max_rep) + logical * max_rep + rank
    flat_indices = (flat_layer_idx * num_logical_experts * max_replicas) + \
                   (flat_log_idx * max_replicas) + \
                   flat_rank_idx

    # Scatter
    log2phy.view(-1).scatter_(0, flat_indices, flat_phy_ids)

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