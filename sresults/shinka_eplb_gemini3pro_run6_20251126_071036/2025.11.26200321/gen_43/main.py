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
                    num_iters: int = 20) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Iterative 2-opt swap refinement to minimize load imbalance.
    """
    num_layers, num_items = weight.shape
    device = weight.device

    # Compute initial loads
    pack_loads = torch.zeros(num_layers, num_packs, device=device, dtype=weight.dtype)
    pack_loads.scatter_add_(1, pack_index, weight)

    batch_indices = torch.arange(num_layers, device=device)

    for _ in range(num_iters):
        # Find max and min packs
        max_load, max_idx = pack_loads.max(dim=1)
        min_load, min_idx = pack_loads.min(dim=1)
        curr_diff = max_load - min_load

        # Identify candidate items
        # mask: [L, N]
        is_max = (pack_index == max_idx.unsqueeze(1))
        is_min = (pack_index == min_idx.unsqueeze(1))

        # w_i - w_j matrix
        # [L, N, 1] - [L, 1, N] -> [L, N, N]
        w_diff = weight.unsqueeze(2) - weight.unsqueeze(1)

        # Target: maximize reduction
        # New Diff = |(Max - Min) - 2(w_i - w_j)|
        # We want to minimize this new diff.

        # Valid only if i in max and j in min
        valid_mask = is_max.unsqueeze(2) & is_min.unsqueeze(1)

        metric = torch.abs(curr_diff.view(-1, 1, 1) - 2 * w_diff)
        # Mask invalid
        metric = torch.where(valid_mask, metric, torch.tensor(float('inf'), device=device))

        # Find best swap
        flat_metric = metric.view(num_layers, -1)
        best_val, best_flat_idx = flat_metric.min(dim=1)

        # Improvement check
        # We need improvement over current diff
        improve_mask = best_val < (curr_diff - 1e-5)

        if not improve_mask.any():
            break

        # Apply swaps
        active_layers = batch_indices[improve_mask]
        active_flat = best_flat_idx[improve_mask]

        idx_i = active_flat // num_items
        idx_j = active_flat % num_items

        p_max = max_idx[active_layers]
        p_min = min_idx[active_layers]

        # Update weights/loads
        w_i = weight[active_layers, idx_i]
        w_j = weight[active_layers, idx_j]
        diff_w = w_i - w_j

        pack_loads[active_layers, p_max] -= diff_w
        pack_loads[active_layers, p_min] += diff_w

        # Swap indices
        pack_index[active_layers, idx_i] = p_min
        pack_index[active_layers, idx_j] = p_max

        # Swap ranks
        r_i = rank_in_pack[active_layers, idx_i]
        r_j = rank_in_pack[active_layers, idx_j]

        rank_in_pack[active_layers, idx_i] = r_j
        rank_in_pack[active_layers, idx_j] = r_i

    return pack_index, rank_in_pack


def balanced_packing(weight: torch.Tensor,
                     num_packs: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pack n weighted objects to m packs using a Massive Parallel Ensemble strategy.

    Generates 128 candidates using LPT, ZigZag, Random Shuffle, and Randomized perturbations.
    Selects the best packing per layer minimizing load imbalance, followed by local refinement.

    Parameters:
        weight: [layers, n], the weight of each item
        num_packs: number of packs

    Returns:
        pack_index: [layers, n], the pack index of each item
        rank_in_pack: [layers, n], the rank of the item in the pack
    """
    num_layers, num_items = weight.shape
    device = weight.device
    num_candidates = 128
    capacity = num_items // num_packs

    # 1. Generate Candidates
    # Base: Sort descending (LPT)
    lpt_val, lpt_idx = weight.sort(dim=-1, descending=True)

    # ZigZag permutation of LPT: 0, N-1, 1, N-2, ...
    zigzag_perm = torch.empty(num_items, device=device, dtype=torch.long)
    half = (num_items + 1) // 2
    arange = torch.arange(num_items, device=device)
    zigzag_perm[0::2] = arange[:half]
    zigzag_perm[1::2] = arange[half:].flip(0)

    # Candidate Indices [L, C, N]
    # C0: lpt_idx
    c0_idx = lpt_idx.unsqueeze(1)

    # C1: lpt_idx gathered by zigzag_perm
    # Permute columns of lpt_idx
    c1_idx = lpt_idx.gather(1, zigzag_perm.view(1, -1).expand(num_layers, -1)).unsqueeze(1)

    # C2: Random Shuffles (Explore non-greedy basins)
    # Using 32 random permutations
    num_random = 32
    rand_perm = torch.rand(num_layers, num_random, num_items, device=device).argsort(dim=-1)

    # C3: Noisy LPT candidates
    num_noisy = num_candidates - 2 - num_random
    noise = torch.rand(num_layers, num_noisy, num_items, device=device) * 0.4 + 0.8
    noisy_weights = weight.unsqueeze(1) * noise
    _, c_noisy_idx = noisy_weights.sort(dim=-1, descending=True)

    all_indices = torch.cat([c0_idx, c1_idx, rand_perm, c_noisy_idx], dim=1)

    # Gather actual weights [L, C, N]
    expanded_weight = weight.unsqueeze(1).expand(-1, num_candidates, -1)
    ordered_weights = expanded_weight.gather(2, all_indices)

    # Flatten for vectorized kernel
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

    # 3. Selection
    loads = pack_loads.view(num_layers, num_candidates, num_packs)
    imbalance = loads.max(dim=-1).values - loads.min(dim=-1).values
    best_candidate_idx = imbalance.argmin(dim=1)

    # Helper to gather
    def gather_best(tensor_lcn, best_idx_l):
        idx_expanded = best_idx_l.view(num_layers, 1, 1).expand(-1, 1, num_items)
        return tensor_lcn.gather(1, idx_expanded).squeeze(1)

    final_ordered_indices = gather_best(all_indices, best_candidate_idx)
    final_packs_ordered = gather_best(flat_assigned_packs.view(num_layers, num_candidates, num_items), best_candidate_idx)
    final_ranks_ordered = gather_best(flat_assigned_ranks.view(num_layers, num_candidates, num_items), best_candidate_idx)

    pack_index = torch.empty(num_layers, num_items, device=device, dtype=torch.int64)
    rank_in_pack = torch.empty(num_layers, num_items, device=device, dtype=torch.int64)

    pack_index.scatter_(1, final_ordered_indices, final_packs_ordered)
    rank_in_pack.scatter_(1, final_ordered_indices, final_ranks_ordered)

    # 4. Refinement
    # Apply local search to further improve the best selected candidate
    pack_index, rank_in_pack = _refine_packing(weight, pack_index, rank_in_pack, num_packs)

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