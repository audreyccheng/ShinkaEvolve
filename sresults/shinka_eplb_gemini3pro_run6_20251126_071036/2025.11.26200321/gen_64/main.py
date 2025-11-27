# EVOLVE-BLOCK-START
"""
Expert parallelism load balancer (EPLB) for vLLM.

This module implements the core rearrangement algorithm using a
Hybrid Ensemble Greedy strategy that evaluates multiple packing
heuristics in parallel to minimize load imbalance.
"""

import torch


def _refine_packing(weights: torch.Tensor,
                    pack_ids: torch.Tensor,
                    pack_loads: torch.Tensor,
                    ranks: torch.Tensor,
                    num_packs: int,
                    num_iters: int = 5) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Refines the packing by attempting to swap a single item between the heaviest
    pack and ANY other pack to reduce the maximum load.
    """
    batch_size, num_items = weights.shape
    device = weights.device
    batch_indices = torch.arange(batch_size, device=device)

    # Precompute pairwise weight differences: [B, N, N]
    # w_diff[b, i, j] = w[b, i] - w[b, j]
    w_diff = weights.unsqueeze(2) - weights.unsqueeze(1)

    for _ in range(num_iters):
        max_load, max_pack_idx = pack_loads.max(dim=1)

        best_gain = torch.zeros(batch_size, device=device, dtype=weights.dtype)
        best_swap_flat = torch.zeros(batch_size, device=device, dtype=torch.int64)
        best_target_pack = torch.full((batch_size,), -1, device=device, dtype=torch.int64)

        # Identify items in max pack [B, N]
        is_in_max = (pack_ids == max_pack_idx.unsqueeze(1))

        # Iterate over all possible target packs
        for p in range(num_packs):
            # Target pack load [B]
            target_load = pack_loads[:, p]

            # Gap available: max_load - target_load
            gap = max_load - target_load

            # Mask valid batches: max_pack != p and gap > 0
            # (Note: if p == max_pack, gap is 0, so gap > 1e-5 covers it)
            active_p_mask = gap > 1e-5

            if not active_p_mask.any():
                continue

            # Identify items in target pack [B, N]
            is_in_target = (pack_ids == torch.tensor(p, device=device))

            # Valid pairs: i in Max, j in Target
            # [B, N, N]
            valid_pair = is_in_max.unsqueeze(2) & is_in_target.unsqueeze(1)

            # We want to swap such that:
            # 1. New Max Load < Old Max Load => (LoadMax - delta) < LoadMax => delta > 0
            # 2. New Target Load < Old Max Load => (LoadTarget + delta) < LoadMax => delta < Gap
            # Objective: Maximize reduction in MaxLoad constrained by TargetLoad not exceeding OldMaxLoad.
            # Reduction = LoadMax - max(LoadMax - delta, LoadTarget + delta)
            #           = min(delta, Gap - delta)

            delta = w_diff

            # Filter deltas
            valid_swap = valid_pair & (delta > 1e-5) & (delta < (gap.view(-1, 1, 1) - 1e-5))

            # Calculate gain
            gain = torch.min(delta, gap.view(-1, 1, 1) - delta)

            # Mask invalid
            gain = torch.where(valid_swap, gain, torch.tensor(-1.0, device=device, dtype=weights.dtype))

            # Find best swap for this target pack
            # Flatten last two dims: [B, N*N]
            p_max_gain, p_flat_idx = gain.view(batch_size, -1).max(dim=1)

            # Update global best if better
            improve_mask = (p_max_gain > best_gain) & active_p_mask

            if improve_mask.any():
                best_gain = torch.where(improve_mask, p_max_gain, best_gain)
                best_swap_flat = torch.where(improve_mask, p_flat_idx, best_swap_flat)
                best_target_pack = torch.where(improve_mask, torch.tensor(p, device=device), best_target_pack)

        # Apply best swaps found across all packs
        active_mask = best_target_pack != -1
        if not active_mask.any():
            break

        active_batch_idx = batch_indices[active_mask]
        active_flat = best_swap_flat[active_mask]
        active_p = best_target_pack[active_mask]

        i_idx = active_flat // num_items
        j_idx = active_flat % num_items

        # Get Max Pack for these batches
        active_max_p = max_pack_idx[active_mask]

        # Weights
        w_i = weights[active_batch_idx, i_idx]
        w_j = weights[active_batch_idx, j_idx]
        delta_val = w_i - w_j

        # Update Loads
        # We need scatter_add or direct indexing. Direct indexing works because active_batch_idx is unique.
        pack_loads[active_batch_idx, active_max_p] -= delta_val
        pack_loads[active_batch_idx, active_p] += delta_val

        # Update IDs
        pack_ids[active_batch_idx, i_idx] = active_p
        pack_ids[active_batch_idx, j_idx] = active_max_p

        # Update Ranks
        r_i = ranks[active_batch_idx, i_idx]
        r_j = ranks[active_batch_idx, j_idx]
        ranks[active_batch_idx, i_idx] = r_j
        ranks[active_batch_idx, j_idx] = r_i

    return pack_ids, ranks, pack_loads


def _vectorized_greedy_packing(weights: torch.Tensor,
                               num_packs: int,
                               capacity: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Vectorized Greedy Packing Kernel.
    """
    batch_size, num_items = weights.shape
    device = weights.device

    # State tracking
    pack_loads = torch.zeros(batch_size, num_packs, device=device, dtype=weights.dtype)
    pack_counts = torch.zeros(batch_size, num_packs, device=device, dtype=torch.int64)

    pack_ids = torch.empty(batch_size, num_items, device=device, dtype=torch.int64)
    ranks = torch.empty(batch_size, num_items, device=device, dtype=torch.int64)

    batch_indices = torch.arange(batch_size, device=device)
    inf_tensor = torch.tensor(float('inf'), device=device, dtype=weights.dtype)

    # Greedy allocation loop
    for i in range(num_items):
        w = weights[:, i]
        valid_mask = pack_counts < capacity
        temp_loads = torch.where(valid_mask, pack_loads, inf_tensor)
        chosen_packs = temp_loads.argmin(dim=1)

        pack_ids[:, i] = chosen_packs
        ranks[:, i] = pack_counts[batch_indices, chosen_packs]

        pack_counts[batch_indices, chosen_packs] += 1
        pack_loads[batch_indices, chosen_packs] += w

    return pack_ids, ranks, pack_loads


def balanced_packing(weight: torch.Tensor,
                     num_packs: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pack n weighted objects to m packs using a Hybrid Ensemble Strategy.
    """
    num_layers, num_items = weight.shape
    device = weight.device
    capacity = num_items // num_packs

    # Configuration
    num_candidates = 128
    num_refine_candidates = 16  # Only refine the top K candidates per layer

    # 1. Base LPT Sort
    lpt_weights, lpt_indices = weight.sort(dim=-1, descending=True)

    # 2. ZigZag (Folded LPT)
    relative_zigzag = torch.empty(num_items, device=device, dtype=torch.long)
    half = (num_items + 1) // 2
    arange = torch.arange(num_items, device=device)
    relative_zigzag[0::2] = arange[:half]
    relative_zigzag[1::2] = arange[half:].flip(0)

    c_zigzag_weights = lpt_weights[:, relative_zigzag].unsqueeze(1)
    c_zigzag_idx = lpt_indices[:, relative_zigzag].unsqueeze(1)

    c_lpt_weights = lpt_weights.unsqueeze(1)
    c_lpt_idx = lpt_indices.unsqueeze(1)

    # 3. Random Shuffles (Exploration)
    num_shuffle = 30
    rand_perm = torch.rand(num_layers, num_shuffle, num_items,
                           device=device).argsort(dim=-1)

    orig_expanded_shuffle = weight.unsqueeze(1).expand(-1, num_shuffle, -1)
    c_shuffle_weights = orig_expanded_shuffle.gather(2, rand_perm)
    c_shuffle_idx = rand_perm

    # 4. Noisy LPT
    num_noisy = num_candidates - 2 - num_shuffle  # 96
    num_small = num_noisy // 2
    num_large = num_noisy - num_small

    noise_small = (torch.rand(num_layers, num_small, num_items, device=device) *
                   0.2) + 0.9
    noise_large = (torch.rand(num_layers, num_large, num_items, device=device) *
                   0.8) + 0.6

    noise = torch.cat([noise_small, noise_large], dim=1)
    noisy_weights_in = weight.unsqueeze(1) * noise
    _, noisy_sorted_idx = noisy_weights_in.sort(dim=-1, descending=True)

    orig_expanded_noisy = weight.unsqueeze(1).expand(-1, num_noisy, -1)
    actual_noisy_weights = orig_expanded_noisy.gather(2, noisy_sorted_idx)

    # Combine all
    all_weights = torch.cat(
        [c_lpt_weights, c_zigzag_weights, c_shuffle_weights, actual_noisy_weights],
        dim=1)
    all_indices = torch.cat(
        [c_lpt_idx, c_zigzag_idx, c_shuffle_idx, noisy_sorted_idx], dim=1)

    # Flatten for Greedy
    flat_weights = all_weights.view(-1, num_items)

    # Greedy Packing
    flat_ids, flat_ranks, flat_loads = _vectorized_greedy_packing(
        flat_weights, num_packs, capacity)

    # --- Pre-Selection for Refinement ---
    loads = flat_loads.view(num_layers, num_candidates, num_packs)
    imbalance = loads.max(dim=-1).values - loads.min(dim=-1).values

    # Select top K candidates
    _, best_k_indices = imbalance.topk(num_refine_candidates,
                                       dim=1,
                                       largest=False)  # [L, K]

    # Flatten indices to select from flat tensors
    layer_offsets = (torch.arange(num_layers, device=device) *
                     num_candidates).unsqueeze(1)
    flat_selected_indices = (best_k_indices + layer_offsets).flatten()

    refined_weights = flat_weights[flat_selected_indices]
    refined_ids = flat_ids[flat_selected_indices]
    refined_ranks = flat_ranks[flat_selected_indices]
    refined_loads = flat_loads[flat_selected_indices]

    # Run Refinement on top K
    refined_ids, refined_ranks, refined_loads = _refine_packing(
        refined_weights, refined_ids, refined_loads, refined_ranks, num_packs=num_packs, num_iters=5)

    # --- Final Selection ---
    loads_final = refined_loads.view(num_layers, num_refine_candidates,
                                     num_packs)
    imbalance_final = loads_final.max(dim=-1).values - loads_final.min(
        dim=-1).values

    best_in_k_idx = imbalance_final.argmin(dim=1)  # [L]

    # Gather results
    idx_view = best_in_k_idx.view(num_layers, 1, 1).expand(-1, 1, num_items)
    final_aligned_ids = refined_ids.view(num_layers, num_refine_candidates,
                                         num_items).gather(1, idx_view).squeeze(1)
    final_aligned_ranks = refined_ranks.view(
        num_layers, num_refine_candidates,
        num_items).gather(1, idx_view).squeeze(1)

    # Recover global indices for scatter
    best_global_cand_idx = best_k_indices.gather(
        1, best_in_k_idx.unsqueeze(1)).squeeze(1)
    global_idx_view = best_global_cand_idx.view(num_layers, 1,
                                                1).expand(-1, 1, num_items)
    final_sorted_idx = all_indices.gather(1, global_idx_view).squeeze(1)

    # Scatter back
    pack_index = torch.empty(num_layers,
                             num_items,
                             device=device,
                             dtype=torch.int64)
    rank_in_pack = torch.empty(num_layers,
                               num_items,
                               device=device,
                               dtype=torch.int64)

    pack_index.scatter_(1, final_sorted_idx, final_aligned_ids)
    rank_in_pack.scatter_(1, final_sorted_idx, final_aligned_ranks)

    return pack_index, rank_in_pack


def replicate_experts(
        weight: torch.Tensor,
        num_phy: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Replicate experts using Binary Search on Max Load followed by Greedy Refinement.
    """
    num_layers, num_log = weight.shape
    device = weight.device

    # Trivial case
    if num_phy == num_log:
        phy2log = torch.arange(num_log, device=device).expand(num_layers, -1)
        rank = torch.zeros(num_layers, num_phy, dtype=torch.int64, device=device)
        logcnt = torch.ones(num_layers, num_log, dtype=torch.int64, device=device)
        return phy2log, rank, logcnt

    # Binary Search
    low = weight.sum(dim=-1, keepdim=True) / num_phy
    high = weight.max(dim=-1, keepdim=True).values
    low = torch.max(low, torch.tensor(1e-4, device=device))

    for _ in range(15):
        mid = (low + high) * 0.5
        counts = torch.ceil(weight / mid)
        total = counts.sum(dim=-1, keepdim=True)
        mask = total <= num_phy
        high = torch.where(mask, mid, high)
        low = torch.where(mask, low, mid)

    logcnt = torch.ceil(weight / high).long().clamp(min=1)

    # Correct sum
    current_sum = logcnt.sum(dim=-1)
    diff = num_phy - current_sum

    # Under-allocation: Add to max density
    max_diff = int(diff.max().item())
    if max_diff > 0:
        rows = torch.arange(num_layers, device=device)
        for _ in range(max_diff):
            active = current_sum < num_phy
            if not active.any(): break

            density = weight / logcnt.float()
            target_idx = density.argmax(dim=-1)

            active_rows = rows[active]
            active_targets = target_idx[active]

            logcnt.index_put_((active_rows, active_targets),
                              torch.tensor(1, device=device, dtype=torch.int64),
                              accumulate=True)
            current_sum[active] += 1

    # Over-allocation: Remove from min cost
    min_diff = int(diff.min().item())
    if min_diff < 0:
        rows = torch.arange(num_layers, device=device)
        for _ in range(abs(min_diff)):
            active = current_sum > num_phy
            if not active.any(): break

            valid = logcnt > 1
            cost = weight / (logcnt - 1).float()
            cost[~valid] = float('inf')

            target_idx = cost.argmin(dim=-1)

            active_rows = rows[active]
            active_targets = target_idx[active]

            logcnt.index_put_((active_rows, active_targets),
                              torch.tensor(-1, device=device, dtype=torch.int64),
                              accumulate=True)
            current_sum[active] -= 1

    # Construct maps
    flat_log_ids = torch.arange(num_log, device=device).repeat(num_layers)
    flat_counts = logcnt.flatten()
    flat_phy2log = torch.repeat_interleave(flat_log_ids, flat_counts)

    target_size = num_layers * num_phy
    if flat_phy2log.numel() != target_size:
        if flat_phy2log.numel() < target_size:
            flat_phy2log = torch.cat([flat_phy2log, torch.zeros(target_size - flat_phy2log.numel(), device=device, dtype=torch.long)])
        else:
            flat_phy2log = flat_phy2log[:target_size]

    phy2log = flat_phy2log.view(num_layers, num_phy)

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
    Hierarchical rebalancing.
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

    # 1. Groups -> Nodes
    tokens_per_group = weight.unflatten(-1, (num_groups, group_size)).sum(-1)
    group_pack_index, group_rank_in_pack = balanced_packing(
        tokens_per_group, num_nodes)

    log2mlog = (((group_pack_index * groups_per_node + group_rank_in_pack) *
                 group_size).unsqueeze(-1) +
                torch.arange(group_size,
                             dtype=torch.int64,
                             device=group_pack_index.device)).flatten(-2)
    mlog2log = inverse(log2mlog)

    # 2. Replicate within nodes
    tokens_per_mlog = weight.gather(-1, mlog2log).view(
        -1, num_logical_experts // num_nodes)
    phy2mlog, phyrank, mlogcnt = replicate_experts(
        tokens_per_mlog, num_physical_experts // num_nodes)

    # 3. Physical -> GPUs
    tokens_per_phy = (tokens_per_mlog / mlogcnt).gather(-1, phy2mlog)
    pack_index, rank_in_pack = balanced_packing(tokens_per_phy,
                                                num_gpus // num_nodes)

    phy2pphy = pack_index * phy_experts_per_gpu + rank_in_pack
    pphy2phy = inverse(phy2pphy)

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
    Entry point.
    """
    num_layers, num_logical_experts = weight.shape
    weight = weight.float().cpu()

    if num_groups % num_nodes == 0:
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, num_groups, num_nodes, num_gpus)
    else:
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, 1, 1, num_gpus)

    max_replicas = int(logcnt.max().item())

    log2phy = torch.full(
        (num_layers, num_logical_experts, max_replicas),
        -1,
        dtype=torch.int64,
        device=logcnt.device,
    )

    flat_layer_idx = torch.arange(num_layers, device=logcnt.device).unsqueeze(-1).expand(-1, num_replicas).flatten()
    flat_log_idx = phy2log.flatten()
    flat_rank_idx = phyrank.flatten()
    flat_phy_ids = torch.arange(num_replicas, dtype=torch.int64, device=logcnt.device).expand(num_layers, -1).flatten()

    indices = (flat_layer_idx * num_logical_experts * max_replicas) + \
              (flat_log_idx * max_replicas) + \
              flat_rank_idx

    log2phy.view(-1).scatter_(0, indices, flat_phy_ids)

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