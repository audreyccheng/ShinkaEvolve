# EVOLVE-BLOCK-START
"""
Expert parallelism load balancer (EPLB) for vLLM.

This module implements the core rearrangement algorithm using a
Massive Parallel Ensemble Greedy packing strategy with Refinement
to maximize load balance.
"""

import torch


def balanced_packing(weight: torch.Tensor,
                     num_packs: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pack n weighted objects to m packs using a Massive Parallel Ensemble strategy
    with Vectorized Refinement.

    Strategies included:
    1. LPT (Longest Processing Time)
    2. ZigZag LPT (Interleaved heavy/light)
    3. Noisy LPT
    4. Random Shuffling

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

    # --- 1. Generate Candidates ---
    # A. Base LPT
    lpt_weights, lpt_indices = weight.sort(dim=-1, descending=True)

    # B. ZigZag Permutation
    zigzag_order = torch.empty(num_items, device=device, dtype=torch.long)
    half = (num_items + 1) // 2
    arange = torch.arange(num_items, device=device)
    zigzag_order[0::2] = arange[:half]
    zigzag_order[1::2] = arange[half:].flip(0)
    zigzag_indices = lpt_indices.gather(1, zigzag_order.unsqueeze(0).expand(num_layers, -1))

    # C. Randomized Candidates
    num_others = num_candidates - 2
    num_noisy = num_others // 2
    num_random = num_others - num_noisy

    # Noisy LPT
    noise = (torch.rand(num_layers, num_noisy, num_items, device=device) * 0.4) + 0.8
    noisy_w = weight.unsqueeze(1) * noise
    _, noisy_indices = noisy_w.sort(dim=-1, descending=True)

    # Random Shuffle
    rand_vals = torch.rand(num_layers, num_random, num_items, device=device)
    _, random_indices = rand_vals.sort(dim=-1)

    # Combine all indices
    c0 = lpt_indices.unsqueeze(1)
    c1 = zigzag_indices.unsqueeze(1)
    all_indices = torch.cat([c0, c1, noisy_indices, random_indices], dim=1)

    # Gather weights [Layers, Candidates, Items]
    expanded_weights = weight.unsqueeze(1).expand(-1, num_candidates, -1)
    ordered_weights = expanded_weights.gather(2, all_indices)

    # Flatten batch dimensions for processing
    batch_size = num_layers * num_candidates
    flat_weights = ordered_weights.view(batch_size, num_items)

    # --- 2. Parallel Chunked Greedy Packing ---
    pack_loads = torch.zeros(batch_size, num_packs, device=device, dtype=weight.dtype)

    # Store indices (into flat_weights rows) and weights per pack for refinement
    # [Batch, Packs, Capacity]
    item_indices_per_pack = torch.empty(batch_size, num_packs, capacity, device=device, dtype=torch.int64)
    weights_per_pack = torch.empty(batch_size, num_packs, capacity, device=device, dtype=weight.dtype)

    chunk_indices_template = torch.arange(num_packs, device=device).expand(batch_size, -1)

    for k in range(capacity):
        start = k * num_packs
        end = start + num_packs
        chunk_w = flat_weights[:, start:end]

        # Sort packs by load: heavier items go to lighter packs
        _, sorted_pack_indices = pack_loads.sort(dim=-1)

        # Map assignments: pack P gets item J from chunk
        # item index relative to start is J.
        # sorted_pack_indices[b, j] = p means item j goes to pack p.
        # We need to fill item_indices_per_pack[b, p, k] = start + j

        # Use scatter to place items into their assigned packs
        # src: indices of items in the chunk (0..P-1) + start
        src_indices = chunk_indices_template + start
        item_indices_per_pack[:, :, k].scatter_(1, sorted_pack_indices, src_indices)

        # Same for weights
        weights_per_pack[:, :, k].scatter_(1, sorted_pack_indices, chunk_w)

        # Update loads
        # We add the weight of the item assigned to pack p
        # weights_per_pack[:, p, k] contains exactly that weight
        pack_loads += weights_per_pack[:, :, k]

    # --- 3. Vectorized Refinement (2-Opt Swap) ---
    # Iteratively swap items between max and min load packs
    for _ in range(20):
        max_load, max_p = pack_loads.max(dim=1)
        min_load, min_p = pack_loads.min(dim=1)

        current_imbalance = max_load - min_load

        # Gather items (weights) in these packs
        # [Batch, Capacity]
        gather_idx_max = max_p.view(-1, 1, 1).expand(-1, 1, capacity)
        gather_idx_min = min_p.view(-1, 1, 1).expand(-1, 1, capacity)

        w_max = weights_per_pack.gather(1, gather_idx_max).squeeze(1)
        w_min = weights_per_pack.gather(1, gather_idx_min).squeeze(1)

        # Compute potential new imbalance for all pairs (i, j)
        # diff = 2 * (w_max[i] - w_min[j])
        # new_imbalance = |current - diff|
        # We want to minimize this.

        # [Batch, Cap, Cap]
        diff_matrix = 2 * (w_max.unsqueeze(2) - w_min.unsqueeze(1))

        # We only care if new imbalance < current imbalance
        # projected = |current - diff|
        # improvement condition: |current - diff| < current
        # equivalent to 0 < diff < 2*current (assuming current > 0)

        projected_imbalance = torch.abs(current_imbalance.view(-1, 1, 1) - diff_matrix)

        # Mask invalid swaps where packs are same (diff=0) or no improvement
        # Adding a small epsilon to ensure strict improvement
        improvement_mask = projected_imbalance < (current_imbalance.view(-1, 1, 1) - 1e-4)

        # Set non-improving to infinity
        valid_projected = torch.where(improvement_mask, projected_imbalance, torch.tensor(float('inf'), device=device))

        # Find best swap
        flat_proj = valid_projected.view(batch_size, -1)
        best_val, best_idx_flat = flat_proj.min(dim=1)

        # Check which batches have a valid swap
        active_mask = best_val != float('inf')
        if not active_mask.any():
            break

        # Perform updates for active batches
        active_indices = torch.nonzero(active_mask).squeeze(1)

        # Decode best_idx_flat -> (i, j)
        best_idx = best_idx_flat[active_indices]
        idx_i = best_idx // capacity # index in max pack
        idx_j = best_idx % capacity  # index in min pack

        batch_idx = active_indices
        p_max = max_p[batch_idx]
        p_min = min_p[batch_idx]

        # Get values to swap
        val_max = w_max[batch_idx, idx_i]
        val_min = w_min[batch_idx, idx_j]

        item_max = item_indices_per_pack[batch_idx, p_max, idx_i]
        item_min = item_indices_per_pack[batch_idx, p_min, idx_j]

        # Update weights_per_pack
        weights_per_pack[batch_idx, p_max, idx_i] = val_min
        weights_per_pack[batch_idx, p_min, idx_j] = val_max

        # Update item_indices
        item_indices_per_pack[batch_idx, p_max, idx_i] = item_min
        item_indices_per_pack[batch_idx, p_min, idx_j] = item_max

        # Update pack_loads
        load_diff = val_max - val_min
        pack_loads[batch_idx, p_max] -= load_diff
        pack_loads[batch_idx, p_min] += load_diff

    # --- 4. Selection and Reconstruction ---
    # Find best candidate per layer
    pack_loads_view = pack_loads.view(num_layers, num_candidates, num_packs)
    imbalance = pack_loads_view.max(dim=-1).values - pack_loads_view.min(dim=-1).values
    best_cand_idx = imbalance.argmin(dim=1) # [Layers]

    # Gather best item_indices_per_pack [Layers, Packs, Capacity]
    # item_indices_per_pack is [L*C, P, K] -> [L, C, P, K]
    reshaped_assignments = item_indices_per_pack.view(num_layers, num_candidates, num_packs, capacity)

    # Select best candidate
    gather_idx_assgn = best_cand_idx.view(-1, 1, 1, 1).expand(-1, 1, num_packs, capacity)
    final_assignments = reshaped_assignments.gather(1, gather_idx_assgn).squeeze(1)

    # Gather best permutation map [Layers, Items]
    # all_indices is [L, C, Items]
    gather_idx_perm = best_cand_idx.view(-1, 1, 1).expand(-1, 1, num_items)
    final_perm = all_indices.gather(1, gather_idx_perm).squeeze(1)

    # Reconstruct Output
    # final_assignments[l, p, k] = i (index in final_perm)
    # The item ID is final_perm[l, i]
    # It is assigned to pack p, rank k

    # Flatten assignment dimensions to scatter
    # [L, P, K]
    flat_indices = final_assignments.flatten() # [L*P*K] = [L*N] values are 0..N-1 (local)

    # We need to map local indices 0..N-1 back to original items
    # We iterate layers or do it vectorized
    # final_perm [L, N]
    # We need to gather from final_perm using flat_indices (adjusted for layer offset)

    layer_offsets = torch.arange(num_layers, device=device).unsqueeze(1).expand(-1, num_items * num_candidates).flatten() # Wait, simpler

    # final_assignments values are 0..N-1 relative to the layer's sorted list
    # We can use gather on final_perm
    # final_assignments is [L, P, K]. Flatten last two dims -> [L, N] (roughly, since N=P*K)
    sorted_assignments = final_assignments.view(num_layers, num_items)

    # Get original item IDs
    # orig_ids[l, x] = final_perm[l, sorted_assignments[l, x]]
    orig_ids = final_perm.gather(1, sorted_assignments)

    # Now we have: for each slot x (which corresponds to pack p, rank k), the item ID is orig_ids[l, x].
    # We need output[l, item_id] = pack_id

    # Construct pack_ids tensor matching sorted_assignments geometry
    # final_assignments index [l, p, k] corresponds to pack p, rank k
    pack_ids_template = torch.arange(num_packs, device=device).view(1, num_packs, 1).expand(num_layers, -1, capacity)
    rank_ids_template = torch.arange(capacity, device=device).view(1, 1, capacity).expand(num_layers, num_packs, -1)

    pack_ids_flat = pack_ids_template.flatten(1) # [L, N]
    rank_ids_flat = rank_ids_template.flatten(1) # [L, N]

    # Scatter
    pack_index = torch.empty(num_layers, num_items, device=device, dtype=torch.int64)
    rank_in_pack = torch.empty(num_layers, num_items, device=device, dtype=torch.int64)

    pack_index.scatter_(1, orig_ids, pack_ids_flat)
    rank_in_pack.scatter_(1, orig_ids, rank_ids_flat)

    return pack_index, rank_in_pack


def replicate_experts(
        weight: torch.Tensor,
        num_phy: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Replicate experts using Binary Search on Max Load followed by Greedy Refinement.
    """
    num_layers, num_log = weight.shape
    device = weight.device

    if num_phy == num_log:
        phy2log = torch.arange(num_log, device=device).expand(num_layers, -1)
        rank = torch.zeros(num_layers, num_phy, dtype=torch.int64, device=device)
        logcnt = torch.ones(num_layers, num_log, dtype=torch.int64, device=device)
        return phy2log, rank, logcnt

    # Binary Search
    low = weight.sum(dim=-1, keepdim=True) / num_phy
    high = weight.max(dim=-1, keepdim=True).values
    low = torch.max(low, torch.tensor(1e-6, device=device))

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
            # Inactive rows shouldn't be selected
            density[~active] = -1.0

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
            cost[~active] = float('inf')

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

    # Reshape safeguard
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