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
                    ranks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Refines the packing by attempting to swap a single item between the heaviest
    and lightest packs in each batch to reduce imbalance.
    """
    batch_size, num_items = weights.shape
    device = weights.device
    batch_indices = torch.arange(batch_size, device=device)

    # Iterative refinement (1 pass is usually sufficient for speed/imbalance tradeoff)
    for _ in range(1):
        # Identify heaviest and lightest packs
        max_load, max_pack_idx = pack_loads.max(dim=1)
        min_load, min_pack_idx = pack_loads.min(dim=1)

        current_diff = max_load - min_load

        # Mask items belonging to these packs
        # [B, N]
        is_in_max = (pack_ids == max_pack_idx.unsqueeze(1))
        is_in_min = (pack_ids == min_pack_idx.unsqueeze(1))

        # We want to swap item i (from max) with item j (from min)
        # Minimize |(L_max - w_i + w_j) - (L_min - w_j + w_i)|
        # = |(L_max - L_min) - 2*(w_i - w_j)|
        # Let target = (L_max - L_min) / 2

        # Expand weights for pairwise diff
        # [B, N, N] -> w[b, i] - w[b, j]
        # Memory warning: N=256, B=2048 -> 128M elements. Safe.
        w_diff = weights.unsqueeze(2) - weights.unsqueeze(1)

        # Calculate improvement metric
        # We want 2 * w_diff to be close to current_diff
        # metric = | current_diff - 2 * w_diff |
        metric = torch.abs(current_diff.view(-1, 1, 1) - 2 * w_diff)

        # Apply validity mask
        valid_swap = is_in_max.unsqueeze(2) & is_in_min.unsqueeze(1)
        metric = torch.where(valid_swap, metric, torch.tensor(float('inf'), device=device))

        # Find best swap per batch
        flat_metric = metric.view(batch_size, -1)
        min_metric, flat_indices = flat_metric.min(dim=1)

        # Only apply if improvement
        # We strictly want the new diff to be smaller than current_diff
        # min_metric is the new diff between these two packs
        improvement = min_metric < current_diff

        batch_indices_active = batch_indices[improvement]
        if batch_indices_active.numel() == 0:
            break

        indices_active = flat_indices[improvement]
        i_idx = indices_active // num_items
        j_idx = indices_active % num_items

        # Perform swap
        p_max = max_pack_idx[batch_indices_active]
        p_min = min_pack_idx[batch_indices_active]

        w_i = weights[batch_indices_active, i_idx]
        w_j = weights[batch_indices_active, j_idx]
        delta = w_i - w_j

        # Update Loads
        pack_loads[batch_indices_active, p_max] -= delta
        pack_loads[batch_indices_active, p_min] += delta

        # Update IDs
        pack_ids[batch_indices_active, i_idx] = p_min
        pack_ids[batch_indices_active, j_idx] = p_max

        # Update Ranks (swap them to maintain 0..capacity-1 set in each pack)
        r_i = ranks[batch_indices_active, i_idx]
        r_j = ranks[batch_indices_active, j_idx]
        ranks[batch_indices_active, i_idx] = r_j
        ranks[batch_indices_active, j_idx] = r_i

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

    # Increase candidates to 128 for broader search
    num_candidates = 128

    # 1. Base LPT Sort
    lpt_weights, lpt_indices = weight.sort(dim=-1, descending=True)

    # 2. ZigZag
    relative_zigzag = torch.empty(num_items, device=device, dtype=torch.long)
    half = (num_items + 1) // 2
    arange = torch.arange(num_items, device=device)
    relative_zigzag[0::2] = arange[:half]
    relative_zigzag[1::2] = arange[half:].flip(0)

    # 3. Noisy LPT
    num_noisy = num_candidates - 2
    noise = (torch.rand(num_layers, num_noisy, num_items, device=device) * 0.3) + 0.85

    noisy_weights_in = weight.unsqueeze(1) * noise
    noisy_sorted_weights, noisy_sorted_idx = noisy_weights_in.sort(dim=-1, descending=True)

    orig_expanded = weight.unsqueeze(1).expand(-1, num_noisy, -1)
    actual_noisy_weights = orig_expanded.gather(2, noisy_sorted_idx)

    c_lpt_weights = lpt_weights.unsqueeze(1)
    c_lpt_idx = lpt_indices.unsqueeze(1)

    c_zigzag_weights = c_lpt_weights[:, :, relative_zigzag]
    c_zigzag_idx = c_lpt_idx[:, :, relative_zigzag]

    all_weights = torch.cat([c_lpt_weights, c_zigzag_weights, actual_noisy_weights], dim=1)
    all_indices = torch.cat([c_lpt_idx, c_zigzag_idx, noisy_sorted_idx], dim=1)

    flat_weights = all_weights.view(-1, num_items)

    # Run Greedy Packing
    flat_ids, flat_ranks, flat_loads = _vectorized_greedy_packing(flat_weights, num_packs, capacity)

    # Run Refinement
    flat_ids, flat_ranks, flat_loads = _refine_packing(flat_weights, flat_ids, flat_loads, flat_ranks)

    # --- Selection ---
    loads = flat_loads.view(num_layers, num_candidates, num_packs)
    imbalance = loads.max(dim=-1).values - loads.min(dim=-1).values

    best_candidate_idx = imbalance.argmin(dim=1)

    idx_view = best_candidate_idx.view(num_layers, 1, 1).expand(-1, 1, num_items)

    final_sorted_idx = all_indices.gather(1, idx_view).squeeze(1)

    aligned_ids = flat_ids.view(num_layers, num_candidates, num_items)
    final_aligned_ids = aligned_ids.gather(1, idx_view).squeeze(1)

    aligned_ranks = flat_ranks.view(num_layers, num_candidates, num_items)
    final_aligned_ranks = aligned_ranks.gather(1, idx_view).squeeze(1)

    pack_index = torch.empty(num_layers, num_items, device=device, dtype=torch.int64)
    rank_in_pack = torch.empty(num_layers, num_items, device=device, dtype=torch.int64)

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