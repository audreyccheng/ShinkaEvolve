# EVOLVE-BLOCK-START
"""
Expert parallelism load balancer (EPLB) for vLLM.

This module implements the core rearrangement algorithm using a
Parallel Ensemble Greedy packing strategy to maximize load balance.
"""

import torch


def _refine_packing(weights: torch.Tensor,
                    pack_ids: torch.Tensor,
                    pack_loads: torch.Tensor,
                    ranks: torch.Tensor,
                    num_iters: int = 20) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Refines the packing by attempting to swap items from the heaviest pack
    to any other pack to strictly reduce the maximum load.
    """
    batch_size, num_items = weights.shape
    device = weights.device

    # Precompute pairwise weight differences: w[b, i] - w[b, j]
    # [B, N, N]
    w_diff = weights.unsqueeze(2) - weights.unsqueeze(1)

    batch_range = torch.arange(batch_size, device=device)

    for _ in range(num_iters):
        # Identify Max Pack
        max_load, max_idx = pack_loads.max(dim=1) # [B]

        # Load of pack where j resides
        # [B, N]
        load_j = pack_loads.gather(1, pack_ids)

        # Gap = MaxLoad - Load_j
        # [B, 1, N]
        gap = (max_load.unsqueeze(1) - load_j).unsqueeze(1)

        # Delta = w_i - w_j
        # [B, N, N]
        delta = w_diff

        # Valid Swap Mask: i is Max, j is NOT Max
        # [B, N, N]
        is_max = (pack_ids == max_idx.unsqueeze(1))
        valid_swap = is_max.unsqueeze(2) & (~is_max.unsqueeze(1))

        # Gain calculation
        # Gain = min(delta, gap - delta)
        gain = torch.min(delta, gap - delta)

        # Filter valid swaps with positive gain
        valid_swap = valid_swap & (gain > 1e-6)

        # Apply mask
        gain = torch.where(valid_swap, gain, torch.tensor(-1.0, device=device, dtype=weights.dtype))

        # Find best swap
        # Flatten last two dims [B, N*N]
        flat_gain = gain.view(batch_size, -1)
        best_gain, best_flat_idx = flat_gain.max(dim=1)

        # Apply swaps for batches with improvement
        active = best_gain > 1e-6
        if not active.any():
            break

        b_idx = batch_range[active]
        flat_idx = best_flat_idx[active]

        idx_i = flat_idx // num_items
        idx_j = flat_idx % num_items

        p_max = max_idx[b_idx]
        p_target = pack_ids[b_idx, idx_j]

        w_i = weights[b_idx, idx_i]
        w_j = weights[b_idx, idx_j]
        d = w_i - w_j

        # Update loads
        pack_loads[b_idx, p_max] -= d
        pack_loads[b_idx, p_target] += d

        # Update indices
        pack_ids[b_idx, idx_i] = p_target
        pack_ids[b_idx, idx_j] = p_max

        # Update ranks
        r_i = ranks[b_idx, idx_i]
        r_j = ranks[b_idx, idx_j]
        ranks[b_idx, idx_i] = r_j
        ranks[b_idx, idx_j] = r_i

    return pack_ids, ranks, pack_loads


def _vectorized_greedy_packing(weights: torch.Tensor,
                               num_packs: int,
                               capacity: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Vectorized Greedy Packing Kernel.
    """
    batch_size, num_items = weights.shape
    device = weights.device

    pack_loads = torch.zeros(batch_size, num_packs, device=device, dtype=weights.dtype)
    pack_counts = torch.zeros(batch_size, num_packs, device=device, dtype=torch.int64)
    pack_ids = torch.empty(batch_size, num_items, device=device, dtype=torch.int64)
    ranks = torch.empty(batch_size, num_items, device=device, dtype=torch.int64)

    batch_indices = torch.arange(batch_size, device=device)
    inf_tensor = torch.tensor(float('inf'), device=device, dtype=weights.dtype)

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
    Pack n weighted objects to m packs using a Massive Parallel Ensemble strategy.
    """
    num_layers, num_items = weight.shape
    device = weight.device
    num_candidates = 128
    num_refine = 8
    capacity = num_items // num_packs

    # 1. Candidate Generation
    # A. LPT
    lpt_val, lpt_idx = weight.sort(dim=-1, descending=True)
    c_lpt = lpt_idx.unsqueeze(1)

    # B. ZigZag
    zigzag_perm = torch.empty(num_items, device=device, dtype=torch.long)
    half = (num_items + 1) // 2
    arange = torch.arange(num_items, device=device)
    zigzag_perm[0::2] = arange[:half]
    zigzag_perm[1::2] = arange[half:].flip(0)
    c_zigzag = lpt_idx.gather(1, zigzag_perm.unsqueeze(0).expand(num_layers, -1)).unsqueeze(1)

    # C. Random Shuffles
    num_random = 30
    rand_perm = torch.rand(num_layers, num_random, num_items, device=device).argsort(dim=-1)

    # D. Noisy LPT
    num_noisy = num_candidates - 2 - num_random
    noise = torch.rand(num_layers, num_noisy, num_items, device=device) * 0.4 + 0.8
    noisy_w = weight.unsqueeze(1) * noise
    _, c_noisy = noisy_w.sort(dim=-1, descending=True)

    all_indices = torch.cat([c_lpt, c_zigzag, rand_perm, c_noisy], dim=1)

    # Gather weights
    expanded_weight = weight.unsqueeze(1).expand(-1, num_candidates, -1)
    ordered_weights = expanded_weight.gather(2, all_indices)

    # Flatten
    batch_size = num_layers * num_candidates
    flat_weights = ordered_weights.view(batch_size, num_items)

    # 2. Greedy Packing
    flat_ids, flat_ranks, flat_loads = _vectorized_greedy_packing(flat_weights, num_packs, capacity)

    # 3. Top-K Selection
    loads = flat_loads.view(num_layers, num_candidates, num_packs)
    imbalance = loads.max(dim=-1).values - loads.min(dim=-1).values

    _, best_k_indices = imbalance.topk(num_refine, dim=1, largest=False)

    layer_offsets = (torch.arange(num_layers, device=device) * num_candidates).unsqueeze(1)
    flat_selected = (layer_offsets + best_k_indices).flatten()

    refined_weights = flat_weights[flat_selected]
    refined_ids = flat_ids[flat_selected]
    refined_ranks = flat_ranks[flat_selected]
    refined_loads = flat_loads[flat_selected]

    # 4. Refinement
    refined_ids, refined_ranks, refined_loads = _refine_packing(
        refined_weights, refined_ids, refined_loads, refined_ranks, num_iters=20
    )

    # 5. Final Selection
    loads_final = refined_loads.view(num_layers, num_refine, num_packs)
    imbalance_final = loads_final.max(dim=-1).values - loads_final.min(dim=-1).values
    best_in_k = imbalance_final.argmin(dim=1)

    # 6. Scatter Back
    winner_cand_idx = best_k_indices.gather(1, best_in_k.unsqueeze(1)).squeeze(1)

    winner_flat_idx = (torch.arange(num_layers, device=device) * num_refine) + best_in_k
    final_aligned_ids = refined_ids[winner_flat_idx]
    final_aligned_ranks = refined_ranks[winner_flat_idx]

    idx_view = winner_cand_idx.view(num_layers, 1, 1).expand(-1, 1, num_items)
    final_sorted_idx = all_indices.gather(1, idx_view).squeeze(1)

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
            # Mask inactive
            # We can't use in-place masking easily on density if we want argmax
            # Set density of inactive to -1
            density[~active] = -1.0

            target_idx = density.argmax(dim=-1)

            # Select only active rows for update
            active_rows = rows[active]
            active_targets = target_idx[active]

            logcnt.index_put_((active_rows, active_targets),
                              torch.tensor(1, device=device, dtype=torch.int64),
                              accumulate=True)
            current_sum[active] += 1

    # Over-allocation: Remove from min cost (new load)
    min_diff = int(diff.min().item())
    if min_diff < 0:
        rows = torch.arange(num_layers, device=device)
        for _ in range(abs(min_diff)):
            active = current_sum > num_phy
            if not active.any(): break

            valid = logcnt > 1
            cost = weight / (logcnt - 1).float()
            # Set invalid or inactive to inf
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