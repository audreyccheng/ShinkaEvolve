# EVOLVE-BLOCK-START
"""
Expert parallelism load balancer (EPLB) for vLLM.

This module implements the core rearrangement algorithm using a
Hybrid Ensemble strategy with Massive Parallelism and Top-K Refinement.
"""
import torch

def _refine_packing(weights: torch.Tensor,
                    pack_ids: torch.Tensor,
                    pack_loads: torch.Tensor,
                    ranks: torch.Tensor,
                    num_iters: int = 20) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Refines packing by attempting pairwise swaps between the heaviest pack and
    ANY other pack to strictly reduce the maximum load.
    """
    batch_size, num_items = weights.shape
    device = weights.device
    num_packs = pack_loads.shape[1]

    # Precompute pairwise weight differences: [B, N, N]
    # w_diff[b, i, j] = w[b, i] - w[b, j]
    w_diff = weights.unsqueeze(2) - weights.unsqueeze(1)

    batch_range = torch.arange(batch_size, device=device)

    for _ in range(num_iters):
        # Identify Max Pack
        max_vals, max_indices = pack_loads.max(dim=1) # [B]

        # Track best swap per batch
        best_gain = torch.full((batch_size,), -1.0, device=device, dtype=weights.dtype)
        best_flat_idx = torch.zeros(batch_size, dtype=torch.long, device=device)
        best_target_p = torch.full((batch_size,), -1, dtype=torch.long, device=device)

        # Mask for items in Max Pack [B, N]
        mask_max = (pack_ids == max_indices.unsqueeze(1))

        # Iterate over all possible target packs
        for p in range(num_packs):
            # Optimization: Skip if no batch has max_pack != p
            # (Basically we iterate p from 0..M-1. If for some batch Max==p, we skip for that batch)
            valid_batch = (max_indices != p)
            if not valid_batch.any():
                continue

            target_vals = pack_loads[:, p]
            load_diff = max_vals - target_vals

            # Gain = min(delta, gap - delta)
            # gap = Max - Target
            # delta = w_i - w_j

            delta = w_diff

            # Mask Target Pack [B, N]
            mask_target = (pack_ids == p)

            # Valid Swap: i in Max, j in Target
            valid_swap = mask_max.unsqueeze(2) & mask_target.unsqueeze(1)
            valid_swap &= valid_batch.view(-1, 1, 1)

            gap = load_diff.view(-1, 1, 1)
            gain = torch.min(delta, gap - delta)

            # Filter for positive gain
            valid_swap &= (gain > 1e-5)

            gain = torch.where(valid_swap, gain, torch.tensor(-1.0, device=device, dtype=weights.dtype))

            # Max gain for this target pack p
            flat_gain = gain.view(batch_size, -1)
            p_max_gain, p_max_idx = flat_gain.max(dim=1)

            # Update best across packs
            improve = p_max_gain > best_gain
            if improve.any():
                best_gain = torch.where(improve, p_max_gain, best_gain)
                best_flat_idx = torch.where(improve, p_max_idx, best_flat_idx)
                best_target_p = torch.where(improve, torch.tensor(p, device=device), best_target_p)

        # Apply best swap found
        active = best_gain > 1e-5
        if not active.any():
            break

        active_batch = batch_range[active]
        active_flat = best_flat_idx[active]
        p_target = best_target_p[active]
        p_max = max_indices[active]

        i_idx = active_flat // num_items
        j_idx = active_flat % num_items

        w_i = weights[active_batch, i_idx]
        w_j = weights[active_batch, j_idx]
        delta_val = w_i - w_j

        pack_loads[active_batch, p_max] -= delta_val
        pack_loads[active_batch, p_target] += delta_val

        pack_ids[active_batch, i_idx] = p_target
        pack_ids[active_batch, j_idx] = p_max

        r_i = ranks[active_batch, i_idx]
        r_j = ranks[active_batch, j_idx]
        ranks[active_batch, i_idx] = r_j
        ranks[active_batch, j_idx] = r_i

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
    Pack n weighted objects to m packs using a Hybrid Ensemble Strategy with Top-K Refinement.
    """
    num_layers, num_items = weight.shape
    device = weight.device
    capacity = num_items // num_packs

    num_candidates = 256

    # 1. Candidate Generation
    # A. LPT
    lpt_val, lpt_idx = weight.sort(dim=-1, descending=True)
    c_lpt = lpt_idx.unsqueeze(1) # [L, 1, N]

    # B. ZigZag
    zigzag_perm = torch.empty(num_items, device=device, dtype=torch.long)
    half = (num_items + 1) // 2
    arange = torch.arange(num_items, device=device)
    zigzag_perm[0::2] = arange[:half]
    zigzag_perm[1::2] = arange[half:].flip(0)
    c_zigzag = lpt_idx.gather(1, zigzag_perm.unsqueeze(0).expand(num_layers, -1)).unsqueeze(1)

    # C. Random Shuffles
    num_random = 64
    rand_keys = torch.randn(num_layers, num_random, num_items, device=device)
    _, c_random = rand_keys.sort(dim=-1)

    # D. Noisy LPT
    num_noisy = num_candidates - 2 - num_random # 190
    noise = torch.rand(num_layers, num_noisy, num_items, device=device) * 0.5 + 0.75
    noisy_w = weight.unsqueeze(1) * noise
    _, c_noisy = noisy_w.sort(dim=-1, descending=True)

    all_indices = torch.cat([c_lpt, c_zigzag, c_random, c_noisy], dim=1)

    # Gather weights for kernel
    expanded_weight = weight.unsqueeze(1).expand(-1, num_candidates, -1)
    ordered_weights = expanded_weight.gather(2, all_indices)

    # Flatten
    flat_weights = ordered_weights.view(-1, num_items)

    # 2. Greedy Packing (Batched)
    flat_ids, flat_ranks, flat_loads = _vectorized_greedy_packing(flat_weights, num_packs, capacity)

    # 3. Top-K Selection with Hierarchical Scoring
    loads = flat_loads.view(num_layers, num_candidates, num_packs)

    # Metric 1: Max Load (Minimize)
    max_loads = loads.max(dim=-1).values
    # Metric 2: L2 Norm (Minimize, tie-breaker)
    l2_norms = loads.pow(2).sum(dim=-1)

    num_refine = 16

    # Step 3a: Pre-select top 4*K candidates by Max Load
    k_pre = num_refine * 4
    _, top_pre_indices = max_loads.topk(k_pre, dim=1, largest=False) # [L, 4K]

    # Step 3b: Select top K candidates by L2 Norm from the pre-selected set
    l2_subset = l2_norms.gather(1, top_pre_indices) # [L, 4K]
    _, top_k_local = l2_subset.topk(num_refine, dim=1, largest=False) # [L, K]

    best_k_indices = top_pre_indices.gather(1, top_k_local) # [L, K]

    # Extract data for these Top K
    layer_offsets = (torch.arange(num_layers, device=device) * num_candidates).unsqueeze(1)
    global_indices = (layer_offsets + best_k_indices).view(-1)

    refined_weights = flat_weights[global_indices]
    refined_ids = flat_ids[global_indices]
    refined_ranks = flat_ranks[global_indices]
    refined_loads = flat_loads[global_indices]

    # 4. Refinement on Top K (Max-Any Swap)
    refined_ids, refined_ranks, refined_loads = _refine_packing(
        refined_weights, refined_ids, refined_loads, refined_ranks, num_iters=20
    )

    # 5. Final Selection
    # Again using Max Load then L2
    refined_loads_view = refined_loads.view(num_layers, num_refine, num_packs)
    rmax = refined_loads_view.max(dim=-1).values
    rl2 = refined_loads_view.pow(2).sum(dim=-1)

    # Lexicographical min: rmax first, then rl2
    # score = rmax + rl2 * 1e-9 (assuming rmax dominate)
    score = rmax + rl2 * 1e-9
    best_in_k = score.argmin(dim=1) # [L]

    # 6. Gather and Scatter Back
    winner_cand_idx = best_k_indices.gather(1, best_in_k.unsqueeze(1)).squeeze(1) # [L]

    idx_view = winner_cand_idx.view(num_layers, 1, 1).expand(-1, 1, num_items)
    final_sorted_idx = all_indices.gather(1, idx_view).squeeze(1) # [L, N]

    winner_flat_idx = (torch.arange(num_layers, device=device) * num_refine) + best_in_k
    final_aligned_ids = refined_ids[winner_flat_idx]
    final_aligned_ranks = refined_ranks[winner_flat_idx]

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

    # Under-allocation
    max_diff = int(diff.max().item())
    if max_diff > 0:
        rows = torch.arange(num_layers, device=device)
        for _ in range(max_diff):
            active = current_sum < num_phy
            if not active.any(): break

            density = weight / logcnt.float()
            # mask inactive rows to avoid selecting them
            density[~active] = -1.0
            target_idx = density.argmax(dim=-1)

            active_rows = rows[active]
            active_targets = target_idx[active]

            logcnt.index_put_((active_rows, active_targets),
                              torch.tensor(1, device=device, dtype=torch.int64),
                              accumulate=True)
            current_sum[active] += 1

    # Over-allocation
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