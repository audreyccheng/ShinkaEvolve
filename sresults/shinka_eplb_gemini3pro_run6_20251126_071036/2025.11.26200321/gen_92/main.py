# EVOLVE-BLOCK-START
"""
Expert parallelism load balancer (EPLB) for vLLM.

This module implements the core rearrangement algorithm using a
Hybrid Ensemble strategy with Massive Parallelism and Top-K Refinement.
"""
import torch

def _pairwise_lpt_refine(weights: torch.Tensor,
                         pack_ids: torch.Tensor,
                         pack_loads: torch.Tensor,
                         num_iters: int = 10) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Refines packing by iteratively identifying the heaviest and lightest packs,
    pooling their items, and re-distributing them using a Greedy LPT strategy
    into two virtual bins. This effectively completely re-balances the two
    most extreme packs.
    """
    batch_size, num_items = weights.shape
    device = weights.device

    for _ in range(num_iters):
        # 1. Identify Max/Min
        sorted_loads, sorted_pack_idx = pack_loads.sort(dim=1)
        p_max = sorted_pack_idx[:, -1]
        p_min = sorted_pack_idx[:, 0]

        current_diff = sorted_loads[:, -1] - sorted_loads[:, 0]

        # Early exit if perfectly balanced
        if current_diff.max() < 1e-5:
            break

        # 2. Mask items belonging to Max or Min packs
        mask_max = (pack_ids == p_max.unsqueeze(1))
        mask_min = (pack_ids == p_min.unsqueeze(1))
        mask_pair = mask_max | mask_min

        # 3. Extract and Sort Weights
        # We set non-participating items to -1 so they sort to the end
        active_weights = torch.where(mask_pair, weights, torch.tensor(-1.0, device=device))
        sorted_w, sorted_idx = active_weights.sort(dim=1, descending=True)

        # 4. Greedy Allocation into Two Virtual Bins
        l0 = torch.zeros(batch_size, device=device, dtype=weights.dtype)
        l1 = torch.zeros(batch_size, device=device, dtype=weights.dtype)

        # Store decisions: False->Bin0 (p_max), True->Bin1 (p_min)
        decisions = torch.zeros(batch_size, num_items, dtype=torch.bool, device=device)

        for k in range(num_items):
            w = sorted_w[:, k]
            valid = (w > -0.5)
            if not valid.any():
                break

            # Assign to lighter bin
            choice = (l1 < l0) # True -> 1

            w_masked = torch.where(valid, w, torch.tensor(0.0, device=device))
            l0 = torch.where(valid & (~choice), l0 + w_masked, l0)
            l1 = torch.where(valid & choice, l1 + w_masked, l1)

            decisions[:, k] = choice

        # 5. Check Improvement
        new_max = torch.maximum(l0, l1)
        new_min = torch.minimum(l0, l1)
        new_diff = new_max - new_min

        improve_mask = new_diff < (current_diff - 1e-5)

        if not improve_mask.any():
            break

        # 6. Apply Updates
        # Only update batches that improved
        active = improve_mask
        if not active.any():
            continue

        active_idx = torch.arange(batch_size, device=device)[active]

        # Prepare scatter indices
        # We need to map sorted_idx back to global positions for pack_ids
        sub_sorted_idx = sorted_idx[active] # [B_sub, N]
        sub_decisions = decisions[active]   # [B_sub, N]
        sub_weights = sorted_w[active]

        sub_p_max = p_max[active].unsqueeze(1)
        sub_p_min = p_min[active].unsqueeze(1)

        # Map choice 0->p_max, 1->p_min
        target_packs = torch.where(sub_decisions, sub_p_min, sub_p_max)

        # Scatter update
        valid_items = (sub_weights > -0.5)

        flat_b = active_idx.unsqueeze(1).expand(-1, num_items)

        # Flat indices for pack_ids [B*N]
        flat_dest = (flat_b * num_items + sub_sorted_idx).flatten()
        flat_src = target_packs.flatten()
        flat_mask = valid_items.flatten()

        final_dest = flat_dest[flat_mask]
        final_src = flat_src[flat_mask]

        pack_ids.view(-1).scatter_(0, final_dest, final_src)

        # Update loads
        pack_loads[active, p_max[active]] = l0[active]
        pack_loads[active, p_min[active]] = l1[active]

    return pack_ids, pack_loads


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
    Pack n weighted objects to m packs using Hybrid Ensemble Strategy with
    Pairwise LPT Refinement.
    """
    num_layers, num_items = weight.shape
    device = weight.device
    capacity = num_items // num_packs
    num_candidates = 128

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

    # C. Random Shuffles (62)
    rand_keys = torch.randn(num_layers, 62, num_items, device=device)
    _, c_random = rand_keys.sort(dim=-1)

    # D. Noisy LPT (64)
    noise = torch.rand(num_layers, 64, num_items, device=device) * 0.4 + 0.8
    noisy_w = weight.unsqueeze(1) * noise
    _, c_noisy = noisy_w.sort(dim=-1, descending=True)

    all_indices = torch.cat([c_lpt, c_zigzag, c_random, c_noisy], dim=1)

    # Gather weights
    expanded_weight = weight.unsqueeze(1).expand(-1, num_candidates, -1)
    ordered_weights = expanded_weight.gather(2, all_indices)

    # Flatten
    flat_weights = ordered_weights.view(-1, num_items)

    # 2. Greedy Packing
    flat_ids, _, flat_loads = _vectorized_greedy_packing(flat_weights, num_packs, capacity)

    # 3. Top-K Selection
    loads = flat_loads.view(num_layers, num_candidates, num_packs)
    imbalance = loads.max(dim=-1).values - loads.min(dim=-1).values
    std_dev = loads.std(dim=-1)

    # Score: Imbalance + 1e-4 * StdDev (to break ties)
    score = imbalance + 1e-4 * std_dev

    k = 8
    _, best_k_indices = score.topk(k, dim=1, largest=False)

    # Extract Top-K
    layer_offsets = (torch.arange(num_layers, device=device) * num_candidates).unsqueeze(1)
    global_indices = (layer_offsets + best_k_indices).view(-1)

    refined_weights = flat_weights[global_indices]
    refined_ids = flat_ids[global_indices]
    refined_loads = flat_loads[global_indices]

    # 4. Refinement
    refined_ids, refined_loads = _pairwise_lpt_refine(
        refined_weights, refined_ids, refined_loads, num_iters=10
    )

    # 5. Final Selection
    final_imbalance = refined_loads.max(dim=1).values - refined_loads.min(dim=1).values
    final_std = refined_loads.std(dim=1)
    final_score = final_imbalance + 1e-4 * final_std
    final_score = final_score.view(num_layers, k)

    best_in_k = final_score.argmin(dim=1)

    # 6. Reconstruct Winner
    winner_cand_idx = best_k_indices.gather(1, best_in_k.unsqueeze(1)).squeeze(1)

    idx_view = winner_cand_idx.view(num_layers, 1, 1).expand(-1, 1, num_items)
    final_sorted_idx = all_indices.gather(1, idx_view).squeeze(1)

    winner_flat_idx = (torch.arange(num_layers, device=device) * k) + best_in_k
    final_aligned_ids = refined_ids[winner_flat_idx]

    # 7. Compute Ranks for Winner
    # Since we dropped ranks during refinement, we compute them now
    final_aligned_ranks = torch.zeros_like(final_aligned_ids)
    pack_counters = torch.zeros((num_layers, num_packs), dtype=torch.long, device=device)
    row_indices = torch.arange(num_layers, device=device)

    for i in range(num_items):
        p = final_aligned_ids[:, i]
        final_aligned_ranks[:, i] = pack_counters[row_indices, p]
        pack_counters[row_indices, p] += 1

    # Scatter back
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