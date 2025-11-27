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


def _refine_packing(weights: torch.Tensor,
                    pack_indices: torch.Tensor,
                    pack_weights: torch.Tensor,
                    num_packs: int,
                    groups_per_pack: int,
                    max_iters: int = 50) -> torch.Tensor:
    """
    Batched vectorized refinement step that minimizes load variance by checking pairwise swaps.
    Operates on a batch of packings in parallel.

    Args:
        weights: [B, num_groups] weights of items (indexed by item_id)
        pack_indices: [B, num_packs, groups_per_pack] item assignments (item_ids)
        pack_weights: [B, num_packs] current load of each pack
    """
    batch_size = weights.shape[0]
    device = weights.device
    num_items_total = num_packs * groups_per_pack

    # Pre-compute pack identifiers for every item slot in the flattened view
    # item_to_pack_id[k] tells which pack the k-th item in flattened view belongs to
    item_to_pack_id = torch.arange(num_packs, device=device).unsqueeze(1).expand(-1, groups_per_pack).reshape(-1)

    # Mask to prevent swapping within same pack [N, N]
    same_pack_mask = item_to_pack_id.unsqueeze(0) == item_to_pack_id.unsqueeze(1)

    batch_arange = torch.arange(batch_size, device=device)

    for _ in range(max_iters):
        # Flatten current items to [B, N]
        current_item_ids = pack_indices.view(batch_size, -1)

        # Get weights: w[b, slot] = weights[b, current_item_ids[b, slot]]
        w_current = weights.gather(1, current_item_ids)

        # Load of the pack each item currently belongs to
        # l_current[b, slot] = pack_weights[b, pack_id[slot]]
        l_current = pack_weights[:, item_to_pack_id]

        # Delta matrix D[b, i, j] = w[b, i] - w[b, j]
        D = w_current.unsqueeze(2) - w_current.unsqueeze(1) # [B, N, N]

        # Load Diff matrix L_diff[b, i, j] = L[b, pack(j)] - L[b, pack(i)]
        L_diff = l_current.unsqueeze(1) - l_current.unsqueeze(2) # [B, N, N]

        # Change in variance (sum of squares)
        # Change = 2 * D * (L_diff + D)
        change = 2 * D * (L_diff + D)

        # Apply mask
        change.masked_fill_(same_pack_mask, float('inf'))

        # Find best swap per batch
        min_val, flat_idx = torch.min(change.view(batch_size, -1), dim=1) # [B]

        # Check convergence
        update_mask = min_val < -1e-6
        if not update_mask.any():
            break

        # Indices to update
        active_batches = batch_arange[update_mask]
        active_indices = flat_idx[update_mask]

        idx_i = active_indices // num_items_total
        idx_j = active_indices % num_items_total

        pid_i = item_to_pack_id[idx_i]
        pid_j = item_to_pack_id[idx_j]

        # Access flattened pack_indices [B, N] to simplify swap
        flat_pack_indices = pack_indices.view(batch_size, -1)

        item_i = flat_pack_indices[active_batches, idx_i]
        item_j = flat_pack_indices[active_batches, idx_j]

        # Execute swap
        flat_pack_indices[active_batches, idx_i] = item_j
        flat_pack_indices[active_batches, idx_j] = item_i

        # Update weights
        delta = D[active_batches, idx_i, idx_j]
        pack_weights[active_batches, pid_i] -= delta
        pack_weights[active_batches, pid_j] += delta

    return pack_indices


def balanced_packing(weight: torch.Tensor,
                     num_packs: int,
                     num_restarts: int = 4) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pack n weighted objects to m packs using Parallel Randomized Greedy LPT
    followed by Variance-Minimizing Refinement on ALL candidates.

    Parameters:
        weight: [X, n], the weight of each item
        num_packs: number of packs
        num_restarts: parameter kept for API compatibility.

    Returns:
        pack_index: [X, n], the pack index of each item
        rank_in_pack: [X, n], the rank of the item in the pack
    """
    num_layers, num_groups = weight.shape
    device = weight.device
    assert num_groups % num_packs == 0
    groups_per_pack = num_groups // num_packs

    # Optimization for trivial case
    if groups_per_pack == 1:
        pack_index = torch.arange(num_groups, dtype=torch.int64,
                                  device=device).expand(num_layers, -1)
        rank_in_pack = torch.zeros_like(weight, dtype=torch.int64)
        return pack_index, rank_in_pack

    # Number of parallel candidates to explore
    num_candidates = 64

    # 1. Parallel Randomized Greedy Initialization
    # We generate multiple candidate packings in parallel by sorting weights with random noise.

    # Expand weights for candidates: [L, R, G]
    B = num_layers * num_candidates

    # Create sort keys
    weights_expanded = weight.unsqueeze(1).expand(-1, num_candidates, -1) # [L, R, G]

    # Noise generation with Linear Schedule
    # candidate 0: 0.0 noise (Pure LPT)
    # candidate N-1: max_noise
    max_noise = 0.15
    noise_levels = torch.linspace(0, max_noise, steps=num_candidates, device=device)
    # [1, R, 1]
    noise_levels = noise_levels.view(1, num_candidates, 1)

    noise = torch.rand_like(weights_expanded) * noise_levels

    sort_keys = weights_expanded * (1.0 + noise)

    # Sort descending
    sorted_res = sort_keys.sort(dim=-1, descending=True)
    sorted_indices = sorted_res.indices # [L, R, G]

    # Flatten to [B, G]
    flat_indices = sorted_indices.view(B, num_groups)
    flat_weights_src = weights_expanded.reshape(B, num_groups)

    # Gather actual weights in sorted order: [B, G]
    flat_sorted_weights = torch.gather(flat_weights_src, 1, flat_indices)

    # Greedy Packing Loop
    pack_loads = torch.zeros((B, num_packs), dtype=weight.dtype, device=device)
    pack_counts = torch.zeros((B, num_packs), dtype=torch.int64, device=device)
    pack_assignment = torch.zeros((B, num_packs, groups_per_pack), dtype=torch.int64, device=device)

    batch_idx = torch.arange(B, device=device)
    inf_tensor = torch.tensor(float('inf'), device=device)

    # Vectorized loop over items
    for j in range(num_groups):
        w = flat_sorted_weights[:, j]
        item_id = flat_indices[:, j]

        # Identify valid packs (not full)
        is_full = (pack_counts >= groups_per_pack)

        # Find min load among valid packs
        masked_loads = torch.where(is_full, inf_tensor, pack_loads)
        best_pack = torch.argmin(masked_loads, dim=1) # [B]

        # Update
        pack_loads[batch_idx, best_pack] += w
        slot = pack_counts[batch_idx, best_pack]
        pack_assignment[batch_idx, best_pack, slot] = item_id
        pack_counts[batch_idx, best_pack] += 1

    # 2. Refinement Loop (Batched on ALL candidates)
    # Now we refine all candidates, not just top-k.
    refined_assignment = _refine_packing(
        flat_weights_src,
        pack_assignment,
        pack_loads,
        num_packs,
        groups_per_pack,
        max_iters=100
    )

    # 3. Final Selection
    # Recalculate max loads after refinement
    final_max_loads, _ = pack_loads.max(dim=1) # [L*K]
    final_max_loads = final_max_loads.view(num_layers, num_candidates)

    best_local_idx = torch.argmin(final_max_loads, dim=1) # [L]

    # Calculate indices into the refined batch
    offsets_refined = torch.arange(num_layers, device=device) * num_candidates
    best_refined_indices = offsets_refined + best_local_idx

    best_assignment = refined_assignment[best_refined_indices] # [L, M, C]

    # 4. Reconstruct Outputs
    pack_index = torch.empty_like(weight, dtype=torch.int64)
    rank_in_pack = torch.empty_like(weight, dtype=torch.int64)

    flat_assignment = best_assignment.view(num_layers, -1)

    grid = torch.arange(num_groups, device=device)
    p_vals = (grid // groups_per_pack).unsqueeze(0).expand(num_layers, -1)
    c_vals = (grid % groups_per_pack).unsqueeze(0).expand(num_layers, -1)

    pack_index.scatter_(1, flat_assignment, p_vals)
    rank_in_pack.scatter_(1, flat_assignment, c_vals)

    return pack_index, rank_in_pack


def replicate_experts(
        weight: torch.Tensor,
        num_phy: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Replicate experts to `num_phy` replicas, minimizing the maximum load.

    Uses a vectorized greedy approach (Jefferson/D'Hondt method).
    """
    n, num_log = weight.shape
    device = weight.device

    # Initialize: Each expert gets at least 1 replica
    logcnt = torch.ones((n, num_log), dtype=torch.int64, device=device)

    # Greedily assign remaining replicas
    for _ in range(num_log, num_phy):
        scores = weight / logcnt
        indices = torch.argmax(scores, dim=-1)
        rows = torch.arange(n, device=device)
        logcnt[rows, indices] += 1

    phy2log = torch.zeros((n, num_phy), dtype=torch.int64, device=device)
    rank = torch.zeros((n, num_phy), dtype=torch.int64, device=device)

    # Reconstruct mappings
    for i in range(n):
        counts = logcnt[i]
        l_ids = torch.repeat_interleave(
            torch.arange(num_log, device=device), counts
        )
        phy2log[i] = l_ids

        curr = 0
        for idx in range(num_log):
            c = counts[idx].item()
            rank[i, curr:curr+c] = torch.arange(c, device=device)
            curr += c

    return phy2log, rank, logcnt


def rebalance_experts_hierarchical(
    weight: torch.Tensor,
    num_physical_experts: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
):
    """
    Hierarchical rebalancing: Groups->Nodes, then Replicas->GPUs.
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
    tokens_per_group = weight.unflatten(-1, (num_groups, group_size)).sum(-1)

    group_pack_index, group_rank_in_pack = balanced_packing(
        tokens_per_group, num_nodes)

    log2mlog = (((group_pack_index * groups_per_node + group_rank_in_pack) *
                 group_size).unsqueeze(-1) +
                torch.arange(group_size,
                             dtype=torch.int64,
                             device=group_pack_index.device)).flatten(-2)
    mlog2log = inverse(log2mlog)

    # Step 2: construct redundant experts within nodes
    tokens_per_mlog = weight.gather(-1, mlog2log).view(
        -1, num_logical_experts // num_nodes)
    phy2mlog, phyrank, mlogcnt = replicate_experts(
        tokens_per_mlog, num_physical_experts // num_nodes)

    # Step 3: pack physical_experts to GPUs
    tokens_per_phy = (tokens_per_mlog / mlogcnt).gather(-1, phy2mlog)

    pack_index, rank_in_pack = balanced_packing(tokens_per_phy,
                                                num_gpus // num_nodes)

    phy2pphy = pack_index * phy_experts_per_gpu + rank_in_pack
    pphy2phy = inverse(phy2pphy)

    pphy2mlog = phy2mlog.gather(-1, pphy2phy)
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
    """
    num_layers, num_logical_experts = weight.shape
    weight = weight.float().cpu()

    if num_groups % num_nodes == 0:
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, num_groups, num_nodes, num_gpus)
    else:
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