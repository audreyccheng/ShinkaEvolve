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


def _refine_packing_batched(weights: torch.Tensor,
                            pack_indices: torch.Tensor,
                            pack_weights: torch.Tensor,
                            num_packs: int,
                            groups_per_pack: int,
                            max_iters: int = 20) -> torch.Tensor:
    """
    Batched version of refinement.

    Args:
        weights: [B, N] weights of items (N = num_packs * groups_per_pack)
        pack_indices: [B, num_packs, groups_per_pack] item assignments
        pack_weights: [B, num_packs] current load of each pack
    """
    B, N = weights.shape
    device = weights.device

    # Pre-compute pack identifiers for every item slot in the flattened view
    item_to_pack_id = torch.arange(num_packs, device=device).repeat_interleave(groups_per_pack)

    # Mask to prevent swapping within same pack [N, N]
    same_pack_mask = item_to_pack_id.unsqueeze(0) == item_to_pack_id.unsqueeze(1)

    # Flatten pack_indices to [B, N]
    current_items = pack_indices.view(B, N)

    for _ in range(max_iters):
        # Gather weights of currently assigned items: [B, N]
        w_current = torch.gather(weights, 1, current_items) # [B, N]

        # Load of the pack each item currently belongs to
        l_current = pack_weights[:, item_to_pack_id] # [B, N]

        # Delta matrix D[b, i, j] = w[b, i] - w[b, j]
        # [B, N, 1] - [B, 1, N] -> [B, N, N]
        D = w_current.unsqueeze(2) - w_current.unsqueeze(1)

        # Load Diff matrix L_diff[b, i, j] = L_j - L_i
        L_diff = l_current.unsqueeze(1) - l_current.unsqueeze(2)

        # Change cost = 2 * D * (L_diff + D)
        change = 2 * D * (L_diff + D)

        # Apply mask
        change.masked_fill_(same_pack_mask.unsqueeze(0), float('inf'))

        # Find best swap per batch
        min_val, flat_idx = torch.min(change.view(B, -1), dim=1) # [B]

        # Mask for batches that have a beneficial swap
        valid_mask = min_val < -1e-5
        if not valid_mask.any():
            break

        # Select valid batch indices
        batch_indices = torch.nonzero(valid_mask).squeeze(1)

        # Indices in [N]
        idx_i = flat_idx[batch_indices] // N
        idx_j = flat_idx[batch_indices] % N

        # Get actual items to swap
        # current_items: [B, N]
        # We need to perform row-wise gather/scatter only on valid batches
        valid_items = current_items[batch_indices] # [V, N]

        item_i = valid_items.gather(1, idx_i.unsqueeze(1)).squeeze(1)
        item_j = valid_items.gather(1, idx_j.unsqueeze(1)).squeeze(1)

        # Execute swap in item tensor
        # In-place update for the selected batches
        valid_items.scatter_(1, idx_i.unsqueeze(1), item_j.unsqueeze(1))
        valid_items.scatter_(1, idx_j.unsqueeze(1), item_i.unsqueeze(1))
        current_items[batch_indices] = valid_items

        # Update weights
        pid_i = item_to_pack_id[idx_i] # [V]
        pid_j = item_to_pack_id[idx_j] # [V]

        # Extract delta
        # D is [B, N, N]. We need D[batch_indices, idx_i, idx_j]
        valid_D = D[batch_indices] # [V, N, N]
        delta = valid_D.view(len(batch_indices), -1).gather(1, (idx_i * N + idx_j).unsqueeze(1)).squeeze(1)

        valid_weights = pack_weights[batch_indices] # [V, P]
        valid_weights.scatter_add_(1, pid_i.unsqueeze(1), -delta.unsqueeze(1))
        valid_weights.scatter_add_(1, pid_j.unsqueeze(1), delta.unsqueeze(1))
        pack_weights[batch_indices] = valid_weights

    return pack_indices.view(B, num_packs, groups_per_pack)


def balanced_packing(weight: torch.Tensor,
                     num_packs: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pack n weighted objects to m packs, such that each bin contains exactly
    n/m objects and the weights of all packs are as balanced as possible.

    Uses Parallel Randomized Snake-Sort Initialization followed by Batched Refinement.

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

    if groups_per_pack == 1:
        pack_index = torch.arange(num_groups, dtype=torch.int64,
                                  device=device).expand(num_layers, -1)
        rank_in_pack = torch.zeros_like(weight, dtype=torch.int64)
        return pack_index, rank_in_pack

    # Parameters
    num_candidates = 8

    # 1. Expand and Perturb
    # [L, 1, N]
    w_expanded = weight.unsqueeze(1).expand(-1, num_candidates, -1)

    # Noise: [L, K, N]. K=0 is clean. K>0 has noise.
    noise = torch.rand(num_layers, num_candidates - 1, num_groups, device=device) * 0.02 + 0.99
    # Prefix with ones
    ones = torch.ones(num_layers, 1, num_groups, device=device)
    noise = torch.cat([ones, noise], dim=1)

    w_perturbed = w_expanded * noise

    # 2. Sort and Initialize (Snake Pattern)
    # [L, K, N]
    sorted_res = w_perturbed.sort(dim=-1, descending=True)
    sorted_indices = sorted_res.indices

    # Snake pattern mapping: Rank r -> Pack p
    # [N]
    ranks = torch.arange(num_groups, device=device)
    cycle_len = 2 * num_packs
    mod_ranks = ranks % cycle_len
    pack_ids_by_rank = torch.where(mod_ranks < num_packs, mod_ranks, cycle_len - 1 - mod_ranks)

    # Slot indices
    slot_indices = (ranks // cycle_len) * 2 + (mod_ranks >= num_packs).long()

    # Flatten destination index: p * C + s
    flat_dest_idx = pack_ids_by_rank * groups_per_pack + slot_indices

    # Assign items to packs based on rank
    # assignments: [L, K, P*G]
    assignments = torch.zeros((num_layers, num_candidates, num_groups), dtype=torch.int64, device=device)

    # Scatter sorted_indices into positions
    dest_expanded = flat_dest_idx.view(1, 1, -1).expand(num_layers, num_candidates, -1)
    assignments.scatter_(2, dest_expanded, sorted_indices)

    # Reshape to [L, K, P, G] -> Flatten to [B, P, G]
    B = num_layers * num_candidates
    assignments_flat = assignments.view(B, num_packs, groups_per_pack)

    # Compute initial pack weights
    # We need actual weights, not perturbed ones for accurate refinement
    w_real = weight.unsqueeze(1).expand(-1, num_candidates, -1).reshape(B, -1)

    # Pack weights: [B, P]
    w_assigned = torch.gather(w_real, 1, assignments_flat.view(B, -1)) # [B, N]
    w_assigned = w_assigned.view(B, num_packs, groups_per_pack)
    pack_weights = w_assigned.sum(dim=2) # [B, P]

    # 3. Batched Refinement
    refined_assignments = _refine_packing_batched(
        w_real, assignments_flat, pack_weights, num_packs, groups_per_pack, max_iters=20
    )

    # 4. Selection
    # Metric: Standard Deviation of pack weights
    scores = pack_weights.std(dim=-1) # [B]

    # Reshape to [L, K]
    scores = scores.view(num_layers, num_candidates)

    # Argmin
    best_k = torch.argmin(scores, dim=1) # [L]

    # Gather best assignments
    # refined_assignments: [B, P, G] -> [L, K, P, G]
    refined_reshaped = refined_assignments.view(num_layers, num_candidates, num_packs, groups_per_pack)

    # Select best K
    # best_k expanded: [L, 1, 1, 1]
    best_assignments = refined_reshaped.gather(1, best_k.view(-1, 1, 1, 1).expand(-1, -1, num_packs, groups_per_pack)).squeeze(1)
    # [L, P, G]

    # 5. Output Construction
    pack_index = torch.empty_like(weight, dtype=torch.int64)
    rank_in_pack = torch.empty_like(weight, dtype=torch.int64)

    flat_assignment = best_assignments.view(num_layers, -1)

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
        scores = weight.float() / logcnt
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

        # Fast rank generation
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