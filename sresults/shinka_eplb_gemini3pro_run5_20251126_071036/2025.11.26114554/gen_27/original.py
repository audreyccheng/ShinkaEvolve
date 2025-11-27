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
    Vectorized refinement step that minimizes load variance by checking pairwise swaps.
    Operates on a single layer's data.

    Args:
        weights: [num_groups] weights of items
        pack_indices: [num_packs, groups_per_pack] item assignments
        pack_weights: [num_packs] current load of each pack
    """
    device = weights.device
    num_items_total = num_packs * groups_per_pack

    # Pre-compute pack identifiers for every item slot in the flattened view
    # item_to_pack_id[k] tells which pack the k-th item in flattened view belongs to
    item_to_pack_id = torch.arange(num_packs, device=device).unsqueeze(1).expand(-1, groups_per_pack).reshape(-1)

    # Mask to prevent swapping within same pack [N, N]
    same_pack_mask = item_to_pack_id.unsqueeze(0) == item_to_pack_id.unsqueeze(1)

    for _ in range(max_iters):
        # Flatten current items to [N]
        current_items = pack_indices.view(-1)
        w_current = weights[current_items]

        # Load of the pack each item currently belongs to
        l_current = pack_weights[item_to_pack_id]

        # Delta matrix D[i, j] = w[i] - w[j]
        # This represents the net weight change for Pack(i) if we swap item i OUT
        # and item j IN. (Note: Pack(i) loses w[i], gains w[j], so net change is w[j]-w[i] = -D[i,j])
        # Let's align with math:
        # Swap i (from P_i) and j (from P_j).
        # New P_i load = L_i - w_i + w_j = L_i - D[i,j]
        # New P_j load = L_j - w_j + w_i = L_j + D[i,j]
        D = w_current.unsqueeze(1) - w_current.unsqueeze(0) # [N, N]

        # Load Diff matrix L_diff[i, j] = L_j - L_i
        L_diff = l_current.unsqueeze(0) - l_current.unsqueeze(1)

        # We want to minimize Sum of Squared Loads.
        # Change = (L_i - D)^2 + (L_j + D)^2 - (L_i^2 + L_j^2)
        #        = -2*L_i*D + D^2 + 2*L_j*D + D^2
        #        = 2*D*(L_j - L_i + D)
        change = 2 * D * (L_diff + D)

        # Apply mask
        change.masked_fill_(same_pack_mask, float('inf'))

        # Find best swap
        min_val, flat_idx = torch.min(change.view(-1), dim=0)

        if min_val > -1e-6:
            break

        idx_i = flat_idx // num_items_total
        idx_j = flat_idx % num_items_total

        # Decode indices
        pid_i = item_to_pack_id[idx_i]
        pid_j = item_to_pack_id[idx_j]

        slot_i = idx_i % groups_per_pack
        slot_j = idx_j % groups_per_pack

        # Execute swap
        item_i = pack_indices[pid_i, slot_i].item()
        item_j = pack_indices[pid_j, slot_j].item()

        pack_indices[pid_i, slot_i] = item_j
        pack_indices[pid_j, slot_j] = item_i

        # Update weights
        delta = D[idx_i, idx_j].item()
        pack_weights[pid_i] -= delta
        pack_weights[pid_j] += delta

    return pack_indices


def balanced_packing(weight: torch.Tensor,
                     num_packs: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pack n weighted objects to m packs, such that each bin contains exactly
    n/m objects and the weights of all packs are as balanced as possible.

    Uses Snake-Sort Initialization followed by Variance-Minimizing Refinement.

    Parameters:
        weight: [X, n], the weight of each item
        num_packs: number of packs

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

    # 1. Snake Sort Initialization (Vectorized)
    # Sort weights descending
    sorted_res = weight.sort(dim=-1, descending=True)
    sorted_weights = sorted_res.values
    sorted_indices = sorted_res.indices

    # Construct snake pattern mapping for ranks
    # Rank r goes to pack p(r).
    # Pattern 0..M-1, M-1..0, repeated
    cycle_len = 2 * num_packs
    ranks = torch.arange(num_groups, device=device)
    mod_ranks = ranks % cycle_len

    # Calculate Pack IDs for each rank
    # if mod < M: p = mod; else: p = 2M - 1 - mod
    pack_ids_by_rank = torch.where(mod_ranks < num_packs, mod_ranks, cycle_len - 1 - mod_ranks)

    # Calculate Slot IDs for each rank
    # Each full cycle consumes 2 slots per pack.
    # slot = (cycle_idx * 2) + (1 if in second half of cycle else 0)
    slot_indices = (ranks // cycle_len) * 2 + (mod_ranks >= num_packs).long()

    # Determine the flattened index in the destination [L, M, C] array
    # dest_idx = pack_id * C + slot_id
    flat_dest_idx = pack_ids_by_rank * groups_per_pack + slot_indices

    # Create the packed assignment structure [L, M, C]
    # We map items from sorted_indices to this structure
    pack_assignment = torch.zeros((num_layers, num_packs, groups_per_pack),
                                  dtype=torch.int64, device=device)

    # Scatter sorted items to their snake-assigned positions
    # pack_assignment.view(L, -1)[:, flat_dest_idx[r]] = sorted_indices[:, r]
    pack_assignment.view(num_layers, -1)[:, flat_dest_idx] = sorted_indices

    # Calculate initial pack weights
    pack_weights = torch.zeros((num_layers, num_packs), dtype=weight.dtype, device=device)
    # We can sum up using index_add_ or scatter_add_
    # Expand pack_ids to [L, N]
    pack_ids_expanded = pack_ids_by_rank.unsqueeze(0).expand(num_layers, -1)
    pack_weights.scatter_add_(1, pack_ids_expanded, sorted_weights)

    # 2. Refinement Loop
    # Process each layer
    for i in range(num_layers):
        pack_assignment[i] = _refine_packing(
            weight[i],
            pack_assignment[i],
            pack_weights[i],
            num_packs,
            groups_per_pack,
            max_iters=50
        )

    # 3. Reconstruct Outputs
    pack_index = torch.empty_like(weight, dtype=torch.int64)
    rank_in_pack = torch.empty_like(weight, dtype=torch.int64)

    # flat_assignment [L, N] contains item IDs
    flat_assignment = pack_assignment.view(num_layers, -1)

    # Map back: We know the Pack ID and Slot ID for each position in flat_assignment
    # Position k corresponds to pack k // C, slot k % C
    grid = torch.arange(num_groups, device=device)
    p_vals = (grid // groups_per_pack).unsqueeze(0).expand(num_layers, -1)
    c_vals = (grid % groups_per_pack).unsqueeze(0).expand(num_layers, -1)

    # pack_index[layer, item_id] = p_val
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