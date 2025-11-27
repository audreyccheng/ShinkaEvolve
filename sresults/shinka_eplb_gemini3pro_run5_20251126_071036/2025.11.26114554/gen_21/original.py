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
    Iteratively refines the packing by swapping items between any two packs
    to reduce the variance of pack weights.

    Args:
        weights: [num_groups] weights of items
        pack_indices: [num_packs, groups_per_pack] indices of items in each pack
        pack_weights: [num_packs] total weight of each pack

    Returns:
        pack_indices: Refined item assignments
    """
    N = num_packs * groups_per_pack
    # Precompute mask for same-pack swaps
    pack_ids = torch.arange(num_packs, device=weights.device).repeat_interleave(groups_per_pack)
    # Mask [N, N] where pack_ids[i] == pack_ids[j]
    mask = pack_ids.view(-1, 1) == pack_ids.view(1, -1)

    for _ in range(max_iters):
        # Flatten indices to [N]
        flat_indices = pack_indices.view(-1)
        w_flat = weights[flat_indices]  # [N]
        l_flat = pack_weights[pack_ids] # [N] (broadcast pack weights to items)

        # Calculate Delta matrix: D[i, j] = w[i] - w[j]
        # Item i is at flat index i, currently in pack pack_ids[i]
        # Item j is at flat index j, currently in pack pack_ids[j]
        # We consider swapping item i into j's pack and item j into i's pack
        D = w_flat.view(-1, 1) - w_flat.view(1, -1)

        # Calculate Load Diff matrix: L_diff[i, j] = L[pack(j)] - L[pack(i)]
        L_diff = l_flat.view(1, -1) - l_flat.view(-1, 1)

        # Change in variance = 2 * D * (L_diff + D)
        # We want to minimize this change.
        change = D * (L_diff + D)

        # Mask invalid swaps (same pack)
        change.masked_fill_(mask, float('inf'))

        # Find best swap
        min_val, min_idx = torch.min(change.view(-1), 0)

        if min_val > -1e-6:
            break

        # Decode indices
        idx_i = min_idx // N
        idx_j = min_idx % N

        p1 = (idx_i // groups_per_pack).item()
        g1 = (idx_i % groups_per_pack).item()

        p2 = (idx_j // groups_per_pack).item()
        g2 = (idx_j % groups_per_pack).item()

        # Execute swap
        item1 = pack_indices[p1, g1].item()
        item2 = pack_indices[p2, g2].item()

        pack_indices[p1, g1] = item2
        pack_indices[p2, g2] = item1

        # Update weights
        delta = D[idx_i, idx_j].item()
        pack_weights[p1] -= delta
        pack_weights[p2] += delta

    return pack_indices


def balanced_packing(weight: torch.Tensor,
                     num_packs: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pack n weighted objects to m packs, such that each bin contains exactly
    n/m objects and the weights of all packs are as balanced as possible.

    Implements a Greedy LPT initialization followed by Iterative Swapping refinement.

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

    # Pre-allocate outputs
    pack_index = torch.full_like(weight, -1, dtype=torch.int64)
    rank_in_pack = torch.full_like(weight, -1, dtype=torch.int64)

    # Sort weights for Greedy LPT (Longest Processing Time) initialization
    # Sorting descending helps placing largest items first
    sorted_res = weight.sort(dim=-1, descending=True)
    sorted_indices = sorted_res.indices
    sorted_weights = sorted_res.values

    # Pre-compute grid indices for scattering results later
    p_ids_grid = torch.arange(num_packs, device=device).unsqueeze(1).expand(-1, groups_per_pack)
    r_ids_grid = torch.arange(groups_per_pack, device=device).unsqueeze(0).expand(num_packs, -1)
    flat_p_ids = p_ids_grid.flatten()
    flat_r_ids = r_ids_grid.flatten()

    for i in range(num_layers):
        # 1. Greedy Initialization
        current_pack_weights = torch.zeros(num_packs, device=device, dtype=weight.dtype)
        current_pack_counts = torch.zeros(num_packs, device=device, dtype=torch.int64)

        # Matrix to store assignments: [Pack, Slot] -> Item Index
        pack_assignment = torch.zeros((num_packs, groups_per_pack),
                                      dtype=torch.int64, device=device)

        layer_indices = sorted_indices[i]
        layer_vals = sorted_weights[i]

        for j in range(num_groups):
            w = layer_vals[j]
            item_idx = layer_indices[j]

            # Vectorized greedy choice: choose the valid pack with min weight
            # Mask out full packs by setting their weight to infinity
            is_full = current_pack_counts >= groups_per_pack
            masked_weights = torch.where(is_full, float('inf'), current_pack_weights)
            best_pack = torch.argmin(masked_weights)

            # Assign
            slot = current_pack_counts[best_pack]
            pack_assignment[best_pack, slot] = item_idx
            current_pack_weights[best_pack] += w
            current_pack_counts[best_pack] += 1

        # 2. Iterative Refinement (Swapping)
        pack_assignment = _refine_packing(
            weight[i], pack_assignment, current_pack_weights,
            num_packs, groups_per_pack
        )

        # 3. Store results
        flat_items = pack_assignment.flatten()
        pack_index[i, flat_items] = flat_p_ids
        rank_in_pack[i, flat_items] = flat_r_ids

    return pack_index, rank_in_pack


def replicate_experts(
        weight: torch.Tensor,
        num_phy: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Replicate experts to `num_phy` replicas, minimizing the maximum load.

    Uses a vectorized greedy approach (Jefferson/D'Hondt method) which is
    optimal for the min-max load objective with discrete allocations.

    Parameters:
        weight: [X, num_log]
        num_phy: total number of experts after replication

    Returns:
        phy2log: [X, num_phy], logical expert id of each physical expert
        rank: [X, num_phy], the replica rank
        logcnt: [X, num_log], number of replicas for each logical expert
    """
    n, num_log = weight.shape
    device = weight.device

    # Initialize: Each expert gets at least 1 replica
    logcnt = torch.ones((n, num_log), dtype=torch.int64, device=device)

    # Initialize phy2log with basic mapping [0, 1, ..., num_log-1] for the first part
    # We will overwrite the redundant parts in the loop
    phy2log = torch.arange(num_phy, dtype=torch.int64, device=device).repeat(n, 1)
    rank = torch.zeros((n, num_phy), dtype=torch.int64, device=device)

    arangen = torch.arange(n, dtype=torch.int64, device=device)

    # Greedily assign remaining replicas
    # We need to assign `num_phy - num_log` replicas.
    # In each step, assign to the expert with the highest current load per replica.
    # This loop is vectorized over layers (dim 0).
    for i in range(num_log, num_phy):
        # Score is current load per replica
        scores = weight / logcnt
        # Find expert with max score in each layer
        indices = torch.argmax(scores, dim=-1)

        # Assign the new replica
        phy2log[:, i] = indices

        # Record the rank for this new replica
        rank[:, i] = logcnt[arangen, indices]

        # Increment replica count
        logcnt[arangen, indices] += 1

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
    # Sum weights within each group
    tokens_per_group = weight.unflatten(-1, (num_groups, group_size)).sum(-1)

    # Use improved balanced packing with swapping
    group_pack_index, group_rank_in_pack = balanced_packing(
        tokens_per_group, num_nodes)

    log2mlog = (((group_pack_index * groups_per_node + group_rank_in_pack) *
                 group_size).unsqueeze(-1) +
                torch.arange(group_size,
                             dtype=torch.int64,
                             device=group_pack_index.device)).flatten(-2)
    mlog2log = inverse(log2mlog)

    # Step 2: construct redundant experts within nodes
    # [num_layers * num_nodes, num_logical_experts // num_nodes]
    tokens_per_mlog = weight.gather(-1, mlog2log).view(
        -1, num_logical_experts // num_nodes)
    phy2mlog, phyrank, mlogcnt = replicate_experts(
        tokens_per_mlog, num_physical_experts // num_nodes)

    # Step 3: pack physical_experts to GPUs
    # [num_layers * num_nodes, num_physical_experts // num_nodes]
    tokens_per_phy = (tokens_per_mlog / mlogcnt).gather(-1, phy2mlog)

    # Use improved balanced packing here as well
    pack_index, rank_in_pack = balanced_packing(tokens_per_phy,
                                                num_gpus // num_nodes)

    phy2pphy = pack_index * phy_experts_per_gpu + rank_in_pack
    pphy2phy = inverse(phy2pphy)

    pphy2mlog = phy2mlog.gather(
        -1, pphy2phy)  # [num_layers * num_nodes, num_log_per_nodes]
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

    # Dispatch policy
    if num_groups % num_nodes == 0:
        # use hierarchical load-balance policy
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, num_groups, num_nodes, num_gpus)
    else:
        # use global load-balance policy
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, 1, 1, num_gpus)

    num_redundant_experts = num_replicas - num_logical_experts
    maxlogcnt = num_redundant_experts + 1

    # Construct reverse mapping log2phy
    log2phy: torch.Tensor = torch.full(
        (num_layers, num_logical_experts, maxlogcnt),
        -1,
        dtype=torch.int64,
        device=logcnt.device,
    )

    # Efficient scatter
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