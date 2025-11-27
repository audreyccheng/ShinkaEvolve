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
                            max_iters: int = 100) -> torch.Tensor:
    """
    Batched version of refinement that minimizes load variance by checking all pairwise swaps.

    Args:
        weights: [B, N] The weight of every item (global indexing).
        pack_indices: [B, num_packs, groups_per_pack] Item IDs assigned to packs.
        pack_weights: [B, num_packs] Current total weight of each pack.
        max_iters: Maximum number of refinement iterations.

    Returns:
        pack_indices: Refined item assignments [B, M, G].
    """
    B, num_packs, groups_per_pack = pack_indices.shape
    device = weights.device

    # Mask for diagonal packs (u=v) to prevent self-swapping in the all-pairs matrix
    # [1, M, M, 1, 1]
    eye_mask = torch.eye(num_packs, device=device).view(1, num_packs, num_packs, 1, 1).bool()

    for _ in range(max_iters):
        # Gather current item weights: [B, M, G]
        w_items = torch.gather(weights, 1, pack_indices.view(B, -1)).view(B, num_packs, groups_per_pack)

        # We want to find best swap u <-> v (items i <-> j) to minimize sum of squared pack weights.
        # Change in L2 = 2 * delta * (W_v - W_u + delta)
        # where delta = w_{u,i} - w_{v,j} (weight moved from u to v)

        # Prepare broadcast tensors
        # W_u: [B, M, 1, 1, 1]
        W = pack_weights.view(B, num_packs, 1, 1, 1)

        # w_u: [B, M, 1, G, 1] (Item i in Pack u)
        w_u = w_items.view(B, num_packs, 1, groups_per_pack, 1)

        # w_v: [B, 1, M, 1, G] (Item j in Pack v)
        w_v = w_items.view(B, 1, num_packs, 1, groups_per_pack)

        # Compute delta for all combinations: [B, M, M, G, G]
        # delta[b, u, v, i, j] is weight change for pack v (and -change for pack u)
        delta = w_u - w_v

        # Pack weight diff: W_v - W_u
        # [B, M, M, 1, 1]
        W_diff = W.permute(0, 2, 1, 3, 4) - W

        # Calculate expected L2 change
        change = 2 * delta * (W_diff + delta)

        # Mask out self-swaps (u == v)
        change.masked_fill_(eye_mask, float('inf'))

        # Find best swap per batch
        # Flatten to [B, M*M*G*G]
        change_flat = change.view(B, -1)
        min_val, min_idx = torch.min(change_flat, dim=1)

        # Convergence check: stop if no beneficial swap
        active = min_val < -1e-5
        if not active.any():
            break

        # Decode indices for active batches
        active_idx = torch.nonzero(active).squeeze(-1)
        best_flat = min_idx[active_idx]

        # Constants for decoding
        G = groups_per_pack
        M = num_packs
        G2 = G * G
        MG2 = M * G2

        # Decode: u, v, i, j
        u = best_flat // MG2
        rem = best_flat % MG2
        v = rem // G2
        rem = rem % G2
        i = rem // G
        j = rem % G

        # Apply swaps
        # Indices: [active_idx, u, i] <-> [active_idx, v, j]
        val_u = pack_indices[active_idx, u, i]
        val_v = pack_indices[active_idx, v, j]

        pack_indices[active_idx, u, i] = val_v
        pack_indices[active_idx, v, j] = val_u

        # Update weights
        # delta was calculated as w_u - w_v
        d_val = delta[active_idx, u, v, i, j]

        pack_weights[active_idx, u] -= d_val
        pack_weights[active_idx, v] += d_val

    return pack_indices


def balanced_packing(weight: torch.Tensor,
                     num_packs: int,
                     num_candidates: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pack n weighted objects to m packs using Parallel Randomized Greedy LPT + Refinement.

    Args:
        weight: [L, N] weights
        num_packs: number of target packs
        num_candidates: number of parallel randomized attempts per layer
    """
    num_layers, num_groups = weight.shape
    device = weight.device
    groups_per_pack = num_groups // num_packs

    # Trivial case optimization
    if groups_per_pack == 1:
        pack_index = torch.arange(num_groups, dtype=torch.int64,
                                  device=device).expand(num_layers, -1)
        rank_in_pack = torch.zeros_like(weight, dtype=torch.int64)
        return pack_index, rank_in_pack

    # Expand inputs for parallel candidates
    # [L, K, N] -> flattened to [B, N] where B = L * K
    B = num_layers * num_candidates
    w_expanded = weight.unsqueeze(1).expand(-1, num_candidates, -1).reshape(B, num_groups)

    # Perturb weights to randomize sort order for LPT
    # Add small multiplicative noise: 0.1%
    noise = torch.rand_like(w_expanded) * 0.001
    w_perturbed = w_expanded + w_expanded * noise

    # Sort descending based on perturbed weights
    sorted_indices = torch.argsort(w_perturbed, dim=-1, descending=True)

    # Gather actual weights in the sorted order for accumulation
    row_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, num_groups)
    w_sorted = w_expanded[row_idx, sorted_indices]

    # Initialize packing state
    pack_weights = torch.zeros(B, num_packs, device=device, dtype=weight.dtype)
    pack_counts = torch.zeros(B, num_packs, device=device, dtype=torch.long)
    pack_assignments = torch.zeros(B, num_packs, groups_per_pack, device=device, dtype=torch.long)

    batch_range = torch.arange(B, device=device)
    inf_val = float('inf')

    # Vectorized Greedy LPT Packing
    for i in range(num_groups):
        val = w_sorted[:, i]       # [B]
        itm_idx = sorted_indices[:, i] # [B]

        # Identify full packs
        is_full = pack_counts >= groups_per_pack

        # Mask weights of full packs
        valid_weights = torch.where(is_full, inf_val, pack_weights)

        # Choose pack with min weight
        best_pack = torch.argmin(valid_weights, dim=1) # [B]

        # Get next available slot in the chosen pack
        slots = pack_counts[batch_range, best_pack]

        # Assign
        pack_assignments[batch_range, best_pack, slots] = itm_idx
        pack_weights[batch_range, best_pack] += val
        pack_counts[batch_range, best_pack] += 1

    # Parallel Refinement
    pack_assignments = _refine_packing_batched(w_expanded, pack_assignments, pack_weights)

    # Selection: Pick the candidate with the best balance (min max_load) for each layer
    # Reshape to [L, K, M]
    pw_reshaped = pack_weights.view(num_layers, num_candidates, num_packs)

    # Calculate max load for each candidate
    max_loads = pw_reshaped.max(dim=2).values # [L, K]

    # Argmin to find best candidate index per layer
    best_k = torch.argmin(max_loads, dim=1) # [L]

    # Extract best assignments
    # [L, K, M, G]
    pa_reshaped = pack_assignments.view(num_layers, num_candidates, num_packs, groups_per_pack)
    best_assignments = pa_reshaped[torch.arange(num_layers, device=device), best_k] # [L, M, G]

    # Convert assignments back to (pack_index, rank_in_pack)
    pack_index = torch.empty((num_layers, num_groups), device=device, dtype=torch.int64)
    rank_in_pack = torch.empty((num_layers, num_groups), device=device, dtype=torch.int64)

    # Flatten assignment map to [L, M*G]
    flat_assignments = best_assignments.view(num_layers, -1)

    # Create coordinate grids
    p_ids = torch.arange(num_packs, device=device).unsqueeze(1).expand(-1, groups_per_pack).reshape(-1)
    r_ids = torch.arange(groups_per_pack, device=device).unsqueeze(0).expand(num_packs, -1).reshape(-1)

    p_ids_expanded = p_ids.unsqueeze(0).expand(num_layers, -1)
    r_ids_expanded = r_ids.unsqueeze(0).expand(num_layers, -1)

    # Scatter results
    pack_index.scatter_(1, flat_assignments, p_ids_expanded)
    rank_in_pack.scatter_(1, flat_assignments, r_ids_expanded)

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

    # Greedily assign remaining replicas
    # We need to assign `num_phy - num_log` replicas.
    # In each step, assign to the expert with the highest current load per replica.
    # This loop runs separately for all layers in parallel using tensor ops.
    for _ in range(num_log, num_phy):
        # Score is current load per replica
        scores = weight / logcnt
        # Find expert with max score in each layer
        indices = torch.argmax(scores, dim=-1)
        # Add replica
        # Advanced indexing: row 0 gets indices[0] incremented, etc.
        rows = torch.arange(n, device=device)
        logcnt[rows, indices] += 1

    # Reconstruction of the mapping tables from counts
    phy2log = torch.zeros((n, num_phy), dtype=torch.int64, device=device)
    rank = torch.zeros((n, num_phy), dtype=torch.int64, device=device)

    # Reconstruct mappings row by row
    # While Python loop over layers exists, operations inside are vectorized or efficient enough
    # for typical MoE dimensions (small number of experts/layers).
    for i in range(n):
        counts = logcnt[i]

        # Create logical IDs: repeat each logical ID 'count' times
        # e.g. [0, 0, 1, 2, 2, 2]
        l_ids = torch.repeat_interleave(
            torch.arange(num_log, device=device), counts
        )
        phy2log[i] = l_ids

        # Create ranks: [0, 1, 0, 0, 1, 2]
        # We compute this by iterating logical experts
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
    # Sum weights within each group
    tokens_per_group = weight.unflatten(-1, (num_groups, group_size)).sum(-1)

    # Use improved balanced packing with parallel randomization
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