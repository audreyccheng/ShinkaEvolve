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
                    max_iters: int = 50) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Refines packing by swapping items between any two packs to minimize the L2 norm
    (variance) of the pack loads. Vectorized over the batch dimension (num_layers).
    """
    num_layers = weights.shape[0]
    batch_idx = torch.arange(num_layers, device=weights.device)

    # Pre-compute mask for diagonal (self-swaps)
    # [1, M, M, 1, 1]
    diag_mask = torch.eye(num_packs, device=weights.device).view(
        1, num_packs, num_packs, 1, 1).bool()

    for _ in range(max_iters):
        # 1. Gather item weights currently in packs
        # w_items: [L, M, G]
        w_items = weights.gather(
            1, pack_indices.view(num_layers, -1)).view(num_layers, num_packs,
                                                       groups_per_pack)

        # 2. Prepare tensors for broadcasting
        # W: [L, M, 1, 1, 1]
        W = pack_weights.view(num_layers, num_packs, 1, 1, 1)
        # w_u: [L, M, 1, G, 1] (Item i in Pack u)
        w_u = w_items.view(num_layers, num_packs, 1, groups_per_pack, 1)
        # w_v: [L, 1, M, 1, G] (Item j in Pack v)
        w_v = w_items.view(num_layers, 1, num_packs, 1, groups_per_pack)

        # 3. Compute Delta and L2 Change
        # delta = w_u - w_v. Weight moved from u to v.
        # [L, M, M, G, G]
        delta = w_u - w_v

        # W_v - W_u. [L, M, M, 1, 1]
        W_diff = W.permute(0, 2, 1, 3, 4) - W

        # Change in L2 = 2 * delta * (W_v - W_u + delta)
        change = 2 * delta * (W_diff + delta)

        # Mask invalid swaps (self-swaps)
        change.masked_fill_(diag_mask, float('inf'))

        # 4. Find best swap per layer
        flat_change = change.view(num_layers, -1)
        min_change, flat_idx = torch.min(flat_change, dim=1)

        # 5. Check convergence
        active_mask = min_change < -1e-5
        if not active_mask.any():
            break

        active_indices = batch_idx[active_mask]

        # 6. Execute Swaps
        best_idx = flat_idx[active_mask]

        G = groups_per_pack
        G2 = G * G
        MG2 = num_packs * G2

        # idx = u * (M*G*G) + v * (G*G) + i * G + j
        u = best_idx // MG2
        rem = best_idx % MG2
        v = rem // G2
        rem = rem % G2
        i = rem // G
        j = rem % G

        # Swap indices
        val_u = pack_indices[active_indices, u, i]
        val_v = pack_indices[active_indices, v, j]

        pack_indices[active_indices, u, i] = val_v
        pack_indices[active_indices, v, j] = val_u

        # Update Weights
        d_val = delta[active_mask, u, v, i, j]
        pack_weights[active_indices, u] -= d_val
        pack_weights[active_indices, v] += d_val

    return pack_indices, pack_weights


def balanced_packing(weight: torch.Tensor,
                     num_packs: int,
                     num_attempts: int = 10) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pack n weighted objects to m packs using Vectorized Iterated Greedy.
    Combines Greedy LPT with iterative re-weighting and all-pairs L2 refinement.
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

    best_assignment = torch.zeros((num_layers, num_packs, groups_per_pack), dtype=torch.int64, device=device)
    best_max_loads = torch.full((num_layers,), float('inf'), device=device)
    virtual_weight = weight.clone().float()
    batch_idx = torch.arange(num_layers, device=device)
    inf_tensor = torch.tensor(float('inf'), device=device)

    # Pre-allocate tensors for speed
    curr_pack_assignment = torch.zeros((num_layers, num_packs, groups_per_pack), dtype=torch.int64, device=device)
    curr_pack_weights = torch.zeros((num_layers, num_packs), dtype=weight.dtype, device=device)
    curr_pack_counts = torch.zeros((num_layers, num_packs), dtype=torch.int64, device=device)

    for attempt in range(num_attempts):
        # 1. Greedy Initialization
        curr_pack_weights.zero_()
        curr_pack_counts.zero_()

        sorted_indices = torch.argsort(virtual_weight, dim=1, descending=True)

        for j in range(num_groups):
            item_idx = sorted_indices[:, j] # [L]
            w_real = weight[batch_idx, item_idx] # [L]

            is_full = (curr_pack_counts >= groups_per_pack)
            masked_weights = torch.where(is_full, inf_tensor, curr_pack_weights)
            best_pack = torch.argmin(masked_weights, dim=1) # [L]

            slot = curr_pack_counts[batch_idx, best_pack]
            curr_pack_assignment[batch_idx, best_pack, slot] = item_idx
            curr_pack_weights[batch_idx, best_pack] += w_real
            curr_pack_counts[batch_idx, best_pack] += 1

        # 2. Refinement
        curr_pack_assignment, curr_pack_weights = _refine_packing(
            weight, curr_pack_assignment, curr_pack_weights, num_packs, groups_per_pack
        )

        # 3. Update Best
        curr_max_load, curr_max_pid = torch.max(curr_pack_weights, dim=1)
        improved = curr_max_load < best_max_loads

        if improved.any():
            mask_exp = improved.view(num_layers, 1, 1)
            best_assignment = torch.where(mask_exp, curr_pack_assignment, best_assignment)
            best_max_loads = torch.where(improved, curr_max_load, best_max_loads)

        # 4. Re-weighting
        if attempt < num_attempts - 1:
            max_p_items = curr_pack_assignment[batch_idx, curr_max_pid] # [L, G]
            mults = torch.ones_like(virtual_weight)
            src = torch.full_like(max_p_items, 1.05, dtype=virtual_weight.dtype)
            mults.scatter_(1, max_p_items, src)
            virtual_weight = virtual_weight * mults

    # Construct final outputs
    pack_index = torch.zeros_like(weight, dtype=torch.int64)
    rank_in_pack = torch.zeros_like(weight, dtype=torch.int64)

    flat_assignment = best_assignment.view(num_layers, -1)

    p_ids = torch.arange(num_packs, device=device).unsqueeze(1).expand(-1, groups_per_pack).reshape(1, -1).expand(num_layers, -1)
    r_ids = torch.arange(groups_per_pack, device=device).unsqueeze(0).expand(num_packs, -1).reshape(1, -1).expand(num_layers, -1)

    pack_index.scatter_(1, flat_assignment, p_ids)
    rank_in_pack.scatter_(1, flat_assignment, r_ids)

    return pack_index, rank_in_pack


def replicate_experts(
        weight: torch.Tensor,
        num_phy: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Replicate `num_log` experts to `num_phy` replicas, such that the maximum
    load of all replicas is minimized.

    Parameters:
        weight: [X, num_log]
        num_phy: total number of experts after replication

    Returns:
        phy2log: [X, num_phy], logical expert id of each physical expert
        rank: [X, num_phy], the replica rank
        logcnt: [X, num_log], number of replicas for each logical expert
    """
    n, num_log = weight.shape
    num_redundant = num_phy - num_log
    assert num_redundant >= 0
    device = weight.device
    phy2log = torch.arange(num_phy, dtype=torch.int64,
                           device=device).repeat(n, 1)
    rank = torch.zeros(n, num_phy, dtype=torch.int64, device=device)
    logcnt = torch.ones(n, num_log, dtype=torch.int64, device=device)
    arangen = torch.arange(n, dtype=torch.int64, device=device)
    for i in range(num_log, num_phy):
        redundant_indices = (weight / logcnt).max(dim=-1).indices
        phy2log[:, i] = redundant_indices
        rank[:, i] = logcnt[arangen, redundant_indices]
        logcnt[arangen, redundant_indices] += 1
    return phy2log, rank, logcnt


def rebalance_experts_hierarchical(
    weight: torch.Tensor,
    num_physical_experts: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
):
    """
    Parameters:
        weight: [num_moe_layers, num_logical_experts]
        num_physical_experts: number of physical experts after replication
        num_groups: number of expert groups
        num_nodes: number of server nodes, where the intra-node network
        (e.g, NVLink) is faster
        num_gpus: number of GPUs, must be a multiple of `num_nodes`

    Returns:
        physical_to_logical_map: [num_moe_layers, num_physical_experts]
        logical_to_physical_map: [num_moe_layers, num_logical_experts, X]
        logical_count: [num_moe_layers, num_logical_experts]
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
    # [num_layers * num_nodes, num_logical_experts // num_nodes]
    tokens_per_mlog = weight.gather(-1, mlog2log).view(
        -1, num_logical_experts // num_nodes)
    phy2mlog, phyrank, mlogcnt = replicate_experts(
        tokens_per_mlog, num_physical_experts // num_nodes)

    # Step 3: pack physical_experts to GPUs
    # [num_layers * num_nodes, num_physical_experts // num_nodes]
    tokens_per_phy = (tokens_per_mlog / mlogcnt).gather(-1, phy2mlog)
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

    Parameters:
        weight: [layers, num_logical_experts], the load statistics for all
            logical experts
        num_replicas: number of physical experts, must be a multiple of
            `num_gpus`
        num_groups: number of expert groups
        num_nodes: number of server nodes, where the intra-node network
            (e.g, NVLink) is faster
        num_gpus: number of GPUs, must be a multiple of `num_nodes`

    Returns:
        physical_to_logical_map: [layers, num_replicas], the expert index of
            each replica
        logical_to_physical_map: [layers, num_logical_experts, X], the replica
            indices for each expert
        expert_count: [layers, num_logical_experts], number of physical
            replicas for each logical expert
    """
    num_layers, num_logical_experts = weight.shape
    weight = weight.float().cpu()
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