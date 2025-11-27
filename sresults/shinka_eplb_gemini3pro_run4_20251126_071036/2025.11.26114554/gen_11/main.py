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


def balanced_packing(weight: torch.Tensor,
                     num_packs: int) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pack n weighted objects to m packs, such that each bin contains exactly
    n/m objects and the weights of all packs are as balanced as possible.

    Parameters:
        weight: [X, n], the weight of each item
        num_packs: number of packs

    Returns:
        pack_index: [X, n], the pack index of each item
        rank_in_pack: [X, n], the rank of the item in the pack
    """
    num_layers, num_groups = weight.shape
    assert num_groups % num_packs == 0
    groups_per_pack = num_groups // num_packs

    if groups_per_pack == 1:
        pack_index = torch.arange(weight.size(-1),
                                  dtype=torch.int64,
                                  device=weight.device).expand(weight.shape)
        rank_in_pack = torch.zeros_like(weight, dtype=torch.int64)
        return pack_index, rank_in_pack

    # Optimization: Move weights to CPU to avoid slow element-wise GPU access
    weight_cpu = weight.to("cpu", dtype=torch.float32)

    # Pre-allocate output tensors
    pack_index = torch.empty((num_layers, num_groups),
                             dtype=torch.int64,
                             device="cpu")
    rank_in_pack = torch.empty((num_layers, num_groups),
                               dtype=torch.int64,
                               device="cpu")

    num_restarts = 10  # Number of attempts with randomization

    for i in range(num_layers):
        row_weight = weight_cpu[i]
        # Precompute list for fast access
        row_weight_list = row_weight.tolist()

        best_diff = float('inf')
        best_packs = None

        # Base indices: sorted by weight descending
        base_indices = torch.argsort(row_weight, descending=True)

        for attempt in range(num_restarts):
            # 1. Randomized Greedy Initialization
            if attempt == 0:
                # First attempt: Pure LPT (Longest Processing Time first)
                indices = base_indices.tolist()
            else:
                # Subsequent attempts: Randomized order
                # Add noise to weights only for sorting purposes
                noise = torch.rand(num_groups, device="cpu") * 0.2 + 0.9
                indices = torch.argsort(row_weight * noise,
                                        descending=True).tolist()

            current_packs = [[] for _ in range(num_packs)]
            pack_weights = [0.0] * num_packs

            # Greedy assignment based on (possibly randomized) sort order
            for group_idx in indices:
                w = row_weight_list[group_idx]

                # Find lightest valid pack
                best_p = -1
                min_val = float('inf')

                # Small optimization: if num_packs is small, linear scan is fast enough
                for p in range(num_packs):
                    if len(current_packs[p]) < groups_per_pack:
                        if pack_weights[p] < min_val:
                            min_val = pack_weights[p]
                            best_p = p

                current_packs[best_p].append(group_idx)
                pack_weights[best_p] += w

            # 2. Refinement Phase (Local Search)
            # Try to swap items to improve balance
            for _ in range(20):
                min_p = min(range(num_packs), key=pack_weights.__getitem__)
                max_p = max(range(num_packs), key=pack_weights.__getitem__)

                diff = pack_weights[max_p] - pack_weights[min_p]
                if diff < 1e-6:
                    break

                target_diff = diff / 2.0
                best_swap = None

                # Search for best swap between max and min packs
                # We want (w_u - w_v) ≈ diff/2
                for idx_u, u in enumerate(current_packs[max_p]):
                    w_u = row_weight_list[u]
                    for idx_v, v in enumerate(current_packs[min_p]):
                        w_v = row_weight_list[v]
                        delta = w_u - w_v
                        if 0 < delta < diff:
                            gap = abs(delta - target_diff)
                            if best_swap is None or gap < best_swap[0]:
                                best_swap = (gap, idx_u, idx_v, delta)
                                if gap < 1e-4: break
                    if best_swap and best_swap[0] < 1e-4: break

                if best_swap:
                    _, idx_u, idx_v, delta = best_swap
                    u = current_packs[max_p][idx_u]
                    v = current_packs[min_p][idx_v]

                    current_packs[max_p][idx_u] = v
                    current_packs[min_p][idx_v] = u
                    pack_weights[max_p] -= delta
                    pack_weights[min_p] += delta
                else:
                    break

            # Check if this attempt is better
            current_max = max(pack_weights)
            current_min = min(pack_weights)
            current_diff = current_max - current_min

            if current_diff < best_diff:
                best_diff = current_diff
                best_packs = current_packs
                if best_diff < 1e-6:
                    break  # Perfect balance found

        # 3. Fill result tensors with best found configuration
        for p in range(num_packs):
            for r, g in enumerate(best_packs[p]):
                pack_index[i, g] = p
                rank_in_pack[i, g] = r

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