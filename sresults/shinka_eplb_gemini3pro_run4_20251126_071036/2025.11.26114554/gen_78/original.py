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
import random


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

    # Optimization: Move weights to CPU and use Python lists for the sequential greedy/swap logic.
    # This avoids high kernel launch overheads for sequential scalar operations on GPU.
    weight_cpu = weight.to("cpu", dtype=torch.float32)
    indices = weight_cpu.sort(-1, descending=True).indices.tolist()
    weight_list = weight_cpu.tolist()

    pack_index = torch.empty(weight.shape, dtype=torch.int64, device="cpu")
    rank_in_pack = torch.empty(weight.shape, dtype=torch.int64, device="cpu")

    for i in range(num_layers):
        row_weight = weight_list[i]
        # current_packs stores tuples of (expert_index, weight)
        current_packs = [[] for _ in range(num_packs)]
        pack_weights = [0.0] * num_packs

        # 1. Greedy packing (LPT: Longest Processing Time first)
        # Assign heaviest items first to the least loaded pack that has space.
        for group in indices[i]:
            w = row_weight[group]
            best_pack = -1
            min_val = float('inf')

            # Find the least loaded pack with capacity
            for p in range(num_packs):
                if len(current_packs[p]) < groups_per_pack:
                    if pack_weights[p] < min_val:
                        min_val = pack_weights[p]
                        best_pack = p

            current_packs[best_pack].append((group, w))
            pack_weights[best_pack] += w

        # 2. Refinement: Pairwise swapping
        # Try to swap items between the heaviest and lightest packs to balance load.
        for _ in range(20):
            sorted_packs = sorted(range(num_packs),
                                  key=pack_weights.__getitem__,
                                  reverse=True)
            found_improvement = False

            for i1 in range(num_packs):
                p1 = sorted_packs[i1]
                # Compare against lighter packs
                for i2 in range(num_packs - 1, i1, -1):
                    p2 = sorted_packs[i2]

                    diff = pack_weights[p1] - pack_weights[p2]
                    if diff < 1e-6:
                        break # Since packs are sorted, subsequent diffs will be smaller

                    target = diff / 2.0
                    best_swap = None
                    best_gap = diff

                    # Find best swap: u from p1, v from p2 such that w_u - w_v approx target
                    for idx_u, (u, w_u) in enumerate(current_packs[p1]):
                        for idx_v, (v, w_v) in enumerate(current_packs[p2]):
                            delta = w_u - w_v
                            # We need w_u > w_v to reduce diff
                            if 0 < delta < diff:
                                gap = abs(delta - target)
                                if gap < best_gap:
                                    best_gap = gap
                                    # Store swap details: indices and the actual values to swap
                                    best_swap = (idx_u, idx_v, delta, v, w_v, u, w_u)
                                    if gap < 1e-5: break
                        if best_swap and best_gap < 1e-5: break

                    if best_swap:
                        idx_u, idx_v, delta, val_v, weight_v, val_u, weight_u = best_swap

                        # Apply swap
                        current_packs[p1][idx_u] = (val_v, weight_v)
                        current_packs[p2][idx_v] = (val_u, weight_u)
                        pack_weights[p1] -= delta
                        pack_weights[p2] += delta
                        found_improvement = True
                        break # Break inner loop to re-sort packs

                if found_improvement: break

            if not found_improvement:
                break

        # 3. Fill result tensors
        for p in range(num_packs):
            for r, (g, _) in enumerate(current_packs[p]):
                pack_index[i, g] = p
                rank_in_pack[i, g] = r

    return pack_index.to(weight.device), rank_in_pack.to(weight.device)


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

    # Optimization: Track current scores (load/count) to avoid full matrix recalculation
    current_scores = weight.clone()

    for i in range(num_log, num_phy):
        # Pick the expert with the max current load per replica
        redundant_indices = current_scores.max(dim=-1).indices

        phy2log[:, i] = redundant_indices
        rank[:, i] = logcnt[arangen, redundant_indices]

        # Increment count
        logcnt[arangen, redundant_indices] += 1

        # Update scores only for the experts that got a new replica
        # score = weight / count
        # We use advanced indexing to update only the modified positions
        updated_weights = weight[arangen, redundant_indices]
        updated_counts = logcnt[arangen, redundant_indices].float()
        current_scores[arangen, redundant_indices] = updated_weights / updated_counts

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