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
import heapq
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

    # Optimization: handle trivial case efficiently
    if groups_per_pack == 1:
        pack_index = torch.arange(num_groups,
                                  dtype=torch.int64,
                                  device=weight.device).expand(num_layers, -1)
        rank_in_pack = torch.zeros_like(pack_index)
        return pack_index, rank_in_pack

    # Move computation to CPU to avoid GPU synchronization overhead for sequential logic
    weight_cpu = weight.to("cpu", dtype=torch.float32)

    # Pre-allocate result tensors on CPU
    pack_index = torch.empty((num_layers, num_groups), dtype=torch.int64, device="cpu")
    rank_in_pack = torch.empty((num_layers, num_groups), dtype=torch.int64, device="cpu")

    # Number of random restarts to escape local optima
    num_restarts = 4

    for i in range(num_layers):
        row_w = weight_cpu[i].tolist()

        # Original items (index, weight)
        original_items = sorted(enumerate(row_w), key=lambda x: x[1], reverse=True)

        best_diff = float('inf')
        best_packs = None

        for attempt in range(num_restarts):
            # 1. Initialization Strategy
            current_packs = [[] for _ in range(num_packs)]
            pack_weights = [0.0] * num_packs

            if attempt == 0:
                # Deterministic LPT
                items = original_items
            else:
                # Randomized LPT
                noise = torch.rand(num_groups).tolist()
                noisy_items = [(idx, w, w * (0.9 + 0.2 * noise[k]))
                               for k, (idx, w) in enumerate(original_items)]
                noisy_items.sort(key=lambda x: x[2], reverse=True)
                items = [(idx, w) for idx, w, _ in noisy_items]

            # Greedy Phase
            for idx, w in items:
                # Assign to the lightest pack that has space
                best_p = -1
                min_w = float('inf')
                for p in range(num_packs):
                    if len(current_packs[p]) < groups_per_pack:
                        if pack_weights[p] < min_w:
                            min_w = pack_weights[p]
                            best_p = p
                current_packs[best_p].append(idx)
                pack_weights[best_p] += w

            # 2. Refinement Phase: Pool-and-Split
            # Instead of single swaps, pick two packs (Max and Min), pool items, and re-distribute.
            for _ in range(30):
                # Identify Max and Min packs
                min_p = 0
                max_p = 0
                min_val = pack_weights[0]
                max_val = pack_weights[0]

                for p in range(1, num_packs):
                    val = pack_weights[p]
                    if val < min_val:
                        min_val = val
                        min_p = p
                    elif val > max_val:
                        max_val = val
                        max_p = p

                if max_val - min_val < 1e-6:
                    break

                # Try to improve by pairing Max Pack with Min Pack
                # Only trying Max-Min pair is usually sufficient and fast.
                p1, p2 = max_p, min_p

                # Pool items
                pooled_items = []
                for idx in current_packs[p1]:
                    pooled_items.append((idx, row_w[idx]))
                for idx in current_packs[p2]:
                    pooled_items.append((idx, row_w[idx]))

                # Sort Descending
                pooled_items.sort(key=lambda x: x[1], reverse=True)

                # Re-distribute using constrained greedy into two temp packs
                t_packs = [[], []]
                t_weights = [0.0, 0.0]

                possible = True
                for idx, w in pooled_items:
                    # Put in valid bin with min weight
                    b_best = -1
                    b_min = float('inf')
                    for b in range(2):
                        if len(t_packs[b]) < groups_per_pack:
                            if t_weights[b] < b_min:
                                b_min = t_weights[b]
                                b_best = b
                    if b_best != -1:
                        t_packs[b_best].append(idx)
                        t_weights[b_best] += w
                    else:
                        possible = False
                        break

                if not possible: break

                # Check criteria
                old_local_max = max(pack_weights[p1], pack_weights[p2])
                new_local_max = max(t_weights[0], t_weights[1])
                old_spread = abs(pack_weights[p1] - pack_weights[p2])
                new_spread = abs(t_weights[0] - t_weights[1])

                # Accept if max load reduced, or if max load unchanged but spread reduced
                if new_local_max < old_local_max - 1e-6 or \
                   (abs(new_local_max - old_local_max) < 1e-6 and new_spread < old_spread - 1e-6):
                    current_packs[p1] = t_packs[0]
                    pack_weights[p1] = t_weights[0]
                    current_packs[p2] = t_packs[1]
                    pack_weights[p2] = t_weights[1]
                else:
                    # If Pool-and-Split on Max/Min didn't help, we are likely in a local optimum
                    # reachable by this move.
                    break

            # Check global result
            curr_diff = max(pack_weights) - min(pack_weights)
            if curr_diff < best_diff:
                best_diff = curr_diff
                best_packs = [list(p) for p in current_packs]
                if best_diff < 1e-6: break

        # Fill result tensors
        for p in range(num_packs):
            for r, idx in enumerate(best_packs[p]):
                pack_index[i, idx] = p
                rank_in_pack[i, idx] = r

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

    if num_redundant == 0:
        return phy2log, rank, logcnt

    # Move to CPU for efficient priority queue operations
    weight_cpu = weight.to("cpu", dtype=torch.float32)
    phy2log_cpu = phy2log.to("cpu")
    rank_cpu = rank.to("cpu")
    logcnt_cpu = logcnt.to("cpu")

    for layer_idx in range(n):
        w_row = weight_cpu[layer_idx].tolist()

        # Priority queue stores (-score, expert_idx, current_count)
        # Score = weight / count
        # Python heapq is a min-heap, so we store negative score to simulate max-heap
        pq = []
        for i in range(num_log):
            heapq.heappush(pq, (-w_row[i], i, 1))

        # Assign replicas one by one
        for r_idx in range(num_log, num_phy):
            score_neg, expert_idx, count = heapq.heappop(pq)

            # Assign the new replica
            phy2log_cpu[layer_idx, r_idx] = expert_idx
            rank_cpu[layer_idx, r_idx] = count

            # Update count and push back with new score
            new_count = count + 1
            logcnt_cpu[layer_idx, expert_idx] = new_count
            new_score_neg = -w_row[expert_idx] / new_count
            heapq.heappush(pq, (new_score_neg, expert_idx, new_count))

    return phy2log_cpu.to(device), rank_cpu.to(device), logcnt_cpu.to(device)


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