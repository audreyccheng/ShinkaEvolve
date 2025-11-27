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

    Uses Iterated Local Search (ILS) with a greedy LPT initialization.
    """
    num_layers, num_groups = weight.shape
    assert num_groups % num_packs == 0
    groups_per_pack = num_groups // num_packs

    # Handle trivial case
    if groups_per_pack == 1:
        pack_index = torch.arange(num_groups, dtype=torch.int64, device=weight.device).expand(num_layers, -1)
        rank_in_pack = torch.zeros_like(pack_index)
        return pack_index, rank_in_pack

    # CPU processing for complex logic
    weight_cpu = weight.to("cpu", dtype=torch.float32)
    pack_index = torch.empty((num_layers, num_groups), dtype=torch.int64, device="cpu")
    rank_in_pack = torch.empty((num_layers, num_groups), dtype=torch.int64, device="cpu")

    import random

    for i in range(num_layers):
        row_w = weight_cpu[i]
        items = sorted([(idx, row_w[idx].item()) for idx in range(num_groups)],
                       key=lambda x: x[1], reverse=True)

        # --- 1. Greedy Initialization (LPT) ---
        # Assign to the lightest pack that has space
        current_packs = [[] for _ in range(num_packs)]
        current_loads = [0.0] * num_packs

        for idx, w in items:
            best_p = -1
            min_load = float('inf')
            for p in range(num_packs):
                if len(current_packs[p]) < groups_per_pack:
                    if current_loads[p] < min_load:
                        min_load = current_loads[p]
                        best_p = p
            current_packs[best_p].append((idx, w))
            current_loads[best_p] += w

        # Track global best
        best_packs = [list(p) for p in current_packs]
        best_loads = list(current_loads)
        best_diff = max(best_loads) - min(best_loads)

        if best_diff < 1e-6:
            # Already optimal
            pass
        else:
            # --- 2. Iterated Local Search ---
            # Try to improve packing via local search and restarts
            num_restarts = 4
            iter_per_restart = 50

            for attempt in range(num_restarts):
                # Perturbation (except first run, which uses greedy)
                if attempt > 0:
                    # Restore best known as base for perturbation
                    current_packs = [list(p) for p in best_packs]
                    current_loads = list(best_loads)

                    # Randomly swap items between k random pairs of packs
                    k_swaps = 2
                    for _ in range(k_swaps):
                        p1 = random.randint(0, num_packs - 1)
                        p2 = random.randint(0, num_packs - 1)
                        if p1 == p2: continue
                        if not current_packs[p1] or not current_packs[p2]: continue

                        i1 = random.randint(0, len(current_packs[p1]) - 1)
                        i2 = random.randint(0, len(current_packs[p2]) - 1)

                        u, w_u = current_packs[p1][i1]
                        v, w_v = current_packs[p2][i2]

                        current_packs[p1][i1] = (v, w_v)
                        current_packs[p2][i2] = (u, w_u)
                        current_loads[p1] = current_loads[p1] - w_u + w_v
                        current_loads[p2] = current_loads[p2] - w_v + w_u

                # Local Search (Descent)
                for _ in range(iter_per_restart):
                    # Sort packs by load
                    sorted_packs = sorted(range(num_packs), key=lambda k: current_loads[k])
                    min_p = sorted_packs[0]
                    max_p = sorted_packs[-1]

                    diff = current_loads[max_p] - current_loads[min_p]
                    if diff < 1e-6: break

                    target = diff / 2.0
                    best_move = None
                    best_gap = diff

                    # Try to swap between max_p and min_p
                    p1 = max_p
                    p2 = min_p
                    found_swap = False

                    for i1, (u, w_u) in enumerate(current_packs[p1]):
                        for i2, (v, w_v) in enumerate(current_packs[p2]):
                            delta = w_u - w_v
                            if 0 < delta < diff:
                                gap = abs(delta - target)
                                if gap < best_gap:
                                    best_gap = gap
                                    best_move = (p1, p2, i1, i2, delta, u, w_u, v, w_v)
                                    if gap < 1e-6:
                                        found_swap = True
                                        break
                        if found_swap: break

                    # If no good swap between extremes, try swapping max_p with random others
                    if not best_move:
                        for _ in range(2):
                            rand_p = random.choice(sorted_packs[1:-1]) if len(sorted_packs) > 2 else min_p
                            if rand_p == max_p: continue

                            limit = current_loads[max_p] - current_loads[rand_p]
                            if limit < 1e-6: continue

                            for i1, (u, w_u) in enumerate(current_packs[max_p]):
                                for i2, (v, w_v) in enumerate(current_packs[rand_p]):
                                    delta = w_u - w_v
                                    if 0 < delta < limit:
                                        best_move = (max_p, rand_p, i1, i2, delta, u, w_u, v, w_v)
                                        found_swap = True
                                        break
                                if found_swap: break
                            if found_swap: break

                    if best_move:
                        p_from, p_to, i_from, i_to, delta, u_idx, u_w, v_idx, v_w = best_move
                        current_packs[p_from][i_from] = (v_idx, v_w)
                        current_packs[p_to][i_to] = (u_idx, u_w)
                        current_loads[p_from] -= delta
                        current_loads[p_to] += delta
                    else:
                        break # Local optimum reached

                # Check against global best
                curr_diff = max(current_loads) - min(current_loads)
                if curr_diff < best_diff:
                    best_diff = curr_diff
                    best_packs = [list(p) for p in current_packs]
                    best_loads = list(current_loads)
                    if best_diff < 1e-6: break

        # Fill output tensors
        for p in range(num_packs):
            for r, (idx, _) in enumerate(best_packs[p]):
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
    arangen = torch.arange(n, dtype=torch.int64, device=device)

    # Optimization: Maintain current scores to avoid repetitive division
    # Initial score is weight / 1
    current_scores = weight.clone().float()

    for i in range(num_log, num_phy):
        # Find expert with highest load per replica
        redundant_indices = current_scores.max(dim=-1).indices

        phy2log[:, i] = redundant_indices
        rank[:, i] = logcnt[arangen, redundant_indices]

        # Increment counts
        logcnt[arangen, redundant_indices] += 1

        # Update scores only for changed experts
        # score = weight / count
        selected_weights = weight[arangen, redundant_indices].float()
        selected_counts = logcnt[arangen, redundant_indices].float()
        current_scores[arangen, redundant_indices] = selected_weights / selected_counts

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