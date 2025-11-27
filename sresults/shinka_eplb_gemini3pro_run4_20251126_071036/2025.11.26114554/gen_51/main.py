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

    # Use CPU for scalar operations (faster for this logic)
    weight_cpu = weight.cpu()

    # Pre-allocate output on CPU
    pack_index = torch.empty((num_layers, num_groups), dtype=torch.int64)
    rank_in_pack = torch.empty((num_layers, num_groups), dtype=torch.int64)

    for i in range(num_layers):
        layer_weights = weight_cpu[i].tolist()

        # --- 1. Initialization Phase ---
        # Generate multiple candidates and pick the best one.
        # Candidates: 1 Deterministic LPT, 4 Randomized LPT

        candidates = []

        # 1a. Deterministic LPT (Sort descending)
        indices_det = sorted(range(num_groups), key=lambda x: layer_weights[x], reverse=True)
        candidates.append(indices_det)

        # 1b. Randomized LPT
        # Perturb weights slightly to change sort order
        for _ in range(4):
            # Sort by w * random_noise
            indices_rand = sorted(range(num_groups),
                                  key=lambda x: layer_weights[x] * (0.9 + 0.2 * random.random()),
                                  reverse=True)
            candidates.append(indices_rand)

        best_pack_contents = None
        best_pack_weights = None
        best_max_load = float('inf')
        best_ss = float('inf')

        for indices in candidates:
            # Greedy Construction
            current_packs = [[] for _ in range(num_packs)]
            current_weights = [0.0] * num_packs
            current_counts = [0] * num_packs

            for idx in indices:
                w = layer_weights[idx]

                # Find valid pack with min weight
                best_p = -1
                min_w = float('inf')

                for p in range(num_packs):
                    if current_counts[p] < groups_per_pack:
                        if current_weights[p] < min_w:
                            min_w = current_weights[p]
                            best_p = p

                current_packs[best_p].append(idx)
                current_weights[best_p] += w
                current_counts[best_p] += 1

            curr_max = max(current_weights)
            curr_ss = sum(w*w for w in current_weights)

            if curr_max < best_max_load - 1e-6:
                best_max_load = curr_max
                best_ss = curr_ss
                best_pack_contents = current_packs
                best_pack_weights = current_weights
            elif abs(curr_max - best_max_load) < 1e-6:
                if curr_ss < best_ss - 1e-6:
                    best_ss = curr_ss
                    best_pack_contents = current_packs
                    best_pack_weights = current_weights

        # Set current best
        pack_contents = best_pack_contents
        pack_weights = best_pack_weights

        # --- 2. Refinement Phase (LNS) ---
        # Iteratively destroy and repair subsets of packs

        # We can afford more iterations because the sub-problem solver is fast (scalar python)
        MAX_STEPS = 50

        for step in range(MAX_STEPS):
            # Identify Max and Min packs
            # Manual min/max for speed
            max_p = -1
            min_p = -1
            max_w = -1.0
            min_w = float('inf')

            for p, w in enumerate(pack_weights):
                if w > max_w:
                    max_w = w
                    max_p = p
                if w < min_w:
                    min_w = w
                    min_p = p

            if max_p == min_p:
                break

            # Convergence check: if almost balanced, maybe try to optimize variance?
            # If diff is small, we might still want to swap to reduce SS, but if very small, break.
            if max_w - min_w < 1e-6:
                break

            # Selection Strategy: Max + Min + (Random)
            selected_packs = [max_p, min_p]

            # Occasionally add a 3rd random pack to allow cyclic swaps,
            # or if we only have 2 packs, we just stick to 2.
            if num_packs > 2:
                # 50% chance to add a 3rd pack
                if random.random() < 0.5:
                    p3 = random.randint(0, num_packs - 1)
                    while p3 in selected_packs:
                        p3 = random.randint(0, num_packs - 1)
                    selected_packs.append(p3)

            # --- Ruin ---
            items_to_repack = []
            for p in selected_packs:
                items_to_repack.extend(pack_contents[p])

            # Calculate metrics for acceptance
            current_sub_max = max(pack_weights[p] for p in selected_packs)
            current_sub_ss = sum(pack_weights[p]**2 for p in selected_packs)

            # --- Recreate (Sub-problem Solver) ---
            # We run multiple randomized greedy attempts on this small subset

            best_sub_assignment = None
            best_sub_weights = None
            improved_sub = False

            # Optimization: Sort items once for deterministic LPT
            # Store as (weight, index)
            items_data = [(layer_weights[idx], idx) for idx in items_to_repack]
            items_data_sorted = sorted(items_data, key=lambda x: x[0], reverse=True)

            SUB_ATTEMPTS = 15
            for attempt in range(SUB_ATTEMPTS):
                if attempt == 0:
                    # Deterministic LPT
                    current_items = items_data_sorted
                else:
                    # Randomized LPT (perturbed sort)
                    # Inline shuffle-like behavior via perturbed keys is robust
                    current_items = sorted(items_data,
                                           key=lambda x: x[0] * (0.8 + 0.4 * random.random()),
                                           reverse=True)

                # Greedy Best Fit on Subset
                temp_packs = {p: [] for p in selected_packs}
                temp_weights = {p: 0.0 for p in selected_packs}
                temp_counts = {p: 0 for p in selected_packs}

                possible = True
                for w, idx in current_items:
                    # Find best valid bin (lightest)
                    best_local_p = -1
                    min_local_w = float('inf')

                    for p in selected_packs:
                        if temp_counts[p] < groups_per_pack:
                            if temp_weights[p] < min_local_w:
                                min_local_w = temp_weights[p]
                                best_local_p = p

                    if best_local_p == -1:
                        # Should not happen if total items = total capacity
                        possible = False
                        break

                    temp_packs[best_local_p].append(idx)
                    temp_weights[best_local_p] += w
                    temp_counts[best_local_p] += 1

                if possible:
                    # Check metrics
                    new_max = max(temp_weights.values())
                    new_ss = sum(v*v for v in temp_weights.values())

                    # Acceptance Criterion
                    if new_max < current_sub_max - 1e-6:
                        current_sub_max = new_max
                        current_sub_ss = new_ss
                        best_sub_assignment = temp_packs
                        best_sub_weights = temp_weights
                        improved_sub = True
                    elif abs(new_max - current_sub_max) < 1e-6:
                        if new_ss < current_sub_ss - 1e-6:
                            current_sub_ss = new_ss
                            best_sub_assignment = temp_packs
                            best_sub_weights = temp_weights
                            improved_sub = True

            # Apply improvement if found
            if improved_sub:
                for p in selected_packs:
                    pack_contents[p] = best_sub_assignment[p]
                    pack_weights[p] = best_sub_weights[p]

        # --- Final Write ---
        for p in range(num_packs):
            for rank, g_idx in enumerate(pack_contents[p]):
                pack_index[i, g_idx] = p
                rank_in_pack[i, g_idx] = rank

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