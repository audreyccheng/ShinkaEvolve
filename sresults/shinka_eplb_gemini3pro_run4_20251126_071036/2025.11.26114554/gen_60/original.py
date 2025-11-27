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
    device = weight.device

    if groups_per_pack == 1:
        pack_index = torch.arange(weight.size(-1),
                                  dtype=torch.int64,
                                  device=device).expand(weight.shape)
        rank_in_pack = torch.zeros_like(weight, dtype=torch.int64)
        return pack_index, rank_in_pack

    # Operations on CPU are generally faster for this sequential/iterative logic
    weight_cpu = weight.cpu()

    # Pre-allocate output tensors
    pack_index = torch.empty(weight.shape, dtype=torch.int64, device=device)
    rank_in_pack = torch.empty(weight.shape, dtype=torch.int64, device=device)

    # Number of restarts for randomized greedy
    NUM_RESTARTS = 20

    for i in range(num_layers):
        layer_weights = weight_cpu[i]

        # Base sort indices (LPT)
        base_indices = layer_weights.argsort(descending=True).tolist()

        best_assignment = None
        best_max_load = float('inf')

        for attempt in range(NUM_RESTARTS):
            # 1. Randomized Candidate Generation
            if attempt == 0:
                indices = base_indices
            else:
                # Add noise to weights to perturb sort order
                # 5% noise is usually sufficient to swap similar items
                noise = torch.rand(num_groups) * 0.05
                indices = (layer_weights * (1 + noise)).argsort(descending=True).tolist()

            # 2. Greedy Construction (LPT / Best Fit)
            # pack_contents: list of lists storing item indices
            pack_contents = [[] for _ in range(num_packs)]
            pack_weights = [0.0] * num_packs
            pack_sizes = [0] * num_packs

            for idx in indices:
                w = layer_weights[idx].item()

                # Find lightest valid pack
                best_p = -1
                min_w = float('inf')

                # Simple linear scan is fast for small num_packs
                for p in range(num_packs):
                    if pack_sizes[p] < groups_per_pack:
                        if pack_weights[p] < min_w:
                            min_w = pack_weights[p]
                            best_p = p

                pack_contents[best_p].append(idx)
                pack_weights[best_p] += w
                pack_sizes[best_p] += 1

            # 3. Refinement: Large Neighborhood Search (LNS)
            # Iteratively improve by repacking subsets of bins

            MAX_LNS_STEPS = 50
            consecutive_no_improve = 0

            for step in range(MAX_LNS_STEPS):
                # Sort packs by weight
                sorted_packs = sorted(range(num_packs), key=lambda x: pack_weights[x], reverse=True)
                max_p = sorted_packs[0]
                min_p = sorted_packs[-1]

                # If balanced enough, stop
                if pack_weights[max_p] - pack_weights[min_p] < 1e-6:
                    break

                # Select candidates for LNS: Heaviest, Lightest, and potentially a random one
                candidates = [max_p, min_p]
                if num_packs > 2:
                    # Pick a random pack that is not max or min to facilitate 3-way/4-way cyclic shifts
                    # Use a simple pseudo-random approach based on step
                    candidates.append(sorted_packs[(step + 1) % (num_packs - 1) + 1])

                # Collect items from candidates
                items_to_repack = []
                for p in candidates:
                    items_to_repack.extend(pack_contents[p])

                # Sort descending (LPT)
                items_to_repack.sort(key=lambda x: layer_weights[x].item(), reverse=True)

                # Repack into these bins using Bounded Best Fit
                new_loads = {p: 0.0 for p in candidates}
                new_contents = {p: [] for p in candidates}
                possible = True

                for item_idx in items_to_repack:
                    w = layer_weights[item_idx].item()
                    # Assign to candidate bin with min current load that has space
                    best_local_p = -1
                    min_local_w = float('inf')
                    for p in candidates:
                        if len(new_contents[p]) < groups_per_pack:
                            if new_loads[p] < min_local_w:
                                min_local_w = new_loads[p]
                                best_local_p = p

                    if best_local_p == -1:
                        possible = False
                        break

                    new_contents[best_local_p].append(item_idx)
                    new_loads[best_local_p] += w

                if possible:
                    # Check improvement criteria
                    old_local_max = max(pack_weights[p] for p in candidates)
                    new_local_max = max(new_loads.values())

                    # Accept if max load decreases OR max load is same but variance decreases (better balance)
                    # Sum of squares is a good proxy for variance
                    old_ss = sum(pack_weights[p]**2 for p in candidates)
                    new_ss = sum(val**2 for val in new_loads.values())

                    if new_local_max < old_local_max - 1e-6 or (abs(new_local_max - old_local_max) < 1e-6 and new_ss < old_ss - 1e-6):
                        # Apply Update
                        for p in candidates:
                            pack_contents[p] = new_contents[p]
                            pack_weights[p] = new_loads[p]
                            pack_sizes[p] = groups_per_pack # Should be full
                        consecutive_no_improve = 0
                    else:
                        consecutive_no_improve += 1
                else:
                    consecutive_no_improve += 1

                if consecutive_no_improve >= 5:
                    break

            # 4. Check global best
            current_max = max(pack_weights)
            if current_max < best_max_load - 1e-6:
                best_max_load = current_max
                best_assignment = [list(p) for p in pack_contents]

        # 5. Write best assignment to output tensors
        # Flatten for scatter
        flat_indices = []
        flat_packs = []
        flat_ranks = []

        for p, items in enumerate(best_assignment):
            for r, idx in enumerate(items):
                flat_indices.append(idx)
                flat_packs.append(p)
                flat_ranks.append(r)

        idx_tensor = torch.tensor(flat_indices, dtype=torch.int64, device=device)
        pack_tensor = torch.tensor(flat_packs, dtype=torch.int64, device=device)
        rank_tensor = torch.tensor(flat_ranks, dtype=torch.int64, device=device)

        pack_index[i].scatter_(0, idx_tensor, pack_tensor)
        rank_in_pack[i].scatter_(0, idx_tensor, rank_tensor)

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