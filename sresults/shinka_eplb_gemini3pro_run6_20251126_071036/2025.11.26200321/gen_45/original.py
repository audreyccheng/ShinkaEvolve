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

    Uses ZigZag initialization followed by a vectorized Max-Any Swap local search.

    Parameters:
        weight: [layers, n], the weight of each item
        num_packs: number of packs

    Returns:
        pack_index: [layers, n], the pack index of each item
        rank_in_pack: [layers, n], the rank of the item in the pack
    """
    num_layers, num_groups = weight.shape
    device = weight.device
    assert num_groups % num_packs == 0
    groups_per_pack = num_groups // num_packs

    # Trivial case
    if groups_per_pack == 1:
        pack_index = torch.arange(num_packs, dtype=torch.int64, device=device).expand(num_layers, -1)
        rank_in_pack = torch.zeros_like(pack_index)
        return pack_index, rank_in_pack

    # 1. Sort weights descending: [L, N]
    sorted_weight, sorted_indices = weight.sort(dim=-1, descending=True)

    # 2. ZigZag Initialization
    pattern = torch.cat([
        torch.arange(num_packs, device=device),
        torch.arange(num_packs - 1, -1, -1, device=device)
    ])
    num_patterns = (num_groups + len(pattern) - 1) // len(pattern)
    assignments = pattern.repeat(num_patterns)[:num_groups] # [N]

    _, perm = assignments.sort(stable=True)
    pack_contents_indices = perm.view(num_packs, groups_per_pack)
    pack_contents = pack_contents_indices.unsqueeze(0).expand(num_layers, -1, -1).clone()

    # 3. Vectorized Local Search
    # Iterate to swap items between the heaviest (Max) and ANY other pack to reduce max load.

    num_iters = 20
    layer_arange = torch.arange(num_layers, device=device)

    for _ in range(num_iters):
        # Gather weights: [L, M, K]
        flat_contents = pack_contents.view(num_layers, -1)
        current_weights = torch.gather(sorted_weight, 1, flat_contents).view(num_layers, num_packs, groups_per_pack)

        # Compute sums: [L, M]
        pack_sums = current_weights.sum(dim=2)

        # Find Max pack: [L]
        val_max, idx_max_pack = pack_sums.max(dim=1)

        # Get weights in Max pack: [L, K]
        w_max = current_weights.gather(1, idx_max_pack.view(num_layers, 1, 1).expand(-1, -1, groups_per_pack)).squeeze(1)

        # Compute diffs: [L, M, K_max, K_other]
        # w_max: [L, 1, K, 1]
        # current_weights: [L, M, 1, K]
        w_max_expanded = w_max.view(num_layers, 1, groups_per_pack, 1)
        current_weights_expanded = current_weights.view(num_layers, num_packs, 1, groups_per_pack)

        diffs = w_max_expanded - current_weights_expanded # [L, M, K, K]

        # Target improvement logic
        # New Max Pack Weight = val_max - diff
        # New Other Pack Weight = pack_sums[other] + diff
        # New pair max = max(val_max - diff, pack_sums[other] + diff)
        # Improvement = val_max - New pair max

        val_max_expanded = val_max.view(num_layers, 1, 1, 1)
        pack_sums_expanded = pack_sums.view(num_layers, num_packs, 1, 1)

        new_pair_max = torch.max(val_max_expanded - diffs, pack_sums_expanded + diffs)
        improvement = val_max_expanded - new_pair_max

        # Mask out invalid swaps
        mask_self = torch.arange(num_packs, device=device).view(1, -1).expand(num_layers, -1) == idx_max_pack.view(-1, 1)
        mask_self = mask_self.view(num_layers, num_packs, 1, 1)

        valid_mask = (diffs > 0) & (improvement > 1e-6) & (~mask_self)
        improvement = torch.where(valid_mask, improvement, torch.tensor(float('-inf'), device=device))

        # Find best swap
        flat_imp = improvement.view(num_layers, -1)
        best_imp, flat_idx = flat_imp.max(dim=1)

        if not (best_imp > float('-inf')).any():
            break

        # Decode indices
        k_sq = groups_per_pack * groups_per_pack
        m_idx = flat_idx // k_sq
        rem = flat_idx % k_sq
        k_max_idx = rem // groups_per_pack
        k_other_idx = rem % groups_per_pack

        l_valid = torch.nonzero(best_imp > float('-inf')).squeeze(1)

        if len(l_valid) > 0:
            p_max = idx_max_pack[l_valid]
            p_other = m_idx[l_valid]

            idx_in_max = k_max_idx[l_valid]
            idx_in_other = k_other_idx[l_valid]

            val_max_item = pack_contents[l_valid, p_max, idx_in_max]
            val_other_item = pack_contents[l_valid, p_other, idx_in_other]

            pack_contents[l_valid, p_max, idx_in_max] = val_other_item
            pack_contents[l_valid, p_other, idx_in_other] = val_max_item

    # 4. Construct outputs
    pack_ids = torch.arange(num_packs, device=device).view(1, num_packs, 1).expand(num_layers, -1, groups_per_pack)
    rank_ids = torch.arange(groups_per_pack, device=device).view(1, 1, groups_per_pack).expand(num_layers, num_packs, -1)

    flat_sorted_idx_ptr = pack_contents.view(num_layers, -1)
    flat_pack_ids = pack_ids.reshape(num_layers, -1)
    flat_rank_ids = rank_ids.reshape(num_layers, -1)

    original_idx = torch.gather(sorted_indices, 1, flat_sorted_idx_ptr)

    pack_index = torch.empty_like(flat_pack_ids)
    rank_in_pack = torch.empty_like(flat_rank_ids)

    pack_index.scatter_(1, original_idx, flat_pack_ids)
    rank_in_pack.scatter_(1, original_idx, flat_rank_ids)

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

    # Initialize with 1 replica per expert
    phy2log = torch.arange(num_phy, dtype=torch.int64, device=device).repeat(n, 1)
    rank = torch.zeros(n, num_phy, dtype=torch.int64, device=device)
    logcnt = torch.ones(n, num_log, dtype=torch.int64, device=device)
    arangen = torch.arange(n, dtype=torch.int64, device=device)

    # Greedily add replicas to the expert with highest current load per replica
    # We execute this sequentially because the greedy choice depends on the previous step.
    # However, for GPU, the operations inside the loop are vectorized over the batch dimension 'n'.

    for i in range(num_log, num_phy):
        # Metric: current load per replica = weight / count
        # Find expert with max metric
        metrics = weight / logcnt
        redundant_indices = metrics.max(dim=-1).indices # [N]

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
        num_nodes: number of server nodes
        num_gpus: number of GPUs

    Returns:
        physical_to_logical_map, logical_to_physical_map, logical_count
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
    """
    num_layers, num_logical_experts = weight.shape
    # NOTE: We keep weight on its original device to allow GPU acceleration.
    weight = weight.float()

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

    # Scatter to create the reverse map
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