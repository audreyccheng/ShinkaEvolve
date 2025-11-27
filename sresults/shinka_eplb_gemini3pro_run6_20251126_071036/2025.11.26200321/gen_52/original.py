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

    Uses a Massive Parallel Hybrid Initialization (Randomized LPT, Random, Folded LPT)
    followed by a Vectorized Max-Any Swap local search.

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

    if groups_per_pack == 1:
        pack_index = torch.arange(num_packs, dtype=torch.int64, device=device).expand(num_layers, -1)
        rank_in_pack = torch.zeros_like(pack_index)
        return pack_index, rank_in_pack

    # --- Configuration ---
    # Partition candidates into strategies
    num_cand_lpt = 64
    num_cand_rnd = 32
    num_cand_fold = 32
    num_candidates = num_cand_lpt + num_cand_rnd + num_cand_fold # 128
    num_problems = num_layers * num_candidates

    # Expand weights [L, C, N] for initialization logic
    w_base = weight.unsqueeze(1).expand(num_layers, num_candidates, num_groups)

    # --- Strategy 1: Randomized LPT (0..63) ---
    scales = torch.linspace(0, 0.5, num_cand_lpt, device=device).view(1, -1, 1)
    noise_lpt = torch.rand_like(w_base[:, :num_cand_lpt, :]) * w_base[:, :num_cand_lpt, :] * scales
    # Pure LPT for first candidate of each layer
    noise_lpt[:, 0, :] = 0
    keys_lpt = w_base[:, :num_cand_lpt, :] + noise_lpt

    # --- Strategy 2: Random Shuffle (64..95) ---
    keys_rnd = torch.rand_like(w_base[:, num_cand_lpt:num_cand_lpt+num_cand_rnd, :])

    # --- Strategy 3: Randomized Folded LPT (96..127) ---
    noise_fold = torch.rand_like(w_base[:, -num_cand_fold:, :]) * w_base[:, -num_cand_fold:, :] * 0.2
    keys_fold = w_base[:, -num_cand_fold:, :] + noise_fold

    # Combine keys and sort
    sort_keys = torch.cat([keys_lpt, keys_rnd, keys_fold], dim=1).reshape(num_problems, num_groups)
    w_expanded = w_base.reshape(num_problems, num_groups)

    _, sorted_indices = sort_keys.sort(dim=-1, descending=True)

    # Apply Folding Permutation to Strategy 3 indices
    sorted_indices_view = sorted_indices.view(num_layers, num_candidates, num_groups)

    fold_perm = torch.empty(num_groups, dtype=torch.long, device=device)
    fold_perm[0::2] = torch.arange((num_groups + 1) // 2, device=device)
    fold_perm[1::2] = torch.arange(num_groups - 1, (num_groups + 1) // 2 - 1, -1, device=device)

    # Gather folded part
    fold_slice = sorted_indices_view[:, -num_cand_fold:, :]
    fold_slice_folded = fold_slice.gather(2, fold_perm.view(1, 1, -1).expand(num_layers, num_cand_fold, -1))

    # Reconstruct sorted_indices with folded part
    sorted_indices = torch.cat([
        sorted_indices_view[:, :-num_cand_fold, :],
        fold_slice_folded
    ], dim=1).reshape(num_problems, num_groups)

    # Gather actual weights for processing
    sorted_weight = torch.gather(w_expanded, 1, sorted_indices)

    # --- Phase 1: Vectorized Greedy Assignment ---
    pack_weights = torch.zeros(num_problems, num_packs, device=device, dtype=weight.dtype)
    pack_counts = torch.zeros(num_problems, num_packs, device=device, dtype=torch.int64)
    sorted_pack_index = torch.zeros_like(sorted_indices)

    for i in range(num_groups):
        w_item = sorted_weight[:, i:i+1] # [LC, 1]

        # Mask full packs
        is_full = (pack_counts >= groups_per_pack)
        candidates = pack_weights.clone()
        candidates[is_full] = float('inf')

        chosen_pack = candidates.argmin(dim=1, keepdim=True)

        sorted_pack_index[:, i:i+1] = chosen_pack
        pack_weights.scatter_add_(1, chosen_pack, w_item)
        pack_counts.scatter_add_(1, chosen_pack, torch.ones_like(chosen_pack))

    # --- Phase 2: Vectorized Local Search (Max-Any Swap) ---
    _, pack_content_sort_idx = sorted_pack_index.sort(dim=1, stable=True)
    pack_contents = pack_content_sort_idx.view(num_problems, num_packs, groups_per_pack)

    num_iters = 20
    for _ in range(num_iters):
        flat_contents = pack_contents.view(num_problems, -1)
        current_weights = torch.gather(sorted_weight, 1, flat_contents).view(num_problems, num_packs, groups_per_pack)

        pack_sums = current_weights.sum(dim=2) # [LC, M]
        val_max, idx_max = pack_sums.max(dim=1) # [LC]

        gather_max = idx_max.view(-1, 1, 1).expand(-1, 1, groups_per_pack)
        w_max = torch.gather(current_weights, 1, gather_max).squeeze(1) # [LC, K]

        # diffs: [LC, M, K_max, K_other]
        w_max_exp = w_max.view(num_problems, 1, groups_per_pack, 1)
        curr_exp = current_weights.view(num_problems, num_packs, 1, groups_per_pack)
        diffs = w_max_exp - curr_exp

        val_max_exp = val_max.view(num_problems, 1, 1, 1)
        pack_sums_exp = pack_sums.view(num_problems, num_packs, 1, 1)

        new_pair_max = torch.max(val_max_exp - diffs, pack_sums_exp + diffs)
        improvement = val_max_exp - new_pair_max

        mask_self = (torch.arange(num_packs, device=device).view(1, -1) == idx_max.view(-1, 1))
        mask_self = mask_self.view(num_problems, num_packs, 1, 1)

        valid = (diffs > 0) & (improvement > 1e-6) & (~mask_self)
        improvement = torch.where(valid, improvement, torch.tensor(float('-inf'), device=device))

        flat_imp = improvement.view(num_problems, -1)
        best_imp, flat_idx = flat_imp.max(dim=1)

        if not (best_imp > float('-inf')).any():
            break

        active = torch.nonzero(best_imp > float('-inf')).squeeze(1)
        if len(active) == 0: break

        sel_idx = flat_idx[active]
        K = groups_per_pack
        K2 = K * K

        p_other = sel_idx // K2
        rem = sel_idx % K2
        idx_in_max = rem // K
        idx_in_other = rem % K

        p_max = idx_max[active]

        v_max = pack_contents[active, p_max, idx_in_max]
        v_other = pack_contents[active, p_other, idx_in_other]

        pack_contents[active, p_max, idx_in_max] = v_other
        pack_contents[active, p_other, idx_in_other] = v_max

    # --- Phase 3: Selection ---
    flat_contents = pack_contents.view(num_problems, -1)
    final_weights = torch.gather(sorted_weight, 1, flat_contents).view(num_problems, num_packs, groups_per_pack)
    final_max_loads = final_weights.sum(dim=2).max(dim=1).values

    reshaped = final_max_loads.view(num_layers, num_candidates)
    best_cand = reshaped.argmin(dim=1)

    best_problem_idx = torch.arange(num_layers, device=device) * num_candidates + best_cand

    best_contents = pack_contents[best_problem_idx]
    best_sorted_indices = sorted_indices[best_problem_idx]

    flat_best_contents = best_contents.view(num_layers, -1)
    original_idx = torch.gather(best_sorted_indices, 1, flat_best_contents)

    pack_ids = torch.arange(num_packs, device=device).view(1, num_packs, 1).expand(num_layers, -1, groups_per_pack).reshape(num_layers, -1)
    rank_ids = torch.arange(groups_per_pack, device=device).view(1, 1, groups_per_pack).expand(num_layers, num_packs, -1).reshape(num_layers, -1)

    pack_index = torch.empty_like(pack_ids)
    rank_in_pack = torch.empty_like(rank_ids)

    pack_index.scatter_(1, original_idx, pack_ids)
    rank_in_pack.scatter_(1, original_idx, rank_ids)

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