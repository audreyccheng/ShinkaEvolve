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

    Uses a Two-Stage Evolutionary Algorithm:
    1. Exploration: Randomized LPT + Interleaved Sorting.
    2. Exploitation: Rank-based Mutation of best candidates.
    Followed by Vectorized Max-Any Swap Local Search.

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

    # --- Configuration ---
    # Stage 1: Exploration
    num_expl_lpt = 16
    num_expl_int = 16
    num_expl = num_expl_lpt + num_expl_int # 32

    # Stage 2: Exploitation
    num_exploit = 96

    total_candidates = num_expl + num_exploit # 128

    # --- Helper: Vectorized Greedy Packer ---
    def run_greedy(indices_in, batch_w):
        """
        indices_in: [Batch, N]
        batch_w: [Batch, N] (weights sorted according to indices_in)
        """
        batch_size = indices_in.shape[0]

        pack_w = torch.zeros(batch_size, num_packs, device=device, dtype=weight.dtype)
        pack_c = torch.zeros(batch_size, num_packs, device=device, dtype=torch.int64)
        sorted_p_idx = torch.zeros_like(indices_in)

        inf_val = torch.tensor(float('inf'), device=device)

        for i in range(num_groups):
            w_item = batch_w[:, i:i+1]

            # Mask full packs
            is_full = (pack_c >= groups_per_pack)
            cand_w = torch.where(is_full, inf_val, pack_w)

            # Choose pack with min weight
            chosen = cand_w.argmin(dim=1, keepdim=True)

            sorted_p_idx[:, i:i+1] = chosen
            pack_w.scatter_add_(1, chosen, w_item)
            pack_c.scatter_add_(1, chosen, torch.ones_like(chosen))

        return sorted_p_idx, pack_w

    # --- Stage 1: Exploration ---
    # [L, 1, N]
    w_base = weight.unsqueeze(1)

    # 1a. Randomized LPT
    scales_lpt = torch.linspace(0, 0.3, num_expl_lpt, device=device).view(1, -1, 1)
    noise_lpt = torch.rand(num_layers, num_expl_lpt, num_groups, device=device, dtype=weight.dtype) * w_base * scales_lpt
    noise_lpt[:, 0, :] = 0 # Cand 0 is pure LPT
    keys_lpt = w_base + noise_lpt
    _, indices_lpt = keys_lpt.sort(dim=-1, descending=True)

    # 1b. Randomized Interleaved
    # We apply noise to LPT keys, sort, then apply interleaving pattern.
    scales_int = torch.linspace(0, 0.3, num_expl_int, device=device).view(1, -1, 1)
    noise_int = torch.rand(num_layers, num_expl_int, num_groups, device=device, dtype=weight.dtype) * w_base * scales_int
    keys_int_pre = w_base + noise_int
    _, indices_int_sorted = keys_int_pre.sort(dim=-1, descending=True)

    # Interleaving Map: 0, N-1, 1, N-2, ...
    interleave_map = torch.empty(num_groups, dtype=torch.long, device=device)
    interleave_map[0::2] = torch.arange((num_groups + 1) // 2, device=device)
    interleave_map[1::2] = torch.arange(num_groups - 1, (num_groups + 1) // 2 - 1, step=-1, device=device)

    indices_int = indices_int_sorted.gather(2, interleave_map.view(1, 1, -1).expand(num_layers, num_expl_int, -1))

    # Combine Stage 1
    # [L, 32, N]
    stage1_indices = torch.cat([indices_lpt, indices_int], dim=1)
    flat_stage1_indices = stage1_indices.reshape(num_layers * num_expl, num_groups)

    # Gather weights for greedy
    w_s1 = weight.repeat_interleave(num_expl, dim=0)
    w_s1_sorted = torch.gather(w_s1, 1, flat_stage1_indices)

    # Run Greedy S1
    s1_pack_idx, s1_loads = run_greedy(flat_stage1_indices, w_s1_sorted)

    # Select Best S1
    s1_max_loads = s1_loads.max(dim=1).values.view(num_layers, num_expl)
    best_s1_idx = s1_max_loads.argmin(dim=1) # [L]

    # Gather best indices: [L, N]
    gather_idx = best_s1_idx.view(num_layers, 1, 1).expand(-1, 1, num_groups)
    best_indices_s1 = stage1_indices.gather(1, gather_idx).squeeze(1)

    # --- Stage 2: Exploitation (Mutation) ---
    # Create rank scores based on best permutation
    # pos_scores[l, i] = rank of item i in the best permutation of layer l
    # To do this efficiently: pos_scores.scatter_(1, best_indices, range)
    pos_scores = torch.empty_like(best_indices_s1, dtype=torch.float32)
    src_range = torch.arange(num_groups, device=device, dtype=torch.float32).expand(num_layers, -1)
    pos_scores.scatter_(1, best_indices_s1, src_range)

    # Expand: [L, 96, N]
    pos_expanded = pos_scores.unsqueeze(1).expand(-1, num_exploit, -1)

    # Add noise to ranks to induce local swaps
    noise_levels = torch.linspace(0.5, 4.0, num_exploit, device=device).view(1, -1, 1)
    rank_noise = torch.randn(num_layers, num_exploit, num_groups, device=device) * noise_levels
    new_scores = pos_expanded + rank_noise

    # Sort to get mutated indices
    _, stage2_indices = new_scores.sort(dim=-1)

    flat_stage2_indices = stage2_indices.reshape(num_layers * num_exploit, num_groups)

    # Run Greedy S2
    w_s2 = weight.repeat_interleave(num_exploit, dim=0)
    w_s2_sorted = torch.gather(w_s2, 1, flat_stage2_indices)
    s2_pack_idx, s2_loads = run_greedy(flat_stage2_indices, w_s2_sorted)

    # --- Combine and Local Search ---
    # We process all 128 candidates.

    # [L, 128, N]
    s1_r = s1_pack_idx.view(num_layers, num_expl, num_groups)
    s2_r = s2_pack_idx.view(num_layers, num_exploit, num_groups)
    combined_pack_idx = torch.cat([s1_r, s2_r], dim=1).view(num_layers * total_candidates, num_groups)

    s1_i = flat_stage1_indices.view(num_layers, num_expl, num_groups)
    s2_i = flat_stage2_indices.view(num_layers, num_exploit, num_groups)
    combined_indices = torch.cat([s1_i, s2_i], dim=1).view(num_layers * total_candidates, num_groups)

    w_all = weight.repeat_interleave(total_candidates, dim=0)
    sorted_weight = torch.gather(w_all, 1, combined_indices)

    # Setup for LS
    _, pack_content_sort_idx = combined_pack_idx.sort(dim=1, stable=True)
    pack_contents = pack_content_sort_idx.view(num_layers * total_candidates, num_packs, groups_per_pack)

    # Vectorized Max-Any Swap
    num_iters = 20
    for _ in range(num_iters):
        flat_contents = pack_contents.view(-1, groups_per_pack * num_packs)
        curr_w = torch.gather(sorted_weight, 1, flat_contents).view(-1, num_packs, groups_per_pack)

        pack_sums = curr_w.sum(dim=2)
        val_max, idx_max = pack_sums.max(dim=1)

        # Identify Max Pack Items
        gather_max = idx_max.view(-1, 1, 1).expand(-1, 1, groups_per_pack)
        w_max = torch.gather(curr_w, 1, gather_max).squeeze(1) # [Batch, K]

        # Calculate Diffs for swapping Max Items with Any Items
        # diffs: [Batch, M, K_max, K_other]
        # w_max: [Batch, 1, K, 1]
        # curr_w: [Batch, M, 1, K]
        diffs = w_max.unsqueeze(1).unsqueeze(3) - curr_w.unsqueeze(2)

        # Objective: minimize max(new_max_pack, new_other_pack)
        v_max_exp = val_max.view(-1, 1, 1, 1)
        p_sums_exp = pack_sums.view(-1, num_packs, 1, 1)

        new_max_load = v_max_exp - diffs
        new_other_load = p_sums_exp + diffs

        pair_max = torch.max(new_max_load, new_other_load)
        improvement = v_max_exp - pair_max

        # Mask self-swaps and invalid moves
        mask_self = (torch.arange(num_packs, device=device).view(1, -1) == idx_max.view(-1, 1))
        mask_self = mask_self.view(-1, num_packs, 1, 1)

        # Requirements: Strictly reduce load on max pack (diffs > 0) AND improvement > threshold
        valid = (diffs > 0) & (improvement > 1e-6) & (~mask_self)

        score = torch.where(valid, improvement, torch.tensor(float('-inf'), device=device))

        # Find best move per problem
        flat_score = score.view(score.shape[0], -1)
        best_imp, best_flat_idx = flat_score.max(dim=1)

        if not (best_imp > 0).any():
            break

        active = torch.nonzero(best_imp > 0).squeeze(1)
        if len(active) == 0: break

        # Execute Swaps
        sel = best_flat_idx[active]
        K = groups_per_pack
        K2 = K*K

        p_other = sel // K2
        rem = sel % K2
        idx_in_max = rem // K
        idx_in_other = rem % K

        p_max_act = idx_max[active]

        # Swap content pointers
        v_max_ptr = pack_contents[active, p_max_act, idx_in_max].clone()
        v_oth_ptr = pack_contents[active, p_other, idx_in_other].clone()

        pack_contents[active, p_max_act, idx_in_max] = v_oth_ptr
        pack_contents[active, p_other, idx_in_other] = v_max_ptr

    # --- Final Selection ---
    flat_contents = pack_contents.view(-1, num_packs * groups_per_pack)
    final_w = torch.gather(sorted_weight, 1, flat_contents).view(-1, num_packs, groups_per_pack)
    final_max = final_w.sum(dim=2).max(dim=1).values

    final_max = final_max.view(num_layers, total_candidates)
    best_cand = final_max.argmin(dim=1)

    best_idx_flat = torch.arange(num_layers, device=device) * total_candidates + best_cand

    best_c = pack_contents[best_idx_flat]
    best_sort_idx = combined_indices[best_idx_flat]

    # Map back to original indices
    flat_best_c = best_c.view(num_layers, -1)
    orig_idx = torch.gather(best_sort_idx, 1, flat_best_c)

    # Generate Output
    pack_ids = torch.arange(num_packs, device=device).view(1, num_packs, 1).expand(num_layers, -1, groups_per_pack).reshape(num_layers, -1)
    rank_ids = torch.arange(groups_per_pack, device=device).view(1, 1, groups_per_pack).expand(num_layers, num_packs, -1).reshape(num_layers, -1)

    pack_index = torch.empty_like(pack_ids)
    rank_in_pack = torch.empty_like(rank_ids)

    pack_index.scatter_(1, orig_idx, pack_ids)
    rank_in_pack.scatter_(1, orig_idx, rank_ids)

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
    for i in range(num_log, num_phy):
        # metric = weight / count
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
    tokens_per_mlog = weight.gather(-1, mlog2log).view(
        -1, num_logical_experts // num_nodes)

    phy2mlog, phyrank, mlogcnt = replicate_experts(
        tokens_per_mlog, num_physical_experts // num_nodes)

    # Step 3: pack physical_experts to GPUs
    tokens_per_phy = (tokens_per_mlog / mlogcnt).gather(-1, phy2mlog)

    pack_index, rank_in_pack = balanced_packing(tokens_per_phy,
                                                num_gpus // num_nodes)

    phy2pphy = pack_index * phy_experts_per_gpu + rank_in_pack
    pphy2phy = inverse(phy2pphy)

    pphy2mlog = phy2mlog.gather(
        -1, pphy2phy)
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
    weight = weight.float()

    if num_groups % num_nodes == 0:
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, num_groups, num_nodes, num_gpus)
    else:
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