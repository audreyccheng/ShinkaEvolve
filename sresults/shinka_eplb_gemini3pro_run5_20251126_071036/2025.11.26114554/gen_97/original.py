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


def _refine_packing_batched(weights: torch.Tensor,
                            pack_indices: torch.Tensor,
                            pack_weights: torch.Tensor,
                            num_packs: int,
                            groups_per_pack: int,
                            max_iters: int = 50) -> torch.Tensor:
    """
    Batched vectorized refinement minimizing max load.
    Uses 1-for-1 swaps (Max vs All) and 2-for-2 swaps (Max vs Min).

    Args:
        weights: [B, num_total_items] Real weights of items.
        pack_indices: [B, num_packs, groups_per_pack] Item IDs in each pack.
        pack_weights: [B, num_packs] Current loads.
    """
    B = weights.shape[0]
    device = weights.device

    # Pre-compute indices for 2-swap upper triangle (distinct pairs)
    if groups_per_pack >= 2:
        g_idx = torch.arange(groups_per_pack, device=device)
        grid_i, grid_j = torch.meshgrid(g_idx, g_idx, indexing='ij')
        mask_upper = grid_i < grid_j
        pair_idx_i = grid_i[mask_upper] # [NumPairs]
        pair_idx_j = grid_j[mask_upper] # [NumPairs]
        num_pairs = pair_idx_i.shape[0]
    else:
        num_pairs = 0

    batch_idx = torch.arange(B, device=device)

    for _ in range(max_iters):
        # --- Step 1: 1-for-1 Swap (Global) ---

        # Identify max load pack
        max_vals, max_pids = torch.max(pack_weights, dim=1)

        # Gather items in max packs: [B, G]
        gather_indices = max_pids.view(B, 1, 1).expand(B, 1, groups_per_pack)
        items_max = torch.gather(pack_indices, 1, gather_indices).squeeze(1)
        w_max = torch.gather(weights, 1, items_max)

        # Gather weights of ALL items: [B, M, G]
        w_all = torch.gather(weights, 1, pack_indices.view(B, -1)).view(B, num_packs, groups_per_pack)

        # Deltas: w_max[i] - w_all[k, j]
        # [B, G, 1, 1] - [B, 1, M, G] -> [B, G, M, G]
        deltas = w_max.unsqueeze(1).unsqueeze(2) - w_all.unsqueeze(3)

        new_max_load = max_vals.view(B, 1, 1, 1) - deltas
        new_other_load = pack_weights.view(B, num_packs, 1, 1) + deltas
        objectives = torch.max(new_max_load, new_other_load)

        # Mask self-swaps
        mask_k = (torch.arange(num_packs, device=device).unsqueeze(0) == max_pids.unsqueeze(1))
        mask = mask_k.view(B, num_packs, 1, 1).expand(B, num_packs, groups_per_pack, groups_per_pack)
        objectives.masked_fill_(mask, float('inf'))

        flat_obj = objectives.view(B, -1)
        min_obj, flat_best_idx = torch.min(flat_obj, dim=1)

        # Check 1-swap improvement
        improve_mask_1 = min_obj < (max_vals - 1e-6)

        if improve_mask_1.any():
            # Apply 1-swaps
            active_batch = batch_idx[improve_mask_1]
            best_idx = flat_best_idx[active_batch]

            G = groups_per_pack
            G2 = G * G

            k_idx = best_idx // G2
            rem = best_idx % G2
            j_idx = rem // G
            i_idx = rem % G

            curr_max_pids = max_pids[active_batch]

            val_from_max = pack_indices[active_batch, curr_max_pids, i_idx]
            val_from_other = pack_indices[active_batch, k_idx, j_idx]

            pack_indices[active_batch, curr_max_pids, i_idx] = val_from_other
            pack_indices[active_batch, k_idx, j_idx] = val_from_max

            w_max_v = weights[active_batch, val_from_max]
            w_other_v = weights[active_batch, val_from_other]
            d_val = w_max_v - w_other_v

            pack_weights[active_batch, curr_max_pids] -= d_val
            pack_weights[active_batch, k_idx] += d_val

            # Continue to next iter, skipping 2-swap for these batches

        # --- Step 2: 2-for-2 Swap (Max vs Min) ---
        # Only run for batches where 1-swap failed or wasn't tried for improvement
        # We need improvement, so we only run on batches that didn't just improve via 1-swap

        if num_pairs == 0:
            if not improve_mask_1.any():
                break
            continue

        candidates_mask = ~improve_mask_1
        if not candidates_mask.any():
            continue

        active_batch = batch_idx[candidates_mask]

        # Identify Min Pack for these candidates
        min_vals, min_pids = torch.min(pack_weights[active_batch], dim=1)
        curr_max_pids = max_pids[active_batch]
        curr_max_vals = max_vals[active_batch]

        # Gather items: [B_sub, G]
        gather_max = curr_max_pids.view(-1, 1, 1).expand(-1, 1, groups_per_pack)
        items_max_sub = torch.gather(pack_indices[active_batch], 1, gather_max).squeeze(1)
        w_max_sub = torch.gather(weights[active_batch], 1, items_max_sub) # [B_sub, G]

        gather_min = min_pids.view(-1, 1, 1).expand(-1, 1, groups_per_pack)
        items_min_sub = torch.gather(pack_indices[active_batch], 1, gather_min).squeeze(1)
        w_min_sub = torch.gather(weights[active_batch], 1, items_min_sub) # [B_sub, G]

        # Compute Pair Sums: [B_sub, NumPairs]
        pair_sum_max = w_max_sub[:, pair_idx_i] + w_max_sub[:, pair_idx_j]
        pair_sum_min = w_min_sub[:, pair_idx_i] + w_min_sub[:, pair_idx_j]

        # Deltas: PairMax - PairMin
        # [B_sub, NumPairs, 1] - [B_sub, 1, NumPairs] -> [B_sub, NP, NP]
        deltas_2 = pair_sum_max.unsqueeze(2) - pair_sum_min.unsqueeze(1)

        # New Loads
        new_max_load_2 = curr_max_vals.view(-1, 1, 1) - deltas_2
        new_min_load_2 = min_vals.view(-1, 1, 1) + deltas_2

        # Objective: minimize max(new_max, new_min)
        obj_2 = torch.max(new_max_load_2, new_min_load_2)

        flat_obj_2 = obj_2.view(active_batch.shape[0], -1)
        min_obj_2, best_idx_2 = torch.min(flat_obj_2, dim=1)

        improve_mask_2 = min_obj_2 < (curr_max_vals - 1e-6)

        if improve_mask_2.any():
            # Execute 2-swaps
            final_active_batch = active_batch[improve_mask_2]
            best_idx_2_sub = best_idx_2[improve_mask_2]

            idx_max_pair = best_idx_2_sub // num_pairs
            idx_min_pair = best_idx_2_sub % num_pairs

            m1 = pair_idx_i[idx_max_pair]
            m2 = pair_idx_j[idx_max_pair]
            n1 = pair_idx_i[idx_min_pair]
            n2 = pair_idx_j[idx_min_pair]

            p_max = curr_max_pids[improve_mask_2]
            p_min = min_pids[improve_mask_2]

            # Read values
            val_max_1 = pack_indices[final_active_batch, p_max, m1]
            val_max_2 = pack_indices[final_active_batch, p_max, m2]
            val_min_1 = pack_indices[final_active_batch, p_min, n1]
            val_min_2 = pack_indices[final_active_batch, p_min, n2]

            # Write values (Swap)
            pack_indices[final_active_batch, p_max, m1] = val_min_1
            pack_indices[final_active_batch, p_max, m2] = val_min_2
            pack_indices[final_active_batch, p_min, n1] = val_max_1
            pack_indices[final_active_batch, p_min, n2] = val_max_2

            # Update weights
            d_val = deltas_2.view(active_batch.shape[0], -1)[improve_mask_2, best_idx_2_sub]

            pack_weights[final_active_batch, p_max] -= d_val
            pack_weights[final_active_batch, p_min] += d_val

        if not improve_mask_1.any() and not improve_mask_2.any():
            break

    return pack_indices


def balanced_packing(weight: torch.Tensor,
                     num_packs: int,
                     num_candidates: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pack n weighted objects to m packs.
    Strategy: Parallel Randomized Greedy LPT + Hybrid 1-item/2-item Refinement.
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

    # Batch Size
    B = num_layers * num_candidates

    # [L, K, G] -> [B, G]
    weights_expanded = weight.unsqueeze(1).expand(-1, num_candidates, -1).reshape(B, num_groups)

    # Randomized Sort Keys
    noise = torch.rand(B, num_groups, device=device) * 0.05
    # Keep one candidate pure greedy per layer
    noise.view(num_layers, num_candidates, num_groups)[:, 0, :] = 0.0

    sort_keys = weights_expanded * (1.0 + noise)
    sorted_res = torch.sort(sort_keys, dim=1, descending=True)
    sorted_indices = sorted_res.indices

    sorted_weights = torch.gather(weights_expanded, 1, sorted_indices)

    # 1. Greedy Packing
    pack_weights = torch.zeros(B, num_packs, device=device, dtype=weight.dtype)
    pack_counts = torch.zeros(B, num_packs, device=device, dtype=torch.int64)
    pack_indices = torch.zeros(B, num_packs, groups_per_pack, device=device, dtype=torch.int64)

    batch_idx = torch.arange(B, device=device)
    inf_val = float('inf')

    for i in range(num_groups):
        w = sorted_weights[:, i]
        idx = sorted_indices[:, i]

        is_full = pack_counts >= groups_per_pack
        cand_weights = torch.where(is_full, inf_val, pack_weights)
        best_pack = torch.argmin(cand_weights, dim=1)

        slots = pack_counts[batch_idx, best_pack]
        pack_indices[batch_idx, best_pack, slots] = idx
        pack_weights[batch_idx, best_pack] += w
        pack_counts[batch_idx, best_pack] += 1

    # 2. Hybrid Refinement (1-swap & 2-swap)
    pack_indices = _refine_packing_batched(
        weights_expanded,
        pack_indices,
        pack_weights,
        num_packs,
        groups_per_pack,
        max_iters=50
    )

    # 3. Select Best Candidate
    pw_reshaped = pack_weights.view(num_layers, num_candidates, num_packs)
    max_loads = torch.max(pw_reshaped, dim=2).values
    best_k = torch.argmin(max_loads, dim=1)

    pi_reshaped = pack_indices.view(num_layers, num_candidates, num_packs, groups_per_pack)
    best_assignment = pi_reshaped[torch.arange(num_layers, device=device), best_k]

    # 4. Construct Output
    flat_assignment = best_assignment.view(num_layers, -1)

    p_ids = torch.arange(num_packs, device=device).unsqueeze(1).expand(-1, groups_per_pack).reshape(1, -1).expand(num_layers, -1)
    r_ids = torch.arange(groups_per_pack, device=device).unsqueeze(0).expand(num_packs, -1).reshape(1, -1).expand(num_layers, -1)

    pack_index = torch.empty_like(weight, dtype=torch.int64)
    rank_in_pack = torch.empty_like(weight, dtype=torch.int64)

    pack_index.scatter_(1, flat_assignment, p_ids)
    rank_in_pack.scatter_(1, flat_assignment, r_ids)

    return pack_index, rank_in_pack


def replicate_experts(
        weight: torch.Tensor,
        num_phy: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Replicate experts to `num_phy` replicas, minimizing the maximum load.
    Vectorized greedy.
    """
    n, num_log = weight.shape
    device = weight.device

    logcnt = torch.ones((n, num_log), dtype=torch.int64, device=device)

    for _ in range(num_log, num_phy):
        scores = weight / logcnt
        indices = torch.argmax(scores, dim=-1)
        rows = torch.arange(n, device=device)
        logcnt[rows, indices] += 1

    phy2log = torch.zeros((n, num_phy), dtype=torch.int64, device=device)
    rank = torch.zeros((n, num_phy), dtype=torch.int64, device=device)

    for i in range(n):
        counts = logcnt[i]
        l_ids = torch.repeat_interleave(
            torch.arange(num_log, device=device), counts
        )
        phy2log[i] = l_ids

        curr = 0
        for idx in range(num_log):
            c = counts[idx].item()
            rank[i, curr:curr+c] = torch.arange(c, device=device)
            curr += c

    return phy2log, rank, logcnt


def rebalance_experts_hierarchical(
    weight: torch.Tensor,
    num_physical_experts: int,
    num_groups: int,
    num_nodes: int,
    num_gpus: int,
):
    """
    Hierarchical rebalancing: Groups->Nodes, then Replicas->GPUs.
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
    weight = weight.float().cpu()

    # Dispatch policy
    if num_groups % num_nodes == 0:
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, num_groups, num_nodes, num_gpus)
    else:
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, 1, 1, num_gpus)

    num_redundant_experts = num_replicas - num_logical_experts
    maxlogcnt = num_redundant_experts + 1

    # Construct reverse mapping log2phy
    log2phy: torch.Tensor = torch.full(
        (num_layers, num_logical_experts, maxlogcnt),
        -1,
        dtype=torch.int64,
        device=logcnt.device,
    )

    # Efficient scatter
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