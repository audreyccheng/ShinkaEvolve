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
    Batched vectorized refinement minimizing max load by swapping items
    between the heaviest pack and any other pack.

    Args:
        weights: [B, num_total_items] Real weights of items.
        pack_indices: [B, num_packs, groups_per_pack] Item IDs in each pack.
        pack_weights: [B, num_packs] Current loads.

    Returns:
        Refined pack_indices.
    """
    B = weights.shape[0]
    device = weights.device

    # Range for batch indexing
    batch_idx = torch.arange(B, device=device)

    for _ in range(max_iters):
        # 1. Identify max load pack for each batch
        # max_vals: [B], max_pids: [B]
        max_vals, max_pids = torch.max(pack_weights, dim=1)

        # 2. Gather items in max packs: [B, G]
        # max_pids shape [B] -> [B, 1, 1] -> expand to [B, 1, G]
        gather_indices = max_pids.view(B, 1, 1).expand(B, 1, groups_per_pack)
        # Gather from pack_indices [B, M, G] -> [B, 1, G] -> squeeze -> [B, G]
        items_max = torch.gather(pack_indices, 1, gather_indices).squeeze(1)

        # Gather weights of these items: [B, G]
        w_max = torch.gather(weights, 1, items_max)

        # 3. Gather weights of ALL items in current packing layout: [B, M, G]
        # Flatten pack_indices to [B, M*G] for gather, then reshape
        w_all = torch.gather(weights, 1, pack_indices.view(B, -1)).view(B, num_packs, groups_per_pack)

        # 4. Compute Deltas
        # We consider swapping item i (from max pack) with item j (from pack k)
        # delta = w_max[i] - w_all[k, j]
        # w_max: [B, G] -> [B, 1, 1, G] (dim 3 is i)
        # w_all: [B, M, G] -> [B, M, G, 1] (dim 2 is j, dim 1 is k)
        # deltas: [B, M, G, G]
        deltas = w_max.unsqueeze(1).unsqueeze(2) - w_all.unsqueeze(3)

        # 5. Compute new loads
        # New Max Load = Old Max - delta
        # New Other Load = Old Other + delta
        new_max_load = max_vals.view(B, 1, 1, 1) - deltas
        new_other_load = pack_weights.view(B, num_packs, 1, 1) + deltas

        # Objective: max(new_max_load, new_other_load)
        objectives = torch.max(new_max_load, new_other_load)

        # 6. Masking
        # Mask out swaps where k == max_pid
        mask_k = (torch.arange(num_packs, device=device).unsqueeze(0) == max_pids.unsqueeze(1))
        mask = mask_k.view(B, num_packs, 1, 1).expand(B, num_packs, groups_per_pack, groups_per_pack)

        objectives.masked_fill_(mask, float('inf'))

        # 7. Find best swap
        # Flatten objectives to [B, M*G*G]
        flat_obj = objectives.view(B, -1)
        min_obj, flat_best_idx = torch.min(flat_obj, dim=1)

        # 8. Check improvement
        improvement_mask = min_obj < (max_vals - 1e-5)

        if not improvement_mask.any():
            break

        # 9. Update (only for batches with improvement)
        active_batch_indices = batch_idx[improvement_mask]

        if len(active_batch_indices) == 0:
            break

        # Decode indices
        best_idx = flat_best_idx[active_batch_indices]
        # Dimensions for decoding
        G = groups_per_pack
        G2 = G * G

        # best_idx is into [M, G, G]
        k_idx = best_idx // G2
        rem = best_idx % G2
        j_idx = rem // G  # item in other pack
        i_idx = rem % G   # item in max pack

        curr_max_pids = max_pids[active_batch_indices]

        # Values to swap
        val_from_max = pack_indices[active_batch_indices, curr_max_pids, i_idx]
        val_from_other = pack_indices[active_batch_indices, k_idx, j_idx]

        # Perform swap
        pack_indices[active_batch_indices, curr_max_pids, i_idx] = val_from_other
        pack_indices[active_batch_indices, k_idx, j_idx] = val_from_max

        # Update weights
        w_max_val = weights[active_batch_indices, val_from_max]
        w_other_val = weights[active_batch_indices, val_from_other]
        d_val = w_max_val - w_other_val

        pack_weights[active_batch_indices, curr_max_pids] -= d_val
        pack_weights[active_batch_indices, k_idx] += d_val

    return pack_indices


def _refine_packing_2swap(weights: torch.Tensor,
                          pack_indices: torch.Tensor,
                          pack_weights: torch.Tensor,
                          num_packs: int,
                          groups_per_pack: int,
                          max_iters: int = 5) -> torch.Tensor:
    """
    Batched vectorized refinement minimizing max load by swapping TWO items
    between the heaviest pack and any other pack.
    """
    L, M, G = pack_indices.shape
    device = weights.device

    if G < 2:
        return pack_indices

    # Pre-compute pairs indices for G items
    triu_indices = torch.triu_indices(G, G, offset=1, device=device)
    u_idx, v_idx = triu_indices[0], triu_indices[1]
    num_pairs = u_idx.shape[0]

    batch_idx = torch.arange(L, device=device)

    for _ in range(max_iters):
        # 1. Identify max load pack
        max_vals, max_pids = torch.max(pack_weights, dim=1) # [L]

        # 2. Gather weights of items in max pack
        items_max = pack_indices[batch_idx, max_pids] # [L, G]
        w_max = torch.gather(weights, 1, items_max)   # [L, G]

        # 3. Compute pair sums for max pack
        w_max_pairs = w_max[:, u_idx] + w_max[:, v_idx] # [L, P]

        # 4. Gather weights of ALL items
        flat_pi = pack_indices.view(L, -1)
        w_all = torch.gather(weights, 1, flat_pi).view(L, M, G)

        # 5. Compute pair sums for all packs
        w_all_pairs = w_all[:, :, u_idx] + w_all[:, :, v_idx] # [L, M, P]

        # 6. Deltas: Pair from Max - Pair from Other
        deltas = w_max_pairs.unsqueeze(1).unsqueeze(3) - w_all_pairs.unsqueeze(2)

        # 7. New Loads
        new_max_load = max_vals.view(L, 1, 1, 1) - deltas
        new_other_load = pack_weights.view(L, M, 1, 1) + deltas

        objectives = torch.max(new_max_load, new_other_load)

        # 8. Mask invalid swaps (same pack)
        mask_k = (torch.arange(M, device=device).unsqueeze(0) == max_pids.unsqueeze(1))
        mask = mask_k.view(L, M, 1, 1).expand(L, M, num_pairs, num_pairs)
        objectives.masked_fill_(mask, float('inf'))

        # 9. Find Best Swap
        flat_obj = objectives.view(L, -1)
        min_obj, best_idx_flat = torch.min(flat_obj, dim=1)

        improve = min_obj < (max_vals - 1e-5)
        if not improve.any():
            break

        active = batch_idx[improve]
        if len(active) == 0:
            break

        # Decode indices
        idx_flat = best_idx_flat[improve]
        P = num_pairs
        P2 = P * P

        k_idx = idx_flat // P2
        rem = idx_flat % P2
        pair_max_idx = rem // P
        pair_other_idx = rem % P

        curr_max_pids = max_pids[active]
        curr_other_pids = k_idx

        u_max = u_idx[pair_max_idx]
        v_max = v_idx[pair_max_idx]
        u_other = u_idx[pair_other_idx]
        v_other = v_idx[pair_other_idx]

        # Values to swap
        val_max_u = pack_indices[active, curr_max_pids, u_max]
        val_max_v = pack_indices[active, curr_max_pids, v_max]
        val_other_u = pack_indices[active, curr_other_pids, u_other]
        val_other_v = pack_indices[active, curr_other_pids, v_other]

        # Swap
        pack_indices[active, curr_max_pids, u_max] = val_other_u
        pack_indices[active, curr_max_pids, v_max] = val_other_v
        pack_indices[active, curr_other_pids, u_other] = val_max_u
        pack_indices[active, curr_other_pids, v_other] = val_max_v

        # Update loads
        d_val = deltas.view(L, -1)[active, idx_flat]
        pack_weights[active, curr_max_pids] -= d_val
        pack_weights[active, curr_other_pids] += d_val

    return pack_indices


def balanced_packing(weight: torch.Tensor,
                     num_packs: int,
                     num_candidates: int = 64) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pack n weighted objects to m packs.

    Strategy:
    1. Parallel Randomized Greedy Initialization (LPT with variable noise).
    2. Batched Refinement (Min-Max Load, 1-swap).
    3. Select best candidate.
    4. Fine-grained Refinement (2-swap) on best candidate.
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

    # Expand for candidates
    B = num_layers * num_candidates
    weights_expanded = weight.unsqueeze(1).expand(-1, num_candidates, -1).reshape(B, num_groups)

    # Generate Sort Keys with Variable Noise (0% to 10%)
    noise_scales = torch.linspace(0.0, 0.1, num_candidates, device=device).view(1, num_candidates, 1)
    rand_noise = torch.rand(num_layers, num_candidates, num_groups, device=device)

    # [B, G]
    noise_vector = (rand_noise * noise_scales).reshape(B, num_groups)

    sort_keys = weights_expanded * (1.0 + noise_vector)

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

    # 2. Batched 1-Swap Refinement
    pack_indices = _refine_packing_batched(
        weights_expanded, pack_indices, pack_weights, num_packs, groups_per_pack
    )

    # 3. Select Best Candidate
    pw_reshaped = pack_weights.view(num_layers, num_candidates, num_packs)
    max_loads = torch.max(pw_reshaped, dim=2).values
    best_k = torch.argmin(max_loads, dim=1)

    pi_reshaped = pack_indices.view(num_layers, num_candidates, num_packs, groups_per_pack)
    best_assignment = pi_reshaped[torch.arange(num_layers, device=device), best_k]

    # 4. Fine-grained 2-Swap Refinement on Best Solution
    # Need to re-compute pack_weights for the best assignment
    flat_best = best_assignment.view(num_layers, -1)
    w_best = torch.gather(weight, 1, flat_best).view(num_layers, num_packs, groups_per_pack)
    pack_weights_best = w_best.sum(dim=2)

    best_assignment = _refine_packing_2swap(
        weight, best_assignment, pack_weights_best,
        num_packs, groups_per_pack
    )

    # 5. Construct Output Maps
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

    Uses a vectorized greedy approach (Jefferson/D'Hondt method).
    """
    n, num_log = weight.shape
    device = weight.device

    # Initialize: Each expert gets at least 1 replica
    logcnt = torch.ones((n, num_log), dtype=torch.int64, device=device)

    # Greedily assign remaining replicas
    for _ in range(num_log, num_phy):
        # Score is current load per replica
        scores = weight / logcnt
        # Find expert with max score in each layer
        indices = torch.argmax(scores, dim=-1)
        # Add replica
        rows = torch.arange(n, device=device)
        logcnt[rows, indices] += 1

    # Reconstruction of the mapping tables from counts
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
    # Sum weights within each group
    tokens_per_group = weight.unflatten(-1, (num_groups, group_size)).sum(-1)

    # Use improved balanced packing
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

    # Use improved balanced packing
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
        # use hierarchical load-balance policy
        phy2log, phyrank, logcnt = rebalance_experts_hierarchical(
            weight, num_replicas, num_groups, num_nodes, num_gpus)
    else:
        # use global load-balance policy
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