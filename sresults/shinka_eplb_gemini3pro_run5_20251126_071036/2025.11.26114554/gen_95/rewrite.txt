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
    Pack n weighted objects to m packs using Parallel Randomized LPT with
    Variance Reduction and MinMax Refinement.

    Parameters:
        weight: [X, n], the weight of each item
        num_packs: number of packs

    Returns:
        pack_index: [X, n], the pack index of each item
        rank_in_pack: [X, n], the rank of the item in the pack
    """
    num_layers, num_groups = weight.shape
    device = weight.device
    assert num_groups % num_packs == 0
    groups_per_pack = num_groups // num_packs

    # Trivial case optimization
    if groups_per_pack == 1:
        pack_index = torch.arange(num_packs, dtype=torch.int64,
                                  device=device).expand(num_layers, -1)
        rank_in_pack = torch.zeros(num_layers,
                                   num_groups,
                                   dtype=torch.int64,
                                   device=device)
        return pack_index, rank_in_pack

    # --- Configuration ---
    num_restarts = 256  # Parallel exploration factor
    num_refined = 32    # Number of candidates to refine fully

    # Expand inputs: [Layers, Groups] -> [Layers * Restarts, Groups]
    batch_size_initial = num_layers * num_restarts
    w_expanded = weight.repeat_interleave(num_restarts, dim=0)

    # Generate Noise for Sorting (Diverse Greedy Orders)
    # Restart 0 per layer is deterministic (noise=0)
    noise_scales = torch.linspace(0.0, 0.15, steps=num_restarts, device=device)
    noise_scales = noise_scales.repeat(num_layers).unsqueeze(1)  # [B, 1]
    
    # Apply noise and sort
    w_noisy = w_expanded * (1.0 + noise_scales * torch.rand_like(w_expanded))
    _, sort_idx = w_noisy.sort(dim=1, descending=True)

    # Gather actual weights in processing order
    w_sorted = torch.gather(w_expanded, 1, sort_idx)

    # --- 1. Vectorized Greedy LPT ---
    pack_weights = torch.zeros(batch_size_initial, num_packs, device=device)
    pack_counts = torch.zeros(batch_size_initial,
                              num_packs,
                              dtype=torch.int64,
                              device=device)
    
    # Store assignments in sorted domain: [Batch, Groups]
    pack_assignments = torch.zeros(batch_size_initial,
                                   num_groups,
                                   dtype=torch.int64,
                                   device=device)
    rank_assignments = torch.zeros(batch_size_initial,
                                   num_groups,
                                   dtype=torch.int64,
                                   device=device)
    
    row_arange = torch.arange(batch_size_initial, device=device)

    # Greedy loop
    for i in range(num_groups):
        item_w = w_sorted[:, i]
        
        # Valid packs mask
        valid_mask = pack_counts < groups_per_pack
        
        # Select pack with min weight among valid ones
        temp_w = pack_weights.clone()
        temp_w[~valid_mask] = float('inf')
        chosen_pack = torch.argmin(temp_w, dim=1)
        
        # Update state
        chosen_rank = pack_counts[row_arange, chosen_pack]
        pack_weights[row_arange, chosen_pack] += item_w
        pack_counts[row_arange, chosen_pack] += 1
        
        pack_assignments[:, i] = chosen_pack
        rank_assignments[:, i] = chosen_rank

    # --- 2. Candidate Pruning ---
    imbalance = pack_weights.max(dim=1).values - pack_weights.min(dim=1).values
    imbalance = imbalance.view(num_layers, num_restarts)
    
    # Select best candidates per layer
    _, best_indices = imbalance.topk(num_refined, dim=1, largest=False)
    
    # Flatten selection indices
    offsets = torch.arange(num_layers, device=device).unsqueeze(1) * num_restarts
    flat_selected = (offsets + best_indices).flatten()
    
    batch_size_ref = num_layers * num_refined
    row_ref = torch.arange(batch_size_ref, device=device)
    
    # Gather data for refinement
    sel_pack_assign = pack_assignments[flat_selected]
    sel_rank_assign = rank_assignments[flat_selected]
    sel_w_sorted = w_sorted[flat_selected]
    sel_orig_idx = sort_idx[flat_selected]
    
    # Construct structured representation: [Batch, Packs, GroupsPerPack]
    pack_contents = torch.zeros(batch_size_ref,
                                num_packs,
                                groups_per_pack,
                                device=device)
    pack_item_ids = torch.zeros(batch_size_ref,
                                num_packs,
                                groups_per_pack,
                                dtype=torch.int64,
                                device=device)
    
    flat_b = row_ref.unsqueeze(1).expand(-1, num_groups).flatten()
    flat_p = sel_pack_assign.flatten()
    flat_r = sel_rank_assign.flatten()
    
    pack_contents.index_put_((flat_b, flat_p, flat_r), sel_w_sorted.flatten())
    pack_item_ids.index_put_((flat_b, flat_p, flat_r), sel_orig_idx.flatten())

    # --- 3. Variance Reduction Refinement (L2 Descent) ---
    # Swap items between random pairs of packs to minimize variance
    
    # Define pairs to check
    if num_packs <= 16:
        # Check all unique pairs
        idx_p1, idx_p2 = torch.triu_indices(num_packs, num_packs, offset=1, device=device)
    else:
        # Check random subset of pairs
        n_pairs = 64
        idx_p1 = torch.randint(0, num_packs, (n_pairs,), device=device)
        idx_p2 = torch.randint(0, num_packs, (n_pairs,), device=device)
        mask = idx_p1 != idx_p2
        idx_p1 = idx_p1[mask]
        idx_p2 = idx_p2[mask]

    num_l2_iters = 10
    if idx_p1.numel() > 0:
        for _ in range(num_l2_iters):
            curr_weights = pack_contents.sum(dim=2) # [B, P]
            
            # W1 - W2
            w_p1 = curr_weights[:, idx_p1]
            w_p2 = curr_weights[:, idx_p2]
            diff = w_p1 - w_p2
            
            # Gather items: [B, N_pairs, G]
            items_p1 = pack_contents[:, idx_p1]
            items_p2 = pack_contents[:, idx_p2]
            
            # Delta matrix: [B, N_pairs, G, G]
            delta = items_p1.unsqueeze(3) - items_p2.unsqueeze(2)
            
            # Maximize: |D| - |D - 2*delta|
            # (Reduction in diff magnitude)
            D_exp = diff.unsqueeze(2).unsqueeze(3)
            improvement = D_exp.abs() - (D_exp - 2 * delta).abs()
            
            # Flatten to find best swap in batch
            imp_flat = improvement.view(batch_size_ref, -1)
            best_imp, best_idx_flat = imp_flat.max(dim=1)
            
            active = best_imp > 1e-5
            if not active.any():
                break
                
            b_idx = torch.where(active)[0]
            flat_idx = best_idx_flat[b_idx]
            
            # Decode indices
            G = groups_per_pack
            GG = G * G
            pair_idx = flat_idx // GG
            rem = flat_idx % GG
            g1 = rem // G
            g2 = rem % G
            
            p1 = idx_p1[pair_idx]
            p2 = idx_p2[pair_idx]
            
            # Swap
            val1 = pack_contents[b_idx, p1, g1]
            val2 = pack_contents[b_idx, p2, g2]
            pack_contents[b_idx, p1, g1] = val2
            pack_contents[b_idx, p2, g2] = val1
            
            id1 = pack_item_ids[b_idx, p1, g1]
            id2 = pack_item_ids[b_idx, p2, g2]
            pack_item_ids[b_idx, p1, g1] = id2
            pack_item_ids[b_idx, p2, g2] = id1

    # --- 4. MinMax Refinement (L-infinity Descent) ---
    # Target Max Load vs K Min Packs using 1-item and 2-item swaps
    
    K_refine = min(4, num_packs - 1)
    can_do_2for2 = (groups_per_pack >= 2) and (groups_per_pack <= 32)
    if can_do_2for2:
        triu_r, triu_c = torch.triu_indices(groups_per_pack, groups_per_pack, offset=1, device=device)
    
    for _ in range(40):
        curr_weights = pack_contents.sum(dim=2)
        
        # Max Pack
        val_max, idx_max = curr_weights.max(dim=1)
        
        # K Min Packs
        val_min_k, idx_min_k = curr_weights.topk(K_refine, dim=1, largest=False)
        
        # Convergence check
        global_diff = val_max - val_min_k[:, 0]
        if (global_diff < 1e-4).all():
            break
            
        active_batch = global_diff > 1e-4
        
        # 1-for-1 Swaps
        items_max = pack_contents[row_ref, idx_max] # [B, G]
        idx_min_exp = idx_min_k.unsqueeze(2).expand(-1, -1, groups_per_pack)
        items_min = pack_contents.gather(1, idx_min_exp) # [B, K, G]
        
        delta_1 = items_max.unsqueeze(1).unsqueeze(3) - items_min.unsqueeze(2) # [B, K, G, G]
        
        diff_k = val_max.unsqueeze(1) - val_min_k
        diff_k_exp = diff_k.unsqueeze(2).unsqueeze(3)
        
        gain_1 = diff_k_exp - (diff_k_exp - 2 * delta_1).abs()
        gain_1[delta_1 <= 0] = -1.0 # Must reduce max pack
        
        best_gain_1, best_flat_1 = gain_1.view(batch_size_ref, -1).max(dim=1)
        
        # 2-for-2 Swaps (Max vs Min only)
        best_gain_2 = torch.full_like(best_gain_1, -1.0)
        best_flat_2 = torch.zeros_like(best_flat_1)
        
        if can_do_2for2:
            p_min_single = idx_min_k[:, 0]
            val_min_single = val_min_k[:, 0]
            items_min_single = pack_contents[row_ref, p_min_single]
            
            pairs_max = items_max[:, triu_r] + items_max[:, triu_c] # [B, N_pairs]
            pairs_min = items_min_single[:, triu_r] + items_min_single[:, triu_c]
            
            delta_2 = pairs_max.unsqueeze(2) - pairs_min.unsqueeze(1) # [B, N_pairs, N_pairs]
            
            diff_2 = (val_max - val_min_single).view(-1, 1, 1)
            gain_2 = diff_2 - (diff_2 - 2 * delta_2).abs()
            gain_2[delta_2 <= 0] = -1.0
            
            best_gain_2, best_flat_2 = gain_2.view(batch_size_ref, -1).max(dim=1)
            
        # Select best move
        use_2 = (best_gain_2 > best_gain_1) & (best_gain_2 > 1e-5)
        use_1 = (~use_2) & (best_gain_1 > 1e-5)
        do_any = (use_1 | use_2) & active_batch
        
        if not do_any.any():
            break
            
        # Execute 1-for-1
        if use_1.any():
            b_idx = row_ref[use_1]
            idx_flat = best_flat_1[use_1]
            
            G = groups_per_pack
            GG = G * G
            k = idx_flat // GG
            rem = idx_flat % GG
            g_max = rem // G
            g_min = rem % G
            
            p_m = idx_max[b_idx]
            p_l = idx_min_k[b_idx, k]
            
            v_m = pack_contents[b_idx, p_m, g_max]
            v_l = pack_contents[b_idx, p_l, g_min]
            
            pack_contents[b_idx, p_m, g_max] = v_l
            pack_contents[b_idx, p_l, g_min] = v_m
            
            i_m = pack_item_ids[b_idx, p_m, g_max]
            i_l = pack_item_ids[b_idx, p_l, g_min]
            pack_item_ids[b_idx, p_m, g_max] = i_l
            pack_item_ids[b_idx, p_l, g_min] = i_m
            
        # Execute 2-for-2
        if use_2.any():
            b_idx = row_ref[use_2]
            idx_flat = best_flat_2[use_2]
            
            n_pairs = triu_r.size(0)
            pair_m = idx_flat // n_pairs
            pair_l = idx_flat % n_pairs
            
            u1, u2 = triu_r[pair_m], triu_c[pair_m]
            v1, v2 = triu_r[pair_l], triu_c[pair_l]
            
            p_m = idx_max[b_idx]
            p_l = idx_min_k[b_idx, 0]
            
            # Swap items u1, v1
            v_u1 = pack_contents[b_idx, p_m, u1]
            v_v1 = pack_contents[b_idx, p_l, v1]
            pack_contents[b_idx, p_m, u1] = v_v1
            pack_contents[b_idx, p_l, v1] = v_u1
            
            id_u1 = pack_item_ids[b_idx, p_m, u1]
            id_v1 = pack_item_ids[b_idx, p_l, v1]
            pack_item_ids[b_idx, p_m, u1] = id_v1
            pack_item_ids[b_idx, p_l, v1] = id_u1
            
            # Swap items u2, v2
            v_u2 = pack_contents[b_idx, p_m, u2]
            v_v2 = pack_contents[b_idx, p_l, v2]
            pack_contents[b_idx, p_m, u2] = v_v2
            pack_contents[b_idx, p_l, v2] = v_u2
            
            id_u2 = pack_item_ids[b_idx, p_m, u2]
            id_v2 = pack_item_ids[b_idx, p_l, v2]
            pack_item_ids[b_idx, p_m, u2] = id_v2
            pack_item_ids[b_idx, p_l, v2] = id_u2

    # --- 5. Final Selection ---
    final_weights = pack_contents.sum(dim=2)
    final_imbalance = final_weights.max(dim=1).values - final_weights.min(dim=1).values
    
    # Reshape to identify best restart per layer
    final_imbalance = final_imbalance.view(num_layers, num_refined)
    best_ref_idx = final_imbalance.argmin(dim=1)
    
    ref_offsets = torch.arange(num_layers, device=device) * num_refined
    best_flat_idx = ref_offsets + best_ref_idx
    
    best_item_ids = pack_item_ids[best_flat_idx]
    
    # Reconstruct output
    pack_index = torch.empty(num_layers, num_groups, dtype=torch.int64, device=device)
    rank_in_pack = torch.empty(num_layers, num_groups, dtype=torch.int64, device=device)
    
    flat_ids = best_item_ids.view(num_layers, -1)
    grid_packs = torch.arange(num_packs, device=device).view(1, -1, 1).expand(num_layers, -1, groups_per_pack).reshape(num_layers, -1)
    grid_ranks = torch.arange(groups_per_pack, device=device).view(1, 1, -1).expand(num_layers, num_packs, -1).reshape(num_layers, -1)
    
    pack_index.scatter_(1, flat_ids, grid_packs)
    rank_in_pack.scatter_(1, flat_ids, grid_ranks)
    
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
    # Pre-compute scores to avoid redundant division
    current_scores = weight.float() / logcnt.float()

    for i in range(num_log, num_phy):
        redundant_indices = current_scores.argmax(dim=-1)
        phy2log[:, i] = redundant_indices
        rank[:, i] = logcnt[arangen, redundant_indices]

        # Update logcnt
        logcnt[arangen, redundant_indices] += 1

        # Incrementally update scores only for modified experts
        new_cnt = logcnt[arangen, redundant_indices].float()
        chosen_weight = weight[arangen, redundant_indices].float()
        current_scores[arangen, redundant_indices] = chosen_weight / new_cnt

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
    weight = weight.float().cpu() # Move to CPU for compatibility if needed, though algo handles device
    # But usually original code moved to CPU for some ops, we keep device-agnostic but often GPU is faster for our vectorized ops.
    # However, to be safe with callers, let's respect device of weight.
    # Actually, original code had `weight = weight.float().cpu()`.
    # Our optimized `balanced_packing` uses torch operations that are much faster on GPU if weight is on GPU.
    # We will assume weight matches device or move it.
    
    # If the input is on CPU, our parallel logic runs on CPU (slower but works).
    # If input is GPU, it runs on GPU.
    # We ensure float type.
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