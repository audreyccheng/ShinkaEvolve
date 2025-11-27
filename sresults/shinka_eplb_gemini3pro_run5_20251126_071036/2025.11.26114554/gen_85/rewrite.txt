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
    Pack n weighted objects to m packs using Greedy LPT followed by 
    Multi-Strategy Vectorized Refinement.

    Parameters:
        weight: [Batch, n], the weight of each item
        num_packs: number of packs

    Returns:
        pack_index: [Batch, n], the pack index of each item
        rank_in_pack: [Batch, n], the rank of the item in the pack
    """
    batch_size, num_items = weight.shape
    device = weight.device
    
    # Validation
    assert num_items % num_packs == 0
    items_per_pack = num_items // num_packs

    # Trivial case
    if items_per_pack == 1:
        pack_index = torch.arange(num_packs, dtype=torch.int64, device=device).unsqueeze(0).expand(batch_size, -1)
        rank_in_pack = torch.zeros(batch_size, num_items, dtype=torch.int64, device=device)
        return pack_index, rank_in_pack

    # --- 1. Greedy LPT ---
    sorted_weight, sorted_indices = weight.sort(dim=-1, descending=True)
    
    pack_weights = torch.zeros(batch_size, num_packs, device=device)
    pack_counts = torch.zeros(batch_size, num_packs, dtype=torch.int64, device=device)
    
    # Store contents: [Batch, Pack, Slot]
    pack_contents = torch.zeros(batch_size, num_packs, items_per_pack, device=device)
    pack_item_ids = torch.zeros(batch_size, num_packs, items_per_pack, dtype=torch.int64, device=device)
    
    row_indices = torch.arange(batch_size, device=device)
    
    # Vectorized assignment loop
    for i in range(num_items):
        w = sorted_weight[:, i]
        original_idx = sorted_indices[:, i]
        
        # Valid packs (not full)
        valid_mask = pack_counts < items_per_pack
        
        # Argmin with masking
        curr_weights = pack_weights.clone()
        curr_weights[~valid_mask] = float('inf')
        chosen_pack = torch.argmin(curr_weights, dim=1)
        
        chosen_rank = pack_counts[row_indices, chosen_pack]
        
        # Update
        # Using advanced indexing for in-place updates
        pack_weights[row_indices, chosen_pack] += w
        pack_counts[row_indices, chosen_pack] += 1
        pack_contents[row_indices, chosen_pack, chosen_rank] = w
        pack_item_ids[row_indices, chosen_pack, chosen_rank] = original_idx

    # --- 2. Refinement ---
    
    # Pre-compute upper triangle indices for 2-item swaps
    # We only use this if the group size is reasonable to avoid O(G^2) explosion
    use_2item = (items_per_pack >= 2 and items_per_pack <= 32)
    triu_r, triu_c = None, None
    if use_2item:
        triu_r, triu_c = torch.triu_indices(items_per_pack, items_per_pack, offset=1, device=device)

    for iter_step in range(20):
        # Update weights from contents to ensure consistency
        p_weights = pack_contents.sum(dim=2)
        
        val_max, idx_max = p_weights.max(dim=1)
        val_min, idx_min = p_weights.min(dim=1)
        
        diff = val_max - val_min
        active_mask = diff > 1e-4
        if not active_mask.any():
            break
            
        # --- Strategy A: Max-vs-TopK 1-Item Swap ---
        # Compare Heaviest Pack against Top-K Lightest Packs
        K = min(4, num_packs - 1)
        val_min_topk, idx_min_topk = p_weights.topk(K, dim=1, largest=False)
        
        # Items in Max Pack: [Batch, G]
        items_max = pack_contents[row_indices, idx_max]
        
        # Items in Candidate Packs: [Batch, K, G]
        idx_cand_expanded = idx_min_topk.unsqueeze(2).expand(-1, -1, items_per_pack)
        items_cand = pack_contents.gather(1, idx_cand_expanded)
        
        # Delta: [Batch, K, G_max, G_cand] (w_max - w_cand)
        delta = items_max.unsqueeze(1).unsqueeze(3) - items_cand.unsqueeze(2)
        
        # Gain metric: maximize improvement in max pack, provided delta > 0
        # gain = diff - |diff - 2*delta|
        # To avoid swapping with a pack that isn't the absolute min and making it the new max,
        # we check the relative diff. 
        # Simplified robust heuristic: Only swap if 0 < delta < diff
        
        diff_view = (val_max.unsqueeze(1) - val_min_topk).unsqueeze(2).unsqueeze(3)
        gain = diff_view - (diff_view - 2 * delta).abs()
        
        # Mask: delta must be positive (reduce max)
        valid_swap = (delta > 0)
        gain = torch.where(valid_swap, gain, -1.0)
        
        best_gain_flat, best_idx_flat = gain.view(batch_size, -1).max(dim=1)
        
        do_swap_1 = (best_gain_flat > 1e-5) & active_mask
        
        if do_swap_1.any():
            b_idx = row_indices[do_swap_1]
            flat_idx = best_idx_flat[do_swap_1]
            
            G = items_per_pack
            idx_g_cand = flat_idx % G
            rem = flat_idx // G
            idx_g_max = rem % G
            idx_k = rem // G
            
            p_m = idx_max[b_idx]
            p_c = idx_min_topk[b_idx, idx_k]
            
            # Swap
            val_u = pack_contents[b_idx, p_m, idx_g_max]
            val_v = pack_contents[b_idx, p_c, idx_g_cand]
            
            pack_contents[b_idx, p_m, idx_g_max] = val_v
            pack_contents[b_idx, p_c, idx_g_cand] = val_u
            
            id_u = pack_item_ids[b_idx, p_m, idx_g_max]
            id_v = pack_item_ids[b_idx, p_c, idx_g_cand]
            
            pack_item_ids[b_idx, p_m, idx_g_max] = id_v
            pack_item_ids[b_idx, p_c, idx_g_cand] = id_u
            
            # If we swapped, continue to next iteration
            continue

        # --- Strategy B: Max-vs-Min 2-Item Swap ---
        if use_2item:
            # Items: [Batch, G]
            items_h = pack_contents[row_indices, idx_max]
            items_l = pack_contents[row_indices, idx_min]
            
            # Pair Sums: [Batch, NumPairs]
            pairs_h = items_h[:, triu_r] + items_h[:, triu_c]
            pairs_l = items_l[:, triu_r] + items_l[:, triu_c]
            
            # Delta: [Batch, NumPairs_H, NumPairs_L]
            delta2 = pairs_h.unsqueeze(2) - pairs_l.unsqueeze(1)
            
            # Gain
            diff_view = diff.view(batch_size, 1, 1)
            imp2 = diff_view - (diff_view - 2 * delta2).abs()
            imp2 = torch.where(delta2 > 0, imp2, -1.0)
            
            best_imp2, best_idx2 = imp2.view(batch_size, -1).max(dim=1)
            
            do_swap_2 = (best_imp2 > 1e-5) & active_mask
            
            if do_swap_2.any():
                b_idx = row_indices[do_swap_2]
                flat_idx = best_idx2[do_swap_2]
                
                num_pairs = pairs_h.shape[1]
                idx_pair_l = flat_idx % num_pairs
                idx_pair_h = flat_idx // num_pairs
                
                p_m = idx_max[b_idx]
                p_l = idx_min[b_idx]
                
                # Indices in G
                u1 = triu_r[idx_pair_h]
                u2 = triu_c[idx_pair_h]
                v1 = triu_r[idx_pair_l]
                v2 = triu_c[idx_pair_l]
                
                # Swap Values
                val_u1 = pack_contents[b_idx, p_m, u1]
                val_u2 = pack_contents[b_idx, p_m, u2]
                val_v1 = pack_contents[b_idx, p_l, v1]
                val_v2 = pack_contents[b_idx, p_l, v2]
                
                pack_contents[b_idx, p_m, u1] = val_v1
                pack_contents[b_idx, p_m, u2] = val_v2
                pack_contents[b_idx, p_l, v1] = val_u1
                pack_contents[b_idx, p_l, v2] = val_u2
                
                # Swap IDs
                id_u1 = pack_item_ids[b_idx, p_m, u1]
                id_u2 = pack_item_ids[b_idx, p_m, u2]
                id_v1 = pack_item_ids[b_idx, p_l, v1]
                id_v2 = pack_item_ids[b_idx, p_l, v2]
                
                pack_item_ids[b_idx, p_m, u1] = id_v1
                pack_item_ids[b_idx, p_m, u2] = id_v2
                pack_item_ids[b_idx, p_l, v1] = id_u1
                pack_item_ids[b_idx, p_l, v2] = id_u2
                
                continue
        
        # If neither swap found, break
        break

    # Reconstruct Output
    pack_index = torch.empty(batch_size, num_items, dtype=torch.int64, device=device)
    rank_in_pack = torch.empty(batch_size, num_items, dtype=torch.int64, device=device)
    
    flat_ids = pack_item_ids.view(batch_size, -1)
    
    grid_packs = torch.arange(num_packs, device=device).view(1, -1, 1).expand(batch_size, -1, items_per_pack).reshape(batch_size, -1)
    grid_ranks = torch.arange(items_per_pack, device=device).view(1, 1, -1).expand(batch_size, num_packs, -1).reshape(batch_size, -1)
    
    pack_index.scatter_(1, flat_ids, grid_packs)
    rank_in_pack.scatter_(1, flat_ids, grid_ranks)
    
    return pack_index, rank_in_pack


def replicate_experts(
        weight: torch.Tensor,
        num_phy: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Replicate `num_log` experts to `num_phy` replicas, such that the maximum
    load of all replicas is minimized.
    """
    n, num_log = weight.shape
    num_redundant = num_phy - num_log
    assert num_redundant >= 0
    device = weight.device
    
    phy2log = torch.arange(num_phy, dtype=torch.int64, device=device).repeat(n, 1)
    rank = torch.zeros(n, num_phy, dtype=torch.int64, device=device)
    logcnt = torch.ones(n, num_log, dtype=torch.int64, device=device)
    arangen = torch.arange(n, dtype=torch.int64, device=device)

    current_scores = weight.float() / logcnt.float()

    for i in range(num_log, num_phy):
        redundant_indices = current_scores.argmax(dim=-1)
        phy2log[:, i] = redundant_indices
        rank[:, i] = logcnt[arangen, redundant_indices]

        logcnt[arangen, redundant_indices] += 1
        
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
    Hierarchical packing: Groups -> Nodes, then Experts -> GPUs.
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

    pphy2mlog = phy2mlog.gather(-1, pphy2phy)
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
    Entry point. Implements Strategic Two-Pass Re-weighting with Parallel Restarts.
    """
    num_layers, num_logical_experts = weight.shape
    weight = weight.float()
    device = weight.device

    # --- Phase 1: Probing Pass ---
    # Run a deterministic pass to identify bottleneck experts
    if num_groups % num_nodes == 0:
        phy2log_probe, _, logcnt_probe = rebalance_experts_hierarchical(
            weight, num_replicas, num_groups, num_nodes, num_gpus)
    else:
        phy2log_probe, _, logcnt_probe = rebalance_experts_hierarchical(
            weight, num_replicas, 1, 1, num_gpus)
            
    # Calculate loads for probing
    w_probe = weight.gather(1, phy2log_probe)
    c_probe = logcnt_probe.gather(1, phy2log_probe)
    load_per_replica_probe = w_probe / c_probe
    experts_per_gpu = num_replicas // num_gpus
    
    # [Layers, NumGPUs]
    gpu_loads_probe = load_per_replica_probe.view(num_layers, num_gpus, experts_per_gpu).sum(dim=-1)
    
    # Identify bottleneck GPU per layer
    max_gpu_idx = gpu_loads_probe.argmax(dim=1) # [Layers]
    
    # Construct Mask for experts on the bottleneck GPU
    # phy2log_probe: [Layers, NumPhy]
    # We want to find which logical experts are on GPU `max_gpu_idx`
    phy_indices = torch.arange(num_replicas, device=device).view(1, num_gpus, experts_per_gpu)
    phy_indices = phy_indices.expand(num_layers, -1, -1)
    
    # Select indices for the max GPU: [Layers, ExpertsPerGPU]
    target_phy_indices = phy_indices.gather(1, max_gpu_idx.view(-1, 1, 1).expand(-1, 1, experts_per_gpu)).squeeze(1)
    
    # Get logical IDs
    target_log_ids = phy2log_probe.gather(1, target_phy_indices) # [Layers, ExpertsPerGPU]
    
    # Create biased weight matrix
    weight_biased = weight.clone()
    # Boost weights of bottleneck experts by 5%
    # We scatter add a small delta
    delta_w = weight.gather(1, target_log_ids) * 0.05
    weight_biased.scatter_add_(1, target_log_ids, delta_w)

    # --- Phase 2: Parallel Randomized Solve ---
    num_restarts_total = 64
    split = num_restarts_total // 2
    
    # Batch 1: Original weights + Noise
    w_batch1 = weight.repeat_interleave(split, dim=0)
    noise1 = torch.rand_like(w_batch1) * 0.05
    w_batch1_noisy = w_batch1 * (1.0 + noise1)
    
    # Batch 2: Biased weights + Noise
    w_batch2 = weight_biased.repeat_interleave(num_restarts_total - split, dim=0)
    noise2 = torch.rand_like(w_batch2) * 0.05
    w_batch2_noisy = w_batch2 * (1.0 + noise2)
    
    # Combine
    weight_input = torch.cat([w_batch1_noisy, w_batch2_noisy], dim=0)
    
    # Run Solver
    if num_groups % num_nodes == 0:
        phy2log_cand, phyrank_cand, logcnt_cand = rebalance_experts_hierarchical(
            weight_input, num_replicas, num_groups, num_nodes, num_gpus)
    else:
        phy2log_cand, phyrank_cand, logcnt_cand = rebalance_experts_hierarchical(
            weight_input, num_replicas, 1, 1, num_gpus)
            
    # --- Phase 3: Selection ---
    # Evaluate candidates on ORIGINAL CLEAN weights
    weight_clean_expanded = weight.repeat_interleave(num_restarts_total, dim=0)
    
    w_assigned = weight_clean_expanded.gather(1, phy2log_cand)
    c_assigned = logcnt_cand.gather(1, phy2log_cand)
    
    load_per_replica = w_assigned / c_assigned
    gpu_loads = load_per_replica.view(-1, num_gpus, experts_per_gpu).sum(dim=-1)
    
    max_loads = gpu_loads.max(dim=1).values # [Batch]
    max_loads_view = max_loads.view(num_layers, num_restarts_total)
    
    best_restart_idx = max_loads_view.argmin(dim=1) # [Layers]
    
    # Gather Results
    base_idx = torch.arange(num_layers, device=device) * num_restarts_total
    best_batch_idx = base_idx + best_restart_idx
    
    phy2log = phy2log_cand[best_batch_idx]
    logcnt = logcnt_cand[best_batch_idx]
    phyrank = phyrank_cand[best_batch_idx]
    
    # Mapping Log -> Phy
    num_redundant_experts = num_replicas - num_logical_experts
    maxlogcnt = num_redundant_experts + 1
    log2phy = torch.full(
        (num_layers, num_logical_experts, maxlogcnt),
        -1,
        dtype=torch.int64,
        device=device,
    )
    
    scatter_idx = phy2log * maxlogcnt + phyrank
    src = torch.arange(num_replicas, dtype=torch.int64, device=device).expand(num_layers, -1)
    
    log2phy.view(num_layers, -1).scatter_(-1, scatter_idx, src)
    
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