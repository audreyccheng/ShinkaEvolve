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
    Pack n weighted objects to m packs using Vectorized Greedy LPT with
    Incremental State Update and Local Search Refinement.

    Parameters:
        weight: [Batch, n], the weight of each item
        num_packs: number of packs

    Returns:
        pack_index: [Batch, n], the pack index of each item
        rank_in_pack: [Batch, n], the rank of the item in the pack
    """
    batch_size, num_items = weight.shape
    device = weight.device

    # Check divisibility as per original requirements
    assert num_items % num_packs == 0
    items_per_pack = num_items // num_packs

    # Trivial case optimization
    if items_per_pack == 1:
        pack_index = torch.arange(num_packs, dtype=torch.int64, device=device).unsqueeze(0).expand(batch_size, -1)
        rank_in_pack = torch.zeros(batch_size, num_items, dtype=torch.int64, device=device)
        return pack_index, rank_in_pack

    # --- 1. Greedy LPT with Incremental Update ---

    # Sort weights (LPT heuristic)
    sorted_weight, sorted_indices = weight.sort(dim=-1, descending=True)

    # State initialization
    pack_weights = torch.zeros(batch_size, num_packs, device=device)
    pack_counts = torch.zeros(batch_size, num_packs, dtype=torch.int64, device=device)

    # Maintain structures incrementally to avoid expensive reconstruction
    # [Batch, Packs, ItemsPerPack]
    pack_contents = torch.zeros(batch_size, num_packs, items_per_pack, device=device)
    pack_item_ids = torch.zeros(batch_size, num_packs, items_per_pack, dtype=torch.int64, device=device)

    row_indices = torch.arange(batch_size, device=device)

    # Assign items one by one
    for i in range(num_items):
        w = sorted_weight[:, i]
        original_idx = sorted_indices[:, i]

        # Mask valid packs (count < limit)
        valid_mask = pack_counts < items_per_pack

        # Select pack with min weight among valid packs
        # Use a temporary view with large penalties for invalid packs to avoid cloning if possible,
        # but cloning is safe and reasonably fast for these dimensions.
        curr_weights = pack_weights.clone()
        curr_weights[~valid_mask] = float('inf')

        chosen_pack = torch.argmin(curr_weights, dim=1)

        # Get the rank (slot) where we will place the item
        chosen_rank = pack_counts[row_indices, chosen_pack]

        # Update State
        # We use advanced indexing for in-place updates
        pack_weights[row_indices, chosen_pack] += w
        pack_counts[row_indices, chosen_pack] += 1

        pack_contents[row_indices, chosen_pack, chosen_rank] = w
        pack_item_ids[row_indices, chosen_pack, chosen_rank] = original_idx

    # --- 2. Vectorized Refinement ---

    # 2.1 1-Item Swaps (Heaviest vs K-Lightest)
    K = min(4, num_packs - 1)

    if K > 0:
        for _ in range(20):
            # Recalculate weights from contents to ensure consistency
            p_weights = pack_contents.sum(dim=2) # [Batch, Packs]

            val_max, idx_max = p_weights.max(dim=1)

            # Get K lightest packs
            val_min_topk, idx_min_topk = p_weights.topk(K, dim=1, largest=False)

            # Check convergence (compare max with absolute min)
            diff = val_max - val_min_topk[:, 0]
            active_mask = diff > 1e-4
            if not active_mask.any():
                break

            # Gather Max Items: [Batch, ItemsPerPack]
            items_max = pack_contents[row_indices, idx_max]

            # Gather Min Items: [Batch, K, ItemsPerPack]
            idx_min_expanded = idx_min_topk.unsqueeze(2).expand(-1, -1, items_per_pack)
            items_min = pack_contents.gather(1, idx_min_expanded)

            # Compute Delta: [Batch, K, G_max, G_min] (w_max - w_min)
            delta = items_max.unsqueeze(1).unsqueeze(3) - items_min.unsqueeze(2)

            # Compute Improvement
            # Objective: minimize abs(diff - 2*delta)
            # Improvement = diff - abs(diff - 2*delta)
            current_diff = (val_max.unsqueeze(1) - val_min_topk).unsqueeze(2).unsqueeze(3)
            improvement = current_diff - (current_diff - 2 * delta).abs()

            # Filter valid swaps: must move weight from max to min (delta > 0)
            valid_swap = (delta > 0)
            improvement = torch.where(valid_swap, improvement, -1.0)

            # Find best swap per batch
            imp_flat = improvement.view(batch_size, -1)
            best_imp, best_idx_flat = imp_flat.max(dim=1)

            # Threshold check
            do_swap = (best_imp > 1e-5) & active_mask

            if not do_swap.any():
                break

            # Perform Swaps
            batch_indices = row_indices[do_swap]
            flat_indices = best_idx_flat[do_swap]

            # Decode indices from flattened K*G*G
            G = items_per_pack
            idx_g_min = flat_indices % G
            rem = flat_indices // G
            idx_g_max = rem % G
            idx_k = rem // G

            # Get Pack IDs
            p_max = idx_max[batch_indices]
            p_min = idx_min_topk[batch_indices, idx_k]

            # Swap Weights
            val_h = pack_contents[batch_indices, p_max, idx_g_max]
            val_l = pack_contents[batch_indices, p_min, idx_g_min]

            pack_contents[batch_indices, p_max, idx_g_max] = val_l
            pack_contents[batch_indices, p_min, idx_g_min] = val_h

            # Swap IDs
            id_h = pack_item_ids[batch_indices, p_max, idx_g_max]
            id_l = pack_item_ids[batch_indices, p_min, idx_g_min]

            pack_item_ids[batch_indices, p_max, idx_g_max] = id_l
            pack_item_ids[batch_indices, p_min, idx_g_min] = id_h

    # 2.2 2-Item Swaps (Heaviest vs Lightest)
    # Only if G is small enough to generally fit in memory/compute (G <= 20)
    G = items_per_pack
    if G >= 2 and G <= 20:
        # Precompute pair indices (upper triangle)
        # u < v
        triu_r, triu_c = torch.triu_indices(G, G, offset=1, device=device)
        num_pairs = triu_r.shape[0]

        # Limit iterations for 2-swap as it's more expensive
        for _ in range(10):
            p_weights = pack_contents.sum(dim=2)
            val_max, idx_max = p_weights.max(dim=1)
            val_min, idx_min = p_weights.min(dim=1)

            diff = val_max - val_min
            active_mask = diff > 1e-4
            if not active_mask.any():
                break

            # Gather items
            items_max = pack_contents[row_indices, idx_max] # [B, G]
            items_min = pack_contents[row_indices, idx_min] # [B, G]

            # Gather pairs
            # pairs_max: [B, num_pairs]
            pair_sum_max = items_max[:, triu_r] + items_max[:, triu_c]
            pair_sum_min = items_min[:, triu_r] + items_min[:, triu_c]

            # Delta [B, num_pairs, num_pairs]
            # row: max pack pair, col: min pack pair
            delta = pair_sum_max.unsqueeze(2) - pair_sum_min.unsqueeze(1)

            # Improvement
            diff_view = diff.view(-1, 1, 1)
            improvement = diff_view - (diff_view - 2 * delta).abs()

            # Valid mask: delta > 0
            valid_swap = delta > 0
            improvement = torch.where(valid_swap, improvement, -1.0)

            # Best swap
            imp_flat = improvement.view(batch_size, -1)
            best_imp, best_idx_flat = imp_flat.max(dim=1)

            do_swap = (best_imp > 1e-5) & active_mask
            if not do_swap.any():
                break

            # Perform Swaps
            batch_indices = row_indices[do_swap]
            flat_indices = best_idx_flat[do_swap]

            idx_pair_min = flat_indices % num_pairs
            idx_pair_max = flat_indices // num_pairs

            p_max = idx_max[batch_indices]
            p_min = idx_min[batch_indices]

            # Indices in G
            u1 = triu_r[idx_pair_max]
            u2 = triu_c[idx_pair_max]
            v1 = triu_r[idx_pair_min]
            v2 = triu_c[idx_pair_min]

            # Values
            val_u1 = pack_contents[batch_indices, p_max, u1]
            val_u2 = pack_contents[batch_indices, p_max, u2]
            val_v1 = pack_contents[batch_indices, p_min, v1]
            val_v2 = pack_contents[batch_indices, p_min, v2]

            # Swap values
            pack_contents[batch_indices, p_max, u1] = val_v1
            pack_contents[batch_indices, p_max, u2] = val_v2
            pack_contents[batch_indices, p_min, v1] = val_u1
            pack_contents[batch_indices, p_min, v2] = val_u2

            # Swap IDs
            id_u1 = pack_item_ids[batch_indices, p_max, u1]
            id_u2 = pack_item_ids[batch_indices, p_max, u2]
            id_v1 = pack_item_ids[batch_indices, p_min, v1]
            id_v2 = pack_item_ids[batch_indices, p_min, v2]

            pack_item_ids[batch_indices, p_max, u1] = id_v1
            pack_item_ids[batch_indices, p_max, u2] = id_v2
            pack_item_ids[batch_indices, p_min, v1] = id_u1
            pack_item_ids[batch_indices, p_min, v2] = id_u2

    # --- 3. Final Formatting ---

    # Invert mapping: pack_item_ids[b, p, r] = item_id  ->  pack_index[b, item_id] = p
    flat_item_ids = pack_item_ids.view(batch_size, num_items)

    pack_index = torch.empty(batch_size, num_items, dtype=torch.int64, device=device)
    rank_in_pack = torch.empty(batch_size, num_items, dtype=torch.int64, device=device)

    # Coordinate grids
    grid_packs = torch.arange(num_packs, device=device).view(1, -1, 1).expand(batch_size, -1, items_per_pack).reshape(batch_size, -1)
    grid_ranks = torch.arange(items_per_pack, device=device).view(1, 1, -1).expand(batch_size, num_packs, -1).reshape(batch_size, -1)

    pack_index.scatter_(1, flat_item_ids, grid_packs)
    rank_in_pack.scatter_(1, flat_item_ids, grid_ranks)

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

    # Pre-compute scores. Score = weight / count. Initially count is 1.
    current_scores = weight.float() / logcnt.float()

    for i in range(num_log, num_phy):
        redundant_indices = current_scores.argmax(dim=-1)
        phy2log[:, i] = redundant_indices
        rank[:, i] = logcnt[arangen, redundant_indices]

        # Update logcnt
        logcnt[arangen, redundant_indices] += 1

        # Incrementally update scores for modified experts
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
    # Ensure float for division
    weight = weight.float()
    device = weight.device

    # --- Parallel Randomized Restarts ---
    num_restarts = 64

    # Expand Inputs: [Layers, G] -> [Layers * Restarts, G]
    weight_expanded = weight.repeat_interleave(num_restarts, dim=0)

    # Inject Noise for restarts 1..N
    if num_restarts > 1:
        # Linear noise schedule from 0.0 to 0.1
        noise_levels = torch.linspace(0.0, 0.1, num_restarts, device=device)
        # Broadcast to [Layers*Restarts]
        noise_levels = noise_levels.repeat(num_layers).unsqueeze(1)

        noise = torch.rand_like(weight_expanded) * noise_levels
        weight_noisy = weight_expanded * (1.0 + noise)
    else:
        weight_noisy = weight_expanded

    # Run Hierarchical Load Balancing on Expanded Batch
    if num_groups % num_nodes == 0:
        phy2log_cand, phyrank_cand, logcnt_cand = rebalance_experts_hierarchical(
            weight_noisy, num_replicas, num_groups, num_nodes, num_gpus)
    else:
        phy2log_cand, phyrank_cand, logcnt_cand = rebalance_experts_hierarchical(
            weight_noisy, num_replicas, 1, 1, num_gpus)

    # --- Select Best Restart ---
    # Metric: Maximum load on any GPU
    # Reconstruct loads using original (clean) weights

    # Assigned Weight: [Batch, NumPhy]
    w_assigned = weight_expanded.gather(1, phy2log_cand)
    # Replica Count: [Batch, NumPhy]
    c_assigned = logcnt_cand.gather(1, phy2log_cand)

    load_per_replica = w_assigned / c_assigned

    experts_per_gpu = num_replicas // num_gpus
    # Sum loads per GPU: [Batch, NumGPUs, ExpertsPerGPU] -> [Batch, NumGPUs]
    gpu_loads = load_per_replica.view(-1, num_gpus, experts_per_gpu).sum(dim=-1)

    # Max load per Batch
    max_loads = gpu_loads.max(dim=1).values # [Batch]

    # Find argmin per Layer
    max_loads_view = max_loads.view(num_layers, num_restarts)
    best_restart_idx = max_loads_view.argmin(dim=1) # [Layers]

    # Gather Best Results
    base_idx = torch.arange(num_layers, device=device) * num_restarts
    best_batch_idx = base_idx + best_restart_idx

    phy2log = phy2log_cand[best_batch_idx]
    logcnt = logcnt_cand[best_batch_idx]
    phyrank = phyrank_cand[best_batch_idx]

    # Construct Logical -> Physical Map
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