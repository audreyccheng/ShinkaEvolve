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
                            max_iters: int = 100) -> torch.Tensor:
    """
    Batched version of refinement that minimizes load variance by checking both:
    1. All pairwise 1-item swaps.
    2. 2-item swaps between the heaviest and lightest packs.

    Args:
        weights: [B, N] The weight of every item (global indexing).
        pack_indices: [B, num_packs, groups_per_pack] Item IDs assigned to packs.
        pack_weights: [B, num_packs] Current total weight of each pack.
        max_iters: Maximum number of refinement iterations.
    """
    B, num_packs, groups_per_pack = pack_indices.shape
    device = weights.device

    # Pre-compute indices for 2-item combinations if pack size is reasonable
    # Limit to groups_per_pack <= 20 to avoid excessive memory usage (20C2 = 190, 190^2 = 36100 pairs)
    if 2 <= groups_per_pack <= 20:
        pair_indices = torch.triu_indices(groups_per_pack, groups_per_pack, offset=1, device=device)
        num_pairs = pair_indices.shape[1]
    else:
        num_pairs = 0

    # Mask for diagonal packs (u=v) to prevent self-swapping in the all-pairs matrix
    eye_mask = torch.eye(num_packs, device=device).view(1, num_packs, num_packs, 1, 1).bool()

    for _ in range(max_iters):
        # Gather current item weights: [B, M, G]
        w_items = torch.gather(weights, 1, pack_indices.view(B, -1)).view(B, num_packs, groups_per_pack)

        # --- 1. Compute 1-Swap Gains (All Pairs) ---
        # w_u: [B, M, 1, G, 1], w_v: [B, 1, M, 1, G]
        w_u = w_items.view(B, num_packs, 1, groups_per_pack, 1)
        w_v = w_items.view(B, 1, num_packs, 1, groups_per_pack)

        # Delta: w_u - w_v [B, M, M, G, G]
        delta_1 = w_u - w_v

        # W_diff: [B, M, M, 1, 1] (W_v - W_u)
        W = pack_weights.view(B, num_packs, 1, 1, 1)
        W_diff = W.permute(0, 2, 1, 3, 4) - W

        # Gain = -Change = -2 * delta * (W_diff + delta)
        gain_1 = -2 * delta_1 * (W_diff + delta_1)
        gain_1.masked_fill_(eye_mask, -float('inf'))

        flat_gain_1 = gain_1.view(B, -1)
        max_gain_1, max_idx_1 = torch.max(flat_gain_1, dim=1)

        # --- 2. Compute 2-Swap Gains (Max-Min Pairs) ---
        max_gain_2 = torch.full((B,), -float('inf'), device=device)
        max_idx_2 = None

        if num_pairs > 0:
            max_load, max_pid = torch.max(pack_weights, dim=1) # [B]
            min_load, min_pid = torch.min(pack_weights, dim=1) # [B]

            # Gather items from max/min packs: [B, G]
            gather_max_idx = max_pid.view(B, 1, 1).expand(-1, 1, groups_per_pack)
            gather_min_idx = min_pid.view(B, 1, 1).expand(-1, 1, groups_per_pack)

            w_max_items = torch.gather(w_items, 1, gather_max_idx).squeeze(1) # [B, G]
            w_min_items = torch.gather(w_items, 1, gather_min_idx).squeeze(1) # [B, G]

            # Pair sums: [B, num_pairs]
            w_max_pairs = w_max_items[:, pair_indices[0]] + w_max_items[:, pair_indices[1]]
            w_min_pairs = w_min_items[:, pair_indices[0]] + w_min_items[:, pair_indices[1]]

            # Delta for 2-swap: (max_pair - min_pair)
            # Shape [B, num_pairs, num_pairs]
            delta_2 = w_max_pairs.unsqueeze(2) - w_min_pairs.unsqueeze(1)

            # W_diff_2 = L_min - L_max (negative)
            W_diff_2 = (min_load - max_load).view(B, 1, 1)

            # Gain = -2 * delta * (W_diff + delta)
            gain_2 = -2 * delta_2 * (W_diff_2 + delta_2)

            flat_gain_2 = gain_2.view(B, -1)
            max_gain_2, max_idx_2 = torch.max(flat_gain_2, dim=1)

        # --- 3. Execute Swaps ---
        threshold = 1e-5

        use_2swap = (max_gain_2 > max_gain_1) & (max_gain_2 > threshold)
        use_1swap = (~use_2swap) & (max_gain_1 > threshold)

        if not (use_2swap.any() or use_1swap.any()):
            break

        if use_1swap.any():
            batches = torch.nonzero(use_1swap).squeeze(-1)
            idx = max_idx_1[batches]

            G = groups_per_pack
            M = num_packs
            G2 = G * G
            MG2 = M * G2

            u = idx // MG2
            rem = idx % MG2
            v = rem // G2
            rem = rem % G2
            i = rem // G
            j = rem % G

            val_u = pack_indices[batches, u, i]
            val_v = pack_indices[batches, v, j]
            pack_indices[batches, u, i] = val_v
            pack_indices[batches, v, j] = val_u

            d = delta_1[batches, u, v, i, j]
            pack_weights[batches, u] -= d
            pack_weights[batches, v] += d

        if use_2swap.any():
            batches = torch.nonzero(use_2swap).squeeze(-1)
            idx = max_idx_2[batches]

            p_max_idx = idx // num_pairs
            p_min_idx = idx % num_pairs

            i1_local = pair_indices[0, p_max_idx]
            i2_local = pair_indices[1, p_max_idx]
            j1_local = pair_indices[0, p_min_idx]
            j2_local = pair_indices[1, p_min_idx]

            u = max_pid[batches]
            v = min_pid[batches]

            val_u1 = pack_indices[batches, u, i1_local]
            val_v1 = pack_indices[batches, v, j1_local]
            val_u2 = pack_indices[batches, u, i2_local]
            val_v2 = pack_indices[batches, v, j2_local]

            pack_indices[batches, u, i1_local] = val_v1
            pack_indices[batches, v, j1_local] = val_u1
            pack_indices[batches, u, i2_local] = val_v2
            pack_indices[batches, v, j2_local] = val_u2

            d = delta_2[batches, p_max_idx, p_min_idx]
            pack_weights[batches, u] -= d
            pack_weights[batches, v] += d

    return pack_indices


def balanced_packing(weight: torch.Tensor,
                     num_packs: int,
                     num_iters: int = 4,
                     num_candidates: int = 32) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Pack n weighted objects to m packs using Iterative Re-weighting LPT + Refinement.

    In each iteration, we run parallel randomized greedy LPT with virtual weights,
    refine the packings, and then boost the virtual weights of items in the heaviest
    packs of the best solutions. This forces the greedy heuristic to prioritize
    problematic items in subsequent iterations.

    Args:
        weight: [L, N] weights
        num_packs: number of target packs
        num_iters: number of re-weighting iterations
        num_candidates: number of parallel randomized attempts per layer per iteration
    """
    num_layers, num_groups = weight.shape
    device = weight.device
    groups_per_pack = num_groups // num_packs

    # Trivial case optimization
    if groups_per_pack == 1:
        pack_index = torch.arange(num_groups, dtype=torch.int64,
                                  device=device).expand(num_layers, -1)
        rank_in_pack = torch.zeros_like(weight, dtype=torch.int64)
        return pack_index, rank_in_pack

    # Globals
    B = num_layers * num_candidates
    batch_range = torch.arange(B, device=device)
    inf_val = float('inf')

    # Coordinate grids for scatter
    p_ids = torch.arange(num_packs, device=device).unsqueeze(1).expand(-1, groups_per_pack).reshape(-1)
    r_ids = torch.arange(groups_per_pack, device=device).unsqueeze(0).expand(num_packs, -1).reshape(-1)
    p_ids_expanded = p_ids.unsqueeze(0).expand(num_layers, -1)
    r_ids_expanded = r_ids.unsqueeze(0).expand(num_layers, -1)

    # State
    best_overall_max_loads = torch.full((num_layers,), float('inf'), device=device)
    best_overall_assignments = torch.zeros((num_layers, num_packs * groups_per_pack), dtype=torch.int64, device=device)

    # Virtual weights start as real weights: [L, N]
    virtual_weights = weight.clone()

    # Pre-expand real weights for batched ops: [B, N]
    w_expanded = weight.unsqueeze(1).expand(-1, num_candidates, -1).reshape(B, num_groups)

    for iteration in range(num_iters):
        # 1. Expand virtual weights: [B, N]
        v_expanded = virtual_weights.unsqueeze(1).expand(-1, num_candidates, -1).reshape(B, num_groups)

        # 2. Add noise to virtual weights for randomization
        # 1% noise
        noise = torch.rand_like(v_expanded) * 0.01
        v_perturbed = v_expanded * (1.0 + noise)

        # 3. Sort indices based on perturbed virtual weights
        sorted_indices = torch.argsort(v_perturbed, dim=-1, descending=True)

        # Gather ACTUAL weights in the sorted order for accumulation
        row_idx = torch.arange(B, device=device).unsqueeze(1).expand(-1, num_groups)
        w_sorted = w_expanded[row_idx, sorted_indices]

        # 4. Greedy Packing
        pack_weights = torch.zeros(B, num_packs, device=device, dtype=weight.dtype)
        pack_counts = torch.zeros(B, num_packs, device=device, dtype=torch.long)
        pack_assignments = torch.zeros(B, num_packs, groups_per_pack, device=device, dtype=torch.long)

        for i in range(num_groups):
            val = w_sorted[:, i]       # [B]
            itm_idx = sorted_indices[:, i] # [B]

            # Identify full packs
            is_full = pack_counts >= groups_per_pack
            # Mask weights of full packs
            valid_weights = torch.where(is_full, inf_val, pack_weights)
            # Choose pack with min weight
            best_pack = torch.argmin(valid_weights, dim=1) # [B]

            # Assign
            slots = pack_counts[batch_range, best_pack]
            pack_assignments[batch_range, best_pack, slots] = itm_idx
            pack_weights[batch_range, best_pack] += val
            pack_counts[batch_range, best_pack] += 1

        # 5. Batched Refinement (Variance reduction using real weights)
        pack_assignments = _refine_packing_batched(w_expanded, pack_assignments, pack_weights)

        # 6. Evaluation & Selection
        # [L, K, M]
        pw_reshaped = pack_weights.view(num_layers, num_candidates, num_packs)
        # Max load per candidate: [L, K]
        candidate_max_loads = pw_reshaped.max(dim=2).values
        # Max pack index per candidate: [L, K]
        candidate_max_pids = pw_reshaped.argmax(dim=2)

        # Best in this batch for each layer
        batch_best_vals, batch_best_k = torch.min(candidate_max_loads, dim=1) # [L]

        # Update overall best
        improved = batch_best_vals < best_overall_max_loads
        if improved.any():
            # Extract best assignments for this batch
            # [L, K, M, G]
            pa_reshaped = pack_assignments.view(num_layers, num_candidates, num_packs, groups_per_pack)
            # [L, M, G]
            curr_best_assignments = pa_reshaped[torch.arange(num_layers, device=device), batch_best_k]
            # Flatten to [L, M*G]
            curr_flat = curr_best_assignments.view(num_layers, -1)

            best_overall_assignments[improved] = curr_flat[improved]
            best_overall_max_loads[improved] = batch_best_vals[improved]

        # 7. Feedback / Re-weighting
        # If not last iteration, update virtual weights
        if iteration < num_iters - 1:
            # We want to boost items in the heaviest pack of the best candidate in this batch
            # batch_best_k: [L] indices of best candidate
            # candidate_max_pids: [L, K]

            # Get max pid for best candidate: [L]
            best_max_pids = candidate_max_pids[torch.arange(num_layers, device=device), batch_best_k]

            # Get items in that max pack: [L, G]
            # pack_assignments: [B, M, G] -> [L, K, M, G]
            pa_reshaped = pack_assignments.view(num_layers, num_candidates, num_packs, groups_per_pack)
            max_pack_items = pa_reshaped[torch.arange(num_layers, device=device), batch_best_k, best_max_pids]

            # Update virtual weights: Multiply by 1.1
            # We need to scatter this update
            # Create a multiplier tensor initialized to 1.0
            mult = torch.ones_like(virtual_weights)
            # Scatter 1.1 into indices
            src = torch.full_like(max_pack_items, 1.1, dtype=virtual_weights.dtype)
            mult.scatter_(1, max_pack_items, src)

            virtual_weights = virtual_weights * mult

    # Construct final output from best_overall_assignments
    pack_index = torch.empty((num_layers, num_groups), device=device, dtype=torch.int64)
    rank_in_pack = torch.empty((num_layers, num_groups), device=device, dtype=torch.int64)

    pack_index.scatter_(1, best_overall_assignments, p_ids_expanded)
    rank_in_pack.scatter_(1, best_overall_assignments, r_ids_expanded)

    return pack_index, rank_in_pack


def replicate_experts(
        weight: torch.Tensor,
        num_phy: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Replicate experts to `num_phy` replicas, minimizing the maximum load.

    Uses a vectorized greedy approach (Jefferson/D'Hondt method) which is
    optimal for the min-max load objective with discrete allocations.

    Parameters:
        weight: [X, num_log]
        num_phy: total number of experts after replication

    Returns:
        phy2log: [X, num_phy], logical expert id of each physical expert
        rank: [X, num_phy], the replica rank
        logcnt: [X, num_log], number of replicas for each logical expert
    """
    n, num_log = weight.shape
    device = weight.device

    # Initialize: Each expert gets at least 1 replica
    logcnt = torch.ones((n, num_log), dtype=torch.int64, device=device)

    # Greedily assign remaining replicas
    # We need to assign `num_phy - num_log` replicas.
    # In each step, assign to the expert with the highest current load per replica.
    # This loop runs separately for all layers in parallel using tensor ops.
    for _ in range(num_log, num_phy):
        # Score is current load per replica
        scores = weight / logcnt
        # Find expert with max score in each layer
        indices = torch.argmax(scores, dim=-1)
        # Add replica
        # Advanced indexing: row 0 gets indices[0] incremented, etc.
        rows = torch.arange(n, device=device)
        logcnt[rows, indices] += 1

    # Reconstruction of the mapping tables from counts
    phy2log = torch.zeros((n, num_phy), dtype=torch.int64, device=device)
    rank = torch.zeros((n, num_phy), dtype=torch.int64, device=device)

    # Reconstruct mappings row by row
    # While Python loop over layers exists, operations inside are vectorized or efficient enough
    # for typical MoE dimensions (small number of experts/layers).
    for i in range(n):
        counts = logcnt[i]

        # Create logical IDs: repeat each logical ID 'count' times
        # e.g. [0, 0, 1, 2, 2, 2]
        l_ids = torch.repeat_interleave(
            torch.arange(num_log, device=device), counts
        )
        phy2log[i] = l_ids

        # Create ranks: [0, 1, 0, 0, 1, 2]
        # We compute this by iterating logical experts
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

    # Use improved balanced packing with parallel randomization
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

    # Use improved balanced packing here as well
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